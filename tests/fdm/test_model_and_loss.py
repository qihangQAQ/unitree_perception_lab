import torch
import torch.nn.functional as functional

from unitree_rl_lab.fdm.config import FDMModelCfg, TrainCfg
from unitree_rl_lab.fdm.models import G1HeightFDM
from unitree_rl_lab.fdm.training.losses import FDMLoss
from unitree_rl_lab.fdm.utils.se2 import integrate_body_twists


def test_body_twist_residual_integration():
    commands = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    pose = integrate_body_twists(commands, 0.5)
    assert torch.allclose(pose[0, :, 0], torch.tensor([0.5, 1.0]))
    assert torch.allclose(pose[0, :, 1:3], torch.zeros(2, 2))
    assert torch.allclose(pose[0, :, 3], torch.ones(2))


def test_all_at_once_model_shapes_and_backward():
    model = G1HeightFDM(FDMModelCfg())
    batch_size = 2
    state = torch.randn(batch_size, 10, 5)
    proprio = torch.randn(batch_size, 10, 96)
    height = torch.randn(batch_size, 1, 60, 46)
    commands = torch.randn(batch_size, 10, 3)
    output = model(state, proprio, height, commands)
    assert output["future_pose"].shape == (batch_size, 10, 4)
    assert output["collision_logits"].shape == (batch_size, 10)
    assert output["correction_twist"].shape == (batch_size, 10, 3)
    target = {
        "future_pose": torch.randn(batch_size, 10, 4),
        "future_collision": torch.zeros(batch_size, 10),
        "valid_mask": torch.ones(batch_size, 10, dtype=torch.bool),
    }
    loss = FDMLoss(TrainCfg())(output, target)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert model.collision_decoder[-1].weight.grad is not None


def test_trajectory_loss_sums_per_step_mse_with_original_weights():
    cfg = TrainCfg()
    predicted_pose = torch.tensor([1.0, 3.0, 2.0, 4.0]).expand(2, 10, 4).clone()
    prediction = {
        "future_pose": predicted_pose,
        "collision_logits": torch.zeros(2, 10),
    }
    target = {
        "future_pose": torch.zeros_like(predicted_pose),
        "future_collision": torch.zeros(2, 10),
        "valid_mask": torch.ones(2, 10, dtype=torch.bool),
    }

    losses = FDMLoss(cfg)(prediction, target)

    # Mean over the two pose channels at each horizon step, then sum ten steps.
    torch.testing.assert_close(losses["position_loss"], torch.tensor(10 * (1.0**2 + 3.0**2) / 2))
    torch.testing.assert_close(losses["heading_loss"], torch.tensor(10 * (2.0**2 + 4.0**2) / 2))
    torch.testing.assert_close(losses["collision_loss"], torch.log(torch.tensor(2.0)))
    torch.testing.assert_close(
        losses["loss"],
        cfg.position_weight * losses["position_loss"]
        + cfg.heading_weight * losses["heading_loss"]
        + cfg.collision_weight * losses["collision_loss"],
    )


def test_trajectory_loss_respects_step_masks_and_keeps_collision_bce_mean():
    predicted_pose = torch.zeros(2, 10, 4)
    predicted_pose[0, 0, :2] = torch.tensor([1.0, 3.0])
    predicted_pose[1, 0, :2] = 100.0  # Invalid at step zero.
    predicted_pose[0, 1, :2] = torch.tensor([2.0, 4.0])
    predicted_pose[1, 1, :2] = torch.tensor([4.0, 6.0])
    predicted_pose[:, 2, :2] = 100.0  # No valid samples at this step.
    predicted_pose[0, 0, 2:] = torch.tensor([2.0, 4.0])
    predicted_pose[0, 1, 2:] = torch.tensor([1.0, 3.0])
    predicted_pose[1, 1, 2:] = torch.tensor([3.0, 5.0])
    logits = torch.zeros(2, 10)
    logits[0, 0] = 0.7
    logits[0, 1] = -0.5
    logits[1, 1] = 1.2
    collision = torch.zeros(2, 10)
    collision[0, 1] = 1.0
    valid = torch.zeros(2, 10, dtype=torch.bool)
    valid[0, 0] = valid[0, 1] = valid[1, 1] = True

    losses = FDMLoss(TrainCfg(), collision_pos_weight=2.5)(
        {"future_pose": predicted_pose, "collision_logits": logits},
        {"future_pose": torch.zeros_like(predicted_pose), "future_collision": collision, "valid_mask": valid},
    )

    expected_position = (1.0**2 + 3.0**2) / 2 + (2.0**2 + 4.0**2 + 4.0**2 + 6.0**2) / 4
    expected_heading = (2.0**2 + 4.0**2) / 2 + (1.0**2 + 3.0**2 + 3.0**2 + 5.0**2) / 4
    expected_collision = functional.binary_cross_entropy_with_logits(
        logits[valid], collision[valid], pos_weight=torch.tensor(2.5)
    )
    torch.testing.assert_close(losses["position_loss"], torch.tensor(expected_position))
    torch.testing.assert_close(losses["heading_loss"], torch.tensor(expected_heading))
    torch.testing.assert_close(losses["collision_loss"], expected_collision)
