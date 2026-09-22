import torch

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
