import numpy as np
import torch
from tensordict import TensorDict

from unitree_rl_lab.painting.inference import PaintingPolicy
from unitree_rl_lab.rsl_rl_ext.algorithms.painting_ppo import PaintingPPO
from unitree_rl_lab.rsl_rl_ext.modules.painting_actor_critic import PaintingActorCritic
from unitree_rl_lab.rsl_rl_ext.runners.painting_runner import PaintingRunner


def _observations(n=8):
    return TensorDict({"policy": torch.randn(n, 5, 93), "painting": torch.randn(n, 27),
                       "critic": torch.randn(n, 128), "velocity_targets": torch.randn(n, 6)}, [n])


def _policy(obs):
    return PaintingActorCritic(obs, {"policy": ["policy", "painting"], "critic": ["critic"]}, 29,
                              actor_hidden_dims=[32], critic_hidden_dims=[32], estimator_hidden_dims=[32],
                              noise_std_type="log")


def test_actor_has_no_truth_leak_and_policy_gradients_do_not_train_estimator():
    obs = _observations()
    policy = _policy(obs)
    before = policy.act_inference(obs)
    alternate = obs.clone()
    alternate["velocity_targets"] *= 100
    alternate["critic"] *= -50
    torch.testing.assert_close(before, policy.act_inference(alternate))
    before.square().mean().backward()
    assert all(p.grad is None for p in policy.estimator.parameters())
    assert all(p.grad is not None for p in policy.actor.parameters())
    assert policy.get_actor_obs(obs).shape == (8, 126)
    assert (before < policy.action_mean_upper).all()
    assert (before > policy.action_mean_lower).all()


def test_action_mean_and_effective_std_are_bounded_and_projected():
    obs = _observations()
    policy = _policy(obs)
    with torch.no_grad():
        for parameter in policy.actor.parameters():
            parameter.mul_(1.0e4)
        policy.log_std.copy_(torch.linspace(-100, 100, 29))
    actions = policy.act_inference(obs)
    assert (actions <= policy.action_mean_upper).all()
    assert (actions >= policy.action_mean_lower).all()
    assert (policy.effective_action_std >= policy.action_std_min).all()
    assert (policy.effective_action_std <= policy.action_std_max).all()
    policy.project_distribution_parameters_()
    torch.testing.assert_close(policy.log_std.exp(), policy.effective_action_std)


def test_real_ppo_update_trains_both_networks_and_estimator_checkpoint(tmp_path):
    obs = _observations()
    policy = _policy(obs)
    algorithm = PaintingPPO(policy, num_learning_epochs=2, num_mini_batches=2)
    algorithm.init_storage("rl", 8, 4, obs, [29])
    before_actor = [p.detach().clone() for p in policy.actor.parameters()]
    before_estimator = [p.detach().clone() for p in policy.estimator.parameters()]
    with torch.no_grad():
        for i in range(4):
            algorithm.act(obs)
            dones = torch.zeros(8)
            dones[0] = i == 3
            algorithm.process_env_step(obs, torch.randn(8), dones, {"time_outs": dones.bool()})
        algorithm.compute_returns(obs)
    losses = algorithm.update()
    assert all(np.isfinite(v) for v in losses.values())
    assert "action/clip_rate" in losses and "action/std_all" in losses
    assert not any("saturation" in name or "joint_" in name for name in losses)
    assert any(not torch.equal(a, b) for a, b in zip(before_actor, policy.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(before_estimator, policy.estimator.parameters()))
    assert all(p.grad is None for p in policy.estimator.parameters())
    runner = PaintingRunner.__new__(PaintingRunner)
    runner.alg, runner.current_learning_iteration, runner.logger_type = algorithm, 7, "tensorboard"
    runner.disable_logs, runner.device = True, "cpu"
    checkpoint = tmp_path / "model.pt"
    runner.save(checkpoint, {"test": True})
    expected = policy.act_inference(obs).detach().clone()
    with torch.no_grad():
        next(policy.estimator.parameters()).add_(1)
    assert runner.load(checkpoint) == {"test": True}
    torch.testing.assert_close(expected, policy.act_inference(obs))
    assert runner.current_learning_iteration == 7
    assert algorithm.estimator_optimizer.state


def test_onnx_and_numpy_adapter_match_history_action_scales_and_reset(tmp_path):
    obs = _observations(1)
    policy = _policy(obs).eval()
    onnx = policy.export_onnx(tmp_path)
    cfg = {"default_joint_pos": [0.1] * 29,
           "actions": {"JointPositionAction": {"scale": [0.25] * 29, "offset": [0.1] * 29}},
           "painting_inference": {
               "interface_version": 2,
               "angular_velocity_scale": .2,
               "joint_velocity_scale": .05,
               "action_clip": 3.0,
               "raw_action_limits": [[-3.0, 3.0]] * 29,
               "target_position_limits": [[-1.0, 1.0]] * 29,
           }}
    adapter = PaintingPolicy(onnx, cfg)
    for _ in range(3):
        targets = np.random.default_rng(1).normal(size=(5, 3)).astype(np.float32)
        result = adapter.step(
            np.ones(3), [0, 0, -1], np.ones(29), np.ones(29), targets, [1, 0, 0], .3,
            [0, -.2, 0], [.1, -.2], 0, .5, .8,
        )
        command = np.concatenate((targets.ravel(), [1, 0, 0, .3, 0, -.2, 0, .1, -.2, 0, .5, .8])).astype(np.float32)
        tensors = {"policy": torch.from_numpy(adapter.history[None]), "painting": torch.from_numpy(command[None])}
        expected = policy.act_inference(tensors).detach().numpy()[0].clip(-3, 3)
        np.testing.assert_allclose(result, .1 + .25 * expected, atol=1e-6)
    adapter.reset()
    assert not adapter.initialized and not adapter.last_action.any()
