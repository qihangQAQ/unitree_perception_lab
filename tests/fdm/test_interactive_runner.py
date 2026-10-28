"""Real Torch/history logic with a deterministic fake simulator and planner."""

from types import SimpleNamespace

import torch

from unitree_rl_lab.fdm.config import RolloutCfg
from unitree_rl_lab.fdm.interactive_goals import SpaceRequest
from unitree_rl_lab.fdm.planning.navigation import NavigationCfg
from unitree_rl_lab.fdm.runner import interactive_demo as demo
from unitree_rl_lab.fdm.runner.navigation_observation import NavigationObservation


class Controls:
    def __init__(self):
        self.gate, self.phase, self.round = SpaceRequest(), "settling", 0
        self.events = []

    def set_waiting(self, enabled):
        self.gate.enable(enabled)

    def consume_next(self):
        return self.gate.consume()

    def update(self, phase, number, goal, distance, reason=""):
        self.phase, self.round = phase, number
        self.events.append((phase, number, goal, reason))


class Scene(dict):
    pass


class Env:
    def __init__(self, controls, *, frozen=False, collide=False, finish_round=2):
        self.num_envs, self.device = 1, "cpu"
        self.steps = self.wait_steps = self.reset_calls = 0
        self.controls, self.frozen, self.collide = controls, frozen, collide
        self.finish_round, self.running = finish_round, True
        self.commands, self.wait_commands = torch.zeros(1, 3), []
        self.waiting_round = None
        self.cfg = SimpleNamespace(decimation=4, scene=SimpleNamespace(
            height_scanner=SimpleNamespace(update_period=0.1)))
        self.sim = SimpleNamespace(is_stopped=lambda: False, is_playing=lambda: True, render=lambda: None)
        data = SimpleNamespace(root_pos_w=torch.tensor([[0., 0., 0.8]]),
                               root_quat_w=torch.tensor([[1., 0., 0., 0.]]),
                               root_lin_vel_w=torch.zeros(1, 3), root_ang_vel_w=torch.zeros(1, 3),
                               projected_gravity_b=torch.tensor([[0., 0., -1.]]))
        self.scene = Scene(robot=SimpleNamespace(data=data))
        names = list(RolloutCfg().collision_body_names) + ["left_ankle_roll_link", "right_ankle_roll_link"]
        self.forces = torch.zeros(1, 6, len(names), 3)
        self.forces[:, :, -2:, 2] = 2
        self.scene.sensors = {
            "contact_forces": SimpleNamespace(body_names=names, data=SimpleNamespace(net_forces_w_history=self.forces)),
            "fdm_height_scanner": SimpleNamespace(reset=lambda ids: None),
        }
        self.scene.terrain = SimpleNamespace(
            demo_scene=SimpleNamespace(contains=lambda xy: True), demo_start=(0, 0, 0),
            demo_goal=None, demo_start_height=0, demo_goal_height=None,
        )
        self.command_manager = SimpleNamespace(get_term=lambda _: SimpleNamespace(
            set_command=lambda values: self.commands.copy_(values)))

    def reset(self, **_):
        self.reset_calls += 1
        raise AssertionError("Interactive rounds must not reset the environment")

    def step(self, actions):
        self.steps += 1
        if self.steps > 1600:
            raise AssertionError("Interactive loop failed to complete its simulated rounds")
        data = self.scene["robot"].data
        data.root_lin_vel_w[:, :2] = 0 if self.frozen else self.commands[:, :2]
        data.root_pos_w[:, :2] += data.root_lin_vel_w[:, :2] * .02
        data.root_ang_vel_w[:, 2] = self.commands[:, 2]
        if self.controls.phase == "navigating" and self.controls.round == 1:
            self.controls.gate.key(True)  # Hold throughout walking into the wait.
        if self.controls.phase == "waiting":
            if self.waiting_round != self.controls.round:
                self.waiting_round, self.wait_steps = self.controls.round, 0
            self.wait_steps += 1
            self.wait_commands.append(self.commands.clone())
            if self.collide and self.wait_steps == 25:
                self.forces[:, :, 0, 0] = 5
            if self.controls.round >= self.finish_round:
                if self.wait_steps == 30:
                    self.running = False
            elif self.wait_steps == 40:
                self.controls.gate.key(False)
            elif self.wait_steps == 300:
                self.controls.gate.key(True)  # Fresh press after six seconds standing.
            elif self.wait_steps == 301:
                self.controls.gate.key(False)
        return ({"policy": torch.full((1, 283), float(self.steps))}, None,
                torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), {})


def run_trial(monkeypatch, *, frozen=False, collide=False, sample_failure=False, stall=False):
    controls = Controls()
    env = Env(controls, frozen=frozen, collide=collide, finish_round=1 if sample_failure else 2)
    centers, input_times, resets, adapters = [], [], [], []

    class Sampler:
        def __init__(self, *_, **__):
            pass

        def sample(self, center):
            centers.append(center.copy())
            if sample_failure and len(centers) == 1:
                return None
            return (float(center[0]) + 2.5, float(center[1]), 0.)

    class Adapter(NavigationObservation):
        def __init__(self, *args):
            super().__init__(*args)
            self.wait_ticks = 0
            adapters.append(self)

        def tick(self, *args):
            self.wait_ticks += controls.phase == "waiting"
            return super().tick(*args)

        def inputs(self, ids, obs, timestamp):
            result = super().inputs(ids, obs, timestamp)
            input_times.append((controls.round, timestamp, self.timestamps.clone()))
            return result

    def maps(sensor, ids, **_):
        return torch.zeros(len(ids), 1, 60, 46), torch.zeros(len(ids), 1, 60, 46, dtype=torch.bool)

    class Planner:
        previous = torch.zeros(1, 3)

        def reset(self, ids=None, **kwargs):
            resets.append((ids, kwargs))

        def plan(self, *_, **__):
            result = torch.zeros(1, 10, 3)
            result[:, :, 0] = .8
            return result

    policy = SimpleNamespace(reset_calls=0, act_calls=0)

    def reset_policy():
        policy.reset_calls += 1

    def act(obs):
        policy.act_calls += 1
        return torch.zeros(1, 29)

    policy.reset, policy.act = reset_policy, act
    monkeypatch.setattr(demo, "LocalGoalSampler", Sampler)
    monkeypatch.setattr(demo, "NavigationObservation", Adapter)
    monkeypatch.setattr("unitree_rl_lab.fdm.runner.navigation_observation.door_aware_height_map", maps)
    navigation = NavigationCfg(timeout_s=10 if stall or not frozen else .2,
                               stall_s=.1 if stall else 15, max_recoveries=1)
    result = demo.run_interactive(
        env, SimpleNamespace(is_running=lambda: env.running),
        SimpleNamespace(seed=0, goal_radius=5., goal_min_distance=2.), policy, Planner(),
        RolloutCfg(num_envs=1), navigation, None, {"policy": torch.zeros(1, 283)}, controls,
    )
    return SimpleNamespace(result=result, env=env, controls=controls, centers=centers,
                           adapters=adapters, inputs=input_times, resets=resets, policy=policy)


def test_repeated_goals_preserve_history_and_do_not_reset_robot(monkeypatch):
    trial = run_trial(monkeypatch)
    assert trial.result is None and len(trial.centers) == 2
    assert trial.env.reset_calls == 0 and trial.policy.reset_calls == 1
    assert trial.policy.act_calls == trial.env.steps
    assert trial.adapters[0].wait_ticks >= 330
    assert all(torch.count_nonzero(c) == 0 for c in trial.env.wait_commands)
    assert 2.0 < trial.centers[1][0] < 2.5  # Actual stop position, not previous target.
    second = next(row for row in trial.inputs if row[0] == 2)
    assert second[1] > 8.0
    assert second[1] - float(second[2].min()) < .6
    assert len(trial.resets) == 3  # Initial seed, first goal, second goal.
    assert "seed" not in trial.resets[-1][1]


def test_timeout_waits_for_space_then_starts_new_round(monkeypatch):
    trial = run_trial(monkeypatch, frozen=True)
    assert trial.result is None and len(trial.centers) == 2
    assert any("timeout" in event[3] for event in trial.controls.events)
    assert trial.env.reset_calls == 0
    assert all(torch.count_nonzero(c) == 0 for c in trial.env.wait_commands)


def test_stall_exhaustion_returns_to_wait(monkeypatch):
    trial = run_trial(monkeypatch, frozen=True, stall=True)
    assert trial.result is None and len(trial.centers) == 2
    assert any("No progress" in event[3] for event in trial.controls.events)


def test_sampling_failure_waits_without_goal(monkeypatch):
    trial = run_trial(monkeypatch, sample_failure=True)
    assert trial.result is None and len(trial.centers) == 2
    assert any(phase == "waiting" and number == 0 and goal is None
               for phase, number, goal, _ in trial.controls.events)
    assert any("No valid target" in event[3] for event in trial.controls.events)


def test_collision_during_wait_stops_instead_of_resuming(monkeypatch):
    trial = run_trial(monkeypatch, collide=True)
    assert trial.result == "Body/hand collision"
    assert len(trial.centers) == 1 and trial.env.reset_calls == 0
    assert not trial.controls.consume_next()
