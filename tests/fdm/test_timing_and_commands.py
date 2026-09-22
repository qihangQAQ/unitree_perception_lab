import torch

from unitree_rl_lab.fdm.config import CommandSamplingCfg
from unitree_rl_lab.fdm.runner.command_planner import CorrelatedCommandPlanner
from unitree_rl_lab.fdm.utils.timing import history_tick_indices


def test_twenty_hz_history_has_ten_ticks_per_command():
    assert history_tick_indices(25) == [3, 5, 8, 10, 13, 15, 18, 20, 23, 25]


def test_command_plan_advances_without_rewriting_future():
    planner = CorrelatedCommandPlanner(4, 10, CommandSamplingCfg(), "cpu", seed=7)
    planner.reset()
    before = planner.plan.clone()
    planner.advance(torch.tensor([1, 3]))
    assert torch.equal(planner.plan[1, :-1], before[1, 1:])
    assert torch.equal(planner.plan[3, :-1], before[3, 1:])
    assert torch.equal(planner.plan[0], before[0])
    assert torch.count_nonzero(planner.plan[..., 1]) > 0
    assert torch.all(planner.plan[..., 1] >= planner.cfg.min_lateral_speed)
    assert torch.all(planner.plan[..., 1] <= planner.cfg.max_lateral_speed)
