from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from unitree_rl_lab.fdm.config import CommandSamplingCfg, RolloutCfg
from unitree_rl_lab.fdm.planning.mppi import MPPICfg, MPPIPlanner
from unitree_rl_lab.fdm.planning.navigation import GoalProgress, NavigationCfg, local_goal
from unitree_rl_lab.fdm.runner.navigation_observation import NavigationObservation, state_from_history
from unitree_rl_lab.fdm.runner.navigation_runner import NavigationRunner
from unitree_rl_lab.fdm.runner.navigation_reporting import compare_trials, summarize_navigation
from test_stage2 import _IdealFDM, _inputs
from test_fixed_collection import _Scene, _maps


def _planner(n=2, **kwargs):
    return MPPIPlanner(_IdealFDM(),n,cfg=MPPICfg(candidates=12),
                       command_cfg=CommandSamplingCfg(max_forward_speed=.8,min_lateral_speed=0,max_lateral_speed=0),
                       forward_only=True,turn_candidates=True,independent_rng=True,record_predictions=True,**kwargs)


def test_forward_constraints_apply_to_every_scored_candidate_and_recovery():
    planner = _planner()
    planner.mean[...,0] = -.2
    planner.mean[...,1] = .2
    planner.initialized[:] = True
    args = list(_inputs())
    args[-1] = torch.tensor([[-2.,0.],[0.,2.]])
    seen = []
    original = planner._score
    def score(*args,**kwargs):
        seen.append(args[1].clone())
        return original(*args,**kwargs)
    with patch.object(planner,'_score',side_effect=score):
        plans = planner.plan(*args,align_mask=torch.tensor([True,True]))
        planner.reset(torch.tensor([0]))
        planner.plan(*args)
    assert all((p[...,0]>=0).all() and (p[...,1]==0).all() and (p[...,0]<=.8).all() for p in seen)
    assert (plans[:,0,0]==0).all()
    population = seen[0].view(2,12,10,3)
    assert (population[:,3,:,2]>0).all() and (population[:,4,:,2]<0).all()
    assert torch.isfinite(planner.last_pose).all()


def test_independent_random_stream_unchanged_when_other_trial_finishes():
    a,b = _planner(),_planner()
    args = _inputs()
    torch.testing.assert_close(a.plan(*args),b.plan(*args))
    one = tuple(v[1:2] for v in args)
    torch.testing.assert_close(a.plan(*args)[1],b.plan(*one)[0],atol=1e-6,rtol=1e-5)


def _raw(yaw=0.):
    return torch.tensor([[0.,0.,1.,0.,0.,torch.sin(torch.tensor(yaw/2)),torch.cos(torch.tensor(yaw/2)),0.]])


def test_turn_progress_is_not_stall_and_arrival_requires_stationary_hold():
    cfg = NavigationCfg(stall_s=1.,arrival_hold_s=.5)
    goal = torch.tensor([[-2.,0.]])
    state = _raw()
    tracker = GoalProgress(goal,state,cfg)
    active = torch.tensor([True])
    for index in range(1,7):
        status = tracker.update(_raw(index*.3),torch.zeros(1),active,index*.4)
        assert not status['recover'].any()
    state[:,:2] = goal
    assert not tracker.update(state,torch.ones(1),active,3.)['reached'].any()
    assert not tracker.update(state,torch.zeros(1),active,3.1,torch.ones(1))['reached'].any()
    assert not tracker.update(state,torch.zeros(1),active,3.2,torch.zeros(1))['reached'].any()
    assert tracker.update(state,torch.zeros(1),active,3.8,torch.zeros(1))['reached'].all()


def test_recovery_budget_ends_fixed_goal_instead_of_resampling_it():
    state = _raw()
    goal = torch.tensor([[2.,0.]])
    progress = GoalProgress(goal,state,NavigationCfg(stall_s=1,max_recoveries=1))
    active = torch.tensor([True])
    assert progress.update(state,torch.zeros(1),active,1.1)['recover'].all()
    assert progress.update(state,torch.zeros(1),active,2.2)['exhausted'].all()
    torch.testing.assert_close(progress.goals,goal)


def test_goal_coordinates_use_root_yaw():
    torch.testing.assert_close(local_goal(torch.tensor([[0.,2.]]),_raw(torch.pi/2)),torch.tensor([[2.,0.]]),atol=1e-6,rtol=0)


class _NavEnv:
    def __init__(self, collision_step=None):
        self.unwrapped = self
        self.num_envs,self.device,self.step_dt = 2,'cpu',.02
        self.common_step_counter = 0
        self.cfg = SimpleNamespace(decimation=4)
        self.termination_manager = SimpleNamespace(active_terms=[])
        self.event_manager = SimpleNamespace(get_term_cfg=lambda name:SimpleNamespace(func=SimpleNamespace(_generator=torch.Generator())))
        self.commands = torch.zeros(2,3)
        self.command_manager = SimpleNamespace(get_term=lambda name:SimpleNamespace(set_command=self._command))
        data = SimpleNamespace(root_pos_w=torch.tensor([[0.,0.,1.],[1.,0.,1.]]),
                               root_quat_w=torch.tensor([[1.,0.,0.,0.]]).repeat(2,1),
                               root_lin_vel_w=torch.zeros(2,3),root_ang_vel_w=torch.zeros(2,3),
                               projected_gravity_b=torch.tensor([[0.,0.,-1.]]).repeat(2,1))
        self.scene = _Scene(robot=SimpleNamespace(data=data))
        names = list(RolloutCfg().collision_body_names)+['left_ankle_roll_link','right_ankle_roll_link']
        self.forces = torch.zeros(2,6,len(names),3)
        self.scene.sensors = {'contact_forces':SimpleNamespace(body_names=names,data=SimpleNamespace(net_forces_w_history=self.forces)),
                              'fdm_height_scanner':SimpleNamespace(reset=lambda ids:None)}
        self.scene.terrain = SimpleNamespace(all_usd_origins=torch.tensor([[0.,0.,0.],[10.,0.,0.],[20.,0.,0.]]),
                                            origin_split_ids=torch.tensor([0,1,2]),cfg=SimpleNamespace(usd_uniform_env_spacing=10.))
        self.collision_step = collision_step
        self.reset_count = 0

    def _command(self, commands):
        self.commands.copy_(commands)

    def reset(self,seed):
        self.reset_count += 1
        self.steps = 0
        self.commands.zero_()
        return {'policy':torch.zeros(2,283)},{}

    def step(self, actions):
        self.steps += 1
        self.common_step_counter += 1
        self.forces.zero_()
        self.forces[:,:,-2:,2] = 2
        if self.steps == self.collision_step:
            self.forces[0,2,0,0] = 5
        data = self.scene['robot'].data
        data.root_lin_vel_w[:,:2] = self.commands[:,:2]
        data.root_ang_vel_w[:,2] = self.commands[:,2]
        data.root_pos_w[:,:2] += self.commands[:,:2]*self.step_dt
        return {'policy':torch.zeros(2,283)},None,torch.zeros(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool),{}


def _run(env,goals,**kwargs):
    cfg = NavigationCfg(initialization_s=1.,timeout_s=.08,arrival_hold_s=.04,log_interval_s=0,**kwargs)
    policy = SimpleNamespace(act=lambda obs:torch.zeros(2,29),reset=lambda:None)
    records = []
    runner = NavigationRunner(env,policy,_planner(),RolloutCfg(num_envs=2),cfg,record=records.append)
    with patch('unitree_rl_lab.fdm.runner.navigation_observation.door_aware_height_map',side_effect=_maps):
        result = runner.run(seed=2,goal_factory=lambda raw:(goals,torch.ones(2,dtype=torch.bool)))
    return result,records


def test_runner_collision_outranks_success_no_autoreset_and_no_dataset_buffer():
    env = _NavEnv(collision_step=51)
    result,records = _run(env,torch.tensor([[0.,0.],[1.,0.]]))
    assert [t['outcome'] for t in result['trials']] == ['collision','reached']
    assert env.reset_count == 1
    assert result['trials'][0]['contact_groups'] == [True,False,False]
    assert result['command_constraint_violations'] == 0
    assert not env.commands.any()
    assert any(r['event']=='terminal' for r in records)


def test_runner_timeout_and_executed_command_constraints():
    env = _NavEnv()
    result,records = _run(env,torch.tensor([[3.,0.],[3.,1.]]))
    assert all(t['outcome']=='timeout' for t in result['trials'])
    decisions = [r for r in records if r['event']=='decision']
    assert decisions
    assert all(c[0]>=0 and c[1]==0 for r in decisions for c in r['commands'])
    assert not env.commands.any()


def test_auto_reset_environment_is_rejected():
    env = _NavEnv()
    env.termination_manager.active_terms=['timeout']
    with pytest.raises(ValueError,match='automatic resets'):
        NavigationRunner(env,None,_planner(),RolloutCfg(num_envs=2))


def test_history_boundary_and_newest_first_match_training():
    env = _NavEnv()
    adapter = NavigationObservation(env,RolloutCfg(num_envs=2))
    obs = {'policy':torch.zeros(2,283)}
    for i in range(10):
        env.scene['robot'].data.root_pos_w[:,0] = i*.01
        obs['policy'][:,:96] = i
        adapter.push(torch.arange(2),adapter.raw(),obs,i*.05)
    assert adapter.proprio[0,:,0].tolist()==list(reversed(range(10)))
    expected = state_from_history(adapter.history)
    with patch('unitree_rl_lab.fdm.runner.navigation_observation.door_aware_height_map',side_effect=_maps):
        actual = adapter.inputs(torch.arange(2),obs,.45)
    torch.testing.assert_close(actual[0],expected)


def test_comparison_does_not_claim_gain_with_changed_starts():
    result,_ = _run(_NavEnv(),torch.tensor([[0.,0.],[1.,0.]]))
    trials = result['trials']
    assert summarize_navigation(trials)['success_rate']==1
    assert compare_trials(trials,trials)['matched_scenarios']
    other = [{**t,'goal':[99.,0.]} for t in trials]
    comparison = compare_trials(trials,other)
    assert not comparison['matched_scenarios'] and 'delta' not in comparison
