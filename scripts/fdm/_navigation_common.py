"""Shared local-navigation CLI. Simulator imports happen only after AppLauncher."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from _common import configure_safe_spawn, require_input_file, sha256_file


def parse_args(*, evaluation=False):
    from isaaclab.app import AppLauncher
    from unitree_rl_lab.fdm.config import DEFAULT_TERRAIN_USD
    from unitree_rl_lab.fdm.benchmarks import SCENE_NAMES, get_scene

    parser = argparse.ArgumentParser(description='Frozen forward-only G1 FDM/MPPI local navigation; no global planner.')
    parser.add_argument('--checkpoint', required=True, help='Frozen G1 locomotion policy checkpoint.')
    if evaluation:
        parser.add_argument('--fdm-checkpoints', nargs='+', required=True, help='Compare in order; first is the baseline.')
    else:
        parser.add_argument('--fdm-checkpoint', required=True)
    terrain = parser.add_mutually_exclusive_group()
    terrain.add_argument('--terrain-usd', help='USD terrain (default: the original navigation map).')
    terrain.add_argument('--scene', choices=SCENE_NAMES,
                         help='One generated 10x10 m benchmark arena with fixed start and goal.')
    parser.add_argument('--num-envs', type=int, help='Default: 1 for generated scenes/demo, 8 for USD evaluation.')
    parser.add_argument('--episodes', type=int, help='Trials per seed (generated scenes default 1; USD evaluation 32).')
    parser.add_argument('--seeds', '--seed', nargs='+', type=int, default=[90042])
    parser.add_argument('--split', choices=('train','val','test'), help='USD spatial split; not used with --scene.')
    goal = parser.add_mutually_exclusive_group()
    goal.add_argument('--goal', type=float, nargs=2, metavar=('X','Y'), help='Final goal in world coordinates, one env.')
    goal.add_argument('--goal-offset', type=float, nargs=2, metavar=('FORWARD','LEFT'),
                      help='One fixed goal relative to the prepared starting pose, one env.')
    goal.add_argument('--scenarios', help='JSON with scenarios [{start:[x,y,yaw], goal:[x,y]}], or start_pose:[x,y,z,w,x,y,z].')
    parser.add_argument('--start', type=float, nargs=3, metavar=('X','Y','YAW'), help='Explicit world spawn, one env.')
    parser.add_argument('--goal-min-distance', type=float, default=1.5)
    parser.add_argument('--goal-max-distance', type=float, default=3.)
    parser.add_argument('--timeout', type=float, default=120.)
    parser.add_argument('--initialization-time', type=float, default=2.)
    parser.add_argument('--stall-time', type=float, default=5.)
    parser.add_argument('--max-recoveries', type=int, default=2)
    parser.add_argument('--arrival-radius', type=float, default=.4)
    parser.add_argument('--max-forward-speed', type=float, default=.8)
    parser.add_argument('--max-yaw-rate', type=float, default=1.)
    parser.add_argument('--mppi-candidates', type=int, default=256)
    parser.add_argument('--mppi-iterations', type=int, default=1)
    parser.add_argument('--inference-chunk', type=int, default=2048)
    parser.add_argument('--risk-threshold', type=float, default=.55)
    parser.add_argument('--log-interval', type=float, default=10.)
    parser.add_argument('--disable-policy-corruption', action='store_true')
    parser.add_argument('--disable-draw', action='store_true')
    parser.add_argument('--output', default='logs/fdm_g1/stage3_navigation/' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    # AppLauncher temporarily removes --help and calls parse_known_args;
    # defer required-option validation until the final parse so --help works.
    required = [action for action in parser._actions if action.required]
    for action in required:
        action.required = False
    try:
        AppLauncher.add_app_launcher_args(parser)
    finally:
        for action in required:
            action.required = True
    args = parser.parse_args()
    args.evaluation = evaluation
    try:
        if args.scene:
            if args.split is not None or any(v is not None for v in (args.start,args.goal,args.goal_offset,args.scenarios)):
                raise ValueError('--scene has fixed start/goal and an independent benchmark split; '
                                 'omit --split/--start/--goal/--goal-offset/--scenarios.')
            if args.num_envs not in (None, 1):
                raise ValueError('--scene requires --num-envs 1; repeat trials with --episodes/--seeds.')
            args.num_envs, args.split = 1, 'benchmark'
            get_scene(args.scene).metadata()  # Check clearance, blocked straight path, and reachability before Kit.
        else:
            args.num_envs = args.num_envs if args.num_envs is not None else (8 if evaluation else 1)
            args.split = args.split or ('test' if evaluation else 'val')
            args.terrain_usd = args.terrain_usd or str(DEFAULT_TERRAIN_USD)
        for name in ('checkpoint','terrain_usd','scenarios'):
            if getattr(args,name):
                setattr(args,name,require_input_file(getattr(args,name),'--'+name.replace('_','-')))
        args.fdm_checkpoints = ([require_input_file(p,'--fdm-checkpoints') for p in args.fdm_checkpoints]
                                if evaluation else [require_input_file(args.fdm_checkpoint,'--fdm-checkpoint')])
        if min(args.num_envs,args.mppi_iterations,args.inference_chunk) < 1 or args.mppi_candidates < 8:
            raise ValueError('Require positive counts and at least 8 MPPI candidates.')
        if args.episodes is not None and args.episodes < 1:
            raise ValueError('--episodes must be positive.')
        if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**31 for s in args.seeds):
            raise ValueError('Seeds must be unique nonnegative 31-bit integers.')
        if args.start or args.goal or args.goal_offset:
            if args.num_envs != 1 or args.scenarios:
                raise ValueError('Explicit --start/--goal/--goal-offset require one env and no --scenarios.')
        for coords in (args.start,args.goal,args.goal_offset):
            if coords is not None and not all(math.isfinite(v) for v in coords):
                raise ValueError('Coordinates must be finite.')
        for name in ('timeout','initialization_time','stall_time','arrival_radius','max_forward_speed','max_yaw_rate',
                     'goal_min_distance','goal_max_distance'):
            if not math.isfinite(getattr(args,name)) or getattr(args,name) <= 0:
                raise ValueError(f'--{name.replace("_","-")} must be finite and positive.')
        if args.initialization_time < 1 or args.max_forward_speed > 1.2 or args.max_yaw_rate > 1:
            raise ValueError('Require initialization >= 1 s, vx <= 1.2 m/s, |wz| <= 1 rad/s.')
        if args.max_recoveries < 0 or not math.isfinite(args.log_interval) or args.log_interval < 0:
            raise ValueError('Invalid recovery count or log interval.')
        if not 0 < args.risk_threshold < 1:
            raise ValueError('Risk threshold must be in (0,1).')
        if not args.arrival_radius < args.goal_min_distance <= args.goal_max_distance:
            raise ValueError('Require arrival radius < automatic goal minimum <= maximum distance.')
        args.output = str(Path(args.output).resolve())
        if Path(args.output).exists():
            raise ValueError(f'Refusing to overwrite output directory: {args.output}')
        if args.scenarios:
            read_scenarios(args.scenarios)  # Fail malformed requests before starting Kit.
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    return args


def read_scenarios(path):
    payload = json.loads(Path(path).read_text())
    rows = payload['scenarios']
    if not rows:
        raise ValueError('Scenario list is empty.')
    for row in rows:
        if ('start' in row) == ('start_pose' in row):
            raise ValueError('Each scenario requires exactly one start or start_pose.')
        start = row.get('start',row.get('start_pose'))
        if len(start) != (3 if 'start' in row else 7) or len(row['goal']) != 2:
            raise ValueError('Invalid scenario coordinate dimensions.')
        if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in start+row['goal']):
            raise ValueError('Scenario coordinates must be finite numbers.')
        if 'start_pose' in row and abs(sum(v*v for v in start[3:])-1) > 1e-3:
            raise ValueError('start_pose quaternion must be normalized wxyz.')
    return payload


def create_environment(args, rollout):
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import unitree_rl_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
    from unitree_rl_lab.fdm.planning.goals import NavigationGrid
    from unitree_rl_lab.fdm.runner import FrozenRecurrentPolicy

    task = rollout.task_name
    cfg = parse_env_cfg(task,device=args.device,num_envs=args.num_envs)
    cfg.seed = args.seeds[0]
    cfg.observations.policy.enable_corruption = rollout.policy_observation_corruption
    if args.scene:
        from unitree_rl_lab.tasks.fdm.benchmark_env_cfg import configure_benchmark
        configure_benchmark(cfg,args.scene)
    else:
        cfg.scene.terrain.usd_path = args.terrain_usd
        cfg.scene.terrain.active_split = args.split
        configure_safe_spawn(cfg,rollout)
        cfg.events.reset_base.func.cfg.retain_navigation_grid = True
    env = gym.make(task,cfg=cfg)
    try:
        observations,_ = env.reset(seed=args.seeds[0])
        policy = FrozenRecurrentPolicy(observations,env.unwrapped.action_manager.total_action_dim,
                                       load_cfg_from_registry(task,'rsl_rl_cfg_entry_point'),
                                       args.checkpoint,env.unwrapped.device)
        grid = None if args.scene else NavigationGrid(env.unwrapped.scene.terrain)
        return env,policy,grid
    except BaseException:
        env.close()
        raise


def place_starts(env, rows, grid, split):
    """Place requested starts without silently snapping them across geometry."""
    import torch
    robot = env.scene['robot']
    resetter = env.event_manager.get_term_cfg('reset_base').func
    ids = torch.arange(env.num_envs,device=env.device)
    xy = torch.tensor([r.get('start',r.get('start_pose'))[:2] for r in rows],device=env.device)
    expected = {'train':0,'val':1,'test':2}[split]
    if (grid.lookup(xy) <= 0).any() or (grid.spatial.lookup(xy) != expected).any():
        raise ValueError('Requested start is outside the selected split or conservative free space.')
    ray_z = max(resetter.bounds_max[2]+1.,resetter.cfg.wall_height+1.)
    hit = resetter._door_filtered_downcast(xy,ray_z,ray_z-resetter.bounds_min[2]+100.)
    if not torch.isfinite(hit).all() or not resetter._filter_wall_clearance(hit).all():
        raise ValueError('Requested start failed terrain/robot-clearance validation; choose another start.')
    pose = robot.data.default_root_state[ids,:7].clone()
    for i,row in enumerate(rows):
        if 'start_pose' in row:
            pose[i] = torch.tensor(row['start_pose'],device=env.device)
        else:
            x,y,yaw = row['start']
            pose[i,:3] = torch.tensor([x,y,float(hit[i,2])+float(pose[i,2])+resetter.cfg.safety_margin],device=env.device)
            pose[i,3:] = torch.tensor([math.cos(yaw/2),0.,0.,math.sin(yaw/2)],device=env.device)
    robot.write_root_pose_to_sim(pose,env_ids=ids)
    robot.write_root_velocity_to_sim(torch.zeros(env.num_envs,6,device=env.device),env_ids=ids)
    terrain = env.scene.terrain
    cell = torch.floor((xy-grid.spatial.minimum)/grid.spatial.spacing).long()
    origin_ids = cell[:,0]*grid.spatial.shape[1]+cell[:,1]
    terrain.env_origin_ids[ids] = origin_ids
    terrain.env_origins[ids] = terrain.all_usd_origins[origin_ids]


def run_navigation(args, app):
    import torch
    from unitree_rl_lab.fdm.config import RolloutCfg, CommandSamplingCfg
    from unitree_rl_lab.fdm.planning.mppi import MPPICfg, MPPIPlanner
    from unitree_rl_lab.fdm.planning.navigation import NavigationCfg
    from unitree_rl_lab.fdm.planning.goals import LocalGoalManager, GoalCfg
    from unitree_rl_lab.fdm.runner.navigation_runner import NavigationRunner
    from unitree_rl_lab.fdm.runner.navigation_reporting import (NavigationRecorder, NavigationVisualizer,
        json_safe, summarize_navigation, compare_trials)
    from unitree_rl_lab.fdm.data.shard_writer import _atomic_json_dump
    from unitree_rl_lab.fdm.training.stage2 import load_initial_model
    from unitree_rl_lab.fdm.utils.se2 import yaw_from_quaternion_xyzw
    from unitree_rl_lab.fdm.benchmarks import BENCHMARK_TASK, get_scene

    def save(path,payload):
        _atomic_json_dump(path,json_safe(payload))

    cfg = NavigationCfg(timeout_s=args.timeout,initialization_s=args.initialization_time,
                        stall_s=args.stall_time,max_recoveries=args.max_recoveries,arrival_radius=args.arrival_radius,
                        forward_speed=args.max_forward_speed,yaw_rate=args.max_yaw_rate,log_interval_s=args.log_interval)
    cfg.validate()
    command_cfg = CommandSamplingCfg(max_forward_speed=cfg.forward_speed,min_lateral_speed=0.,max_lateral_speed=0.,
                                     max_yaw_rate=cfg.yaw_rate)
    mppi_cfg = MPPICfg(candidates=args.mppi_candidates,iterations=args.mppi_iterations,
                       inference_chunk=args.inference_chunk,high_risk_threshold=args.risk_threshold,noise_std=(.45,0.,.65))
    benchmark = get_scene(args.scene) if args.scene else None
    benchmark_info = benchmark.metadata() if benchmark else None
    rollout = RolloutCfg(task_name=BENCHMARK_TASK if benchmark else 'Unitree-G1-29dof-FDM-Navigation',
                         policy_checkpoint=args.checkpoint,
                         terrain_usd_path=args.terrain_usd or '',num_envs=args.num_envs,seed=args.seeds[0],split=args.split,
                         command=command_cfg,policy_observation_corruption=not args.disable_policy_corruption)
    rollout.validate()
    terrain_hash = sha256_file(args.terrain_usd) if args.terrain_usd else None
    payload = read_scenarios(args.scenarios) if args.scenarios else None
    if payload and (payload.get('terrain_sha256',terrain_hash) != terrain_hash or payload.get('split',args.split) != args.split):
        raise ValueError('Scenario terrain/split differs from this run.')
    rows = payload['scenarios'] if payload else None
    count = len(rows) if rows else args.episodes or (32 if args.evaluation and not benchmark else args.num_envs)
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    save(output/'config.json',{'version':1,'planner':'local_fdm_mppi','global_planner':None,
         'args':vars(args),'navigation':asdict(cfg),'mppi':asdict(mppi_cfg),'rollout':rollout.as_dict(),
         'benchmark':benchmark_info,
         'terrain_sha256':terrain_hash,'policy_sha256':sha256_file(args.checkpoint),
         'checkpoints':[{'path':p,'sha256':sha256_file(p)} for p in args.fdm_checkpoints]})
    if benchmark:
        save(output/'scene.json',benchmark_info)
        print(f'[FDM] Generated scene={benchmark.name}, size={benchmark.size}m, '
              f'start={benchmark.start}, goal={benchmark.goal}, '
              f'direct_path_blocked={benchmark_info["direct_path_blocked"]}.',flush=True)
    env = None
    camera_initialized = bool(benchmark)  # Generated task already frames the complete arena at startup.
    reports = []
    all_trials = []
    try:
        env,policy,grid = create_environment(args,rollout)
        for model_index,checkpoint in enumerate(args.fdm_checkpoints):
            model,_ = load_initial_model(checkpoint)
            model = model.to(env.unwrapped.device).eval().requires_grad_(False)
            planner = MPPIPlanner(model,args.num_envs,cfg=mppi_cfg,command_cfg=command_cfg,
                                  forward_only=True,turn_candidates=True,independent_rng=True,record_predictions=True)
            model_dir = output/f'model_{model_index:02d}_{Path(checkpoint).stem}'
            model_dir.mkdir()
            trials,batches = [],[]
            for seed in args.seeds:
                for offset in range(0,count,args.num_envs):
                    size = min(args.num_envs,count-offset)
                    batch_rows = rows[offset:offset+size] if rows else None
                    if batch_rows:
                        batch_rows = batch_rows + [batch_rows[-1]]*(args.num_envs-size)
                    elif args.start:
                        batch_rows = [{'start':args.start}]
                    setter = (lambda e:place_starts(e,batch_rows,grid,args.split)) if batch_rows else None
                    batch_seed = seed + offset

                    def goals(raw):
                        nonlocal camera_initialized
                        camera = env.unwrapped.viewport_camera_controller
                        if camera is not None and not camera_initialized:
                            # Frame the actual start (including --start/scenarios)
                            # once, then preserve the user's manual camera view.
                            viewer = env.unwrapped.cfg.viewer
                            origin = raw[viewer.env_index,:3].cpu().tolist()
                            camera.update_view_location(
                                eye=[p+d for p,d in zip(origin,viewer.eye)],
                                lookat=[p+d for p,d in zip(origin,viewer.lookat)])
                            camera_initialized = True
                        sampled_valid = torch.ones(args.num_envs,dtype=torch.bool,device=raw.device)
                        if benchmark:
                            values = torch.tensor([benchmark.goal],device=raw.device)
                        elif rows:
                            values = torch.tensor([r['goal'] for r in batch_rows],device=raw.device)
                        elif args.goal is not None:
                            values = torch.tensor([args.goal],device=raw.device)
                        elif args.goal_offset is not None:
                            yaw = yaw_from_quaternion_xyzw(raw[:,3:7])
                            x,y = args.goal_offset
                            values = raw[:,:2] + torch.stack((yaw.cos()*x-yaw.sin()*y,yaw.sin()*x+yaw.cos()*y),dim=-1)
                        else:
                            sampler = LocalGoalManager(grid,args.num_envs,seed=batch_seed+200003,
                                                       cfg=GoalCfg(minimum_distance=args.goal_min_distance,
                                                                   maximum_distance=args.goal_max_distance,
                                                                   arrival_radius=cfg.arrival_radius))
                            sampler.update(torch.arange(args.num_envs,device=raw.device),raw,0.)
                            values = sampler.goal
                            sampled_valid = sampler.active
                        if benchmark:
                            valid = benchmark.free(raw[:,:2]) & benchmark.free(values)
                        else:
                            anchor = grid.lookup(raw[:,:2])
                            valid = sampled_valid & (anchor > 0) & (grid.lookup(values) == anchor)
                        valid[size:] = False
                        return values,valid

                    recorder = NavigationRecorder(model_dir/f'seed_{seed}_batch_{offset:04d}')
                    visualizer = NavigationVisualizer(args.num_envs) if not args.headless and not args.disable_draw else None
                    try:
                        runner = NavigationRunner(env,policy,planner,rollout,cfg,record=recorder,visualize=visualizer,
                                                  running=app.is_running,
                                                  region_contains=benchmark.contains if benchmark else None)
                        result = runner.run(seed=batch_seed,goal_factory=goals,start_setter=setter,
                                            trial_ids=list(range(offset,offset+args.num_envs)))
                        result['trials'] = result['trials'][:size]
                        for t in result['trials']:
                            t['batch_seed'] = t['seed']; t['seed'] = seed
                            if benchmark:
                                t['scene'] = benchmark.name
                                t['reference_path_length_m'] = None
                                t['path_length_ratio'] = None
                                if t['outcome'] not in ('initialization_failed','invalid_goal'):
                                    try:
                                        _, length = benchmark.reference_path(t['start'])
                                    except ValueError:
                                        pass  # Record the failure without inventing a valid reference route.
                                    else:
                                        t['reference_path_length_m'] = length
                                        if t['outcome'] == 'reached':
                                            t['path_length_ratio'] = t['path_length_m']/length
                        trials.extend(result['trials'])
                        batches.append({k:v for k,v in result.items() if k != 'trials'})
                        save(recorder.directory/'result.json',result)
                        print(f"[FDM] Navigation model={model_index} seed={seed} trials={len(trials)} "
                              f"{summarize_navigation(result['trials'])}",flush=True)
                    finally:
                        recorder.close()
                    if not app.is_running():
                        raise InterruptedError('Navigation interrupted before completing the requested evaluation.')
            report = {'checkpoint':checkpoint,'summary':summarize_navigation(trials),'batches':batches,'trials':trials}
            if all_trials:
                report['comparison_to_first'] = compare_trials(all_trials[0],trials)
            save(model_dir/'summary.json',report)
            reports.append({k:v for k,v in report.items() if k != 'trials'})
            all_trials.append(trials)
            del runner,planner,model
        save(output/'summary.json',{'status':'completed','benchmark':benchmark_info,'models':reports})
        print(f'[FDM] Navigation complete: {output / "summary.json"}',flush=True)
    except BaseException as exc:
        save(output/'failure.json',{'status':'failed','type':type(exc).__name__,'message':str(exc)})
        raise
    finally:
        if env is not None:
            env.close()


def main(*, evaluation=False):
    from isaaclab.app import AppLauncher
    args = parse_args(evaluation=evaluation)
    launcher = AppLauncher(args)
    try:
        run_navigation(args,launcher.app)
    finally:
        launcher.app.close(wait_for_replicator=False)
