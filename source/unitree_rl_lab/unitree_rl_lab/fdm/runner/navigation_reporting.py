"""Streaming navigation diagnostics and optional Isaac Sim debug drawing."""

from collections import Counter, deque
import json
import math
from pathlib import Path

import torch

from ..utils.se2 import yaw_from_quaternion_xyzw


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key:json_safe(item) for key,item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


class NavigationRecorder:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.events = (self.directory / 'events.jsonl').open('w', buffering=1)
        self.trajectory = (self.directory / 'trajectory.jsonl').open('w', buffering=1)

    def __call__(self, record):
        stream = self.trajectory if record['event'] == 'decision' else self.events
        stream.write(json.dumps(json_safe(record), allow_nan=False) + '\n')

    def close(self):
        self.events.close()
        self.trajectory.close()


def summarize_navigation(trials):
    counts = Counter(t['outcome'] for t in trials)
    reached = [t for t in trials if t['outcome'] == 'reached']
    ratios = [t['path_length_ratio'] for t in reached if t.get('path_length_ratio') is not None]
    return {'trials':len(trials), 'outcomes':dict(counts),
            'success_rate':counts['reached']/max(1,len(trials)),
            'collision_rate':counts['collision']/max(1,len(trials)),
            'stall_rate':counts['stalled']/max(1,len(trials)),
            'timeout_rate':counts['timeout']/max(1,len(trials)),
            'fall_rate':counts['fall']/max(1,len(trials)),
            'initialization_failure_rate':counts['initialization_failed']/max(1,len(trials)),
            'invalid_goal_rate':counts['invalid_goal']/max(1,len(trials)),
            'mean_success_time_s':sum(t['elapsed_s'] for t in reached)/len(reached) if reached else None,
            'mean_success_path_length_m':sum(t['path_length_m'] for t in reached)/len(reached) if reached else None,
            'mean_success_path_length_ratio':sum(ratios)/len(ratios) if ratios else None,
            'recoveries':sum(t['recoveries'] for t in trials),
            'collision_groups':{name:sum(t['contact_groups'][i] for t in trials)
                                for i,name in enumerate(('torso','left_hand','right_hand'))}}


def compare_trials(baseline, current, tolerance=1e-3):
    left = {(t['seed'],t['trial']):t for t in baseline}
    right = {(t['seed'],t['trial']):t for t in current}
    mismatches = []
    for key in sorted(left.keys() | right.keys()):
        if key not in left or key not in right:
            mismatches.append(list(key))
            continue
        a, b = left[key], right[key]
        x = torch.tensor(a['start'][:7] + a['goal'])
        y = torch.tensor(b['start'][:7] + b['goal'])
        if not torch.allclose(x, y, atol=tolerance, rtol=0):
            mismatches.append(list(key))
    result = {'matched_scenarios':not mismatches, 'mismatches':mismatches, 'tolerance':tolerance}
    if not mismatches:
        a, b = summarize_navigation(baseline), summarize_navigation(current)
        result['delta'] = {key:b[key]-a[key] for key in ('success_rate','collision_rate','stall_rate')}
    return result


class NavigationVisualizer:
    """Selected prediction (cyan), actual path (green), final goal (red)."""

    def __init__(self, num_envs):
        from isaacsim.util.debug_draw import _debug_draw
        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.trails = [deque(maxlen=600) for _ in range(num_envs)]

    def __call__(self, raw, goals, commands, prediction, active, align):
        self.draw.clear_lines()
        starts, ends, colors, widths = [], [], [], []

        def line(a,b,color,width=3.):
            starts.append(tuple(a)); ends.append(tuple(b)); colors.append(color); widths.append(width)

        yaw = yaw_from_quaternion_xyzw(raw[:,3:7])
        for i in range(len(raw)):
            xyz = raw[i,:3].cpu().tolist()
            self.trails[i].append(xyz)
            for a,b in zip(self.trails[i],list(self.trails[i])[1:]):
                line(a,b,(.2,1.,.2,1.),2.)
            gx,gy = goals[i].cpu().tolist()
            line((gx-.2,gy,xyz[2]),(gx+.2,gy,xyz[2]),(1.,.1,.1,1.),5.)
            line((gx,gy-.2,xyz[2]),(gx,gy+.2,xyz[2]),(1.,.1,.1,1.),5.)
            if not bool(active[i]):
                continue
            local = prediction[i,:,:2]
            c,s = float(yaw[i].cos()),float(yaw[i].sin())
            points = [[xyz[0]+c*x-s*y,xyz[1]+s*x+c*y,xyz[2]] for x,y in local.cpu().tolist()]
            for a,b in zip([xyz]+points,points):
                if all(math.isfinite(v) for v in a+b):
                    line(a,b,(.1,.8,1.,1.))
            line(xyz,(xyz[0]+.5*c,xyz[1]+.5*s,xyz[2]),
                 (1.,.7,0.,1.) if bool(align[i]) else (.2,.4,1.,1.),5.)
        if starts:
            self.draw.draw_lines(starts,ends,colors,widths)
