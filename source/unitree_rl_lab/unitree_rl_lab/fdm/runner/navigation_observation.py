"""Bounded inference histories with the same conventions as FDM collection."""

import torch

from ..utils.height_map import door_aware_height_map
from ..utils.se2 import relative_pose_sequence, yaw_from_quaternion_xyzw


def state_from_history(history):
    relative = relative_pose_sequence(history[..., :3], history[..., 3:7])
    return torch.cat((relative, history[..., 7:8]), dim=-1)


def root_to_sensor_transform(sensor, raw_state, ids):
    """Root-yaw prediction frame -> torso-yaw ray-caster frame."""
    data = getattr(sensor, 'data', None)
    if data is None:
        return None
    sensor_yaw = yaw_from_quaternion_xyzw(data.quat_w[ids][:, (1, 2, 3, 0)])
    root_yaw = yaw_from_quaternion_xyzw(raw_state[:, 3:7])
    delta = raw_state[:, :2] - data.pos_w[ids, :2]
    return torch.stack((sensor_yaw.cos() * delta[:, 0] + sensor_yaw.sin() * delta[:, 1],
                        -sensor_yaw.sin() * delta[:, 0] + sensor_yaw.cos() * delta[:, 1],
                        root_yaw - sensor_yaw), dim=-1)


class NavigationObservation:
    def __init__(self, env, cfg):
        self.env, self.cfg = env, cfg
        self.device = torch.device(env.device)
        self.robot = env.scene['robot']
        self.sensor = env.scene.sensors['fdm_height_scanner']
        self.contact = env.scene.sensors['contact_forces']
        self.command = env.command_manager.get_term('base_velocity')
        names = self.contact.body_names
        groups = [('torso_link',), tuple(n for n in cfg.collision_body_names if n.startswith('left_')),
                  tuple(n for n in cfg.collision_body_names if n.startswith('right_'))]
        self.group_ids = [torch.tensor([names.index(n) for n in group], device=self.device) for group in groups]
        self.feet = [names.index(n) for n in ('left_ankle_roll_link', 'right_ankle_roll_link')]
        shape = (env.num_envs, cfg.history_length)
        self.history = torch.zeros(*shape, 8, device=self.device)
        self.proprio = torch.zeros(*shape, 96, device=self.device)
        self.timestamps = torch.zeros(*shape, dtype=torch.float64, device=self.device)
        self.count = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.elapsed = 0.

    def reset(self):
        self.history.zero_()
        self.proprio.zero_()
        self.timestamps.zero_()
        self.count.zero_()
        self.elapsed = 0.

    def contacts(self):
        forces = self.contact.data.net_forces_w_history[:, :self.env.cfg.decimation].norm(dim=-1).amax(dim=1)
        groups = torch.stack([(forces[:, ids] > self.cfg.collision_force_threshold).any(dim=1)
                              for ids in self.group_ids], dim=-1)
        feet = (forces[:, self.feet] > self.cfg.collision_force_threshold).all(dim=-1)
        return groups.any(dim=-1), groups, feet

    def raw(self, collision=None):
        if collision is None:
            collision = torch.zeros(self.env.num_envs, device=self.device)
        return torch.cat((self.robot.data.root_pos_w, self.robot.data.root_quat_w[:, (1, 2, 3, 0)],
                          collision.float()[:, None]), dim=-1)

    def push(self, ids, raw, observations, timestamp):
        if not len(ids):
            return
        if observations['policy'].shape != (self.env.num_envs, 283):
            raise ValueError("Expected the frozen G1 policy's [N, 283] observations.")
        self.history[ids, 1:] = self.history[ids, :-1].clone()
        self.proprio[ids, 1:] = self.proprio[ids, :-1].clone()
        self.timestamps[ids, 1:] = self.timestamps[ids, :-1].clone()
        self.history[ids, 0] = raw[ids]
        self.proprio[ids, 0] = observations['policy'][ids, :96]
        self.timestamps[ids, 0] = timestamp
        self.count[ids] = (self.count[ids] + 1).clamp_max(self.cfg.history_length)

    def tick(self, ids, raw, observations, timestamp):
        self.elapsed += self.cfg.policy_dt
        if self.elapsed + 1e-7 >= self.cfg.history_timestep:
            self.elapsed -= self.cfg.history_timestep
            self.push(ids, raw, observations, timestamp)

    def set_commands(self, commands, observations):
        self.command.set_command(commands)
        # Preserve the rest of the current observation (including its noise).
        observations['policy'][:, 6:9] = commands

    def inputs(self, ids, observations, timestamp):
        raw = self.raw()
        stale = ids[self.timestamps[ids, 0] != timestamp]
        self.push(stale, raw, observations, timestamp)
        if torch.any(self.count[ids] < self.cfg.history_length):
            raise RuntimeError("Planning before the FDM history is full.")
        self.sensor.reset(ids)
        height, invalid = door_aware_height_map(
            self.sensor, ids, shape=(self.cfg.map_height, self.cfg.map_width),
            clip=(self.cfg.map_clip_min, self.cfg.map_clip_max),
            invalid_sentinel=self.cfg.invalid_height_sentinel,
            door_probe_height=self.cfg.map_door_probe_height,
            door_height_threshold=self.cfg.map_door_height_threshold,
        )
        return (state_from_history(self.history[ids]), self.proprio[ids], height.half(), invalid,
                root_to_sensor_transform(self.sensor, raw[ids], ids))
