"""Place the base using the freshly sampled TCP distance."""

import torch


def reset_painting_root(robot, env_ids, origins, distance, forward_reach, lateral_offset, yaw_range):
    state = robot.data.default_root_state[env_ids].clone()
    state[:, :3] += origins[env_ids]
    state[:, 0] = origins[env_ids, 0] - distance - forward_reach
    state[:, 1] = origins[env_ids, 1] + lateral_offset
    yaw = torch.empty(len(env_ids), device=state.device).uniform_(*yaw_range)
    state[:, 3:7] = 0
    state[:, 3] = torch.cos(yaw / 2)
    state[:, 6] = torch.sin(yaw / 2)
    state[:, 7:13] = 0
    robot.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(state[:, 7:13], env_ids=env_ids)
