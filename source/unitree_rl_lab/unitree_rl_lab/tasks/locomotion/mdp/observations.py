from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.envs import ManagerBasedEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def feet_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.5) -> torch.Tensor:
    """返回左右脚的布尔接触状态（与 LeggedLab 对齐）。

    每只脚：如果接触力范数 > threshold，则为 True(1.0)，否则为 False(0.0)。

    Args:
        env: 环境实例
        sensor_cfg: 传感器配置，指定传感器名称和身体名称（如 ".*ankle_roll.*"）
        threshold: 接触力阈值（N），默认 0.5N

    Returns:
        torch.Tensor: 布尔接触状态，形状为 (num_envs, num_feet)，如 (num_envs, 2)
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # net_forces_w 形状: (num_envs, num_bodies, 3)，取指定body的力
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    # 计算每只脚的力范数，取历史最大值，然后与阈值比较
    force_norms = torch.norm(forces, dim=-1)  # (num_envs, num_bodies)
    contact = torch.max(force_norms, dim=1)[0] > threshold  # (num_envs,)
    # 扩展为 (num_envs, num_feet) 以便与 LeggedLab 的2D输出对齐
    num_feet = forces.shape[1]
    return contact.unsqueeze(1).expand(-1, num_feet).float()


def height_scan_hpc(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.78  # 默认值改为 G1 的合理站立高度
) -> torch.Tensor:
    """
    获取基于传感器坐标系的高度扫描图，并包含 NaN/Inf 工业级清洗。
    """
    # 提取传感器
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # 1. 原始物理计算：地形相对高度 = 传感器Z - 击中点Z - 目标离地偏移量
    # 结果含义：0.0 代表完美平地，正数代表脚下有坑（击中点低），负数代表踩到台阶（击中点高）
    heights = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset

    # 2.  核心防护：清洗 NaN 和 Inf (防御物理引擎异常或射线射穿地图)
    # nan=0.0: 如果没测到，保守假设脚下是平地
    # posinf/neginf: 限制在一个物理上不可能达到的极限值，后续会被 config 中的 clip 截断
    heights = torch.nan_to_num(heights, nan=0.0, posinf=10.0, neginf=-10.0)

    return heights
