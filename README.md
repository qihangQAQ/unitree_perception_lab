# Unitree_perception_lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## 项目简介

Perception_lab 是一个面向复杂地形人形机器人运动控制的感知强化学习框架。项目基于 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 构建，并在 [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab) 的基础上进行开发，主要研究对象为宇树 G1 29 自由度人形机器人。

项目包含高程图循环策略基线、三个独立消融任务，以及结合状态估计、交叉注意力和混合专家网络的高程图与深度图感知策略。此外，项目还保留了 Go2、H1 和 G1 29 自由度机器人的基础速度控制与动作模仿任务。


## 观测维度

G1 感知任务使用以下观测：

- **本体观测（96 维）：** 机身角速度（3）、投影重力（3）、速度指令（3）、关节位置（29）、关节速度（29）和上一帧动作（29）。
- **高程图（187 维）：** `17 x 11` 地形高度采样点。
- **深度图（384 维）：** `16 x 24 x 1` 单通道深度图像。
- **Critic 特权观测（404 维）：** 包含无噪声机器人状态、机身线速度、随机化物理参数、足端接触信息、足端局部高度扫描和全局高程图。

## 核心感知运动任务

三个消融任务均直接继承感知基线，每个任务只加入一种实验机制，彼此之间不叠加。

| 任务 | 技术栈与作用 |
|---|---|
| `Unitree-G1-29dof-Velocity-perception` | 96 维本体观测 + `17 x 11` 高程图，Actor 每步输入 283 维；LSTM + PPO；Critic 使用 404 维完整特权观测。 |
| `Unitree-G1-29dof-Velocity-perception-Exp1` | 特权观测消融：Actor 改用与 Critic 相同的 404 维完整特权观测，作为性能上界对照。 |
| `Unitree-G1-29dof-Velocity-perception-Exp2` | 地形边缘约束消融：在感知基线上加入虚拟地形边缘、脚踝体积点检测和穿透惩罚。 |
| `Unitree-G1-29dof-Velocity-perception-Exp3` | 下楼奖励消融：在感知基线上加入下楼前进奖励和停滞惩罚。 |
| `Unitree-G1-29dof-Velocity-perception-predict` | 在Unitree-velocity-perception的基础上，加入SSR落足点预测和虚拟膨胀体机制。目前在基础感知（网络结构整体并不复杂）任务中取得了较好的实验效果，可以作为elevation-mapping任务的baseline |
| `Unitree-G1-29dof-Velocity-perception-pro` | `5 x 96` 本体历史 + `17 x 11` 高程图；Multi-Head Cross-Attention + Old-HIM + 4-expert MoE；融合后的 Actor 输入为 147 维，Critic 为 404 维。 |
| `Unitree-G1-29dof-Velocity-perception-pro-Upgrade` | 继承 perception-pro，移除遍地梅花桩，使用姿态与交叉梅花桩掉落终止，并加入 yaw 角速度误差惩罚。 |
| `Unitree-G1-29dof-Velocity-depth` | `5 x 96` 本体历史 + `16 x 24 x 1` 深度图（384 维）；CNN + Multi-Head Cross-Attention + Old-HIM + 4-expert MoE；融合后的 Actor 输入为 147 维，Critic 为 404 维。 |
| `Unitree-G1-29dof-Velocity-depth-Upgrade` | 继承 depth，显式使用 Upgrade-terrain2，使用姿态与交叉梅花桩掉落终止，并加入 yaw 角速度误差惩罚。 |

`perception-pro` 和 `depth` 共用项目级 `unitree_rl_lab.rsl_rl_ext` 实现。扩展按 `algorithms`、
`modules`、`runners`、`storage` 和 `exporters` 分层，两个任务的 agent 配置只负责选择高程图或
深度图输入适配器；上游 `rsl_rl` 仍作为唯一训练后端，不在项目中复制其源码。

## 任务总结
| 任务 | 效果分析 |
|---|---|
|Unitree-velocity-perception|经过修改，解决了action_rate爆炸的问题；但是存在下楼滑步下楼的风险|
|Unitree-velocity-perception-predict（基础感知任务）|在Unitree-velocity-perception的基础上，加入SSR落足点预测和虚拟膨胀体机制。目前在基础感知（网络结构整体并不复杂）任务中取得了较好的实验效果，可以作为elevation-mapping任务的baseline|
|Unitree-velocity-perception-pro（进阶感知任务）|升级整体网络架构（CNN +Multi-Head cross-attention + MOE HIM ）目前在基础地形（Upgrade-terrain1）上表现良好，可以作为升级架构baseline，在进阶地形（Upgrade-terrain2）上，地形等级很高，但是表现一般，正在开发中|
|Unitree-velocity-perception-pro-Upgrade（进阶感知任务）|在pro的基础上针对进阶地形（梅花桩 + gap）进行了调整，整体表现较好，但是步态不是很好，有一些奇怪，但是可以完成任务|
|Unitree-velocity-Depth（深度图感知任务）|整体网络架构（CNN +Multi-Head cross-attention + MOE HIM ）目前在基础地形（Upgrade-terrain1）上表现良好|


## 其他任务

| 任务 | 机器人 | 功能 |
|---|---|---|
| `Unitree-Go2-Velocity` | Go2 | 四足机器人速度跟踪。 |
| `Unitree-H1-Velocity` | H1 | 人形机器人速度与步态控制。 |
| `Unitree-G1-29dof-Velocity` | G1 29 自由度 | 基础本体感知速度控制。 |
| `Unitree-G1-29dof-Velocity-Extra` | G1 29 自由度 | 使用另一套 G1 模型和运动奖励设计。 |
| `Unitree-G1-29dof-Mimic-Dance-102` | G1 29 自由度 | Dance-102 全身动作跟踪。 |
| `Unitree-G1-29dof-Mimic-Gangnanm-Style` | G1 29 自由度 | Gangnam Style 全身动作跟踪。 |

## 混合柱障碍地形预览

`MIXED_PILLARS_TERRAIN` 生成一块 12×12 m 平地，随机混合摆放长方体柱、圆柱和三棱柱各 10 根。
每根柱高独立在 0.5～1.5 m 内随机采样，朝向也随机。
圆柱半径随机取 0.15～0.5 m（直径 0.3～1.0 m），长方体截面的长、宽分别随机取 0.3～1.0 m，
等边三棱柱的截面边长随机取 0.3～0.9 m。
三类柱子的截面尺寸均偏向较小值采样，减少粗柱；`size_sampling_power=2.0`，调回 1.0 为均匀采样，增大则更偏细。
下面的预览命令直接读取
`Unitree-G1-29dof-Velocity-depth` 的 play 配置中的地面材质、天空光照及渲染配置，不按形状着色。
柱子之间至少间隔 0.8 m，距离场地边缘至少 0.5 m。固定种子可复现同一布局。

```bash
python scripts/rsl_rl/visualize_terrain.py \
  --terrain_cfg unitree_rl_lab.terrains.terrain_generator_cfg:MIXED_PILLARS_TERRAIN \
  --appearance-task Unitree-G1-29dof-Velocity-depth \
  --num_levels 1 --seed 0 --rendering_mode balanced
```

`--appearance-task` 只借用外观配置，不加载机器人或策略模型。这里使用与 `play.py` 默认相同的
`balanced` 渲染模式；`performance` 会改变灯光采样和降噪设置，不适合用来对照训练任务的外观。

场地大小和默认种子在 `terrains/terrain_generator_cfg.py` 的 `MIXED_PILLARS_TERRAIN` 中调整，
各类数量、尺寸、高度及间距在 `terrains/custom_terrain/mixed_pillars_cfg.py` 中调整。
换 `--seed` 可生成另一种随机布局。当前仅用于场地预览，几何中心不代表安全机器人出生点。

## 拼接地图 MPPI + FDM 联合演示

`play_mixed_pillars.py` 保留原脚本名，现默认使用 `FDM_ATLAS_TERRAIN`：48×36 m、12 个相连分区，
包含柱阵、楼梯、坡道、多层平台、墙体及迷宫。运行一个 G1，加载冻结的 perception-predict 行走策略和 FDM。
地图生成代码为 `source/unitree_rl_lab/unitree_rl_lab/fdm/terrain_atlas.py`，
推理环境为 `tasks/fdm/atlas_env_cfg.py`，注册任务为 `Unitree-G1-29dof-FDM-Atlas`。
不指定起终点时默认使用 B1 的楼梯平台场景，起点 `(-22.8, 0)`，目标 `(-13.2, 0)`，观察上楼梯还是平地绕行。
几何检查只用于起终点的局部支撑检查，不给 MPPI 提供路线、可通行分类或全局地图。

```bash
python scripts/fdm/play_mixed_pillars.py \
  --checkpoint logs/rsl_rl/Unitree-Velocity_perception/2026-08-30_12-12-16_perception-predict/model_23500.pt \
  --fdm-checkpoint logs/fdm_g1/stage2_mppi_v1/fdm_round_004.pt \
  --seed 0 --mppi-candidates 256 --rendering_mode balanced
```

机器人先站稳 2 秒、填满历史观测；底层策略以 50 Hz 执行，MPPI 每 0.5 秒更新速度序列并执行第一段。
蓝点显示 FDM 大范围扫描，红点显示策略局部扫描；绿色直箭头显示实际下发的线速度，绿色弧箭头显示转向角速度；
黄色圆环是目标的 0.4 m 到达范围。蓝点保留最近一次规划的采样位置，红点随局部扫描更新。
材质、灯光沿用 depth 任务，相机可自由操作。

可添加 `--goal X Y` 指定目标，或同时添加 `--start X Y YAW` 指定出生点（世界坐标，偏航角单位为弧度）。
地图范围为 `X ∈ [-24,24]、Y ∈ [-18,18]`。`--zone A1` 至 `--zone C4` 可选分区预设起终点；
显式 `--start`、`--goal` 分别覆盖对应预设，位置不会自动挪动。
未指定 `--zone` 时，根据显式目标（或仅提供的起点）选择所在分区，其余端点采用该区预设。
例如原来的 `--goal 3.95 -1.85` 会选择 B3 起点 `(1.4,0)`，目标位于该区缓坡上，地面高度约 0.4625 m。
起点要求半径 0.55 m 的平坦支撑区域；目标支持平地、平台或不超过 30° 的连续坡面，
要求半径 0.50 m 内无墙体、楼梯边或落差。机器人重置高度和黄色目标标记均根据实际网格表面计算。
这些是端点设置要求，沿途楼梯和陡坡是否通过仍由 FDM 与策略决定。
固定种子可复现地图；相机启动时对准所选起终点，之后可自由操作。

```bash
# 楼梯与平地绕行对比：使用上面命令，添加 --zone B1，不指定 --goal。
# 陡坡与普通楼梯对比：--zone B2
# 高台阶与缓坡对比：--zone B3
# 多层平台：--zone C2

# 查看同一张完整地图，不加载机器人和模型。
python scripts/rsl_rl/visualize_terrain.py \
  --terrain_cfg unitree_rl_lab.terrains.terrain_generator_cfg:FDM_ATLAS_TERRAIN \
  --appearance-task Unitree-G1-29dof-Velocity-depth \
  --num_levels 1 --seed 0 --rendering_mode balanced
```

到达后持续执行零速度站立，关闭窗口退出。身体/手碰撞、摔倒、越界、恢复后仍无进展或超时会暂停并保留现场。
`--timeout` 默认 120 仿真秒；`--stall-time` 默认 15 仿真秒，最多重启 MPPI 候选序列两次。
脚本不创建输出目录或指标文件，仅在终端提示启动、到达或停止原因。无需 `--output`。

## 空格切换随机目标的连续联合推理

新入口为 `scripts/fdm/play_interactive_goals.py`，使用同一张 48×36 m 地图和同一套模型。
默认从 B1 楼梯旁的平地出生，站稳约 2 秒后自动生成第一个目标并出发。

```bash
python scripts/fdm/play_interactive_goals.py \
  --checkpoint logs/rsl_rl/Unitree-Velocity_perception/2026-08-30_12-12-16_perception-predict/model_23500.pt \
  --fdm-checkpoint logs/fdm_g1/stage2_mppi_v1/fdm_round_004.pt \
  --seed 0 \
  --goal-radius 5.0 \
  --goal-min-distance 2.0 \
  --mppi-candidates 256 \
  --rendering_mode balanced
```

每次从机器人当时的真实 XY 位置周围 2～5 m 环形区域内，按面积均匀采样目标；
地形筛选后保留有效落点。目标支持平地、宽敞平台和缓坡，排除墙体、柱顶、地图边缘、
悬空或支撑面不连续的位置。落点筛选不判断整条路线是否可通行，上楼梯还是绕行仍由模型决定。
目标采样使用独立随机数流，后续换目标不会重置这个随机数流。

操作流程：

1. 首个目标自动开始。黄色圆环显示目标，状态窗口显示轮次、目标坐标和剩余平面距离。
2. 进入目标 0.4 m 内、线速度低于 0.15 m/s、转向速度低于 0.2 rad/s，持续 0.5 秒后判定到达。
3. 到达后保持零速度站立，仿真继续运行。点击仿真视口获得焦点，按一次 **空格**生成下一目标并出发。
4. 行走过程中按空格会被忽略；长按不会连续换目标。空格由演示脚本接管，手动暂停或恢复使用仿真工具栏。

每轮结束后机器人保持当前位置，策略记忆和 FDM 观测历史持续更新；新目标只重置 MPPI 的上一轮计划及导航计时。
相机在每轮开始时对准当前机器人和新目标，此后可以自由操作。
`--zone A1` 至 `--zone C4` 只选择初始平地出生区域，后续目标可以跨分区；
也可用 `--start X Y YAW` 指定平地出生位置。这个入口不接受固定 `--goal` 或 `--headless`。

单轮超时或长期无进展后，机器人停止行走并等待空格换目标；
在指定半径内尝试 256 次仍没有有效落点时，保持等待，按空格可再试。
身体/手碰撞、摔倒、越界或异常观测会暂停并保留现场。
键盘订阅采用 Carbonite 的输入事件接口，参见 [NVIDIA 输入接口文档](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/109.0.6/carb.input/carb.input.IInput.html)。

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
- Install the Unitree RL IsaacLab standalone environments.

    ```bash
    conda activate lab
    ./unitree_rl_lab.sh -i
    # restart your shell to activate the environment changes.
    ```



## Training and Evaluation
 Listing the available tasks:

    ```bash
    ./unitree_rl_lab.sh -l # This is a faster version than isaaclab

Train the recurrent height-map baseline:

```bash
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity-perception
```

Train one of the independent ablations:

```bash
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity-perception-Exp1
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity-perception-Exp2
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity-perception-Exp3
```

Train the perception task with foothold prediction and penetration guidance:

```bash
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity-perception-predict
```

Train a structured cross-attention policy:

```bash
# Height-map + Old-HIM + cross-attention + MoE
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity-perception-pro

# perception-pro + revised terrain/termination/yaw objectives
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity-perception-pro-Upgrade

# Depth + Old-HIM + cross-attention + MoE
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity-depth

# depth + terrain2/revised termination/yaw objectives
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity-depth-Upgrade
```

Run a trained policy by replacing `-t` with `-p`:

```bash
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity-perception-pro
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity-perception-pro-Upgrade
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity-depth
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity-depth-Upgrade
```

### Motion-imitation data preparation

The repository tracks the reference motions as CSV files. Generate the NPZ file expected by a mimic task before training it:

```bash
python scripts/mimic/csv_to_npz.py \
    -f source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/dance_102/G1_Take_102.bvh_60hz.csv \
    --input_fps 60
```

Use the corresponding Gangnam Style CSV path to prepare the other mimic task.

## Deploy

After the model training is completed, we need to perform sim2sim on the trained strategy in Mujoco to test the performance of the model.
Then deploy sim2real.

### Setup

```bash
# Install dependencies
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
# Install unitree_sdk2
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF # Install on the /usr/local directory
sudo make install
# Compile the robot_controller
cd unitree_rl_lab/deploy/robots/g1_29dof # or other robots
mkdir build && cd build
cmake .. && make
```

### Sim2Sim

Installing the [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco?tab=readme-ov-file#installation).

- Set the `robot` at `/simulate/config.yaml` to g1
- Set `domain_id` to 0
- Set `enable_elastic_hand` to 1
- Set `use_joystck` to 1.

```bash
# start simulation
cd unitree_mujoco/simulate/build
./unitree_mujoco
# ./unitree_mujoco -i 0 -n eth0 -r g1 -s scene_29dof.xml # alternative
```

```bash
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl
# 1. press [L2 + Up] to set the robot to stand up
# 2. Click the mujoco window, and then press 8 to make the robot feet touch the ground.
# 3. Press [R1 + X] to run the policy.
# 4. Click the mujoco window, and then press 9 to disable the elastic band.
```

### Sim2Real

You can use this program to control the robot directly, but make sure the on-borad control program has been closed.

```bash
./g1_ctrl --network eth0 # eth0 is the network interface name.
```
