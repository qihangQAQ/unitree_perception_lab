# Unitree G1 感知前向动力学模型（FDM）实现方案


> 目标任务：`Unitree-G1-29dof-FDM-Rollout`
>
> 基础任务：`Unitree-G1-29dof-Velocity-perception-predict`


## 1. 目标与范围

本项目的第一阶段目标是搭建一条完整、可验证的 FDM 流水线：

1. 加载已经训练好的 G1 高程图感知策略，并冻结其参数。
2. 在专门的 FDM 仿真任务中，向该策略输入以“前进 + 转向”为主的时序速度指令。
3. 按照原 FDM 项目的多时间尺度逻辑在线采集机器人状态、本体感知、FDM 大范围高程图、速度指令和碰撞标签。
4. 将每一轮 rollout 切分为固定预测长度的监督学习样本，并立即训练当前 FDM。
5. 按照“在线采集一轮 → 训练若干 epoch → 验证并保存 → 继续采集”的方式迭代训练感知 FDM，预测未来 SE(2) 运动轨迹与碰撞概率。
6. 建立离线评估和仿真闭环评估，确保模型可以在后续接入采样式规划器。

注意， 修改中不包含：

- 不重新训练或联合训练原感知 locomotion policy；这里冻结的是 locomotion policy，**不是** FDM。FDM 网络是本项目必须训练的主体。
- 不修改原任务 `Unitree-G1-29dof-Velocity-perception-predict` 的行为。
- 不实现深度图 FDM；先完成高程图版本。
- 不在第一阶段实现 MPPI/采样式导航规划器，只预留接口。
- 不直接照搬 ANYmal 的机器人状态维度；碰撞 link 按 G1 调整，地形直接使用原 FDM 的合并 USD。

## 2. 总体架构

系统由 rollout、数据切分和 FDM 训练三个相互配合的部分组成，并由 `FDMRunner` 按 collection round 循环调度：

```text
冻结的 G1 感知策略
        │  50 Hz，输出 29 维关节目标
        ▼
Unitree-G1-29dof-FDM-Rollout
        │  采集状态、历史、本体感知、大范围高程图、command、碰撞
        ▼
轨迹数据集与窗口切分
        │  history=10，future horizon=10
        ▼
G1 感知 FDM
        ├─ 未来 SE(2) 轨迹
        └─ 未来累计碰撞概率
                │
        └── 训练若干 epoch、验证、保存 checkpoint
                    │
                    └── 返回下一轮在线 rollout
```

核心原则如下：

- locomotion policy 负责把高层速度指令转换为 29 维关节动作。
- locomotion policy 全程冻结；每一轮在线采集结束后，只更新 FDM 网络参数。
- FDM 学习的是“冻结策略 + G1 机器人 + 地形”构成的闭环动力学，而不是裸机器人动力学。
- FDM 的 action 是高层速度指令 `[v_x, v_y, \omega_z]`，不是 29 维关节动作。
- FDM 的轨迹输出保留横向位移 `y`，即使 `v_y` 固定为零；人形机器人仍可能发生侧向漂移、滑动或碰撞偏转。

## 3. 新任务的职责和继承关系

新任务注册名：

```text
Unitree-G1-29dof-FDM-Rollout
```

新任务配置继承：

```text
velocity_perception_predict_env_cfg.RobotEnvCfg
```

继承后保持不变的部分：

- G1 29-DoF 机器人资产与关节顺序。
- 原感知策略的 283 维 actor 输入接口。
- 原策略使用的局部高程扫描器。
- 关节位置 action：29 维。
- `sim.dt = 0.005 s` 和 `decimation = 4`。
- 原策略训练时的 observation scale、clip 和 corruption 语义。

只在新 FDM 任务中覆盖或增加：

- FDM 专用的 0.5 s 时序 command 生成器。
- `1 × 60 × 46` 大范围高程图扫描器。
- FDM state/proprioception 数据接口。
- 即时碰撞标签与延迟碰撞 termination。
- 原 FDM 合并 USD 的加载、environment origin 分配和 reset 分布。
- rollout 所需的 episode 长度、数据记录信息和元数据。
- 关闭 reward、训练 curriculum 和随机 push 等与 rollout 无关的机制。

冻结策略 checkpoint 由采集脚本通过命令行传入，不在环境配置中硬编码。环境依然接收 29 维关节 action，采集 runner 每 0.02 s 调用一次冻结策略并把结果送入环境。

这样不会改变原任务；原任务仍然保留立即 termination 等原始设定。

## 4. 时间尺度与频率

这是实现中最容易混淆、也必须通过自动测试锁死的部分。

| 层级 | 周期 | 频率 | 作用 |
|---|---:|---:|---|
| 物理仿真 | 0.005 s | 200 Hz | PhysX 积分；接触传感器在此频率更新 |
| locomotion policy / env step | 0.020 s | 50 Hz | 每 4 个物理步执行一次冻结策略，输出一次 29 维关节 action |
| 原策略局部高程图更新 | 0.100 s | 10 Hz | 保持当前任务配置；策略在 50 Hz 下读取最近一次缓存的局部高程图 |
| FDM history 采样 | 平均 0.050 s | 平均 20 Hz | 保存 state/proprioception 历史；每个 command 周期得到 10 个历史点 |
| FDM command 更新 | 0.500 s | 2 Hz | 更新一次 `[v_x, v_y, \omega_z]`，随后保持 25 个 policy step |
| 常规完整 FDM 帧 | 0.500 s | 2 Hz | 保存 history、大范围高程图、当前状态以及下一段要执行的 command |
| FDM 预测 horizon | 5.000 s | 10 个 command step | `H = 10`，每步 0.5 s |

### 4.1 20 Hz 到底采集什么

20 Hz 不是完整数据集帧频率，只是 FDM 输入历史的内部采样频率。

每个 history tick 采集：

- 当前机器人位姿/碰撞状态。
- 当前策略可见的 96 维本体感知向量。

不会在 20 Hz 下重复保存 `60 × 46` 大范围高程图，也不会在 20 Hz 下更换 command。

由于 policy step 是 0.02 s，而 0.05 s 等于 2.5 个 policy step，无法直接做到严格等间隔。因此第一版复现原 FDM 的做法，在 2 个和 3 个 policy step 之间交替采样：

```text
0.04 s, 0.06 s, 0.04 s, 0.06 s, ...
```

长期平均周期为 0.05 s，即平均 20 Hz。每条记录同时保存时间戳，不能只依赖数组索引推断真实时间。

后续可以增加“50 Hz 暂存 + 精确重采样到 20 Hz”的消融版本，但不作为第一版默认实现。

### 4.2 一个 0.5 s command step 内发生什么

设完整 FDM 帧 `F_k` 在 `t_k` 被记录：

1. 在 `t_k` 保存当前状态、10 点历史和当前大范围高程图。
2. 生成并记录 command `u_k = [v_x, v_y, \omega_z]`。
3. 在接下来的 25 个 50 Hz policy step 中保持 `u_k` 不变。
4. 冻结策略每 0.02 s 根据自己的 283 维 observation 产生一次 29 维 action。
5. 接触传感器在内部 200 Hz 检查碰撞。
6. 正常情况下在 `t_k + 0.5 s` 记录 `F_{k+1}`。
7. `F_{k+1}` 中的状态是 `u_k` 对应的监督结果；`F_{k+1}` 中保存的新 command 是下一段要执行的 `u_{k+1}`。

因此监督关系必须是：

```text
(F_k 的历史、地图、u_k)  ──预测──>  F_{k+1} 的状态和碰撞
```

不能把 `F_{k+1}` 中的新 command 错配给已经发生的运动。

### 4.3 碰撞造成的非周期事件帧

如果 command 尚未执行满 0.5 s 就发生目标 link 碰撞：

- 立即把碰撞状态写成一个额外事件帧，不等待下一个 0.5 s 边界。
- 该碰撞帧是前一个 command 的结果。
- 保存实际 `delta_t`，因为这一次转移可能短于 0.5 s。
- 随后延迟一个 policy step 才 termination/reset。
- 切片时不允许窗口跨越 reset；碰撞后的未来 target 使用首次碰撞状态冻结填充。

这与原 FDM 的核心逻辑一致：先记录碰撞结果，再 reset，而不是让 termination 把碰撞观测吞掉。

项目中不定义 5 Hz 数据采集层。若以后规划器需要 5 Hz replanning，它属于规划频率，不属于本次 rollout 数据格式。

## 5. Observation 与数据维度

### 5.1 冻结 locomotion policy 输入

原策略接口完全不变：

| 字段 | 维度 |
|---|---:|
| base angular velocity | 3 |
| projected gravity | 3 |
| velocity command | 3 |
| joint position relative | 29 |
| joint velocity relative | 29 |
| last action | 29 |
| 原策略局部高程图 | 187 (`17 × 11`) |
| 合计 | **283** |

策略输入频率是 50 Hz，但其 187 维局部高程图按照当前任务配置每 0.1 s 更新一次。

### 5.2 FDM 本体感知

“本体感知不变”的具体实现是：直接取冻结策略 283 维 observation 的前 96 维，而不是为 FDM 再定义一套不同的缩放和噪声。

| 字段 | 维度 |
|---|---:|
| base angular velocity | 3 |
| projected gravity | 3 |
| velocity command | 3 |
| joint position relative | 29 |
| scaled joint velocity relative | 29 |
| last policy action | 29 |
| 合计 | **96** |

一个完整输入历史的形状为：

```text
proprio_history: [10, 96]
```

这样做有三个优点：

- 不改变现有感知策略接口。
- FDM 训练时看到的 proprioception 与实际闭环运行时一致。
- 避免照搬 ANYmal 针对 12 个关节设计的 132 维 proprioception，并避免按 29 个关节机械扩展到 302 维。

### 5.3 原始 FDM state

每个 history tick 保存干净的监督状态：

| 字段 | 维度 | 说明 |
|---|---:|---|
| base position in world | 3 | `[x, y, z]` |
| base quaternion in world | 4 | `[q_x, q_y, q_z, q_w]` |
| navigation collision | 1 | 当前时刻 torso 或左右 wrist/hand 是否发生目标碰撞 |
| 合计 | **8** | |

构造网络样本时，把最近 10 个 raw state 转到当前机器人坐标系，并只保留导航所需量：

```text
relative_state_history: [10, 5]
每点 = [x_rel, y_rel, sin(yaw_rel), cos(yaw_rel), collision]
```

因此 state/proprioception GRU 的单时刻输入维度初始为：

```text
5 + 96 = 101
```

`z/roll/pitch` 暂不作为 FDM 导航轨迹输出，但原始数据中保留完整 3D pose，后续可以做台阶/摔倒消融而不必重新采集。

### 5.4 FDM 大范围高程图

第一版固定为：

```text
shape      = [1, 60, 46]
resolution = 0.10 m
physical x = [-0.5 m, 4.0 m]    # 机器人后方少量区域 + 主要前向区域
physical y = [-2.95 m, 2.95 m]
alignment  = torso yaw frame
```

这对应原 FDM 高程图版本的 `60 × 46`，而不是 `64 × 46`。原项目使用 `(4.5 m, 5.9 m)`、0.1 m 分辨率的 grid，经轴顺序整理后得到 `60 × 46`。

该扫描器与策略的 `17 × 11` 局部扫描器同时存在：

- `height_scanner`：原策略使用，不能替换。
- `fdm_height_scanner`：仅供数据记录和 FDM 使用。

初始实现保存相对机器人参考高度的 float16 高程，shape 为 `[1, 60, 46]`。建议 clip 范围 `[-1.0, 1.5] m`，并保存无效射线 mask 或使用固定 sentinel，禁止把无命中点静默当作平地。

采集 FDM 大范围图时复现原项目的门洞识别：在原始向下射线命中点的水平坐标，从世界高度 0.5 m 再向下、向上探测；若上方命中低于原始顶部命中且上下表面净空大于 1.25 m，就用下方地面命中构造该点高程。该修正只作用于 FDM 数据和后续 FDM 推理，冻结 locomotion policy 的局部高程图接口保持训练时定义。门洞阈值须在 G1 的实际通行场景中验证。

### 5.5 Command

为了与原 FDM 和未来 planner 接口兼容，command 仍保留 3 维：

```text
command = [v_x, v_y, omega_z]
shape   = [3]
```

推荐初始范围：

```text
v_x     ∈ [0.0, 1.2] m/s
v_y     ∈ [-0.2, 0.2] m/s
omega_z ∈ [-1.0, 1.0] rad/s
```

command 以时间相关方式生成，而不是每 0.5 s 完全独立跳变：

- 60%：相关的前进 + 转向圆弧。
- 20%：近似直行。
- 15%：低速或原地转向。
- 5%：停止。

具体比例和范围全部进入配置文件，先通过 rollout 验证策略稳定范围，再扩大覆盖范围。`v_y` 虽然固定为零，仍然保留在数据和模型接口中。

## 6. 碰撞标签与延迟 termination

### 6.1 第一版碰撞定义

Isaac Lab 的接触传感器检测 rigid body link，而不是 joint。G1 29-DoF 的 wrist roll/pitch/yaw 是关节；碰撞检测应匹配这些关节对应的 wrist link 和末端 rubber-hand link。

第一版主碰撞 link 集合为：

```text
body_names = [
    "torso_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "left_rubber_hand",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_rubber_hand",
]
force threshold = 1.0 N
```

正式运行前必须从 articulation/contact sensor 查询实际 body names，并断言上述 link 全部匹配成功，禁止因正则或资产命名不一致而静默漏检。

只要该时刻集合中任意 link 的接触力超过阈值，`collision_now = 1`，否则为 0。主标签仍然是一个标量，不按 link 数量展开。

标签是“该时间点/区间是否发生碰撞”的布尔值，不是碰撞次数：

- 同一时刻多个 torso/hand link 或多个接触点同时命中，仍然只记一个 `1`。
- 接触连续存在多个物理步，不会被解释成多次独立碰撞样本。
- shoulder、elbow、前臂和腿部 link 暂不加入主碰撞集合；后续若扩大集合，仍通过 OR 生成单一主标签。

为了后续分析，可以额外记录但不用于第一版训练的诊断字段：

```text
contact_groups: [torso, left_hand, right_hand]
shape: [3]
```

诊断分组不改变训练标签：`collision_now` 等于三个分组的 OR。

### 6.2 延迟 termination

新任务中同时维护两个判定：

1. `collision_now`：读取最新接触，立即作为标签。
2. `navigation_contact_delayed`：对相同的 torso + 双侧 wrist/hand link 集合读取延迟后的接触历史，用于 termination。

默认延迟：

```text
delay = 1 policy step = 0.02 s = 4 physics steps
```

FDM 新任务将 contact sensor 的 `history_length` 从当前的 3 增加到至少 6，以满足延迟检查；该修改只存在于 FDM 子类中。碰撞标签和 delayed termination 必须共用同一组 link、同一力阈值，避免出现“已经标为碰撞但仍持续卡在障碍物上”的语义不一致。

事件顺序必须是：

```text
碰撞发生
  → collision_now = 1
  → 强制写入碰撞事件帧
  → 下一 policy step delayed termination = 1
  → env reset
```

普通 timeout、人工 watchdog reset 或初始化失败不标为 collision，必须用独立的 `termination_reason` 保存。

### 6.3 训练标签语义

对预测 horizon 内的碰撞标签使用累计语义：

```text
collision_target[h] = 1
当且仅当从预测开始到第 h 个 command step 已经发生过碰撞
```

一旦首次碰撞：

- 后续 pose target 固定为首次碰撞 pose。
- 后续 collision target 保持为 1。
- 后续 action 仍保留原 command 序列，避免网络通过人为清零 action 直接识别碰撞。
- 窗口绝不跨越 reset 后的新 episode。

## 7. 地形方案

### 7.1 直接使用原 FDM 合并 USD

第一版直接使用原 FDM 当前启用的合并 USD，不使用本项目的程序化 terrain generator 生成训练地形：

```text
/home/qihang/code/fdm/exts/fdm/data/Terrains/
navigation_terrain_wall_usd_merge_large_single_object_maze.usd
```

该文件约 26 MB，包含 wall、single-object 和 maze 等原 FDM 训练场景。第一版保持 USD 几何、比例和障碍尺寸不变，只针对 G1 调整出生高度、姿态和 reset 安全检查。

USD 路径必须作为配置项 `terrain_usd_path` 或命令行参数传入。开发环境可以默认指向上述原仓库文件，但运行代码不能在多个模块中散落硬编码绝对路径。若后续需要仓库自包含，可以复制同一个 USD 文件或用外部资产目录部署；无论资产放置位置如何，使用的都必须是同一个原始 USD。

### 7.2 移植原 FDM 的 USD importer

原 FDM 不是仅用标准 `TerrainImporterCfg` 打开文件，而是使用：

```text
NavTerrainImporterCfg(
    terrain_type="usd",
    usd_path=terrain_usd_path,
    usd_uniform_env_spacing=10.0,
)
```

当前仓库尚未包含 `NavTerrainImporter`。为了忠实使用该 USD，需要从原 FDM 的 nav-suite 移植最小必要功能：

- 读取 USD world bounding box。
- 以 `usd_uniform_env_spacing = 10.0 m` 在 bounding box 内建立可用 environment origins。
- 为并行环境分配 origin，并向 reset 逻辑提供 origin/region id。
- 支持固定 origin 列表和 split，保证 validation/test 可复现。
- 移植原 FDM 的 `TerrainAnalysisRootReset` 出生保护：构建高度图，过滤墙内/近墙候选点，按机器人占地范围的局部最大高度设置出生高度。
- 在上述地图预筛选后继续执行 G1 落地检查，验证双脚着地、机身直立且 navigation links 没有与墙体或障碍物初始穿透。

新 rollout 任务不使用 `UPGRADE_TERRAIN1`，也不使用 `TerrainImporterCfg(terrain_type="generator")`。

当前 `tasks/fdm/mdp/terrains` 中的 pillar、single-object、stairs/ramp 生成器仍保留，因为现有感知配置已经导入它们；但它们不进入本任务的主训练地形配置。

### 7.3 单个 USD 内的数据划分

数据集不能先随机切窗口再划分，否则相邻窗口和同一地形会泄漏到不同集合。

正确顺序：

1. importer 根据 10 m spacing 为每个可用 origin 分配稳定的 `usd_origin_id` 和二维 grid 坐标。
2. 在采集前，把完整 origin/cell 固定分到 train、validation、test；同一个 origin 永远只能属于一个 split。
3. rollout episode 继承所在 origin 的 split。
4. 在各 split 内部完成 trajectory window 切分，禁止切片后再随机划分。

推荐使用空间分块而不是简单逐格交错，以降低相邻区域的几何泄漏：

- train：约 80% 的连续空间块。
- validation：约 10% 的独立空间块，训练开始前一次性采集并保持固定。
- test：约 10% 的独立空间块，仅用于最终评估。

manifest 必须记录 USD 文件路径、文件 hash、bounding box、10 m spacing、每个 split 的 origin id 和坐标。若 USD 内存在可识别的场景类别，还应记录 region/semantic 名称并按类别分层划分。

## 8. 原始 rollout 数据格式

每个环境保存变长 trajectory，而不是把 reset 前后拼成一条有效轨迹。一个 command-level frame 的推荐字段如下：

| 字段 | shape | dtype | 说明 |
|---|---:|---|---|
| `state_history_raw` | `[10, 8]` | float32 | 10 个 state history 点 |
| `proprio_history` | `[10, 96]` | float32 | 策略可见的本体感知历史 |
| `history_timestamps` | `[10]` | float64 | 最新点在 index 0；保留 2/3 policy step 交替采样的真实时刻 |
| `height_map` | `[1, 60, 46]` | float16 | 当前大范围高程图 |
| `height_map_invalid` | `[1, 60, 46]` | bool | 无命中射线 mask |
| `command` | `[3]` | float32 | 从当前帧开始执行的 command |
| `command_plan` | `[10, 3]` | float32 | 碰撞前已经生成的未来 10 步指令计划 |
| `has_outgoing_command` | `[1]` | bool | 事件/terminal 帧为 false，禁止作为窗口起点 |
| `timestamp` | `[1]` | float64 | 仿真时间 |
| `delta_t` | `[1]` | float32 | 距上一完整/事件帧的真实间隔 |
| `collision_now` | `[1]` | bool | 当前帧是否为目标碰撞 |
| `contact_groups` | `[3]` | bool | 诊断用 link 分组接触 |
| `terminated` | `[1]` | bool | 是否触发 termination |
| `truncated` | `[1]` | bool | 是否为时间或采集长度截断 |
| `termination_reason` | `[1]` | int8 | none/collision/timeout/watchdog |
| `episode_id` | `[1]` | int64 | 防止跨 reset 切片 |
| `usd_origin_id` | `[1]` | int32 | importer 分配的 10 m origin/cell 索引 |
| `usd_region_split` | `[1]` | int8 | train/validation/test 空间分区 |

额外的 dataset-level metadata：

- schema version。
- 任务名、git commit、Isaac Lab/Isaac Sim 版本。
- policy checkpoint 路径及 hash。
- joint/body 名称和顺序。
- observation scale/clip 配置。
- 所有频率、地图范围和 command 范围。
- collision link 列表、分组和 force threshold。
- 原 USD 路径与文件 hash、world bounding box、10 m origin spacing 和 split origin 列表。
- 随机种子、reset 配置和 domain randomization 配置。

第一版使用分片文件和 manifest，不创建一个超大的 pickle：

```text
datasets/fdm_g1/<dataset_name>/
├── manifest.json
├── train/
│   ├── shard_00000.pt
│   └── ...
├── val/
└── test/
```

每个 shard 只包含完整 episode，写盘采用临时文件 + 原子 rename，避免采集中断后留下看似有效的半文件。

## 9. 监督样本切分

默认参数：

```text
history_length      = 10
history_rate        = 20 Hz（平均）
history_duration    = 0.5 s
prediction_horizon  = 10
command_timestep    = 0.5 s
prediction_duration = 5.0 s
```

对 command-level trajectory 中的起点 `k`，生成：

### 输入

```text
relative_state_history : [10, 5]
proprio_history        : [10, 96]
height_map             : [1, 60, 46]
future_commands        : [10, 3]
```

### 监督目标

```text
future_pose     : [10, 4]  # [x_rel, y_rel, sin(yaw_rel), cos(yaw_rel)]
future_collision: [10]     # 累计碰撞标签
valid_mask      : [10]     # padding/loss mask
```

切片约束：

- 起点自身不能已经 collision。
- 正常窗口需要同一 episode 内的未来 10 个 command step。
- 碰撞窗口在首次碰撞处截断并冻结填充到 horizon 末尾。
- timeout/watchdog/reset 不用碰撞冻结逻辑，缺失未来部分通过 `valid_mask` 排除；默认不使用这类不完整窗口训练。
- 不允许窗口跨 `episode_id`、`usd_origin_id` 或 reset。
- 训练窗口可以随机重叠采样；验证/测试使用固定、可复现的起点。

类别平衡建议：

- 原始数据保留自然碰撞率。
- DataLoader 通过 sampler 把含碰撞窗口比例调整到约 30%～40%，不复制或修改原始文件。
- 保留无碰撞、直行、转向和低运动量样本。
- 小运动窗口最多占训练 batch 的约 10%，与原 FDM 的过滤思路一致。

## 10. FDM 网络设计

第一版以原高程图 FDM 的结构为参考，但按 G1 输入维度重建，而不是直接复制 ANYmal checkpoint。

### 10.1 编码器

1. State + proprioception encoder

```text
input  : [B, 10, 101]
GRU    : 2 layers
hidden : 64
output : [B, 64]
```

2. Height-map encoder

```text
input  : [B, 1, 60, 46]
CNN    : 4 stages，通道建议 [32, 64, 128, 256]
output : [B, 512]
```

3. Command encoder

```text
input  : [B, 10, 3]
MLP    : 3 → 16
output : [B, 10, 16]
```

### 10.2 时序预测器

将 proprio/state latent、height-map latent 和每一步 command latent 输入 command-level GRU：

```text
hidden size = 128
steps       = 10
```

保留 command GRU 的全部 10 个输出并展平为一个联合 latent，再用两个 all-at-once decoder 同时输出完整 horizon；不是对每一步独立复用同一个 head。这样后面时刻的预测可以显式利用完整的未来 command plan。

输出两个 head：

- motion head：每步输出 3 维速度/位姿修正 `[delta_x, delta_y, delta_yaw]`。
- collision head：每步输出 1 个 collision logit。

motion head 使用理想 command 积分作为 residual baseline：

```text
ideal twist = [v_x, v_y, omega_z]
ideal step  = ideal twist × 0.5 s
prediction  = ideal step + learned correction
```

再按照 SE(2) 逐步积分得到未来累计轨迹。该设计让模型重点学习速度跟踪误差、地形影响、滑动、转弯误差和碰撞后的停止，而不是从零学习基本积分。

### 10.3 损失

第一版损失：

```text
L = 1.7 * L_position
  + 1.7 * L_heading
  + 2.0 * L_collision
  + 1.0 * L_stop_after_collision
```

- `L_position`：未来相对 `x/y` 的 Smooth-L1 或 MSE。
- `L_heading`：`sin/cos(yaw)` 损失，并监控实际 wrap 后 yaw error。
- `L_collision`：`BCEWithLogitsLoss`，通过 sampler 或 `pos_weight` 处理类别不平衡。
- `L_stop_after_collision`：首次碰撞后预测轨迹不再继续穿过障碍。
- 所有 loss 应用 `valid_mask`。

先实现单模型；ensemble、不确定性估计和 energy prediction 放到后续阶段。

## 11. 在线交替采集与训练

FDM 是本项目必须训练的模型。冻结的只有 locomotion policy，FDM 参数会在每个 collection round 后更新。

第一版按照原 FDM `FDMRunner.train()` 的控制流实现，而不是先收集全部数据、最后只训练一次：

```text
初始化冻结的 locomotion policy
初始化 G1 FDM 网络

采集一次固定 validation dataset

for collection_round in range(collection_rounds):
    1. 使用冻结 locomotion policy 在原 FDM USD 上在线 rollout
    2. 收集这一轮新的 trajectory replay buffer
    3. 切分成本轮 FDM 监督样本
    4. 用本轮样本训练当前 FDM 若干 epochs
    5. 在固定 validation dataset 上评估
    6. 保存本轮 dataset manifest、指标和 FDM checkpoint
```

与“每一个仿真 step 都进行反向传播”不同，梯度更新发生在一轮 rollout 完成之后。仿真采集阶段使用 `torch.inference_mode()`，训练阶段再启动 DataLoader 和反向传播。

与原项目对齐的默认值为：

```text
collection_rounds = 20
epochs_per_round  = 8
num_samples       = 80000  # 每轮切出的目标窗口数
```

smoke test 和 pilot 阶段可以临时减小这些数值，但正式默认逻辑保持一致。第一版每轮训练使用本轮新采集的数据；每轮原始 shard 和 checkpoint 都保留，后续若需要累计 replay 或跨轮混采，应作为显式配置和消融实验加入。

第一版 command 数据来自时间相关随机 command，FDM 不控制 locomotion policy。后续如果接入原项目的 sampling planner，可以逐步混入由当前 FDM 规划产生的 command，但这不属于首版验收范围。

`FDMRunner` 负责整个在线循环，职责划分为：

- `Unitree-G1-29dof-FDM-Rollout`：仿真、传感器、command、碰撞和 reset。
- frozen policy runner：50 Hz 产生 29 维关节 action。
- replay/dataset：收集并切分当前轮数据。
- `FDMTrainer`：更新 FDM、验证和保存 checkpoint。

## 12. 数据采集规模与内存策略

原 FDM 默认 replay trajectory 长度为 150 个 command step：

```text
150 × 0.5 s = 75 s / environment
```

G1 第一版也保留 `trajectory_length = 150`，但采用流式分片写盘，不把 4096 个环境的全部高程图长期放在 GPU。

推荐逐步扩展：

1. smoke test：16 env，2～5 个 command step。
2. 数据链路测试：64 env，完整 150 step。
3. pilot dataset：256～512 env，生成约 10 万个切片窗口。
4. 正式采集：根据 CPU RAM、磁盘吞吐和 collision rate 决定是否扩大 env 数。

仅大范围高程图一项，float16 下每帧约为：

```text
60 × 46 × 2 bytes = 5520 bytes
```

如果一次缓存 `4096 × 150` 帧，仅地图就约 3.16 GiB，还未包含历史和 Python/Tensor 开销。因此不应默认照搬 4096 env 的整轮内存 replay buffer。

## 13. 项目目录规划

根目录新建的 `fdm/` 用于保存设计、用户配置和 FDM 子项目说明；Isaac Lab task、可安装 Python 包和入口脚本仍放到仓库原有约定位置。

```text
unitree_perception_lab/
├── fdm/
│   ├── FDM_IMPLEMENTATION_DESIGN.md       # 本文档
│   ├── README.md                          # 后续补充使用说明
│   └── configs/
│       ├── rollout_g1_height.yaml
│       ├── model_g1_height.yaml
│       └── train_g1_height.yaml
│   # terrain_usd_path 指向原 FDM 的合并 USD；大文件不复制进源码包
│
├── source/unitree_rl_lab/unitree_rl_lab/
│   ├── tasks/fdm/
│   │   ├── __init__.py                    # 注册 FDM rollout task
│   │   ├── g1/
│   │   │   ├── __init__.py
│   │   │   └── fdm_rollout_env_cfg.py
│   │   └── mdp/
│   │       ├── __init__.py
│   │       ├── commands.py
│   │       ├── observations.py
│   │       ├── terminations.py
│   │       └── terrains/
│   │           ├── nav_terrain_importer.py
│   │           ├── nav_terrain_importer_cfg.py
│   │           └── ...                    # 保留已有生成器，但本任务不使用
│   │
│   └── fdm/                               # 不依赖启动 Isaac Sim 的核心包
│       ├── __init__.py
│       ├── data/
│       │   ├── schema.py
│       │   ├── rollout_buffer.py
│       │   ├── shard_writer.py
│       │   └── trajectory_dataset.py
│       ├── models/
│       │   ├── config.py
│       │   ├── encoders.py
│       │   └── g1_fdm.py
│       ├── training/
│       │   ├── losses.py
│       │   ├── metrics.py
│       │   └── trainer.py
│       ├── runner/
│       │   ├── config.py
│       │   └── online_runner.py            # collection round 在线采集—训练循环
│       └── utils/
│           ├── se2.py
│           └── timing.py
│
├── scripts/fdm/
│   ├── collect_rollouts.py                 # 独立采集/调试入口
│   ├── inspect_dataset.py
│   ├── train_fdm.py                        # 主入口：在线交替采集和训练
│   └── evaluate_fdm.py
│
├── tests/fdm/
│   ├── test_timing.py
│   ├── test_schema.py
│   ├── test_collision_delay.py
│   ├── test_window_slicing.py
│   └── test_model_shapes.py
│
└── logs/fdm/                               # git ignore，不放源码
    ├── datasets/
    ├── checkpoints/
    └── runs/
```

这样划分的原因：

- `tasks/fdm` 只负责 Isaac Lab 环境、传感器、command、termination 和地形。
- `unitree_rl_lab/fdm` 负责纯 PyTorch 数据与网络逻辑；在线 runner 在每轮仿真采集后调用这些训练组件，组件本身不依赖启动 Isaac Sim。
- `scripts/fdm` 是可执行入口。
- 根目录 `fdm` 保存方案和可版本控制的实验配置。
- 大数据、日志和 checkpoint 放 `logs/fdm`，不进入源码包和 Git。

当前 `tasks/fdm/mdp/terrains` 中已有生成器代码全部保留，以避免破坏现有导入；但 `Unitree-G1-29dof-FDM-Rollout` 只使用原 FDM 合并 USD 和移植的 `NavTerrainImporter`。

## 14. 实施阶段

### 阶段 A：接口与任务骨架

- 注册 `Unitree-G1-29dof-FDM-Rollout`。
- 验证其继承后 actor observation 仍为 283、action 仍为 29。
- 通过移植的 `NavTerrainImporter` 加载原 FDM `navigation_terrain_wall_usd_merge_large_single_object_maze.usd`，使用 10 m origin spacing。
- 加入 `60 × 46` scanner，但不替换原 scanner。
- 实现 0.5 s command 更新和三维 `[v_x, v_y, \omega_z]` 时间相关采样。
- 从指定 checkpoint 加载冻结策略并完成短 rollout。

验收：原任务配置无变化；同一 observation 下冻结策略输出与原 play/runner 一致；USD bounding box 和 environment origins 正确，G1 reset 后无初始穿透。

### 阶段 B：碰撞与多频率采集

- 实现 torso + 双侧 wrist/hand link 的即时统一碰撞标签和三组诊断标志。
- 实现 1 个 policy step 的延迟 termination。
- 实现 20 Hz 平均历史 ring buffer、2 Hz 完整帧和碰撞事件帧。
- 保存时间戳、episode id 和 termination reason。

验收：分别用 torso、左手和右手接触 USD 障碍，日志顺序都必须是“碰撞标签帧 → 下一步 done → reset”，且碰撞帧不丢失；所有配置的 body name 必须匹配成功。

### 阶段 C：数据落盘与切片

- 实现 versioned schema、shard writer 和 manifest。
- 实现 train/val/test 的 USD origin 空间分块，并把 split 固化到 manifest。
- 实现 horizon=10 的窗口切分、碰撞冻结和 valid mask。
- 提供 dataset inspection 脚本，显示频率、shape、碰撞率和随机样本轨迹。

验收：任何训练窗口都不能跨 episode；command 与下一状态的时序配对通过单元测试。

### 阶段 D：网络与训练

- 实现三个 encoder、command-level GRU、motion/collision heads。
- 实现 SE(2) residual integration 和损失。
- 先在小数据上过拟合，确认 loss、shape 和梯度正确。
- 实现 `FDMRunner` 的在线循环：固定验证集一次，之后每轮“新采集 → 切片 → 训练 8 epochs → 验证 → 保存”。
- 再运行多轮 pilot collection/training。

验收：模型可以稳定过拟合一个小 batch；所有输出无 NaN/Inf；checkpoint 可恢复；连续两个 collection round 能完成采集、训练和恢复，且 locomotion policy 参数始终不变。

### 阶段 E：评估

- 轨迹指标：ADE、FDE、yaw MAE，分别报告 0.5/1/2/3/4/5 s。
- 碰撞指标：AUROC、AUPRC、F1、recall、precision、Brier score。
- 分 USD region、速度区间和转弯强度报告。
- 与 constant-velocity baseline 对比。
- 在固定 test origin 上做可视化 rollout。

验收：FDM 相对 constant-velocity baseline 在非平地和碰撞场景有明确提升，且碰撞 recall 达到后续规划使用要求。

## 15. 必须实现的防错检查

以下条件不满足时应直接报错，而不是继续生成可能错误的数据：

- `sim.dt * decimation == 0.02`。
- `command_timestep / policy_dt == 25`。
- `history_length == 10`。
- 大范围高程图 shape 必须为 `[1, 60, 46]`。
- policy observation 必须为 283，切出的 proprioception 必须为 96。
- joint action 必须为 29，command 必须为 3，且三维速度均处于配置的采样范围内。
- 原 USD 文件存在且 hash 与 manifest 一致；origin spacing 必须为 10 m。
- `torso_link`、左右 wrist links 和左右 `rubber_hand` 必须全部被 contact sensor 匹配。
- contact history 必须足够支持 4 个 physics step 的 termination 延迟。
- 数据窗口内 `episode_id`、`usd_origin_id` 和 split 必须一致。
- 首次 collision 后 cumulative collision target 必须单调不减。
- checkpoint 中的 observation/action 维度和当前任务必须一致。

## 16. 默认设计决策汇总

| 项目 | 第一版决定 |
|---|---|
| 基础策略 | 冻结 `Unitree-G1-29dof-Velocity-perception-predict` checkpoint |
| rollout 任务 | `Unitree-G1-29dof-FDM-Rollout` |
| policy 频率 | 50 Hz |
| history 频率 | 平均 20 Hz |
| command/完整帧频率 | 2 Hz |
| command duration | 0.5 s |
| history length | 10（0.5 s） |
| prediction horizon | 10（5.0 s） |
| 策略 observation | 283 |
| FDM proprioception | `[10, 96]` |
| FDM state history | `[10, 5]`（由 `[10, 8]` raw state 转换） |
| FDM 高程图 | `[1, 60, 46]`，0.1 m |
| command | `[v_x, v_y, omega_z]`，其中 `v_y \in [-0.2, 0.2] m/s` |
| 主碰撞 link | `torso_link` + 左右 wrist roll/pitch/yaw links + 左右 `rubber_hand` |
| 碰撞诊断 | `[torso, left_hand, right_hand]`，主标签为三者 OR |
| termination 延迟 | 1 policy step = 0.02 s |
| 地形 | 直接使用原 FDM `navigation_terrain_wall_usd_merge_large_single_object_maze.usd` |
| USD origin | 移植 `NavTerrainImporter`，`usd_uniform_env_spacing = 10.0 m` |
| 训练方式 | 20 个 collection rounds；每轮在线采集新数据后训练 FDM 8 epochs |
| 模型输出 | 未来 SE(2) 轨迹 + 累计碰撞概率 |
| 第一阶段不做 | depth FDM、energy head、ensemble、planner、policy 联合训练 |

## 17. 执行前仍可调整的参数

不影响框架结构、但在正式执行前可以继续讨论的参数：

- `v_x` 和 `omega_z` 的最终范围与 command mode 比例。
- torso/hand 的最终接触力阈值，以及后续是否把 elbow/shoulder 加入主碰撞集合。
- 原 FDM USD 在当前机器上的最终部署路径，以及是否复制到独立资产目录。
- USD origin 的 train/validation/test 空间分块比例和具体区域。
- pilot 阶段 env 数、每轮窗口数和正式数据规模。
- 高程图无效点编码和第一版是否加入遮挡/噪声增强。
- collision sampler 的目标比例。

上述参数都必须配置化，不能散落为训练代码中的硬编码常量。
