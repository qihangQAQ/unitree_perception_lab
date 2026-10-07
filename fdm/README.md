# G1 FDM 第一阶段

这一目录对应“不带 MPPI”的第一阶段：冻结已有的 G1 高程图 locomotion policy，在线 rollout，按 episode 分片保存数据，并交替训练单个 height-map FDM。

训练地形随本项目保存在 `fdm/assets/terrains/navigation_terrain_wall_usd_merge_large_single_object_maze.usd`，与参考 FDM 的原始 USD 内容一致。USD 通过 Git LFS 管理；克隆或更新仓库后如果只得到指针文件，先在仓库根目录执行 `git lfs pull`。两个入口默认使用此地形，只有切换地形时才需要传 `--terrain-usd`。

核心约束已经固化在代码和测试中：

- 策略仍使用 283 维 observation、29 维关节 action 和 recurrent hidden state。
- FDM 输入为最新优先的 `[10, 5]` state history、`[10, 96]` proprio history、`[1, 60, 46]` height map 和预先生成的 `[10, 3]` command plan。
- 模型使用参考 FDM 的 all-at-once 解码：command GRU 的 10 个输出一起展平，再联合预测 10 步速度修正和碰撞 logits。
- 碰撞标签和 termination 统一检查当前 policy step 内 4 个物理子步的接触力；碰撞事件立即写盘，termination 延迟一个 policy step。碰撞 target 后续 pose 冻结，command plan 保持碰撞前的原计划。
- 位置与航向损失对每个预测步分别计算 MSE 均值，再对 10 步求和；碰撞 BCE 对有效预测步取均值。
- USD origin 先按连续空间块固定划分 train/val/test，再在 episode 内切窗口。
- 正式采集默认保留 policy observation corruption。调试时才使用 `--disable-policy-corruption`。
- FDM 的 `60 × 46` 大范围高程图按参考项目进行门洞识别：从世界高度 0.5 m 再向下、向上探测，满足 1.25 m 净空条件时用下方地面命中替换顶部横梁命中。采集器批量处理同一仿真步的记录帧，不修改冻结策略使用的局部高程图或射线传感器原始命中。数据 schema 为 v3，旧版数据须单独保存。

## 入口

只采集一个 split：

```bash
conda activate unitree-lab
python scripts/fdm/collect_rollouts.py \
  --headless --device cuda:0 \
  --checkpoint logs/rsl_rl/Unitree-Velocity_perception/2026-08-30_12-12-16_perception-predict/model_23500.pt \
  --dataset datasets/fdm_g1/baseline \
  --split train --num-envs 256 --num-episodes 256
```

执行完整的“固定 validation → 每轮采集 → 8 epoch 训练 → 验证 → checkpoint”流程：

```bash
python scripts/fdm/train_fdm.py \
  --headless --device cuda:0 \
  --checkpoint logs/rsl_rl/Unitree-Velocity_perception/2026-08-30_12-12-16_perception-predict/model_23500.pt \
  --dataset datasets/fdm_g1/baseline \
  --output logs/fdm_g1/baseline
```

检查数据和离线评估：

```bash
python scripts/fdm/inspect_dataset.py --dataset datasets/fdm_g1/baseline --split train
python scripts/fdm/evaluate_fdm.py \
  --checkpoint logs/fdm_g1/baseline/fdm_round_019.pt \
  --dataset datasets/fdm_g1/baseline --split val --device cuda
```

## 小显卡调试与进度

`train_fdm.py` 先采集固定 validation，再采集第一轮 train，最后才进行 FDM 训练。
`--num-envs` 是并行环境数，`--validation-episodes` 和 `--episodes-per-round` 是轨迹总数；
把环境从 4096 减到 1，并不会减少需要采集的 4096 条轨迹。`--batch-size` 仅影响后面的模型训练。
单条轨迹最多执行 150 个 0.5 秒 command，因此单环境采集大量轨迹可能长时间停留在 validation 阶段。

本机先用独立目录跑一个小规模完整流程：

```bash
python -u scripts/fdm/train_fdm.py \
  --headless \
  --checkpoint logs/rsl_rl/Unitree-Velocity_perception/2026-08-30_12-12-16_perception-predict/model_23500.pt \
  --dataset datasets/fdm_g1/stage1_v3_smoke \
  --output logs/fdm_g1/stage1_v3_smoke \
  --num-envs 1 \
  --validation-episodes 4 \
  --episodes-per-round 8 \
  --collection-rounds 1 \
  --epochs-per-round 1 \
  --batch-size 32 \
  --workers 0 \
  --log-interval 10
```

采集、窗口索引、训练和验证默认每 10 秒输出进度（`--log-interval 0` 关闭周期日志）。
采集日志包含完成 episode 数、仿真步数、各环境所处阶段和估计剩余时间；
即使第一条轨迹还没结束，也能通过 `steps` 判断是否推进。若某个仿真调用本身阻塞，周期日志也会停止。
数据默认累计约 20,000 帧才写一个 shard，采集结束时也会写入剩余完整轨迹，因此尚无 shard 不代表没有运行。
在线训练默认 `--workers 0` 以减少内存占用；需要多进程加载时可显式设置 `--workers 4`，使用 `spawn`，
worker 不会重新启动 Isaac Sim，也不会通过 `fork` 继承正在运行的仿真进程。
改变 rollout 参数时使用新数据目录，避免与已有 manifest 的实验元数据冲突。

## 数据帧语义

普通帧 `F_k` 是 command 决策时刻。它的 history/map 是 `t_k` 的观测，`command_plan[0]` 是紧接着执行的 `u_k`。正常情况下下一帧 `F_{k+1}` 是 `u_k` 执行 0.5 秒后的监督结果。发生碰撞时，下一帧可能是小于 0.5 秒的事件帧，实际间隔保存在 `delta_t`。

每个 shard 只保存完整 episode，并通过临时文件加原子 rename 发布。manifest 同时记录策略和 USD 的 SHA-256、空间 split、body/joint 顺序和 rollout 参数。已有 manifest 的元数据不一致时会拒绝追加，避免混入不同实验条件。

## Safe spawn

安全出生采用“地图预分析 + 落地后二次检查”：第一次 reset 时按原 FDM 的 `TerrainAnalysisRootReset` 逻辑构建 5 cm 高度图，过滤墙内点、门洞顶面和距离墙小于 0.7 m 的点，并按 G1 的 0.6 m 占地范围计算安全出生高度。候选点始终属于当前 train/val/test 空间 split。collector 随后等待机器人落地，只有双脚接触、重力方向直立、root 高度有效且 navigation link 无碰撞时才开始 warmup；失败候选不写入数据并重新采样。

## 测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/fdm
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 用于避免系统 ROS pytest 插件污染当前 Conda 环境。

## 采集统计与本地报告

两个采集入口默认输出实时进度，并在每次采集结束后按自然窗口分布打印统计表。
`train_fdm.py` 使用 `--output` 保存报告；`collect_rollouts.py` 也接受 `--output`（默认 `logs/fdm_g1/collection`）。
每次运行创建独立目录：

```text
<output>/collection/<UTC时间_随机ID>/
  progress.jsonl
  val_summary.json
  train_round_000_summary.json
```

`progress.jsonl` 在采集开始、每个日志周期和结束/失败/中断时追加记录；`--log-interval 0` 只保留这些阶段边界记录。
失败时保存异常类型与消息，堆栈在 Isaac Sim 关闭之前立即打印。
记录包含 split、round（从 0 开始）、episode 完成比例、当前调用记录的帧数、写盘 shard 数、步数、速度、ETA、阶段环境数，
以及出生检查重试、warmup 碰撞和 reset 计数。`warmup_resets` 指 reset 发生时仍处于 warmup 的次数；
已检测到的 warmup 碰撞单独计入 `warmup_collisions`。episode/s、steps/s、ETA 使用本次采集累计墙钟时间。

耗时拆为冻结策略推理、仿真、指令生成、数据处理、写盘。这是 CPU 侧程序段墙钟耗时，
不逐段强制同步 GPU，因此异步 GPU 工作可能计入后续同步所在的程序段；不能将其当作精确 CUDA kernel profiling。
接触力峰值以 N 为单位，覆盖本次接收的完整轨迹 active 阶段内的物理子步，分别记录 torso、左手和右手。
跨采集轮保留的未完成轨迹在完成时计入对应轮；它的力峰值和完整轨迹统计包含此前已采集的部分。

每份 `*_summary.json` 记录来源 manifest/shard、统计定义及以下内容：

- episode、帧、候选/有效/排除窗口数量，episode 长度和持续时间（不含 settling/warmup）。
- 碰撞 episode 比例、含碰撞窗口比例、低运动窗口比例（当前阈值 0.05 m），结束原因及碰撞部位。
- 窗口末端平面位移的均值、标准差和范围，以及距离、绝对 x/y 位移的 1 m 半开区间分布（仅列非空桶）。
- 窗口 command plan `[vx, vy, wz]` 的均值、标准差、最小值、最大值；标准差为总体标准差 `ddof=0`。
- 原始相邻帧的线速度/角速度、加速度范围；线速度坐标系为 episode 初始 yaw 坐标系。
  使用真实时间戳，角速度处理角度环绕，加速度按有符号速度差除以相邻速度区间的中点时间差。
  不跨 reset，不将碰撞后补齐帧当成物理采样。
- 理想速度基线的位置/航向误差：按 nominal command timestep 积分原 command plan，与训练窗口 target 比较；
  碰撞 target 冻结，基线继续执行计划。航向误差单位为度。
- 无效高程像素比例、各浮点字段的非有限值数量、无效时间间隔数量。非有限数据不会被统计静默删除出训练集。

数据统计基于重采样前的有效窗口；训练 sampler 的目标比例单列为 `sampler_targets`。
碰撞部位按发生碰撞的 episode 统计，各组可能重叠。空分布用 JSON `null`，单样本标准差为 0。
复用 validation 时标记 `reused=true`，不虚构本次采集耗时或力峰值。
统计不会改变数据 schema、采集分布、模型输入或速度约束。

需要参考项目的“采完即评估当前模型”流程时，在训练命令中增加：

```bash
--eval-after-collection
```

默认关闭此项以减少额外推理。开启后，验证集和每轮新训练数据均按自然顺序评估，
结果写入对应报告的 `evaluation_before_training`，附带当时的 `global_epoch`（0 代表尚未训练）。
原有训练后的验证始终保留。模型指标新增末端位置误差、理想速度基线误差及模型/基线位置 MSE 比值；
基线 MSE 为 0 时比值记为 `null`。验证损失按各项实际分母聚合，碰撞指标按全局 TP/FP/FN 聚合，避免最后一个小 batch 改变统计权重。

已有数据可以使用相同统计逻辑离线检查：

```bash
python scripts/fdm/inspect_dataset.py \
  --dataset datasets/fdm_g1/stage1_v3 --split val \
  --output logs/fdm_g1/stage1_v3_inspect
```

离线检查无法从旧版轨迹恢复运行耗时、出生重试或连续接触力峰值，这些字段标为不可用。

采集循环、训练/验证地形切换以及冻结策略的 recurrent reset 统一使用 `torch.inference_mode()`。
这是因为环境观测与 recurrent hidden state 会在推理模式下创建，后续 command 切片修改和 reset 也必须在相同模式下进行。
采集入口关闭时不等待未使用的 Replicator 任务，避免异常被关闭流程遮住。
