# G1 FDM 二阶段：MPPI 混合 rollout

二阶段仍然冻结 G1 locomotion policy，只更新 FDM。每轮先用上一轮 FDM 的固定推理副本采集，
再暂停仿真、训练、验证；下轮换成更新后的副本。入口为 `scripts/fdm/train_fdm_stage2.py`。
一阶段入口及模型参数名称保持兼容，初始 FDM 只加载权重，不继承一阶段的 optimizer、scheduler 或 epoch。

## 单张 4090 / CPU 内存 50 GiB 的启动配置

在仓库根目录运行，使用新的数据和输出目录：

```bash
conda activate unitree-lab
python scripts/fdm/train_fdm_stage2.py \
  --headless --device cuda:0 \
  --checkpoint logs/rsl_rl/Unitree-Velocity_perception/2026-08-30_12-12-16_perception-predict/model_23500.pt \
  --fdm-checkpoint logs/fdm_g1/stage1_fixed_v1/fdm_round_018.pt \
  --dataset datasets/fdm_g1/stage2_mppi_v1 \
  --output logs/fdm_g1/stage2_mppi_v1 \
  --num-envs 256 --collection-batches 4 \
  --frames-per-env 150 --validation-frames-per-env 150 \
  --mppi-candidates 128 --mppi-iterations 1 --inference-chunk 2048 \
  --initial-mppi-fraction 0.25 --initial-mix-rounds 2 --mppi-fraction 0.40 \
  --collection-rounds 10 --epochs-per-round 4 --learning-rate 1e-4 \
  --samples-per-round 80000 --batch-size 2048 --workers 0 \
  --dataset-cache auto --dataset-cache-gb 8 --navigation-eval-interval 5
```

`--checkpoint` 是冻结的运动策略，`--fdm-checkpoint` 是用于初始化的 FDM，两者不能互换。
默认保留策略观测噪声。训练的轮次从 0 重新计数，最终 checkpoint 为 `fdm_round_009.pt`。

每轮最多记录 `256 × 150 × 4 = 153600` 个真实帧。4 个子批次各自 reset、封口、释放缓冲，
不会跨子批次或跨训练轮保留半条 episode。原始缓冲每次约 0.454 GiB；这不包括仿真、写盘暂存、
验证缓存、训练样本与 GPU 张量。`--dataset-cache-gb 8` 约束 RAM 样本缓存，不能当作总进程内存上限。
如需减少显存，可先降低 `--inference-chunk`、`--batch-size`；若将环境数降到 128，
使用 `--collection-batches 8` 可保留每轮最大真实帧量。

首次部署可使用独立目录做两轮小测试：

```bash
python scripts/fdm/train_fdm_stage2.py \
  --headless --device cuda:0 \
  --checkpoint logs/rsl_rl/Unitree-Velocity_perception/2026-08-30_12-12-16_perception-predict/model_23500.pt \
  --fdm-checkpoint logs/fdm_g1/stage1_fixed_v1/fdm_round_018.pt \
  --dataset datasets/fdm_g1/stage2_smoke_server \
  --output logs/fdm_g1/stage2_smoke_server \
  --num-envs 4 --collection-batches 2 \
  --frames-per-env 24 --validation-frames-per-env 24 \
  --collection-rounds 2 --epochs-per-round 1 \
  --mppi-candidates 8 --inference-chunk 16 \
  --samples-per-round 64 --batch-size 16 --workers 0 \
  --dataset-cache auto --dataset-cache-gb 1 --navigation-eval-interval 1
```

小测试仅检验流程，不用于评价导航能力或估算正式配置的收敛效果。

## 采集与规划

- FDM 命令周期为 0.5 秒，预测 10 步、共 5 秒。MPPI 每个命令周期重规划，只执行首条命令。
  0.5 秒是仿真时间，实际墙钟耗时由规划和仿真速度决定。
- 前两轮 25% 环境使用 MPPI，之后为 40%；其余环境使用一阶段的相关随机指令。
  分配在一个子批次内固定，包括 reset 后；下一批重新分配。
- MPPI 环境先用零指令积累完整历史，再进入规划。已有计划移位作 warm start，加入时间相关噪声，
  同时包含停止与朝向目标的候选。指令范围为 `vx ∈ [0,1.2]`、`vy ∈ [-0.2,0.2]`、`wz ∈ [-1,1]`。
- 成本包含目标距离、连续碰撞概率、高风险惩罚、指令变化和地图无效/越界惩罚。
  FDM 的碰撞概率按累计事件概率处理，不把它当作独立 hazard 重复相乘。
- 加权均值计划会再次经过 FDM 评分；更差、非有限或不安全的均值退回最佳候选，全部候选无效则停止。
  预测碰撞概率只影响规划，真实碰撞标签来自接触力传感器。
- 每次决策只计算一次 history/map 编码，候选命令分块解码。
  规划和写盘复用同一份时间对齐、float16 量化后的门洞修正地图；无效像素查询考虑 torso 与 root 的坐标差异。
- 复用 safe-spawn 的地形射线结果，以 0.2 m 栅格和 0.4 m 净空构建可走区域连通分量。
  目标距当前位置 1.5–3 m，位于同一空间 split 和连通分量。目标世界坐标保持不变，直到到达 0.4 m 范围、
  超时 20 秒或连续 5 秒无足够进展后重新选择。它是局部目标采集器，不包含全局路径搜索。
- 如果持续 5 秒无法选择目标，以 `GOAL_UNAVAILABLE` 截断并换出生点。
  实际机器人位置越过当前 split 时，以 `SPLIT_EXIT` 截断并 reset；这两种情况都不标为碰撞。

## 二阶段标签与样本池

数据使用独立 **v4** schema。一阶段 v3 数据和读取方式仍保留；两种数据不能追加到同一个 manifest。
新字段记录 `command_source`（0=random / 1=MPPI）、目标世界坐标和 `planner_model_version`。
模型版本 0 对应初始化权重，版本 1 对应完成二阶段第一轮后的权重，以此类推。

无碰撞窗口使用每一步**实际下发的 command** 拼成模型输入，不能直接使用窗口起点的旧 `command_plan`。
例如第 2 步重规划改变了速度，第 2 步监督也必须使用改变后的速度。

若第 `n` 条命令执行时发生碰撞：输入的前 `n` 条使用实际执行命令；余下部分从碰撞前最后一次决策的
`command_plan[1:]` 补足。碰撞后的目标 pose 冻结、collision 累计标签保持 1。
不会取事件/reset 帧的命令，也不会把未执行尾部统一补成零。没有碰撞而未来不足的窗口排除，不跨 reset 或 ROUND_CUT 拼接。
样本缓存版本已更新，旧的预计算目标缓存不会被误用。

每轮仅使用本轮各子批次的新 shard。先按随机/MPPI 配额抽取最多 80000 个窗口，然后缓存，训练中打乱复用。
碰撞与低运动的目标占比仍为 35% / 10%，但每个原始窗口在样本池内最多重复 4 次。
某一来源的碰撞窗口太少时，优先使用该来源的其他窗口；整个来源不足时保留较小配额并报告 shortfall，
不会靠无限复制凑出看似满足要求的比例，也不会复制或拼接原始 episode。

## 验证与闭环对照

新训练启动时，用初始化的 FDM 固定生成两套 validation：纯随机命令和纯 MPPI 命令。
它们均在 val 地形区域采集，训练全过程保持不变；无需下载一阶段原始 dataset。
`validation.json` 保存初始指标，各轮 checkpoint JSON 分别保存两套验证指标。
指标包含位置/航向误差、位置 P95、各预测步误差、碰撞 precision/recall 和阈值曲线。
`by_horizon` 中首碰撞预测步的 recall 是按命令步分桶，`nominal_time_s` 不是物理子步精度的碰撞提前量。

默认还在训练前、每 5 轮及最后一轮做单目标闭环对照：初始化 FDM+MPPI 对比当前 FDM+同参数 MPPI。
所有环境先完成相同的准备阶段再一起启动规划，固定 reset/goal/MPPI 种子，并保存起点和目标。
报告检查 `matched_scenarios`；不匹配时不会输出能力提升的差值。
每个有效目标记录到达、碰撞、超时、停滞、越界等结果和耗时。到达率分母为成功分配的目标数，
同时应检查 `goal_unavailable`，不能忽略目标分配失败的环境。

正式训练后可在独立 test 地形区域评估：

```bash
python scripts/fdm/evaluate_mppi.py \
  --headless --device cuda:0 \
  --checkpoint logs/rsl_rl/Unitree-Velocity_perception/2026-08-30_12-12-16_perception-predict/model_23500.pt \
  --baseline logs/fdm_g1/stage1_fixed_v1/fdm_round_018.pt \
  --fdm-checkpoint logs/fdm_g1/stage2_mppi_v1/fdm_round_009.pt \
  --num-envs 64 --mppi-candidates 128 --inference-chunk 2048 \
  --split test --seed 90042 \
  --output logs/fdm_g1/stage2_mppi_v1/navigation_test_90042.json
```

使用多个 seed 和足够多有效目标来评价导航效果，不能仅根据训练 loss 或少量 smoke trial 判断二阶段优于一阶段。

## 文件与续训

```text
datasets/fdm_g1/stage2_mppi_v1/
  train/manifest.json              # v4 训练数据，每轮各自选择新 shards
  validation_random/manifest.json  # 固定随机命令验证数据
  validation_mppi/manifest.json    # 固定 MPPI 验证数据
  validation.json                 # 初始化模型在两套数据上的基线指标
  navigation_baseline.json         # 初始闭环评估，启用时生成
logs/fdm_g1/stage2_mppi_v1/
  fdm_round_000.pt / .json
  ...
  navigation_round_004.json
  navigation_round_009.json
  collection/<run_id>/progress.jsonl
  collection/<run_id>/train_round_000_summary.json
```

日志包含每批实际帧数、内存、MPPI 推理批次均值/P95 耗时、均值回退和全部候选无效次数、
目标结果，以及两类来源的自然窗口数、采样数、唯一数、碰撞占比和配额不足量。
`goal_unavailable` 统计失败尝试，`anchor_unavailable` 表示当前位置不在可走栅格，
`goal_unavailable_resets` 是由持续目标失败触发的 reset 数。

需要中断续训时，使用同样的二阶段采集/规划/训练计划参数，将原命令中的
`--fdm-checkpoint ...` 换成 `--resume logs/fdm_g1/stage2_mppi_v1/fdm_round_004.pt`。
保留原 `--dataset`、固定验证数据和总轮数 10。恢复下一轮、optimizer、scheduler 与当前格式 RNG 状态。
未完成轮的孤立 shards 保留在磁盘上，恢复后重新采集该轮；不会混入新轮训练池。
这只是二阶段自身的恢复机制，没有新增旧一阶段 episode 机制的 checkpoint 兼容。

## 本地验证

- FDM 回归测试 92 项通过，包含 v3 保持、v4 指令对齐、碰撞尾部、分块推理一致性、均值重评分、
  无效候选停止、目标连通性、越界/无目标截断、固定缓冲释放、训练入口与二阶段自身恢复。
- 本地 8 GiB RTX 4060 Ti 已跑通 4 环境、两子批次、两轮的真实 Isaac Sim 采集/训练/双验证，
  并完成训练前和两轮后的闭环对照，起点/目标匹配检查通过。
- 独立评估入口另以 4 环境、128 候选完成 GPU 验证；上述结果不代表已经测得 4090、256 环境的正式吞吐或峰值显存。
