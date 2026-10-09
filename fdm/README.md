# G1 FDM 第一阶段

这一目录对应“不带 MPPI”的第一阶段：冻结已有的 G1 高程图 locomotion policy，在线 rollout，按 episode 分片保存数据，并交替训练单个 height-map FDM。

新数据集默认使用每环境固定容量采集：每轮重新 reset 环境和策略 hidden state，最多记录
`num_envs × frames_per_env` 个真实帧（默认每环境 150 帧，可包含多条 episode）。达到容量后进入训练；
平均填充率达到 95% 且最后 10% 进度节点后的等待超过此前平均进度间隔的 1.5 倍时，也会结束采集。
另有 policy step 上限；未达到最低填充率就触发上限会报错，不会默默训练异常数据。

轮末未完成轨迹以 `ROUND_CUT` 封口，释放原始缓冲，再准备训练样本；下轮不接着上轮的半条轨迹采集。
训练期间不调用 `env.step()`，不会同时产生 rollout 数据。已填满的环境在等待其他环境期间仍参加向量化仿真，
但不再记录数据。指令生成和 CPU 帧拷贝按同一步的环境批量执行。

参考项目每轮重置 replay buffer，并有慢尾提前结束及复制数据补齐的机制。这里仅在有效窗口采样时允许重复，
原始 shard 不拼接其他环境的轨迹。每轮先按碰撞/低运动权重抽取 **80,000 个训练窗口**，再预计算缓存，
8 个 epoch 内打乱并复用同一窗口池；固定 validation 保持全部自然分布窗口。`--samples-per-round 0`
可保留原来的全窗口加权采样训练。80,000 是本实现的最终训练池大小，参考项目的 80,000 则是后续过滤前的候选数，二者不完全等价。

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
  --dataset datasets/fdm_g1/fixed_v1 \
  --split train --num-envs 256 --frames-per-env 150
```

执行完整的“固定 validation → 每轮采集 → 8 epoch 训练 → 验证 → checkpoint”流程：

```bash
python scripts/fdm/train_fdm.py \
  --headless --device cuda:0 \
  --checkpoint logs/rsl_rl/Unitree-Velocity_perception/2026-08-30_12-12-16_perception-predict/model_23500.pt \
  --dataset datasets/fdm_g1/fixed_v1 \
  --output logs/fdm_g1/fixed_v1 \
  --frames-per-env 150 --validation-frames-per-env 150 \
  --samples-per-round 80000 --dataset-cache auto --workers 0
```

检查数据和离线评估：

```bash
python scripts/fdm/inspect_dataset.py --dataset datasets/fdm_g1/fixed_v1 --split train
python scripts/fdm/evaluate_fdm.py \
  --checkpoint logs/fdm_g1/fixed_v1/fdm_round_019.pt \
  --dataset datasets/fdm_g1/fixed_v1 --split val --device cuda
```

50 GiB CPU 内存的服务器可以先使用 `--num-envs 1024 --batch-size 2048 --dataset-cache-gb 8`，
确认采集、缓存和训练的内存日志后再增加环境数。按当前字段计算，1024 × 150 的原始帧缓冲约 1.81 GiB，
4096 × 150 约 7.26 GiB；80,000 个 float16 高程图训练样本约 0.95 GiB。
这些都不包括 Isaac Sim、固定 validation、写盘暂存、模型和 batch；50 GiB 不是所有配置下的容量保证。
`--batch-size 2048` 还取决于 GPU 显存。配置示例见 `configs/collection.yaml`；入口当前通过 CLI 配置，不自动读取 YAML。

## Checkpoint 与后续二阶段

本次从头训练，使用新的 dataset/output 目录，不传 `--resume`。入口只使用固定容量采集，
不兼容旧 episodes 机制的数据目录或 checkpoint，也没有旧轮次离线补训入口。

新 checkpoint 继续保存 `model_state_dict`、`model_cfg`、优化器、scheduler、累计 epoch、
随机数状态和数据来源，并记录格式版本。同一阶段中断后仍可用 `--resume` 恢复当前格式的完整轮 checkpoint；
`--collection-rounds`、`--epochs-per-round` 必须与原训练计划一致。
可选的 `--resume-round-summary` 仅用于当前固定容量机制下一轮已完成采集的数据，不支持旧报告。
仿真状态没有保存，因此恢复 rollout 不保证逐位复现。

未来接入 MPPI 的二阶段可以使用 checkpoint 中的 `model_cfg` 构建 FDM，加载
`model_state_dict` 作为初始化权重，并设置新的二阶段数据与训练计划。
这条模型加载路径不依赖一阶段 dataset 或一阶段 scheduler 是否已经结束；
优化器状态也仍保存在文件中，二阶段实现时可按训练方案选择是否恢复。
当前入口的 `--resume` 用于同一阶段的精确训练状态恢复，MPPI 采集与二阶段入口后续再实现。

## 小显卡调试与进度

`train_fdm.py` 先采集固定 validation，再采集第一轮 train，最后才进行 FDM 训练。
固定容量模式中，`--num-envs` 是并行环境数，`--frames-per-env` 是每环境每轮总帧数，
`--validation-frames-per-env` 是验证集的每环境帧数；`--batch-size` 仅影响后面的模型训练。

本机先用独立目录跑一个小规模完整流程：

```bash
python -u scripts/fdm/train_fdm.py \
  --headless \
  --checkpoint logs/rsl_rl/Unitree-Velocity_perception/2026-08-30_12-12-16_perception-predict/model_23500.pt \
  --dataset datasets/fdm_g1/fixed_v1_smoke \
  --output logs/fdm_g1/fixed_v1_smoke \
  --num-envs 4 --frames-per-env 24 --validation-frames-per-env 24 \
  --samples-per-round 256 \
  --collection-rounds 2 \
  --epochs-per-round 1 \
  --batch-size 32 \
  --workers 0 \
  --log-interval 10
```

采集、窗口索引、训练和验证默认每 10 秒输出进度（`--log-interval 0` 关闭周期日志）。
采集日志包含实际帧/目标帧、完成 episode 数、仿真步数、各环境所处阶段和估计剩余时间；
即使第一条轨迹还没结束，也能通过 `steps` 判断是否推进。若某个仿真调用本身阻塞，周期日志也会停止。
数据默认累计约 20,000 帧才写一个 shard，采集结束时也会写入剩余已封口轨迹，因此尚无 shard 不代表没有运行。
在线训练默认 `--workers 0` 以减少内存占用；需要多进程加载时可显式设置 `--workers 4`，使用 `spawn`，
worker 不会重新启动 Isaac Sim，也不会通过 `fork` 继承正在运行的仿真进程。
改变 rollout 参数时使用新数据目录，避免与已有 manifest 的实验元数据冲突。

训练进度会输出实际 `device`，以及 `last_s(data=...,step=...)`（最近一批）和
`avg_s(data=...,step=...)`（本 epoch 每批平均耗时）。`data` 包含等待 DataLoader 和拼接 batch，
分片模式下还包含读取和构造样本；`step` 包含传入设备、前向、损失、反向和参数更新。
CUDA 损失读取完成后才结束 step 计时，避免异步 GPU 计算被误计入下一批数据等待。
多 worker 时 data 是主进程实际等待时间，不是各 worker 的工作时间之和。
`shard_loads` 是当前 epoch 累计完整分片加载次数（仅 workers=0 可直接统计）。

在线训练与测速默认 `--dataset-cache auto`：内存足够时使用 `memory`，不足时使用预计算的 `mmap` 文件。
索引/统计完成并选定训练窗口池后，顺序扫描相关分片一次，预先计算相对状态、未来轨迹、碰撞标签和掩码。
固定 validation 缓存只准备一次；每轮仅缓存新采集的 train 数据，连续用于该轮全部 epoch。
保存轮次 checkpoint 后释放 train 缓存，再采集下一轮；不会累积缓存所有轮次。
训练/验证时索引预计算张量，不再反复反序列化分片或变换窗口，训练日志应显示 `cache=memory` 或 `cache=mmap`、`shard_loads=0`。

样本缓存直接预分配为最终大小，预处理临时数据限制为一个原始分片和小批窗口。
常规采集的高程图保留 float16，取 batch 时才转 float32；已有 v3 文件若包含 float32 高程图，则保留其精度。
使用受限样本池时仍按原来的碰撞/低运动目标权重抽样，只抽一次，然后每个 epoch 打乱该池；
报告单列源窗口数、池大小、独立窗口数和重复次数。原始窗口标签不变。

缓存前打印当前 split、已常驻缓存及二者合计的预计 GiB。可用 `--dataset-cache-gb 8` 将
**固定 validation + 当前 train 的常驻 RAM 样本张量**限制在 8 GiB；不设置时没有显式缓存容量上限。
预算不包括 mmap 的动态驻留页、仿真、模型、Python 窗口索引、原始分片和临时 batch，不能作为整个进程的 RAM 上限。
此外会尽可能读取系统可用 RAM 和常见容器内存限制，在分配前检查预处理空间并预留 20% 可用 RAM；
这只是预检查，不能保证其他进程同时分配内存时不发生竞争。强制 memory 模式预算不足会报错；auto 会尝试 mmap。
默认继续使用 `--workers 0`；多进程模式仍用 spawn，大缓存跨进程传递还需要足够的系统共享内存。

容器可用内存估算同时读取 `memory.stat`，计入保守的可回收 inactive file 缓存，扣除共享内存和
dirty/writeback 限制，并受宿主机 `MemAvailable` 与容器限额约束；不再只用 `limit-current`。
每次索引/缓存前后及每轮释放后，`Memory stage=...` 显示进程 RSS、宿主机可用量、容器 anon/file 占用、
可回收估计和未完成采集轨迹的帧数，也写入 `progress.jsonl`。这些值是采样时刻的估计，不是分配保证。

`--dataset-cache auto` 优先使用与 memory 模式相同的预计算 CPU 张量。若预检查认为容量不足，
当前数据集改用 mmap，保留已准备的 validation 缓存。若连一个原始 shard 的预处理空间都不足则明确报错。
`mmap` 将结果原子发布到 dataset 下的 `.fdm_sample_cache/<hash>/`，key 包含源文件大小/修改时间、
窗口索引与变换版本；相同数据和窗口池可以直接复用。spawn worker 自行打开映射，不复制整个缓存。
它仍会占用磁盘和操作系统页缓存，内存压力大时吞吐受磁盘性能影响，但不会退回每批重读原始 shard 的路径。
这些文件可在没有训练使用时删除；原始 shard 足以重新构建。不同轮次的 mmap 文件会留在磁盘，不自动清理。

`--dataset-cache shard` 仅保留作旧机制对照。此模式保留两个分片的 LRU 缓存，
同一批内按分片/episode 集中读取，再恢复原始采样顺序和重复样本，每个分片每批至多加载一次。
它仍有跨 batch 的重复读盘，吞吐通常低于内存模式。`--dataset-cache-gb` 不限制此模式的原始分片缓存。

如果训练异常缓慢，可直接在已有 train 数据上测量几批完整前向/反向：

```bash
python -u scripts/fdm/benchmark_fdm.py \
  --dataset datasets/fdm_g1/fixed_v1 \
  --batch-size 2048 --batches 3 --epochs 2 --device cuda:0 \
  --samples-per-round 80000 --dataset-cache auto --dataset-cache-gb 8
```

测速默认跑 2 个 epoch、每个 3 批，检查缓存能否跨 epoch 复用。指定报告可只测当前轮；
省略 `--round-summary` 才会索引全部 train 分片。窗口池默认 80,000，准备缓存后才取测试 batch。
使用临时随机初始化模型，不启动 Isaac Sim，不写原始数据或 checkpoint；mmap 模式会写预计算文件缓存。
因此它用于定位瓶颈，不能恢复当前训练权重；独立测速也不包含在线仿真常驻时的资源竞争。
首先比较 `data` 与 `step`，第一批可能包含 GPU 初始化开销，再观察后续批次。

## 数据帧语义

普通帧 `F_k` 是 command 决策时刻。它的 history/map 是 `t_k` 的观测，`command_plan[0]` 是紧接着执行的 `u_k`。正常情况下下一帧 `F_{k+1}` 是 `u_k` 执行 0.5 秒后的监督结果。发生碰撞时，下一帧可能是小于 0.5 秒的事件帧，实际间隔保存在 `delta_t`。

每个 shard 保存完整 episode 或已封口的真实 episode 片段，并通过临时文件加原子 rename 发布。
`ROUND_CUT` 与碰撞分开编码，不能作为完整未来或碰撞补齐标签；窗口不会跨 episode/reset/轮次。
manifest 同时记录策略和 USD 的 SHA-256、空间 split、body/joint 顺序、rollout 参数及固定采集配置。
已有 manifest 的元数据不一致时会拒绝追加，避免混入不同实验条件。

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
记录包含 split、round（从 0 开始）、填充比例、当前调用记录的帧数、写盘 shard 数、步数、速度、ETA、阶段环境数，
以及出生检查重试、warmup 碰撞和 reset 计数。`warmup_resets` 指 reset 发生时仍处于 warmup 的次数；
已检测到的 warmup 碰撞单独计入 `warmup_collisions`。episode/s、steps/s、ETA 使用本次采集累计墙钟时间。
固定容量模式另有 `target_frames`、`collection_mode=fixed` 和 `stop_reason`（capacity/slow_tail/step_limit），
完成比例与 ETA 按帧容量计算。

耗时拆为冻结策略推理、仿真、指令生成、数据处理、写盘。这是 CPU 侧程序段墙钟耗时，
不逐段强制同步 GPU，因此异步 GPU 工作可能计入后续同步所在的程序段；不能将其当作精确 CUDA kernel profiling。
接触力峰值以 N 为单位，覆盖本次接收的完整轨迹 active 阶段内的物理子步，分别记录 torso、左手和右手。
每轮仅统计本轮真实轨迹和封口片段。

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
