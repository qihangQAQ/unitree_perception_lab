# G1 FDM 第一阶段

这一目录对应“不带 MPPI”的第一阶段：冻结已有的 G1 高程图 locomotion policy，在线 rollout，按 episode 分片保存数据，并交替训练单个 height-map FDM。

核心约束已经固化在代码和测试中：

- 策略仍使用 283 维 observation、29 维关节 action 和 recurrent hidden state。
- FDM 输入为最新优先的 `[10, 5]` state history、`[10, 96]` proprio history、`[1, 60, 46]` height map 和预先生成的 `[10, 3]` command plan。
- 模型使用参考 FDM 的 all-at-once 解码：command GRU 的 10 个输出一起展平，再联合预测 10 步速度修正和碰撞 logits。
- 碰撞事件立即写盘，termination 延迟一个 policy step；碰撞 target 后续 pose 冻结，但 command plan 保持碰撞前的原计划。
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
  --terrain-usd /home/qihang/code/fdm/exts/fdm/data/Terrains/navigation_terrain_wall_usd_merge_large_single_object_maze.usd \
  --dataset datasets/fdm_g1/baseline \
  --split train --num-envs 256 --num-episodes 256
```

执行完整的“固定 validation → 每轮采集 → 8 epoch 训练 → 验证 → checkpoint”流程：

```bash
python scripts/fdm/train_fdm.py \
  --headless --device cuda:0 \
  --checkpoint logs/rsl_rl/Unitree-Velocity_perception/2026-08-30_12-12-16_perception-predict/model_23500.pt \
  --terrain-usd /home/qihang/code/fdm/exts/fdm/data/Terrains/navigation_terrain_wall_usd_merge_large_single_object_maze.usd \
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
