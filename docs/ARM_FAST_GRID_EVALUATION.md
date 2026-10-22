# ARM 平台前端优化评估：FAST-Grid 与 KLT 金字塔复用

本文记录参考 `open_vins`、`SchurVINS` 后，在 VINS-Fusion 上实施的前端优化，
以及在 Orange Pi Zero 3（4×Cortex-A53）上使用 `~/vicon_room2/bags` 三条
EuRoC V2 完整序列做的同二进制 A/B 验证。

> 结论边界：**降载与降延迟成立；精度没有一致提升。** FAST-Grid 仍是实验性
> opt-in 路径，默认检测器仍为 GFTT。

---

## 1. 参考项目的可借鉴点与取舍

### 1.1 OpenVINS（`ov_core/src/track/TrackKLT.cpp`、`Grider_GRID`）

- 前端用 **FAST + 网格（grid）检测**，而非全图 `goodFeaturesToTrack`。
  每个网格单元扫描 FAST 角点，再按响应选点，能显著减少全图排序/亚像素细化开销。
- KLT 跟踪前后都裁剪/复用**金字塔**，避免重复构建。
- 后端是 MSCKF + FEJ（first-estimate Jacobian），与 VINS 的滑窗 BA 是不同体系。

**采用的**：FAST + grid 检测器、KLT 金字塔缓存、`min_dist` 约束。
**未采用的**：MSCKF/FEJ 后端。原因见 §1.3。

### 1.2 SchurVINS（`svo/src/schur_vins.cpp` 等）

- 使用 **EKF + Schur complement** 的滤波式后端，状态规模随时间不增长，
  适合算力受限平台（ARM）。
- 前端基于 SVO 半直接法，与 VINS 的稀疏 KLT 前端差异很大。

**关键事实**：VINS-Fusion 的 Ceres 后端**已经使用 `DENSE_SCHUR`**
（`estimator.cpp`），所以"补一个 Schur complement"并不是缺失项。真正的差异是
**滑窗 BA/边缘化** vs **EKF 滤波**，属于后端重写，不能作为本轮低风险改动。

### 1.3 为什么本轮不做后端重写

| 方向 | 潜在收益 | 风险 / 成本 |
|---|---|---|
| 移植 OpenVINS MSCKF+FEJ | 状态不增长，ARM 上后端开销更可控 | 需重写初始化、边缘化、外参/时延、失败检测；回归面极大 |
| 移植 SchurVINS EKF 后端 | 计算量随状态线性、适合 ARM | 与现有因子图/边缘化不兼容，精度行为需重新标定 |
| 保留现有 BA，仅做前端/工程优化 | 低风险、可增量验证 | 收益上限低于后端重写 |

因此本轮聚焦**前端检测、KLT 金字塔复用、参数化与可测试性**。

---

## 2. 已实现的改动

### 2.1 新增参数（`parameters.cpp/.h`、EuRoC 配置）

```yaml
feature_detector: "gftt"          # gftt | fast_grid
fast_threshold: 20
grid_x: 5
grid_y: 5
detection_min_feature_ratio: 0.0  # 已有特征数低于该比例才补点
```

参数读取带范围校验；非法值会直接抛异常，避免静默配置错误。

### 2.2 KLT 金字塔缓存（`feature_tracker.cpp`）

- 用 `buildOpticalFlowPyramid` + `calcOpticalFlowPyrLK(pyr)` 替代逐次重建：
  temporal forward / backward、prediction、stereo 复用同一套金字塔。
- GFTT 路径只构建 **image-only** 金字塔（无导数），避免
  `goodFeaturesToTrack` 在 ARM 上分配 752×480 `CV_32F` 临时量导致的回退
  （代码注释记录 ~1.75 ms → ~4.45 ms 的劣化）。
- FAST-Grid 路径构建**带导数**金字塔供后续 KLT 直接使用。

### 2.3 FAST-Grid 检测器（opt-in）

- 5×5 网格，每格独立 `cv::FAST`。
- 每格按响应 `partial_sort` 保留候选，避免全图排序。
- **轮转选点**：每轮每格至多取一个点，先铺满全图，再取第二点。
- 选点后统一 `cornerSubPix`（5×5），再次校验 mask 与 `min_dist`。
- mask 语义沿用 VINS（255=可用），与 OpenVINS 的 Grider mask 相反，已处理。

### 2.4 测试

- `test_feature_tracker.cpp`：参数/检测器/网格/门限行为（8 个用例）。
- `test_gftt_pyramid_perf.cpp`：GFTT 金字塔性能回归（1 个用例）。
- `CMakeLists.txt` 注册新测试。
- 本机复验：`ctest` 5/5 可执行文件通过（`test_feature_tracker` 8/8、
  `test_gftt_pyramid_perf` 1/1）；ARM 上亦完成同一套构建与回归。

---

## 3. ARM 实测方法

### 3.1 环境

- 板卡：Orange Pi Zero 3，4×Cortex-A53 @ ~1.512 GHz，3.8 GiB。
- 主机：`pinydrone0@192.168.5.60`（远端结果根 `~/vins_remote_eval_20261019`）。
- 数据：`~/vicon_room2/bags/{V2_01_easy,V2_02_medium,V2_03_difficult}`。
- 回放：**完整序列、1 倍速**，`max_cnt=120`，`flow_back=1`，`max_solver_time=0.04`。

### 3.2 公平性控制

- 三个二进制均在 ARM 上**本机构建**，A/B 使用同一脚本、同一配置路径、同一回放器。
- 旧基线 `current_head_vins_node`（源码 `8c5e7d6`，SHA-256
  `7627574c…8697b`）与实际候选分别测。
- 候选 `gfttfix_fastgrid_vins_node`（SHA-256 `78f67064…832f`）同时支持
  GFTT/FAST-Grid，因此：
  - **GFTT 对照**：同一候选二进制，仅切换检测器 → 隔离检测器影响。
  - **旧→新**：反映"旧基线 → 优化后二进制"的端到端总量。

### 3.3 指标

- `cpu_seconds`：进程 CPU 时间（含多线程累计）。
- 延迟分位：图像发布到 VIO 输出的时间差。
- ATE：刚性位置对齐 RMSE；另做**共同时间窗口**重算，避免覆盖长度不同。

---

## 4. 实测结果

### 4.1 旧基线 GFTT（`baseline_fair`，1 次）

| 序列 | CPU s | P50 ms | P95 ms | P99 ms | ATE m | poses |
|---|---:|---:|---:|---:|---:|---:|
| V2_01_easy | 226.73 | 423.9 | 856.0 | 1025.1 | 0.20117 | 1078 |
| V2_02_medium | 219.23 | 406.8 | 605.0 | 1265.0 | 0.27057 | 1039 |
| V2_03_difficult | 200.17 | 384.7 | 588.7 | 1197.3 | 0.49805 | 1077 |

### 4.2 新二进制 GFTT（2 次完整运行，表中为两次）

| 序列 | CPU s (r1/r2) | P95 ms (r1/r2) | ATE m (r1/r2) | poses |
|---|---|---|---:|---:|
| V2_01_easy | 224.91 / 224.39 | 597.19 / 606.60 | 0.19179 / 0.16622 | 1083 / 1084 |
| V2_02_medium | 219.66 / 221.18 | 436.09 / 526.86 | 0.29981 / 0.26047 | 1136 / 1094 |
| V2_03_difficult | 198.36 / 197.78 | 540.73 / 520.51 | 0.48311 / 0.48556 | 1089 / 1087 |

- CPU 与旧基线基本持平（-0.9% ~ +0.2%）。
- **P95 明显下降**：easy -30.2%、medium -27.9%、difficult -8.2%（r1 对照）。
- ATE 两次波动：easy 14.3%、medium 14.0%、difficult 0.5%（极差/均值）。

### 4.3 新二进制 FAST-Grid（easy/difficult 2 次，medium 4 次）

| 序列 | CPU s 均值 | P95 ms 均值 | ATE m 均值 | ATE 极差% | poses |
|---|---:|---:|---:|---:|---:|
| V2_01_easy | 183.41 | 237.37 | 0.21239 | 0.15% | 1083 |
| V2_02_medium | 176.72 | 158.12 | 0.26386 | 2.18% | 1137 |
| V2_03_difficult | 134.03 | 123.60 | 0.52521 | 1.07% | 1091 |

- 运行间 **CPU 极差 <0.3 s、P95 波动 <8 ms**，重复性很好。
- ATE 重复性明显优于 GFTT（≤2.2% vs 最高 14%）。
- medium 4 次全部正常，无 `numerical unstable`。

### 4.4 关键对比（第一次运行，共同时间窗口 ATE）

| 对比 | 序列 | CPU | P95 | ATE（共同窗口） |
|---|---|---:|---:|---:|
| 旧 GFTT → 新 GFTT | easy | -0.8% | -30.2% | -4.7%（0.2012→0.1917） |
| | medium | +0.2% | -27.9% | +10.8%（0.2706→0.2999） |
| | difficult | -0.9% | -8.2% | -3.1%（0.4980→0.4826） |
| 旧 GFTT → 新 FAST-Grid | easy | **-19.2%** | **-73.3%** | +5.5%（0.2012→0.2122） |
| | medium | **-19.4%** | **-74.8%** | -4.9%（0.2706→0.2573） |
| | difficult | **-32.9%** | **-79.1%** | +6.1%（0.4976→0.5280） |
| 新 GFTT → 新 FAST-Grid | easy | **-18.6%** | **-61.7%** | +10.7%（0.1918→0.2122） |
| | medium | **-19.6%** | **-65.0%** | -13.4%（0.2998→0.2597） |
| | difficult | **-32.3%** | **-77.3%** | +9.5%（0.4821→0.5280） |

峰值 RSS 在 FAST-Grid 下下降约 5%–7%。

---

## 5. 精度结论

1. **降载/降延迟是确定收益。** 在同一候选二进制的检测器隔离对比中，FAST-Grid
   相对 GFTT 降低 CPU 18.6%–32.3%、P95 61.7%–77.3%，且运行间方差很小。
2. **精度没有稳定提升。** 共同窗口 ATE 在 -13.4% ~ +10.7% 之间摆动：
   medium 变好、easy/difficult 变差。方向随序列变化，**不能声称 FAST-Grid 提高精度**。
3. **旧→新 GFTT 的 P95 下降不等于 CPU 下降。** CPU 与旧基线持平，
   说明金字塔复用主要改善**调度/尾延迟**，而不是减少总计算量。
4. **difficult 的绝对值最差**（~0.52 m），三种方案均如此，属于该序列本身难度，
   不是本轮引入的退化。

---

## 6. 必须记录的风险

### 6.1 Ceres 截断解进入边缘化

新二进制日志每约 5 s 出现：

```
Ceres hit max_solver_time=32/40 ms ... the truncated solution is still used
```

三条完整序列各约 **23 次**。旧二进制没有该日志，是因为**缺少该告警文本**，
不能推断旧版没有超时。风险在于：截断解被直接 `double2vector()` 并参与
`marginalization`，可能污染先验。这是后续最值得处理的后端问题（见 §7）。

### 6.2 失败检测未实现

`failureDetection()` 仍直接 `return false`，因此 `failure_reboots=0`
**不能证明未发散**，只能说明没有触发重启路径。

### 6.3 历史 medium 数值不稳未根治

旧基线出现过一次约 108 s 后 `numerical unstable in preintegration` 与
`1e21` 量级坐标。本次新二进制 medium 共 4 次完整运行均正常，但样本量不足以
宣称已修复。

### 6.4 配置混杂

旧→新对比同时包含 KLT cache、detection gate、检测器、构建条件变化；
只有 **新 GFTT ↔ 新 FAST-Grid** 是干净的检测器隔离 A/B。

---

## 7. 后续优化优先级

> 后续 P0/P1/P2 项的完成情况与 ARM 验证结果见
> `docs/ARM_P0_P1_OPTIMIZATION.md`。

| 优先级 | 方向 | 说明 |
|---|---|---|
| P0 | **Ceres 超时事务化** | 超时后拒绝截断解、保留上一状态或跳过该次更新，避免污染边缘化先验 |
| P1 | 可配置边缘化线程数 | `NUM_THREADS=4` 在 4×A53 上未必最优，应做成参数并 A/B |
| P1 | 删除投影 factor 中的 `TicToc` | 每帧每特征计时开销，低风险 |
| P2 | FAST-Grid 跳格/候选裁剪 | 在保持覆盖的前提下进一步降载 |
| P2 | 阶段 timing | 分离前端/后端/边缘化耗时，指导后续投入 |
| P3 | opt-in RANSAC 外点剔除 | 提升 difficult 精度，但增加开销 |
| — | 后端 MSCKF/EKF 重写 | 收益最大但风险最高，需独立项目周期 |

**不建议**：默认 `-march=native`、`fast-math`、继续降低 `max_cnt`、关闭
`flow_back`。这些以精度或可移植性换取速度，与本次"可验证的工程优化"目标冲突。

---

## 8. 复现

```bash
# 远端配置必须用相对远端 home 的路径（不要用 ~/...，本地 shell 会展开）
python3 scripts/benchmark_remote_arm.sh \
  --host pinydrone0@192.168.5.60 \
  --config Code/Piny_drone_nav/src/VINS-Fusion/src/vins_estimator/config/euroc/euroc_mono_imu_fast_grid_config.yaml \
  --label arm_fastgrid \
  --duration full
```

本地汇总：

```bash
python3 output/opt_20261020/arm_analysis/make_final_comparison.py
```

## 9. 数据位置

| 内容 | 路径 |
|---|---|
| 旧基线 | `output/opt_20261020/arm_analysis/baseline_fair/` |
| 新 GFTT r1/r2 | `final_gftt_control/`、`final_gftt_repeat/` |
| 新 FAST-Grid r1/r2 | `final_fastgrid/`、`final_fastgrid_repeat/` |
| FAST-Grid medium 稳定性 | `final_fastgrid_stability/` |
| 汇总 JSON | `output/opt_20261020/arm_analysis/final_comparison.json` |
| Groundtruth | `output/opt_20261020/arm_analysis/groundtruth/` |
| FAST-Grid 配置 | `src/vins_estimator/config/euroc/euroc_mono_imu_fast_grid_config.yaml` |
