# ARM 平衡优化最终实施结果（2026-11-10）
> 最终状态：本地产品代码已冻结；本地 frozen 与 ARM `.62` default 三序列均已完成，ARM profile 由另一代理测试。本文以实测表、证据边界和使用命令为主。所有回放均为 **单次筛选，不是重复统计**，不得宣称统计显著。CPU governor 未修改。

## 1. 最终结论摘要
- 本地 frozen 主表（baseline → `optimized_frozen`）：三条 EuRoC V2 序列的 8 列轨迹（含 `t/q`）逐位相同，共同时间戳 ATE 完全相同。
- CPU：`-2.725% / -3.388% / -1.028%`；P95：`-9.677% / -6.974% / -1.288%`（easy / medium / difficult）。
- Peak RSS：`+0.1016 / +0.0898 / +0.0547 MiB`；没有 RSS 下降。
- 分配探针：malloc/calloc/realloc calls `477079 → 314192`（`-34.1%`），累计请求字节 `141576788 → 123944316`（`-12.45%`）；这是进程总请求量，不是 peak/live RSS。
- 微基准 3 轮：reduce `4.58 → 4.49 us`（小幅）；update `265.7 → 269.2 us`（无稳定收益）。
- 初次 optimized 的 difficult CPU 为 `+2.7%`，frozen 为 `-1.03%`；该差异用于说明单次运行波动，不作为加速证据。
- ARM `.62` default：CPU `-3.72% / -2.543% / -3.283%`，P95 `-1.557% / -2.251% / -2.091%`，Peak RSS `-0.0508 / -0.2617 / +0.4141 MiB`；三条 8 列轨迹 `np.array_equal=True`，ATE 相同，restart requests/reset entries 为 `0/0/3`（两版本相同）。
- ARM `.62` default 的 difficult 仍有 `ATE=19.615 m`、`max_output_gap=8.2 s`；这不是可部署结果。
- `pin67 + threads1` difficult 首轮：CPU `75.98 s`、P50/P95/P99 `36.419/55.656/68.088 ms`、RSS `100.047 MiB`、ATE `19.5772 m`、gap `8.2 s`、poses `994`。相对 default opt（`109.90 s`、P95 `88.703 ms`、ATE `19.6152 m`）性能明显提高（CPU 约 `-30.9%`、P95 约 `-37.3%`），但精度仍然失败。
- `pin67 + threads1` medium：CPU `83.06 s` / P95 `39.689 ms` / ATE `0.275096`，相对 opt_default（`122.25 s` / `79.673 ms`）约 CPU `-32.1%`、P95 `-50.2%`，精度不变。
- default easy 重复：CPU `122.74 s` / P95 `80.700 ms` / ATE `0.159895`；相对首轮 `120.35 s` / `80.726 ms`，P95 稳定、CPU 约 `+1.99%`。
- pin easy 重复：CPU `82.87 s` / P95 `38.8965 ms` / ATE `0.159895`；初轮为 `82.98 s` / `40.2403 ms`。相对 default 两轮，CPU 约 `-31%/-32%`、P95 约 `-50%/-52%`，RSS 约 `+0.7 MiB`，绑核收益可重复。
- 性能 opt-in 建议为 `pin67 + threads1`：easy 重复可稳定降低 CPU/P95，medium 单次也明显改善。但 difficult + Schur **不可靠**（`ATE≈19.58 m`、`8.2 s` 空档、3 次 restart），不能称全局最佳或精度达标。
- Ceres difficult 对照已完成：完整覆盖 ATE `11.3486 m`（1103 poses）；共同 994 timestamps 上 Schur `19.6152 m`、Ceres `10.5793 m`，不能直接用全覆盖 ATE 跨覆盖排名。Ceres 仍远高于本轮额外筛选线 `<1 m`（不是用户验收标准）；未补 easy/medium，不推荐默认 Ceres。
- 最终边界：已找到可重复的性能 opt-in，但未找到在 difficult 同时满足精度的全场景最优。产品默认与校准不变；初始化/Schur 问题仍未解决。
- ARM microbench 3 轮：reduce `-5.26%`、update `-5.21%`，checksum 相同。
- 设备 `.62` 有 PowerVR OpenCL/Vulkan 和 `/dev/vipcore` NPU，但当前 VINS 没有 GPU/NPU 数据路径或端到端精度证据。
- 早期 `metrics.failure_reboots=0` 不能代表 Schur 后端没有受控重启；最终表以原始日志解析为准。ARM baseline difficult 已识别 **3 次 restart requests / 3 次 reset entries**；`actual_resets_completed` 无可靠标记，不能写成“3 次完成”。

### 1.1 实际实现要点
- 前端复用路径：`FeatureTracker::undistortedPtsInto()`、`ptsVelocityInto()` 直接写入复用输出向量，避免每帧返回临时 vector；`detection_half_mask_scratch_` 复用半分辨率 mask。金字塔 buffer 复用是此前已有能力，本轮不把它当新增。见 `src/vins_estimator/src/featureTracker/feature_tracker.cpp` 与 `feature_tracker.h`。
- 后端复用路径：`SchurUpdater::reduceTrackImpl()` 成功时保留 `_jacobians`，`update()` 直接复用，避免第二遍 Jacobian 计算；`TrackScratch_s` 复用 `Hx/Hf/r/N/S0/LDLT` 等 `thread_local` scratch，`schurBlockScratch()` 复用每 track 的 3×3 Schur 块。见 `src/vins_estimator/src/backend/schur_update.cpp` 与 `schur_update.h`。
- 这两项是 allocation/reuse 改动，不是新的求解器或精度改动；checksum 与旧 harness 相同，性能收益以第 6/7 节实测为准。

## 2. 平台与后端边界
旧 `.60` 结果、当前本地 x86 结果和 `.62` ARM 结果必须分开看：

| 证据 | 平台 | 后端 | 边界 |
|---|---|---|---|
| `docs/ARM_FAST_GRID_EVALUATION.md` | 历史 `.60` Orange Pi Zero 3，4×A53 | Ceres | 不能外推到 A76 异构平台 |
| `docs/REMOTE_ARM_PERFORMANCE.md` | 历史 `.60`，4×A53 | Ceres | 队列/freq 结论仅历史参考 |
| `docs/ARM_P0_P1_OPTIMIZATION.md` | 历史 `.60`，4×A53 | Ceres | solve/边缘化结论仅历史参考 |
| `docs/ARM_TRACKER_ACCELERATION.md` | 历史 `.60`，4×A53 | Schur EKF（另有 Ceres 单次对照） | medium 失稳不能直接套到 `.62` |
| 本文本地回放 | x86_64 工作站，32 逻辑核 | Schur EKF | 只用于本地筛选 |
| 本文 `.62` ARM | 2×A76 + 6×A55 | Schur EKF | default 三序列与 profile 验证完成 |

`.60` 是 4×A53 同构平台；`.62` 是 2×A76 + 6×A55 异构平台，且带 PowerVR GPU/NPU。线程、热、频率和求解器行为不能混算。

## 3. `.62` 设备实测
探测命令与原始输出：

```bash
scripts/probe_arm_acceleration.sh --host pinydrone@192.168.5.62
# 原始输出：output/balance_20261110/probe_arm.txt
```

| 项 | 实测值 |
|---|---|
| 板卡 | Orange Pi 4 Pro / `sun60iw2` |
| 系统 | Armbian 26.11 / Ubuntu 24.04，kernel `6.6.98-vendor-sun60iw2` |
| CPU | 2× Cortex-A76（最高 2.002 GHz）+ 6× Cortex-A55（最高 1.794 GHz） |
| `cpu_capacity` | A55 = 385；A76 = 1024 |
| governor | `schedutil`，未修改 |
| 内存 | MemTotal `3,930,724 kB` = **3.75 GiB**（free 3838 MiB） |
| Swap | SwapTotal `1,965,360 kB` = **1.87 GiB** |
| 可用内存 | 约 2.9–3.1 GiB（随负载变化） |
| NEON/FP16 flags | `asimd`、`asimdhp`、`fphp`、`asimddp` |
| OpenCV | 4.6.0，Baseline `NEON FP16`，TBB，OpenCL `YES` |
| OpenCL | `haveOpenCL=True`，PowerVR B-Series BXM-4-64，OpenCL 3.0 |
| GPU | DRM `card1` / `pvrsrvkm` / `renderD128`；Vulkan PowerVR BXM-4-64 MC1 |
| GPUs devfreq | 400–1008 MHz，当前 400 MHz，`simple_ondemand` |
| GPU 计算单元 | 1 CU，400 MHz，约 3.75 GiB 共享全局内存 |
| NPU | `/dev/vipcore`，`vipcore` 1.13.0，devfreq 最高 852 MHz |
| NPU 厂商库 | `/usr/lib/sun60iw2-npu/libVIPhal.so`、`libNBGlinker.so` |
| NPU 运行时 | 未发现标准 CLI；**unverified** |

GPU 型号来自 `clinfo`/`vulkaninfo` 的实际枚举，不是从 CPU 或板卡名称推断。

### 3.1 Thermal
| zone | 类型 | 探测读数 |
|---|---|---:|
| thermal_zone0 | cpub | 84.0–88.6 °C |
| thermal_zone1 | ddr | 80.4–81.5 °C |
| thermal_zone2 | npu | 81.4–82.7 °C |
| thermal_zone3 | cpul | 85.3–88.5 °C |
| thermal_zone4 | gpu | 82.4–83.3 °C |

构建期间读取到约 **83 °C**；探测期间 CPU 进一步达到 **88.5–88.6 °C**。散热/降频是 ARM 性能可复现的关键变量，必须记录开始/峰值/结束温度和频率，不要通过修改 governor 来“稳定”结果。

## 4. NEON、OpenCL、GPU/NPU 证据边界
NEON 已有路径来自 OpenCV AArch64 内核（`goodFeaturesToTrack`、`buildOpticalFlowPyramid`、`calcOpticalFlowPyrLK`、`cornerSubPix`、`pyrDown`）；VINS 没有手写 NEON。baseline `NEON FP16` 不等于 VINS 使用 FP16 数值；`-march=native` 默认 OFF，且有确定性发散记录。

OpenCL：VINS 使用 `cv::Mat` 和 `std::vector<cv::Point2f>`，透明 API 只对 `cv::UMat` 生效；即使改 UMat 也可能逐 kernel 回退，上传/下载成本必须实测。探索阶段组合 UMat smoke 曾以 **139** 退出，repro/输出见 `output/balance_20261110/opencl_umat_crash*`；该现象未定位，不能断言驱动 bug 或所有 UMat 不可用。最终 probe 不执行 UMat 计算，ARM 独占期间不重跑。

GPU/NPU：GPU 有 OpenCL/Vulkan/DRM，NPU 有 `/dev/vipcore` 和厂商库，但 VINS 没有 GPU/NPU 数据路径、往返计时或精度回归；NPU 无标准 CLI/模型。不能宣称加速或精度收益。

## 5. 本地最终二进制、测试与格式
| 项 | 值 |
|---|---|
| Git revision | `e28af31be74d9a46d32e6a028dc2f12abada784a` |
| 最终 frozen 二进制 | `7a4ce0e08c0584b19386070eb2b9c124190d0d75c1c160c42f66b8aa46ecd7d3` |
| 配置 SHA256 | `d7924795643d792865869705792ac671645dc2b5822f18cd9598549576af01e8` |
| CTest | 12 个程序，136 tests，0 failures / 0 disabled（gtest XML 计数） |
| Python 全量 | 45 passed（`output/balance_20261110/python_tests_verified.log`） |
| 新增 summary 测试 | 5 passed（本次运行） |
| format | `BreakConstructorInitializers` 使用复数键名；`SpacesBeforeTrailingComments: 2` 与 tidy 对齐；format-diff = 0 |
| clang-tidy | 两个 invalid check 名已修，`--verify-config` 为 No config errors；4 TU 变更行在修后规则无 warning/error |
| benchmark cpp | 本地 quick compile + format + tidy 均通过；校验值与旧 harness 相同 |
| 本地产品代码 | 已冻结，后续不再改产品代码 |

## 6. 本地 frozen 主表（baseline → optimized_frozen）
数据：EuRoC V2，完整序列，1× 速率，Schur EKF，单次运行。baseline 与 frozen 使用同一配置和 git revision；二进制分别为 `d2846b3b...` 与 `7a4ce0e0...`。

| 序列 | 版本 | CPU s | Peak RSS MiB | P50 / P95 / P99 ms | 输出/匹配 | ATE m |
|---|---|---:|---:|---|---:|---:|
| V2_01_easy | baseline | 14.68 | 90.3203 | 6.454 / 8.109 / 10.544 | 1083/1080 | 0.178288 |
| V2_01_easy | frozen | 14.28 | 90.4219 | 6.299 / 7.325 / 9.306 | 1083/1080 | 0.178288 |
| V2_02_medium | baseline | 15.35 | 90.8750 | 6.440 / 8.453 / 13.722 | 1130/1128 | 0.257460 |
| V2_02_medium | frozen | 14.83 | 90.9648 | 6.219 / 7.863 / 13.554 | 1130/1128 | 0.257460 |
| V2_03_difficult | baseline | 13.62 | 90.8594 | 7.049 / 10.847 / 14.079 | 1103/1102 | 0.323517 |
| V2_03_difficult | frozen | 13.48 | 90.9141 | 6.943 / 10.707 / 13.284 | 1103/1102 | 0.323517 |

变化：CPU `-2.725% / -3.388% / -1.028%`；P95 `-9.677% / -6.974% / -1.288%`；Peak RSS `+0.1016 / +0.0898 / +0.0547 MiB`。三条轨迹全部 8 列逐位相同，共同时间戳数 `1083 / 1130 / 1103`，位置最大差 `0.0 m`，共同时间戳 ATE 完全相同。

### 6.1 frames 加权 tracker 均值
| 序列 | 版本 | track | detect | flow | pyramid | input_image | features tracked/detected |
|---|---|---:|---:|---:|---:|---:|---|
| easy | baseline | 4.9186 | 2.2403 | 2.5057 | 0.1336 | 4.9212 | 182.563 / 18.968 |
| easy | frozen | 4.7289 | 2.1354 | 2.4346 | 0.1246 | 4.7320 | 182.563 / 18.968 |
| medium | baseline | 5.0154 | 2.3408 | 2.5148 | 0.1264 | 5.0197 | 158.434 / 41.177 |
| medium | frozen | 4.8282 | 2.2252 | 2.4527 | 0.1200 | 4.8316 | 158.434 / 41.177 |
| difficult | baseline | 5.4629 | 2.4344 | 2.8667 | 0.1300 | 5.4670 | 123.598 / 63.839 |
| difficult | frozen | 5.4145 | 2.4009 | 2.8556 | 0.1273 | 5.4177 | 123.598 / 63.839 |

`sum_frames == total_frames` 三条均成立（`2280 / 2348 / 1922`），累计 overflow dropped pairs 均为 0。本地日志经 `benchmark_euroc.parse_restart_diagnostics` 解析：三条序列的 `restart_requests=0`、`reset_entry_events=0`、`actual_resets_completed=None`。**不能再用 `metrics.failure_reboots=0` 代替 Schur 重启证据。**

## 7. RSS 稳定窗口、分配与微基准
### 7.1 RSS 稳定窗口
定义：丢弃启动前 10 s；early 窗口 `[10, 20) s`，late 窗口为末尾 10 s；分别取平均 RSS。

| 序列 | baseline early/late/delta MiB | frozen early/late/delta MiB |
|---|---|---|
| easy | 90.259 / 90.303 / +0.044 | 90.349 / 90.405 / +0.056 |
| medium | 90.815 / 90.859 / +0.044 | 90.897 / 90.946 / +0.050 |
| difficult | 90.770 / 90.829 / +0.059 | 90.844 / 90.893 / +0.049 |

首尾稳定窗口差约 `+0.04～0.06 MiB`，属于短序列内的小幅波动；本轮未观察到大项增长，但**2 分钟序列不足以证明长期无泄漏**。不能把 scratch 常驻、跨版本差异或 peak RSS 变化直接归因为泄漏因果。

### 7.2 分配探针
| 指标 | baseline | candidate | 变化 |
|---|---:|---:|---:|
| malloc/calloc/realloc calls | 477079 | 314192 | **-34.1%** |
| 累计请求字节 | 141576788 | 123944316 | **-12.45%** |
| checksum | 446728580.03886902 | 同左 | 一致 |

这是进程总请求分配量，不是 peak/live RSS；`LD_PRELOAD` 探针有 overhead，不能把该文件的 elapsed_ms 当正常耗时结果。它证明“减少 allocation”，不证明“降低 RSS”或“加速”。

### 7.3 微基准 3 轮
相同确定场景：5 clones、24 tracks、seed 12345、12000 reduce + 100 update。

| 操作 | baseline 3 轮 us/call | candidate 3 轮 us/call | 3 轮均值 |
|---|---|---|---:|
| reduce | 4.451 / 4.585 / 4.696 | 4.680 / 4.437 / 4.344 | 4.577 → 4.487 |
| update | 265.414 / 263.475 / 268.243 | 274.290 / 267.891 / 265.278 | 265.710 → 269.153 |

update 没有稳定收益，不能宣称加速；reduce 只有小幅变化。

### 7.4 现有内存上限
无需新增重复的内存上限：

- `src/vins_estimator/src/utility/runtime_utils.h:61` 已有 `BoundedHistory`；`visualization.cpp:45` 的 `path_history` 使用它。
- `src/vins_estimator/src/rosNodeTest.cpp:396` 图像队列由 `image_queue_size`（默认 100）限制。
- `src/vins_estimator/src/rosNodeTest.cpp:567` tracking 队列由 `tracking_queue_size`（默认 5）限制。
- `src/vins_estimator/src/estimator/estimator.cpp:288` 后端 feature 队列由 `max_feature_queue_size`（默认 10）限制。

## 8. 初次 optimized 的波动证据
原 `optimized` 在最终 tidy 修复前，只有 const-ref/参数名/format 变化；保留它作为波动证据，不作为主表。

| 序列 | optimized → frozen CPU | P95 | P99 | Peak RSS |
|---|---|---:|---:|---:|
| easy | -1.52% | -1.33% | -0.53% | -0.16% |
| medium | -2.43% | -4.66% | +0.84% | +0.03% |
| difficult | -3.65% | -2.17% | -0.59% | -0.03% |

两次 candidate 的轨迹/ATE 仍完全相同；但初次 optimized 相对 baseline 的 difficult CPU 为 `+2.7%`，frozen 为 `-1.03%`。这说明单次 CPU/P95 波动不可忽略，所有对比均不能称统计显著。

## 9. 默认检测/队列不冒然调整
当前默认：`feature_detector: gftt`、`opencv_threads: 1`、`tracking_queue_size: 5`、`drop_oldest`、`tracker_detection_interval: 1`。不直接改默认的原因必须按平台/后端注明：

- `[历史 .60 / 4×A53 / Schur EKF]` `gftt_pyramid` tracker 均值降 42%–46%、检测降 66%–69%，但 medium 两次候选回放失稳（ATE 17.4 m、3 次受控重启）。保留 opt-in。
- `[历史 .60 / 4×A53 / Ceres]` `fast_grid` CPU 降 18.6%–32.3%、P95 降 61.7%–79.1%，但共同窗口 ATE 在 -13.4%～+10.7% 间摆动。保留 opt-in。
- `[历史 .60 / 4×A53 / Ceres]` q1 约丢弃一半 20 Hz 图像；30 Hz 压力测试输出约 7.7 Hz；5 Hz 在 difficult 上 ATE 0.762 m。只可显式 opt-in。
- `[历史 .60 / 4×A53 / Ceres]` `opencv_threads=2` CPU 更高，P95 无实质改善；`freq=20` 更差；backend queue=2 P95 约 540 ms。
- `[本地 x86_64 / Schur EKF]` frozen 单次筛选无 ATE 变化、无 RSS 下降、CPU/P95 仅小幅变化；不足以改默认。
- `[.62 ARM / Schur EKF]` default 三序列与性能 profile 已完成；仍不能把 `.60` 的线程/求解器结论直接套用。

OpenVINS 的可借鉴点是每帧只建一次金字塔、网格 FAST、按网格并行和 `min_px_dist` 占用图；VINS 已有金字塔复用和 opt-in 检测器。这里的“并行不能消除后端 BA/边缘化成本”特指历史 `.60` 的 Ceres 后端；当前 `.62` 使用 Schur EKF，成本结构不同，必须单独测量。

## 10. 使用命令
```bash
# 设备只读探测
scripts/probe_arm_acceleration.sh --host pinydrone@192.168.5.62

# 主表：baseline vs optimized_frozen
python3 scripts/summarize_balanced_optimization.py \
  --baseline output/balance_20261110/local_replay/baseline \
  --candidate output/balance_20261110/local_replay/optimized_frozen \
  --output output/balance_20261110/local_replay/summary_frozen.json

# 波动证据：optimized vs optimized_frozen
python3 scripts/summarize_balanced_optimization.py \
  --baseline output/balance_20261110/local_replay/optimized \
  --candidate output/balance_20261110/local_replay/optimized_frozen \
  --output output/balance_20261110/local_replay/summary_compare_tidy.json

# 语法与测试
bash -n scripts/probe_arm_acceleration.sh
python3 -m pytest -q scripts/tests/test_probe_arm_acceleration.py
python3 -m pytest -q scripts/tests/test_summarize_balanced_optimization.py
```

summary 按 `(dataset, repeat)` 保留 r1/r2，不聚合；`features_avg` 拆成 tracked/detected 两个 frames 加权均值；只有 `*_ms` 输出 max；重启诊断直接读原 `estimator.log`，不重写历史 `metrics.json`。新 parser 稳定字段为 `restart_requests`、`reset_entry_events`、`actual_resets_completed`（无完成 marker，固定 `None`）、`failure_detection_triggers`、`starvation_restarts`、`soft_restart_escalations`、`legacy_failure_reboots`、`by_reason`、`events`；误导性的 `restart_executions` alias 已移除。

## 11. ARM `.62` default 主表与最终 profile 结论
ARM default 三序列已完成（`baseline_default → optimized_default`，Schur EKF，单次筛选）。baseline / optimized 二进制 SHA256 分别为 `00772c54…` / `f500174b…`，源码 head `e28af31…`，配置 SHA256 `d79247…`。

| 序列 | 版本 | CPU s | Peak RSS MiB | P50 / P95 / P99 ms | 输出/匹配 | ATE m | max gap s | restart req/entry |
|---|---|---:|---:|---|---:|---:|---:|---:|
| V2_01_easy | baseline | 125.00 | 98.6367 | 68.944 / 82.003 / 88.140 | 1083/1080 | 0.159895 | 0.10 | 0/0 |
| V2_01_easy | optimized | 120.35 | 98.5859 | 64.151 / 80.726 / 87.830 | 1083/1080 | 0.159895 | 0.10 | 0/0 |
| V2_02_medium | baseline | 125.44 | 102.105 | 65.370 / 81.508 / 92.263 | 1133/1131 | 0.275096 | 0.10 | 0/0 |
| V2_02_medium | optimized | 122.25 | 101.844 | 61.934 / 79.673 / 88.398 | 1133/1131 | 0.275096 | 0.10 | 0/0 |
| V2_03_difficult | baseline | 113.63 | 99.2109 | 66.941 / 90.598 / 106.702 | 994/993 | 19.6152 | **8.20** | 3/3 |
| V2_03_difficult | optimized | 109.90 | 99.6250 | 62.487 / 88.703 / 109.058 | 994/993 | 19.6152 | **8.20** | 3/3 |

变化：CPU `-3.720% / -2.543% / -3.283%`；P95 `-1.557% / -2.251% / -2.091%`；Peak RSS `-0.0508 / -0.2617 / +0.4141 MiB`。三条轨迹 8 列 `np.array_equal=True`，位置最大差 `0.0 m`，ATE/max gap/重启计数两版本相同。因此 default 候选没有引入轨迹差异，但也没有解决 difficult 的精度/长空档问题。

difficult 的 8.2 s 空档直接对应第 2 次 Schur reset 后的重初始化（约 8.25 s）。第 1/3 次分别约 1.61 s、1.46 s。8.25 s 窗口内出现 14 条 `Not enough features or parallax`、15 条 `unstable features tracking`、3 条 `misalign visual structure with IMU`、3 条 `Unable to perform dense Cholesky factorization`。ZUPT 唯一发生在 frame 627、`t=1413394997.606`，不在该空档窗口。

原 difficult log 中 `Backend overloaded` 行数为 0，只能称**未观察到该显式 backend overload 标记**；tracking `overflow_events=0`、`received_images==total_frames` 只证明该 tracking/输入路径未溢出，**不能排除所有队列/调度影响，也不能证明 backend 没丢 feature**。`submit=false` 是后端频率门控，不是 drop。根因仍未证明，不能把长空档简单归因于“只跟性能有关”。

### 11.1 ARM microbench
同场景：5 clones、24 tracks、seed 12345、12000 reduce + 100 update；3 轮，checksum 相同。

| 操作 | baseline 3 轮 us/call | optimized 3 轮 us/call | 均值变化 |
|---|---|---|---:|
| reduce | 13.608 / 13.824 / 14.340 | 13.106 / 13.115 / 13.356 | 13.924 → 13.192（**-5.26%**） |
| update | 729.035 / 732.613 / 731.295 | 693.519 / 691.240 / 693.955 | 730.981 → 692.905（**-5.21%**） |

证据：`output/balance_20261110/arm/evidence/microbench_all.txt`。

### 11.2 ARM profile 初筛与后续
ARM tests：`test_feature_tracker` 21 passed，`test_schur_update` 27 passed。

| 序列 | 配置 | CPU s | P95 ms | ATE m | gap s | poses | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| easy | opt_default r1 | 120.35 | 80.726 | 0.159895 | 0.1 | 1083 | 首轮 |
| easy | opt_default repeat | 122.74 | 80.700 | 0.159895 | 0.1 | 1083 | 与首轮稳定 |
| easy | threads2 | 170.11 | 85.36 | 0.159895 | 0.1 | 1083 | 更差 |
| easy | pin67 + threads1 r1 | 82.98 | 40.2403 | 0.159895 | 0.1 | 1083 | 初轮 |
| easy | pin67 + threads1 repeat | 82.87 | 38.8965 | 0.159895 | 0.1 | 1083 | repeat 稳定 |
| medium | opt_default | 122.25 | 79.673 | 0.275096 | 0.1 | 1133 | 单次 |
| medium | pin67 + threads1 | 83.06 | 39.689 | 0.275096 | 0.1 | 1133 | 单次，ATE 相同 |
| difficult | opt_default | 109.90 | 88.703 | 19.6152 | 8.2 | 994 | 3 restarts |
| difficult | pin67 + threads1 | 75.98 | 55.656 | 19.5772 | 8.2 | 994 | 性能更好但精度失败 |
| difficult | Ceres 实验 | 96.88 | 86.0108 | 11.3486 | 0.15 | 1103 | 完整覆盖；共同 994 timestamps ATE `10.5793`，仍不推荐 |

pin medium 相对 default opt：CPU 约 `-32.1%`，P95 约 `-50.2%`，ATE 不变。easy 的 pin 相对 default 两轮：CPU 约 `-31.05%` / `-32.48%`，P95 约 `-50.15%` / `-51.80%`，ATE 不变，RSS 约 `+0.7 MiB`；绑核收益可重复。

最终结论：性能 opt-in 为 `pin67 + threads1 + schur_ekf`。easy 两次重复相对 default 两次重复：CPU 约 `-31.05%/-32.48%`，P95 约 `-50.15%/-51.80%`，ATE 不变，RSS 约 `+0.7 MiB`；收益可重复。medium 单次：CPU 约 `-32.1%`、P95 约 `-50.2%`，ATE 不变。difficult + Schur 仍不可靠（ATE `19.5772 m`、gap `8.2 s`、3 次 restart）。Ceres difficult 完整覆盖 ATE 为 `11.3486 m`（1103 poses），但 Schur 同交集只有 994 poses；在共同 994 timestamps 上，Schur ATE `19.6152 m`、Ceres ATE `10.5793 m`，因此不能直接用 `11.3486` 与 Schur 的 994-pose ATE 排名。Ceres 仍远高于本轮额外筛选线 `<1 m`（不是用户验收标准），故不补 easy/medium、不推荐默认 Ceres。日志中 10 条 timeout warn 只是**行数**，不是精确 timeout 次数。最终未找到 difficult 同时满足精度的全场景最优；初始化/Schur 问题仍未解决。产品默认/校准不改。

### 11.3 可直接使用的 opt-in 运行命令
先确认本机大小核编号，不要在其他板卡上照抄 `6,7`：

```bash
cat /sys/devices/system/cpu/online
for c in /sys/devices/system/cpu/cpu[0-9]*/cpu_capacity; do
  printf '%s %s\n' "$c" "$(cat "$c")"
done
```

本 `.62` 板上 `cpu6/cpu7` 的 capacity 为 1024，`cpu0-5` 为 385；因此把已验证的性能 profile 固定到大核：

```bash
VINS_NODE=/path/to/verified/vins_node
CONFIG=/path/to/src/vins_estimator/config/euroc/euroc_mono_imu_config.yaml

taskset -c 6,7 "$VINS_NODE" --ros-args \
  -p config_file:="$CONFIG" \
  -p opencv_threads:=1 \
  -p backend_type:=schur_ekf
```

Ceres difficult 完整覆盖 ATE `11.3486 m`、共同 994 timestamps ATE `10.5793 m`，仍不足，故不推荐切换；Ceres 只保留为实验结果。该命令仅作为 Schur 性能 opt-in，适用 easy/medium 的已验证方向；difficult + Schur 仍不可靠，不能用于困难场景精度验收。`cpu6/7` 仅适用本机，先在目标机 probe `cpu_capacity`；不要把 affinity 硬编码进默认 launch/CMake/systemd。`config_file` 原 YAML 不改，`backend_type` 通过 ROS 参数覆盖；不覆盖部署目录/`deployed_vins_node`；这是未提交 git 的已验证 opt-in 运行方案，不是产品默认。无需新增通用 launcher，现有 `taskset` 足够。


## 12. 证据路径
- 本结果 JSON：`docs/ARM_BALANCED_OPTIMIZATION_RESULTS_20261110.json`
- 设备原始探测：`output/balance_20261110/probe_arm.txt`
- OpenCL 139 最小 repro：`output/balance_20261110/opencl_umat_crash_probe.py`、`opencl_umat_crash.txt`
- 本地 frozen 摘要：`output/balance_20261110/local_replay/summary_frozen.json`
- 波动证据摘要：`output/balance_20261110/local_replay/summary_compare_tidy.json`
- 分配探针：`output/balance_20261110/alloc_probe.c`、`alloc_probe.so`、`local_schur_allocations.txt`
- 微基准：`output/balance_20261110/local_schur_microbench.txt`
- CTest XML：`build/vins/test_results/vins/*.gtest.xml`（12 程序 / 136 tests）
- Python 最终验证日志：`output/balance_20261110/python_tests_verified.log`
- 最终源码 hash 清单：`output/balance_20261110/final_source_hashes.json`
- ARM default 证据目录：`output/balance_20261110/arm`；对比摘要 `arm/default_compare.json`
- ARM tests：`arm/evidence/test_feature_tracker_optimized.log`（21 passed）、`arm/evidence/test_schur_update_optimized.log`（27 passed）
- ARM difficult 长空档诊断：`output/balance_20261110/arm_difficult_diagnosis.md`
- 历史 ARM 证据：`docs/ARM_FAST_GRID_EVALUATION.md`、`docs/ARM_TRACKER_ACCELERATION.md`、`docs/REMOTE_ARM_PERFORMANCE.md`、`docs/ARM_P0_P1_OPTIMIZATION.md`
- OpenVINS 参考：`/home/sprout_stars/rmcode/open_vins/ov_core/src/track/TrackKLT.cpp`、`Grider_GRID.h`、`ov_core/CMakeLists.txt`
