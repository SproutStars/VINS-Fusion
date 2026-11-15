# ARM MH 双目最终结果（2026-11-15）

> **Phase status:** 早期 MH01 mono-only 压缩 bag /结果**无效**，不能用于 stereo 初始化或公平对比；仅保留为历史。本报告全部最终结果使用 **双目 MH01（cam0+cam1）** 与 **MH05（cam0+cam1）**，ARM 使用 MH official intrinsics。机器可读结果见 `output/arm_perf_20261115/final_arm_results_20261115.json`。

## 1. 环境与不变条件

- 远端：`pinydrone@192.168.5.62`，Orange Pi Zero 4，aarch64，6×Cortex-A55 + 2×Cortex-A76，3.8 GiB RAM，ROS 2 Jazzy，GCC 13.3。
- 构建：Release `-O3 -DNDEBUG`，`-j1` memory-safe，native/LTO OFF；estimator 固定 A76 CPU 6,7；player/observer 未绑定。
- 未修改 governor/thermal；未启动飞控、loop/global fusion 或 RViz。
- 所有 ATE 使用 Leica `T_BS`、sensor/prism origin；**GT 不参与初始化**。

## 2. 源码 / 二进制 / 配置 hash

| 项 | 值 |
|---|---|
| Baseline HEAD | `ccf2e69c03f295c4db8b4308be9addd9eae2cafa` |
| Baseline binary | `mh01_baseline_ccf2e69_vins_node` |
| Baseline SHA-256 | `e752363045fcbe672857e27f3d02bcdc7c1215ba8d70049456a3bbe3b6e50a57` |
| Candidate source tree | `a08c7cf00abd3e23f1ecef300bada9c97ee5afd1ed688999c28631af223a138a` |
| Candidate `estimator.cpp` | `e75941ce5ad571211fbd293393e6109741ce4ad90712df40b983f415abe96d5f` |
| Candidate final binary | `candidate_final_e75941ce_vins_node` |
| Candidate final SHA-256 | `cd9d367d46d4b4c8e076698bfa65fae923136cc5e1eeb4404fc56688a70f95bf` |
| fixed config | `stereo_fixed_frozen_abs.yaml`，SHA `943e8cc91cd83e5cb63acd5e6b6b42979c2bba5061a68b71e9e4663da416f7f0` |
| batchtrue config | `stereo_fixed_batchtrue_frozen_abs.yaml`，SHA `b8710404b2206c47468b61e5abce756cec460233b7e3200a75f210ba7393758e` |
| online config | `machine_hall_online_abs.yaml`，SHA `eb1949d6cbdca1e28170db23371b4cb6907fbffc3aba5de5d10c1ffd29090e3d` |

`Estimator::resampleImuIntervalSamples` 已确认 `bool` + 7 参数，本地/远端一致。

## 3. 标定差异（必须分开报告）

| 文件 | SHA-256 | fx / fy / cx / cy | k1 / k2 |
|---|---|---|---|
| `cam0_v1_mh.yaml`（ARM official） | `ff2b75a7…` | 458.654 / 457.296 / 367.215 / 248.375 | −0.28340811 / 0.07395907 |
| `cam0_machine_hall.yaml`（official） | `6fb75ee1…` | 同上 | 同上 |
| `cam1_v1_mh.yaml`（ARM official） | `2c5a2132…` | 457.587 / 456.134 / 379.999 / 255.238 | −0.28368365 / 0.07451284 |
| `cam1_machine_hall.yaml`（official） | `85adf2e2…` | 同上 | 同上 |
| `cam0_pinhole.yaml`（本机 generic/Vicon） | `04bf3968…` | **461.1586 / 459.7529 / 362.6593 / 248.5211** | **−0.295456 / 0.086623** |

- `cam0_v1_mh` 与 `cam0_machine_hall` 的内参/畸变数值一致；`cam1_v1_mh` 与 `cam1_machine_hall` 亦一致。
- 本机 `cam0_pinhole` 是 Vicon/generic，和 MH official **不同**；local generic 与 ARM official 不得混比。
- 本报告 ARM baseline/candidate fixed 均使用 official `cam0_v1_mh`/`cam1_v1_mh`；online 组使用 `cam0_machine_hall`/`cam1_machine_hall`（数值相同）。

## 4. 最终双目结果

| 组别 | 序列 | poses | cam0 actual/expected | cam1 actual/expected | CPU s | RSS MiB | P50/P95/P99 ms | ATE RMSE m | init s | reset | latency miss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline fixed | MH01 | 1831 | 3682/3682 | 3682/3682 | 132.53 | 106.34 | 41.92/47.85/51.92 | 0.16935 | 1.004 | 0 | 0 |
| **Candidate fixed** | MH01 | 1831 | 3682/3682 | 3682/3682 | **132.19** | **100.95** | 41.74/47.71/51.78 | **0.16913** | 1.004 | 0 | 0 |
| Candidate batch true | MH01 | 1831 | 3682/3682 | 3682/3682 | 131.91 | 100.82 | 41.54/47.42/52.24 | 0.169127 | 1.004 | 0 | 0 |
| Candidate online-calib | MH01 | 1831 | **3681/3682** | **3681/3682** | 135.13 | 101.51 | **n/a**（3 个有效样本） | 0.18820 | 1.004 | 0 | 1828 |
| Baseline fixed | MH05 | 1126 | 2273/2273 | 2273/2273 | 82.79 | 97.50 | 42.18/50.33/55.11 | 0.64241 | 0.98 | 0 | 0 |
| **Candidate fixed** | MH05 | 1122 | 2273/2273 | 2273/2273 | **82.62** | **95.75** | 41.92/49.62/52.51 | 0.64407 | 1.38 | 0 | 0 |

IMU 期望数：MH01 36820、MH05 22721；benchmark observer 直接统计的是 cam0/cam1 图像，IMU 数来自 bag metadata，未单独计数。

## 5. 对比结论（ARM 同环境、同 official 内参）

### MH01 fixed candidate vs baseline

- ATE：`0.16913` vs `0.16935`，**−0.00022 m（−0.13%）**。
- CPU：`132.19` vs `132.53 s`，**−0.34 s（−0.26%）**。
- RSS：`100.95` vs `106.34 MiB`，**−5.39 MiB（−5.07%）**。
- P95：`47.71` vs `47.85 ms`。
- 结论：ARM 上 fixed candidate **基本 CPU 中性、精度略好**；local x86 的 −19.6% 不能外推到 ARM。

### MH05 fixed candidate vs baseline

- ATE：`0.64407` vs `0.64241`，**+0.00166 m（+0.26%）**。
- CPU：`82.62` vs `82.79 s`，**−0.17 s（−0.21%）**。
- RSS：`95.75` vs `97.50 MiB`；P95 `49.62` vs `50.33 ms`。
- 结论：difficult 序列上 fixed candidate 也基本中性；精度小幅回退，需按序列看。

### QR + batch true vs batch false（fixed）

- ATE：`0.16912707827` vs `0.16912707819`，差异可忽略。
- CPU：`131.91` vs `132.19 s`，**−0.28 s（−0.21%）**。
- 结论：batch IMU covariance 在本 ARM MH01 上**没有可测收益**；保持 opt-in/ablation，不建议默认开启。

### Online calibration（estimate_extrinsic=1, estimate_td=1, QR）

- ATE：`0.18820` vs fixed `0.16913`，**+0.01907 m（+11.28%）**。
- CPU：`135.13` vs `132.19 s`，**+2.94 s（+2.22%）**；RSS +0.56 MiB。
- latency：**n/a**。`latency_misses=1828/1831`，仅 3 个有效样本（odometry header=image+td，旧 observer 精确匹配失效），不得与 fixed latency 比较。
- 结论：本轮 ARM MH01 online-calib **精度退化**；保持 **opt-in**，不通过调参掩盖该结果。

## 6. Stereo topic 实际接收

- 所有 fixed/batchtrue MH01：cam0 **3682/3682**，cam1 **3682/3682**。
- MH05 baseline/candidate fixed：cam0 **2273/2273**，cam1 **2273/2273**。
- Online MH01：cam0 **3681/3682**，cam1 **3681/3682**（少 1 帧，属接收/尾帧时序，不影响 1831 poses 与 ATE）。
- IMU 由 bag 播放并由 estimator 消费；benchmark 未直接计数，期望数见上。

## 7. 证据与限制

- 机器可读：`output/arm_perf_20261115/final_arm_results_20261115.json`
- 逐 run 结果：`output/arm_perf_20261115/results/<label>/`
- 构建/运行证据：`output/arm_perf_20261115/evidence/`
- 配置/二进制/truth：`output/arm_perf_20261115/{configs,binaries,truth}/`
- 远端证据：`~/vins_arm_perf_20261115/evidence/`
- 单次全序列；未改 governor/thermal；A76 6,7 固定；player/observer 未绑定。
- 本机 x86、local generic/Vicon 内参、SVD 早期结果与本次 ARM official/QR 结果不可跨环境或跨内参排名。
- 早期 mono-only MH01 结果无效，已从最终比较中排除。
