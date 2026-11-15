# ARM MH 基线、环境与优化审计（2026-11-15）

> **Phase status（2026-11-15 final）：** 早期 MH01 **mono-only** 压缩/评测无效，不能用于 stereo 初始化或公平对比；本文第 4 节的早期 mono-only 数字仅保留为历史，**不进入最终结论**。最终双目 baseline/candidate、batch ablation 与 online-calib 结果见：
> - `docs/ARM_FINAL_RESULTS_20261115.md`
> - `output/arm_perf_20261115/final_arm_results_20261115.json`
>
> 最终 ARM fixed：MH01 ATE 0.16913 m / CPU 132.19 s / cam0=cam1=3682；MH05 ATE 0.64407 m / CPU 82.62 s / cam0=cam1=2273。Online-calib 为 opt-in，ARM MH01 ATE 0.18820 m、latency n/a（1828/1831 misses），不得与 fixed latency 比较；batch true 无实质收益。ARM 使用 MH official intrinsics；本机 generic/Vicon 内参与 ARM official 分开报告。

## 1. 结论摘要

- 远端 `pinydrone@192.168.5.62` 可达且空闲；Orange Pi Zero 4，aarch64，6×Cortex-A55 + 2×Cortex-A76，3.8 GiB RAM，ROS 2 Jazzy，GCC 13.3。
- 使用 **clean HEAD `ccf2e69c03f295c4db8b4308be9addd9eae2cafa` 的隔离源**构建 ARM `vins_node`，未触碰远端原部署和其他子代理工作区。
- 早期 mono-only 阶段完成 MH05 全程与 3 s 静止烟测；这些数字**不作为最终 stereo 结论**。
- **早期 MH01 mono-only 尝试未产出可用指标**：离线 truth 读取器在 message-zstd MCAP 上反序列化 `/leica/position` 失败。该 mono-only 路线已被最终双目 MH01 实验替代；最终结果见最终报告/JSON。
- 当前最可落地的 ARM 消耗优化不是重复实现 OpenVINS 的 Givens/QR、batch IMU、压缩或 KLT 金字塔，而是：`schur_max_update_observations`、`schur_batch_imu_covariance`、`max_cnt`/`tracking_queue_size` 的 YAML A/B，以及少量后端 chi2/缓存代码优化。

## 2. 隔离构建与来源

| 项 | 值 |
|---|---|
| 源码 | `git archive ccf2e69` 最小源（仅 `src/vins_estimator`、`scripts`） |
| 远端源码 | `/home/pinydrone/vins_arm_perf_20261115/workspace/VINS-Fusion` |
| 构建树 | `/tmp/vins_arm_build_ccf2e69`（tmpfs，重启会丢失；二进制已快照） |
| 二进制 | `/home/pinydrone/vins_arm_perf_20261115/binaries/mh01_baseline_ccf2e69_vins_node` |
| SHA-256 | `e752363045fcbe672857e27f3d02bcdc7c1215ba8d70049456a3bbe3b6e50a57` |
| ELF | ELF64 AArch64 PIE；`ldd` 无 missing |
| 编译 | `Release -O3 -DNDEBUG -std=c++17`；`BUILD_TESTING=OFF`；native/LTO OFF |
| 依赖 | `vins_common`/`camera_models` 来自 `/home/pinydrone/vins_stereo_20261112/workspace/install` |
| 原部署 | `~/pinydrone_ws`、其他 `~/vins_*` 未修改 |

`vins_node` 在 84% 时已链接完成；随后停止了本轮自己的 KITTI/sample 额外目标编译，未杀死其他任务。`candidate_clean_vins_node` 实际位于 `~/vins_stereo_20261112/binaries/`，源码为 `fc607a9` + 12 个 dirty 文件，**不是** `ccf2e69`，因此未复用。

## 3. 数据与 truth

| 序列 | 远端路径 | 规模 | 时长 | cam0 / IMU / Leica |
|---|---|---|---|---|
| MH05 difficult | `/home/pinydrone/vins_qr_imu_budget_20261115/bags/MH_05_difficult` | 1.5 GiB MCAP | 114.773 s | 2273 / 22721 / 1665 |
| MH01 easy | `/home/pinydrone/vicon_room2/bags/machine_hall/MH_01_easy` | 1.1 GiB message-zstd MCAP | 187.749 s | 3682 / 36820 / 3099 |

- MH01 为 mono-only（cam0+IMU+Leica），由原始 2.5 GiB bag 转换；`ros2 bag play` 已在远端验证可读。
- MH01 的 truth 额外提供 CSV：`/home/pinydrone/vins_arm_perf_20261115/truth/machine_hall/MH_01_easy_leica.csv`，3099 行，SHA-256 `161253408db7beb3868bd1205d8ee1a29a8da1776afa63fc73a1c89d8a0e0b72`。
- Leica 外参：`/home/pinydrone/vicon_room2/bags/machine_hall/mav0/leica0/sensor.yaml`，SHA-256 `41de3d21b59e5711e4912f1f9e0d84b6824346f69b14eaab84def80656ac35ff`。
  `T_BS`（prism → body）平移 `[0.0748903, -0.0184772, -0.120209]`。
- 所有 ATE 使用 `sensor_type: position`、`--truth-sensor-config`，比较 `p_sensor = p_body + R_est * t_body_sensor` 与 Leica 位置；**GT 不参与初始化**。
- 远端剩余约 758 MB；machine_hall 全量约 12 GB，当前存储不允许全量同步。MH01 + MH05 已覆盖“MH01 / MH05 / 至少一条困难全程”。

## 4. 早期基线（其中 mono-only MH01 已废弃）

### 4.1 MH05 前 3 s 静止段

- 61 张 cam0，**0 输出位姿**；无崩溃、无 failure/restart、无初始化误触发。
- CPU 1.85 s，峰值 RSS 85.65 MiB，18 threads，wall 8.48 s。
- estimator log 持续输出 `Not enough features or parallax; Move device around`；符合纯静止段不产生 VIO 初始化的预期。
- Tracker：`features_avg=191.3`，`track_ms=25.49/34.02 ms`，`detect_ms=7.05/25.71 ms`，`overflow=0`。

### 4.2 MH05 difficult 全程（clean HEAD）

| 指标 | 值 |
|---|---:|
| output_poses | 1091 |
| received_images | 2273 / 2273（cam0 100%） |
| CPU seconds | 73.5 s |
| avg CPU (one core) | 61.96% |
| peak RSS | 99.18 MiB |
| peak threads | 22 |
| wall | 118.63 s |
| latency P50 / P95 / P99 | 32.41 / 37.68 / 43.99 ms |
| ATE RMSE (Leica T_BS) | 1.1595 m |
| ATE median / P95 | 1.0708 / 1.5888 m |
| matched poses | 786 |
| trajectory duration | 108.8 s |
| max output gap | 2.8 s |
| init latency (first output - bag start) | 4.48 s |
| failure/reboot/starvation | 0 / 0 / 0 |
| latency_misses | 0 |

日志确认：`Backend type: schur_ekf (schur window=5 ... max_features=120 ...)`；Leica T_BS 已写入 `manifest.json` 的 `truth.MH_05_difficult.extrinsic`。

### 4.3 MH01 状态（早期 mono-only，已废弃，不用于最终对比）

首次 MH01 回放在 `read_bag_topics`/truth 反序列化处失败，错误为 message-level zstd MCAP 的 `/leica/position` `Fast CDR exception`；**尚未产生 estimator 指标**。bag 播放本身已验证。现在使用外部 truth CSV，重跑命令：

```bash
ssh pinydrone@192.168.5.62 \
  'bash /home/pinydrone/vins_arm_perf_20261115/evidence/run_mh01_full.sh'
```

该脚本默认：
- `BIN=/home/pinydrone/vins_arm_perf_20261115/binaries/mh01_baseline_ccf2e69_vins_node`
- `CFG=/home/pinydrone/vins_arm_perf_20261115/configs/baseline_mh_abs.yaml`
- `LABEL=baseline_ccf2e69_mh01_full`
- `--cpu-affinity 6,7`
- `--truth-csv MH_01_easy:=.../MH_01_easy_leica.csv`
- `--truth-sensor-config .../leica0/sensor.yaml`

## 5. 复现与最终候选矩阵

### 5.1 配置边界

- 算法参数由 `config_file` YAML 读取；**`--parameter stage_timing:=1` 这类算法参数不一定生效**（实际 log 仍为 0）。
- `opencv_threads`、`tracking_queue_size` 是节点参数，CLI 可用；但为可审计性，最终 ablation 也应写入 YAML。
- 所有 ablation YAML 必须：
  1. 从 `configs/baseline_mh_abs.yaml` 复制；
  2. 将 `cam0_calib`/`cam1_calib` 改为绝对路径；
  3. 修改目标算法键（`schur_nullspace_method`、`schur_batch_imu_covariance`、`schur_max_update_observations`、初始化/时偏等）；
  4. 在 estimator log 中核查实际值后再采信。

### 5.2 MH01/MH05 最终候选

远端已准备参数化脚本：`/home/pinydrone/vins_arm_perf_20261115/evidence/run_arm_candidate_matrix.sh`。

```bash
BIN=/path/to/candidate_vins_node \
CFG=/home/pinydrone/vins_arm_perf_20261115/configs/candidate_xxx.yaml \
LABEL=candidate_xxx \
VINS_GIT_REVISION=<candidate_git_head> \
bash /home/pinydrone/vins_arm_perf_20261115/evidence/run_arm_candidate_matrix.sh
```

脚本按顺序跑 MH05 difficult 全程和 MH01 全程，统一 `--cpu-affinity 6,7`：
- MH05 使用 bag 内 `/leica/position` + Leica T_BS；
- MH01 使用 external truth CSV + Leica T_BS；
- 输出到 `$ROOT/results/$LABEL_mh05_full` 与 `$LABEL_mh01_full`。

## 6. OpenVINS 对照与优化建议

### 6.1 已落地，勿重复

| OpenVINS 机制 | 当前实现 |
|---|---|
| `UpdaterHelper::nullspace_project_inplace` Givens QR | `src/vins_estimator/src/backend/schur_update.cpp` 的 `givens_nullspace_project_inplace`；`schur_nullspace_method: qr` |
| Givens measurement compression | 当前后端用 rank-revealing Schur 信息矩阵压缩（`schur_update.h`），旧的 H=A 信息形式已替换；不要再叠一层 |
| batch IMU propagation | `ekf_state.cpp` 的 `batch_imu_covariance_`，`schur_batch_imu_covariance`（默认 false） |
| KLT 金字塔复用 | `feature_tracker.cpp` 的 `gftt_pyramid`、`buildFlowPyramid`、derivative cache、scratch buffers |
| OpenCV 线程控制 | `rosNodeTest.cpp` 的 `opencv_threads`（默认 1） |
| 观测预算 | `schur_max_update_observations`（0=不限；已有 240/480 实验入口） |
| CPU 亲和 | benchmark `--cpu-affinity`，estimator 固定到 A76 6,7 |

OpenVINS 参考位置：`ov_msckf/src/update/UpdaterHelper.cpp`、`ov_msckf/src/update/UpdaterMSCKF.cpp`、`ov_core/src/track/TrackKLT.cpp`、`ov_msckf/src/core/VioManager.cpp`。

### 6.2 建议的 A/B（优先顺序）

1. **`schur_max_update_observations=240` 与 `480`**：当前默认 0；预计显著降低后端每帧观测数/CPU，风险是精度与更新频率下降。
2. **`schur_batch_imu_covariance=true`**：已有实现，默认 false；需与 false 做同序列 A/B，关注 ATE 与 CPU/RSS。
3. **`max_cnt=150` 或 `120`**：后端 `max_features=120` 已封顶；降低前端检测数可能直接省 GFTT/LK CPU，但会改变特征分布，必须看精度。
4. **`tracking_queue_size=2` 或 `1`**：降低排队延迟/丢帧策略；`1` 可能减少可用帧，必须记录 `received_images`、`output_poses`、`max_output_gap`。
5. **`opencv_threads=1`**：默认已是 1；2 在 2 核 affinity 下可能反噬。
6. **`marginalization_threads`**：只对 Ceres 后端有意义；`schur_ekf` 下不要把它当作 ARM 优化项。
7. **平台编译实验**：`-mcpu=cortex-a76.cortex-a55` 或 `-mtune=cortex-a76`（保持 `-march=native` 关闭以保证可复现）；LTO 仅作为独立实验，不混入基线。

### 6.3 代码级候选（交给主代理）

- 每个 feature 的 chi2 `H0 P H0^T` 只涉及少数 active clone；可抽取 active-clone 子协方差，避免对完整 `n×n` 状态反复乘。
- `Hx/Hf/r` 的 `allFinite()` 全矩阵扫描与异常检查可在填充时增量完成；当前 scratch 已复用，收益中等。
- 可评估 raw reprojection pre-gate，先剔除明显 outlier 再做 nullspace/SVD；这会改变 gating 语义，必须与当前 chi2 分布对照。
- 不建议在当前已很紧凑的 tracker 上再加缓存层；先通过 YAML 预算/特征数取得可测收益。

## 7. 限制

- MH01 为 mono-only message-zstd bag，truth 用 external CSV；MH05 为原始 MCAP，truth 用 bag topic。两者 ATE 定义一致（Leica T_BS，sensor_origin），但输入存储形式不同。
- 单次测量；未改 governor/thermal，未做重复统计；A76 6,7 固定，player/observer 未绑定。
- 远端存储仅约 758 MB，MH02/03/04 未同步；如需更多序列必须先释放空间。
- 本报告不把 MH05 的 1.1595 m 与旧设备/旧配置结果直接排名；旧结果可能使用不同 truth 基准或不同二进制。
