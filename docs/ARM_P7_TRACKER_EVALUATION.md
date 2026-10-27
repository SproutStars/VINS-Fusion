# ARM P7 保持策略与 tracker 验证（2026-10-27）

本报告区分机制验证、定位有效性和前端性能。输出持续并不等于定位有效；
单次运行的延迟下降也不能抵消轨迹发散或重新初始化造成的长时间无输出。

## 环境与复现边界

- Orange Pi Zero 3 / AArch64 / ROS 2 Jazzy，SSH `pinydrone0@192.168.5.60`。
- 独立工作区 `~/vins_p7_eval_20261027/workspace`，未修改原部署工作区。
- 数据 `~/vicon_room2/bags/{V2_01_easy,V2_02_medium,V2_03_difficult}`。
- 全序列、1 倍速、ROS domain 173，串行执行；每次回放前读一遍对应 mcap 预热。
- 相同二进制 `~/vins_p7_eval_20261027/results/binaries/p7_tracker`，SHA-256：
  `a7a4acda3edf48bd1a9a8edfdfbe6d1221002605f2a28e2d7eb51015f037b587`。
- 源码基点 `a8a08064224d0bd55c732be7e97cbb54fc4d5d32` + P7/tracker 未提交补丁；
  补丁 SHA-256 `4513eebae28c9605c40ee5adbeb5b300c5397e939d689868a136b862d7910718`。
- `Release`、LTO 关闭、native 优化关闭；同一 EuRoC mono 配置，`max_cnt=150`、
  `flow_back=1`、GFTT、OpenCV 1 线程、后端 `schur_ekf`、P5/P6 关闭。
- 结果根 `~/vins_p7_eval_20261027/results/benchmark`；本地原始副本
  `output/arm_p7_20261027/benchmark`（不提交原始大日志）。每轮 manifest 保留二进制哈希、
  完整配置、参数与命令。构建脚本/测试夹具后续修复不改变评测二进制。后续补齐的 P7 参数热修改拒绝
  检查也不在此快照中；所有回放参数均在启动时设置，因此不影响本矩阵。

## 测量口径

ATE 是 Vicon 时间插值后刚性对齐的 RMSE，不拟合尺度或时间偏移。
同时报告输出数量、覆盖时长和最大时间空档；覆盖不同的原始 ATE 不能直接当作
同一时间窗口上的精度增益。CPU 百分比采用单核口径，可超过 100%。

tracker 阶段耗时由所有 periodic/final 窗口按 `frames` 加权，最终
`sum(frames) == total_frames`；`final` 的均值只代表最后一段，不能代表全程。
`detect_ms` 包含 mask/补点检测，`pyramid_ms` 是 KLT 金字塔构建；它们不等同于
单独计时的 GFTT 内部步骤。队列统计使用累计 `overflow_events` 和
`overflow_dropped_pairs`，不数节流告警行。

## 运行结果

共 **13 次完整 bag 回放**。P7 与 control 在 difficult 各 3 次；其他配置为探索性的单次筛选，
不足以支持修改默认值。所有控制组均使用此轮新 tracker 实现，P7 关闭、queue=5、
drop_oldest、检测间隔=1；因此不是相对旧版本代码的整体性能 A/B。

参数差异：`p7hold` 只启用 P7；`detect2` 只设检测间隔 2；`trackeropt` 同时设
检测间隔 2 与 `latest`；`queue1` 只把队列设为 1。完整参数、CPU/RSS、P99、精确计数、
首尾时间戳见 [机器可读结果](ARM_P7_TRACKER_RESULTS.json)。下表按实际运行顺序排列。

### difficult：P7 重复验证与 tracker 筛选

| run | ATE RMSE m | 匹配覆盖 s | 输出/匹配 | P50/P95 ms | CPU % | 最大空档 s | 受控重启 | hold/退出/重建 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p7ctl_difficult_1 | 0.4368 | 74.45 | 732/731 | 328.4/514.3 | 135.6 | 0.40 | 0 | 0/0/0 |
| p7hold_difficult_1 | 11556.6946 | 110.05 | 1084/1083 | 322.5/498.8 | 109.2 | 0.30 | 0 | 30/30/30 |
| trackeropt_difficult_1 | 4.0699 | 74.75 | 315/314 | 173.9/466.9 | 142.7 | 39.50 | 1 | 0/0/0 |
| detect2_difficult_1 | 25.1489 | 110.00 | 628/627 | 275.0/467.4 | 132.8 | 26.90 | 3 | 0/0/0 |
| p7ctl_difficult_2 | 12.4635 | 109.15 | 173/172 | 346.0/1202.0 | 168.8 | 46.70 | 2 | 0/0/0 |
| p7hold_difficult_2 | 4303.5176 | 110.00 | 1082/1081 | 321.7/486.5 | 109.3 | 0.25 | 0 | 30/30/30 |
| p7hold_difficult_3 | 3467.5963 | 82.45 | 812/811 | 320.9/516.2 | 127.3 | 0.50 | 0 | 21/20/20 |
| p7ctl_difficult_3 | 6.8837 | 19.45 | 189/189 | 324.2/548.2 | 166.0 | 0.20 | 1 | 0/0/0 |
| queue1_difficult_1 | 1.8670 | 98.95 | 800/799 | 121.0/170.2 | 130.2 | 17.45 | 1 | 0/0/0 |

### easy/medium：仅改检测间隔

| run | ATE RMSE m | 匹配覆盖 s | 输出/匹配 | P50/P95/P99 ms | CPU % | 最大空档 s | 受控重启 |
|---|---:|---:|---:|---:|---:|---:|---:|
| p7ctl_easy_1 | 0.2273 | 105.20 | 1051/1048 | 318.6/343.2/408.0 | 118.3 | 0.40 | 0 |
| detect2_easy_1 | 0.2151 | 107.80 | 1082/1079 | 281.4/324.7/355.6 | 114.9 | 0.15 | 0 |
| p7ctl_medium_1 | 0.3266 | 109.15 | 1087/1085 | 323.7/354.9/449.1 | 117.9 | 0.45 | 0 |
| detect2_medium_1 | 2.6070 | 110.00 | 1070/1068 | 288.0/338.7/889.1 | 121.0 | 2.85 | 1 |

### tracker 全程分阶段耗时

以下均为按处理帧数加权的均值，单位 ms；溢出次数和丢弃帧对数为精确累计值。

| run | 处理帧 | 溢出/丢弃帧对 | 排队 | track | 检测 | 光流 | KLT 金字塔 |
|---|---:|---:|---:|---:|---:|---:|---:|
| p7ctl_difficult_1 | 1313 | 609/609 | 210.74 | 87.62 | 45.58 | 39.05 | 2.72 |
| p7hold_difficult_1 | 1278 | 604/604 | 215.68 | 88.83 | 46.80 | 39.09 | 2.68 |
| trackeropt_difficult_1 | 1552 | 74/370 | 102.57 | 69.29 | 31.19 | 35.06 | 2.79 |
| detect2_difficult_1 | 1532 | 390/390 | 155.92 | 70.60 | 32.40 | 35.16 | 2.79 |
| p7ctl_difficult_2 | 1313 | 609/609 | 216.62 | 87.40 | 45.39 | 39.03 | 2.70 |
| p7hold_difficult_2 | 1277 | 605/605 | 211.09 | 88.86 | 46.79 | 39.11 | 2.69 |
| p7hold_difficult_3 | 1316 | 606/606 | 207.89 | 87.26 | 45.44 | 38.85 | 2.70 |
| p7ctl_difficult_3 | 1313 | 609/609 | 210.13 | 87.37 | 45.44 | 38.95 | 2.71 |
| queue1_difficult_1 | 1282 | 640/640 | 18.90 | 88.05 | 46.16 | 38.84 | 2.76 |
| p7ctl_easy_1 | 1410 | 870/870 | 205.94 | 80.02 | 44.51 | 32.53 | 2.69 |
| detect2_easy_1 | 1976 | 304/304 | 201.38 | 56.95 | 23.57 | 30.43 | 2.69 |
| p7ctl_medium_1 | 1360 | 988/988 | 220.36 | 86.52 | 47.58 | 35.96 | 2.69 |
| detect2_medium_1 | 1946 | 402/402 | 211.68 | 60.24 | 26.45 | 30.88 | 2.67 |

### 输入与首尾覆盖

bag 的相机 header 跨度：easy 113.95 s / 2280 帧，medium 117.35 s / 2348 帧，
difficult 116.75 s / 1922 帧。首输出延迟 = 首位姿 header − 首相机 header；尾部缺失 =
末相机 header − 末位姿 header。这不是内部初始化耗时。所有 run 的 latency_misses=0。

| run | 接收/相机总帧 | 首输出延迟 s | 末尾缺失 s | 已发布位姿跨度 s |
|---|---:|---:|---:|---:|
| p7ctl_difficult_1 | 1922/1922 | 42.20 | 0.00 | 74.55 |
| p7hold_difficult_1 | 1917/1922 | 6.60 | 0.00 | 110.15 |
| trackeropt_difficult_1 | 1922/1922 | 41.90 | 0.00 | 74.85 |
| detect2_difficult_1 | 1922/1922 | 6.65 | 0.00 | 110.10 |
| p7ctl_difficult_2 | 1922/1922 | 7.50 | 0.00 | 109.25 |
| p7hold_difficult_2 | 1917/1922 | 6.65 | 0.00 | 110.10 |
| p7hold_difficult_3 | 1922/1922 | 34.20 | 0.00 | 82.55 |
| p7ctl_difficult_3 | 1922/1922 | 41.60 | 55.70 | 19.45 |
| queue1_difficult_1 | 1922/1922 | 17.70 | 0.00 | 99.05 |
| p7ctl_easy_1 | 2280/2280 | 8.45 | 0.00 | 105.50 |
| detect2_easy_1 | 2280/2280 | 5.85 | 0.00 | 108.10 |
| p7ctl_medium_1 | 2348/2348 | 8.00 | 0.00 | 109.35 |
| detect2_medium_1 | 2348/2348 | 7.15 | 0.00 | 110.20 |

### 结论与默认值

- **P7：机制存在，定位恢复不通过。** 三次 difficult 的 ATE 为 11556.69 / 4303.52 /
  3467.60 m，不能作为可用定位输出。保留 `schur_starvation_hold=false`，不部署为默认恢复策略。
  进入/退出 hold 与 clone 重建日志证明状态机实际触发；它们不证明几何恢复或有效校正。
- **控制组也不稳定，不能夸大因果归属。** control 的 ATE 为 0.4368 / 12.4635 / 6.8837 m，
  匹配覆盖分别为 74.45 / 109.15 / 19.45 s。第一轮的较低 ATE 漏掉了前 42.2 s，
  第二轮存在 46.7 s 空档，第三轮的最大空档虽只有 0.2 s，却只输出了短片段。
  不把某一次 control 当作全程稳定的基准，也不将所有预先出现的漂移归因于 P7。
- **检测间隔 2：有降耗效果，尚不能作为默认。** difficult 的 tracker 均值从首轮 control
  87.62 ms 降至 70.60 ms，检测阶段从 45.58 ms 降至 32.40 ms；但仍超出 20 Hz 的
  50 ms 预算，该轮 ATE 25.15 m、空档 26.9 s。easy 的单次结果较好，不能覆盖 difficult
  的反例；medium 的 ATE 从 0.3266 m 升至 2.6070 m，P99 从 449.1 ms 升至 889.1 ms，
  出现一次受控重启。三个序列 tracker 均值虽分别降至 57.0 / 60.2 / 70.6 ms，
  仍未达到 50 ms 帧预算。保留 `tracker_detection_interval=1`。
- **队列 1：减少等待，不是算力修复。** 单轮 difficult P50/P95 为 121.0/170.2 ms，
  但 P99 仍为 789.0 ms、ATE 1.867 m、空档 17.45 s。保留默认 queue=5；queue=1 仅作为
  后续重复验证的候选，不据此声称稳定性改善。
- **latest + 间隔 2：不能只看 overflow 次数。** 74 次 overflow 实际丢弃 370 帧对，
  P50 173.9 ms，但空档 39.5 s、ATE 4.07 m。两个参数同时改变，不能将效果单独归因于
  `latest`。保留默认 `drop_oldest`。
- **优化重点是检测与光流。** 首轮 control 检测约 45.6 ms、光流 39.0 ms，KLT 金字塔
  约 2.7 ms。缓存机制仍有复用价值，但此次没有旧缓存实现的 A/B，不能报出其独立加速比。


两次 P7 回放的观察端仅收到 1917/1922 帧，且 tracker 处理数加 overflow 丢弃数
也不等于观察端输入数；输入订阅独立，存在未归因的投递/退出差异。未将这些差额
伪装为 tracker overflow，亦未把输入缺失作为失败的已证实原因。

## 测试与复现修复

ARM 首次测试中 `SchurUpdate.T27WeakUpdateRescuesGateCollapse` 失败，其他测试通过。
夹具在 `Eigen::Vector3d(...)` 的多个参数里依次调用随机分布，C++ 未规定参数求值顺序，
导致 ARM 与 x86 实际生成不同的点云与扰动。将取样拆成显式顺序后，两平台输出一致：
普通门控拒绝 16 条轨迹，弱更新接受 14 条；不放宽断言、不改变生产算法。

修复后 ARM `colcon test-result`：**110 tests, 0 errors, 0 failures, 0 skipped**；
Python 脚本测试 **24/24**。本地 Python **24/24**、Schur update **24/24**。

全新隔离构建还暴露了 `--packages-select camera_models vins_common vins` 遗漏声明依赖：
colcon 需要 `loop_fusion/global_fusion` 的 `package.sh`。脚本改为 `--packages-up-to vins`，
包含声明依赖闭包但不启动相应节点，11 项 wrapper 测试通过。本轮二进制是在补构建这两个
依赖后完成的；未另做一次从零构建来测量改后脚本的完整构建耗时。

另补齐 `schur_starvation_hold` / `schur_hold_recovery_frames` 的启动后修改拒绝检查，
避免 ROS 参数服务报告成功而算法仍沿用启动值。本地与 ARM 均重编译节点并实际调用
参数服务，确认启动覆盖生效、热修改被拒绝、存储值不变，节点正常退出；这项接口修复
未替换 ARM 回放快照，复核快照 SHA-256 保持不变。

用户要求的 ds 子代理两次因服务余额不足退出，本轮复验和修复由主代理执行。

## 当时实现的边界

以下为本次 P7 实验快照的限制。后续已补充 IMU 回调独立协方差传播及里程计
协方差，见 [实现与中断验证](IMU_COVARIANCE_PROPAGATION.md)；不改变本报告的历史测量。

- P7 保留 live 状态与 15×15 协方差、删除 clone；保持期间使用名义过程噪声，
  对每个被处理的图像帧写回状态。已有 IMU 预测话题继续发布，但本轮延迟/空档
  测量订阅的是 `/vins_estimator/odometry`，没有独立验证 IMU 话题的每个发布周期。
  `processMeasurements()` 仍由 feature 队列驱动：完全没有图像时，Schur P 不会独立
  按每个 IMU 回调递推；`inputIMU()` 的 `fastPredictIMU()` 只更新预测名义状态。
  因此当前实现尚未完整覆盖“图像彻底中断仍实时递推协方差”的要求。
  当前两个 odometry 发布函数均未填入 EKF 协方差；这里的“P 自然增长”指内部状态，
  不代表下游已能从 ROS 消息读到增长后的不确定度。
- “恢复”仅检查连续两帧 camera-0 观测数量达到 12，不检查轨迹寿命、几何内点、
  创新或恢复后的实际视觉更新。因此日志中的 `tracker recovered`/`clone window rebuilt`
  只能证明数量门槛满足和窗口结构重建，**不能证明视觉定位已恢复**。
- 第一次 P7 失败轮在进入 hold 前位置范数已约 130 m，说明已有状态/初始化失真。
  保持策略无法自行纠正这种失真；反复按观测数量退出 hold 会重置饥饿时间基线。
  本轮不把所有误差都归因于 P7 开关，但这一反例已经否定“当前实现可以稳定救回”的结论。
- `failure_reboots=0` 只对应旧式日志关键字，不能代替受控重启统计，也不能证明未发散。
  本报告另数 `scheduling a controlled estimator restart`。
- 金字塔缓存属于所有配置共同使用的实现，本矩阵不能单独归因其性能收益。
  阶段数据只说明当前 KLT 金字塔占比低，下一步应优先解决检测/光流成本。

## 下一步优先级

1. **先解决 tracker 超预算**：以 20 Hz 的 50 ms 帧预算衡量。继续细分 GFTT 补点的
   金字塔/响应图/候选排序成本，评估只在缺点区域补点及候选复用；保留正反向光流与
   几何质量检查。单纯减小队列只能压缩等待，不能减少每帧计算量。
2. **把状态有效性纳入恢复判据**：先定位初始化/接管后的偏置、速度和首次漂移，
   恢复不能只数新检测到的点。需要轨迹寿命、几何内点、创新检验以及实际成功更新的
   连续性证据；不得用重新锚定饥饿时钟掩盖持续没有校正。
3. **补齐发布语义与专项测试**：分别验证图像中断时 IMU 预测输出与内部 P 的增长，
   再决定如何对下游暴露降级状态/协方差。对定位已经失真的状态，需要可靠的重定位或
   初始化策略，不能承诺靠纯 IMU 保持恢复。
4. **候选通过重复回放后再改默认值**：difficult 至少交错各 3 次，并同时检查
   easy/medium。所有结论须包含首尾覆盖、最长空档、ATE 和精确丢帧数；不能用
   `failure_reboots=0` 或较少节流告警当作成功依据。

## 复现

当前已有快照可直接执行；每轮使用新 label，保持串行，回放前预读对应 mcap。
以下例子仅切 P7，其余配置取相同快照的默认值：

```bash
ssh pinydrone0@192.168.5.60 \
  'cat ~/vicon_room2/bags/V2_03_difficult/V2_03_difficult.mcap > /dev/null'
bash scripts/benchmark_remote_arm.sh benchmark \
  --host pinydrone0@192.168.5.60 \
  --repo vins_p7_eval_20261027/workspace \
  --output vins_p7_eval_20261027/results \
  --bags vicon_room2/bags --snapshot-name p7_tracker \
  --datasets V2_03_difficult --label p7hold_recheck_1 \
  --parameter backend_type:=schur_ekf \
  --parameter schur_starvation_hold:=true
```

control 去掉最后一行；检测间隔试验改为 `tracker_detection_interval:=2`；
队列试验只改为 `tracking_queue_size:=1`；组合试验是检测间隔 2 加
`tracking_queue_policy:=latest`。这些节点级参数须经 `--parameter` 传入。
本地汇总脚本与相机 header 边界记录保存在 `output/arm_p7_20261027/evidence/`，
统计来源为每轮原始 `metrics.json`、`estimator.log`、`trajectory.csv` 和 manifest。
