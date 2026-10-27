# 图像中断时的 IMU 协方差传播

Schur EKF 激活后，`/vins_estimator/imu_propagate` 由有效 IMU 回调持续发布均值和协方差，无需等待图像。`/vins_estimator/odometry` 仍按处理后的图像发布，也填入 EKF 协方差。无需新增配置开关。

## 两条时间线

- `schur_state_` 保持图像时间线，包含 clone 与交叉协方差，由原后端推进并进行视觉更新。
- `latest_schur_state_` 只保存实时 15 维 IMU 边缘状态、FEJ 与 15×15 协方差。IMU 回调调用同一个 EKF 传播模型，同时推进均值和 P；没有图像时仍然继续。
- 每次视觉处理完成，从校正后的 EKF 复制 live marginal（不重置、不重膨胀、不复制 clone），再重放状态时间戳之后缓存的 IMU。旧样本和重复时间戳不会二次积分。下一次视觉更新仍在自己的历史时间线上进行，不对已经推进到未来的状态直接应用旧观测。
- 锁顺序为 `mProcess → mPropagate → mBuf`；回调的入队和预测与视觉重同步互斥。IMU 回调通过锁内 readiness 判断是否可发布，不再无锁读取图像线程的 `solver_flag`。重置/受控重启立即禁用实时输出，重新初始化后再启用。

当前 P3/P4 过程噪声倍率与图像滤波器保持一致：复制最新视觉处理阶段的倍率，在两次视觉同步之间保持不变；P7 已进入 hold 时使用倍率 1。此改动不从 IMU 回调触发 hold/restart，也不修改已有视觉恢复判据。

重复、倒序或非有限 IMU 被拒绝。连续 IMU 超过 1 秒空档时禁用该实时预测，避免用新时间戳发布未传播的旧协方差；之后需从有效视觉状态重新同步。相机中断而 IMU 连续时没有这一限制。

## ROS 消息约定

两类 Schur 里程计的 `header.frame_id="world"`，`child_frame_id="body"`。时间戳与用于生成消息的 EKF 状态一致：IMU 消息使用最新 IMU 时刻，图像消息使用图像时刻加 `td`。原 Ceres 输出路径保持原有行为。

EKF 误差顺序为 `[dtheta_body, dp_world, dv_world, dba, dbg]`。消息中的完整 6×6 数组按行保存，包含非零交叉项：

- pose 均值为世界系位置与 body→world 四元数。误差变换为 `[dp_world; R*dtheta_body]`，转换到 ROS 的固定世界轴旋转误差。
- twist 的线速度为 `Rᵀ*v_world`，角速度为 `gyro-bg`，都属于 body 系。线速度雅可比包含 `skew(Rᵀ*v)*dtheta + Rᵀ*dv`；角速度对 bias 的雅可比为 `-I`。
- 两者均以 `J*P_live*Jᵀ` 计算，保留姿态、位置、速度、bias 的相关项。角速度附加采样噪声方差 `gyr_noise²/Δt_imu`；Δt 来自真实采样间隔而非图像边界的部分积分步。把当前 gyro 噪声作为独立测量噪声处理，未增加其与滤波状态的相关模型。

下游若曾把 Schur 消息中标为 `body` 的线速度当作世界系速度，需要调整读取方式；本轮同步修正了均值坐标，避免 covariance 与数值属于不同坐标系。轨迹 CSV 仍保留世界系速度和图像时间戳，以兼容现有离线评测。

## ARM 成本与边界

每个 IMU 样本新增一次 15 维传播，计算不随 clone 数增长。只在视觉同步时复制 live marginal；没有每个 IMU 样本复制整个滑窗。消息转换只在发布 IMU 消息存在订阅者时执行。无图像期间仍保留后端所需的原始 IMU 缓冲，缓冲长度随中断时间增长；本轮没有引入截断历史或丢弃 IMU 的策略。

持续发布与 P 增长不意味着纯惯性轨迹保持准确，也不证明恢复后的视觉定位可靠。回放验证针对发布连续性、协方差、线程同步与恢复后输出，未据此宣称定位精度改善。

## 验证与复现

本地完整 10 个 CTest 程序（103 个测试用例）通过。新增回归覆盖连续 10 秒无图像、含 clone 状态复制为 live marginal、延迟视觉校正后的 IMU 尾部重放、重复/过期样本、并发回调与重同步、图像边界部分积分时的真实 gyro 采样间隔、重置及 IMU 长空档。消息测试覆盖旋转坐标、交叉项、gyro 噪声、对称与半正定性。

`verify_imu_blackout.py` 只启动 VINS 和 rosbag；IMU 直接输入估计器，摄像头经中继在指定时间段完全停止转发。默认对 easy 前 55 秒以 1× 回放，第 30–40 秒切断全部图像，最后恢复。检查丢弃图像数、中断期间无图像里程计时间戳、IMU 输出连续性和实际接收间隔、消息时间戳/坐标系、协方差有限/对称/半正定、位置不确定度增长，以及恢复后的图像里程计协方差。统计中断中心 8 秒以排除边界处已入队的图像处理。

```bash
source /opt/ros/jazzy/setup.bash
source /path/to/install/setup.bash
ROS_DOMAIN_ID=173 python3 scripts/verify_imu_blackout.py \
  --binary /path/to/vins_node --config /path/to/euroc_gftt.yaml \
  --bag ~/vicon_room2/bags/V2_01_easy --output /path/to/new_result_directory
```

本地最终回放：丢弃 200 张图像，中断中心 8 秒收到 1601 条 IMU 里程计，最大相邻传感器时间戳间隔 5.0004 ms，实际接收间隔 P95/最大值为 48.675/49.428 ms；位置协方差 trace 从 0.7518 增至 1.2044 m²。两种消息的协方差均有限、对称、半正定且非零，图像恢复后重新发布带协方差的图像里程计。中断期间估计器进程 CPU 约为单核的 1.58%，采样峰值 RSS 为 102.86 MiB。这是整个估计器进程在断图时的绝对占用，未做旧版本 CPU 对照，不能当作新增传播的净开销。

配置取自基线提交 `56ea726` 的 `src/vins_estimator/config/euroc/euroc_mono_imu_config.yaml`（`feature_detector=gftt`），独立副本搭配相邻的相机标定文件。命令行覆盖 Schur 后端、OpenCV 单线程、P7 开启、P5/P6 关闭及相机中继话题；仓库配置的 `backend_type` 默认值仍是 `ceres`，本功能在 Schur 激活后生效。用户工作区已有的 `gftt_pyramid` 修改未用于本次回放。


### ARM 最终回放（2026-10-27）

设备为 Orange Pi Zero 3（Cortex-A53、aarch64、4 GiB RAM），ROS 2 Jazzy / OpenCV 4.6。源码及测试位于独立的 `~/vins_p7_eval_20261027/workspace/src/VINS-Fusion`，实际运行 `~/vins_p7_eval_20261027/results/build/vins/vins_node`；未修改部署目录 `~/Code/Piny_drone_nav`。采用串行构建以控制编译内存。11 个改动源码/测试/脚本文件的 SHA-256 与本地一致。

| 最终版本回放指标 | 本地 | ARM |
| --- | ---: | ---: |
| 回放长度 / 完全断图长度 | 55 s / 10 s | 55 s / 10 s |
| 丢弃的图像数 | 200 | 200 |
| 中断中心 8 s 的 IMU 里程计条数 | 1601 | 1601 |
| 最大相邻消息时间戳间隔 | 5.0004 ms | 5.0004 ms |
| 实际接收间隔 P95 / 最大值 | 48.675 / 49.428 ms | 25.737 / 38.506 ms |
| 位置协方差 trace 起点 → 终点（m²） | 0.7518 → 1.2044 | 0.7521 → 1.2428 |
| 被检查协方差的最小特征值 | 3.78e-7 | 4.51e-7 |
| 断图期间 CPU（单核百分比） | 1.58% | 12.31% |
| 断图期间采样峰值 RSS | 102.86 MiB | 98.27 MiB |

ARM 的估计器恢复、里程计协方差、EKF 状态三个测试程序共 41 个用例通过，包括最后新增的 gyro 采样间隔检查。本地完整 10 个程序共 103 个用例通过。两端均通过脚本的全部 10 项检查。恢复图像后，图像里程计重新输出非零协方差；最终 ARM 回放中未出现受控重启或 hold 进入日志。CPU 为中断中心窗口内进程 CPU 时间增量除以墙钟时间；RSS 每约 0.5 秒采样。这些数值不包含 rosbag 和 Python 检查进程，也没有减去旧实现的基线开销。

ARM 的 tracker 仍累计溢出 340 次、丢弃 340 对待处理图像；独立 IMU 传播未解决检测/光流吞吐问题。采样间隔修正之前的初次 ARM 回放同样通过功能检查，但出现多次 hold，位置协方差 trace 在断图窗口从 328.51 增至 1578.58 m²，远大于最终回放。该差异没有做单因素定位，不能归因于仅影响消息角速度噪声的修正，也不能用两次运行宣称定位稳定或精度改善。两轮结果均保留供追溯。

最终本地和 ARM 的结果、命令、二进制/配置/源码哈希见 [机器可读结果](IMU_COVARIANCE_RESULTS.json)。本地完整日志与消息样本位于 `output/imu_covariance_20261027/`，ARM 原始结果位于 `~/vins_p7_eval_20261027/results/imu_covariance/`；原始大文件不纳入 Git。
