# Schur 校准状态、独立初始化与 machine_hall 验证（2026-11-15）

本次实现由用户指定的 DeepSeek ds 子代理承担状态更新、独立初始化、远程 ARM 评估三个任务，主代理负责审查、集成、回归与证据核查。参考本机 `/home/sprout_stars/rmcode/open_vins`，没有复制它的 Ceres 动态初始化依赖。

## 实现

- 可选状态布局 `[IMU 15][clones 6N][camera extrinsics 6C][shared td 1]`；同步双目共用时偏。固定标定不增加协方差列。
- 外参/td 进入视觉雅可比、卡方门控、特征点 Schur 消元、平方根更新和 Joseph 协方差。FEJ 与 nominal 分离；解析雅可比对测量预测作有限差分验证。
- 特征历史保留归一化图像速度与采样时偏；三角化、最终 RMS 和发布质量使用同一时偏补偿。ZUPT、实时预测器、克隆硬删除和软重启适配校准。
- `schur_initialization: standalone` 默认启用：验证静止窗口或双目 metric PnP + IMU 重力/速度对齐；不调用 Ceres，失败也不静默回退。`legacy_ceres` 可显式恢复原始启动路径。
- strict PnP 返回成功标志并检查重投影，窗口保存每帧视觉来源有效标志，失败帧需滑出后才允许动态 seed；静止 seed 不依赖 PnP。
- Schur IMU 样本在图像+td 边界插值，传播 dt 精确到当前边界；单个未来样本不会导致越界传播，缺失边界拒绝更新。
- td 物理范围默认 ±0.10 s；不合法校准提议整次测量均值/协方差回滚，保留本帧克隆和观测索引。

## 独立性的边界

可执行文件仍链接 Ceres，因为项目保留原 Ceres 后端和初始化对照。独立性指 Schur 的启动、传播、测量和恢复运行路径。单目运动初始化尚未实现；单目需要可验证的静止窗口，或显式选 `legacy_ceres`。动态双目初始化需要已知可用的双目几何，未知外参配置等待旋转标定完成。外参/td 的观测能力仍依赖运动激励。

## ARM 消耗优化

对照 OpenVINS `UpdaterHelper.cpp`、`UpdaterMSCKF.cpp`、`Propagator.cpp`、`StateHelper.cpp`、`TrackKLT.cpp` 与本项目已有实现后：Givens QR、观测预算、金字塔缓存、OpenCV 单线程和批量 IMU 协方差在原仓库已经存在，本次不把这些旧机制算作新实现。

本次新增/适配的计算节省：

1. standalone 激活后停止计算供 Ceres 使用的 IMU 预积分和对应样本缓冲；保留 EKF 必需的传播。
2. 校准状态克隆增广用 O(n²) 分块拷贝，不用稠密选择矩阵乘积。
3. 开启校准时只分解 clone+calibration 信息块，跳过 15 维未直接观测 live IMU 零列；条件数必要时用无量纲列归一化检查，更新仍使用原始信息矩阵。
4. QR/batch/观测预算只在相同输入上测量后选用；不能把降低输出覆盖率算作性能收益。

## 测试与评估方法

本地 Release 构建，14 个 CTest 目标、203 个 gtest 用例通过，87 个 Python 脚本测试通过。测试覆盖 no-Ceres 静止初始化、失败 PnP 窗口、未来 IMU 边界插值、解析 Jacobian、含交叉项的卡方参考、实际校准修正后残差下降、克隆增删与预测器协方差、校准点云和 ZUPT。

EuRoC machine_hall，双目 + IMU，完整 1× 回放，Leica `/leica/position` 作为评估 truth，应用 `mav0/leica0/sensor.yaml` 的棱镜→body 杠杆臂；按传感器时间对齐，刚体 SE(3) 位置配准，不拟合尺度或时差。GT 不参与初始化。

CPU 时间仅统计 estimator，100% CPU 表示一个核。图像计数为 observer 实际收到的话题消息，不是 tracker 全处理证明；同时审查 overflow、输出数量和间隔。单次回放的时间/RSS 存在调度和热状态波动，不代表跨设备通用收益。

## 本地初步全序列结果

源 HEAD `ccf2e69`；变更构建和原二进制 SHA、manifest、轨迹、日志位于 `output/schur_complete_20261115/`。下表 fixed 使用最终 release 复测；online 使用功能相同的前一快照，最后改动仅修正 seed 帧的空更新门控与新增日志。此表均使用原通用相机内参，非 dataset 官方 Machine Hall 内参；ARM 组使用官方内参，不能跨表归因。

| 数据 | 版本 | ATE RMSE m | CPU s | 输出数 | 最大空洞 s | 重置 |
|---|---|---:|---:|---:|---:|---:|
| MH01 | 原版 fixed | 0.7795 | 36.26 | 1831 | 0.70 | 0 |
| MH01 | standalone release fixed | 0.7788 | 27.60 | 1831 | 0.70 | 0 |
| MH01 | 最终校准功能版，外参+td+QR | 0.1677 | 28.02 | 1831 | 0.7002 | 0 |
| MH05 | 原版 fixed | 2.3327 | 18.90 | 1126 | 2.80 | 0 |
| MH05 | standalone release fixed | 2.0418 | 16.02 | 1122 | 2.80 | 0 |
| MH05 | 最终校准功能版，外参+td+QR | 0.7337 | 17.24 | 1122 | 2.8001 | 0 |

外参+td+QR 组与 fixed 同时改变三项，不能把精度改善单独归因于某一项。开启时偏后里程计时间戳是 image+td，旧脚本仅按完全相等时间戳关联延迟：MH01 1831 个输出中有 1829 个无法匹配，MH05 有 1114/1122 个无法匹配，因此这两组延迟 P50/P95/P99 **不可用**，不得引用为改善证据。固定标定组延迟匹配正常。

完整输出仍有 0.7/2.8 s 的原有长间隔；无重启并不等同于每帧输出。

## 远程 ARM

由 ds 在 `pinydrone@192.168.5.62` 的独立目录 `/home/pinydrone/vins_arm_perf_20261115` 原生 Release 构建，保持原部署二进制。设备为 Orange Pi Zero 4，6×A55 + 2×A76、3.8 GiB RAM、ROS 2 Jazzy。远程测试固定 A76 6/7、OpenCV 1 线程。早期 mono-only 压缩 MH01 及错误输入模式结果不纳入双目成对对比；原始审计见 `ARM_PERF_BASELINE_MH_20261115.md`，最终双目结果另附。

固定标定同机回放，MH01 estimator CPU 时间减少 23.9%，P95 从 20.58 ms 降至 16.66 ms；MH05 CPU 时间减少 15.2%，P95 从 20.29 ms 降至 18.57 ms。MH05 输出比基线少 4 个，但相同总覆盖 111.7 s、相同最大空洞 2.8 s；不能声称严格逐帧等价。

新增 `machine_hall_stereo_imu_schur_online_config.yaml` 使用数据 `mav0/cam0/sensor.yaml` 与 `mav0/cam1/sensor.yaml` 的官方内参；原 `cam0_pinhole.yaml` / `cam1_pinhole.yaml` 不被覆盖。Machine Hall cam0 为 fx=458.654、fy=457.296、cx=367.215、cy=248.375，与通用配置 fx=461.1586、fy=459.7529、cx=362.6593 不同。

## 五条序列的本地独立启动覆盖

以下均为在线外参+td+QR，使用原通用相机内参的同一配置族；MH01/MH02 运动初始化，MH03/MH04/MH05 静止初始化。所有 run 无重置，日志确认不调用 Ceres。该表是运行覆盖和精度诊断，不是官方 EuRoC 排行榜结果。

| 数据 | ATE RMSE m | 输出数 | 轨迹跨度 s | 最大输出间隔 s | CPU s |
|---|---:|---:|---:|---:|---:|
| MH_01_easy | 0.1677 | 1831 | 182.50 | 0.7002 | 28.02 |
| MH_05_difficult | 0.7337 | 1122 | 111.70 | 2.8001 | 17.24 |
| MH_02_easy | 0.3980 | 1510 | 150.30 | 1.3999 | 24.52 |
| MH_03_medium | 0.7533 | 1337 | 132.40 | 3.1996 | 20.28 |
| MH_04_difficult | 0.6159 | 1003 | 99.70 | 1.8000 | 14.82 |

MH04 observer 收到 cam0=2033、cam1=2032 条；其他四序列两路计数一致。图像计数与输出数定义不同。所有在线组的延迟精确匹配覆盖都很低，故仅列 CPU/RSS/轨迹数据。

## ARM 同输入成对结果（官方 Machine Hall 内参）

以下两组使用同一 clean HEAD baseline binary 和同一候选 binary、同一 QR、batch=false、特征 120、窗口 5、官方 Machine Hall 内参；候选为 standalone，baseline 为原 Ceres 初始化后激活 Schur。测试未订阅点云，故不包括可视化点云重建成本。

| 数据 | 版本 | ATE m | CPU s | RSS MiB | P95 ms | 输出数 | 最大间隔 s | 重置 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MH01 | 原版 | 0.16935 | 132.53 | 106.34 | 47.85 | 1831 | 0.70 | 0 |
| MH01 | standalone fixed | 0.16913 | 132.19 | 100.95 | 47.71 | 1831 | 0.70 | 0 |
| MH05 | 原版 | 0.64241 | 82.79 | 97.50 | 50.33 | 1126 | 2.80 | 0 |
| MH05 | standalone fixed | 0.64407 | 82.62 | 95.75 | 49.62 | 1122 | 2.80 | 0 |

CPU 差异仅约 0.2%–0.3%，不足以声明明显降耗；单次 P95 改善也很小。收益首先是运行初始化独立与校准状态完整，ARM 热路径仍主要由双目跟踪占用。候选 MH01 的 RSS 减少约 5.4 MiB、MH05 减少约 1.75 MiB。两路图像均完整收到，MH01 3682×2、MH05 2273×2。后续 batch true 和 online 组用于评估额外开销/配置选择，不能把本地 x86 的 CPU 降幅宣传为 ARM 收益。

### ARM 批量传播消融

同一候选 MH01 fixed + QR，仅改 `schur_batch_imu_covariance`：false CPU=132.19 s，true CPU=131.91 s；ATE 分别 0.16912707819 / 0.16912707827 m，均输出 1831 个位姿，无重置。P95 分别 47.71 / 47.42 ms，RSS 100.95 / 100.82 MiB。数值结果一致，性能变化约 0.2%，属于单次运行波动量级，**保留 batch=false 默认**。

### 新配置使用

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run vins vins_node --ros-args -p config_file:=$PWD/src/vins_estimator/config/euroc/machine_hall_stereo_imu_schur_online_config.yaml -p opencv_threads:=1
```

新通用/ Machine Hall 在线配置不默认绑定设备 CPU；ARM A76 6/7 绑定由本次 benchmark 的 `--cpu-affinity 6,7` 完成。设备部署前应明确实际可用核心；原 balanced ARM 配置保持现有 launch affinity。在线校准配置不会覆盖原固定外参配置。

### 官方内参的本地 online 复测

专用 Machine Hall 配置在 x86 full replay：MH01 ATE=0.20858 m、CPU=26.68 s、1831 输出；MH05 ATE=0.77942 m、CPU=16.78 s、1122 输出，均无重置。在线估计不保证在已经良好标定的数据上提升精度，仍应按传感器标定质量与运动激励选择；没有把它强制开启到原 balanced fixed 配置中。在线组的图像/里程计延迟统计仍因时偏关联不足而不可用。

### ARM 在线校准结果与建议

MH01 同一候选、同一官方内参、同一 QR/batch=false，开启所有相机外参和共享 td：ATE=0.18820 m（fixed=0.16913 m），CPU=135.13 s（fixed=132.19 s，增加约 2.2%），RSS=101.51 MiB（fixed=100.95 MiB），仍输出 1831 个位姿、无重置、最大空洞 0.7004 s。在线组 `latency_misses=1828/1831`，只有 3 个精确匹配，因此 P95 不可用。状态 58×58，最终外参文件和变化的里程计时间戳确认估计在更新（本组 stage_timing=0，不输出周期校准日志）。

**建议**：已有可靠标定时继续使用 fixed 配置；需要在线校准时显式使用新增 online 配置，并按传感器不确定性设置先验。在线校准完整且可用，但这次数据上没有改善已经准确标定的 ARM 精度。batch 默认仍保持 false；不推荐为这约 0.2% 的单次变化增加配置复杂度。当前证据支持更低 RSS 和独立初始化，不能声称显著降低 ARM CPU。

ARM 二进制 SHA-256 为 `cd9d367d46d4b4c8e076698bfa65fae923136cc5e1eeb4404fc56688a70f95bf`；estimator.cpp SHA-256 为 `e75941ce5ad571211fbd293393e6109741ce4ad90712df40b983f415abe96d5f`，与主工作区一致。证据根目录 `output/arm_perf_20261115/`，含 baseline/candidate manifests、完整轨迹、资源、日志、来源和相机标定对照。旧 mono-only/错误输入模式结果仅作问题审计，不用于此表比较。

ARM 完整逐组来源、配置哈希和收尾审计另见 [ARM_FINAL_RESULTS_20261115.md](ARM_FINAL_RESULTS_20261115.md)，机器可读汇总见 `output/arm_perf_20261115/final_arm_results_20261115.json`。在线 MH01 observer 两路各收到 3681/3682 条图像消息（fixed/batch 为 3682/3682）；不将少 1 条接收消息武断归因到算法，也不据此声称全帧实时。
