# 双目 ARM 优化报告：研究与本地结果（2026-11-12）

**当前设备的平衡推荐：双目Schur EKF + `gftt_pyramid` + 前端120特征 + 后端最多120 tracks/W=5 + OpenCV单线程 + A76核心6/7。** ARM三序列均无重启和跟踪溢出；相对修复后的200特征，CPU降低7%–10%、P95降低16%–54%、RSS降低4%–8%，代价是difficult ATE由0.346 m升至0.477 m。medium三次运行轨迹完全一致，P95稳定在53.39–53.53 ms。需要更好difficult精度时保留200特征；它在easy/medium会积压和丢对。

首先修复了此前“前端处理右目、Schur只消费左目”的数据链路。原用户YAML逐字保留，独立配置和完整结果在下文；旧only-left后端在ARM medium出现10.46 m误差及2次重启。本文误差使用未校正杆臂的Vicon标记体位置参考，适合同口径筛选，不作为官方EuRoC body真值排名。

> 本文包含源码研究、硬件核查和已完成的本地回放结果。优化建议中的预期收益仍需重复验证；ARM 最终结果等待 clean run 完成，不从本地结果外推。

## 1. 版本、配置与证据边界

- VINS-Fusion 基础提交为 `b026fdb0dd6954a77ed7f9e10db7a1fac6fe0f77`；本章节描述其上已完成修复的工作树，源码引用对应当前文件。仅用 HEAD 无法唯一标识这份实现，复现应同时保存源码差异、实际 ELF SHA256 和配置全文。
- OpenVINS 对比版本为本地 `/home/sprout_stars/rmcode/open_vins` 的 `69488123ed9362dd44b6f28e7f4680abbff1442b`，属于实现研究，不包含在本机或 ARM 上运行 OpenVINS 的性能对照。
- 产品 EuRoC 配置为双目+IMU、`gftt_pyramid`、`max_cnt=200`、Schur W=5、最多 120 tracks、后端 10 Hz；产品 YAML 的 `estimate_td=1`，主对照派生配置固定为 `estimate_td=0`，另完成 difficult 的 `estimate_td=1` 对照（见§8）。固定td是采用已知EuRoC标定的实验选择，不保证适用于其他相机或未标定时间偏移。比较必须采用相同配置口径。[配置来源](/home/sprout_stars/rmcode/VINS-Fusion/src/vins_estimator/config/euroc/euroc_stereo_imu_config.yaml:31)
- 本轮 local baseline 是**真实 stereo 输入、双目前端及初始化，但 Schur 阶段只消费左目**的混合流程。它能衡量修复前整条流程的行为，不能简称为纯 mono，也不能当作完整 stereo Schur 后端。其 manifest 确认播放 IMU、cam0、cam1 三个 topic。[输入证据](/home/sprout_stars/rmcode/VINS-Fusion/output/stereo_arm_20261112/local_results/baseline_stereo/manifest.json)
- 历史 `.62` 报告和 `.60` 4×A53 报告中的回放结果为 mono，既不能与上述混合 baseline 合并，也不能代替新 ARM stereo 结果。本地 x86 与 ARM 数据分表；远端新证据根目录为 `/home/pinydrone/vins_stereo_20261112/`。

## 2. 已完成的产品与基准能力

三重 stereo 断点已经修复，不再列为后续待实施优化：

| 已完成内容 | 当前行为与源码证据 |
|---|---|
| 双目观测收集 | 按已配置 camera 收集同一 feature 的左右观测，共享一个 body clone；hold 恢复门槛仍按独立 feature 数计算。[estimator.cpp](/home/sprout_stars/rmcode/VINS-Fusion/src/vins_estimator/src/estimator/estimator.cpp:1447) |
| 历史去重 | 按 `(frame_index, camera_id)` 去重并排序，保留同帧左右观测，正常有序追加使用快速路径。[feature_history.h](/home/sprout_stars/rmcode/VINS-Fusion/src/vins_estimator/src/backend/feature_history.h:74) |
| 同帧双目三角化 | 同一 clone、不同 camera 的外参基线可用于初始化；相同视图仍拒绝，后续保留多视图精化和几何门控。[schur_update.cpp](/home/sprout_stars/rmcode/VINS-Fusion/src/vins_estimator/src/backend/schur_update.cpp:286) |
| Loop 元数据 | keyframe 的二维字段明确取 camera0，避免左右共享 feature id 时把右目坐标写进单相机 loop 消息。[schur_visualization.cpp](/home/sprout_stars/rmcode/VINS-Fusion/src/vins_estimator/src/backend/schur_visualization.cpp:109) |
| 前端复用 | KLT/RANSAC/检测输出复用 scratch；上一帧 lifted points 保留双精度以避免重复投影及 float 往返改变 RANSAC 输入。[feature_tracker.h](/home/sprout_stars/rmcode/VINS-Fusion/src/vins_estimator/src/featureTracker/feature_tracker.h:132)、[RANSAC](/home/sprout_stars/rmcode/VINS-Fusion/src/vins_estimator/src/featureTracker/feature_tracker.cpp:737) |
| Benchmark | 从有效配置和覆盖参数解析 stereo/custom topics，分别计数左右消息；CPU affinity 通过 `taskset` 前缀施加于 estimator，manifest 哈希仍取实际 ELF，记录 mask、topics 和计数语义。[主题解析](/home/sprout_stars/rmcode/VINS-Fusion/scripts/benchmark_euroc.py:112)、[命令构造](/home/sprout_stars/rmcode/VINS-Fusion/scripts/benchmark_euroc.py:194)、[manifest](/home/sprout_stars/rmcode/VINS-Fusion/scripts/benchmark_euroc.py:623) |

降落静止检测继续只看 camera0 像素是有意的：它在同一图像平面匹配光流，不能混入右目像素；这不代表视觉更新仍为 only-left。[静止检测输入](/home/sprout_stars/rmcode/VINS-Fusion/src/vins_estimator/src/estimator/estimator.cpp:1483)

现有 `SHOW_TRACK && trackImageRequested()` 订阅门控、bounded path、有界队列、`cv_bridge::toCvShare`、左右目金字塔及 track 级 scratch 复用均已具备，不能重复计为本轮新增收益。[绘图门控](/home/sprout_stars/rmcode/VINS-Fusion/src/vins_estimator/src/estimator/estimator.cpp:267)、[路径上限](/home/sprout_stars/rmcode/VINS-Fusion/src/vins_estimator/src/utility/visualization.cpp:81)、[零拷贝](/home/sprout_stars/rmcode/VINS-Fusion/src/vins_estimator/src/rosNodeTest.cpp:444)

回归证据：**12/12 个 CTest 程序通过**；对 12 份 gtest XML 根节点求和为 **149 个用例，0 failures、0 errors、0 disabled**；Python **73 项通过**。这是功能回归证据，不等于完整序列精度或 ARM 实时性验收。

- [最终 CTest 日志](/home/sprout_stars/rmcode/VINS-Fusion/output/stereo_arm_20261112/evidence/local_candidate_final_ctest.log)
- [gtest XML 目录](/home/sprout_stars/rmcode/VINS-Fusion/output/stereo_arm_20261112/local_candidate/build/vins/test_results/vins)
- [Python 日志](/home/sprout_stars/rmcode/VINS-Fusion/output/stereo_arm_20261112/evidence/python_tests.log)

## 3. OpenVINS 后端对比

| 维度 | 当前 VINS Schur EKF | OpenVINS 对比版本 | 可借鉴点与限制 |
|---|---|---|---|
| 状态 | 15 维活动 IMU + 每 clone 6 维，无常驻 landmark 状态 | IMU、clones、可选 SLAM/ARUCO 和标定状态 | 保持当前紧凑状态；完整移植 SLAM 状态会增加协方差及更新成本 |
| 传播 | 固定单步 IMU 输入的 RK4 均值；三阶截断 `Phi`、离散 `Qd`；逐样本更新 P 的活动行列 | RK4 使用区间两端输入；解析 F/G，累积 `Phi/Qd` 后统一传播协方差 | 均值输入模型和协方差离散化并不完全相同，批处理与解析传播应拆开验证 |
| FEJ | 传播后前推 live FEJ；clone 在增广时冻结；残差当前值、雅可比 frozen clone 值 | 启用 FEJ，传播和观测保留相应线性化点 | 已有一致性机制；不应以省计算为由取消 FEJ 或刷新 clone FEJ |
| 特征消元 | 每 track 对 Hf 做 full-U JacobiSVD 以门控；3×3 点块 Schur 消元；全局信息矩阵特征分解取平方根 | Givens QR 消去 Hf；堆叠后再次 QR 压缩测量 | QR 是可研究的替代数值路线，不能仅凭算法名称推断 ARM 更快 |
| 协方差更新 | 创新 LDLT + Joseph，检查有限性、正定创新和 PSD | `P -= K M^T`，维护上三角并检查对角 | OpenVINS 本版本不是 Joseph；不宜为速度直接移除当前数值保护 |
| Clone 管理 | 增广→更新→删除最旧行列；W=5，更新时最多暂存 6 clones | 同类增广/更新/删除流程；EuRoC 配置保留 11 clones | 双目共享同一时刻的 body clone，无需按 camera 复制状态 |
| 特征预算 | 左目检测上限 200，右目匹配；Schur 最多 120 tracks，按观测长度优先 | EuRoC 配置 MSCKF 40、SLAM 50、SLAM 单批25；`num_pts=200` 初始化后按相机数拆分 | track 数、观测数、常驻 landmark 数是不同预算，不能直接对照 120 与 40 |

具体依据：

- 当前传播/FEJ：[ekf_state.cpp:152](/home/sprout_stars/rmcode/VINS-Fusion/src/vins_estimator/src/backend/ekf_state.cpp:152)；clone 增广/删除：[241](/home/sprout_stars/rmcode/VINS-Fusion/src/vins_estimator/src/backend/ekf_state.cpp:241)、[287](/home/sprout_stars/rmcode/VINS-Fusion/src/vins_estimator/src/backend/ekf_state.cpp:287)。
- 当前 SVD、全局 eigen、Joseph：[schur_update.cpp:531](/home/sprout_stars/rmcode/VINS-Fusion/src/vins_estimator/src/backend/schur_update.cpp:531)、[783](/home/sprout_stars/rmcode/VINS-Fusion/src/vins_estimator/src/backend/schur_update.cpp:783)、[857](/home/sprout_stars/rmcode/VINS-Fusion/src/vins_estimator/src/backend/schur_update.cpp:857)。
- OpenVINS 累积传播、FEJ：[Propagator.cpp:71](/home/sprout_stars/rmcode/open_vins/ov_msckf/src/state/Propagator.cpp:71)、[473](/home/sprout_stars/rmcode/open_vins/ov_msckf/src/state/Propagator.cpp:473)、[728](/home/sprout_stars/rmcode/open_vins/ov_msckf/src/state/Propagator.cpp:728)。
- OpenVINS 两级 Givens QR：[UpdaterHelper.cpp:426](/home/sprout_stars/rmcode/open_vins/ov_msckf/src/update/UpdaterHelper.cpp:426)、[456](/home/sprout_stars/rmcode/open_vins/ov_msckf/src/update/UpdaterHelper.cpp:456)；协方差更新：[StateHelper.cpp:166](/home/sprout_stars/rmcode/open_vins/ov_msckf/src/state/StateHelper.cpp:166)。
- OpenVINS 预算：[EuRoC 配置:16](/home/sprout_stars/rmcode/open_vins/config/euroc_mav/estimator_config.yaml:16)、[初始化后拆分:124](/home/sprout_stars/rmcode/open_vins/ov_msckf/src/core/VioManagerHelper.cpp:124)、[MSCKF 截断:523](/home/sprout_stars/rmcode/open_vins/ov_msckf/src/core/VioManager.cpp:523)。配置的 `num_pts` 注释写“per camera”，但实际初始化后执行除以相机数，应以代码为准。

### 内存与计算量级

W=5 时更新前状态维度为 `15+6×6=51`，单份 double P 仅 `51²×8=20,808 bytes≈20.3 KiB`；裁剪后 45×45 约15.8 KiB。复用 P 或帧级 `A/b/L/S/K/P_new` 可以减少分配和抖动，**不足以单独承诺明显降低进程 RSS**。RSS 更应从图像/导数金字塔、ROS 队列、Ceres 初始化、发布缓存和分配器高水位调查。[帧级 A/b](/home/sprout_stars/rmcode/VINS-Fusion/src/vins_estimator/src/backend/schur_update.cpp:654)

120 tracks 并非120或240条历史观测：6 clones、每帧左右均成功时，每 track 最多12条，合计约1,440条二维观测、2,880个残差标量。完整率决定实际数量；full-U SVD 的每 track 临时矩阵也随历史长度增长。因此增加窗口与增加特征不能只看 P 的尺寸。

图像队列的理论payload也应与RSS分开：`752×480×1=360,960 bytes=352.5 KiB`（mono8，不含stride填充、消息头与分配开销）；左右各100张的同步队列payload容量上界约 **68.85 MiB**，左右各6张约 **4.13 MiB**。这是假设各槽位持有独立图像且队列全满的容量估算，不是实际占用或RSS差值。消息共享、真实队列水位、跟踪队列和DDS保留会改变驻留量；DDS image QoS仍是`keep_last(100)`，因此不能据此宣称“节省约65 MB”。[同步队列与DDS深度](/home/sprout_stars/rmcode/VINS-Fusion/src/vins_estimator/src/rosNodeTest.cpp:196)


## 4. `.62` 硬件实际能力与 API 限制

2026-11-12 通过 `ssh -o BatchMode=yes pinydrone@192.168.5.62` 只读核查。设备主机名为 `orangepizero4`；设备树报告 `sun60iw2`、`xunlong,orangepi-4-pro`，型号命名差异不影响以下 CPU/驱动枚举事实。

| 项 | 读取结果与边界 |
|---|---|
| CPU/系统 | aarch64，Linux `6.6.98-vendor-sun60iw2`，Armbian 26.11 noble；6×A55最高1.794 GHz、2×A76最高2.002 GHz |
| 调度能力 | cpu0–5 capacity=385，cpu6–7=1024；governor=`schedutil`；这是异构8核，不能按8个等性能核估计 |
| 内存 | MemTotal=3,930,724 kB，约3.75 GiB；zram swap=1,965,360 kB；GPU共享主存，枚举的全局显存不是额外容量 |
| OpenCV | 4.6.0，Release，NEON/FP16 baseline，TBB 2021.11，编译启用 OpenCL |
| OpenCL | PowerVR B-Series BXM-4-64，OpenCL 3.0，报告 OpenCL C版本1.2；1 CU，未报告FP64支持；探测时400 MHz，devfreq可到1008 MHz |
| GPU 内存/API | host/device unified memory；报告单次最大分配约959.6 MiB；Vulkan设备API 1.3.277，驱动24.2@6603887 |
| NPU | `/dev/vipcore`，驱动1.13.0，492/852 MHz；存在 `libVIPhal.so`、`libNBGlinker.so`，标准模型编译/运行工具链未验证 |
| 温度 | 采样时CPU约68.6–69.3 °C；这是瞬时读数，不能证明持续双目负载的热稳定性 |

可复现的只读查询：

```bash
ssh -o BatchMode=yes pinydrone@192.168.5.62 'uname -srm; lscpu; head -5 /proc/meminfo; cat /sys/devices/system/cpu/cpu*/cpu_capacity'
ssh -o BatchMode=yes pinydrone@192.168.5.62 'clinfo -l; vulkaninfo --summary; opencv_version --verbose'
ssh -o BatchMode=yes pinydrone@192.168.5.62 'ls -l /dev/dri /dev/vipcore; cat /sys/module/vipcore/version; cat /sys/class/devfreq/*/available_frequencies'
```

**API可见不等于VINS已加速。** 当前图像路径使用 `cv::Mat` 和 CPU LK，几何后端使用 Eigen double；改成 UMat 仍可能逐算子回退或支付同步/转换成本。统一内存不保证零拷贝，也不保证 CPU/GPU 同时访问无同步。1 CU及缺少FP64支持使双精度 Schur 后端迁移缺乏直接收益依据；Vulkan需要另写计算和数据管理路径。NEON/FP16 baseline也不代表当前估计器使用FP16。

历史组合 UMat smoke 曾退出139，原因未定位，不能据此认定所有OpenCL不可用；本轮未重跑计算负载。[历史证据与边界](/home/sprout_stars/rmcode/VINS-Fusion/docs/ARM_BALANCED_OPTIMIZATION_20261110.md:84) NPU节点和动态库尚不能证明有适用于KLT/QR的算子或可部署模型，暂不作为当前几何后端的加速选项。

## 5. 剩余优化：按预期收益与实施成本排序

| 优先级 | 建议 | 预期价值/成本 | 验证方式与风险 |
|---|---|---|---|
| P1 | 细分前端/三角化/门控SVD/全局eigen/Joseph/传播计时，记录每相机进入和使用的观测数 | 定位真实瓶颈，成本低 | 采用低频汇总；测量自身开销；不能用总track数证明右目使用量 |
| P1 | 固定前端200，先扫后端80/100/120 tracks，再对优选点扫前端160/200 | 直接控制双目计算量，成本低 | 同时看历史观测量、几何覆盖、门控通过率和饥饿；按长度优先可能偏向局部区域 |
| P1 | A76 affinity与OpenCV线程做独立2×2对照 | mono先验显示值得复测，配置成本低 | 四组为不绑核/1线程、不绑核/2线程、绑6,7/1线程、绑6,7/2线程；四组互异 |
| P2 | 复用帧级更新工作区与clone临时P | 降分配/尾延迟，成本较低；RSS收益预计有限 | 统计malloc次数和P95/P99；保持赋值次序、alias安全与并发隔离 |
| P2 | 替换每track full-U SVD为等价零空间变换，再独立评估全局QR压缩 | stereo历史观测较多时可能有价值，成本中高 | 分两步改；核对秩、门控、dx和P，而非仅checksum；不能直接移除rank/PSD保护 |
| P2 | 累积每IMU步Phi/Qd，在图像边界统一传播完整P | 可能减少clone交叉块的重复计算，成本中高 | 先保持相同单步动力学；不得同时更换RK4输入模型或噪声离散化 |
| P2 | 在优选预算下比较W=5/7/9 | 调整时间基线与观测成本，配置成本低、精度风险中 | 避免照搬OpenVINS的11；同时记录窗口时间跨度、历史观测量、rank/condition |
| P3 | GPU图像算子原型；NPU留作独立研究 | 收益未知、成本高 | 仅在CPU profile证明热点后投入；计入转换/传输/同步/回退和精度成本 |

若新增“总观测预算”，应按完整track选择或按明确规则重建track，避免任意截断左右对或移除最佳基线；在没有数据前不固定200/240为总历史观测上限。后端10/20 Hz对照仅在已测吞吐允许时进行，保持输入20 Hz和前端全速跟踪的口径一致。

## 6. 后续数值与精度风险

| 风险 | 必须保留或检查的约束 |
|---|---|
| FEJ/可观性 | clone FEJ在增广时冻结；双目提供尺度基线，但整体视觉姿态gauge仍在；同帧双目可三角化不等于能独立约束全部body位姿 |
| Schur与QR等价性 | 正规方程会平方条件数；当前eigen阈值与QR奇异值/对角阈值不能直接照抄；比较的是投影后信息、门控和后验，而非Q矩阵本身 |
| 门控自由度 | 按`2N-rank(Hf)`解释chi2；N是实际左右历史观测数，不能用track数替代；相关匹配误差会影响独立噪声假设 |
| 三角化质量 | 首末视图有基线不代表基线最佳；远点、弱视差、错误左右匹配仍需深度、RMS、点块rank/condition保护 |
| 标定与同步 | 相机外参方向、左右时间差、固定td与在线td配置必须一致；错外参/错时间可能被误判为计算预算不足 |
| 传播批处理 | 使用`Phi_total=Phi_i Phi_total`、`Q_total=Phi_i Q_total Phi_iᵀ+Q_i`；保留FEJ、图像边界插值、视觉校正后的重锚和IMU-rate发布语义 |
| 故障连续性 | hold/ZUPT门槛按独立feature或单相机像素定义，不因双目观测翻倍而降低；统计restart request、reset入口和输出gap，不能用单一failure计数代替 |
| Loop接口 | 双目参与三角化时，loop的二维元数据仍须与camera0图像匹配；不能把cloud fallback坐标误用于keyframe二维观测 |

## 7. Pareto 验证方案

1. **分离对照语义。** 混合baseline与修复后的stereo Schur比较整条流程收益；保持双目输入和派生配置相同。另设前端也为单目的控制组时，应单独命名；Ceres stereo是后端参照，不能混入同一微优化结论。
2. **先固定预算，再调参数。** 三条EuRoC V2序列先做同预算对照；再选择少量track预算、W和调度候选，避免全维度网格。局部替换数值路径需独立ablation，不能把双目修复和QR收益合并归因。
3. **精度与连续性。** 报告SE(3)刚体对齐ATE、不拟合尺度/时间偏移，同时给平移/旋转RPE、初始化时间、完整输出及共同时间戳误差、覆盖和最大gap。按现有Vicon插值/原点口径解释，不冒充官方EuRoC榜单精度。
4. **资源与有效工作量。** estimator进程CPU秒、端到端P50/P95/P99、分阶段时间、Peak RSS、processed pairs、左右topic消息数、使用观测数及队列丢弃共同报告。`received_images`为左/主相机帧数，`received_image_messages`为所有相机消息数；不能因stereo计数翻倍虚增处理率。
5. **排队与测量口径。** `submit=false`是后端频率门控，不是overflow；记录同步丢弃、tracking pair drop和backend feature drop。bag/observer不绑核且不计入estimator CPU；正式测试不与build/其他benchmark并发。
6. **调度与热。** 使用上表四组互异调度设置，记录开始/峰值/结束温度与频率，保持governor不变。CPU6/7仅适用于本板，换硬件先读capacity；进程级taskset不等于tracker/backend分别固定到一个核。
7. **重复与决策。** 初筛至少2次、最终候选至少3次，轮换顺序；报告中位数与离散度。精度/连续性先过约定门槛，再比较CPU、P95和RSS的非支配点；不以单次easy胜出宣称全场景最优。
8. **复现材料。** 保存实际ELF SHA256、源码revision与差异、配置全文及hash、覆盖参数、播放topic、affinity mask、OpenCV版本和完整日志。本地证据位于 `output/stereo_arm_20261112/`，ARM证据位于 `/home/pinydrone/vins_stereo_20261112/`；历史mono资料保持独立。

研究建议的落点是：先利用已完成的双目链路与回归基础，测清观测预算和A76调度的折中；随后依据热点选择数值优化。单份协方差仅约20 KiB，GPU/NPU也尚无当前VINS数据路径，因此内存和异构加速收益都应由端到端证据支持。

## 8. 本地最终数据与适用结论

### 8.1 主对照：baseline → stereo n200 → n120

本地x86，完整1×回放，各配置每序列1次。baseline与n200均为前端`max_cnt=200`；n120仅将此前端预算改为120，三者后端`schur_max_features=120`、W=5、`estimate_td=0`。候选二进制n200/n120相同（ELF SHA256 `bf6bff4e…`），baseline为`c3272062…`；完整hash与配置见各label的manifest。CPU为estimator累计CPU秒，P95为观察者图像到里程计延迟，RSS为进程峰值。

| 序列 | 配置 | ATE m（各自全输出） | CPU s | P95 ms | RSS MiB | restart请求/reset入口 | 最大gap s | poses |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| easy | baseline（only-left Schur） | 0.27904 | 24.09 | 21.217 | 91.539 | 0/0 | 0.10 | 1130 |
| easy | stereo n200 | 0.29242 | 22.38 | 20.770 | 91.250 | 0/0 | 0.10 | 1130 |
| easy | stereo n120 | 0.29033 | 17.67 | 18.467 | 90.066 | 0/0 | 0.10 | 1130 |
| medium | baseline（only-left Schur） | 0.29383 | 21.74 | 23.208 | 90.297 | 0/0 | 0.10 | 1164 |
| medium | stereo n200 | 0.27419 | 22.94 | 24.012 | 90.531 | 0/0 | 0.10 | 1164 |
| medium | stereo n120 | 0.26200 | 18.08 | 21.211 | 89.684 | 0/0 | 0.10 | 1164 |
| difficult | baseline（only-left Schur） | 33.19376 | 16.98 | 16.896 | 91.703 | 5/5 | 1.20 | 1102 |
| difficult | stereo n200 | 0.37534 | 17.47 | 17.600 | 91.199 | 0/0 | 0.15 | 1157 |
| difficult | stereo n120 | 0.44546 | 15.66 | 17.613 | 90.215 | 0/0 | 0.15 | 1157 |

数据直接取各运行的`metrics.json`：[local_results](/home/sprout_stars/rmcode/VINS-Fusion/output/stereo_arm_20261112/local_results)。reset入口不等于已确认完成重置；`actual_resets_completed`无可靠标记。baseline easy与初始构建末尾重叠，其他上述回放未与本地构建并发；所有性能数字仅为单次筛选，不能宣称统计显著。[实验边界](/home/sprout_stars/rmcode/VINS-Fusion/output/stereo_arm_20261112/evidence/method.json)

baseline与candidate的轨迹输出集合不完全相同：easy/medium的输出时间戳集合相同；difficult为1102→1157 poses，baseline内部有重启空洞，虽然两者首末输出跨度均约115.55 s，也不代表覆盖与连续性相同。不能直接用全输出ATE跨覆盖排名。下表由`make_summary.py`产物分别读取baseline↔n200、baseline↔n120的**两标签时间戳交集**，每个交集重新对齐GT；不是两条估计轨迹互相对齐的误差。

| 序列 | 两组交集各自的时间戳数/GT匹配数 | baseline共同ATE m | n200共同ATE m | n120共同ATE m |
|---|---:|---:|---:|---:|
| easy | 1130/1127 | 0.27904 | 0.29242 | 0.29033 |
| medium | 1164/1162 | 0.29383 | 0.27419 | 0.26200 |
| difficult | 1102/1101 | 33.19376 | 0.38038 | 0.44987 |

[共同时间戳汇总JSON](/home/sprout_stars/rmcode/VINS-Fusion/output/stereo_arm_20261112/evidence/STEREO_RESULTS_20261112.json)。n200解决本次difficult的失稳与长gap，但easy共同ATE比baseline高约4.8%；n120相对n200的CPU约下降21%/21%/10%（easy/medium/difficult），RSS约减少1.18/0.85/0.98 MiB，easy/medium P95降低，difficult P95基本不变。n120的difficult全输出ATE比n200高约18.7%，因此是资源/精度折中，不能称无损或全场景最佳。

### 8.2 用户td1配置与前端隔离

`candidate_user_td1`已完成difficult：ATE **0.37534 m**、CPU18.63 s、P95 18.022 ms、RSS90.648 MiB、restart/reset=0/0、gap0.15 s。其1157×8轨迹与fixedtd n200逐元素全等且文件SHA256相同；本例未观察到轨迹差异，不构成“td0/td1对其他相机等价”的保证。Schur状态不含在线td，已知EuRoC时空标定下固定td能减少实验自由度；其他相机仍须校准并核查初始化与切换后的时间模型。[td1对照证据](/home/sprout_stars/rmcode/VINS-Fusion/output/stereo_arm_20261112/evidence/local_n120_and_user_td1_compare.json)

前端隔离仅在baseline源码上替换`feature_tracker.cpp/.h`，difficult的 **1102×8列轨迹与baseline全等**，仍为ATE33.19376 m、5次饥饿重启。CPU16.98→17.34 s、P95 16.896→17.006 ms、RSS91.703→91.969 MiB，**未证明CPU/P95/RSS收益**。因此本次difficult的大幅改善来自stereo/backend改动组合，不能归因于前端复用；隔离实验只支持本序列轨迹保持，不代表全部场景。[源码隔离与比较JSON](/home/sprout_stars/rmcode/VINS-Fusion/output/stereo_arm_20261112/evidence/local_frontend_only_compare.json)

### 8.3 数值检查与ARM结果边界

独立数值检查覆盖三clone双目、非零扰动残差：Schur消元与独立联合Jacobian/左零空间参考的A、b相对误差约`4.29e-16`、`4.53e-15`。同clone双目24点的三种场景（精确观测、扰动、current位置偏离FEJ）均未应用更新，`ΔP=0、Δpose=0`；参考投影pose Jacobian约`7e-16`，符合自由特征下单时刻双目不提供绝对pose信息。生产拒绝原因为`indefinite Schur information`，该有限检查不证明所有退化排列均安全；近零信息依然只有相对eigen保护、没有绝对信息下限，后续QR/数值改造必须继续覆盖此风险。[完整numeric checks报告](/home/sprout_stars/rmcode/VINS-Fusion/output/stereo_arm_20261112/numeric_checks/REPORT.md)

ARM结果采用clean构建后的有效回放，见第9节。早期 `candidate_a76` 因增量构建混用旧类布局，跟踪分阶段统计异常为零，已保留原始材料并排除；其结果不参与推荐。


### 8.4 双目 Ceres 参照

本地使用相同 candidate ELF、200个前端特征，仅将 `backend_type` 改为 `ceres`，三条完整1×回放均无restart/reset入口。easy/medium/difficult的ATE为 **0.4731/0.4262/0.4697 m**，CPU为 **41.89/36.31/25.34 s**，P95为 **40.30/39.93/28.69 ms**，RSS为 **102.84/100.32/97.54 MiB**。本轮中Ceres三条的精度、CPU、P95和RSS均未优于Schur n200；这是相同前端与当前参数下的单次实测，不能推广为所有NLS/EKF实现的排名。[完整Ceres对照](/home/sprout_stars/rmcode/VINS-Fusion/output/stereo_arm_20261112/evidence/local_ceres_comparison.json)


## 9. ARM 有效完整回放

设备、配置、标定、后端频率与本报告前述条件一致；以下均为真实左右目+IMU输入、OpenCV单线程、estimator绑定A76核心6/7。baseline使用原后端，n200/n120使用同一clean构建ELF（SHA256 `877af86098163104529bdbf30be9795a53e30e934134def8c3434f9af2ab1570`），仅前端`max_cnt`不同。完整干净构建后12/12 CTest通过，跟踪stage统计正常。CPU秒仅统计estimator，不含播放、observer或编译。

| 序列 | 配置 | ATE m（各自输出） | CPU s | P95 ms | RSS MiB | restart请求/入口 | 跟踪溢出丢对 | 最大gap s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| easy | baseline | 0.26659 | 92.56 | 104.12 | 103.57 | 0/0 | 14 | 0.15 |
| easy | stereo n200 | 0.30336 | 93.01 | 107.06 | 103.84 | 0/0 | 13 | 0.15 |
| easy | stereo n120 | 0.30318 | 84.19 | 49.50 | 95.91 | 0/0 | 0 | 0.10 |
| medium | baseline | 10.46305 | 94.64 | 66.64 | 109.44 | 2/2 | 8 | 1.20 |
| medium | stereo n200 | 0.27698 | 95.80 | 65.27 | 99.41 | 0/0 | 9 | 0.15 |
| medium | stereo n120 | 0.29711 | 86.37 | 53.39 | 95.61 | 0/0 | 0 | 0.10 |
| difficult | baseline | 0.28276 | 81.61 | 64.21 | 98.95 | 0/0 | 0 | 0.15 |
| difficult | stereo n200 | 0.34554 | 82.27 | 63.81 | 99.62 | 0/0 | 0 | 0.15 |
| difficult | stereo n120 | 0.47704 | 76.21 | 53.48 | 95.49 | 0/0 | 0 | 0.15 |

n120相对n200：CPU约下降 **9.5%/9.8%/7.4%**，P95约下降 **53.8%/18.2%/16.2%**，RSS约下降 **7.64%/3.82%/4.15%**；对应ATE变化约 **−0.1%/+7.3%/+38.1%**（easy/medium/difficult）。这是一台设备、每序列单次筛选；n120节省资源但difficult精度有明确代价，不能表述为无损优化。

n200的easy/medium跟踪队列最大等待约249/255 ms，部分帧处理超过50 ms输入周期；n120三序列最大等待约14/29/36 ms且无跟踪溢出。缩小前端预算改善了积压，不能把n200丢帧后的较少工作量计作性能优势。`V2_03_difficult`录包本身cam0=1922、cam1=2336条消息，observer完整接收，实际配对处理1921对；左右计数不对称来自输入，非自动判作传输丢失。

两组输出集合存在排队丢帧差异，精度比较还应查看[逐对共同时间戳ATE及处理帧数](/home/sprout_stars/rmcode/VINS-Fusion/output/stereo_arm_20261112/evidence/STEREO_RESULTS_20261112.json)。延迟为主相机消息在observer到达至里程计到达，并非传感器曝光至飞控执行的完整时延。实验未更改governor；调度、队列与重复验证结果在最终推荐中另列。


### 9.1 运行范围与独立配置

本次只测估计器，未启动RViz、loop_fusion或global_fusion。正常launch的默认值会启动RViz和loop_fusion，因此复现性能必须显式关闭：

```bash
ros2 launch vins vins.launch.py mode:=stereo_imu \
  config_file:=/home/sprout_stars/rmcode/VINS-Fusion/src/vins_estimator/config/euroc/euroc_stereo_imu_arm_balanced_config.yaml \
  opencv_threads:=1 loop_fusion:=false global_fusion:=false rviz:=false
```

上例为本地源码路径，远端使用配置在设备上的绝对路径，并先构建/加载匹配源码。本轮保留原工作区build/install不变，验证二进制位于独立目录；直接运行旧install不会自动包含源码修复。独立n200配置为同目录 `euroc_stereo_imu_arm_accuracy_config.yaml`（名称表示200预算，不保证各场景精度最高）。用户原配置未覆盖。两个配置保持原相机参数、`flow_back`和RANSAC，只固定EuRoC `td=0`并启用本设备6/7大核绑定；balanced另将`max_cnt`改为120。它们使用EuRoC V2标定，不能直接用于用户实际相机。

没有图像跟踪订阅时，原有绘图门控已避免drawTrack；关闭RViz还避免额外点云/路径/Schur可视化工作。若开启loop，二维元数据修复已覆盖camera0接口，但本轮性能和ATE不包含回环校正。将`opencv_threads`、跟踪队列或DDS深度写进算法`config_file`并不保证节点读取；应使用对应launch参数或节点`--ros-args -p`。


### 9.2 调度对照（medium，n200）

同一clean ELF与相同算法配置，仅改变是否绑6/7和OpenCV线程数。四组均为ATE约0.277 m、无restart/reset入口；每组仅一轮。

| 调度 | CPU s | P95 ms | RSS MiB |
|---|---:|---:|---:|
| A76 6/7，OpenCV 1线程 | 95.80 | 65.27 | 99.41 |
| 不绑核，OpenCV 1线程 | 134.97 | 104.44 | 99.25 |
| A76 6/7，OpenCV 2线程 | 130.70 | 52.77 | 96.47 |
| 不绑核，OpenCV 2线程 | 185.39 | 82.12 | 95.62 |

n200绑核双线程可降低尾延迟，但CPU比绑核单线程增加约36.4%。n120绑核单线程为CPU86.37 s、P9553.39 ms、RSS95.61 MiB，延迟接近n200双线程而CPU低约33.9%；medium ATE约0.297 m，存在约0.020 m误差差异。这些是性能/精度的取舍，RSS单次高水位差不能归因于线程数本身。绑定仅限本机CPU6/7，其他ARM按capacity重新选择。

### 9.3 精度参考坐标系

bag的 `vicon/firefly_sbx/firefly_sbx` TransformStamped提供Vicon标记体位姿，VIO里程计child为body/IMU。现有ATE采用GT位置插值和SE(3)位置对齐，不估尺度或时间偏移，但未校正标记体到IMU的杆臂；这些数值用于同一数据、同一参考下的筛选，不能等同官方EuRoC body真值ATE。difficult录包还存在左右图数量不对称，应保留输入覆盖审计。

新增1秒SE(3) RPE离线脚本通过合成验证，但未校正的marker/body旋转轴映射会使姿态RPE明显失真；因此原始姿态RPE不参与方案排名。已找到的V1 `vicon0/sensor.yaml` 外参不能未经核实直接套用V2。本报告不以这些未对齐RPE声称算法精度；正式验收需V2对应的body真值或确切marker→IMU标定。[RPE诊断及边界](/home/sprout_stars/rmcode/VINS-Fusion/output/stereo_arm_20261112/evidence/rpe_results.json)


### 9.4 n120 medium 重复验证

首次筛选加两个复测，共三次运行：ATE均为 **0.29711443 m**，1164×8列轨迹逐元素相同。CPU为 **86.37/86.59/86.36 s**（中位86.37 s），P95为 **53.389/53.474/53.530 ms**（中位53.474 ms），RSS为 **95.613/95.566/95.797 MiB**；三次均无restart/reset入口和跟踪溢出。可认为本设备、本数据上的n120单线程性能收益具有重复证据；easy/difficult仍只筛选一次，不能外推为所有序列均已重复验证。


已验证的ARM ELF位于 `/home/pinydrone/vins_stereo_20261112/binaries/candidate_clean_vins_node`。如使用独立二进制复现，显式传入配置及节点参数，避免launch使用旧install：

```bash
source /opt/ros/jazzy/setup.bash
taskset -c 6,7 /home/pinydrone/vins_stereo_20261112/binaries/candidate_clean_vins_node \
  --ros-args -r __node:=vins_estimator \
  -p config_file:=/home/pinydrone/vins_stereo_20261112/configs/arm_stereo_fixedtd_n120.yaml \
  -p opencv_threads:=1
```

完整回放命令和配置全文由每个label的manifest保存。该命令只启动估计器，不主动启动相机、飞控或回放。


### 9.5 小队列对照

保持n120、A76绑定和OpenCV单线程，仅将节点参数改为`image_queue_size=6`、`tracking_queue_size=2`、`tracking_queue_policy=latest`。easy/medium的输出轨迹与正常队列逐元素全等、无restart和跟踪溢出；CPU **84.06/86.43 s**，P95 **49.885/54.016 ms**，RSS **96.117/95.715 MiB**。与n120正常队列 **49.495/53.389 ms、95.906/95.613 MiB**相比，没有额外的延迟或内存收益证据，故推荐维持已测默认队列策略。

较小队列能限制过载时保留的旧帧，但本次n120没有过载，不能验证极端负载下latest的精度与丢帧代价。算法侧左右各100张752×480 mono8图像的像素payload容量上界约68.85 MiB，左右各6张约4.13 MiB；这是容量估计，不是已分配RSS。DDS image history仍固定为100，缩小同步队列不等于缩小DDS history，也没有实测节省约65 MiB。

## 10. 最终选择与后续优化顺序

- **当前资源平衡选项：** 独立balanced配置，120前端特征、Schur最多120 tracks、W=5、10 Hz后端、双向LK与RANSAC保留，OpenCV单线程，estimator绑定本设备A76核心6/7。完整三序列ATE约0.303/0.297/0.477 m，P95约49.5/53.4/53.5 ms，RSS约95.9/95.6/95.5 MiB；medium重复验证稳定。该结论限定当前数据与标定。
- **difficult精度优先：** 独立n200配置，difficult约0.346 m；需要接受更高资源消耗及easy/medium的积压。n200双线程可降延迟但CPU明显增大，不作为资源平衡默认。
- **已做而未证明提速：** 前端缓冲复用本地difficult轨迹不变，但CPU/P95/RSS没有稳定收益；不以代码层面的减少分配代替端到端性能证据。
- **后续最高价值：** 完成真实相机时空标定与Vicon marker/body参考统一；细化每目/三角化/门控/传播热点；再考虑前端与后端观测预算分开控制。当前结果不支持取消反向LK、RANSAC或FEJ。
- **后端数值优化：** 在独立消元/后验/秩与退化测试下研究Givens QR替代full-U SVD，以及IMU Phi/Qd累积。优先保留既有数值保护，不能仅为更低计算量取消门控或放宽失败阈值。
- **硬件路线：** 使用现有AArch64 OpenCV NEON和大核调度；GPU只适合下一阶段独立图像内核原型，须计入拷贝/同步。PowerVR无FP64且当前VINS无GPU路径；NPU编译模型和适用算子未验证，不能作为现成KLT/Schur加速开关。

所有更改留在工作树，未提交、未替换远端原部署，也未启动飞控。原用户YAML哈希保持 `d9e504c55d3feb777bc91f4850891d89cbcd63a9e0fdb516581bb59f904a0c4f`。可审查的结果摘要：[STEREO_ARM_RESULTS_20261112.json](/home/sprout_stars/rmcode/VINS-Fusion/docs/STEREO_ARM_RESULTS_20261112.json)；完整原始CSV/日志/manifest及重算工具保存在 `output/stereo_arm_20261112/`，远端保存在 `/home/pinydrone/vins_stereo_20261112/`。


热与频率连续采样：有效n200三序列回放的大核热区约65.8–84.2 °C，n120初筛约65.9–77.7 °C，n120重复约61.8–66.7 °C；均为原`schedutil` governor，采样频率在416–2002 MHz间变化，含播放/结束空闲时段。这不是固定频率实验，也不能由温度相关性单独认定硬件热节流；调度和负载是本轮实际运行条件。相对较冷时的n120复测仍得到相近CPU/P95，支持其稳定性。[分阶段温度/频率摘要](/home/sprout_stars/rmcode/VINS-Fusion/output/stereo_arm_20261112/evidence/arm_thermal_summary.json)


最终复现审计：ARM共16次有效完整回放（3次baseline、3次n200、3次n120、3次调度对照、2次n120复测、2次队列对照）；失效增量构建结果另存且不计入。全部有效candidate运行的ELF SHA256一致；最终源码树前后哈希一致，12/12 ARM CTest通过，`ldd`无缺失依赖。最终远端进程审计未发现遗留估计器、bag播放、编译或采样进程。[远端最终审计](/home/sprout_stars/rmcode/VINS-Fusion/output/stereo_arm_20261112/arm_evidence/candidate_final_audit.txt)

![ARM accuracy/resource trade-offs](/home/sprout_stars/rmcode/VINS-Fusion/output/stereo_arm_20261112/figures/arm_tradeoffs.png)

图仅展示三序列初筛，坐标/数值按各序列单独解释，不平均轨迹误差；可导出[SVG](/home/sprout_stars/rmcode/VINS-Fusion/output/stereo_arm_20261112/figures/arm_tradeoffs.svg)。
