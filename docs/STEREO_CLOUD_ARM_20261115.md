# 双目 + IMU 点云改进与 ARM 验证（2026-11-15）

实现由用户指定的 DeepSeek ds 子代理执行；主代理负责代码审阅和 192.168.5.62 的 ARM 原生验证。

## 范围与参考

改进路径为 `backend_type: schur_ekf`、`imu: 1`、`num_of_cam: 2`。仓库现有 EuRoC 双目 IMU 配置已经使用该后端。Ceres 点云路径不增加新逻辑，单目不启用额外历史。输出依然是实际跟踪特征的稀疏点云，不是稠密深度图。

参考 `/home/sprout_stars/rmcode/open_vins/ov_core/src/feat/FeatureInitializer.cpp` 的多视图三角化、条件检查及非线性细化思路，并参考 OpenVINS 将特征生命周期与输出分开的管理方式。复用本项目已有的 `SchurUpdater::triangulate()`（当前 clone 位姿、双目基线、迭代细化），不引入 OpenVINS 状态估计器或额外求解依赖。

## 点云生成

测量历史 `FeatureHistory` 必须在滤波更新消费后清除，防止重复使用观测。新增 `SchurCloudHistory` 独立保存当前 clone 窗口内的真实观测，跨越消费段；只用于点云发布，不进入 EKF 更新或回环关键帧。绝对 frame index 在读取时转换为 clone index，滑窗时删除过期记录，reset、hold 和 soft restart 清空。

内存与计算预算由现有 `max_cnt` 和 `schur_window_size` 推导，不增加前端特征或求解器预算。每特征观测数及总特征数均有上限，重建候选最多 `2 * max_cnt`，按观测数、特征 ID 确定次序。候选上限不保证覆盖窗口中所有历史特征。无可视化订阅者时不三角化点云；只有回环订阅时不执行额外点云重建。单目使用相同输入向量时只三角化一次。

双目运行中的发布点统一检查：

| 条件 | point_cloud | margin_cloud |
|---|---:|---:|
| 重投影 RMS | ≤2.5 px | ≤2 px |
| 任一观测重投影误差 | ≤4 px | ≤3 px |
| 视线夹角 | ≥0.5° | ≥1° |
| 固定位姿的相对深度标准差 | ≤25% | ≤15% |

实际双目配对可由同一时刻的两相机基线满足两视图约束，单目 margin 仍要求至少 3 个 clone。点云视差检查采用线性复杂度的两参考射线扫描，保守拒绝不足视差的候选；margin 保留原有精确两两视差检查。所有点仍经过已有三角化的深度、有限值和条件检查。

`margin_cloud` 仍仅发布滤波真正接受并消费的 ID，每个连续特征生命周期最多一次；额外历史只增加 `point_cloud` 来源，不扩大回环输入。`keyframe_point` 的 camera-0/raw-pixel 约定不变。

条件深度标准差只传播像素噪声，假定当前位姿固定，不包含位姿、外参、时间同步或噪声模型误差，不能当作绝对地图精度。仅两视图的沿极线错匹配可以产生零残差的错误深度，几何门控无法保证识别；跨时刻观测提供额外约束。

## 验证口径

远程工作目录：`/home/pinydrone/vins_stereo_cloud_20261115`。基线来自同一初始 HEAD 的已有 ARM 二进制，另存副本；候选在独立源码和 build 目录原生构建，不覆盖设备已有工作区。编译为 Release、portable code generation，未使用 `-march=native` 或 fast-math。

采用 EuRoC V2 easy/medium/difficult 完整 1× 回放，双目配置 `euroc_stereo_imu_arm_balanced_config.yaml`，`max_cnt=120`、窗口 5、OpenCV 单线程、估计器绑定 A76 核 6/7，前端队列 5。两组均订阅 point_cloud 和 margin_cloud，以测到真实重建/发布成本。算法配置仅开启阶段耗时并将标定路径解析为相同文件的绝对路径。

脚本增加可选 `--cloud-metrics`，保存全程与第一张左图 +20 秒后的点云统计；`--save-cloud-csv` 可输出逐帧统计。关闭选项时不新增点云订阅，原有图像计数、延迟、ATE 和 QoS 口径不变。观察器只保留 O(消息数) 的标量元数据，不保留所有 XYZ。

逐帧累计点数会重复统计同一特征，不能等同于唯一地图点数。图像计数是 observer 的消息数，不证明 estimator 实际接收或 tracker 处理每张图像。CPU 的 100% 表示一个核，仅统计估计器，包含现有脚本的启动/排空时间口径。

V2 原始 Vicon marker 与 IMU body 的确切外参在当前数据中未提供；保留现有脚本的原始位置 ATE 作为成对诊断，不把它解释为已校正的 body 绝对误差。将同时检查两组输出时间戳、轨迹、覆盖跨度、输出间隔、重启和 tracker overflow。Vicon 轨迹不能验证三维表面误差。

## 结果

原始证据位于 `output/stereo_cloud_20261115/`。

本地构建成功；CTest 13/13 测试目标通过，点云测试 18/18 通过；Python 脚本回归测试 87/87 通过；修改的 C++ 文件通过 clang-format-18 检查，`git diff --check` 干净。

固定真实位姿、0.11 m 双目基线、4 m 真值点、5 个时刻、每轴 0.25 px 独立高斯噪声、100 次合成试验：最新单对双目观测的平均三角化位置误差为 0.095639 m，使用全部 5 个时刻为 0.00701147 m。测试验证更多真实观测能降低这个受控条件下的误差，不代表真实表面重建的绝对精度。

ARM Release 原生构建成功，板上点云 18 项、Schur 更新 37 项、估计器恢复/IMU 10 项测试全部通过。板上相同合成试验的平均误差为 0.0940151 m → 0.007489 m；平台标准库随机数/浮点实现存在差异，不将跨平台数值变化当作算法优劣。

三序列第一轮对照（点数统计取左图 +20s 后；CPU 的 100%=一个核）：

| 序列 | 每帧 point_cloud 基线→候选 | 增量 | margin 累计点数 基线→候选 | P95 延迟 ms 基线→候选 | CPU % 基线→候选 | RSS MiB 基线→候选 | ATE m 基线→候选 |
|---|---:|---:|---:|---:|---:|---:|---:|
| easy | 111.78 → 126.55 | +13.2% | 4128 → 6288 | 51.89 → 54.94 | 72.23 → 74.26 | 99.17 → 98.41 | 0.3032 → 0.3018 |
| medium | 124.05 → 165.20 | +33.2% | 6241 → 13683 | 54.48 → 56.89 | 71.38 → 73.83 | 95.66 → 98.14 | 0.2971 → 0.2944 |
| difficult | 88.82 → 121.39 | +36.7% | 6042 → 10255 | 53.69 → 55.73 | 62.54 → 64.49 | 96.73 → 96.79 | 0.4467 → 0.4770 |

两组各序列 observer 左图数为 2280/2348/1922，右图数为 2280/2348/2336，与 bag 各自真实消息数一致。difficult 原始双目流存在缺帧，不能用理想帧率推算左图数量。两组各序列 tracker 总帧为 2280/2348/1921，difficult 比左图 observer 少 1；overflow_events 和 overflow_dropped_pairs 均为 0，不能把这 1 帧解释为已确认的队列溢出。

两组输出数均为 1130/1164/1157，输出时间戳逐个相同，无 >250ms 输出间隔；latency_misses、重启请求、reset entry 均为 0。全部发布点均有限。首个输出相对首张左图为约 1.05/1.05/1.10 s，末端覆盖到最后一张左图。

轨迹数值并非逐位相同，成对最大位置差为 0.113/0.146/0.678 m。easy/medium 的 ATE 相近，difficult 约增加 0.030 m（6.8%），不能宣称 VIO 精度无回退。新增代码不向滤波写入点，差异原因未定；三个序列从首个发布位姿起就有差异（约 0.317/0.049/1.131 mm），早于 Schur 额外点云路径启用。当前运行仍包含有时间预算的 Ceres 初始化及多线程调度，不能仅据代码隔离就断言轨迹不变，也不能把这些差异直接归因于额外历史。单轮三序列结果不建立统计显著性，A55 性能未测。

构建候选时使用格式化前源码；随后 ds 仅对这 8 个 C++ 文件运行 clang-format-18。`format_only_source_audit.json` 核对各文件移除空白后完全一致，本地格式化后重新构建和测试通过。原生候选二进制及其构建源码 SHA256 均已保存，便于区分源文件格式与可执行行为。

## easy 复测

反向运行顺序中，候选复测为 126.55 点/帧、margin 6290 点、CPU 74.39%、P95 54.51 ms、RSS 96.13 MiB、ATE 0.3031809116 m。其全部 1130 行轨迹（时间戳、位置、四元数）与第一轮基线逐项完全一致，同时仍比基线多约 13.2% 的每帧发布点，margin 多约 52.4%。这支持发布历史与滤波隔离的实现，但不消除 difficult 单轮 ATE 差异的不确定性。

基线 easy 复测为 111.78 点/帧、margin 4128 点、CPU 72.16%、P95 51.37 ms、RSS 95.98 MiB、ATE 0.3031809116 m。它与候选复测的全部轨迹数值也完全一致。复测差值为 +13.2% 点/帧、+52.4% margin 点，CPU +2.23 个百分点，P95 +3.14 ms，RSS +0.16 MiB。

## 使用

重新构建后，使用仓库现有双目 IMU Schur 配置即可自动生效，无需额外参数：

```bash
source /opt/ros/jazzy/setup.bash
source /home/sprout_stars/rmcode/VINS-Fusion/install/setup.bash
ros2 launch vins vins.launch.py mode:=stereo_imu \
  config_file:=/home/sprout_stars/rmcode/VINS-Fusion/src/vins_estimator/config/euroc/euroc_stereo_imu_arm_balanced_config.yaml \
  opencv_threads:=1
```

该 ARM 配置绑定核心 6/7，只适用于已核对 CPU 拓扑的设备；实际相机请使用自己的双目标定及 IMU 参数。RViz 的 PointCloud 显示订阅 `/vins_estimator/point_cloud`；需要累计已消费的可靠特征点时订阅 `/vins_estimator/margin_cloud`。相机模式、话题名和标定仍由配置决定。

新增代码是窗口内特征点重建；若目标是表面级稠密地图，仍需独立深度/建图模块，不能用本结果声称已实现稠密重建。当前实测验证限于 A76 核 6/7，不能外推到 A55。
