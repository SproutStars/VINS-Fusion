# VINS-Fusion for ROS 2 Jazzy

本项目将基于 ROS 1 Kinetic 的 VINS-Fusion 迁移到 **Ubuntu 24.04 / ROS 2 Jazzy**。
保留特征跟踪、IMU 预积分、滑动窗口优化、边缘化、回环检测及 GPS 全局融合算法，
使用 `ament_cmake` / `colcon`、原生 `rclcpp` 消息和 `tf2_ros`，不需要 ROS 1 或 `ros1_bridge`。

上游论文、作者和历史说明保存在 [ROS 1 README](docs/README_ROS1.md)，
旧 launch/RViz 文件归档在 `docs/ros1_assets`。代码沿用各文件的上游许可证。

## 安装与编译

先安装 ROS 2 Jazzy；在本工作区根目录执行：

```bash
source /opt/ros/jazzy/setup.bash
sudo apt update
sudo apt install python3-colcon-common-extensions python3-rosdep \
  libceres-dev libeigen3-dev libopencv-dev libboost-all-dev \
  ros-jazzy-cv-bridge ros-jazzy-tf2-ros ros-jazzy-rviz2
# 如果机器还没有初始化 rosdep，先执行 sudo rosdep init，再执行 rosdep update。
rosdep install --from-paths src --ignore-src --rosdistro jazzy -r -y
colcon build --symlink-install --executor sequential \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
source install/setup.bash
```

使用 zsh 时将上述 `setup.bash` 换为 `setup.zsh`。内存较少时可在编译命令前设置
`MAKEFLAGS=-j2`，限制 C++ 编译并发数。

| 包名 | 作用 | 可执行程序 |
| --- | --- | --- |
| `camera_models` | 相机投影模型与标定 | `Calibrations` |
| `vins_common` | 时间、配置路径等 ROS 2 公共工具 | — |
| `vins` | 视觉惯性估计，源码目录为 `src/vins_estimator` | `vins_node`, `kitti_odom_test`, `kitti_gps_test` |
| `loop_fusion` | 回环检测和位姿图优化 | `loop_fusion_node` |
| `global_fusion` | GPS 全局融合 | `global_fusion_node` |

Ubuntu 24.04 的 Ceres 2.2 已删除 `LocalParameterization`。本实现使用 `Manifold`，
并将 VINS 的解析 Jacobian 在切空间与环境空间之间转换；不能简单替换类名。

## 启动

单目 + IMU（默认 EuRoC）：

```bash
ros2 launch vins vins_mono_imu_launch.py
```

双目 + IMU、双目无 IMU：

```bash
ros2 launch vins vins_stereo_imu_launch.py
ros2 launch vins vins_stereo_launch.py
```

指定实际相机配置，同时开启回环和 RViz：

```bash
ros2 launch vins vins.launch.py \
  config_file:=/absolute/path/to/config.yaml \
  loop_fusion:=true rviz:=true
```

启动参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `config_file` | 安装包中的 EuRoC 单目 IMU 配置 | 算法 YAML（ROS 2 参数格式，兼容旧版 OpenCV 格式） |
| `use_sim_time` | `false` | 使用 rosbag 的 `/clock` |
| `loop_fusion` | `false` | 启动回环节点 |
| `global_fusion` | `false` | 启动 GPS 融合，需要 `/gps` |
| `rviz` | `false` | 启动 RViz 2 |
| `output_path` | `$HOME/.ros/vins_fusion` | 轨迹及位姿图输出目录，自动创建 |

也可直接运行节点；仍支持旧版的配置文件位置参数：

```bash
ros2 run vins vins_node --ros-args \
  -p config_file:=/absolute/path/to/config.yaml \
  -p output_path:=/tmp/vins_output
ros2 run loop_fusion loop_fusion_node --ros-args \
  -p config_file:=/absolute/path/to/config.yaml
ros2 run global_fusion global_fusion_node --ros-args \
  -p output_file:=/tmp/vins_output/vio_global.csv
```

直接运行估计节点时还可设置 `image0_topic`、`image1_topic`、`imu_topic`、
`image_queue_size`（默认 100）、`stereo_sync_tolerance`（默认 0.003 秒）。
输入图像和 IMU 使用 best-effort / volatile QoS，兼容 ROS 2 传感器驱动和可靠 rosbag 发布者。
输出轨迹和可视化消息使用 reliable / volatile QoS。

## 配置相机和 IMU

算法配置使用 ROS 2 参数 YAML 格式（`/**` 下的 `ros__parameters`），并作为
`config_file` 参数传给节点；旧版 OpenCV `%YAML:1.0` 文件仍然兼容。请用
`config_file` 而不是 `--params-file` 传入：前者会把相机标定、输出目录等相对路径
解析到配置文件所在目录，并让多个节点共用同一份配置。

修改 `image0_topic`、`image1_topic`、`imu_topic`、分辨率、内参和 `body_T_cam*`，
使它们与实际传感器对应。`image_width` / `image_height` 需要与相机（相机标定文件）
的分辨率一致；仓库中的 EuRoC 配置按 1280 × 960 标定。

`cam0_calib` / `cam1_calib` 既可以是 camodocal 格式的相机模型文件（例如
`cam0_pinhole.yaml` / `cam0_mei.yaml`），也可以直接指向 ROS `sensor_msgs/CameraInfo`
标定文件（例如 `camera0_info.yaml` / `camera1_info.yaml`）。程序会自动识别并读取
`camera_matrix` 与畸变系数：`plumb_bob` / `rational_polynomial` 映射为针孔模型，
`equidistant` / `fisheye` 映射为 Kannala-Brandt 模型。不要混用不同相机的内参和
EuRoC 的外参。

标定路径支持绝对路径；相对路径按算法配置所在目录解析。直接运行节点时，配置内的
相对输出路径也按配置目录解析；launch 会用 `output_path` 覆盖它，避免写入安装包目录。
单目无 IMU 不受支持。运行时切换双目必须在启动配置中已提供两个相机的标定；
启用 IMU 需要在配置中提供 IMU 话题及有效噪声参数。传感器切换会重新初始化估计器。

主要输入为 `/cam0/image_raw`、`/cam1/image_raw` 和 `/imu0`，按配置修改。
可选的 `/feature_tracker/feature` 保留 `sensor_msgs/msg/PointCloud` 输入格式。
主要输出：

- `/vins_estimator/odometry`：优化后的位姿和速度。
- `/vins_estimator/imu_propagate`：IMU 高频传播位姿。
- `/vins_estimator/path`、`point_cloud`、`image_track`：轨迹、三维特征和跟踪图像。
- `/vins_estimator/keyframe_pose`、`keyframe_point`、`extrinsic`：回环输入。
- `/loop_fusion/pose_graph_path`、`odometry_rect`：回环修正结果。
- `/global_fusion/global_odometry`、`global_path`：GPS 融合结果。
- `/tf`：`world → body → camera`。

回环操作改为 ROS 2 服务，无需终端键盘线程：

```bash
ros2 service call /loop_fusion/save_pose_graph std_srvs/srv/Trigger '{}'
ros2 service call /loop_fusion/new_sequence std_srvs/srv/Trigger '{}'
```

重启和传感器切换仍保留 Bool 话题，例如：

```bash
ros2 topic pub --once /vins_restart std_msgs/msg/Bool '{data: true}'
```

## 数据集回放

在两个终端分别执行：

```bash
ros2 launch vins vins_stereo_imu_launch.py use_sim_time:=true loop_fusion:=true rviz:=true
ros2 bag play /path/to/ros2_bag --clock
```

ROS 1 `.bag` 需要先转换为 ROS 2 rosbag2，或使用另行安装的兼容存储插件。
确认转换后图像、IMU 的话题名和原始时间戳保持一致；此仓库不提供数据集转换器。

KITTI 离线程序：

```bash
ros2 run vins kitti_odom_test /absolute/path/to/kitti_config.yaml /path/to/sequence/
ros2 run vins kitti_gps_test /absolute/path/to/kitti_config.yaml /path/to/raw_sequence/
```

GPS 数据需要搭配 `global_fusion_node`；可在另一个终端启动它。KITTI 的相机配置在
`src/vins_estimator/config/kitti_odom` 和 `config/kitti_raw` 中。

## 验证

```bash
colcon test --packages-select vins --event-handlers console_direct+
colcon test-result --verbose

# 在已 source install/setup.bash 的终端中验证节点通信。
# 需要 rclpy、python3-numpy、python3-opencv。
python3 scripts/ros2_smoke_test.py
```

测试覆盖位姿流形的单位四元数、Plus/Minus 往返、Jacobian 数值差分、投影及 IMU 因子的
坐标转换和 ROS 时间单位。集成测试用合成双目图像和 GPS 数据验证特征跟踪、
滑动窗口优化、边缘化、重启、回环服务、全局融合输出、双目 IMU 预积分、运行时 IMU 切换及等待 IMU 时的退出；
使用独立的 ROS_DOMAIN_ID（默认 73）和临时输出目录，无需数据集。
这些验证不能替代完整数据集精度评测；实机或
EuRoC 回放时还需检查初始化、轨迹、回环及 GPS 融合结果。

## Docker

```bash
docker build -f docker/Dockerfile -t vins-fusion:jazzy .
./docker/run.sh vins_mono_imu_launch.py
```

容器使用 host 网络访问 ROS 2 话题；此脚本不启动 GUI。数据集目录可用额外的
`docker run -v` 参数挂载。容器运行的输出写入 Docker 命名卷 `vins-fusion-output`。

## 性能评测

工程优化范围、回归测试与 `~/vicon_room2/bags` 的可复现比较方法见
[性能优化与 EuRoC 回归测试](docs/PERFORMANCE.md)。保留算法参数不变，分别检查
CPU 时间、延迟分位数、峰值内存与轨迹精度，避免以丢帧或精度退化换取表面加速。
