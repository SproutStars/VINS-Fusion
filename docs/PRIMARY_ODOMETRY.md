# 回环修正作为主里程计输出

通过 `vins.launch.py`（包括三个旧 launch 包装入口）启动时，
`/vins_estimator/odometry` 是统一的主里程计接口：

| launch 参数 | 主话题发布者 | 主话题内容 | 原始 VIO |
| --- | --- | --- | --- |
| `loop_fusion:=true` | `loop_fusion` | 应用最新位姿图修正的里程计 | `/vins_estimator/odometry_raw` |
| `loop_fusion:=false` | `vins_estimator` | 原始 VIO | `/vins_estimator/odometry` |

默认 `loop_fusion:=true`。尚未形成回环时修正为单位变换，正常输出当前 VIO；
形成回环后应用最新修正。每条原始图像频率里程计生成一条主输出，保持原始时间戳。
主话题只由一个节点发布。回环节点订阅 raw 话题，避免把修正结果再次送入回环。
`/loop_fusion/odometry_rect` 保留，与主输出内容相同。

修正包含位姿、完整 6×6 位姿协方差及其中的交叉项。速度与速度协方差按
`nav_msgs/Odometry` 约定位于 `child_frame_id=body`，切换世界坐标系时保持不变。
Ceres 原始主里程计的线速度也修正为 body 坐标表达（此前消息声明 body、数值却
位于 world）；Schur 已遵循该约定。位姿协方差仅对原始估计做坐标变换，不包含
位姿图修正本身的不确定性。

`path` 与主 TF 也同步切换：开启回环时使用主里程计的修正位姿；原始路径保留为
`path_raw`，原始消息采用 `world_vio` / `body_vio` / `camera_vio` 坐标名。
`imu_propagate` 仍为局部 IMU 高频传播；`vio.csv` 保留原始 VIO 数值。
全局 GPS 融合仍连接连续的 raw VIO，避免把回环跳变再次融合。
`vio_loop.csv` 和 `pose_graph_path` 保留位姿图的结果。

该开关在启动时生效。回环进程退出时主输出会停止，不能静默切回另一坐标系。

## 独立运行节点

不使用 launch 时，默认保留旧的独立节点行为。如需相同的主输出语义，显式配置：

```bash
ros2 run vins vins_node --ros-args -r __node:=vins_estimator \
  -p config_file:=/absolute/path/to/config.yaml \
  -r /vins_estimator/odometry:=/vins_estimator/odometry_raw \
  -r /vins_estimator/path:=/vins_estimator/path_raw -p use_raw_frames:=true
ros2 run loop_fusion loop_fusion_node --ros-args \
  -p config_file:=/absolute/path/to/config.yaml \
  -p input_odometry_topic:=/vins_estimator/odometry_raw \
  -p publish_primary_odometry:=true -p publish_primary_visualization:=true
```

回环节点会拒绝输入话题解析为修正输出话题的配置，包含 ROS remapping 后的名称。

## 验证

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 scripts/validate_schur_loop.py --launch-mode on \
  --datasets V2_01_easy --output output/primary_odometry_validation/on
python3 scripts/validate_schur_loop.py --launch-mode off --duration 35 \
  --datasets V2_01_easy --output output/primary_odometry_validation/off
```

实际启动 launch、回放 bag，并检查主话题仅有一个发布者；开回环时主输出与
`odometry_rect` 逐消息相同、raw 输入与修正输出时间戳一一对应，body 速度与协方差
保留，真实回环修正非零；关回环时主输出为 VIO 且没有回环输出。

构建与针对性检查：`vins` / `loop_fusion` 的 `colcon` 测试汇总 133 项通过，
Python 测试 25 项通过。覆盖旋转后的 body 速度、单位与非单位修正、完整位姿协方差
交叉项、速度协方差保留及开关路由。实际进程启动也验证了直接输入反馈配置与经过
ROS remapping 的反馈配置都会在加载词袋之前以错误退出。

Schur 开回环完整 `V2_01_easy` 实测：2280 张图像、1083 条 raw 里程计与 1083 条
修正里程计一一对应；主输出只有 `loop_fusion` 一个发布者，主消息与
`odometry_rect` 完全相同，速度、速度协方差、child frame、时间戳全部保留。
67 条回环边，最大位置修正 0.37948 m。原始 `vio.csv` 与功能变更前逐字节一致。

关闭回环的 35 秒 easy 回放也通过：700 张图像，293 条主里程计，唯一发布者为
`vins_estimator`，无修正输出。首次验证采集到 DDS 临时节点名
`_NODE_NAME_UNKNOWN_`，脚本增加等待完整 ROS 图发现后复跑，路由检查通过。

Ceres 开回环完整 easy 回放通过：1083 条 raw 与 1083 条修正消息一一对应，
主话题唯一发布者为 loop_fusion，并与 odometry_rect 完全相同；body 速度和速度
协方差保留。52 条回环边，最大位置修正 0.86771 m。

最终三种模式测量摘要保存在 `docs/PRIMARY_ODOMETRY_RESULTS.json`，原始日志和
配置位于 `output/primary_odometry_validation/`（关闭回环最终结果为 `off_verified`）。

## 路径与 TF 同步

| 输出 | 回环开启 | 回环关闭 |
| --- | --- | --- |
| `/vins_estimator/path` | 回环节点发布实时修正轨迹 | 估计节点发布原始轨迹 |
| `/vins_estimator/path_raw` | 原始轨迹，`world_vio` | 不发布 |
| `world → body → camera` | 主里程计位姿 + 相机外参 | 原始 VIO 位姿 + 相机外参 |
| `world → world_vio` | 当前合成回环修正 | 不发布 |
| `world_vio → body_vio → camera_vio` | 原始 VIO 链 | 不发布 |

主 TF 与主里程计逐帧采用相同时间戳及位姿，只有一个发布源。相机 TF 使用已收到
的真实外参，首次外参到达前不发布。RViz 固定坐标系仍用 `world`；原始点云和相机
marker 使用 `world_vio`，通过回环变换接入同一 TF 树，不再把未修正数据标成 world。

主 path 默认保留 2000 个采样位姿、每秒发布一次，末端匹配该时刻的主里程计。
时间回跳清空主 path 历史。其旧采样保持当时使用的修正，不随之后的优化回写；
整体重优化历史请查看 `/loop_fusion/pose_graph_path`（RViz 的 Loop Path）。
RViz 原 VIO Path 显示更名为 Primary Path，保持订阅 `/vins_estimator/path`。

路径/TF 功能与主里程计一样由 launch 的 `loop_fusion` 开关控制。独立启动时，
需同时使用 `use_raw_frames:=true`、path 的 remap 及
`publish_primary_visualization:=true`，避免重复广播主 TF。

路径/TF 扩展的 C++ 测试汇总 140 项通过，新增测试包含有界历史、重复时间戳、
时间回跳、路径与 TF 同源、外参有效性及参数校验。启动参数使用只读描述符，
避免运行时修改坐标名，同时允许 ROS 在启动阶段设置 `use_raw_frames`。

Schur 开回环的路径/TF 完整 easy 回放通过：109 条主 path 与 109 条 raw path，
主 path 唯一发布者为 loop_fusion，所有路径采样与相同时间戳的主里程计一致。
1083 条主里程计均有唯一且数值一致的 `world → body` TF；原始 TF 链、连接变换及
相机外参 TF 均存在。原始 `vio.csv` 与扩展前逐字节一致。

关闭回环的 35 秒回放通过：30 条主 path 仅由 vins_estimator 发布，全部采样与
主里程计一致，293 条里程计均有唯一匹配的主 TF，相机 TF 正常，无回环输出。

实际参数服务验证：`use_raw_frames:=true` 启动成功，运行时改为 false 被只读参数
规则拒绝。查看新 TF 树前建议重启 RViz，清除旧 world/body 关系及历史路径缓存。

Ceres 开回环完整 easy 回放通过：1083 条里程计、109 条主路径消息，路径采样
与主里程计一致，每条里程计具有唯一一致的主 TF；独立原始链和连接变换检查通过，
保持 52 条回环边。三个模式的最终测量见 `docs/PRIMARY_VISUALIZATION_RESULTS.json`，
日志位于 `output/primary_visualization_validation/{on_final,off_final,ceres_final}/`。
