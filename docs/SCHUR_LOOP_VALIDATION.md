# Schur EKF 回环与点云验证

## 复现与验收方式

使用 `~/vicon_room2/bags` 中的 ROS 2 EuRoC bag，单目 + IMU，源码中的
`euroc_mono_imu_config.yaml`，1 倍速回放，同时启动估计节点和 `loop_fusion`。
验证脚本使用独立 `ROS_DOMAIN_ID`，关闭节点后保留日志、配置及二进制 SHA-256。

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 scripts/validate_schur_loop.py \
  --output output/schur_loop_validation/after
```

脚本默认完整回放 easy、medium、difficult 三个序列。`--datasets` 可选单个序列，
`--duration 35` 可用于快速复现点云缺失；短片段不保证经过回环，仍然会保留严格回环
验收项并在未检测到回环时返回非零。`--backend ceres` 可检查原有后端。

验收依据：

- 回放 20 秒之后仍有非空 `point_cloud` 和 `keyframe_point`，避免把 Ceres 初始化
  阶段的输出误认为 Schur 正常输出。
- 点坐标有限；关键帧每点有 5 项 channel（归一化坐标、像素坐标、特征 ID）。
- `keyframe_pose` 与 `keyframe_point` 成对，时间戳能在相机帧中找到匹配
  （考虑 epoch 时间经 double 转换，容许 1 微秒误差）。
- 位姿图持续增长，并包含真正的回环边；`odometry_rect` 相对同时间戳原始
  `odometry` 出现非零修正。仅存在修正话题或路径不算回环成功。
- 轨迹精度以 bag 中 Vicon 真值进行时间插值及 SE(3) 刚性位置对齐，不拟合尺度
  或时间偏移。修正后的实时里程计不等同于最终优化后的整条历史轨迹。

## 修复前复现

`V2_01_easy` 前 35 秒，Schur 成功激活：

| 项目 | 结果 |
| --- | ---: |
| `point_cloud` 消息 / 非空 | 350 / 13 |
| 20 秒后 `point_cloud` 消息 / 非空 | 150 / 0 |
| `keyframe_point` 消息 / 非空 | 226 / 3 |
| 20 秒后 `keyframe_point` 消息 / 非空 | 121 / 0 |
| 位姿图最大节点数 | 216 |
| 回环边 | 0 |
| 最大位姿修正 | 0 m |

原始记录位于 `output/schur_loop_validation/before/V2_01_easy/summary.json`。

完整 easy 序列同样复现：2280 张图像全部接收，20 秒后的 940 条点云消息和
750 条关键帧点消息全部为空；位姿图达到 845 个节点但回环边仍为 0，最大修正
仍为 0 m。原始和修正后实时轨迹的 ATE RMSE 都为 0.21021 m。
完整记录位于 `output/schur_loop_validation/before_full/V2_01_easy/summary.json`。

修复前 Ceres 对照（同一完整 easy 序列）通过全部验收项：20 秒后的 940 条
点云全部非空，677 条关键帧点消息全部非空，检测到 52 条回环边，最大位置修正
0.86771 m。原始实时轨迹 ATE RMSE 为 0.24050 m，修正后为 0.19271 m。
该对照说明 bag、相机参数和词袋回环链路可用。

## 实现说明

Ceres 的 `FeatureManager::setDepth()` 会设置 `solve_flag=1`，原有点云和
关键帧发布依赖该标志。切换 Schur 后不再执行这一步，旧的已求解特征离开窗口后，
发布端只能生成空点云。Schur 的 clone 窗口也不能按传统固定窗口下标读取。

修复为 Schur 增加独立的发布快照，在视觉更新之后、消费观测和裁剪 clone 之前，
用当前 clone 位姿对保留轨迹的副本进行三角化。复用现有深度、有限值和重投影
误差检查，提供世界坐标三维点及当前图像的归一化/像素坐标。仅有订阅者时生成
快照，不修改滤波状态、协方差或观测消费规则。重置、重新初始化和 hold 时清空
快照。

关键帧使用当前帧的位姿和原始图像时间戳；回环端以整数纳秒进行 2 微秒容差匹配，
防止浮点时间转换导致使用下一帧图像。Ceres 保留原有发布逻辑。

## 测试范围与已有问题

`colcon build --packages-select vins loop_fusion` 成功；
`colcon test` / `colcon test-result` 汇总 122 项，0 errors、0 failures、0 skipped。
新增 5 个 GTest 用例覆盖快照三角化和输入不被修改、缺失像素时的过滤、窗口未满时
的边缘化判定、原始像素历史及时间容差。

仓库原有 `scripts/tests/test_launch_structure.py` 的两处断言仍期望
`loop_fusion` 默认关闭，但 `HEAD` 中的 `vins.launch.py` 已默认开启，存在既有
不一致；本修复未修改 launch 默认值。远程 benchmark wrapper 的测试带有进程
互斥检查，应在 bag 回放完全退出之后运行。

## 最终版本完整回放结果（2026-10-29）

单目 + IMU、1 倍速；三组均通过脚本全部验收项。表中点云计数只统计回放 20 秒后，ATE 单位为米。

| 序列 | 非空点云 / 消息 | 非空边缘化点云 / 消息 | 配对关键帧 | 回环边 | 原始 VIO ATE | 实时修正 ATE | 最终位姿图 ATE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2_01_easy | 940/940 | 938/940 | 855 | 67 | 0.21021 | 0.17226 | 0.14136 |
| V2_02_medium | 974/974 | 971/974 | 1071 | 83 | 0.27331 | 0.25938 | 0.22564 |
| V2_03_difficult | 962/968 | 910/968 | 1066 | 16 | 0.31782 | 0.32839 | 0.29782 |

所有点坐标和关键帧 channel 均为有限值，关键帧 pose/point 无未配对消息，全部能匹配到源图像。三组图像接收数分别为 2280、2348、1922，与 bag 元数据一致；均无跟踪队列溢出。

三组原始 `vio.csv` 与修复前逐字节一致，说明发布快照没有改变滤波结果。最大位置修正分别约为 0.379、0.397、0.480 m。

difficult 存在视觉退化帧，因此少数帧无通过三角化检查的点，边缘化点云也不要求每帧非空。其最终位姿图 ATE 改善，但实时修正轨迹 ATE 从 0.31782 m 上升到 0.32839 m；本次修复证明回环正常建立和应用，并不保证每条序列的实时 ATE 都改善。

最终原始日志、配置及逐序列 summary 位于 `output/schur_loop_validation/final/`，可版本控制的测量摘要保存在 `docs/SCHUR_LOOP_RESULTS.json`。所有最终序列使用同一组二进制 SHA-256。

最终二进制的 Ceres 完整 easy 回归同样通过：后期点云 940/940 非空、763 组
关键帧全部配对、52 条回环边，与修复前 Ceres 对照一致。原始 VIO ATE 为
0.24050 m，实时修正 ATE 为 0.19263 m，最终位姿图 ATE 为 0.16924 m。
记录位于 `output/schur_loop_validation/ceres_final/V2_01_easy/summary.json`。

全部回放退出后复跑 `python3 -m pytest scripts/tests -q`：22 passed、2 failed；
失败仅为上述已有 launch 默认值断言，远程 wrapper 测试全部通过。

后续 `margin_cloud` 质量修复增加了发布资格、几何门控与生命周期去重，因此当前
历史点云数量少于本表。最新点云质量对比及回环回归见
[SCHUR_MARGIN_CLOUD.md](SCHUR_MARGIN_CLOUD.md)。本报告保留第一次恢复回环时的测量记录。
