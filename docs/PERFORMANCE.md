# 性能优化与 EuRoC 回归测试

## 结论

在 `~/vicon_room2/bags` 的三条完整 EuRoC 序列上，最终版本相对保留基线降低了
6.7%–9.2% 的 estimator CPU 时间，轨迹结果与输出数量保持一致。代价是观测延迟
分位数上升约 1.4%–4.6%，但绝对 P95 仍低于 30 ms，小于约 50 ms 的相机帧间隔（后端目标周期为 100 ms）。

测量环境：AMD Ryzen 9 7940HX、ROS 2 Jazzy、Release 构建、1 倍速完整回放、每条序列
重复 2 次。基线是任务开始前实际编译并保存的二进制，而不是用 Git 提交号代替。

| 序列 | CPU 秒（基线 → 最终） | CPU 变化 | P95 ms | P99 ms | 峰值 RSS MiB | ATE RMSE m |
|---|---:|---:|---:|---:|---:|---:|
| V2_01_easy | 33.270 → 31.045 | -6.7% | 28.064 → 28.632 | 30.167 → 30.597 | 110.81 → 110.91 | 0.253444 → 0.253444 |
| V2_02_medium | 30.785 → 28.070 | -8.8% | 25.125 → 25.572 | 29.056 → 29.102 | 107.29 → 107.18 | 0.297495 → 0.297495 |
| V2_03_difficult | 24.470 → 22.210 | -9.2% | 21.822 → 22.387 | 25.518 → 26.102 | 106.86 → 106.81 | 0.879242 → 0.879242 |

六次最终回放的 `latency_misses` 和 `failure_reboots` 均为 0；输出位姿数、匹配位姿数
和接收图像数与基线逐序列一致。`max_output_gap_s` 仍为 0.10 s（difficult 为 0.15 s），
没有通过丢帧或降低后端频率换取加速。

原始数据位于：

- `output/optimization_20261019/baseline_reference/summary.json`
- `output/optimization_20261019/final/summary.json`
- `output/optimization_20261019/comparison_final.json`

## 保留的优化

### 1. OpenCV 单线程（主要收益）

该工作负载只有 752×480 单目图像、约 200 个特征，OpenCV 线程池的唤醒和同步开销
高于并行收益。节点新增 `opencv_threads` 参数，默认值为 1，launch 文件同步暴露该参数：

```bash
ros2 launch vins vins.launch.py opencv_threads:=1
```

隔离测量（关闭 `-march=native`，仅 V2_02_medium）显示，单线程相对 2 线程：
CPU 30.23 s → 28.11 s（-7.0%），P95 24.45 ms → 25.70 ms（+5.1%）。
最终表里的总收益还包含下面这些不改变算法的微优化。

### 2. 数据结构与缓冲复用

- `FeatureManager` 保留 `std::list` 的遍历顺序（从而保持求解参数顺序），增加
  `feature_id → list iterator` 哈希索引，删除原来的线性 `find_if` 查找。
- 光流 mask 的排序缓冲复用容量；过滤状态按常量引用传递，避免每帧复制 `status`。
- 归一化观测 map 在前后帧之间 `swap`，不再深拷贝。
- 没有轨迹可视化订阅者时不维护箭头历史、不绘制跟踪图。
- 回环 BRIEF 模板按配置路径缓存，描述子提取器复用以避免逐关键帧重建。

未降低 `max_cnt`、`freq`、光流反向检查、滑窗大小或求解迭代预算。

## 已尝试但拒绝的方案

### OpenCV 金字塔复用（已删除）

曾缓存 `buildOpticalFlowPyramid` 的结果供前向、反向和双目 LK 复用，但该 OpenCV
LK 调用路径无法实际消费预构建金字塔，额外构建成本无法被节省。三条序列 CPU 回归约
+19%–+21%（例如 medium 30.79 s → 37.0–37.3 s）。相关源码和测试已全部删除，
最终版本恢复原有 LK 调用方式。

### `-march=native`（默认关闭）

`-march=native` 在 easy 上仅有很小且不稳定的收益，却改变了浮点指令组合（FMA/向量宽度）。
V2_02_medium 在两次重复中均确定性发散：ATE 从 0.297 m 变为 22783.6 m，输出从
1127 变为 1128，CPU 因进入错误状态反而降到约 17.3 s。因此该选项现在默认 `OFF`，
只作为高级用户显式实验选项保留：

```bash
-DVINS_ENABLE_NATIVE_OPTIMIZATION=ON
```

不要分发 `-march=native` 编译的二进制；它既不可跨机器运行，也可能改变数值结果。
最终评测二进制使用默认的通用 x86-64 Release 代码生成。

### LTO（未保留）

`-flto=auto` 在 easy 上与 `-march=native` 的结果无差异（32.54 s vs 32.49 s），
没有可复现收益，最终构建显式关闭。

## 复现方法

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-up-to vins loop_fusion --executor sequential \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF
source install/setup.bash
colcon test --packages-select vins loop_fusion --event-handlers console_direct+
colcon test-result --verbose

python3 scripts/benchmark_euroc.py \
  --bags-root ~/vicon_room2/bags \
  --binary "$PWD/install/vins/lib/vins/vins_node" \
  --output "$PWD/output/performance" \
  --label final --repeats 2

python3 scripts/compare_benchmarks.py \
  output/performance/baseline/summary.json \
  output/performance/final/summary.json \
  --json-output output/performance/comparison.json
```

注意：

1. 必须在测量前把 `install/vins/lib/vins/vins_node` 复制成独立文件；它是符号链接，
   边编译边测试会读到不同二进制。
2. 对比相同配置、完整序列、回放速度和重复次数；测量期间不并发编译、运行另一个
   测量实例或 RViz。
3. `benchmark_euroc.py` 只启动 estimator，不启动 `loop_fusion` 或 RViz。CPU 秒是
   estimator 进程全部线程的用户态+内核态累计时间，不包含播放器和观察者。
4. `latency_misses` 表示里程计时间戳找不到对应图像到达时间；不能忽略非零值。
5. 回环 BRIEF 缓存目前只有单元测试，未纳入端到端 benchmark，不能据此声称回环整体
   加速比例。
6. `opencv_threads` 是 ROS 参数，需通过 launch 或 `--parameter opencv_threads:=1`
   传入；算法文件由 `config_file` 读取，其中任意字段不会自动成为节点参数。

## 未实施的后续方向

以下方向可能继续降低 CPU 或内存，但会改变线程/锁语义或优化问题结构，本轮未实施：

- 将 `inputIMU` 在缓冲锁内同步重放 IMU 的工作移出临界区，减少图像回调阻塞。
- 为回环关键帧数据库和历史地图增加有界裁剪策略。
- 重构边缘化中的 Schur 补与矩阵复制，减少大矩阵临时对象。
- 将后端状态提交与发布解耦，但需重新设计多线程可见性与失败回滚。

## ARM 设备验证

Orange Pi Zero 3 的部署与回放结果另见 [远程 ARM 验证](REMOTE_ARM_PERFORMANCE.md)。
本页的 x86 测量不能替代 ARM 实测，源码更新也不等同于设备已重建安装。
