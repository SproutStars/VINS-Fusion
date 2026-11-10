# ARM 前端模式与配置对比（2026-11-10）

已完成 **21 次完整 1× ARM 回放**。在本轮设备、标定和后端组合下，推荐独立配置 **`gftt_pyramid + 200 特征 + OpenCV 单线程 + A76 大核绑定`**。

三个 V2 序列各运行两次，金字塔方案的轨迹全部 8 列逐位一致，均无重启。样本有限；这是已测配置中的综合折中，不是所有相机/平台的全局最优，也不是每条序列最低误差。

## 测试条件

- 设备：`pinydrone@192.168.5.62`，aarch64，2×Cortex-A76 + 6×Cortex-A55，约 3.75 GiB RAM；CPU 6/7 为 A76。
- 完整冻结源码来自 `a0dfc4d` 加当前未提交 FAST 修复；相机模型/公共依赖也在独立工作区重建。
- 二进制 SHA256：`b7174d459f5e767e92930f30a03971591e8995e8a48f445c110f344589afc70a`。21 次运行均相同，退出码、配置全文/哈希、命令已核对。
- 固定：Schur EKF，warmup 10、`estimate_td=0`、`max_cnt=200`、`min_dist=20`、`freq=10`、双向 LK 与 RANSAC 开启、OpenCV 1 线程、估计器所有线程绑定 6/7。
- 首轮三配置只有检测器不同；150/120 特征测试仅改变 `max_cnt`。相机标定是 EuRoC V2 的 `cam0_pinhole.yaml`，光流/几何均使用原图坐标。
- 三模式×三序列首轮 9 次，金字塔 n150/n120 各三序列 6 次，GFTT/金字塔 n200 各三序列复测 6 次。全部串行，未与编译重叠；按序列轮换模式次序。
- CPU 秒仅估计器进程；CPU% 采用单核口径。延迟由图像接收到里程计到达测量。ATE 用 Vicon 时间插值、刚体对齐，不拟合尺度或时间偏移。
- `max output gap` 单独表示连续性；端点覆盖时长比例不是无空档覆盖，`poses/images` 也不是覆盖率。重启请求与 reset 入口分别统计，无可靠完成标记。

## 三模式首轮结果

ATE 为各自完整输出的 RMSE；不同输出集合还以相同时间戳交集复算，结果保存于 JSON。

| 序列 | 模式 | ATE m | 输出/匹配 | 最大空档 s | 重启请求/入口 |
|---|---|---:|---:|---:|---:|
| V2_01_easy | gftt | 0.2306 | 1083/1080 | 0.10 | 0/0 |
| V2_01_easy | fast_grid | 0.2034 | 1077/1074 | 0.10 | 0/0 |
| V2_01_easy | gftt_pyramid | 0.2748 | 1083/1080 | 0.10 | 0/0 |
| V2_02_medium | gftt | 1.0880 | 1133/1131 | 0.10 | 0/0 |
| V2_02_medium | fast_grid | 0.2955 | 1133/1131 | 0.10 | 0/0 |
| V2_02_medium | gftt_pyramid | 0.3472 | 1128/1126 | 0.10 | 0/0 |
| V2_03_difficult | gftt | 0.4882 | 1103/1102 | 0.15 | 0/0 |
| V2_03_difficult | fast_grid | 16.6532 | 1077/1076 | 1.60 | 2/2 |
| V2_03_difficult | gftt_pyramid | 0.3333 | 1101/1100 | 0.15 | 0/0 |

| 序列 | 模式 | CPU s | P50/P95/P99 ms | 峰值 RSS MiB |
|---|---|---:|---|---:|
| V2_01_easy | gftt | 82.75 | 34.3/40.3/46.3 | 99.38 |
| V2_01_easy | fast_grid | 80.93 | 33.7/42.9/53.4 | 103.84 |
| V2_01_easy | gftt_pyramid | 73.81 | 31.4/35.8/43.1 | 103.30 |
| V2_02_medium | gftt | 82.89 | 32.8/40.4/48.9 | 102.72 |
| V2_02_medium | fast_grid | 84.78 | 33.8/42.3/48.6 | 107.00 |
| V2_02_medium | gftt_pyramid | 75.50 | 30.0/37.1/48.6 | 97.69 |
| V2_03_difficult | gftt | 74.61 | 35.9/55.0/66.0 | 99.63 |
| V2_03_difficult | fast_grid | 77.31 | 37.0/55.9/70.5 | 104.05 |
| V2_03_difficult | gftt_pyramid | 66.32 | 31.2/48.5/61.5 | 103.40 |

三模式共同时间戳 ATE（GFTT / FAST / pyramid）：easy `0.2305 / 0.2034 / 0.2752 m`，medium `1.0850 / 0.2954 / 0.3472 m`，difficult `0.4896 / 16.6670 / 0.3355 m`。每次共同 ATE 的具体交集不同，不得将三模式交集替作两模式比较。

## 减少特征数的代价

| 特征数 | 序列 | ATE m | CPU s | P95 ms | 最大空档 s | 重启请求 |
|---:|---|---:|---:|---:|---:|---:|
| 150 | V2_01_easy | 3.4351 | 71.40 | 35.5 | 1.20 | 2 |
| 150 | V2_02_medium | 0.3978 | 73.71 | 36.3 | 0.10 | 0 |
| 150 | V2_03_difficult | 25.5672 | 65.30 | 46.0 | 5.80 | 1 |
| 120 | V2_01_easy | 0.3071 | 74.06 | 34.7 | 0.10 | 0 |
| 120 | V2_02_medium | 0.4239 | 72.59 | 35.5 | 0.10 | 0 |
| 120 | V2_03_difficult | 0.6815 | 65.10 | 43.7 | 0.15 | 0 |

n150 的 easy/difficult 出现视觉饥饿重启；n120 虽无重启，三序列误差均比 n200 更高，difficult 约翻倍。CPU 只小幅节省，easy n120 甚至略增，因此不选择 150/120。

## 重复验证

| 序列 | 模式 | 首轮/复测 ATE m | 首轮/复测 CPU s | 首轮/复测 P95 ms | 复测重启请求 |
|---|---|---|---|---|---:|
| V2_01_easy | gftt | 0.2306/0.2316 | 82.75/82.78 | 40.3/40.2 | 0 |
| V2_01_easy | gftt_pyramid | 0.2748/0.2748 | 73.81/73.99 | 35.8/35.7 | 0 |
| V2_02_medium | gftt | 1.0880/1.0880 | 82.89/82.43 | 40.4/40.3 | 0 |
| V2_02_medium | gftt_pyramid | 0.3472/0.3472 | 75.50/75.54 | 37.1/36.8 | 0 |
| V2_03_difficult | gftt | 0.4882/13.5721 | 74.61/75.42 | 55.0/55.1 | 1 |
| V2_03_difficult | gftt_pyramid | 0.3333/0.3333 | 66.32/66.23 | 48.5/47.4 | 0 |

- 金字塔 n200 三序列两次轨迹逐位一致、无重启。相对 GFTT，CPU 时间约降低 8%–12%，P95 约降低 8%–14%；easy ATE 多约 4.4 cm，medium/difficult 则更稳定。
- GFTT medium 的约 1.088 m 可重复；difficult 首轮 0.488 m 后复测 13.572 m、1 次重启、5.9 s 空档，不能只保留成功样本或平均掩盖失败。原因未定位，不单独归因于 ARM SIMD。
- 最新 FAST 是带 GFTT 恢复的 hybrid，不能引用旧 FAST-only 性能。其 difficult 首轮失败，不作为本轮通用 ARM 推荐。

## 性能来源、内存与热状态

- 逐窗口耗时按处理帧数加权，核对 `sum(frames)==total_frames`。pyramid 相对 GFTT 检测阶段约少 5.2–8.7 ms/frame；金字塔构建增加约 1.8–1.9 ms，跟踪总时间仍降低约 4.6–6.3 ms。收益主要来自较低检测开销。
- n150/n120 的检测开销并未同比下降；较少光流点带来的收益不足以抵消精度和连续性代价。
- RSS 是进程峰值，不是导数缓存大小。pyramid 相对 GFTT 约变化 −5～+4 MiB，方向随序列变化；不能称作固定更省内存。
- 回放温度约 75–91°C，采样频率受动态调频影响。保存每次开始/峰值/结束温度及频率；没有直接 throttle 标记，不把温度/频率相关性当作因果。重复及轮换顺序能减小偏差，但样本仅两次，不作统计显著声明。
- 所有 21 次运行观察者收到序列完整图像数，跟踪队列无溢出；这不证明所有后端帧都被接受。精度仍须检查 ATE、输出数量、空档、视觉更新与重启日志。

## 可直接使用的配置

新增独立文件：`src/vins_estimator/config/euroc/euroc_mono_imu_arm_balanced_config.yaml`。相对实测 pyramid n200 配置，仅把 `cpu_affinity_enabled` 从 false 改为 true；其余算法参数完全一致。没有覆盖现有 mono YAML。

```yaml
feature_detector: "gftt_pyramid"
max_cnt: 200
min_dist: 20
flow_back: 1
ransac_outlier_rejection: 1
backend_type: "schur_ekf"
schur_warmup_frames: 10
estimate_td: 0
cpu_affinity_enabled: true
cpu_affinity_cores: [6, 7]
```

```bash
ros2 launch vins vins.launch.py \
  config_file:=/home/sprout_stars/rmcode/VINS-Fusion/src/vins_estimator/config/euroc/euroc_mono_imu_arm_balanced_config.yaml \
  opencv_threads:=1
```

上面是本地源码路径；远端需将 `config_file` 改为该文件在设备上的实际路径。该配置使用 EuRoC V2 标定及本机 CPU ID，其他相机必须替换标定，其他 ARM 板必须先核对核心编号。通过 launch 才应用 YAML 中的绑核设置；直接运行节点需另用 taskset。

## 验证与适用范围

- ARM：前端 25、Schur 更新 27、EKF 状态 31，共 83 个 C++ 用例通过（3 个 CTest 程序）。本地 60 项 Python 测试通过，包含配置读取、核心绑定及只读比较工具测试。
- 21 次完整 1× 回放、同一二进制及原始配置/命令/哈希通过核对；没有遗留 ROS、编译或采样进程。
- 只验证一台 ARM、EuRoC V2 三条单目 IMU 序列。GFTT/pyramid n200 各两次，其余只筛选一次，不能保证其他光照、运动、相机或长期运行同样稳定。
- 本轮选择配置，不更改算法、后端门限、生产默认或用户当前配置。若控制侧精度指标不同，应按同样方法重新验收。

证据：`docs/ARM_FRONTEND_COMPARISON_RESULTS_20261110.json`；原始输出 `output/arm_frontends_20261110/{results,evidence}`；配置差异 `profile_semantic_diff.json`；哈希审计 `final_manifest_audit.json`；温度 `thermal_summary_final.json`。
