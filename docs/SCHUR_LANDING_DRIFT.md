# Schur EKF 降落后静止漂移

## 验证方法

数据为 `~/vicon_room2/bags/{V2_01_easy,V2_02_medium,V2_03_difficult}`，完整序列、1 倍速、单独 ROS domain 173、OpenCV 单线程。使用仓库 EuRoC 单目 IMU 配置，显式指定 `backend_type:=schur_ekf`。估计器订阅相机和 IMU；Vicon 只用于离线评估，不输入估计器。

全程 ATE 沿用 `scripts/benchmark_euroc.py` 的 SE(3) 刚体对齐，不拟合尺度或时差。新增 `scripts/analyze_landing_drift.py` 只用结束前 10 秒以外的数据拟合坐标系变换，然后计算末尾 10/5/3/1/0.5 秒的误差。尾段相对末点误差为 `||(p_est,end-p_est,start)-(p_gt,end-p_gt,start)||`，可显示落地后的新增漂移，不用尾部重新拟合。脚本也记录真实结束时间与输出结束时间之差、最大输出间隔，避免把提前停止输出误当作改善。

原始 IMU 中的降落冲击峰值：easy 35.26 m/s²（距 IMU 结束 5.385 s），medium 22.78 m/s²（3.075 s），difficult 32.84 m/s²（2.395 s）。Vicon 显示 easy 最后 3 秒、另两组最后 1 秒的各轴位置变化范围小于约 1.5 mm。

## 修改前结果

| 序列 | 全程 ATE RMSE (m) | 最后 1 秒相对末点误差 (m) | 最后 3 秒相对末点误差 (m) |
| --- | ---: | ---: | ---: |
| V2_01_easy | 0.2102 | 0.0186 | 0.0596 |
| V2_02_medium | 0.2733 | 0.0328 | 0.0395 |
| V2_03_difficult | 0.3178 | 0.0811 | 0.0856 |

`difficult` 落地后出现 `tracks=120 used=0 rejected=120` 和 `no gated tracks`，之后进入 deep starvation。末帧估计速度约 0.103 m/s，而真实设备已经静止。easy 和 medium 的末帧速度分别约 0.044 和 0.051 m/s。

## 卡方检验单项对照

点消元后的 gate 改为显式左零空间投影：`r0=Nᵀr`、`H0=NᵀHx`，`S0=H0PH0ᵀ+I`，自由度 `2N-rank(Hf)`。该修正避免旧式 `HxPHxᵀ+I-Hf(HfᵀHf)⁻¹Hfᵀ` 在不可观测方向上的奇异性，但仍保留三角化/点块退化拒绝。

完整三组单项对照末 1 秒误差分别为 0.0166、0.0327、0.0805 m；全程 ATE 为 0.2102、0.2735、0.3155 m。该修正单独不能解决降落后漂移，不能把它描述为有效的落地修复。

## 落地静止约束

新增 `LandingStaticDetector` 使用图像与 IMU 的传感器时间。启用条件同时包括：最近 6 秒出现原始加速度范数 ≥20 m/s²；至少 30 个跨帧匹配特征、至少 70% 特征运动很小且覆盖至少 3 个图像象限；去偏 IMU 在 0.3 秒窗口中低动态；估计速度 ≤0.25 m/s；这些条件持续 0.3 秒。光流阈值为每 0.05 秒 0.25 像素，实际位移按帧间隔归一化。

启用后把 `v=0` 作为标准差 0.005 m/s 的三维伪观测，使用完整状态及交叉协方差进行 Kalman 更新和 Joseph 协方差更新。位置、姿态与偏置按滤波增益一并修正，不直接写死位置或速度。校正后的图像状态通过已有同步路径进入 IMU 实时预测；只有图像提供新静止证据时才更新。成功的零速度更新抑制饥饿重启；检测到图像运动、IMU 动态、过高速度或数据断档时退出，重新进入需要再次累计静止证据。

该功能不需要 Vicon、不识别数据集名称或录制时刻。近期冲击和多传感器门控降低误判概率，但不能从根本上消除视觉/惯性观测歧义：远景或错误跟踪配合低估计速度仍可能混淆缓慢运动。阈值需针对其他相机、分辨率和振动水平验证。

通用代码和缺少配置项时默认关闭，EuRoC 配置启用 `schur_landing_zupt_enable: true`，可用启动参数 `-p schur_landing_zupt_enable:=false` 禁用。其他 `schur_landing_zupt_*` 阈值在配置文件中设置。

## 最终结果（2026-11-05）

| 序列 | 最后 0.5s 新增误差 (mm)，前 → 后 | 最后 1s 新增误差 (mm)，前 → 后 | 末帧速度 (m/s)，前 → 后 | 全程 ATE RMSE (m)，前 → 后 |
| --- | ---: | ---: | ---: | ---: |
| V2_01_easy | 9.79 → 0.72 | 18.63 → 1.05 | 0.0444 → 0.0058 | 0.2102 → 0.2100 |
| V2_02_medium | 16.82 → 0.71 | 32.76 → 0.41 | 0.0511 → 0.0041 | 0.2733 → 0.2735 |
| V2_03_difficult | 39.05 → 1.20 | 81.09 → 35.66 | 0.1035 → 0.0043 | 0.3178 → 0.3150 |

“最后 0.5s”统计到的图像位姿跨度约为 0.4s（10 Hz 后端），最后 1s 约为 0.9s。上述误差是尾段相对位移误差，不代表绝对定位误差。三组最终各启用一次，触发附近 Vicon 各轴变化均小于 1.5mm；无退出抖动、无拒绝 ZUPT 更新、无重启。匹配位姿数、输出持续时间、最大输出间隔与修改前一致。difficult 在结束前约 0.61s 开始约束，最后 1s 含首次校正回调，因此仍有 35.7mm 相对末点误差；稳定后的最后半秒为 1.2mm。

机器可读的配置全文、二进制/源码哈希、各轮完整与尾段指标见 [SCHUR_LANDING_RESULTS.json](SCHUR_LANDING_RESULTS.json)。最终二进制 SHA-256 为 `d857f7219c4223d51f09c519ca30bd41aae00dc1369a2f6c7c6327905a8fd449`。回放只比较位置/速度与输出连续性，未验证闭环控制效果。

## 复现命令

在仓库根目录运行，先保留修改前二进制，再分别回放修改前/后程序：

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 scripts/benchmark_euroc.py \
  --binary output/landing_impact_20261105/binaries/vins_before \
  --output output/landing_impact_20261105 --label before \
  --datasets V2_03_difficult V2_02_medium V2_01_easy \
  --parameter backend_type:=schur_ekf --parameter opencv_threads:=1
python3 scripts/benchmark_euroc.py \
  --output output/landing_impact_20261105 --label after \
  --datasets V2_03_difficult V2_02_medium V2_01_easy \
  --parameter backend_type:=schur_ekf --parameter opencv_threads:=1
python3 scripts/analyze_landing_drift.py \
  --root output/landing_impact_20261105 --labels before after --plot
```

首次零速度更新会通过状态交叉协方差修正位置，可能出现短暂回调。因此必须同时观察末点误差、最大偏移、路径长度和速度；末点误差变小本身不足以证明轨迹始终稳定。本方法针对冲击后静止的持续漂移，不消除飞行期间积累的位置/航向误差，也不约束尚未静止的冲击过程。

原始日志、轨迹、二进制及配置清单保存在 `output/landing_impact_20261105/`（Git 忽略）。


## 回归测试

最终本地构建的 12 个 CTest 程序共 130 个 C++ 用例通过，`scripts/tests` 的 27 个 Python 用例通过。新增覆盖包括：无冲击不触发、冲击后持续运动不触发、空间覆盖与 IMU 稳定性、重复 IMU 样本、短窗口/采样缺口、重新运动退出后重新计时、时间阈值浮点边界、默认关闭、重置清理、Joseph 更新保持协方差半正定，以及非零残差的正确卡方检验。

本次回放在本地 x86 环境完成，未进行 ARM 实机回归。完整 bag 只有最后一段落地；落地后重新起飞的行为由合成测试覆盖，未由这些 bag 验证。
