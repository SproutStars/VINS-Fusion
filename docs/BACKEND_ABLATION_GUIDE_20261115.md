# QR、IMU 协方差批量传播与独立观测预算验证（2026-11-15）

本轮在基础版本 `ee68322` 上分别实现三项可选优化，以 V1 三条、machine_hall 五条完整序列逐项比较。所有新开关默认关闭，原有算法配置保持不变；完整数值表由 `scripts/summarize_backend_ablation.py` 从原始轨迹重算，见 [逐次结果](BACKEND_ABLATION_20261115.md) 和 [JSON](BACKEND_ABLATION_RESULTS_20261115.json)。全部计划回放已结束；结论与适用边界见第5节。

## 1. 实现范围

| 项目 | 实现 | 保留的约束 |
|---|---|---|
| QR | 用 Givens 旋转同时变换每 track 的 `Hf/Hx/r`，保留消去三维点后的残差行，不构造 full-U | 点块秩/条件门、chi² 自由度不变；3×3 三角因子仍做小 SVD 秩检查；全局 eigen/Joseph 更新不变 |
| IMU batch | 每采样仍计算原 RK4 均值、FEJ、Phi 和 Qd，累计图像间隔的 Phi/Q 后一次传播 P | covariance 访问、clone 操作和实时 predictor 重锚前 flush；实时 predictor 仍逐 IMU sample 更新协方差 |
| 观测预算 | 前端特征数、后端 track 上限、每次更新历史观测总数三者独立 | 只选完整 track，左右目观测不截断；放不下的非 margin track 延后，到期或永远超预算的 track 丢弃但不记作 outlier |

批量传播使用 `A_next = Phi_i A`、`Q_next = Phi_i Q Phi_iᵀ + Qd_i`，最终 `P_next = A_full P A_fullᵀ + Q_full`。采样间隔变化、噪声膨胀与图像边界部分采样都在每步 Phi/Qd 中处理。浮点运算顺序与逐步更新不同，数学等价不保证长序列逐位一致。

预算的正数模式在三角化前选入完整 track，能减少三角化和门控计算；也会改变信息集合与轨迹。`0` 走原选择顺序和生命周期，不能将“更少观测”表述成精度无损。

## 2. 开关与复现

三个独立节点参数也可写入算法 YAML：

```yaml
schur_nullspace_method: "svd"       # "qr" 开启 QR
schur_batch_imu_covariance: false   # true 开启图像时间滤波器批量传播
schur_max_update_observations: 0   # 0 不限；本轮筛选 240/480
```

直接通过节点参数逐项开启，避免改动同一份基准配置：

```bash
# 假设已 source ROS 和匹配源码的 install
ros2 run vins vins_node --ros-args \
  -p config_file:=/absolute/path/to/stereo_config.yaml \
  -p schur_nullspace_method:=qr

# 独立 batch：只传此开关，其余保持默认
# -p schur_batch_imu_covariance:=true

# 独立观测预算：前端 max_cnt 和 schur_max_features 保持基线值
# -p schur_max_update_observations:=240
```

换用旧 install 的二进制不会自动包含本轮实现；需从当前源码构建并记录实际 ELF SHA256。完整实验位于本地 `output/qr_imu_budget_20261115/`，ARM 独立目录 `/home/pinydrone/vins_qr_imu_budget_20261115/`，没有替换原部署。

## 3. 测试与口径

- 本地候选干净构建通过 **13/13 CTest、172 个 gtest 用例**；Python **80 项通过**。新增测试覆盖 QR/SVD 非零残差与后验、同帧双目及退化秩、批量/逐步多采样与 clone 交叉块、reset/copy/reanchor、完整轨迹预算及延后/到期计数。
- 原版本与候选关闭全部开关的轨迹，在全部 8 条本地序列上逐元素相同。MH_02 的原有重启也被相同复现，因此不能因新代码而忽略基线自身的失稳。
- 本地完整回放采用 **2×速率**，用于固定输入序列的算法与本地负载对照；ARM 采用 **1×速率、核心6/7、OpenCV单线程**。CPU 只比较相同平台、速率和真值配置；本地延迟不能当成 ARM 延迟。
- 固定前端120特征、Schur最多120 tracks、W=5、后端10 Hz，固定已知 EuRoC `td=0`，使用 V1/MH 相机标定；不开 RViz、loop 或 global fusion。
- V1 使用本组 Vicon `T_BS`，MH 使用本组 Leica `T_BS`。通过 `p_sensor = p_body + R_est t_BS` 将估计转换到真实被跟踪原点，再做位置插值、刚体对齐 ATE，不拟合尺度或时间偏移。Leica 只有位置，不生成虚构姿态真值。
- **实际输出 gap** 从全部 `trajectory.csv` 时间戳计算；`metrics.max_output_gap_s` 是真值有效区间匹配后的 gap，MH 中可因 Leica 缺测而很大。报告将二者分开，不把真值缺测当作里程计断流。
- 结果中的5%误差线只是本轮工程筛选标记，并非用户规定的验收标准；绝对误差、重启和实际 gap 需要单独看。
- 一次外部 SIGINT 中断的 QR/MH_05 样本已保留并排除，完整重跑另记；中断不能当作算法崩溃。

## 4. 数学与 API 边界

`covariance() const` 会 lazy flush mutable 缓存，`EkfState` 仍依赖现有单线程状态所有权或外部锁；不能并发只读同一状态，也不能把取出的 P 引用跨 `propagate()` 长期保留。实时 predictor 明确关闭 batching，避免向控制端发布尚未更新的协方差。

QR 和 SVD 的 chi² 投影基可以不同，真正应该比较的是投影后的统计量、后验和接受集合。恰在门限附近的观测可能因舍入不同而改变门控，从而在真实长序列中产生不同轨迹；不能由微基准 checksum 相同推出所有硬件逐位相同。

观测预算到期丢弃是信息损失而非 outlier。预算过小可导致少于最小有效特征/观测门槛，引发饥饿重启；增加预算也不保证误差单调变好，因为完整 track 的选入和到期集合会改变。临时 helper 的 vector 压缩接口不用于 live FeatureHistory；产品使用选择计划并显式管理生命周期。

独立审查未发现当前产品路径中漏 flush、QR 方向错误或已应用观测重复消费的明确 bug；该结论不替代实际回放验证。审查证据位于 `output/qr_imu_budget_20261115/evidence/implementation_review.md`。


## 5. 实测结果与建议

有效样本：本地 **44 次完整回放**，ARM **17 次完整性能回放**，另有ARM一条30秒debug诊断（不参与性能比较）。本地一条外部SIGINT中断样本保留但排除。原配置未改、产品默认不开启三项优化；本次没有提交或覆盖原部署。

### 5.1 本地八序列精度（2×回放）

下表为各自完整输出的传感器原点ATE（m），共同时间戳ATE与GT匹配覆盖在详细JSON中逐对列出。

| 序列 | 关闭优化 | QR | IMU batch | obs240 | 重启请求：关闭/obs240 |
|---|---:|---:|---:|---:|---:|
| V1_01_easy | 0.12955 | 0.12955 | 0.12955 | 0.16487 | 0/0 |
| V1_02_medium | 0.15115 | 0.15115 | 0.15115 | 0.15707 | 0/0 |
| V1_03_difficult | 0.24236 | 0.24236 | 0.24236 | 0.26708 | 0/0 |
| MH_01_easy | 0.18070 | 0.18070 | 0.18070 | 0.17165 | 0/0 |
| MH_02_easy | 6.77604 | 6.77604 | 6.77604 | 3.50060 | 1/2 |
| MH_03_medium | 0.56238 | 0.56238 | 0.56238 | 0.55421 | 0/0 |
| MH_04_difficult | 0.58275 | 0.58275 | 0.58275 | 0.60221 | 0/0 |
| MH_05_difficult | 0.75244 | 0.75244 | 0.75244 | 0.76945 | 0/0 |

- **默认兼容性：** 候选关闭全部开关与原二进制，在8条序列的全部8列轨迹逐元素一致。
- **QR：** 本地8条序列轨迹均逐位一致，无新增重启；8条累计CPU由152.26 s到146.15 s，约−4.0%，单次筛选、逐序列存在波动。
- **IMU batch：** 本地8条最大ATE绝对差约`1.2e-8 m`、最大位置分量差约`1.6e-6 m`；重启数量相同。累计CPU148.38 s，约−2.5%；V1_01/V1_03/MH_01单次CPU反而略增，不能宣称普遍提速。
- **obs240：** V1_01/V1_03误差约增加27.3%/10.2%；MH_02误差下降但重启由1次增至2次，不能当作鲁棒精度改善。8条性能和接受集合均有改变，不推荐统一默认预算240。
- **obs480：** 仅在V1_02/MH_05筛选，ATE约0.16742/0.75104 m；V1_02相对关闭优化增加约10.8%。预算更大也不保证信息集合/误差单调改进。
- **QR+batch：** 两条组合筛选误差约0.15115/0.75244 m，CPU12.72/15.47 s，接近batch单项；只覆盖两条，不扩大声称8条组合精度已验证。

MH_02所有关闭/QR/batch路径均有1次重启，ATE约6.78 m、真实输出gap约1.2 s，当前初始化/观测/后端组合在此序列仍不可靠。优化不等于解决了原有算法失稳。

### 5.2 ARM：收益被前端与固定开销覆盖

设备仍为2×A76+6×A55，固定A76核心6/7、OpenCV单线程、1×回放；每组均使用同一份候选ELF，SHA256为 `2a61a4f7a2d9c28064de95454cc93fa9a3dccfde2bfcba33c6c202367cd4facb`。13/13 ARM CTest通过，未启用native/fast-math/LTO。

| 模式 | V1_02 CPU s / P95 ms | MH_05 CPU s / P95 ms |
|---|---:|---:|
| 关闭优化（首轮） | 63.52 / 45.89 | 82.86 / 49.45 |
| QR | 63.41 / 45.62 | 82.54 / 48.96 |
| IMU batch | 63.35 / 45.97 | 82.56 / 49.51 |
| obs240 | 63.37 / 45.59 | 82.40 / 49.51 |
| obs480 | 63.51 / 45.75 | 82.84 / 49.64 |
| QR+batch | 63.38、63.45 / 46.20、46.17 | 82.20 / 49.20 |

以上精确值以详细结果JSON为准；单项端到端CPU差小于1%，没有稳定的CPU/P95/RSS收益证据。V1关闭优化复测为63.64/63.47 s、P95为45.19/45.98 ms，与组合的差异处于同量级；峰值RSS也在约98.4–98.6 MiB波动，不能宣称显著降内存。

V1首轮关闭优化ATE约0.19419 m，但复测关闭优化、QR、batch、组合都约0.172394 m。日志表明首轮Ceres初始化触发壁钟预算时只完成约2次迭代，后续约8次，不能把该误差下降归因于后端优化。MH_05各QR/batch/组合约0.642415 m；obs240约0.628083 m，obs480约0.649556 m。所有ARM完整样本均无重启、实际输出gap约0.1 s；MH的2.8 s GT-matched gap来自Leica缺测。

三轮ARM微基准中，QR更新从约6006–6030 us降到5832–5884 us，约−2%至−3%；IMU batch在5/10 clones分别约1.62–1.63×/2.42–2.47×，协方差相对差约`3e-15`。它们是生产内核局部收益，不能替代端到端结果。观测240/480能大幅减少合成更新工作量，但真实整机由前端跟踪和其他开销主导。

### 5.3 预算诊断与最终选择

ARM单独的30秒obs240诊断记录458个Schur step，实际使用观测最大240、超预算0；348步更新成功，selected/deferred/expired轨迹累计计数为15228/6196/2012。诊断只证明预算确实约束进入更新的观测与生命周期，不证明其精度优于原模式；debug日志/额外矩阵诊断开销使其不能参与性能排名。

**建议保持现有默认：SVD、逐步IMU协方差、观测预算0。** QR可作为最先启用的实验选项；batch适合后端占比更高、clone窗口更大的条件下单独复验；观测预算仅在明确接受精度与鲁棒性损失时使用。没有依据为了本设备提速而同时默认打开三项。

下一步若目标是明显降低ARM端到端延迟，应继续针对前端光流/图像金字塔和线程调度，而非仅替换小型后端代数内核。MH_02的初始化与视觉饥饿需要单独修复，并以不改变其他序列行为的回归验证。

## 6. 证据与清理

完整本地结果：`output/qr_imu_budget_20261115/evidence/results/runs.jsonl`；ARM原始结果及审计：`output/qr_imu_budget_20261115/arm_results/`、`arm_evidence/`；远端根：`/home/pinydrone/vins_qr_imu_budget_20261115/`。全部旧部署、bags和原始失败样本保留。ARM最终审计确认无估计器、bag播放、编译、采样残留；governor保持`schedutil`，性能回放温度不超过约62°C。

![Local backend ablation](/home/sprout_stars/rmcode/VINS-Fusion/output/qr_imu_budget_20261115/figures/ablation_local.png)
