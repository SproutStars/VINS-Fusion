# Schur 补 EKF 后端（`backend_type: schur_ekf`）

本文描述 VINS-Fusion 中新增的、面向 ARM 平台的后端。它参考
`open_vins`（OpenVINS MSCKF）与 `SchurVINS` 的状态管理与更新结构，把默认的
Ceres 滑窗非线性最小二乘后端替换为**紧耦合误差状态 EKF + 逐特征 Schur 消元**。

> 边界声明：默认后端仍然是 `ceres`。新后端是 **opt-in**（配置里显式写
> `backend_type: "schur_ekf"` 才启用），失败时会回退到受控重启路径，不影响
> 现有默认行为。

---

## 1. 动机：为什么在 ARM 上换后端

默认后端每个图像帧都要在 Ceres 里求解一个包含滑窗内所有位姿、速度、偏置、
逆深度和全部观测的稀疏 NLS 问题。在 Orange Pi Zero 3（4×Cortex-A53）上实测
（`docs/REMOTE_ARM_PERFORMANCE.md`、`docs/ARM_FAST_GRID_EVALUATION.md`）：

- 单帧 p50 延迟约 380–425 ms，p95 约 590–856 ms，远超 20 Hz 图像周期；
- estimator 进程 CPU 约 158–187%（单核百分比），4 核平台持续高占用；
- 峰值 RSS 约 100–104 MiB。

EKF 后端的计算量结构不同：传播是固定 15×15（+clone）的矩阵运算，视觉更新
是**逐特征小规模**的 Schur 消元（3×3 特征点块 + 状态块），单帧成本与特征数
近似线性，且没有迭代求解、没有反复的雅可比重建，也不需要为每个特征分配
Ceres 残差块。

---

## 2. 状态与约定

### 2.1 误差状态布局

活动 IMU 状态 15 维（右乘扰动，body 系）：

| 索引 | 含义 |
|---:|---|
| 0–2 | `dtheta`（姿态误差，`q_true = q_hat ⊗ dq`, `dq = [1, dtheta/2]`） |
| 3–5 | `dp` |
| 6–8 | `dv` |
| 9–11 | `dba` |
| 12–14 | `dbg` |

每个 clone（一帧一个，最旧在前）追加 6 维：`dtheta_i`、`dp_i`。总维度
`15 + 6 * num_clones`。四元数为 Hamilton 约定 `[w,x,y,z]`，表示 body→world；
重力 `g_w = (0, 0, -|g|)`。

### 2.2 FEJ（First-Estimate Jacobian）

- 传播时雅可比在**区间起点**的 FEJ 值上线性化，随后把 `imu_fej()` 前推到传播
  后的均值，与 OpenVINS `Propagator::propagate()` 一致。
- clone 的 FEJ 位姿在**增广那一刻冻结**，之后不再更新
  （`refresh_imu_fej()` 只刷新活动块）。
- 视觉雅可比使用 FEJ 值，残差使用当前值。这是保证可观性/一致性的关键：
  如果雅可比跟着当前估计漂移，会引入虚假的可观方向，典型症状是速度通道被
  系统性带偏。

### 2.3 Clone 窗口

顺序为 **增广 → 视觉更新 → 裁剪**（与 OpenVINS / SchurVINS 相同）：

1. 当前帧位姿作为最新 clone 增广进协方差（新 clone 与活动位姿完全相关）；
2. 用"本帧丢失的特征"和"最旧 clone 上的特征"做一次 MSCKF 更新；
3. 视觉更新结束后才裁掉最旧 clone。

裁剪方式是**直接删除对应行列**（`P(keep, keep)`），与 OpenVINS
`StateHelper::marginalize()` 一致，**不做**条件 Schur 补——条件补会把剩余协方差
人为缩小，导致后续更新过度自信。

---

## 3. 视觉更新（Schur 消元）

对每个被选中的 track：

1. 用窗口内所有观测三角化 3D 点（Gauss–Newton 迭代），并做卡方外点门控；
2. 构建观测对 `[dtheta_i, dp_i, ..., dtheta_j, dp_j, ...]` 与 3D 点的雅可比；
3. 对 3D 点做 Schur 消元，得到只含状态的压缩信息
   （等价于 `(P^-1 + A) dx = b`，特征点块被消掉）；
4. 用平方根信息形式更新，避免显式求逆；协方差用 Joseph 形式更新保证 PSD。

### 3.1 Track 选择（避免重复计量）

同一个 track 只能被消费一次。选择规则：

- **窗口未满**（`num_clones() <= window_size`）：只取**本帧丢失**的 track，
  `marginalized_count = 0`。此时没有 clone 会被裁掉，用最旧 clone 的 track 会
  在后续帧被重复使用。
- **窗口已满**：额外允许"最旧观测落在即将被裁掉的 clone 上"的 track
  （OpenVINS `VioManager.cpp:366-369`）。

只有真正进入更新的 track 才被消费；被拒绝的 track 保留在窗口内等待重试，
其生命周期由窗口裁剪兜底。

---

## 4. 数值保护与回退

| 保护 | 参数 | 行为 |
|---|---|---|
| 帧内特征数下限 | `schur_min_features` | 低于该值的帧直接跳过 |
| 更新观测数下限 | `schur_min_update_observations` | 有效观测不足时不更新 |
| 信息矩阵秩揭示 | `schur_info_rank_rtol` | 丢弃 gauge/尺度零空间 |
| 负特征值容差 | `schur_info_indefinite_tol` | 显著负模态则拒绝该帧 |
| 条件数上限 | `schur_condition_limit` | 超限拒绝该帧（0 关闭） |
| 连续失败重启（次要） | `schur_max_consecutive_failures` | 连续 N 帧无更新则受控重启 |
| 时间饥饿重启（主要） | `schur_max_no_update_seconds` | 距上次成功更新超过该秒数则受控重启（0 关闭） |
| 饥饿期过程噪声放大 | `schur_starvation_seconds` / `schur_starvation_noise_scale` | 饥饿超过前者后，每次 IMU 递推把 Q 乘以后者 |

此外，`is_positive_definite` / `is_positive_semidefinite` / `symmetrize` 在每次
更新后校验协方差；`schurStateFinite()` 检查状态有限性。

---

## 5. 参数

全部参数位于 `src/vins_estimator/config/euroc/euroc_mono_imu_config.yaml` 的
`estimator` 段，均有默认值，缺省即保持 Ceres 行为。

| 参数 | 默认 | 含义 |
|---|---:|---|
| `backend_type` | `"ceres"` | `"ceres"` 或 `"schur_ekf"` |
| `schur_window_size` | 5 | 协方差里保留的 clone 数 |
| `schur_warmup_frames` | 0 | 初始化后用 Ceres 再解多少帧再切 EKF |
| `schur_max_features` | 120 | 单帧最多参与更新的 track 数 |
| `schur_min_observations` | 2 | 一个 track 进入更新的最少观测数 |
| `schur_min_features` | 12 | 帧内特征数下限 |
| `schur_min_update_observations` | 12 | 一次更新所需最少有效观测 |
| `schur_pixel_noise` | 1.5 | 归一化坐标观测噪声（像素） |
| `schur_chi2_multiplier` | 5.0 | 卡方外点门控倍数 |
| `schur_triangulation_iterations` | 5 | 三角化迭代次数 |
| `schur_info_rank_rtol` | 1e-10 | 秩揭示相对阈值 |
| `schur_info_indefinite_tol` | 1e-6 | 负模态容差 |
| `schur_point_block_rtol` | 1e-8 | 特征点块相对阈值 |
| `schur_condition_limit` | 1e6 | 条件数上限 |
| `schur_max_consecutive_failures` | 60 | 连续失败重启阈值（**次要兜底**，按帧数；受重启冷却抑制） |
| `schur_max_no_update_seconds` | 3.0 | 时间饥饿重启阈值（秒，**主要防线**，`<=0` 关闭） |
| `schur_restart_cooldown_seconds` | 8.0 | 两次受控重启之间的最小传感器时间间隔（`>=0`，`0` 关闭） |
| `schur_starvation_seconds` | 0.25 | 超过该秒数无成功更新即进入"饥饿"（`<=0` 关闭） |
| `schur_starvation_noise_scale` | 3.0 | 饥饿期每次 IMU 递推的 Q 放大倍数（`>=1.0`） |
| `schur_starvation_deep_seconds` | 1.0 | 超过该秒数进入"深度饥饿"，Q 倍数升级（`<=0` 关闭） |
| `schur_starvation_deep_noise_scale` | 10.0 | 深度饥饿期的 Q 放大倍数（`>=1.0`） |
| `schur_starvation_soft_restart` | false | 饥饿重启是否走**软重启**（保留 IMU/EKF 状态、不重初始化；见 5.1） |
| `schur_soft_restart_max_seconds` | 10.0 | 软重启后若该秒数内仍无成功视觉更新，升级为完整受控重启（`<=0` 关闭升级） |
| `schur_starvation_weak_update` | false | 饥饿期是否启用 **P6 弱更新**（放宽门控 + 对接受观测降权；见 5.1） |
| `schur_weak_update_max_scale` | 10.0 | P6 弱更新的最大 scale（finite 且 `>= 1.0`）：门控阈值 ×scale、观测信息 ÷scale² |
| `schur_init_pos_sigma` | 0.5 | 初始位置标准差 |
| `schur_init_vel_sigma` | 0.5 | 初始速度标准差 |
| `schur_init_rot_sigma` | 0.05 | 初始姿态标准差 |
| `schur_init_ba_sigma` | 0.5 | 初始加计偏置标准差 |
| `schur_init_bg_sigma` | 0.05 | 初始陀螺偏置标准差 |
| `schur_acc_noise` | 2.0e-3 | 加计连续时间噪声密度（unit²/Hz，OpenVINS EuRoC 值） |
| `schur_gyr_noise` | 1.6968e-4 | 陀螺连续时间噪声密度（unit²/Hz，OpenVINS EuRoC 值） |
| `schur_acc_bias_noise` | 3.0e-3 | 加计偏置随机游走密度 |
| `schur_gyr_bias_noise` | 1.9393e-5 | 陀螺偏置随机游走密度 |

初始协方差按上述 σ **放大**，因为切换时刻的偏置/速度来自初始化阶段，精度粗糙；
放大后由视觉观测自行收敛。

`schur_*_noise` 与 `acc_n`/`gyr_n`/`acc_w`/`gyr_w` **不是**同一套约定：后者由 VINS
按每采样噪声（经 `IntegrationBase` 的 V 矩阵、乘 `0.5*dt`）使用，若直接当作谱密度
喂给 EKF，会把速度/姿态过程噪声放大 ~1/dt，导致每次视觉更新都过度修正速度。因此
EKF 单独使用 OpenVINS 的 EuRoC MAV 连续时间噪声密度。

`schur_min_update_observations` 的 A/B：在完整 V2_01_easy / V2_02_medium 上
`12` 与 `24` 的 ATE 逐位相同（ATE 0.2295 / 0.3809），故取更宽松的 12，让视觉更
稀疏的帧也能参与更新。

### 5.1 饥饿判据与重启策略（P1 / P3 / P4 / P5 / P6）

**问题（ARM 证据）。** 见 7.2.1：同一 binary 仅切参数，Orange Pi Zero 3 上
EuRoC V2_03_difficult 有 182 帧被跳过（insufficient observations 162 /
condition limit 11），最长连续 **20** 帧没有视觉更新。旧实现只按**帧数**计
（`schur_max_consecutive_failures = 20`），于是"20 帧"在 x86 20 Hz 下约 1 s、
在 ARM 上却是 **2 s 以上**纯 IMU 递推，触发受控重启，产生 24.4 s 输出空洞
（ATE 3.89 m）。x86 与 ARM 的差别只来自处理频率，帧数阈值本身无法同时适配两端。
对照实验（7.2.2）也说明放宽 `schur_min_update_observations` 只会更差
（受控重启 1→3 次、36.25 s 空洞、ATE 16.02 m），因此不能靠放宽门槛解决。

**时间判据（P1，主要防线）。** `schur_max_no_update_seconds` 以**传感器时间**
（帧的 `header + td`，与 `schurProcessFrame` 的时间域一致）度量"距上次**成功应用**
视觉更新"的间隔：

- 每次成功应用更新（`UpdateSummary::applied == true`）都把基准重置为当前帧时间；
- EKF 播种（`initSchurBackend`）时基准取**播种帧时间**，因此跨 reset 不带脏状态；
- 只在 `schurProcessFrame` 里、且**本帧没有成功应用更新**时判定，绝不在
  `schurPropagate` 的 IMU 步里判定——所以**单次大 IMU 间隔本身不会触发重启**；
- 间隔超过阈值即打印 ERROR（含距上次成功更新的秒数、连续失败帧数、累计跳过
  帧数、最长无更新间隔）并调用 `scheduleControlledRestart()`；
- **抗跳变/单调性**：若当前帧时间戳**小于**基准（bag 重放、时钟回退），只
  **重置基准**（并清除冷却锚点），不触发重启；比较用严格 `>`，间隔恰好等于
  阈值也不触发；
- `<= 0` 关闭时间判据，退回纯帧数判据。

默认值 **3.0 s** 来自 ARM 实测（§7.2.4）：饥饿片段多数在 1–2 s 内自愈，而 1.0 s
的阈值会把"本来能自愈的短饥饿"也变成重启。

**饥饿期过程噪声放大（P3，两级）。** 纯 IMU 递推期间协方差增长不足会让随后的视觉
更新被卡方门控大量拒绝（"状态漂移 → 门控拒绝 → 漂移更大"的螺旋）。语义：

| 距上次成功更新的间隔 | Q 倍数 |
|---|---|
| `< schur_starvation_seconds`（默认 0.25） | 1.0（不放大） |
| `> schur_starvation_seconds` 且 `< schur_starvation_deep_seconds`（默认 1.0） | `schur_starvation_noise_scale`（默认 3.0） |
| `> schur_starvation_deep_seconds` | `schur_starvation_deep_noise_scale`（默认 10.0） |

**一旦有成功视觉更新立即恢复 1.0**；`schur_starvation_seconds <= 0` 关闭整条放大链
（深度升级以基础饥饿为前提）。实现上 `EkfState::propagate()` 的 `noise_scale`
参数（默认 1.0）由 `Estimator::schurPropagate` 按 `StarvationMonitor::noise_scale()`
传入；非有限或 `< 1` 的倍数被中和为 1.0。Q 为 PSD，乘一个正标量仍对称 PSD、有限。
饥饿状态每帧评估一次、帧间分段常数，因此一次大间隔不会让"是否放大"抖动。

**重启冷却（P4，禁止重启风暴）。** ARM 上每次受控重启都要重走 Ceres 初始化，
V2_03 上需要 10–44 s 才恢复发布里程计，所以重启必须是**最后手段**。
`schur_restart_cooldown_seconds`（默认 8.0，传感器时间）规定两次受控重启之间的
最小间隔：

- 冷却期内，时间判据与帧数兜底**都只计数不动作**（`cooldown_suppressions`），
  并打印节流 WARN；`restart` 保持 `false`；
- 冷却锚点由 `note_restart(t)` 设置；**`reset(t)` 刻意不清除锚点**——受控重启会走
  `clearState()` → `reset()` → `initSchurBackend()` → `reset(seed_time)`，如果
  `reset()` 清锚点，冷却就永远无效。时间戳回退（bag 重放）则清除锚点；
- 由于重新初始化本身就要 10–44 s 的传感器时间，冷却在重启完成后通常已经过期，
  不会妨碍真正需要的下一次重启；
- `0` 关闭冷却（退回 v3 行为，用于 A/B）。

**帧数兜底。** `schur_max_consecutive_failures` 默认从 20 提到 **60**，纯粹是为了
不让它在时间判据之前触发：ARM 后端处理频率约 10 Hz（`freq: 10` 限流），60 帧
≈ 6 s，x86 20 Hz 下 ≈ 3 s，与 3.0 s 的时间判据同量级。它同样受冷却抑制。

**软饥饿重启（P5 / P5b，去掉"重启 → 重初始化"的高代价）。** ARM 实测（§7.2.5）
说明 P4 已把重启风暴限制到 1 次，但那一次重启仍要付 **27–37 s 无里程计输出**的
代价：`scheduleControlledRestart()` 会 `clearState()`（`schur_ekf_active_ = false`、
`solver_flag = INITIAL`）并重新走 Ceres 初始化。饥饿时坏掉的是**视觉结构**
（clone 窗口 / 特征历史），IMU/EKF 的名义状态 q/p/v/ba/bg 仍然可用，所以 P5 让
饥饿重启只丢视觉结构：

- 参数 `schur_starvation_soft_restart`（默认 **false**，opt-in）：
  - `false`（默认）：保持 v4 行为，饥饿判据命中即 `scheduleControlledRestart()`；
  - `true`：饥饿判据命中时调用 `Estimator::schurSoftRestart()`——保留
    `schur_state_` 的 IMU 名义状态，按 `init_*_sigma` 重新膨胀 P，清空 clone 窗口
    与 `schur_history_`，重置帧索引与失败计数；**不**调用 `clearState()`、不改
    `solver_flag`、不抑制发布、`schur_ekf_active_` 保持 `true`。因此 IMU 递推与
    里程计输出在饥饿期**不间断**，重启代价从"数十秒无输出"降为"几十毫秒"。
- 升级兜底 `schur_soft_restart_max_seconds`（默认 **10.0**，`<=0` 关闭升级）：
  软重启的锚点由**本次未恢复 episode 的第一次软重启**写入（`schurSoftRestart()`
  仅在 `schur_soft_restart_time_ < 0` 时 arm），此后若在该秒数内仍没有**成功应用**
  的视觉更新，则升级为一次完整受控重启（`clearState()` + Ceres 初始化）。这样软
  重启既不会无限重试，也不会把完整重启彻底取消。
- **帧数兜底同样软化（P5b）**：`schur_max_consecutive_failures` 在 soft 模式下也走
  `schurSoftRestart()`（日志 `produced N consecutive frames without a visual
  update ... performing a soft restart`），非 soft 模式逐字保留旧的完整重启路径。
  soft 模式下**只有**升级兜底能触发完整重启。该决策抽成 ROS-free 纯函数
  `backend::frameCountBackstopAction()`，由 T26 覆盖。
- 软重启沿用 P4 冷却（`note_restart()`），且当帧 `schurProcessFrame()` 正常返回
  `true`（继续 `prune_clone_window()` → `schurWriteBackWindow()`），因此**不会**
  进入 `runOptimization()` 的 Ceres fallback；只有软重启后 `schurStateFinite()`
  失败（未观察到）才会走既有的 false fallback。
- **累计计数修正**：`StarvationMonitor::reset()` 不再清零 `time_gate_triggers_` /
  `deep_escalations_` / `cooldown_suppressions_`，因此周期汇总日志里的累计值在
  v5 起是真实值（v4 及以前每次重启后被清零，见 §7.2.5 的 `time_gate_triggers=0`）。
- **ARM 实测结论（§7.2.6）：不可行，不要启用。** 3 个触发软重启的 run 里，估计
  在软重启后一步跳到 674–1065 m 之外，ATE 变成 148–194 m（对照：同一 binary 未触发
  软重启的 run 为 0.51–0.53 m）。根因是"保留已经漂移的名义状态 + 把 P 重新膨胀到
  `schur_init_*_sigma` + 只能相信降级 tracker 的弱几何观测"这个组合：大 P 让小几何
  的卡尔曼增益接近 1，状态被"吸附"到错误的视觉解上。它消除了 27–37 s 的输出空洞，
  代价却是永久发散，因此 `schur_starvation_soft_restart` 保持默认 `false`，该路径仅
  作为诊断开关保留。

**饥饿期自适应弱更新（P6，opt-in）。** 饥饿的根因是 tracker 降级后可用特征骤减、帧内几何
退化，于是卡方门控把整帧判成外点（日志 `no gated tracks`）、这一帧就没有更新、漂移更大。
P6 的思路是在"饥饿但还没漂飞"的窗口（0.25–1.0 s）里**有条件地**放松接受门槛，并**同时**
把接受的观测降权，避免把噪声当信号：

- 参数 `schur_starvation_weak_update`（默认 **false**，opt-in）与
  `schur_weak_update_max_scale`（默认 **10.0**，必须 finite 且 `>= 1.0`，否则 `readParameters`
  直接抛错）。
- 每帧的 scale 由**当前饥饿时长**（距上次成功应用更新的秒数）决定，复用既有阈值，不引入
  新的 ramp 参数：`gap <= schur_starvation_seconds` → 1.0；`starvation < gap < deep` 线性
  ramp；`gap >= deep` → `schur_weak_update_max_scale`；最后 clamp 到 `[1, max]`。实现是
  ROS-free 纯函数 `backend::weakUpdateScale()`：关闭标志、非有限参数、`max_scale < 1`、
  负/NaN gap 一律返回 1.0，因此默认配置是严格 no-op。
- 三个作用点都在 `SchurUpdater::update()`，且都以 `weak_scale == 1.0` 时**逐位不变**为前提：
  1. **门控阈值 ×scale**：`red.chi2 > chi2_multiplier * scale * chi2_quantile_95(dof)` 才判为
     外点。`red.chi2` 仍用**名义** sigma（`pixel_noise`/焦距）构造，所以门控放宽不是靠
     "把观测噪声放大"实现的。
  2. **接受观测降权**：通过门控后 `A *= 1/scale²`、`b *= 1/scale²`，等价于把每个接受观测的
     sigma 乘 scale（`R * scale²`）。它在门控**之后**施加，因此不改变门控判定本身。
  3. **接受门槛下降**：`min_features_eff = max(2, floor(min_features/scale))`、
     `min_update_observations_eff = max(4, floor(min_update_observations/scale))`；其余
     rank / indefinite / condition / 三角化判定逐字不变。
- 可观测性：`schurProcessFrame()` 的 skip WARN 与周期汇总行都带 `weak_scale`；`weak_scale > 1`
  且该帧仍被跳过时，按"第 1 次 + 每 20 次"节流打印一条
  `Schur EKF weak-update frame ... still skipped (... weak_scale=..., tracks=..., used=..., rejected=...): <reason>`。
- **实测结论（x86 见 §7.1.4，ARM 见 §7.2.7）：不可行，不要启用。** 放松门控会把原本被
  正确拒绝的**外点/误跟踪**放进来，而 `1/scale²` 的降权不足以抵消一整帧系统性偏差；
  在 x86 上它甚至把原本 0 重启、ATE 0.42 m 的 V2_03_difficult 变成 ATE 69.44 m。

**A/B 方法（ROS 参数覆盖，不改 YAML）。** 六组配置（`--parameter NAME:=VALUE`）：

| 目的 | 参数 |
|---|---|
| **legacy**（旧行为，只留 20 帧判据） | `schur_max_no_update_seconds:=0.0` `schur_starvation_seconds:=0.0` `schur_restart_cooldown_seconds:=0.0` `schur_max_consecutive_failures:=20` |
| **timed1**（v3：1.0 s 时间判据 + 单级放大） | `schur_max_no_update_seconds:=1.0` `schur_starvation_seconds:=0.25` `schur_starvation_noise_scale:=3.0` `schur_restart_cooldown_seconds:=0.0` `schur_max_consecutive_failures:=20` |
| **ride**（P4 默认值） | `schur_max_no_update_seconds:=3.0` `schur_starvation_seconds:=0.25` `schur_starvation_noise_scale:=3.0` `schur_starvation_deep_seconds:=1.0` `schur_starvation_deep_noise_scale:=10.0` `schur_restart_cooldown_seconds:=8.0` `schur_max_consecutive_failures:=60` |
| **soft**（P5/P5b 软重启：ride 全部参数 + 以下两项） | `schur_starvation_soft_restart:=true` `schur_soft_restart_max_seconds:=10.0` |
| **timeonly / noiseonly**（归因） | 只启用其中一半（见 7.2.4） |
| **weak**（P6 弱更新：ride 全部参数 + 以下两项） | `schur_starvation_weak_update:=true` `schur_weak_update_max_scale:=10.0` |

x86 与 ARM 的实测结论严格分开陈述（见第 7 节）。

---

## 6. 测试

- `src/vins_estimator/test/test_ekf_state.cpp`（27 个用例）：传播、增广、裁剪、FEJ、
  PSD 性质，`StarvationMonitor`（时间判据：帧数多但时间跨度小不触发、超过阈值
  触发、时间戳回退只重置基准；T17/T18 深度饥饿升级与关闭；T19–T22 重启冷却：
  冷却内抑制时间判据与帧数兜底、`reset()` 不清锚点、时间戳回退清锚点、`0` 关闭
  冷却；T23 累计计数不被 `reset()` 清零；T24 软重启升级判据
  `softRestartEscalationDue()`，含 `<=0` 关闭与时间戳回退；T25 `schurSoftRestart`
  的等价性质：保留 IMU 名义状态、清空 clone、P 回到 init σ；T26 帧数兜底的
  soft/full 决策 `frameCountBackstopAction()`；T27 P6 弱更新 ramp
  `weakUpdateScale()`：关闭/非法参数/负 gap/严格 `>` 边界/线性插值/`deep <=
  starvation` 阶跃/`max_scale == 1`），以及 `EkfState::propagate` 的
  饥饿期 Q 放大（线性放大、对称/正定、非法倍数被中和）。
- `src/vins_estimator/test/test_schur_update.cpp`（24 个用例）：三角化、卡方门控、
  Schur 消元等价性、track 选择（含窗口未满/已满两种模式）；T27 用合成数据证明
  P6 能救回门控崩溃——同一帧在 `weak_update_scale = 1.0` 时 `applied=false`、
  `features_used=0`（复现 `no gated tracks`），在 `10.0` 时 `applied=true`、
  `features_used=14`、相对位姿误差 2.284e-2 → 1.993e-2 下降、协方差保持对称
  PSD，且与"放宽门控但不降权"的对照相比修正量 |dx| 从 2.072e-1 降到
  1.439e-2（比值 0.069）。

运行：

```bash
colcon test --packages-select vins --ctest-args -R "test_(ekf_state|schur_update)"
```

开发机（x86_64）全量：`colcon test --packages-select vins` →
**95 tests, 0 errors, 0 failures, 0 skipped**（`test_ekf_state` 27、`test_schur_update`
24、其余 44）。ARM 上不跑 gtest（`BUILD_TESTING=ON` 只保证编译通过）。

---

## 7. 实测结果

> 阅读约定：**x86 与 ARM 的结果严格分开陈述**，两组数据的硬件、CPU 占用口径和
> 绝对延迟不可混用。所有 ATE 均为 Vicon ground truth 上的刚性对齐 RMSE（不估计
> 尺度/时间偏置）。

### 7.1 x86（开发机，ROS 2 Jazzy）

命令：`scripts/benchmark_euroc.py`，`--rate 1.0`，三个 EuRoC 序列完整回放。
原始产物在 `output/backend_rewrite/round6_full_ceres/`、`round6_full_schur/`、
`round6_full_schur_rerun/`、`round6_full_schur_rerun3/`。

| 数据集 | 后端 | ATE RMSE (m) | p50 延迟 (ms) | CPU (%·单核) | 峰值 RSS (MiB) |
|---|---|---:|---:|---:|---:|
| V2_01_easy | ceres | 0.2164 | 14.01 | 18.2 | 106.6 |
| V2_01_easy | schur_ekf | 0.2295 | 5.12 | 9.6 | 102.0 |
| V2_02_medium | ceres | **9284.25（求解失败）** | 32.56 | 11.2 | 104.4 |
| V2_02_medium | schur_ekf | 0.3809 | 5.15 | 10.0 | 103.7 |
| V2_03_difficult | ceres | 0.5414 | 11.15 | 14.0 | 106.1 |
| V2_03_difficult | schur_ekf | 0.4691 | 5.60 | 9.1 | 105.2 |

要点：

- **Ceres 在 V2_02_medium 上确定性失败**（两次独立回放均为 ATE ≈ 9284 m），
  日志中反复出现
  `Linear solver failure ... Unable to perform dense Cholesky factorization`；
  Schur 后端在同一序列上稳定收敛（0.3809 m）。这是当前配置下 Ceres 的可复现
  缺陷，不是偶发。
- Schur 后端的单帧 p50 延迟约为 Ceres 的 **1/3**（5.1–5.6 ms vs 11.2–32.6 ms），
  峰值 RSS 略低。
- 困难序列 V2_03_difficult 上 Schur 的 ATE 反而更好（0.4691 vs 0.5414）。

配置 A/B（完整 V2_01_easy / V2_02_medium，产物 `output/backend_rewrite/round6_ab/`）：

| 方案 | V2_01 ATE (m) | V2_02 ATE (m) |
|---|---:|---:|
| 冻结 clone 协方差（旧 `VINS_SCHUR_FREEZE`） | 0.2444 | 0.3895 |
| `schur_min_update_observations: 12`（**当前默认**） | 0.2295 | 0.3809 |
| `schur_window_size: 8` | 0.1917 | 0.3843 |

`window=8` 只在 V2_01 略优、V2_02 略差，且协方差维度更高（ARM 上更贵），因此
默认仍为 `window_size: 5`、`min_update_observations: 12`。

#### 7.1.1 x86 A/B：P1（时间判据）+ P3（饥饿期 Q 放大）

binary 为本地 `build/vins/vins_node`（含 P1/P3），产物
`output/backend_rewrite/v3_base_x86/`、`output/backend_rewrite/v3_timed_x86/`。
两组用**同一个 binary**，只用 ROS 参数切换：

- `base`：`schur_max_no_update_seconds:=0.0`、`schur_starvation_seconds:=0.0`（关闭 P1/P3，等价旧行为）
- `timed`：`schur_max_no_update_seconds:=1.0`、`schur_starvation_seconds:=0.25`、`schur_starvation_noise_scale:=3.0`

| 数据集 | 配置 | ATE RMSE (m) | p50 (ms) | p95 (ms) | p99 (ms) | CPU (%·单核) | 峰值 RSS (MiB) | poses/images | 最大输出空洞 (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V2_01_easy | base | 0.2191 | 5.39 | 5.96 | 7.28 | 10.45 | 102.1 | 1083/2280 | 0.10 |
| V2_01_easy | timed | 0.2191 | 5.35 | 5.95 | 7.08 | 10.18 | 102.3 | 1083/2280 | 0.10 |
| V2_02_medium | base | 0.3506 | 5.38 | 6.50 | 16.07 | 10.62 | 103.9 | 1130/2348 | 0.10 |
| V2_02_medium | timed | 0.3477 | 5.38 | 6.46 | 15.96 | 10.61 | 103.9 | 1130/2348 | 0.10 |
| V2_03_difficult | base | 0.4209 | 5.93 | 12.26 | 13.89 | 9.69 | 105.1 | 1096/1922 | 0.15 |
| V2_03_difficult | timed | 0.4216 | 5.94 | 12.22 | 13.90 | 9.81 | 105.0 | 1092/1922 | 0.15 |

要点：x86 上 base 与 timed **逐位级一致**（ATE 差 ≤ 0.004 m，p50 差 ≤ 0.04 ms，
CPU/RSS 差在噪声内）。x86 的 V2_03 timed 日志里出现 1 次 time_gate 受控重启，
但最大输出空洞仍是 0.15 s，对结果无影响。结论：新判据在 x86 的 20 Hz 低延迟
场景下**不改变行为**，没有引入回归。

> 可复现性提示：A/B 各 run 的 `estimator.log` 启动横幅里的
> `Backend type: ... / schur window=... min_update_obs=...` 是**静态回显**（由
> `readParameters` 在 ROS 参数覆盖之前打印 YAML 值），因此不能用来确认 `--parameter`
> 是否生效；变体归属是由 label 与 ATE 量级推断的。真正生效的证据是同一次运行里的
> `Backend selection: schur_ekf` 与各变体可区分的 ATE。

#### 7.1.2 x86 A/B：P4（重启冷却 + 两级 Q 放大 + 新默认阈值）

binary 为本地 `build/vins/vins_node`（含 P4），产物
`output/backend_rewrite/v4_ride_x86/`、`output/backend_rewrite/v4_base_x86/`。
两组用**同一个 binary**，只用 ROS 参数切换（`ride` = P4 默认值，`base` = legacy
参数，见 5.1 的 A/B 表）。另有一次被中断的残留
`output/backend_rewrite/_incomplete_v4_ride_x86_1946/`，不计入。

| 数据集 | 配置 | ATE RMSE (m) | p50 (ms) | p95 (ms) | p99 (ms) | CPU (%·单核) | 峰值 RSS (MiB) | poses/images | 最大输出空洞 (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V2_01_easy | ride | 0.2191 | 5.41 | 6.15 | 7.47 | 10.38 | 102.2 | 1083/2280 | 0.10 |
| V2_01_easy | base | 0.2191 | 5.38 | 6.26 | 8.11 | 10.51 | 102.2 | 1083/2280 | 0.10 |
| V2_02_medium | ride | 0.3477 | 5.53 | 7.18 | 16.09 | 11.06 | 103.7 | 1130/2348 | 0.10 |
| V2_02_medium | base | 0.3506 | 5.48 | 6.60 | 15.98 | 10.83 | 103.7 | 1130/2348 | 0.10 |
| V2_03_difficult | ride | 0.4219 | 6.14 | 12.18 | 13.89 | 10.23 | 105.0 | 1096/1922 | 0.15 |
| V2_03_difficult | base | 0.4209 | 6.03 | 12.22 | 14.12 | 9.85 | 105.0 | 1096/1922 | 0.15 |

要点：ride 与 base 在 x86 上仍然**逐位一致**（ATE 差 ≤ 0.003 m，poses 逐位相同，
p50 差 ≤ 0.16 ms）。日志统计：两组都是 **0 次重启、0 次冷却抑制、0 次
`Tracking queue overflow`**；ride 的 V2_03_difficult 有 1 次
`deep starvation started`（两级 Q 放大生效），base 没有。结论：P4 在 x86 上不改变
行为，无回归。

#### 7.1.3 x86 A/B：P5/P5b（软饥饿重启，opt-in）

产物 `output/backend_rewrite/v5_soft_x86/`。同一个 binary，参数 = `ride` 全部参数
再加 `schur_starvation_soft_restart:=true` `schur_soft_restart_max_seconds:=10.0`。

| 数据集 | 配置 | ATE RMSE (m) | p50 (ms) | p95 (ms) | p99 (ms) | CPU (%·单核) | 峰值 RSS (MiB) | poses/images | 最大输出空洞 (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V2_01_easy | soft | 0.2191 | 5.45 | 6.20 | 7.35 | 10.54 | 101.6 | 1083/2280 | 0.10 |
| V2_01_easy | ride（7.1.2） | 0.2191 | 5.41 | 6.15 | 7.47 | 10.38 | 102.2 | 1083/2280 | 0.10 |
| V2_02_medium | soft | 0.3477 | 5.49 | 6.44 | 15.96 | 10.76 | 103.2 | 1130/2348 | 0.10 |
| V2_02_medium | ride（7.1.2） | 0.3477 | 5.53 | 7.18 | 16.09 | 11.06 | 103.7 | 1130/2348 | 0.10 |
| V2_03_difficult | soft | 0.4219 | 6.10 | 12.21 | 13.78 | 10.10 | 104.5 | 1096/1922 | 0.15 |
| V2_03_difficult | ride（7.1.2） | 0.4219 | 6.14 | 12.18 | 13.89 | 10.23 | 105.0 | 1096/1922 | 0.15 |

要点：x86 上三个序列都是 **0 次重启**，因此**软重启路径一次都没有被触发**
（`grep -c "soft restart at frame"` = 0、`grep -c "performing a soft restart"` = 0），
ATE / poses / 输出空洞与 §7.1.2 的 `ride` **完全相同**（0.2191 / 0.3477 / 0.4219 m，
pose 数逐位相同）。这证明 P5/P5b 在不需要它的平台上**是纯 no-op**：只多两个不生效
的参数。它真正的验证在 ARM（§7.2.6）。

#### 7.1.4 x86 A/B：P6 饥饿自适应弱更新（opt-in，结论：**不可行**）

产物 `output/backend_rewrite/v6_weak_x86/`（`weak`）与
`output/backend_rewrite/v6_control_x86/`（`ride` 对照，只跑 V2_03_difficult）。
**同一个 binary**（本地 `build/vins/vins_node`，含 P1–P6），唯一差别是
`schur_starvation_weak_update` / `schur_weak_update_max_scale` 两个参数（见 5.1 的
A/B 表）。`v6_control_x86` 的 ATE 与 §7.1.2 的 `v4_ride_x86`、§7.1.3 的 `v5_soft_x86`
**逐位相同**（0.421859 m），说明 P6 补丁在关闭时确实是 no-op。

| 数据集 | 配置 | ATE RMSE (m) | p50 (ms) | p95 (ms) | p99 (ms) | CPU (%·单核) | 峰值 RSS (MiB) | poses/images | 最大输出空洞 (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V2_01_easy | weak | 0.219390 | 5.48 | 6.36 | 7.11 | 10.60 | 101.6 | 1083/2280 | 0.10 |
| V2_01_easy | ride（7.1.2） | 0.219104 | 5.41 | 6.15 | 7.47 | 10.38 | 102.2 | 1083/2280 | 0.10 |
| V2_02_medium | weak | 0.340113 | 5.57 | 6.91 | 15.90 | 10.99 | 103.1 | 1130/2348 | 0.10 |
| V2_02_medium | ride（7.1.2） | 0.347736 | 5.53 | 7.18 | 16.09 | 11.06 | 103.7 | 1130/2348 | 0.10 |
| V2_03_difficult | weak | **69.437068** | 6.14 | 12.30 | 14.22 | 10.12 | 104.2 | 1070/1922 | **2.70** |
| V2_03_difficult | ride（同 binary 对照，`v6_control_x86`） | 0.421859 | 6.09 | 12.34 | 13.87 | 10.03 | 104.3 | 1096/1922 | 0.15 |

要点：

- **V2_01_easy / V2_02_medium：P6 只各命中 1 帧，量级可忽略但并非逐位不变。**
  两个序列各有 1 帧的饥饿间隔落在 `0.25–1.0 s` 之间，日志里出现一次
  `weak-update frame ... still skipped (weak_scale=1.60, ...)`；同一帧在 ride 配置下
  是被跳过的（skip WARN 条数 v4=95 → v6=94），所以轨迹确实变了：ATE
  0.219104 → 0.219390 m（+0.0003，略差）、0.347736 → 0.340113 m（−0.008，略好），
  pose 数与匹配数逐位相同（1083/1080、1130/1128）。即单次弱更新的影响是
  "同量级抖动"，方向不定。
- **V2_03_difficult：P6 把可用的结果变成不可用。** 日志里 17 帧带 `weak_scale > 1`
  （4.00 / 5.20 / 8.80 / 10.00），ATE 从 **0.421859 m 劣化到 69.437068 m**，
  并且出现 1 次 `scheduling a controlled estimator restart`（frame 426，
  `max_no_update_gap=3.100 s`）。对照（同 binary、`ride`）是 **0 次重启、
  1 次 `deep starvation started`、`weak_scale` 全程 1.00**；weak 则是
  **6 次 deep starvation + 1 次受控重启**。x86 上 ride / soft / base / control
  四个配置的 V2_03 ATE 逐位相同（0.421859 m），因此这次劣化只能归因于 P6 本身，
  不是 run 间噪声。
- **机理**：V2_03 的饥饿窗口里 tracker 已经降级，被卡方门控拒绝的恰恰是**误跟踪**
  的特征。门控阈值 ×10 之后这些外点通过了门控；`1/scale²` 的降权只降低权重，
  并不能抵消"一整帧系统性偏差"（相对位姿/尺度偏置）被注入状态。随后漂移更大，
  连放宽 10× 的门控也开始整帧拒绝（日志 `no gated tracks`，`tracks=120 used=0
  rejected=120`），直到 3 s 时间判据触发受控重启并重新走 Ceres 初始化。
  **"多收一点、少信一点"在真实退化数据上不成立**——这与 T27 合成实验（干净数据、
  只有低秩几何问题）给出的乐观结论相反。
- **行动项**：`schur_starvation_weak_update` **保持默认 `false`**，该路径仅作为
  诊断开关保留（见 §5.1、§7.4）。

### 7.2 ARM（Orange Pi Zero 3，4×Cortex-A53，aarch64）

平台：Orange Pi Zero 3，aarch64，4 核，ROS 2 Jazzy。
二进制 `schur_ekf_v2`，SHA256 `5214559fd8145ac53544d6c5fcc870f56abe6f24a215c0b24119ff876a8284fb`
（aarch64 ELF，Release、LTO OFF、`VINS_ENABLE_NATIVE_OPTIMIZATION=OFF`）。
Ceres 与 Schur 是**同一个 binary**，只用 `backend_type` 参数切换，因此两者差异只
来自后端本身。每个序列回放前都预热了 mcap 缓存。原始产物在 ARM 上
`~/vins_remote_eval_20261019/benchmark/arm_{ceres,schur}_v2_*/`。

| 数据集 | 后端 | ATE RMSE (m) | p50 (ms) | p95 (ms) | p99 (ms) | CPU (%·单核) | 峰值 RSS (MiB) | poses/images | 最大输出空洞 (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V2_01_easy | ceres | 0.1675 | 409.5 | 566.4 | 909.5 | 178.3 | 103.5 | 1082/2280 | 0.15 |
| V2_01_easy | schur_ekf | 0.2623 | 311.1 | 336.5 | 355.0 | 114.0 | 97.2 | 1083/2280 | 0.20 |
| V2_02_medium | ceres | 0.2377 | 382.8 | 485.7 | 1052.8 | 172.9 | 101.2 | 1076/2348 | 0.50 |
| V2_02_medium | schur_ekf | 0.4235 | 315.1 | 342.9 | 378.0 | 116.3 | 97.8 | 1100/2348 | 0.15 |
| V2_03_difficult | ceres | 0.4740 | 365.3 | 529.9 | 1172.5 | 154.5 | 101.1 | 1083/1922 | 0.35 |
| V2_03_difficult | schur_ekf | **3.8907（触发受控重启）** | 315.3 | 439.4 | 645.1 | 122.7 | 99.2 | 851/1922 | **24.40** |

ARM 上的结论：

- **资源收益成立**：Schur 相对 Ceres 降低 CPU 21–36%（114–123% vs 155–178%），
  p50 延迟降低 14–24%（311–315 ms vs 365–410 ms），峰值 RSS 低 2–6 MiB，并且
  p95/p99 明显更平稳（V2_01 p99 355 ms vs 910 ms，V2_02 p99 378 ms vs 1053 ms）。
- **精度在 ARM 上不如 Ceres**：三个序列 ATE 都更差（0.262 vs 0.168、0.424 vs
  0.238、3.891 vs 0.474）。
- **V2_03_difficult 上不可用**：出现 20 连续帧无视觉更新 → 受控重启 → 24.4 s 输出
  空洞，重启后轨迹整体偏移 ~1.1 m。x86 同序列未触发该现象。

#### 7.2.1 ARM 上 V2_03 失败的机理（已定位）

ARM 单帧 p50 ≈ 315 ms，而 bag 以 20 Hz 播图，estimator 实际只能处理约 3 帧/s，
日志中出现 16 次 `Tracking queue overflow: dropping stale image pair`（x86 同序列
为 0 次）。注意溢出次数只取决于 tracker 处理速率与输入频率，**与后端无关**：同一
序列 Schur 与 Ceres 两次运行的溢出次数完全相同（V2_01 22、V2_02 23、V2_03 16）。
由此产生连锁：

1. 丢帧使初始化窗口的帧间距被拉长，EKF 的初始种子与 x86 略有不同；
2. 每帧可用视觉更新频率从 20 Hz 降到约 3 Hz，两次视觉修正之间纯 IMU 递推长达
   ~0.3 s；
3. 卡方门控大量拒绝特征（日志中反复出现 `used=3..11, rejected=40..90`），观测数
   达不到 `schur_min_update_observations=12` → 该帧被跳过（只做 IMU 递推）；
4. `schur_max_consecutive_failures=20` 是**按帧数**计的，在 ARM 3 Hz 处理频率下
   等价于 **~6.4 s 纯 IMU 递推**（x86 20 Hz 下只有 ~1 s），于是触发受控重启。

同一序列在 x86 上也有 231 次跳过，但从未连续 20 次，因此没有重启。**这是 ARM
特有的失效模式，不是 x86 的复现。**

各 run 的跳过统计（`estimator.log` 逐条 grep）：

| run | 数据集 | 跳过帧数 | 最长连续失败 | 受控重启 | queue overflow | 主因 |
|---|---|---:|---:|---:|---:|---|
| arm_schur_v2_easy | V2_01 | 177 | 1 | 0 | 22 | insufficient 173 |
| arm_schur_v2_medium | V2_02 | 157 | 1 | 0 | 23 | insufficient 147 / condition 8 |
| arm_schur_v2_difficult | V2_03 | 182 | **20** | **1** | 16 | insufficient 162 / condition 11 |
| arm_schur_v2_difficult_minobs4 | V2_03 | 99 | **20** | **3** | 16 | insufficient 70 / condition 16 |
| x86 round6_full_schur（对照） | V2_03 | 231 | 1 | 0 | 0 | insufficient 214 / condition 11 |

注：`summary.json` 的 `failure_reboots` 是**进程崩溃/重启**计数，与受控 estimator
重启不是一回事；受控重启只体现在 `max_output_gap_s` 与日志里。

#### 7.2.2 缓解实验：放宽更新门槛反而更差

`schur_min_update_observations:=4`（其余不变）在 ARM V2_03_difficult 上：
ATE **16.0219 m**、输出 492/1894、最大输出空洞 36.25 s、CPU 145.7%。
放宽门槛会让低秩/病态更新被接受，滤波器发散得更快：受控重启从 1 次增加到
**3 次**，且触发原因不再是"观测数不足"，而是 `no gated tracks`（frame 78/128，
`used=0 rejected=97/35`）和 `Schur information condition limit exceeded`
（frame 121）。CPU 也从 122.7% 升到 145.7%、p99 从 645 ms 升到 1297 ms。
因此「20 连续失败重启」不是 `12` 这个阈值造成的假象，而是该序列在 ARM 上确实
缺乏可用视觉约束，**不应把放宽门槛当作补救手段**。

#### 7.2.3 与 v1 ARM 结果的对比

v1（`schur_ekf_v1`，`min_update_observations=24` + 旧 `VINS_SCHUR_FREEZE`）在 ARM
V2_01_easy 上 ATE 0.5470 m、p50 310.8 ms、CPU 114.1%、RSS 101.0 MiB；v2 把 ATE
降到 **0.2623 m**（−52%），资源占用基本不变（CPU 114.0%、RSS 97.2 MiB）。
分段 ATE 显示改善主要来自早期：v1 的 0–10 s RMSE 为 1.4913 m，v2 为 0.2231 m
（x86 同序列 0–10 s 为 0.2121 m）。

#### 7.2.4 ARM A/B：P1（时间判据）+ P3（饥饿期 Q 放大）

binary `schur_ekf_v3`，SHA256
`7aa06517666256d53a3391c900f7504530f0b18a859c66123b85e7e832e23cf7`（aarch64 ELF，
远端 HEAD `82b4ba38caeaea799709886a1b88b43a17e2f628` + 未提交改动）。两组用**同一个
binary**，只用 ROS 参数切换：

- `base`：`schur_max_no_update_seconds:=0.0`、`schur_starvation_seconds:=0.0`（关闭 P1/P3，只留旧的 20 连续帧判据）
- `timed`：`schur_max_no_update_seconds:=1.0`、`schur_starvation_seconds:=0.25`、`schur_starvation_noise_scale:=3.0`

原始产物：`~/vins_remote_eval_20261019/benchmark/arm_v3_{base,timed}_*/`。每个序列回放前
预热 mcap；x86 与 ARM 不并发 benchmark。

**V2_01_easy / V2_02_medium（无重启，无副作用）**

| 数据集 | 配置 | run | ATE RMSE (m) | 轨迹覆盖 (s) | poses/images | p50 (ms) | CPU (%·单核) | 最大输出空洞 (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| V2_01_easy | base | r1 | 0.3733 | 107.95 | 1084/2280 | 311.8 | 114.4 | 0.15 |
| V2_01_easy | base | r2 | 0.2151 | 107.85 | 1080/2280 | 311.7 | 113.4 | 0.15 |
| V2_01_easy | timed | r1 | 0.2995 | 107.95 | 1083/2280 | 312.0 | 114.1 | 0.15 |
| V2_01_easy | timed | r2 | 0.2805 | 108.00 | 1083/2280 | 312.7 | 111.9 | 0.20 |
| V2_02_medium | base | r1 | 0.4031 | 109.40 | 1095/2348 | 316.6 | 117.4 | 0.25 |
| V2_02_medium | base | r2 | 0.4269 | 107.60 | 1073/2348 | 315.4 | 120.6 | 0.70 |
| V2_02_medium | timed | r1 | 0.5094 | 108.30 | 1081/2348 | 315.6 | 120.4 | 0.65 |
| V2_02_medium | timed | r2 | 0.3054 | 113.40 | 1137/2348 | 315.1 | 114.6 | 0.15 |

两个序列都是 **0 次重启**（base/timed 都是），轨迹覆盖 ≥ 107 s（全程约 108–113 s）。
ATE 的 base/timed 差异（V2_01 0.29/0.22 均值，V2_02 0.41/0.41 均值）落在 run 间
波动范围内，没有系统性回归。延迟、CPU、RSS 在两种配置下基本一致。

**V2_03_difficult（关键序列，4 次重复）**

| 配置 | run | ATE RMSE (m) | 轨迹覆盖 (s) | poses/images | p50 (ms) | p95 (ms) | p99 (ms) | CPU (%·单核) | RSS (MiB) | 最大输出空洞 (s) | 受控重启 | 触发判据 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base | r1 | 7.2332 | 110.05 | 869/1922 | 313.1 | 432.0 | 873.9 | 122.9 | 97.9 | 21.35 | 2 | frame_gate |
| base | r2 | 5.0809 | 109.45 | 815/1922 | 315.7 | 481.5 | 883.3 | 127.8 | 99.8 | 22.95 | 2 | frame_gate |
| base | r3 | 7.3713 | 110.00 | 943/1922 | 314.9 | 446.8 | 869.7 | 120.7 | 98.5 | 9.05 | 2 | frame_gate |
| base | r4 | 0.3205 | **64.10** | 639/1922 | 310.5 | 452.0 | 795.0 | 140.3 | 98.2 | 0.35 | 0 | — |
| timed | r1 | **0.4551** | 109.00 | 1085/1922 | 313.7 | 432.1 | 535.4 | 111.7 | 99.2 | **0.20** | 1 | time_gate |
| timed | r2 | 3.0868 | 109.20 | 966/1922 | 315.1 | 468.1 | 976.7 | 116.9 | 99.4 | 9.25 | 4 | time_gate |
| timed | r3 | 12.4428 | 101.90 | 400/1922 | 319.3 | 1111.1 | 1526.9 | 149.5 | 100.2 | 44.50 | 4 | time_gate |
| timed | r4 | 2.5731 | 108.10 | 691/1922 | 314.6 | 460.8 | 1188.1 | 136.6 | 99.9 | 38.45 | 4 | time_gate |

归因实验（V2_03_difficult，各 1 次）：

| 配置 | 参数 | ATE RMSE (m) | 轨迹覆盖 (s) | poses/images | 最大输出空洞 (s) | 受控重启 | 触发判据 |
|---|---|---:|---:|---:|---:|---:|---|
| timeonly | `schur_max_no_update_seconds:=1.0`，P3 关闭 | 2.2710 | 109.00 | 605/1922 | 36.75 | 4 | time_gate |
| noiseonly | P1 关闭，`schur_starvation_seconds:=0.25`、`scale:=3.0` | 1.7356 | 109.50 | 856/1922 | 23.90 | 1 | frame_gate |

日志统计（`estimator.log` 逐条 grep）：

| 配置 | 受控重启 | time_gate | frame_gate | 跳过帧总数 |
|---|---:|---:|---:|---:|
| base（4 次 run） | 2 / 2 / 2 / 0 | 0 | 6 | 168–187 |
| timed（4 次 run） | 1 / 4 / 4 / 4 | 13 | 0 | 208–303 |

**结论（V2_03_difficult）**

- **机制可行**：timed 的最好一次 run 为 ATE **0.4551 m**、最大输出空洞 **0.20 s**、
  轨迹覆盖 109 s，显著优于 base 的 5.08–7.37 m / 9.05–22.95 s 空洞。
- **但不稳定**：4 次重复里 3 次出现 4 次受控重启，输出空洞 9.25/38.45/44.5 s，
  ATE 3.09/2.57/12.44 m。1.0 s 的时间判据把「本来 1–2 s 就能自愈的短饥饿」也
  变成了重启。
- **重启是代价最高的动作**：日志显示每次 time_gate 都在 1.05–1.10 s 触发
  （样例：`Schur EKF visually starved for 1.100 s (> 1.000 s) at frame 1085: streak=11
  skipped_total=303 max_no_update_gap=1.100 s; scheduling a controlled estimator restart`），
  而重启后 estimator 要重新走 Ceres 初始化，V2_03 上需要 10–44 s 才恢复发布里程计
  （`Schur EKF seeded at t=...` 之间的间隔即空洞）。于是「短饥饿 → 重启 → 重新初始化
  → 又短饥饿」形成**重启风暴**，空洞被放大一个数量级。
- **必须同时用 P1 与 P3**：timeonly 更差（4 次重启、36.75 s 空洞、2.2710 m），
  noiseonly 仍被旧的 20 连续帧判据拖住（1 次重启、23.90 s 空洞、1.7356 m）。
- **ATE 口径警告**：ATE 只在**被匹配到的位姿**上统计，重启导致轨迹覆盖下降时 ATE
  反而可能"变好"。例如 base r4 的 ATE 0.3205 m 对应的轨迹覆盖只有 **64.10 s / 110 s**，
  不能与覆盖 109–110 s 的 run 直接比较。所有 ARM 对比都必须同时看
  `trajectory_duration_s` 与 `poses/images`。
- base 的 2 次重启全部由**旧的 20 连续帧判据**触发（frame_gate），产生的空洞
  9.05–22.95 s 与 timed 的重启空洞同量级——即旧判据同样会踩这个坑，只是触发频率更低。

#### 7.2.5 ARM：P4 矩阵（重启冷却 + 两级 Q 放大 + 新默认阈值）

binary `schur_ekf_v4`，SHA256
`ccd9dd02663f534d369b0ed5831194f5127ac9bea7391dc9179fc94f4c7343f1`
（aarch64 ELF，BuildID `61f87d49eb23030fe8b7abcc59fc13199eb5e4a2`，Release、
LTO OFF、`VINS_ENABLE_NATIVE_OPTIMIZATION=OFF`）。远端 HEAD
`82b4ba38caeaea799709886a1b88b43a17e2f628` + 未提交改动。原始产物
`~/vins_remote_eval_20261019/benchmark/arm_v4_*/`；每个序列回放前预热 mcap；
x86 与 ARM 不并发 benchmark；`ROS_DOMAIN_ID=173`。

**V2_01_easy / V2_02_medium（`ride` 配置，无重启）**

| 数据集 | 配置 | ATE RMSE (m) | 轨迹覆盖 (s) | poses/images | p50 (ms) | p95 (ms) | p99 (ms) | CPU (%·单核) | 峰值 RSS (MiB) | 最大输出空洞 (s) | 受控重启 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V2_01_easy | ride | 0.3064 | 108.0 | 1084/2280 | 311.8 | 337.7 | 354.0 | 114.2 | 97.3 | 0.15 | 0 |
| V2_02_medium | ride | 0.3057 | 113.4 | 1132/2348 | 315.1 | 341.7 | 360.9 | 113.6 | 100.4 | 0.20 | 0 |

**V2_03_difficult（关键序列，13 次 run，跨 4 种配置）**

| 配置 | run | ATE RMSE (m) | 轨迹覆盖 (s) | poses/images | p50 (ms) | p95 (ms) | p99 (ms) | CPU (%·单核) | 峰值 RSS (MiB) | 最大输出空洞 (s) | 受控重启 | 深度饥饿 | queue overflow |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy | r1 | 0.3689 | 109.3 | 1088/1922 | 313.3 | 439.1 | 545.3 | 111.2 | 99.0 | 0.20 | 0 | 0 | 16 |
| legacy | r2 | 0.4828 | 108.4 | 1079/1922 | 312.6 | 441.2 | 532.4 | 112.8 | 99.7 | 0.35 | 0 | 0 | 16 |
| legacy | r3 | 3.0763 | 109.5 | 1021/1922 | 314.0 | 437.2 | 821.2 | 116.6 | 99.1 | 7.05 | 1（帧数兜底） | 0 | 16 |
| legacy | r4 | 8.7793 | 109.5 | 833/1922 | 310.8 | 486.2 | 1263.5 | 126.5 | 99.1 | 20.75 | 3（帧数兜底） | 0 | 16 |
| timed1（v3 参数） | r1 | 0.4932 | 108.7 | 1081/1922 | 315.0 | 438.0 | 540.3 | 112.1 | 99.5 | 0.25 | 1（时间判据） | 1 | 16 |
| ride | r1 | 1.6379 | 109.5 | 820/1922 | 312.4 | 448.7 | 910.2 | 127.7 | 99.0 | **27.10** | 1（时间判据） | 2 | 16 |
| ride | r2 | 3.9873 | 109.6 | 724/1922 | 313.8 | 440.7 | 978.0 | 135.0 | 99.0 | **36.75** | 1（时间判据） | 2 | 16 |
| ride | r3 | 3.0590 | 109.3 | 822/1922 | 313.8 | 453.7 | 903.6 | 126.9 | 99.9 | **27.25** | 1（时间判据） | 1 | 16 |
| ride | r4 | 15.6841 | 109.3 | 588/1922 | 317.6 | 557.2 | 1169.5 | 138.2 | 99.6 | **32.75** | 3（时间判据） | 7 | 16 |
| ride | r5 | 0.3897 | 109.5 | 1094/1922 | 313.3 | 428.1 | 523.9 | 111.7 | 98.3 | 0.20 | 0 | 1 | 16 |
| nogate（关闭时间判据，只留 Q 放大） | r1 | 0.3620 | 109.5 | 1091/1922 | 312.2 | 435.9 | 524.0 | 111.6 | 98.3 | 0.20 | 0 | 1 | 16 |
| nogate | r2 | 36.3404 | 109.5 | 1049/1922 | 312.0 | 440.8 | 812.3 | 114.0 | 98.0 | 4.50 | 1（帧数兜底） | 5 | 16 |

**结论（ARM P4）**

- **冷却有效，风暴被限幅**：`ride` 的 5 次 run 里，4 次的重启次数从 v3 `timed` 的
  「最多 4 次」降到**恰好 1 次**（r4 例外，3 次，间隔已超过 8 s 冷却）。日志中
  `time_gate_triggers` 与 `cooldown_suppressions` 在 v4 里被 `reset()` 清零，
  所以汇总行读数为 0（这是 P5 修掉的缺陷，见 5.1）。
- **但单次完整重启的代价仍不可接受**：`ride` 的 5 次 run 里只有 **1/5** 成功
  （r5：ATE 0.3897 m、空洞 0.20 s）；另外 4 次的重启后空洞为 27.10 / 36.75 /
  27.25 / 32.75 s，ATE 1.64 / 3.99 / 3.06 / 15.68 m，`poses` 掉到 588–822/1922。
  即"重启 → 重新走 Ceres 初始化 → 数十秒无里程计输出"这一机理**没有被 P4 改变**，
  只是频率下降了。
- **只靠 Q 放大不足以修复**：`nogate`（关掉时间判据、保留两级 Q 放大）2 次 run 为
  0.3620 m / 0.20 s 与 36.3404 m / 4.50 s，说明饥饿**可以**自愈（r1 无重启且精度
  最好），但也可能恶化到必须重启；Q 放大本身不能保证收敛。
- **legacy 也在同一水平**：4 次 run 为 0.3689 / 0.4828 / 3.0763 / 8.7793 m，
  空洞 0.20 / 0.35 / 7.05 / 20.75 s，受控重启 0/0/1/3 次（全部来自帧数兜底）。
  也就是说旧判据同样不稳定，P1/P3/P4 并没有把 legacy 变差，但也没能解决根因。
- **`Tracking queue overflow` 与后端无关**：V2_03 上每次 run 都是 16 条（5 s 节流
  后的条数），V2_01/V2_02 为 22/23 条。它是视觉更新降频（饥饿）的根本来源。
- **口径警告**：`trajectory_duration_s` 一直是 108–110 s（bag 时间），但
  `poses/images` 只有 588–1094/1922——空洞体现在 `max_output_gap_s` 与 pose 数上，
  必须与 ATE 一起看。

#### 7.2.6 ARM：P5/P5b 软饥饿重启矩阵（结论：**不可行**，不要启用）

binary `schur_ekf_v5`，SHA256
`e3221e93ae3e1a5ed56fdaa6063c46a8251f6d71835c4b7656a060ad40df4a28`
（aarch64 ELF，BuildID `44ae9aa8d1d0a37903a36e2f3ebcbcc1abd16ac4`，Release、
LTO OFF、`VINS_ENABLE_NATIVE_OPTIMIZATION=OFF`）。远端 HEAD
`82b4ba38caeaea799709886a1b88b43a17e2f628` + 未提交改动。原始产物
`~/vins_remote_eval_20261019/benchmark/arm_v5_*/`；每个序列回放前预热 mcap；
x86 与 ARM 不并发 benchmark；`ROS_DOMAIN_ID=173`。

`soft` 配置 = `ride` 的全部参数 + `schur_starvation_soft_restart:=true`
`schur_soft_restart_max_seconds:=10.0`；`arm_v5_ride_difficult` 是软重启**关闭**的
对照（其余参数与 `ride` 相同）。"最大单步位移"由 `trajectory.csv` 相邻位姿的
欧氏距离算出，用来量化状态是否发生过"跳飞"。

| 数据集 | 配置 | ATE RMSE (m) | 轨迹覆盖 (s) | poses/images | p50 (ms) | p95 (ms) | p99 (ms) | CPU (%·单核) | 峰值 RSS (MiB) | 最大输出空洞 (s) | 软重启 | 升级完整重启 | 深度饥饿 | queue overflow | 最大单步位移 (m) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V2_03_difficult | soft | **194.0411** | 109.4 | 996/1913 | 314.5 | 435.9 | 785.9 | 117.4 | 98.9 | 9.20 | 3 | 1 | 6 | 16 | **1065.23** |
| V2_03_difficult | soft | **179.2457** | 109.3 | 672/1922 | 317.6 | 534.8 | 1157.6 | 136.1 | 99.5 | **41.55** | 5 | 1 | 10 | 17 | **674.35** |
| V2_03_difficult | soft | 0.5304 | 109.5 | 1094/1922 | 313.3 | 441.3 | 564.8 | 111.6 | 98.2 | 0.20 | 0 | 0 | 1 | 16 | 0.33 |
| V2_03_difficult | ride（对照） | 1.7724 | 109.3 | 825/1922 | 315.2 | 465.7 | 898.4 | 127.2 | 99.2 | 26.60 | 0 | 1 | 2 | 16 | 12.51 |
| V2_01_easy | soft | **147.8960** | 108.0 | 1069/2280 | 309.9 | 336.3 | 397.6 | 114.1 | 101.1 | 1.45 | 4 | 1 | 9 | 22 | **856.87** |
| V2_02_medium | soft | 0.5083 | 109.2 | 1093/2348 | 315.8 | 342.2 | 434.5 | 116.8 | 99.4 | 0.20 | 0 | 0 | 0 | 23 | 0.29 |

**结论（ARM P5）**

- **软重启一旦触发，估计就永久损坏。** 3 个触发了软重启的 run（V2_03 两次、
  V2_01 一次）ATE 为 **194.04 / 179.25 / 147.90 m**，最大单步位移
  **1065.23 / 674.35 / 856.87 m**（即状态在一步之内跳到一公里外）。而同一 binary、
  同一序列、软重启**没有**触发的 run（V2_03 rep3、V2_02 medium）ATE 为
  0.5304 / 0.5083 m、最大单步位移 0.33 / 0.29 m。对照组
  `arm_v4_ride_easy`（V2_01_easy、ride 配置）的最大单步位移只有 0.14 m。
- **机理（与 P5 的设计假设正好相反）。** 软重启保留的是**已经漂移**的 IMU 名义
  状态，同时把 P 按 `schur_init_*_sigma` 重新膨胀（pos 0.5 m、vel 0.5 m/s、
  ba 0.5 m/s²），而 clone 窗口被清空后最早的视觉更新只能来自**降级 tracker** 的
  少量弱几何特征。大 P + 弱几何 ⇒ 卡尔曼增益接近 1 ⇒ 状态一步"吸附"到错误的
  视觉解上，此后轨迹再也回不来。也就是说"保留状态 + 重新膨胀 P + 相信降级观测"
  是最坏的组合；P5 去掉了 27–37 s 的输出空洞，却把它换成了永久性发散。
- **空洞也没有真正消失。** 触发软重启的 3 个 run 最大输出空洞
  9.20 / 41.55 / 1.45 s；其中 rep2 的 **41.55 s 比 ride 对照的 26.60 s 更差**——
  10 s 的升级兜底本身要重走一次 Ceres 初始化，而在这 10 s 里状态已经跳飞。
- **升级兜底按设计工作**：每个触发软重启的 run 恰好出现 1 次
  `escalating to a full controlled estimator restart`（`scheduling a controlled
  estimator restart` 为 0，因为升级走的是另一条日志分支）。它证明了 P5b 的判定
  链路是通的，但 10 s 这个窗口太长。
- **资源口径没有变化**：CPU 111.6–136.1%、峰值 RSS 98.2–101.1 MiB、p50
  309.9–317.6 ms，与 v4 同量级 —— 软重启既无性能收益，也无精度收益。
- **行动项**：`schur_starvation_soft_restart` **保持默认 `false`**，并把这条
  代码路径标记为"实测不可行，不要启用"（见 §5.1、§7.4）。
- **附带发现：ARM 的 run 间方差极大。** 同一 binary、同一序列
  （V2_01_easy）在 v4 `ride` 配置下 0 重启、ATE 0.3064 m，在 v5 `soft` 配置下
  触发 4 次软重启、ATE 147.90 m。**任何 ARM 结论都必须 ≥3 次重复**，单次 run 的
  0.3 m 与 3 m 之间的差别在统计上没有意义。

#### 7.2.7 ARM：P6 饥饿自适应弱更新矩阵（结论：**无收益，不要启用**）

binary `schur_ekf_v6`，SHA256
`2b0e4cd792f8ce22cf10aeb8639bee61b0ced149a8091e63899ad2ece6d6eeac`
（aarch64 ELF，BuildID `144551d232c739433872bafda36035f4beb841a4`，Release、
LTO OFF、`VINS_ENABLE_NATIVE_OPTIMIZATION=OFF`；`strings` 中可见
`weak-update frame %d still skipped` 与汇总行的 `weak_update_scale=%.2f`）。远端
HEAD `82b4ba38caeaea799709886a1b88b43a17e2f628` + 未提交改动；原始产物
`~/vins_remote_eval_20261019/benchmark/arm_v6_weak_*/`、`arm_v6_control_*/`；
每个序列回放前预热 mcap；x86 与 ARM 不并发 benchmark；`ROS_DOMAIN_ID=173`。

`weak` = `ride` 全部参数 + `schur_starvation_weak_update:=true`
`schur_weak_update_max_scale:=10.0`；`control` = **同一个 binary** 只跑 `ride`
（weak 关闭），用于把 P6 的影响从"会话级方差"里分离出来。"最大单步位移"由
`trajectory.csv` 相邻位姿的欧氏距离算出。

| 数据集 | 配置 | ATE RMSE (m) | 轨迹覆盖 (s) | poses/images | p50 (ms) | p95 (ms) | p99 (ms) | CPU (%·单核) | 峰值 RSS (MiB) | 最大输出空洞 (s) | 受控重启 | 深度饥饿 | 弱更新帧数 | queue overflow | 最大单步位移 (m) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V2_03_difficult | weak | 0.3602 | 109.4 | 1090/1922 | 314.0 | 430.1 | 535.1 | 110.1 | 99.4 | 0.30 | 0 | 0 | 2 | 16 | 0.32 |
| V2_03_difficult | weak | **50.1979** | 109.5 | 1011/1922 | 313.6 | 434.6 | 1004.5 | 117.0 | 99.1 | 8.10 | 1 | 2 | 7 | 16 | **260.57** |
| V2_03_difficult | weak | **301.5650** | 109.5 | 739/1922 | 317.2 | 573.5 | 1096.5 | 130.7 | 99.0 | **35.30** | 1 | 7 | 14 | 17 | **1139.20** |
| V2_03_difficult | weak | **131.7229** | 109.3 | 1076/1922 | 313.9 | 434.9 | 559.2 | 111.9 | 99.5 | 1.40 | 1 | 9 | 10 | 16 | **664.85** |
| V2_03_difficult | weak | **214.4459** | 109.5 | 1007/1922 | 313.1 | 472.2 | 1029.0 | 115.2 | 99.0 | 3.95 | 2 | 17 | 23 | 16 | **741.04** |
| V2_03_difficult | control（同 binary，weak 关闭） | 7.5082 | 109.5 | 932/1922 | 314.7 | 458.4 | 914.6 | 118.6 | 99.2 | 16.10 | 1 | 3 | 0 | 16 | 54.96 |
| V2_03_difficult | control（同 binary，weak 关闭） | 10.2641 | 108.4 | 1017/1922 | 315.1 | 455.4 | 766.5 | 116.9 | 99.8 | 5.95 | 1 | 3 | 0 | 16 | 71.01 |
| V2_03_difficult | control（同 binary，weak 关闭） | 15.5086 | 108.4 | 626/1922 | 317.4 | 527.1 | 1232.1 | 138.2 | 100.5 | **38.70** | 2 | 5 | 0 | 16 | 61.80 |
| V2_03_difficult | control（同 binary，weak 关闭） | 5.3506 | 109.5 | 946/1922 | 313.0 | 435.6 | 1045.3 | 121.6 | 99.2 | 13.35 | 2 | 3 | 0 | 16 | 27.14 |
| V2_01_easy | weak | 0.2329 | 108.0 | 1083/2280 | 310.6 | 335.0 | 346.4 | 114.2 | 101.1 | 0.20 | 0 | 0 | 1 | 22 | 0.13 |
| V2_02_medium | weak | 0.3816 | 113.4 | 1136/2348 | 316.5 | 341.7 | 360.6 | 113.6 | 100.4 | 0.20 | 0 | 0 | 1 | 23 | 0.24 |

**结论（ARM P6）**

- **V2_01_easy / V2_02_medium：P6 是纯 no-op 级别。** 两个序列各只有 1 帧进入弱更新
  （`weak_scale=1.60`），0 重启、0 深度饥饿、输出空洞 0.20 s，ATE
  0.2329 / 0.3816 m —— 与 v4 `ride` 的 0.3064 / 0.3057 m 同量级（差异在 run 间波动内）。
- **V2_03_difficult：P6 没有救回任何东西，反而更差。** weak 5 次里 4 次发散
  （ATE 50.20 / 301.57 / 131.72 / 214.45 m，中位数 131.72 m），唯一成功的一次是
  0.3602 m；受控重启 0/1/1/1/2 次，输出空洞 0.30–35.30 s。也就是说 **P6 既没有
  消除重启、也没有消除空洞**，只是把失败时的幅度从"米级"推到"百米级"。
- **但这次 ARM 会话的对照组同样发散，所以 ARM 数据不能单独归因。** 同一 binary、
  weak 关闭（`ride`）的 4 次 run 全部发散（ATE 5.35 / 7.51 / 10.26 / 15.51 m，
  最大单步位移 27–71 m，空洞 5.95–38.70 s）。这说明**本会话**里 V2_03 的失败由
  "完整受控重启 → Ceres 重新初始化 → 状态跳飞"这条 P6 完全没碰到的路径主导；
  它同时也是 §7.2.5 里 v4 `ride` 的老问题，只是本次会话更严重（v4 `ride` 会话是
  0.39–15.68 m、6 次重启 / 5 runs；v6 `control` 是 5.35–15.51 m、6 次重启 / 4 runs）。
  这再次印证 §7.2.6 的提醒：**ARM 存在会话级方差，单次 run 不能用于归因。**
- **可归因的结论来自 x86（§7.1.4）**：同一个 binary，唯一差别是 P6 两个参数，
  V2_03_difficult 从 0.421859 m / 0 重启 / `weak_scale` 全程 1.00 变成
  69.437068 m / 1 次受控重启 / 17 帧 `weak_scale > 1`（x86 上 ride / soft / base /
  control 四个配置逐位相同，可排除 run 间噪声）。ARM 侧只能给出"无收益、且失败时
  幅度更大"的旁证。
- **行动项**：`schur_starvation_weak_update` **保持默认 `false`**；P6 代码保留为
  诊断/研究开关（默认路径逐位不变，已由 §7.1.4 的 `control` 与 x86 逐位一致性验证）。
- **资源口径**：weak 与 control 的 CPU 110.1–138.2%、RSS 99.0–101.1 MiB、
  p50 313.0–317.4 ms 完全同量级 —— P6 既没有性能收益，也没有性能代价。

### 7.3 测量注意事项（ARM）

- **冷缓存预热**：ARM 上 mcap 冷读只有 23–45 MB/s，会让 rosbag2 报
  `Message queue starved`，看起来像 estimator 卡死。每次 benchmark 前必须
  `sync; cat $HOME/vicon_room2/bags/<dataset>/*.mcap > /dev/null` 预热。
- **DDS domain 隔离**：x86 与 ARM 在同一网段且默认
  `ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET` 时，两边 `ROS_DOMAIN_ID` 相同会互相
  发现并干扰（表现为 rosbag2 player 中途停住）。因此 **x86 与 ARM 绝不并发
  benchmark**，并建议分域（`--domain-id`）运行。

### 7.4 结论与建议

1. **后端重写本身成立（x86 已验证）**：三个 EuRoC 序列 ATE 与 Ceres 同级（V2_03
   反而更好 0.4691 vs 0.5414），p50 延迟约为 Ceres 的 1/3，且规避了 Ceres 在
   V2_02_medium 上的确定性 Cholesky 求解失败。
2. **ARM 上资源收益明确**：CPU −21~36%、p50 −14~24%、p95/p99 更平稳、RSS 略降；
   但当前实现的精度不足——V2_01/V2_02 的 ATE 差于 Ceres，V2_03_difficult 触发
   受控重启后不可用。
3. **因此默认后端仍为 `ceres`**，`schur_ekf` 保持 opt-in；不建议在实机上把
   ARM 平台的默认后端切到 `schur_ekf`。
4. **饥饿判据（P1 时间判据 + P3 饥饿期 Q 放大）的结论：机制有效，但不充分。**
   - x86：base 与 timed 逐位一致，无回归（§7.1.1）。
   - ARM V2_01/V2_02：两种配置都 0 重启，精度/延迟/CPU 差异落在 run 间波动内（§7.2.4）。
   - ARM V2_03：最好一次 run 达到 ATE 0.4551 m、最大输出空洞 0.20 s、轨迹覆盖
     109 s，**证明机制本身可行**；但 4 次重复里 3 次出现 4 次重启、9.25–44.5 s
     输出空洞、ATE 2.57–12.44 m，**不稳定**。
   - 归因：单独时间判据（timeonly）更差，单独 Q 放大（noiseonly）仍被旧的帧数
     判据限制，二者必须同时用。
5. **v3 定位到的失败机理**：ARM 上后端处理频率约 10 Hz（`freq: 10`），单帧 Schur
   更新只要 3–5 ms（`stage timing: update=2.9–4.8 ms`），但**每次受控重启后
   estimator 要重新走 Ceres 初始化**，V2_03 上要 10–44 s 才恢复发布里程计。因此
   「重启」是代价最高的动作，而 1.0 s 的时间判据把 1–2 s 的短饥饿也变成重启，
   形成「短饥饿 → 重启 → 重新初始化 → 又短饥饿」的重启风暴。旧判据（20 连续帧）
   同样会踩这个坑，只是触发频率更低（base 4 次 run 共 6 次 frame_gate 重启）。
6. **优化优先级与状态（P1→P6）**：
   - **P1 时间判据 + P3 饥饿期 Q 放大（已实现、已实测）**：机制有效但不充分
     （§7.1.1、§7.2.4）。
   - **P4 重启冷却 + 两级 Q 放大 + 新默认阈值（已实现、已实测）**：把 ARM 上的重启
     风暴从"最多 4 次"压到"恰好 1 次"，但没有改变"一次完整重启 = 27–37 s 无里程计
     输出"这个代价（§7.2.5）。
   - **P5/P5b 软饥饿重启（已实现、已实测 → 判定不可行）**：ARM 上 3/3 次触发即永久
     发散（ATE 148–194 m，最大单步位移 674–1065 m），默认 `false` 且不建议启用
     （§7.2.6）。结论：**"保留已经漂移的状态 + 重新膨胀 P"这条路不可行**。
   - **P6 饥饿自适应弱更新（已实现、已实测 → 判定无收益/不可行）**：见 §5.1、
     §7.1.4、§7.2.7。思路是在状态**还没漂飞之前**（0.25–1.0 s 的饥饿窗口）用
     "多收一点、少信一点"（门控阈值 ×scale，接受观测信息 ÷scale²）打破
     "饥饿 → 漂移 → 门控崩溃（`no gated tracks`）→ 完整重启"的正反馈；它**不动 P、
     不重置状态**，因此不共享 P5 的失效机理。但实测结果相反：x86 上同一个 binary 只
     切换 P6 两个参数，V2_03_difficult 从 0.4219 m / 0 重启劣化到 69.44 m / 1 重启
     （§7.1.4，可严格归因）；ARM 侧 P6 既没有减少重启和空洞，失败时的幅度还从米级
     变成百米级（§7.2.7，同会话 control 也发散，故 ARM 只作旁证）。原因是饥饿窗口里
     被门控拒绝的正是**误跟踪**外点，放宽 10× 后它们带着整帧系统性偏差进入状态。
     默认 `false`，路径仅作诊断保留。
   - **不要无条件放宽 `schur_min_update_observations`**（§7.2.2 实测更差）。P6 的
     "饥饿期有条件放宽 + 同时降权"实测也没有救回来（§7.1.4），因此"放宽接受门槛"
     这一族方向（无条件 / 有条件 / 降权补偿）在现有数据上都不成立。
   - **根本瓶颈仍未解决**：ARM 上 tracker 跟不上 20 Hz
     （`Tracking queue overflow`，日志按 5 s 节流，实际丢帧数远多于日志条数），使可用
     特征数骤降。这是饥饿的根源，与后端无关，需要从 `tracking_queue_size` / 检测频率 /
     特征跟踪耗时入手。后端只能减轻症状，不能消除。
7. **未覆盖范围**：只跑了 3 个 EuRoC 序列；ARM 上 V2_03_difficult 累计 **25 次
   重复**（P4 矩阵 12 次、P5 矩阵 4 次、P6 矩阵 9 次，已见显著会话级方差），
   V2_01/V2_02 各 3–4 次；未做故障注入，也没有实机 1280×960/30 Hz 数据；
   `Estimator::failureDetection()` 目前仍直接 `return false`，所以
   `failure_reboots=0` 不能作为未发散的证据。另外 ATE 只在被匹配到的位姿上统计，
   **轨迹覆盖（`trajectory_duration_s`）必须与 ATE 一起看**（§7.2.4 口径警告）。

### 7.5 复现命令

x86：

```bash
python3 scripts/benchmark_euroc.py --bags-root ~/vicon_room2/bags \
  --config src/vins_estimator/config/euroc/euroc_mono_imu_config.yaml \
  --binary <vins_node> --output output/backend_rewrite --label <label> \
  --datasets V2_01_easy V2_02_medium V2_03_difficult \
  --parameter backend_type:=schur_ekf
```

ARM（每个序列前先预热 mcap）：

```bash
ssh pinydrone0@192.168.5.60 "sync; cat \$HOME/vicon_room2/bags/V2_01_easy/*.mcap > /dev/null"
scripts/benchmark_remote_arm.sh benchmark --reuse-workspace-build \
  --snapshot-name schur_ekf_v2 --datasets V2_01_easy --label arm_schur_v2_easy \
  --parameter backend_type:=schur_ekf
```

ARM 上的 P4 基线（`ride`，即当前默认值；§7.2.7 里同时作为 P6 的对照组）：

```bash
scripts/benchmark_remote_arm.sh benchmark --reuse-workspace-build \
  --snapshot-name schur_ekf_v6 --datasets V2_03_difficult --label arm_v6_control_difficult \
  --parameter backend_type:=schur_ekf \
  --parameter schur_max_no_update_seconds:=3.0 \
  --parameter schur_starvation_seconds:=0.25 \
  --parameter schur_starvation_noise_scale:=3.0 \
  --parameter schur_starvation_deep_seconds:=1.0 \
  --parameter schur_starvation_deep_noise_scale:=10.0 \
  --parameter schur_restart_cooldown_seconds:=8.0 \
  --parameter schur_max_consecutive_failures:=60
```

P5 软重启（**仅诊断，实测不可行**，§7.2.6）：在上面基础上再加

```bash
  --parameter schur_starvation_soft_restart:=true \
  --parameter schur_soft_restart_max_seconds:=10.0
```

P6 弱更新（§7.2.7）：在上面基础上再加

```bash
  --parameter schur_starvation_weak_update:=true \
  --parameter schur_weak_update_max_scale:=10.0
```

> A/B 纪律：P5/P6 这类 opt-in 开关必须用**同一个 binary**、同一会话内 weak on/off
> 交替各跑 ≥3 次，否则 ARM 的会话级方差（§7.2.6、§7.2.7）会把开关效果淹没。

ARM 结果解析（每个 run 目录下的 `metrics.json` + `estimator.log`）：

```bash
# 关键计数：软重启 / 升级 / 完整重启 / 深度饥饿 / 弱更新 / queue overflow
for pat in 'soft restart at frame' 'escalating to a full' \
           'scheduling a controlled estimator restart' 'deep starvation started' \
           'weak-update frame' 'Tracking queue overflow'; do
  printf '%-45s %s\n' "$pat" "$(grep -c "$pat" estimator.log)"
done
# 周期汇总（累计计数 + 当前噪声倍数 + 最长无更新间隔）
grep 'starvation summary' estimator.log | tail -1
# 状态是否跳飞：相邻位姿的最大欧氏距离（trajectory.csv）
python3 -c "
import csv,math
rows=list(csv.DictReader(open('trajectory.csv')))
prev=None; mx=0.0
for r in rows:
    p=(float(r['x']),float(r['y']),float(r['z']))
    if prev is not None: mx=max(mx, math.dist(p,prev))
    prev=p
print('max_step=%.2f m' % mx)"
```
