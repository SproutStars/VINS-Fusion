# ARM 平台 P0/P1/P2 优化与故障注入验证

本文记录在 `docs/ARM_FAST_GRID_EVALUATION.md` 给出的优先级基础上，对
VINS-Fusion 做的第二轮加固与性能分析，并在 Orange Pi Zero 3（4×Cortex-A53）
上使用 `~/vicon_room2/bags` 的 EuRoC V2 数据完成同二进制验证。

> 结论边界：**崩溃根因已修复并通过故障注入验证；CPU 与精度没有统计显著的
> 普遍提升。** `retry`/`revert` 在本板卡的可用算力预算下无法完成初始化，
> 生产默认仍应保持 `warn`。

---

## 1. 本轮优先级与完成情况

| 优先级 | 项目 | 状态 | 说明 |
|---|---|---|---|
| P0 | Ceres 超时后的事务化 | 完成 | 失败帧不再执行 `double2vector`、外点剔除、边缘化与状态发布 |
| P0 | 初始化失败后的历史失控 | 完成 | 修复 `RegionsAlias` 崩溃根因（见 §2） |
| P1 | RANSAC 越界 | 完成 | 删除过期 `cur_un_pts` 的 `reduceVector`，增加 mask 长度/退化防御 |
| P1 | 超时判定依赖英文文本 | 完成 | 新增 `solver_timeout.h`，改用耗时 + `NO_CONVERGENCE` |
| P2 | `openExEstimation` 事务泄漏 | 完成 | 仅在 solve 成功后置位 |
| P2 | 边缘化线程数可配置 | 完成 | `marginalization_threads`，ARM A/B 结论见 §5.4 |
| P2 | 阶段耗时分解 | 完成 | `stage_timing` 输出 solve/边缘化等分段耗时 |
| P2 | 投影因子 `TicToc` 开销 | 完成 | 删除 4 个投影因子每残差构造的计时对象 |

未做：MSCKF/EKF 后端重写、默认开启 FAST-Grid。原因见
`docs/ARM_FAST_GRID_EVALUATION.md` 与本文 §7。

---

## 2. P0 崩溃根因（`RegionsAlias`）

故障注入日志（修复前）：

```text
Ceres timeout (policy=retry): first 5 ms timed out, retry 10 ms failed
Optimization failed: ... current-frame inputs already advanced ...
F ... problem_impl.cc:78 Check failed: !RegionsAlias(... size 4 ... size 3 ...)
```

根因链条：

1. INITIAL 阶段 `optimization()` 失败后直接 `return`，没有走 `initialStructure()`
   失败路径中的 `slideWindow()`。
2. 连续失败使 `all_image_frame` 持续累积，旧帧的 `is_key_frame=true` 仍然保留。
3. 下一次 `visualInitialAlign()` 遍历整个 map 得到 `kv`，缺少上界保护，
   越界写 `Vs[kv]`，破坏栈上的 VLA。
4. 相邻 VLA `c_rotation[frame_num][4]` 与 `c_translation[frame_num][3]` 地址重叠，
   最终触发 Ceres 的参数块别名检查。

修复：

- 初始化优化失败时调用 `maintainInitializationHistoryAfterFailure()`：
  沿用原 `slideWindow()` 语义，再按 `Headers[0..frame_count]` 裁剪
  `all_image_frame`，同步释放被裁剪帧的预积分对象；仍异常则受控 reset。
- `visualInitialAlign()` 先清除历史 keyframe 标记，并校验 `kv` 上界与最终数量。
- `GlobalSFM` 的 `c_rotation`/`c_translation` 改为独立 `std::vector<std::array>`
  存储，从结构上消除相邻 VLA 别名。
- 非线性优化失败不再在 `processImage()` 锁内 reset，而是由
  `processMeasurements()` 释放锁后执行 `clearState()+setParameter()`。

**已知局限**：失败帧的 `f_manager`/`all_image_frame`/IMU 输入在 solve 前已消费，
无法完整事务回滚。INITIAL 走"滑窗维护 + 裁剪 + 重试"，NON_LINEAR 走受控 reset；
两者都在日志中显式说明。

---

## 3. 验证环境

| 项 | 值 |
|---|---|
| 板卡 | Orange Pi Zero 3，4×Cortex-A53 AArch64 |
| 远端 | `pinydrone0@192.168.5.60` |
| 数据 | `~/vicon_room2/bags/{V2_01_easy,V2_02_medium,V2_03_difficult}` |
| 结果根 | `~/vins_remote_eval_20261022` |
| 候选二进制 | `binaries/p0p1fix2_vins_node` |
| SHA-256 | `8035e836b7c0e9df85cae9a8e9af2ae6a702dda4e9a8932a4a2b4085ca43c815` |
| 基线二进制 | `binaries/baseline_gftt_vins_node`（旧 GFTT） |
| 公共参数 | `max_cnt=120`、`flow_back=1`、`marginalization_threads=4`、`show_track=1`、`save_image=1` |

ATE 为刚性位置对齐 RMSE；跨运行比较使用**共同时间窗口**重算（不同运行覆盖
长度不同，直接比较原始 ATE 不公平）。

### 3.1 回归测试

| 环境 | 结果 |
|---|---|
| 本地 x86_64 `ctest` | 7/7 通过 |
| 本地 `pytest scripts/tests` | 24 passed |
| **ARM（同一构建）`colcon test`** | **49 tests, 0 errors, 0 failures**，`ctest` 7/7 |
| ARM `pytest scripts/tests` | 24 passed（19.8 s） |

证据：`~/vins_remote_eval_20261022/evidence/test_20261022T131840Z.log`。

---

## 4. 故障注入结果（P0 验证）

配置：medium、1 倍速、`duration=40 s`、`max_solver_time=0.006`。

| 策略 | 标签 | 崩溃 | `RegionsAlias` | 超时次数 | 输出位姿 | ATE |
|---|---|---|---:|---:|---:|---:|---:|
| retry | `arm_p0p1fix2_stress_retry_40s` | 无 | 0 | 9 | 0 | — |
| revert | `arm_p0p1fix2_stress_revert_40s` | 无 | 0 | 17 | 0 | — |

- 修复前同一场景在 retry 下触发 `Check failed: !RegionsAlias` 并退出；
  修复后进程正常结束，`failure_reboots=0`，日志中无 `Check failed`/`terminate`。
- 两组的 `Optimization failed` 次数与超时次数一致，说明每个失败帧都被正确
  拦在 `double2vector`/边缘化之前。
- 40 s 内输出 0 个位姿：6 ms 预算下初始化 BA 无法在 1–2 次迭代内收敛，
  属于故障注入的预期表现，不是新回归。

### 4.1 重要负面发现：retry/revert 在本板卡不可用

进一步用较宽松预算验证：

| 配置 | 序列 | 首次预算 | 重试预算 | 重试成功 | 输出位姿 |
|---|---|---:|---:|---:|---:|
| `retry_mild.yaml` | easy 40 s | 16 ms | 32 ms | 0 | 0 |
| `retry.yaml` | easy 40 s | 32 ms | 64 ms | 0 | 0 |

典型日志：

```text
Ceres timeout (policy=retry): first solve hit max_solver_time=32 ms after 41.9 ms
and 2 iterations; retry with 64 ms failed after 67.1 ms and 3 iterations;
solver parameter rollback=true
```

原因：`retry`/`revert` 把"超时"直接判为失败，而本板卡上窗口 BA 几乎每次都会
超时（`max_solver_time=0.04` 下 warn 模式累计超时 >1000 次/序列）。因此这两个
策略会拒绝所有"截断但可用"的解，初始化永远无法完成。**它们只能作为故障注入
手段，不能作为生产默认。**

---

## 5. 全序列性能与精度

配置：`stage.yaml`（= `base.yaml` + `stage_timing=1`），`max_solver_time=0.04`，
`solver_timeout_policy=warn`，1 倍速完整序列，各 1 次。

### 5.1 端到端指标

| 序列 | 运行 | CPU s | P95 ms | P99 ms | ATE m | poses |
|---|---|---:|---:|---:|---:|---:|
| easy | 旧 GFTT | 225.39 | 709.34 | — | 0.21211 | — |
| easy | p0p1 | 224.60 | 659.54 | — | 0.17517 | — |
| easy | p0p1fix | 226.23 | 801.88 | — | 0.18915 | — |
| easy | **p0p1fix2** | **221.63** | 715.70 | 849.50 | 0.21824 | 1084 |
| medium | 旧 GFTT | 221.48 | 471.70 | — | 0.27873 | — |
| medium | p0p1 | 221.03 | 499.72 | — | 0.26637 | — |
| medium | p0p1fix | 219.96 | 455.79 | — | 0.29526 | — |
| medium | **p0p1fix2** | **220.37** | 578.66 | 941.11 | 0.26553 | 1107 |
| difficult | 旧 GFTT | 197.31 | 525.54 | — | 0.53257 | — |
| difficult | p0p1 | 196.96 | 528.18 | — | 0.43368 | — |
| difficult | p0p1fix | 197.94 | 538.23 | — | 0.52103 | — |
| difficult | **p0p1fix2** | **197.06** | 532.36 | 1229.69 | 0.48228 | 1087 |

- CPU 与旧基线持平（-1.7% / -0.5% / -0.1%），在运行噪声范围内。
- P95 运行间波动大（历史同配置 411–546 ms），单次差异不能作为显著性结论。
- `failure_reboots=0` 只说明未触发重启；`failureDetection()` 仍直接返回
  `false`，不能据此断言未发散。

### 5.2 共同时间窗口 ATE

| 序列 | 旧 GFTT | p0p1 | p0p1fix | **p0p1fix2** | 窗口 |
|---|---:|---:|---:|---:|---:|
| easy | 0.21217 | **0.17570** | 0.18950 | 0.21782 | 107.50 s |
| medium | 0.27742 | 0.26656 | 0.28664 | **0.25607** | 107.20 s |
| difficult | 0.53109 | **0.43333** | 0.51960 | 0.48147 | 109.25 s |

相对旧基线：easy +2.7%、medium -7.7%、difficult -9.3%。方向仍随序列变化，
且 GFTT 历史重复运行间 ATE 波动可达 14%，因此**不能宣称精度提升**；
可以确认的是**没有引入稳定的精度退化**。

### 5.3 阶段耗时分解（每 100 帧平均）

| 序列 | vector2double | problem | **solve** | double2vector | **边缘化** |
|---|---:|---:|---:|---:|---:|
| easy | 0.31–0.41 ms | 2.7–3.8 ms | **43.7–48.5 ms** | 0.38–0.52 ms | **25.4–34.7 ms** |
| medium | 0.44–0.74 ms | 1.8–3.1 ms | **40.4–45.5 ms** | 0.53–0.87 ms | **14.9–28.7 ms** |
| difficult | 0.53–0.71 ms | 0.86–1.94 ms | **33.4–41.2 ms** | 0.70–0.88 ms | **8.3–18.6 ms** |

结论：**后端 solve 与边缘化是绝对主项**，两者合计远大于前端与状态拷贝
（vector2double/double2vector 均 <1 ms）。在不动后端结构的前提下，
ARM 性能优化空间有限。

### 5.4 其它 A/B 结论（沿用上一轮数据）

- 边缘化线程：4 线程 P95 455.79/499.72 ms，1 线程 522.97/546.58 ms，
  2 线程 474.57/476.11 ms；CPU 差异很小，P99/ATE 不稳定。
  **保持 4 线程**。
- FAST-Grid 前端：CPU -18.6%～-32.3%、P95 -61.7%～-79.1%，但共同窗口 ATE
  在 -13.4%～+10.7% 间摆动。保持 **opt-in**。

---

## 6. RANSAC 外点剔除

配置：`ransac.yaml`（`ransac_outlier_rejection=1`），medium 40 s。

| 运行 | CPU s | P95 ms | 位姿 | ATE m |
|---|---:|---:|---:|---:|
| base（RANSAC 关） | 75.21 | 824.62 | 336 | **0.24260** |
| RANSAC（开） | 57.84 | 385.79 | 355 | **404.98** |

- 修复后代码路径**不再越界、不再崩溃**（`cur_un_pts` 的过期 `reduceVector`
  已删除，退化 F 矩阵有防御），这是本轮 P1 的目标。
- 但在 `max_cnt=120` 下开启 RANSAC 会显著减少可跟踪点，medium 该段**发散**。
  因此 `ransac_outlier_rejection` 必须保持默认 **0**，不能作为精度改进项。

---

## 7. 生产建议

```yaml
solver_timeout_policy: "warn"      # retry/revert 在本板卡无法完成初始化
marginalization_threads: 4
ransac_outlier_rejection: 0
stage_timing: 0                    # 仅诊断时置 1
feature_detector: "gftt"           # FAST-Grid 仍需按序列验证精度
```

- `warn` 的代价是接受大量截断解（easy/medium 每序列 >1000 次超时记录），
  会污染边缘化先验；这是下一阶段最值得投入的后端问题。
- 若要真正降低 ARM 后端开销，需要重写优化/边缘化调度（如按残差稀疏性
  分块、减少线性化点重建），而不是继续调前端参数。

---

## 8. 复现

```bash
cd /home/sprout_stars/rmcode/VINS-Fusion

# 1) 同步源码到远端工作区（示例）
rsync -az src/vins_estimator/src/ pinydrone0@192.168.5.60:Code/Piny_drone_nav/src/VINS-Fusion/src/vins_estimator/src/
rsync -az src/vins_estimator/test/ pinydrone0@192.168.5.60:Code/Piny_drone_nav/src/VINS-Fusion/src/vins_estimator/test/

# 2) ARM 构建新快照
./scripts/benchmark_remote_arm.sh build --repo Code/Piny_drone_nav \
  --output vins_remote_eval_20261022 --snapshot-name p0p1fix2_vins_node \
  --reuse-workspace-build --allow-dirty-source --jobs 2

# 3) 故障注入（预期：无崩溃、无 RegionsAlias）
./scripts/benchmark_remote_arm.sh benchmark --repo Code/Piny_drone_nav \
  --output vins_remote_eval_20261022 --snapshot-name p0p1fix2_vins_node \
  --config vins_remote_eval_20261022/configs/stress_retry.yaml \
  --datasets V2_02_medium --label arm_p0p1fix2_stress_retry_40s --duration 40

# 4) 全序列阶段分解
./scripts/benchmark_remote_arm.sh benchmark --repo Code/Piny_drone_nav \
  --output vins_remote_eval_20261022 --snapshot-name p0p1fix2_vins_node \
  --config vins_remote_eval_20261022/configs/stage.yaml \
  --datasets V2_02_medium --label arm_p0p1fix2_stage_medium
```

本地回归：

```bash
cd build/vins && ctest --output-on-failure -j4
cd ../.. && git diff --check
env -u COLCON_CURRENT_PREFIX -u AMENT_PREFIX_PATH -u CMAKE_PREFIX_PATH \
  -u LD_LIBRARY_PATH -u PYTHONPATH bash -lc \
  'source /opt/ros/jazzy/setup.bash && source install/local_setup.bash && \
   python3 -m pytest scripts/tests -q'
```

---

## 9. 数据位置

| 内容 | 路径 |
|---|---|
| ARM 结果根 | `~/vins_remote_eval_20261022/benchmark/` |
| 阶段分解 | `arm_p0p1fix2_stage_{easy,medium,difficult}/` |
| 故障注入 | `arm_p0p1fix2_stress_{retry,revert}_40s/` |
| RANSAC A/B | `arm_p0p1fix2_ransac_40s/`、`arm_p0p1fix2_base_medium_40s/` |
| 二进制快照 | `~/vins_remote_eval_20261022/binaries/p0p1fix2_vins_node` |
| 构建证据 | `~/vins_remote_eval_20261022/evidence/build_provenance.txt` |
