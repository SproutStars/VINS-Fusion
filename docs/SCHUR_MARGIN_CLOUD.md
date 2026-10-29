# Schur 历史点云质量

`/vins_estimator/margin_cloud` 在 RViz 中以 `History Points` 显示，默认累积
1000 秒。它用于展示离开估计窗口的历史特征；持续跟踪中、深度不可靠的点如果反复
发布，会在历史图中形成散点和拖影。它也不是全局一致的稠密地图，原始世界坐标点
不会随着未来的回环优化自动更新。

前一版为恢复 Schur 点云输出，将本次消费的轨迹，以及任何含最旧 clone 观测的
轨迹都加入 margin_cloud。后者可能并未通过更新门控，甚至仍在窗口中继续跟踪；
三角化通过 5 像素 RMS 检查也不能证明深度可靠，低视差时仍可出现很大的深度误差。

## 验证方法

使用同一 EuRoC 相机配置、1 倍速完整回放三组 `~/vicon_room2/bags` 数据，记录
原始点坐标、时间戳和原始 VIO 轨迹：

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 scripts/validate_schur_loop.py --save-margin \
  --output output/schur_margin_validation/after
```

`margin_cloud.npz` 可用 `scripts/compare_margin_clouds.py` 生成不裁剪的等比例
前后对比图和距离分位数。例如：

```bash
python3 scripts/compare_margin_clouds.py \
  --inputs output/schur_margin_validation/before/V2_01_easy/margin_cloud.npz \
           output/schur_margin_validation/after/V2_01_easy/margin_cloud.npz \
  --labels before after \
  --output output/schur_margin_validation/easy_comparison
```

统计剔除首次 VIO 位姿之后的前 20 秒，避免 Ceres 初始化期混入 Schur 结果。
图中使用相同坐标轴且不裁剪点，橙色线为 VIO 轨迹。距离统计是点到发布时刻机体
位置的距离，不是到真实表面的误差；EuRoC Vicon 只有轨迹真值，因此不能仅靠这些
统计宣称地图精度达到 Ceres。需同时检查有足够有效点、原始 VIO 不变和回环回归。

## 修复规则

只将 `summary.applied` 为真时实际使用的轨迹作为 margin 候选，不再因为观测包含
clone 0 就发布。候选还需满足：

- 至少 3 个不同 clone 的观测；
- 实际相机到点的世界射线最大夹角至少 1°；
- 重投影 RMS 不超过 2 px，任意单帧残差不超过 3 px；
- 由像素噪声和投影雅可比形成的三维点信息矩阵可逆，各视角的深度标准差/深度
  不超过 15%。该值以相机位姿固定为条件，不包括全部位姿不确定性。

同一连续跟踪特征 ID 首次满足这些条件后只发布一次，跨 EKF 消费和后续测量段
仍保留去重记录。首次不合格不会阻止后来合格的测量段发布。ID 同时离开当前图像
和保留历史后释放记录，内存随存活轨迹数而非运行时间增长。完整估计器重置或
重新初始化时清空；hold/测量窗口重建期间对仍存活的 ID 保持去重。

这些规则只作用于历史点云，普通点云、关键帧和 EKF 更新不受此质量门控影响。
不会裁剪固定距离，也没有空间平滑。代价是历史点数减少，未产生新的可靠点时
发布空消息是正常行为。

`vins` 构建成功，其测试汇总 123 项、零失败；`loop_fusion` 另有 4 项通过，共 127 项，其中快照/历史点云专门测试 9 项。
新增场景覆盖弱视差、重投影误差、深度精度不足、远但可靠的点、消费资格、跨测量
段去重和内存边界。

## easy 实测及 Ceres 对照

剔除首次 VIO 后 20 秒，统一统计点到发布时间机体位置的距离：

| 模式 | 历史点数 | 距离 p95 | 距离 p99 | 超过 20 m 的点数 |
| --- | ---: | ---: | ---: | ---: |
| Schur 修改前 | 31200 | 10.126 m | 39.833 m | 919 |
| Schur 修改后 | 6762 | 5.082 m | 6.917 m | 0 |
| Ceres 对照 | 11548 | 4.923 m | 7.120 m | 12 |

点数下降包含去除重复发布与排除不可靠几何两部分，不能直接解释为丢失同等数量的
独立地图特征。修复后的距离尾部接近 Ceres，但这不是点到真实表面的精度证明。
对比图保存在 `output/schur_margin_validation/easy_comparison.png`。

修复需要重启估计节点加载新二进制。RViz 已累积的旧历史点不会因为收到新点云
自动消失；对比时请清空 `History Points` 显示的缓存或重启 RViz，再回放数据。
无需改变原来的 1000 秒显示累积时间。

## 三组最终验收结果

| 序列 | 历史点数（前 → 后） | 距离 p99（前 → 后） | 超过 20 m 点数（前 → 后） | 回环边（前/后） |
| --- | ---: | ---: | ---: | ---: |
| V2_01_easy | 31200 → 6762 | 39.83 → 6.92 m | 919 → 0 | 67/67 |
| V2_02_medium | 32140 → 10662 | 35.77 → 7.46 m | 567 → 0 | 83/83 |
| V2_03_difficult | 25607 → 8039 | 14.32 → 6.31 m | 152 → 0 | 16/16 |

三组均收到 bag 中全部图像，无跟踪队列溢出。原始 `vio.csv` 均与修改前逐字节一致；普通点云与关键帧的全部统计一致；回环边分别保持 67、83、16 条。所有功能验收项通过。

结果摘要：`docs/SCHUR_MARGIN_RESULTS.json`；完整日志、NPZ 点坐标、配置与二进制 SHA-256：`output/schur_margin_validation/`。
