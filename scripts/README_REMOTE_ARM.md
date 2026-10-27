# Remote ARM benchmark wrapper

`scripts/benchmark_remote_arm.sh` is a foreground SSH wrapper for reproducing
the ARM build, unit tests, and EuRoC estimator benchmark. It does not start
flight control, `loop_fusion`, `global_fusion`, or RViz, and it does not use
`sudo` or delete files.

## Inputs

The defaults target the current development setup:

- SSH host: `pinydrone0@192.168.5.60` (key-based; no password is stored)
- remote workspace: `$HOME/Code/Piny_drone_nav`
- bags: `$HOME/vicon_room2/bags`
- results: `$HOME/vins_remote_eval_20261019`
- ROS domain: `173`
- datasets: `V2_01_easy V2_02_medium V2_03_difficult`

Every path and the domain can be overridden by a command-line option or the
corresponding environment variable (`SSH_HOST`, `REMOTE_REPO`, `REMOTE_BAGS`,
`REMOTE_OUTPUT`, `ROS_DOMAIN_ID`, `DATASETS`, `LABEL`, `RATE`, `REPEATS`,
`BUILD_JOBS`, `SNAPSHOT_NAME`, `CONFIG`, `REUSE_WORKSPACE_BUILD`).
Repeatable `--parameter NAME:=VALUE` is CLI-only: values are passed as individual
arguments to the existing benchmark, without editing the YAML or using `eval`.

## Steps

Build against the existing workspace build/install directories (incremental,
as used for the current-head build):

```bash
scripts/benchmark_remote_arm.sh build \
  --jobs 2 --reuse-workspace-build \
  --snapshot-name current_head_vins_node
```

The build is fixed to Release, `BUILD_TESTING=ON`, LTO OFF, and
`VINS_ENABLE_NATIVE_OPTIMIZATION=OFF`. `--jobs` is restricted to 1 or 2.
Without `--reuse-workspace-build`, the script uses `<output>/build` and
`<output>/install` instead of the workspace directories. An existing
snapshot is rejected before compilation; use `--overwrite-snapshot` only
when replacing a snapshot you own. A submodule `.git` file is accepted
(the wrapper uses `git -C ... rev-parse --is-inside-work-tree`).

Run the C++ tests for `vins_common` and `vins`, then the Python script tests:

```bash
scripts/benchmark_remote_arm.sh test --reuse-workspace-build
```

Run one complete sequence or the default three sequences:

```bash
scripts/benchmark_remote_arm.sh benchmark \
  --reuse-workspace-build \
  --datasets V2_01_easy \
  --label current_head_easy
```

The benchmark step calls only `scripts/benchmark_euroc.py`. By default it
uses the EuRoC mono-IMU configuration in the repository. A candidate config
outside the repository can be passed explicitly and is recorded with its
SHA256 in `benchmark_provenance.txt`:

```bash
scripts/benchmark_remote_arm.sh benchmark \
  --reuse-workspace-build \
  --datasets V2_01_easy \
  --label max120_easy \
  --config vins_remote_eval_20261019/configs/max120.yaml
```

The relative `--config` path is resolved against the remote `$HOME`; absolute
paths are also accepted. Every step takes a non-blocking per-repo `flock` and
refuses to overlap another wrapper step. The test and benchmark steps never
run a build.

## Artifacts

The wrapper creates or uses:

- `<output>/evidence/` — build/test/benchmark commands, console logs, status,
  binary provenance, and the resolved config path/SHA256
- `<output>/binaries/<snapshot-name>` — copied with `cp -L` so the benchmark
  does not depend on the install symlink
- `<output>/benchmark/<label>/` — existing `benchmark_euroc.py` output

`deployed_vins_node` is reserved for the old deployment and is never
overwritten by the wrapper. Choose a new benchmark label for each comparison;
the wrapper refuses to reuse an existing label directory.

## Stage 2: single-variable OpenCV thread experiment (prepared, not started)

**Wait for explicit release of the remote CPU after the main current-head
three-bag run.** An idle process check is not permission to start. Do not run a
build or tests alongside either benchmark, and do not use the `all` step here.
The commands below are manual, not a scheduled/background experiment.

Before executing, verify that the source HEAD is
`8c5e7d629a41d8e0f23a4a67ef5076c8e04eeb5b` and the existing
`binaries/current_head_vins_node` SHA256 is
`7627574c5a12e05486dafec1d84fea08ce1f9a7bf1a72d702f9a291c6764867b`.
Do not rebuild or replace this snapshot for the comparison. The wrapper records
the observed HEAD/hash/config hash/command, and the benchmark's `manifest.json`
records the parameter list and full config text; check them against the main
baseline artifacts, not just the label.

From the local `src/VINS-Fusion` directory, **only after release**:

```bash
scripts/benchmark_remote_arm.sh benchmark \
  --host pinydrone0@192.168.5.60 \
  --repo /home/pinydrone0/Code/Piny_drone_nav \
  --bags /home/pinydrone0/vicon_room2/bags \
  --output /home/pinydrone0/vins_remote_eval_20261019 \
  --reuse-workspace-build \
  --snapshot-name current_head_vins_node \
  --domain-id 173 --rate 1.0 --repeats 1 \
  --datasets V2_01_easy --label arm_cv2_easy \
  --parameter opencv_threads:=2
```

This runs the complete easy bag with the same binary/default EuRoC mono YAML.
Only the ROS parameter `opencv_threads` changes from 1 to 2. Feature count 200,
`freq=10`, `flow_back=1`, solver budget 0.04 s, 8 iterations, 752x480 images, and
calibration remain unchanged. The camera input is 20 Hz, not 10 Hz. There is no
`--duration` limit and no automatic max150/max120 follow-up.

benchmark 退出后，只读取已有 metrics 与 `estimator.log`，不要启动 ROS，也不要
重写 metrics。下面的只读命令同样不应在主 benchmark 运行期间执行。每个 run 优先
解析最后一条带累计字段的 `Tracker final stats:`；若旧产物没有 final 行，则回退
到最后一条 `Tracker periodic stats:`：

```bash
ssh -o BatchMode=yes pinydrone0@192.168.5.60 'python3 -' <<'PYREPORT'
import json
import re
from pathlib import Path

root = Path.home() / "vins_remote_eval_20261019/benchmark/arm_cv2_easy"
print("metrics: " + str(root / "summary.json"))
print((root / "summary.json").read_text())
expected_images = {"V2_01_easy": 2280, "V2_02_medium": 2348, "V2_03_difficult": 1922}
runs = sorted(root.glob("*_r*/metrics.json"))
if not runs:
    raise SystemExit("No complete run metrics found")
for metrics_path in runs:
    metrics = json.loads(metrics_path.read_text())
    expected = expected_images[metrics["dataset"]]
    received = metrics["received_images"]
    log = metrics_path.parent / "estimator.log"
    lines = log.read_text(errors="replace").splitlines()
    warnings = [line for line in lines if "Tracking queue overflow:" in line]
    overflow_events = None
    overflow_dropped_pairs = None
    overflow_stats_source = "missing"
    for source, prefix in (("final", "Tracker final stats:"),
                           ("periodic", "Tracker periodic stats:")):
        for line in reversed([item for item in lines if prefix in item]):
            match = re.search(
                r"overflow_events=(\d+) overflow_dropped_pairs=(\d+)", line)
            if match:
                overflow_events = int(match.group(1))
                overflow_dropped_pairs = int(match.group(2))
                overflow_stats_source = source
                break
        if overflow_events is not None:
            break
    print(json.dumps({
        "run": metrics_path.parent.name,
        "bag_expected_images": expected,
        "observer_received_images": received,
        "expected_minus_received": expected - received,
        "observer_image_coverage_percent": 100 * received / expected,
        "latency_misses": metrics["latency_misses"],
        "estimator_log": str(log),
        "exact_overflow_events": overflow_events,
        "exact_overflow_dropped_pairs": overflow_dropped_pairs,
        "overflow_stats_source": overflow_stats_source,
        "tracking_overflow_warning_lines": len(warnings),
        "first_overflow_warning": warnings[0] if warnings else None,
        "last_overflow_warning": warnings[-1] if warnings else None,
    }, indent=2))
PYREPORT
```

精确丢帧量必须看周期或 final stats 中的累计字段，不能数告警行：
`overflow_events` 是 tracking queue 达到上限并执行丢弃策略的累计次数，
`overflow_dropped_pairs` 是实际丢弃的 stereo frame pair 累计数。`latest` 策略
一次 overflow 可能清掉多帧，因此 dropped pair 可以大于 event 数。周期日志消费
的是窗口计时/均值，不会清零这两个累计值；`Tracker final stats` 可能显示
`frames=0`，但 `total_frames` 与 overflow 累计仍然有效。可直接检查原始行：

```bash
grep -E 'Tracker (periodic|final) stats:' estimator.log | tail -5
grep 'overflow_events=' estimator.log | tail -1
```

`Tracking queue overflow` 告警现在只在该进程第 1 次以及每 100 次 overflow 时
输出，不再按 5 s 时间节流；它只是提示，不是精确丢帧计数。旧产物中的
“5 s 节流告警行数”不能与新累计字段直接等同。

tracker 的默认语义保持不变：`tracking_queue_policy=drop_oldest`（opt-in
`latest` 时才丢弃排队旧帧、只处理最新帧）、`tracker_detection_interval=1`
（每帧按原逻辑补点；大于 1 时常规检测最多每 N 帧一次，但特征亏空超过
`0.5*MAX_CNT` 仍会紧急检测）、`tracker_stats_interval=100`。stats 中
`queue_wait_ms`/`decode_ms`/`track_ms`/`pyramid_ms`/`flow_ms`/`detect_ms`/
`input_image_ms` 均为 `mean/max`：分别表示入队到 worker 开始的等待、图像解码、
`trackImage()` 总耗时、KLT 金字塔构建、时序/双目光流、特征拒绝与检测，以及整个
`Estimator::inputImage()` 调用；不是百分位。

Input coverage 仍必须伴随每次候选比较：raw bag 期望帧数为 easy **2280**、
medium **2348**、difficult **1922**。Difficult 原始记录本身已有缺帧，1922 是
实际 bag 帧数，不是 20 Hz × 时长算出的理论值。比较各次 run 的
`received_images`，并将未修改 metrics 中的 `latency_misses` 与延迟百分位一起
报告。未匹配位姿不进入这些延迟百分位。这些是 observer 计数，不能证明 estimator
送达或 tracker 实际接收；即使总数相等，也不能证明所有原始帧时间戳都唯一送达。

此前报告的历史 current-head medium run（旧 5 s 节流口径）观测到
**2319/2348 images (29 fewer)**。
The cause is undetermined; absence of a player starvation log does not establish
loss-free playback/delivery. Treat this as a coverage limitation in comparisons,
not as proven tracker loss or a clean full-frame-input baseline. Do not change
the existing benchmark's QoS, counters, or timing to conceal this difference.

同样，`received_images - output_poses` 不是 tracker 丢帧计数：初始化和后端抽帧
也会影响它。旧告警行变少本身不能证明实时性改善；必须比较精确
`overflow_events`、`overflow_dropped_pairs`、queue wait 与 track/detect 的
mean/max，并同时报告输入覆盖和延迟。

### Initialization and trajectory coverage

Do not rank configurations by ATE alone. The reported current-head difficult
run has only 661 poses and `trajectory_duration_s=74.75` over an approximately
116.81 s bag: its low ATE does not describe coverage of the entire sequence.
For each comparison, additionally record:

- The first and last camera **header** timestamps from the bag, not an assumed
  timestamp obtained by multiplying frame count by 50 ms (especially difficult).
- The first/last published trajectory header timestamps and their span. Report
  first-pose minus first-camera time as **first-output delay in sensor time**,
  a proxy for initialization/startup coverage, not exact internal init latency.
- The last-camera minus last-pose time (tail not covered), output span relative
  to camera span, and any `Initialization finish!` log lines as separate evidence.
- Existing `trajectory_duration_s` and `matched_poses`, unchanged. The existing
  ATE function filters to Vicon-valid timestamps; its matched span need not be
  identical to the full published trajectory span. Neither span alone proves
  continuous coverage, so include output gaps and pose counts as well.

Do post-processing only after the measured run exits. Read existing artifacts;
do not instrument or alter the benchmark, adjust time/scale alignment, or
silently change the accuracy comparison to a different trajectory interval.

Compare full metrics (wall/CPU, P50/P95/P99, received images/output/matched poses,
latency misses, RSS/threads, ATE and output gaps/reboots) against the same-binary
cv1 easy run. A53 results decide, not local/x86 timings. If cv2 is worse, do not
recommend it. If latency remains above the reference 100–200 ms range (not a
user-defined acceptance criterion), report that result first; max150 is the next candidate **only after a separate go-ahead**,
with the OpenCV setting explicitly chosen so that the next comparison changes
only one variable. Do not silently combine cv2 and feature reduction.

## Guardrails

- No password is embedded; SSH runs with `BatchMode=yes`.
- No background process, daemon, or cleanup step is created.
- Build, test, and benchmark share a non-blocking lock keyed by the
  resolved remote repo path, so they cannot overlap. Conflict detection
  inspects real executables/argv tokens rather than fuzzy-searching an
  ancestor shell's command line; a snapshot argument containing
  `vins_node` does not trigger a false positive.
- The build uses `--packages-up-to vins` to include its declared dependency
  closure. On a fresh isolated workspace, colcon also needs the `package.sh`
  files for the `loop_fusion`/`global_fusion` runtime dependencies. These
  packages are built but their nodes are never launched by this wrapper.
- The wrapper never edits calibration/config files, CPU governor, or
  flight-controller state. Explicit `--parameter` ROS overrides are recorded.
- Run build, test, and benchmark separately when collecting measurements.


## 2026-10-27 P7 / tracker 验证

最新的独立工作区、二进制哈希、逐轮结果和限制见
[ARM_P7_TRACKER_EVALUATION.md](../docs/ARM_P7_TRACKER_EVALUATION.md)。
P7 已出现严重漂移，保持默认关闭；隔帧补点与 `latest` 队列也不能仅凭延迟降低推荐。
上面的 Stage 2 是旧快照实验，不能与本轮数据混用。

计算整段 tracker 耗时时，应汇总所有 `Tracker periodic stats` 和 `Tracker final stats`：
对每个 `mean` 取 `sum(frames * mean) / sum(frames)`，`max` 取所有窗口最大值，
并检查 `sum(frames) == final.total_frames`。最终行仅可直接读取**累计**溢出计数。
