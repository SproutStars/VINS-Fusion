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

After the benchmark exits, read the **complete** existing metrics and count the
emitted overflow warnings without launching ROS or rewriting metrics. This
separate read-only command is also intentionally not run during the main run:

```bash
ssh -o BatchMode=yes pinydrone0@192.168.5.60 'python3 -' <<'PYREPORT'
import json
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
    print(json.dumps({
        "run": metrics_path.parent.name,
        "bag_expected_images": expected,
        "observer_received_images": received,
        "expected_minus_received": expected - received,
        "observer_image_coverage_percent": 100 * received / expected,
        "latency_misses": metrics["latency_misses"],
        "estimator_log": str(log),
        "tracking_overflow_warning_lines": len(warnings),
        "first_overflow_warning": warnings[0] if warnings else None,
        "last_overflow_warning": warnings[-1] if warnings else None,
        "warning_throttle_ms": 5000,
        "exact_dropped_images": None,
    }, indent=2))
PYREPORT
```

The tracking overflow warning uses a **5000 ms throttle on the node clock**
(`rosNodeTest.cpp`, `dispatch`). Warning-line counts are **not exact dropped
image counts**; first/last warning lines retain the original log timestamps.
Input coverage must accompany every candidate comparison: expected raw-bag
counts are easy **2280**, medium **2348**, difficult **1922**. Difficult already
has missing images in the original recording; 1922 is its actual bag count,
not a theoretical 20 Hz duration product. Compare `received_images` with those
counts and report `latency_misses` from the unmodified metrics alongside latency
percentiles. Unmatched poses do not contribute to those latency percentiles.
These are observer counts, not proof of estimator delivery or tracker ingestion;
even equal counts alone do not prove all original frame stamps arrived uniquely.

The reported current-head medium run observed **2319/2348 images (29 fewer)**.
The cause is undetermined; absence of a player starvation log does not establish
loss-free playback/delivery. Treat this as a coverage limitation in comparisons,
not as proven tracker loss or a clean full-frame-input baseline. Do not change
the existing benchmark's QoS, counters, or timing to conceal this difference.

Likewise, `received_images - output_poses` is not a tracker-drop counter:
initialization and backend frame decimation also affect it. Fewer warning lines
alone do not prove improved real-time behavior.

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
- The build is limited to `camera_models vins_common vins` with
  `--packages-select`. `vins` lists `loop_fusion`/`global_fusion` only as
  `exec_depend`, and its `CMakeLists.txt` has no build-time reference to
  them, so the mono benchmark does not rebuild those packages.
- The wrapper never edits calibration/config files, CPU governor, or
  flight-controller state. Explicit `--parameter` ROS overrides are recorded.
- Run build, test, and benchmark separately when collecting measurements.
