#!/usr/bin/env bash
# Reproducible SSH workflow for building, testing, and benchmarking VINS on ARM.
# This is intentionally a foreground wrapper: it never leaves a remote daemon.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/benchmark_remote_arm.sh [build|test|benchmark|all] [options]

Defaults:
  step=build, host=pinydrone0@192.168.5.60
  remote repo=Code/Piny_drone_nav (relative to remote $HOME)
  remote bags=vicon_room2/bags, output=vins_remote_eval_20261019
  domain=173, datasets="V2_01_easy V2_02_medium V2_03_difficult"
  label=current_head, rate=1.0, repeats=1, jobs=1, snapshot=current_head_vins_node

The script only invokes vins_node through benchmark_euroc.py. It never starts
flight control, loop_fusion, global_fusion, or RViz. It does not use sudo,
delete user files, or edit calibration/algorithm configuration files.

Options (environment variables with the same names are also accepted):
  --host HOST              SSH destination, default pinydrone0@192.168.5.60
  --repo PATH              Remote workspace (absolute or relative to $HOME)
  --bags PATH              Remote EuRoC bags root
  --output PATH            Remote result/evidence root
  --domain-id N            ROS_DOMAIN_ID, default 173
  --datasets "A B C"       Space-separated bag names
  --label NAME             Benchmark label; must be new
  --rate R                 Bag playback rate, default 1.0
  --repeats N              Benchmark repetitions, default 1
  --duration S             Optional bag-time seconds; omit for full sequences
  --jobs N                 Build jobs, only 1 or 2; default 1
  --snapshot-name NAME     Binary snapshot under <output>/binaries
  --config PATH            Remote config YAML (default: EuRoC mono-IMU)
  --parameter NAME:=VALUE  Extra ROS parameter (repeatable, CLI only)
  --reuse-workspace-build  Reuse <repo>/build and <repo>/install
  --allow-dirty-source     Permit a dirty VINS-Fusion worktree
  --overwrite-snapshot     Replace an existing snapshot (never deployed_vins_node)
  -h, --help               Show this help

Examples:
  scripts/benchmark_remote_arm.sh build --jobs 2 --reuse-workspace-build
  scripts/benchmark_remote_arm.sh test --reuse-workspace-build
  scripts/benchmark_remote_arm.sh benchmark --datasets V2_01_easy --label head_easy

Build and test are separate. Every step takes a non-blocking per-repo flock
and refuses to start if a conflicting compiler/benchmark/VINS process is
running. The benchmark uses --config (default: the EuRoC mono-IMU config).
--parameter values are forwarded unchanged; no config file is edited.
USAGE
}

die() { printf 'error: %s\n' "$*" >&2; exit 2; }

SSH_HOST=${SSH_HOST:-pinydrone0@192.168.5.60}
REMOTE_REPO=${REMOTE_REPO:-Code/Piny_drone_nav}
REMOTE_BAGS=${REMOTE_BAGS:-vicon_room2/bags}
REMOTE_OUTPUT=${REMOTE_OUTPUT:-vins_remote_eval_20261019}
ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-173}
DATASETS=${DATASETS:-"V2_01_easy V2_02_medium V2_03_difficult"}
LABEL=${LABEL:-current_head}
RATE=${RATE:-1.0}
REPEATS=${REPEATS:-1}
DURATION=${DURATION:-}
BUILD_JOBS=${BUILD_JOBS:-1}
SNAPSHOT_NAME=${SNAPSHOT_NAME:-current_head_vins_node}
CONFIG_PATH=${CONFIG:-}
ROS_PARAMETERS=()
REUSE_WORKSPACE_BUILD=${REUSE_WORKSPACE_BUILD:-0}
ALLOW_DIRTY_SOURCE=${ALLOW_DIRTY_SOURCE:-0}
OVERWRITE_SNAPSHOT=${OVERWRITE_SNAPSHOT:-0}

STEP=build
if (( $# > 0 )) && [[ $1 != -* ]]; then
  STEP=$1
  shift
fi

while (( $# > 0 )); do
  case "$1" in
    --host) SSH_HOST=$2; shift 2 ;;
    --repo) REMOTE_REPO=$2; shift 2 ;;
    --bags) REMOTE_BAGS=$2; shift 2 ;;
    --output) REMOTE_OUTPUT=$2; shift 2 ;;
    --domain-id) ROS_DOMAIN_ID=$2; shift 2 ;;
    --datasets) DATASETS=$2; shift 2 ;;
    --label) LABEL=$2; shift 2 ;;
    --rate) RATE=$2; shift 2 ;;
    --repeats) REPEATS=$2; shift 2 ;;
    --duration) DURATION=$2; shift 2 ;;
    --jobs) BUILD_JOBS=$2; shift 2 ;;
    --snapshot-name) SNAPSHOT_NAME=$2; shift 2 ;;
    --config) CONFIG_PATH=$2; shift 2 ;;
    --parameter)
      [[ $# -ge 2 && "$2" == ?*:=* ]] || die "--parameter requires NAME:=VALUE"
      ROS_PARAMETERS+=("$2"); shift 2 ;;
    --reuse-workspace-build) REUSE_WORKSPACE_BUILD=1; shift ;;
    --allow-dirty-source) ALLOW_DIRTY_SOURCE=1; shift ;;
    --overwrite-snapshot) OVERWRITE_SNAPSHOT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$STEP" in build|test|benchmark|all) ;; *) die "unknown step: $STEP" ;; esac
case "$BUILD_JOBS" in 1|2) ;; *) die "BUILD_JOBS must be 1 or 2" ;; esac
[[ "$ROS_DOMAIN_ID" =~ ^[0-9]+$ ]] || die "ROS_DOMAIN_ID must be an integer"
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || die "REPEATS must be positive"
[[ "$RATE" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "RATE must be positive"
[[ -z "$DURATION" || "$DURATION" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "DURATION must be a positive number"
[[ -n "$DATASETS" ]] || die "DATASETS must not be empty"
[[ -n "$LABEL" && "$LABEL" != */* ]] || die "LABEL must be a non-empty path component"
[[ -n "$SNAPSHOT_NAME" && "$SNAPSHOT_NAME" != */* ]] || die "SNAPSHOT_NAME must be a non-empty path component"
[[ "$SNAPSHOT_NAME" != deployed_vins_node ]] || die "deployed_vins_node is reserved for the old deployment"

if [[ "$REUSE_WORKSPACE_BUILD" == 1 ]]; then BUILD_MODE=workspace; else BUILD_MODE=isolated; fi

remote_run() {
  local script=$1
  shift
  local -a quoted=()
  local arg
  for arg in "$@"; do quoted+=("$(printf '%q' "$arg")"); done
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$SSH_HOST" "bash -s -- ${quoted[*]}" <<<"$script"
}

run_build() {
  local script
  script=$(cat <<'REMOTE'
set -eo pipefail
resolve_path() {
  case "$1" in
    /*) printf '%s' "$1" ;;
    "~/"*) printf '%s/%s' "$HOME" "${1#\~/}" ;;
    *) printf '%s/%s' "$HOME" "$1" ;;
  esac
}
REPO=$(resolve_path "$1")
OUTPUT=$(resolve_path "$2")
JOBS=$3
MODE=$4
SNAPSHOT=$5
ALLOW_DIRTY=$6
OVERWRITE=$7
SRC="$REPO/src/VINS-Fusion"
EVID="$OUTPUT/evidence"
mkdir -p "$EVID" "$OUTPUT/binaries"
LOG="$EVID/build_$(date -u +%Y%m%dT%H%M%SZ).log"
STATUS="$EVID/build.status"
exec > >(tee -a "$LOG") 2>&1
on_exit() { rc=$?; printf 'exit=%s\nfinished_utc=%s\nlog=%s\n' "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$LOG" > "$STATUS"; }
trap on_exit EXIT

# One non-blocking lock per remote repo path prevents build/test/benchmark
# steps (and two wrapper invocations) from overlapping.
vins_remote_guard() {
  local lock out rc
  command -v flock >/dev/null 2>&1 || {
    echo "error: flock is required to serialize remote-arm steps" >&2
    exit 3
  }
  lock="${TMPDIR:-/tmp}/vins_remote_arm_$(printf '%s' "$REPO" | sha256sum | cut -c1-16).lock"
  exec 9>"$lock"
  if ! flock -n 9; then
    echo "error: another VINS remote-arm step is using $REPO (lock: $lock)" >&2
    exit 3
  fi
  # Detect real executables/argv tokens, never a fuzzy match against an
  # ancestor shell's command line (which may contain a snapshot argument).
  rc=0
  out=$(/usr/bin/python3 - "$REPO" "$OUTPUT/binaries" <<'PYCHECK'
import os, sys
repo = os.path.realpath(sys.argv[1])
binary_root = os.path.realpath(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else ''
compilers = {'cc1plus', 'cc1', 'gcc', 'g++', 'clang', 'clang++',
             'cmake', 'gmake', 'make', 'ninja'}
found = []
for pid in os.listdir('/proc'):
    if not pid.isdigit() or int(pid) == os.getpid():
        continue
    try:
        raw = open('/proc/%s/cmdline' % pid, 'rb').read()
    except OSError:
        continue
    argv = [part.decode('utf-8', 'replace') for part in raw.split(b'\0') if part]
    if not argv:
        continue
    try:
        exe_path = os.readlink('/proc/%s/exe' % pid)
    except OSError:
        exe_path = ''
    exe = os.path.basename(exe_path).replace(' (deleted)', '')
    names = {exe, os.path.basename(argv[0])}
    try:
        cwd = os.readlink('/proc/%s/cwd' % pid)
    except OSError:
        cwd = ''
    cmd = ' '.join(argv)
    in_repo = repo in cmd or cwd == repo or cwd.startswith(repo + os.sep)
    is_colcon_build = any(os.path.basename(a) == 'colcon' for a in argv) and 'build' in argv
    if in_repo and (names & compilers or is_colcon_build):
        found.append(('compiler/build', pid, cmd))
        continue
    in_bin_root = bool(binary_root) and bool(exe_path) and (
        os.path.realpath(exe_path) == binary_root
        or os.path.realpath(exe_path).startswith(binary_root + os.sep))
    is_vins_name = any(
        n == 'vins_node' or n.endswith('_vins_node') or n.endswith('-vins_node')
        for n in names)
    if is_vins_name or in_bin_root:
        found.append(('vins_node', pid, cmd))
        continue
    if any(os.path.basename(a) == 'benchmark_euroc.py' for a in argv[1:]):
        found.append(('benchmark_euroc', pid, cmd))
        continue
    for i, a in enumerate(argv):
        if os.path.basename(a) == 'ros2':
            rest = argv[i + 1:]
            while rest and rest[0] == '--':
                rest = rest[1:]
            if len(rest) >= 2 and rest[0] == 'bag' and rest[1] == 'play':
                found.append(('ros2 bag play', pid, cmd))
                break
for kind, pid, cmd in found:
    print('%s\t%s\t%s' % (kind, pid, cmd))
sys.exit(0 if found else 1)
PYCHECK
  ) || rc=$?
  if (( rc == 0 )); then
    printf 'error: conflicting VINS/build process detected; refusing to overlap:\n%s\n' "$out" >&2
    exit 3
  elif (( rc != 1 )); then
    echo "error: process conflict scan failed (rc=$rc)" >&2
    exit 3
  fi
}
vins_remote_guard

SNAP="$OUTPUT/binaries/$SNAPSHOT"
if [[ -e "$SNAP" && "$OVERWRITE" != 1 ]]; then
  echo "error: snapshot already exists; choose --snapshot-name or --overwrite-snapshot: $SNAP" >&2
  exit 6
fi

command -v git >/dev/null || { echo "error: git not found" >&2; exit 3; }
git -C "$SRC" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "error: VINS-Fusion git worktree not found: $SRC" >&2
  exit 3
}
[[ -f "$SRC/scripts/benchmark_euroc.py" ]] || { echo "error: benchmark script missing" >&2; exit 3; }
[[ -f "$SRC/src/vins_estimator/config/euroc/euroc_mono_imu_config.yaml" ]] || { echo "error: EuRoC mono IMU config missing" >&2; exit 3; }

HEAD=$(git -C "$SRC" rev-parse HEAD)
DIRTY=$(git -C "$SRC" status --porcelain)
if [[ -n "$DIRTY" && "$ALLOW_DIRTY" != 1 ]]; then
  printf 'error: VINS-Fusion worktree is dirty; commit/stash it or use --allow-dirty-source\n%s\n' "$DIRTY" >&2
  exit 4
fi

source /opt/ros/jazzy/setup.bash
command -v colcon >/dev/null || { echo "error: colcon not found" >&2; exit 3; }
command -v cmake >/dev/null || { echo "error: cmake not found" >&2; exit 3; }

if [[ "$MODE" == workspace ]]; then
  BUILD_BASE="$REPO/build"
  INSTALL_BASE="$REPO/install"
  MAP_ARGS=()
else
  BUILD_BASE="$OUTPUT/build"
  INSTALL_BASE="$OUTPUT/install"
  MAP_ARGS=(--build-base "$BUILD_BASE" --install-base "$INSTALL_BASE")
fi

CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_TESTING=ON
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF
  -DVINS_ENABLE_NATIVE_OPTIMIZATION=OFF
)
# Build only the packages needed by the mono estimator binary and its tests.
# loop_fusion/global_fusion are exec_depend-only for "vins"; its CMakeLists.txt
# has no build-time reference to either, so they are deliberately excluded.
BUILD_CMD=(colcon build --base-paths "$SRC/src"
  --packages-select camera_models vins_common vins
  --executor sequential "${MAP_ARGS[@]}" --event-handlers console_direct+
  --cmake-args "${CMAKE_ARGS[@]}")
printf 'head=%s\nmode=%s\njobs=%s\ncommand=' "$HEAD" "$MODE" "$JOBS"
printf '%q ' "${BUILD_CMD[@]}"
printf '\n'
printf '%q ' "${BUILD_CMD[@]}" > "$EVID/build_command.txt"

cd "$REPO"
MAKEFLAGS="-j$JOBS" CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" "${BUILD_CMD[@]}"

BIN="$INSTALL_BASE/vins/lib/vins/vins_node"
[[ -e "$BIN" ]] || { echo "error: installed vins_node not found: $BIN" >&2; exit 5; }
cp -Lf "$BIN" "$OUTPUT/binaries/$SNAPSHOT"
FILE_INFO=$(file -b "$OUTPUT/binaries/$SNAPSHOT")
[[ "$FILE_INFO" == *aarch64* ]] || { echo "error: snapshot is not an aarch64 binary: $FILE_INFO" >&2; exit 7; }
SHA=$(sha256sum "$OUTPUT/binaries/$SNAPSHOT" | awk '{print $1}')
{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo=$REPO"
  echo "head=$HEAD"
  echo "source_dirty=$([[ -n "$DIRTY" ]] && echo yes || echo no)"
  echo "build_mode=$MODE"
  echo "jobs=$JOBS"
  echo "build_base=$BUILD_BASE"
  echo "install_base=$INSTALL_BASE"
  echo "binary=$OUTPUT/binaries/$SNAPSHOT"
  echo "binary_sha256=$SHA"
  echo "binary_file=$FILE_INFO"
  echo "command=$(cat "$EVID/build_command.txt")"
} | tee "$EVID/build_provenance.txt"
echo "snapshot=$OUTPUT/binaries/$SNAPSHOT"
echo "sha256=$SHA"
REMOTE
)
  remote_run "$script" "$REMOTE_REPO" "$REMOTE_OUTPUT" "$BUILD_JOBS" "$BUILD_MODE" "$SNAPSHOT_NAME" "$ALLOW_DIRTY_SOURCE" "$OVERWRITE_SNAPSHOT"
}

run_test() {
  local script
  script=$(cat <<'REMOTE'
set -eo pipefail
resolve_path() {
  case "$1" in
    /*) printf '%s' "$1" ;;
    "~/"*) printf '%s/%s' "$HOME" "${1#\~/}" ;;
    *) printf '%s/%s' "$HOME" "$1" ;;
  esac
}
REPO=$(resolve_path "$1")
OUTPUT=$(resolve_path "$2")
MODE=$3
DOMAIN=$4
SRC="$REPO/src/VINS-Fusion"
EVID="$OUTPUT/evidence"
mkdir -p "$EVID"
LOG="$EVID/test_$(date -u +%Y%m%dT%H%M%SZ).log"
STATUS="$EVID/test.status"
exec > >(tee -a "$LOG") 2>&1
on_exit() { rc=$?; printf 'exit=%s\nfinished_utc=%s\nlog=%s\n' "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$LOG" > "$STATUS"; }
trap on_exit EXIT

# One non-blocking lock per remote repo path prevents build/test/benchmark
# steps (and two wrapper invocations) from overlapping.
vins_remote_guard() {
  local lock out rc
  command -v flock >/dev/null 2>&1 || {
    echo "error: flock is required to serialize remote-arm steps" >&2
    exit 3
  }
  lock="${TMPDIR:-/tmp}/vins_remote_arm_$(printf '%s' "$REPO" | sha256sum | cut -c1-16).lock"
  exec 9>"$lock"
  if ! flock -n 9; then
    echo "error: another VINS remote-arm step is using $REPO (lock: $lock)" >&2
    exit 3
  fi
  # Detect real executables/argv tokens, never a fuzzy match against an
  # ancestor shell's command line (which may contain a snapshot argument).
  rc=0
  out=$(/usr/bin/python3 - "$REPO" "$OUTPUT/binaries" <<'PYCHECK'
import os, sys
repo = os.path.realpath(sys.argv[1])
binary_root = os.path.realpath(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else ''
compilers = {'cc1plus', 'cc1', 'gcc', 'g++', 'clang', 'clang++',
             'cmake', 'gmake', 'make', 'ninja'}
found = []
for pid in os.listdir('/proc'):
    if not pid.isdigit() or int(pid) == os.getpid():
        continue
    try:
        raw = open('/proc/%s/cmdline' % pid, 'rb').read()
    except OSError:
        continue
    argv = [part.decode('utf-8', 'replace') for part in raw.split(b'\0') if part]
    if not argv:
        continue
    try:
        exe_path = os.readlink('/proc/%s/exe' % pid)
    except OSError:
        exe_path = ''
    exe = os.path.basename(exe_path).replace(' (deleted)', '')
    names = {exe, os.path.basename(argv[0])}
    try:
        cwd = os.readlink('/proc/%s/cwd' % pid)
    except OSError:
        cwd = ''
    cmd = ' '.join(argv)
    in_repo = repo in cmd or cwd == repo or cwd.startswith(repo + os.sep)
    is_colcon_build = any(os.path.basename(a) == 'colcon' for a in argv) and 'build' in argv
    if in_repo and (names & compilers or is_colcon_build):
        found.append(('compiler/build', pid, cmd))
        continue
    in_bin_root = bool(binary_root) and bool(exe_path) and (
        os.path.realpath(exe_path) == binary_root
        or os.path.realpath(exe_path).startswith(binary_root + os.sep))
    is_vins_name = any(
        n == 'vins_node' or n.endswith('_vins_node') or n.endswith('-vins_node')
        for n in names)
    if is_vins_name or in_bin_root:
        found.append(('vins_node', pid, cmd))
        continue
    if any(os.path.basename(a) == 'benchmark_euroc.py' for a in argv[1:]):
        found.append(('benchmark_euroc', pid, cmd))
        continue
    for i, a in enumerate(argv):
        if os.path.basename(a) == 'ros2':
            rest = argv[i + 1:]
            while rest and rest[0] == '--':
                rest = rest[1:]
            if len(rest) >= 2 and rest[0] == 'bag' and rest[1] == 'play':
                found.append(('ros2 bag play', pid, cmd))
                break
for kind, pid, cmd in found:
    print('%s\t%s\t%s' % (kind, pid, cmd))
sys.exit(0 if found else 1)
PYCHECK
  ) || rc=$?
  if (( rc == 0 )); then
    printf 'error: conflicting VINS/build process detected; refusing to overlap:\n%s\n' "$out" >&2
    exit 3
  elif (( rc != 1 )); then
    echo "error: process conflict scan failed (rc=$rc)" >&2
    exit 3
  fi
}
vins_remote_guard

if [[ "$MODE" == workspace ]]; then
  BUILD_BASE="$REPO/build"
  INSTALL_BASE="$REPO/install"
  MAP_ARGS=()
else
  BUILD_BASE="$OUTPUT/build"
  INSTALL_BASE="$OUTPUT/install"
  MAP_ARGS=(--build-base "$BUILD_BASE" --install-base "$INSTALL_BASE")
fi
[[ -f "$INSTALL_BASE/setup.bash" ]] || { echo "error: install setup missing: $INSTALL_BASE/setup.bash" >&2; exit 4; }

source /opt/ros/jazzy/setup.bash
source "$INSTALL_BASE/setup.bash"
export ROS_DOMAIN_ID="$DOMAIN"
HEAD=$(git -C "$SRC" rev-parse HEAD)
TEST_CMD=(colcon test --base-paths "$SRC/src" --packages-select vins_common vins
  --executor sequential --parallel-workers 1 "${MAP_ARGS[@]}"
  --event-handlers console_direct+ --return-code-on-test-failure)
printf 'head=%s\nmode=%s\ndomain=%s\ncommand=' "$HEAD" "$MODE" "$DOMAIN"
printf '%q ' "${TEST_CMD[@]}"
printf '\n'
printf '%q ' "${TEST_CMD[@]}" > "$EVID/test_command.txt"

cd "$REPO"
"${TEST_CMD[@]}"
colcon test-result --test-result-base "$BUILD_BASE" --verbose

cd "$SRC"
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m unittest discover -s scripts/tests -v
echo "tests_completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
REMOTE
)
  remote_run "$script" "$REMOTE_REPO" "$REMOTE_OUTPUT" "$BUILD_MODE" "$ROS_DOMAIN_ID"
}

run_benchmark() {
  local script
  script=$(cat <<'REMOTE'
set -eo pipefail
resolve_path() {
  case "$1" in
    /*) printf '%s' "$1" ;;
    "~/"*) printf '%s/%s' "$HOME" "${1#\~/}" ;;
    *) printf '%s/%s' "$HOME" "$1" ;;
  esac
}
REPO=$(resolve_path "$1")
OUTPUT=$(resolve_path "$2")
BAGS=$(resolve_path "$3")
MODE=$4
DOMAIN=$5
DATASETS=$6
LABEL=$7
RATE=$8
REPEATS=$9
DURATION=${10}
SNAPSHOT=${11}
CONFIG_ARG=${12}
shift 12
ROS_PARAMETERS=("$@")
SRC="$REPO/src/VINS-Fusion"
EVID="$OUTPUT/evidence"
mkdir -p "$EVID"
LOG="$EVID/benchmark_$(date -u +%Y%m%dT%H%M%SZ).log"
STATUS="$EVID/benchmark.status"
exec > >(tee -a "$LOG") 2>&1
on_exit() { rc=$?; printf 'exit=%s\nfinished_utc=%s\nlog=%s\n' "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$LOG" > "$STATUS"; }
trap on_exit EXIT

# One non-blocking lock per remote repo path prevents build/test/benchmark
# steps (and two wrapper invocations) from overlapping.
vins_remote_guard() {
  local lock out rc
  command -v flock >/dev/null 2>&1 || {
    echo "error: flock is required to serialize remote-arm steps" >&2
    exit 3
  }
  lock="${TMPDIR:-/tmp}/vins_remote_arm_$(printf '%s' "$REPO" | sha256sum | cut -c1-16).lock"
  exec 9>"$lock"
  if ! flock -n 9; then
    echo "error: another VINS remote-arm step is using $REPO (lock: $lock)" >&2
    exit 3
  fi
  # Detect real executables/argv tokens, never a fuzzy match against an
  # ancestor shell's command line (which may contain a snapshot argument).
  rc=0
  out=$(/usr/bin/python3 - "$REPO" "$OUTPUT/binaries" <<'PYCHECK'
import os, sys
repo = os.path.realpath(sys.argv[1])
binary_root = os.path.realpath(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else ''
compilers = {'cc1plus', 'cc1', 'gcc', 'g++', 'clang', 'clang++',
             'cmake', 'gmake', 'make', 'ninja'}
found = []
for pid in os.listdir('/proc'):
    if not pid.isdigit() or int(pid) == os.getpid():
        continue
    try:
        raw = open('/proc/%s/cmdline' % pid, 'rb').read()
    except OSError:
        continue
    argv = [part.decode('utf-8', 'replace') for part in raw.split(b'\0') if part]
    if not argv:
        continue
    try:
        exe_path = os.readlink('/proc/%s/exe' % pid)
    except OSError:
        exe_path = ''
    exe = os.path.basename(exe_path).replace(' (deleted)', '')
    names = {exe, os.path.basename(argv[0])}
    try:
        cwd = os.readlink('/proc/%s/cwd' % pid)
    except OSError:
        cwd = ''
    cmd = ' '.join(argv)
    in_repo = repo in cmd or cwd == repo or cwd.startswith(repo + os.sep)
    is_colcon_build = any(os.path.basename(a) == 'colcon' for a in argv) and 'build' in argv
    if in_repo and (names & compilers or is_colcon_build):
        found.append(('compiler/build', pid, cmd))
        continue
    in_bin_root = bool(binary_root) and bool(exe_path) and (
        os.path.realpath(exe_path) == binary_root
        or os.path.realpath(exe_path).startswith(binary_root + os.sep))
    is_vins_name = any(
        n == 'vins_node' or n.endswith('_vins_node') or n.endswith('-vins_node')
        for n in names)
    if is_vins_name or in_bin_root:
        found.append(('vins_node', pid, cmd))
        continue
    if any(os.path.basename(a) == 'benchmark_euroc.py' for a in argv[1:]):
        found.append(('benchmark_euroc', pid, cmd))
        continue
    for i, a in enumerate(argv):
        if os.path.basename(a) == 'ros2':
            rest = argv[i + 1:]
            while rest and rest[0] == '--':
                rest = rest[1:]
            if len(rest) >= 2 and rest[0] == 'bag' and rest[1] == 'play':
                found.append(('ros2 bag play', pid, cmd))
                break
for kind, pid, cmd in found:
    print('%s\t%s\t%s' % (kind, pid, cmd))
sys.exit(0 if found else 1)
PYCHECK
  ) || rc=$?
  if (( rc == 0 )); then
    printf 'error: conflicting VINS/build process detected; refusing to overlap:\n%s\n' "$out" >&2
    exit 3
  elif (( rc != 1 )); then
    echo "error: process conflict scan failed (rc=$rc)" >&2
    exit 3
  fi
}
vins_remote_guard

if [[ "$MODE" == workspace ]]; then
  INSTALL_BASE="$REPO/install"
else
  INSTALL_BASE="$OUTPUT/install"
fi
BIN="$OUTPUT/binaries/$SNAPSHOT"
if [[ -z "$CONFIG_ARG" ]]; then
  CONFIG="$SRC/src/vins_estimator/config/euroc/euroc_mono_imu_config.yaml"
else
  CONFIG=$(resolve_path "$CONFIG_ARG")
fi
TARGET="$OUTPUT/benchmark/$LABEL"
git -C "$SRC" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "error: VINS-Fusion git worktree not found: $SRC" >&2
  exit 3
}
[[ -f "$BIN" ]] || { echo "error: binary snapshot missing: $BIN" >&2; exit 5; }
[[ -f "$CONFIG" ]] || { echo "error: config missing: $CONFIG" >&2; exit 6; }
[[ -f "$SRC/scripts/benchmark_euroc.py" ]] || { echo "error: benchmark script missing" >&2; exit 7; }
[[ ! -e "$TARGET" ]] || { echo "error: label already exists, choose a new label: $TARGET" >&2; exit 8; }

read -r -a DATASET_ARRAY <<< "$DATASETS"
(( ${#DATASET_ARRAY[@]} > 0 )) || { echo "error: no datasets supplied" >&2; exit 9; }
for dataset in "${DATASET_ARRAY[@]}"; do
  [[ -d "$BAGS/$dataset" ]] || { echo "error: bag directory missing: $BAGS/$dataset" >&2; exit 10; }
done

FILE_INFO=$(file -b "$BIN")
[[ "$FILE_INFO" == *aarch64* ]] || { echo "error: snapshot is not aarch64: $FILE_INFO" >&2; exit 11; }
SHA=$(sha256sum "$BIN" | awk '{print $1}')
CONFIG_SHA=$(sha256sum "$CONFIG" | awk '{print $1}')
BENCH_CMD=(/usr/bin/python3 "$SRC/scripts/benchmark_euroc.py"
  --bags-root "$BAGS" --config "$CONFIG" --binary "$BIN"
  --output "$OUTPUT/benchmark" --label "$LABEL"
  --datasets "${DATASET_ARRAY[@]}" --rate "$RATE" --repeats "$REPEATS"
  --domain-id "$DOMAIN")
if [[ -n "$DURATION" ]]; then
  BENCH_CMD+=(--duration "$DURATION")
fi
for parameter in "${ROS_PARAMETERS[@]}"; do
  BENCH_CMD+=(--parameter "$parameter")
done
{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "head=$(git -C "$SRC" rev-parse HEAD)"
  echo "binary=$BIN"
  echo "binary_sha256=$SHA"
  echo "binary_file=$FILE_INFO"
  echo "config=$CONFIG"
  echo "config_sha256=$CONFIG_SHA"
  echo "domain=$DOMAIN"
  echo "datasets=$DATASETS"
  echo "label=$LABEL"
  echo "rate=$RATE"
  echo "repeats=$REPEATS"
  echo "duration=${DURATION:-full}"
  printf 'command='; printf '%q ' "${BENCH_CMD[@]}"; printf '\n'
} | tee "$EVID/benchmark_provenance.txt"

source /opt/ros/jazzy/setup.bash
command -v ros2 >/dev/null || { echo "error: ros2 not found" >&2; exit 12; }
if [[ -f "$INSTALL_BASE/setup.bash" ]]; then source "$INSTALL_BASE/setup.bash"; fi
/usr/bin/python3 - <<'PY'
import numpy
import psutil
import rclpy
import rosbag2_py
PY
export ROS_DOMAIN_ID="$DOMAIN"

cd "$SRC"
PYTHONDONTWRITEBYTECODE=1 "${BENCH_CMD[@]}"
echo "benchmark_completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
REMOTE
)
  remote_run "$script" "$REMOTE_REPO" "$REMOTE_OUTPUT" "$REMOTE_BAGS" "$BUILD_MODE" "$ROS_DOMAIN_ID" "$DATASETS" "$LABEL" "$RATE" "$REPEATS" "$DURATION" "$SNAPSHOT_NAME" "$CONFIG_PATH" "${ROS_PARAMETERS[@]}"
}

case "$STEP" in
  build) run_build ;;
  test) run_test ;;
  benchmark) run_benchmark ;;
  all) run_build; run_test; run_benchmark ;;
esac
