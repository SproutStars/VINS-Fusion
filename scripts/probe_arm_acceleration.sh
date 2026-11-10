#!/usr/bin/env bash
# Read-only ARM acceleration probe for VINS-Fusion.
# No installs, governor/permission/sysfs writes, or ROS replay.
# Use --host for SSH (key auth, BatchMode=yes; no password is stored).
set -uo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/probe_arm_acceleration.sh [--host HOST]
Read-only probe: CPU topology/frequency/capacity, thermal, memory, OpenCV
NEON/OpenCL, DRM/GPU/NPU nodes, and installed runtimes/ICDs.
No package install, governor or permission change, sysfs write, or ROS run.
USAGE
}
die() { printf 'error: %s\n' "$*" >&2; exit 2; }
val() {
  local f=$1
  if [[ -r "$f" ]]; then tr '\n' ' ' < "$f" | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//'; else printf unavailable; fi
}
kv() { printf '%s=%s\n' "$1" "$2"; }

HOST=${PROBE_HOST:-local}
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  --host)
    [[ $# -eq 2 ]] || die "--host requires exactly one value"
    HOST=$2
    [[ "$HOST" =~ ^[A-Za-z0-9_.@:-]+$ ]] || die "invalid host"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" "PROBE_HOST='$HOST' bash -s" < "$0"
    exit $?
    ;;
  "") ;;
  *) die "unknown argument: $1" ;;
esac

echo "host=$HOST"
echo "== system =="
uname -a 2>/dev/null || echo "uname=unavailable"
[[ -r /etc/os-release ]] && grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release || echo "os_release=unavailable"
echo
echo "== memory =="
free -b 2>/dev/null || echo "free=unavailable"
grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo 2>/dev/null || true
echo
echo "== cpu topology =="
lscpu 2>/dev/null || echo "lscpu=unavailable"
kv cpu_online "$(val /sys/devices/system/cpu/online)"
kv cpu_present "$(val /sys/devices/system/cpu/present)"
kv cpu_count_online "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo unavailable)"
echo
echo "== cpu capacity =="
found=0
for f in /sys/devices/system/cpu/cpu[0-9]*/cpu_capacity; do
  [[ -r "$f" ]] || continue
  found=1
  kv "$(basename "$(dirname "$f")")_capacity" "$(cat "$f")"
done
(( found )) || echo "cpu_capacity=unavailable"
echo
echo "== cpufreq =="
found=0
for p in /sys/devices/system/cpu/cpufreq/policy*; do
  [[ -d "$p" ]] || continue
  found=1
  kv "policy_$(basename "$p" | sed 's/^policy//')" "cpus=$(val "$p/affected_cpus") governor=$(val "$p/scaling_governor") cur_khz=$(val "$p/scaling_cur_freq") min_khz=$(val "$p/cpuinfo_min_freq") max_khz=$(val "$p/cpuinfo_max_freq") available=$(val "$p/scaling_available_frequencies")"
done
(( found )) || echo "cpufreq=unavailable"
echo
echo "== thermal =="
found=0
for z in /sys/class/thermal/thermal_zone*; do
  [[ -d "$z" ]] || continue
  found=1
  temp=$(val "$z/temp")
  temp_c=unavailable
  [[ "$temp" =~ ^-?[0-9]+$ ]] && temp_c=$(awk -v t="$temp" 'BEGIN{printf "%.1f", t/1000}')
  kv "$(basename "$z")" "type=$(val "$z/type") temp_mc=$temp temp_c=$temp_c"
done
(( found )) || echo "thermal=unavailable"
echo
echo "== opencv =="
kv pkgconfig_opencv4 "$(pkg-config --modversion opencv4 2>/dev/null || echo unavailable)"
opencv_version --verbose 2>/dev/null | grep -E 'Baseline|Parallel framework|OpenCL|Version control' || echo "opencv_version=unavailable"
python3 - <<'PY' 2>/dev/null || echo "python_cv2=unavailable"
try:
    import cv2
except Exception as exc:
    print("python_cv2=unavailable:", exc)
    raise SystemExit(0)
print("python_cv2_version=", cv2.__version__)
for line in cv2.getBuildInformation().splitlines():
    text = line.strip()
    if text.startswith(("Baseline:", "Parallel framework:", "OpenCL:")):
        print("build_" + text.replace(":", "=", 1))
print("haveOpenCL=", cv2.ocl.haveOpenCL())
print("useOpenCL=", cv2.ocl.useOpenCL())
try:
    dev = cv2.ocl.Device.getDefault()
    print("opencl_device=", dev.name())
    print("opencl_vendor=", dev.vendorName())
    print("opencl_version=", dev.version())
    print("opencl_driver=", dev.driverVersion())
except Exception as exc:
    print("opencl_device=unavailable:", exc)
PY
echo
echo "== gpu / dri =="
ls -l /dev/dri 2>/dev/null || echo "dri_nodes=unavailable"
for n in /dev/mali* /dev/npu* /dev/rknpu* /dev/accel* /dev/vipcore; do
  [[ -e "$n" ]] && ls -l "$n"
done
for c in /sys/class/drm/card*; do
  [[ -d "$c" ]] || continue
  name=$(basename "$c")
  [[ "$name" =~ ^card[0-9]+$ ]] || continue
  kv "$name" "driver=$(awk -F= '$1=="DRIVER"{print $2}' "$c/device/uevent" 2>/dev/null || echo unavailable) compatible=$(awk -F= '$1=="OF_COMPATIBLE_0"{print $2}' "$c/device/uevent" 2>/dev/null || echo unavailable)"
done
for d in /sys/class/devfreq/*; do
  [[ -d "$d" ]] || continue
  n=$(val "$d/name")
  [[ "$n" == *gpu* || "$(basename "$d")" == *gpu* ]] || continue
  kv "gpu_devfreq" "name=$n cur_hz=$(val "$d/cur_freq") max_hz=$(val "$d/max_freq") governor=$(val "$d/governor")"
done
echo
echo "== npu =="
for d in /sys/class/devfreq/*; do
  [[ -d "$d" ]] || continue
  n=$(val "$d/name")
  [[ "$n" == *npu* || "$(basename "$d")" == *npu* ]] || continue
  kv "npu_devfreq" "name=$n cur_hz=$(val "$d/cur_freq") max_hz=$(val "$d/max_freq") governor=$(val "$d/governor")"
done
kv npu_vipcore_version "$(val /sys/module/vipcore/version)"
ls -l /usr/lib/sun60iw2-npu 2>/dev/null || echo "npu_vendor_libs=unavailable"
echo
echo "== runtimes / icds =="
if command -v clinfo >/dev/null 2>&1; then clinfo -l 2>&1 || echo "clinfo=failed"; else echo "clinfo=unavailable"; fi
if command -v vulkaninfo >/dev/null 2>&1; then vulkaninfo --summary 2>/dev/null | sed -n '/^Devices:/,/^$/p' || echo "vulkaninfo=failed"; else echo "vulkaninfo=unavailable"; fi
echo "-- opencl icd --"
ls -l /etc/OpenCL/vendors 2>/dev/null && cat /etc/OpenCL/vendors/* 2>/dev/null || echo "opencl_icd=unavailable"
echo "-- vulkan icd --"
ls -l /usr/share/vulkan/icd.d 2>/dev/null && cat /usr/share/vulkan/icd.d/*.json 2>/dev/null || echo "vulkan_icd=unavailable"
echo "-- ldconfig matches --"
ldconfig -p 2>/dev/null | grep -Ei 'OpenCL|vulkan|PVR|Mali|EGL|GLES' || echo "ldconfig_matches=unavailable"
echo
echo "== device tree =="
if [[ -r /proc/device-tree/model ]]; then tr -d '\0' < /proc/device-tree/model; echo; else echo "model=unavailable"; fi
if [[ -r /proc/device-tree/compatible ]]; then tr '\0' '\n' < /proc/device-tree/compatible; else echo "compatible=unavailable"; fi
