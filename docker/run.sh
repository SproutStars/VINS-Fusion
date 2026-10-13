#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == '-h' || ${1:-} == '--help' ]]; then
    echo 'Usage: docker/run.sh [launch_file] [name:=value ...]'
    exit 0
fi
launch_file=${1:-vins_mono_imu_launch.py}
if [[ $# -gt 0 ]]; then shift; fi
exec docker run --rm -it --network host \
    -v vins-fusion-output:/output \
    vins-fusion:jazzy ros2 launch vins "$launch_file" output_path:=/output "$@"
