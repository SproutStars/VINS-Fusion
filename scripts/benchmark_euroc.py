#!/usr/bin/env python3
"""Replay EuRoC ROS 2 bags and measure the estimator, not RViz or the player.

Source /opt/ros/jazzy/setup.bash and install/setup.bash before running.
ATE uses sensor header timestamps, interpolated Vicon positions, and rigid SE(3)
position alignment (no fitted scale or time offset). Arrival latency is observed
at ROS subscriptions, not inferred from bag /clock (record/header times differ).
"""
import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import time

import numpy as np
import psutil


class ArrivalTimes:
    """Match integer camera stamps to round-tripped double estimator stamps.

    A double at EuRoC's epoch has ~238 ns spacing. Rounding both values to a
    microsecond can land on opposite sides of a bucket boundary. Use nearest
    matching within 1 us instead, far below the interval between camera frames.
    """
    def __init__(self, capacity=1000):
        if capacity < 1:
            raise ValueError("Arrival history capacity must be positive")
        self.capacity = capacity
        self.arrivals = OrderedDict()

    def record(self, stamp_ns, arrival_ns):
        self.arrivals[stamp_ns] = arrival_ns
        while len(self.arrivals) > self.capacity:
            self.arrivals.popitem(last=False)

    def latency_ms(self, stamp_ns, now_ns):
        if not self.arrivals:
            return None
        closest = min(self.arrivals, key=lambda candidate: abs(candidate - stamp_ns))
        if abs(closest - stamp_ns) > 1000 or now_ns < self.arrivals[closest]:
            return None
        return (now_ns - self.arrivals[closest]) / 1e6


def rigid_align(estimated, reference):
    """Return rigidly aligned Nx3 positions without changing metric scale."""
    source_mean, target_mean = estimated.mean(axis=0), reference.mean(axis=0)
    u, _, vt = np.linalg.svd((estimated - source_mean).T @ (reference - target_mean))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    return (estimated - source_mean) @ rotation + target_mean


def trajectory_metrics(estimated, truth):
    if len(estimated) < 3 or len(truth) < 3:
        return {"matched_poses": 0, "ate_rmse_m": None}
    estimated = np.asarray(estimated, dtype=float)
    truth = np.asarray(truth, dtype=float)
    truth = truth[np.argsort(truth[:, 0])]
    truth = truth[np.r_[True, np.diff(truth[:, 0]) > 0]]
    valid = (estimated[:, 0] >= truth[0, 0]) & (estimated[:, 0] <= truth[-1, 0])
    estimated = estimated[valid]
    if len(estimated) < 3:
        return {"matched_poses": len(estimated), "ate_rmse_m": None}
    t = estimated[:, 0]
    right = np.searchsorted(truth[:, 0], t, side="right").clip(1, len(truth) - 1)
    # Do not interpolate across Vicon outages (> 100 ms).
    valid = truth[right, 0] - truth[right - 1, 0] <= 0.1
    estimated, t = estimated[valid], t[valid]
    if len(estimated) < 3:
        return {"matched_poses": len(estimated), "ate_rmse_m": None}
    reference = np.column_stack([np.interp(t, truth[:, 0], truth[:, i]) for i in range(1, 4)])
    error = np.linalg.norm(rigid_align(estimated[:, 1:4], reference) - reference, axis=1)
    return {
        "matched_poses": len(error), "ate_rmse_m": float(np.sqrt(np.mean(error**2))),
        "ate_median_m": float(np.median(error)), "ate_p95_m": float(np.percentile(error, 95)),
        "trajectory_duration_s": float(t[-1] - t[0]),
        "max_output_gap_s": float(np.max(np.diff(t))),
    }


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def extract_truth(bag, destination):
    import rosbag2_py
    from geometry_msgs.msg import TransformStamped
    from rclpy.serialization import deserialize_message
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    topics = reader.get_all_topics_and_types()
    candidates = [t.name for t in topics if "vicon" in t.name and t.type == "geometry_msgs/msg/TransformStamped"]
    if len(candidates) != 1:
        raise ValueError(f"Expected one Vicon TransformStamped topic in {bag}, found {candidates}")
    reader.set_filter(rosbag2_py.StorageFilter(topics=candidates))
    rows = []
    while reader.has_next():
        _, data, _ = reader.read_next()
        msg = deserialize_message(data, TransformStamped)
        p, q = msg.transform.translation, msg.transform.rotation
        rows.append([msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                     p.x, p.y, p.z, q.x, q.y, q.z, q.w])
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(destination, rows, delimiter=",", header="t,x,y,z,qx,qy,qz,qw", comments="")
    return rows


def run_bag(args, bag, output, truth):
    import rclpy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    output.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    observer = rclpy.create_node("vins_benchmark_observer")
    arrivals = ArrivalTimes()
    trajectory, latencies = [], []
    counters = {"received_images": 0, "latency_misses": 0}
    last_output_wall = [time.monotonic()]

    def image(data):
        # CDR encapsulation followed by Header.stamp. No image-array decoding.
        endian = "<" if data[1] & 1 else ">"
        sec, nsec = struct.unpack_from(endian + "iI", data, 4)
        arrivals.record(sec * 10**9 + nsec, time.monotonic_ns())
        counters["received_images"] += 1

    def odometry(msg):
        now = time.monotonic_ns()
        sec, nsec = msg.header.stamp.sec, msg.header.stamp.nanosec
        stamp = sec * 10**9 + nsec
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        trajectory.append([sec + nsec * 1e-9, p.x, p.y, p.z, q.x, q.y, q.z, q.w])
        last_output_wall[0] = time.monotonic()
        latency = arrivals.latency_ms(stamp, now)
        if latency is not None:
            latencies.append(latency)
        else:
            counters["latency_misses"] += 1

    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100,
                     reliability=ReliabilityPolicy.BEST_EFFORT)
    subscriptions = [observer.create_subscription(Image, "/cam0/image_raw", image, qos, raw=True),
                     observer.create_subscription(Odometry, "/vins_estimator/odometry", odometry, 100)]
    estimator = player = None
    resource_rows = []
    env = os.environ.copy()
    with open(output / "estimator.log", "w") as estimator_log, open(output / "player.log", "w") as player_log:
        try:
            command = [str(args.binary), "--ros-args", "-r", "__node:=vins_estimator",
                       "-p", f"config_file:={args.config}", "-p", f"output_path:={output}",
                       "-p", "use_sim_time:=true"]
            if args.parameter:
                for parameter in args.parameter:
                    command += ["-p", parameter]
            estimator = subprocess.Popen(command, stdout=estimator_log, stderr=subprocess.STDOUT,
                                         env=env, start_new_session=True)
            process = psutil.Process(estimator.pid)
            deadline = time.monotonic() + 20
            while observer.count_publishers("/vins_estimator/odometry") == 0:
                rclpy.spin_once(observer, timeout_sec=0.05)
                if estimator.poll() is not None:
                    raise RuntimeError(f"Estimator exited: {output / 'estimator.log'}")
                if time.monotonic() > deadline:
                    raise TimeoutError("Estimator publisher discovery timed out")
            time.sleep(1.0)
            initial_cpu = process.cpu_times()
            start = time.monotonic()
            command = ["ros2", "bag", "play", str(bag), "--rate", str(args.rate),
                       "--clock", "100", "--disable-keyboard-controls", "--delay", "1",
                       "--topics", "/imu0", "/cam0/image_raw"]
            if args.duration:
                command += ["--playback-duration", str(args.duration)]
            player = subprocess.Popen(command, stdout=player_log, stderr=subprocess.STDOUT,
                                      env=env, start_new_session=True)
            player_end = None
            sample_at = 0.0
            while True:
                rclpy.spin_once(observer, timeout_sec=0.01)
                now = time.monotonic()
                if estimator.poll() is not None:
                    raise RuntimeError(f"Estimator exited during replay: {output / 'estimator.log'}")
                if now >= sample_at:
                    cpu = process.cpu_times()
                    resource_rows.append([now - start, cpu.user - initial_cpu.user,
                                          cpu.system - initial_cpu.system,
                                          process.memory_info().rss / 2**20, process.num_threads()])
                    sample_at = now + 0.1
                if player.poll() is not None:
                    if player.returncode != 0:
                        raise RuntimeError(f"Bag player exited {player.returncode}: {output / 'player.log'}")
                    if player_end is None:
                        player_end = now
                    if now - max(player_end, last_output_wall[0]) >= 3.0:
                        break
                    if now - player_end > args.drain_timeout:
                        raise TimeoutError("Estimator did not drain its output queue")
                if now - start > args.timeout:
                    raise TimeoutError("Replay timeout")
            final_cpu = process.cpu_times()
            elapsed = time.monotonic() - start
            # Include cooperative shutdown time in logs but not CPU/latency metrics.
        finally:
            stop_process(player)
            stop_process(estimator)
            observer.destroy_node()
            rclpy.try_shutdown()
    np.savetxt(output / "trajectory.csv", trajectory, delimiter=",",
               header="t,x,y,z,qx,qy,qz,qw", comments="")
    np.savetxt(output / "resources.csv", resource_rows, delimiter=",",
               header="wall_s,user_cpu_s,system_cpu_s,rss_mib,threads", comments="")
    np.savetxt(output / "latencies_ms.csv", latencies, delimiter=",")
    cpu_seconds = final_cpu.user + final_cpu.system - initial_cpu.user - initial_cpu.system
    log = (output / "estimator.log").read_text(errors="replace")
    result = {
        "dataset": bag.name, "label": args.label, "rate": args.rate,
        "wall_s": elapsed, "cpu_seconds": cpu_seconds,
        "average_cpu_percent_one_core": 100 * cpu_seconds / elapsed,
        "peak_rss_mib": max(row[3] for row in resource_rows),
        "peak_threads": max(row[4] for row in resource_rows),
        "output_poses": len(trajectory), **counters,
        "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "latency_p99_ms": float(np.percentile(latencies, 99)) if latencies else None,
        "failure_reboots": log.count("failure detection!") + log.count("system reboot!"),
        **trajectory_metrics(trajectory, truth),
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--bags-root", type=Path, default=Path.home() / "vicon_room2/bags")
    parser.add_argument("--config", type=Path, default=root / "src/vins_estimator/config/euroc/euroc_mono_imu_config.yaml")
    parser.add_argument("--binary", type=Path, default=root / "install/vins/lib/vins/vins_node")
    parser.add_argument("--output", type=Path, default=root / "output/performance")
    parser.add_argument("--label", required=True)
    parser.add_argument("--datasets", nargs="+", default=["V2_01_easy", "V2_02_medium", "V2_03_difficult"])
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=None, help="Optional bag-time seconds; omit for complete sequences")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--domain-id", type=int, default=173)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--drain-timeout", type=float, default=30)
    parser.add_argument("--parameter", action="append", help="Extra ROS parameter, e.g. opencv_threads:=1")
    args = parser.parse_args()
    args.binary, args.config, args.output = args.binary.resolve(), args.config.resolve(), args.output.resolve()
    if args.rate <= 0 or args.repeats < 1:
        parser.error("rate and repeats must be positive")
    os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)
    # Only benchmark processes are launched into this domain. Do not run parallel
    # contenders on the same CPU: that would make CPU/latency comparisons unfair.
    target = args.output / args.label
    target.mkdir(parents=True, exist_ok=True)
    manifest = {"binary": str(args.binary), "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                "config": str(args.config), "config_text": args.config.read_text(),
                "argv": vars(args), "cpu_count": os.cpu_count(),
                "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "method": "ROS subscriptions; Vicon header-time interpolation; rigid position alignment, no scale/time fit"}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    results = []
    for dataset in args.datasets:
        bag = args.bags_root / dataset
        truth_path = args.output / "groundtruth" / f"{dataset}.csv"
        truth = np.loadtxt(truth_path, delimiter=",", skiprows=1) if truth_path.exists() else extract_truth(bag, truth_path)
        for repeat in range(1, args.repeats + 1):
            output = target / f"{dataset}_r{repeat}"
            results.append(run_bag(args, bag, output, truth))
            (target / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
