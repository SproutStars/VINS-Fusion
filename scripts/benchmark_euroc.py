#!/usr/bin/env python3
"""Replay EuRoC ROS 2 bags and measure the estimator, not RViz or the player.

Source /opt/ros/jazzy/setup.bash and install/setup.bash before running.
ATE uses sensor header timestamps, interpolated Vicon positions, and rigid SE(3)
position alignment (no fitted scale or time offset). Arrival latency is observed
at ROS subscriptions, not inferred from bag /clock (record/header times differ).

Image counters are message diagnostics: received_images counts primary/left
camera image messages and does not prove synchronization or processing, while
received_image_messages counts all enabled camera messages and
received_images_by_topic reports each stream separately.
"""
import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import time

import numpy as np
import psutil
import yaml


_REPLAY_CONFIG_KEYS = ("imu", "num_of_cam", "imu_topic", "image0_topic", "image1_topic")
_UNSUPPORTED_MODE_OVERRIDES = frozenset(("imu", "num_of_cam"))
_IMAGE_COUNT_SEMANTICS = {
    "received_images": "primary/left camera image messages; not synchronized or processed pairs",
    "received_image_messages": "total image messages across enabled camera topics",
    "received_images_by_topic": "image messages for each enabled camera topic",
}


def load_ros2_parameters(config_path):
    """Return the ros__parameters mapping from a ROS 2 YAML config file."""
    config_path = Path(config_path)
    try:
        document = yaml.safe_load(config_path.read_text())
    except OSError as error:
        raise ValueError(f"Cannot read ROS 2 configuration {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in ROS 2 configuration {config_path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"ROS 2 configuration {config_path} must contain a mapping")

    sections = []
    for node_name, node_config in document.items():
        if not isinstance(node_config, dict) or "ros__parameters" not in node_config:
            continue
        parameters = node_config["ros__parameters"]
        if not isinstance(parameters, dict):
            raise ValueError(
                f"ros__parameters for {node_name!r} in {config_path} must be a mapping")
        sections.append((node_name, parameters))
    if not sections:
        raise ValueError(f"ROS 2 configuration {config_path} has no ros__parameters mapping")
    if len(sections) != 1:
        names = ", ".join(repr(name) for name, _ in sections)
        raise ValueError(
            f"Expected exactly one ros__parameters mapping in {config_path}, found {names}")
    return dict(sections[0][1])


def parse_parameter_overrides(overrides):
    """Parse repeated ROS 2 NAME:=VALUE overrides into typed values."""
    parsed = {}
    for override in overrides or ():
        if not isinstance(override, str):
            raise ValueError("ROS parameter overrides must be strings")
        name, separator, raw_value = override.partition(":=")
        if not separator or not name or name != name.strip():
            raise ValueError(f"Invalid ROS parameter override {override!r}; expected NAME:=VALUE")
        try:
            parsed[name] = yaml.safe_load(raw_value)
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid YAML value in ROS parameter override {override!r}") from error
    return parsed


def _integer_parameter(parameters, name):
    value = parameters.get(name)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value)
    raise ValueError(f"{name} must be an integer, got {value!r}")


def _topic_parameter(parameters, name, required):
    value = parameters.get(name)
    if value is None:
        if required:
            raise ValueError(f"{name} is required for the selected replay mode")
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    if value != value.strip() or re.search(r"\s", value):
        raise ValueError(f"{name} must not contain whitespace: {value!r}")
    if not value.startswith("/"):
        raise ValueError(f"{name} must be an absolute ROS topic starting with '/': {value!r}")
    return value


def resolve_replay_config(parameters, overrides=None):
    """Resolve the streams that must be played for the effective VINS mode.

    ``imu`` and ``num_of_cam`` select the mode from ``config_file``. The ROS 2
    node only accepts topic overrides through ``--parameter``; mode overrides
    are rejected because they would not change the running estimator.
    """
    if not isinstance(parameters, dict):
        raise ValueError("ROS parameters must be a mapping")
    override_values = parse_parameter_overrides(overrides)
    unsupported = sorted(_UNSUPPORTED_MODE_OVERRIDES.intersection(override_values))
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(
            f"--parameter {names} is not supported by vins_node; set {names} in the YAML config")
    effective = dict(parameters)
    effective.update(override_values)

    use_imu = _integer_parameter(effective, "imu")
    num_of_cam = _integer_parameter(effective, "num_of_cam")
    if use_imu not in (0, 1):
        raise ValueError(f"imu must be 0 or 1, got {use_imu}")
    if num_of_cam not in (1, 2):
        raise ValueError(f"num_of_cam must be 1 or 2, got {num_of_cam}")

    topics = {}
    if use_imu:
        topics["imu"] = _topic_parameter(effective, "imu_topic", required=True)
    topics["image0"] = _topic_parameter(effective, "image0_topic", required=True)
    if num_of_cam == 2:
        topics["image1"] = _topic_parameter(effective, "image1_topic", required=True)

    topic_values = list(topics.values())
    if len(topic_values) != len(set(topic_values)):
        raise ValueError(f"Replay topics must be distinct, got {topics}")

    mode = ("mono" if num_of_cam == 1 else "stereo")
    if use_imu:
        mode += "_imu"
    playback_topics = list(topics.values())
    return {
        "mode": mode,
        "use_imu": bool(use_imu),
        "imu": use_imu,
        "num_of_cam": num_of_cam,
        "topics": topics,
        "image_topics": [topics["image0"], topics["image1"]] if num_of_cam == 2 else [topics["image0"]],
        "playback_topics": playback_topics,
        "effective_parameters": {name: effective[name] for name in _REPLAY_CONFIG_KEYS if name in effective},
    }


def load_replay_config(config_path, overrides=None):
    return resolve_replay_config(load_ros2_parameters(config_path), overrides)


def parse_cpu_affinity(value, available_cpus=None):
    """Parse comma-separated CPU IDs and require them in the current mask."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("CPU affinity must be a comma-separated string")
    tokens = value.split(",")
    if not tokens or any(not token.strip() for token in tokens):
        raise ValueError("CPU affinity must contain comma-separated CPU IDs")
    try:
        cpu_ids = [int(token) for token in tokens]
    except ValueError as error:
        raise ValueError("CPU affinity must contain only integer CPU IDs") from error
    if any(cpu_id < 0 for cpu_id in cpu_ids):
        raise ValueError("CPU affinity CPU IDs must be nonnegative")
    if len(cpu_ids) != len(set(cpu_ids)):
        raise ValueError(f"CPU affinity contains duplicate IDs: {cpu_ids}")
    available = set(os.sched_getaffinity(0) if available_cpus is None else available_cpus)
    unavailable = sorted(set(cpu_ids) - available)
    if unavailable:
        raise ValueError(
            f"CPU affinity contains IDs unavailable to this process: {unavailable}; "
            f"available IDs are {sorted(available)}")
    return cpu_ids


def build_estimator_command(binary, config, output, parameters=None, cpu_affinity=None,
                            taskset="taskset"):
    """Build the estimator argv; affinity is applied before exec by taskset."""
    command = [str(binary), "--ros-args", "-r", "__node:=vins_estimator",
               "-p", f"config_file:={config}", "-p", f"output_path:={output}",
               "-p", "use_sim_time:=true"]
    for parameter in parameters or ():
        command += ["-p", parameter]
    if cpu_affinity:
        command = [str(taskset), "-c", ",".join(str(cpu_id) for cpu_id in cpu_affinity), *command]
    return command


class ImageMessageCounter:
    """Count image messages without changing frame-pair metrics."""
    def __init__(self, image_topics):
        if not image_topics:
            raise ValueError("At least one image topic is required")
        self.primary_topic = image_topics[0]
        self.counts = {topic: 0 for topic in image_topics}
        self.received_images = 0
        self.received_image_messages = 0

    def record(self, topic):
        if topic not in self.counts:
            raise ValueError(f"Image message from unexpected topic {topic!r}")
        self.counts[topic] += 1
        self.received_image_messages += 1
        if topic == self.primary_topic:
            self.received_images += 1

    def metrics(self):
        return {
            "received_images": self.received_images,
            "received_image_messages": self.received_image_messages,
            "received_images_by_topic": dict(self.counts),
        }


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


_RESTART_REQUEST_PHRASES = (
    "scheduling a controlled estimator reset",
    "scheduling a controlled estimator restart",
    "controlled estimator reset is scheduled",
    "escalating to a full controlled estimator restart",
)

_RESTART_REASONS = (
    "solver_failure",
    "initialization",
    "failure_detection",
    "starvation",
    "backstop",
    "soft_escalation",
    "legacy",
    "other",
)


def _classify_restart_request(line):
    if "Schur EKF visually starved" in line:
        return "starvation"
    if "consecutive frames without a visual update" in line:
        return "backstop"
    if "escalating to a full controlled estimator restart" in line:
        return "soft_escalation"
    if "Failure detection triggered" in line:
        return "failure_detection"
    if "Solver failed during" in line:
        return "solver_failure"
    if "initialization" in line.lower():
        return "initialization"
    return "other"


def _event_timestamp(line):
    match = re.search(r"\[(\d+(?:\.\d+)?)\]", line)
    return float(match.group(1)) if match else None


def parse_restart_diagnostics(log_text):
    """Summarize explicit controlled-restart requests and reset entries.

    Request lines only say the estimator *scheduled* a reset/restart; several
    can be logged before one reset starts. The "Resetting estimator state
    after solver failure" line is emitted before clearState(), so it proves a
    reset was entered, not that it completed. No completion marker exists here,
    hence actual_resets_completed is None.
    """
    by_reason = {reason: 0 for reason in _RESTART_REASONS}
    events = []
    restart_requests = 0
    reset_entry_events = 0
    failure_detection_triggers = 0
    starvation_restarts = 0
    soft_restart_escalations = 0
    legacy_failure_reboots = 0

    for line_no, line in enumerate(log_text.splitlines(), start=1):
        legacy_hits = line.count("failure detection!") + line.count("system reboot!")
        if legacy_hits:
            legacy_failure_reboots += legacy_hits
            by_reason["legacy"] += legacy_hits
            events.append({
                "line_no": line_no,
                "kind": "legacy",
                "reason": "legacy",
                "message": line.strip(),
                "timestamp": _event_timestamp(line),
            })

        if "Failure detection triggered" in line:
            failure_detection_triggers += 1
        if ("Schur EKF visually starved" in line or
                "consecutive frames without a visual update" in line):
            starvation_restarts += 1
        if "escalating to a full controlled estimator restart" in line:
            soft_restart_escalations += 1

        if any(phrase in line for phrase in _RESTART_REQUEST_PHRASES):
            reason = _classify_restart_request(line)
            restart_requests += 1
            by_reason[reason] += 1
            events.append({
                "line_no": line_no,
                "kind": "request",
                "reason": reason,
                "message": line.strip(),
                "timestamp": _event_timestamp(line),
            })
        elif "Resetting estimator state after solver failure" in line:
            # Emitted before clearState(), so this is a reset-entry marker,
            # not proof of completion. The message does not name the request
            # that caused the reset, so keep it out of by_reason.
            reset_entry_events += 1
            events.append({
                "line_no": line_no,
                "kind": "reset_entry",
                # The entry message does not reveal which request won (Schur
                # starvation also reaches this path), so do not guess a reason.
                "reason": "unknown",
                "message": line.strip(),
                "timestamp": _event_timestamp(line),
            })

    return {
        "restart_requests": restart_requests,
        "reset_entry_events": reset_entry_events,
        "actual_resets_completed": None,
        "failure_detection_triggers": failure_detection_triggers,
        "starvation_restarts": starvation_restarts,
        "soft_restart_escalations": soft_restart_escalations,
        "legacy_failure_reboots": legacy_failure_reboots,
        "by_reason": by_reason,
        "events": events,
    }


def parse_restart_diagnostics_file(path):
    return parse_restart_diagnostics(Path(path).read_text(errors="replace"))


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


def run_bag(args, replay, bag, output, truth):
    import rclpy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    output.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    observer = rclpy.create_node("vins_benchmark_observer")
    arrivals = ArrivalTimes()
    trajectory, latencies = [], []
    image_topics = replay["image_topics"]
    image_counter = ImageMessageCounter(image_topics)
    counters = {"latency_misses": 0}
    last_output_wall = [time.monotonic()]

    def image(topic, data):
        # CDR encapsulation followed by Header.stamp. No image-array decoding.
        endian = "<" if data[1] & 1 else ">"
        sec, nsec = struct.unpack_from(endian + "iI", data, 4)
        if topic == image_topics[0]:
            arrivals.record(sec * 10**9 + nsec, time.monotonic_ns())
        image_counter.record(topic)

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
    subscriptions = [
        observer.create_subscription(
            Image, topic, lambda data, topic=topic: image(topic, data), qos, raw=True)
        for topic in image_topics
    ]
    subscriptions.append(
        observer.create_subscription(Odometry, "/vins_estimator/odometry", odometry, 100))
    estimator = player = None
    resource_rows = []
    env = os.environ.copy()
    with open(output / "estimator.log", "w") as estimator_log, open(output / "player.log", "w") as player_log:
        try:
            command = build_estimator_command(
                args.binary, args.config, output, args.parameter, args.cpu_affinity, args.taskset)
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
                       "--topics", *replay["playback_topics"]]
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
    restart_diagnostics = parse_restart_diagnostics(log)
    result = {
        "dataset": bag.name, "label": args.label, "rate": args.rate,
        "wall_s": elapsed, "cpu_seconds": cpu_seconds,
        "average_cpu_percent_one_core": 100 * cpu_seconds / elapsed,
        "peak_rss_mib": max(row[3] for row in resource_rows),
        "peak_threads": max(row[4] for row in resource_rows),
        "output_poses": len(trajectory), **image_counter.metrics(), **counters,
        "image_count_semantics": _IMAGE_COUNT_SEMANTICS,
        "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "latency_p99_ms": float(np.percentile(latencies, 99)) if latencies else None,
        "failure_reboots": restart_diagnostics["legacy_failure_reboots"],
        **restart_diagnostics,
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
    parser.add_argument("--cpu-affinity", help="Comma-separated estimator CPU IDs, e.g. 6,7")
    args = parser.parse_args()
    args.binary, args.config, args.output = args.binary.resolve(), args.config.resolve(), args.output.resolve()
    if args.rate <= 0 or args.repeats < 1:
        parser.error("rate and repeats must be positive")
    try:
        replay = load_replay_config(args.config, args.parameter)
        args.cpu_affinity = parse_cpu_affinity(args.cpu_affinity)
        args.taskset = None
        if args.cpu_affinity:
            args.taskset = shutil.which("taskset")
            if not args.taskset:
                raise ValueError("--cpu-affinity requires the taskset executable in PATH")
    except ValueError as error:
        parser.error(str(error))
    os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)
    # Only benchmark processes are launched into this domain. Do not run parallel
    # contenders on the same CPU: that would make CPU/latency comparisons unfair.
    target = args.output / args.label
    target.mkdir(parents=True, exist_ok=True)
    manifest = {"binary": str(args.binary), "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                "config": str(args.config), "config_text": args.config.read_text(),
                "mode": replay["mode"], "topics": replay["topics"],
                "playback_topics": replay["playback_topics"], "replay": replay,
                "image_count_semantics": _IMAGE_COUNT_SEMANTICS,
                "cpu_affinity": args.cpu_affinity,
                "cpu_affinity_method": (
                    "taskset -c prefix on estimator only; observer and bag player unbound"
                    if args.cpu_affinity else None),
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
            results.append(run_bag(args, replay, bag, output, truth))
            (target / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
