#!/usr/bin/env python3
"""Replay a EuRoC bag with an image-only blackout; validate EKF odometry.

Source ROS and the workspace first. Runs only vins_node and rosbag, in the
caller's ROS domain. IMU goes directly to the estimator; only the camera topic
is relayed. Use a separate output directory and a baseline GFTT config.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import psutil
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def stop(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--blackout-start', type=float, default=30.0)
    parser.add_argument('--blackout-end', type=float, default=40.0)
    parser.add_argument('--duration', type=float, default=55.0)
    args = parser.parse_args()
    if not (2 < args.blackout_start and
            args.blackout_end - args.blackout_start > 3 and
            args.duration - args.blackout_end >= 5):
        parser.error('require start>2, blackout>3 s and >=5 s recovery')
    for path in (args.binary, args.config, args.bag):
        if not path.exists():
            parser.error(f'input does not exist: {path}')
    args.output.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    node = rclpy.create_node('imu_blackout_check')
    camera = node.create_publisher(Image, '/blackout/cam0', qos_profile_sensor_data)
    origin = [None]
    counts = {'images': 0, 'forwarded': 0, 'dropped': 0}
    samples = {'imu': [], 'visual': []}
    resources = []
    process = [None]

    def image(msg):
        if origin[0] is None:
            origin[0] = stamp(msg)
        elapsed = stamp(msg) - origin[0]
        counts['images'] += 1
        if args.blackout_start <= elapsed < args.blackout_end:
            counts['dropped'] += 1
        else:
            camera.publish(msg)
            counts['forwarded'] += 1

    def odometry(kind, msg):
        if kind == 'imu' and origin[0] is not None and process[0] is not None:
            elapsed = stamp(msg) - origin[0]
            if args.blackout_start + 1 <= elapsed <= args.blackout_end - 1:
                if not resources or time.monotonic() - resources[-1]['wall'] >= .5:
                    cpu = process[0].cpu_times()
                    resources.append({'wall': time.monotonic(), 'cpu': cpu.user + cpu.system,
                                      'rss_mib': process[0].memory_info().rss / 2**20})
        samples[kind].append({'t': stamp(msg), 'arrival': time.monotonic(),
                              'frame': msg.header.frame_id,
                              'child': msg.child_frame_id,
                              'pose_cov': list(msg.pose.covariance),
                              'twist_cov': list(msg.twist.covariance)})

    subscriptions = [node.create_subscription(Image, '/cam0/image_raw', image,
                                              qos_profile_sensor_data)]
    for kind, topic in [('imu', 'imu_propagate'), ('visual', 'odometry')]:
        subscriptions.append(node.create_subscription(
            Odometry, '/vins_estimator/' + topic,
            lambda msg, kind=kind: odometry(kind, msg), 2000))
    estimator = player = None
    try:
        with (args.output / 'estimator.log').open('w') as elog, \
                (args.output / 'player.log').open('w') as plog:
            command = [str(args.binary.resolve()), '--ros-args',
                       '-p', 'config_file:=' + str(args.config.resolve()),
                       '-p', 'output_path:=' + str(args.output.resolve()),
                       '-p', 'backend_type:=schur_ekf', '-p', 'opencv_threads:=1',
                       '-p', 'image0_topic:=/blackout/cam0',
                       '-p', 'schur_starvation_hold:=true',
                       '-p', 'schur_starvation_soft_restart:=false',
                       '-p', 'schur_starvation_weak_update:=false',
                       '-p', 'use_sim_time:=true']
            estimator = subprocess.Popen(command, stdout=elog, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            process[0] = psutil.Process(estimator.pid)
            deadline = time.monotonic() + 30
            while node.count_publishers('/vins_estimator/imu_propagate') == 0:
                rclpy.spin_once(node, timeout_sec=.05)
                if estimator.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('estimator discovery failed; see estimator.log')
            player = subprocess.Popen(
                ['ros2', 'bag', 'play', str(args.bag.resolve()), '--rate', '1',
                 '--clock', '100', '--disable-keyboard-controls', '--delay', '2',
                 '--playback-duration', str(args.duration), '--topics',
                 '/imu0', '/cam0/image_raw'], stdout=plog, stderr=subprocess.STDOUT,
                start_new_session=True)
            deadline = time.monotonic() + args.duration + 40
            ended = None
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.01)
                if estimator.poll() is not None:
                    raise RuntimeError('estimator exited during replay')
                if player.poll() is not None:
                    if ended is None:
                        ended = time.monotonic()
                    if time.monotonic() - ended > 3:
                        break
            else:
                raise RuntimeError('replay timeout')
            if player.returncode != 0:
                raise RuntimeError('bag player failed')
    finally:
        stop(player)
        stop(estimator)
        node.destroy_node()
        rclpy.shutdown()
        (args.output / 'samples.json').write_text(json.dumps(samples))

    if origin[0] is None:
        raise RuntimeError('no camera input')
    # Exclude a one-second boundary on either end: the last queued visual
    # frame may finish just after camera cutoff, and recovery can re-anchor P.
    window = [s for s in samples['imu'] if args.blackout_start + 1 <=
              s['t'] - origin[0] <= args.blackout_end - 1]
    checks = {}
    checks['camera_cutoff_exercised'] = counts['dropped'] > 0
    checks['imu_samples_during_blackout'] = len(window) >= (args.blackout_end - args.blackout_start - 2) * 180
    checks['imu_timestamps_monotonic'] = all(b['t'] > a['t'] for a, b in zip(samples['imu'], samples['imu'][1:]))
    max_gap = max((b['t'] - a['t'] for a, b in zip(window, window[1:])), default=float('inf'))
    checks['blackout_max_gap_under_50ms'] = max_gap < .05
    wall_gaps = [b['arrival'] - a['arrival'] for a, b in zip(window, window[1:])]
    checks['blackout_arrival_gap_under_500ms'] = bool(wall_gaps) and max(wall_gaps) < .5
    checks['no_visual_headers_inside_blackout'] = not any(
        args.blackout_start + 1 <= s['t'] - origin[0] <= args.blackout_end - 1
        for s in samples['visual'])
    # Scope the visual-message checks to after seeding; early Ceres warmup has
    # no EKF covariance. A nonzero visual covariance must also return afterwards.
    visual = [s for s in samples['visual'] if s['t'] - origin[0] >= args.blackout_end]
    checks['visual_covariance_after_recovery'] = any(np.trace(np.array(s['pose_cov']).reshape(6, 6)) > 0 for s in visual)
    selected = window + [s for s in visual if any(s['pose_cov'])]
    checks['world_body_frames'] = bool(selected) and all(s['frame'] == 'world' and s['child'] == 'body' for s in selected)
    valid = True
    minimum_eigenvalue = float('inf')
    for sample in selected:
        for field in ['pose_cov', 'twist_cov']:
            matrix = np.array(sample[field]).reshape(6, 6)
            if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-9):
                valid = False
                continue
            eigenvalue = float(np.linalg.eigvalsh(matrix)[0])
            minimum_eigenvalue = min(minimum_eigenvalue, eigenvalue)
            valid &= eigenvalue >= -1e-8 * max(1, np.max(np.abs(matrix))) and np.trace(matrix) > 0
    checks['finite_symmetric_psd_nonzero_covariance'] = bool(selected) and bool(valid)
    traces = [float(np.trace(np.array(s['pose_cov']).reshape(6, 6)[:3, :3])) for s in window]
    checks['position_uncertainty_grows'] = len(traces) > 1 and traces[-1] > traces[0]
    report = {'checks': checks, 'passed': all(checks.values()), 'counts': counts,
              'imu_messages': len(samples['imu']), 'visual_messages': len(samples['visual']),
              'blackout_imu_messages': len(window), 'blackout_max_gap_seconds': max_gap,
              'blackout_arrival_gap_ms_p50_p95_max': list(np.percentile(wall_gaps, [50, 95, 100]) * 1000) if wall_gaps else [],
              'position_covariance_trace_start_end': [traces[0], traces[-1]] if traces else [],
              'minimum_covariance_eigenvalue': minimum_eigenvalue,
              'camera_origin': origin[0], 'blackout_seconds': [args.blackout_start, args.blackout_end],
              'duration_seconds': args.duration,
              'blackout_cpu_percent_single_core': 100 * (resources[-1]['cpu'] - resources[0]['cpu']) / (resources[-1]['wall'] - resources[0]['wall']) if len(resources) > 1 else None,
              'blackout_peak_rss_mib': max((r['rss_mib'] for r in resources), default=None),
              'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(),
              'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
              'config_text': args.config.read_text(), 'estimator_command': command,
              'bag': str(args.bag.resolve()), 'binary': str(args.binary.resolve())}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
