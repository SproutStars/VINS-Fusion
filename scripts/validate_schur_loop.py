#!/usr/bin/env python3
"""Replay EuRoC with loop_fusion and audit real cloud/keyframe/loop outputs.

Source ROS and this workspace's install/setup.bash first. Outputs include the
exact config, binary hashes, node logs and per-sequence JSON measurements.
A short --duration is useful for reproducing missing clouds, but cannot prove
loop closure. The default uses complete bags and requires a real loop edge.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import bisect
import math
import os
from pathlib import Path
import subprocess
import time

from benchmark_euroc import extract_truth, stop_process, trajectory_metrics


def stamp_ns(msg):
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


def run(args, dataset):
    import rclpy
    from nav_msgs.msg import Odometry, Path as RosPath
    from sensor_msgs.msg import Image, PointCloud
    from visualization_msgs.msg import MarkerArray, Marker
    from tf2_msgs.msg import TFMessage
    from rclpy.qos import qos_profile_sensor_data

    output = args.output / dataset
    output.mkdir(parents=True, exist_ok=True)
    bag = args.bags_root / dataset
    binary_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (args.estimator, args.loop)}
    truth = extract_truth(bag, output / 'groundtruth.csv')
    node = rclpy.create_node('schur_loop_validation')
    counts = defaultdict(int)
    clouds = defaultdict(lambda: {'messages': 0, 'nonempty': 0, 'points': 0,
                                  'last_stamp': 0, 'last_nonempty_stamp': 0, 'invalid': 0})
    poses, rectified, images, keyposes, keypoints = {}, {}, set(), set(), set()
    trajectory, rect_trajectory, graph_trajectory = [], [], []
    graph = {'max_path_poses': 0, 'max_loop_edges': 0}
    primary_messages, raw_messages, corrected_messages = {}, {}, {}
    publisher_names = []
    path_publishers = []
    primary_paths, raw_paths, transforms = [], [], {}
    margin_samples = []
    first_image = [None]
    last_output = [time.monotonic()]

    def image(msg):
        stamp = stamp_ns(msg)
        images.add(stamp)
        if first_image[0] is None:
            first_image[0] = stamp
        counts['images'] += 1

    def odom(msg, corrected=False):
        stamp = stamp_ns(msg)
        p = msg.pose.pose.position
        xyz = (p.x, p.y, p.z)
        (corrected_messages if corrected else raw_messages)[stamp] = msg
        (rectified if corrected else poses)[stamp] = xyz
        (rect_trajectory if corrected else trajectory).append([stamp / 1e9, *xyz])
        counts['rectified_odometry' if corrected else 'odometry'] += 1
        last_output[0] = time.monotonic()

    def cloud(msg, name):
        stats = clouds[name]
        stamp = stamp_ns(msg)
        stats['messages'] += 1
        stats['points'] += len(msg.points)
        stats['last_stamp'] = stamp
        if msg.points:
            stats['nonempty'] += 1
            stats['last_nonempty_stamp'] = stamp
        stats['invalid'] += sum(not all(math.isfinite(v) for v in (p.x, p.y, p.z))
                                for p in msg.points)
        # Warmup uses Ceres; keep separate evidence after Schur takes over.
        if first_image[0] is not None and stamp - first_image[0] > 20 * 10**9:
            stats['post_20s_messages'] = stats.get('post_20s_messages', 0) + 1
            stats['post_20s_nonempty'] = stats.get('post_20s_nonempty', 0) + bool(msg.points)
        if args.save_margin and name == 'margin_cloud':
            margin_samples.extend((stamp, p.x, p.y, p.z) for p in msg.points)
        if name == 'keyframe_point':
            keypoints.add(stamp)
            valid = len(msg.channels) == len(msg.points) and all(
                len(c.values) == 5 and all(math.isfinite(v) for v in c.values)
                for c in msg.channels)
            counts['invalid_keyframe_channels'] += not valid

    def keypose(msg):
        counts['keyframe_pose'] += 1
        keyposes.add(stamp_ns(msg))

    def path(msg):
        counts['pose_graph_path'] += 1
        graph['max_path_poses'] = max(graph['max_path_poses'], len(msg.poses))
        graph_trajectory[:] = [[stamp_ns(p) / 1e9, p.pose.position.x,
                                p.pose.position.y, p.pose.position.z] for p in msg.poses]

    def markers(msg):
        edges = sum(m.type == Marker.LINE_STRIP and m.color.r == 1.0
                    and len(m.points) == 2 for m in msg.markers)
        graph['max_loop_edges'] = max(graph['max_loop_edges'], edges)

    def tf_message(msg):
        for transform in msg.transforms:
            key = (transform.header.frame_id, transform.child_frame_id, stamp_ns(transform))
            transforms.setdefault(key, []).append(transform)

    raw_topic = '/vins_estimator/odometry_raw' if args.launch_mode == 'on' else '/vins_estimator/odometry'
    subscriptions = [
        node.create_subscription(RosPath, '/vins_estimator/path', primary_paths.append, 100),
        node.create_subscription(RosPath, '/vins_estimator/path_raw', raw_paths.append, 100),
        node.create_subscription(TFMessage, '/tf', tf_message, 1000),
        node.create_subscription(Odometry, '/vins_estimator/odometry',
                                 lambda m: primary_messages.__setitem__(stamp_ns(m), m), 1000),
        node.create_subscription(Image, '/cam0/image_raw', image, qos_profile_sensor_data),
        node.create_subscription(Odometry, raw_topic, odom, 1000),
        node.create_subscription(Odometry, '/loop_fusion/odometry_rect', lambda m: odom(m, True), 1000),
        node.create_subscription(Odometry, '/vins_estimator/keyframe_pose', keypose, 1000),
        node.create_subscription(RosPath, '/loop_fusion/pose_graph_path', path, 100),
        node.create_subscription(MarkerArray, '/loop_fusion/pose_graph', markers, 100),
    ]
    for name in ('point_cloud', 'margin_cloud', 'keyframe_point'):
        subscriptions.append(node.create_subscription(
            PointCloud, '/vins_estimator/' + name, lambda m, n=name: cloud(m, n), 1000))
    processes, logs = [], []
    error = None
    try:
        if args.launch_mode != 'direct':
            import yaml
            config_data = yaml.safe_load(args.config.read_text())
            parameters = config_data['/**']['ros__parameters']
            parameters['backend_type'] = args.backend
            for key in ('cam0_calib', 'cam1_calib'):
                if key in parameters:
                    parameters[key] = str((args.config.parent / parameters[key]).resolve())
            launch_config = output / 'launch_config.yaml'
            launch_config.write_text(yaml.safe_dump(config_data))
            log = open(output / 'launch.log', 'w')
            logs.append(log)
            processes.append(subprocess.Popen([
                'ros2', 'launch', 'vins', 'vins.launch.py',
                'config_file:=' + str(launch_config), 'output_path:=' + str(output),
                'loop_fusion:=' + ('true' if args.launch_mode == 'on' else 'false'),
                'rviz:=false', 'use_sim_time:=true'], stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True))
        for binary, name in (() if args.launch_mode != 'direct' else
                             ((args.estimator, 'vins_estimator'), (args.loop, 'loop_fusion'))):
            log = open(output / (name + '.log'), 'w')
            logs.append(log)
            command = [str(binary), '--ros-args', '-r', '__node:=' + name,
                       '-p', 'config_file:=' + str(args.config), '-p', 'use_sim_time:=true',
                       '-p', 'output_path:=' + str(output)]
            if name == 'vins_estimator':
                command += ['-p', 'backend_type:=' + args.backend]
            else:
                command += ['-p', 'pose_graph_save_path:=' + str(output / 'pose_graph')]
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True))
        deadline = time.monotonic() + 40
        while not (node.count_publishers(raw_topic) and node.count_publishers('/vins_estimator/odometry') and
                   (args.launch_mode == 'off' or node.count_subscribers('/vins_estimator/keyframe_point') >= 2)):
            rclpy.spin_once(node, timeout_sec=0.05)
            if any(p.poll() is not None for p in processes):
                raise RuntimeError('Node exited during startup; see logs')
            if time.monotonic() > deadline:
                raise TimeoutError('ROS discovery timed out')
        # DDS endpoint discovery can precede node-name discovery. Wait for the
        # graph metadata instead of treating _NODE_NAME_UNKNOWN_ as a publisher.
        while True:
            publisher_names = [info.node_name for info in
                               node.get_publishers_info_by_topic('/vins_estimator/odometry')]
            if publisher_names and all(name != '_NODE_NAME_UNKNOWN_' for name in publisher_names):
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Primary publisher node-name discovery timed out')
            rclpy.spin_once(node, timeout_sec=0.05)
        path_publishers = [info.node_name for info in node.get_publishers_info_by_topic('/vins_estimator/path')]
        command = ['ros2', 'bag', 'play', str(bag), '--rate', str(args.rate),
                   '--clock', '100', '--disable-keyboard-controls', '--delay', '2',
                   '--topics', '/imu0', '/cam0/image_raw', '/cam1/image_raw']
        if args.duration:
            command += ['--playback-duration', str(args.duration)]
        log = open(output / 'player.log', 'w')
        logs.append(log)
        player = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(player)
        end = None
        deadline = time.monotonic() + args.timeout
        while True:
            rclpy.spin_once(node, timeout_sec=0.01)
            now = time.monotonic()
            if any(p.poll() is not None for p in processes[:-1]):
                raise RuntimeError('Node exited during replay; see logs')
            if player.poll() is not None:
                if player.returncode != 0:
                    raise RuntimeError('Bag player failed; see player.log')
                end = end or now
                if now - max(end, last_output[0]) > 8:
                    break
            if now > deadline:
                raise TimeoutError('Replay timed out')
    except Exception as exc:
        error = str(exc)
    finally:
        for process in reversed(processes):
            stop_process(process)
        for log in logs:
            log.close()
        node.destroy_node()
    del subscriptions
    if args.save_margin:
        import numpy as np
        np.savez_compressed(output / 'margin_cloud.npz',
                            points=np.asarray(margin_samples, dtype=float).reshape(-1, 4),
                            odometry=np.asarray(trajectory, dtype=float).reshape(-1, 4))
    matches = keyposes & keypoints
    # Headers round-trip through double in VINS, so allow 1 us at epoch scale.
    sorted_images = sorted(images)
    def has_image(stamp):
        index = bisect.bisect_left(sorted_images, stamp)
        return any(abs(sorted_images[i] - stamp) <= 1000
                   for i in (index - 1, index) if 0 <= i < len(sorted_images))
    image_matches = sum(has_image(stamp) for stamp in matches)
    corrections = [math.dist(poses[t], rectified[t]) for t in poses.keys() & rectified.keys()]
    estimator_log = (output / ('vins_estimator.log' if args.launch_mode == 'direct' else 'launch.log')).read_text()
    summary = dict(dataset=dataset, backend=args.backend, rate=args.rate,
                   launch_mode=args.launch_mode, primary_publishers=publisher_names,
                   primary_path_publishers=path_publishers, primary_path_messages=len(primary_paths),
                   playback_duration=args.duration, binary_sha256=binary_hashes,
                   error=error, counts=dict(counts),
                   clouds=dict(clouds), graph=graph, paired_keyframes=len(matches),
                   keyframes_matching_image=image_matches,
                   unpaired_keyframes=len(keyposes ^ keypoints),
                   max_position_correction_m=max(corrections, default=0),
                   odometry_metrics=trajectory_metrics(trajectory, truth),
                   rectified_metrics=trajectory_metrics(rect_trajectory, truth),
                   final_pose_graph_metrics=trajectory_metrics(graph_trajectory, truth),
                   schur_activation_logged='Schur EKF seeded' in estimator_log,
                   schur_seed_count=estimator_log.count('Schur EKF seeded'))
    checks = {
        'processes_healthy': error is None,
        'backend_active': args.backend != 'schur_ekf' or summary['schur_activation_logged'],
        'point_cloud_after_warmup': clouds['point_cloud'].get('post_20s_nonempty', 0) > 0,
        'margin_cloud_after_warmup': clouds['margin_cloud'].get('post_20s_nonempty', 0) > 0,
        'keyframes_after_warmup': clouds['keyframe_point'].get('post_20s_nonempty', 0) > 0,
        'valid_points': all(c['invalid'] == 0 for c in clouds.values()),
        'valid_keyframe_channels': counts['invalid_keyframe_channels'] == 0,
        'paired_keyframes': bool(matches) and not (keyposes ^ keypoints),
        'keyframe_images_available': bool(matches) and image_matches == len(matches),
        'pose_graph_grows': graph['max_path_poses'] > 10,
        'loop_constraint_detected': graph['max_loop_edges'] > 0,
        'loop_correction_applied': max(corrections, default=0) > 1e-5,
    }
    if args.launch_mode == 'off':
        for name in ('pose_graph_grows', 'loop_constraint_detected', 'loop_correction_applied'):
            checks.pop(name)
        checks['no_loop_output'] = not corrected_messages
    if args.launch_mode != 'direct':
        expected = corrected_messages if args.launch_mode == 'on' else raw_messages
        checks['single_primary_publisher'] = publisher_names == [
            'loop_fusion' if args.launch_mode == 'on' else 'vins_estimator']
        checks['primary_matches_selected_stream'] = bool(expected) and primary_messages == expected
        if args.launch_mode == 'on':
            checks['all_raw_messages_corrected'] = raw_messages.keys() == corrected_messages.keys()
            checks['body_twist_and_covariance_preserved'] = bool(raw_messages) and all(
                m.twist == corrected_messages[t].twist and
                corrected_messages[t].child_frame_id == ('body' if m.child_frame_id == 'body_vio' else m.child_frame_id) and
                m.header.stamp == corrected_messages[t].header.stamp
                for t, m in raw_messages.items() if t in corrected_messages)
    if args.launch_mode != 'direct':
        checks['single_primary_path_publisher'] = path_publishers == publisher_names
        checks['primary_path_matches_odometry'] = bool(primary_paths) and all(
            p.header.frame_id == 'world' and p.poses and len(p.poses) <= 2000 and
            stamp_ns(p) == stamp_ns(p.poses[-1]) and all(
                stamp_ns(pose) in primary_messages and
                pose.pose == primary_messages[stamp_ns(pose)].pose.pose and
                pose.header.frame_id == 'world' for pose in p.poses)
            for p in primary_paths)
        def tf_matches(t, msg):
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            a, b = t.transform.translation, t.transform.rotation
            return max(abs(a.x-p.x), abs(a.y-p.y), abs(a.z-p.z),
                       abs(b.x-q.x), abs(b.y-q.y), abs(b.z-q.z), abs(b.w-q.w)) < 1e-9
        checks['primary_tf_matches_odometry'] = bool(primary_messages) and all(
            len(transforms.get(('world', 'body', stamp), [])) == 1 and
            tf_matches(transforms[('world', 'body', stamp)][0], msg)
            for stamp, msg in primary_messages.items())
        checks['camera_tf_available'] = any(a == 'body' and b == 'camera' for a,b,_ in transforms)
        if args.launch_mode == 'on':
            checks['raw_path_in_separate_frame'] = bool(raw_paths) and all(
                p.header.frame_id == 'world_vio' for p in raw_paths)
            checks['raw_tf_separate_and_connected'] = all(
                ('world_vio', 'body_vio', stamp) in transforms and
                ('world', 'world_vio', stamp) in transforms for stamp in primary_messages)
            summary['raw_path_messages'] = len(raw_paths)
    summary['checks'] = checks
    summary['passed'] = all(checks.values())
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    return summary['passed']


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bags-root', type=Path, default=Path.home() / 'vicon_room2/bags')
    parser.add_argument('--config', type=Path, default=root / 'src/vins_estimator/config/euroc/euroc_mono_imu_config.yaml')
    parser.add_argument('--estimator', type=Path, default=root / 'install/vins/lib/vins/vins_node')
    parser.add_argument('--loop', type=Path, default=root / 'install/loop_fusion/lib/loop_fusion/loop_fusion_node')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', default=['V2_01_easy', 'V2_02_medium', 'V2_03_difficult'])
    parser.add_argument('--launch-mode', choices=['direct', 'on', 'off'], default='direct',
                        help='Use actual launch to verify primary odometry routing with loop on/off')
    parser.add_argument('--save-margin', action='store_true',
                        help='Save timestamped margin points and VIO poses for geometry inspection')
    parser.add_argument('--rate', type=float, default=1.0)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--backend', choices=['ceres', 'schur_ekf'], default='schur_ekf')
    parser.add_argument('--domain-id', type=int, default=184)
    parser.add_argument('--timeout', type=float, default=900)
    args = parser.parse_args()
    if args.rate <= 0 or (args.duration is not None and args.duration <= 0):
        parser.error('rate and duration must be positive')
    for name in ('bags_root', 'config', 'estimator', 'loop', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {**vars(args), 'config_text': args.config.read_text(),
                'binary_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (args.estimator, args.loop)}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n')
    os.environ['ROS_DOMAIN_ID'] = str(args.domain_id)
    import rclpy
    rclpy.init()
    try:
        passed = [run(args, dataset) for dataset in args.datasets]
    finally:
        rclpy.try_shutdown()
    return 0 if all(passed) else 1


if __name__ == '__main__':
    raise SystemExit(main())
