#!/usr/bin/env python3
"""Run after sourcing install/setup.bash. Requires rclpy, cv2 and numpy.

Uses a separate ROS domain and temporary files. Synthetic stereo exercises
tracking, Ceres optimization and marginalization; it is not an accuracy test.
"""
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("ROS_DOMAIN_ID", "73")

import rclpy
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu, NavSatFix, PointCloud
from std_msgs.msg import Bool
from std_srvs.srv import Trigger


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    processes = []
    logs = []
    with tempfile.TemporaryDirectory(prefix="vins-smoke-") as directory:
        task_path = Path(directory)
        os.environ["ROS_LOG_DIR"] = str(task_path / "log")
        rclpy.init()
        node = rclpy.create_node("vins_smoke_test")

        def start(package, executable, *arguments):
            log = open(task_path / f"{executable}-{len(logs)}.log", "w+")
            binary = Path(get_package_prefix(package)) / "lib" / package / executable
            process = subprocess.Popen([str(binary), *arguments], stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            logs.append(log)
            return process

        def spin_for(duration):
            end = time.monotonic() + duration
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.01)
                for process in processes:
                    check(process.poll() is None, f"Node exited unexpectedly: {process.args}")

        def wait_until(predicate, description, timeout=15):
            end = time.monotonic() + timeout
            while not predicate() and time.monotonic() < end:
                spin_for(0.05)
            check(predicate(), f"Timed out waiting for {description}")

        def stop(process):
            process.send_signal(signal.SIGINT)
            code = process.wait(timeout=10)
            processes.remove(process)
            check(code == 0, f"Node did not exit cleanly: {process.args}, code={code}")

        try:
            mono_config = str(Path(get_package_share_directory("vins")) /
                              "config/euroc/euroc_mono_imu_config.yaml")
            mono = start("vins", "vins_node", "--ros-args", "-p",
                         f"config_file:={mono_config}", "-p", f"output_path:={directory}")
            wait_until(lambda: "vins_estimator" in node.get_node_names(), "mono startup")
            stop(mono)
            print("PASS mono initialization and clean shutdown")

            # Build a rectified synthetic stereo setup. Cameras are 10 cm apart.
            # Exercises ROS 2 parameter YAML configs plus native CameraInfo files.
            for name, camera in (("left_info.yaml", "smoke_left"),
                                 ("right_info.yaml", "smoke_right")):
                (task_path / name).write_text(f"""image_width: 320
image_height: 240
camera_name: {camera}
camera_matrix:
  rows: 3
  cols: 3
  data: [200.0, 0.0, 160.0, 0.0, 200.0, 120.0, 0.0, 0.0, 1.0]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [0.0, 0.0, 0.0, 0.0, 0.0]
""")

            def ros2_scalar(value):
                if isinstance(value, bool):
                    return "true" if value else "false"
                if isinstance(value, str):
                    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
                return repr(value)

            values = [("imu", 0), ("num_of_cam", 2), ("image0_topic", "/smoke/left"),
                      ("image1_topic", "/smoke/right"), ("imu_topic", "/smoke/imu"),
                      ("output_path", directory),
                      ("pose_graph_save_path", str(task_path / "graph")),
                      ("cam0_calib", "left_info.yaml"), ("cam1_calib", "right_info.yaml"),
                      ("image_width", 320), ("image_height", 240), ("estimate_extrinsic", 0),
                      ("multiple_thread", 1), ("max_cnt", 100), ("min_dist", 10),
                      ("F_threshold", 1.0), ("show_track", 1), ("flow_back", 1),
                      ("max_solver_time", 0.05), ("max_num_iterations", 5),
                      ("keyframe_parallax", 10.0), ("estimate_td", 0), ("td", 0.0),
                      ("load_previous_pose_graph", 0), ("save_image", 0),
                      ("acc_n", 0.1), ("gyr_n", 0.01), ("acc_w", 0.001),
                      ("gyr_w", 0.0001), ("g_norm", 9.81)]
            extrinsic = np.eye(4)
            extrinsic[0, 3] = 0.1
            values.append(("body_T_cam0", np.eye(4).flatten().tolist()))
            values.append(("body_T_cam1", extrinsic.flatten().tolist()))
            stereo_config = task_path / "stereo.yaml"
            lines = ["/**:", "  ros__parameters:"]
            for key, value in values:
                if isinstance(value, list):
                    rendered = "[" + ", ".join(ros2_scalar(v) for v in value) + "]"
                else:
                    rendered = ros2_scalar(value)
                lines.append(f"    {key}: {rendered}")
            stereo_config.write_text("\n".join(lines) + "\n")

            start("vins", "vins_node", "--ros-args", "-p", f"config_file:={stereo_config}")
            start("loop_fusion", "loop_fusion_node", "--ros-args", "-p",
                  f"config_file:={stereo_config}")
            start("global_fusion", "global_fusion_node", "--ros-args", "-r",
                  "__node:=global_fusion", "-p", f"output_file:={task_path / 'global.csv'}")
            pub_left = node.create_publisher(Image, "/smoke/left", qos_profile_sensor_data)
            pub_right = node.create_publisher(Image, "/smoke/right", qos_profile_sensor_data)
            pub_restart = node.create_publisher(Bool, "/vins_restart", 10)
            pub_vio = node.create_publisher(Odometry, "/vins_estimator/odometry", 10)
            pub_gps = node.create_publisher(NavSatFix, "/gps", qos_profile_sensor_data)
            counts = {"track": 0, "odometry": 0, "cloud": 0, "global": 0}
            last_stamp = {}

            def receive(name, message):
                counts[name] += 1
                if hasattr(message, "header"):
                    last_stamp[name] = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9

            subscriptions = [
                node.create_subscription(Image, "/vins_estimator/image_track",
                                         lambda m: receive("track", m), 10),
                node.create_subscription(Odometry, "/vins_estimator/odometry",
                                         lambda m: receive("odometry", m), 10),
                node.create_subscription(PointCloud, "/vins_estimator/point_cloud",
                                         lambda m: receive("cloud", m), 10),
                node.create_subscription(Odometry, "/global_fusion/global_odometry",
                                         lambda m: receive("global", m), 10),
            ]
            wait_until(lambda: pub_left.get_subscription_count() >= 2 and
                       pub_right.get_subscription_count() >= 1 and
                       pub_gps.get_subscription_count() >= 1, "sensor subscriptions")

            rng = np.random.default_rng(42)
            left = rng.integers(0, 256, (240, 320), dtype=np.uint8)
            left = cv2.GaussianBlur(left, (5, 5), 1)
            right = np.zeros_like(left)
            right[:, :-5] = left[:, 5:]

            def publish_images(frame):
                for pub, image in ((pub_left, left), (pub_right, right)):
                    msg = Image()
                    msg.header.stamp.sec = 100 + frame // 20
                    msg.header.stamp.nanosec = (frame % 20) * 50_000_000
                    msg.header.frame_id = "camera"
                    msg.height, msg.width = image.shape
                    msg.encoding, msg.step = "mono8", image.shape[1]
                    msg.data = image.tobytes()
                    pub.publish(msg)
                spin_for(0.05)

            for frame in range(60):
                publish_images(frame)
            wait_until(lambda: counts["track"] > 10 and counts["cloud"] > 0,
                       "stereo tracking, optimization and point cloud")
            check(last_stamp["track"] >= 100, "ROS timestamp conversion is wrong")
            print(f"PASS stereo tracking, Ceres optimization and marginalization: {counts}")

            count_before = counts["track"]
            pub_restart.publish(Bool(data=True))
            spin_for(0.2)
            for frame in range(60, 80):
                publish_images(frame)
            wait_until(lambda: counts["track"] > count_before + 5, "tracking after restart")
            print("PASS estimator restart")

            for service_name in ("save_pose_graph", "new_sequence"):
                client = node.create_client(Trigger, f"/loop_fusion/{service_name}")
                check(client.wait_for_service(timeout_sec=10), f"Missing service {service_name}")
                future = client.call_async(Trigger.Request())
                wait_until(future.done, f"service {service_name}")
                check(future.result().success, future.result().message)
            check((task_path / "graph" / "pose_graph.txt").is_file(), "Pose graph file missing")
            print("PASS loop fusion services and resource loading")

            for index in range(8):
                gps = NavSatFix()
                gps.header.stamp.sec = 200 + index
                gps.status.status = 0
                gps.latitude, gps.longitude, gps.altitude = 22.3, 114.2, 20.0
                gps.position_covariance[0] = 1.0
                pub_gps.publish(gps)
                spin_for(0.03)
                vio = Odometry()
                vio.header.stamp = gps.header.stamp
                vio.pose.pose.orientation.w = 1.0
                pub_vio.publish(vio)
                spin_for(0.03)
            wait_until(lambda: last_stamp.get("global", 0) >= 200, "GPS fusion output")
            spin_for(2.1)  # Let the global optimizer consume the GPS constraints.
            check((task_path / "global.csv").stat().st_size > 0, "Global output file missing")
            print("PASS GPS/global fusion")

            # Restart with an IMU configuration and no arriving IMU. Stop while
            # the estimator waits inside its measurement loop.
            for process in processes[:]:
                stop(process)
            print("PASS all stereo/loop/global nodes exit cleanly")
            text = stereo_config.read_text().replace("imu: 0", "imu: 1")
            imu_config = task_path / "stereo_imu.yaml"
            imu_config.write_text(text)
            imu_process = start("vins", "vins_node", "--ros-args", "-p",
                                f"config_file:={imu_config}")
            wait_until(lambda: pub_right.get_subscription_count() > 0, "stereo IMU startup")
            for frame in range(4):
                publish_images(frame)
            stop(imu_process)
            print("PASS shutdown while waiting for IMU")

            # Now provide a 200 Hz stationary IMU stream alongside 20 Hz stereo.
            # Stereo scale makes stationary initialization observable here.
            imu_process = start("vins", "vins_node", "--ros-args", "-p",
                                f"config_file:={imu_config}")
            pub_imu = node.create_publisher(Imu, "/smoke/imu", qos_profile_sensor_data)
            wait_until(lambda: pub_imu.get_subscription_count() > 0 and
                       pub_right.get_subscription_count() > 0, "IMU subscriptions")
            odometry_before = counts["odometry"]
            def publish_imu(frame):
                for sample in range(10):
                    msg = Imu()
                    stamp_ns = 100_000_000_000 + frame * 50_000_000 + sample * 5_000_000
                    msg.header.stamp.sec = stamp_ns // 1_000_000_000
                    msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
                    msg.linear_acceleration.z = 9.81
                    pub_imu.publish(msg)

            for frame in range(60):
                publish_imu(frame)
                publish_images(frame)
            wait_until(lambda: counts["odometry"] > odometry_before + 5,
                       "stereo IMU initialization and optimization")
            print("PASS stereo IMU preintegration and optimization")
            pub_switch = node.create_publisher(Bool, "/vins_imu_switch", 10)
            wait_until(lambda: pub_switch.get_subscription_count() > 0, "sensor switch subscription")
            for enabled, frames in ((False, range(60, 100)), (True, range(100, 140))):
                pub_switch.publish(Bool(data=enabled))
                spin_for(0.2)
                before = counts["odometry"]
                for frame in frames:
                    if enabled:
                        publish_imu(frame)
                    publish_images(frame)
                wait_until(lambda: counts["odometry"] > before + 5, "sensor switch reinitialization")
            stop(imu_process)
            print("PASS runtime IMU switches and clean shutdown")
        except Exception:
            for log in logs:
                log.flush()
                log.seek(0)
                print(f"\n--- {Path(log.name).name} ---\n{log.read()[-18000:]}")
            raise
        finally:
            for process in processes:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            node.destroy_node()
            rclpy.shutdown()
            for log in logs:
                log.close()


if __name__ == "__main__":
    main()
