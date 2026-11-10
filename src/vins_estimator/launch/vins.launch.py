"""ROS 2 Jazzy VINS-Fusion entry point; algorithm YAML also accepts OpenCV layout."""
import os
import shlex
import shutil

import cv2
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitution import Substitution
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


class CpuAffinityPrefix(Substitution):
    """Build a taskset prefix from launch-only keys in the algorithm YAML."""

    def __init__(self, config_file):
        super().__init__()
        self._config_file = config_file

    def perform(self, context):
        path = perform_substitutions(
            context, normalize_to_list_of_substitutions(self._config_file))
        enabled, cores = self._read(path)
        if not self._truth(enabled):
            return ""
        cores = self._cores(cores)
        available = os.sched_getaffinity(0)
        unavailable = [cpu for cpu in cores if cpu not in available]
        if unavailable:
            raise RuntimeError(
                "CPU affinity cores {} are unavailable; process may use {}".format(
                    unavailable, sorted(available)))
        taskset = shutil.which("taskset")
        if taskset is None:
            raise RuntimeError("CPU affinity requires 'taskset', but it was not found")
        return shlex.join([taskset, "--cpu-list", ",".join(map(str, cores))])

    def _read(self, path):
        try:
            with open(path, "r", encoding="utf-8") as stream:
                text = stream.read()
        except OSError as error:
            raise RuntimeError("Cannot read CPU affinity config {}: {}".format(
                path, error)) from error
        if text.lstrip().startswith("%YAML:1.0"):
            return self._read_legacy(path)
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise RuntimeError("Invalid YAML in CPU affinity config {}: {}".format(
                path, error)) from error
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise RuntimeError(
                "Invalid YAML in CPU affinity config {}: expected a mapping".format(path))
        parameters = {}
        for _, node in sorted(data.items(), key=lambda item: item[0] != "/**"):
            if isinstance(node, dict) and isinstance(node.get("ros__parameters"), dict):
                parameters.update(node["ros__parameters"])
        return (parameters.get("cpu_affinity_enabled", False),
                parameters.get("cpu_affinity_cores", []))

    @staticmethod
    def _truth(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        raise ValueError(
            "cpu_affinity_enabled must be a bool or 0/1, got {!r}".format(value))

    @staticmethod
    def _cores(value):
        if (not isinstance(value, list) or not value or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in value)):
            raise ValueError(
                "cpu_affinity_cores must be a non-empty list of non-negative "
                "integers, got {!r}".format(value))
        return sorted(set(value))

    @staticmethod
    def _read_legacy(path):
        try:
            storage = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        except cv2.error as error:
            raise RuntimeError("Invalid OpenCV YAML in CPU affinity config {}: {}".format(
                path, error)) from error
        try:
            if not storage.isOpened():
                raise RuntimeError("Cannot open CPU affinity config {}".format(path))
            enabled = storage.getNode("cpu_affinity_enabled")
            if enabled.empty():
                enabled = False
            elif enabled.isInt() or enabled.isReal():
                enabled = int(enabled.real()) if enabled.isInt() else enabled.real()
            else:
                raise ValueError("cpu_affinity_enabled must be a bool or 0/1")
            if not CpuAffinityPrefix._truth(enabled):
                return enabled, []
            cores = storage.getNode("cpu_affinity_cores")
            values = []
            if not cores.empty():
                if not cores.isSeq():
                    raise ValueError(
                        "cpu_affinity_cores must be a sequence of integers")
                for index in range(cores.size()):
                    item = cores.at(index)
                    if item.isInt():
                        values.append(int(item.real()))
                    elif item.isReal():
                        values.append(item.real())
                    else:
                        raise ValueError(
                            "cpu_affinity_cores must contain integers")
            return enabled, values
        finally:
            storage.release()


def generate_launch_description():
    share = FindPackageShare("vins")
    config = LaunchConfiguration("config_file")
    clock = ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)
    output = LaunchConfiguration("output_path")
    loop_enabled = LaunchConfiguration("loop_fusion")
    raw_odometry = PythonExpression([
        "'/vins_estimator/odometry_raw' if '", loop_enabled,
        "'.lower() in ('true', '1') else '/vins_estimator/odometry'",
    ])
    raw_path = PythonExpression([
        "'/vins_estimator/path_raw' if '", loop_enabled,
        "'.lower() in ('true', '1') else '/vins_estimator/path'",
    ])
    default_config = PathJoinSubstitution([
        share, "config", "euroc",
        ["euroc_", LaunchConfiguration("mode"), "_config.yaml"],
    ])
    return LaunchDescription([
        DeclareLaunchArgument(
            "mode", default_value="mono_imu",
            choices=["mono_imu", "stereo_imu", "stereo"],
            description="VINS mode used to select the default EuRoC config file"),
        DeclareLaunchArgument(
            "config_file", default_value=default_config,
            description="Algorithm YAML; an explicit value overrides the mode default"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("loop_fusion", default_value="true"),
        DeclareLaunchArgument("global_fusion", default_value="false"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("output_path", default_value=PathJoinSubstitution([
            share, "..", "..", "output"])),
        DeclareLaunchArgument("opencv_threads", default_value="1",
                              description="OpenCV worker threads; 1 minimizes CPU for mono EuRoC"),
        DeclareLaunchArgument("tracking_queue_size", default_value="5",
                              description="Maximum queued image pairs before stale frames are dropped"),
        Node(
            package="vins",
            executable="vins_node",
            name="vins_estimator",
            output="screen",
            prefix=CpuAffinityPrefix(config),
            remappings=[("~/odometry", raw_odometry), ("~/path", raw_path)],
            parameters=[{
                "config_file": config,
                "use_sim_time": clock,
                "output_path": output,
                "opencv_threads": ParameterValue(
                    LaunchConfiguration("opencv_threads"), value_type=int),
                "use_raw_frames": ParameterValue(loop_enabled, value_type=bool),
                "tracking_queue_size": ParameterValue(
                    LaunchConfiguration("tracking_queue_size"), value_type=int)
                }]),
        Node(
            package="loop_fusion",
            executable="loop_fusion_node",
            name="loop_fusion",
            output="screen",
            condition=IfCondition(LaunchConfiguration("loop_fusion")),
            parameters=[{
                "config_file": config,
                "use_sim_time": clock,
                "output_path": output,
                "input_odometry_topic": "/vins_estimator/odometry_raw",
                "publish_primary_odometry": True,
                "publish_primary_visualization": True,
                "pose_graph_save_path": PathJoinSubstitution([
                    output, "pose_graph"
                    ])
                }]),
        Node(
            package="global_fusion",
            executable="global_fusion_node",
            name="global_fusion",
            output="screen",
            # GPS fusion expects continuous local VIO, not loop-closure jumps.
            remappings=[("/vins_estimator/odometry", raw_odometry)],
            condition=IfCondition(LaunchConfiguration("global_fusion")),
            parameters=[{
                "use_sim_time": clock,
                "output_file": PathJoinSubstitution([
                    output, "vio_global.csv"])
                }]),
        Node(
            package="rviz2",
            executable="rviz2",
            output="screen",
            condition=IfCondition(LaunchConfiguration("rviz")),
            arguments=["-d", PathJoinSubstitution([share, "config", "vins_ros2.rviz"])],
            parameters=[{"use_sim_time": clock}]),
    ])
