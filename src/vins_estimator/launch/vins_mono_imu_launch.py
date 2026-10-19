"""Compatibility entry point for Mono IMU estimation; forwards to vins.launch.py."""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare("vins"), "launch", "vins.launch.py"])),
            launch_arguments={"mode": "mono_imu"}.items()),
    ])
