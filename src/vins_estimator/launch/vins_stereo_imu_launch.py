"""Convenience launch for stereo imu estimation."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("vins")
    return LaunchDescription([
        DeclareLaunchArgument("config_file", default_value=PathJoinSubstitution([
            share, "config", "euroc", "euroc_stereo_imu_config.yaml"])),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            PathJoinSubstitution([share, "launch", "vins.launch.py"])),
            launch_arguments={"config_file": LaunchConfiguration("config_file")}.items()),
    ])
