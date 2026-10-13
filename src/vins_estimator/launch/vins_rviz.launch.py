from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    return LaunchDescription([Node(package="rviz2", executable="rviz2",
        arguments=["-d", PathJoinSubstitution([FindPackageShare("vins"),
            "config", "vins_ros2.rviz"])])])
