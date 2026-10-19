"""ROS 2 Jazzy VINS-Fusion entry point (OpenCV YAML is not ROS parameter YAML)."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("vins")
    config = LaunchConfiguration("config_file")
    clock = ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)
    output = LaunchConfiguration("output_path")
    return LaunchDescription([
        DeclareLaunchArgument("config_file", default_value=PathJoinSubstitution([
            share, "config", "euroc", "euroc_mono_imu_config.yaml"])),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("loop_fusion", default_value="false"),
        DeclareLaunchArgument("global_fusion", default_value="false"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("output_path", default_value=PathJoinSubstitution([
            share, "..", "..", "output"])),
        DeclareLaunchArgument("opencv_threads", default_value="1",
                              description="OpenCV worker threads; 1 minimizes CPU for mono EuRoC"),
        Node(
            package="vins",
            executable="vins_node",
            name="vins_estimator",
            output="screen",
            parameters=[{
                "config_file": config,
                "use_sim_time": clock,
                "output_path": output,
                "opencv_threads": ParameterValue(
                    LaunchConfiguration("opencv_threads"), value_type=int)
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
                "pose_graph_save_path": PathJoinSubstitution([
                    output, "pose_graph"
                    ])
                }]),
        Node(
            package="global_fusion",
            executable="global_fusion_node",
            name="global_fusion",
            output="screen",
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
