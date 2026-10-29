"""ROS 2 Jazzy VINS-Fusion entry point (OpenCV YAML is not ROS parameter YAML)."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


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
