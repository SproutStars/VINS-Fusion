#pragma once

#include <rclcpp/rclcpp.hpp>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>

namespace vins
{
inline rclcpp::Logger logger() { return rclcpp::get_logger("vins_fusion"); }

// ROS 2 Time(int64_t) takes nanoseconds, not the ROS 1 floating-point seconds.
inline builtin_interfaces::msg::Time stamp(double seconds)
{
    if (!std::isfinite(seconds) || seconds < 0.0 ||
        seconds >= static_cast<double>(std::numeric_limits<int32_t>::max()))
        throw std::invalid_argument("Timestamp is outside the ROS message range");
    return rclcpp::Time(static_cast<int64_t>(std::llround(seconds * 1e9)), RCL_ROS_TIME);
}
inline double seconds(const builtin_interfaces::msg::Time &time)
{
    return rclcpp::Time(time, RCL_ROS_TIME).seconds();
}
inline std::string configFile(const rclcpp::Node::SharedPtr &node,
                              int argc, char **argv)
{
    const auto args = rclcpp::remove_ros_arguments(argc, argv);
    const std::string fallback = args.size() > 1 ? args[1] : "";
    auto path = node->declare_parameter<std::string>("config_file", fallback);
    if (path.empty() || !std::filesystem::is_regular_file(path))
        throw std::invalid_argument("Set -p config_file:=/absolute/path/to/config.yaml");
    return std::filesystem::absolute(path).string();
}
inline std::string resolvePath(const std::string &config, const std::string &path)
{
    auto result = std::filesystem::path(path);
    if (result.is_relative()) result = std::filesystem::path(config).parent_path() / result;
    return result.lexically_normal().string();
}
}  // namespace vins
