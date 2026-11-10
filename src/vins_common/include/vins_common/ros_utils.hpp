#pragma once

#include <cmath>
#include <filesystem>
#include <limits>
#include <rclcpp/rclcpp.hpp>
#include <stdexcept>
#include <string>

namespace vins
{
inline rclcpp::Logger logger() { return rclcpp::get_logger("vins_fusion"); }

// ROS 2 Time(int64_t) takes nanoseconds, not the ROS 1 floating-point seconds.
inline builtin_interfaces::msg::Time stamp(double _seconds)
{
  if (!std::isfinite(_seconds) || _seconds < 0.0 || _seconds >= static_cast<double>(std::numeric_limits<int32_t>::max()))
    throw std::invalid_argument("Timestamp is outside the ROS message range");
  return rclcpp::Time(static_cast<int64_t>(std::llround(_seconds * 1e9)), RCL_ROS_TIME);
}
inline double seconds(const builtin_interfaces::msg::Time & _time)
{
  return rclcpp::Time(_time, RCL_ROS_TIME).seconds();
}
inline std::string configFile(const rclcpp::Node::SharedPtr & _node, int _argc, char ** _argv)
{
  const auto args = rclcpp::remove_ros_arguments(_argc, _argv);
  const std::string fallback = args.size() > 1 ? args[1] : "";
  auto path = _node->declare_parameter<std::string>("config_file", fallback);
  if (path.empty() || !std::filesystem::is_regular_file(path))
    throw std::invalid_argument("Set -p config_file:=/absolute/path/to/config.yaml");
  return std::filesystem::absolute(path).string();
}
inline std::string resolvePath(const std::string & _config, const std::string & _path)
{
  auto result = std::filesystem::path(_path);
  if (result.is_relative())
    result = std::filesystem::path(_config).parent_path() / result;
  return result.lexically_normal().string();
}
} // namespace vins
