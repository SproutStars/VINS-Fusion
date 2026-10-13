#pragma once

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <rclcpp/parameter.hpp>
#include <rclcpp/parameter_map.hpp>
#include <rclcpp/parameter_value.hpp>

namespace vins
{

/// Reads a VINS algorithm configuration file.
///
/// Two layouts are accepted:
///   * ROS 2 parameter YAML (`/**` -> `ros__parameters`), the format shipped
///     with this ROS 2 port and usable directly with `--params-file`;
///   * the legacy OpenCV `%YAML:1.0` layout from the ROS 1 version, so existing
///     configurations keep working.
///
/// The reader normalizes both layouts into a flat name -> value map and offers
/// small typed accessors with fallbacks. Missing keys never throw; callers
/// decide which parameters are mandatory.
class Config
{
public:
    static Config fromFile(const std::string & path)
    {
        std::ifstream stream(path, std::ios::binary);
        if (!stream)
            throw std::runtime_error("Cannot open VINS configuration: " + path);
        const std::string text((std::istreambuf_iterator<char>(stream)),
                               std::istreambuf_iterator<char>());
        Config config;
        if (text.find("ros__parameters") != std::string::npos)
            config.loadRos2(path);
        else
            config.loadOpenCv(path);
        if (config.values_.empty())
            throw std::runtime_error("Empty or invalid VINS configuration: " + path);
        return config;
    }

    bool has(const std::string & key) const { return values_.count(key) != 0; }

    int getInt(const std::string & key, int fallback = 0) const
    {
        const auto it = values_.find(key);
        return it == values_.end() ? fallback : static_cast<int>(std::llround(asDouble(it->second)));
    }

    double getDouble(const std::string & key, double fallback = 0.0) const
    {
        const auto it = values_.find(key);
        return it == values_.end() ? fallback : asDouble(it->second);
    }

    bool getBool(const std::string & key, bool fallback = false) const
    {
        const auto it = values_.find(key);
        if (it == values_.end())
            return fallback;
        if (it->second.get_type() == rclcpp::ParameterType::PARAMETER_BOOL)
            return it->second.get<bool>();
        return asDouble(it->second) != 0.0;
    }

    std::string getString(const std::string & key,
                          const std::string & fallback = std::string()) const
    {
        const auto it = values_.find(key);
        if (it == values_.end())
            return fallback;
        if (it->second.get_type() == rclcpp::ParameterType::PARAMETER_STRING)
            return it->second.get<std::string>();
        return fallback;
    }

    std::vector<double> getDoubleArray(const std::string & key) const
    {
        const auto it = values_.find(key);
        if (it == values_.end())
            return {};
        const auto & value = it->second;
        if (value.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY)
            return value.get<std::vector<double>>();
        if (value.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY)
        {
            const auto ints = value.get<std::vector<int64_t>>();
            return std::vector<double>(ints.begin(), ints.end());
        }
        if (value.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE)
            return {value.get<double>()};
        return {};
    }

private:
    static double asDouble(const rclcpp::ParameterValue & value)
    {
        switch (value.get_type())
        {
            case rclcpp::ParameterType::PARAMETER_INTEGER:
                return static_cast<double>(value.get<int64_t>());
            case rclcpp::ParameterType::PARAMETER_DOUBLE:
                return value.get<double>();
            case rclcpp::ParameterType::PARAMETER_BOOL:
                return value.get<bool>() ? 1.0 : 0.0;
            case rclcpp::ParameterType::PARAMETER_STRING:
                try
                {
                    return std::stod(value.get<std::string>());
                }
                catch (const std::exception &)
                {
                    return 0.0;
                }
            default:
                return 0.0;
        }
    }

    void loadRos2(const std::string & path)
    {
        const auto map = rclcpp::parameter_map_from_yaml_file(path);
        for (const auto & entry : map)
            for (const auto & parameter : entry.second)
                values_[parameter.get_name()] = parameter.get_parameter_value();
    }

    void loadOpenCv(const std::string & path)
    {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened())
            throw std::runtime_error("Invalid OpenCV YAML: " + path);
        const cv::FileNode root = fs.root();
        for (auto it = root.begin(); it != root.end(); ++it)
        {
            const cv::FileNode node = *it;
            if (node.isMap())
            {
                if (!node["data"].empty() && !node["rows"].empty())
                {
                    cv::Mat matrix;
                    node >> matrix;
                    cv::Mat as_double;
                    matrix.convertTo(as_double, CV_64F);
                    values_[node.name()] = rclcpp::ParameterValue(
                        std::vector<double>(as_double.begin<double>(), as_double.end<double>()));
                }
                continue;
            }
            if (node.isSeq())
            {
                cv::Mat matrix;
                node >> matrix;
                cv::Mat as_double;
                matrix.convertTo(as_double, CV_64F);
                values_[node.name()] = rclcpp::ParameterValue(
                    std::vector<double>(as_double.begin<double>(), as_double.end<double>()));
                continue;
            }
            if (node.isString())
                values_[node.name()] = rclcpp::ParameterValue(static_cast<std::string>(node));
            else if (node.isInt())
                values_[node.name()] = rclcpp::ParameterValue(static_cast<int64_t>(static_cast<int>(node)));
            else if (node.isReal())
                values_[node.name()] = rclcpp::ParameterValue(static_cast<double>(node));
        }
    }

    std::map<std::string, rclcpp::ParameterValue> values_;
};

}  // namespace vins
