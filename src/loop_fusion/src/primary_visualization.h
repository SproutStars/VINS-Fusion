#pragma once

#include <cmath>
#include <cstdint>
#include <deque>
#include <optional>
#include <stdexcept>
#include <vector>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include "odometry_correction.h"

namespace loop_fusion
{
// A bounded live trajectory: each sample keeps the correction available when
// that odometry arrived. Later graph optimization does not rewrite this history.
// pose_graph_path remains the globally reoptimized trajectory.
class PrimaryPath
{
public:
    explicit PrimaryPath(int64_t capacity = 2000, double rate = 1.0)
    {
        if (capacity < 0 || !std::isfinite(rate) || rate < 0.0)
            throw std::invalid_argument("path_max_poses must be nonnegative; path_publish_rate must be finite and nonnegative");
        capacity_ = static_cast<size_t>(capacity);
        period_seconds_ = rate > 0.0 ? 1.0L / rate : 0.0L;
    }

    void clear()
    {
        poses_.clear();
        last_seen_.reset();
        last_published_.reset();
    }

    // Zero capacity disables path history/output, zero rate publishes every
    // sample. Integer stamps preserve nanoseconds even at EuRoC epoch times.
    std::optional<nav_msgs::msg::Path> append(const nav_msgs::msg::Odometry &odometry)
    {
        const int64_t time = static_cast<int64_t>(odometry.header.stamp.sec) * 1000000000LL +
                             odometry.header.stamp.nanosec;
        if (last_seen_ && time < *last_seen_) clear();
        last_seen_ = time;
        if (capacity_ == 0) return std::nullopt;
        geometry_msgs::msg::PoseStamped pose;
        pose.header = odometry.header;
        pose.pose = odometry.pose.pose;
        if (!poses_.empty() && poses_.back().header.stamp == pose.header.stamp)
            poses_.back() = pose;
        else
        {
            if (poses_.size() == capacity_) poses_.pop_front();
            poses_.push_back(pose);
        }
        if (last_published_ &&
            static_cast<long double>(time - *last_published_) / 1e9L < period_seconds_)
            return std::nullopt;
        last_published_ = time;
        nav_msgs::msg::Path path;
        path.header = odometry.header;
        path.poses.assign(poses_.begin(), poses_.end());
        return path;
    }

private:
    size_t capacity_;
    long double period_seconds_;
    std::optional<int64_t> last_seen_, last_published_;
    std::deque<geometry_msgs::msg::PoseStamped> poses_;
};

inline std::vector<geometry_msgs::msg::TransformStamped> primaryTransforms(
    const nav_msgs::msg::Odometry &odometry, const Correction &correction,
    const Eigen::Vector3d &camera_translation, const Eigen::Matrix3d &camera_rotation,
    bool extrinsic_ready)
{
    geometry_msgs::msg::TransformStamped body;
    body.header = odometry.header;
    body.child_frame_id = odometry.child_frame_id;
    body.transform.translation.x = odometry.pose.pose.position.x;
    body.transform.translation.y = odometry.pose.pose.position.y;
    body.transform.translation.z = odometry.pose.pose.position.z;
    body.transform.rotation = odometry.pose.pose.orientation;
    geometry_msgs::msg::TransformStamped raw_world;
    raw_world.header = odometry.header;
    raw_world.child_frame_id = "world_vio";
    raw_world.transform.translation.x = correction.translation.x();
    raw_world.transform.translation.y = correction.translation.y();
    raw_world.transform.translation.z = correction.translation.z();
    const Eigen::Quaterniond raw_q(correction.rotation);
    raw_world.transform.rotation.w = raw_q.w();
    raw_world.transform.rotation.x = raw_q.x();
    raw_world.transform.rotation.y = raw_q.y();
    raw_world.transform.rotation.z = raw_q.z();
    std::vector<geometry_msgs::msg::TransformStamped> transforms{body, raw_world};
    if (extrinsic_ready)
    {
        geometry_msgs::msg::TransformStamped camera;
        camera.header = odometry.header;
        camera.header.frame_id = odometry.child_frame_id;
        camera.child_frame_id = "camera";
        camera.transform.translation.x = camera_translation.x();
        camera.transform.translation.y = camera_translation.y();
        camera.transform.translation.z = camera_translation.z();
        const Eigen::Quaterniond q(camera_rotation);
        camera.transform.rotation.w = q.w();
        camera.transform.rotation.x = q.x();
        camera.transform.rotation.y = q.y();
        camera.transform.rotation.z = q.z();
        transforms.push_back(camera);
    }
    return transforms;
}
}  // namespace loop_fusion
