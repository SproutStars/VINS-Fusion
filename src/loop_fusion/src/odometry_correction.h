#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <nav_msgs/msg/odometry.hpp>
#include <stdexcept>
#include <string>

namespace loop_fusion
{
struct Correction
{
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
};

inline Correction composeCorrection(const Eigen::Matrix3d &r_drift,
                                    const Eigen::Vector3d &t_drift,
                                    const Eigen::Matrix3d &w_r_vio,
                                    const Eigen::Vector3d &w_t_vio)
{
    return {r_drift * w_r_vio, r_drift * w_t_vio + t_drift};
}

inline nav_msgs::msg::Odometry correctOdometry(const nav_msgs::msg::Odometry &input,
                                             const Correction &correction)
{
    // Twist and its covariance are in child_frame_id (body): changing the world
    // frame does not rotate them. Preserve all metadata, including the stamp.
    nav_msgs::msg::Odometry output = input;
    output.header.frame_id = "world";
    // Raw and corrected body frames use identical local axes; only the world
    // pose differs. Preserve body twist when renaming this known raw frame.
    if (output.child_frame_id == "body_vio") output.child_frame_id = "body";
    auto &pose = output.pose.pose;
    const auto &source = input.pose.pose;
    const Eigen::Vector3d position = correction.rotation *
        Eigen::Vector3d(source.position.x, source.position.y, source.position.z) +
        correction.translation;
    const Eigen::Quaterniond orientation = Eigen::Quaterniond(correction.rotation) *
        Eigen::Quaterniond(source.orientation.w, source.orientation.x,
                           source.orientation.y, source.orientation.z);
    pose.position.x = position.x();
    pose.position.y = position.y();
    pose.position.z = position.z();
    pose.orientation.w = orientation.w();
    pose.orientation.x = orientation.x();
    pose.orientation.y = orientation.y();
    pose.orientation.z = orientation.z();

    // Pose errors use world-frame [translation, fixed-axis rotation]. Rotate
    // the complete covariance, including translation/rotation cross terms.
    // This conditions on the correction; loop-transform uncertainty is NOT
    // included (the pose graph does not estimate/publish that covariance).
    using Matrix6 = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>;
    Matrix6 transform = Matrix6::Zero();
    transform.topLeftCorner<3, 3>() = correction.rotation;
    transform.bottomRightCorner<3, 3>() = correction.rotation;
    Eigen::Map<Matrix6> covariance(output.pose.covariance.data());
    covariance = transform * Eigen::Map<const Matrix6>(input.pose.covariance.data()) *
                 transform.transpose();
    return output;
}

// Arguments must be ROS-resolved topic names, including any remapping.
inline void validateOdometryRouting(const std::string &input, const std::string &rectified,
                                    bool publish_primary, const std::string &primary)
{
    if (publish_primary && primary == rectified)
        throw std::invalid_argument("Primary and rectified odometry outputs resolve to the same topic");
    if (input == rectified || (publish_primary && input == primary))
        throw std::invalid_argument("Odometry input resolves to a corrected output topic; "
                                    "set input_odometry_topic to the raw estimator topic");
}
}  // namespace loop_fusion
