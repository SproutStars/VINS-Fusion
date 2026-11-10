#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <nav_msgs/msg/odometry.hpp>
#include <stdexcept>
#include <string>

namespace loop_fusion
{
struct Correction_s {
  Eigen::Matrix3d rotation_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d translation_ = Eigen::Vector3d::Zero();
};

inline Correction_s composeCorrection(
  const Eigen::Matrix3d & _r_drift, const Eigen::Vector3d & _t_drift, const Eigen::Matrix3d & _w_r_vio,
  const Eigen::Vector3d & _w_t_vio)
{
  return {_r_drift * _w_r_vio, _r_drift * _w_t_vio + _t_drift};
}

inline nav_msgs::msg::Odometry
correctOdometry(const nav_msgs::msg::Odometry & _input, const Correction_s & _correction)
{
  // Twist and its covariance are in child_frame_id (body): changing the world
  // frame does not rotate them. Preserve all metadata, including the stamp.
  nav_msgs::msg::Odometry output = _input;
  output.header.frame_id = "world";
  // Raw and corrected body frames use identical local axes; only the world
  // pose differs. Preserve body twist when renaming this known raw frame.
  if (output.child_frame_id == "body_vio")
    output.child_frame_id = "body";
  auto & pose = output.pose.pose;
  const auto & source = _input.pose.pose;
  const Eigen::Vector3d position =
    _correction.rotation_ * Eigen::Vector3d(source.position.x, source.position.y, source.position.z) +
    _correction.translation_;
  const Eigen::Quaterniond orientation =
    Eigen::Quaterniond(_correction.rotation_) *
    Eigen::Quaterniond(source.orientation.w, source.orientation.x, source.orientation.y, source.orientation.z);
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
  transform.topLeftCorner<3, 3>() = _correction.rotation_;
  transform.bottomRightCorner<3, 3>() = _correction.rotation_;
  Eigen::Map<Matrix6> covariance(output.pose.covariance.data());
  covariance = transform * Eigen::Map<const Matrix6>(_input.pose.covariance.data()) * transform.transpose();
  return output;
}

// Arguments must be ROS-resolved topic names, including any remapping.
inline void validateOdometryRouting(
  const std::string & _input, const std::string & _rectified, bool _publish_primary,
  const std::string & _primary)
{
  if (_publish_primary && _primary == _rectified)
    throw std::invalid_argument("Primary and rectified odometry outputs resolve to the same topic");
  if (_input == _rectified || (_publish_primary && _input == _primary))
    throw std::invalid_argument("Odometry input resolves to a corrected output topic; "
                                "set input_odometry_topic to the raw estimator topic");
}
} // namespace loop_fusion
