#pragma once

#include <cmath>

#include <eigen3/Eigen/Core>
#include <nav_msgs/msg/odometry.hpp>

#include "backend/ekf_state.h"

namespace vins
{

// nav_msgs/Odometry twist uses child_frame_id, which is body for VINS.
inline void fillBodyLinearVelocity(
  nav_msgs::msg::Odometry & _msg, const Eigen::Matrix3d & _R_world_body, const Eigen::Vector3d & _world_velocity)
{
  const Eigen::Vector3d velocity = _R_world_body.transpose() * _world_velocity;
  _msg.twist.twist.linear.x = velocity.x();
  _msg.twist.twist.linear.y = velocity.y();
  _msg.twist.twist.linear.z = velocity.z();
}

// Fills the pose/twist mean and covariance of an Odometry message from the live
// Schur-EKF state. The caller owns the header and child_frame_id.
//
// EkfState uses a right (body-frame) attitude perturbation dtheta and stores
// position/velocity errors in the world frame. nav_msgs/Odometry uses fixed-axis
// world angular errors for pose, so the pose covariance is mapped with
//     e_pose = [dp; R * dtheta].
// The twist is expressed in the child (body) frame:
//     linear  = R^T * v
//     angular = raw_gyro - bg.
// Its error Jacobian is
//     dlinear  = skew(R^T * v) * dtheta + R^T * dv,
//     dangular = -dbg.
//
// Both 6x6 covariance blocks are obtained by congruence with these Jacobians
// from the live 15x15 marginal (theta, p, v, ba, bg), which retains the live
// cross-covariances. The caller-provided gyro is one sample with continuous
// noise density `options().gyr_noise`; its discrete angular-rate variance
// gyr_noise^2 / imu_dt is added to the angular block only when imu_dt is finite
// and positive. Otherwise only the state's gyro-bias uncertainty is reported.
// The helper does not reset or otherwise modify the filter state.
inline void fillEkfOdometry(
  nav_msgs::msg::Odometry & _msg, const backend::EkfState & _state, const Eigen::Vector3d & _raw_gyro,
  double _imu_dt)
{
  using Mat3 = Eigen::Matrix3d;
  using Mat6 = Eigen::Matrix<double, 6, 6>;
  using Mat6x15 = Eigen::Matrix<double, 6, backend::K_IMU_STATE_SIZE>;
  using Mat15 = Eigen::Matrix<double, backend::K_IMU_STATE_SIZE, backend::K_IMU_STATE_SIZE>;

  const backend::ImuNominal_s & imu = _state.imu();
  const Mat3 R = backend::EkfState::rotation(imu.q_);

  _msg.pose.pose.position.x = imu.p_.x();
  _msg.pose.pose.position.y = imu.p_.y();
  _msg.pose.pose.position.z = imu.p_.z();
  _msg.pose.pose.orientation.w = imu.q_(0);
  _msg.pose.pose.orientation.x = imu.q_(1);
  _msg.pose.pose.orientation.y = imu.q_(2);
  _msg.pose.pose.orientation.z = imu.q_(3);

  const Eigen::Vector3d body_velocity = R.transpose() * imu.v_;
  _msg.twist.twist.linear.x = body_velocity.x();
  _msg.twist.twist.linear.y = body_velocity.y();
  _msg.twist.twist.linear.z = body_velocity.z();
  _msg.twist.twist.angular.x = _raw_gyro.x() - imu.bg_.x();
  _msg.twist.twist.angular.y = _raw_gyro.y() - imu.bg_.y();
  _msg.twist.twist.angular.z = _raw_gyro.z() - imu.bg_.z();

  Mat6x15 J_pose = Mat6x15::Zero();
  J_pose.block<3, 3>(0, backend::K_IMU_P) = Mat3::Identity();
  J_pose.block<3, 3>(3, backend::K_IMU_THETA) = R;

  Mat6x15 J_twist = Mat6x15::Zero();
  J_twist.block<3, 3>(0, backend::K_IMU_THETA) = backend::EkfState::skew(body_velocity);
  J_twist.block<3, 3>(0, backend::K_IMU_V) = R.transpose();
  J_twist.block<3, 3>(3, backend::K_IMU_BG) = -Mat3::Identity();

  Mat6 pose_covariance = Mat6::Zero();
  Mat6 twist_covariance = Mat6::Zero();

  // An initialized EkfState always has at least the live 15x15 block. Keep an
  // uninitialized state from reading outside its covariance matrix; its means
  // are still filled and both covariance blocks remain zero.
  if (_state.covariance().rows() >= backend::K_IMU_STATE_SIZE && _state.covariance().cols() >= backend::K_IMU_STATE_SIZE) {
    const Mat15 P_live =
      _state.covariance().topLeftCorner(backend::K_IMU_STATE_SIZE, backend::K_IMU_STATE_SIZE);
    pose_covariance.noalias() = J_pose * P_live * J_pose.transpose();
    twist_covariance.noalias() = J_twist * P_live * J_twist.transpose();
  }

  if (std::isfinite(_imu_dt) && _imu_dt > 0.0) {
    const double gyro_rate_variance = _state.options().gyr_noise_ * _state.options().gyr_noise_ / _imu_dt;
    twist_covariance.block<3, 3>(3, 3).diagonal().array() += gyro_rate_variance;
  }

  // Covariance messages are symmetric by contract. The state covariance is
  // symmetric up to roundoff, so average it before exposing it to consumers.
  pose_covariance = (0.5 * (pose_covariance + pose_covariance.transpose())).eval();
  twist_covariance = (0.5 * (twist_covariance + twist_covariance.transpose())).eval();

  for (int row = 0; row < 6; ++row) {
    for (int col = 0; col < 6; ++col) {
      _msg.pose.covariance[6 * row + col] = pose_covariance(row, col);
      _msg.twist.covariance[6 * row + col] = twist_covariance(row, col);
    }
  }
}

}  // namespace vins
