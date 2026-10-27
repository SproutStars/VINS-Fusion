#pragma once

#include <cmath>

#include <eigen3/Eigen/Core>
#include <nav_msgs/msg/odometry.hpp>

#include "backend/ekf_state.h"

namespace vins
{

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
inline void fillEkfOdometry(nav_msgs::msg::Odometry &msg,
                            const backend::EkfState &state,
                            const Eigen::Vector3d &raw_gyro,
                            double imu_dt)
{
    using Mat3 = Eigen::Matrix3d;
    using Mat6 = Eigen::Matrix<double, 6, 6>;
    using Mat6x15 = Eigen::Matrix<double, 6, backend::kImuStateSize>;
    using Mat15 = Eigen::Matrix<double, backend::kImuStateSize,
                                backend::kImuStateSize>;

    const backend::ImuNominal &imu = state.imu();
    const Mat3 R = backend::EkfState::rotation(imu.q);

    msg.pose.pose.position.x = imu.p.x();
    msg.pose.pose.position.y = imu.p.y();
    msg.pose.pose.position.z = imu.p.z();
    msg.pose.pose.orientation.w = imu.q(0);
    msg.pose.pose.orientation.x = imu.q(1);
    msg.pose.pose.orientation.y = imu.q(2);
    msg.pose.pose.orientation.z = imu.q(3);

    const Eigen::Vector3d body_velocity = R.transpose() * imu.v;
    msg.twist.twist.linear.x = body_velocity.x();
    msg.twist.twist.linear.y = body_velocity.y();
    msg.twist.twist.linear.z = body_velocity.z();
    msg.twist.twist.angular.x = raw_gyro.x() - imu.bg.x();
    msg.twist.twist.angular.y = raw_gyro.y() - imu.bg.y();
    msg.twist.twist.angular.z = raw_gyro.z() - imu.bg.z();

    Mat6x15 J_pose = Mat6x15::Zero();
    J_pose.block<3, 3>(0, backend::kImuP) = Mat3::Identity();
    J_pose.block<3, 3>(3, backend::kImuTheta) = R;

    Mat6x15 J_twist = Mat6x15::Zero();
    J_twist.block<3, 3>(0, backend::kImuTheta) =
        backend::EkfState::skew(body_velocity);
    J_twist.block<3, 3>(0, backend::kImuV) = R.transpose();
    J_twist.block<3, 3>(3, backend::kImuBg) = -Mat3::Identity();

    Mat6 pose_covariance = Mat6::Zero();
    Mat6 twist_covariance = Mat6::Zero();

    // An initialized EkfState always has at least the live 15x15 block. Keep an
    // uninitialized state from reading outside its covariance matrix; its means
    // are still filled and both covariance blocks remain zero.
    if (state.covariance().rows() >= backend::kImuStateSize &&
        state.covariance().cols() >= backend::kImuStateSize)
    {
        const Mat15 P_live =
            state.covariance().topLeftCorner(backend::kImuStateSize,
                                             backend::kImuStateSize);
        pose_covariance.noalias() = J_pose * P_live * J_pose.transpose();
        twist_covariance.noalias() = J_twist * P_live * J_twist.transpose();
    }

    if (std::isfinite(imu_dt) && imu_dt > 0.0)
    {
        const double gyro_rate_variance =
            state.options().gyr_noise * state.options().gyr_noise / imu_dt;
        twist_covariance.block<3, 3>(3, 3).diagonal().array() +=
            gyro_rate_variance;
    }

    // Covariance messages are symmetric by contract. The state covariance is
    // symmetric up to roundoff, so average it before exposing it to consumers.
    pose_covariance = (0.5 * (pose_covariance + pose_covariance.transpose())).eval();
    twist_covariance = (0.5 * (twist_covariance + twist_covariance.transpose())).eval();

    for (int row = 0; row < 6; ++row)
    {
        for (int col = 0; col < 6; ++col)
        {
            msg.pose.covariance[6 * row + col] = pose_covariance(row, col);
            msg.twist.covariance[6 * row + col] = twist_covariance(row, col);
        }
    }
}

}  // namespace vins
