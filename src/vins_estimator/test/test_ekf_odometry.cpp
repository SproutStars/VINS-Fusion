#include <gtest/gtest.h>

#include <eigen3/Eigen/Dense>
#include <nav_msgs/msg/odometry.hpp>

#include <array>
#include <cmath>
#include <limits>

#include "backend/ekf_state.h"
#include "utility/ekf_odometry.h"

namespace
{

using Eigen::Matrix3d;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Mat15 = Eigen::Matrix<double, vins::backend::kImuStateSize,
                             vins::backend::kImuStateSize>;
using Mat6x15 = Eigen::Matrix<double, 6, vins::backend::kImuStateSize>;

constexpr double kPi = 3.14159265358979323846;

vins::backend::EkfOptions testOptions()
{
    vins::backend::EkfOptions options;
    options.gyr_noise = 0.03;
    options.init_rotation_sigma = 0.02;
    options.init_position_sigma = 0.03;
    options.init_velocity_sigma = 0.04;
    options.init_acc_bias_sigma = 0.05;
    options.init_gyr_bias_sigma = 0.01;
    return options;
}

Mat15 correlatedCovariance()
{
    Eigen::Matrix<double, vins::backend::kImuStateSize, 1> direction;
    direction << 0.020, -0.010, 0.030,
                 0.040, 0.020, -0.030,
                 0.050, -0.020, 0.010,
                 0.010, 0.015, -0.010,
                 0.030, 0.020, -0.025;
    return 1e-3 * Mat15::Identity() + direction * direction.transpose();
}

Mat6x15 poseJacobian(const Matrix3d &R)
{
    Mat6x15 J = Mat6x15::Zero();
    J.block<3, 3>(0, vins::backend::kImuP) = Matrix3d::Identity();
    J.block<3, 3>(3, vins::backend::kImuTheta) = R;
    return J;
}

Mat6x15 twistJacobian(const Matrix3d &R, const Vector3d &body_velocity)
{
    Mat6x15 J = Mat6x15::Zero();
    J.block<3, 3>(0, vins::backend::kImuTheta) =
        vins::backend::EkfState::skew(body_velocity);
    J.block<3, 3>(0, vins::backend::kImuV) = R.transpose();
    J.block<3, 3>(3, vins::backend::kImuBg) = -Matrix3d::Identity();
    return J;
}

Matrix6d messageCovariance(const std::array<double, 36> &covariance)
{
    Matrix6d result;
    for (int row = 0; row < 6; ++row)
        for (int col = 0; col < 6; ++col)
            result(row, col) = covariance[6 * row + col];
    return result;
}

void expectSymmetricPsd(const Matrix6d &covariance)
{
    EXPECT_LT((covariance - covariance.transpose()).cwiseAbs().maxCoeff(), 1e-12);
    const Eigen::SelfAdjointEigenSolver<Matrix6d> solver(covariance);
    ASSERT_EQ(solver.info(), Eigen::Success);
    EXPECT_GT(solver.eigenvalues().minCoeff(), -1e-10);
}

}  // namespace

TEST(EkfOdometry, MapsRotatedMeansAndCovarianceWithoutTouchingHeader)
{
    vins::backend::EkfState state;
    const Vector4d q(std::cos(kPi / 4.0), 0.0, 0.0, std::sin(kPi / 4.0));
    const Vector3d p(1.0, -2.0, 0.5);
    const Vector3d v(0.5, -0.2, 1.2);
    const Vector3d ba(0.02, -0.01, 0.03);
    const Vector3d bg(0.01, -0.02, 0.03);
    state.reset(testOptions(), 42.0, q, p, v, ba, bg);

    const Mat15 P = correlatedCovariance();
    state.covariance() = P;

    nav_msgs::msg::Odometry msg;
    msg.header.stamp.sec = 123;
    msg.header.stamp.nanosec = 456;
    msg.header.frame_id = "world_test";
    msg.child_frame_id = "body_test";

    const Vector3d raw_gyro(0.11, 0.08, -0.07);
    const double imu_dt = 0.01;
    vins::fillEkfOdometry(msg, state, raw_gyro, imu_dt);

    EXPECT_EQ(msg.header.stamp.sec, 123);
    EXPECT_EQ(msg.header.stamp.nanosec, 456u);
    EXPECT_EQ(msg.header.frame_id, "world_test");
    EXPECT_EQ(msg.child_frame_id, "body_test");

    EXPECT_NEAR(msg.pose.pose.position.x, p.x(), 1e-15);
    EXPECT_NEAR(msg.pose.pose.position.y, p.y(), 1e-15);
    EXPECT_NEAR(msg.pose.pose.position.z, p.z(), 1e-15);
    EXPECT_NEAR(msg.pose.pose.orientation.w, q(0), 1e-15);
    EXPECT_NEAR(msg.pose.pose.orientation.x, q(1), 1e-15);
    EXPECT_NEAR(msg.pose.pose.orientation.y, q(2), 1e-15);
    EXPECT_NEAR(msg.pose.pose.orientation.z, q(3), 1e-15);

    const Matrix3d R = vins::backend::EkfState::rotation(q);
    const Vector3d body_velocity = R.transpose() * v;
    const Vector3d expected_angular = raw_gyro - bg;
    EXPECT_NEAR(msg.twist.twist.linear.x, body_velocity.x(), 1e-14);
    EXPECT_NEAR(msg.twist.twist.linear.y, body_velocity.y(), 1e-14);
    EXPECT_NEAR(msg.twist.twist.linear.z, body_velocity.z(), 1e-14);
    EXPECT_NEAR(msg.twist.twist.angular.x, expected_angular.x(), 1e-15);
    EXPECT_NEAR(msg.twist.twist.angular.y, expected_angular.y(), 1e-15);
    EXPECT_NEAR(msg.twist.twist.angular.z, expected_angular.z(), 1e-15);

    Matrix6d expected_pose = poseJacobian(R) * P * poseJacobian(R).transpose();
    Matrix6d expected_twist =
        twistJacobian(R, body_velocity) * P * twistJacobian(R, body_velocity).transpose();
    const double gyro_rate_variance =
        state.options().gyr_noise * state.options().gyr_noise / imu_dt;
    expected_twist.block<3, 3>(3, 3).diagonal().array() += gyro_rate_variance;

    const Matrix6d pose_covariance = messageCovariance(msg.pose.covariance);
    const Matrix6d twist_covariance = messageCovariance(msg.twist.covariance);
    EXPECT_LT((pose_covariance - expected_pose).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_LT((twist_covariance - expected_twist).cwiseAbs().maxCoeff(), 1e-12);

    // The correlated P must leave non-zero cross-covariances in both messages.
    EXPECT_GT(std::abs(pose_covariance(0, 3)), 1e-6);
    EXPECT_GT(std::abs(twist_covariance(0, 4)), 1e-6);
    expectSymmetricPsd(pose_covariance);
    expectSymmetricPsd(twist_covariance);
}

TEST(EkfOdometry, UsesOnlyLiveFifteenStateMarginalWhenClonesExist)
{
    vins::backend::EkfState state;
    const Vector4d q(std::cos(kPi / 6.0), 0.0, std::sin(kPi / 6.0), 0.0);
    state.reset(testOptions(), 7.0, q, Vector3d(0.2, 0.4, -0.1),
                Vector3d(0.3, -0.5, 0.8), Vector3d::Zero(),
                Vector3d(0.02, -0.01, 0.04));
    state.augment_clone(7.1);
    state.augment_clone(7.2);
    ASSERT_EQ(state.covariance().rows(), 27);

    const Mat15 P_live = correlatedCovariance();
    state.covariance().topLeftCorner<vins::backend::kImuStateSize,
                                     vins::backend::kImuStateSize>() = P_live;
    state.covariance().bottomRightCorner(12, 12).setConstant(1e6);
    state.covariance().topRightCorner(15, 12).setConstant(-1e5);
    state.covariance().bottomLeftCorner(12, 15).setConstant(-1e5);

    nav_msgs::msg::Odometry msg;
    const Vector3d raw_gyro(0.2, -0.1, 0.3);
    vins::fillEkfOdometry(msg, state, raw_gyro, 0.005);

    const Matrix3d R = vins::backend::EkfState::rotation(q);
    const Vector3d body_velocity = R.transpose() * state.imu().v;
    Matrix6d expected_pose = poseJacobian(R) * P_live * poseJacobian(R).transpose();
    Matrix6d expected_twist =
        twistJacobian(R, body_velocity) * P_live * twistJacobian(R, body_velocity).transpose();
    expected_twist.block<3, 3>(3, 3).diagonal().array() +=
        state.options().gyr_noise * state.options().gyr_noise / 0.005;

    EXPECT_LT((messageCovariance(msg.pose.covariance) - expected_pose).cwiseAbs().maxCoeff(),
              1e-12);
    EXPECT_LT((messageCovariance(msg.twist.covariance) - expected_twist).cwiseAbs().maxCoeff(),
              1e-12);
}

TEST(EkfOdometry, AddsGyroNoiseOnlyForFinitePositiveDt)
{
    vins::backend::EkfState state;
    state.reset(testOptions(), 0.0, Vector4d(1.0, 0.0, 0.0, 0.0),
                Vector3d::Zero(), Vector3d(0.4, -0.2, 0.1), Vector3d::Zero(),
                Vector3d(0.01, 0.02, -0.03));
    const Mat15 P = correlatedCovariance();
    state.covariance() = P;

    const Matrix3d R = Matrix3d::Identity();
    const Vector3d body_velocity = state.imu().v;
    const Matrix6d bias_covariance =
        twistJacobian(R, body_velocity) * P * twistJacobian(R, body_velocity).transpose();

    nav_msgs::msg::Odometry positive_dt_msg;
    vins::fillEkfOdometry(positive_dt_msg, state, Vector3d::Zero(), 0.02);
    const Matrix6d positive_dt_covariance = messageCovariance(positive_dt_msg.twist.covariance);
    const double expected_extra =
        state.options().gyr_noise * state.options().gyr_noise / 0.02;
    for (int axis = 0; axis < 3; ++axis)
        EXPECT_NEAR(positive_dt_covariance(3 + axis, 3 + axis),
                    bias_covariance(3 + axis, 3 + axis) + expected_extra, 1e-14);

    for (double invalid_dt : {0.0, -0.01,
                              std::numeric_limits<double>::quiet_NaN(),
                              std::numeric_limits<double>::infinity()})
    {
        nav_msgs::msg::Odometry msg;
        vins::fillEkfOdometry(msg, state, Vector3d::Zero(), invalid_dt);
        const Matrix6d covariance = messageCovariance(msg.twist.covariance);
        EXPECT_LT((covariance - bias_covariance).cwiseAbs().maxCoeff(), 1e-14)
            << "imu_dt=" << invalid_dt;
    }
}
