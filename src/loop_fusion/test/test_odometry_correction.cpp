#include <gtest/gtest.h>
#include "odometry_correction.h"

namespace
{
nav_msgs::msg::Odometry sample()
{
    nav_msgs::msg::Odometry message;
    message.header.frame_id = "world";
    message.header.stamp.sec = 1403636579;
    message.header.stamp.nanosec = 123456789;
    message.child_frame_id = "body";
    message.pose.pose.position.x = 1;
    message.pose.pose.position.y = 2;
    message.pose.pose.position.z = 3;
    const Eigen::Quaterniond q(Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitX()));
    message.pose.pose.orientation.w = q.w();
    message.pose.pose.orientation.x = q.x();
    message.twist.twist.linear.x = 2;
    message.twist.twist.linear.y = -3;
    message.twist.twist.linear.z = 4;
    message.twist.twist.angular.x = -0.2;
    message.twist.twist.angular.y = 0.3;
    message.twist.twist.angular.z = -0.4;
    for (int i = 0; i < 36; ++i)
        message.twist.covariance[i] = 0.01 * (i + 1);
    // Anisotropic SPD pose covariance with nonzero cross terms.
    Eigen::Matrix<double, 6, 6> factor;
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c)
            factor(r, c) = (r == c ? 2.0 + r : 0.03 * (r + 1) * (c + 2));
    using Matrix6 = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>;
    Eigen::Map<Matrix6>(message.pose.covariance.data()) = factor * factor.transpose();
    return message;
}
}

TEST(OdometryCorrection, IdentityPreservesCompleteMessage)
{
    const auto input = sample();
    const auto output = loop_fusion::correctOdometry(input, {});
    EXPECT_EQ(output, input);
}

TEST(OdometryCorrection, ComposesAlignmentAndDriftAndPreservesBodyTwist)
{
    const auto input = sample();
    const Eigen::Matrix3d alignment = Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::Matrix3d drift = Eigen::AngleAxisd(-0.3, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const Eigen::Vector3d align_t(1, -2, 3), drift_t(-4, 5, -6);
    const auto correction = loop_fusion::composeCorrection(drift, drift_t, alignment, align_t);
    const auto output = loop_fusion::correctOdometry(input, correction);
    const Eigen::Vector3d expected = drift * (alignment * Eigen::Vector3d(1, 2, 3) + align_t) + drift_t;
    EXPECT_NEAR(output.pose.pose.position.x, expected.x(), 1e-12);
    EXPECT_NEAR(output.pose.pose.position.y, expected.y(), 1e-12);
    EXPECT_NEAR(output.pose.pose.position.z, expected.z(), 1e-12);
    const auto &q = output.pose.pose.orientation;
    EXPECT_TRUE(Eigen::Quaterniond(q.w, q.x, q.y, q.z).toRotationMatrix().isApprox(
        drift * alignment * Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitX()).toRotationMatrix(), 1e-12));
    EXPECT_EQ(output.header.stamp, input.header.stamp);
    EXPECT_EQ(output.child_frame_id, input.child_frame_id);
    EXPECT_EQ(output.twist, input.twist);
}

TEST(OdometryCorrection, RotatesEveryPoseCovarianceBlockIncludingCrossTerms)
{
    const auto input = sample();
    loop_fusion::Correction correction;
    correction.rotation = Eigen::AngleAxisd(0.8, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
    correction.translation = Eigen::Vector3d(5, -6, 7);
    const auto output = loop_fusion::correctOdometry(input, correction);
    // Independent scalar reference for all four 3x3 blocks, including cross terms.
    for (int br = 0; br < 2; ++br)
        for (int bc = 0; bc < 2; ++bc)
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                {
                    double expected = 0;
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j)
                            expected += correction.rotation(r, i) *
                                input.pose.covariance[(3 * br + i) * 6 + 3 * bc + j] *
                                correction.rotation(c, j);
                    EXPECT_NEAR(output.pose.covariance[(3 * br + r) * 6 + 3 * bc + c], expected, 1e-12);
                }
    // Translation alone does not change pose-error covariance at the same body origin.
    correction.rotation.setIdentity();
    EXPECT_EQ(loop_fusion::correctOdometry(input, correction).pose.covariance, input.pose.covariance);
    EXPECT_EQ(output.twist, input.twist);
}

TEST(OdometryRouting, AllowsLegacyAndRawInputButRejectsResolvedFeedback)
{
    using loop_fusion::validateOdometryRouting;
    const std::string primary = "/vins_estimator/odometry", rect = "/loop_fusion/odometry_rect";
    EXPECT_NO_THROW(validateOdometryRouting(primary, rect, false, primary));
    EXPECT_NO_THROW(validateOdometryRouting("/vins_estimator/odometry_raw", rect, true, primary));
    EXPECT_THROW(validateOdometryRouting(primary, rect, true, primary), std::invalid_argument);
    EXPECT_THROW(validateOdometryRouting(rect, rect, false, primary), std::invalid_argument);
    EXPECT_THROW(validateOdometryRouting("/raw", primary, true, primary), std::invalid_argument);
}
