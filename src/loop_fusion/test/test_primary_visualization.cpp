#include <gtest/gtest.h>
#include <limits>
#include "primary_visualization.h"

namespace
{
nav_msgs::msg::Odometry sample(int seconds = 1403636579, uint32_t nanos = 123456789)
{
    nav_msgs::msg::Odometry input;
    input.header.frame_id = "world_vio";
    input.child_frame_id = "body_vio";
    input.header.stamp.sec = seconds;
    input.header.stamp.nanosec = nanos;
    input.pose.pose.position.x = 1;
    input.pose.pose.position.y = -2;
    input.pose.pose.position.z = 3;
    input.pose.pose.orientation.w = 1;
    input.twist.twist.linear.x = 4;
    input.twist.covariance[0] = 0.1;
    return input;
}
loop_fusion::Correction correction()
{
    return {Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitZ()).toRotationMatrix(), Eigen::Vector3d(2, 3, 4)};
}
}

TEST(PrimaryVisualization, SameCorrectedOdometryDrivesPathAndBothWorldTransforms)
{
    const auto raw = sample();
    const auto transform = correction();
    const auto odometry = loop_fusion::correctOdometry(raw, transform);
    EXPECT_EQ(odometry.child_frame_id, "body");
    EXPECT_EQ(odometry.header.frame_id, "world");
    EXPECT_EQ(odometry.twist, raw.twist);
    loop_fusion::PrimaryPath history;
    const auto path = history.append(odometry);
    ASSERT_TRUE(path);
    ASSERT_EQ(path->poses.size(), 1u);
    EXPECT_EQ(path->header, odometry.header);
    EXPECT_EQ(path->poses.back().header, odometry.header);
    EXPECT_EQ(path->poses.back().pose, odometry.pose.pose);
    const auto tf = loop_fusion::primaryTransforms(odometry, transform,
        Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(), false);
    ASSERT_EQ(tf.size(), 2u); // No fabricated camera extrinsic before callback.
    EXPECT_EQ(tf[0].header, odometry.header);
    EXPECT_EQ(tf[0].child_frame_id, "body");
    EXPECT_DOUBLE_EQ(tf[0].transform.translation.x, odometry.pose.pose.position.x);
    EXPECT_DOUBLE_EQ(tf[0].transform.translation.y, odometry.pose.pose.position.y);
    EXPECT_DOUBLE_EQ(tf[0].transform.translation.z, odometry.pose.pose.position.z);
    EXPECT_EQ(tf[0].transform.rotation, odometry.pose.pose.orientation);
    EXPECT_EQ(tf[1].header, odometry.header);
    EXPECT_EQ(tf[1].child_frame_id, "world_vio");
    EXPECT_DOUBLE_EQ(tf[1].transform.translation.x, transform.translation.x());
    EXPECT_DOUBLE_EQ(tf[1].transform.translation.y, transform.translation.y());
    EXPECT_DOUBLE_EQ(tf[1].transform.translation.z, transform.translation.z());
    const auto &q = tf[1].transform.rotation;
    EXPECT_TRUE(Eigen::Quaterniond(q.w, q.x, q.y, q.z).toRotationMatrix().isApprox(transform.rotation));
}

TEST(PrimaryVisualization, CameraTransformUsesBodyExtrinsicsAndSameSensorStamp)
{
    const auto odometry = loop_fusion::correctOdometry(sample(), correction());
    const Eigen::Vector3d translation(0.1, -0.2, 0.3);
    const Eigen::Matrix3d rotation = Eigen::AngleAxisd(-0.4, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const auto tf = loop_fusion::primaryTransforms(odometry, correction(), translation, rotation, true);
    ASSERT_EQ(tf.size(), 3u);
    const auto &camera = tf.back();
    EXPECT_EQ(camera.header.frame_id, "body");
    EXPECT_EQ(camera.child_frame_id, "camera");
    EXPECT_EQ(camera.header.stamp, odometry.header.stamp);
    EXPECT_DOUBLE_EQ(camera.transform.translation.x, translation.x());
    EXPECT_DOUBLE_EQ(camera.transform.translation.y, translation.y());
    EXPECT_DOUBLE_EQ(camera.transform.translation.z, translation.z());
    const auto &q = camera.transform.rotation;
    EXPECT_TRUE(Eigen::Quaterniond(q.w, q.x, q.y, q.z).toRotationMatrix().isApprox(rotation));
}

TEST(PrimaryPath, BoundedHistoryRateGateAndLatestExactlyMatchesInput)
{
    loop_fusion::PrimaryPath history(3, 1.0);
    auto odometry = loop_fusion::correctOdometry(sample(), correction());
    ASSERT_TRUE(history.append(odometry));
    odometry.header.stamp.nanosec += 100000000;
    EXPECT_FALSE(history.append(odometry));
    odometry.header.stamp.nanosec += 100000000;
    EXPECT_FALSE(history.append(odometry));
    odometry.header.stamp.sec += 1;
    odometry.pose.pose.position.x = 99;
    const auto path = history.append(odometry);
    ASSERT_TRUE(path);
    EXPECT_EQ(path->poses.size(), 3u);
    EXPECT_EQ(path->header, odometry.header);
    EXPECT_EQ(path->poses.back().pose, odometry.pose.pose);
    EXPECT_EQ(path->poses.back().header, odometry.header);
}

TEST(PrimaryPath, BackwardNanosecondClearsHistoryAndReopensGate)
{
    loop_fusion::PrimaryPath history(3, 1.0);
    auto odometry = loop_fusion::correctOdometry(sample(), correction());
    ASSERT_TRUE(history.append(odometry));
    // Even a 1 ns rollback at epoch scale is detected without double conversion.
    --odometry.header.stamp.nanosec;
    const auto path = history.append(odometry);
    ASSERT_TRUE(path);
    ASSERT_EQ(path->poses.size(), 1u);
    EXPECT_EQ(path->poses.back().header.stamp, odometry.header.stamp);
}

TEST(PrimaryPath, KeepsHistoricalCorrectionsAndReplacesDuplicateTimestamp)
{
    loop_fusion::PrimaryPath history(5, 0.0);
    const auto first = loop_fusion::correctOdometry(sample(), {});
    ASSERT_TRUE(history.append(first));
    const auto second = loop_fusion::correctOdometry(sample(1403636580), correction());
    const auto path = history.append(second);
    ASSERT_TRUE(path);
    ASSERT_EQ(path->poses.size(), 2u);
    EXPECT_EQ(path->poses.front().pose, first.pose.pose);
    EXPECT_EQ(path->poses.back().pose, second.pose.pose);
    auto updated = second;
    updated.pose.pose.position.x += 1;
    const auto duplicate = history.append(updated);
    ASSERT_TRUE(duplicate);
    EXPECT_EQ(duplicate->poses.size(), 2u);
    EXPECT_EQ(duplicate->poses.back().pose, updated.pose.pose);
    history.clear();
    EXPECT_EQ(history.append(updated)->poses.size(), 1u);
}

TEST(PrimaryPath, ValidatesLimitsAndZeroCapacityDisablesHistory)
{
    EXPECT_THROW(loop_fusion::PrimaryPath(-1, 1), std::invalid_argument);
    EXPECT_THROW(loop_fusion::PrimaryPath(2, -1), std::invalid_argument);
    EXPECT_THROW(loop_fusion::PrimaryPath(2, std::numeric_limits<double>::infinity()), std::invalid_argument);
    EXPECT_THROW(loop_fusion::PrimaryPath(2, std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    loop_fusion::PrimaryPath disabled(0, 0);
    EXPECT_FALSE(disabled.append(sample()));
    loop_fusion::PrimaryPath each_sample(1, 0);
    EXPECT_TRUE(each_sample.append(sample()));
    EXPECT_TRUE(each_sample.append(sample(1403636580)));
}
