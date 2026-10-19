/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <gtest/gtest.h>

#include <vins_common/utility.hpp>

#include <Eigen/Geometry>

#include <cmath>

TEST(UtilityRotation, YprRoundTrip)
{
    const Eigen::Vector3d ypr(30.0, -20.0, 10.0);
    const Eigen::Matrix3d R = Utility::ypr2R(ypr);

    EXPECT_TRUE((R.transpose() * R).isApprox(Eigen::Matrix3d::Identity(), 1e-12));
    EXPECT_NEAR(R.determinant(), 1.0, 1e-12);
    EXPECT_TRUE(Utility::R2ypr(R).isApprox(ypr, 1e-9));
}

TEST(UtilityRotation, GravityAlignment)
{
    const Eigen::Vector3d g(0.32, -0.17, 9.81);
    const Eigen::Matrix3d R = Utility::g2R(g);

    EXPECT_TRUE((R.transpose() * R).isApprox(Eigen::Matrix3d::Identity(), 1e-12));
    EXPECT_NEAR(R.determinant(), 1.0, 1e-12);
    EXPECT_TRUE((R * g.normalized()).isApprox(Eigen::Vector3d::UnitZ(), 1e-12));
    EXPECT_NEAR(Utility::R2ypr(R).x(), 0.0, 1e-9);
}

TEST(UtilityRotation, QuaternionAndSkewHelpers)
{
    const Eigen::Vector3d theta(0.02, -0.04, 0.06);
    const Eigen::Quaterniond dq = Utility::deltaQ(theta);
    EXPECT_DOUBLE_EQ(dq.w(), 1.0);
    EXPECT_DOUBLE_EQ(dq.x(), theta.x() / 2.0);
    EXPECT_DOUBLE_EQ(dq.y(), theta.y() / 2.0);
    EXPECT_DOUBLE_EQ(dq.z(), theta.z() / 2.0);

    const Eigen::Vector3d a(1.0, -2.0, 3.0);
    const Eigen::Vector3d b(4.0, 5.0, -6.0);
    Eigen::Matrix3d expected_skew;
    expected_skew << 0.0, -a.z(), a.y(),
        a.z(), 0.0, -a.x(),
        -a.y(), a.x(), 0.0;
    EXPECT_TRUE(Utility::skewSymmetric(a).isApprox(expected_skew, 1e-12));
    EXPECT_TRUE((Utility::skewSymmetric(a) * b).isApprox(a.cross(b), 1e-12));

    const Eigen::Quaterniond q(Eigen::AngleAxisd(0.3, Eigen::Vector3d(1.0, 2.0, 3.0).normalized()));
    const Eigen::Quaterniond p(Eigen::AngleAxisd(-0.2, Eigen::Vector3d(-2.0, 1.0, 0.5).normalized()));
    const Eigen::Vector4d qv(q.w(), q.x(), q.y(), q.z());
    const Eigen::Vector4d pv(p.w(), p.x(), p.y(), p.z());
    EXPECT_TRUE((Utility::Qleft(q) * pv).isApprox(Utility::Qright(p) * qv, 1e-12));
    EXPECT_DOUBLE_EQ(Utility::positify(q).w(), q.w());
    EXPECT_DOUBLE_EQ(Utility::positify(q).x(), q.x());
}

TEST(UtilityAngle, NormalizeAngle)
{
    EXPECT_NEAR(Utility::normalizeAngle(190.0), -170.0, 1e-12);
    EXPECT_NEAR(Utility::normalizeAngle(-190.0), 170.0, 1e-12);
    EXPECT_NEAR(Utility::normalizeAngle(30.0), 30.0, 1e-12);
}
