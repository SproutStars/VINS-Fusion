#include <gtest/gtest.h>
#include "factor/pose_local_parameterization.h"
#include <vins_common/ros_utils.hpp>

TEST(PoseManifold, RoundTripAndUnitQuaternion)
{
    double x[7] = {1, 2, 3, 0.2, -0.3, 0.1, 0.9273618495495703};
    double delta[6] = {0.1, -0.2, 0.4, 0.2, -0.1, 0.3};
    double y[7], recovered[6];
    PoseManifold manifold;
    ASSERT_TRUE(manifold.Plus(x, delta, y));
    ASSERT_TRUE(manifold.Minus(y, x, recovered));
    EXPECT_NEAR(Eigen::Map<Eigen::Quaterniond>(y + 3).norm(), 1.0, 1e-12);
    for (int i = 0; i < 6; ++i) EXPECT_NEAR(delta[i], recovered[i], 1e-12);
}
TEST(PoseManifold, JacobianMatchesFiniteDifference)
{
    double x[7] = {1, 2, 3, 0.2, -0.3, 0.1, 0.9273618495495703};
    auto J = PoseManifold::plusJacobian(x);
    PoseManifold manifold;
    for (int col = 0; col < 6; ++col) {
        double delta[6] = {}, positive[7], negative[7];
        delta[col] = 1e-7; manifold.Plus(x, delta, positive);
        delta[col] = -1e-7; manifold.Plus(x, delta, negative);
        for (int row = 0; row < 7; ++row)
            EXPECT_NEAR(J(row, col), (positive[row] - negative[row]) / 2e-7, 1e-8);
    }
    EXPECT_TRUE((PoseManifold::minusJacobian(x) * J).isApprox(Eigen::Matrix<double, 6, 6>::Identity(), 1e-12));
}
TEST(PoseManifold, AnalyticFactorLiftPreservesTangentJacobian)
{
    double x[7] = {1, 2, 3, 0.2, -0.3, 0.1, 0.9273618495495703};
    Eigen::Matrix<double, 2, 7, Eigen::RowMajor> J;
    J.setRandom(); J.col(6).setZero();
    Eigen::Matrix<double, 2, 6> local = J.leftCols<6>();
    PoseManifold::liftJacobian(x, J.data(), 2);
    EXPECT_TRUE((J * PoseManifold::plusJacobian(x)).isApprox(local, 1e-12));
}
TEST(RosTime, SecondsAreNotNanoseconds)
{
    auto time = vins::stamp(1403636579.123456);
    EXPECT_EQ(time.sec, 1403636579);
    EXPECT_NEAR(vins::seconds(time), 1403636579.123456, 1e-6);
    EXPECT_THROW(vins::stamp(-1.0), std::invalid_argument);
}

#include "factor/projectionTwoFrameOneCamFactor.h"
#include "factor/projectionTwoFrameTwoCamFactor.h"
#include "factor/projectionOneFrameTwoCamFactor.h"
#include "factor/initial_pose_factor.h"
#include "factor/imu_factor.h"
#include "factor/marginalization_factor.h"

// Exercise the actual cost functions with non-identity orientations. Testing
// only identity poses would miss quaternion layout and perturbation errors.
static void checkPoseJacobians(ceres::CostFunction &factor,
                               std::vector<double *> parameters)
{
    const int n = factor.num_residuals();
    std::vector<double> residual(n);
    std::vector<std::vector<double>> data;
    for (int size : factor.parameter_block_sizes()) data.emplace_back(n * size);
    std::vector<double *> jacobians;
    for (auto &block : data) jacobians.push_back(block.data());
    ASSERT_TRUE(factor.Evaluate(parameters.data(), residual.data(), jacobians.data()));
    PoseManifold manifold;
    for (size_t block = 0; block < parameters.size(); ++block) {
        if (factor.parameter_block_sizes()[block] != 7) continue;
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 7, Eigen::RowMajor>> ambient(data[block].data(), n, 7);
        const Eigen::MatrixXd tangent = ambient * PoseManifold::plusJacobian(parameters[block]);
        const double *original = parameters[block];
        for (int col = 0; col < 6; ++col) {
            double delta[6] = {}, positive[7], negative[7];
            std::vector<double> r_plus(n), r_minus(n);
            delta[col] = 1e-7; manifold.Plus(original, delta, positive);
            delta[col] = -1e-7; manifold.Plus(original, delta, negative);
            parameters[block] = positive;
            ASSERT_TRUE(factor.Evaluate(parameters.data(), r_plus.data(), nullptr));
            parameters[block] = negative;
            ASSERT_TRUE(factor.Evaluate(parameters.data(), r_minus.data(), nullptr));
            parameters[block] = const_cast<double *>(original);
            for (int row = 0; row < n; ++row)
                EXPECT_NEAR(tangent(row, col), (r_plus[row] - r_minus[row]) / 2e-7, 1e-5)
                    << "block=" << block << " row=" << row << " col=" << col;
        }
    }
}

TEST(ProjectionFactors, TangentJacobiansMatchFiniteDifference)
{
    double pose0[7] = {0.1, 0.2, 0.3, 0.2, -0.3, 0.1, 0.9273618495495703};
    double pose1[7] = {0.3, 0.1, 0.2, -0.1, 0.2, -0.3, 0.9273618495495703};
    double cam0[7] = {0.01, 0.02, 0.03, 0.05, -0.1, 0.03, std::sqrt(1 - 0.0134)};
    double cam1[7] = {0.11, 0.02, 0.03, -0.04, 0.02, 0.1, std::sqrt(1 - 0.012)};
    double depth = 0.2, td = 0.01;
    const Eigen::Vector3d p0(0.1, 0.2, 1), p1(0.05, 0.1, 1);
    const Eigen::Vector2d v0(0.01, 0.02), v1(-0.02, 0.01);
    ProjectionTwoFrameOneCamFactor::sqrt_info.setIdentity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info.setIdentity();
    ProjectionOneFrameTwoCamFactor::sqrt_info.setIdentity();
    ProjectionTwoFrameOneCamFactor mono(p0, p1, v0, v1, 0.001, -0.001);
    ProjectionTwoFrameTwoCamFactor stereo(p0, p1, v0, v1, 0.001, -0.001);
    ProjectionOneFrameTwoCamFactor pair(p0, p1, v0, v1, 0.001, -0.001);
    checkPoseJacobians(mono, {pose0, pose1, cam0, &depth, &td});
    checkPoseJacobians(stereo, {pose0, pose1, cam0, cam1, &depth, &td});
    checkPoseJacobians(pair, {cam0, cam1, &depth, &td});
}

TEST(Marginalization, RestoresTangentCoordinatesForSchurComplement)
{
    double pose0[7] = {0.1, 0.2, 0.3, 0.2, -0.3, 0.1, 0.9273618495495703};
    double pose1[7] = {0.3, 0.1, 0.2, -0.1, 0.2, -0.3, 0.9273618495495703};
    double cam[7] = {0.01, 0.02, 0.03, 0, 0, 0, 1};
    double depth = 0.2, td = 0;
    ProjectionTwoFrameOneCamFactor::sqrt_info.setIdentity();
    ProjectionTwoFrameOneCamFactor factor(Eigen::Vector3d(0.1, 0.2, 1), Eigen::Vector3d(0.05, 0.1, 1),
        Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero(), 0, 0);
    std::vector<double *> parameters{pose0, pose1, cam, &depth, &td};
    ResidualBlockInfo block(&factor, nullptr, parameters, {0});
    block.Evaluate();
    double residual[2], pose_J[14];
    double *jacobians[] = {pose_J, nullptr, nullptr, nullptr, nullptr};
    factor.Evaluate(parameters.data(), residual, jacobians);
    Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> ambient(pose_J);
    EXPECT_TRUE(block.jacobians[0].leftCols(6).isApprox(ambient * PoseManifold::plusJacobian(pose0), 1e-12));
    EXPECT_TRUE(block.jacobians[0].col(6).isZero());
    delete[] block.raw_jacobians;
}

TEST(ImuFactor, PreintegrationAndPoseJacobians)
{
    // Constant velocity with a tilted body exercises both pose blocks and
    // gravity compensation, using a non-singular preintegration covariance.
    G = Eigen::Vector3d(0, 0, 9.81);
    ACC_N = 0.1; GYR_N = 0.01; ACC_W = 0.001; GYR_W = 0.0001;
    double pose0[7] = {0.1, 0.2, 0.3, 0.2, -0.3, 0.1, 0.9273618495495703};
    double pose1[7];
    std::copy(pose0, pose0 + 7, pose1);
    double speed_bias[9] = {0.1, -0.2, 0.3, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 3; ++i) pose1[i] += 0.1 * speed_bias[i];
    const Eigen::Vector3d acc = Eigen::Map<Eigen::Quaterniond>(pose0 + 3).inverse() * G;
    IntegrationBase integration(acc, Eigen::Vector3d::Zero(),
                                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    for (int i = 0; i < 20; ++i) integration.push_back(0.005, acc, Eigen::Vector3d::Zero());
    IMUFactor factor(&integration);
    const double *parameters[] = {pose0, speed_bias, pose1, speed_bias};
    double residual[15];
    ASSERT_TRUE(factor.Evaluate(parameters, residual, nullptr));
    EXPECT_LT((Eigen::Map<Eigen::Matrix<double, 15, 1>>(residual).norm()), 1e-8);
    checkPoseJacobians(factor, {pose0, speed_bias, pose1, speed_bias});
}
