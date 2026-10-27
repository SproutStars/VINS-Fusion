// Standalone gtest suite for the ARM-oriented Schur-complement EKF state.
// Eigen + std + gtest only (no ROS / Ceres / OpenCV).
//
// The tests re-derive the analytic Jacobian / Schur complement locally instead
// of reusing the implementation, so sign, ordering and indexing mistakes in
// ekf_state.cpp are actually caught.

#include <gtest/gtest.h>
#include <eigen3/Eigen/Dense>

#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "backend/ekf_state.h"

namespace vins
{
namespace backend
{
namespace
{

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Mat15 = Eigen::Matrix<double, kImuStateSize, kImuStateSize>;
using Vec15 = Eigen::Matrix<double, kImuStateSize, 1>;
using Vec6 = Eigen::Matrix<double, kCloneStateSize, 1>;

constexpr double kPi = 3.14159265358979323846;

// --------------------------------------------------------------------------
// Test-local primitives (independent re-implementation of the conventions).
// --------------------------------------------------------------------------

Vector4d random_unit_quaternion(std::mt19937 &rng)
{
    std::normal_distribution<double> normal(0.0, 1.0);
    Vector4d q;
    for (int i = 0; i < 4; ++i)
        q(i) = normal(rng);
    return q / q.norm();
}

Vector3d random_axis_angle(std::mt19937 &rng)
{
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    Vector3d axis;
    for (int i = 0; i < 3; ++i)
        axis(i) = uni(rng);
    if (axis.norm() < 1e-6)
        axis = Vector3d::UnitX();
    axis.normalize();
    std::uniform_real_distribution<double> mag(0.0, 0.99 * kPi);
    return axis * mag(rng);
}

Matrix3d rotation_of(const Vector4d &q)
{
    const Eigen::Quaterniond eq(q(0), q(1), q(2), q(3));
    return eq.normalized().toRotationMatrix();
}

Matrix3d skew_of(const Vector3d &v)
{
    Matrix3d S;
    S << 0.0, -v(2), v(1), v(2), 0.0, -v(0), -v(1), v(0), 0.0;
    return S;
}

Vector4d quat_multiply_local(const Vector4d &a, const Vector4d &b)
{
    const Eigen::Quaterniond ea(a(0), a(1), a(2), a(3));
    const Eigen::Quaterniond eb(b(0), b(1), b(2), b(3));
    const Eigen::Quaterniond e = ea * eb;
    return Vector4d(e.w(), e.x(), e.y(), e.z());
}

Vector4d quat_conjugate(const Vector4d &q)
{
    return Vector4d(q(0), -q(1), -q(2), -q(3));
}

// One RK4 step of q' = 0.5 Omega(gyr - bg) q, p' = v, v' = R(q)(acc - ba) + g_w
// with the biases frozen inside the step (mirrors the spec equations).
void test_rk4_step(const Vector3d &w, const Vector3d &a, const Vector3d &g_w,
                   double dt, Vector4d &q, Vector3d &p, Vector3d &v)
{
    Matrix4d omega = Matrix4d::Zero();
    omega.block<1, 3>(0, 1) = -w.transpose();
    omega.block<3, 1>(1, 0) = w;
    omega.block<3, 3>(1, 1) = -skew_of(w);

    // Explicit return types: an `auto` lambda would deduce an Eigen expression
    // template that dangles once the lambda returns.
    const auto dq_dt = [&omega](const Vector4d &qq) -> Vector4d {
        return 0.5 * (omega * qq);
    };
    const auto dv_dt = [&a, &g_w](const Vector4d &qq) -> Vector3d {
        return rotation_of(qq) * a + g_w;
    };

    const Vector4d k1q = dq_dt(q), k2q = dq_dt(q + 0.5 * dt * k1q),
                   k3q = dq_dt(q + 0.5 * dt * k2q), k4q = dq_dt(q + dt * k3q);
    const Vector3d k1v = dv_dt(q), k2v = dv_dt(q + 0.5 * dt * k1q),
                   k3v = dv_dt(q + 0.5 * dt * k2q), k4v = dv_dt(q + dt * k3q);
    const Vector3d k1p = v, k2p = v + 0.5 * dt * k1v, k3p = v + 0.5 * dt * k2v,
                   k4p = v + dt * k3v;

    q += (dt / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q);
    p += (dt / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);
    v += (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
    q /= q.norm();
}

// Test-local F from the spec, evaluated at the FEJ (= nominal) values.
Mat15 build_F(const Vector3d &w, const Vector3d &a, const Matrix3d &R)
{
    const Matrix3d I3 = Matrix3d::Identity();
    Mat15 F = Mat15::Zero();
    F.block<3, 3>(kImuTheta, kImuTheta) = -skew_of(w);
    F.block<3, 3>(kImuTheta, kImuBg) = -I3;
    F.block<3, 3>(kImuP, kImuV) = I3;
    F.block<3, 3>(kImuV, kImuTheta) = -R * skew_of(a);
    F.block<3, 3>(kImuV, kImuBa) = -R;
    return F;
}

EkfOptions test_options()
{
    EkfOptions options;
    options.gravity = 9.81007;
    options.acc_noise = 0.1;
    options.gyr_noise = 0.01;
    options.acc_bias_noise = 0.001;
    options.gyr_bias_noise = 0.0001;
    options.init_position_sigma = 0.05;
    options.init_velocity_sigma = 0.05;
    options.init_rotation_sigma = 0.01;
    options.init_acc_bias_sigma = 0.1;
    options.init_gyr_bias_sigma = 0.01;
    return options;
}

void check_symmetric(const Eigen::MatrixXd &P, double tol = 1e-12)
{
    EXPECT_LT((P - P.transpose()).cwiseAbs().maxCoeff(), tol);
}

// Smallest eigenvalue of the symmetric part (test-local PSD check).
double min_eigenvalue(const Eigen::MatrixXd &P)
{
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (P + P.transpose()));
    return es.eigenvalues().minCoeff();
}

}  // namespace

// ---------------------------------------------------------------- T1 --------

TEST(EkfStateHelpers, T1RotationMatchesEigenQuaternion)
{
    std::mt19937 rng(1234);
    for (int k = 0; k < 200; ++k)
    {
        const Vector4d q = random_unit_quaternion(rng);
        const Matrix3d R_impl = EkfState::rotation(q);
        const Matrix3d R_eigen = rotation_of(q);
        EXPECT_NEAR((R_impl - R_eigen).cwiseAbs().maxCoeff(), 0.0, 1e-12);
        EXPECT_NEAR(R_impl.determinant(), 1.0, 1e-12);
    }
    // Near-zero quaternion falls back to identity instead of producing NaNs.
    EXPECT_TRUE(EkfState::rotation(Vector4d(0, 0, 0, 0))
                    .isApprox(Matrix3d::Identity(), 1e-12));
}

// ---------------------------------------------------------------- T2 --------

TEST(EkfStateHelpers, T2QuaternionRoundTripsAndNorm)
{
    std::mt19937 rng(4321);
    for (int k = 0; k < 200; ++k)
    {
        const Vector3d axis_angle = random_axis_angle(rng);
        const Vector4d q = EkfState::quaternion_from_axis_angle(axis_angle);
        EXPECT_NEAR(q.norm(), 1.0, 1e-12);
        const Vector3d back = EkfState::quaternion_to_axis_angle(q);
        EXPECT_NEAR((back - axis_angle).norm(), 0.0, 1e-9);
    }
    // Small angles stay finite and round trip through the series branch.
    for (double mag : {0.0, 1e-12, 1e-9, 1e-6})
    {
        const Vector3d axis_angle = Vector3d(1.0, -2.0, 0.5).normalized() * mag;
        const Vector4d q = EkfState::quaternion_from_axis_angle(axis_angle);
        EXPECT_NEAR((EkfState::quaternion_to_axis_angle(q) - axis_angle).norm(), 0.0, 1e-9);
    }

    // Hamilton product against Eigen, and q (x) q* = identity.
    for (int k = 0; k < 100; ++k)
    {
        const Vector4d a = random_unit_quaternion(rng);
        const Vector4d b = random_unit_quaternion(rng);
        EXPECT_NEAR((EkfState::quaternion_multiply(a, b) - quat_multiply_local(a, b)).norm(),
                    0.0, 1e-12);
        EXPECT_NEAR((EkfState::quaternion_multiply(a, quat_conjugate(a)) -
                     Vector4d(1, 0, 0, 0)).norm(),
                    0.0, 1e-12);
    }

    // The mean quaternion must stay a unit quaternion over a long run.
    EkfOptions options = test_options();
    options.gravity = 0.0;
    EkfState state;
    state.reset(options, 0.0, random_unit_quaternion(rng), Vector3d(0.1, 0.2, 0.3),
                Vector3d(0.0, 0.0, 0.0), Vector3d::Zero(), Vector3d::Zero());
    std::normal_distribution<double> normal(0.0, 1.0);
    for (int k = 0; k < 10000; ++k)
    {
        const Vector3d acc(normal(rng), normal(rng), normal(rng));
        const Vector3d gyr(normal(rng), normal(rng), normal(rng));
        state.propagate(acc, gyr, 0.01);
    }
    EXPECT_NEAR(state.imu().q.norm(), 1.0, 1e-8);
}

// ---------------------------------------------------------------- T3 --------

TEST(EkfStateHelpers, T3PoseCorrectionMatchesRightPerturbation)
{
    std::mt19937 rng(99);
    for (int k = 0; k < 50; ++k)
    {
        const Vector4d q = random_unit_quaternion(rng);
        const Vector3d p = Vector3d::Random();
        const Vector3d dtheta = random_axis_angle(rng).normalized() * 1e-5;

        Vec6 dx = Vec6::Zero();
        dx.head<3>() = dtheta;
        Vector4d q_c = q;
        Vector3d p_c = p;
        EkfState::apply_pose_correction(q_c, p_c, dx);

        const Matrix3d expected = rotation_of(q) * (Matrix3d::Identity() + skew_of(dtheta));
        EXPECT_NEAR((EkfState::rotation(q_c) - expected).cwiseAbs().maxCoeff(), 0.0, 1e-6);
        EXPECT_NEAR((p_c - p).norm(), 0.0, 1e-15);
    }
}

// ---------------------------------------------------------------- T4 --------

TEST(EkfStatePropagation, T4RestStaysAtRestAndCovarianceGrows)
{
    EkfOptions options = test_options();
    options.gravity = 0.0;
    EkfState state;
    state.reset(options, 1.0, Vector4d(1, 0, 0, 0), Vector3d::Zero(), Vector3d::Zero(),
                Vector3d::Zero(), Vector3d::Zero());

    Eigen::Matrix<double, kImuStateSize, 1> previous =
        state.covariance().diagonal().head<kImuStateSize>();
    for (int k = 0; k < 100; ++k)
    {
        state.propagate(Vector3d::Zero(), Vector3d::Zero(), 0.01);

        EXPECT_NEAR((state.imu().q - Vector4d(1, 0, 0, 0)).norm(), 0.0, 1e-12);
        EXPECT_NEAR(state.imu().p.norm(), 0.0, 1e-15);
        EXPECT_NEAR(state.imu().v.norm(), 0.0, 1e-15);
        EXPECT_NEAR(state.timestamp(), 1.0 + 0.01 * (k + 1), 1e-12);

        const auto diagonal = state.covariance().diagonal().head<kImuStateSize>();
        for (int i = kImuP; i < kImuV + 3; ++i)
            EXPECT_GE(diagonal(i), previous(i) - 1e-15);
        previous = diagonal;
    }
}

// ---------------------------------------------------------------- T5 --------

TEST(EkfStatePropagation, T5ConstantAccelerationIsIntegratedExactly)
{
    EkfOptions options = test_options();
    options.gravity = 0.0;
    const Vector3d acc(0.3, -0.2, 0.5);
    const double total_time = 1.0;
    const int steps = 100;
    const double dt = total_time / steps;

    EkfState state;
    state.reset(options, 0.0, Vector4d(1, 0, 0, 0), Vector3d::Zero(), Vector3d::Zero(),
                Vector3d::Zero(), Vector3d::Zero());
    for (int k = 0; k < steps; ++k)
        state.propagate(acc, Vector3d::Zero(), dt);

    EXPECT_NEAR((state.imu().v - acc * total_time).norm(), 0.0, 1e-9);
    EXPECT_NEAR((state.imu().p - 0.5 * acc * total_time * total_time).norm(), 0.0, 1e-9);
    EXPECT_NEAR((state.imu().q - Vector4d(1, 0, 0, 0)).norm(), 0.0, 1e-12);
}

// ---------------------------------------------------------------- T6 --------

TEST(EkfStateCovariance, T6SymmetryAndPositiveDefinitenessOverChains)
{
    EkfOptions options = test_options();
    options.window_size = 3;
    EkfState state;
    state.reset(options, 0.0, Vector4d(1, 0, 0, 0), Vector3d::Zero(), Vector3d::Zero(),
                Vector3d::Zero(), Vector3d::Zero());

    std::mt19937 rng(7);
    std::normal_distribution<double> normal(0.0, 0.1);
    double timestamp = 0.0;
    int steps_since_augment = 100;  // no clone yet

    for (int k = 0; k < 60; ++k)
    {
        const Vector3d acc(0.1, -0.05, 9.8 + normal(rng));
        const Vector3d gyr(0.2, -0.1, 0.05);
        state.propagate(acc, gyr, 0.01);
        timestamp += 0.01;
        ++steps_since_augment;

        check_symmetric(state.covariance());

        // A fresh clone is an exact copy of the live pose, so right after
        // augment_clone() the covariance is only PSD: the relative pose error
        // (dp_live - dp_clone) has zero variance and the process noise enters
        // dp only through dv, so the null direction survives one more
        // propagation and is lifted by the second one.  Everywhere else the
        // covariance is positive definite.
        const bool expect_pd = state.num_clones() == 0 || steps_since_augment >= 2;
        if (expect_pd)
            EXPECT_TRUE(EkfState::is_positive_definite(state.covariance()));
        else
            EXPECT_GE(min_eigenvalue(state.covariance()), -1e-9);

        if (k % 5 == 4)
        {
            state.augment_clone(timestamp);
            steps_since_augment = 0;
            state.prune_clone_window();
            check_symmetric(state.covariance());
            EXPECT_LE(state.num_clones(), options.window_size);
            EXPECT_EQ(state.size(), kImuStateSize + kCloneStateSize * state.num_clones());
        }
    }

    // The last iteration may have ended with augment_clone(), which appends a
    // clone that is an *exact* copy of the live pose.  That copy is perfectly
    // correlated with the live pose, so the covariance is only PSD right after
    // the augmentation: the difference dp_live - dp_clone has zero variance
    // until process noise (entering dp through dv) decorrelates the pair.
    // Propagate until that null direction is lifted before testing PD.
    for (int i = steps_since_augment; i < 2; ++i)
        state.propagate(Vector3d(0.1, -0.05, 9.8), Vector3d(0.2, -0.1, 0.05), 0.01);

    // Removing the oldest clone deletes its rows/columns; a PD covariance
    // remains symmetric positive definite.
    ASSERT_GT(state.num_clones(), 0);
    ASSERT_TRUE(EkfState::is_positive_definite(state.covariance()));
    state.marginalize_oldest_clone();
    check_symmetric(state.covariance());
    EXPECT_TRUE(EkfState::is_positive_definite(state.covariance()));
}

// ---------------------------------------------------------------- T7 --------

TEST(EkfStateCloneWindow, T7AugmentCloneCopiesTheLivePose)
{
    EkfOptions options = test_options();
    options.window_size = 8;
    EkfState state;
    state.reset(options, 0.0, Vector4d(1, 0, 0, 0), Vector3d::Zero(), Vector3d::Zero(),
                Vector3d::Zero(), Vector3d::Zero());

    double timestamp = 0.0;
    for (int k = 0; k < 3; ++k)
    {
        for (int s = 0; s < 4; ++s)
        {
            state.propagate(Vector3d(0.1, 0.0, 9.8), Vector3d(0.1, 0.05, -0.02), 0.01);
            timestamp += 0.01;
        }
        const int n_before = state.size();
        const Eigen::MatrixXd P_before = state.covariance();
        state.augment_clone(timestamp);

        ASSERT_EQ(state.num_clones(), k + 1);
        EXPECT_EQ(state.size(), n_before + kCloneStateSize);
        EXPECT_EQ(state.timestamp(), timestamp);
        EXPECT_EQ(state.clone_timestamp(k), timestamp);

        // The new clone is a frozen copy of the live pose (mean and FEJ).
        EXPECT_NEAR((state.clone(k).q - state.imu().q).norm(), 0.0, 1e-15);
        EXPECT_NEAR((state.clone(k).p - state.imu().p).norm(), 0.0, 1e-15);
        EXPECT_NEAR((state.clone_fej(k).q - state.imu_fej().q).norm(), 0.0, 1e-15);
        EXPECT_NEAR((state.clone_fej(k).p - state.imu_fej().p).norm(), 0.0, 1e-15);

        // Its covariance block (and the cross terms) equal the live pose block.
        const int b = state.clone_offset(k);
        const Eigen::MatrixXd &P = state.covariance();
        EXPECT_LT((P.block(b, b, kCloneStateSize, kCloneStateSize) -
                   P.topLeftCorner(kCloneStateSize, kCloneStateSize))
                      .cwiseAbs()
                      .maxCoeff(),
                  1e-15);
        EXPECT_LT((P.block(b, 0, kCloneStateSize, n_before) - P_before.topRows(kCloneStateSize))
                      .cwiseAbs()
                      .maxCoeff(),
                  1e-15);
        EXPECT_LT((P.block(0, b, n_before, kCloneStateSize) -
                   P_before.leftCols(kCloneStateSize))
                      .cwiseAbs()
                      .maxCoeff(),
                  1e-15);
    }

    // augment_clone() only appends; pruning is an explicit post-update step.
    options.window_size = 2;
    EkfState trimmed;
    trimmed.reset(options, 0.0, Vector4d(1, 0, 0, 0), Vector3d::Zero(), Vector3d::Zero(),
                  Vector3d::Zero(), Vector3d::Zero());
    for (int k = 0; k < 10; ++k)
    {
        trimmed.propagate(Vector3d(0.0, 0.0, 9.8), Vector3d(0.1, 0.0, 0.0), 0.01);
        trimmed.augment_clone(0.01 * (k + 1));
        // Pruning happens after every augment below, so the pre-prune size is
        // the previous (already bounded) window plus this new clone.
        EXPECT_EQ(trimmed.num_clones(), std::min(k + 1, 3));
        EXPECT_EQ(trimmed.size(),
                  kImuStateSize + kCloneStateSize * trimmed.num_clones());
        const int removed = trimmed.prune_clone_window();
        EXPECT_EQ(removed, k >= 2 ? 1 : 0);
        EXPECT_EQ(trimmed.num_clones(), std::min(k + 1, 2));
    }
    EXPECT_EQ(trimmed.num_clones(), 2);
}

// ---------------------------------------------------------------- T8 --------

TEST(EkfStateCloneWindow, T8MarginalizeOldestDeletesRowsAndColumns)
{
    EkfOptions options = test_options();
    options.window_size = 4;
    EkfState state;
    state.reset(options, 0.0, Vector4d(1, 0, 0, 0), Vector3d::Zero(), Vector3d::Zero(),
                Vector3d::Zero(), Vector3d::Zero());

    const Vector3d acc(0.2, -0.1, 9.8);
    const Vector3d gyr(0.3, -0.2, 0.15);
    double timestamp = 0.0;
    for (int k = 0; k < 3; ++k)
    {
        state.propagate(acc, gyr, 0.01);
        timestamp += 0.01;
    }
    state.augment_clone(timestamp);
    for (int k = 0; k < 3; ++k)
    {
        state.propagate(acc, gyr, 0.01);
        timestamp += 0.01;
    }
    state.augment_clone(timestamp);
    for (int k = 0; k < 3; ++k)
    {
        state.propagate(acc, gyr, 0.01);
        timestamp += 0.01;
    }

    ASSERT_EQ(state.num_clones(), 2);
    const int n = state.size();
    const int b = state.clone_offset(0);
    const int m = n - kCloneStateSize;
    const Eigen::MatrixXd P = state.covariance();

    // OpenVINS/SchurVINS remove the clone by direct row/column deletion.
    Eigen::VectorXi keep(m);
    int k = 0;
    for (int i = 0; i < n; ++i)
        if (i < b || i >= b + kCloneStateSize)
            keep(k++) = i;
    const Eigen::MatrixXd expected = P(keep, keep);

    const CloneNominal retained_clone = state.clone(1);
    const CloneNominal retained_clone_fej = state.clone_fej(1);
    const double retained_timestamp = state.clone_timestamp(1);

    state.marginalize_oldest_clone();

    ASSERT_EQ(state.size(), n - kCloneStateSize);
    ASSERT_EQ(state.num_clones(), 1);
    EXPECT_LT((state.covariance() - expected).cwiseAbs().maxCoeff(), 1e-9);
    check_symmetric(state.covariance());
    EXPECT_TRUE(EkfState::is_positive_definite(state.covariance()));

    // Remaining clone shifted down to index 0 and keeps its values.
    EXPECT_NEAR((state.clone(0).q - retained_clone.q).norm(), 0.0, 1e-15);
    EXPECT_NEAR((state.clone(0).p - retained_clone.p).norm(), 0.0, 1e-15);
    EXPECT_NEAR((state.clone_fej(0).q - retained_clone_fej.q).norm(), 0.0, 1e-15);
    EXPECT_NEAR((state.clone_fej(0).p - retained_clone_fej.p).norm(), 0.0, 1e-15);
    EXPECT_EQ(state.clone_timestamp(0), retained_timestamp);

    // Dropping the last clone leaves the live block; further calls are no-ops.
    state.marginalize_oldest_clone();
    ASSERT_EQ(state.num_clones(), 0);
    EXPECT_EQ(state.size(), kImuStateSize);
    const Eigen::MatrixXd live_only = state.covariance();
    state.marginalize_oldest_clone();
    EXPECT_EQ(state.size(), kImuStateSize);
    EXPECT_EQ(state.covariance(), live_only);
}

// ---------------------------------------------------------------- T9 --------

TEST(EkfStatePropagation, T9FiniteDifferenceMatchesAnalyticPhi)
{
    EkfOptions options = test_options();
    options.gravity = 9.81007;

    std::mt19937 rng(2024);
    const Vector4d q0 = EkfState::quaternion_from_axis_angle(Vector3d(0.2, -0.3, 0.15));
    const Vector3d p0(0.4, -1.2, 0.7);
    const Vector3d v0(0.3, 0.1, -0.2);
    const Vector3d ba(0.02, -0.01, 0.03);
    const Vector3d bg(0.001, -0.002, 0.0015);
    const Vector3d acc(0.5, -0.25, 9.9);
    const Vector3d gyr(0.35, -0.2, 0.45);
    const double dt = 0.01;
    const Vector3d g_w(0.0, 0.0, -options.gravity);

    // Analytic Phi from the spec (test-local F).
    const Vector3d w = gyr - bg;
    const Vector3d a = acc - ba;
    const Mat15 F = build_F(w, a, rotation_of(q0));
    const Mat15 Phi = Mat15::Identity() + F * dt + 0.5 * (F * F) * (dt * dt) +
                      (1.0 / 6.0) * (F * F * F) * (dt * dt * dt);

    // Nominal integration with the test-local RK4 (same equations as propagate).
    const auto integrate = [&](const Vector4d &q_in, const Vector3d &p_in,
                               const Vector3d &v_in, const Vector3d &ba_in,
                               const Vector3d &bg_in, Vector4d &q_out,
                               Vector3d &p_out, Vector3d &v_out) {
        q_out = q_in;
        p_out = p_in;
        v_out = v_in;
        test_rk4_step(gyr - bg_in, acc - ba_in, g_w, dt, q_out, p_out, v_out);
    };

    Vector4d q_base, q_plus, q_minus;
    Vector3d p_base, p_plus, p_minus, v_base, v_plus, v_minus;
    integrate(q0, p0, v0, ba, bg, q_base, p_base, v_base);

    const double eps = 1e-7;
    Mat15 phi_fd = Mat15::Zero();
    for (int i = 0; i < kImuStateSize; ++i)
    {
        Vec15 step = Vec15::Zero();
        Vec15 delta_plus = Vec15::Zero();
        Vec15 delta_minus = Vec15::Zero();
        for (double sign : {1.0, -1.0})
        {
            step(i) = sign * eps;
            Vector4d q_p = q0;
            Vector3d p_p = p0;
            // Rotation uses the right (body frame) perturbation; the remaining
            // components are additive.
            EkfState::apply_pose_correction(q_p, p_p, step.head<6>());
            const Vector3d v_p = v0 + step.segment<3>(kImuV);
            const Vector3d ba_p = ba + step.segment<3>(kImuBa);
            const Vector3d bg_p = bg + step.segment<3>(kImuBg);

            Vector4d q_i;
            Vector3d p_i, v_i;
            integrate(q_p, p_p, v_p, ba_p, bg_p, q_i, p_i, v_i);

            // Error state of the perturbed trajectory w.r.t. the nominal one,
            // using the same right perturbation convention.
            const Vector4d dq =
                quat_multiply_local(quat_conjugate(q_base), q_i);
            Vec15 delta = Vec15::Zero();
            delta.segment<3>(kImuTheta) = EkfState::quaternion_to_axis_angle(dq);
            delta.segment<3>(kImuP) = p_i - p_base;
            delta.segment<3>(kImuV) = v_i - v_base;
            delta.segment<3>(kImuBa) = ba_p - ba;
            delta.segment<3>(kImuBg) = bg_p - bg;
            if (sign > 0.0)
                delta_plus = delta;
            else
                delta_minus = delta;
        }
        phi_fd.col(i) = (delta_plus - delta_minus) / (2.0 * eps);
    }

    double max_error = 0.0;
    for (int r = 0; r < kImuStateSize; ++r)
        for (int c = 0; c < kImuStateSize; ++c)
            max_error = std::max(max_error, std::abs(phi_fd(r, c) - Phi(r, c)));
    std::cout << "[T9] max |Phi_fd - Phi| = " << max_error << std::endl;
    EXPECT_LT(max_error, 5e-3);

    // The implementation's own RK4 must reproduce the test-local integration,
    // otherwise the finite-difference comparison above would be meaningless.
    EkfState state;
    state.reset(options, 5.0, q0, p0, v0, ba, bg);
    state.propagate(acc, gyr, dt);
    EXPECT_NEAR((state.imu().q - q_base).norm(), 0.0, 1e-12);
    EXPECT_NEAR((state.imu().p - p_base).norm(), 0.0, 1e-12);
    EXPECT_NEAR((state.imu().v - v_base).norm(), 0.0, 1e-12);
    EXPECT_NEAR(state.timestamp(), 5.0 + dt, 1e-12);
    // Propagation advances the live FEJ to the new mean (OpenVINS behavior).
    EXPECT_NEAR((state.imu_fej().q - state.imu().q).norm(), 0.0, 1e-15);
    EXPECT_NEAR((state.imu_fej().p - state.imu().p).norm(), 0.0, 1e-15);
    EXPECT_NEAR((state.imu_fej().v - state.imu().v).norm(), 0.0, 1e-15);
    EXPECT_NEAR((state.imu_fej().ba - state.imu().ba).norm(), 0.0, 1e-15);
    EXPECT_NEAR((state.imu_fej().bg - state.imu().bg).norm(), 0.0, 1e-15);
}

// --------------------------------------------------------------- T10 --------

TEST(EkfStatePropagation, T10PropagationRefreshesImuFejButFreezesCloneFej)
{
    EkfOptions options = test_options();
    options.window_size = 4;
    EkfState state;
    state.reset(options, 0.0, Vector4d(1, 0, 0, 0), Vector3d::Zero(), Vector3d::Zero(),
                Vector3d::Zero(), Vector3d::Zero());

    state.propagate(Vector3d(0.1, -0.2, 9.8), Vector3d(0.2, -0.1, 0.05), 0.01);
    state.augment_clone(0.01);
    const CloneNominal frozen_clone = state.clone(0);
    const CloneNominal frozen_clone_fej = state.clone_fej(0);

    state.propagate(Vector3d(0.2, 0.1, 9.9), Vector3d(-0.1, 0.2, 0.03), 0.01);

    EXPECT_NEAR((state.imu_fej().q - state.imu().q).norm(), 0.0, 1e-15);
    EXPECT_NEAR((state.imu_fej().p - state.imu().p).norm(), 0.0, 1e-15);
    EXPECT_NEAR((state.imu_fej().v - state.imu().v).norm(), 0.0, 1e-15);
    EXPECT_NEAR((state.clone(0).q - frozen_clone.q).norm(), 0.0, 1e-15);
    EXPECT_NEAR((state.clone(0).p - frozen_clone.p).norm(), 0.0, 1e-15);
    EXPECT_NEAR((state.clone_fej(0).q - frozen_clone_fej.q).norm(), 0.0, 1e-15);
    EXPECT_NEAR((state.clone_fej(0).p - frozen_clone_fej.p).norm(), 0.0, 1e-15);
}

// --------------------------------------------------------------- T11 --------

TEST(EkfStateCovariance, T11PositiveSemidefiniteCheckAllowsSingularCovariance)
{
    Eigen::MatrixXd psd(4, 4);
    psd << 1.0, 1.0, 0.0, 0.0,
           1.0, 1.0, 0.0, 0.0,
           0.0, 0.0, 2.0, 0.5,
           0.0, 0.0, 0.5, 1.0;
    EXPECT_TRUE(EkfState::is_positive_semidefinite(psd));
    EXPECT_FALSE(EkfState::is_positive_definite(psd));

    Eigen::MatrixXd indefinite = psd;
    indefinite(2, 2) = -1e-2;
    EXPECT_FALSE(EkfState::is_positive_semidefinite(indefinite));
}

// ----------------------------------------------- T12-T14: starvation monitor --

// Reopen the anonymous namespace for a file-local helper.
namespace
{
// One propagate() step from an identical seed, with a configurable noise scale.
Eigen::MatrixXd propagate_one_step(double noise_scale)
{
    EkfOptions options = test_options();
    options.window_size = 2;
    EkfState state;
    state.reset(options, 0.0, Vector4d(1, 0, 0, 0), Vector3d(0.1, 0.2, 0.3),
                Vector3d(1.0, 0.0, 0.0), Vector3d(0.01, 0.02, 0.03),
                Vector3d(0.001, 0.002, 0.003));
    state.propagate(Vector3d(0.1, -0.05, 9.8), Vector3d(0.2, -0.1, 0.05), 0.01,
                    noise_scale);
    return state.covariance();
}
}  // namespace

TEST(StarvationMonitor, T12TimeCriterionUsesTimeNotFrameCount)
{
    StarvationMonitor monitor;
    monitor.configure({1.0, 0.25});
    monitor.reset(100.0);

    // 30 failed frames but only 0.9375 s of sensor time (all steps are exact
    // binary fractions): the frame *count* is large while the *time* span stays
    // below the 1.0 s threshold, so the time gate must not fire. (The old
    // frame-count guard would have needed 20 frames.)
    for (int i = 0; i < 30; ++i)
    {
        const StarvationMonitor::Decision d =
            monitor.observe_failed_frame(100.0 + 0.03125 * (i + 1));
        EXPECT_FALSE(d.restart);
        EXPECT_FALSE(d.rebaselined);
    }
    EXPECT_EQ(monitor.time_gate_triggers(), 0);
    // 0.9375 s > starvation_seconds (0.25) -> process-noise inflation is on.
    EXPECT_TRUE(monitor.starvation_active());

    // A frame whose gap exceeds the time threshold fires exactly once.
    const StarvationMonitor::Decision fired = monitor.observe_failed_frame(101.5);
    EXPECT_TRUE(fired.restart);
    EXPECT_FALSE(fired.rebaselined);
    EXPECT_DOUBLE_EQ(fired.gap_seconds, 1.5);
    EXPECT_EQ(monitor.time_gate_triggers(), 1);

    // A successful update re-baselines the monitor and clears starvation.
    monitor.note_applied_update(101.5);
    EXPECT_FALSE(monitor.starvation_active());
    EXPECT_DOUBLE_EQ(monitor.last_update_time(), 101.5);

    // A small forward gap afterwards does not fire and leaves inflation off.
    const StarvationMonitor::Decision small = monitor.observe_failed_frame(101.5625);
    EXPECT_FALSE(small.restart);
    EXPECT_FALSE(monitor.starvation_active());

    // A gap exactly at the threshold (strict '>') must not fire either.
    const StarvationMonitor::Decision edge = monitor.observe_failed_frame(102.5);
    EXPECT_FALSE(edge.restart);
    EXPECT_DOUBLE_EQ(edge.gap_seconds, 1.0);
    EXPECT_EQ(monitor.time_gate_triggers(), 1);
}

TEST(StarvationMonitor, T13BackwardsTimestampRebaselinesInsteadOfRestarting)
{
    StarvationMonitor monitor;
    monitor.configure({1.0, 0.25});
    monitor.reset(500.0);

    // Starve the monitor first (0.5 s gap > starvation_seconds) so we can prove
    // that a rollback also clears the noise-inflation flag.
    const StarvationMonitor::Decision starved = monitor.observe_failed_frame(500.5);
    EXPECT_FALSE(starved.restart);
    EXPECT_TRUE(monitor.starvation_active());

    // A large backwards jump (bag replay / clock rollback) must re-baseline the
    // monitor, never schedule a restart, and reset the starvation state to the
    // new (earlier) baseline.
    const StarvationMonitor::Decision back = monitor.observe_failed_frame(400.0);
    EXPECT_TRUE(back.rebaselined);
    EXPECT_FALSE(back.restart);
    EXPECT_FALSE(monitor.starvation_active());
    EXPECT_EQ(monitor.time_gate_triggers(), 0);
    EXPECT_DOUBLE_EQ(monitor.last_update_time(), 400.0);

    // After re-baselining, a normal forward gap behaves as usual.
    const StarvationMonitor::Decision forward = monitor.observe_failed_frame(400.125);
    EXPECT_FALSE(forward.restart);
    EXPECT_FALSE(forward.rebaselined);
    EXPECT_DOUBLE_EQ(forward.gap_seconds, 0.125);
}

TEST(StarvationMonitor, T14DisabledCriteriaNeverFire)
{
    StarvationMonitor monitor;
    monitor.configure({0.0, 0.0});  // both criteria disabled
    monitor.reset(0.0);
    const StarvationMonitor::Decision d = monitor.observe_failed_frame(1000.0);
    EXPECT_FALSE(d.restart);
    EXPECT_FALSE(monitor.starvation_active());
    EXPECT_EQ(monitor.time_gate_triggers(), 0);
}

// ------------------------------------------ T15-T16: starvation noise scale --

TEST(EkfStatePropagation, T15StarvationNoiseScaleInflatesQLinearly)
{
    const Eigen::MatrixXd p1 = propagate_one_step(1.0);
    const Eigen::MatrixXd p2 = propagate_one_step(2.0);
    const Eigen::MatrixXd p3 = propagate_one_step(3.0);

    // P(scale) = A + scale * Qd with the same A = Phi P0 Phi^T, so the increment
    // over the neutral step is linear in the scale: (P3 - P1) == 2 * (P2 - P1).
    const Eigen::MatrixXd d1 = p2 - p1;
    const Eigen::MatrixXd d2 = p3 - p1;
    const double tol = 1e-12 * (1.0 + d2.cwiseAbs().maxCoeff());
    EXPECT_LT((d2 - 2.0 * d1).cwiseAbs().maxCoeff(), tol);
    EXPECT_GT(d2.trace(), 0.0);

    // The extra noise is a PSD block, and the scaled covariance stays symmetric
    // and positive definite (no asymmetric / non-finite result).
    EXPECT_GE(min_eigenvalue(d1), -1e-18);
    EXPECT_GE(min_eigenvalue(d2), -1e-18);
    check_symmetric(p3);
    EXPECT_TRUE(EkfState::is_positive_definite(p3));
    EXPECT_TRUE(p3.allFinite());
}

TEST(EkfStatePropagation, T16NoiseScaleGuardNeutralisesBadValues)
{
    const Eigen::MatrixXd neutral = propagate_one_step(1.0);
    // A scale < 1 or non-finite would shrink/poison Q; the implementation
    // neutralises it to 1.0, so the covariance is bit-identical to the neutral
    // step (this is what "recovered" propagation must look like).
    for (double bad : {0.5, 0.0, -3.0,
                       std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity()})
    {
        EXPECT_EQ((propagate_one_step(bad) - neutral).cwiseAbs().maxCoeff(), 0.0)
            << "bad noise scale " << bad;
    }
}


// ------------------------------------------------ T17-T22: deep + cooldown --

namespace
{
// Explicit config so the tests do not depend on the aggregate-init field order.
StarvationMonitor::Config starvation_config(double max_no_update_seconds,
                                            double starvation_seconds,
                                            double noise_scale,
                                            double deep_starvation_seconds,
                                            double deep_noise_scale,
                                            double restart_cooldown_seconds)
{
    StarvationMonitor::Config config;
    config.max_no_update_seconds = max_no_update_seconds;
    config.starvation_seconds = starvation_seconds;
    config.noise_scale = noise_scale;
    config.deep_starvation_seconds = deep_starvation_seconds;
    config.deep_noise_scale = deep_noise_scale;
    config.restart_cooldown_seconds = restart_cooldown_seconds;
    return config;
}
}  // namespace

TEST(StarvationMonitor, T17DeepStarvationEscalatesTheNoiseScale)
{
    StarvationMonitor monitor;
    // Time gate pushed far away so this test isolates the inflation stages;
    // cooldown off so nothing is suppressed.
    monitor.configure(starvation_config(100.0, 0.25, 3.0, 1.0, 10.0, 0.0));
    monitor.reset(0.0);

    // Below starvation_seconds: neutral.
    EXPECT_EQ(monitor.observe_failed_frame(0.2).deep_escalated, false);
    EXPECT_FALSE(monitor.starvation_active());
    EXPECT_DOUBLE_EQ(monitor.noise_scale(), 1.0);

    // Between starvation_seconds and deep_starvation_seconds: base scale.
    EXPECT_EQ(monitor.observe_failed_frame(0.5).deep_escalated, false);
    EXPECT_TRUE(monitor.starvation_active());
    EXPECT_FALSE(monitor.deep_starvation_active());
    EXPECT_DOUBLE_EQ(monitor.noise_scale(), 3.0);
    EXPECT_EQ(monitor.deep_escalations(), 0);

    // Exactly at deep_starvation_seconds: strict '>' keeps it at the base scale.
    EXPECT_EQ(monitor.observe_failed_frame(1.0).deep_escalated, false);
    EXPECT_DOUBLE_EQ(monitor.noise_scale(), 3.0);
    EXPECT_EQ(monitor.deep_escalations(), 0);

    // Beyond deep_starvation_seconds: escalated, reported exactly once.
    const StarvationMonitor::Decision deep = monitor.observe_failed_frame(1.5);
    EXPECT_TRUE(deep.deep_escalated);
    EXPECT_TRUE(monitor.deep_starvation_active());
    EXPECT_DOUBLE_EQ(monitor.noise_scale(), 10.0);
    EXPECT_EQ(monitor.deep_escalations(), 1);

    // Still deep on the next starved frame, but not a *new* escalation.
    EXPECT_FALSE(monitor.observe_failed_frame(1.6).deep_escalated);
    EXPECT_DOUBLE_EQ(monitor.noise_scale(), 10.0);
    EXPECT_EQ(monitor.deep_escalations(), 1);

    // A successful update resets everything back to 1.0 immediately.
    monitor.note_applied_update(1.6);
    EXPECT_FALSE(monitor.starvation_active());
    EXPECT_FALSE(monitor.deep_starvation_active());
    EXPECT_DOUBLE_EQ(monitor.noise_scale(), 1.0);
    // 1.7 - 1.6 is not exact in binary, so compare with a tolerance.
    EXPECT_NEAR(monitor.observe_failed_frame(1.7).gap_seconds, 0.1, 1e-12);
    EXPECT_DOUBLE_EQ(monitor.noise_scale(), 1.0);
}

TEST(StarvationMonitor, T18DeepEscalationCanBeDisabled)
{
    StarvationMonitor monitor;
    // deep_starvation_seconds <= 0 disables the escalation; the base scale still
    // applies for an arbitrarily long starvation.
    monitor.configure(starvation_config(1000.0, 0.25, 3.0, 0.0, 10.0, 0.0));
    monitor.reset(0.0);

    const StarvationMonitor::Decision deep = monitor.observe_failed_frame(50.0);
    EXPECT_FALSE(deep.deep_escalated);
    EXPECT_FALSE(monitor.deep_starvation_active());
    EXPECT_TRUE(monitor.starvation_active());
    EXPECT_DOUBLE_EQ(monitor.noise_scale(), 3.0);
    EXPECT_EQ(monitor.deep_escalations(), 0);
}

TEST(StarvationMonitor, T19RestartCooldownSuppressesTheTimeCriterion)
{
    StarvationMonitor monitor;
    monitor.configure(starvation_config(1.0, 0.25, 3.0, 1.0, 10.0, 8.0));
    monitor.reset(0.0);

    // First starvation stretch fires the time criterion normally.
    const StarvationMonitor::Decision fired = monitor.observe_failed_frame(1.5);
    EXPECT_TRUE(fired.restart);
    EXPECT_FALSE(fired.cooldown_suppressed);
    EXPECT_EQ(monitor.time_gate_triggers(), 1);
    monitor.note_restart(1.5);

    // Inside the 8 s window the trigger is real but must not restart.
    const StarvationMonitor::Decision held = monitor.observe_failed_frame(6.5);
    EXPECT_FALSE(held.restart);
    EXPECT_TRUE(held.cooldown_suppressed);
    EXPECT_DOUBLE_EQ(held.gap_seconds, 6.5);
    EXPECT_EQ(monitor.time_gate_triggers(), 1);
    EXPECT_EQ(monitor.cooldown_suppressions(), 1);
    EXPECT_TRUE(monitor.cooldown_active(1.5));
    EXPECT_TRUE(monitor.cooldown_active(9.4));   // 7.9 s < 8 s
    EXPECT_FALSE(monitor.cooldown_active(9.5));  // 8.0 s is not < 8 s

    // Once the cooldown has elapsed a restart fires again.
    const StarvationMonitor::Decision again = monitor.observe_failed_frame(9.5);
    EXPECT_TRUE(again.restart);
    EXPECT_FALSE(again.cooldown_suppressed);
    EXPECT_EQ(monitor.time_gate_triggers(), 2);
    EXPECT_EQ(monitor.cooldown_suppressions(), 1);
}

TEST(StarvationMonitor, T20CooldownSurvivesResetButNotClockRollback)
{
    StarvationMonitor monitor;
    monitor.configure(starvation_config(1.0, 0.25, 3.0, 1.0, 10.0, 8.0));
    monitor.reset(0.0);

    EXPECT_TRUE(monitor.observe_failed_frame(1.5).restart);
    monitor.note_restart(1.5);

    // A controlled restart runs clearState() and initSchurBackend(), both of
    // which call reset(); the anchor must survive that re-seed or the cooldown
    // could never prevent a restart storm.
    monitor.reset(1.6);
    EXPECT_TRUE(monitor.cooldown_active(1.6));
    const StarvationMonitor::Decision held = monitor.observe_failed_frame(3.0);
    EXPECT_FALSE(held.restart);
    EXPECT_TRUE(held.cooldown_suppressed);
    // reset() is a re-seed, not a counter reset: the cumulative diagnostic
    // counters (and the cooldown anchor) survive it (see T23).
    EXPECT_EQ(monitor.time_gate_triggers(), 1);
    EXPECT_EQ(monitor.cooldown_suppressions(), 1);

    // A backwards timestamp (bag replay) drops the anchor: keeping it would keep
    // the cooldown active for the whole replay.
    StarvationMonitor replay;
    replay.configure(starvation_config(1.0, 0.25, 3.0, 1.0, 10.0, 8.0));
    replay.reset(500.0);
    replay.note_restart(500.0);
    EXPECT_TRUE(replay.cooldown_active(501.0));
    const StarvationMonitor::Decision rollback = replay.observe_failed_frame(100.0);
    EXPECT_TRUE(rollback.rebaselined);
    EXPECT_FALSE(rollback.restart);
    EXPECT_FALSE(replay.cooldown_active(100.0));
    EXPECT_DOUBLE_EQ(replay.last_update_time(), 100.0);
}

TEST(StarvationMonitor, T21CooldownCanBeDisabled)
{
    StarvationMonitor monitor;
    // restart_cooldown_seconds == 0 disables the cooldown: every trigger fires,
    // i.e. the pre-P4 behaviour is preserved.
    monitor.configure(starvation_config(1.0, 0.25, 3.0, 1.0, 10.0, 0.0));
    monitor.reset(0.0);

    EXPECT_TRUE(monitor.observe_failed_frame(1.5).restart);
    monitor.note_restart(1.5);
    EXPECT_FALSE(monitor.cooldown_active(1.5));
    const StarvationMonitor::Decision second = monitor.observe_failed_frame(3.0);
    EXPECT_TRUE(second.restart);
    EXPECT_FALSE(second.cooldown_suppressed);
    EXPECT_EQ(monitor.time_gate_triggers(), 2);
    EXPECT_EQ(monitor.cooldown_suppressions(), 0);
}

TEST(StarvationMonitor, T22FrameCountBackstopIsCooldownSuppressed)
{
    // Pure-logic mirror of the decision rule in Estimator::schurProcessFrame:
    // the frame-count backstop restarts only when the streak reaches the limit
    // AND the monitor's cooldown is inactive. (The Estimator itself needs ROS, so
    // the rule is locked down here against the monitor's public predicate.)
    const auto backstop_would_restart = [](const StarvationMonitor &monitor,
                                           double timestamp, int streak, int limit) {
        return !monitor.cooldown_active(timestamp) && streak >= limit;
    };

    StarvationMonitor monitor;
    // Time criterion disabled so only the frame-count backstop is in play.
    monitor.configure(starvation_config(0.0, 0.25, 3.0, 1.0, 10.0, 8.0));
    monitor.reset(0.0);
    monitor.note_restart(0.0);

    // 60 consecutive failures inside the cooldown: the backstop is held back.
    EXPECT_FALSE(backstop_would_restart(monitor, 1.0, 60, 60));
    EXPECT_FALSE(backstop_would_restart(monitor, 7.9, 120, 60));
    // ... and it is the cooldown, not the streak, that holds it back.
    EXPECT_FALSE(backstop_would_restart(monitor, 1.0, 59, 60));
    // Past the cooldown the backstop fires again.
    EXPECT_TRUE(backstop_would_restart(monitor, 9.0, 60, 60));

    // The time criterion never fires here (it is disabled), so the frame-count
    // path cannot be confused with a suppression of the time criterion.
    for (double t : {1.0, 2.0, 3.0})
    {
        const StarvationMonitor::Decision d = monitor.observe_failed_frame(t);
        EXPECT_FALSE(d.restart);
        EXPECT_FALSE(d.cooldown_suppressed);
    }
    EXPECT_EQ(monitor.cooldown_suppressions(), 0);
}

// ------------------------------------- T23-T24: cumulative counters + P5 ----

TEST(StarvationMonitor, T23DiagnosticCountersAccumulateAcrossResets)
{
    // A controlled restart runs clearState() and initSchurBackend(), both of
    // which call reset(). The end-of-run schurReportStarvation() summary must
    // still see the triggers that happened before that re-seed, so reset() only
    // re-baselines the starvation window and leaves the cumulative counters
    // alone. The three counters are exercised in separate monitors so each
    // expectation is an exact, easy-to-read count.

    // Time-gate triggers accumulate across resets (cooldown disabled so every
    // trigger fires immediately).
    {
        StarvationMonitor monitor;
        monitor.configure(starvation_config(1.0, 0.25, 3.0, 1.0, 10.0, 0.0));
        monitor.reset(0.0);
        EXPECT_TRUE(monitor.observe_failed_frame(1.5).restart);
        monitor.reset(10.0);
        EXPECT_EQ(monitor.time_gate_triggers(), 1);
        EXPECT_TRUE(monitor.observe_failed_frame(11.5).restart);
        monitor.reset(20.0);
        EXPECT_EQ(monitor.time_gate_triggers(), 2);
    }

    // Deep escalations accumulate across resets (time criterion pushed away so
    // this monitor only counts the deep stage).
    {
        StarvationMonitor monitor;
        monitor.configure(starvation_config(100.0, 0.25, 3.0, 1.0, 10.0, 0.0));
        monitor.reset(0.0);
        EXPECT_TRUE(monitor.observe_failed_frame(1.5).deep_escalated);
        monitor.reset(10.0);
        EXPECT_EQ(monitor.deep_escalations(), 1);
        EXPECT_TRUE(monitor.observe_failed_frame(11.5).deep_escalated);
        monitor.reset(20.0);
        EXPECT_EQ(monitor.deep_escalations(), 2);
    }

    // Cooldown suppressions accumulate across resets too. deep_starvation is
    // pushed away so the suppressed frame does not also bump the deep counter.
    {
        StarvationMonitor monitor;
        monitor.configure(starvation_config(1.0, 0.25, 3.0, 100.0, 10.0, 8.0));
        monitor.reset(0.0);
        EXPECT_TRUE(monitor.observe_failed_frame(1.5).restart);
        monitor.note_restart(1.5);
        EXPECT_TRUE(monitor.observe_failed_frame(4.0).cooldown_suppressed);
        monitor.reset(4.0);
        EXPECT_EQ(monitor.cooldown_suppressions(), 1);
        EXPECT_EQ(monitor.time_gate_triggers(), 1);
        // The 8 s cooldown from 1.5 s ends at 9.5 s, so 9.6 s fires again.
        EXPECT_TRUE(monitor.observe_failed_frame(9.6).restart);
        monitor.note_restart(9.6);
        EXPECT_TRUE(monitor.observe_failed_frame(12.0).cooldown_suppressed);
        monitor.reset(20.0);
        EXPECT_EQ(monitor.cooldown_suppressions(), 2);
        EXPECT_EQ(monitor.time_gate_triggers(), 2);
    }
}

TEST(SoftRestartEscalation, T24EscalationDueAfterMaxSecondsWithoutUpdate)
{
    // Pure-logic mirror of the P5 escalation rule used by
    // Estimator::schurProcessFrame: a soft restart escalates once it has gone
    // strictly longer than max_seconds without an applied visual update.

    // No pending soft restart (negative sentinel): never escalates.
    EXPECT_FALSE(softRestartEscalationDue(100.0, -1.0, 10.0));

    // Exactly at the threshold: the strict '>' keeps it from escalating.
    EXPECT_FALSE(softRestartEscalationDue(110.0, 100.0, 10.0));
    // Just past the threshold: escalate.
    EXPECT_TRUE(softRestartEscalationDue(110.001, 100.0, 10.0));

    // max_seconds <= 0 disables the escalation.
    EXPECT_FALSE(softRestartEscalationDue(1000.0, 100.0, 0.0));
    EXPECT_FALSE(softRestartEscalationDue(1000.0, 100.0, -5.0));

    // Backwards timestamp (bag replay / clock rollback): no escalation.
    EXPECT_FALSE(softRestartEscalationDue(50.0, 100.0, 10.0));

    // A successfully applied visual update clears the anchor in the Estimator;
    // this is modelled here by the negative sentinel again.
    EXPECT_FALSE(softRestartEscalationDue(5000.0, -1.0, 10.0));
}

TEST(EkfStateCloneWindow, T25SoftRestartKeepsMotionStateAndDropsClones)
{
    // Mirrors Estimator::schurSoftRestart(): the state is re-reset from its own
    // current IMU nominal values. The motion state must survive while the clone
    // window and the old covariance are discarded and P is re-inflated from the
    // configured init_*_sigma values.
    EkfOptions options = test_options();
    options.window_size = 5;
    options.init_velocity_sigma = 0.5;

    EkfState state;
    state.reset(options, 0.0, Vector4d(1, 0, 0, 0), Vector3d::Zero(), Vector3d::Zero(),
                Vector3d::Zero(), Vector3d::Zero());
    for (int k = 0; k < 2; ++k)
    {
        for (int s = 0; s < 5; ++s)
            state.propagate(Vector3d(0.2, 0.0, 9.8), Vector3d(0.0, 0.1, 0.0), 0.01);
        state.augment_clone(0.05 * (k + 1));
    }
    ASSERT_EQ(state.num_clones(), 2);
    const ImuNominal before = state.imu();
    // The propagation must have moved the nominal state before the re-seed.
    EXPECT_GT(before.v.norm(), 0.0);
    EXPECT_GT(before.q.head<3>().norm(), 0.0);

    // Re-seed exactly as schurSoftRestart() does.
    state.reset(state.options(), 2.0, before.q, before.p, before.v, before.ba, before.bg);

    const ImuNominal &after = state.imu();
    EXPECT_EQ(state.num_clones(), 0);
    EXPECT_EQ(state.size(), kImuStateSize);
    EXPECT_DOUBLE_EQ(state.timestamp(), 2.0);
    EXPECT_NEAR((after.q - before.q).norm(), 0.0, 1e-15);
    EXPECT_NEAR((after.p - before.p).norm(), 0.0, 1e-15);
    EXPECT_NEAR((after.v - before.v).norm(), 0.0, 1e-15);
    EXPECT_NEAR((after.ba - before.ba).norm(), 0.0, 1e-15);
    EXPECT_NEAR((after.bg - before.bg).norm(), 0.0, 1e-15);
    // The FEJ point follows the preserved state.
    EXPECT_NEAR((state.imu_fej().p - before.p).norm(), 0.0, 1e-15);

    // P is back to the init_*_sigma diagonal (the old covariance is gone).
    const Eigen::MatrixXd &P = state.covariance();
    EXPECT_EQ(P.rows(), kImuStateSize);
    EXPECT_EQ(P.cols(), kImuStateSize);
    EXPECT_NEAR(P(kImuTheta, kImuTheta), options.init_rotation_sigma * options.init_rotation_sigma,
                1e-15);
    EXPECT_NEAR(P(kImuP, kImuP), options.init_position_sigma * options.init_position_sigma,
                1e-15);
    EXPECT_NEAR(P(kImuV, kImuV), options.init_velocity_sigma * options.init_velocity_sigma,
                1e-15);
    EXPECT_NEAR(P(kImuBa, kImuBa), options.init_acc_bias_sigma * options.init_acc_bias_sigma,
                1e-15);
    EXPECT_NEAR(P(kImuBg, kImuBg), options.init_gyr_bias_sigma * options.init_gyr_bias_sigma,
                1e-15);
    Eigen::MatrixXd off_diagonal = P;
    off_diagonal.diagonal().setZero();
    EXPECT_NEAR(off_diagonal.cwiseAbs().maxCoeff(), 0.0, 1e-15);
}

TEST(StarvationRestartPolicy, T26FrameCountBackstopSoftVersusFull)
{
    // P5b: the frame-count backstop must take the soft path in soft mode instead
    // of scheduling a full clearState() restart, while the historical full
    // restart is preserved when soft mode is off. Only the
    // SCHUR_SOFT_RESTART_MAX_SECONDS escalation may schedule a full restart in
    // soft mode.
    using Action = StarvationRestartAction;

    // At/after the limit: soft mode -> kSoft, otherwise kFull.
    EXPECT_EQ(frameCountBackstopAction(false, false, 60, 60, true), Action::kSoft);
    EXPECT_EQ(frameCountBackstopAction(false, false, 60, 60, false), Action::kFull);
    EXPECT_EQ(frameCountBackstopAction(false, false, 600, 60, true), Action::kSoft);
    EXPECT_EQ(frameCountBackstopAction(false, false, 600, 60, false), Action::kFull);

    // Below the limit: nothing happens in either mode.
    EXPECT_EQ(frameCountBackstopAction(false, false, 59, 60, true), Action::kNone);
    EXPECT_EQ(frameCountBackstopAction(false, false, 59, 60, false), Action::kNone);

    // The time criterion already restarted this frame: backstop is skipped.
    EXPECT_EQ(frameCountBackstopAction(true, false, 600, 60, true), Action::kNone);
    EXPECT_EQ(frameCountBackstopAction(true, false, 600, 60, false), Action::kNone);

    // The P4 cooldown holds the backstop back in both modes.
    EXPECT_EQ(frameCountBackstopAction(false, true, 600, 60, true), Action::kNone);
    EXPECT_EQ(frameCountBackstopAction(false, true, 600, 60, false), Action::kNone);
}

// ===========================================================================
// T27 -- P6 starvation-adaptive weak-update ramp (weakUpdateScale).
// ===========================================================================
TEST(WeakUpdateScale, T27PiecewiseRampAndGuards)
{
    // Defaults reused by the estimator: starvation=0.25 s, deep=1.0 s, max=10.
    const double starve = 0.25;
    const double deep = 1.0;
    const double max_scale = 10.0;

    auto scale = [](double gap, bool enabled, double s, double d, double m) {
        return weakUpdateScale(gap, enabled, s, d, m);
    };
    auto def = [&](double gap) { return scale(gap, true, starve, deep, max_scale); };

    // Disabled -> 1.0 regardless of the gap.
    EXPECT_DOUBLE_EQ(scale(0.0, false, starve, deep, max_scale), 1.0);
    EXPECT_DOUBLE_EQ(scale(10.0, false, starve, deep, max_scale), 1.0);

    // gap < starvation_seconds -> 1.0.
    EXPECT_DOUBLE_EQ(def(0.0), 1.0);
    EXPECT_DOUBLE_EQ(def(0.249999), 1.0);

    // gap == starvation_seconds -> 1.0 (strict inequality).
    EXPECT_DOUBLE_EQ(def(starve), 1.0);

    // gap == deep_seconds -> max_scale.
    EXPECT_DOUBLE_EQ(def(deep), max_scale);

    // gap > deep_seconds -> clamped to max_scale.
    EXPECT_DOUBLE_EQ(def(5.0), max_scale);

    // Linear interpolation in between: 1 + 9*(0.625-0.25)/(1.0-0.25) = 5.5.
    EXPECT_NEAR(def(0.625), 5.5, 1e-12);
    EXPECT_NEAR(def(0.4), 1.0 + 9.0 * (0.4 - starve) / (deep - starve), 1e-12);
    EXPECT_NEAR(def(0.9), 1.0 + 9.0 * (0.9 - starve) / (deep - starve), 1e-12);
    std::printf("[T27] scale(0.25)=%.3f scale(0.4)=%.3f scale(0.625)=%.3f scale(1.0)=%.3f\n",
                def(0.25), def(0.4), def(0.625), def(1.0));

    // max_scale == 1.0 is equivalent to disabled.
    EXPECT_DOUBLE_EQ(scale(0.625, true, starve, deep, 1.0), 1.0);
    EXPECT_DOUBLE_EQ(scale(5.0, true, starve, deep, 1.0), 1.0);

    // deep <= starvation collapses the ramp to a step at starvation_seconds.
    EXPECT_DOUBLE_EQ(scale(starve, true, starve, starve, max_scale), 1.0);  // gap == starvation
    EXPECT_DOUBLE_EQ(scale(starve + 1e-9, true, starve, starve, max_scale), max_scale);
    EXPECT_DOUBLE_EQ(scale(starve + 1e-9, true, starve, starve - 0.1, max_scale), max_scale);

    // Invalid parameters -> 1.0.
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_DOUBLE_EQ(scale(5.0, true, nan, deep, max_scale), 1.0);
    EXPECT_DOUBLE_EQ(scale(5.0, true, starve, nan, max_scale), 1.0);
    EXPECT_DOUBLE_EQ(scale(5.0, true, starve, deep, nan), 1.0);
    EXPECT_DOUBLE_EQ(scale(5.0, true, inf, deep, max_scale), 1.0);
    EXPECT_DOUBLE_EQ(scale(5.0, true, starve, deep, inf), 1.0);
    EXPECT_DOUBLE_EQ(scale(5.0, true, starve, deep, 0.5), 1.0);   // max_scale < 1
    EXPECT_DOUBLE_EQ(scale(5.0, true, starve, deep, -3.0), 1.0);

    // Negative / NaN gap -> 1.0.
    EXPECT_DOUBLE_EQ(def(-1.0), 1.0);
    EXPECT_DOUBLE_EQ(def(nan), 1.0);
}


// ===========================================================================
// T28 -- P7 hold: drop_all_clones() must preserve the live 15x15 P block.
// ===========================================================================
TEST(EkfStateCloneWindow, T28DropAllClonesPreservesLiveCovariance)
{
    // The hold policy must be able to throw away stale visual structure without
    // resetting or re-inflating the covariance: exactly what
    // marginalize_oldest_clone() (hard delete) guarantees.
    EkfOptions options = test_options();
    options.window_size = 5;
    EkfState state;
    state.reset(options, 0.0, Vector4d(1, 0, 0, 0), Vector3d::Zero(), Vector3d::Zero(),
                Vector3d::Zero(), Vector3d::Zero());
    for (int k = 0; k < 4; ++k)
    {
        for (int s = 0; s < 5; ++s)
            state.propagate(Vector3d(0.1, 0.0, 9.8), Vector3d(0.02, 0.01, -0.03), 0.01);
        state.augment_clone(0.05 * (k + 1));
    }
    ASSERT_EQ(state.num_clones(), 4);

    const Mat15 live_before = state.covariance().topLeftCorner(kImuStateSize, kImuStateSize);
    const ImuNominal imu_before = state.imu();

    EXPECT_EQ(state.drop_all_clones(), 4);
    EXPECT_EQ(state.num_clones(), 0);
    EXPECT_EQ(state.size(), kImuStateSize);
    EXPECT_EQ(state.covariance().rows(), kImuStateSize);
    EXPECT_EQ(state.covariance().cols(), kImuStateSize);

    // Bit-for-bit: no reset, no re-inflation, no Schur-complement shrinkage.
    EXPECT_LT((state.covariance() - live_before).cwiseAbs().maxCoeff(), 1e-15);
    EXPECT_NEAR((state.imu().p - imu_before.p).norm(), 0.0, 1e-15);
    EXPECT_NEAR((state.imu().v - imu_before.v).norm(), 0.0, 1e-15);
    EXPECT_NEAR((state.imu().q - imu_before.q).norm(), 0.0, 1e-15);
    EXPECT_TRUE(EkfState::is_positive_definite(state.covariance()));
    // Idempotent.
    EXPECT_EQ(state.drop_all_clones(), 0);

    // The reduced state keeps propagating normally after the drop.
    const ImuNominal before_propagation = state.imu();
    state.propagate(Vector3d(0.0, 0.0, 9.8), Vector3d(0.0, 0.0, 0.0), 0.01);
    EXPECT_GT(state.covariance().trace(), live_before.trace());
    EXPECT_GT((state.imu().p - before_propagation.p).norm(), 0.0);
}

// ===========================================================================
// T29 -- P7 hold state machine: consecutive recovered frames are required.
// ===========================================================================
TEST(StarvationHold, T29StateMachineRequiresConsecutiveRecoveredFrames)
{
    StarvationHold::Config config;
    config.enabled = true;
    config.recovery_min_features = 12;
    config.recovery_frames = 2;
    StarvationHold hold;
    hold.configure(config);
    EXPECT_TRUE(hold.enabled());
    EXPECT_FALSE(hold.active());
    // A frame outside hold reports kNone and is not counted.
    EXPECT_EQ(hold.observe_frame(0.0, 100), StarvationHold::Action::kNone);
    EXPECT_EQ(hold.frames(), 0);

    // Entering hold records one episode; re-entering is a no-op.
    EXPECT_TRUE(hold.enter(10.0));
    EXPECT_TRUE(hold.active());
    EXPECT_FALSE(hold.enter(10.1));
    EXPECT_EQ(hold.episodes(), 1);
    EXPECT_DOUBLE_EQ(hold.current_duration(12.5), 2.5);

    // Too few observations break the streak and keep holding.
    EXPECT_EQ(hold.observe_frame(11.0, 11), StarvationHold::Action::kHold);
    EXPECT_EQ(hold.good_streak(), 0);
    EXPECT_EQ(hold.observe_frame(11.1, 12), StarvationHold::Action::kHold);
    EXPECT_EQ(hold.good_streak(), 1);
    EXPECT_EQ(hold.observe_frame(11.2, 5), StarvationHold::Action::kHold);
    EXPECT_EQ(hold.good_streak(), 0);
    EXPECT_EQ(hold.observe_frame(11.3, 50), StarvationHold::Action::kHold);
    EXPECT_EQ(hold.observe_frame(11.4, 50), StarvationHold::Action::kRecover);
    EXPECT_EQ(hold.frames(), 5);
    EXPECT_EQ(hold.good_streak(), 2);

    // Leaving hold records the duration; later frames are not hold frames.
    EXPECT_DOUBLE_EQ(hold.leave(11.5), 1.5);
    EXPECT_FALSE(hold.active());
    EXPECT_DOUBLE_EQ(hold.last_duration(), 1.5);
    EXPECT_DOUBLE_EQ(hold.total_duration(), 1.5);
    EXPECT_DOUBLE_EQ(hold.max_duration(), 1.5);
    EXPECT_EQ(hold.observe_frame(11.6, 100), StarvationHold::Action::kNone);
    EXPECT_EQ(hold.frames(), 5);

    // The recovery streak does not leak into the next episode.
    EXPECT_TRUE(hold.enter(20.0));
    EXPECT_EQ(hold.good_streak(), 0);
    EXPECT_EQ(hold.episodes(), 2);
    // recovery_frames < 1 is clamped to 1.
    config.recovery_frames = 0;
    hold.configure(config);
    EXPECT_EQ(hold.observe_frame(20.1, 12), StarvationHold::Action::kRecover);
    hold.leave(20.2);
    // A shorter episode must not lower the recorded maximum.
    EXPECT_NEAR(hold.total_duration(), 1.7, 1e-12);
    EXPECT_DOUBLE_EQ(hold.max_duration(), 1.5);

    // recovery_min_features <= 0 accepts any frame (even one with no tracks).
    config.recovery_min_features = 0;
    hold.configure(config);
    EXPECT_TRUE(hold.enter(21.0));
    EXPECT_EQ(hold.episodes(), 3);
    EXPECT_EQ(hold.observe_frame(21.1, 0), StarvationHold::Action::kRecover);
    hold.deactivate();
    EXPECT_FALSE(hold.active());
    // Cumulative diagnostics survive deactivate() (used on a full re-seed).
    EXPECT_EQ(hold.episodes(), 3);
    EXPECT_EQ(hold.frames(), 7);
}

// ===========================================================================
// T30 -- P7 hold mode takes precedence in the frame-count backstop.
// ===========================================================================
TEST(StarvationRestartPolicy, T30HoldModeBackstopNeverRestarts)
{
    using Action = StarvationRestartAction;
    // At/after the limit: hold instead of soft/full, even when both are on.
    EXPECT_EQ(frameCountBackstopAction(false, false, 60, 60, false, true), Action::kHold);
    EXPECT_EQ(frameCountBackstopAction(false, false, 600, 60, true, true), Action::kHold);
    EXPECT_EQ(frameCountBackstopAction(false, false, 60, 60, true, true), Action::kHold);
    // Same suppression rules as the restart paths.
    EXPECT_EQ(frameCountBackstopAction(true, false, 600, 60, false, true), Action::kNone);
    EXPECT_EQ(frameCountBackstopAction(false, true, 600, 60, false, true), Action::kNone);
    EXPECT_EQ(frameCountBackstopAction(false, false, 59, 60, false, true), Action::kNone);
    // The five-argument call keeps the historical semantics.
    EXPECT_EQ(frameCountBackstopAction(false, false, 60, 60, false), Action::kFull);
}

// ===========================================================================
// T31 -- P7 hold exit re-anchors the baseline but keeps the diagnostics.
// ===========================================================================
TEST(StarvationMonitor, T31RebaselineKeepsMaximumGapAndCounters)
{
    StarvationMonitor::Config config;
    config.max_no_update_seconds = 3.0;
    config.starvation_seconds = 0.25;
    config.noise_scale = 3.0;
    StarvationMonitor monitor;
    monitor.configure(config);
    monitor.reset(10.0);

    monitor.observe_failed_frame(10.1);
    const StarvationMonitor::Decision decision = monitor.observe_failed_frame(14.0);
    EXPECT_TRUE(decision.restart);
    EXPECT_EQ(monitor.time_gate_triggers(), 1);

    // Hold exit: re-anchor without claiming an update was applied.
    monitor.rebaseline(14.0);
    EXPECT_DOUBLE_EQ(monitor.starvation_duration(14.0), 0.0);
    EXPECT_FALSE(monitor.starvation_active());
    // The hold duration is folded into the diagnostic maximum.
    EXPECT_DOUBLE_EQ(monitor.max_gap_seconds(), 4.0);
    // Counters survive and a full grace period is available again.
    EXPECT_EQ(monitor.time_gate_triggers(), 1);
    EXPECT_FALSE(monitor.observe_failed_frame(16.9).restart);
    EXPECT_TRUE(monitor.observe_failed_frame(17.1).restart);
    EXPECT_EQ(monitor.time_gate_triggers(), 2);
}

}  // namespace backend
}  // namespace vins
