// Standalone gtest suite for the ARM-oriented Schur-complement EKF state.
// Eigen + std + gtest only (no ROS / Ceres / OpenCV).
//
// The tests re-derive the analytic Jacobian / Schur complement locally instead
// of reusing the implementation, so sign, ordering and indexing mistakes in
// ekf_state.cpp are actually caught.

#include <eigen3/Eigen/Dense>
#include <gtest/gtest.h>

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
using Mat15 = Eigen::Matrix<double, K_IMU_STATE_SIZE, K_IMU_STATE_SIZE>;
using Vec15 = Eigen::Matrix<double, K_IMU_STATE_SIZE, 1>;
using Vec6 = Eigen::Matrix<double, K_CLONE_STATE_SIZE, 1>;

constexpr double K_PI = 3.14159265358979323846;

// --------------------------------------------------------------------------
// Test-local primitives (independent re-implementation of the conventions).
// --------------------------------------------------------------------------

Vector4d random_unit_quaternion(std::mt19937 & _rng)
{
  std::normal_distribution<double> normal(0.0, 1.0);
  Vector4d q;
  for (int i = 0; i < 4; ++i)
    q(i) = normal(_rng);
  return q / q.norm();
}

Vector3d random_axis_angle(std::mt19937 & _rng)
{
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  Vector3d axis;
  for (int i = 0; i < 3; ++i)
    axis(i) = uni(_rng);
  if (axis.norm() < 1e-6)
    axis = Vector3d::UnitX();
  axis.normalize();
  std::uniform_real_distribution<double> mag(0.0, 0.99 * K_PI);
  return axis * mag(_rng);
}

Matrix3d rotation_of(const Vector4d & _q)
{
  const Eigen::Quaterniond eq(_q(0), _q(1), _q(2), _q(3));
  return eq.normalized().toRotationMatrix();
}

Matrix3d skew_of(const Vector3d & _v)
{
  Matrix3d S;
  S << 0.0, -_v(2), _v(1), _v(2), 0.0, -_v(0), -_v(1), _v(0), 0.0;
  return S;
}

Vector4d quat_multiply_local(const Vector4d & _a, const Vector4d & _b)
{
  const Eigen::Quaterniond ea(_a(0), _a(1), _a(2), _a(3));
  const Eigen::Quaterniond eb(_b(0), _b(1), _b(2), _b(3));
  const Eigen::Quaterniond e = ea * eb;
  return Vector4d(e.w(), e.x(), e.y(), e.z());
}

Vector4d quat_conjugate(const Vector4d & _q) { return Vector4d(_q(0), -_q(1), -_q(2), -_q(3)); }

// One RK4 step of q' = 0.5 Omega(gyr - bg) q, p' = v, v' = R(q)(acc - ba) + g_w
// with the biases frozen inside the step (mirrors the spec equations).
void test_rk4_step(
  const Vector3d & _w, const Vector3d & _a, const Vector3d & _g_w, double _dt, Vector4d & _q, Vector3d & _p,
  Vector3d & _v)
{
  Matrix4d omega = Matrix4d::Zero();
  omega.block<1, 3>(0, 1) = -_w.transpose();
  omega.block<3, 1>(1, 0) = _w;
  omega.block<3, 3>(1, 1) = -skew_of(_w);

  // Explicit return types: an `auto` lambda would deduce an Eigen expression
  // template that dangles once the lambda returns.
  const auto dq_dt = [&omega](const Vector4d & _qq) -> Vector4d { return 0.5 * (omega * _qq); };
  const auto dv_dt = [&_a, &_g_w](const Vector4d & _qq) -> Vector3d { return rotation_of(_qq) * _a + _g_w; };

  const Vector4d k1q = dq_dt(_q), k2q = dq_dt(_q + 0.5 * _dt * k1q), k3q = dq_dt(_q + 0.5 * _dt * k2q),
                 k4q = dq_dt(_q + _dt * k3q);
  const Vector3d k1v = dv_dt(_q), k2v = dv_dt(_q + 0.5 * _dt * k1q), k3v = dv_dt(_q + 0.5 * _dt * k2q),
                 k4v = dv_dt(_q + _dt * k3q);
  const Vector3d k1p = _v, k2p = _v + 0.5 * _dt * k1v, k3p = _v + 0.5 * _dt * k2v, k4p = _v + _dt * k3v;

  _q += (_dt / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q);
  _p += (_dt / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);
  _v += (_dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
  _q /= _q.norm();
}

// Test-local F from the spec, evaluated at the FEJ (= nominal) values.
Mat15 build_F(const Vector3d & _w, const Vector3d & _a, const Matrix3d & _R)
{
  const Matrix3d I3 = Matrix3d::Identity();
  Mat15 F = Mat15::Zero();
  F.block<3, 3>(K_IMU_THETA, K_IMU_THETA) = -skew_of(_w);
  F.block<3, 3>(K_IMU_THETA, K_IMU_BG) = -I3;
  F.block<3, 3>(K_IMU_P, K_IMU_V) = I3;
  F.block<3, 3>(K_IMU_V, K_IMU_THETA) = -_R * skew_of(_a);
  F.block<3, 3>(K_IMU_V, K_IMU_BA) = -_R;
  return F;
}

EkfOptions_s test_options()
{
  EkfOptions_s options;
  options.gravity_ = 9.81007;
  options.acc_noise_ = 0.1;
  options.gyr_noise_ = 0.01;
  options.acc_bias_noise_ = 0.001;
  options.gyr_bias_noise_ = 0.0001;
  options.init_position_sigma_ = 0.05;
  options.init_velocity_sigma_ = 0.05;
  options.init_rotation_sigma_ = 0.01;
  options.init_acc_bias_sigma_ = 0.1;
  options.init_gyr_bias_sigma_ = 0.01;
  return options;
}

void check_symmetric(const Eigen::MatrixXd & _P, double _tol = 1e-12)
{
  EXPECT_LT((_P - _P.transpose()).cwiseAbs().maxCoeff(), _tol);
}

// Smallest eigenvalue of the symmetric part (test-local PSD check).
double min_eigenvalue(const Eigen::MatrixXd & _P)
{
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (_P + _P.transpose()));
  return es.eigenvalues().minCoeff();
}

}  // namespace

// ---------------------------------------------------------------- T1 --------

TEST(EkfStateHelpers, T1RotationMatchesEigenQuaternion)
{
  std::mt19937 rng(1234);
  for (int k = 0; k < 200; ++k) {
    const Vector4d q = random_unit_quaternion(rng);
    const Matrix3d R_impl = EkfState::rotation(q);
    const Matrix3d R_eigen = rotation_of(q);
    EXPECT_NEAR((R_impl - R_eigen).cwiseAbs().maxCoeff(), 0.0, 1e-12);
    EXPECT_NEAR(R_impl.determinant(), 1.0, 1e-12);
  }
  // Near-zero quaternion falls back to identity instead of producing NaNs.
  EXPECT_TRUE(EkfState::rotation(Vector4d(0, 0, 0, 0)).isApprox(Matrix3d::Identity(), 1e-12));
}

// ---------------------------------------------------------------- T2 --------

TEST(EkfStateHelpers, T2QuaternionRoundTripsAndNorm)
{
  std::mt19937 rng(4321);
  for (int k = 0; k < 200; ++k) {
    const Vector3d axis_angle = random_axis_angle(rng);
    const Vector4d q = EkfState::quaternionFromAxisAngle(axis_angle);
    EXPECT_NEAR(q.norm(), 1.0, 1e-12);
    const Vector3d back = EkfState::quaternionToAxisAngle(q);
    EXPECT_NEAR((back - axis_angle).norm(), 0.0, 1e-9);
  }
  // Small angles stay finite and round trip through the series branch.
  for (double mag : {0.0, 1e-12, 1e-9, 1e-6}) {
    const Vector3d axis_angle = Vector3d(1.0, -2.0, 0.5).normalized() * mag;
    const Vector4d q = EkfState::quaternionFromAxisAngle(axis_angle);
    EXPECT_NEAR((EkfState::quaternionToAxisAngle(q) - axis_angle).norm(), 0.0, 1e-9);
  }

  // Hamilton product against Eigen, and q (x) q* = identity.
  for (int k = 0; k < 100; ++k) {
    const Vector4d a = random_unit_quaternion(rng);
    const Vector4d b = random_unit_quaternion(rng);
    EXPECT_NEAR((EkfState::quaternionMultiply(a, b) - quat_multiply_local(a, b)).norm(), 0.0, 1e-12);
    EXPECT_NEAR((EkfState::quaternionMultiply(a, quat_conjugate(a)) - Vector4d(1, 0, 0, 0)).norm(), 0.0, 1e-12);
  }

  // The mean quaternion must stay a unit quaternion over a long run.
  EkfOptions_s options = test_options();
  options.gravity_ = 0.0;
  EkfState state;
  state.reset(
    options, 0.0, random_unit_quaternion(rng), Vector3d(0.1, 0.2, 0.3), Vector3d(0.0, 0.0, 0.0),
    Vector3d::Zero(), Vector3d::Zero());
  std::normal_distribution<double> normal(0.0, 1.0);
  for (int k = 0; k < 10000; ++k) {
    const Vector3d acc(normal(rng), normal(rng), normal(rng));
    const Vector3d gyr(normal(rng), normal(rng), normal(rng));
    state.propagate(acc, gyr, 0.01);
  }
  EXPECT_NEAR(state.imu().q_.norm(), 1.0, 1e-8);
}

// ---------------------------------------------------------------- T3 --------

TEST(EkfStateHelpers, T3PoseCorrectionMatchesRightPerturbation)
{
  std::mt19937 rng(99);
  for (int k = 0; k < 50; ++k) {
    const Vector4d q = random_unit_quaternion(rng);
    const Vector3d p = Vector3d::Random();
    const Vector3d dtheta = random_axis_angle(rng).normalized() * 1e-5;

    Vec6 dx = Vec6::Zero();
    dx.head<3>() = dtheta;
    Vector4d q_c = q;
    Vector3d p_c = p;
    EkfState::applyPoseCorrection(q_c, p_c, dx);

    const Matrix3d expected = rotation_of(q) * (Matrix3d::Identity() + skew_of(dtheta));
    EXPECT_NEAR((EkfState::rotation(q_c) - expected).cwiseAbs().maxCoeff(), 0.0, 1e-6);
    EXPECT_NEAR((p_c - p).norm(), 0.0, 1e-15);
  }
}

// ---------------------------------------------------------------- T4 --------

TEST(EkfStatePropagation, T4RestStaysAtRestAndCovarianceGrows)
{
  EkfOptions_s options = test_options();
  options.gravity_ = 0.0;
  EkfState state;
  state.reset(
    options, 1.0, Vector4d(1, 0, 0, 0), Vector3d::Zero(), Vector3d::Zero(), Vector3d::Zero(), Vector3d::Zero());

  Eigen::Matrix<double, K_IMU_STATE_SIZE, 1> previous = state.covariance().diagonal().head<K_IMU_STATE_SIZE>();
  for (int k = 0; k < 100; ++k) {
    state.propagate(Vector3d::Zero(), Vector3d::Zero(), 0.01);

    EXPECT_NEAR((state.imu().q_ - Vector4d(1, 0, 0, 0)).norm(), 0.0, 1e-12);
    EXPECT_NEAR(state.imu().p_.norm(), 0.0, 1e-15);
    EXPECT_NEAR(state.imu().v_.norm(), 0.0, 1e-15);
    EXPECT_NEAR(state.timestamp(), 1.0 + 0.01 * (k + 1), 1e-12);

    const auto diagonal = state.covariance().diagonal().head<K_IMU_STATE_SIZE>();
    for (int i = K_IMU_P; i < K_IMU_V + 3; ++i)
      EXPECT_GE(diagonal(i), previous(i) - 1e-15);
    previous = diagonal;
  }
}

// ---------------------------------------------------------------- T5 --------

TEST(EkfStatePropagation, T5ConstantAccelerationIsIntegratedExactly)
{
  EkfOptions_s options = test_options();
  options.gravity_ = 0.0;
  const Vector3d acc(0.3, -0.2, 0.5);
  const double total_time = 1.0;
  const int steps = 100;
  const double dt = total_time / steps;

  EkfState state;
  state.reset(
    options, 0.0, Vector4d(1, 0, 0, 0), Vector3d::Zero(), Vector3d::Zero(), Vector3d::Zero(), Vector3d::Zero());
  for (int k = 0; k < steps; ++k)
    state.propagate(acc, Vector3d::Zero(), dt);

  EXPECT_NEAR((state.imu().v_ - acc * total_time).norm(), 0.0, 1e-9);
  EXPECT_NEAR((state.imu().p_ - 0.5 * acc * total_time * total_time).norm(), 0.0, 1e-9);
  EXPECT_NEAR((state.imu().q_ - Vector4d(1, 0, 0, 0)).norm(), 0.0, 1e-12);
}

// ---------------------------------------------------------------- T6 --------

TEST(EkfStateCovariance, T6SymmetryAndPositiveDefinitenessOverChains)
{
  EkfOptions_s options = test_options();
  options.window_size_ = 3;
  EkfState state;
  state.reset(
    options, 0.0, Vector4d(1, 0, 0, 0), Vector3d::Zero(), Vector3d::Zero(), Vector3d::Zero(), Vector3d::Zero());

  std::mt19937 rng(7);
  std::normal_distribution<double> normal(0.0, 0.1);
  double timestamp = 0.0;
  int steps_since_augment = 100;  // no clone yet

  for (int k = 0; k < 60; ++k) {
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
    const bool expect_pd = state.numClones() == 0 || steps_since_augment >= 2;
    if (expect_pd)
      EXPECT_TRUE(EkfState::isPositiveDefinite(state.covariance()));
    else
      EXPECT_GE(min_eigenvalue(state.covariance()), -1e-9);

    if (k % 5 == 4) {
      state.augmentClone(timestamp);
      steps_since_augment = 0;
      state.pruneCloneWindow();
      check_symmetric(state.covariance());
      EXPECT_LE(state.numClones(), options.window_size_);
      EXPECT_EQ(state.size(), K_IMU_STATE_SIZE + K_CLONE_STATE_SIZE * state.numClones());
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
  ASSERT_GT(state.numClones(), 0);
  ASSERT_TRUE(EkfState::isPositiveDefinite(state.covariance()));
  state.marginalizeOldestClone();
  check_symmetric(state.covariance());
  EXPECT_TRUE(EkfState::isPositiveDefinite(state.covariance()));
}

// ---------------------------------------------------------------- T7 --------

TEST(EkfStateCloneWindow, T7AugmentCloneCopiesTheLivePose)
{
  EkfOptions_s options = test_options();
  options.window_size_ = 8;
  EkfState state;
  state.reset(
    options, 0.0, Vector4d(1, 0, 0, 0), Vector3d::Zero(), Vector3d::Zero(), Vector3d::Zero(), Vector3d::Zero());

  double timestamp = 0.0;
  for (int k = 0; k < 3; ++k) {
    for (int s = 0; s < 4; ++s) {
      state.propagate(Vector3d(0.1, 0.0, 9.8), Vector3d(0.1, 0.05, -0.02), 0.01);
      timestamp += 0.01;
    }
    const int n_before = state.size();
    const Eigen::MatrixXd P_before = state.covariance();
    state.augmentClone(timestamp);

    ASSERT_EQ(state.numClones(), k + 1);
    EXPECT_EQ(state.size(), n_before + K_CLONE_STATE_SIZE);
    EXPECT_EQ(state.timestamp(), timestamp);
    EXPECT_EQ(state.cloneTimestamp(k), timestamp);

    // The new clone is a frozen copy of the live pose (mean and FEJ).
    EXPECT_NEAR((state.clone(k).q_ - state.imu().q_).norm(), 0.0, 1e-15);
    EXPECT_NEAR((state.clone(k).p_ - state.imu().p_).norm(), 0.0, 1e-15);
    EXPECT_NEAR((state.cloneFej(k).q_ - state.imuFej().q_).norm(), 0.0, 1e-15);
    EXPECT_NEAR((state.cloneFej(k).p_ - state.imuFej().p_).norm(), 0.0, 1e-15);

    // Its covariance block (and the cross terms) equal the live pose block.
    const int b = state.cloneOffset(k);
    const Eigen::MatrixXd & P = state.covariance();
    EXPECT_LT(
      (P.block(b, b, K_CLONE_STATE_SIZE, K_CLONE_STATE_SIZE) -
       P.topLeftCorner(K_CLONE_STATE_SIZE, K_CLONE_STATE_SIZE))
        .cwiseAbs()
        .maxCoeff(),
      1e-15);
    EXPECT_LT(
      (P.block(b, 0, K_CLONE_STATE_SIZE, n_before) - P_before.topRows(K_CLONE_STATE_SIZE)).cwiseAbs().maxCoeff(),
      1e-15);
    EXPECT_LT(
      (P.block(0, b, n_before, K_CLONE_STATE_SIZE) - P_before.leftCols(K_CLONE_STATE_SIZE)).cwiseAbs().maxCoeff(),
      1e-15);
  }

  // augment_clone() only appends; pruning is an explicit post-update step.
  options.window_size_ = 2;
  EkfState trimmed;
  trimmed.reset(
    options, 0.0, Vector4d(1, 0, 0, 0), Vector3d::Zero(), Vector3d::Zero(), Vector3d::Zero(), Vector3d::Zero());
  for (int k = 0; k < 10; ++k) {
    trimmed.propagate(Vector3d(0.0, 0.0, 9.8), Vector3d(0.1, 0.0, 0.0), 0.01);
    trimmed.augmentClone(0.01 * (k + 1));
    // Pruning happens after every augment below, so the pre-prune size is
    // the previous (already bounded) window plus this new clone.
    EXPECT_EQ(trimmed.numClones(), std::min(k + 1, 3));
    EXPECT_EQ(trimmed.size(), K_IMU_STATE_SIZE + K_CLONE_STATE_SIZE * trimmed.numClones());
    const int removed = trimmed.pruneCloneWindow();
    EXPECT_EQ(removed, k >= 2 ? 1 : 0);
    EXPECT_EQ(trimmed.numClones(), std::min(k + 1, 2));
  }
  EXPECT_EQ(trimmed.numClones(), 2);
}

// ---------------------------------------------------------------- T8 --------

TEST(EkfStateCloneWindow, T8MarginalizeOldestDeletesRowsAndColumns)
{
  EkfOptions_s options = test_options();
  options.window_size_ = 4;
  EkfState state;
  state.reset(
    options, 0.0, Vector4d(1, 0, 0, 0), Vector3d::Zero(), Vector3d::Zero(), Vector3d::Zero(), Vector3d::Zero());

  const Vector3d acc(0.2, -0.1, 9.8);
  const Vector3d gyr(0.3, -0.2, 0.15);
  double timestamp = 0.0;
  for (int k = 0; k < 3; ++k) {
    state.propagate(acc, gyr, 0.01);
    timestamp += 0.01;
  }
  state.augmentClone(timestamp);
  for (int k = 0; k < 3; ++k) {
    state.propagate(acc, gyr, 0.01);
    timestamp += 0.01;
  }
  state.augmentClone(timestamp);
  for (int k = 0; k < 3; ++k) {
    state.propagate(acc, gyr, 0.01);
    timestamp += 0.01;
  }

  ASSERT_EQ(state.numClones(), 2);
  const int n = state.size();
  const int b = state.cloneOffset(0);
  const int m = n - K_CLONE_STATE_SIZE;
  const Eigen::MatrixXd P = state.covariance();

  // OpenVINS/SchurVINS remove the clone by direct row/column deletion.
  Eigen::VectorXi keep(m);
  int k = 0;
  for (int i = 0; i < n; ++i)
    if (i < b || i >= b + K_CLONE_STATE_SIZE)
      keep(k++) = i;
  const Eigen::MatrixXd expected = P(keep, keep);

  const CloneNominal_s retained_clone = state.clone(1);
  const CloneNominal_s retained_clone_fej = state.cloneFej(1);
  const double retained_timestamp = state.cloneTimestamp(1);

  state.marginalizeOldestClone();

  ASSERT_EQ(state.size(), n - K_CLONE_STATE_SIZE);
  ASSERT_EQ(state.numClones(), 1);
  EXPECT_LT((state.covariance() - expected).cwiseAbs().maxCoeff(), 1e-9);
  check_symmetric(state.covariance());
  EXPECT_TRUE(EkfState::isPositiveDefinite(state.covariance()));

  // Remaining clone shifted down to index 0 and keeps its values.
  EXPECT_NEAR((state.clone(0).q_ - retained_clone.q_).norm(), 0.0, 1e-15);
  EXPECT_NEAR((state.clone(0).p_ - retained_clone.p_).norm(), 0.0, 1e-15);
  EXPECT_NEAR((state.cloneFej(0).q_ - retained_clone_fej.q_).norm(), 0.0, 1e-15);
  EXPECT_NEAR((state.cloneFej(0).p_ - retained_clone_fej.p_).norm(), 0.0, 1e-15);
  EXPECT_EQ(state.cloneTimestamp(0), retained_timestamp);

  // Dropping the last clone leaves the live block; further calls are no-ops.
  state.marginalizeOldestClone();
  ASSERT_EQ(state.numClones(), 0);
  EXPECT_EQ(state.size(), K_IMU_STATE_SIZE);
  const Eigen::MatrixXd live_only = state.covariance();
  state.marginalizeOldestClone();
  EXPECT_EQ(state.size(), K_IMU_STATE_SIZE);
  EXPECT_EQ(state.covariance(), live_only);
}

// ---------------------------------------------------------------- T9 --------

TEST(EkfStatePropagation, T9FiniteDifferenceMatchesAnalyticPhi)
{
  EkfOptions_s options = test_options();
  options.gravity_ = 9.81007;

  std::mt19937 rng(2024);
  const Vector4d q0 = EkfState::quaternionFromAxisAngle(Vector3d(0.2, -0.3, 0.15));
  const Vector3d p0(0.4, -1.2, 0.7);
  const Vector3d v0(0.3, 0.1, -0.2);
  const Vector3d ba(0.02, -0.01, 0.03);
  const Vector3d bg(0.001, -0.002, 0.0015);
  const Vector3d acc(0.5, -0.25, 9.9);
  const Vector3d gyr(0.35, -0.2, 0.45);
  const double dt = 0.01;
  const Vector3d g_w(0.0, 0.0, -options.gravity_);

  // Analytic Phi from the spec (test-local F).
  const Vector3d w = gyr - bg;
  const Vector3d a = acc - ba;
  const Mat15 F = build_F(w, a, rotation_of(q0));
  const Mat15 Phi =
    Mat15::Identity() + F * dt + 0.5 * (F * F) * (dt * dt) + (1.0 / 6.0) * (F * F * F) * (dt * dt * dt);

  // Nominal integration with the test-local RK4 (same equations as propagate).
  const auto integrate = [&](
                           const Vector4d & _q_in, const Vector3d & _p_in, const Vector3d & _v_in,
                           const Vector3d & _ba_in, const Vector3d & _bg_in, Vector4d & _q_out,
                           Vector3d & _p_out, Vector3d & _v_out)
  {
    _q_out = _q_in;
    _p_out = _p_in;
    _v_out = _v_in;
    test_rk4_step(gyr - _bg_in, acc - _ba_in, g_w, dt, _q_out, _p_out, _v_out);
  };

  Vector4d q_base, q_plus, q_minus;
  Vector3d p_base, p_plus, p_minus, v_base, v_plus, v_minus;
  integrate(q0, p0, v0, ba, bg, q_base, p_base, v_base);

  const double eps = 1e-7;
  Mat15 phi_fd = Mat15::Zero();
  for (int i = 0; i < K_IMU_STATE_SIZE; ++i) {
    Vec15 step = Vec15::Zero();
    Vec15 delta_plus = Vec15::Zero();
    Vec15 delta_minus = Vec15::Zero();
    for (double sign : {1.0, -1.0}) {
      step(i) = sign * eps;
      Vector4d q_p = q0;
      Vector3d p_p = p0;
      // Rotation uses the right (body frame) perturbation; the remaining
      // components are additive.
      EkfState::applyPoseCorrection(q_p, p_p, step.head<6>());
      const Vector3d v_p = v0 + step.segment<3>(K_IMU_V);
      const Vector3d ba_p = ba + step.segment<3>(K_IMU_BA);
      const Vector3d bg_p = bg + step.segment<3>(K_IMU_BG);

      Vector4d q_i;
      Vector3d p_i, v_i;
      integrate(q_p, p_p, v_p, ba_p, bg_p, q_i, p_i, v_i);

      // Error state of the perturbed trajectory w.r.t. the nominal one,
      // using the same right perturbation convention.
      const Vector4d dq = quat_multiply_local(quat_conjugate(q_base), q_i);
      Vec15 delta = Vec15::Zero();
      delta.segment<3>(K_IMU_THETA) = EkfState::quaternionToAxisAngle(dq);
      delta.segment<3>(K_IMU_P) = p_i - p_base;
      delta.segment<3>(K_IMU_V) = v_i - v_base;
      delta.segment<3>(K_IMU_BA) = ba_p - ba;
      delta.segment<3>(K_IMU_BG) = bg_p - bg;
      if (sign > 0.0)
        delta_plus = delta;
      else
        delta_minus = delta;
    }
    phi_fd.col(i) = (delta_plus - delta_minus) / (2.0 * eps);
  }

  double max_error = 0.0;
  for (int r = 0; r < K_IMU_STATE_SIZE; ++r)
    for (int c = 0; c < K_IMU_STATE_SIZE; ++c)
      max_error = std::max(max_error, std::abs(phi_fd(r, c) - Phi(r, c)));
  std::cout << "[T9] max |Phi_fd - Phi| = " << max_error << '\n';
  EXPECT_LT(max_error, 5e-3);

  // The implementation's own RK4 must reproduce the test-local integration,
  // otherwise the finite-difference comparison above would be meaningless.
  EkfState state;
  state.reset(options, 5.0, q0, p0, v0, ba, bg);
  state.propagate(acc, gyr, dt);
  EXPECT_NEAR((state.imu().q_ - q_base).norm(), 0.0, 1e-12);
  EXPECT_NEAR((state.imu().p_ - p_base).norm(), 0.0, 1e-12);
  EXPECT_NEAR((state.imu().v_ - v_base).norm(), 0.0, 1e-12);
  EXPECT_NEAR(state.timestamp(), 5.0 + dt, 1e-12);
  // Propagation advances the live FEJ to the new mean (OpenVINS behavior).
  EXPECT_NEAR((state.imuFej().q_ - state.imu().q_).norm(), 0.0, 1e-15);
  EXPECT_NEAR((state.imuFej().p_ - state.imu().p_).norm(), 0.0, 1e-15);
  EXPECT_NEAR((state.imuFej().v_ - state.imu().v_).norm(), 0.0, 1e-15);
  EXPECT_NEAR((state.imuFej().ba_ - state.imu().ba_).norm(), 0.0, 1e-15);
  EXPECT_NEAR((state.imuFej().bg_ - state.imu().bg_).norm(), 0.0, 1e-15);
}

// --------------------------------------------------------------- T10 --------

TEST(EkfStatePropagation, T10PropagationRefreshesImuFejButFreezesCloneFej)
{
  EkfOptions_s options = test_options();
  options.window_size_ = 4;
  EkfState state;
  state.reset(
    options, 0.0, Vector4d(1, 0, 0, 0), Vector3d::Zero(), Vector3d::Zero(), Vector3d::Zero(), Vector3d::Zero());

  state.propagate(Vector3d(0.1, -0.2, 9.8), Vector3d(0.2, -0.1, 0.05), 0.01);
  state.augmentClone(0.01);
  const CloneNominal_s frozen_clone = state.clone(0);
  const CloneNominal_s frozen_clone_fej = state.cloneFej(0);

  state.propagate(Vector3d(0.2, 0.1, 9.9), Vector3d(-0.1, 0.2, 0.03), 0.01);

  EXPECT_NEAR((state.imuFej().q_ - state.imu().q_).norm(), 0.0, 1e-15);
  EXPECT_NEAR((state.imuFej().p_ - state.imu().p_).norm(), 0.0, 1e-15);
  EXPECT_NEAR((state.imuFej().v_ - state.imu().v_).norm(), 0.0, 1e-15);
  EXPECT_NEAR((state.clone(0).q_ - frozen_clone.q_).norm(), 0.0, 1e-15);
  EXPECT_NEAR((state.clone(0).p_ - frozen_clone.p_).norm(), 0.0, 1e-15);
  EXPECT_NEAR((state.cloneFej(0).q_ - frozen_clone_fej.q_).norm(), 0.0, 1e-15);
  EXPECT_NEAR((state.cloneFej(0).p_ - frozen_clone_fej.p_).norm(), 0.0, 1e-15);
}

// --------------------------------------------------------------- T11 --------

TEST(EkfStateCovariance, T11PositiveSemidefiniteCheckAllowsSingularCovariance)
{
  Eigen::MatrixXd psd(4, 4);
  psd << 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 0.5, 1.0;
  EXPECT_TRUE(EkfState::isPositiveSemidefinite(psd));
  EXPECT_FALSE(EkfState::isPositiveDefinite(psd));

  Eigen::MatrixXd indefinite = psd;
  indefinite(2, 2) = -1e-2;
  EXPECT_FALSE(EkfState::isPositiveSemidefinite(indefinite));
}

// ----------------------------------------------- T12-T14: starvation monitor --

// Reopen the anonymous namespace for a file-local helper.
namespace
{
// One propagate() step from an identical seed, with a configurable noise scale.
Eigen::MatrixXd propagate_one_step(double _noise_scale)
{
  EkfOptions_s options = test_options();
  options.window_size_ = 2;
  EkfState state;
  state.reset(
    options, 0.0, Vector4d(1, 0, 0, 0), Vector3d(0.1, 0.2, 0.3), Vector3d(1.0, 0.0, 0.0),
    Vector3d(0.01, 0.02, 0.03), Vector3d(0.001, 0.002, 0.003));
  state.propagate(Vector3d(0.1, -0.05, 9.8), Vector3d(0.2, -0.1, 0.05), 0.01, _noise_scale);
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
  for (int i = 0; i < 30; ++i) {
    const StarvationMonitor::Decision_s d = monitor.observeFailedFrame(100.0 + 0.03125 * (i + 1));
    EXPECT_FALSE(d.restart_);
    EXPECT_FALSE(d.rebaselined_);
  }
  EXPECT_EQ(monitor.timeGateTriggers(), 0);
  // 0.9375 s > starvation_seconds (0.25) -> process-noise inflation is on.
  EXPECT_TRUE(monitor.starvationActive());

  // A frame whose gap exceeds the time threshold fires exactly once.
  const StarvationMonitor::Decision_s fired = monitor.observeFailedFrame(101.5);
  EXPECT_TRUE(fired.restart_);
  EXPECT_FALSE(fired.rebaselined_);
  EXPECT_DOUBLE_EQ(fired.gap_seconds_, 1.5);
  EXPECT_EQ(monitor.timeGateTriggers(), 1);

  // A successful update re-baselines the monitor and clears starvation.
  monitor.noteAppliedUpdate(101.5);
  EXPECT_FALSE(monitor.starvationActive());
  EXPECT_DOUBLE_EQ(monitor.lastUpdateTime(), 101.5);

  // A small forward gap afterwards does not fire and leaves inflation off.
  const StarvationMonitor::Decision_s small = monitor.observeFailedFrame(101.5625);
  EXPECT_FALSE(small.restart_);
  EXPECT_FALSE(monitor.starvationActive());

  // A gap exactly at the threshold (strict '>') must not fire either.
  const StarvationMonitor::Decision_s edge = monitor.observeFailedFrame(102.5);
  EXPECT_FALSE(edge.restart_);
  EXPECT_DOUBLE_EQ(edge.gap_seconds_, 1.0);
  EXPECT_EQ(monitor.timeGateTriggers(), 1);
}

TEST(StarvationMonitor, T13BackwardsTimestampRebaselinesInsteadOfRestarting)
{
  StarvationMonitor monitor;
  monitor.configure({1.0, 0.25});
  monitor.reset(500.0);

  // Starve the monitor first (0.5 s gap > starvation_seconds) so we can prove
  // that a rollback also clears the noise-inflation flag.
  const StarvationMonitor::Decision_s starved = monitor.observeFailedFrame(500.5);
  EXPECT_FALSE(starved.restart_);
  EXPECT_TRUE(monitor.starvationActive());

  // A large backwards jump (bag replay / clock rollback) must re-baseline the
  // monitor, never schedule a restart, and reset the starvation state to the
  // new (earlier) baseline.
  const StarvationMonitor::Decision_s back = monitor.observeFailedFrame(400.0);
  EXPECT_TRUE(back.rebaselined_);
  EXPECT_FALSE(back.restart_);
  EXPECT_FALSE(monitor.starvationActive());
  EXPECT_EQ(monitor.timeGateTriggers(), 0);
  EXPECT_DOUBLE_EQ(monitor.lastUpdateTime(), 400.0);

  // After re-baselining, a normal forward gap behaves as usual.
  const StarvationMonitor::Decision_s forward = monitor.observeFailedFrame(400.125);
  EXPECT_FALSE(forward.restart_);
  EXPECT_FALSE(forward.rebaselined_);
  EXPECT_DOUBLE_EQ(forward.gap_seconds_, 0.125);
}

TEST(StarvationMonitor, T14DisabledCriteriaNeverFire)
{
  StarvationMonitor monitor;
  monitor.configure({0.0, 0.0});  // both criteria disabled
  monitor.reset(0.0);
  const StarvationMonitor::Decision_s d = monitor.observeFailedFrame(1000.0);
  EXPECT_FALSE(d.restart_);
  EXPECT_FALSE(monitor.starvationActive());
  EXPECT_EQ(monitor.timeGateTriggers(), 0);
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
  EXPECT_TRUE(EkfState::isPositiveDefinite(p3));
  EXPECT_TRUE(p3.allFinite());
}

TEST(EkfStatePropagation, T16NoiseScaleGuardNeutralisesBadValues)
{
  const Eigen::MatrixXd neutral = propagate_one_step(1.0);
  // A scale < 1 or non-finite would shrink/poison Q; the implementation
  // neutralises it to 1.0, so the covariance is bit-identical to the neutral
  // step (this is what "recovered" propagation must look like).
  for (double bad :
       {0.5, 0.0, -3.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    EXPECT_EQ((propagate_one_step(bad) - neutral).cwiseAbs().maxCoeff(), 0.0) << "bad noise scale " << bad;
  }
}


// ------------------------------------------------ T17-T22: deep + cooldown --

namespace
{
// Explicit config so the tests do not depend on the aggregate-init field order.
StarvationMonitor::Config_s starvation_config(
  double _max_no_update_seconds, double _starvation_seconds, double _noise_scale,
  double _deep_starvation_seconds, double _deep_noise_scale, double _restart_cooldown_seconds)
{
  StarvationMonitor::Config_s config;
  config.max_no_update_seconds_ = _max_no_update_seconds;
  config.starvation_seconds_ = _starvation_seconds;
  config.noise_scale_ = _noise_scale;
  config.deep_starvation_seconds_ = _deep_starvation_seconds;
  config.deep_noise_scale_ = _deep_noise_scale;
  config.restart_cooldown_seconds_ = _restart_cooldown_seconds;
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
  EXPECT_EQ(monitor.observeFailedFrame(0.2).deep_escalated_, false);
  EXPECT_FALSE(monitor.starvationActive());
  EXPECT_DOUBLE_EQ(monitor.noiseScale(), 1.0);

  // Between starvation_seconds and deep_starvation_seconds: base scale.
  EXPECT_EQ(monitor.observeFailedFrame(0.5).deep_escalated_, false);
  EXPECT_TRUE(monitor.starvationActive());
  EXPECT_FALSE(monitor.deepStarvationActive());
  EXPECT_DOUBLE_EQ(monitor.noiseScale(), 3.0);
  EXPECT_EQ(monitor.deepEscalations(), 0);

  // Exactly at deep_starvation_seconds: strict '>' keeps it at the base scale.
  EXPECT_EQ(monitor.observeFailedFrame(1.0).deep_escalated_, false);
  EXPECT_DOUBLE_EQ(monitor.noiseScale(), 3.0);
  EXPECT_EQ(monitor.deepEscalations(), 0);

  // Beyond deep_starvation_seconds: escalated, reported exactly once.
  const StarvationMonitor::Decision_s deep = monitor.observeFailedFrame(1.5);
  EXPECT_TRUE(deep.deep_escalated_);
  EXPECT_TRUE(monitor.deepStarvationActive());
  EXPECT_DOUBLE_EQ(monitor.noiseScale(), 10.0);
  EXPECT_EQ(monitor.deepEscalations(), 1);

  // Still deep on the next starved frame, but not a *new* escalation.
  EXPECT_FALSE(monitor.observeFailedFrame(1.6).deep_escalated_);
  EXPECT_DOUBLE_EQ(monitor.noiseScale(), 10.0);
  EXPECT_EQ(monitor.deepEscalations(), 1);

  // A successful update resets everything back to 1.0 immediately.
  monitor.noteAppliedUpdate(1.6);
  EXPECT_FALSE(monitor.starvationActive());
  EXPECT_FALSE(monitor.deepStarvationActive());
  EXPECT_DOUBLE_EQ(monitor.noiseScale(), 1.0);
  // 1.7 - 1.6 is not exact in binary, so compare with a tolerance.
  EXPECT_NEAR(monitor.observeFailedFrame(1.7).gap_seconds_, 0.1, 1e-12);
  EXPECT_DOUBLE_EQ(monitor.noiseScale(), 1.0);
}

TEST(StarvationMonitor, T18DeepEscalationCanBeDisabled)
{
  StarvationMonitor monitor;
  // deep_starvation_seconds <= 0 disables the escalation; the base scale still
  // applies for an arbitrarily long starvation.
  monitor.configure(starvation_config(1000.0, 0.25, 3.0, 0.0, 10.0, 0.0));
  monitor.reset(0.0);

  const StarvationMonitor::Decision_s deep = monitor.observeFailedFrame(50.0);
  EXPECT_FALSE(deep.deep_escalated_);
  EXPECT_FALSE(monitor.deepStarvationActive());
  EXPECT_TRUE(monitor.starvationActive());
  EXPECT_DOUBLE_EQ(monitor.noiseScale(), 3.0);
  EXPECT_EQ(monitor.deepEscalations(), 0);
}

TEST(StarvationMonitor, T19RestartCooldownSuppressesTheTimeCriterion)
{
  StarvationMonitor monitor;
  monitor.configure(starvation_config(1.0, 0.25, 3.0, 1.0, 10.0, 8.0));
  monitor.reset(0.0);

  // First starvation stretch fires the time criterion normally.
  const StarvationMonitor::Decision_s fired = monitor.observeFailedFrame(1.5);
  EXPECT_TRUE(fired.restart_);
  EXPECT_FALSE(fired.cooldown_suppressed_);
  EXPECT_EQ(monitor.timeGateTriggers(), 1);
  monitor.noteRestart(1.5);

  // Inside the 8 s window the trigger is real but must not restart.
  const StarvationMonitor::Decision_s held = monitor.observeFailedFrame(6.5);
  EXPECT_FALSE(held.restart_);
  EXPECT_TRUE(held.cooldown_suppressed_);
  EXPECT_DOUBLE_EQ(held.gap_seconds_, 6.5);
  EXPECT_EQ(monitor.timeGateTriggers(), 1);
  EXPECT_EQ(monitor.cooldownSuppressions(), 1);
  EXPECT_TRUE(monitor.cooldownActive(1.5));
  EXPECT_TRUE(monitor.cooldownActive(9.4));   // 7.9 s < 8 s
  EXPECT_FALSE(monitor.cooldownActive(9.5));  // 8.0 s is not < 8 s

  // Once the cooldown has elapsed a restart fires again.
  const StarvationMonitor::Decision_s again = monitor.observeFailedFrame(9.5);
  EXPECT_TRUE(again.restart_);
  EXPECT_FALSE(again.cooldown_suppressed_);
  EXPECT_EQ(monitor.timeGateTriggers(), 2);
  EXPECT_EQ(monitor.cooldownSuppressions(), 1);
}

TEST(StarvationMonitor, T20CooldownSurvivesResetButNotClockRollback)
{
  StarvationMonitor monitor;
  monitor.configure(starvation_config(1.0, 0.25, 3.0, 1.0, 10.0, 8.0));
  monitor.reset(0.0);

  EXPECT_TRUE(monitor.observeFailedFrame(1.5).restart_);
  monitor.noteRestart(1.5);

  // A controlled restart runs clearState() and initSchurBackend(), both of
  // which call reset(); the anchor must survive that re-seed or the cooldown
  // could never prevent a restart storm.
  monitor.reset(1.6);
  EXPECT_TRUE(monitor.cooldownActive(1.6));
  const StarvationMonitor::Decision_s held = monitor.observeFailedFrame(3.0);
  EXPECT_FALSE(held.restart_);
  EXPECT_TRUE(held.cooldown_suppressed_);
  // reset() is a re-seed, not a counter reset: the cumulative diagnostic
  // counters (and the cooldown anchor) survive it (see T23).
  EXPECT_EQ(monitor.timeGateTriggers(), 1);
  EXPECT_EQ(monitor.cooldownSuppressions(), 1);

  // A backwards timestamp (bag replay) drops the anchor: keeping it would keep
  // the cooldown active for the whole replay.
  StarvationMonitor replay;
  replay.configure(starvation_config(1.0, 0.25, 3.0, 1.0, 10.0, 8.0));
  replay.reset(500.0);
  replay.noteRestart(500.0);
  EXPECT_TRUE(replay.cooldownActive(501.0));
  const StarvationMonitor::Decision_s rollback = replay.observeFailedFrame(100.0);
  EXPECT_TRUE(rollback.rebaselined_);
  EXPECT_FALSE(rollback.restart_);
  EXPECT_FALSE(replay.cooldownActive(100.0));
  EXPECT_DOUBLE_EQ(replay.lastUpdateTime(), 100.0);
}

TEST(StarvationMonitor, T21CooldownCanBeDisabled)
{
  StarvationMonitor monitor;
  // restart_cooldown_seconds == 0 disables the cooldown: every trigger fires,
  // i.e. the pre-P4 behaviour is preserved.
  monitor.configure(starvation_config(1.0, 0.25, 3.0, 1.0, 10.0, 0.0));
  monitor.reset(0.0);

  EXPECT_TRUE(monitor.observeFailedFrame(1.5).restart_);
  monitor.noteRestart(1.5);
  EXPECT_FALSE(monitor.cooldownActive(1.5));
  const StarvationMonitor::Decision_s second = monitor.observeFailedFrame(3.0);
  EXPECT_TRUE(second.restart_);
  EXPECT_FALSE(second.cooldown_suppressed_);
  EXPECT_EQ(monitor.timeGateTriggers(), 2);
  EXPECT_EQ(monitor.cooldownSuppressions(), 0);
}

TEST(StarvationMonitor, T22FrameCountBackstopIsCooldownSuppressed)
{
  // Pure-logic mirror of the decision rule in Estimator::schurProcessFrame:
  // the frame-count backstop restarts only when the streak reaches the limit
  // AND the monitor's cooldown is inactive. (The Estimator itself needs ROS, so
  // the rule is locked down here against the monitor's public predicate.)
  const auto backstop_would_restart =
    [](const StarvationMonitor & _monitor, double _timestamp, int _streak, int _limit)
  { return !_monitor.cooldownActive(_timestamp) && _streak >= _limit; };

  StarvationMonitor monitor;
  // Time criterion disabled so only the frame-count backstop is in play.
  monitor.configure(starvation_config(0.0, 0.25, 3.0, 1.0, 10.0, 8.0));
  monitor.reset(0.0);
  monitor.noteRestart(0.0);

  // 60 consecutive failures inside the cooldown: the backstop is held back.
  EXPECT_FALSE(backstop_would_restart(monitor, 1.0, 60, 60));
  EXPECT_FALSE(backstop_would_restart(monitor, 7.9, 120, 60));
  // ... and it is the cooldown, not the streak, that holds it back.
  EXPECT_FALSE(backstop_would_restart(monitor, 1.0, 59, 60));
  // Past the cooldown the backstop fires again.
  EXPECT_TRUE(backstop_would_restart(monitor, 9.0, 60, 60));

  // The time criterion never fires here (it is disabled), so the frame-count
  // path cannot be confused with a suppression of the time criterion.
  for (double t : {1.0, 2.0, 3.0}) {
    const StarvationMonitor::Decision_s d = monitor.observeFailedFrame(t);
    EXPECT_FALSE(d.restart_);
    EXPECT_FALSE(d.cooldown_suppressed_);
  }
  EXPECT_EQ(monitor.cooldownSuppressions(), 0);
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
    EXPECT_TRUE(monitor.observeFailedFrame(1.5).restart_);
    monitor.reset(10.0);
    EXPECT_EQ(monitor.timeGateTriggers(), 1);
    EXPECT_TRUE(monitor.observeFailedFrame(11.5).restart_);
    monitor.reset(20.0);
    EXPECT_EQ(monitor.timeGateTriggers(), 2);
  }

  // Deep escalations accumulate across resets (time criterion pushed away so
  // this monitor only counts the deep stage).
  {
    StarvationMonitor monitor;
    monitor.configure(starvation_config(100.0, 0.25, 3.0, 1.0, 10.0, 0.0));
    monitor.reset(0.0);
    EXPECT_TRUE(monitor.observeFailedFrame(1.5).deep_escalated_);
    monitor.reset(10.0);
    EXPECT_EQ(monitor.deepEscalations(), 1);
    EXPECT_TRUE(monitor.observeFailedFrame(11.5).deep_escalated_);
    monitor.reset(20.0);
    EXPECT_EQ(monitor.deepEscalations(), 2);
  }

  // Cooldown suppressions accumulate across resets too. deep_starvation is
  // pushed away so the suppressed frame does not also bump the deep counter.
  {
    StarvationMonitor monitor;
    monitor.configure(starvation_config(1.0, 0.25, 3.0, 100.0, 10.0, 8.0));
    monitor.reset(0.0);
    EXPECT_TRUE(monitor.observeFailedFrame(1.5).restart_);
    monitor.noteRestart(1.5);
    EXPECT_TRUE(monitor.observeFailedFrame(4.0).cooldown_suppressed_);
    monitor.reset(4.0);
    EXPECT_EQ(monitor.cooldownSuppressions(), 1);
    EXPECT_EQ(monitor.timeGateTriggers(), 1);
    // The 8 s cooldown from 1.5 s ends at 9.5 s, so 9.6 s fires again.
    EXPECT_TRUE(monitor.observeFailedFrame(9.6).restart_);
    monitor.noteRestart(9.6);
    EXPECT_TRUE(monitor.observeFailedFrame(12.0).cooldown_suppressed_);
    monitor.reset(20.0);
    EXPECT_EQ(monitor.cooldownSuppressions(), 2);
    EXPECT_EQ(monitor.timeGateTriggers(), 2);
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
  EkfOptions_s options = test_options();
  options.window_size_ = 5;
  options.init_velocity_sigma_ = 0.5;

  EkfState state;
  state.reset(
    options, 0.0, Vector4d(1, 0, 0, 0), Vector3d::Zero(), Vector3d::Zero(), Vector3d::Zero(), Vector3d::Zero());
  for (int k = 0; k < 2; ++k) {
    for (int s = 0; s < 5; ++s)
      state.propagate(Vector3d(0.2, 0.0, 9.8), Vector3d(0.0, 0.1, 0.0), 0.01);
    state.augmentClone(0.05 * (k + 1));
  }
  ASSERT_EQ(state.numClones(), 2);
  const ImuNominal_s before = state.imu();
  // The propagation must have moved the nominal state before the re-seed.
  EXPECT_GT(before.v_.norm(), 0.0);
  EXPECT_GT(before.q_.head<3>().norm(), 0.0);

  // Re-seed exactly as schurSoftRestart() does.
  state.reset(state.options(), 2.0, before.q_, before.p_, before.v_, before.ba_, before.bg_);

  const ImuNominal_s & after = state.imu();
  EXPECT_EQ(state.numClones(), 0);
  EXPECT_EQ(state.size(), K_IMU_STATE_SIZE);
  EXPECT_DOUBLE_EQ(state.timestamp(), 2.0);
  EXPECT_NEAR((after.q_ - before.q_).norm(), 0.0, 1e-15);
  EXPECT_NEAR((after.p_ - before.p_).norm(), 0.0, 1e-15);
  EXPECT_NEAR((after.v_ - before.v_).norm(), 0.0, 1e-15);
  EXPECT_NEAR((after.ba_ - before.ba_).norm(), 0.0, 1e-15);
  EXPECT_NEAR((after.bg_ - before.bg_).norm(), 0.0, 1e-15);
  // The FEJ point follows the preserved state.
  EXPECT_NEAR((state.imuFej().p_ - before.p_).norm(), 0.0, 1e-15);

  // P is back to the init_*_sigma diagonal (the old covariance is gone).
  const Eigen::MatrixXd & P = state.covariance();
  EXPECT_EQ(P.rows(), K_IMU_STATE_SIZE);
  EXPECT_EQ(P.cols(), K_IMU_STATE_SIZE);
  EXPECT_NEAR(P(K_IMU_THETA, K_IMU_THETA), options.init_rotation_sigma_ * options.init_rotation_sigma_, 1e-15);
  EXPECT_NEAR(P(K_IMU_P, K_IMU_P), options.init_position_sigma_ * options.init_position_sigma_, 1e-15);
  EXPECT_NEAR(P(K_IMU_V, K_IMU_V), options.init_velocity_sigma_ * options.init_velocity_sigma_, 1e-15);
  EXPECT_NEAR(P(K_IMU_BA, K_IMU_BA), options.init_acc_bias_sigma_ * options.init_acc_bias_sigma_, 1e-15);
  EXPECT_NEAR(P(K_IMU_BG, K_IMU_BG), options.init_gyr_bias_sigma_ * options.init_gyr_bias_sigma_, 1e-15);
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
  using Action = StarvationRestartAction_e;

  // At/after the limit: soft mode -> kSoft, otherwise kFull.
  EXPECT_EQ(frameCountBackstopAction(false, false, 60, 60, true), Action::K_SOFT);
  EXPECT_EQ(frameCountBackstopAction(false, false, 60, 60, false), Action::K_FULL);
  EXPECT_EQ(frameCountBackstopAction(false, false, 600, 60, true), Action::K_SOFT);
  EXPECT_EQ(frameCountBackstopAction(false, false, 600, 60, false), Action::K_FULL);

  // Below the limit: nothing happens in either mode.
  EXPECT_EQ(frameCountBackstopAction(false, false, 59, 60, true), Action::K_NONE);
  EXPECT_EQ(frameCountBackstopAction(false, false, 59, 60, false), Action::K_NONE);

  // The time criterion already restarted this frame: backstop is skipped.
  EXPECT_EQ(frameCountBackstopAction(true, false, 600, 60, true), Action::K_NONE);
  EXPECT_EQ(frameCountBackstopAction(true, false, 600, 60, false), Action::K_NONE);

  // The P4 cooldown holds the backstop back in both modes.
  EXPECT_EQ(frameCountBackstopAction(false, true, 600, 60, true), Action::K_NONE);
  EXPECT_EQ(frameCountBackstopAction(false, true, 600, 60, false), Action::K_NONE);
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

  auto scale = [](double _gap, bool _enabled, double _s, double _d, double _m)
  { return weakUpdateScale(_gap, _enabled, _s, _d, _m); };
  auto def = [&](double _gap) { return scale(_gap, true, starve, deep, max_scale); };

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
  std::printf(
    "[T27] scale(0.25)=%.3f scale(0.4)=%.3f scale(0.625)=%.3f scale(1.0)=%.3f\n", def(0.25), def(0.4),
    def(0.625), def(1.0));

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
  EXPECT_DOUBLE_EQ(scale(5.0, true, starve, deep, 0.5), 1.0);  // max_scale < 1
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
  EkfOptions_s options = test_options();
  options.window_size_ = 5;
  EkfState state;
  state.reset(
    options, 0.0, Vector4d(1, 0, 0, 0), Vector3d::Zero(), Vector3d::Zero(), Vector3d::Zero(), Vector3d::Zero());
  for (int k = 0; k < 4; ++k) {
    for (int s = 0; s < 5; ++s)
      state.propagate(Vector3d(0.1, 0.0, 9.8), Vector3d(0.02, 0.01, -0.03), 0.01);
    state.augmentClone(0.05 * (k + 1));
  }
  ASSERT_EQ(state.numClones(), 4);

  const Mat15 live_before = state.covariance().topLeftCorner(K_IMU_STATE_SIZE, K_IMU_STATE_SIZE);
  const ImuNominal_s imu_before = state.imu();

  EXPECT_EQ(state.dropAllClones(), 4);
  EXPECT_EQ(state.numClones(), 0);
  EXPECT_EQ(state.size(), K_IMU_STATE_SIZE);
  EXPECT_EQ(state.covariance().rows(), K_IMU_STATE_SIZE);
  EXPECT_EQ(state.covariance().cols(), K_IMU_STATE_SIZE);

  // Bit-for-bit: no reset, no re-inflation, no Schur-complement shrinkage.
  EXPECT_LT((state.covariance() - live_before).cwiseAbs().maxCoeff(), 1e-15);
  EXPECT_NEAR((state.imu().p_ - imu_before.p_).norm(), 0.0, 1e-15);
  EXPECT_NEAR((state.imu().v_ - imu_before.v_).norm(), 0.0, 1e-15);
  EXPECT_NEAR((state.imu().q_ - imu_before.q_).norm(), 0.0, 1e-15);
  EXPECT_TRUE(EkfState::isPositiveDefinite(state.covariance()));
  // Idempotent.
  EXPECT_EQ(state.dropAllClones(), 0);

  // The reduced state keeps propagating normally after the drop.
  const ImuNominal_s before_propagation = state.imu();
  state.propagate(Vector3d(0.0, 0.0, 9.8), Vector3d(0.0, 0.0, 0.0), 0.01);
  EXPECT_GT(state.covariance().trace(), live_before.trace());
  EXPECT_GT((state.imu().p_ - before_propagation.p_).norm(), 0.0);
}

// ===========================================================================
// T29 -- P7 hold state machine: consecutive recovered frames are required.
// ===========================================================================
TEST(StarvationHold, T29StateMachineRequiresConsecutiveRecoveredFrames)
{
  StarvationHold::Config_s config;
  config.enabled_ = true;
  config.recovery_min_features_ = 12;
  config.recovery_frames_ = 2;
  StarvationHold hold;
  hold.configure(config);
  EXPECT_TRUE(hold.enabled());
  EXPECT_FALSE(hold.active());
  // A frame outside hold reports kNone and is not counted.
  EXPECT_EQ(hold.observeFrame(0.0, 100), StarvationHold::Action_e::K_NONE);
  EXPECT_EQ(hold.frames(), 0);

  // Entering hold records one episode; re-entering is a no-op.
  EXPECT_TRUE(hold.enter(10.0));
  EXPECT_TRUE(hold.active());
  EXPECT_FALSE(hold.enter(10.1));
  EXPECT_EQ(hold.episodes(), 1);
  EXPECT_DOUBLE_EQ(hold.currentDuration(12.5), 2.5);

  // Too few observations break the streak and keep holding.
  EXPECT_EQ(hold.observeFrame(11.0, 11), StarvationHold::Action_e::K_HOLD);
  EXPECT_EQ(hold.goodStreak(), 0);
  EXPECT_EQ(hold.observeFrame(11.1, 12), StarvationHold::Action_e::K_HOLD);
  EXPECT_EQ(hold.goodStreak(), 1);
  EXPECT_EQ(hold.observeFrame(11.2, 5), StarvationHold::Action_e::K_HOLD);
  EXPECT_EQ(hold.goodStreak(), 0);
  EXPECT_EQ(hold.observeFrame(11.3, 50), StarvationHold::Action_e::K_HOLD);
  EXPECT_EQ(hold.observeFrame(11.4, 50), StarvationHold::Action_e::K_RECOVER);
  EXPECT_EQ(hold.frames(), 5);
  EXPECT_EQ(hold.goodStreak(), 2);

  // Leaving hold records the duration; later frames are not hold frames.
  EXPECT_DOUBLE_EQ(hold.leave(11.5), 1.5);
  EXPECT_FALSE(hold.active());
  EXPECT_DOUBLE_EQ(hold.lastDuration(), 1.5);
  EXPECT_DOUBLE_EQ(hold.totalDuration(), 1.5);
  EXPECT_DOUBLE_EQ(hold.maxDuration(), 1.5);
  EXPECT_EQ(hold.observeFrame(11.6, 100), StarvationHold::Action_e::K_NONE);
  EXPECT_EQ(hold.frames(), 5);

  // The recovery streak does not leak into the next episode.
  EXPECT_TRUE(hold.enter(20.0));
  EXPECT_EQ(hold.goodStreak(), 0);
  EXPECT_EQ(hold.episodes(), 2);
  // recovery_frames < 1 is clamped to 1.
  config.recovery_frames_ = 0;
  hold.configure(config);
  EXPECT_EQ(hold.observeFrame(20.1, 12), StarvationHold::Action_e::K_RECOVER);
  hold.leave(20.2);
  // A shorter episode must not lower the recorded maximum.
  EXPECT_NEAR(hold.totalDuration(), 1.7, 1e-12);
  EXPECT_DOUBLE_EQ(hold.maxDuration(), 1.5);

  // recovery_min_features <= 0 accepts any frame (even one with no tracks).
  config.recovery_min_features_ = 0;
  hold.configure(config);
  EXPECT_TRUE(hold.enter(21.0));
  EXPECT_EQ(hold.episodes(), 3);
  EXPECT_EQ(hold.observeFrame(21.1, 0), StarvationHold::Action_e::K_RECOVER);
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
  using Action = StarvationRestartAction_e;
  // At/after the limit: hold instead of soft/full, even when both are on.
  EXPECT_EQ(frameCountBackstopAction(false, false, 60, 60, false, true), Action::K_HOLD);
  EXPECT_EQ(frameCountBackstopAction(false, false, 600, 60, true, true), Action::K_HOLD);
  EXPECT_EQ(frameCountBackstopAction(false, false, 60, 60, true, true), Action::K_HOLD);
  // Same suppression rules as the restart paths.
  EXPECT_EQ(frameCountBackstopAction(true, false, 600, 60, false, true), Action::K_NONE);
  EXPECT_EQ(frameCountBackstopAction(false, true, 600, 60, false, true), Action::K_NONE);
  EXPECT_EQ(frameCountBackstopAction(false, false, 59, 60, false, true), Action::K_NONE);
  // The five-argument call keeps the historical semantics.
  EXPECT_EQ(frameCountBackstopAction(false, false, 60, 60, false), Action::K_FULL);
}

// ===========================================================================
// T31 -- P7 hold exit re-anchors the baseline but keeps the diagnostics.
// ===========================================================================
TEST(StarvationMonitor, T31RebaselineKeepsMaximumGapAndCounters)
{
  StarvationMonitor::Config_s config;
  config.max_no_update_seconds_ = 3.0;
  config.starvation_seconds_ = 0.25;
  config.noise_scale_ = 3.0;
  StarvationMonitor monitor;
  monitor.configure(config);
  monitor.reset(10.0);

  monitor.observeFailedFrame(10.1);
  const StarvationMonitor::Decision_s decision = monitor.observeFailedFrame(14.0);
  EXPECT_TRUE(decision.restart_);
  EXPECT_EQ(monitor.timeGateTriggers(), 1);

  // Hold exit: re-anchor without claiming an update was applied.
  monitor.rebaseline(14.0);
  EXPECT_DOUBLE_EQ(monitor.starvationDuration(14.0), 0.0);
  EXPECT_FALSE(monitor.starvationActive());
  // The hold duration is folded into the diagnostic maximum.
  EXPECT_DOUBLE_EQ(monitor.maxGapSeconds(), 4.0);
  // Counters survive and a full grace period is available again.
  EXPECT_EQ(monitor.timeGateTriggers(), 1);
  EXPECT_FALSE(monitor.observeFailedFrame(16.9).restart_);
  EXPECT_TRUE(monitor.observeFailedFrame(17.1).restart_);
  EXPECT_EQ(monitor.timeGateTriggers(), 2);
}

}  // namespace backend
}  // namespace vins
