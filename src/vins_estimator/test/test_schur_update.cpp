/*******************************************************
 * Unit tests for the ARM-oriented Schur-complement visual update
 * (src/vins_estimator/src/backend/schur_update.{h,cpp}).
 *
 * Standalone: gtest + Eigen only, no ROS / Ceres / OpenCV.
 *
 *   g++ -std=c++17 -O2 -Wall -Wextra -I src/vins_estimator/src -I /usr/include/eigen3 \
 *       src/vins_estimator/src/backend/ekf_state.cpp \
 *       src/vins_estimator/src/backend/schur_update.cpp \
 *       src/vins_estimator/test/test_schur_update.cpp \
 *       -lgtest -lgtest_main -pthread -o /tmp/test_schur_update
 *
 * Conventions: q = [w,x,y,z] (Hamilton, body -> world), right/body
 * perturbation, error state [dtheta, dp, dv, dba, dbg] + 6 per clone.
 *******************************************************/

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include "backend/feature_history.h"
#include "backend/schur_update.h"

namespace
{

using namespace vins::backend;

constexpr double K_FOCAL = 460.0;
constexpr double K_PIXEL_NOISE = 1.5;

UpdateOptions_s default_options()
{
  UpdateOptions_s o;
  o.max_features_ = 120;
  o.min_observations_ = 2;
  // The algorithm-level tests below use one or very few tracks.  Keep the
  // production frame-level starvation gates from turning those tests into
  // no-op updates; the dedicated gate test overrides these values.
  o.min_features_ = 1;
  o.min_update_observations_ = 1;
  o.pixel_noise_ = K_PIXEL_NOISE;
  o.focal_length_ = K_FOCAL;
  o.chi2_multiplier_ = 5.0;
  o.triangulation_iterations_ = 5;
  o.min_depth_ = 0.1;
  o.max_depth_ = 60.0;
  o.condition_limit_ = 1e6;
  return o;
}

EkfOptions_s default_ekf_options(int _window_size)
{
  EkfOptions_s o;
  o.window_size_ = _window_size;
  return o;
}

Eigen::Vector4d quat(const Eigen::Vector3d & _axis_angle)
{
  return EkfState::quaternionFromAxisAngle(_axis_angle);
}

Eigen::Vector4d random_quat(std::mt19937 & _rng, double _max_angle)
{
  std::uniform_real_distribution<double> d(-_max_angle, _max_angle);
  return quat(Eigen::Vector3d(d(_rng), d(_rng), d(_rng)));
}

// Projection of a world point into the camera attached to the clone (q, p).
Eigen::Vector2d project_world(
  const Eigen::Vector4d & _q, const Eigen::Vector3d & _p, const CameraExtrinsics_s & _cam,
  const Eigen::Vector3d & _world_point)
{
  const Eigen::Matrix3d R_i = EkfState::rotation(_q);
  const Eigen::Matrix3d R_cb = EkfState::rotation(_cam.q_cb_);
  const Eigen::Vector3d u = R_i.transpose() * (_world_point - _p);
  const Eigen::Vector3d p_c = R_cb.transpose() * (u - _cam.p_cb_);
  return Eigen::Vector2d(p_c.x() / p_c.z(), p_c.y() / p_c.z());
}

// Independent (test local) weighted residual model: (uv - h)/sigma.
Eigen::Vector2d weighted_residual_model(
  const Eigen::Vector4d & _q, const Eigen::Vector3d & _p, const CameraExtrinsics_s & _cam,
  const Eigen::Vector3d & _world_point, const Eigen::Vector2d & _uv, double _sigma)
{
  return (_uv - project_world(_q, _p, _cam, _world_point)) / _sigma;
}

// Inverse of a unit quaternion.  Note that Eigen::Vector4d::conjugate()
// conjugates each scalar entry (a no-op for real doubles); it does NOT negate
// the vector part, so it cannot be used as a quaternion inverse.
Eigen::Vector4d quat_inverse(const Eigen::Vector4d & _q)
{
  Eigen::Vector4d inv = _q;
  inv.tail<3>() *= -1.0;
  return inv;
}

// Exact inverse of EkfState::apply_pose_correction's first-order update
//   q_after = q_before (x) [1, dtheta/2] / norm  =>  dtheta = 2 * vec / w.
// Using this (rather than the exact log map) removes the O(|dtheta|^2)
// mismatch between the first-order mean update and the reconstruction.
Eigen::Vector3d correction_from_quat(const Eigen::Vector4d & _before, const Eigen::Vector4d & _after)
{
  const Eigen::Vector4d qrel =
    EkfState::quaternionMultiply(quat_inverse(EkfState::quaternionNormalize(_before)), _after);
  if (std::abs(qrel(0)) < 1e-12)
    return EkfState::quaternionToAxisAngle(qrel);
  return 2.0 * qrel.tail<3>() / qrel(0);
}

// Rotation vector of a quaternion (inverse of quaternion_from_axis_angle).
Eigen::Vector3d log_quat(const Eigen::Vector4d & _q_in)
{
  const Eigen::Vector4d q = _q_in.normalized();
  const double w = std::max(-1.0, std::min(1.0, q(0)));  // [w,x,y,z], not Eigen .w()
  const double s = std::sqrt(std::max(0.0, 1.0 - w * w));
  if (s < 1e-12)
    return Eigen::Vector3d::Zero();
  const double angle = 2.0 * std::atan2(s, w);
  return (angle / s) * q.tail<3>();
}

// Builds a state whose clones sit exactly at the given poses (window large
// enough that nothing is marginalised) and whose live pose is the last one.
void build_state(
  EkfState & _state, const std::vector<Eigen::Vector4d> & _qs, const std::vector<Eigen::Vector3d> & _ps,
  double _vel_noise = 0.0, std::mt19937 * _rng = nullptr)
{
  ASSERT_EQ(_qs.size(), _ps.size());
  const int n = static_cast<int>(_qs.size());
  EkfOptions_s opt = default_ekf_options(n + 2);
  Eigen::Vector3d v = Eigen::Vector3d::Zero();
  if (_rng != nullptr && _vel_noise > 0.0) {
    std::uniform_real_distribution<double> d(-_vel_noise, _vel_noise);
    v << d(*_rng), d(*_rng), d(*_rng);
  }
  _state.reset(opt, 0.0, _qs[0], _ps[0], v, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  _state.augmentClone(0.0);
  for (int i = 1; i < n; ++i) {
    _state.imu().q_ = _qs[static_cast<std::size_t>(i)];
    _state.imu().p_ = _ps[static_cast<std::size_t>(i)];
    // This test helper bypasses propagate(), which is what normally advances
    // imu_fej(). Keep the live FEJ synchronized with the pose being cloned.
    _state.refreshImuFej();
    _state.augmentClone(static_cast<double>(i));
  }
}

// A freshly augmented clone is an exact copy of the live pose, so the raw
// covariance has zero-variance live/clone difference directions and is
// singular.  A real filter always adds process noise between frames; emulate
// that with a tiny isotropic jitter so that the information-form reference
// P^-1 used by T4 is well defined and S is not numerically indefinite.
void regularize_covariance(EkfState & _state, double _sigma = 1e-4)
{
  _state.covariance().diagonal().array() += _sigma * _sigma;
}

CameraExtrinsics_s make_camera(const Eigen::Vector3d & _axis_angle, const Eigen::Vector3d & _p_cb)
{
  CameraExtrinsics_s cam;
  cam.q_cb_ = quat(_axis_angle);
  cam.p_cb_ = _p_cb;
  return cam;
}

FeatureObservation_s make_observation(int _clone_index, int _camera_id, const Eigen::Vector2d & _uv)
{
  FeatureObservation_s obs;
  obs.clone_index_ = _clone_index;
  obs.camera_id_ = _camera_id;
  obs.uv_ = _uv;
  return obs;
}

// A straight, smooth camera trajectory in front of a static point cloud.
void smooth_trajectory(int _num_frames, std::vector<Eigen::Vector4d> & _qs, std::vector<Eigen::Vector3d> & _ps)
{
  _qs.clear();
  _ps.clear();
  for (int i = 0; i < _num_frames; ++i) {
    const double t = static_cast<double>(i);
    _ps.push_back(Eigen::Vector3d(0.25 * t, 0.05 * t * t, 0.01 * t));
    _qs.push_back(quat(Eigen::Vector3d(0.0, -0.02 * t, 0.01 * t)));
  }
}

// ---------------------------------------------------------------------------
// Joint (state + point) system of one track, built independently from the
// observation jacobians, used to check the Schur reduction and the EKF update.
// ---------------------------------------------------------------------------
struct JointSystem_s {
  Eigen::MatrixXd Hx_;  // 2M x n
  Eigen::MatrixXd Hf_;  // 2M x 3
  Eigen::VectorXd r_;   // 2M
  int num_obs_ = 0;
};

JointSystem_s build_joint(
  const SchurUpdater & _updater, const EkfState & _state, const std::vector<CameraExtrinsics_s> & _cameras,
  const FeatureTrack_s & _track)
{
  JointSystem_s joint;
  const int m = static_cast<int>(_track.observations_.size());
  const int n = _state.size();
  joint.num_obs_ = m;
  joint.Hx_ = Eigen::MatrixXd::Zero(2 * m, n);
  joint.Hf_ = Eigen::MatrixXd::Zero(2 * m, 3);
  joint.r_ = Eigen::VectorXd::Zero(2 * m);
  for (int j = 0; j < m; ++j) {
    const FeatureObservation_s & obs = _track.observations_[static_cast<std::size_t>(j)];
    const ObservationJacobian_s J = _updater.observationJacobian(
      _state, _cameras[static_cast<std::size_t>(obs.camera_id_)], _track.world_point_, obs);
    const int off = _state.cloneOffset(obs.clone_index_);
    joint.Hx_.block<2, 6>(2 * j, off) = J.H_pose_;
    joint.Hf_.block<2, 3>(2 * j, 0) = J.H_point_;
    joint.r_.segment<2>(2 * j) = J.residual_;
  }
  return joint;
}

// A_red = H_x^T H_x - W A_ff^-1 W^T, b_red = H_x^T r - W A_ff^-1 b_f  (step D
// of the specification, using the blocks reported by reduce_track()).
void reduce_with_blocks(
  const EkfState & _state, const JointSystem_s & _joint, const TrackReduction_s & _red,
  Eigen::MatrixXd & _A_red, Eigen::VectorXd & _b_red)
{
  _A_red = _joint.Hx_.transpose() * _joint.Hx_;
  _b_red = _joint.Hx_.transpose() * _joint.r_;
  const Eigen::Matrix3d Ainv = _red.A_point_.inverse();
  for (const auto & wi : _red.W_) {
    const int oi = _state.cloneOffset(wi.first);
    _b_red.segment<6>(oi) -= wi.second * Ainv * _red.b_point_;
    for (const auto & wj : _red.W_) {
      const int oj = _state.cloneOffset(wj.first);
      _A_red.block<6, 6>(oi, oj) -= wi.second * Ainv * wj.second.transpose();
    }
  }
}

// Exact Schur complement of the joint system (point block eliminated).
void reduce_exact(const JointSystem_s & _joint, Eigen::MatrixXd & _A_ref, Eigen::VectorXd & _b_ref)
{
  const Eigen::MatrixXd A_xx = _joint.Hx_.transpose() * _joint.Hx_;
  const Eigen::MatrixXd A_xf = _joint.Hx_.transpose() * _joint.Hf_;
  const Eigen::MatrixXd A_ff = _joint.Hf_.transpose() * _joint.Hf_;
  const Eigen::VectorXd b_x = _joint.Hx_.transpose() * _joint.r_;
  const Eigen::VectorXd b_f = _joint.Hf_.transpose() * _joint.r_;
  const Eigen::Matrix3d Ainv = A_ff.inverse();
  _A_ref = A_xx - A_xf * Ainv * A_xf.transpose();
  _b_ref = b_x - A_xf * Ainv * b_f;
}

// ---------------------------------------------------------------------------
// State snapshot for exact before/after comparisons.
// ---------------------------------------------------------------------------
struct Snapshot_s {
  Eigen::Vector4d q_;
  Eigen::Vector3d p_, v_, ba_, bg_;
  std::vector<Eigen::Vector4d> clone_q_;
  std::vector<Eigen::Vector3d> clone_p_;
  Eigen::MatrixXd P_;
};

Snapshot_s snapshot(const EkfState & _state)
{
  Snapshot_s s;
  s.q_ = _state.imu().q_;
  s.p_ = _state.imu().p_;
  s.v_ = _state.imu().v_;
  s.ba_ = _state.imu().ba_;
  s.bg_ = _state.imu().bg_;
  for (int i = 0; i < _state.numClones(); ++i) {
    s.clone_q_.push_back(_state.clone(i).q_);
    s.clone_p_.push_back(_state.clone(i).p_);
  }
  s.P_ = _state.covariance();
  return s;
}

double snapshot_max_diff(const Snapshot_s & _a, const Snapshot_s & _b)
{
  double d = std::max((_a.q_ - _b.q_).cwiseAbs().maxCoeff(), (_a.p_ - _b.p_).cwiseAbs().maxCoeff());
  d = std::max(d, (_a.v_ - _b.v_).cwiseAbs().maxCoeff());
  d = std::max(d, (_a.ba_ - _b.ba_).cwiseAbs().maxCoeff());
  d = std::max(d, (_a.bg_ - _b.bg_).cwiseAbs().maxCoeff());
  for (std::size_t i = 0; i < _a.clone_q_.size(); ++i) {
    d = std::max(d, (_a.clone_q_[i] - _b.clone_q_[i]).cwiseAbs().maxCoeff());
    d = std::max(d, (_a.clone_p_[i] - _b.clone_p_[i]).cwiseAbs().maxCoeff());
  }
  if (_a.P_.size() > 0 && _b.P_.size() > 0)
    d = std::max(d, (_a.P_ - _b.P_).cwiseAbs().maxCoeff());
  return d;
}

int clone_theta_offset(const EkfState & _state, int _i) { return _state.cloneOffset(_i); }

bool state_is_finite(const EkfState & _state)
{
  bool finite = _state.imu().q_.allFinite() && _state.imu().p_.allFinite() && _state.imu().v_.allFinite() &&
                _state.imu().ba_.allFinite() && _state.imu().bg_.allFinite() &&
                _state.covariance().allFinite();
  for (int i = 0; i < _state.numClones(); ++i) {
    finite = finite && _state.clone(i).q_.allFinite() && _state.clone(i).p_.allFinite();
  }
  return finite;
}

double min_covariance_eigenvalue(const Eigen::MatrixXd & _P)
{
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(_P);
  return eig.info() == Eigen::Success ? eig.eigenvalues().minCoeff() : -std::numeric_limits<double>::infinity();
}

// Reconstructs the error-state correction dx from the nominal state change.
Eigen::VectorXd reconstruct_dx(const Snapshot_s & _snap, const EkfState & _after)
{
  const int n = _after.size();
  Eigen::VectorXd dx = Eigen::VectorXd::Zero(n);
  dx.segment<3>(K_IMU_THETA) = correction_from_quat(_snap.q_, _after.imu().q_);
  dx.segment<3>(K_IMU_P) = _after.imu().p_ - _snap.p_;
  dx.segment<3>(K_IMU_V) = _after.imu().v_ - _snap.v_;
  dx.segment<3>(K_IMU_BA) = _after.imu().ba_ - _snap.ba_;
  dx.segment<3>(K_IMU_BG) = _after.imu().bg_ - _snap.bg_;
  for (int i = 0; i < _after.numClones(); ++i) {
    dx.segment<3>(clone_theta_offset(_after, i)) =
      correction_from_quat(_snap.clone_q_[static_cast<std::size_t>(i)], _after.clone(i).q_);
    dx.segment<3>(clone_theta_offset(_after, i) + 3) =
      _after.clone(i).p_ - _snap.clone_p_[static_cast<std::size_t>(i)];
  }
  return dx;
}

// Removes the 6-dof global rigid gauge from a per-clone perturbation field
// (gauge rotation dphi: dtheta_i = R_i^T dphi, dp_i = dphi x p_i; gauge
// translation dt: dp_i = dt).  The visual system cannot observe this motion,
// so a purely observable perturbation is used in the convergence test.
void remove_gauge(
  const std::vector<Eigen::Vector4d> & _qs, const std::vector<Eigen::Vector3d> & _ps,
  std::vector<Eigen::Vector3d> & _dtheta, std::vector<Eigen::Vector3d> & _dp)
{
  const int n = static_cast<int>(_qs.size());
  Eigen::MatrixXd G = Eigen::MatrixXd::Zero(6 * n, 6);
  Eigen::VectorXd v(6 * n);
  for (int i = 0; i < n; ++i) {
    const Eigen::Matrix3d R = EkfState::rotation(_qs[static_cast<std::size_t>(i)]);
    const Eigen::Vector3d & p = _ps[static_cast<std::size_t>(i)];
    for (int k = 0; k < 3; ++k) {
      const Eigen::Vector3d e = Eigen::Vector3d::Unit(k);
      G.block<3, 1>(6 * i, k) = R.transpose() * e;
      G.block<3, 1>(6 * i + 3, k) = e.cross(p);
      G.block<3, 1>(6 * i + 3, 3 + k) = e;
    }
    v.segment<3>(6 * i) = _dtheta[static_cast<std::size_t>(i)];
    v.segment<3>(6 * i + 3) = _dp[static_cast<std::size_t>(i)];
  }
  const Eigen::VectorXd coef = (G.transpose() * G).ldlt().solve(G.transpose() * v);
  v -= G * coef;
  for (int i = 0; i < n; ++i) {
    _dtheta[static_cast<std::size_t>(i)] = v.segment<3>(6 * i);
    _dp[static_cast<std::size_t>(i)] = v.segment<3>(6 * i + 3);
  }
}


// Sets an isotropic, block-diagonal prior on the live pose and every clone.
// After augment_clone() the raw covariance is singular (a fresh clone is a
// copy of the live pose).  A real filter adds process noise between frames;
// this helper models that as independent pose priors, which keeps the
// multi-frame visual-only convergence test well conditioned.
void set_pose_diagonal_covariance(EkfState & _state, double _rot_sigma_rad, double _pos_sigma_m)
{
  const int n = _state.size();
  _state.covariance().setZero(n, n);
  _state.covariance().block<3, 3>(K_IMU_THETA, K_IMU_THETA) =
    _rot_sigma_rad * _rot_sigma_rad * Eigen::Matrix3d::Identity();
  _state.covariance().block<3, 3>(K_IMU_P, K_IMU_P) = _pos_sigma_m * _pos_sigma_m * Eigen::Matrix3d::Identity();
  _state.covariance().block<3, 3>(K_IMU_V, K_IMU_V) = 0.1 * 0.1 * Eigen::Matrix3d::Identity();
  _state.covariance().block<3, 3>(K_IMU_BA, K_IMU_BA) = 0.1 * 0.1 * Eigen::Matrix3d::Identity();
  _state.covariance().block<3, 3>(K_IMU_BG, K_IMU_BG) = 0.01 * 0.01 * Eigen::Matrix3d::Identity();
  for (int i = 0; i < _state.numClones(); ++i) {
    const int off = _state.cloneOffset(i);
    _state.covariance().block<3, 3>(off, off) = _rot_sigma_rad * _rot_sigma_rad * Eigen::Matrix3d::Identity();
    _state.covariance().block<3, 3>(off + 3, off + 3) =
      _pos_sigma_m * _pos_sigma_m * Eigen::Matrix3d::Identity();
  }
}

// Absolute and gauge-invariant pose errors.  A visual-only problem (no IMU)
// cannot observe the six-dof world gauge.  The relative metric anchors clone 0
// at the true trajectory, so it measures exactly what the images constrain.
void pose_errors(
  const EkfState & _state, const std::vector<Eigen::Vector4d> & _q_true,
  const std::vector<Eigen::Vector3d> & _p_true, bool _relative, double & _rms_pos, double & _rms_rot)
{
  double sq_pos = 0.0;
  double sq_rot = 0.0;
  const Eigen::Matrix3d R0t = EkfState::rotation(_q_true.front());
  const Eigen::Vector3d & p0t = _p_true.front();
  const Eigen::Matrix3d R0e = EkfState::rotation(_state.clone(0).q_);
  const Eigen::Vector3d p0e = _state.clone(0).p_;
  for (int i = 0; i < _state.numClones(); ++i) {
    if (_relative && i == 0)
      continue;
    Eigen::Vector3d p_t = _p_true[static_cast<std::size_t>(i)];
    Eigen::Vector4d q_t = _q_true[static_cast<std::size_t>(i)];
    Eigen::Vector3d p_e = _state.clone(i).p_;
    Eigen::Vector4d q_e = _state.clone(i).q_;
    if (_relative) {
      p_t = R0t.transpose() * (p_t - p0t);
      p_e = R0e.transpose() * (p_e - p0e);
      q_t = EkfState::quaternionMultiply(quat_inverse(_q_true.front()), q_t);
      q_e = EkfState::quaternionMultiply(quat_inverse(_state.clone(0).q_), q_e);
    }
    sq_pos += (p_e - p_t).squaredNorm();
    sq_rot += log_quat(EkfState::quaternionMultiply(quat_inverse(q_t), q_e)).squaredNorm();
  }
  const double count = _relative ? std::max(1, _state.numClones() - 1) : _state.numClones();
  _rms_pos = std::sqrt(sq_pos / count);
  _rms_rot = std::sqrt(sq_rot / count);
}

// ===========================================================================
// T13 -- residual uses the current clone pose, Jacobians use the frozen FEJ.
// ===========================================================================
TEST(SchurUpdate, T13JacobianUsesCloneFejWhileResidualUsesCurrentEstimate)
{
  const UpdateOptions_s opts = default_options();
  SchurUpdater updater(opts);
  const CameraExtrinsics_s cam =
    make_camera(Eigen::Vector3d(0.02, -0.01, 0.03), Eigen::Vector3d(0.01, 0.02, -0.01));
  const double sigma = opts.pixel_noise_ / opts.focal_length_;

  std::vector<Eigen::Vector4d> qs;
  std::vector<Eigen::Vector3d> ps;
  smooth_trajectory(3, qs, ps);
  EkfState state;
  build_state(state, qs, ps);
  regularize_covariance(state);

  // Move the current clone estimates away from their frozen FEJ poses.
  for (int i = 0; i < state.numClones(); ++i) {
    state.clone(i).q_ =
      EkfState::quaternionMultiply(state.clone(i).q_, quat(Eigen::Vector3d(0.004 * (i + 1), -0.003, 0.002)));
    state.clone(i).p_ += Eigen::Vector3d(0.01 * (i + 1), -0.008, 0.006);
  }

  const Eigen::Vector3d world_point(0.8, 0.5, 4.5);
  FeatureTrack_s track;
  track.id_ = 1300;
  for (int i = 0; i < 3; ++i) {
    track.observations_.push_back(
      make_observation(i, 0, project_world(state.clone(i).q_, state.clone(i).p_, cam, world_point)));
  }

  const int ci = 1;
  const FeatureObservation_s & obs = track.observations_[static_cast<std::size_t>(ci)];
  const ObservationJacobian_s J = updater.observationJacobian(state, cam, world_point, obs);
  ASSERT_TRUE(J.valid_);

  const Eigen::Vector2d expected_residual =
    weighted_residual_model(state.clone(ci).q_, state.clone(ci).p_, cam, world_point, obs.uv_, sigma);
  EXPECT_LT((J.residual_ - expected_residual).norm(), 1e-12);

  const Eigen::Vector4d q_fej = state.cloneFej(ci).q_;
  const Eigen::Vector3d p_fej = state.cloneFej(ci).p_;
  const double eps = 1e-6;
  Eigen::Matrix<double, 2, 6> fd_pose = Eigen::Matrix<double, 2, 6>::Zero();
  for (int k = 0; k < 6; ++k) {
    Eigen::Vector4d q_p = q_fej, q_m = q_fej;
    Eigen::Vector3d p_p = p_fej, p_m = p_fej;
    if (k < 3) {
      Eigen::Vector3d d = Eigen::Vector3d::Zero();
      d[k] = eps;
      q_p = EkfState::quaternionMultiply(q_fej, quat(d)).normalized();
      q_m = EkfState::quaternionMultiply(q_fej, quat(-d)).normalized();
    } else {
      p_p[k - 3] += eps;
      p_m[k - 3] -= eps;
    }
    const Eigen::Vector2d r_p = weighted_residual_model(q_p, p_p, cam, world_point, obs.uv_, sigma);
    const Eigen::Vector2d r_m = weighted_residual_model(q_m, p_m, cam, world_point, obs.uv_, sigma);
    fd_pose.col(k) = (r_p - r_m) / (2.0 * eps);
  }
  EXPECT_LT((fd_pose + J.H_pose_).norm() / J.H_pose_.norm(), 1e-7);

  std::vector<CloneNominal_s> frozen_fej(state.numClones());
  for (int i = 0; i < state.numClones(); ++i)
    frozen_fej[static_cast<std::size_t>(i)] = state.cloneFej(i);

  std::vector<FeatureTrack_s> tracks{track};
  const UpdateSummary_s summary = updater.update(state, tracks, {cam});
  EXPECT_TRUE(summary.applied_);
  ASSERT_EQ(summary.consumed_feature_ids_.size(), 1u);
  EXPECT_EQ(summary.consumed_feature_ids_.front(), track.id_);
  for (int i = 0; i < state.numClones(); ++i) {
    EXPECT_LT((state.cloneFej(i).q_ - frozen_fej[static_cast<std::size_t>(i)].q_).norm(), 1e-15);
    EXPECT_LT((state.cloneFej(i).p_ - frozen_fej[static_cast<std::size_t>(i)].p_).norm(), 1e-15);
  }
}

// ===========================================================================
// T14 -- excessive visual information conditioning rejects the whole frame.
// ===========================================================================
TEST(SchurUpdate, T14ConditionLimitRejectsWithoutTouchingState)
{
  UpdateOptions_s opts = default_options();
  opts.condition_limit_ = 1.0;
  SchurUpdater updater(opts);
  const CameraExtrinsics_s cam =
    make_camera(Eigen::Vector3d(0.01, 0.02, -0.01), Eigen::Vector3d(0.02, 0.0, 0.01));
  std::vector<Eigen::Vector4d> qs;
  std::vector<Eigen::Vector3d> ps;
  smooth_trajectory(4, qs, ps);
  EkfState state;
  build_state(state, qs, ps);
  regularize_covariance(state);

  std::vector<FeatureTrack_s> tracks;
  for (int f = 0; f < 8; ++f) {
    FeatureTrack_s track;
    track.id_ = 1400 + f;
    const Eigen::Vector3d world_point(-0.7 + 0.2 * f, 0.15 * (f % 3), 4.0 + 0.1 * f);
    for (int i = 0; i < 4; ++i)
      track.observations_.push_back(make_observation(
        i, 0,
        project_world(qs[static_cast<std::size_t>(i)], ps[static_cast<std::size_t>(i)], cam, world_point)));
    tracks.push_back(track);
  }

  const Snapshot_s before = snapshot(state);
  const UpdateSummary_s summary = updater.update(state, tracks, {cam});
  std::printf(
    "[T14] applied=%d rank=%d cond=%.6e reason=%s\n", summary.applied_ ? 1 : 0, summary.rank_,
    summary.condition_, summary.failure_reason_.c_str());
  EXPECT_FALSE(summary.applied_);
  EXPECT_GT(summary.condition_, opts.condition_limit_);
  EXPECT_NE(summary.failure_reason_.find("condition"), std::string::npos);
  EXPECT_EQ(summary.consumed_feature_ids_.size(), tracks.size());
  EXPECT_LT(snapshot_max_diff(before, snapshot(state)), 1e-15);
}

// ===========================================================================
// T15 -- Joseph output must be PSD; otherwise nothing is written back.
// ===========================================================================
TEST(SchurUpdate, T15NonPsdJosephCovarianceRejectsWithoutWritingBack)
{
  const UpdateOptions_s opts = default_options();
  SchurUpdater updater(opts);
  const CameraExtrinsics_s cam = make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  std::vector<Eigen::Vector4d> qs;
  std::vector<Eigen::Vector3d> ps;
  smooth_trajectory(3, qs, ps);
  EkfState state;
  build_state(state, qs, ps);
  set_pose_diagonal_covariance(state, 0.01, 0.1);

  // An indefinite mode in the live state is not touched by the visual L
  // matrix, so the Joseph result stays indefinite in that direction.
  state.covariance()(K_IMU_P, K_IMU_P) = -1.0;

  const Eigen::Vector3d world_point(0.6, 0.3, 4.2);
  FeatureTrack_s track;
  track.id_ = 1500;
  for (int i = 0; i < 3; ++i)
    track.observations_.push_back(make_observation(
      i, 0, project_world(qs[static_cast<std::size_t>(i)], ps[static_cast<std::size_t>(i)], cam, world_point)));

  const Snapshot_s before = snapshot(state);
  std::vector<FeatureTrack_s> tracks{track};
  const UpdateSummary_s summary = updater.update(state, tracks, {cam});
  std::printf("[T15] applied=%d reason=%s\n", summary.applied_ ? 1 : 0, summary.failure_reason_.c_str());
  EXPECT_FALSE(summary.applied_);
  EXPECT_NE(summary.failure_reason_.find("PSD"), std::string::npos);
  EXPECT_LT(snapshot_max_diff(before, snapshot(state)), 1e-15);
}

// ===========================================================================
// T16 -- consumed_feature_ids covers accepted, rejected and capped tracks.
// ===========================================================================
TEST(SchurUpdate, T16ConsumedFeatureIdsAreComplete)
{
  UpdateOptions_s opts = default_options();
  opts.max_features_ = 1;
  opts.min_features_ = 2;  // force a visually starved frame, but still consume all ids
  SchurUpdater updater(opts);
  const CameraExtrinsics_s cam = make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  std::vector<Eigen::Vector4d> qs;
  std::vector<Eigen::Vector3d> ps;
  smooth_trajectory(3, qs, ps);
  EkfState state;
  build_state(state, qs, ps);
  regularize_covariance(state);

  auto make_track = [&](int _id, const Eigen::Vector3d & _world_point)
  {
    FeatureTrack_s track;
    track.id_ = _id;
    for (int i = 0; i < 3; ++i)
      track.observations_.push_back(make_observation(
        i, 0,
        project_world(qs[static_cast<std::size_t>(i)], ps[static_cast<std::size_t>(i)], cam, _world_point)));
    return track;
  };

  std::vector<FeatureTrack_s> tracks;
  tracks.push_back(make_track(1601, Eigen::Vector3d(0.4, 0.3, 4.0)));
  tracks.push_back(make_track(1602, Eigen::Vector3d(-0.3, 0.5, 4.5)));
  FeatureTrack_s outlier = make_track(1603, Eigen::Vector3d(0.7, -0.2, 3.8));
  outlier.observations_[1].uv_.x() += 50.0 / K_FOCAL;
  tracks.push_back(outlier);
  FeatureTrack_s short_track;
  short_track.id_ = 1604;
  short_track.observations_.push_back(make_observation(0, 0, Eigen::Vector2d(0.1, 0.2)));
  tracks.push_back(short_track);

  const UpdateSummary_s summary = updater.update(state, tracks, {cam});
  std::vector<int> expected{1601, 1602, 1603, 1604};
  std::vector<int> consumed = summary.consumed_feature_ids_;
  std::sort(consumed.begin(), consumed.end());
  EXPECT_EQ(consumed, expected);
  EXPECT_FALSE(summary.applied_);
  EXPECT_NE(summary.failure_reason_.find("insufficient"), std::string::npos);
  // used_feature_ids tracks A/b accumulation, not final acceptance: this
  // frame accumulated one track and was then rejected by the starvation gate.
  ASSERT_EQ(summary.used_feature_ids_.size(), 1u);
  EXPECT_EQ(summary.used_feature_ids_.front(), 1601);
}


// ===========================================================================
// T17 -- only features lost in the newest frame are selected as lost.
// ===========================================================================
TEST(SchurUpdate, T17FeatureHistorySelectsLostOnly)
{
  FeatureHistory history;
  const Eigen::Vector2d uv(0.1, -0.05);
  history.addFrame(10, 0, {{171, uv}, {172, uv}});
  history.addFrame(11, 0, {{172, uv}});

  const FeatureHistory::TrackSelection selection = history.selectTracks(9, 11, 1, 10);
  ASSERT_EQ(selection.tracks_.size(), 1u);
  EXPECT_EQ(selection.tracks_.front().id_, 171);
  EXPECT_EQ(selection.lost_count_, 1);
  EXPECT_EQ(selection.marginalized_count_, 0);
}

// ===========================================================================
// T18 -- a still-tracked feature is selected when it constrains the oldest
// clone that is about to be marginalized.
// ===========================================================================
TEST(SchurUpdate, T18FeatureHistorySelectsMarginalized)
{
  FeatureHistory history;
  const Eigen::Vector2d uv(0.2, 0.1);
  history.addFrame(10, 0, {{181, uv}});
  history.addFrame(11, 0, {{182, uv}});
  history.addFrame(12, 0, {{181, uv}, {182, uv}});

  const FeatureHistory::TrackSelection selection = history.selectTracks(10, 12, 2, 10);
  ASSERT_EQ(selection.tracks_.size(), 1u);
  EXPECT_EQ(selection.tracks_.front().id_, 181);
  EXPECT_EQ(selection.lost_count_, 0);
  EXPECT_EQ(selection.marginalized_count_, 1);
}

// ===========================================================================
// T19 -- trim is clone-window based only; it must not delete a track merely
// because it has been unobserved for more than one frame.
// ===========================================================================
TEST(SchurUpdate, T19FeatureHistoryTrimDoesNotDropTwoFrameOldTracks)
{
  FeatureHistory history;
  const Eigen::Vector2d uv(0.3, -0.2);
  history.addFrame(0, 0, {{191, uv}, {192, uv}});
  history.addFrame(2, 0, {{191, uv}});
  history.addFrame(4, 0, {});  // newest frame: feature 191 is now lost

  history.trim(1, 4);
  ASSERT_EQ(history.numFeatures(), 1u);
  EXPECT_EQ(history.newestFrameIndex(), 4);

  const FeatureHistory::TrackSelection selection = history.selectTracks(1, 4, 1, 10);
  ASSERT_EQ(selection.tracks_.size(), 1u);
  EXPECT_EQ(selection.tracks_.front().id_, 191);
  ASSERT_EQ(selection.tracks_.front().observations_.size(), 1u);
  EXPECT_EQ(selection.tracks_.front().observations_.front().clone_index_, 1);
  EXPECT_EQ(selection.lost_count_, 1);
}

// ===========================================================================
// T20 -- once a lost track is consumed and erased, a later frame cannot select
// it as lost again.
// ===========================================================================
TEST(SchurUpdate, T20FeatureHistoryLostTrackIsNotSelectedAgainAfterErase)
{
  FeatureHistory history;
  const Eigen::Vector2d uv(0.05, 0.15);
  history.addFrame(20, 0, {{201, uv}});
  history.addFrame(21, 0, {});

  const FeatureHistory::TrackSelection first = history.selectTracks(19, 21, 1, 10);
  ASSERT_EQ(first.tracks_.size(), 1u);
  EXPECT_EQ(first.tracks_.front().id_, 201);
  EXPECT_EQ(first.lost_count_, 1);

  std::vector<int> used_ids;
  used_ids.push_back(first.tracks_.front().id_);
  history.eraseFeatures(used_ids);
  history.addFrame(22, 0, {});

  const FeatureHistory::TrackSelection second = history.selectTracks(19, 22, 1, 10);
  EXPECT_TRUE(second.tracks_.empty());
  EXPECT_EQ(second.lost_count_, 0);
  EXPECT_EQ(second.marginalized_count_, 0);
}

// ===========================================================================
// T21 -- used_feature_ids contains exactly the tracks accumulated into A/b,
// never a triangulation/chi2/gating rejection.
// ===========================================================================
TEST(SchurUpdate, T21UsedFeatureIdsDisjointFromRejected)
{
  UpdateOptions_s opts = default_options();
  opts.max_features_ = 10;
  opts.min_features_ = 1;
  opts.min_update_observations_ = 1;
  SchurUpdater updater(opts);
  const CameraExtrinsics_s cam = make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  std::vector<Eigen::Vector4d> qs;
  std::vector<Eigen::Vector3d> ps;
  smooth_trajectory(3, qs, ps);
  EkfState state;
  build_state(state, qs, ps);
  regularize_covariance(state);

  auto make_track = [&](int _id, const Eigen::Vector3d & _world_point, double _outlier_px)
  {
    FeatureTrack_s track;
    track.id_ = _id;
    for (int i = 0; i < 3; ++i) {
      Eigen::Vector2d uv =
        project_world(qs[static_cast<std::size_t>(i)], ps[static_cast<std::size_t>(i)], cam, _world_point);
      if (i == 1)
        uv.x() += _outlier_px / K_FOCAL;
      track.observations_.push_back(make_observation(i, 0, uv));
    }
    return track;
  };

  const FeatureTrack_s outlier = make_track(2102, Eigen::Vector3d(0.7, -0.3, 4.0), 50.0);
  const FeatureTrack_s good = make_track(2101, Eigen::Vector3d(-0.2, 0.4, 4.2), 0.0);
  FeatureTrack_s short_track;
  short_track.id_ = 2103;
  short_track.observations_.push_back(make_observation(0, 0, Eigen::Vector2d(0.1, 0.2)));

  std::vector<FeatureTrack_s> tracks{outlier, good, short_track};
  const UpdateSummary_s summary = updater.update(state, tracks, {cam});
  EXPECT_TRUE(summary.applied_);
  EXPECT_EQ(summary.features_used_, 1);
  ASSERT_EQ(summary.used_feature_ids_.size(), 1u);
  EXPECT_EQ(summary.used_feature_ids_.front(), good.id_);
  EXPECT_EQ(summary.consumed_feature_ids_.size(), tracks.size());

  EXPECT_NE(
    std::find(summary.rejected_feature_ids_.begin(), summary.rejected_feature_ids_.end(), outlier.id_),
    summary.rejected_feature_ids_.end());
  EXPECT_NE(
    std::find(summary.rejected_feature_ids_.begin(), summary.rejected_feature_ids_.end(), short_track.id_),
    summary.rejected_feature_ids_.end());
  for (const int used_id : summary.used_feature_ids_) {
    EXPECT_EQ(
      std::find(summary.rejected_feature_ids_.begin(), summary.rejected_feature_ids_.end(), used_id),
      summary.rejected_feature_ids_.end());
  }
}


// ===========================================================================
// T22 -- marginalization-eligible tracks are selected only after the clone
// window is full. While filling, only tracks lost in the newest frame qualify.
// ===========================================================================
TEST(SchurUpdate, T22WindowFullControlsMarginalizedSelection)
{
  FeatureHistory history;
  const Eigen::Vector2d uv(0.12, -0.08);
  history.addFrame(10, 0, {{221, uv}});
  history.addFrame(11, 0, {{222, uv}});
  history.addFrame(12, 0, {{221, uv}});

  const FeatureHistory::TrackSelection filling = history.tracks(10, 12, 1, 10, false);
  ASSERT_EQ(filling.tracks_.size(), 1u);
  EXPECT_EQ(filling.tracks_.front().id_, 222);
  EXPECT_EQ(filling.lost_count_, 1);
  EXPECT_EQ(filling.marginalized_count_, 0);

  const FeatureHistory::TrackSelection full = history.tracks(10, 12, 1, 10, true);
  ASSERT_EQ(full.tracks_.size(), 2u);
  EXPECT_NE(
    std::find_if(
      full.tracks_.begin(), full.tracks_.end(), [](const FeatureTrack_s & track) { return track.id_ == 221; }),
    full.tracks_.end());
  EXPECT_NE(
    std::find_if(
      full.tracks_.begin(), full.tracks_.end(), [](const FeatureTrack_s & track) { return track.id_ == 222; }),
    full.tracks_.end());
  EXPECT_EQ(full.lost_count_, 1);
  EXPECT_EQ(full.marginalized_count_, 1);
}

// ===========================================================================
// T31 -- FeatureHistory keeps both same-frame camera views, drops only an
// exact (frame, camera) duplicate, and restores (frame, camera) order even
// when the right view is registered first.
// ===========================================================================
TEST(SchurUpdate, T31FeatureHistoryKeepsStereoPairAndOrdersByFrameCamera)
{
  FeatureHistory history;
  const Eigen::Vector2d uv(0.1, -0.2);
  const Eigen::Vector2d right_shift(0.03, 0.0);

  history.addFrameWithPixels(
    10, 1, {{310, uv + right_shift, Eigen::Vector2d(345.0, 240.0), true}});
  history.addFrameWithPixels(
    10, 0, {{310, uv, Eigen::Vector2d(320.0, 240.0), true}});
  // Same frame and camera: must be ignored without disturbing the order.
  history.addFrameWithPixels(
    10, 0, {{310, uv + Eigen::Vector2d(0.2, 0.0), Eigen::Vector2d(400.0, 240.0), true}});
  // Right view first again, now for the next frame (out-of-order fallback).
  history.addFrameWithPixels(
    11, 1, {{310, uv + Eigen::Vector2d(0.01, 0.0), Eigen::Vector2d(350.0, 240.0), true}});
  history.addFrameWithPixels(
    11, 0, {{310, uv + Eigen::Vector2d(0.0, 0.01), Eigen::Vector2d(322.0, 241.0), true}});

  const std::vector<FeatureTrack_s> tracks = history.allTracks(10);
  ASSERT_EQ(tracks.size(), 1u);
  ASSERT_EQ(tracks[0].observations_.size(), 4u);
  const int expected_clone[4] = {0, 0, 1, 1};
  const int expected_camera[4] = {0, 1, 0, 1};
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(tracks[0].observations_[static_cast<std::size_t>(i)].clone_index_, expected_clone[i]);
    EXPECT_EQ(tracks[0].observations_[static_cast<std::size_t>(i)].camera_id_, expected_camera[i]);
  }
  EXPECT_NEAR(tracks[0].observations_[0].pixel_uv_.x(), 320.0, 1e-12);
  EXPECT_NEAR(tracks[0].observations_[1].pixel_uv_.x(), 345.0, 1e-12);
  EXPECT_EQ(history.newestFrameIndex(), 11);
}

// ===========================================================================
// T32 -- min_observations counts raw measurements, so a one-frame stereo pair
// is two observations (and can qualify for a later lost-track update).
// ===========================================================================
TEST(SchurUpdate, T32FeatureHistoryStereoPairCountsAsTwoObservations)
{
  FeatureHistory history;
  const Eigen::Vector2d uv(0.12, -0.08);
  history.addFrame(10, 0, {{321, uv}});
  history.addFrame(10, 1, {{321, uv + Eigen::Vector2d(0.03, 0.0)}});
  history.addFrame(11, 0, {});
  history.addFrame(11, 1, {});

  const FeatureHistory::TrackSelection two = history.selectTracks(10, 11, 2, 10, false);
  ASSERT_EQ(two.tracks_.size(), 1u);
  EXPECT_EQ(two.tracks_.front().id_, 321);
  ASSERT_EQ(two.tracks_.front().observations_.size(), 2u);
  EXPECT_EQ(two.tracks_.front().observations_[0].clone_index_, 0);
  EXPECT_EQ(two.tracks_.front().observations_[0].camera_id_, 0);
  EXPECT_EQ(two.tracks_.front().observations_[1].clone_index_, 0);
  EXPECT_EQ(two.tracks_.front().observations_[1].camera_id_, 1);
  EXPECT_EQ(two.lost_count_, 1);
  EXPECT_EQ(two.marginalized_count_, 0);

  const FeatureHistory::TrackSelection three = history.selectTracks(10, 11, 3, 10, false);
  EXPECT_TRUE(three.tracks_.empty());
}

// ===========================================================================
// T33 -- a same-timestamp stereo pair has one clone but two cameras. The
// extrinsic baseline must be accepted, while an identical (clone, camera)
// pair still has no baseline. View selection must not depend on vector order.
// ===========================================================================
TEST(SchurUpdate, T33TriangulationSupportsSameCloneStereoBaseline)
{
  const SchurUpdater updater(default_options());
  const CameraExtrinsics_s cam0 = make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  const CameraExtrinsics_s cam1 =
    make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d(-0.11, 0.0, 0.0));
  const std::vector<Eigen::Vector4d> qs{quat(Eigen::Vector3d::Zero())};
  const std::vector<Eigen::Vector3d> ps{Eigen::Vector3d::Zero()};
  EkfState state;
  build_state(state, qs, ps);

  const Eigen::Vector3d world_point(0.4, -0.2, 4.0);
  const Eigen::Vector2d uv0 = project_world(qs[0], ps[0], cam0, world_point);
  const Eigen::Vector2d uv1 = project_world(qs[0], ps[0], cam1, world_point);

  FeatureTrack_s track;
  track.id_ = 330;
  track.observations_.push_back(make_observation(0, 0, uv0));
  track.observations_.push_back(make_observation(0, 1, uv1));
  ASSERT_TRUE(updater.triangulate(state, {cam0, cam1}, track));
  EXPECT_LT((track.world_point_ - world_point).norm(), 1e-6);
  EXPECT_TRUE(track.has_point_);

  FeatureTrack_s reversed;
  reversed.id_ = 331;
  reversed.observations_.push_back(make_observation(0, 1, uv1));
  reversed.observations_.push_back(make_observation(0, 0, uv0));
  ASSERT_TRUE(updater.triangulate(state, {cam0, cam1}, reversed));
  EXPECT_LT((reversed.world_point_ - world_point).norm(), 1e-6);

  FeatureTrack_s mono_duplicate;
  mono_duplicate.id_ = 332;
  mono_duplicate.observations_.push_back(make_observation(0, 0, uv0));
  mono_duplicate.observations_.push_back(
    make_observation(0, 0, uv0 + Eigen::Vector2d(0.05, 0.01)));
  EXPECT_FALSE(updater.triangulate(state, {cam0, cam1}, mono_duplicate));
  EXPECT_FALSE(mono_duplicate.has_point_);
}

// ===========================================================================
// T34 -- the Schur update consumes both cameras at every clone and reports
// the true raw-observation count.
// ===========================================================================
TEST(SchurUpdate, T34UpdateConsumesStereoObservationsPerClone)
{
  UpdateOptions_s opts = default_options();
  opts.condition_limit_ = 0.0;
  SchurUpdater updater(opts);
  const CameraExtrinsics_s cam0 =
    make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d(0.02, 0.0, 0.0));
  const CameraExtrinsics_s cam1 =
    make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d(-0.09, 0.0, 0.0));
  std::vector<Eigen::Vector4d> qs;
  std::vector<Eigen::Vector3d> ps;
  smooth_trajectory(4, qs, ps);
  EkfState state;
  build_state(state, qs, ps);
  regularize_covariance(state);

  const int num_tracks = 6;
  std::vector<FeatureTrack_s> tracks;
  for (int f = 0; f < num_tracks; ++f) {
    FeatureTrack_s track;
    track.id_ = 3400 + f;
    const Eigen::Vector3d world_point(-0.6 + 0.25 * f, 0.15 * ((f % 3) - 1), 4.0 + 0.1 * f);
    for (int i = 0; i < 4; ++i) {
      track.observations_.push_back(make_observation(
        i, 0, project_world(qs[static_cast<std::size_t>(i)], ps[static_cast<std::size_t>(i)], cam0,
                            world_point)));
      track.observations_.push_back(make_observation(
        i, 1, project_world(qs[static_cast<std::size_t>(i)], ps[static_cast<std::size_t>(i)], cam1,
                            world_point)));
    }
    tracks.push_back(track);
  }

  const UpdateSummary_s summary = updater.update(state, tracks, {cam0, cam1});
  EXPECT_TRUE(summary.applied_);
  EXPECT_EQ(summary.features_used_, num_tracks);
  EXPECT_EQ(summary.observations_used_, 2 * 4 * num_tracks);
  EXPECT_EQ(summary.used_feature_ids_.size(), static_cast<std::size_t>(num_tracks));
  EXPECT_TRUE(state_is_finite(state));
  for (const FeatureTrack_s & track : tracks)
    EXPECT_TRUE(track.has_point_);
}

}  // namespace

// ===========================================================================
// T1 -- analytic Jacobians against central finite differences.
// ===========================================================================
TEST(SchurUpdate, T1ObservationJacobianMatchesFiniteDifferences)
{
  const UpdateOptions_s opts = default_options();
  const SchurUpdater updater(opts);
  const double sigma = opts.pixel_noise_ / opts.focal_length_;

  std::mt19937 rng(20261025);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);

  const Eigen::Vector4d q_i = random_quat(rng, 0.8);
  const Eigen::Vector3d p_i(unit(rng), unit(rng), unit(rng));
  const CameraExtrinsics_s cam =
    make_camera(Eigen::Vector3d(0.05, -0.1, 0.03), Eigen::Vector3d(0.02, -0.01, 0.05));
  // Point ~5 m in front of the camera, off-centre.
  const Eigen::Vector3d uv_c(0.25 * unit(rng), 0.25 * unit(rng), 1.0);
  const Eigen::Vector3d p_c = 5.0 * uv_c;
  const Eigen::Vector3d world_point =
    p_i + EkfState::rotation(q_i) * (EkfState::rotation(cam.q_cb_) * p_c + cam.p_cb_);

  std::vector<Eigen::Vector4d> qs{q_i};
  std::vector<Eigen::Vector3d> ps{p_i};
  EkfState state;
  build_state(state, qs, ps);
  state.clone(0).q_ = q_i;
  state.clone(0).p_ = p_i;

  const Eigen::Vector2d uv = project_world(q_i, p_i, cam, world_point);
  const FeatureObservation_s obs = make_observation(0, 0, uv);

  const ObservationJacobian_s J = updater.observationJacobian(state, cam, world_point, obs);
  ASSERT_TRUE(J.valid_);
  EXPECT_LT(std::abs(J.p_c_.z() - p_c.z()), 1e-12);

  const double eps = 1e-6;
  Eigen::Matrix<double, 2, 6> fd_pose = Eigen::Matrix<double, 2, 6>::Zero();
  Eigen::Matrix<double, 2, 3> fd_point = Eigen::Matrix<double, 2, 3>::Zero();
  for (int k = 0; k < 6; ++k) {
    Eigen::Vector4d q_p = q_i, q_m = q_i;
    Eigen::Vector3d p_p = p_i, p_m = p_i;
    if (k < 3) {
      // Right (body) perturbation: q' = q (x) dq.
      Eigen::Vector3d d = Eigen::Vector3d::Zero();
      d[k] = eps;
      q_p = EkfState::quaternionMultiply(q_i, quat(d)).normalized();
      q_m = EkfState::quaternionMultiply(q_i, quat(-d)).normalized();
    } else {
      p_p[k - 3] += eps;
      p_m[k - 3] -= eps;
    }
    const Eigen::Vector2d r_p = weighted_residual_model(q_p, p_p, cam, world_point, uv, sigma);
    const Eigen::Vector2d r_m = weighted_residual_model(q_m, p_m, cam, world_point, uv, sigma);
    fd_pose.col(k) = (r_p - r_m) / (2.0 * eps);
  }
  for (int k = 0; k < 3; ++k) {
    Eigen::Vector3d wp = world_point, wm = world_point;
    wp[k] += eps;
    wm[k] -= eps;
    const Eigen::Vector2d r_p = weighted_residual_model(q_i, p_i, cam, wp, uv, sigma);
    const Eigen::Vector2d r_m = weighted_residual_model(q_i, p_i, cam, wm, uv, sigma);
    fd_point.col(k) = (r_p - r_m) / (2.0 * eps);
  }

  // residual = (uv - h)/sigma  =>  d(residual)/dx = -H.
  const double pose_err = (fd_pose + J.H_pose_).norm() / J.H_pose_.norm();
  const double point_err = (fd_point + J.H_point_).norm() / J.H_point_.norm();
  std::printf("[T1] rel err pose=%.3e point=%.3e\n", pose_err, point_err);
  EXPECT_LT(pose_err, 1e-7);
  EXPECT_LT(point_err, 1e-7);
}

// ===========================================================================
// T2 -- triangulation of a known point, noise free and with 0.5 px noise.
// ===========================================================================
TEST(SchurUpdate, T2TriangulationRecoversKnownPoint)
{
  const UpdateOptions_s opts = default_options();
  const SchurUpdater updater(opts);
  const CameraExtrinsics_s cam =
    make_camera(Eigen::Vector3d(0.02, -0.03, 0.01), Eigen::Vector3d(0.01, 0.0, 0.02));
  std::vector<Eigen::Vector4d> qs;
  std::vector<Eigen::Vector3d> ps;
  smooth_trajectory(5, qs, ps);
  EkfState state;
  build_state(state, qs, ps);

  const Eigen::Vector3d world_point(1.0, 0.8, 4.5);

  // ---- noise free -----------------------------------------------------
  FeatureTrack_s track;
  track.id_ = 1;
  for (int i = 0; i < 5; ++i)
    track.observations_.push_back(make_observation(
      i, 0, project_world(qs[static_cast<std::size_t>(i)], ps[static_cast<std::size_t>(i)], cam, world_point)));
  ASSERT_TRUE(updater.triangulate(state, {cam}, track));
  const double err_clean = (track.world_point_ - world_point).norm();
  std::printf("[T2] noise-free point error = %.3e m\n", err_clean);
  EXPECT_LT(err_clean, 1e-6);

  // ---- 0.5 px noise ---------------------------------------------------
  std::mt19937 rng(11);
  std::normal_distribution<double> noise(0.0, 0.5 / K_FOCAL);
  double worst = 0.0;
  for (int trial = 0; trial < 20; ++trial) {
    FeatureTrack_s noisy;
    noisy.id_ = 2;
    for (int i = 0; i < 5; ++i) {
      Eigen::Vector2d uv =
        project_world(qs[static_cast<std::size_t>(i)], ps[static_cast<std::size_t>(i)], cam, world_point);
      uv += Eigen::Vector2d(noise(rng), noise(rng));
      noisy.observations_.push_back(make_observation(i, 0, uv));
    }
    ASSERT_TRUE(updater.triangulate(state, {cam}, noisy));
    worst = std::max(worst, (noisy.world_point_ - world_point).norm());
  }
  std::printf("[T2] worst 0.5 px noise error = %.4f m (20 trials)\n", worst);
  EXPECT_LT(worst, 0.05);
}

// ===========================================================================
// T3 -- the reduce_track blocks reproduce the exact Schur complement.
// ===========================================================================
TEST(SchurUpdate, T3SchurReductionMatchesJointSchurComplement)
{
  const UpdateOptions_s opts = default_options();
  const SchurUpdater updater(opts);
  const CameraExtrinsics_s cam =
    make_camera(Eigen::Vector3d(0.01, 0.02, -0.01), Eigen::Vector3d(0.03, -0.02, 0.01));
  std::vector<Eigen::Vector4d> qs;
  std::vector<Eigen::Vector3d> ps;
  smooth_trajectory(3, qs, ps);
  EkfState state;
  build_state(state, qs, ps);

  const Eigen::Vector3d world_point(0.6, 1.1, 4.0);
  FeatureTrack_s track;
  track.id_ = 3;
  for (int i = 0; i < 3; ++i)
    track.observations_.push_back(make_observation(
      i, 0, project_world(qs[static_cast<std::size_t>(i)], ps[static_cast<std::size_t>(i)], cam, world_point)));
  ASSERT_TRUE(updater.triangulate(state, {cam}, track));

  const TrackReduction_s red = updater.reduceTrack(state, {cam}, track);
  ASSERT_TRUE(red.valid_);
  const JointSystem_s joint = build_joint(updater, state, {cam}, track);

  // reduce_track's W must equal H_x^T H_f restricted to the clone blocks.
  Eigen::MatrixXd A_xf = Eigen::MatrixXd::Zero(state.size(), 3);
  for (const auto & w : red.W_)
    A_xf.block<6, 3>(state.cloneOffset(w.first), 0) = w.second;
  const double W_err = (A_xf - joint.Hx_.transpose() * joint.Hf_).norm();
  const double W_scale = (joint.Hx_.transpose() * joint.Hf_).norm();
  std::printf(
    "[T3] W rel err = %.3e, A_point rel err = %.3e\n", W_err / W_scale,
    (red.A_point_ - joint.Hf_.transpose() * joint.Hf_).norm() / (joint.Hf_.transpose() * joint.Hf_).norm());
  EXPECT_LT(W_err, 1e-9 * std::max(1.0, W_scale));
  EXPECT_LT(
    (red.A_point_ - joint.Hf_.transpose() * joint.Hf_).norm(),
    1e-9 * std::max(1.0, (joint.Hf_.transpose() * joint.Hf_).norm()));
  EXPECT_LT(
    (red.b_point_ - joint.Hf_.transpose() * joint.r_).norm(),
    1e-9 * std::max(1.0, (joint.Hf_.transpose() * joint.r_).norm()));

  Eigen::MatrixXd A_red, A_ref;
  Eigen::VectorXd b_red, b_ref;
  reduce_with_blocks(state, joint, red, A_red, b_red);
  reduce_exact(joint, A_ref, b_ref);

  const double a_err = (A_red - A_ref).norm() / A_ref.norm();
  const double b_err = (b_red - b_ref).norm() / b_ref.norm();
  std::printf("[T3] Schur A rel err = %.3e, b rel err = %.3e\n", a_err, b_err);
  EXPECT_LT(a_err, 1e-9);
  EXPECT_LT(b_err, 1e-9);
}

// ===========================================================================
// T4 -- one update() call reproduces the information-form solution.
// ===========================================================================
TEST(SchurUpdate, T4EkfUpdateMatchesInformationForm)
{
  const UpdateOptions_s opts = default_options();
  SchurUpdater updater(opts);
  const CameraExtrinsics_s cam = make_camera(Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(0.02, 0.0, 0.0));

  // Truth poses, perturbed nominal poses (the measurement residual is then
  // non-trivial).
  std::vector<Eigen::Vector4d> q_true, q_nom;
  std::vector<Eigen::Vector3d> p_true, p_nom;
  smooth_trajectory(3, q_true, p_true);
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  for (int i = 0; i < 3; ++i) {
    q_nom.push_back(EkfState::quaternionMultiply(
      q_true[static_cast<std::size_t>(i)],
      quat(Eigen::Vector3d(0.004 * d(rng), 0.004 * d(rng), 0.004 * d(rng)))));
    p_nom.push_back(
      p_true[static_cast<std::size_t>(i)] + Eigen::Vector3d(0.02 * d(rng), 0.02 * d(rng), 0.02 * d(rng)));
  }
  EkfState state;
  build_state(state, q_nom, p_nom);
  regularize_covariance(state);  // fresh clones make the raw P singular

  const Eigen::Vector3d world_point(0.9, 0.7, 4.2);
  FeatureTrack_s track;
  track.id_ = 4;
  for (int i = 0; i < 3; ++i)
    track.observations_.push_back(make_observation(
      i, 0,
      project_world(q_true[static_cast<std::size_t>(i)], p_true[static_cast<std::size_t>(i)], cam, world_point)));
  ASSERT_TRUE(updater.triangulate(state, {cam}, track));

  const TrackReduction_s red = updater.reduceTrack(state, {cam}, track);
  ASSERT_TRUE(red.valid_);
  const JointSystem_s joint = build_joint(updater, state, {cam}, track);
  Eigen::MatrixXd A_red;
  Eigen::VectorXd b_red;
  reduce_with_blocks(state, joint, red, A_red, b_red);

  const Snapshot_s before = snapshot(state);
  const Eigen::MatrixXd P_before = state.covariance();
  const int n = state.size();

  std::vector<FeatureTrack_s> tracks{track};
  const UpdateSummary_s summary = updater.update(state, tracks, {cam});
  EXPECT_EQ(summary.features_used_, 1);
  EXPECT_EQ(summary.observations_used_, 3);
  EXPECT_EQ(summary.degrees_of_freedom_, 6);

  const Eigen::VectorXd dx = reconstruct_dx(before, state);
  const Eigen::MatrixXd P_inv = P_before.ldlt().solve(Eigen::MatrixXd::Identity(n, n));
  const Eigen::VectorXd lhs = (P_inv + A_red) * dx;
  const double info_err = (lhs - b_red).norm() / b_red.norm();
  std::printf(
    "[T4] information-form rel err = %.3e (|dx| = %.3e, |b| = %.3e)\n", info_err, dx.norm(), b_red.norm());
  EXPECT_LT(info_err, 1e-8);

  // Joint least squares of the same system including the prior (state part).
  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n + 3, n + 3);
  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n + 3);
  M.topLeftCorner(n, n) = joint.Hx_.transpose() * joint.Hx_ + P_inv;
  M.topRightCorner(n, 3) = joint.Hx_.transpose() * joint.Hf_;
  M.bottomLeftCorner(3, n) = joint.Hf_.transpose() * joint.Hx_;
  M.bottomRightCorner(3, 3) = joint.Hf_.transpose() * joint.Hf_;
  rhs.head(n) = joint.Hx_.transpose() * joint.r_;
  rhs.tail(3) = joint.Hf_.transpose() * joint.r_;
  const Eigen::VectorXd joint_sol = M.ldlt().solve(rhs);
  const double joint_err = (joint_sol.head(n) - dx).norm() / dx.norm();
  std::printf("[T4] joint LS state-part rel err = %.3e\n", joint_err);
  EXPECT_LT(joint_err, 1e-8);
}

// ===========================================================================
// T5 -- covariance stays symmetric/PD and the observed blocks shrink.
// ===========================================================================
TEST(SchurUpdate, T5CovarianceSymmetricPositiveDefiniteAndShrinks)
{
  const UpdateOptions_s opts = default_options();
  SchurUpdater updater(opts);
  const CameraExtrinsics_s cam = make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d(0.02, 0.0, 0.0));
  std::vector<Eigen::Vector4d> qs;
  std::vector<Eigen::Vector3d> ps;
  smooth_trajectory(4, qs, ps);
  EkfState state;
  build_state(state, qs, ps);
  regularize_covariance(state);

  std::vector<FeatureTrack_s> tracks;
  for (int f = 0; f < 6; ++f) {
    FeatureTrack_s track;
    track.id_ = 100 + f;
    const Eigen::Vector3d world_point(-1.0 + 0.4 * f, 0.5 * ((f % 3) - 1), 4.0 + 0.2 * f);
    for (int i = 0; i < 4; ++i)
      track.observations_.push_back(make_observation(
        i, 0,
        project_world(qs[static_cast<std::size_t>(i)], ps[static_cast<std::size_t>(i)], cam, world_point)));
    tracks.push_back(track);
  }

  const Snapshot_s before = snapshot(state);
  const UpdateSummary_s summary = updater.update(state, tracks, {cam});
  EXPECT_EQ(summary.features_used_, 6);

  const Eigen::MatrixXd P = state.covariance();
  EXPECT_LT((P - P.transpose()).norm(), 1e-12 * std::max(1.0, P.norm()));
  Eigen::LDLT<Eigen::MatrixXd> ldlt(P);
  EXPECT_EQ(ldlt.info(), Eigen::Success);
  EXPECT_GT(ldlt.vectorD().minCoeff(), 1e-12);
  EXPECT_TRUE(P.allFinite());

  for (int i = 0; i < state.numClones(); ++i) {
    const int off = state.cloneOffset(i);
    const double tr_before = before.P_.block<6, 6>(off, off).trace();
    const double tr_after = P.block<6, 6>(off, off).trace();
    std::printf("[T5] clone %d trace %.6e -> %.6e\n", i, tr_before, tr_after);
    EXPECT_LT(tr_after, tr_before);
  }
}

// ===========================================================================
// T6 -- outlier gate: a 50 px offset track is rejected, state unchanged.
// ===========================================================================
TEST(SchurUpdate, T6Chi2GateRejectsOutlierWithoutTouchingState)
{
  const UpdateOptions_s opts = default_options();
  SchurUpdater updater(opts);
  const CameraExtrinsics_s cam = make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  std::vector<Eigen::Vector4d> qs;
  std::vector<Eigen::Vector3d> ps;
  smooth_trajectory(3, qs, ps);
  EkfState state;
  build_state(state, qs, ps);
  regularize_covariance(state);

  const Eigen::Vector3d world_point(0.4, 0.9, 4.0);
  const double offset_px = 50.0;

  // (a) three observations, the second one shifted by 50 px.
  FeatureTrack_s track;
  track.id_ = 77;
  for (int i = 0; i < 3; ++i) {
    Eigen::Vector2d uv =
      project_world(qs[static_cast<std::size_t>(i)], ps[static_cast<std::size_t>(i)], cam, world_point);
    if (i == 1)
      uv.x() += offset_px / K_FOCAL;
    track.observations_.push_back(make_observation(i, 0, uv));
  }

  {
    const Snapshot_s before = snapshot(state);
    std::vector<FeatureTrack_s> tracks{track};
    const UpdateSummary_s summary = updater.update(state, tracks, {cam});
    std::printf(
      "[T6] (3 obs, 1 x 50 px) used=%d rejected=%d ids=%zu\n", summary.features_used_,
      summary.features_rejected_, summary.rejected_feature_ids_.size());
    EXPECT_EQ(summary.features_used_, 0);
    EXPECT_EQ(summary.features_rejected_, 1);
    ASSERT_EQ(summary.rejected_feature_ids_.size(), 1u);
    EXPECT_EQ(summary.rejected_feature_ids_.front(), 77);
    EXPECT_LT(snapshot_max_diff(before, snapshot(state)), 1e-12);
  }

  // (b) a single-observation track (below min_observations) is also dropped.
  {
    FeatureTrack_s single;
    single.id_ = 78;
    Eigen::Vector2d uv = project_world(qs[0], ps[0], cam, world_point);
    uv.x() += offset_px / K_FOCAL;
    single.observations_.push_back(make_observation(0, 0, uv));
    const Snapshot_s before = snapshot(state);
    std::vector<FeatureTrack_s> tracks{single};
    const UpdateSummary_s summary = updater.update(state, tracks, {cam});
    EXPECT_EQ(summary.features_used_, 0);
    EXPECT_EQ(summary.features_rejected_, 1);
    EXPECT_LT(snapshot_max_diff(before, snapshot(state)), 1e-12);
  }

  // (c) the chi2 statistic itself: with a confident (small) covariance the
  //     clone poses cannot explain a 50 px residual, so the gate trips.
  {
    EkfState confident;
    build_state(confident, qs, ps);
    confident.covariance() *= 1e-6;          // shrink every block consistently
    regularize_covariance(confident, 1e-6);  // ... but keep P PD
    FeatureTrack_s gate_track;
    gate_track.id_ = 79;
    for (int i = 0; i < 3; ++i)
      gate_track.observations_.push_back(make_observation(
        i, 0,
        project_world(qs[static_cast<std::size_t>(i)], ps[static_cast<std::size_t>(i)], cam, world_point)));
    // Triangulate clean data first, then corrupt one *measurement*: this is
    // the situation the chi2 gate sees at update time.
    ASSERT_TRUE(updater.triangulate(confident, {cam}, gate_track));
    gate_track.observations_[1].uv_.x() += offset_px / K_FOCAL;
    const TrackReduction_s red = updater.reduceTrack(confident, {cam}, gate_track);
    const double gate = opts.chi2_multiplier_ * SchurUpdater::chi2Quantile95(6);
    std::printf("[T6] (gate) chi2=%.3f gate=%.3f rms=%.3f px\n", red.chi2_, gate, red.rms_residual_px_);
    EXPECT_GT(red.chi2_, gate);
  }
}

// ===========================================================================
// T7 -- convergence smoke test on a synthetic static scene.
// ===========================================================================
TEST(SchurUpdate, T7ConvergesOnSyntheticStaticScene)
{
  UpdateOptions_s opts = default_options();
  opts.triangulation_iterations_ = 20;
  SchurUpdater updater(opts);
  const CameraExtrinsics_s cam = make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

  const int num_frames = 10;
  std::vector<Eigen::Vector4d> q_true;
  std::vector<Eigen::Vector3d> p_true;
  smooth_trajectory(num_frames, q_true, p_true);

  // 20 static features spread in front of the trajectory.
  std::vector<Eigen::Vector3d> features;
  features.reserve(20);
  for (int i = 0; i < 20; ++i) {
    features.push_back(
      Eigen::Vector3d(-1.5 + 0.3 * (i % 5) + 0.7, 0.4 * ((i / 5) - 2) + 0.9, 4.0 + 0.3 * (i % 3)));
  }

  // Small, smooth, gauge-free perturbations.  With 1.5 px measurement noise
  // and the fixed 5 px triangulation gate, a 0.5 deg / 5 cm per-clone error at
  // 4 m is already ~6 px of inconsistent reprojection; 0.1 deg / 1 cm keeps
  // the synthetic measurements inside the configured noise model while still
  // making the filter start clearly away from the truth.
  std::vector<Eigen::Vector3d> dtheta, dp;
  for (int i = 0; i < num_frames; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(num_frames - 1);
    Eigen::Vector3d rot_dir(
      std::sin(2.0 * M_PI * t), 0.5 * std::sin(4.0 * M_PI * t + 0.3), 0.3 * std::cos(2.0 * M_PI * t));
    Eigen::Vector3d pos_dir(
      std::sin(2.0 * M_PI * t + 0.7), 0.4 * std::cos(4.0 * M_PI * t), 0.2 * std::sin(6.0 * M_PI * t));
    dtheta.push_back(rot_dir.normalized() * (0.1 * M_PI / 180.0));
    dp.push_back(pos_dir.normalized() * 0.01);
  }
  remove_gauge(q_true, p_true, dtheta, dp);

  EkfOptions_s ekf_options = default_ekf_options(num_frames + 2);
  EkfState state;
  state.reset(
    ekf_options, 0.0, q_true[0], p_true[0], Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
    Eigen::Vector3d::Zero());

  // All ten frames are pre-augmented as the window; the ten update() calls
  // below then represent one visual update per frame.
  for (int frame = 0; frame < num_frames; ++frame) {
    state.imu().q_ = EkfState::quaternionMultiply(
                       q_true[static_cast<std::size_t>(frame)], quat(dtheta[static_cast<std::size_t>(frame)]))
                       .normalized();
    state.imu().p_ = p_true[static_cast<std::size_t>(frame)] + dp[static_cast<std::size_t>(frame)];
    // We set the nominal pose directly instead of calling propagate().
    state.refreshImuFej();
    state.augmentClone(static_cast<double>(frame));
  }
  set_pose_diagonal_covariance(state, 2.0 * M_PI / 180.0, 0.2);

  // Every static feature is observed by every clone in the window.
  std::vector<FeatureTrack_s> tracks;
  for (std::size_t f = 0; f < features.size(); ++f) {
    FeatureTrack_s track;
    track.id_ = static_cast<int>(f);
    for (int i = 0; i < num_frames; ++i) {
      track.observations_.push_back(make_observation(
        i, 0,
        project_world(
          q_true[static_cast<std::size_t>(i)], p_true[static_cast<std::size_t>(i)], cam, features[f])));
    }
    tracks.push_back(track);
  }

  double before_pos = 0.0;
  double before_rot = 0.0;
  double before_rel_pos = 0.0;
  double before_rel_rot = 0.0;
  pose_errors(state, q_true, p_true, false, before_pos, before_rot);
  pose_errors(state, q_true, p_true, true, before_rel_pos, before_rel_rot);

  for (int frame = 0; frame < num_frames; ++frame) {
    const UpdateSummary_s summary = updater.update(state, tracks, {cam});
    EXPECT_EQ(summary.features_used_, static_cast<int>(features.size()));
  }

  double after_pos = 0.0;
  double after_rot = 0.0;
  double after_rel_pos = 0.0;
  double after_rel_rot = 0.0;
  pose_errors(state, q_true, p_true, false, after_pos, after_rot);
  pose_errors(state, q_true, p_true, true, after_rel_pos, after_rel_rot);

  std::printf(
    "[T7] absolute pos %.4f -> %.4f cm (x%.1f), rot %.4f -> %.4f deg (x%.1f)\n", before_pos * 100.0,
    after_pos * 100.0, before_pos / after_pos, before_rot * 180.0 / M_PI, after_rot * 180.0 / M_PI,
    before_rot / after_rot);
  std::printf(
    "[T7] gauge-invariant pos %.4f -> %.4f cm (x%.1f), rot %.4f -> %.4f deg (x%.1f)\n",
    before_rel_pos * 100.0, after_rel_pos * 100.0, before_rel_pos / after_rel_pos,
    before_rel_rot * 180.0 / M_PI, after_rel_rot * 180.0 / M_PI, before_rel_rot / after_rel_rot);

  EXPECT_LT(before_pos, 0.02);
  EXPECT_LT(before_rot, 0.0021);
  EXPECT_GT(before_rel_pos / after_rel_pos, 3.0);
  EXPECT_GT(before_rel_rot / after_rel_rot, 3.0);
}

// ===========================================================================
// T8 -- robustness: degenerate inputs must not crash or produce NaN.
// ===========================================================================
TEST(SchurUpdate, T8DegenerateInputsAreHandled)
{
  const UpdateOptions_s opts = default_options();
  SchurUpdater updater(opts);
  const CameraExtrinsics_s cam = make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  std::vector<Eigen::Vector4d> qs;
  std::vector<Eigen::Vector3d> ps;
  smooth_trajectory(3, qs, ps);
  EkfState state;
  build_state(state, qs, ps);
  regularize_covariance(state);

  const Eigen::Vector3d world_point(0.5, 0.5, 4.0);
  const Eigen::Vector2d uv0 = project_world(qs[0], ps[0], cam, world_point);

  std::vector<FeatureTrack_s> tracks;

  // (a) empty track list
  {
    const Snapshot_s before = snapshot(state);
    const UpdateSummary_s summary = updater.update(state, tracks, {cam});
    EXPECT_EQ(summary.features_used_, 0);
    EXPECT_LT(snapshot_max_diff(before, snapshot(state)), 1e-12);
  }

  // (b) single observation
  FeatureTrack_s single;
  single.id_ = 1;
  single.observations_.push_back(make_observation(0, 0, uv0));
  tracks.push_back(single);

  // (c) duplicate observation at the same clone
  FeatureTrack_s dup;
  dup.id_ = 2;
  dup.observations_.push_back(make_observation(0, 0, uv0));
  dup.observations_.push_back(make_observation(0, 0, uv0 + Eigen::Vector2d(0.01, -0.01)));
  dup.observations_.push_back(make_observation(1, 0, project_world(qs[1], ps[1], cam, world_point)));
  tracks.push_back(dup);

  // (d) all observations identical
  FeatureTrack_s identical;
  identical.id_ = 3;
  for (int i = 0; i < 3; ++i)
    identical.observations_.push_back(make_observation(i, 0, uv0));
  tracks.push_back(identical);

  // (e) a camera id that does not exist
  FeatureTrack_s bad_camera;
  bad_camera.id_ = 4;
  bad_camera.observations_.push_back(make_observation(0, 7, uv0));
  bad_camera.observations_.push_back(make_observation(1, 7, uv0));
  tracks.push_back(bad_camera);

  // (f) a valid track so the update path is really exercised as well
  FeatureTrack_s good;
  good.id_ = 5;
  for (int i = 0; i < 3; ++i)
    good.observations_.push_back(make_observation(
      i, 0, project_world(qs[static_cast<std::size_t>(i)], ps[static_cast<std::size_t>(i)], cam, world_point)));
  tracks.push_back(good);

  const UpdateSummary_s summary = updater.update(state, tracks, {cam});
  std::printf(
    "[T8] used=%d rejected=%d obs=%d\n", summary.features_used_, summary.features_rejected_,
    summary.observations_used_);
  // Degenerate tracks may be rejected at triangulation or survive; the
  // contract is that every input track is accounted for and nothing is NaN.
  EXPECT_EQ(summary.features_used_ + summary.features_rejected_, static_cast<int>(tracks.size()));

  EXPECT_TRUE(state.imu().q_.allFinite());
  EXPECT_TRUE(state.imu().p_.allFinite());
  EXPECT_TRUE(state.imu().v_.allFinite());
  EXPECT_TRUE(state.covariance().allFinite());
  for (int i = 0; i < state.numClones(); ++i) {
    EXPECT_TRUE(state.clone(i).q_.allFinite());
    EXPECT_TRUE(state.clone(i).p_.allFinite());
  }
  EXPECT_NEAR(state.imu().q_.norm(), 1.0, 1e-12);
}

// ===========================================================================
// T9 -- frame-level feature/observation starvation gates must be no-ops.
// ===========================================================================
TEST(SchurUpdate, T9MinimumFeatureAndObservationGatesSkipUpdate)
{
  const CameraExtrinsics_s cam = make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  std::vector<Eigen::Vector4d> qs;
  std::vector<Eigen::Vector3d> ps;
  smooth_trajectory(3, qs, ps);
  const Eigen::Vector3d world_point(0.5, 0.8, 4.0);

  auto make_state = [&]()
  {
    EkfState state;
    build_state(state, qs, ps);
    regularize_covariance(state);
    return state;
  };
  auto make_tracks = [&]()
  {
    FeatureTrack_s track;
    track.id_ = 900;
    for (int i = 0; i < 3; ++i) {
      track.observations_.push_back(make_observation(
        i, 0,
        project_world(qs[static_cast<std::size_t>(i)], ps[static_cast<std::size_t>(i)], cam, world_point)));
    }
    return std::vector<FeatureTrack_s>{track};
  };

  // One usable track is below min_features=2.
  {
    UpdateOptions_s opts = default_options();
    opts.min_features_ = 2;
    opts.min_update_observations_ = 1;
    SchurUpdater updater(opts);
    EkfState state = make_state();
    std::vector<FeatureTrack_s> tracks = make_tracks();
    const Snapshot_s before = snapshot(state);
    const UpdateSummary_s summary = updater.update(state, tracks, {cam});
    std::printf(
      "[T9] feature gate applied=%d features=%d obs=%d\n", summary.applied_ ? 1 : 0, summary.features_used_,
      summary.observations_used_);
    EXPECT_FALSE(summary.applied_);
    EXPECT_LT(snapshot_max_diff(before, snapshot(state)), 1e-15);
  }

  // The same three-observation track is below min_update_observations=4.
  {
    UpdateOptions_s opts = default_options();
    opts.min_features_ = 1;
    opts.min_update_observations_ = 4;
    SchurUpdater updater(opts);
    EkfState state = make_state();
    std::vector<FeatureTrack_s> tracks = make_tracks();
    const Snapshot_s before = snapshot(state);
    const UpdateSummary_s summary = updater.update(state, tracks, {cam});
    std::printf(
      "[T9] observation gate applied=%d features=%d obs=%d\n", summary.applied_ ? 1 : 0,
      summary.features_used_, summary.observations_used_);
    EXPECT_FALSE(summary.applied_);
    EXPECT_LT(snapshot_max_diff(before, snapshot(state)), 1e-15);
  }
}

// ===========================================================================
// T10 -- one track has a structurally rank-deficient information matrix.
// ===========================================================================
TEST(SchurUpdate, T10SingleTrackStructuralNullspaceIsRankRevealing)
{
  const UpdateOptions_s opts = default_options();
  SchurUpdater updater(opts);
  const CameraExtrinsics_s cam = make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  std::vector<Eigen::Vector4d> qs;
  std::vector<Eigen::Vector3d> ps;
  smooth_trajectory(3, qs, ps);
  EkfState state;
  build_state(state, qs, ps);
  regularize_covariance(state);

  const Eigen::Vector3d world_point(0.7, 0.4, 4.0);
  FeatureTrack_s track;
  track.id_ = 1000;
  for (int i = 0; i < 3; ++i) {
    track.observations_.push_back(make_observation(
      i, 0, project_world(qs[static_cast<std::size_t>(i)], ps[static_cast<std::size_t>(i)], cam, world_point)));
  }

  std::vector<FeatureTrack_s> tracks{track};
  const UpdateSummary_s summary = updater.update(state, tracks, {cam});
  const Eigen::MatrixXd P = state.covariance();
  const double symmetry_error = (P - P.transpose()).norm();
  const double min_eigenvalue = min_covariance_eigenvalue(P);
  const double psd_tolerance = 1e-9 * std::max(1.0, P.cwiseAbs().maxCoeff());

  std::printf(
    "[T10] applied=%d rank=%d n=%d cond=%.6e min_eig=%.3e\n", summary.applied_ ? 1 : 0, summary.rank_,
    state.size(), summary.condition_, min_eigenvalue);
  EXPECT_TRUE(summary.applied_);
  EXPECT_GT(summary.rank_, 0);
  EXPECT_LT(summary.rank_, state.size());
  EXPECT_TRUE(std::isfinite(summary.condition_));
  EXPECT_GE(summary.condition_, 1.0);
  EXPECT_TRUE(state_is_finite(state));
  EXPECT_TRUE(P.allFinite());
  EXPECT_LT(symmetry_error, 1e-12 * std::max(1.0, P.norm()));
  EXPECT_GE(min_eigenvalue, -psd_tolerance);

  // A single track cannot constrain the visual pose gauge or the mono scale.
  // The rank-revealing square-root update must discard those structural null
  // directions; unlike the old information-form path, no ridge is added to
  // inject fictitious information (and a spurious covariance update) there.
}

// ===========================================================================
// T11 -- an invalid point block rejects the track and leaves the frame intact.
// ===========================================================================
TEST(SchurUpdate, T11DegeneratePointBlockSkipsWholeFrame)
{
  const UpdateOptions_s opts = default_options();
  SchurUpdater updater(opts);
  const CameraExtrinsics_s cam = make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  std::vector<Eigen::Vector4d> qs;
  std::vector<Eigen::Vector3d> ps;
  smooth_trajectory(3, qs, ps);
  EkfState state;
  build_state(state, qs, ps);
  regularize_covariance(state);

  // Repeating the same observation from one clone leaves the 3x3 point block
  // with rank two, even though the track has enough observations to pass the
  // frame-level observation gate.
  const Eigen::Vector3d world_point(0.5, 0.5, 4.0);
  const Eigen::Vector2d uv = project_world(qs[0], ps[0], cam, world_point);
  FeatureTrack_s duplicate;
  duplicate.id_ = 1100;
  duplicate.world_point_ = world_point;
  duplicate.has_point_ = true;
  duplicate.observations_.push_back(make_observation(0, 0, uv));
  duplicate.observations_.push_back(make_observation(0, 0, uv));

  const TrackReduction_s red = updater.reduceTrack(state, {cam}, duplicate);
  EXPECT_FALSE(red.valid_);

  const Snapshot_s before = snapshot(state);
  std::vector<FeatureTrack_s> tracks{duplicate};
  const UpdateSummary_s summary = updater.update(state, tracks, {cam});
  std::printf(
    "[T11] point_block_valid=%d applied=%d rank=%d\n", red.valid_ ? 1 : 0, summary.applied_ ? 1 : 0,
    summary.rank_);
  EXPECT_FALSE(summary.applied_);
  EXPECT_LT(snapshot_max_diff(before, snapshot(state)), 1e-15);
}

// ===========================================================================
// T12 -- multiple valid tracks produce a finite, applied, non-empty update.
// ===========================================================================
TEST(SchurUpdate, T12MultipleTracksApplyWithFiniteCondition)
{
  UpdateOptions_s opts = default_options();
  // This test exercises the applied path for several tracks; the dedicated
  // T14 covers the production condition_limit rejection.
  opts.condition_limit_ = 0.0;
  SchurUpdater updater(opts);
  const CameraExtrinsics_s cam = make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d(0.02, 0.0, 0.0));
  std::vector<Eigen::Vector4d> qs;
  std::vector<Eigen::Vector3d> ps;
  smooth_trajectory(4, qs, ps);
  EkfState state;
  build_state(state, qs, ps);
  regularize_covariance(state);

  std::vector<FeatureTrack_s> tracks;
  const int num_tracks = 8;
  for (int f = 0; f < num_tracks; ++f) {
    FeatureTrack_s track;
    track.id_ = 1200 + f;
    const Eigen::Vector3d world_point(-0.8 + 0.25 * f, 0.2 * ((f % 4) - 1.5), 3.8 + 0.15 * f);
    for (int i = 0; i < 4; ++i) {
      track.observations_.push_back(make_observation(
        i, 0,
        project_world(qs[static_cast<std::size_t>(i)], ps[static_cast<std::size_t>(i)], cam, world_point)));
    }
    tracks.push_back(track);
  }

  const UpdateSummary_s summary = updater.update(state, tracks, {cam});
  std::printf(
    "[T12] applied=%d rank=%d n=%d cond=%.6e features=%d obs=%d\n", summary.applied_ ? 1 : 0, summary.rank_,
    state.size(), summary.condition_, summary.features_used_, summary.observations_used_);
  EXPECT_TRUE(summary.applied_);
  EXPECT_GT(summary.rank_, 0);
  EXPECT_GE(summary.condition_, 1.0);
  EXPECT_TRUE(std::isfinite(summary.condition_));
  EXPECT_EQ(summary.features_used_, num_tracks);
  EXPECT_EQ(summary.observations_used_, 4 * num_tracks);
  EXPECT_TRUE(state_is_finite(state));
}

// ===========================================================================
// T28 -- the chi2 gate projects the residual and the state Jacobian into the
// left null space of Hf.  It must not use the old unprojected profiled
// covariance C - Hf (Hf^T Hf)^-1 Hf^T.
// ===========================================================================
TEST(SchurUpdate, T28Chi2GateUsesExplicitPointNullspaceProjection)
{
  const UpdateOptions_s opts = default_options();
  SchurUpdater updater(opts);
  const CameraExtrinsics_s cam =
    make_camera(Eigen::Vector3d(0.02, -0.01, 0.03), Eigen::Vector3d(0.0, 0.0, 0.0));
  std::vector<Eigen::Vector4d> qs;
  std::vector<Eigen::Vector3d> ps;
  smooth_trajectory(4, qs, ps);
  EkfState state;
  build_state(state, qs, ps);
  regularize_covariance(state);

  const Eigen::Vector3d world_point(0.8, -0.4, 4.5);
  FeatureTrack_s track;
  track.id_ = 2800;
  // Non-zero residual: a fixed sub-pixel offset per observation.  With a
  // zero residual this test would only check 0 == 0.
  const Eigen::Vector2d offset_px(0.62, -0.41);
  for (int i = 0; i < 4; ++i) {
    Eigen::Vector2d uv =
      project_world(qs[static_cast<std::size_t>(i)], ps[static_cast<std::size_t>(i)], cam, world_point);
    uv += offset_px * (i % 2 == 0 ? 1.0 : -0.5) / K_FOCAL;
    track.observations_.push_back(make_observation(i, 0, uv));
  }
  ASSERT_TRUE(updater.triangulate(state, {cam}, track));
  const TrackReduction_s red = updater.reduceTrack(state, {cam}, track);
  ASSERT_TRUE(red.valid_);
  EXPECT_GT(std::abs(red.chi2_), 1e-6);

  const JointSystem_s joint = build_joint(updater, state, {cam}, track);
  const int m = 2 * static_cast<int>(track.observations_.size());
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(joint.Hf_, Eigen::ComputeFullU);
  const Eigen::VectorXd & singular = svd.singularValues();
  const double smax = singular.maxCoeff();
  // A_point = Hf^T Hf: the singular-value threshold is sqrt of the
  // eigenvalue threshold used by the point-block rank test.
  const double singular_rtol = std::sqrt(opts.point_block_rtol_);
  int rank = 0;
  for (int i = 0; i < singular.size(); ++i)
    if (singular(i) > singular_rtol * smax)
      ++rank;
  ASSERT_EQ(rank, 3);
  ASSERT_EQ(red.point_rank_, rank);
  ASSERT_EQ(red.gate_dof_, m - rank);

  const Eigen::MatrixXd N = svd.matrixU().rightCols(m - rank);
  const Eigen::VectorXd r0 = N.transpose() * joint.r_;
  const Eigen::MatrixXd H0 = N.transpose() * joint.Hx_;
  Eigen::MatrixXd S0 = H0 * state.covariance() * H0.transpose();
  S0.diagonal().array() += 1.0;
  Eigen::LDLT<Eigen::MatrixXd> ldlt(S0);
  ASSERT_EQ(ldlt.info(), Eigen::Success);
  EXPECT_GT(ldlt.vectorD().minCoeff(), 0.0);
  const Eigen::VectorXd y = ldlt.solve(r0);
  ASSERT_TRUE(y.allFinite());
  const double chi2_ref = r0.dot(y);

  std::printf(
    "[T28] rank=%d gate_dof=%d chi2=%.9f ref=%.9f diff=%.3e\n", red.point_rank_, red.gate_dof_, red.chi2_,
    chi2_ref, std::abs(red.chi2_ - chi2_ref));
  EXPECT_NEAR(red.chi2_, chi2_ref, 1e-9 * std::max(1.0, std::abs(chi2_ref)));

  // Residual components in range(Hf) are exactly what moving the 3-D point
  // can absorb.  They must not change the gate statistic.
  const Eigen::Vector3d point_shift(2.0e-3, -1.0e-3, 4.0e-4);
  const Eigen::VectorXd r_absorbed = joint.r_ + joint.Hf_ * point_shift;
  const Eigen::VectorXd r0_absorbed = N.transpose() * r_absorbed;
  EXPECT_LT((r0_absorbed - r0).norm(), 1e-12 * std::max(1.0, r0.norm()));
  const double chi2_absorbed = r0_absorbed.dot(ldlt.solve(r0_absorbed));
  EXPECT_NEAR(chi2_absorbed, chi2_ref, 1e-9 * std::max(1.0, std::abs(chi2_ref)));

  // The previous expression C - Hf (Hf^T Hf)^-1 Hf^T does not have this
  // invariance; the assertion below records the property that separated the
  // two formulations.
  Eigen::MatrixXd C = joint.Hx_ * state.covariance() * joint.Hx_.transpose();
  C.diagonal().array() += 1.0;
  const Eigen::MatrixXd S_old =
    C - joint.Hf_ * (joint.Hf_.transpose() * joint.Hf_).inverse() * joint.Hf_.transpose();
  const double chi2_old = joint.r_.dot(S_old.ldlt().solve(joint.r_));
  const double chi2_old_absorbed = r_absorbed.dot(S_old.ldlt().solve(r_absorbed));
  std::printf("[T28] old_profiled_chi2=%.9f absorbed=%.9f\n", chi2_old, chi2_old_absorbed);
  EXPECT_GT(std::abs(chi2_old_absorbed - chi2_old), 1e-6);
}

// ===========================================================================
// chi2_quantile_95 sanity (Wilson-Hilferty closed form).
// ===========================================================================
TEST(SchurUpdate, Chi2Quantile95ClosedForm)
{
  EXPECT_DOUBLE_EQ(SchurUpdater::chi2Quantile95(0), 0.0);
  EXPECT_DOUBLE_EQ(SchurUpdater::chi2Quantile95(-3), 0.0);
  // Reference values of the chi-square 0.95 quantile.
  EXPECT_NEAR(SchurUpdater::chi2Quantile95(1), 3.8415, 1.0e-1);
  EXPECT_NEAR(SchurUpdater::chi2Quantile95(2), 5.9915, 6.0e-2);
  EXPECT_NEAR(SchurUpdater::chi2Quantile95(6), 12.5916, 3.0e-2);
  EXPECT_NEAR(SchurUpdater::chi2Quantile95(20), 31.4104, 2.0e-1);
}

// ===========================================================================
// T27 -- P6 weak update rescues a chi2 gate collapse ("accept more, weight
// less").
//
// A drifted, over-confident filter whose clones no longer match the
// (truth-derived) observations produces per-track chi2 values beyond the
// already-loose 5x gate, so every track is rejected ("no gated tracks") and the
// filter would keep coasting on IMU alone until a full restart.  Setting
// UpdateOptions::weak_update_scale > 1.0 relaxes *only* the chi2 threshold,
// down-weights the accepted observations (sigma -> sigma * scale) and lowers
// the two starvation floors.  The same tracks then produce a small correction
// that still reduces the observable (gauge-invariant) pose error.
//
// Nominal sigma note: with the point profiled out of S the residual direction
// sees S = (I - Hf (Hf^T Hf)^-1 Hf^T) + Hx P Hx^T, and triangulate() rejects a
// point whose RMS reprojection error exceeds 5 px.  Together those bound chi2
// above by n_obs * (5 px / sigma_px)^2, so tripping the
// 5.0 * chi2_quantile_95(dof) gate on a *triangulatable* 6-observation track
// requires sigma_px < ~1.2 px.  This scenario therefore uses a 1 px nominal
// sigma while keeping the production 5.0 gate multiplier.
// ===========================================================================
TEST(SchurUpdate, T27WeakUpdateRescuesGateCollapse)
{
  const CameraExtrinsics_s cam = make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  const int kClones = 6;
  std::vector<Eigen::Vector4d> q_true;
  std::vector<Eigen::Vector3d> p_true;
  smooth_trajectory(kClones, q_true, p_true);

  // A static point cloud in front of the trajectory.
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> xy(-0.6, 0.6);
  std::uniform_real_distribution<double> z(3.0, 5.0);
  std::vector<Eigen::Vector3d> points;
  // Sequence draws explicitly: constructor argument evaluation order differs
  // between compilers/architectures. Preserve the original x86 fixture.
  for (int i = 0; i < 16; ++i) {
    const double pz = z(rng);
    const double py = xy(rng);
    const double px = xy(rng);
    points.emplace_back(px, py, pz);
  }

  // Observations are always generated from the truth poses.
  auto make_tracks = [&]()
  {
    std::vector<FeatureTrack_s> tracks;
    int id = 1000;
    for (const Eigen::Vector3d & wp : points) {
      FeatureTrack_s track;
      track.id_ = id++;
      for (int i = 0; i < kClones; ++i)
        track.observations_.push_back(make_observation(
          i, 0,
          project_world(q_true[static_cast<std::size_t>(i)], p_true[static_cast<std::size_t>(i)], cam, wp)));
      tracks.push_back(track);
    }
    return tracks;
  };

  // The drifted filter: nominal poses = truth perturbed *independently per
  // clone* (non-rigid, so the residual cannot be absorbed by the pose/scale
  // gauge or by moving the triangulated point).  The 14.5 mm / 14.5 mrad
  // magnitude puts the 6-observation chi2 just above the 5x gate while
  // keeping the triangulation RMS under the 5 px cap.
  std::vector<Eigen::Vector4d> q_nom = q_true;
  std::vector<Eigen::Vector3d> p_nom = p_true;
  std::mt19937 perturb_rng(7);
  std::uniform_real_distribution<double> off(-1.0, 1.0);
  for (int i = 0; i < kClones; ++i) {
    const double pz = 0.0145 * off(perturb_rng);
    const double py = 0.0145 * off(perturb_rng);
    const double px = 0.0145 * off(perturb_rng);
    p_nom[static_cast<std::size_t>(i)] += Eigen::Vector3d(px, py, pz);
    const double rz = 0.0145 * off(perturb_rng);
    const double ry = 0.0145 * off(perturb_rng);
    const double rx = 0.0145 * off(perturb_rng);
    q_nom[static_cast<std::size_t>(i)] =
      EkfState::quaternionMultiply(q_true[static_cast<std::size_t>(i)], quat(Eigen::Vector3d(rx, ry, rz)));
  }

  auto scenario_options = []()
  {
    UpdateOptions_s o = default_options();
    o.pixel_noise_ = 1.0;  // see the nominal-sigma note above
    return o;
  };
  auto build = [&](const std::vector<Eigen::Vector4d> & _qs, const std::vector<Eigen::Vector3d> & _ps)
  {
    EkfState state;
    build_state(state, _qs, _ps);
    regularize_covariance(state);
    return state;
  };

  // (1) Consistent state + consistent tracks -> a normal applied update, and
  //     the weak-update flag stays off (weak_update_scale defaults to 1.0).
  {
    UpdateOptions_s opts = scenario_options();
    SchurUpdater updater(opts);
    EkfState state = build(q_true, p_true);
    std::vector<FeatureTrack_s> tracks = make_tracks();
    const UpdateSummary_s summary = updater.update(state, tracks, {cam});
    std::printf(
      "[T27] (1) consistent applied=%d used=%d obs=%d weak=%d\n", summary.applied_ ? 1 : 0,
      summary.features_used_, summary.observations_used_, summary.weak_update_ ? 1 : 0);
    EXPECT_TRUE(summary.applied_);
    EXPECT_GT(summary.features_used_, 0);
    EXPECT_FALSE(summary.weak_update_);
    EXPECT_DOUBLE_EQ(summary.weak_update_scale_, 1.0);
  }

  // (2) Drifted state, weak update OFF (scale = 1.0): the 5x gate rejects
  //     every track and the state is untouched ("no gated tracks").
  {
    UpdateOptions_s opts = scenario_options();
    EXPECT_DOUBLE_EQ(opts.weak_update_scale_, 1.0);
    SchurUpdater updater(opts);
    EkfState state = build(q_nom, p_nom);
    const Snapshot_s before = snapshot(state);
    std::vector<FeatureTrack_s> tracks = make_tracks();
    const UpdateSummary_s summary = updater.update(state, tracks, {cam});
    std::printf(
      "[T27] (2) gated applied=%d used=%d rejected=%d reason=\"%s\"\n", summary.applied_ ? 1 : 0,
      summary.features_used_, summary.features_rejected_, summary.failure_reason_.c_str());
    EXPECT_FALSE(summary.applied_);
    EXPECT_EQ(summary.features_used_, 0);
    EXPECT_EQ(summary.failure_reason_, std::string("no gated tracks"));
    EXPECT_FALSE(summary.weak_update_);
    EXPECT_LT(snapshot_max_diff(before, snapshot(state)), 1e-12);
  }

  // (3) Same drifted state + tracks, weak update ON (scale = 10): the gate is
  //     relaxed, tracks are accepted, the update applies, the gauge-invariant
  //     pose error shrinks and P stays symmetric PSD.
  {
    UpdateOptions_s opts = scenario_options();
    opts.weak_update_scale_ = 10.0;
    SchurUpdater updater(opts);
    EkfState state = build(q_nom, p_nom);
    double before_pos = 0.0, before_rot = 0.0, after_pos = 0.0, after_rot = 0.0;
    pose_errors(state, q_true, p_true, true, before_pos, before_rot);
    std::vector<FeatureTrack_s> tracks = make_tracks();
    const UpdateSummary_s summary = updater.update(state, tracks, {cam});
    pose_errors(state, q_true, p_true, true, after_pos, after_rot);
    const Eigen::MatrixXd & P = state.covariance();
    const double asym = (P - P.transpose()).cwiseAbs().maxCoeff();
    std::printf(
      "[T27] (3) weak applied=%d used=%d obs=%d weak=%d scale=%.1f "
      "rel-pos %.4e -> %.4e asym=%.1e\n",
      summary.applied_ ? 1 : 0, summary.features_used_, summary.observations_used_,
      summary.weak_update_ ? 1 : 0, summary.weak_update_scale_, before_pos, after_pos, asym);
    EXPECT_TRUE(summary.applied_);
    EXPECT_GT(summary.features_used_, 0);
    EXPECT_TRUE(summary.weak_update_);
    EXPECT_DOUBLE_EQ(summary.weak_update_scale_, 10.0);
    // A visual-only update cannot observe the six-dof world gauge, so the
    // gauge-invariant (relative) position error is the meaningful metric.
    EXPECT_LT(after_pos, before_pos);
    EXPECT_LT(asym, 1e-12);
    EXPECT_TRUE(EkfState::isPositiveSemidefinite(P));
    EXPECT_TRUE(state_is_finite(state));
  }

  // (4) Down-weighting: compare the weak update against a control that
  //     relaxes the gate by the *same* factor (chi2_multiplier * 10) but does
  //     NOT inflate sigma.  Both accept the same tracks; the weak correction
  //     must be far smaller because each observation carries sigma * scale.
  {
    UpdateOptions_s ctrl = scenario_options();
    ctrl.chi2_multiplier_ = ctrl.chi2_multiplier_ * 10.0;  // same relaxed gate
    ctrl.weak_update_scale_ = 1.0;                         // no down-weight
    SchurUpdater ctrl_updater(ctrl);
    EkfState ctrl_state = build(q_nom, p_nom);
    const Snapshot_s ctrl_before = snapshot(ctrl_state);
    std::vector<FeatureTrack_s> ctrl_tracks = make_tracks();
    const UpdateSummary_s ctrl_summary = ctrl_updater.update(ctrl_state, ctrl_tracks, {cam});
    const double dx_ctrl = reconstruct_dx(ctrl_before, ctrl_state).norm();

    UpdateOptions_s weak = scenario_options();
    weak.weak_update_scale_ = 10.0;
    SchurUpdater weak_updater(weak);
    EkfState weak_state = build(q_nom, p_nom);
    const Snapshot_s weak_before = snapshot(weak_state);
    std::vector<FeatureTrack_s> weak_tracks = make_tracks();
    const UpdateSummary_s weak_summary = weak_updater.update(weak_state, weak_tracks, {cam});
    const double dx_weak = reconstruct_dx(weak_before, weak_state).norm();

    std::printf(
      "[T27] (4) |dx| ctrl(no dw)=%.3e weak(dw)=%.3e ratio=%.3f\n", dx_ctrl, dx_weak,
      dx_ctrl > 0.0 ? dx_weak / dx_ctrl : 0.0);
    ASSERT_TRUE(ctrl_summary.applied_);
    ASSERT_FALSE(ctrl_summary.weak_update_);
    ASSERT_TRUE(weak_summary.applied_);
    ASSERT_TRUE(weak_summary.weak_update_);
    // Same accepted set, so the only remaining difference is the weight.
    EXPECT_EQ(weak_summary.features_used_, ctrl_summary.features_used_);
    ASSERT_GT(dx_ctrl, 0.0);
    EXPECT_GT(dx_weak, 0.0);
    EXPECT_LT(dx_weak, dx_ctrl / 5.0);
  }
}

// ===========================================================================
// T29 -- the thread-local reduce scratch is invisible: same-shape repeats,
// alternating track/window shapes, interleaved SchurUpdater instances and a
// failed reduction must all reproduce the independent joint Schur complement.
// Uses only the public (pre-optimization) API so the same source can be linked
// against the baseline and optimized implementations.
// ===========================================================================
namespace
{

void expect_joint_reduction(
  const SchurUpdater & _updater, const EkfState & _state, const std::vector<CameraExtrinsics_s> & _cameras,
  const FeatureTrack_s & _track, const TrackReduction_s & _red, const char * _label)
{
  EXPECT_TRUE(_red.valid_) << _label;
  if (!_red.valid_)
    return;

  const JointSystem_s joint = build_joint(_updater, _state, _cameras, _track);
  Eigen::MatrixXd A_xf = Eigen::MatrixXd::Zero(_state.size(), 3);
  for (const auto & w : _red.W_)
    A_xf.block<6, 3>(_state.cloneOffset(w.first), 0) = w.second;
  const Eigen::MatrixXd A_xf_ref = joint.Hx_.transpose() * joint.Hf_;
  const Eigen::Matrix3d A_ff_ref = joint.Hf_.transpose() * joint.Hf_;
  const Eigen::Vector3d b_f_ref = joint.Hf_.transpose() * joint.r_;
  EXPECT_LT((A_xf - A_xf_ref).norm(), 1e-9 * std::max(1.0, A_xf_ref.norm())) << _label;
  EXPECT_LT((_red.A_point_ - A_ff_ref).norm(), 1e-9 * std::max(1.0, A_ff_ref.norm())) << _label;
  EXPECT_LT((_red.b_point_ - b_f_ref).norm(), 1e-9 * std::max(1.0, b_f_ref.norm())) << _label;

  Eigen::MatrixXd A_red, A_ref;
  Eigen::VectorXd b_red, b_ref;
  reduce_with_blocks(_state, joint, _red, A_red, b_red);
  reduce_exact(joint, A_ref, b_ref);
  // Noiseless synthetic tracks can make b_ref exactly zero, so use a scale
  // that stays well defined instead of dividing by b_ref.norm().
  EXPECT_LT((A_red - A_ref).norm(), 1e-9 * std::max(1.0, A_ref.norm())) << _label;
  EXPECT_LT((b_red - b_ref).norm(), 1e-9 * std::max(1.0, b_ref.norm())) << _label;
}

}  // namespace

TEST(SchurUpdate, T29ReduceScratchReuseAcrossShapesInstancesAndFailures)
{
  const UpdateOptions_s opts = default_options();
  SchurUpdater updater(opts);
  UpdateOptions_s noisy_opts = default_options();
  noisy_opts.pixel_noise_ = 2.0;  // different sigma -> an independent reference
  SchurUpdater updater_noisy(noisy_opts);
  const CameraExtrinsics_s cam =
    make_camera(Eigen::Vector3d(0.01, -0.02, 0.015), Eigen::Vector3d(0.02, 0.01, -0.01));

  std::vector<Eigen::Vector4d> qs_short, qs_long;
  std::vector<Eigen::Vector3d> ps_short, ps_long;
  smooth_trajectory(4, qs_short, ps_short);
  smooth_trajectory(6, qs_long, ps_long);
  EkfState state_short, state_long;
  build_state(state_short, qs_short, ps_short);
  build_state(state_long, qs_long, ps_long);
  regularize_covariance(state_short);
  regularize_covariance(state_long);

  const Eigen::Vector3d point(0.8, 0.6, 4.5);
  FeatureTrack_s track_short;
  track_short.id_ = 2900;
  for (int i = 0; i < state_short.numClones(); ++i) {
    track_short.observations_.push_back(
      make_observation(i, 0, project_world(state_short.clone(i).q_, state_short.clone(i).p_, cam, point)));
  }
  ASSERT_TRUE(updater.triangulate(state_short, {cam}, track_short));

  FeatureTrack_s track_long;
  track_long.id_ = 2901;
  for (int i = 0; i < state_long.numClones(); ++i) {
    track_long.observations_.push_back(
      make_observation(i, 0, project_world(state_long.clone(i).q_, state_long.clone(i).p_, cam, point)));
  }
  ASSERT_TRUE(updater.triangulate(state_long, {cam}, track_long));

  // Observes only clones 0..2 of the six-clone window, so the remaining Hx
  // columns must be zero-filled rather than inherit the previous track.
  FeatureTrack_s track_partial;
  track_partial.id_ = 2903;
  for (int i = 0; i < 3; ++i) {
    track_partial.observations_.push_back(
      make_observation(i, 0, project_world(state_long.clone(i).q_, state_long.clone(i).p_, cam, point)));
  }
  ASSERT_TRUE(updater.triangulate(state_long, {cam}, track_partial));

  // (1) Same shape repeated: identical reduction values.
  const TrackReduction_s r1 = updater.reduceTrack(state_short, {cam}, track_short);
  const TrackReduction_s r2 = updater.reduceTrack(state_short, {cam}, track_short);
  expect_joint_reduction(updater, state_short, {cam}, track_short, r1, "short-1");
  expect_joint_reduction(updater, state_short, {cam}, track_short, r2, "short-2");
  EXPECT_DOUBLE_EQ(r1.chi2_, r2.chi2_);
  EXPECT_DOUBLE_EQ(r1.rms_residual_px_, r2.rms_residual_px_);
  EXPECT_EQ(r1.gate_dof_, r2.gate_dof_);
  EXPECT_EQ(r1.point_rank_, r2.point_rank_);
  EXPECT_LT((r1.A_point_ - r2.A_point_).norm(), 1e-15 * std::max(1.0, r1.A_point_.norm()));
  EXPECT_LT((r1.b_point_ - r2.b_point_).norm(), 1e-15 * std::max(1.0, r1.b_point_.norm()));

  // (2) Alternate window/observation shapes. The partial track leaves Hx
  //     columns for clones 3..5 untouched, which catches stale scratch data.
  const TrackReduction_s long_red = updater.reduceTrack(state_long, {cam}, track_long);
  expect_joint_reduction(updater, state_long, {cam}, track_long, long_red, "long");
  const TrackReduction_s partial_red = updater.reduceTrack(state_long, {cam}, track_partial);
  expect_joint_reduction(updater, state_long, {cam}, track_partial, partial_red, "partial");
  const TrackReduction_s short_again = updater.reduceTrack(state_short, {cam}, track_short);
  expect_joint_reduction(updater, state_short, {cam}, track_short, short_again, "short-again");
  const TrackReduction_s long_again = updater.reduceTrack(state_long, {cam}, track_long);
  expect_joint_reduction(updater, state_long, {cam}, track_long, long_again, "long-again");
  EXPECT_DOUBLE_EQ(long_red.chi2_, long_again.chi2_);
  EXPECT_LT((long_red.A_point_ - long_again.A_point_).norm(), 1e-15 * std::max(1.0, long_red.A_point_.norm()));

  // (3) Two updater instances share this thread's scratch; interleave them and
  //     compare each against its own sigma-dependent joint reference.
  const TrackReduction_s noisy_red = updater_noisy.reduceTrack(state_short, {cam}, track_short);
  expect_joint_reduction(updater_noisy, state_short, {cam}, track_short, noisy_red, "noisy");
  const TrackReduction_s nominal_red = updater.reduceTrack(state_short, {cam}, track_short);
  expect_joint_reduction(updater, state_short, {cam}, track_short, nominal_red, "nominal-after-noisy");
  // The exact synthetic residuals make chi2 ~ 0 for both sigmas; A_point
  // scales as 1/sigma^2 and therefore still distinguishes the two updaters.
  EXPECT_GT(
    (noisy_red.A_point_ - nominal_red.A_point_).norm(), 1e-6 * std::max(1.0, nominal_red.A_point_.norm()));

  // (4) A failing reduction must not poison the next valid one.
  FeatureTrack_s degenerate;
  degenerate.id_ = 2902;
  degenerate.world_point_ = point;
  degenerate.has_point_ = true;
  const Eigen::Vector2d uv = project_world(state_short.clone(0).q_, state_short.clone(0).p_, cam, point);
  degenerate.observations_.push_back(make_observation(0, 0, uv));
  degenerate.observations_.push_back(make_observation(0, 0, uv));
  EXPECT_FALSE(updater.reduceTrack(state_short, {cam}, degenerate).valid_);
  const TrackReduction_s after_failure = updater.reduceTrack(state_short, {cam}, track_short);
  expect_joint_reduction(updater, state_short, {cam}, track_short, after_failure, "after-failure");
  EXPECT_DOUBLE_EQ(r1.chi2_, after_failure.chi2_);
}

// ===========================================================================
// T30 -- deterministic repeated reduce/update workload. This is the replayable
// driver for an external baseline-vs-optimized numeric/timing comparison; the
// same source links against either implementation through the public API.
// There is deliberately no wall-clock assertion: timing is only recorded.
// ===========================================================================
TEST(SchurUpdate, T30DeterministicRepeatReduceUpdateWorkload)
{
  UpdateOptions_s opts = default_options();
  opts.max_features_ = 40;
  opts.condition_limit_ = 0.0;  // exercise the full update on this tight scene
  SchurUpdater updater(opts);
  const CameraExtrinsics_s cam =
    make_camera(Eigen::Vector3d(0.01, -0.01, 0.005), Eigen::Vector3d(0.02, 0.0, -0.01));

  std::vector<Eigen::Vector4d> qs;
  std::vector<Eigen::Vector3d> ps;
  smooth_trajectory(5, qs, ps);
  EkfState state;
  build_state(state, qs, ps);
  regularize_covariance(state);

  std::vector<FeatureTrack_s> tracks;
  for (int f = 0; f < 24; ++f) {
    const int len = 3 + (f % 3);  // 3..5 observations
    const int start = f % 3;      // first observing clone
    const int stop = std::min(start + len, state.numClones());
    const Eigen::Vector3d point(0.55 + 0.07 * (f % 6), 0.35 + 0.09 * (f / 6), 3.8 + 0.13 * (f % 4));
    FeatureTrack_s track;
    track.id_ = 3000 + f;
    for (int ci = start; ci < stop; ++ci) {
      Eigen::Vector2d uv = project_world(state.clone(ci).q_, state.clone(ci).p_, cam, point);
      const double jitter = 0.0006 * std::sin(0.7 * (f + 1) * (ci + 1));
      track.observations_.push_back(make_observation(ci, 0, uv + Eigen::Vector2d(jitter, -0.5 * jitter)));
    }
    if (!updater.triangulate(state, {cam}, track))
      continue;
    tracks.push_back(track);
  }
  ASSERT_GE(tracks.size(), 12u);

  // Warm up both paths once (scratch allocation happens here, and it also
  // confirms the synthetic scene exercises an applied update).
  {
    EkfState warm_state = state;
    std::vector<FeatureTrack_s> warm_tracks = tracks;
    const UpdateSummary_s warm = updater.update(warm_state, warm_tracks, {cam});
    ASSERT_TRUE(warm.applied_);
    ASSERT_GT(warm.features_used_, 0);
  }

  constexpr int kReduceIterations = 200;
  constexpr int kUpdateIterations = 40;
  const int reduce_calls_per_iter = static_cast<int>(tracks.size());

  double reduce_checksum = 0.0;
  const auto reduce_t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < kReduceIterations; ++it) {
    for (const FeatureTrack_s & track : tracks) {
      const TrackReduction_s red = updater.reduceTrack(state, {cam}, track);
      if (!red.valid_)
        continue;
      reduce_checksum += red.chi2_ + red.rms_residual_px_ + red.A_point_.sum() + red.b_point_.sum() +
                         static_cast<double>(red.gate_dof_);
    }
  }
  const auto reduce_t1 = std::chrono::steady_clock::now();
  const double reduce_ms = std::chrono::duration<double, std::milli>(reduce_t1 - reduce_t0).count();

  double update_checksum = 0.0;
  const auto update_t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < kUpdateIterations; ++it) {
    EkfState work_state = state;
    std::vector<FeatureTrack_s> work_tracks = tracks;
    const UpdateSummary_s summary = updater.update(work_state, work_tracks, {cam});
    update_checksum += work_state.covariance().trace() + static_cast<double>(summary.rank_) +
                       static_cast<double>(summary.features_used_) +
                       static_cast<double>(summary.observations_used_);
  }
  const auto update_t1 = std::chrono::steady_clock::now();
  const double update_ms = std::chrono::duration<double, std::milli>(update_t1 - update_t0).count();

  char reduce_text[32];
  char update_text[32];
  std::snprintf(reduce_text, sizeof(reduce_text), "%.17g", reduce_checksum);
  std::snprintf(update_text, sizeof(update_text), "%.17g", update_checksum);
  RecordProperty("reduce_iterations", kReduceIterations);
  RecordProperty("reduce_calls", kReduceIterations * reduce_calls_per_iter);
  RecordProperty("reduce_elapsed_ms", reduce_ms);
  RecordProperty("reduce_checksum", reduce_text);
  RecordProperty("update_iterations", kUpdateIterations);
  RecordProperty("update_elapsed_ms", update_ms);
  RecordProperty("update_checksum", update_text);
  std::printf(
    "[T30] reduce: iters=%d calls=%d elapsed_ms=%.3f checksum=%s\n", kReduceIterations,
    kReduceIterations * reduce_calls_per_iter, reduce_ms, reduce_text);
  std::printf("[T30] update: iters=%d elapsed_ms=%.3f checksum=%s\n", kUpdateIterations, update_ms, update_text);
  EXPECT_TRUE(std::isfinite(reduce_checksum));
  EXPECT_TRUE(std::isfinite(update_checksum));
}
