// Standalone QR/SVD and sequential/batch IMU backend microbenchmark.
//
// Eigen only: no ROS, no gtest, no CMake target.  Link this source against the
// candidate backend sources (or a candidate library exporting the same symbols):
//
//   g++ -std=c++17 -O2 -DNDEBUG -I src/vins_estimator/src -I /usr/include/eigen3 \
//       scripts/benchmark_backend_ablation.cpp \
//       src/vins_estimator/src/backend/ekf_state.cpp \
//       src/vins_estimator/src/backend/schur_update.cpp \
//       -o benchmark_backend_ablation
//
// The historical SVD path is the reference for the QR comparison.  The
// sequential IMU covariance path is the reference for the batch path.  All
// timing regions contain only the production calls under test: scene building,
// state/track reset copies, warmup and validation are outside the timed regions.
//
// Fixed scene: 5 clones, 120 stereo tracks, nonzero pixel noise.  The scene is
// deterministically generated from --seed.  Output is machine-readable JSON
// and/or readable key=value lines; checksums must match across builds with the
// same source and compiler/runtime, while wall-clock timings are not expected
// to match.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend/ekf_state.h"
#include "backend/schur_update.h"

namespace
{

using namespace vins::backend;

constexpr unsigned int K_SEED = 20261115u;
constexpr int K_CLONES = 5;
constexpr int K_TRACKS = 120;
constexpr int K_CAMERAS = 2;
constexpr int K_FLUSH_SAMPLES = 20;
constexpr int K_DEFAULT_ITERATIONS = 200;
constexpr int K_DEFAULT_WARMUP = 5;
constexpr double K_DT = 0.005;
constexpr double K_GRAVITY = 9.81007;
constexpr double K_FOCAL_LENGTH = 460.0;
constexpr double K_PIXEL_NOISE_PX = 0.35;
constexpr double K_TWO_PI = 6.283185307179586476925286766559;
constexpr int K_BUDGETS[] = {240, 480, 0};

// std::normal_distribution is not specified bit-for-bit across standard
// libraries.  This small Box-Muller generator uses the standardized mt19937
// stream and elementary functions so repeated local/ARM checksums are stable.
class DeterministicNoise
{
public:
  explicit DeterministicNoise(unsigned int _seed) : rng_(_seed) {}

  double uniform()
  {
    return (static_cast<double>(rng_()) + 0.5) / 4294967296.0;
  }

  double normal()
  {
    if (has_spare_) {
      has_spare_ = false;
      return spare_;
    }
    const double u1 = std::max(uniform(), 1e-300);
    const double u2 = uniform();
    const double radius = std::sqrt(-2.0 * std::log(u1));
    const double angle = K_TWO_PI * u2;
    spare_ = radius * std::sin(angle);
    has_spare_ = true;
    return radius * std::cos(angle);
  }

private:
  std::mt19937 rng_;
  bool has_spare_ = false;
  double spare_ = 0.0;
};

Eigen::Vector4d quatFromAxisAngle(const Eigen::Vector3d & _axis_angle)
{
  return EkfState::quaternionFromAxisAngle(_axis_angle);
}

CameraExtrinsics_s makeCamera(int _id)
{
  CameraExtrinsics_s cam;
  if (_id == 0) {
    cam.q_cb_ = quatFromAxisAngle(Eigen::Vector3d(0.006, -0.004, 0.003));
    cam.p_cb_ = Eigen::Vector3d::Zero();
  } else {
    cam.q_cb_ = quatFromAxisAngle(Eigen::Vector3d(0.004, -0.003, 0.002));
    cam.p_cb_ = Eigen::Vector3d(-0.15, 0.0, 0.0);  // 15 cm stereo baseline
  }
  return cam;
}

Eigen::Vector2d projectWorld(
  const Eigen::Vector4d & _q, const Eigen::Vector3d & _p, const CameraExtrinsics_s & _cam,
  const Eigen::Vector3d & _world_point)
{
  const Eigen::Matrix3d R_i = EkfState::rotation(_q);
  const Eigen::Matrix3d R_cb = EkfState::rotation(_cam.q_cb_);
  const Eigen::Vector3d u = R_i.transpose() * (_world_point - _p);
  const Eigen::Vector3d p_c = R_cb.transpose() * (u - _cam.p_cb_);
  if (std::abs(p_c.z()) < 1e-9)
    return Eigen::Vector2d::Zero();
  return Eigen::Vector2d(p_c.x() / p_c.z(), p_c.y() / p_c.z());
}

struct ScalarError_s {
  double max_abs = 0.0;
  double rms = 0.0;
  double sum_sq = 0.0;
  int count = 0;
  bool finite = true;

  void add(const Eigen::Vector3d & _diff)
  {
    if (!_diff.allFinite()) {
      finite = false;
      return;
    }
    max_abs = std::max(max_abs, _diff.cwiseAbs().maxCoeff());
    sum_sq += _diff.squaredNorm();
    count += 3;
  }

  void finish() { rms = count > 0 ? std::sqrt(sum_sq / static_cast<double>(count)) : 0.0; }
};

struct MatrixError_s {
  double rel_fro = 0.0;
  double max_abs = 0.0;
  double ref_fro = 0.0;
  double diff_fro = 0.0;
  bool finite = true;
};

ScalarError_s compareMeans(const EkfState & _ref, const EkfState & _test)
{
  ScalarError_s error;
  if (_ref.numClones() != _test.numClones()) {
    error.finite = false;
    return error;
  }

  auto add_quat = [&error](const Eigen::Vector4d & _qr, Eigen::Vector4d _qt)
  {
    if (!_qr.allFinite() || !_qt.allFinite()) {
      error.finite = false;
      return;
    }
    if (_qr.dot(_qt) < 0.0)
      _qt = -_qt;
    const Eigen::Vector4d inverse_ref(_qr[0], -_qr[1], -_qr[2], -_qr[3]);
    const Eigen::Vector4d dq = EkfState::quaternionMultiply(inverse_ref, _qt);
    error.add(EkfState::quaternionToAxisAngle(dq));
  };

  add_quat(_ref.imu().q_, _test.imu().q_);
  error.add(_ref.imu().p_ - _test.imu().p_);
  error.add(_ref.imu().v_ - _test.imu().v_);
  error.add(_ref.imu().ba_ - _test.imu().ba_);
  error.add(_ref.imu().bg_ - _test.imu().bg_);
  for (int i = 0; i < _ref.numClones(); ++i) {
    add_quat(_ref.clone(i).q_, _test.clone(i).q_);
    error.add(_ref.clone(i).p_ - _test.clone(i).p_);
  }
  error.finish();
  return error;
}

MatrixError_s compareCovariances(const EkfState & _ref, const EkfState & _test)
{
  MatrixError_s error;
  const Eigen::MatrixXd & ref = _ref.covariance();
  const Eigen::MatrixXd & test = _test.covariance();
  if (ref.rows() != test.rows() || ref.cols() != test.cols() || !ref.allFinite() || !test.allFinite()) {
    error.finite = false;
    return error;
  }
  const Eigen::MatrixXd diff = ref - test;
  error.ref_fro = ref.norm();
  error.diff_fro = diff.norm();
  error.rel_fro = error.diff_fro / std::max(error.ref_fro, 1e-12);
  error.max_abs = diff.size() > 0 ? diff.cwiseAbs().maxCoeff() : 0.0;
  return error;
}

EkfState buildTruthState(
  const std::vector<Eigen::Vector4d> & _qs, const std::vector<Eigen::Vector3d> & _ps,
  const EkfOptions_s & _options)
{
  EkfState state;
  state.reset(
    _options, 0.0, _qs.front(), _ps.front(), Eigen::Vector3d(0.3, 0.0, 0.0), Eigen::Vector3d::Zero(),
    Eigen::Vector3d::Zero());
  state.augmentClone(0.0);
  for (std::size_t i = 1; i < _qs.size(); ++i) {
    state.imu().q_ = _qs[i];
    state.imu().p_ = _ps[i];
    state.refreshImuFej();
    state.augmentClone(0.2 * static_cast<double>(i));
  }

  // A short deterministic IMU stretch gives the visual update nontrivial live
  // to clone cross-covariance instead of a block-diagonal fresh state.
  state.covariance().diagonal().array() += 1e-9;
  DeterministicNoise noise(K_SEED ^ 0x51u);
  for (int k = 0; k < 40; ++k) {
    const double t = K_DT * static_cast<double>(k);
    Eigen::Vector3d acc(
      0.02 * std::sin(0.3 * t), 0.01 * std::cos(0.2 * t), _options.gravity_ + 0.03 * std::sin(0.17 * t));
    Eigen::Vector3d gyr(0.002 * std::sin(0.4 * t), 0.001 * std::cos(0.35 * t), 0.001 * std::sin(0.2 * t));
    acc += 0.02 * Eigen::Vector3d(noise.normal(), noise.normal(), noise.normal());
    gyr += 0.001 * Eigen::Vector3d(noise.normal(), noise.normal(), noise.normal());
    state.propagate(acc, gyr, K_DT);
  }
  return state;
}

void perturbState(EkfState & _state)
{
  _state.imu().q_ = EkfState::quaternionMultiply(
    _state.imu().q_, quatFromAxisAngle(Eigen::Vector3d(0.003, -0.002, 0.0015)));
  _state.imu().p_ += Eigen::Vector3d(0.012, -0.009, 0.007);
  _state.imu().v_ += Eigen::Vector3d(0.010, -0.008, 0.006);
  _state.imu().ba_ += Eigen::Vector3d(0.002, -0.001, 0.001);
  _state.imu().bg_ += Eigen::Vector3d(0.0002, -0.0001, 0.0001);

  // Keep clone FEJ values at the truth poses: the residual is evaluated at the
  // perturbed mean while the Jacobian uses the frozen linearisation point.
  for (int i = 0; i < _state.numClones(); ++i) {
    const double fi = static_cast<double>(i);
    const Eigen::Vector3d axis_angle(
      0.0025 * std::sin(0.9 * fi), -0.0020 * std::cos(0.7 * fi), 0.0015 * std::sin(1.1 * fi));
    _state.clone(i).q_ = EkfState::quaternionMultiply(_state.clone(i).q_, quatFromAxisAngle(axis_angle));
    _state.clone(i).p_ +=
      Eigen::Vector3d(0.008 * std::cos(0.5 * fi), 0.006 * std::sin(0.8 * fi), 0.004 * std::cos(1.2 * fi));
  }
}

struct Scene_s {
  EkfState state_;
  std::vector<FeatureTrack_s> tracks_;
  std::vector<CameraExtrinsics_s> cameras_;
  int intended_tracks_ = 0;
  int intended_observations_ = 0;
  int actual_observations_ = 0;
  int triangulation_failures_ = 0;
  ScalarError_s initial_mean_perturbation_;
};

Scene_s makeScene(unsigned int _seed)
{
  Scene_s scene;
  scene.cameras_.push_back(makeCamera(0));
  scene.cameras_.push_back(makeCamera(1));

  std::vector<Eigen::Vector4d> qs;
  std::vector<Eigen::Vector3d> ps;
  for (int i = 0; i < K_CLONES; ++i) {
    const double t = 0.2 * static_cast<double>(i);
    ps.push_back(Eigen::Vector3d(0.30 * t, 0.03 * t * t, 0.02 * t));
    qs.push_back(quatFromAxisAngle(Eigen::Vector3d(0.0, -0.018 * t, 0.012 * t)));
  }

  EkfOptions_s options;
  options.window_size_ = K_CLONES;
  options.batch_imu_covariance_ = false;
  const EkfState truth = buildTruthState(qs, ps, options);
  scene.state_ = truth;
  perturbState(scene.state_);
  scene.initial_mean_perturbation_ = compareMeans(truth, scene.state_);

  UpdateOptions_s triangulation_options;
  triangulation_options.max_features_ = K_TRACKS;
  triangulation_options.min_observations_ = 2;
  triangulation_options.min_features_ = 1;
  triangulation_options.min_update_observations_ = 1;
  triangulation_options.condition_limit_ = 0.0;
  SchurUpdater triangulator(triangulation_options);

  DeterministicNoise pixel_noise(_seed ^ 0x9e3779b9u);
  const double sigma_uv = K_PIXEL_NOISE_PX / K_FOCAL_LENGTH;
  scene.intended_tracks_ = K_TRACKS;
  for (int f = 0; f < K_TRACKS; ++f) {
    const int col = f % 12;
    const int row = f / 12;
    const Eigen::Vector3d point(
      -0.66 + 0.12 * static_cast<double>(col), -0.54 + 0.108 * static_cast<double>(row),
      3.0 + 0.10 * static_cast<double>(f % 7));

    FeatureTrack_s track;
    track.id_ = 7000 + f;
    for (int ci = 0; ci < K_CLONES; ++ci) {
      for (int cam = 0; cam < K_CAMERAS; ++cam) {
        FeatureObservation_s observation;
        observation.clone_index_ = ci;
        observation.camera_id_ = cam;
        observation.uv_ = projectWorld(
          truth.clone(ci).q_, truth.clone(ci).p_, scene.cameras_[static_cast<std::size_t>(cam)], point);
        observation.uv_ += Eigen::Vector2d(sigma_uv * pixel_noise.normal(), sigma_uv * pixel_noise.normal());
        track.observations_.push_back(observation);
        ++scene.intended_observations_;
      }
    }
    if (triangulator.triangulate(scene.state_, scene.cameras_, track)) {
      scene.tracks_.push_back(track);
    } else {
      ++scene.triangulation_failures_;
    }
  }

  for (const FeatureTrack_s & track : scene.tracks_)
    scene.actual_observations_ += static_cast<int>(track.observations_.size());
  return scene;
}

struct ReduceResult_s {
  std::string backend_;
  int iterations_ = 0;
  int tracks_ = 0;
  long long calls_ = 0;
  double elapsed_ms_ = 0.0;
  double us_per_call_ = 0.0;
  double checksum_ = 0.0;
};

ReduceResult_s measureReduce(
  const std::string & _backend, SchurUpdater & _updater, const EkfState & _state,
  const std::vector<CameraExtrinsics_s> & _cameras, const std::vector<FeatureTrack_s> & _tracks,
  int _iterations)
{
  ReduceResult_s result;
  result.backend_ = _backend;
  result.iterations_ = _iterations;
  result.tracks_ = static_cast<int>(_tracks.size());
  result.calls_ = static_cast<long long>(_iterations) * static_cast<long long>(_tracks.size());

  double checksum = 0.0;
  const auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < _iterations; ++it) {
    for (const FeatureTrack_s & track : _tracks) {
      const TrackReduction_s red = _updater.reduceTrack(_state, _cameras, track);
      if (!red.valid_)
        continue;
      checksum += red.chi2_ + red.rms_residual_px_ + red.A_point_.sum() + red.b_point_.sum() +
                  static_cast<double>(red.gate_dof_);
    }
  }
  const auto t1 = std::chrono::steady_clock::now();
  result.elapsed_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
  result.us_per_call_ =
    result.calls_ > 0 ? 1000.0 * result.elapsed_ms_ / static_cast<double>(result.calls_) : 0.0;
  result.checksum_ = checksum;
  return result;
}

struct ReduceValidation_s {
  int svd_valid_ = 0;
  int qr_valid_ = 0;
  int invalid_mismatch_ = 0;
  int gate_dof_mismatch_ = 0;
  int point_rank_mismatch_ = 0;
  int w_key_mismatch_ = 0;
  int finite_violations_ = 0;
  int chi2_finite_svd_ = 0;
  int chi2_nonfinite_svd_ = 0;
  int chi2_finite_qr_ = 0;
  int chi2_nonfinite_qr_ = 0;
  double max_a_point_rel_fro_ = 0.0;
  double max_b_point_rel_ = 0.0;
  double max_chi2_abs_ = 0.0;
  double max_rms_abs_ = 0.0;
  double max_w_rel_fro_ = 0.0;
  bool finite_ = true;
};

ReduceValidation_s validateReductions(
  SchurUpdater & _svd, SchurUpdater & _qr, const EkfState & _state,
  const std::vector<CameraExtrinsics_s> & _cameras, const std::vector<FeatureTrack_s> & _tracks)
{
  ReduceValidation_s result;
  for (const FeatureTrack_s & track : _tracks) {
    const TrackReduction_s ref = _svd.reduceTrack(_state, _cameras, track);
    const TrackReduction_s test = _qr.reduceTrack(_state, _cameras, track);
    if (ref.valid_)
      ++result.svd_valid_;
    if (test.valid_)
      ++result.qr_valid_;
    if (ref.valid_ != test.valid_)
      ++result.invalid_mismatch_;

    if (ref.valid_) {
      if (std::isfinite(ref.chi2_))
        ++result.chi2_finite_svd_;
      else
        ++result.chi2_nonfinite_svd_;
    }
    if (test.valid_) {
      if (std::isfinite(test.chi2_))
        ++result.chi2_finite_qr_;
      else
        ++result.chi2_nonfinite_qr_;
    }

    if (!ref.A_point_.allFinite() || !ref.b_point_.allFinite() || !std::isfinite(ref.chi2_) ||
        !std::isfinite(ref.rms_residual_px_) || !test.A_point_.allFinite() || !test.b_point_.allFinite() ||
        !std::isfinite(test.chi2_) || !std::isfinite(test.rms_residual_px_)) {
      ++result.finite_violations_;
      result.finite_ = false;
    }

    if (!ref.valid_ || !test.valid_)
      continue;

    if (ref.gate_dof_ != test.gate_dof_)
      ++result.gate_dof_mismatch_;
    if (ref.point_rank_ != test.point_rank_)
      ++result.point_rank_mismatch_;

    const double a_point_rel = (test.A_point_ - ref.A_point_).norm() / std::max(ref.A_point_.norm(), 1e-12);
    const double b_point_rel = (test.b_point_ - ref.b_point_).norm() / std::max(ref.b_point_.norm(), 1e-12);
    result.max_a_point_rel_fro_ = std::max(result.max_a_point_rel_fro_, a_point_rel);
    result.max_b_point_rel_ = std::max(result.max_b_point_rel_, b_point_rel);
    result.max_chi2_abs_ = std::max(result.max_chi2_abs_, std::abs(test.chi2_ - ref.chi2_));
    result.max_rms_abs_ = std::max(result.max_rms_abs_, std::abs(test.rms_residual_px_ - ref.rms_residual_px_));

    std::set<int> keys;
    for (const auto & item : ref.W_)
      keys.insert(item.first);
    for (const auto & item : test.W_)
      keys.insert(item.first);
    for (const int key : keys) {
      const auto ref_it = ref.W_.find(key);
      const auto test_it = test.W_.find(key);
      if (ref_it == ref.W_.end() || test_it == test.W_.end()) {
        ++result.w_key_mismatch_;
        continue;
      }
      const double w_rel = (test_it->second - ref_it->second).norm() / std::max(ref_it->second.norm(), 1e-12);
      result.max_w_rel_fro_ = std::max(result.max_w_rel_fro_, w_rel);
    }
  }
  return result;
}

struct UpdateResult_s {
  std::string backend_;
  int iterations_ = 0;
  long long calls_ = 0;
  double elapsed_ms_ = 0.0;
  double us_per_update_ = 0.0;
  double checksum_ = 0.0;
  UpdateSummary_s last_summary_;
};

UpdateResult_s measureUpdate(
  const std::string & _backend, SchurUpdater & _updater, const EkfState & _state,
  const std::vector<FeatureTrack_s> & _tracks, const std::vector<CameraExtrinsics_s> & _cameras,
  int _iterations)
{
  UpdateResult_s result;
  result.backend_ = _backend;
  result.iterations_ = _iterations;
  result.calls_ = _iterations;
  double checksum = 0.0;
  double elapsed_ms = 0.0;
  UpdateSummary_s last_summary;
  for (int it = 0; it < _iterations; ++it) {
    // The reset copies are deliberately outside the timed region.
    EkfState work_state = _state;
    std::vector<FeatureTrack_s> work_tracks = _tracks;
    const auto t0 = std::chrono::steady_clock::now();
    last_summary = _updater.update(work_state, work_tracks, _cameras);
    const auto t1 = std::chrono::steady_clock::now();
    elapsed_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    const Eigen::MatrixXd & P = work_state.covariance();
    checksum += P.trace() + work_state.imu().p_.sum() + work_state.imu().v_.sum() +
                work_state.imu().ba_.sum() + work_state.imu().bg_.sum() +
                static_cast<double>(last_summary.features_used_) +
                static_cast<double>(last_summary.features_rejected_) +
                static_cast<double>(last_summary.observations_used_) +
                static_cast<double>(last_summary.rank_) + (last_summary.applied_ ? 1.0 : 0.0) +
                last_summary.mean_residual_px_ + last_summary.max_residual_px_;
  }
  result.elapsed_ms_ = elapsed_ms;
  result.us_per_update_ = _iterations > 0 ? 1000.0 * elapsed_ms / static_cast<double>(_iterations) : 0.0;
  result.checksum_ = checksum;
  result.last_summary_ = last_summary;
  return result;
}

struct UpdateValidation_s {
  int applied_mismatch_ = 0;
  int used_mismatch_ = 0;
  int rejected_mismatch_ = 0;
  int observations_mismatch_ = 0;
  int rank_mismatch_ = 0;
  int dof_mismatch_ = 0;
  int used_ids_mismatch_ = 0;
  int rejected_ids_mismatch_ = 0;
  int svd_used_ = 0;
  int svd_rejected_ = 0;
  int svd_observations_ = 0;
  int qr_used_ = 0;
  int qr_rejected_ = 0;
  int qr_observations_ = 0;
  bool svd_applied_ = false;
  bool qr_applied_ = false;
  double condition_rel_diff_ = 0.0;
  ScalarError_s mean_;
  MatrixError_s covariance_;
  bool finite_ = true;
};

UpdateValidation_s validateUpdates(
  SchurUpdater & _svd, SchurUpdater & _qr, const EkfState & _state,
  const std::vector<FeatureTrack_s> & _tracks, const std::vector<CameraExtrinsics_s> & _cameras)
{
  UpdateValidation_s result;
  EkfState ref_state = _state;
  EkfState test_state = _state;
  std::vector<FeatureTrack_s> ref_tracks = _tracks;
  std::vector<FeatureTrack_s> test_tracks = _tracks;
  const UpdateSummary_s ref = _svd.update(ref_state, ref_tracks, _cameras);
  const UpdateSummary_s test = _qr.update(test_state, test_tracks, _cameras);

  result.svd_used_ = ref.features_used_;
  result.svd_rejected_ = ref.features_rejected_;
  result.svd_observations_ = ref.observations_used_;
  result.qr_used_ = test.features_used_;
  result.qr_rejected_ = test.features_rejected_;
  result.qr_observations_ = test.observations_used_;
  result.svd_applied_ = ref.applied_;
  result.qr_applied_ = test.applied_;

  result.applied_mismatch_ = ref.applied_ != test.applied_ ? 1 : 0;
  result.used_mismatch_ = ref.features_used_ != test.features_used_ ? 1 : 0;
  result.rejected_mismatch_ = ref.features_rejected_ != test.features_rejected_ ? 1 : 0;
  result.observations_mismatch_ = ref.observations_used_ != test.observations_used_ ? 1 : 0;
  result.rank_mismatch_ = ref.rank_ != test.rank_ ? 1 : 0;
  result.dof_mismatch_ = ref.degrees_of_freedom_ != test.degrees_of_freedom_ ? 1 : 0;
  result.condition_rel_diff_ =
    std::abs(ref.condition_ - test.condition_) / std::max({std::abs(ref.condition_), std::abs(test.condition_), 1e-12});

  auto equal_ids = [](std::vector<int> _lhs, std::vector<int> _rhs)
  {
    std::sort(_lhs.begin(), _lhs.end());
    std::sort(_rhs.begin(), _rhs.end());
    return _lhs == _rhs;
  };
  result.used_ids_mismatch_ = equal_ids(ref.used_feature_ids_, test.used_feature_ids_) ? 0 : 1;
  result.rejected_ids_mismatch_ = equal_ids(ref.rejected_feature_ids_, test.rejected_feature_ids_) ? 0 : 1;

  result.mean_ = compareMeans(ref_state, test_state);
  result.covariance_ = compareCovariances(ref_state, test_state);
  result.finite_ = result.mean_.finite && result.covariance_.finite && ref_state.covariance().allFinite() &&
                   test_state.covariance().allFinite() && std::isfinite(ref.condition_) &&
                   std::isfinite(test.condition_) && std::isfinite(ref.mean_residual_px_) &&
                   std::isfinite(ref.max_residual_px_) && std::isfinite(test.mean_residual_px_) &&
                   std::isfinite(test.max_residual_px_);
  return result;
}

struct BudgetResult_s {
  int max_observations_ = 0;
  int iterations_ = 0;
  double elapsed_ms_ = 0.0;
  double us_per_update_ = 0.0;
  double us_per_processed_track_ = 0.0;
  double us_per_processed_obs_ = 0.0;
  double checksum_ = 0.0;
  int input_tracks_ = 0;
  int input_obs_ = 0;
  int processed_tracks_ = 0;
  int processed_obs_ = 0;
  int used_tracks_ = 0;
  int used_obs_ = 0;
  int rejected_ = 0;
  int deferred_tracks_ = 0;
  int expired_tracks_ = 0;
  int rank_ = 0;
  bool applied_ = false;
  bool budget_limited_ = false;
};

BudgetResult_s measureBudget(
  const EkfState & _state, const std::vector<FeatureTrack_s> & _tracks,
  const std::vector<CameraExtrinsics_s> & _cameras, const UpdateOptions_s & _base_options, int _budget,
  int _iterations, int _warmup)
{
  BudgetResult_s result;
  result.max_observations_ = _budget;
  result.iterations_ = _iterations;
  result.input_tracks_ = static_cast<int>(_tracks.size());
  for (const FeatureTrack_s & track : _tracks)
    result.input_obs_ += static_cast<int>(track.observations_.size());

  UpdateOptions_s options = _base_options;
  options.max_update_observations_ = _budget;
  options.nullspace_method_ = "svd";
  SchurUpdater updater(options);

  for (int w = 0; w < _warmup; ++w) {
    EkfState warm_state = _state;
    std::vector<FeatureTrack_s> warm_tracks = _tracks;
    (void)updater.update(warm_state, warm_tracks, _cameras);
  }

  double checksum = 0.0;
  double elapsed_ms = 0.0;
  UpdateSummary_s last_summary;
  for (int it = 0; it < _iterations; ++it) {
    EkfState work_state = _state;
    std::vector<FeatureTrack_s> work_tracks = _tracks;
    const auto t0 = std::chrono::steady_clock::now();
    last_summary = updater.update(work_state, work_tracks, _cameras);
    const auto t1 = std::chrono::steady_clock::now();
    elapsed_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    const Eigen::MatrixXd & P = work_state.covariance();
    checksum += P.trace() + static_cast<double>(last_summary.features_used_) +
                static_cast<double>(last_summary.observations_used_) +
                static_cast<double>(last_summary.features_rejected_) +
                static_cast<double>(last_summary.budget_selected_tracks_) +
                static_cast<double>(last_summary.budget_selected_observations_) +
                (last_summary.applied_ ? 1.0 : 0.0);
  }

  result.elapsed_ms_ = elapsed_ms;
  result.us_per_update_ = _iterations > 0 ? 1000.0 * elapsed_ms / static_cast<double>(_iterations) : 0.0;
  result.checksum_ = checksum;
  result.used_tracks_ = last_summary.features_used_;
  result.used_obs_ = last_summary.observations_used_;
  result.rejected_ = last_summary.features_rejected_;
  result.deferred_tracks_ = last_summary.budget_deferred_tracks_;
  result.expired_tracks_ = last_summary.budget_expired_tracks_;
  result.rank_ = last_summary.rank_;
  result.applied_ = last_summary.applied_;
  result.budget_limited_ = last_summary.budget_limited_;
  if (last_summary.budget_active_) {
    result.input_obs_ = last_summary.budget_input_observations_;
    result.processed_tracks_ = last_summary.budget_selected_tracks_;
    result.processed_obs_ = last_summary.budget_selected_observations_;
  } else {
    result.processed_tracks_ = static_cast<int>(last_summary.consumed_feature_ids_.size());
    result.processed_obs_ = result.input_obs_;
  }
  result.us_per_processed_track_ =
    (_iterations > 0 && result.processed_tracks_ > 0)
      ? 1000.0 * elapsed_ms / (static_cast<double>(_iterations) * static_cast<double>(result.processed_tracks_))
      : 0.0;
  result.us_per_processed_obs_ =
    (_iterations > 0 && result.processed_obs_ > 0)
      ? 1000.0 * elapsed_ms / (static_cast<double>(_iterations) * static_cast<double>(result.processed_obs_))
      : 0.0;
  return result;
}

struct ImuSample_s {
  Eigen::Vector3d acc_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyr_ = Eigen::Vector3d::Zero();
};

EkfState makeImuBaseState(int _clones, unsigned int _seed)
{
  EkfOptions_s options;
  options.window_size_ = _clones;
  options.batch_imu_covariance_ = false;

  std::vector<Eigen::Vector4d> qs;
  std::vector<Eigen::Vector3d> ps;
  for (int i = 0; i < _clones; ++i) {
    const double t = 0.1 * static_cast<double>(i);
    ps.push_back(Eigen::Vector3d(0.2 * t, 0.02 * t * t, 0.01 * t));
    qs.push_back(quatFromAxisAngle(Eigen::Vector3d(0.0, -0.012 * t, 0.008 * t)));
  }

  EkfState state;
  state.reset(
    options, 0.0, qs.front(), ps.front(), Eigen::Vector3d(0.2, 0.0, 0.0),
    Eigen::Vector3d(0.001, -0.001, 0.0005), Eigen::Vector3d(0.0001, -0.0001, 0.00005));
  state.augmentClone(0.0);
  for (int i = 1; i < _clones; ++i) {
    state.imu().q_ = qs[static_cast<std::size_t>(i)];
    state.imu().p_ = ps[static_cast<std::size_t>(i)];
    state.refreshImuFej();
    state.augmentClone(0.1 * static_cast<double>(i));
  }

  state.covariance().diagonal().array() += 1e-9;
  DeterministicNoise noise(_seed ^ 0x51u ^ static_cast<unsigned int>(_clones));
  for (int k = 0; k < 40; ++k) {
    const double t = K_DT * static_cast<double>(k);
    Eigen::Vector3d acc(
      0.03 * std::sin(0.3 * t), 0.02 * std::cos(0.2 * t), K_GRAVITY + 0.04 * std::sin(0.17 * t));
    Eigen::Vector3d gyr(0.003 * std::sin(0.4 * t), 0.002 * std::cos(0.35 * t), 0.001 * std::sin(0.2 * t));
    acc += 0.02 * Eigen::Vector3d(noise.normal(), noise.normal(), noise.normal());
    gyr += 0.001 * Eigen::Vector3d(noise.normal(), noise.normal(), noise.normal());
    state.propagate(acc, gyr, K_DT);
  }
  return state;
}

std::vector<ImuSample_s> makeImuSamples(int _count, unsigned int _seed)
{
  DeterministicNoise noise(_seed);
  std::vector<ImuSample_s> samples;
  samples.reserve(static_cast<std::size_t>(std::max(0, _count)));
  for (int k = 0; k < _count; ++k) {
    const double t = K_DT * static_cast<double>(k);
    ImuSample_s sample;
    sample.acc_ = Eigen::Vector3d(
      0.04 * std::sin(0.31 * t), 0.03 * std::cos(0.23 * t), K_GRAVITY + 0.05 * std::sin(0.19 * t));
    sample.gyr_ = Eigen::Vector3d(0.004 * std::sin(0.41 * t), 0.003 * std::cos(0.37 * t), 0.002 * std::sin(0.29 * t));
    sample.acc_ += 0.02 * Eigen::Vector3d(noise.normal(), noise.normal(), noise.normal());
    sample.gyr_ += 0.001 * Eigen::Vector3d(noise.normal(), noise.normal(), noise.normal());
    samples.push_back(sample);
  }
  return samples;
}

void runImuWarmup(const EkfState & _base, const std::vector<ImuSample_s> & _samples)
{
  EkfState sequential = _base;
  EkfState batch = _base;
  batch.setBatchImuCovarianceEnabled(true);
  for (std::size_t i = 0; i < _samples.size(); ++i) {
    sequential.propagate(_samples[i].acc_, _samples[i].gyr_, K_DT);
    batch.propagate(_samples[i].acc_, _samples[i].gyr_, K_DT);
    if ((i + 1) % static_cast<std::size_t>(K_FLUSH_SAMPLES) == 0)
      batch.flushImuCovariance();
  }
  batch.flushImuCovariance();
}

struct ImuResult_s {
  int clones_ = 0;
  int samples_ = 0;
  int groups_ = 0;
  int warmup_samples_ = 0;
  double seq_propagate_ms_ = 0.0;
  double seq_total_ms_ = 0.0;
  double batch_propagate_ms_ = 0.0;
  double batch_flush_ms_ = 0.0;
  double batch_total_ms_ = 0.0;
  double seq_us_per_sample_ = 0.0;
  double batch_us_per_sample_ = 0.0;
  double speedup_ = 0.0;
  ScalarError_s mean_;
  MatrixError_s covariance_;
  double checksum_ = 0.0;
  bool finite_ = true;
  bool pending_after_flush_ = false;
  bool seq_psd_ = false;
  bool batch_psd_ = false;
};

ImuResult_s measureImu(const EkfState & _base, const std::vector<ImuSample_s> & _samples, int _warmup_samples)
{
  ImuResult_s result;
  result.clones_ = _base.numClones();
  result.samples_ = static_cast<int>(_samples.size());
  result.groups_ = (result.samples_ + K_FLUSH_SAMPLES - 1) / K_FLUSH_SAMPLES;
  result.warmup_samples_ = _warmup_samples;

  EkfState sequential = _base;
  EkfState batch = _base;
  batch.setBatchImuCovarianceEnabled(true);

  double seq_propagate_ms = 0.0;
  double batch_propagate_ms = 0.0;
  double batch_flush_ms = 0.0;
  for (int start = 0; start < result.samples_; start += K_FLUSH_SAMPLES) {
    const int end = std::min(start + K_FLUSH_SAMPLES, result.samples_);

    const auto t0 = std::chrono::steady_clock::now();
    for (int k = start; k < end; ++k)
      sequential.propagate(_samples[static_cast<std::size_t>(k)].acc_, _samples[static_cast<std::size_t>(k)].gyr_, K_DT);
    const auto t1 = std::chrono::steady_clock::now();
    seq_propagate_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    const auto t2 = std::chrono::steady_clock::now();
    for (int k = start; k < end; ++k)
      batch.propagate(_samples[static_cast<std::size_t>(k)].acc_, _samples[static_cast<std::size_t>(k)].gyr_, K_DT);
    const auto t3 = std::chrono::steady_clock::now();
    batch_propagate_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();

    const auto t4 = std::chrono::steady_clock::now();
    batch.flushImuCovariance();
    const auto t5 = std::chrono::steady_clock::now();
    batch_flush_ms += std::chrono::duration<double, std::milli>(t5 - t4).count();
  }

  result.seq_propagate_ms_ = seq_propagate_ms;
  result.seq_total_ms_ = seq_propagate_ms;
  result.batch_propagate_ms_ = batch_propagate_ms;
  result.batch_flush_ms_ = batch_flush_ms;
  result.batch_total_ms_ = batch_propagate_ms + batch_flush_ms;
  result.seq_us_per_sample_ =
    result.samples_ > 0 ? 1000.0 * result.seq_total_ms_ / static_cast<double>(result.samples_) : 0.0;
  result.batch_us_per_sample_ =
    result.samples_ > 0 ? 1000.0 * result.batch_total_ms_ / static_cast<double>(result.samples_) : 0.0;
  result.speedup_ = result.batch_total_ms_ > 0.0 ? result.seq_total_ms_ / result.batch_total_ms_ : 0.0;
  result.mean_ = compareMeans(sequential, batch);
  result.covariance_ = compareCovariances(sequential, batch);
  result.pending_after_flush_ = batch.hasPendingImuCovariance();
  result.seq_psd_ = EkfState::isPositiveSemidefinite(sequential.covariance(), 1e-8);
  result.batch_psd_ = EkfState::isPositiveSemidefinite(batch.covariance(), 1e-8);
  result.checksum_ = sequential.covariance().trace() + batch.covariance().trace() +
                     sequential.imu().p_.sum() + batch.imu().p_.sum() + sequential.imu().v_.sum() +
                     batch.imu().v_.sum() + sequential.imu().ba_.sum() + batch.imu().ba_.sum();
  result.finite_ = result.mean_.finite && result.covariance_.finite && sequential.covariance().allFinite() &&
                   batch.covariance().allFinite() && std::isfinite(result.checksum_);
  return result;
}

enum class Mode_e { K_ALL, K_QR, K_IMU };

struct Config_s {
  Mode_e mode_ = Mode_e::K_ALL;
  int iterations_ = K_DEFAULT_ITERATIONS;
  int warmup_ = K_DEFAULT_WARMUP;
  unsigned int seed_ = K_SEED;
  std::string format_ = "both";
  bool help_ = false;
};

const char * modeName(Mode_e _mode)
{
  switch (_mode) {
    case Mode_e::K_QR:
      return "qr";
    case Mode_e::K_IMU:
      return "imu";
    case Mode_e::K_ALL:
    default:
      return "all";
  }
}

void printUsage()
{
  std::printf(
    "Usage: benchmark_backend_ablation [options]\n"
    "\n"
    "  --mode all|qr|imu   all=visual+IMU; qr=QR/SVD reduce+update and budget sweep;\n"
    "                      imu=sequential/batch propagate only (default: all)\n"
    "  --iterations N      measured repetitions: QR update calls (each with all tracks),\n"
    "                      IMU propagated samples, budget update calls (default: %d)\n"
    "  --warmup N          QR/budget warmup update passes; IMU warmup groups of %d samples\n"
    "                      (default: %d)\n"
    "  --format both|json|kv  both prints key=value then JSON_BEGIN/JSON_END\n"
    "  --seed N            deterministic scene/noise seed (default: %u)\n"
    "  --help\n"
    "\n"
    "Timing regions contain only the production calls under test.  Scene setup,\n"
    "state/track reset copies, warmup and validation are outside the timed regions.\n",
    K_DEFAULT_ITERATIONS, K_FLUSH_SAMPLES, K_DEFAULT_WARMUP, K_SEED);
}

bool parseInt(const char * _text, int & _value)
{
  char * end = nullptr;
  const long value = std::strtol(_text, &end, 10);
  if (end == _text || *end != '\0' || value < 0 || value > std::numeric_limits<int>::max())
    return false;
  _value = static_cast<int>(value);
  return true;
}

bool parseUint(const char * _text, unsigned int & _value)
{
  char * end = nullptr;
  const unsigned long value = std::strtoul(_text, &end, 10);
  if (end == _text || *end != '\0' || value > std::numeric_limits<unsigned int>::max())
    return false;
  _value = static_cast<unsigned int>(value);
  return true;
}

bool parseConfig(int _argc, char ** _argv, Config_s & _config)
{
  for (int i = 1; i < _argc; ++i) {
    const std::string arg = _argv[i];
    if (arg == "--help" || arg == "-h") {
      _config.help_ = true;
      return true;
    }
    if (arg == "--mode") {
      if (++i >= _argc)
        return false;
      const std::string mode = _argv[i];
      if (mode == "all")
        _config.mode_ = Mode_e::K_ALL;
      else if (mode == "qr")
        _config.mode_ = Mode_e::K_QR;
      else if (mode == "imu")
        _config.mode_ = Mode_e::K_IMU;
      else
        return false;
      continue;
    }
    if (arg == "--iterations") {
      if (++i >= _argc || !parseInt(_argv[i], _config.iterations_) || _config.iterations_ <= 0)
        return false;
      continue;
    }
    if (arg == "--warmup") {
      if (++i >= _argc || !parseInt(_argv[i], _config.warmup_))
        return false;
      continue;
    }
    if (arg == "--seed") {
      if (++i >= _argc || !parseUint(_argv[i], _config.seed_))
        return false;
      continue;
    }
    if (arg == "--format") {
      if (++i >= _argc)
        return false;
      _config.format_ = _argv[i];
      if (_config.format_ != "both" && _config.format_ != "json" && _config.format_ != "kv")
        return false;
      continue;
    }
    return false;
  }
  return true;
}

std::string jsonNumber(double _value)
{
  if (!std::isfinite(_value))
    return "null";
  std::ostringstream stream;
  stream << std::setprecision(17) << _value;
  return stream.str();
}

std::string jsonBool(bool _value) { return _value ? "true" : "false"; }

void printSceneKv(const Scene_s & _scene)
{
  std::printf(
    "scene clones=%d intended_tracks=%d triangulated_tracks=%d cameras=%d intended_observations=%d "
    "actual_observations=%d triangulation_failures=%d pixel_noise_px=%.17g "
    "initial_mean_max_abs=%.17g initial_mean_rms=%.17g\n",
    K_CLONES, _scene.intended_tracks_, static_cast<int>(_scene.tracks_.size()), K_CAMERAS,
    _scene.intended_observations_, _scene.actual_observations_, _scene.triangulation_failures_, K_PIXEL_NOISE_PX,
    _scene.initial_mean_perturbation_.max_abs, _scene.initial_mean_perturbation_.rms);
}

void printReduceKv(const ReduceResult_s & _result)
{
  std::printf(
    "reduce backend=%s iterations=%d tracks=%d calls=%lld elapsed_ms=%.17g us_per_call=%.17g checksum=%.17g\n",
    _result.backend_.c_str(), _result.iterations_, _result.tracks_, _result.calls_, _result.elapsed_ms_,
    _result.us_per_call_, _result.checksum_);
}

void printReduceValidationKv(const ReduceValidation_s & _result)
{
  std::printf(
    "compare_reduce ref=svd test=qr svd_valid=%d qr_valid=%d invalid_mismatch=%d gate_dof_mismatch=%d "
    "point_rank_mismatch=%d w_key_mismatch=%d finite_violations=%d chi2_finite_svd=%d "
    "chi2_nonfinite_svd=%d chi2_finite_qr=%d chi2_nonfinite_qr=%d max_a_point_rel_fro=%.17g "
    "max_b_point_rel=%.17g max_chi2_abs=%.17g max_rms_abs=%.17g max_w_rel_fro=%.17g finite=%d\n",
    _result.svd_valid_, _result.qr_valid_, _result.invalid_mismatch_, _result.gate_dof_mismatch_,
    _result.point_rank_mismatch_, _result.w_key_mismatch_, _result.finite_violations_, _result.chi2_finite_svd_,
    _result.chi2_nonfinite_svd_, _result.chi2_finite_qr_, _result.chi2_nonfinite_qr_,
    _result.max_a_point_rel_fro_, _result.max_b_point_rel_, _result.max_chi2_abs_, _result.max_rms_abs_,
    _result.max_w_rel_fro_, _result.finite_ ? 1 : 0);
}

void printUpdateKv(const UpdateResult_s & _result)
{
  const UpdateSummary_s & summary = _result.last_summary_;
  std::printf(
    "update backend=%s iterations=%d calls=%lld elapsed_ms=%.17g us_per_update=%.17g checksum=%.17g "
    "applied=%d features_used=%d features_rejected=%d observations_used=%d rank=%d condition=%.17g "
    "mean_residual_px=%.17g max_residual_px=%.17g\n",
    _result.backend_.c_str(), _result.iterations_, _result.calls_, _result.elapsed_ms_, _result.us_per_update_,
    _result.checksum_, summary.applied_ ? 1 : 0, summary.features_used_, summary.features_rejected_,
    summary.observations_used_, summary.rank_, summary.condition_, summary.mean_residual_px_,
    summary.max_residual_px_);
}

void printUpdateValidationKv(const UpdateValidation_s & _result)
{
  std::printf(
    "compare_update ref=svd test=qr applied_mismatch=%d used_mismatch=%d rejected_mismatch=%d "
    "observations_mismatch=%d rank_mismatch=%d dof_mismatch=%d used_ids_mismatch=%d "
    "rejected_ids_mismatch=%d condition_rel_diff=%.17g svd_used=%d svd_rejected=%d svd_observations=%d "
    "qr_used=%d qr_rejected=%d qr_observations=%d svd_applied=%d qr_applied=%d mean_max_abs=%.17g "
    "mean_rms=%.17g cov_rel_fro=%.17g cov_max_abs=%.17g finite=%d\n",
    _result.applied_mismatch_, _result.used_mismatch_, _result.rejected_mismatch_, _result.observations_mismatch_,
    _result.rank_mismatch_, _result.dof_mismatch_, _result.used_ids_mismatch_, _result.rejected_ids_mismatch_,
    _result.condition_rel_diff_, _result.svd_used_, _result.svd_rejected_, _result.svd_observations_,
    _result.qr_used_, _result.qr_rejected_, _result.qr_observations_, _result.svd_applied_ ? 1 : 0,
    _result.qr_applied_ ? 1 : 0, _result.mean_.max_abs, _result.mean_.rms, _result.covariance_.rel_fro,
    _result.covariance_.max_abs, _result.finite_ ? 1 : 0);
}

void printImuKv(const ImuResult_s & _result)
{
  std::printf(
    "imu clones=%d samples=%d groups=%d warmup_samples=%d seq_propagate_ms=%.17g seq_total_ms=%.17g "
    "batch_propagate_ms=%.17g batch_flush_ms=%.17g batch_total_ms=%.17g seq_us_per_sample=%.17g "
    "batch_us_per_sample=%.17g speedup=%.17g mean_max_abs=%.17g mean_rms=%.17g cov_rel_fro=%.17g "
    "cov_max_abs=%.17g checksum=%.17g finite=%d pending_after_flush=%d seq_psd=%d batch_psd=%d\n",
    _result.clones_, _result.samples_, _result.groups_, _result.warmup_samples_, _result.seq_propagate_ms_,
    _result.seq_total_ms_, _result.batch_propagate_ms_, _result.batch_flush_ms_, _result.batch_total_ms_,
    _result.seq_us_per_sample_, _result.batch_us_per_sample_, _result.speedup_, _result.mean_.max_abs,
    _result.mean_.rms, _result.covariance_.rel_fro, _result.covariance_.max_abs, _result.checksum_,
    _result.finite_ ? 1 : 0, _result.pending_after_flush_ ? 1 : 0, _result.seq_psd_ ? 1 : 0,
    _result.batch_psd_ ? 1 : 0);
}

void printBudgetKv(const BudgetResult_s & _result)
{
  std::printf(
    "budget max_observations=%d iterations=%d elapsed_ms=%.17g us_per_update=%.17g "
    "us_per_processed_track=%.17g us_per_processed_obs=%.17g checksum=%.17g input_tracks=%d input_obs=%d "
    "processed_tracks=%d processed_obs=%d used_tracks=%d used_obs=%d rejected=%d deferred_tracks=%d "
    "expired_tracks=%d applied=%d budget_limited=%d rank=%d\n",
    _result.max_observations_, _result.iterations_, _result.elapsed_ms_, _result.us_per_update_,
    _result.us_per_processed_track_, _result.us_per_processed_obs_, _result.checksum_, _result.input_tracks_,
    _result.input_obs_, _result.processed_tracks_, _result.processed_obs_, _result.used_tracks_, _result.used_obs_,
    _result.rejected_, _result.deferred_tracks_, _result.expired_tracks_, _result.applied_ ? 1 : 0,
    _result.budget_limited_ ? 1 : 0, _result.rank_);
}

void printJson(
  const Config_s & _config, const Scene_s * _scene, const std::vector<ReduceResult_s> & _reduces,
  const ReduceValidation_s & _reduce_validation, const std::vector<UpdateResult_s> & _updates,
  const UpdateValidation_s & _update_validation, const std::vector<ImuResult_s> & _imu,
  const std::vector<BudgetResult_s> & _budgets)
{
  std::ostringstream out;
  out << "{\"schema\":\"backend_ablation_v1\",\"meta\":{";
  out << "\"seed\":" << _config.seed_ << ",\"mode\":\"" << modeName(_config.mode_) << "\"";
  out << ",\"iterations\":" << _config.iterations_ << ",\"warmup\":" << _config.warmup_;
  out << ",\"timing_scope\":\"production_calls_only\"";
  out << ",\"timing_excludes\":\"scene_build,state_copy,track_copy,warmup,validation\"";
  out << "},\"scene\":";
  if (_scene == nullptr) {
    out << "null";
  } else {
    out << "{\"clones\":" << K_CLONES << ",\"intended_tracks\":" << _scene->intended_tracks_
        << ",\"triangulated_tracks\":" << _scene->tracks_.size() << ",\"cameras\":" << K_CAMERAS
        << ",\"intended_observations\":" << _scene->intended_observations_
        << ",\"actual_observations\":" << _scene->actual_observations_
        << ",\"triangulation_failures\":" << _scene->triangulation_failures_
        << ",\"pixel_noise_px\":" << jsonNumber(K_PIXEL_NOISE_PX)
        << ",\"initial_mean_max_abs\":" << jsonNumber(_scene->initial_mean_perturbation_.max_abs)
        << ",\"initial_mean_rms\":" << jsonNumber(_scene->initial_mean_perturbation_.rms) << "}";
  }

  out << ",\"qr_svd\":";
  if (_reduces.empty() && _updates.empty()) {
    out << "null";
  } else {
    out << "{\"reduce\":[";
    for (std::size_t i = 0; i < _reduces.size(); ++i) {
      if (i != 0)
        out << ',';
      const ReduceResult_s & item = _reduces[i];
      out << "{\"backend\":\"" << item.backend_ << "\",\"iterations\":" << item.iterations_
          << ",\"tracks\":" << item.tracks_ << ",\"calls\":" << item.calls_
          << ",\"elapsed_ms\":" << jsonNumber(item.elapsed_ms_)
          << ",\"us_per_call\":" << jsonNumber(item.us_per_call_)
          << ",\"checksum\":" << jsonNumber(item.checksum_) << "}";
    }
    out << "],\"reduce_validation\":{";
    out << "\"svd_valid\":" << _reduce_validation.svd_valid_ << ",\"qr_valid\":"
        << _reduce_validation.qr_valid_ << ",\"invalid_mismatch\":" << _reduce_validation.invalid_mismatch_
        << ",\"gate_dof_mismatch\":" << _reduce_validation.gate_dof_mismatch_
        << ",\"point_rank_mismatch\":" << _reduce_validation.point_rank_mismatch_
        << ",\"w_key_mismatch\":" << _reduce_validation.w_key_mismatch_
        << ",\"finite_violations\":" << _reduce_validation.finite_violations_
        << ",\"chi2_finite_svd\":" << _reduce_validation.chi2_finite_svd_
        << ",\"chi2_nonfinite_svd\":" << _reduce_validation.chi2_nonfinite_svd_
        << ",\"chi2_finite_qr\":" << _reduce_validation.chi2_finite_qr_
        << ",\"chi2_nonfinite_qr\":" << _reduce_validation.chi2_nonfinite_qr_
        << ",\"max_a_point_rel_fro\":" << jsonNumber(_reduce_validation.max_a_point_rel_fro_)
        << ",\"max_b_point_rel\":" << jsonNumber(_reduce_validation.max_b_point_rel_)
        << ",\"max_chi2_abs\":" << jsonNumber(_reduce_validation.max_chi2_abs_)
        << ",\"max_rms_abs\":" << jsonNumber(_reduce_validation.max_rms_abs_)
        << ",\"max_w_rel_fro\":" << jsonNumber(_reduce_validation.max_w_rel_fro_)
        << ",\"finite\":" << jsonBool(_reduce_validation.finite_) << "},\"update\":[";
    for (std::size_t i = 0; i < _updates.size(); ++i) {
      if (i != 0)
        out << ',';
      const UpdateResult_s & item = _updates[i];
      const UpdateSummary_s & summary = item.last_summary_;
      out << "{\"backend\":\"" << item.backend_ << "\",\"iterations\":" << item.iterations_
          << ",\"calls\":" << item.calls_ << ",\"elapsed_ms\":" << jsonNumber(item.elapsed_ms_)
          << ",\"us_per_update\":" << jsonNumber(item.us_per_update_)
          << ",\"checksum\":" << jsonNumber(item.checksum_)
          << ",\"applied\":" << jsonBool(summary.applied_) << ",\"features_used\":" << summary.features_used_
          << ",\"features_rejected\":" << summary.features_rejected_
          << ",\"observations_used\":" << summary.observations_used_ << ",\"rank\":" << summary.rank_
          << ",\"condition\":" << jsonNumber(summary.condition_)
          << ",\"mean_residual_px\":" << jsonNumber(summary.mean_residual_px_)
          << ",\"max_residual_px\":" << jsonNumber(summary.max_residual_px_) << "}";
    }
    out << "],\"update_validation\":{";
    out << "\"applied_mismatch\":" << _update_validation.applied_mismatch_
        << ",\"used_mismatch\":" << _update_validation.used_mismatch_
        << ",\"rejected_mismatch\":" << _update_validation.rejected_mismatch_
        << ",\"observations_mismatch\":" << _update_validation.observations_mismatch_
        << ",\"rank_mismatch\":" << _update_validation.rank_mismatch_
        << ",\"dof_mismatch\":" << _update_validation.dof_mismatch_
        << ",\"used_ids_mismatch\":" << _update_validation.used_ids_mismatch_
        << ",\"rejected_ids_mismatch\":" << _update_validation.rejected_ids_mismatch_
        << ",\"condition_rel_diff\":" << jsonNumber(_update_validation.condition_rel_diff_)
        << ",\"svd_used\":" << _update_validation.svd_used_
        << ",\"svd_rejected\":" << _update_validation.svd_rejected_
        << ",\"svd_observations\":" << _update_validation.svd_observations_
        << ",\"qr_used\":" << _update_validation.qr_used_
        << ",\"qr_rejected\":" << _update_validation.qr_rejected_
        << ",\"qr_observations\":" << _update_validation.qr_observations_
        << ",\"svd_applied\":" << jsonBool(_update_validation.svd_applied_)
        << ",\"qr_applied\":" << jsonBool(_update_validation.qr_applied_)
        << ",\"mean_max_abs\":" << jsonNumber(_update_validation.mean_.max_abs)
        << ",\"mean_rms\":" << jsonNumber(_update_validation.mean_.rms)
        << ",\"cov_rel_fro\":" << jsonNumber(_update_validation.covariance_.rel_fro)
        << ",\"cov_max_abs\":" << jsonNumber(_update_validation.covariance_.max_abs)
        << ",\"finite\":" << jsonBool(_update_validation.finite_) << "},\"budget\":[";
    for (std::size_t i = 0; i < _budgets.size(); ++i) {
      if (i != 0)
        out << ',';
      const BudgetResult_s & item = _budgets[i];
      out << "{\"max_observations\":" << item.max_observations_ << ",\"iterations\":" << item.iterations_
          << ",\"elapsed_ms\":" << jsonNumber(item.elapsed_ms_)
          << ",\"us_per_update\":" << jsonNumber(item.us_per_update_)
          << ",\"us_per_processed_track\":" << jsonNumber(item.us_per_processed_track_)
          << ",\"us_per_processed_obs\":" << jsonNumber(item.us_per_processed_obs_)
          << ",\"checksum\":" << jsonNumber(item.checksum_)
          << ",\"input_tracks\":" << item.input_tracks_ << ",\"input_obs\":" << item.input_obs_
          << ",\"processed_tracks\":" << item.processed_tracks_ << ",\"processed_obs\":" << item.processed_obs_
          << ",\"used_tracks\":" << item.used_tracks_ << ",\"used_obs\":" << item.used_obs_
          << ",\"rejected\":" << item.rejected_ << ",\"deferred_tracks\":" << item.deferred_tracks_
          << ",\"expired_tracks\":" << item.expired_tracks_ << ",\"applied\":" << jsonBool(item.applied_)
          << ",\"budget_limited\":" << jsonBool(item.budget_limited_) << ",\"rank\":" << item.rank_ << "}";
    }
    out << "]}";
  }

  out << ",\"imu\":[";
  for (std::size_t i = 0; i < _imu.size(); ++i) {
    if (i != 0)
      out << ',';
    const ImuResult_s & item = _imu[i];
    out << "{\"clones\":" << item.clones_ << ",\"samples\":" << item.samples_ << ",\"groups\":" << item.groups_
        << ",\"warmup_samples\":" << item.warmup_samples_
        << ",\"seq_propagate_ms\":" << jsonNumber(item.seq_propagate_ms_)
        << ",\"seq_total_ms\":" << jsonNumber(item.seq_total_ms_)
        << ",\"batch_propagate_ms\":" << jsonNumber(item.batch_propagate_ms_)
        << ",\"batch_flush_ms\":" << jsonNumber(item.batch_flush_ms_)
        << ",\"batch_total_ms\":" << jsonNumber(item.batch_total_ms_)
        << ",\"seq_us_per_sample\":" << jsonNumber(item.seq_us_per_sample_)
        << ",\"batch_us_per_sample\":" << jsonNumber(item.batch_us_per_sample_)
        << ",\"speedup\":" << jsonNumber(item.speedup_)
        << ",\"mean_max_abs\":" << jsonNumber(item.mean_.max_abs)
        << ",\"mean_rms\":" << jsonNumber(item.mean_.rms)
        << ",\"cov_rel_fro\":" << jsonNumber(item.covariance_.rel_fro)
        << ",\"cov_max_abs\":" << jsonNumber(item.covariance_.max_abs)
        << ",\"checksum\":" << jsonNumber(item.checksum_) << ",\"finite\":" << jsonBool(item.finite_)
        << ",\"pending_after_flush\":" << jsonBool(item.pending_after_flush_)
        << ",\"seq_psd\":" << jsonBool(item.seq_psd_) << ",\"batch_psd\":" << jsonBool(item.batch_psd_) << "}";
  }
  out << "]}";
  std::printf("JSON_BEGIN\n%s\nJSON_END\n", out.str().c_str());
}

}  // namespace

int main(int argc, char ** argv)
{
  Config_s config;
  if (!parseConfig(argc, argv, config)) {
    std::fprintf(stderr, "error: invalid command line\n");
    printUsage();
    return 2;
  }
  if (config.help_) {
    printUsage();
    return 0;
  }

  const bool run_qr = config.mode_ == Mode_e::K_ALL || config.mode_ == Mode_e::K_QR;
  const bool run_imu = config.mode_ == Mode_e::K_ALL || config.mode_ == Mode_e::K_IMU;

  Scene_s scene;
  bool has_scene = false;
  std::vector<ReduceResult_s> reduces;
  ReduceValidation_s reduce_validation;
  std::vector<UpdateResult_s> updates;
  UpdateValidation_s update_validation;
  std::vector<BudgetResult_s> budgets;

  if (run_qr) {
    scene = makeScene(config.seed_);
    has_scene = true;
    if (scene.tracks_.empty()) {
      std::fprintf(stderr, "error: no triangulated synthetic stereo tracks\n");
      return 1;
    }

    UpdateOptions_s base_options;
    base_options.max_features_ = K_TRACKS;
    base_options.min_observations_ = 2;
    base_options.min_features_ = 12;
    base_options.min_update_observations_ = 24;
    base_options.condition_limit_ = 0.0;
    base_options.pixel_noise_ = 1.5;
    base_options.focal_length_ = K_FOCAL_LENGTH;
    base_options.chi2_multiplier_ = 5.0;
    base_options.info_rank_rtol_ = 1e-10;
    base_options.point_block_rtol_ = 1e-8;
    base_options.max_update_observations_ = 0;

    UpdateOptions_s svd_options = base_options;
    svd_options.nullspace_method_ = "svd";
    UpdateOptions_s qr_options = base_options;
    qr_options.nullspace_method_ = "qr";
    SchurUpdater svd_updater(svd_options);
    SchurUpdater qr_updater(qr_options);

    // Warm the thread-local reduction/update scratch.  This pass is outside all
    // timed regions.
    (void)measureReduce("warmup", svd_updater, scene.state_, scene.cameras_, scene.tracks_, 1);
    (void)measureReduce("warmup", qr_updater, scene.state_, scene.cameras_, scene.tracks_, 1);
    for (int w = 0; w < config.warmup_; ++w) {
      EkfState svd_state = scene.state_;
      EkfState qr_state = scene.state_;
      std::vector<FeatureTrack_s> svd_tracks = scene.tracks_;
      std::vector<FeatureTrack_s> qr_tracks = scene.tracks_;
      (void)svd_updater.update(svd_state, svd_tracks, scene.cameras_);
      (void)qr_updater.update(qr_state, qr_tracks, scene.cameras_);
    }

    reduces.push_back(measureReduce("svd", svd_updater, scene.state_, scene.cameras_, scene.tracks_, config.iterations_));
    reduces.push_back(measureReduce("qr", qr_updater, scene.state_, scene.cameras_, scene.tracks_, config.iterations_));
    reduce_validation = validateReductions(svd_updater, qr_updater, scene.state_, scene.cameras_, scene.tracks_);

    updates.push_back(
      measureUpdate("svd", svd_updater, scene.state_, scene.tracks_, scene.cameras_, config.iterations_));
    updates.push_back(
      measureUpdate("qr", qr_updater, scene.state_, scene.tracks_, scene.cameras_, config.iterations_));
    update_validation = validateUpdates(svd_updater, qr_updater, scene.state_, scene.tracks_, scene.cameras_);

    for (const int budget : K_BUDGETS) {
      budgets.push_back(
        measureBudget(scene.state_, scene.tracks_, scene.cameras_, base_options, budget, config.iterations_, config.warmup_));
    }
  }

  std::vector<ImuResult_s> imu;
  if (run_imu) {
    const int warmup_samples = config.warmup_ * K_FLUSH_SAMPLES;
    for (const int clones : {5, 10}) {
      const EkfState base = makeImuBaseState(clones, config.seed_);
      if (warmup_samples > 0) {
        const std::vector<ImuSample_s> warmup = makeImuSamples(warmup_samples, config.seed_ ^ 0xabcdu ^ clones);
        runImuWarmup(base, warmup);
      }
      const std::vector<ImuSample_s> samples =
        makeImuSamples(config.iterations_, config.seed_ ^ 0x1357u ^ static_cast<unsigned int>(clones));
      imu.push_back(measureImu(base, samples, warmup_samples));
    }
  }

  if (config.format_ == "kv" || config.format_ == "both") {
    std::printf(
      "meta schema=backend_ablation_v1 seed=%u mode=%s iterations=%d warmup=%d "
      "timing_scope=production_calls_only timing_excludes=scene_build,state_copy,track_copy,warmup,validation\n",
      config.seed_, modeName(config.mode_), config.iterations_, config.warmup_);
    if (has_scene)
      printSceneKv(scene);
    for (const ReduceResult_s & result : reduces)
      printReduceKv(result);
    if (!reduces.empty())
      printReduceValidationKv(reduce_validation);
    for (const UpdateResult_s & result : updates)
      printUpdateKv(result);
    if (!updates.empty())
      printUpdateValidationKv(update_validation);
    for (const ImuResult_s & result : imu)
      printImuKv(result);
    for (const BudgetResult_s & result : budgets)
      printBudgetKv(result);
  }
  if (config.format_ == "json" || config.format_ == "both")
    printJson(
      config, has_scene ? &scene : nullptr, reduces, reduce_validation, updates, update_validation, imu, budgets);
  return 0;
}
