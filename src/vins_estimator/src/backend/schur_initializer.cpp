#include "schur_initializer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <unordered_map>

namespace vins
{
namespace backend
{
namespace
{

constexpr double K_PI = 3.14159265358979323846;
constexpr double K_EPS = 1e-12;

Eigen::Vector4d quaternionOf(const Eigen::Matrix3d & _R)
{
  const Eigen::Quaterniond q(_R);
  const Eigen::Quaterniond normalized = q.normalized();
  return Eigen::Vector4d(normalized.w(), normalized.x(), normalized.y(), normalized.z());
}

Eigen::Vector3d axisAngleOf(const Eigen::Quaterniond & _q)
{
  const Eigen::Quaterniond q = _q.normalized();
  const double w = std::max(-1.0, std::min(1.0, std::abs(q.w())));
  const double s = std::sqrt(std::max(0.0, 1.0 - w * w));
  if (s < K_EPS)
    return Eigen::Vector3d::Zero();
  const double angle = 2.0 * std::atan2(s, w);
  Eigen::Vector3d axis(q.x(), q.y(), q.z());
  if (q.w() < 0.0)
    axis = -axis;
  return axis.normalized() * angle;
}

double rotationAngle(const Eigen::Matrix3d & _R) { return axisAngleOf(Eigen::Quaterniond(_R)).norm(); }

double median(std::vector<double> _values)
{
  if (_values.empty())
    return std::numeric_limits<double>::quiet_NaN();
  const std::size_t mid = _values.size() / 2;
  std::nth_element(_values.begin(), _values.begin() + static_cast<std::ptrdiff_t>(mid), _values.end());
  double value = _values[mid];
  if (_values.size() % 2 == 0) {
    std::nth_element(
      _values.begin(), _values.begin() + static_cast<std::ptrdiff_t>(mid - 1),
      _values.begin() + static_cast<std::ptrdiff_t>(mid));
    value = 0.5 * (value + _values[mid - 1]);
  }
  return value;
}

Eigen::Matrix<double, 3, 2> tangentBasis(const Eigen::Vector3d & _g)
{
  Eigen::Matrix<double, 3, 2> basis;
  const Eigen::Vector3d a = _g.normalized();
  Eigen::Vector3d tmp(0.0, 0.0, 1.0);
  if (std::abs(a.dot(tmp)) > 0.9)
    tmp = Eigen::Vector3d(1.0, 0.0, 0.0);
  const Eigen::Vector3d b = (tmp - a * a.dot(tmp)).normalized();
  basis.col(0) = b;
  basis.col(1) = a.cross(b);
  return basis;
}


// Three-point Lagrange derivative.  Exact for a quadratic position segment,
// which is what a constant-acceleration window is.  Unlike a one-sided
// difference this keeps the endpoint velocity unbiased enough for the gravity
// regression.
Eigen::Vector3d quadraticDerivativeAt(
  double _t, double _ta, double _tb, double _tc, const Eigen::Vector3d & _pa, const Eigen::Vector3d & _pb,
  const Eigen::Vector3d & _pc)
{
  const double denom_a = (_ta - _tb) * (_ta - _tc);
  const double denom_b = (_tb - _ta) * (_tb - _tc);
  const double denom_c = (_tc - _ta) * (_tc - _tb);
  const double ca = (2.0 * _t - _tb - _tc) / denom_a;
  const double cb = (2.0 * _t - _ta - _tc) / denom_b;
  const double cc = (2.0 * _t - _ta - _tb) / denom_c;
  return ca * _pa + cb * _pb + cc * _pc;
}

bool finiteMotionFrame(const SchurInitMotionFrame_s & _f)
{
  return std::isfinite(_f.t0_) && std::isfinite(_f.t1_) && _f.t1_ > _f.t0_ && _f.R0_.allFinite() &&
         _f.R1_.allFinite() && _f.p0_.allFinite() && _f.p1_.allFinite() && _f.delta_p_.allFinite() &&
         _f.delta_v_.allFinite() && _f.dq_dbg_.allFinite() && _f.p_ci_.allFinite() &&
         std::isfinite(_f.delta_q_.w()) && std::isfinite(_f.delta_q_.x()) && std::isfinite(_f.delta_q_.y()) &&
         std::isfinite(_f.delta_q_.z());
}

}  // namespace

SchurInitializer::SchurInitializer(const SchurInitConfig_s & _config) : config_(_config) {}

void SchurInitializer::configure(const SchurInitConfig_s & _config)
{
  config_ = _config;
  reset();
}

void SchurInitializer::reset()
{
  imu_.clear();
  visual_.clear();
  last_imu_time_ = -1.0;
  last_visual_time_ = -1.0;
  imu_gap_invalidated_ = false;
  last_reason_.clear();
}

void SchurInitializer::addImuSample(double _timestamp, const Eigen::Vector3d & _acc, const Eigen::Vector3d & _gyr)
{
  if (!std::isfinite(_timestamp) || !_acc.allFinite() || !_gyr.allFinite())
    return;
  if (last_imu_time_ >= 0.0 && _timestamp <= last_imu_time_)
    return;  // strict boundary: duplicates/backwards samples never enter a window

  if (
    last_imu_time_ >= 0.0 && config_.static_max_imu_gap_seconds_ > 0.0 &&
    _timestamp - last_imu_time_ > config_.static_max_imu_gap_seconds_) {
    imu_gap_invalidated_ = true;
  }
  last_imu_time_ = _timestamp;
  imu_.push_back(SchurInitImuSample_s{_timestamp, _acc, _gyr});
  if (imu_.size() > 20000)
    imu_.pop_front();
}

void SchurInitializer::addVisualFrame(
  double _timestamp, const std::vector<SchurInitVisualObservation_s> & _observations)
{
  if (!std::isfinite(_timestamp))
    return;
  if (last_visual_time_ >= 0.0 && _timestamp <= last_visual_time_) {
    // A backwards clock invalidates the visual motion gate. Never reuse a
    // mixed timeline that could make a moving platform look static.
    visual_.clear();
    last_reason_ = "visual clock moved backwards; static/motion window reset";
  }
  last_visual_time_ = _timestamp;
  visual_.push_back(SchurInitVisualFrame_s{_timestamp, _observations});
  if (visual_.size() > 200)
    visual_.pop_front();
}

void SchurInitializer::prune(double _now)
{
  if (!std::isfinite(_now))
    return;
  const double keep = std::max(1.0, 2.0 * config_.static_window_seconds_);
  while (!imu_.empty() && imu_.front().timestamp_ < _now - keep)
    imu_.pop_front();
  while (!visual_.empty() && visual_.front().timestamp_ < _now - keep)
    visual_.pop_front();
}

SchurInitializer::StaticImuStats_s SchurInitializer::staticImuStats(double _now) const
{
  StaticImuStats_s stats;
  if (!std::isfinite(_now) || config_.static_window_seconds_ <= 0.0) {
    stats.reason_ = "invalid static window";
    return stats;
  }

  const double start = _now - config_.static_window_seconds_;
  std::vector<const SchurInitImuSample_s *> window;
  for (const auto & sample : imu_) {
    if (sample.timestamp_ >= start && sample.timestamp_ <= _now)
      window.push_back(&sample);
  }
  stats.samples_ = static_cast<int>(window.size());
  if (window.size() < 2) {
    stats.reason_ = "not enough IMU samples in the static window";
    return stats;
  }
  stats.span_seconds_ = window.back()->timestamp_ - window.front()->timestamp_;
  if (stats.span_seconds_ < 0.9 * config_.static_window_seconds_) {
    stats.reason_ = "static IMU window is not full yet";
    return stats;
  }
  if (stats.samples_ < config_.static_min_samples_) {
    stats.reason_ = "too few IMU samples in the static window";
    return stats;
  }

  Eigen::Vector3d acc_mean = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyr_mean = Eigen::Vector3d::Zero();
  for (const auto * sample : window) {
    acc_mean += sample->acc_;
    gyr_mean += sample->gyr_;
  }
  acc_mean /= static_cast<double>(window.size());
  gyr_mean /= static_cast<double>(window.size());

  double acc_var = 0.0;
  double gyr_var = 0.0;
  for (const auto * sample : window) {
    const Eigen::Vector3d da = sample->acc_ - acc_mean;
    const Eigen::Vector3d dg = sample->gyr_ - gyr_mean;
    acc_var += da.squaredNorm();
    gyr_var += dg.squaredNorm();
  }
  const double denom = static_cast<double>(window.size() - 1);
  stats.accel_std_ = std::sqrt(acc_var / denom);
  stats.gyro_std_ = std::sqrt(gyr_var / denom);
  stats.mean_acc_ = acc_mean;
  stats.mean_gyr_ = gyr_mean;
  stats.accel_norm_ = acc_mean.norm();

  for (std::size_t i = 1; i < window.size(); ++i) {
    const double gap = window[i]->timestamp_ - window[i - 1]->timestamp_;
    stats.max_gap_seconds_ = std::max(stats.max_gap_seconds_, gap);
  }

  if (config_.static_max_imu_gap_seconds_ > 0.0 && stats.max_gap_seconds_ > config_.static_max_imu_gap_seconds_) {
    stats.reason_ = "IMU gap inside the static window";
    return stats;
  }
  if (!std::isfinite(stats.accel_std_) || stats.accel_std_ > config_.static_accel_std_) {
    stats.reason_ = "accelerometer variance exceeds the static gate";
    return stats;
  }
  if (!std::isfinite(stats.gyro_std_) || stats.gyro_std_ > config_.static_gyro_std_) {
    stats.reason_ = "gyro variance exceeds the static gate";
    return stats;
  }
  if (config_.static_accel_norm_tol_ > 0.0 && std::abs(stats.accel_norm_ - config_.gravity_) > config_.static_accel_norm_tol_) {
    stats.reason_ = "measured specific force is inconsistent with gravity";
    return stats;
  }
  if (stats.mean_gyr_.norm() > config_.static_gyro_mean_max_) {
    stats.reason_ = "mean gyro exceeds the static bias gate";
    return stats;
  }
  stats.valid_ = true;
  return stats;
}

SchurInitializer::VisualStaticStats_s SchurInitializer::visualStaticStats(double _now) const
{
  VisualStaticStats_s stats;
  if (!std::isfinite(_now) || config_.static_window_seconds_ <= 0.0) {
    stats.reason_ = "invalid visual window";
    return stats;
  }

  const double start = _now - config_.static_window_seconds_;
  std::vector<const SchurInitVisualFrame_s *> window;
  for (const auto & frame : visual_) {
    if (frame.timestamp_ >= start && frame.timestamp_ <= _now && !frame.observations_.empty())
      window.push_back(&frame);
  }
  if (window.size() < 2) {
    stats.reason_ = "not enough visual frames in the static window";
    return stats;
  }
  stats.span_seconds_ = window.back()->timestamp_ - window.front()->timestamp_;
  if (stats.span_seconds_ < config_.static_visual_min_span_seconds_) {
    stats.reason_ = "visual static window is not full yet";
    return stats;
  }

  std::unordered_map<int, Eigen::Vector2d> first;
  first.reserve(window.front()->observations_.size() * 2);
  for (const auto & obs : window.front()->observations_)
    first.emplace(obs.feature_id_ * 8 + obs.camera_id_, obs.pixel_uv_);

  // Compare every newer frame with the oldest reference.  A platform that
  // moves away and returns to the reference by the last frame must not pass:
  // the maximum per-frame median flow over the whole window is what matters.
  double max_median_flow = 0.0;
  int max_matches = 0;
  int quadrants = 0;
  bool have_flow = false;
  for (std::size_t frame_index = 1; frame_index < window.size(); ++frame_index) {
    std::vector<double> flows;
    for (const auto & obs : window[frame_index]->observations_) {
      const auto it = first.find(obs.feature_id_ * 8 + obs.camera_id_);
      if (it == first.end())
        continue;
      const Eigen::Vector2d delta = obs.pixel_uv_ - it->second;
      if (!delta.allFinite())
        continue;
      flows.push_back(delta.norm());
    }
    if (flows.empty())
      continue;
    const double frame_flow = median(flows);
    if (!std::isfinite(frame_flow))
      continue;
    if (!have_flow || frame_flow > max_median_flow) {
      max_median_flow = frame_flow;
      have_flow = true;
    }
    max_matches = std::max(max_matches, static_cast<int>(flows.size()));
  }

  // Coverage from the newest frame against the oldest reference.
  if (config_.static_image_width_ > 0 && config_.static_image_height_ > 0) {
    bool quadrant[4] = {false, false, false, false};
    for (const auto & obs : window.back()->observations_) {
      const auto it = first.find(obs.feature_id_ * 8 + obs.camera_id_);
      if (it == first.end())
        continue;
      const int qx = obs.pixel_uv_.x() >= 0.5 * config_.static_image_width_ ? 1 : 0;
      const int qy = obs.pixel_uv_.y() >= 0.5 * config_.static_image_height_ ? 1 : 0;
      quadrant[qy * 2 + qx] = true;
    }
    for (const bool filled : quadrant)
      quadrants += filled ? 1 : 0;
  }

  stats.matches_ = max_matches;
  stats.quadrants_ = quadrants;
  stats.median_flow_px_ = max_median_flow;
  if (!have_flow) {
    stats.reason_ = "no common features across the visual static window";
    return stats;
  }
  if (stats.matches_ < config_.static_visual_min_matches_) {
    stats.reason_ = "too few matched features for the visual static gate";
    return stats;
  }
  if (!std::isfinite(stats.median_flow_px_) || stats.median_flow_px_ > config_.static_flow_max_px_) {
    stats.reason_ = "visual flow exceeds the static gate";
    return stats;
  }
  if (config_.static_visual_min_quadrants_ > 0 && stats.quadrants_ < config_.static_visual_min_quadrants_) {
    stats.reason_ = "visual static matches do not cover enough image quadrants";
    return stats;
  }
  stats.valid_ = true;
  return stats;
}

bool SchurInitializer::tryStatic(double _now, bool _stereo, SchurInitSeed_s & _seed) const
{
  if (!config_.allow_static_)
    return false;
  if (!_stereo && !config_.allow_mono_static_) {
    _seed.reason_ = "monocular static initialization is disabled (scale unobservable)";
    return false;
  }

  const StaticImuStats_s imu_stats = staticImuStats(_now);
  if (!imu_stats.valid_) {
    _seed.reason_ = imu_stats.reason_;
    return false;
  }
  const VisualStaticStats_s visual_stats = visualStaticStats(_now);
  if (!visual_stats.valid_) {
    _seed.reason_ = visual_stats.reason_;
    return false;
  }

  const double gravity = config_.gravity_;
  const Eigen::Vector3d up = Eigen::Vector3d(0.0, 0.0, 1.0);
  if (imu_stats.mean_acc_.norm() < 1e-6) {
    _seed.reason_ = "mean acceleration is degenerate";
    return false;
  }
  const Eigen::Matrix3d R0 =
    Eigen::Quaterniond::FromTwoVectors(imu_stats.mean_acc_.normalized(), up).toRotationMatrix();
  const Eigen::Vector3d ba = imu_stats.mean_acc_ - R0.transpose() * Eigen::Vector3d(0.0, 0.0, gravity);
  const Eigen::Vector3d bg = imu_stats.mean_gyr_;
  if (ba.norm() > config_.static_ba_norm_max_) {
    _seed.reason_ = "estimated accelerometer bias exceeds the static gate";
    return false;
  }
  if (!ba.allFinite() || !bg.allFinite()) {
    _seed.reason_ = "static bias estimate is non-finite";
    return false;
  }

  _seed = SchurInitSeed_s{};
  _seed.kind_ = SchurInitResultKind_e::K_STATIC;
  _seed.q_ = quaternionOf(R0);
  _seed.p_.setZero();
  _seed.v_.setZero();
  _seed.ba_ = ba;
  _seed.bg_ = bg;
  _seed.gravity_visual_ = Eigen::Vector3d(0.0, 0.0, gravity);
  _seed.world_rotation_.setIdentity();
  _seed.scale_ = 1.0;
  SchurInitSeed_s::FrameState_s frame;
  frame.index_ = -1;
  frame.q_ = _seed.q_;
  frame.p_ = _seed.p_;
  frame.v_ = _seed.v_;
  _seed.frame_states_.push_back(frame);
  _seed.reason_ = "verified static window";
  return true;
}

bool SchurInitializer::solveGyroBias(
  const std::vector<SchurInitMotionFrame_s> & _frames, Eigen::Vector3d & _delta_bg, std::string & _reason)
{
  _delta_bg.setZero();
  if (_frames.empty()) {
    _reason = "no motion frames for gyro-bias estimation";
    return false;
  }

  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d b = Eigen::Vector3d::Zero();
  double max_rotation_residual = 0.0;
  for (const auto & frame : _frames) {
    if (!finiteMotionFrame(frame)) {
      _reason = "non-finite motion frame";
      return false;
    }
    const Eigen::Quaterniond q_ij(frame.R0_.transpose() * frame.R1_);
    const Eigen::Quaterniond delta_q = frame.delta_q_.normalized();
    const Eigen::Vector3d residual = 2.0 * (delta_q.inverse() * q_ij).vec();
    if (!residual.allFinite()) {
      _reason = "non-finite gyro residual";
      return false;
    }
    max_rotation_residual =
      std::max(max_rotation_residual, rotationAngle((q_ij * delta_q.inverse()).toRotationMatrix()));
    A += frame.dq_dbg_.transpose() * frame.dq_dbg_;
    b += frame.dq_dbg_.transpose() * residual;
  }

  if (A.norm() < 1e-12) {
    if (max_rotation_residual < 1e-4) {
      _reason.clear();
      return true;  // perfectly consistent samples and no gyro excitation needed
    }
    _reason = "gyro-bias Jacobian is unobservable";
    return false;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(A);
  if (eig.info() != Eigen::Success || eig.eigenvalues().minCoeff() < 1e-12) {
    _reason = "gyro-bias normal matrix is singular";
    return false;
  }
  _delta_bg = A.ldlt().solve(b);
  if (!_delta_bg.allFinite()) {
    _reason = "gyro-bias solve produced a non-finite correction";
    return false;
  }
  _reason.clear();
  return true;
}

bool SchurInitializer::alignStereoMotion(
  const std::vector<SchurInitMotionFrame_s> & _frames, const SchurInitConfig_s & _config,
  const Eigen::Vector3d & _bg_guess, const Eigen::Vector3d & _ba_guess, SchurInitSeed_s & _seed,
  std::string & _reason)
{
  _seed = SchurInitSeed_s{};
  _reason.clear();
  if (static_cast<int>(_frames.size()) < _config.dynamic_min_frames_) {
    _reason = "not enough stereo visual/IMU pairs for dynamic initialization";
    return false;
  }
  for (std::size_t i = 0; i < _frames.size(); ++i) {
    if (!finiteMotionFrame(_frames[i])) {
      _reason = "non-finite or out-of-order stereo motion frame";
      return false;
    }
    if (i > 0 && _frames[i].t0_ <= _frames[i - 1].t0_) {
      _reason = "stereo motion frames are not strictly time ordered";
      return false;
    }
  }

  const double rotation_excitation = rotationAngle(_frames.front().R0_.transpose() * _frames.back().R1_);
  const double translation_excitation = (_frames.back().p1_ - _frames.front().p0_).norm();
  if (
    rotation_excitation < _config.dynamic_min_rotation_deg_ * K_PI / 180.0 &&
    translation_excitation < _config.dynamic_min_translation_m_) {
    _reason = "insufficient visual excitation for dynamic initialization";
    return false;
  }

  Eigen::Vector3d delta_bg = Eigen::Vector3d::Zero();
  if (!solveGyroBias(_frames, delta_bg, _reason))
    return false;
  if (delta_bg.norm() > _config.dynamic_max_gyro_bias_norm_) {
    _reason = "gyro-bias correction exceeds the dynamic initialization gate";
    return false;
  }

  const int n = static_cast<int>(_frames.size());
  const double gravity_mag = _config.gravity_;
  for (std::size_t i = 0; i < _frames.size(); ++i) {
    if (!finiteMotionFrame(_frames[i])) {
      _reason = "non-finite or out-of-order stereo motion frame";
      return false;
    }
    if (i > 0) {
      const double continuity = std::abs(_frames[i].t0_ - _frames[i - 1].t1_);
      if (continuity > std::max(1e-4, 0.1 * (_frames[i].t1_ - _frames[i].t0_))) {
        _reason = "stereo motion frames are not contiguous in time";
        return false;
      }
    }
  }

  // Stereo PnP poses are metric, so velocities are observed directly by
  // central differences in the visual world.  This avoids the rank-deficient
  // joint (per-frame velocity, gravity, scale) system: with metric stereo the
  // scale is known to be one, and gravity is then a small 3-DOF linear solve
  // from the IMU preintegration velocity equations.
  const int num_frames = n + 1;
  std::vector<double> times(static_cast<std::size_t>(num_frames), 0.0);
  std::vector<Eigen::Matrix3d> rotations(static_cast<std::size_t>(num_frames));
  std::vector<Eigen::Vector3d> positions(static_cast<std::size_t>(num_frames));
  std::vector<Eigen::Vector3d> velocities(static_cast<std::size_t>(num_frames), Eigen::Vector3d::Zero());
  for (int i = 0; i < n; ++i) {
    times[static_cast<std::size_t>(i)] = _frames[static_cast<std::size_t>(i)].t0_;
    rotations[static_cast<std::size_t>(i)] = _frames[static_cast<std::size_t>(i)].R0_;
    positions[static_cast<std::size_t>(i)] = _frames[static_cast<std::size_t>(i)].p0_;
  }
  times.back() = _frames.back().t1_;
  rotations.back() = _frames.back().R1_;
  positions.back() = _frames.back().p1_;
  for (int i = 0; i < num_frames; ++i) {
    if (num_frames >= 3 && i == 0) {
      velocities[0] =
        quadraticDerivativeAt(times[0], times[0], times[1], times[2], positions[0], positions[1], positions[2]);
    } else if (num_frames >= 3 && i == num_frames - 1) {
      velocities[static_cast<std::size_t>(i)] = quadraticDerivativeAt(
        times[static_cast<std::size_t>(i)], times[static_cast<std::size_t>(i)],
        times[static_cast<std::size_t>(i - 1)], times[static_cast<std::size_t>(i - 2)],
        positions[static_cast<std::size_t>(i)], positions[static_cast<std::size_t>(i - 1)],
        positions[static_cast<std::size_t>(i - 2)]);
    } else {
      velocities[static_cast<std::size_t>(i)] = quadraticDerivativeAt(
        times[static_cast<std::size_t>(i)], times[static_cast<std::size_t>(i)],
        times[static_cast<std::size_t>(i - 1)], times[static_cast<std::size_t>(i + 1)],
        positions[static_cast<std::size_t>(i)], positions[static_cast<std::size_t>(i - 1)],
        positions[static_cast<std::size_t>(i + 1)]);
    }
    if (!velocities[static_cast<std::size_t>(i)].allFinite()) {
      _reason = "visual velocity is non-finite";
      return false;
    }
  }

  // dt * R_i^T g = delta_v - R_i^T (v_j - v_i), stacked over all pairs.
  Eigen::Matrix3d gravity_normal = Eigen::Matrix3d::Zero();
  Eigen::Vector3d gravity_rhs = Eigen::Vector3d::Zero();
  for (int i = 0; i < n; ++i) {
    const SchurInitMotionFrame_s & frame = _frames[static_cast<std::size_t>(i)];
    const double dt = frame.t1_ - frame.t0_;
    const Eigen::Matrix3d R0t = frame.R0_.transpose();
    const Eigen::Vector3d dv_vis =
      R0t * (velocities[static_cast<std::size_t>(i + 1)] - velocities[static_cast<std::size_t>(i)]);
    const Eigen::Vector3d rhs = frame.delta_v_ - dv_vis;
    // A = dt R0t, so A^T A = dt^2 I and A^T rhs = dt R0 rhs.
    gravity_normal += dt * dt * Eigen::Matrix3d::Identity();
    gravity_rhs += dt * frame.R0_ * rhs;
  }
  Eigen::Vector3d g_visual = gravity_normal.ldlt().solve(gravity_rhs);
  if (!g_visual.allFinite() || g_visual.norm() < 1e-6) {
    _reason = "visual/IMU velocity equations produced a degenerate gravity";
    return false;
  }
  if (std::abs(g_visual.norm() - gravity_mag) > _config.dynamic_gravity_tol_) {
    _reason = "linear gravity norm is inconsistent with the configured gravity";
    return false;
  }
  g_visual = g_visual.normalized() * gravity_mag;

  for (int iter = 0; iter < std::max(1, _config.dynamic_gravity_refine_iterations_); ++iter) {
    const Eigen::Matrix<double, 3, 2> B = tangentBasis(g_visual);
    Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
    Eigen::Vector2d y = Eigen::Vector2d::Zero();
    for (int i = 0; i < n; ++i) {
      const SchurInitMotionFrame_s & frame = _frames[static_cast<std::size_t>(i)];
      const double dt = frame.t1_ - frame.t0_;
      const Eigen::Matrix3d R0t = frame.R0_.transpose();
      const Eigen::Vector3d & vi = velocities[static_cast<std::size_t>(i)];
      const Eigen::Vector3d & vj = velocities[static_cast<std::size_t>(i + 1)];

      const Eigen::Matrix<double, 3, 2> Jv = dt * R0t * B;
      const Eigen::Vector3d rv = frame.delta_v_ - R0t * (vj - vi) - dt * R0t * g_visual;
      H.noalias() += Jv.transpose() * Jv;
      y.noalias() += Jv.transpose() * rv;

      const Eigen::Matrix<double, 3, 2> Jp = 0.5 * dt * dt * R0t * B;
      const Eigen::Vector3d measured_p = frame.delta_p_ + R0t * frame.R1_ * frame.p_ci_ - frame.p_ci_;
      const Eigen::Vector3d rp =
        measured_p - R0t * (frame.p1_ - frame.p0_ - vi * dt + 0.5 * dt * dt * g_visual);
      H.noalias() += Jp.transpose() * Jp;
      y.noalias() += Jp.transpose() * rp;
    }
    const Eigen::Vector2d dg = H.ldlt().solve(y);
    if (!dg.allFinite()) {
      _reason = "gravity refinement solve is singular";
      return false;
    }
    g_visual = (g_visual + B * dg).normalized() * gravity_mag;
  }

  double max_position_residual = 0.0;
  double max_velocity_residual = 0.0;
  double max_rotation_residual = 0.0;
  for (int i = 0; i < n; ++i) {
    const SchurInitMotionFrame_s & frame = _frames[static_cast<std::size_t>(i)];
    const double dt = frame.t1_ - frame.t0_;
    const Eigen::Matrix3d R0t = frame.R0_.transpose();
    const Eigen::Vector3d & vi = velocities[static_cast<std::size_t>(i)];
    const Eigen::Vector3d & vj = velocities[static_cast<std::size_t>(i + 1)];
    const Eigen::Vector3d measured_p = frame.delta_p_ + R0t * frame.R1_ * frame.p_ci_ - frame.p_ci_;
    const Eigen::Vector3d predicted_p = R0t * (frame.p1_ - frame.p0_ - vi * dt + 0.5 * dt * dt * g_visual);
    const Eigen::Vector3d measured_v = frame.delta_v_;
    const Eigen::Vector3d predicted_v = R0t * (vj - vi + g_visual * dt);
    const double rotation_residual = rotationAngle(
      (frame.delta_q_.normalized().inverse() * Eigen::Quaterniond(frame.R0_.transpose() * frame.R1_))
        .normalized()
        .toRotationMatrix());
    max_position_residual = std::max(max_position_residual, (predicted_p - measured_p).norm());
    max_velocity_residual = std::max(max_velocity_residual, (predicted_v - measured_v).norm());
    max_rotation_residual = std::max(max_rotation_residual, rotation_residual);
  }

  if (max_position_residual > _config.dynamic_max_position_residual_m_) {
    _reason = "dynamic position residual exceeds the gate";
    return false;
  }
  if (max_velocity_residual > _config.dynamic_max_velocity_residual_mps_) {
    _reason = "dynamic velocity residual exceeds the gate";
    return false;
  }
  if (max_rotation_residual > _config.dynamic_max_rotation_residual_deg_ * K_PI / 180.0) {
    _reason = "dynamic rotation residual exceeds the gate";
    return false;
  }

  // g_visual is the VINS positive-down gravity compensation recovered from
  // delta_v = R_i^T (v_j - v_i + g_visual dt).  The VINS world z axis points
  // down, so map it to EKF +z.  EKF physical gravity is then -z, matching
  // EkfState's propagator g_w = (0, 0, -|g|).
  const Eigen::Vector3d vins_down_axis(0.0, 0.0, gravity_mag);
  const Eigen::Matrix3d R_align =
    Eigen::Quaterniond::FromTwoVectors(g_visual.normalized(), vins_down_axis.normalized()).toRotationMatrix();
  const Eigen::Vector3d p_ref = positions.front();

  _seed = SchurInitSeed_s{};
  _seed.kind_ = SchurInitResultKind_e::K_DYNAMIC;
  _seed.ba_ = _ba_guess;
  _seed.bg_ = _bg_guess + delta_bg;
  _seed.gravity_visual_ = g_visual;
  _seed.world_rotation_ = R_align;
  _seed.scale_ = 1.0;
  _seed.frame_states_.reserve(static_cast<std::size_t>(num_frames));
  for (int i = 0; i < num_frames; ++i) {
    SchurInitSeed_s::FrameState_s state;
    state.index_ = i;
    state.q_ = quaternionOf(R_align * rotations[static_cast<std::size_t>(i)]);
    state.p_ = R_align * (positions[static_cast<std::size_t>(i)] - p_ref);
    state.v_ = R_align * velocities[static_cast<std::size_t>(i)];
    _seed.frame_states_.push_back(state);
  }
  const auto & last = _seed.frame_states_.back();
  _seed.q_ = last.q_;
  _seed.p_ = last.p_;
  _seed.v_ = last.v_;
  _seed.reason_ = "stereo visual velocity + IMU gravity alignment";
  return true;
}

SchurInitSeed_s SchurInitializer::attempt(
  double _now, bool _stereo, bool _has_imu, const std::vector<SchurInitMotionFrame_s> & _motion_frames,
  const Eigen::Vector3d & _bg_guess, const Eigen::Vector3d & _ba_guess)
{
  prune(_now);
  last_reason_.clear();
  if (!_has_imu) {
    last_reason_ = "no IMU data";
    return SchurInitSeed_s{};
  }

  SchurInitSeed_s static_seed;
  if (_stereo || config_.allow_mono_static_) {
    if (tryStatic(_now, _stereo, static_seed)) {
      last_reason_ = static_seed.reason_;
      return static_seed;
    }
    last_reason_ = static_seed.reason_;
  } else {
    last_reason_ = "monocular static initialization is disabled (scale unobservable)";
  }

  if (!_stereo) {
    if (last_reason_.empty())
      last_reason_ = "monocular dynamic initialization is not implemented";
    return SchurInitSeed_s{};
  }
  if (!config_.allow_dynamic_) {
    last_reason_ += "; dynamic initialization disabled";
    return SchurInitSeed_s{};
  }
  if (static_cast<int>(_motion_frames.size()) < config_.dynamic_min_frames_) {
    if (last_reason_.empty())
      last_reason_ = "waiting for enough stereo visual/IMU pairs";
    else
      last_reason_ += "; waiting for enough stereo visual/IMU pairs";
    return SchurInitSeed_s{};
  }

  SchurInitSeed_s dynamic_seed;
  std::string dynamic_reason;
  if (alignStereoMotion(_motion_frames, config_, _bg_guess, _ba_guess, dynamic_seed, dynamic_reason)) {
    last_reason_ = dynamic_seed.reason_;
    return dynamic_seed;
  }

  dynamic_seed.kind_ = SchurInitResultKind_e::K_REJECTED;
  dynamic_seed.reason_ = dynamic_reason;
  last_reason_ = dynamic_reason;
  return dynamic_seed;
}

}  // namespace backend
}  // namespace vins
