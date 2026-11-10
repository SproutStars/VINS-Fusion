/*******************************************************
 * Conservative landing static constraint for the Schur EKF.
 *
 * This module is deliberately ROS-free so the state machine and the EKF
 * pseudo-measurement can be unit tested directly.
 *
 * The constraint is applied only when all independent cues agree:
 *   1. a recent accelerometer impact (|a| above a configured threshold),
 *   2. sustained low image flow with enough matched features covering enough
 *      image quadrants,
 *   3. low debiased IMU dynamics (angular rate and specific-force magnitude),
 *   4. a low current velocity estimate.
 *
 * IMU cues alone cannot distinguish a stationary platform from constant-
 * velocity motion, and distant uniform motion can produce little image flow.
 * The joint gates make these ambiguous cases unlikely (the recent impact arms
 * the episode and the speed limit excludes obvious motion), but they are not a
 * proof of stationarity.  The finite-covariance pseudo-measurement and the
 * immediate motion exit reduce the risk of a rare false positive.
 *
 * The zero-velocity measurement is applied as a normal EKF update with a
 * finite covariance (Joseph form); it is not a hard state reset/freeze.
 *******************************************************/

#pragma once

#include <algorithm>
#include <cmath>
#include <deque>
#include <eigen3/Eigen/Dense>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "ekf_state.h"

namespace vins
{
namespace backend
{

struct LandingPixelObservation_s {
  int id_ = -1;
  Eigen::Vector2d pixel_ = Eigen::Vector2d::Zero();
};

struct LandingStaticConfig_s {
  bool enabled_ = false;

  // Sustained visual stillness. Flow is measured per image frame (pixels).
  double min_still_seconds_ = 0.5;
  // Image flow is normalised to this reference interval before comparison,
  // so the threshold means pixels per 50 ms regardless of the backend rate.
  double flow_reference_seconds_ = 0.05;
  double max_flow_px_ = 0.25;
  double min_static_fraction_ = 0.7;
  int min_features_ = 30;
  int min_quadrants_ = 3;
  int image_width_ = 752;
  int image_height_ = 480;

  // Recent impact that arms the landing episode.
  double impact_window_seconds_ = 6.0;
  double impact_accel_ = 20.0;

  // Timing/IMU continuity. A long image or IMU gap clears the evidence.
  double max_image_gap_seconds_ = 0.25;
  double imu_window_seconds_ = 0.5;
  double max_imu_sample_gap_seconds_ = 0.05;
  int min_imu_samples_ = 10;

  // Debiased IMU low-dynamics tests (entry thresholds).
  double max_gyro_rad_s_ = 0.08;
  double max_accel_dev_ = 0.60;
  double max_accel_std_ = 0.35;
  double max_gyro_std_ = 0.03;
  double max_speed_ = 0.25;

  // Wider exit thresholds provide hysteresis but still leave promptly.
  double exit_flow_px_ = 0.50;
  double exit_speed_ = 0.35;
  double exit_gyro_rad_s_ = 0.15;
  double exit_accel_dev_ = 1.00;

  double gravity_ = 9.81007;
};

struct LandingFrameResult_s {
  bool active_ = false;
  bool entered_ = false;
  bool exited_ = false;
  bool visual_still_ = false;
  bool imu_still_ = false;
  bool impact_recent_ = false;
  bool speed_ok_ = false;

  int matched_features_ = 0;
  int still_features_ = 0;
  int quadrants_ = 0;
  double median_flow_px_ = 0.0;
  double static_fraction_ = 0.0;
  double still_duration_ = 0.0;
  double impact_age_ = 0.0;
  double gyro_norm_ = 0.0;
  double accel_dev_ = 0.0;
  double accel_std_ = 0.0;
  double gyro_std_ = 0.0;
  std::string exit_reason_;
};

class LandingStaticDetector {
public:
  void configure(const LandingStaticConfig_s & _config) { config_ = _config; }
  const LandingStaticConfig_s & config() const { return config_; }

  // Clears every piece of evidence and the active state. Called on estimator
  // restarts/resets and on a backwards sensor-time jump.
  void reset()
  {
    previous_frame_time_ = -1.0;
    previous_pixels_.clear();
    still_since_ = -1.0;
    active_ = false;
    has_impact_ = false;
    last_impact_time_ = -1.0;
    previous_imu_time_ = -1.0;
    imu_samples_.clear();
  }

  bool active() const { return active_; }

  void observeImu(double _t, const Eigen::Vector3d & _acc, const Eigen::Vector3d & _gyr)
  {
    if (!config_.enabled_ || !std::isfinite(_t) || !_acc.allFinite() || !_gyr.allFinite())
      return;
    if (previous_imu_time_ >= 0.0) {
      constexpr double kTimeEps = 1e-9;
      if (_t < previous_imu_time_ - kTimeEps) {
        reset();  // clock rollback: no zero-velocity evidence survives
        previous_imu_time_ = _t;
        imu_samples_.push_back(ImuSample_s{_t, _acc, _gyr});
        return;
      }
      if (_t <= previous_imu_time_ + kTimeEps)
        return;  // getIMUInterval may replay the boundary sample
    }
    if (previous_imu_time_ >= 0.0 && _t - previous_imu_time_ > 1.0)
      imu_samples_.clear();  // camera/IMU outage: require a fresh window
    previous_imu_time_ = _t;
    imu_samples_.push_back(ImuSample_s{_t, _acc, _gyr});

    if (_acc.norm() >= config_.impact_accel_) {
      has_impact_ = true;
      last_impact_time_ = _t;
    }

    const double keep = std::max(1.0, 2.0 * config_.imu_window_seconds_);
    while (!imu_samples_.empty() && imu_samples_.front().t_ < _t - keep)
      imu_samples_.pop_front();
  }

  LandingFrameResult_s observeFrame(
    double _t, const std::vector<LandingPixelObservation_s> & _observations,
    const Eigen::Vector3d & _velocity, const Eigen::Vector3d & _ba, const Eigen::Vector3d & _bg)
  {
    LandingFrameResult_s result;
    if (!config_.enabled_) {
      if (active_) {
        active_ = false;
        result.exited_ = true;
        result.exit_reason_ = "disabled";
      }
      previous_frame_time_ = -1.0;
      previous_pixels_.clear();
      still_since_ = -1.0;
      return result;
    }
    if (!std::isfinite(_t)) {
      deactivate(result, "non-finite frame time");
      return result;
    }

    std::unordered_map<int, Eigen::Vector2d> current;
    current.reserve(_observations.size());
    for (const LandingPixelObservation_s & observation : _observations) {
      if (observation.id_ < 0 || !observation.pixel_.allFinite())
        continue;
      current.emplace(observation.id_, observation.pixel_);
    }

    const bool has_previous = previous_frame_time_ >= 0.0;
    const double frame_gap = has_previous ? _t - previous_frame_time_ : -1.0;
    if (
      has_previous && (!(frame_gap > 0.0) || frame_gap > config_.max_image_gap_seconds_ ||
                       !_velocity.allFinite() || !_ba.allFinite() || !_bg.allFinite())) {
      // Clock rollback, image loss, or an invalid state: drop the
      // evidence and do not carry a zero-velocity constraint across it.
      if (frame_gap < 0.0) {
        const bool was_active = active_;
        reset();
        if (was_active) {
          result.exited_ = true;
          result.exit_reason_ = "clock rollback";
        }
        previous_frame_time_ = _t;
        previous_pixels_ = std::move(current);
        return result;
      }
      previous_frame_time_ = _t;
      previous_pixels_ = std::move(current);
      still_since_ = -1.0;
      deactivate(result, "image gap");
      return result;
    }

    if (!has_previous) {
      previous_frame_time_ = _t;
      previous_pixels_ = std::move(current);
      still_since_ = -1.0;
      if (active_)
        deactivate(result, "missing previous frame");
      return result;
    }

    const double flow_scale = config_.flow_reference_seconds_ / frame_gap;
    std::vector<double> flows;
    flows.reserve(previous_pixels_.size());
    int quadrant_count[4] = {0, 0, 0, 0};
    for (const auto & entry : current) {
      const auto previous = previous_pixels_.find(entry.first);
      if (previous == previous_pixels_.end())
        continue;
      const double flow = (entry.second - previous->second).norm() * flow_scale;
      if (!std::isfinite(flow))
        continue;
      flows.push_back(flow);
      if (flow <= config_.max_flow_px_) {
        const int qx = entry.second.x() < 0.5 * config_.image_width_ ? 0 : 1;
        const int qy = entry.second.y() < 0.5 * config_.image_height_ ? 0 : 1;
        ++quadrant_count[2 * qy + qx];
      }
    }

    result.matched_features_ = static_cast<int>(flows.size());
    result.still_features_ = 0;
    if (!flows.empty()) {
      std::vector<double> sorted = flows;
      const std::size_t middle = sorted.size() / 2;
      std::nth_element(sorted.begin(), sorted.begin() + middle, sorted.end());
      result.median_flow_px_ = sorted[middle];
      for (double flow : flows)
        if (flow <= config_.max_flow_px_)
          ++result.still_features_;
      result.static_fraction_ = static_cast<double>(result.still_features_) / static_cast<double>(flows.size());
    }
    const int quadrant_min = std::max(1, config_.min_features_ / 8);
    for (int i : quadrant_count)
      if (i >= quadrant_min)
        ++result.quadrants_;

    const bool enough_visual =
      result.matched_features_ >= config_.min_features_ && result.still_features_ >= config_.min_features_ &&
      result.static_fraction_ >= config_.min_static_fraction_ &&
      result.median_flow_px_ <= config_.max_flow_px_ && result.quadrants_ >= config_.min_quadrants_;
    result.visual_still_ = enough_visual;

    double gyro_norm = 0.0;
    double accel_dev = 0.0;
    double accel_std = 0.0;
    double gyro_std = 0.0;
    const bool imu_entry = evaluateImu(_t, _ba, _bg, false, gyro_norm, accel_dev, accel_std, gyro_std);
    const bool imu_exit = evaluateImu(_t, _ba, _bg, true, gyro_norm, accel_dev, accel_std, gyro_std);
    result.imu_still_ = imu_entry;
    result.gyro_norm_ = gyro_norm;
    result.accel_dev_ = accel_dev;
    result.accel_std_ = accel_std;
    result.gyro_std_ = gyro_std;

    result.speed_ok_ = _velocity.allFinite() && _velocity.norm() <= config_.max_speed_;
    result.impact_recent_ = has_impact_ && std::isfinite(last_impact_time_) && _t >= last_impact_time_ &&
                            (_t - last_impact_time_) <= config_.impact_window_seconds_;
    result.impact_age_ = has_impact_ ? _t - last_impact_time_ : std::numeric_limits<double>::infinity();

    if (active_) {
      const bool exit_visual = result.matched_features_ < std::max(10, config_.min_features_ / 2) ||
                               result.median_flow_px_ > config_.exit_flow_px_ ||
                               result.static_fraction_ < 0.5 * config_.min_static_fraction_ ||
                               result.quadrants_ < std::max(2, config_.min_quadrants_ - 1);
      const bool exit_speed = !_velocity.allFinite() || _velocity.norm() > config_.exit_speed_;
      if (exit_visual || !imu_exit || exit_speed) {
        deactivate(result, exit_visual ? "visual motion" : (!imu_exit ? "imu motion" : "speed"));
      } else {
        result.active_ = true;
        result.still_duration_ = still_since_ >= 0.0 ? _t - still_since_ : 0.0;
      }
    } else {
      const bool entry = result.visual_still_ && imu_entry && result.impact_recent_ && result.speed_ok_;
      if (entry) {
        if (still_since_ < 0.0)
          still_since_ = _t;
        result.still_duration_ = _t - still_since_;
        constexpr double kStillTimeEps = 1e-6;
        if (result.still_duration_ + kStillTimeEps >= config_.min_still_seconds_) {
          active_ = true;
          result.active_ = true;
          result.entered_ = true;
        }
      } else {
        still_since_ = -1.0;
      }
    }

    previous_frame_time_ = _t;
    previous_pixels_ = std::move(current);
    return result;
  }

private:
  struct ImuSample_s {
    double t_ = 0.0;
    Eigen::Vector3d acc_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyr_ = Eigen::Vector3d::Zero();
  };

  void deactivate(LandingFrameResult_s & _result, const char * _reason)
  {
    if (active_) {
      _result.exited_ = true;
      _result.exit_reason_ = _reason;
    }
    active_ = false;
    still_since_ = -1.0;
  }

  bool evaluateImu(
    double _t, const Eigen::Vector3d & _ba, const Eigen::Vector3d & _bg, bool _exit_mode, double & _gyro_norm,
    double & _accel_dev, double & _accel_std, double & _gyro_std) const
  {
    _gyro_norm = 0.0;
    _accel_dev = 0.0;
    _accel_std = 0.0;
    _gyro_std = 0.0;
    if (imu_samples_.empty() || !std::isfinite(_t))
      return false;
    if (_t - imu_samples_.back().t_ > config_.max_image_gap_seconds_ || _t < imu_samples_.back().t_ - config_.max_image_gap_seconds_)
      return false;

    const double window_start = _t - config_.imu_window_seconds_;
    std::vector<Eigen::Vector3d> gyro;
    std::vector<double> accel_norms;
    double first_sample_time = std::numeric_limits<double>::infinity();
    double last_sample_time = -std::numeric_limits<double>::infinity();
    double previous_sample_time = -std::numeric_limits<double>::infinity();
    for (const ImuSample_s & sample : imu_samples_) {
      if (sample.t_ < window_start)
        continue;
      if (sample.t_ > _t + 1e-6)
        break;
      if (std::isfinite(previous_sample_time) && sample.t_ - previous_sample_time > config_.max_imu_sample_gap_seconds_)
        return false;
      previous_sample_time = sample.t_;
      first_sample_time = std::min(first_sample_time, sample.t_);
      last_sample_time = std::max(last_sample_time, sample.t_);
      gyro.push_back(sample.gyr_ - _bg);
      accel_norms.push_back((sample.acc_ - _ba).norm());
    }
    if (
      static_cast<int>(gyro.size()) < config_.min_imu_samples_ ||
      static_cast<int>(accel_norms.size()) < config_.min_imu_samples_)
      return false;
    // Validate the actual selected-sample span and gaps. Counting samples
    // alone lets a short burst after a reset masquerade as a 0.5 s window.
    const double sample_span = last_sample_time - first_sample_time;
    if (!(sample_span >= 0.8 * config_.imu_window_seconds_))
      return false;

    Eigen::Vector3d gyro_mean = Eigen::Vector3d::Zero();
    double accel_mean = 0.0;
    for (std::size_t i = 0; i < gyro.size(); ++i) {
      gyro_mean += gyro[i];
      accel_mean += accel_norms[i];
    }
    gyro_mean /= static_cast<double>(gyro.size());
    accel_mean /= static_cast<double>(accel_norms.size());
    _gyro_norm = gyro_mean.norm();
    _accel_dev = std::abs(accel_mean - config_.gravity_);

    double gyro_var = 0.0;
    double accel_var = 0.0;
    for (std::size_t i = 0; i < gyro.size(); ++i) {
      gyro_var += (gyro[i] - gyro_mean).squaredNorm();
      const double d = accel_norms[i] - accel_mean;
      accel_var += d * d;
    }
    _gyro_std = std::sqrt(gyro_var / static_cast<double>(gyro.size()));
    _accel_std = std::sqrt(accel_var / static_cast<double>(accel_norms.size()));

    const double max_gyro = _exit_mode ? config_.exit_gyro_rad_s_ : config_.max_gyro_rad_s_;
    const double max_accel = _exit_mode ? config_.exit_accel_dev_ : config_.max_accel_dev_;
    return _gyro_norm <= max_gyro && _accel_dev <= max_accel && _gyro_std <= config_.max_gyro_std_ &&
           _accel_std <= config_.max_accel_std_;
  }

  LandingStaticConfig_s config_;
  double previous_frame_time_ = -1.0;
  std::unordered_map<int, Eigen::Vector2d> previous_pixels_;
  double still_since_ = -1.0;
  bool active_ = false;
  bool has_impact_ = false;
  double last_impact_time_ = -1.0;
  double previous_imu_time_ = -1.0;
  std::deque<ImuSample_s> imu_samples_;
};

struct ZeroVelocityUpdateResult_s {
  bool applied_ = false;
  double chi2_ = 0.0;
  double sigma_ = 0.0;
  Eigen::Vector3d dv_ = Eigen::Vector3d::Zero();
  std::string failure_reason_;
};

// Applies z = 0 - v with covariance R = sigma^2 I as a regular EKF update.
// The full covariance is updated with the Joseph form; the nominal correction
// is injected into the same error-state layout used by the visual update.
inline ZeroVelocityUpdateResult_s applyZeroVelocityUpdate(EkfState & _state, double _sigma)
{
  ZeroVelocityUpdateResult_s result;
  result.sigma_ = _sigma;
  if (!_state.initialized() || _state.size() <= 0) {
    result.failure_reason_ = "uninitialized state";
    return result;
  }
  if (!std::isfinite(_sigma) || _sigma <= 0.0) {
    result.failure_reason_ = "invalid sigma";
    return result;
  }
  const int n = _state.size();
  Eigen::MatrixXd & P = _state.covariance();
  if (P.rows() != n || P.cols() != n || !P.allFinite() || !_state.imu().v_.allFinite()) {
    result.failure_reason_ = "invalid covariance/state";
    return result;
  }

  const Eigen::Matrix3d R = _sigma * _sigma * Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d S = P.block<3, 3>(K_IMU_V, K_IMU_V) + R;
  Eigen::LDLT<Eigen::Matrix3d> ldlt(S);
  if (ldlt.info() != Eigen::Success || !(ldlt.vectorD().minCoeff() > 0.0)) {
    result.failure_reason_ = "singular innovation covariance";
    return result;
  }

  const Eigen::Vector3d z = -_state.imu().v_;
  const Eigen::MatrixXd K = P.middleCols<3>(K_IMU_V) * ldlt.solve(Eigen::Matrix3d::Identity());
  if (!K.allFinite()) {
    result.failure_reason_ = "non-finite gain";
    return result;
  }
  const Eigen::VectorXd dx = K * z;
  result.chi2_ = z.dot(ldlt.solve(z));
  if (!dx.allFinite()) {
    result.failure_reason_ = "non-finite correction";
    return result;
  }

  Eigen::MatrixXd IKH = Eigen::MatrixXd::Identity(n, n);
  IKH.middleCols<3>(K_IMU_V) -= K;
  Eigen::MatrixXd P_new = IKH * P * IKH.transpose() + K * R * K.transpose();
  if (!P_new.allFinite()) {
    result.failure_reason_ = "non-finite covariance";
    return result;
  }
  EkfState::symmetrize(P_new);
  if (!EkfState::isPositiveSemidefinite(P_new)) {
    result.failure_reason_ = "non-PSD covariance";
    return result;
  }

  Eigen::Matrix<double, 6, 1> dpose;
  dpose.head<3>() = dx.segment<3>(K_IMU_THETA);
  dpose.tail<3>() = dx.segment<3>(K_IMU_P);
  EkfState::applyPoseCorrection(_state.imu().q_, _state.imu().p_, dpose);
  result.dv_ = dx.segment<3>(K_IMU_V);
  _state.imu().v_ += result.dv_;
  _state.imu().ba_ += dx.segment<3>(K_IMU_BA);
  _state.imu().bg_ += dx.segment<3>(K_IMU_BG);
  for (int i = 0; i < _state.numClones(); ++i) {
    const Eigen::Matrix<double, 6, 1> dclone = dx.segment<6>(_state.cloneOffset(i));
    EkfState::applyPoseCorrection(_state.clone(i).q_, _state.clone(i).p_, dclone);
  }
  P = std::move(P_new);
  EkfState::symmetrize(P);
  result.applied_ = true;
  return result;
}

}  // namespace backend
}  // namespace vins
