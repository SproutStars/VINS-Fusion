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

#include <eigen3/Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "ekf_state.h"

namespace vins
{
namespace backend
{

struct LandingPixelObservation
{
    int id = -1;
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
};

struct LandingStaticConfig
{
    bool enabled = false;

    // Sustained visual stillness. Flow is measured per image frame (pixels).
    double min_still_seconds = 0.5;
    // Image flow is normalised to this reference interval before comparison,
    // so the threshold means pixels per 50 ms regardless of the backend rate.
    double flow_reference_seconds = 0.05;
    double max_flow_px = 0.25;
    double min_static_fraction = 0.7;
    int min_features = 30;
    int min_quadrants = 3;
    int image_width = 752;
    int image_height = 480;

    // Recent impact that arms the landing episode.
    double impact_window_seconds = 6.0;
    double impact_accel = 20.0;

    // Timing/IMU continuity. A long image or IMU gap clears the evidence.
    double max_image_gap_seconds = 0.25;
    double imu_window_seconds = 0.5;
    double max_imu_sample_gap_seconds = 0.05;
    int min_imu_samples = 10;

    // Debiased IMU low-dynamics tests (entry thresholds).
    double max_gyro_rad_s = 0.08;
    double max_accel_dev = 0.60;
    double max_accel_std = 0.35;
    double max_gyro_std = 0.03;
    double max_speed = 0.25;

    // Wider exit thresholds provide hysteresis but still leave promptly.
    double exit_flow_px = 0.50;
    double exit_speed = 0.35;
    double exit_gyro_rad_s = 0.15;
    double exit_accel_dev = 1.00;

    double gravity = 9.81007;
};

struct LandingFrameResult
{
    bool active = false;
    bool entered = false;
    bool exited = false;
    bool visual_still = false;
    bool imu_still = false;
    bool impact_recent = false;
    bool speed_ok = false;

    int matched_features = 0;
    int still_features = 0;
    int quadrants = 0;
    double median_flow_px = 0.0;
    double static_fraction = 0.0;
    double still_duration = 0.0;
    double impact_age = 0.0;
    double gyro_norm = 0.0;
    double accel_dev = 0.0;
    double accel_std = 0.0;
    double gyro_std = 0.0;
    std::string exit_reason;
};

class LandingStaticDetector
{
  public:
    void configure(const LandingStaticConfig &config) { config_ = config; }
    const LandingStaticConfig &config() const { return config_; }

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

    void observe_imu(double t, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
    {
        if (!config_.enabled || !std::isfinite(t) || !acc.allFinite() || !gyr.allFinite())
            return;
        if (previous_imu_time_ >= 0.0)
        {
            constexpr double kTimeEps = 1e-9;
            if (t < previous_imu_time_ - kTimeEps)
            {
                reset();  // clock rollback: no zero-velocity evidence survives
                previous_imu_time_ = t;
                imu_samples_.push_back(ImuSample{t, acc, gyr});
                return;
            }
            if (t <= previous_imu_time_ + kTimeEps)
                return;  // getIMUInterval may replay the boundary sample
        }
        if (previous_imu_time_ >= 0.0 && t - previous_imu_time_ > 1.0)
            imu_samples_.clear();  // camera/IMU outage: require a fresh window
        previous_imu_time_ = t;
        imu_samples_.push_back(ImuSample{t, acc, gyr});

        if (acc.norm() >= config_.impact_accel)
        {
            has_impact_ = true;
            last_impact_time_ = t;
        }

        const double keep = std::max(1.0, 2.0 * config_.imu_window_seconds);
        while (!imu_samples_.empty() && imu_samples_.front().t < t - keep)
            imu_samples_.pop_front();
    }

    LandingFrameResult observe_frame(double t,
                                     const std::vector<LandingPixelObservation> &observations,
                                     const Eigen::Vector3d &velocity,
                                     const Eigen::Vector3d &ba,
                                     const Eigen::Vector3d &bg)
    {
        LandingFrameResult result;
        if (!config_.enabled)
        {
            if (active_)
            {
                active_ = false;
                result.exited = true;
                result.exit_reason = "disabled";
            }
            previous_frame_time_ = -1.0;
            previous_pixels_.clear();
            still_since_ = -1.0;
            return result;
        }
        if (!std::isfinite(t))
        {
            deactivate(result, "non-finite frame time");
            return result;
        }

        std::unordered_map<int, Eigen::Vector2d> current;
        current.reserve(observations.size());
        for (const LandingPixelObservation &observation : observations)
        {
            if (observation.id < 0 || !observation.pixel.allFinite())
                continue;
            current.emplace(observation.id, observation.pixel);
        }

        const bool has_previous = previous_frame_time_ >= 0.0;
        const double frame_gap = has_previous ? t - previous_frame_time_ : -1.0;
        if (has_previous && (!(frame_gap > 0.0) ||
                             frame_gap > config_.max_image_gap_seconds ||
                             !velocity.allFinite() || !ba.allFinite() || !bg.allFinite()))
        {
            // Clock rollback, image loss, or an invalid state: drop the
            // evidence and do not carry a zero-velocity constraint across it.
            if (frame_gap < 0.0)
            {
                const bool was_active = active_;
                reset();
                if (was_active)
                {
                    result.exited = true;
                    result.exit_reason = "clock rollback";
                }
                previous_frame_time_ = t;
                previous_pixels_ = std::move(current);
                return result;
            }
            previous_frame_time_ = t;
            previous_pixels_ = std::move(current);
            still_since_ = -1.0;
            deactivate(result, "image gap");
            return result;
        }

        if (!has_previous)
        {
            previous_frame_time_ = t;
            previous_pixels_ = std::move(current);
            still_since_ = -1.0;
            if (active_)
                deactivate(result, "missing previous frame");
            return result;
        }

        const double flow_scale = config_.flow_reference_seconds / frame_gap;
        std::vector<double> flows;
        flows.reserve(previous_pixels_.size());
        int quadrant_count[4] = {0, 0, 0, 0};
        for (const auto &entry : current)
        {
            const auto previous = previous_pixels_.find(entry.first);
            if (previous == previous_pixels_.end())
                continue;
            const double flow =
                (entry.second - previous->second).norm() * flow_scale;
            if (!std::isfinite(flow))
                continue;
            flows.push_back(flow);
            if (flow <= config_.max_flow_px)
            {
                const int qx = entry.second.x() < 0.5 * config_.image_width ? 0 : 1;
                const int qy = entry.second.y() < 0.5 * config_.image_height ? 0 : 1;
                ++quadrant_count[2 * qy + qx];
            }
        }

        result.matched_features = static_cast<int>(flows.size());
        result.still_features = 0;
        if (!flows.empty())
        {
            std::vector<double> sorted = flows;
            const std::size_t middle = sorted.size() / 2;
            std::nth_element(sorted.begin(), sorted.begin() + middle, sorted.end());
            result.median_flow_px = sorted[middle];
            for (double flow : flows)
                if (flow <= config_.max_flow_px)
                    ++result.still_features;
            result.static_fraction =
                static_cast<double>(result.still_features) / static_cast<double>(flows.size());
        }
        const int quadrant_min = std::max(1, config_.min_features / 8);
        for (int i = 0; i < 4; ++i)
            if (quadrant_count[i] >= quadrant_min)
                ++result.quadrants;

        const bool enough_visual =
            result.matched_features >= config_.min_features &&
            result.still_features >= config_.min_features &&
            result.static_fraction >= config_.min_static_fraction &&
            result.median_flow_px <= config_.max_flow_px &&
            result.quadrants >= config_.min_quadrants;
        result.visual_still = enough_visual;

        double gyro_norm = 0.0;
        double accel_dev = 0.0;
        double accel_std = 0.0;
        double gyro_std = 0.0;
        const bool imu_entry =
            evaluate_imu(t, ba, bg, false, gyro_norm, accel_dev, accel_std, gyro_std);
        const bool imu_exit =
            evaluate_imu(t, ba, bg, true, gyro_norm, accel_dev, accel_std, gyro_std);
        result.imu_still = imu_entry;
        result.gyro_norm = gyro_norm;
        result.accel_dev = accel_dev;
        result.accel_std = accel_std;
        result.gyro_std = gyro_std;

        result.speed_ok = velocity.allFinite() && velocity.norm() <= config_.max_speed;
        result.impact_recent = has_impact_ && std::isfinite(last_impact_time_) &&
                               t >= last_impact_time_ &&
                               (t - last_impact_time_) <= config_.impact_window_seconds;
        result.impact_age = has_impact_ ? t - last_impact_time_
                                        : std::numeric_limits<double>::infinity();

        if (active_)
        {
            const bool exit_visual =
                result.matched_features < std::max(10, config_.min_features / 2) ||
                result.median_flow_px > config_.exit_flow_px ||
                result.static_fraction < 0.5 * config_.min_static_fraction ||
                result.quadrants < std::max(2, config_.min_quadrants - 1);
            const bool exit_speed =
                !velocity.allFinite() || velocity.norm() > config_.exit_speed;
            if (exit_visual || !imu_exit || exit_speed)
            {
                deactivate(result, exit_visual ? "visual motion"
                                               : (!imu_exit ? "imu motion" : "speed"));
            }
            else
            {
                result.active = true;
                result.still_duration = still_since_ >= 0.0 ? t - still_since_ : 0.0;
            }
        }
        else
        {
            const bool entry = result.visual_still && imu_entry && result.impact_recent &&
                               result.speed_ok;
            if (entry)
            {
                if (still_since_ < 0.0)
                    still_since_ = t;
                result.still_duration = t - still_since_;
                constexpr double kStillTimeEps = 1e-6;
                if (result.still_duration + kStillTimeEps >= config_.min_still_seconds)
                {
                    active_ = true;
                    result.active = true;
                    result.entered = true;
                }
            }
            else
            {
                still_since_ = -1.0;
            }
        }

        previous_frame_time_ = t;
        previous_pixels_ = std::move(current);
        return result;
    }

  private:
    struct ImuSample
    {
        double t = 0.0;
        Eigen::Vector3d acc = Eigen::Vector3d::Zero();
        Eigen::Vector3d gyr = Eigen::Vector3d::Zero();
    };

    void deactivate(LandingFrameResult &result, const char *reason)
    {
        if (active_)
        {
            result.exited = true;
            result.exit_reason = reason;
        }
        active_ = false;
        still_since_ = -1.0;
    }

    bool evaluate_imu(double t, const Eigen::Vector3d &ba, const Eigen::Vector3d &bg,
                      bool exit_mode, double &gyro_norm, double &accel_dev,
                      double &accel_std, double &gyro_std) const
    {
        gyro_norm = 0.0;
        accel_dev = 0.0;
        accel_std = 0.0;
        gyro_std = 0.0;
        if (imu_samples_.empty() || !std::isfinite(t))
            return false;
        if (t - imu_samples_.back().t > config_.max_image_gap_seconds ||
            t < imu_samples_.back().t - config_.max_image_gap_seconds)
            return false;

        const double window_start = t - config_.imu_window_seconds;
        std::vector<Eigen::Vector3d> gyro;
        std::vector<double> accel_norms;
        double first_sample_time = std::numeric_limits<double>::infinity();
        double last_sample_time = -std::numeric_limits<double>::infinity();
        double previous_sample_time = -std::numeric_limits<double>::infinity();
        for (const ImuSample &sample : imu_samples_)
        {
            if (sample.t < window_start)
                continue;
            if (sample.t > t + 1e-6)
                break;
            if (std::isfinite(previous_sample_time) &&
                sample.t - previous_sample_time > config_.max_imu_sample_gap_seconds)
                return false;
            previous_sample_time = sample.t;
            first_sample_time = std::min(first_sample_time, sample.t);
            last_sample_time = std::max(last_sample_time, sample.t);
            gyro.push_back(sample.gyr - bg);
            accel_norms.push_back((sample.acc - ba).norm());
        }
        if (static_cast<int>(gyro.size()) < config_.min_imu_samples ||
            static_cast<int>(accel_norms.size()) < config_.min_imu_samples)
            return false;
        // Validate the actual selected-sample span and gaps. Counting samples
        // alone lets a short burst after a reset masquerade as a 0.5 s window.
        const double sample_span = last_sample_time - first_sample_time;
        if (!(sample_span >= 0.8 * config_.imu_window_seconds))
            return false;

        Eigen::Vector3d gyro_mean = Eigen::Vector3d::Zero();
        double accel_mean = 0.0;
        for (std::size_t i = 0; i < gyro.size(); ++i)
        {
            gyro_mean += gyro[i];
            accel_mean += accel_norms[i];
        }
        gyro_mean /= static_cast<double>(gyro.size());
        accel_mean /= static_cast<double>(accel_norms.size());
        gyro_norm = gyro_mean.norm();
        accel_dev = std::abs(accel_mean - config_.gravity);

        double gyro_var = 0.0;
        double accel_var = 0.0;
        for (std::size_t i = 0; i < gyro.size(); ++i)
        {
            gyro_var += (gyro[i] - gyro_mean).squaredNorm();
            const double d = accel_norms[i] - accel_mean;
            accel_var += d * d;
        }
        gyro_std = std::sqrt(gyro_var / static_cast<double>(gyro.size()));
        accel_std = std::sqrt(accel_var / static_cast<double>(accel_norms.size()));

        const double max_gyro = exit_mode ? config_.exit_gyro_rad_s : config_.max_gyro_rad_s;
        const double max_accel = exit_mode ? config_.exit_accel_dev : config_.max_accel_dev;
        return gyro_norm <= max_gyro && accel_dev <= max_accel &&
               gyro_std <= config_.max_gyro_std && accel_std <= config_.max_accel_std;
    }

    LandingStaticConfig config_;
    double previous_frame_time_ = -1.0;
    std::unordered_map<int, Eigen::Vector2d> previous_pixels_;
    double still_since_ = -1.0;
    bool active_ = false;
    bool has_impact_ = false;
    double last_impact_time_ = -1.0;
    double previous_imu_time_ = -1.0;
    std::deque<ImuSample> imu_samples_;
};

struct ZeroVelocityUpdateResult
{
    bool applied = false;
    double chi2 = 0.0;
    double sigma = 0.0;
    Eigen::Vector3d dv = Eigen::Vector3d::Zero();
    std::string failure_reason;
};

// Applies z = 0 - v with covariance R = sigma^2 I as a regular EKF update.
// The full covariance is updated with the Joseph form; the nominal correction
// is injected into the same error-state layout used by the visual update.
inline ZeroVelocityUpdateResult applyZeroVelocityUpdate(EkfState &state, double sigma)
{
    ZeroVelocityUpdateResult result;
    result.sigma = sigma;
    if (!state.initialized() || state.size() <= 0)
    {
        result.failure_reason = "uninitialized state";
        return result;
    }
    if (!std::isfinite(sigma) || sigma <= 0.0)
    {
        result.failure_reason = "invalid sigma";
        return result;
    }
    const int n = state.size();
    Eigen::MatrixXd &P = state.covariance();
    if (P.rows() != n || P.cols() != n || !P.allFinite() || !state.imu().v.allFinite())
    {
        result.failure_reason = "invalid covariance/state";
        return result;
    }

    const Eigen::Matrix3d R = sigma * sigma * Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d S =
        P.block<3, 3>(kImuV, kImuV) + R;
    Eigen::LDLT<Eigen::Matrix3d> ldlt(S);
    if (ldlt.info() != Eigen::Success || !(ldlt.vectorD().minCoeff() > 0.0))
    {
        result.failure_reason = "singular innovation covariance";
        return result;
    }

    const Eigen::Vector3d z = -state.imu().v;
    const Eigen::MatrixXd K = P.middleCols<3>(kImuV) * ldlt.solve(Eigen::Matrix3d::Identity());
    if (!K.allFinite())
    {
        result.failure_reason = "non-finite gain";
        return result;
    }
    const Eigen::VectorXd dx = K * z;
    result.chi2 = z.dot(ldlt.solve(z));
    if (!dx.allFinite())
    {
        result.failure_reason = "non-finite correction";
        return result;
    }

    Eigen::MatrixXd IKH = Eigen::MatrixXd::Identity(n, n);
    IKH.middleCols<3>(kImuV) -= K;
    Eigen::MatrixXd P_new = IKH * P * IKH.transpose() + K * R * K.transpose();
    if (!P_new.allFinite())
    {
        result.failure_reason = "non-finite covariance";
        return result;
    }
    EkfState::symmetrize(P_new);
    if (!EkfState::is_positive_semidefinite(P_new))
    {
        result.failure_reason = "non-PSD covariance";
        return result;
    }

    Eigen::Matrix<double, 6, 1> dpose;
    dpose.head<3>() = dx.segment<3>(kImuTheta);
    dpose.tail<3>() = dx.segment<3>(kImuP);
    EkfState::apply_pose_correction(state.imu().q, state.imu().p, dpose);
    result.dv = dx.segment<3>(kImuV);
    state.imu().v += result.dv;
    state.imu().ba += dx.segment<3>(kImuBa);
    state.imu().bg += dx.segment<3>(kImuBg);
    for (int i = 0; i < state.num_clones(); ++i)
    {
        const Eigen::Matrix<double, 6, 1> dclone = dx.segment<6>(state.clone_offset(i));
        EkfState::apply_pose_correction(state.clone(i).q, state.clone(i).p, dclone);
    }
    P = std::move(P_new);
    EkfState::symmetrize(P);
    result.applied = true;
    return result;
}

}  // namespace backend
}  // namespace vins
