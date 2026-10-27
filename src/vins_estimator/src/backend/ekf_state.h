/*******************************************************
 * ARM-oriented Schur-complement EKF backend for VINS-Fusion.
 *
 * The interface below is frozen: implementation lives in ekf_state.cpp.
 * Conventions (do not change without updating docs/BACKEND_SCHUR_EKF.md):
 *
 *  - Quaternion q = [w, x, y, z] (Hamilton product), body -> world, R = R(q).
 *  - Error state uses a *right* (body frame) perturbation:
 *        q_true = q_hat (x) dq,  dq = [1, dtheta/2]
 *    so R(q_true) ~= R(q_hat) (I + [dtheta]x).
 *  - Error-state layout of the live IMU state (15):
 *        [0..2]   dtheta
 *        [3..5]   dp
 *        [6..8]   dv
 *        [9..11]  dba
 *        [12..14] dbg
 *  - Clone blocks are appended after the live state, oldest first, 6 each:
 *        [0..2] dtheta_i, [3..5] dp_i  (poses frozen in the mean)
 *  - Gravity acts along -z of the world frame: g_w = (0, 0, -|g|).
 *******************************************************/

#pragma once

#include <eigen3/Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>

namespace vins
{
namespace backend
{

// Covariance indices of the live IMU state.
enum ImuErrorIndex
{
    kImuTheta = 0,
    kImuP = 3,
    kImuV = 6,
    kImuBa = 9,
    kImuBg = 12
};

constexpr int kImuStateSize = 15;
constexpr int kCloneStateSize = 6;

struct EkfOptions
{
    int window_size = 5;        // clones kept in the covariance (one per image frame)
    double gravity = 9.81007;   // |g| in m/s^2
    double acc_noise = 0.1;     // continuous-time accel noise density
    double gyr_noise = 0.01;    // continuous-time gyro noise density
    double acc_bias_noise = 0.001;  // accel bias random walk
    double gyr_bias_noise = 0.0001; // gyro bias random walk
    double init_position_sigma = 0.05;
    double init_velocity_sigma = 0.05;
    double init_rotation_sigma = 0.01;
    double init_acc_bias_sigma = 0.1;
    double init_gyr_bias_sigma = 0.01;
};

struct ImuNominal
{
    Eigen::Vector4d q = Eigen::Vector4d(1, 0, 0, 0);
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    Eigen::Vector3d v = Eigen::Vector3d::Zero();
    Eigen::Vector3d ba = Eigen::Vector3d::Zero();
    Eigen::Vector3d bg = Eigen::Vector3d::Zero();
};

struct CloneNominal
{
    Eigen::Vector4d q = Eigen::Vector4d(1, 0, 0, 0);
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
};

// Filter state: nominal values + covariance of the error state.
class EkfState
{
  public:
    // Resets the state to a single live IMU block (no clones) and re-initialises
    // the covariance from EkfOptions::init_*_sigma.
    void reset(const EkfOptions &options, double timestamp, const Eigen::Vector4d &q,
               const Eigen::Vector3d &p, const Eigen::Vector3d &v,
               const Eigen::Vector3d &ba, const Eigen::Vector3d &bg);

    // Copy only the live marginal and its FEJ linearisation point. Used by the
    // IMU-rate predictor: no clone window copy and no covariance reset/inflation.
    void copyLiveFrom(const EkfState &source);

    bool initialized() const { return initialized_; }
    const EkfOptions &options() const { return options_; }
    double timestamp() const { return timestamp_; }

    int size() const { return kImuStateSize + kCloneStateSize * num_clones(); }
    int num_clones() const { return static_cast<int>(clones_.size()); }
    int clone_offset(int index) const { return kImuStateSize + kCloneStateSize * index; }

    ImuNominal &imu() { return imu_; }
    const ImuNominal &imu() const { return imu_; }
    // First-estimate (FEJ) linearisation point of the live state. Propagation
    // linearises at the interval-start values and then advances imu_fej() to the
    // propagated mean, matching OpenVINS Propagator::propagate().
    ImuNominal &imu_fej() { return imu_fej_; }
    const ImuNominal &imu_fej() const { return imu_fej_; }

    CloneNominal &clone(int index) { return clones_.at(index); }
    const CloneNominal &clone(int index) const { return clones_.at(index); }
    CloneNominal &clone_fej(int index) { return clone_fej_.at(index); }
    const CloneNominal &clone_fej(int index) const { return clone_fej_.at(index); }
    double clone_timestamp(int index) const { return clone_timestamps_.at(index); }

    Eigen::MatrixXd &covariance() { return covariance_; }
    const Eigen::MatrixXd &covariance() const { return covariance_; }

    // Propagates the nominal state (RK4) and the covariance (linearised at the FEJ
    // values) by one IMU sample, then advances timestamp_ by dt.
    // acc/gyr are the raw body-frame measurements; dt > 0 seconds.
    //
    // `noise_scale` (>= 1.0) inflates the discrete process-noise covariance Qd of
    // this step by that factor. The Estimator passes > 1 while the filter has been
    // visually starved for `schur_starvation_seconds` (see
    // docs/BACKEND_SCHUR_EKF.md): the covariance is allowed to grow during a pure
    // IMU stretch so that a later visual update is not over-gated by an
    // over-confident P. Q is PSD and the factor is a positive scalar, so the
    // scaled Qd stays symmetric PSD and finite (symmetrize() runs afterwards).
    // The default 1.0 leaves every existing caller unchanged.
    void propagate(const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr, double dt,
                   double noise_scale = 1.0);

    // Appends the current IMU pose as the newest clone and grows the covariance
    // by a 6x6 block. The new clone starts perfectly correlated with the live
    // pose. This method does NOT prune the window; call prune_clone_window()
    // after the visual update.
    void augment_clone(double timestamp);

    // Hard-removes the oldest clone by deleting its covariance rows/columns.
    // This matches OpenVINS StateHelper::marginalize() and SchurVINS
    // Management(): the remaining covariance is simply P(keep, keep). No
    // conditional Schur complement (which would artificially shrink P) is used.
    void marginalize_oldest_clone();

    // Removes the oldest clones until num_clones() <= options().window_size.
    // Returns the number of clones removed.
    int prune_clone_window();

    // Hard-removes every clone block, leaving exactly the 15-dim live IMU
    // state. Each removal is a hard delete (see marginalize_oldest_clone()), so
    // the live 15x15 covariance block is preserved bit-for-bit: P is neither
    // reset nor re-inflated. This is what the starvation hold policy uses to
    // drop stale visual structure before rebuilding the window.
    // Returns the number of clones removed.
    int drop_all_clones();

    // Copies current nominal values into the live IMU FEJ copy only. Clone FEJ
    // values remain frozen at their augmentation poses.
    void refresh_imu_fej();

    // Copies current nominal values into all FEJ copies. Do not call this after
    // a visual update: it would overwrite the frozen clone linearisation points.
    void refresh_fej();

    // ---- helpers shared with the measurement update and the tests ----
    static Eigen::Matrix3d rotation(const Eigen::Vector4d &q);
    static Eigen::Matrix3d skew(const Eigen::Vector3d &v);
    static Eigen::Vector4d quaternion_multiply(const Eigen::Vector4d &a, const Eigen::Vector4d &b);
    static Eigen::Vector4d quaternion_from_axis_angle(const Eigen::Vector3d &axis_angle);
    static Eigen::Vector3d quaternion_to_axis_angle(const Eigen::Vector4d &q);
    static Eigen::Vector4d quaternion_normalize(const Eigen::Vector4d &q);
    static Eigen::Matrix4d omega_matrix(const Eigen::Vector3d &omega);  // JPL kinematics
    // Applies a 6-dim pose correction [dtheta; dp] with the right perturbation.
    static void apply_pose_correction(Eigen::Vector4d &q, Eigen::Vector3d &p,
                                      const Eigen::Matrix<double, 6, 1> &dx);
    // Makes the covariance exactly symmetric.
    static void symmetrize(Eigen::MatrixXd &P);
    // True when the symmetric part is positive definite within `tolerance`.
    static bool is_positive_definite(const Eigen::MatrixXd &P, double tolerance = 1e-12);
    // True when the symmetric part is positive semidefinite within `tolerance`.
    // A relative tolerance is used so singular augmented covariances are valid.
    static bool is_positive_semidefinite(const Eigen::MatrixXd &P, double tolerance = 1e-9);

  private:
    EkfOptions options_;
    bool initialized_ = false;
    double timestamp_ = 0.0;
    ImuNominal imu_;
    ImuNominal imu_fej_;
    std::vector<CloneNominal> clones_;
    std::vector<CloneNominal> clone_fej_;
    std::vector<double> clone_timestamps_;
    Eigen::MatrixXd covariance_;
};


// ---------------------------------------------------------------------------
// Visual-starvation monitor (P1 time-based restart + P3 noise inflation +
// P4 deep escalation and restart cooldown).
//
// Lives here, next to EkfState, because ekf_state.h is the ROS-free header that
// both Estimator::schurProcessFrame and the unit tests include; the policy is
// implemented once and tested directly (see test/test_ekf_state.cpp).
//
// Semantics (all times are sensor times in the frame header+td domain):
//   * reset(t) / note_applied_update(t) set the baseline to t.
//   * observe_failed_frame(t) is called for an image frame that did NOT apply a
//     visual update:
//       - t < baseline  -> bag replay / clock rollback: re-baseline, never
//         restart, and drop the restart-cooldown anchor (the returned Decision
//         reports rebaselined == true);
//       - gap = t - baseline > max_no_update_seconds (> 0) -> restart, unless
//         the restart cooldown is active, in which case the trigger is recorded
//         as suppressed (cooldown_suppressed == true, restart stays false);
//       - starvation becomes active once gap > starvation_seconds (> 0), which
//         is what makes Estimator::schurPropagate inflate Q;
//       - once gap > deep_starvation_seconds (> 0, and starvation is already
//         active) the multiplier is escalated to deep_noise_scale.
//   * A successful update (note_applied_update) always clears starvation.
//   * note_restart(t) records the cooldown anchor and clears starvation.
//   * reset(t) re-seeds the baseline but deliberately KEEPS the cooldown anchor
//     and the cumulative diagnostic counters (time_gate_triggers,
//     deep_escalations, cooldown_suppressions): a controlled restart runs
//     clearState() and then initSchurBackend(), both of which call reset(), and
//     the cooldown must survive that re-seed or it could never prevent a
//     restart storm.  The counters are cumulative for the whole run so the
//     end-of-run schurReportStarvation() summary still sees triggers that
//     happened before a restart re-seeded the monitor (T23).
//   * The strict ">" comparisons mean a gap exactly at the threshold does not
//     fire, and a single large gap never fires during IMU propagation because
//     observe_failed_frame() is only called once per image frame.
// ---------------------------------------------------------------------------
class StarvationMonitor
{
  public:
    struct Config
    {
        // Seconds without an applied update before a controlled restart.
        // <= 0 disables the time criterion.
        double max_no_update_seconds = 3.0;
        // Seconds without an applied update before process-noise inflation is
        // switched on. <= 0 disables inflation.
        double starvation_seconds = 0.25;
        // Q multiplier applied while starved. Defensive: a non-finite or < 1
        // value is treated as 1.0 (no inflation).
        double noise_scale = 1.0;
        // Seconds without an applied update before the multiplier is escalated
        // to deep_noise_scale. <= 0 disables the escalation. Only meaningful
        // while starvation is already active (deep is a strictly later stage).
        double deep_starvation_seconds = 1.0;
        // Q multiplier applied during deep starvation. Defensive: a non-finite
        // or < 1 value is treated as 1.0.
        double deep_noise_scale = 10.0;
        // Minimum sensor-time separation between two controlled restarts. While
        // active, both the time criterion and the frame-count backstop are
        // suppressed (they are counted, not acted on). 0 disables the cooldown.
        double restart_cooldown_seconds = 8.0;
    };

    struct Decision
    {
        bool rebaselined = false;         // backwards timestamp: baseline reset
        bool restart = false;             // time criterion fired for this frame
        bool cooldown_suppressed = false; // would have fired, blocked by cooldown
        bool deep_escalated = false;      // deep starvation started this frame
        double gap_seconds = 0.0;         // gap since the baseline (>= 0)
    };

    void configure(const Config &config) { config_ = config; }

    // (Re)seeds the monitor. The seeding frame counts as the last applied
    // update, so the first starvation window starts here. The restart-cooldown
    // anchor and the cumulative diagnostic counters (time_gate_triggers,
    // deep_escalations, cooldown_suppressions) are intentionally NOT touched
    // (see the class comment); use a backwards timestamp to drop the anchor.
    void reset(double timestamp)
    {
        last_update_time_ = timestamp;
        max_gap_seconds_ = 0.0;
        starvation_active_ = false;
        deep_active_ = false;
    }

    // Records a frame that DID apply a visual update.
    void note_applied_update(double timestamp) { rebaseline(timestamp); }

    // Re-anchors the baseline to `timestamp` without claiming that an update was
    // applied. Used by the P7 hold policy when the tracker recovers: the rebuilt
    // clone window needs a full SCHUR_MAX_NO_UPDATE_SECONDS grace period before
    // the time criterion may fire again. The elapsed gap is folded into the
    // diagnostic maximum exactly like a real inter-update gap.
    void rebaseline(double timestamp)
    {
        const double gap = timestamp - last_update_time_;
        if (gap > max_gap_seconds_)
            max_gap_seconds_ = gap;
        last_update_time_ = timestamp;
        starvation_active_ = false;
        deep_active_ = false;
    }

    // Records a controlled restart at sensor time `timestamp`. This is the only
    // way the cooldown anchor is set; reset() leaves it alone so the cooldown
    // survives the clearState()/initSchurBackend() re-seed that follows.
    void note_restart(double timestamp)
    {
        last_restart_time_ = timestamp;
        has_restart_anchor_ = true;
        starvation_active_ = false;
        deep_active_ = false;
    }

    // True while a restart is still inside the configured cooldown window.
    bool cooldown_active(double timestamp) const
    {
        return config_.restart_cooldown_seconds > 0.0 && has_restart_anchor_ &&
               (timestamp - last_restart_time_) < config_.restart_cooldown_seconds;
    }

    // Feeds a frame that did NOT apply a visual update.
    Decision observe_failed_frame(double timestamp)
    {
        Decision decision;
        const double gap = timestamp - last_update_time_;
        if (gap < 0.0)
        {
            // Bag replay / clock rollback: re-baseline instead of restarting.
            // The new baseline is "now", so the starvation window restarts from
            // zero and any active inflation is cleared until a fresh gap is
            // measured forward in time. The cooldown anchor is dropped as well:
            // it belongs to a timeline that no longer exists, and keeping it
            // would keep the cooldown permanently active during a replay.
            last_update_time_ = timestamp;
            starvation_active_ = false;
            deep_active_ = false;
            has_restart_anchor_ = false;
            decision.rebaselined = true;
            return decision;
        }
        if (gap > max_gap_seconds_)
            max_gap_seconds_ = gap;
        decision.gap_seconds = gap;

        starvation_active_ =
            config_.starvation_seconds > 0.0 && gap > config_.starvation_seconds;
        const bool deep_now = starvation_active_ &&
                              config_.deep_starvation_seconds > 0.0 &&
                              gap > config_.deep_starvation_seconds;
        decision.deep_escalated = deep_now && !deep_active_;
        if (decision.deep_escalated)
            ++deep_escalations_;
        deep_active_ = deep_now;

        if (config_.max_no_update_seconds > 0.0 &&
            gap > config_.max_no_update_seconds)
        {
            if (cooldown_active(timestamp))
            {
                // The trigger is real but a restart just happened: record it and
                // stay put instead of tearing the estimator down again.
                ++cooldown_suppressions_;
                decision.cooldown_suppressed = true;
            }
            else
            {
                ++time_gate_triggers_;
                decision.restart = true;
            }
        }
        return decision;
    }

    bool starvation_active() const { return starvation_active_; }
    bool deep_starvation_active() const { return deep_active_; }
    // Q multiplier for the current starvation stage: 1.0 when not starved,
    // noise_scale while starved, deep_noise_scale once deep starvation is on.
    // Invalid (non-finite or < 1) configured scales defensively fall back to
    // 1.0 so a bad parameter can never shrink or poison the covariance.
    double noise_scale() const
    {
        if (deep_active_)
            return sanitized_scale(config_.deep_noise_scale);
        if (starvation_active_)
            return sanitized_scale(config_.noise_scale);
        return 1.0;
    }
    double max_gap_seconds() const { return max_gap_seconds_; }
    int time_gate_triggers() const { return time_gate_triggers_; }
    int deep_escalations() const { return deep_escalations_; }
    int cooldown_suppressions() const { return cooldown_suppressions_; }
    double last_update_time() const { return last_update_time_; }
    double last_restart_time() const { return last_restart_time_; }
    // Current starvation duration for a given sensor time (clamped at 0).
    double starvation_duration(double timestamp) const
    {
        return std::max(0.0, timestamp - last_update_time_);
    }

  private:
    static double sanitized_scale(double scale)
    {
        return (std::isfinite(scale) && scale >= 1.0) ? scale : 1.0;
    }

    Config config_;
    double last_update_time_ = 0.0;
    double max_gap_seconds_ = 0.0;
    double last_restart_time_ = 0.0;
    int time_gate_triggers_ = 0;
    int deep_escalations_ = 0;
    int cooldown_suppressions_ = 0;
    bool starvation_active_ = false;
    bool deep_active_ = false;
    bool has_restart_anchor_ = false;
};

// ---------------------------------------------------------------------------
// P7 hold policy ("do not restart, just hold").
//
// A starvation-triggered restart on ARM costs a multi-second Ceres
// re-initialisation with no odometry, and P5 (re-inflate P and immediately
// absorb the degraded observation) was measured to be worse than the starvation
// itself. The hold policy is deliberately different from both:
//   * the time/frame-count criterion does NOT call clearState(), does NOT
//     re-seed the filter and does NOT touch the covariance: the live IMU state
//     keeps propagating with the nominal process noise and keeps being
//     published every frame;
//   * visual updates and clone augmentation stop, so a stale tracker cannot
//     corrupt the filter (the clone blocks are hard-deleted, the live 15x15 P
//     block is preserved);
//   * once the tracker reports enough observations for enough consecutive
//     frames, hold is left and the clone window is rebuilt frame by frame.
//
// This class owns only the decision logic (ROS-free, unit tested); the Estimator
// performs the EkfState mutation.
// ---------------------------------------------------------------------------
class StarvationHold
{
  public:
    struct Config
    {
        bool enabled = false;
        // Minimum camera-0 observations (tracks) a frame must carry to count
        // towards recovery. <= 0 accepts any frame.
        int recovery_min_features = 12;
        // Consecutive recovered frames required before leaving hold. Values
        // below 1 are treated as 1.
        int recovery_frames = 2;
    };

    enum class Action
    {
        kNone = 0,     // not holding
        kHold = 1,     // still holding: this frame must skip the visual update
        kRecover = 2,  // leave hold and rebuild the clone window from this frame
    };

    void configure(const Config &config) { config_ = config; }
    bool enabled() const { return config_.enabled; }
    bool active() const { return active_; }

    // Enters hold at sensor time `timestamp`. Returns false when already
    // holding. Cumulative diagnostics survive a later full restart.
    bool enter(double timestamp)
    {
        if (active_)
            return false;
        active_ = true;
        start_time_ = timestamp;
        last_time_ = timestamp;
        good_streak_ = 0;
        ++episodes_;
        return true;
    }

    // Feeds one holding frame and reports whether the window may be rebuilt.
    Action observe_frame(double timestamp, int observations)
    {
        if (!active_)
            return Action::kNone;
        ++frames_;
        last_time_ = timestamp;
        const bool enough = config_.recovery_min_features <= 0 ||
                            observations >= config_.recovery_min_features;
        good_streak_ = enough ? good_streak_ + 1 : 0;
        if (good_streak_ >= std::max(1, config_.recovery_frames))
            return Action::kRecover;
        return Action::kHold;
    }

    // Leaves hold and returns the hold duration in seconds. Idempotent.
    double leave(double timestamp)
    {
        if (!active_)
            return 0.0;
        const double duration = std::max(0.0, timestamp - start_time_);
        active_ = false;
        good_streak_ = 0;
        last_time_ = timestamp;
        last_duration_ = duration;
        total_duration_ += duration;
        if (duration > max_duration_)
            max_duration_ = duration;
        return duration;
    }

    // Drops the holding state without recording a duration. Used when the
    // filter is re-seeded by a full restart.
    void deactivate()
    {
        active_ = false;
        good_streak_ = 0;
        start_time_ = -1.0;
        last_time_ = -1.0;
    }

    int episodes() const { return episodes_; }
    int frames() const { return frames_; }
    int good_streak() const { return good_streak_; }
    double start_time() const { return start_time_; }
    double last_duration() const { return last_duration_; }
    double total_duration() const { return total_duration_; }
    double max_duration() const { return max_duration_; }
    double current_duration(double timestamp) const
    {
        return active_ ? std::max(0.0, timestamp - start_time_) : 0.0;
    }

  private:
    Config config_;
    bool active_ = false;
    double start_time_ = -1.0;
    double last_time_ = -1.0;
    double total_duration_ = 0.0;
    double max_duration_ = 0.0;
    double last_duration_ = 0.0;
    int episodes_ = 0;
    int frames_ = 0;
    int good_streak_ = 0;
};

// P5 soft-restart escalation predicate, shared by Estimator::schurProcessFrame
// and the unit tests (T24) so the rule is ROS-free and directly testable.
// `soft_restart_time` is the sensor time of the pending (unrecovered) soft
// restart, or a negative sentinel when none is pending.  Returns true once
// `frame_time` is strictly more than `max_seconds` past that anchor, i.e. the
// soft restart has not been followed by an applied visual update for too long.
// `max_seconds <= 0` disables the escalation, and a backwards timestamp (bag
// replay / clock rollback) never escalates because the stall is negative.
inline bool softRestartEscalationDue(double frame_time, double soft_restart_time,
                                     double max_seconds)
{
    if (!(max_seconds > 0.0) || !(soft_restart_time >= 0.0))
        return false;
    return (frame_time - soft_restart_time) > max_seconds;
}

// P6 starvation-adaptive weak update ("accept more, weight less"), shared by
// Estimator::schurProcessFrame() and the unit tests (T27) so the ramp is
// ROS-free and directly testable.  `gap_seconds` is the sensor-time gap since
// the last applied visual update (StarvationMonitor::starvation_duration).
//
// The returned factor scales *this frame's* visual update only:
//   * the chi2 gate threshold becomes chi2_multiplier * scale (the gate
//     statistic itself still uses the nominal sigma), and
//   * the observation weight is divided by scale (sigma -> sigma * scale, i.e.
//     R * scale^2 / information / scale^2), and
//   * the acceptance floors are relaxed (min_features / scale, floored, with a
//     hard lower bound applied by the caller).
//
// The ramp reuses the existing starvation parameters (no new ramp knobs):
//   scale = 1.0                                            , gap <= starvation
//   scale = 1 + (max_scale-1) * (gap-starvation)/(deep-starvation)
//                                                          , starvation < gap < deep
//   scale = max_scale                                      , gap >= deep
// and is clamped to [1.0, max_scale].  `deep_seconds <= starvation_seconds`
// collapses the ramp to a step (gap > starvation_seconds => max_scale).  A
// disabled flag, non-finite parameters, max_scale < 1, or a negative/NaN gap all
// return 1.0, so the default configuration is a strict no-op.
inline double weakUpdateScale(double gap_seconds, bool enabled,
                              double starvation_seconds, double deep_seconds,
                              double max_scale)
{
    if (!enabled)
        return 1.0;
    if (!std::isfinite(starvation_seconds) || !std::isfinite(deep_seconds) ||
        !std::isfinite(max_scale) || max_scale < 1.0)
        return 1.0;
    // Strictly greater than starvation_seconds (also rejects gap < 0 and NaN).
    if (!(gap_seconds > starvation_seconds))
        return 1.0;
    // Past deep_seconds (including the degenerate deep <= starvation step) the
    // scale saturates at max_scale.
    if (gap_seconds >= deep_seconds)
        return max_scale;
    const double denom = deep_seconds - starvation_seconds;
    if (!(denom > 0.0))
        return max_scale;  // defensive: step instead of dividing by <= 0
    const double frac = (gap_seconds - starvation_seconds) / denom;
    const double scale = 1.0 + (max_scale - 1.0) * frac;
    return std::min(max_scale, std::max(1.0, scale));
}

// What a starvation restart trigger should do this frame.
enum class StarvationRestartAction
{
    kNone = 0,  // no restart this frame
    kSoft = 1,  // P5 soft restart: keep the IMU/EKF state, drop the visual state
    kFull = 2,  // historical full clearState() + Ceres re-initialisation restart
    kHold = 3,  // P7 hold: keep propagating/publishing, drop visual updates only
};

// P5b frame-count-backstop policy, shared by Estimator::schurProcessFrame() and
// the unit tests (T26) so the rule is ROS-free and directly testable.  It
// mirrors the gate around SCHUR_MAX_CONSECUTIVE_FAILURES: the backstop is only
// considered when the time criterion did not already restart this frame and the
// P4 restart cooldown is inactive, and it fires once the failure streak reaches
// the configured limit.  In soft mode (SCHUR_STARVATION_SOFT_RESTART while the
// Schur EKF is active) the backstop takes the soft path instead of the full
// clearState() restart; otherwise the historical full restart is kept verbatim.
// The soft-restart escalation anchor is untouched here (schurSoftRestart() only
// arms it when none is pending), so SCHUR_SOFT_RESTART_MAX_SECONDS still
// escalates to a full restart.
// In P7 hold mode (hold_mode) the backstop never restarts either: it enters the
// hold state, which keeps publishing pure-IMU propagation. hold_mode takes
// precedence over soft_mode when both are enabled.
inline StarvationRestartAction frameCountBackstopAction(bool already_restarted,
                                                        bool cooldown_active,
                                                        int consecutive_failures,
                                                        int max_consecutive_failures,
                                                        bool soft_mode,
                                                        bool hold_mode = false)
{
    if (already_restarted || cooldown_active ||
        consecutive_failures < max_consecutive_failures)
        return StarvationRestartAction::kNone;
    if (hold_mode)
        return StarvationRestartAction::kHold;
    return soft_mode ? StarvationRestartAction::kSoft : StarvationRestartAction::kFull;
}

}  // namespace backend
}  // namespace vins
