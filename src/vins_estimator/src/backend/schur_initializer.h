/*******************************************************
 * ROS-free initialization helpers for the Schur EKF backend.
 *
 * The initializer deliberately never calls Ceres and never depends on the
 * sliding-window state arrays.  It supports two independent paths:
 *
 *   * static: an IMU window with low acceleration/gyro variance plus a visual
 *     low-flow gate.  The initial roll/pitch comes from the mean measured
 *     specific force, the gyro bias from the mean gyro, and v = 0.
 *   * dynamic stereo: visual body poses from stereo triangulation + PnP are
 *     aligned against IMU preintegration by a linear gravity/velocity solve.
 *     The solve is Ceres-free and the same equations as VINS init alignment.
 *
 * All outputs use the Schur EKF world convention: gravity is (0, 0, -|g|),
 * q is body -> world in [w, x, y, z] order, and p/v are expressed in the
 * world frame.  A failed attempt is reported explicitly; it is never silently
 * converted into a static initialization.
 *******************************************************/

#pragma once

#include <deque>
#include <eigen3/Eigen/Dense>
#include <string>
#include <vector>

namespace vins
{
namespace backend
{

enum class SchurInitResultKind_e : int {
  K_WAITING = 0,  // not enough data / not yet excited
  K_STATIC = 1,   // initialized from a verified static window
  K_DYNAMIC = 2,  // initialized from stereo visual + IMU alignment
  K_REJECTED = 3  // enough data, but a consistency gate rejected it
};

struct SchurInitConfig_s {
  bool allow_static_ = true;
  bool allow_dynamic_ = true;
  // A monocular static seed has no metric scale/translation observability.
  // It is disabled by default and must be opted into explicitly.
  bool allow_mono_static_ = false;

  double gravity_ = 9.81007;
  double static_window_seconds_ = 1.5;
  int static_min_samples_ = 30;
  double static_max_imu_gap_seconds_ = 0.05;
  double static_accel_std_ = 0.8;  // sqrt(sum_i var(a_i)); MH01/02 fail
  double static_gyro_std_ = 0.08;
  double static_accel_norm_tol_ = 0.6;
  double static_gyro_mean_max_ = 0.30;
  double static_ba_norm_max_ = 0.50;

  // Visual motion gate. Check the maximum median pixel displacement over
  // each newer frame relative to the oldest frame in the static window.
  double static_flow_max_px_ = 1.5;
  int static_visual_min_matches_ = 20;
  int static_visual_min_quadrants_ = 2;
  double static_visual_min_span_seconds_ = 0.20;
  int static_image_width_ = 752;
  int static_image_height_ = 480;

  int dynamic_min_frames_ = 6;
  double dynamic_min_rotation_deg_ = 3.0;
  double dynamic_min_translation_m_ = 0.05;
  double dynamic_gravity_tol_ = 0.5;
  double dynamic_max_gyro_bias_norm_ = 0.3;
  double dynamic_max_position_residual_m_ = 0.25;
  double dynamic_max_velocity_residual_mps_ = 0.50;
  double dynamic_max_rotation_residual_deg_ = 5.0;
  int dynamic_gravity_refine_iterations_ = 4;
};

struct SchurInitImuSample_s {
  double timestamp_ = 0.0;
  Eigen::Vector3d acc_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyr_ = Eigen::Vector3d::Zero();
};

struct SchurInitVisualObservation_s {
  int feature_id_ = -1;
  int camera_id_ = 0;
  Eigen::Vector2d pixel_uv_ = Eigen::Vector2d::Zero();
};

struct SchurInitVisualFrame_s {
  double timestamp_ = 0.0;
  std::vector<SchurInitVisualObservation_s> observations_;
};

// One adjacent image pair with its IMU preintegration summary.  `delta_*`
// must already be bias-corrected relative to the supplied bias guess.
struct SchurInitMotionFrame_s {
  double t0_ = 0.0;
  double t1_ = 0.0;
  Eigen::Matrix3d R0_ = Eigen::Matrix3d::Identity();  // body -> visual world
  Eigen::Matrix3d R1_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d p0_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d p1_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d delta_p_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond delta_q_ = Eigen::Quaterniond::Identity();
  Eigen::Vector3d delta_v_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d dq_dbg_ = Eigen::Matrix3d::Zero();
  Eigen::Vector3d p_ci_ = Eigen::Vector3d::Zero();  // body_T_cam0 translation
};

struct SchurInitSeed_s {
  SchurInitResultKind_e kind_ = SchurInitResultKind_e::K_WAITING;
  Eigen::Vector4d q_ = Eigen::Vector4d(1, 0, 0, 0);
  Eigen::Vector3d p_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d ba_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d bg_ = Eigen::Vector3d::Zero();
  // Dynamic-only diagnostics.  Static seeds leave world_rotation_ identity.
  Eigen::Vector3d gravity_visual_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d world_rotation_ = Eigen::Matrix3d::Identity();
  double scale_ = 1.0;
  // Per-frame states for the active visual window, oldest first.  Static
  // seeds provide one entry at the seed pose; dynamic seeds provide every
  // aligned frame.  `index` is the caller's active frame index.
  struct FrameState_s {
    int index_ = -1;
    Eigen::Vector4d q_ = Eigen::Vector4d(1, 0, 0, 0);
    Eigen::Vector3d p_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d v_ = Eigen::Vector3d::Zero();
  };
  std::vector<FrameState_s> frame_states_;
  std::string reason_;
};

class SchurInitializer {
public:
  explicit SchurInitializer(const SchurInitConfig_s & _config = SchurInitConfig_s());

  void configure(const SchurInitConfig_s & _config);
  const SchurInitConfig_s & config() const { return config_; }
  void reset();

  // Strictly monotonic IMU ingestion.  Non-finite or non-increasing samples
  // are dropped; a gap larger than the configured limit invalidates the
  // current static candidate (the next attempt waits for a clean window).
  void addImuSample(double _timestamp, const Eigen::Vector3d & _acc, const Eigen::Vector3d & _gyr);
  void addVisualFrame(double _timestamp, const std::vector<SchurInitVisualObservation_s> & _observations);

  // Attempts static first, then dynamic.  `motion_frames` are the active
  // stereo alignment pairs, oldest first.  `bg_guess`/`ba_guess` are the
  // biases used to preintegrate those pairs.
  SchurInitSeed_s attempt(
    double _now, bool _stereo, bool _has_imu, const std::vector<SchurInitMotionFrame_s> & _motion_frames,
    const Eigen::Vector3d & _bg_guess = Eigen::Vector3d::Zero(),
    const Eigen::Vector3d & _ba_guess = Eigen::Vector3d::Zero());

  // Pure, independently testable dynamic alignment.  Returns false and sets
  // `_reason` when a gate fails.  The input frames must be consistent with
  // VINS' visual/IMU convention (g is the positive-down gravity vector).
  static bool alignStereoMotion(
    const std::vector<SchurInitMotionFrame_s> & _frames, const SchurInitConfig_s & _config,
    const Eigen::Vector3d & _bg_guess, const Eigen::Vector3d & _ba_guess, SchurInitSeed_s & _seed,
    std::string & _reason);

  // Returns the gyro-bias correction and residual excitation statistics used
  // by alignStereoMotion().  Also used by the tests.
  static bool solveGyroBias(
    const std::vector<SchurInitMotionFrame_s> & _frames, Eigen::Vector3d & _delta_bg, std::string & _reason);

  // Static window diagnostics.  These are public so estimator logging and
  // tests can explain why an attempt is waiting rather than guessing.
  struct StaticImuStats_s {
    bool valid_ = false;
    int samples_ = 0;
    double span_seconds_ = 0.0;
    double max_gap_seconds_ = 0.0;
    Eigen::Vector3d mean_acc_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d mean_gyr_ = Eigen::Vector3d::Zero();
    double accel_std_ = 0.0;
    double gyro_std_ = 0.0;
    double accel_norm_ = 0.0;
    std::string reason_;
  };
  StaticImuStats_s staticImuStats(double _now) const;

  struct VisualStaticStats_s {
    bool valid_ = false;
    int matches_ = 0;
    int quadrants_ = 0;
    double span_seconds_ = 0.0;
    double median_flow_px_ = 0.0;
    std::string reason_;
  };
  VisualStaticStats_s visualStaticStats(double _now) const;

  // Last reason populated by attempt().  Useful for throttled logs.
  const std::string & lastReason() const { return last_reason_; }

private:
  void prune(double _now);
  bool tryStatic(double _now, bool _stereo, SchurInitSeed_s & _seed) const;

  SchurInitConfig_s config_;
  std::deque<SchurInitImuSample_s> imu_;
  std::deque<SchurInitVisualFrame_s> visual_;
  double last_imu_time_ = -1.0;
  double last_visual_time_ = -1.0;
  bool imu_gap_invalidated_ = false;
  std::string last_reason_;
};

}  // namespace backend
}  // namespace vins
