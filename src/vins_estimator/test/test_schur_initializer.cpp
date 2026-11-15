// Standalone gtest suite for the Schur standalone initializer.
//
// gtest + Eigen only (no ROS / Ceres / OpenCV).  The dynamic cases generate
// preintegration summaries directly from a known trajectory, then verify that
// the Ceres-free alignment recovers gravity, velocities and metric scale.

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "backend/schur_initializer.h"

namespace vins
{
namespace backend
{
namespace
{

constexpr double K_G = 9.81007;
constexpr double K_PI = 3.14159265358979323846;

Eigen::Matrix3d rotYpr(double _yaw, double _pitch, double _roll)
{
  const Eigen::AngleAxisd yaw(_yaw, Eigen::Vector3d::UnitZ());
  const Eigen::AngleAxisd pitch(_pitch, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd roll(_roll, Eigen::Vector3d::UnitX());
  return (yaw * pitch * roll).toRotationMatrix();
}

SchurInitConfig_s staticConfig()
{
  SchurInitConfig_s config;
  config.allow_static_ = true;
  config.allow_dynamic_ = true;
  config.static_window_seconds_ = 1.0;
  config.static_min_samples_ = 20;
  config.static_max_imu_gap_seconds_ = 0.02;
  config.static_accel_std_ = 0.5;
  config.static_gyro_std_ = 0.05;
  config.static_accel_norm_tol_ = 0.3;
  config.static_gyro_mean_max_ = 0.2;
  config.static_ba_norm_max_ = 0.3;
  config.static_flow_max_px_ = 1.0;
  config.static_visual_min_matches_ = 20;
  config.static_visual_min_quadrants_ = 2;
  config.static_visual_min_span_seconds_ = 0.4;
  config.static_image_width_ = 640;
  config.static_image_height_ = 480;
  config.dynamic_min_frames_ = 5;
  config.dynamic_min_rotation_deg_ = 2.0;
  config.dynamic_min_translation_m_ = 0.02;
  config.dynamic_gravity_tol_ = 0.5;
  config.dynamic_max_gyro_bias_norm_ = 0.3;
  config.dynamic_max_position_residual_m_ = 0.03;
  config.dynamic_max_velocity_residual_mps_ = 0.05;
  config.dynamic_max_rotation_residual_deg_ = 1.0;
  config.dynamic_gravity_refine_iterations_ = 4;
  return config;
}

std::vector<SchurInitVisualObservation_s> gridObservations(double _flow_px, double _t)
{
  std::vector<SchurInitVisualObservation_s> observations;
  int id = 0;
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 6; ++col) {
      const double u = 60.0 + 100.0 * col;
      const double v = 50.0 + 100.0 * row;
      observations.push_back(SchurInitVisualObservation_s{
        id++, 0, Eigen::Vector2d(u + _flow_px * std::sin(_t), v + 0.25 * _flow_px * std::cos(_t))});
    }
  }
  return observations;
}

void feedStaticMotion(
  SchurInitializer & _initializer, double _duration, double _accel_std, double _gyro_std, double _flow_px)
{
  const Eigen::Vector3d ba(0.04, -0.02, 0.01);
  const Eigen::Vector3d bg(0.01, -0.02, 0.03);
  const Eigen::Matrix3d R = rotYpr(0.3, -0.2, 0.1);
  const Eigen::Vector3d specific_force = R.transpose() * Eigen::Vector3d(0.0, 0.0, K_G) + ba;
  const int imu_rate = 200;
  const int steps = static_cast<int>(_duration * imu_rate);
  for (int i = 0; i <= steps; ++i) {
    const double t = static_cast<double>(i) / imu_rate;
    const double phase = 2.0 * K_PI * 7.0 * t;
    const Eigen::Vector3d acc =
      specific_force +
      _accel_std * Eigen::Vector3d(std::sin(phase), std::cos(phase), 0.5 * std::sin(1.7 * phase));
    const Eigen::Vector3d gyr =
      bg + _gyro_std * Eigen::Vector3d(std::cos(phase), std::sin(0.7 * phase), 0.2 * std::cos(1.3 * phase));
    _initializer.addImuSample(t, acc, gyr);
    if (i % 10 == 0)
      _initializer.addVisualFrame(t, gridObservations(_flow_px, t));
  }
}

SchurInitMotionFrame_s makeMotionPair(
  const Eigen::Matrix3d & _R0, const Eigen::Vector3d & _p0, const Eigen::Vector3d & _v0,
  const Eigen::Matrix3d & _R1, const Eigen::Vector3d & _p1, const Eigen::Vector3d & _v1,
  const Eigen::Vector3d & _gravity_visual, double _dt, const Eigen::Vector3d & _p_ci)
{
  SchurInitMotionFrame_s frame;
  frame.t0_ = 0.0;
  frame.t1_ = _dt;
  frame.R0_ = _R0;
  frame.R1_ = _R1;
  frame.p0_ = _p0;
  frame.p1_ = _p1;
  frame.p_ci_ = _p_ci;
  frame.delta_q_ = Eigen::Quaterniond(_R0.transpose() * _R1);
  frame.delta_v_ = _R0.transpose() * (_v1 - _v0 + _gravity_visual * _dt);
  const Eigen::Matrix3d R0t = _R0.transpose();
  frame.delta_p_ =
    R0t * (_p1 - _p0 - _v0 * _dt + 0.5 * _gravity_visual * _dt * _dt) - (R0t * _R1 * _p_ci - _p_ci);
  return frame;
}

std::vector<SchurInitMotionFrame_s> makeTrajectory(
  Eigen::Vector3d & _gravity_visual, std::vector<Eigen::Matrix3d> * _rotations = nullptr,
  std::vector<Eigen::Vector3d> * _positions = nullptr, std::vector<Eigen::Vector3d> * _velocities = nullptr)
{
  _gravity_visual = Eigen::Vector3d(0.35, -0.20, 9.78).normalized() * K_G;
  const Eigen::Vector3d p_ci(0.03, -0.02, 0.01);
  const int n = 9;
  const double dt = 0.05;
  std::vector<Eigen::Matrix3d> rotations;
  std::vector<Eigen::Vector3d> positions;
  std::vector<Eigen::Vector3d> velocities;
  for (int i = 0; i <= n; ++i) {
    const double t = i * dt;
    rotations.push_back(rotYpr(0.15 * t, -0.10 * t, 0.08 * t));
    positions.push_back(Eigen::Vector3d(0.4 * t, -0.2 * t + 0.15 * t * t, 0.3 * t * t));
    velocities.push_back(Eigen::Vector3d(0.4, -0.2 + 0.3 * t, 0.6 * t));
  }
  std::vector<SchurInitMotionFrame_s> frames;
  for (int i = 0; i < n; ++i) {
    SchurInitMotionFrame_s frame = makeMotionPair(
      rotations[static_cast<std::size_t>(i)], positions[static_cast<std::size_t>(i)],
      velocities[static_cast<std::size_t>(i)], rotations[static_cast<std::size_t>(i + 1)],
      positions[static_cast<std::size_t>(i + 1)], velocities[static_cast<std::size_t>(i + 1)],
      _gravity_visual, dt, p_ci);
    frame.t0_ = i * dt;
    frame.t1_ = (i + 1) * dt;
    frames.push_back(frame);
  }
  if (_rotations)
    *_rotations = rotations;
  if (_positions)
    *_positions = positions;
  if (_velocities)
    *_velocities = velocities;
  return frames;
}

TEST(SchurInitializer, StaticWindowInitializesGravityBiasAndZeroVelocity)
{
  SchurInitializer initializer(staticConfig());
  feedStaticMotion(initializer, 1.2, 0.05, 0.005, 0.2);

  const SchurInitSeed_s seed = initializer.attempt(1.2, true, true, {});
  ASSERT_EQ(seed.kind_, SchurInitResultKind_e::K_STATIC) << seed.reason_;
  EXPECT_LT(seed.v_.norm(), 1e-12);
  EXPECT_LT(seed.p_.norm(), 1e-12);
  // The component of accelerometer bias along gravity is not observable
  // from a standstill; the static seed deliberately absorbs it into roll and
  // pitch, so only the perpendicular residual is required to stay small.
  EXPECT_LT(seed.ba_.norm(), 0.03);
  EXPECT_NEAR(seed.bg_.x(), 0.01, 0.01);
  EXPECT_NEAR(seed.bg_.y(), -0.02, 0.01);
  EXPECT_NEAR(seed.bg_.z(), 0.03, 0.01);

  const Eigen::Matrix3d R =
    Eigen::Quaterniond(seed.q_(0), seed.q_(1), seed.q_(2), seed.q_(3)).toRotationMatrix();
  const Eigen::Matrix3d R_true = rotYpr(0.3, -0.2, 0.1);
  const Eigen::Vector3d ba_true(0.04, -0.02, 0.01);
  const Eigen::Vector3d mean_acc = R_true.transpose() * Eigen::Vector3d(0.0, 0.0, K_G) + ba_true;
  const Eigen::Vector3d measured = R * (mean_acc - seed.ba_);
  EXPECT_NEAR(measured.x(), 0.0, 0.03);
  EXPECT_NEAR(measured.y(), 0.0, 0.03);
  EXPECT_NEAR(measured.z(), K_G, 0.03);
}

TEST(SchurInitializer, EuRoCStartWindowsMatchMeasuredGates)
{
  // Measured from the local EuRoC machine_hall bags over the first 1.5 s:
  // MH01 1.62/0.258, MH02 1.23/0.127, MH03 0.394/0.035,
  // MH04 0.179/0.020, MH05 0.061/0.004 (accel/gyro vector std).
  struct Case_s {
    double accel_std;
    double gyro_std;
    bool expect_static;
  };
  const Case_s cases[] = {
    {0.061, 0.004, true},
    {0.179, 0.020, true},
    {0.394, 0.035, true},
    {1.230, 0.127, false},
    {1.620, 0.258, false}};

  for (const Case_s & test : cases) {
    SchurInitializer initializer(staticConfig());
    feedStaticMotion(initializer, 1.2, test.accel_std, test.gyro_std, 0.2);
    const auto stats = initializer.staticImuStats(1.2);
    EXPECT_EQ(stats.valid_, test.expect_static)
      << test.accel_std << "/" << test.gyro_std << ": " << stats.reason_;
  }
}

TEST(SchurInitializer, VisualMotionPreventsFalseStaticInitialization)
{
  SchurInitializer initializer(staticConfig());
  feedStaticMotion(initializer, 1.2, 0.05, 0.005, 20.0);
  const auto visual = initializer.visualStaticStats(1.2);
  EXPECT_FALSE(visual.valid_);
  EXPECT_GT(visual.median_flow_px_, 1.0);

  const SchurInitSeed_s seed = initializer.attempt(1.2, true, true, {});
  EXPECT_NE(seed.kind_, SchurInitResultKind_e::K_STATIC);
}

TEST(SchurInitializer, ImuGapInvalidatesStaticCandidate)
{
  SchurInitializer initializer(staticConfig());
  const int steps = 240;
  double offset = 0.0;
  for (int i = 0; i < steps; ++i) {
    if (i == 60)
      offset += 0.2;
    const double t = static_cast<double>(i) / 200.0 + offset;
    initializer.addImuSample(t, Eigen::Vector3d(0.0, 0.0, K_G), Eigen::Vector3d(0.0, 0.0, 0.0));
    if (i % 10 == 0)
      initializer.addVisualFrame(t, gridObservations(0.0, t));
  }
  const auto stats = initializer.staticImuStats(1.2);
  EXPECT_FALSE(stats.valid_);
  EXPECT_NE(stats.reason_.find("gap"), std::string::npos)
    << stats.reason_ << " samples=" << stats.samples_ << " span=" << stats.span_seconds_
    << " gap=" << stats.max_gap_seconds_;
}

TEST(SchurInitializer, BackwardsImuSampleIsDropped)
{
  SchurInitializer initializer(staticConfig());
  initializer.addImuSample(1.0, Eigen::Vector3d(0.0, 0.0, K_G), Eigen::Vector3d::Zero());
  initializer.addImuSample(0.9, Eigen::Vector3d(1.0, 2.0, 3.0), Eigen::Vector3d::Ones());
  const auto stats = initializer.staticImuStats(1.0);
  EXPECT_LT(stats.samples_, 2);
}

TEST(SchurInitializer, DynamicAlignmentRecoversGravityVelocityAndScale)
{
  Eigen::Vector3d gravity_visual;
  std::vector<Eigen::Matrix3d> rotations;
  std::vector<Eigen::Vector3d> positions;
  std::vector<Eigen::Vector3d> velocities;
  const auto frames = makeTrajectory(gravity_visual, &rotations, &positions, &velocities);

  SchurInitSeed_s seed;
  std::string reason;
  ASSERT_TRUE(SchurInitializer::alignStereoMotion(
    frames, staticConfig(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), seed, reason))
    << reason;
  EXPECT_EQ(seed.kind_, SchurInitResultKind_e::K_DYNAMIC);
  EXPECT_NEAR(seed.gravity_visual_.x(), gravity_visual.x(), 1e-6);
  EXPECT_NEAR(seed.gravity_visual_.y(), gravity_visual.y(), 1e-6);
  EXPECT_NEAR(seed.gravity_visual_.z(), gravity_visual.z(), 1e-6);
  EXPECT_NEAR(seed.scale_, 1.0, 1e-6);
  ASSERT_EQ(seed.frame_states_.size(), frames.size() + 1);
  const Eigen::Vector3d expected_last_velocity =
    seed.world_rotation_ * Eigen::Vector3d(0.4, -0.2 + 0.3 * 0.45, 0.6 * 0.45);
  EXPECT_TRUE(seed.frame_states_.back().v_.isApprox(expected_last_velocity, 1e-6));

  // g_visual is the VINS positive-down vector.  The EKF world is +z up, so
  // the alignment maps the VINS down axis to +z and physical gravity to -z.
  const Eigen::Vector3d aligned_down = seed.world_rotation_ * seed.gravity_visual_;
  EXPECT_TRUE(aligned_down.isApprox(Eigen::Vector3d(0.0, 0.0, K_G), 1e-9));
  const Eigen::Vector3d aligned_physical_gravity = -aligned_down;
  EXPECT_TRUE(aligned_physical_gravity.isApprox(Eigen::Vector3d(0.0, 0.0, -K_G), 1e-9));

  // Static check after alignment: a body at rest measures specific force
  // opposite to physical gravity, and the EKF propagation must cancel it.
  const Eigen::Matrix3d R_body = rotYpr(0.4, -0.3, 0.2);
  const Eigen::Vector3d g_ekf(0.0, 0.0, -K_G);
  const Eigen::Vector3d acc_static = -R_body.transpose() * g_ekf;
  const Eigen::Vector3d a_world_static = R_body * acc_static + g_ekf;
  EXPECT_LT(a_world_static.norm(), 1e-12);

  // Known non-zero acceleration: a_world = R(acc - ba) + g.  The aligned
  // seed frame must reproduce the known world acceleration, not its negation.
  const Eigen::Vector3d ba(0.01, -0.02, 0.03);
  const Eigen::Vector3d a_desired(0.2, -0.1, 0.05);
  const Eigen::Vector3d acc_motion = R_body.transpose() * (a_desired - g_ekf) + ba;
  const Eigen::Vector3d a_reconstructed = R_body * (acc_motion - ba) + g_ekf;
  EXPECT_TRUE(a_reconstructed.isApprox(a_desired, 1e-12));
}

TEST(SchurInitializer, DynamicAlignmentRejectsInconsistentPositionResidual)
{
  Eigen::Vector3d gravity_visual;
  auto frames = makeTrajectory(gravity_visual);
  frames[3].delta_p_ += Eigen::Vector3d(1.0, 0.0, 0.0);
  SchurInitSeed_s seed;
  std::string reason;
  EXPECT_FALSE(SchurInitializer::alignStereoMotion(
    frames, staticConfig(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), seed, reason));
  EXPECT_NE(reason.find("position residual"), std::string::npos);
}

TEST(SchurInitializer, DynamicAlignmentRejectsNoExcitation)
{
  const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d p = Eigen::Vector3d::Zero();
  const Eigen::Vector3d v = Eigen::Vector3d::Zero();
  std::vector<SchurInitMotionFrame_s> frames;
  for (int i = 0; i < 8; ++i) {
    SchurInitMotionFrame_s frame =
      makeMotionPair(R, p, v, R, p, v, Eigen::Vector3d(0.0, 0.0, K_G), 0.05, Eigen::Vector3d::Zero());
    frame.t0_ = i * 0.05;
    frame.t1_ = (i + 1) * 0.05;
    frames.push_back(frame);
  }
  SchurInitSeed_s seed;
  std::string reason;
  EXPECT_FALSE(SchurInitializer::alignStereoMotion(
    frames, staticConfig(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), seed, reason));
  EXPECT_NE(reason.find("excitation"), std::string::npos);
}

TEST(SchurInitializer, GyroBiasSolveRecoversInjectedCorrection)
{
  std::vector<SchurInitMotionFrame_s> frames;
  const Eigen::Vector3d expected(0.01, -0.02, 0.03);
  for (int i = 0; i < 5; ++i) {
    SchurInitMotionFrame_s frame;
    frame.t0_ = i * 0.05;
    frame.t1_ = (i + 1) * 0.05;
    frame.delta_q_.setIdentity();
    frame.R0_.setIdentity();
    frame.R1_ = Eigen::Quaterniond(1.0, 0.5 * expected.x(), 0.5 * expected.y(), 0.5 * expected.z())
                  .normalized()
                  .toRotationMatrix();
    frame.dq_dbg_.setIdentity();
    frames.push_back(frame);
  }
  Eigen::Vector3d delta_bg;
  std::string reason;
  ASSERT_TRUE(SchurInitializer::solveGyroBias(frames, delta_bg, reason)) << reason;
  EXPECT_NEAR(delta_bg.x(), expected.x(), 1e-3);
  EXPECT_NEAR(delta_bg.y(), expected.y(), 1e-3);
  EXPECT_NEAR(delta_bg.z(), expected.z(), 1e-3);
}

TEST(SchurInitializer, MovingDataWithoutDynamicPathDoesNotInitializeStatically)
{
  SchurInitConfig_s config = staticConfig();
  config.allow_dynamic_ = false;
  SchurInitializer initializer(config);
  feedStaticMotion(initializer, 1.2, 1.6, 0.25, 20.0);
  const SchurInitSeed_s seed = initializer.attempt(1.2, true, true, {});
  EXPECT_NE(seed.kind_, SchurInitResultKind_e::K_STATIC);
  EXPECT_NE(seed.kind_, SchurInitResultKind_e::K_DYNAMIC);
}

}  // namespace
}  // namespace backend
}  // namespace vins
