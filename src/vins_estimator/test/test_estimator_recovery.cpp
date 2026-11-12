#include <gtest/gtest.h>

#include <map>
#include <set>
#include <vector>

#include "estimator/estimator.h"

namespace
{
struct EstimatorParameterGuard_s {
  EstimatorParameterGuard_s()
    : estimate_extrinsic_(ESTIMATE_EXTRINSIC), estimate_td_(ESTIMATE_TD), num_of_cam_(NUM_OF_CAM),
      stereo_(STEREO), use_imu_(USE_IMU), multiple_thread_(MULTIPLE_THREAD), td_(TD), g_(G), ric_(RIC),
      tic_(TIC), cam_names_(CAM_NAMES)
  {
  }

  ~EstimatorParameterGuard_s()
  {
    ESTIMATE_EXTRINSIC = estimate_extrinsic_;
    ESTIMATE_TD = estimate_td_;
    NUM_OF_CAM = num_of_cam_;
    STEREO = stereo_;
    USE_IMU = use_imu_;
    MULTIPLE_THREAD = multiple_thread_;
    TD = td_;
    G = g_;
    RIC = ric_;
    TIC = tic_;
    CAM_NAMES = cam_names_;
  }

  int estimate_extrinsic_;
  int estimate_td_;
  int num_of_cam_;
  int stereo_;
  int use_imu_;
  int multiple_thread_;
  double td_;
  Eigen::Vector3d g_;
  std::vector<Eigen::Matrix3d> ric_;
  std::vector<Eigen::Vector3d> tic_;
  std::vector<std::string> cam_names_;
};

class InjectableEstimator : public Estimator {
public:
  bool initialStructure() override
  {
    ++initialization_calls_;
    return next_initialization_result_;
  }

  bool runOptimization() override
  {
    ++optimization_calls_;
    last_optimization_frame_ = frame_count_;
    return next_optimization_result_;
  }

  bool next_initialization_result_ = true;
  bool next_optimization_result_ = true;
  int initialization_calls_ = 0;
  int optimization_calls_ = 0;
  int last_optimization_frame_ = -1;
};

const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> K_EMPTY_IMAGE;

void configureInitialEstimator(InjectableEstimator & _estimator)
{
  _estimator.setParameter();
  _estimator.acc_0_.setZero();
  _estimator.gyr_0_.setZero();
  _estimator.solver_flag_ = Estimator::INITIAL;
  _estimator.frame_count_ = WINDOW_SIZE;
  for (int i = 0; i <= WINDOW_SIZE; ++i)
    _estimator.Headers_[i] = static_cast<double>(i);
}
}  // namespace

TEST(EstimatorRecovery, RepeatedInitializationFailuresStayBoundedAndRetry)
{
  EstimatorParameterGuard_s guard;
  ESTIMATE_EXTRINSIC = 0;
  ESTIMATE_TD = 0;
  NUM_OF_CAM = 1;
  STEREO = 1;
  USE_IMU = 0;
  MULTIPLE_THREAD = 0;
  TD = 0.0;
  G.setZero();
  RIC.assign(1, Eigen::Matrix3d::Identity());
  TIC.assign(1, Eigen::Vector3d::Zero());
  CAM_NAMES.clear();

  InjectableEstimator estimator;
  configureInitialEstimator(estimator);

  // Reproduce a stale initialization history whose image frames are still
  // marked as keyframes when the solver fails and the window is retried.
  for (int i = 0; i <= WINDOW_SIZE; ++i) {
    ImageFrame stale(K_EMPTY_IMAGE, static_cast<double>(i));
    stale.pre_integration_ = nullptr;
    stale.R_.setIdentity();
    stale.T_.setZero();
    stale.is_key_frame_ = true;
    estimator.all_image_frame_.emplace(stale.t_, stale);
  }

  constexpr int kFailures = 12;
  estimator.next_optimization_result_ = false;
  for (int attempt = 0; attempt < kFailures; ++attempt) {
    estimator.processImage(K_EMPTY_IMAGE, 100.0 + attempt);

    EXPECT_FALSE(estimator.restartPending()) << "attempt " << attempt;
    EXPECT_TRUE(estimator.statePublicationSuppressed()) << "attempt " << attempt;
    EXPECT_EQ(estimator.solver_flag_, Estimator::INITIAL) << "attempt " << attempt;
    EXPECT_EQ(estimator.frame_count_, WINDOW_SIZE) << "attempt " << attempt;
    // Failed initialization must leave only headers that still belong to
    // the active window; stale image frames would otherwise index Pose /
    // SpeedBias arrays out of bounds in the next initialization pass.
    EXPECT_LE(estimator.all_image_frame_.size(), static_cast<size_t>(WINDOW_SIZE + 1)) << "attempt " << attempt;
    std::set<double> active_headers;
    for (int i = 0; i <= estimator.frame_count_; ++i)
      active_headers.insert(estimator.Headers_[i]);
    for (const auto & frame_entry : estimator.all_image_frame_)
      EXPECT_NE(active_headers.find(frame_entry.first), active_headers.end())
        << "stale image frame at " << frame_entry.first << " on attempt " << attempt;
  }

  EXPECT_EQ(estimator.optimization_calls_, kFailures);
  EXPECT_EQ(estimator.last_optimization_frame_, WINDOW_SIZE);

  // After repeated failures the next successful frame still uses the normal
  // initialization path and transitions to NON_LINEAR without a crash.
  estimator.next_optimization_result_ = true;
  estimator.processImage(K_EMPTY_IMAGE, 100.0 + kFailures);

  EXPECT_FALSE(estimator.restartPending());
  EXPECT_FALSE(estimator.statePublicationSuppressed());
  EXPECT_EQ(estimator.solver_flag_, Estimator::NON_LINEAR);
  EXPECT_EQ(estimator.frame_count_, WINDOW_SIZE);
  EXPECT_EQ(estimator.optimization_calls_, kFailures + 2);
  EXPECT_LE(estimator.all_image_frame_.size(), static_cast<size_t>(WINDOW_SIZE + 1));
}

TEST(EstimatorRecovery, NonLinearFailureSchedulesControlledReset)
{
  EstimatorParameterGuard_s guard;
  ESTIMATE_EXTRINSIC = 0;
  ESTIMATE_TD = 0;
  NUM_OF_CAM = 1;
  STEREO = 1;
  USE_IMU = 0;
  MULTIPLE_THREAD = 0;
  TD = 0.0;
  G.setZero();
  RIC.assign(1, Eigen::Matrix3d::Identity());
  TIC.assign(1, Eigen::Vector3d::Zero());
  CAM_NAMES.clear();

  InjectableEstimator estimator;
  configureInitialEstimator(estimator);
  estimator.solver_flag_ = Estimator::NON_LINEAR;
  estimator.last_marginalization_info_ = new MarginalizationInfo();
  estimator.last_marginalization_parameter_blocks_.push_back(estimator.para_Pose_[0]);
  estimator.pre_integrations_[3] = new IntegrationBase{
    Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};

  estimator.next_optimization_result_ = false;
  estimator.processImage(K_EMPTY_IMAGE, 200.0);

  EXPECT_TRUE(estimator.restartPending());
  EXPECT_TRUE(estimator.statePublicationSuppressed());
  EXPECT_EQ(estimator.solver_flag_, Estimator::NON_LINEAR);
  EXPECT_EQ(estimator.frame_count_, WINDOW_SIZE);
  EXPECT_EQ(estimator.optimization_calls_, 1);

  // This is the production recovery slot used by processMeasurements() after
  // releasing mProcess.
  EXPECT_TRUE(estimator.restartAfterSolverFailure());
  EXPECT_FALSE(estimator.restartPending());
  EXPECT_EQ(estimator.solver_flag_, Estimator::INITIAL);
  EXPECT_EQ(estimator.frame_count_, 0);
  EXPECT_TRUE(estimator.all_image_frame_.empty());
  EXPECT_EQ(estimator.last_marginalization_info_, nullptr);
  EXPECT_TRUE(estimator.last_marginalization_parameter_blocks_.empty());
  EXPECT_EQ(estimator.pre_integrations_[3], nullptr);

  // The frame after a controlled reset must be able to start from a clean
  // INITIAL window instead of touching the failed window's prior or history.
  estimator.next_optimization_result_ = true;
  estimator.processImage(K_EMPTY_IMAGE, 201.0);

  EXPECT_FALSE(estimator.restartPending());
  EXPECT_EQ(estimator.solver_flag_, Estimator::INITIAL);
  EXPECT_EQ(estimator.frame_count_, 1);
  EXPECT_EQ(estimator.optimization_calls_, 2);
  EXPECT_TRUE(estimator.all_image_frame_.size() <= static_cast<size_t>(WINDOW_SIZE + 1));
}

TEST(EstimatorRecovery, MonocularInitializationFailureKeepsWindowBoundedAndRetries)
{
  EstimatorParameterGuard_s guard;
  ESTIMATE_EXTRINSIC = 0;
  ESTIMATE_TD = 0;
  NUM_OF_CAM = 1;
  STEREO = 0;
  USE_IMU = 1;
  MULTIPLE_THREAD = 0;
  TD = 0.0;
  G.setZero();
  RIC.assign(1, Eigen::Matrix3d::Identity());
  TIC.assign(1, Eigen::Vector3d::Zero());
  CAM_NAMES.clear();

  InjectableEstimator estimator;
  configureInitialEstimator(estimator);
  for (int i = 0; i <= WINDOW_SIZE; ++i) {
    ImageFrame stale(K_EMPTY_IMAGE, static_cast<double>(i));
    stale.pre_integration_ = nullptr;
    stale.R_.setIdentity();
    stale.T_.setZero();
    stale.is_key_frame_ = true;
    estimator.all_image_frame_.emplace(stale.t_, stale);
  }

  constexpr int kFailures = 12;
  estimator.next_optimization_result_ = false;
  for (int attempt = 0; attempt < kFailures; ++attempt) {
    estimator.processImage(K_EMPTY_IMAGE, 300.0 + attempt);
    EXPECT_FALSE(estimator.restartPending()) << "attempt " << attempt;
    EXPECT_TRUE(estimator.statePublicationSuppressed()) << "attempt " << attempt;
    EXPECT_EQ(estimator.solver_flag_, Estimator::INITIAL) << "attempt " << attempt;
    EXPECT_EQ(estimator.frame_count_, WINDOW_SIZE) << "attempt " << attempt;
    EXPECT_LE(estimator.all_image_frame_.size(), static_cast<size_t>(WINDOW_SIZE + 1)) << "attempt " << attempt;
  }

  EXPECT_EQ(estimator.initialization_calls_, kFailures);
  EXPECT_EQ(estimator.optimization_calls_, kFailures);

  estimator.next_optimization_result_ = true;
  estimator.processImage(K_EMPTY_IMAGE, 300.0 + kFailures);
  EXPECT_FALSE(estimator.restartPending());
  EXPECT_FALSE(estimator.statePublicationSuppressed());
  EXPECT_EQ(estimator.solver_flag_, Estimator::NON_LINEAR);
  EXPECT_EQ(estimator.frame_count_, WINDOW_SIZE);
  EXPECT_EQ(estimator.initialization_calls_, kFailures + 1);
  EXPECT_EQ(estimator.optimization_calls_, kFailures + 1);
  EXPECT_LE(estimator.all_image_frame_.size(), static_cast<size_t>(WINDOW_SIZE + 1));
}

namespace
{
void seedRealtimeSchur(Estimator & _estimator, double _time)
{
  using namespace vins::backend;
  _estimator.solver_flag_ = Estimator::NON_LINEAR;
  _estimator.frame_count_ = WINDOW_SIZE;
  _estimator.Headers_[WINDOW_SIZE] = _time;
  _estimator.td_ = 0.0;
  _estimator.g_ = Eigen::Vector3d(0, 0, 9.81);
  _estimator.acc_0_ = _estimator.g_;
  _estimator.gyr_0_.setZero();
  EkfOptions_s options;
  options.gravity_ = 9.81;
  _estimator.schur_state_.reset(
    options, _time, Eigen::Vector4d(1, 0, 0, 0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
    Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  _estimator.schur_state_.augmentClone(_time);
  // Keep a non-default marginal and FEJ to catch reset/re-inflation or lost
  // correlations when copying the live predictor from the image-time EKF.
  _estimator.schur_state_.covariance()(3, 6) = 0.0001;
  _estimator.schur_state_.covariance()(6, 3) = 0.0001;
  _estimator.schur_state_.imuFej().p_.x() = 0.01;
  _estimator.schur_ekf_active_ = true;
  _estimator.updateLatestStates();
}
}  // namespace

TEST(EstimatorImuCovariance, CameraBlackoutPropagatesWithoutTouchingImageWindow)
{
  Estimator estimator;
  seedRealtimeSchur(estimator, 10.0);
  const auto window_covariance = estimator.schur_state_.covariance();
  const auto initial_covariance = estimator.latest_schur_state_.covariance();
  auto reference = estimator.schur_state_;
  ASSERT_EQ(estimator.latest_schur_state_.numClones(), 0);
  EXPECT_TRUE(initial_covariance.isApprox(window_covariance.topLeftCorner(15, 15), 0));
  EXPECT_TRUE(estimator.latest_schur_state_.imuFej().p_.isApprox(reference.imuFej().p_, 0));
  for (int i = 1; i <= 2000; ++i) {
    const double time = 10.0 + i * 0.005;
    const Eigen::Vector3d acc(0.01, 0.02, 9.81), gyro(0, 0, 0.01);
    reference.propagate(acc, gyro, time - reference.timestamp());
    // No processImage/processMeasurements calls during this ten-second gap.
    estimator.inputIMU(time, acc, gyro);
  }
  EXPECT_NEAR(estimator.latest_time_, 20.0, 1e-12);
  EXPECT_NEAR(estimator.latest_schur_state_.timestamp(), 20.0, 1e-12);
  EXPECT_TRUE(
    estimator.latest_schur_state_.covariance().isApprox(reference.covariance().topLeftCorner(15, 15), 1e-10));
  EXPECT_TRUE(estimator.latest_P_.isApprox(reference.imu().p_, 1e-10));
  EXPECT_GT(estimator.latest_schur_state_.covariance()(3, 3), initial_covariance(3, 3));
  EXPECT_TRUE(vins::backend::EkfState::isPositiveSemidefinite(estimator.latest_schur_state_.covariance()));
  EXPECT_DOUBLE_EQ(estimator.schur_state_.timestamp(), 10.0);
  EXPECT_TRUE(estimator.schur_state_.covariance().isApprox(window_covariance, 0));
  EXPECT_FALSE(estimator.restartPending());
}

TEST(EstimatorImuCovariance, ReanchorReplaysOnlyTailAndResetDisablesPublication)
{
  Estimator estimator;
  seedRealtimeSchur(estimator, 10.0);
  const Eigen::Vector3d acc(0, 0, 9.81), gyro = Eigen::Vector3d::Zero();
  for (int i = 1; i <= 20; ++i)
    estimator.inputIMU(10.0 + i * 0.005, acc, gyro);
  // Model a delayed visual correction at 10.05 while IMU is already at 10.10.
  for (int i = 1; i <= 10; ++i)
    estimator.schur_state_.propagate(acc, gyro, 10.0 + i * 0.005 - estimator.schur_state_.timestamp());
  estimator.schur_state_.imu().p_.x() = 2.0;
  estimator.schur_state_.covariance() *= 0.5;
  auto reference = estimator.schur_state_;
  for (int i = 11; i <= 20; ++i)
    reference.propagate(acc, gyro, 10.0 + i * 0.005 - reference.timestamp());
  estimator.updateLatestStates();
  EXPECT_NEAR(estimator.latest_time_, 10.1, 1e-12);
  EXPECT_TRUE(estimator.latest_P_.isApprox(reference.imu().p_, 1e-12));
  EXPECT_TRUE(
    estimator.latest_schur_state_.covariance().isApprox(reference.covariance().topLeftCorner(15, 15), 1e-12));
  const auto covariance = estimator.latest_schur_state_.covariance();
  const auto buffered = estimator.accBuf_.size();
  estimator.inputIMU(10.1, acc, gyro);   // duplicate
  estimator.inputIMU(10.09, acc, gyro);  // out of order
  EXPECT_EQ(estimator.accBuf_.size(), buffered);
  EXPECT_TRUE(estimator.latest_schur_state_.covariance().isApprox(covariance, 0));
  // The next callback must integrate exactly once after replay.
  estimator.inputIMU(10.105, acc, gyro);
  reference.propagate(acc, gyro, 10.105 - reference.timestamp());
  EXPECT_TRUE(
    estimator.latest_schur_state_.covariance().isApprox(reference.covariance().topLeftCorner(15, 15), 1e-12));
  estimator.clearState();
  EXPECT_FALSE(estimator.latest_state_valid_);
  EXPECT_FALSE(estimator.latest_schur_state_.initialized());
  estimator.inputIMU(1.0, acc, gyro);  // reset permits a new timestamp epoch
  EXPECT_EQ(estimator.accBuf_.size(), 1u);
  EXPECT_FALSE(estimator.latest_schur_state_.initialized());
}

TEST(EstimatorImuCovariance, ConcurrentCallbacksAndVisualReanchorsDoNotDoubleIntegrate)
{
  Estimator estimator;
  seedRealtimeSchur(estimator, 10.0);
  const Eigen::Vector3d acc(0.01, -0.02, 9.81), gyro(0.01, 0.02, 0.03);
  std::thread producer(
    [&]
    {
      for (int i = 1; i <= 400; ++i)
        estimator.inputIMU(10.0 + i * 0.005, acc, gyro);
    });
  for (int i = 0; i < 40; ++i)
    estimator.updateLatestStates();
  producer.join();
  auto reference = estimator.schur_state_;
  for (int i = 1; i <= 400; ++i)
    reference.propagate(acc, gyro, 10.0 + i * 0.005 - reference.timestamp());
  EXPECT_NEAR(estimator.latest_time_, 12.0, 1e-12);
  EXPECT_TRUE(
    estimator.latest_schur_state_.covariance().isApprox(reference.covariance().topLeftCorner(15, 15), 1e-12));
  EXPECT_TRUE(estimator.latest_P_.isApprox(reference.imu().p_, 1e-12));
  // A missing IMU second must not publish stale P under a new timestamp.
  estimator.inputIMU(14.0, acc, gyro);
  EXPECT_FALSE(estimator.latest_state_valid_);
  EXPECT_NEAR(estimator.latest_schur_state_.timestamp(), 12.0, 1e-12);
}

TEST(EstimatorImuCovariance, GyroNoiseUsesSensorPeriodAcrossFractionalReanchor)
{
  Estimator estimator;
  seedRealtimeSchur(estimator, 10.0);
  const Eigen::Vector3d acc(0, 0, 9.81), gyro = Eigen::Vector3d::Zero();
  estimator.inputIMU(10.005, acc, gyro);
  estimator.inputIMU(10.010, acc, gyro);
  // The image filter has used a partial step with the 10.010 sample; its
  // replay to 10.010 lasts only 1 ms, while gyro noise still spans 5 ms.
  estimator.schur_state_.propagate(acc, gyro, 0.009);
  estimator.updateLatestStates();
  EXPECT_NEAR(estimator.latest_time_, 10.010, 1e-12);
  EXPECT_NEAR(estimator.latest_imu_dt_, 0.005, 1e-12);
  estimator.inputIMU(10.015, acc, gyro);
  EXPECT_NEAR(estimator.latest_imu_dt_, 0.005, 1e-12);
  // Even without a buffered endpoint, the next integration step may be
  // fractional. It must not become the measurement's noise period.
  estimator.schur_state_.propagate(acc, gyro, 0.010);
  estimator.updateLatestStates();
  estimator.inputIMU(10.020, acc, gyro);
  EXPECT_NEAR(estimator.latest_imu_dt_, 0.005, 1e-12);
  estimator.clearState();
  EXPECT_DOUBLE_EQ(estimator.latest_imu_dt_, 0.0);
}

TEST(EstimatorLandingDetector, ClearStateDropsActiveStaticEvidence)
{
  Estimator estimator;
  vins::backend::LandingStaticConfig_s config;
  config.enabled_ = true;
  config.min_still_seconds_ = 0.5;
  config.max_flow_px_ = 0.25;
  config.min_features_ = 20;
  config.min_quadrants_ = 3;
  config.impact_window_seconds_ = 6.0;
  config.impact_accel_ = 20.0;
  config.max_image_gap_seconds_ = 0.2;
  config.imu_window_seconds_ = 0.2;
  config.max_imu_sample_gap_seconds_ = 0.05;
  config.min_imu_samples_ = 10;
  config.max_gyro_rad_s_ = 0.08;
  config.max_accel_dev_ = 0.6;
  config.max_gyro_std_ = 0.03;
  config.max_accel_std_ = 0.35;
  config.max_speed_ = 0.3;
  estimator.schur_landing_static_.configure(config);

  std::vector<vins::backend::LandingPixelObservation_s> observations;
  int id = 0;
  for (int row = 0; row < 6; ++row)
    for (int col = 0; col < 8; ++col)
      observations.push_back(
        vins::backend::LandingPixelObservation_s{id++, Eigen::Vector2d(60.0 + 80.0 * col, 40.0 + 70.0 * row)});

  estimator.schur_landing_static_.observeImu(0.0, Eigen::Vector3d(0.0, 0.0, 25.0), Eigen::Vector3d::Zero());
  double imu_t = 0.005;
  for (int i = 0; i <= 20; ++i) {
    const double t = 0.10 + 0.05 * i;
    for (double sample_t = imu_t; sample_t <= t + 1e-9; sample_t += 0.005)
      estimator.schur_landing_static_.observeImu(
        sample_t, Eigen::Vector3d(0.0, 0.0, 9.81007), Eigen::Vector3d::Zero());
    imu_t = t + 0.005;
    estimator.schur_landing_static_.observeFrame(
      t, observations, Eigen::Vector3d(0.05, 0.0, 0.0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  }
  ASSERT_TRUE(estimator.schur_landing_static_.active());

  estimator.clearState();
  EXPECT_FALSE(estimator.schur_landing_static_.active());
}

TEST(EstimatorSchurStereo, SchurProcessFrameCollectsBothConfiguredCameras)
{
  EstimatorParameterGuard_s guard;
  ESTIMATE_EXTRINSIC = 0;
  ESTIMATE_TD = 0;
  NUM_OF_CAM = 2;
  STEREO = 1;
  USE_IMU = 1;
  MULTIPLE_THREAD = 0;
  TD = 0.0;
  G = Eigen::Vector3d(0.0, 0.0, -9.81);
  RIC.assign(2, Eigen::Matrix3d::Identity());
  TIC.assign(2, Eigen::Vector3d::Zero());
  TIC[1] = Eigen::Vector3d(-0.11, 0.0, 0.0);
  CAM_NAMES.clear();

  Estimator estimator;
  estimator.setParameter();
  estimator.frame_count_ = 0;
  estimator.Headers_[0] = 0.0;
  estimator.Rs_[0].setIdentity();
  estimator.Ps_[0].setZero();
  estimator.initSchurBackend();
  ASSERT_TRUE(estimator.schurBackendActive());
  ASSERT_EQ(estimator.schur_cameras_.size(), 2u);
  EXPECT_NEAR(estimator.schur_cameras_[1].p_cb_.x(), -0.11, 1e-12);

  Estimator::FeatureFrame image;
  Eigen::Matrix<double, 7, 1> left;
  left << 0.10, -0.05, 1.0, 320.0, 240.0, 0.0, 0.0;
  Eigen::Matrix<double, 7, 1> right;
  right << 0.05, -0.05, 1.0, 280.0, 240.0, 0.0, 0.0;
  // Right first: the estimator must still add both views and restore the
  // deterministic (frame, camera) order in FeatureHistory.
  image[7].emplace_back(1, right);
  image[7].emplace_back(0, left);

  ASSERT_TRUE(estimator.schurProcessFrame(image, 0.0));
  const std::vector<vins::backend::FeatureTrack_s> tracks = estimator.schur_history_.allTracks(0);
  ASSERT_EQ(tracks.size(), 1u);
  ASSERT_EQ(tracks[0].observations_.size(), 2u);
  EXPECT_EQ(tracks[0].observations_[0].clone_index_, 0);
  EXPECT_EQ(tracks[0].observations_[0].camera_id_, 0);
  EXPECT_EQ(tracks[0].observations_[1].clone_index_, 0);
  EXPECT_EQ(tracks[0].observations_[1].camera_id_, 1);
  EXPECT_NEAR(tracks[0].observations_[0].pixel_uv_.x(), 320.0, 1e-12);
  EXPECT_NEAR(tracks[0].observations_[1].pixel_uv_.x(), 280.0, 1e-12);
}
