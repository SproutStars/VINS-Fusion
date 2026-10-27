#include <gtest/gtest.h>

#include <map>
#include <set>
#include <vector>

#include "estimator/estimator.h"

namespace
{
struct EstimatorParameterGuard
{
    EstimatorParameterGuard()
        : estimate_extrinsic(ESTIMATE_EXTRINSIC), estimate_td(ESTIMATE_TD),
          num_of_cam(NUM_OF_CAM), stereo(STEREO), use_imu(USE_IMU),
          multiple_thread(MULTIPLE_THREAD), td(TD), g(G), ric(RIC), tic(TIC),
          cam_names(CAM_NAMES) {}

    ~EstimatorParameterGuard()
    {
        ESTIMATE_EXTRINSIC = estimate_extrinsic;
        ESTIMATE_TD = estimate_td;
        NUM_OF_CAM = num_of_cam;
        STEREO = stereo;
        USE_IMU = use_imu;
        MULTIPLE_THREAD = multiple_thread;
        TD = td;
        G = g;
        RIC = ric;
        TIC = tic;
        CAM_NAMES = cam_names;
    }

    int estimate_extrinsic;
    int estimate_td;
    int num_of_cam;
    int stereo;
    int use_imu;
    int multiple_thread;
    double td;
    Eigen::Vector3d g;
    std::vector<Eigen::Matrix3d> ric;
    std::vector<Eigen::Vector3d> tic;
    std::vector<std::string> cam_names;
};

class InjectableEstimator : public Estimator
{
  public:
    bool initialStructure() override
    {
        ++initialization_calls;
        return next_initialization_result;
    }

    bool runOptimization() override
    {
        ++optimization_calls;
        last_optimization_frame = frame_count;
        return next_optimization_result;
    }

    bool next_initialization_result = true;
    bool next_optimization_result = true;
    int initialization_calls = 0;
    int optimization_calls = 0;
    int last_optimization_frame = -1;
};

const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> kEmptyImage;

void configureInitialEstimator(InjectableEstimator &estimator)
{
    estimator.setParameter();
    estimator.acc_0.setZero();
    estimator.gyr_0.setZero();
    estimator.solver_flag = Estimator::INITIAL;
    estimator.frame_count = WINDOW_SIZE;
    for (int i = 0; i <= WINDOW_SIZE; ++i)
        estimator.Headers[i] = static_cast<double>(i);
}
}

TEST(EstimatorRecovery, RepeatedInitializationFailuresStayBoundedAndRetry)
{
    EstimatorParameterGuard guard;
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
    for (int i = 0; i <= WINDOW_SIZE; ++i)
    {
        ImageFrame stale(kEmptyImage, static_cast<double>(i));
        stale.pre_integration = nullptr;
        stale.R.setIdentity();
        stale.T.setZero();
        stale.is_key_frame = true;
        estimator.all_image_frame.emplace(stale.t, stale);
    }

    constexpr int kFailures = 12;
    estimator.next_optimization_result = false;
    for (int attempt = 0; attempt < kFailures; ++attempt)
    {
        estimator.processImage(kEmptyImage, 100.0 + attempt);

        EXPECT_FALSE(estimator.restartPending()) << "attempt " << attempt;
        EXPECT_TRUE(estimator.statePublicationSuppressed()) << "attempt " << attempt;
        EXPECT_EQ(estimator.solver_flag, Estimator::INITIAL) << "attempt " << attempt;
        EXPECT_EQ(estimator.frame_count, WINDOW_SIZE) << "attempt " << attempt;
        // Failed initialization must leave only headers that still belong to
        // the active window; stale image frames would otherwise index Pose /
        // SpeedBias arrays out of bounds in the next initialization pass.
        EXPECT_LE(estimator.all_image_frame.size(), static_cast<size_t>(WINDOW_SIZE + 1))
            << "attempt " << attempt;
        std::set<double> active_headers;
        for (int i = 0; i <= estimator.frame_count; ++i)
            active_headers.insert(estimator.Headers[i]);
        for (const auto &frame_entry : estimator.all_image_frame)
            EXPECT_NE(active_headers.find(frame_entry.first), active_headers.end())
                << "stale image frame at " << frame_entry.first << " on attempt " << attempt;
    }

    EXPECT_EQ(estimator.optimization_calls, kFailures);
    EXPECT_EQ(estimator.last_optimization_frame, WINDOW_SIZE);

    // After repeated failures the next successful frame still uses the normal
    // initialization path and transitions to NON_LINEAR without a crash.
    estimator.next_optimization_result = true;
    estimator.processImage(kEmptyImage, 100.0 + kFailures);

    EXPECT_FALSE(estimator.restartPending());
    EXPECT_FALSE(estimator.statePublicationSuppressed());
    EXPECT_EQ(estimator.solver_flag, Estimator::NON_LINEAR);
    EXPECT_EQ(estimator.frame_count, WINDOW_SIZE);
    EXPECT_EQ(estimator.optimization_calls, kFailures + 2);
    EXPECT_LE(estimator.all_image_frame.size(), static_cast<size_t>(WINDOW_SIZE + 1));
}

TEST(EstimatorRecovery, NonLinearFailureSchedulesControlledReset)
{
    EstimatorParameterGuard guard;
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
    estimator.solver_flag = Estimator::NON_LINEAR;
    estimator.last_marginalization_info = new MarginalizationInfo();
    estimator.last_marginalization_parameter_blocks.push_back(estimator.para_Pose[0]);
    estimator.pre_integrations[3] = new IntegrationBase{
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};

    estimator.next_optimization_result = false;
    estimator.processImage(kEmptyImage, 200.0);

    EXPECT_TRUE(estimator.restartPending());
    EXPECT_TRUE(estimator.statePublicationSuppressed());
    EXPECT_EQ(estimator.solver_flag, Estimator::NON_LINEAR);
    EXPECT_EQ(estimator.frame_count, WINDOW_SIZE);
    EXPECT_EQ(estimator.optimization_calls, 1);

    // This is the production recovery slot used by processMeasurements() after
    // releasing mProcess.
    EXPECT_TRUE(estimator.restartAfterSolverFailure());
    EXPECT_FALSE(estimator.restartPending());
    EXPECT_EQ(estimator.solver_flag, Estimator::INITIAL);
    EXPECT_EQ(estimator.frame_count, 0);
    EXPECT_TRUE(estimator.all_image_frame.empty());
    EXPECT_EQ(estimator.last_marginalization_info, nullptr);
    EXPECT_TRUE(estimator.last_marginalization_parameter_blocks.empty());
    EXPECT_EQ(estimator.pre_integrations[3], nullptr);

    // The frame after a controlled reset must be able to start from a clean
    // INITIAL window instead of touching the failed window's prior or history.
    estimator.next_optimization_result = true;
    estimator.processImage(kEmptyImage, 201.0);

    EXPECT_FALSE(estimator.restartPending());
    EXPECT_EQ(estimator.solver_flag, Estimator::INITIAL);
    EXPECT_EQ(estimator.frame_count, 1);
    EXPECT_EQ(estimator.optimization_calls, 2);
    EXPECT_TRUE(estimator.all_image_frame.size() <= static_cast<size_t>(WINDOW_SIZE + 1));
}

TEST(EstimatorRecovery, MonocularInitializationFailureKeepsWindowBoundedAndRetries)
{
    EstimatorParameterGuard guard;
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
    for (int i = 0; i <= WINDOW_SIZE; ++i)
    {
        ImageFrame stale(kEmptyImage, static_cast<double>(i));
        stale.pre_integration = nullptr;
        stale.R.setIdentity();
        stale.T.setZero();
        stale.is_key_frame = true;
        estimator.all_image_frame.emplace(stale.t, stale);
    }

    constexpr int kFailures = 12;
    estimator.next_optimization_result = false;
    for (int attempt = 0; attempt < kFailures; ++attempt)
    {
        estimator.processImage(kEmptyImage, 300.0 + attempt);
        EXPECT_FALSE(estimator.restartPending()) << "attempt " << attempt;
        EXPECT_TRUE(estimator.statePublicationSuppressed()) << "attempt " << attempt;
        EXPECT_EQ(estimator.solver_flag, Estimator::INITIAL) << "attempt " << attempt;
        EXPECT_EQ(estimator.frame_count, WINDOW_SIZE) << "attempt " << attempt;
        EXPECT_LE(estimator.all_image_frame.size(), static_cast<size_t>(WINDOW_SIZE + 1))
            << "attempt " << attempt;
    }

    EXPECT_EQ(estimator.initialization_calls, kFailures);
    EXPECT_EQ(estimator.optimization_calls, kFailures);

    estimator.next_optimization_result = true;
    estimator.processImage(kEmptyImage, 300.0 + kFailures);
    EXPECT_FALSE(estimator.restartPending());
    EXPECT_FALSE(estimator.statePublicationSuppressed());
    EXPECT_EQ(estimator.solver_flag, Estimator::NON_LINEAR);
    EXPECT_EQ(estimator.frame_count, WINDOW_SIZE);
    EXPECT_EQ(estimator.initialization_calls, kFailures + 1);
    EXPECT_EQ(estimator.optimization_calls, kFailures + 1);
    EXPECT_LE(estimator.all_image_frame.size(), static_cast<size_t>(WINDOW_SIZE + 1));
}

namespace
{
void seedRealtimeSchur(Estimator &estimator, double time)
{
    using namespace vins::backend;
    estimator.solver_flag = Estimator::NON_LINEAR;
    estimator.frame_count = WINDOW_SIZE;
    estimator.Headers[WINDOW_SIZE] = time;
    estimator.td = 0.0;
    estimator.g = Eigen::Vector3d(0, 0, 9.81);
    estimator.acc_0 = estimator.g;
    estimator.gyr_0.setZero();
    EkfOptions options;
    options.gravity = 9.81;
    estimator.schur_state_.reset(options, time, Eigen::Vector4d(1, 0, 0, 0),
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    estimator.schur_state_.augment_clone(time);
    // Keep a non-default marginal and FEJ to catch reset/re-inflation or lost
    // correlations when copying the live predictor from the image-time EKF.
    estimator.schur_state_.covariance()(3, 6) = 0.0001;
    estimator.schur_state_.covariance()(6, 3) = 0.0001;
    estimator.schur_state_.imu_fej().p.x() = 0.01;
    estimator.schur_ekf_active_ = true;
    estimator.updateLatestStates();
}
}

TEST(EstimatorImuCovariance, CameraBlackoutPropagatesWithoutTouchingImageWindow)
{
    Estimator estimator;
    seedRealtimeSchur(estimator, 10.0);
    const auto window_covariance = estimator.schur_state_.covariance();
    const auto initial_covariance = estimator.latest_schur_state_.covariance();
    auto reference = estimator.schur_state_;
    ASSERT_EQ(estimator.latest_schur_state_.num_clones(), 0);
    EXPECT_TRUE(initial_covariance.isApprox(window_covariance.topLeftCorner(15, 15), 0));
    EXPECT_TRUE(estimator.latest_schur_state_.imu_fej().p.isApprox(reference.imu_fej().p, 0));
    for (int i = 1; i <= 2000; ++i)
    {
        const double time = 10.0 + i * 0.005;
        const Eigen::Vector3d acc(0.01, 0.02, 9.81), gyro(0, 0, 0.01);
        reference.propagate(acc, gyro, time - reference.timestamp());
        // No processImage/processMeasurements calls during this ten-second gap.
        estimator.inputIMU(time, acc, gyro);
    }
    EXPECT_NEAR(estimator.latest_time, 20.0, 1e-12);
    EXPECT_NEAR(estimator.latest_schur_state_.timestamp(), 20.0, 1e-12);
    EXPECT_TRUE(estimator.latest_schur_state_.covariance().isApprox(
        reference.covariance().topLeftCorner(15, 15), 1e-10));
    EXPECT_TRUE(estimator.latest_P.isApprox(reference.imu().p, 1e-10));
    EXPECT_GT(estimator.latest_schur_state_.covariance()(3, 3), initial_covariance(3, 3));
    EXPECT_TRUE(vins::backend::EkfState::is_positive_semidefinite(
        estimator.latest_schur_state_.covariance()));
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
    estimator.schur_state_.imu().p.x() = 2.0;
    estimator.schur_state_.covariance() *= 0.5;
    auto reference = estimator.schur_state_;
    for (int i = 11; i <= 20; ++i)
        reference.propagate(acc, gyro, 10.0 + i * 0.005 - reference.timestamp());
    estimator.updateLatestStates();
    EXPECT_NEAR(estimator.latest_time, 10.1, 1e-12);
    EXPECT_TRUE(estimator.latest_P.isApprox(reference.imu().p, 1e-12));
    EXPECT_TRUE(estimator.latest_schur_state_.covariance().isApprox(
        reference.covariance().topLeftCorner(15, 15), 1e-12));
    const auto covariance = estimator.latest_schur_state_.covariance();
    const auto buffered = estimator.accBuf.size();
    estimator.inputIMU(10.1, acc, gyro);  // duplicate
    estimator.inputIMU(10.09, acc, gyro); // out of order
    EXPECT_EQ(estimator.accBuf.size(), buffered);
    EXPECT_TRUE(estimator.latest_schur_state_.covariance().isApprox(covariance, 0));
    // The next callback must integrate exactly once after replay.
    estimator.inputIMU(10.105, acc, gyro);
    reference.propagate(acc, gyro, 10.105 - reference.timestamp());
    EXPECT_TRUE(estimator.latest_schur_state_.covariance().isApprox(
        reference.covariance().topLeftCorner(15, 15), 1e-12));
    estimator.clearState();
    EXPECT_FALSE(estimator.latest_state_valid_);
    EXPECT_FALSE(estimator.latest_schur_state_.initialized());
    estimator.inputIMU(1.0, acc, gyro); // reset permits a new timestamp epoch
    EXPECT_EQ(estimator.accBuf.size(), 1u);
    EXPECT_FALSE(estimator.latest_schur_state_.initialized());
}

TEST(EstimatorImuCovariance, ConcurrentCallbacksAndVisualReanchorsDoNotDoubleIntegrate)
{
    Estimator estimator;
    seedRealtimeSchur(estimator, 10.0);
    const Eigen::Vector3d acc(0.01, -0.02, 9.81), gyro(0.01, 0.02, 0.03);
    std::thread producer([&] {
        for (int i = 1; i <= 400; ++i)
            estimator.inputIMU(10.0 + i * 0.005, acc, gyro);
    });
    for (int i = 0; i < 40; ++i)
        estimator.updateLatestStates();
    producer.join();
    auto reference = estimator.schur_state_;
    for (int i = 1; i <= 400; ++i)
        reference.propagate(acc, gyro, 10.0 + i * 0.005 - reference.timestamp());
    EXPECT_NEAR(estimator.latest_time, 12.0, 1e-12);
    EXPECT_TRUE(estimator.latest_schur_state_.covariance().isApprox(
        reference.covariance().topLeftCorner(15, 15), 1e-12));
    EXPECT_TRUE(estimator.latest_P.isApprox(reference.imu().p, 1e-12));
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
    EXPECT_NEAR(estimator.latest_time, 10.010, 1e-12);
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
