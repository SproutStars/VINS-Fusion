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
