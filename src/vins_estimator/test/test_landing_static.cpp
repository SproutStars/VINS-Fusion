#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "backend/landing_static.h"

namespace
{
using namespace vins::backend;

LandingStaticConfig test_config()
{
    LandingStaticConfig config;
    config.enabled = true;
    config.min_still_seconds = 0.5;
    config.max_flow_px = 0.25;
    config.min_static_fraction = 0.7;
    config.min_features = 20;
    config.min_quadrants = 3;
    config.impact_window_seconds = 6.0;
    config.impact_accel = 20.0;
    config.max_image_gap_seconds = 0.20;
    config.imu_window_seconds = 0.20;
    config.min_imu_samples = 10;
    config.max_gyro_rad_s = 0.08;
    config.max_accel_dev = 0.60;
    config.max_accel_std = 0.35;
    config.max_gyro_std = 0.03;
    config.max_speed = 0.30;
    config.exit_flow_px = 0.50;
    config.exit_speed = 0.40;
    config.exit_gyro_rad_s = 0.15;
    config.exit_accel_dev = 1.0;
    config.gravity = 9.81007;
    return config;
}

std::vector<LandingPixelObservation> static_grid(double x_shift = 0.0,
                                                 bool one_quadrant = false)
{
    std::vector<LandingPixelObservation> observations;
    int id = 0;
    for (int row = 0; row < 6; ++row)
    {
        for (int col = 0; col < 8; ++col)
        {
            const double x = 60.0 + 80.0 * col + x_shift;
            const double y = 40.0 + 70.0 * row;
            if (one_quadrant && (x >= 376.0 || y >= 240.0))
                continue;
            observations.push_back(LandingPixelObservation{id++, Eigen::Vector2d(x, y)});
        }
    }
    return observations;
}

void feed_imu(LandingStaticDetector &detector, double t0, double t1,
              const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
{
    constexpr double dt = 0.005;
    for (double t = t0; t <= t1 + 1e-9; t += dt)
        detector.observe_imu(t, acc, gyr);
}

LandingFrameResult feed_frame(LandingStaticDetector &detector, double t,
                              const std::vector<LandingPixelObservation> &observations,
                              const Eigen::Vector3d &velocity, double &imu_t,
                              const Eigen::Vector3d &acc =
                                  Eigen::Vector3d(0.0, 0.0, 9.81007),
                              const Eigen::Vector3d &gyr = Eigen::Vector3d::Zero())
{
    feed_imu(detector, imu_t, t, acc, gyr);
    imu_t = t + 0.005;
    return detector.observe_frame(t, observations, velocity, Eigen::Vector3d::Zero(),
                                  Eigen::Vector3d::Zero());
}

TEST(LandingStaticDetector, RequiresSustainedVisualStillness)
{
    LandingStaticDetector detector;
    detector.configure(test_config());
    detector.observe_imu(0.0, Eigen::Vector3d(0.0, 0.0, 25.0), Eigen::Vector3d::Zero());

    const auto observations = static_grid();
    bool active_before = false;
    bool active_after = false;
    double imu_t = 0.005;
    for (int i = 0; i <= 24; ++i)
    {
        const double t = 0.10 + 0.05 * i;
        const LandingFrameResult result =
            feed_frame(detector, t, observations, Eigen::Vector3d(0.05, 0.0, 0.0), imu_t);
        if (t <= 0.50)
            active_before = active_before || result.active;
        if (t >= 0.85)
            active_after = active_after || result.active;
    }
    EXPECT_FALSE(active_before);
    EXPECT_TRUE(active_after);
    EXPECT_TRUE(detector.active());
}

TEST(LandingStaticDetector, ImpactAloneDoesNotActivate)
{
    LandingStaticDetector detector;
    detector.configure(test_config());
    detector.observe_imu(0.0, Eigen::Vector3d(0.0, 0.0, 25.0), Eigen::Vector3d::Zero());

    const auto observations = static_grid();
    double imu_t = 0.005;
    for (int i = 0; i <= 24; ++i)
    {
        // Non-zero velocity keeps the platform moving even though the image is
        // static, e.g. a uniform far-field motion ambiguity.
        const LandingFrameResult result =
            feed_frame(detector, 0.10 + 0.05 * i, observations,
                       Eigen::Vector3d(0.60, 0.0, 0.0), imu_t);
        EXPECT_FALSE(result.active);
    }
    EXPECT_FALSE(detector.active());
}

TEST(LandingStaticDetector, ContinuousMotionAfterImpactCannotActivate)
{
    LandingStaticDetector detector;
    detector.configure(test_config());
    detector.observe_imu(0.0, Eigen::Vector3d(0.0, 0.0, 25.0), Eigen::Vector3d::Zero());

    double imu_t = 0.005;
    for (int i = 0; i <= 24; ++i)
    {
        const double t = 0.10 + 0.05 * i;
        const auto observations = static_grid(3.0 * i);
        const LandingFrameResult result =
            feed_frame(detector, t, observations, Eigen::Vector3d(0.05, 0.0, 0.0), imu_t);
        EXPECT_FALSE(result.active);
    }
    EXPECT_FALSE(detector.active());
}

TEST(LandingStaticDetector, MotionExitsImmediately)
{
    LandingStaticDetector detector;
    detector.configure(test_config());
    detector.observe_imu(0.0, Eigen::Vector3d(0.0, 0.0, 25.0), Eigen::Vector3d::Zero());

    bool entered = false;
    double t = 0.10;
    double imu_t = 0.005;
    for (int i = 0; i < 24; ++i)
    {
        t = 0.10 + 0.05 * i;
        const LandingFrameResult result =
            feed_frame(detector, t, static_grid(), Eigen::Vector3d(0.05, 0.0, 0.0), imu_t);
        entered = entered || result.entered;
        if (entered)
            break;
    }
    ASSERT_TRUE(entered);
    ASSERT_TRUE(detector.active());

    const LandingFrameResult moving = feed_frame(
        detector, t + 0.05, static_grid(2.0), Eigen::Vector3d(0.05, 0.0, 0.0), imu_t);
    EXPECT_FALSE(moving.active);
    EXPECT_TRUE(moving.exited);
    EXPECT_FALSE(detector.active());
}

TEST(LandingStaticDetector, ImageGapClearsEvidence)
{
    LandingStaticDetector detector;
    detector.configure(test_config());
    detector.observe_imu(0.0, Eigen::Vector3d(0.0, 0.0, 25.0), Eigen::Vector3d::Zero());

    bool entered = false;
    double t = 0.10;
    double imu_t = 0.005;
    for (int i = 0; i < 24; ++i)
    {
        t = 0.10 + 0.05 * i;
        const LandingFrameResult result =
            feed_frame(detector, t, static_grid(), Eigen::Vector3d(0.05, 0.0, 0.0), imu_t);
        entered = entered || result.entered;
        if (entered)
            break;
    }
    ASSERT_TRUE(entered);

    const LandingFrameResult after_gap =
        feed_frame(detector, t + 0.5, static_grid(), Eigen::Vector3d(0.05, 0.0, 0.0), imu_t);
    EXPECT_FALSE(after_gap.active);
    EXPECT_TRUE(after_gap.exited);
    EXPECT_FALSE(detector.active());
}

TEST(LandingStaticDetector, SpatialCoverageIsRequired)
{
    LandingStaticDetector detector;
    detector.configure(test_config());
    detector.observe_imu(0.0, Eigen::Vector3d(0.0, 0.0, 25.0), Eigen::Vector3d::Zero());

    const auto observations = static_grid(0.0, true);
    double imu_t = 0.005;
    for (int i = 0; i <= 24; ++i)
    {
        const LandingFrameResult result =
            feed_frame(detector, 0.10 + 0.05 * i, observations,
                       Eigen::Vector3d(0.05, 0.0, 0.0), imu_t);
        EXPECT_FALSE(result.active);
    }
    EXPECT_FALSE(detector.active());
}

TEST(LandingStaticDetector, DebiasedImuMustBeQuiet)
{
    LandingStaticDetector detector;
    detector.configure(test_config());
    detector.observe_imu(0.0, Eigen::Vector3d(0.0, 0.0, 25.0), Eigen::Vector3d::Zero());

    const auto observations = static_grid();
    double imu_t = 0.005;
    for (int i = 0; i <= 24; ++i)
    {
        const LandingFrameResult result =
            feed_frame(detector, 0.10 + 0.05 * i, observations,
                       Eigen::Vector3d(0.05, 0.0, 0.0), imu_t,
                       Eigen::Vector3d(0.0, 0.0, 9.81007),
                       Eigen::Vector3d(0.0, 0.0, 0.2));
        EXPECT_FALSE(result.active);
    }
    EXPECT_FALSE(detector.active());
}

TEST(LandingStaticDetector, NoImpactNeverActivates)
{
    LandingStaticDetector detector;
    detector.configure(test_config());

    const auto observations = static_grid();
    double imu_t = 0.005;
    for (int i = 0; i <= 24; ++i)
    {
        const LandingFrameResult result =
            feed_frame(detector, 0.10 + 0.05 * i, observations,
                       Eigen::Vector3d(0.05, 0.0, 0.0), imu_t);
        EXPECT_FALSE(result.active);
    }
    EXPECT_FALSE(detector.active());
}

TEST(LandingStaticDetector, DuplicateImuTimestampIsIgnoredNotAReset)
{
    LandingStaticDetector detector;
    detector.configure(test_config());
    detector.observe_imu(0.0, Eigen::Vector3d(0.0, 0.0, 25.0), Eigen::Vector3d::Zero());

    const auto observations = static_grid();
    double imu_t = 0.005;
    double t = 0.10;
    bool entered = false;
    for (int i = 0; i < 24; ++i)
    {
        t = 0.10 + 0.05 * i;
        const LandingFrameResult result =
            feed_frame(detector, t, observations, Eigen::Vector3d(0.05, 0.0, 0.0), imu_t);
        entered = entered || result.entered;
        if (entered)
            break;
    }
    ASSERT_TRUE(entered);
    ASSERT_TRUE(detector.active());

    // getIMUInterval may replay the boundary sample with the same timestamp.
    // This must be ignored, not treated as a backwards clock jump.
    detector.observe_imu(t, Eigen::Vector3d(0.0, 0.0, 9.81007), Eigen::Vector3d::Zero());
    const LandingFrameResult next =
        feed_frame(detector, t + 0.05, observations, Eigen::Vector3d(0.05, 0.0, 0.0), imu_t);
    EXPECT_TRUE(next.active);
    EXPECT_FALSE(next.exited);
}

TEST(LandingStaticDetector, ShortImuWindowOrGapDoesNotQualify)
{
    LandingStaticDetector detector;
    LandingStaticConfig config = test_config();
    config.imu_window_seconds = 0.5;
    config.max_imu_sample_gap_seconds = 0.05;
    config.min_imu_samples = 10;
    detector.configure(config);
    detector.observe_imu(0.0, Eigen::Vector3d(0.0, 0.0, 25.0), Eigen::Vector3d::Zero());

    // Ten samples in 10 ms satisfy the count but span too little of the 0.5 s
    // window. A second case inserts a 100 ms hole between samples.
    for (int i = 0; i < 5; ++i)
        detector.observe_imu(0.100 + 0.001 * i, Eigen::Vector3d(0.0, 0.0, 9.81007),
                             Eigen::Vector3d::Zero());
    for (int i = 0; i < 5; ++i)
        detector.observe_imu(0.300 + 0.001 * i, Eigen::Vector3d(0.0, 0.0, 9.81007),
                             Eigen::Vector3d::Zero());
    const LandingFrameResult result =
        detector.observe_frame(0.310, static_grid(), Eigen::Vector3d(0.05, 0.0, 0.0),
                               Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    EXPECT_FALSE(result.active);
}

TEST(LandingStaticDetector, MotionExitRequiresFullRearm)
{
    LandingStaticDetector detector;
    detector.configure(test_config());
    detector.observe_imu(0.0, Eigen::Vector3d(0.0, 0.0, 25.0), Eigen::Vector3d::Zero());

    const auto observations = static_grid();
    double imu_t = 0.005;
    double t = 0.10;
    bool entered = false;
    for (int i = 0; i < 24; ++i)
    {
        t = 0.10 + 0.05 * i;
        const LandingFrameResult result =
            feed_frame(detector, t, observations, Eigen::Vector3d(0.05, 0.0, 0.0), imu_t);
        entered = entered || result.entered;
        if (entered)
            break;
    }
    ASSERT_TRUE(entered);
    const LandingFrameResult moving = feed_frame(
        detector, t + 0.05, static_grid(2.0), Eigen::Vector3d(0.05, 0.0, 0.0), imu_t);
    ASSERT_TRUE(moving.exited);

    // Six static frames = 0.30 s < 0.5 s must not re-enter.
    double last = t + 0.05;
    for (int i = 0; i < 6; ++i)
    {
        last += 0.05;
        const LandingFrameResult result =
            feed_frame(detector, last, static_grid(), Eigen::Vector3d(0.05, 0.0, 0.0), imu_t);
        EXPECT_FALSE(result.active);
    }
    // Six more frames complete (and slightly exceed) the 0.5 s re-arm window.
    bool reentered = false;
    for (int i = 0; i < 6; ++i)
    {
        last += 0.05;
        const LandingFrameResult result =
            feed_frame(detector, last, static_grid(), Eigen::Vector3d(0.05, 0.0, 0.0), imu_t);
        reentered = reentered || result.active;
    }
    EXPECT_TRUE(reentered);
}

TEST(LandingStaticDetector, ResetClearsEvidence)
{
    LandingStaticDetector detector;
    detector.configure(test_config());
    detector.observe_imu(0.0, Eigen::Vector3d(0.0, 0.0, 25.0), Eigen::Vector3d::Zero());

    const auto observations = static_grid();
    double imu_t = 0.005;
    double t = 0.10;
    bool entered = false;
    for (int i = 0; i < 24; ++i)
    {
        t = 0.10 + 0.05 * i;
        const LandingFrameResult result =
            feed_frame(detector, t, observations, Eigen::Vector3d(0.05, 0.0, 0.0), imu_t);
        entered = entered || result.entered;
        if (entered)
            break;
    }
    ASSERT_TRUE(entered);
    detector.reset();
    EXPECT_FALSE(detector.active());

    const LandingFrameResult after_reset =
        feed_frame(detector, t + 0.05, observations, Eigen::Vector3d(0.05, 0.0, 0.0), imu_t);
    EXPECT_FALSE(after_reset.active);
}

TEST(LandingStaticDetector, DefaultConfigIsDisabled)
{
    LandingStaticDetector detector;  // safe default: disabled
    EXPECT_FALSE(detector.config().enabled);
    detector.observe_imu(0.0, Eigen::Vector3d(0.0, 0.0, 25.0), Eigen::Vector3d::Zero());

    const auto observations = static_grid();
    double imu_t = 0.005;
    for (int i = 0; i <= 24; ++i)
    {
        const LandingFrameResult result =
            feed_frame(detector, 0.10 + 0.05 * i, observations,
                       Eigen::Vector3d(0.05, 0.0, 0.0), imu_t);
        EXPECT_FALSE(result.active);
    }
    EXPECT_FALSE(detector.active());
}

TEST(LandingStaticDetector, StillTimeBoundaryUsesEpsilon)
{
    LandingStaticDetector detector;
    LandingStaticConfig config = test_config();
    config.min_still_seconds = 0.3;
    config.max_image_gap_seconds = 0.5;
    config.imu_window_seconds = 0.1;
    detector.configure(config);
    detector.observe_imu(0.0, Eigen::Vector3d(0.0, 0.0, 25.0), Eigen::Vector3d::Zero());

    const auto observations = static_grid();
    double imu_t = 0.005;
    // The first frame establishes the previous image. The second frame starts
    // the stillness timer at t=0.20.
    feed_frame(detector, 0.10, observations, Eigen::Vector3d(0.05, 0.0, 0.0), imu_t);
    const LandingFrameResult started =
        feed_frame(detector, 0.20, observations, Eigen::Vector3d(0.05, 0.0, 0.0), imu_t);
    ASSERT_FALSE(started.active);

    // 0.299998 s is outside the epsilon; 0.2999995 s is inside it and must
    // activate. Without the tolerance the exact 0.3 s boundary can require an
    // extra image frame due to binary floating-point accumulation.
    const LandingFrameResult below =
        feed_frame(detector, 0.499998, observations, Eigen::Vector3d(0.05, 0.0, 0.0), imu_t);
    EXPECT_FALSE(below.active);
    const LandingFrameResult boundary =
        feed_frame(detector, 0.4999995, observations, Eigen::Vector3d(0.05, 0.0, 0.0), imu_t);
    EXPECT_TRUE(boundary.active);
    EXPECT_TRUE(boundary.entered);
}

TEST(LandingZeroVelocityUpdate, JosephUpdateReducesVelocityAndKeepsPsd)
{
    EkfOptions options;
    options.window_size = 2;
    EkfState state;
    state.reset(options, 10.0, Eigen::Vector4d(1, 0, 0, 0), Eigen::Vector3d::Zero(),
                Eigen::Vector3d(0.5, -0.3, 0.2), Eigen::Vector3d::Zero(),
                Eigen::Vector3d::Zero());
    state.augment_clone(10.0);
    state.augment_clone(10.1);
    const Eigen::MatrixXd P_before = state.covariance();

    const ZeroVelocityUpdateResult result = applyZeroVelocityUpdate(state, 0.02);
    ASSERT_TRUE(result.applied);
    EXPECT_LT(state.imu().v.norm(), 0.1);
    EXPECT_LT(state.imu().v.norm(), Eigen::Vector3d(0.5, -0.3, 0.2).norm());
    EXPECT_TRUE(state.covariance().allFinite());
    EXPECT_TRUE(EkfState::is_positive_semidefinite(state.covariance()));
    EXPECT_EQ(state.num_clones(), 2);
    const double before_trace = P_before.block<3, 3>(kImuV, kImuV).trace();
    const double after_trace =
        state.covariance().block<3, 3>(kImuV, kImuV).trace();
    EXPECT_LT(after_trace, before_trace);
}

}  // namespace
