#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "backend/feature_history.h"
#include "backend/schur_visualization.h"

namespace
{
using namespace vins::backend;

UpdateOptions visualization_options()
{
    UpdateOptions options;
    options.min_observations = 2;
    options.triangulation_iterations = 5;
    options.min_depth = 0.1;
    options.max_depth = 60.0;
    return options;
}

FeatureObservation observation(int clone_index, const Eigen::Vector2d &uv,
                               const Eigen::Vector2d &pixel_uv, bool has_pixel)
{
    FeatureObservation result;
    result.clone_index = clone_index;
    result.uv = uv;
    result.pixel_uv = pixel_uv;
    result.has_pixel_uv = has_pixel;
    return result;
}

EkfState two_clone_state()
{
    EkfOptions options;
    options.window_size = 4;
    EkfState state;
    state.reset(options, 0.0, Eigen::Vector4d(1, 0, 0, 0), Eigen::Vector3d::Zero(),
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    state.augment_clone(0.0);
    state.imu().p = Eigen::Vector3d(1.0, 0.0, 0.0);
    state.refresh_imu_fej();
    state.augment_clone(1.0);
    return state;
}
}  // namespace

TEST(SchurVisualization, TriangulatesCopiesAndKeepsKeyframePixelObservations)
{
    const EkfState state = two_clone_state();
    const std::vector<CameraExtrinsics> cameras(1);
    const SchurUpdater updater(visualization_options());

    // A static world point at (0, 0, 5) observed after the camera moved +1 m in x.
    FeatureTrack track;
    track.id = 7;
    track.observations.push_back(observation(0, Eigen::Vector2d(0.0, 0.0),
                                             Eigen::Vector2d(320.0, 240.0), true));
    track.observations.push_back(observation(1, Eigen::Vector2d(-0.2, 0.0),
                                             Eigen::Vector2d(228.0, 240.0), true));
    const FeatureTrack original = track;

    const SchurVisualizationSnapshot snapshot = makeSchurVisualizationSnapshot(
        state, cameras, updater, {track}, 1, 12.5, {7});

    ASSERT_TRUE(snapshot.valid);
    ASSERT_EQ(snapshot.cloud_points.size(), 1u);
    EXPECT_TRUE(snapshot.margin_points.empty());
    ASSERT_EQ(snapshot.keyframe_points.size(), 1u);
    EXPECT_TRUE(snapshot.cloud_points[0].world_point.allFinite());
    EXPECT_NEAR(snapshot.cloud_points[0].world_point.x(), 0.0, 1e-6);
    EXPECT_NEAR(snapshot.cloud_points[0].world_point.y(), 0.0, 1e-6);
    EXPECT_NEAR(snapshot.cloud_points[0].world_point.z(), 5.0, 1e-6);
    EXPECT_NEAR(snapshot.keyframe_points[0].normalized_uv.x(), -0.2, 1e-12);
    EXPECT_NEAR(snapshot.keyframe_points[0].pixel_uv.x(), 228.0, 1e-12);

    // The publication path must never turn a copied track into measurement state.
    EXPECT_FALSE(track.has_point);
    EXPECT_TRUE(track.world_point.isZero());
    EXPECT_EQ(track.observations.size(), original.observations.size());
}

TEST(SchurVisualization, RequiresCurrentFramePixelsAndRejectsUnacceptedMargin)
{
    const EkfState state = two_clone_state();
    const std::vector<CameraExtrinsics> cameras(1);
    const SchurUpdater updater(visualization_options());

    FeatureTrack no_pixel;
    no_pixel.id = 9;
    no_pixel.observations.push_back(observation(0, Eigen::Vector2d(0.0, 0.0),
                                                Eigen::Vector2d::Zero(), false));
    no_pixel.observations.push_back(observation(1, Eigen::Vector2d(-0.2, 0.0),
                                                Eigen::Vector2d::Zero(), false));

    FeatureTrack too_short;
    too_short.id = 10;
    too_short.observations.push_back(observation(0, Eigen::Vector2d(0.0, 0.0),
                                                 Eigen::Vector2d(320.0, 240.0), true));

    const SchurVisualizationSnapshot snapshot = makeSchurVisualizationSnapshot(
        state, cameras, updater, {no_pixel, too_short}, 1, 13.0);

    ASSERT_TRUE(snapshot.valid);
    ASSERT_EQ(snapshot.cloud_points.size(), 1u);
    EXPECT_EQ(snapshot.cloud_points[0].feature_id, 9);
    EXPECT_TRUE(snapshot.keyframe_points.empty());
    EXPECT_TRUE(snapshot.margin_points.empty());
}

TEST(SchurVisualization, CloneZeroAloneDoesNotQualifyForMargin)
{
    const EkfState state = two_clone_state();
    const std::vector<CameraExtrinsics> cameras(1);
    const SchurUpdater updater(visualization_options());

    FeatureTrack track;
    track.id = 11;
    track.observations.push_back(observation(0, Eigen::Vector2d(0.0, 0.0),
                                             Eigen::Vector2d(320.0, 240.0), true));
    track.observations.push_back(observation(1, Eigen::Vector2d(-0.2, 0.0),
                                             Eigen::Vector2d(228.0, 240.0), true));

    const SchurVisualizationSnapshot snapshot = makeSchurVisualizationSnapshot(
        state, cameras, updater, {track}, 1, 14.0);
    ASSERT_TRUE(snapshot.valid);
    EXPECT_EQ(snapshot.cloud_points.size(), 1u);
    EXPECT_TRUE(snapshot.margin_points.empty());
}

TEST(FeatureHistoryPublication, PreservesRawPixelsAndResolvesCloneIndices)
{
    FeatureHistory history;
    history.add_frame_with_pixels(
        20, 0,
        {FeatureHistory::FrameObservation{1, Eigen::Vector2d(0.0, 0.0),
                                          Eigen::Vector2d(320.0, 240.0), true}});
    history.add_frame_with_pixels(
        21, 0,
        {FeatureHistory::FrameObservation{1, Eigen::Vector2d(-0.2, 0.0),
                                          Eigen::Vector2d(228.0, 240.0), true}});

    const std::vector<FeatureTrack> tracks = history.all_tracks(20);
    ASSERT_EQ(tracks.size(), 1u);
    ASSERT_EQ(tracks[0].observations.size(), 2u);
    EXPECT_EQ(tracks[0].observations[0].clone_index, 0);
    EXPECT_EQ(tracks[0].observations[1].clone_index, 1);
    EXPECT_TRUE(tracks[0].observations[1].has_pixel_uv);
    EXPECT_NEAR(tracks[0].observations[1].pixel_uv.x(), 228.0, 1e-12);
}


namespace
{
struct MarginScene
{
    EkfState state;
    std::vector<CameraExtrinsics> cameras{CameraExtrinsics{}};
    FeatureTrack track;

    explicit MarginScene(double baseline = 1.0, double depth = 5.0)
    {
        EkfOptions options;
        options.window_size = 3;
        state.reset(options, 0.0, Eigen::Vector4d(1, 0, 0, 0), Eigen::Vector3d::Zero(),
                    Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
        track.id = 42;
        for (int i = 0; i < 3; ++i)
        {
            state.imu().p = Eigen::Vector3d(i * baseline / 2.0, 0, 0);
            state.refresh_imu_fej();
            state.augment_clone(i);
            const Eigen::Vector2d uv(-state.imu().p.x() / depth, 0);
            track.observations.push_back(observation(i, uv, uv * 460.0 + Eigen::Vector2d(320, 240), true));
        }
    }

    SchurVisualizationSnapshot snapshot(const std::vector<int> &accepted = {42},
                                        MarginPublicationHistory *history = nullptr) const
    {
        return makeSchurVisualizationSnapshot(state, cameras,
            SchurUpdater(visualization_options()), {track}, 2, 2.0, accepted, history);
    }
};
}

TEST(SchurMargin, OnlyAcceptedConsumedTracksQualify)
{
    MarginScene scene;
    const auto rejected = scene.snapshot({});
    const auto accepted = scene.snapshot();
    ASSERT_EQ(accepted.margin_points.size(), 1u);
    EXPECT_TRUE(rejected.margin_points.empty());
    ASSERT_EQ(rejected.cloud_points.size(), 1u);
    ASSERT_EQ(rejected.keyframe_points.size(), 1u);
    EXPECT_TRUE(rejected.cloud_points[0].world_point.isApprox(accepted.cloud_points[0].world_point));
}

TEST(SchurMargin, RejectsWeakParallaxAndHighRelativeDepthUncertaintyOnlyForMargin)
{
    // 0.57 degree: insufficient parallax even though triangulation is exact.
    MarginScene weak(0.05);
    // 1.37 degrees passes the angle gate, but 1.5 px noise gives >15% depth sigma.
    MarginScene uncertain(0.12);
    for (const MarginScene *scene : {&weak, &uncertain})
    {
        const auto result = scene->snapshot();
        ASSERT_EQ(result.cloud_points.size(), 1u);
        ASSERT_EQ(result.keyframe_points.size(), 1u);
        EXPECT_TRUE(result.margin_points.empty());
    }
    // Large distance alone does not reject a well constrained point.
    MarginScene far_but_precise(10.0, 50.0);
    EXPECT_EQ(far_but_precise.snapshot().margin_points.size(), 1u);
}

TEST(SchurMargin, RejectsReprojectionTailThatPassesOrdinaryCloudGate)
{
    MarginScene scene;
    // The middle observation has a 6 px orthogonal error. Refined RMS remains
    // below the cloud's 5 px gate, but exceeds the margin's 2 px RMS / 3 px max.
    scene.track.observations[1].uv.y() += 6.0 / 460.0;
    const auto result = scene.snapshot();
    ASSERT_EQ(result.cloud_points.size(), 1u);
    ASSERT_EQ(result.keyframe_points.size(), 1u);
    EXPECT_TRUE(result.margin_points.empty());
}

TEST(SchurMargin, PublishesOnceAcrossConsumedSegmentsAndBoundsLifetime)
{
    MarginScene scene;
    MarginPublicationHistory history;
    history.retain({42});
    EXPECT_EQ(scene.snapshot({42}, &history).margin_points.size(), 1u);
    // Accepted track erased and rebuilt by EKF, same feature still tracked.
    history.retain({42});
    const auto repeated = scene.snapshot({42}, &history);
    EXPECT_TRUE(repeated.margin_points.empty());
    EXPECT_EQ(repeated.cloud_points.size(), 1u);
    EXPECT_EQ(repeated.keyframe_points.size(), 1u);
    EXPECT_EQ(history.size(), 1u);
    // No current image or retained history can refer to this lifetime anymore.
    history.retain({});
    EXPECT_EQ(history.size(), 0u);
    EXPECT_EQ(scene.snapshot({42}, &history).margin_points.size(), 1u);
    history.clear();
    EXPECT_EQ(history.size(), 0u);
    // Long runs cannot accumulate an unbounded set of historical feature IDs.
    for (int id = 0; id < 10000; ++id)
    {
        history.retain({id});
        EXPECT_TRUE(history.mark(id));
        EXPECT_EQ(history.size(), 1u);
    }
}

TEST(SchurMargin, PoorFirstSegmentDoesNotBlockLaterReliableSegmentOrMutateFilter)
{
    MarginScene weak(0.05), good;
    MarginPublicationHistory history;
    const auto covariance = good.state.covariance();
    const auto pose = good.state.imu().p;
    history.retain({42});
    EXPECT_TRUE(weak.snapshot({42}, &history).margin_points.empty());
    EXPECT_EQ(history.size(), 0u);
    EXPECT_EQ(good.snapshot({42}, &history).margin_points.size(), 1u);
    EXPECT_TRUE((good.state.covariance().array() == covariance.array()).all());
    EXPECT_TRUE((good.state.imu().p.array() == pose.array()).all());
    EXPECT_FALSE(good.track.has_point);
    EXPECT_TRUE(good.track.world_point.isZero());
}
