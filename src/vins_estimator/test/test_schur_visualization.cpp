#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "backend/feature_history.h"
#include "backend/schur_visualization.h"

namespace
{
using namespace vins::backend;

UpdateOptions_s visualization_options()
{
  UpdateOptions_s options;
  options.min_observations_ = 2;
  options.triangulation_iterations_ = 5;
  options.min_depth_ = 0.1;
  options.max_depth_ = 60.0;
  return options;
}

FeatureObservation_s
observation(int _clone_index, const Eigen::Vector2d & _uv, const Eigen::Vector2d & _pixel_uv, bool _has_pixel)
{
  FeatureObservation_s result;
  result.clone_index_ = _clone_index;
  result.uv_ = _uv;
  result.pixel_uv_ = _pixel_uv;
  result.has_pixel_uv_ = _has_pixel;
  return result;
}

EkfState two_clone_state()
{
  EkfOptions_s options;
  options.window_size_ = 4;
  EkfState state;
  state.reset(
    options, 0.0, Eigen::Vector4d(1, 0, 0, 0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
    Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  state.augmentClone(0.0);
  state.imu().p_ = Eigen::Vector3d(1.0, 0.0, 0.0);
  state.refreshImuFej();
  state.augmentClone(1.0);
  return state;
}
}  // namespace

TEST(SchurVisualization, TriangulatesCopiesAndKeepsKeyframePixelObservations)
{
  const EkfState state = two_clone_state();
  const std::vector<CameraExtrinsics_s> cameras(1);
  const SchurUpdater updater(visualization_options());

  // A static world point at (0, 0, 5) observed after the camera moved +1 m in x.
  FeatureTrack_s track;
  track.id_ = 7;
  track.observations_.push_back(observation(0, Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(320.0, 240.0), true));
  track.observations_.push_back(observation(1, Eigen::Vector2d(-0.2, 0.0), Eigen::Vector2d(228.0, 240.0), true));
  const FeatureTrack_s original = track;

  const SchurVisualizationSnapshot_s snapshot =
    makeSchurVisualizationSnapshot(state, cameras, updater, {track}, 1, 12.5, {7});

  ASSERT_TRUE(snapshot.valid_);
  ASSERT_EQ(snapshot.cloud_points_.size(), 1u);
  EXPECT_TRUE(snapshot.margin_points_.empty());
  ASSERT_EQ(snapshot.keyframe_points_.size(), 1u);
  EXPECT_TRUE(snapshot.cloud_points_[0].world_point_.allFinite());
  EXPECT_NEAR(snapshot.cloud_points_[0].world_point_.x(), 0.0, 1e-6);
  EXPECT_NEAR(snapshot.cloud_points_[0].world_point_.y(), 0.0, 1e-6);
  EXPECT_NEAR(snapshot.cloud_points_[0].world_point_.z(), 5.0, 1e-6);
  EXPECT_NEAR(snapshot.keyframe_points_[0].normalized_uv_.x(), -0.2, 1e-12);
  EXPECT_NEAR(snapshot.keyframe_points_[0].pixel_uv_.x(), 228.0, 1e-12);

  // The publication path must never turn a copied track into measurement state.
  EXPECT_FALSE(track.has_point_);
  EXPECT_TRUE(track.world_point_.isZero());
  EXPECT_EQ(track.observations_.size(), original.observations_.size());
}

TEST(SchurVisualization, RequiresCurrentFramePixelsAndRejectsUnacceptedMargin)
{
  const EkfState state = two_clone_state();
  const std::vector<CameraExtrinsics_s> cameras(1);
  const SchurUpdater updater(visualization_options());

  FeatureTrack_s no_pixel;
  no_pixel.id_ = 9;
  no_pixel.observations_.push_back(observation(0, Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d::Zero(), false));
  no_pixel.observations_.push_back(observation(1, Eigen::Vector2d(-0.2, 0.0), Eigen::Vector2d::Zero(), false));

  FeatureTrack_s too_short;
  too_short.id_ = 10;
  too_short.observations_.push_back(
    observation(0, Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(320.0, 240.0), true));

  const SchurVisualizationSnapshot_s snapshot =
    makeSchurVisualizationSnapshot(state, cameras, updater, {no_pixel, too_short}, 1, 13.0);

  ASSERT_TRUE(snapshot.valid_);
  ASSERT_EQ(snapshot.cloud_points_.size(), 1u);
  EXPECT_EQ(snapshot.cloud_points_[0].feature_id_, 9);
  EXPECT_TRUE(snapshot.keyframe_points_.empty());
  EXPECT_TRUE(snapshot.margin_points_.empty());
}

TEST(SchurVisualization, CloneZeroAloneDoesNotQualifyForMargin)
{
  const EkfState state = two_clone_state();
  const std::vector<CameraExtrinsics_s> cameras(1);
  const SchurUpdater updater(visualization_options());

  FeatureTrack_s track;
  track.id_ = 11;
  track.observations_.push_back(observation(0, Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(320.0, 240.0), true));
  track.observations_.push_back(observation(1, Eigen::Vector2d(-0.2, 0.0), Eigen::Vector2d(228.0, 240.0), true));

  const SchurVisualizationSnapshot_s snapshot =
    makeSchurVisualizationSnapshot(state, cameras, updater, {track}, 1, 14.0);
  ASSERT_TRUE(snapshot.valid_);
  EXPECT_EQ(snapshot.cloud_points_.size(), 1u);
  EXPECT_TRUE(snapshot.margin_points_.empty());
}

TEST(FeatureHistoryPublication, PreservesRawPixelsAndResolvesCloneIndices)
{
  FeatureHistory history;
  history.addFrameWithPixels(
    20, 0,
    {FeatureHistory::FrameObservation_s{1, Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(320.0, 240.0), true}});
  history.addFrameWithPixels(
    21, 0,
    {FeatureHistory::FrameObservation_s{1, Eigen::Vector2d(-0.2, 0.0), Eigen::Vector2d(228.0, 240.0), true}});

  const std::vector<FeatureTrack_s> tracks = history.allTracks(20);
  ASSERT_EQ(tracks.size(), 1u);
  ASSERT_EQ(tracks[0].observations_.size(), 2u);
  EXPECT_EQ(tracks[0].observations_[0].clone_index_, 0);
  EXPECT_EQ(tracks[0].observations_[1].clone_index_, 1);
  EXPECT_TRUE(tracks[0].observations_[1].has_pixel_uv_);
  EXPECT_NEAR(tracks[0].observations_[1].pixel_uv_.x(), 228.0, 1e-12);
}


namespace
{
struct MarginScene_s {
  EkfState state_;
  std::vector<CameraExtrinsics_s> cameras_{CameraExtrinsics_s{}};
  FeatureTrack_s track_;

  explicit MarginScene_s(double _baseline = 1.0, double _depth = 5.0)
  {
    EkfOptions_s options;
    options.window_size_ = 3;
    state_.reset(
      options, 0.0, Eigen::Vector4d(1, 0, 0, 0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    track_.id_ = 42;
    for (int i = 0; i < 3; ++i) {
      state_.imu().p_ = Eigen::Vector3d(i * _baseline / 2.0, 0, 0);
      state_.refreshImuFej();
      state_.augmentClone(i);
      const Eigen::Vector2d uv(-state_.imu().p_.x() / _depth, 0);
      track_.observations_.push_back(observation(i, uv, uv * 460.0 + Eigen::Vector2d(320, 240), true));
    }
  }

  SchurVisualizationSnapshot_s
  snapshot(const std::vector<int> & _accepted = {42}, MarginPublicationHistory * _history = nullptr) const
  {
    return makeSchurVisualizationSnapshot(
      state_, cameras_, SchurUpdater(visualization_options()), {track_}, 2, 2.0, _accepted, _history);
  }
};
}  // namespace

TEST(SchurMargin, OnlyAcceptedConsumedTracksQualify)
{
  MarginScene_s scene;
  const auto rejected = scene.snapshot({});
  const auto accepted = scene.snapshot();
  ASSERT_EQ(accepted.margin_points_.size(), 1u);
  EXPECT_TRUE(rejected.margin_points_.empty());
  ASSERT_EQ(rejected.cloud_points_.size(), 1u);
  ASSERT_EQ(rejected.keyframe_points_.size(), 1u);
  EXPECT_TRUE(rejected.cloud_points_[0].world_point_.isApprox(accepted.cloud_points_[0].world_point_));
}

TEST(SchurMargin, RejectsWeakParallaxAndHighRelativeDepthUncertaintyOnlyForMargin)
{
  // 0.57 degree: insufficient parallax even though triangulation is exact.
  MarginScene_s weak(0.05);
  // 1.37 degrees passes the angle gate, but 1.5 px noise gives >15% depth sigma.
  MarginScene_s uncertain(0.12);
  for (const MarginScene_s * scene : {&weak, &uncertain}) {
    const auto result = scene->snapshot();
    ASSERT_EQ(result.cloud_points_.size(), 1u);
    ASSERT_EQ(result.keyframe_points_.size(), 1u);
    EXPECT_TRUE(result.margin_points_.empty());
  }
  // Large distance alone does not reject a well constrained point.
  MarginScene_s far_but_precise(10.0, 50.0);
  EXPECT_EQ(far_but_precise.snapshot().margin_points_.size(), 1u);
}

TEST(SchurMargin, RejectsReprojectionTailThatPassesOrdinaryCloudGate)
{
  MarginScene_s scene;
  // The middle observation has a 6 px orthogonal error. Refined RMS remains
  // below the cloud's 5 px gate, but exceeds the margin's 2 px RMS / 3 px max.
  scene.track_.observations_[1].uv_.y() += 6.0 / 460.0;
  const auto result = scene.snapshot();
  ASSERT_EQ(result.cloud_points_.size(), 1u);
  ASSERT_EQ(result.keyframe_points_.size(), 1u);
  EXPECT_TRUE(result.margin_points_.empty());
}

TEST(SchurMargin, PublishesOnceAcrossConsumedSegmentsAndBoundsLifetime)
{
  MarginScene_s scene;
  MarginPublicationHistory history;
  history.retain({42});
  EXPECT_EQ(scene.snapshot({42}, &history).margin_points_.size(), 1u);
  // Accepted track erased and rebuilt by EKF, same feature still tracked.
  history.retain({42});
  const auto repeated = scene.snapshot({42}, &history);
  EXPECT_TRUE(repeated.margin_points_.empty());
  EXPECT_EQ(repeated.cloud_points_.size(), 1u);
  EXPECT_EQ(repeated.keyframe_points_.size(), 1u);
  EXPECT_EQ(history.size(), 1u);
  // No current image or retained history can refer to this lifetime anymore.
  history.retain({});
  EXPECT_EQ(history.size(), 0u);
  EXPECT_EQ(scene.snapshot({42}, &history).margin_points_.size(), 1u);
  history.clear();
  EXPECT_EQ(history.size(), 0u);
  // Long runs cannot accumulate an unbounded set of historical feature IDs.
  for (int id = 0; id < 10000; ++id) {
    history.retain({id});
    EXPECT_TRUE(history.mark(id));
    EXPECT_EQ(history.size(), 1u);
  }
}

TEST(SchurMargin, PoorFirstSegmentDoesNotBlockLaterReliableSegmentOrMutateFilter)
{
  MarginScene_s weak(0.05), good;
  MarginPublicationHistory history;
  const auto covariance = good.state_.covariance();
  const auto pose = good.state_.imu().p_;
  history.retain({42});
  EXPECT_TRUE(weak.snapshot({42}, &history).margin_points_.empty());
  EXPECT_EQ(history.size(), 0u);
  EXPECT_EQ(good.snapshot({42}, &history).margin_points_.size(), 1u);
  EXPECT_TRUE((good.state_.covariance().array() == covariance.array()).all());
  EXPECT_TRUE((good.state_.imu().p_.array() == pose.array()).all());
  EXPECT_FALSE(good.track_.has_point_);
  EXPECT_TRUE(good.track_.world_point_.isZero());
}
