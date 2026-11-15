#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "backend/feature_history.h"
#include "backend/schur_cloud_history.h"
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

FeatureObservation_s observation(
  int _clone_index, int _camera_id, const Eigen::Vector2d & _uv, const Eigen::Vector2d & _pixel_uv,
  bool _has_pixel)
{
  FeatureObservation_s result = observation(_clone_index, _uv, _pixel_uv, _has_pixel);
  result.camera_id_ = _camera_id;
  return result;
}

Eigen::Vector2d project_world(
  const EkfState & _state, int _clone_index, const CameraExtrinsics_s & _camera,
  const Eigen::Vector3d & _world_point)
{
  const auto & clone = _state.clone(_clone_index);
  const Eigen::Matrix3d R_i = EkfState::rotation(clone.q_);
  const Eigen::Matrix3d R_cb = EkfState::rotation(_camera.q_cb_);
  const Eigen::Vector3d p_c = R_cb.transpose() * (R_i.transpose() * (_world_point - clone.p_) - _camera.p_cb_);
  return Eigen::Vector2d(p_c.x() / p_c.z(), p_c.y() / p_c.z());
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

TEST(SchurVisualization, StereoKeyframePointsUseCameraZeroOrAreOmitted)
{
  const EkfState state = two_clone_state();
  const CameraExtrinsics_s cam0;
  CameraExtrinsics_s cam1;
  cam1.p_cb_ = Eigen::Vector3d(-0.11, 0.0, 0.0);
  const std::vector<CameraExtrinsics_s> cameras{cam0, cam1};
  const SchurUpdater updater(visualization_options());

  const Eigen::Vector3d world_point(0.0, 0.0, 5.0);
  const Eigen::Vector2d uv00 = project_world(state, 0, cam0, world_point);
  const Eigen::Vector2d uv10 = project_world(state, 1, cam0, world_point);
  const Eigen::Vector2d uv11 = project_world(state, 1, cam1, world_point);

  FeatureTrack_s both_cameras;
  both_cameras.id_ = 21;
  both_cameras.observations_.push_back(observation(0, 0, uv00, Eigen::Vector2d(320.0, 240.0), true));
  both_cameras.observations_.push_back(observation(1, 0, uv10, Eigen::Vector2d(228.0, 240.0), true));
  both_cameras.observations_.push_back(observation(1, 1, uv11, Eigen::Vector2d(999.0, 777.0), true));

  FeatureTrack_s right_only_at_keyframe;
  right_only_at_keyframe.id_ = 22;
  right_only_at_keyframe.observations_.push_back(observation(0, 0, uv00, Eigen::Vector2d(320.0, 240.0), true));
  right_only_at_keyframe.observations_.push_back(observation(1, 1, uv11, Eigen::Vector2d(238.0, 240.0), true));

  const SchurVisualizationSnapshot_s snapshot =
    makeSchurVisualizationSnapshot(state, cameras, updater, {both_cameras, right_only_at_keyframe}, 1, 15.0);

  ASSERT_TRUE(snapshot.valid_);
  ASSERT_EQ(snapshot.cloud_points_.size(), 2u);
  ASSERT_EQ(snapshot.keyframe_points_.size(), 1u);

  // The loop-fusion point must be camera 0 even when a right-camera view also
  // exists at the keyframe clone.
  EXPECT_EQ(snapshot.keyframe_points_[0].feature_id_, 21);
  EXPECT_NEAR(snapshot.keyframe_points_[0].pixel_uv_.x(), 228.0, 1e-12);
  EXPECT_NEAR(snapshot.keyframe_points_[0].pixel_uv_.y(), 240.0, 1e-12);

  // A right-only keyframe view still contributes a world point, but no
  // keyframe point. Its cloud metadata falls back to the last camera-0 view,
  // never the right-camera pixel.
  EXPECT_EQ(snapshot.cloud_points_[1].feature_id_, 22);
  EXPECT_NEAR(snapshot.cloud_points_[1].pixel_uv_.x(), 320.0, 1e-12);
  EXPECT_NEAR(snapshot.cloud_points_[1].pixel_uv_.y(), 240.0, 1e-12);
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


TEST(SchurCloudHistory, SurvivesConsumptionAndPrunesByAbsoluteFrame)
{
  FeatureHistory measurement;
  SchurCloudHistory cloud;
  cloud.configure(64, 32);

  measurement.addFrame(20, 0, {std::make_pair(1, Eigen::Vector2d(0.0, 0.0))});
  cloud.addFrame(20, 0, {std::make_pair(1, Eigen::Vector2d(0.0, 0.0))});
  measurement.addFrame(21, 0, {std::make_pair(1, Eigen::Vector2d(-0.2, 0.0))});
  cloud.addFrame(21, 0, {std::make_pair(1, Eigen::Vector2d(-0.2, 0.0))});

  // Simulate the Schur update consuming the track. Measurement history forgets
  // it, publication history must not.
  measurement.eraseFeatures({1});
  measurement.addFrame(22, 0, {std::make_pair(1, Eigen::Vector2d(-0.4, 0.0))});
  cloud.addFrame(22, 0, {std::make_pair(1, Eigen::Vector2d(-0.4, 0.0))});

  EXPECT_TRUE(measurement.allTracks(20).at(0).observations_.size() == 1u);
  const auto cloud_tracks = cloud.tracks(20, 10);
  ASSERT_EQ(cloud_tracks.size(), 1u);
  ASSERT_EQ(cloud_tracks[0].observations_.size(), 3u);
  EXPECT_EQ(cloud_tracks[0].observations_[0].clone_index_, 0);
  EXPECT_EQ(cloud_tracks[0].observations_[2].clone_index_, 2);

  // Sliding the clone window rebases clone indices without a remap.
  cloud.prune(21);
  const auto pruned = cloud.tracks(21, 10);
  ASSERT_EQ(pruned.size(), 1u);
  ASSERT_EQ(pruned[0].observations_.size(), 2u);
  EXPECT_EQ(pruned[0].observations_[0].clone_index_, 0);
  EXPECT_EQ(pruned[0].observations_[1].clone_index_, 1);

  // A single remaining observation is no longer a triangulatable track.
  cloud.prune(22);
  EXPECT_TRUE(cloud.tracks(22, 10).empty());
  EXPECT_EQ(cloud.numFeatures(), 1u);
  cloud.prune(23);
  EXPECT_EQ(cloud.numFeatures(), 0u);
}

TEST(SchurCloudHistory, EnforcesCapsAndIgnoresDuplicateViews)
{
  SchurCloudHistory cloud;
  cloud.configure(2, 4);
  cloud.addFrameWithPixels(
    0, 0,
    {FeatureHistory::FrameObservation_s{1, Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(320.0, 240.0), true}});
  cloud.addFrameWithPixels(
    0, 0,
    {FeatureHistory::FrameObservation_s{1, Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(320.0, 240.0), true}});
  EXPECT_EQ(cloud.numObservations(), 1u);

  for (int frame = 1; frame < 8; ++frame)
    cloud.addFrameWithPixels(
      frame, 0,
      {FeatureHistory::FrameObservation_s{
        1, Eigen::Vector2d(0.01 * frame, 0.0), Eigen::Vector2d(320.0 + frame, 240.0), true}});
  EXPECT_LE(cloud.numObservations(), 4u);
  const auto bounded = cloud.tracks(0, 10);
  ASSERT_EQ(bounded.size(), 1u);
  ASSERT_EQ(bounded[0].observations_.size(), 4u);
  EXPECT_EQ(bounded[0].observations_.back().clone_index_, 7);

  cloud.addFrameWithPixels(
    0, 1,
    {FeatureHistory::FrameObservation_s{2, Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(320.0, 240.0), true},
     FeatureHistory::FrameObservation_s{3, Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(320.0, 240.0), true}});
  EXPECT_LE(cloud.numFeatures(), 2u);

  cloud.clear();
  EXPECT_EQ(cloud.numFeatures(), 0u);
  EXPECT_EQ(cloud.newestFrameIndex(), -1);
}

TEST(SchurCloudHistory, CloudOnlyTracksNeverBecomeLoopKeyframePoints)
{
  const EkfState state = two_clone_state();
  const std::vector<CameraExtrinsics_s> cameras(1);
  const SchurUpdater updater(visualization_options());
  const Eigen::Vector3d world_point(0.0, 0.0, 5.0);
  const Eigen::Vector2d uv0 = project_world(state, 0, cameras[0], world_point);
  const Eigen::Vector2d uv1 = project_world(state, 1, cameras[0], world_point);

  FeatureTrack_s measurement_track;
  measurement_track.id_ = 21;
  measurement_track.observations_.push_back(observation(0, 0, uv0, Eigen::Vector2d(320.0, 240.0), true));
  measurement_track.observations_.push_back(observation(1, 0, uv1, Eigen::Vector2d(228.0, 240.0), true));

  FeatureTrack_s cloud_only = measurement_track;
  cloud_only.id_ = 99;
  cloud_only.observations_[1].pixel_uv_ = Eigen::Vector2d(999.0, 777.0);

  const SchurVisualizationSnapshot_s snapshot = makeSchurVisualizationSnapshot(
    state, cameras, updater, {measurement_track}, {measurement_track, cloud_only}, 1, 20.0, {21});
  ASSERT_TRUE(snapshot.valid_);
  ASSERT_EQ(snapshot.cloud_points_.size(), 2u);
  ASSERT_EQ(snapshot.keyframe_points_.size(), 1u);
  EXPECT_EQ(snapshot.keyframe_points_[0].feature_id_, 21);
  EXPECT_NEAR(snapshot.keyframe_points_[0].pixel_uv_.x(), 228.0, 1e-12);
}

TEST(SchurCloudQuality, StereoPairSatisfiesTwoViewMarginAndCloudGate)
{
  const EkfState state = two_clone_state();
  const CameraExtrinsics_s cam0;
  CameraExtrinsics_s cam1;
  cam1.p_cb_ = Eigen::Vector3d(-0.11, 0.0, 0.0);
  const std::vector<CameraExtrinsics_s> cameras{cam0, cam1};
  const SchurUpdater updater(visualization_options());
  const Eigen::Vector3d world_point(0.0, 0.0, 2.0);

  FeatureTrack_s track;
  track.id_ = 31;
  track.observations_.push_back(
    observation(1, 0, project_world(state, 1, cam0, world_point), Eigen::Vector2d(320.0, 240.0), true));
  track.observations_.push_back(
    observation(1, 1, project_world(state, 1, cam1, world_point), Eigen::Vector2d(238.0, 240.0), true));

  const SchurVisualizationSnapshot_s snapshot =
    makeSchurVisualizationSnapshot(state, cameras, updater, {track}, {track}, 1, 21.0, {31});
  ASSERT_TRUE(snapshot.valid_);
  ASSERT_EQ(snapshot.cloud_points_.size(), 1u);
  ASSERT_EQ(snapshot.margin_points_.size(), 1u);
  EXPECT_EQ(snapshot.margin_points_[0].feature_id_, 31);
}

TEST(SchurCloudQuality, StereoCloudGateRejectsWrongMatch)
{
  const EkfState state = two_clone_state();
  const CameraExtrinsics_s cam0;
  CameraExtrinsics_s cam1;
  cam1.p_cb_ = Eigen::Vector3d(-0.11, 0.0, 0.0);
  const std::vector<CameraExtrinsics_s> cameras{cam0, cam1};
  const SchurUpdater updater(visualization_options());
  const Eigen::Vector3d world_point(0.0, 0.0, 2.0);

  FeatureTrack_s track;
  track.id_ = 32;
  track.observations_.push_back(
    observation(1, 0, project_world(state, 1, cam0, world_point), Eigen::Vector2d(320.0, 240.0), true));
  track.observations_.push_back(
    observation(1, 1, project_world(state, 1, cam1, world_point), Eigen::Vector2d(238.0, 240.0), true));
  // Off-epipolar (y) error. An x error along the epipolar line is not
  // geometrically rejectable from two views alone (it is just a different
  // depth); only cross-frame redundancy can address that ambiguity.
  track.observations_[1].uv_.y() += 0.2;  // ~92 px off-epipolar mismatch

  const SchurVisualizationSnapshot_s snapshot =
    makeSchurVisualizationSnapshot(state, cameras, updater, {track}, {track}, 1, 22.0, {32});
  EXPECT_TRUE(snapshot.cloud_points_.empty());
  EXPECT_TRUE(snapshot.margin_points_.empty());
  EXPECT_TRUE(snapshot.keyframe_points_.empty());
}

TEST(SchurCloudQuality, MarginRetentionIncludesCloudOnlyIds)
{
  SchurCloudHistory cloud;
  cloud.configure(8, 8);
  cloud.addFrameWithPixels(
    0, 0,
    {FeatureHistory::FrameObservation_s{5, Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(320.0, 240.0), true}});

  MarginPublicationHistory history;
  history.mark(5);
  ASSERT_EQ(history.size(), 1u);
  std::unordered_set<int> live_ids;
  cloud.appendFeatureIds(live_ids);
  history.retain(live_ids);
  EXPECT_EQ(history.size(), 1u);
  EXPECT_TRUE(cloud.contains(5));
}


TEST(SchurCloudQuality, RejectsWeakParallaxZeroBaselineAndNegativeDepth)
{
  const SchurUpdater updater(visualization_options());
  SchurVisualizationQuality_s quality;
  quality.gate_all_cloud_tracks_ = true;

  // Weak parallax: 2 cm baseline at 5 m, exact observations. Triangulation
  // succeeds but the publication profiles reject the geometry.
  {
    EkfOptions_s options;
    options.window_size_ = 4;
    EkfState state;
    state.reset(
      options, 0.0, Eigen::Vector4d(1, 0, 0, 0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    state.augmentClone(0.0);
    state.imu().p_ = Eigen::Vector3d(0.02, 0.0, 0.0);
    state.refreshImuFej();
    state.augmentClone(1.0);
    const std::vector<CameraExtrinsics_s> cameras(1);
    const Eigen::Vector3d world_point(0.0, 0.0, 5.0);
    FeatureTrack_s track;
    track.id_ = 41;
    track.observations_.push_back(observation(
      0, 0, project_world(state, 0, cameras[0], world_point), Eigen::Vector2d(320.0, 240.0), true));
    track.observations_.push_back(observation(
      1, 0, project_world(state, 1, cameras[0], world_point), Eigen::Vector2d(318.0, 240.0), true));
    const auto snapshot = makeSchurVisualizationSnapshot(
      state, cameras, updater, {track}, {track}, 1, 30.0, {41}, nullptr, quality);
    EXPECT_TRUE(snapshot.cloud_points_.empty());
    EXPECT_TRUE(snapshot.margin_points_.empty());
    EXPECT_EQ(snapshot.keyframe_points_.size(), 1u);
  }

  // Zero stereo baseline: different camera ids but identical extrinsic pose.
  {
    const EkfState state = two_clone_state();
    const CameraExtrinsics_s cam0;
    const CameraExtrinsics_s cam1 = cam0;
    const std::vector<CameraExtrinsics_s> cameras{cam0, cam1};
    const Eigen::Vector3d world_point(0.0, 0.0, 2.0);
    FeatureTrack_s track;
    track.id_ = 42;
    const Eigen::Vector2d uv = project_world(state, 1, cam0, world_point);
    track.observations_.push_back(observation(1, 0, uv, Eigen::Vector2d(320.0, 240.0), true));
    track.observations_.push_back(observation(1, 1, uv, Eigen::Vector2d(320.0, 240.0), true));
    const auto snapshot = makeSchurVisualizationSnapshot(
      state, cameras, updater, {track}, {track}, 1, 31.0, {42}, nullptr, quality);
    EXPECT_TRUE(snapshot.cloud_points_.empty());
    EXPECT_TRUE(snapshot.margin_points_.empty());
    EXPECT_TRUE(snapshot.keyframe_points_.empty());
  }

  // Negative depth: the second observation is on the opposite side of the
  // epipole, so the two-view initialisation returns lambda <= 0.
  {
    const EkfState state = two_clone_state();
    const std::vector<CameraExtrinsics_s> cameras(1);
    FeatureTrack_s track;
    track.id_ = 43;
    track.observations_.push_back(
      observation(0, 0, Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(320.0, 240.0), true));
    track.observations_.push_back(
      observation(1, 0, Eigen::Vector2d(0.2, 0.0), Eigen::Vector2d(412.0, 240.0), true));
    const auto snapshot = makeSchurVisualizationSnapshot(
      state, cameras, updater, {track}, {track}, 1, 32.0, {43}, nullptr, quality);
    EXPECT_TRUE(snapshot.cloud_points_.empty());
    EXPECT_TRUE(snapshot.margin_points_.empty());
    EXPECT_TRUE(snapshot.keyframe_points_.empty());
  }
}

TEST(SchurCloudQuality, CrossSegmentObservationsReduceSyntheticTriangulationError)
{
  EkfOptions_s options;
  options.window_size_ = 5;
  EkfState state;
  state.reset(
    options, 0.0, Eigen::Vector4d(1, 0, 0, 0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
    Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  for (int i = 0; i < 5; ++i) {
    state.imu().p_ = Eigen::Vector3d(0.2 * i, 0.0, 0.0);
    state.refreshImuFej();
    state.augmentClone(i);
  }

  CameraExtrinsics_s cam0;
  CameraExtrinsics_s cam1;
  cam1.p_cb_ = Eigen::Vector3d(-0.11, 0.0, 0.0);
  const std::vector<CameraExtrinsics_s> cameras{cam0, cam1};
  const SchurUpdater updater(visualization_options());
  const Eigen::Vector3d truth(0.0, 0.0, 4.0);
  std::mt19937 rng(12345);
  std::normal_distribution<double> pixel_noise(0.0, 0.25);
  double short_error = 0.0;
  double full_error = 0.0;
  int trials = 0;

  // No seed selection: all 100 deterministic trials contribute to the mean.
  for (int trial = 0; trial < 100; ++trial) {
    FeatureTrack_s short_track;
    short_track.id_ = 51;
    FeatureTrack_s full_track;
    full_track.id_ = 52;
    for (int i = 0; i < 5; ++i) {
      for (int camera_id = 0; camera_id < 2; ++camera_id) {
        Eigen::Vector2d uv = project_world(state, i, cameras[camera_id], truth);
        uv += Eigen::Vector2d(pixel_noise(rng), pixel_noise(rng)) / 460.0;
        const FeatureObservation_s obs =
          observation(i, camera_id, uv, uv * 460.0 + Eigen::Vector2d(320.0, 240.0), true);
        full_track.observations_.push_back(obs);
        if (i == 4)
          short_track.observations_.push_back(obs);
      }
    }

    FeatureTrack_s short_solution = short_track;
    FeatureTrack_s full_solution = full_track;
    const bool short_ok = updater.triangulate(state, cameras, short_solution);
    const bool full_ok = updater.triangulate(state, cameras, full_solution);
    if (!short_ok || !full_ok)
      continue;
    short_error += (short_solution.world_point_ - truth).norm();
    full_error += (full_solution.world_point_ - truth).norm();
    ++trials;
  }

  ASSERT_GT(trials, 90);
  const double short_mean = short_error / trials;
  const double full_mean = full_error / trials;
  std::cout << "[cloud-history] synthetic stereo trials=" << trials << " short_mean=" << short_mean
            << " full_mean=" << full_mean << std::endl;
  EXPECT_LT(full_mean, short_mean);
}

TEST(SchurFeatureHistory, PreservesCalibrationMeasurementsThroughSelectionAndCloudHistory)
{
  FeatureHistory history;
  SchurCloudHistory cloud;
  FeatureHistory::FrameObservation_s view;
  view.feature_id_ = 731;
  view.uv_ = Eigen::Vector2d(0.12, -0.03);
  view.pixel_uv_ = Eigen::Vector2d(340.0, 222.0);
  view.has_pixel_uv_ = true;
  view.velocity_ = Eigen::Vector2d(0.32, -0.17);
  view.cur_td_ = 0.006;
  history.addFrameWithPixels(10, 0, {view});
  cloud.addFrameWithPixels(10, 0, {view});
  view.cur_td_ = 0.007;
  view.velocity_ *= 0.5;
  history.addFrameWithPixels(11, 0, {view});
  cloud.addFrameWithPixels(11, 0, {view});
  // A lost track enters the actual measurement selection path; the allTracks
  // publication path and the longer-lived cloud history must retain the same
  // acquisition metadata after the measurement history is consumed.
  const auto selected = history.selectTracks(10, 12, 2, 10);
  ASSERT_EQ(selected.tracks_.size(), 1u);
  ASSERT_EQ(selected.tracks_[0].observations_.size(), 2u);
  EXPECT_DOUBLE_EQ(selected.tracks_[0].observations_[0].cur_td_, 0.006);
  EXPECT_TRUE(selected.tracks_[0].observations_[0].velocity_.isApprox(Eigen::Vector2d(0.32, -0.17)));
  EXPECT_DOUBLE_EQ(selected.tracks_[0].observations_[1].cur_td_, 0.007);
  const auto published = history.allTracks(10);
  ASSERT_EQ(published.size(), 1u);
  EXPECT_TRUE(published[0].observations_[1].velocity_.isApprox(view.velocity_));
  history.eraseFeatures({731});
  EXPECT_EQ(history.numFeatures(), 0u);
  const auto retained = cloud.tracks(10, 10);
  ASSERT_EQ(retained.size(), 1u);
  EXPECT_DOUBLE_EQ(retained[0].observations_[0].cur_td_, 0.006);
  EXPECT_TRUE(retained[0].observations_[1].velocity_.isApprox(view.velocity_));
}

TEST(SchurVisualizationCalibration, UsesEstimatedCalibrationForCloudQuality)
{
  EkfOptions_s options;
  options.num_cameras_ = 1;
  options.estimate_extrinsics_ = true;
  options.estimate_time_offset_ = true;
  options.window_size_ = 4;
  CameraCalibrationNominal_s calibration;
  calibration.p_cb_ = Eigen::Vector3d(0.3, 0.1, 0.0);
  calibration.q_cb_ = EkfState::quaternionFromAxisAngle(Eigen::Vector3d(0.0, 0.15, 0.0));
  calibration.td_ = 0.02;
  EkfState state;
  state.reset(
    options, 0.0, Eigen::Vector4d(1, 0, 0, 0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
    Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), {calibration});
  state.augmentClone(0.0);
  state.imu().p_ = Eigen::Vector3d(0.8, 0.0, 0.0);
  state.refreshImuFej();
  state.augmentClone(1.0);
  state.imu().p_ = Eigen::Vector3d(1.6, 0.1, 0.0);
  state.refreshImuFej();
  state.augmentClone(2.0);
  CameraExtrinsics_s estimated;
  estimated.q_cb_ = calibration.q_cb_;
  estimated.p_cb_ = calibration.p_cb_;
  estimated.td_ = calibration.td_;
  FeatureTrack_s track;
  track.id_ = 881;
  const Eigen::Vector3d truth(0.6, 0.2, 4.0);
  for (int i = 0; i < 3; ++i) {
    FeatureObservation_s obs;
    obs.clone_index_ = i;
    obs.velocity_ = Eigen::Vector2d(0.6, -0.3);
    obs.cur_td_ = 0.0;
    obs.uv_ = project_world(state, i, estimated, truth) + calibration.td_ * obs.velocity_;
    obs.pixel_uv_ = obs.uv_ * 460.0 + Eigen::Vector2d(320.0, 240.0);
    obs.has_pixel_uv_ = true;
    track.observations_.push_back(obs);
  }
  const SchurUpdater updater(visualization_options());
  // The legacy caller vector deliberately holds stale config calibration.
  // Triangulation and quality filtering must both use the filter estimate.
  const std::vector<FeatureTrack_s> tracks{track};
  SchurVisualizationQuality_s quality;
  quality.gate_all_cloud_tracks_ = true;
  const auto snapshot = makeSchurVisualizationSnapshot(
    state, {CameraExtrinsics_s()}, updater, tracks, tracks, 2, 2.0, {}, nullptr, quality);
  ASSERT_TRUE(snapshot.valid_);
  ASSERT_EQ(snapshot.cloud_points_.size(), 1u);
  EXPECT_TRUE(snapshot.cloud_points_[0].world_point_.isApprox(truth, 1e-6));
}
