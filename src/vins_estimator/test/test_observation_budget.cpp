/*******************************************************
 * Unit tests for the deterministic observation-budget helper
 * (src/vins_estimator/src/backend/observation_budget.h).
 *
 * Standalone: gtest + Eigen only.
 *
 *   g++ -std=c++17 -O2 -Wall -Wextra -I src/vins_estimator/src \
 *       -I /usr/include/eigen3 src/vins_estimator/test/test_observation_budget.cpp \
 *       -lgtest -lgtest_main -pthread -o /tmp/test_observation_budget
 *
 * The helper never truncates a track. The tests therefore check complete
 * mono/stereo observation vectors, not only selected id counts.
 *******************************************************/

#include <gtest/gtest.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "backend/feature_history.h"
#include "backend/observation_budget.h"

namespace
{

using namespace vins::backend;

FeatureObservation_s
observation(int _clone, int _camera, int _seed)
{
  FeatureObservation_s result;
  result.clone_index_ = _clone;
  result.camera_id_ = _camera;
  result.uv_ = Eigen::Vector2d(0.01 * (_seed + 1), -0.02 * (_seed + 1));
  result.pixel_uv_ = Eigen::Vector2d(320.0 + _seed, 240.0 - _seed);
  result.has_pixel_uv_ = true;
  return result;
}

FeatureTrack_s
makeMonoTrack(int _id, int _first_clone, int _clone_count)
{
  FeatureTrack_s track;
  track.id_ = _id;
  track.observations_.reserve(static_cast<std::size_t>(_clone_count));
  for (int i = 0; i < _clone_count; ++i)
    track.observations_.push_back(observation(_first_clone + i, 0, i));
  return track;
}

FeatureTrack_s
makeStereoTrack(int _id, int _first_clone, int _clone_count)
{
  FeatureTrack_s track;
  track.id_ = _id;
  track.observations_.reserve(static_cast<std::size_t>(2 * _clone_count));
  for (int i = 0; i < _clone_count; ++i) {
    track.observations_.push_back(observation(_first_clone + i, 0, i));
    track.observations_.push_back(observation(_first_clone + i, 1, i));
  }
  return track;
}

std::vector<int>
ids(const std::vector<FeatureTrack_s> & _tracks)
{
  std::vector<int> result;
  result.reserve(_tracks.size());
  for (const FeatureTrack_s & track : _tracks)
    result.push_back(track.id_);
  return result;
}

int
totalObservations(const std::vector<FeatureTrack_s> & _tracks)
{
  int total = 0;
  for (const FeatureTrack_s & track : _tracks)
    total += static_cast<int>(track.observations_.size());
  return total;
}

void
expectSameTrackShape(const FeatureTrack_s & _actual, const FeatureTrack_s & _expected)
{
  ASSERT_EQ(_actual.id_, _expected.id_);
  ASSERT_EQ(_actual.observations_.size(), _expected.observations_.size());
  for (std::size_t i = 0; i < _actual.observations_.size(); ++i) {
    const FeatureObservation_s & actual = _actual.observations_[i];
    const FeatureObservation_s & expected = _expected.observations_[i];
    EXPECT_EQ(actual.clone_index_, expected.clone_index_);
    EXPECT_EQ(actual.camera_id_, expected.camera_id_);
    EXPECT_TRUE(actual.uv_.isApprox(expected.uv_));
    EXPECT_TRUE(actual.pixel_uv_.isApprox(expected.pixel_uv_));
    EXPECT_EQ(actual.has_pixel_uv_, expected.has_pixel_uv_);
  }
}

void
expectSamePlan(const ObservationBudgetPlan_s & _a, const ObservationBudgetPlan_s & _b)
{
  EXPECT_EQ(_a.selected_indices_, _b.selected_indices_);
  EXPECT_EQ(_a.selected_feature_ids_, _b.selected_feature_ids_);
  EXPECT_EQ(_a.deferred_feature_ids_, _b.deferred_feature_ids_);
  EXPECT_EQ(_a.expired_feature_ids_, _b.expired_feature_ids_);
  EXPECT_EQ(_a.input_track_count_, _b.input_track_count_);
  EXPECT_EQ(_a.input_observation_count_, _b.input_observation_count_);
  EXPECT_EQ(_a.selected_track_count_, _b.selected_track_count_);
  EXPECT_EQ(_a.selected_observation_count_, _b.selected_observation_count_);
  EXPECT_EQ(_a.deferred_track_count_, _b.deferred_track_count_);
  EXPECT_EQ(_a.deferred_observation_count_, _b.deferred_observation_count_);
  EXPECT_EQ(_a.expired_track_count_, _b.expired_track_count_);
  EXPECT_EQ(_a.expired_observation_count_, _b.expired_observation_count_);
  EXPECT_EQ(_a.oversized_expired_track_count_, _b.oversized_expired_track_count_);
  EXPECT_EQ(_a.margin_expired_track_count_, _b.margin_expired_track_count_);
  EXPECT_EQ(_a.limited_, _b.limited_);
}

TEST(ObservationBudget, ZeroBudgetIsExactIdentity)
{
  const std::vector<FeatureTrack_s> input{
    makeMonoTrack(42, 1, 2), makeStereoTrack(7, 0, 3), makeMonoTrack(99, 2, 1)};
  const std::vector<FeatureTrack_s> before = input;
  const ObservationBudgetOptions_s options;  // max_observations_ == 0, max_tracks_ == 0

  const ObservationBudgetPlan_s plan = planObservationBudget(input, options);
  ASSERT_EQ(plan.selected_indices_.size(), input.size());
  EXPECT_EQ(plan.selected_indices_, (std::vector<std::size_t>{0, 1, 2}));
  EXPECT_EQ(plan.selected_feature_ids_, (std::vector<int>{42, 7, 99}));
  EXPECT_TRUE(plan.deferred_feature_ids_.empty());
  EXPECT_TRUE(plan.expired_feature_ids_.empty());
  EXPECT_EQ(plan.input_track_count_, 3);
  EXPECT_EQ(plan.input_observation_count_, totalObservations(input));
  EXPECT_EQ(plan.selected_track_count_, 3);
  EXPECT_EQ(plan.selected_observation_count_, plan.input_observation_count_);
  EXPECT_EQ(plan.deferred_track_count_, 0);
  EXPECT_EQ(plan.expired_track_count_, 0);
  EXPECT_FALSE(plan.limited_);

  const ObservationBudgetSelection_s selection = selectObservationBudgetTracks(input, options);
  ASSERT_EQ(selection.tracks_.size(), before.size());
  for (std::size_t i = 0; i < before.size(); ++i)
    expectSameTrackShape(selection.tracks_[i], before[i]);

  std::vector<FeatureTrack_s> applied = input;
  const ObservationBudgetPlan_s applied_plan = applyObservationBudget(applied, options);
  expectSamePlan(applied_plan, plan);
  ASSERT_EQ(applied.size(), before.size());
  for (std::size_t i = 0; i < before.size(); ++i)
    expectSameTrackShape(applied[i], before[i]);
}

TEST(ObservationBudget, SelectsCompleteStereoTrackAndPreservesShape)
{
  const std::vector<FeatureTrack_s> input{
    makeStereoTrack(10, 0, 3), makeMonoTrack(11, 0, 3), makeMonoTrack(12, 0, 2)};
  ObservationBudgetOptions_s options;
  options.max_observations_ = 6;
  options.margin_active_ = false;

  const ObservationBudgetSelection_s selection = selectObservationBudgetTracks(input, options);
  ASSERT_EQ(selection.tracks_.size(), 1u);
  EXPECT_EQ(selection.tracks_.front().id_, 10);
  ASSERT_EQ(selection.tracks_.front().observations_.size(), 6u);
  expectSameTrackShape(selection.tracks_.front(), input.front());

  const int expected_camera[6] = {0, 1, 0, 1, 0, 1};
  const int expected_clone[6] = {0, 0, 1, 1, 2, 2};
  for (std::size_t i = 0; i < selection.tracks_.front().observations_.size(); ++i) {
    EXPECT_EQ(selection.tracks_.front().observations_[i].camera_id_, expected_camera[i]);
    EXPECT_EQ(selection.tracks_.front().observations_[i].clone_index_, expected_clone[i]);
  }
  EXPECT_EQ(selection.plan_.selected_observation_count_, 6);
  EXPECT_EQ(selection.plan_.deferred_feature_ids_, (std::vector<int>{11, 12}));
  EXPECT_TRUE(selection.plan_.expired_feature_ids_.empty());
}

TEST(ObservationBudget, SkipsOversizedTrackAndSelectsLaterCompleteTracks)
{
  const std::vector<FeatureTrack_s> input{
    makeStereoTrack(10, 0, 3), makeMonoTrack(11, 0, 3), makeMonoTrack(12, 0, 2)};
  ObservationBudgetOptions_s options;
  options.max_observations_ = 5;
  options.margin_active_ = false;

  const ObservationBudgetSelection_s selection = selectObservationBudgetTracks(input, options);
  ASSERT_EQ(selection.tracks_.size(), 2u);
  EXPECT_EQ(selection.tracks_[0].id_, 11);
  EXPECT_EQ(selection.tracks_[1].id_, 12);
  expectSameTrackShape(selection.tracks_[0], input[1]);
  expectSameTrackShape(selection.tracks_[1], input[2]);
  EXPECT_EQ(selection.plan_.selected_observation_count_, 5);
  EXPECT_EQ(selection.plan_.expired_feature_ids_, (std::vector<int>{10}));
  EXPECT_EQ(selection.plan_.oversized_expired_track_count_, 1);
  EXPECT_TRUE(selection.plan_.deferred_feature_ids_.empty());
}

TEST(ObservationBudget, DeterministicLongestFirstStableOrder)
{
  FeatureHistory history;
  for (int frame = 0; frame <= 4; ++frame) {
    std::vector<std::pair<int, Eigen::Vector2d>> observations;
    const Eigen::Vector2d uv(0.1, -0.05);
    observations.emplace_back(101, uv);
    if (frame <= 2) {
      observations.emplace_back(102, uv);
      observations.emplace_back(103, uv);
    }
    if (frame <= 1)
      observations.emplace_back(104, uv);
    history.addFrame(frame, 0, observations);
  }

  const FeatureHistory::TrackSelection candidates = history.tracks(0, 4, 1, 10, true);
  ASSERT_EQ(ids(candidates.tracks_), (std::vector<int>{101, 102, 103, 104}));

  ObservationBudgetOptions_s options;
  options.max_observations_ = 8;
  options.margin_active_ = false;
  const ObservationBudgetPlan_s expected = planObservationBudget(candidates.tracks_, options);
  EXPECT_EQ(expected.selected_feature_ids_, (std::vector<int>{101, 102}));
  EXPECT_EQ(expected.deferred_feature_ids_, (std::vector<int>{103, 104}));
  EXPECT_EQ(expected.selected_observation_count_, 8);
  EXPECT_EQ(expected.deferred_observation_count_, 5);

  for (int repeat = 0; repeat < 50; ++repeat)
    expectSamePlan(planObservationBudget(candidates.tracks_, options), expected);
}

TEST(ObservationBudget, DefersNonMarginTrackAndSelectsLaterSmaller)
{
  const std::vector<FeatureTrack_s> input{
    makeMonoTrack(201, 1, 4), makeMonoTrack(202, 1, 3), makeMonoTrack(203, 1, 1)};
  ObservationBudgetOptions_s options;
  options.max_observations_ = 5;

  const ObservationBudgetPlan_s plan = planObservationBudget(input, options);
  EXPECT_EQ(plan.selected_feature_ids_, (std::vector<int>{201, 203}));
  EXPECT_EQ(plan.deferred_feature_ids_, (std::vector<int>{202}));
  EXPECT_TRUE(plan.expired_feature_ids_.empty());
  EXPECT_EQ(plan.selected_observation_count_, 5);
  EXPECT_EQ(plan.deferred_observation_count_, 3);
}

TEST(ObservationBudget, StrictPrefixStopsAtFirstNonFit)
{
  const std::vector<FeatureTrack_s> input{
    makeMonoTrack(211, 1, 4), makeMonoTrack(212, 1, 3), makeMonoTrack(213, 1, 1)};
  ObservationBudgetOptions_s options;
  options.max_observations_ = 5;
  options.skip_oversized_ = false;

  const ObservationBudgetPlan_s plan = planObservationBudget(input, options);
  EXPECT_EQ(plan.selected_feature_ids_, (std::vector<int>{211}));
  EXPECT_EQ(plan.deferred_feature_ids_, (std::vector<int>{212, 213}));
  EXPECT_TRUE(plan.expired_feature_ids_.empty());
  EXPECT_EQ(plan.selected_observation_count_, 4);
  EXPECT_EQ(plan.deferred_observation_count_, 4);
}

TEST(ObservationBudget, StrictPrefixStopsAtOversizedTrack)
{
  const std::vector<FeatureTrack_s> input{
    makeMonoTrack(221, 1, 4), makeMonoTrack(222, 1, 6), makeMonoTrack(223, 1, 1)};
  ObservationBudgetOptions_s options;
  options.max_observations_ = 5;
  options.skip_oversized_ = false;
  options.margin_active_ = false;

  const ObservationBudgetPlan_s plan = planObservationBudget(input, options);
  EXPECT_EQ(plan.selected_feature_ids_, (std::vector<int>{221}));
  EXPECT_EQ(plan.deferred_feature_ids_, (std::vector<int>{223}));
  EXPECT_EQ(plan.expired_feature_ids_, (std::vector<int>{222}));
  EXPECT_EQ(plan.selected_observation_count_, 4);
  EXPECT_EQ(plan.deferred_observation_count_, 1);
  EXPECT_EQ(plan.expired_observation_count_, 6);
  EXPECT_EQ(plan.oversized_expired_track_count_, 1);
}

TEST(ObservationBudget, ExpiresOversizedAndOldestMarginTracks)
{
  const std::vector<FeatureTrack_s> input{
    makeMonoTrack(301, 1, 4), makeMonoTrack(302, 0, 2), makeMonoTrack(303, 1, 1),
    makeMonoTrack(304, 1, 8)};
  ObservationBudgetOptions_s options;
  options.max_observations_ = 5;
  options.margin_active_ = true;
  options.margin_clone_index_ = 0;

  const ObservationBudgetPlan_s plan = planObservationBudget(input, options);
  EXPECT_EQ(plan.selected_feature_ids_, (std::vector<int>{301, 303}));
  EXPECT_TRUE(plan.deferred_feature_ids_.empty());
  EXPECT_EQ(plan.expired_feature_ids_, (std::vector<int>{302, 304}));
  EXPECT_EQ(plan.selected_observation_count_, 5);
  EXPECT_EQ(plan.expired_observation_count_, 10);
  EXPECT_EQ(plan.margin_expired_track_count_, 1);
  EXPECT_EQ(plan.oversized_expired_track_count_, 1);
  EXPECT_EQ(plan.input_observation_count_, 15);

  options.margin_active_ = false;
  const ObservationBudgetPlan_s without_margin = planObservationBudget(input, options);
  EXPECT_EQ(without_margin.selected_feature_ids_, (std::vector<int>{301, 303}));
  EXPECT_EQ(without_margin.deferred_feature_ids_, (std::vector<int>{302}));
  EXPECT_EQ(without_margin.expired_feature_ids_, (std::vector<int>{304}));
}

TEST(ObservationBudget, MinimumPositiveBudgetExpiresAllUnbudgetableTracks)
{
  const std::vector<FeatureTrack_s> input{makeMonoTrack(401, 1, 2), makeMonoTrack(402, 1, 3)};
  ObservationBudgetOptions_s options;
  options.max_observations_ = 1;
  options.margin_active_ = false;

  const ObservationBudgetPlan_s plan = planObservationBudget(input, options);
  EXPECT_TRUE(plan.selected_feature_ids_.empty());
  EXPECT_TRUE(plan.deferred_feature_ids_.empty());
  EXPECT_EQ(plan.expired_feature_ids_, (std::vector<int>{401, 402}));
  EXPECT_EQ(plan.expired_observation_count_, 5);
  EXPECT_EQ(plan.oversized_expired_track_count_, 2);
  EXPECT_EQ(plan.selected_observation_count_, 0);
}

TEST(ObservationBudget, TrackCapIsIndependentFromObservationBudget)
{
  const std::vector<FeatureTrack_s> input{
    makeMonoTrack(501, 1, 2), makeMonoTrack(502, 1, 2), makeMonoTrack(503, 1, 2)};
  ObservationBudgetOptions_s options;
  options.max_observations_ = 0;  // unlimited observations
  options.max_tracks_ = 2;
  options.margin_active_ = false;

  const ObservationBudgetPlan_s plan = planObservationBudget(input, options);
  EXPECT_EQ(plan.selected_feature_ids_, (std::vector<int>{501, 502}));
  EXPECT_EQ(plan.deferred_feature_ids_, (std::vector<int>{503}));
  EXPECT_TRUE(plan.expired_feature_ids_.empty());
  EXPECT_EQ(plan.selected_track_count_, 2);
  EXPECT_EQ(plan.deferred_track_count_, 1);
  EXPECT_EQ(plan.selected_observation_count_, 4);
  EXPECT_EQ(plan.deferred_observation_count_, 2);
  EXPECT_TRUE(plan.limited_);
}

TEST(ObservationBudget, TrackCapStillExpiresPermanentlyOversizedTrack)
{
  const std::vector<FeatureTrack_s> input{
    makeMonoTrack(511, 1, 2), makeMonoTrack(512, 1, 2), makeMonoTrack(513, 1, 8)};
  ObservationBudgetOptions_s options;
  options.max_observations_ = 5;
  options.max_tracks_ = 2;
  options.margin_active_ = false;

  const ObservationBudgetPlan_s plan = planObservationBudget(input, options);
  EXPECT_EQ(plan.selected_feature_ids_, (std::vector<int>{511, 512}));
  EXPECT_TRUE(plan.deferred_feature_ids_.empty());
  EXPECT_EQ(plan.expired_feature_ids_, (std::vector<int>{513}));
  EXPECT_EQ(plan.oversized_expired_track_count_, 1);
  EXPECT_EQ(plan.selected_observation_count_, 4);
  EXPECT_EQ(plan.expired_observation_count_, 8);
}

TEST(ObservationBudget, ApplyCompactsInInputOrderWithoutRejectingDeferredTracks)
{
  std::vector<FeatureTrack_s> tracks{
    makeMonoTrack(601, 1, 2), makeMonoTrack(602, 1, 5), makeMonoTrack(603, 1, 1)};
  ObservationBudgetOptions_s options;
  options.max_observations_ = 3;
  options.margin_active_ = false;

  const ObservationBudgetPlan_s plan = applyObservationBudget(tracks, options);
  ASSERT_EQ(tracks.size(), 2u);
  EXPECT_EQ(ids(tracks), (std::vector<int>{601, 603}));
  EXPECT_EQ(plan.selected_indices_, (std::vector<std::size_t>{0, 2}));
  EXPECT_EQ(plan.expired_feature_ids_, (std::vector<int>{602}));
  EXPECT_TRUE(plan.deferred_feature_ids_.empty());
  expectSameTrackShape(tracks[0], makeMonoTrack(601, 1, 2));
  expectSameTrackShape(tracks[1], makeMonoTrack(603, 1, 1));
}

}  // namespace
