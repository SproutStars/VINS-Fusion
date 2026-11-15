/*******************************************************
 * Deterministic total-observation budget for one visual update.
 *
 * The frontend feature count, the backend track cap (`max_features_`), and the
 * number of historical observations admitted to one update are independent
 * limits. This helper enforces only the last limit. It does not gate outliers,
 * does not gate minimum track length, and does not gate minimum frame counts.
 *
 * Contract:
 * - `max_observations_ <= 0` disables the observation budget. The input order,
 *   every track, and every observation vector are preserved exactly.
 * - A track is atomic. It contributes `observations_.size()` observations and
 *   is never truncated. A same-clone stereo pair therefore contributes two
 *   observations and its baseline stays intact.
 * - The caller supplies FeatureHistory's deterministic longest-first stable
 *   order. Selected indices stay in that input order.
 * - Call the planner once per frame after FeatureHistory::trim() and before the
 *   update. The planner does not mutate FeatureHistory. Set margin_active_ from
 *   the caller's window_full state. Clone index 0 is the outgoing oldest clone.
 * - `skip_oversized_` defers a complete track that does not fit the remaining
 *   budget and keeps scanning for later smaller tracks. With false, selection
 *   stops at the first track that does not fit.
 * - A track larger than the whole observation budget can never be selected.
 *   It is reported in `expired_feature_ids_`, not in deferred or rejected ids.
 * - With `margin_active_` set, a track whose oldest observation is at or before
 *   `margin_clone_index_` is at the oldest-clone margin. If it does not fit,
 *   it is expired instead of deferred. Otherwise it could stay pending across
 *   successive trims while new observations keep it alive. A non-margin track
 *   becomes a margin track after at most one clone window, so every deferred
 *   track is selected or expired within that window.
 * - Deferred tracks are not outliers and are not consumed. Keep them in
 *   FeatureHistory. Never add them to used, rejected, or consumed ids.
 * - Expired tracks are not outliers and are not consumed either. Erase them
 *   from FeatureHistory without passing them to the measurement update, even
 *   when the update is rejected. FeatureHistory::trim() remains the hard bound
 *   on observation lifetime.
 *******************************************************/

#pragma once

#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "schur_update.h"

namespace vins
{
namespace backend
{

struct ObservationBudgetOptions_s {
  // <= 0 means unlimited. Positive values cap the sum of complete observation
  // vectors admitted by one plan.
  int max_observations_ = 0;
  // <= 0 means unlimited. This is independent of UpdateOptions_s::max_features_.
  int max_tracks_ = 0;
  // Continue scanning after a complete track does not fit. With false, all
  // later tracks are classified but not admitted.
  bool skip_oversized_ = true;
  // Set from the caller's clone-window state. A margin track that cannot fit
  // is expired instead of deferred.
  bool margin_active_ = true;
  // The outgoing oldest clone index. After FeatureHistory resolution this is 0.
  int margin_clone_index_ = 0;
};

struct ObservationBudgetPlan_s {
  // Ascending indices into the input vector. Selected tracks remain complete.
  std::vector<std::size_t> selected_indices_;
  std::vector<int> selected_feature_ids_;
  // Non-margin tracks that did not fit. They stay in FeatureHistory.
  std::vector<int> deferred_feature_ids_;
  // Permanently unselectable tracks and margin tracks that did not fit.
  // Erase them from FeatureHistory without update or outlier accounting.
  std::vector<int> expired_feature_ids_;
  int input_track_count_ = 0;
  int input_observation_count_ = 0;
  int selected_track_count_ = 0;
  int selected_observation_count_ = 0;
  int deferred_track_count_ = 0;
  int deferred_observation_count_ = 0;
  int expired_track_count_ = 0;
  int expired_observation_count_ = 0;
  int oversized_expired_track_count_ = 0;
  int margin_expired_track_count_ = 0;
  bool limited_ = false;  // true when at least one track was deferred or expired
};

namespace observation_budget_detail
{

inline int
saturatingCount(std::size_t _count)
{
  const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  return _count > limit ? std::numeric_limits<int>::max() : static_cast<int>(_count);
}

inline int
saturatingAdd(int _lhs, int _rhs)
{
  const int limit = std::numeric_limits<int>::max();
  return _lhs > limit - _rhs ? limit : _lhs + _rhs;
}

inline bool
isMarginTrack(const FeatureTrack_s & _track, const ObservationBudgetOptions_s & _options)
{
  if (!_options.margin_active_ || _track.observations_.empty())
    return false;
  return _track.observations_.front().clone_index_ <= _options.margin_clone_index_;
}

inline void
recordDeferred(ObservationBudgetPlan_s & _plan, const FeatureTrack_s & _track, int _observations)
{
  _plan.deferred_feature_ids_.push_back(_track.id_);
  ++_plan.deferred_track_count_;
  _plan.deferred_observation_count_ = saturatingAdd(_plan.deferred_observation_count_, _observations);
  _plan.limited_ = true;
}

inline void
recordExpired(
  ObservationBudgetPlan_s & _plan, const FeatureTrack_s & _track, int _observations, bool _oversized,
  bool _margin)
{
  _plan.expired_feature_ids_.push_back(_track.id_);
  ++_plan.expired_track_count_;
  _plan.expired_observation_count_ = saturatingAdd(_plan.expired_observation_count_, _observations);
  // Mutually exclusive diagnostics: a track larger than the whole budget is
  // classified as oversized first, independent of its oldest clone.
  if (_oversized)
    ++_plan.oversized_expired_track_count_;
  else if (_margin)
    ++_plan.margin_expired_track_count_;
  _plan.limited_ = true;
}

inline void
recordSelected(ObservationBudgetPlan_s & _plan, const FeatureTrack_s & _track, std::size_t _index, int _observations)
{
  _plan.selected_indices_.push_back(_index);
  _plan.selected_feature_ids_.push_back(_track.id_);
  ++_plan.selected_track_count_;
  _plan.selected_observation_count_ = saturatingAdd(_plan.selected_observation_count_, _observations);
}

}  // namespace observation_budget_detail

// Plans one deterministic selection without modifying the input.
inline ObservationBudgetPlan_s
planObservationBudget(
  const std::vector<FeatureTrack_s> & _tracks,
  const ObservationBudgetOptions_s & _options = ObservationBudgetOptions_s())
{
  using namespace observation_budget_detail;
  ObservationBudgetPlan_s plan;
  plan.input_track_count_ = saturatingCount(_tracks.size());
  const bool observation_unlimited = _options.max_observations_ <= 0;
  const bool track_unlimited = _options.max_tracks_ <= 0;
  int remaining_observations = observation_unlimited ? 0 : _options.max_observations_;
  bool stop = false;
  auto classify_unselected = [&](const FeatureTrack_s & _track, int _observations, bool _margin)
  {
    // A track larger than the total budget can never be selected under this
    // configuration. Expire it instead of deferring it forever.
    if (!observation_unlimited && _observations > _options.max_observations_) {
      recordExpired(plan, _track, _observations, true, _margin);
      return;
    }
    if (_margin)
      recordExpired(plan, _track, _observations, false, true);
    else
      recordDeferred(plan, _track, _observations);
  };

  for (std::size_t index = 0; index < _tracks.size(); ++index) {
    const FeatureTrack_s & track = _tracks[index];
    const int observations = saturatingCount(track.observations_.size());
    const bool margin = isMarginTrack(track, _options);

    plan.input_observation_count_ = saturatingAdd(plan.input_observation_count_, observations);

    if (stop || (!track_unlimited && plan.selected_track_count_ >= _options.max_tracks_)) {
      classify_unselected(track, observations, margin);
      continue;
    }

    const bool oversized = !observation_unlimited && observations > _options.max_observations_;
    if (oversized) {
      classify_unselected(track, observations, margin);
      if (!_options.skip_oversized_)
        stop = true;
      continue;
    }

    if (!observation_unlimited && observations > remaining_observations) {
      classify_unselected(track, observations, margin);
      if (!_options.skip_oversized_)
        stop = true;
      continue;
    }

    recordSelected(plan, track, index, observations);
    if (!observation_unlimited)
      remaining_observations -= observations;
  }

  return plan;
}

// Plans and then compacts `_tracks` in place to the selected tracks, preserving
// their original relative order. The returned indices refer to the input
// before compaction.
inline ObservationBudgetPlan_s
applyObservationBudget(
  std::vector<FeatureTrack_s> & _tracks,
  const ObservationBudgetOptions_s & _options = ObservationBudgetOptions_s())
{
  const ObservationBudgetPlan_s plan = planObservationBudget(_tracks, _options);
  std::vector<FeatureTrack_s> selected;
  selected.reserve(plan.selected_indices_.size());
  for (const std::size_t index : plan.selected_indices_)
    selected.push_back(std::move(_tracks[index]));
  _tracks.swap(selected);
  return plan;
}

struct ObservationBudgetSelection_s {
  std::vector<FeatureTrack_s> tracks_;
  ObservationBudgetPlan_s plan_;
};

// Convenience wrapper for callers that want selected copies plus diagnostics.
inline ObservationBudgetSelection_s
selectObservationBudgetTracks(
  const std::vector<FeatureTrack_s> & _tracks,
  const ObservationBudgetOptions_s & _options = ObservationBudgetOptions_s())
{
  ObservationBudgetSelection_s selection;
  selection.plan_ = planObservationBudget(_tracks, _options);
  selection.tracks_.reserve(selection.plan_.selected_indices_.size());
  for (const std::size_t index : selection.plan_.selected_indices_)
    selection.tracks_.push_back(_tracks[index]);
  return selection;
}

}  // namespace backend
}  // namespace vins
