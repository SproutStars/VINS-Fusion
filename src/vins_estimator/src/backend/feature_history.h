/*******************************************************
 * Per-feature observation history for the Schur-complement EKF backend.
 *
 * The estimator augments exactly one clone per image frame, so the clone index
 * of a frame is (frame_index - oldest_retained_frame_index). Observations are
 * therefore stored with monotonic frame indices and resolved to clone indices
 * only when the tracks are handed to SchurUpdater::update().
 *
 * Feature reuse policy follows OpenVINS: a track is selected only when it was
 * lost in the newest frame, or when it contains an observation at the oldest
 * clone that is about to be marginalized. Selected tracks must be erased after
 * the update so the same measurements are never accumulated twice.
 *
 * Feature points are NOT cached here: SchurUpdater re-triangulates every
 * selected track from the current clone poses, matching OpenVINS (a feature is
 * consumed by at most one update, so freezing the first triangulation was
 * measured to be no better on full sequences -- see schur_update.cpp).
 *******************************************************/

#pragma once

#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <unordered_map>
#include <utility>
#include <vector>

#include "schur_update.h"

namespace vins
{
namespace backend
{

struct TrackSelection_s {
  std::vector<FeatureTrack_s> tracks_;
  // Eligible tracks before max_features truncation. A track that is both
  // lost and marginalizable is counted once as marginalized, matching
  // OpenVINS' overlap removal between feats_lost and feats_marg.
  int lost_count_ = 0;
  int marginalized_count_ = 0;
};

class FeatureHistory {
public:
  struct Observation_s {
    int frame_index_ = 0;
    int camera_id_ = 0;
    Eigen::Vector2d uv_ = Eigen::Vector2d::Zero();  // normalised, undistorted
    Eigen::Vector2d pixel_uv_ = Eigen::Vector2d::Zero();
    bool has_pixel_uv_ = false;
  };

  struct FrameObservation_s {
    int feature_id_ = -1;
    Eigen::Vector2d uv_ = Eigen::Vector2d::Zero();
    Eigen::Vector2d pixel_uv_ = Eigen::Vector2d::Zero();
    bool has_pixel_uv_ = false;
  };

  using TrackSelection = ::vins::backend::TrackSelection_s;

  // Registers one image frame. `observations` maps feature id -> normalised uv.
  void
  addFrame(int _frame_index, int _camera_id, const std::vector<std::pair<int, Eigen::Vector2d>> & _observations)
  {
    std::vector<FrameObservation_s> converted;
    converted.reserve(_observations.size());
    for (const auto & observation : _observations)
      converted.push_back(FrameObservation_s{observation.first, observation.second});
    addFrameWithPixels(_frame_index, _camera_id, converted);
  }

  // Registers one camera's observations for a frame while retaining the raw
  // pixel coordinates needed by the loop-fusion keyframe message. This is
  // publication-only metadata; the measurement update consumes `uv` exactly as
  // before. A stereo pair is registered with two calls at the same
  // `_frame_index` and different `_camera_id` values; each (frame, camera) view
  // is kept so both measurements enter the Schur update. The per-feature
  // history is kept ordered by (frame_index, camera_id), independent of call
  // order, because triangulation anchors on the first view and consumes the
  // last one for its two-view initialisation.
  void addFrameWithPixels(int _frame_index, int _camera_id, const std::vector<FrameObservation_s> & _observations)
  {
    newest_frame_index_ = std::max(newest_frame_index_, _frame_index);
    for (const auto & observation : _observations) {
      Entry_s & entry = features_[observation.feature_id_];
      auto & history = entry.observations_;
      const Observation_s record{
        _frame_index, _camera_id, observation.uv_, observation.pixel_uv_, observation.has_pixel_uv_};
      // Normal tracking is monotonic: frames advance and the estimator adds
      // camera 0 before camera 1, so the common case is a plain append.
      const bool append_to_back =
        history.empty() || history.back().frame_index_ < _frame_index ||
        (history.back().frame_index_ == _frame_index && history.back().camera_id_ < _camera_id);
      if (append_to_back) {
        history.push_back(record);
      } else {
        // Out-of-order fallback (reverse camera order, replayed/older frame):
        // keep the deterministic (frame_index, camera_id) order and drop only
        // an exact duplicate view.
        auto insert_at = history.begin();
        while (
          insert_at != history.end() &&
          (insert_at->frame_index_ < _frame_index ||
           (insert_at->frame_index_ == _frame_index && insert_at->camera_id_ < _camera_id)))
          ++insert_at;
        if (insert_at != history.end() && insert_at->frame_index_ == _frame_index &&
            insert_at->camera_id_ == _camera_id)
          continue;  // duplicate (frame, camera) view: keep the first observation
        history.insert(insert_at, record);
      }
      entry.last_frame_ = std::max(entry.last_frame_, _frame_index);
    }
  }

  // Drops observations older than the oldest retained clone. Entries that
  // become empty are removed. A track that has merely been unobserved for a
  // frame or two is retained until it is selected and erased by the caller.
  void trim(int _oldest_frame_index)
  {
    for (auto it = features_.begin(); it != features_.end();) {
      Entry_s & entry = it->second;
      auto first_kept = entry.observations_.begin();
      while (first_kept != entry.observations_.end() && first_kept->frame_index_ < _oldest_frame_index)
        ++first_kept;
      entry.observations_.erase(entry.observations_.begin(), first_kept);
      if (entry.observations_.empty()) {
        it = features_.erase(it);
        continue;
      }
      ++it;
    }
  }

  // Compatibility overload retained only for source compatibility with old
  // callers. The `newest_frame_index` argument is intentionally ignored and
  // this call is exactly equivalent to trim(oldest_frame_index): there is no
  // newest-frame pruning rule any more.
  void trim(int _oldest_frame_index, [[maybe_unused]] int _newest_frame_index) { trim(_oldest_frame_index); }

  // Selects tracks for exactly one MSCKF-style visual update.
  //
  // eligible = (last_frame < newest_frame_index) ||
  //            (window_full &&
  //             observations.front().frame_index <= oldest_frame_index)
  //
  // The first case is a feature lost in the newest image. The second case is
  // enabled only when the clone window is full and the oldest clone is about
  // to be marginalized; while the window is filling, old features must not be
  // selected merely because they touch clone zero.
  // Both eligibility classes are deduplicated into one updated track list.
  // The result is sorted by observation count descending, then feature id
  // ascending, and finally capped at max_features.
  TrackSelection selectTracks(
    int _oldest_frame_index, int _newest_frame_index, int _min_observations, size_t _max_features,
    bool _window_full = true) const
  {
    TrackSelection selection;
    for (const auto & feature : features_) {
      const Entry_s & entry = feature.second;
      if (static_cast<int>(entry.observations_.size()) < _min_observations)
        continue;

      const bool marginalized = _window_full && entry.observations_.front().frame_index_ <= _oldest_frame_index;
      const bool lost = entry.last_frame_ < _newest_frame_index;
      if (!marginalized && !lost)
        continue;

      if (marginalized)
        ++selection.marginalized_count_;
      else
        ++selection.lost_count_;

      FeatureTrack_s track;
      track.id_ = feature.first;
      track.observations_.reserve(entry.observations_.size());
      for (const Observation_s & observation : entry.observations_) {
        FeatureObservation_s resolved;
        resolved.clone_index_ = observation.frame_index_ - _oldest_frame_index;
        resolved.camera_id_ = observation.camera_id_;
        resolved.uv_ = observation.uv_;
        track.observations_.push_back(resolved);
      }
      selection.tracks_.push_back(std::move(track));
    }

    std::sort(
      selection.tracks_.begin(), selection.tracks_.end(),
      [](const FeatureTrack_s & _a, const FeatureTrack_s & _b)
      {
        if (_a.observations_.size() != _b.observations_.size())
          return _a.observations_.size() > _b.observations_.size();
        return _a.id_ < _b.id_;
      });
    if (selection.tracks_.size() > _max_features)
      selection.tracks_.resize(_max_features);
    return selection;
  }

  // Primary selection entry point used by the estimator. It returns both the
  // selected tracks and the lost/marginalized diagnostics. Passing
  // window_full=false suppresses the marginalization eligibility class.
  TrackSelection tracks(
    int _oldest_frame_index, int _newest_frame_index, int _min_observations, size_t _max_features,
    bool _window_full) const
  {
    return selectTracks(
      _oldest_frame_index, _newest_frame_index, _min_observations, _max_features, _window_full);
  }

  // Compatibility overload: old callers that do not know whether the window
  // is full keep the original lost-or-marginalized behavior.
  TrackSelection
  tracks(int _oldest_frame_index, int _newest_frame_index, int _min_observations, size_t _max_features) const
  {
    return tracks(_oldest_frame_index, _newest_frame_index, _min_observations, _max_features, true);
  }

  // Backwards-compatible wrapper using the newest frame registered by
  // add_frame(). New code should prefer select_tracks() so it can log the
  // lost/marginalized counts.
  std::vector<FeatureTrack_s> tracks(int _oldest_frame_index, int _min_observations, size_t _max_features) const
  {
    return selectTracks(_oldest_frame_index, newest_frame_index_, _min_observations, _max_features).tracks_;
  }

  // Removes tracks whose measurements were actually accumulated by the
  // update. Call with UpdateSummary::used_feature_ids, not with rejected or
  // merely consumed ids, so unused tracks remain available for a later frame.
  void eraseFeatures(const std::vector<int> & _feature_ids)
  {
    for (const int id : _feature_ids)
      features_.erase(id);
  }

  // Copies every retained track and resolves frame indices to clone indices.
  // Used only to build a publication snapshot before measurement tracks are
  // erased; callers must not feed the result back into an EKF update.
  std::vector<FeatureTrack_s> allTracks(int _oldest_frame_index) const
  {
    std::vector<FeatureTrack_s> result;
    result.reserve(features_.size());
    for (const auto & feature : features_) {
      FeatureTrack_s track;
      track.id_ = feature.first;
      track.observations_.reserve(feature.second.observations_.size());
      for (const Observation_s & observation : feature.second.observations_) {
        if (observation.frame_index_ < _oldest_frame_index)
          continue;
        FeatureObservation_s resolved;
        resolved.clone_index_ = observation.frame_index_ - _oldest_frame_index;
        resolved.camera_id_ = observation.camera_id_;
        resolved.uv_ = observation.uv_;
        resolved.pixel_uv_ = observation.pixel_uv_;
        resolved.has_pixel_uv_ = observation.has_pixel_uv_;
        track.observations_.push_back(resolved);
      }
      if (!track.observations_.empty())
        result.push_back(std::move(track));
    }
    std::sort(
      result.begin(), result.end(),
      [](const FeatureTrack_s & _a, const FeatureTrack_s & _b)
      {
        if (_a.observations_.size() != _b.observations_.size())
          return _a.observations_.size() > _b.observations_.size();
        return _a.id_ < _b.id_;
      });
    return result;
  }

  size_t numFeatures() const { return features_.size(); }
  int newestFrameIndex() const { return newest_frame_index_; }
  void clear()
  {
    features_.clear();
    newest_frame_index_ = -1;
  }

private:
  struct Entry_s {
    std::vector<Observation_s> observations_;  // sorted by (frame_index, camera_id)
    int last_frame_ = -1;
  };
  std::unordered_map<int, Entry_s> features_;
  int newest_frame_index_ = -1;
};

}  // namespace backend
}  // namespace vins
