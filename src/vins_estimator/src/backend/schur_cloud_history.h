/*******************************************************
 * Publication-only cloud observation history for the Schur backend.
 *
 * FeatureHistory owns the measurements that may enter a filter update and
 * erases a track as soon as it has been consumed. That is correct for the EKF,
 * but it shortens the history available to the point-cloud publication path:
 * a feature consumed in frame N and seen again in frame N+1 would only have a
 * one-frame baseline for triangulation. This class keeps a second, bounded copy
 * of the *real* observations (never interpolated or duplicated) so publication
 * can triangulate across consumption boundaries.
 *
 * It is deliberately isolated from the filter:
 * - FeatureHistory::eraseFeatures() does not touch this history.
 * - Nothing here is ever passed to SchurUpdater::update().
 * - The estimator only materialises tracks from it while a visualization
 *   consumer exists.
 *
 * Observations are stored by absolute frame index and resolved to clone
 * indices on read, so a sliding clone window needs only prune(); no remap is
 * required. Storage is bounded by both the clone window and hard per-track /
 * total-track caps, and containers are reserved from those caps.
 *******************************************************/

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "feature_history.h"
#include "schur_update.h"

namespace vins
{
namespace backend
{

class SchurCloudHistory {
public:
  static constexpr size_t kDefaultMaxTracks = 2048;
  static constexpr size_t kDefaultMaxObservationsPerTrack = 64;

  void configure(
    size_t _max_tracks = kDefaultMaxTracks, size_t _max_observations_per_track = kDefaultMaxObservationsPerTrack)
  {
    max_tracks_ = std::max<size_t>(1, _max_tracks);
    max_observations_per_track_ = std::max<size_t>(2, _max_observations_per_track);
    features_.reserve(std::min(max_tracks_, static_cast<size_t>(4096)));
  }

  // Convenience overload without raw pixels (tests / non-loop consumers).
  void
  addFrame(int _frame_index, int _camera_id, const std::vector<std::pair<int, Eigen::Vector2d>> & _observations)
  {
    std::vector<FeatureHistory::FrameObservation_s> converted;
    converted.reserve(_observations.size());
    for (const auto & observation : _observations)
      converted.push_back(FeatureHistory::FrameObservation_s{observation.first, observation.second});
    addFrameWithPixels(_frame_index, _camera_id, converted);
  }

  // Registers one camera's real observations for a frame. A stereo pair is two
  // calls with the same frame index and different camera ids. Both calls are
  // retained; only an exact duplicate (frame, camera) view is ignored.
  void addFrameWithPixels(
    int _frame_index, int _camera_id, const std::vector<FeatureHistory::FrameObservation_s> & _observations)
  {
    newest_frame_index_ = std::max(newest_frame_index_, _frame_index);
    for (const auto & observation : _observations) {
      auto found = features_.find(observation.feature_id_);
      if (found == features_.end()) {
        if (features_.size() >= max_tracks_)
          continue;  // hard bound; a live ID will re-enter when another expires
        Entry_s entry;
        entry.observations_.reserve(std::min(max_observations_per_track_, static_cast<size_t>(16)));
        found = features_.emplace(observation.feature_id_, std::move(entry)).first;
      }

      Entry_s & entry = found->second;
      auto & history = entry.observations_;
      const FeatureHistory::Observation_s record{
        _frame_index, _camera_id, observation.uv_, observation.pixel_uv_, observation.has_pixel_uv_};

      const bool append_to_back =
        history.empty() || history.back().frame_index_ < _frame_index ||
        (history.back().frame_index_ == _frame_index && history.back().camera_id_ < _camera_id);
      if (append_to_back) {
        history.push_back(record);
      } else {
        auto insert_at = history.begin();
        while (insert_at != history.end() &&
               (insert_at->frame_index_ < _frame_index ||
                (insert_at->frame_index_ == _frame_index && insert_at->camera_id_ < _camera_id)))
          ++insert_at;
        if (insert_at != history.end() && insert_at->frame_index_ == _frame_index && insert_at->camera_id_ == _camera_id)
          continue;  // duplicate view
        history.insert(insert_at, record);
      }
      entry.last_frame_ = std::max(entry.last_frame_, _frame_index);

      // Keep the newest observations. Tracks normally have at most
      // cameras * clone_window entries, so this only fires for abnormal IDs.
      if (history.size() > max_observations_per_track_) {
        const size_t drop = history.size() - max_observations_per_track_;
        history.erase(history.begin(), history.begin() + static_cast<std::ptrdiff_t>(drop));
      }
    }
  }

  // Mirrors FeatureHistory::trim(). Observations older than the oldest clone
  // cannot be resolved to a clone index and are removed. A feature disappears
  // once its last real observation is pruned.
  void prune(int _oldest_frame_index)
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

  // Materialises copies for publication. No filter state is read or written.
  // The return is deterministic (longest track first, then id) and capped.
  std::vector<FeatureTrack_s> tracks(int _oldest_frame_index, size_t _max_points) const
  {
    std::vector<FeatureTrack_s> result;
    if (_max_points == 0)
      return result;
    result.reserve(std::min(_max_points, features_.size()));
    for (const auto & feature : features_) {
      FeatureTrack_s track;
      track.id_ = feature.first;
      track.observations_.reserve(feature.second.observations_.size());
      for (const FeatureHistory::Observation_s & observation : feature.second.observations_) {
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
      if (track.observations_.size() >= 2)
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
    if (result.size() > _max_points)
      result.resize(_max_points);
    return result;
  }

  // Used by MarginPublicationHistory::retain() so a feature consumed by the
  // filter is still considered alive while its cloud history exists.
  void appendFeatureIds(std::unordered_set<int> & _out) const
  {
    _out.reserve(_out.size() + features_.size());
    for (const auto & feature : features_)
      _out.insert(feature.first);
  }

  bool contains(int _feature_id) const { return features_.count(_feature_id) != 0; }
  size_t numFeatures() const { return features_.size(); }
  int newestFrameIndex() const { return newest_frame_index_; }

  size_t numObservations() const
  {
    size_t total = 0;
    for (const auto & feature : features_)
      total += feature.second.observations_.size();
    return total;
  }

  size_t maxTracks() const { return max_tracks_; }
  size_t maxObservationsPerTrack() const { return max_observations_per_track_; }

  void clear()
  {
    features_.clear();
    newest_frame_index_ = -1;
  }

private:
  struct Entry_s {
    std::vector<FeatureHistory::Observation_s> observations_;  // sorted by (frame, camera)
    int last_frame_ = -1;
  };

  std::unordered_map<int, Entry_s> features_;
  int newest_frame_index_ = -1;
  size_t max_tracks_ = kDefaultMaxTracks;
  size_t max_observations_per_track_ = kDefaultMaxObservationsPerTrack;
};

}  // namespace backend
}  // namespace vins
