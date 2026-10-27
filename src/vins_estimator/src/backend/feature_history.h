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

struct TrackSelection
{
    std::vector<FeatureTrack> tracks;
    // Eligible tracks before max_features truncation. A track that is both
    // lost and marginalizable is counted once as marginalized, matching
    // OpenVINS' overlap removal between feats_lost and feats_marg.
    int lost_count = 0;
    int marginalized_count = 0;
};

class FeatureHistory
{
  public:
    struct Observation
    {
        int frame_index = 0;
        int camera_id = 0;
        Eigen::Vector2d uv = Eigen::Vector2d::Zero();  // normalised, undistorted
    };

    using TrackSelection = ::vins::backend::TrackSelection;

    // Registers one image frame. `observations` maps feature id -> normalised uv.
    void add_frame(int frame_index, int camera_id,
                   const std::vector<std::pair<int, Eigen::Vector2d>> &observations)
    {
        newest_frame_index_ = std::max(newest_frame_index_, frame_index);
        for (const auto &observation : observations)
        {
            Entry &entry = features_[observation.first];
            if (!entry.observations.empty() &&
                entry.observations.back().frame_index == frame_index)
                continue;  // duplicate id in one frame: keep the first observation
            entry.observations.push_back(Observation{frame_index, camera_id, observation.second});
            entry.last_frame = std::max(entry.last_frame, frame_index);
        }
    }

    // Drops observations older than the oldest retained clone. Entries that
    // become empty are removed. A track that has merely been unobserved for a
    // frame or two is retained until it is selected and erased by the caller.
    void trim(int oldest_frame_index)
    {
        for (auto it = features_.begin(); it != features_.end();)
        {
            Entry &entry = it->second;
            auto first_kept = entry.observations.begin();
            while (first_kept != entry.observations.end() &&
                   first_kept->frame_index < oldest_frame_index)
                ++first_kept;
            entry.observations.erase(entry.observations.begin(), first_kept);
            if (entry.observations.empty())
            {
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
    void trim(int oldest_frame_index, int /*newest_frame_index*/)
    {
        trim(oldest_frame_index);
    }

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
    TrackSelection select_tracks(int oldest_frame_index, int newest_frame_index,
                                 int min_observations, size_t max_features,
                                 bool window_full = true) const
    {
        TrackSelection selection;
        for (const auto &feature : features_)
        {
            const Entry &entry = feature.second;
            if (static_cast<int>(entry.observations.size()) < min_observations)
                continue;

            const bool marginalized =
                window_full &&
                entry.observations.front().frame_index <= oldest_frame_index;
            const bool lost = entry.last_frame < newest_frame_index;
            if (!marginalized && !lost)
                continue;

            if (marginalized)
                ++selection.marginalized_count;
            else
                ++selection.lost_count;

            FeatureTrack track;
            track.id = feature.first;
            track.observations.reserve(entry.observations.size());
            for (const Observation &observation : entry.observations)
            {
                FeatureObservation resolved;
                resolved.clone_index = observation.frame_index - oldest_frame_index;
                resolved.camera_id = observation.camera_id;
                resolved.uv = observation.uv;
                track.observations.push_back(resolved);
            }
            selection.tracks.push_back(std::move(track));
        }

        std::sort(selection.tracks.begin(), selection.tracks.end(),
                  [](const FeatureTrack &a, const FeatureTrack &b) {
                      if (a.observations.size() != b.observations.size())
                          return a.observations.size() > b.observations.size();
                      return a.id < b.id;
                  });
        if (selection.tracks.size() > max_features)
            selection.tracks.resize(max_features);
        return selection;
    }

    // Primary selection entry point used by the estimator. It returns both the
    // selected tracks and the lost/marginalized diagnostics. Passing
    // window_full=false suppresses the marginalization eligibility class.
    TrackSelection tracks(int oldest_frame_index, int newest_frame_index,
                          int min_observations, size_t max_features,
                          bool window_full) const
    {
        return select_tracks(oldest_frame_index, newest_frame_index,
                             min_observations, max_features, window_full);
    }

    // Compatibility overload: old callers that do not know whether the window
    // is full keep the original lost-or-marginalized behavior.
    TrackSelection tracks(int oldest_frame_index, int newest_frame_index,
                          int min_observations, size_t max_features) const
    {
        return tracks(oldest_frame_index, newest_frame_index,
                      min_observations, max_features, true);
    }

    // Backwards-compatible wrapper using the newest frame registered by
    // add_frame(). New code should prefer select_tracks() so it can log the
    // lost/marginalized counts.
    std::vector<FeatureTrack> tracks(int oldest_frame_index, int min_observations,
                                     size_t max_features) const
    {
        return select_tracks(oldest_frame_index, newest_frame_index_,
                             min_observations, max_features)
            .tracks;
    }

    // Removes tracks whose measurements were actually accumulated by the
    // update. Call with UpdateSummary::used_feature_ids, not with rejected or
    // merely consumed ids, so unused tracks remain available for a later frame.
    void erase_features(const std::vector<int> &feature_ids)
    {
        for (const int id : feature_ids)
            features_.erase(id);
    }

    size_t num_features() const { return features_.size(); }
    int newest_frame_index() const { return newest_frame_index_; }
    void clear()
    {
        features_.clear();
        newest_frame_index_ = -1;
    }

  private:
    struct Entry
    {
        std::vector<Observation> observations;  // sorted by frame_index
        int last_frame = -1;
    };
    std::unordered_map<int, Entry> features_;
    int newest_frame_index_ = -1;
};

}  // namespace backend
}  // namespace vins
