/*******************************************************
 * Publication-only visual snapshot for the Schur EKF backend.
 *
 * The Schur backend keeps its visual state in FeatureHistory and EkfState, not
 * in the Ceres FeatureManager. The visualisation/loop-fusion topics therefore
 * cannot use FeatureManager::estimated_depth/solve_flag once Ceres stops
 * running. This module triangulates copies of the retained tracks with the
 * current clone poses and packages the finite world points plus the current
 * frame's 2D observations. It never mutates the filter or its measurement
 * history.
 *******************************************************/

#pragma once

#include <eigen3/Eigen/Dense>
#include <unordered_set>
#include <vector>

#include "ekf_state.h"
#include "schur_update.h"

namespace vins
{
namespace backend
{

struct SchurVisualizationPoint_s {
  int feature_id_ = -1;
  Eigen::Vector3d world_point_ = Eigen::Vector3d::Zero();
  Eigen::Vector2d normalized_uv_ = Eigen::Vector2d::Zero();
  Eigen::Vector2d pixel_uv_ = Eigen::Vector2d::Zero();
};

struct SchurVisualizationSnapshot_s {
  bool valid_ = false;
  double timestamp_ = 0.0;  // sensor time used for diagnostics
  Eigen::Vector4d q_wxyz_ = Eigen::Vector4d(1, 0, 0, 0);
  Eigen::Vector3d p_ = Eigen::Vector3d::Zero();
  // All finite tracks retained in the current clone window.
  std::vector<SchurVisualizationPoint_s> cloud_points_;
  // Accepted, actually consumed tracks passing the stricter margin geometry
  // gate. At most one point per continuously tracked feature lifetime.
  std::vector<SchurVisualizationPoint_s> margin_points_;
  // Tracks that also have a camera-0 raw-pixel observation in the keyframe
  // clone. The loop-fusion keyframe message carries no per-point camera id, so
  // right-only keyframe observations are deliberately omitted here.
  std::vector<SchurVisualizationPoint_s> keyframe_points_;
};

// Publication bookkeeping only. Retain IDs while they occur in the current
// image OR retained measurement history, including across consumed segments.
// Storage is bounded by the live tracks, not by the length of the run. A lost
// feature is forgotten once no retained observations can publish it again.
class MarginPublicationHistory {
public:
  void retain(const std::unordered_set<int> & _live_ids);
  bool mark(int _feature_id) { return published_.insert(_feature_id).second; }
  void clear() { published_.clear(); }
  size_t size() const { return published_.size(); }

private:
  std::unordered_set<int> published_;
};

// Margin-only quality thresholds: >=3 distinct clones, >=1 degree of actual
// camera-ray parallax (rotation alone is insufficient), RMS <=2 px and every
// residual <=3 px. The point information matrix uses pixel_noise/focal_length;
// sqrt(ray_z^T information^-1 ray_z)/depth must be <=15% in every view.
// This is a conditional depth precision estimate with fixed poses, not a claim
// about total world-point uncertainty. No distance clipping or smoothing.
// Builds a publication snapshot from copies of `tracks`. `keyframe_clone_index`
// is the clone corresponding to the image frame being published; pass a
// negative value to skip keyframe-point selection. The returned snapshot is
// invalid only when the filter state itself is not usable.
SchurVisualizationSnapshot_s makeSchurVisualizationSnapshot(
  const EkfState & _state, const std::vector<CameraExtrinsics_s> & _cameras, const SchurUpdater & _updater,
  const std::vector<FeatureTrack_s> & _tracks, int _keyframe_clone_index, double _timestamp,
  const std::vector<int> & _accepted_consumed_feature_ids = {},
  MarginPublicationHistory * _margin_history = nullptr);

}  // namespace backend
}  // namespace vins
