#include "backend/schur_visualization.h"

#include <algorithm>
#include <cmath>

namespace vins
{
namespace backend
{

void MarginPublicationHistory::retain(const std::unordered_set<int> & _live_ids)
{
  for (auto it = published_.begin(); it != published_.end();) {
    if (_live_ids.count(*it) == 0)
      it = published_.erase(it);
    else
      ++it;
  }
}

namespace
{
bool reliableMarginPoint(
  const EkfState & _state, const std::vector<CameraExtrinsics_s> & _cameras, const UpdateOptions_s & _options,
  const FeatureTrack_s & _track)
{
  std::unordered_set<int> clones;
  std::vector<Eigen::Vector3d> rays, depth_axes;
  std::vector<double> depths;
  Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
  double squared_error_px = 0.0;
  if (!(_options.pixel_noise_ > 0.0) || !(_options.focal_length_ > 0.0))
    return false;
  const double sigma = _options.pixel_noise_ / _options.focal_length_;
  for (const auto & obs : _track.observations_) {
    if (
      obs.clone_index_ < 0 || obs.clone_index_ >= _state.numClones() || obs.camera_id_ < 0 ||
      static_cast<size_t>(obs.camera_id_) >= _cameras.size())
      return false;
    const auto & clone = _state.clone(obs.clone_index_);
    const auto & camera = _cameras[obs.camera_id_];
    const Eigen::Matrix3d Rwb = EkfState::rotation(clone.q_);
    const Eigen::Matrix3d Rwc = Rwb * EkfState::rotation(camera.q_cb_);
    const Eigen::Vector3d ray = _track.world_point_ - (clone.p_ + Rwb * camera.p_cb_);
    const Eigen::Vector3d pc = Rwc.transpose() * ray;
    if (!pc.allFinite() || !(pc.z() > 0.0))
      return false;
    const Eigen::Vector2d projected = pc.head<2>() / pc.z();
    const double error_px = (projected - obs.uv_).norm() * _options.focal_length_;
    if (!std::isfinite(error_px) || error_px > 3.0)
      return false;
    squared_error_px += error_px * error_px;
    Eigen::Matrix<double, 2, 3> projection;
    projection << 1.0, 0.0, -projected.x(), 0.0, 1.0, -projected.y();
    const Eigen::Matrix<double, 2, 3> J = projection * Rwc.transpose() / (pc.z() * sigma);
    information.noalias() += J.transpose() * J;
    clones.insert(obs.clone_index_);
    rays.push_back(ray.normalized());
    depth_axes.push_back(Rwc.col(2));
    depths.push_back(pc.z());
  }
  if (clones.size() < 3 || std::sqrt(squared_error_px / rays.size()) > 2.0)
    return false;
  double max_parallax = 0.0;
  for (size_t i = 0; i < rays.size(); ++i)
    for (size_t j = i + 1; j < rays.size(); ++j)
      max_parallax = std::max(max_parallax, std::atan2(rays[i].cross(rays[j]).norm(), rays[i].dot(rays[j])));
  constexpr double kOneDegree = 0.017453292519943295;
  if (max_parallax < kOneDegree || !information.allFinite())
    return false;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(information);
  if (eig.info() != Eigen::Success || !(eig.eigenvalues().minCoeff() > 1e-12 * eig.eigenvalues().maxCoeff()))
    return false;
  const Eigen::Matrix3d covariance =
    eig.eigenvectors() * eig.eigenvalues().cwiseInverse().asDiagonal() * eig.eigenvectors().transpose();
  if (!covariance.allFinite())
    return false;
  for (size_t i = 0; i < depths.size(); ++i) {
    const double variance = depth_axes[i].dot(covariance * depth_axes[i]);
    if (!std::isfinite(variance) || variance < 0.0 || std::sqrt(variance) / depths[i] > 0.15)
      return false;
  }
  return true;
}
}  // namespace

SchurVisualizationSnapshot_s makeSchurVisualizationSnapshot(
  const EkfState & _state, const std::vector<CameraExtrinsics_s> & _cameras, const SchurUpdater & _updater,
  const std::vector<FeatureTrack_s> & _tracks, int _keyframe_clone_index, double _timestamp,
  const std::vector<int> & _accepted_consumed_feature_ids, MarginPublicationHistory * _margin_history)
{
  SchurVisualizationSnapshot_s snapshot;
  if (!_state.initialized() || !_state.imu().q_.allFinite() || !_state.imu().p_.allFinite())
    return snapshot;

  snapshot.valid_ = true;
  snapshot.timestamp_ = _timestamp;
  snapshot.q_wxyz_ = _state.imu().q_;
  snapshot.p_ = _state.imu().p_;

  for (const FeatureTrack_s & source : _tracks) {
    if (source.observations_.size() < 2)
      continue;

    FeatureTrack_s track = source;
    if (!_updater.triangulate(_state, _cameras, track) || !track.world_point_.allFinite())
      continue;

    const FeatureObservation_s * keyframe_observation = nullptr;
    if (_keyframe_clone_index >= 0) {
      for (const FeatureObservation_s & observation : source.observations_) {
        if (observation.clone_index_ == _keyframe_clone_index) {
          keyframe_observation = &observation;
          break;
        }
      }
    }

    const FeatureObservation_s * display_observation =
      keyframe_observation != nullptr ? keyframe_observation : &source.observations_.back();
    if (!display_observation->uv_.allFinite())
      continue;

    SchurVisualizationPoint_s point;
    point.feature_id_ = source.id_;
    point.world_point_ = track.world_point_;
    point.normalized_uv_ = display_observation->uv_;
    point.pixel_uv_ = display_observation->pixel_uv_;
    snapshot.cloud_points_.push_back(point);

    const bool consumed =
      std::find(_accepted_consumed_feature_ids.begin(), _accepted_consumed_feature_ids.end(), source.id_) !=
      _accepted_consumed_feature_ids.end();
    if (
      consumed && reliableMarginPoint(_state, _cameras, _updater.options(), track) &&
      (_margin_history == nullptr || _margin_history->mark(source.id_)))
      snapshot.margin_points_.push_back(point);

    if (
      keyframe_observation != nullptr && keyframe_observation->has_pixel_uv_ &&
      keyframe_observation->pixel_uv_.allFinite()) {
      snapshot.keyframe_points_.push_back(point);
    }
  }

  return snapshot;
}

}  // namespace backend
}  // namespace vins
