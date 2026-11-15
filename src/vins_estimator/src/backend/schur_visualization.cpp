#include "backend/schur_visualization.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

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
constexpr double kPi = 3.14159265358979323846;
constexpr int kPublicationCameraId = 0;

struct ReliabilityProfile_s {
  double max_rms_px = 0.0;
  double max_residual_px = 0.0;
  double min_parallax_rad = 0.0;
  double max_depth_sigma_ratio = 0.0;
  int min_clones = 0;
  bool stereo_pair_satisfies_two_views = false;
  bool exact_parallax = false;
};

CameraExtrinsics_s
publicationCamera(const EkfState & _state, const std::vector<CameraExtrinsics_s> & _cameras, int _camera_id)
{
  CameraExtrinsics_s camera = _cameras.at(_camera_id);
  if (_state.calibrationEnabled() && _camera_id < _state.cameraCount()) {
    const auto & calibration = _state.cameraCalib(_camera_id);
    camera.q_cb_ = calibration.q_cb_;
    camera.p_cb_ = calibration.p_cb_;
    camera.td_ = _state.sharedTimeOffset();
  }
  return camera;
}

double rayAngle(const Eigen::Vector3d & _a, const Eigen::Vector3d & _b)
{
  return std::atan2(_a.cross(_b).norm(), _a.dot(_b));
}

// Cheap O(observations) stereo test. A track with two different camera ids at
// one clone has a real extrinsic baseline and does not need three timestamps.
bool hasStereoView(const std::vector<CameraExtrinsics_s> & _cameras, const FeatureTrack_s & _track)
{
  std::unordered_map<int, int> clone_camera;
  clone_camera.reserve(_track.observations_.size());
  for (const auto & obs : _track.observations_) {
    if (obs.camera_id_ < 0 || static_cast<size_t>(obs.camera_id_) >= _cameras.size())
      continue;
    const auto inserted = clone_camera.emplace(obs.clone_index_, obs.camera_id_);
    if (!inserted.second && inserted.first->second != obs.camera_id_)
      return true;
  }
  return false;
}

// Fixed-pose conditional reliability. The clone poses are held constant while
// the pixel noise is propagated through the reprojection Jacobian. Parallax is
// evaluated in O(n) using two reference rays rather than all view pairs.
bool reliablePoint(
  const EkfState & _state, const std::vector<CameraExtrinsics_s> & _cameras, const UpdateOptions_s & _options,
  const FeatureTrack_s & _track, const ReliabilityProfile_s & _profile)
{
  if (!(_options.pixel_noise_ > 0.0) || !(_options.focal_length_ > 0.0))
    return false;
  if (!_track.world_point_.allFinite() || _track.observations_.size() < 2)
    return false;
  if (
    !(_profile.max_rms_px > 0.0) || !(_profile.max_residual_px > 0.0) ||
    !(_profile.max_depth_sigma_ratio > 0.0) || _profile.min_clones < 2)
    return false;

  const double sigma = _options.pixel_noise_ / _options.focal_length_;
  std::unordered_set<std::uint64_t> views;
  std::unordered_map<int, int> clone_camera;
  std::vector<Eigen::Vector3d> rays;
  std::vector<Eigen::Vector3d> depth_axes;
  std::vector<double> depths;
  rays.reserve(_track.observations_.size());
  depth_axes.reserve(_track.observations_.size());
  depths.reserve(_track.observations_.size());
  views.reserve(_track.observations_.size());
  clone_camera.reserve(_track.observations_.size());

  Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
  double squared_error_px = 0.0;
  bool has_stereo_view = false;

  for (const auto & obs : _track.observations_) {
    if (
      obs.clone_index_ < 0 || obs.clone_index_ >= _state.numClones() || obs.camera_id_ < 0 ||
      static_cast<size_t>(obs.camera_id_) >= _cameras.size())
      return false;
    if (!obs.uv_.allFinite())
      return false;

    const auto & clone = _state.clone(obs.clone_index_);
    const auto camera = publicationCamera(_state, _cameras, obs.camera_id_);
    const Eigen::Matrix3d Rwb = EkfState::rotation(clone.q_);
    const Eigen::Matrix3d Rwc = Rwb * EkfState::rotation(camera.q_cb_);
    const Eigen::Vector3d camera_center = clone.p_ + Rwb * camera.p_cb_;
    const Eigen::Vector3d ray = _track.world_point_ - camera_center;
    const Eigen::Vector3d pc = Rwc.transpose() * ray;
    if (!pc.allFinite() || !(pc.z() > 0.0))
      return false;

    const Eigen::Vector2d projected = pc.head<2>() / pc.z();
    const double error_px =
      (projected - SchurUpdater::correctedObservation(_state, obs)).norm() * _options.focal_length_;
    if (!std::isfinite(error_px) || error_px > _profile.max_residual_px)
      return false;
    squared_error_px += error_px * error_px;

    Eigen::Matrix<double, 2, 3> projection;
    projection << 1.0, 0.0, -projected.x(), 0.0, 1.0, -projected.y();
    const Eigen::Matrix<double, 2, 3> J = projection * Rwc.transpose() / (pc.z() * sigma);
    information.noalias() += J.transpose() * J;

    const std::uint64_t view_key =
      (static_cast<std::uint64_t>(static_cast<std::uint32_t>(obs.clone_index_)) << 32) |
      static_cast<std::uint32_t>(obs.camera_id_);
    views.insert(view_key);
    const auto inserted = clone_camera.emplace(obs.clone_index_, obs.camera_id_);
    if (!inserted.second && inserted.first->second != obs.camera_id_)
      has_stereo_view = true;

    rays.push_back(ray.normalized());
    depth_axes.push_back(Rwc.col(2));
    depths.push_back(pc.z());
  }

  if (views.size() < 2 || rays.empty())
    return false;
  if (std::sqrt(squared_error_px / static_cast<double>(rays.size())) > _profile.max_rms_px)
    return false;

  const bool two_view_stereo = _profile.stereo_pair_satisfies_two_views && has_stereo_view && views.size() >= 2;
  if (static_cast<int>(clone_camera.size()) < _profile.min_clones && !two_view_stereo)
    return false;

  double max_parallax = 0.0;
  if (_profile.exact_parallax) {
    // Margin keeps the historical all-pairs definition so the default mono
    // gate is bit-for-bit unchanged. The accepted set is bounded by
    // max_features and each track by the clone window, so this stays cheap.
    for (size_t i = 0; i < rays.size(); ++i)
      for (size_t j = i + 1; j < rays.size(); ++j)
        max_parallax = std::max(max_parallax, rayAngle(rays[i], rays[j]));
  } else {
    // Cloud publication uses an O(n) two-reference approximation.
    size_t farthest = 0;
    for (size_t i = 1; i < rays.size(); ++i) {
      const double angle = rayAngle(rays.front(), rays[i]);
      if (angle > max_parallax) {
        max_parallax = angle;
        farthest = i;
      }
    }
    for (size_t i = 0; i < rays.size(); ++i)
      max_parallax = std::max(max_parallax, rayAngle(rays[farthest], rays[i]));
  }
  if (!std::isfinite(max_parallax) || max_parallax < _profile.min_parallax_rad)
    return false;

  if (!information.allFinite())
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
    if (!std::isfinite(variance) || variance < 0.0 || std::sqrt(variance) / depths[i] > _profile.max_depth_sigma_ratio)
      return false;
  }
  return true;
}

ReliabilityProfile_s marginProfile(const SchurVisualizationQuality_s & _quality)
{
  ReliabilityProfile_s profile;
  profile.max_rms_px = _quality.margin_max_rms_px_;
  profile.max_residual_px = _quality.margin_max_residual_px_;
  profile.min_parallax_rad = _quality.margin_min_parallax_deg_ * kPi / 180.0;
  profile.max_depth_sigma_ratio = _quality.margin_max_depth_sigma_ratio_;
  profile.min_clones = 3;
  profile.stereo_pair_satisfies_two_views = _quality.stereo_pair_satisfies_two_views_;
  profile.exact_parallax = true;
  return profile;
}

ReliabilityProfile_s cloudProfile(const SchurVisualizationQuality_s & _quality)
{
  ReliabilityProfile_s profile;
  profile.max_rms_px = _quality.cloud_max_rms_px_;
  profile.max_residual_px = _quality.cloud_max_residual_px_;
  profile.min_parallax_rad = _quality.cloud_min_parallax_deg_ * kPi / 180.0;
  profile.max_depth_sigma_ratio = _quality.cloud_max_depth_sigma_ratio_;
  profile.min_clones = 2;
  profile.stereo_pair_satisfies_two_views = true;
  return profile;
}

const FeatureObservation_s *
displayObservation(const FeatureTrack_s & _track, const FeatureObservation_s * _keyframe_observation)
{
  if (_keyframe_observation != nullptr && _keyframe_observation->uv_.allFinite())
    return _keyframe_observation;
  for (auto it = _track.observations_.rbegin(); it != _track.observations_.rend(); ++it) {
    if (it->camera_id_ == kPublicationCameraId && it->uv_.allFinite())
      return &*it;
  }
  for (auto it = _track.observations_.rbegin(); it != _track.observations_.rend(); ++it) {
    if (it->uv_.allFinite())
      return &*it;
  }
  return nullptr;
}

const FeatureObservation_s * keyframeObservation(const FeatureTrack_s & _track, int _keyframe_clone_index)
{
  if (_keyframe_clone_index < 0)
    return nullptr;
  for (const FeatureObservation_s & observation : _track.observations_) {
    if (
      observation.clone_index_ == _keyframe_clone_index && observation.camera_id_ == kPublicationCameraId &&
      observation.uv_.allFinite())
      return &observation;
  }
  return nullptr;
}

}  // namespace

SchurVisualizationSnapshot_s makeSchurVisualizationSnapshot(
  const EkfState & _state, const std::vector<CameraExtrinsics_s> & _cameras, const SchurUpdater & _updater,
  const std::vector<FeatureTrack_s> & _tracks, int _keyframe_clone_index, double _timestamp,
  const std::vector<int> & _accepted_consumed_feature_ids, MarginPublicationHistory * _margin_history)
{
  return makeSchurVisualizationSnapshot(
    _state, _cameras, _updater, _tracks, _tracks, _keyframe_clone_index, _timestamp,
    _accepted_consumed_feature_ids, _margin_history, SchurVisualizationQuality_s());
}

SchurVisualizationSnapshot_s makeSchurVisualizationSnapshot(
  const EkfState & _state, const std::vector<CameraExtrinsics_s> & _cameras, const SchurUpdater & _updater,
  const std::vector<FeatureTrack_s> & _measurement_tracks, const std::vector<FeatureTrack_s> & _cloud_tracks,
  int _keyframe_clone_index, double _timestamp, const std::vector<int> & _accepted_consumed_feature_ids,
  MarginPublicationHistory * _margin_history, const SchurVisualizationQuality_s & _quality)
{
  SchurVisualizationSnapshot_s snapshot;
  if (!_state.initialized() || !_state.imu().q_.allFinite() || !_state.imu().p_.allFinite())
    return snapshot;

  snapshot.valid_ = true;
  snapshot.timestamp_ = _timestamp;
  snapshot.q_wxyz_ = _state.imu().q_;
  snapshot.p_ = _state.imu().p_;

  std::unordered_set<int> accepted_ids;
  accepted_ids.reserve(_accepted_consumed_feature_ids.size());
  accepted_ids.insert(_accepted_consumed_feature_ids.begin(), _accepted_consumed_feature_ids.end());

  const ReliabilityProfile_s margin_profile = marginProfile(_quality);
  const ReliabilityProfile_s cloud_profile = cloudProfile(_quality);

  snapshot.margin_points_.reserve(std::min(_measurement_tracks.size(), static_cast<size_t>(4096)));
  snapshot.keyframe_points_.reserve(std::min(_measurement_tracks.size(), static_cast<size_t>(4096)));
  snapshot.cloud_points_.reserve(std::min(_cloud_tracks.size(), static_cast<size_t>(4096)));

  // One triangulation per source track. When measurement and cloud sources are
  // literally the same vector (the historical overload and the mono path), all
  // outputs share the same refined world point instead of triangulating twice.
  const auto publish_track =
    [&](const FeatureTrack_s & source, bool _want_cloud, bool _want_margin, bool _want_keyframe)
  {
    if (source.observations_.size() < 2)
      return;
    FeatureTrack_s track = source;
    if (!_updater.triangulate(_state, _cameras, track) || !track.world_point_.allFinite())
      return;

    const FeatureObservation_s * keyframe_observation =
      _want_keyframe ? keyframeObservation(source, _keyframe_clone_index) : nullptr;
    const FeatureObservation_s * display_observation = displayObservation(source, keyframe_observation);
    if (display_observation == nullptr)
      return;

    SchurVisualizationPoint_s point;
    point.feature_id_ = source.id_;
    point.world_point_ = track.world_point_;
    point.normalized_uv_ = display_observation->uv_;
    point.pixel_uv_ = display_observation->pixel_uv_;

    if (
      _want_margin && accepted_ids.count(source.id_) != 0 &&
      reliablePoint(_state, _cameras, _updater.options(), track, margin_profile) &&
      (_margin_history == nullptr || _margin_history->mark(source.id_)))
      snapshot.margin_points_.push_back(point);

    if (
      _want_keyframe && keyframe_observation != nullptr && keyframe_observation->has_pixel_uv_ &&
      keyframe_observation->pixel_uv_.allFinite()) {
      SchurVisualizationPoint_s keyframe_point = point;
      keyframe_point.normalized_uv_ = keyframe_observation->uv_;
      keyframe_point.pixel_uv_ = keyframe_observation->pixel_uv_;
      snapshot.keyframe_points_.push_back(keyframe_point);
    }

    if (
      _want_cloud && !((_quality.gate_all_cloud_tracks_ || hasStereoView(_cameras, source)) &&
                       !reliablePoint(_state, _cameras, _updater.options(), track, cloud_profile)))
      snapshot.cloud_points_.push_back(point);
  };

  if (&_measurement_tracks == &_cloud_tracks) {
    // Non-extended-history path: a single triangulation serves cloud, margin
    // and loop outputs. This keeps the default mono publication cost unchanged.
    for (const FeatureTrack_s & source : _measurement_tracks)
      publish_track(source, true, true, true);
  } else {
    // Measurement tracks are the only source of margin and loop-fusion points.
    // A longer publication history is intentionally invisible there.
    for (const FeatureTrack_s & source : _measurement_tracks)
      publish_track(source, false, true, true);
    // Cloud-only tracks are publication data. They never become margin or
    // loop points, even if they contain a camera-0 view at the keyframe clone.
    for (const FeatureTrack_s & source : _cloud_tracks)
      publish_track(source, true, false, false);
  }

  return snapshot;
}

}  // namespace backend
}  // namespace vins
