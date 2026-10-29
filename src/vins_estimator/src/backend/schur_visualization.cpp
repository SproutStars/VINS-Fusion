#include "backend/schur_visualization.h"

#include <algorithm>
#include <cmath>

namespace vins
{
namespace backend
{

void MarginPublicationHistory::retain(const std::unordered_set<int> &live_ids)
{
    for (auto it = published_.begin(); it != published_.end();)
    {
        if (live_ids.count(*it) == 0)
            it = published_.erase(it);
        else
            ++it;
    }
}

namespace
{
bool reliableMarginPoint(const EkfState &state,
                         const std::vector<CameraExtrinsics> &cameras,
                         const UpdateOptions &options, const FeatureTrack &track)
{
    std::unordered_set<int> clones;
    std::vector<Eigen::Vector3d> rays, depth_axes;
    std::vector<double> depths;
    Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
    double squared_error_px = 0.0;
    if (!(options.pixel_noise > 0.0) || !(options.focal_length > 0.0)) return false;
    const double sigma = options.pixel_noise / options.focal_length;
    for (const auto &obs : track.observations)
    {
        if (obs.clone_index < 0 || obs.clone_index >= state.num_clones() ||
            obs.camera_id < 0 || static_cast<size_t>(obs.camera_id) >= cameras.size())
            return false;
        const auto &clone = state.clone(obs.clone_index);
        const auto &camera = cameras[obs.camera_id];
        const Eigen::Matrix3d Rwb = EkfState::rotation(clone.q);
        const Eigen::Matrix3d Rwc = Rwb * EkfState::rotation(camera.q_cb);
        const Eigen::Vector3d ray = track.world_point - (clone.p + Rwb * camera.p_cb);
        const Eigen::Vector3d pc = Rwc.transpose() * ray;
        if (!pc.allFinite() || !(pc.z() > 0.0)) return false;
        const Eigen::Vector2d projected = pc.head<2>() / pc.z();
        const double error_px = (projected - obs.uv).norm() * options.focal_length;
        if (!std::isfinite(error_px) || error_px > 3.0) return false;
        squared_error_px += error_px * error_px;
        Eigen::Matrix<double, 2, 3> projection;
        projection << 1.0, 0.0, -projected.x(), 0.0, 1.0, -projected.y();
        const Eigen::Matrix<double, 2, 3> J = projection * Rwc.transpose() / (pc.z() * sigma);
        information.noalias() += J.transpose() * J;
        clones.insert(obs.clone_index);
        rays.push_back(ray.normalized());
        depth_axes.push_back(Rwc.col(2));
        depths.push_back(pc.z());
    }
    if (clones.size() < 3 || std::sqrt(squared_error_px / rays.size()) > 2.0)
        return false;
    double max_parallax = 0.0;
    for (size_t i = 0; i < rays.size(); ++i)
        for (size_t j = i + 1; j < rays.size(); ++j)
            max_parallax = std::max(max_parallax,
                std::atan2(rays[i].cross(rays[j]).norm(), rays[i].dot(rays[j])));
    constexpr double kOneDegree = 0.017453292519943295;
    if (max_parallax < kOneDegree || !information.allFinite()) return false;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(information);
    if (eig.info() != Eigen::Success ||
        !(eig.eigenvalues().minCoeff() > 1e-12 * eig.eigenvalues().maxCoeff()))
        return false;
    const Eigen::Matrix3d covariance = eig.eigenvectors() *
        eig.eigenvalues().cwiseInverse().asDiagonal() * eig.eigenvectors().transpose();
    if (!covariance.allFinite()) return false;
    for (size_t i = 0; i < depths.size(); ++i)
    {
        const double variance = depth_axes[i].dot(covariance * depth_axes[i]);
        if (!std::isfinite(variance) || variance < 0.0 ||
            std::sqrt(variance) / depths[i] > 0.15)
            return false;
    }
    return true;
}
}  // namespace

SchurVisualizationSnapshot makeSchurVisualizationSnapshot(
    const EkfState &state, const std::vector<CameraExtrinsics> &cameras,
    const SchurUpdater &updater, const std::vector<FeatureTrack> &tracks,
    int keyframe_clone_index, double timestamp,
    const std::vector<int> &accepted_consumed_feature_ids,
    MarginPublicationHistory *margin_history)
{
    SchurVisualizationSnapshot snapshot;
    if (!state.initialized() || !state.imu().q.allFinite() || !state.imu().p.allFinite())
        return snapshot;

    snapshot.valid = true;
    snapshot.timestamp = timestamp;
    snapshot.q_wxyz = state.imu().q;
    snapshot.p = state.imu().p;

    for (const FeatureTrack &source : tracks)
    {
        if (source.observations.size() < 2)
            continue;

        FeatureTrack track = source;
        if (!updater.triangulate(state, cameras, track) || !track.world_point.allFinite())
            continue;

        const FeatureObservation *keyframe_observation = nullptr;
        if (keyframe_clone_index >= 0)
        {
            for (const FeatureObservation &observation : source.observations)
            {
                if (observation.clone_index == keyframe_clone_index)
                {
                    keyframe_observation = &observation;
                    break;
                }
            }
        }

        const FeatureObservation *display_observation =
            keyframe_observation != nullptr ? keyframe_observation : &source.observations.back();
        if (!display_observation->uv.allFinite())
            continue;

        SchurVisualizationPoint point;
        point.feature_id = source.id;
        point.world_point = track.world_point;
        point.normalized_uv = display_observation->uv;
        point.pixel_uv = display_observation->pixel_uv;
        snapshot.cloud_points.push_back(point);

        const bool consumed = std::find(accepted_consumed_feature_ids.begin(),
                                        accepted_consumed_feature_ids.end(), source.id) !=
                              accepted_consumed_feature_ids.end();
        if (consumed && reliableMarginPoint(state, cameras, updater.options(), track) &&
            (margin_history == nullptr || margin_history->mark(source.id)))
            snapshot.margin_points.push_back(point);

        if (keyframe_observation != nullptr && keyframe_observation->has_pixel_uv &&
            keyframe_observation->pixel_uv.allFinite())
        {
            snapshot.keyframe_points.push_back(point);
        }
    }

    return snapshot;
}

}  // namespace backend
}  // namespace vins
