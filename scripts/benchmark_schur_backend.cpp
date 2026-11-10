// Standalone Schur-backend microbenchmark (Eigen only, no gtest / ROS).
//
// Build (no CMake required):
//   g++ -std=c++17 -O2 -DNDEBUG -I src/vins_estimator/src -I
//   /usr/include/eigen3
//       src/vins_estimator/src/backend/ekf_state.cpp
//       src/vins_estimator/src/backend/schur_update.cpp
//       scripts/benchmark_schur_backend.cpp -o benchmark_schur_backend
//
// The same source links against either the baseline or the optimized
// schur_update.cpp.  It runs a fixed synthetic scene (5 clones, 24 tracks,
// fixed RNG seed) through repeated reduce_track() and update() calls and prints
// machine-readable key=value lines.  The checksums must match across builds;
// the timings are meant to be compared externally.  There are no pass/fail
// thresholds based on wall-clock time.
//
// State/track copies used to reset the update scene are outside the update
// timing region, so the reported update latency is the update call itself.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "backend/ekf_state.h"
#include "backend/schur_update.h"

namespace
{

using namespace vins::backend;

constexpr unsigned int K_SEED = 12345u;
constexpr int K_CLONES = 5;
constexpr int K_TRACKS = 24;
constexpr int K_WARMUP_ITERATIONS = 5;
constexpr int K_REDUCE_ITERATIONS = 500;
constexpr int K_UPDATE_ITERATIONS = 100;

Eigen::Vector4d quat(const Eigen::Vector3d & _axis_angle)
{
  return EkfState::quaternion_from_axis_angle(_axis_angle);
}

void buildState(
  EkfState & _state, const std::vector<Eigen::Vector4d> & _qs, const std::vector<Eigen::Vector3d> & _ps)
{
  const int n = static_cast<int>(_qs.size());
  EkfOptions opt;
  opt.window_size = n + 2;
  _state.reset(
    opt, 0.0, _qs[0], _ps[0], Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  _state.augment_clone(0.0);
  for (int i = 1; i < n; ++i) {
    _state.imu().q = _qs[static_cast<std::size_t>(i)];
    _state.imu().p = _ps[static_cast<std::size_t>(i)];
    _state.refresh_imu_fej();
    _state.augment_clone(static_cast<double>(i));
  }
}

void regularizeCovariance(EkfState & _state) { _state.covariance().diagonal().array() += 1e-8; }

CameraExtrinsics makeCamera()
{
  CameraExtrinsics cam;
  cam.q_cb = quat(Eigen::Vector3d(0.01, -0.01, 0.005));
  cam.p_cb = Eigen::Vector3d(0.02, 0.0, -0.01);
  return cam;
}

Eigen::Vector2d projectWorld(
  const Eigen::Vector4d & _q, const Eigen::Vector3d & _p, const CameraExtrinsics & _cam,
  const Eigen::Vector3d & _world_point)
{
  const Eigen::Matrix3d R_i = EkfState::rotation(_q);
  const Eigen::Matrix3d R_cb = EkfState::rotation(_cam.q_cb);
  const Eigen::Vector3d u = R_i.transpose() * (_world_point - _p);
  const Eigen::Vector3d p_c = R_cb.transpose() * (u - _cam.p_cb);
  return Eigen::Vector2d(p_c.x() / p_c.z(), p_c.y() / p_c.z());
}

struct Scene_s {
  EkfState state_;
  std::vector<FeatureTrack> tracks_;
  CameraExtrinsics cam_;
};

Scene_s makeScene(const UpdateOptions & _options)
{
  Scene_s scene;
  scene.cam_ = makeCamera();

  std::vector<Eigen::Vector4d> qs;
  std::vector<Eigen::Vector3d> ps;
  for (int i = 0; i < K_CLONES; ++i) {
    const double t = static_cast<double>(i);
    ps.push_back(Eigen::Vector3d(0.25 * t, 0.05 * t * t, 0.01 * t));
    qs.push_back(quat(Eigen::Vector3d(0.0, -0.02 * t, 0.01 * t)));
  }
  buildState(scene.state_, qs, ps);
  regularizeCovariance(scene.state_);

  SchurUpdater updater(_options);
  std::mt19937 rng(K_SEED);
  std::normal_distribution<double> pixel_jitter(0.0, 0.5 / 460.0);  // 0.5 px
  for (int f = 0; f < K_TRACKS; ++f) {
    const int len = 3 + (f % 3);
    const int start = f % 3;
    const int stop = std::min(start + len, scene.state_.num_clones());
    const Eigen::Vector3d point(0.55 + 0.07 * (f % 6), 0.35 + 0.09 * (f / 6), 3.8 + 0.13 * (f % 4));
    FeatureTrack track;
    track.id = 4000 + f;
    for (int ci = start; ci < stop; ++ci) {
      FeatureObservation obs;
      obs.clone_index = ci;
      obs.camera_id = 0;
      obs.uv = projectWorld(scene.state_.clone(ci).q, scene.state_.clone(ci).p, scene.cam_, point) +
               Eigen::Vector2d(pixel_jitter(rng), pixel_jitter(rng));
      track.observations.push_back(obs);
    }

    if (updater.triangulate(scene.state_, {scene.cam_}, track))
      scene.tracks_.push_back(track);
  }
  return scene;
}

}  // namespace

int main()
{
  UpdateOptions options;
  options.max_features = 40;
  options.min_features = 1;
  options.min_update_observations = 1;
  options.condition_limit = 0.0;

  SchurUpdater updater(options);
  const Scene_s scene = makeScene(options);
  const int num_tracks = static_cast<int>(scene.tracks_.size());
  if (num_tracks == 0) {
    std::fprintf(stderr, "error: no triangulated synthetic tracks\n");
    return 1;
  }

  // Warm up the scratch buffers once.
  double reduce_checksum = 0.0;
  for (int it = 0; it < K_WARMUP_ITERATIONS; ++it) {
    for (const FeatureTrack & track : scene.tracks_) {
      const TrackReduction red = updater.reduce_track(scene.state_, {scene.cam_}, track);
      if (red.valid)
        reduce_checksum += red.chi2;
    }
  }
  {
    EkfState warm_state = scene.state_;
    std::vector<FeatureTrack> warm_tracks = scene.tracks_;
    (void)updater.update(warm_state, warm_tracks, {scene.cam_});
  }

  reduce_checksum = 0.0;
  const auto reduce_t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < K_REDUCE_ITERATIONS; ++it) {
    for (const FeatureTrack & track : scene.tracks_) {
      const TrackReduction red = updater.reduce_track(scene.state_, {scene.cam_}, track);
      if (!red.valid)
        continue;
      reduce_checksum += red.chi2 + red.rms_residual_px + red.A_point.sum() + red.b_point.sum() +
                         static_cast<double>(red.gate_dof);
    }
  }
  const auto reduce_t1 = std::chrono::steady_clock::now();
  const double reduce_ms = std::chrono::duration<double, std::milli>(reduce_t1 - reduce_t0).count();
  const int reduce_calls = K_REDUCE_ITERATIONS * num_tracks;

  double update_checksum = 0.0;
  double update_ms = 0.0;
  for (int it = 0; it < K_UPDATE_ITERATIONS; ++it) {
    EkfState work_state = scene.state_;
    std::vector<FeatureTrack> work_tracks = scene.tracks_;
    const auto t0 = std::chrono::steady_clock::now();
    const UpdateSummary summary = updater.update(work_state, work_tracks, {scene.cam_});
    const auto t1 = std::chrono::steady_clock::now();
    update_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    update_checksum += work_state.covariance().trace() + static_cast<double>(summary.rank) +
                       static_cast<double>(summary.features_used) +
                       static_cast<double>(summary.observations_used) +
                       static_cast<double>(summary.applied ? 1 : 0);
  }

  std::printf("scene clones=%d tracks=%d seed=%u\n", K_CLONES, num_tracks, K_SEED);
  std::printf(
    "reduce iterations=%d calls=%d elapsed_ms=%.6f us_per_call=%.6f "
    "checksum=%.17g\n",
    K_REDUCE_ITERATIONS, reduce_calls, reduce_ms, 1000.0 * reduce_ms / static_cast<double>(reduce_calls),
    reduce_checksum);
  std::printf(
    "update iterations=%d calls=%d elapsed_ms=%.6f us_per_call=%.6f "
    "checksum=%.17g\n",
    K_UPDATE_ITERATIONS, K_UPDATE_ITERATIONS, update_ms,
    1000.0 * update_ms / static_cast<double>(K_UPDATE_ITERATIONS), update_checksum);
  return 0;
}
