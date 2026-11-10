/*******************************************************
 * Schur-complement visual update for the ARM-oriented EKF backend.
 *
 * Features are nuisance parameters: they are NOT part of the covariance.  For
 * every feature track the joint (clone-window + 3D point) normal equations are
 * built and the point is eliminated with a 3x3 Schur complement, yielding an
 * equivalent measurement information matrix A (n x n) and vector b (n) for the
 * clone window.
 *
 * A is structurally singular: every observation row touches only one clone, so
 * the live IMU block is empty, and the visual problem carries a 6-DoF pose
 * gauge plus (for mono) an unobservable scale.  The update therefore builds a
 * rank-revealing square root A_cc = Lc^T Lc of the clone block (eigenvalue
 * truncation, relative tolerance `info_rank_rtol`) and runs a standard Kalman
 * update with unit measurement noise:
 *
 *     S = L P L^T + I,  K = P L^T S^-1,  dx = K z,
 *     P <- (I - K L) P (I - K L)^T + K K^T.
 *
 * Unlike the previous information-form compression (H = A, R = A) plus a
 * 1e-12 ridge, the discarded null directions receive *no* information, which
 * is what keeps a long run from slowly drifting and then diverging.
 *
 * Conventions follow ekf_state.h (right/body perturbation, q = [w,x,y,z]).
 *******************************************************/

#pragma once

#include <eigen3/Eigen/Dense>
#include <string>
#include <unordered_map>
#include <vector>

#include "ekf_state.h"

namespace vins
{
namespace backend
{

// Camera-to-body extrinsics (constant, from the config).
struct CameraExtrinsics_s {
  Eigen::Vector4d q_cb_ = Eigen::Vector4d(1, 0, 0, 0);  // body_T_cam rotation
  Eigen::Vector3d p_cb_ = Eigen::Vector3d::Zero();      // body_T_cam translation
};

struct FeatureObservation_s {
  int clone_index_ = 0;  // index into EkfState::clone(i)
  int camera_id_ = 0;
  Eigen::Vector2d uv_ = Eigen::Vector2d::Zero();  // normalised, undistorted (x/z, y/z)
  // Raw image coordinates for loop-fusion keyframes. Measurement updates only
  // use `uv`; this field is publication metadata and never enters the EKF.
  Eigen::Vector2d pixel_uv_ = Eigen::Vector2d::Zero();
  bool has_pixel_uv_ = false;
};

struct FeatureTrack_s {
  int id_ = -1;
  std::vector<FeatureObservation_s> observations_;  // sorted by clone_index ascending
  // World-frame feature point, written by SchurUpdater::triangulate() from
  // the current clone poses on every update (OpenVINS re-triangulates a
  // feature each time it is used, and a feature is consumed by at most one
  // update, so no cross-frame cache is kept). `has_point` reports whether the
  // point came from a successful triangulation rather than from an earlier
  // caller-supplied value.
  Eigen::Vector3d world_point_ = Eigen::Vector3d::Zero();
  bool has_point_ = false;
};

struct UpdateOptions_s {
  int max_features_ = 120;            // cap on features used per frame (longest tracks first)
  int min_observations_ = 2;          // minimum observations per track
  int min_features_ = 12;             // skip the visual update below this many gated tracks
  int min_update_observations_ = 24;  // ... and below this many total observations
  double pixel_noise_ = 1.5;          // 1-sigma reprojection noise in pixels
  double focal_length_ = 460.0;       // pixels, converts pixel noise to normalised noise
  double chi2_multiplier_ = 5.0;      // gate = multiplier * chi2_quantile_0.95(dof)
  int triangulation_iterations_ = 5;
  double min_depth_ = 0.1;   // metres
  double max_depth_ = 60.0;  // metres
  // Eigenvalues of the Schur-reduced information matrix below this fraction
  // of the largest one are treated as the structural null space (pose gauge,
  // mono scale, unobserved modes) and receive no update.
  double info_rank_rtol_ = 1e-10;
  // A relative negative eigenvalue below this is treated as roundoff; a
  // larger one means the per-track Schur complements are inconsistent.
  double info_indefinite_tol_ = 1e-6;
  // 3x3 point-block conditioning: reject a track when lambda_min is below
  // this fraction of lambda_max (a near-degenerate triangulation).
  double point_block_rtol_ = 1e-8;
  // Reject the frame when lambda_max / lambda_min of the retained visual
  // information exceeds this value. Values <= 0 disable the check.
  double condition_limit_ = 1e6;
  // P6 starvation-adaptive weak update. When > 1.0 the chi2 gate threshold is
  // multiplied by this factor and the applied observation weight is divided by
  // it (sigma -> sigma * scale). The default 1.0 leaves the update bit-for-bit
  // unchanged; it is set per frame by Estimator::schurProcessFrame() from the
  // starvation gap. Appended last so existing aggregate initialisation is not
  // disturbed.
  double weak_update_scale_ = 1.0;
};

struct UpdateSummary_s {
  bool applied_ = false;    // false when the frame was visually starved or the
                            // information matrix was numerically unusable
  int rank_ = 0;            // rank of the Schur-reduced clone information
  double condition_ = 0.0;  // lambda_max / lambda_min over the retained modes
  int features_used_ = 0;
  int features_rejected_ = 0;
  int observations_used_ = 0;
  int degrees_of_freedom_ = 0;
  double mean_residual_px_ = 0.0;
  double max_residual_px_ = 0.0;
  // Human-readable reason when applied == false. Empty on success.
  std::string failure_reason_;
  // Track ids that successfully contributed to the accumulated A/b. A track
  // is listed here only after triangulation, chi2 gating, point-block and
  // Jacobian checks have all passed, even if the frame-level update is later
  // abandoned (for example by the condition or PSD guards). Rejected tracks
  // never appear here.
  std::vector<int> used_feature_ids_;
  // Tracks rejected during triangulation, gating or Schur reduction.
  std::vector<int> rejected_feature_ids_;
  // Every track id passed to update() on this call, including accepted,
  // rejected and max_features-capped tracks. This describes the input set;
  // it is not the list of tracks whose measurements were actually applied.
  std::vector<int> consumed_feature_ids_;
  // P6 weak update: true when this call used weak_update_scale > 1.0, i.e. the
  // gate was relaxed and the accepted observations were down-weighted.
  bool weak_update_ = false;
  double weak_update_scale_ = 1.0;
};

// One weighted observation row of the visual measurement model.
struct ObservationJacobian_s {
  Eigen::Matrix<double, 2, 6> H_pose_ = Eigen::Matrix<double, 2, 6>::Zero();
  Eigen::Matrix<double, 2, 3> H_point_ = Eigen::Matrix<double, 2, 3>::Zero();
  Eigen::Vector2d residual_ = Eigen::Vector2d::Zero();  // (z - h) / sigma
  Eigen::Vector3d p_c_ = Eigen::Vector3d::Zero();       // point in the camera frame
  bool valid_ = false;
};

// Per-feature bookkeeping of the Schur elimination (exposed for tests).
struct TrackReduction_s {
  bool valid_ = false;
  Eigen::Matrix3d A_point_ = Eigen::Matrix3d::Zero();  // H_f^T H_f
  Eigen::Vector3d b_point_ = Eigen::Vector3d::Zero();  // H_f^T r
  // clone_index -> H_xi^T H_f (6x3)
  std::unordered_map<int, Eigen::Matrix<double, 6, 3>> W_;
  // Numerical rank of H_f (= rank of A_point). The point-depth direction is
  // unobservable when this drops below 3; such tracks are rejected by this
  // updater.
  int point_rank_ = 0;
  // Dimension of the point-nullspace-projected residual used by the chi2
  // gate: gate_dof = 2 * observations - point_rank.
  int gate_dof_ = 0;
  double chi2_ = 0.0;
  double rms_residual_px_ = 0.0;
};

class SchurUpdater {
public:
  explicit SchurUpdater(const UpdateOptions_s & _options = UpdateOptions_s()) : options_(_options) {}
  const UpdateOptions_s & options() const { return options_; }

  // P6: refresh the per-frame weak update scale without rebuilding the updater.
  // The default (1.0) is a strict no-op, so existing callers are unchanged.
  void setWeakUpdateScale(double _scale) { options_.weak_update_scale_ = _scale; }

  // Triangulates and refines `track.world_point` from its observations using the
  // current clone poses. Returns false when the track cannot be triangulated.
  bool triangulate(
    const EkfState & _state, const std::vector<CameraExtrinsics_s> & _cameras, FeatureTrack_s & _track) const;

  // Weighted residual/Jacobians of a single observation. The residual uses
  // the current clone pose; the Jacobian uses the frozen clone FEJ pose.
  ObservationJacobian_s observationJacobian(
    const EkfState & _state, const CameraExtrinsics_s & _camera, const Eigen::Vector3d & _world_point,
    const FeatureObservation_s & _observation) const;

  // Builds the per-track Schur reduction (A_point, b_point, W blocks, chi2 gate
  // value). `_state` is not modified.
  TrackReduction_s reduceTrack(
    const EkfState & _state, const std::vector<CameraExtrinsics_s> & _cameras,
    const FeatureTrack_s & _track) const;

  // Full measurement update: triangulate + gate + Schur-eliminate + EKF update.
  // `tracks` are updated in place (world_point) and `state` is corrected.
  UpdateSummary_s update(
    EkfState & _state, std::vector<FeatureTrack_s> & _tracks, const std::vector<CameraExtrinsics_s> & _cameras);

  // 0.95 quantile of the chi-square distribution (Wilson-Hilferty approximation,
  // no Boost dependency). Used for the outlier gate.
  static double chi2Quantile95(int _dof);

private:
  // Shared implementation of reduce_track() and update(). Fills `_jacobians`
  // with this track's weighted observation rows and `_reduction` with the
  // point block, chi2 gate and per-clone W blocks. Returns false when the
  // track cannot be reduced. `_reduction` is reset first; `_jacobians` is left
  // populated for a successful reduction so update() can reuse it instead of
  // recomputing every Jacobian a second time.
  bool reduceTrackImpl(
    const EkfState & _state, const std::vector<CameraExtrinsics_s> & _cameras, const FeatureTrack_s & _track,
    std::vector<ObservationJacobian_s> & _jacobians, TrackReduction_s & _reduction) const;

  UpdateOptions_s options_;
};

}  // namespace backend
}  // namespace vins
