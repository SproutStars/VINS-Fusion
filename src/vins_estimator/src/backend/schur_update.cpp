/*******************************************************
 * Schur-complement visual update for the ARM-oriented EKF backend.
 *
 * Implementation of the frozen interface in schur_update.h.  Conventions
 * (right/body perturbation, q = [w,x,y,z], body -> world) follow ekf_state.h.
 *
 * ===========================================================================
 * 1. Measurement model and Jacobians
 * ===========================================================================
 * A clone i stores a body pose (R_i, p_i).  The camera is rigidly attached,
 * body_T_cam = (R_cb, p_cb):
 *
 *     u   = R_i^T (P_w - p_i)              point in the clone body frame
 *     p_c = R_cb^T (u - p_cb)              point in the camera frame
 *     h   = (p_c.x/p_c.z, p_c.y/p_c.z)     normalised (undistorted) projection
 *     r   = (uv - h) / sigma               weighted residual
 *     sigma = pixel_noise / focal_length   normalised 1-sigma pixel noise
 *
 * Right/body perturbation: q_true = q_hat (x) [1, dtheta/2], i.e.
 * R_true = R_hat (I + [dtheta]_x).  The residual is evaluated at the current
 * clone mean, while the Jacobian blocks use the frozen clone FEJ pose. Hence
 *
 *     du/d(dtheta_i) = [u]_x        (R_i^T -> (I - [dtheta]_x) R_i^T)
 *     du/d(dp_i)     = -R_i^T
 *     du/d(P_w)      =  R_i^T
 *
 * and with d(h)/d(p_c) = [[1/z, 0, -x/z^2], [0, 1/z, -y/z^2]]:
 *
 *     H_pose  = (1/sigma) d(h)/d(p_c) [ R_cb^T [u]_x | -R_cb^T R_i^T ]   (2x6)
 *     H_point = (1/sigma) d(h)/d(p_c) R_cb^T R_i^T                       (2x3)
 *
 * T1 of the unit test validates both blocks against central finite
 * differences (the residual derivative is -H, since r = (uv - h)/sigma).
 *
 * ===========================================================================
 * 2. Triangulation
 * ===========================================================================
 * The track is anchored on its FIRST observing clone.  The point is
 * parameterised in that clone's camera frame by (alpha, beta, rho):
 *
 *     p_C = (alpha/rho, beta/rho, 1/rho)     depth = 1/rho
 *
 * which is exactly open_vins' FeatureInitializer::single_gaussnewton
 * parameterisation (alpha, beta are the anchor pixel coordinates, rho the
 * inverse depth).  Initialisation is the closed-form two-view solution: the
 * depth lambda along the first bearing such that the point also lies on the
 * last bearing, i.e. solve  (m + lambda d) x b_last = 0  in least squares.
 * Refinement minimises the unweighted normalised reprojection error over all
 * observations with plain Gauss-Newton.
 *
 * ===========================================================================
 * 3. Schur reduction and EKF update
 * ===========================================================================
 * Feature points are nuisance parameters.  Per track the joint system
 * [H_x H_f]^T [H_x H_f] is reduced by eliminating the 3x3 point block
 *
 *     A_red = H_x^T H_x - W A_ff^-1 W^T ,   b_red = H_x^T r - W A_ff^-1 b_f
 *
 * with A_ff = sum H_f^T H_f and W_i = sum_{obs of clone i} H_xi^T H_f.  Note
 * that H_x^T H_x = sum_a H_a^T H_a is *block diagonal* (every observation row
 * only touches the 6 columns of its own clone), so it is accumulated per
 * observation; only the Schur elimination couples different clones.
 *
 * A is structurally singular: the live IMU block and unobserved clones carry
 * no visual information, and the visual problem has a pose/scale gauge.  The
 * accumulated (A, b) is therefore decomposed as
 *
 *     A = V diag(lambda) V^T,   lambda_i > info_rank_rtol * lambda_max,
 *
 * and a rank-revealing square root L = diag(sqrt(lambda)) V^T is used as the
 * measurement Jacobian:
 *
 *     z = diag(1/sqrt(lambda)) V^T b,
 *     S = L P L^T + I,   K = P L^T S^-1,   dx = K z,
 *     P <- (I - K L) P (I - K L)^T + K K^T   (Joseph form).
 *
 * This is equivalent to (P^-1 + L^T L) dx = b on the observed subspace.  The
 * discarded null directions receive no information.  In particular, this
 * avoids the old information-form ridge (A P A^T + A + 1e-12 I), which added
 * fake certainty to ~n/2 gauge directions and slowly corrupted long runs.
 * Significant negative eigenvalues are treated as an inconsistent Schur
 * reduction and the frame update is skipped rather than repaired implicitly.
 *
 * The per-track chi2 gate needs a basis of the left null space of the 2N x 3
 * point Jacobian Hf.  The default `nullspace_method_ == "svd"` path computes
 * it from JacobiSVD(Hf, ComputeFullU), which materialises the full 2N x 2N U.
 * The optional `"qr"` path applies OpenVINS-style Givens rotations to Hf, Hx
 * and r, dropping the eliminated rows, and never forms the full U.  The
 * projection subspace (hence chi2 and its dof) is identical; the point-block
 * rank test, point_block_rtol_ and the global eigen/Joseph update are shared
 * by both paths.  No ridge is ever used to remove the structural gauge.
 *
 * ===========================================================================
 * 4. P6 starvation-adaptive weak update ("accept more, weight less")
 * ===========================================================================
 * When UpdateOptions::weak_update_scale > 1.0 the chi2 gate *threshold* is
 * multiplied by that factor (the gate statistic keeps the nominal sigma) and the
 * accumulated information (A, b) is divided by scale^2, i.e. the observation
 * weight uses sigma * scale. This is exactly R * scale^2 on the observed
 * subspace and lets a starved filter accept a few tracks without trusting them
 * fully. The default 1.0 reproduces the previous behaviour bit-for-bit.
 *******************************************************/

#include "backend/schur_update.h"

#include "observation_budget.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <unordered_map>
#include <vector>

namespace vins
{
namespace backend
{
namespace
{

// 0.95 quantile of the standard normal distribution (Wilson-Hilferty).
constexpr double K_NORMAL_QUANTILE95 = 1.6448536269514722;
// Depth below which a camera-frame point counts as "behind the camera" [m].
constexpr double K_MIN_CAMERA_Z = 1e-6;
// A triangulated point whose RMS reprojection error exceeds this is dropped.
constexpr double K_MAX_TRIANGULATION_RMS_PX = 5.0;

// True when an observation carries an explicit image velocity for td
// compensation.  A non-zero vector is accepted even if the optional flag was
// not set by an older caller.
inline bool observation_has_velocity(const FeatureObservation_s & _observation)
{
  return _observation.has_velocity_ ||
         (_observation.velocity_.allFinite() && _observation.velocity_.squaredNorm() > 0.0);
}

// Current/linearisation camera calibration used by the measurement model.  A
// state with calibration disabled falls back to the caller-supplied constant
// extrinsics, preserving the historical API bit-for-bit.
struct CameraView_s {
  CameraExtrinsics_s current_;
  CameraExtrinsics_s fej_;
  int camera_id_ = -1;
  int calib_size_ = 0;
  int calib_offset_ = -1;
  int td_offset_ = -1;
  int td_column_ = -1;
};

CameraView_s resolve_camera_view(const EkfState & _state, int _camera_id, const CameraExtrinsics_s & _fallback)
{
  CameraView_s view;
  view.current_ = _fallback;
  view.fej_ = _fallback;
  view.camera_id_ = _camera_id;
  if (!_state.calibrationEnabled() || _camera_id < 0 || _camera_id >= _state.cameraCount())
    return view;

  const CameraCalibrationNominal_s & cur = _state.cameraCalib(_camera_id);
  const CameraCalibrationNominal_s & fej = _state.cameraCalibFej(_camera_id);
  view.current_.q_cb_ = cur.q_cb_;
  view.current_.p_cb_ = cur.p_cb_;
  view.current_.td_ = _state.cameraTimeOffset(_camera_id);
  view.fej_.q_cb_ = fej.q_cb_;
  view.fej_.p_cb_ = fej.p_cb_;
  view.fej_.td_ = fej.td_;
  view.calib_size_ = _state.cameraCalibSize();
  view.calib_offset_ = view.calib_size_ > 0 ? _state.cameraCalibOffset(_camera_id) : -1;
  view.td_offset_ = _state.timeOffsetOffset();
  if (view.td_offset_ >= 0)
    view.td_column_ = view.calib_size_ > 0 ? K_CAMERA_EXTRINSIC_STATE_SIZE : 0;
  return view;
}

CameraExtrinsics_s camera_from_state(const EkfState & _state, int _camera_id)
{
  CameraExtrinsics_s camera;
  if (_camera_id < 0 || _camera_id >= _state.cameraCount())
    return camera;
  const CameraCalibrationNominal_s & calib = _state.cameraCalib(_camera_id);
  camera.q_cb_ = calib.q_cb_;
  camera.p_cb_ = calib.p_cb_;
  camera.td_ = _state.cameraTimeOffset(_camera_id);
  return camera;
}

// When calibration is being estimated the state is authoritative.  When it is
// disabled the caller's constant vector is used unchanged.
std::vector<CameraExtrinsics_s>
resolved_cameras(const EkfState & _state, const std::vector<CameraExtrinsics_s> & _cameras)
{
  if (!_state.calibrationEnabled() || _state.cameraCount() <= 0)
    return _cameras;
  std::vector<CameraExtrinsics_s> cameras(static_cast<std::size_t>(_state.cameraCount()));
  for (int cam = 0; cam < _state.cameraCount(); ++cam)
    cameras[static_cast<std::size_t>(cam)] = camera_from_state(_state, cam);
  return cameras;
}

// Normalised projection h = (x/z, y/z).
inline Eigen::Vector2d project_normalized(const Eigen::Vector3d & _p_c)
{
  return Eigen::Vector2d(_p_c.x() / _p_c.z(), _p_c.y() / _p_c.z());
}

// d(h)/d(p_c) for the normalised projection (2x3).
inline Eigen::Matrix<double, 2, 3> projection_jacobian(const Eigen::Vector3d & _p_c)
{
  const double inv_z = 1.0 / _p_c.z();
  Eigen::Matrix<double, 2, 3> J;
  J(0, 0) = inv_z;
  J(0, 1) = 0.0;
  J(0, 2) = -_p_c.x() * inv_z * inv_z;
  J(1, 0) = 0.0;
  J(1, 1) = inv_z;
  J(1, 2) = -_p_c.y() * inv_z * inv_z;
  return J;
}

// ---------------------------------------------------------------------------
// Scratch storage.  The frozen header leaves no room for member buffers, so the
// reusable work buffers live here (thread-local, one set per thread).  They are
// reused across tracks so the hot path performs no per-observation allocation.
// ---------------------------------------------------------------------------
std::vector<ObservationJacobian_s> & jacobian_scratch()
{
  static thread_local std::vector<ObservationJacobian_s> scratch;
  return scratch;
}

// Reusable per-track work buffers for reduceTrackImpl(). Eigen keeps dynamic
// storage across a resize(source-shape) -> resize(same-shape) round trip, so
// repeating tracks of the same shape avoid the allocations; a shape change may
// still reallocate. Buffers are fully overwritten (Hx is zero-filled) in the
// same order as the previous per-track temporaries, so the arithmetic and its
// rounding are unchanged. The SVD and LDLT objects keep their internal buffers
// between same-shape tracks through compute().
struct TrackScratch_s {
  Eigen::MatrixXd Hx_;
  Eigen::MatrixXd Hf_;
  Eigen::VectorXd r_;
  Eigen::MatrixXd N_;
  Eigen::VectorXd r0_;
  Eigen::MatrixXd H0_;
  Eigen::MatrixXd S0_;
  Eigen::VectorXd y_;
  Eigen::JacobiSVD<Eigen::MatrixXd> hf_svd_;
  Eigen::LDLT<Eigen::MatrixXd> gate_ldlt_;
};

TrackScratch_s & trackScratch()
{
  static thread_local TrackScratch_s scratch;
  return scratch;
}

// Fills `out` with the weighted jacobian rows of one track (resized, not
// reallocated when the track length does not grow).  Returns false when an
// observation is degenerate: unknown clone or camera index, point behind the
// camera, or non-finite numbers.
bool fill_track_jacobians(
  const SchurUpdater & _updater, const EkfState & _state, const std::vector<CameraExtrinsics_s> & _cameras,
  const FeatureTrack_s & _track, std::vector<ObservationJacobian_s> & _out)
{
  const int num_obs = static_cast<int>(_track.observations_.size());
  _out.resize(static_cast<std::size_t>(num_obs));
  for (int j = 0; j < num_obs; ++j) {
    const FeatureObservation_s & obs = _track.observations_[static_cast<std::size_t>(j)];
    if (obs.clone_index_ < 0 || obs.clone_index_ >= _state.numClones())
      return false;
    if (obs.camera_id_ < 0 || obs.camera_id_ >= static_cast<int>(_cameras.size()))
      return false;
    _out[static_cast<std::size_t>(j)] = _updater.observationJacobian(
      _state, _cameras[static_cast<std::size_t>(obs.camera_id_)], _track.world_point_, obs);
    if (!_out[static_cast<std::size_t>(j)].valid_)
      return false;
  }
  return true;
}

// OpenVINS UpdaterHelper::nullspace_project_inplace style Givens QR.  Hf is
// m x 3, Hx is m x n and r is m.  The same orthogonal Q^T is applied to all
// three matrices, so Hf becomes upper triangular in its first 3 rows and the
// trailing rows of Hx/r are exactly N^T Hx and N^T r, where N spans the left
// null space of Hf.  Rotating and truncating avoids the full m x m U that
// JacobiSVD(Hf, ComputeFullU) materialises.  `_Hf` is modified in place.
void givens_nullspace_project_inplace(Eigen::MatrixXd & _Hf, Eigen::MatrixXd & _Hx, Eigen::VectorXd & _r)
{
  Eigen::JacobiRotation<double> gr;
  const int rows = static_cast<int>(_Hf.rows());
  const int cols = static_cast<int>(_Hf.cols());
  for (int col = 0; col < cols; ++col) {
    for (int row = rows - 1; row > col; --row) {
      gr.makeGivens(_Hf(row - 1, col), _Hf(row, col));
      _Hf.block(row - 1, col, 2, cols - col).applyOnTheLeft(0, 1, gr.adjoint());
      _Hx.block(row - 1, 0, 2, _Hx.cols()).applyOnTheLeft(0, 1, gr.adjoint());
      _r.block(row - 1, 0, 2, 1).applyOnTheLeft(0, 1, gr.adjoint());
    }
  }
}

// Rank of the 3x3 triangular factor produced by the Givens projection.  The
// threshold is identical to the SVD path (singular value > sqrt(rtol) *
// sigma_max, because A_point = Hf^T Hf), but only the small triangular block
// is decomposed.  Returns 0 for a non-finite/empty factor.
int triangular_column_rank(const Eigen::MatrixXd & _Hf, double _rtol)
{
  const int rank_dim = static_cast<int>(_Hf.cols());
  if (rank_dim <= 0 || _Hf.rows() < rank_dim)
    return 0;
  Eigen::MatrixXd R = _Hf.topRows(rank_dim);
  if (!R.allFinite())
    return 0;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(R, 0);  // singular values only
  const auto & singular = svd.singularValues();
  if (!singular.allFinite() || singular.size() == 0)
    return 0;
  const double singular_max = singular.maxCoeff();
  if (!(singular_max > 0.0))
    return 0;
  const double singular_rtol = std::sqrt(std::max(0.0, _rtol));
  int rank = 0;
  for (int i = 0; i < singular.size(); ++i)
    if (singular(i) > singular_rtol * singular_max)
      ++rank;
  return rank;
}

}  // namespace

double SchurUpdater::chi2Quantile95(int _dof)
{
  if (_dof <= 0)
    return 0.0;
  const double d = static_cast<double>(_dof);
  const double t = 1.0 - 2.0 / (9.0 * d) + K_NORMAL_QUANTILE95 * std::sqrt(2.0 / (9.0 * d));
  return d * t * t * t;
}

ObservationJacobian_s SchurUpdater::observationJacobian(
  const EkfState & _state, const CameraExtrinsics_s & _camera, const Eigen::Vector3d & _world_point,
  const FeatureObservation_s & _observation) const
{
  ObservationJacobian_s J;
  const double sigma = options_.pixel_noise_ / options_.focal_length_;
  if (!(sigma > 0.0) || !(options_.focal_length_ > 0.0))
    return J;
  if (_observation.clone_index_ < 0 || _observation.clone_index_ >= _state.numClones())
    return J;
  if (!_world_point.allFinite() || !_observation.uv_.allFinite() || !_observation.velocity_.allFinite())
    return J;

  const CameraView_s view = resolve_camera_view(_state, _observation.camera_id_, _camera);
  if (view.camera_id_ < 0)
    return J;
  const CameraExtrinsics_s & cur = view.current_;
  const CameraExtrinsics_s & lin = view.fej_;
  const CloneNominal_s & clone = _state.clone(_observation.clone_index_);
  const CloneNominal_s & clone_lin = _state.cloneFej(_observation.clone_index_);
  const Eigen::Matrix3d R_cb = EkfState::rotation(cur.q_cb_);

  // Residual at the current clone and calibration estimate.  The optional
  // VINS-style td term uses the shared td state and the observation's sampled
  // td/velocity; it is THE only td measurement term (no clone-td Jacobian is
  // added, so information is not double counted).
  const Eigen::Matrix3d R_i = EkfState::rotation(clone.q_);
  const Eigen::Vector3d u = R_i.transpose() * (_world_point - clone.p_);
  const Eigen::Vector3d p_c = R_cb.transpose() * (u - cur.p_cb_);
  J.p_c_ = p_c;
  if (!(p_c.z() > K_MIN_CAMERA_Z))
    return J;  // behind / grazing the camera

  Eigen::Vector2d h = project_normalized(p_c);
  if (view.td_offset_ >= 0 && observation_has_velocity(_observation))
    h += (_state.cameraTimeOffset(_observation.camera_id_) - _observation.cur_td_) * _observation.velocity_;
  J.residual_ = (_observation.uv_ - h) / sigma;

  // Jacobian at the frozen clone/calibration FEJ, matching OpenVINS
  // UpdaterHelper::get_feature_jacobian_3d().
  const Eigen::Matrix3d R_i_lin = EkfState::rotation(clone_lin.q_);
  const Eigen::Vector3d u_lin = R_i_lin.transpose() * (_world_point - clone_lin.p_);
  const Eigen::Matrix3d R_cb_lin = EkfState::rotation(lin.q_cb_);
  const Eigen::Vector3d p_c_lin = R_cb_lin.transpose() * (u_lin - lin.p_cb_);
  if (!(p_c_lin.z() > K_MIN_CAMERA_Z))
    return J;

  const Eigen::Matrix<double, 2, 3> dh_dpc = projection_jacobian(p_c_lin);

  // dp_c/d[dtheta_i | dp_i] and dp_c/d(P_w), all evaluated at the FEJ pose.
  Eigen::Matrix<double, 3, 6> dpc_dpose;
  dpc_dpose.block<3, 3>(0, 0) = R_cb_lin.transpose() * EkfState::skew(u_lin);
  dpc_dpose.block<3, 3>(0, 3) = -R_cb_lin.transpose() * R_i_lin.transpose();
  const Eigen::Matrix3d dpc_dpoint = R_cb_lin.transpose() * R_i_lin.transpose();

  J.H_pose_ = (dh_dpc * dpc_dpose) / sigma;
  J.H_point_ = (dh_dpc * dpc_dpoint) / sigma;
  J.camera_id_ = view.camera_id_;
  J.calib_size_ = view.calib_size_;
  J.calib_offset_ = view.calib_offset_;
  J.td_offset_ = view.td_offset_;
  J.td_column_ = view.td_column_;

  // body_T_cam right perturbation:
  //   R_cb_true = R_cb (I + [dtheta]x), p_cb_true = p_cb + dp
  //   p_c = R_cb^T (u - p_cb)
  //   dp_c/dtheta = [p_c]x, dp_c/dp_cb = -R_cb^T.
  if (J.calib_size_ == K_CAMERA_EXTRINSIC_STATE_SIZE) {
    Eigen::Matrix<double, 3, 6> dpc_dcalib;
    dpc_dcalib.block<3, 3>(0, 0) = EkfState::skew(p_c_lin);
    dpc_dcalib.block<3, 3>(0, 3) = -R_cb_lin.transpose();
    J.H_calib_.block<2, 6>(0, 0) = (dh_dpc * dpc_dcalib) / sigma;
  }

  // Explicit VINS-style td Jacobian.  All H fields in this updater are
  // dh/dx (residual is z-h, and the EKF uses dx = K r), so:
  //   h_effective = h + (td-cur_td)*velocity  =>  H_td = +velocity/sigma.
  // The same shared td column is used by every camera.
  if (J.td_column_ >= 0) {
    if (observation_has_velocity(_observation))
      J.H_calib_.col(J.td_column_) = _observation.velocity_ / sigma;
    else
      J.H_calib_.col(J.td_column_).setZero();
  }

  J.valid_ =
    J.H_pose_.allFinite() && J.H_point_.allFinite() && J.H_calib_.allFinite() && J.residual_.allFinite();
  return J;
}

ObservationJacobian_s SchurUpdater::observationJacobian(
  const EkfState & _state, const Eigen::Vector3d & _world_point, const FeatureObservation_s & _observation) const
{
  const CameraExtrinsics_s camera = camera_from_state(_state, _observation.camera_id_);
  return observationJacobian(_state, camera, _world_point, _observation);
}

Eigen::Vector2d
SchurUpdater::correctedObservation(const EkfState & _state, const FeatureObservation_s & _observation)
{
  Eigen::Vector2d corrected = _observation.uv_;
  if (
    _state.hasTimeOffsetState() && observation_has_velocity(_observation) && _observation.camera_id_ >= 0 &&
    _observation.camera_id_ < _state.cameraCount()) {
    corrected -=
      (_state.cameraTimeOffset(_observation.camera_id_) - _observation.cur_td_) * _observation.velocity_;
  }
  return corrected;
}

bool SchurUpdater::triangulate(
  const EkfState & _state, const std::vector<CameraExtrinsics_s> & _cameras, FeatureTrack_s & _track) const
{
  _track.has_point_ = false;
  const std::vector<CameraExtrinsics_s> cameras = resolved_cameras(_state, _cameras);
  const int num_obs = static_cast<int>(_track.observations_.size());
  if (num_obs < 2)
    return false;
  if (!(options_.min_depth_ > 0.0) || !(options_.max_depth_ > options_.min_depth_))
    return false;

  for (const FeatureObservation_s & obs : _track.observations_) {
    if (obs.clone_index_ < 0 || obs.clone_index_ >= _state.numClones())
      return false;
    if (obs.camera_id_ < 0 || obs.camera_id_ >= static_cast<int>(cameras.size()))
      return false;
    if (!obs.uv_.allFinite())
      return false;
  }

  // ---- two-view linear initialisation (anchor = earliest view) ----------
  // Pick the views explicitly by (clone, camera) rather than trusting the
  // caller's vector order. A same-timestamp stereo pair shares one clone but
  // uses two cameras; the extrinsic baseline makes the two-view solve well
  // posed, so only an identical (clone, camera) pair is rejected here.
  const FeatureObservation_s * first_view = &_track.observations_.front();
  const FeatureObservation_s * last_view = &_track.observations_.front();
  for (const FeatureObservation_s & obs : _track.observations_) {
    if (
      obs.clone_index_ < first_view->clone_index_ ||
      (obs.clone_index_ == first_view->clone_index_ && obs.camera_id_ < first_view->camera_id_))
      first_view = &obs;
    if (
      obs.clone_index_ > last_view->clone_index_ ||
      (obs.clone_index_ == last_view->clone_index_ && obs.camera_id_ > last_view->camera_id_))
      last_view = &obs;
  }
  const FeatureObservation_s & first = *first_view;
  const FeatureObservation_s & last = *last_view;
  if (first.clone_index_ == last.clone_index_ && first.camera_id_ == last.camera_id_)
    return false;  // identical view: no baseline

  const CloneNominal_s & c0 = _state.clone(first.clone_index_);
  const CloneNominal_s & c1 = _state.clone(last.clone_index_);
  const CameraExtrinsics_s & cam0 = cameras[static_cast<std::size_t>(first.camera_id_)];
  const CameraExtrinsics_s & cam1 = cameras[static_cast<std::size_t>(last.camera_id_)];
  const Eigen::Matrix3d R0 = EkfState::rotation(c0.q_);
  const Eigen::Matrix3d R1 = EkfState::rotation(c1.q_);
  const Eigen::Matrix3d R_cb0 = EkfState::rotation(cam0.q_cb_);
  const Eigen::Matrix3d R_cb1 = EkfState::rotation(cam1.q_cb_);

  const Eigen::Vector2d first_uv = correctedObservation(_state, first);
  const Eigen::Vector2d last_uv = correctedObservation(_state, last);
  const Eigen::Vector3d b0(first_uv.x(), first_uv.y(), 1.0);
  // Point in the anchor camera frame at depth lambda:  p_C = lambda * b0.
  // Seen from the last clone: p_c1 = m + lambda * d, which must be parallel
  // to b1, i.e. [b1]_x (m + lambda d) = 0 solved in least squares.
  const Eigen::Vector3d t0 = R1.transpose() * ((c0.p_ - c1.p_) + R0 * cam0.p_cb_);
  const Eigen::Vector3d m = R_cb1.transpose() * (t0 - cam1.p_cb_);
  const Eigen::Vector3d d = R_cb1.transpose() * R1.transpose() * R0 * R_cb0 * b0;
  const Eigen::Vector3d b1(last_uv.x(), last_uv.y(), 1.0);
  const Eigen::Matrix3d Sx = EkfState::skew(b1);
  const Eigen::Vector3d lhs = Sx * d;
  const double denom = lhs.squaredNorm();
  if (!(denom > 1e-18))
    return false;  // (nearly) parallel rays
  const double lambda = -lhs.dot(Sx * m) / denom;
  if (!std::isfinite(lambda) || !(lambda > 0.0))
    return false;  // behind the anchor

  // ---- Gauss-Newton on the anchored inverse-depth parameterisation -------
  double alpha = first_uv.x();
  double beta = first_uv.y();
  double rho = 1.0 / lambda;
  if (!std::isfinite(rho) || !(rho > 0.0))
    return false;

  for (int iter = 0; iter < options_.triangulation_iterations_; ++iter) {
    const Eigen::Vector3d p_C(alpha / rho, beta / rho, 1.0 / rho);
    const Eigen::Vector3d P_w = c0.p_ + R0 * (R_cb0 * p_C + cam0.p_cb_);

    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
    for (const FeatureObservation_s & obs : _track.observations_) {
      const CloneNominal_s & ci = _state.clone(obs.clone_index_);
      const CameraExtrinsics_s & cam = cameras[static_cast<std::size_t>(obs.camera_id_)];
      const Eigen::Matrix3d Ri = EkfState::rotation(ci.q_);
      const Eigen::Matrix3d Rc = EkfState::rotation(cam.q_cb_);
      const Eigen::Vector3d u = Ri.transpose() * (P_w - ci.p_);
      const Eigen::Vector3d p_c = Rc.transpose() * (u - cam.p_cb_);
      if (!(p_c.z() > K_MIN_CAMERA_Z))
        return false;

      // dp_c/d(alpha, beta, rho): chain rule from the anchor camera frame
      // through the world frame into this camera frame.
      const Eigen::Matrix3d chain = Rc.transpose() * Ri.transpose() * R0 * R_cb0;
      Eigen::Matrix3d dpc_dp;
      dpc_dp.col(0) = chain * Eigen::Vector3d(1.0 / rho, 0.0, 0.0);
      dpc_dp.col(1) = chain * Eigen::Vector3d(0.0, 1.0 / rho, 0.0);
      dpc_dp.col(2) = -(chain * p_C) / rho;

      const Eigen::Vector2d res = correctedObservation(_state, obs) - project_normalized(p_c);
      // J = d(residual)/d(alpha, beta, rho).  Normal equations are
      // H = sum J^T J, g = sum J^T res and the Gauss-Newton step is
      // x <- x - H^-1 g (a *descent* step, hence the subtraction below).
      const Eigen::Matrix<double, 2, 3> J = -projection_jacobian(p_c) * dpc_dp;
      H.noalias() += J.transpose() * J;
      g.noalias() += J.transpose() * res;
    }
    if (!H.allFinite() || !g.allFinite())
      return false;

    Eigen::LDLT<Eigen::Matrix3d> ldlt(H);
    if (ldlt.info() != Eigen::Success)
      return false;
    const Eigen::Vector3d pivots = ldlt.vectorD();
    if (!(pivots.minCoeff() > 1e-12 * std::max(1.0, pivots.maxCoeff())))
      return false;
    const Eigen::Vector3d step = ldlt.solve(g);
    if (!step.allFinite())
      return false;
    alpha -= step(0);
    beta -= step(1);
    rho -= step(2);
    if (!std::isfinite(alpha) || !std::isfinite(beta) || !std::isfinite(rho))
      return false;
  }

  // ---- depth clamp, RMS check and output --------------------------------
  if (!(rho > 0.0) || !std::isfinite(rho))
    return false;
  const double depth = 1.0 / rho;
  if (!std::isfinite(depth))
    return false;
  if (depth < options_.min_depth_ || depth > options_.max_depth_)
    return false;  // clamped

  const Eigen::Vector3d p_C(alpha / rho, beta / rho, 1.0 / rho);
  const Eigen::Vector3d P_w = c0.p_ + R0 * (R_cb0 * p_C + cam0.p_cb_);
  if (!P_w.allFinite())
    return false;

  double sq_res = 0.0;
  for (const FeatureObservation_s & obs : _track.observations_) {
    const CloneNominal_s & ci = _state.clone(obs.clone_index_);
    const CameraExtrinsics_s & cam = cameras[static_cast<std::size_t>(obs.camera_id_)];
    const Eigen::Matrix3d Ri = EkfState::rotation(ci.q_);
    const Eigen::Matrix3d Rc = EkfState::rotation(cam.q_cb_);
    const Eigen::Vector3d u = Ri.transpose() * (P_w - ci.p_);
    const Eigen::Vector3d p_c = Rc.transpose() * (u - cam.p_cb_);
    if (!(p_c.z() > K_MIN_CAMERA_Z))
      return false;
    const Eigen::Vector2d res = correctedObservation(_state, obs) - project_normalized(p_c);
    if (!res.allFinite())
      return false;
    sq_res += res.squaredNorm();
  }
  const double rms_residual_px = std::sqrt(sq_res / static_cast<double>(num_obs)) * options_.focal_length_;
  if (!(rms_residual_px <= K_MAX_TRIANGULATION_RMS_PX))
    return false;

  // Triangulated from the current clone poses on every use (OpenVINS
  // UpdaterMSCKF behaviour). The point is intentionally *not* cached across
  // frames: a feature is consumed by at most one update, and freezing the
  // point (VINS_SCHUR_FREEZE A/B) was measured to be slightly worse on full
  // sequences (V2_01 0.2444 vs 0.2295 m, V2_02 0.3895 vs 0.3809 m), so the
  // freeze branch was removed.
  _track.world_point_ = P_w;
  _track.has_point_ = true;
  return true;
}

bool SchurUpdater::triangulate(const EkfState & _state, FeatureTrack_s & _track) const
{
  std::vector<CameraExtrinsics_s> cameras(static_cast<std::size_t>(std::max(0, _state.cameraCount())));
  for (int cam = 0; cam < _state.cameraCount(); ++cam)
    cameras[static_cast<std::size_t>(cam)] = camera_from_state(_state, cam);
  return triangulate(_state, cameras, _track);
}

bool SchurUpdater::reduceTrackImpl(
  const EkfState & _state, const std::vector<CameraExtrinsics_s> & _cameras, const FeatureTrack_s & _track,
  std::vector<ObservationJacobian_s> & _jacobians, TrackReduction_s & _reduction) const
{
  _reduction = TrackReduction_s();
  const int num_obs = static_cast<int>(_track.observations_.size());
  if (num_obs == 0)
    return false;
  const Eigen::MatrixXd & P = _state.covariance();
  if (P.rows() != _state.size() || P.cols() != _state.size())
    return false;

  const std::vector<CameraExtrinsics_s> cameras = resolved_cameras(_state, _cameras);
  if (!fill_track_jacobians(*this, _state, cameras, _track, _jacobians))
    return false;

  // ---- point block and per-clone W blocks -------------------------------
  double sq_res = 0.0;
  for (int j = 0; j < num_obs; ++j) {
    const ObservationJacobian_s & J = _jacobians[static_cast<std::size_t>(j)];
    _reduction.A_point_.noalias() += J.H_point_.transpose() * J.H_point_;
    _reduction.b_point_.noalias() += J.H_point_.transpose() * J.residual_;
    sq_res += J.residual_.squaredNorm();

    const int ci = _track.observations_[static_cast<std::size_t>(j)].clone_index_;
    auto it = _reduction.W_.find(ci);
    if (it == _reduction.W_.end()) {
      it = _reduction.W_.emplace(ci, Eigen::Matrix<double, 6, 3>::Zero()).first;
    }
    it->second.noalias() += J.H_pose_.transpose() * J.H_point_;

    if (J.calib_size_ == K_CAMERA_EXTRINSIC_STATE_SIZE) {
      auto ical = _reduction.W_calib_.find(J.camera_id_);
      if (ical == _reduction.W_calib_.end())
        ical = _reduction.W_calib_.emplace(J.camera_id_, Eigen::Matrix<double, 6, 3>::Zero()).first;
      ical->second.noalias() += J.H_calib_.block<2, 6>(0, 0).transpose() * J.H_point_;
    }
    if (J.td_column_ >= 0)
      _reduction.W_td_.noalias() += J.H_calib_.col(J.td_column_).transpose() * J.H_point_;
  }

  // Weighted -> pixel residual: ||r|| * sigma * focal_length.
  const double sigma = options_.pixel_noise_ / options_.focal_length_;
  _reduction.rms_residual_px_ =
    std::sqrt(sq_res / static_cast<double>(num_obs)) * sigma * options_.focal_length_;

  // Reject near-degenerate triangulations before they are used in the Schur
  // complement.  The relative eigenvalue test is scale free and much more
  // informative than raw LDLT pivots.
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> point_eig(_reduction.A_point_);
  if (point_eig.info() != Eigen::Success)
    return false;
  const Eigen::Vector3d & point_lambda = point_eig.eigenvalues();
  const double point_max = point_lambda.maxCoeff();
  const double point_min = point_lambda.minCoeff();
  if (
    !_reduction.A_point_.allFinite() || !std::isfinite(point_max) || !(point_max > 0.0) ||
    !(point_min > options_.point_block_rtol_ * point_max))
    return false;

  // Numerical rank of the point block from the same relative tolerance used
  // by the conditioning check above.  Rank 3 means the window has enough
  // parallax to estimate the point depth; rank < 3 is not observable here
  // and is rejected by this updater.
  _reduction.point_rank_ = 0;
  for (int i = 0; i < 3; ++i)
    if (point_lambda(i) > options_.point_block_rtol_ * point_max)
      ++_reduction.point_rank_;
  if (_reduction.point_rank_ < 3)
    return false;

  // ---- chi2 gate: explicit point-nullspace projection -------------------
  // A feature is a free nuisance parameter, so a residual component that can
  // be absorbed by moving the point carries no information about the state
  // and must not enter the gate.  Let Hf be the 2N x 3 point Jacobian and N
  // an orthonormal basis of its left null space (N^T Hf = 0).  Then
  //
  //     r0 = N^T r,   H0 = N^T Hx,
  //     S0 = H0 P H0^T + I,   chi2 = r0^T S0^-1 r0,
  //     dof = 2N - rank(Hf).
  //
  // S0 is positive definite by construction, so the numerical singularity of
  // the old profiled covariance C - Hf (Hf^T Hf)^-1 Hf^T cannot turn a valid
  // track into chi2 = inf.  The old expression also left the state covariance
  // in the point-depth direction unprojected; projecting both r and Hx is the
  // statistically correct marginalisation.
  const int m = 2 * num_obs;
  const int n = _state.size();
  TrackScratch_s & scratch = trackScratch();
  Eigen::MatrixXd & Hx = scratch.Hx_;
  Eigen::MatrixXd & Hf = scratch.Hf_;
  Eigen::VectorXd & r = scratch.r_;
  Hx.setZero(m, n);
  Hf.setZero(m, 3);
  r.resize(m);
  for (int a = 0; a < num_obs; ++a) {
    const ObservationJacobian_s & Ja = _jacobians[static_cast<std::size_t>(a)];
    const int oa = _state.cloneOffset(_track.observations_[static_cast<std::size_t>(a)].clone_index_);
    Hx.block<2, 6>(2 * a, oa) = Ja.H_pose_;
    if (Ja.calib_size_ == K_CAMERA_EXTRINSIC_STATE_SIZE)
      Hx.block<2, 6>(2 * a, Ja.calib_offset_) = Ja.H_calib_.block<2, 6>(0, 0);
    if (Ja.td_column_ >= 0)
      Hx.block<2, 1>(2 * a, Ja.td_offset_) = Ja.H_calib_.col(Ja.td_column_);
    Hf.block<2, 3>(2 * a, 0) = Ja.H_point_;
    r.segment<2>(2 * a) = Ja.residual_;
  }
  if (!Hx.allFinite() || !Hf.allFinite() || !r.allFinite())
    return false;

  // Point-nullspace projection for the chi2 gate.  "svd" is the historical
  // path and stays bit-for-bit compatible; "qr" applies the same orthogonal
  // projection with Givens rotations and never forms the full m x m U.
  int hf_rank = 0;
  Eigen::VectorXd & r0 = scratch.r0_;
  Eigen::MatrixXd & H0 = scratch.H0_;
  if (options_.nullspace_method_ == "qr") {
    givens_nullspace_project_inplace(Hf, Hx, r);
    if (!Hf.allFinite() || !Hx.allFinite() || !r.allFinite())
      return false;
    // Rank is measured on the 3x3 triangular factor with the same relative
    // singular-value threshold as the point-block test.  This keeps the gate
    // dof and the near-degenerate rejection identical to the SVD oracle.
    hf_rank = triangular_column_rank(Hf, options_.point_block_rtol_);
    if (hf_rank != _reduction.point_rank_)
      return false;
    const int null_dim = m - hf_rank;
    if (null_dim <= 0)
      return false;
    // The trailing rows are already N^T Hx and N^T r from the shared Q^T.
    H0 = Hx.bottomRows(null_dim);
    r0 = r.tail(null_dim);
  } else {
    Eigen::JacobiSVD<Eigen::MatrixXd> & hf_svd = scratch.hf_svd_;
    hf_svd.compute(Hf, Eigen::ComputeFullU);
    const auto & singular = hf_svd.singularValues();
    if (!singular.allFinite() || singular.size() == 0)
      return false;
    const double singular_max = singular.maxCoeff();
    if (!(singular_max > 0.0))
      return false;
    // A_point = Hf^T Hf, so an eigenvalue threshold of rtol*lambda_max
    // corresponds to a singular-value threshold of sqrt(rtol)*sigma_max.
    const double singular_rtol = std::sqrt(std::max(0.0, options_.point_block_rtol_));
    for (int i = 0; i < singular.size(); ++i)
      if (singular(i) > singular_rtol * singular_max)
        ++hf_rank;
    // Keep the gate consistent with the point-block rank test.  A rank < 3
    // track is a low-parallax case and is rejected by this updater.
    if (hf_rank != _reduction.point_rank_)
      return false;

    const Eigen::MatrixXd & U = hf_svd.matrixU();
    Eigen::MatrixXd & N = scratch.N_;
    N = U.rightCols(m - hf_rank);
    r0 = N.transpose() * r;
    H0 = N.transpose() * Hx;
  }
  Eigen::MatrixXd & S0 = scratch.S0_;
  S0 = H0 * P * H0.transpose();
  S0.diagonal().array() += 1.0;
  _reduction.gate_dof_ = static_cast<int>(r0.size());

  Eigen::LDLT<Eigen::MatrixXd> & ldlt = scratch.gate_ldlt_;
  ldlt.compute(S0);
  const auto s_pivots = ldlt.vectorD();
  if (ldlt.info() != Eigen::Success || !s_pivots.allFinite() || !(s_pivots.minCoeff() > 0.0)) {
    _reduction.chi2_ = std::numeric_limits<double>::infinity();
  } else {
    Eigen::VectorXd & y = scratch.y_;
    y = ldlt.solve(r0);
    _reduction.chi2_ = y.allFinite() ? r0.dot(y) : std::numeric_limits<double>::infinity();
  }

  _reduction.valid_ = std::isfinite(_reduction.chi2_) && std::isfinite(_reduction.rms_residual_px_) &&
                      _reduction.b_point_.allFinite();
  return _reduction.valid_;
}

TrackReduction_s SchurUpdater::reduceTrack(
  const EkfState & _state, const std::vector<CameraExtrinsics_s> & _cameras, const FeatureTrack_s & _track) const
{
  TrackReduction_s red;
  std::vector<ObservationJacobian_s> & jac = jacobian_scratch();
  reduceTrackImpl(_state, _cameras, _track, jac, red);
  return red;
}

TrackReduction_s SchurUpdater::reduceTrack(const EkfState & _state, const FeatureTrack_s & _track) const
{
  std::vector<CameraExtrinsics_s> cameras(static_cast<std::size_t>(std::max(0, _state.cameraCount())));
  for (int cam = 0; cam < _state.cameraCount(); ++cam)
    cameras[static_cast<std::size_t>(cam)] = camera_from_state(_state, cam);
  return reduceTrack(_state, cameras, _track);
}

UpdateSummary_s SchurUpdater::update(
  EkfState & _state, std::vector<FeatureTrack_s> & _tracks, const std::vector<CameraExtrinsics_s> & _cameras)
{
  UpdateSummary_s summary;
  const std::vector<CameraExtrinsics_s> cameras = resolved_cameras(_state, _cameras);
  // P6 starvation-adaptive weak update. A value > 1.0 relaxes the chi2 gate
  // threshold and down-weights the accepted observations (sigma -> sigma *
  // scale, i.e. R * scale^2). Defensive sanitisation: non-finite or <= 1.0
  // leaves the update bit-for-bit unchanged.
  const double weak_scale = (std::isfinite(options_.weak_update_scale_) && options_.weak_update_scale_ > 1.0)
                              ? options_.weak_update_scale_
                              : 1.0;
  if (weak_scale > 1.0) {
    summary.weak_update_ = true;
    summary.weak_update_scale_ = weak_scale;
  }
  // Record every input track before any early return. This describes the
  // input set of this call; only tracks that reach the A/b accumulation are
  // eventually reported through used_feature_ids.
  summary.consumed_feature_ids_.reserve(_tracks.size());
  summary.used_feature_ids_.reserve(_tracks.size());
  summary.rejected_feature_ids_.reserve(_tracks.size());
  for (const FeatureTrack_s & track : _tracks)
    summary.consumed_feature_ids_.push_back(track.id_);

  if (!_state.initialized() || _state.size() <= 0)
    return summary;
  const int n = _state.size();
  Eigen::MatrixXd & P = _state.covariance();
  if (P.rows() != n || P.cols() != n)
    return summary;

  // ---- 1. candidate selection, optional observation budget, triangulation -
  const int max_features = std::max(0, options_.max_features_);
  const int max_budget_observations = std::max(0, options_.max_update_observations_);
  const bool budgeted = max_budget_observations > 0;
  std::vector<int> accepted;
  accepted.reserve(_tracks.size());
  if (budgeted) {
    // Select complete tracks BEFORE triangulation so deferred/expired tracks
    // are not re-triangulated on every frame. The policy (atomic tracks,
    // stable longest-first order, margin expiry) lives in observation_budget.h.
    std::vector<int> candidates;
    candidates.reserve(_tracks.size());
    for (std::size_t i = 0; i < _tracks.size(); ++i) {
      if (static_cast<int>(_tracks[i].observations_.size()) < options_.min_observations_) {
        summary.rejected_feature_ids_.push_back(_tracks[i].id_);  // too short
      } else {
        candidates.push_back(static_cast<int>(i));
      }
    }
    std::stable_sort(
      candidates.begin(), candidates.end(),
      [&_tracks](int _lhs, int _rhs)
      {
        return _tracks[static_cast<std::size_t>(_lhs)].observations_.size() >
               _tracks[static_cast<std::size_t>(_rhs)].observations_.size();
      });
    // max_features_ remains a separate track cap applied before the budget.
    if (static_cast<int>(candidates.size()) > max_features)
      candidates.resize(static_cast<std::size_t>(max_features));

    // The helper takes FeatureTrack_s by value/const-ref; candidate tracks are
    // small (one observation per clone window) and this keeps `_tracks` order
    // intact for the caller.
    std::vector<FeatureTrack_s> candidate_tracks;
    candidate_tracks.reserve(candidates.size());
    for (const int index : candidates)
      candidate_tracks.push_back(_tracks[static_cast<std::size_t>(index)]);

    ObservationBudgetOptions_s budget;
    budget.max_observations_ = max_budget_observations;
    budget.max_tracks_ = 0;  // max_features_ was applied above
    budget.skip_oversized_ = true;
    // Same window-full test as Estimator::schurProcessFrame(): the state has
    // already been augmented for this frame, so numClones() > window_size_
    // means the oldest clone (clone index 0) will be marginalised next.
    budget.margin_active_ = _state.numClones() > _state.options().window_size_;
    budget.margin_clone_index_ = 0;
    const ObservationBudgetPlan_s plan = planObservationBudget(candidate_tracks, budget);

    summary.budget_active_ = true;
    summary.budget_limited_ = plan.limited_;
    summary.budget_input_observations_ = plan.input_observation_count_;
    summary.budget_selected_observations_ = plan.selected_observation_count_;
    summary.budget_deferred_observations_ = plan.deferred_observation_count_;
    summary.budget_expired_observations_ = plan.expired_observation_count_;
    summary.budget_selected_tracks_ = plan.selected_track_count_;
    summary.budget_deferred_tracks_ = plan.deferred_track_count_;
    summary.budget_expired_tracks_ = plan.expired_track_count_;
    summary.budget_deferred_feature_ids_ = plan.deferred_feature_ids_;
    summary.budget_expired_feature_ids_ = plan.expired_feature_ids_;
    // Positive budget: only admitted tracks reach triangulation, so only they
    // count as consumed. Deferred/expired ids are never used or rejected.
    summary.consumed_feature_ids_ = plan.selected_feature_ids_;

    accepted.reserve(plan.selected_indices_.size());
    for (const std::size_t selected : plan.selected_indices_)
      accepted.push_back(candidates[selected]);
  } else {
    // Legacy zero-budget path: triangulate every eligible track, then keep the
    // longest ones. consumed_feature_ids_ stays the full input list.
    for (std::size_t i = 0; i < _tracks.size(); ++i) {
      if (static_cast<int>(_tracks[i].observations_.size()) < options_.min_observations_) {
        summary.rejected_feature_ids_.push_back(_tracks[i].id_);  // too short to triangulate
        continue;
      }
      // Re-triangulate from the current clone poses every time the track is
      // selected, matching OpenVINS (UpdaterHelper::triangulate_feature is
      // called once per feature per update, and a feature is consumed by at
      // most one update). Freezing the first triangulation instead
      // (VINS_SCHUR_FREEZE=1, removed after the A/B) measured 0.1055 m vs
      // 0.1116 m on V2_01_easy 20 s and 0.2444 m vs 0.2295 m / 0.3895 m vs
      // 0.3809 m on the full V2_01_easy / V2_02_medium sequences, i.e. it was
      // never better on a full sequence, so the reference behaviour stays.
      if (!triangulate(_state, cameras, _tracks[i])) {
        summary.rejected_feature_ids_.push_back(_tracks[i].id_);  // no usable point
        continue;
      }
      accepted.push_back(static_cast<int>(i));
    }
    std::stable_sort(
      accepted.begin(), accepted.end(),
      [&_tracks](int _lhs, int _rhs)
      {
        return _tracks[static_cast<std::size_t>(_lhs)].observations_.size() >
               _tracks[static_cast<std::size_t>(_rhs)].observations_.size();
      });
    if (static_cast<int>(accepted.size()) > max_features)
      accepted.resize(static_cast<std::size_t>(max_features));
  }

  if (budgeted) {
    // Only budget-admitted tracks are triangulated. A failure here is a normal
    // rejection and does not return the reserved rows to the budget.
    std::vector<int> triangulated;
    triangulated.reserve(accepted.size());
    for (const int index : accepted) {
      if (!triangulate(_state, cameras, _tracks[static_cast<std::size_t>(index)])) {
        summary.rejected_feature_ids_.push_back(_tracks[static_cast<std::size_t>(index)].id_);
        continue;
      }
      triangulated.push_back(index);
    }
    accepted.swap(triangulated);
  }

  // ---- 2. per-track gate + Schur-reduced information accumulation -------
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
  double residual_sum_px = 0.0;
  double residual_max_px = 0.0;
  int residual_count = 0;
  const double sigma = options_.pixel_noise_ / options_.focal_length_;

  for (const int index : accepted) {
    FeatureTrack_s & track = _tracks[static_cast<std::size_t>(index)];
    TrackReduction_s red;
    std::vector<ObservationJacobian_s> & jac = jacobian_scratch();
    if (!reduceTrackImpl(_state, cameras, track, jac, red)) {
      summary.rejected_feature_ids_.push_back(track.id_);
      continue;
    }

    const int num_obs = static_cast<int>(track.observations_.size());
    // The point is marginalized in the gate, so the testable residual has
    // 2*num_obs - rank(Hf) degrees of freedom, not 2*num_obs.
    const int gate_dof = red.gate_dof_ > 0 ? red.gate_dof_ : 2 * num_obs;
    // P6: only the *threshold* is relaxed by weak_scale. red.chi2 is built
    // from the nominal sigma (pixel_noise / focal_length), so the gate is
    // never widened by the inflated measurement noise.
    if (red.chi2_ > options_.chi2_multiplier_ * weak_scale * chi2Quantile95(gate_dof)) {
      summary.rejected_feature_ids_.push_back(track.id_);  // chi2 outlier gate
      continue;
    }

    // reduceTrackImpl() already filled and validated `jac` for this exact
    // track, so the accumulation below reuses it instead of recomputing
    // every Jacobian a second time.

    // H_x^T H_x is block diagonal (one 6x6 block per observing clone), so
    // it is accumulated one observation at a time.  H_x^T r likewise.
    for (int j = 0; j < num_obs; ++j) {
      const ObservationJacobian_s & J = jac[static_cast<std::size_t>(j)];
      const int off = _state.cloneOffset(track.observations_[static_cast<std::size_t>(j)].clone_index_);
      A.block<6, 6>(off, off) += J.H_pose_.transpose() * J.H_pose_;
      b.segment<6>(off) += J.H_pose_.transpose() * J.residual_;

      if (J.calib_size_ == K_CAMERA_EXTRINSIC_STATE_SIZE) {
        const int offc = J.calib_offset_;
        const Eigen::Matrix<double, 2, 6> Hc = J.H_calib_.block<2, 6>(0, 0);
        A.block<6, 6>(offc, offc) += Hc.transpose() * Hc;
        b.segment<6>(offc) += Hc.transpose() * J.residual_;
        A.block<6, 6>(off, offc) += J.H_pose_.transpose() * Hc;
        A.block<6, 6>(offc, off) += Hc.transpose() * J.H_pose_;
      }
      if (J.td_column_ >= 0) {
        const int offt = J.td_offset_;
        const Eigen::Vector2d Htd = J.H_calib_.col(J.td_column_);
        A(offt, offt) += Htd.squaredNorm();
        b(offt) += Htd.dot(J.residual_);
        A.block<6, 1>(off, offt) += J.H_pose_.transpose() * Htd;
        A.block<1, 6>(offt, off) += Htd.transpose() * J.H_pose_;
        if (J.calib_size_ == K_CAMERA_EXTRINSIC_STATE_SIZE) {
          const int offc = J.calib_offset_;
          const Eigen::Matrix<double, 2, 6> Hc = J.H_calib_.block<2, 6>(0, 0);
          A.block<6, 1>(offc, offt) += Hc.transpose() * Htd;
          A.block<1, 6>(offt, offc) += Htd.transpose() * Hc;
        }
      }

      const double px = J.residual_.norm() * sigma * options_.focal_length_;
      residual_sum_px += px;
      residual_max_px = std::max(residual_max_px, px);
      ++residual_count;
    }

    // Schur elimination of this track's 3D point: the only step that
    // couples different clones.  Solve the 3x3 block once, without forming
    // an explicit inverse, and reuse the solutions for every W_i block.
    Eigen::LDLT<Eigen::Matrix3d> point_ldlt(red.A_point_);
    const Eigen::Vector3d point_rhs = point_ldlt.solve(red.b_point_);
    struct SchurBlock_s {
      int offset_ = 0;
      int dim_ = 0;
      Eigen::MatrixXd W_;   // dim x 3
      Eigen::MatrixXd Xi_;  // 3 x dim
    };
    std::vector<SchurBlock_s> schur_blocks;
    schur_blocks.reserve(red.W_.size() + red.W_calib_.size() + 1);
    bool schur_ok = point_ldlt.info() == Eigen::Success && point_rhs.allFinite();
    for (const auto & wi : red.W_) {
      SchurBlock_s block;
      block.offset_ = _state.cloneOffset(wi.first);
      block.dim_ = 6;
      block.W_ = wi.second;
      block.Xi_ = point_ldlt.solve(wi.second.transpose());
      schur_ok = schur_ok && block.Xi_.allFinite();
      schur_blocks.push_back(std::move(block));
    }
    for (const auto & wi : red.W_calib_) {
      SchurBlock_s block;
      block.offset_ = _state.cameraCalibOffset(wi.first);
      block.dim_ = 6;
      block.W_ = wi.second;
      block.Xi_ = point_ldlt.solve(wi.second.transpose());
      schur_ok = schur_ok && block.Xi_.allFinite();
      schur_blocks.push_back(std::move(block));
    }
    if (_state.hasTimeOffsetState()) {
      SchurBlock_s block;
      block.offset_ = _state.timeOffsetOffset();
      block.dim_ = 1;
      block.W_ = red.W_td_;
      block.Xi_ = point_ldlt.solve(red.W_td_.transpose());
      schur_ok = schur_ok && block.Xi_.allFinite();
      schur_blocks.push_back(std::move(block));
    }
    if (!schur_ok) {
      summary.rejected_feature_ids_.push_back(track.id_);
      continue;
    }

    for (const SchurBlock_s & wi : schur_blocks) {
      b.segment(wi.offset_, wi.dim_).noalias() -= wi.W_ * point_rhs;
      for (const SchurBlock_s & xj : schur_blocks) {
        A.block(wi.offset_, xj.offset_, wi.dim_, xj.dim_).noalias() -= wi.W_ * xj.Xi_;
      }
    }

    // This track passed triangulation, chi2, point-block and Jacobian
    // checks and its terms have just been accumulated into A/b.
    summary.used_feature_ids_.push_back(track.id_);
    summary.features_used_ += 1;
    summary.observations_used_ += num_obs;
    summary.degrees_of_freedom_ += 2 * num_obs;
  }
  summary.features_rejected_ = static_cast<int>(summary.rejected_feature_ids_.size());

  if (summary.features_used_ == 0) {
    summary.failure_reason_ = "no gated tracks";
    return summary;
  }

  // P6 down-weighting: the accumulated information is divided by scale^2,
  // exactly equivalent to weighting each accepted observation with
  // sigma * weak_scale (R * scale^2). It is applied here, after the nominal-
  // sigma chi2 gate, so the gate decision itself is never widened. When
  // weak_scale == 1.0 this is a no-op and A/b are untouched.
  if (weak_scale > 1.0) {
    const double inv_scale_sq = 1.0 / (weak_scale * weak_scale);
    A *= inv_scale_sq;
    b *= inv_scale_sq;
  }

  // A visually starved frame must not move the filter.  In particular, a
  // handful of tracks can constrain only a low-rank subspace whose spurious
  // tail previously caused velocity/scale runaway after ~12 s.
  // P6 lowers the floors (with hard lower bounds) only while weak_scale > 1.0;
  // with weak_scale == 1.0 the historical thresholds are used verbatim.
  int min_features_gate = std::max(1, options_.min_features_);
  int min_observations_gate = std::max(1, options_.min_update_observations_);
  if (weak_scale > 1.0) {
    min_features_gate = std::max(2, static_cast<int>(std::floor(options_.min_features_ / weak_scale)));
    min_observations_gate =
      std::max(4, static_cast<int>(std::floor(options_.min_update_observations_ / weak_scale)));
  }
  if (summary.features_used_ < min_features_gate || summary.observations_used_ < min_observations_gate) {
    summary.failure_reason_ = "insufficient tracks/observations";
    return summary;
  }

  // ---- 3. rank-revealing square-root EKF update --------------------------
  auto abandon = [&summary](const char * _reason)
  {
    summary.failure_reason_ = _reason;
    return summary;
  };

  if (!A.allFinite() || !b.allFinite())
    return abandon("non-finite A/b");

  // A is symmetric by construction (including the W A_ff^-1 W^T Schur
  // terms); symmetrise before an eigensolver to suppress roundoff asymmetry.
  // Visual rows never touch the 15 live IMU columns, so with calibration
  // enabled only the clone+calibration bottom-right block is decomposed.  The
  // fixed-layout path keeps the historical full decomposition.
  EkfState::symmetrize(A);
  const bool reduced_eig = _state.calibrationEnabled() && n > K_IMU_STATE_SIZE;
  const int red_dim = reduced_eig ? n - K_IMU_STATE_SIZE : n;
  Eigen::MatrixXd A_red;
  if (reduced_eig) {
    A_red = A.bottomRightCorner(red_dim, red_dim);
    EkfState::symmetrize(A_red);
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig;
  eig.compute(reduced_eig ? A_red : A);
  if (eig.info() != Eigen::Success)
    return abandon("A eigensolver failed");
  const Eigen::VectorXd & lambda = eig.eigenvalues();
  if (!lambda.allFinite())
    return abandon("non-finite A eigenvalues");

  const double lambda_scale = lambda.cwiseAbs().maxCoeff();
  if (!std::isfinite(lambda_scale) || !(lambda_scale > 0.0))
    return abandon("zero A scale");
  // A real negative mode means the per-track Schur complements disagree
  // beyond roundoff.  Do not repair it with a ridge and do not apply a
  // partially corrupted update.
  if (lambda.minCoeff() < -options_.info_indefinite_tol_ * lambda_scale)
    return abandon("indefinite Schur information");

  const double rank_tol = options_.info_rank_rtol_ * lambda_scale;
  int rank = 0;
  double lambda_min_kept = std::numeric_limits<double>::infinity();
  double lambda_max_kept = 0.0;
  for (int i = 0; i < red_dim; ++i) {
    if (lambda(i) > rank_tol) {
      ++rank;
      lambda_min_kept = std::min(lambda_min_kept, lambda(i));
      lambda_max_kept = std::max(lambda_max_kept, lambda(i));
    }
  }
  if (rank == 0 || !(lambda_min_kept > 0.0))
    return abandon("zero-rank Schur information");
  summary.rank_ = rank;
  summary.condition_ = lambda_max_kept / lambda_min_kept;
  if (reduced_eig && summary.condition_ > options_.condition_limit_ && options_.condition_limit_ > 0.0) {
    // The reduced information mixes rotation/translation/td physical units.
    // Only when the raw metric would reject the frame, re-measure it after
    // diagonal normalising EVERY reduced column to unit information.  This is
    // a dimensionless correlation condition, used for the gate only; the
    // update still uses the unscaled A/eigenvectors above.  Keeping the second
    // eigensolve behind the raw-limit check avoids its cost on normal frames.
    Eigen::VectorXd D = Eigen::VectorXd::Ones(red_dim);
    const double diag_scale = std::max(1.0, A_red.diagonal().cwiseAbs().maxCoeff());
    for (int i = 0; i < red_dim; ++i) {
      const double d = std::max(0.0, A_red(i, i));
      if (d > options_.info_rank_rtol_ * diag_scale)
        D(i) = 1.0 / std::sqrt(d);
    }
    const Eigen::MatrixXd A_scaled = D.asDiagonal() * A_red * D.asDiagonal();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> scaled_eig(A_scaled);
    if (scaled_eig.info() != Eigen::Success)
      return abandon("scaled A eigensolver failed");
    const Eigen::VectorXd & scaled_lambda = scaled_eig.eigenvalues();
    const double scaled_scale = scaled_lambda.cwiseAbs().maxCoeff();
    if (scaled_scale > 0.0) {
      const double scaled_tol = options_.info_rank_rtol_ * scaled_scale;
      double scaled_min = std::numeric_limits<double>::infinity();
      double scaled_max = 0.0;
      for (int i = 0; i < red_dim; ++i) {
        if (scaled_lambda(i) > scaled_tol) {
          scaled_min = std::min(scaled_min, scaled_lambda(i));
          scaled_max = std::max(scaled_max, scaled_lambda(i));
        }
      }
      if (scaled_min > 0.0 && std::isfinite(scaled_max))
        summary.condition_ = scaled_max / scaled_min;
    }
  }
  if (!std::isfinite(summary.condition_) || (options_.condition_limit_ > 0.0 && summary.condition_ > options_.condition_limit_))
    return abandon("Schur information condition limit exceeded");

  // Square-root information L: L^T L is the retained part of A.  The
  // measurement residual z is chosen so that L^T z = V V^T b.
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(rank, n);
  Eigen::VectorXd z(rank);
  int row = 0;
  const Eigen::VectorXd & b_eig = reduced_eig ? b.tail(red_dim) : b;
  for (int i = 0; i < red_dim; ++i) {
    if (!(lambda(i) > rank_tol))
      continue;
    const double sqrt_lambda = std::sqrt(lambda(i));
    if (reduced_eig)
      L.block(row, K_IMU_STATE_SIZE, 1, red_dim) = sqrt_lambda * eig.eigenvectors().col(i).transpose();
    else
      L.row(row) = sqrt_lambda * eig.eigenvectors().col(i).transpose();
    z(row) = eig.eigenvectors().col(i).dot(b_eig) / sqrt_lambda;
    ++row;
  }
  if (!L.allFinite() || !z.allFinite())
    return abandon("non-finite square-root information");

  // Standard KF update with R = I.  K is never formed by an explicit inverse:
  // X = S^-1 (L P), K^T = X.
  const Eigen::MatrixXd LP = L * P;
  Eigen::MatrixXd S = LP * L.transpose();
  S.diagonal().array() += 1.0;
  Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
  if (ldlt.info() != Eigen::Success)
    return abandon("innovation LDLT failed");
  const Eigen::VectorXd s_pivots = ldlt.vectorD();
  if (!s_pivots.allFinite() || !(s_pivots.minCoeff() > 0.0))
    return abandon("non-positive innovation covariance");

  const Eigen::MatrixXd X = ldlt.solve(LP);  // rank x n
  const Eigen::VectorXd y = ldlt.solve(z);
  if (!X.allFinite() || !y.allFinite())
    return abandon("non-finite innovation solve");

  const Eigen::MatrixXd K = X.transpose();  // n x rank
  const Eigen::VectorXd dx = K * z;
  if (!dx.allFinite())
    return abandon("non-finite state correction");

  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
  const Eigen::MatrixXd IKL = I - K * L;
  Eigen::MatrixXd P_new = IKL * P * IKL.transpose() + K * K.transpose();
  if (!P_new.allFinite())
    return abandon("non-finite Joseph covariance");
  EkfState::symmetrize(P_new);
  if (!P_new.allFinite())
    return abandon("non-finite symmetrised covariance");
  if (!EkfState::isPositiveSemidefinite(P_new))
    return abandon("non-PSD Joseph covariance");

  // Apply the correction to the nominal state (right perturbation). FEJ stays
  // frozen: clone FEJ values were captured at augmentation and must not be
  // overwritten after a visual update.
  Eigen::Matrix<double, 6, 1> dpose;
  dpose.head<3>() = dx.segment<3>(K_IMU_THETA);
  dpose.tail<3>() = dx.segment<3>(K_IMU_P);
  EkfState::applyPoseCorrection(_state.imu().q_, _state.imu().p_, dpose);
  _state.imu().v_ += dx.segment<3>(K_IMU_V);
  _state.imu().ba_ += dx.segment<3>(K_IMU_BA);
  _state.imu().bg_ += dx.segment<3>(K_IMU_BG);
  for (int i = 0; i < _state.numClones(); ++i) {
    const Eigen::Matrix<double, 6, 1> dclone = dx.segment<6>(_state.cloneOffset(i));
    EkfState::applyPoseCorrection(_state.clone(i).q_, _state.clone(i).p_, dclone);
  }
  if (_state.calibrationEnabled()) {
    if (_state.cameraCalibSize() == K_CAMERA_EXTRINSIC_STATE_SIZE) {
      for (int cam = 0; cam < _state.cameraCount(); ++cam) {
        const Eigen::Matrix<double, 6, 1> dcalib = dx.segment<6>(_state.cameraCalibOffset(cam));
        // dtd is injected once below; camera != 0 ignores it by construction.
        _state.injectCameraCalibrationDelta(cam, dcalib, 0.0);
      }
    }
    if (_state.hasTimeOffsetState())
      _state.injectSharedTimeOffsetDelta(dx(_state.timeOffsetOffset()));
  }
  P = P_new;
  EkfState::symmetrize(P);
  summary.applied_ = true;

  // ---- 4. summary -------------------------------------------------------
  if (residual_count > 0) {
    summary.mean_residual_px_ = residual_sum_px / static_cast<double>(residual_count);
    summary.max_residual_px_ = residual_max_px;
  }
  return summary;
}


UpdateSummary_s SchurUpdater::update(EkfState & _state, std::vector<FeatureTrack_s> & _tracks)
{
  std::vector<CameraExtrinsics_s> cameras(static_cast<std::size_t>(std::max(0, _state.cameraCount())));
  for (int cam = 0; cam < _state.cameraCount(); ++cam)
    cameras[static_cast<std::size_t>(cam)] = camera_from_state(_state, cam);
  return update(_state, _tracks, cameras);
}

}  // namespace backend
}  // namespace vins
