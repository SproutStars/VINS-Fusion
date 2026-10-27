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
constexpr double kNormalQuantile95 = 1.6448536269514722;
// Depth below which a camera-frame point counts as "behind the camera" [m].
constexpr double kMinCameraZ = 1e-6;
// A triangulated point whose RMS reprojection error exceeds this is dropped.
constexpr double kMaxTriangulationRmsPx = 5.0;

// Normalised projection h = (x/z, y/z).
inline Eigen::Vector2d project_normalized(const Eigen::Vector3d &p_c)
{
    return Eigen::Vector2d(p_c.x() / p_c.z(), p_c.y() / p_c.z());
}

// d(h)/d(p_c) for the normalised projection (2x3).
inline Eigen::Matrix<double, 2, 3> projection_jacobian(const Eigen::Vector3d &p_c)
{
    const double inv_z = 1.0 / p_c.z();
    Eigen::Matrix<double, 2, 3> J;
    J(0, 0) = inv_z;
    J(0, 1) = 0.0;
    J(0, 2) = -p_c.x() * inv_z * inv_z;
    J(1, 0) = 0.0;
    J(1, 1) = inv_z;
    J(1, 2) = -p_c.y() * inv_z * inv_z;
    return J;
}

// ---------------------------------------------------------------------------
// Scratch storage.  The frozen header leaves no room for member buffers, so the
// reusable work buffers live here (thread-local, one set per thread).  They are
// reused across tracks so the hot path performs no per-observation allocation.
// ---------------------------------------------------------------------------
std::vector<ObservationJacobian> &jacobian_scratch()
{
    static thread_local std::vector<ObservationJacobian> scratch;
    return scratch;
}

// Fills `out` with the weighted jacobian rows of one track (resized, not
// reallocated when the track length does not grow).  Returns false when an
// observation is degenerate: unknown clone or camera index, point behind the
// camera, or non-finite numbers.
bool fill_track_jacobians(const SchurUpdater &updater, const EkfState &state,
                          const std::vector<CameraExtrinsics> &cameras, const FeatureTrack &track,
                          std::vector<ObservationJacobian> &out)
{
    const int num_obs = static_cast<int>(track.observations.size());
    out.resize(static_cast<std::size_t>(num_obs));
    for (int j = 0; j < num_obs; ++j)
    {
        const FeatureObservation &obs = track.observations[static_cast<std::size_t>(j)];
        if (obs.clone_index < 0 || obs.clone_index >= state.num_clones()) return false;
        if (obs.camera_id < 0 || obs.camera_id >= static_cast<int>(cameras.size())) return false;
        out[static_cast<std::size_t>(j)] =
            updater.observation_jacobian(state, cameras[static_cast<std::size_t>(obs.camera_id)],
                                         track.world_point, obs);
        if (!out[static_cast<std::size_t>(j)].valid) return false;
    }
    return true;
}

}  // namespace

double SchurUpdater::chi2_quantile_95(int dof)
{
    if (dof <= 0) return 0.0;
    const double d = static_cast<double>(dof);
    const double t = 1.0 - 2.0 / (9.0 * d) + kNormalQuantile95 * std::sqrt(2.0 / (9.0 * d));
    return d * t * t * t;
}

ObservationJacobian SchurUpdater::observation_jacobian(const EkfState &state,
                                                      const CameraExtrinsics &camera,
                                                      const Eigen::Vector3d &world_point,
                                                      const FeatureObservation &observation) const
{
    ObservationJacobian J;
    const double sigma = options_.pixel_noise / options_.focal_length;
    if (!(sigma > 0.0) || !(options_.focal_length > 0.0)) return J;
    if (observation.clone_index < 0 || observation.clone_index >= state.num_clones()) return J;
    if (!world_point.allFinite() || !observation.uv.allFinite()) return J;

    const CloneNominal &clone = state.clone(observation.clone_index);
    const CloneNominal &clone_lin = state.clone_fej(observation.clone_index);
    const Eigen::Matrix3d R_cb = EkfState::rotation(camera.q_cb);

    // Residual at the current clone estimate, matching the OpenVINS FEJ update
    // (the residual is evaluated at the current state).
    const Eigen::Matrix3d R_i = EkfState::rotation(clone.q);
    const Eigen::Vector3d u = R_i.transpose() * (world_point - clone.p);
    const Eigen::Vector3d p_c = R_cb.transpose() * (u - camera.p_cb);
    J.p_c = p_c;
    if (!(p_c.z() > kMinCameraZ)) return J;  // behind / grazing the camera
    const Eigen::Vector2d h = project_normalized(p_c);
    J.residual = (observation.uv - h) / sigma;

    // Jacobian at the frozen clone FEJ, matching OpenVINS
    // UpdaterHelper::get_feature_jacobian_3d().
    const Eigen::Matrix3d R_i_lin = EkfState::rotation(clone_lin.q);
    const Eigen::Vector3d u_lin = R_i_lin.transpose() * (world_point - clone_lin.p);
    const Eigen::Vector3d p_c_lin = R_cb.transpose() * (u_lin - camera.p_cb);
    if (!(p_c_lin.z() > kMinCameraZ)) return J;

    const Eigen::Matrix<double, 2, 3> dh_dpc = projection_jacobian(p_c_lin);

    // dp_c/d[dtheta_i | dp_i] and dp_c/d(P_w), all evaluated at the FEJ pose.
    Eigen::Matrix<double, 3, 6> dpc_dpose;
    dpc_dpose.block<3, 3>(0, 0) = R_cb.transpose() * EkfState::skew(u_lin);
    dpc_dpose.block<3, 3>(0, 3) = -R_cb.transpose() * R_i_lin.transpose();
    const Eigen::Matrix3d dpc_dpoint = R_cb.transpose() * R_i_lin.transpose();

    J.H_pose = (dh_dpc * dpc_dpose) / sigma;
    J.H_point = (dh_dpc * dpc_dpoint) / sigma;
    J.valid = J.H_pose.allFinite() && J.H_point.allFinite() && J.residual.allFinite();
    return J;
}

bool SchurUpdater::triangulate(const EkfState &state, const std::vector<CameraExtrinsics> &cameras,
                               FeatureTrack &track) const
{
    track.has_point = false;
    const int num_obs = static_cast<int>(track.observations.size());
    if (num_obs < 2) return false;
    if (!(options_.min_depth > 0.0) || !(options_.max_depth > options_.min_depth)) return false;

    for (const FeatureObservation &obs : track.observations)
    {
        if (obs.clone_index < 0 || obs.clone_index >= state.num_clones()) return false;
        if (obs.camera_id < 0 || obs.camera_id >= static_cast<int>(cameras.size())) return false;
        if (!obs.uv.allFinite()) return false;
    }

    // ---- two-view linear initialisation (anchor = first observation) -------
    const FeatureObservation &first = track.observations.front();
    const FeatureObservation &last = track.observations.back();
    if (first.clone_index == last.clone_index) return false;  // no baseline

    const CloneNominal &c0 = state.clone(first.clone_index);
    const CloneNominal &c1 = state.clone(last.clone_index);
    const CameraExtrinsics &cam0 = cameras[static_cast<std::size_t>(first.camera_id)];
    const CameraExtrinsics &cam1 = cameras[static_cast<std::size_t>(last.camera_id)];
    const Eigen::Matrix3d R0 = EkfState::rotation(c0.q);
    const Eigen::Matrix3d R1 = EkfState::rotation(c1.q);
    const Eigen::Matrix3d R_cb0 = EkfState::rotation(cam0.q_cb);
    const Eigen::Matrix3d R_cb1 = EkfState::rotation(cam1.q_cb);

    const Eigen::Vector3d b0(first.uv.x(), first.uv.y(), 1.0);
    // Point in the anchor camera frame at depth lambda:  p_C = lambda * b0.
    // Seen from the last clone: p_c1 = m + lambda * d, which must be parallel
    // to b1, i.e. [b1]_x (m + lambda d) = 0 solved in least squares.
    const Eigen::Vector3d t0 = R1.transpose() * ((c0.p - c1.p) + R0 * cam0.p_cb);
    const Eigen::Vector3d m = R_cb1.transpose() * (t0 - cam1.p_cb);
    const Eigen::Vector3d d =
        R_cb1.transpose() * R1.transpose() * R0 * R_cb0 * b0;
    const Eigen::Vector3d b1(last.uv.x(), last.uv.y(), 1.0);
    const Eigen::Matrix3d Sx = EkfState::skew(b1);
    const Eigen::Vector3d lhs = Sx * d;
    const double denom = lhs.squaredNorm();
    if (!(denom > 1e-18)) return false;  // (nearly) parallel rays
    const double lambda = -lhs.dot(Sx * m) / denom;
    if (!std::isfinite(lambda) || !(lambda > 0.0)) return false;  // behind the anchor

    // ---- Gauss-Newton on the anchored inverse-depth parameterisation -------
    double alpha = first.uv.x();
    double beta = first.uv.y();
    double rho = 1.0 / lambda;
    if (!std::isfinite(rho) || !(rho > 0.0)) return false;

    for (int iter = 0; iter < options_.triangulation_iterations; ++iter)
    {
        const Eigen::Vector3d p_C(alpha / rho, beta / rho, 1.0 / rho);
        const Eigen::Vector3d P_w = c0.p + R0 * (R_cb0 * p_C + cam0.p_cb);

        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        Eigen::Vector3d g = Eigen::Vector3d::Zero();
        for (const FeatureObservation &obs : track.observations)
        {
            const CloneNominal &ci = state.clone(obs.clone_index);
            const CameraExtrinsics &cam = cameras[static_cast<std::size_t>(obs.camera_id)];
            const Eigen::Matrix3d Ri = EkfState::rotation(ci.q);
            const Eigen::Matrix3d Rc = EkfState::rotation(cam.q_cb);
            const Eigen::Vector3d u = Ri.transpose() * (P_w - ci.p);
            const Eigen::Vector3d p_c = Rc.transpose() * (u - cam.p_cb);
            if (!(p_c.z() > kMinCameraZ)) return false;

            // dp_c/d(alpha, beta, rho): chain rule from the anchor camera frame
            // through the world frame into this camera frame.
            const Eigen::Matrix3d chain = Rc.transpose() * Ri.transpose() * R0 * R_cb0;
            Eigen::Matrix3d dpc_dp;
            dpc_dp.col(0) = chain * Eigen::Vector3d(1.0 / rho, 0.0, 0.0);
            dpc_dp.col(1) = chain * Eigen::Vector3d(0.0, 1.0 / rho, 0.0);
            dpc_dp.col(2) = -(chain * p_C) / rho;

            const Eigen::Vector2d res = obs.uv - project_normalized(p_c);
            // J = d(residual)/d(alpha, beta, rho).  Normal equations are
            // H = sum J^T J, g = sum J^T res and the Gauss-Newton step is
            // x <- x - H^-1 g (a *descent* step, hence the subtraction below).
            const Eigen::Matrix<double, 2, 3> J = -projection_jacobian(p_c) * dpc_dp;
            H.noalias() += J.transpose() * J;
            g.noalias() += J.transpose() * res;
        }
        if (!H.allFinite() || !g.allFinite()) return false;

        Eigen::LDLT<Eigen::Matrix3d> ldlt(H);
        if (ldlt.info() != Eigen::Success) return false;
        const Eigen::Vector3d pivots = ldlt.vectorD();
        if (!(pivots.minCoeff() > 1e-12 * std::max(1.0, pivots.maxCoeff()))) return false;
        const Eigen::Vector3d step = ldlt.solve(g);
        if (!step.allFinite()) return false;
        alpha -= step(0);
        beta -= step(1);
        rho -= step(2);
        if (!std::isfinite(alpha) || !std::isfinite(beta) || !std::isfinite(rho)) return false;
    }

    // ---- depth clamp, RMS check and output --------------------------------
    if (!(rho > 0.0) || !std::isfinite(rho)) return false;
    const double depth = 1.0 / rho;
    if (!std::isfinite(depth)) return false;
    if (depth < options_.min_depth || depth > options_.max_depth) return false;  // clamped

    const Eigen::Vector3d p_C(alpha / rho, beta / rho, 1.0 / rho);
    const Eigen::Vector3d P_w = c0.p + R0 * (R_cb0 * p_C + cam0.p_cb);
    if (!P_w.allFinite()) return false;

    double sq_res = 0.0;
    for (const FeatureObservation &obs : track.observations)
    {
        const CloneNominal &ci = state.clone(obs.clone_index);
        const CameraExtrinsics &cam = cameras[static_cast<std::size_t>(obs.camera_id)];
        const Eigen::Matrix3d Ri = EkfState::rotation(ci.q);
        const Eigen::Matrix3d Rc = EkfState::rotation(cam.q_cb);
        const Eigen::Vector3d u = Ri.transpose() * (P_w - ci.p);
        const Eigen::Vector3d p_c = Rc.transpose() * (u - cam.p_cb);
        if (!(p_c.z() > kMinCameraZ)) return false;
        const Eigen::Vector2d res = obs.uv - project_normalized(p_c);
        if (!res.allFinite()) return false;
        sq_res += res.squaredNorm();
    }
    const double rms_residual_px =
        std::sqrt(sq_res / static_cast<double>(num_obs)) * options_.focal_length;
    if (!(rms_residual_px <= kMaxTriangulationRmsPx)) return false;

    // Triangulated from the current clone poses on every use (OpenVINS
    // UpdaterMSCKF behaviour). The point is intentionally *not* cached across
    // frames: a feature is consumed by at most one update, and freezing the
    // point (VINS_SCHUR_FREEZE A/B) was measured to be slightly worse on full
    // sequences (V2_01 0.2444 vs 0.2295 m, V2_02 0.3895 vs 0.3809 m), so the
    // freeze branch was removed.
    track.world_point = P_w;
    track.has_point = true;
    return true;
}

TrackReduction SchurUpdater::reduce_track(const EkfState &state,
                                          const std::vector<CameraExtrinsics> &cameras,
                                          const FeatureTrack &track) const
{
    TrackReduction red;
    const int num_obs = static_cast<int>(track.observations.size());
    if (num_obs == 0) return red;
    const Eigen::MatrixXd &P = state.covariance();
    if (P.rows() != state.size() || P.cols() != state.size()) return red;

    std::vector<ObservationJacobian> &jac = jacobian_scratch();
    if (!fill_track_jacobians(*this, state, cameras, track, jac)) return red;

    // ---- point block and per-clone W blocks -------------------------------
    double sq_res = 0.0;
    for (int j = 0; j < num_obs; ++j)
    {
        const ObservationJacobian &J = jac[static_cast<std::size_t>(j)];
        red.A_point.noalias() += J.H_point.transpose() * J.H_point;
        red.b_point.noalias() += J.H_point.transpose() * J.residual;
        sq_res += J.residual.squaredNorm();

        const int ci = track.observations[static_cast<std::size_t>(j)].clone_index;
        auto it = red.W.find(ci);
        if (it == red.W.end())
        {
            it = red.W.emplace(ci, Eigen::Matrix<double, 6, 3>::Zero()).first;
        }
        it->second.noalias() += J.H_pose.transpose() * J.H_point;
    }

    // Weighted -> pixel residual: ||r|| * sigma * focal_length.
    const double sigma = options_.pixel_noise / options_.focal_length;
    red.rms_residual_px =
        std::sqrt(sq_res / static_cast<double>(num_obs)) * sigma * options_.focal_length;

    // Reject near-degenerate triangulations before they are used in the Schur
    // complement.  The relative eigenvalue test is scale free and much more
    // informative than raw LDLT pivots.
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> point_eig(red.A_point);
    if (point_eig.info() != Eigen::Success) return red;
    const Eigen::Vector3d point_lambda = point_eig.eigenvalues();
    const double point_max = point_lambda.maxCoeff();
    const double point_min = point_lambda.minCoeff();
    if (!red.A_point.allFinite() || !std::isfinite(point_max) ||
        !(point_max > 0.0) || !(point_min > options_.point_block_rtol * point_max))
        return red;

    Eigen::LDLT<Eigen::Matrix3d> point_ldlt(red.A_point);
    if (point_ldlt.info() != Eigen::Success) return red;

    // ---- chi2 gate ---------------------------------------------------------
    // Profile the 3-D point out of the innovation covariance as well.  Using
    // only H_x P H_x^T + I would make short tracks appear artificially certain
    // (their residual can always be absorbed by moving the point).
    const int m = 2 * num_obs;
    Eigen::MatrixXd Hf = Eigen::MatrixXd::Zero(m, 3);
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(m, m);
    Eigen::VectorXd r(m);
    for (int a = 0; a < num_obs; ++a)
    {
        const ObservationJacobian &Ja = jac[static_cast<std::size_t>(a)];
        Hf.block<2, 3>(2 * a, 0) = Ja.H_point;
        const int oa =
            state.clone_offset(track.observations[static_cast<std::size_t>(a)].clone_index);
        for (int b = 0; b < num_obs; ++b)
        {
            const int ob =
                state.clone_offset(track.observations[static_cast<std::size_t>(b)].clone_index);
            S.block<2, 2>(2 * a, 2 * b) =
                Ja.H_pose * P.block<6, 6>(oa, ob) *
                jac[static_cast<std::size_t>(b)].H_pose.transpose();
        }
        S.block<2, 2>(2 * a, 2 * a) += Eigen::Matrix2d::Identity();
        r.segment<2>(2 * a) = Ja.residual;
    }
    const Eigen::MatrixXd point_solve = point_ldlt.solve(Hf.transpose());
    if (!point_solve.allFinite()) return red;
    S.noalias() -= Hf * point_solve;

    Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
    const Eigen::VectorXd s_pivots = ldlt.vectorD();
    const double s_scale = s_pivots.cwiseAbs().maxCoeff();
    if (ldlt.info() != Eigen::Success || !std::isfinite(s_scale) || !(s_scale > 0.0) ||
        !(s_pivots.minCoeff() > 1e-12 * s_scale))
    {
        red.chi2 = std::numeric_limits<double>::infinity();
    }
    else
    {
        const Eigen::VectorXd y = ldlt.solve(r);
        red.chi2 = y.allFinite() ? r.dot(y) : std::numeric_limits<double>::infinity();
    }

    red.valid = std::isfinite(red.chi2) && std::isfinite(red.rms_residual_px) &&
                red.b_point.allFinite();
    return red;
}

UpdateSummary SchurUpdater::update(EkfState &state, std::vector<FeatureTrack> &tracks,
                                   const std::vector<CameraExtrinsics> &cameras)
{
    UpdateSummary summary;
    // P6 starvation-adaptive weak update. A value > 1.0 relaxes the chi2 gate
    // threshold and down-weights the accepted observations (sigma -> sigma *
    // scale, i.e. R * scale^2). Defensive sanitisation: non-finite or <= 1.0
    // leaves the update bit-for-bit unchanged.
    const double weak_scale =
        (std::isfinite(options_.weak_update_scale) && options_.weak_update_scale > 1.0)
            ? options_.weak_update_scale
            : 1.0;
    if (weak_scale > 1.0)
    {
        summary.weak_update = true;
        summary.weak_update_scale = weak_scale;
    }
    // Record every input track before any early return. This describes the
    // input set of this call; only tracks that reach the A/b accumulation are
    // eventually reported through used_feature_ids.
    summary.consumed_feature_ids.reserve(tracks.size());
    summary.used_feature_ids.reserve(tracks.size());
    for (const FeatureTrack &track : tracks)
        summary.consumed_feature_ids.push_back(track.id);

    if (!state.initialized() || state.size() <= 0) return summary;
    const int n = state.size();
    Eigen::MatrixXd &P = state.covariance();
    if (P.rows() != n || P.cols() != n) return summary;

    // ---- 1. triangulate, drop unusable tracks, keep the longest ones ------
    std::vector<int> accepted;
    accepted.reserve(tracks.size());
    for (std::size_t i = 0; i < tracks.size(); ++i)
    {
        if (static_cast<int>(tracks[i].observations.size()) < options_.min_observations)
        {
            summary.rejected_feature_ids.push_back(tracks[i].id);  // too short to triangulate
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
        if (!triangulate(state, cameras, tracks[i]))
        {
            summary.rejected_feature_ids.push_back(tracks[i].id);  // no usable point
            continue;
        }
        accepted.push_back(static_cast<int>(i));
    }
    std::stable_sort(accepted.begin(), accepted.end(), [&tracks](int lhs, int rhs) {
        return tracks[static_cast<std::size_t>(lhs)].observations.size() >
               tracks[static_cast<std::size_t>(rhs)].observations.size();
    });
    const int max_features = std::max(0, options_.max_features);
    if (static_cast<int>(accepted.size()) > max_features)
    {
        accepted.resize(static_cast<std::size_t>(max_features));
    }

    // ---- 2. per-track gate + Schur-reduced information accumulation -------
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
    double residual_sum_px = 0.0;
    double residual_max_px = 0.0;
    int residual_count = 0;
    const double sigma = options_.pixel_noise / options_.focal_length;

    for (const int index : accepted)
    {
        FeatureTrack &track = tracks[static_cast<std::size_t>(index)];
        const TrackReduction red = reduce_track(state, cameras, track);
        if (!red.valid)
        {
            summary.rejected_feature_ids.push_back(track.id);
            continue;
        }

        const int num_obs = static_cast<int>(track.observations.size());
        const int dof = 2 * num_obs;
        // P6: only the *threshold* is relaxed by weak_scale. red.chi2 is built
        // from the nominal sigma (pixel_noise / focal_length), so the gate is
        // never widened by the inflated measurement noise.
        if (red.chi2 > options_.chi2_multiplier * weak_scale * chi2_quantile_95(dof))
        {
            summary.rejected_feature_ids.push_back(track.id);  // chi2 outlier gate
            continue;
        }

        std::vector<ObservationJacobian> &jac = jacobian_scratch();
        if (!fill_track_jacobians(*this, state, cameras, track, jac))
        {
            summary.rejected_feature_ids.push_back(track.id);
            continue;
        }

        // H_x^T H_x is block diagonal (one 6x6 block per observing clone), so
        // it is accumulated one observation at a time.  H_x^T r likewise.
        for (int j = 0; j < num_obs; ++j)
        {
            const ObservationJacobian &J = jac[static_cast<std::size_t>(j)];
            const int off =
                state.clone_offset(track.observations[static_cast<std::size_t>(j)].clone_index);
            A.block<6, 6>(off, off) += J.H_pose.transpose() * J.H_pose;
            b.segment<6>(off) += J.H_pose.transpose() * J.residual;

            const double px = J.residual.norm() * sigma * options_.focal_length;
            residual_sum_px += px;
            residual_max_px = std::max(residual_max_px, px);
            ++residual_count;
        }

        // Schur elimination of this track's 3D point: the only step that
        // couples different clones.  Solve the 3x3 block once, without forming
        // an explicit inverse, and reuse the solutions for every W_i block.
        Eigen::LDLT<Eigen::Matrix3d> point_ldlt(red.A_point);
        const Eigen::Vector3d point_rhs = point_ldlt.solve(red.b_point);
        std::vector<std::pair<int, Eigen::Matrix<double, 3, 6>>> schur_blocks;
        schur_blocks.reserve(red.W.size());
        bool schur_ok = point_ldlt.info() == Eigen::Success && point_rhs.allFinite();
        for (const auto &wi : red.W)
        {
            const Eigen::Matrix<double, 3, 6> Xi = point_ldlt.solve(wi.second.transpose());
            schur_ok = schur_ok && Xi.allFinite();
            schur_blocks.emplace_back(wi.first, Xi);
        }
        if (!schur_ok)
        {
            summary.rejected_feature_ids.push_back(track.id);
            continue;
        }

        for (const auto &wi : red.W)
        {
            const int oi = state.clone_offset(wi.first);
            b.segment<6>(oi).noalias() -= wi.second * point_rhs;
            for (const auto &xj : schur_blocks)
            {
                const int oj = state.clone_offset(xj.first);
                A.block<6, 6>(oi, oj).noalias() -= wi.second * xj.second;
            }
        }

        // This track passed triangulation, chi2, point-block and Jacobian
        // checks and its terms have just been accumulated into A/b.
        summary.used_feature_ids.push_back(track.id);
        summary.features_used += 1;
        summary.observations_used += num_obs;
        summary.degrees_of_freedom += dof;
    }
    summary.features_rejected = static_cast<int>(summary.rejected_feature_ids.size());

    if (summary.features_used == 0)
    {
        summary.failure_reason = "no gated tracks";
        return summary;
    }

    // P6 down-weighting: the accumulated information is divided by scale^2,
    // exactly equivalent to weighting each accepted observation with
    // sigma * weak_scale (R * scale^2). It is applied here, after the nominal-
    // sigma chi2 gate, so the gate decision itself is never widened. When
    // weak_scale == 1.0 this is a no-op and A/b are untouched.
    if (weak_scale > 1.0)
    {
        const double inv_scale_sq = 1.0 / (weak_scale * weak_scale);
        A *= inv_scale_sq;
        b *= inv_scale_sq;
    }

    // A visually starved frame must not move the filter.  In particular, a
    // handful of tracks can constrain only a low-rank subspace whose spurious
    // tail previously caused velocity/scale runaway after ~12 s.
    // P6 lowers the floors (with hard lower bounds) only while weak_scale > 1.0;
    // with weak_scale == 1.0 the historical thresholds are used verbatim.
    int min_features_gate = std::max(1, options_.min_features);
    int min_observations_gate = std::max(1, options_.min_update_observations);
    if (weak_scale > 1.0)
    {
        min_features_gate =
            std::max(2, static_cast<int>(std::floor(options_.min_features / weak_scale)));
        min_observations_gate = std::max(
            4, static_cast<int>(std::floor(options_.min_update_observations / weak_scale)));
    }
    if (summary.features_used < min_features_gate ||
        summary.observations_used < min_observations_gate)
    {
        summary.failure_reason = "insufficient tracks/observations";
        return summary;
    }

    // ---- 3. rank-revealing square-root EKF update --------------------------
    auto abandon = [&summary](const char *reason) {
        summary.failure_reason = reason;
        return summary;
    };

    if (!A.allFinite() || !b.allFinite()) return abandon("non-finite A/b");

    // A is symmetric by construction (including the W A_ff^-1 W^T Schur
    // terms); symmetrise before an eigensolver to suppress roundoff asymmetry.
    EkfState::symmetrize(A);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(A);
    if (eig.info() != Eigen::Success) return abandon("A eigensolver failed");
    const Eigen::VectorXd lambda = eig.eigenvalues();
    if (!lambda.allFinite()) return abandon("non-finite A eigenvalues");

    const double lambda_scale = lambda.cwiseAbs().maxCoeff();
    if (!std::isfinite(lambda_scale) || !(lambda_scale > 0.0))
        return abandon("zero A scale");
    // A real negative mode means the per-track Schur complements disagree
    // beyond roundoff.  Do not repair it with a ridge and do not apply a
    // partially corrupted update.
    if (lambda.minCoeff() < -options_.info_indefinite_tol * lambda_scale)
        return abandon("indefinite Schur information");

    const double rank_tol = options_.info_rank_rtol * lambda_scale;
    int rank = 0;
    double lambda_min_kept = std::numeric_limits<double>::infinity();
    double lambda_max_kept = 0.0;
    for (int i = 0; i < n; ++i)
    {
        if (lambda(i) > rank_tol)
        {
            ++rank;
            lambda_min_kept = std::min(lambda_min_kept, lambda(i));
            lambda_max_kept = std::max(lambda_max_kept, lambda(i));
        }
    }
    if (rank == 0 || !(lambda_min_kept > 0.0))
        return abandon("zero-rank Schur information");
    summary.rank = rank;
    summary.condition = lambda_max_kept / lambda_min_kept;
    if (!std::isfinite(summary.condition) ||
        (options_.condition_limit > 0.0 && summary.condition > options_.condition_limit))
        return abandon("Schur information condition limit exceeded");

    // Square-root information L: L^T L is the retained part of A.  The
    // measurement residual z is chosen so that L^T z = V V^T b.
    Eigen::MatrixXd L(rank, n);
    Eigen::VectorXd z(rank);
    int row = 0;
    for (int i = 0; i < n; ++i)
    {
        if (!(lambda(i) > rank_tol))
            continue;
        const double sqrt_lambda = std::sqrt(lambda(i));
        L.row(row) = sqrt_lambda * eig.eigenvectors().col(i).transpose();
        z(row) = eig.eigenvectors().col(i).dot(b) / sqrt_lambda;
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
    if (ldlt.info() != Eigen::Success) return abandon("innovation LDLT failed");
    const Eigen::VectorXd s_pivots = ldlt.vectorD();
    if (!s_pivots.allFinite() || !(s_pivots.minCoeff() > 0.0))
        return abandon("non-positive innovation covariance");

    const Eigen::MatrixXd X = ldlt.solve(LP);  // rank x n
    const Eigen::VectorXd y = ldlt.solve(z);
    if (!X.allFinite() || !y.allFinite())
        return abandon("non-finite innovation solve");

    const Eigen::MatrixXd K = X.transpose();  // n x rank
    const Eigen::VectorXd dx = K * z;
    if (!dx.allFinite()) return abandon("non-finite state correction");

    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    const Eigen::MatrixXd IKL = I - K * L;
    Eigen::MatrixXd P_new = IKL * P * IKL.transpose() + K * K.transpose();
    if (!P_new.allFinite()) return abandon("non-finite Joseph covariance");
    EkfState::symmetrize(P_new);
    if (!P_new.allFinite()) return abandon("non-finite symmetrised covariance");
    if (!EkfState::is_positive_semidefinite(P_new))
        return abandon("non-PSD Joseph covariance");

    // Apply the correction to the nominal state (right perturbation). FEJ stays
    // frozen: clone FEJ values were captured at augmentation and must not be
    // overwritten after a visual update.
    Eigen::Matrix<double, 6, 1> dpose;
    dpose.head<3>() = dx.segment<3>(kImuTheta);
    dpose.tail<3>() = dx.segment<3>(kImuP);
    EkfState::apply_pose_correction(state.imu().q, state.imu().p, dpose);
    state.imu().v += dx.segment<3>(kImuV);
    state.imu().ba += dx.segment<3>(kImuBa);
    state.imu().bg += dx.segment<3>(kImuBg);
    for (int i = 0; i < state.num_clones(); ++i)
    {
        const Eigen::Matrix<double, 6, 1> dclone = dx.segment<6>(state.clone_offset(i));
        EkfState::apply_pose_correction(state.clone(i).q, state.clone(i).p, dclone);
    }
    P = P_new;
    EkfState::symmetrize(P);
    summary.applied = true;

    // ---- 4. summary -------------------------------------------------------
    if (residual_count > 0)
    {
        summary.mean_residual_px = residual_sum_px / static_cast<double>(residual_count);
        summary.max_residual_px = residual_max_px;
    }
    return summary;
}

}  // namespace backend
}  // namespace vins
