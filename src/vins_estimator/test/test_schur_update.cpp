/*******************************************************
 * Unit tests for the ARM-oriented Schur-complement visual update
 * (src/vins_estimator/src/backend/schur_update.{h,cpp}).
 *
 * Standalone: gtest + Eigen only, no ROS / Ceres / OpenCV.
 *
 *   g++ -std=c++17 -O2 -Wall -Wextra -I src/vins_estimator/src -I /usr/include/eigen3 \
 *       src/vins_estimator/src/backend/ekf_state.cpp \
 *       src/vins_estimator/src/backend/schur_update.cpp \
 *       src/vins_estimator/test/test_schur_update.cpp \
 *       -lgtest -lgtest_main -pthread -o /tmp/test_schur_update
 *
 * Conventions: q = [w,x,y,z] (Hamilton, body -> world), right/body
 * perturbation, error state [dtheta, dp, dv, dba, dbg] + 6 per clone.
 *******************************************************/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include "backend/feature_history.h"
#include "backend/schur_update.h"

namespace
{

using namespace vins::backend;

constexpr double kFocal = 460.0;
constexpr double kPixelNoise = 1.5;

UpdateOptions default_options()
{
    UpdateOptions o;
    o.max_features = 120;
    o.min_observations = 2;
    // The algorithm-level tests below use one or very few tracks.  Keep the
    // production frame-level starvation gates from turning those tests into
    // no-op updates; the dedicated gate test overrides these values.
    o.min_features = 1;
    o.min_update_observations = 1;
    o.pixel_noise = kPixelNoise;
    o.focal_length = kFocal;
    o.chi2_multiplier = 5.0;
    o.triangulation_iterations = 5;
    o.min_depth = 0.1;
    o.max_depth = 60.0;
    o.condition_limit = 1e6;
    return o;
}

EkfOptions default_ekf_options(int window_size)
{
    EkfOptions o;
    o.window_size = window_size;
    return o;
}

Eigen::Vector4d quat(const Eigen::Vector3d &axis_angle)
{
    return EkfState::quaternion_from_axis_angle(axis_angle);
}

Eigen::Vector4d random_quat(std::mt19937 &rng, double max_angle)
{
    std::uniform_real_distribution<double> d(-max_angle, max_angle);
    return quat(Eigen::Vector3d(d(rng), d(rng), d(rng)));
}

// Projection of a world point into the camera attached to the clone (q, p).
Eigen::Vector2d project_world(const Eigen::Vector4d &q, const Eigen::Vector3d &p,
                              const CameraExtrinsics &cam, const Eigen::Vector3d &world_point)
{
    const Eigen::Matrix3d R_i = EkfState::rotation(q);
    const Eigen::Matrix3d R_cb = EkfState::rotation(cam.q_cb);
    const Eigen::Vector3d u = R_i.transpose() * (world_point - p);
    const Eigen::Vector3d p_c = R_cb.transpose() * (u - cam.p_cb);
    return Eigen::Vector2d(p_c.x() / p_c.z(), p_c.y() / p_c.z());
}

// Independent (test local) weighted residual model: (uv - h)/sigma.
Eigen::Vector2d weighted_residual_model(const Eigen::Vector4d &q, const Eigen::Vector3d &p,
                                        const CameraExtrinsics &cam, const Eigen::Vector3d &world_point,
                                        const Eigen::Vector2d &uv, double sigma)
{
    return (uv - project_world(q, p, cam, world_point)) / sigma;
}

// Inverse of a unit quaternion.  Note that Eigen::Vector4d::conjugate()
// conjugates each scalar entry (a no-op for real doubles); it does NOT negate
// the vector part, so it cannot be used as a quaternion inverse.
Eigen::Vector4d quat_inverse(const Eigen::Vector4d &q)
{
    Eigen::Vector4d inv = q;
    inv.tail<3>() *= -1.0;
    return inv;
}

// Exact inverse of EkfState::apply_pose_correction's first-order update
//   q_after = q_before (x) [1, dtheta/2] / norm  =>  dtheta = 2 * vec / w.
// Using this (rather than the exact log map) removes the O(|dtheta|^2)
// mismatch between the first-order mean update and the reconstruction.
Eigen::Vector3d correction_from_quat(const Eigen::Vector4d &before,
                                     const Eigen::Vector4d &after)
{
    const Eigen::Vector4d qrel = EkfState::quaternion_multiply(
        quat_inverse(EkfState::quaternion_normalize(before)), after);
    if (std::abs(qrel(0)) < 1e-12) return EkfState::quaternion_to_axis_angle(qrel);
    return 2.0 * qrel.tail<3>() / qrel(0);
}

// Rotation vector of a quaternion (inverse of quaternion_from_axis_angle).
Eigen::Vector3d log_quat(const Eigen::Vector4d &q_in)
{
    const Eigen::Vector4d q = q_in.normalized();
    const double w = std::max(-1.0, std::min(1.0, q(0)));  // [w,x,y,z], not Eigen .w()
    const double s = std::sqrt(std::max(0.0, 1.0 - w * w));
    if (s < 1e-12) return Eigen::Vector3d::Zero();
    const double angle = 2.0 * std::atan2(s, w);
    return (angle / s) * q.tail<3>();
}

// Builds a state whose clones sit exactly at the given poses (window large
// enough that nothing is marginalised) and whose live pose is the last one.
void build_state(EkfState &state, const std::vector<Eigen::Vector4d> &qs,
                 const std::vector<Eigen::Vector3d> &ps, double vel_noise = 0.0,
                 std::mt19937 *rng = nullptr)
{
    ASSERT_EQ(qs.size(), ps.size());
    const int n = static_cast<int>(qs.size());
    EkfOptions opt = default_ekf_options(n + 2);
    Eigen::Vector3d v = Eigen::Vector3d::Zero();
    if (rng != nullptr && vel_noise > 0.0)
    {
        std::uniform_real_distribution<double> d(-vel_noise, vel_noise);
        v << d(*rng), d(*rng), d(*rng);
    }
    state.reset(opt, 0.0, qs[0], ps[0], v, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    state.augment_clone(0.0);
    for (int i = 1; i < n; ++i)
    {
        state.imu().q = qs[static_cast<std::size_t>(i)];
        state.imu().p = ps[static_cast<std::size_t>(i)];
        // This test helper bypasses propagate(), which is what normally advances
        // imu_fej(). Keep the live FEJ synchronized with the pose being cloned.
        state.refresh_imu_fej();
        state.augment_clone(static_cast<double>(i));
    }
}

// A freshly augmented clone is an exact copy of the live pose, so the raw
// covariance has zero-variance live/clone difference directions and is
// singular.  A real filter always adds process noise between frames; emulate
// that with a tiny isotropic jitter so that the information-form reference
// P^-1 used by T4 is well defined and S is not numerically indefinite.
void regularize_covariance(EkfState &state, double sigma = 1e-4)
{
    state.covariance().diagonal().array() += sigma * sigma;
}

CameraExtrinsics make_camera(const Eigen::Vector3d &axis_angle, const Eigen::Vector3d &p_cb)
{
    CameraExtrinsics cam;
    cam.q_cb = quat(axis_angle);
    cam.p_cb = p_cb;
    return cam;
}

FeatureObservation make_observation(int clone_index, int camera_id, const Eigen::Vector2d &uv)
{
    FeatureObservation obs;
    obs.clone_index = clone_index;
    obs.camera_id = camera_id;
    obs.uv = uv;
    return obs;
}

// A straight, smooth camera trajectory in front of a static point cloud.
void smooth_trajectory(int num_frames, std::vector<Eigen::Vector4d> &qs, std::vector<Eigen::Vector3d> &ps)
{
    qs.clear();
    ps.clear();
    for (int i = 0; i < num_frames; ++i)
    {
        const double t = static_cast<double>(i);
        ps.push_back(Eigen::Vector3d(0.25 * t, 0.05 * t * t, 0.01 * t));
        qs.push_back(quat(Eigen::Vector3d(0.0, -0.02 * t, 0.01 * t)));
    }
}

// ---------------------------------------------------------------------------
// Joint (state + point) system of one track, built independently from the
// observation jacobians, used to check the Schur reduction and the EKF update.
// ---------------------------------------------------------------------------
struct JointSystem
{
    Eigen::MatrixXd Hx;  // 2M x n
    Eigen::MatrixXd Hf;  // 2M x 3
    Eigen::VectorXd r;   // 2M
    int num_obs = 0;
};

JointSystem build_joint(const SchurUpdater &updater, const EkfState &state,
                        const std::vector<CameraExtrinsics> &cameras, const FeatureTrack &track)
{
    JointSystem joint;
    const int m = static_cast<int>(track.observations.size());
    const int n = state.size();
    joint.num_obs = m;
    joint.Hx = Eigen::MatrixXd::Zero(2 * m, n);
    joint.Hf = Eigen::MatrixXd::Zero(2 * m, 3);
    joint.r = Eigen::VectorXd::Zero(2 * m);
    for (int j = 0; j < m; ++j)
    {
        const FeatureObservation &obs = track.observations[static_cast<std::size_t>(j)];
        const ObservationJacobian J =
            updater.observation_jacobian(state, cameras[static_cast<std::size_t>(obs.camera_id)],
                                         track.world_point, obs);
        const int off = state.clone_offset(obs.clone_index);
        joint.Hx.block<2, 6>(2 * j, off) = J.H_pose;
        joint.Hf.block<2, 3>(2 * j, 0) = J.H_point;
        joint.r.segment<2>(2 * j) = J.residual;
    }
    return joint;
}

// A_red = H_x^T H_x - W A_ff^-1 W^T, b_red = H_x^T r - W A_ff^-1 b_f  (step D
// of the specification, using the blocks reported by reduce_track()).
void reduce_with_blocks(const EkfState &state, const JointSystem &joint, const TrackReduction &red,
                        Eigen::MatrixXd &A_red, Eigen::VectorXd &b_red)
{
    A_red = joint.Hx.transpose() * joint.Hx;
    b_red = joint.Hx.transpose() * joint.r;
    const Eigen::Matrix3d Ainv = red.A_point.inverse();
    for (const auto &wi : red.W)
    {
        const int oi = state.clone_offset(wi.first);
        b_red.segment<6>(oi) -= wi.second * Ainv * red.b_point;
        for (const auto &wj : red.W)
        {
            const int oj = state.clone_offset(wj.first);
            A_red.block<6, 6>(oi, oj) -= wi.second * Ainv * wj.second.transpose();
        }
    }
}

// Exact Schur complement of the joint system (point block eliminated).
void reduce_exact(const JointSystem &joint, Eigen::MatrixXd &A_ref, Eigen::VectorXd &b_ref)
{
    const Eigen::MatrixXd A_xx = joint.Hx.transpose() * joint.Hx;
    const Eigen::MatrixXd A_xf = joint.Hx.transpose() * joint.Hf;
    const Eigen::MatrixXd A_ff = joint.Hf.transpose() * joint.Hf;
    const Eigen::VectorXd b_x = joint.Hx.transpose() * joint.r;
    const Eigen::VectorXd b_f = joint.Hf.transpose() * joint.r;
    const Eigen::Matrix3d Ainv = A_ff.inverse();
    A_ref = A_xx - A_xf * Ainv * A_xf.transpose();
    b_ref = b_x - A_xf * Ainv * b_f;
}

// ---------------------------------------------------------------------------
// State snapshot for exact before/after comparisons.
// ---------------------------------------------------------------------------
struct Snapshot
{
    Eigen::Vector4d q;
    Eigen::Vector3d p, v, ba, bg;
    std::vector<Eigen::Vector4d> clone_q;
    std::vector<Eigen::Vector3d> clone_p;
    Eigen::MatrixXd P;
};

Snapshot snapshot(const EkfState &state)
{
    Snapshot s;
    s.q = state.imu().q;
    s.p = state.imu().p;
    s.v = state.imu().v;
    s.ba = state.imu().ba;
    s.bg = state.imu().bg;
    for (int i = 0; i < state.num_clones(); ++i)
    {
        s.clone_q.push_back(state.clone(i).q);
        s.clone_p.push_back(state.clone(i).p);
    }
    s.P = state.covariance();
    return s;
}

double snapshot_max_diff(const Snapshot &a, const Snapshot &b)
{
    double d = std::max((a.q - b.q).cwiseAbs().maxCoeff(), (a.p - b.p).cwiseAbs().maxCoeff());
    d = std::max(d, (a.v - b.v).cwiseAbs().maxCoeff());
    d = std::max(d, (a.ba - b.ba).cwiseAbs().maxCoeff());
    d = std::max(d, (a.bg - b.bg).cwiseAbs().maxCoeff());
    for (std::size_t i = 0; i < a.clone_q.size(); ++i)
    {
        d = std::max(d, (a.clone_q[i] - b.clone_q[i]).cwiseAbs().maxCoeff());
        d = std::max(d, (a.clone_p[i] - b.clone_p[i]).cwiseAbs().maxCoeff());
    }
    if (a.P.size() > 0 && b.P.size() > 0) d = std::max(d, (a.P - b.P).cwiseAbs().maxCoeff());
    return d;
}

int clone_theta_offset(const EkfState &state, int i) { return state.clone_offset(i); }

bool state_is_finite(const EkfState &state)
{
    bool finite = state.imu().q.allFinite() && state.imu().p.allFinite() &&
                  state.imu().v.allFinite() && state.imu().ba.allFinite() &&
                  state.imu().bg.allFinite() && state.covariance().allFinite();
    for (int i = 0; i < state.num_clones(); ++i)
    {
        finite = finite && state.clone(i).q.allFinite() && state.clone(i).p.allFinite();
    }
    return finite;
}

double min_covariance_eigenvalue(const Eigen::MatrixXd &P)
{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(P);
    return eig.info() == Eigen::Success ? eig.eigenvalues().minCoeff()
                                        : -std::numeric_limits<double>::infinity();
}

// Reconstructs the error-state correction dx from the nominal state change.
Eigen::VectorXd reconstruct_dx(const Snapshot &snap, const EkfState &after)
{
    const int n = after.size();
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(n);
    dx.segment<3>(kImuTheta) = correction_from_quat(snap.q, after.imu().q);
    dx.segment<3>(kImuP) = after.imu().p - snap.p;
    dx.segment<3>(kImuV) = after.imu().v - snap.v;
    dx.segment<3>(kImuBa) = after.imu().ba - snap.ba;
    dx.segment<3>(kImuBg) = after.imu().bg - snap.bg;
    for (int i = 0; i < after.num_clones(); ++i)
    {
        dx.segment<3>(clone_theta_offset(after, i)) =
            correction_from_quat(snap.clone_q[static_cast<std::size_t>(i)], after.clone(i).q);
        dx.segment<3>(clone_theta_offset(after, i) + 3) =
            after.clone(i).p - snap.clone_p[static_cast<std::size_t>(i)];
    }
    return dx;
}

// Removes the 6-dof global rigid gauge from a per-clone perturbation field
// (gauge rotation dphi: dtheta_i = R_i^T dphi, dp_i = dphi x p_i; gauge
// translation dt: dp_i = dt).  The visual system cannot observe this motion,
// so a purely observable perturbation is used in the convergence test.
void remove_gauge(const std::vector<Eigen::Vector4d> &qs, const std::vector<Eigen::Vector3d> &ps,
                  std::vector<Eigen::Vector3d> &dtheta, std::vector<Eigen::Vector3d> &dp)
{
    const int n = static_cast<int>(qs.size());
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(6 * n, 6);
    Eigen::VectorXd v(6 * n);
    for (int i = 0; i < n; ++i)
    {
        const Eigen::Matrix3d R = EkfState::rotation(qs[static_cast<std::size_t>(i)]);
        const Eigen::Vector3d p = ps[static_cast<std::size_t>(i)];
        for (int k = 0; k < 3; ++k)
        {
            const Eigen::Vector3d e = Eigen::Vector3d::Unit(k);
            G.block<3, 1>(6 * i, k) = R.transpose() * e;
            G.block<3, 1>(6 * i + 3, k) = e.cross(p);
            G.block<3, 1>(6 * i + 3, 3 + k) = e;
        }
        v.segment<3>(6 * i) = dtheta[static_cast<std::size_t>(i)];
        v.segment<3>(6 * i + 3) = dp[static_cast<std::size_t>(i)];
    }
    const Eigen::VectorXd coef = (G.transpose() * G).ldlt().solve(G.transpose() * v);
    v -= G * coef;
    for (int i = 0; i < n; ++i)
    {
        dtheta[static_cast<std::size_t>(i)] = v.segment<3>(6 * i);
        dp[static_cast<std::size_t>(i)] = v.segment<3>(6 * i + 3);
    }
}


// Sets an isotropic, block-diagonal prior on the live pose and every clone.
// After augment_clone() the raw covariance is singular (a fresh clone is a
// copy of the live pose).  A real filter adds process noise between frames;
// this helper models that as independent pose priors, which keeps the
// multi-frame visual-only convergence test well conditioned.
void set_pose_diagonal_covariance(EkfState &state, double rot_sigma_rad,
                                 double pos_sigma_m)
{
    const int n = state.size();
    state.covariance().setZero(n, n);
    state.covariance().block<3, 3>(kImuTheta, kImuTheta) =
        rot_sigma_rad * rot_sigma_rad * Eigen::Matrix3d::Identity();
    state.covariance().block<3, 3>(kImuP, kImuP) =
        pos_sigma_m * pos_sigma_m * Eigen::Matrix3d::Identity();
    state.covariance().block<3, 3>(kImuV, kImuV) =
        0.1 * 0.1 * Eigen::Matrix3d::Identity();
    state.covariance().block<3, 3>(kImuBa, kImuBa) =
        0.1 * 0.1 * Eigen::Matrix3d::Identity();
    state.covariance().block<3, 3>(kImuBg, kImuBg) =
        0.01 * 0.01 * Eigen::Matrix3d::Identity();
    for (int i = 0; i < state.num_clones(); ++i)
    {
        const int off = state.clone_offset(i);
        state.covariance().block<3, 3>(off, off) =
            rot_sigma_rad * rot_sigma_rad * Eigen::Matrix3d::Identity();
        state.covariance().block<3, 3>(off + 3, off + 3) =
            pos_sigma_m * pos_sigma_m * Eigen::Matrix3d::Identity();
    }
}

// Absolute and gauge-invariant pose errors.  A visual-only problem (no IMU)
// cannot observe the six-dof world gauge.  The relative metric anchors clone 0
// at the true trajectory, so it measures exactly what the images constrain.
void pose_errors(const EkfState &state, const std::vector<Eigen::Vector4d> &q_true,
                 const std::vector<Eigen::Vector3d> &p_true, bool relative,
                 double &rms_pos, double &rms_rot)
{
    double sq_pos = 0.0;
    double sq_rot = 0.0;
    const Eigen::Matrix3d R0t = EkfState::rotation(q_true.front());
    const Eigen::Vector3d p0t = p_true.front();
    const Eigen::Matrix3d R0e = EkfState::rotation(state.clone(0).q);
    const Eigen::Vector3d p0e = state.clone(0).p;
    for (int i = 0; i < state.num_clones(); ++i)
    {
        if (relative && i == 0) continue;
        Eigen::Vector3d p_t = p_true[static_cast<std::size_t>(i)];
        Eigen::Vector4d q_t = q_true[static_cast<std::size_t>(i)];
        Eigen::Vector3d p_e = state.clone(i).p;
        Eigen::Vector4d q_e = state.clone(i).q;
        if (relative)
        {
            p_t = R0t.transpose() * (p_t - p0t);
            p_e = R0e.transpose() * (p_e - p0e);
            q_t = EkfState::quaternion_multiply(quat_inverse(q_true.front()), q_t);
            q_e = EkfState::quaternion_multiply(quat_inverse(state.clone(0).q), q_e);
        }
        sq_pos += (p_e - p_t).squaredNorm();
        sq_rot += log_quat(EkfState::quaternion_multiply(quat_inverse(q_t), q_e)).squaredNorm();
    }
    const double count = relative ? std::max(1, state.num_clones() - 1) : state.num_clones();
    rms_pos = std::sqrt(sq_pos / count);
    rms_rot = std::sqrt(sq_rot / count);
}

// ===========================================================================
// T13 -- residual uses the current clone pose, Jacobians use the frozen FEJ.
// ===========================================================================
TEST(SchurUpdate, T13JacobianUsesCloneFejWhileResidualUsesCurrentEstimate)
{
    const UpdateOptions opts = default_options();
    SchurUpdater updater(opts);
    const CameraExtrinsics cam =
        make_camera(Eigen::Vector3d(0.02, -0.01, 0.03), Eigen::Vector3d(0.01, 0.02, -0.01));
    const double sigma = opts.pixel_noise / opts.focal_length;

    std::vector<Eigen::Vector4d> qs;
    std::vector<Eigen::Vector3d> ps;
    smooth_trajectory(3, qs, ps);
    EkfState state;
    build_state(state, qs, ps);
    regularize_covariance(state);

    // Move the current clone estimates away from their frozen FEJ poses.
    for (int i = 0; i < state.num_clones(); ++i)
    {
        state.clone(i).q = EkfState::quaternion_multiply(
            state.clone(i).q, quat(Eigen::Vector3d(0.004 * (i + 1), -0.003, 0.002)));
        state.clone(i).p += Eigen::Vector3d(0.01 * (i + 1), -0.008, 0.006);
    }

    const Eigen::Vector3d world_point(0.8, 0.5, 4.5);
    FeatureTrack track;
    track.id = 1300;
    for (int i = 0; i < 3; ++i)
    {
        track.observations.push_back(
            make_observation(i, 0, project_world(state.clone(i).q, state.clone(i).p, cam,
                                                 world_point)));
    }

    const int ci = 1;
    const FeatureObservation &obs = track.observations[static_cast<std::size_t>(ci)];
    const ObservationJacobian J = updater.observation_jacobian(state, cam, world_point, obs);
    ASSERT_TRUE(J.valid);

    const Eigen::Vector2d expected_residual =
        weighted_residual_model(state.clone(ci).q, state.clone(ci).p, cam, world_point, obs.uv,
                                sigma);
    EXPECT_LT((J.residual - expected_residual).norm(), 1e-12);

    const Eigen::Vector4d q_fej = state.clone_fej(ci).q;
    const Eigen::Vector3d p_fej = state.clone_fej(ci).p;
    const double eps = 1e-6;
    Eigen::Matrix<double, 2, 6> fd_pose = Eigen::Matrix<double, 2, 6>::Zero();
    for (int k = 0; k < 6; ++k)
    {
        Eigen::Vector4d q_p = q_fej, q_m = q_fej;
        Eigen::Vector3d p_p = p_fej, p_m = p_fej;
        if (k < 3)
        {
            Eigen::Vector3d d = Eigen::Vector3d::Zero();
            d[k] = eps;
            q_p = EkfState::quaternion_multiply(q_fej, quat(d)).normalized();
            q_m = EkfState::quaternion_multiply(q_fej, quat(-d)).normalized();
        }
        else
        {
            p_p[k - 3] += eps;
            p_m[k - 3] -= eps;
        }
        const Eigen::Vector2d r_p =
            weighted_residual_model(q_p, p_p, cam, world_point, obs.uv, sigma);
        const Eigen::Vector2d r_m =
            weighted_residual_model(q_m, p_m, cam, world_point, obs.uv, sigma);
        fd_pose.col(k) = (r_p - r_m) / (2.0 * eps);
    }
    EXPECT_LT((fd_pose + J.H_pose).norm() / J.H_pose.norm(), 1e-7);

    std::vector<CloneNominal> frozen_fej(state.num_clones());
    for (int i = 0; i < state.num_clones(); ++i)
        frozen_fej[static_cast<std::size_t>(i)] = state.clone_fej(i);

    std::vector<FeatureTrack> tracks{track};
    const UpdateSummary summary = updater.update(state, tracks, {cam});
    EXPECT_TRUE(summary.applied);
    ASSERT_EQ(summary.consumed_feature_ids.size(), 1u);
    EXPECT_EQ(summary.consumed_feature_ids.front(), track.id);
    for (int i = 0; i < state.num_clones(); ++i)
    {
        EXPECT_LT((state.clone_fej(i).q - frozen_fej[static_cast<std::size_t>(i)].q).norm(),
                  1e-15);
        EXPECT_LT((state.clone_fej(i).p - frozen_fej[static_cast<std::size_t>(i)].p).norm(),
                  1e-15);
    }
}

// ===========================================================================
// T14 -- excessive visual information conditioning rejects the whole frame.
// ===========================================================================
TEST(SchurUpdate, T14ConditionLimitRejectsWithoutTouchingState)
{
    UpdateOptions opts = default_options();
    opts.condition_limit = 1.0;
    SchurUpdater updater(opts);
    const CameraExtrinsics cam =
        make_camera(Eigen::Vector3d(0.01, 0.02, -0.01), Eigen::Vector3d(0.02, 0.0, 0.01));
    std::vector<Eigen::Vector4d> qs;
    std::vector<Eigen::Vector3d> ps;
    smooth_trajectory(4, qs, ps);
    EkfState state;
    build_state(state, qs, ps);
    regularize_covariance(state);

    std::vector<FeatureTrack> tracks;
    for (int f = 0; f < 8; ++f)
    {
        FeatureTrack track;
        track.id = 1400 + f;
        const Eigen::Vector3d world_point(-0.7 + 0.2 * f, 0.15 * (f % 3), 4.0 + 0.1 * f);
        for (int i = 0; i < 4; ++i)
            track.observations.push_back(
                make_observation(i, 0, project_world(qs[static_cast<std::size_t>(i)],
                                                     ps[static_cast<std::size_t>(i)], cam,
                                                     world_point)));
        tracks.push_back(track);
    }

    const Snapshot before = snapshot(state);
    const UpdateSummary summary = updater.update(state, tracks, {cam});
    std::printf("[T14] applied=%d rank=%d cond=%.6e reason=%s\n", summary.applied ? 1 : 0,
                summary.rank, summary.condition, summary.failure_reason.c_str());
    EXPECT_FALSE(summary.applied);
    EXPECT_GT(summary.condition, opts.condition_limit);
    EXPECT_NE(summary.failure_reason.find("condition"), std::string::npos);
    EXPECT_EQ(summary.consumed_feature_ids.size(), tracks.size());
    EXPECT_LT(snapshot_max_diff(before, snapshot(state)), 1e-15);
}

// ===========================================================================
// T15 -- Joseph output must be PSD; otherwise nothing is written back.
// ===========================================================================
TEST(SchurUpdate, T15NonPsdJosephCovarianceRejectsWithoutWritingBack)
{
    const UpdateOptions opts = default_options();
    SchurUpdater updater(opts);
    const CameraExtrinsics cam = make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector4d> qs;
    std::vector<Eigen::Vector3d> ps;
    smooth_trajectory(3, qs, ps);
    EkfState state;
    build_state(state, qs, ps);
    set_pose_diagonal_covariance(state, 0.01, 0.1);

    // An indefinite mode in the live state is not touched by the visual L
    // matrix, so the Joseph result stays indefinite in that direction.
    state.covariance()(kImuP, kImuP) = -1.0;

    const Eigen::Vector3d world_point(0.6, 0.3, 4.2);
    FeatureTrack track;
    track.id = 1500;
    for (int i = 0; i < 3; ++i)
        track.observations.push_back(
            make_observation(i, 0, project_world(qs[static_cast<std::size_t>(i)],
                                                 ps[static_cast<std::size_t>(i)], cam,
                                                 world_point)));

    const Snapshot before = snapshot(state);
    std::vector<FeatureTrack> tracks{track};
    const UpdateSummary summary = updater.update(state, tracks, {cam});
    std::printf("[T15] applied=%d reason=%s\n", summary.applied ? 1 : 0,
                summary.failure_reason.c_str());
    EXPECT_FALSE(summary.applied);
    EXPECT_NE(summary.failure_reason.find("PSD"), std::string::npos);
    EXPECT_LT(snapshot_max_diff(before, snapshot(state)), 1e-15);
}

// ===========================================================================
// T16 -- consumed_feature_ids covers accepted, rejected and capped tracks.
// ===========================================================================
TEST(SchurUpdate, T16ConsumedFeatureIdsAreComplete)
{
    UpdateOptions opts = default_options();
    opts.max_features = 1;
    opts.min_features = 2;  // force a visually starved frame, but still consume all ids
    SchurUpdater updater(opts);
    const CameraExtrinsics cam = make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector4d> qs;
    std::vector<Eigen::Vector3d> ps;
    smooth_trajectory(3, qs, ps);
    EkfState state;
    build_state(state, qs, ps);
    regularize_covariance(state);

    auto make_track = [&](int id, const Eigen::Vector3d &world_point) {
        FeatureTrack track;
        track.id = id;
        for (int i = 0; i < 3; ++i)
            track.observations.push_back(
                make_observation(i, 0, project_world(qs[static_cast<std::size_t>(i)],
                                                     ps[static_cast<std::size_t>(i)], cam,
                                                     world_point)));
        return track;
    };

    std::vector<FeatureTrack> tracks;
    tracks.push_back(make_track(1601, Eigen::Vector3d(0.4, 0.3, 4.0)));
    tracks.push_back(make_track(1602, Eigen::Vector3d(-0.3, 0.5, 4.5)));
    FeatureTrack outlier = make_track(1603, Eigen::Vector3d(0.7, -0.2, 3.8));
    outlier.observations[1].uv.x() += 50.0 / kFocal;
    tracks.push_back(outlier);
    FeatureTrack short_track;
    short_track.id = 1604;
    short_track.observations.push_back(make_observation(0, 0, Eigen::Vector2d(0.1, 0.2)));
    tracks.push_back(short_track);

    const UpdateSummary summary = updater.update(state, tracks, {cam});
    std::vector<int> expected{1601, 1602, 1603, 1604};
    std::vector<int> consumed = summary.consumed_feature_ids;
    std::sort(consumed.begin(), consumed.end());
    EXPECT_EQ(consumed, expected);
    EXPECT_FALSE(summary.applied);
    EXPECT_NE(summary.failure_reason.find("insufficient"), std::string::npos);
    // used_feature_ids tracks A/b accumulation, not final acceptance: this
    // frame accumulated one track and was then rejected by the starvation gate.
    ASSERT_EQ(summary.used_feature_ids.size(), 1u);
    EXPECT_EQ(summary.used_feature_ids.front(), 1601);
}


// ===========================================================================
// T17 -- only features lost in the newest frame are selected as lost.
// ===========================================================================
TEST(SchurUpdate, T17FeatureHistorySelectsLostOnly)
{
    FeatureHistory history;
    const Eigen::Vector2d uv(0.1, -0.05);
    history.add_frame(10, 0, {{171, uv}, {172, uv}});
    history.add_frame(11, 0, {{172, uv}});

    const FeatureHistory::TrackSelection selection =
        history.select_tracks(9, 11, 1, 10);
    ASSERT_EQ(selection.tracks.size(), 1u);
    EXPECT_EQ(selection.tracks.front().id, 171);
    EXPECT_EQ(selection.lost_count, 1);
    EXPECT_EQ(selection.marginalized_count, 0);
}

// ===========================================================================
// T18 -- a still-tracked feature is selected when it constrains the oldest
// clone that is about to be marginalized.
// ===========================================================================
TEST(SchurUpdate, T18FeatureHistorySelectsMarginalized)
{
    FeatureHistory history;
    const Eigen::Vector2d uv(0.2, 0.1);
    history.add_frame(10, 0, {{181, uv}});
    history.add_frame(11, 0, {{182, uv}});
    history.add_frame(12, 0, {{181, uv}, {182, uv}});

    const FeatureHistory::TrackSelection selection =
        history.select_tracks(10, 12, 2, 10);
    ASSERT_EQ(selection.tracks.size(), 1u);
    EXPECT_EQ(selection.tracks.front().id, 181);
    EXPECT_EQ(selection.lost_count, 0);
    EXPECT_EQ(selection.marginalized_count, 1);
}

// ===========================================================================
// T19 -- trim is clone-window based only; it must not delete a track merely
// because it has been unobserved for more than one frame.
// ===========================================================================
TEST(SchurUpdate, T19FeatureHistoryTrimDoesNotDropTwoFrameOldTracks)
{
    FeatureHistory history;
    const Eigen::Vector2d uv(0.3, -0.2);
    history.add_frame(0, 0, {{191, uv}, {192, uv}});
    history.add_frame(2, 0, {{191, uv}});
    history.add_frame(4, 0, {});  // newest frame: feature 191 is now lost

    history.trim(1, 4);
    ASSERT_EQ(history.num_features(), 1u);
    EXPECT_EQ(history.newest_frame_index(), 4);

    const FeatureHistory::TrackSelection selection =
        history.select_tracks(1, 4, 1, 10);
    ASSERT_EQ(selection.tracks.size(), 1u);
    EXPECT_EQ(selection.tracks.front().id, 191);
    ASSERT_EQ(selection.tracks.front().observations.size(), 1u);
    EXPECT_EQ(selection.tracks.front().observations.front().clone_index, 1);
    EXPECT_EQ(selection.lost_count, 1);
}

// ===========================================================================
// T20 -- once a lost track is consumed and erased, a later frame cannot select
// it as lost again.
// ===========================================================================
TEST(SchurUpdate, T20FeatureHistoryLostTrackIsNotSelectedAgainAfterErase)
{
    FeatureHistory history;
    const Eigen::Vector2d uv(0.05, 0.15);
    history.add_frame(20, 0, {{201, uv}});
    history.add_frame(21, 0, {});

    const FeatureHistory::TrackSelection first =
        history.select_tracks(19, 21, 1, 10);
    ASSERT_EQ(first.tracks.size(), 1u);
    EXPECT_EQ(first.tracks.front().id, 201);
    EXPECT_EQ(first.lost_count, 1);

    std::vector<int> used_ids;
    used_ids.push_back(first.tracks.front().id);
    history.erase_features(used_ids);
    history.add_frame(22, 0, {});

    const FeatureHistory::TrackSelection second =
        history.select_tracks(19, 22, 1, 10);
    EXPECT_TRUE(second.tracks.empty());
    EXPECT_EQ(second.lost_count, 0);
    EXPECT_EQ(second.marginalized_count, 0);
}

// ===========================================================================
// T21 -- used_feature_ids contains exactly the tracks accumulated into A/b,
// never a triangulation/chi2/gating rejection.
// ===========================================================================
TEST(SchurUpdate, T21UsedFeatureIdsDisjointFromRejected)
{
    UpdateOptions opts = default_options();
    opts.max_features = 10;
    opts.min_features = 1;
    opts.min_update_observations = 1;
    SchurUpdater updater(opts);
    const CameraExtrinsics cam =
        make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector4d> qs;
    std::vector<Eigen::Vector3d> ps;
    smooth_trajectory(3, qs, ps);
    EkfState state;
    build_state(state, qs, ps);
    regularize_covariance(state);

    auto make_track = [&](int id, const Eigen::Vector3d &world_point, double outlier_px) {
        FeatureTrack track;
        track.id = id;
        for (int i = 0; i < 3; ++i)
        {
            Eigen::Vector2d uv = project_world(qs[static_cast<std::size_t>(i)],
                                               ps[static_cast<std::size_t>(i)], cam,
                                               world_point);
            if (i == 1) uv.x() += outlier_px / kFocal;
            track.observations.push_back(make_observation(i, 0, uv));
        }
        return track;
    };

    const FeatureTrack outlier =
        make_track(2102, Eigen::Vector3d(0.7, -0.3, 4.0), 50.0);
    const FeatureTrack good =
        make_track(2101, Eigen::Vector3d(-0.2, 0.4, 4.2), 0.0);
    FeatureTrack short_track;
    short_track.id = 2103;
    short_track.observations.push_back(
        make_observation(0, 0, Eigen::Vector2d(0.1, 0.2)));

    std::vector<FeatureTrack> tracks{outlier, good, short_track};
    const UpdateSummary summary = updater.update(state, tracks, {cam});
    EXPECT_TRUE(summary.applied);
    EXPECT_EQ(summary.features_used, 1);
    ASSERT_EQ(summary.used_feature_ids.size(), 1u);
    EXPECT_EQ(summary.used_feature_ids.front(), good.id);
    EXPECT_EQ(summary.consumed_feature_ids.size(), tracks.size());

    EXPECT_NE(std::find(summary.rejected_feature_ids.begin(),
                        summary.rejected_feature_ids.end(), outlier.id),
              summary.rejected_feature_ids.end());
    EXPECT_NE(std::find(summary.rejected_feature_ids.begin(),
                        summary.rejected_feature_ids.end(), short_track.id),
              summary.rejected_feature_ids.end());
    for (const int used_id : summary.used_feature_ids)
    {
        EXPECT_EQ(std::find(summary.rejected_feature_ids.begin(),
                            summary.rejected_feature_ids.end(), used_id),
                  summary.rejected_feature_ids.end());
    }
}


// ===========================================================================
// T22 -- marginalization-eligible tracks are selected only after the clone
// window is full. While filling, only tracks lost in the newest frame qualify.
// ===========================================================================
TEST(SchurUpdate, T22WindowFullControlsMarginalizedSelection)
{
    FeatureHistory history;
    const Eigen::Vector2d uv(0.12, -0.08);
    history.add_frame(10, 0, {{221, uv}});
    history.add_frame(11, 0, {{222, uv}});
    history.add_frame(12, 0, {{221, uv}});

    const FeatureHistory::TrackSelection filling =
        history.tracks(10, 12, 1, 10, false);
    ASSERT_EQ(filling.tracks.size(), 1u);
    EXPECT_EQ(filling.tracks.front().id, 222);
    EXPECT_EQ(filling.lost_count, 1);
    EXPECT_EQ(filling.marginalized_count, 0);

    const FeatureHistory::TrackSelection full =
        history.tracks(10, 12, 1, 10, true);
    ASSERT_EQ(full.tracks.size(), 2u);
    EXPECT_NE(std::find_if(full.tracks.begin(), full.tracks.end(),
                           [](const FeatureTrack &track) { return track.id == 221; }),
              full.tracks.end());
    EXPECT_NE(std::find_if(full.tracks.begin(), full.tracks.end(),
                           [](const FeatureTrack &track) { return track.id == 222; }),
              full.tracks.end());
    EXPECT_EQ(full.lost_count, 1);
    EXPECT_EQ(full.marginalized_count, 1);
}

}  // namespace

// ===========================================================================
// T1 -- analytic Jacobians against central finite differences.
// ===========================================================================
TEST(SchurUpdate, T1ObservationJacobianMatchesFiniteDifferences)
{
    const UpdateOptions opts = default_options();
    const SchurUpdater updater(opts);
    const double sigma = opts.pixel_noise / opts.focal_length;

    std::mt19937 rng(20261025);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);

    const Eigen::Vector4d q_i = random_quat(rng, 0.8);
    const Eigen::Vector3d p_i(unit(rng), unit(rng), unit(rng));
    const CameraExtrinsics cam = make_camera(Eigen::Vector3d(0.05, -0.1, 0.03),
                                             Eigen::Vector3d(0.02, -0.01, 0.05));
    // Point ~5 m in front of the camera, off-centre.
    const Eigen::Vector3d uv_c(0.25 * unit(rng), 0.25 * unit(rng), 1.0);
    const Eigen::Vector3d p_c = 5.0 * uv_c;
    const Eigen::Vector3d world_point =
        p_i + EkfState::rotation(q_i) * (EkfState::rotation(cam.q_cb) * p_c + cam.p_cb);

    std::vector<Eigen::Vector4d> qs{q_i};
    std::vector<Eigen::Vector3d> ps{p_i};
    EkfState state;
    build_state(state, qs, ps);
    state.clone(0).q = q_i;
    state.clone(0).p = p_i;

    const Eigen::Vector2d uv = project_world(q_i, p_i, cam, world_point);
    const FeatureObservation obs = make_observation(0, 0, uv);

    const ObservationJacobian J = updater.observation_jacobian(state, cam, world_point, obs);
    ASSERT_TRUE(J.valid);
    EXPECT_LT(std::abs(J.p_c.z() - p_c.z()), 1e-12);

    const double eps = 1e-6;
    Eigen::Matrix<double, 2, 6> fd_pose = Eigen::Matrix<double, 2, 6>::Zero();
    Eigen::Matrix<double, 2, 3> fd_point = Eigen::Matrix<double, 2, 3>::Zero();
    for (int k = 0; k < 6; ++k)
    {
        Eigen::Vector4d q_p = q_i, q_m = q_i;
        Eigen::Vector3d p_p = p_i, p_m = p_i;
        if (k < 3)
        {
            // Right (body) perturbation: q' = q (x) dq.
            Eigen::Vector3d d = Eigen::Vector3d::Zero();
            d[k] = eps;
            q_p = EkfState::quaternion_multiply(q_i, quat(d)).normalized();
            q_m = EkfState::quaternion_multiply(q_i, quat(-d)).normalized();
        }
        else
        {
            p_p[k - 3] += eps;
            p_m[k - 3] -= eps;
        }
        const Eigen::Vector2d r_p = weighted_residual_model(q_p, p_p, cam, world_point, uv, sigma);
        const Eigen::Vector2d r_m = weighted_residual_model(q_m, p_m, cam, world_point, uv, sigma);
        fd_pose.col(k) = (r_p - r_m) / (2.0 * eps);
    }
    for (int k = 0; k < 3; ++k)
    {
        Eigen::Vector3d wp = world_point, wm = world_point;
        wp[k] += eps;
        wm[k] -= eps;
        const Eigen::Vector2d r_p = weighted_residual_model(q_i, p_i, cam, wp, uv, sigma);
        const Eigen::Vector2d r_m = weighted_residual_model(q_i, p_i, cam, wm, uv, sigma);
        fd_point.col(k) = (r_p - r_m) / (2.0 * eps);
    }

    // residual = (uv - h)/sigma  =>  d(residual)/dx = -H.
    const double pose_err = (fd_pose + J.H_pose).norm() / J.H_pose.norm();
    const double point_err = (fd_point + J.H_point).norm() / J.H_point.norm();
    std::printf("[T1] rel err pose=%.3e point=%.3e\n", pose_err, point_err);
    EXPECT_LT(pose_err, 1e-7);
    EXPECT_LT(point_err, 1e-7);
}

// ===========================================================================
// T2 -- triangulation of a known point, noise free and with 0.5 px noise.
// ===========================================================================
TEST(SchurUpdate, T2TriangulationRecoversKnownPoint)
{
    const UpdateOptions opts = default_options();
    const SchurUpdater updater(opts);
    const CameraExtrinsics cam = make_camera(Eigen::Vector3d(0.02, -0.03, 0.01),
                                             Eigen::Vector3d(0.01, 0.0, 0.02));
    std::vector<Eigen::Vector4d> qs;
    std::vector<Eigen::Vector3d> ps;
    smooth_trajectory(5, qs, ps);
    EkfState state;
    build_state(state, qs, ps);

    const Eigen::Vector3d world_point(1.0, 0.8, 4.5);

    // ---- noise free -----------------------------------------------------
    FeatureTrack track;
    track.id = 1;
    for (int i = 0; i < 5; ++i)
        track.observations.push_back(
            make_observation(i, 0, project_world(qs[static_cast<std::size_t>(i)],
                                                 ps[static_cast<std::size_t>(i)], cam, world_point)));
    ASSERT_TRUE(updater.triangulate(state, {cam}, track));
    const double err_clean = (track.world_point - world_point).norm();
    std::printf("[T2] noise-free point error = %.3e m\n", err_clean);
    EXPECT_LT(err_clean, 1e-6);

    // ---- 0.5 px noise ---------------------------------------------------
    std::mt19937 rng(11);
    std::normal_distribution<double> noise(0.0, 0.5 / kFocal);
    double worst = 0.0;
    for (int trial = 0; trial < 20; ++trial)
    {
        FeatureTrack noisy;
        noisy.id = 2;
        for (int i = 0; i < 5; ++i)
        {
            Eigen::Vector2d uv = project_world(qs[static_cast<std::size_t>(i)],
                                               ps[static_cast<std::size_t>(i)], cam, world_point);
            uv += Eigen::Vector2d(noise(rng), noise(rng));
            noisy.observations.push_back(make_observation(i, 0, uv));
        }
        ASSERT_TRUE(updater.triangulate(state, {cam}, noisy));
        worst = std::max(worst, (noisy.world_point - world_point).norm());
    }
    std::printf("[T2] worst 0.5 px noise error = %.4f m (20 trials)\n", worst);
    EXPECT_LT(worst, 0.05);
}

// ===========================================================================
// T3 -- the reduce_track blocks reproduce the exact Schur complement.
// ===========================================================================
TEST(SchurUpdate, T3SchurReductionMatchesJointSchurComplement)
{
    const UpdateOptions opts = default_options();
    const SchurUpdater updater(opts);
    const CameraExtrinsics cam = make_camera(Eigen::Vector3d(0.01, 0.02, -0.01),
                                             Eigen::Vector3d(0.03, -0.02, 0.01));
    std::vector<Eigen::Vector4d> qs;
    std::vector<Eigen::Vector3d> ps;
    smooth_trajectory(3, qs, ps);
    EkfState state;
    build_state(state, qs, ps);

    const Eigen::Vector3d world_point(0.6, 1.1, 4.0);
    FeatureTrack track;
    track.id = 3;
    for (int i = 0; i < 3; ++i)
        track.observations.push_back(
            make_observation(i, 0, project_world(qs[static_cast<std::size_t>(i)],
                                                 ps[static_cast<std::size_t>(i)], cam, world_point)));
    ASSERT_TRUE(updater.triangulate(state, {cam}, track));

    const TrackReduction red = updater.reduce_track(state, {cam}, track);
    ASSERT_TRUE(red.valid);
    const JointSystem joint = build_joint(updater, state, {cam}, track);

    // reduce_track's W must equal H_x^T H_f restricted to the clone blocks.
    Eigen::MatrixXd A_xf = Eigen::MatrixXd::Zero(state.size(), 3);
    for (const auto &w : red.W) A_xf.block<6, 3>(state.clone_offset(w.first), 0) = w.second;
    const double W_err = (A_xf - joint.Hx.transpose() * joint.Hf).norm();
    const double W_scale = (joint.Hx.transpose() * joint.Hf).norm();
    std::printf("[T3] W rel err = %.3e, A_point rel err = %.3e\n", W_err / W_scale,
                (red.A_point - joint.Hf.transpose() * joint.Hf).norm() /
                    (joint.Hf.transpose() * joint.Hf).norm());
    EXPECT_LT(W_err, 1e-9 * std::max(1.0, W_scale));
    EXPECT_LT((red.A_point - joint.Hf.transpose() * joint.Hf).norm(),
              1e-9 * std::max(1.0, (joint.Hf.transpose() * joint.Hf).norm()));
    EXPECT_LT((red.b_point - joint.Hf.transpose() * joint.r).norm(),
              1e-9 * std::max(1.0, (joint.Hf.transpose() * joint.r).norm()));

    Eigen::MatrixXd A_red, A_ref;
    Eigen::VectorXd b_red, b_ref;
    reduce_with_blocks(state, joint, red, A_red, b_red);
    reduce_exact(joint, A_ref, b_ref);

    const double a_err = (A_red - A_ref).norm() / A_ref.norm();
    const double b_err = (b_red - b_ref).norm() / b_ref.norm();
    std::printf("[T3] Schur A rel err = %.3e, b rel err = %.3e\n", a_err, b_err);
    EXPECT_LT(a_err, 1e-9);
    EXPECT_LT(b_err, 1e-9);
}

// ===========================================================================
// T4 -- one update() call reproduces the information-form solution.
// ===========================================================================
TEST(SchurUpdate, T4EkfUpdateMatchesInformationForm)
{
    const UpdateOptions opts = default_options();
    SchurUpdater updater(opts);
    const CameraExtrinsics cam = make_camera(Eigen::Vector3d(0.0, 0.0, 0.0),
                                             Eigen::Vector3d(0.02, 0.0, 0.0));

    // Truth poses, perturbed nominal poses (the measurement residual is then
    // non-trivial).
    std::vector<Eigen::Vector4d> q_true, q_nom;
    std::vector<Eigen::Vector3d> p_true, p_nom;
    smooth_trajectory(3, q_true, p_true);
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (int i = 0; i < 3; ++i)
    {
        q_nom.push_back(EkfState::quaternion_multiply(q_true[static_cast<std::size_t>(i)],
                                                      quat(Eigen::Vector3d(0.004 * d(rng), 0.004 * d(rng), 0.004 * d(rng)))));
        p_nom.push_back(p_true[static_cast<std::size_t>(i)] + Eigen::Vector3d(0.02 * d(rng), 0.02 * d(rng), 0.02 * d(rng)));
    }
    EkfState state;
    build_state(state, q_nom, p_nom);
    regularize_covariance(state);  // fresh clones make the raw P singular

    const Eigen::Vector3d world_point(0.9, 0.7, 4.2);
    FeatureTrack track;
    track.id = 4;
    for (int i = 0; i < 3; ++i)
        track.observations.push_back(
            make_observation(i, 0, project_world(q_true[static_cast<std::size_t>(i)],
                                                 p_true[static_cast<std::size_t>(i)], cam, world_point)));
    ASSERT_TRUE(updater.triangulate(state, {cam}, track));

    const TrackReduction red = updater.reduce_track(state, {cam}, track);
    ASSERT_TRUE(red.valid);
    const JointSystem joint = build_joint(updater, state, {cam}, track);
    Eigen::MatrixXd A_red;
    Eigen::VectorXd b_red;
    reduce_with_blocks(state, joint, red, A_red, b_red);

    const Snapshot before = snapshot(state);
    const Eigen::MatrixXd P_before = state.covariance();
    const int n = state.size();

    std::vector<FeatureTrack> tracks{track};
    const UpdateSummary summary = updater.update(state, tracks, {cam});
    EXPECT_EQ(summary.features_used, 1);
    EXPECT_EQ(summary.observations_used, 3);
    EXPECT_EQ(summary.degrees_of_freedom, 6);

    const Eigen::VectorXd dx = reconstruct_dx(before, state);
    const Eigen::MatrixXd P_inv = P_before.ldlt().solve(Eigen::MatrixXd::Identity(n, n));
    const Eigen::VectorXd lhs = (P_inv + A_red) * dx;
    const double info_err = (lhs - b_red).norm() / b_red.norm();
    std::printf("[T4] information-form rel err = %.3e (|dx| = %.3e, |b| = %.3e)\n", info_err,
                dx.norm(), b_red.norm());
    EXPECT_LT(info_err, 1e-8);

    // Joint least squares of the same system including the prior (state part).
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n + 3, n + 3);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n + 3);
    M.topLeftCorner(n, n) = joint.Hx.transpose() * joint.Hx + P_inv;
    M.topRightCorner(n, 3) = joint.Hx.transpose() * joint.Hf;
    M.bottomLeftCorner(3, n) = joint.Hf.transpose() * joint.Hx;
    M.bottomRightCorner(3, 3) = joint.Hf.transpose() * joint.Hf;
    rhs.head(n) = joint.Hx.transpose() * joint.r;
    rhs.tail(3) = joint.Hf.transpose() * joint.r;
    const Eigen::VectorXd joint_sol = M.ldlt().solve(rhs);
    const double joint_err = (joint_sol.head(n) - dx).norm() / dx.norm();
    std::printf("[T4] joint LS state-part rel err = %.3e\n", joint_err);
    EXPECT_LT(joint_err, 1e-8);
}

// ===========================================================================
// T5 -- covariance stays symmetric/PD and the observed blocks shrink.
// ===========================================================================
TEST(SchurUpdate, T5CovarianceSymmetricPositiveDefiniteAndShrinks)
{
    const UpdateOptions opts = default_options();
    SchurUpdater updater(opts);
    const CameraExtrinsics cam = make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d(0.02, 0.0, 0.0));
    std::vector<Eigen::Vector4d> qs;
    std::vector<Eigen::Vector3d> ps;
    smooth_trajectory(4, qs, ps);
    EkfState state;
    build_state(state, qs, ps);
    regularize_covariance(state);

    std::vector<FeatureTrack> tracks;
    for (int f = 0; f < 6; ++f)
    {
        FeatureTrack track;
        track.id = 100 + f;
        const Eigen::Vector3d world_point(-1.0 + 0.4 * f, 0.5 * ((f % 3) - 1), 4.0 + 0.2 * f);
        for (int i = 0; i < 4; ++i)
            track.observations.push_back(
                make_observation(i, 0, project_world(qs[static_cast<std::size_t>(i)],
                                                     ps[static_cast<std::size_t>(i)], cam, world_point)));
        tracks.push_back(track);
    }

    const Snapshot before = snapshot(state);
    const UpdateSummary summary = updater.update(state, tracks, {cam});
    EXPECT_EQ(summary.features_used, 6);

    const Eigen::MatrixXd P = state.covariance();
    EXPECT_LT((P - P.transpose()).norm(), 1e-12 * std::max(1.0, P.norm()));
    Eigen::LDLT<Eigen::MatrixXd> ldlt(P);
    EXPECT_EQ(ldlt.info(), Eigen::Success);
    EXPECT_GT(ldlt.vectorD().minCoeff(), 1e-12);
    EXPECT_TRUE(P.allFinite());

    for (int i = 0; i < state.num_clones(); ++i)
    {
        const int off = state.clone_offset(i);
        const double tr_before = before.P.block<6, 6>(off, off).trace();
        const double tr_after = P.block<6, 6>(off, off).trace();
        std::printf("[T5] clone %d trace %.6e -> %.6e\n", i, tr_before, tr_after);
        EXPECT_LT(tr_after, tr_before);
    }
}

// ===========================================================================
// T6 -- outlier gate: a 50 px offset track is rejected, state unchanged.
// ===========================================================================
TEST(SchurUpdate, T6Chi2GateRejectsOutlierWithoutTouchingState)
{
    const UpdateOptions opts = default_options();
    SchurUpdater updater(opts);
    const CameraExtrinsics cam = make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector4d> qs;
    std::vector<Eigen::Vector3d> ps;
    smooth_trajectory(3, qs, ps);
    EkfState state;
    build_state(state, qs, ps);
    regularize_covariance(state);

    const Eigen::Vector3d world_point(0.4, 0.9, 4.0);
    const double offset_px = 50.0;

    // (a) three observations, the second one shifted by 50 px.
    FeatureTrack track;
    track.id = 77;
    for (int i = 0; i < 3; ++i)
    {
        Eigen::Vector2d uv = project_world(qs[static_cast<std::size_t>(i)],
                                           ps[static_cast<std::size_t>(i)], cam, world_point);
        if (i == 1) uv.x() += offset_px / kFocal;
        track.observations.push_back(make_observation(i, 0, uv));
    }

    {
        const Snapshot before = snapshot(state);
        std::vector<FeatureTrack> tracks{track};
        const UpdateSummary summary = updater.update(state, tracks, {cam});
        std::printf("[T6] (3 obs, 1 x 50 px) used=%d rejected=%d ids=%zu\n", summary.features_used,
                    summary.features_rejected, summary.rejected_feature_ids.size());
        EXPECT_EQ(summary.features_used, 0);
        EXPECT_EQ(summary.features_rejected, 1);
        ASSERT_EQ(summary.rejected_feature_ids.size(), 1u);
        EXPECT_EQ(summary.rejected_feature_ids.front(), 77);
        EXPECT_LT(snapshot_max_diff(before, snapshot(state)), 1e-12);
    }

    // (b) a single-observation track (below min_observations) is also dropped.
    {
        FeatureTrack single;
        single.id = 78;
        Eigen::Vector2d uv = project_world(qs[0], ps[0], cam, world_point);
        uv.x() += offset_px / kFocal;
        single.observations.push_back(make_observation(0, 0, uv));
        const Snapshot before = snapshot(state);
        std::vector<FeatureTrack> tracks{single};
        const UpdateSummary summary = updater.update(state, tracks, {cam});
        EXPECT_EQ(summary.features_used, 0);
        EXPECT_EQ(summary.features_rejected, 1);
        EXPECT_LT(snapshot_max_diff(before, snapshot(state)), 1e-12);
    }

    // (c) the chi2 statistic itself: with a confident (small) covariance the
    //     clone poses cannot explain a 50 px residual, so the gate trips.
    {
        EkfState confident;
        build_state(confident, qs, ps);
        confident.covariance() *= 1e-6;        // shrink every block consistently
        regularize_covariance(confident, 1e-6);  // ... but keep P PD
        FeatureTrack gate_track;
        gate_track.id = 79;
        for (int i = 0; i < 3; ++i)
            gate_track.observations.push_back(
                make_observation(i, 0, project_world(qs[static_cast<std::size_t>(i)],
                                                     ps[static_cast<std::size_t>(i)], cam,
                                                     world_point)));
        // Triangulate clean data first, then corrupt one *measurement*: this is
        // the situation the chi2 gate sees at update time.
        ASSERT_TRUE(updater.triangulate(confident, {cam}, gate_track));
        gate_track.observations[1].uv.x() += offset_px / kFocal;
        const TrackReduction red = updater.reduce_track(confident, {cam}, gate_track);
        const double gate = opts.chi2_multiplier * SchurUpdater::chi2_quantile_95(6);
        std::printf("[T6] (gate) chi2=%.3f gate=%.3f rms=%.3f px\n", red.chi2, gate,
                    red.rms_residual_px);
        EXPECT_GT(red.chi2, gate);
    }
}

// ===========================================================================
// T7 -- convergence smoke test on a synthetic static scene.
// ===========================================================================
TEST(SchurUpdate, T7ConvergesOnSyntheticStaticScene)
{
    UpdateOptions opts = default_options();
    opts.triangulation_iterations = 20;
    SchurUpdater updater(opts);
    const CameraExtrinsics cam = make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    const int num_frames = 10;
    std::vector<Eigen::Vector4d> q_true;
    std::vector<Eigen::Vector3d> p_true;
    smooth_trajectory(num_frames, q_true, p_true);

    // 20 static features spread in front of the trajectory.
    std::vector<Eigen::Vector3d> features;
    for (int i = 0; i < 20; ++i)
    {
        features.push_back(Eigen::Vector3d(-1.5 + 0.3 * (i % 5) + 0.7,
                                           0.4 * ((i / 5) - 2) + 0.9,
                                           4.0 + 0.3 * (i % 3)));
    }

    // Small, smooth, gauge-free perturbations.  With 1.5 px measurement noise
    // and the fixed 5 px triangulation gate, a 0.5 deg / 5 cm per-clone error at
    // 4 m is already ~6 px of inconsistent reprojection; 0.1 deg / 1 cm keeps
    // the synthetic measurements inside the configured noise model while still
    // making the filter start clearly away from the truth.
    std::vector<Eigen::Vector3d> dtheta, dp;
    for (int i = 0; i < num_frames; ++i)
    {
        const double t = static_cast<double>(i) / static_cast<double>(num_frames - 1);
        Eigen::Vector3d rot_dir(std::sin(2.0 * M_PI * t),
                                0.5 * std::sin(4.0 * M_PI * t + 0.3),
                                0.3 * std::cos(2.0 * M_PI * t));
        Eigen::Vector3d pos_dir(std::sin(2.0 * M_PI * t + 0.7),
                                0.4 * std::cos(4.0 * M_PI * t),
                                0.2 * std::sin(6.0 * M_PI * t));
        dtheta.push_back(rot_dir.normalized() * (0.1 * M_PI / 180.0));
        dp.push_back(pos_dir.normalized() * 0.01);
    }
    remove_gauge(q_true, p_true, dtheta, dp);

    EkfOptions ekf_options = default_ekf_options(num_frames + 2);
    EkfState state;
    state.reset(ekf_options, 0.0, q_true[0], p_true[0], Eigen::Vector3d::Zero(),
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    // All ten frames are pre-augmented as the window; the ten update() calls
    // below then represent one visual update per frame.
    for (int frame = 0; frame < num_frames; ++frame)
    {
        state.imu().q = EkfState::quaternion_multiply(q_true[static_cast<std::size_t>(frame)],
                                                      quat(dtheta[static_cast<std::size_t>(frame)]))
                            .normalized();
        state.imu().p = p_true[static_cast<std::size_t>(frame)] + dp[static_cast<std::size_t>(frame)];
        // We set the nominal pose directly instead of calling propagate().
        state.refresh_imu_fej();
        state.augment_clone(static_cast<double>(frame));
    }
    set_pose_diagonal_covariance(state, 2.0 * M_PI / 180.0, 0.2);

    // Every static feature is observed by every clone in the window.
    std::vector<FeatureTrack> tracks;
    for (std::size_t f = 0; f < features.size(); ++f)
    {
        FeatureTrack track;
        track.id = static_cast<int>(f);
        for (int i = 0; i < num_frames; ++i)
        {
            track.observations.push_back(
                make_observation(i, 0, project_world(q_true[static_cast<std::size_t>(i)],
                                                     p_true[static_cast<std::size_t>(i)], cam,
                                                     features[f])));
        }
        tracks.push_back(track);
    }

    double before_pos = 0.0;
    double before_rot = 0.0;
    double before_rel_pos = 0.0;
    double before_rel_rot = 0.0;
    pose_errors(state, q_true, p_true, false, before_pos, before_rot);
    pose_errors(state, q_true, p_true, true, before_rel_pos, before_rel_rot);

    for (int frame = 0; frame < num_frames; ++frame)
    {
        const UpdateSummary summary = updater.update(state, tracks, {cam});
        EXPECT_EQ(summary.features_used, static_cast<int>(features.size()));
    }

    double after_pos = 0.0;
    double after_rot = 0.0;
    double after_rel_pos = 0.0;
    double after_rel_rot = 0.0;
    pose_errors(state, q_true, p_true, false, after_pos, after_rot);
    pose_errors(state, q_true, p_true, true, after_rel_pos, after_rel_rot);

    std::printf("[T7] absolute pos %.4f -> %.4f cm (x%.1f), rot %.4f -> %.4f deg (x%.1f)\n",
                before_pos * 100.0, after_pos * 100.0, before_pos / after_pos,
                before_rot * 180.0 / M_PI, after_rot * 180.0 / M_PI, before_rot / after_rot);
    std::printf("[T7] gauge-invariant pos %.4f -> %.4f cm (x%.1f), rot %.4f -> %.4f deg (x%.1f)\n",
                before_rel_pos * 100.0, after_rel_pos * 100.0, before_rel_pos / after_rel_pos,
                before_rel_rot * 180.0 / M_PI, after_rel_rot * 180.0 / M_PI,
                before_rel_rot / after_rel_rot);

    EXPECT_LT(before_pos, 0.02);
    EXPECT_LT(before_rot, 0.0021);
    EXPECT_GT(before_rel_pos / after_rel_pos, 3.0);
    EXPECT_GT(before_rel_rot / after_rel_rot, 3.0);
}

// ===========================================================================
// T8 -- robustness: degenerate inputs must not crash or produce NaN.
// ===========================================================================
TEST(SchurUpdate, T8DegenerateInputsAreHandled)
{
    const UpdateOptions opts = default_options();
    SchurUpdater updater(opts);
    const CameraExtrinsics cam = make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector4d> qs;
    std::vector<Eigen::Vector3d> ps;
    smooth_trajectory(3, qs, ps);
    EkfState state;
    build_state(state, qs, ps);
    regularize_covariance(state);

    const Eigen::Vector3d world_point(0.5, 0.5, 4.0);
    const Eigen::Vector2d uv0 = project_world(qs[0], ps[0], cam, world_point);

    std::vector<FeatureTrack> tracks;

    // (a) empty track list
    {
        const Snapshot before = snapshot(state);
        const UpdateSummary summary = updater.update(state, tracks, {cam});
        EXPECT_EQ(summary.features_used, 0);
        EXPECT_LT(snapshot_max_diff(before, snapshot(state)), 1e-12);
    }

    // (b) single observation
    FeatureTrack single;
    single.id = 1;
    single.observations.push_back(make_observation(0, 0, uv0));
    tracks.push_back(single);

    // (c) duplicate observation at the same clone
    FeatureTrack dup;
    dup.id = 2;
    dup.observations.push_back(make_observation(0, 0, uv0));
    dup.observations.push_back(make_observation(0, 0, uv0 + Eigen::Vector2d(0.01, -0.01)));
    dup.observations.push_back(make_observation(1, 0, project_world(qs[1], ps[1], cam, world_point)));
    tracks.push_back(dup);

    // (d) all observations identical
    FeatureTrack identical;
    identical.id = 3;
    for (int i = 0; i < 3; ++i) identical.observations.push_back(make_observation(i, 0, uv0));
    tracks.push_back(identical);

    // (e) a camera id that does not exist
    FeatureTrack bad_camera;
    bad_camera.id = 4;
    bad_camera.observations.push_back(make_observation(0, 7, uv0));
    bad_camera.observations.push_back(make_observation(1, 7, uv0));
    tracks.push_back(bad_camera);

    // (f) a valid track so the update path is really exercised as well
    FeatureTrack good;
    good.id = 5;
    for (int i = 0; i < 3; ++i)
        good.observations.push_back(make_observation(i, 0, project_world(qs[static_cast<std::size_t>(i)],
                                                                        ps[static_cast<std::size_t>(i)], cam, world_point)));
    tracks.push_back(good);

    const UpdateSummary summary = updater.update(state, tracks, {cam});
    std::printf("[T8] used=%d rejected=%d obs=%d\n", summary.features_used, summary.features_rejected,
                summary.observations_used);
    // Degenerate tracks may be rejected at triangulation or survive; the
    // contract is that every input track is accounted for and nothing is NaN.
    EXPECT_EQ(summary.features_used + summary.features_rejected,
              static_cast<int>(tracks.size()));

    EXPECT_TRUE(state.imu().q.allFinite());
    EXPECT_TRUE(state.imu().p.allFinite());
    EXPECT_TRUE(state.imu().v.allFinite());
    EXPECT_TRUE(state.covariance().allFinite());
    for (int i = 0; i < state.num_clones(); ++i)
    {
        EXPECT_TRUE(state.clone(i).q.allFinite());
        EXPECT_TRUE(state.clone(i).p.allFinite());
    }
    EXPECT_NEAR(state.imu().q.norm(), 1.0, 1e-12);
}

// ===========================================================================
// T9 -- frame-level feature/observation starvation gates must be no-ops.
// ===========================================================================
TEST(SchurUpdate, T9MinimumFeatureAndObservationGatesSkipUpdate)
{
    const CameraExtrinsics cam = make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector4d> qs;
    std::vector<Eigen::Vector3d> ps;
    smooth_trajectory(3, qs, ps);
    const Eigen::Vector3d world_point(0.5, 0.8, 4.0);

    auto make_state = [&]() {
        EkfState state;
        build_state(state, qs, ps);
        regularize_covariance(state);
        return state;
    };
    auto make_tracks = [&]() {
        FeatureTrack track;
        track.id = 900;
        for (int i = 0; i < 3; ++i)
        {
            track.observations.push_back(
                make_observation(i, 0, project_world(qs[static_cast<std::size_t>(i)],
                                                     ps[static_cast<std::size_t>(i)], cam,
                                                     world_point)));
        }
        return std::vector<FeatureTrack>{track};
    };

    // One usable track is below min_features=2.
    {
        UpdateOptions opts = default_options();
        opts.min_features = 2;
        opts.min_update_observations = 1;
        SchurUpdater updater(opts);
        EkfState state = make_state();
        std::vector<FeatureTrack> tracks = make_tracks();
        const Snapshot before = snapshot(state);
        const UpdateSummary summary = updater.update(state, tracks, {cam});
        std::printf("[T9] feature gate applied=%d features=%d obs=%d\n",
                    summary.applied ? 1 : 0, summary.features_used,
                    summary.observations_used);
        EXPECT_FALSE(summary.applied);
        EXPECT_LT(snapshot_max_diff(before, snapshot(state)), 1e-15);
    }

    // The same three-observation track is below min_update_observations=4.
    {
        UpdateOptions opts = default_options();
        opts.min_features = 1;
        opts.min_update_observations = 4;
        SchurUpdater updater(opts);
        EkfState state = make_state();
        std::vector<FeatureTrack> tracks = make_tracks();
        const Snapshot before = snapshot(state);
        const UpdateSummary summary = updater.update(state, tracks, {cam});
        std::printf("[T9] observation gate applied=%d features=%d obs=%d\n",
                    summary.applied ? 1 : 0, summary.features_used,
                    summary.observations_used);
        EXPECT_FALSE(summary.applied);
        EXPECT_LT(snapshot_max_diff(before, snapshot(state)), 1e-15);
    }
}

// ===========================================================================
// T10 -- one track has a structurally rank-deficient information matrix.
// ===========================================================================
TEST(SchurUpdate, T10SingleTrackStructuralNullspaceIsRankRevealing)
{
    const UpdateOptions opts = default_options();
    SchurUpdater updater(opts);
    const CameraExtrinsics cam = make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector4d> qs;
    std::vector<Eigen::Vector3d> ps;
    smooth_trajectory(3, qs, ps);
    EkfState state;
    build_state(state, qs, ps);
    regularize_covariance(state);

    const Eigen::Vector3d world_point(0.7, 0.4, 4.0);
    FeatureTrack track;
    track.id = 1000;
    for (int i = 0; i < 3; ++i)
    {
        track.observations.push_back(
            make_observation(i, 0, project_world(qs[static_cast<std::size_t>(i)],
                                                 ps[static_cast<std::size_t>(i)], cam,
                                                 world_point)));
    }

    std::vector<FeatureTrack> tracks{track};
    const UpdateSummary summary = updater.update(state, tracks, {cam});
    const Eigen::MatrixXd P = state.covariance();
    const double symmetry_error = (P - P.transpose()).norm();
    const double min_eigenvalue = min_covariance_eigenvalue(P);
    const double psd_tolerance = 1e-9 * std::max(1.0, P.cwiseAbs().maxCoeff());

    std::printf("[T10] applied=%d rank=%d n=%d cond=%.6e min_eig=%.3e\n",
                summary.applied ? 1 : 0, summary.rank, state.size(), summary.condition,
                min_eigenvalue);
    EXPECT_TRUE(summary.applied);
    EXPECT_GT(summary.rank, 0);
    EXPECT_LT(summary.rank, state.size());
    EXPECT_TRUE(std::isfinite(summary.condition));
    EXPECT_GE(summary.condition, 1.0);
    EXPECT_TRUE(state_is_finite(state));
    EXPECT_TRUE(P.allFinite());
    EXPECT_LT(symmetry_error, 1e-12 * std::max(1.0, P.norm()));
    EXPECT_GE(min_eigenvalue, -psd_tolerance);

    // A single track cannot constrain the visual pose gauge or the mono scale.
    // The rank-revealing square-root update must discard those structural null
    // directions; unlike the old information-form path, no ridge is added to
    // inject fictitious information (and a spurious covariance update) there.
}

// ===========================================================================
// T11 -- an invalid point block rejects the track and leaves the frame intact.
// ===========================================================================
TEST(SchurUpdate, T11DegeneratePointBlockSkipsWholeFrame)
{
    const UpdateOptions opts = default_options();
    SchurUpdater updater(opts);
    const CameraExtrinsics cam = make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector4d> qs;
    std::vector<Eigen::Vector3d> ps;
    smooth_trajectory(3, qs, ps);
    EkfState state;
    build_state(state, qs, ps);
    regularize_covariance(state);

    // Repeating the same observation from one clone leaves the 3x3 point block
    // with rank two, even though the track has enough observations to pass the
    // frame-level observation gate.
    const Eigen::Vector3d world_point(0.5, 0.5, 4.0);
    const Eigen::Vector2d uv = project_world(qs[0], ps[0], cam, world_point);
    FeatureTrack duplicate;
    duplicate.id = 1100;
    duplicate.world_point = world_point;
    duplicate.has_point = true;
    duplicate.observations.push_back(make_observation(0, 0, uv));
    duplicate.observations.push_back(make_observation(0, 0, uv));

    const TrackReduction red = updater.reduce_track(state, {cam}, duplicate);
    EXPECT_FALSE(red.valid);

    const Snapshot before = snapshot(state);
    std::vector<FeatureTrack> tracks{duplicate};
    const UpdateSummary summary = updater.update(state, tracks, {cam});
    std::printf("[T11] point_block_valid=%d applied=%d rank=%d\n",
                red.valid ? 1 : 0, summary.applied ? 1 : 0, summary.rank);
    EXPECT_FALSE(summary.applied);
    EXPECT_LT(snapshot_max_diff(before, snapshot(state)), 1e-15);
}

// ===========================================================================
// T12 -- multiple valid tracks produce a finite, applied, non-empty update.
// ===========================================================================
TEST(SchurUpdate, T12MultipleTracksApplyWithFiniteCondition)
{
    UpdateOptions opts = default_options();
    // This test exercises the applied path for several tracks; the dedicated
    // T14 covers the production condition_limit rejection.
    opts.condition_limit = 0.0;
    SchurUpdater updater(opts);
    const CameraExtrinsics cam = make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d(0.02, 0.0, 0.0));
    std::vector<Eigen::Vector4d> qs;
    std::vector<Eigen::Vector3d> ps;
    smooth_trajectory(4, qs, ps);
    EkfState state;
    build_state(state, qs, ps);
    regularize_covariance(state);

    std::vector<FeatureTrack> tracks;
    const int num_tracks = 8;
    for (int f = 0; f < num_tracks; ++f)
    {
        FeatureTrack track;
        track.id = 1200 + f;
        const Eigen::Vector3d world_point(-0.8 + 0.25 * f, 0.2 * ((f % 4) - 1.5),
                                          3.8 + 0.15 * f);
        for (int i = 0; i < 4; ++i)
        {
            track.observations.push_back(
                make_observation(i, 0, project_world(qs[static_cast<std::size_t>(i)],
                                                     ps[static_cast<std::size_t>(i)], cam,
                                                     world_point)));
        }
        tracks.push_back(track);
    }

    const UpdateSummary summary = updater.update(state, tracks, {cam});
    std::printf("[T12] applied=%d rank=%d n=%d cond=%.6e features=%d obs=%d\n",
                summary.applied ? 1 : 0, summary.rank, state.size(), summary.condition,
                summary.features_used, summary.observations_used);
    EXPECT_TRUE(summary.applied);
    EXPECT_GT(summary.rank, 0);
    EXPECT_GE(summary.condition, 1.0);
    EXPECT_TRUE(std::isfinite(summary.condition));
    EXPECT_EQ(summary.features_used, num_tracks);
    EXPECT_EQ(summary.observations_used, 4 * num_tracks);
    EXPECT_TRUE(state_is_finite(state));
}

// ===========================================================================
// chi2_quantile_95 sanity (Wilson-Hilferty closed form).
// ===========================================================================
TEST(SchurUpdate, Chi2Quantile95ClosedForm)
{
    EXPECT_DOUBLE_EQ(SchurUpdater::chi2_quantile_95(0), 0.0);
    EXPECT_DOUBLE_EQ(SchurUpdater::chi2_quantile_95(-3), 0.0);
    // Reference values of the chi-square 0.95 quantile.
    EXPECT_NEAR(SchurUpdater::chi2_quantile_95(1), 3.8415, 1.0e-1);
    EXPECT_NEAR(SchurUpdater::chi2_quantile_95(2), 5.9915, 6.0e-2);
    EXPECT_NEAR(SchurUpdater::chi2_quantile_95(6), 12.5916, 3.0e-2);
    EXPECT_NEAR(SchurUpdater::chi2_quantile_95(20), 31.4104, 2.0e-1);
}

// ===========================================================================
// T27 -- P6 weak update rescues a chi2 gate collapse ("accept more, weight
// less").
//
// A drifted, over-confident filter whose clones no longer match the
// (truth-derived) observations produces per-track chi2 values beyond the
// already-loose 5x gate, so every track is rejected ("no gated tracks") and the
// filter would keep coasting on IMU alone until a full restart.  Setting
// UpdateOptions::weak_update_scale > 1.0 relaxes *only* the chi2 threshold,
// down-weights the accepted observations (sigma -> sigma * scale) and lowers
// the two starvation floors.  The same tracks then produce a small correction
// that still reduces the observable (gauge-invariant) pose error.
//
// Nominal sigma note: with the point profiled out of S the residual direction
// sees S = (I - Hf (Hf^T Hf)^-1 Hf^T) + Hx P Hx^T, and triangulate() rejects a
// point whose RMS reprojection error exceeds 5 px.  Together those bound chi2
// above by n_obs * (5 px / sigma_px)^2, so tripping the
// 5.0 * chi2_quantile_95(dof) gate on a *triangulatable* 6-observation track
// requires sigma_px < ~1.2 px.  This scenario therefore uses a 1 px nominal
// sigma while keeping the production 5.0 gate multiplier.
// ===========================================================================
TEST(SchurUpdate, T27WeakUpdateRescuesGateCollapse)
{
    const CameraExtrinsics cam = make_camera(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    const int kClones = 6;
    std::vector<Eigen::Vector4d> q_true;
    std::vector<Eigen::Vector3d> p_true;
    smooth_trajectory(kClones, q_true, p_true);

    // A static point cloud in front of the trajectory.
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> xy(-0.6, 0.6);
    std::uniform_real_distribution<double> z(3.0, 5.0);
    std::vector<Eigen::Vector3d> points;
    for (int i = 0; i < 16; ++i)
        points.emplace_back(xy(rng), xy(rng), z(rng));

    // Observations are always generated from the truth poses.
    auto make_tracks = [&]() {
        std::vector<FeatureTrack> tracks;
        int id = 1000;
        for (const Eigen::Vector3d &wp : points)
        {
            FeatureTrack track;
            track.id = id++;
            for (int i = 0; i < kClones; ++i)
                track.observations.push_back(make_observation(
                    i, 0, project_world(q_true[static_cast<std::size_t>(i)],
                                        p_true[static_cast<std::size_t>(i)], cam, wp)));
            tracks.push_back(track);
        }
        return tracks;
    };

    // The drifted filter: nominal poses = truth perturbed *independently per
    // clone* (non-rigid, so the residual cannot be absorbed by the pose/scale
    // gauge or by moving the triangulated point).  The 14.5 mm / 14.5 mrad
    // magnitude puts the 6-observation chi2 just above the 5x gate while
    // keeping the triangulation RMS under the 5 px cap.
    std::vector<Eigen::Vector4d> q_nom = q_true;
    std::vector<Eigen::Vector3d> p_nom = p_true;
    std::mt19937 perturb_rng(7);
    std::uniform_real_distribution<double> off(-1.0, 1.0);
    for (int i = 0; i < kClones; ++i)
    {
        p_nom[static_cast<std::size_t>(i)] +=
            Eigen::Vector3d(0.0145 * off(perturb_rng), 0.0145 * off(perturb_rng),
                            0.0145 * off(perturb_rng));
        q_nom[static_cast<std::size_t>(i)] = EkfState::quaternion_multiply(
            q_true[static_cast<std::size_t>(i)],
            quat(Eigen::Vector3d(0.0145 * off(perturb_rng), 0.0145 * off(perturb_rng),
                                 0.0145 * off(perturb_rng))));
    }

    auto scenario_options = []() {
        UpdateOptions o = default_options();
        o.pixel_noise = 1.0;  // see the nominal-sigma note above
        return o;
    };
    auto build = [&](const std::vector<Eigen::Vector4d> &qs,
                     const std::vector<Eigen::Vector3d> &ps) {
        EkfState state;
        build_state(state, qs, ps);
        regularize_covariance(state);
        return state;
    };

    // (1) Consistent state + consistent tracks -> a normal applied update, and
    //     the weak-update flag stays off (weak_update_scale defaults to 1.0).
    {
        UpdateOptions opts = scenario_options();
        SchurUpdater updater(opts);
        EkfState state = build(q_true, p_true);
        std::vector<FeatureTrack> tracks = make_tracks();
        const UpdateSummary summary = updater.update(state, tracks, {cam});
        std::printf("[T27] (1) consistent applied=%d used=%d obs=%d weak=%d\n",
                    summary.applied ? 1 : 0, summary.features_used, summary.observations_used,
                    summary.weak_update ? 1 : 0);
        EXPECT_TRUE(summary.applied);
        EXPECT_GT(summary.features_used, 0);
        EXPECT_FALSE(summary.weak_update);
        EXPECT_DOUBLE_EQ(summary.weak_update_scale, 1.0);
    }

    // (2) Drifted state, weak update OFF (scale = 1.0): the 5x gate rejects
    //     every track and the state is untouched ("no gated tracks").
    {
        UpdateOptions opts = scenario_options();
        EXPECT_DOUBLE_EQ(opts.weak_update_scale, 1.0);
        SchurUpdater updater(opts);
        EkfState state = build(q_nom, p_nom);
        const Snapshot before = snapshot(state);
        std::vector<FeatureTrack> tracks = make_tracks();
        const UpdateSummary summary = updater.update(state, tracks, {cam});
        std::printf("[T27] (2) gated applied=%d used=%d rejected=%d reason=\"%s\"\n",
                    summary.applied ? 1 : 0, summary.features_used, summary.features_rejected,
                    summary.failure_reason.c_str());
        EXPECT_FALSE(summary.applied);
        EXPECT_EQ(summary.features_used, 0);
        EXPECT_EQ(summary.failure_reason, std::string("no gated tracks"));
        EXPECT_FALSE(summary.weak_update);
        EXPECT_LT(snapshot_max_diff(before, snapshot(state)), 1e-12);
    }

    // (3) Same drifted state + tracks, weak update ON (scale = 10): the gate is
    //     relaxed, tracks are accepted, the update applies, the gauge-invariant
    //     pose error shrinks and P stays symmetric PSD.
    {
        UpdateOptions opts = scenario_options();
        opts.weak_update_scale = 10.0;
        SchurUpdater updater(opts);
        EkfState state = build(q_nom, p_nom);
        double before_pos = 0.0, before_rot = 0.0, after_pos = 0.0, after_rot = 0.0;
        pose_errors(state, q_true, p_true, true, before_pos, before_rot);
        std::vector<FeatureTrack> tracks = make_tracks();
        const UpdateSummary summary = updater.update(state, tracks, {cam});
        pose_errors(state, q_true, p_true, true, after_pos, after_rot);
        const Eigen::MatrixXd &P = state.covariance();
        const double asym = (P - P.transpose()).cwiseAbs().maxCoeff();
        std::printf("[T27] (3) weak applied=%d used=%d obs=%d weak=%d scale=%.1f "
                    "rel-pos %.4e -> %.4e asym=%.1e\n",
                    summary.applied ? 1 : 0, summary.features_used, summary.observations_used,
                    summary.weak_update ? 1 : 0, summary.weak_update_scale, before_pos, after_pos,
                    asym);
        EXPECT_TRUE(summary.applied);
        EXPECT_GT(summary.features_used, 0);
        EXPECT_TRUE(summary.weak_update);
        EXPECT_DOUBLE_EQ(summary.weak_update_scale, 10.0);
        // A visual-only update cannot observe the six-dof world gauge, so the
        // gauge-invariant (relative) position error is the meaningful metric.
        EXPECT_LT(after_pos, before_pos);
        EXPECT_LT(asym, 1e-12);
        EXPECT_TRUE(EkfState::is_positive_semidefinite(P));
        EXPECT_TRUE(state_is_finite(state));
    }

    // (4) Down-weighting: compare the weak update against a control that
    //     relaxes the gate by the *same* factor (chi2_multiplier * 10) but does
    //     NOT inflate sigma.  Both accept the same tracks; the weak correction
    //     must be far smaller because each observation carries sigma * scale.
    {
        UpdateOptions ctrl = scenario_options();
        ctrl.chi2_multiplier = ctrl.chi2_multiplier * 10.0;  // same relaxed gate
        ctrl.weak_update_scale = 1.0;                        // no down-weight
        SchurUpdater ctrl_updater(ctrl);
        EkfState ctrl_state = build(q_nom, p_nom);
        const Snapshot ctrl_before = snapshot(ctrl_state);
        std::vector<FeatureTrack> ctrl_tracks = make_tracks();
        const UpdateSummary ctrl_summary = ctrl_updater.update(ctrl_state, ctrl_tracks, {cam});
        const double dx_ctrl = reconstruct_dx(ctrl_before, ctrl_state).norm();

        UpdateOptions weak = scenario_options();
        weak.weak_update_scale = 10.0;
        SchurUpdater weak_updater(weak);
        EkfState weak_state = build(q_nom, p_nom);
        const Snapshot weak_before = snapshot(weak_state);
        std::vector<FeatureTrack> weak_tracks = make_tracks();
        const UpdateSummary weak_summary = weak_updater.update(weak_state, weak_tracks, {cam});
        const double dx_weak = reconstruct_dx(weak_before, weak_state).norm();

        std::printf("[T27] (4) |dx| ctrl(no dw)=%.3e weak(dw)=%.3e ratio=%.3f\n", dx_ctrl, dx_weak,
                    dx_ctrl > 0.0 ? dx_weak / dx_ctrl : 0.0);
        ASSERT_TRUE(ctrl_summary.applied);
        ASSERT_FALSE(ctrl_summary.weak_update);
        ASSERT_TRUE(weak_summary.applied);
        ASSERT_TRUE(weak_summary.weak_update);
        // Same accepted set, so the only remaining difference is the weight.
        EXPECT_EQ(weak_summary.features_used, ctrl_summary.features_used);
        ASSERT_GT(dx_ctrl, 0.0);
        EXPECT_GT(dx_weak, 0.0);
        EXPECT_LT(dx_weak, dx_ctrl / 5.0);
    }
}
