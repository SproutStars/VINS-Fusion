/*******************************************************
 * ARM-oriented Schur-complement EKF backend: filter state.
 *
 * Conventions (frozen, see ekf_state.h):
 *   q = [w, x, y, z] Hamilton product, body -> world, R = R(q).
 *   Right (body frame) error perturbation: q_true = q_hat (x) [1, dtheta/2],
 *   hence R(q_true) ~= R(q_hat) * (I + [dtheta]x).
 *   Live error state (15): [dtheta | dp | dv | dba | dbg] and clones follow
 *   oldest first, 6 each: [dtheta_i | dp_i].
 *
 * Cost notes (ARM): every hot-path matrix is a fixed-size Eigen type
 * (4x4, 6x6, 12x12, 15x15, 15x12).  The window covariance is dynamic-sized by
 * construction (the clone window length is a runtime option), but the block
 * update in propagate() reuses per-thread scratch buffers, so a steady-state
 * propagate() performs no heap allocation at all.
 *******************************************************/

#include "backend/ekf_state.h"

#include <cmath>
#include <utility>

namespace vins
{
namespace backend
{
namespace
{

using Mat3 = Eigen::Matrix3d;
using Mat15 = Eigen::Matrix<double, kImuStateSize, kImuStateSize>;
using Mat15x12 = Eigen::Matrix<double, kImuStateSize, 12>;
using Mat12 = Eigen::Matrix<double, 12, 12>;

// Squared quaternion norm below which a quaternion is considered unusable.
constexpr double kNormEps = 1e-24;
// |axis * angle| below which the small-angle series is used.
constexpr double kSmallAngle = 1e-8;

// Scratch buffers for the block update of the window covariance.  They are
// resized in place, so after the first call with a given window size the
// propagate() hot path stops allocating (Eigen::resize is a no-op when the
// shape is unchanged).  thread_local keeps concurrent filters independent.
thread_local Eigen::MatrixXd g_row_scratch;
thread_local Eigen::MatrixXd g_col_scratch;

// One RK4 step of the nominal IMU kinematics with the biases held constant over
// the step (same structure as SchurVINS PredictionState / open_vins
// predict_mean_rk4):
//     q' = 0.5 * Omega(gyr - bg) * q,   p' = v,
//     v' = R(q) * (acc - ba) + g_w,     ba' = bg' = 0.
// `w` and `a` are the bias-corrected measurements, `g_w` the world gravity.
void rk4_imu_step(const Eigen::Vector3d &w, const Eigen::Vector3d &a,
                  const Eigen::Vector3d &g_w, double dt, Eigen::Vector4d &q,
                  Eigen::Vector3d &p, Eigen::Vector3d &v)
{
    const Eigen::Matrix4d omega = EkfState::omega_matrix(w);
    const auto dq_dt = [&omega](const Eigen::Vector4d &qq) -> Eigen::Vector4d {
        return 0.5 * (omega * qq);
    };
    const auto dv_dt = [&a, &g_w](const Eigen::Vector4d &qq) -> Eigen::Vector3d {
        return EkfState::rotation(qq) * a + g_w;
    };

    const double h = dt;
    const double h2 = 0.5 * dt;
    const double h6 = dt / 6.0;

    const Eigen::Vector4d k1q = dq_dt(q);
    const Eigen::Vector3d k1p = v;
    const Eigen::Vector3d k1v = dv_dt(q);

    const Eigen::Vector4d q2 = q + h2 * k1q;
    const Eigen::Vector4d k2q = dq_dt(q2);
    const Eigen::Vector3d k2p = v + h2 * k1v;
    const Eigen::Vector3d k2v = dv_dt(q2);

    const Eigen::Vector4d q3 = q + h2 * k2q;
    const Eigen::Vector4d k3q = dq_dt(q3);
    const Eigen::Vector3d k3p = v + h2 * k2v;
    const Eigen::Vector3d k3v = dv_dt(q3);

    const Eigen::Vector4d q4 = q + h * k3q;
    const Eigen::Vector4d k4q = dq_dt(q4);
    const Eigen::Vector3d k4p = v + h * k3v;
    const Eigen::Vector3d k4v = dv_dt(q4);

    q += h6 * (k1q + 2.0 * k2q + 2.0 * k3q + k4q);
    p += h6 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);
    v += h6 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);

    // The RK4 step is norm preserving only to O(dt^5) (Omega is skew), so the
    // quaternion is renormalised every step to keep R(q) a rotation.
    q = EkfState::quaternion_normalize(q);
}

}  // namespace

// ---------------------------------------------------------------- reset -----

void EkfState::reset(const EkfOptions &options, double timestamp,
                     const Eigen::Vector4d &q, const Eigen::Vector3d &p,
                     const Eigen::Vector3d &v, const Eigen::Vector3d &ba,
                     const Eigen::Vector3d &bg)
{
    options_ = options;
    initialized_ = true;

    imu_.q = q;
    imu_.p = p;
    imu_.v = v;
    imu_.ba = ba;
    imu_.bg = bg;
    imu_fej_ = imu_;

    clones_.clear();
    clone_fej_.clear();
    clone_timestamps_.clear();
    timestamp_ = timestamp;

    // P0 = diag(rot, pos, vel, acc bias, gyr bias), all three-axis blocks.
    const Mat3 I3 = Mat3::Identity();
    covariance_ = Eigen::MatrixXd::Zero(kImuStateSize, kImuStateSize);
    covariance_.block<3, 3>(kImuTheta, kImuTheta) =
        options_.init_rotation_sigma * options_.init_rotation_sigma * I3;
    covariance_.block<3, 3>(kImuP, kImuP) =
        options_.init_position_sigma * options_.init_position_sigma * I3;
    covariance_.block<3, 3>(kImuV, kImuV) =
        options_.init_velocity_sigma * options_.init_velocity_sigma * I3;
    covariance_.block<3, 3>(kImuBa, kImuBa) =
        options_.init_acc_bias_sigma * options_.init_acc_bias_sigma * I3;
    covariance_.block<3, 3>(kImuBg, kImuBg) =
        options_.init_gyr_bias_sigma * options_.init_gyr_bias_sigma * I3;
}

void EkfState::copyLiveFrom(const EkfState &source)
{
    if (this == &source) return;
    options_ = source.options_;
    initialized_ = source.initialized_;
    timestamp_ = source.timestamp_;
    imu_ = source.imu_;
    imu_fej_ = source.imu_fej_;
    clones_.clear();
    clone_fej_.clear();
    clone_timestamps_.clear();
    if (initialized_)
        covariance_ = source.covariance_.topLeftCorner<kImuStateSize, kImuStateSize>();
    else
        covariance_.resize(0, 0);
}

// ------------------------------------------------------------- propagate ----

void EkfState::propagate(const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr,
                         double dt, double noise_scale)
{
    if (!initialized_ || !(dt > 0.0))
        return;

    // A non-finite or < 1 factor would either break the covariance or shrink the
    // process noise; fall back to the neutral value. The Estimator already
    // validates the parameter, so this is purely defensive.
    if (!std::isfinite(noise_scale) || noise_scale < 1.0)
        noise_scale = 1.0;

    const Eigen::Vector3d g_w(0.0, 0.0, -options_.gravity);

    // ---- nominal mean: RK4 with the biases frozen inside the step ----------
    const Eigen::Vector3d w = gyr - imu_.bg;
    const Eigen::Vector3d a = acc - imu_.ba;
    rk4_imu_step(w, a, g_w, dt, imu_.q, imu_.p, imu_.v);

    // ---- covariance: one Euler/RK-style step linearised at the FEJ --------
    // The interval-start FEJ is used for F/G below. Once the mean and covariance
    // have been propagated, the live FEJ is advanced to the new mean; this is
    // the same relinearisation used by OpenVINS Propagator::propagate().
    const Eigen::Vector3d w_f = gyr - imu_fej_.bg;
    const Eigen::Vector3d a_f = acc - imu_fej_.ba;
    const Mat3 R = rotation(imu_fej_.q);
    const Mat3 I3 = Mat3::Identity();

    // F: d/dt [dtheta, dp, dv, dba, dbg]
    //   dtheta' = -[w]x dtheta - dbg
    //   dp'     = dv
    //   dv'     = -R [a]x dtheta - R dba
    Mat15 F = Mat15::Zero();
    F.block<3, 3>(kImuTheta, kImuTheta) = -skew(w_f);
    F.block<3, 3>(kImuTheta, kImuBg) = -I3;
    F.block<3, 3>(kImuP, kImuV) = I3;
    F.block<3, 3>(kImuV, kImuTheta) = -R * skew(a_f);
    F.block<3, 3>(kImuV, kImuBa) = -R;

    // G maps the process noise n = [n_g, n_a, n_bg, n_ba] into the state.
    Mat15x12 G = Mat15x12::Zero();
    G.block<3, 3>(kImuTheta, 0) = -I3;
    G.block<3, 3>(kImuV, 3) = -R;
    G.block<3, 3>(kImuBg, 6) = I3;
    G.block<3, 3>(kImuBa, 9) = I3;

    // Spectral densities of the four noise inputs, already integrated over this
    // step: Qd = Phi G Qc G^T Phi^T below carries the step length implicitly
    // (PhiG ~= G, so the diagonal blocks of Qd come out as sigma^2 * dt).  The
    // /dt form used previously inflated the gyro/accel blocks by 1/dt^2, which
    // made attitude and velocity effectively unobservable.
    Mat12 Qc = Mat12::Zero();
    Qc.block<3, 3>(0, 0) = (options_.gyr_noise * options_.gyr_noise * dt) * I3;
    Qc.block<3, 3>(3, 3) = (options_.acc_noise * options_.acc_noise * dt) * I3;
    Qc.block<3, 3>(6, 6) =
        (options_.gyr_bias_noise * options_.gyr_bias_noise * dt) * I3;
    Qc.block<3, 3>(9, 9) =
        (options_.acc_bias_noise * options_.acc_bias_noise * dt) * I3;

    // Visual-starvation inflation: scale the whole process-noise covariance by a
    // positive factor (>= 1). Qc is symmetric PSD by construction, so a positive
    // scalar keeps it symmetric PSD, and Qd = Phi G Qc G^T Phi^T inherits that.
    if (noise_scale != 1.0)
        Qc *= noise_scale;

    // Phi = exp(F dt) truncated after the cubic term; Qd = Phi G Qc G^T Phi^T.
    const Mat15 F2 = F * F;
    const Mat15 Phi =
        Mat15::Identity() + F * dt + 0.5 * F2 * (dt * dt) +
        (1.0 / 6.0) * (F2 * F) * (dt * dt * dt);
    const Mat15x12 PhiG = Phi * G;
    const Mat15 Qd = PhiG * Qc * PhiG.transpose();

    // P <- Phi_full P Phi_full^T + Qd_full, where Phi_full = I + D and D lives
    // only in the leading 15x15 block.  Left and right multiplication therefore
    // touch only the live rows/columns; the clone blocks change only through
    // the cross terms, exactly as for an explicit size()-sized Phi_full.
    const int n = size();
    g_row_scratch.resize(kImuStateSize, n);
    g_col_scratch.resize(n, kImuStateSize);
    g_row_scratch.noalias() = Phi * covariance_.topRows(kImuStateSize);
    covariance_.topRows(kImuStateSize) = g_row_scratch;
    g_col_scratch.noalias() =
        covariance_.leftCols(kImuStateSize) * Phi.transpose();
    covariance_.leftCols(kImuStateSize) = g_col_scratch;
    covariance_.topLeftCorner(kImuStateSize, kImuStateSize) += Qd;

    symmetrize(covariance_);
    imu_fej_ = imu_;
    timestamp_ += dt;
}

// ----------------------------------------------------------- clone window ---

void EkfState::augment_clone(double timestamp)
{
    if (!initialized_)
        return;

    const int n = size();

    // The clone mean is frozen at the current live pose.
    clones_.push_back(CloneNominal{imu_.q, imu_.p});
    clone_fej_.push_back(CloneNominal{imu_fej_.q, imu_fej_.p});
    clone_timestamps_.push_back(timestamp);

    // The new clone is a perfect copy of the live pose, hence its covariance
    // block and all cross terms are copies of rows/cols 0:6.
    Eigen::MatrixXd P_new =
        Eigen::MatrixXd::Zero(n + kCloneStateSize, n + kCloneStateSize);
    P_new.topLeftCorner(n, n) = covariance_;
    P_new.block(0, n, n, kCloneStateSize) = covariance_.leftCols(kCloneStateSize);
    P_new.block(n, 0, kCloneStateSize, n) = covariance_.topRows(kCloneStateSize);
    P_new.block(n, n, kCloneStateSize, kCloneStateSize) =
        covariance_.topLeftCorner(kCloneStateSize, kCloneStateSize);
    covariance_ = std::move(P_new);

    timestamp_ = timestamp;
}

int EkfState::drop_all_clones()
{
    int removed = 0;
    while (num_clones() > 0)
    {
        marginalize_oldest_clone();
        ++removed;
    }
    return removed;
}

int EkfState::prune_clone_window()
{
    int removed = 0;
    const int keep = std::max(0, options_.window_size);
    while (num_clones() > keep)
    {
        marginalize_oldest_clone();
        ++removed;
    }
    return removed;
}

void EkfState::marginalize_oldest_clone()
{
    if (num_clones() == 0)
        return;

    const int b = clone_offset(0);      // always kImuStateSize for the oldest clone
    const int n = size();
    const int m = n - kCloneStateSize;
    const int tail = m - b;             // newer clones after the removed block

    // Hard deletion, as in OpenVINS StateHelper::marginalize() and SchurVINS
    // Management(). The old conditional Schur complement P_rr - P_rc P_cc^-1
    // P_cr injected a mathematically different, over-confident covariance.
    Eigen::MatrixXd P_new(m, m);
    P_new.topLeftCorner(b, b) = covariance_.topLeftCorner(b, b);
    if (tail > 0)
    {
        P_new.topRightCorner(b, tail) =
            covariance_.block(0, b + kCloneStateSize, b, tail);
        P_new.bottomLeftCorner(tail, b) =
            covariance_.block(b + kCloneStateSize, 0, tail, b);
        P_new.bottomRightCorner(tail, tail) =
            covariance_.block(b + kCloneStateSize, b + kCloneStateSize, tail, tail);
    }
    symmetrize(P_new);
    covariance_ = std::move(P_new);

    clones_.erase(clones_.begin());
    clone_fej_.erase(clone_fej_.begin());
    clone_timestamps_.erase(clone_timestamps_.begin());
}

void EkfState::refresh_imu_fej()
{
    imu_fej_ = imu_;
}

void EkfState::refresh_fej()
{
    imu_fej_ = imu_;
    clone_fej_ = clones_;
}

// --------------------------------------------------------------- helpers ----

Eigen::Matrix3d EkfState::rotation(const Eigen::Vector4d &q)
{
    const Eigen::Vector4d n = quaternion_normalize(q);
    if (n.squaredNorm() < kNormEps)
        return Eigen::Matrix3d::Identity();

    const double w = n(0), x = n(1), y = n(2), z = n(3);
    Eigen::Matrix3d R;
    R << 1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
        2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
        2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y);
    return R;
}

Eigen::Matrix3d EkfState::skew(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d S;
    S << 0.0, -v(2), v(1),
         v(2), 0.0, -v(0),
         -v(1), v(0), 0.0;
    return S;
}

Eigen::Vector4d EkfState::quaternion_multiply(const Eigen::Vector4d &a,
                                             const Eigen::Vector4d &b)
{
    // Hamilton product a (x) b with q = [w, x, y, z].
    const double aw = a(0), ax = a(1), ay = a(2), az = a(3);
    const double bw = b(0), bx = b(1), by = b(2), bz = b(3);
    Eigen::Vector4d out;
    out(0) = aw * bw - ax * bx - ay * by - az * bz;
    out(1) = aw * bx + ax * bw + ay * bz - az * by;
    out(2) = aw * by - ax * bz + ay * bw + az * bx;
    out(3) = aw * bz + ax * by - ay * bx + az * bw;
    return quaternion_normalize(out);
}

Eigen::Vector4d EkfState::quaternion_from_axis_angle(const Eigen::Vector3d &axis_angle)
{
    const double theta = axis_angle.norm();
    Eigen::Vector4d q;
    if (theta < kSmallAngle)
    {
        // sin(t/2)/t -> 1/2 - t^2/48 and cos(t/2) -> 1 - t^2/8.
        q(0) = 1.0 - theta * theta / 8.0;
        q.tail<3>() = axis_angle * (0.5 - theta * theta / 48.0);
    }
    else
    {
        const double half = 0.5 * theta;
        q(0) = std::cos(half);
        q.tail<3>() = axis_angle * (std::sin(half) / theta);
    }
    return quaternion_normalize(q);
}

Eigen::Vector3d EkfState::quaternion_to_axis_angle(const Eigen::Vector4d &q)
{
    Eigen::Vector4d n = quaternion_normalize(q);
    if (n.squaredNorm() < kNormEps)
        return Eigen::Vector3d::Zero();
    if (n(0) < 0.0)  // pick the shortest rotation
        n = -n;

    const Eigen::Vector3d vec = n.tail<3>();
    const double vec_norm = vec.norm();
    if (vec_norm < kSmallAngle)
        return 2.0 * vec;  // ~ 2 * sin(t/2) * axis for small angles

    const double theta = 2.0 * std::atan2(vec_norm, n(0));
    return vec * (theta / vec_norm);
}

Eigen::Vector4d EkfState::quaternion_normalize(const Eigen::Vector4d &q)
{
    const double n2 = q.squaredNorm();
    if (n2 < kNormEps)
        return Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
    return q / std::sqrt(n2);
}

Eigen::Matrix4d EkfState::omega_matrix(const Eigen::Vector3d &omega)
{
    // JPL kinematics: q' = 0.5 * Omega(omega) * q for q = [w, x, y, z] with
    // q (x) [0, omega] = Omega(omega) * q.
    Eigen::Matrix4d M;
    M.setZero();
    M.block<1, 3>(0, 1) = -omega.transpose();
    M.block<3, 1>(1, 0) = omega;
    M.block<3, 3>(1, 1) = -skew(omega);
    return M;
}

void EkfState::apply_pose_correction(Eigen::Vector4d &q, Eigen::Vector3d &p,
                                     const Eigen::Matrix<double, 6, 1> &dx)
{
    // Right (body frame) perturbation: q <- q (x) [1, dtheta/2].
    Eigen::Vector4d dq;
    dq(0) = 1.0;
    dq.tail<3>() = 0.5 * dx.head<3>();
    q = quaternion_normalize(quaternion_multiply(q, dq));
    p += dx.tail<3>();
}

void EkfState::symmetrize(Eigen::MatrixXd &P)
{
    if (P.rows() != P.cols())
        return;
    Eigen::MatrixXd sym = 0.5 * (P + P.transpose());
    P.swap(sym);
}

bool EkfState::is_positive_definite(const Eigen::MatrixXd &P, double tolerance)
{
    if (P.rows() != P.cols() || P.rows() == 0 || !P.allFinite())
        return false;

    const Eigen::MatrixXd sym = 0.5 * (P + P.transpose());
    Eigen::LDLT<Eigen::MatrixXd> ldlt(sym);
    if (ldlt.info() != Eigen::Success)
        return false;
    const auto &D = ldlt.vectorD();
    return D.allFinite() && (D.minCoeff() > tolerance);
}

bool EkfState::is_positive_semidefinite(const Eigen::MatrixXd &P, double tolerance)
{
    if (P.rows() != P.cols() || P.rows() == 0 || !P.allFinite())
        return false;

    const Eigen::MatrixXd sym = 0.5 * (P + P.transpose());
    const double scale = std::max(1.0, sym.cwiseAbs().maxCoeff());
    // A plain LDLT can report NumericalIssue on a valid singular PSD matrix
    // (e.g. a fresh clone is an exact copy of the live pose). Use eigenvalues
    // so zero-variance directions are accepted while truly negative modes are
    // still rejected. This check runs once per visual update, not per track.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(
        sym, Eigen::EigenvaluesOnly);
    if (eig.info() != Eigen::Success)
        return false;
    const Eigen::VectorXd &eigenvalues = eig.eigenvalues();
    return eigenvalues.allFinite() &&
           eigenvalues.minCoeff() >= -tolerance * scale;
}

}  // namespace backend
}  // namespace vins
