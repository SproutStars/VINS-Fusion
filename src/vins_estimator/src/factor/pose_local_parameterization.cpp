#include "pose_local_parameterization.h"

bool PoseManifold::Plus(const double *x, const double *delta, double *result) const
{
    Eigen::Map<Eigen::Vector3d> p(result);
    Eigen::Map<Eigen::Quaterniond> q(result + 3);
    p = Eigen::Map<const Eigen::Vector3d>(x) + Eigen::Map<const Eigen::Vector3d>(delta);
    q = (Eigen::Map<const Eigen::Quaterniond>(x + 3) *
         Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3))).normalized();
    return true;
}

Eigen::Matrix<double, 7, 6> PoseManifold::plusJacobian(const double *x)
{
    const Eigen::Map<const Eigen::Quaterniond> q(x + 3);
    Eigen::Matrix<double, 7, 6> J = Eigen::Matrix<double, 7, 6>::Zero();
    J.topLeftCorner<3, 3>().setIdentity();
    J.block<3, 3>(3, 3) = 0.5 * (q.w() * Eigen::Matrix3d::Identity() + Utility::skewSymmetric(q.vec()));
    J.block<1, 3>(6, 3) = -0.5 * q.vec().transpose();
    return J;
}

Eigen::Matrix<double, 6, 7> PoseManifold::minusJacobian(const double *x)
{
    Eigen::Matrix<double, 6, 7> J = plusJacobian(x).transpose();
    J.bottomRows<3>() *= 4.0;
    return J;
}

bool PoseManifold::PlusJacobian(const double *x, double *data) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(data);
    J = plusJacobian(x);
    return true;
}

bool PoseManifold::Minus(const double *y, const double *x, double *result) const
{
    Eigen::Map<Eigen::Vector3d> dp(result);
    Eigen::Map<Eigen::Vector3d> dr(result + 3);
    dp = Eigen::Map<const Eigen::Vector3d>(y) - Eigen::Map<const Eigen::Vector3d>(x);
    Eigen::Quaterniond relative = Eigen::Map<const Eigen::Quaterniond>(x + 3).conjugate() *
                                 Eigen::Map<const Eigen::Quaterniond>(y + 3);
    if (std::abs(relative.w()) < 1e-12) return false;
    dr = 2.0 * relative.vec() / relative.w();
    return true;
}

bool PoseManifold::MinusJacobian(const double *x, double *data) const
{
    Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(data);
    J = minusJacobian(x);
    return true;
}
