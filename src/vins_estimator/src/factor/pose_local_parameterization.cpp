#include "pose_local_parameterization.h"

bool PoseManifold::Plus(const double * _x, const double * _delta, double * _result) const
{
  Eigen::Map<Eigen::Vector3d> p(_result);
  Eigen::Map<Eigen::Quaterniond> q(_result + 3);
  p = Eigen::Map<const Eigen::Vector3d>(_x) + Eigen::Map<const Eigen::Vector3d>(_delta);
  q = (Eigen::Map<const Eigen::Quaterniond>(_x + 3) *
       Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(_delta + 3)))
        .normalized();
  return true;
}

Eigen::Matrix<double, 7, 6> PoseManifold::plusJacobian(const double * _x)
{
  const Eigen::Map<const Eigen::Quaterniond> q(_x + 3);
  Eigen::Matrix<double, 7, 6> J = Eigen::Matrix<double, 7, 6>::Zero();
  J.topLeftCorner<3, 3>().setIdentity();
  J.block<3, 3>(3, 3) = 0.5 * (q.w() * Eigen::Matrix3d::Identity() + Utility::skewSymmetric(q.vec()));
  J.block<1, 3>(6, 3) = -0.5 * q.vec().transpose();
  return J;
}

Eigen::Matrix<double, 6, 7> PoseManifold::minusJacobian(const double * _x)
{
  Eigen::Matrix<double, 6, 7> J = plusJacobian(_x).transpose();
  J.bottomRows<3>() *= 4.0;
  return J;
}

bool PoseManifold::PlusJacobian(const double * _x, double * _data) const
{
  Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(_data);
  J = plusJacobian(_x);
  return true;
}

bool PoseManifold::Minus(const double * _y, const double * _x, double * _result) const
{
  Eigen::Map<Eigen::Vector3d> dp(_result);
  Eigen::Map<Eigen::Vector3d> dr(_result + 3);
  dp = Eigen::Map<const Eigen::Vector3d>(_y) - Eigen::Map<const Eigen::Vector3d>(_x);
  Eigen::Quaterniond relative =
    Eigen::Map<const Eigen::Quaterniond>(_x + 3).conjugate() * Eigen::Map<const Eigen::Quaterniond>(_y + 3);
  if (std::abs(relative.w()) < 1e-12)
    return false;
  dr = 2.0 * relative.vec() / relative.w();
  return true;
}

bool PoseManifold::MinusJacobian(const double * _x, double * _data) const
{
  Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(_data);
  J = minusJacobian(_x);
  return true;
}
