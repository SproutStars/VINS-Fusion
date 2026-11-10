/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "pose_local_parameterization.h"

#include "projectionTwoFrameTwoCamFactor.h"

Eigen::Matrix2d ProjectionTwoFrameTwoCamFactor::sqrt_info_;
double ProjectionTwoFrameTwoCamFactor::sum_t_;

ProjectionTwoFrameTwoCamFactor::ProjectionTwoFrameTwoCamFactor(
  const Eigen::Vector3d & _pts_i, const Eigen::Vector3d & _pts_j, const Eigen::Vector2d & _velocity_i,
  const Eigen::Vector2d & _velocity_j, const double _td_i, const double _td_j)
  : pts_i_(_pts_i), pts_j_(_pts_j), td_i_(_td_i), td_j_(_td_j)
{
  velocity_i_.x() = _velocity_i.x();
  velocity_i_.y() = _velocity_i.y();
  velocity_i_.z() = 0;
  velocity_j_.x() = _velocity_j.x();
  velocity_j_.y() = _velocity_j.y();
  velocity_j_.z() = 0;

#ifdef UNIT_SPHERE_ERROR
  Eigen::Vector3d b1, b2;
  Eigen::Vector3d a = pts_j.normalized();
  Eigen::Vector3d tmp(0, 0, 1);
  if (a == tmp)
    tmp << 1, 0, 0;
  b1 = (tmp - a * (a.transpose() * tmp)).normalized();
  b2 = a.cross(b1);
  tangent_base.block<1, 3>(0, 0) = b1.transpose();
  tangent_base.block<1, 3>(1, 0) = b2.transpose();
#endif
};

bool ProjectionTwoFrameTwoCamFactor::Evaluate(
  double const * const * _parameters, double * _residuals, double ** _jacobians) const
{
  Eigen::Vector3d Pi(_parameters[0][0], _parameters[0][1], _parameters[0][2]);
  Eigen::Quaterniond Qi(_parameters[0][6], _parameters[0][3], _parameters[0][4], _parameters[0][5]);

  Eigen::Vector3d Pj(_parameters[1][0], _parameters[1][1], _parameters[1][2]);
  Eigen::Quaterniond Qj(_parameters[1][6], _parameters[1][3], _parameters[1][4], _parameters[1][5]);

  Eigen::Vector3d tic(_parameters[2][0], _parameters[2][1], _parameters[2][2]);
  Eigen::Quaterniond qic(_parameters[2][6], _parameters[2][3], _parameters[2][4], _parameters[2][5]);

  Eigen::Vector3d tic2(_parameters[3][0], _parameters[3][1], _parameters[3][2]);
  Eigen::Quaterniond qic2(_parameters[3][6], _parameters[3][3], _parameters[3][4], _parameters[3][5]);

  double inv_dep_i = _parameters[4][0];

  double td = _parameters[5][0];

  Eigen::Vector3d pts_i_td, pts_j_td;
  pts_i_td = pts_i_ - (td - td_i_) * velocity_i_;
  pts_j_td = pts_j_ - (td - td_j_) * velocity_j_;

  Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
  Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
  Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
  Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
  Eigen::Vector3d pts_camera_j = qic2.inverse() * (pts_imu_j - tic2);
  Eigen::Map<Eigen::Vector2d> residual(_residuals);

#ifdef UNIT_SPHERE_ERROR
  residual = tangent_base * (pts_camera_j.normalized() - pts_j_td.normalized());
#else
  double dep_j = pts_camera_j.z();
  residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();
#endif

  residual = sqrt_info_ * residual;

  if (_jacobians) {
    Eigen::Matrix3d Ri = Qi.toRotationMatrix();
    Eigen::Matrix3d Rj = Qj.toRotationMatrix();
    Eigen::Matrix3d ric = qic.toRotationMatrix();
    Eigen::Matrix3d ric2 = qic2.toRotationMatrix();
    Eigen::Matrix<double, 2, 3> reduce(2, 3);
#ifdef UNIT_SPHERE_ERROR
    double norm = pts_camera_j.norm();
    Eigen::Matrix3d norm_jaco;
    double x1, x2, x3;
    x1 = pts_camera_j(0);
    x2 = pts_camera_j(1);
    x3 = pts_camera_j(2);
    norm_jaco << 1.0 / norm - x1 * x1 / pow(norm, 3), -x1 * x2 / pow(norm, 3), -x1 * x3 / pow(norm, 3),
      -x1 * x2 / pow(norm, 3), 1.0 / norm - x2 * x2 / pow(norm, 3), -x2 * x3 / pow(norm, 3),
      -x1 * x3 / pow(norm, 3), -x2 * x3 / pow(norm, 3), 1.0 / norm - x3 * x3 / pow(norm, 3);
    reduce = tangent_base * norm_jaco;
#else
    reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j), 0, 1. / dep_j,
      -pts_camera_j(1) / (dep_j * dep_j);
#endif
    reduce = sqrt_info_ * reduce;

    if (_jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(_jacobians[0]);

      Eigen::Matrix<double, 3, 6> jaco_i;
      jaco_i.leftCols<3>() = ric2.transpose() * Rj.transpose();
      jaco_i.rightCols<3>() = ric2.transpose() * Rj.transpose() * Ri * -Utility::skewSymmetric(pts_imu_i);

      jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
      jacobian_pose_i.rightCols<1>().setZero();
    }

    if (_jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(_jacobians[1]);

      Eigen::Matrix<double, 3, 6> jaco_j;
      jaco_j.leftCols<3>() = ric2.transpose() * -Rj.transpose();
      jaco_j.rightCols<3>() = ric2.transpose() * Utility::skewSymmetric(pts_imu_j);

      jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
      jacobian_pose_j.rightCols<1>().setZero();
    }
    if (_jacobians[2]) {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(_jacobians[2]);
      Eigen::Matrix<double, 3, 6> jaco_ex;
      jaco_ex.leftCols<3>() = ric2.transpose() * Rj.transpose() * Ri;
      jaco_ex.rightCols<3>() =
        ric2.transpose() * Rj.transpose() * Ri * ric * -Utility::skewSymmetric(pts_camera_i);
      jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
      jacobian_ex_pose.rightCols<1>().setZero();
    }
    if (_jacobians[3]) {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose1(_jacobians[3]);
      Eigen::Matrix<double, 3, 6> jaco_ex;
      jaco_ex.leftCols<3>() = -ric2.transpose();
      jaco_ex.rightCols<3>() = Utility::skewSymmetric(pts_camera_j);
      jacobian_ex_pose1.leftCols<6>() = reduce * jaco_ex;
      jacobian_ex_pose1.rightCols<1>().setZero();
    }
    if (_jacobians[4]) {
      Eigen::Map<Eigen::Vector2d> jacobian_feature(_jacobians[4]);
#if 1
      jacobian_feature =
        reduce * ric2.transpose() * Rj.transpose() * Ri * ric * pts_i_td * -1.0 / (inv_dep_i * inv_dep_i);
#else
      jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i;
#endif
    }
    if (_jacobians[5]) {
      Eigen::Map<Eigen::Vector2d> jacobian_td(_jacobians[5]);
      jacobian_td = reduce * ric2.transpose() * Rj.transpose() * Ri * ric * velocity_i_ / inv_dep_i * -1.0 +
                    sqrt_info_ * velocity_j_.head(2);
    }
  }

  if (_jacobians) {
    if (_jacobians[0])
      PoseManifold::liftJacobian(_parameters[0], _jacobians[0], 2);
    if (_jacobians[1])
      PoseManifold::liftJacobian(_parameters[1], _jacobians[1], 2);
    if (_jacobians[2])
      PoseManifold::liftJacobian(_parameters[2], _jacobians[2], 2);
    if (_jacobians[3])
      PoseManifold::liftJacobian(_parameters[3], _jacobians[3], 2);
  }
  return true;
}

void ProjectionTwoFrameTwoCamFactor::check(double ** _parameters)
{
  double * res = new double[15];
  double ** jaco = new double *[6];
  jaco[0] = new double[2 * 7];
  jaco[1] = new double[2 * 7];
  jaco[2] = new double[2 * 7];
  jaco[3] = new double[2 * 7];
  jaco[4] = new double[2 * 1];
  jaco[5] = new double[2 * 1];
  Evaluate(_parameters, res, jaco);
  puts("check begins");

  puts("my");

  std::cout << Eigen::Map<Eigen::Matrix<double, 2, 1>>(res).transpose() << '\n' << '\n';
  std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[0]) << '\n' << '\n';
  std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[1]) << '\n' << '\n';
  std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[2]) << '\n' << '\n';
  std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[3]) << '\n' << '\n';
  std::cout << Eigen::Map<Eigen::Vector2d>(jaco[4]) << '\n' << '\n';
  std::cout << Eigen::Map<Eigen::Vector2d>(jaco[5]) << '\n' << '\n';

  Eigen::Vector3d Pi(_parameters[0][0], _parameters[0][1], _parameters[0][2]);
  Eigen::Quaterniond Qi(_parameters[0][6], _parameters[0][3], _parameters[0][4], _parameters[0][5]);

  Eigen::Vector3d Pj(_parameters[1][0], _parameters[1][1], _parameters[1][2]);
  Eigen::Quaterniond Qj(_parameters[1][6], _parameters[1][3], _parameters[1][4], _parameters[1][5]);

  Eigen::Vector3d tic(_parameters[2][0], _parameters[2][1], _parameters[2][2]);
  Eigen::Quaterniond qic(_parameters[2][6], _parameters[2][3], _parameters[2][4], _parameters[2][5]);

  Eigen::Vector3d tic2(_parameters[3][0], _parameters[3][1], _parameters[3][2]);
  Eigen::Quaterniond qic2(_parameters[3][6], _parameters[3][3], _parameters[3][4], _parameters[3][5]);

  double inv_dep_i = _parameters[4][0];

  double td = _parameters[5][0];

  Eigen::Vector3d pts_i_td, pts_j_td;
  pts_i_td = pts_i_ - (td - td_i_) * velocity_i_;
  pts_j_td = pts_j_ - (td - td_j_) * velocity_j_;

  Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
  Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
  Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
  Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
  Eigen::Vector3d pts_camera_j = qic2.inverse() * (pts_imu_j - tic2);


  Eigen::Vector2d residual;
#ifdef UNIT_SPHERE_ERROR
  residual = tangent_base * (pts_camera_j.normalized() - pts_j_td.normalized());
#else
  double dep_j = pts_camera_j.z();
  residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();
#endif
  residual = sqrt_info_ * residual;

  puts("num");
  std::cout << residual.transpose() << '\n';

  const double eps = 1e-6;
  Eigen::Matrix<double, 2, 26> num_jacobian;
  for (int k = 0; k < 26; k++) {
    Eigen::Vector3d Pi(_parameters[0][0], _parameters[0][1], _parameters[0][2]);
    Eigen::Quaterniond Qi(_parameters[0][6], _parameters[0][3], _parameters[0][4], _parameters[0][5]);

    Eigen::Vector3d Pj(_parameters[1][0], _parameters[1][1], _parameters[1][2]);
    Eigen::Quaterniond Qj(_parameters[1][6], _parameters[1][3], _parameters[1][4], _parameters[1][5]);

    Eigen::Vector3d tic(_parameters[2][0], _parameters[2][1], _parameters[2][2]);
    Eigen::Quaterniond qic(_parameters[2][6], _parameters[2][3], _parameters[2][4], _parameters[2][5]);

    Eigen::Vector3d tic2(_parameters[3][0], _parameters[3][1], _parameters[3][2]);
    Eigen::Quaterniond qic2(_parameters[3][6], _parameters[3][3], _parameters[3][4], _parameters[3][5]);

    double inv_dep_i = _parameters[4][0];

    double td = _parameters[5][0];

    int a = k / 3, b = k % 3;
    Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

    if (a == 0)
      Pi += delta;
    else if (a == 1)
      Qi = Qi * Utility::deltaQ(delta);
    else if (a == 2)
      Pj += delta;
    else if (a == 3)
      Qj = Qj * Utility::deltaQ(delta);
    else if (a == 4)
      tic += delta;
    else if (a == 5)
      qic = qic * Utility::deltaQ(delta);
    else if (a == 6)
      tic2 += delta;
    else if (a == 7)
      qic2 = qic2 * Utility::deltaQ(delta);
    else if (a == 8) {
      if (b == 0)
        inv_dep_i += delta.x();
      else
        td += delta.y();
    }

    Eigen::Vector3d pts_i_td, pts_j_td;
    pts_i_td = pts_i_ - (td - td_i_) * velocity_i_;
    pts_j_td = pts_j_ - (td - td_j_) * velocity_j_;

    Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Eigen::Vector3d pts_camera_j = qic2.inverse() * (pts_imu_j - tic2);

    Eigen::Vector2d tmp_residual;
#ifdef UNIT_SPHERE_ERROR
    tmp_residual = tangent_base * (pts_camera_j.normalized() - pts_j_td.normalized());
#else
    double dep_j = pts_camera_j.z();
    tmp_residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();
#endif
    tmp_residual = sqrt_info_ * tmp_residual;
    num_jacobian.col(k) = (tmp_residual - residual) / eps;
  }
  std::cout << num_jacobian.block<2, 6>(0, 0) << '\n';
  std::cout << num_jacobian.block<2, 6>(0, 6) << '\n';
  std::cout << num_jacobian.block<2, 6>(0, 12) << '\n';
  std::cout << num_jacobian.block<2, 6>(0, 18) << '\n';
  std::cout << num_jacobian.block<2, 1>(0, 24) << '\n';
  std::cout << num_jacobian.block<2, 1>(0, 25) << '\n';
}
