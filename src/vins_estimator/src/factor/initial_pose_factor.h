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

#pragma once

#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"
#include "../utility/utility.h"
#include "pose_local_parameterization.h"
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <vins_common/ros_utils.hpp>

class InitialPoseFactor : public ceres::SizedCostFunction<6, 7> {
public:
  InitialPoseFactor(const Eigen::Vector3d & _P, const Eigen::Quaterniond & _Q)
  {
    init_P_ = _P;
    init_Q_ = _Q;
    sqrt_info_ = 1000 * Eigen::Matrix<double, 6, 6>::Identity();
  }
  bool Evaluate(double const * const * _parameters, double * _residuals, double ** _jacobians) const override
  {
    Eigen::Vector3d P(_parameters[0][0], _parameters[0][1], _parameters[0][2]);
    Eigen::Quaterniond Q(_parameters[0][6], _parameters[0][3], _parameters[0][4], _parameters[0][5]);

    Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(_residuals);
    residual.block<3, 1>(0, 0) = P - init_P_;
    residual.block<3, 1>(3, 0) = 2 * (init_Q_.inverse() * Q).vec();
    residual = sqrt_info_ * residual;

    if (_jacobians) {
      if (_jacobians[0]) {
        Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose(_jacobians[0]);
        jacobian_pose.setZero();
        jacobian_pose.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        jacobian_pose.block<3, 3>(3, 3) = Utility::qleft(init_Q_.inverse() * Q).bottomRightCorner<3, 3>();
        jacobian_pose = sqrt_info_ * jacobian_pose;
        PoseManifold::liftJacobian(_parameters[0], _jacobians[0], 6);
      }
    }
    return true;
  }

  void check(double ** _parameters)
  {
    double * res = new double[6];
    double ** jaco = new double *[1];
    jaco[0] = new double[6 * 7];
    Evaluate(_parameters, res, jaco);
    puts("check begins");

    puts("my");

    std::cout << Eigen::Map<Eigen::Matrix<double, 6, 1>>(res).transpose() << '\n' << '\n';
    std::cout << Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>>(jaco[0]) << '\n' << '\n';

    Eigen::Vector3d P(_parameters[0][0], _parameters[0][1], _parameters[0][2]);
    Eigen::Quaterniond Q(_parameters[0][6], _parameters[0][3], _parameters[0][4], _parameters[0][5]);

    Eigen::Matrix<double, 6, 1> residual;
    residual.block<3, 1>(0, 0) = P - init_P_;
    residual.block<3, 1>(3, 0) = 2 * (init_Q_.inverse() * Q).vec();
    residual = sqrt_info_ * residual;

    puts("num");
    std::cout << residual.transpose() << '\n';

    const double eps = 1e-6;
    Eigen::Matrix<double, 6, 6> num_jacobian;
    for (int k = 0; k < 6; k++) {
      Eigen::Vector3d P(_parameters[0][0], _parameters[0][1], _parameters[0][2]);
      Eigen::Quaterniond Q(_parameters[0][6], _parameters[0][3], _parameters[0][4], _parameters[0][5]);

      int a = k / 3, b = k % 3;
      Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

      if (a == 0)
        P += delta;
      else if (a == 1)
        Q = Q * Utility::deltaQ(delta);


      Eigen::Matrix<double, 6, 1> tmp_residual;
      tmp_residual.block<3, 1>(0, 0) = P - init_P_;
      tmp_residual.block<3, 1>(3, 0) = 2 * (init_Q_.inverse() * Q).vec();
      tmp_residual = sqrt_info_ * tmp_residual;

      num_jacobian.col(k) = (tmp_residual - residual) / eps;
    }
    std::cout << num_jacobian << '\n';
  }

  Eigen::Vector3d init_P_;
  Eigen::Quaterniond init_Q_;
  Eigen::Matrix<double, 6, 6> sqrt_info_;
};
