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
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <vins_common/ros_utils.hpp>

class InitialBiasFactor : public ceres::SizedCostFunction<6, 9> {
public:
  InitialBiasFactor(const Eigen::Vector3d & _Ba, const Eigen::Vector3d & _Bg)
  {
    init_Ba_ = _Ba;
    init_Bg_ = _Bg;
    sqrt_info_ = 1.0 / (0.001) * Eigen::Matrix<double, 6, 6>::Identity();
  }
  virtual bool Evaluate(double const * const * _parameters, double * _residuals, double ** _jacobians) const
  {
    Eigen::Vector3d Ba(_parameters[0][3], _parameters[0][4], _parameters[0][5]);
    Eigen::Vector3d Bg(_parameters[0][6], _parameters[0][7], _parameters[0][8]);

    Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(_residuals);
    residual.block<3, 1>(0, 0) = Ba - init_Ba_;
    residual.block<3, 1>(3, 0) = Bg - init_Bg_;
    residual = sqrt_info_ * residual;

    if (_jacobians) {
      if (_jacobians[0]) {
        Eigen::Map<Eigen::Matrix<double, 6, 9, Eigen::RowMajor>> jacobian_bias(_jacobians[0]);
        jacobian_bias.setZero();
        jacobian_bias.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
        jacobian_bias.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();
        jacobian_bias = sqrt_info_ * jacobian_bias;
      }
    }
    return true;
  }

  Eigen::Vector3d init_Ba_, init_Bg_;
  Eigen::Matrix<double, 6, 6> sqrt_info_;
};
