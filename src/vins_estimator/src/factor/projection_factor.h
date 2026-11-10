/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"
#include "../utility/utility.h"
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <vins_common/ros_utils.hpp>

class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1> {
public:
  ProjectionFactor(const Eigen::Vector3d & _pts_i, const Eigen::Vector3d & _pts_j);
  bool Evaluate(double const * const * _parameters, double * _residuals, double ** _jacobians) const override;
  void check(double ** _parameters);

  Eigen::Vector3d pts_i_, pts_j_;
  Eigen::Matrix<double, 2, 3> tangent_base_;
  static Eigen::Matrix2d sqrt_info_;
  static double sum_t_;
};
