/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <ceres/ceres.h>
#include <cstdlib>
#include <unordered_map>
#include <vins_common/ros_utils.hpp>

#include "../utility/tic_toc.h"
#include "../utility/utility.h"

constexpr int MAX_MARGINALIZATION_THREADS = 8;

struct ResidualBlockInfo_s {
  ResidualBlockInfo_s(
    ceres::CostFunction * _cost_function, ceres::LossFunction * _loss_function,
    std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
    : cost_function_(_cost_function), loss_function_(_loss_function), parameter_blocks_(_parameter_blocks),
      drop_set_(_drop_set)
  {
  }

  void evaluate();

  ceres::CostFunction * cost_function_;
  ceres::LossFunction * loss_function_;
  std::vector<double *> parameter_blocks_;
  std::vector<int> drop_set_;

  double ** raw_jacobians_;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians_;
  Eigen::VectorXd residuals_;

  int localSize(int _size) { return _size == 7 ? 6 : _size; }
};

struct ThreadsStruct_s {
  std::vector<ResidualBlockInfo_s *> sub_factors_;
  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;
  const std::unordered_map<long, int> * parameter_block_size_ = nullptr;
  const std::unordered_map<long, int> * parameter_block_idx_ = nullptr;
};

class MarginalizationInfo {
public:
  MarginalizationInfo() { valid_ = true; };
  ~MarginalizationInfo();
  int localSize(int _size) const;
  int globalSize(int _size) const;
  void addResidualBlockInfo(ResidualBlockInfo_s * _residual_block_info);
  void preMarginalize();
  void marginalize();
  std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> & _addr_shift);

  std::vector<ResidualBlockInfo_s *> factors_;
  int m_, n_;
  std::unordered_map<long, int> parameter_block_size_;  //global size
  int sum_block_size_;
  std::unordered_map<long, int> parameter_block_idx_;  //local size
  std::unordered_map<long, double *> parameter_block_data_;

  std::vector<int> keep_block_size_;  //global size
  std::vector<int> keep_block_idx_;   //local size
  std::vector<double *> keep_block_data_;

  Eigen::MatrixXd linearized_jacobians_;
  Eigen::VectorXd linearized_residuals_;
  const double eps_ = 1e-8;
  bool valid_;
};

class MarginalizationFactor : public ceres::CostFunction {
public:
  MarginalizationFactor(MarginalizationInfo * _marginalization_info);
  bool Evaluate(double const * const * _parameters, double * _residuals, double ** _jacobians) const override;

  MarginalizationInfo * marginalization_info_;
};
