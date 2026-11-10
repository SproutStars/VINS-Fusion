/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "pose_local_parameterization.h"
#include <vins_common/ros_utils.hpp>

#include "../estimator/parameters.h"
#include "../utility/runtime_utils.h"
#include "marginalization_factor.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>

void ResidualBlockInfo_s::evaluate()
{
  residuals_.resize(cost_function_->num_residuals());

  std::vector<int> block_sizes = cost_function_->parameter_block_sizes();
  raw_jacobians_ = new double *[block_sizes.size()];
  jacobians_.resize(block_sizes.size());

  for (int i = 0; i < static_cast<int>(block_sizes.size()); i++) {
    jacobians_[i].resize(cost_function_->num_residuals(), block_sizes[i]);
    raw_jacobians_[i] = jacobians_[i].data();
    //dim += block_sizes[i] == 7 ? 6 : block_sizes[i];
  }
  cost_function_->Evaluate(parameter_blocks_.data(), residuals_.data(), raw_jacobians_);
  // Convert ambient Jacobians back to tangent coordinates for the Schur prior.
  for (size_t i = 0; i < block_sizes.size(); ++i) {
    if (block_sizes[i] == 7) {
      const Eigen::MatrixXd local = jacobians_[i] * PoseManifold::plusJacobian(parameter_blocks_[i]);
      jacobians_[i].setZero();
      jacobians_[i].leftCols(6) = local;
    }
  }


  //std::vector<int> tmp_idx(block_sizes.size());
  //Eigen::MatrixXd tmp(dim, dim);
  //for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
  //{
  //    int size_i = localSize(block_sizes[i]);
  //    Eigen::MatrixXd jacobian_i = jacobians[i].leftCols(size_i);
  //    for (int j = 0, sub_idx = 0; j < static_cast<int>(parameter_blocks.size()); sub_idx += block_sizes[j] == 7 ? 6 : block_sizes[j], j++)
  //    {
  //        int size_j = localSize(block_sizes[j]);
  //        Eigen::MatrixXd jacobian_j = jacobians[j].leftCols(size_j);
  //        tmp_idx[j] = sub_idx;
  //        tmp.block(tmp_idx[i], tmp_idx[j], size_i, size_j) = jacobian_i.transpose() * jacobian_j;
  //    }
  //}
  //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(tmp);
  //std::cout << saes.eigenvalues() << std::endl;
  //assert(saes.eigenvalues().minCoeff() >= -1e-6);

  if (loss_function_) {
    double residual_scaling_, alpha_sq_norm_;

    double sq_norm, rho[3];

    sq_norm = residuals_.squaredNorm();
    loss_function_->Evaluate(sq_norm, rho);
    //printf("sq_norm: %f, rho[0]: %f, rho[1]: %f, rho[2]: %f\n", sq_norm, rho[0], rho[1], rho[2]);

    double sqrt_rho1_ = sqrt(rho[1]);

    if ((sq_norm == 0.0) || (rho[2] <= 0.0)) {
      residual_scaling_ = sqrt_rho1_;
      alpha_sq_norm_ = 0.0;
    } else {
      const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
      const double alpha = 1.0 - sqrt(D);
      residual_scaling_ = sqrt_rho1_ / (1 - alpha);
      alpha_sq_norm_ = alpha / sq_norm;
    }

    for (int i = 0; i < static_cast<int>(parameter_blocks_.size()); i++) {
      jacobians_[i] =
        sqrt_rho1_ * (jacobians_[i] - alpha_sq_norm_ * residuals_ * (residuals_.transpose() * jacobians_[i]));
    }

    residuals_ *= residual_scaling_;
  }
}

MarginalizationInfo::~MarginalizationInfo()
{
  //RCLCPP_WARN(vins::logger(), "release marginlizationinfo");

  for (auto & it : parameter_block_data_)
    delete[] it.second;

  for (auto & factor : factors_) {

    delete[] factor->raw_jacobians_;

    delete factor->cost_function_;

    delete factor;
  }
}

void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo_s * _residual_block_info)
{
  factors_.emplace_back(_residual_block_info);

  std::vector<double *> & parameter_blocks = _residual_block_info->parameter_blocks_;
  std::vector<int> parameter_block_sizes = _residual_block_info->cost_function_->parameter_block_sizes();

  for (int i = 0; i < static_cast<int>(_residual_block_info->parameter_blocks_.size()); i++) {
    double * addr = parameter_blocks[i];
    int size = parameter_block_sizes[i];
    parameter_block_size_[reinterpret_cast<long>(addr)] = size;
  }

  for (int i : _residual_block_info->drop_set_) {
    double * addr = parameter_blocks[i];
    parameter_block_idx_[reinterpret_cast<long>(addr)] = 0;
  }
}

void MarginalizationInfo::preMarginalize()
{
  for (auto it : factors_) {
    it->evaluate();

    std::vector<int> block_sizes = it->cost_function_->parameter_block_sizes();
    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++) {
      long addr = reinterpret_cast<long>(it->parameter_blocks_[i]);
      int size = block_sizes[i];
      if (parameter_block_data_.find(addr) == parameter_block_data_.end()) {
        double * data = new double[size];
        memcpy(data, it->parameter_blocks_[i], sizeof(double) * size);
        parameter_block_data_[addr] = data;
      }
    }
  }
}

int MarginalizationInfo::localSize(int _size) const { return _size == 7 ? 6 : _size; }

int MarginalizationInfo::globalSize(int _size) const { return _size == 6 ? 7 : _size; }

void constructNormalEquations(ThreadsStruct_s * _p)
{
  for (auto it : _p->sub_factors_) {
    for (int i = 0; i < static_cast<int>(it->parameter_blocks_.size()); i++) {
      int idx_i = _p->parameter_block_idx_->at(reinterpret_cast<long>(it->parameter_blocks_[i]));
      int size_i = _p->parameter_block_size_->at(reinterpret_cast<long>(it->parameter_blocks_[i]));
      if (size_i == 7)
        size_i = 6;
      const auto jacobian_i = it->jacobians_[i].leftCols(size_i);
      for (int j = i; j < static_cast<int>(it->parameter_blocks_.size()); j++) {
        int idx_j = _p->parameter_block_idx_->at(reinterpret_cast<long>(it->parameter_blocks_[j]));
        int size_j = _p->parameter_block_size_->at(reinterpret_cast<long>(it->parameter_blocks_[j]));
        if (size_j == 7)
          size_j = 6;
        const auto jacobian_j = it->jacobians_[j].leftCols(size_j);
        if (i == j)
          _p->A_.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
        else {
          _p->A_.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
          _p->A_.block(idx_j, idx_i, size_j, size_i) = _p->A_.block(idx_i, idx_j, size_i, size_j).transpose();
        }
      }
      _p->b_.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals_;
    }
  }
}

void MarginalizationInfo::marginalize()
{
  int pos = 0;
  for (auto & it : parameter_block_idx_) {
    it.second = pos;
    pos += localSize(parameter_block_size_[it.first]);
  }

  m_ = pos;

  for (const auto & it : parameter_block_size_) {
    if (parameter_block_idx_.find(it.first) == parameter_block_idx_.end()) {
      parameter_block_idx_[it.first] = pos;
      pos += localSize(it.second);
    }
  }

  n_ = pos - m_;
  //RCLCPP_INFO(vins::logger(), "marginalization, pos: %d, m: %d, n: %d, size: %d", pos, m, n, (int)parameter_block_idx.size());
  if (m_ == 0) {
    valid_ = false;
    printf("unstable tracking...\n");
    return;
  }

  TicToc t_summing;
  Eigen::MatrixXd A(pos, pos);
  Eigen::VectorXd b(pos);
  A.setZero();
  b.setZero();
  /*
    for (auto it : factors)
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])]);
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])]);
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    A.block(idx_j, idx_i, size_j, size_i) = A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    RCLCPP_INFO(vins::logger(), "summing up costs %f ms", t_summing.toc());
    */
  //multi thread


  // Each estimator processing thread owns its own pool and reusable buffers.
  // A pool never retains factor pointers beyond this synchronous barrier.
  struct Workspace_s {
    int count_ = 0;
    std::array<ThreadsStruct_s, MAX_MARGINALIZATION_THREADS> partial_;
    std::unique_ptr<vins::WorkerPool> pool_;
  };
  thread_local Workspace_s workspace;

  const int num_threads = std::clamp(MARGINALIZATION_THREADS, 1, MAX_MARGINALIZATION_THREADS);
  const size_t num_workers = static_cast<size_t>(num_threads);
  if (workspace.count_ != num_threads) {
    workspace.pool_.reset();
    workspace.count_ = 0;
    if (num_threads > 1)
      workspace.pool_ = std::make_unique<vins::WorkerPool>(num_workers);
    workspace.count_ = num_threads;
  }

  auto & partial = workspace.partial_;
  const size_t factors_per_worker = (factors_.size() + num_workers - 1) / num_workers;
  for (size_t i = 0; i < num_workers; ++i) {
    ThreadsStruct_s & worker = partial[i];
    worker.sub_factors_.clear();
    if (worker.sub_factors_.capacity() < factors_per_worker)
      worker.sub_factors_.reserve(factors_per_worker);
    worker.A_.setZero(pos, pos);
    worker.b_.setZero(pos);
    worker.parameter_block_size_ = &parameter_block_size_;
    worker.parameter_block_idx_ = &parameter_block_idx_;
  }

  size_t worker = 0;
  for (auto * factor : factors_) {
    partial[worker].sub_factors_.push_back(factor);
    worker = (worker + 1) % num_workers;
  }

  if (num_threads == 1) {
    constructNormalEquations(&partial[0]);
  } else {
    workspace.pool_->run([&](size_t _index) { constructNormalEquations(&partial[_index]); });
  }

  for (size_t i = num_workers; i-- > 0;) {
    A += partial[i].A_;
    b += partial[i].b_;
    partial[i].sub_factors_.clear();
  }
  //RCLCPP_DEBUG(vins::logger(), "thread summing up costs %f ms", t_thread_summing.toc());
  //RCLCPP_INFO(vins::logger(), "A diff %f , b diff %f ", (A - tmp_A).sum(), (b - tmp_b).sum());


  //TODO
  Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m_, m_) + A.block(0, 0, m_, m_).transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);

  //assert(saes.eigenvalues().minCoeff() >= -1e-4, "min eigenvalue %f", saes.eigenvalues().minCoeff());

  Eigen::MatrixXd Amm_inv =
    saes.eigenvectors() *
    Eigen::VectorXd((saes.eigenvalues().array() > eps_).select(saes.eigenvalues().array().inverse(), 0))
      .asDiagonal() *
    saes.eigenvectors().transpose();
  //printf("error1: %f\n", (Amm * Amm_inv - Eigen::MatrixXd::Identity(m, m)).sum());

  Eigen::VectorXd bmm = b.segment(0, m_);
  Eigen::MatrixXd Amr = A.block(0, m_, m_, n_);
  Eigen::MatrixXd Arm = A.block(m_, 0, n_, m_);
  Eigen::MatrixXd Arr = A.block(m_, m_, n_, n_);
  Eigen::VectorXd brr = b.segment(m_, n_);
  A = Arr - Arm * Amm_inv * Amr;
  b = brr - Arm * Amm_inv * bmm;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
  Eigen::VectorXd S =
    Eigen::VectorXd((saes2.eigenvalues().array() > eps_).select(saes2.eigenvalues().array(), 0));
  Eigen::VectorXd S_inv =
    Eigen::VectorXd((saes2.eigenvalues().array() > eps_).select(saes2.eigenvalues().array().inverse(), 0));

  Eigen::VectorXd S_sqrt = S.cwiseSqrt();
  Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

  linearized_jacobians_ = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
  linearized_residuals_ = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
  //std::cout << A << std::endl
  //          << std::endl;
  //std::cout << linearized_jacobians << std::endl;
  //printf("error2: %f %f\n", (linearized_jacobians.transpose() * linearized_jacobians - A).sum(),
  //      (linearized_jacobians.transpose() * linearized_residuals - b).sum());
}

std::vector<double *> MarginalizationInfo::getParameterBlocks(std::unordered_map<long, double *> & _addr_shift)
{
  std::vector<double *> keep_block_addr;
  keep_block_size_.clear();
  keep_block_idx_.clear();
  keep_block_data_.clear();

  for (const auto & it : parameter_block_idx_) {
    if (it.second >= m_) {
      keep_block_size_.push_back(parameter_block_size_[it.first]);
      keep_block_idx_.push_back(parameter_block_idx_[it.first]);
      keep_block_data_.push_back(parameter_block_data_[it.first]);
      keep_block_addr.push_back(_addr_shift[it.first]);
    }
  }
  sum_block_size_ = std::accumulate(std::begin(keep_block_size_), std::end(keep_block_size_), 0);

  return keep_block_addr;
}

MarginalizationFactor::MarginalizationFactor(MarginalizationInfo * _marginalization_info)
  : marginalization_info_(_marginalization_info)
{
  int cnt = 0;
  for (auto it : marginalization_info_->keep_block_size_) {
    mutable_parameter_block_sizes()->push_back(it);
    cnt += it;
  }
  //printf("residual size: %d, %d\n", cnt, n);
  set_num_residuals(marginalization_info_->n_);
};

bool MarginalizationFactor::Evaluate(
  double const * const * _parameters, double * _residuals, double ** _jacobians) const
{
  //printf("internal addr,%d, %d\n", (int)parameter_block_sizes().size(), num_residuals());
  //for (int i = 0; i < static_cast<int>(keep_block_size.size()); i++)
  //{
  //    //printf("unsigned %x\n", reinterpret_cast<unsigned long>(parameters[i]));
  //    //printf("signed %x\n", reinterpret_cast<long>(parameters[i]));
  //printf("jacobian %x\n", reinterpret_cast<long>(jacobians));
  //printf("residual %x\n", reinterpret_cast<long>(residuals));
  //}
  int n = marginalization_info_->n_;
  int m = marginalization_info_->m_;
  Eigen::VectorXd dx(n);
  for (int i = 0; i < static_cast<int>(marginalization_info_->keep_block_size_.size()); i++) {
    int size = marginalization_info_->keep_block_size_[i];
    int idx = marginalization_info_->keep_block_idx_[i] - m;
    Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(_parameters[i], size);
    Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info_->keep_block_data_[i], size);
    if (size != 7)
      dx.segment(idx, size) = x - x0;
    else {
      dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
      dx.segment<3>(idx + 3) = 2.0 * Utility::positify(
                                       Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() *
                                       Eigen::Quaterniond(x(6), x(3), x(4), x(5)))
                                       .vec();
      if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() *
             Eigen::Quaterniond(x(6), x(3), x(4), x(5)))
              .w() >= 0)) {
        dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(
                                          Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() *
                                          Eigen::Quaterniond(x(6), x(3), x(4), x(5)))
                                          .vec();
      }
    }
  }
  Eigen::Map<Eigen::VectorXd>(_residuals, n) =
    marginalization_info_->linearized_residuals_ + marginalization_info_->linearized_jacobians_ * dx;
  if (_jacobians) {

    for (int i = 0; i < static_cast<int>(marginalization_info_->keep_block_size_.size()); i++) {
      if (_jacobians[i]) {
        int size = marginalization_info_->keep_block_size_[i],
            local_size = marginalization_info_->localSize(size);
        int idx = marginalization_info_->keep_block_idx_[i] - m;
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(
          _jacobians[i], n, size);
        jacobian.setZero();
        jacobian.leftCols(local_size) =
          marginalization_info_->linearized_jacobians_.middleCols(idx, local_size);
        if (size == 7)
          PoseManifold::liftJacobian(_parameters[i], _jacobians[i], n);
      }
    }
  }
  return true;
}
