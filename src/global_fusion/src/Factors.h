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
#include <ceres/ceres.h>
#include <ceres/rotation.h>

template <typename T> inline void QuaternionInverse(const T _q[4], T _q_inverse[4])
{
  _q_inverse[0] = _q[0];
  _q_inverse[1] = -_q[1];
  _q_inverse[2] = -_q[2];
  _q_inverse[3] = -_q[3];
};


struct TError_s {
  TError_s(double _t_x, double _t_y, double _t_z, double _var)
    : t_x_(_t_x), t_y_(_t_y), t_z_(_t_z), var_(_var)
  {
  }

  template <typename T> bool operator()(const T * _tj, T * _residuals) const
  {
    _residuals[0] = (_tj[0] - T(t_x_)) / T(var_);
    _residuals[1] = (_tj[1] - T(t_y_)) / T(var_);
    _residuals[2] = (_tj[2] - T(t_z_)) / T(var_);

    return true;
  }

  static ceres::CostFunction * create(const double _t_x, const double _t_y, const double _t_z, const double _var)
  {
    return (new ceres::AutoDiffCostFunction<TError_s, 3, 3>(new TError_s(_t_x, _t_y, _t_z, _var)));
  }

  double t_x_, t_y_, t_z_, var_;
};

struct RelativeRtError_s {
  RelativeRtError_s(
    double _t_x, double _t_y, double _t_z, double _q_w, double _q_x, double _q_y, double _q_z, double _t_var,
    double _q_var)
    : t_x_(_t_x), t_y_(_t_y), t_z_(_t_z), q_w_(_q_w), q_x_(_q_x), q_y_(_q_y), q_z_(_q_z), t_var_(_t_var),
      q_var_(_q_var)
  {
  }

  template <typename T>
  bool operator()(const T * const _w_q_i, const T * _ti, const T * _w_q_j, const T * _tj, T * _residuals) const
  {
    T t_w_ij[3];
    t_w_ij[0] = _tj[0] - _ti[0];
    t_w_ij[1] = _tj[1] - _ti[1];
    t_w_ij[2] = _tj[2] - _ti[2];

    T i_q_w[4];
    QuaternionInverse(_w_q_i, i_q_w);

    T t_i_ij[3];
    ceres::QuaternionRotatePoint(i_q_w, t_w_ij, t_i_ij);

    _residuals[0] = (t_i_ij[0] - T(t_x_)) / T(t_var_);
    _residuals[1] = (t_i_ij[1] - T(t_y_)) / T(t_var_);
    _residuals[2] = (t_i_ij[2] - T(t_z_)) / T(t_var_);

    T relative_q[4];
    relative_q[0] = T(q_w_);
    relative_q[1] = T(q_x_);
    relative_q[2] = T(q_y_);
    relative_q[3] = T(q_z_);

    T q_i_j[4];
    ceres::QuaternionProduct(i_q_w, _w_q_j, q_i_j);

    T relative_q_inv[4];
    QuaternionInverse(relative_q, relative_q_inv);

    T error_q[4];
    ceres::QuaternionProduct(relative_q_inv, q_i_j, error_q);

    _residuals[3] = T(2) * error_q[1] / T(q_var_);
    _residuals[4] = T(2) * error_q[2] / T(q_var_);
    _residuals[5] = T(2) * error_q[3] / T(q_var_);

    return true;
  }

  static ceres::CostFunction * create(
    const double _t_x, const double _t_y, const double _t_z, const double _q_w, const double _q_x,
    const double _q_y, const double _q_z, const double _t_var, const double _q_var)
  {
    return (new ceres::AutoDiffCostFunction<RelativeRtError_s, 6, 4, 3, 4, 3>(
      new RelativeRtError_s(_t_x, _t_y, _t_z, _q_w, _q_x, _q_y, _q_z, _t_var, _q_var)));
  }

  double t_x_, t_y_, t_z_, t_norm_;
  double q_w_, q_x_, q_y_, q_z_;
  double t_var_, q_var_;
};