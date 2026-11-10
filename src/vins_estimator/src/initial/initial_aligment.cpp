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

#include <vins_common/ros_utils.hpp>

#include "initial_alignment.h"

void solveGyroscopeBias(map<double, ImageFrame> & _all_image_frame, Vector3d * _Bgs)
{
  Matrix3d A;
  Vector3d b;
  Vector3d delta_bg;
  A.setZero();
  b.setZero();
  map<double, ImageFrame>::iterator frame_i;
  map<double, ImageFrame>::iterator frame_j;
  for (frame_i = _all_image_frame.begin(); next(frame_i) != _all_image_frame.end(); frame_i++) {
    frame_j = next(frame_i);
    MatrixXd tmp_A(3, 3);
    tmp_A.setZero();
    VectorXd tmp_b(3);
    tmp_b.setZero();
    Eigen::Quaterniond q_ij(frame_i->second.R_.transpose() * frame_j->second.R_);
    tmp_A = frame_j->second.pre_integration_->jacobian_.template block<3, 3>(O_R, O_BG);
    tmp_b = 2 * (frame_j->second.pre_integration_->delta_q_.inverse() * q_ij).vec();
    A += tmp_A.transpose() * tmp_A;
    b += tmp_A.transpose() * tmp_b;
  }
  delta_bg = A.ldlt().solve(b);
  RCLCPP_WARN_STREAM(vins::logger(), "gyroscope bias initial calibration " << delta_bg.transpose());

  for (int i = 0; i <= WINDOW_SIZE; i++)
    _Bgs[i] += delta_bg;

  for (frame_i = _all_image_frame.begin(); next(frame_i) != _all_image_frame.end(); frame_i++) {
    frame_j = next(frame_i);
    frame_j->second.pre_integration_->repropagate(Vector3d::Zero(), _Bgs[0]);
  }
}


MatrixXd TangentBasis(Vector3d & _g0)
{
  Vector3d b, c;
  Vector3d a = _g0.normalized();
  Vector3d tmp(0, 0, 1);
  if (a == tmp)
    tmp << 1, 0, 0;
  b = (tmp - a * (a.transpose() * tmp)).normalized();
  c = a.cross(b);
  MatrixXd bc(3, 2);
  bc.block<3, 1>(0, 0) = b;
  bc.block<3, 1>(0, 1) = c;
  return bc;
}

void RefineGravity(map<double, ImageFrame> & _all_image_frame, Vector3d & _g, VectorXd & _x)
{
  Vector3d g0 = _g.normalized() * G.norm();
  Vector3d lx, ly;
  //VectorXd x;
  int all_frame_count = _all_image_frame.size();
  int n_state = all_frame_count * 3 + 2 + 1;

  MatrixXd A{n_state, n_state};
  A.setZero();
  VectorXd b{n_state};
  b.setZero();

  map<double, ImageFrame>::iterator frame_i;
  map<double, ImageFrame>::iterator frame_j;
  for (int k = 0; k < 4; k++) {
    MatrixXd lxly(3, 2);
    lxly = TangentBasis(g0);
    int i = 0;
    for (frame_i = _all_image_frame.begin(); next(frame_i) != _all_image_frame.end(); frame_i++, i++) {
      frame_j = next(frame_i);

      MatrixXd tmp_A(6, 9);
      tmp_A.setZero();
      VectorXd tmp_b(6);
      tmp_b.setZero();

      double dt = frame_j->second.pre_integration_->sum_dt_;


      tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
      tmp_A.block<3, 2>(0, 6) = frame_i->second.R_.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
      tmp_A.block<3, 1>(0, 8) =
        frame_i->second.R_.transpose() * (frame_j->second.T_ - frame_i->second.T_) / 100.0;
      tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration_->delta_p_ +
                                frame_i->second.R_.transpose() * frame_j->second.R_ * TIC[0] - TIC[0] -
                                frame_i->second.R_.transpose() * dt * dt / 2 * g0;

      tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
      tmp_A.block<3, 3>(3, 3) = frame_i->second.R_.transpose() * frame_j->second.R_;
      tmp_A.block<3, 2>(3, 6) = frame_i->second.R_.transpose() * dt * Matrix3d::Identity() * lxly;
      tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration_->delta_v_ -
                                frame_i->second.R_.transpose() * dt * Matrix3d::Identity() * g0;


      Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
      //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
      //MatrixXd cov_inv = cov.inverse();
      cov_inv.setIdentity();

      MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
      VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

      A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
      b.segment<6>(i * 3) += r_b.head<6>();

      A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
      b.tail<3>() += r_b.tail<3>();

      A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
      A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    _x = A.ldlt().solve(b);
    VectorXd dg = _x.segment<2>(n_state - 3);
    g0 = (g0 + lxly * dg).normalized() * G.norm();
    //double s = x(n_state - 1);
  }
  _g = g0;
}

bool LinearAlignment(map<double, ImageFrame> & _all_image_frame, Vector3d & _g, VectorXd & _x)
{
  int all_frame_count = _all_image_frame.size();
  int n_state = all_frame_count * 3 + 3 + 1;

  MatrixXd A{n_state, n_state};
  A.setZero();
  VectorXd b{n_state};
  b.setZero();

  map<double, ImageFrame>::iterator frame_i;
  map<double, ImageFrame>::iterator frame_j;
  int i = 0;
  for (frame_i = _all_image_frame.begin(); next(frame_i) != _all_image_frame.end(); frame_i++, i++) {
    frame_j = next(frame_i);

    MatrixXd tmp_A(6, 10);
    tmp_A.setZero();
    VectorXd tmp_b(6);
    tmp_b.setZero();

    double dt = frame_j->second.pre_integration_->sum_dt_;

    tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
    tmp_A.block<3, 3>(0, 6) = frame_i->second.R_.transpose() * dt * dt / 2 * Matrix3d::Identity();
    tmp_A.block<3, 1>(0, 9) =
      frame_i->second.R_.transpose() * (frame_j->second.T_ - frame_i->second.T_) / 100.0;
    tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration_->delta_p_ +
                              frame_i->second.R_.transpose() * frame_j->second.R_ * TIC[0] - TIC[0];
    //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
    tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
    tmp_A.block<3, 3>(3, 3) = frame_i->second.R_.transpose() * frame_j->second.R_;
    tmp_A.block<3, 3>(3, 6) = frame_i->second.R_.transpose() * dt * Matrix3d::Identity();
    tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration_->delta_v_;
    //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

    Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
    //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
    //MatrixXd cov_inv = cov.inverse();
    cov_inv.setIdentity();

    MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
    VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

    A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
    b.segment<6>(i * 3) += r_b.head<6>();

    A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
    b.tail<4>() += r_b.tail<4>();

    A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
    A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
  }
  A = A * 1000.0;
  b = b * 1000.0;
  _x = A.ldlt().solve(b);
  double s = _x(n_state - 1) / 100.0;
  RCLCPP_DEBUG(vins::logger(), "estimated scale: %f", s);
  _g = _x.segment<3>(n_state - 4);
  RCLCPP_DEBUG_STREAM(vins::logger(), " result g     " << _g.norm() << " " << _g.transpose());
  if (fabs(_g.norm() - G.norm()) > 0.5 || s < 0) {
    return false;
  }

  RefineGravity(_all_image_frame, _g, _x);
  s = (_x.tail<1>())(0) / 100.0;
  (_x.tail<1>())(0) = s;
  RCLCPP_DEBUG_STREAM(vins::logger(), " refine     " << _g.norm() << " " << _g.transpose());
  return s >= 0.0;
}

bool VisualIMUAlignment(map<double, ImageFrame> & _all_image_frame, Vector3d * _Bgs, Vector3d & _g, VectorXd & _x)
{
  solveGyroscopeBias(_all_image_frame, _Bgs);

  return LinearAlignment(_all_image_frame, _g, _x);
}
