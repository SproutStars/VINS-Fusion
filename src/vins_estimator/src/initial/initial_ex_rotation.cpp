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

#include "initial_ex_rotation.h"

InitialEXRotation::InitialEXRotation()
{
  frame_count_ = 0;
  Rc_.push_back(Matrix3d::Identity());
  Rc_g_.push_back(Matrix3d::Identity());
  Rimu_.push_back(Matrix3d::Identity());
  ric_ = Matrix3d::Identity();
}

bool InitialEXRotation::calibrationExRotation(
  vector<pair<Vector3d, Vector3d>> _corres, Quaterniond _delta_q_imu, Matrix3d & _calib_ric_result)
{
  frame_count_++;
  Rc_.push_back(solveRelativeR(_corres));
  Rimu_.push_back(_delta_q_imu.toRotationMatrix());
  Rc_g_.push_back(ric_.inverse() * _delta_q_imu * ric_);

  Eigen::MatrixXd A(frame_count_ * 4, 4);
  A.setZero();
  int sum_ok = 0;
  for (int i = 1; i <= frame_count_; i++) {
    Quaterniond r1(Rc_[i]);
    Quaterniond r2(Rc_g_[i]);

    double angular_distance = 180 / M_PI * r1.angularDistance(r2);
    RCLCPP_DEBUG(vins::logger(), "%d %f", i, angular_distance);

    double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;
    ++sum_ok;
    Matrix4d L, R;

    double w = Quaterniond(Rc_[i]).w();
    Vector3d q = Quaterniond(Rc_[i]).vec();
    L.block<3, 3>(0, 0) = w * Matrix3d::Identity() + Utility::skewSymmetric(q);
    L.block<3, 1>(0, 3) = q;
    L.block<1, 3>(3, 0) = -q.transpose();
    L(3, 3) = w;

    Quaterniond R_ij(Rimu_[i]);
    w = R_ij.w();
    q = R_ij.vec();
    R.block<3, 3>(0, 0) = w * Matrix3d::Identity() - Utility::skewSymmetric(q);
    R.block<3, 1>(0, 3) = q;
    R.block<1, 3>(3, 0) = -q.transpose();
    R(3, 3) = w;

    A.block<4, 4>((i - 1) * 4, 0) = huber * (L - R);
  }

  JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
  Matrix<double, 4, 1> x = svd.matrixV().col(3);
  Quaterniond estimated_R(x);
  ric_ = estimated_R.toRotationMatrix().inverse();
  //cout << svd.singularValues().transpose() << endl;
  //cout << ric << endl;
  Vector3d ric_cov;
  ric_cov = svd.singularValues().tail<3>();
  if (frame_count_ >= WINDOW_SIZE && ric_cov(1) > 0.25) {
    _calib_ric_result = ric_;
    return true;
  } else
    return false;
}

Matrix3d InitialEXRotation::solveRelativeR(const vector<pair<Vector3d, Vector3d>> & _corres)
{
  if (_corres.size() >= 9) {
    vector<cv::Point2f> ll, rr;
    for (const auto & _corre : _corres) {
      ll.push_back(cv::Point2f(_corre.first(0), _corre.first(1)));
      rr.push_back(cv::Point2f(_corre.second(0), _corre.second(1)));
    }
    cv::Mat E = cv::findFundamentalMat(ll, rr);
    cv::Mat_<double> R1, R2, t1, t2;
    decomposeE(E, R1, R2, t1, t2);

    if (determinant(R1) + 1.0 < 1e-09) {
      E = -E;
      decomposeE(E, R1, R2, t1, t2);
    }
    double ratio1 = max(testTriangulation(ll, rr, R1, t1), testTriangulation(ll, rr, R1, t2));
    double ratio2 = max(testTriangulation(ll, rr, R2, t1), testTriangulation(ll, rr, R2, t2));
    cv::Mat_<double> ans_R_cv = ratio1 > ratio2 ? R1 : R2;

    Matrix3d ans_R_eigen;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        ans_R_eigen(j, i) = ans_R_cv(i, j);
    return ans_R_eigen;
  }
  return Matrix3d::Identity();
}

double InitialEXRotation::testTriangulation(
  const vector<cv::Point2f> & _l, const vector<cv::Point2f> & _r, cv::Mat_<double> _R, cv::Mat_<double> _t)
{
  cv::Mat pointcloud;
  cv::Matx34f P = cv::Matx34f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
  cv::Matx34f P1 = cv::Matx34f(
    _R(0, 0), _R(0, 1), _R(0, 2), _t(0), _R(1, 0), _R(1, 1), _R(1, 2), _t(1), _R(2, 0), _R(2, 1), _R(2, 2),
    _t(2));
  cv::triangulatePoints(P, P1, _l, _r, pointcloud);
  int front_count = 0;
  for (int i = 0; i < pointcloud.cols; i++) {
    double normal_factor = pointcloud.col(i).at<float>(3);

    cv::Mat_<double> p_3d_l = cv::Mat(P) * (pointcloud.col(i) / normal_factor);
    cv::Mat_<double> p_3d_r = cv::Mat(P1) * (pointcloud.col(i) / normal_factor);
    if (p_3d_l(2) > 0 && p_3d_r(2) > 0)
      front_count++;
  }
  RCLCPP_DEBUG(vins::logger(), "MotionEstimator: %f", 1.0 * front_count / pointcloud.cols);
  return 1.0 * front_count / pointcloud.cols;
}

void InitialEXRotation::decomposeE(
  cv::Mat _E, cv::Mat_<double> & _R1, cv::Mat_<double> & _R2, cv::Mat_<double> & _t1, cv::Mat_<double> & _t2)
{
  cv::SVD svd(_E, cv::SVD::MODIFY_A);
  cv::Matx33d W(0, -1, 0, 1, 0, 0, 0, 0, 1);
  cv::Matx33d Wt(0, 1, 0, -1, 0, 0, 0, 0, 1);
  _R1 = svd.u * cv::Mat(W) * svd.vt;
  _R2 = svd.u * cv::Mat(Wt) * svd.vt;
  _t1 = svd.u.col(2);
  _t2 = -svd.u.col(2);
}
