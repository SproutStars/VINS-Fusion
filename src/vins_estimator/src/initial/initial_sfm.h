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
#include <cstdlib>
#include <deque>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
using namespace Eigen;
using namespace std;



struct SfmFeature_s {
  bool state_;
  int id_;
  vector<pair<int, Vector2d>> observation_;
  double position_[3];
  double depth_;
};

struct ReprojectionError3D_s {
  ReprojectionError3D_s(double _observed_u, double _observed_v)
    : observed_u_(_observed_u), observed_v_(_observed_v)
  {
  }

  template <typename T>
  bool operator()(const T * const _camera_R, const T * const _camera_T, const T * _point, T * _residuals) const
  {
    T p[3];
    ceres::QuaternionRotatePoint(_camera_R, _point, p);
    p[0] += _camera_T[0];
    p[1] += _camera_T[1];
    p[2] += _camera_T[2];
    T xp = p[0] / p[2];
    T yp = p[1] / p[2];
    _residuals[0] = xp - T(observed_u_);
    _residuals[1] = yp - T(observed_v_);
    return true;
  }

  static ceres::CostFunction * create(const double _observed_x, const double _observed_y)
  {
    return (new ceres::AutoDiffCostFunction<ReprojectionError3D_s, 2, 4, 3, 3>(
      new ReprojectionError3D_s(_observed_x, _observed_y)));
  }

  double observed_u_;
  double observed_v_;
};

class GlobalSFM {
public:
  GlobalSFM();
  bool construct(
    int _frame_num, Quaterniond * _q, Vector3d * _T, int _l, const Matrix3d _relative_R,
    const Vector3d _relative_T, vector<SfmFeature_s> & _sfm_f, map<int, Vector3d> & _sfm_tracked_points);

private:
  bool solveFrameByPnP(Matrix3d & _R_initial, Vector3d & _P_initial, int _i, vector<SfmFeature_s> & _sfm_f) const;

  void triangulatePoint(
    Eigen::Matrix<double, 3, 4> & _Pose0, Eigen::Matrix<double, 3, 4> & _Pose1, Vector2d & _point0,
    Vector2d & _point1, Vector3d & _point_3d);
  void triangulateTwoFrames(
    int _frame0, Eigen::Matrix<double, 3, 4> & _Pose0, int _frame1, Eigen::Matrix<double, 3, 4> & _Pose1,
    vector<SfmFeature_s> & _sfm_f);

  int feature_num_;
};