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
#include <vector>
using namespace std;

#include <opencv2/opencv.hpp>

#include <eigen3/Eigen/Dense>
using namespace Eigen;
#include <vins_common/ros_utils.hpp>

/* This class help you to calibrate extrinsic rotation between imu and camera when your totally don't konw the extrinsic parameter */
class InitialEXRotation {
public:
  InitialEXRotation();
  bool calibrationExRotation(
    vector<pair<Vector3d, Vector3d>> _corres, Quaterniond _delta_q_imu, Matrix3d & _calib_ric_result);

private:
  Matrix3d solveRelativeR(const vector<pair<Vector3d, Vector3d>> & _corres);

  double testTriangulation(
    const vector<cv::Point2f> & _l, const vector<cv::Point2f> & _r, cv::Mat_<double> _R, cv::Mat_<double> _t);
  void decomposeE(
    cv::Mat _E, cv::Mat_<double> & _R1, cv::Mat_<double> & _R2, cv::Mat_<double> & _t1, cv::Mat_<double> & _t2);
  int frame_count_;

  vector<Matrix3d> Rc_;
  vector<Matrix3d> Rimu_;
  vector<Matrix3d> Rc_g_;
  Matrix3d ric_;
};
