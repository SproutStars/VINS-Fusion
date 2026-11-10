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

#include <vector>
using namespace std;

#include <opencv2/opencv.hpp>
//#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <vins_common/ros_utils.hpp>

class MotionEstimator {
public:
  bool solveRelativeRT(const vector<pair<Vector3d, Vector3d>> & _corres, Matrix3d & _R, Vector3d & _T);

private:
  double testTriangulation(
    const vector<cv::Point2f> & _l, const vector<cv::Point2f> & _r, cv::Mat_<double> _R, cv::Mat_<double> _t);
  void decomposeE(
    cv::Mat _E, cv::Mat_<double> & _R1, cv::Mat_<double> & _R2, cv::Mat_<double> & _t1, cv::Mat_<double> & _t2);
};
