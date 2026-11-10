/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <algorithm>
#include <iterator>
#include <list>
#include <numeric>
#include <unordered_map>
#include <vector>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <vins_common/ros_utils.hpp>

#include "../utility/tic_toc.h"
#include "parameters.h"

class FeaturePerFrame {
public:
  FeaturePerFrame(const Eigen::Matrix<double, 7, 1> & _point, double _td)
  {
    point_.x() = _point(0);
    point_.y() = _point(1);
    point_.z() = _point(2);
    uv_.x() = _point(3);
    uv_.y() = _point(4);
    velocity_.x() = _point(5);
    velocity_.y() = _point(6);
    cur_td_ = _td;
    is_stereo_ = false;
  }
  void rightObservation(const Eigen::Matrix<double, 7, 1> & _point)
  {
    pointRight_.x() = _point(0);
    pointRight_.y() = _point(1);
    pointRight_.z() = _point(2);
    uvRight_.x() = _point(3);
    uvRight_.y() = _point(4);
    velocityRight_.x() = _point(5);
    velocityRight_.y() = _point(6);
    is_stereo_ = true;
  }
  double cur_td_;
  Vector3d point_, pointRight_;
  Vector2d uv_, uvRight_;
  Vector2d velocity_, velocityRight_;
  bool is_stereo_;
};

class FeaturePerId {
public:
  const int feature_id_;
  int start_frame_;
  vector<FeaturePerFrame> feature_per_frame_;
  int used_num_;
  double estimated_depth_;
  int solve_flag_;  // 0 haven't solve yet; 1 solve succ; 2 solve fail;

  FeaturePerId(int _feature_id, int _start_frame)
    : feature_id_(_feature_id), start_frame_(_start_frame), used_num_(0), estimated_depth_(-1.0),
      solve_flag_(0)
  {
  }

  int endFrame() const;
};

class FeatureManager {
public:
  FeatureManager(Matrix3d _Rs[]);
  FeatureManager(const FeatureManager &) = delete;
  FeatureManager & operator=(const FeatureManager &) = delete;

  void setRic(Matrix3d _ric[]);
  void clearState();
  int getFeatureCount();
  bool addFeatureCheckParallax(
    int _frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> & _image, double _td);
  vector<pair<Vector3d, Vector3d>> getCorresponding(int _frame_count_l, int _frame_count_r);
  //void updateDepth(const VectorXd &x);
  void setDepth(const VectorXd & _x);
  void removeFailures();
  void clearDepth();
  VectorXd getDepthVector();
  void triangulate(int _frameCnt, Vector3d _Ps[], Matrix3d _Rs[], Vector3d _tic[], Matrix3d _ric[]);
  void triangulatePoint(
    Eigen::Matrix<double, 3, 4> & _Pose0, Eigen::Matrix<double, 3, 4> & _Pose1, Eigen::Vector2d & _point0,
    Eigen::Vector2d & _point1, Eigen::Vector3d & _point_3d);
  void initFramePoseByPnP(int _frameCnt, Vector3d _Ps[], Matrix3d _Rs[], Vector3d _tic[], Matrix3d _ric[]);
  bool solvePoseByPnP(
    Eigen::Matrix3d & _R_initial, Eigen::Vector3d & _P_initial, vector<cv::Point2f> & _pts2D,
    vector<cv::Point3f> & _pts3D);
  void removeBackShiftDepth(
    Eigen::Matrix3d _marg_R, Eigen::Vector3d _marg_P, Eigen::Matrix3d _new_R, Eigen::Vector3d _new_P);
  void removeBack();
  void removeFront(int _frame_count);
  void removeOutlier(set<int> & _outlierIndex);
  list<FeaturePerId> feature_;
  int last_track_num_;
  double last_average_parallax_;
  int new_feature_num_;
  int long_track_num_;

private:
  // Preserve list order (and thus solver parameter order); only lookup is hashed.
  std::unordered_map<int, list<FeaturePerId>::iterator> feature_index_;
  void eraseFeature(list<FeaturePerId>::iterator _it);
  double compensatedParallax2(const FeaturePerId & _it_per_id, int _frame_count);
  const Matrix3d * Rs_;
  Matrix3d ric_[2];
};

#endif