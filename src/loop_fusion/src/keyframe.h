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

#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"
#include "brief_extractor.h"
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "parameters.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#define MIN_LOOP_NUM 25

using namespace Eigen;
using namespace std;
using namespace DVision;


class KeyFrame {
public:
  KeyFrame(
    double _time_stamp, int _index, Vector3d & _vio_T_w_i, Matrix3d & _vio_R_w_i, cv::Mat & _image,
    vector<cv::Point3f> & _point_3d, vector<cv::Point2f> & _point_2d_uv,
    vector<cv::Point2f> & _point_2d_normal, vector<double> & _point_id, int _sequence);
  KeyFrame(
    double _time_stamp, int _index, Vector3d & _vio_T_w_i, Matrix3d & _vio_R_w_i, Vector3d & _T_w_i,
    Matrix3d & _R_w_i, cv::Mat & _image, int _loop_index, Eigen::Matrix<double, 8, 1> & _loop_info,
    vector<cv::KeyPoint> & _keypoints, vector<cv::KeyPoint> & _keypoints_norm,
    vector<BRIEF::bitset> & _brief_descriptors);
  bool findConnection(KeyFrame * _old_kf);
  void computeWindowBRIEFPoint();
  void computeBRIEFPoint();
  //void extractBrief();
  int hammingDis(const BRIEF::bitset & _a, const BRIEF::bitset & _b);
  bool searchInAera(
    const BRIEF::bitset _window_descriptor, const std::vector<BRIEF::bitset> & _descriptors_old,
    const std::vector<cv::KeyPoint> & _keypoints_old, const std::vector<cv::KeyPoint> & _keypoints_old_norm,
    cv::Point2f & _best_match, cv::Point2f & _best_match_norm);
  void searchByBRIEFDes(
    std::vector<cv::Point2f> & _matched_2d_old, std::vector<cv::Point2f> & _matched_2d_old_norm,
    std::vector<uchar> & _status, const std::vector<BRIEF::bitset> & _descriptors_old,
    const std::vector<cv::KeyPoint> & _keypoints_old, const std::vector<cv::KeyPoint> & _keypoints_old_norm);
  void fundmantalMatrixRansac(
    const std::vector<cv::Point2f> & _matched_2d_cur_norm,
    const std::vector<cv::Point2f> & _matched_2d_old_norm, vector<uchar> & _status);
  void pnPransac(
    const vector<cv::Point2f> & _matched_2d_old_norm, const std::vector<cv::Point3f> & _matched_3d,
    std::vector<uchar> & _status, Eigen::Vector3d & _PnP_T_old, Eigen::Matrix3d & _PnP_R_old);
  void getVioPose(Eigen::Vector3d & _T_w_i, Eigen::Matrix3d & _R_w_i) const;
  void getPose(Eigen::Vector3d & _T_w_i, Eigen::Matrix3d & _R_w_i) const;
  void updatePose(const Eigen::Vector3d & _T_w_i, const Eigen::Matrix3d & _R_w_i);
  void updateVioPose(const Eigen::Vector3d & _T_w_i, const Eigen::Matrix3d & _R_w_i);
  void updateLoop(Eigen::Matrix<double, 8, 1> & _loop_info);

  Eigen::Vector3d getLoopRelativeT();
  double getLoopRelativeYaw();
  Eigen::Quaterniond getLoopRelativeQ();



  double time_stamp_;
  int index_;
  int local_index_;
  Eigen::Vector3d vio_T_w_i_;
  Eigen::Matrix3d vio_R_w_i_;
  Eigen::Vector3d T_w_i_;
  Eigen::Matrix3d R_w_i_;
  Eigen::Vector3d origin_vio_T_;
  Eigen::Matrix3d origin_vio_R_;
  cv::Mat image_;
  cv::Mat thumbnail_;
  vector<cv::Point3f> point_3d_;
  vector<cv::Point2f> point_2d_uv_;
  vector<cv::Point2f> point_2d_norm_;
  vector<double> point_id_;
  vector<cv::KeyPoint> keypoints_;
  vector<cv::KeyPoint> keypoints_norm_;
  vector<cv::KeyPoint> window_keypoints_;
  vector<BRIEF::bitset> brief_descriptors_;
  vector<BRIEF::bitset> window_brief_descriptors_;
  bool has_fast_point_;
  int sequence_;

  bool has_loop_;
  int loop_index_;
  Eigen::Matrix<double, 8, 1> loop_info_;
};
