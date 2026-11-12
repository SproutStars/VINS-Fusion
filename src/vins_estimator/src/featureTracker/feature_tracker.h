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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <execinfo.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <queue>

#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f & _pt);
void reduceVector(vector<cv::Point2f> & _v, const vector<uchar> & _status);
void reduceVector(vector<int> & _v, const vector<uchar> & _status);

struct FeatureTrackerTimings_s {
  double pyramid_ms_ = 0.0;
  double flow_ms_ = 0.0;
  double detect_ms_ = 0.0;
  double total_ms_ = 0.0;
  int tracked_features_ = 0;
  int detected_features_ = 0;
};

class FeatureTracker {
public:
  FeatureTracker();
  map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(
    double _cur_time, const cv::Mat & _img, const cv::Mat & _img1 = cv::Mat(), bool _extract_features = true,
    bool _draw_tracks = true);
  void setMask();
  void
  detectFeatures(const cv::Mat & _img, const cv::Mat & _mask, int _max_features, vector<cv::Point2f> & _points);
  const vector<cv::Mat> & previousPyramid() const { return prev_pyr_; }
  const FeatureTrackerTimings_s & lastTimings() const { return last_timings_; }
  // Opt-in temporal detection gate. 1 (default) runs the detector exactly as
  // before on every frame that has a deficit. Values > 1 run it at most every
  // N frames unless the feature deficit becomes urgent, which keeps recovery
  // bounded in difficult scenes.
  void setDetectionInterval(int _frames)
  {
    detection_interval_.store(std::max(1, _frames), std::memory_order_relaxed);
  }
  int detectionInterval() const { return detection_interval_.load(std::memory_order_relaxed); }
  void readIntrinsicParameter(const vector<string> & _calib_file);
  void showUndistortion(const string & _name);
  void rejectWithF();
  void undistortedPoints();
  vector<cv::Point2f> undistortedPts(vector<cv::Point2f> & _pts, camodocal::CameraPtr _cam);
  vector<cv::Point2f> ptsVelocity(
    vector<int> & _ids, vector<cv::Point2f> & _pts, map<int, cv::Point2f> & _cur_id_pts,
    map<int, cv::Point2f> & _prev_id_pts);
  void showTwoImage(
    const cv::Mat & _img1, const cv::Mat & _img2, vector<cv::Point2f> _pts1, vector<cv::Point2f> _pts2);
  void drawTrack(
    const cv::Mat & _imLeft, const cv::Mat & _imRight, vector<int> & _curLeftIds,
    vector<cv::Point2f> & _curLeftPts, vector<cv::Point2f> & _curRightPts,
    map<int, cv::Point2f> & _prevLeftPtsMap);
  void setPrediction(map<int, Eigen::Vector3d> & _predictPts);
  double distance(cv::Point2f & _pt1, cv::Point2f & _pt2);
  void removeOutliers(set<int> & _removePtsIds);
  cv::Mat getTrackImage() const;
  bool inBorder(const cv::Point2f & _pt) const;

  int row_, col_;
  cv::Mat imTrack_;
  cv::Mat mask_;
  cv::Mat fisheye_mask_;
  cv::Mat prev_img_, cur_img_;
  vector<cv::Point2f> n_pts_;
  vector<cv::Point2f> predict_pts_;
  vector<cv::Point2f> predict_pts_debug_;
  vector<cv::Point2f> prev_pts_, cur_pts_, cur_right_pts_;
  vector<cv::Point2f> prev_un_pts_, cur_un_pts_, cur_un_right_pts_;
  vector<cv::Point2f> pts_velocity_, right_pts_velocity_;
  vector<int> ids_, ids_right_;
  vector<int> track_cnt_;
  map<int, cv::Point2f> cur_un_pts_map_, prev_un_pts_map_;
  map<int, cv::Point2f> cur_un_right_pts_map_, prev_un_right_pts_map_;
  map<int, cv::Point2f> prevLeftPtsMap_;
  vector<camodocal::CameraPtr> m_camera_;
  double cur_time_;
  double prev_time_;
  bool stereo_cam_;
  int n_id_;
  bool hasPrediction_;

private:
  friend struct FeatureTrackerTestAccess;
  void detectFastGridFeatures(
    const cv::Mat & _img, const cv::Mat & _mask, int _max_features, vector<cv::Point2f> & _points);
  void detectPyramidFeatures(
    const cv::Mat & _img, const cv::Mat & _mask, int _max_features, vector<cv::Point2f> & _points,
    const cv::Mat & _half_image);
  void undistortedPtsInto(
    const vector<cv::Point2f> & _pts, const camodocal::CameraPtr & _cam, vector<cv::Point2f> & _out,
    vector<cv::Point3d> * _lifted = nullptr) const;
  void ptsVelocityInto(
    const vector<int> & _ids, const vector<cv::Point2f> & _pts, map<int, cv::Point2f> & _cur_id_pts,
    const map<int, cv::Point2f> & _prev_id_pts, vector<cv::Point2f> & _velocity) const;

  vector<pair<int, pair<cv::Point2f, int>>> mask_order_;
  cv::Mat detection_half_mask_scratch_;
  vector<cv::Mat> prev_pyr_;
  // Scratch pyramids are kept between calls so buildOpticalFlowPyramid()
  // reuses its level buffers instead of allocating/freeing ~0.5 MB per frame.
  vector<cv::Mat> scratch_pyr_;
  vector<cv::Mat> scratch_right_pyr_;
  // KLT and RANSAC outputs are reused between frames. OpenCV and
  // findFundamentalMat overwrite the contents, so only capacity is retained.
  vector<uchar> flow_status_scratch_;
  vector<uchar> reverse_status_scratch_;
  vector<float> flow_err_scratch_;
  vector<cv::Point2f> reverse_pts_scratch_;
  vector<cv::Point2f> ransac_cur_pts_scratch_;
  vector<cv::Point2f> ransac_prev_pts_scratch_;
  vector<uchar> ransac_status_scratch_;
  vector<cv::Point2f> detection_candidates_scratch_;
  // Keep the exact double-precision lifted points behind prev_un_pts_. RANSAC
  // can then reuse the previous frame's liftProjective() results without the
  // float round-trip through prev_un_pts_ changing its input coordinates.
  vector<cv::Point3d> cur_lifted_pts_;
  vector<cv::Point3d> prev_lifted_pts_;
  FeatureTrackerTimings_s last_timings_;
  std::atomic<int> detection_interval_{1};
  int frame_counter_ = 0;
};
