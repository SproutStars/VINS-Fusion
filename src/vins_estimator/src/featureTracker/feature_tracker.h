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
#include <cstdio>
#include <iostream>
#include <queue>
#include <chrono>
#include <execinfo.h>
#include <csignal>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);
void reduceVector(vector<cv::Point2f> &v, const vector<uchar> &status);
void reduceVector(vector<int> &v, const vector<uchar> &status);

struct FeatureTrackerTimings
{
    double pyramid_ms = 0.0;
    double flow_ms = 0.0;
    double detect_ms = 0.0;
    double total_ms = 0.0;
    int tracked_features = 0;
    int detected_features = 0;
};

class FeatureTracker
{
public:
    FeatureTracker();
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat(), bool extract_features = true, bool draw_tracks = true);
    void setMask();
    void detectFeatures(const cv::Mat &img, const cv::Mat &mask, int max_features,
                        vector<cv::Point2f> &points);
    const vector<cv::Mat> &previousPyramid() const { return prev_pyr_; }
    const FeatureTrackerTimings &lastTimings() const { return last_timings_; }
    // Opt-in temporal detection gate. 1 (default) runs the detector exactly as
    // before on every frame that has a deficit. Values > 1 run it at most every
    // N frames unless the feature deficit becomes urgent, which keeps recovery
    // bounded in difficult scenes.
    void setDetectionInterval(int frames)
    {
        detection_interval_.store(std::max(1, frames), std::memory_order_relaxed);
    }
    int detectionInterval() const
    {
        return detection_interval_.load(std::memory_order_relaxed);
    }
    void readIntrinsicParameter(const vector<string> &calib_file);
    void showUndistortion(const string &name);
    void rejectWithF();
    void undistortedPoints();
    vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);
    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts);
    void showTwoImage(const cv::Mat &img1, const cv::Mat &img2, 
                      vector<cv::Point2f> pts1, vector<cv::Point2f> pts2);
    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                                   vector<int> &curLeftIds,
                                   vector<cv::Point2f> &curLeftPts, 
                                   vector<cv::Point2f> &curRightPts,
                                   map<int, cv::Point2f> &prevLeftPtsMap);
    void setPrediction(map<int, Eigen::Vector3d> &predictPts);
    double distance(cv::Point2f &pt1, cv::Point2f &pt2);
    void removeOutliers(set<int> &removePtsIds);
    cv::Mat getTrackImage();
    bool inBorder(const cv::Point2f &pt);

    int row, col;
    cv::Mat imTrack;
    cv::Mat mask;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> predict_pts;
    vector<cv::Point2f> predict_pts_debug;
    vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts;
    vector<cv::Point2f> pts_velocity, right_pts_velocity;
    vector<int> ids, ids_right;
    vector<int> track_cnt;
    map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;
    map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map;
    map<int, cv::Point2f> prevLeftPtsMap;
    vector<camodocal::CameraPtr> m_camera;
    double cur_time;
    double prev_time;
    bool stereo_cam;
    int n_id;
    bool hasPrediction;

private:
    void detectFastGridFeatures(const cv::Mat &img, const cv::Mat &mask, int max_features,
                                vector<cv::Point2f> &points);

    vector<pair<int, pair<cv::Point2f, int>>> mask_order_;
    vector<cv::Mat> prev_pyr_;
    // Scratch pyramids are kept between calls so buildOpticalFlowPyramid()
    // reuses its level buffers instead of allocating/freeing ~0.5 MB per frame.
    vector<cv::Mat> scratch_pyr_;
    vector<cv::Mat> scratch_right_pyr_;
    FeatureTrackerTimings last_timings_;
    std::atomic<int> detection_interval_{1};
    int frame_counter_ = 0;
};
