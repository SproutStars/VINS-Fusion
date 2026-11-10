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

#include <atomic>

#pragma once
#include "LocalCartesian.hpp"
#include "tic_toc.h"
#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <map>
#include <mutex>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <thread>
#include <vector>

using namespace std;

class GlobalOptimization {
  std::atomic<bool> stopping_{false};

public:
  GlobalOptimization();
  ~GlobalOptimization();
  void inputGPS(double _t, double _latitude, double _longitude, double _altitude, double _posAccuracy);
  void inputOdom(double _t, Eigen::Vector3d _OdomP, Eigen::Quaterniond _OdomQ);
  void getGlobalOdom(Eigen::Vector3d & _odomP, Eigen::Quaterniond & _odomQ);
  nav_msgs::msg::Path global_path_;
  nav_msgs::msg::Path getGlobalPath();

private:
  void gpS2Xyz(double _latitude, double _longitude, double _altitude, double * _xyz);
  void optimize();
  void updateGlobalPath();

  // format t, tx,ty,tz,qw,qx,qy,qz
  map<double, vector<double>> localPoseMap_;
  map<double, vector<double>> globalPoseMap_;
  map<double, vector<double>> GPSPositionMap_;
  bool initGPS_;
  std::atomic<bool> newGPS_{false};
  GeographicLib::LocalCartesian geoConverter_;
  std::mutex mPoseMap_;
  Eigen::Matrix4d WGPS_T_WVIO_;
  Eigen::Vector3d lastP_;
  Eigen::Quaterniond lastQ_;
  std::thread threadOpt_;
};