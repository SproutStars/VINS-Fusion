/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include "../estimator/estimator.h"
#include "../estimator/parameters.h"
#include "CameraPoseVisualization.h"
#include <cv_bridge/cv_bridge.hpp>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <vins_common/ros_utils.hpp>
#include <visualization_msgs/msg/marker.hpp>

extern rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr PUB_ODOMETRY;
extern rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr PUB_PATH;

extern rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr PUB_KEY_POSES;


extern nav_msgs::msg::Path PATH;
extern rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr PUB_POSE_GRAPH;
extern int IMAGE_ROW, IMAGE_COL;

void unregisterPub();
void resetVisualization();
bool trackImageRequested();
// True when any Schur-backed point-cloud/keyframe topic has a subscriber. The
// estimator uses this to avoid triangulating publication-only snapshots when
// loop_fusion and RViz are not connected.
bool schurVisualizationRequested();

void registerPub(const rclcpp::Node::SharedPtr & _n);

void pubLatestOdometry(
  const Eigen::Vector3d & _P, const Eigen::Quaterniond & _Q, const Eigen::Vector3d & _V, double _t);

// IMU-rate Schur state and covariance have the same sensor timestamp.
void pubLatestOdometry(const vins::backend::EkfState & _state, const Eigen::Vector3d & _raw_gyro, double _imu_dt);

void pubTrackImage(const cv::Mat & _imgTrack, const double _t);

void printStatistics(const Estimator & _estimator, double _t);

void pubOdometry(const Estimator & _estimator, const std_msgs::msg::Header & _header);

void pubInitialGuess(const Estimator & _estimator, const std_msgs::msg::Header & _header);

void pubKeyPoses(const Estimator & _estimator, const std_msgs::msg::Header & _header);

void pubCameraPose(const Estimator & _estimator, const std_msgs::msg::Header & _header);

void pubPointCloud(const Estimator & _estimator, const std_msgs::msg::Header & _header);

void pubTF(const Estimator & _estimator, const std_msgs::msg::Header & _header);

void pubKeyframe(const Estimator & _estimator);

void pubRelocalization(const Estimator & _estimator);

void pubCar(const Estimator & _estimator, const std_msgs::msg::Header & _header);
