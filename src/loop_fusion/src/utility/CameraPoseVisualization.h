/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include "../parameters.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <vins_common/ros_utils.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

class CameraPoseVisualization {
public:
  std::string m_marker_ns_;

  CameraPoseVisualization(float _r, float _g, float _b, float _a);

  void setImageBoundaryColor(float _r, float _g, float _b, float _a = 1.0);
  void setOpticalCenterConnectorColor(float _r, float _g, float _b, float _a = 1.0);
  void setScale(double _s);
  void setLineWidth(double _width);

  void addPose(const Eigen::Vector3d & _p, const Eigen::Quaterniond & _q);
  void reset();

  void publishBy(
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr & _pub,
    const std_msgs::msg::Header & _header);
  void addEdge(const Eigen::Vector3d & _p0, const Eigen::Vector3d & _p1);
  void addLoopedge(const Eigen::Vector3d & _p0, const Eigen::Vector3d & _p1);
  //void add_image(const Eigen::Vector3d& T, const Eigen::Matrix3d& R, const cv::Mat &src);
  void publishImageBy(
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr & _pub,
    const std_msgs::msg::Header & _header);

private:
  std::vector<visualization_msgs::msg::Marker> m_markers_;
  std_msgs::msg::ColorRGBA m_image_boundary_color_;
  std_msgs::msg::ColorRGBA m_optical_center_connector_color_;
  double m_scale_;
  double m_line_width_;
  visualization_msgs::msg::Marker image_;
  int LOOP_EDGE_NUM_;
  int tmp_loop_edge_num_;

  static const Eigen::Vector3d IMLT;
  static const Eigen::Vector3d IMLB;
  static const Eigen::Vector3d IMRT;
  static const Eigen::Vector3d IMRB;
  static const Eigen::Vector3d OC;
  static const Eigen::Vector3d LT0;
  static const Eigen::Vector3d LT1;
  static const Eigen::Vector3d LT2;
};
