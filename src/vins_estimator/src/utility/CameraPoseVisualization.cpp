/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <vins_common/ros_utils.hpp>

#include "CameraPoseVisualization.h"

const Eigen::Vector3d CameraPoseVisualization::IMLT = Eigen::Vector3d(-1.0, -0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualization::IMRT = Eigen::Vector3d(1.0, -0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualization::IMLB = Eigen::Vector3d(-1.0, 0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualization::IMRB = Eigen::Vector3d(1.0, 0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualization::LT0 = Eigen::Vector3d(-0.7, -0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualization::LT1 = Eigen::Vector3d(-0.7, -0.2, 1.0);
const Eigen::Vector3d CameraPoseVisualization::LT2 = Eigen::Vector3d(-1.0, -0.2, 1.0);
const Eigen::Vector3d CameraPoseVisualization::OC = Eigen::Vector3d(0.0, 0.0, 0.0);

void Eigen2Point(const Eigen::Vector3d & _v, geometry_msgs::msg::Point & _p)
{
  _p.x = _v.x();
  _p.y = _v.y();
  _p.z = _v.z();
}

CameraPoseVisualization::CameraPoseVisualization(float _r, float _g, float _b, float _a)
  : m_marker_ns_("CameraPoseVisualization"), m_scale_(0.2), m_line_width_(0.01)
{
  m_image_boundary_color_.r = _r;
  m_image_boundary_color_.g = _g;
  m_image_boundary_color_.b = _b;
  m_image_boundary_color_.a = _a;
  m_optical_center_connector_color_.r = _r;
  m_optical_center_connector_color_.g = _g;
  m_optical_center_connector_color_.b = _b;
  m_optical_center_connector_color_.a = _a;
}

void CameraPoseVisualization::setImageBoundaryColor(float _r, float _g, float _b, float _a)
{
  m_image_boundary_color_.r = _r;
  m_image_boundary_color_.g = _g;
  m_image_boundary_color_.b = _b;
  m_image_boundary_color_.a = _a;
}

void CameraPoseVisualization::setOpticalCenterConnectorColor(float _r, float _g, float _b, float _a)
{
  m_optical_center_connector_color_.r = _r;
  m_optical_center_connector_color_.g = _g;
  m_optical_center_connector_color_.b = _b;
  m_optical_center_connector_color_.a = _a;
}

void CameraPoseVisualization::setScale(double _s) { m_scale_ = _s; }
void CameraPoseVisualization::setLineWidth(double _width) { m_line_width_ = _width; }
void CameraPoseVisualization::addEdge(const Eigen::Vector3d & _p0, const Eigen::Vector3d & _p1)
{
  visualization_msgs::msg::Marker marker;

  marker.ns = m_marker_ns_;
  marker.id = m_markers_.size() + 1;
  marker.type = visualization_msgs::msg::Marker::LINE_LIST;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.scale.x = 0.005;

  marker.color.g = 1.0f;
  marker.color.a = 1.0;

  geometry_msgs::msg::Point point0, point1;

  Eigen2Point(_p0, point0);
  Eigen2Point(_p1, point1);

  marker.points.push_back(point0);
  marker.points.push_back(point1);

  m_markers_.push_back(marker);
}

void CameraPoseVisualization::addLoopedge(const Eigen::Vector3d & _p0, const Eigen::Vector3d & _p1)
{
  visualization_msgs::msg::Marker marker;

  marker.ns = m_marker_ns_;
  marker.id = m_markers_.size() + 1;
  marker.type = visualization_msgs::msg::Marker::LINE_LIST;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.scale.x = 0.04;
  //marker.scale.x = 0.3;

  marker.color.r = 1.0f;
  marker.color.b = 1.0f;
  marker.color.a = 1.0;

  geometry_msgs::msg::Point point0, point1;

  Eigen2Point(_p0, point0);
  Eigen2Point(_p1, point1);

  marker.points.push_back(point0);
  marker.points.push_back(point1);

  m_markers_.push_back(marker);
}


void CameraPoseVisualization::addPose(const Eigen::Vector3d & _p, const Eigen::Quaterniond & _q)
{
  visualization_msgs::msg::Marker marker;

  marker.ns = m_marker_ns_;
  marker.id = m_markers_.size() + 1;
  marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.scale.x = m_line_width_;

  marker.pose.position.x = 0.0;
  marker.pose.position.y = 0.0;
  marker.pose.position.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;


  geometry_msgs::msg::Point pt_lt, pt_lb, pt_rt, pt_rb, pt_oc, pt_lt0, pt_lt1, pt_lt2;

  Eigen2Point(_q * (m_scale_ * IMLT) + _p, pt_lt);
  Eigen2Point(_q * (m_scale_ * IMLB) + _p, pt_lb);
  Eigen2Point(_q * (m_scale_ * IMRT) + _p, pt_rt);
  Eigen2Point(_q * (m_scale_ * IMRB) + _p, pt_rb);
  Eigen2Point(_q * (m_scale_ * LT0) + _p, pt_lt0);
  Eigen2Point(_q * (m_scale_ * LT1) + _p, pt_lt1);
  Eigen2Point(_q * (m_scale_ * LT2) + _p, pt_lt2);
  Eigen2Point(_q * (m_scale_ * OC) + _p, pt_oc);

  // image boundaries
  marker.points.push_back(pt_lt);
  marker.points.push_back(pt_lb);
  marker.colors.push_back(m_image_boundary_color_);
  marker.colors.push_back(m_image_boundary_color_);

  marker.points.push_back(pt_lb);
  marker.points.push_back(pt_rb);
  marker.colors.push_back(m_image_boundary_color_);
  marker.colors.push_back(m_image_boundary_color_);

  marker.points.push_back(pt_rb);
  marker.points.push_back(pt_rt);
  marker.colors.push_back(m_image_boundary_color_);
  marker.colors.push_back(m_image_boundary_color_);

  marker.points.push_back(pt_rt);
  marker.points.push_back(pt_lt);
  marker.colors.push_back(m_image_boundary_color_);
  marker.colors.push_back(m_image_boundary_color_);

  // top-left indicator
  marker.points.push_back(pt_lt0);
  marker.points.push_back(pt_lt1);
  marker.colors.push_back(m_image_boundary_color_);
  marker.colors.push_back(m_image_boundary_color_);

  marker.points.push_back(pt_lt1);
  marker.points.push_back(pt_lt2);
  marker.colors.push_back(m_image_boundary_color_);
  marker.colors.push_back(m_image_boundary_color_);

  // optical center connector
  marker.points.push_back(pt_lt);
  marker.points.push_back(pt_oc);
  marker.colors.push_back(m_optical_center_connector_color_);
  marker.colors.push_back(m_optical_center_connector_color_);


  marker.points.push_back(pt_lb);
  marker.points.push_back(pt_oc);
  marker.colors.push_back(m_optical_center_connector_color_);
  marker.colors.push_back(m_optical_center_connector_color_);

  marker.points.push_back(pt_rt);
  marker.points.push_back(pt_oc);
  marker.colors.push_back(m_optical_center_connector_color_);
  marker.colors.push_back(m_optical_center_connector_color_);

  marker.points.push_back(pt_rb);
  marker.points.push_back(pt_oc);
  marker.colors.push_back(m_optical_center_connector_color_);
  marker.colors.push_back(m_optical_center_connector_color_);

  m_markers_.push_back(marker);
}

void CameraPoseVisualization::reset() { m_markers_.clear(); }

void CameraPoseVisualization::publishBy(
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr & _pub,
  const std_msgs::msg::Header & _header)
{
  visualization_msgs::msg::MarkerArray markerArray_msg;

  for (auto & marker : m_markers_) {
    marker.header = _header;
    markerArray_msg.markers.push_back(marker);
  }

  _pub->publish(markerArray_msg);
}