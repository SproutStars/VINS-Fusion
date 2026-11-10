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

#include "globalOpt.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <fstream>
#include <iostream>
#include <mutex>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <queue>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <stdio.h>
#include <vins_common/ros_utils.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

std::unique_ptr<GlobalOptimization> GLOBAL_ESTIMATOR;
std::string OUTPUT_FILE;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr PUB_GLOBAL_ODOMETRY;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr PUB_GLOBAL_PATH;
rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr PUB_CAR;

double LAST_VIO_T = -1;
std::queue<sensor_msgs::msg::NavSatFix::ConstSharedPtr> GPS_QUEUE;
std::mutex M_BUF;

void publish_car_model(double _t, Eigen::Vector3d _t_w_car, Eigen::Quaterniond _q_w_car)
{
  visualization_msgs::msg::MarkerArray markerArray_msg;
  visualization_msgs::msg::Marker car_mesh;
  car_mesh.header.stamp = vins::stamp(_t);
  car_mesh.header.frame_id = "world";
  car_mesh.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
  car_mesh.action = visualization_msgs::msg::Marker::ADD;
  car_mesh.id = 0;

  car_mesh.mesh_resource = "package://global_fusion/models/car.dae";

  Eigen::Matrix3d rot;
  rot << 0, 0, -1, 0, -1, 0, -1, 0, 0;

  Eigen::Quaterniond Q;
  Q = _q_w_car * rot;
  car_mesh.pose.position.x = _t_w_car.x();
  car_mesh.pose.position.y = _t_w_car.y();
  car_mesh.pose.position.z = _t_w_car.z();
  car_mesh.pose.orientation.w = Q.w();
  car_mesh.pose.orientation.x = Q.x();
  car_mesh.pose.orientation.y = Q.y();
  car_mesh.pose.orientation.z = Q.z();

  car_mesh.color.a = 1.0;
  car_mesh.color.r = 1.0;
  car_mesh.color.g = 0.0;
  car_mesh.color.b = 0.0;

  float major_scale = 2.0;

  car_mesh.scale.x = major_scale;
  car_mesh.scale.y = major_scale;
  car_mesh.scale.z = major_scale;
  markerArray_msg.markers.push_back(car_mesh);
  PUB_CAR->publish(markerArray_msg);
}

void GPS_callback(const sensor_msgs::msg::NavSatFix::ConstSharedPtr & _GPS_msg)
{
  if (
    _GPS_msg->status.status < 0 || !std::isfinite(_GPS_msg->latitude) ||
    !std::isfinite(_GPS_msg->longitude) || !std::isfinite(_GPS_msg->altitude))
    return;
  M_BUF.lock();
  if (GPS_QUEUE.size() >= 1000)
    GPS_QUEUE.pop();
  GPS_QUEUE.push(_GPS_msg);
  M_BUF.unlock();
}

void vio_callback(const nav_msgs::msg::Odometry::ConstSharedPtr & _pose_msg)
{
  //printf("vio_callback! \n");
  double t = vins::seconds(_pose_msg->header.stamp);
  LAST_VIO_T = t;
  Eigen::Vector3d vio_t(
    _pose_msg->pose.pose.position.x, _pose_msg->pose.pose.position.y, _pose_msg->pose.pose.position.z);
  Eigen::Quaterniond vio_q;
  vio_q.w() = _pose_msg->pose.pose.orientation.w;
  vio_q.x() = _pose_msg->pose.pose.orientation.x;
  vio_q.y() = _pose_msg->pose.pose.orientation.y;
  vio_q.z() = _pose_msg->pose.pose.orientation.z;
  GLOBAL_ESTIMATOR->inputOdom(t, vio_t, vio_q);


  M_BUF.lock();
  while (!GPS_QUEUE.empty()) {
    sensor_msgs::msg::NavSatFix::ConstSharedPtr GPS_msg = GPS_QUEUE.front();
    double gps_t = vins::seconds(GPS_msg->header.stamp);
    printf("vio t: %f, gps t: %f \n", t, gps_t);
    // 10ms sync tolerance
    if (gps_t >= t - 0.01 && gps_t <= t + 0.01) {
      //printf("receive GPS with timestamp %f\n", vins::seconds(GPS_msg->header.stamp));
      double latitude = GPS_msg->latitude;
      double longitude = GPS_msg->longitude;
      double altitude = GPS_msg->altitude;
      //int numSats = GPS_msg->status.service;
      double pos_accuracy = GPS_msg->position_covariance[0];
      if (pos_accuracy <= 0)
        pos_accuracy = 1;
      //printf("receive covariance %lf \n", pos_accuracy);
      //if(GPS_msg->status.status > 8)
      GLOBAL_ESTIMATOR->inputGPS(t, latitude, longitude, altitude, pos_accuracy);
      GPS_QUEUE.pop();
      break;
    } else if (gps_t < t - 0.01)
      GPS_QUEUE.pop();
    else if (gps_t > t + 0.01)
      break;
  }
  M_BUF.unlock();

  Eigen::Vector3d global_t;
  Eigen::Quaterniond global_q;
  GLOBAL_ESTIMATOR->getGlobalOdom(global_t, global_q);

  nav_msgs::msg::Odometry odometry;
  odometry.header = _pose_msg->header;
  odometry.header.frame_id = "world";
  odometry.child_frame_id = "world";
  odometry.pose.pose.position.x = global_t.x();
  odometry.pose.pose.position.y = global_t.y();
  odometry.pose.pose.position.z = global_t.z();
  odometry.pose.pose.orientation.x = global_q.x();
  odometry.pose.pose.orientation.y = global_q.y();
  odometry.pose.pose.orientation.z = global_q.z();
  odometry.pose.pose.orientation.w = global_q.w();
  PUB_GLOBAL_ODOMETRY->publish(odometry);
  PUB_GLOBAL_PATH->publish(GLOBAL_ESTIMATOR->getGlobalPath());
  publish_car_model(t, global_t, global_q);


  // write result to file
  if (OUTPUT_FILE.empty())
    return;
  std::ofstream foutC(OUTPUT_FILE, ios::app);
  foutC.setf(ios::fixed, ios::floatfield);
  foutC.precision(0);
  foutC << vins::seconds(_pose_msg->header.stamp) * 1e9 << ",";
  foutC.precision(5);
  foutC << global_t.x() << "," << global_t.y() << "," << global_t.z() << "," << global_q.w() << ","
        << global_q.x() << "," << global_q.y() << "," << global_q.z() << '\n';
  foutC.close();
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto n = std::make_shared<rclcpp::Node>("globalEstimator");

  GLOBAL_ESTIMATOR = std::make_unique<GlobalOptimization>();
  OUTPUT_FILE = n->declare_parameter<std::string>("output_file", "");
  if (!OUTPUT_FILE.empty()) {
    const auto parent = std::filesystem::path(OUTPUT_FILE).parent_path();
    if (!parent.empty())
      std::filesystem::create_directories(parent);
  }

  auto sub_GPS = n->create_subscription<sensor_msgs::msg::NavSatFix>(
    "/gps", rclcpp::SensorDataQoS().keep_last(100), GPS_callback);
  auto sub_vio =
    n->create_subscription<nav_msgs::msg::Odometry>("/vins_estimator/odometry", rclcpp::QoS(100), vio_callback);
  PUB_GLOBAL_PATH = n->create_publisher<nav_msgs::msg::Path>("~/global_path", rclcpp::QoS(100));
  PUB_GLOBAL_ODOMETRY = n->create_publisher<nav_msgs::msg::Odometry>("~/global_odometry", rclcpp::QoS(100));
  PUB_CAR = n->create_publisher<visualization_msgs::msg::MarkerArray>("~/car_model", rclcpp::QoS(100));
  rclcpp::spin(n);
  rclcpp::shutdown();
  GLOBAL_ESTIMATOR.reset();
  PUB_GLOBAL_ODOMETRY.reset();
  PUB_GLOBAL_PATH.reset();
  PUB_CAR.reset();
  return 0;
}
