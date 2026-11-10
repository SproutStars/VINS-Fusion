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

#include <std_srvs/srv/trigger.hpp>

#include "keyframe.h"
#include "parameters.h"
#include "pose_graph.h"
#include "primary_visualization.h"
#include "timestamp_sync.h"
#include "utility/CameraPoseVisualization.h"
#include "utility/tic_toc.h"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <mutex>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <std_msgs/msg/bool.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <thread>
#include <vector>
#include <vins_common/parameters.hpp>
#include <vins_common/ros_utils.hpp>
#include <visualization_msgs/msg/marker.hpp>
#define SKIP_FIRST_CNT 10
using namespace std;

queue<sensor_msgs::msg::Image::ConstSharedPtr> IMAGE_BUF;
queue<sensor_msgs::msg::PointCloud::ConstSharedPtr> POINT_BUF;
queue<nav_msgs::msg::Odometry::ConstSharedPtr> POSE_BUF;
queue<Eigen::Vector3d> ODOMETRY_BUF;
std::mutex M_BUF;
std::mutex M_PROCESS;
int FRAME_INDEX = 0;
int SEQUENCE = 1;
std::unique_ptr<PoseGraph> POSEGRAPH;
int SKIPPED_INITIAL_FRAME_COUNT = 0;
int SKIP_CNT;
int SKIPPED_FRAME_COUNT = 0;
bool LOAD_FLAG = 0;
bool START_FLAG = 0;
double SKIP_DIS = 0;

int VISUALIZATION_SHIFT_X;
int VISUALIZATION_SHIFT_Y;
int ROW;
int COL;
int DEBUG_IMAGE;

camodocal::CameraPtr M_CAMERA;
Eigen::Vector3d TIC = Eigen::Vector3d::Zero();
Eigen::Matrix3d QIC = Eigen::Matrix3d::Identity();
bool EXTRINSIC_READY = false;
std::unique_ptr<loop_fusion::PrimaryPath> PRIMARY_PATH;
std::unique_ptr<tf2_ros::TransformBroadcaster> PRIMARY_TF;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr PUB_PRIMARY_PATH;
rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr PUB_MATCH_IMG;
rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr PUB_CAMERA_POSE_VISUAL;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr PUB_ODOMETRY_RECT;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr PUB_PRIMARY_ODOMETRY;

std::string BRIEF_PATTERN_FILE;
std::string POSE_GRAPH_SAVE_PATH;
std::string VINS_RESULT_PATH;
CameraPoseVisualization CAMERAPOSEVISUAL(1, 0, 0, 1);
Eigen::Vector3d LAST_T(-100, -100, -100);
double LAST_IMAGE_TIME = -1;

rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr PUB_POINT_CLOUD;
rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr PUB_MARGIN_CLOUD;

void new_sequence()
{
  printf("new sequence\n");
  if (PRIMARY_PATH)
    PRIMARY_PATH->clear();
  SEQUENCE++;
  printf("sequence cnt %d \n", SEQUENCE);
  if (SEQUENCE > 5) {
    RCLCPP_WARN(vins::logger(), "only support 5 sequences since it's boring to copy code for more sequences.");
    throw std::runtime_error("VINS invariant violated");
  }
  POSEGRAPH->posegraph_visualization_->reset();
  POSEGRAPH->publish();
  M_BUF.lock();
  while (!IMAGE_BUF.empty())
    IMAGE_BUF.pop();
  while (!POINT_BUF.empty())
    POINT_BUF.pop();
  while (!POSE_BUF.empty())
    POSE_BUF.pop();
  while (!ODOMETRY_BUF.empty())
    ODOMETRY_BUF.pop();
  M_BUF.unlock();
}

void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & _image_msg)
{
  //RCLCPP_INFO(vins::logger(), "image_callback!");
  M_BUF.lock();
  IMAGE_BUF.push(_image_msg);
  M_BUF.unlock();
  //printf(" image time %f \n", vins::seconds(image_msg->header.stamp));

  // detect unstable camera stream
  if (LAST_IMAGE_TIME == -1)
    LAST_IMAGE_TIME = vins::seconds(_image_msg->header.stamp);
  else if (vins::seconds(_image_msg->header.stamp) - LAST_IMAGE_TIME > 1.0 || vins::seconds(_image_msg->header.stamp) < LAST_IMAGE_TIME) {
    RCLCPP_WARN(vins::logger(), "image discontinue! detect a new sequence!");
    new_sequence();
  }
  LAST_IMAGE_TIME = vins::seconds(_image_msg->header.stamp);
}

loop_fusion::Correction_s correctionSnapshot()
{
  std::lock_guard<std::mutex> lock(M_PROCESS);
  return POSEGRAPH->getCorrection();
}

void point_callback(const sensor_msgs::msg::PointCloud::ConstSharedPtr & _point_msg)
{
  //RCLCPP_INFO(vins::logger(), "point_callback!");
  M_BUF.lock();
  POINT_BUF.push(_point_msg);
  M_BUF.unlock();
  /*
    for (unsigned int i = 0; i < point_msg->points.size(); i++)
    {
        printf("%d, 3D point: %f, %f, %f 2D point %f, %f \n",i , point_msg->points[i].x, 
                                                     point_msg->points[i].y,
                                                     point_msg->points[i].z,
                                                     point_msg->channels[i].values[0],
                                                     point_msg->channels[i].values[1]);
    }
    */
  // for visualization
  sensor_msgs::msg::PointCloud point_cloud;
  point_cloud.header = _point_msg->header;
  point_cloud.header.frame_id = "world";
  const auto correction = correctionSnapshot();
  for (auto point : _point_msg->points) {
    cv::Point3f p_3d;
    p_3d.x = point.x;
    p_3d.y = point.y;
    p_3d.z = point.z;
    Eigen::Vector3d tmp =
      correction.rotation_ * Eigen::Vector3d(p_3d.x, p_3d.y, p_3d.z) + correction.translation_;
    geometry_msgs::msg::Point32 p;
    p.x = tmp(0);
    p.y = tmp(1);
    p.z = tmp(2);
    point_cloud.points.push_back(p);
  }
  PUB_POINT_CLOUD->publish(point_cloud);
}

// only for visualization
void margin_point_callback(const sensor_msgs::msg::PointCloud::ConstSharedPtr & _point_msg)
{
  sensor_msgs::msg::PointCloud point_cloud;
  point_cloud.header = _point_msg->header;
  point_cloud.header.frame_id = "world";
  const auto correction = correctionSnapshot();
  for (auto point : _point_msg->points) {
    cv::Point3f p_3d;
    p_3d.x = point.x;
    p_3d.y = point.y;
    p_3d.z = point.z;
    Eigen::Vector3d tmp =
      correction.rotation_ * Eigen::Vector3d(p_3d.x, p_3d.y, p_3d.z) + correction.translation_;
    geometry_msgs::msg::Point32 p;
    p.x = tmp(0);
    p.y = tmp(1);
    p.z = tmp(2);
    point_cloud.points.push_back(p);
  }
  PUB_MARGIN_CLOUD->publish(point_cloud);
}

void pose_callback(const nav_msgs::msg::Odometry::ConstSharedPtr & _pose_msg)
{
  //RCLCPP_INFO(vins::logger(), "pose_callback!");
  M_BUF.lock();
  POSE_BUF.push(_pose_msg);
  M_BUF.unlock();
  /*
    printf("pose t: %f, %f, %f   q: %f, %f, %f %f \n", pose_msg->pose.pose.position.x,
                                                       pose_msg->pose.pose.position.y,
                                                       pose_msg->pose.pose.position.z,
                                                       pose_msg->pose.pose.orientation.w,
                                                       pose_msg->pose.pose.orientation.x,
                                                       pose_msg->pose.pose.orientation.y,
                                                       pose_msg->pose.pose.orientation.z);
    */
}

void vio_callback(const nav_msgs::msg::Odometry::ConstSharedPtr & _pose_msg)
{
  loop_fusion::Correction_s correction;
  Vector3d camera_translation;
  Matrix3d camera_rotation;
  bool camera_ready;
  {
    std::lock_guard<std::mutex> lock(M_PROCESS);
    correction = POSEGRAPH->getCorrection();
    camera_translation = TIC;
    camera_rotation = QIC;
    camera_ready = EXTRINSIC_READY;
  }
  const auto odometry = loop_fusion::correctOdometry(*_pose_msg, correction);
  PUB_ODOMETRY_RECT->publish(odometry);
  if (PUB_PRIMARY_ODOMETRY)
    PUB_PRIMARY_ODOMETRY->publish(odometry);
  if (PRIMARY_PATH) {
    const auto path_message = PRIMARY_PATH->append(odometry);
    if (path_message)
      PUB_PRIMARY_PATH->publish(*path_message);
    PRIMARY_TF->sendTransform(
      loop_fusion::primaryTransforms(odometry, correction, camera_translation, camera_rotation, camera_ready));
  }

  const auto & pose = odometry.pose.pose;
  const Vector3d position(pose.position.x, pose.position.y, pose.position.z);
  const Quaterniond orientation(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
  CAMERAPOSEVISUAL.reset();
  CAMERAPOSEVISUAL.addPose(
    position + orientation * camera_translation, orientation * Quaterniond(camera_rotation));
  CAMERAPOSEVISUAL.publishBy(PUB_CAMERA_POSE_VISUAL, odometry.header);
}

void extrinsic_callback(const nav_msgs::msg::Odometry::ConstSharedPtr & _pose_msg)
{
  M_PROCESS.lock();
  TIC =
    Vector3d(_pose_msg->pose.pose.position.x, _pose_msg->pose.pose.position.y, _pose_msg->pose.pose.position.z);
  QIC = Quaterniond(
          _pose_msg->pose.pose.orientation.w, _pose_msg->pose.pose.orientation.x,
          _pose_msg->pose.pose.orientation.y, _pose_msg->pose.pose.orientation.z)
          .toRotationMatrix();
  EXTRINSIC_READY = true;
  M_PROCESS.unlock();
}

void process()
{
  while (rclcpp::ok()) {
    sensor_msgs::msg::Image::ConstSharedPtr image_msg = nullptr;
    sensor_msgs::msg::PointCloud::ConstSharedPtr point_msg = nullptr;
    nav_msgs::msg::Odometry::ConstSharedPtr pose_msg = nullptr;

    // Find the nearest image/point pair for the front keyframe pose. Exact
    // equality is not safe here: VINS carries the image time through a
    // double, which shifts an epoch timestamp by roughly 100 ns. A small
    // tolerance avoids consuming the next image while remaining much
    // smaller than the camera period.
    M_BUF.lock();
    if (!IMAGE_BUF.empty() && !POINT_BUF.empty() && !POSE_BUF.empty()) {
      const int64_t pose_ns = loop_fusion::stampNs(POSE_BUF.front()->header.stamp);
      while (!IMAGE_BUF.empty() && loop_fusion::stampNs(IMAGE_BUF.front()->header.stamp) <
                                     pose_ns - loop_fusion::K_STAMP_MATCH_TOLERANCE_NS)
        IMAGE_BUF.pop();
      while (!POINT_BUF.empty() && loop_fusion::stampNs(POINT_BUF.front()->header.stamp) <
                                     pose_ns - loop_fusion::K_STAMP_MATCH_TOLERANCE_NS)
        POINT_BUF.pop();

      if (!IMAGE_BUF.empty() && !POINT_BUF.empty()) {
        const int64_t image_ns = loop_fusion::stampNs(IMAGE_BUF.front()->header.stamp);
        const int64_t point_ns = loop_fusion::stampNs(POINT_BUF.front()->header.stamp);
        if (image_ns > pose_ns + loop_fusion::K_STAMP_MATCH_TOLERANCE_NS || point_ns > pose_ns + loop_fusion::K_STAMP_MATCH_TOLERANCE_NS) {
          // No image/point for this pose: drop the orphan pose.
          POSE_BUF.pop();
          printf("throw pose at beginning\n");
        } else {
          pose_msg = POSE_BUF.front();
          POSE_BUF.pop();
          image_msg = IMAGE_BUF.front();
          IMAGE_BUF.pop();
          point_msg = POINT_BUF.front();
          POINT_BUF.pop();
        }
      }
    }
    M_BUF.unlock();

    if (pose_msg != nullptr) {
      //printf(" pose time %f \n", vins::seconds(pose_msg->header.stamp));
      //printf(" point time %f \n", vins::seconds(point_msg->header.stamp));
      //printf(" image time %f \n", vins::seconds(image_msg->header.stamp));
      // skip fisrt few
      if (SKIPPED_INITIAL_FRAME_COUNT < SKIP_FIRST_CNT) {
        SKIPPED_INITIAL_FRAME_COUNT++;
        continue;
      }

      if (SKIPPED_FRAME_COUNT < SKIP_CNT) {
        SKIPPED_FRAME_COUNT++;
        continue;
      } else {
        SKIPPED_FRAME_COUNT = 0;
      }

      cv_bridge::CvImageConstPtr ptr;
      if (image_msg->encoding == "8UC1") {
        sensor_msgs::msg::Image img;
        img.header = image_msg->header;
        img.height = image_msg->height;
        img.width = image_msg->width;
        img.is_bigendian = image_msg->is_bigendian;
        img.step = image_msg->step;
        img.data = image_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
      } else
        ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);

      cv::Mat image = ptr->image;
      // build keyframe
      Vector3d T = Vector3d(
        pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
      Matrix3d R = Quaterniond(
                     pose_msg->pose.pose.orientation.w, pose_msg->pose.pose.orientation.x,
                     pose_msg->pose.pose.orientation.y, pose_msg->pose.pose.orientation.z)
                     .toRotationMatrix();
      if ((T - LAST_T).norm() > SKIP_DIS) {
        vector<cv::Point3f> point_3d;
        vector<cv::Point2f> point_2d_uv;
        vector<cv::Point2f> point_2d_normal;
        vector<double> point_id;

        for (unsigned int i = 0; i < point_msg->points.size(); i++) {
          cv::Point3f p_3d;
          p_3d.x = point_msg->points[i].x;
          p_3d.y = point_msg->points[i].y;
          p_3d.z = point_msg->points[i].z;
          point_3d.push_back(p_3d);

          cv::Point2f p_2d_uv, p_2d_normal;
          double p_id;
          p_2d_normal.x = point_msg->channels[i].values[0];
          p_2d_normal.y = point_msg->channels[i].values[1];
          p_2d_uv.x = point_msg->channels[i].values[2];
          p_2d_uv.y = point_msg->channels[i].values[3];
          p_id = point_msg->channels[i].values[4];
          point_2d_normal.push_back(p_2d_normal);
          point_2d_uv.push_back(p_2d_uv);
          point_id.push_back(p_id);

          //printf("u %f, v %f \n", p_2d_uv.x, p_2d_uv.y);
        }

        KeyFrame * keyframe = new KeyFrame(
          vins::seconds(pose_msg->header.stamp), FRAME_INDEX, T, R, image, point_3d, point_2d_uv,
          point_2d_normal, point_id, SEQUENCE);
        M_PROCESS.lock();
        START_FLAG = 1;
        POSEGRAPH->addKeyFrame(keyframe, 1);
        M_PROCESS.unlock();
        FRAME_INDEX++;
        LAST_T = T;
      }
    }
    std::chrono::milliseconds dura(5);
    std::this_thread::sleep_for(dura);
  }
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto n = std::make_shared<rclcpp::Node>("loop_fusion");
  const bool publish_primary = n->declare_parameter<bool>("publish_primary_odometry", false);
  const bool publish_visualization = n->declare_parameter<bool>("publish_primary_visualization", false);
  const int64_t path_capacity = n->declare_parameter<int64_t>("path_max_poses", 2000);
  const double path_rate = n->declare_parameter<double>("path_publish_rate", 1.0);
  const std::string input_odometry_topic =
    n->declare_parameter<std::string>("input_odometry_topic", "/vins_estimator/odometry");
  const std::string primary_odometry_topic = "/vins_estimator/odometry";
  try {
    // Validate even when output is disabled; invalid settings never hide
    // behind the launch switch. Zero capacity disables path, not TF.
    loop_fusion::PrimaryPath validated_path(path_capacity, path_rate);
    if (publish_visualization)
      PRIMARY_PATH = std::make_unique<loop_fusion::PrimaryPath>(validated_path);
    const auto topics = n->get_node_topics_interface();
    loop_fusion::validateOdometryRouting(
      topics->resolve_topic_name(input_odometry_topic), topics->resolve_topic_name("~/odometry_rect"),
      publish_primary, topics->resolve_topic_name(primary_odometry_topic));
  } catch (const std::exception & e) {
    RCLCPP_ERROR(n->get_logger(), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  POSEGRAPH = std::make_unique<PoseGraph>();
  POSEGRAPH->registerPub(n);

  VISUALIZATION_SHIFT_X = 0;
  VISUALIZATION_SHIFT_Y = 0;
  SKIP_CNT = 0;
  SKIP_DIS = 0;

  std::string config_file;
  try {
    config_file = vins::configFile(n, argc, argv);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(n->get_logger(), "%s", e.what());
    POSEGRAPH.reset();
    rclcpp::shutdown();
    return 1;
  }

  vins::Config config;
  try {
    config = vins::Config::fromFile(config_file);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(n->get_logger(), "%s", e.what());
    POSEGRAPH.reset();
    rclcpp::shutdown();
    return 1;
  }

  CAMERAPOSEVISUAL.setScale(0.1);
  CAMERAPOSEVISUAL.setLineWidth(0.01);

  std::string IMAGE_TOPIC;
  int LOAD_PREVIOUS_POSE_GRAPH;

  ROW = config.getInt("image_height");
  COL = config.getInt("image_width");
  std::string pkg_path = ament_index_cpp::get_package_share_directory("loop_fusion");
  string vocabulary_file = pkg_path + "/support_files/brief_k10L6.bin";
  cout << "vocabulary_file" << vocabulary_file << '\n';
  POSEGRAPH->loadVocabulary(vocabulary_file);

  BRIEF_PATTERN_FILE = pkg_path + "/support_files/brief_pattern.yml";
  cout << "BRIEF_PATTERN_FILE" << BRIEF_PATTERN_FILE << '\n';

  std::string cam0Calib = config.getString("cam0_calib");
  std::string cam0Path = vins::resolvePath(config_file, cam0Calib);
  printf("cam calib path: %s\n", cam0Path.c_str());
  M_CAMERA = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(cam0Path);

  IMAGE_TOPIC = config.getString("image0_topic");
  POSE_GRAPH_SAVE_PATH = config.getString("pose_graph_save_path");
  VINS_RESULT_PATH = config.getString("output_path");
  DEBUG_IMAGE = config.getInt("save_image");

  LOAD_PREVIOUS_POSE_GRAPH = config.getInt("load_previous_pose_graph");
  IMAGE_TOPIC = n->declare_parameter<std::string>("image0_topic", IMAGE_TOPIC);
  POSE_GRAPH_SAVE_PATH = n->declare_parameter<std::string>("pose_graph_save_path", POSE_GRAPH_SAVE_PATH);
  VINS_RESULT_PATH = n->declare_parameter<std::string>("output_path", VINS_RESULT_PATH);
  POSE_GRAPH_SAVE_PATH = vins::resolvePath(config_file, POSE_GRAPH_SAVE_PATH);
  VINS_RESULT_PATH = vins::resolvePath(config_file, VINS_RESULT_PATH);
  std::filesystem::create_directories(POSE_GRAPH_SAVE_PATH);
  std::filesystem::create_directories(VINS_RESULT_PATH);
  // Legacy save code concatenates filenames, so keep a trailing separator.
  POSE_GRAPH_SAVE_PATH += "/";
  VINS_RESULT_PATH += "/vio_loop.csv";
  std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
  fout.close();

  int USE_IMU = config.getInt("imu");
  POSEGRAPH->setIMUFlag(USE_IMU);

  if (LOAD_PREVIOUS_POSE_GRAPH) {
    printf("load pose graph\n");
    M_PROCESS.lock();
    POSEGRAPH->loadPoseGraph();
    M_PROCESS.unlock();
    printf("load pose graph finish\n");
    LOAD_FLAG = 1;
  } else {
    printf("no previous pose graph\n");
    LOAD_FLAG = 1;
  }

  auto sub_vio =
    n->create_subscription<nav_msgs::msg::Odometry>(input_odometry_topic, rclcpp::QoS(100), vio_callback);
  auto sub_image = n->create_subscription<sensor_msgs::msg::Image>(
    IMAGE_TOPIC, rclcpp::SensorDataQoS().keep_last(2000), image_callback);
  auto sub_pose = n->create_subscription<nav_msgs::msg::Odometry>(
    "/vins_estimator/keyframe_pose", rclcpp::QoS(100), pose_callback);
  auto sub_extrinsic = n->create_subscription<nav_msgs::msg::Odometry>(
    "/vins_estimator/extrinsic", rclcpp::QoS(100), extrinsic_callback);
  auto sub_point = n->create_subscription<sensor_msgs::msg::PointCloud>(
    "/vins_estimator/keyframe_point", rclcpp::QoS(100), point_callback);
  auto sub_margin_point = n->create_subscription<sensor_msgs::msg::PointCloud>(
    "/vins_estimator/margin_cloud", rclcpp::QoS(100), margin_point_callback);

  PUB_MATCH_IMG = n->create_publisher<sensor_msgs::msg::Image>("~/match_image", rclcpp::QoS(100));
  PUB_CAMERA_POSE_VISUAL =
    n->create_publisher<visualization_msgs::msg::MarkerArray>("~/camera_pose_visual", rclcpp::QoS(100));
  PUB_POINT_CLOUD =
    n->create_publisher<sensor_msgs::msg::PointCloud>("~/point_cloud_loop_rect", rclcpp::QoS(100));
  PUB_MARGIN_CLOUD =
    n->create_publisher<sensor_msgs::msg::PointCloud>("~/margin_cloud_loop_rect", rclcpp::QoS(100));
  PUB_ODOMETRY_RECT = n->create_publisher<nav_msgs::msg::Odometry>("~/odometry_rect", rclcpp::QoS(100));
  if (publish_primary)
    PUB_PRIMARY_ODOMETRY =
      n->create_publisher<nav_msgs::msg::Odometry>(primary_odometry_topic, rclcpp::QoS(100));

  if (publish_visualization) {
    PUB_PRIMARY_PATH = n->create_publisher<nav_msgs::msg::Path>("/vins_estimator/path", rclcpp::QoS(1));
    PRIMARY_TF = std::make_unique<tf2_ros::TransformBroadcaster>(n);
  }

  std::thread measurement_process;
  auto save_service = n->create_service<std_srvs::srv::Trigger>(
    "~/save_pose_graph",
    [](const std_srvs::srv::Trigger::Request::SharedPtr, std_srvs::srv::Trigger::Response::SharedPtr _response)
    {
      std::lock_guard<std::mutex> lock(M_PROCESS);
      POSEGRAPH->savePoseGraph();
      _response->success = true;
      _response->message = "Pose graph saved";
    });
  auto sequence_service = n->create_service<std_srvs::srv::Trigger>(
    "~/new_sequence",
    [](const std_srvs::srv::Trigger::Request::SharedPtr, std_srvs::srv::Trigger::Response::SharedPtr _response)
    {
      std::lock_guard<std::mutex> lock(M_PROCESS);
      new_sequence();
      _response->success = true;
      _response->message = "New sequence started";
    });

  measurement_process = std::thread(process);


  rclcpp::spin(n);
  rclcpp::shutdown();
  measurement_process.join();
  POSEGRAPH.reset();
  PRIMARY_TF.reset();
  PRIMARY_PATH.reset();
  PUB_PRIMARY_PATH.reset();
  PUB_MATCH_IMG.reset();
  PUB_CAMERA_POSE_VISUAL.reset();
  PUB_POINT_CLOUD.reset();
  PUB_MARGIN_CLOUD.reset();
  PUB_ODOMETRY_RECT.reset();
  PUB_PRIMARY_ODOMETRY.reset();
  return 0;
}
