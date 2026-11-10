/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <vins_common/ros_utils.hpp>

#include "ekf_odometry.h"
#include "runtime_utils.h"
#include "visualization.h"


using namespace Eigen;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr PUB_ODOMETRY;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr PUB_LATEST_ODOMETRY;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr PUB_PATH;
rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr PUB_POINT_CLOUD;
rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr PUB_MARGIN_CLOUD;
rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr PUB_KEY_POSES;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr PUB_CAMERA_POSE;
rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr PUB_CAMERA_POSE_VISUAL;
nav_msgs::msg::Path PATH;

rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr PUB_KEYFRAME_POSE;
rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr PUB_KEYFRAME_POINT;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr PUB_EXTRINSIC;

rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr PUB_IMAGE_TRACK;

CameraPoseVisualization CAMERAPOSEVISUAL(1, 0, 0, 1);
static double SUM_OF_PATH = 0;
static Vector3d LAST_PATH(0.0, 0.0, 0.0);

size_t PUB_COUNTER = 0;
std::unique_ptr<tf2_ros::TransformBroadcaster> TF_BROADCASTER;

namespace
{
std::ofstream RESULT_FILE;
size_t RESULT_ROWS = 0;
vins::BoundedHistory<geometry_msgs::msg::PoseStamped> PATH_HISTORY;
vins::FrameRateGate PATH_GATE;
double PATH_PUBLISH_RATE = 1.0;
std::string WORLD_FRAME = "world", BODY_FRAME = "body", CAMERA_FRAME = "camera";

template <class T> bool hasSubscribers(const std::shared_ptr<rclcpp::Publisher<T>> & _publisher)
{
  return _publisher && (_publisher->get_subscription_count() != 0 ||
                        _publisher->get_intra_process_subscription_count() != 0);
}
}  // namespace

void resetVisualization()
{
  PATH.poses.clear();
  PATH_HISTORY.clear();
  // Permit immediate publication after a restart, even at the same timestamp.
  PATH_GATE.reset(PATH_PUBLISH_RATE);
}

bool trackImageRequested() { return hasSubscribers(PUB_IMAGE_TRACK); }

bool schurVisualizationRequested()
{
  return hasSubscribers(PUB_POINT_CLOUD) || hasSubscribers(PUB_MARGIN_CLOUD) ||
         hasSubscribers(PUB_KEYFRAME_POSE) || hasSubscribers(PUB_KEYFRAME_POINT);
}

void registerPub(const rclcpp::Node::SharedPtr & _n)
{
  rcl_interfaces::msg::ParameterDescriptor frame_descriptor;
  frame_descriptor.read_only = true;
  const bool use_raw_frames = _n->declare_parameter<bool>("use_raw_frames", false, frame_descriptor);
  WORLD_FRAME = use_raw_frames ? "world_vio" : "world";
  BODY_FRAME = use_raw_frames ? "body_vio" : "body";
  CAMERA_FRAME = use_raw_frames ? "camera_vio" : "camera";
  const int path_capacity = _n->declare_parameter<int>("path_max_poses", 2000);
  PATH_PUBLISH_RATE = _n->declare_parameter<double>("path_publish_rate", 1.0);
  if (path_capacity < 0)
    throw std::invalid_argument("path_max_poses must be nonnegative");
  PATH_HISTORY.reset(static_cast<size_t>(path_capacity));
  PATH_GATE.reset(PATH_PUBLISH_RATE);
  RESULT_FILE.open(VINS_RESULT_PATH, std::ios::app);
  if (!RESULT_FILE)
    throw std::runtime_error("Cannot open trajectory: " + VINS_RESULT_PATH);
  RESULT_FILE.setf(std::ios::fixed, std::ios::floatfield);
  RESULT_ROWS = 0;
  PUB_LATEST_ODOMETRY = _n->create_publisher<nav_msgs::msg::Odometry>("~/imu_propagate", rclcpp::QoS(100));
  PUB_PATH = _n->create_publisher<nav_msgs::msg::Path>("~/path", rclcpp::QoS(1));
  PUB_ODOMETRY = _n->create_publisher<nav_msgs::msg::Odometry>("~/odometry", rclcpp::QoS(100));
  PUB_POINT_CLOUD = _n->create_publisher<sensor_msgs::msg::PointCloud>("~/point_cloud", rclcpp::QoS(100));
  PUB_MARGIN_CLOUD = _n->create_publisher<sensor_msgs::msg::PointCloud>("~/margin_cloud", rclcpp::QoS(100));
  PUB_KEY_POSES = _n->create_publisher<visualization_msgs::msg::Marker>("~/key_poses", rclcpp::QoS(100));
  PUB_CAMERA_POSE = _n->create_publisher<nav_msgs::msg::Odometry>("~/camera_pose", rclcpp::QoS(100));
  PUB_CAMERA_POSE_VISUAL =
    _n->create_publisher<visualization_msgs::msg::MarkerArray>("~/camera_pose_visual", rclcpp::QoS(100));
  PUB_KEYFRAME_POSE = _n->create_publisher<nav_msgs::msg::Odometry>("~/keyframe_pose", rclcpp::QoS(100));
  PUB_KEYFRAME_POINT = _n->create_publisher<sensor_msgs::msg::PointCloud>("~/keyframe_point", rclcpp::QoS(100));
  PUB_EXTRINSIC = _n->create_publisher<nav_msgs::msg::Odometry>("~/extrinsic", rclcpp::QoS(100));
  PUB_IMAGE_TRACK = _n->create_publisher<sensor_msgs::msg::Image>("~/image_track", rclcpp::QoS(1));

  TF_BROADCASTER = std::make_unique<tf2_ros::TransformBroadcaster>(_n);
  CAMERAPOSEVISUAL.setScale(0.1);
  CAMERAPOSEVISUAL.setLineWidth(0.01);
}

void unregisterPub()
{
  RESULT_FILE.close();
  resetVisualization();
  TF_BROADCASTER.reset();
  PUB_ODOMETRY.reset();
  PUB_LATEST_ODOMETRY.reset();
  PUB_PATH.reset();
  PUB_POINT_CLOUD.reset();
  PUB_MARGIN_CLOUD.reset();
  PUB_KEY_POSES.reset();
  PUB_CAMERA_POSE.reset();
  PUB_CAMERA_POSE_VISUAL.reset();
  PUB_KEYFRAME_POSE.reset();
  PUB_KEYFRAME_POINT.reset();
  PUB_EXTRINSIC.reset();
  PUB_IMAGE_TRACK.reset();
}

void pubLatestOdometry(
  const Eigen::Vector3d & _P, const Eigen::Quaterniond & _Q, const Eigen::Vector3d & _V, double _t)
{
  if (!hasSubscribers(PUB_LATEST_ODOMETRY))
    return;
  nav_msgs::msg::Odometry odometry;
  odometry.header.stamp = vins::stamp(_t);
  odometry.header.frame_id = WORLD_FRAME;
  odometry.pose.pose.position.x = _P.x();
  odometry.pose.pose.position.y = _P.y();
  odometry.pose.pose.position.z = _P.z();
  odometry.pose.pose.orientation.x = _Q.x();
  odometry.pose.pose.orientation.y = _Q.y();
  odometry.pose.pose.orientation.z = _Q.z();
  odometry.pose.pose.orientation.w = _Q.w();
  odometry.twist.twist.linear.x = _V.x();
  odometry.twist.twist.linear.y = _V.y();
  odometry.twist.twist.linear.z = _V.z();
  PUB_LATEST_ODOMETRY->publish(odometry);
}

void pubLatestOdometry(const vins::backend::EkfState & _state, const Eigen::Vector3d & _raw_gyro, double _imu_dt)
{
  if (!hasSubscribers(PUB_LATEST_ODOMETRY))
    return;
  nav_msgs::msg::Odometry odometry;
  odometry.header.stamp = vins::stamp(_state.timestamp());
  odometry.header.frame_id = WORLD_FRAME;
  odometry.child_frame_id = BODY_FRAME;
  vins::fillEkfOdometry(odometry, _state, _raw_gyro, _imu_dt);
  PUB_LATEST_ODOMETRY->publish(odometry);
}

void pubTrackImage(const cv::Mat & _imgTrack, const double _t)
{
  if (!trackImageRequested())
    return;
  std_msgs::msg::Header header;
  header.frame_id = WORLD_FRAME;
  header.stamp = vins::stamp(_t);
  sensor_msgs::msg::Image::SharedPtr imgTrackMsg = cv_bridge::CvImage(header, "bgr8", _imgTrack).toImageMsg();
  PUB_IMAGE_TRACK->publish(*imgTrackMsg);
}


void printStatistics(const Estimator & _estimator, double _t)
{
  if (_estimator.solver_flag_ != Estimator::SolverFlag_e::NON_LINEAR)
    return;
  //printf("position: %f, %f, %f\r", estimator.Ps[WINDOW_SIZE].x(), estimator.Ps[WINDOW_SIZE].y(), estimator.Ps[WINDOW_SIZE].z());
  RCLCPP_DEBUG_STREAM(vins::logger(), "position: " << _estimator.Ps_[WINDOW_SIZE].transpose());
  RCLCPP_DEBUG_STREAM(vins::logger(), "orientation: " << _estimator.Vs_[WINDOW_SIZE].transpose());
  if (ESTIMATE_EXTRINSIC) {
    cv::FileStorage fs(EX_CALIB_RESULT_PATH, cv::FileStorage::WRITE);
    for (int i = 0; i < NUM_OF_CAM; i++) {
      //RCLCPP_DEBUG(vins::logger(), "calibration result for camera %d", i);
      RCLCPP_DEBUG_STREAM(vins::logger(), "extirnsic tic: " << _estimator.tic_[i].transpose());
      RCLCPP_DEBUG_STREAM(vins::logger(), "extrinsic ric: " << Utility::r2ypr(_estimator.ric_[i]).transpose());

      Eigen::Matrix4d eigen_T = Eigen::Matrix4d::Identity();
      eigen_T.block<3, 3>(0, 0) = _estimator.ric_[i];
      eigen_T.block<3, 1>(0, 3) = _estimator.tic_[i];
      cv::Mat cv_T;
      cv::eigen2cv(eigen_T, cv_T);
      if (i == 0)
        fs << "body_T_cam0" << cv_T;
      else
        fs << "body_T_cam1" << cv_T;
    }
    fs.release();
  }

  static double sum_of_time = 0;
  static int sum_of_calculation = 0;
  sum_of_time += _t;
  sum_of_calculation++;
  RCLCPP_DEBUG(vins::logger(), "vo solver costs: %f ms", _t);
  RCLCPP_DEBUG(vins::logger(), "average of time %f ms", sum_of_time / sum_of_calculation);

  SUM_OF_PATH += (_estimator.Ps_[WINDOW_SIZE] - LAST_PATH).norm();
  LAST_PATH = _estimator.Ps_[WINDOW_SIZE];
  RCLCPP_DEBUG(vins::logger(), "sum of path %f", SUM_OF_PATH);
  if (ESTIMATE_TD)
    RCLCPP_DEBUG(vins::logger(), "td %f", _estimator.td_);
}

void pubOdometry(const Estimator & _estimator, const std_msgs::msg::Header & _header)
{
  if (_estimator.solver_flag_ == Estimator::SolverFlag_e::NON_LINEAR) {
    nav_msgs::msg::Odometry odometry;
    odometry.header = _header;
    odometry.header.frame_id = WORLD_FRAME;
    odometry.child_frame_id = BODY_FRAME;
    Quaterniond tmp_Q;
    tmp_Q = Quaterniond(_estimator.Rs_[WINDOW_SIZE]);
    odometry.pose.pose.position.x = _estimator.Ps_[WINDOW_SIZE].x();
    odometry.pose.pose.position.y = _estimator.Ps_[WINDOW_SIZE].y();
    odometry.pose.pose.position.z = _estimator.Ps_[WINDOW_SIZE].z();
    odometry.pose.pose.orientation.x = tmp_Q.x();
    odometry.pose.pose.orientation.y = tmp_Q.y();
    odometry.pose.pose.orientation.z = tmp_Q.z();
    odometry.pose.pose.orientation.w = tmp_Q.w();
    vins::fillBodyLinearVelocity(odometry, _estimator.Rs_[WINDOW_SIZE], _estimator.Vs_[WINDOW_SIZE]);
    if (_estimator.schurBackendActive()) {
      // The EKF state lives at image time + td, including its covariance.
      odometry.header.stamp = vins::stamp(_estimator.schur_state_.timestamp());
      vins::fillEkfOdometry(odometry, _estimator.schur_state_, _estimator.gyr_0_, _estimator.schur_last_imu_dt_);
    }
    PUB_ODOMETRY->publish(odometry);

    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header = odometry.header;
    pose_stamped.header.frame_id = WORLD_FRAME;
    pose_stamped.pose = odometry.pose.pose;
    PATH_HISTORY.push(pose_stamped);
    if (PATH_HISTORY.size() && hasSubscribers(PUB_PATH) && PATH_GATE.accept(vins::seconds(_header.stamp))) {
      PATH.header = odometry.header;
      PATH.header.frame_id = WORLD_FRAME;
      PATH.poses.assign(PATH_HISTORY.items().begin(), PATH_HISTORY.items().end());
      PUB_PATH->publish(PATH);
    }

    // write result to file
    auto & foutC = RESULT_FILE;
    foutC.precision(0);
    foutC << vins::seconds(_header.stamp) * 1e9 << ",";
    foutC.precision(5);
    foutC << _estimator.Ps_[WINDOW_SIZE].x() << "," << _estimator.Ps_[WINDOW_SIZE].y() << ","
          << _estimator.Ps_[WINDOW_SIZE].z() << "," << tmp_Q.w() << "," << tmp_Q.x() << "," << tmp_Q.y()
          << "," << tmp_Q.z() << "," << _estimator.Vs_[WINDOW_SIZE].x() << ","
          << _estimator.Vs_[WINDOW_SIZE].y() << "," << _estimator.Vs_[WINDOW_SIZE].z() << ",\n";
    if (++RESULT_ROWS % 50 == 0)
      foutC.flush();
    Eigen::Vector3d tmp_T = _estimator.Ps_[WINDOW_SIZE];
    RCLCPP_DEBUG(
      vins::logger(), "time: %f, t: %f %f %f q: %f %f %f %f", vins::seconds(_header.stamp), tmp_T.x(),
      tmp_T.y(), tmp_T.z(), tmp_Q.w(), tmp_Q.x(), tmp_Q.y(), tmp_Q.z());
  }
}

void pubKeyPoses(const Estimator & _estimator, const std_msgs::msg::Header & _header)
{
  if (!hasSubscribers(PUB_KEY_POSES))
    return;
  if (_estimator.key_poses_.empty())
    return;
  visualization_msgs::msg::Marker key_poses;
  key_poses.header = _header;
  key_poses.header.frame_id = WORLD_FRAME;
  key_poses.ns = "key_poses";
  key_poses.type = visualization_msgs::msg::Marker::SPHERE_LIST;
  key_poses.action = visualization_msgs::msg::Marker::ADD;
  key_poses.pose.orientation.w = 1.0;
  key_poses.lifetime = builtin_interfaces::msg::Duration{};

  //static int key_poses_id = 0;
  key_poses.id = 0;  //key_poses_id++;
  key_poses.scale.x = 0.05;
  key_poses.scale.y = 0.05;
  key_poses.scale.z = 0.05;
  key_poses.color.r = 1.0;
  key_poses.color.a = 1.0;

  for (int i = 0; i <= WINDOW_SIZE; i++) {
    geometry_msgs::msg::Point pose_marker;
    Vector3d correct_pose;
    correct_pose = _estimator.key_poses_[i];
    pose_marker.x = correct_pose.x();
    pose_marker.y = correct_pose.y();
    pose_marker.z = correct_pose.z();
    key_poses.points.push_back(pose_marker);
  }
  PUB_KEY_POSES->publish(key_poses);
}

void pubCameraPose(const Estimator & _estimator, const std_msgs::msg::Header & _header)
{
  if (!hasSubscribers(PUB_CAMERA_POSE) && !hasSubscribers(PUB_CAMERA_POSE_VISUAL))
    return;
  int idx2 = WINDOW_SIZE - 1;

  if (_estimator.solver_flag_ == Estimator::SolverFlag_e::NON_LINEAR) {
    int i = idx2;
    Vector3d P = _estimator.Ps_[i] + _estimator.Rs_[i] * _estimator.tic_[0];
    Quaterniond R = Quaterniond(_estimator.Rs_[i] * _estimator.ric_[0]);

    nav_msgs::msg::Odometry odometry;
    odometry.header = _header;
    odometry.header.frame_id = WORLD_FRAME;
    odometry.pose.pose.position.x = P.x();
    odometry.pose.pose.position.y = P.y();
    odometry.pose.pose.position.z = P.z();
    odometry.pose.pose.orientation.x = R.x();
    odometry.pose.pose.orientation.y = R.y();
    odometry.pose.pose.orientation.z = R.z();
    odometry.pose.pose.orientation.w = R.w();

    PUB_CAMERA_POSE->publish(odometry);

    CAMERAPOSEVISUAL.reset();
    CAMERAPOSEVISUAL.addPose(P, R);
    if (STEREO) {
      Vector3d P = _estimator.Ps_[i] + _estimator.Rs_[i] * _estimator.tic_[1];
      Quaterniond R = Quaterniond(_estimator.Rs_[i] * _estimator.ric_[1]);
      CAMERAPOSEVISUAL.addPose(P, R);
    }
    CAMERAPOSEVISUAL.publishBy(PUB_CAMERA_POSE_VISUAL, odometry.header);
  }
}


void pubPointCloud(const Estimator & _estimator, const std_msgs::msg::Header & _header)
{
  if (!hasSubscribers(PUB_POINT_CLOUD) && !hasSubscribers(PUB_MARGIN_CLOUD))
    return;

  if (_estimator.schurBackendActive()) {
    const auto & snapshot = _estimator.schurVisualizationSnapshot();
    sensor_msgs::msg::PointCloud point_cloud;
    point_cloud.header = _header;
    point_cloud.header.frame_id = WORLD_FRAME;
    if (snapshot.valid_) {
      point_cloud.header.stamp = _estimator.schurVisualizationStamp();
      point_cloud.points.reserve(snapshot.cloud_points_.size());
      for (const auto & point : snapshot.cloud_points_) {
        if (!point.world_point_.allFinite())
          continue;
        geometry_msgs::msg::Point32 p;
        p.x = static_cast<float>(point.world_point_.x());
        p.y = static_cast<float>(point.world_point_.y());
        p.z = static_cast<float>(point.world_point_.z());
        point_cloud.points.push_back(p);
      }
    }
    if (hasSubscribers(PUB_POINT_CLOUD))
      PUB_POINT_CLOUD->publish(point_cloud);

    sensor_msgs::msg::PointCloud margin_cloud;
    margin_cloud.header = point_cloud.header;
    margin_cloud.points.reserve(snapshot.margin_points_.size());
    for (const auto & point : snapshot.margin_points_) {
      if (!point.world_point_.allFinite())
        continue;
      geometry_msgs::msg::Point32 p;
      p.x = static_cast<float>(point.world_point_.x());
      p.y = static_cast<float>(point.world_point_.y());
      p.z = static_cast<float>(point.world_point_.z());
      margin_cloud.points.push_back(p);
    }
    if (hasSubscribers(PUB_MARGIN_CLOUD))
      PUB_MARGIN_CLOUD->publish(margin_cloud);
    return;
  }

  sensor_msgs::msg::PointCloud point_cloud, loop_point_cloud;
  point_cloud.header = _header;
  loop_point_cloud.header = _header;
  point_cloud.header.frame_id = WORLD_FRAME;
  loop_point_cloud.header.frame_id = WORLD_FRAME;


  for (auto & it_per_id : _estimator.f_manager_.feature_) {
    int used_num;
    used_num = it_per_id.feature_per_frame_.size();
    if (used_num < 2 || it_per_id.start_frame_ >= WINDOW_SIZE - 2)
      continue;
    if (it_per_id.start_frame_ > WINDOW_SIZE * 3.0 / 4.0 || it_per_id.solve_flag_ != 1)
      continue;
    int imu_i = it_per_id.start_frame_;
    Vector3d pts_i = it_per_id.feature_per_frame_[0].point_ * it_per_id.estimated_depth_;
    Vector3d w_pts_i =
      _estimator.Rs_[imu_i] * (_estimator.ric_[0] * pts_i + _estimator.tic_[0]) + _estimator.Ps_[imu_i];

    geometry_msgs::msg::Point32 p;
    p.x = w_pts_i(0);
    p.y = w_pts_i(1);
    p.z = w_pts_i(2);
    point_cloud.points.push_back(p);
  }
  PUB_POINT_CLOUD->publish(point_cloud);


  // pub margined potin
  sensor_msgs::msg::PointCloud margin_cloud;
  margin_cloud.header = _header;
  margin_cloud.header.frame_id = WORLD_FRAME;

  for (auto & it_per_id : _estimator.f_manager_.feature_) {
    int used_num;
    used_num = it_per_id.feature_per_frame_.size();
    if (used_num < 2 || it_per_id.start_frame_ >= WINDOW_SIZE - 2)
      continue;
    //if (it_per_id->start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id->solve_flag != 1)
    //        continue;

    if (it_per_id.start_frame_ == 0 && it_per_id.feature_per_frame_.size() <= 2 && it_per_id.solve_flag_ == 1) {
      int imu_i = it_per_id.start_frame_;
      Vector3d pts_i = it_per_id.feature_per_frame_[0].point_ * it_per_id.estimated_depth_;
      Vector3d w_pts_i =
        _estimator.Rs_[imu_i] * (_estimator.ric_[0] * pts_i + _estimator.tic_[0]) + _estimator.Ps_[imu_i];

      geometry_msgs::msg::Point32 p;
      p.x = w_pts_i(0);
      p.y = w_pts_i(1);
      p.z = w_pts_i(2);
      margin_cloud.points.push_back(p);
    }
  }
  PUB_MARGIN_CLOUD->publish(margin_cloud);
}


void pubTF(const Estimator & _estimator, const std_msgs::msg::Header & _header)
{
  if (_estimator.solver_flag_ != Estimator::SolverFlag_e::NON_LINEAR)
    return;
  geometry_msgs::msg::TransformStamped body, camera;
  body.header = _header;
  if (_estimator.schurBackendActive())
    body.header.stamp = vins::stamp(_estimator.schur_state_.timestamp());
  body.header.frame_id = WORLD_FRAME;
  body.child_frame_id = BODY_FRAME;
  body.transform.translation.x = _estimator.Ps_[WINDOW_SIZE].x();
  body.transform.translation.y = _estimator.Ps_[WINDOW_SIZE].y();
  body.transform.translation.z = _estimator.Ps_[WINDOW_SIZE].z();
  Quaterniond q_body(_estimator.Rs_[WINDOW_SIZE]);
  if (_estimator.schurBackendActive()) {
    const auto & imu = _estimator.schur_state_.imu();
    body.transform.translation.x = imu.p_.x();
    body.transform.translation.y = imu.p_.y();
    body.transform.translation.z = imu.p_.z();
    q_body = Quaterniond(imu.q_(0), imu.q_(1), imu.q_(2), imu.q_(3));
  }
  body.transform.rotation.x = q_body.x();
  body.transform.rotation.y = q_body.y();
  body.transform.rotation.z = q_body.z();
  body.transform.rotation.w = q_body.w();
  camera.header = body.header;
  camera.header.frame_id = BODY_FRAME;
  camera.child_frame_id = CAMERA_FRAME;
  camera.transform.translation.x = _estimator.tic_[0].x();
  camera.transform.translation.y = _estimator.tic_[0].y();
  camera.transform.translation.z = _estimator.tic_[0].z();
  const Quaterniond q_camera(_estimator.ric_[0]);
  camera.transform.rotation.x = q_camera.x();
  camera.transform.rotation.y = q_camera.y();
  camera.transform.rotation.z = q_camera.z();
  camera.transform.rotation.w = q_camera.w();
  TF_BROADCASTER->sendTransform(std::vector<geometry_msgs::msg::TransformStamped>{body, camera});

  nav_msgs::msg::Odometry odometry;
  odometry.header = _header;
  odometry.header.frame_id = WORLD_FRAME;
  odometry.pose.pose.position.x = _estimator.tic_[0].x();
  odometry.pose.pose.position.y = _estimator.tic_[0].y();
  odometry.pose.pose.position.z = _estimator.tic_[0].z();
  Quaterniond tmp_q{_estimator.ric_[0]};
  odometry.pose.pose.orientation.x = tmp_q.x();
  odometry.pose.pose.orientation.y = tmp_q.y();
  odometry.pose.pose.orientation.z = tmp_q.z();
  odometry.pose.pose.orientation.w = tmp_q.w();
  PUB_EXTRINSIC->publish(odometry);
}

void pubKeyframe(const Estimator & _estimator)
{
  if (!hasSubscribers(PUB_KEYFRAME_POSE) && !hasSubscribers(PUB_KEYFRAME_POINT))
    return;

  if (_estimator.schurBackendActive()) {
    if (_estimator.marginalization_flag_ != Estimator::MARGIN_OLD)
      return;

    const auto & snapshot = _estimator.schurVisualizationSnapshot();
    if (!snapshot.valid_ || snapshot.keyframe_points_.empty())
      return;

    const builtin_interfaces::msg::Time stamp = _estimator.schurVisualizationStamp();
    const Eigen::Quaterniond q(
      snapshot.q_wxyz_(0), snapshot.q_wxyz_(1), snapshot.q_wxyz_(2), snapshot.q_wxyz_(3));

    nav_msgs::msg::Odometry odometry;
    odometry.header.stamp = stamp;
    odometry.header.frame_id = WORLD_FRAME;
    odometry.child_frame_id = BODY_FRAME;
    odometry.pose.pose.position.x = snapshot.p_.x();
    odometry.pose.pose.position.y = snapshot.p_.y();
    odometry.pose.pose.position.z = snapshot.p_.z();
    odometry.pose.pose.orientation.x = q.x();
    odometry.pose.pose.orientation.y = q.y();
    odometry.pose.pose.orientation.z = q.z();
    odometry.pose.pose.orientation.w = q.w();
    if (hasSubscribers(PUB_KEYFRAME_POSE))
      PUB_KEYFRAME_POSE->publish(odometry);

    sensor_msgs::msg::PointCloud point_cloud;
    point_cloud.header.stamp = stamp;
    point_cloud.header.frame_id = WORLD_FRAME;
    point_cloud.points.reserve(snapshot.keyframe_points_.size());
    point_cloud.channels.reserve(snapshot.keyframe_points_.size());
    for (const auto & point : snapshot.keyframe_points_) {
      if (!point.world_point_.allFinite() || !point.normalized_uv_.allFinite() || !point.pixel_uv_.allFinite())
        continue;
      geometry_msgs::msg::Point32 p;
      p.x = static_cast<float>(point.world_point_.x());
      p.y = static_cast<float>(point.world_point_.y());
      p.z = static_cast<float>(point.world_point_.z());
      point_cloud.points.push_back(p);

      sensor_msgs::msg::ChannelFloat32 p_2d;
      p_2d.values = {
        static_cast<float>(point.normalized_uv_.x()), static_cast<float>(point.normalized_uv_.y()),
        static_cast<float>(point.pixel_uv_.x()), static_cast<float>(point.pixel_uv_.y()),
        static_cast<float>(point.feature_id_)};
      point_cloud.channels.push_back(std::move(p_2d));
    }
    if (!point_cloud.points.empty() && hasSubscribers(PUB_KEYFRAME_POINT))
      PUB_KEYFRAME_POINT->publish(point_cloud);
    return;
  }

  // pub camera pose, 2D-3D points of keyframe
  if (_estimator.solver_flag_ == Estimator::SolverFlag_e::NON_LINEAR && _estimator.marginalization_flag_ == 0) {
    int i = WINDOW_SIZE - 2;
    //Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
    Vector3d P = _estimator.Ps_[i];
    Quaterniond R = Quaterniond(_estimator.Rs_[i]);

    nav_msgs::msg::Odometry odometry;
    odometry.header.stamp = vins::stamp(_estimator.Headers_[WINDOW_SIZE - 2]);
    odometry.header.frame_id = WORLD_FRAME;
    odometry.pose.pose.position.x = P.x();
    odometry.pose.pose.position.y = P.y();
    odometry.pose.pose.position.z = P.z();
    odometry.pose.pose.orientation.x = R.x();
    odometry.pose.pose.orientation.y = R.y();
    odometry.pose.pose.orientation.z = R.z();
    odometry.pose.pose.orientation.w = R.w();
    //printf("time: %f t: %f %f %f r: %f %f %f %f\n", vins::seconds(odometry.header.stamp), P.x(), P.y(), P.z(), R.w(), R.x(), R.y(), R.z());

    PUB_KEYFRAME_POSE->publish(odometry);


    sensor_msgs::msg::PointCloud point_cloud;
    point_cloud.header.stamp = vins::stamp(_estimator.Headers_[WINDOW_SIZE - 2]);
    point_cloud.header.frame_id = WORLD_FRAME;
    for (auto & it_per_id : _estimator.f_manager_.feature_) {
      int frame_size = it_per_id.feature_per_frame_.size();
      if (
        it_per_id.start_frame_ < WINDOW_SIZE - 2 &&
        it_per_id.start_frame_ + frame_size - 1 >= WINDOW_SIZE - 2 && it_per_id.solve_flag_ == 1) {

        int imu_i = it_per_id.start_frame_;
        Vector3d pts_i = it_per_id.feature_per_frame_[0].point_ * it_per_id.estimated_depth_;
        Vector3d w_pts_i =
          _estimator.Rs_[imu_i] * (_estimator.ric_[0] * pts_i + _estimator.tic_[0]) + _estimator.Ps_[imu_i];
        geometry_msgs::msg::Point32 p;
        p.x = w_pts_i(0);
        p.y = w_pts_i(1);
        p.z = w_pts_i(2);
        point_cloud.points.push_back(p);

        int imu_j = WINDOW_SIZE - 2 - it_per_id.start_frame_;
        sensor_msgs::msg::ChannelFloat32 p_2d;
        p_2d.values.push_back(it_per_id.feature_per_frame_[imu_j].point_.x());
        p_2d.values.push_back(it_per_id.feature_per_frame_[imu_j].point_.y());
        p_2d.values.push_back(it_per_id.feature_per_frame_[imu_j].uv_.x());
        p_2d.values.push_back(it_per_id.feature_per_frame_[imu_j].uv_.y());
        p_2d.values.push_back(it_per_id.feature_id_);
        point_cloud.channels.push_back(p_2d);
      }
    }
    PUB_KEYFRAME_POINT->publish(point_cloud);
  }
}
