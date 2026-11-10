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

#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DBoW/TemplatedDatabase.h"
#include "ThirdParty/DBoW/TemplatedVocabulary.h"
#include "ThirdParty/DVision/DVision.h"
#include "keyframe.h"
#include "odometry_correction.h"
#include "utility/CameraPoseVisualization.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include <assert.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <mutex>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <stdio.h>
#include <string>
#include <thread>
#include <vins_common/ros_utils.hpp>


#define SHOW_S_EDGE false
#define SHOW_L_EDGE true
#define SAVE_LOOP_PATH true

using namespace DVision;
using namespace DBoW2;

class PoseGraph {
  std::atomic<bool> stopping_{false};

public:
  PoseGraph();
  ~PoseGraph();
  void registerPub(const rclcpp::Node::SharedPtr & _n);
  void addKeyFrame(KeyFrame * _cur_kf, bool _flag_detect_loop);
  void loadKeyFrame(KeyFrame * _cur_kf, bool _flag_detect_loop);
  void loadVocabulary(std::string _voc_path);
  void setIMUFlag(bool _use_imu);
  KeyFrame * getKeyFrame(int _index);
  nav_msgs::msg::Path path_[10];
  nav_msgs::msg::Path base_path_;
  CameraPoseVisualization * posegraph_visualization_;
  void savePoseGraph();
  void loadPoseGraph();
  void publish();
  // Caller must hold the node's m_process to stabilize w_* (sequence
  // alignment). Internally locks only m_drift, never m_keyframelist.
  loop_fusion::Correction_s getCorrection();
  Vector3d t_drift_;
  double yaw_drift_;
  Matrix3d r_drift_;
  // world frame( base sequence or first sequence)<----> cur sequence frame
  Vector3d w_t_vio_;
  Matrix3d w_r_vio_;


private:
  int detectLoop(KeyFrame * _keyframe, int _frame_index);
  void addKeyFrameIntoVoc(KeyFrame * _keyframe);
  void optimize4DoF();
  void optimize6DoF();
  void updatePath();
  list<KeyFrame *> keyframelist_;
  std::mutex m_keyframelist_;
  std::mutex m_optimize_buf_;
  std::mutex m_path_;
  std::mutex m_drift_;
  std::thread t_optimization_;
  std::queue<int> optimize_buf_;

  int global_index_;
  int sequence_cnt_;
  vector<bool> sequence_loop_;
  map<int, cv::Mat> image_pool_;
  int earliest_loop_index_;
  int base_sequence_;
  bool use_imu_;

  BriefDatabase db_;
  BriefVocabulary * voc_;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_pg_path_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_base_path_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_pose_graph_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_[10];
};

template <typename T> inline void QuaternionInverse(const T _q[4], T _q_inverse[4])
{
  _q_inverse[0] = _q[0];
  _q_inverse[1] = -_q[1];
  _q_inverse[2] = -_q[2];
  _q_inverse[3] = -_q[3];
};

template <typename T> T NormalizeAngle(const T & _angle_degrees)
{
  if (_angle_degrees > T(180.0))
    return _angle_degrees - T(360.0);
  else if (_angle_degrees < T(-180.0))
    return _angle_degrees + T(360.0);
  else
    return _angle_degrees;
};

struct AngleManifold_s {
  template <typename T>
  // Ceres AutoDiffManifold requires this method name.
  // NOLINTNEXTLINE(readability-identifier-naming)
  bool Plus(const T * _x, const T * _delta, T * _result) const
  {
    _result[0] = NormalizeAngle(_x[0] + _delta[0]);
    return true;
  }
  template <typename T>
  // Ceres AutoDiffManifold requires this method name.
  // NOLINTNEXTLINE(readability-identifier-naming)
  bool Minus(const T * _y, const T * _x, T * _result) const
  {
    _result[0] = NormalizeAngle(_y[0] - _x[0]);
    return true;
  }
  static ceres::Manifold * create() { return new ceres::AutoDiffManifold<AngleManifold_s, 1, 1>; }
};

template <typename T> void YawPitchRollToRotationMatrix(const T _yaw, const T _pitch, const T _roll, T _R[9])
{

  T y = _yaw / T(180.0) * T(M_PI);
  T p = _pitch / T(180.0) * T(M_PI);
  T r = _roll / T(180.0) * T(M_PI);


  _R[0] = cos(y) * cos(p);
  _R[1] = -sin(y) * cos(r) + cos(y) * sin(p) * sin(r);
  _R[2] = sin(y) * sin(r) + cos(y) * sin(p) * cos(r);
  _R[3] = sin(y) * cos(p);
  _R[4] = cos(y) * cos(r) + sin(y) * sin(p) * sin(r);
  _R[5] = -cos(y) * sin(r) + sin(y) * sin(p) * cos(r);
  _R[6] = -sin(p);
  _R[7] = cos(p) * sin(r);
  _R[8] = cos(p) * cos(r);
};

template <typename T> void RotationMatrixTranspose(const T _R[9], T _inv_R[9])
{
  _inv_R[0] = _R[0];
  _inv_R[1] = _R[3];
  _inv_R[2] = _R[6];
  _inv_R[3] = _R[1];
  _inv_R[4] = _R[4];
  _inv_R[5] = _R[7];
  _inv_R[6] = _R[2];
  _inv_R[7] = _R[5];
  _inv_R[8] = _R[8];
};

template <typename T> void RotationMatrixRotatePoint(const T _R[9], const T _t[3], T _r_t[3])
{
  _r_t[0] = _R[0] * _t[0] + _R[1] * _t[1] + _R[2] * _t[2];
  _r_t[1] = _R[3] * _t[0] + _R[4] * _t[1] + _R[5] * _t[2];
  _r_t[2] = _R[6] * _t[0] + _R[7] * _t[1] + _R[8] * _t[2];
};

struct FourDofError_s {
  FourDofError_s(double _t_x, double _t_y, double _t_z, double _relative_yaw, double _pitch_i, double _roll_i)
    : t_x_(_t_x), t_y_(_t_y), t_z_(_t_z), relative_yaw_(_relative_yaw), pitch_i_(_pitch_i), roll_i_(_roll_i)
  {
  }

  template <typename T>
  bool operator()(const T * const _yaw_i, const T * _ti, const T * _yaw_j, const T * _tj, T * _residuals) const
  {
    T t_w_ij[3];
    t_w_ij[0] = _tj[0] - _ti[0];
    t_w_ij[1] = _tj[1] - _ti[1];
    t_w_ij[2] = _tj[2] - _ti[2];

    // euler to rotation
    T w_R_i[9];
    YawPitchRollToRotationMatrix(_yaw_i[0], T(pitch_i_), T(roll_i_), w_R_i);
    // rotation transpose
    T i_R_w[9];
    RotationMatrixTranspose(w_R_i, i_R_w);
    // rotation matrix rotate point
    T t_i_ij[3];
    RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

    _residuals[0] = (t_i_ij[0] - T(t_x_));
    _residuals[1] = (t_i_ij[1] - T(t_y_));
    _residuals[2] = (t_i_ij[2] - T(t_z_));
    _residuals[3] = NormalizeAngle(_yaw_j[0] - _yaw_i[0] - T(relative_yaw_));

    return true;
  }

  static ceres::CostFunction * create(
    const double _t_x, const double _t_y, const double _t_z, const double _relative_yaw,
    const double _pitch_i, const double _roll_i)
  {
    return (new ceres::AutoDiffCostFunction<FourDofError_s, 4, 1, 3, 1, 3>(
      new FourDofError_s(_t_x, _t_y, _t_z, _relative_yaw, _pitch_i, _roll_i)));
  }

  double t_x_, t_y_, t_z_;
  double relative_yaw_, pitch_i_, roll_i_;
};

struct FourDofWeightError_s {
  FourDofWeightError_s(double _t_x, double _t_y, double _t_z, double _relative_yaw, double _pitch_i, double _roll_i)
    : t_x_(_t_x), t_y_(_t_y), t_z_(_t_z), relative_yaw_(_relative_yaw), pitch_i_(_pitch_i), roll_i_(_roll_i)
  {
    weight_ = 1;
  }

  template <typename T>
  bool operator()(const T * const _yaw_i, const T * _ti, const T * _yaw_j, const T * _tj, T * _residuals) const
  {
    T t_w_ij[3];
    t_w_ij[0] = _tj[0] - _ti[0];
    t_w_ij[1] = _tj[1] - _ti[1];
    t_w_ij[2] = _tj[2] - _ti[2];

    // euler to rotation
    T w_R_i[9];
    YawPitchRollToRotationMatrix(_yaw_i[0], T(pitch_i_), T(roll_i_), w_R_i);
    // rotation transpose
    T i_R_w[9];
    RotationMatrixTranspose(w_R_i, i_R_w);
    // rotation matrix rotate point
    T t_i_ij[3];
    RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

    _residuals[0] = (t_i_ij[0] - T(t_x_)) * T(weight_);
    _residuals[1] = (t_i_ij[1] - T(t_y_)) * T(weight_);
    _residuals[2] = (t_i_ij[2] - T(t_z_)) * T(weight_);
    _residuals[3] = NormalizeAngle((_yaw_j[0] - _yaw_i[0] - T(relative_yaw_))) * T(weight_) / T(10.0);

    return true;
  }

  static ceres::CostFunction * create(
    const double _t_x, const double _t_y, const double _t_z, const double _relative_yaw,
    const double _pitch_i, const double _roll_i)
  {
    return (new ceres::AutoDiffCostFunction<FourDofWeightError_s, 4, 1, 3, 1, 3>(
      new FourDofWeightError_s(_t_x, _t_y, _t_z, _relative_yaw, _pitch_i, _roll_i)));
  }

  double t_x_, t_y_, t_z_;
  double relative_yaw_, pitch_i_, roll_i_;
  double weight_;
};

struct RelativeRtError_s {
  RelativeRtError_s(
    double _t_x, double _t_y, double _t_z, double _q_w, double _q_x, double _q_y, double _q_z, double _t_var,
    double _q_var)
    : t_x_(_t_x), t_y_(_t_y), t_z_(_t_z), q_w_(_q_w), q_x_(_q_x), q_y_(_q_y), q_z_(_q_z), t_var_(_t_var),
      q_var_(_q_var)
  {
  }

  template <typename T>
  bool operator()(const T * const _w_q_i, const T * _ti, const T * _w_q_j, const T * _tj, T * _residuals) const
  {
    T t_w_ij[3];
    t_w_ij[0] = _tj[0] - _ti[0];
    t_w_ij[1] = _tj[1] - _ti[1];
    t_w_ij[2] = _tj[2] - _ti[2];

    T i_q_w[4];
    QuaternionInverse(_w_q_i, i_q_w);

    T t_i_ij[3];
    ceres::QuaternionRotatePoint(i_q_w, t_w_ij, t_i_ij);

    _residuals[0] = (t_i_ij[0] - T(t_x_)) / T(t_var_);
    _residuals[1] = (t_i_ij[1] - T(t_y_)) / T(t_var_);
    _residuals[2] = (t_i_ij[2] - T(t_z_)) / T(t_var_);

    T relative_q[4];
    relative_q[0] = T(q_w_);
    relative_q[1] = T(q_x_);
    relative_q[2] = T(q_y_);
    relative_q[3] = T(q_z_);

    T q_i_j[4];
    ceres::QuaternionProduct(i_q_w, _w_q_j, q_i_j);

    T relative_q_inv[4];
    QuaternionInverse(relative_q, relative_q_inv);

    T error_q[4];
    ceres::QuaternionProduct(relative_q_inv, q_i_j, error_q);

    _residuals[3] = T(2) * error_q[1] / T(q_var_);
    _residuals[4] = T(2) * error_q[2] / T(q_var_);
    _residuals[5] = T(2) * error_q[3] / T(q_var_);

    return true;
  }

  static ceres::CostFunction * create(
    const double _t_x, const double _t_y, const double _t_z, const double _q_w, const double _q_x,
    const double _q_y, const double _q_z, const double _t_var, const double _q_var)
  {
    return (new ceres::AutoDiffCostFunction<RelativeRtError_s, 6, 4, 3, 4, 3>(
      new RelativeRtError_s(_t_x, _t_y, _t_z, _q_w, _q_x, _q_y, _q_z, _t_var, _q_var)));
  }

  double t_x_, t_y_, t_z_, t_norm_;
  double q_w_, q_x_, q_y_, q_z_;
  double t_var_, q_var_;
};