/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <vins_common/ros_utils.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>

#include "../utility/visualization.h"
#include "estimator.h"
#include "solver_timeout.h"

namespace
{
struct CeresParameterSnapshot_s {
  std::array<double, (WINDOW_SIZE + 1) * SIZE_POSE> pose_;
  std::array<double, (WINDOW_SIZE + 1) * SIZE_SPEEDBIAS> speed_bias_;
  std::array<double, NUM_OF_F * SIZE_FEATURE> feature_;
  std::array<double, 2 * SIZE_POSE> ex_pose_;
  std::array<double, 1> td_;
};

// Frame period of the Schur-backend starvation summary log (see
// Estimator::schurReportStarvation). Chosen to match the existing
// stage_timing_interval default of 100 frames.
constexpr int K_SCHUR_STARVATION_REPORT_FRAMES = 100;
}  // namespace

Estimator::Estimator() : f_manager_{Rs_}
{
  RCLCPP_INFO(vins::logger(), "init begins");
  initThreadFlag_ = false;
  clearState();
}

Estimator::~Estimator()
{
  stopping_ = true;
  measurements_ready_.notify_all();
  if (processThread_.joinable())
    processThread_.join();
  clearState();
}


void Estimator::clearState()
{
  std::lock_guard<std::mutex> process_lock(mProcess_);
  std::lock_guard<std::mutex> propagate_lock(mPropagate_);
  latest_state_valid_ = false;
  latest_schur_active_ = false;
  latest_schur_state_ = vins::backend::EkfState();
  latest_imu_dt_ = 0.0;
  latest_input_time_ = -1.0;
  schur_last_imu_dt_ = 0.0;
  schur_last_imu_time_ = -1.0;
  {
    std::lock_guard<std::mutex> lock(mBuf_);
    while (!accBuf_.empty())
      accBuf_.pop();
    while (!gyrBuf_.empty())
      gyrBuf_.pop();
    while (!featureBuf_.empty())
      featureBuf_.pop();
  }

  prevTime_ = -1;
  curTime_ = 0;
  openExEstimation_ = 0;
  initP_ = Eigen::Vector3d(0, 0, 0);
  initR_ = Eigen::Matrix3d::Identity();
  inputImageCnt_ = 0;
  initFirstPoseFlag_ = false;

  for (int i = 0; i < WINDOW_SIZE + 1; i++) {
    Rs_[i].setIdentity();
    Ps_[i].setZero();
    Vs_[i].setZero();
    Bas_[i].setZero();
    Bgs_[i].setZero();
    dt_buf_[i].clear();
    linear_acceleration_buf_[i].clear();
    angular_velocity_buf_[i].clear();

    if (pre_integrations_[i] != nullptr) {
      delete pre_integrations_[i];
    }
    pre_integrations_[i] = nullptr;
  }

  for (int i = 0; i < NUM_OF_CAM; i++) {
    tic_[i] = Vector3d::Zero();
    ric_[i] = Matrix3d::Identity();
  }

  first_imu_ = false, sum_of_back_ = 0;
  sum_of_front_ = 0;
  frame_count_ = 0;
  solver_flag_ = INITIAL;
  initial_timestamp_ = 0;
  all_image_frame_.clear();

  if (tmp_pre_integration_ != nullptr)
    delete tmp_pre_integration_;
  if (last_marginalization_info_ != nullptr)
    delete last_marginalization_info_;

  tmp_pre_integration_ = nullptr;
  last_marginalization_info_ = nullptr;
  last_marginalization_parameter_blocks_.clear();

  f_manager_.clearState();

  failure_occur_ = 0;
  restart_requested_ = false;
  suppress_state_publication_ = false;

  schur_ekf_active_ = false;
  schur_frame_index_ = 0;
  schur_oldest_frame_ = 0;
  schur_update_count_ = 0;
  schur_stage_total_ms_ = 0.0;
  schur_features_total_ = 0.0;
  schur_rejected_total_ = 0.0;
  schur_soft_restart_time_ = -1.0;
  schur_starvation_monitor_.reset(0.0);
  schur_landing_static_.reset();
  schur_landing_static_updates_ = 0;
  schur_landing_static_failures_ = 0;
  schur_history_.clear();
  schur_cameras_.clear();
  schur_margin_history_.clear();
  schur_visual_snapshot_ = vins::backend::SchurVisualizationSnapshot_s();
  schur_visualization_stamp_ = builtin_interfaces::msg::Time{};
  current_feature_stamp_.reset();
}

bool Estimator::restartAfterSolverFailure()
{
  if (!restart_requested_)
    return false;

  restart_requested_ = false;
  RCLCPP_WARN(
    vins::logger(), "Resetting estimator state after solver failure; current-frame observations and IMU "
                    "preintegration inputs were already consumed, so the next frame starts a clean "
                    "initialization instead of extending a stale image-frame history");
  clearState();
  setParameter();
  return true;
}

bool Estimator::runOptimization() { return optimization(); }

void Estimator::maintainInitializationHistoryAfterFailure()
{
  slideWindow();

  // MARGIN_SECOND_NEW does not evict all_image_frame entries, and a
  // MARGIN_OLD slide can leave the discarded oldest frame behind. Keep only
  // headers that belong to the active window so the next initialization pass
  // cannot index pose/velocity arrays using stale image frames.
  trimInitializationHistoryToActiveWindow();

  const size_t max_history = static_cast<size_t>(frame_count_ + 1);
  if (all_image_frame_.size() > max_history) {
    RCLCPP_WARN(
      vins::logger(),
      "Initialization image history still has %zu frames after trimming (limit %zu); "
      "scheduling a controlled estimator reset",
      all_image_frame_.size(), max_history);
    scheduleControlledRestart();
  }
}

void Estimator::trimInitializationHistoryToActiveWindow()
{
  std::set<double> active_headers;
  for (int i = 0; i <= frame_count_; ++i)
    active_headers.insert(Headers_[i]);

  for (auto frame_it = all_image_frame_.begin(); frame_it != all_image_frame_.end();) {
    auto next_it = std::next(frame_it);
    if (active_headers.find(frame_it->first) == active_headers.end()) {
      delete frame_it->second.pre_integration_;
      frame_it->second.pre_integration_ = nullptr;
      all_image_frame_.erase(frame_it);
    }
    frame_it = next_it;
  }
}

void Estimator::setParameter()
{
  mProcess_.lock();
  for (int i = 0; i < NUM_OF_CAM; i++) {
    tic_[i] = TIC[i];
    ric_[i] = RIC[i];
    cout << " exitrinsic cam " << i << '\n' << ric_[i] << '\n' << tic_[i].transpose() << '\n';
  }
  f_manager_.setRic(ric_);
  ProjectionTwoFrameOneCamFactor::sqrt_info_ = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
  ProjectionTwoFrameTwoCamFactor::sqrt_info_ = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
  ProjectionOneFrameTwoCamFactor::sqrt_info_ = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
  td_ = TD;
  g_ = G;
  cout << "set g " << g_.transpose() << '\n';
  featureTracker_.readIntrinsicParameter(CAM_NAMES);

  std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';
  if (MULTIPLE_THREAD && !initThreadFlag_) {
    initThreadFlag_ = true;
    processThread_ = std::thread(&Estimator::processMeasurements, this);
  }
  mProcess_.unlock();
}

void Estimator::changeSensorType(int _use_imu, int _use_stereo)
{
  bool restart = false;
  mProcess_.lock();
  if (!_use_imu && !_use_stereo)
    printf("at least use two sensors! \n");
  else {
    if (USE_IMU != _use_imu) {
      USE_IMU = _use_imu;
      if (USE_IMU) {
        // reuse imu; restart system
        restart = true;
      } else {
        if (last_marginalization_info_ != nullptr)
          delete last_marginalization_info_;

        tmp_pre_integration_ = nullptr;
        last_marginalization_info_ = nullptr;
        last_marginalization_parameter_blocks_.clear();
      }
    }

    STEREO = _use_stereo;
    printf("use imu %d use stereo %d\n", USE_IMU, STEREO);
  }
  mProcess_.unlock();
  if (restart) {
    clearState();
    setParameter();
  }
}

void Estimator::inputImage(
  double _t, const cv::Mat & _img, const cv::Mat & _img1,
  const std::optional<builtin_interfaces::msg::Time> & _stamp)
{
  inputImageCnt_++;
  // Tracking remains full-rate; only backend observation packaging is gated.
  const bool submit = !MULTIPLE_THREAD || backend_gate_.accept(_t);
  const bool draw = SHOW_TRACK && trackImageRequested();
  auto featureFrame = featureTracker_.trackImage(_t, _img, _img1, submit, draw);
  if (draw)
    pubTrackImage(featureTracker_.getTrackImage(), _t);
  if (!submit)
    return;
  {
    std::lock_guard<std::mutex> lock(mBuf_);
    if (featureBuf_.size() >= static_cast<size_t>(MAX_FEATURE_QUEUE)) {
      featureBuf_.pop();
      if (++dropped_features_ == 1 || dropped_features_ % 100 == 0)
        RCLCPP_WARN(vins::logger(), "Backend overloaded: dropped %zu queued feature frames", dropped_features_);
    }
    FeatureFrameInput_s input;
    input.time_ = _t;
    input.stamp_ = _stamp;
    input.features_ = std::move(featureFrame);
    featureBuf_.push(std::move(input));
  }
  measurements_ready_.notify_one();
  if (!MULTIPLE_THREAD)
    processMeasurements();
}

void Estimator::inputIMU(double _t, const Vector3d & _linearAcceleration, const Vector3d & _angularVelocity)
{
  if (!std::isfinite(_t) || !_linearAcceleration.allFinite() || !_angularVelocity.allFinite())
    return;
  // Serialize enqueue + prediction with visual re-anchoring. Otherwise a
  // sample can be replayed by updateLatestStates and then integrated twice
  // by the callback that originally enqueued it.
  std::lock_guard<std::mutex> propagate_lock(mPropagate_);
  if (_t <= latest_input_time_)
    return;
  // Measurement noise uses the sensor period, independent of the shorter
  // integration step after an image-time re-anchor or IMU tail replay.
  latest_imu_dt_ = latest_input_time_ >= 0.0 ? _t - latest_input_time_ : 0.0;
  latest_input_time_ = _t;
  {
    std::lock_guard<std::mutex> buffer_lock(mBuf_);
    accBuf_.emplace(_t, _linearAcceleration);
    gyrBuf_.emplace(_t, _angularVelocity);
  }
  measurements_ready_.notify_one();

  if (latest_state_valid_ && fastPredictIMU(_t, _linearAcceleration, _angularVelocity)) {
    if (latest_schur_active_)
      pubLatestOdometry(latest_schur_state_, _angularVelocity, latest_imu_dt_);
    else
      pubLatestOdometry(latest_P_, latest_Q_, latest_V_, latest_time_);
  }
}

void Estimator::inputFeature(
  double _t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> & _featureFrame)
{
  inputFeature(_t, _featureFrame, vins::stamp(_t));
}

void Estimator::inputFeature(
  double _t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> & _featureFrame,
  const builtin_interfaces::msg::Time & _stamp)
{
  {
    std::lock_guard<std::mutex> lock(mBuf_);
    if (featureBuf_.size() >= static_cast<size_t>(MAX_FEATURE_QUEUE))
      featureBuf_.pop();
    FeatureFrameInput_s input;
    input.time_ = _t;
    input.stamp_ = _stamp;
    input.features_ = _featureFrame;
    featureBuf_.push(std::move(input));
  }
  measurements_ready_.notify_one();
  if (!MULTIPLE_THREAD)
    processMeasurements();
}


bool Estimator::getIMUInterval(
  double _t0, double _t1, vector<pair<double, Eigen::Vector3d>> & _accVector,
  vector<pair<double, Eigen::Vector3d>> & _gyrVector)
{
  if (accBuf_.empty()) {
    printf("not receive imu\n");
    return false;
  }
  //printf("get imu from %f %f\n", t0, t1);
  //printf("imu fornt time %f   imu end time %f\n", accBuf.front().first, accBuf.back().first);
  if (_t1 <= accBuf_.back().first) {
    while (accBuf_.front().first <= _t0) {
      accBuf_.pop();
      gyrBuf_.pop();
    }
    while (accBuf_.front().first < _t1) {
      _accVector.push_back(accBuf_.front());
      accBuf_.pop();
      _gyrVector.push_back(gyrBuf_.front());
      gyrBuf_.pop();
    }
    _accVector.push_back(accBuf_.front());
    _gyrVector.push_back(gyrBuf_.front());
  } else {
    printf("wait for imu\n");
    return false;
  }
  return true;
}

bool Estimator::imuAvailable(double _t)
{
  std::lock_guard<std::mutex> lock(mBuf_);
  return !accBuf_.empty() && _t <= accBuf_.back().first;
}

void Estimator::processMeasurements()
{
  while (!stopping_ && rclcpp::ok()) {
    FeatureFrameInput_s feature;
    vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;
    {
      std::unique_lock<std::mutex> lock(mBuf_);
      auto ready = [this]
      {
        return !featureBuf_.empty() &&
               (!USE_IMU || (!accBuf_.empty() && featureBuf_.front().time_ + td_ <= accBuf_.back().first));
      };
      if (MULTIPLE_THREAD)
        measurements_ready_.wait(lock, [&] { return stopping_ || !rclcpp::ok() || ready(); });
      else if (!ready())
        return;
      if (stopping_ || !rclcpp::ok())
        return;
      feature = std::move(featureBuf_.front());
      featureBuf_.pop();
      curTime_ = feature.time_ + td_;
      if (USE_IMU)
        getIMUInterval(prevTime_, curTime_, accVector, gyrVector);
    }
    TicToc process_time;
    if (USE_IMU) {
      if (!initFirstPoseFlag_)
        initFirstIMUPose(accVector);
      for (size_t i = 0; i < accVector.size(); ++i) {
        double dt;
        if (i == 0)
          dt = accVector[i].first - prevTime_;
        else if (i == accVector.size() - 1)
          dt = curTime_ - accVector[i - 1].first;
        else
          dt = accVector[i].first - accVector[i - 1].first;
        processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);
      }
    }
    bool restart_after_failure = false;
    bool suppress_publication = false;
    {
      std::lock_guard<std::mutex> lock(mProcess_);
      processImage(feature.features_, feature.time_, feature.stamp_);
      restart_after_failure = restart_requested_;
      suppress_publication = suppress_state_publication_;
      if (!restart_after_failure) {
        // Frame inputs were consumed even when the solve failed, so the
        // IMU timeline must still advance on a retryable failure.
        prevTime_ = curTime_;
      }
      if (!restart_after_failure && !suppress_publication) {
        printStatistics(*this, process_time.toc());
        std_msgs::msg::Header header;
        header.frame_id = "world";
        header.stamp = vins::stamp(feature.time_);
        if (!rclcpp::ok())
          return;
        pubOdometry(*this, header);
        pubKeyPoses(*this, header);
        pubCameraPose(*this, header);
        pubPointCloud(*this, header);
        pubKeyframe(*this);
        pubTF(*this, header);
      }
    }
    if (restart_after_failure) {
      // clearState()/setParameter() lock mProcess themselves, so the
      // reset must happen after releasing the processImage lock.
      restartAfterSolverFailure();
      if (!MULTIPLE_THREAD)
        break;
      continue;
    }
    if (!MULTIPLE_THREAD)
      break;
  }
}


void Estimator::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> & _accVector)
{
  printf("init first imu pose\n");
  initFirstPoseFlag_ = true;
  //return;
  Eigen::Vector3d averAcc(0, 0, 0);
  int n = (int)_accVector.size();
  for (auto & i : _accVector) {
    averAcc = averAcc + i.second;
  }
  averAcc = averAcc / n;
  printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
  Matrix3d R0 = Utility::g2R(averAcc);
  double yaw = Utility::r2ypr(R0).x();
  R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
  Rs_[0] = R0;
  cout << "init R0 " << '\n' << Rs_[0] << '\n';
  //Vs[0] = Vector3d(5, 0, 0);
}

void Estimator::initFirstPose(Eigen::Vector3d _p, Eigen::Matrix3d _r)
{
  Ps_[0] = _p;
  Rs_[0] = _r;
  initP_ = _p;
  initR_ = _r;
}


void Estimator::processIMU(
  double _t, double _dt, const Vector3d & _linear_acceleration, const Vector3d & _angular_velocity)
{
  if (!first_imu_) {
    first_imu_ = true;
    acc_0_ = _linear_acceleration;
    gyr_0_ = _angular_velocity;
  }

  if (!pre_integrations_[frame_count_]) {
    pre_integrations_[frame_count_] =
      new IntegrationBase{acc_0_, gyr_0_, Bas_[frame_count_], Bgs_[frame_count_]};
  }
  if (frame_count_ != 0) {
    pre_integrations_[frame_count_]->pushBack(_dt, _linear_acceleration, _angular_velocity);
    //if(solver_flag != NON_LINEAR)
    tmp_pre_integration_->pushBack(_dt, _linear_acceleration, _angular_velocity);

    dt_buf_[frame_count_].push_back(_dt);
    linear_acceleration_buf_[frame_count_].push_back(_linear_acceleration);
    angular_velocity_buf_[frame_count_].push_back(_angular_velocity);

    int j = frame_count_;
    Vector3d un_acc_0 = Rs_[j] * (acc_0_ - Bas_[j]) - g_;
    Vector3d un_gyr = 0.5 * (gyr_0_ + _angular_velocity) - Bgs_[j];
    Rs_[j] *= Utility::deltaQ(un_gyr * _dt).toRotationMatrix();
    Vector3d un_acc_1 = Rs_[j] * (_linear_acceleration - Bas_[j]) - g_;
    Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    Ps_[j] += _dt * Vs_[j] + 0.5 * _dt * _dt * un_acc;
    Vs_[j] += _dt * un_acc;
  }
  acc_0_ = _linear_acceleration;
  gyr_0_ = _angular_velocity;

  // The gyro rate variance uses the sensor sampling interval, not the
  // possibly fractional interval ending exactly at an image timestamp.
  if (schur_last_imu_time_ >= 0.0 && _t > schur_last_imu_time_)
    schur_last_imu_dt_ = _t - schur_last_imu_time_;
  schur_last_imu_time_ = _t;
  if (schur_ekf_active_) {
    schur_landing_static_.observeImu(_t, _linear_acceleration, _angular_velocity);
    schurPropagate(_dt, _linear_acceleration, _angular_velocity);
  }
}

void Estimator::processImage(
  const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> & _image, const double _header,
  const std::optional<builtin_interfaces::msg::Time> & _stamp)
{
  RCLCPP_DEBUG(vins::logger(), "new image coming ------------------------------------------");
  RCLCPP_DEBUG(vins::logger(), "Adding feature points %lu", _image.size());
  suppress_state_publication_ = false;
  current_feature_stamp_ = _stamp;
  const auto log_initialization_failure = [this](const char * _stage)
  {
    suppress_state_publication_ = true;
    RCLCPP_WARN(
      vins::logger(),
      "Solver failed during %s; Ceres parameter blocks were rolled back, and "
      "double2vector, outlier rejection and state publication were skipped. The "
      "marginalization prior was not replaced. The initialization window is slid to keep "
      "the image-frame history bounded before the next retry; current-frame "
      "feature/image/IMU inputs were already advanced and are not rollback-capable",
      _stage);
  };
  const auto log_odometry_failure = [this](const char * _stage)
  {
    suppress_state_publication_ = true;
    RCLCPP_WARN(
      vins::logger(),
      "Solver failed during %s; Ceres parameter blocks were rolled back, and "
      "double2vector, outlier rejection, state publication and window sliding were skipped. "
      "The marginalization prior was not replaced. Current-frame feature/image/IMU inputs "
      "were already advanced and are not rollback-capable, so a controlled estimator reset "
      "is scheduled before the next frame",
      _stage);
    scheduleControlledRestart();
  };
  if (f_manager_.addFeatureCheckParallax(frame_count_, _image, td_)) {
    marginalization_flag_ = MARGIN_OLD;
    //printf("keyframe\n");
  } else {
    marginalization_flag_ = MARGIN_SECOND_NEW;
    //printf("non-keyframe\n");
  }

  RCLCPP_DEBUG(vins::logger(), "%s", marginalization_flag_ ? "Non-keyframe" : "Keyframe");
  RCLCPP_DEBUG(vins::logger(), "Solving %d", frame_count_);
  RCLCPP_DEBUG(vins::logger(), "number of feature: %d", f_manager_.getFeatureCount());
  Headers_[frame_count_] = _header;

  // These frame-level inputs are consumed before the solver runs. They are
  // intentionally not part of the solver rollback: a failed solve may leave
  // the current feature/image/IMU observations advanced, but it must not
  // replace the marginalization prior. INITIAL retries still apply the
  // window-maintenance slide below so all_image_frame stays bounded;
  // NON_LINEAR failures skip sliding and request a controlled reset.
  ImageFrame imageframe(_image, _header);
  imageframe.pre_integration_ = tmp_pre_integration_;
  all_image_frame_.insert(make_pair(_header, imageframe));
  tmp_pre_integration_ = new IntegrationBase{acc_0_, gyr_0_, Bas_[frame_count_], Bgs_[frame_count_]};

  if (ESTIMATE_EXTRINSIC == 2) {
    RCLCPP_INFO(vins::logger(), "calibrating extrinsic param, rotation movement is needed");
    if (frame_count_ != 0) {
      vector<pair<Vector3d, Vector3d>> corres = f_manager_.getCorresponding(frame_count_ - 1, frame_count_);
      Matrix3d calib_ric;
      if (initial_ex_rotation_.calibrationExRotation(
            corres, pre_integrations_[frame_count_]->delta_q_, calib_ric)) {
        RCLCPP_WARN(vins::logger(), "initial extrinsic rotation calib success");
        RCLCPP_WARN_STREAM(vins::logger(), "initial extrinsic rotation: " << '\n' << calib_ric);
        ric_[0] = calib_ric;
        RIC[0] = calib_ric;
        ESTIMATE_EXTRINSIC = 1;
      }
    }
  }

  if (solver_flag_ == INITIAL) {
    // monocular + IMU initilization
    if (!STEREO && USE_IMU) {
      if (frame_count_ == WINDOW_SIZE) {
        bool result = false;
        if (ESTIMATE_EXTRINSIC != 2 && (_header - initial_timestamp_) > 0.1) {
          result = initialStructure();
          initial_timestamp_ = _header;
        }
        if (result) {
          if (!runOptimization()) {
            log_initialization_failure("monocular IMU initialization");
            maintainInitializationHistoryAfterFailure();
            return;
          }
          solver_flag_ = NON_LINEAR;
          activateSchurBackend(_image, _header);
          updateLatestStates();
          slideWindow();
          RCLCPP_INFO(vins::logger(), "Initialization finish!");
        } else if (restart_requested_) {
          // initialStructure() already found an inconsistent history.
          // Do not touch the window again; processMeasurements() will
          // perform the controlled reset after releasing mProcess.
          suppress_state_publication_ = true;
          RCLCPP_WARN(
            vins::logger(), "Initialization requested a controlled reset; skipping window sliding "
                            "for the current frame");
          return;
        } else
          slideWindow();
      }
    }

    // stereo + IMU initilization
    if (STEREO && USE_IMU) {
      f_manager_.initFramePoseByPnP(frame_count_, Ps_, Rs_, tic_, ric_);
      f_manager_.triangulate(frame_count_, Ps_, Rs_, tic_, ric_);
      if (frame_count_ == WINDOW_SIZE) {
        if (all_image_frame_.size() != static_cast<size_t>(frame_count_ + 1)) {
          suppress_state_publication_ = true;
          RCLCPP_WARN(
            vins::logger(),
            "Stereo IMU initialization image history has %zu frames for an active "
            "window of %d; scheduling a controlled estimator reset",
            all_image_frame_.size(), frame_count_ + 1);
          scheduleControlledRestart();
          return;
        }
        map<double, ImageFrame>::iterator frame_it;
        int i = 0;
        for (frame_it = all_image_frame_.begin(); frame_it != all_image_frame_.end(); frame_it++) {
          if (i > frame_count_ || (i > 0 && frame_it->second.pre_integration_ == nullptr)) {
            suppress_state_publication_ = true;
            RCLCPP_WARN(
              vins::logger(),
              "Stereo IMU initialization has an invalid image frame at window "
              "index %d; scheduling a controlled estimator reset",
              i);
            scheduleControlledRestart();
            return;
          }
          frame_it->second.R_ = Rs_[i];
          frame_it->second.T_ = Ps_[i];
          i++;
        }
        solveGyroscopeBias(all_image_frame_, Bgs_);
        for (int i = 0; i <= WINDOW_SIZE; i++) {
          pre_integrations_[i]->repropagate(Vector3d::Zero(), Bgs_[i]);
        }
        if (!runOptimization()) {
          log_initialization_failure("stereo IMU initialization");
          maintainInitializationHistoryAfterFailure();
          return;
        }
        solver_flag_ = NON_LINEAR;
        activateSchurBackend(_image, _header);
        updateLatestStates();
        slideWindow();
        RCLCPP_INFO(vins::logger(), "Initialization finish!");
      }
    }

    // stereo only initilization
    if (STEREO && !USE_IMU) {
      f_manager_.initFramePoseByPnP(frame_count_, Ps_, Rs_, tic_, ric_);
      f_manager_.triangulate(frame_count_, Ps_, Rs_, tic_, ric_);
      const bool initial_solve_ok = runOptimization();
      if (!initial_solve_ok)
        log_initialization_failure("stereo-only initialization");

      if (frame_count_ == WINDOW_SIZE) {
        if (!initial_solve_ok) {
          maintainInitializationHistoryAfterFailure();
          return;
        }
        if (!runOptimization()) {
          log_initialization_failure("stereo-only initialization refinement");
          maintainInitializationHistoryAfterFailure();
          return;
        }
        solver_flag_ = NON_LINEAR;
        activateSchurBackend(_image, _header);
        updateLatestStates();
        slideWindow();
        RCLCPP_INFO(vins::logger(), "Initialization finish!");
      }
    }

    if (frame_count_ < WINDOW_SIZE) {
      frame_count_++;
      int prev_frame = frame_count_ - 1;
      Ps_[frame_count_] = Ps_[prev_frame];
      Vs_[frame_count_] = Vs_[prev_frame];
      Rs_[frame_count_] = Rs_[prev_frame];
      Bas_[frame_count_] = Bas_[prev_frame];
      Bgs_[frame_count_] = Bgs_[prev_frame];
    }

  } else {
    TicToc t_solve;
    if (!USE_IMU)
      f_manager_.initFramePoseByPnP(frame_count_, Ps_, Rs_, tic_, ric_);
    f_manager_.triangulate(frame_count_, Ps_, Rs_, tic_, ric_);
    bool solve_ok = false;
    if (schur_ekf_active_) {
      // ARM-oriented path: propagate + Schur-complement EKF update, no
      // Ceres solve and no marginalization factor.
      solve_ok = schurProcessFrame(_image, _header);
      if (!solve_ok) {
        RCLCPP_WARN(
          vins::logger(), "Schur EKF could not process this frame; re-seeding the filter from a "
                          "Ceres solve to keep the estimate valid");
        solve_ok = runOptimization();
        if (solve_ok)
          initSchurBackend();
      }
    } else {
      solve_ok = runOptimization();
    }
    if (!solve_ok) {
      log_odometry_failure("nonlinear odometry");
      return;
    }
    set<int> removeIndex;
    outliersRejection(removeIndex);
    f_manager_.removeOutlier(removeIndex);
    if (!MULTIPLE_THREAD) {
      featureTracker_.removeOutliers(removeIndex);
      predictPtsInNextFrame();
    }

    RCLCPP_DEBUG(vins::logger(), "solver costs: %fms", t_solve.toc());

    if (failureDetection()) {
      suppress_state_publication_ = true;
      RCLCPP_WARN(vins::logger(), "Failure detection triggered; scheduling a controlled estimator reset");
      failure_occur_ = 1;
      scheduleControlledRestart();
      return;
    }

    slideWindow();
    f_manager_.removeFailures();
    // Optional Ceres warm-up: start the EKF from a converged window state
    // once the configured number of post-initialisation frames has passed.
    if (!schur_ekf_active_ && BACKEND_TYPE == "schur_ekf")
      activateSchurBackend(_image, _header);
    // prepare output of VINS
    key_poses_.clear();
    for (int i = 0; i <= WINDOW_SIZE; i++)
      key_poses_.push_back(Ps_[i]);

    last_R_ = Rs_[WINDOW_SIZE];
    last_P_ = Ps_[WINDOW_SIZE];
    last_R0_ = Rs_[0];
    last_P0_ = Ps_[0];
    updateLatestStates();
    if (std::getenv("VINS_SCHUR_DEBUG") != nullptr) {
      static int ceres_dbg = 0;
      if (++ceres_dbg % 5 == 0)
        fprintf(
          stderr,
          "[ceres-state] frame=%d t=%.3f |p|=%.4f |v|=%.4f |ba|=%.5f |bg|=%.6f "
          "ba=[%.4f %.4f %.4f] bg=[%.5f %.5f %.5f]\n",
          ceres_dbg, _header + td_, Ps_[WINDOW_SIZE].norm(), Vs_[WINDOW_SIZE].norm(),
          Bas_[WINDOW_SIZE].norm(), Bgs_[WINDOW_SIZE].norm(), Bas_[WINDOW_SIZE].x(), Bas_[WINDOW_SIZE].y(),
          Bas_[WINDOW_SIZE].z(), Bgs_[WINDOW_SIZE].x(), Bgs_[WINDOW_SIZE].y(), Bgs_[WINDOW_SIZE].z());
    }
  }
}

bool Estimator::initialStructure()
{
  TicToc t_sfm;
  if (frame_count_ < 0 || frame_count_ > WINDOW_SIZE || all_image_frame_.empty()) {
    RCLCPP_WARN(
      vins::logger(),
      "Initialization has an invalid frame history (frame_count=%d, image_frames=%zu); "
      "scheduling a controlled estimator reset",
      frame_count_, all_image_frame_.size());
    scheduleControlledRestart();
    return false;
  }
  //check imu observibility
  {
    map<double, ImageFrame>::iterator frame_it;
    Vector3d sum_g;
    for (frame_it = all_image_frame_.begin(), frame_it++; frame_it != all_image_frame_.end(); frame_it++) {
      double dt = frame_it->second.pre_integration_->sum_dt_;
      Vector3d tmp_g = frame_it->second.pre_integration_->delta_v_ / dt;
      sum_g += tmp_g;
    }
    Vector3d aver_g;
    aver_g = sum_g * 1.0 / ((int)all_image_frame_.size() - 1);
    double var = 0;
    for (frame_it = all_image_frame_.begin(), frame_it++; frame_it != all_image_frame_.end(); frame_it++) {
      double dt = frame_it->second.pre_integration_->sum_dt_;
      Vector3d tmp_g = frame_it->second.pre_integration_->delta_v_ / dt;
      var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
      //cout << "frame g " << tmp_g.transpose() << endl;
    }
    var = sqrt(var / ((int)all_image_frame_.size() - 1));
    //RCLCPP_WARN(vins::logger(), "IMU variation %f!", var);
    if (var < 0.25) {
      RCLCPP_INFO(vins::logger(), "IMU excitation not enouth!");
      //return false;
    }
  }
  // global sfm
  Quaterniond Q[frame_count_ + 1];
  Vector3d T[frame_count_ + 1];
  map<int, Vector3d> sfm_tracked_points;
  vector<SfmFeature_s> sfm_f;
  for (auto & it_per_id : f_manager_.feature_) {
    int imu_j = it_per_id.start_frame_ - 1;
    SfmFeature_s tmp_feature;
    tmp_feature.state_ = false;
    tmp_feature.id_ = it_per_id.feature_id_;
    for (auto & it_per_frame : it_per_id.feature_per_frame_) {
      imu_j++;
      Vector3d pts_j = it_per_frame.point_;
      tmp_feature.observation_.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
    }
    sfm_f.push_back(tmp_feature);
  }
  Matrix3d relative_R;
  Vector3d relative_T;
  int l;
  if (!relativePose(relative_R, relative_T, l)) {
    RCLCPP_INFO(vins::logger(), "Not enough features or parallax; Move device around");
    return false;
  }
  GlobalSFM sfm;
  if (!sfm.construct(frame_count_ + 1, Q, T, l, relative_R, relative_T, sfm_f, sfm_tracked_points)) {
    RCLCPP_DEBUG(vins::logger(), "global SFM failed!");
    marginalization_flag_ = MARGIN_OLD;
    return false;
  }

  //solve pnp for all frame
  map<double, ImageFrame>::iterator frame_it;
  map<int, Vector3d>::iterator it;
  frame_it = all_image_frame_.begin();
  for (int i = 0; frame_it != all_image_frame_.end(); frame_it++) {
    if (i > frame_count_) {
      RCLCPP_WARN(
        vins::logger(),
        "Initialization image history has frames beyond the active window "
        "(index=%d, frame_count=%d); scheduling a controlled estimator reset",
        i, frame_count_);
      scheduleControlledRestart();
      return false;
    }
    // provide initial guess
    cv::Mat r, rvec, t, D, tmp_r;
    if ((frame_it->first) == Headers_[i]) {
      frame_it->second.is_key_frame_ = true;
      frame_it->second.R_ = Q[i].toRotationMatrix() * RIC[0].transpose();
      frame_it->second.T_ = T[i];
      i++;
      continue;
    }
    if ((frame_it->first) > Headers_[i]) {
      i++;
      if (i > frame_count_) {
        RCLCPP_WARN(
          vins::logger(), "Initialization frame timestamps do not match the active window; "
                          "scheduling a controlled estimator reset");
        scheduleControlledRestart();
        return false;
      }
    }
    Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
    Vector3d P_inital = -R_inital * T[i];
    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    frame_it->second.is_key_frame_ = false;
    vector<cv::Point3f> pts_3_vector;
    vector<cv::Point2f> pts_2_vector;
    for (auto & id_pts : frame_it->second.points_) {
      int feature_id = id_pts.first;
      for (auto & i_p : id_pts.second) {
        it = sfm_tracked_points.find(feature_id);
        if (it != sfm_tracked_points.end()) {
          Vector3d world_pts = it->second;
          cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
          pts_3_vector.push_back(pts_3);
          Vector2d img_pts = i_p.second.head<2>();
          cv::Point2f pts_2(img_pts(0), img_pts(1));
          pts_2_vector.push_back(pts_2);
        }
      }
    }
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    if (pts_3_vector.size() < 6) {
      cout << "pts_3_vector size " << pts_3_vector.size() << '\n';
      RCLCPP_DEBUG(vins::logger(), "Not enough points for solve pnp !");
      return false;
    }
    if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1)) {
      RCLCPP_DEBUG(vins::logger(), "solve pnp fail!");
      return false;
    }
    cv::Rodrigues(rvec, r);
    MatrixXd R_pnp, tmp_R_pnp;
    cv::cv2eigen(r, tmp_R_pnp);
    R_pnp = tmp_R_pnp.transpose();
    MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);
    T_pnp = R_pnp * (-T_pnp);
    frame_it->second.R_ = R_pnp * RIC[0].transpose();
    frame_it->second.T_ = T_pnp;
  }
  if (visualInitialAlign())
    return true;
  else {
    RCLCPP_INFO(vins::logger(), "misalign visual structure with IMU");
    return false;
  }
}

bool Estimator::visualInitialAlign()
{
  TicToc t_g;
  VectorXd x;
  //solve scale
  bool result = VisualIMUAlignment(all_image_frame_, Bgs_, g_, x);
  if (!result) {
    RCLCPP_DEBUG(vins::logger(), "solve g failed!");
    return false;
  }

  // Old image frames can outlive the active window when initialization was
  // retried. Clear stale keyframe flags first so the velocity slot below can
  // never be indexed past the active window.
  for (auto & frame_entry : all_image_frame_)
    frame_entry.second.is_key_frame_ = false;

  // change state
  for (int i = 0; i <= frame_count_; i++) {
    auto frame_entry = all_image_frame_.find(Headers_[i]);
    if (frame_entry == all_image_frame_.end()) {
      RCLCPP_WARN(
        vins::logger(),
        "Initialization window header %d is missing from all_image_frame; "
        "scheduling a controlled estimator reset",
        i);
      scheduleControlledRestart();
      return false;
    }
    Matrix3d Ri = frame_entry->second.R_;
    Vector3d Pi = frame_entry->second.T_;
    Ps_[i] = Pi;
    Rs_[i] = Ri;
    frame_entry->second.is_key_frame_ = true;
  }

  double s = (x.tail<1>())(0);
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    pre_integrations_[i]->repropagate(Vector3d::Zero(), Bgs_[i]);
  }
  for (int i = frame_count_; i >= 0; i--)
    Ps_[i] = s * Ps_[i] - Rs_[i] * TIC[0] - (s * Ps_[0] - Rs_[0] * TIC[0]);
  int kv = -1;
  map<double, ImageFrame>::iterator frame_i;
  for (frame_i = all_image_frame_.begin(); frame_i != all_image_frame_.end(); frame_i++) {
    if (frame_i->second.is_key_frame_) {
      kv++;
      if (kv > frame_count_ || kv > WINDOW_SIZE) {
        RCLCPP_WARN(
          vins::logger(),
          "Initialization velocity index %d exceeds the active window (frame_count=%d); "
          "scheduling a controlled estimator reset",
          kv, frame_count_);
        scheduleControlledRestart();
        return false;
      }
      Vs_[kv] = frame_i->second.R_ * x.segment<3>(kv * 3);
    }
  }
  if (kv != frame_count_) {
    RCLCPP_WARN(
      vins::logger(),
      "Initialization found %d keyframes for %d active frames; scheduling a controlled "
      "estimator reset",
      kv + 1, frame_count_ + 1);
    scheduleControlledRestart();
    return false;
  }

  Matrix3d R0 = Utility::g2R(g_);
  double yaw = Utility::r2ypr(R0 * Rs_[0]).x();
  R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
  g_ = R0 * g_;
  //Matrix3d rot_diff = R0 * Rs[0].transpose();
  Matrix3d rot_diff = R0;
  for (int i = 0; i <= frame_count_; i++) {
    Ps_[i] = rot_diff * Ps_[i];
    Rs_[i] = rot_diff * Rs_[i];
    Vs_[i] = rot_diff * Vs_[i];
  }
  RCLCPP_DEBUG_STREAM(vins::logger(), "g0     " << g_.transpose());
  RCLCPP_DEBUG_STREAM(vins::logger(), "my R0  " << Utility::r2ypr(Rs_[0]).transpose());

  f_manager_.clearDepth();
  f_manager_.triangulate(frame_count_, Ps_, Rs_, tic_, ric_);

  return true;
}

bool Estimator::relativePose(Matrix3d & _relative_R, Vector3d & _relative_T, int & _l)
{
  // find previous frame which contians enough correspondance and parallex with newest frame
  for (int i = 0; i < WINDOW_SIZE; i++) {
    vector<pair<Vector3d, Vector3d>> corres;
    corres = f_manager_.getCorresponding(i, WINDOW_SIZE);
    if (corres.size() > 20) {
      double sum_parallax = 0;
      double average_parallax;
      for (auto & corre : corres) {
        Vector2d pts_0(corre.first(0), corre.first(1));
        Vector2d pts_1(corre.second(0), corre.second(1));
        double parallax = (pts_0 - pts_1).norm();
        sum_parallax = sum_parallax + parallax;
      }
      average_parallax = 1.0 * sum_parallax / int(corres.size());
      if (average_parallax * 460 > 30 && m_estimator_.solveRelativeRT(corres, _relative_R, _relative_T)) {
        _l = i;
        RCLCPP_DEBUG(
          vins::logger(), "average_parallax %f choose l %d and newest frame to triangulate the whole structure",
          average_parallax * 460, _l);
        return true;
      }
    }
  }
  return false;
}

void Estimator::vector2double()
{
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    para_Pose_[i][0] = Ps_[i].x();
    para_Pose_[i][1] = Ps_[i].y();
    para_Pose_[i][2] = Ps_[i].z();
    Quaterniond q{Rs_[i]};
    para_Pose_[i][3] = q.x();
    para_Pose_[i][4] = q.y();
    para_Pose_[i][5] = q.z();
    para_Pose_[i][6] = q.w();

    if (USE_IMU) {
      para_SpeedBias_[i][0] = Vs_[i].x();
      para_SpeedBias_[i][1] = Vs_[i].y();
      para_SpeedBias_[i][2] = Vs_[i].z();

      para_SpeedBias_[i][3] = Bas_[i].x();
      para_SpeedBias_[i][4] = Bas_[i].y();
      para_SpeedBias_[i][5] = Bas_[i].z();

      para_SpeedBias_[i][6] = Bgs_[i].x();
      para_SpeedBias_[i][7] = Bgs_[i].y();
      para_SpeedBias_[i][8] = Bgs_[i].z();
    }
  }

  for (int i = 0; i < NUM_OF_CAM; i++) {
    para_Ex_Pose_[i][0] = tic_[i].x();
    para_Ex_Pose_[i][1] = tic_[i].y();
    para_Ex_Pose_[i][2] = tic_[i].z();
    Quaterniond q{ric_[i]};
    para_Ex_Pose_[i][3] = q.x();
    para_Ex_Pose_[i][4] = q.y();
    para_Ex_Pose_[i][5] = q.z();
    para_Ex_Pose_[i][6] = q.w();
  }


  VectorXd dep = f_manager_.getDepthVector();
  for (int i = 0; i < f_manager_.getFeatureCount(); i++)
    para_Feature_[i][0] = dep(i);

  para_Td_[0][0] = td_;
}

void Estimator::double2vector()
{
  Vector3d origin_R0 = Utility::r2ypr(Rs_[0]);
  Vector3d origin_P0 = Ps_[0];

  if (failure_occur_) {
    origin_R0 = Utility::r2ypr(last_R0_);
    origin_P0 = last_P0_;
    failure_occur_ = 0;
  }

  if (USE_IMU) {
    Vector3d origin_R00 = Utility::r2ypr(
      Quaterniond(para_Pose_[0][6], para_Pose_[0][3], para_Pose_[0][4], para_Pose_[0][5]).toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0) {
      RCLCPP_DEBUG(vins::logger(), "euler singular point!");
      rot_diff = Rs_[0] * Quaterniond(para_Pose_[0][6], para_Pose_[0][3], para_Pose_[0][4], para_Pose_[0][5])
                            .toRotationMatrix()
                            .transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++) {

      Rs_[i] = rot_diff * Quaterniond(para_Pose_[i][6], para_Pose_[i][3], para_Pose_[i][4], para_Pose_[i][5])
                            .normalized()
                            .toRotationMatrix();

      Ps_[i] = rot_diff * Vector3d(
                            para_Pose_[i][0] - para_Pose_[0][0], para_Pose_[i][1] - para_Pose_[0][1],
                            para_Pose_[i][2] - para_Pose_[0][2]) +
               origin_P0;


      Vs_[i] = rot_diff * Vector3d(para_SpeedBias_[i][0], para_SpeedBias_[i][1], para_SpeedBias_[i][2]);

      Bas_[i] = Vector3d(para_SpeedBias_[i][3], para_SpeedBias_[i][4], para_SpeedBias_[i][5]);

      Bgs_[i] = Vector3d(para_SpeedBias_[i][6], para_SpeedBias_[i][7], para_SpeedBias_[i][8]);
    }
  } else {
    for (int i = 0; i <= WINDOW_SIZE; i++) {
      Rs_[i] = Quaterniond(para_Pose_[i][6], para_Pose_[i][3], para_Pose_[i][4], para_Pose_[i][5])
                 .normalized()
                 .toRotationMatrix();

      Ps_[i] = Vector3d(para_Pose_[i][0], para_Pose_[i][1], para_Pose_[i][2]);
    }
  }

  if (USE_IMU) {
    for (int i = 0; i < NUM_OF_CAM; i++) {
      tic_[i] = Vector3d(para_Ex_Pose_[i][0], para_Ex_Pose_[i][1], para_Ex_Pose_[i][2]);
      ric_[i] = Quaterniond(para_Ex_Pose_[i][6], para_Ex_Pose_[i][3], para_Ex_Pose_[i][4], para_Ex_Pose_[i][5])
                  .normalized()
                  .toRotationMatrix();
    }
  }

  VectorXd dep = f_manager_.getDepthVector();
  for (int i = 0; i < f_manager_.getFeatureCount(); i++)
    dep(i) = para_Feature_[i][0];
  f_manager_.setDepth(dep);

  if (USE_IMU)
    td_ = para_Td_[0][0];
}

bool Estimator::failureDetection()
{
  return false;
  if (f_manager_.last_track_num_ < 2) {
    RCLCPP_INFO(vins::logger(), " little feature %d", f_manager_.last_track_num_);
    //return true;
  }
  if (Bas_[WINDOW_SIZE].norm() > 2.5) {
    RCLCPP_INFO(vins::logger(), " big IMU acc bias estimation %f", Bas_[WINDOW_SIZE].norm());
    return true;
  }
  if (Bgs_[WINDOW_SIZE].norm() > 1.0) {
    RCLCPP_INFO(vins::logger(), " big IMU gyr bias estimation %f", Bgs_[WINDOW_SIZE].norm());
    return true;
  }
  /*
    if (tic(0) > 1)
    {
        RCLCPP_INFO(vins::logger(), " big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
  Vector3d tmp_P = Ps_[WINDOW_SIZE];
  if ((tmp_P - last_P_).norm() > 5) {
    //RCLCPP_INFO(vins::logger(), " big translation");
    //return true;
  }
  if (abs(tmp_P.z() - last_P_.z()) > 1) {
    //RCLCPP_INFO(vins::logger(), " big z translation");
    //return true;
  }
  Matrix3d tmp_R = Rs_[WINDOW_SIZE];
  Matrix3d delta_R = tmp_R.transpose() * last_R_;
  Quaterniond delta_Q(delta_R);
  double delta_angle;
  delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
  if (delta_angle > 50) {
    RCLCPP_INFO(vins::logger(), " big delta_angle ");
    //return true;
  }
  return false;
}

// ================= Schur-complement EKF backend (ARM-oriented) =============
//
// Opt-in replacement for the Ceres sliding-window solve, selected with
// `backend_type: schur_ekf`.  The window is a 15-state IMU block plus 6 states
// per retained clone; feature points are nuisance parameters that never enter
// the covariance (Schur-eliminated per track).  See docs/BACKEND_SCHUR_EKF.md.

void Estimator::initSchurBackend()
{
  vins::backend::EkfOptions_s options;
  options.window_size_ = SCHUR_WINDOW_SIZE;
  options.gravity_ = g_.norm() > 1e-3 ? g_.norm() : G.norm();
  // Continuous-time spectral densities (unit^2/Hz).  ACC_N/GYR_N/ACC_W/GYR_W
  // follow VINS's per-sample convention (they enter IntegrationBase through a
  // V matrix scaled by 0.5*dt) and must not be fed to the EKF as densities:
  // doing so inflates the velocity/attitude process noise by ~1/dt and makes
  // every visual update over-correct the velocity.
  options.acc_noise_ = SCHUR_ACC_NOISE;
  options.gyr_noise_ = SCHUR_GYR_NOISE;
  options.acc_bias_noise_ = SCHUR_ACC_BIAS_NOISE;
  options.gyr_bias_noise_ = SCHUR_GYR_BIAS_NOISE;
  options.init_position_sigma_ = SCHUR_INIT_POS_SIGMA;
  options.init_velocity_sigma_ = SCHUR_INIT_VEL_SIGMA;
  options.init_rotation_sigma_ = SCHUR_INIT_ROT_SIGMA;
  options.init_acc_bias_sigma_ = SCHUR_INIT_BA_SIGMA;
  options.init_gyr_bias_sigma_ = SCHUR_INIT_BG_SIGMA;

  const Eigen::Quaterniond q_init(Rs_[frame_count_]);
  const Eigen::Vector4d q_vec(q_init.w(), q_init.x(), q_init.y(), q_init.z());
  // Sensor time (frame header + td) of the seeding frame. It is the baseline
  // of the time-based starvation criterion until the first applied update.
  const double seed_time = Headers_[frame_count_] + td_;
  schur_state_.reset(
    options, seed_time, q_vec, Ps_[frame_count_], Vs_[frame_count_], Bas_[frame_count_], Bgs_[frame_count_]);

  vins::backend::UpdateOptions_s update_options;
  update_options.max_features_ = SCHUR_MAX_FEATURES;
  update_options.min_observations_ = SCHUR_MIN_OBSERVATIONS;
  update_options.pixel_noise_ = SCHUR_PIXEL_NOISE;
  update_options.focal_length_ = FOCAL_LENGTH;
  update_options.chi2_multiplier_ = SCHUR_CHI2_MULTIPLIER;
  update_options.triangulation_iterations_ = static_cast<int>(std::lround(SCHUR_TRIANGULATION_ITERATIONS));
  update_options.min_features_ = SCHUR_MIN_FEATURES;
  update_options.min_update_observations_ = SCHUR_MIN_UPDATE_OBSERVATIONS;
  update_options.info_rank_rtol_ = SCHUR_INFO_RANK_RTOL;
  update_options.info_indefinite_tol_ = SCHUR_INFO_INDEFINITE_TOL;
  update_options.point_block_rtol_ = SCHUR_POINT_BLOCK_RTOL;
  update_options.condition_limit_ = SCHUR_CONDITION_LIMIT;
  schur_updater_ = vins::backend::SchurUpdater(update_options);

  schur_cameras_.clear();
  for (int i = 0; i < std::max(NUM_OF_CAM, 1); ++i) {
    vins::backend::CameraExtrinsics_s camera;
    if (i < NUM_OF_CAM) {
      const Eigen::Quaterniond q_cb(ric_[i]);
      camera.q_cb_ = Eigen::Vector4d(q_cb.w(), q_cb.x(), q_cb.y(), q_cb.z());
      camera.p_cb_ = tic_[i];
    }
    schur_cameras_.push_back(camera);
  }

  schur_history_.clear();
  schur_margin_history_.clear();
  schur_frame_index_ = 0;
  schur_oldest_frame_ = 0;
  schur_visual_snapshot_ = vins::backend::SchurVisualizationSnapshot_s();
  schur_visualization_stamp_ = builtin_interfaces::msg::Time{};
  schur_update_count_ = 0;
  schur_stage_total_ms_ = 0.0;
  schur_features_total_ = 0.0;
  schur_rejected_total_ = 0.0;
  schur_lost_total_ = 0.0;
  schur_marginalized_total_ = 0.0;
  schur_consecutive_failures_ = 0;
  schur_skipped_frames_ = 0;
  // A full (re-)seed ends any pending soft-restart episode.
  schur_soft_restart_time_ = -1.0;
  // Time-based starvation policy: configure the thresholds and re-seed the
  // monitor so the seed frame counts as the last applied update. The
  // diagnostic counters start fresh for the new filter instance; the restart
  // cooldown anchor deliberately survives reset() (see StarvationMonitor), so
  // a controlled restart cannot be followed by another one within
  // SCHUR_RESTART_COOLDOWN_SECONDS of sensor time.
  vins::backend::StarvationMonitor::Config_s starvation_config;
  starvation_config.max_no_update_seconds_ = SCHUR_MAX_NO_UPDATE_SECONDS;
  starvation_config.starvation_seconds_ = SCHUR_STARVATION_SECONDS;
  starvation_config.noise_scale_ = SCHUR_STARVATION_NOISE_SCALE;
  starvation_config.deep_starvation_seconds_ = SCHUR_STARVATION_DEEP_SECONDS;
  starvation_config.deep_noise_scale_ = SCHUR_STARVATION_DEEP_NOISE_SCALE;
  starvation_config.restart_cooldown_seconds_ = SCHUR_RESTART_COOLDOWN_SECONDS;
  schur_starvation_monitor_.configure(starvation_config);
  schur_starvation_monitor_.reset(seed_time);
  // Conservative landing ZUPT detector. The detector is reset together with
  // every filter instance; it cannot carry an impact or stillness window
  // across a restart or clock discontinuity.
  vins::backend::LandingStaticConfig_s landing_config;
  landing_config.enabled_ = SCHUR_LANDING_ZUPT_ENABLE;
  landing_config.min_still_seconds_ = SCHUR_LANDING_ZUPT_MIN_STILL_SECONDS;
  landing_config.flow_reference_seconds_ = SCHUR_LANDING_ZUPT_FLOW_REFERENCE_SECONDS;
  landing_config.max_flow_px_ = SCHUR_LANDING_ZUPT_MAX_FLOW_PX;
  landing_config.min_static_fraction_ = SCHUR_LANDING_ZUPT_MIN_STATIC_FRACTION;
  landing_config.min_features_ = SCHUR_LANDING_ZUPT_MIN_FEATURES;
  landing_config.min_quadrants_ = SCHUR_LANDING_ZUPT_MIN_QUADRANTS;
  landing_config.image_width_ = COL;
  landing_config.image_height_ = ROW;
  landing_config.impact_window_seconds_ = SCHUR_LANDING_ZUPT_IMPACT_WINDOW_SECONDS;
  landing_config.impact_accel_ = SCHUR_LANDING_ZUPT_IMPACT_ACCEL;
  landing_config.max_image_gap_seconds_ = SCHUR_LANDING_ZUPT_MAX_IMAGE_GAP_SECONDS;
  landing_config.imu_window_seconds_ = SCHUR_LANDING_ZUPT_IMU_WINDOW_SECONDS;
  landing_config.max_imu_sample_gap_seconds_ = SCHUR_LANDING_ZUPT_MAX_IMU_GAP_SECONDS;
  landing_config.min_imu_samples_ = SCHUR_LANDING_ZUPT_MIN_IMU_SAMPLES;
  landing_config.max_gyro_rad_s_ = SCHUR_LANDING_ZUPT_MAX_GYRO;
  landing_config.max_accel_dev_ = SCHUR_LANDING_ZUPT_MAX_ACCEL_DEV;
  landing_config.max_accel_std_ = SCHUR_LANDING_ZUPT_MAX_ACCEL_STD;
  landing_config.max_gyro_std_ = SCHUR_LANDING_ZUPT_MAX_GYRO_STD;
  landing_config.max_speed_ = SCHUR_LANDING_ZUPT_MAX_SPEED;
  landing_config.exit_flow_px_ = SCHUR_LANDING_ZUPT_EXIT_FLOW_PX;
  landing_config.exit_speed_ = SCHUR_LANDING_ZUPT_EXIT_SPEED;
  landing_config.exit_gyro_rad_s_ = SCHUR_LANDING_ZUPT_EXIT_GYRO;
  landing_config.exit_accel_dev_ = SCHUR_LANDING_ZUPT_EXIT_ACCEL_DEV;
  landing_config.gravity_ = options.gravity_;
  schur_landing_static_.configure(landing_config);
  schur_landing_static_.reset();
  schur_landing_static_updates_ = 0;
  schur_landing_static_failures_ = 0;
  // P7 hold policy: configure and drop any holding state from a previous
  // filter instance (cumulative diagnostics survive).
  vins::backend::StarvationHold::Config_s hold_config;
  hold_config.enabled_ = SCHUR_STARVATION_HOLD;
  hold_config.recovery_min_features_ = SCHUR_MIN_FEATURES;
  hold_config.recovery_frames_ = SCHUR_HOLD_RECOVERY_FRAMES;
  schur_hold_.configure(hold_config);
  schur_hold_.deactivate();
  schur_rebuild_frames_ = 0;
  schur_rebuild_pending_ = false;
  schur_ekf_active_ = true;

  RCLCPP_INFO(
    vins::logger(),
    "Schur EKF seeded at t=%.6f: p=[%.3f %.3f %.3f] v=[%.3f %.3f %.3f] "
    "|g|=%.4f window=%d",
    seed_time, Ps_[frame_count_].x(), Ps_[frame_count_].y(), Ps_[frame_count_].z(), Vs_[frame_count_].x(),
    Vs_[frame_count_].y(), Vs_[frame_count_].z(), options.gravity_, options.window_size_);
}

void Estimator::activateSchurBackend(
  const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> & _image, double _header)
{
  if (BACKEND_TYPE != "schur_ekf")
    return;
  if (!USE_IMU) {
    RCLCPP_WARN(vins::logger(), "backend_type=schur_ekf requires an IMU; keeping the Ceres backend");
    return;
  }
  if (schur_ekf_active_)
    return;
  if (inputImageCnt_ < SCHUR_WARMUP_FRAMES)
    return;

  initSchurBackend();
  if (!schurProcessFrame(_image, _header)) {
    schur_ekf_active_ = false;
    RCLCPP_WARN(
      vins::logger(), "Schur EKF could not consume the initialisation frame; keeping the "
                      "Ceres backend");
    return;
  }
  RCLCPP_INFO(
    vins::logger(), "Schur-complement EKF backend active (window=%d clones, max_features=%d)",
    SCHUR_WINDOW_SIZE, SCHUR_MAX_FEATURES);
}

void Estimator::schurPropagate(double _dt, const Eigen::Vector3d & _acc, const Eigen::Vector3d & _gyr)
{
  if (!schur_ekf_active_)
    return;
  // Guard against clock jumps and long IMU gaps: a single step that large
  // would be dominated by linearisation error.
  if (!(_dt > 0.0) || _dt > 1.0)
    return;
  // Visual-starvation process-noise inflation (P3/P4). The monitor returns the
  // multiplier of the current starvation stage: 1.0 when not starved,
  // SCHUR_STARVATION_NOISE_SCALE while starved, and
  // SCHUR_STARVATION_DEEP_NOISE_SCALE once deep starvation
  // (SCHUR_STARVATION_DEEP_SECONDS) is reached. The stage is evaluated once
  // per image frame in schurProcessFrame(), where the sensor-time gap since the
  // last applied update is known; between frames it is piecewise constant, so
  // every IMU step of a starved stretch uses the same scale. A successful
  // update clears the stage (scale back to 1.0) in the same frame.
  // P7 hold: the hold policy must let P grow naturally instead of being
  // re-inflated, so the starvation Q multiplier is bypassed while holding.
  const double noise_scale = schur_hold_.active() ? 1.0 : schur_starvation_monitor_.noiseScale();
  schur_state_.propagate(_acc, _gyr, _dt, noise_scale);
}

bool Estimator::schurProcessFrame(
  const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> & _image, double _header)
{
  if (!schur_ekf_active_)
    return false;

  TicToc t_update;
  // Sensor time of this frame (frame header + td). The time-based starvation
  // criterion compares it against the baseline held by schur_starvation_monitor_
  // (the time of the last *applied* update, or the seed time right after
  // initSchurBackend()).
  const double frame_time = _header + td_;
  // P6 starvation-adaptive weak update: the scale for *this* frame comes from
  // the current starvation gap (time since the last applied visual update).
  // 1.0 while the gap is at or below SCHUR_STARVATION_SECONDS, ramping to
  // SCHUR_WEAK_UPDATE_MAX_SCALE at SCHUR_STARVATION_DEEP_SECONDS. Disabled or
  // invalid parameters give 1.0, which is a strict no-op.
  double weak_scale = vins::backend::weakUpdateScale(
    schur_starvation_monitor_.starvationDuration(frame_time), SCHUR_STARVATION_WEAK_UPDATE,
    SCHUR_STARVATION_SECONDS, SCHUR_STARVATION_DEEP_SECONDS, SCHUR_WEAK_UPDATE_MAX_SCALE);

  // Collect this frame's observations for every configured camera. Column
  // layout is [x, y, 1, u, v, vx, vy] where (x, y) are normalised
  // coordinates. Left/right views share a feature id and a clone index; both
  // are real measurements and must be registered. `feature_count` stays the
  // number of distinct tracks for the hold policy, while FeatureHistory's
  // min_observations keeps counting raw observations (a stereo pair is two).
  const std::size_t num_cameras = std::max<std::size_t>(1, schur_cameras_.size());
  std::vector<std::vector<vins::backend::FeatureHistory::FrameObservation_s>> frame_observations(
    num_cameras);
  for (auto & camera_observations : frame_observations)
    camera_observations.reserve(_image.size());
  int feature_count = 0;
  for (const auto & feature : _image) {
    bool feature_observed = false;
    for (const auto & observation : feature.second) {
      const int camera_id = observation.first;
      if (camera_id < 0 || static_cast<std::size_t>(camera_id) >= num_cameras)
        continue;  // camera not configured in this estimator
      const Eigen::Matrix<double, 7, 1> & value = observation.second;
      vins::backend::FeatureHistory::FrameObservation_s observation_record;
      observation_record.feature_id_ = feature.first;
      observation_record.uv_ = Eigen::Vector2d(value(0), value(1));
      observation_record.pixel_uv_ = Eigen::Vector2d(value(3), value(4));
      observation_record.has_pixel_uv_ = observation_record.pixel_uv_.allFinite();
      frame_observations[static_cast<std::size_t>(camera_id)].push_back(observation_record);
      feature_observed = true;
    }
    if (feature_observed)
      ++feature_count;
  }

  // Landing static detector. It consumes the same image-time timeline as the
  // EKF and is evaluated on every frame, including frames where the visual
  // update is rejected. The detector only *recommends* a zero-velocity
  // pseudo-measurement; the actual EKF update happens below after the visual
  // update.
  // The landing detector matches feature ids across images in one image
  // plane, so it deliberately keeps consuming camera 0 only. Mixing the right
  // camera's pixels into the same id map would corrupt the flow estimate.
  const std::vector<vins::backend::FeatureHistory::FrameObservation_s> & camera0_observations =
    frame_observations.front();
  std::vector<vins::backend::LandingPixelObservation_s> landing_pixels;
  landing_pixels.reserve(camera0_observations.size());
  for (const auto & observation : camera0_observations) {
    if (observation.has_pixel_uv_)
      landing_pixels.push_back(
        vins::backend::LandingPixelObservation_s{observation.feature_id_, observation.pixel_uv_});
  }
  const vins::backend::LandingFrameResult_s landing = schur_landing_static_.observeFrame(
    frame_time, landing_pixels, schur_state_.imu().v_, schur_state_.imu().ba_, schur_state_.imu().bg_);
  if (landing.entered_) {
    RCLCPP_WARN(
      vins::logger(),
      "Landing ZUPT armed at frame %d (t=%.3f): impact_age=%.2fs "
      "still=%.2fs matched=%d still_features=%d quadrants=%d flow=%.3fpx "
      "gyro=%.4f accel_dev=%.3f |v|=%.3f",
      schur_frame_index_, frame_time, landing.impact_age_, landing.still_duration_, landing.matched_features_,
      landing.still_features_, landing.quadrants_, landing.median_flow_px_, landing.gyro_norm_,
      landing.accel_dev_, schur_state_.imu().v_.norm());
  }
  if (landing.exited_) {
    RCLCPP_INFO(
      vins::logger(),
      "Landing ZUPT exited at frame %d (t=%.3f): reason=%s matched=%d "
      "quadrants=%d flow=%.3fpx gyro=%.4f accel_dev=%.3f |v|=%.3f",
      schur_frame_index_, frame_time, landing.exit_reason_.c_str(), landing.matched_features_,
      landing.quadrants_, landing.median_flow_px_, landing.gyro_norm_, landing.accel_dev_,
      schur_state_.imu().v_.norm());
  }

  // Keep publication dedup across EKF track consumption and hold/rebuild.
  // Current image IDs bridge the history erase/reinsert boundary. Lost IDs
  // remain only while retained observations could still publish a point.
  if (schur_margin_history_.size() != 0) {
    std::unordered_set<int> margin_live_ids;
    for (const auto & camera_observations : frame_observations)
      for (const auto & observation : camera_observations)
        margin_live_ids.insert(observation.feature_id_);
    for (const auto & track : schur_history_.allTracks(schur_oldest_frame_))
      margin_live_ids.insert(track.id_);
    schur_margin_history_.retain(margin_live_ids);
  }

  // P7 hold fast path. While holding, this frame is a pure-IMU frame: no
  // history entry, no clone augmentation, no visual update and no
  // marginalisation. The live state keeps propagating (schurPropagate()) and
  // is written back/published every frame, so odometry stays continuous. Once
  // the tracker reports enough distinct features for enough consecutive
  // frames, the stale clone blocks/history are dropped (live P untouched) and
  // this same frame starts rebuilding the window by falling through to the
  // normal path. That frame starts with an empty history, and the first
  // rebuilt entry is neither lost nor marginalisable while the window fills,
  // so stale observations are never absorbed.
  if (schur_hold_.active()) {
    const vins::backend::StarvationHold::Action_e hold_action =
      schur_hold_.observeFrame(frame_time, feature_count);
    if (hold_action == vins::backend::StarvationHold::Action_e::K_HOLD) {
      // Hold has no visual structure; never expose the previous frame's
      // snapshot as if it belonged to this image.
      schur_visual_snapshot_ = vins::backend::SchurVisualizationSnapshot_s();
      schur_visualization_stamp_ = builtin_interfaces::msg::Time{};
      if (!schurStateFinite()) {
        schur_ekf_active_ = false;
        RCLCPP_WARN(
          vins::logger(),
          "Schur EKF state became non-finite while holding at t=%.3f; "
          "falling back to the Ceres backend",
          frame_time);
        return false;
      }
      if (schur_hold_.frames() % K_SCHUR_STARVATION_REPORT_FRAMES == 0)
        schurReportStarvation(frame_time);
      ++schur_update_count_;
      schurWriteBackWindow();
      schurReportStageTiming(t_update.toc());
      return true;
    }
    schurHoldRecover(frame_time, feature_count);
    // The monitor was re-anchored, so the weak-update ramp starts at zero.
    weak_scale = vins::backend::weakUpdateScale(
      schur_starvation_monitor_.starvationDuration(frame_time), SCHUR_STARVATION_WEAK_UPDATE,
      SCHUR_STARVATION_SECONDS, SCHUR_STARVATION_DEEP_SECONDS, SCHUR_WEAK_UPDATE_MAX_SCALE);
  }
  const bool debug_schur = std::getenv("VINS_SCHUR_DEBUG") != nullptr;
  auto covariance_bounds = [](const Eigen::MatrixXd & _P, double & _trace, double & _min_eig, double & _max_eig)
  {
    _trace = _P.trace();
    _min_eig = std::numeric_limits<double>::quiet_NaN();
    _max_eig = std::numeric_limits<double>::quiet_NaN();
    if (!_P.allFinite())
      return;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(0.5 * (_P + _P.transpose()));
    if (eig.info() == Eigen::Success) {
      _min_eig = eig.eigenvalues().minCoeff();
      _max_eig = eig.eigenvalues().maxCoeff();
    }
  };
  double p_trace_before = 0.0, p_min_before = 0.0, p_max_before = 0.0;
  double p_trace_augmented = 0.0, p_min_augmented = 0.0, p_max_augmented = 0.0;
  if (debug_schur)
    covariance_bounds(schur_state_.covariance(), p_trace_before, p_min_before, p_max_before);
  const vins::backend::ImuNominal_s imu_before_update = schur_state_.imu();

  // 1. Record this frame's normalised, undistorted observations for every
  //    configured camera under the same clone/frame index. FeatureHistory
  //    keeps one row per (frame, camera), so a stereo pair contributes two
  //    real measurements while still sharing a single clone.
  for (std::size_t camera_id = 0; camera_id < frame_observations.size(); ++camera_id)
    schur_history_.addFrameWithPixels(
      schur_frame_index_, static_cast<int>(camera_id), frame_observations[camera_id]);

  // 2. Append the current pose as the newest clone. Pruning is deferred until
  //    after the visual update so the oldest clone can still be observed by
  //    the tracks that are about to be marginalised (OpenVINS/SchurVINS
  //    ordering: augment -> update -> marginalise).
  schur_state_.augmentClone(_header + td_);
  if (debug_schur)
    covariance_bounds(schur_state_.covariance(), p_trace_augmented, p_min_augmented, p_max_augmented);

  // 3. Clone i corresponds to frame index (schur_oldest_frame_ + i), so the
  //    oldest retained clone is the one that will be pruned in step 6.
  schur_oldest_frame_ = schur_frame_index_ - (schur_state_.numClones() - 1);
  schur_history_.trim(schur_oldest_frame_);

  // 4. MSCKF track selection. Only features that are lost in this frame, or
  //    whose oldest observation belongs to the clone that is about to be
  //    margined out, may enter the update. Each such track is consumed
  //    exactly once, which is what keeps the filter consistent: re-inserting
  //    the same rows on consecutive frames double counts the measurements,
  //    collapses the covariance and produces spurious velocity corrections.
  // OpenVINS/SchurVINS rule (VioManager.cpp:366-369): while the clone window
  // is still filling up there is nothing to marginalise, so only features
  // that are actually lost this frame may be used. Only once the window is
  // full (i.e. this frame will prune the oldest clone) do we also admit
  // tracks whose oldest observation sits on that outgoing clone.
  const bool window_full = schur_state_.numClones() > schur_state_.options().window_size_;
  vins::backend::TrackSelection_s selection = schur_history_.tracks(
    schur_oldest_frame_, schur_frame_index_, schur_updater_.options().min_observations_,
    static_cast<size_t>(schur_updater_.options().max_features_), window_full);

  vins::backend::UpdateSummary_s summary;
  // P6: inject the per-frame weak update scale into the updater options before
  // the update. With weak_scale == 1.0 this leaves the update unchanged.
  schur_updater_.setWeakUpdateScale(weak_scale);
  if (!selection.tracks_.empty()) {
    summary = schur_updater_.update(schur_state_, selection.tracks_, schur_cameras_);
    schur_features_total_ += summary.features_used_;
    schur_rejected_total_ += summary.features_rejected_;
  }
  schur_lost_total_ += selection.lost_count_;
  schur_marginalized_total_ += selection.marginalized_count_;

  // Landing zero-velocity pseudo-measurement. It is applied only while the
  // detector is active, after the visual update and before publication, so
  // both the image-time window and the latest real-time state see the same
  // corrected velocity. A successful update also rebaselines the starvation
  // monitor: a physically supported stationary update is not visual
  // starvation and must not trigger a restart.
  bool zupt_applied = false;
  if (landing.active_) {
    const vins::backend::ZeroVelocityUpdateResult_s zupt =
      vins::backend::applyZeroVelocityUpdate(schur_state_, SCHUR_LANDING_ZUPT_SIGMA);
    zupt_applied = zupt.applied_;
    if (zupt.applied_) {
      ++schur_landing_static_updates_;
      const double before = (schur_state_.imu().v_ - zupt.dv_).norm();
      if (
        schur_landing_static_updates_ == 1 ||
        (SCHUR_LANDING_ZUPT_UPDATE_LOG_INTERVAL > 0 &&
         schur_landing_static_updates_ % SCHUR_LANDING_ZUPT_UPDATE_LOG_INTERVAL == 0)) {
        RCLCPP_INFO(
          vins::logger(),
          "Landing ZUPT update #%d at frame %d (t=%.3f): "
          "|v| %.4f -> %.4f m/s, |dv|=%.4f, chi2=%.3f, sigma=%.3f",
          schur_landing_static_updates_, schur_frame_index_, frame_time, before, schur_state_.imu().v_.norm(),
          zupt.dv_.norm(), zupt.chi2_, zupt.sigma_);
      }
    } else {
      ++schur_landing_static_failures_;
      if (
        schur_landing_static_failures_ == 1 ||
        (SCHUR_LANDING_ZUPT_UPDATE_LOG_INTERVAL > 0 &&
         schur_landing_static_failures_ % SCHUR_LANDING_ZUPT_UPDATE_LOG_INTERVAL == 0))
        RCLCPP_WARN(
          vins::logger(), "Landing ZUPT update rejected at frame %d (t=%.3f): %s", schur_frame_index_,
          frame_time, zupt.failure_reason_.c_str());
    }
  }

  // Publication-only snapshot. Build it before consumed tracks are erased so
  // loop_fusion still sees the complete feature set of this image, but
  // triangulate copies only: neither the filter history nor the accepted
  // measurement tracks are modified. The snapshot is deliberately separate
  // from FeatureManager because Schur never runs setDepth(), which is the
  // only place that sets Ceres solve_flag=1.
  if (schurVisualizationRequested()) {
    const int keyframe_clone_index = schur_frame_index_ - schur_oldest_frame_;
    const std::vector<int> consumed_for_margin =
      summary.applied_ ? summary.used_feature_ids_ : std::vector<int>{};
    schur_visual_snapshot_ = vins::backend::makeSchurVisualizationSnapshot(
      schur_state_, schur_cameras_, schur_updater_, schur_history_.allTracks(schur_oldest_frame_),
      keyframe_clone_index, frame_time, consumed_for_margin, &schur_margin_history_);
    schur_visualization_stamp_ = current_feature_stamp_.value_or(vins::stamp(_header));
  } else {
    schur_visual_snapshot_ = vins::backend::SchurVisualizationSnapshot_s();
    schur_visualization_stamp_ = builtin_interfaces::msg::Time{};
  }

  // 5. Consume the tracks that actually entered the update. A rejected track
  //    is kept so it can be retried while it is still inside the window; the
  //    window trim bounds its lifetime.
  if (summary.applied_) {
    schur_history_.eraseFeatures(summary.used_feature_ids_);
    schur_consecutive_failures_ = 0;
    // A successful update ends any starvation: it re-baselines the time
    // criterion and switches the process-noise inflation back off (the
    // inter-update gap is folded into the monitor's diagnostic maximum). It
    // also clears the P5 soft-restart escalation timer.
    schur_soft_restart_time_ = -1.0;
    schur_starvation_monitor_.noteAppliedUpdate(frame_time);
  } else if (zupt_applied) {
    // The zero-velocity pseudo-measurement corrected the state even though
    // no visual track was applied. Treat the frame as a successful update
    // so the starvation time gate and frame-count backstop stay suppressed
    // while the platform is stationary.
    schur_consecutive_failures_ = 0;
    schur_soft_restart_time_ = -1.0;
    schur_starvation_monitor_.noteAppliedUpdate(frame_time);
  } else {
    ++schur_consecutive_failures_;
    ++schur_skipped_frames_;
    if (schur_consecutive_failures_ == 1 || schur_consecutive_failures_ % 10 == 0)
      RCLCPP_WARN(
        vins::logger(),
        "Schur EKF skipped frame %d (streak=%d, tracks=%zu lost=%d marg=%d, "
        "used=%d rejected=%d, weak_scale=%.2f): %s",
        schur_frame_index_, schur_consecutive_failures_, selection.tracks_.size(), selection.lost_count_,
        selection.marginalized_count_, summary.features_used_, summary.features_rejected_, weak_scale,
        summary.failure_reason_.empty() ? "no eligible tracks" : summary.failure_reason_.c_str());

    // P6: a weak-update frame that is *still* rejected is the case the
    // feature exists to fix, so surface it (count-throttled: 1st and every
    // 20th occurrence). The counter is process-wide and diagnostic only.
    if (weak_scale > 1.0) {
      static int weak_skip_warn_count = 0;
      ++weak_skip_warn_count;
      if (weak_skip_warn_count == 1 || weak_skip_warn_count % 20 == 0)
        RCLCPP_WARN(
          vins::logger(),
          "Schur EKF weak-update frame %d still skipped (occurrence=%d, "
          "weak_scale=%.2f, tracks=%zu used=%d rejected=%d): %s",
          schur_frame_index_, weak_skip_warn_count, weak_scale, selection.tracks_.size(),
          summary.features_used_, summary.features_rejected_,
          summary.failure_reason_.empty() ? "no eligible tracks" : summary.failure_reason_.c_str());
    }

    // Time-based starvation criterion (P1/P4). Semantics: evaluated only
    // here, i.e. once per image frame that did NOT apply a visual update; it
    // is never evaluated inside schurPropagate(), so a single large IMU gap
    // cannot trigger a restart by itself. A frame that DOES apply an update
    // re-baselines the monitor above and never reaches this code.
    // The monitor re-baselines on a backwards timestamp (bag replay / clock
    // rollback) instead of restarting, and drives the process-noise
    // inflation stage (P3/P4) from the gap.
    // A restart is a *last resort*: on ARM it costs a full Ceres
    // re-initialisation (10-44 s without odometry), so the time criterion is
    // the primary guard (SCHUR_MAX_NO_UPDATE_SECONDS, default 3 s, versus
    // starvation stretches that usually self-heal within 1-2 s) and
    // SCHUR_MAX_CONSECUTIVE_FAILURES is only a frame-count backstop
    // (default 60). Both are suppressed while the restart cooldown
    // (SCHUR_RESTART_COOLDOWN_SECONDS) is active, which bounds a restart
    // storm to at most one restart per cooldown window.
    const vins::backend::StarvationMonitor::Decision_s starvation =
      schur_starvation_monitor_.observeFailedFrame(frame_time);
    if (starvation.deep_escalated_)
      RCLCPP_WARN(
        vins::logger(),
        "Schur EKF deep starvation started at frame %d (%.3f s without a "
        "visual update > %.3f s): escalating the process-noise scale to %.2f",
        schur_frame_index_, starvation.gap_seconds_, SCHUR_STARVATION_DEEP_SECONDS,
        SCHUR_STARVATION_DEEP_NOISE_SCALE);
    bool restarted = false;
    // P5 soft starvation restart. When enabled while the Schur EKF is
    // active, the time-criterion restart keeps the IMU/EKF nominal state
    // instead of tearing the estimator down (see schurSoftRestart()).
    // Escalation backstop: a soft restart that is not followed by an applied
    // visual update within SCHUR_SOFT_RESTART_MAX_SECONDS of sensor time is
    // given up on and replaced by one full controlled restart. The anchor is
    // only armed by the *first* soft restart of an unrecovered episode (see
    // schurSoftRestart()), so a chain of repeat soft restarts cannot postpone
    // the escalation forever.
    const bool soft_mode = SCHUR_STARVATION_SOFT_RESTART && schur_ekf_active_;
    // P7 hold mode never restarts: the time criterion and the frame-count
    // backstop both enter the hold state instead. It takes precedence over
    // the P5 soft restart (and therefore over its escalation backstop).
    const bool hold_mode = SCHUR_STARVATION_HOLD && schur_ekf_active_;
    if (
      !hold_mode && soft_mode &&
      vins::backend::softRestartEscalationDue(
        frame_time, schur_soft_restart_time_, SCHUR_SOFT_RESTART_MAX_SECONDS)) {
      RCLCPP_ERROR(
        vins::logger(),
        "Schur EKF soft restart at t=%.3f did not apply a visual update "
        "for %.3f s (> %.3f s) at frame %d; escalating to a full "
        "controlled estimator restart",
        schur_soft_restart_time_, frame_time - schur_soft_restart_time_, SCHUR_SOFT_RESTART_MAX_SECONDS,
        schur_frame_index_);
      schur_soft_restart_time_ = -1.0;
      schur_consecutive_failures_ = 0;
      schur_starvation_monitor_.noteRestart(frame_time);
      scheduleControlledRestart();
      restarted = true;
    } else if (starvation.restart_) {
      if (hold_mode) {
        // "不重启、只保持": keep the live state and P, stop accepting
        // visual updates and keep publishing pure-IMU propagation.
        // schurHoldEnter() logs the gap and dropped clone count.
        schurHoldEnter(frame_time, starvation.gap_seconds_);
      } else if (soft_mode) {
        // Keep propagating/publishing from the current IMU state; only
        // the visual structure is dropped. schurSoftRestart() logs the
        // gap, frame index and dropped clone count itself.
        schurSoftRestart(frame_time, starvation.gap_seconds_);
      } else {
        RCLCPP_ERROR(
          vins::logger(),
          "Schur EKF visually starved for %.3f s (> %.3f s) at frame %d: "
          "streak=%d skipped_total=%d max_no_update_gap=%.3f s; scheduling "
          "a controlled estimator restart",
          starvation.gap_seconds_, SCHUR_MAX_NO_UPDATE_SECONDS, schur_frame_index_,
          schur_consecutive_failures_, schur_skipped_frames_, schur_starvation_monitor_.maxGapSeconds());
        schur_consecutive_failures_ = 0;
        // Arm the cooldown before scheduling: the restart runs
        // clearState()/initSchurBackend(), which re-seed the monitor via
        // reset() and must not drop the anchor.
        schur_starvation_monitor_.noteRestart(frame_time);
        scheduleControlledRestart();
      }
      restarted = true;
    } else if (starvation.cooldown_suppressed_) {
      if (hold_mode) {
        // The cooldown only bounds restarts; entering hold is free, so a
        // cooldown-suppressed trigger still enters hold.
        schurHoldEnter(frame_time, starvation.gap_seconds_);
        restarted = true;
      } else {
        const int suppressed = schur_starvation_monitor_.cooldownSuppressions();
        if (suppressed == 1 || suppressed % 10 == 0)
          RCLCPP_WARN(
            vins::logger(),
            "Schur EKF restart suppressed by the cooldown (%.3f s without a "
            "visual update, %.3f s of %.3f s cooldown elapsed) at frame %d: "
            "suppressed_total=%d streak=%d skipped_total=%d",
            starvation.gap_seconds_, frame_time - schur_starvation_monitor_.lastRestartTime(),
            SCHUR_RESTART_COOLDOWN_SECONDS, schur_frame_index_, suppressed, schur_consecutive_failures_,
            schur_skipped_frames_);
      }
    }
    // Frame-count backstop: only when the time criterion did not restart this
    // frame AND the cooldown is not holding restarts back. The failure streak
    // keeps accumulating while suppressed so the diagnostic is not lost. In
    // P5 soft mode the backstop also takes the soft path (keep the IMU/EKF
    // state) instead of clearState(); only the escalation backstop is allowed
    // to schedule a full restart in that mode.
    const vins::backend::StarvationRestartAction_e backstop_action = vins::backend::frameCountBackstopAction(
      restarted, schur_starvation_monitor_.cooldownActive(frame_time), schur_consecutive_failures_,
      SCHUR_MAX_CONSECUTIVE_FAILURES, soft_mode, hold_mode);
    if (backstop_action == vins::backend::StarvationRestartAction_e::K_HOLD) {
      RCLCPP_ERROR(
        vins::logger(),
        "Schur EKF produced %d consecutive frames without a visual update "
        "(%d skipped in total) at frame %d; entering hold "
        "(no restart, keeping the IMU/EKF state and covariance)",
        schur_consecutive_failures_, schur_skipped_frames_, schur_frame_index_);
      schurHoldEnter(frame_time, starvation.gap_seconds_);
    } else if (backstop_action == vins::backend::StarvationRestartAction_e::K_SOFT) {
      RCLCPP_ERROR(
        vins::logger(),
        "Schur EKF produced %d consecutive frames without a visual update "
        "(%d skipped in total) at frame %d; performing a soft restart "
        "(keeping the IMU/EKF state, no re-initialisation)",
        schur_consecutive_failures_, schur_skipped_frames_, schur_frame_index_);
      schurSoftRestart(frame_time, starvation.gap_seconds_);
    } else if (backstop_action == vins::backend::StarvationRestartAction_e::K_FULL) {
      RCLCPP_ERROR(
        vins::logger(),
        "Schur EKF produced %d consecutive frames without a visual update "
        "(%d skipped in total); scheduling a controlled estimator restart",
        schur_consecutive_failures_, schur_skipped_frames_);
      schur_consecutive_failures_ = 0;
      schur_starvation_monitor_.noteRestart(frame_time);
      scheduleControlledRestart();
    }
  }

  // 6. Drop the oldest clone now that the tracks observing it have been used.
  //    A frame that just entered hold has no clones left, so skip the index
  //    bookkeeping (schurHoldRecover() re-seeds it on recovery).
  if (schur_state_.numClones() > 0) {
    schur_state_.pruneCloneWindow();
    schur_oldest_frame_ = schur_frame_index_ - (schur_state_.numClones() - 1);
    schur_history_.trim(schur_oldest_frame_);
  }

  // 6b. P7 rebuild accounting: after leaving hold the window is rebuilt one
  //     clone per frame. Report once it is back to full size so the recovery
  //     cost is visible in the logs.
  if (schur_rebuild_pending_) {
    ++schur_rebuild_frames_;
    if (schur_state_.numClones() >= schur_state_.options().window_size_) {
      schur_rebuild_pending_ = false;
      schur_rebuild_last_frames_ = schur_rebuild_frames_;
      RCLCPP_INFO(
        vins::logger(),
        "Schur EKF clone window rebuilt after hold in %d frames "
        "(%d clones, t=%.3f): visual updates resume from the next frame",
        schur_rebuild_frames_, schur_state_.numClones(), frame_time);
    }
  }

  if (debug_schur) {
    double p_trace_after = 0.0, p_min_after = 0.0, p_max_after = 0.0;
    covariance_bounds(schur_state_.covariance(), p_trace_after, p_min_after, p_max_after);
    const auto & im = schur_state_.imu();
    const Eigen::Vector3d dv = im.v_ - imu_before_update.v_;
    const Eigen::Vector3d dba = im.ba_ - imu_before_update.ba_;
    const Eigen::Vector3d dbg = im.bg_ - imu_before_update.bg_;
    const Eigen::Vector3d dp = im.p_ - imu_before_update.p_;
    fprintf(
      stderr,
      "[schur-step] frame=%d t=%.3f n=%d tracks=%zu lost=%d marg=%d used=%d rej=%d "
      "obs=%d dof=%d applied=%d rank=%d cond=%.3g reason='%s' "
      "P=[%.4g/%.3g/%.3g] -> aug=[%.4g/%.3g/%.3g] -> post=[%.4g/%.3g/%.3g] "
      "|dp|=%.3g |dv|=%.3g |dba|=%.3g |dbg|=%.3g "
      "p=%.4f v=%.4f ba=%.4f bg=%.5f rms=%.3f maxres=%.3f\n",
      schur_frame_index_, _header + td_, schur_state_.size(), selection.tracks_.size(), selection.lost_count_,
      selection.marginalized_count_, summary.features_used_, summary.features_rejected_,
      summary.observations_used_, summary.degrees_of_freedom_, static_cast<int>(summary.applied_),
      summary.rank_, summary.condition_, summary.failure_reason_.c_str(), p_trace_before, p_min_before,
      p_max_before, p_trace_augmented, p_min_augmented, p_max_augmented, p_trace_after, p_min_after,
      p_max_after, dp.norm(), dv.norm(), dba.norm(), dbg.norm(), im.p_.norm(), im.v_.norm(), im.ba_.norm(),
      im.bg_.norm(), summary.mean_residual_px_, summary.max_residual_px_);
  }
  // Periodic starvation summary (every kSchurStarvationReportFrames frames).
  if (schur_frame_index_ > 0 && schur_frame_index_ % K_SCHUR_STARVATION_REPORT_FRAMES == 0)
    schurReportStarvation(frame_time);

  ++schur_frame_index_;
  ++schur_update_count_;

  if (!schurStateFinite()) {
    schur_ekf_active_ = false;
    schur_visual_snapshot_ = vins::backend::SchurVisualizationSnapshot_s();
    schur_visualization_stamp_ = builtin_interfaces::msg::Time{};
    RCLCPP_WARN(
      vins::logger(),
      "Schur EKF state became non-finite at frame %d; falling back to the "
      "Ceres backend",
      schur_frame_index_ - 1);
    return false;
  }

  // 7. Publish the corrected state through the existing window arrays so the
  //    visualisation and odometry output paths stay unchanged.
  schurWriteBackWindow();
  schurReportStageTiming(t_update.toc());
  return true;
}

void Estimator::schurHoldEnter(double _frame_time, double _gap_seconds)
{
  if (!schur_hold_.enter(_frame_time))
    return;
  const int frame_index = schur_frame_index_;
  // Drop only the visual structure. Every clone block is hard-deleted, so the
  // live 15-dim IMU block of P survives bit-for-bit: no reset(), no
  // re-inflation. This also keeps the per-IMU propagation cost at 15x15 while
  // holding instead of growing with a stale window.
  const int dropped_clones = schur_state_.dropAllClones();
  schur_history_.clear();
  schur_landing_static_.reset();
  schur_visual_snapshot_ = vins::backend::SchurVisualizationSnapshot_s();
  schur_visualization_stamp_ = builtin_interfaces::msg::Time{};
  schur_frame_index_ = 0;
  schur_oldest_frame_ = 0;
  schur_consecutive_failures_ = 0;
  // Hold supersedes any pending P5 escalation: nothing is escalated while we
  // hold, because holding never restarts.
  schur_soft_restart_time_ = -1.0;
  RCLCPP_WARN(
    vins::logger(),
    "Schur EKF hold entered at frame %d (t=%.3f): %.3f s without a visual "
    "update (> %.3f s). No restart and no covariance re-inflation: dropped %d "
    "clones, live state and P kept; pure-IMU propagation continues and the "
    "estimate keeps being published. Episode=%d",
    frame_index, _frame_time, _gap_seconds, SCHUR_MAX_NO_UPDATE_SECONDS, dropped_clones,
    schur_hold_.episodes());
}

void Estimator::schurHoldRecover(double _frame_time, int _features)
{
  const int episode = schur_hold_.episodes();
  const double duration = schur_hold_.leave(_frame_time);
  // The tracker recovered: drop whatever visual structure is left and start a
  // fresh frame numbering. The live state and its covariance (which grew
  // naturally during the hold) are deliberately kept untouched.
  schur_state_.dropAllClones();
  schur_history_.clear();
  schur_landing_static_.reset();
  schur_visual_snapshot_ = vins::backend::SchurVisualizationSnapshot_s();
  schur_visualization_stamp_ = builtin_interfaces::msg::Time{};
  schur_frame_index_ = 0;
  schur_oldest_frame_ = 0;
  schur_consecutive_failures_ = 0;
  // Re-anchor the time criterion so the rebuilt window has a full
  // SCHUR_MAX_NO_UPDATE_SECONDS grace period to fill before it can fire again.
  schur_starvation_monitor_.rebaseline(_frame_time);
  schur_rebuild_frames_ = 0;
  schur_rebuild_pending_ = true;
  RCLCPP_WARN(
    vins::logger(),
    "Schur EKF hold exit after %.3f s (episode=%d, t=%.3f): tracker recovered "
    "with %d distinct features (>= %d for %d consecutive frames). Rebuilding the "
    "clone window from this frame, live state and P kept",
    duration, episode, _frame_time, _features, SCHUR_MIN_FEATURES, SCHUR_HOLD_RECOVERY_FRAMES);
}

void Estimator::schurSoftRestart(double _frame_time, double _gap_seconds)
{
  // P5 soft starvation restart. Keep the IMU/EKF nominal state (q/p/v/ba/bg)
  // and drop only the visual structure: no clearState(), no Ceres/SfM
  // re-initialisation, no solver_flag change and no publication suppression,
  // so IMU propagation and odometry output continue through the starvation
  // stretch. This is what removes the 27-37 s odometry gap that a full
  // controlled restart costs on ARM.
  const vins::backend::ImuNominal_s imu = schur_state_.imu();
  const int dropped_clones = schur_state_.numClones();
  const int frame_index = schur_frame_index_;
  // Discard the clone window and the old covariance, re-inflating P from the
  // configured init_*_sigma values while preserving the motion state.
  schur_state_.reset(schur_state_.options(), _frame_time, imu.q_, imu.p_, imu.v_, imu.ba_, imu.bg_);
  schur_history_.clear();
  schur_landing_static_.reset();
  schur_visual_snapshot_ = vins::backend::SchurVisualizationSnapshot_s();
  schur_visualization_stamp_ = builtin_interfaces::msg::Time{};
  schur_frame_index_ = 0;
  schur_oldest_frame_ = 0;
  schur_consecutive_failures_ = 0;
  schur_skipped_frames_ = 0;
  // Arm the escalation timer only when no soft restart is already pending:
  // repeated soft restarts during one unrecovered episode must not postpone
  // the full-restart escalation (see softRestartEscalationDue()).
  if (schur_soft_restart_time_ < 0.0)
    schur_soft_restart_time_ = _frame_time;
  // Same cooldown semantics as a full restart (P4).
  schur_starvation_monitor_.noteRestart(_frame_time);
  // schur_ekf_active_ is deliberately left true.
  RCLCPP_WARN(
    vins::logger(),
    "Schur EKF soft restart at frame %d: visually starved for %.3f s "
    "(> %.3f s); dropped %d clones, keeping the IMU/EKF state "
    "(q/p/v/ba/bg) without re-initialisation. A full restart is scheduled "
    "if no visual update is applied within %.3f s",
    frame_index, _gap_seconds, SCHUR_MAX_NO_UPDATE_SECONDS, dropped_clones, SCHUR_SOFT_RESTART_MAX_SECONDS);
}

void Estimator::schurWriteBackWindow()
{
  const vins::backend::ImuNominal_s & imu = schur_state_.imu();
  const Eigen::Quaterniond q_imu(imu.q_(0), imu.q_(1), imu.q_(2), imu.q_(3));
  Rs_[frame_count_] = q_imu.toRotationMatrix();
  Ps_[frame_count_] = imu.p_;
  Vs_[frame_count_] = imu.v_;
  Bas_[frame_count_] = imu.ba_;
  Bgs_[frame_count_] = imu.bg_;

  // Refresh the retained clones into the matching window slots so the
  // visualiser shows the corrected (not the propagated) poses.
  const int num_clones = schur_state_.numClones();
  for (int i = 0; i < num_clones; ++i) {
    const int window_index = frame_count_ - (num_clones - 1 - i);
    if (window_index < 0)
      continue;
    const vins::backend::CloneNominal_s & clone = schur_state_.clone(i);
    const Eigen::Quaterniond q_clone(clone.q_(0), clone.q_(1), clone.q_(2), clone.q_(3));
    Rs_[window_index] = q_clone.toRotationMatrix();
    Ps_[window_index] = clone.p_;
  }
}

bool Estimator::schurStateFinite() const
{
  const vins::backend::ImuNominal_s & imu = schur_state_.imu();
  if (!imu.q_.allFinite() || !imu.p_.allFinite() || !imu.v_.allFinite() || !imu.ba_.allFinite() || !imu.bg_.allFinite())
    return false;
  if (!schur_state_.covariance().allFinite())
    return false;
  for (int i = 0; i < schur_state_.numClones(); ++i) {
    const vins::backend::CloneNominal_s & clone = schur_state_.clone(i);
    if (!clone.q_.allFinite() || !clone.p_.allFinite())
      return false;
  }
  return true;
}

void Estimator::schurReportStageTiming(double _update_ms)
{
  schur_stage_total_ms_ += _update_ms;
  if (!STAGE_TIMING || STAGE_TIMING_INTERVAL < 1)
    return;
  if (schur_update_count_ % STAGE_TIMING_INTERVAL != 0)
    return;

  const double samples = static_cast<double>(STAGE_TIMING_INTERVAL);
  RCLCPP_INFO(
    vins::logger(),
    "schur_ekf stage timing (%d samples): update=%.3f ms, used=%.1f, "
    "rejected=%.1f, lost=%.1f, marg=%.1f, skipped=%d, state=%dx%d",
    STAGE_TIMING_INTERVAL, schur_stage_total_ms_ / samples, schur_features_total_ / samples,
    schur_rejected_total_ / samples, schur_lost_total_ / samples, schur_marginalized_total_ / samples,
    schur_skipped_frames_, schur_state_.size(), schur_state_.size());
  schur_stage_total_ms_ = 0.0;
  schur_features_total_ = 0.0;
  schur_rejected_total_ = 0.0;
  schur_lost_total_ = 0.0;
  schur_marginalized_total_ = 0.0;
}

void Estimator::schurReportStarvation(double _frame_time) const
{
  // P6: report the weak-update scale this frame would use, recomputed from the
  // same starvation gap so the summary stays consistent with schurProcessFrame.
  const double weak_scale = vins::backend::weakUpdateScale(
    schur_starvation_monitor_.starvationDuration(_frame_time), SCHUR_STARVATION_WEAK_UPDATE,
    SCHUR_STARVATION_SECONDS, SCHUR_STARVATION_DEEP_SECONDS, SCHUR_WEAK_UPDATE_MAX_SCALE);
  RCLCPP_INFO(
    vins::logger(),
    "schur_ekf starvation summary (%d frames): time_gate_triggers=%d "
    "deep_escalations=%d cooldown_suppressions=%d noise_scale=%.2f "
    "weak_update_scale=%.2f max_no_update_gap=%.3f s current_starvation=%.3f s "
    "consecutive_failures=%d skipped_total=%d "
    "thresholds(max_no_update=%.2f s cooldown=%.2f s) "
    "hold(enabled=%d active=%d episodes=%d frames=%d current=%.3f s "
    "total=%.3f s last=%.3f s max=%.3f s streak=%d/%d rebuild=%s "
    "rebuild_frames=%d)",
    schur_frame_index_, schur_starvation_monitor_.timeGateTriggers(),
    schur_starvation_monitor_.deepEscalations(), schur_starvation_monitor_.cooldownSuppressions(),
    schur_starvation_monitor_.noiseScale(), weak_scale, schur_starvation_monitor_.maxGapSeconds(),
    schur_starvation_monitor_.starvationDuration(_frame_time), schur_consecutive_failures_,
    schur_skipped_frames_, SCHUR_MAX_NO_UPDATE_SECONDS, SCHUR_RESTART_COOLDOWN_SECONDS,
    static_cast<int>(SCHUR_STARVATION_HOLD), static_cast<int>(schur_hold_.active()), schur_hold_.episodes(),
    schur_hold_.frames(), schur_hold_.currentDuration(_frame_time), schur_hold_.totalDuration(),
    schur_hold_.lastDuration(), schur_hold_.maxDuration(), schur_hold_.goodStreak(),
    SCHUR_HOLD_RECOVERY_FRAMES, schur_rebuild_pending_ ? "pending" : "idle", schur_rebuild_last_frames_);
}

bool Estimator::optimization()
{
  TicToc t_whole, t_prepare;
  const bool stage_timing_enabled = STAGE_TIMING;
  TicToc stage_timer;
  double stage_vector2double_ms = 0.0;
  double stage_problem_ms = 0.0;
  double stage_solve_ms = 0.0;
  double stage_double2vector_ms = 0.0;
  double stage_marginalization_ms = 0.0;
  static double stage_vector2double_total_ms = 0.0;
  static double stage_problem_total_ms = 0.0;
  static double stage_solve_total_ms = 0.0;
  static double stage_double2vector_total_ms = 0.0;
  static double stage_marginalization_total_ms = 0.0;
  static std::size_t stage_timing_samples = 0;
  auto finish_stage_timing = [&]()
  {
    if (!stage_timing_enabled)
      return;
    stage_vector2double_total_ms += stage_vector2double_ms;
    stage_problem_total_ms += stage_problem_ms;
    stage_solve_total_ms += stage_solve_ms;
    stage_double2vector_total_ms += stage_double2vector_ms;
    stage_marginalization_total_ms += stage_marginalization_ms;
    ++stage_timing_samples;
    if (stage_timing_samples >= static_cast<std::size_t>(STAGE_TIMING_INTERVAL)) {
      const double samples = static_cast<double>(stage_timing_samples);
      RCLCPP_INFO(
        vins::logger(),
        "optimization stage timing (%zu samples): vector2double=%.3f ms, problem=%.3f ms, "
        "solve=%.3f ms, double2vector=%.3f ms, marginalization=%.3f ms",
        stage_timing_samples, stage_vector2double_total_ms / samples, stage_problem_total_ms / samples,
        stage_solve_total_ms / samples, stage_double2vector_total_ms / samples,
        stage_marginalization_total_ms / samples);
      stage_vector2double_total_ms = 0.0;
      stage_problem_total_ms = 0.0;
      stage_solve_total_ms = 0.0;
      stage_double2vector_total_ms = 0.0;
      stage_marginalization_total_ms = 0.0;
      stage_timing_samples = 0;
    }
  };

  if (stage_timing_enabled)
    stage_timer.tic();
  vector2double();
  if (stage_timing_enabled)
    stage_vector2double_ms = stage_timer.toc();

  if (stage_timing_enabled)
    stage_timer.tic();
  ceres::Problem problem;
  ceres::LossFunction * loss_function;
  //loss_function = NULL;
  loss_function = new ceres::HuberLoss(1.0);
  //loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
  //ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
  for (int i = 0; i < frame_count_ + 1; i++) {
    ceres::Manifold * local_parameterization = new PoseManifold();
    problem.AddParameterBlock(para_Pose_[i], SIZE_POSE, local_parameterization);
    if (USE_IMU)
      problem.AddParameterBlock(para_SpeedBias_[i], SIZE_SPEEDBIAS);
  }
  if (!USE_IMU)
    problem.SetParameterBlockConstant(para_Pose_[0]);

  const bool estimate_ex_pose =
    (ESTIMATE_EXTRINSIC && frame_count_ == WINDOW_SIZE && Vs_[0].norm() > 0.2) || openExEstimation_;
  for (int i = 0; i < NUM_OF_CAM; i++) {
    ceres::Manifold * local_parameterization = new PoseManifold();
    problem.AddParameterBlock(para_Ex_Pose_[i], SIZE_POSE, local_parameterization);
    if (!estimate_ex_pose)
      problem.SetParameterBlockConstant(para_Ex_Pose_[i]);
  }
  problem.AddParameterBlock(para_Td_[0], 1);

  if (!ESTIMATE_TD || Vs_[0].norm() < 0.2)
    problem.SetParameterBlockConstant(para_Td_[0]);

  if (last_marginalization_info_ && last_marginalization_info_->valid_) {
    // construct new marginlization_factor
    MarginalizationFactor * marginalization_factor = new MarginalizationFactor(last_marginalization_info_);
    problem.AddResidualBlock(marginalization_factor, nullptr, last_marginalization_parameter_blocks_);
  }
  if (USE_IMU) {
    for (int i = 0; i < frame_count_; i++) {
      int j = i + 1;
      if (pre_integrations_[j]->sum_dt_ > 10.0)
        continue;
      IMUFactor * imu_factor = new IMUFactor(pre_integrations_[j]);
      problem.AddResidualBlock(
        imu_factor, nullptr, para_Pose_[i], para_SpeedBias_[i], para_Pose_[j], para_SpeedBias_[j]);
    }
  }

  int f_m_cnt = 0;
  int feature_index = -1;
  for (auto & it_per_id : f_manager_.feature_) {
    it_per_id.used_num_ = it_per_id.feature_per_frame_.size();
    if (it_per_id.used_num_ < 4)
      continue;

    ++feature_index;

    int imu_i = it_per_id.start_frame_, imu_j = imu_i - 1;

    Vector3d pts_i = it_per_id.feature_per_frame_[0].point_;

    for (auto & it_per_frame : it_per_id.feature_per_frame_) {
      imu_j++;
      if (imu_i != imu_j) {
        Vector3d pts_j = it_per_frame.point_;
        ProjectionTwoFrameOneCamFactor * f_td = new ProjectionTwoFrameOneCamFactor(
          pts_i, pts_j, it_per_id.feature_per_frame_[0].velocity_, it_per_frame.velocity_,
          it_per_id.feature_per_frame_[0].cur_td_, it_per_frame.cur_td_);
        problem.AddResidualBlock(
          f_td, loss_function, para_Pose_[imu_i], para_Pose_[imu_j], para_Ex_Pose_[0],
          para_Feature_[feature_index], para_Td_[0]);
      }

      if (STEREO && it_per_frame.is_stereo_) {
        Vector3d pts_j_right = it_per_frame.pointRight_;
        if (imu_i != imu_j) {
          ProjectionTwoFrameTwoCamFactor * f = new ProjectionTwoFrameTwoCamFactor(
            pts_i, pts_j_right, it_per_id.feature_per_frame_[0].velocity_, it_per_frame.velocityRight_,
            it_per_id.feature_per_frame_[0].cur_td_, it_per_frame.cur_td_);
          problem.AddResidualBlock(
            f, loss_function, para_Pose_[imu_i], para_Pose_[imu_j], para_Ex_Pose_[0], para_Ex_Pose_[1],
            para_Feature_[feature_index], para_Td_[0]);
        } else {
          ProjectionOneFrameTwoCamFactor * f = new ProjectionOneFrameTwoCamFactor(
            pts_i, pts_j_right, it_per_id.feature_per_frame_[0].velocity_, it_per_frame.velocityRight_,
            it_per_id.feature_per_frame_[0].cur_td_, it_per_frame.cur_td_);
          problem.AddResidualBlock(
            f, loss_function, para_Ex_Pose_[0], para_Ex_Pose_[1], para_Feature_[feature_index], para_Td_[0]);
        }
      }
      f_m_cnt++;
    }
  }

  RCLCPP_DEBUG(vins::logger(), "visual measurement count: %d", f_m_cnt);
  //printf("prepare for ceres: %f \n", t_prepare.toc());
  if (stage_timing_enabled)
    stage_problem_ms = stage_timer.toc();

  const bool transactional_solver = (SOLVER_TIMEOUT_POLICY != "warn");
  CeresParameterSnapshot_s parameter_snapshot{};
  std::memcpy(parameter_snapshot.pose_.data(), para_Pose_, sizeof(para_Pose_));
  std::memcpy(parameter_snapshot.speed_bias_.data(), para_SpeedBias_, sizeof(para_SpeedBias_));
  std::memcpy(parameter_snapshot.feature_.data(), para_Feature_, sizeof(para_Feature_));
  std::memcpy(parameter_snapshot.ex_pose_.data(), para_Ex_Pose_, sizeof(para_Ex_Pose_));
  std::memcpy(parameter_snapshot.td_.data(), para_Td_, sizeof(para_Td_));
  auto restore_parameter_snapshot = [&]()
  {
    std::memcpy(para_Pose_, parameter_snapshot.pose_.data(), sizeof(para_Pose_));
    std::memcpy(para_SpeedBias_, parameter_snapshot.speed_bias_.data(), sizeof(para_SpeedBias_));
    std::memcpy(para_Feature_, parameter_snapshot.feature_.data(), sizeof(para_Feature_));
    std::memcpy(para_Ex_Pose_, parameter_snapshot.ex_pose_.data(), sizeof(para_Ex_Pose_));
    std::memcpy(para_Td_, parameter_snapshot.td_.data(), sizeof(para_Td_));
  };
  static std::size_t timeout_count = 0;
  static std::size_t retry_attempt_count = 0;
  static std::size_t retry_success_count = 0;
  static std::size_t revert_count = 0;
  bool reverted = false;
  auto rollback_parameters = [&]()
  {
    if (reverted)
      return;
    restore_parameter_snapshot();
    reverted = true;
    ++revert_count;
  };

  ceres::Solver::Options options;

  options.linear_solver_type = ceres::DENSE_SCHUR;
  //options.num_threads = 2;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.max_num_iterations = NUM_ITERATIONS;
  //options.use_explicit_schur_complement = true;
  //options.minimizer_progress_to_stdout = true;
  //options.use_nonmonotonic_steps = true;
  if (marginalization_flag_ == MARGIN_OLD)
    options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
  else
    options.max_solver_time_in_seconds = SOLVER_TIME;
  const double first_solver_budget = options.max_solver_time_in_seconds;
  TicToc t_solver;
  ceres::Solver::Summary summary;
  if (stage_timing_enabled)
    stage_timer.tic();
  ceres::Solve(options, &problem, &summary);
  if (stage_timing_enabled)
    stage_solve_ms += stage_timer.toc();
  const double first_solver_elapsed_s = t_solver.toc() / 1000.0;

  const bool timed_out = vins::solverTimedOut(summary, first_solver_elapsed_s, first_solver_budget);
  bool solved = summary.IsSolutionUsable();

  if (timed_out) {
    ++timeout_count;

    if (!transactional_solver) {
      static rclcpp::Clock throttle_clock(RCL_STEADY_TIME);
      if (solved) {
        RCLCPP_WARN_THROTTLE(
          vins::logger(), throttle_clock, 5000,
          "Ceres hit max_solver_time=%.0f ms after %.1f ms and %zu iterations "
          "(policy=warn); accepting the truncated but usable solution, which can bias "
          "the marginalization prior",
          first_solver_budget * 1000.0, first_solver_elapsed_s * 1000.0, summary.iterations.size());
      } else {
        RCLCPP_WARN_THROTTLE(
          vins::logger(), throttle_clock, 5000,
          "Ceres hit max_solver_time=%.0f ms after %.1f ms and %zu iterations "
          "(policy=warn); the solution is unusable and will be rolled back",
          first_solver_budget * 1000.0, first_solver_elapsed_s * 1000.0, summary.iterations.size());
      }
    } else if (SOLVER_TIMEOUT_POLICY == "retry") {
      ++retry_attempt_count;
      restore_parameter_snapshot();

      const double retry_solver_budget = first_solver_budget * SOLVER_TIMEOUT_RETRY_MULTIPLIER;
      options.max_solver_time_in_seconds = retry_solver_budget;
      ceres::Solver::Summary retry_summary;
      TicToc retry_solver_timer;
      if (stage_timing_enabled)
        stage_timer.tic();
      ceres::Solve(options, &problem, &retry_summary);
      if (stage_timing_enabled)
        stage_solve_ms += stage_timer.toc();
      const double retry_solver_elapsed_s = retry_solver_timer.toc() / 1000.0;

      const bool retry_timed_out =
        vins::solverTimedOut(retry_summary, retry_solver_elapsed_s, retry_solver_budget);
      solved = !retry_timed_out && retry_summary.IsSolutionUsable();
      if (solved) {
        ++retry_success_count;
      } else {
        rollback_parameters();
      }

      RCLCPP_WARN(
        vins::logger(),
        "Ceres timeout (policy=retry): first solve hit max_solver_time=%.0f ms after "
        "%.1f ms and %zu iterations; retry with %.0f ms %s after %.1f ms and %zu "
        "iterations; solver parameter rollback=%s",
        first_solver_budget * 1000.0, first_solver_elapsed_s * 1000.0, summary.iterations.size(),
        retry_solver_budget * 1000.0, solved ? "succeeded" : "failed", retry_solver_elapsed_s * 1000.0,
        retry_summary.iterations.size(), reverted ? "true" : "false");
    } else {
      rollback_parameters();
      solved = false;
      RCLCPP_WARN(
        vins::logger(),
        "Ceres timeout (policy=%s): hit max_solver_time=%.0f ms after %.1f ms and "
        "%zu iterations; retry_attempted=false, solver parameter rollback=true. "
        "Marginalization prior and sliding window remain unchanged; current-frame "
        "feature/image/IMU inputs were already advanced and are not rolled back",
        SOLVER_TIMEOUT_POLICY.c_str(), first_solver_budget * 1000.0, first_solver_elapsed_s * 1000.0,
        summary.iterations.size());
    }

    if (timeout_count % 100 == 0) {
      RCLCPP_INFO(
        vins::logger(),
        "Ceres timeout cumulative summary (policy=%s): timeouts=%zu retry_attempts=%zu "
        "retry_successes=%zu reverts=%zu",
        SOLVER_TIMEOUT_POLICY.c_str(), timeout_count, retry_attempt_count, retry_success_count, revert_count);
    }
  } else if (!solved) {
    rollback_parameters();
    RCLCPP_WARN(
      vins::logger(),
      "Ceres solution unusable (policy=%s) after %.1f ms and %zu iterations with "
      "max_solver_time=%.0f ms; solver parameter rollback=true. Marginalization prior "
      "and sliding window remain unchanged; current-frame feature/image/IMU inputs "
      "were already advanced and are not rolled back",
      SOLVER_TIMEOUT_POLICY.c_str(), first_solver_elapsed_s * 1000.0, summary.iterations.size(),
      first_solver_budget * 1000.0);
  }

  //cout << summary.BriefReport() << endl;
  RCLCPP_DEBUG(vins::logger(), "Iterations : %d", static_cast<int>(summary.iterations.size()));
  //printf("solver costs: %f \n", t_solver.toc());

  if (!solved) {
    rollback_parameters();
    RCLCPP_WARN(
      vins::logger(), "Optimization failed: reverted solver parameter blocks only; marginalization prior "
                      "and sliding window were not advanced, and no state will be published. Current-frame "
                      "feature/image/IMU inputs were already advanced and are not part of this rollback");
    finish_stage_timing();
    return false;
  }

  if (stage_timing_enabled)
    stage_timer.tic();
  double2vector();
  if (stage_timing_enabled)
    stage_double2vector_ms = stage_timer.toc();
  if (estimate_ex_pose)
    openExEstimation_ = 1;
  //printf("frame_count: %d \n", frame_count);

  if (frame_count_ < WINDOW_SIZE) {
    finish_stage_timing();
    return true;
  }

  if (stage_timing_enabled)
    stage_timer.tic();
  TicToc t_whole_marginalization;
  if (marginalization_flag_ == MARGIN_OLD) {
    MarginalizationInfo * marginalization_info = new MarginalizationInfo();
    vector2double();

    if (last_marginalization_info_ && last_marginalization_info_->valid_) {
      vector<int> drop_set;
      for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks_.size()); i++) {
        if (
          last_marginalization_parameter_blocks_[i] == para_Pose_[0] ||
          last_marginalization_parameter_blocks_[i] == para_SpeedBias_[0])
          drop_set.push_back(i);
      }
      // construct new marginlization_factor
      MarginalizationFactor * marginalization_factor = new MarginalizationFactor(last_marginalization_info_);
      ResidualBlockInfo_s * residual_block_info = new ResidualBlockInfo_s(
        marginalization_factor, nullptr, last_marginalization_parameter_blocks_, drop_set);
      marginalization_info->addResidualBlockInfo(residual_block_info);
    }

    if (USE_IMU) {
      if (pre_integrations_[1]->sum_dt_ < 10.0) {
        IMUFactor * imu_factor = new IMUFactor(pre_integrations_[1]);
        ResidualBlockInfo_s * residual_block_info = new ResidualBlockInfo_s(
          imu_factor, nullptr,
          vector<double *>{para_Pose_[0], para_SpeedBias_[0], para_Pose_[1], para_SpeedBias_[1]},
          vector<int>{0, 1});
        marginalization_info->addResidualBlockInfo(residual_block_info);
      }
    }

    {
      int feature_index = -1;
      for (auto & it_per_id : f_manager_.feature_) {
        it_per_id.used_num_ = it_per_id.feature_per_frame_.size();
        if (it_per_id.used_num_ < 4)
          continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame_, imu_j = imu_i - 1;
        if (imu_i != 0)
          continue;

        Vector3d pts_i = it_per_id.feature_per_frame_[0].point_;

        for (auto & it_per_frame : it_per_id.feature_per_frame_) {
          imu_j++;
          if (imu_i != imu_j) {
            Vector3d pts_j = it_per_frame.point_;
            ProjectionTwoFrameOneCamFactor * f_td = new ProjectionTwoFrameOneCamFactor(
              pts_i, pts_j, it_per_id.feature_per_frame_[0].velocity_, it_per_frame.velocity_,
              it_per_id.feature_per_frame_[0].cur_td_, it_per_frame.cur_td_);
            ResidualBlockInfo_s * residual_block_info = new ResidualBlockInfo_s(
              f_td, loss_function,
              vector<double *>{
                para_Pose_[imu_i], para_Pose_[imu_j], para_Ex_Pose_[0], para_Feature_[feature_index],
                para_Td_[0]},
              vector<int>{0, 3});
            marginalization_info->addResidualBlockInfo(residual_block_info);
          }
          if (STEREO && it_per_frame.is_stereo_) {
            Vector3d pts_j_right = it_per_frame.pointRight_;
            if (imu_i != imu_j) {
              ProjectionTwoFrameTwoCamFactor * f = new ProjectionTwoFrameTwoCamFactor(
                pts_i, pts_j_right, it_per_id.feature_per_frame_[0].velocity_, it_per_frame.velocityRight_,
                it_per_id.feature_per_frame_[0].cur_td_, it_per_frame.cur_td_);
              ResidualBlockInfo_s * residual_block_info = new ResidualBlockInfo_s(
                f, loss_function,
                vector<double *>{
                  para_Pose_[imu_i], para_Pose_[imu_j], para_Ex_Pose_[0], para_Ex_Pose_[1],
                  para_Feature_[feature_index], para_Td_[0]},
                vector<int>{0, 4});
              marginalization_info->addResidualBlockInfo(residual_block_info);
            } else {
              ProjectionOneFrameTwoCamFactor * f = new ProjectionOneFrameTwoCamFactor(
                pts_i, pts_j_right, it_per_id.feature_per_frame_[0].velocity_, it_per_frame.velocityRight_,
                it_per_id.feature_per_frame_[0].cur_td_, it_per_frame.cur_td_);
              ResidualBlockInfo_s * residual_block_info = new ResidualBlockInfo_s(
                f, loss_function,
                vector<double *>{para_Ex_Pose_[0], para_Ex_Pose_[1], para_Feature_[feature_index], para_Td_[0]},
                vector<int>{2});
              marginalization_info->addResidualBlockInfo(residual_block_info);
            }
          }
        }
      }
    }

    TicToc t_pre_margin;
    marginalization_info->preMarginalize();
    RCLCPP_DEBUG(vins::logger(), "pre marginalization %f ms", t_pre_margin.toc());

    TicToc t_margin;
    marginalization_info->marginalize();
    RCLCPP_DEBUG(vins::logger(), "marginalization %f ms", t_margin.toc());

    std::unordered_map<long, double *> addr_shift;
    for (int i = 1; i <= WINDOW_SIZE; i++) {
      addr_shift[reinterpret_cast<long>(para_Pose_[i])] = para_Pose_[i - 1];
      if (USE_IMU)
        addr_shift[reinterpret_cast<long>(para_SpeedBias_[i])] = para_SpeedBias_[i - 1];
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
      addr_shift[reinterpret_cast<long>(para_Ex_Pose_[i])] = para_Ex_Pose_[i];

    addr_shift[reinterpret_cast<long>(para_Td_[0])] = para_Td_[0];

    vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

    if (last_marginalization_info_)
      delete last_marginalization_info_;
    last_marginalization_info_ = marginalization_info;
    last_marginalization_parameter_blocks_ = parameter_blocks;

  } else {
    if (
      last_marginalization_info_ &&
      std::count(
        std::begin(last_marginalization_parameter_blocks_), std::end(last_marginalization_parameter_blocks_),
        para_Pose_[WINDOW_SIZE - 1])) {

      MarginalizationInfo * marginalization_info = new MarginalizationInfo();
      vector2double();
      if (last_marginalization_info_ && last_marginalization_info_->valid_) {
        vector<int> drop_set;
        for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks_.size()); i++) {
          assert(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
          if (last_marginalization_parameter_blocks_[i] == para_Pose_[WINDOW_SIZE - 1])
            drop_set.push_back(i);
        }
        // construct new marginlization_factor
        MarginalizationFactor * marginalization_factor = new MarginalizationFactor(last_marginalization_info_);
        ResidualBlockInfo_s * residual_block_info = new ResidualBlockInfo_s(
          marginalization_factor, nullptr, last_marginalization_parameter_blocks_, drop_set);

        marginalization_info->addResidualBlockInfo(residual_block_info);
      }

      TicToc t_pre_margin;
      RCLCPP_DEBUG(vins::logger(), "begin marginalization");
      marginalization_info->preMarginalize();
      RCLCPP_DEBUG(vins::logger(), "end pre marginalization, %f ms", t_pre_margin.toc());

      TicToc t_margin;
      RCLCPP_DEBUG(vins::logger(), "begin marginalization");
      marginalization_info->marginalize();
      RCLCPP_DEBUG(vins::logger(), "end marginalization, %f ms", t_margin.toc());

      std::unordered_map<long, double *> addr_shift;
      for (int i = 0; i <= WINDOW_SIZE; i++) {
        if (i == WINDOW_SIZE - 1)
          continue;
        else if (i == WINDOW_SIZE) {
          addr_shift[reinterpret_cast<long>(para_Pose_[i])] = para_Pose_[i - 1];
          if (USE_IMU)
            addr_shift[reinterpret_cast<long>(para_SpeedBias_[i])] = para_SpeedBias_[i - 1];
        } else {
          addr_shift[reinterpret_cast<long>(para_Pose_[i])] = para_Pose_[i];
          if (USE_IMU)
            addr_shift[reinterpret_cast<long>(para_SpeedBias_[i])] = para_SpeedBias_[i];
        }
      }
      for (int i = 0; i < NUM_OF_CAM; i++)
        addr_shift[reinterpret_cast<long>(para_Ex_Pose_[i])] = para_Ex_Pose_[i];

      addr_shift[reinterpret_cast<long>(para_Td_[0])] = para_Td_[0];


      vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
      if (last_marginalization_info_)
        delete last_marginalization_info_;
      last_marginalization_info_ = marginalization_info;
      last_marginalization_parameter_blocks_ = parameter_blocks;
    }
  }
  if (stage_timing_enabled)
    stage_marginalization_ms = stage_timer.toc();
  finish_stage_timing();
  //printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
  //printf("whole time for ceres: %f \n", t_whole.toc());
  return true;
}

void Estimator::slideWindow()
{
  TicToc t_margin;
  if (marginalization_flag_ == MARGIN_OLD) {
    double t_0 = Headers_[0];
    back_R0_ = Rs_[0];
    back_P0_ = Ps_[0];
    if (frame_count_ == WINDOW_SIZE) {
      if (all_image_frame_.find(t_0) == all_image_frame_.end()) {
        RCLCPP_WARN(
          vins::logger(),
          "Cannot slide initialization window because oldest header %.9f is missing "
          "from all_image_frame; scheduling a controlled estimator reset",
          t_0);
        scheduleControlledRestart();
        return;
      }
      for (int i = 0; i < WINDOW_SIZE; i++) {
        Headers_[i] = Headers_[i + 1];
        Rs_[i].swap(Rs_[i + 1]);
        Ps_[i].swap(Ps_[i + 1]);
        if (USE_IMU) {
          std::swap(pre_integrations_[i], pre_integrations_[i + 1]);

          dt_buf_[i].swap(dt_buf_[i + 1]);
          linear_acceleration_buf_[i].swap(linear_acceleration_buf_[i + 1]);
          angular_velocity_buf_[i].swap(angular_velocity_buf_[i + 1]);

          Vs_[i].swap(Vs_[i + 1]);
          Bas_[i].swap(Bas_[i + 1]);
          Bgs_[i].swap(Bgs_[i + 1]);
        }
      }
      Headers_[WINDOW_SIZE] = Headers_[WINDOW_SIZE - 1];
      Ps_[WINDOW_SIZE] = Ps_[WINDOW_SIZE - 1];
      Rs_[WINDOW_SIZE] = Rs_[WINDOW_SIZE - 1];

      if (USE_IMU) {
        Vs_[WINDOW_SIZE] = Vs_[WINDOW_SIZE - 1];
        Bas_[WINDOW_SIZE] = Bas_[WINDOW_SIZE - 1];
        Bgs_[WINDOW_SIZE] = Bgs_[WINDOW_SIZE - 1];

        delete pre_integrations_[WINDOW_SIZE];
        pre_integrations_[WINDOW_SIZE] =
          new IntegrationBase{acc_0_, gyr_0_, Bas_[WINDOW_SIZE], Bgs_[WINDOW_SIZE]};

        dt_buf_[WINDOW_SIZE].clear();
        linear_acceleration_buf_[WINDOW_SIZE].clear();
        angular_velocity_buf_[WINDOW_SIZE].clear();
      }

      {
        auto it_0 = all_image_frame_.find(t_0);
        auto erase_end = std::next(it_0);
        for (auto frame_it = all_image_frame_.begin(); frame_it != erase_end; ++frame_it) {
          delete frame_it->second.pre_integration_;
          frame_it->second.pre_integration_ = nullptr;
        }
        all_image_frame_.erase(all_image_frame_.begin(), erase_end);
      }
      slideWindowOld();
    }
  } else {
    if (frame_count_ == WINDOW_SIZE) {
      Headers_[frame_count_ - 1] = Headers_[frame_count_];
      Ps_[frame_count_ - 1] = Ps_[frame_count_];
      Rs_[frame_count_ - 1] = Rs_[frame_count_];

      if (USE_IMU) {
        for (unsigned int i = 0; i < dt_buf_[frame_count_].size(); i++) {
          double tmp_dt = dt_buf_[frame_count_][i];
          Vector3d tmp_linear_acceleration = linear_acceleration_buf_[frame_count_][i];
          Vector3d tmp_angular_velocity = angular_velocity_buf_[frame_count_][i];

          pre_integrations_[frame_count_ - 1]->pushBack(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

          dt_buf_[frame_count_ - 1].push_back(tmp_dt);
          linear_acceleration_buf_[frame_count_ - 1].push_back(tmp_linear_acceleration);
          angular_velocity_buf_[frame_count_ - 1].push_back(tmp_angular_velocity);
        }

        Vs_[frame_count_ - 1] = Vs_[frame_count_];
        Bas_[frame_count_ - 1] = Bas_[frame_count_];
        Bgs_[frame_count_ - 1] = Bgs_[frame_count_];

        delete pre_integrations_[WINDOW_SIZE];
        pre_integrations_[WINDOW_SIZE] =
          new IntegrationBase{acc_0_, gyr_0_, Bas_[WINDOW_SIZE], Bgs_[WINDOW_SIZE]};

        dt_buf_[WINDOW_SIZE].clear();
        linear_acceleration_buf_[WINDOW_SIZE].clear();
        angular_velocity_buf_[WINDOW_SIZE].clear();
      }
      slideWindowNew();
    }
  }
}

void Estimator::slideWindowNew()
{
  sum_of_front_++;
  f_manager_.removeFront(frame_count_);
}

void Estimator::slideWindowOld()
{
  sum_of_back_++;

  bool shift_depth = solver_flag_ == NON_LINEAR;
  if (shift_depth) {
    Matrix3d R0, R1;
    Vector3d P0, P1;
    R0 = back_R0_ * ric_[0];
    R1 = Rs_[0] * ric_[0];
    P0 = back_P0_ + back_R0_ * tic_[0];
    P1 = Ps_[0] + Rs_[0] * tic_[0];
    f_manager_.removeBackShiftDepth(R0, P0, R1, P1);
  } else
    f_manager_.removeBack();
}


void Estimator::getPoseInWorldFrame(Eigen::Matrix4d & _T)
{
  _T = Eigen::Matrix4d::Identity();
  _T.block<3, 3>(0, 0) = Rs_[frame_count_];
  _T.block<3, 1>(0, 3) = Ps_[frame_count_];
}

void Estimator::getPoseInWorldFrame(int _index, Eigen::Matrix4d & _T)
{
  _T = Eigen::Matrix4d::Identity();
  _T.block<3, 3>(0, 0) = Rs_[_index];
  _T.block<3, 1>(0, 3) = Ps_[_index];
}

void Estimator::predictPtsInNextFrame()
{
  //printf("predict pts in next frame\n");
  if (frame_count_ < 2)
    return;
  // predict next pose. Assume constant velocity motion
  Eigen::Matrix4d curT, prevT, nextT;
  getPoseInWorldFrame(curT);
  getPoseInWorldFrame(frame_count_ - 1, prevT);
  nextT = curT * (prevT.inverse() * curT);
  map<int, Eigen::Vector3d> predictPts;

  for (auto & it_per_id : f_manager_.feature_) {
    if (it_per_id.estimated_depth_ > 0) {
      int firstIndex = it_per_id.start_frame_;
      int lastIndex = it_per_id.start_frame_ + it_per_id.feature_per_frame_.size() - 1;
      //printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
      if ((int)it_per_id.feature_per_frame_.size() >= 2 && lastIndex == frame_count_) {
        double depth = it_per_id.estimated_depth_;
        Vector3d pts_j = ric_[0] * (depth * it_per_id.feature_per_frame_[0].point_) + tic_[0];
        Vector3d pts_w = Rs_[firstIndex] * pts_j + Ps_[firstIndex];
        Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
        Vector3d pts_cam = ric_[0].transpose() * (pts_local - tic_[0]);
        int ptsIndex = it_per_id.feature_id_;
        predictPts[ptsIndex] = pts_cam;
      }
    }
  }
  featureTracker_.setPrediction(predictPts);
  //printf("estimator output %d predict pts\n",(int)predictPts.size());
}

double Estimator::reprojectionError(
  Matrix3d & _Ri, Vector3d & _Pi, Matrix3d & _rici, Vector3d & _tici, Matrix3d & _Rj, Vector3d & _Pj,
  Matrix3d & _ricj, Vector3d & _ticj, double _depth, Vector3d & _uvi, Vector3d & _uvj)
{
  Vector3d pts_w = _Ri * (_rici * (_depth * _uvi) + _tici) + _Pi;
  Vector3d pts_cj = _ricj.transpose() * (_Rj.transpose() * (pts_w - _Pj) - _ticj);
  Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - _uvj.head<2>();
  double rx = residual.x();
  double ry = residual.y();
  return sqrt(rx * rx + ry * ry);
}

void Estimator::outliersRejection(set<int> & _removeIndex)
{
  //return;
  int feature_index = -1;
  for (auto & it_per_id : f_manager_.feature_) {
    double err = 0;
    int errCnt = 0;
    it_per_id.used_num_ = it_per_id.feature_per_frame_.size();
    if (it_per_id.used_num_ < 4)
      continue;
    feature_index++;
    int imu_i = it_per_id.start_frame_, imu_j = imu_i - 1;
    Vector3d pts_i = it_per_id.feature_per_frame_[0].point_;
    double depth = it_per_id.estimated_depth_;
    for (auto & it_per_frame : it_per_id.feature_per_frame_) {
      imu_j++;
      if (imu_i != imu_j) {
        Vector3d pts_j = it_per_frame.point_;
        double tmp_error = reprojectionError(
          Rs_[imu_i], Ps_[imu_i], ric_[0], tic_[0], Rs_[imu_j], Ps_[imu_j], ric_[0], tic_[0], depth, pts_i,
          pts_j);
        err += tmp_error;
        errCnt++;
        //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
      }
      // need to rewrite projecton factor.........
      if (STEREO && it_per_frame.is_stereo_) {

        Vector3d pts_j_right = it_per_frame.pointRight_;
        if (imu_i != imu_j) {
          double tmp_error = reprojectionError(
            Rs_[imu_i], Ps_[imu_i], ric_[0], tic_[0], Rs_[imu_j], Ps_[imu_j], ric_[1], tic_[1], depth, pts_i,
            pts_j_right);
          err += tmp_error;
          errCnt++;
          //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
        } else {
          double tmp_error = reprojectionError(
            Rs_[imu_i], Ps_[imu_i], ric_[0], tic_[0], Rs_[imu_j], Ps_[imu_j], ric_[1], tic_[1], depth, pts_i,
            pts_j_right);
          err += tmp_error;
          errCnt++;
          //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
        }
      }
    }
    double ave_err = err / errCnt;
    if (ave_err * FOCAL_LENGTH > 3)
      _removeIndex.insert(it_per_id.feature_id_);
  }
}

bool Estimator::fastPredictIMU(double _t, Eigen::Vector3d _linear_acceleration, Eigen::Vector3d _angular_velocity)
{
  const double dt = _t - latest_time_;
  if (!std::isfinite(dt) || !(dt > 0.0) || !_linear_acceleration.allFinite() || !_angular_velocity.allFinite())
    return false;
  if (latest_schur_active_) {
    // An IMU outage is distinct from a camera outage. Do not label an
    // unpropagated/invalid estimate with a new timestamp after a clock jump.
    if (dt > 1.0) {
      latest_state_valid_ = false;
      return false;
    }
    latest_schur_state_.propagate(_linear_acceleration, _angular_velocity, dt, latest_schur_noise_scale_);
    const auto & imu = latest_schur_state_.imu();
    latest_P_ = imu.p_;
    latest_Q_ = Eigen::Quaterniond(imu.q_[0], imu.q_[1], imu.q_[2], imu.q_[3]);
    latest_V_ = imu.v_;
    latest_Ba_ = imu.ba_;
    latest_Bg_ = imu.bg_;
  } else {
    Eigen::Vector3d un_acc_0 = latest_Q_ * (latest_acc_0_ - latest_Ba_) - latest_gravity_;
    Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0_ + _angular_velocity) - latest_Bg_;
    latest_Q_ = latest_Q_ * Utility::deltaQ(un_gyr * dt);
    Eigen::Vector3d un_acc_1 = latest_Q_ * (_linear_acceleration - latest_Ba_) - latest_gravity_;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    latest_P_ += dt * latest_V_ + 0.5 * dt * dt * un_acc;
    latest_V_ += dt * un_acc;
  }
  latest_time_ = _t;
  latest_acc_0_ = _linear_acceleration;
  latest_gyr_0_ = _angular_velocity;
  return true;
}

void Estimator::updateLatestStates()
{
  std::lock_guard<std::mutex> propagate_lock(mPropagate_);
  latest_state_valid_ = solver_flag_ == NON_LINEAR && !restart_requested_ && !suppress_state_publication_;
  latest_schur_active_ = latest_state_valid_ && schur_ekf_active_;
  latest_time_ = Headers_[frame_count_] + td_;
  latest_P_ = Ps_[frame_count_];
  latest_Q_ = Rs_[frame_count_];
  latest_V_ = Vs_[frame_count_];
  latest_Ba_ = Bas_[frame_count_];
  latest_Bg_ = Bgs_[frame_count_];
  latest_acc_0_ = acc_0_;
  latest_gyr_0_ = gyr_0_;
  latest_gravity_ = g_;
  if (latest_schur_active_) {
    latest_schur_state_.copyLiveFrom(schur_state_);
    latest_time_ = latest_schur_state_.timestamp();
    const auto & imu = latest_schur_state_.imu();
    latest_P_ = imu.p_;
    latest_Q_ = Eigen::Quaterniond(imu.q_[0], imu.q_[1], imu.q_[2], imu.q_[3]);
    latest_V_ = imu.v_;
    latest_Ba_ = imu.ba_;
    latest_Bg_ = imu.bg_;
    latest_schur_noise_scale_ = schur_hold_.active() ? 1.0 : schur_starvation_monitor_.noiseScale();
  }
  if (!latest_state_valid_)
    return;
  queue<pair<double, Eigen::Vector3d>> tmp_accBuf, tmp_gyrBuf;
  {
    std::lock_guard<std::mutex> buffer_lock(mBuf_);
    tmp_accBuf = accBuf_;
    tmp_gyrBuf = gyrBuf_;
  }
  while (!tmp_accBuf.empty() && latest_state_valid_) {
    // Older/equal samples are deliberately skipped, including the endpoint
    // retained by getIMUInterval. Only the tail after the corrected state
    // is replayed; the image-time EKF is never advanced by this predictor.
    fastPredictIMU(tmp_accBuf.front().first, tmp_accBuf.front().second, tmp_gyrBuf.front().second);
    tmp_accBuf.pop();
    tmp_gyrBuf.pop();
  }
}
