/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <atomic>
#include <optional>

#pragma once

#include "../utility/runtime_utils.h"
#include <ceres/ceres.h>
#include <condition_variable>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <mutex>
#include <opencv2/core/eigen.hpp>
#include <queue>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/header.hpp>
#include <thread>
#include <unordered_map>

#include "../backend/ekf_state.h"
#include "../backend/feature_history.h"
#include "../backend/landing_static.h"
#include "../backend/schur_update.h"
#include "../backend/schur_visualization.h"
#include "../factor/imu_factor.h"
#include "../factor/marginalization_factor.h"
#include "../factor/pose_local_parameterization.h"
#include "../factor/projectionOneFrameTwoCamFactor.h"
#include "../factor/projectionTwoFrameOneCamFactor.h"
#include "../factor/projectionTwoFrameTwoCamFactor.h"
#include "../featureTracker/feature_tracker.h"
#include "../initial/initial_alignment.h"
#include "../initial/initial_ex_rotation.h"
#include "../initial/initial_sfm.h"
#include "../initial/solve_5pts.h"
#include "../utility/tic_toc.h"
#include "../utility/utility.h"
#include "feature_manager.h"
#include "parameters.h"


class Estimator {
  std::atomic<bool> stopping_{false};
  std::condition_variable measurements_ready_;
  vins::FrameRateGate backend_gate_{BACKEND_FREQ, true};
  size_t dropped_features_ = 0;

public:
  Estimator();
  ~Estimator();
  void setParameter();

  // interface
  void initFirstPose(Eigen::Vector3d _p, Eigen::Matrix3d _r);
  void inputIMU(double _t, const Vector3d & _linearAcceleration, const Vector3d & _angularVelocity);
  void inputFeature(double _t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> & _featureFrame);
  // Exact ROS stamp overload used by the node so keyframe_pose/keyframe_point
  // can match the source image without an epoch-double round trip.
  void inputFeature(
    double _t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> & _featureFrame,
    const builtin_interfaces::msg::Time & _stamp);
  void inputImage(
    double _t, const cv::Mat & _img, const cv::Mat & _img1 = cv::Mat(),
    const std::optional<builtin_interfaces::msg::Time> & _stamp = std::nullopt);
  void
  processIMU(double _t, double _dt, const Vector3d & _linear_acceleration, const Vector3d & _angular_velocity);
  void processImage(
    const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> & _image, const double _header,
    const std::optional<builtin_interfaces::msg::Time> & _stamp = std::nullopt);
  void processMeasurements();
  void changeSensorType(int _use_imu, int _use_stereo);
  // Solver entry point used by processImage(). Tests may override this to
  // exercise failure/recovery paths without running a real Ceres solve.
  virtual bool runOptimization();
  bool restartPending() const { return restart_requested_; }
  bool statePublicationSuppressed() const { return suppress_state_publication_; }
  bool restartAfterSolverFailure();
  // ARM-oriented Schur-complement EKF backend (opt-in through the
  // `backend_type` parameter). Returns true once the filter is running.
  bool schurBackendActive() const { return schur_ekf_active_; }
  const vins::backend::SchurVisualizationSnapshot_s & schurVisualizationSnapshot() const
  {
    return schur_visual_snapshot_;
  }
  const builtin_interfaces::msg::Time & schurVisualizationStamp() const { return schur_visualization_stamp_; }
  void maintainInitializationHistoryAfterFailure();
  void trimInitializationHistoryToActiveWindow();

  // internal
  void clearState();
  // Virtual so recovery tests can enter the real processImage() initialization
  // branch without fabricating a full visual SFM problem.
  virtual bool initialStructure();
  bool visualInitialAlign();
  bool relativePose(Matrix3d & _relative_R, Vector3d & _relative_T, int & _l);
  void slideWindow();
  void slideWindowNew();
  void slideWindowOld();
  bool optimization();
  void vector2double();
  void double2vector();
  bool failureDetection();
  bool getIMUInterval(
    double _t0, double _t1, vector<pair<double, Eigen::Vector3d>> & _accVector,
    vector<pair<double, Eigen::Vector3d>> & _gyrVector);
  void getPoseInWorldFrame(Eigen::Matrix4d & _T);
  void getPoseInWorldFrame(int _index, Eigen::Matrix4d & _T);
  void predictPtsInNextFrame();
  void outliersRejection(set<int> & _removeIndex);
  double reprojectionError(
    Matrix3d & _Ri, Vector3d & _Pi, Matrix3d & _rici, Vector3d & _tici, Matrix3d & _Rj, Vector3d & _Pj,
    Matrix3d & _ricj, Vector3d & _ticj, double _depth, Vector3d & _uvi, Vector3d & _uvj);
  void updateLatestStates();
  // --- Schur-complement EKF backend (see docs/BACKEND_SCHUR_EKF.md) ---
  // Seeds the filter from the current Ceres-initialised window state.
  void initSchurBackend();
  // Seeds and activates the filter once Ceres initialisation has produced a
  // usable window state. No-op when the backend is not selected.
  void
  activateSchurBackend(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> & _image, double _header);
  // Propagates the filter by one IMU sample (no-op until initialised).
  void schurPropagate(double _dt, const Eigen::Vector3d & _acc, const Eigen::Vector3d & _gyr);
  // Runs the visual EKF update for one image frame. Returns false when the
  // filter cannot be used for this frame and the caller must fall back.
  bool schurProcessFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> & _image, double _header);
  // P5 soft starvation restart: keep the IMU/EKF nominal state and only drop
  // the clone window and covariance (P re-inflated from init_*_sigma). Called
  // from schurProcessFrame() instead of scheduleControlledRestart() when
  // SCHUR_STARVATION_SOFT_RESTART is enabled.
  void schurSoftRestart(double _frame_time, double _gap_seconds);
  // P7 hold policy ("不重启、只保持"). enter: stop accepting visual updates and
  // hard-delete the stale clone blocks/history while keeping the live state and
  // covariance untouched (no clearState(), no reset(), no P re-inflation).
  // recover: leave hold, re-seed the history and re-baseline the starvation
  // monitor so the clone window is rebuilt frame by frame. Neither method
  // restarts the estimator; processed image frames still write back and publish.
  // The image-time window is propagated in processMeasurements(); inputIMU()
  // independently propagates the live marginal used by imu_propagate.
  void schurHoldEnter(double _frame_time, double _gap_seconds);
  void schurHoldRecover(double _frame_time, int _features);
  // Copies the filter state back into the Ceres/visualisation window arrays.
  void schurWriteBackWindow();
  bool schurStateFinite() const;
  void schurReportStageTiming(double _update_ms);
  // Periodic (every kSchurStarvationReportFrames frames) summary of the
  // starvation policy: restart-trigger count, deep-escalation count,
  // cooldown-suppressed trigger count, the currently active Q multiplier, the
  // longest observed no-update interval, the current starvation duration and
  // the active thresholds.
  void schurReportStarvation(double _frame_time) const;
  bool fastPredictIMU(double _t, Eigen::Vector3d _linear_acceleration, Eigen::Vector3d _angular_velocity);
  bool imuAvailable(double _t);
  void initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> & _accVector);

  // Keep the existing enum representation for interface compatibility.
  // NOLINTNEXTLINE(performance-enum-size)
  enum SolverFlag_e : unsigned int { INITIAL, NON_LINEAR };

  // Keep the existing enum representation for interface compatibility.
  // NOLINTNEXTLINE(performance-enum-size)
  enum MarginalizationFlag_e : unsigned int { MARGIN_OLD = 0, MARGIN_SECOND_NEW = 1 };

  std::mutex mProcess_;
  std::mutex mBuf_;
  std::mutex mPropagate_;
  queue<pair<double, Eigen::Vector3d>> accBuf_;
  queue<pair<double, Eigen::Vector3d>> gyrBuf_;
  using FeatureFrame = map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>;
  struct FeatureFrameInput_s {
    double time_ = 0.0;
    std::optional<builtin_interfaces::msg::Time> stamp_;
    FeatureFrame features_;
  };
  queue<FeatureFrameInput_s> featureBuf_;
  double prevTime_, curTime_;
  bool openExEstimation_;

  std::thread trackThread_;
  std::thread processThread_;

  FeatureTracker featureTracker_;

  SolverFlag_e solver_flag_;
  MarginalizationFlag_e marginalization_flag_;
  Vector3d g_;

  Matrix3d ric_[2];
  Vector3d tic_[2];

  Vector3d Ps_[(WINDOW_SIZE + 1)];
  Vector3d Vs_[(WINDOW_SIZE + 1)];
  Matrix3d Rs_[(WINDOW_SIZE + 1)];
  Vector3d Bas_[(WINDOW_SIZE + 1)];
  Vector3d Bgs_[(WINDOW_SIZE + 1)];
  double td_;

  Matrix3d back_R0_, last_R_, last_R0_;
  Vector3d back_P0_, last_P_, last_P0_;
  double Headers_[(WINDOW_SIZE + 1)];

  IntegrationBase * pre_integrations_[(WINDOW_SIZE + 1)]{};
  Vector3d acc_0_, gyr_0_;

  vector<double> dt_buf_[(WINDOW_SIZE + 1)];
  vector<Vector3d> linear_acceleration_buf_[(WINDOW_SIZE + 1)];
  vector<Vector3d> angular_velocity_buf_[(WINDOW_SIZE + 1)];

  int frame_count_;
  int sum_of_outlier_, sum_of_back_, sum_of_front_, sum_of_invalid_;
  int inputImageCnt_;

  FeatureManager f_manager_;
  MotionEstimator m_estimator_;
  InitialEXRotation initial_ex_rotation_;

  bool first_imu_;
  bool is_valid_, is_key_;
  bool failure_occur_;

  vector<Vector3d> point_cloud_;
  vector<Vector3d> margin_cloud_;
  vector<Vector3d> key_poses_;
  double initial_timestamp_;


  double para_Pose_[WINDOW_SIZE + 1][SIZE_POSE];
  double para_SpeedBias_[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
  double para_Feature_[NUM_OF_F][SIZE_FEATURE];
  double para_Ex_Pose_[2][SIZE_POSE];
  double para_Retrive_Pose_[SIZE_POSE];
  double para_Td_[1][1];
  double para_Tr_[1][1];

  int loop_window_index_;

  MarginalizationInfo * last_marginalization_info_ = nullptr;
  vector<double *> last_marginalization_parameter_blocks_;

  map<double, ImageFrame> all_image_frame_;
  IntegrationBase * tmp_pre_integration_ = nullptr;

  Eigen::Vector3d initP_;
  Eigen::Matrix3d initR_;

  double latest_time_;
  Eigen::Vector3d latest_P_, latest_V_, latest_Ba_, latest_Bg_, latest_acc_0_, latest_gyr_0_;
  Eigen::Quaterniond latest_Q_;
  // Protected by mPropagate, including readiness (inputIMU never reads the
  // image-thread solver_flag). Lock order: mProcess -> mPropagate -> mBuf.
  vins::backend::EkfState latest_schur_state_;
  bool latest_state_valid_ = false;
  bool latest_schur_active_ = false;
  double latest_imu_dt_ = 0.0;
  double latest_input_time_ = -1.0;
  double latest_schur_noise_scale_ = 1.0;
  Eigen::Vector3d latest_gravity_ = Eigen::Vector3d::Zero();

  bool initFirstPoseFlag_;
  bool initThreadFlag_;

  // --- Schur-complement EKF backend state ---
  vins::backend::EkfState schur_state_;
  vins::backend::SchurUpdater schur_updater_;
  vins::backend::FeatureHistory schur_history_;
  std::vector<vins::backend::CameraExtrinsics_s> schur_cameras_;
  vins::backend::SchurVisualizationSnapshot_s schur_visual_snapshot_;
  vins::backend::MarginPublicationHistory schur_margin_history_;
  builtin_interfaces::msg::Time schur_visualization_stamp_{};
  std::optional<builtin_interfaces::msg::Time> current_feature_stamp_;
  bool schur_ekf_active_ = false;  // filter seeded and driving the estimate
  int schur_frame_index_ = 0;      // monotonic image-frame counter for the filter
  int schur_oldest_frame_ = 0;     // oldest frame index retained in the history
  int schur_update_count_ = 0;
  double schur_last_imu_time_ = -1.0;
  double schur_last_imu_dt_ = 0.0;
  double schur_stage_total_ms_ = 0.0;
  double schur_features_total_ = 0.0;
  double schur_rejected_total_ = 0.0;
  double schur_lost_total_ = 0.0;
  double schur_marginalized_total_ = 0.0;
  int schur_consecutive_failures_ = 0;  // frames skipped without a visual update
  int schur_skipped_frames_ = 0;        // diagnostic counter (never reset)
  // Sensor time of the pending (unrecovered) P5 soft restart, or < 0 when no
  // soft restart is waiting for its first applied visual update. Kept at the
  // *first* soft restart of an unrecovered episode so repeated soft restarts
  // cannot postpone the escalation forever; cleared on a successful update,
  // on a full restart, and in initSchurBackend().
  double schur_soft_restart_time_ = -1.0;
  // Time-based starvation policy (P1 time-based restart + P3 process-noise
  // inflation). Owns the sensor-time baseline, the longest-observed-gap and the
  // trigger counters so they are reset atomically in initSchurBackend(); a
  // backwards timestamp re-baselines instead of triggering a restart.
  vins::backend::StarvationMonitor schur_starvation_monitor_;
  // Conservative landing static constraint. It is armed by a recent impact
  // and only updates while visual/IMU/speed evidence all agree.
  vins::backend::LandingStaticDetector schur_landing_static_;
  int schur_landing_static_updates_ = 0;
  int schur_landing_static_failures_ = 0;
  // P7 hold policy state machine and rebuild diagnostics.
  vins::backend::StarvationHold schur_hold_;
  int schur_rebuild_frames_ = 0;       // frames since the last hold exit
  int schur_rebuild_last_frames_ = 0;  // frames the last rebuild needed
  bool schur_rebuild_pending_ = false;

private:
  void scheduleControlledRestart()
  {
    restart_requested_ = true;
    std::lock_guard<std::mutex> lock(mPropagate_);
    latest_state_valid_ = false;
  }
  bool restart_requested_ = false;
  bool suppress_state_publication_ = false;
};
