/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <atomic>

#pragma once
 
#include <thread>
#include <mutex>
#include <condition_variable>
#include "../utility/runtime_utils.h"
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/float32.hpp>
#include <ceres/ceres.h>
#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "parameters.h"
#include "feature_manager.h"
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../initial/solve_5pts.h"
#include "../initial/initial_sfm.h"
#include "../initial/initial_alignment.h"
#include "../initial/initial_ex_rotation.h"
#include "../factor/imu_factor.h"
#include "../factor/pose_local_parameterization.h"
#include "../factor/marginalization_factor.h"
#include "../factor/projectionTwoFrameOneCamFactor.h"
#include "../factor/projectionTwoFrameTwoCamFactor.h"
#include "../factor/projectionOneFrameTwoCamFactor.h"
#include "../featureTracker/feature_tracker.h"
#include "../backend/ekf_state.h"
#include "../backend/schur_update.h"
#include "../backend/feature_history.h"


class Estimator
{
    std::atomic<bool> stopping_{false};
    std::condition_variable measurements_ready_;
    vins::FrameRateGate backend_gate_{BACKEND_FREQ, true};
    size_t dropped_features_ = 0;
  public:
    Estimator();
    ~Estimator();
    void setParameter();

    // interface
    void initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r);
    void inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity);
    void inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame);
    void inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    void processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header);
    void processMeasurements();
    void changeSensorType(int use_imu, int use_stereo);
    // Solver entry point used by processImage(). Tests may override this to
    // exercise failure/recovery paths without running a real Ceres solve.
    virtual bool runOptimization();
    bool restartPending() const { return restart_requested_; }
    bool statePublicationSuppressed() const { return suppress_state_publication_; }
    bool restartAfterSolverFailure();
    // ARM-oriented Schur-complement EKF backend (opt-in through the
    // `backend_type` parameter). Returns true once the filter is running.
    bool schurBackendActive() const { return schur_ekf_active_; }
    void maintainInitializationHistoryAfterFailure();
    void trimInitializationHistoryToActiveWindow();

    // internal
    void clearState();
    // Virtual so recovery tests can enter the real processImage() initialization
    // branch without fabricating a full visual SFM problem.
    virtual bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void slideWindowNew();
    void slideWindowOld();
    bool optimization();
    void vector2double();
    void double2vector();
    bool failureDetection();
    bool getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                              vector<pair<double, Eigen::Vector3d>> &gyrVector);
    void getPoseInWorldFrame(Eigen::Matrix4d &T);
    void getPoseInWorldFrame(int index, Eigen::Matrix4d &T);
    void predictPtsInNextFrame();
    void outliersRejection(set<int> &removeIndex);
    double reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                     Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                     double depth, Vector3d &uvi, Vector3d &uvj);
    void updateLatestStates();
    // --- Schur-complement EKF backend (see docs/BACKEND_SCHUR_EKF.md) ---
    // Seeds the filter from the current Ceres-initialised window state.
    void initSchurBackend();
    // Seeds and activates the filter once Ceres initialisation has produced a
    // usable window state. No-op when the backend is not selected.
    void activateSchurBackend(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
                              double header);
    // Propagates the filter by one IMU sample (no-op until initialised).
    void schurPropagate(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr);
    // Runs the visual EKF update for one image frame. Returns false when the
    // filter cannot be used for this frame and the caller must fall back.
    bool schurProcessFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
                           double header);
    // P5 soft starvation restart: keep the IMU/EKF nominal state and only drop
    // the clone window and covariance (P re-inflated from init_*_sigma). Called
    // from schurProcessFrame() instead of scheduleControlledRestart() when
    // SCHUR_STARVATION_SOFT_RESTART is enabled.
    void schurSoftRestart(double frame_time, double gap_seconds);
    // P7 hold policy ("不重启、只保持"). enter: stop accepting visual updates and
    // hard-delete the stale clone blocks/history while keeping the live state and
    // covariance untouched (no clearState(), no reset(), no P re-inflation).
    // recover: leave hold, re-seed the history and re-baseline the starvation
    // monitor so the clone window is rebuilt frame by frame. Neither method
    // restarts the estimator; processed image frames still write back and publish.
    // The image-time window is propagated in processMeasurements(); inputIMU()
    // independently propagates the live marginal used by imu_propagate.
    void schurHoldEnter(double frame_time, double gap_seconds);
    void schurHoldRecover(double frame_time, int observations);
    // Copies the filter state back into the Ceres/visualisation window arrays.
    void schurWriteBackWindow();
    bool schurStateFinite() const;
    void schurReportStageTiming(double update_ms);
    // Periodic (every kSchurStarvationReportFrames frames) summary of the
    // starvation policy: restart-trigger count, deep-escalation count,
    // cooldown-suppressed trigger count, the currently active Q multiplier, the
    // longest observed no-update interval, the current starvation duration and
    // the active thresholds.
    void schurReportStarvation(double frame_time) const;
    bool fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity);
    bool IMUAvailable(double t);
    void initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector);

    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    std::mutex mProcess;
    std::mutex mBuf;
    std::mutex mPropagate;
    queue<pair<double, Eigen::Vector3d>> accBuf;
    queue<pair<double, Eigen::Vector3d>> gyrBuf;
    queue<pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > > featureBuf;
    double prevTime, curTime;
    bool openExEstimation;

    std::thread trackThread;
    std::thread processThread;

    FeatureTracker featureTracker;

    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    Vector3d g;

    Matrix3d ric[2];
    Vector3d tic[2];

    Vector3d        Ps[(WINDOW_SIZE + 1)];
    Vector3d        Vs[(WINDOW_SIZE + 1)];
    Matrix3d        Rs[(WINDOW_SIZE + 1)];
    Vector3d        Bas[(WINDOW_SIZE + 1)];
    Vector3d        Bgs[(WINDOW_SIZE + 1)];
    double td;

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    double Headers[(WINDOW_SIZE + 1)];

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)]{};
    Vector3d acc_0, gyr_0;

    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;
    int inputImageCnt;

    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;


    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[2][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];

    int loop_window_index;

    MarginalizationInfo *last_marginalization_info = nullptr;
    vector<double *> last_marginalization_parameter_blocks;

    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration = nullptr;

    Eigen::Vector3d initP;
    Eigen::Matrix3d initR;

    double latest_time;
    Eigen::Vector3d latest_P, latest_V, latest_Ba, latest_Bg, latest_acc_0, latest_gyr_0;
    Eigen::Quaterniond latest_Q;
    // Protected by mPropagate, including readiness (inputIMU never reads the
    // image-thread solver_flag). Lock order: mProcess -> mPropagate -> mBuf.
    vins::backend::EkfState latest_schur_state_;
    bool latest_state_valid_ = false;
    bool latest_schur_active_ = false;
    double latest_imu_dt_ = 0.0;
    double latest_input_time_ = -1.0;
    double latest_schur_noise_scale_ = 1.0;
    Eigen::Vector3d latest_gravity_ = Eigen::Vector3d::Zero();

    bool initFirstPoseFlag;
    bool initThreadFlag;

    // --- Schur-complement EKF backend state ---
    vins::backend::EkfState schur_state_;
    vins::backend::SchurUpdater schur_updater_;
    vins::backend::FeatureHistory schur_history_;
    std::vector<vins::backend::CameraExtrinsics> schur_cameras_;
    bool schur_ekf_active_ = false;   // filter seeded and driving the estimate
    int schur_frame_index_ = 0;       // monotonic image-frame counter for the filter
    int schur_oldest_frame_ = 0;      // oldest frame index retained in the history
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
    // P7 hold policy state machine and rebuild diagnostics.
    vins::backend::StarvationHold schur_hold_;
    int schur_rebuild_frames_ = 0;       // frames since the last hold exit
    int schur_rebuild_last_frames_ = 0;  // frames the last rebuild needed
    bool schur_rebuild_pending_ = false;

  private:
    void scheduleControlledRestart()
    {
        restart_requested_ = true;
        std::lock_guard<std::mutex> lock(mPropagate);
        latest_state_valid_ = false;
    }
    bool restart_requested_ = false;
    bool suppress_state_publication_ = false;
};
