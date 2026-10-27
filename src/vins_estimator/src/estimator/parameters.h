/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <vins_common/ros_utils.hpp>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "../utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <map>

using namespace std;

const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
const int NUM_OF_F = 1000;
//#define UNIT_SPHERE_ERROR

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string OUTPUT_FOLDER;
extern std::string IMU_TOPIC;
extern double TD;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;
extern int ROW, COL;
extern int NUM_OF_CAM;
extern int STEREO;
extern int USE_IMU;
extern int MULTIPLE_THREAD;
extern double BACKEND_FREQ;
extern int MAX_FEATURE_QUEUE;
// pts_gt for debug purpose;
extern map<int, Eigen::Vector3d> pts_gt;

extern std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int MIN_DIST;
extern std::string FEATURE_DETECTOR;
extern int FAST_THRESHOLD;
extern int GRID_X, GRID_Y;
extern double DETECTION_MIN_FEATURE_RATIO;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern int FLOW_BACK;

extern int MARGINALIZATION_THREADS;
extern std::string SOLVER_TIMEOUT_POLICY;
extern double SOLVER_TIMEOUT_RETRY_MULTIPLIER;
extern bool STAGE_TIMING;
extern int STAGE_TIMING_INTERVAL;
extern int RANSAC_OUTLIER_REJECTION;

// Backend selection and ARM-oriented Schur-complement EKF parameters.
// BACKEND_TYPE is "ceres" (default, unchanged sliding-window NLS) or
// "schur_ekf" (opt-in Schur-complement EKF described in
// docs/BACKEND_SCHUR_EKF.md).
extern std::string BACKEND_TYPE;
extern int SCHUR_WINDOW_SIZE;
extern int SCHUR_WARMUP_FRAMES;
extern int SCHUR_MAX_FEATURES;
extern int SCHUR_MIN_OBSERVATIONS;
extern int SCHUR_MIN_FEATURES;
extern int SCHUR_MIN_UPDATE_OBSERVATIONS;
extern double SCHUR_PIXEL_NOISE;
extern double SCHUR_CHI2_MULTIPLIER;
extern double SCHUR_TRIANGULATION_ITERATIONS;
extern double SCHUR_INFO_RANK_RTOL;
extern double SCHUR_INFO_INDEFINITE_TOL;
extern double SCHUR_POINT_BLOCK_RTOL;
extern double SCHUR_CONDITION_LIMIT;
// Frame-count backstop for a controlled restart.  This is only a *secondary*
// guard (default 60): the time criterion below is the primary one.  60 frames
// is 6 s of pure IMU propagation at the ~10 Hz the ARM backend actually runs
// at, and 3 s at 20 Hz on x86, so the time criterion normally fires first.
// Both guards are suppressed while SCHUR_RESTART_COOLDOWN_SECONDS is active.
extern int SCHUR_MAX_CONSECUTIVE_FAILURES;
// Wall/sensor-time starvation criterion (primary restart guard): restart when
// this many seconds have elapsed (in the frame header+td domain) since the last
// *applied* visual update.  <= 0 disables the time criterion and leaves the
// frame-count one (SCHUR_MAX_CONSECUTIVE_FAILURES) as the only guard.  Default
// 3.0 s: on ARM the starvation stretches mostly self-heal within 1-2 s, so a
// 1 s criterion restarts far too eagerly (and every restart costs a full Ceres
// re-initialisation, i.e. 10-44 s without odometry on ARM).
extern double SCHUR_MAX_NO_UPDATE_SECONDS;
// Visual-starvation process-noise inflation: once this many seconds have
// elapsed since the last applied visual update, each subsequent IMU
// propagation scales its process-noise covariance Q by
// SCHUR_STARVATION_NOISE_SCALE until the next successful update.  <= 0 disables
// the inflation (scale is never applied).
extern double SCHUR_STARVATION_SECONDS;
// Multiplier applied to Q while starved.  Must be >= 1.0 (1.0 is a no-op).
extern double SCHUR_STARVATION_NOISE_SCALE;
// Deep starvation: once this many seconds have elapsed since the last applied
// visual update (and the base starvation stage is already active), the Q
// multiplier is escalated from SCHUR_STARVATION_NOISE_SCALE to
// SCHUR_STARVATION_DEEP_NOISE_SCALE.  <= 0 disables the escalation.
extern double SCHUR_STARVATION_DEEP_SECONDS;
// Q multiplier applied during deep starvation.  Must be finite and >= 1.0.
extern double SCHUR_STARVATION_DEEP_NOISE_SCALE;
// P6 starvation-adaptive weak update ("accept more, weight less").  When true,
// a frame whose sensor-time gap since the last applied visual update exceeds
// SCHUR_STARVATION_SECONDS relaxes the chi2 gate by weakUpdateScale(...) and
// down-weights the accepted observations by the same factor, so the filter can
// recover from a gate collapse without a full controlled restart.  Default
// false preserves the previous behaviour exactly.
extern bool SCHUR_STARVATION_WEAK_UPDATE;
// Upper bound of the weak-update scale (>= 1.0; 1.0 is equivalent to disabling
// the feature).  The ramp reuses SCHUR_STARVATION_SECONDS /
// SCHUR_STARVATION_DEEP_SECONDS and saturates at this value.
extern double SCHUR_WEAK_UPDATE_MAX_SCALE;
// Restart cooldown, in sensor time: at most one controlled restart per this
// many seconds.  While the cooldown is active both restart guards are
// suppressed (they are counted, not acted on), which turns a restart storm into
// a bounded number of restarts.  Must be finite and >= 0; 0 disables it.
extern double SCHUR_RESTART_COOLDOWN_SECONDS;
// P5 soft starvation restart.  When true, a starvation-triggered restart (the
// time criterion above) performs a *soft* restart instead of the full
// clearState() + Ceres re-initialisation: the EKF keeps its current IMU nominal
// state (q/p/v/ba/bg) and only drops the clone window and covariance (P is
// re-inflated from the init_*_sigma values), so odometry keeps flowing through
// the starvation stretch.  Default false preserves the previous behaviour.
extern bool SCHUR_STARVATION_SOFT_RESTART;
// P5 soft-restart escalation backstop: if a soft restart is not followed by an
// applied visual update within this many seconds of sensor time, escalate once
// to a full controlled restart (clearState()).  Must be finite; <= 0 disables
// the escalation.  Only meaningful when SCHUR_STARVATION_SOFT_RESTART is true.
extern double SCHUR_SOFT_RESTART_MAX_SECONDS;
extern double SCHUR_INIT_POS_SIGMA;
extern double SCHUR_INIT_VEL_SIGMA;
extern double SCHUR_INIT_ROT_SIGMA;
extern double SCHUR_INIT_BA_SIGMA;
extern double SCHUR_INIT_BG_SIGMA;
// Continuous-time IMU noise densities of the EKF.  These are NOT acc_n/gyr_n:
// VINS applies acc_n/gyr_n through IntegrationBase's V matrix, i.e. as a
// per-sample noise scaled by 0.5*dt, which is a different convention from the
// spectral densities an EKF needs (sigma^2 [unit^2/Hz]).  The defaults below
// are the EuRoC MAV values used by open_vins (config/euroc_mav/kalibr_imu_chain.yaml).
extern double SCHUR_ACC_NOISE;
extern double SCHUR_GYR_NOISE;
extern double SCHUR_ACC_BIAS_NOISE;
extern double SCHUR_GYR_BIAS_NOISE;

void readParameters(std::string config_file, const std::string &output_path = "");

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
