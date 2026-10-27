/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <vins_common/parameters.hpp>
#include <vins_common/ros_utils.hpp>

#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string IMU_TOPIC;
int ROW, COL;
double TD;
int NUM_OF_CAM;
int STEREO;
int USE_IMU;
int MULTIPLE_THREAD;
double BACKEND_FREQ = 10.0;
int MAX_FEATURE_QUEUE = 10;
int MARGINALIZATION_THREADS = 4;
std::string SOLVER_TIMEOUT_POLICY = "warn";
double SOLVER_TIMEOUT_RETRY_MULTIPLIER = 2.0;
bool STAGE_TIMING = false;
int STAGE_TIMING_INTERVAL = 100;
int RANSAC_OUTLIER_REJECTION = 0;
std::string BACKEND_TYPE = "ceres";
int SCHUR_WINDOW_SIZE = 5;
int SCHUR_WARMUP_FRAMES = 0;
int SCHUR_MAX_FEATURES = 120;
int SCHUR_MIN_OBSERVATIONS = 2;
int SCHUR_MIN_FEATURES = 12;
int SCHUR_MIN_UPDATE_OBSERVATIONS = 24;
double SCHUR_PIXEL_NOISE = 1.5;
double SCHUR_CHI2_MULTIPLIER = 5.0;
double SCHUR_TRIANGULATION_ITERATIONS = 5.0;
double SCHUR_INFO_RANK_RTOL = 1e-10;
double SCHUR_INFO_INDEFINITE_TOL = 1e-6;
double SCHUR_POINT_BLOCK_RTOL = 1e-8;
double SCHUR_CONDITION_LIMIT = 1e6;
int SCHUR_MAX_CONSECUTIVE_FAILURES = 60;
double SCHUR_MAX_NO_UPDATE_SECONDS = 3.0;
double SCHUR_STARVATION_SECONDS = 0.25;
double SCHUR_STARVATION_NOISE_SCALE = 3.0;
double SCHUR_STARVATION_DEEP_SECONDS = 1.0;
double SCHUR_STARVATION_DEEP_NOISE_SCALE = 10.0;
bool SCHUR_STARVATION_WEAK_UPDATE = false;
double SCHUR_WEAK_UPDATE_MAX_SCALE = 10.0;
double SCHUR_RESTART_COOLDOWN_SECONDS = 8.0;
bool SCHUR_STARVATION_SOFT_RESTART = false;
double SCHUR_SOFT_RESTART_MAX_SECONDS = 10.0;
bool SCHUR_STARVATION_HOLD = false;
int SCHUR_HOLD_RECOVERY_FRAMES = 2;
double SCHUR_INIT_POS_SIGMA = 0.05;
double SCHUR_INIT_VEL_SIGMA = 0.05;
double SCHUR_INIT_ROT_SIGMA = 0.01;
double SCHUR_INIT_BA_SIGMA = 0.05;
double SCHUR_INIT_BG_SIGMA = 0.005;
double SCHUR_ACC_NOISE = 2.0e-3;
double SCHUR_GYR_NOISE = 1.6968e-4;
double SCHUR_ACC_BIAS_NOISE = 3.0e-3;
double SCHUR_GYR_BIAS_NOISE = 1.9393e-5;
map<int, Eigen::Vector3d> pts_gt;
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
std::string FISHEYE_MASK;
std::vector<std::string> CAM_NAMES;
int MAX_CNT;
int MIN_DIST;
std::string FEATURE_DETECTOR;
int FAST_THRESHOLD;
int GRID_X, GRID_Y;
double DETECTION_MIN_FEATURE_RATIO;
double F_THRESHOLD;
int SHOW_TRACK;
int FLOW_BACK;

namespace
{
Eigen::Matrix4d readBodyTCam(const vins::Config &config, const std::string &key)
{
    const std::vector<double> values = config.getDoubleArray(key);
    if (values.size() != 16)
        throw std::runtime_error("Parameter " + key + " must contain 16 values (4x4 row-major)");
    Eigen::Matrix4d T;
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            T(row, col) = values[row * 4 + col];
    return T;
}
}  // namespace

void readParameters(std::string config_file, const std::string &output_path)
{
    const vins::Config config = vins::Config::fromFile(config_file);

    RIC.clear(); TIC.clear(); CAM_NAMES.clear(); STEREO = 0;

    IMAGE0_TOPIC = config.getString("image0_topic");
    IMAGE1_TOPIC = config.getString("image1_topic");
    MAX_CNT = config.getInt("max_cnt");
    MIN_DIST = config.getInt("min_dist");
    FEATURE_DETECTOR = config.getString("feature_detector", "gftt");
    FAST_THRESHOLD = config.getInt("fast_threshold", 20);
    GRID_X = config.getInt("grid_x", 5);
    GRID_Y = config.getInt("grid_y", 5);
    DETECTION_MIN_FEATURE_RATIO = config.getDouble("detection_min_feature_ratio", 0.0);
    F_THRESHOLD = config.getDouble("F_threshold");
    SHOW_TRACK = config.getInt("show_track");
    FLOW_BACK = config.getInt("flow_back");
    if (MAX_CNT < 1 || MIN_DIST < 0)
        throw std::invalid_argument("max_cnt must be positive and min_dist nonnegative");
    if (FEATURE_DETECTOR != "gftt" && FEATURE_DETECTOR != "fast_grid")
        throw std::invalid_argument("feature_detector must be 'gftt' or 'fast_grid'");
    if (FAST_THRESHOLD < 1 || FAST_THRESHOLD > 255)
        throw std::invalid_argument("fast_threshold must be in [1, 255]");
    if (GRID_X < 1 || GRID_Y < 1)
        throw std::invalid_argument("grid_x and grid_y must be positive");
    if (!std::isfinite(DETECTION_MIN_FEATURE_RATIO) ||
        DETECTION_MIN_FEATURE_RATIO < 0.0 || DETECTION_MIN_FEATURE_RATIO > 1.0)
        throw std::invalid_argument("detection_min_feature_ratio must be in [0, 1]");
    RCLCPP_INFO(vins::logger(),
                "Feature tracker: max_cnt=%d min_dist=%d flow_back=%d detector=%s "
                "fast_threshold=%d grid=%dx%d detection_min_feature_ratio=%.3f",
                MAX_CNT, MIN_DIST, FLOW_BACK, FEATURE_DETECTOR.c_str(), FAST_THRESHOLD,
                GRID_X, GRID_Y, DETECTION_MIN_FEATURE_RATIO);

    MULTIPLE_THREAD = config.getInt("multiple_thread");
    BACKEND_FREQ = config.getDouble("freq", 10.0);
    MAX_FEATURE_QUEUE = config.getInt("max_feature_queue_size", 10);
    if (!std::isfinite(BACKEND_FREQ) || BACKEND_FREQ < 0.0 || MAX_FEATURE_QUEUE < 1)
        throw std::invalid_argument("freq must be nonnegative and max_feature_queue_size positive");

    MARGINALIZATION_THREADS = config.getInt("marginalization_threads", 4);
    if (MARGINALIZATION_THREADS < 1 || MARGINALIZATION_THREADS > 8)
        throw std::invalid_argument("marginalization_threads must be in [1, 8]");
    SOLVER_TIMEOUT_POLICY = config.getString("solver_timeout_policy", "warn");
    if (SOLVER_TIMEOUT_POLICY != "warn" && SOLVER_TIMEOUT_POLICY != "retry" &&
        SOLVER_TIMEOUT_POLICY != "revert")
        throw std::invalid_argument("solver_timeout_policy must be 'warn', 'retry' or 'revert'");
    SOLVER_TIMEOUT_RETRY_MULTIPLIER = config.getDouble("solver_timeout_retry_multiplier", 2.0);
    if (!std::isfinite(SOLVER_TIMEOUT_RETRY_MULTIPLIER) || SOLVER_TIMEOUT_RETRY_MULTIPLIER <= 1.0)
        throw std::invalid_argument("solver_timeout_retry_multiplier must be greater than 1");
    STAGE_TIMING = config.getBool("stage_timing", false);
    STAGE_TIMING_INTERVAL = config.getInt("stage_timing_interval", 100);
    if (STAGE_TIMING_INTERVAL < 1)
        throw std::invalid_argument("stage_timing_interval must be positive");
    RANSAC_OUTLIER_REJECTION = config.getInt("ransac_outlier_rejection", 0);
    if (RANSAC_OUTLIER_REJECTION != 0 && RANSAC_OUTLIER_REJECTION != 1)
        throw std::invalid_argument("ransac_outlier_rejection must be 0 or 1");
    BACKEND_TYPE = config.getString("backend_type", "ceres");
    if (BACKEND_TYPE != "ceres" && BACKEND_TYPE != "schur_ekf")
        throw std::invalid_argument("backend_type must be 'ceres' or 'schur_ekf'");
    SCHUR_WINDOW_SIZE = config.getInt("schur_window_size", 5);
    if (SCHUR_WINDOW_SIZE < 2 || SCHUR_WINDOW_SIZE > WINDOW_SIZE)
        throw std::invalid_argument("schur_window_size must be in [2, WINDOW_SIZE]");
    SCHUR_WARMUP_FRAMES = config.getInt("schur_warmup_frames", 0);
    if (SCHUR_WARMUP_FRAMES < 0)
        throw std::invalid_argument("schur_warmup_frames must be nonnegative");
    SCHUR_MAX_FEATURES = config.getInt("schur_max_features", 120);
    if (SCHUR_MAX_FEATURES < 4)
        throw std::invalid_argument("schur_max_features must be at least 4");
    SCHUR_MIN_OBSERVATIONS = config.getInt("schur_min_observations", 2);
    if (SCHUR_MIN_OBSERVATIONS < 2 || SCHUR_MIN_OBSERVATIONS > SCHUR_WINDOW_SIZE)
        throw std::invalid_argument("schur_min_observations must be in [2, schur_window_size]");
    SCHUR_MIN_FEATURES = config.getInt("schur_min_features", 12);
    if (SCHUR_MIN_FEATURES < 1 || SCHUR_MIN_FEATURES > SCHUR_MAX_FEATURES)
        throw std::invalid_argument("schur_min_features must be in [1, schur_max_features]");
    // 12 was A/B-tested against 24 on the full EuRoC V2_01_easy / V2_02_medium
    // sequences: the ATE was bit-identical (0.2295 / 0.3809 m) while the looser
    // threshold keeps a few more visually thin frames updating, so 12 is the
    // default (see docs/BACKEND_SCHUR_EKF.md section 7).
    SCHUR_MIN_UPDATE_OBSERVATIONS =
        config.getInt("schur_min_update_observations", 12);
    if (SCHUR_MIN_UPDATE_OBSERVATIONS < 1)
        throw std::invalid_argument("schur_min_update_observations must be positive");
    SCHUR_PIXEL_NOISE = config.getDouble("schur_pixel_noise", 1.5);
    if (!std::isfinite(SCHUR_PIXEL_NOISE) || SCHUR_PIXEL_NOISE <= 0.0)
        throw std::invalid_argument("schur_pixel_noise must be positive");
    SCHUR_CHI2_MULTIPLIER = config.getDouble("schur_chi2_multiplier", 5.0);
    if (!std::isfinite(SCHUR_CHI2_MULTIPLIER) || SCHUR_CHI2_MULTIPLIER <= 0.0)
        throw std::invalid_argument("schur_chi2_multiplier must be positive");
    SCHUR_TRIANGULATION_ITERATIONS = config.getDouble("schur_triangulation_iterations", 5.0);
    if (!std::isfinite(SCHUR_TRIANGULATION_ITERATIONS) || SCHUR_TRIANGULATION_ITERATIONS < 1.0)
        throw std::invalid_argument("schur_triangulation_iterations must be at least 1");
    SCHUR_INFO_RANK_RTOL = config.getDouble("schur_info_rank_rtol", 1e-10);
    SCHUR_INFO_INDEFINITE_TOL = config.getDouble("schur_info_indefinite_tol", 1e-6);
    SCHUR_POINT_BLOCK_RTOL = config.getDouble("schur_point_block_rtol", 1e-8);
    SCHUR_CONDITION_LIMIT = config.getDouble("schur_condition_limit", 1e6);
    if (!std::isfinite(SCHUR_CONDITION_LIMIT) || SCHUR_CONDITION_LIMIT < 0.0)
        throw std::invalid_argument("schur_condition_limit must be nonnegative (0 disables)");
    // Frame-count backstop (secondary restart guard; the time criterion below
    // is the primary one).
    SCHUR_MAX_CONSECUTIVE_FAILURES = config.getInt("schur_max_consecutive_failures", 60);
    if (SCHUR_MAX_CONSECUTIVE_FAILURES < 1)
        throw std::invalid_argument("schur_max_consecutive_failures must be positive");
    // Time-based starvation restart: seconds since the last applied visual
    // update. Non-finite is rejected; <= 0 disables the time criterion.
    SCHUR_MAX_NO_UPDATE_SECONDS =
        config.getDouble("schur_max_no_update_seconds", 3.0);
    if (!std::isfinite(SCHUR_MAX_NO_UPDATE_SECONDS))
        throw std::invalid_argument("schur_max_no_update_seconds must be finite (<=0 disables)");
    // Process-noise inflation during visual starvation. Non-finite is rejected;
    // <= 0 disables the inflation entirely.
    SCHUR_STARVATION_SECONDS = config.getDouble("schur_starvation_seconds", 0.25);
    if (!std::isfinite(SCHUR_STARVATION_SECONDS))
        throw std::invalid_argument("schur_starvation_seconds must be finite (<=0 disables)");
    SCHUR_STARVATION_NOISE_SCALE =
        config.getDouble("schur_starvation_noise_scale", 3.0);
    if (!std::isfinite(SCHUR_STARVATION_NOISE_SCALE) || SCHUR_STARVATION_NOISE_SCALE < 1.0)
        throw std::invalid_argument("schur_starvation_noise_scale must be finite and >= 1.0");
    // Deep starvation escalation of the Q multiplier. Non-finite is rejected;
    // <= 0 disables the escalation.
    SCHUR_STARVATION_DEEP_SECONDS =
        config.getDouble("schur_starvation_deep_seconds", 1.0);
    if (!std::isfinite(SCHUR_STARVATION_DEEP_SECONDS))
        throw std::invalid_argument("schur_starvation_deep_seconds must be finite (<=0 disables)");
    SCHUR_STARVATION_DEEP_NOISE_SCALE =
        config.getDouble("schur_starvation_deep_noise_scale", 10.0);
    if (!std::isfinite(SCHUR_STARVATION_DEEP_NOISE_SCALE) ||
        SCHUR_STARVATION_DEEP_NOISE_SCALE < 1.0)
        throw std::invalid_argument(
            "schur_starvation_deep_noise_scale must be finite and >= 1.0");
    SCHUR_STARVATION_WEAK_UPDATE =
        config.getBool("schur_starvation_weak_update", false);
    SCHUR_WEAK_UPDATE_MAX_SCALE =
        config.getDouble("schur_weak_update_max_scale", 10.0);
    if (!std::isfinite(SCHUR_WEAK_UPDATE_MAX_SCALE) || SCHUR_WEAK_UPDATE_MAX_SCALE < 1.0)
        throw std::invalid_argument("schur_weak_update_max_scale must be finite and >= 1.0");
    // Restart cooldown in sensor time: at most one controlled restart per this
    // many seconds. Non-finite or negative is rejected; 0 disables the cooldown.
    SCHUR_RESTART_COOLDOWN_SECONDS =
        config.getDouble("schur_restart_cooldown_seconds", 8.0);
    if (!std::isfinite(SCHUR_RESTART_COOLDOWN_SECONDS) || SCHUR_RESTART_COOLDOWN_SECONDS < 0.0)
        throw std::invalid_argument(
            "schur_restart_cooldown_seconds must be finite and nonnegative (0 disables)");
    // P5 soft starvation restart: when enabled, a starvation-triggered restart
    // keeps the IMU/EKF nominal state and only drops the clone window and
    // covariance. Default false keeps the previous full-restart behaviour.
    SCHUR_STARVATION_SOFT_RESTART =
        config.getBool("schur_starvation_soft_restart", false);
    // Escalation backstop for the soft restart, in sensor time. Non-finite is
    // rejected; <= 0 disables the escalation.
    SCHUR_SOFT_RESTART_MAX_SECONDS =
        config.getDouble("schur_soft_restart_max_seconds", 10.0);
    if (!std::isfinite(SCHUR_SOFT_RESTART_MAX_SECONDS))
        throw std::invalid_argument(
            "schur_soft_restart_max_seconds must be finite (<=0 disables)");
    // P7 hold policy: stop accepting visual updates instead of restarting.
    // Default false keeps the previous restart behaviour bit-identical.
    SCHUR_STARVATION_HOLD = config.getBool("schur_starvation_hold", false);
    SCHUR_HOLD_RECOVERY_FRAMES = config.getInt("schur_hold_recovery_frames", 2);
    if (SCHUR_HOLD_RECOVERY_FRAMES < 1)
        throw std::invalid_argument("schur_hold_recovery_frames must be positive");
    if (!std::isfinite(SCHUR_INFO_RANK_RTOL) || SCHUR_INFO_RANK_RTOL <= 0.0 ||
        SCHUR_INFO_RANK_RTOL >= 1.0)
        throw std::invalid_argument("schur_info_rank_rtol must be in (0, 1)");
    if (!std::isfinite(SCHUR_INFO_INDEFINITE_TOL) || SCHUR_INFO_INDEFINITE_TOL < 0.0 ||
        SCHUR_INFO_INDEFINITE_TOL >= 1.0)
        throw std::invalid_argument("schur_info_indefinite_tol must be in [0, 1)");
    if (!std::isfinite(SCHUR_POINT_BLOCK_RTOL) || SCHUR_POINT_BLOCK_RTOL <= 0.0 ||
        SCHUR_POINT_BLOCK_RTOL >= 1.0)
        throw std::invalid_argument("schur_point_block_rtol must be in (0, 1)");
    const auto read_positive = [&config](const char *name, double &value, double fallback) {
        value = config.getDouble(name, fallback);
        if (!std::isfinite(value) || value <= 0.0)
            throw std::invalid_argument(std::string(name) + " must be positive");
    };
    read_positive("schur_init_pos_sigma", SCHUR_INIT_POS_SIGMA, 0.05);
    read_positive("schur_init_vel_sigma", SCHUR_INIT_VEL_SIGMA, 0.05);
    read_positive("schur_init_rot_sigma", SCHUR_INIT_ROT_SIGMA, 0.01);
    read_positive("schur_init_ba_sigma", SCHUR_INIT_BA_SIGMA, 0.05);
    read_positive("schur_init_bg_sigma", SCHUR_INIT_BG_SIGMA, 0.005);
    read_positive("schur_acc_noise", SCHUR_ACC_NOISE, 2.0e-3);
    read_positive("schur_gyr_noise", SCHUR_GYR_NOISE, 1.6968e-4);
    read_positive("schur_acc_bias_noise", SCHUR_ACC_BIAS_NOISE, 3.0e-3);
    read_positive("schur_gyr_bias_noise", SCHUR_GYR_BIAS_NOISE, 1.9393e-5);
    RCLCPP_INFO(vins::logger(),
                "Backend type: %s (schur window=%d warmup=%d max_features=%d "
                "min_observations=%d min_features=%d min_update_obs=%d pixel_noise=%.2f "
                "chi2_multiplier=%.2f rank_rtol=%.1e indefinite_tol=%.1e "
                "point_block_rtol=%.1e condition_limit=%.1e max_consecutive_failures=%d "
                "max_no_update_seconds=%.3f starvation_seconds=%.3f "
                "starvation_noise_scale=%.2f deep_starvation_seconds=%.3f "
                "deep_noise_scale=%.2f weak_update=%d weak_update_max_scale=%.2f "
                "restart_cooldown_seconds=%.3f "
                "soft_restart=%d soft_restart_max_seconds=%.3f "
                "starvation_hold=%d hold_recovery_frames=%d "
                "imu_noise(acc=%.3e gyr=%.3e acc_walk=%.3e gyr_walk=%.3e))",
                BACKEND_TYPE.c_str(), SCHUR_WINDOW_SIZE, SCHUR_WARMUP_FRAMES,
                SCHUR_MAX_FEATURES, SCHUR_MIN_OBSERVATIONS, SCHUR_MIN_FEATURES,
                SCHUR_MIN_UPDATE_OBSERVATIONS, SCHUR_PIXEL_NOISE, SCHUR_CHI2_MULTIPLIER,
                SCHUR_INFO_RANK_RTOL, SCHUR_INFO_INDEFINITE_TOL, SCHUR_POINT_BLOCK_RTOL,
                SCHUR_CONDITION_LIMIT, SCHUR_MAX_CONSECUTIVE_FAILURES,
                SCHUR_MAX_NO_UPDATE_SECONDS, SCHUR_STARVATION_SECONDS,
                SCHUR_STARVATION_NOISE_SCALE, SCHUR_STARVATION_DEEP_SECONDS,
                SCHUR_STARVATION_DEEP_NOISE_SCALE,
                static_cast<int>(SCHUR_STARVATION_WEAK_UPDATE), SCHUR_WEAK_UPDATE_MAX_SCALE,
                SCHUR_RESTART_COOLDOWN_SECONDS,
                static_cast<int>(SCHUR_STARVATION_SOFT_RESTART),
                SCHUR_SOFT_RESTART_MAX_SECONDS,
                static_cast<int>(SCHUR_STARVATION_HOLD), SCHUR_HOLD_RECOVERY_FRAMES,
                SCHUR_ACC_NOISE,
                SCHUR_GYR_NOISE, SCHUR_ACC_BIAS_NOISE, SCHUR_GYR_BIAS_NOISE);
    RCLCPP_INFO(vins::logger(),
                "Backend policy: marginalization_threads=%d solver_timeout_policy=%s "
                "retry_multiplier=%.2f stage_timing=%d interval=%d ransac=%d",
                MARGINALIZATION_THREADS, SOLVER_TIMEOUT_POLICY.c_str(),
                SOLVER_TIMEOUT_RETRY_MULTIPLIER, static_cast<int>(STAGE_TIMING),
                STAGE_TIMING_INTERVAL, RANSAC_OUTLIER_REJECTION);

    USE_IMU = config.getInt("imu");
    printf("USE_IMU: %d\n", USE_IMU);
    IMU_TOPIC = config.getString("imu_topic");
    if (USE_IMU) printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
    // Retain the IMU calibration when starting in stereo-only mode so a
    // runtime sensor switch can initialize the IMU estimator correctly.
    ACC_N = config.getDouble("acc_n");
    ACC_W = config.getDouble("acc_w");
    GYR_N = config.getDouble("gyr_n");
    GYR_W = config.getDouble("gyr_w");
    if (config.has("g_norm")) G.z() = config.getDouble("g_norm");

    SOLVER_TIME = config.getDouble("max_solver_time");
    NUM_ITERATIONS = config.getInt("max_num_iterations");
    if (!std::isfinite(SOLVER_TIME) || SOLVER_TIME <= 0.0 || NUM_ITERATIONS < 1)
        throw std::invalid_argument("max_solver_time must be positive and max_num_iterations at least 1");
    RCLCPP_INFO(vins::logger(), "Backend: max_solver_time=%.3f s max_num_iterations=%d",
                SOLVER_TIME, NUM_ITERATIONS);
    MIN_PARALLAX = config.getDouble("keyframe_parallax");
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    OUTPUT_FOLDER = config.getString("output_path");
    if (!output_path.empty()) OUTPUT_FOLDER = output_path;
    OUTPUT_FOLDER = vins::resolvePath(config_file, OUTPUT_FOLDER);
    std::filesystem::create_directories(OUTPUT_FOLDER);
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    ESTIMATE_EXTRINSIC = config.getInt("estimate_extrinsic");
    if (ESTIMATE_EXTRINSIC == 2)
    {
        RCLCPP_WARN(vins::logger(), "have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    else
    {
        if (ESTIMATE_EXTRINSIC == 1)
        {
            RCLCPP_WARN(vins::logger(), " Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            RCLCPP_WARN(vins::logger(), " fix extrinsic param ");

        const Eigen::Matrix4d T = readBodyTCam(config, "body_T_cam0");
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }

    NUM_OF_CAM = config.getInt("num_of_cam");
    printf("camera number %d\n", NUM_OF_CAM);

    if (NUM_OF_CAM != 1 && NUM_OF_CAM != 2)
    {
        printf("num_of_cam should be 1 or 2\n");
        throw std::invalid_argument("num_of_cam must be 1 or 2");
    }

    std::string cam0Calib = config.getString("cam0_calib");
    std::string cam0Path = vins::resolvePath(config_file, cam0Calib);
    CAM_NAMES.push_back(cam0Path);

    if (NUM_OF_CAM == 2)
    {
        STEREO = 1;
        std::string cam1Calib = config.getString("cam1_calib");
        std::string cam1Path = vins::resolvePath(config_file, cam1Calib);
        CAM_NAMES.push_back(cam1Path);

        const Eigen::Matrix4d T = readBodyTCam(config, "body_T_cam1");
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = config.getDouble("td");
    ESTIMATE_TD = config.getInt("estimate_td");
    if (ESTIMATE_TD)
        RCLCPP_INFO_STREAM(vins::logger(), "Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        RCLCPP_INFO_STREAM(vins::logger(), "Synchronized sensors, fix time offset: " << TD);

    ROW = config.getInt("image_height");
    COL = config.getInt("image_width");
    RCLCPP_INFO(vins::logger(), "ROW: %d COL: %d ", ROW, COL);

    if (!USE_IMU && !STEREO) throw std::invalid_argument("Mono mode requires an IMU");
    for (const auto &camera : CAM_NAMES)
        if (!std::filesystem::is_regular_file(camera)) throw std::runtime_error("Missing camera calibration: " + camera);

    if (!USE_IMU)
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }
}
