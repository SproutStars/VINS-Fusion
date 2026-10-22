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
