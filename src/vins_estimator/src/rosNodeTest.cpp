/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, HKUST
 * Licensed under the GNU General Public License v3.0.
 *******************************************************/
#include <vins_common/ros_utils.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <std_msgs/msg/bool.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <condition_variable>
#include <mutex>
#include <thread>
#include "estimator/estimator.h"
#include "utility/visualization.h"

class VinsNode final : public rclcpp::Node
{
    using Image = sensor_msgs::msg::Image;
    using Imu = sensor_msgs::msg::Imu;
    using Cloud = sensor_msgs::msg::PointCloud;
    using Bool = std_msgs::msg::Bool;
public:
    VinsNode() : Node("vins_estimator") {}
    ~VinsNode() override
    {
        finalizeTracking();
        estimator_.reset();
        last_frame_ = {};
        unregisterPub();
    }

    // Emit the last statistics window before the ROS context and its logger
    // are torn down. This is idempotent so the destructor can call it too.
    void finalizeTracking()
    {
        if (tracking_finalized_)
            return;
        stopTracking();
        tracking_finalized_ = true;
        reportTrackingStats("final");
    }

    // Optional ROS-parameter overrides for the backend selection. They default
    // to the values read from the configuration file so `-p` is only needed
    // when a run should deviate from the YAML.
    void applyBackendParameterOverrides()
    {
        const std::string backend = declare_parameter<std::string>("backend_type", BACKEND_TYPE);
        if (backend != "ceres" && backend != "schur_ekf")
            throw std::invalid_argument("backend_type must be 'ceres' or 'schur_ekf'");
        BACKEND_TYPE = backend;

        SCHUR_WINDOW_SIZE = declare_parameter<int>("schur_window_size", SCHUR_WINDOW_SIZE);
        if (SCHUR_WINDOW_SIZE < 2 || SCHUR_WINDOW_SIZE > WINDOW_SIZE)
            throw std::invalid_argument("schur_window_size must be in [2, WINDOW_SIZE]");
        SCHUR_WARMUP_FRAMES =
            declare_parameter<int>("schur_warmup_frames", SCHUR_WARMUP_FRAMES);
        if (SCHUR_WARMUP_FRAMES < 0)
            throw std::invalid_argument("schur_warmup_frames must be nonnegative");
        SCHUR_MAX_FEATURES = declare_parameter<int>("schur_max_features", SCHUR_MAX_FEATURES);
        if (SCHUR_MAX_FEATURES < 4)
            throw std::invalid_argument("schur_max_features must be at least 4");
        SCHUR_MIN_OBSERVATIONS =
            declare_parameter<int>("schur_min_observations", SCHUR_MIN_OBSERVATIONS);
        if (SCHUR_MIN_OBSERVATIONS < 2 || SCHUR_MIN_OBSERVATIONS > SCHUR_WINDOW_SIZE)
            throw std::invalid_argument("schur_min_observations must be in [2, schur_window_size]");
        SCHUR_MIN_FEATURES =
            declare_parameter<int>("schur_min_features", SCHUR_MIN_FEATURES);
        if (SCHUR_MIN_FEATURES < 1 || SCHUR_MIN_FEATURES > SCHUR_MAX_FEATURES)
            throw std::invalid_argument("schur_min_features must be in [1, schur_max_features]");
        SCHUR_MIN_UPDATE_OBSERVATIONS = declare_parameter<int>(
            "schur_min_update_observations", SCHUR_MIN_UPDATE_OBSERVATIONS);
        if (SCHUR_MIN_UPDATE_OBSERVATIONS < 1)
            throw std::invalid_argument("schur_min_update_observations must be positive");
        SCHUR_PIXEL_NOISE = declare_parameter<double>("schur_pixel_noise", SCHUR_PIXEL_NOISE);
        if (!std::isfinite(SCHUR_PIXEL_NOISE) || SCHUR_PIXEL_NOISE <= 0.0)
            throw std::invalid_argument("schur_pixel_noise must be positive");
        SCHUR_CHI2_MULTIPLIER =
            declare_parameter<double>("schur_chi2_multiplier", SCHUR_CHI2_MULTIPLIER);
        if (!std::isfinite(SCHUR_CHI2_MULTIPLIER) || SCHUR_CHI2_MULTIPLIER <= 0.0)
            throw std::invalid_argument("schur_chi2_multiplier must be positive");
        SCHUR_TRIANGULATION_ITERATIONS = declare_parameter<double>(
            "schur_triangulation_iterations", SCHUR_TRIANGULATION_ITERATIONS);
        if (!std::isfinite(SCHUR_TRIANGULATION_ITERATIONS) || SCHUR_TRIANGULATION_ITERATIONS < 1.0)
            throw std::invalid_argument("schur_triangulation_iterations must be at least 1");
        SCHUR_INFO_RANK_RTOL =
            declare_parameter<double>("schur_info_rank_rtol", SCHUR_INFO_RANK_RTOL);
        if (!std::isfinite(SCHUR_INFO_RANK_RTOL) || SCHUR_INFO_RANK_RTOL <= 0.0 ||
            SCHUR_INFO_RANK_RTOL >= 1.0)
            throw std::invalid_argument("schur_info_rank_rtol must be in (0, 1)");
        SCHUR_INFO_INDEFINITE_TOL = declare_parameter<double>(
            "schur_info_indefinite_tol", SCHUR_INFO_INDEFINITE_TOL);
        if (!std::isfinite(SCHUR_INFO_INDEFINITE_TOL) ||
            SCHUR_INFO_INDEFINITE_TOL < 0.0 || SCHUR_INFO_INDEFINITE_TOL >= 1.0)
            throw std::invalid_argument("schur_info_indefinite_tol must be in [0, 1)");
        SCHUR_POINT_BLOCK_RTOL = declare_parameter<double>(
            "schur_point_block_rtol", SCHUR_POINT_BLOCK_RTOL);
        if (!std::isfinite(SCHUR_POINT_BLOCK_RTOL) || SCHUR_POINT_BLOCK_RTOL <= 0.0 ||
            SCHUR_POINT_BLOCK_RTOL >= 1.0)
            throw std::invalid_argument("schur_point_block_rtol must be in (0, 1)");
        SCHUR_CONDITION_LIMIT =
            declare_parameter<double>("schur_condition_limit", SCHUR_CONDITION_LIMIT);
        if (!std::isfinite(SCHUR_CONDITION_LIMIT) || SCHUR_CONDITION_LIMIT < 0.0)
            throw std::invalid_argument("schur_condition_limit must be nonnegative");
        SCHUR_MAX_CONSECUTIVE_FAILURES = declare_parameter<int>(
            "schur_max_consecutive_failures", SCHUR_MAX_CONSECUTIVE_FAILURES);
        if (SCHUR_MAX_CONSECUTIVE_FAILURES < 1)
            throw std::invalid_argument("schur_max_consecutive_failures must be positive");
        // Primary starvation guard in sensor time; <= 0 disables it.
        SCHUR_MAX_NO_UPDATE_SECONDS = declare_parameter<double>(
            "schur_max_no_update_seconds", SCHUR_MAX_NO_UPDATE_SECONDS);
        if (!std::isfinite(SCHUR_MAX_NO_UPDATE_SECONDS))
            throw std::invalid_argument(
                "schur_max_no_update_seconds must be finite (<=0 disables)");
        // Starvation onset for process-noise inflation; <= 0 disables it.
        SCHUR_STARVATION_SECONDS = declare_parameter<double>(
            "schur_starvation_seconds", SCHUR_STARVATION_SECONDS);
        if (!std::isfinite(SCHUR_STARVATION_SECONDS))
            throw std::invalid_argument(
                "schur_starvation_seconds must be finite (<=0 disables)");
        SCHUR_STARVATION_NOISE_SCALE = declare_parameter<double>(
            "schur_starvation_noise_scale", SCHUR_STARVATION_NOISE_SCALE);
        if (!std::isfinite(SCHUR_STARVATION_NOISE_SCALE) ||
            SCHUR_STARVATION_NOISE_SCALE < 1.0)
            throw std::invalid_argument(
                "schur_starvation_noise_scale must be finite and >= 1.0");
        // Deep starvation onset for the escalated Q multiplier; <= 0 disables.
        SCHUR_STARVATION_DEEP_SECONDS = declare_parameter<double>(
            "schur_starvation_deep_seconds", SCHUR_STARVATION_DEEP_SECONDS);
        if (!std::isfinite(SCHUR_STARVATION_DEEP_SECONDS))
            throw std::invalid_argument(
                "schur_starvation_deep_seconds must be finite (<=0 disables)");
        SCHUR_STARVATION_DEEP_NOISE_SCALE = declare_parameter<double>(
            "schur_starvation_deep_noise_scale", SCHUR_STARVATION_DEEP_NOISE_SCALE);
        if (!std::isfinite(SCHUR_STARVATION_DEEP_NOISE_SCALE) ||
            SCHUR_STARVATION_DEEP_NOISE_SCALE < 1.0)
            throw std::invalid_argument(
                "schur_starvation_deep_noise_scale must be finite and >= 1.0");
        // Minimum sensor-time separation between controlled restarts; 0 disables.
        SCHUR_RESTART_COOLDOWN_SECONDS = declare_parameter<double>(
            "schur_restart_cooldown_seconds", SCHUR_RESTART_COOLDOWN_SECONDS);
        if (!std::isfinite(SCHUR_RESTART_COOLDOWN_SECONDS) ||
            SCHUR_RESTART_COOLDOWN_SECONDS < 0.0)
            throw std::invalid_argument(
                "schur_restart_cooldown_seconds must be finite and nonnegative "
                "(0 disables)");
        // P5 soft starvation restart; default false keeps the full-restart path.
        SCHUR_STARVATION_SOFT_RESTART = declare_parameter<bool>(
            "schur_starvation_soft_restart", SCHUR_STARVATION_SOFT_RESTART);
        // Escalation backstop for the soft restart; non-finite is rejected and
        // <= 0 disables the escalation.
        SCHUR_SOFT_RESTART_MAX_SECONDS = declare_parameter<double>(
            "schur_soft_restart_max_seconds", SCHUR_SOFT_RESTART_MAX_SECONDS);
        if (!std::isfinite(SCHUR_SOFT_RESTART_MAX_SECONDS))
            throw std::invalid_argument(
                "schur_soft_restart_max_seconds must be finite (<=0 disables)");
        // P7 hold policy: no restart, no covariance re-inflation; keep
        // publishing pure-IMU propagation until the tracker recovers.
        SCHUR_STARVATION_HOLD = declare_parameter<bool>(
            "schur_starvation_hold", SCHUR_STARVATION_HOLD);
        SCHUR_HOLD_RECOVERY_FRAMES = declare_parameter<int>(
            "schur_hold_recovery_frames", SCHUR_HOLD_RECOVERY_FRAMES);
        if (SCHUR_HOLD_RECOVERY_FRAMES < 1)
            throw std::invalid_argument("schur_hold_recovery_frames must be positive");
        // P6 starvation-adaptive weak update; default false is a strict no-op.
        SCHUR_STARVATION_WEAK_UPDATE = declare_parameter<bool>(
            "schur_starvation_weak_update", SCHUR_STARVATION_WEAK_UPDATE);
        // Upper bound of the weak-update scale; 1.0 is equivalent to disabling.
        SCHUR_WEAK_UPDATE_MAX_SCALE = declare_parameter<double>(
            "schur_weak_update_max_scale", SCHUR_WEAK_UPDATE_MAX_SCALE);
        if (!std::isfinite(SCHUR_WEAK_UPDATE_MAX_SCALE) || SCHUR_WEAK_UPDATE_MAX_SCALE < 1.0)
            throw std::invalid_argument(
                "schur_weak_update_max_scale must be finite and >= 1.0");
        SCHUR_INIT_POS_SIGMA =
            declare_parameter<double>("schur_init_pos_sigma", SCHUR_INIT_POS_SIGMA);
        SCHUR_INIT_VEL_SIGMA =
            declare_parameter<double>("schur_init_vel_sigma", SCHUR_INIT_VEL_SIGMA);
        SCHUR_INIT_ROT_SIGMA =
            declare_parameter<double>("schur_init_rot_sigma", SCHUR_INIT_ROT_SIGMA);
        SCHUR_INIT_BA_SIGMA =
            declare_parameter<double>("schur_init_ba_sigma", SCHUR_INIT_BA_SIGMA);
        SCHUR_INIT_BG_SIGMA =
            declare_parameter<double>("schur_init_bg_sigma", SCHUR_INIT_BG_SIGMA);
        SCHUR_ACC_NOISE = declare_parameter<double>("schur_acc_noise", SCHUR_ACC_NOISE);
        SCHUR_GYR_NOISE = declare_parameter<double>("schur_gyr_noise", SCHUR_GYR_NOISE);
        SCHUR_ACC_BIAS_NOISE =
            declare_parameter<double>("schur_acc_bias_noise", SCHUR_ACC_BIAS_NOISE);
        SCHUR_GYR_BIAS_NOISE =
            declare_parameter<double>("schur_gyr_bias_noise", SCHUR_GYR_BIAS_NOISE);
        const double imu_noise[] = {SCHUR_ACC_NOISE, SCHUR_GYR_NOISE,
                                    SCHUR_ACC_BIAS_NOISE, SCHUR_GYR_BIAS_NOISE};
        if (std::any_of(std::begin(imu_noise), std::end(imu_noise),
                        [](double value) { return !std::isfinite(value) || value <= 0.0; }))
            throw std::invalid_argument("schur_*_noise parameters must be positive");
        const double init_sigmas[] = {
            SCHUR_INIT_POS_SIGMA, SCHUR_INIT_VEL_SIGMA, SCHUR_INIT_ROT_SIGMA,
            SCHUR_INIT_BA_SIGMA, SCHUR_INIT_BG_SIGMA};
        if (std::any_of(std::begin(init_sigmas), std::end(init_sigmas),
                        [](double value) { return !std::isfinite(value) || value <= 0.0; }))
            throw std::invalid_argument("schur_init_*_sigma parameters must be positive");
        RCLCPP_INFO(this->get_logger(), "Backend selection: %s", BACKEND_TYPE.c_str());
    }

    void start(int argc, char **argv)
    {
        readParameters(vins::configFile(shared_from_this(), argc, argv),
            declare_parameter<std::string>("output_path", ""));
        applyBackendParameterOverrides();
        IMAGE0_TOPIC = declare_parameter<std::string>("image0_topic", IMAGE0_TOPIC);
        IMAGE1_TOPIC = declare_parameter<std::string>("image1_topic", IMAGE1_TOPIC);
        IMU_TOPIC = declare_parameter<std::string>("imu_topic", IMU_TOPIC);
        sync_tolerance_ = declare_parameter<double>("stereo_sync_tolerance", 0.003);
        max_queue_ = declare_parameter<int>("image_queue_size", 100);
        if (max_queue_ < 1 || sync_tolerance_ < 0.0)
            throw std::invalid_argument("Invalid queue size or stereo synchronization tolerance");
        tracking_queue_size_ = declare_parameter<int>("tracking_queue_size", 5);
        tracker_detection_interval_ =
            declare_parameter<int>("tracker_detection_interval", 1);
        tracker_stats_interval_ =
            declare_parameter<int>("tracker_stats_interval", 100);
        tracking_queue_policy_ =
            declare_parameter<std::string>("tracking_queue_policy", "drop_oldest");
        const int opencv_threads = declare_parameter<int>("opencv_threads", 1);
        if (tracking_queue_size_ < 1 || opencv_threads < 0)
            throw std::invalid_argument("tracking_queue_size must be positive and opencv_threads nonnegative");
        if (tracker_detection_interval_ < 1 || tracker_stats_interval_ < 1)
            throw std::invalid_argument(
                "tracker_detection_interval and tracker_stats_interval must be positive");
        if (tracking_queue_policy_ != "drop_oldest" &&
            tracking_queue_policy_ != "latest")
            throw std::invalid_argument(
                "tracking_queue_policy must be 'drop_oldest' or 'latest'");
        cv::setNumThreads(opencv_threads);
        parameter_callback_ = add_on_set_parameters_callback(
            [this](const std::vector<rclcpp::Parameter> & parameters) {
                rcl_interfaces::msg::SetParametersResult result;
                result.successful = true;
                for (const auto & parameter : parameters) {
                    const auto & name = parameter.get_name();
                    if (name == "config_file" || name == "output_path" ||
                        name == "image0_topic" || name == "image1_topic" ||
                        name == "imu_topic" || name == "backend_type" ||
                        name == "schur_window_size" || name == "schur_warmup_frames" ||
                        name == "schur_max_features" ||
                        name == "schur_min_observations" ||
                        name == "schur_min_features" ||
                        name == "schur_min_update_observations" ||
                        name == "schur_pixel_noise" ||
                        name == "schur_chi2_multiplier" ||
                        name == "schur_triangulation_iterations" ||
                        name == "schur_info_rank_rtol" ||
                        name == "schur_info_indefinite_tol" ||
                        name == "schur_point_block_rtol" ||
                        name == "schur_condition_limit" ||
                        name == "schur_max_consecutive_failures" ||
                        name == "schur_max_no_update_seconds" ||
                        name == "schur_starvation_seconds" ||
                        name == "schur_starvation_noise_scale" ||
                        name == "schur_starvation_deep_seconds" ||
                        name == "schur_starvation_deep_noise_scale" ||
                        name == "schur_restart_cooldown_seconds" ||
                        name == "schur_starvation_soft_restart" ||
                        name == "schur_soft_restart_max_seconds" ||
                        name == "schur_starvation_hold" ||
                        name == "schur_hold_recovery_frames" ||
                        name == "schur_starvation_weak_update" ||
                        name == "schur_weak_update_max_scale" ||
                        name == "schur_init_pos_sigma" || name == "schur_init_vel_sigma" ||
                        name == "schur_init_rot_sigma" || name == "schur_init_ba_sigma" ||
                        name == "schur_init_bg_sigma" ||
                        name == "tracker_stats_interval") {
                        result.successful = false;
                        result.reason = name + " is fixed after startup; stop and relaunch the node";
                        return result;
                    }
                    if (name == "tracking_queue_size" && parameter.as_int() < 1) {
                        result.successful = false;
                        result.reason = "tracking_queue_size must be positive";
                        return result;
                    }
                    if (name == "tracker_detection_interval" && parameter.as_int() < 1) {
                        result.successful = false;
                        result.reason = "tracker_detection_interval must be positive";
                        return result;
                    }
                    if (name == "tracking_queue_policy") {
                        const std::string policy = parameter.as_string();
                        if (policy != "drop_oldest" && policy != "latest") {
                            result.successful = false;
                            result.reason =
                                "tracking_queue_policy must be 'drop_oldest' or 'latest'";
                            return result;
                        }
                    }
                    if (name == "opencv_threads" && parameter.as_int() < 0) {
                        result.successful = false;
                        result.reason = "opencv_threads must be nonnegative";
                        return result;
                    }
                }
                for (const auto & parameter : parameters) {
                    if (parameter.get_name() == "tracking_queue_size") {
                        std::lock_guard<std::mutex> lock(tracking_mutex_);
                        tracking_queue_size_ = static_cast<int>(parameter.as_int());
                    } else if (parameter.get_name() == "tracking_queue_policy") {
                        std::lock_guard<std::mutex> lock(tracking_mutex_);
                        tracking_queue_policy_ = parameter.as_string();
                    } else if (parameter.get_name() == "tracker_detection_interval") {
                        tracker_detection_interval_ =
                            static_cast<int>(parameter.as_int());
                        applyTrackerConfiguration();
                    } else if (parameter.get_name() == "opencv_threads") {
                        cv::setNumThreads(static_cast<int>(parameter.as_int()));
                    }
                }
                return result;
            });
        registerPub(shared_from_this());
        estimator_ = std::make_unique<Estimator>();
        estimator_->setParameter();
        applyTrackerConfiguration();
        // Best effort accepts both reliable rosbag publishers and sensor drivers.
        auto qos = rclcpp::SensorDataQoS().keep_last(100);
        sub_imu_ = create_subscription<Imu>(IMU_TOPIC.empty() ? "/imu0" : IMU_TOPIC,
            rclcpp::SensorDataQoS().keep_last(2000), [this](Imu::ConstSharedPtr m) {
                if (USE_IMU) estimator_->inputIMU(vins::seconds(m->header.stamp),
                    Eigen::Vector3d(m->linear_acceleration.x, m->linear_acceleration.y, m->linear_acceleration.z),
                    Eigen::Vector3d(m->angular_velocity.x, m->angular_velocity.y, m->angular_velocity.z));
            });
        sub_image0_ = create_subscription<Image>(IMAGE0_TOPIC, qos,
            [this](Image::ConstSharedPtr m) { enqueue(images0_, std::move(m)); });
        if (NUM_OF_CAM == 2)
            sub_image1_ = create_subscription<Image>(IMAGE1_TOPIC, qos,
                [this](Image::ConstSharedPtr m) { if (STEREO) enqueue(images1_, std::move(m)); });
        sub_feature_ = create_subscription<Cloud>("/feature_tracker/feature", qos,
            [this](Cloud::ConstSharedPtr m) { feature(m); });
        sub_restart_ = create_subscription<Bool>("/vins_restart", 10,
            [this](Bool::ConstSharedPtr m) { if (m->data) restart(); });
        sub_imu_switch_ = create_subscription<Bool>("/vins_imu_switch", 10,
            [this](Bool::ConstSharedPtr m) { switchSensors(m->data, STEREO); });
        sub_cam_switch_ = create_subscription<Bool>("/vins_cam_switch", 10,
            [this](Bool::ConstSharedPtr m) {
                if (m->data && NUM_OF_CAM != 2) {
                    RCLCPP_WARN(get_logger(), "Stereo requires two calibrated cameras");
                    return;
                }
                switchSensors(USE_IMU, m->data);
                images0_.clear(); images1_.clear();
            });
        startTracking();
        RCLCPP_INFO(get_logger(), "Ready: %s, image=%s, imu=%s", STEREO ? "stereo" : "mono", IMAGE0_TOPIC.c_str(), IMU_TOPIC.c_str());
        RCLCPP_INFO(get_logger(),
                    "Tracker config: queue_policy=%s detection_interval=%d stats_interval=%d",
                    tracking_queue_policy_.c_str(), tracker_detection_interval_,
                    tracker_stats_interval_);
    }

private:
    void enqueue(std::deque<Image::ConstSharedPtr> &queue, Image::ConstSharedPtr image)
    {
        if (queue.size() >= static_cast<size_t>(max_queue_)) queue.pop_front();
        queue.push_back(std::move(image));
        syncImages();
    }
    struct FramePair {
        Image::ConstSharedPtr left, right;
        std::chrono::steady_clock::time_point enqueued_at;
    };
    struct TrackingStatsWindow {
        uint64_t frames = 0;
        double queue_wait_sum_ms = 0.0, queue_wait_max_ms = 0.0;
        double decode_sum_ms = 0.0, decode_max_ms = 0.0;
        double track_sum_ms = 0.0, track_max_ms = 0.0;
        double pyramid_sum_ms = 0.0, pyramid_max_ms = 0.0;
        double flow_sum_ms = 0.0, flow_max_ms = 0.0;
        double detect_sum_ms = 0.0, detect_max_ms = 0.0;
        double input_image_sum_ms = 0.0, input_image_max_ms = 0.0;
        int tracked_feature_sum = 0, detected_feature_sum = 0;
    };
    static double elapsedMs(const std::chrono::steady_clock::time_point &start,
                            const std::chrono::steady_clock::time_point &end)
    {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    void applyTrackerConfiguration()
    {
        if (estimator_)
            estimator_->featureTracker.setDetectionInterval(tracker_detection_interval_);
    }
    void recordTrackingStats(double queue_wait_ms, double decode_ms,
                             const FeatureTrackerTimings &timings,
                             double input_image_ms)
    {
        TrackingStatsWindow &stats = tracking_stats_;
        ++stats.frames;
        ++tracking_total_frames_;
        stats.queue_wait_sum_ms += queue_wait_ms;
        stats.queue_wait_max_ms = std::max(stats.queue_wait_max_ms, queue_wait_ms);
        stats.decode_sum_ms += decode_ms;
        stats.decode_max_ms = std::max(stats.decode_max_ms, decode_ms);
        stats.track_sum_ms += timings.total_ms;
        stats.track_max_ms = std::max(stats.track_max_ms, timings.total_ms);
        stats.pyramid_sum_ms += timings.pyramid_ms;
        stats.pyramid_max_ms = std::max(stats.pyramid_max_ms, timings.pyramid_ms);
        stats.flow_sum_ms += timings.flow_ms;
        stats.flow_max_ms = std::max(stats.flow_max_ms, timings.flow_ms);
        stats.detect_sum_ms += timings.detect_ms;
        stats.detect_max_ms = std::max(stats.detect_max_ms, timings.detect_ms);
        stats.input_image_sum_ms += input_image_ms;
        stats.input_image_max_ms = std::max(stats.input_image_max_ms, input_image_ms);
        stats.tracked_feature_sum += timings.tracked_features;
        stats.detected_feature_sum += timings.detected_features;
        if (tracking_total_frames_ %
                static_cast<uint64_t>(tracker_stats_interval_) == 0)
            reportTrackingStats("periodic");
    }
    void reportTrackingStats(const char *phase)
    {
        const TrackingStatsWindow &stats = tracking_stats_;
        const uint64_t overflow_events =
            tracking_overflow_events_.load(std::memory_order_relaxed);
        const uint64_t dropped_pairs =
            tracking_overflow_dropped_pairs_.load(std::memory_order_relaxed);
        const bool final_report = std::string(phase) == "final";
        if (!final_report && stats.frames == 0 && overflow_events == 0 &&
            dropped_pairs == 0)
            return;
        const double frames = static_cast<double>(std::max<uint64_t>(stats.frames, 1));
        char line[1024];
        std::snprintf(
            line, sizeof(line),
            "Tracker %s stats: frames=%llu total_frames=%llu "
            "queue_wait_ms=%.2f/%.2f decode_ms=%.2f/%.2f "
            "track_ms=%.2f/%.2f pyramid_ms=%.2f/%.2f "
            "flow_ms=%.2f/%.2f detect_ms=%.2f/%.2f "
            "input_image_ms=%.2f/%.2f features_avg=%.1f/%.1f "
            "overflow_events=%llu overflow_dropped_pairs=%llu",
            phase, static_cast<unsigned long long>(stats.frames),
            static_cast<unsigned long long>(tracking_total_frames_),
            stats.queue_wait_sum_ms / frames, stats.queue_wait_max_ms,
            stats.decode_sum_ms / frames, stats.decode_max_ms,
            stats.track_sum_ms / frames, stats.track_max_ms,
            stats.pyramid_sum_ms / frames, stats.pyramid_max_ms,
            stats.flow_sum_ms / frames, stats.flow_max_ms,
            stats.detect_sum_ms / frames, stats.detect_max_ms,
            stats.input_image_sum_ms / frames, stats.input_image_max_ms,
            stats.tracked_feature_sum / frames, stats.detected_feature_sum / frames,
            static_cast<unsigned long long>(overflow_events),
            static_cast<unsigned long long>(dropped_pairs));
        // SIGINT normally tears down the ROS logging subsystem before main()
        // returns. Keep the exact final counters in stderr so a benchmark or
        // field run still records the complete overflow total.
        if (final_report && !rclcpp::ok())
            std::fprintf(stderr, "%s\n", line);
        else
            RCLCPP_INFO(get_logger(), "%s", line);
        tracking_stats_ = TrackingStatsWindow{};
    }
    static cv::Mat decode(const Image::ConstSharedPtr &image)
    {
        // CvShare can alias ROS-owned pixels. FramePair ownership is retained
        // through the NEXT tracking call because FeatureTracker keeps prev_img.
        const auto encoding = image->encoding == "8UC1" ? image->encoding :
            sensor_msgs::image_encodings::MONO8;
        return cv_bridge::toCvShare(image, encoding)->image;
    }
    void track(FramePair frame)
    {
        const auto track_start = std::chrono::steady_clock::now();
        const double queue_wait_ms = elapsedMs(frame.enqueued_at, track_start);
        double decode_ms = 0.0;
        double input_image_ms = 0.0;
        FeatureTrackerTimings timings;
        try {
            const auto decode_start = std::chrono::steady_clock::now();
            const cv::Mat left = decode(frame.left);
            const cv::Mat right = frame.right ? decode(frame.right) : cv::Mat();
            decode_ms = elapsedMs(decode_start, std::chrono::steady_clock::now());
            const auto input_start = std::chrono::steady_clock::now();
            estimator_->inputImage(vins::seconds(frame.left->header.stamp), left, right);
            input_image_ms = elapsedMs(input_start, std::chrono::steady_clock::now());
            timings = estimator_->featureTracker.lastTimings();
        } catch (const cv::Exception &e) {
            RCLCPP_ERROR(get_logger(), "Image conversion/tracking failed: %s", e.what());
        } catch (const cv_bridge::Exception &e) {
            RCLCPP_ERROR(get_logger(), "Image conversion failed: %s", e.what());
        }
        last_frame_ = std::move(frame);
        recordTrackingStats(queue_wait_ms, decode_ms, timings, input_image_ms);
    }
    void startTracking()
    {
        tracking_stopping_ = false;
        if (!MULTIPLE_THREAD) return;
        tracking_thread_ = std::thread([this] {
            while (true) {
                FramePair frame;
                {
                    std::unique_lock<std::mutex> lock(tracking_mutex_);
                    tracking_ready_.wait(lock, [this] { return tracking_stopping_ || !tracking_frames_.empty(); });
                    if (tracking_stopping_) return;
                    frame = std::move(tracking_frames_.front());
                    tracking_frames_.pop_front();
                }
                track(std::move(frame));
            }
        });
    }
    void stopTracking()
    {
        {
            std::lock_guard<std::mutex> lock(tracking_mutex_);
            tracking_stopping_ = true;
            tracking_frames_.clear();
        }
        tracking_ready_.notify_all();
        if (tracking_thread_.joinable()) tracking_thread_.join();
    }
    void dispatch(FramePair frame)
    {
        if (!MULTIPLE_THREAD) { track(std::move(frame)); return; }
        bool overflowed = false;
        bool latest_policy = false;
        size_t dropped_this_event = 0;
        uint64_t overflow_events = 0;
        uint64_t overflow_dropped_pairs = 0;
        {
            std::lock_guard<std::mutex> lock(tracking_mutex_);
            if (tracking_frames_.size() >= static_cast<size_t>(tracking_queue_size_)) {
                overflowed = true;
                latest_policy = tracking_queue_policy_ == "latest";
                if (latest_policy) {
                    dropped_this_event = tracking_frames_.size();
                    tracking_frames_.clear();
                } else {
                    tracking_frames_.pop_front();
                    dropped_this_event = 1;
                }
                overflow_events = ++tracking_overflow_events_;
                const uint64_t dropped =
                    static_cast<uint64_t>(dropped_this_event);
                overflow_dropped_pairs =
                    tracking_overflow_dropped_pairs_.fetch_add(
                        dropped, std::memory_order_relaxed) + dropped;
            }
            tracking_frames_.push_back(std::move(frame));
        }
        // No time-based throttle: the exact cumulative counts remain visible
        // even when overflow events are much denser than the old 5 s warning.
        if (overflowed && (overflow_events == 1 || overflow_events % 100 == 0)) {
            RCLCPP_WARN(
                get_logger(),
                "Tracking queue overflow: events=%llu dropped_this_event=%zu "
                "dropped_pairs=%llu policy=%s",
                static_cast<unsigned long long>(overflow_events), dropped_this_event,
                static_cast<unsigned long long>(overflow_dropped_pairs),
                latest_policy ? "latest" : "drop_oldest");
        }
        tracking_ready_.notify_one();
    }
    void syncImages()
    {
        // Called only by the single-threaded ROS executor. The worker never
        // touches synchronization queues or sensor-switch/restart state.
        while (!images0_.empty() && (!STEREO || !images1_.empty())) {
            if (STEREO) {
                const double dt = vins::seconds(images0_.front()->header.stamp) - vins::seconds(images1_.front()->header.stamp);
                if (dt < -sync_tolerance_) { images0_.pop_front(); continue; }
                if (dt > sync_tolerance_) { images1_.pop_front(); continue; }
            }
            FramePair frame{images0_.front(), {},
                            std::chrono::steady_clock::now()};
            images0_.pop_front();
            if (STEREO) { frame.right = images1_.front(); images1_.pop_front(); }
            dispatch(std::move(frame));
        }
    }
    void restart()
    {
        RCLCPP_WARN(get_logger(),
                    "Restarting estimator with already-loaded configuration; "
                    "edit the YAML and relaunch the node to apply parameter changes");
        stopTracking();
        estimator_.reset(); // Join both workers before replacing their estimator.
        last_frame_ = {};
        images0_.clear(); images1_.clear();
        resetVisualization();
        estimator_ = std::make_unique<Estimator>();
        estimator_->setParameter();
        applyTrackerConfiguration();
        startTracking();
    }
    void switchSensors(bool imu, bool stereo)
    {
        if (!imu && !stereo) {
            RCLCPP_WARN(get_logger(), "Mono mode requires an IMU");
            return;
        }
        if (imu && (IMU_TOPIC.empty() || ACC_N <= 0 || GYR_N <= 0 || ACC_W <= 0 || GYR_W <= 0)) {
            RCLCPP_WARN(get_logger(), "Enabling IMU requires an IMU topic and positive noise parameters in the configuration");
            return;
        }
        if (imu == static_cast<bool>(USE_IMU) && stereo == static_cast<bool>(STEREO)) return;
        // Join both workers before changing flags or discarding shared state.
        stopTracking();
        estimator_.reset();
        last_frame_ = {};
        USE_IMU = imu;
        STEREO = stereo;
        images0_.clear(); images1_.clear();
        resetVisualization();
        estimator_ = std::make_unique<Estimator>();
        estimator_->setParameter();
        applyTrackerConfiguration();
        startTracking();
    }
    void feature(const Cloud::ConstSharedPtr &m)
    {
        if (m->channels.size() < 6) return;
        for (const auto &channel : m->channels)
            if (channel.values.size() < m->points.size()) return;
        std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> frame;
        for (size_t i = 0; i < m->points.size(); ++i) {
            if (std::abs(m->points[i].z - 1.0) > 1e-6) return;
            const int id = static_cast<int>(m->channels[0].values[i]);
            const int camera = static_cast<int>(m->channels[1].values[i]);
            if (camera < 0 || camera >= NUM_OF_CAM) return;
            Eigen::Matrix<double, 7, 1> value;
            value << m->points[i].x, m->points[i].y, m->points[i].z,
                     m->channels[2].values[i], m->channels[3].values[i],
                     m->channels[4].values[i], m->channels[5].values[i];
            frame[id].emplace_back(camera, value);
        }
        estimator_->inputFeature(vins::seconds(m->header.stamp), frame);
    }
    std::unique_ptr<Estimator> estimator_;
    std::deque<Image::ConstSharedPtr> images0_, images1_;
    int max_queue_ = 100;
    double sync_tolerance_ = 0.003;
    rclcpp::Subscription<Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<Image>::SharedPtr sub_image0_, sub_image1_;
    rclcpp::Subscription<Cloud>::SharedPtr sub_feature_;
    rclcpp::Subscription<Bool>::SharedPtr sub_restart_, sub_imu_switch_, sub_cam_switch_;
    FramePair last_frame_;
    std::deque<FramePair> tracking_frames_;
    std::mutex tracking_mutex_;
    std::condition_variable tracking_ready_;
    std::thread tracking_thread_;
    bool tracking_stopping_ = false;
    int tracking_queue_size_ = 5;
    std::string tracking_queue_policy_ = "drop_oldest";
    int tracker_detection_interval_ = 1;
    int tracker_stats_interval_ = 100;
    uint64_t tracking_total_frames_ = 0;
    TrackingStatsWindow tracking_stats_;
    std::atomic<uint64_t> tracking_overflow_events_{0};
    std::atomic<uint64_t> tracking_overflow_dropped_pairs_{0};
    bool tracking_finalized_ = false;
    rclcpp::Node::OnSetParametersCallbackHandle::SharedPtr parameter_callback_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    int result = 0;
    try {
        auto node = std::make_shared<VinsNode>();
        node->start(argc, argv);
        rclcpp::spin(node);
        node->finalizeTracking();
    } catch (const std::exception &e) {
        RCLCPP_ERROR(vins::logger(), "%s", e.what());
        result = 1;
    }
    rclcpp::shutdown();
    return result;
}
