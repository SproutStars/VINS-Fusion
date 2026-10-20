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
    ~VinsNode() override { stopTracking(); estimator_.reset(); last_frame_ = {}; unregisterPub(); }

    void start(int argc, char **argv)
    {
        readParameters(vins::configFile(shared_from_this(), argc, argv),
            declare_parameter<std::string>("output_path", ""));
        IMAGE0_TOPIC = declare_parameter<std::string>("image0_topic", IMAGE0_TOPIC);
        IMAGE1_TOPIC = declare_parameter<std::string>("image1_topic", IMAGE1_TOPIC);
        IMU_TOPIC = declare_parameter<std::string>("imu_topic", IMU_TOPIC);
        sync_tolerance_ = declare_parameter<double>("stereo_sync_tolerance", 0.003);
        max_queue_ = declare_parameter<int>("image_queue_size", 100);
        if (max_queue_ < 1 || sync_tolerance_ < 0.0)
            throw std::invalid_argument("Invalid queue size or stereo synchronization tolerance");
        tracking_queue_size_ = declare_parameter<int>("tracking_queue_size", 5);
        const int opencv_threads = declare_parameter<int>("opencv_threads", 1);
        if (tracking_queue_size_ < 1 || opencv_threads < 0)
            throw std::invalid_argument("tracking_queue_size must be positive and opencv_threads nonnegative");
        cv::setNumThreads(opencv_threads);
        parameter_callback_ = add_on_set_parameters_callback(
            [this](const std::vector<rclcpp::Parameter> & parameters) {
                rcl_interfaces::msg::SetParametersResult result;
                result.successful = true;
                for (const auto & parameter : parameters) {
                    const auto & name = parameter.get_name();
                    if (name == "config_file" || name == "output_path" ||
                        name == "image0_topic" || name == "image1_topic" ||
                        name == "imu_topic") {
                        result.successful = false;
                        result.reason = name + " is fixed after startup; stop and relaunch the node";
                        return result;
                    }
                    if (name == "tracking_queue_size" && parameter.as_int() < 1) {
                        result.successful = false;
                        result.reason = "tracking_queue_size must be positive";
                        return result;
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
                    } else if (parameter.get_name() == "opencv_threads") {
                        cv::setNumThreads(static_cast<int>(parameter.as_int()));
                    }
                }
                return result;
            });
        registerPub(shared_from_this());
        estimator_ = std::make_unique<Estimator>();
        estimator_->setParameter();
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
    };
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
        try {
            estimator_->inputImage(vins::seconds(frame.left->header.stamp), decode(frame.left),
                                   frame.right ? decode(frame.right) : cv::Mat());
        } catch (const cv::Exception &e) {
            RCLCPP_ERROR(get_logger(), "Image conversion/tracking failed: %s", e.what());
        } catch (const cv_bridge::Exception &e) {
            RCLCPP_ERROR(get_logger(), "Image conversion failed: %s", e.what());
        }
        last_frame_ = std::move(frame);
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
        {
            std::lock_guard<std::mutex> lock(tracking_mutex_);
            if (tracking_frames_.size() >= static_cast<size_t>(tracking_queue_size_)) {
                tracking_frames_.pop_front();
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                                    "Tracking queue overflow: dropping stale image pair");
            }
            tracking_frames_.push_back(std::move(frame));
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
            FramePair frame{images0_.front(), {}};
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
    } catch (const std::exception &e) {
        RCLCPP_ERROR(vins::logger(), "%s", e.what());
        result = 1;
    }
    rclcpp::shutdown();
    return result;
}
