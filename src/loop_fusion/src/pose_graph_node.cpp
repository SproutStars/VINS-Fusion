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

#include <std_srvs/srv/trigger.hpp>

#include <vector>
#include <vins_common/parameters.hpp>
#include <vins_common/ros_utils.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <std_msgs/msg/bool.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <iostream>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <mutex>
#include <queue>
#include <thread>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "keyframe.h"
#include "utility/tic_toc.h"
#include "pose_graph.h"
#include "utility/CameraPoseVisualization.h"
#include "parameters.h"
#define SKIP_FIRST_CNT 10
using namespace std;

queue<sensor_msgs::msg::Image::ConstSharedPtr> image_buf;
queue<sensor_msgs::msg::PointCloud::ConstSharedPtr> point_buf;
queue<nav_msgs::msg::Odometry::ConstSharedPtr> pose_buf;
queue<Eigen::Vector3d> odometry_buf;
std::mutex m_buf;
std::mutex m_process;
int frame_index  = 0;
int sequence = 1;
std::unique_ptr<PoseGraph> posegraph;
int skip_first_cnt = 0;
int SKIP_CNT;
int skip_cnt = 0;
bool load_flag = 0;
bool start_flag = 0;
double SKIP_DIS = 0;

int VISUALIZATION_SHIFT_X;
int VISUALIZATION_SHIFT_Y;
int ROW;
int COL;
int DEBUG_IMAGE;

camodocal::CameraPtr m_camera;
Eigen::Vector3d tic = Eigen::Vector3d::Zero();
Eigen::Matrix3d qic = Eigen::Matrix3d::Identity();
rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_match_img;
rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_camera_pose_visual;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odometry_rect;

std::string BRIEF_PATTERN_FILE;
std::string POSE_GRAPH_SAVE_PATH;
std::string VINS_RESULT_PATH;
CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
Eigen::Vector3d last_t(-100, -100, -100);
double last_image_time = -1;

rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_point_cloud;
rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_margin_cloud;

void new_sequence()
{
    printf("new sequence\n");
    sequence++;
    printf("sequence cnt %d \n", sequence);
    if (sequence > 5)
    {
        RCLCPP_WARN(vins::logger(), "only support 5 sequences since it's boring to copy code for more sequences.");
        throw std::runtime_error("VINS invariant violated");
    }
    posegraph->posegraph_visualization->reset();
    posegraph->publish();
    m_buf.lock();
    while(!image_buf.empty())
        image_buf.pop();
    while(!point_buf.empty())
        point_buf.pop();
    while(!pose_buf.empty())
        pose_buf.pop();
    while(!odometry_buf.empty())
        odometry_buf.pop();
    m_buf.unlock();
}

void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr &image_msg)
{
    //RCLCPP_INFO(vins::logger(), "image_callback!");
    m_buf.lock();
    image_buf.push(image_msg);
    m_buf.unlock();
    //printf(" image time %f \n", vins::seconds(image_msg->header.stamp));

    // detect unstable camera stream
    if (last_image_time == -1)
        last_image_time = vins::seconds(image_msg->header.stamp);
    else if (vins::seconds(image_msg->header.stamp) - last_image_time > 1.0 || vins::seconds(image_msg->header.stamp) < last_image_time)
    {
        RCLCPP_WARN(vins::logger(), "image discontinue! detect a new sequence!");
        new_sequence();
    }
    last_image_time = vins::seconds(image_msg->header.stamp);
}

void point_callback(const sensor_msgs::msg::PointCloud::ConstSharedPtr &point_msg)
{
    //RCLCPP_INFO(vins::logger(), "point_callback!");
    m_buf.lock();
    point_buf.push(point_msg);
    m_buf.unlock();
    /*
    for (unsigned int i = 0; i < point_msg->points.size(); i++)
    {
        printf("%d, 3D point: %f, %f, %f 2D point %f, %f \n",i , point_msg->points[i].x, 
                                                     point_msg->points[i].y,
                                                     point_msg->points[i].z,
                                                     point_msg->channels[i].values[0],
                                                     point_msg->channels[i].values[1]);
    }
    */
    // for visualization
    sensor_msgs::msg::PointCloud point_cloud;
    point_cloud.header = point_msg->header;
    for (unsigned int i = 0; i < point_msg->points.size(); i++)
    {
        cv::Point3f p_3d;
        p_3d.x = point_msg->points[i].x;
        p_3d.y = point_msg->points[i].y;
        p_3d.z = point_msg->points[i].z;
        Eigen::Vector3d tmp = posegraph->r_drift * Eigen::Vector3d(p_3d.x, p_3d.y, p_3d.z) + posegraph->t_drift;
        geometry_msgs::msg::Point32 p;
        p.x = tmp(0);
        p.y = tmp(1);
        p.z = tmp(2);
        point_cloud.points.push_back(p);
    }
    pub_point_cloud->publish(point_cloud);
}

// only for visualization
void margin_point_callback(const sensor_msgs::msg::PointCloud::ConstSharedPtr &point_msg)
{
    sensor_msgs::msg::PointCloud point_cloud;
    point_cloud.header = point_msg->header;
    for (unsigned int i = 0; i < point_msg->points.size(); i++)
    {
        cv::Point3f p_3d;
        p_3d.x = point_msg->points[i].x;
        p_3d.y = point_msg->points[i].y;
        p_3d.z = point_msg->points[i].z;
        Eigen::Vector3d tmp = posegraph->r_drift * Eigen::Vector3d(p_3d.x, p_3d.y, p_3d.z) + posegraph->t_drift;
        geometry_msgs::msg::Point32 p;
        p.x = tmp(0);
        p.y = tmp(1);
        p.z = tmp(2);
        point_cloud.points.push_back(p);
    }
    pub_margin_cloud->publish(point_cloud);
}

void pose_callback(const nav_msgs::msg::Odometry::ConstSharedPtr &pose_msg)
{
    //RCLCPP_INFO(vins::logger(), "pose_callback!");
    m_buf.lock();
    pose_buf.push(pose_msg);
    m_buf.unlock();
    /*
    printf("pose t: %f, %f, %f   q: %f, %f, %f %f \n", pose_msg->pose.pose.position.x,
                                                       pose_msg->pose.pose.position.y,
                                                       pose_msg->pose.pose.position.z,
                                                       pose_msg->pose.pose.orientation.w,
                                                       pose_msg->pose.pose.orientation.x,
                                                       pose_msg->pose.pose.orientation.y,
                                                       pose_msg->pose.pose.orientation.z);
    */
}

void vio_callback(const nav_msgs::msg::Odometry::ConstSharedPtr &pose_msg)
{
    //RCLCPP_INFO(vins::logger(), "vio_callback!");
    Vector3d vio_t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
    Quaterniond vio_q;
    vio_q.w() = pose_msg->pose.pose.orientation.w;
    vio_q.x() = pose_msg->pose.pose.orientation.x;
    vio_q.y() = pose_msg->pose.pose.orientation.y;
    vio_q.z() = pose_msg->pose.pose.orientation.z;

    vio_t = posegraph->w_r_vio * vio_t + posegraph->w_t_vio;
    vio_q = posegraph->w_r_vio *  vio_q;

    vio_t = posegraph->r_drift * vio_t + posegraph->t_drift;
    vio_q = posegraph->r_drift * vio_q;

    nav_msgs::msg::Odometry odometry;
    odometry.header = pose_msg->header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = vio_t.x();
    odometry.pose.pose.position.y = vio_t.y();
    odometry.pose.pose.position.z = vio_t.z();
    odometry.pose.pose.orientation.x = vio_q.x();
    odometry.pose.pose.orientation.y = vio_q.y();
    odometry.pose.pose.orientation.z = vio_q.z();
    odometry.pose.pose.orientation.w = vio_q.w();
    pub_odometry_rect->publish(odometry);

    Vector3d vio_t_cam;
    Quaterniond vio_q_cam;
    vio_t_cam = vio_t + vio_q * tic;
    vio_q_cam = vio_q * qic;        

    cameraposevisual.reset();
    cameraposevisual.add_pose(vio_t_cam, vio_q_cam);
    cameraposevisual.publish_by(pub_camera_pose_visual, pose_msg->header);


}

void extrinsic_callback(const nav_msgs::msg::Odometry::ConstSharedPtr &pose_msg)
{
    m_process.lock();
    tic = Vector3d(pose_msg->pose.pose.position.x,
                   pose_msg->pose.pose.position.y,
                   pose_msg->pose.pose.position.z);
    qic = Quaterniond(pose_msg->pose.pose.orientation.w,
                      pose_msg->pose.pose.orientation.x,
                      pose_msg->pose.pose.orientation.y,
                      pose_msg->pose.pose.orientation.z).toRotationMatrix();
    m_process.unlock();
}

void process()
{
    while (rclcpp::ok())
    {
        sensor_msgs::msg::Image::ConstSharedPtr image_msg = NULL;
        sensor_msgs::msg::PointCloud::ConstSharedPtr point_msg = NULL;
        nav_msgs::msg::Odometry::ConstSharedPtr pose_msg = NULL;

        // find out the messages with same time stamp
        m_buf.lock();
        if(!image_buf.empty() && !point_buf.empty() && !pose_buf.empty())
        {
            if (vins::seconds(image_buf.front()->header.stamp) > vins::seconds(pose_buf.front()->header.stamp))
            {
                pose_buf.pop();
                printf("throw pose at beginning\n");
            }
            else if (vins::seconds(image_buf.front()->header.stamp) > vins::seconds(point_buf.front()->header.stamp))
            {
                point_buf.pop();
                printf("throw point at beginning\n");
            }
            else if (vins::seconds(image_buf.back()->header.stamp) >= vins::seconds(pose_buf.front()->header.stamp) 
                && vins::seconds(point_buf.back()->header.stamp) >= vins::seconds(pose_buf.front()->header.stamp))
            {
                pose_msg = pose_buf.front();
                pose_buf.pop();
                while (!pose_buf.empty())
                    pose_buf.pop();
                while (vins::seconds(image_buf.front()->header.stamp) < vins::seconds(pose_msg->header.stamp))
                    image_buf.pop();
                image_msg = image_buf.front();
                image_buf.pop();

                while (vins::seconds(point_buf.front()->header.stamp) < vins::seconds(pose_msg->header.stamp))
                    point_buf.pop();
                point_msg = point_buf.front();
                point_buf.pop();
            }
        }
        m_buf.unlock();

        if (pose_msg != NULL)
        {
            //printf(" pose time %f \n", vins::seconds(pose_msg->header.stamp));
            //printf(" point time %f \n", vins::seconds(point_msg->header.stamp));
            //printf(" image time %f \n", vins::seconds(image_msg->header.stamp));
            // skip fisrt few
            if (skip_first_cnt < SKIP_FIRST_CNT)
            {
                skip_first_cnt++;
                continue;
            }

            if (skip_cnt < SKIP_CNT)
            {
                skip_cnt++;
                continue;
            }
            else
            {
                skip_cnt = 0;
            }

            cv_bridge::CvImageConstPtr ptr;
            if (image_msg->encoding == "8UC1")
            {
                sensor_msgs::msg::Image img;
                img.header = image_msg->header;
                img.height = image_msg->height;
                img.width = image_msg->width;
                img.is_bigendian = image_msg->is_bigendian;
                img.step = image_msg->step;
                img.data = image_msg->data;
                img.encoding = "mono8";
                ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
            }
            else
                ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);
            
            cv::Mat image = ptr->image;
            // build keyframe
            Vector3d T = Vector3d(pose_msg->pose.pose.position.x,
                                  pose_msg->pose.pose.position.y,
                                  pose_msg->pose.pose.position.z);
            Matrix3d R = Quaterniond(pose_msg->pose.pose.orientation.w,
                                     pose_msg->pose.pose.orientation.x,
                                     pose_msg->pose.pose.orientation.y,
                                     pose_msg->pose.pose.orientation.z).toRotationMatrix();
            if((T - last_t).norm() > SKIP_DIS)
            {
                vector<cv::Point3f> point_3d; 
                vector<cv::Point2f> point_2d_uv; 
                vector<cv::Point2f> point_2d_normal;
                vector<double> point_id;

                for (unsigned int i = 0; i < point_msg->points.size(); i++)
                {
                    cv::Point3f p_3d;
                    p_3d.x = point_msg->points[i].x;
                    p_3d.y = point_msg->points[i].y;
                    p_3d.z = point_msg->points[i].z;
                    point_3d.push_back(p_3d);

                    cv::Point2f p_2d_uv, p_2d_normal;
                    double p_id;
                    p_2d_normal.x = point_msg->channels[i].values[0];
                    p_2d_normal.y = point_msg->channels[i].values[1];
                    p_2d_uv.x = point_msg->channels[i].values[2];
                    p_2d_uv.y = point_msg->channels[i].values[3];
                    p_id = point_msg->channels[i].values[4];
                    point_2d_normal.push_back(p_2d_normal);
                    point_2d_uv.push_back(p_2d_uv);
                    point_id.push_back(p_id);

                    //printf("u %f, v %f \n", p_2d_uv.x, p_2d_uv.y);
                }

                KeyFrame* keyframe = new KeyFrame(vins::seconds(pose_msg->header.stamp), frame_index, T, R, image,
                                   point_3d, point_2d_uv, point_2d_normal, point_id, sequence);   
                m_process.lock();
                start_flag = 1;
                posegraph->addKeyFrame(keyframe, 1);
                m_process.unlock();
                frame_index++;
                last_t = T;
            }
        }
        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto n = std::make_shared<rclcpp::Node>("loop_fusion");
    posegraph = std::make_unique<PoseGraph>();
    posegraph->registerPub(n);
    
    VISUALIZATION_SHIFT_X = 0;
    VISUALIZATION_SHIFT_Y = 0;
    SKIP_CNT = 0;
    SKIP_DIS = 0;

    std::string config_file;
    try { config_file = vins::configFile(n, argc, argv); }
    catch (const std::exception &e) {
        RCLCPP_ERROR(n->get_logger(), "%s", e.what());
        posegraph.reset();
        rclcpp::shutdown(); return 1;
    }

    vins::Config config;
    try { config = vins::Config::fromFile(config_file); }
    catch (const std::exception &e) {
        RCLCPP_ERROR(n->get_logger(), "%s", e.what());
        posegraph.reset();
        rclcpp::shutdown(); return 1;
    }

    cameraposevisual.setScale(0.1);
    cameraposevisual.setLineWidth(0.01);

    std::string IMAGE_TOPIC;
    int LOAD_PREVIOUS_POSE_GRAPH;

    ROW = config.getInt("image_height");
    COL = config.getInt("image_width");
    std::string pkg_path = ament_index_cpp::get_package_share_directory("loop_fusion");
    string vocabulary_file = pkg_path + "/support_files/brief_k10L6.bin";
    cout << "vocabulary_file" << vocabulary_file << endl;
    posegraph->loadVocabulary(vocabulary_file);

    BRIEF_PATTERN_FILE = pkg_path + "/support_files/brief_pattern.yml";
    cout << "BRIEF_PATTERN_FILE" << BRIEF_PATTERN_FILE << endl;

    std::string cam0Calib = config.getString("cam0_calib");
    std::string cam0Path = vins::resolvePath(config_file, cam0Calib);
    printf("cam calib path: %s\n", cam0Path.c_str());
    m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(cam0Path.c_str());

    IMAGE_TOPIC = config.getString("image0_topic");
    POSE_GRAPH_SAVE_PATH = config.getString("pose_graph_save_path");
    VINS_RESULT_PATH = config.getString("output_path");
    DEBUG_IMAGE = config.getInt("save_image");

    LOAD_PREVIOUS_POSE_GRAPH = config.getInt("load_previous_pose_graph");
    IMAGE_TOPIC = n->declare_parameter<std::string>("image0_topic", IMAGE_TOPIC);
    POSE_GRAPH_SAVE_PATH = n->declare_parameter<std::string>("pose_graph_save_path", POSE_GRAPH_SAVE_PATH);
    VINS_RESULT_PATH = n->declare_parameter<std::string>("output_path", VINS_RESULT_PATH);
    POSE_GRAPH_SAVE_PATH = vins::resolvePath(config_file, POSE_GRAPH_SAVE_PATH);
    VINS_RESULT_PATH = vins::resolvePath(config_file, VINS_RESULT_PATH);
    std::filesystem::create_directories(POSE_GRAPH_SAVE_PATH);
    std::filesystem::create_directories(VINS_RESULT_PATH);
    // Legacy save code concatenates filenames, so keep a trailing separator.
    POSE_GRAPH_SAVE_PATH += "/";
    VINS_RESULT_PATH += "/vio_loop.csv";
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    int USE_IMU = config.getInt("imu");
    posegraph->setIMUFlag(USE_IMU);

    if (LOAD_PREVIOUS_POSE_GRAPH)
    {
        printf("load pose graph\n");
        m_process.lock();
        posegraph->loadPoseGraph();
        m_process.unlock();
        printf("load pose graph finish\n");
        load_flag = 1;
    }
    else
    {
        printf("no previous pose graph\n");
        load_flag = 1;
    }

    auto sub_vio = n->create_subscription<nav_msgs::msg::Odometry>("/vins_estimator/odometry", rclcpp::QoS(100), vio_callback);
    auto sub_image = n->create_subscription<sensor_msgs::msg::Image>(IMAGE_TOPIC, rclcpp::SensorDataQoS().keep_last(2000), image_callback);
    auto sub_pose = n->create_subscription<nav_msgs::msg::Odometry>("/vins_estimator/keyframe_pose", rclcpp::QoS(100), pose_callback);
    auto sub_extrinsic = n->create_subscription<nav_msgs::msg::Odometry>("/vins_estimator/extrinsic", rclcpp::QoS(100), extrinsic_callback);
    auto sub_point = n->create_subscription<sensor_msgs::msg::PointCloud>("/vins_estimator/keyframe_point", rclcpp::QoS(100), point_callback);
    auto sub_margin_point = n->create_subscription<sensor_msgs::msg::PointCloud>("/vins_estimator/margin_cloud", rclcpp::QoS(100), margin_point_callback);

    pub_match_img = n->create_publisher<sensor_msgs::msg::Image>("~/match_image", rclcpp::QoS(100));
    pub_camera_pose_visual = n->create_publisher<visualization_msgs::msg::MarkerArray>("~/camera_pose_visual", rclcpp::QoS(100));
    pub_point_cloud = n->create_publisher<sensor_msgs::msg::PointCloud>("~/point_cloud_loop_rect", rclcpp::QoS(100));
    pub_margin_cloud = n->create_publisher<sensor_msgs::msg::PointCloud>("~/margin_cloud_loop_rect", rclcpp::QoS(100));
    pub_odometry_rect = n->create_publisher<nav_msgs::msg::Odometry>("~/odometry_rect", rclcpp::QoS(100));

    std::thread measurement_process;
    auto save_service = n->create_service<std_srvs::srv::Trigger>("~/save_pose_graph",
        [](const std_srvs::srv::Trigger::Request::SharedPtr,
           std_srvs::srv::Trigger::Response::SharedPtr response) {
            std::lock_guard<std::mutex> lock(m_process);
            posegraph->savePoseGraph();
            response->success = true;
            response->message = "Pose graph saved";
        });
    auto sequence_service = n->create_service<std_srvs::srv::Trigger>("~/new_sequence",
        [](const std_srvs::srv::Trigger::Request::SharedPtr,
           std_srvs::srv::Trigger::Response::SharedPtr response) {
            std::lock_guard<std::mutex> lock(m_process);
            new_sequence();
            response->success = true;
            response->message = "New sequence started";
        });

    measurement_process = std::thread(process);

    
    rclcpp::spin(n);
    rclcpp::shutdown();
    measurement_process.join();
    posegraph.reset();
    pub_match_img.reset(); pub_camera_pose_visual.reset();
    pub_point_cloud.reset(); pub_margin_cloud.reset(); pub_odometry_rect.reset();
    return 0;
}
