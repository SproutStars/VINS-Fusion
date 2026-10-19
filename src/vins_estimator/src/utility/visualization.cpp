/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <vins_common/ros_utils.hpp>

#include "visualization.h"
#include "runtime_utils.h"


using namespace Eigen;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odometry;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_latest_odometry;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path;
rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_point_cloud;
rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_margin_cloud;
rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_key_poses;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_camera_pose;
rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_camera_pose_visual;
nav_msgs::msg::Path path;

rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_keyframe_pose;
rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_keyframe_point;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_extrinsic;

rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_image_track;

CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
static double sum_of_path = 0;
static Vector3d last_path(0.0, 0.0, 0.0);

size_t pub_counter = 0;
std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

namespace
{
std::ofstream result_file;
size_t result_rows = 0;
vins::BoundedHistory<geometry_msgs::msg::PoseStamped> path_history;
vins::FrameRateGate path_gate;
double path_publish_rate = 1.0;

template<class T>
bool hasSubscribers(const std::shared_ptr<rclcpp::Publisher<T>> &publisher)
{
    return publisher && (publisher->get_subscription_count() != 0 ||
                         publisher->get_intra_process_subscription_count() != 0);
}
}  // namespace

void resetVisualization()
{
    path.poses.clear();
    path_history.clear();
    // Permit immediate publication after a restart, even at the same timestamp.
    path_gate.reset(path_publish_rate);
}

bool trackImageRequested() { return hasSubscribers(pub_image_track); }

void registerPub(const rclcpp::Node::SharedPtr &n)
{
    const int path_capacity = n->declare_parameter<int>("path_max_poses", 2000);
    path_publish_rate = n->declare_parameter<double>("path_publish_rate", 1.0);
    if (path_capacity < 0) throw std::invalid_argument("path_max_poses must be nonnegative");
    path_history.reset(static_cast<size_t>(path_capacity));
    path_gate.reset(path_publish_rate);
    result_file.open(VINS_RESULT_PATH, std::ios::app);
    if (!result_file) throw std::runtime_error("Cannot open trajectory: " + VINS_RESULT_PATH);
    result_file.setf(std::ios::fixed, std::ios::floatfield);
    result_rows = 0;
    pub_latest_odometry = n->create_publisher<nav_msgs::msg::Odometry>("~/imu_propagate", rclcpp::QoS(100));
    pub_path = n->create_publisher<nav_msgs::msg::Path>("~/path", rclcpp::QoS(1));
    pub_odometry = n->create_publisher<nav_msgs::msg::Odometry>("~/odometry", rclcpp::QoS(100));
    pub_point_cloud = n->create_publisher<sensor_msgs::msg::PointCloud>("~/point_cloud", rclcpp::QoS(100));
    pub_margin_cloud = n->create_publisher<sensor_msgs::msg::PointCloud>("~/margin_cloud", rclcpp::QoS(100));
    pub_key_poses = n->create_publisher<visualization_msgs::msg::Marker>("~/key_poses", rclcpp::QoS(100));
    pub_camera_pose = n->create_publisher<nav_msgs::msg::Odometry>("~/camera_pose", rclcpp::QoS(100));
    pub_camera_pose_visual = n->create_publisher<visualization_msgs::msg::MarkerArray>("~/camera_pose_visual", rclcpp::QoS(100));
    pub_keyframe_pose = n->create_publisher<nav_msgs::msg::Odometry>("~/keyframe_pose", rclcpp::QoS(100));
    pub_keyframe_point = n->create_publisher<sensor_msgs::msg::PointCloud>("~/keyframe_point", rclcpp::QoS(100));
    pub_extrinsic = n->create_publisher<nav_msgs::msg::Odometry>("~/extrinsic", rclcpp::QoS(100));
    pub_image_track = n->create_publisher<sensor_msgs::msg::Image>("~/image_track", rclcpp::QoS(1));

    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(n);
    cameraposevisual.setScale(0.1);
    cameraposevisual.setLineWidth(0.01);
}

void unregisterPub()
{
    result_file.close();
    resetVisualization();
    tf_broadcaster.reset();
    pub_odometry.reset();
    pub_latest_odometry.reset();
    pub_path.reset();
    pub_point_cloud.reset();
    pub_margin_cloud.reset();
    pub_key_poses.reset();
    pub_camera_pose.reset();
    pub_camera_pose_visual.reset();
    pub_keyframe_pose.reset();
    pub_keyframe_point.reset();
    pub_extrinsic.reset();
    pub_image_track.reset();
}

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, double t)
{
    if (!hasSubscribers(pub_latest_odometry)) return;
    nav_msgs::msg::Odometry odometry;
    odometry.header.stamp = vins::stamp(t);
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = P.x();
    odometry.pose.pose.position.y = P.y();
    odometry.pose.pose.position.z = P.z();
    odometry.pose.pose.orientation.x = Q.x();
    odometry.pose.pose.orientation.y = Q.y();
    odometry.pose.pose.orientation.z = Q.z();
    odometry.pose.pose.orientation.w = Q.w();
    odometry.twist.twist.linear.x = V.x();
    odometry.twist.twist.linear.y = V.y();
    odometry.twist.twist.linear.z = V.z();
    pub_latest_odometry->publish(odometry);
}

void pubTrackImage(const cv::Mat &imgTrack, const double t)
{
    if (!trackImageRequested()) return;
    std_msgs::msg::Header header;
    header.frame_id = "world";
    header.stamp = vins::stamp(t);
    sensor_msgs::msg::Image::SharedPtr imgTrackMsg = cv_bridge::CvImage(header, "bgr8", imgTrack).toImageMsg();
    pub_image_track->publish(*imgTrackMsg);
}


void printStatistics(const Estimator &estimator, double t)
{
    if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    //printf("position: %f, %f, %f\r", estimator.Ps[WINDOW_SIZE].x(), estimator.Ps[WINDOW_SIZE].y(), estimator.Ps[WINDOW_SIZE].z());
    RCLCPP_DEBUG_STREAM(vins::logger(), "position: " << estimator.Ps[WINDOW_SIZE].transpose());
    RCLCPP_DEBUG_STREAM(vins::logger(), "orientation: " << estimator.Vs[WINDOW_SIZE].transpose());
    if (ESTIMATE_EXTRINSIC)
    {
        cv::FileStorage fs(EX_CALIB_RESULT_PATH, cv::FileStorage::WRITE);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            //RCLCPP_DEBUG(vins::logger(), "calibration result for camera %d", i);
            RCLCPP_DEBUG_STREAM(vins::logger(), "extirnsic tic: " << estimator.tic[i].transpose());
            RCLCPP_DEBUG_STREAM(vins::logger(), "extrinsic ric: " << Utility::R2ypr(estimator.ric[i]).transpose());

            Eigen::Matrix4d eigen_T = Eigen::Matrix4d::Identity();
            eigen_T.block<3, 3>(0, 0) = estimator.ric[i];
            eigen_T.block<3, 1>(0, 3) = estimator.tic[i];
            cv::Mat cv_T;
            cv::eigen2cv(eigen_T, cv_T);
            if(i == 0)
                fs << "body_T_cam0" << cv_T ;
            else
                fs << "body_T_cam1" << cv_T ;
        }
        fs.release();
    }

    static double sum_of_time = 0;
    static int sum_of_calculation = 0;
    sum_of_time += t;
    sum_of_calculation++;
    RCLCPP_DEBUG(vins::logger(), "vo solver costs: %f ms", t);
    RCLCPP_DEBUG(vins::logger(), "average of time %f ms", sum_of_time / sum_of_calculation);

    sum_of_path += (estimator.Ps[WINDOW_SIZE] - last_path).norm();
    last_path = estimator.Ps[WINDOW_SIZE];
    RCLCPP_DEBUG(vins::logger(), "sum of path %f", sum_of_path);
    if (ESTIMATE_TD)
        RCLCPP_DEBUG(vins::logger(), "td %f", estimator.td);
}

void pubOdometry(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        nav_msgs::msg::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.child_frame_id = "body";
        Quaterniond tmp_Q;
        tmp_Q = Quaterniond(estimator.Rs[WINDOW_SIZE]);
        odometry.pose.pose.position.x = estimator.Ps[WINDOW_SIZE].x();
        odometry.pose.pose.position.y = estimator.Ps[WINDOW_SIZE].y();
        odometry.pose.pose.position.z = estimator.Ps[WINDOW_SIZE].z();
        odometry.pose.pose.orientation.x = tmp_Q.x();
        odometry.pose.pose.orientation.y = tmp_Q.y();
        odometry.pose.pose.orientation.z = tmp_Q.z();
        odometry.pose.pose.orientation.w = tmp_Q.w();
        odometry.twist.twist.linear.x = estimator.Vs[WINDOW_SIZE].x();
        odometry.twist.twist.linear.y = estimator.Vs[WINDOW_SIZE].y();
        odometry.twist.twist.linear.z = estimator.Vs[WINDOW_SIZE].z();
        pub_odometry->publish(odometry);

        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header = header;
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose = odometry.pose.pose;
        path_history.push(pose_stamped);
        if (path_history.size() && hasSubscribers(pub_path) && path_gate.accept(vins::seconds(header.stamp))) {
            path.header = header;
            path.header.frame_id = "world";
            path.poses.assign(path_history.items().begin(), path_history.items().end());
            pub_path->publish(path);
        }

        // write result to file
        auto &foutC = result_file;
        foutC.precision(0);
        foutC << vins::seconds(header.stamp) * 1e9 << ",";
        foutC.precision(5);
        foutC << estimator.Ps[WINDOW_SIZE].x() << ","
              << estimator.Ps[WINDOW_SIZE].y() << ","
              << estimator.Ps[WINDOW_SIZE].z() << ","
              << tmp_Q.w() << ","
              << tmp_Q.x() << ","
              << tmp_Q.y() << ","
              << tmp_Q.z() << ","
              << estimator.Vs[WINDOW_SIZE].x() << ","
              << estimator.Vs[WINDOW_SIZE].y() << ","
              << estimator.Vs[WINDOW_SIZE].z() << ",\n";
        if (++result_rows % 50 == 0) foutC.flush();
        Eigen::Vector3d tmp_T = estimator.Ps[WINDOW_SIZE];
        RCLCPP_DEBUG(vins::logger(), "time: %f, t: %f %f %f q: %f %f %f %f", vins::seconds(header.stamp), tmp_T.x(), tmp_T.y(), tmp_T.z(),
                                                          tmp_Q.w(), tmp_Q.x(), tmp_Q.y(), tmp_Q.z());
    }
}

void pubKeyPoses(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    if (!hasSubscribers(pub_key_poses)) return;
    if (estimator.key_poses.size() == 0)
        return;
    visualization_msgs::msg::Marker key_poses;
    key_poses.header = header;
    key_poses.header.frame_id = "world";
    key_poses.ns = "key_poses";
    key_poses.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    key_poses.action = visualization_msgs::msg::Marker::ADD;
    key_poses.pose.orientation.w = 1.0;
    key_poses.lifetime = builtin_interfaces::msg::Duration{};

    //static int key_poses_id = 0;
    key_poses.id = 0; //key_poses_id++;
    key_poses.scale.x = 0.05;
    key_poses.scale.y = 0.05;
    key_poses.scale.z = 0.05;
    key_poses.color.r = 1.0;
    key_poses.color.a = 1.0;

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        geometry_msgs::msg::Point pose_marker;
        Vector3d correct_pose;
        correct_pose = estimator.key_poses[i];
        pose_marker.x = correct_pose.x();
        pose_marker.y = correct_pose.y();
        pose_marker.z = correct_pose.z();
        key_poses.points.push_back(pose_marker);
    }
    pub_key_poses->publish(key_poses);
}

void pubCameraPose(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    if (!hasSubscribers(pub_camera_pose) && !hasSubscribers(pub_camera_pose_visual)) return;
    int idx2 = WINDOW_SIZE - 1;

    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        int i = idx2;
        Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
        Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[0]);

        nav_msgs::msg::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();

        pub_camera_pose->publish(odometry);

        cameraposevisual.reset();
        cameraposevisual.add_pose(P, R);
        if(STEREO)
        {
            Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[1];
            Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[1]);
            cameraposevisual.add_pose(P, R);
        }
        cameraposevisual.publish_by(pub_camera_pose_visual, odometry.header);
    }
}


void pubPointCloud(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    if (!hasSubscribers(pub_point_cloud) && !hasSubscribers(pub_margin_cloud)) return;
    sensor_msgs::msg::PointCloud point_cloud, loop_point_cloud;
    point_cloud.header = header;
    loop_point_cloud.header = header;


    for (auto &it_per_id : estimator.f_manager.feature)
    {
        int used_num;
        used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        if (it_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id.solve_flag != 1)
            continue;
        int imu_i = it_per_id.start_frame;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
        Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];

        geometry_msgs::msg::Point32 p;
        p.x = w_pts_i(0);
        p.y = w_pts_i(1);
        p.z = w_pts_i(2);
        point_cloud.points.push_back(p);
    }
    pub_point_cloud->publish(point_cloud);


    // pub margined potin
    sensor_msgs::msg::PointCloud margin_cloud;
    margin_cloud.header = header;

    for (auto &it_per_id : estimator.f_manager.feature)
    { 
        int used_num;
        used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        //if (it_per_id->start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id->solve_flag != 1)
        //        continue;

        if (it_per_id.start_frame == 0 && it_per_id.feature_per_frame.size() <= 2 
            && it_per_id.solve_flag == 1 )
        {
            int imu_i = it_per_id.start_frame;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
            Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];

            geometry_msgs::msg::Point32 p;
            p.x = w_pts_i(0);
            p.y = w_pts_i(1);
            p.z = w_pts_i(2);
            margin_cloud.points.push_back(p);
        }
    }
    pub_margin_cloud->publish(margin_cloud);
}


void pubTF(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    if( estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    geometry_msgs::msg::TransformStamped body, camera;
    body.header = header;
    body.header.frame_id = "world";
    body.child_frame_id = "body";
    body.transform.translation.x = estimator.Ps[WINDOW_SIZE].x();
    body.transform.translation.y = estimator.Ps[WINDOW_SIZE].y();
    body.transform.translation.z = estimator.Ps[WINDOW_SIZE].z();
    const Quaterniond q_body(estimator.Rs[WINDOW_SIZE]);
    body.transform.rotation.x = q_body.x();
    body.transform.rotation.y = q_body.y();
    body.transform.rotation.z = q_body.z();
    body.transform.rotation.w = q_body.w();
    camera.header = header;
    camera.header.frame_id = "body";
    camera.child_frame_id = "camera";
    camera.transform.translation.x = estimator.tic[0].x();
    camera.transform.translation.y = estimator.tic[0].y();
    camera.transform.translation.z = estimator.tic[0].z();
    const Quaterniond q_camera(estimator.ric[0]);
    camera.transform.rotation.x = q_camera.x();
    camera.transform.rotation.y = q_camera.y();
    camera.transform.rotation.z = q_camera.z();
    camera.transform.rotation.w = q_camera.w();
    tf_broadcaster->sendTransform(std::vector<geometry_msgs::msg::TransformStamped>{body, camera});

    nav_msgs::msg::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = estimator.tic[0].x();
    odometry.pose.pose.position.y = estimator.tic[0].y();
    odometry.pose.pose.position.z = estimator.tic[0].z();
    Quaterniond tmp_q{estimator.ric[0]};
    odometry.pose.pose.orientation.x = tmp_q.x();
    odometry.pose.pose.orientation.y = tmp_q.y();
    odometry.pose.pose.orientation.z = tmp_q.z();
    odometry.pose.pose.orientation.w = tmp_q.w();
    pub_extrinsic->publish(odometry);

}

void pubKeyframe(const Estimator &estimator)
{
    if (!hasSubscribers(pub_keyframe_pose) && !hasSubscribers(pub_keyframe_point)) return;
    // pub camera pose, 2D-3D points of keyframe
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR && estimator.marginalization_flag == 0)
    {
        int i = WINDOW_SIZE - 2;
        //Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
        Vector3d P = estimator.Ps[i];
        Quaterniond R = Quaterniond(estimator.Rs[i]);

        nav_msgs::msg::Odometry odometry;
        odometry.header.stamp = vins::stamp(estimator.Headers[WINDOW_SIZE - 2]);
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();
        //printf("time: %f t: %f %f %f r: %f %f %f %f\n", vins::seconds(odometry.header.stamp), P.x(), P.y(), P.z(), R.w(), R.x(), R.y(), R.z());

        pub_keyframe_pose->publish(odometry);


        sensor_msgs::msg::PointCloud point_cloud;
        point_cloud.header.stamp = vins::stamp(estimator.Headers[WINDOW_SIZE - 2]);
        point_cloud.header.frame_id = "world";
        for (auto &it_per_id : estimator.f_manager.feature)
        {
            int frame_size = it_per_id.feature_per_frame.size();
            if(it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.start_frame + frame_size - 1 >= WINDOW_SIZE - 2 && it_per_id.solve_flag == 1)
            {

                int imu_i = it_per_id.start_frame;
                Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
                Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0])
                                      + estimator.Ps[imu_i];
                geometry_msgs::msg::Point32 p;
                p.x = w_pts_i(0);
                p.y = w_pts_i(1);
                p.z = w_pts_i(2);
                point_cloud.points.push_back(p);

                int imu_j = WINDOW_SIZE - 2 - it_per_id.start_frame;
                sensor_msgs::msg::ChannelFloat32 p_2d;
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.x());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.y());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.x());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.y());
                p_2d.values.push_back(it_per_id.feature_id);
                point_cloud.channels.push_back(p_2d);
            }

        }
        pub_keyframe_point->publish(point_cloud);
    }
}