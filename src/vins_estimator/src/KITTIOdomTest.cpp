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

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <string>
#include <vins_common/ros_utils.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include "estimator/estimator.h"
#include "utility/visualization.h"

using namespace std;
using namespace Eigen;


Eigen::Matrix3d c1Rc0, c0Rc1;
Eigen::Vector3d c1Tc0, c0Tc1;

int run(int argc, char** argv)
{

    auto n = std::make_shared<rclcpp::Node>("vins_estimator");

	auto pubLeftImage = n->create_publisher<sensor_msgs::msg::Image>("/leftImage", rclcpp::QoS(100));
	auto pubRightImage = n->create_publisher<sensor_msgs::msg::Image>("/rightImage", rclcpp::QoS(100));

	const auto args = rclcpp::remove_ros_arguments(argc, argv);
	if(args.size() != 3)
	{
		printf("please intput: ros2 run vins kitti_odom_test [config file] [data folder] \n"
			   "for example: ros2 run vins kitti_odom_test "
			   "/path/to/VINS-Fusion/src/vins_estimator/config/kitti_odom/kitti_config00-02.yaml "
			   "/media/tony-ws1/disk_D/kitti/odometry/sequences/00/ \n");
		return 1;
	}

	string config_file = args[1].c_str();
	printf("config_file: %s\n", args[1].c_str());
	string sequence = args[2].c_str();
	printf("read sequence: %s\n", args[2].c_str());
	string dataPath = sequence + "/";

	readParameters(config_file);
    MULTIPLE_THREAD = 0; // Offline output must use the pose for this frame.
    registerPub(n);
    Estimator estimator;
	estimator.setParameter();

	// load image list
	FILE* file;
	file = std::fopen((dataPath + "times.txt").c_str() , "r");
	if(file == NULL){
	    printf("cannot find file: %stimes.txt\n", dataPath.c_str());
	    throw std::runtime_error("VINS invariant violated");
	    return 0;          
	}
	double imageTime;
	vector<double> imageTimeList;
	while ( fscanf(file, "%lf", &imageTime) == 1)
	{
	    imageTimeList.push_back(imageTime);
	}
	std::fclose(file);

	string leftImagePath, rightImagePath;
	cv::Mat imLeft, imRight;
	FILE* outFile;
	outFile = fopen((OUTPUT_FOLDER + "/vio.txt").c_str(),"w");
	if(outFile == NULL)
		printf("Output path dosen't exist: %s\n", OUTPUT_FOLDER.c_str());

	for (size_t i = 0; i < imageTimeList.size(); i++)
	{	
		if(rclcpp::ok())
		{
			printf("\nprocess image %d\n", (int)i);
			stringstream ss;
			ss << setfill('0') << setw(6) << i;
			leftImagePath = dataPath + "image_0/" + ss.str() + ".png";
			rightImagePath = dataPath + "image_1/" + ss.str() + ".png";
			//printf("%lu  %f \n", i, imageTimeList[i]);
			//printf("%s\n", leftImagePath.c_str() );
			//printf("%s\n", rightImagePath.c_str() );

			imLeft = cv::imread(leftImagePath, cv::IMREAD_GRAYSCALE );
            if (imLeft.empty()) throw std::runtime_error("Cannot read image: " + leftImagePath);
			sensor_msgs::msg::Image::SharedPtr imLeftMsg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", imLeft).toImageMsg();
			imLeftMsg->header.stamp = vins::stamp(imageTimeList[i]);
			pubLeftImage->publish(*imLeftMsg);

			imRight = cv::imread(rightImagePath, cv::IMREAD_GRAYSCALE );
            if (imRight.empty()) throw std::runtime_error("Cannot read image: " + rightImagePath);
			sensor_msgs::msg::Image::SharedPtr imRightMsg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", imRight).toImageMsg();
			imRightMsg->header.stamp = vins::stamp(imageTimeList[i]);
			pubRightImage->publish(*imRightMsg);


			estimator.inputImage(imageTimeList[i], imLeft, imRight);
			
			Eigen::Matrix<double, 4, 4> pose;
			estimator.getPoseInWorldFrame(pose);
			if(outFile != NULL)
				fprintf (outFile, "%f %f %f %f %f %f %f %f %f %f %f %f \n",pose(0,0), pose(0,1), pose(0,2),pose(0,3),
																	       pose(1,0), pose(1,1), pose(1,2),pose(1,3),
																	       pose(2,0), pose(2,1), pose(2,2),pose(2,3));
			
			//cv::imshow("leftImage", imLeft);
			//cv::imshow("rightImage", imRight);
			//cv::waitKey(2);
		}
		else
			break;
	}
	if(outFile != NULL)
		fclose (outFile);
	return 0;
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    int result = 0;
    try { result = run(argc, argv); }
    catch (const std::exception &e) {
        RCLCPP_ERROR(vins::logger(), "%s", e.what());
        result = 1;
    }
    unregisterPub();
    rclcpp::shutdown();
    return result;
}
