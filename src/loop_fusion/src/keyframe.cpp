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

#include <vins_common/ros_utils.hpp>

#include "keyframe.h"

template <typename Derived> static void reduceVector(vector<Derived> & _v, vector<uchar> _status)
{
  int j = 0;
  for (int i = 0; i < int(_v.size()); i++)
    if (_status[i])
      _v[j++] = _v[i];
  _v.resize(j);
}

// create keyframe online
KeyFrame::KeyFrame(
  double _time_stamp, int _index, Vector3d & _vio_T_w_i, Matrix3d & _vio_R_w_i, cv::Mat & _image,
  vector<cv::Point3f> & _point_3d, vector<cv::Point2f> & _point_2d_uv, vector<cv::Point2f> & _point_2d_norm,
  vector<double> & _point_id, int _sequence)
{
  time_stamp_ = _time_stamp;
  index_ = _index;
  vio_T_w_i_ = _vio_T_w_i;
  vio_R_w_i_ = _vio_R_w_i;
  T_w_i_ = vio_T_w_i_;
  R_w_i_ = vio_R_w_i_;
  origin_vio_T_ = vio_T_w_i_;
  origin_vio_R_ = vio_R_w_i_;
  image_ = _image.clone();
  cv::resize(image_, thumbnail_, cv::Size(80, 60));
  point_3d_ = _point_3d;
  point_2d_uv_ = _point_2d_uv;
  point_2d_norm_ = _point_2d_norm;
  point_id_ = _point_id;
  has_loop_ = false;
  loop_index_ = -1;
  has_fast_point_ = false;
  loop_info_ << 0, 0, 0, 0, 0, 0, 0, 0;
  sequence_ = _sequence;
  computeWindowBRIEFPoint();
  computeBRIEFPoint();
  if (!DEBUG_IMAGE)
    image_.release();
}

// load previous keyframe
KeyFrame::KeyFrame(
  double _time_stamp, int _index, [[maybe_unused]] Vector3d & _vio_T_w_i,
  [[maybe_unused]] Matrix3d & _vio_R_w_i, Vector3d & _T_w_i, Matrix3d & _R_w_i, cv::Mat & _image,
  int _loop_index, Eigen::Matrix<double, 8, 1> & _loop_info, vector<cv::KeyPoint> & _keypoints,
  vector<cv::KeyPoint> & _keypoints_norm, vector<BRIEF::bitset> & _brief_descriptors)
{
  time_stamp_ = _time_stamp;
  index_ = _index;
  //vio_T_w_i = _vio_T_w_i;
  //vio_R_w_i = _vio_R_w_i;
  vio_T_w_i_ = _T_w_i;
  vio_R_w_i_ = _R_w_i;
  T_w_i_ = _T_w_i;
  R_w_i_ = _R_w_i;
  if (DEBUG_IMAGE) {
    image_ = _image.clone();
    cv::resize(image_, thumbnail_, cv::Size(80, 60));
  }
  has_loop_ = _loop_index != -1;
  loop_index_ = _loop_index;
  loop_info_ = _loop_info;
  has_fast_point_ = false;
  sequence_ = 0;
  keypoints_ = _keypoints;
  keypoints_norm_ = _keypoints_norm;
  brief_descriptors_ = _brief_descriptors;
}


void KeyFrame::computeWindowBRIEFPoint()
{
  const auto extractor = cachedBriefExtractor(BRIEF_PATTERN_FILE);
  for (auto i : point_2d_uv_) {
    cv::KeyPoint key;
    key.pt = i;
    window_keypoints_.push_back(key);
  }
  (*extractor)(image_, window_keypoints_, window_brief_descriptors_);
}

void KeyFrame::computeBRIEFPoint()
{
  const auto extractor = cachedBriefExtractor(BRIEF_PATTERN_FILE);
  const int fast_th = 20; // corner detector response threshold
  if (1)
    cv::FAST(image_, keypoints_, fast_th, true);
  else {
    vector<cv::Point2f> tmp_pts;
    cv::goodFeaturesToTrack(image_, tmp_pts, 500, 0.01, 10);
    for (auto tmp_pt : tmp_pts) {
      cv::KeyPoint key;
      key.pt = tmp_pt;
      keypoints_.push_back(key);
    }
  }
  (*extractor)(image_, keypoints_, brief_descriptors_);
  for (auto & keypoint : keypoints_) {
    Eigen::Vector3d tmp_p;
    M_CAMERA->liftProjective(Eigen::Vector2d(keypoint.pt.x, keypoint.pt.y), tmp_p);
    cv::KeyPoint tmp_norm;
    tmp_norm.pt = cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z());
    keypoints_norm_.push_back(tmp_norm);
  }
}

bool KeyFrame::searchInAera(
  const BRIEF::bitset _window_descriptor, const std::vector<BRIEF::bitset> & _descriptors_old,
  const std::vector<cv::KeyPoint> & _keypoints_old, const std::vector<cv::KeyPoint> & _keypoints_old_norm,
  cv::Point2f & _best_match, cv::Point2f & _best_match_norm)
{
  cv::Point2f best_pt;
  int bestDist = 128;
  int bestIndex = -1;
  for (int i = 0; i < (int)_descriptors_old.size(); i++) {

    int dis = hammingDis(_window_descriptor, _descriptors_old[i]);
    if (dis < bestDist) {
      bestDist = dis;
      bestIndex = i;
    }
  }
  //printf("best dist %d", bestDist);
  if (bestIndex != -1 && bestDist < 80) {
    _best_match = _keypoints_old[bestIndex].pt;
    _best_match_norm = _keypoints_old_norm[bestIndex].pt;
    return true;
  } else
    return false;
}

void KeyFrame::searchByBRIEFDes(
  std::vector<cv::Point2f> & _matched_2d_old, std::vector<cv::Point2f> & _matched_2d_old_norm,
  std::vector<uchar> & _status, const std::vector<BRIEF::bitset> & _descriptors_old,
  const std::vector<cv::KeyPoint> & _keypoints_old, const std::vector<cv::KeyPoint> & _keypoints_old_norm)
{
  for (const auto & window_brief_descriptor : window_brief_descriptors_) {
    cv::Point2f pt(0.f, 0.f);
    cv::Point2f pt_norm(0.f, 0.f);
    if (searchInAera(window_brief_descriptor, _descriptors_old, _keypoints_old, _keypoints_old_norm, pt, pt_norm))
      _status.push_back(1);
    else
      _status.push_back(0);
    _matched_2d_old.push_back(pt);
    _matched_2d_old_norm.push_back(pt_norm);
  }
}


void KeyFrame::fundmantalMatrixRansac(
  const std::vector<cv::Point2f> & _matched_2d_cur_norm,
  const std::vector<cv::Point2f> & _matched_2d_old_norm, vector<uchar> & _status)
{
  int n = (int)_matched_2d_cur_norm.size();
  for (int i = 0; i < n; i++)
    _status.push_back(0);
  if (n >= 8) {
    vector<cv::Point2f> tmp_cur(n), tmp_old(n);
    for (int i = 0; i < (int)_matched_2d_cur_norm.size(); i++) {
      double FOCAL_LENGTH = 460.0;
      double tmp_x, tmp_y;
      tmp_x = FOCAL_LENGTH * _matched_2d_cur_norm[i].x + COL / 2.0;
      tmp_y = FOCAL_LENGTH * _matched_2d_cur_norm[i].y + ROW / 2.0;
      tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);

      tmp_x = FOCAL_LENGTH * _matched_2d_old_norm[i].x + COL / 2.0;
      tmp_y = FOCAL_LENGTH * _matched_2d_old_norm[i].y + ROW / 2.0;
      tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
    }
    cv::findFundamentalMat(tmp_cur, tmp_old, cv::FM_RANSAC, 3.0, 0.9, _status);
  }
}

void KeyFrame::pnPransac(
  const vector<cv::Point2f> & _matched_2d_old_norm, const std::vector<cv::Point3f> & _matched_3d,
  std::vector<uchar> & _status, Eigen::Vector3d & _PnP_T_old, Eigen::Matrix3d & _PnP_R_old)
{
  //for (int i = 0; i < matched_3d.size(); i++)
  //	printf("3d x: %f, y: %f, z: %f\n",matched_3d[i].x, matched_3d[i].y, matched_3d[i].z );
  //printf("match size %d \n", matched_3d.size());
  cv::Mat r, rvec, t, D, tmp_r;
  cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
  Matrix3d R_inital;
  Vector3d P_inital;
  Matrix3d R_w_c = origin_vio_R_ * QIC;
  Vector3d T_w_c = origin_vio_T_ + origin_vio_R_ * TIC;

  R_inital = R_w_c.inverse();
  P_inital = -(R_inital * T_w_c);

  cv::eigen2cv(R_inital, tmp_r);
  cv::Rodrigues(tmp_r, rvec);
  cv::eigen2cv(P_inital, t);

  cv::Mat inliers;
  TicToc t_pnp_ransac;

  if (CV_MAJOR_VERSION < 3)
    solvePnPRansac(_matched_3d, _matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 100, inliers);
  else {
    if (CV_MINOR_VERSION < 2)
      solvePnPRansac(
        _matched_3d, _matched_2d_old_norm, K, D, rvec, t, true, 100, sqrt(10.0 / 460.0), 0.99, inliers);
    else
      solvePnPRansac(_matched_3d, _matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 0.99, inliers);
  }

  for (int i = 0; i < (int)_matched_2d_old_norm.size(); i++)
    _status.push_back(0);

  for (int i = 0; i < inliers.rows; i++) {
    int n = inliers.at<int>(i);
    _status[n] = 1;
  }

  cv::Rodrigues(rvec, r);
  Matrix3d R_pnp, R_w_c_old;
  cv::cv2eigen(r, R_pnp);
  R_w_c_old = R_pnp.transpose();
  Vector3d T_pnp, T_w_c_old;
  cv::cv2eigen(t, T_pnp);
  T_w_c_old = R_w_c_old * (-T_pnp);

  _PnP_R_old = R_w_c_old * QIC.transpose();
  _PnP_T_old = T_w_c_old - _PnP_R_old * TIC;
}


bool KeyFrame::findConnection(KeyFrame * _old_kf)
{
  TicToc tmp_t;
  //printf("find Connection\n");
  vector<cv::Point2f> matched_2d_cur, matched_2d_old;
  vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
  vector<cv::Point3f> matched_3d;
  vector<double> matched_id;
  vector<uchar> status;

  matched_3d = point_3d_;
  matched_2d_cur = point_2d_uv_;
  matched_2d_cur_norm = point_2d_norm_;
  matched_id = point_id_;

  TicToc t_match;
#if 0
		if (DEBUG_IMAGE)    
	    {
	        cv::Mat gray_img, loop_match_img;
	        cv::Mat old_img = old_kf->image;
	        cv::hconcat(image, old_img, gray_img);
	        cvtColor(gray_img, loop_match_img, cv::COLOR_GRAY2RGB);
	        for(int i = 0; i< (int)point_2d_uv.size(); i++)
	        {
	            cv::Point2f cur_pt = point_2d_uv[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)old_kf->keypoints.size(); i++)
	        {
	            cv::Point2f old_pt = old_kf->keypoints[i].pt;
	            old_pt.x += COL;
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        ostringstream path;
	        path << "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "0raw_point.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
#endif
  //printf("search by des\n");
  searchByBRIEFDes(
    matched_2d_old, matched_2d_old_norm, status, _old_kf->brief_descriptors_, _old_kf->keypoints_,
    _old_kf->keypoints_norm_);
  reduceVector(matched_2d_cur, status);
  reduceVector(matched_2d_old, status);
  reduceVector(matched_2d_cur_norm, status);
  reduceVector(matched_2d_old_norm, status);
  reduceVector(matched_3d, status);
  reduceVector(matched_id, status);
  //printf("search by des finish\n");

#if 0
		if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, cv::COLOR_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap);
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path, path1, path2;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	        /*
	        path1 <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_1.jpg";
	        cv::imwrite( path1.str().c_str(), image);
	        path2 <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_2.jpg";
	        cv::imwrite( path2.str().c_str(), old_img);	        
	        */
	        
	    }
#endif
  status.clear();
/*
	FundmantalMatrixRANSAC(matched_2d_cur_norm, matched_2d_old_norm, status);
	reduceVector(matched_2d_cur, status);
	reduceVector(matched_2d_old, status);
	reduceVector(matched_2d_cur_norm, status);
	reduceVector(matched_2d_old_norm, status);
	reduceVector(matched_3d, status);
	reduceVector(matched_id, status);
	*/
#if 0
		if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, cv::COLOR_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap) ;
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "2fundamental_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
#endif
  Eigen::Vector3d PnP_T_old;
  Eigen::Matrix3d PnP_R_old;
  Eigen::Vector3d relative_t;
  Quaterniond relative_q;
  double relative_yaw;
  if ((int)matched_2d_cur.size() > MIN_LOOP_NUM) {
    status.clear();
    pnPransac(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);
    reduceVector(matched_2d_cur, status);
    reduceVector(matched_2d_old, status);
    reduceVector(matched_2d_cur_norm, status);
    reduceVector(matched_2d_old_norm, status);
    reduceVector(matched_3d, status);
    reduceVector(matched_id, status);
#if 1
    if (DEBUG_IMAGE) {
      int gap = 10;
      cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
      cv::Mat gray_img, loop_match_img;
      cv::Mat old_img = _old_kf->image_;
      cv::hconcat(image_, gap_image, gap_image);
      cv::hconcat(gap_image, old_img, gray_img);
      cvtColor(gray_img, loop_match_img, cv::COLOR_GRAY2RGB);
      for (auto cur_pt : matched_2d_cur) {
        cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
      }
      for (auto old_pt : matched_2d_old) {
        old_pt.x += (COL + gap);
        cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
      }
      for (int i = 0; i < (int)matched_2d_cur.size(); i++) {
        cv::Point2f old_pt = matched_2d_old[i];
        old_pt.x += (COL + gap);
        cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 2, 8, 0);
      }
      cv::Mat notation(50, COL + gap + COL, CV_8UC3, cv::Scalar(255, 255, 255));
      putText(
        notation, "current frame: " + to_string(index_) + "  sequence: " + to_string(sequence_),
        cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);

      putText(
        notation,
        "previous frame: " + to_string(_old_kf->index_) + "  sequence: " + to_string(_old_kf->sequence_),
        cv::Point2f(20 + COL + gap, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
      cv::vconcat(notation, loop_match_img, loop_match_img);

      /*
	            ostringstream path;
	            path <<  "/home/tony-ws1/raw_data/loop_image/"
	                    << index << "-"
	                    << old_kf->index << "-" << "3pnp_match.jpg";
	            cv::imwrite( path.str().c_str(), loop_match_img);
	            */
      if ((int)matched_2d_cur.size() > MIN_LOOP_NUM) {
        /*
	            	cv::imshow("loop connection",loop_match_img);  
	            	cv::waitKey(10);  
	            	*/
        cv::Mat thumbimage;
        cv::resize(loop_match_img, thumbimage, cv::Size(loop_match_img.cols / 2, loop_match_img.rows / 2));
        sensor_msgs::msg::Image::SharedPtr msg =
          cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", thumbimage).toImageMsg();
        msg->header.stamp = vins::stamp(time_stamp_);
        PUB_MATCH_IMG->publish(*msg);
      }
    }
#endif
  }

  if ((int)matched_2d_cur.size() > MIN_LOOP_NUM) {
    relative_t = PnP_R_old.transpose() * (origin_vio_T_ - PnP_T_old);
    relative_q = PnP_R_old.transpose() * origin_vio_R_;
    relative_yaw = Utility::normalizeAngle(Utility::r2ypr(origin_vio_R_).x() - Utility::r2ypr(PnP_R_old).x());
    //printf("PNP relative\n");
    //cout << "pnp relative_t " << relative_t.transpose() << endl;
    //cout << "pnp relative_yaw " << relative_yaw << endl;
    if (abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0) {

      has_loop_ = true;
      loop_index_ = _old_kf->index_;
      loop_info_ << relative_t.x(), relative_t.y(), relative_t.z(), relative_q.w(), relative_q.x(),
        relative_q.y(), relative_q.z(), relative_yaw;
      //cout << "pnp relative_t " << relative_t.transpose() << endl;
      //cout << "pnp relative_q " << relative_q.w() << " " << relative_q.vec().transpose() << endl;
      return true;
    }
  }
  //printf("loop final use num %d %lf--------------- \n", (int)matched_2d_cur.size(), t_match.toc());
  return false;
}


int KeyFrame::hammingDis(const BRIEF::bitset & _a, const BRIEF::bitset & _b)
{
  BRIEF::bitset xor_of_bitset = _a ^ _b;
  int dis = xor_of_bitset.count();
  return dis;
}

void KeyFrame::getVioPose(Eigen::Vector3d & _T_w_i, Eigen::Matrix3d & _R_w_i) const
{
  _T_w_i = vio_T_w_i_;
  _R_w_i = vio_R_w_i_;
}

void KeyFrame::getPose(Eigen::Vector3d & _T_w_i, Eigen::Matrix3d & _R_w_i) const
{
  _T_w_i = T_w_i_;
  _R_w_i = R_w_i_;
}

void KeyFrame::updatePose(const Eigen::Vector3d & _T_w_i, const Eigen::Matrix3d & _R_w_i)
{
  T_w_i_ = _T_w_i;
  R_w_i_ = _R_w_i;
}

void KeyFrame::updateVioPose(const Eigen::Vector3d & _T_w_i, const Eigen::Matrix3d & _R_w_i)
{
  vio_T_w_i_ = _T_w_i;
  vio_R_w_i_ = _R_w_i;
  T_w_i_ = vio_T_w_i_;
  R_w_i_ = vio_R_w_i_;
}

Eigen::Vector3d KeyFrame::getLoopRelativeT()
{
  return Eigen::Vector3d(loop_info_(0), loop_info_(1), loop_info_(2));
}

Eigen::Quaterniond KeyFrame::getLoopRelativeQ()
{
  return Eigen::Quaterniond(loop_info_(3), loop_info_(4), loop_info_(5), loop_info_(6));
}

double KeyFrame::getLoopRelativeYaw() { return loop_info_(7); }

void KeyFrame::updateLoop(Eigen::Matrix<double, 8, 1> & _loop_info)
{
  if (abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0) {
    //printf("update loop info\n");
    loop_info_ = _loop_info;
  }
}
