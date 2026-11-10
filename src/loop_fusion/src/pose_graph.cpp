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

#include "pose_graph.h"

PoseGraph::PoseGraph()
{
  posegraph_visualization_ = new CameraPoseVisualization(1.0, 0.0, 1.0, 1.0);
  posegraph_visualization_->setScale(0.1);
  posegraph_visualization_->setLineWidth(0.01);
  earliest_loop_index_ = -1;
  t_drift_ = Eigen::Vector3d(0, 0, 0);
  yaw_drift_ = 0;
  r_drift_ = Eigen::Matrix3d::Identity();
  w_t_vio_ = Eigen::Vector3d(0, 0, 0);
  w_r_vio_ = Eigen::Matrix3d::Identity();
  global_index_ = 0;
  sequence_cnt_ = 0;
  sequence_loop_.push_back(0);
  base_sequence_ = 1;
  use_imu_ = 0;
}

PoseGraph::~PoseGraph()
{
  stopping_ = true;
  if (t_optimization_.joinable())
    t_optimization_.join();
  for (auto * keyframe : keyframelist_)
    delete keyframe;
  delete posegraph_visualization_;
}


void PoseGraph::registerPub(const rclcpp::Node::SharedPtr & _n)
{
  pub_pg_path_ = _n->create_publisher<nav_msgs::msg::Path>("~/pose_graph_path", rclcpp::QoS(100));
  pub_base_path_ = _n->create_publisher<nav_msgs::msg::Path>("~/base_path", rclcpp::QoS(100));
  pub_pose_graph_ =
    _n->create_publisher<visualization_msgs::msg::MarkerArray>("~/pose_graph", rclcpp::QoS(100));
  for (int i = 1; i < 10; i++)
    pub_path_[i] = _n->create_publisher<nav_msgs::msg::Path>("~/path_" + to_string(i), rclcpp::QoS(100));
}

void PoseGraph::setIMUFlag(bool _use_imu)
{
  use_imu_ = _use_imu;
  if (use_imu_) {
    printf("VIO input, perfrom 4 DoF (x, y, z, yaw) pose graph optimization\n");
    t_optimization_ = std::thread(&PoseGraph::optimize4DoF, this);
  } else {
    printf("VO input, perfrom 6 DoF pose graph optimization\n");
    t_optimization_ = std::thread(&PoseGraph::optimize6DoF, this);
  }
}

void PoseGraph::loadVocabulary(std::string _voc_path)
{
  voc_ = new BriefVocabulary(_voc_path);
  db_.setVocabulary(*voc_, false, 0);
}

loop_fusion::Correction_s PoseGraph::getCorrection()
{
  std::lock_guard<std::mutex> lock(m_drift_);
  return loop_fusion::composeCorrection(r_drift_, t_drift_, w_r_vio_, w_t_vio_);
}

void PoseGraph::addKeyFrame(KeyFrame * _cur_kf, bool _flag_detect_loop)
{
  //shift to base frame
  Vector3d vio_P_cur;
  Matrix3d vio_R_cur;
  if (sequence_cnt_ != _cur_kf->sequence_) {
    sequence_cnt_++;
    sequence_loop_.push_back(0);
    w_t_vio_ = Eigen::Vector3d(0, 0, 0);
    w_r_vio_ = Eigen::Matrix3d::Identity();
    m_drift_.lock();
    t_drift_ = Eigen::Vector3d(0, 0, 0);
    r_drift_ = Eigen::Matrix3d::Identity();
    m_drift_.unlock();
  }

  _cur_kf->getVioPose(vio_P_cur, vio_R_cur);
  vio_P_cur = w_r_vio_ * vio_P_cur + w_t_vio_;
  vio_R_cur = w_r_vio_ * vio_R_cur;
  _cur_kf->updateVioPose(vio_P_cur, vio_R_cur);
  _cur_kf->index_ = global_index_;
  global_index_++;
  int loop_index = -1;
  if (_flag_detect_loop) {
    TicToc tmp_t;
    loop_index = detectLoop(_cur_kf, _cur_kf->index_);
  } else {
    addKeyFrameIntoVoc(_cur_kf);
  }
  if (loop_index != -1) {
    //printf(" %d detect loop with %d \n", cur_kf->index, loop_index);
    KeyFrame * old_kf = getKeyFrame(loop_index);

    if (_cur_kf->findConnection(old_kf)) {
      if (earliest_loop_index_ > loop_index || earliest_loop_index_ == -1)
        earliest_loop_index_ = loop_index;

      Vector3d w_P_old, w_P_cur, vio_P_cur;
      Matrix3d w_R_old, w_R_cur, vio_R_cur;
      old_kf->getVioPose(w_P_old, w_R_old);
      _cur_kf->getVioPose(vio_P_cur, vio_R_cur);

      Vector3d relative_t;
      Quaterniond relative_q;
      relative_t = _cur_kf->getLoopRelativeT();
      relative_q = (_cur_kf->getLoopRelativeQ()).toRotationMatrix();
      w_P_cur = w_R_old * relative_t + w_P_old;
      w_R_cur = w_R_old * relative_q;
      double shift_yaw;
      Matrix3d shift_r;
      Vector3d shift_t;
      if (use_imu_) {
        shift_yaw = Utility::r2ypr(w_R_cur).x() - Utility::r2ypr(vio_R_cur).x();
        shift_r = Utility::ypr2R(Vector3d(shift_yaw, 0, 0));
      } else
        shift_r = w_R_cur * vio_R_cur.transpose();
      shift_t = w_P_cur - w_R_cur * vio_R_cur.transpose() * vio_P_cur;
      // shift vio pose of whole sequence to the world frame
      if (old_kf->sequence_ != _cur_kf->sequence_ && sequence_loop_[_cur_kf->sequence_] == 0) {
        w_r_vio_ = shift_r;
        w_t_vio_ = shift_t;
        vio_P_cur = w_r_vio_ * vio_P_cur + w_t_vio_;
        vio_R_cur = w_r_vio_ * vio_R_cur;
        _cur_kf->updateVioPose(vio_P_cur, vio_R_cur);
        list<KeyFrame *>::iterator it = keyframelist_.begin();
        for (; it != keyframelist_.end(); it++) {
          if ((*it)->sequence_ == _cur_kf->sequence_) {
            Vector3d vio_P_cur;
            Matrix3d vio_R_cur;
            (*it)->getVioPose(vio_P_cur, vio_R_cur);
            vio_P_cur = w_r_vio_ * vio_P_cur + w_t_vio_;
            vio_R_cur = w_r_vio_ * vio_R_cur;
            (*it)->updateVioPose(vio_P_cur, vio_R_cur);
          }
        }
        sequence_loop_[_cur_kf->sequence_] = 1;
      }
      m_optimize_buf_.lock();
      optimize_buf_.push(_cur_kf->index_);
      m_optimize_buf_.unlock();
    }
  }
  m_keyframelist_.lock();
  Vector3d P;
  Matrix3d R;
  _cur_kf->getVioPose(P, R);
  {
    std::lock_guard<std::mutex> lock(m_drift_);
    P = r_drift_ * P + t_drift_;
    R = r_drift_ * R;
  }
  _cur_kf->updatePose(P, R);
  Quaterniond Q{R};
  geometry_msgs::msg::PoseStamped pose_stamped;
  pose_stamped.header.stamp = vins::stamp(_cur_kf->time_stamp_);
  pose_stamped.header.frame_id = "world";
  pose_stamped.pose.position.x = P.x() + VISUALIZATION_SHIFT_X;
  pose_stamped.pose.position.y = P.y() + VISUALIZATION_SHIFT_Y;
  pose_stamped.pose.position.z = P.z();
  pose_stamped.pose.orientation.x = Q.x();
  pose_stamped.pose.orientation.y = Q.y();
  pose_stamped.pose.orientation.z = Q.z();
  pose_stamped.pose.orientation.w = Q.w();
  path_[sequence_cnt_].poses.push_back(pose_stamped);
  path_[sequence_cnt_].header = pose_stamped.header;

  if (SAVE_LOOP_PATH) {
    ofstream loop_path_file(VINS_RESULT_PATH, ios::app);
    loop_path_file.setf(ios::fixed, ios::floatfield);
    loop_path_file.precision(0);
    loop_path_file << _cur_kf->time_stamp_ * 1e9 << ",";
    loop_path_file.precision(5);
    loop_path_file << P.x() << "," << P.y() << "," << P.z() << "," << Q.w() << "," << Q.x() << "," << Q.y()
                   << "," << Q.z() << "," << '\n';
    loop_path_file.close();
  }
  //draw local connection
  if (SHOW_S_EDGE) {
    list<KeyFrame *>::reverse_iterator rit = keyframelist_.rbegin();
    for (int i = 0; i < 4; i++) {
      if (rit == keyframelist_.rend())
        break;
      Vector3d conncected_P;
      Matrix3d connected_R;
      if ((*rit)->sequence_ == _cur_kf->sequence_) {
        (*rit)->getPose(conncected_P, connected_R);
        posegraph_visualization_->addEdge(P, conncected_P);
      }
      rit++;
    }
  }
  if (SHOW_L_EDGE) {
    if (_cur_kf->has_loop_) {
      //printf("has loop \n");
      KeyFrame * connected_KF = getKeyFrame(_cur_kf->loop_index_);
      Vector3d connected_P, P0;
      Matrix3d connected_R, R0;
      connected_KF->getPose(connected_P, connected_R);
      //cur_kf->getVioPose(P0, R0);
      _cur_kf->getPose(P0, R0);
      if (_cur_kf->sequence_ > 0) {
        //printf("add loop into visual \n");
        posegraph_visualization_->addLoopedge(
          P0, connected_P + Vector3d(VISUALIZATION_SHIFT_X, VISUALIZATION_SHIFT_Y, 0));
      }
    }
  }
  //posegraph_visualization->add_pose(P + Vector3d(VISUALIZATION_SHIFT_X, VISUALIZATION_SHIFT_Y, 0), Q);

  keyframelist_.push_back(_cur_kf);
  publish();
  m_keyframelist_.unlock();
}


void PoseGraph::loadKeyFrame(KeyFrame * _cur_kf, bool _flag_detect_loop)
{
  _cur_kf->index_ = global_index_;
  global_index_++;
  int loop_index = -1;
  if (_flag_detect_loop)
    loop_index = detectLoop(_cur_kf, _cur_kf->index_);
  else {
    addKeyFrameIntoVoc(_cur_kf);
  }
  if (loop_index != -1) {
    printf(" %d detect loop with %d \n", _cur_kf->index_, loop_index);
    KeyFrame * old_kf = getKeyFrame(loop_index);
    if (_cur_kf->findConnection(old_kf)) {
      if (earliest_loop_index_ > loop_index || earliest_loop_index_ == -1)
        earliest_loop_index_ = loop_index;
      m_optimize_buf_.lock();
      optimize_buf_.push(_cur_kf->index_);
      m_optimize_buf_.unlock();
    }
  }
  m_keyframelist_.lock();
  Vector3d P;
  Matrix3d R;
  _cur_kf->getPose(P, R);
  Quaterniond Q{R};
  geometry_msgs::msg::PoseStamped pose_stamped;
  pose_stamped.header.stamp = vins::stamp(_cur_kf->time_stamp_);
  pose_stamped.header.frame_id = "world";
  pose_stamped.pose.position.x = P.x() + VISUALIZATION_SHIFT_X;
  pose_stamped.pose.position.y = P.y() + VISUALIZATION_SHIFT_Y;
  pose_stamped.pose.position.z = P.z();
  pose_stamped.pose.orientation.x = Q.x();
  pose_stamped.pose.orientation.y = Q.y();
  pose_stamped.pose.orientation.z = Q.z();
  pose_stamped.pose.orientation.w = Q.w();
  base_path_.poses.push_back(pose_stamped);
  base_path_.header = pose_stamped.header;

  //draw local connection
  if (SHOW_S_EDGE) {
    list<KeyFrame *>::reverse_iterator rit = keyframelist_.rbegin();
    for (int i = 0; i < 1; i++) {
      if (rit == keyframelist_.rend())
        break;
      Vector3d conncected_P;
      Matrix3d connected_R;
      if ((*rit)->sequence_ == _cur_kf->sequence_) {
        (*rit)->getPose(conncected_P, connected_R);
        posegraph_visualization_->addEdge(P, conncected_P);
      }
      rit++;
    }
  }
  /*
    if (cur_kf->has_loop)
    {
        KeyFrame* connected_KF = getKeyFrame(cur_kf->loop_index);
        Vector3d connected_P;
        Matrix3d connected_R;
        connected_KF->getPose(connected_P,  connected_R);
        posegraph_visualization->add_loopedge(P, connected_P, SHIFT);
    }
    */

  keyframelist_.push_back(_cur_kf);
  //publish();
  m_keyframelist_.unlock();
}

KeyFrame * PoseGraph::getKeyFrame(int _index)
{
  //    unique_lock<mutex> lock(m_keyframelist);
  list<KeyFrame *>::iterator it = keyframelist_.begin();
  for (; it != keyframelist_.end(); it++) {
    if ((*it)->index_ == _index)
      break;
  }
  if (it != keyframelist_.end())
    return *it;
  else
    return nullptr;
}

int PoseGraph::detectLoop(KeyFrame * _keyframe, int _frame_index)
{
  // put image into image_pool; for visualization
  cv::Mat compressed_image;
  if (DEBUG_IMAGE) {
    int feature_num = _keyframe->keypoints_.size();
    cv::resize(_keyframe->image_, compressed_image, cv::Size(376, 240));
    putText(
      compressed_image, "feature_num:" + to_string(feature_num), cv::Point2f(10, 10),
      cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
    image_pool_[_frame_index] = compressed_image;
  }
  TicToc tmp_t;
  //first query; then add this frame into database!
  QueryResults ret;
  TicToc t_query;
  db_.query(_keyframe->brief_descriptors_, ret, 4, _frame_index - 50);
  //printf("query time: %f", t_query.toc());
  //cout << "Searching for Image " << frame_index << ". " << ret << endl;

  TicToc t_add;
  db_.add(_keyframe->brief_descriptors_);
  //printf("add feature time: %f", t_add.toc());
  // ret[0] is the nearest neighbour's score. threshold change with neighour score
  bool find_loop = false;
  cv::Mat loop_result;
  if (DEBUG_IMAGE) {
    loop_result = compressed_image.clone();
    if (!ret.empty())
      putText(
        loop_result, "neighbour score:" + to_string(ret[0].Score), cv::Point2f(10, 50),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
  }
  // visual loop result
  if (DEBUG_IMAGE) {
    for (auto & i : ret) {
      int tmp_index = i.Id;
      auto it = image_pool_.find(tmp_index);
      cv::Mat tmp_image = (it->second).clone();
      putText(
        tmp_image, "index:  " + to_string(tmp_index) + "loop score:" + to_string(i.Score),
        cv::Point2f(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
      cv::hconcat(loop_result, tmp_image, loop_result);
    }
  }
  // a good match with its nerghbour
  if (!ret.empty() && ret[0].Score > 0.05)
    for (unsigned int i = 1; i < ret.size(); i++) {
      //if (ret[i].Score > ret[0].Score * 0.3)
      if (ret[i].Score > 0.015) {
        find_loop = true;
        int tmp_index = ret[i].Id;
        if (DEBUG_IMAGE && 0) {
          auto it = image_pool_.find(tmp_index);
          cv::Mat tmp_image = (it->second).clone();
          putText(
            tmp_image, "loop score:" + to_string(ret[i].Score), cv::Point2f(10, 50), cv::FONT_HERSHEY_SIMPLEX,
            0.4, cv::Scalar(255));
          cv::hconcat(loop_result, tmp_image, loop_result);
        }
      }
    }
  /*
    if (DEBUG_IMAGE)
    {
        cv::imshow("loop_result", loop_result);
        cv::waitKey(20);
    }
*/
  if (find_loop && _frame_index > 50) {
    int min_index = -1;
    for (auto & i : ret) {
      if (min_index == -1 || (i.Id < min_index && i.Score > 0.015))
        min_index = i.Id;
    }
    return min_index;
  } else
    return -1;
}

void PoseGraph::addKeyFrameIntoVoc(KeyFrame * _keyframe)
{
  // put image into image_pool; for visualization
  cv::Mat compressed_image;
  if (DEBUG_IMAGE) {
    int feature_num = _keyframe->keypoints_.size();
    cv::resize(_keyframe->image_, compressed_image, cv::Size(376, 240));
    putText(
      compressed_image, "feature_num:" + to_string(feature_num), cv::Point2f(10, 10),
      cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
    image_pool_[_keyframe->index_] = compressed_image;
  }

  db_.add(_keyframe->brief_descriptors_);
}

void PoseGraph::optimize4DoF()
{
  while (!stopping_ && rclcpp::ok()) {
    int cur_index = -1;
    int first_looped_index = -1;
    m_optimize_buf_.lock();
    while (!optimize_buf_.empty()) {
      cur_index = optimize_buf_.front();
      first_looped_index = earliest_loop_index_;
      optimize_buf_.pop();
    }
    m_optimize_buf_.unlock();
    if (cur_index != -1) {
      printf("optimize pose graph \n");
      TicToc tmp_t;
      m_keyframelist_.lock();
      KeyFrame * cur_kf = getKeyFrame(cur_index);

      int max_length = cur_index + 1;

      // w^t_i   w^q_i
      double t_array[max_length][3];
      Quaterniond q_array[max_length];
      double euler_array[max_length][3];
      double sequence_array[max_length];

      ceres::Problem problem;
      ceres::Solver::Options options;
      options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
      //options.minimizer_progress_to_stdout = true;
      //options.max_solver_time_in_seconds = SOLVER_TIME * 3;
      options.max_num_iterations = 5;
      ceres::Solver::Summary summary;
      ceres::LossFunction * loss_function;
      loss_function = new ceres::HuberLoss(0.1);
      //loss_function = new ceres::CauchyLoss(1.0);
      ceres::Manifold * angle_local_parameterization = AngleManifold_s::create();

      list<KeyFrame *>::iterator it;

      int i = 0;
      for (it = keyframelist_.begin(); it != keyframelist_.end(); it++) {
        if ((*it)->index_ < first_looped_index)
          continue;
        (*it)->local_index_ = i;
        Quaterniond tmp_q;
        Matrix3d tmp_r;
        Vector3d tmp_t;
        (*it)->getVioPose(tmp_t, tmp_r);
        tmp_q = tmp_r;
        t_array[i][0] = tmp_t(0);
        t_array[i][1] = tmp_t(1);
        t_array[i][2] = tmp_t(2);
        q_array[i] = tmp_q;

        Vector3d euler_angle = Utility::r2ypr(tmp_q.toRotationMatrix());
        euler_array[i][0] = euler_angle.x();
        euler_array[i][1] = euler_angle.y();
        euler_array[i][2] = euler_angle.z();

        sequence_array[i] = (*it)->sequence_;

        problem.AddParameterBlock(euler_array[i], 1, angle_local_parameterization);
        problem.AddParameterBlock(t_array[i], 3);

        if ((*it)->index_ == first_looped_index || (*it)->sequence_ == 0) {
          problem.SetParameterBlockConstant(euler_array[i]);
          problem.SetParameterBlockConstant(t_array[i]);
        }

        //add edge
        for (int j = 1; j < 5; j++) {
          if (i - j >= 0 && sequence_array[i] == sequence_array[i - j]) {
            Vector3d euler_conncected = Utility::r2ypr(q_array[i - j].toRotationMatrix());
            Vector3d relative_t(
              t_array[i][0] - t_array[i - j][0], t_array[i][1] - t_array[i - j][1],
              t_array[i][2] - t_array[i - j][2]);
            relative_t = q_array[i - j].inverse() * relative_t;
            double relative_yaw = euler_array[i][0] - euler_array[i - j][0];
            ceres::CostFunction * cost_function = FourDofError_s::create(
              relative_t.x(), relative_t.y(), relative_t.z(), relative_yaw, euler_conncected.y(),
              euler_conncected.z());
            problem.AddResidualBlock(
              cost_function, nullptr, euler_array[i - j], t_array[i - j], euler_array[i], t_array[i]);
          }
        }

        //add loop edge

        if ((*it)->has_loop_) {
          assert((*it)->loop_index >= first_looped_index);
          int connected_index = getKeyFrame((*it)->loop_index_)->local_index_;
          Vector3d euler_conncected = Utility::r2ypr(q_array[connected_index].toRotationMatrix());
          Vector3d relative_t;
          relative_t = (*it)->getLoopRelativeT();
          double relative_yaw = (*it)->getLoopRelativeYaw();
          ceres::CostFunction * cost_function = FourDofWeightError_s::create(
            relative_t.x(), relative_t.y(), relative_t.z(), relative_yaw, euler_conncected.y(),
            euler_conncected.z());
          problem.AddResidualBlock(
            cost_function, loss_function, euler_array[connected_index], t_array[connected_index],
            euler_array[i], t_array[i]);
        }

        if ((*it)->index_ == cur_index)
          break;
        i++;
      }
      m_keyframelist_.unlock();

      ceres::Solve(options, &problem, &summary);
      //std::cout << summary.BriefReport() << "\n";

      //printf("pose optimization time: %f \n", tmp_t.toc());
      /*
            for (int j = 0 ; j < i; j++)
            {
                printf("optimize i: %d p: %f, %f, %f\n", j, t_array[j][0], t_array[j][1], t_array[j][2] );
            }
            */
      m_keyframelist_.lock();
      i = 0;
      for (it = keyframelist_.begin(); it != keyframelist_.end(); it++) {
        if ((*it)->index_ < first_looped_index)
          continue;
        Quaterniond tmp_q;
        tmp_q = Utility::ypr2R(Vector3d(euler_array[i][0], euler_array[i][1], euler_array[i][2]));
        Vector3d tmp_t = Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
        Matrix3d tmp_r = tmp_q.toRotationMatrix();
        (*it)->updatePose(tmp_t, tmp_r);

        if ((*it)->index_ == cur_index)
          break;
        i++;
      }

      Vector3d cur_t, vio_t;
      Matrix3d cur_r, vio_r;
      cur_kf->getPose(cur_t, cur_r);
      cur_kf->getVioPose(vio_t, vio_r);
      m_drift_.lock();
      yaw_drift_ = Utility::r2ypr(cur_r).x() - Utility::r2ypr(vio_r).x();
      r_drift_ = Utility::ypr2R(Vector3d(yaw_drift_, 0, 0));
      t_drift_ = cur_t - r_drift_ * vio_t;
      const Matrix3d correction_rotation = r_drift_;
      const Vector3d correction_translation = t_drift_;
      m_drift_.unlock();
      //cout << "t_drift " << t_drift.transpose() << endl;
      //cout << "r_drift " << Utility::R2ypr(r_drift).transpose() << endl;
      //cout << "yaw drift " << yaw_drift << endl;

      it++;
      for (; it != keyframelist_.end(); it++) {
        Vector3d P;
        Matrix3d R;
        (*it)->getVioPose(P, R);
        P = correction_rotation * P + correction_translation;
        R = correction_rotation * R;
        (*it)->updatePose(P, R);
      }
      m_keyframelist_.unlock();
      updatePath();
    }

    std::chrono::milliseconds dura(2000);
    std::this_thread::sleep_for(dura);
  }
  return;
}


void PoseGraph::optimize6DoF()
{
  while (!stopping_ && rclcpp::ok()) {
    int cur_index = -1;
    int first_looped_index = -1;
    m_optimize_buf_.lock();
    while (!optimize_buf_.empty()) {
      cur_index = optimize_buf_.front();
      first_looped_index = earliest_loop_index_;
      optimize_buf_.pop();
    }
    m_optimize_buf_.unlock();
    if (cur_index != -1) {
      printf("optimize pose graph \n");
      TicToc tmp_t;
      m_keyframelist_.lock();
      KeyFrame * cur_kf = getKeyFrame(cur_index);

      int max_length = cur_index + 1;

      // w^t_i   w^q_i
      double t_array[max_length][3];
      double q_array[max_length][4];
      double sequence_array[max_length];

      ceres::Problem problem;
      ceres::Solver::Options options;
      options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
      //ptions.minimizer_progress_to_stdout = true;
      //options.max_solver_time_in_seconds = SOLVER_TIME * 3;
      options.max_num_iterations = 5;
      ceres::Solver::Summary summary;
      ceres::LossFunction * loss_function;
      loss_function = new ceres::HuberLoss(0.1);
      //loss_function = new ceres::CauchyLoss(1.0);
      ceres::Manifold * local_parameterization = new ceres::QuaternionManifold();

      list<KeyFrame *>::iterator it;

      int i = 0;
      for (it = keyframelist_.begin(); it != keyframelist_.end(); it++) {
        if ((*it)->index_ < first_looped_index)
          continue;
        (*it)->local_index_ = i;
        Quaterniond tmp_q;
        Matrix3d tmp_r;
        Vector3d tmp_t;
        (*it)->getVioPose(tmp_t, tmp_r);
        tmp_q = tmp_r;
        t_array[i][0] = tmp_t(0);
        t_array[i][1] = tmp_t(1);
        t_array[i][2] = tmp_t(2);
        q_array[i][0] = tmp_q.w();
        q_array[i][1] = tmp_q.x();
        q_array[i][2] = tmp_q.y();
        q_array[i][3] = tmp_q.z();

        sequence_array[i] = (*it)->sequence_;

        problem.AddParameterBlock(q_array[i], 4, local_parameterization);
        problem.AddParameterBlock(t_array[i], 3);

        if ((*it)->index_ == first_looped_index || (*it)->sequence_ == 0) {
          problem.SetParameterBlockConstant(q_array[i]);
          problem.SetParameterBlockConstant(t_array[i]);
        }

        //add edge
        for (int j = 1; j < 5; j++) {
          if (i - j >= 0 && sequence_array[i] == sequence_array[i - j]) {
            Vector3d relative_t(
              t_array[i][0] - t_array[i - j][0], t_array[i][1] - t_array[i - j][1],
              t_array[i][2] - t_array[i - j][2]);
            Quaterniond q_i_j =
              Quaterniond(q_array[i - j][0], q_array[i - j][1], q_array[i - j][2], q_array[i - j][3]);
            Quaterniond q_i = Quaterniond(q_array[i][0], q_array[i][1], q_array[i][2], q_array[i][3]);
            relative_t = q_i_j.inverse() * relative_t;
            Quaterniond relative_q = q_i_j.inverse() * q_i;
            ceres::CostFunction * vo_function = RelativeRtError_s::create(
              relative_t.x(), relative_t.y(), relative_t.z(), relative_q.w(), relative_q.x(), relative_q.y(),
              relative_q.z(), 0.1, 0.01);
            problem.AddResidualBlock(
              vo_function, nullptr, q_array[i - j], t_array[i - j], q_array[i], t_array[i]);
          }
        }

        //add loop edge

        if ((*it)->has_loop_) {
          assert((*it)->loop_index >= first_looped_index);
          int connected_index = getKeyFrame((*it)->loop_index_)->local_index_;
          Vector3d relative_t;
          relative_t = (*it)->getLoopRelativeT();
          Quaterniond relative_q;
          relative_q = (*it)->getLoopRelativeQ();
          ceres::CostFunction * loop_function = RelativeRtError_s::create(
            relative_t.x(), relative_t.y(), relative_t.z(), relative_q.w(), relative_q.x(), relative_q.y(),
            relative_q.z(), 0.1, 0.01);
          problem.AddResidualBlock(
            loop_function, loss_function, q_array[connected_index], t_array[connected_index], q_array[i],
            t_array[i]);
        }

        if ((*it)->index_ == cur_index)
          break;
        i++;
      }
      m_keyframelist_.unlock();

      ceres::Solve(options, &problem, &summary);
      //std::cout << summary.BriefReport() << "\n";

      //printf("pose optimization time: %f \n", tmp_t.toc());
      /*
            for (int j = 0 ; j < i; j++)
            {
                printf("optimize i: %d p: %f, %f, %f\n", j, t_array[j][0], t_array[j][1], t_array[j][2] );
            }
            */
      m_keyframelist_.lock();
      i = 0;
      for (it = keyframelist_.begin(); it != keyframelist_.end(); it++) {
        if ((*it)->index_ < first_looped_index)
          continue;
        Quaterniond tmp_q(q_array[i][0], q_array[i][1], q_array[i][2], q_array[i][3]);
        Vector3d tmp_t = Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
        Matrix3d tmp_r = tmp_q.toRotationMatrix();
        (*it)->updatePose(tmp_t, tmp_r);

        if ((*it)->index_ == cur_index)
          break;
        i++;
      }

      Vector3d cur_t, vio_t;
      Matrix3d cur_r, vio_r;
      cur_kf->getPose(cur_t, cur_r);
      cur_kf->getVioPose(vio_t, vio_r);
      m_drift_.lock();
      r_drift_ = cur_r * vio_r.transpose();
      t_drift_ = cur_t - r_drift_ * vio_t;
      const Matrix3d correction_rotation = r_drift_;
      const Vector3d correction_translation = t_drift_;
      m_drift_.unlock();
      //cout << "t_drift " << t_drift.transpose() << endl;
      //cout << "r_drift " << Utility::R2ypr(r_drift).transpose() << endl;

      it++;
      for (; it != keyframelist_.end(); it++) {
        Vector3d P;
        Matrix3d R;
        (*it)->getVioPose(P, R);
        P = correction_rotation * P + correction_translation;
        R = correction_rotation * R;
        (*it)->updatePose(P, R);
      }
      m_keyframelist_.unlock();
      updatePath();
    }

    std::chrono::milliseconds dura(2000);
    std::this_thread::sleep_for(dura);
  }
  return;
}

void PoseGraph::updatePath()
{
  m_keyframelist_.lock();
  list<KeyFrame *>::iterator it;
  for (int i = 1; i <= sequence_cnt_; i++) {
    path_[i].poses.clear();
  }
  base_path_.poses.clear();
  posegraph_visualization_->reset();

  if (SAVE_LOOP_PATH) {
    ofstream loop_path_file_tmp(VINS_RESULT_PATH, ios::out);
    loop_path_file_tmp.close();
  }

  for (it = keyframelist_.begin(); it != keyframelist_.end(); it++) {
    Vector3d P;
    Matrix3d R;
    (*it)->getPose(P, R);
    Quaterniond Q;
    Q = R;
    //        printf("path p: %f, %f, %f\n",  P.x(),  P.z(),  P.y() );

    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header.stamp = vins::stamp((*it)->time_stamp_);
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose.position.x = P.x() + VISUALIZATION_SHIFT_X;
    pose_stamped.pose.position.y = P.y() + VISUALIZATION_SHIFT_Y;
    pose_stamped.pose.position.z = P.z();
    pose_stamped.pose.orientation.x = Q.x();
    pose_stamped.pose.orientation.y = Q.y();
    pose_stamped.pose.orientation.z = Q.z();
    pose_stamped.pose.orientation.w = Q.w();
    if ((*it)->sequence_ == 0) {
      base_path_.poses.push_back(pose_stamped);
      base_path_.header = pose_stamped.header;
    } else {
      path_[(*it)->sequence_].poses.push_back(pose_stamped);
      path_[(*it)->sequence_].header = pose_stamped.header;
    }

    if (SAVE_LOOP_PATH) {
      ofstream loop_path_file(VINS_RESULT_PATH, ios::app);
      loop_path_file.setf(ios::fixed, ios::floatfield);
      loop_path_file.precision(0);
      loop_path_file << (*it)->time_stamp_ * 1e9 << ",";
      loop_path_file.precision(5);
      loop_path_file << P.x() << "," << P.y() << "," << P.z() << "," << Q.w() << "," << Q.x() << "," << Q.y()
                     << "," << Q.z() << "," << '\n';
      loop_path_file.close();
    }
    //draw local connection
    if (SHOW_S_EDGE) {
      list<KeyFrame *>::reverse_iterator rit = keyframelist_.rbegin();
      list<KeyFrame *>::reverse_iterator lrit;
      for (; rit != keyframelist_.rend(); rit++) {
        if ((*rit)->index_ == (*it)->index_) {
          lrit = rit;
          lrit++;
          for (int i = 0; i < 4; i++) {
            if (lrit == keyframelist_.rend())
              break;
            if ((*lrit)->sequence_ == (*it)->sequence_) {
              Vector3d conncected_P;
              Matrix3d connected_R;
              (*lrit)->getPose(conncected_P, connected_R);
              posegraph_visualization_->addEdge(P, conncected_P);
            }
            lrit++;
          }
          break;
        }
      }
    }
    if (SHOW_L_EDGE) {
      if ((*it)->has_loop_ && (*it)->sequence_ == sequence_cnt_) {

        KeyFrame * connected_KF = getKeyFrame((*it)->loop_index_);
        Vector3d connected_P;
        Matrix3d connected_R;
        connected_KF->getPose(connected_P, connected_R);
        //(*it)->getVioPose(P, R);
        (*it)->getPose(P, R);
        if ((*it)->sequence_ > 0) {
          posegraph_visualization_->addLoopedge(
            P, connected_P + Vector3d(VISUALIZATION_SHIFT_X, VISUALIZATION_SHIFT_Y, 0));
        }
      }
    }
  }
  publish();
  m_keyframelist_.unlock();
}


void PoseGraph::savePoseGraph()
{
  m_keyframelist_.lock();
  TicToc tmp_t;
  FILE * pFile;
  printf("pose graph path: %s\n", POSE_GRAPH_SAVE_PATH.c_str());
  printf("pose graph saving... \n");
  string file_path = POSE_GRAPH_SAVE_PATH + "pose_graph.txt";
  pFile = fopen(file_path.c_str(), "w");
  //fprintf(pFile, "index time_stamp Tx Ty Tz Qw Qx Qy Qz loop_index loop_info\n");
  list<KeyFrame *>::iterator it;
  for (it = keyframelist_.begin(); it != keyframelist_.end(); it++) {
    std::string image_path, descriptor_path, brief_path, keypoints_path;
    if (DEBUG_IMAGE) {
      image_path = POSE_GRAPH_SAVE_PATH + to_string((*it)->index_) + "_image.png";
      imwrite(image_path, (*it)->image_);
    }
    Quaterniond VIO_tmp_Q{(*it)->vio_R_w_i_};
    Quaterniond PG_tmp_Q{(*it)->R_w_i_};
    Vector3d VIO_tmp_T = (*it)->vio_T_w_i_;
    Vector3d PG_tmp_T = (*it)->T_w_i_;

    fprintf(
      pFile, " %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d %f %f %f %f %f %f %f %f %d\n",
      (*it)->index_, (*it)->time_stamp_, VIO_tmp_T.x(), VIO_tmp_T.y(), VIO_tmp_T.z(), PG_tmp_T.x(),
      PG_tmp_T.y(), PG_tmp_T.z(), VIO_tmp_Q.w(), VIO_tmp_Q.x(), VIO_tmp_Q.y(), VIO_tmp_Q.z(), PG_tmp_Q.w(),
      PG_tmp_Q.x(), PG_tmp_Q.y(), PG_tmp_Q.z(), (*it)->loop_index_, (*it)->loop_info_(0),
      (*it)->loop_info_(1), (*it)->loop_info_(2), (*it)->loop_info_(3), (*it)->loop_info_(4),
      (*it)->loop_info_(5), (*it)->loop_info_(6), (*it)->loop_info_(7), (int)(*it)->keypoints_.size());

    // write keypoints, brief_descriptors   vector<cv::KeyPoint> keypoints vector<BRIEF::bitset> brief_descriptors;
    assert((*it)->keypoints.size() == (*it)->brief_descriptors.size());
    brief_path = POSE_GRAPH_SAVE_PATH + to_string((*it)->index_) + "_briefdes.dat";
    std::ofstream brief_file(brief_path, std::ios::binary);
    keypoints_path = POSE_GRAPH_SAVE_PATH + to_string((*it)->index_) + "_keypoints.txt";
    FILE * keypoints_file;
    keypoints_file = fopen(keypoints_path.c_str(), "w");
    for (int i = 0; i < (int)(*it)->keypoints_.size(); i++) {
      brief_file << (*it)->brief_descriptors_[i] << '\n';
      fprintf(
        keypoints_file, "%f %f %f %f\n", (*it)->keypoints_[i].pt.x, (*it)->keypoints_[i].pt.y,
        (*it)->keypoints_norm_[i].pt.x, (*it)->keypoints_norm_[i].pt.y);
    }
    brief_file.close();
    fclose(keypoints_file);
  }
  fclose(pFile);

  printf("save pose graph time: %f s\n", tmp_t.toc() / 1000);
  m_keyframelist_.unlock();
}
void PoseGraph::loadPoseGraph()
{
  TicToc tmp_t;
  FILE * pFile;
  string file_path = POSE_GRAPH_SAVE_PATH + "pose_graph.txt";
  printf("lode pose graph from: %s \n", file_path.c_str());
  printf("pose graph loading...\n");
  pFile = fopen(file_path.c_str(), "r");
  if (pFile == nullptr) {
    printf("lode previous pose graph error: wrong previous pose graph path or no previous pose graph \n the "
           "system will start with new pose graph \n");
    return;
  }
  int index;
  double time_stamp;
  double VIO_Tx, VIO_Ty, VIO_Tz;
  double PG_Tx, PG_Ty, PG_Tz;
  double VIO_Qw, VIO_Qx, VIO_Qy, VIO_Qz;
  double PG_Qw, PG_Qx, PG_Qy, PG_Qz;
  double loop_info_0, loop_info_1, loop_info_2, loop_info_3;
  double loop_info_4, loop_info_5, loop_info_6, loop_info_7;
  int loop_index;
  int keypoints_num;
  Eigen::Matrix<double, 8, 1> loop_info;
  int cnt = 0;
  while (
    fscanf(
      pFile,
      "%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %d",
      &index, &time_stamp, &VIO_Tx, &VIO_Ty, &VIO_Tz, &PG_Tx, &PG_Ty, &PG_Tz, &VIO_Qw, &VIO_Qx, &VIO_Qy,
      &VIO_Qz, &PG_Qw, &PG_Qx, &PG_Qy, &PG_Qz, &loop_index, &loop_info_0, &loop_info_1, &loop_info_2,
      &loop_info_3, &loop_info_4, &loop_info_5, &loop_info_6, &loop_info_7, &keypoints_num) != EOF) {
    /*
        printf("I read: %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %d\n", index, time_stamp, 
                                    VIO_Tx, VIO_Ty, VIO_Tz, 
                                    PG_Tx, PG_Ty, PG_Tz, 
                                    VIO_Qw, VIO_Qx, VIO_Qy, VIO_Qz, 
                                    PG_Qw, PG_Qx, PG_Qy, PG_Qz, 
                                    loop_index,
                                    loop_info_0, loop_info_1, loop_info_2, loop_info_3, 
                                    loop_info_4, loop_info_5, loop_info_6, loop_info_7,
                                    keypoints_num);
        */
    cv::Mat image;
    std::string image_path, descriptor_path;
    if (DEBUG_IMAGE) {
      image_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_image.png";
      image = cv::imread(image_path, 0);
    }

    Vector3d VIO_T(VIO_Tx, VIO_Ty, VIO_Tz);
    Vector3d PG_T(PG_Tx, PG_Ty, PG_Tz);
    Quaterniond VIO_Q;
    VIO_Q.w() = VIO_Qw;
    VIO_Q.x() = VIO_Qx;
    VIO_Q.y() = VIO_Qy;
    VIO_Q.z() = VIO_Qz;
    Quaterniond PG_Q;
    PG_Q.w() = PG_Qw;
    PG_Q.x() = PG_Qx;
    PG_Q.y() = PG_Qy;
    PG_Q.z() = PG_Qz;
    Matrix3d VIO_R, PG_R;
    VIO_R = VIO_Q.toRotationMatrix();
    PG_R = PG_Q.toRotationMatrix();
    Eigen::Matrix<double, 8, 1> loop_info;
    loop_info << loop_info_0, loop_info_1, loop_info_2, loop_info_3, loop_info_4, loop_info_5, loop_info_6,
      loop_info_7;

    if (loop_index != -1)
      if (earliest_loop_index_ > loop_index || earliest_loop_index_ == -1) {
        earliest_loop_index_ = loop_index;
      }

    // load keypoints, brief_descriptors
    string brief_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_briefdes.dat";
    std::ifstream brief_file(brief_path, std::ios::binary);
    string keypoints_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_keypoints.txt";
    FILE * keypoints_file;
    keypoints_file = fopen(keypoints_path.c_str(), "r");
    vector<cv::KeyPoint> keypoints;
    vector<cv::KeyPoint> keypoints_norm;
    vector<BRIEF::bitset> brief_descriptors;
    for (int i = 0; i < keypoints_num; i++) {
      BRIEF::bitset tmp_des;
      brief_file >> tmp_des;
      brief_descriptors.push_back(tmp_des);
      cv::KeyPoint tmp_keypoint;
      cv::KeyPoint tmp_keypoint_norm;
      double p_x, p_y, p_x_norm, p_y_norm;
      if (!fscanf(keypoints_file, "%lf %lf %lf %lf", &p_x, &p_y, &p_x_norm, &p_y_norm))
        printf(" fail to load pose graph \n");
      tmp_keypoint.pt.x = p_x;
      tmp_keypoint.pt.y = p_y;
      tmp_keypoint_norm.pt.x = p_x_norm;
      tmp_keypoint_norm.pt.y = p_y_norm;
      keypoints.push_back(tmp_keypoint);
      keypoints_norm.push_back(tmp_keypoint_norm);
    }
    brief_file.close();
    fclose(keypoints_file);

    KeyFrame * keyframe = new KeyFrame(
      time_stamp, index, VIO_T, VIO_R, PG_T, PG_R, image, loop_index, loop_info, keypoints, keypoints_norm,
      brief_descriptors);
    loadKeyFrame(keyframe, 0);
    if (cnt % 20 == 0) {
      publish();
    }
    cnt++;
  }
  fclose(pFile);
  printf("load pose graph time: %f s\n", tmp_t.toc() / 1000);
  base_sequence_ = 0;
}

void PoseGraph::publish()
{
  for (int i = 1; i <= sequence_cnt_; i++) {
    //if (sequence_loop[i] == true || i == base_sequence)
    if (1 || i == base_sequence_) {
      pub_pg_path_->publish(path_[i]);
      pub_path_[i]->publish(path_[i]);
      posegraph_visualization_->publishBy(pub_pose_graph_, path_[sequence_cnt_].header);
    }
  }
  pub_base_path_->publish(base_path_);
  //posegraph_visualization->publish_by(pub_pose_graph, path[sequence_cnt].header);
}
