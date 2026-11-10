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

#include "initial_sfm.h"

#include <array>
#include <vector>

GlobalSFM::GlobalSFM() = default;

void GlobalSFM::triangulatePoint(
  Eigen::Matrix<double, 3, 4> & _Pose0, Eigen::Matrix<double, 3, 4> & _Pose1, Vector2d & _point0,
  Vector2d & _point1, Vector3d & _point_3d)
{
  Matrix4d design_matrix = Matrix4d::Zero();
  design_matrix.row(0) = _point0[0] * _Pose0.row(2) - _Pose0.row(0);
  design_matrix.row(1) = _point0[1] * _Pose0.row(2) - _Pose0.row(1);
  design_matrix.row(2) = _point1[0] * _Pose1.row(2) - _Pose1.row(0);
  design_matrix.row(3) = _point1[1] * _Pose1.row(2) - _Pose1.row(1);
  Vector4d triangulated_point;
  triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
  _point_3d(0) = triangulated_point(0) / triangulated_point(3);
  _point_3d(1) = triangulated_point(1) / triangulated_point(3);
  _point_3d(2) = triangulated_point(2) / triangulated_point(3);
}


bool GlobalSFM::solveFrameByPnP(
  Matrix3d & _R_initial, Vector3d & _P_initial, int _i, vector<SfmFeature_s> & _sfm_f) const
{
  vector<cv::Point2f> pts_2_vector;
  vector<cv::Point3f> pts_3_vector;
  for (int j = 0; j < feature_num_; j++) {
    if (!_sfm_f[j].state_)
      continue;
    Vector2d point2d;
    for (int k = 0; k < (int)_sfm_f[j].observation_.size(); k++) {
      if (_sfm_f[j].observation_[k].first == _i) {
        Vector2d img_pts = _sfm_f[j].observation_[k].second;
        cv::Point2f pts_2(img_pts(0), img_pts(1));
        pts_2_vector.push_back(pts_2);
        cv::Point3f pts_3(_sfm_f[j].position_[0], _sfm_f[j].position_[1], _sfm_f[j].position_[2]);
        pts_3_vector.push_back(pts_3);
        break;
      }
    }
  }
  if (int(pts_2_vector.size()) < 15) {
    printf("unstable features tracking, please slowly move you device!\n");
    if (int(pts_2_vector.size()) < 10)
      return false;
  }
  cv::Mat r, rvec, t, D, tmp_r;
  cv::eigen2cv(_R_initial, tmp_r);
  cv::Rodrigues(tmp_r, rvec);
  cv::eigen2cv(_P_initial, t);
  cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
  bool pnp_succ;
  pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
  if (!pnp_succ) {
    return false;
  }
  cv::Rodrigues(rvec, r);
  //cout << "r " << endl << r << endl;
  MatrixXd R_pnp;
  cv::cv2eigen(r, R_pnp);
  MatrixXd T_pnp;
  cv::cv2eigen(t, T_pnp);
  _R_initial = R_pnp;
  _P_initial = T_pnp;
  return true;
}

void GlobalSFM::triangulateTwoFrames(
  int _frame0, Eigen::Matrix<double, 3, 4> & _Pose0, int _frame1, Eigen::Matrix<double, 3, 4> & _Pose1,
  vector<SfmFeature_s> & _sfm_f)
{
  assert(frame0 != frame1);
  for (int j = 0; j < feature_num_; j++) {
    if (_sfm_f[j].state_)
      continue;
    bool has_0 = false, has_1 = false;
    Vector2d point0;
    Vector2d point1;
    for (auto & k : _sfm_f[j].observation_) {
      if (k.first == _frame0) {
        point0 = k.second;
        has_0 = true;
      }
      if (k.first == _frame1) {
        point1 = k.second;
        has_1 = true;
      }
    }
    if (has_0 && has_1) {
      Vector3d point_3d;
      triangulatePoint(_Pose0, _Pose1, point0, point1, point_3d);
      _sfm_f[j].state_ = true;
      _sfm_f[j].position_[0] = point_3d(0);
      _sfm_f[j].position_[1] = point_3d(1);
      _sfm_f[j].position_[2] = point_3d(2);
      //cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
    }
  }
}

// 	 q w_R_cam t w_R_cam
//  c_rotation cam_R_w
//  c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)
bool GlobalSFM::construct(
  int _frame_num, Quaterniond * _q, Vector3d * _T, int _l, const Matrix3d _relative_R,
  const Vector3d _relative_T, vector<SfmFeature_s> & _sfm_f, map<int, Vector3d> & _sfm_tracked_points)
{
  if (_frame_num <= 0 || _q == nullptr || _T == nullptr || _l < 0 || _l >= _frame_num)
    return false;
  feature_num_ = _sfm_f.size();
  //cout << "set 0 and " << l << " as known " << endl;
  // have relative_r relative_t
  // intial two view
  _q[_l].w() = 1;
  _q[_l].x() = 0;
  _q[_l].y() = 0;
  _q[_l].z() = 0;
  _T[_l].setZero();
  _q[_frame_num - 1] = _q[_l] * Quaterniond(_relative_R);
  _T[_frame_num - 1] = _relative_T;
  //cout << "init q_l " << q[l].w() << " " << q[l].vec().transpose() << endl;
  //cout << "init t_l " << T[l].transpose() << endl;

  //rotate to cam frame
  Matrix3d c_Rotation[_frame_num];
  Vector3d c_Translation[_frame_num];
  Quaterniond c_Quat[_frame_num];
  // Ceres stores these addresses as parameter blocks. Keep each block in a
  // separately owned, non-overlapping allocation instead of adjacent
  // variable-length stack arrays.
  std::vector<std::array<double, 4>> c_rotation(static_cast<size_t>(_frame_num));
  std::vector<std::array<double, 3>> c_translation(static_cast<size_t>(_frame_num));
  Eigen::Matrix<double, 3, 4> Pose[_frame_num];

  c_Quat[_l] = _q[_l].inverse();
  c_Rotation[_l] = c_Quat[_l].toRotationMatrix();
  c_Translation[_l] = -1 * (c_Rotation[_l] * _T[_l]);
  Pose[_l].block<3, 3>(0, 0) = c_Rotation[_l];
  Pose[_l].block<3, 1>(0, 3) = c_Translation[_l];

  c_Quat[_frame_num - 1] = _q[_frame_num - 1].inverse();
  c_Rotation[_frame_num - 1] = c_Quat[_frame_num - 1].toRotationMatrix();
  c_Translation[_frame_num - 1] = -1 * (c_Rotation[_frame_num - 1] * _T[_frame_num - 1]);
  Pose[_frame_num - 1].block<3, 3>(0, 0) = c_Rotation[_frame_num - 1];
  Pose[_frame_num - 1].block<3, 1>(0, 3) = c_Translation[_frame_num - 1];


  //1: trangulate between l ----- frame_num - 1
  //2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1;
  for (int i = _l; i < _frame_num - 1; i++) {
    // solve pnp
    if (i > _l) {
      Matrix3d R_initial = c_Rotation[i - 1];
      Vector3d P_initial = c_Translation[i - 1];
      if (!solveFrameByPnP(R_initial, P_initial, i, _sfm_f))
        return false;
      c_Rotation[i] = R_initial;
      c_Translation[i] = P_initial;
      c_Quat[i] = c_Rotation[i];
      Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
      Pose[i].block<3, 1>(0, 3) = c_Translation[i];
    }

    // triangulate point based on the solve pnp result
    triangulateTwoFrames(i, Pose[i], _frame_num - 1, Pose[_frame_num - 1], _sfm_f);
  }
  //3: triangulate l-----l+1 l+2 ... frame_num -2
  for (int i = _l + 1; i < _frame_num - 1; i++)
    triangulateTwoFrames(_l, Pose[_l], i, Pose[i], _sfm_f);
  //4: solve pnp l-1; triangulate l-1 ----- l
  //             l-2              l-2 ----- l
  for (int i = _l - 1; i >= 0; i--) {
    //solve pnp
    Matrix3d R_initial = c_Rotation[i + 1];
    Vector3d P_initial = c_Translation[i + 1];
    if (!solveFrameByPnP(R_initial, P_initial, i, _sfm_f))
      return false;
    c_Rotation[i] = R_initial;
    c_Translation[i] = P_initial;
    c_Quat[i] = c_Rotation[i];
    Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
    Pose[i].block<3, 1>(0, 3) = c_Translation[i];
    //triangulate
    triangulateTwoFrames(i, Pose[i], _l, Pose[_l], _sfm_f);
  }
  //5: triangulate all other points
  for (int j = 0; j < feature_num_; j++) {
    if (_sfm_f[j].state_)
      continue;
    if ((int)_sfm_f[j].observation_.size() >= 2) {
      Vector2d point0, point1;
      int frame_0 = _sfm_f[j].observation_[0].first;
      point0 = _sfm_f[j].observation_[0].second;
      int frame_1 = _sfm_f[j].observation_.back().first;
      point1 = _sfm_f[j].observation_.back().second;
      Vector3d point_3d;
      triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
      _sfm_f[j].state_ = true;
      _sfm_f[j].position_[0] = point_3d(0);
      _sfm_f[j].position_[1] = point_3d(1);
      _sfm_f[j].position_[2] = point_3d(2);
      //cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
    }
  }

  /*
	for (int i = 0; i < frame_num; i++)
	{
		q[i] = c_Rotation[i].transpose(); 
		cout << "solvePnP  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{
		Vector3d t_tmp;
		t_tmp = -1 * (q[i] * c_Translation[i]);
		cout << "solvePnP  t" << " i " << i <<"  " << t_tmp.x() <<"  "<< t_tmp.y() <<"  "<< t_tmp.z() << endl;
	}
*/
  //full BA
  ceres::Problem problem;
  ceres::Manifold * local_parameterization = new ceres::QuaternionManifold();
  //cout << " begin full BA " << endl;
  for (int i = 0; i < _frame_num; i++) {
    //double array for ceres
    c_translation[i][0] = c_Translation[i].x();
    c_translation[i][1] = c_Translation[i].y();
    c_translation[i][2] = c_Translation[i].z();
    c_rotation[i][0] = c_Quat[i].w();
    c_rotation[i][1] = c_Quat[i].x();
    c_rotation[i][2] = c_Quat[i].y();
    c_rotation[i][3] = c_Quat[i].z();
    problem.AddParameterBlock(c_rotation[i].data(), 4, local_parameterization);
    problem.AddParameterBlock(c_translation[i].data(), 3);
    if (i == _l) {
      problem.SetParameterBlockConstant(c_rotation[i].data());
    }
    if (i == _l || i == _frame_num - 1) {
      problem.SetParameterBlockConstant(c_translation[i].data());
    }
  }

  for (int i = 0; i < feature_num_; i++) {
    if (!_sfm_f[i].state_)
      continue;
    for (int j = 0; j < int(_sfm_f[i].observation_.size()); j++) {
      int l = _sfm_f[i].observation_[j].first;
      ceres::CostFunction * cost_function = ReprojectionError3D_s::create(
        _sfm_f[i].observation_[j].second.x(), _sfm_f[i].observation_[j].second.y());

      problem.AddResidualBlock(
        cost_function, nullptr, c_rotation[l].data(), c_translation[l].data(), _sfm_f[i].position_);
    }
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  //options.minimizer_progress_to_stdout = true;
  options.max_solver_time_in_seconds = 0.2;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  //std::cout << summary.BriefReport() << "\n";
  if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03) {
    //cout << "vision only BA converge" << endl;
  } else {
    //cout << "vision only BA not converge " << endl;
    return false;
  }
  for (int i = 0; i < _frame_num; i++) {
    _q[i].w() = c_rotation[i][0];
    _q[i].x() = c_rotation[i][1];
    _q[i].y() = c_rotation[i][2];
    _q[i].z() = c_rotation[i][3];
    _q[i] = _q[i].inverse();
    //cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
  }
  for (int i = 0; i < _frame_num; i++) {

    _T[i] = -1 * (_q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
    //cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"  "<< T[i](2) << endl;
  }
  for (auto & i : _sfm_f) {
    if (i.state_)
      _sfm_tracked_points[i.id_] = Vector3d(i.position_[0], i.position_[1], i.position_[2]);
  }
  return true;
}
