/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "pose_local_parameterization.h"

#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vins_common/ros_utils.hpp>

#include "../estimator/parameters.h"
#include "../utility/utility.h"
#include "integration_base.h"

#include <ceres/ceres.h>

class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9> {
public:
  IMUFactor() = delete;
  IMUFactor(IntegrationBase * _pre_integration) : pre_integration_(_pre_integration) {}
  bool Evaluate(double const * const * _parameters, double * _residuals, double ** _jacobians) const override
  {

    Eigen::Vector3d Pi(_parameters[0][0], _parameters[0][1], _parameters[0][2]);
    Eigen::Quaterniond Qi(_parameters[0][6], _parameters[0][3], _parameters[0][4], _parameters[0][5]);

    Eigen::Vector3d Vi(_parameters[1][0], _parameters[1][1], _parameters[1][2]);
    Eigen::Vector3d Bai(_parameters[1][3], _parameters[1][4], _parameters[1][5]);
    Eigen::Vector3d Bgi(_parameters[1][6], _parameters[1][7], _parameters[1][8]);

    Eigen::Vector3d Pj(_parameters[2][0], _parameters[2][1], _parameters[2][2]);
    Eigen::Quaterniond Qj(_parameters[2][6], _parameters[2][3], _parameters[2][4], _parameters[2][5]);

    Eigen::Vector3d Vj(_parameters[3][0], _parameters[3][1], _parameters[3][2]);
    Eigen::Vector3d Baj(_parameters[3][3], _parameters[3][4], _parameters[3][5]);
    Eigen::Vector3d Bgj(_parameters[3][6], _parameters[3][7], _parameters[3][8]);

    //Eigen::Matrix<double, 15, 15> Fd;
    //Eigen::Matrix<double, 15, 12> Gd;

    //Eigen::Vector3d pPj = Pi + Vi * sum_t - 0.5 * g * sum_t * sum_t + corrected_delta_p;
    //Eigen::Quaterniond pQj = Qi * delta_q;
    //Eigen::Vector3d pVj = Vi - g * sum_t + corrected_delta_v;
    //Eigen::Vector3d pBaj = Bai;
    //Eigen::Vector3d pBgj = Bgi;

    //Vi + Qi * delta_v - g * sum_dt = Vj;
    //Qi * delta_q = Qj;

    //delta_p = Qi.inverse() * (0.5 * g * sum_dt * sum_dt + Pj - Pi);
    //delta_v = Qi.inverse() * (g * sum_dt + Vj - Vi);
    //delta_q = Qi.inverse() * Qj;

#if 0
        if ((Bai - pre_integration->linearized_ba).norm() > 0.10 ||
            (Bgi - pre_integration->linearized_bg).norm() > 0.01)
        {
            pre_integration->repropagate(Bai, Bgi);
        }
#endif

    Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(_residuals);
    residual = pre_integration_->evaluate(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj);

    Eigen::Matrix<double, 15, 15> sqrt_info =
      Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integration_->covariance_.inverse()).matrixL().transpose();
    //sqrt_info.setIdentity();
    residual = sqrt_info * residual;

    if (_jacobians) {
      double sum_dt = pre_integration_->sum_dt_;
      Eigen::Matrix3d dp_dba = pre_integration_->jacobian_.template block<3, 3>(O_P, O_BA);
      Eigen::Matrix3d dp_dbg = pre_integration_->jacobian_.template block<3, 3>(O_P, O_BG);

      Eigen::Matrix3d dq_dbg = pre_integration_->jacobian_.template block<3, 3>(O_R, O_BG);

      Eigen::Matrix3d dv_dba = pre_integration_->jacobian_.template block<3, 3>(O_V, O_BA);
      Eigen::Matrix3d dv_dbg = pre_integration_->jacobian_.template block<3, 3>(O_V, O_BG);

      if (pre_integration_->jacobian_.maxCoeff() > 1e8 || pre_integration_->jacobian_.minCoeff() < -1e8) {
        RCLCPP_WARN(vins::logger(), "numerical unstable in preintegration");
        //std::cout << pre_integration->jacobian << std::endl;
        ///                throw std::runtime_error("VINS invariant violated");
      }

      if (_jacobians[0]) {
        Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_i(_jacobians[0]);
        jacobian_pose_i.setZero();

        jacobian_pose_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();
        jacobian_pose_i.block<3, 3>(O_P, O_R) =
          Utility::skewSymmetric(Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));

#if 0
            jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Qj.inverse() * Qi).toRotationMatrix();
#else
        Eigen::Quaterniond corrected_delta_q =
          pre_integration_->delta_q_ * Utility::deltaQ(dq_dbg * (Bgi - pre_integration_->linearized_bg_));
        jacobian_pose_i.block<3, 3>(O_R, O_R) =
          -(Utility::qleft(Qj.inverse() * Qi) * Utility::qright(corrected_delta_q)).bottomRightCorner<3, 3>();
#endif

        jacobian_pose_i.block<3, 3>(O_V, O_R) = Utility::skewSymmetric(Qi.inverse() * (G * sum_dt + Vj - Vi));

        jacobian_pose_i = sqrt_info * jacobian_pose_i;

        if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8) {
          RCLCPP_WARN(vins::logger(), "numerical unstable in preintegration");
          //std::cout << sqrt_info << std::endl;
          //throw std::runtime_error("VINS invariant violated");
        }
      }
      if (_jacobians[1]) {
        Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(_jacobians[1]);
        jacobian_speedbias_i.setZero();
        jacobian_speedbias_i.block<3, 3>(O_P, O_V - O_V) = -Qi.inverse().toRotationMatrix() * sum_dt;
        jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
        jacobian_speedbias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;

#if 0
            jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -dq_dbg;
#else
        //Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
        //jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Utility::Qleft(Qj.inverse() * Qi * corrected_delta_q).bottomRightCorner<3, 3>() * dq_dbg;
        jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) =
          -Utility::qleft(Qj.inverse() * Qi * pre_integration_->delta_q_).bottomRightCorner<3, 3>() * dq_dbg;
#endif

        jacobian_speedbias_i.block<3, 3>(O_V, O_V - O_V) = -Qi.inverse().toRotationMatrix();
        jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
        jacobian_speedbias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;

        jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_V) = -Eigen::Matrix3d::Identity();

        jacobian_speedbias_i.block<3, 3>(O_BG, O_BG - O_V) = -Eigen::Matrix3d::Identity();

        jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;

        //assert(fabs(jacobian_speedbias_i.maxCoeff()) < 1e8);
        //assert(fabs(jacobian_speedbias_i.minCoeff()) < 1e8);
      }
      if (_jacobians[2]) {
        Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_j(_jacobians[2]);
        jacobian_pose_j.setZero();

        jacobian_pose_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();

#if 0
            jacobian_pose_j.block<3, 3>(O_R, O_R) = Eigen::Matrix3d::Identity();
#else
        Eigen::Quaterniond corrected_delta_q =
          pre_integration_->delta_q_ * Utility::deltaQ(dq_dbg * (Bgi - pre_integration_->linearized_bg_));
        jacobian_pose_j.block<3, 3>(O_R, O_R) =
          Utility::qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();
#endif

        jacobian_pose_j = sqrt_info * jacobian_pose_j;

        //assert(fabs(jacobian_pose_j.maxCoeff()) < 1e8);
        //assert(fabs(jacobian_pose_j.minCoeff()) < 1e8);
      }
      if (_jacobians[3]) {
        Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(_jacobians[3]);
        jacobian_speedbias_j.setZero();

        jacobian_speedbias_j.block<3, 3>(O_V, O_V - O_V) = Qi.inverse().toRotationMatrix();

        jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_V) = Eigen::Matrix3d::Identity();

        jacobian_speedbias_j.block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix3d::Identity();

        jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;

        //assert(fabs(jacobian_speedbias_j.maxCoeff()) < 1e8);
        //assert(fabs(jacobian_speedbias_j.minCoeff()) < 1e8);
      }
    }

    if (_jacobians) {
      if (_jacobians[0])
        PoseManifold::liftJacobian(_parameters[0], _jacobians[0], 15);
      if (_jacobians[2])
        PoseManifold::liftJacobian(_parameters[2], _jacobians[2], 15);
    }
    return true;
  }

  //bool Evaluate_Direct(double const *const *parameters, Eigen::Matrix<double, 15, 1> &residuals, Eigen::Matrix<double, 15, 30> &jacobians);

  //void checkCorrection();
  //void checkTransition();
  //void checkJacobian(double **parameters);
  IntegrationBase * pre_integration_;
};
