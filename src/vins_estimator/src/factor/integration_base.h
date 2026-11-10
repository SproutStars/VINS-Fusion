/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <vins_common/ros_utils.hpp>

#pragma once

#include "../estimator/parameters.h"
#include "../utility/utility.h"

#include <ceres/ceres.h>
using namespace Eigen;

class IntegrationBase {
public:
  IntegrationBase() = delete;
  IntegrationBase(
    const Eigen::Vector3d & _acc_0, const Eigen::Vector3d & _gyr_0, const Eigen::Vector3d & _linearized_ba,
    const Eigen::Vector3d & _linearized_bg)
    : acc_0_{_acc_0}, gyr_0_{_gyr_0}, linearized_acc_{_acc_0}, linearized_gyr_{_gyr_0},
      linearized_ba_{_linearized_ba}, linearized_bg_{_linearized_bg},
      jacobian_{Eigen::Matrix<double, 15, 15>::Identity()},
      covariance_{Eigen::Matrix<double, 15, 15>::Zero()}, sum_dt_{0.0}, delta_p_{Eigen::Vector3d::Zero()},
      delta_q_{Eigen::Quaterniond::Identity()}, delta_v_{Eigen::Vector3d::Zero()}

  {
    noise_ = Eigen::Matrix<double, 18, 18>::Zero();
    noise_.block<3, 3>(0, 0) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
    noise_.block<3, 3>(3, 3) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
    noise_.block<3, 3>(6, 6) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
    noise_.block<3, 3>(9, 9) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
    noise_.block<3, 3>(12, 12) = (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
    noise_.block<3, 3>(15, 15) = (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
  }

  void pushBack(double _dt, const Eigen::Vector3d & _acc, const Eigen::Vector3d & _gyr)
  {
    dt_buf_.push_back(_dt);
    acc_buf_.push_back(_acc);
    gyr_buf_.push_back(_gyr);
    propagate(_dt, _acc, _gyr);
  }

  void repropagate(const Eigen::Vector3d & _linearized_ba, const Eigen::Vector3d & _linearized_bg)
  {
    sum_dt_ = 0.0;
    acc_0_ = linearized_acc_;
    gyr_0_ = linearized_gyr_;
    delta_p_.setZero();
    delta_q_.setIdentity();
    delta_v_.setZero();
    linearized_ba_ = _linearized_ba;
    linearized_bg_ = _linearized_bg;
    jacobian_.setIdentity();
    covariance_.setZero();
    for (int i = 0; i < static_cast<int>(dt_buf_.size()); i++)
      propagate(dt_buf_[i], acc_buf_[i], gyr_buf_[i]);
  }

  void midPointIntegration(
    double _dt, const Eigen::Vector3d & _acc_0, const Eigen::Vector3d & _gyr_0,
    const Eigen::Vector3d & _acc_1, const Eigen::Vector3d & _gyr_1, const Eigen::Vector3d & _delta_p,
    const Eigen::Quaterniond & _delta_q, const Eigen::Vector3d & _delta_v,
    const Eigen::Vector3d & _linearized_ba, const Eigen::Vector3d & _linearized_bg,
    Eigen::Vector3d & _result_delta_p, Eigen::Quaterniond & _result_delta_q, Eigen::Vector3d & _result_delta_v,
    Eigen::Vector3d & _result_linearized_ba, Eigen::Vector3d & _result_linearized_bg, bool _update_jacobian)
  {
    //RCLCPP_INFO(vins::logger(), "midpoint integration");
    Vector3d un_acc_0 = _delta_q * (_acc_0 - _linearized_ba);
    Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - _linearized_bg;
    _result_delta_q = _delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
    Vector3d un_acc_1 = _result_delta_q * (_acc_1 - _linearized_ba);
    Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    _result_delta_p = _delta_p + _delta_v * _dt + 0.5 * un_acc * _dt * _dt;
    _result_delta_v = _delta_v + un_acc * _dt;
    _result_linearized_ba = _linearized_ba;
    _result_linearized_bg = _linearized_bg;

    if (_update_jacobian) {
      Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - _linearized_bg;
      Vector3d a_0_x = _acc_0 - _linearized_ba;
      Vector3d a_1_x = _acc_1 - _linearized_ba;
      Matrix3d R_w_x, R_a_0_x, R_a_1_x;

      R_w_x << 0, -w_x(2), w_x(1), w_x(2), 0, -w_x(0), -w_x(1), w_x(0), 0;
      R_a_0_x << 0, -a_0_x(2), a_0_x(1), a_0_x(2), 0, -a_0_x(0), -a_0_x(1), a_0_x(0), 0;
      R_a_1_x << 0, -a_1_x(2), a_1_x(1), a_1_x(2), 0, -a_1_x(0), -a_1_x(1), a_1_x(0), 0;

      MatrixXd F = MatrixXd::Zero(15, 15);
      F.block<3, 3>(0, 0) = Matrix3d::Identity();
      F.block<3, 3>(0, 3) = -0.25 * _delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt +
                            -0.25 * _result_delta_q.toRotationMatrix() * R_a_1_x *
                              (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
      F.block<3, 3>(0, 6) = MatrixXd::Identity(3, 3) * _dt;
      F.block<3, 3>(0, 9) =
        -0.25 * (_delta_q.toRotationMatrix() + _result_delta_q.toRotationMatrix()) * _dt * _dt;
      F.block<3, 3>(0, 12) = -0.25 * _result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
      F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;
      F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3, 3) * _dt;
      F.block<3, 3>(6, 3) =
        -0.5 * _delta_q.toRotationMatrix() * R_a_0_x * _dt +
        -0.5 * _result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt;
      F.block<3, 3>(6, 6) = Matrix3d::Identity();
      F.block<3, 3>(6, 9) = -0.5 * (_delta_q.toRotationMatrix() + _result_delta_q.toRotationMatrix()) * _dt;
      F.block<3, 3>(6, 12) = -0.5 * _result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
      F.block<3, 3>(9, 9) = Matrix3d::Identity();
      F.block<3, 3>(12, 12) = Matrix3d::Identity();
      //cout<<"A"<<endl<<A<<endl;

      MatrixXd V = MatrixXd::Zero(15, 18);
      V.block<3, 3>(0, 0) = 0.25 * _delta_q.toRotationMatrix() * _dt * _dt;
      V.block<3, 3>(0, 3) = 0.25 * -_result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * 0.5 * _dt;
      V.block<3, 3>(0, 6) = 0.25 * _result_delta_q.toRotationMatrix() * _dt * _dt;
      V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);
      V.block<3, 3>(3, 3) = 0.5 * MatrixXd::Identity(3, 3) * _dt;
      V.block<3, 3>(3, 9) = 0.5 * MatrixXd::Identity(3, 3) * _dt;
      V.block<3, 3>(6, 0) = 0.5 * _delta_q.toRotationMatrix() * _dt;
      V.block<3, 3>(6, 3) = 0.5 * -_result_delta_q.toRotationMatrix() * R_a_1_x * _dt * 0.5 * _dt;
      V.block<3, 3>(6, 6) = 0.5 * _result_delta_q.toRotationMatrix() * _dt;
      V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);
      V.block<3, 3>(9, 12) = MatrixXd::Identity(3, 3) * _dt;
      V.block<3, 3>(12, 15) = MatrixXd::Identity(3, 3) * _dt;

      //step_jacobian = F;
      //step_V = V;
      jacobian_ = F * jacobian_;
      covariance_ = F * covariance_ * F.transpose() + V * noise_ * V.transpose();
    }
  }

  void propagate(double _dt, const Eigen::Vector3d & _acc_1, const Eigen::Vector3d & _gyr_1)
  {
    dt_ = _dt;
    acc_1_ = _acc_1;
    gyr_1_ = _gyr_1;
    Vector3d result_delta_p;
    Quaterniond result_delta_q;
    Vector3d result_delta_v;
    Vector3d result_linearized_ba;
    Vector3d result_linearized_bg;

    midPointIntegration(
      _dt, acc_0_, gyr_0_, _acc_1, _gyr_1, delta_p_, delta_q_, delta_v_, linearized_ba_, linearized_bg_,
      result_delta_p, result_delta_q, result_delta_v, result_linearized_ba, result_linearized_bg, 1);

    //checkJacobian(_dt, acc_0, gyr_0, acc_1, gyr_1, delta_p, delta_q, delta_v,
    //                    linearized_ba, linearized_bg);
    delta_p_ = result_delta_p;
    delta_q_ = result_delta_q;
    delta_v_ = result_delta_v;
    linearized_ba_ = result_linearized_ba;
    linearized_bg_ = result_linearized_bg;
    delta_q_.normalize();
    sum_dt_ += dt_;
    acc_0_ = acc_1_;
    gyr_0_ = gyr_1_;
  }

  Eigen::Matrix<double, 15, 1> evaluate(
    const Eigen::Vector3d & _Pi, const Eigen::Quaterniond & _Qi, const Eigen::Vector3d & _Vi,
    const Eigen::Vector3d & _Bai, const Eigen::Vector3d & _Bgi, const Eigen::Vector3d & _Pj,
    const Eigen::Quaterniond & _Qj, const Eigen::Vector3d & _Vj, const Eigen::Vector3d & _Baj,
    const Eigen::Vector3d & _Bgj)
  {
    Eigen::Matrix<double, 15, 1> residuals;

    Eigen::Matrix3d dp_dba = jacobian_.block<3, 3>(O_P, O_BA);
    Eigen::Matrix3d dp_dbg = jacobian_.block<3, 3>(O_P, O_BG);

    Eigen::Matrix3d dq_dbg = jacobian_.block<3, 3>(O_R, O_BG);

    Eigen::Matrix3d dv_dba = jacobian_.block<3, 3>(O_V, O_BA);
    Eigen::Matrix3d dv_dbg = jacobian_.block<3, 3>(O_V, O_BG);

    Eigen::Vector3d dba = _Bai - linearized_ba_;
    Eigen::Vector3d dbg = _Bgi - linearized_bg_;

    Eigen::Quaterniond corrected_delta_q = delta_q_ * Utility::deltaQ(dq_dbg * dbg);
    Eigen::Vector3d corrected_delta_v = delta_v_ + dv_dba * dba + dv_dbg * dbg;
    Eigen::Vector3d corrected_delta_p = delta_p_ + dp_dba * dba + dp_dbg * dbg;

    residuals.block<3, 1>(O_P, 0) =
      _Qi.inverse() * (0.5 * G * sum_dt_ * sum_dt_ + _Pj - _Pi - _Vi * sum_dt_) - corrected_delta_p;
    residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (_Qi.inverse() * _Qj)).vec();
    residuals.block<3, 1>(O_V, 0) = _Qi.inverse() * (G * sum_dt_ + _Vj - _Vi) - corrected_delta_v;
    residuals.block<3, 1>(O_BA, 0) = _Baj - _Bai;
    residuals.block<3, 1>(O_BG, 0) = _Bgj - _Bgi;
    return residuals;
  }

  double dt_;
  Eigen::Vector3d acc_0_, gyr_0_;
  Eigen::Vector3d acc_1_, gyr_1_;

  const Eigen::Vector3d linearized_acc_, linearized_gyr_;
  Eigen::Vector3d linearized_ba_, linearized_bg_;

  Eigen::Matrix<double, 15, 15> jacobian_, covariance_;
  Eigen::Matrix<double, 15, 15> step_jacobian_;
  Eigen::Matrix<double, 15, 18> step_V_;
  Eigen::Matrix<double, 18, 18> noise_;

  double sum_dt_;
  Eigen::Vector3d delta_p_;
  Eigen::Quaterniond delta_q_;
  Eigen::Vector3d delta_v_;

  std::vector<double> dt_buf_;
  std::vector<Eigen::Vector3d> acc_buf_;
  std::vector<Eigen::Vector3d> gyr_buf_;
};
/*

    void eulerIntegration(double _dt, const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                            const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                            const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                            const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                            Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                            Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg, bool update_jacobian)
    {
        result_delta_p = delta_p + delta_v * _dt + 0.5 * (delta_q * (_acc_1 - linearized_ba)) * _dt * _dt;
        result_delta_v = delta_v + delta_q * (_acc_1 - linearized_ba) * _dt;
        Vector3d omg = _gyr_1 - linearized_bg;
        omg = omg * _dt / 2;
        Quaterniond dR(1, omg(0), omg(1), omg(2));
        result_delta_q = (delta_q * dR);   
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;         

        if(update_jacobian)
        {
            Vector3d w_x = _gyr_1 - linearized_bg;
            Vector3d a_x = _acc_1 - linearized_ba;
            Matrix3d R_w_x, R_a_x;

            R_w_x<<0, -w_x(2), w_x(1),
                w_x(2), 0, -w_x(0),
                -w_x(1), w_x(0), 0;
            R_a_x<<0, -a_x(2), a_x(1),
                a_x(2), 0, -a_x(0),
                -a_x(1), a_x(0), 0;

            MatrixXd A = MatrixXd::Zero(15, 15);
            // one step euler 0.5
            A.block<3, 3>(0, 3) = 0.5 * (-1 * delta_q.toRotationMatrix()) * R_a_x * _dt;
            A.block<3, 3>(0, 6) = MatrixXd::Identity(3,3);
            A.block<3, 3>(0, 9) = 0.5 * (-1 * delta_q.toRotationMatrix()) * _dt;
            A.block<3, 3>(3, 3) = -R_w_x;
            A.block<3, 3>(3, 12) = -1 * MatrixXd::Identity(3,3);
            A.block<3, 3>(6, 3) = (-1 * delta_q.toRotationMatrix()) * R_a_x;
            A.block<3, 3>(6, 9) = (-1 * delta_q.toRotationMatrix());
            //cout<<"A"<<endl<<A<<endl;

            MatrixXd U = MatrixXd::Zero(15,12);
            U.block<3, 3>(0, 0) =  0.5 * delta_q.toRotationMatrix() * _dt;
            U.block<3, 3>(3, 3) =  MatrixXd::Identity(3,3);
            U.block<3, 3>(6, 0) =  delta_q.toRotationMatrix();
            U.block<3, 3>(9, 6) = MatrixXd::Identity(3,3);
            U.block<3, 3>(12, 9) = MatrixXd::Identity(3,3);

            // put outside
            Eigen::Matrix<double, 12, 12> noise = Eigen::Matrix<double, 12, 12>::Zero();
            noise.block<3, 3>(0, 0) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
            noise.block<3, 3>(3, 3) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
            noise.block<3, 3>(6, 6) =  (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
            noise.block<3, 3>(9, 9) =  (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();

            //write F directly
            MatrixXd F, V;
            F = (MatrixXd::Identity(15,15) + _dt * A);
            V = _dt * U;
            step_jacobian = F;
            step_V = V;
            jacobian = F * jacobian;
            covariance = F * covariance * F.transpose() + V * noise * V.transpose();
        }

    }     


    void checkJacobian(double _dt, const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0, 
                                   const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                            const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                            const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg)
    {
        Vector3d result_delta_p;
        Quaterniond result_delta_q;
        Vector3d result_delta_v;
        Vector3d result_linearized_ba;
        Vector3d result_linearized_bg;
        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, 0);

        Vector3d turb_delta_p;
        Quaterniond turb_delta_q;
        Vector3d turb_delta_v;
        Vector3d turb_linearized_ba;
        Vector3d turb_linearized_bg;

        Vector3d turb(0.0001, -0.003, 0.003);

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p + turb, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb p       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 0) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 0) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 0) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 0) * turb).transpose() << endl;
        cout << "bg diff " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff " << (step_jacobian.block<3, 3>(12, 0) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q * Quaterniond(1, turb(0) / 2, turb(1) / 2, turb(2) / 2), delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb q       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 3) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 3) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 3) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 3) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_jacobian.block<3, 3>(12, 3) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v + turb,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb v       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 6) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 6) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 6) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 6) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_jacobian.block<3, 3>(12, 6) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba + turb, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb ba       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 9) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 9) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 9) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 9) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_jacobian.block<3, 3>(12, 9) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg + turb,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb bg       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 12) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 12) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 12) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 12) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_jacobian.block<3, 3>(12, 12) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0 + turb, _gyr_0, _acc_1 , _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb acc_0       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_V.block<3, 3>(0, 0) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_V.block<3, 3>(3, 0) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_V.block<3, 3>(6, 0) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_V.block<3, 3>(9, 0) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_V.block<3, 3>(12, 0) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0 + turb, _acc_1 , _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb _gyr_0       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_V.block<3, 3>(0, 3) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_V.block<3, 3>(3, 3) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_V.block<3, 3>(6, 3) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_V.block<3, 3>(9, 3) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_V.block<3, 3>(12, 3) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1 + turb, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb acc_1       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_V.block<3, 3>(0, 6) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_V.block<3, 3>(3, 6) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_V.block<3, 3>(6, 6) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_V.block<3, 3>(9, 6) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_V.block<3, 3>(12, 6) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1 , _gyr_1 + turb, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb _gyr_1       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_V.block<3, 3>(0, 9) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_V.block<3, 3>(3, 9) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_V.block<3, 3>(6, 9) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_V.block<3, 3>(9, 9) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_V.block<3, 3>(12, 9) * turb).transpose() << endl;
    }
    */