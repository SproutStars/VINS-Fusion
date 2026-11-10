/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <cstring>

class Utility {
public:
  template <typename Derived>
  static Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> & _theta)
  {
    typedef typename Derived::Scalar Scalar_t;

    Eigen::Quaternion<Scalar_t> dq;
    Eigen::Matrix<Scalar_t, 3, 1> half_theta = _theta;
    half_theta /= static_cast<Scalar_t>(2.0);
    dq.w() = static_cast<Scalar_t>(1.0);
    dq.x() = half_theta.x();
    dq.y() = half_theta.y();
    dq.z() = half_theta.z();
    return dq;
  }

  template <typename Derived>
  static Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(const Eigen::MatrixBase<Derived> & _q)
  {
    Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
    ans << typename Derived::Scalar(0), -_q(2), _q(1), _q(2), typename Derived::Scalar(0), -_q(0), -_q(1),
      _q(0), typename Derived::Scalar(0);
    return ans;
  }

  template <typename Derived>
  static Eigen::Quaternion<typename Derived::Scalar> positify(const Eigen::QuaternionBase<Derived> & _q)
  {
    //printf("a: %f %f %f %f", q.w(), q.x(), q.y(), q.z());
    //Eigen::Quaternion<typename Derived::Scalar> p(-q.w(), -q.x(), -q.y(), -q.z());
    //printf("b: %f %f %f %f", p.w(), p.x(), p.y(), p.z());
    //return q.template w() >= (typename Derived::Scalar)(0.0) ? q : Eigen::Quaternion<typename Derived::Scalar>(-q.w(), -q.x(), -q.y(), -q.z());
    return _q;
  }

  template <typename Derived>
  static Eigen::Matrix<typename Derived::Scalar, 4, 4> qleft(const Eigen::QuaternionBase<Derived> & _q)
  {
    Eigen::Quaternion<typename Derived::Scalar> qq = positify(_q);
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = qq.w(), ans.template block<1, 3>(0, 1) = -qq.vec().transpose();
    ans.template block<3, 1>(1, 0) = qq.vec(),
                                ans.template block<3, 3>(1, 1) =
                                  qq.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() +
                                  skewSymmetric(qq.vec());
    return ans;
  }

  template <typename Derived>
  static Eigen::Matrix<typename Derived::Scalar, 4, 4> qright(const Eigen::QuaternionBase<Derived> & _p)
  {
    Eigen::Quaternion<typename Derived::Scalar> pp = positify(_p);
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = pp.w(), ans.template block<1, 3>(0, 1) = -pp.vec().transpose();
    ans.template block<3, 1>(1, 0) = pp.vec(),
                                ans.template block<3, 3>(1, 1) =
                                  pp.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() -
                                  skewSymmetric(pp.vec());
    return ans;
  }

  static Eigen::Vector3d r2ypr(const Eigen::Matrix3d & _R)
  {
    Eigen::Vector3d n = _R.col(0);
    Eigen::Vector3d o = _R.col(1);
    Eigen::Vector3d a = _R.col(2);

    Eigen::Vector3d ypr(3);
    double y = atan2(n(1), n(0));
    double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return ypr / M_PI * 180.0;
  }

  template <typename Derived>
  static Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(const Eigen::MatrixBase<Derived> & _ypr)
  {
    typedef typename Derived::Scalar Scalar_t;

    Scalar_t y = _ypr(0) / 180.0 * M_PI;
    Scalar_t p = _ypr(1) / 180.0 * M_PI;
    Scalar_t r = _ypr(2) / 180.0 * M_PI;

    Eigen::Matrix<Scalar_t, 3, 3> Rz;
    Rz << cos(y), -sin(y), 0, sin(y), cos(y), 0, 0, 0, 1;

    Eigen::Matrix<Scalar_t, 3, 3> Ry;
    Ry << cos(p), 0., sin(p), 0., 1., 0., -sin(p), 0., cos(p);

    Eigen::Matrix<Scalar_t, 3, 3> Rx;
    Rx << 1., 0., 0., 0., cos(r), -sin(r), 0., sin(r), cos(r);

    return Rz * Ry * Rx;
  }

  static Eigen::Matrix3d g2R(const Eigen::Vector3d & _g)
  {
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = _g.normalized();
    Eigen::Vector3d ng2{0, 0, 1.0};
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    double yaw = Utility::r2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    return R0;
  }

  template <size_t N> struct Uint_s {};

  template <size_t N, typename Lambda, typename IterT>
  void unroller(const Lambda & _f, const IterT & _iter, Uint_s<N> /*unused*/)
  {
    unroller(_f, _iter, Uint_s<N - 1>());
    _f(_iter + N);
  }

  template <typename Lambda, typename IterT>
  void unroller(const Lambda & _f, const IterT & _iter, Uint_s<0> /*unused*/)
  {
    _f(_iter);
  }

  template <typename T> static T normalizeAngle(const T & _angle_degrees)
  {
    T two_pi(2.0 * 180);
    if (_angle_degrees > 0)
      return _angle_degrees - two_pi * std::floor((_angle_degrees + T(180)) / two_pi);
    else
      return _angle_degrees + two_pi * std::floor((-_angle_degrees + T(180)) / two_pi);
  };
};
