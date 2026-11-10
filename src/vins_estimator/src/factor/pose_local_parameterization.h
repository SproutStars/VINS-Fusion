/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, HKUST
 * Licensed under the GNU General Public License v3.0.
 *******************************************************/
#pragma once
#include "../utility/utility.h"
#include <Eigen/Dense>
#include <ceres/manifold.h>

// Pose layout: [px, py, pz, qx, qy, qz, qw]. Right quaternion perturbation,
// in radians, retains VINS's normalized deltaQ retraction.
class PoseManifold final : public ceres::Manifold {
public:
  int AmbientSize() const override { return 7; }
  int TangentSize() const override { return 6; }
  bool Plus(const double * _x, const double * _delta, double * _result) const override;
  bool PlusJacobian(const double * _x, double * _jacobian) const override;
  bool Minus(const double * _y, const double * _x, double * _result) const override;
  bool MinusJacobian(const double * _x, double * _jacobian) const override;

  static Eigen::Matrix<double, 7, 6> plusJacobian(const double * _x);
  static Eigen::Matrix<double, 6, 7> minusJacobian(const double * _x);

  // Original analytic factors store tangent Jacobians in their first six
  // columns. Lift them to ambient coordinates for Ceres's real manifold.
  static void liftJacobian(const double * _x, double * _data, int _residual_size)
  {
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 7, Eigen::RowMajor>> J(_data, _residual_size, 7);
    const Eigen::MatrixXd tangent = J.leftCols<6>();
    J = tangent * minusJacobian(_x);
  }
};
