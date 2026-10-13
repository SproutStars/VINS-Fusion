/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, HKUST
 * Licensed under the GNU General Public License v3.0.
 *******************************************************/
#pragma once
#include <ceres/manifold.h>
#include <Eigen/Dense>
#include "../utility/utility.h"

// Pose layout: [px, py, pz, qx, qy, qz, qw]. Right quaternion perturbation,
// in radians, retains VINS's normalized deltaQ retraction.
class PoseManifold final : public ceres::Manifold
{
public:
    int AmbientSize() const override { return 7; }
    int TangentSize() const override { return 6; }
    bool Plus(const double *x, const double *delta, double *result) const override;
    bool PlusJacobian(const double *x, double *jacobian) const override;
    bool Minus(const double *y, const double *x, double *result) const override;
    bool MinusJacobian(const double *x, double *jacobian) const override;

    static Eigen::Matrix<double, 7, 6> plusJacobian(const double *x);
    static Eigen::Matrix<double, 6, 7> minusJacobian(const double *x);

    // Original analytic factors store tangent Jacobians in their first six
    // columns. Lift them to ambient coordinates for Ceres's real manifold.
    static void liftJacobian(const double *x, double *data, int residual_size)
    {
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 7, Eigen::RowMajor>> J(data, residual_size, 7);
        const Eigen::MatrixXd tangent = J.leftCols<6>();
        J = tangent * minusJacobian(x);
    }
};
