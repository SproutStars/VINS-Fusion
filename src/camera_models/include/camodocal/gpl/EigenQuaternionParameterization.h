#pragma once
#include <ceres/manifold.h>
namespace camodocal {
// Ceres stores this quaternion in Eigen's [x,y,z,w] layout.
using EigenQuaternionParameterization = ceres::EigenQuaternionManifold;
}
