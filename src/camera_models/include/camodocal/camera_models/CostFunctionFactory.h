#ifndef COSTFUNCTIONFACTORY_H
#define COSTFUNCTIONFACTORY_H

#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>

#include "camodocal/camera_models/Camera.h"

namespace ceres
{
class CostFunction;
}

namespace camodocal
{

// Keep the existing enum representation for interface compatibility.
// NOLINTNEXTLINE(performance-enum-size)
enum : unsigned int {
  CAMERA_INTRINSICS = 1 << 0,
  CAMERA_POSE = 1 << 1,
  POINT_3D = 1 << 2,
  ODOMETRY_INTRINSICS = 1 << 3,
  ODOMETRY_3D_POSE = 1 << 4,
  ODOMETRY_6D_POSE = 1 << 5,
  CAMERA_ODOMETRY_TRANSFORM = 1 << 6
};

class CostFunctionFactory {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CostFunctionFactory();

  static boost::shared_ptr<CostFunctionFactory> instance();

  ceres::CostFunction * generateCostFunction(
    const CameraConstPtr & _camera, const Eigen::Vector3d & _observed_P, const Eigen::Vector2d & _observed_p,
    int _flags) const;

  ceres::CostFunction * generateCostFunction(
    const CameraConstPtr & _camera, const Eigen::Vector3d & _observed_P, const Eigen::Vector2d & _observed_p,
    const Eigen::Matrix2d & _sqrtPrecisionMat, int _flags) const;

  ceres::CostFunction * generateCostFunction(
    const CameraConstPtr & _camera, const Eigen::Vector2d & _observed_p, int _flags,
    bool _optimize_cam_odo_z = true) const;

  ceres::CostFunction * generateCostFunction(
    const CameraConstPtr & _camera, const Eigen::Vector2d & _observed_p,
    const Eigen::Matrix2d & _sqrtPrecisionMat, int _flags, bool _optimize_cam_odo_z = true) const;

  ceres::CostFunction * generateCostFunction(
    const CameraConstPtr & _camera, const Eigen::Vector3d & _odo_pos, const Eigen::Vector3d & _odo_att,
    const Eigen::Vector2d & _observed_p, int _flags, bool _optimize_cam_odo_z = true) const;

  ceres::CostFunction * generateCostFunction(
    const CameraConstPtr & _camera, const Eigen::Quaterniond & _cam_odo_q, const Eigen::Vector3d & _cam_odo_t,
    const Eigen::Vector3d & _odo_pos, const Eigen::Vector3d & _odo_att, const Eigen::Vector2d & _observed_p,
    int _flags) const;

  ceres::CostFunction * generateCostFunction(
    const CameraConstPtr & _cameraLeft, const CameraConstPtr & _cameraRight,
    const Eigen::Vector3d & _observed_P, const Eigen::Vector2d & _observed_p_left,
    const Eigen::Vector2d & _observed_p_right) const;

private:
  static boost::shared_ptr<CostFunctionFactory> m_instance_;
};

} // namespace camodocal

#endif
