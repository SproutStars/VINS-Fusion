#include "camodocal/camera_models/CostFunctionFactory.h"

#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/EquidistantCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "camodocal/camera_models/PinholeFullCamera.h"
#include "camodocal/camera_models/ScaramuzzaCamera.h"
#include "ceres/ceres.h"

namespace camodocal
{

template <typename T>
void worldToCameraTransform(
  const T * const _q_cam_odo, const T * const _t_cam_odo, const T * const _p_odo, const T * const _att_odo,
  T * _q, T * _t, bool _optimize_cam_odo_z = true)
{
  Eigen::Quaternion<T> q_z_inv(cos(_att_odo[0] / T(2)), T(0), T(0), -sin(_att_odo[0] / T(2)));
  Eigen::Quaternion<T> q_y_inv(cos(_att_odo[1] / T(2)), T(0), -sin(_att_odo[1] / T(2)), T(0));
  Eigen::Quaternion<T> q_x_inv(cos(_att_odo[2] / T(2)), -sin(_att_odo[2] / T(2)), T(0), T(0));

  Eigen::Quaternion<T> q_zyx_inv = q_x_inv * q_y_inv * q_z_inv;

  T q_odo[4] = {q_zyx_inv.w(), q_zyx_inv.x(), q_zyx_inv.y(), q_zyx_inv.z()};

  T q_odo_cam[4] = {_q_cam_odo[3], -_q_cam_odo[0], -_q_cam_odo[1], -_q_cam_odo[2]};

  T q0[4];
  ceres::QuaternionProduct(q_odo_cam, q_odo, q0);

  T t0[3];
  T t_odo[3] = {_p_odo[0], _p_odo[1], _p_odo[2]};

  ceres::QuaternionRotatePoint(q_odo, t_odo, t0);

  t0[0] += _t_cam_odo[0];
  t0[1] += _t_cam_odo[1];

  if (_optimize_cam_odo_z) {
    t0[2] += _t_cam_odo[2];
  }

  ceres::QuaternionRotatePoint(q_odo_cam, t0, _t);
  _t[0] = -_t[0];
  _t[1] = -_t[1];
  _t[2] = -_t[2];

  // Convert quaternion from Ceres convention (w, x, y, z)
  // to Eigen convention (x, y, z, w)
  _q[0] = q0[1];
  _q[1] = q0[2];
  _q[2] = q0[3];
  _q[3] = q0[0];
}

template <class CameraT> class ReprojectionError1 {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ReprojectionError1(const Eigen::Vector3d & _observed_P, const Eigen::Vector2d & _observed_p)
    : m_observed_P_(_observed_P), m_observed_p_(_observed_p), m_sqrtPrecisionMat_(Eigen::Matrix2d::Identity())
  {
  }

  ReprojectionError1(
    const Eigen::Vector3d & _observed_P, const Eigen::Vector2d & _observed_p,
    const Eigen::Matrix2d & _sqrtPrecisionMat)
    : m_observed_P_(_observed_P), m_observed_p_(_observed_p), m_sqrtPrecisionMat_(_sqrtPrecisionMat)
  {
  }

  ReprojectionError1(
    const std::vector<double> & _intrinsic_params, const Eigen::Vector3d & _observed_P,
    const Eigen::Vector2d & _observed_p)
    : m_intrinsic_params_(_intrinsic_params), m_observed_P_(_observed_P), m_observed_p_(_observed_p)
  {
  }

  // variables: camera intrinsics and camera extrinsics
  template <typename T>
  bool operator()(const T * const _intrinsic_params, const T * const _q, const T * const _t, T * _residuals) const
  {
    Eigen::Matrix<T, 3, 1> P = m_observed_P_.cast<T>();

    Eigen::Matrix<T, 2, 1> predicted_p;
    CameraT::spaceToPlane(_intrinsic_params, _q, _t, P, predicted_p);

    Eigen::Matrix<T, 2, 1> e = predicted_p - m_observed_p_.cast<T>();

    Eigen::Matrix<T, 2, 1> e_weighted = m_sqrtPrecisionMat_.cast<T>() * e;

    _residuals[0] = e_weighted(0);
    _residuals[1] = e_weighted(1);

    return true;
  }

  // variables: camera-odometry transforms and odometry poses
  template <typename T>
  bool operator()(
    const T * const _q_cam_odo, const T * const _t_cam_odo, const T * const _p_odo, const T * const _att_odo,
    T * _residuals) const
  {
    T q[4], t[3];
    worldToCameraTransform(_q_cam_odo, _t_cam_odo, _p_odo, _att_odo, q, t);

    Eigen::Matrix<T, 3, 1> P = m_observed_P_.cast<T>();

    std::vector<T> intrinsic_params(m_intrinsic_params_.begin(), m_intrinsic_params_.end());

    // project 3D object point to the image plane
    Eigen::Matrix<T, 2, 1> predicted_p;
    CameraT::spaceToPlane(intrinsic_params.data(), q, t, P, predicted_p);

    _residuals[0] = predicted_p(0) - T(m_observed_p_(0));
    _residuals[1] = predicted_p(1) - T(m_observed_p_(1));

    return true;
  }

  // private:
  // camera intrinsics
  std::vector<double> m_intrinsic_params_;

  // observed 3D point
  Eigen::Vector3d m_observed_P_;

  // observed 2D point
  Eigen::Vector2d m_observed_p_;

  // square root of precision matrix
  Eigen::Matrix2d m_sqrtPrecisionMat_;
};

// variables: camera extrinsics, 3D point
template <class CameraT> class ReprojectionError2 {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ReprojectionError2(const std::vector<double> & _intrinsic_params, const Eigen::Vector2d & _observed_p)
    : m_intrinsic_params_(_intrinsic_params), m_observed_p_(_observed_p)
  {
  }

  template <typename T>
  bool operator()(const T * const _q, const T * const _t, const T * const _point, T * _residuals) const
  {
    Eigen::Matrix<T, 3, 1> P;
    P(0) = T(_point[0]);
    P(1) = T(_point[1]);
    P(2) = T(_point[2]);

    std::vector<T> intrinsic_params(m_intrinsic_params_.begin(), m_intrinsic_params_.end());

    // project 3D object point to the image plane
    Eigen::Matrix<T, 2, 1> predicted_p;
    CameraT::spaceToPlane(intrinsic_params.data(), _q, _t, P, predicted_p);

    _residuals[0] = predicted_p(0) - T(m_observed_p_(0));
    _residuals[1] = predicted_p(1) - T(m_observed_p_(1));

    return true;
  }

private:
  // camera intrinsics
  std::vector<double> m_intrinsic_params_;

  // observed 2D point
  Eigen::Vector2d m_observed_p_;
};

template <class CameraT> class ReprojectionError3 {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ReprojectionError3(const Eigen::Vector2d & _observed_p)
    : m_observed_p_(_observed_p), m_sqrtPrecisionMat_(Eigen::Matrix2d::Identity()), m_optimize_cam_odo_z_(true)
  {
  }

  ReprojectionError3(const Eigen::Vector2d & _observed_p, const Eigen::Matrix2d & _sqrtPrecisionMat)
    : m_observed_p_(_observed_p), m_sqrtPrecisionMat_(_sqrtPrecisionMat), m_optimize_cam_odo_z_(true)
  {
  }

  ReprojectionError3(const std::vector<double> & _intrinsic_params, const Eigen::Vector2d & _observed_p)
    : m_intrinsic_params_(_intrinsic_params), m_observed_p_(_observed_p),
      m_sqrtPrecisionMat_(Eigen::Matrix2d::Identity()), m_optimize_cam_odo_z_(true)
  {
  }

  ReprojectionError3(
    const std::vector<double> & _intrinsic_params, const Eigen::Vector2d & _observed_p,
    const Eigen::Matrix2d & _sqrtPrecisionMat)
    : m_intrinsic_params_(_intrinsic_params), m_observed_p_(_observed_p),
      m_sqrtPrecisionMat_(_sqrtPrecisionMat), m_optimize_cam_odo_z_(true)
  {
  }

  ReprojectionError3(
    const std::vector<double> & _intrinsic_params, const Eigen::Vector3d & _odo_pos,
    const Eigen::Vector3d & _odo_att, const Eigen::Vector2d & _observed_p, bool _optimize_cam_odo_z)
    : m_intrinsic_params_(_intrinsic_params), m_odo_pos_(_odo_pos), m_odo_att_(_odo_att),
      m_observed_p_(_observed_p), m_optimize_cam_odo_z_(_optimize_cam_odo_z)
  {
  }

  ReprojectionError3(
    const std::vector<double> & _intrinsic_params, const Eigen::Quaterniond & _cam_odo_q,
    const Eigen::Vector3d & _cam_odo_t, const Eigen::Vector3d & _odo_pos, const Eigen::Vector3d & _odo_att,
    const Eigen::Vector2d & _observed_p)
    : m_intrinsic_params_(_intrinsic_params), m_cam_odo_q_(_cam_odo_q), m_cam_odo_t_(_cam_odo_t),
      m_odo_pos_(_odo_pos), m_odo_att_(_odo_att), m_observed_p_(_observed_p), m_optimize_cam_odo_z_(true)
  {
  }

  // variables: camera intrinsics, camera-to-odometry transform,
  //            odometry extrinsics, 3D point
  template <typename T>
  bool operator()(
    const T * const _intrinsic_params, const T * const _q_cam_odo, const T * const _t_cam_odo,
    const T * const _p_odo, const T * const _att_odo, const T * const _point, T * _residuals) const
  {
    T q[4], t[3];
    worldToCameraTransform(_q_cam_odo, _t_cam_odo, _p_odo, _att_odo, q, t, m_optimize_cam_odo_z_);

    Eigen::Matrix<T, 3, 1> P(_point[0], _point[1], _point[2]);

    // project 3D object point to the image plane
    Eigen::Matrix<T, 2, 1> predicted_p;
    CameraT::spaceToPlane(_intrinsic_params, q, t, P, predicted_p);

    Eigen::Matrix<T, 2, 1> err = predicted_p - m_observed_p_.cast<T>();
    Eigen::Matrix<T, 2, 1> err_weighted = m_sqrtPrecisionMat_.cast<T>() * err;

    _residuals[0] = err_weighted(0);
    _residuals[1] = err_weighted(1);

    return true;
  }

  // variables: camera-to-odometry transform, 3D point
  template <typename T>
  bool
  operator()(const T * const _q_cam_odo, const T * const _t_cam_odo, const T * const _point, T * _residuals) const
  {
    T p_odo[3] = {T(m_odo_pos_(0)), T(m_odo_pos_(1)), T(m_odo_pos_(2))};
    T att_odo[3] = {T(m_odo_att_(0)), T(m_odo_att_(1)), T(m_odo_att_(2))};
    T q[4], t[3];

    worldToCameraTransform(_q_cam_odo, _t_cam_odo, p_odo, att_odo, q, t, m_optimize_cam_odo_z_);

    std::vector<T> intrinsic_params(m_intrinsic_params_.begin(), m_intrinsic_params_.end());
    Eigen::Matrix<T, 3, 1> P(_point[0], _point[1], _point[2]);

    // project 3D object point to the image plane
    Eigen::Matrix<T, 2, 1> predicted_p;
    CameraT::spaceToPlane(intrinsic_params.data(), q, t, P, predicted_p);

    _residuals[0] = predicted_p(0) - T(m_observed_p_(0));
    _residuals[1] = predicted_p(1) - T(m_observed_p_(1));

    return true;
  }

  // variables: camera-to-odometry transform, odometry extrinsics, 3D point
  template <typename T>
  bool operator()(
    const T * const _q_cam_odo, const T * const _t_cam_odo, const T * const _p_odo, const T * const _att_odo,
    const T * const _point, T * _residuals) const
  {
    T q[4], t[3];
    worldToCameraTransform(_q_cam_odo, _t_cam_odo, _p_odo, _att_odo, q, t, m_optimize_cam_odo_z_);

    std::vector<T> intrinsic_params(m_intrinsic_params_.begin(), m_intrinsic_params_.end());
    Eigen::Matrix<T, 3, 1> P(_point[0], _point[1], _point[2]);

    // project 3D object point to the image plane
    Eigen::Matrix<T, 2, 1> predicted_p;
    CameraT::spaceToPlane(intrinsic_params.data(), q, t, P, predicted_p);

    Eigen::Matrix<T, 2, 1> err = predicted_p - m_observed_p_.cast<T>();
    Eigen::Matrix<T, 2, 1> err_weighted = m_sqrtPrecisionMat_.cast<T>() * err;

    _residuals[0] = err_weighted(0);
    _residuals[1] = err_weighted(1);

    return true;
  }

  // variables: 3D point
  template <typename T> bool operator()(const T * const _point, T * _residuals) const
  {
    T q_cam_odo[4] = {
      T(m_cam_odo_q_.coeffs()(0)), T(m_cam_odo_q_.coeffs()(1)), T(m_cam_odo_q_.coeffs()(2)),
      T(m_cam_odo_q_.coeffs()(3))};
    T t_cam_odo[3] = {T(m_cam_odo_t_(0)), T(m_cam_odo_t_(1)), T(m_cam_odo_t_(2))};
    T p_odo[3] = {T(m_odo_pos_(0)), T(m_odo_pos_(1)), T(m_odo_pos_(2))};
    T att_odo[3] = {T(m_odo_att_(0)), T(m_odo_att_(1)), T(m_odo_att_(2))};
    T q[4], t[3];

    worldToCameraTransform(q_cam_odo, t_cam_odo, p_odo, att_odo, q, t, m_optimize_cam_odo_z_);

    std::vector<T> intrinsic_params(m_intrinsic_params_.begin(), m_intrinsic_params_.end());
    Eigen::Matrix<T, 3, 1> P(_point[0], _point[1], _point[2]);

    // project 3D object point to the image plane
    Eigen::Matrix<T, 2, 1> predicted_p;
    CameraT::spaceToPlane(intrinsic_params.data(), q, t, P, predicted_p);

    _residuals[0] = predicted_p(0) - T(m_observed_p_(0));
    _residuals[1] = predicted_p(1) - T(m_observed_p_(1));

    return true;
  }

private:
  // camera intrinsics
  std::vector<double> m_intrinsic_params_;

  // observed camera-odometry transform
  Eigen::Quaterniond m_cam_odo_q_;
  Eigen::Vector3d m_cam_odo_t_;

  // observed odometry
  Eigen::Vector3d m_odo_pos_;
  Eigen::Vector3d m_odo_att_;

  // observed 2D point
  Eigen::Vector2d m_observed_p_;

  Eigen::Matrix2d m_sqrtPrecisionMat_;

  bool m_optimize_cam_odo_z_;
};

// variables: camera intrinsics and camera extrinsics
template <class CameraT> class StereoReprojectionError {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  StereoReprojectionError(
    const Eigen::Vector3d & _observed_P, const Eigen::Vector2d & _observed_p_l,
    const Eigen::Vector2d & _observed_p_r)
    : m_observed_P_(_observed_P), m_observed_p_l_(_observed_p_l), m_observed_p_r_(_observed_p_r)
  {
  }

  template <typename T>
  bool operator()(
    const T * const _intrinsic_params_l, const T * const _intrinsic_params_r, const T * const _q_l,
    const T * const _t_l, const T * const _q_l_r, const T * const _t_l_r, T * _residuals) const
  {
    Eigen::Matrix<T, 3, 1> P;
    P(0) = T(m_observed_P_(0));
    P(1) = T(m_observed_P_(1));
    P(2) = T(m_observed_P_(2));

    Eigen::Matrix<T, 2, 1> predicted_p_l;
    CameraT::spaceToPlane(_intrinsic_params_l, _q_l, _t_l, P, predicted_p_l);

    Eigen::Quaternion<T> q_r = Eigen::Quaternion<T>(_q_l_r) * Eigen::Quaternion<T>(_q_l);

    Eigen::Matrix<T, 3, 1> t_r;
    t_r(0) = _t_l[0];
    t_r(1) = _t_l[1];
    t_r(2) = _t_l[2];

    t_r = Eigen::Quaternion<T>(_q_l_r) * t_r;
    t_r(0) += _t_l_r[0];
    t_r(1) += _t_l_r[1];
    t_r(2) += _t_l_r[2];

    Eigen::Matrix<T, 2, 1> predicted_p_r;
    CameraT::spaceToPlane(_intrinsic_params_r, q_r.coeffs().data(), t_r.data(), P, predicted_p_r);

    _residuals[0] = predicted_p_l(0) - T(m_observed_p_l_(0));
    _residuals[1] = predicted_p_l(1) - T(m_observed_p_l_(1));
    _residuals[2] = predicted_p_r(0) - T(m_observed_p_r_(0));
    _residuals[3] = predicted_p_r(1) - T(m_observed_p_r_(1));

    return true;
  }

private:
  // observed 3D point
  Eigen::Vector3d m_observed_P_;

  // observed 2D point
  Eigen::Vector2d m_observed_p_l_;
  Eigen::Vector2d m_observed_p_r_;
};

template <class CameraT> class ComprehensionError {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ComprehensionError(const Eigen::Vector3d & _observed_P, const Eigen::Vector2d & _observed_p)
    : m_observed_P_(_observed_P), m_observed_p_(_observed_p), m_sqrtPrecisionMat_(Eigen::Matrix2d::Identity())
  {
  }

  template <typename T>
  bool operator()(const T * const _intrinsic_params, const T * const _q, const T * const _t, T * _residuals) const
  {
    {
      Eigen::Matrix<T, 2, 1> p = m_observed_p_.cast<T>();

      Eigen::Matrix<T, 3, 1> predicted_img_P;
      CameraT::liftToSphere(_intrinsic_params, p, predicted_img_P);

      Eigen::Matrix<T, 2, 1> predicted_p;
      CameraT::sphereToPlane(_intrinsic_params, predicted_img_P, predicted_p);

      Eigen::Matrix<T, 2, 1> e = predicted_p - m_observed_p_.cast<T>();

      Eigen::Matrix<T, 2, 1> e_weighted = m_sqrtPrecisionMat_.cast<T>() * e;

      _residuals[0] = e_weighted(0);
      _residuals[1] = e_weighted(1);
    }

    {
      Eigen::Matrix<T, 3, 1> P = m_observed_P_.cast<T>();

      Eigen::Matrix<T, 2, 1> predicted_p;
      CameraT::spaceToPlane(_intrinsic_params, _q, _t, P, predicted_p);

      Eigen::Matrix<T, 2, 1> e = predicted_p - m_observed_p_.cast<T>();

      Eigen::Matrix<T, 2, 1> e_weighted = m_sqrtPrecisionMat_.cast<T>() * e;

      _residuals[2] = e_weighted(0);
      _residuals[3] = e_weighted(1);
    }

    return true;
  }

  // private:
  // camera intrinsics
  // std::vector<double> m_intrinsic_params;

  // observed 3D point
  Eigen::Vector3d m_observed_P_;

  // observed 2D point
  Eigen::Vector2d m_observed_p_;

  // square root of precision matrix
  Eigen::Matrix2d m_sqrtPrecisionMat_;
};

boost::shared_ptr<CostFunctionFactory> CostFunctionFactory::m_instance_;

CostFunctionFactory::CostFunctionFactory() = default;

boost::shared_ptr<CostFunctionFactory> CostFunctionFactory::instance()
{
  if (m_instance_.get() == nullptr) {
    m_instance_.reset(new CostFunctionFactory);
  }

  return m_instance_;
}

ceres::CostFunction * CostFunctionFactory::generateCostFunction(
  const CameraConstPtr & _camera, const Eigen::Vector3d & _observed_P, const Eigen::Vector2d & _observed_p,
  int _flags) const
{
  ceres::CostFunction * costFunction = nullptr;

  std::vector<double> intrinsic_params;
  _camera->writeParameters(intrinsic_params);

  switch (_flags) {
  case CAMERA_INTRINSICS | CAMERA_POSE:
    switch (_camera->modelType()) {
    case Camera::KANNALA_BRANDT:
      costFunction = new ceres::AutoDiffCostFunction<ReprojectionError1<EquidistantCamera>, 2, 8, 4, 3>(
        new ReprojectionError1<EquidistantCamera>(_observed_P, _observed_p));
      break;
    case Camera::PINHOLE:
      costFunction = new ceres::AutoDiffCostFunction<ReprojectionError1<PinholeCamera>, 2, 8, 4, 3>(
        new ReprojectionError1<PinholeCamera>(_observed_P, _observed_p));
      break;
    case Camera::PINHOLE_FULL:
      costFunction = new ceres::AutoDiffCostFunction<ReprojectionError1<PinholeFullCamera>, 2, 12, 4, 3>(
        new ReprojectionError1<PinholeFullCamera>(_observed_P, _observed_p));
      break;
    case Camera::MEI:
      costFunction = new ceres::AutoDiffCostFunction<ReprojectionError1<CataCamera>, 2, 9, 4, 3>(
        new ReprojectionError1<CataCamera>(_observed_P, _observed_p));
      break;
    case Camera::SCARAMUZZA:
      costFunction =
        new ceres::AutoDiffCostFunction<ComprehensionError<OCAMCamera>, 4, SCARAMUZZA_CAMERA_NUM_PARAMS, 4, 3>(
          new ComprehensionError<OCAMCamera>(_observed_P, _observed_p));
      // new ceres::AutoDiffCostFunction<ReprojectionError1<OCAMCamera>, 2,
      // SCARAMUZZA_CAMERA_NUM_PARAMS, 4, 3>( new
      // ReprojectionError1<OCAMCamera>(observed_P, observed_p));
      break;
    }
    break;
  case CAMERA_ODOMETRY_TRANSFORM | ODOMETRY_6D_POSE:
    switch (_camera->modelType()) {
    case Camera::KANNALA_BRANDT:
      costFunction = new ceres::AutoDiffCostFunction<ReprojectionError1<EquidistantCamera>, 2, 4, 3, 3, 3>(
        new ReprojectionError1<EquidistantCamera>(intrinsic_params, _observed_P, _observed_p));
      break;
    case Camera::PINHOLE:
      costFunction = new ceres::AutoDiffCostFunction<ReprojectionError1<PinholeCamera>, 2, 4, 3, 3, 3>(
        new ReprojectionError1<PinholeCamera>(intrinsic_params, _observed_P, _observed_p));
      break;
    case Camera::PINHOLE_FULL:
      costFunction = new ceres::AutoDiffCostFunction<ReprojectionError1<PinholeFullCamera>, 2, 4, 3, 3, 3>(
        new ReprojectionError1<PinholeFullCamera>(intrinsic_params, _observed_P, _observed_p));
      break;
    case Camera::MEI:
      costFunction = new ceres::AutoDiffCostFunction<ReprojectionError1<CataCamera>, 2, 4, 3, 3, 3>(
        new ReprojectionError1<CataCamera>(intrinsic_params, _observed_P, _observed_p));
      break;
    case Camera::SCARAMUZZA:
      costFunction = new ceres::AutoDiffCostFunction<ReprojectionError1<OCAMCamera>, 2, 4, 3, 3, 3>(
        new ReprojectionError1<OCAMCamera>(intrinsic_params, _observed_P, _observed_p));
      break;
    }
    break;
  }

  return costFunction;
}

ceres::CostFunction * CostFunctionFactory::generateCostFunction(
  const CameraConstPtr & _camera, const Eigen::Vector3d & _observed_P, const Eigen::Vector2d & _observed_p,
  const Eigen::Matrix2d & _sqrtPrecisionMat, int _flags) const
{
  ceres::CostFunction * costFunction = nullptr;

  std::vector<double> intrinsic_params;
  _camera->writeParameters(intrinsic_params);

  switch (_flags) {
  case CAMERA_INTRINSICS | CAMERA_POSE:
    switch (_camera->modelType()) {
    case Camera::KANNALA_BRANDT:
      costFunction = new ceres::AutoDiffCostFunction<ReprojectionError1<EquidistantCamera>, 2, 8, 4, 3>(
        new ReprojectionError1<EquidistantCamera>(_observed_P, _observed_p, _sqrtPrecisionMat));
      break;
    case Camera::PINHOLE:
      costFunction = new ceres::AutoDiffCostFunction<ReprojectionError1<PinholeCamera>, 2, 8, 4, 3>(
        new ReprojectionError1<PinholeCamera>(_observed_P, _observed_p, _sqrtPrecisionMat));
      break;
    case Camera::PINHOLE_FULL:
      costFunction = new ceres::AutoDiffCostFunction<ReprojectionError1<PinholeFullCamera>, 2, 12, 4, 3>(
        new ReprojectionError1<PinholeFullCamera>(_observed_P, _observed_p, _sqrtPrecisionMat));
      break;
    case Camera::MEI:
      costFunction = new ceres::AutoDiffCostFunction<ReprojectionError1<CataCamera>, 2, 9, 4, 3>(
        new ReprojectionError1<CataCamera>(_observed_P, _observed_p, _sqrtPrecisionMat));
      break;
    case Camera::SCARAMUZZA:
      costFunction =
        new ceres::AutoDiffCostFunction<ReprojectionError1<OCAMCamera>, 2, SCARAMUZZA_CAMERA_NUM_PARAMS, 4, 3>(
          new ReprojectionError1<OCAMCamera>(_observed_P, _observed_p, _sqrtPrecisionMat));
      break;
    }
    break;
  }

  return costFunction;
}

ceres::CostFunction * CostFunctionFactory::generateCostFunction(
  const CameraConstPtr & _camera, const Eigen::Vector2d & _observed_p, int _flags, bool _optimize_cam_odo_z) const
{
  ceres::CostFunction * costFunction = nullptr;

  std::vector<double> intrinsic_params;
  _camera->writeParameters(intrinsic_params);

  switch (_flags) {
  case CAMERA_POSE | POINT_3D:
    switch (_camera->modelType()) {
    case Camera::KANNALA_BRANDT:
      costFunction = new ceres::AutoDiffCostFunction<ReprojectionError2<EquidistantCamera>, 2, 4, 3, 3>(
        new ReprojectionError2<EquidistantCamera>(intrinsic_params, _observed_p));
      break;
    case Camera::PINHOLE:
      costFunction = new ceres::AutoDiffCostFunction<ReprojectionError2<PinholeCamera>, 2, 4, 3, 3>(
        new ReprojectionError2<PinholeCamera>(intrinsic_params, _observed_p));
      break;
    case Camera::PINHOLE_FULL:
      costFunction = new ceres::AutoDiffCostFunction<ReprojectionError2<PinholeFullCamera>, 2, 4, 3, 3>(
        new ReprojectionError2<PinholeFullCamera>(intrinsic_params, _observed_p));
      break;
    case Camera::MEI:
      costFunction = new ceres::AutoDiffCostFunction<ReprojectionError2<CataCamera>, 2, 4, 3, 3>(
        new ReprojectionError2<CataCamera>(intrinsic_params, _observed_p));
      break;
    case Camera::SCARAMUZZA:
      costFunction = new ceres::AutoDiffCostFunction<ReprojectionError2<OCAMCamera>, 2, 4, 3, 3>(
        new ReprojectionError2<OCAMCamera>(intrinsic_params, _observed_p));
      break;
    }
    break;
  case CAMERA_ODOMETRY_TRANSFORM | ODOMETRY_3D_POSE | POINT_3D:
    switch (_camera->modelType()) {
    case Camera::KANNALA_BRANDT:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<EquidistantCamera>, 2, 4, 3, 2, 1, 3>(
          new ReprojectionError3<EquidistantCamera>(intrinsic_params, _observed_p));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 4, 2, 2, 1, 3>(
          new ReprojectionError3<PinholeCamera>(intrinsic_params, _observed_p));
      }
      break;
    case Camera::PINHOLE:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 4, 3, 2, 1, 3>(
          new ReprojectionError3<PinholeCamera>(intrinsic_params, _observed_p));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 4, 2, 2, 1, 3>(
          new ReprojectionError3<PinholeCamera>(intrinsic_params, _observed_p));
      }
      break;
    case Camera::PINHOLE_FULL:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeFullCamera>, 2, 4, 3, 2, 1, 3>(
          new ReprojectionError3<PinholeFullCamera>(intrinsic_params, _observed_p));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeFullCamera>, 2, 4, 2, 2, 1, 3>(
          new ReprojectionError3<PinholeFullCamera>(intrinsic_params, _observed_p));
      }
      break;
    case Camera::MEI:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 4, 3, 2, 1, 3>(
          new ReprojectionError3<CataCamera>(intrinsic_params, _observed_p));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 4, 2, 2, 1, 3>(
          new ReprojectionError3<CataCamera>(intrinsic_params, _observed_p));
      }
      break;
    case Camera::SCARAMUZZA:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<OCAMCamera>, 2, 4, 3, 2, 1, 3>(
          new ReprojectionError3<OCAMCamera>(intrinsic_params, _observed_p));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<OCAMCamera>, 2, 4, 2, 2, 1, 3>(
          new ReprojectionError3<OCAMCamera>(intrinsic_params, _observed_p));
      }
      break;
    }
    break;
  case CAMERA_ODOMETRY_TRANSFORM | ODOMETRY_6D_POSE | POINT_3D:
    switch (_camera->modelType()) {
    case Camera::KANNALA_BRANDT:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<EquidistantCamera>, 2, 4, 3, 3, 3, 3>(
          new ReprojectionError3<EquidistantCamera>(intrinsic_params, _observed_p));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 4, 2, 3, 3, 3>(
          new ReprojectionError3<PinholeCamera>(intrinsic_params, _observed_p));
      }
      break;
    case Camera::PINHOLE:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 4, 3, 3, 3, 3>(
          new ReprojectionError3<PinholeCamera>(intrinsic_params, _observed_p));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 4, 2, 3, 3, 3>(
          new ReprojectionError3<PinholeCamera>(intrinsic_params, _observed_p));
      }
      break;
    case Camera::PINHOLE_FULL:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeFullCamera>, 2, 4, 3, 3, 3, 3>(
          new ReprojectionError3<PinholeFullCamera>(intrinsic_params, _observed_p));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeFullCamera>, 2, 4, 2, 3, 3, 3>(
          new ReprojectionError3<PinholeFullCamera>(intrinsic_params, _observed_p));
      }
      break;
    case Camera::MEI:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 4, 3, 3, 3, 3>(
          new ReprojectionError3<CataCamera>(intrinsic_params, _observed_p));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 4, 2, 3, 3, 3>(
          new ReprojectionError3<CataCamera>(intrinsic_params, _observed_p));
      }
      break;
    case Camera::SCARAMUZZA:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<OCAMCamera>, 2, 4, 3, 3, 3, 3>(
          new ReprojectionError3<OCAMCamera>(intrinsic_params, _observed_p));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<OCAMCamera>, 2, 4, 2, 3, 3, 3>(
          new ReprojectionError3<OCAMCamera>(intrinsic_params, _observed_p));
      }
      break;
    }
    break;
  case CAMERA_INTRINSICS | CAMERA_ODOMETRY_TRANSFORM | ODOMETRY_3D_POSE | POINT_3D:
    switch (_camera->modelType()) {
    case Camera::KANNALA_BRANDT:
      if (_optimize_cam_odo_z) {
        costFunction =
          new ceres::AutoDiffCostFunction<ReprojectionError3<EquidistantCamera>, 2, 8, 4, 3, 2, 1, 3>(
            new ReprojectionError3<EquidistantCamera>(_observed_p));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 8, 4, 2, 2, 1, 3>(
          new ReprojectionError3<PinholeCamera>(_observed_p));
      }
      break;
    case Camera::PINHOLE:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 8, 4, 3, 2, 1, 3>(
          new ReprojectionError3<PinholeCamera>(_observed_p));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 8, 4, 2, 2, 1, 3>(
          new ReprojectionError3<PinholeCamera>(_observed_p));
      }
      break;
    case Camera::PINHOLE_FULL:
      if (_optimize_cam_odo_z) {
        costFunction =
          new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeFullCamera>, 2, 12, 4, 3, 2, 1, 3>(
            new ReprojectionError3<PinholeFullCamera>(_observed_p));
      } else {
        costFunction =
          new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeFullCamera>, 2, 12, 4, 2, 2, 1, 3>(
            new ReprojectionError3<PinholeFullCamera>(_observed_p));
      }
      break;
    case Camera::MEI:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 9, 4, 3, 2, 1, 3>(
          new ReprojectionError3<CataCamera>(_observed_p));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 9, 4, 2, 2, 1, 3>(
          new ReprojectionError3<CataCamera>(_observed_p));
      }
      break;
    case Camera::SCARAMUZZA:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<
          ReprojectionError3<OCAMCamera>, 2, SCARAMUZZA_CAMERA_NUM_PARAMS, 4, 3, 2, 1, 3>(
          new ReprojectionError3<OCAMCamera>(_observed_p));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<
          ReprojectionError3<OCAMCamera>, 2, SCARAMUZZA_CAMERA_NUM_PARAMS, 4, 2, 2, 1, 3>(
          new ReprojectionError3<OCAMCamera>(_observed_p));
      }
      break;
    }
    break;
  case CAMERA_INTRINSICS | CAMERA_ODOMETRY_TRANSFORM | ODOMETRY_6D_POSE | POINT_3D:
    switch (_camera->modelType()) {
    case Camera::KANNALA_BRANDT:
      if (_optimize_cam_odo_z) {
        costFunction =
          new ceres::AutoDiffCostFunction<ReprojectionError3<EquidistantCamera>, 2, 8, 4, 3, 3, 3, 3>(
            new ReprojectionError3<EquidistantCamera>(_observed_p));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 8, 4, 2, 3, 3, 3>(
          new ReprojectionError3<PinholeCamera>(_observed_p));
      }
      break;
    case Camera::PINHOLE:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 8, 4, 3, 3, 3, 3>(
          new ReprojectionError3<PinholeCamera>(_observed_p));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 8, 4, 2, 3, 3, 3>(
          new ReprojectionError3<PinholeCamera>(_observed_p));
      }
      break;
    case Camera::PINHOLE_FULL:
      if (_optimize_cam_odo_z) {
        costFunction =
          new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeFullCamera>, 2, 12, 4, 3, 3, 3, 3>(
            new ReprojectionError3<PinholeFullCamera>(_observed_p));
      } else {
        costFunction =
          new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeFullCamera>, 2, 12, 4, 2, 3, 3, 3>(
            new ReprojectionError3<PinholeFullCamera>(_observed_p));
      }
      break;
    case Camera::MEI:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 9, 4, 3, 3, 3, 3>(
          new ReprojectionError3<CataCamera>(_observed_p));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 9, 4, 2, 3, 3, 3>(
          new ReprojectionError3<CataCamera>(_observed_p));
      }
      break;
    case Camera::SCARAMUZZA:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<
          ReprojectionError3<OCAMCamera>, 2, SCARAMUZZA_CAMERA_NUM_PARAMS, 4, 3, 3, 3, 3>(
          new ReprojectionError3<OCAMCamera>(_observed_p));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<
          ReprojectionError3<OCAMCamera>, 2, SCARAMUZZA_CAMERA_NUM_PARAMS, 4, 2, 3, 3, 3>(
          new ReprojectionError3<OCAMCamera>(_observed_p));
      }
      break;
    }
    break;
  }

  return costFunction;
}

ceres::CostFunction * CostFunctionFactory::generateCostFunction(
  const CameraConstPtr & _camera, const Eigen::Vector2d & _observed_p,
  const Eigen::Matrix2d & _sqrtPrecisionMat, int _flags, bool _optimize_cam_odo_z) const
{
  ceres::CostFunction * costFunction = nullptr;

  std::vector<double> intrinsic_params;
  _camera->writeParameters(intrinsic_params);

  switch (_flags) {
  case CAMERA_ODOMETRY_TRANSFORM | ODOMETRY_6D_POSE | POINT_3D:
    switch (_camera->modelType()) {
    case Camera::KANNALA_BRANDT:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<EquidistantCamera>, 2, 4, 3, 3, 3, 3>(
          new ReprojectionError3<EquidistantCamera>(intrinsic_params, _observed_p, _sqrtPrecisionMat));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 4, 2, 3, 3, 3>(
          new ReprojectionError3<PinholeCamera>(intrinsic_params, _observed_p, _sqrtPrecisionMat));
      }
      break;
    case Camera::PINHOLE:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 4, 3, 3, 3, 3>(
          new ReprojectionError3<PinholeCamera>(intrinsic_params, _observed_p, _sqrtPrecisionMat));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 4, 2, 3, 3, 3>(
          new ReprojectionError3<PinholeCamera>(intrinsic_params, _observed_p, _sqrtPrecisionMat));
      }
      break;
    case Camera::PINHOLE_FULL:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeFullCamera>, 2, 4, 3, 3, 3, 3>(
          new ReprojectionError3<PinholeFullCamera>(intrinsic_params, _observed_p, _sqrtPrecisionMat));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeFullCamera>, 2, 4, 2, 3, 3, 3>(
          new ReprojectionError3<PinholeFullCamera>(intrinsic_params, _observed_p, _sqrtPrecisionMat));
      }
      break;
    case Camera::MEI:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 4, 3, 3, 3, 3>(
          new ReprojectionError3<CataCamera>(intrinsic_params, _observed_p, _sqrtPrecisionMat));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 4, 2, 3, 3, 3>(
          new ReprojectionError3<CataCamera>(intrinsic_params, _observed_p, _sqrtPrecisionMat));
      }
      break;
    case Camera::SCARAMUZZA:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<OCAMCamera>, 2, 4, 3, 3, 3, 3>(
          new ReprojectionError3<OCAMCamera>(intrinsic_params, _observed_p, _sqrtPrecisionMat));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<OCAMCamera>, 2, 4, 2, 3, 3, 3>(
          new ReprojectionError3<OCAMCamera>(intrinsic_params, _observed_p, _sqrtPrecisionMat));
      }
      break;
    }
    break;
  case CAMERA_INTRINSICS | CAMERA_ODOMETRY_TRANSFORM | ODOMETRY_6D_POSE | POINT_3D:
    switch (_camera->modelType()) {
    case Camera::KANNALA_BRANDT:
      if (_optimize_cam_odo_z) {
        costFunction =
          new ceres::AutoDiffCostFunction<ReprojectionError3<EquidistantCamera>, 2, 8, 4, 3, 3, 3, 3>(
            new ReprojectionError3<EquidistantCamera>(_observed_p, _sqrtPrecisionMat));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 8, 4, 2, 3, 3, 3>(
          new ReprojectionError3<PinholeCamera>(_observed_p, _sqrtPrecisionMat));
      }
      break;
    case Camera::PINHOLE:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 8, 4, 3, 3, 3, 3>(
          new ReprojectionError3<PinholeCamera>(_observed_p, _sqrtPrecisionMat));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 8, 4, 2, 3, 3, 3>(
          new ReprojectionError3<PinholeCamera>(_observed_p, _sqrtPrecisionMat));
      }
      break;
    case Camera::PINHOLE_FULL:
      if (_optimize_cam_odo_z) {
        costFunction =
          new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeFullCamera>, 2, 12, 4, 3, 3, 3, 3>(
            new ReprojectionError3<PinholeFullCamera>(_observed_p, _sqrtPrecisionMat));
      } else {
        costFunction =
          new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeFullCamera>, 2, 12, 4, 2, 3, 3, 3>(
            new ReprojectionError3<PinholeFullCamera>(_observed_p, _sqrtPrecisionMat));
      }
      break;
    case Camera::MEI:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 9, 4, 3, 3, 3, 3>(
          new ReprojectionError3<CataCamera>(_observed_p, _sqrtPrecisionMat));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 9, 4, 2, 3, 3, 3>(
          new ReprojectionError3<CataCamera>(_observed_p, _sqrtPrecisionMat));
      }
      break;
    case Camera::SCARAMUZZA:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<
          ReprojectionError3<OCAMCamera>, 2, SCARAMUZZA_CAMERA_NUM_PARAMS, 4, 3, 3, 3, 3>(
          new ReprojectionError3<OCAMCamera>(_observed_p, _sqrtPrecisionMat));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<
          ReprojectionError3<OCAMCamera>, 2, SCARAMUZZA_CAMERA_NUM_PARAMS, 4, 2, 3, 3, 3>(
          new ReprojectionError3<OCAMCamera>(_observed_p, _sqrtPrecisionMat));
      }
      break;
    }
    break;
  }

  return costFunction;
}

ceres::CostFunction * CostFunctionFactory::generateCostFunction(
  const CameraConstPtr & _camera, const Eigen::Vector3d & _odo_pos, const Eigen::Vector3d & _odo_att,
  const Eigen::Vector2d & _observed_p, int _flags, bool _optimize_cam_odo_z) const
{
  ceres::CostFunction * costFunction = nullptr;

  std::vector<double> intrinsic_params;
  _camera->writeParameters(intrinsic_params);

  switch (_flags) {
  case CAMERA_ODOMETRY_TRANSFORM | POINT_3D:
    switch (_camera->modelType()) {
    case Camera::KANNALA_BRANDT:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<EquidistantCamera>, 2, 4, 3, 3>(
          new ReprojectionError3<EquidistantCamera>(
            intrinsic_params, _odo_pos, _odo_att, _observed_p, _optimize_cam_odo_z));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<EquidistantCamera>, 2, 4, 2, 3>(
          new ReprojectionError3<EquidistantCamera>(
            intrinsic_params, _odo_pos, _odo_att, _observed_p, _optimize_cam_odo_z));
      }
      break;
    case Camera::PINHOLE:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 4, 3, 3>(
          new ReprojectionError3<PinholeCamera>(
            intrinsic_params, _odo_pos, _odo_att, _observed_p, _optimize_cam_odo_z));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 4, 2, 3>(
          new ReprojectionError3<PinholeCamera>(
            intrinsic_params, _odo_pos, _odo_att, _observed_p, _optimize_cam_odo_z));
      }
      break;
    case Camera::PINHOLE_FULL:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeFullCamera>, 2, 4, 3, 3>(
          new ReprojectionError3<PinholeFullCamera>(
            intrinsic_params, _odo_pos, _odo_att, _observed_p, _optimize_cam_odo_z));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeFullCamera>, 2, 4, 2, 3>(
          new ReprojectionError3<PinholeFullCamera>(
            intrinsic_params, _odo_pos, _odo_att, _observed_p, _optimize_cam_odo_z));
      }
      break;
    case Camera::MEI:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 4, 3, 3>(
          new ReprojectionError3<CataCamera>(
            intrinsic_params, _odo_pos, _odo_att, _observed_p, _optimize_cam_odo_z));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 4, 2, 3>(
          new ReprojectionError3<CataCamera>(
            intrinsic_params, _odo_pos, _odo_att, _observed_p, _optimize_cam_odo_z));
      }
      break;
    case Camera::SCARAMUZZA:
      if (_optimize_cam_odo_z) {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<OCAMCamera>, 2, 4, 3, 3>(
          new ReprojectionError3<OCAMCamera>(
            intrinsic_params, _odo_pos, _odo_att, _observed_p, _optimize_cam_odo_z));
      } else {
        costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<OCAMCamera>, 2, 4, 2, 3>(
          new ReprojectionError3<OCAMCamera>(
            intrinsic_params, _odo_pos, _odo_att, _observed_p, _optimize_cam_odo_z));
      }
      break;
    }
    break;
  }

  return costFunction;
}

ceres::CostFunction * CostFunctionFactory::generateCostFunction(
  const CameraConstPtr & _camera, const Eigen::Quaterniond & _cam_odo_q, const Eigen::Vector3d & _cam_odo_t,
  const Eigen::Vector3d & _odo_pos, const Eigen::Vector3d & _odo_att, const Eigen::Vector2d & _observed_p,
  int _flags) const
{
  ceres::CostFunction * costFunction = nullptr;

  std::vector<double> intrinsic_params;
  _camera->writeParameters(intrinsic_params);

  switch (_flags) {
  case POINT_3D:
    switch (_camera->modelType()) {
    case Camera::KANNALA_BRANDT:
      costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<EquidistantCamera>, 2, 3>(
        new ReprojectionError3<EquidistantCamera>(
          intrinsic_params, _cam_odo_q, _cam_odo_t, _odo_pos, _odo_att, _observed_p));
      break;
    case Camera::PINHOLE:
      costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeCamera>, 2, 3>(
        new ReprojectionError3<PinholeCamera>(
          intrinsic_params, _cam_odo_q, _cam_odo_t, _odo_pos, _odo_att, _observed_p));
      break;
    case Camera::PINHOLE_FULL:
      costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<PinholeFullCamera>, 2, 3>(
        new ReprojectionError3<PinholeFullCamera>(
          intrinsic_params, _cam_odo_q, _cam_odo_t, _odo_pos, _odo_att, _observed_p));
      break;
    case Camera::MEI:
      costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<CataCamera>, 2, 3>(
        new ReprojectionError3<CataCamera>(
          intrinsic_params, _cam_odo_q, _cam_odo_t, _odo_pos, _odo_att, _observed_p));
      break;
    case Camera::SCARAMUZZA:
      costFunction = new ceres::AutoDiffCostFunction<ReprojectionError3<OCAMCamera>, 2, 3>(
        new ReprojectionError3<OCAMCamera>(
          intrinsic_params, _cam_odo_q, _cam_odo_t, _odo_pos, _odo_att, _observed_p));
      break;
    }
    break;
  }

  return costFunction;
}

ceres::CostFunction * CostFunctionFactory::generateCostFunction(
  const CameraConstPtr & _cameraL, const CameraConstPtr & _cameraR, const Eigen::Vector3d & _observed_P,
  const Eigen::Vector2d & _observed_p_l, const Eigen::Vector2d & _observed_p_r) const
{
  ceres::CostFunction * costFunction = nullptr;

  if (_cameraL->modelType() != _cameraR->modelType()) {
    return costFunction;
  }

  switch (_cameraL->modelType()) {
  case Camera::KANNALA_BRANDT:
    costFunction =
      new ceres::AutoDiffCostFunction<StereoReprojectionError<EquidistantCamera>, 4, 8, 8, 4, 3, 4, 3>(
        new StereoReprojectionError<EquidistantCamera>(_observed_P, _observed_p_l, _observed_p_r));
    break;
  case Camera::PINHOLE:
    costFunction = new ceres::AutoDiffCostFunction<StereoReprojectionError<PinholeCamera>, 4, 8, 8, 4, 3, 4, 3>(
      new StereoReprojectionError<PinholeCamera>(_observed_P, _observed_p_l, _observed_p_r));
    break;
  case Camera::PINHOLE_FULL:
    costFunction =
      new ceres::AutoDiffCostFunction<StereoReprojectionError<PinholeFullCamera>, 4, 12, 12, 4, 3, 4, 3>(
        new StereoReprojectionError<PinholeFullCamera>(_observed_P, _observed_p_l, _observed_p_r));
    break;
  case Camera::MEI:
    costFunction = new ceres::AutoDiffCostFunction<StereoReprojectionError<CataCamera>, 4, 9, 9, 4, 3, 4, 3>(
      new StereoReprojectionError<CataCamera>(_observed_P, _observed_p_l, _observed_p_r));
    break;
  case Camera::SCARAMUZZA:
    costFunction = new ceres::AutoDiffCostFunction<
      StereoReprojectionError<OCAMCamera>, 4, SCARAMUZZA_CAMERA_NUM_PARAMS, SCARAMUZZA_CAMERA_NUM_PARAMS, 4,
      3, 4, 3>(new StereoReprojectionError<OCAMCamera>(_observed_P, _observed_p_l, _observed_p_r));
    break;
  }

  return costFunction;
}
} // namespace camodocal
