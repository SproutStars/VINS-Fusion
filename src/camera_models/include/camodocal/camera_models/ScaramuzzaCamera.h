#ifndef SCARAMUZZACAMERA_H
#define SCARAMUZZACAMERA_H

#include <opencv2/core/core.hpp>
#include <string>

#include "Camera.h"
#include "ceres/rotation.h"

namespace camodocal
{

#define SCARAMUZZA_POLY_SIZE 5
#define SCARAMUZZA_INV_POLY_SIZE 20

#define SCARAMUZZA_CAMERA_NUM_PARAMS                                                                         \
  (SCARAMUZZA_POLY_SIZE + SCARAMUZZA_INV_POLY_SIZE + 2 /*center*/ + 3 /*affine*/)

/**
 * Scaramuzza Camera (Omnidirectional)
 * https://sites.google.com/site/scarabotix/ocamcalib-toolbox
 */

class OCAMCamera : public Camera {
public:
  class Parameters : public Camera::Parameters {
  public:
    Parameters();

    double & c() { return m_C_; }
    double & d() { return m_D_; }
    double & e() { return m_E_; }

    double & centerX() { return m_center_x_; }
    double & centerY() { return m_center_y_; }

    double & poly(int _idx) { return m_poly_[_idx]; }
    double & invPoly(int _idx) { return m_inv_poly_[_idx]; }

    double c() const { return m_C_; }
    double d() const { return m_D_; }
    double e() const { return m_E_; }

    double centerX() const { return m_center_x_; }
    double centerY() const { return m_center_y_; }

    double poly(int _idx) const { return m_poly_[_idx]; }
    double invPoly(int _idx) const { return m_inv_poly_[_idx]; }

    bool readFromYamlFile(const std::string & _filename) override;
    void writeToYamlFile(const std::string & _filename) const override;

    Parameters & operator=(const Parameters & _other);
    friend std::ostream & operator<<(std::ostream & _out, const Parameters & _params);

  private:
    double m_poly_[SCARAMUZZA_POLY_SIZE];
    double m_inv_poly_[SCARAMUZZA_INV_POLY_SIZE];
    double m_C_;
    double m_D_;
    double m_E_;
    double m_center_x_;
    double m_center_y_;
  };

  OCAMCamera();

  /**
    * \brief Constructor from the projection model parameters
    */
  OCAMCamera(const Parameters & _params);

  Camera::ModelType_e modelType() const override;
  const std::string & cameraName() const override;
  int imageWidth() const override;
  int imageHeight() const override;

  void estimateIntrinsics(
    const cv::Size & _boardSize, const std::vector<std::vector<cv::Point3f>> & _objectPoints,
    const std::vector<std::vector<cv::Point2f>> & _imagePoints) override;

  // Lift points from the image plane to the sphere
  void liftSphere(const Eigen::Vector2d & _p, Eigen::Vector3d & _P) const override;
  //%output P

  // Lift points from the image plane to the projective space
  void liftProjective(const Eigen::Vector2d & _p, Eigen::Vector3d & _P) const override;
  //%output P

  // Projects 3D points to the image plane (Pi function)
  void spaceToPlane(const Eigen::Vector3d & _P, Eigen::Vector2d & _p) const override;
  //%output p

  // Projects 3D points to the image plane (Pi function)
  // and calculates jacobian
  //void spaceToPlane(const Eigen::Vector3d& P, Eigen::Vector2d& p,
  //                  Eigen::Matrix<double,2,3>& J) const;
  //%output p
  //%output J

  void undistToPlane(const Eigen::Vector2d & _p_u, Eigen::Vector2d & _p) const override;
  //%output p

  template <typename T>
  static void spaceToPlane(
    const T * const _params, const T * const _q, const T * const _t, const Eigen::Matrix<T, 3, 1> & _P,
    Eigen::Matrix<T, 2, 1> & _p);
  template <typename T>
  static void spaceToSphere(
    const T * const _params, const T * const _q, const T * const _t, const Eigen::Matrix<T, 3, 1> & _P,
    Eigen::Matrix<T, 3, 1> & _P_s);
  template <typename T>
  static void
  liftToSphere(const T * const _params, const Eigen::Matrix<T, 2, 1> & _p, Eigen::Matrix<T, 3, 1> & _P);

  template <typename T>
  static void
  sphereToPlane(const T * const _params, const Eigen::Matrix<T, 3, 1> & _P, Eigen::Matrix<T, 2, 1> & _p);


  void initUndistortMap(cv::Mat & _map1, cv::Mat & _map2, double _fScale = 1.0) const;
  cv::Mat initUndistortRectifyMap(
    cv::Mat & _map1, cv::Mat & _map2, float _fx = -1.0f, float _fy = -1.0f,
    cv::Size _imageSize = cv::Size(0, 0), float _cx = -1.0f, float _cy = -1.0f,
    cv::Mat _rmat = cv::Mat::eye(3, 3, CV_32F)) const override;

  int parameterCount() const override;

  const Parameters & getParameters() const;
  void setParameters(const Parameters & _parameters);

  void readParameters(const std::vector<double> & _parameterVec) override;
  void writeParameters(std::vector<double> & _parameterVec) const override;

  void writeParametersToYamlFile(const std::string & _filename) const override;

  std::string parametersToString() const override;

private:
  Parameters mParameters_;

  double m_inv_scale_;
};

typedef boost::shared_ptr<OCAMCamera> OCAMCameraPtr;
typedef boost::shared_ptr<const OCAMCamera> OCAMCameraConstPtr;

template <typename T>
void OCAMCamera::spaceToPlane(
  const T * const _params, const T * const _q, const T * const _t, const Eigen::Matrix<T, 3, 1> & _P,
  Eigen::Matrix<T, 2, 1> & _p)
{
  T P_c[3];
  {
    T P_w[3];
    P_w[0] = T(_P(0));
    P_w[1] = T(_P(1));
    P_w[2] = T(_P(2));

    // Convert quaternion from Eigen convention (x, y, z, w)
    // to Ceres convention (w, x, y, z)
    T q_ceres[4] = {_q[3], _q[0], _q[1], _q[2]};

    ceres::QuaternionRotatePoint(q_ceres, P_w, P_c);

    P_c[0] += _t[0];
    P_c[1] += _t[1];
    P_c[2] += _t[2];
  }

  T c = _params[0];
  T d = _params[1];
  T e = _params[2];
  T xc[2] = {_params[3], _params[4]};

  //T poly[SCARAMUZZA_POLY_SIZE];
  //for (int i=0; i < SCARAMUZZA_POLY_SIZE; i++)
  //    poly[i] = params[5+i];

  T inv_poly[SCARAMUZZA_INV_POLY_SIZE];
  for (int i = 0; i < SCARAMUZZA_INV_POLY_SIZE; i++)
    inv_poly[i] = _params[5 + SCARAMUZZA_POLY_SIZE + i];

  T norm_sqr = P_c[0] * P_c[0] + P_c[1] * P_c[1];
  T norm = T(0.0);
  if (norm_sqr > T(0.0))
    norm = sqrt(norm_sqr);

  T theta = atan2(-P_c[2], norm);
  T rho = T(0.0);
  T theta_i = T(1.0);

  for (auto & i : inv_poly) {
    rho += theta_i * i;
    theta_i *= theta;
  }

  T invNorm = T(1.0) / norm;
  T xn[2] = {P_c[0] * invNorm * rho, P_c[1] * invNorm * rho};

  _p(0) = xn[0] * c + xn[1] * d + xc[0];
  _p(1) = xn[0] * e + xn[1] + xc[1];
}

template <typename T>
void OCAMCamera::spaceToSphere(
  [[maybe_unused]] const T * const _params, const T * const _q, const T * const _t,
  const Eigen::Matrix<T, 3, 1> & _P, Eigen::Matrix<T, 3, 1> & _P_s)
{
  T P_c[3];
  {
    T P_w[3];
    P_w[0] = T(_P(0));
    P_w[1] = T(_P(1));
    P_w[2] = T(_P(2));

    // Convert quaternion from Eigen convention (x, y, z, w)
    // to Ceres convention (w, x, y, z)
    T q_ceres[4] = {_q[3], _q[0], _q[1], _q[2]};

    ceres::QuaternionRotatePoint(q_ceres, P_w, P_c);

    P_c[0] += _t[0];
    P_c[1] += _t[1];
    P_c[2] += _t[2];
  }

  //T poly[SCARAMUZZA_POLY_SIZE];
  //for (int i=0; i < SCARAMUZZA_POLY_SIZE; i++)
  //    poly[i] = params[5+i];

  T norm_sqr = P_c[0] * P_c[0] + P_c[1] * P_c[1] + P_c[2] * P_c[2];
  T norm = T(0.0);
  if (norm_sqr > T(0.0))
    norm = sqrt(norm_sqr);

  _P_s(0) = P_c[0] / norm;
  _P_s(1) = P_c[1] / norm;
  _P_s(2) = P_c[2] / norm;
}

template <typename T>
void OCAMCamera::liftToSphere(
  const T * const _params, const Eigen::Matrix<T, 2, 1> & _p, Eigen::Matrix<T, 3, 1> & _P)
{
  T c = _params[0];
  T d = _params[1];
  T e = _params[2];
  T cc[2] = {_params[3], _params[4]};
  T poly[SCARAMUZZA_POLY_SIZE];
  for (int i = 0; i < SCARAMUZZA_POLY_SIZE; i++)
    poly[i] = _params[5 + i];

  // Relative to Center
  T p_2d[2];
  p_2d[0] = T(_p(0));
  p_2d[1] = T(_p(1));

  T xc[2] = {p_2d[0] - cc[0], p_2d[1] - cc[1]};

  T inv_scale = T(1.0) / (c - d * e);

  // Affine Transformation
  T xc_a[2];

  xc_a[0] = inv_scale * (xc[0] - d * xc[1]);
  xc_a[1] = inv_scale * (-e * xc[0] + c * xc[1]);

  T norm_sqr = xc_a[0] * xc_a[0] + xc_a[1] * xc_a[1];
  T phi = sqrt(norm_sqr);
  T phi_i = T(1.0);
  T z = T(0.0);

  for (int i = 0; i < SCARAMUZZA_POLY_SIZE; i++) {
    if (i != 1) {
      z += phi_i * poly[i];
    }
    phi_i *= phi;
  }

  T p_3d[3];
  p_3d[0] = xc[0];
  p_3d[1] = xc[1];
  p_3d[2] = -z;

  T p_3d_norm_sqr = p_3d[0] * p_3d[0] + p_3d[1] * p_3d[1] + p_3d[2] * p_3d[2];
  T p_3d_norm = sqrt(p_3d_norm_sqr);

  _P << p_3d[0] / p_3d_norm, p_3d[1] / p_3d_norm, p_3d[2] / p_3d_norm;
}

template <typename T>
void OCAMCamera::sphereToPlane(
  const T * const _params, const Eigen::Matrix<T, 3, 1> & _P, Eigen::Matrix<T, 2, 1> & _p)
{
  T P_c[3];
  {
    P_c[0] = T(_P(0));
    P_c[1] = T(_P(1));
    P_c[2] = T(_P(2));
  }

  T c = _params[0];
  T d = _params[1];
  T e = _params[2];
  T xc[2] = {_params[3], _params[4]};

  T inv_poly[SCARAMUZZA_INV_POLY_SIZE];
  for (int i = 0; i < SCARAMUZZA_INV_POLY_SIZE; i++)
    inv_poly[i] = _params[5 + SCARAMUZZA_POLY_SIZE + i];

  T norm_sqr = P_c[0] * P_c[0] + P_c[1] * P_c[1];
  T norm = T(0.0);
  if (norm_sqr > T(0.0))
    norm = sqrt(norm_sqr);

  T theta = atan2(-P_c[2], norm);
  T rho = T(0.0);
  T theta_i = T(1.0);

  for (auto & i : inv_poly) {
    rho += theta_i * i;
    theta_i *= theta;
  }

  T invNorm = T(1.0) / norm;
  T xn[2] = {P_c[0] * invNorm * rho, P_c[1] * invNorm * rho};

  _p(0) = xn[0] * c + xn[1] * d + xc[0];
  _p(1) = xn[0] * e + xn[1] + xc[1];
}
} // namespace camodocal

#endif
