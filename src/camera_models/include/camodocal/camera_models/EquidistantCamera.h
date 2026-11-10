#ifndef EQUIDISTANTCAMERA_H
#define EQUIDISTANTCAMERA_H

#include <opencv2/core/core.hpp>
#include <string>

#include "Camera.h"
#include "ceres/rotation.h"

namespace camodocal
{

/**
 * J. Kannala, and S. Brandt, A Generic Camera Model and Calibration Method
 * for Conventional, Wide-Angle, and Fish-Eye Lenses, PAMI 2006
 */

class EquidistantCamera : public Camera {
public:
  class Parameters : public Camera::Parameters {
  public:
    Parameters();
    Parameters(
      const std::string & _cameraName, int _w, int _h, double _k2, double _k3, double _k4, double _k5,
      double _mu, double _mv, double _u0, double _v0);

    double & k2();
    double & k3();
    double & k4();
    double & k5();
    double & mu();
    double & mv();
    double & u0();
    double & v0();

    double k2() const;
    double k3() const;
    double k4() const;
    double k5() const;
    double mu() const;
    double mv() const;
    double u0() const;
    double v0() const;

    bool readFromYamlFile(const std::string & _filename) override;
    void writeToYamlFile(const std::string & _filename) const override;

    Parameters & operator=(const Parameters & _other);
    friend std::ostream & operator<<(std::ostream & _out, const Parameters & _params);

  private:
    // projection
    double m_k2_;
    double m_k3_;
    double m_k4_;
    double m_k5_;

    double m_mu_;
    double m_mv_;
    double m_u0_;
    double m_v0_;
  };

  EquidistantCamera();

  /**
    * \brief Constructor from the projection model parameters
    */
  EquidistantCamera(
    const std::string & _cameraName, int _imageWidth, int _imageHeight, double _k2, double _k3, double _k4,
    double _k5, double _mu, double _mv, double _u0, double _v0);
  /**
    * \brief Constructor from the projection model parameters
    */
  EquidistantCamera(const Parameters & _params);

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
  void spaceToPlane(const Eigen::Vector3d & _P, Eigen::Vector2d & _p, Eigen::Matrix<double, 2, 3> & _J) const;
  //%output p
  //%output J

  void undistToPlane(const Eigen::Vector2d & _p_u, Eigen::Vector2d & _p) const override;
  //%output p

  template <typename T>
  static void spaceToPlane(
    const T * const _params, const T * const _q, const T * const _t, const Eigen::Matrix<T, 3, 1> & _P,
    Eigen::Matrix<T, 2, 1> & _p);

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
  template <typename T> static T r(T _k2, T _k3, T _k4, T _k5, T _theta);


  void fitOddPoly(
    const std::vector<double> & _x, const std::vector<double> & _y, int _n, std::vector<double> & _coeffs) const;

  void backprojectSymmetric(const Eigen::Vector2d & _p_u, double & _theta, double & _phi) const;

  Parameters mParameters_;

  double m_inv_K11_, m_inv_K13_, m_inv_K22_, m_inv_K23_;
};

typedef boost::shared_ptr<EquidistantCamera> EquidistantCameraPtr;
typedef boost::shared_ptr<const EquidistantCamera> EquidistantCameraConstPtr;

template <typename T> T EquidistantCamera::r(T _k2, T _k3, T _k4, T _k5, T _theta)
{
  // k1 = 1
  return _theta + _k2 * _theta * _theta * _theta + _k3 * _theta * _theta * _theta * _theta * _theta +
         _k4 * _theta * _theta * _theta * _theta * _theta * _theta * _theta +
         _k5 * _theta * _theta * _theta * _theta * _theta * _theta * _theta * _theta * _theta;
}

template <typename T>
void EquidistantCamera::spaceToPlane(
  const T * const _params, const T * const _q, const T * const _t, const Eigen::Matrix<T, 3, 1> & _P,
  Eigen::Matrix<T, 2, 1> & _p)
{
  T P_w[3];
  P_w[0] = T(_P(0));
  P_w[1] = T(_P(1));
  P_w[2] = T(_P(2));

  // Convert quaternion from Eigen convention (x, y, z, w)
  // to Ceres convention (w, x, y, z)
  T q_ceres[4] = {_q[3], _q[0], _q[1], _q[2]};

  T P_c[3];
  ceres::QuaternionRotatePoint(q_ceres, P_w, P_c);

  P_c[0] += _t[0];
  P_c[1] += _t[1];
  P_c[2] += _t[2];

  // project 3D object point to the image plane;
  T k2 = _params[0];
  T k3 = _params[1];
  T k4 = _params[2];
  T k5 = _params[3];
  T mu = _params[4];
  T mv = _params[5];
  T u0 = _params[6];
  T v0 = _params[7];

  T len = sqrt(P_c[0] * P_c[0] + P_c[1] * P_c[1] + P_c[2] * P_c[2]);
  T theta = acos(P_c[2] / len);
  T phi = atan2(P_c[1], P_c[0]);

  Eigen::Matrix<T, 2, 1> p_u = r(k2, k3, k4, k5, theta) * Eigen::Matrix<T, 2, 1>(cos(phi), sin(phi));

  _p(0) = mu * p_u(0) + u0;
  _p(1) = mv * p_u(1) + v0;
}

} // namespace camodocal

#endif
