#ifndef PINHOLECAMERA_H
#define PINHOLECAMERA_H

#include <opencv2/core/core.hpp>
#include <string>

#include "Camera.h"
#include "ceres/rotation.h"

namespace camodocal
{

class PinholeCamera : public Camera {
public:
  class Parameters : public Camera::Parameters {
  public:
    Parameters();
    Parameters(
      const std::string & _cameraName, int _w, int _h, double _k1, double _k2, double _p1, double _p2,
      double _fx, double _fy, double _cx, double _cy);

    double & k1();
    double & k2();
    double & p1();
    double & p2();
    double & fx();
    double & fy();
    double & cx();
    double & cy();

    double xi() const;
    double k1() const;
    double k2() const;
    double p1() const;
    double p2() const;
    double fx() const;
    double fy() const;
    double cx() const;
    double cy() const;

    bool readFromYamlFile(const std::string & _filename) override;
    void writeToYamlFile(const std::string & _filename) const override;

    Parameters & operator=(const Parameters & _other);
    friend std::ostream & operator<<(std::ostream & _out, const Parameters & _params);

  private:
    double m_k1_;
    double m_k2_;
    double m_p1_;
    double m_p2_;
    double m_fx_;
    double m_fy_;
    double m_cx_;
    double m_cy_;
  };

  PinholeCamera();

  /**
    * \brief Constructor from the projection model parameters
    */
  PinholeCamera(
    const std::string & _cameraName, int _imageWidth, int _imageHeight, double _k1, double _k2, double _p1,
    double _p2, double _fx, double _fy, double _cx, double _cy);
  /**
    * \brief Constructor from the projection model parameters
    */
  PinholeCamera(const Parameters & _params);

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

  void distortion(const Eigen::Vector2d & _p_u, Eigen::Vector2d & _d_u) const;
  void distortion(const Eigen::Vector2d & _p_u, Eigen::Vector2d & _d_u, Eigen::Matrix2d & _J) const;

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

  double m_inv_K11_, m_inv_K13_, m_inv_K22_, m_inv_K23_;
  bool m_noDistortion_;
};

typedef boost::shared_ptr<PinholeCamera> PinholeCameraPtr;
typedef boost::shared_ptr<const PinholeCamera> PinholeCameraConstPtr;

template <typename T>
void PinholeCamera::spaceToPlane(
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

  // project 3D object point to the image plane
  T k1 = _params[0];
  T k2 = _params[1];
  T p1 = _params[2];
  T p2 = _params[3];
  T fx = _params[4];
  T fy = _params[5];
  T alpha = T(0); //cameraParams.alpha();
  T cx = _params[6];
  T cy = _params[7];

  // Transform to model plane
  T u = P_c[0] / P_c[2];
  T v = P_c[1] / P_c[2];

  T rho_sqr = u * u + v * v;
  T L = T(1.0) + k1 * rho_sqr + k2 * rho_sqr * rho_sqr;
  T du = T(2.0) * p1 * u * v + p2 * (rho_sqr + T(2.0) * u * u);
  T dv = p1 * (rho_sqr + T(2.0) * v * v) + T(2.0) * p2 * u * v;

  u = L * u + du;
  v = L * v + dv;
  _p(0) = fx * (u + alpha * v) + cx;
  _p(1) = fy * v + cy;
}

} // namespace camodocal

#endif
