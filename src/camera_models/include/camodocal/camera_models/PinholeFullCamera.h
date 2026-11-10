#ifndef PinholeFullCAMERA_H
#define PinholeFullCAMERA_H

#include <opencv2/core/core.hpp>
#include <string>

#include "Camera.h"
#include "ceres/rotation.h"

namespace camodocal
{

class PinholeFullCamera : public Camera {
public:
  class Parameters : public Camera::Parameters {
  public:
    Parameters();
    Parameters(
      const std::string & _cameraName, int _w, int _h, double _k1, double _k2, double _k3, double _k4,
      double _k5, double _k6, double _p1, double _p2, double _fx, double _fy, double _cx, double _cy);

    double & k1();
    double & k2();
    double & k3();
    double & k4();
    double & k5();
    double & k6();
    double & p1();
    double & p2();
    double & fx();
    double & fy();
    double & cx();
    double & cy();

    double xi() const;
    double k1() const;
    double k2() const;
    double k3() const;
    double k4() const;
    double k5() const;
    double k6() const;
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
    double m_k3_;
    double m_k4_;
    double m_k5_;
    double m_k6_;
  };

  PinholeFullCamera();

  /**
     * \brief Constructor from the projection model parameters
     */
  PinholeFullCamera(
    const std::string & _cameraName, int _imageWidth, int _imageHeight, double _k1, double _k2, double _k3,
    double _k4, double _k5, double _k6, double _p1, double _p2, double _fx, double _fy, double _cx, double _cy);
  /**
     * \brief Constructor from the projection model parameters
     */
  PinholeFullCamera(const Parameters & _params);

  Camera::ModelType_e modelType() const override;
  const std::string & cameraName() const override;
  int imageWidth() const override;
  int imageHeight() const override;
  cv::Size imageSize() const { return cv::Size(imageWidth(), imageHeight()); }
  cv::Point2f getPrinciple() const { return cv::Point2f(mParameters_.cx(), mParameters_.cy()); }

  void estimateIntrinsics(
    const cv::Size & _boardSize, const std::vector<std::vector<cv::Point3f>> & _objectPoints,
    const std::vector<std::vector<cv::Point2f>> & _imagePoints) override;

  void setInitIntrinsics(
    [[maybe_unused]] const std::vector<std::vector<cv::Point3f>> & _objectPoints,
    [[maybe_unused]] const std::vector<std::vector<cv::Point2f>> & _imagePoints)
  {
    Parameters params = getParameters();

    params.k1() = 0.0;
    params.k2() = 0.0;
    params.k3() = 0.0;
    params.k4() = 0.0;
    params.k5() = 0.0;
    params.k6() = 0.0;
    params.p1() = 0.0;
    params.p2() = 0.0;

    double cx = params.imageWidth() / 2.0;
    double cy = params.imageHeight() / 2.0;
    params.cx() = cx;
    params.cy() = cy;
    params.fx() = 1200;
    params.fy() = 1200;

    setParameters(params);
  }

  // Lift points from the image plane to the sphere
  void liftSphere(const Eigen::Vector2d & _p, Eigen::Vector3d & _P) const override;
  //%output P

  // Lift points from the image plane to the projective space
  void liftProjective(const Eigen::Vector2d & _p, Eigen::Vector3d & _P) const override;
  //%output P

  void liftProjective(const Eigen::Vector2d & _p, Eigen::Vector3d & _P, float _image_scale) const;

  // Projects 3D points to the image plane (Pi function)
  void spaceToPlane(const Eigen::Vector3d & _P, Eigen::Vector2d & _p) const override;
  //%output p

  void spaceToPlane(const Eigen::Vector3d & _P, Eigen::Vector2d & _p, float _image_scalse) const;

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

typedef boost::shared_ptr<PinholeFullCamera> PinholeFullCameraPtr;
typedef boost::shared_ptr<const PinholeFullCamera> PinholeFullCameraConstPtr;

template <typename T>
void PinholeFullCamera::spaceToPlane(
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
  T k3 = _params[2];
  T k4 = _params[3];
  T k5 = _params[4];
  T k6 = _params[5];
  T p1 = _params[6];
  T p2 = _params[7];
  T fx = _params[8];
  T fy = _params[9];
  T alpha = T(0); // cameraParams.alpha();
  T cx = _params[10];
  T cy = _params[11];

  // Transform to model plane
  T x = P_c[0] / P_c[2];
  T y = P_c[1] / P_c[2];
  // T z = P_c[2] / P_c[2];

  T r2 = x * x + y * y;
  T r4 = r2 * r2;
  T r6 = r4 * r2;
  T a1 = T(2) * x * y;
  T a2 = r2 + T(2) * x * x;
  T a3 = r2 + T(2) * y * y;
  T cdist = T(1) + k1 * r2 + k2 * r4 + k3 * r6;
  T icdist2 = T(1.) / (T(1) + k4 * r2 + k5 * r4 + k6 * r6);
  T xd0 = x * cdist * icdist2 + p1 * a1 + p2 * a2; // + k[8] * r2 + k[9] * r4;
  T yd0 = y * cdist * icdist2 + p1 * a3 + p2 * a1; // + k[10] * r2 + k[11] * r4;

  _p(0) = xd0 * fx + cx;
  _p(1) = yd0 * fy + cy;
}
} // namespace camodocal

#endif
