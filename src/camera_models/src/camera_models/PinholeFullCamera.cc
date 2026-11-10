#include "camodocal/camera_models/PinholeFullCamera.h"

#include <cmath>
#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camodocal/gpl/gpl.h"

namespace camodocal
{

PinholeFullCamera::Parameters::Parameters()
  : Camera::Parameters(PINHOLE_FULL), m_k1_(0.0), m_k2_(0.0), m_k3_(0.0), m_k4_(0.0), m_k5_(0.0), m_k6_(0.0),
    m_p1_(0.0), m_p2_(0.0), m_fx_(0.0), m_fy_(0.0), m_cx_(0.0), m_cy_(0.0)
{
}

PinholeFullCamera::Parameters::Parameters(
  const std::string & _cameraName, //
  int _w, int _h, double _k1, double _k2, double _k3, double _k4, double _k5, double _k6, double _p1,
  double _p2, double _fx, double _fy, double _cx, double _cy)
  : Camera::Parameters(PINHOLE_FULL, _cameraName, _w, _h), m_k1_(_k1), m_k2_(_k2), m_k3_(_k3), m_k4_(_k4),
    m_k5_(_k5), m_k6_(_k6), m_p1_(_p1), m_p2_(_p2), m_fx_(_fx), m_fy_(_fy), m_cx_(_cx), m_cy_(_cy)
{
}

double & PinholeFullCamera::Parameters::k1() { return m_k1_; }

double & PinholeFullCamera::Parameters::k2() { return m_k2_; }

double & PinholeFullCamera::Parameters::k3() { return m_k3_; }

double & PinholeFullCamera::Parameters::k4() { return m_k4_; }

double & PinholeFullCamera::Parameters::k5() { return m_k5_; }

double & PinholeFullCamera::Parameters::k6() { return m_k6_; }

double & PinholeFullCamera::Parameters::p1() { return m_p1_; }

double & PinholeFullCamera::Parameters::p2() { return m_p2_; }

double & PinholeFullCamera::Parameters::fx() { return m_fx_; }

double & PinholeFullCamera::Parameters::fy() { return m_fy_; }

double & PinholeFullCamera::Parameters::cx() { return m_cx_; }

double & PinholeFullCamera::Parameters::cy() { return m_cy_; }

double PinholeFullCamera::Parameters::k1() const { return m_k1_; }

double PinholeFullCamera::Parameters::k2() const { return m_k2_; }

double PinholeFullCamera::Parameters::p1() const { return m_p1_; }

double PinholeFullCamera::Parameters::p2() const { return m_p2_; }

double PinholeFullCamera::Parameters::fx() const { return m_fx_; }

double PinholeFullCamera::Parameters::fy() const { return m_fy_; }

double PinholeFullCamera::Parameters::cx() const { return m_cx_; }

double PinholeFullCamera::Parameters::cy() const { return m_cy_; }

double PinholeFullCamera::Parameters::k3() const { return m_k3_; }

double PinholeFullCamera::Parameters::k4() const { return m_k4_; }

double PinholeFullCamera::Parameters::k5() const { return m_k5_; }

double PinholeFullCamera::Parameters::k6() const { return m_k6_; }

bool PinholeFullCamera::Parameters::readFromYamlFile(const std::string & _filename)
{
  cv::FileStorage fs(_filename, cv::FileStorage::READ);

  if (!fs.isOpened()) {
    return false;
  }

  if (!fs["model_type"].isNone()) {
    std::string sModelType;
    fs["model_type"] >> sModelType;

    if (sModelType.compare("PINHOLE_FULL") != 0) {
      return false;
    }
  }

  m_modelType_ = PINHOLE_FULL;
  fs["camera_name"] >> m_cameraName_;
  m_imageWidth_ = static_cast<int>(fs["image_width"]);
  m_imageHeight_ = static_cast<int>(fs["image_height"]);

  cv::FileNode n = fs["distortion_parameters"];
  m_k1_ = static_cast<double>(n["k1"]);
  m_k2_ = static_cast<double>(n["k2"]);
  m_k3_ = static_cast<double>(n["k3"]);
  m_k4_ = static_cast<double>(n["k4"]);
  m_k5_ = static_cast<double>(n["k5"]);
  m_k6_ = static_cast<double>(n["k6"]);
  m_p1_ = static_cast<double>(n["p1"]);
  m_p2_ = static_cast<double>(n["p2"]);

  n = fs["projection_parameters"];
  m_fx_ = static_cast<double>(n["fx"]);
  m_fy_ = static_cast<double>(n["fy"]);
  m_cx_ = static_cast<double>(n["cx"]);
  m_cy_ = static_cast<double>(n["cy"]);

  return true;
}

void PinholeFullCamera::Parameters::writeToYamlFile(const std::string & _filename) const
{
  cv::FileStorage fs(_filename, cv::FileStorage::WRITE);

  fs << "model_type" << "PINHOLE_FULL";
  fs << "camera_name" << m_cameraName_;
  fs << "image_width" << m_imageWidth_;
  fs << "image_height" << m_imageHeight_;

  // radial distortion: k1, k2
  // tangential distortion: p1, p2
  fs << "distortion_parameters";
  fs << "{" << "k1" << m_k1_ << "k2" << m_k2_ << "k3" << m_k3_ << "k4" << m_k4_ << "k5" << m_k5_ << "k6"
     << m_k6_ << "p1" << m_p1_ << "p2" << m_p2_ << "}";

  // projection: fx, fy, cx, cy
  fs << "projection_parameters";
  fs << "{" << "fx" << m_fx_ << "fy" << m_fy_ << "cx" << m_cx_ << "cy" << m_cy_ << "}";

  fs.release();
}

PinholeFullCamera::Parameters &
PinholeFullCamera::Parameters::operator=(const PinholeFullCamera::Parameters & _other)
{
  if (this != &_other) {
    m_modelType_ = _other.m_modelType_;
    m_cameraName_ = _other.m_cameraName_;
    m_imageWidth_ = _other.m_imageWidth_;
    m_imageHeight_ = _other.m_imageHeight_;
    m_k1_ = _other.m_k1_;
    m_k2_ = _other.m_k2_;
    m_k3_ = _other.m_k3_;
    m_k4_ = _other.m_k4_;
    m_k5_ = _other.m_k5_;
    m_k6_ = _other.m_k6_;
    m_p1_ = _other.m_p1_;
    m_p2_ = _other.m_p2_;
    m_fx_ = _other.m_fx_;
    m_fy_ = _other.m_fy_;
    m_cx_ = _other.m_cx_;
    m_cy_ = _other.m_cy_;
  }

  return *this;
}

std::ostream & operator<<(std::ostream & _out, const PinholeFullCamera::Parameters & _params)
{
  _out << "Camera Parameters:" << '\n';
  _out << "    model_type " << "PINHOLE_FULL" << '\n';
  _out << "   camera_name " << _params.m_cameraName_ << '\n';
  _out << "   image_width " << _params.m_imageWidth_ << '\n';
  _out << "  image_height " << _params.m_imageHeight_ << '\n';

  // radial distortion: k1, k2
  // tangential distortion: p1, p2
  _out << "Distortion Parameters" << '\n';
  _out << "            k1 " << _params.m_k1_ << '\n'
       << "            k2 " << _params.m_k2_ << '\n'
       << "            k3 " << _params.m_k3_ << '\n'
       << "            k4 " << _params.m_k4_ << '\n'
       << "            k5 " << _params.m_k5_ << '\n'
       << "            k6 " << _params.m_k6_ << '\n'
       << "            p1 " << _params.m_p1_ << '\n'
       << "            p2 " << _params.m_p2_ << '\n';

  // projection: fx, fy, cx, cy
  _out << "Projection Parameters" << '\n';
  _out << "            fx " << _params.m_fx_ << '\n'
       << "            fy " << _params.m_fy_ << '\n'
       << "            cx " << _params.m_cx_ << '\n'
       << "            cy " << _params.m_cy_ << '\n';

  return _out;
}

PinholeFullCamera::PinholeFullCamera()
  : m_inv_K11_(1.0), m_inv_K13_(0.0), m_inv_K22_(1.0), m_inv_K23_(0.0), m_noDistortion_(true)
{
}

PinholeFullCamera::PinholeFullCamera(
  const std::string & _cameraName, int _imageWidth, int _imageHeight, double _k1, double _k2, double _k3,
  double _k4, double _k5, double _k6, double _p1, double _p2, double _fx, double _fy, double _cx, double _cy)
  : mParameters_(
      _cameraName, _imageWidth, _imageHeight, _k1, _k2, _k3, _k4, _k5, _k6, _p1, _p2, _fx, _fy, _cx, _cy)
{
  m_noDistortion_ = (mParameters_.k1() == 0.0) && (mParameters_.k2() == 0.0) && (mParameters_.p1() == 0.0) &&
                    (mParameters_.p2() == 0.0);

  // Inverse camera projection matrix parameters
  m_inv_K11_ = 1.0 / mParameters_.fx();
  m_inv_K13_ = -mParameters_.cx() / mParameters_.fx();
  m_inv_K22_ = 1.0 / mParameters_.fy();
  m_inv_K23_ = -mParameters_.cy() / mParameters_.fy();
}

PinholeFullCamera::PinholeFullCamera(const PinholeFullCamera::Parameters & _params) : mParameters_(_params)
{
  m_noDistortion_ = (mParameters_.k1() == 0.0) && (mParameters_.k2() == 0.0) && (mParameters_.p1() == 0.0) &&
                    (mParameters_.p2() == 0.0);

  // Inverse camera projection matrix parameters
  m_inv_K11_ = 1.0 / mParameters_.fx();
  m_inv_K13_ = -mParameters_.cx() / mParameters_.fx();
  m_inv_K22_ = 1.0 / mParameters_.fy();
  m_inv_K23_ = -mParameters_.cy() / mParameters_.fy();
}

Camera::ModelType_e PinholeFullCamera::modelType() const { return mParameters_.modelType(); }

const std::string & PinholeFullCamera::cameraName() const { return mParameters_.cameraName(); }

int PinholeFullCamera::imageWidth() const { return mParameters_.imageWidth(); }

int PinholeFullCamera::imageHeight() const { return mParameters_.imageHeight(); }

void PinholeFullCamera::estimateIntrinsics(
  [[maybe_unused]] const cv::Size & _boardSize, const std::vector<std::vector<cv::Point3f>> & _objectPoints,
  const std::vector<std::vector<cv::Point2f>> & _imagePoints)
{
  // Z. Zhang, A Flexible New Technique for Camera Calibration, PAMI 2000

  Parameters params = getParameters();

  params.k1() = 0.0;
  params.k2() = 0.0;
  params.p1() = 0.0;
  params.p2() = 0.0;

  double cx = params.imageWidth() / 2.0;
  double cy = params.imageHeight() / 2.0;
  params.cx() = cx;
  params.cy() = cy;

  size_t nImages = _imagePoints.size();

  cv::Mat A(nImages * 2, 2, CV_64F);
  cv::Mat b(nImages * 2, 1, CV_64F);

  for (size_t i = 0; i < nImages; ++i) {
    const std::vector<cv::Point3f> & oPoints = _objectPoints.at(i);

    std::vector<cv::Point2f> M(oPoints.size());
    for (size_t j = 0; j < M.size(); ++j) {
      M.at(j) = cv::Point2f(oPoints.at(j).x, oPoints.at(j).y);
    }

    cv::Mat H = cv::findHomography(M, _imagePoints.at(i));

    H.at<double>(0, 0) -= H.at<double>(2, 0) * cx;
    H.at<double>(0, 1) -= H.at<double>(2, 1) * cx;
    H.at<double>(0, 2) -= H.at<double>(2, 2) * cx;
    H.at<double>(1, 0) -= H.at<double>(2, 0) * cy;
    H.at<double>(1, 1) -= H.at<double>(2, 1) * cy;
    H.at<double>(1, 2) -= H.at<double>(2, 2) * cy;

    double h[3], v[3], d1[3], d2[3];
    double n[4] = {0, 0, 0, 0};

    for (int j = 0; j < 3; ++j) {
      double t0 = H.at<double>(j, 0);
      double t1 = H.at<double>(j, 1);
      h[j] = t0;
      v[j] = t1;
      d1[j] = (t0 + t1) * 0.5;
      d2[j] = (t0 - t1) * 0.5;
      n[0] += t0 * t0;
      n[1] += t1 * t1;
      n[2] += d1[j] * d1[j];
      n[3] += d2[j] * d2[j];
    }

    for (double & j : n) {
      j = 1.0 / sqrt(j);
    }

    for (int j = 0; j < 3; ++j) {
      h[j] *= n[0];
      v[j] *= n[1];
      d1[j] *= n[2];
      d2[j] *= n[3];
    }

    A.at<double>(i * 2, 0) = h[0] * v[0];
    A.at<double>(i * 2, 1) = h[1] * v[1];
    A.at<double>(i * 2 + 1, 0) = d1[0] * d2[0];
    A.at<double>(i * 2 + 1, 1) = d1[1] * d2[1];
    b.at<double>(i * 2, 0) = -h[2] * v[2];
    b.at<double>(i * 2 + 1, 0) = -d1[2] * d2[2];
  }

  cv::Mat f(2, 1, CV_64F);
  cv::solve(A, b, f, cv::DECOMP_NORMAL | cv::DECOMP_LU);

  params.fx() = sqrt(fabs(1.0 / f.at<double>(0)));
  params.fy() = sqrt(fabs(1.0 / f.at<double>(1)));

  setParameters(params);
}

/**
 * \brief Lifts a point from the image plane to the unit sphere
 *
 * \param p image coordinates
 * \param P coordinates of the point on the sphere
 */
void PinholeFullCamera::liftSphere(const Eigen::Vector2d & _p, Eigen::Vector3d & _P) const
{
  liftProjective(_p, _P);

  _P.normalize();
}

/**
 * \brief Lifts a point from the image plane to its projective ray
 *
 * \param p image coordinates
 * \param P coordinates of the projective ray
 */
void PinholeFullCamera::liftProjective(const Eigen::Vector2d & _p, Eigen::Vector3d & _P) const
{
  double k1 = mParameters_.k1();
  double k2 = mParameters_.k2();
  double k3 = mParameters_.k3();
  double k4 = mParameters_.k4();
  double k5 = mParameters_.k5();
  double k6 = mParameters_.k6();
  double p1 = mParameters_.p1();
  double p2 = mParameters_.p2();

  double fx = mParameters_.fx();
  double fy = mParameters_.fy();
  double ifx = 1. / fx;
  double ify = 1. / fy;
  double cx = mParameters_.cx();
  double cy = mParameters_.cy();

  // Lift points to normalised plane
  double mx_d = ifx * _p(0) + m_inv_K13_;
  double my_d = ify * _p(1) + m_inv_K23_;
  double u = _p(0);
  double v = _p(1);
  double x = mx_d;
  double y = my_d;
  double x0 = x;
  double y0 = y;

  double error = std::numeric_limits<double>::max();

  int max_cnt = 8; // 5
  int min_error = 0.01;
  for (int j = 0;; j++) {
    if (j > max_cnt)
      break;
    if (error < min_error)
      break;

    double r2 = x * x + y * y;
    double icdist = (1 + ((k6 * r2 + k5) * r2 + k4) * r2) / (1 + ((k3 * r2 + k2) * r2 + k1) * r2);
    double deltaX = 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
    double deltaY = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;

    x = (x0 - deltaX) * icdist;
    y = (y0 - deltaY) * icdist;

    if (1) {
      double r4, r6, a1, a2, a3, cdist, icdist2;
      double xd, yd, xd0, yd0;

      r2 = x * x + y * y;
      r4 = r2 * r2;
      r6 = r4 * r2;
      a1 = 2 * x * y;
      a2 = r2 + 2 * x * x;
      a3 = r2 + 2 * y * y;
      cdist = 1 + k1 * r2 + k2 * r4 + k3 * r6;
      icdist2 = 1. / (1 + k4 * r2 + k5 * r4 + k6 * r6);
      xd0 = x * cdist * icdist2 + p1 * a1 + p2 * a2;
      yd0 = y * cdist * icdist2 + p1 * a3 + p2 * a1;

      double x_proj = xd * fx + cx;
      double y_proj = yd * fy + cy;

      error = sqrt(pow(x_proj - u, 2) + pow(y_proj - v, 2));
    }
  }

  _P << x, y, 1.0;
}

void PinholeFullCamera::liftProjective(const Eigen::Vector2d & _p, Eigen::Vector3d & _P, float _image_scale) const
{
  Eigen::Vector2d p_tmp = _p / _image_scale; // p_tmp is without resize, p is with resize
  liftProjective(p_tmp, _P);                 // p_tmp is without resize
}

/**
 * \brief Project a 3D point (\a x,\a y,\a z) to the image plane in (\a u,\a v)
 *
 * \param P 3D point coordinates
 * \param p return value, contains the image point coordinates
 */
void PinholeFullCamera::spaceToPlane(const Eigen::Vector3d & _P, Eigen::Vector2d & _p) const
{
  Eigen::Vector2d p_u, p_d;

  // Project points to the normalised plane
  p_u << _P(0) / _P(2), _P(1) / _P(2);

  if (m_noDistortion_) {
    p_d = p_u;
  } else {
    // Apply distortion
    Eigen::Vector2d d_u;
    distortion(p_u, d_u);
    p_d = p_u + d_u;
  }

  // Apply generalised projection matrix
  _p << mParameters_.fx() * p_d(0) + mParameters_.cx(), mParameters_.fy() * p_d(1) + mParameters_.cy();
}

void PinholeFullCamera::spaceToPlane(const Eigen::Vector3d & _P, Eigen::Vector2d & _p, float _image_scalse) const
{
  Eigen::Vector2d p_tmp;
  spaceToPlane(_P, p_tmp);
  _p = p_tmp * _image_scalse;
}

/**
 * \brief Project a 3D point to the image plane and calculate Jacobian
 *
 * \param P 3D point coordinates
 * \param p return value, contains the image point coordinates
 */
void PinholeFullCamera::spaceToPlane(
  const Eigen::Vector3d & _P, Eigen::Vector2d & _p, Eigen::Matrix<double, 2, 3> & _J) const
{
  Eigen::Vector2d p_u, p_d;
  double norm, inv_denom;
  double dxdmx, dydmx, dxdmy, dydmy;

  norm = _P.norm();
  // Project points to the normalised plane
  inv_denom = 1.0 / _P(2);
  p_u << inv_denom * _P(0), inv_denom * _P(1);

  // Calculate jacobian
  double dudx = inv_denom;
  double dvdx = 0.0;
  double dudy = 0.0;
  double dvdy = inv_denom;
  inv_denom = -inv_denom * inv_denom;
  double dudz = _P(0) * inv_denom;
  double dvdz = _P(1) * inv_denom;

  if (m_noDistortion_) {
    p_d = p_u;
  } else {
    // Apply distortion
    Eigen::Vector2d d_u;
    distortion(p_u, d_u);
    p_d = p_u + d_u;
  }

  double fx = mParameters_.fx();
  double fy = mParameters_.fy();

  // Make the product of the jacobians
  // and add projection matrix jacobian
  inv_denom = fx * (dudx * dxdmx + dvdx * dxdmy); // reuse
  dvdx = fy * (dudx * dydmx + dvdx * dydmy);
  dudx = inv_denom;

  inv_denom = fx * (dudy * dxdmx + dvdy * dxdmy); // reuse
  dvdy = fy * (dudy * dydmx + dvdy * dydmy);
  dudy = inv_denom;

  inv_denom = fx * (dudz * dxdmx + dvdz * dxdmy); // reuse
  dvdz = fy * (dudz * dydmx + dvdz * dydmy);
  dudz = inv_denom;

  // Apply generalised projection matrix
  _p << fx * p_d(0) + mParameters_.cx(), fy * p_d(1) + mParameters_.cy();

  _J << dudx, dudy, dudz, dvdx, dvdy, dvdz;
}

/**
 * \brief Projects an undistorted 2D point p_u to the image plane
 *
 * \param p_u 2D point coordinates
 * \return image point coordinates
 */
void PinholeFullCamera::undistToPlane(const Eigen::Vector2d & _p_u, Eigen::Vector2d & _p) const
{
  Eigen::Vector2d p_d;

  if (m_noDistortion_) {
    p_d = _p_u;
  } else {
    // Apply distortion
    Eigen::Vector2d d_u;
    distortion(_p_u, d_u);
    p_d = _p_u + d_u;
  }

  // Apply generalised projection matrix
  _p << mParameters_.fx() * p_d(0) + mParameters_.cx(), mParameters_.fy() * p_d(1) + mParameters_.cy();
}

/**
 * \brief Apply distortion to input point (from the normalised plane)
 *
 * \param p_u undistorted coordinates of point on the normalised plane
 * \return to obtain the distorted point: p_d = p_u + d_u
 */
void PinholeFullCamera::distortion(const Eigen::Vector2d & _p_u, Eigen::Vector2d & _d_u) const
{
  // project 3D object point to the image plane
  double k1 = mParameters_.k1();
  double k2 = mParameters_.k2();
  double k3 = mParameters_.k3();
  double k4 = mParameters_.k4();
  double k5 = mParameters_.k5();
  double k6 = mParameters_.k6();
  double p1 = mParameters_.p1();
  double p2 = mParameters_.p2();

  // Transform to model plane
  double x = _p_u(0);
  double y = _p_u(1);

  double r2 = x * x + y * y;
  double r4 = r2 * r2;
  double r6 = r4 * r2;
  double a1 = 2 * x * y;
  double a2 = r2 + 2 * x * x;
  double a3 = r2 + 2 * y * y;
  double cdist = 1 + k1 * r2 + k2 * r4 + k3 * r6;
  double icdist2 = 1. / (1 + k4 * r2 + k5 * r4 + k6 * r6);

  _d_u << x * cdist * icdist2 + p1 * a1 + p2 * a2 - x, //
    y * cdist * icdist2 + p1 * a3 + p2 * a1 - y;
}

/**
 * \brief Apply distortion to input point (from the normalised plane)
 *        and calculate Jacobian
 *
 * \param p_u undistorted coordinates of point on the normalised plane
 * \return to obtain the distorted point: p_d = p_u + d_u
 */
void PinholeFullCamera::distortion(
  const Eigen::Vector2d & _p_u, Eigen::Vector2d & _d_u, [[maybe_unused]] Eigen::Matrix2d & _J) const
{
  // project 3D object point to the image plane
  double k1 = mParameters_.k1();
  double k2 = mParameters_.k2();
  double k3 = mParameters_.k3();
  double k4 = mParameters_.k4();
  double k5 = mParameters_.k5();
  double k6 = mParameters_.k6();
  double p1 = mParameters_.p1();
  double p2 = mParameters_.p2();

  // Transform to model plane
  double x = _p_u(0);
  double y = _p_u(1);

  double r2 = x * x + y * y;
  double r4 = r2 * r2;
  double r6 = r4 * r2;
  double a1 = 2 * x * y;
  double a2 = r2 + 2 * x * x;
  double a3 = r2 + 2 * y * y;
  double cdist = 1 + k1 * r2 + k2 * r4 + k3 * r6;
  double icdist2 = 1. / (1 + k4 * r2 + k5 * r4 + k6 * r6);

  _d_u << x * cdist * icdist2 + p1 * a1 + p2 * a2 - x, //
    y * cdist * icdist2 + p1 * a3 + p2 * a1 - y;       //
}

void PinholeFullCamera::initUndistortMap(cv::Mat & _map1, cv::Mat & _map2, double _fScale) const
{
  cv::Size imageSize(mParameters_.imageWidth(), mParameters_.imageHeight());

  cv::Mat mapX = cv::Mat::zeros(imageSize, CV_32F);
  cv::Mat mapY = cv::Mat::zeros(imageSize, CV_32F);

  for (int v = 0; v < imageSize.height; ++v) {
    for (int u = 0; u < imageSize.width; ++u) {
      double mx_u = m_inv_K11_ / _fScale * u + m_inv_K13_ / _fScale;
      double my_u = m_inv_K22_ / _fScale * v + m_inv_K23_ / _fScale;

      Eigen::Vector3d P;
      P << mx_u, my_u, 1.0;

      Eigen::Vector2d p;
      spaceToPlane(P, p);

      mapX.at<float>(v, u) = p(0);
      mapY.at<float>(v, u) = p(1);
    }
  }

  cv::convertMaps(mapX, mapY, _map1, _map2, CV_32FC1, false);
}

cv::Mat PinholeFullCamera::initUndistortRectifyMap(
  cv::Mat & _map1, cv::Mat & _map2, float _fx, float _fy, cv::Size _imageSize, float _cx, float _cy,
  cv::Mat _rmat) const
{
  if (_imageSize == cv::Size(0, 0)) {
    _imageSize = cv::Size(mParameters_.imageWidth(), mParameters_.imageHeight());
  }

  cv::Mat mapX = cv::Mat::zeros(_imageSize.height, _imageSize.width, CV_32F);
  cv::Mat mapY = cv::Mat::zeros(_imageSize.height, _imageSize.width, CV_32F);

  Eigen::Matrix3f R, R_inv;
  cv::cv2eigen(_rmat, R);
  R_inv = R.inverse();

  // assume no skew
  Eigen::Matrix3f K_rect;

  if (_cx == -1.0f || _cy == -1.0f) {
    K_rect << _fx, 0, _imageSize.width / 2, 0, _fy, _imageSize.height / 2, 0, 0, 1;
  } else {
    K_rect << _fx, 0, _cx, 0, _fy, _cy, 0, 0, 1;
  }

  if (_fx == -1.0f || _fy == -1.0f) {
    K_rect(0, 0) = mParameters_.fx();
    K_rect(1, 1) = mParameters_.fy();
  }

  Eigen::Matrix3f K_rect_inv = K_rect.inverse();

  for (int v = 0; v < _imageSize.height; ++v) {
    for (int u = 0; u < _imageSize.width; ++u) {
      Eigen::Vector3f xo;
      xo << u, v, 1;

      Eigen::Vector3f uo = R_inv * K_rect_inv * xo;

      Eigen::Vector2d p;
      spaceToPlane(uo.cast<double>(), p);

      mapX.at<float>(v, u) = p(0);
      mapY.at<float>(v, u) = p(1);
    }
  }

  cv::convertMaps(mapX, mapY, _map1, _map2, CV_32FC1, false);

  cv::Mat K_rect_cv;
  cv::eigen2cv(K_rect, K_rect_cv);
  return K_rect_cv;
}

int PinholeFullCamera::parameterCount() const { return 12; }

const PinholeFullCamera::Parameters & PinholeFullCamera::getParameters() const { return mParameters_; }

void PinholeFullCamera::setParameters(const PinholeFullCamera::Parameters & _parameters)
{
  mParameters_ = _parameters;

  m_noDistortion_ = (mParameters_.k1() == 0.0) && (mParameters_.k2() == 0.0) && (mParameters_.p1() == 0.0) &&
                    (mParameters_.p2() == 0.0);

  m_inv_K11_ = 1.0 / mParameters_.fx();
  m_inv_K13_ = -mParameters_.cx() / mParameters_.fx();
  m_inv_K22_ = 1.0 / mParameters_.fy();
  m_inv_K23_ = -mParameters_.cy() / mParameters_.fy();
}

void PinholeFullCamera::readParameters(const std::vector<double> & _parameterVec)
{
  if ((int)_parameterVec.size() != parameterCount()) {
    return;
  }

  Parameters params = getParameters();

  params.k1() = _parameterVec.at(0);
  params.k2() = _parameterVec.at(1);
  params.k3() = _parameterVec.at(2);
  params.k4() = _parameterVec.at(3);
  params.k5() = _parameterVec.at(4);
  params.k6() = _parameterVec.at(5);
  params.p1() = _parameterVec.at(6);
  params.p2() = _parameterVec.at(7);
  params.fx() = _parameterVec.at(8);
  params.fy() = _parameterVec.at(9);
  params.cx() = _parameterVec.at(10);
  params.cy() = _parameterVec.at(11);

  setParameters(params);
}

void PinholeFullCamera::writeParameters(std::vector<double> & _parameterVec) const
{
  _parameterVec.resize(parameterCount());
  _parameterVec.at(0) = mParameters_.k1();
  _parameterVec.at(1) = mParameters_.k2();
  _parameterVec.at(2) = mParameters_.k3();
  _parameterVec.at(3) = mParameters_.k4();
  _parameterVec.at(4) = mParameters_.k5();
  _parameterVec.at(5) = mParameters_.k6();
  _parameterVec.at(6) = mParameters_.p1();
  _parameterVec.at(7) = mParameters_.p2();
  _parameterVec.at(8) = mParameters_.fx();
  _parameterVec.at(9) = mParameters_.fy();
  _parameterVec.at(10) = mParameters_.cx();
  _parameterVec.at(11) = mParameters_.cy();
}

void PinholeFullCamera::writeParametersToYamlFile(const std::string & _filename) const
{
  mParameters_.writeToYamlFile(_filename);
}

std::string PinholeFullCamera::parametersToString() const
{
  std::ostringstream oss;
  oss << mParameters_;

  return oss.str();
}
} // namespace camodocal
