#include "camodocal/camera_models/PinholeCamera.h"

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

PinholeCamera::Parameters::Parameters()
  : Camera::Parameters(PINHOLE), m_k1_(0.0), m_k2_(0.0), m_p1_(0.0), m_p2_(0.0), m_fx_(0.0), m_fy_(0.0),
    m_cx_(0.0), m_cy_(0.0)
{
}

PinholeCamera::Parameters::Parameters(
  const std::string & _cameraName, int _w, int _h, double _k1, double _k2, double _p1, double _p2, double _fx,
  double _fy, double _cx, double _cy)
  : Camera::Parameters(PINHOLE, _cameraName, _w, _h), m_k1_(_k1), m_k2_(_k2), m_p1_(_p1), m_p2_(_p2),
    m_fx_(_fx), m_fy_(_fy), m_cx_(_cx), m_cy_(_cy)
{
}

double & PinholeCamera::Parameters::k1() { return m_k1_; }

double & PinholeCamera::Parameters::k2() { return m_k2_; }

double & PinholeCamera::Parameters::p1() { return m_p1_; }

double & PinholeCamera::Parameters::p2() { return m_p2_; }

double & PinholeCamera::Parameters::fx() { return m_fx_; }

double & PinholeCamera::Parameters::fy() { return m_fy_; }

double & PinholeCamera::Parameters::cx() { return m_cx_; }

double & PinholeCamera::Parameters::cy() { return m_cy_; }

double PinholeCamera::Parameters::k1() const { return m_k1_; }

double PinholeCamera::Parameters::k2() const { return m_k2_; }

double PinholeCamera::Parameters::p1() const { return m_p1_; }

double PinholeCamera::Parameters::p2() const { return m_p2_; }

double PinholeCamera::Parameters::fx() const { return m_fx_; }

double PinholeCamera::Parameters::fy() const { return m_fy_; }

double PinholeCamera::Parameters::cx() const { return m_cx_; }

double PinholeCamera::Parameters::cy() const { return m_cy_; }

bool PinholeCamera::Parameters::readFromYamlFile(const std::string & _filename)
{
  cv::FileStorage fs(_filename, cv::FileStorage::READ);

  if (!fs.isOpened()) {
    return false;
  }

  if (!fs["model_type"].isNone()) {
    std::string sModelType;
    fs["model_type"] >> sModelType;

    if (sModelType.compare("PINHOLE") != 0) {
      return false;
    }
  }

  m_modelType_ = PINHOLE;
  fs["camera_name"] >> m_cameraName_;
  m_imageWidth_ = static_cast<int>(fs["image_width"]);
  m_imageHeight_ = static_cast<int>(fs["image_height"]);

  cv::FileNode n = fs["distortion_parameters"];
  m_k1_ = static_cast<double>(n["k1"]);
  m_k2_ = static_cast<double>(n["k2"]);
  m_p1_ = static_cast<double>(n["p1"]);
  m_p2_ = static_cast<double>(n["p2"]);

  n = fs["projection_parameters"];
  m_fx_ = static_cast<double>(n["fx"]);
  m_fy_ = static_cast<double>(n["fy"]);
  m_cx_ = static_cast<double>(n["cx"]);
  m_cy_ = static_cast<double>(n["cy"]);

  return true;
}

void PinholeCamera::Parameters::writeToYamlFile(const std::string & _filename) const
{
  cv::FileStorage fs(_filename, cv::FileStorage::WRITE);

  fs << "model_type" << "PINHOLE";
  fs << "camera_name" << m_cameraName_;
  fs << "image_width" << m_imageWidth_;
  fs << "image_height" << m_imageHeight_;

  // radial distortion: k1, k2
  // tangential distortion: p1, p2
  fs << "distortion_parameters";
  fs << "{" << "k1" << m_k1_ << "k2" << m_k2_ << "p1" << m_p1_ << "p2" << m_p2_ << "}";

  // projection: fx, fy, cx, cy
  fs << "projection_parameters";
  fs << "{" << "fx" << m_fx_ << "fy" << m_fy_ << "cx" << m_cx_ << "cy" << m_cy_ << "}";

  fs.release();
}

PinholeCamera::Parameters & PinholeCamera::Parameters::operator=(const PinholeCamera::Parameters & _other)
{
  if (this != &_other) {
    m_modelType_ = _other.m_modelType_;
    m_cameraName_ = _other.m_cameraName_;
    m_imageWidth_ = _other.m_imageWidth_;
    m_imageHeight_ = _other.m_imageHeight_;
    m_k1_ = _other.m_k1_;
    m_k2_ = _other.m_k2_;
    m_p1_ = _other.m_p1_;
    m_p2_ = _other.m_p2_;
    m_fx_ = _other.m_fx_;
    m_fy_ = _other.m_fy_;
    m_cx_ = _other.m_cx_;
    m_cy_ = _other.m_cy_;
  }

  return *this;
}

std::ostream & operator<<(std::ostream & _out, const PinholeCamera::Parameters & _params)
{
  _out << "Camera Parameters:" << '\n';
  _out << "    model_type " << "PINHOLE" << '\n';
  _out << "   camera_name " << _params.m_cameraName_ << '\n';
  _out << "   image_width " << _params.m_imageWidth_ << '\n';
  _out << "  image_height " << _params.m_imageHeight_ << '\n';

  // radial distortion: k1, k2
  // tangential distortion: p1, p2
  _out << "Distortion Parameters" << '\n';
  _out << "            k1 " << _params.m_k1_ << '\n'
       << "            k2 " << _params.m_k2_ << '\n'
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

PinholeCamera::PinholeCamera()
  : m_inv_K11_(1.0), m_inv_K13_(0.0), m_inv_K22_(1.0), m_inv_K23_(0.0), m_noDistortion_(true)
{
}

PinholeCamera::PinholeCamera(
  const std::string & _cameraName, int _imageWidth, int _imageHeight, double _k1, double _k2, double _p1,
  double _p2, double _fx, double _fy, double _cx, double _cy)
  : mParameters_(_cameraName, _imageWidth, _imageHeight, _k1, _k2, _p1, _p2, _fx, _fy, _cx, _cy)
{
  m_noDistortion_ = (mParameters_.k1() == 0.0) && (mParameters_.k2() == 0.0) && (mParameters_.p1() == 0.0) &&
                    (mParameters_.p2() == 0.0);

  // Inverse camera projection matrix parameters
  m_inv_K11_ = 1.0 / mParameters_.fx();
  m_inv_K13_ = -mParameters_.cx() / mParameters_.fx();
  m_inv_K22_ = 1.0 / mParameters_.fy();
  m_inv_K23_ = -mParameters_.cy() / mParameters_.fy();
}

PinholeCamera::PinholeCamera(const PinholeCamera::Parameters & _params) : mParameters_(_params)
{
  m_noDistortion_ = (mParameters_.k1() == 0.0) && (mParameters_.k2() == 0.0) && (mParameters_.p1() == 0.0) &&
                    (mParameters_.p2() == 0.0);

  // Inverse camera projection matrix parameters
  m_inv_K11_ = 1.0 / mParameters_.fx();
  m_inv_K13_ = -mParameters_.cx() / mParameters_.fx();
  m_inv_K22_ = 1.0 / mParameters_.fy();
  m_inv_K23_ = -mParameters_.cy() / mParameters_.fy();
}

Camera::ModelType_e PinholeCamera::modelType() const { return mParameters_.modelType(); }

const std::string & PinholeCamera::cameraName() const { return mParameters_.cameraName(); }

int PinholeCamera::imageWidth() const { return mParameters_.imageWidth(); }

int PinholeCamera::imageHeight() const { return mParameters_.imageHeight(); }

void PinholeCamera::estimateIntrinsics(
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
void PinholeCamera::liftSphere(const Eigen::Vector2d & _p, Eigen::Vector3d & _P) const
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
void PinholeCamera::liftProjective(const Eigen::Vector2d & _p, Eigen::Vector3d & _P) const
{
  double mx_d, my_d, mx2_d, mxy_d, my2_d, mx_u, my_u;
  double rho2_d, rho4_d, radDist_d, Dx_d, Dy_d, inv_denom_d;
  //double lambda;

  // Lift points to normalised plane
  mx_d = m_inv_K11_ * _p(0) + m_inv_K13_;
  my_d = m_inv_K22_ * _p(1) + m_inv_K23_;

  if (m_noDistortion_) {
    mx_u = mx_d;
    my_u = my_d;
  } else {
    if (0) {
      double k1 = mParameters_.k1();
      double k2 = mParameters_.k2();
      double p1 = mParameters_.p1();
      double p2 = mParameters_.p2();

      // Apply inverse distortion model
      // proposed by Heikkila
      mx2_d = mx_d * mx_d;
      my2_d = my_d * my_d;
      mxy_d = mx_d * my_d;
      rho2_d = mx2_d + my2_d;
      rho4_d = rho2_d * rho2_d;
      radDist_d = k1 * rho2_d + k2 * rho4_d;
      Dx_d = mx_d * radDist_d + p2 * (rho2_d + 2 * mx2_d) + 2 * p1 * mxy_d;
      Dy_d = my_d * radDist_d + p1 * (rho2_d + 2 * my2_d) + 2 * p2 * mxy_d;
      inv_denom_d = 1 / (1 + 4 * k1 * rho2_d + 6 * k2 * rho4_d + 8 * p1 * my_d + 8 * p2 * mx_d);

      mx_u = mx_d - inv_denom_d * Dx_d;
      my_u = my_d - inv_denom_d * Dy_d;
    } else {
      // Recursive distortion model
      int n = 8;
      Eigen::Vector2d d_u;
      distortion(Eigen::Vector2d(mx_d, my_d), d_u);
      // Approximate value
      mx_u = mx_d - d_u(0);
      my_u = my_d - d_u(1);

      for (int i = 1; i < n; ++i) {
        distortion(Eigen::Vector2d(mx_u, my_u), d_u);
        mx_u = mx_d - d_u(0);
        my_u = my_d - d_u(1);
      }
    }
  }

  // Obtain a projective ray
  _P << mx_u, my_u, 1.0;
}


/**
 * \brief Project a 3D point (\a x,\a y,\a z) to the image plane in (\a u,\a v)
 *
 * \param P 3D point coordinates
 * \param p return value, contains the image point coordinates
 */
void PinholeCamera::spaceToPlane(const Eigen::Vector3d & _P, Eigen::Vector2d & _p) const
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

#if 0
/**
 * \brief Project a 3D point to the image plane and calculate Jacobian
 *
 * \param P 3D point coordinates
 * \param p return value, contains the image point coordinates
 */
void
PinholeCamera::spaceToPlane(const Eigen::Vector3d& P, Eigen::Vector2d& p,
                            Eigen::Matrix<double,2,3>& J) const
{
    Eigen::Vector2d p_u, p_d;
    double norm, inv_denom;
    double dxdmx, dydmx, dxdmy, dydmy;

    norm = P.norm();
    // Project points to the normalised plane
    inv_denom = 1.0 / P(2);
    p_u << inv_denom * P(0), inv_denom * P(1);

    // Calculate jacobian
    double dudx = inv_denom;
    double dvdx = 0.0;
    double dudy = 0.0;
    double dvdy = inv_denom;
    inv_denom = - inv_denom * inv_denom;
    double dudz = P(0) * inv_denom;
    double dvdz = P(1) * inv_denom;

    if (m_noDistortion)
    {
        p_d = p_u;
    }
    else
    {
        // Apply distortion
        Eigen::Vector2d d_u;
        distortion(p_u, d_u);
        p_d = p_u + d_u;
    }

    double fx = mParameters.fx();
    double fy = mParameters.fy();

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
    p << fx * p_d(0) + mParameters.cx(),
         fy * p_d(1) + mParameters.cy();

    J << dudx, dudy, dudz,
         dvdx, dvdy, dvdz;
}
#endif

/**
 * \brief Projects an undistorted 2D point p_u to the image plane
 *
 * \param p_u 2D point coordinates
 * \return image point coordinates
 */
void PinholeCamera::undistToPlane(const Eigen::Vector2d & _p_u, Eigen::Vector2d & _p) const
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
void PinholeCamera::distortion(const Eigen::Vector2d & _p_u, Eigen::Vector2d & _d_u) const
{
  double k1 = mParameters_.k1();
  double k2 = mParameters_.k2();
  double p1 = mParameters_.p1();
  double p2 = mParameters_.p2();

  double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

  mx2_u = _p_u(0) * _p_u(0);
  my2_u = _p_u(1) * _p_u(1);
  mxy_u = _p_u(0) * _p_u(1);
  rho2_u = mx2_u + my2_u;
  rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
  _d_u << _p_u(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
    _p_u(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);
}

/**
 * \brief Apply distortion to input point (from the normalised plane)
 *        and calculate Jacobian
 *
 * \param p_u undistorted coordinates of point on the normalised plane
 * \return to obtain the distorted point: p_d = p_u + d_u
 */
void PinholeCamera::distortion(const Eigen::Vector2d & _p_u, Eigen::Vector2d & _d_u, Eigen::Matrix2d & _J) const
{
  double k1 = mParameters_.k1();
  double k2 = mParameters_.k2();
  double p1 = mParameters_.p1();
  double p2 = mParameters_.p2();

  double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

  mx2_u = _p_u(0) * _p_u(0);
  my2_u = _p_u(1) * _p_u(1);
  mxy_u = _p_u(0) * _p_u(1);
  rho2_u = mx2_u + my2_u;
  rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
  _d_u << _p_u(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
    _p_u(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);

  double dxdmx =
    1.0 + rad_dist_u + k1 * 2.0 * mx2_u + k2 * rho2_u * 4.0 * mx2_u + 2.0 * p1 * _p_u(1) + 6.0 * p2 * _p_u(0);
  double dydmx = k1 * 2.0 * _p_u(0) * _p_u(1) + k2 * 4.0 * rho2_u * _p_u(0) * _p_u(1) + p1 * 2.0 * _p_u(0) +
                 2.0 * p2 * _p_u(1);
  double dxdmy = dydmx;
  double dydmy =
    1.0 + rad_dist_u + k1 * 2.0 * my2_u + k2 * rho2_u * 4.0 * my2_u + 6.0 * p1 * _p_u(1) + 2.0 * p2 * _p_u(0);

  _J << dxdmx, dxdmy, dydmx, dydmy;
}

void PinholeCamera::initUndistortMap(cv::Mat & _map1, cv::Mat & _map2, double _fScale) const
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

cv::Mat PinholeCamera::initUndistortRectifyMap(
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

int PinholeCamera::parameterCount() const { return 8; }

const PinholeCamera::Parameters & PinholeCamera::getParameters() const { return mParameters_; }

void PinholeCamera::setParameters(const PinholeCamera::Parameters & _parameters)
{
  mParameters_ = _parameters;

  m_noDistortion_ = (mParameters_.k1() == 0.0) && (mParameters_.k2() == 0.0) && (mParameters_.p1() == 0.0) &&
                    (mParameters_.p2() == 0.0);

  m_inv_K11_ = 1.0 / mParameters_.fx();
  m_inv_K13_ = -mParameters_.cx() / mParameters_.fx();
  m_inv_K22_ = 1.0 / mParameters_.fy();
  m_inv_K23_ = -mParameters_.cy() / mParameters_.fy();
}

void PinholeCamera::readParameters(const std::vector<double> & _parameterVec)
{
  if ((int)_parameterVec.size() != parameterCount()) {
    return;
  }

  Parameters params = getParameters();

  params.k1() = _parameterVec.at(0);
  params.k2() = _parameterVec.at(1);
  params.p1() = _parameterVec.at(2);
  params.p2() = _parameterVec.at(3);
  params.fx() = _parameterVec.at(4);
  params.fy() = _parameterVec.at(5);
  params.cx() = _parameterVec.at(6);
  params.cy() = _parameterVec.at(7);

  setParameters(params);
}

void PinholeCamera::writeParameters(std::vector<double> & _parameterVec) const
{
  _parameterVec.resize(parameterCount());
  _parameterVec.at(0) = mParameters_.k1();
  _parameterVec.at(1) = mParameters_.k2();
  _parameterVec.at(2) = mParameters_.p1();
  _parameterVec.at(3) = mParameters_.p2();
  _parameterVec.at(4) = mParameters_.fx();
  _parameterVec.at(5) = mParameters_.fy();
  _parameterVec.at(6) = mParameters_.cx();
  _parameterVec.at(7) = mParameters_.cy();
}

void PinholeCamera::writeParametersToYamlFile(const std::string & _filename) const
{
  mParameters_.writeToYamlFile(_filename);
}

std::string PinholeCamera::parametersToString() const
{
  std::ostringstream oss;
  oss << mParameters_;

  return oss.str();
}

} // namespace camodocal
