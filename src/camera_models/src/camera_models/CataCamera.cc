#include "camodocal/camera_models/CataCamera.h"

#include <cmath>
#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camodocal/gpl/gpl.h"

namespace camodocal
{

CataCamera::Parameters::Parameters()
  : Camera::Parameters(MEI), m_xi_(0.0), m_k1_(0.0), m_k2_(0.0), m_p1_(0.0), m_p2_(0.0), m_gamma1_(0.0),
    m_gamma2_(0.0), m_u0_(0.0), m_v0_(0.0)
{
}

CataCamera::Parameters::Parameters(
  const std::string & _cameraName, int _w, int _h, double _xi, double _k1, double _k2, double _p1, double _p2,
  double _gamma1, double _gamma2, double _u0, double _v0)
  : Camera::Parameters(MEI, _cameraName, _w, _h), m_xi_(_xi), m_k1_(_k1), m_k2_(_k2), m_p1_(_p1), m_p2_(_p2),
    m_gamma1_(_gamma1), m_gamma2_(_gamma2), m_u0_(_u0), m_v0_(_v0)
{
}

double & CataCamera::Parameters::xi() { return m_xi_; }

double & CataCamera::Parameters::k1() { return m_k1_; }

double & CataCamera::Parameters::k2() { return m_k2_; }

double & CataCamera::Parameters::p1() { return m_p1_; }

double & CataCamera::Parameters::p2() { return m_p2_; }

double & CataCamera::Parameters::gamma1() { return m_gamma1_; }

double & CataCamera::Parameters::gamma2() { return m_gamma2_; }

double & CataCamera::Parameters::u0() { return m_u0_; }

double & CataCamera::Parameters::v0() { return m_v0_; }

double CataCamera::Parameters::xi() const { return m_xi_; }

double CataCamera::Parameters::k1() const { return m_k1_; }

double CataCamera::Parameters::k2() const { return m_k2_; }

double CataCamera::Parameters::p1() const { return m_p1_; }

double CataCamera::Parameters::p2() const { return m_p2_; }

double CataCamera::Parameters::gamma1() const { return m_gamma1_; }

double CataCamera::Parameters::gamma2() const { return m_gamma2_; }

double CataCamera::Parameters::u0() const { return m_u0_; }

double CataCamera::Parameters::v0() const { return m_v0_; }

bool CataCamera::Parameters::readFromYamlFile(const std::string & _filename)
{
  cv::FileStorage fs(_filename, cv::FileStorage::READ);

  if (!fs.isOpened()) {
    return false;
  }

  if (!fs["model_type"].isNone()) {
    std::string sModelType;
    fs["model_type"] >> sModelType;

    if (sModelType.compare("MEI") != 0) {
      return false;
    }
  }

  m_modelType_ = MEI;
  fs["camera_name"] >> m_cameraName_;
  m_imageWidth_ = static_cast<int>(fs["image_width"]);
  m_imageHeight_ = static_cast<int>(fs["image_height"]);

  cv::FileNode n = fs["mirror_parameters"];
  m_xi_ = static_cast<double>(n["xi"]);

  n = fs["distortion_parameters"];
  m_k1_ = static_cast<double>(n["k1"]);
  m_k2_ = static_cast<double>(n["k2"]);
  m_p1_ = static_cast<double>(n["p1"]);
  m_p2_ = static_cast<double>(n["p2"]);

  n = fs["projection_parameters"];
  m_gamma1_ = static_cast<double>(n["gamma1"]);
  m_gamma2_ = static_cast<double>(n["gamma2"]);
  m_u0_ = static_cast<double>(n["u0"]);
  m_v0_ = static_cast<double>(n["v0"]);

  return true;
}

void CataCamera::Parameters::writeToYamlFile(const std::string & _filename) const
{
  cv::FileStorage fs(_filename, cv::FileStorage::WRITE);

  fs << "model_type" << "MEI";
  fs << "camera_name" << m_cameraName_;
  fs << "image_width" << m_imageWidth_;
  fs << "image_height" << m_imageHeight_;

  // mirror: xi
  fs << "mirror_parameters";
  fs << "{" << "xi" << m_xi_ << "}";

  // radial distortion: k1, k2
  // tangential distortion: p1, p2
  fs << "distortion_parameters";
  fs << "{" << "k1" << m_k1_ << "k2" << m_k2_ << "p1" << m_p1_ << "p2" << m_p2_ << "}";

  // projection: gamma1, gamma2, u0, v0
  fs << "projection_parameters";
  fs << "{" << "gamma1" << m_gamma1_ << "gamma2" << m_gamma2_ << "u0" << m_u0_ << "v0" << m_v0_ << "}";

  fs.release();
}

CataCamera::Parameters & CataCamera::Parameters::operator=(const CataCamera::Parameters & _other)
{
  if (this != &_other) {
    m_modelType_ = _other.m_modelType_;
    m_cameraName_ = _other.m_cameraName_;
    m_imageWidth_ = _other.m_imageWidth_;
    m_imageHeight_ = _other.m_imageHeight_;
    m_xi_ = _other.m_xi_;
    m_k1_ = _other.m_k1_;
    m_k2_ = _other.m_k2_;
    m_p1_ = _other.m_p1_;
    m_p2_ = _other.m_p2_;
    m_gamma1_ = _other.m_gamma1_;
    m_gamma2_ = _other.m_gamma2_;
    m_u0_ = _other.m_u0_;
    m_v0_ = _other.m_v0_;
  }

  return *this;
}

std::ostream & operator<<(std::ostream & _out, const CataCamera::Parameters & _params)
{
  _out << "Camera Parameters:" << '\n';
  _out << "    model_type " << "MEI" << '\n';
  _out << "   camera_name " << _params.m_cameraName_ << '\n';
  _out << "   image_width " << _params.m_imageWidth_ << '\n';
  _out << "  image_height " << _params.m_imageHeight_ << '\n';

  _out << "Mirror Parameters" << '\n';
  _out << std::fixed << std::setprecision(10);
  _out << "            xi " << _params.m_xi_ << '\n';

  // radial distortion: k1, k2
  // tangential distortion: p1, p2
  _out << "Distortion Parameters" << '\n';
  _out << "            k1 " << _params.m_k1_ << '\n'
       << "            k2 " << _params.m_k2_ << '\n'
       << "            p1 " << _params.m_p1_ << '\n'
       << "            p2 " << _params.m_p2_ << '\n';

  // projection: gamma1, gamma2, u0, v0
  _out << "Projection Parameters" << '\n';
  _out << "        gamma1 " << _params.m_gamma1_ << '\n'
       << "        gamma2 " << _params.m_gamma2_ << '\n'
       << "            u0 " << _params.m_u0_ << '\n'
       << "            v0 " << _params.m_v0_ << '\n';

  return _out;
}

CataCamera::CataCamera()
  : m_inv_K11_(1.0), m_inv_K13_(0.0), m_inv_K22_(1.0), m_inv_K23_(0.0), m_noDistortion_(true)
{
}

CataCamera::CataCamera(
  const std::string & _cameraName, int _imageWidth, int _imageHeight, double _xi, double _k1, double _k2,
  double _p1, double _p2, double _gamma1, double _gamma2, double _u0, double _v0)
  : mParameters_(_cameraName, _imageWidth, _imageHeight, _xi, _k1, _k2, _p1, _p2, _gamma1, _gamma2, _u0, _v0)
{
  m_noDistortion_ = (mParameters_.k1() == 0.0) && (mParameters_.k2() == 0.0) && (mParameters_.p1() == 0.0) &&
                    (mParameters_.p2() == 0.0);

  // Inverse camera projection matrix parameters
  m_inv_K11_ = 1.0 / mParameters_.gamma1();
  m_inv_K13_ = -mParameters_.u0() / mParameters_.gamma1();
  m_inv_K22_ = 1.0 / mParameters_.gamma2();
  m_inv_K23_ = -mParameters_.v0() / mParameters_.gamma2();
}

CataCamera::CataCamera(const CataCamera::Parameters & _params) : mParameters_(_params)
{
  m_noDistortion_ = (mParameters_.k1() == 0.0) && (mParameters_.k2() == 0.0) && (mParameters_.p1() == 0.0) &&
                    (mParameters_.p2() == 0.0);

  // Inverse camera projection matrix parameters
  m_inv_K11_ = 1.0 / mParameters_.gamma1();
  m_inv_K13_ = -mParameters_.u0() / mParameters_.gamma1();
  m_inv_K22_ = 1.0 / mParameters_.gamma2();
  m_inv_K23_ = -mParameters_.v0() / mParameters_.gamma2();
}

Camera::ModelType_e CataCamera::modelType() const { return mParameters_.modelType(); }

const std::string & CataCamera::cameraName() const { return mParameters_.cameraName(); }

int CataCamera::imageWidth() const { return mParameters_.imageWidth(); }

int CataCamera::imageHeight() const { return mParameters_.imageHeight(); }

void CataCamera::estimateIntrinsics(
  const cv::Size & _boardSize, const std::vector<std::vector<cv::Point3f>> & _objectPoints,
  const std::vector<std::vector<cv::Point2f>> & _imagePoints)
{
  Parameters params = getParameters();

  double u0 = params.imageWidth() / 2.0;
  double v0 = params.imageHeight() / 2.0;

  double gamma0 = 0.0;
  double minReprojErr = std::numeric_limits<double>::max();

  std::vector<cv::Mat> rvecs, tvecs;
  rvecs.assign(_objectPoints.size(), cv::Mat());
  tvecs.assign(_objectPoints.size(), cv::Mat());

  params.xi() = 1.0;
  params.k1() = 0.0;
  params.k2() = 0.0;
  params.p1() = 0.0;
  params.p2() = 0.0;
  params.u0() = u0;
  params.v0() = v0;

  // Initialize gamma (focal length)
  // Use non-radial line image and xi = 1
  for (size_t i = 0; i < _imagePoints.size(); ++i) {
    for (int r = 0; r < _boardSize.height; ++r) {
      cv::Mat P(_boardSize.width, 4, CV_64F);
      for (int c = 0; c < _boardSize.width; ++c) {
        const cv::Point2f & imagePoint = _imagePoints.at(i).at(r * _boardSize.width + c);

        double u = imagePoint.x - u0;
        double v = imagePoint.y - v0;

        P.at<double>(c, 0) = u;
        P.at<double>(c, 1) = v;
        P.at<double>(c, 2) = 0.5;
        P.at<double>(c, 3) = -0.5 * (square(u) + square(v));
      }

      cv::Mat C;
      cv::SVD::solveZ(P, C);

      double t = square(C.at<double>(0)) + square(C.at<double>(1)) + C.at<double>(2) * C.at<double>(3);
      if (t < 0.0) {
        continue;
      }

      // check that line image is not radial
      double d = sqrt(1.0 / t);
      double nx = C.at<double>(0) * d;
      double ny = C.at<double>(1) * d;
      if (hypot(nx, ny) > 0.95) {
        continue;
      }

      double gamma = sqrt(C.at<double>(2) / C.at<double>(3));

      params.gamma1() = gamma;
      params.gamma2() = gamma;
      setParameters(params);

      for (size_t j = 0; j < _objectPoints.size(); ++j) {
        estimateExtrinsics(_objectPoints.at(j), _imagePoints.at(j), rvecs.at(j), tvecs.at(j));
      }

      double reprojErr = reprojectionError(_objectPoints, _imagePoints, rvecs, tvecs, cv::noArray());

      if (reprojErr < minReprojErr) {
        minReprojErr = reprojErr;
        gamma0 = gamma;
      }
    }
  }

  if (gamma0 <= 0.0 && minReprojErr >= std::numeric_limits<double>::max()) {
    std::cout << "[" << params.cameraName() << "] " << "# INFO: CataCamera model fails with given data. "
              << '\n';

    return;
  }

  params.gamma1() = gamma0;
  params.gamma2() = gamma0;
  setParameters(params);
}

/** 
 * \brief Lifts a point from the image plane to the unit sphere
 *
 * \param p image coordinates
 * \param P coordinates of the point on the sphere
 */
void CataCamera::liftSphere(const Eigen::Vector2d & _p, Eigen::Vector3d & _P) const
{
  double mx_d, my_d, mx2_d, mxy_d, my2_d, mx_u, my_u;
  double rho2_d, rho4_d, radDist_d, Dx_d, Dy_d, inv_denom_d;
  double lambda;

  // Lift points to normalised plane
  mx_d = m_inv_K11_ * _p(0) + m_inv_K13_;
  my_d = m_inv_K22_ * _p(1) + m_inv_K23_;

  if (m_noDistortion_) {
    mx_u = mx_d;
    my_u = my_d;
  } else {
    // Apply inverse distortion model
    if (0) {
      double k1 = mParameters_.k1();
      double k2 = mParameters_.k2();
      double p1 = mParameters_.p1();
      double p2 = mParameters_.p2();

      // Inverse distortion model
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
      int n = 6;
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

  // Lift normalised points to the sphere (inv_hslash)
  double xi = mParameters_.xi();
  if (xi == 1.0) {
    lambda = 2.0 / (mx_u * mx_u + my_u * my_u + 1.0);
    _P << lambda * mx_u, lambda * my_u, lambda - 1.0;
  } else {
    lambda =
      (xi + sqrt(1.0 + (1.0 - xi * xi) * (mx_u * mx_u + my_u * my_u))) / (1.0 + mx_u * mx_u + my_u * my_u);
    _P << lambda * mx_u, lambda * my_u, lambda - xi;
  }
}

/** 
 * \brief Lifts a point from the image plane to its projective ray
 *
 * \param p image coordinates
 * \param P coordinates of the projective ray
 */
void CataCamera::liftProjective(const Eigen::Vector2d & _p, Eigen::Vector3d & _P) const
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
  double xi = mParameters_.xi();
  if (xi == 1.0) {
    _P << mx_u, my_u, (1.0 - mx_u * mx_u - my_u * my_u) / 2.0;
  } else {
    // Reuse variable
    rho2_d = mx_u * mx_u + my_u * my_u;
    _P << mx_u, my_u, 1.0 - xi * (rho2_d + 1.0) / (xi + sqrt(1.0 + (1.0 - xi * xi) * rho2_d));
  }
}


/** 
 * \brief Project a 3D point (\a x,\a y,\a z) to the image plane in (\a u,\a v)
 *
 * \param P 3D point coordinates
 * \param p return value, contains the image point coordinates
 */
void CataCamera::spaceToPlane(const Eigen::Vector3d & _P, Eigen::Vector2d & _p) const
{
  Eigen::Vector2d p_u, p_d;

  // Project points to the normalised plane
  double z = _P(2) + mParameters_.xi() * _P.norm();
  p_u << _P(0) / z, _P(1) / z;

  if (m_noDistortion_) {
    p_d = p_u;
  } else {
    // Apply distortion
    Eigen::Vector2d d_u;
    distortion(p_u, d_u);
    p_d = p_u + d_u;
  }

  // Apply generalised projection matrix
  _p << mParameters_.gamma1() * p_d(0) + mParameters_.u0(), mParameters_.gamma2() * p_d(1) + mParameters_.v0();
}

#if 0
/** 
 * \brief Project a 3D point to the image plane and calculate Jacobian
 *
 * \param P 3D point coordinates
 * \param p return value, contains the image point coordinates
 */
void
CataCamera::spaceToPlane(const Eigen::Vector3d& P, Eigen::Vector2d& p,
                        Eigen::Matrix<double,2,3>& J) const
{
    double xi = mParameters.xi();

    Eigen::Vector2d p_u, p_d;
    double norm, inv_denom;
    double dxdmx, dydmx, dxdmy, dydmy;

    norm = P.norm();
    // Project points to the normalised plane
    inv_denom = 1.0 / (P(2) + xi * norm);
    p_u << inv_denom * P(0), inv_denom * P(1);

    // Calculate jacobian
    inv_denom = inv_denom * inv_denom / norm;
    double dudx = inv_denom * (norm * P(2) + xi * (P(1) * P(1) + P(2) * P(2)));
    double dvdx = -inv_denom * xi * P(0) * P(1);
    double dudy = dvdx;
    double dvdy = inv_denom * (norm * P(2) + xi * (P(0) * P(0) + P(2) * P(2)));
    inv_denom = inv_denom * (-xi * P(2) - norm); // reuse variable
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

    double gamma1 = mParameters.gamma1();
    double gamma2 = mParameters.gamma2();

    // Make the product of the jacobians
    // and add projection matrix jacobian
    inv_denom = gamma1 * (dudx * dxdmx + dvdx * dxdmy); // reuse
    dvdx = gamma2 * (dudx * dydmx + dvdx * dydmy);
    dudx = inv_denom;

    inv_denom = gamma1 * (dudy * dxdmx + dvdy * dxdmy); // reuse
    dvdy = gamma2 * (dudy * dydmx + dvdy * dydmy);
    dudy = inv_denom;

    inv_denom = gamma1 * (dudz * dxdmx + dvdz * dxdmy); // reuse
    dvdz = gamma2 * (dudz * dydmx + dvdz * dydmy);
    dudz = inv_denom;
    
    // Apply generalised projection matrix
    p << gamma1 * p_d(0) + mParameters.u0(),
         gamma2 * p_d(1) + mParameters.v0();

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
void CataCamera::undistToPlane(const Eigen::Vector2d & _p_u, Eigen::Vector2d & _p) const
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
  _p << mParameters_.gamma1() * p_d(0) + mParameters_.u0(), mParameters_.gamma2() * p_d(1) + mParameters_.v0();
}

/** 
 * \brief Apply distortion to input point (from the normalised plane)
 *  
 * \param p_u undistorted coordinates of point on the normalised plane
 * \return to obtain the distorted point: p_d = p_u + d_u
 */
void CataCamera::distortion(const Eigen::Vector2d & _p_u, Eigen::Vector2d & _d_u) const
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
void CataCamera::distortion(const Eigen::Vector2d & _p_u, Eigen::Vector2d & _d_u, Eigen::Matrix2d & _J) const
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

void CataCamera::initUndistortMap(cv::Mat & _map1, cv::Mat & _map2, double _fScale) const
{
  cv::Size imageSize(mParameters_.imageWidth(), mParameters_.imageHeight());

  cv::Mat mapX = cv::Mat::zeros(imageSize, CV_32F);
  cv::Mat mapY = cv::Mat::zeros(imageSize, CV_32F);

  for (int v = 0; v < imageSize.height; ++v) {
    for (int u = 0; u < imageSize.width; ++u) {
      double mx_u = m_inv_K11_ / _fScale * u + m_inv_K13_ / _fScale;
      double my_u = m_inv_K22_ / _fScale * v + m_inv_K23_ / _fScale;

      double xi = mParameters_.xi();
      double d2 = mx_u * mx_u + my_u * my_u;

      Eigen::Vector3d P;
      P << mx_u, my_u, 1.0 - xi * (d2 + 1.0) / (xi + sqrt(1.0 + (1.0 - xi * xi) * d2));

      Eigen::Vector2d p;
      spaceToPlane(P, p);

      mapX.at<float>(v, u) = p(0);
      mapY.at<float>(v, u) = p(1);
    }
  }

  cv::convertMaps(mapX, mapY, _map1, _map2, CV_32FC1, false);
}

cv::Mat CataCamera::initUndistortRectifyMap(
  cv::Mat & _map1, cv::Mat & _map2, float _fx, float _fy, cv::Size _imageSize, float _cx, float _cy,
  cv::Mat _rmat) const
{
  if (_imageSize == cv::Size(0, 0)) {
    _imageSize = cv::Size(mParameters_.imageWidth(), mParameters_.imageHeight());
  }

  cv::Mat mapX = cv::Mat::zeros(_imageSize.height, _imageSize.width, CV_32F);
  cv::Mat mapY = cv::Mat::zeros(_imageSize.height, _imageSize.width, CV_32F);

  Eigen::Matrix3f K_rect;

  if (_cx == -1.0f && _cy == -1.0f) {
    K_rect << _fx, 0, _imageSize.width / 2, 0, _fy, _imageSize.height / 2, 0, 0, 1;
  } else {
    K_rect << _fx, 0, _cx, 0, _fy, _cy, 0, 0, 1;
  }

  if (_fx == -1.0f || _fy == -1.0f) {
    K_rect(0, 0) = mParameters_.gamma1();
    K_rect(1, 1) = mParameters_.gamma2();
  }

  Eigen::Matrix3f K_rect_inv = K_rect.inverse();

  Eigen::Matrix3f R, R_inv;
  cv::cv2eigen(_rmat, R);
  R_inv = R.inverse();

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

int CataCamera::parameterCount() const { return 9; }

const CataCamera::Parameters & CataCamera::getParameters() const { return mParameters_; }

void CataCamera::setParameters(const CataCamera::Parameters & _parameters)
{
  mParameters_ = _parameters;

  m_noDistortion_ = (mParameters_.k1() == 0.0) && (mParameters_.k2() == 0.0) && (mParameters_.p1() == 0.0) &&
                    (mParameters_.p2() == 0.0);

  m_inv_K11_ = 1.0 / mParameters_.gamma1();
  m_inv_K13_ = -mParameters_.u0() / mParameters_.gamma1();
  m_inv_K22_ = 1.0 / mParameters_.gamma2();
  m_inv_K23_ = -mParameters_.v0() / mParameters_.gamma2();
}

void CataCamera::readParameters(const std::vector<double> & _parameterVec)
{
  if ((int)_parameterVec.size() != parameterCount()) {
    return;
  }

  Parameters params = getParameters();

  params.xi() = _parameterVec.at(0);
  params.k1() = _parameterVec.at(1);
  params.k2() = _parameterVec.at(2);
  params.p1() = _parameterVec.at(3);
  params.p2() = _parameterVec.at(4);
  params.gamma1() = _parameterVec.at(5);
  params.gamma2() = _parameterVec.at(6);
  params.u0() = _parameterVec.at(7);
  params.v0() = _parameterVec.at(8);

  setParameters(params);
}

void CataCamera::writeParameters(std::vector<double> & _parameterVec) const
{
  _parameterVec.resize(parameterCount());
  _parameterVec.at(0) = mParameters_.xi();
  _parameterVec.at(1) = mParameters_.k1();
  _parameterVec.at(2) = mParameters_.k2();
  _parameterVec.at(3) = mParameters_.p1();
  _parameterVec.at(4) = mParameters_.p2();
  _parameterVec.at(5) = mParameters_.gamma1();
  _parameterVec.at(6) = mParameters_.gamma2();
  _parameterVec.at(7) = mParameters_.u0();
  _parameterVec.at(8) = mParameters_.v0();
}

void CataCamera::writeParametersToYamlFile(const std::string & _filename) const
{
  mParameters_.writeToYamlFile(_filename);
}

std::string CataCamera::parametersToString() const
{
  std::ostringstream oss;
  oss << mParameters_;

  return oss.str();
}

} // namespace camodocal
