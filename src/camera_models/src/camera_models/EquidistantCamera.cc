#include "camodocal/camera_models/EquidistantCamera.h"

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

EquidistantCamera::Parameters::Parameters()
  : Camera::Parameters(KANNALA_BRANDT), m_k2_(0.0), m_k3_(0.0), m_k4_(0.0), m_k5_(0.0), m_mu_(0.0),
    m_mv_(0.0), m_u0_(0.0), m_v0_(0.0)
{
}

EquidistantCamera::Parameters::Parameters(
  const std::string & _cameraName, int _w, int _h, double _k2, double _k3, double _k4, double _k5, double _mu,
  double _mv, double _u0, double _v0)
  : Camera::Parameters(KANNALA_BRANDT, _cameraName, _w, _h), m_k2_(_k2), m_k3_(_k3), m_k4_(_k4), m_k5_(_k5),
    m_mu_(_mu), m_mv_(_mv), m_u0_(_u0), m_v0_(_v0)
{
}

double & EquidistantCamera::Parameters::k2() { return m_k2_; }

double & EquidistantCamera::Parameters::k3() { return m_k3_; }

double & EquidistantCamera::Parameters::k4() { return m_k4_; }

double & EquidistantCamera::Parameters::k5() { return m_k5_; }

double & EquidistantCamera::Parameters::mu() { return m_mu_; }

double & EquidistantCamera::Parameters::mv() { return m_mv_; }

double & EquidistantCamera::Parameters::u0() { return m_u0_; }

double & EquidistantCamera::Parameters::v0() { return m_v0_; }

double EquidistantCamera::Parameters::k2() const { return m_k2_; }

double EquidistantCamera::Parameters::k3() const { return m_k3_; }

double EquidistantCamera::Parameters::k4() const { return m_k4_; }

double EquidistantCamera::Parameters::k5() const { return m_k5_; }

double EquidistantCamera::Parameters::mu() const { return m_mu_; }

double EquidistantCamera::Parameters::mv() const { return m_mv_; }

double EquidistantCamera::Parameters::u0() const { return m_u0_; }

double EquidistantCamera::Parameters::v0() const { return m_v0_; }

bool EquidistantCamera::Parameters::readFromYamlFile(const std::string & _filename)
{
  cv::FileStorage fs(_filename, cv::FileStorage::READ);

  if (!fs.isOpened()) {
    return false;
  }

  if (!fs["model_type"].isNone()) {
    std::string sModelType;
    fs["model_type"] >> sModelType;

    if (sModelType.compare("KANNALA_BRANDT") != 0) {
      return false;
    }
  }

  m_modelType_ = KANNALA_BRANDT;
  fs["camera_name"] >> m_cameraName_;
  m_imageWidth_ = static_cast<int>(fs["image_width"]);
  m_imageHeight_ = static_cast<int>(fs["image_height"]);

  cv::FileNode n = fs["projection_parameters"];
  m_k2_ = static_cast<double>(n["k2"]);
  m_k3_ = static_cast<double>(n["k3"]);
  m_k4_ = static_cast<double>(n["k4"]);
  m_k5_ = static_cast<double>(n["k5"]);
  m_mu_ = static_cast<double>(n["mu"]);
  m_mv_ = static_cast<double>(n["mv"]);
  m_u0_ = static_cast<double>(n["u0"]);
  m_v0_ = static_cast<double>(n["v0"]);

  return true;
}

void EquidistantCamera::Parameters::writeToYamlFile(const std::string & _filename) const
{
  cv::FileStorage fs(_filename, cv::FileStorage::WRITE);

  fs << "model_type" << "KANNALA_BRANDT";
  fs << "camera_name" << m_cameraName_;
  fs << "image_width" << m_imageWidth_;
  fs << "image_height" << m_imageHeight_;

  // projection: k2, k3, k4, k5, mu, mv, u0, v0
  fs << "projection_parameters";
  fs << "{" << "k2" << m_k2_ << "k3" << m_k3_ << "k4" << m_k4_ << "k5" << m_k5_ << "mu" << m_mu_ << "mv"
     << m_mv_ << "u0" << m_u0_ << "v0" << m_v0_ << "}";

  fs.release();
}

EquidistantCamera::Parameters &
EquidistantCamera::Parameters::operator=(const EquidistantCamera::Parameters & _other)
{
  if (this != &_other) {
    m_modelType_ = _other.m_modelType_;
    m_cameraName_ = _other.m_cameraName_;
    m_imageWidth_ = _other.m_imageWidth_;
    m_imageHeight_ = _other.m_imageHeight_;
    m_k2_ = _other.m_k2_;
    m_k3_ = _other.m_k3_;
    m_k4_ = _other.m_k4_;
    m_k5_ = _other.m_k5_;
    m_mu_ = _other.m_mu_;
    m_mv_ = _other.m_mv_;
    m_u0_ = _other.m_u0_;
    m_v0_ = _other.m_v0_;
  }

  return *this;
}

std::ostream & operator<<(std::ostream & _out, const EquidistantCamera::Parameters & _params)
{
  _out << "Camera Parameters:" << '\n';
  _out << "    model_type " << "KANNALA_BRANDT" << '\n';
  _out << "   camera_name " << _params.m_cameraName_ << '\n';
  _out << "   image_width " << _params.m_imageWidth_ << '\n';
  _out << "  image_height " << _params.m_imageHeight_ << '\n';

  // projection: k2, k3, k4, k5, mu, mv, u0, v0
  _out << "Projection Parameters" << '\n';
  _out << "            k2 " << _params.m_k2_ << '\n'
       << "            k3 " << _params.m_k3_ << '\n'
       << "            k4 " << _params.m_k4_ << '\n'
       << "            k5 " << _params.m_k5_ << '\n'
       << "            mu " << _params.m_mu_ << '\n'
       << "            mv " << _params.m_mv_ << '\n'
       << "            u0 " << _params.m_u0_ << '\n'
       << "            v0 " << _params.m_v0_ << '\n';

  return _out;
}

EquidistantCamera::EquidistantCamera() : m_inv_K11_(1.0), m_inv_K13_(0.0), m_inv_K22_(1.0), m_inv_K23_(0.0) {}

EquidistantCamera::EquidistantCamera(
  const std::string & _cameraName, int _imageWidth, int _imageHeight, double _k2, double _k3, double _k4,
  double _k5, double _mu, double _mv, double _u0, double _v0)
  : mParameters_(_cameraName, _imageWidth, _imageHeight, _k2, _k3, _k4, _k5, _mu, _mv, _u0, _v0)
{
  // Inverse camera projection matrix parameters
  m_inv_K11_ = 1.0 / mParameters_.mu();
  m_inv_K13_ = -mParameters_.u0() / mParameters_.mu();
  m_inv_K22_ = 1.0 / mParameters_.mv();
  m_inv_K23_ = -mParameters_.v0() / mParameters_.mv();
}

EquidistantCamera::EquidistantCamera(const EquidistantCamera::Parameters & _params) : mParameters_(_params)
{
  // Inverse camera projection matrix parameters
  m_inv_K11_ = 1.0 / mParameters_.mu();
  m_inv_K13_ = -mParameters_.u0() / mParameters_.mu();
  m_inv_K22_ = 1.0 / mParameters_.mv();
  m_inv_K23_ = -mParameters_.v0() / mParameters_.mv();
}

Camera::ModelType_e EquidistantCamera::modelType() const { return mParameters_.modelType(); }

const std::string & EquidistantCamera::cameraName() const { return mParameters_.cameraName(); }

int EquidistantCamera::imageWidth() const { return mParameters_.imageWidth(); }

int EquidistantCamera::imageHeight() const { return mParameters_.imageHeight(); }

void EquidistantCamera::estimateIntrinsics(
  const cv::Size & _boardSize, const std::vector<std::vector<cv::Point3f>> & _objectPoints,
  const std::vector<std::vector<cv::Point2f>> & _imagePoints)
{
  Parameters params = getParameters();

  double u0 = params.imageWidth() / 2.0;
  double v0 = params.imageHeight() / 2.0;

  double minReprojErr = std::numeric_limits<double>::max();

  std::vector<cv::Mat> rvecs, tvecs;
  rvecs.assign(_objectPoints.size(), cv::Mat());
  tvecs.assign(_objectPoints.size(), cv::Mat());

  params.k2() = 0.0;
  params.k3() = 0.0;
  params.k4() = 0.0;
  params.k5() = 0.0;
  params.u0() = u0;
  params.v0() = v0;

  // Initialize focal length
  // C. Hughes, P. Denny, M. Glavin, and E. Jones,
  // Equidistant Fish-Eye Calibration and Rectification by Vanishing Point
  // Extraction, PAMI 2010
  // Find circles from rows of chessboard corners, and for each pair
  // of circles, find vanishing points: v1 and v2.
  // f = ||v1 - v2|| / PI;
  double f0 = 0.0;
  for (size_t i = 0; i < _imagePoints.size(); ++i) {
    std::vector<Eigen::Vector2d> center(_boardSize.height);
    double radius[_boardSize.height];
    for (int r = 0; r < _boardSize.height; ++r) {
      std::vector<cv::Point2d> circle;
      circle.reserve(_boardSize.width);
      for (int c = 0; c < _boardSize.width; ++c) {
        circle.push_back(_imagePoints.at(i).at(r * _boardSize.width + c));
      }

      fitCircle(circle, center[r](0), center[r](1), radius[r]);
    }

    for (int j = 0; j < _boardSize.height; ++j) {
      for (int k = j + 1; k < _boardSize.height; ++k) {
        // find distance between pair of vanishing points which
        // correspond to intersection points of 2 circles
        std::vector<cv::Point2d> ipts;
        ipts = intersectCircles(center[j](0), center[j](1), radius[j], center[k](0), center[k](1), radius[k]);

        if (ipts.size() < 2) {
          continue;
        }

        double f = cv::norm(ipts.at(0) - ipts.at(1)) / M_PI;

        params.mu() = f;
        params.mv() = f;

        setParameters(params);

        for (size_t l = 0; l < _objectPoints.size(); ++l) {
          estimateExtrinsics(_objectPoints.at(l), _imagePoints.at(l), rvecs.at(l), tvecs.at(l));
        }

        double reprojErr = reprojectionError(_objectPoints, _imagePoints, rvecs, tvecs, cv::noArray());

        if (reprojErr < minReprojErr) {
          minReprojErr = reprojErr;
          f0 = f;
        }
      }
    }
  }

  if (f0 <= 0.0 && minReprojErr >= std::numeric_limits<double>::max()) {
    std::cout << "[" << params.cameraName() << "] " << "# INFO: kannala-Brandt model fails with given data. "
              << '\n';

    return;
  }

  params.mu() = f0;
  params.mv() = f0;

  setParameters(params);
}

/**
 * \brief Lifts a point from the image plane to the unit sphere
 *
 * \param p image coordinates
 * \param P coordinates of the point on the sphere
 */
void EquidistantCamera::liftSphere(const Eigen::Vector2d & _p, Eigen::Vector3d & _P) const
{
  liftProjective(_p, _P);
}

/** 
 * \brief Lifts a point from the image plane to its projective ray
 *
 * \param p image coordinates
 * \param P coordinates of the projective ray
 */
void EquidistantCamera::liftProjective(const Eigen::Vector2d & _p, Eigen::Vector3d & _P) const
{
  // Lift points to normalised plane
  Eigen::Vector2d p_u;
  p_u << m_inv_K11_ * _p(0) + m_inv_K13_, m_inv_K22_ * _p(1) + m_inv_K23_;

  // Obtain a projective ray
  double theta, phi;
  backprojectSymmetric(p_u, theta, phi);

  _P(0) = sin(theta) * cos(phi);
  _P(1) = sin(theta) * sin(phi);
  _P(2) = cos(theta);
}

/** 
 * \brief Project a 3D point (\a x,\a y,\a z) to the image plane in (\a u,\a v)
 *
 * \param P 3D point coordinates
 * \param p return value, contains the image point coordinates
 */
void EquidistantCamera::spaceToPlane(const Eigen::Vector3d & _P, Eigen::Vector2d & _p) const
{
  double theta = acos(_P(2) / _P.norm());
  double phi = atan2(_P(1), _P(0));

  Eigen::Vector2d p_u = r(mParameters_.k2(), mParameters_.k3(), mParameters_.k4(), mParameters_.k5(), theta) *
                        Eigen::Vector2d(cos(phi), sin(phi));

  // Apply generalised projection matrix
  _p << mParameters_.mu() * p_u(0) + mParameters_.u0(), mParameters_.mv() * p_u(1) + mParameters_.v0();
}


/** 
 * \brief Project a 3D point to the image plane and calculate Jacobian
 *
 * \param P 3D point coordinates
 * \param p return value, contains the image point coordinates
 */
void EquidistantCamera::spaceToPlane(
  const Eigen::Vector3d & _P, Eigen::Vector2d & _p, [[maybe_unused]] Eigen::Matrix<double, 2, 3> & _J) const
{
  double theta = acos(_P(2) / _P.norm());
  double phi = atan2(_P(1), _P(0));

  Eigen::Vector2d p_u = r(mParameters_.k2(), mParameters_.k3(), mParameters_.k4(), mParameters_.k5(), theta) *
                        Eigen::Vector2d(cos(phi), sin(phi));

  // Apply generalised projection matrix
  _p << mParameters_.mu() * p_u(0) + mParameters_.u0(), mParameters_.mv() * p_u(1) + mParameters_.v0();
}

/** 
 * \brief Projects an undistorted 2D point p_u to the image plane
 *
 * \param p_u 2D point coordinates
 * \return image point coordinates
 */
void EquidistantCamera::undistToPlane(
  [[maybe_unused]] const Eigen::Vector2d & _p_u, [[maybe_unused]] Eigen::Vector2d & _p) const
{
  //    Eigen::Vector2d p_d;
  //
  //    if (m_noDistortion)
  //    {
  //        p_d = p_u;
  //    }
  //    else
  //    {
  //        // Apply distortion
  //        Eigen::Vector2d d_u;
  //        distortion(p_u, d_u);
  //        p_d = p_u + d_u;
  //    }
  //
  //    // Apply generalised projection matrix
  //    p << mParameters.gamma1() * p_d(0) + mParameters.u0(),
  //         mParameters.gamma2() * p_d(1) + mParameters.v0();
}

void EquidistantCamera::initUndistortMap(cv::Mat & _map1, cv::Mat & _map2, double _fScale) const
{
  cv::Size imageSize(mParameters_.imageWidth(), mParameters_.imageHeight());

  cv::Mat mapX = cv::Mat::zeros(imageSize, CV_32F);
  cv::Mat mapY = cv::Mat::zeros(imageSize, CV_32F);

  for (int v = 0; v < imageSize.height; ++v) {
    for (int u = 0; u < imageSize.width; ++u) {
      double mx_u = m_inv_K11_ / _fScale * u + m_inv_K13_ / _fScale;
      double my_u = m_inv_K22_ / _fScale * v + m_inv_K23_ / _fScale;

      double theta, phi;
      backprojectSymmetric(Eigen::Vector2d(mx_u, my_u), theta, phi);

      Eigen::Vector3d P;
      P << sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta);

      Eigen::Vector2d p;
      spaceToPlane(P, p);

      mapX.at<float>(v, u) = p(0);
      mapY.at<float>(v, u) = p(1);
    }
  }

  cv::convertMaps(mapX, mapY, _map1, _map2, CV_32FC1, false);
}

cv::Mat EquidistantCamera::initUndistortRectifyMap(
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
    K_rect(0, 0) = mParameters_.mu();
    K_rect(1, 1) = mParameters_.mv();
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

int EquidistantCamera::parameterCount() const { return 8; }

const EquidistantCamera::Parameters & EquidistantCamera::getParameters() const { return mParameters_; }

void EquidistantCamera::setParameters(const EquidistantCamera::Parameters & _parameters)
{
  mParameters_ = _parameters;

  // Inverse camera projection matrix parameters
  m_inv_K11_ = 1.0 / mParameters_.mu();
  m_inv_K13_ = -mParameters_.u0() / mParameters_.mu();
  m_inv_K22_ = 1.0 / mParameters_.mv();
  m_inv_K23_ = -mParameters_.v0() / mParameters_.mv();
}

void EquidistantCamera::readParameters(const std::vector<double> & _parameterVec)
{
  if (_parameterVec.size() != parameterCount()) {
    return;
  }

  Parameters params = getParameters();

  params.k2() = _parameterVec.at(0);
  params.k3() = _parameterVec.at(1);
  params.k4() = _parameterVec.at(2);
  params.k5() = _parameterVec.at(3);
  params.mu() = _parameterVec.at(4);
  params.mv() = _parameterVec.at(5);
  params.u0() = _parameterVec.at(6);
  params.v0() = _parameterVec.at(7);

  setParameters(params);
}

void EquidistantCamera::writeParameters(std::vector<double> & _parameterVec) const
{
  _parameterVec.resize(parameterCount());
  _parameterVec.at(0) = mParameters_.k2();
  _parameterVec.at(1) = mParameters_.k3();
  _parameterVec.at(2) = mParameters_.k4();
  _parameterVec.at(3) = mParameters_.k5();
  _parameterVec.at(4) = mParameters_.mu();
  _parameterVec.at(5) = mParameters_.mv();
  _parameterVec.at(6) = mParameters_.u0();
  _parameterVec.at(7) = mParameters_.v0();
}

void EquidistantCamera::writeParametersToYamlFile(const std::string & _filename) const
{
  mParameters_.writeToYamlFile(_filename);
}

std::string EquidistantCamera::parametersToString() const
{
  std::ostringstream oss;
  oss << mParameters_;

  return oss.str();
}

void EquidistantCamera::fitOddPoly(
  const std::vector<double> & _x, const std::vector<double> & _y, int _n, std::vector<double> & _coeffs) const
{
  std::vector<int> pows;
  for (int i = 1; i <= _n; i += 2) {
    pows.push_back(i);
  }

  Eigen::MatrixXd X(_x.size(), pows.size());
  Eigen::MatrixXd Y(_y.size(), 1);
  for (size_t i = 0; i < _x.size(); ++i) {
    for (size_t j = 0; j < pows.size(); ++j) {
      X(i, j) = pow(_x.at(i), pows.at(j));
    }
    Y(i, 0) = _y.at(i);
  }

  Eigen::MatrixXd A = (X.transpose() * X).inverse() * X.transpose() * Y;

  _coeffs.resize(A.rows());
  for (int i = 0; i < A.rows(); ++i) {
    _coeffs.at(i) = A(i, 0);
  }
}

void EquidistantCamera::backprojectSymmetric(const Eigen::Vector2d & _p_u, double & _theta, double & _phi) const
{
  double tol = 1e-10;
  double p_u_norm = _p_u.norm();

  if (p_u_norm < 1e-10) {
    _phi = 0.0;
  } else {
    _phi = atan2(_p_u(1), _p_u(0));
  }

  int npow = 9;
  if (mParameters_.k5() == 0.0) {
    npow -= 2;
  }
  if (mParameters_.k4() == 0.0) {
    npow -= 2;
  }
  if (mParameters_.k3() == 0.0) {
    npow -= 2;
  }
  if (mParameters_.k2() == 0.0) {
    npow -= 2;
  }

  Eigen::MatrixXd coeffs(npow + 1, 1);
  coeffs.setZero();
  coeffs(0) = -p_u_norm;
  coeffs(1) = 1.0;

  if (npow >= 3) {
    coeffs(3) = mParameters_.k2();
  }
  if (npow >= 5) {
    coeffs(5) = mParameters_.k3();
  }
  if (npow >= 7) {
    coeffs(7) = mParameters_.k4();
  }
  if (npow >= 9) {
    coeffs(9) = mParameters_.k5();
  }

  if (npow == 1) {
    _theta = p_u_norm;
  } else {
    // Get eigenvalues of companion matrix corresponding to polynomial.
    // Eigenvalues correspond to roots of polynomial.
    Eigen::MatrixXd A(npow, npow);
    A.setZero();
    A.block(1, 0, npow - 1, npow - 1).setIdentity();
    A.col(npow - 1) = -coeffs.block(0, 0, npow, 1) / coeffs(npow);

    Eigen::EigenSolver<Eigen::MatrixXd> es(A);
    Eigen::MatrixXcd eigval = es.eigenvalues();

    std::vector<double> thetas;
    for (int i = 0; i < eigval.rows(); ++i) {
      if (fabs(eigval(i).imag()) > tol) {
        continue;
      }

      double t = eigval(i).real();

      if (t < -tol) {
        continue;
      } else if (t < 0.0) {
        t = 0.0;
      }

      thetas.push_back(t);
    }

    if (thetas.empty()) {
      _theta = p_u_norm;
    } else {
      _theta = *std::min_element(thetas.begin(), thetas.end());
    }
  }
}

} // namespace camodocal
