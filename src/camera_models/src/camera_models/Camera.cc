#include "camodocal/camera_models/Camera.h"
#include "camodocal/camera_models/ScaramuzzaCamera.h"

#include <opencv2/calib3d/calib3d.hpp>

namespace camodocal
{

Camera::Parameters::Parameters(ModelType_e _modelType)
  : m_modelType_(_modelType), m_imageWidth_(0), m_imageHeight_(0)
{
  switch (_modelType) {
  case KANNALA_BRANDT:
    m_nIntrinsics_ = 8;
    break;
  case PINHOLE:
    m_nIntrinsics_ = 8;
    break;
  case SCARAMUZZA:
    m_nIntrinsics_ = SCARAMUZZA_CAMERA_NUM_PARAMS;
    break;
  case MEI:
  default:
    m_nIntrinsics_ = 9;
  }
}

Camera::Parameters::Parameters(ModelType_e _modelType, const std::string & _cameraName, int _w, int _h)
  : m_modelType_(_modelType), m_cameraName_(_cameraName), m_imageWidth_(_w), m_imageHeight_(_h)
{
  switch (_modelType) {
  case KANNALA_BRANDT:
    m_nIntrinsics_ = 8;
    break;
  case PINHOLE:
    m_nIntrinsics_ = 8;
    break;
  case SCARAMUZZA:
    m_nIntrinsics_ = SCARAMUZZA_CAMERA_NUM_PARAMS;
    break;
  case MEI:
  default:
    m_nIntrinsics_ = 9;
  }
}

Camera::ModelType_e & Camera::Parameters::modelType() { return m_modelType_; }

std::string & Camera::Parameters::cameraName() { return m_cameraName_; }

int & Camera::Parameters::imageWidth() { return m_imageWidth_; }

int & Camera::Parameters::imageHeight() { return m_imageHeight_; }

Camera::ModelType_e Camera::Parameters::modelType() const { return m_modelType_; }

const std::string & Camera::Parameters::cameraName() const { return m_cameraName_; }

int Camera::Parameters::imageWidth() const { return m_imageWidth_; }

int Camera::Parameters::imageHeight() const { return m_imageHeight_; }

int Camera::Parameters::nIntrinsics() const { return m_nIntrinsics_; }

cv::Mat & Camera::mask() { return m_mask_; }

const cv::Mat & Camera::mask() const { return m_mask_; }

void Camera::estimateExtrinsics(
  const std::vector<cv::Point3f> & _objectPoints, const std::vector<cv::Point2f> & _imagePoints,
  cv::Mat & _rvec, cv::Mat & _tvec) const
{
  std::vector<cv::Point2f> Ms(_imagePoints.size());
  for (size_t i = 0; i < Ms.size(); ++i) {
    Eigen::Vector3d P;
    liftProjective(Eigen::Vector2d(_imagePoints.at(i).x, _imagePoints.at(i).y), P);

    P /= P(2);

    Ms.at(i).x = P(0);
    Ms.at(i).y = P(1);
  }

  // assume unit focal length, zero principal point, and zero distortion
  cv::solvePnP(_objectPoints, Ms, cv::Mat::eye(3, 3, CV_64F), cv::noArray(), _rvec, _tvec);
}

double Camera::reprojectionDist(const Eigen::Vector3d & _P1, const Eigen::Vector3d & _P2) const
{
  Eigen::Vector2d p1, p2;

  spaceToPlane(_P1, p1);
  spaceToPlane(_P2, p2);

  return (p1 - p2).norm();
}

double Camera::reprojectionError(
  const std::vector<std::vector<cv::Point3f>> & _objectPoints,
  const std::vector<std::vector<cv::Point2f>> & _imagePoints, const std::vector<cv::Mat> & _rvecs,
  const std::vector<cv::Mat> & _tvecs, cv::OutputArray _perViewErrors) const
{
  int imageCount = _objectPoints.size();
  size_t pointsSoFar = 0;
  double totalErr = 0.0;

  bool computePerViewErrors = _perViewErrors.needed();
  cv::Mat perViewErrors;
  if (computePerViewErrors) {
    _perViewErrors.create(imageCount, 1, CV_64F);
    perViewErrors = _perViewErrors.getMat();
  }

  for (int i = 0; i < imageCount; ++i) {
    size_t pointCount = _imagePoints.at(i).size();

    pointsSoFar += pointCount;

    std::vector<cv::Point2f> estImagePoints;
    projectPoints(_objectPoints.at(i), _rvecs.at(i), _tvecs.at(i), estImagePoints);

    double err = 0.0;
    for (size_t j = 0; j < _imagePoints.at(i).size(); ++j) {
      err += cv::norm(_imagePoints.at(i).at(j) - estImagePoints.at(j));
    }

    if (computePerViewErrors) {
      perViewErrors.at<double>(i) = err / pointCount;
    }

    totalErr += err;
  }

  return totalErr / pointsSoFar;
}

double Camera::reprojectionError(
  const Eigen::Vector3d & _P, const Eigen::Quaterniond & _camera_q, const Eigen::Vector3d & _camera_t,
  const Eigen::Vector2d & _observed_p) const
{
  Eigen::Vector3d P_cam = _camera_q.toRotationMatrix() * _P + _camera_t;

  Eigen::Vector2d p;
  spaceToPlane(P_cam, p);

  return (p - _observed_p).norm();
}

void Camera::projectPoints(
  const std::vector<cv::Point3f> & _objectPoints, const cv::Mat & _rvec, const cv::Mat & _tvec,
  std::vector<cv::Point2f> & _imagePoints) const
{
  // project 3D object points to the image plane
  _imagePoints.reserve(_objectPoints.size());

  //double
  cv::Mat R0;
  cv::Rodrigues(_rvec, R0);

  Eigen::MatrixXd R(3, 3);
  R << R0.at<double>(0, 0), R0.at<double>(0, 1), R0.at<double>(0, 2), R0.at<double>(1, 0),
    R0.at<double>(1, 1), R0.at<double>(1, 2), R0.at<double>(2, 0), R0.at<double>(2, 1), R0.at<double>(2, 2);

  Eigen::Vector3d t;
  t << _tvec.at<double>(0), _tvec.at<double>(1), _tvec.at<double>(2);

  for (auto objectPoint : _objectPoints) {
    // Rotate and translate
    Eigen::Vector3d P;
    P << objectPoint.x, objectPoint.y, objectPoint.z;

    P = R * P + t;

    Eigen::Vector2d p;
    spaceToPlane(P, p);

    _imagePoints.push_back(cv::Point2f(p(0), p(1)));
  }
}

} // namespace camodocal
