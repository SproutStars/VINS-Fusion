#include "camodocal/calib/CameraCalibration.h"

#include <algorithm>
#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CostFunctionFactory.h"
#include "camodocal/gpl/EigenQuaternionParameterization.h"
#include "camodocal/gpl/EigenUtils.h"
#include "camodocal/sparse_graph/Transform.h"

#include "ceres/ceres.h"
namespace camodocal
{

CameraCalibration::CameraCalibration() : m_boardSize_(cv::Size(0, 0)), m_squareSize_(0.0f), m_verbose_(false)
{
}

CameraCalibration::CameraCalibration(
  const Camera::ModelType_e _modelType, const std::string & _cameraName, const cv::Size & _imageSize,
  const cv::Size & _boardSize, float _squareSize)
  : m_boardSize_(_boardSize), m_squareSize_(_squareSize), m_verbose_(false)
{
  m_camera_ = CameraFactory::instance()->generateCamera(_modelType, _cameraName, _imageSize);
}

void CameraCalibration::clear()
{
  m_imagePoints_.clear();
  m_scenePoints_.clear();
}

void CameraCalibration::addChessboardData(const std::vector<cv::Point2f> & _corners)
{
  m_imagePoints_.push_back(_corners);

  std::vector<cv::Point3f> scenePointsInView;
  for (int i = 0; i < m_boardSize_.height; ++i) {
    for (int j = 0; j < m_boardSize_.width; ++j) {
      scenePointsInView.push_back(cv::Point3f(i * m_squareSize_, j * m_squareSize_, 0.0));
    }
  }
  m_scenePoints_.push_back(scenePointsInView);
}

bool CameraCalibration::calibrate()
{
  int imageCount = m_imagePoints_.size();

  // compute intrinsic camera parameters and extrinsic parameters for each of the views
  std::vector<cv::Mat> rvecs;
  std::vector<cv::Mat> tvecs;
  bool ret = calibrateHelper(m_camera_, rvecs, tvecs);

  m_cameraPoses_ = cv::Mat(imageCount, 6, CV_64F);
  for (int i = 0; i < imageCount; ++i) {
    m_cameraPoses_.at<double>(i, 0) = rvecs.at(i).at<double>(0);
    m_cameraPoses_.at<double>(i, 1) = rvecs.at(i).at<double>(1);
    m_cameraPoses_.at<double>(i, 2) = rvecs.at(i).at<double>(2);
    m_cameraPoses_.at<double>(i, 3) = tvecs.at(i).at<double>(0);
    m_cameraPoses_.at<double>(i, 4) = tvecs.at(i).at<double>(1);
    m_cameraPoses_.at<double>(i, 5) = tvecs.at(i).at<double>(2);
  }

  // Compute measurement covariance.
  std::vector<std::vector<cv::Point2f>> errVec(m_imagePoints_.size());
  Eigen::Vector2d errSum = Eigen::Vector2d::Zero();
  size_t errCount = 0;
  for (size_t i = 0; i < m_imagePoints_.size(); ++i) {
    std::vector<cv::Point2f> estImagePoints;
    m_camera_->projectPoints(m_scenePoints_.at(i), rvecs.at(i), tvecs.at(i), estImagePoints);

    for (size_t j = 0; j < m_imagePoints_.at(i).size(); ++j) {
      cv::Point2f pObs = m_imagePoints_.at(i).at(j);
      cv::Point2f pEst = estImagePoints.at(j);

      cv::Point2f err = pObs - pEst;

      errVec.at(i).push_back(err);

      errSum += Eigen::Vector2d(err.x, err.y);
    }

    errCount += m_imagePoints_.at(i).size();
  }

  Eigen::Vector2d errMean = errSum / static_cast<double>(errCount);

  Eigen::Matrix2d measurementCovariance = Eigen::Matrix2d::Zero();
  for (auto & i : errVec) {
    for (auto err : i) {
      double d0 = err.x - errMean(0);
      double d1 = err.y - errMean(1);

      measurementCovariance(0, 0) += d0 * d0;
      measurementCovariance(0, 1) += d0 * d1;
      measurementCovariance(1, 1) += d1 * d1;
    }
  }
  measurementCovariance /= static_cast<double>(errCount);
  measurementCovariance(1, 0) = measurementCovariance(0, 1);

  m_measurementCovariance_ = measurementCovariance;

  return ret;
}

int CameraCalibration::sampleCount() const { return m_imagePoints_.size(); }

std::vector<std::vector<cv::Point2f>> & CameraCalibration::imagePoints() { return m_imagePoints_; }

const std::vector<std::vector<cv::Point2f>> & CameraCalibration::imagePoints() const
{
  return m_imagePoints_;
}

std::vector<std::vector<cv::Point3f>> & CameraCalibration::scenePoints() { return m_scenePoints_; }

const std::vector<std::vector<cv::Point3f>> & CameraCalibration::scenePoints() const
{
  return m_scenePoints_;
}

CameraPtr & CameraCalibration::camera() { return m_camera_; }

const CameraConstPtr CameraCalibration::camera() const { return m_camera_; }

Eigen::Matrix2d & CameraCalibration::measurementCovariance() { return m_measurementCovariance_; }

const Eigen::Matrix2d & CameraCalibration::measurementCovariance() const { return m_measurementCovariance_; }

cv::Mat & CameraCalibration::cameraPoses() { return m_cameraPoses_; }

const cv::Mat & CameraCalibration::cameraPoses() const { return m_cameraPoses_; }

void CameraCalibration::drawResults(std::vector<cv::Mat> & _images) const
{
  std::vector<cv::Mat> rvecs, tvecs;

  for (size_t i = 0; i < _images.size(); ++i) {
    cv::Mat rvec(3, 1, CV_64F);
    rvec.at<double>(0) = m_cameraPoses_.at<double>(i, 0);
    rvec.at<double>(1) = m_cameraPoses_.at<double>(i, 1);
    rvec.at<double>(2) = m_cameraPoses_.at<double>(i, 2);

    cv::Mat tvec(3, 1, CV_64F);
    tvec.at<double>(0) = m_cameraPoses_.at<double>(i, 3);
    tvec.at<double>(1) = m_cameraPoses_.at<double>(i, 4);
    tvec.at<double>(2) = m_cameraPoses_.at<double>(i, 5);

    rvecs.push_back(rvec);
    tvecs.push_back(tvec);
  }

  int drawShiftBits = 4;
  int drawMultiplier = 1 << drawShiftBits;

  cv::Scalar green(0, 255, 0);
  cv::Scalar red(0, 0, 255);

  for (size_t i = 0; i < _images.size(); ++i) {
    cv::Mat & image = _images.at(i);
    if (image.channels() == 1) {
      cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
    }

    std::vector<cv::Point2f> estImagePoints;
    m_camera_->projectPoints(m_scenePoints_.at(i), rvecs.at(i), tvecs.at(i), estImagePoints);

    float errorSum = 0.0f;
    float errorMax = std::numeric_limits<float>::min();

    for (size_t j = 0; j < m_imagePoints_.at(i).size(); ++j) {
      cv::Point2f pObs = m_imagePoints_.at(i).at(j);
      cv::Point2f pEst = estImagePoints.at(j);

      cv::circle(
        image, cv::Point(cvRound(pObs.x * drawMultiplier), cvRound(pObs.y * drawMultiplier)), 5, green, 2,
        cv::LINE_AA, drawShiftBits);

      cv::circle(
        image, cv::Point(cvRound(pEst.x * drawMultiplier), cvRound(pEst.y * drawMultiplier)), 5, red, 2,
        cv::LINE_AA, drawShiftBits);

      float error = cv::norm(pObs - pEst);

      errorSum += error;
      if (error > errorMax) {
        errorMax = error;
      }
    }

    std::ostringstream oss;
    oss << "Reprojection error: avg = " << errorSum / m_imagePoints_.at(i).size() << "   max = " << errorMax;

    cv::putText(
      image, oss.str(), cv::Point(10, image.rows - 10), cv::FONT_HERSHEY_COMPLEX, 0.5,
      cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  }
}

void CameraCalibration::writeParams(const std::string & _filename) const
{
  m_camera_->writeParametersToYamlFile(_filename);
}

bool CameraCalibration::writeChessboardData(const std::string & _filename) const
{
  std::ofstream ofs(_filename.c_str(), std::ios::out | std::ios::binary);
  if (!ofs.is_open()) {
    return false;
  }

  writeData(ofs, m_boardSize_.width);
  writeData(ofs, m_boardSize_.height);
  writeData(ofs, m_squareSize_);

  writeData(ofs, m_measurementCovariance_(0, 0));
  writeData(ofs, m_measurementCovariance_(0, 1));
  writeData(ofs, m_measurementCovariance_(1, 0));
  writeData(ofs, m_measurementCovariance_(1, 1));

  writeData(ofs, m_cameraPoses_.rows);
  writeData(ofs, m_cameraPoses_.cols);
  writeData(ofs, m_cameraPoses_.type());
  for (int i = 0; i < m_cameraPoses_.rows; ++i) {
    for (int j = 0; j < m_cameraPoses_.cols; ++j) {
      writeData(ofs, m_cameraPoses_.at<double>(i, j));
    }
  }

  writeData(ofs, m_imagePoints_.size());
  for (const auto & m_imagePoint : m_imagePoints_) {
    writeData(ofs, m_imagePoint.size());
    for (auto ipt : m_imagePoint) {
      writeData(ofs, ipt.x);
      writeData(ofs, ipt.y);
    }
  }

  writeData(ofs, m_scenePoints_.size());
  for (const auto & m_scenePoint : m_scenePoints_) {
    writeData(ofs, m_scenePoint.size());
    for (auto spt : m_scenePoint) {
      writeData(ofs, spt.x);
      writeData(ofs, spt.y);
      writeData(ofs, spt.z);
    }
  }

  return true;
}

bool CameraCalibration::readChessboardData(const std::string & _filename)
{
  std::ifstream ifs(_filename.c_str(), std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    return false;
  }

  readData(ifs, m_boardSize_.width);
  readData(ifs, m_boardSize_.height);
  readData(ifs, m_squareSize_);

  readData(ifs, m_measurementCovariance_(0, 0));
  readData(ifs, m_measurementCovariance_(0, 1));
  readData(ifs, m_measurementCovariance_(1, 0));
  readData(ifs, m_measurementCovariance_(1, 1));

  int rows, cols, type;
  readData(ifs, rows);
  readData(ifs, cols);
  readData(ifs, type);
  m_cameraPoses_ = cv::Mat(rows, cols, type);

  for (int i = 0; i < m_cameraPoses_.rows; ++i) {
    for (int j = 0; j < m_cameraPoses_.cols; ++j) {
      readData(ifs, m_cameraPoses_.at<double>(i, j));
    }
  }

  size_t nImagePointSets;
  readData(ifs, nImagePointSets);

  m_imagePoints_.clear();
  m_imagePoints_.resize(nImagePointSets);
  for (auto & m_imagePoint : m_imagePoints_) {
    size_t nImagePoints;
    readData(ifs, nImagePoints);
    m_imagePoint.resize(nImagePoints);

    for (auto & ipt : m_imagePoint) {
      readData(ifs, ipt.x);
      readData(ifs, ipt.y);
    }
  }

  size_t nScenePointSets;
  readData(ifs, nScenePointSets);

  m_scenePoints_.clear();
  m_scenePoints_.resize(nScenePointSets);
  for (auto & m_scenePoint : m_scenePoints_) {
    size_t nScenePoints;
    readData(ifs, nScenePoints);
    m_scenePoint.resize(nScenePoints);

    for (auto & spt : m_scenePoint) {
      readData(ifs, spt.x);
      readData(ifs, spt.y);
      readData(ifs, spt.z);
    }
  }

  return true;
}

void CameraCalibration::setVerbose(bool _verbose) { m_verbose_ = _verbose; }

bool CameraCalibration::calibrateHelper(
  CameraPtr & _camera, std::vector<cv::Mat> & _rvecs, std::vector<cv::Mat> & _tvecs) const
{
  _rvecs.assign(m_scenePoints_.size(), cv::Mat());
  _tvecs.assign(m_scenePoints_.size(), cv::Mat());

  // STEP 1: Estimate intrinsics
  _camera->estimateIntrinsics(m_boardSize_, m_scenePoints_, m_imagePoints_);

  // STEP 2: Estimate extrinsics
  for (size_t i = 0; i < m_scenePoints_.size(); ++i) {
    _camera->estimateExtrinsics(m_scenePoints_.at(i), m_imagePoints_.at(i), _rvecs.at(i), _tvecs.at(i));
  }

  if (m_verbose_) {
    std::cout << "[" << _camera->cameraName() << "] "
              << "# INFO: " << "Initial reprojection error: " << std::fixed << std::setprecision(3)
              << _camera->reprojectionError(m_scenePoints_, m_imagePoints_, _rvecs, _tvecs) << " pixels"
              << '\n';
  }

  // STEP 3: optimization using ceres
  optimize(_camera, _rvecs, _tvecs);

  if (m_verbose_) {
    double err = _camera->reprojectionError(m_scenePoints_, m_imagePoints_, _rvecs, _tvecs);
    std::cout << "[" << _camera->cameraName() << "] " << "# INFO: Final reprojection error: " << err
              << " pixels" << '\n';
    std::cout << "[" << _camera->cameraName() << "] " << "# INFO: " << _camera->parametersToString() << '\n';
  }

  return true;
}

void CameraCalibration::optimize(
  CameraPtr & _camera, std::vector<cv::Mat> & _rvecs, std::vector<cv::Mat> & _tvecs) const
{
  // Use ceres to do optimization
  ceres::Problem problem;

  std::vector<Transform, Eigen::aligned_allocator<Transform>> transformVec(_rvecs.size());
  for (size_t i = 0; i < _rvecs.size(); ++i) {
    Eigen::Vector3d rvec;
    cv::cv2eigen(_rvecs.at(i), rvec);

    transformVec.at(i).rotation() = Eigen::AngleAxisd(rvec.norm(), rvec.normalized());
    transformVec.at(i).translation() << _tvecs[i].at<double>(0), _tvecs[i].at<double>(1),
      _tvecs[i].at<double>(2);
  }

  std::vector<double> intrinsicCameraParams;
  m_camera_->writeParameters(intrinsicCameraParams);

  // create residuals for each observation
  for (size_t i = 0; i < m_imagePoints_.size(); ++i) {
    for (size_t j = 0; j < m_imagePoints_.at(i).size(); ++j) {
      const cv::Point3f & spt = m_scenePoints_.at(i).at(j);
      const cv::Point2f & ipt = m_imagePoints_.at(i).at(j);

      ceres::CostFunction * costFunction = CostFunctionFactory::instance()->generateCostFunction(
        _camera, Eigen::Vector3d(spt.x, spt.y, spt.z), Eigen::Vector2d(ipt.x, ipt.y),
        CAMERA_INTRINSICS | CAMERA_POSE);

      ceres::LossFunction * lossFunction = new ceres::CauchyLoss(1.0);
      problem.AddResidualBlock(
        costFunction, lossFunction, intrinsicCameraParams.data(), transformVec.at(i).rotationData(),
        transformVec.at(i).translationData());
    }

    ceres::Manifold * quaternionParameterization = new EigenQuaternionParameterization;

    problem.SetManifold(transformVec.at(i).rotationData(), quaternionParameterization);
  }

  std::cout << "begin ceres" << '\n';
  ceres::Solver::Options options;
  options.max_num_iterations = 1000;
  options.num_threads = 1;

  if (m_verbose_) {
    options.minimizer_progress_to_stdout = true;
  }

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << "end ceres" << '\n';

  if (m_verbose_) {
    std::cout << summary.FullReport() << '\n';
  }

  _camera->readParameters(intrinsicCameraParams);

  for (size_t i = 0; i < _rvecs.size(); ++i) {
    Eigen::AngleAxisd aa(transformVec.at(i).rotation());

    Eigen::Vector3d rvec = aa.angle() * aa.axis();
    cv::eigen2cv(rvec, _rvecs.at(i));

    cv::Mat & tvec = _tvecs.at(i);
    tvec.at<double>(0) = transformVec.at(i).translation()(0);
    tvec.at<double>(1) = transformVec.at(i).translation()(1);
    tvec.at<double>(2) = transformVec.at(i).translation()(2);
  }
}

template <typename T> void CameraCalibration::readData(std::ifstream & _ifs, T & _data) const
{
  char * buffer = new char[sizeof(T)];

  _ifs.read(buffer, sizeof(T));

  _data = *(reinterpret_cast<T *>(buffer));

  delete[] buffer;
}

template <typename T> void CameraCalibration::writeData(std::ofstream & _ofs, T _data) const
{
  char * pData = reinterpret_cast<char *>(&_data);

  _ofs.write(pData, sizeof(T));
}

} // namespace camodocal
