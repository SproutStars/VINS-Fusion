#ifndef CAMERACALIBRATION_H
#define CAMERACALIBRATION_H

#include <opencv2/core/core.hpp>

#include "camodocal/camera_models/Camera.h"

namespace camodocal
{

class CameraCalibration {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CameraCalibration();

  CameraCalibration(
    Camera::ModelType_e _modelType, const std::string & _cameraName, const cv::Size & _imageSize,
    const cv::Size & _boardSize, float _squareSize);

  void clear();

  void addChessboardData(const std::vector<cv::Point2f> & _corners);

  bool calibrate();

  int sampleCount() const;
  std::vector<std::vector<cv::Point2f>> & imagePoints();
  const std::vector<std::vector<cv::Point2f>> & imagePoints() const;
  std::vector<std::vector<cv::Point3f>> & scenePoints();
  const std::vector<std::vector<cv::Point3f>> & scenePoints() const;
  CameraPtr & camera();
  const CameraConstPtr camera() const;

  Eigen::Matrix2d & measurementCovariance();
  const Eigen::Matrix2d & measurementCovariance() const;

  cv::Mat & cameraPoses();
  const cv::Mat & cameraPoses() const;

  void drawResults(std::vector<cv::Mat> & _images) const;

  void writeParams(const std::string & _filename) const;

  bool writeChessboardData(const std::string & _filename) const;
  bool readChessboardData(const std::string & _filename);

  void setVerbose(bool _verbose);

private:
  bool calibrateHelper(CameraPtr & _camera, std::vector<cv::Mat> & _rvecs, std::vector<cv::Mat> & _tvecs) const;

  void optimize(CameraPtr & _camera, std::vector<cv::Mat> & _rvecs, std::vector<cv::Mat> & _tvecs) const;

  template <typename T> void readData(std::ifstream & _ifs, T & _data) const;

  template <typename T> void writeData(std::ofstream & _ofs, T _data) const;

  cv::Size m_boardSize_;
  float m_squareSize_;

  CameraPtr m_camera_;
  cv::Mat m_cameraPoses_;

  std::vector<std::vector<cv::Point2f>> m_imagePoints_;
  std::vector<std::vector<cv::Point3f>> m_scenePoints_;

  Eigen::Matrix2d m_measurementCovariance_;

  bool m_verbose_;
};

} // namespace camodocal

#endif
