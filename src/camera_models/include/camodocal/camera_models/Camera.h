#ifndef CAMERA_H
#define CAMERA_H

#include <boost/shared_ptr.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <vector>

namespace camodocal
{

class Camera {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // Keep the existing enum representation for interface compatibility.
  // NOLINTNEXTLINE(performance-enum-size)
  enum ModelType_e : unsigned int { KANNALA_BRANDT, MEI, PINHOLE, PINHOLE_FULL, SCARAMUZZA };

  class Parameters {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Parameters(ModelType_e _modelType);

    Parameters(ModelType_e _modelType, const std::string & _cameraName, int _w, int _h);

    ModelType_e & modelType();
    std::string & cameraName();
    int & imageWidth();
    int & imageHeight();

    ModelType_e modelType() const;
    const std::string & cameraName() const;
    int imageWidth() const;
    int imageHeight() const;

    int nIntrinsics() const;

    virtual bool readFromYamlFile(const std::string & _filename) = 0;
    virtual void writeToYamlFile(const std::string & _filename) const = 0;

  protected:
    ModelType_e m_modelType_;
    int m_nIntrinsics_;
    std::string m_cameraName_;
    int m_imageWidth_;
    int m_imageHeight_;
  };

  virtual ModelType_e modelType() const = 0;
  virtual const std::string & cameraName() const = 0;
  virtual int imageWidth() const = 0;
  virtual int imageHeight() const = 0;

  virtual cv::Mat & mask();
  virtual const cv::Mat & mask() const;

  virtual void estimateIntrinsics(
    const cv::Size & _boardSize, const std::vector<std::vector<cv::Point3f>> & _objectPoints,
    const std::vector<std::vector<cv::Point2f>> & _imagePoints) = 0;
  virtual void estimateExtrinsics(
    const std::vector<cv::Point3f> & _objectPoints, const std::vector<cv::Point2f> & _imagePoints,
    cv::Mat & _rvec, cv::Mat & _tvec) const;

  // Lift points from the image plane to the sphere
  virtual void liftSphere(const Eigen::Vector2d & _p, Eigen::Vector3d & _P) const = 0;
  //%output P

  // Lift points from the image plane to the projective space
  virtual void liftProjective(const Eigen::Vector2d & _p, Eigen::Vector3d & _P) const = 0;
  //%output P

  // Projects 3D points to the image plane (Pi function)
  virtual void spaceToPlane(const Eigen::Vector3d & _P, Eigen::Vector2d & _p) const = 0;
  //%output p

  // Projects 3D points to the image plane (Pi function)
  // and calculates jacobian
  // virtual void spaceToPlane(const Eigen::Vector3d& P, Eigen::Vector2d& p,
  //                          Eigen::Matrix<double,2,3>& J) const = 0;
  //%output p
  //%output J

  virtual void undistToPlane(const Eigen::Vector2d & _p_u, Eigen::Vector2d & _p) const = 0;
  //%output p

  // virtual void initUndistortMap(cv::Mat& map1, cv::Mat& map2, double fScale = 1.0)
  // const = 0;
  virtual cv::Mat initUndistortRectifyMap(
    cv::Mat & _map1, cv::Mat & _map2, float _fx = -1.0f, float _fy = -1.0f,
    cv::Size _imageSize = cv::Size(0, 0), float _cx = -1.0f, float _cy = -1.0f,
    cv::Mat _rmat = cv::Mat::eye(3, 3, CV_32F)) const = 0;

  virtual int parameterCount() const = 0;

  virtual void readParameters(const std::vector<double> & _parameters) = 0;
  virtual void writeParameters(std::vector<double> & _parameters) const = 0;

  virtual void writeParametersToYamlFile(const std::string & _filename) const = 0;

  virtual std::string parametersToString() const = 0;

  /**
     * \brief Calculates the reprojection distance between points
     *
     * \param P1 first 3D point coordinates
     * \param P2 second 3D point coordinates
     * \return euclidean distance in the plane
     */
  double reprojectionDist(const Eigen::Vector3d & _P1, const Eigen::Vector3d & _P2) const;

  double reprojectionError(
    const std::vector<std::vector<cv::Point3f>> & _objectPoints,
    const std::vector<std::vector<cv::Point2f>> & _imagePoints, const std::vector<cv::Mat> & _rvecs,
    const std::vector<cv::Mat> & _tvecs, cv::OutputArray _perViewErrors = cv::noArray()) const;

  double reprojectionError(
    const Eigen::Vector3d & _P, const Eigen::Quaterniond & _camera_q, const Eigen::Vector3d & _camera_t,
    const Eigen::Vector2d & _observed_p) const;

  void projectPoints(
    const std::vector<cv::Point3f> & _objectPoints, const cv::Mat & _rvec, const cv::Mat & _tvec,
    std::vector<cv::Point2f> & _imagePoints) const;

protected:
  cv::Mat m_mask_;
};

typedef boost::shared_ptr<Camera> CameraPtr;
typedef boost::shared_ptr<const Camera> CameraConstPtr;
} // namespace camodocal

#endif
