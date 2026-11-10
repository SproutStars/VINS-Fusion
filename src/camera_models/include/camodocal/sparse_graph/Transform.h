#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <boost/shared_ptr.hpp>
#include <eigen3/Eigen/Dense>
#include <stdint.h>

namespace camodocal
{

class Transform {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Transform();
  Transform(const Eigen::Matrix4d & _H);

  Eigen::Quaterniond & rotation();
  const Eigen::Quaterniond & rotation() const;
  double * rotationData();
  const double * const rotationData() const;

  Eigen::Vector3d & translation();
  const Eigen::Vector3d & translation() const;
  double * translationData();
  const double * const translationData() const;

  Eigen::Matrix4d toMatrix() const;

private:
  Eigen::Quaterniond m_q_;
  Eigen::Vector3d m_t_;
};

} // namespace camodocal

#endif
