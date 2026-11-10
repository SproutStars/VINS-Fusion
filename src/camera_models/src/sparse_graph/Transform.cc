#include <camodocal/sparse_graph/Transform.h>

namespace camodocal
{

Transform::Transform()
{
  m_q_.setIdentity();
  m_t_.setZero();
}

Transform::Transform(const Eigen::Matrix4d & _H)
{
  m_q_ = Eigen::Quaterniond(_H.block<3, 3>(0, 0));
  m_t_ = _H.block<3, 1>(0, 3);
}

Eigen::Quaterniond & Transform::rotation() { return m_q_; }

const Eigen::Quaterniond & Transform::rotation() const { return m_q_; }

double * Transform::rotationData() { return m_q_.coeffs().data(); }

const double * const Transform::rotationData() const { return m_q_.coeffs().data(); }

Eigen::Vector3d & Transform::translation() { return m_t_; }

const Eigen::Vector3d & Transform::translation() const { return m_t_; }

double * Transform::translationData() { return m_t_.data(); }

const double * const Transform::translationData() const { return m_t_.data(); }

Eigen::Matrix4d Transform::toMatrix() const
{
  Eigen::Matrix4d H;
  H.setIdentity();
  H.block<3, 3>(0, 0) = m_q_.toRotationMatrix();
  H.block<3, 1>(0, 3) = m_t_;

  return H;
}

} // namespace camodocal
