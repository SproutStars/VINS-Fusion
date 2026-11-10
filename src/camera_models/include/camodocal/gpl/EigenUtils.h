#ifndef EIGENUTILS_H
#define EIGENUTILS_H

#include <eigen3/Eigen/Dense>

#include "camodocal/gpl/gpl.h"
#include "ceres/rotation.h"

namespace camodocal
{

// Returns the 3D cross product skew symmetric matrix of a given 3D vector
template <typename T> Eigen::Matrix<T, 3, 3> skew(const Eigen::Matrix<T, 3, 1> & _vec)
{
  return (Eigen::Matrix<T, 3, 3>() << T(0), -_vec(2), _vec(1), _vec(2), T(0), -_vec(0), -_vec(1), _vec(0), T(0))
    .finished();
}

template <typename Derived>
typename Eigen::MatrixBase<Derived>::PlainObject sqrtm(const Eigen::MatrixBase<Derived> & _A)
{
  Eigen::SelfAdjointEigenSolver<typename Derived::PlainObject> es(_A);

  return es.operatorSqrt();
}

template <typename T> Eigen::Matrix<T, 3, 3> AngleAxisToRotationMatrix(const Eigen::Matrix<T, 3, 1> & _rvec)
{
  T angle = _rvec.norm();
  if (angle == T(0)) {
    return Eigen::Matrix<T, 3, 3>::Identity();
  }

  Eigen::Matrix<T, 3, 1> axis;
  axis = _rvec.normalized();

  Eigen::Matrix<T, 3, 3> rmat;
  rmat = Eigen::AngleAxis<T>(angle, axis);

  return rmat;
}

template <typename T> Eigen::Quaternion<T> AngleAxisToQuaternion(const Eigen::Matrix<T, 3, 1> & _rvec)
{
  Eigen::Matrix<T, 3, 3> rmat = AngleAxisToRotationMatrix<T>(_rvec);

  return Eigen::Quaternion<T>(rmat);
}

template <typename T> void AngleAxisToQuaternion(const Eigen::Matrix<T, 3, 1> & _rvec, T * _q)
{
  Eigen::Quaternion<T> quat = AngleAxisToQuaternion<T>(_rvec);

  _q[0] = quat.x();
  _q[1] = quat.y();
  _q[2] = quat.z();
  _q[3] = quat.w();
}

template <typename T> Eigen::Matrix<T, 3, 1> RotationToAngleAxis(const Eigen::Matrix<T, 3, 3> & _rmat)
{
  Eigen::AngleAxis<T> angleaxis;
  angleaxis.fromRotationMatrix(_rmat);
  return angleaxis.angle() * angleaxis.axis();
}

template <typename T> void QuaternionToAngleAxis(const T * const _q, Eigen::Matrix<T, 3, 1> & _rvec)
{
  Eigen::Quaternion<T> quat(_q[3], _q[0], _q[1], _q[2]);

  Eigen::Matrix<T, 3, 3> rmat = quat.toRotationMatrix();

  Eigen::AngleAxis<T> angleaxis;
  angleaxis.fromRotationMatrix(rmat);

  _rvec = angleaxis.angle() * angleaxis.axis();
}

template <typename T> Eigen::Matrix<T, 3, 3> QuaternionToRotation(const T * const _q)
{
  T R[9];
  ceres::QuaternionToRotation(_q, R);

  Eigen::Matrix<T, 3, 3> rmat;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      rmat(i, j) = R[i * 3 + j];
    }
  }

  return rmat;
}

template <typename T> void QuaternionToRotation(const T * const _q, T * _rot)
{
  ceres::QuaternionToRotation(_q, _rot);
}

template <typename T> Eigen::Matrix<T, 4, 4> QuaternionMultMatLeft(const Eigen::Quaternion<T> & _q)
{
  return (Eigen::Matrix<T, 4, 4>() << _q.w(), -_q.z(), _q.y(), _q.x(), _q.z(), _q.w(), -_q.x(), _q.y(),
          -_q.y(), _q.x(), _q.w(), _q.z(), -_q.x(), -_q.y(), -_q.z(), _q.w())
    .finished();
}

template <typename T> Eigen::Matrix<T, 4, 4> QuaternionMultMatRight(const Eigen::Quaternion<T> & _q)
{
  return (Eigen::Matrix<T, 4, 4>() << _q.w(), _q.z(), -_q.y(), _q.x(), -_q.z(), _q.w(), _q.x(), _q.y(),
          _q.y(), -_q.x(), _q.w(), _q.z(), -_q.x(), -_q.y(), -_q.z(), _q.w())
    .finished();
}

/// @param theta - rotation about screw axis
/// @param d - projection of tvec on the rotation axis
/// @param l - screw axis direction
/// @param m - screw axis moment
template <typename T>
void AngleAxisAndTranslationToScrew(
  const Eigen::Matrix<T, 3, 1> & _rvec, const Eigen::Matrix<T, 3, 1> & _tvec, T & _theta, T & _d,
  Eigen::Matrix<T, 3, 1> & _l, Eigen::Matrix<T, 3, 1> & _m)
{

  _theta = _rvec.norm();
  if (_theta == 0) {
    _l.setZero();
    _m.setZero();
    std::cout << "Warning: Undefined screw! Returned 0. " << '\n';
  }

  _l = _rvec.normalized();

  Eigen::Matrix<T, 3, 1> t = _tvec;

  _d = t.transpose() * _l;

  // point on screw axis - projection of origin on screw axis
  Eigen::Matrix<T, 3, 1> c;
  c = 0.5 * (t - _d * _l + (1.0 / tan(_theta / 2.0) * _l).cross(t));

  // c and hence the screw axis is not defined if theta is either 0 or M_PI
  _m = c.cross(_l);
}

template <typename T> Eigen::Matrix<T, 3, 3> RPY2mat(T _roll, T _pitch, T _yaw)
{
  Eigen::Matrix<T, 3, 3> m;

  T cr = cos(_roll);
  T sr = sin(_roll);
  T cp = cos(_pitch);
  T sp = sin(_pitch);
  T cy = cos(_yaw);
  T sy = sin(_yaw);

  m(0, 0) = cy * cp;
  m(0, 1) = cy * sp * sr - sy * cr;
  m(0, 2) = cy * sp * cr + sy * sr;
  m(1, 0) = sy * cp;
  m(1, 1) = sy * sp * sr + cy * cr;
  m(1, 2) = sy * sp * cr - cy * sr;
  m(2, 0) = -sp;
  m(2, 1) = cp * sr;
  m(2, 2) = cp * cr;
  return m;
}

template <typename T> void mat2RPY(const Eigen::Matrix<T, 3, 3> & _m, T & _roll, T & _pitch, T & _yaw)
{
  _roll = atan2(_m(2, 1), _m(2, 2));
  _pitch = atan2(-_m(2, 0), sqrt(_m(2, 1) * _m(2, 1) + _m(2, 2) * _m(2, 2)));
  _yaw = atan2(_m(1, 0), _m(0, 0));
}

template <typename T>
Eigen::Matrix<T, 4, 4> homogeneousTransform(const Eigen::Matrix<T, 3, 3> & _R, const Eigen::Matrix<T, 3, 1> & _t)
{
  Eigen::Matrix<T, 4, 4> H;
  H.setIdentity();

  H.block(0, 0, 3, 3) = _R;
  H.block(0, 3, 3, 1) = _t;

  return H;
}

template <typename T>
Eigen::Matrix<T, 4, 4> poseWithCartesianTranslation(const T * const _q, const T * const _p)
{
  Eigen::Matrix<T, 4, 4> pose = Eigen::Matrix<T, 4, 4>::Identity();

  T rotation[9];
  ceres::QuaternionToRotation(_q, rotation);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      pose(i, j) = rotation[i * 3 + j];
    }
  }

  pose(0, 3) = _p[0];
  pose(1, 3) = _p[1];
  pose(2, 3) = _p[2];

  return pose;
}

template <typename T>
Eigen::Matrix<T, 4, 4>
poseWithSphericalTranslation(const T * const _q, const T * const _p, const T _scale = T(1.0))
{
  Eigen::Matrix<T, 4, 4> pose = Eigen::Matrix<T, 4, 4>::Identity();

  T rotation[9];
  ceres::QuaternionToRotation(_q, rotation);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      pose(i, j) = rotation[i * 3 + j];
    }
  }

  T theta = _p[0];
  T phi = _p[1];
  pose(0, 3) = sin(theta) * cos(phi) * _scale;
  pose(1, 3) = sin(theta) * sin(phi) * _scale;
  pose(2, 3) = cos(theta) * _scale;

  return pose;
}

// Returns the Sampson error of a given essential matrix and 2 image points
template <typename T>
T sampsonError(
  const Eigen::Matrix<T, 3, 3> & _E, const Eigen::Matrix<T, 3, 1> & _p1, const Eigen::Matrix<T, 3, 1> & _p2)
{
  Eigen::Matrix<T, 3, 1> Ex1 = _E * _p1;
  Eigen::Matrix<T, 3, 1> Etx2 = _E.transpose() * _p2;

  T x2tEx1 = _p2.dot(Ex1);

  // compute Sampson error
  T err = square(x2tEx1) / (square(Ex1(0, 0)) + square(Ex1(1, 0)) + square(Etx2(0, 0)) + square(Etx2(1, 0)));

  return err;
}

// Returns the Sampson error of a given rotation/translation and 2 image points
template <typename T>
T sampsonError(
  const Eigen::Matrix<T, 3, 3> & _R, const Eigen::Matrix<T, 3, 1> & _t, const Eigen::Matrix<T, 3, 1> & _p1,
  const Eigen::Matrix<T, 3, 1> & _p2)
{
  // construct essential matrix
  Eigen::Matrix<T, 3, 3> E = skew(_t) * _R;

  Eigen::Matrix<T, 3, 1> Ex1 = E * _p1;
  Eigen::Matrix<T, 3, 1> Etx2 = E.transpose() * _p2;

  T x2tEx1 = _p2.dot(Ex1);

  // compute Sampson error
  T err = square(x2tEx1) / (square(Ex1(0, 0)) + square(Ex1(1, 0)) + square(Etx2(0, 0)) + square(Etx2(1, 0)));

  return err;
}

// Returns the Sampson error of a given rotation/translation and 2 image points
template <typename T>
T sampsonError(
  const Eigen::Matrix<T, 4, 4> & _H, const Eigen::Matrix<T, 3, 1> & _p1, const Eigen::Matrix<T, 3, 1> & _p2)
{
  Eigen::Matrix<T, 3, 3> R = _H.block(0, 0, 3, 3);
  Eigen::Matrix<T, 3, 1> t = _H.block(0, 3, 3, 1);

  return sampsonError(R, t, _p1, _p2);
}

template <typename T>
Eigen::Matrix<T, 3, 1> transformPoint(const Eigen::Matrix<T, 4, 4> & _H, const Eigen::Matrix<T, 3, 1> & _P)
{
  Eigen::Matrix<T, 3, 1> P_trans = _H.block(0, 0, 3, 3) * _P + _H.block(0, 3, 3, 1);

  return P_trans;
}

template <typename T>
Eigen::Matrix<T, 4, 4> estimate3DRigidTransform(
  const std::vector<Eigen::Matrix<T, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, 3, 1>>> & _points1,
  const std::vector<Eigen::Matrix<T, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, 3, 1>>> & _points2)
{
  // compute centroids
  Eigen::Matrix<T, 3, 1> c1, c2;
  c1.setZero();
  c2.setZero();

  for (size_t i = 0; i < _points1.size(); ++i) {
    c1 += _points1.at(i);
    c2 += _points2.at(i);
  }

  c1 /= _points1.size();
  c2 /= _points1.size();

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> X(3, _points1.size());
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Y(3, _points1.size());
  for (size_t i = 0; i < _points1.size(); ++i) {
    X.col(i) = _points1.at(i) - c1;
    Y.col(i) = _points2.at(i) - c2;
  }

  Eigen::Matrix<T, 3, 3> H = X * Y.transpose();

  Eigen::JacobiSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd(
    H, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::Matrix<T, 3, 3> U = svd.matrixU();
  Eigen::Matrix<T, 3, 3> V = svd.matrixV();
  if (U.determinant() * V.determinant() < 0.0) {
    V.col(2) *= -1.0;
  }

  Eigen::Matrix<T, 3, 3> R = V * U.transpose();
  Eigen::Matrix<T, 3, 1> t = c2 - R * c1;

  return homogeneousTransform(R, t);
}

template <typename T>
Eigen::Matrix<T, 4, 4> estimate3DRigidSimilarityTransform(
  const std::vector<Eigen::Matrix<T, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, 3, 1>>> & _points1,
  const std::vector<Eigen::Matrix<T, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, 3, 1>>> & _points2)
{
  // compute centroids
  Eigen::Matrix<T, 3, 1> c1, c2;
  c1.setZero();
  c2.setZero();

  for (size_t i = 0; i < _points1.size(); ++i) {
    c1 += _points1.at(i);
    c2 += _points2.at(i);
  }

  c1 /= _points1.size();
  c2 /= _points1.size();

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> X(3, _points1.size());
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Y(3, _points1.size());
  for (size_t i = 0; i < _points1.size(); ++i) {
    X.col(i) = _points1.at(i) - c1;
    Y.col(i) = _points2.at(i) - c2;
  }

  Eigen::Matrix<T, 3, 3> H = X * Y.transpose();

  Eigen::JacobiSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd(
    H, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::Matrix<T, 3, 3> U = svd.matrixU();
  Eigen::Matrix<T, 3, 3> V = svd.matrixV();
  if (U.determinant() * V.determinant() < 0.0) {
    V.col(2) *= -1.0;
  }

  Eigen::Matrix<T, 3, 3> R = V * U.transpose();

  std::vector<Eigen::Matrix<T, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, 3, 1>>> rotatedPoints1(
    _points1.size());
  for (size_t i = 0; i < _points1.size(); ++i) {
    rotatedPoints1.at(i) = R * (_points1.at(i) - c1);
  }

  double sum_ss = 0.0, sum_tt = 0.0;
  for (size_t i = 0; i < _points1.size(); ++i) {
    sum_ss += (_points1.at(i) - c1).squaredNorm();
    sum_tt += (_points2.at(i) - c2).dot(rotatedPoints1.at(i));
  }

  double scale = sum_tt / sum_ss;

  Eigen::Matrix<T, 3, 3> sR = scale * R;
  Eigen::Matrix<T, 3, 1> t = c2 - sR * c1;

  return homogeneousTransform(sR, t);
}

} // namespace camodocal

#endif
