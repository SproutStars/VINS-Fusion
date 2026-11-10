#ifndef GPL_H
#define GPL_H

#include <algorithm>
#include <cmath>
#include <opencv2/core/core.hpp>

namespace camodocal
{

template <class T> const T clamp(const T & _v, const T & _a, const T & _b)
{
  return std::min(_b, std::max(_a, _v));
}

double hypot3(double _x, double _y, double _z);
float hypot3f(float _x, float _y, float _z);

template <class T> const T normalizeTheta(const T & _theta)
{
  T normTheta = _theta;

  while (normTheta < -M_PI) {
    normTheta += 2.0 * M_PI;
  }
  while (normTheta > M_PI) {
    normTheta -= 2.0 * M_PI;
  }

  return normTheta;
}

double d2r(double _deg);
float d2r(float _deg);
double r2d(double _rad);
float r2d(float _rad);

double sinc(double _theta);

template <class T> const T square(const T & _x) { return _x * _x; }

template <class T> const T cube(const T & _x) { return _x * _x * _x; }

template <class T> const T random(const T & _a, const T & _b)
{
  return static_cast<double>(rand()) / RAND_MAX * (_b - _a) + _a;
}

template <class T> const T randomNormal(const T & _sigma)
{
  T x1, x2, w;

  do {
    x1 = 2.0 * random(0.0, 1.0) - 1.0;
    x2 = 2.0 * random(0.0, 1.0) - 1.0;
    w = x1 * x1 + x2 * x2;
  } while (w >= 1.0 || w == 0.0);

  w = sqrt((-2.0 * log(w)) / w);

  return x1 * w * _sigma;
}

unsigned long long timeInMicroseconds();

double timeInSeconds();

void colorDepthImage(cv::Mat & _imgDepth, cv::Mat & _imgColoredDepth, float _minRange, float _maxRange);

bool colormap(const std::string & _name, unsigned char _idx, float & _r, float & _g, float & _b);

std::vector<cv::Point2i> bresLine(int _x0, int _y0, int _x1, int _y1);
std::vector<cv::Point2i> bresCircle(int _x0, int _y0, int _r);

void fitCircle(const std::vector<cv::Point2d> & _points, double & _centerX, double & _centerY, double & _radius);

std::vector<cv::Point2d> intersectCircles(double _x1, double _y1, double _r1, double _x2, double _y2, double _r2);

void LLtoUTM(
  double _latitude, double _longitude, double & _utmNorthing, double & _utmEasting, std::string & _utmZone);
void UTMtoLL(
  double _utmNorthing, double _utmEasting, const std::string & _utmZone, double & _latitude,
  double & _longitude);

long int timestampDiff(uint64_t _t1, uint64_t _t2);

} // namespace camodocal

#endif
