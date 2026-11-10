#pragma once

#include <cmath>
#include <opencv2/core.hpp>

namespace vins
{
namespace tracking
{
// Shi-Tomasi response at a FAST candidate, matching cornerMinEigenVal's
// 3x3 Sobel / 3x3 block normalization without a full-frame float response map.
inline float fastCornerMinEigenValue(const cv::Mat & _image, const cv::Point2f & _point)
{
  CV_Assert(_image.type() == CV_8UC1);
  if (
    !std::isfinite(_point.x) || !std::isfinite(_point.y) || _point.x < 2 || _point.y < 2 ||
    _point.x >= _image.cols - 2 || _point.y >= _image.rows - 2)
    return 0.0f;
  const int x = cvRound(_point.x);
  const int y = cvRound(_point.y);
  if (x < 2 || y < 2 || x >= _image.cols - 2 || y >= _image.rows - 2)
    return 0.0f;
  double xx = 0.0, xy = 0.0, yy = 0.0;
  for (int dy = -1; dy <= 1; ++dy) {
    const auto * top = _image.ptr<uchar>(y + dy - 1);
    const auto * middle = _image.ptr<uchar>(y + dy);
    const auto * bottom = _image.ptr<uchar>(y + dy + 1);
    for (int dx = -1; dx <= 1; ++dx) {
      const int u = x + dx;
      const double gx =
        top[u + 1] - top[u - 1] + 2 * (middle[u + 1] - middle[u - 1]) + bottom[u + 1] - bottom[u - 1];
      const double gy = bottom[u - 1] - top[u - 1] + 2 * (bottom[u] - top[u]) + bottom[u + 1] - top[u + 1];
      xx += gx * gx;
      xy += gx * gy;
      yy += gy * gy;
    }
  }
  constexpr double K_SCALE = 1.0 / (12.0 * 255.0 * 12.0 * 255.0);
  return static_cast<float>(0.5 * (xx + yy - std::sqrt((xx - yy) * (xx - yy) + 4.0 * xy * xy)) * K_SCALE);
}
}  // namespace tracking
}  // namespace vins
