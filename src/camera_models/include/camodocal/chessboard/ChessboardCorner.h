#ifndef CHESSBOARDCORNER_H
#define CHESSBOARDCORNER_H

#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>

namespace camodocal
{

class ChessboardCorner;
typedef boost::shared_ptr<ChessboardCorner> ChessboardCornerPtr;

class ChessboardCorner {
public:
  ChessboardCorner() : row_(0), column_(0), needsNeighbor_(true), count_(0) {}

  float meanDist(int & _n) const
  {
    float sum = 0;
    _n = 0;
    for (const auto & neighbor : neighbors_) {
      if (neighbor.get()) {
        float dx = neighbor->pt_.x - pt_.x;
        float dy = neighbor->pt_.y - pt_.y;
        sum += sqrt(static_cast<double>(dx * dx + dy * dy));
        _n++;
      }
    }
    return sum / std::max(_n, 1);
  }

  cv::Point2f pt_;                   // X and y coordinates
  int row_;                          // Row and column of the corner
  int column_;                       // in the found pattern
  bool needsNeighbor_;               // Does the corner require a neighbor?
  int count_;                        // number of corner neighbors
  ChessboardCornerPtr neighbors_[4]; // pointer to all corner neighbors
};

} // namespace camodocal

#endif
