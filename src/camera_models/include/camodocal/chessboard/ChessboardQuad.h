#ifndef CHESSBOARDQUAD_H
#define CHESSBOARDQUAD_H

#include <boost/shared_ptr.hpp>

#include "camodocal/chessboard/ChessboardCorner.h"

namespace camodocal
{

class ChessboardQuad;
typedef boost::shared_ptr<ChessboardQuad> ChessboardQuadPtr;

class ChessboardQuad {
public:
  ChessboardQuad() : count_(0), group_idx_(-1), edge_len_(FLT_MAX), labeled_(false) {}

  int count_;                      // Number of quad neighbors
  int group_idx_;                  // Quad group ID
  float edge_len_;                 // Smallest side length^2
  ChessboardCornerPtr corners_[4]; // Coordinates of quad corners
  ChessboardQuadPtr neighbors_[4]; // Pointers of quad neighbors
  bool labeled_;                   // Has this corner been labeled?
};

} // namespace camodocal

#endif
