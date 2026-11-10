#ifndef CHESSBOARD_H
#define CHESSBOARD_H

#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>

namespace camodocal
{

// forward declarations
class ChessboardCorner;
typedef boost::shared_ptr<ChessboardCorner> ChessboardCornerPtr;
class ChessboardQuad;
typedef boost::shared_ptr<ChessboardQuad> ChessboardQuadPtr;

class Chessboard {
public:
  Chessboard(cv::Size _boardSize, cv::Mat & _image);

  void findCorners(bool _useOpenCV = false);
  const std::vector<cv::Point2f> & getCorners() const;
  bool cornersFound() const;

  const cv::Mat & getImage() const;
  const cv::Mat & getSketch() const;

private:
  bool findChessboardCorners(
    const cv::Mat & _image, const cv::Size & _patternSize, std::vector<cv::Point2f> & _corners, int _flags,
    bool _useOpenCV);

  bool findChessboardCornersImproved(
    const cv::Mat & _image, const cv::Size & _patternSize, std::vector<cv::Point2f> & _corners, int _flags);

  void cleanFoundConnectedQuads(std::vector<ChessboardQuadPtr> & _quadGroup, cv::Size _patternSize);

  void findConnectedQuads(
    std::vector<ChessboardQuadPtr> & _quads, std::vector<ChessboardQuadPtr> & _group, int _group_idx,
    int _dilation);

  //    int checkQuadGroup(std::vector<ChessboardQuadPtr>& quadGroup,
  //                       std::vector<ChessboardCornerPtr>& outCorners,
  //                       cv::Size patternSize);

  void labelQuadGroup(std::vector<ChessboardQuadPtr> & _quad_group, cv::Size _patternSize, bool _firstRun);

  void findQuadNeighbors(std::vector<ChessboardQuadPtr> & _quads, int _dilation);

  int augmentBestRun(
    std::vector<ChessboardQuadPtr> & _candidateQuads, int _candidateDilation,
    std::vector<ChessboardQuadPtr> & _existingQuads, int _existingDilation);

  void generateQuads(
    std::vector<ChessboardQuadPtr> & _quads, cv::Mat & _image, int _flags, int _dilation, bool _firstRun);

  bool checkQuadGroup(
    std::vector<ChessboardQuadPtr> & _quads, std::vector<ChessboardCornerPtr> & _corners,
    cv::Size _patternSize) const;

  void getQuadrangleHypotheses(
    const std::vector<std::vector<cv::Point>> & _contours, std::vector<std::pair<float, int>> & _quads,
    int _classId) const;

  bool checkChessboard(const cv::Mat & _image, cv::Size _patternSize) const;

  bool checkBoardMonotony(std::vector<ChessboardCornerPtr> & _corners, cv::Size _patternSize);

  bool matchCorners(ChessboardQuadPtr & _quad1, int _corner1, ChessboardQuadPtr & _quad2, int _corner2) const;

  cv::Mat mImage_;
  cv::Mat mSketch_;
  std::vector<cv::Point2f> mCorners_;
  cv::Size mBoardSize_;
  bool mCornersFound_;
};

} // namespace camodocal

#endif
