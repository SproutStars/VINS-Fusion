#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <utility>

#include "featureTracker/fast_grid_quality.h"
#include "featureTracker/feature_tracker.h"

struct FeatureTrackerTestAccess {
  static void setPreviousLiftedPoints(FeatureTracker & _tracker, std::vector<cv::Point3d> _points)
  {
    _tracker.prev_lifted_pts_ = std::move(_points);
  }

  static const std::vector<cv::Point3d> & previousLiftedPoints(const FeatureTracker & _tracker)
  {
    return _tracker.prev_lifted_pts_;
  }

  static const std::vector<uchar> & flowStatus(const FeatureTracker & _tracker)
  {
    return _tracker.flow_status_scratch_;
  }

  static const std::vector<float> & flowError(const FeatureTracker & _tracker)
  {
    return _tracker.flow_err_scratch_;
  }

  static const std::vector<uchar> & reverseStatus(const FeatureTracker & _tracker)
  {
    return _tracker.reverse_status_scratch_;
  }

  static const std::vector<cv::Point2f> & reversePoints(const FeatureTracker & _tracker)
  {
    return _tracker.reverse_pts_scratch_;
  }

  static const std::vector<cv::Point2f> & ransacCurrentPoints(const FeatureTracker & _tracker)
  {
    return _tracker.ransac_cur_pts_scratch_;
  }

  static const std::vector<cv::Point2f> & ransacPreviousPoints(const FeatureTracker & _tracker)
  {
    return _tracker.ransac_prev_pts_scratch_;
  }

  static const std::vector<uchar> & ransacStatus(const FeatureTracker & _tracker)
  {
    return _tracker.ransac_status_scratch_;
  }
};

namespace
{
// buildOpticalFlowPyramid(withDerivatives=true) interleaves an image and a
// derivative buffer per logical level, i.e. 2 * (maxLevel + 1) Mats. The GFTT
// path intentionally caches only the image levels.
constexpr size_t K_KLT_DERIVATIVE_PYRAMID_MATS = 2u * (3u + 1u);
constexpr size_t K_KLT_IMAGE_PYRAMID_MATS = 3u + 1u;

struct TrackerParameterGuard_s {
  TrackerParameterGuard_s()
    : max_cnt_(MAX_CNT), min_dist_(MIN_DIST), feature_detector_(FEATURE_DETECTOR),
      fast_threshold_(FAST_THRESHOLD), grid_x_(GRID_X), grid_y_(GRID_Y),
      detection_min_feature_ratio_(DETECTION_MIN_FEATURE_RATIO), flow_back_(FLOW_BACK),
      show_track_(SHOW_TRACK), ransac_outlier_rejection_(RANSAC_OUTLIER_REJECTION), f_threshold_(F_THRESHOLD)
  {
  }

  ~TrackerParameterGuard_s()
  {
    MAX_CNT = max_cnt_;
    MIN_DIST = min_dist_;
    FEATURE_DETECTOR = feature_detector_;
    FAST_THRESHOLD = fast_threshold_;
    GRID_X = grid_x_;
    GRID_Y = grid_y_;
    DETECTION_MIN_FEATURE_RATIO = detection_min_feature_ratio_;
    FLOW_BACK = flow_back_;
    SHOW_TRACK = show_track_;
    RANSAC_OUTLIER_REJECTION = ransac_outlier_rejection_;
    F_THRESHOLD = f_threshold_;
  }

  int max_cnt_;
  int min_dist_;
  std::string feature_detector_;
  int fast_threshold_;
  int grid_x_;
  int grid_y_;
  double detection_min_feature_ratio_;
  int flow_back_;
  int show_track_;
  int ransac_outlier_rejection_;
  double f_threshold_;
};

cv::Mat makeCornerTexture(int _rows = 240, int _cols = 320)
{
  cv::Mat image(_rows, _cols, CV_8UC1);
  for (int y = 0; y < _rows; ++y)
    for (int x = 0; x < _cols; ++x)
      image.at<uchar>(y, x) = ((x / 12 + y / 12) & 1) ? 255 : 0;
  cv::GaussianBlur(image, image, cv::Size(3, 3), 0.5);
  return image;
}

void makeTwoViewCorrespondences(std::vector<cv::Point2f> & _previous, std::vector<cv::Point2f> & _current)
{
  std::vector<cv::Point3f> object_points;
  object_points.reserve(32);
  for (int i = 0; i < 32; ++i) {
    object_points.emplace_back((i % 8 - 3.5f) * 0.25f, (i / 8 - 1.5f) * 0.30f, 3.0f + 0.10f * (i % 5));
  }
  const cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 460.0, 0.0, 320.0, 0.0, 460.0, 240.0, 0.0, 0.0, 1.0);
  const cv::Mat distortion = cv::Mat::zeros(1, 5, CV_64F);
  const cv::Mat first_rvec = cv::Mat::zeros(3, 1, CV_64F);
  const cv::Mat first_tvec = cv::Mat::zeros(3, 1, CV_64F);
  const cv::Mat second_rvec = (cv::Mat_<double>(3, 1) << 0.08, -0.04, 0.03);
  const cv::Mat second_tvec = (cv::Mat_<double>(3, 1) << 0.35, 0.05, 0.10);
  cv::projectPoints(object_points, first_rvec, first_tvec, camera_matrix, distortion, _previous);
  cv::projectPoints(object_points, second_rvec, second_tvec, camera_matrix, distortion, _current);
}

void configureRansacState(FeatureTracker & _tracker, bool _cache_previous_lift)
{
  _tracker.row_ = 480;
  _tracker.col_ = 640;
  _tracker.m_camera_.emplace_back(
    new camodocal::PinholeCamera("left", 640, 480, 0, 0, 0, 0, 460, 460, 320, 240));

  makeTwoViewCorrespondences(_tracker.prev_pts_, _tracker.cur_pts_);
  constexpr size_t outlier_index = 9;
  _tracker.cur_pts_[outlier_index].x += 80.0f;
  _tracker.cur_pts_[outlier_index].y -= 55.0f;

  const int base_id = 1000;
  for (size_t i = 0; i < _tracker.cur_pts_.size(); ++i) {
    _tracker.ids_.push_back(base_id + static_cast<int>(i));
    // Non-monotonic lifetimes force setMask() to reorder correspondences.
    _tracker.track_cnt_.push_back(static_cast<int>((i * 7) % 13) + 1);
  }

  std::vector<cv::Point3d> lifted;
  if (_cache_previous_lift) {
    lifted.reserve(_tracker.prev_pts_.size());
    for (const cv::Point2f & point : _tracker.prev_pts_) {
      Eigen::Vector3d lifted_point;
      _tracker.m_camera_[0]->liftProjective(Eigen::Vector2d(point.x, point.y), lifted_point);
      lifted.emplace_back(lifted_point.x(), lifted_point.y(), lifted_point.z());
    }
  }
  FeatureTrackerTestAccess::setPreviousLiftedPoints(_tracker, std::move(lifted));
}

class TrackerBuffers : public ::testing::TestWithParam<bool> {
protected:
  void SetUp() override
  {
    MAX_CNT = 100;
    MIN_DIST = 12;
    FLOW_BACK = 1;
    SHOW_TRACK = 1;
    FEATURE_DETECTOR = "gftt";
    FAST_THRESHOLD = 20;
    GRID_X = 5;
    GRID_Y = 5;
    DETECTION_MIN_FEATURE_RATIO = 0.0;
    cv::setNumThreads(1);
  }
};
}  // namespace

TEST(FeatureTrackerOptions, RansacOutlierRejectionDefaultsOffAndGuardRestoresSetting)
{
  EXPECT_EQ(RANSAC_OUTLIER_REJECTION, 0);

  {
    TrackerParameterGuard_s guard;
    RANSAC_OUTLIER_REJECTION = 1;
    EXPECT_EQ(RANSAC_OUTLIER_REJECTION, 1);
  }

  EXPECT_EQ(RANSAC_OUTLIER_REJECTION, 0);
}

TEST(FeatureTrackerOptions, DetectionIntervalClampsToOneAndDefaultsToLegacyEveryFrame)
{
  FeatureTracker tracker;
  EXPECT_EQ(tracker.detectionInterval(), 1);
  tracker.setDetectionInterval(0);
  EXPECT_EQ(tracker.detectionInterval(), 1);
  tracker.setDetectionInterval(-4);
  EXPECT_EQ(tracker.detectionInterval(), 1);
  tracker.setDetectionInterval(3);
  EXPECT_EQ(tracker.detectionInterval(), 3);
}

TEST(FeatureTrackerDiagnostics, ReportsStageTimingsAndUpdatesOnNonExtractingFrames)
{
  TrackerParameterGuard_s guard;
  MAX_CNT = 80;
  MIN_DIST = 8;
  FLOW_BACK = 1;
  SHOW_TRACK = 0;
  FEATURE_DETECTOR = "gftt";
  DETECTION_MIN_FEATURE_RATIO = 0.0;
  cv::setNumThreads(1);

  FeatureTracker tracker;
  tracker.m_camera_.emplace_back(new camodocal::PinholeCamera("left", 320, 240, 0, 0, 0, 0, 200, 200, 160, 120));
  const cv::Mat first = makeCornerTexture();
  const auto observations = tracker.trackImage(1.0, first);
  EXPECT_FALSE(observations.empty());

  const FeatureTrackerTimings_s first_timings = tracker.lastTimings();
  EXPECT_GT(first_timings.total_ms_, 0.0);
  EXPECT_GT(first_timings.pyramid_ms_, 0.0);
  EXPECT_GE(first_timings.flow_ms_, 0.0);
  EXPECT_GT(first_timings.detect_ms_, 0.0);
  EXPECT_GE(
    first_timings.total_ms_, first_timings.pyramid_ms_ + first_timings.flow_ms_ + first_timings.detect_ms_);
  EXPECT_EQ(first_timings.tracked_features_, 0);
  EXPECT_GT(first_timings.detected_features_, 0);

  cv::Mat second;
  const cv::Mat shift = (cv::Mat_<double>(2, 3) << 1, 0, 0.5, 0, 1, -0.25);
  cv::warpAffine(first, second, shift, first.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
  EXPECT_TRUE(tracker.trackImage(2.0, second, cv::Mat(), false, false).empty());

  const FeatureTrackerTimings_s second_timings = tracker.lastTimings();
  EXPECT_GT(second_timings.total_ms_, 0.0);
  EXPECT_GT(second_timings.pyramid_ms_, 0.0);
  EXPECT_GT(second_timings.flow_ms_, 0.0);
  EXPECT_GT(second_timings.tracked_features_, 0);
  EXPECT_GE(
    second_timings.total_ms_, second_timings.pyramid_ms_ + second_timings.flow_ms_ + second_timings.detect_ms_);
  EXPECT_GE(second_timings.detected_features_, 0);
}

TEST(FeatureTrackerDetectionGate, DefaultIntervalStillTopsUpEveryDeficitFrame)
{
  TrackerParameterGuard_s guard;
  MAX_CNT = 100;
  MIN_DIST = 8;
  FLOW_BACK = 1;
  SHOW_TRACK = 0;
  FEATURE_DETECTOR = "gftt";
  DETECTION_MIN_FEATURE_RATIO = 0.0;
  cv::setNumThreads(1);

  FeatureTracker tracker;
  tracker.m_camera_.emplace_back(new camodocal::PinholeCamera("left", 320, 240, 0, 0, 0, 0, 200, 200, 160, 120));
  const cv::Mat image = makeCornerTexture();
  tracker.trackImage(1.0, image);
  ASSERT_GT(tracker.prev_pts_.size(), 60u);

  // Recreate a routine deficit below the urgency threshold. The legacy
  // interval (1) must still run the detector on this frame.
  tracker.prev_pts_.resize(60);
  tracker.ids_.resize(60);
  tracker.track_cnt_.resize(60);
  tracker.trackImage(2.0, image);
  EXPECT_GT(tracker.lastTimings().detected_features_, 0);
}

TEST(FeatureTrackerDetectionGate, IntervalSkipsRoutineTopUpButUrgentDeficitBypasses)
{
  TrackerParameterGuard_s guard;
  MAX_CNT = 100;
  MIN_DIST = 8;
  FLOW_BACK = 1;
  SHOW_TRACK = 0;
  FEATURE_DETECTOR = "gftt";
  DETECTION_MIN_FEATURE_RATIO = 0.0;
  cv::setNumThreads(1);

  FeatureTracker gated;
  gated.m_camera_.emplace_back(new camodocal::PinholeCamera("left", 320, 240, 0, 0, 0, 0, 200, 200, 160, 120));
  gated.setDetectionInterval(2);
  const cv::Mat image = makeCornerTexture();
  gated.trackImage(1.0, image);
  ASSERT_GT(gated.prev_pts_.size(), 60u);

  gated.prev_pts_.resize(60);
  gated.ids_.resize(60);
  gated.track_cnt_.resize(60);
  gated.trackImage(2.0, image);
  EXPECT_EQ(gated.lastTimings().detected_features_, 0);
  EXPECT_LE(gated.prev_pts_.size(), 61u);
  gated.trackImage(3.0, image);
  EXPECT_GT(gated.lastTimings().detected_features_, 0);

  FeatureTracker urgent;
  urgent.m_camera_.emplace_back(new camodocal::PinholeCamera("left", 320, 240, 0, 0, 0, 0, 200, 200, 160, 120));
  urgent.setDetectionInterval(100);
  urgent.trackImage(1.0, image);
  ASSERT_GT(urgent.prev_pts_.size(), 30u);

  urgent.prev_pts_.resize(30);
  urgent.ids_.resize(30);
  urgent.track_cnt_.resize(30);
  urgent.trackImage(2.0, image);
  EXPECT_GT(urgent.lastTimings().detected_features_, 0);
}

TEST_P(TrackerBuffers, PreservesVelocityAcrossGatedFramesAndPrediction)
{
  FeatureTracker tracker;
  tracker.m_camera_.emplace_back(new camodocal::PinholeCamera("left", 320, 240, 0, 0, 0, 0, 200, 200, 160, 120));
  const bool stereo = GetParam();
  tracker.stereo_cam_ = stereo;
  if (stereo)
    tracker.m_camera_.push_back(tracker.m_camera_.front());
  cv::Mat texture(240, 320, CV_8UC1);
  cv::RNG rng(42);
  rng.fill(texture, cv::RNG::UNIFORM, 0, 256);
  cv::GaussianBlur(texture, texture, cv::Size(3, 3), 0.7);
  std::map<int, cv::Point2f> previous_left, previous_right;
  for (int frame = 0; frame < 12; ++frame) {
    cv::Mat image, right;
    cv::Mat transform = (cv::Mat_<double>(2, 3) << 1, 0, frame * 0.5, 0, 1, frame * 0.25);
    cv::warpAffine(texture, image, transform, texture.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
    if (stereo) {
      transform.at<double>(0, 2) -= 3;
      cv::warpAffine(texture, right, transform, texture.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
    }
    if (frame == 6) {
      std::map<int, Eigen::Vector3d> prediction;
      for (const auto & [id, point] : previous_left)
        prediction.emplace(id, Eigen::Vector3d(point.x, point.y, 1));
      tracker.setPrediction(prediction);
    }
    const bool extract = frame % 2 == 1;
    const bool draw = frame >= 8;
    const auto observations = tracker.trackImage(1 + frame * 0.05, image, right, extract, draw);
    EXPECT_EQ(observations.empty(), !extract);
    EXPECT_EQ(tracker.imTrack_.empty(), !draw);
    EXPECT_EQ(tracker.prevLeftPtsMap_.empty(), !draw);
    ASSERT_GE(tracker.ids_.size(), 50u);
    ASSERT_EQ(tracker.pts_velocity_.size(), tracker.ids_.size());
    for (size_t i = 0; i < tracker.ids_.size(); ++i) {
      const auto found = previous_left.find(tracker.ids_[i]);
      cv::Point2f expected(0, 0);
      if (found != previous_left.end())
        expected = (tracker.cur_un_pts_[i] - found->second) * 20;
      EXPECT_NEAR(tracker.pts_velocity_[i].x, expected.x, 1e-5);
      EXPECT_NEAR(tracker.pts_velocity_[i].y, expected.y, 1e-5);
    }
    if (stereo) {
      ASSERT_GE(tracker.ids_right_.size(), 40u);
      ASSERT_GE(tracker.right_pts_velocity_.size(), tracker.ids_right_.size());
      for (size_t i = 0; i < tracker.ids_right_.size(); ++i) {
        const auto found = previous_right.find(tracker.ids_right_[i]);
        cv::Point2f expected(0, 0);
        if (found != previous_right.end())
          expected = (tracker.cur_un_right_pts_[i] - found->second) * 20;
        EXPECT_NEAR(tracker.right_pts_velocity_[i].x, expected.x, 1e-5);
        EXPECT_NEAR(tracker.right_pts_velocity_[i].y, expected.y, 1e-5);
      }
    }
    previous_left.clear();
    previous_right.clear();
    for (size_t i = 0; i < tracker.ids_.size(); ++i)
      previous_left.emplace(tracker.ids_[i], tracker.cur_un_pts_[i]);
    for (size_t i = 0; i < tracker.ids_right_.size(); ++i)
      previous_right.emplace(tracker.ids_right_[i], tracker.cur_un_right_pts_[i]);
    // A zero-copy caller may recycle both images after the call returns.
    image.setTo(0);
    if (!right.empty())
      right.setTo(0);
  }
}
TEST_P(TrackerBuffers, ReusesUndistortedAndVelocityBuffersAcrossFrames)
{
  FeatureTracker tracker;
  tracker.m_camera_.emplace_back(new camodocal::PinholeCamera("left", 320, 240, 0, 0, 0, 0, 200, 200, 160, 120));
  const bool stereo = GetParam();
  tracker.stereo_cam_ = stereo;
  if (stereo)
    tracker.m_camera_.push_back(tracker.m_camera_.front());

  cv::Mat texture(240, 320, CV_8UC1);
  cv::RNG rng(4242);
  rng.fill(texture, cv::RNG::UNIFORM, 0, 256);
  cv::GaussianBlur(texture, texture, cv::Size(3, 3), 0.8);

  // Repeat an identical frame so the feature count settles; the next frame
  // must reuse the existing output buffers instead of rebuilding them.
  for (double stamp = 1.0; stamp <= 3.0; stamp += 1.0)
    tracker.trackImage(stamp, texture, stereo ? texture : cv::Mat());
  ASSERT_GE(tracker.cur_pts_.size(), 40u);
  ASSERT_EQ(tracker.cur_un_pts_.size(), tracker.cur_pts_.size());
  ASSERT_EQ(tracker.pts_velocity_.size(), tracker.cur_pts_.size());
  if (stereo) {
    ASSERT_GE(tracker.cur_right_pts_.size(), 20u);
    ASSERT_EQ(tracker.cur_un_right_pts_.size(), tracker.cur_right_pts_.size());
    ASSERT_GE(tracker.right_pts_velocity_.size(), tracker.ids_right_.size());
  }

  const cv::Point2f * left_un_pts = tracker.cur_un_pts_.data();
  const cv::Point2f * left_velocity = tracker.pts_velocity_.data();
  const cv::Point2f * right_un_pts = stereo ? tracker.cur_un_right_pts_.data() : nullptr;
  const cv::Point2f * right_velocity = stereo ? tracker.right_pts_velocity_.data() : nullptr;
  const size_t left_un_pts_size = tracker.cur_un_pts_.size();
  const size_t left_velocity_size = tracker.pts_velocity_.size();
  const size_t right_un_pts_size = tracker.cur_un_right_pts_.size();
  const size_t right_velocity_size = tracker.right_pts_velocity_.size();

  tracker.trackImage(4.0, texture, stereo ? texture : cv::Mat());

  EXPECT_EQ(tracker.cur_un_pts_.size(), left_un_pts_size);
  EXPECT_EQ(tracker.cur_un_pts_.data(), left_un_pts);
  EXPECT_EQ(tracker.pts_velocity_.size(), left_velocity_size);
  EXPECT_EQ(tracker.pts_velocity_.data(), left_velocity);
  if (stereo) {
    EXPECT_EQ(tracker.cur_un_right_pts_.size(), right_un_pts_size);
    EXPECT_EQ(tracker.cur_un_right_pts_.data(), right_un_pts);
    EXPECT_EQ(tracker.right_pts_velocity_.size(), right_velocity_size);
    EXPECT_EQ(tracker.right_pts_velocity_.data(), right_velocity);
  }
}

INSTANTIATE_TEST_SUITE_P(MonoAndStereo, TrackerBuffers, ::testing::Bool());

TEST(FeatureTrackerBuffers, ReusesFlowAndRansacScratchAcrossFrames)
{
  TrackerParameterGuard_s guard;
  MAX_CNT = 120;
  MIN_DIST = 10;
  FLOW_BACK = 1;
  SHOW_TRACK = 0;
  FEATURE_DETECTOR = "gftt";
  DETECTION_MIN_FEATURE_RATIO = 0.0;
  RANSAC_OUTLIER_REJECTION = 1;
  cv::setNumThreads(1);

  FeatureTracker tracker;
  tracker.m_camera_.emplace_back(new camodocal::PinholeCamera("left", 320, 240, 0, 0, 0, 0, 200, 200, 160, 120));
  cv::Mat texture(240, 320, CV_8UC1);
  cv::RNG rng(5150);
  rng.fill(texture, cv::RNG::UNIFORM, 0, 256);
  cv::GaussianBlur(texture, texture, cv::Size(3, 3), 0.8);

  for (double stamp = 1.0; stamp <= 3.0; stamp += 1.0)
    tracker.trackImage(stamp, texture);

  const uchar * status_data = FeatureTrackerTestAccess::flowStatus(tracker).data();
  const float * error_data = FeatureTrackerTestAccess::flowError(tracker).data();
  const uchar * reverse_status_data = FeatureTrackerTestAccess::reverseStatus(tracker).data();
  const cv::Point2f * reverse_pts_data = FeatureTrackerTestAccess::reversePoints(tracker).data();
  const cv::Point2f * ransac_cur_data = FeatureTrackerTestAccess::ransacCurrentPoints(tracker).data();
  const cv::Point2f * ransac_prev_data = FeatureTrackerTestAccess::ransacPreviousPoints(tracker).data();
  ASSERT_NE(status_data, nullptr);
  ASSERT_NE(error_data, nullptr);
  ASSERT_NE(reverse_status_data, nullptr);
  ASSERT_NE(reverse_pts_data, nullptr);
  ASSERT_NE(ransac_cur_data, nullptr);
  ASSERT_NE(ransac_prev_data, nullptr);

  tracker.trackImage(4.0, texture);

  EXPECT_EQ(FeatureTrackerTestAccess::flowStatus(tracker).data(), status_data);
  EXPECT_EQ(FeatureTrackerTestAccess::flowError(tracker).data(), error_data);
  EXPECT_EQ(FeatureTrackerTestAccess::reverseStatus(tracker).data(), reverse_status_data);
  EXPECT_EQ(FeatureTrackerTestAccess::reversePoints(tracker).data(), reverse_pts_data);
  EXPECT_EQ(FeatureTrackerTestAccess::ransacCurrentPoints(tracker).data(), ransac_cur_data);
  EXPECT_EQ(FeatureTrackerTestAccess::ransacPreviousPoints(tracker).data(), ransac_prev_data);
}

TEST(FeatureTrackerBuffers, RemoveOutliersKeepsLiftedCacheAlignedForPrediction)
{
  TrackerParameterGuard_s guard;
  MAX_CNT = 100;
  MIN_DIST = 10;
  FLOW_BACK = 1;
  SHOW_TRACK = 0;
  FEATURE_DETECTOR = "gftt";
  DETECTION_MIN_FEATURE_RATIO = 0.0;
  RANSAC_OUTLIER_REJECTION = 1;
  cv::setNumThreads(1);

  FeatureTracker tracker;
  tracker.m_camera_.emplace_back(new camodocal::PinholeCamera("left", 320, 240, 0, 0, 0, 0, 200, 200, 160, 120));
  cv::Mat texture(240, 320, CV_8UC1);
  cv::RNG rng(8080);
  rng.fill(texture, cv::RNG::UNIFORM, 0, 256);
  cv::GaussianBlur(texture, texture, cv::Size(3, 3), 0.8);
  tracker.trackImage(1.0, texture);
  ASSERT_GT(tracker.prev_pts_.size(), 30u);
  ASSERT_EQ(FeatureTrackerTestAccess::previousLiftedPoints(tracker).size(), tracker.prev_pts_.size());

  std::set<int> removed_ids;
  for (size_t i = 0; i < tracker.ids_.size(); i += 3)
    removed_ids.insert(tracker.ids_[i]);
  const size_t expected_count = tracker.prev_pts_.size() - removed_ids.size();
  tracker.removeOutliers(removed_ids);

  ASSERT_EQ(tracker.prev_pts_.size(), expected_count);
  EXPECT_EQ(tracker.ids_.size(), expected_count);
  EXPECT_EQ(tracker.track_cnt_.size(), expected_count);
  EXPECT_EQ(FeatureTrackerTestAccess::previousLiftedPoints(tracker).size(), expected_count);
  const std::vector<cv::Point3d> & compacted_lift = FeatureTrackerTestAccess::previousLiftedPoints(tracker);
  for (size_t i = 0; i < tracker.prev_pts_.size(); ++i) {
    Eigen::Vector3d expected_lift;
    tracker.m_camera_[0]->liftProjective(
      Eigen::Vector2d(tracker.prev_pts_[i].x, tracker.prev_pts_[i].y), expected_lift);
    EXPECT_EQ(compacted_lift[i], cv::Point3d(expected_lift.x(), expected_lift.y(), expected_lift.z()));
  }

  std::map<int, Eigen::Vector3d> predictions;
  for (size_t i = 0; i < tracker.ids_.size(); ++i) {
    Eigen::Vector3d lifted;
    tracker.m_camera_[0]->liftProjective(
      Eigen::Vector2d(tracker.prev_pts_[i].x, tracker.prev_pts_[i].y), lifted);
    predictions.emplace(tracker.ids_[i], lifted);
  }
  tracker.setPrediction(predictions);
  ASSERT_EQ(tracker.predict_pts_.size(), tracker.ids_.size());

  cv::Mat shifted;
  const cv::Mat shift = (cv::Mat_<double>(2, 3) << 1, 0, 0.75, 0, 1, -0.5);
  cv::warpAffine(texture, shifted, shift, texture.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
  EXPECT_GT(tracker.trackImage(1.05, shifted).size(), 20u);
  EXPECT_EQ(FeatureTrackerTestAccess::previousLiftedPoints(tracker).size(), tracker.prev_pts_.size());
}

TEST(FeatureTrackerPyramidCache, FollowsCurrentFrameAndResetsOnResize)
{
  TrackerParameterGuard_s guard;
  MAX_CNT = 70;
  MIN_DIST = 8;
  FLOW_BACK = 1;
  SHOW_TRACK = 0;
  FEATURE_DETECTOR = "gftt";
  DETECTION_MIN_FEATURE_RATIO = 0.0;
  cv::setNumThreads(1);

  FeatureTracker tracker;
  tracker.m_camera_.emplace_back(new camodocal::PinholeCamera("left", 320, 240, 0, 0, 0, 0, 200, 200, 160, 120));
  cv::Mat image = makeCornerTexture();

  tracker.trackImage(1.0, image);
  const auto & first_pyramid = tracker.previousPyramid();
  ASSERT_EQ(first_pyramid.size(), K_KLT_IMAGE_PYRAMID_MATS);
  EXPECT_EQ(first_pyramid.front().size(), image.size());
  EXPECT_EQ(first_pyramid[0].type(), CV_8UC1);
  EXPECT_EQ(first_pyramid[1].size(), cv::Size(image.cols / 2, image.rows / 2));
  EXPECT_EQ(first_pyramid[2].size(), cv::Size(image.cols / 4, image.rows / 4));
  EXPECT_EQ(first_pyramid[3].size(), cv::Size(image.cols / 8, image.rows / 8));
  for (const cv::Mat & level : first_pyramid)
    EXPECT_EQ(level.type(), CV_8UC1);
  // The cache must not alias a caller-owned frame that can be recycled.
  EXPECT_NE(first_pyramid.front().data, image.data);
  const double cached_mean = cv::mean(first_pyramid.front())[0];
  image.setTo(0);
  EXPECT_NEAR(cv::mean(first_pyramid.front())[0], cached_mean, 1e-9);

  cv::Mat next = makeCornerTexture();
  const cv::Mat shift = (cv::Mat_<double>(2, 3) << 1, 0, 1, 0, 1, 0);
  cv::warpAffine(next, next, shift, next.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
  tracker.trackImage(2.0, next);
  ASSERT_EQ(tracker.previousPyramid().size(), K_KLT_IMAGE_PYRAMID_MATS);
  EXPECT_EQ(tracker.previousPyramid().front().size(), next.size());
  EXPECT_NE(tracker.previousPyramid().front().data, next.data);

  cv::Mat resized;
  cv::resize(next, resized, cv::Size(360, 280));
  tracker.trackImage(3.0, resized);
  ASSERT_EQ(tracker.previousPyramid().size(), K_KLT_IMAGE_PYRAMID_MATS);
  EXPECT_EQ(tracker.previousPyramid().front().size(), resized.size());
  EXPECT_EQ(tracker.prev_pts_.size(), tracker.ids_.size());
}

TEST(FeatureTrackerPyramidCache, FastGridRetainsDerivativePyramid)
{
  TrackerParameterGuard_s guard;
  MAX_CNT = 70;
  MIN_DIST = 8;
  FLOW_BACK = 1;
  SHOW_TRACK = 0;
  FEATURE_DETECTOR = "fast_grid";
  FAST_THRESHOLD = 10;
  GRID_X = 5;
  GRID_Y = 5;
  DETECTION_MIN_FEATURE_RATIO = 0.0;
  cv::setNumThreads(1);

  FeatureTracker tracker;
  tracker.m_camera_.emplace_back(new camodocal::PinholeCamera("left", 320, 240, 0, 0, 0, 0, 200, 200, 160, 120));
  const cv::Mat image = makeCornerTexture();
  tracker.trackImage(1.0, image);

  const auto & pyramid = tracker.previousPyramid();
  ASSERT_EQ(pyramid.size(), K_KLT_DERIVATIVE_PYRAMID_MATS);
  EXPECT_EQ(pyramid[0].type(), CV_8UC1);
  EXPECT_EQ(pyramid[1].type(), CV_16SC2);  // level-0 derivatives
  EXPECT_EQ(pyramid[2].size(), cv::Size(image.cols / 2, image.rows / 2));
  EXPECT_EQ(pyramid[3].size(), cv::Size(image.cols / 2, image.rows / 2));
}

TEST(FeatureTrackerPyramidCache, DetectorChangeDiscardsIncompatibleCache)
{
  TrackerParameterGuard_s guard;
  MAX_CNT = 40;
  MIN_DIST = 8;
  FLOW_BACK = 0;
  SHOW_TRACK = 0;
  FEATURE_DETECTOR = "gftt";
  DETECTION_MIN_FEATURE_RATIO = 0.0;
  cv::setNumThreads(1);

  FeatureTracker tracker;
  tracker.m_camera_.emplace_back(new camodocal::PinholeCamera("left", 320, 240, 0, 0, 0, 0, 200, 200, 160, 120));
  const cv::Mat image = makeCornerTexture();
  tracker.trackImage(1.0, image);
  ASSERT_EQ(tracker.previousPyramid().size(), K_KLT_IMAGE_PYRAMID_MATS);
  ASSERT_FALSE(tracker.ids_.empty());

  FEATURE_DETECTOR = "fast_grid";
  FAST_THRESHOLD = 10;
  GRID_X = 5;
  GRID_Y = 5;
  tracker.trackImage(2.0, image);
  EXPECT_EQ(tracker.previousPyramid().size(), K_KLT_DERIVATIVE_PYRAMID_MATS);
  EXPECT_EQ(tracker.prev_pts_.size(), tracker.ids_.size());

  // The reverse transition must also discard the derivative layout.
  FEATURE_DETECTOR = "gftt";
  tracker.trackImage(3.0, image);
  EXPECT_EQ(tracker.previousPyramid().size(), K_KLT_IMAGE_PYRAMID_MATS);
  EXPECT_EQ(tracker.prev_pts_.size(), tracker.ids_.size());
}

TEST(FeatureTrackerKLTCache, ImageOnlyPyramidMatchesDirectMatOpticalFlow)
{
  TrackerParameterGuard_s guard;
  MAX_CNT = 80;
  MIN_DIST = 8;
  FLOW_BACK = 1;
  SHOW_TRACK = 0;
  FEATURE_DETECTOR = "gftt";
  cv::setNumThreads(1);

  const cv::Mat first = makeCornerTexture();
  cv::Mat second;
  const cv::Mat shift = (cv::Mat_<double>(2, 3) << 1, 0, 1.25, 0, 1, -0.75);
  cv::warpAffine(first, second, shift, first.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
  cv::Mat mask(first.size(), CV_8UC1, cv::Scalar(255));
  std::vector<cv::Point2f> points;
  cv::goodFeaturesToTrack(first, points, MAX_CNT, 0.01, MIN_DIST, mask);
  ASSERT_FALSE(points.empty());

  std::vector<cv::Mat> first_pyramid, second_pyramid;
  cv::buildOpticalFlowPyramid(
    first, first_pyramid, cv::Size(21, 21), 3, false, cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, false);
  cv::buildOpticalFlowPyramid(
    second, second_pyramid, cv::Size(21, 21), 3, false, cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, false);
  ASSERT_EQ(first_pyramid.size(), K_KLT_IMAGE_PYRAMID_MATS);
  ASSERT_EQ(second_pyramid.size(), K_KLT_IMAGE_PYRAMID_MATS);

  const cv::Size win(21, 21);
  const cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
  std::vector<float> error;

  // Mat overload: the reference behavior used before pyramid caching.
  std::vector<cv::Point2f> direct_tracked, direct_reverse;
  std::vector<uchar> direct_status, direct_reverse_status;
  cv::calcOpticalFlowPyrLK(first, second, points, direct_tracked, direct_status, error, win, 3, criteria);
  direct_reverse = points;
  cv::calcOpticalFlowPyrLK(
    second, first, direct_tracked, direct_reverse, direct_reverse_status, error, win, 1, criteria,
    cv::OPTFLOW_USE_INITIAL_FLOW);

  // Cached image-only pyramid: the GFTT front-end path.
  std::vector<cv::Point2f> cached_tracked, cached_reverse;
  std::vector<uchar> cached_status, cached_reverse_status;
  cv::calcOpticalFlowPyrLK(
    first_pyramid, second_pyramid, points, cached_tracked, cached_status, error, win, 3, criteria);
  cached_reverse = points;
  cv::calcOpticalFlowPyrLK(
    second_pyramid, first_pyramid, cached_tracked, cached_reverse, cached_reverse_status, error, win, 1,
    criteria, cv::OPTFLOW_USE_INITIAL_FLOW);

  ASSERT_EQ(cached_tracked.size(), direct_tracked.size());
  ASSERT_EQ(cached_reverse.size(), direct_reverse.size());
  for (size_t i = 0; i < cached_tracked.size(); ++i) {
    EXPECT_EQ(cached_status[i], direct_status[i]);
    EXPECT_FLOAT_EQ(cached_tracked[i].x, direct_tracked[i].x);
    EXPECT_FLOAT_EQ(cached_tracked[i].y, direct_tracked[i].y);
  }
  for (size_t i = 0; i < cached_reverse.size(); ++i) {
    EXPECT_EQ(cached_reverse_status[i], direct_reverse_status[i]);
    EXPECT_FLOAT_EQ(cached_reverse[i].x, direct_reverse[i].x);
    EXPECT_FLOAT_EQ(cached_reverse[i].y, direct_reverse[i].y);
  }
}

TEST(FeatureTrackerFastGrid, RespectsAllowedMaskMinDistanceAndRequestedCount)
{
  TrackerParameterGuard_s guard;
  MAX_CNT = 100;
  MIN_DIST = 12;
  FLOW_BACK = 1;
  SHOW_TRACK = 0;
  FEATURE_DETECTOR = "fast_grid";
  FAST_THRESHOLD = 10;
  GRID_X = 5;
  GRID_Y = 5;
  cv::setNumThreads(1);

  const cv::Mat image = makeCornerTexture();
  cv::Mat allowed_mask(image.size(), CV_8UC1, cv::Scalar(255));
  const cv::Rect allowed_region(40, 30, 240, 180);
  allowed_mask.setTo(0);
  allowed_mask(allowed_region).setTo(255);

  FeatureTracker tracker;
  std::vector<cv::Point2f> points;
  constexpr int requested = 37;
  tracker.detectFeatures(image, allowed_mask, requested, points);

  ASSERT_FALSE(points.empty());
  EXPECT_EQ(points.size(), static_cast<size_t>(requested));

  std::set<std::pair<int, int>> occupied_cells;
  for (size_t i = 0; i < points.size(); ++i) {
    const cv::Point2f & pt = points[i];
    ASSERT_TRUE(std::isfinite(pt.x));
    ASSERT_TRUE(std::isfinite(pt.y));
    EXPECT_GE(pt.x, 1.0f);
    EXPECT_LT(pt.x, static_cast<float>(image.cols - 1));
    EXPECT_GE(pt.y, 1.0f);
    EXPECT_LT(pt.y, static_cast<float>(image.rows - 1));
    const int x = cvRound(pt.x);
    const int y = cvRound(pt.y);
    ASSERT_GE(x, 1);
    ASSERT_LT(x, image.cols - 1);
    ASSERT_GE(y, 1);
    ASSERT_LT(y, image.rows - 1);
    EXPECT_GE(allowed_mask.at<uchar>(y, x), 128);
    EXPECT_TRUE(allowed_region.contains(cv::Point(x, y)));
    occupied_cells.emplace(
      std::min(GRID_X - 1, x * GRID_X / image.cols), std::min(GRID_Y - 1, y * GRID_Y / image.rows));

    for (size_t j = 0; j < i; ++j)
      EXPECT_GE(cv::norm(pt - points[j]), static_cast<float>(MIN_DIST) - 1e-3f);
  }
  // A periodic checkerboard can leave no FAST seeds at Shi-Tomasi maxima.
  // Recovery must reproduce the coherent GFTT set instead of mixing rankings;
  // a complete FAST set retains round-robin spatial distribution.
  std::vector<cv::Point2f> recovery;
  cv::goodFeaturesToTrack(image, recovery, requested, 0.01, MIN_DIST, allowed_mask);
  if (points != recovery)
    EXPECT_GE(occupied_cells.size(), 20u);
}

TEST(FeatureTrackerFastGrid, DetectionRatioDefersTopUpUntilThresholdExceeded)
{
  TrackerParameterGuard_s guard;
  MAX_CNT = 100;
  MIN_DIST = 8;
  FLOW_BACK = 0;
  SHOW_TRACK = 0;
  FEATURE_DETECTOR = "fast_grid";
  FAST_THRESHOLD = 10;
  GRID_X = 5;
  GRID_Y = 5;
  DETECTION_MIN_FEATURE_RATIO = 0.5;
  cv::setNumThreads(1);

  FeatureTracker tracker;
  tracker.m_camera_.emplace_back(new camodocal::PinholeCamera("left", 320, 240, 0, 0, 0, 0, 200, 200, 160, 120));
  const cv::Mat image = makeCornerTexture();
  tracker.trackImage(1.0, image);
  ASSERT_EQ(tracker.ids_.size(), static_cast<size_t>(MAX_CNT));

  // Simulate 20 lost tracks: 20 <= 0.5 * max_cnt, so this frame may run KLT
  // but must not top up with a detector.
  tracker.prev_pts_.resize(80);
  tracker.ids_.resize(80);
  tracker.track_cnt_.resize(80);
  tracker.trackImage(2.0, image);
  EXPECT_EQ(tracker.ids_.size(), 80u);

  // The same 20-point deficit is above a zero ratio and triggers FAST top-up.
  DETECTION_MIN_FEATURE_RATIO = 0.0;
  tracker.trackImage(3.0, image);
  EXPECT_GT(tracker.ids_.size(), 80u);
  EXPECT_LE(tracker.ids_.size(), static_cast<size_t>(MAX_CNT));
}

TEST(FeatureTrackerFastGridQuality, SparseMinEigenMatchesOpenCvOracle)
{
  cv::Mat image(80, 96, CV_8UC1);
  cv::RNG rng(20261110);
  rng.fill(image, cv::RNG::UNIFORM, 0, 256);
  cv::GaussianBlur(image, image, cv::Size(3, 3), 0.8);

  cv::Mat oracle;
  cv::cornerMinEigenVal(image, oracle, 3, 3);

  int checked = 0;
  for (int y = 4; y < image.rows - 4; y += 5) {
    for (int x = 4; x < image.cols - 4; x += 5) {
      const float sparse = vins::tracking::fastCornerMinEigenValue(image, cv::Point2f(x, y));
      const float expected = oracle.at<float>(y, x);
      EXPECT_NEAR(sparse, expected, 1e-6f + 1e-4f * std::abs(expected)) << "at (" << x << ", " << y << ")";
      ++checked;
    }
  }
  EXPECT_GT(checked, 100);

  // The 5x5 Sobel/block support must not fabricate responses at the image
  // border or for a non-finite probe.
  const float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(vins::tracking::fastCornerMinEigenValue(image, cv::Point2f(nan, 10)), 0.0f);
  EXPECT_EQ(vins::tracking::fastCornerMinEigenValue(image, cv::Point2f(1, 1)), 0.0f);
  EXPECT_EQ(vins::tracking::fastCornerMinEigenValue(image, cv::Point2f(image.cols - 2, image.rows - 2)), 0.0f);
}

TEST(FeatureTrackerFastGridQuality, RejectsRankOneEdgeButKeepsTwoDimensionalCorner)
{
  cv::Mat image(120, 160, CV_8UC1, cv::Scalar(0));
  // Strong vertical step: high FAST contrast, but the structure tensor has one
  // dominant gradient direction and therefore (near) zero min eigenvalue.
  image.colRange(100, image.cols).setTo(255);
  // A real corner with the same contrast magnitude.
  image(cv::Rect(22, 22, 16, 16)).setTo(255);
  cv::GaussianBlur(image, image, cv::Size(3, 3), 0.8);

  const float edge = vins::tracking::fastCornerMinEigenValue(image, cv::Point2f(101, 90));
  const float corner = vins::tracking::fastCornerMinEigenValue(image, cv::Point2f(22, 22));
  EXPECT_GT(corner, 0.0f);
  // This is exactly the 0.01 relative gate applied to FAST candidates.
  EXPECT_LT(edge, 0.01f * corner);
}

TEST(FeatureTrackerFastGrid, CompleteFastSetUsesMeasuredCornerMaxima)
{
  TrackerParameterGuard_s guard;
  MIN_DIST = 12;
  FEATURE_DETECTOR = "fast_grid";
  FAST_THRESHOLD = 10;
  GRID_X = 1;
  GRID_Y = 1;
  cv::Mat image(120, 160, CV_8UC1);
  cv::RNG rng(20261110);
  rng.fill(image, cv::RNG::UNIFORM, 0, 256);
  cv::GaussianBlur(image, image, cv::Size(3, 3), 0.8);
  std::vector<cv::KeyPoint> seeds;
  cv::FAST(image, seeds, FAST_THRESHOLD, true);
  cv::Mat oracle;
  cv::cornerMinEigenVal(image, oracle, 3, 3);
  FeatureTracker tracker;
  std::vector<cv::Point2f> actual;
  tracker.detectFeatures(image, cv::Mat(), 1, actual);
  ASSERT_EQ(actual.size(), 1u);
  const auto point = actual.front();
  EXPECT_TRUE(std::any_of(
    seeds.begin(), seeds.end(), [point](const cv::KeyPoint & _seed) { return _seed.pt == point; }));
  const int x = cvRound(point.x), y = cvRound(point.y);
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx)
      EXPECT_GE(oracle.at<float>(y, x) + 1e-7f, oracle.at<float>(y + dy, x + dx));
}

TEST(FeatureTrackerFastGrid, RecoversLowContrastCornersWithGftt)
{
  TrackerParameterGuard_s guard;
  MIN_DIST = 12;
  FEATURE_DETECTOR = "fast_grid";
  FAST_THRESHOLD = 60;
  GRID_X = 5;
  GRID_Y = 5;
  cv::setNumThreads(1);
  cv::Mat image = makeCornerTexture();
  image.convertTo(image, CV_8UC1, 0.1, 100);
  std::vector<cv::KeyPoint> raw_fast;
  cv::FAST(image, raw_fast, FAST_THRESHOLD, true);
  ASSERT_TRUE(raw_fast.empty());
  cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(255));
  mask.colRange(0, 40).setTo(0);
  std::vector<cv::Point2f> expected, actual;
  cv::goodFeaturesToTrack(image, expected, 80, 0.01, MIN_DIST, mask);
  ASSERT_GT(expected.size(), 40u);
  FeatureTracker tracker;
  tracker.detectFeatures(image, mask, 80, actual);
  EXPECT_EQ(actual, expected);
}

TEST(FeatureTrackerRansac, RemovesOutlierWithoutTouchingCachedUndistortedPoints)
{
  TrackerParameterGuard_s guard;
  F_THRESHOLD = 1.0;
  RANSAC_OUTLIER_REJECTION = 1;

  FeatureTracker tracker;
  tracker.row_ = 480;
  tracker.col_ = 640;
  tracker.m_camera_.emplace_back(new camodocal::PinholeCamera("left", 640, 480, 0, 0, 0, 0, 460, 460, 320, 240));

  makeTwoViewCorrespondences(tracker.prev_pts_, tracker.cur_pts_);
  ASSERT_EQ(tracker.prev_pts_.size(), tracker.cur_pts_.size());
  constexpr size_t outlier_index = 9;
  tracker.cur_pts_[outlier_index].x += 80.0f;
  tracker.cur_pts_[outlier_index].y -= 55.0f;

  const int base_id = 1000;
  for (size_t i = 0; i < tracker.cur_pts_.size(); ++i) {
    tracker.ids_.push_back(base_id + static_cast<int>(i));
    tracker.track_cnt_.push_back(static_cast<int>(i) + 1);
  }
  // This cache belongs to the previous frame and can legitimately have a
  // different length from the current correspondence arrays.
  tracker.cur_un_pts_.assign(3, cv::Point2f(42.0f, 24.0f));
  const std::vector<cv::Point2f> cached_un_pts = tracker.cur_un_pts_;
  const std::vector<cv::Point2f> prev_before = tracker.prev_pts_;
  const std::vector<cv::Point2f> cur_before = tracker.cur_pts_;
  const size_t correspondence_count = tracker.cur_pts_.size();

  tracker.rejectWithF();

  ASSERT_EQ(tracker.prev_pts_.size(), tracker.cur_pts_.size());
  ASSERT_EQ(tracker.ids_.size(), tracker.cur_pts_.size());
  ASSERT_EQ(tracker.track_cnt_.size(), tracker.cur_pts_.size());
  EXPECT_EQ(tracker.cur_pts_.size(), correspondence_count - 1);
  EXPECT_EQ(tracker.cur_un_pts_, cached_un_pts);

  const int outlier_id = base_id + static_cast<int>(outlier_index);
  EXPECT_EQ(std::find(tracker.ids_.begin(), tracker.ids_.end(), outlier_id), tracker.ids_.end());
  for (size_t i = 0; i < tracker.ids_.size(); ++i) {
    const int original_index = tracker.ids_[i] - base_id;
    EXPECT_EQ(tracker.track_cnt_[i], original_index + 1);
    EXPECT_EQ(tracker.prev_pts_[i], prev_before[original_index]);
    EXPECT_EQ(tracker.cur_pts_[i], cur_before[original_index]);
  }
}

TEST(FeatureTrackerRansac, CachedDoubleLiftMatchesLegacyRecomputationThroughSetMask)
{
  TrackerParameterGuard_s guard;
  F_THRESHOLD = 1.0;
  MIN_DIST = 12;
  RANSAC_OUTLIER_REJECTION = 1;
  cv::setNumThreads(1);

  FeatureTracker cached;
  FeatureTracker recomputed;
  configureRansacState(cached, true);
  configureRansacState(recomputed, false);

  // Both trackers receive identical double-precision RANSAC inputs. The first
  // reuses the exact lifted points cached by undistortedPtsInto(), while the
  // second exercises the legacy per-frame liftProjective() fallback.
  cv::setRNGSeed(20261112);
  cached.rejectWithF();
  cv::setRNGSeed(20261112);
  recomputed.rejectWithF();

  ASSERT_LT(cached.cur_pts_.size(), 32u);
  EXPECT_EQ(
    FeatureTrackerTestAccess::ransacCurrentPoints(cached),
    FeatureTrackerTestAccess::ransacCurrentPoints(recomputed));
  EXPECT_EQ(
    FeatureTrackerTestAccess::ransacPreviousPoints(cached),
    FeatureTrackerTestAccess::ransacPreviousPoints(recomputed));
  EXPECT_EQ(FeatureTrackerTestAccess::ransacStatus(cached), FeatureTrackerTestAccess::ransacStatus(recomputed));
  EXPECT_EQ(cached.prev_pts_, recomputed.prev_pts_);
  EXPECT_EQ(cached.cur_pts_, recomputed.cur_pts_);
  EXPECT_EQ(cached.ids_, recomputed.ids_);
  EXPECT_EQ(cached.track_cnt_, recomputed.track_cnt_);

  // setMask() compacts by track lifetime and redraws the exclusion mask. This
  // covers the reorder that happens between RANSAC and the next frame's cache.
  cached.setMask();
  recomputed.setMask();
  EXPECT_EQ(cached.cur_pts_, recomputed.cur_pts_);
  EXPECT_EQ(cached.ids_, recomputed.ids_);
  EXPECT_EQ(cached.track_cnt_, recomputed.track_cnt_);
  ASSERT_EQ(cached.mask_.size(), recomputed.mask_.size());
  EXPECT_EQ(cv::norm(cached.mask_, recomputed.mask_, cv::NORM_INF), 0.0);
}

TEST(FeatureTrackerRansac, DegenerateFundamentalMatrixLeavesBuffersUntouched)
{
  TrackerParameterGuard_s guard;
  F_THRESHOLD = 1.0;
  RANSAC_OUTLIER_REJECTION = 1;

  FeatureTracker tracker;
  tracker.row_ = 480;
  tracker.col_ = 640;
  tracker.m_camera_.emplace_back(new camodocal::PinholeCamera("left", 640, 480, 0, 0, 0, 0, 460, 460, 320, 240));

  tracker.prev_pts_.assign(16, cv::Point2f(320.0f, 240.0f));
  tracker.cur_pts_.assign(16, cv::Point2f(320.0f, 240.0f));
  for (int i = 0; i < 16; ++i) {
    tracker.ids_.push_back(2000 + i);
    tracker.track_cnt_.push_back(i + 1);
  }
  tracker.cur_un_pts_.assign(2, cv::Point2f(1.0f, 2.0f));

  const auto prev_before = tracker.prev_pts_;
  const auto cur_before = tracker.cur_pts_;
  const auto ids_before = tracker.ids_;
  const auto track_before = tracker.track_cnt_;
  const auto cached_before = tracker.cur_un_pts_;

  tracker.rejectWithF();

  EXPECT_EQ(tracker.prev_pts_, prev_before);
  EXPECT_EQ(tracker.cur_pts_, cur_before);
  EXPECT_EQ(tracker.ids_, ids_before);
  EXPECT_EQ(tracker.track_cnt_, track_before);
  EXPECT_EQ(tracker.cur_un_pts_, cached_before);
}

TEST(FeatureTrackerPyramidDetection, RefinedPointsRespectOddSizeMaskAndSpacing)
{
  TrackerParameterGuard_s guard;
  FEATURE_DETECTOR = "gftt_pyramid";
  MIN_DIST = 12;
  cv::setNumThreads(1);
  const cv::Mat image = makeCornerTexture(241, 321);
  cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(255));
  mask(cv::Rect(100, 0, 80, image.rows)).setTo(0);
  FeatureTracker tracker;
  std::vector<cv::Point2f> points;
  tracker.detectFeatures(image, mask, 70, points);
  ASSERT_GT(points.size(), 20u);
  EXPECT_LE(points.size(), 70u);
  for (size_t i = 0; i < points.size(); ++i) {
    const auto & p = points[i];
    ASSERT_GE(p.x, 1);
    ASSERT_GE(p.y, 1);
    ASSERT_LT(p.x, image.cols - 1);
    ASSERT_LT(p.y, image.rows - 1);
    EXPECT_NE(mask.at<uchar>(cvRound(p.y), cvRound(p.x)), 0);
    for (size_t j = 0; j < i; ++j)
      EXPECT_GE(cv::norm(p - points[j]), MIN_DIST - 1e-5);
  }
  mask.setTo(0);
  tracker.detectFeatures(image, mask, 70, points);
  EXPECT_TRUE(points.empty());
  tracker.detectFeatures(image, cv::Mat(), 0, points);
  EXPECT_TRUE(points.empty());
  tracker.detectFeatures(cv::Mat(), cv::Mat(), 70, points);
  EXPECT_TRUE(points.empty());
}

TEST(FeatureTrackerPyramidDetection, UsesFullResolutionCoordinatesAndOwnsCachedPixels)
{
  TrackerParameterGuard_s guard;
  FEATURE_DETECTOR = "gftt_pyramid";
  MAX_CNT = 80;
  MIN_DIST = 12;
  FLOW_BACK = 1;
  SHOW_TRACK = 0;
  RANSAC_OUTLIER_REJECTION = 0;
  DETECTION_MIN_FEATURE_RATIO = 0;
  cv::setNumThreads(1);
  FeatureTracker tracker;
  tracker.m_camera_.emplace_back(new camodocal::PinholeCamera("left", 321, 241, 0, 0, 0, 0, 200, 200, 160, 120));
  // A non-periodic texture avoids the checkerboard's equally valid matches
  // one or more tiles away when testing absolute pixel coordinates.
  cv::Mat first(241, 321, CV_8UC1);
  cv::RNG rng(7143);
  rng.fill(first, cv::RNG::UNIFORM, 0, 256);
  cv::GaussianBlur(first, first, cv::Size(5, 5), 1.0);
  const auto observations = tracker.trackImage(1.0, first);
  ASSERT_GT(observations.size(), 30u);
  EXPECT_EQ(tracker.previousPyramid().size(), 8u);
  const cv::Mat saved = tracker.previousPyramid()[0].clone();
  cv::Mat second;
  const cv::Mat shift = (cv::Mat_<double>(2, 3) << 1, 0, 2, 0, 1, 1);
  cv::warpAffine(first, second, shift, first.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
  first.setTo(0);
  EXPECT_EQ(cv::norm(saved, tracker.previousPyramid()[0], cv::NORM_INF), 0);
  const auto next = tracker.trackImage(1.05, second);
  int matched = 0;
  for (const auto & entry : next) {
    const auto previous = observations.find(entry.first);
    if (previous == observations.end())
      continue;
    const auto & pixel = previous->second[0].second;
    // Reflected image borders are not a rigid translation. Check the
    // interior so LK patches see the same translated texture.
    if (pixel[3] < 40 || pixel[3] > 281 || pixel[4] < 40 || pixel[4] > 201)
      continue;
    const auto delta = entry.second[0].second - pixel;
    EXPECT_NEAR(delta[3], 2, 0.15);
    EXPECT_NEAR(delta[4], 1, 0.15);
    ++matched;
  }
  EXPECT_GT(matched, 25);
  // Changing detector layout must discard incompatible cached derivatives.
  FEATURE_DETECTOR = "gftt";
  tracker.trackImage(1.1, second);
  EXPECT_EQ(tracker.previousPyramid().size(), 4u);
}

TEST(FeatureTrackerBuffers, PublicUndistortAndVelocityMatchReusedInternalBuffers)
{
  TrackerParameterGuard_s guard;
  MAX_CNT = 90;
  MIN_DIST = 8;
  FLOW_BACK = 1;
  SHOW_TRACK = 0;
  FEATURE_DETECTOR = "gftt";
  DETECTION_MIN_FEATURE_RATIO = 0.0;
  cv::setNumThreads(1);

  FeatureTracker tracker;
  tracker.m_camera_.emplace_back(new camodocal::PinholeCamera("left", 320, 240, 0, 0, 0, 0, 200, 200, 160, 120));

  cv::Mat texture(240, 320, CV_8UC1);
  cv::RNG rng(20241110);
  rng.fill(texture, cv::RNG::UNIFORM, 0, 256);
  cv::GaussianBlur(texture, texture, cv::Size(3, 3), 0.8);
  tracker.trackImage(1.0, texture);

  // trackImage() swaps prev/cur maps and sets prev_time = cur_time before
  // returning, so keep the real first-frame state for the public helper.
  std::map<int, cv::Point2f> first_map = tracker.prev_un_pts_map_;
  const double first_time = tracker.cur_time_;
  ASSERT_FALSE(first_map.empty());

  cv::Mat shifted;
  const cv::Mat transform = (cv::Mat_<double>(2, 3) << 1, 0, 0.75, 0, 1, -0.5);
  cv::warpAffine(texture, shifted, transform, texture.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
  tracker.trackImage(2.0, shifted);
  ASSERT_GT(tracker.cur_pts_.size(), 20u);
  ASSERT_EQ(tracker.pts_velocity_.size(), tracker.ids_.size());

  const std::vector<cv::Point2f> expected_un_pts = tracker.cur_un_pts_;
  const std::vector<cv::Point2f> expected_velocity = tracker.pts_velocity_;
  const std::vector<cv::Point2f> un_pts = tracker.undistortedPts(tracker.cur_pts_, tracker.m_camera_[0]);
  EXPECT_EQ(un_pts, expected_un_pts);

  std::map<int, cv::Point2f> current_map;
  tracker.prev_time_ = first_time;
  const std::vector<cv::Point2f> velocity =
    tracker.ptsVelocity(tracker.ids_, tracker.cur_un_pts_, current_map, first_map);
  ASSERT_EQ(velocity.size(), expected_velocity.size());
  for (size_t i = 0; i < velocity.size(); ++i) {
    EXPECT_FLOAT_EQ(velocity[i].x, expected_velocity[i].x);
    EXPECT_FLOAT_EQ(velocity[i].y, expected_velocity[i].y);
  }
}

TEST(FeatureTrackerPyramidDetection, ReusedMaskScratchFollowsGeometryChanges)
{
  TrackerParameterGuard_s guard;
  FEATURE_DETECTOR = "gftt_pyramid";
  MIN_DIST = 12;
  cv::setNumThreads(1);

  FeatureTracker tracker;
  const cv::Size sizes[] = {cv::Size(321, 241), cv::Size(360, 280), cv::Size(321, 241)};
  for (const cv::Size & size : sizes) {
    const cv::Mat image = makeCornerTexture(size.height, size.width);
    cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(255));
    mask(cv::Rect(size.width / 3, 0, size.width / 3, size.height)).setTo(0);
    std::vector<cv::Point2f> points;
    tracker.detectFeatures(image, mask, 70, points);
    ASSERT_FALSE(points.empty());
    for (const cv::Point2f & point : points) {
      ASSERT_GE(point.x, 1);
      ASSERT_GE(point.y, 1);
      ASSERT_LT(point.x, image.cols - 1);
      ASSERT_LT(point.y, image.rows - 1);
      EXPECT_NE(mask.at<uchar>(cvRound(point.y), cvRound(point.x)), 0);
    }
  }
}
