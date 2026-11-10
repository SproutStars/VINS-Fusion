#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "featureTracker/feature_tracker.h"

namespace
{
struct TrackerParameterGuard_s {
  TrackerParameterGuard_s()
    : max_cnt_(MAX_CNT), min_dist_(MIN_DIST), feature_detector_(FEATURE_DETECTOR),
      fast_threshold_(FAST_THRESHOLD), grid_x_(GRID_X), grid_y_(GRID_Y),
      detection_min_feature_ratio_(DETECTION_MIN_FEATURE_RATIO), flow_back_(FLOW_BACK), show_track_(SHOW_TRACK)
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
};

std::string testDataPath(const std::string & _name)
{
  const std::string source_file(__FILE__);
  return source_file.substr(0, source_file.find_last_of('/')) + "/data/" + _name;
}

double medianGfttMs(const cv::Mat & _image, const cv::Mat & _mask, int _iterations)
{
  std::vector<double> samples;
  samples.reserve(_iterations);
  for (int i = 0; i < _iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    std::vector<cv::Point2f> points;
    cv::goodFeaturesToTrack(_image, points, MAX_CNT, 0.01, MIN_DIST, _mask);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}
}  // namespace

// Reproduces the allocator interaction behind the GFTT regression: repeated
// 752x480 detections followed by a live KLT pyramid. Derivative-augmented
// pyramids put goodFeaturesToTrack()'s 1.44 MB temporaries at the heap top and
// glibc trims/refaults them on every call (~4.45 ms instead of ~1.75 ms on the
// local x86 host). The GFTT path must cache image-only levels.
TEST(FeatureTrackerGfttPerformance, ImageOnlyPyramidDoesNotSlowDetection)
{
  TrackerParameterGuard_s guard;
  MAX_CNT = 120;
  MIN_DIST = 25;
  FLOW_BACK = 1;
  SHOW_TRACK = 0;
  FEATURE_DETECTOR = "gftt";
  FAST_THRESHOLD = 20;
  GRID_X = 5;
  GRID_Y = 5;
  DETECTION_MIN_FEATURE_RATIO = 0.0;
  cv::setNumThreads(1);

  const cv::Mat image = cv::imread(testDataPath("euroc_v2_01_first.png"), cv::IMREAD_GRAYSCALE);
  ASSERT_FALSE(image.empty()) << testDataPath("euroc_v2_01_first.png");
  ASSERT_EQ(image.size(), cv::Size(752, 480));
  cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(255));

  // Warm the allocator exactly as a running tracker does before its first
  // cache is populated.
  (void)medianGfttMs(image, mask, 20);
  const double baseline_ms = medianGfttMs(image, mask, 30);

  double cached_ms = 0.0;
  {
    FeatureTracker tracker;
    tracker.m_camera_.emplace_back(
      new camodocal::PinholeCamera("left", 752, 480, 0, 0, 0, 0, 460, 460, 376, 240));
    tracker.trackImage(1.0, image);

    const std::vector<cv::Mat> & pyramid = tracker.previousPyramid();
    ASSERT_EQ(pyramid.size(), 4u);
    for (const cv::Mat & level : pyramid) {
      EXPECT_EQ(level.type(), CV_8UC1);
    }

    cached_ms = medianGfttMs(image, mask, 30);
  }

  const double released_ms = medianGfttMs(image, mask, 30);
  RecordProperty("gftt_baseline_ms", baseline_ms);
  RecordProperty("gftt_cached_image_pyramid_ms", cached_ms);
  RecordProperty("gftt_after_release_ms", released_ms);
  std::cout << "GFTT median: baseline=" << baseline_ms << " ms, image-only cache=" << cached_ms
            << " ms, released=" << released_ms << " ms" << '\n';

  const double reference_ms = std::max(baseline_ms, released_ms);
  EXPECT_LT(cached_ms, 1.6 * reference_ms + 0.3)
    << "Caching the KLT image pyramid must not regress GFTT detection";
}
