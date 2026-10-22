#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "featureTracker/feature_tracker.h"

namespace
{
struct TrackerParameterGuard
{
    TrackerParameterGuard()
        : max_cnt(MAX_CNT), min_dist(MIN_DIST), feature_detector(FEATURE_DETECTOR),
          fast_threshold(FAST_THRESHOLD), grid_x(GRID_X), grid_y(GRID_Y),
          detection_min_feature_ratio(DETECTION_MIN_FEATURE_RATIO),
          flow_back(FLOW_BACK), show_track(SHOW_TRACK) {}

    ~TrackerParameterGuard()
    {
        MAX_CNT = max_cnt; MIN_DIST = min_dist; FEATURE_DETECTOR = feature_detector;
        FAST_THRESHOLD = fast_threshold; GRID_X = grid_x; GRID_Y = grid_y;
        DETECTION_MIN_FEATURE_RATIO = detection_min_feature_ratio;
        FLOW_BACK = flow_back; SHOW_TRACK = show_track;
    }

    int max_cnt;
    int min_dist;
    std::string feature_detector;
    int fast_threshold;
    int grid_x;
    int grid_y;
    double detection_min_feature_ratio;
    int flow_back;
    int show_track;
};

std::string testDataPath(const std::string &name)
{
    const std::string source_file(__FILE__);
    return source_file.substr(0, source_file.find_last_of('/')) + "/data/" + name;
}

double medianGfttMs(const cv::Mat &image, const cv::Mat &mask, int iterations)
{
    std::vector<double> samples;
    samples.reserve(iterations);
    for (int i = 0; i < iterations; ++i)
    {
        const auto start = std::chrono::steady_clock::now();
        std::vector<cv::Point2f> points;
        cv::goodFeaturesToTrack(image, points, MAX_CNT, 0.01, MIN_DIST, mask);
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
    TrackerParameterGuard guard;
    MAX_CNT = 120; MIN_DIST = 25; FLOW_BACK = 1; SHOW_TRACK = 0;
    FEATURE_DETECTOR = "gftt"; FAST_THRESHOLD = 20; GRID_X = 5; GRID_Y = 5;
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
        tracker.m_camera.emplace_back(new camodocal::PinholeCamera(
            "left", 752, 480, 0, 0, 0, 0, 460, 460, 376, 240));
        tracker.trackImage(1.0, image);

        const std::vector<cv::Mat> &pyramid = tracker.previousPyramid();
        ASSERT_EQ(pyramid.size(), 4u);
        for (const cv::Mat &level : pyramid)
        {
            EXPECT_EQ(level.type(), CV_8UC1);
        }

        cached_ms = medianGfttMs(image, mask, 30);
    }

    const double released_ms = medianGfttMs(image, mask, 30);
    RecordProperty("gftt_baseline_ms", baseline_ms);
    RecordProperty("gftt_cached_image_pyramid_ms", cached_ms);
    RecordProperty("gftt_after_release_ms", released_ms);
    std::cout << "GFTT median: baseline=" << baseline_ms
              << " ms, image-only cache=" << cached_ms
              << " ms, released=" << released_ms << " ms" << std::endl;

    const double reference_ms = std::max(baseline_ms, released_ms);
    EXPECT_LT(cached_ms, 1.6 * reference_ms + 0.3)
        << "Caching the KLT image pyramid must not regress GFTT detection";
}
