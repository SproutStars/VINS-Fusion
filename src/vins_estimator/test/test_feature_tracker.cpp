#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <string>
#include <utility>

#include "featureTracker/feature_tracker.h"

namespace
{
// buildOpticalFlowPyramid(withDerivatives=true) interleaves an image and a
// derivative buffer per logical level, i.e. 2 * (maxLevel + 1) Mats. The GFTT
// path intentionally caches only the image levels.
constexpr size_t kKltDerivativePyramidMats = 2u * (3u + 1u);
constexpr size_t kKltImagePyramidMats = 3u + 1u;

struct TrackerParameterGuard
{
    TrackerParameterGuard()
        : max_cnt(MAX_CNT), min_dist(MIN_DIST), feature_detector(FEATURE_DETECTOR),
          fast_threshold(FAST_THRESHOLD), grid_x(GRID_X), grid_y(GRID_Y),
          detection_min_feature_ratio(DETECTION_MIN_FEATURE_RATIO),
          flow_back(FLOW_BACK), show_track(SHOW_TRACK),
          ransac_outlier_rejection(RANSAC_OUTLIER_REJECTION), f_threshold(F_THRESHOLD) {}

    ~TrackerParameterGuard()
    {
        MAX_CNT = max_cnt; MIN_DIST = min_dist; FEATURE_DETECTOR = feature_detector;
        FAST_THRESHOLD = fast_threshold; GRID_X = grid_x; GRID_Y = grid_y;
        DETECTION_MIN_FEATURE_RATIO = detection_min_feature_ratio;
        FLOW_BACK = flow_back; SHOW_TRACK = show_track;
        RANSAC_OUTLIER_REJECTION = ransac_outlier_rejection;
        F_THRESHOLD = f_threshold;
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
    int ransac_outlier_rejection;
    double f_threshold;
};

cv::Mat makeCornerTexture(int rows = 240, int cols = 320)
{
    cv::Mat image(rows, cols, CV_8UC1);
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            image.at<uchar>(y, x) = ((x / 12 + y / 12) & 1) ? 255 : 0;
    cv::GaussianBlur(image, image, cv::Size(3, 3), 0.5);
    return image;
}

void makeTwoViewCorrespondences(std::vector<cv::Point2f> &previous,
                                std::vector<cv::Point2f> &current)
{
    std::vector<cv::Point3f> object_points;
    for (int i = 0; i < 32; ++i)
    {
        object_points.emplace_back((i % 8 - 3.5f) * 0.25f,
                                   (i / 8 - 1.5f) * 0.30f,
                                   3.0f + 0.10f * (i % 5));
    }
    const cv::Mat camera_matrix =
        (cv::Mat_<double>(3, 3) << 460.0, 0.0, 320.0, 0.0, 460.0, 240.0, 0.0, 0.0, 1.0);
    const cv::Mat distortion = cv::Mat::zeros(1, 5, CV_64F);
    const cv::Mat first_rvec = cv::Mat::zeros(3, 1, CV_64F);
    const cv::Mat first_tvec = cv::Mat::zeros(3, 1, CV_64F);
    const cv::Mat second_rvec = (cv::Mat_<double>(3, 1) << 0.08, -0.04, 0.03);
    const cv::Mat second_tvec = (cv::Mat_<double>(3, 1) << 0.35, 0.05, 0.10);
    cv::projectPoints(object_points, first_rvec, first_tvec, camera_matrix, distortion, previous);
    cv::projectPoints(object_points, second_rvec, second_tvec, camera_matrix, distortion, current);
}

class TrackerBuffers : public ::testing::TestWithParam<bool>
{
protected:
    void SetUp() override
    {
        MAX_CNT = 100; MIN_DIST = 12; FLOW_BACK = 1; SHOW_TRACK = 1;
        FEATURE_DETECTOR = "gftt"; FAST_THRESHOLD = 20; GRID_X = 5; GRID_Y = 5;
        DETECTION_MIN_FEATURE_RATIO = 0.0;
        cv::setNumThreads(1);
    }
};
}

TEST(FeatureTrackerOptions, RansacOutlierRejectionDefaultsOffAndGuardRestoresSetting)
{
    EXPECT_EQ(RANSAC_OUTLIER_REJECTION, 0);

    {
        TrackerParameterGuard guard;
        RANSAC_OUTLIER_REJECTION = 1;
        EXPECT_EQ(RANSAC_OUTLIER_REJECTION, 1);
    }

    EXPECT_EQ(RANSAC_OUTLIER_REJECTION, 0);
}

TEST_P(TrackerBuffers, PreservesVelocityAcrossGatedFramesAndPrediction)
{
    FeatureTracker tracker;
    tracker.m_camera.emplace_back(new camodocal::PinholeCamera("left", 320, 240,
                                                             0, 0, 0, 0, 200, 200, 160, 120));
    const bool stereo = GetParam();
    tracker.stereo_cam = stereo;
    if (stereo) tracker.m_camera.push_back(tracker.m_camera.front());
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
            for (const auto &[id, point] : previous_left)
                prediction.emplace(id, Eigen::Vector3d(point.x, point.y, 1));
            tracker.setPrediction(prediction);
        }
        const bool extract = frame % 2 == 1;
        const bool draw = frame >= 8;
        const auto observations = tracker.trackImage(1 + frame * 0.05, image, right, extract, draw);
        EXPECT_EQ(observations.empty(), !extract);
        EXPECT_EQ(tracker.imTrack.empty(), !draw);
        EXPECT_EQ(tracker.prevLeftPtsMap.empty(), !draw);
        ASSERT_GE(tracker.ids.size(), 50u);
        ASSERT_EQ(tracker.pts_velocity.size(), tracker.ids.size());
        for (size_t i = 0; i < tracker.ids.size(); ++i) {
            const auto found = previous_left.find(tracker.ids[i]);
            cv::Point2f expected(0, 0);
            if (found != previous_left.end())
                expected = (tracker.cur_un_pts[i] - found->second) * 20;
            EXPECT_NEAR(tracker.pts_velocity[i].x, expected.x, 1e-5);
            EXPECT_NEAR(tracker.pts_velocity[i].y, expected.y, 1e-5);
        }
        if (stereo) {
            ASSERT_GE(tracker.ids_right.size(), 40u);
            ASSERT_GE(tracker.right_pts_velocity.size(), tracker.ids_right.size());
            for (size_t i = 0; i < tracker.ids_right.size(); ++i) {
                const auto found = previous_right.find(tracker.ids_right[i]);
                cv::Point2f expected(0, 0);
                if (found != previous_right.end())
                    expected = (tracker.cur_un_right_pts[i] - found->second) * 20;
                EXPECT_NEAR(tracker.right_pts_velocity[i].x, expected.x, 1e-5);
                EXPECT_NEAR(tracker.right_pts_velocity[i].y, expected.y, 1e-5);
            }
        }
        previous_left.clear(); previous_right.clear();
        for (size_t i = 0; i < tracker.ids.size(); ++i)
            previous_left.emplace(tracker.ids[i], tracker.cur_un_pts[i]);
        for (size_t i = 0; i < tracker.ids_right.size(); ++i)
            previous_right.emplace(tracker.ids_right[i], tracker.cur_un_right_pts[i]);
        // A zero-copy caller may recycle both images after the call returns.
        image.setTo(0);
        if (!right.empty()) right.setTo(0);
    }
}
INSTANTIATE_TEST_SUITE_P(MonoAndStereo, TrackerBuffers, ::testing::Bool());

TEST(FeatureTrackerPyramidCache, FollowsCurrentFrameAndResetsOnResize)
{
    TrackerParameterGuard guard;
    MAX_CNT = 70; MIN_DIST = 8; FLOW_BACK = 1; SHOW_TRACK = 0;
    FEATURE_DETECTOR = "gftt"; DETECTION_MIN_FEATURE_RATIO = 0.0;
    cv::setNumThreads(1);

    FeatureTracker tracker;
    tracker.m_camera.emplace_back(new camodocal::PinholeCamera("left", 320, 240,
                                                             0, 0, 0, 0, 200, 200, 160, 120));
    cv::Mat image = makeCornerTexture();

    tracker.trackImage(1.0, image);
    const auto &first_pyramid = tracker.previousPyramid();
    ASSERT_EQ(first_pyramid.size(), kKltImagePyramidMats);
    EXPECT_EQ(first_pyramid.front().size(), image.size());
    EXPECT_EQ(first_pyramid[0].type(), CV_8UC1);
    EXPECT_EQ(first_pyramid[1].size(), cv::Size(image.cols / 2, image.rows / 2));
    EXPECT_EQ(first_pyramid[2].size(), cv::Size(image.cols / 4, image.rows / 4));
    EXPECT_EQ(first_pyramid[3].size(), cv::Size(image.cols / 8, image.rows / 8));
    for (const cv::Mat &level : first_pyramid)
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
    ASSERT_EQ(tracker.previousPyramid().size(), kKltImagePyramidMats);
    EXPECT_EQ(tracker.previousPyramid().front().size(), next.size());
    EXPECT_NE(tracker.previousPyramid().front().data, next.data);

    cv::Mat resized;
    cv::resize(next, resized, cv::Size(360, 280));
    tracker.trackImage(3.0, resized);
    ASSERT_EQ(tracker.previousPyramid().size(), kKltImagePyramidMats);
    EXPECT_EQ(tracker.previousPyramid().front().size(), resized.size());
    EXPECT_EQ(tracker.prev_pts.size(), tracker.ids.size());
}

TEST(FeatureTrackerPyramidCache, FastGridRetainsDerivativePyramid)
{
    TrackerParameterGuard guard;
    MAX_CNT = 70; MIN_DIST = 8; FLOW_BACK = 1; SHOW_TRACK = 0;
    FEATURE_DETECTOR = "fast_grid"; FAST_THRESHOLD = 10; GRID_X = 5; GRID_Y = 5;
    DETECTION_MIN_FEATURE_RATIO = 0.0;
    cv::setNumThreads(1);

    FeatureTracker tracker;
    tracker.m_camera.emplace_back(new camodocal::PinholeCamera("left", 320, 240,
                                                             0, 0, 0, 0, 200, 200, 160, 120));
    const cv::Mat image = makeCornerTexture();
    tracker.trackImage(1.0, image);

    const auto &pyramid = tracker.previousPyramid();
    ASSERT_EQ(pyramid.size(), kKltDerivativePyramidMats);
    EXPECT_EQ(pyramid[0].type(), CV_8UC1);
    EXPECT_EQ(pyramid[1].type(), CV_16SC2);  // level-0 derivatives
    EXPECT_EQ(pyramid[2].size(), cv::Size(image.cols / 2, image.rows / 2));
    EXPECT_EQ(pyramid[3].size(), cv::Size(image.cols / 2, image.rows / 2));
}

TEST(FeatureTrackerPyramidCache, DetectorChangeDiscardsIncompatibleCache)
{
    TrackerParameterGuard guard;
    MAX_CNT = 40; MIN_DIST = 8; FLOW_BACK = 0; SHOW_TRACK = 0;
    FEATURE_DETECTOR = "gftt"; DETECTION_MIN_FEATURE_RATIO = 0.0;
    cv::setNumThreads(1);

    FeatureTracker tracker;
    tracker.m_camera.emplace_back(new camodocal::PinholeCamera("left", 320, 240,
                                                             0, 0, 0, 0, 200, 200, 160, 120));
    const cv::Mat image = makeCornerTexture();
    tracker.trackImage(1.0, image);
    ASSERT_EQ(tracker.previousPyramid().size(), kKltImagePyramidMats);
    ASSERT_FALSE(tracker.ids.empty());

    FEATURE_DETECTOR = "fast_grid";
    FAST_THRESHOLD = 10; GRID_X = 5; GRID_Y = 5;
    tracker.trackImage(2.0, image);
    EXPECT_EQ(tracker.previousPyramid().size(), kKltDerivativePyramidMats);
    EXPECT_EQ(tracker.prev_pts.size(), tracker.ids.size());
}

TEST(FeatureTrackerKLTCache, ImageOnlyPyramidMatchesDirectMatOpticalFlow)
{
    TrackerParameterGuard guard;
    MAX_CNT = 80; MIN_DIST = 8; FLOW_BACK = 1; SHOW_TRACK = 0;
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
    cv::buildOpticalFlowPyramid(first, first_pyramid, cv::Size(21, 21), 3, false,
                                cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, false);
    cv::buildOpticalFlowPyramid(second, second_pyramid, cv::Size(21, 21), 3, false,
                                cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, false);
    ASSERT_EQ(first_pyramid.size(), kKltImagePyramidMats);
    ASSERT_EQ(second_pyramid.size(), kKltImagePyramidMats);

    const cv::Size win(21, 21);
    const cv::TermCriteria criteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    std::vector<float> error;

    // Mat overload: the reference behavior used before pyramid caching.
    std::vector<cv::Point2f> direct_tracked, direct_reverse;
    std::vector<uchar> direct_status, direct_reverse_status;
    cv::calcOpticalFlowPyrLK(first, second, points, direct_tracked, direct_status, error,
                             win, 3, criteria);
    direct_reverse = points;
    cv::calcOpticalFlowPyrLK(second, first, direct_tracked, direct_reverse,
                             direct_reverse_status, error, win, 1, criteria,
                             cv::OPTFLOW_USE_INITIAL_FLOW);

    // Cached image-only pyramid: the GFTT front-end path.
    std::vector<cv::Point2f> cached_tracked, cached_reverse;
    std::vector<uchar> cached_status, cached_reverse_status;
    cv::calcOpticalFlowPyrLK(first_pyramid, second_pyramid, points, cached_tracked,
                             cached_status, error, win, 3, criteria);
    cached_reverse = points;
    cv::calcOpticalFlowPyrLK(second_pyramid, first_pyramid, cached_tracked, cached_reverse,
                             cached_reverse_status, error, win, 1, criteria,
                             cv::OPTFLOW_USE_INITIAL_FLOW);

    ASSERT_EQ(cached_tracked.size(), direct_tracked.size());
    ASSERT_EQ(cached_reverse.size(), direct_reverse.size());
    for (size_t i = 0; i < cached_tracked.size(); ++i)
    {
        EXPECT_EQ(cached_status[i], direct_status[i]);
        EXPECT_FLOAT_EQ(cached_tracked[i].x, direct_tracked[i].x);
        EXPECT_FLOAT_EQ(cached_tracked[i].y, direct_tracked[i].y);
    }
    for (size_t i = 0; i < cached_reverse.size(); ++i)
    {
        EXPECT_EQ(cached_reverse_status[i], direct_reverse_status[i]);
        EXPECT_FLOAT_EQ(cached_reverse[i].x, direct_reverse[i].x);
        EXPECT_FLOAT_EQ(cached_reverse[i].y, direct_reverse[i].y);
    }
}

TEST(FeatureTrackerFastGrid, RespectsAllowedMaskMinDistanceAndRequestedCount)
{
    TrackerParameterGuard guard;
    MAX_CNT = 100; MIN_DIST = 12; FLOW_BACK = 1; SHOW_TRACK = 0;
    FEATURE_DETECTOR = "fast_grid"; FAST_THRESHOLD = 10; GRID_X = 5; GRID_Y = 5;
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
    for (size_t i = 0; i < points.size(); ++i)
    {
        const cv::Point2f &pt = points[i];
        const int x = cvRound(pt.x);
        const int y = cvRound(pt.y);
        ASSERT_GE(x, 0); ASSERT_LT(x, image.cols);
        ASSERT_GE(y, 0); ASSERT_LT(y, image.rows);
        EXPECT_GE(allowed_mask.at<uchar>(y, x), 128);
        EXPECT_TRUE(allowed_region.contains(cv::Point(x, y)));
        occupied_cells.emplace(
            std::min(GRID_X - 1, x * GRID_X / image.cols),
            std::min(GRID_Y - 1, y * GRID_Y / image.rows));

        for (size_t j = 0; j < i; ++j)
            EXPECT_GE(cv::norm(pt - points[j]), static_cast<float>(MIN_DIST) - 1e-3f);
    }
    // Round-robin grid selection should spread these points over most cells.
    EXPECT_GE(occupied_cells.size(), 20u);
}

TEST(FeatureTrackerFastGrid, DetectionRatioDefersTopUpUntilThresholdExceeded)
{
    TrackerParameterGuard guard;
    MAX_CNT = 100; MIN_DIST = 8; FLOW_BACK = 0; SHOW_TRACK = 0;
    FEATURE_DETECTOR = "fast_grid"; FAST_THRESHOLD = 10; GRID_X = 5; GRID_Y = 5;
    DETECTION_MIN_FEATURE_RATIO = 0.5;
    cv::setNumThreads(1);

    FeatureTracker tracker;
    tracker.m_camera.emplace_back(new camodocal::PinholeCamera("left", 320, 240,
                                                             0, 0, 0, 0, 200, 200, 160, 120));
    const cv::Mat image = makeCornerTexture();
    tracker.trackImage(1.0, image);
    ASSERT_EQ(tracker.ids.size(), static_cast<size_t>(MAX_CNT));

    // Simulate 20 lost tracks: 20 <= 0.5 * max_cnt, so this frame may run KLT
    // but must not top up with a detector.
    tracker.prev_pts.resize(80);
    tracker.ids.resize(80);
    tracker.track_cnt.resize(80);
    tracker.trackImage(2.0, image);
    EXPECT_EQ(tracker.ids.size(), 80u);

    // The same 20-point deficit is above a zero ratio and triggers FAST top-up.
    DETECTION_MIN_FEATURE_RATIO = 0.0;
    tracker.trackImage(3.0, image);
    EXPECT_GT(tracker.ids.size(), 80u);
    EXPECT_LE(tracker.ids.size(), static_cast<size_t>(MAX_CNT));
}

TEST(FeatureTrackerRansac, RemovesOutlierWithoutTouchingCachedUndistortedPoints)
{
    TrackerParameterGuard guard;
    F_THRESHOLD = 1.0;
    RANSAC_OUTLIER_REJECTION = 1;

    FeatureTracker tracker;
    tracker.row = 480;
    tracker.col = 640;
    tracker.m_camera.emplace_back(new camodocal::PinholeCamera(
        "left", 640, 480, 0, 0, 0, 0, 460, 460, 320, 240));

    makeTwoViewCorrespondences(tracker.prev_pts, tracker.cur_pts);
    ASSERT_EQ(tracker.prev_pts.size(), tracker.cur_pts.size());
    constexpr size_t outlier_index = 9;
    tracker.cur_pts[outlier_index].x += 80.0f;
    tracker.cur_pts[outlier_index].y -= 55.0f;

    const int base_id = 1000;
    for (size_t i = 0; i < tracker.cur_pts.size(); ++i)
    {
        tracker.ids.push_back(base_id + static_cast<int>(i));
        tracker.track_cnt.push_back(static_cast<int>(i) + 1);
    }
    // This cache belongs to the previous frame and can legitimately have a
    // different length from the current correspondence arrays.
    tracker.cur_un_pts.assign(3, cv::Point2f(42.0f, 24.0f));
    const std::vector<cv::Point2f> cached_un_pts = tracker.cur_un_pts;
    const std::vector<cv::Point2f> prev_before = tracker.prev_pts;
    const std::vector<cv::Point2f> cur_before = tracker.cur_pts;
    const size_t correspondence_count = tracker.cur_pts.size();

    tracker.rejectWithF();

    ASSERT_EQ(tracker.prev_pts.size(), tracker.cur_pts.size());
    ASSERT_EQ(tracker.ids.size(), tracker.cur_pts.size());
    ASSERT_EQ(tracker.track_cnt.size(), tracker.cur_pts.size());
    EXPECT_EQ(tracker.cur_pts.size(), correspondence_count - 1);
    EXPECT_EQ(tracker.cur_un_pts, cached_un_pts);

    const int outlier_id = base_id + static_cast<int>(outlier_index);
    EXPECT_EQ(std::find(tracker.ids.begin(), tracker.ids.end(), outlier_id), tracker.ids.end());
    for (size_t i = 0; i < tracker.ids.size(); ++i)
    {
        const int original_index = tracker.ids[i] - base_id;
        EXPECT_EQ(tracker.track_cnt[i], original_index + 1);
        EXPECT_EQ(tracker.prev_pts[i], prev_before[original_index]);
        EXPECT_EQ(tracker.cur_pts[i], cur_before[original_index]);
    }
}

TEST(FeatureTrackerRansac, DegenerateFundamentalMatrixLeavesBuffersUntouched)
{
    TrackerParameterGuard guard;
    F_THRESHOLD = 1.0;
    RANSAC_OUTLIER_REJECTION = 1;

    FeatureTracker tracker;
    tracker.row = 480;
    tracker.col = 640;
    tracker.m_camera.emplace_back(new camodocal::PinholeCamera(
        "left", 640, 480, 0, 0, 0, 0, 460, 460, 320, 240));

    tracker.prev_pts.assign(16, cv::Point2f(320.0f, 240.0f));
    tracker.cur_pts.assign(16, cv::Point2f(320.0f, 240.0f));
    for (int i = 0; i < 16; ++i)
    {
        tracker.ids.push_back(2000 + i);
        tracker.track_cnt.push_back(i + 1);
    }
    tracker.cur_un_pts.assign(2, cv::Point2f(1.0f, 2.0f));

    const auto prev_before = tracker.prev_pts;
    const auto cur_before = tracker.cur_pts;
    const auto ids_before = tracker.ids;
    const auto track_before = tracker.track_cnt;
    const auto cached_before = tracker.cur_un_pts;

    tracker.rejectWithF();

    EXPECT_EQ(tracker.prev_pts, prev_before);
    EXPECT_EQ(tracker.cur_pts, cur_before);
    EXPECT_EQ(tracker.ids, ids_before);
    EXPECT_EQ(tracker.track_cnt, track_before);
    EXPECT_EQ(tracker.cur_un_pts, cached_before);
}
