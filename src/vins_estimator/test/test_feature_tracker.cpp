#include <gtest/gtest.h>
#include "featureTracker/feature_tracker.h"

namespace
{
class TrackerBuffers : public ::testing::TestWithParam<bool>
{
protected:
    void SetUp() override
    {
        MAX_CNT = 100; MIN_DIST = 12; FLOW_BACK = 1; SHOW_TRACK = 1;
        cv::setNumThreads(1);
    }
};
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
