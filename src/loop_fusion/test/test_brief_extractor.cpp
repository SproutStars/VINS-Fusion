#include <gtest/gtest.h>
#include <filesystem>
#include <stdexcept>
#include <unistd.h>
#include "brief_extractor.h"

TEST(BriefExtractor, CacheReusesPatternAndPreservesDescriptors)
{
    const auto path = std::filesystem::temp_directory_path() /
        ("vins_brief_test_" + std::to_string(getpid()) + ".yml");
    std::filesystem::copy_file(BRIEF_TEST_PATTERN, path, std::filesystem::copy_options::overwrite_existing);
    const auto cached = cachedBriefExtractor(path.string());
    const BriefExtractor reference(path.string());
    std::filesystem::remove(path);
    // A second keyframe must not try to reopen the template file.
    EXPECT_EQ(cached, cachedBriefExtractor(path.string()));
    cv::Mat image(120, 160, CV_8UC1);
    cv::RNG rng(123);
    rng.fill(image, cv::RNG::UNIFORM, 0, 256);
    const std::vector<cv::KeyPoint> points{{30, 30, 1}, {50, 80, 1}, {1, 1, 1}, {159, 119, 1}};
    std::vector<DVision::BRIEF::bitset> expected, actual;
    reference(image, points, expected);
    (*cached)(image, points, actual);
    EXPECT_EQ(actual, expected);
    const auto other = cachedBriefExtractor(BRIEF_TEST_PATTERN);
    EXPECT_NE(other, cached);
    EXPECT_THROW(cachedBriefExtractor(path.string()), std::runtime_error);
    EXPECT_EQ(other, cachedBriefExtractor(BRIEF_TEST_PATTERN));
    // The previous extractor remains valid even after the cache changes path.
    (*cached)(image, points, actual);
    EXPECT_EQ(actual, expected);
}
