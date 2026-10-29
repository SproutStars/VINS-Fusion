#include <gtest/gtest.h>

#include "timestamp_sync.h"

TEST(TimestampSync, AcceptsEpochDoubleRoundTripErrorButNotAnotherFrame)
{
    builtin_interfaces::msg::Time image;
    image.sec = 1403636579;
    image.nanosec = 123456789;

    builtin_interfaces::msg::Time pose = image;
    pose.nanosec += 119;  // observed epoch-double round-trip error
    EXPECT_TRUE(loop_fusion::stampsMatch(loop_fusion::stampNs(image),
                                         loop_fusion::stampNs(pose)));
    EXPECT_EQ(loop_fusion::kStampMatchToleranceNs, 2000);

    builtin_interfaces::msg::Time next_image = image;
    next_image.nanosec += 50'000'000;  // 20 Hz frame period
    EXPECT_FALSE(loop_fusion::stampsMatch(loop_fusion::stampNs(image),
                                          loop_fusion::stampNs(next_image)));
}
