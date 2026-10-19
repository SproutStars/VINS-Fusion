/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <gtest/gtest.h>

#include <vins_common/tic_toc.hpp>

#include <cmath>

TEST(TicToc, ReportsNonNegativeMilliseconds)
{
    TicToc timer;
    const double first = timer.toc();
    EXPECT_TRUE(std::isfinite(first));
    EXPECT_GE(first, 0.0);

    // Re-arm the timer through the public API and verify repeated use.
    timer.tic();
    const double second = timer.toc();
    EXPECT_TRUE(std::isfinite(second));
    EXPECT_GE(second, 0.0);
}
