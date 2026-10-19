/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include <memory>
#include <string>
#include "ThirdParty/DVision/BRIEF.h"

class BriefExtractor
{
public:
    explicit BriefExtractor(const std::string &pattern_file);
    void operator()(const cv::Mat &image, const std::vector<cv::KeyPoint> &keys,
                    std::vector<DVision::BRIEF::bitset> &descriptors) const;
private:
    DVision::BRIEF brief_;
};

// The pattern is configuration, not per-frame data. A new path reloads it;
// replacing the contents at the same path requires restarting the node.
// Shared ownership also makes a path change safe for in-flight extraction.
std::shared_ptr<const BriefExtractor> cachedBriefExtractor(const std::string &pattern_file);
