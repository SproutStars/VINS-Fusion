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

#include "ThirdParty/DVision/BRIEF.h"
#include <memory>
#include <string>

class BriefExtractor {
public:
  explicit BriefExtractor(const std::string & _pattern_file);
  void operator()(
    const cv::Mat & _image, const std::vector<cv::KeyPoint> & _keys,
    std::vector<DVision::BRIEF::bitset> & _descriptors) const;

private:
  DVision::BRIEF brief_;
};

// The pattern is configuration, not per-frame data. A new path reloads it;
// replacing the contents at the same path requires restarting the node.
// Shared ownership also makes a path change safe for in-flight extraction.
std::shared_ptr<const BriefExtractor> cachedBriefExtractor(const std::string & _pattern_file);
