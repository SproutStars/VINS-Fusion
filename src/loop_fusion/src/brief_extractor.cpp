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

#include "brief_extractor.h"
#include <mutex>
#include <stdexcept>

BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
    cv::FileStorage file(pattern_file, cv::FileStorage::READ);
    if (!file.isOpened())
        throw std::runtime_error("Could not open BRIEF pattern: " + pattern_file);
    std::vector<int> x1, x2, y1, y2;
    file["x1"] >> x1;
    file["x2"] >> x2;
    file["y1"] >> y1;
    file["y2"] >> y2;
    if (x1.empty() || x1.size() != x2.size() || x1.size() != y1.size() || x1.size() != y2.size())
        throw std::runtime_error("Invalid BRIEF pattern: " + pattern_file);
    brief_.importPairs(x1, y1, x2, y2);
}

void BriefExtractor::operator()(const cv::Mat &image, const std::vector<cv::KeyPoint> &keys,
                                std::vector<DVision::BRIEF::bitset> &descriptors) const
{
    brief_.compute(image, keys, descriptors);
}

std::shared_ptr<const BriefExtractor> cachedBriefExtractor(const std::string &pattern_file)
{
    // Also serializes the legacy BRIEF constructor's global random generator.
    static std::mutex mutex;
    static std::string loaded_path;
    static std::shared_ptr<const BriefExtractor> extractor;
    std::lock_guard<std::mutex> lock(mutex);
    if (!extractor || loaded_path != pattern_file) {
        auto replacement = std::make_shared<const BriefExtractor>(pattern_file);
        loaded_path = pattern_file;
        extractor = std::move(replacement);
    }
    return extractor;
}
