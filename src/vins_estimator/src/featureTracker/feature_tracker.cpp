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

#include <vins_common/ros_utils.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>

#include "feature_tracker.h"

namespace
{
constexpr int K_KLT_WIN_SIZE = 21;
constexpr int K_KLT_MAX_LEVEL = 3;
// With a temporal detection gate enabled, a deficit this large is treated as
// urgent and bypasses the interval so tracking cannot starve for many frames.
constexpr double K_URGENT_DETECTION_DEFICIT_RATIO = 0.5;

double elapsedMs(const std::chrono::steady_clock::time_point & _start)
{
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _start).count();
}

cv::TermCriteria kltTermCriteria()
{
  return cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
}

void buildFlowPyramid(const cv::Mat & _img, vector<cv::Mat> & _pyramid, bool _with_derivatives)
{
  if (_img.empty()) {
    _pyramid.clear();
    return;
  }
  // Do not clear the vector first: buildOpticalFlowPyramid() reuses existing
  // level Mats when their geometry/type match. The caller alternates two
  // persistent vectors (current and previous), so steady-state frames avoid
  // allocating and freeing the pyramid buffers on every call.
  // The cache must own its pixels: callers may recycle the input frame as
  // soon as trackImage() returns. Reusing the input ROI would make level 0
  // alias a later image while the upper levels still describe the old frame.
  //
  // GFTT deliberately uses image-only pyramids. Retaining the four
  // interleaved CV_16SC2 derivative buffers changes the process heap layout:
  // goodFeaturesToTrack() allocates 752x480 CV_32F temporaries at the heap
  // top, which glibc then trims back on every call (M_TRIM_THRESHOLD),
  // turning a ~1.75 ms detection into ~4.45 ms. calcOpticalFlowPyrLK()
  // accepts image-only pyramids and computes derivatives internally, so the
  // GFTT path keeps pyramid reuse without that allocator pathology.
  cv::buildOpticalFlowPyramid(
    _img, _pyramid, cv::Size(K_KLT_WIN_SIZE, K_KLT_WIN_SIZE), K_KLT_MAX_LEVEL, _with_derivatives,
    cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, false);
}

void calcFlow(
  const vector<cv::Mat> & _prev_pyr, const vector<cv::Mat> & _cur_pyr, const vector<cv::Point2f> & _prev_pts,
  vector<cv::Point2f> & _cur_pts, vector<uchar> & _status, vector<float> & _err, int _max_level, int _flags = 0)
{
  cv::calcOpticalFlowPyrLK(
    _prev_pyr, _cur_pyr, _prev_pts, _cur_pts, _status, _err, cv::Size(K_KLT_WIN_SIZE, K_KLT_WIN_SIZE),
    _max_level, kltTermCriteria(), _flags);
}

float pointDistanceSquared(const cv::Point2f & _a, const cv::Point2f & _b)
{
  const float dx = _a.x - _b.x;
  const float dy = _a.y - _b.y;
  return dx * dx + dy * dy;
}

}  // namespace

// Coarse detection followed by full-resolution refinement. Like OpenVINS'
// downsampling / refinement and SchurVINS' pyramid-level detection, this avoids
// computing a full-resolution corner response. Tracking and calibration stay
// in original pixel coordinates. It is opt-in because the selected points differ.
void FeatureTracker::detectPyramidFeatures(
  const cv::Mat & _img, const cv::Mat & _mask, int _max_features, vector<cv::Point2f> & _points,
  const cv::Mat & _half_image)
{
  if (_img.cols < 24 || _img.rows < 24) {
    cv::goodFeaturesToTrack(_img, _points, _max_features, 0.01, MIN_DIST, _mask);
    return;
  }
  cv::Mat detection_half_image;
  cv::Mat coarse = _half_image;
  if (coarse.empty()) {
    cv::pyrDown(_img, detection_half_image);
    coarse = detection_half_image;
  }
  CV_Assert(
    coarse.type() == CV_8UC1 && coarse.cols == (_img.cols + 1) / 2 && coarse.rows == (_img.rows + 1) / 2);
  // pyrDown samples (2*x, 2*y), including odd image sizes. INTER_NEAREST
  // resize would shift this mapping on odd sizes; explicitly sample instead.
  // The scratch buffer is reused across detections to avoid a per-frame
  // allocation of the half-resolution mask.
  cv::Mat coarse_mask;
  if (!_mask.empty()) {
    CV_Assert(_mask.type() == CV_8UC1 && _mask.size() == _img.size());
    detection_half_mask_scratch_.create(coarse.size(), CV_8UC1);
    for (int y = 0; y < coarse.rows; ++y) {
      const uchar * source = _mask.ptr<uchar>(2 * y);
      uchar * target = detection_half_mask_scratch_.ptr<uchar>(y);
      for (int x = 0; x < coarse.cols; ++x)
        target[x] = source[2 * x];
    }
    coarse_mask = detection_half_mask_scratch_;
  }
  vector<cv::Point2f> candidates;
  cv::goodFeaturesToTrack(coarse, candidates, _max_features, 0.01, std::max(1.0, MIN_DIST * 0.5), coarse_mask);
  for (auto & point : candidates)
    point *= 2.0f;
  if (candidates.empty())
    return;
  cv::cornerSubPix(
    _img, candidates, cv::Size(5, 5), cv::Size(-1, -1),
    cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.01));
  // Refinement can cross a mask boundary or reduce the spacing. Validate in
  // full-resolution coordinates before assigning IDs or adding observations.
  const float min_dist_sq = static_cast<float>(MIN_DIST) * MIN_DIST;
  for (const auto & point : candidates) {
    if (
      !std::isfinite(point.x) || !std::isfinite(point.y) || point.x < 1 || point.y < 1 ||
      point.x >= _img.cols - 1 || point.y >= _img.rows - 1)
      continue;
    if (!_mask.empty() && _mask.at<uchar>(cvRound(point.y), cvRound(point.x)) == 0)
      continue;
    bool separated = true;
    for (const auto & accepted : _points)
      if (pointDistanceSquared(point, accepted) < min_dist_sq) {
        separated = false;
        break;
      }
    if (separated)
      _points.push_back(point);
  }
}

bool FeatureTracker::inBorder(const cv::Point2f & _pt) const
{
  const int BORDER_SIZE = 1;
  int img_x = cvRound(_pt.x);
  int img_y = cvRound(_pt.y);
  return BORDER_SIZE <= img_x && img_x < col_ - BORDER_SIZE && BORDER_SIZE <= img_y &&
         img_y < row_ - BORDER_SIZE;
}

double distance(cv::Point2f _pt1, cv::Point2f _pt2)
{
  //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
  double dx = _pt1.x - _pt2.x;
  double dy = _pt1.y - _pt2.y;
  return sqrt(dx * dx + dy * dy);
}

void reduceVector(vector<cv::Point2f> & _v, const vector<uchar> & _status)
{
  int j = 0;
  for (int i = 0; i < int(_v.size()); i++)
    if (_status[i])
      _v[j++] = _v[i];
  _v.resize(j);
}

void reduceVector(vector<int> & _v, const vector<uchar> & _status)
{
  int j = 0;
  for (int i = 0; i < int(_v.size()); i++)
    if (_status[i])
      _v[j++] = _v[i];
  _v.resize(j);
}

FeatureTracker::FeatureTracker()
{
  stereo_cam_ = 0;
  n_id_ = 0;
  hasPrediction_ = false;
}

void FeatureTracker::detectFeatures(
  const cv::Mat & _img, const cv::Mat & _mask, int _max_features, vector<cv::Point2f> & _points)
{
  _points.clear();
  if (_max_features <= 0 || _img.empty())
    return;

  if (FEATURE_DETECTOR == "fast_grid")
    detectFastGridFeatures(_img, _mask, _max_features, _points);
  else if (FEATURE_DETECTOR == "gftt_pyramid")
    detectPyramidFeatures(_img, _mask, _max_features, _points, cv::Mat());
  else
    cv::goodFeaturesToTrack(_img, _points, _max_features, 0.01, MIN_DIST, _mask);
}

void FeatureTracker::detectFastGridFeatures(
  const cv::Mat & _img, const cv::Mat & _mask, int _max_features, vector<cv::Point2f> & _points)
{
  _points.clear();
  if (_max_features <= 0 || _img.empty() || _img.channels() != 1)
    return;

  const int grid_x = std::max(1, std::min(GRID_X, _img.cols));
  const int grid_y = std::max(1, std::min(GRID_Y, _img.rows));
  const int grid_count = grid_x * grid_y;
  if (grid_count <= 0)
    return;

  // VINS masks use 255 for allowed pixels and 0 for forbidden pixels,
  // matching setMask(). This is the inverse of OpenVINS' Grider_GRID mask.
  const auto mask_allows = [&_mask](const cv::Point2f & _pt)
  {
    if (_mask.empty())
      return true;
    const int x = cvRound(_pt.x);
    const int y = cvRound(_pt.y);
    return x >= 0 && y >= 0 && x < _mask.cols && y < _mask.rows && _mask.at<uchar>(y, x) >= 128;
  };
  const auto response_desc = [](const cv::KeyPoint & _a, const cv::KeyPoint & _b)
  { return _a.response > _b.response; };

  // Keep only the best candidates from each cell. This avoids sorting or
  // refining every FAST corner in the image while still letting sparse
  // cells be backfilled by their next-best candidates.
  vector<vector<cv::KeyPoint>> candidates(grid_count);
  for (int gy = 0; gy < grid_y; ++gy) {
    const int y0 = gy * _img.rows / grid_y;
    const int y1 = (gy + 1) * _img.rows / grid_y;
    for (int gx = 0; gx < grid_x; ++gx) {
      const int x0 = gx * _img.cols / grid_x;
      const int x1 = (gx + 1) * _img.cols / grid_x;
      // cv::FAST needs a 3-pixel border around a candidate.
      if (x1 - x0 < 7 || y1 - y0 < 7)
        continue;

      const cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
      vector<cv::KeyPoint> cell_pts;
      cv::FAST(_img(roi), cell_pts, FAST_THRESHOLD, true);

      vector<cv::KeyPoint> valid;
      valid.reserve(cell_pts.size());
      for (cv::KeyPoint & pt : cell_pts) {
        pt.pt.x += static_cast<float>(x0);
        pt.pt.y += static_cast<float>(y0);
        if (mask_allows(pt.pt))
          valid.push_back(pt);
      }

      const size_t keep = std::min(valid.size(), static_cast<size_t>(_max_features));
      if (keep > 0) {
        std::partial_sort(valid.begin(), valid.begin() + keep, valid.end(), response_desc);
        valid.resize(keep);
      }
      candidates[gy * grid_x + gx] = std::move(valid);
    }
  }

  // Draw one candidate from each cell per round. This distributes the
  // requested features over the image before taking a second point from any
  // already-covered cell.
  vector<size_t> cursor(grid_count, 0);
  vector<cv::Point2f> selected;
  selected.reserve(_max_features);
  const float min_dist = static_cast<float>(MIN_DIST);
  const float min_dist_sq = min_dist * min_dist;
  while (static_cast<int>(selected.size()) < _max_features) {
    bool advanced = false;
    for (int cell = 0; cell < grid_count && static_cast<int>(selected.size()) < _max_features; ++cell) {
      while (cursor[cell] < candidates[cell].size()) {
        const cv::Point2f pt = candidates[cell][cursor[cell]++].pt;
        advanced = true;

        if (!mask_allows(pt))
          continue;
        if (MIN_DIST > 0) {
          bool too_close = false;
          for (const cv::Point2f & chosen : selected) {
            if (pointDistanceSquared(pt, chosen) < min_dist_sq) {
              too_close = true;
              break;
            }
          }
          if (too_close)
            continue;
        }
        selected.push_back(pt);
        break;
      }
    }
    if (!advanced)
      break;
  }

  if (selected.empty())
    return;

  cv::cornerSubPix(
    _img, selected, cv::Size(5, 5), cv::Size(-1, -1),
    cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.001));

  // Refinement can move a point across a mask boundary or closer to another
  // point, so enforce both constraints once more on the final coordinates.
  _points.reserve(selected.size());
  for (const cv::Point2f & pt : selected) {
    if (!mask_allows(pt))
      continue;
    if (MIN_DIST > 0) {
      bool too_close = false;
      for (const cv::Point2f & chosen : _points) {
        if (pointDistanceSquared(pt, chosen) < min_dist_sq) {
          too_close = true;
          break;
        }
      }
      if (too_close)
        continue;
    }
    _points.push_back(pt);
    if (static_cast<int>(_points.size()) >= _max_features)
      break;
  }
}

void FeatureTracker::setMask()
{
  mask_.create(row_, col_, CV_8UC1);
  mask_.setTo(255);

  // prefer to keep features that are tracked for long time
  auto & cnt_pts_id = mask_order_;
  cnt_pts_id.clear();
  cnt_pts_id.reserve(cur_pts_.size());

  for (unsigned int i = 0; i < cur_pts_.size(); i++)
    cnt_pts_id.push_back(make_pair(track_cnt_[i], make_pair(cur_pts_[i], ids_[i])));

  sort(
    cnt_pts_id.begin(), cnt_pts_id.end(),
    [](const pair<int, pair<cv::Point2f, int>> & _a, const pair<int, pair<cv::Point2f, int>> & _b)
    { return _a.first > _b.first; });

  cur_pts_.clear();
  ids_.clear();
  track_cnt_.clear();

  for (auto & it : cnt_pts_id) {
    if (mask_.at<uchar>(it.second.first) == 255) {
      cur_pts_.push_back(it.second.first);
      ids_.push_back(it.second.second);
      track_cnt_.push_back(it.first);
      cv::circle(mask_, it.second.first, MIN_DIST, 0, -1);
    }
  }
}

double FeatureTracker::distance(cv::Point2f & _pt1, cv::Point2f & _pt2)
{
  //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
  double dx = _pt1.x - _pt2.x;
  double dy = _pt1.y - _pt2.y;
  return sqrt(dx * dx + dy * dy);
}

map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> FeatureTracker::trackImage(
  double _cur_time, const cv::Mat & _img, const cv::Mat & _img1, bool _extract_features, bool _draw_tracks)
{
  // Per-stage diagnostics use steady-clock timestamps. The
  // node aggregates these and only logs at a low rate.
  const auto track_start = std::chrono::steady_clock::now();
  FeatureTrackerTimings_s timings;
  int tracked_feature_count = 0;
  int detected_feature_count = 0;
  const auto finish_timings = [&]()
  {
    timings.total_ms_ = elapsedMs(track_start);
    timings.tracked_features_ = tracked_feature_count;
    timings.detected_features_ = detected_feature_count;
    last_timings_ = timings;
  };
  cur_time_ = _cur_time;
  cur_img_ = _img;
  row_ = cur_img_.rows;
  col_ = cur_img_.cols;
  const cv::Mat & rightImg = _img1;
  /*
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(cur_img, cur_img);
        if(!rightImg.empty())
            clahe->apply(rightImg, rightImg);
    }
    */
  const bool same_image_geometry =
    !prev_img_.empty() && prev_img_.size() == cur_img_.size() && prev_img_.type() == cur_img_.type();
  if (!same_image_geometry) {
    prev_pts_.clear();
    ids_.clear();
    track_cnt_.clear();
    prev_un_pts_.clear();
    prev_un_pts_map_.clear();
    cur_un_pts_map_.clear();
    prev_un_right_pts_map_.clear();
    cur_un_right_pts_map_.clear();
    prev_pyr_.clear();
    detection_half_mask_scratch_.release();
  }

  // Coarse GFTT uses quarter-size response buffers; like FAST+grid it avoids
  // the full-resolution allocation regression described above. Cache image
  // derivatives for both forward and backward LK, including the next frame.
  const bool with_derivatives = FEATURE_DETECTOR != "gftt";
  const size_t expected_pyr_mats = static_cast<size_t>(K_KLT_MAX_LEVEL + 1) * (with_derivatives ? 2u : 1u);
  if (!prev_pyr_.empty() && prev_pyr_.size() != expected_pyr_mats) {
    // The detector (and therefore the cache layout) changed at runtime.
    prev_pts_.clear();
    ids_.clear();
    track_cnt_.clear();
    prev_pyr_.clear();
  }

  const auto pyramid_start = std::chrono::steady_clock::now();
  // A detector switch can change the expected interleaved layout. Drop stale
  // derivative Mats before asking OpenCV to reuse the scratch vector.
  if (scratch_pyr_.size() != expected_pyr_mats)
    scratch_pyr_.clear();
  buildFlowPyramid(cur_img_, scratch_pyr_, with_derivatives);
  timings.pyramid_ms_ += elapsedMs(pyramid_start);
  vector<cv::Mat> & cur_pyr = scratch_pyr_;

  // A missing previous pyramid is a cache-state failure rather than a cue to
  // rebuild it from a possibly recycled caller buffer. Restart tracking.
  if (!prev_pts_.empty() && prev_pyr_.empty()) {
    prev_pts_.clear();
    ids_.clear();
    track_cnt_.clear();
  }
  cur_pts_.clear();

  const auto flow_start = std::chrono::steady_clock::now();
  if (!prev_pts_.empty()) {
    TicToc t_o;
    vector<uchar> status;
    vector<float> err;
    if (hasPrediction_) {
      cur_pts_ = predict_pts_;
      calcFlow(prev_pyr_, cur_pyr, prev_pts_, cur_pts_, status, err, 1, cv::OPTFLOW_USE_INITIAL_FLOW);

      int succ_num = 0;
      for (unsigned char statu : status) {
        if (statu)
          succ_num++;
      }
      if (succ_num < 10)
        calcFlow(prev_pyr_, cur_pyr, prev_pts_, cur_pts_, status, err, 3);
    } else
      calcFlow(prev_pyr_, cur_pyr, prev_pts_, cur_pts_, status, err, 3);
    // reverse check
    if (FLOW_BACK) {
      vector<uchar> reverse_status;
      vector<cv::Point2f> reverse_pts = prev_pts_;
      calcFlow(cur_pyr, prev_pyr_, cur_pts_, reverse_pts, reverse_status, err, 1, cv::OPTFLOW_USE_INITIAL_FLOW);
      for (size_t i = 0; i < status.size(); i++) {
        if (status[i] && reverse_status[i] && distance(prev_pts_[i], reverse_pts[i]) <= 0.5) {
          status[i] = 1;
        } else
          status[i] = 0;
      }
    }

    for (int i = 0; i < int(cur_pts_.size()); i++)
      if (status[i] && !inBorder(cur_pts_[i]))
        status[i] = 0;
    reduceVector(prev_pts_, status);
    reduceVector(cur_pts_, status);
    reduceVector(ids_, status);
    reduceVector(track_cnt_, status);
    RCLCPP_DEBUG(vins::logger(), "temporal optical flow costs: %fms", t_o.toc());
    //printf("track cnt %d\n", (int)ids.size());
  }
  timings.flow_ms_ += elapsedMs(flow_start);
  tracked_feature_count = static_cast<int>(cur_pts_.size());

  for (auto & n : track_cnt_)
    n++;

  if (1) {
    const auto detect_start = std::chrono::steady_clock::now();
    if (RANSAC_OUTLIER_REJECTION)
      rejectWithF();
    RCLCPP_DEBUG(vins::logger(), "set mask begins");
    TicToc t_m;
    setMask();
    RCLCPP_DEBUG(vins::logger(), "set mask costs %fms", t_m.toc());

    RCLCPP_DEBUG(vins::logger(), "detect feature begins");
    TicToc t_t;
    const int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts_.size());
    bool should_detect = n_max_cnt > 0 && n_max_cnt > DETECTION_MIN_FEATURE_RATIO * MAX_CNT;
    const int detection_interval = detection_interval_.load(std::memory_order_relaxed);
    if (should_detect && detection_interval > 1) {
      // Default interval 1 keeps the exact legacy behavior. For an
      // opt-in interval > 1, skip routine top-ups on intermediate frames
      // but never starve: a large deficit bypasses the gate.
      const bool interval_due = (frame_counter_ % detection_interval) == 0;
      const bool urgent = n_max_cnt > K_URGENT_DETECTION_DEFICIT_RATIO * MAX_CNT;
      should_detect = interval_due || urgent;
    }
    ++frame_counter_;
    if (should_detect) {
      if (mask_.empty())
        cout << "mask is empty " << '\n';
      if (mask_.type() != CV_8UC1)
        cout << "mask type wrong " << '\n';
      // Interleaved derivative pyramids store the half-size image at 2.
      const cv::Mat half_image =
        FEATURE_DETECTOR == "gftt_pyramid" && cur_pyr.size() > 2 ? cur_pyr[2] : cv::Mat();
      if (FEATURE_DETECTOR == "gftt_pyramid") {
        n_pts_.clear();
        detectPyramidFeatures(cur_img_, mask_, n_max_cnt, n_pts_, half_image);
      } else
        detectFeatures(cur_img_, mask_, n_max_cnt, n_pts_);
    } else
      n_pts_.clear();
    detected_feature_count = static_cast<int>(n_pts_.size());
    RCLCPP_DEBUG(vins::logger(), "detect feature costs: %f ms", t_t.toc());

    for (auto & p : n_pts_) {
      cur_pts_.push_back(p);
      ids_.push_back(n_id_++);
      track_cnt_.push_back(1);
    }
    //printf("feature cnt after add %d\n", (int)ids.size());
    timings.detect_ms_ += elapsedMs(detect_start);
  }

  undistortedPtsInto(cur_pts_, m_camera_[0], cur_un_pts_);
  ptsVelocityInto(ids_, cur_un_pts_, cur_un_pts_map_, prev_un_pts_map_, pts_velocity_);

  if (!_img1.empty() && stereo_cam_) {
    ids_right_.clear();
    cur_right_pts_.clear();
    cur_un_right_pts_.clear();
    right_pts_velocity_.clear();
    cur_un_right_pts_map_.clear();
    if (!cur_pts_.empty()) {
      const auto right_pyramid_start = std::chrono::steady_clock::now();
      if (scratch_right_pyr_.size() != expected_pyr_mats)
        scratch_right_pyr_.clear();
      buildFlowPyramid(rightImg, scratch_right_pyr_, with_derivatives);
      timings.pyramid_ms_ += elapsedMs(right_pyramid_start);
      vector<cv::Mat> & right_pyr = scratch_right_pyr_;
      //printf("stereo image; track feature on right image\n");
      const auto right_flow_start = std::chrono::steady_clock::now();
      vector<cv::Point2f> reverseLeftPts;
      vector<uchar> status, statusRightLeft;
      vector<float> err;
      // cur left ---- cur right
      calcFlow(cur_pyr, right_pyr, cur_pts_, cur_right_pts_, status, err, 3);
      // reverse check cur right ---- cur left
      if (FLOW_BACK) {
        calcFlow(right_pyr, cur_pyr, cur_right_pts_, reverseLeftPts, statusRightLeft, err, 3);
        for (size_t i = 0; i < status.size(); i++) {
          if (status[i] && statusRightLeft[i] && inBorder(cur_right_pts_[i]) && distance(cur_pts_[i], reverseLeftPts[i]) <= 0.5)
            status[i] = 1;
          else
            status[i] = 0;
        }
      }

      ids_right_ = ids_;
      reduceVector(cur_right_pts_, status);
      reduceVector(ids_right_, status);
      // only keep left-right pts
      /*
            reduceVector(cur_pts, status);
            reduceVector(ids, status);
            reduceVector(track_cnt, status);
            reduceVector(cur_un_pts, status);
            reduceVector(pts_velocity, status);
            */
      undistortedPtsInto(cur_right_pts_, m_camera_[1], cur_un_right_pts_);
      ptsVelocityInto(
        ids_right_, cur_un_right_pts_, cur_un_right_pts_map_, prev_un_right_pts_map_, right_pts_velocity_);
      timings.flow_ms_ += elapsedMs(right_flow_start);
    }
    prev_un_right_pts_map_.swap(cur_un_right_pts_map_);
  }
  if (SHOW_TRACK && _draw_tracks)
    drawTrack(cur_img_, rightImg, ids_, cur_pts_, cur_right_pts_, prevLeftPtsMap_);

  // prev_img is retained as a geometry marker; the reusable pixel data and
  // derivatives live in the owned pyramid (caller buffers may be recycled).
  prev_img_ = cur_img_;
  prev_pyr_.swap(cur_pyr);
  // cur_pyr is a reference to scratch_pyr_; after the swap scratch_pyr_
  // holds the old previous-frame buffers ready for reuse by the next build.
  prev_pts_ = cur_pts_;
  prev_un_pts_ = cur_un_pts_;
  prev_un_pts_map_.swap(cur_un_pts_map_);
  prev_time_ = cur_time_;
  hasPrediction_ = false;

  prevLeftPtsMap_.clear();
  // A newly connected viewer gets arrows from its second frame onward.
  if (SHOW_TRACK && _draw_tracks)
    for (size_t i = 0; i < cur_pts_.size(); i++)
      prevLeftPtsMap_.emplace(ids_[i], cur_pts_[i]);

  if (!_extract_features) {
    finish_timings();
    return {};
  }

  map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
  for (size_t i = 0; i < ids_.size(); i++) {
    int feature_id = ids_[i];
    double x, y, z;
    x = cur_un_pts_[i].x;
    y = cur_un_pts_[i].y;
    z = 1;
    double p_u, p_v;
    p_u = cur_pts_[i].x;
    p_v = cur_pts_[i].y;
    int camera_id = 0;
    double velocity_x, velocity_y;
    velocity_x = pts_velocity_[i].x;
    velocity_y = pts_velocity_[i].y;

    Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
    xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
    featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
  }

  if (!_img1.empty() && stereo_cam_) {
    for (size_t i = 0; i < ids_right_.size(); i++) {
      int feature_id = ids_right_[i];
      double x, y, z;
      x = cur_un_right_pts_[i].x;
      y = cur_un_right_pts_[i].y;
      z = 1;
      double p_u, p_v;
      p_u = cur_right_pts_[i].x;
      p_v = cur_right_pts_[i].y;
      int camera_id = 1;
      double velocity_x, velocity_y;
      velocity_x = right_pts_velocity_[i].x;
      velocity_y = right_pts_velocity_[i].y;

      Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
      xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
      featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
    }
  }

  finish_timings();
  return featureFrame;
}

void FeatureTracker::rejectWithF()
{
  if (cur_pts_.size() >= 8) {
    const size_t correspondence_count = cur_pts_.size();
    if (prev_pts_.size() != correspondence_count || ids_.size() != correspondence_count || track_cnt_.size() != correspondence_count) {
      RCLCPP_WARN(
        vins::logger(),
        "Skipping FM RANSAC: inconsistent correspondence sizes "
        "cur=%zu prev=%zu ids=%zu track=%zu",
        cur_pts_.size(), prev_pts_.size(), ids_.size(), track_cnt_.size());
      return;
    }

    RCLCPP_DEBUG(vins::logger(), "FM ransac begins");
    TicToc t_f;
    vector<cv::Point2f> un_cur_pts(correspondence_count), un_prev_pts(correspondence_count);
    for (size_t i = 0; i < correspondence_count; i++) {
      Eigen::Vector3d tmp_p;
      m_camera_[0]->liftProjective(Eigen::Vector2d(cur_pts_[i].x, cur_pts_[i].y), tmp_p);
      tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col_ / 2.0;
      tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row_ / 2.0;
      un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

      m_camera_[0]->liftProjective(Eigen::Vector2d(prev_pts_[i].x, prev_pts_[i].y), tmp_p);
      tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col_ / 2.0;
      tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row_ / 2.0;
      un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
    }

    vector<uchar> status;
    const cv::Mat fundamental =
      cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
    if (fundamental.empty() || status.size() != correspondence_count) {
      RCLCPP_WARN(
        vins::logger(),
        "Skipping FM RANSAC: degenerate model (empty=%d) or status mismatch "
        "(%zu statuses for %zu correspondences)",
        fundamental.empty(), status.size(), correspondence_count);
      return;
    }

    const int size_a = static_cast<int>(correspondence_count);
    reduceVector(prev_pts_, status);
    reduceVector(cur_pts_, status);
    reduceVector(ids_, status);
    reduceVector(track_cnt_, status);
    RCLCPP_DEBUG(
      vins::logger(), "FM ransac: %d -> %lu: %f", size_a, cur_pts_.size(), 1.0 * cur_pts_.size() / size_a);
    RCLCPP_DEBUG(vins::logger(), "FM ransac costs: %fms", t_f.toc());
  }
}

void FeatureTracker::readIntrinsicParameter(const vector<string> & _calib_file)
{
  for (const auto & i : _calib_file) {
    RCLCPP_INFO(vins::logger(), "reading paramerter of camera %s", i.c_str());
    camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(i);
    m_camera_.push_back(camera);
  }
  if (_calib_file.size() == 2)
    stereo_cam_ = 1;
}

void FeatureTracker::showUndistortion([[maybe_unused]] const string & _name)
{
  cv::Mat undistortedImg(row_ + 600, col_ + 600, CV_8UC1, cv::Scalar(0));
  vector<Eigen::Vector2d> distortedp, undistortedp;
  for (int i = 0; i < col_; i++)
    for (int j = 0; j < row_; j++) {
      Eigen::Vector2d a(i, j);
      Eigen::Vector3d b;
      m_camera_[0]->liftProjective(a, b);
      distortedp.push_back(a);
      undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
      //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
    }
  for (int i = 0; i < int(undistortedp.size()); i++) {
    cv::Mat pp(3, 1, CV_32FC1);
    pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col_ / 2;
    pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row_ / 2;
    pp.at<float>(2, 0) = 1.0;
    //cout << trackerData[0].K << endl;
    //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
    //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
    if (
      pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row_ + 600 &&
      pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col_ + 600) {
      undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) =
        cur_img_.at<uchar>(distortedp[i].y(), distortedp[i].x());
    } else {
      //RCLCPP_ERROR(vins::logger(), "(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
    }
  }
  // turn the following code on if you need
  // cv::imshow(name, undistortedImg);
  // cv::waitKey(0);
}

void FeatureTracker::undistortedPtsInto(
  const vector<cv::Point2f> & _pts, const camodocal::CameraPtr & _cam, vector<cv::Point2f> & _out) const
{
  _out.resize(_pts.size());
  for (size_t i = 0; i < _pts.size(); i++) {
    Eigen::Vector2d a(_pts[i].x, _pts[i].y);
    Eigen::Vector3d b;
    _cam->liftProjective(a, b);
    _out[i] = cv::Point2f(b.x() / b.z(), b.y() / b.z());
  }
}

vector<cv::Point2f> FeatureTracker::undistortedPts(vector<cv::Point2f> & _pts, camodocal::CameraPtr _cam)
{
  vector<cv::Point2f> un_pts;
  undistortedPtsInto(_pts, _cam, un_pts);
  return un_pts;
}

void FeatureTracker::ptsVelocityInto(
  const vector<int> & _ids, const vector<cv::Point2f> & _pts, map<int, cv::Point2f> & _cur_id_pts,
  const map<int, cv::Point2f> & _prev_id_pts, vector<cv::Point2f> & _velocity) const
{
  _cur_id_pts.clear();
  for (size_t i = 0; i < _ids.size(); i++) {
    _cur_id_pts.insert(make_pair(_ids[i], _pts[i]));
  }

  // caculate points velocity
  if (!_prev_id_pts.empty()) {
    const double dt = cur_time_ - prev_time_;
    _velocity.resize(_pts.size());
    for (size_t i = 0; i < _pts.size(); i++) {
      const auto it = _prev_id_pts.find(_ids[i]);
      if (it != _prev_id_pts.end()) {
        const double v_x = (_pts[i].x - it->second.x) / dt;
        const double v_y = (_pts[i].y - it->second.y) / dt;
        _velocity[i] = cv::Point2f(v_x, v_y);
      } else
        _velocity[i] = cv::Point2f(0, 0);
    }
  } else {
    // Preserve the legacy buffer length, which used the left-camera point
    // count even when computing right-camera velocities.
    _velocity.assign(cur_pts_.size(), cv::Point2f(0, 0));
  }
}

vector<cv::Point2f> FeatureTracker::ptsVelocity(
  vector<int> & _ids, vector<cv::Point2f> & _pts, map<int, cv::Point2f> & _cur_id_pts,
  map<int, cv::Point2f> & _prev_id_pts)
{
  vector<cv::Point2f> velocity;
  ptsVelocityInto(_ids, _pts, _cur_id_pts, _prev_id_pts, velocity);
  return velocity;
}

void FeatureTracker::drawTrack(
  const cv::Mat & _imLeft, const cv::Mat & _imRight, vector<int> & _curLeftIds,
  vector<cv::Point2f> & _curLeftPts, vector<cv::Point2f> & _curRightPts,
  map<int, cv::Point2f> & _prevLeftPtsMap)
{
  //int rows = imLeft.rows;
  int cols = _imLeft.cols;
  if (!_imRight.empty() && stereo_cam_)
    cv::hconcat(_imLeft, _imRight, imTrack_);
  else
    imTrack_ = _imLeft.clone();
  cv::cvtColor(imTrack_, imTrack_, cv::COLOR_GRAY2RGB);

  for (size_t j = 0; j < _curLeftPts.size(); j++) {
    double len = std::min(1.0, 1.0 * track_cnt_[j] / 20);
    cv::circle(imTrack_, _curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
  }
  if (!_imRight.empty() && stereo_cam_) {
    for (auto rightPt : _curRightPts) {
      rightPt.x += cols;
      cv::circle(imTrack_, rightPt, 2, cv::Scalar(0, 255, 0), 2);
      //cv::Point2f leftPt = curLeftPtsTrackRight[i];
      //cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
    }
  }

  map<int, cv::Point2f>::iterator mapIt;
  for (size_t i = 0; i < _curLeftIds.size(); i++) {
    int id = _curLeftIds[i];
    mapIt = _prevLeftPtsMap.find(id);
    if (mapIt != _prevLeftPtsMap.end()) {
      cv::arrowedLine(imTrack_, _curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
    }
  }

  //draw prediction
  /*
    for(size_t i = 0; i < predict_pts_debug.size(); i++)
    {
        cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
    }
    */
  //printf("predict pts size %d \n", (int)predict_pts_debug.size());

  //cv::Mat imCur2Compress;
  //cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}


void FeatureTracker::setPrediction(map<int, Eigen::Vector3d> & _predictPts)
{
  hasPrediction_ = true;
  predict_pts_.clear();
  predict_pts_debug_.clear();
  map<int, Eigen::Vector3d>::iterator itPredict;
  for (size_t i = 0; i < ids_.size(); i++) {
    //printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
    int id = ids_[i];
    itPredict = _predictPts.find(id);
    if (itPredict != _predictPts.end()) {
      Eigen::Vector2d tmp_uv;
      m_camera_[0]->spaceToPlane(itPredict->second, tmp_uv);
      predict_pts_.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
      predict_pts_debug_.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
    } else
      predict_pts_.push_back(prev_pts_[i]);
  }
}


void FeatureTracker::removeOutliers(set<int> & _removePtsIds)
{
  std::set<int>::iterator itSet;
  vector<uchar> status;
  for (int id : ids_) {
    itSet = _removePtsIds.find(id);
    if (itSet != _removePtsIds.end())
      status.push_back(0);
    else
      status.push_back(1);
  }

  reduceVector(prev_pts_, status);
  reduceVector(ids_, status);
  reduceVector(track_cnt_, status);
}


cv::Mat FeatureTracker::getTrackImage() const { return imTrack_; }