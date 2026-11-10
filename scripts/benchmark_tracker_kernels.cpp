// Standalone OpenCV tracker-kernel microbenchmark.
//
// Build (no CMake required):
//   g++ -std=c++17 -O2 -DNDEBUG scripts/benchmark_tracker_kernels.cpp -o benchmark_tracker_kernels $(pkg-config --cflags --libs opencv4)
//
// The benchmark intentionally does not depend on VINS-Fusion headers.  It
// loads a sorted directory of grayscale PNG/PGM frames, fixes one GFTT point
// set per consecutive image pair, and measures the same six CPU/OpenCV cases:
//
//   threads = 1, 2, 4  x  image-only / derivative-cached KLT pyramids
//
// Image loading is outside every timed region.  The GFTT point set used by all
// LK calls is precomputed from the previous frame of each pair with the exact
// requested parameters.  The timed GFTT call is still measured separately so
// that GFTT cost can be compared without changing the LK input points.
//
// Additional independent timings for cornerMinEigenVal(blockSize=3,
// ksize=3) and dilate(3x3) are reported with reusable output buffers.  These
// are auxiliary measurements only; they are not subtracted from the GFTT
// timing and do not alter the GFTT measurement scope.

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace
{
constexpr int K_MAX_IMAGES = 120;
constexpr int K_DEFAULT_ROUNDS = 2;
constexpr int K_DEFAULT_WARMUP_ROUNDS = 2;

constexpr int K_GFTT_MAX_CORNERS = 150;
constexpr double K_GFTT_QUALITY_LEVEL = 0.01;
constexpr int K_GFTT_MIN_DISTANCE = 25;
constexpr int K_GFTT_BLOCK_SIZE = 3;
constexpr bool K_GFTT_USE_HARRIS = false;
constexpr double K_GFTT_HARRIS_K = 0.04;

constexpr int K_KLT_WIN_SIZE = 21;
constexpr int K_PYRAMID_MAX_LEVEL = 3;
constexpr int K_FORWARD_MAX_LEVEL = 3;
constexpr int K_REVERSE_MAX_LEVEL = 1;
constexpr int K_TERM_MAX_COUNT = 30;
constexpr double K_TERM_EPSILON = 0.01;

constexpr int K_CORNER_BLOCK_SIZE = 3;
constexpr int K_CORNER_K_SIZE = 3;

struct Options_s {
  fs::path image_dir_;
  int max_images_ = K_MAX_IMAGES;
  int rounds_ = K_DEFAULT_ROUNDS;
  int warmup_rounds_ = K_DEFAULT_WARMUP_ROUNDS;
  bool request_opencl_ = false;
  std::string format_ = "json";
  std::string output_path_;
};

struct OpenClInfo_s {
  bool requested_ = false;
  bool have_opencl_ = false;
  bool context_initialized_ = false;
  bool device_available_ = false;
  bool runtime_enabled_ = false;
  std::string device_name_;
  std::string device_vendor_;
  std::string device_version_;
  std::string device_type_;
  std::string error_;
};

struct InputInfo_s {
  std::string directory_;
  std::size_t images_available_ = 0;
  std::size_t images_used_ = 0;
  bool truncated_ = false;
  std::size_t pair_count_ = 0;
  int width_ = 0;
  int height_ = 0;
  std::string first_image_;
  std::string last_image_;
};

struct StageSamples_s {
  std::vector<double> milliseconds_;
  std::vector<int> valid_points_;
  std::size_t skipped_empty_points_ = 0;

  void add(double _ms, int _valid = -1)
  {
    milliseconds_.push_back(_ms);
    if (_valid >= 0)
      valid_points_.push_back(_valid);
  }
};

struct Summary_s {
  std::size_t count_ = 0;
  double mean_ = 0.0;
  double p50_ = 0.0;
  double p95_ = 0.0;
  double min_ = 0.0;
  double max_ = 0.0;
};

struct CountSummary_s {
  std::size_t count_ = 0;
  double mean_ = 0.0;
  double p50_ = 0.0;
  double p95_ = 0.0;
  int min_ = 0;
  int max_ = 0;
};

struct PointCompare_s {
  std::size_t samples_ = 0;
  std::size_t count_mismatch_samples_ = 0;
  int max_abs_count_delta_ = 0;
  std::size_t comparisons_ = 0;
  double max_coordinate_diff_px_ = 0.0;
  double mean_coordinate_diff_px_ = 0.0;
};

struct FlowCompare_s {
  std::size_t samples_ = 0;
  std::size_t status_mismatch_total_ = 0;
  std::size_t status_mismatch_samples_ = 0;
  std::int64_t valid_count_delta_total_ = 0;
  double valid_count_delta_mean_ = 0.0;
  std::size_t coordinate_comparisons_ = 0;
  double max_coordinate_diff_px_ = 0.0;
  double mean_coordinate_diff_px_ = 0.0;
  double p95_coordinate_diff_px_ = 0.0;
  std::vector<double> coordinate_diffs_;
};

struct StageResult_s {
  StageSamples_s samples_;
  PointCompare_s gftt_reference_check_;
  FlowCompare_s baseline_check_;
};

struct CaseConfig_s {
  std::string name_;
  int threads_ = 1;
  bool derivative_pyramid_ = false;
};

struct FlowOutput_s {
  std::vector<cv::Point2f> points_;
  std::vector<uchar> status_;
};

struct BaselineSample_s {
  FlowOutput_s forward_;
  FlowOutput_s reverse_;
};

struct CaseRuntime_s {
  std::vector<cv::Mat> previous_pyramid_;
  std::vector<cv::Mat> current_pyramid_;

  cv::Mat response_;
  cv::Mat dilated_;
  cv::Mat dilate_kernel_;

  std::vector<cv::Point2f> gftt_points_;
  std::vector<cv::Point2f> forward_points_;
  std::vector<uchar> forward_status_;
  std::vector<float> forward_error_;
  std::vector<cv::Point2f> reverse_points_;
  std::vector<uchar> reverse_status_;
  std::vector<float> reverse_error_;

  CaseRuntime_s() : dilate_kernel_(cv::Mat::ones(3, 3, CV_8U)) {}
};

struct CaseResult_s {
  CaseConfig_s config_;
  int actual_threads_ = 0;
  int previous_pyramid_levels_ = 0;
  int current_pyramid_levels_ = 0;
  std::size_t previous_pyramid_mats_ = 0;
  std::size_t current_pyramid_mats_ = 0;
  StageResult_s gftt_;
  StageResult_s pyramid_;
  StageResult_s lk_forward_;
  StageResult_s lk_reverse_;
  StageResult_s corner_min_eigenval_;
  StageResult_s dilate_3x3_;
};

struct BenchmarkResult_s {
  Options_s options_;
  OpenClInfo_s opencl_;
  InputInfo_s input_;
  std::vector<CaseResult_s> cases_;
  std::string baseline_case_name_;
};

double elapsedMilliseconds(
  const std::chrono::steady_clock::time_point & _start, const std::chrono::steady_clock::time_point & _end)
{
  return std::chrono::duration<double, std::milli>(_end - _start).count();
}

Summary_s summarize(const std::vector<double> & _values)
{
  Summary_s out;
  out.count_ = _values.size();
  if (_values.empty())
    return out;

  out.min_ = _values.front();
  out.max_ = _values.front();
  double sum = 0.0;
  for (double value : _values) {
    sum += value;
    out.min_ = std::min(out.min_, value);
    out.max_ = std::max(out.max_, value);
  }
  out.mean_ = sum / static_cast<double>(_values.size());

  std::vector<double> sorted = _values;
  std::sort(sorted.begin(), sorted.end());
  const auto percentile = [&sorted](double _p)
  {
    if (sorted.empty())
      return 0.0;
    const double pos = _p * static_cast<double>(sorted.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(pos));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(pos));
    if (lower == upper)
      return sorted[lower];
    const double fraction = pos - static_cast<double>(lower);
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
  };
  out.p50_ = percentile(0.50);
  out.p95_ = percentile(0.95);
  return out;
}

CountSummary_s summarizeCounts(const std::vector<int> & _values)
{
  CountSummary_s out;
  out.count_ = _values.size();
  if (_values.empty())
    return out;

  out.min_ = _values.front();
  out.max_ = _values.front();
  double sum = 0.0;
  for (int value : _values) {
    sum += static_cast<double>(value);
    out.min_ = std::min(out.min_, value);
    out.max_ = std::max(out.max_, value);
  }
  out.mean_ = sum / static_cast<double>(_values.size());

  std::vector<int> sorted = _values;
  std::sort(sorted.begin(), sorted.end());
  const auto percentile = [&sorted](double _p)
  {
    if (sorted.empty())
      return 0.0;
    const double pos = _p * static_cast<double>(sorted.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(pos));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(pos));
    if (lower == upper)
      return static_cast<double>(sorted[lower]);
    const double fraction = pos - static_cast<double>(lower);
    return static_cast<double>(sorted[lower]) * (1.0 - fraction) +
           static_cast<double>(sorted[upper]) * fraction;
  };
  out.p50_ = percentile(0.50);
  out.p95_ = percentile(0.95);
  return out;
}

double pointDistance(const cv::Point2f & _a, const cv::Point2f & _b)
{
  const double dx = static_cast<double>(_a.x) - static_cast<double>(_b.x);
  const double dy = static_cast<double>(_a.y) - static_cast<double>(_b.y);
  return std::sqrt(dx * dx + dy * dy);
}

int countValid(const std::vector<uchar> & _status)
{
  return static_cast<int>(
    std::count_if(_status.begin(), _status.end(), [](uchar _value) { return _value != 0; }));
}

void comparePoints(
  const std::vector<cv::Point2f> & _actual, const std::vector<cv::Point2f> & _reference,
  PointCompare_s & _aggregate)
{
  ++_aggregate.samples_;
  const int count_delta = static_cast<int>(_actual.size()) - static_cast<int>(_reference.size());
  _aggregate.max_abs_count_delta_ = std::max(_aggregate.max_abs_count_delta_, std::abs(count_delta));
  if (count_delta != 0)
    ++_aggregate.count_mismatch_samples_;

  const std::size_t count = std::min(_actual.size(), _reference.size());
  double sum = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const double diff = pointDistance(_actual[i], _reference[i]);
    sum += diff;
    _aggregate.max_coordinate_diff_px_ = std::max(_aggregate.max_coordinate_diff_px_, diff);
    ++_aggregate.comparisons_;
  }
  if (_aggregate.comparisons_ != 0) {
    const double previous_total =
      _aggregate.mean_coordinate_diff_px_ * static_cast<double>(_aggregate.comparisons_ - count);
    _aggregate.mean_coordinate_diff_px_ = (previous_total + sum) / static_cast<double>(_aggregate.comparisons_);
  }
}

void compareFlow(
  const std::vector<cv::Point2f> & _actual_points, const std::vector<uchar> & _actual_status,
  const FlowOutput_s & _baseline, FlowCompare_s & _aggregate)
{
  ++_aggregate.samples_;

  const int actual_valid = countValid(_actual_status);
  const int baseline_valid = countValid(_baseline.status_);
  const std::int64_t delta =
    static_cast<std::int64_t>(actual_valid) - static_cast<std::int64_t>(baseline_valid);
  _aggregate.valid_count_delta_total_ += delta;

  std::size_t mismatches = 0;
  const std::size_t status_count = std::max(_actual_status.size(), _baseline.status_.size());
  for (std::size_t i = 0; i < status_count; ++i) {
    const bool actual_ok = i < _actual_status.size() && _actual_status[i] != 0;
    const bool baseline_ok = i < _baseline.status_.size() && _baseline.status_[i] != 0;
    if (actual_ok != baseline_ok)
      ++mismatches;
  }
  _aggregate.status_mismatch_total_ += mismatches;
  if (mismatches != 0)
    ++_aggregate.status_mismatch_samples_;

  const std::size_t coordinate_count = std::min(
    {_actual_points.size(), _actual_status.size(), _baseline.points_.size(), _baseline.status_.size()});
  for (std::size_t i = 0; i < coordinate_count; ++i) {
    if (!_actual_status[i] || !_baseline.status_[i])
      continue;
    const double diff = pointDistance(_actual_points[i], _baseline.points_[i]);
    _aggregate.coordinate_diffs_.push_back(diff);
    _aggregate.max_coordinate_diff_px_ = std::max(_aggregate.max_coordinate_diff_px_, diff);
  }
}

void finalizeFlowCompare(FlowCompare_s & _compare)
{
  if (_compare.samples_ != 0)
    _compare.valid_count_delta_mean_ =
      static_cast<double>(_compare.valid_count_delta_total_) / static_cast<double>(_compare.samples_);

  _compare.coordinate_comparisons_ = _compare.coordinate_diffs_.size();
  if (!_compare.coordinate_diffs_.empty()) {
    double sum = 0.0;
    for (double diff : _compare.coordinate_diffs_)
      sum += diff;
    _compare.mean_coordinate_diff_px_ = sum / static_cast<double>(_compare.coordinate_diffs_.size());
    _compare.p95_coordinate_diff_px_ = summarize(_compare.coordinate_diffs_).p95_;
  }
}

std::vector<fs::path> findImages(const fs::path & _directory)
{
  std::error_code ec;
  if (!fs::is_directory(_directory, ec) || ec)
    throw std::runtime_error("image directory does not exist or is not a directory: " + _directory.string());

  std::vector<fs::path> images;
  for (const auto & entry : fs::directory_iterator(_directory, ec)) {
    if (ec)
      throw std::runtime_error("failed to scan image directory: " + ec.message());
    if (!entry.is_regular_file(ec) || ec)
      continue;

    std::string extension = entry.path().extension().string();
    std::transform(
      extension.begin(), extension.end(), extension.begin(),
      [](unsigned char _c) { return static_cast<char>(std::tolower(_c)); });
    if (extension == ".png" || extension == ".pgm")
      images.push_back(entry.path());
  }
  if (ec)
    throw std::runtime_error("failed to scan image directory: " + ec.message());

  std::sort(
    images.begin(), images.end(),
    [](const fs::path & _a, const fs::path & _b)
    {
      const std::string a_name = _a.filename().string();
      const std::string b_name = _b.filename().string();
      if (a_name != b_name)
        return a_name < b_name;
      return _a.generic_string() < _b.generic_string();
    });
  return images;
}

std::vector<cv::Mat> loadImages(const std::vector<fs::path> & _paths)
{
  std::vector<cv::Mat> images;
  images.reserve(_paths.size());
  for (const auto & path : _paths) {
    cv::Mat image = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
    if (image.empty())
      throw std::runtime_error("failed to read grayscale image: " + path.string());
    if (image.type() != CV_8UC1)
      throw std::runtime_error("image is not CV_8UC1 after grayscale load: " + path.string());
    images.push_back(std::move(image));
  }
  return images;
}

std::vector<std::vector<cv::Point2f>> computeReferencePoints(const std::vector<cv::Mat> & _images)
{
  cv::setNumThreads(1);
  std::vector<std::vector<cv::Point2f>> points;
  points.reserve(_images.size() - 1);
  for (std::size_t i = 0; i + 1 < _images.size(); ++i) {
    std::vector<cv::Point2f> pair_points;
    pair_points.reserve(K_GFTT_MAX_CORNERS);
    cv::goodFeaturesToTrack(
      _images[i], pair_points, K_GFTT_MAX_CORNERS, K_GFTT_QUALITY_LEVEL, K_GFTT_MIN_DISTANCE);
    points.push_back(std::move(pair_points));
  }
  return points;
}

std::string deviceTypeName(int _type)
{
  if (_type & cv::ocl::Device::TYPE_GPU)
    return "GPU";
  if (_type & cv::ocl::Device::TYPE_CPU)
    return "CPU";
  if (_type & cv::ocl::Device::TYPE_ACCELERATOR)
    return "ACCELERATOR";
  return "OTHER";
}

OpenClInfo_s inspectOpenCL(bool _requested)
{
  OpenClInfo_s info;
  info.requested_ = _requested;
  info.have_opencl_ = cv::ocl::haveOpenCL();
  if (info.have_opencl_) {
    try {
      cv::ocl::Context & context = cv::ocl::Context::getDefault(true);
      info.context_initialized_ = context.ptr() != nullptr;
      if (info.context_initialized_ && context.ndevices() > 0) {
        cv::ocl::Device device = cv::ocl::Device::getDefault();
        if (!device.empty()) {
          info.device_available_ = device.available();
          info.device_name_ = device.name();
          info.device_vendor_ = device.vendorName();
          info.device_version_ = device.version();
          info.device_type_ = deviceTypeName(device.type());
        }
      }
    } catch (const std::exception & e) {
      info.error_ = e.what();
    }
  }

  const bool can_enable = info.have_opencl_ && info.device_available_;
  cv::ocl::setUseOpenCL(_requested && can_enable);
  // This is a runtime setting only. All measured kernels receive Mat (or
  // vector<Mat>) inputs/outputs and therefore execute on the CPU.
  info.runtime_enabled_ = cv::ocl::useOpenCL();
  return info;
}

void runPair(
  CaseRuntime_s & _runtime, const cv::Mat & _previous, const cv::Mat & _current,
  const std::vector<cv::Point2f> & _fixed_points, const CaseConfig_s & _config, bool _record,
  CaseResult_s & _result)
{
  const cv::Size win(K_KLT_WIN_SIZE, K_KLT_WIN_SIZE);
  const cv::TermCriteria criteria(
    cv::TermCriteria::COUNT + cv::TermCriteria::EPS, K_TERM_MAX_COUNT, K_TERM_EPSILON);

  // The GFTT measurement is intentionally independent of the fixed point set
  // used by LK.  This keeps every LK case on exactly the same input points
  // while still measuring GFTT under the current cv::setNumThreads() value.
  _runtime.gftt_points_.clear();
  _runtime.gftt_points_.reserve(K_GFTT_MAX_CORNERS);
  auto start = std::chrono::steady_clock::now();
  cv::goodFeaturesToTrack(
    _previous, _runtime.gftt_points_, K_GFTT_MAX_CORNERS, K_GFTT_QUALITY_LEVEL, K_GFTT_MIN_DISTANCE);
  auto end = std::chrono::steady_clock::now();
  const double gftt_ms = elapsedMilliseconds(start, end);

  if (_record) {
    _result.gftt_.samples_.add(gftt_ms, static_cast<int>(_runtime.gftt_points_.size()));
    comparePoints(_runtime.gftt_points_, _fixed_points, _result.gftt_.gftt_reference_check_);
  }

  // Reuse the same response/dilation buffers across all samples.  These are
  // auxiliary measurements and do not alter the GFTT timing above.
  _runtime.response_.create(_previous.size(), CV_32F);
  _runtime.dilated_.create(_previous.size(), CV_32F);

  start = std::chrono::steady_clock::now();
  cv::cornerMinEigenVal(_previous, _runtime.response_, K_CORNER_BLOCK_SIZE, K_CORNER_K_SIZE, cv::BORDER_DEFAULT);
  end = std::chrono::steady_clock::now();
  if (_record)
    _result.corner_min_eigenval_.samples_.add(elapsedMilliseconds(start, end));

  start = std::chrono::steady_clock::now();
  cv::dilate(_runtime.response_, _runtime.dilated_, _runtime.dilate_kernel_);
  end = std::chrono::steady_clock::now();
  if (_record)
    _result.dilate_3x3_.samples_.add(elapsedMilliseconds(start, end));

  // One timed sample builds BOTH images of a pair. Production caching
  // builds only the new image each frame; this is not that per-frame cost.
  // Persistent vectors let OpenCV reuse level Mat allocations between pairs.
  start = std::chrono::steady_clock::now();
  const int previous_levels = cv::buildOpticalFlowPyramid(
    _previous, _runtime.previous_pyramid_, win, K_PYRAMID_MAX_LEVEL, _config.derivative_pyramid_,
    cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, false);
  const int current_levels = cv::buildOpticalFlowPyramid(
    _current, _runtime.current_pyramid_, win, K_PYRAMID_MAX_LEVEL, _config.derivative_pyramid_,
    cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, false);
  end = std::chrono::steady_clock::now();
  if (_record)
    _result.pyramid_.samples_.add(elapsedMilliseconds(start, end));

  if (_record) {
    _result.previous_pyramid_levels_ = previous_levels;
    _result.current_pyramid_levels_ = current_levels;
    _result.previous_pyramid_mats_ = _runtime.previous_pyramid_.size();
    _result.current_pyramid_mats_ = _runtime.current_pyramid_.size();
  }

  // Clear both directions before checking for empty input so a cornerless
  // pair cannot reuse the preceding pair's flow or reach LK's point assert.
  _runtime.forward_points_.clear();
  _runtime.forward_status_.clear();
  _runtime.forward_error_.clear();
  _runtime.reverse_points_.clear();
  _runtime.reverse_status_.clear();
  _runtime.reverse_error_.clear();
  if (_fixed_points.empty()) {
    if (_record) {
      ++_result.lk_forward_.samples_.skipped_empty_points_;
      ++_result.lk_reverse_.samples_.skipped_empty_points_;
      _result.lk_forward_.samples_.valid_points_.push_back(0);
      _result.lk_reverse_.samples_.valid_points_.push_back(0);
    }
    return;  // No timing sample: a skipped call must not look like zero-cost LK.
  }

  // Forward LK: previous -> current, maxLevel=3, no initial-flow flag.
  _runtime.forward_points_.reserve(_fixed_points.size());
  _runtime.forward_status_.reserve(_fixed_points.size());
  _runtime.forward_error_.reserve(_fixed_points.size());
  start = std::chrono::steady_clock::now();
  cv::calcOpticalFlowPyrLK(
    _runtime.previous_pyramid_, _runtime.current_pyramid_, _fixed_points, _runtime.forward_points_,
    _runtime.forward_status_, _runtime.forward_error_, win, K_FORWARD_MAX_LEVEL, criteria, 0);
  end = std::chrono::steady_clock::now();
  const double forward_ms = elapsedMilliseconds(start, end);

  // Reverse LK: current -> previous, maxLevel=1.  The output is initialized
  // with the original fixed points (not the forward result), and
  // OPTFLOW_USE_INITIAL_FLOW is set, matching the tracker's reverse-check
  // convention.  prevPts are the forward-tracked points in the current frame.
  _runtime.reverse_points_.assign(_fixed_points.begin(), _fixed_points.end());
  _runtime.reverse_status_.reserve(_fixed_points.size());
  _runtime.reverse_error_.reserve(_fixed_points.size());
  start = std::chrono::steady_clock::now();
  cv::calcOpticalFlowPyrLK(
    _runtime.current_pyramid_, _runtime.previous_pyramid_, _runtime.forward_points_, _runtime.reverse_points_,
    _runtime.reverse_status_, _runtime.reverse_error_, win, K_REVERSE_MAX_LEVEL, criteria,
    cv::OPTFLOW_USE_INITIAL_FLOW);
  end = std::chrono::steady_clock::now();
  const double reverse_ms = elapsedMilliseconds(start, end);

  if (_record) {
    _result.lk_forward_.samples_.add(forward_ms, countValid(_runtime.forward_status_));
    _result.lk_reverse_.samples_.add(reverse_ms, countValid(_runtime.reverse_status_));
  }
}

void runCase(
  const std::vector<cv::Mat> & _images, const std::vector<std::vector<cv::Point2f>> & _fixed_points,
  const CaseConfig_s & _config, int _rounds, int _warmup_rounds, bool _baseline,
  std::vector<BaselineSample_s> & _baseline_samples, CaseResult_s & _result)
{
  cv::setNumThreads(_config.threads_);
  _result.config_ = _config;
  _result.actual_threads_ = cv::getNumThreads();

  CaseRuntime_s runtime;
  const std::size_t pair_count = _images.size() - 1;

  // Warmup rounds are full passes over the image pairs.  They use the same
  // persistent buffers as the measured passes but do not contribute samples.
  for (int warmup = 0; warmup < _warmup_rounds; ++warmup) {
    for (std::size_t pair = 0; pair < pair_count; ++pair) {
      runPair(runtime, _images[pair], _images[pair + 1], _fixed_points[pair], _config, false, _result);
    }
  }

  // Store the first case's outputs per (round, pair); all cases compare
  // against those outputs outside the timed regions, including empty pairs.
  for (int round = 0; round < _rounds; ++round) {
    for (std::size_t pair = 0; pair < pair_count; ++pair) {
      runPair(runtime, _images[pair], _images[pair + 1], _fixed_points[pair], _config, true, _result);

      const std::size_t sample_index = static_cast<std::size_t>(round) * pair_count + pair;
      BaselineSample_s & sample = _baseline_samples[sample_index];
      if (_baseline) {
        sample.forward_.points_ = runtime.forward_points_;
        sample.forward_.status_ = runtime.forward_status_;
        sample.reverse_.points_ = runtime.reverse_points_;
        sample.reverse_.status_ = runtime.reverse_status_;
      }
      compareFlow(
        runtime.forward_points_, runtime.forward_status_, sample.forward_, _result.lk_forward_.baseline_check_);
      compareFlow(
        runtime.reverse_points_, runtime.reverse_status_, sample.reverse_, _result.lk_reverse_.baseline_check_);
    }
  }

  finalizeFlowCompare(_result.lk_forward_.baseline_check_);
  finalizeFlowCompare(_result.lk_reverse_.baseline_check_);
}

std::string jsonEscape(const std::string & _value)
{
  std::ostringstream out;
  for (unsigned char c : _value) {
    switch (c) {
    case '"':
      out << "\\\"";
      break;
    case '\\':
      out << "\\\\";
      break;
    case '\b':
      out << "\\b";
      break;
    case '\f':
      out << "\\f";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      if (c < 0x20)
        out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec
            << std::setfill(' ');
      else
        out << static_cast<char>(c);
      break;
    }
  }
  return out.str();
}

std::string quote(const std::string & _value) { return "\"" + jsonEscape(_value) + "\""; }

void writeCountSummary(std::ostream & _out, const CountSummary_s & _summary)
{
  _out << "{\"count\":" << _summary.count_ << ",\"mean\":" << _summary.mean_ << ",\"p50\":" << _summary.p50_
       << ",\"p95\":" << _summary.p95_ << ",\"min\":" << _summary.min_ << ",\"max\":" << _summary.max_ << "}";
}

void writePointCompare(std::ostream & _out, const PointCompare_s & _compare)
{
  _out << "{\"samples\":" << _compare.samples_
       << ",\"count_mismatch_samples\":" << _compare.count_mismatch_samples_
       << ",\"max_abs_count_delta\":" << _compare.max_abs_count_delta_
       << ",\"coordinate_comparisons\":" << _compare.comparisons_
       << ",\"max_coordinate_diff_px\":" << _compare.max_coordinate_diff_px_
       << ",\"mean_coordinate_diff_px\":" << _compare.mean_coordinate_diff_px_ << "}";
}

void writeFlowCompare(std::ostream & _out, const FlowCompare_s & _compare)
{
  _out << "{\"samples\":" << _compare.samples_
       << ",\"status_mismatch_total\":" << _compare.status_mismatch_total_
       << ",\"status_mismatch_samples\":" << _compare.status_mismatch_samples_
       << ",\"valid_count_delta_total\":" << _compare.valid_count_delta_total_
       << ",\"valid_count_delta_mean\":" << _compare.valid_count_delta_mean_
       << ",\"coordinate_comparisons\":" << _compare.coordinate_comparisons_
       << ",\"max_coordinate_diff_px\":" << _compare.max_coordinate_diff_px_
       << ",\"mean_coordinate_diff_px\":" << _compare.mean_coordinate_diff_px_
       << ",\"p95_coordinate_diff_px\":" << _compare.p95_coordinate_diff_px_ << "}";
}

void writeStage(
  std::ostream & _out, const StageResult_s & _stage, bool _has_valid_points, bool _has_reference_check,
  bool _has_baseline_check)
{
  const Summary_s timing = summarize(_stage.samples_.milliseconds_);
  _out << "{\"samples\":" << timing.count_
       << ",\"skipped_empty_points\":" << _stage.samples_.skipped_empty_points_;
  if (timing.count_ != 0)
    _out << ",\"mean_ms\":" << timing.mean_ << ",\"p50_ms\":" << timing.p50_ << ",\"p95_ms\":" << timing.p95_
         << ",\"min_ms\":" << timing.min_ << ",\"max_ms\":" << timing.max_;
  else
    _out << ",\"mean_ms\":null,\"p50_ms\":null,\"p95_ms\":null,\"min_ms\":null,\"max_ms\":null";
  _out << ",\"valid_points\":";
  if (_has_valid_points && !_stage.samples_.valid_points_.empty())
    writeCountSummary(_out, summarizeCounts(_stage.samples_.valid_points_));
  else
    _out << "null";

  if (_has_reference_check) {
    _out << ",\"reference_check\":";
    writePointCompare(_out, _stage.gftt_reference_check_);
  }
  if (_has_baseline_check) {
    _out << ",\"baseline_check\":";
    writeFlowCompare(_out, _stage.baseline_check_);
  }
  _out << "}";
}

void writeJson(std::ostream & _out, const BenchmarkResult_s & _result)
{
  _out << std::setprecision(12);
  _out << "{\n";
  _out << "  \"schema\":\"tracker_kernels_microbench_v1\",\n";
  _out << "  \"opencv_version\":" << quote(CV_VERSION) << ",\n";
  _out << "  \"opencl\":{\"requested\":" << (_result.opencl_.requested_ ? "true" : "false")
       << ",\"have_opencl\":" << (_result.opencl_.have_opencl_ ? "true" : "false")
       << ",\"context_initialized\":" << (_result.opencl_.context_initialized_ ? "true" : "false")
       << ",\"device_available\":" << (_result.opencl_.device_available_ ? "true" : "false")
       << ",\"runtime_enabled\":" << (_result.opencl_.runtime_enabled_ ? "true" : "false")
       << ",\"kernels_use_opencl\":false" << ",\"execution_backend\":\"cpu_mat\""
       << ",\"device_name\":" << quote(_result.opencl_.device_name_)
       << ",\"device_vendor\":" << quote(_result.opencl_.device_vendor_)
       << ",\"device_version\":" << quote(_result.opencl_.device_version_)
       << ",\"device_type\":" << quote(_result.opencl_.device_type_)
       << ",\"error\":" << quote(_result.opencl_.error_) << "},\n";
  _out << "  \"input\":{\"directory\":" << quote(_result.input_.directory_)
       << ",\"images_available\":" << _result.input_.images_available_
       << ",\"images_used\":" << _result.input_.images_used_
       << ",\"truncated\":" << (_result.input_.truncated_ ? "true" : "false")
       << ",\"pair_count\":" << _result.input_.pair_count_ << ",\"width\":" << _result.input_.width_
       << ",\"height\":" << _result.input_.height_ << ",\"first_image\":" << quote(_result.input_.first_image_)
       << ",\"last_image\":" << quote(_result.input_.last_image_) << "},\n";
  _out << "  \"method\":{\"rounds\":" << _result.options_.rounds_
       << ",\"warmup_rounds\":" << _result.options_.warmup_rounds_
       << ",\"max_images\":" << _result.options_.max_images_
       << ",\"baseline_case\":" << quote(_result.baseline_case_name_)
       << ",\"percentile\":\"linear interpolation over sorted samples\""
       << ",\"empty_fixed_points\":\"skip both LK calls; timing excludes skips; valid_points includes zeros\""
       << ",\"gftt\":{\"max_corners\":" << K_GFTT_MAX_CORNERS << ",\"quality_level\":" << K_GFTT_QUALITY_LEVEL
       << ",\"min_distance\":" << K_GFTT_MIN_DISTANCE << ",\"block_size\":" << K_GFTT_BLOCK_SIZE
       << ",\"use_harris\":" << (K_GFTT_USE_HARRIS ? "true" : "false") << ",\"k\":" << K_GFTT_HARRIS_K << "}"
       << ",\"pyramid\":{\"win_size\":" << K_KLT_WIN_SIZE << ",\"max_level\":" << K_PYRAMID_MAX_LEVEL
       << ",\"try_reuse_input_image\":false" << ",\"images_built_per_sample\":2"
       << ",\"timing_scope\":\"previous_and_current_images_per_pair\""
       << ",\"production_cached_images_built_per_frame\":1}"
       << ",\"lk_forward\":{\"max_level\":" << K_FORWARD_MAX_LEVEL << ",\"criteria\":\"COUNT+EPS,"
       << K_TERM_MAX_COUNT << "," << K_TERM_EPSILON << "\",\"flags\":0}"
       << ",\"lk_reverse\":{\"max_level\":" << K_REVERSE_MAX_LEVEL << ",\"criteria\":\"COUNT+EPS,"
       << K_TERM_MAX_COUNT << "," << K_TERM_EPSILON << "\",\"initial\":\"original_fixed_points\""
       << ",\"flags\":\"OPTFLOW_USE_INITIAL_FLOW\"}"
       << ",\"auxiliary\":{\"corner_min_eigenval\":{\"block_size\":" << K_CORNER_BLOCK_SIZE
       << ",\"ksize\":" << K_CORNER_K_SIZE << "},\"dilate_3x3\":true}" << "},\n";
  _out << "  \"cases\":[\n";
  for (std::size_t i = 0; i < _result.cases_.size(); ++i) {
    const CaseResult_s & case_result = _result.cases_[i];
    _out << "    {\"name\":" << quote(case_result.config_.name_)
         << ",\"threads\":" << case_result.config_.threads_
         << ",\"actual_threads\":" << case_result.actual_threads_
         << ",\"derivative_pyramid\":" << (case_result.config_.derivative_pyramid_ ? "true" : "false")
         << ",\"previous_pyramid_levels\":" << case_result.previous_pyramid_levels_
         << ",\"current_pyramid_levels\":" << case_result.current_pyramid_levels_
         << ",\"previous_pyramid_mats\":" << case_result.previous_pyramid_mats_
         << ",\"current_pyramid_mats\":" << case_result.current_pyramid_mats_ << ",\"stages\":{";

    _out << "\"gftt\":";
    writeStage(_out, case_result.gftt_, true, true, false);
    _out << ",\"pyramid\":";
    writeStage(_out, case_result.pyramid_, false, false, false);
    _out << ",\"lk_forward\":";
    writeStage(_out, case_result.lk_forward_, true, false, true);
    _out << ",\"lk_reverse\":";
    writeStage(_out, case_result.lk_reverse_, true, false, true);
    _out << ",\"corner_min_eigenval\":";
    writeStage(_out, case_result.corner_min_eigenval_, false, false, false);
    _out << ",\"dilate_3x3\":";
    writeStage(_out, case_result.dilate_3x3_, false, false, false);
    _out << "}}";
    if (i + 1 < _result.cases_.size())
      _out << ",";
    _out << "\n";
  }
  _out << "  ]\n}\n";
}

std::string csvEscape(const std::string & _value)
{
  if (_value.find_first_of(",\"\n\r") == std::string::npos)
    return _value;
  std::string escaped = "\"";
  for (char c : _value) {
    if (c == '"')
      escaped += "\"\"";
    else
      escaped += c;
  }
  escaped += "\"";
  return escaped;
}

void writeCsvStage(
  std::ostream & _out, const BenchmarkResult_s & _result, const CaseResult_s & _case_result,
  const std::string & _stage_name, const StageResult_s & _stage, bool _has_valid_points,
  bool _has_reference_check, bool _has_baseline_check)
{
  const Summary_s timing = summarize(_stage.samples_.milliseconds_);
  const CountSummary_s counts = summarizeCounts(_stage.samples_.valid_points_);
  const PointCompare_s & point_compare = _stage.gftt_reference_check_;
  const FlowCompare_s & flow_compare = _stage.baseline_check_;

  _out << csvEscape(_case_result.config_.name_) << ',' << _case_result.config_.threads_ << ','
       << (_case_result.config_.derivative_pyramid_ ? 1 : 0) << ',' << _stage_name << ',' << timing.count_
       << ',' << _stage.samples_.skipped_empty_points_ << ',';
  if (timing.count_ != 0)
    _out << timing.mean_ << ',' << timing.p50_ << ',' << timing.p95_ << ',' << timing.min_ << ','
         << timing.max_;
  else
    _out << ",,,,";
  _out << ',';
  if (_has_valid_points)
    _out << counts.mean_ << ',' << counts.p50_ << ',' << counts.p95_ << ',' << counts.min_ << ','
         << counts.max_;
  else
    _out << ",,,,";
  _out << ',';
  if (_has_reference_check)
    _out << point_compare.count_mismatch_samples_ << ',' << point_compare.max_abs_count_delta_ << ','
         << point_compare.max_coordinate_diff_px_;
  else
    _out << ",,";
  _out << ',';
  if (_has_baseline_check)
    _out << flow_compare.status_mismatch_total_ << ',' << flow_compare.status_mismatch_samples_ << ','
         << flow_compare.valid_count_delta_total_ << ',' << flow_compare.valid_count_delta_mean_ << ','
         << flow_compare.coordinate_comparisons_ << ',' << flow_compare.max_coordinate_diff_px_ << ','
         << flow_compare.p95_coordinate_diff_px_;
  else
    _out << ",,,,,,";
  _out << ',' << _result.input_.images_used_ << ',' << _result.input_.pair_count_ << ','
       << _result.options_.rounds_ << ',' << _result.options_.warmup_rounds_ << ','
       << (_result.opencl_.requested_ ? 1 : 0) << ',' << (_result.opencl_.have_opencl_ ? 1 : 0) << ','
       << (_result.opencl_.device_available_ ? 1 : 0) << ',' << (_result.opencl_.runtime_enabled_ ? 1 : 0)
       << ',' << csvEscape(_result.opencl_.device_name_)
       << ",cpu_mat,0,2,previous_and_current_images_per_pair,1" << '\n';
}

void writeCsv(std::ostream & _out, const BenchmarkResult_s & _result)
{
  _out << "case,threads,derivative_pyramid,stage,samples,skipped_empty_points,mean_ms,p50_ms,p95_ms,min_ms,"
          "max_ms,"
          "valid_points_mean,valid_points_p50,valid_points_p95,valid_points_min,valid_points_max,"
          "reference_count_mismatch_samples,reference_max_abs_count_delta,reference_max_coordinate_diff_px,"
          "status_mismatch_total,status_mismatch_samples,valid_count_delta_total,valid_count_delta_mean,"
          "coordinate_comparisons,max_coordinate_diff_px,coordinate_diff_p95_px,"
          "images_used,pair_count,rounds,warmup_rounds,opencl_requested,opencl_have,opencl_device_available,"
          "opencl_runtime_enabled,opencl_device,execution_backend,kernels_use_opencl,"
          "pyramid_images_built_per_sample,pyramid_timing_scope,production_cached_images_built_per_frame\n";
  for (const CaseResult_s & case_result : _result.cases_) {
    writeCsvStage(_out, _result, case_result, "gftt", case_result.gftt_, true, true, false);
    writeCsvStage(_out, _result, case_result, "pyramid", case_result.pyramid_, false, false, false);
    writeCsvStage(_out, _result, case_result, "lk_forward", case_result.lk_forward_, true, false, true);
    writeCsvStage(_out, _result, case_result, "lk_reverse", case_result.lk_reverse_, true, false, true);
    writeCsvStage(
      _out, _result, case_result, "corner_min_eigenval", case_result.corner_min_eigenval_, false, false, false);
    writeCsvStage(_out, _result, case_result, "dilate_3x3", case_result.dilate_3x3_, false, false, false);
  }
}

void printUsage(const char * _program)
{
  std::cerr << "Usage: " << _program << " [options] <image-directory>\n"
            << "\n"
            << "Options:\n"
            << "  --max-images N       Maximum images to load (default 120, hard max 120)\n"
            << "  --rounds N           Measured rounds (default 2)\n"
            << "  --warmup-rounds N    Warmup rounds per case (default 2)\n"
            << "  --format json|csv    Output format (default json)\n"
            << "  --output PATH        Write output to PATH instead of stdout\n"
            << "  --opencl             Enable OpenCV OpenCL runtime if available (diagnostics only;\n"
            << "                       all timed kernels use Mat/vector<Mat> on CPU, not GPU)\n"
            << "  -h, --help           Show this help\n"
            << "\nPyramid timing builds BOTH images per pair; production caching builds only\n"
            << "one new image per frame. Empty fixed points skip both LK calls; skipped\n"
            << "calls have no timing sample, but contribute zero to valid-point counts.\n";
}

Options_s parseOptions(int _argc, char ** _argv)
{
  Options_s options;
  bool have_directory = false;

  for (int i = 1; i < _argc; ++i) {
    const std::string arg = _argv[i];
    const auto requireValue = [&](const std::string & _name) -> std::string
    {
      if (i + 1 >= _argc)
        throw std::runtime_error(_name + " requires a value");
      return _argv[++i];
    };

    if (arg == "-h" || arg == "--help") {
      printUsage(_argv[0]);
      std::exit(0);
    } else if (arg == "--max-images")
      options.max_images_ = std::stoi(requireValue(arg));
    else if (arg == "--rounds")
      options.rounds_ = std::stoi(requireValue(arg));
    else if (arg == "--warmup-rounds")
      options.warmup_rounds_ = std::stoi(requireValue(arg));
    else if (arg == "--format")
      options.format_ = requireValue(arg);
    else if (arg == "--output")
      options.output_path_ = requireValue(arg);
    else if (arg == "--opencl")
      options.request_opencl_ = true;
    else if (!arg.empty() && arg[0] == '-')
      throw std::runtime_error("unknown option: " + arg);
    else if (!have_directory) {
      options.image_dir_ = arg;
      have_directory = true;
    } else
      throw std::runtime_error("unexpected extra positional argument: " + arg);
  }

  if (!have_directory)
    throw std::runtime_error("missing image directory");
  if (options.max_images_ < 2 || options.max_images_ > K_MAX_IMAGES)
    throw std::runtime_error("--max-images must be in [2, 120]");
  if (options.rounds_ < 1)
    throw std::runtime_error("--rounds must be >= 1");
  if (options.warmup_rounds_ < 0)
    throw std::runtime_error("--warmup-rounds must be >= 0");
  if (options.format_ != "json" && options.format_ != "csv")
    throw std::runtime_error("--format must be json or csv");
  return options;
}

BenchmarkResult_s runBenchmark(const Options_s & _options)
{
  BenchmarkResult_s result;
  result.options_ = _options;
  result.opencl_ = inspectOpenCL(_options.request_opencl_);

  std::cerr << "execution_backend=cpu_mat; OpenCL runtime_enabled=" << result.opencl_.runtime_enabled_
            << " (runtime flag only; no GPU kernels)\n"
            << "pyramid timing: two images per pair; production cache: one new image per frame\n";

  std::vector<fs::path> all_images = findImages(_options.image_dir_);
  result.input_.directory_ = _options.image_dir_.string();
  result.input_.images_available_ = all_images.size();
  if (all_images.size() > static_cast<std::size_t>(_options.max_images_)) {
    all_images.resize(static_cast<std::size_t>(_options.max_images_));
    result.input_.truncated_ = true;
  }
  if (all_images.size() < 2)
    throw std::runtime_error("need at least two grayscale PNG/PGM images");

  result.input_.images_used_ = all_images.size();
  result.input_.pair_count_ = all_images.size() - 1;
  result.input_.first_image_ = all_images.front().filename().string();
  result.input_.last_image_ = all_images.back().filename().string();

  std::cerr << "loading " << all_images.size() << " images from " << _options.image_dir_ << "\n";
  std::vector<cv::Mat> images = loadImages(all_images);
  result.input_.width_ = images.front().cols;
  result.input_.height_ = images.front().rows;
  for (const cv::Mat & image : images) {
    if (image.size() != images.front().size())
      throw std::runtime_error("all benchmark images must have the same size");
  }

  std::cerr << "fixing GFTT points for " << result.input_.pair_count_ << " image pairs\n";
  const std::vector<std::vector<cv::Point2f>> fixed_points = computeReferencePoints(images);

  std::vector<CaseConfig_s> configs = {
    {"threads1_image_only", 1, false}, {"threads1_derivatives", 1, true}, {"threads2_image_only", 2, false},
    {"threads2_derivatives", 2, true}, {"threads4_image_only", 4, false}, {"threads4_derivatives", 4, true},
  };
  result.baseline_case_name_ = configs.front().name_;

  const std::size_t baseline_count = static_cast<std::size_t>(_options.rounds_) * result.input_.pair_count_;
  std::vector<BaselineSample_s> baseline_samples(baseline_count);
  result.cases_.reserve(configs.size());

  for (std::size_t i = 0; i < configs.size(); ++i) {
    const bool baseline = i == 0;
    result.cases_.emplace_back();
    CaseResult_s & case_result = result.cases_.back();
    std::cerr << "running " << configs[i].name_ << " (warmup=" << _options.warmup_rounds_
              << ", measured=" << _options.rounds_ << ")\n";
    runCase(
      images, fixed_points, configs[i], _options.rounds_, _options.warmup_rounds_, baseline, baseline_samples,
      case_result);
  }

  cv::setNumThreads(1);
  return result;
}

void writeOutput(const BenchmarkResult_s & _result)
{
  std::ofstream file;
  std::ostream * out = &std::cout;
  if (!_result.options_.output_path_.empty()) {
    file.open(_result.options_.output_path_, std::ios::out | std::ios::trunc);
    if (!file)
      throw std::runtime_error("failed to open output file: " + _result.options_.output_path_);
    out = &file;
  }

  if (_result.options_.format_ == "csv")
    writeCsv(*out, _result);
  else
    writeJson(*out, _result);
  out->flush();
  if (!*out)
    throw std::runtime_error("failed to write benchmark output");
}

}  // namespace

int main(int argc, char ** argv)
{
  try {
    const Options_s options = parseOptions(argc, argv);
    const BenchmarkResult_s result = runBenchmark(options);
    writeOutput(result);
    return 0;
  } catch (const std::exception & e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
