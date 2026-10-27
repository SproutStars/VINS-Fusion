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
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
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
constexpr int kMaxImages = 120;
constexpr int kDefaultRounds = 2;
constexpr int kDefaultWarmupRounds = 2;

constexpr int kGfttMaxCorners = 150;
constexpr double kGfttQualityLevel = 0.01;
constexpr int kGfttMinDistance = 25;
constexpr int kGfttBlockSize = 3;
constexpr bool kGfttUseHarris = false;
constexpr double kGfttHarrisK = 0.04;

constexpr int kKltWinSize = 21;
constexpr int kPyramidMaxLevel = 3;
constexpr int kForwardMaxLevel = 3;
constexpr int kReverseMaxLevel = 1;
constexpr int kTermMaxCount = 30;
constexpr double kTermEpsilon = 0.01;

constexpr int kCornerBlockSize = 3;
constexpr int kCornerKSize = 3;

struct Options
{
    fs::path image_dir;
    int max_images = kMaxImages;
    int rounds = kDefaultRounds;
    int warmup_rounds = kDefaultWarmupRounds;
    bool request_opencl = false;
    std::string format = "json";
    std::string output_path;
};

struct OpenCLInfo
{
    bool requested = false;
    bool have_opencl = false;
    bool context_initialized = false;
    bool device_available = false;
    bool runtime_enabled = false;
    std::string device_name;
    std::string device_vendor;
    std::string device_version;
    std::string device_type;
    std::string error;
};

struct InputInfo
{
    std::string directory;
    std::size_t images_available = 0;
    std::size_t images_used = 0;
    bool truncated = false;
    std::size_t pair_count = 0;
    int width = 0;
    int height = 0;
    std::string first_image;
    std::string last_image;
};

struct StageSamples
{
    std::vector<double> milliseconds;
    std::vector<int> valid_points;
    std::size_t skipped_empty_points = 0;

    void add(double ms, int valid = -1)
    {
        milliseconds.push_back(ms);
        if (valid >= 0)
            valid_points.push_back(valid);
    }
};

struct Summary
{
    std::size_t count = 0;
    double mean = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double min = 0.0;
    double max = 0.0;
};

struct CountSummary
{
    std::size_t count = 0;
    double mean = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    int min = 0;
    int max = 0;
};

struct PointCompare
{
    std::size_t samples = 0;
    std::size_t count_mismatch_samples = 0;
    int max_abs_count_delta = 0;
    std::size_t comparisons = 0;
    double max_coordinate_diff_px = 0.0;
    double mean_coordinate_diff_px = 0.0;
};

struct FlowCompare
{
    std::size_t samples = 0;
    std::size_t status_mismatch_total = 0;
    std::size_t status_mismatch_samples = 0;
    std::int64_t valid_count_delta_total = 0;
    double valid_count_delta_mean = 0.0;
    std::size_t coordinate_comparisons = 0;
    double max_coordinate_diff_px = 0.0;
    double mean_coordinate_diff_px = 0.0;
    double p95_coordinate_diff_px = 0.0;
    std::vector<double> coordinate_diffs;
};

struct StageResult
{
    StageSamples samples;
    PointCompare gftt_reference_check;
    FlowCompare baseline_check;
};

struct CaseConfig
{
    std::string name;
    int threads = 1;
    bool derivative_pyramid = false;
};

struct FlowOutput
{
    std::vector<cv::Point2f> points;
    std::vector<uchar> status;
};

struct BaselineSample
{
    FlowOutput forward;
    FlowOutput reverse;
};

struct CaseRuntime
{
    std::vector<cv::Mat> previous_pyramid;
    std::vector<cv::Mat> current_pyramid;

    cv::Mat response;
    cv::Mat dilated;
    cv::Mat dilate_kernel;

    std::vector<cv::Point2f> gftt_points;
    std::vector<cv::Point2f> forward_points;
    std::vector<uchar> forward_status;
    std::vector<float> forward_error;
    std::vector<cv::Point2f> reverse_points;
    std::vector<uchar> reverse_status;
    std::vector<float> reverse_error;

    CaseRuntime()
        : dilate_kernel(cv::Mat::ones(3, 3, CV_8U))
    {
    }
};

struct CaseResult
{
    CaseConfig config;
    int actual_threads = 0;
    int previous_pyramid_levels = 0;
    int current_pyramid_levels = 0;
    std::size_t previous_pyramid_mats = 0;
    std::size_t current_pyramid_mats = 0;
    StageResult gftt;
    StageResult pyramid;
    StageResult lk_forward;
    StageResult lk_reverse;
    StageResult corner_min_eigenval;
    StageResult dilate_3x3;
};

struct BenchmarkResult
{
    Options options;
    OpenCLInfo opencl;
    InputInfo input;
    std::vector<CaseResult> cases;
    std::string baseline_case_name;
};

double elapsedMilliseconds(const std::chrono::steady_clock::time_point &start,
                           const std::chrono::steady_clock::time_point &end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

Summary summarize(const std::vector<double> &values)
{
    Summary out;
    out.count = values.size();
    if (values.empty())
        return out;

    out.min = values.front();
    out.max = values.front();
    double sum = 0.0;
    for (double value : values)
    {
        sum += value;
        out.min = std::min(out.min, value);
        out.max = std::max(out.max, value);
    }
    out.mean = sum / static_cast<double>(values.size());

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const auto percentile = [&sorted](double p) {
        if (sorted.empty())
            return 0.0;
        const double pos = p * static_cast<double>(sorted.size() - 1);
        const std::size_t lower = static_cast<std::size_t>(std::floor(pos));
        const std::size_t upper = static_cast<std::size_t>(std::ceil(pos));
        if (lower == upper)
            return sorted[lower];
        const double fraction = pos - static_cast<double>(lower);
        return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
    };
    out.p50 = percentile(0.50);
    out.p95 = percentile(0.95);
    return out;
}

CountSummary summarizeCounts(const std::vector<int> &values)
{
    CountSummary out;
    out.count = values.size();
    if (values.empty())
        return out;

    out.min = values.front();
    out.max = values.front();
    double sum = 0.0;
    for (int value : values)
    {
        sum += static_cast<double>(value);
        out.min = std::min(out.min, value);
        out.max = std::max(out.max, value);
    }
    out.mean = sum / static_cast<double>(values.size());

    std::vector<int> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const auto percentile = [&sorted](double p) {
        if (sorted.empty())
            return 0.0;
        const double pos = p * static_cast<double>(sorted.size() - 1);
        const std::size_t lower = static_cast<std::size_t>(std::floor(pos));
        const std::size_t upper = static_cast<std::size_t>(std::ceil(pos));
        if (lower == upper)
            return static_cast<double>(sorted[lower]);
        const double fraction = pos - static_cast<double>(lower);
        return static_cast<double>(sorted[lower]) * (1.0 - fraction) +
               static_cast<double>(sorted[upper]) * fraction;
    };
    out.p50 = percentile(0.50);
    out.p95 = percentile(0.95);
    return out;
}

double pointDistance(const cv::Point2f &a, const cv::Point2f &b)
{
    const double dx = static_cast<double>(a.x) - static_cast<double>(b.x);
    const double dy = static_cast<double>(a.y) - static_cast<double>(b.y);
    return std::sqrt(dx * dx + dy * dy);
}

int countValid(const std::vector<uchar> &status)
{
    return static_cast<int>(std::count_if(status.begin(), status.end(),
                                         [](uchar value) { return value != 0; }));
}

void comparePoints(const std::vector<cv::Point2f> &actual,
                   const std::vector<cv::Point2f> &reference,
                   PointCompare &aggregate)
{
    ++aggregate.samples;
    const int count_delta = static_cast<int>(actual.size()) -
                            static_cast<int>(reference.size());
    aggregate.max_abs_count_delta =
        std::max(aggregate.max_abs_count_delta, std::abs(count_delta));
    if (count_delta != 0)
        ++aggregate.count_mismatch_samples;

    const std::size_t count = std::min(actual.size(), reference.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i)
    {
        const double diff = pointDistance(actual[i], reference[i]);
        sum += diff;
        aggregate.max_coordinate_diff_px =
            std::max(aggregate.max_coordinate_diff_px, diff);
        ++aggregate.comparisons;
    }
    if (aggregate.comparisons != 0)
    {
        const double previous_total = aggregate.mean_coordinate_diff_px *
                                      static_cast<double>(aggregate.comparisons - count);
        aggregate.mean_coordinate_diff_px =
            (previous_total + sum) / static_cast<double>(aggregate.comparisons);
    }
}

void compareFlow(const std::vector<cv::Point2f> &actual_points,
                 const std::vector<uchar> &actual_status,
                 const FlowOutput &baseline,
                 FlowCompare &aggregate)
{
    ++aggregate.samples;

    const int actual_valid = countValid(actual_status);
    const int baseline_valid = countValid(baseline.status);
    const std::int64_t delta = static_cast<std::int64_t>(actual_valid) -
                               static_cast<std::int64_t>(baseline_valid);
    aggregate.valid_count_delta_total += delta;

    std::size_t mismatches = 0;
    const std::size_t status_count = std::max(actual_status.size(), baseline.status.size());
    for (std::size_t i = 0; i < status_count; ++i)
    {
        const bool actual_ok = i < actual_status.size() && actual_status[i] != 0;
        const bool baseline_ok = i < baseline.status.size() && baseline.status[i] != 0;
        if (actual_ok != baseline_ok)
            ++mismatches;
    }
    aggregate.status_mismatch_total += mismatches;
    if (mismatches != 0)
        ++aggregate.status_mismatch_samples;

    const std::size_t coordinate_count =
        std::min({actual_points.size(), actual_status.size(),
                  baseline.points.size(), baseline.status.size()});
    for (std::size_t i = 0; i < coordinate_count; ++i)
    {
        if (!actual_status[i] || !baseline.status[i])
            continue;
        const double diff = pointDistance(actual_points[i], baseline.points[i]);
        aggregate.coordinate_diffs.push_back(diff);
        aggregate.max_coordinate_diff_px =
            std::max(aggregate.max_coordinate_diff_px, diff);
    }
}

void finalizeFlowCompare(FlowCompare &compare)
{
    if (compare.samples != 0)
        compare.valid_count_delta_mean =
            static_cast<double>(compare.valid_count_delta_total) /
            static_cast<double>(compare.samples);

    compare.coordinate_comparisons = compare.coordinate_diffs.size();
    if (!compare.coordinate_diffs.empty())
    {
        double sum = 0.0;
        for (double diff : compare.coordinate_diffs)
            sum += diff;
        compare.mean_coordinate_diff_px =
            sum / static_cast<double>(compare.coordinate_diffs.size());
        compare.p95_coordinate_diff_px =
            summarize(compare.coordinate_diffs).p95;
    }
}

std::vector<fs::path> findImages(const fs::path &directory)
{
    std::error_code ec;
    if (!fs::is_directory(directory, ec) || ec)
        throw std::runtime_error("image directory does not exist or is not a directory: " +
                                 directory.string());

    std::vector<fs::path> images;
    for (const auto &entry : fs::directory_iterator(directory, ec))
    {
        if (ec)
            throw std::runtime_error("failed to scan image directory: " + ec.message());
        if (!entry.is_regular_file(ec) || ec)
            continue;

        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (extension == ".png" || extension == ".pgm")
            images.push_back(entry.path());
    }
    if (ec)
        throw std::runtime_error("failed to scan image directory: " + ec.message());

    std::sort(images.begin(), images.end(),
              [](const fs::path &a, const fs::path &b) {
                  const std::string a_name = a.filename().string();
                  const std::string b_name = b.filename().string();
                  if (a_name != b_name)
                      return a_name < b_name;
                  return a.generic_string() < b.generic_string();
              });
    return images;
}

std::vector<cv::Mat> loadImages(const std::vector<fs::path> &paths)
{
    std::vector<cv::Mat> images;
    images.reserve(paths.size());
    for (const auto &path : paths)
    {
        cv::Mat image = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
        if (image.empty())
            throw std::runtime_error("failed to read grayscale image: " + path.string());
        if (image.type() != CV_8UC1)
            throw std::runtime_error("image is not CV_8UC1 after grayscale load: " +
                                     path.string());
        images.push_back(std::move(image));
    }
    return images;
}

std::vector<std::vector<cv::Point2f>> computeReferencePoints(
    const std::vector<cv::Mat> &images)
{
    cv::setNumThreads(1);
    std::vector<std::vector<cv::Point2f>> points;
    points.reserve(images.size() - 1);
    for (std::size_t i = 0; i + 1 < images.size(); ++i)
    {
        std::vector<cv::Point2f> pair_points;
        pair_points.reserve(kGfttMaxCorners);
        cv::goodFeaturesToTrack(images[i], pair_points, kGfttMaxCorners,
                                kGfttQualityLevel, kGfttMinDistance);
        points.push_back(std::move(pair_points));
    }
    return points;
}

std::string deviceTypeName(int type)
{
    if (type & cv::ocl::Device::TYPE_GPU)
        return "GPU";
    if (type & cv::ocl::Device::TYPE_CPU)
        return "CPU";
    if (type & cv::ocl::Device::TYPE_ACCELERATOR)
        return "ACCELERATOR";
    return "OTHER";
}

OpenCLInfo inspectOpenCL(bool requested)
{
    OpenCLInfo info;
    info.requested = requested;
    info.have_opencl = cv::ocl::haveOpenCL();
    if (info.have_opencl)
    {
        try
        {
            cv::ocl::Context &context = cv::ocl::Context::getDefault(true);
            info.context_initialized = context.ptr() != nullptr;
            if (info.context_initialized && context.ndevices() > 0)
            {
                cv::ocl::Device device = cv::ocl::Device::getDefault();
                if (!device.empty())
                {
                    info.device_available = device.available();
                    info.device_name = device.name();
                    info.device_vendor = device.vendorName();
                    info.device_version = device.version();
                    info.device_type = deviceTypeName(device.type());
                }
            }
        }
        catch (const std::exception &e)
        {
            info.error = e.what();
        }
    }

    const bool can_enable = info.have_opencl && info.device_available;
    cv::ocl::setUseOpenCL(requested && can_enable);
    // This is a runtime setting only. All measured kernels receive Mat (or
    // vector<Mat>) inputs/outputs and therefore execute on the CPU.
    info.runtime_enabled = cv::ocl::useOpenCL();
    return info;
}

void runPair(CaseRuntime &runtime,
             const cv::Mat &previous,
             const cv::Mat &current,
             const std::vector<cv::Point2f> &fixed_points,
             const CaseConfig &config,
             bool record,
             CaseResult &result)
{
    const cv::Size win(kKltWinSize, kKltWinSize);
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                    kTermMaxCount, kTermEpsilon);

    // The GFTT measurement is intentionally independent of the fixed point set
    // used by LK.  This keeps every LK case on exactly the same input points
    // while still measuring GFTT under the current cv::setNumThreads() value.
    runtime.gftt_points.clear();
    runtime.gftt_points.reserve(kGfttMaxCorners);
    auto start = std::chrono::steady_clock::now();
    cv::goodFeaturesToTrack(previous, runtime.gftt_points, kGfttMaxCorners,
                            kGfttQualityLevel, kGfttMinDistance);
    auto end = std::chrono::steady_clock::now();
    const double gftt_ms = elapsedMilliseconds(start, end);

    if (record)
    {
        result.gftt.samples.add(gftt_ms, static_cast<int>(runtime.gftt_points.size()));
        comparePoints(runtime.gftt_points, fixed_points,
                      result.gftt.gftt_reference_check);
    }

    // Reuse the same response/dilation buffers across all samples.  These are
    // auxiliary measurements and do not alter the GFTT timing above.
    runtime.response.create(previous.size(), CV_32F);
    runtime.dilated.create(previous.size(), CV_32F);

    start = std::chrono::steady_clock::now();
    cv::cornerMinEigenVal(previous, runtime.response, kCornerBlockSize,
                          kCornerKSize, cv::BORDER_DEFAULT);
    end = std::chrono::steady_clock::now();
    if (record)
        result.corner_min_eigenval.samples.add(elapsedMilliseconds(start, end));

    start = std::chrono::steady_clock::now();
    cv::dilate(runtime.response, runtime.dilated, runtime.dilate_kernel);
    end = std::chrono::steady_clock::now();
    if (record)
        result.dilate_3x3.samples.add(elapsedMilliseconds(start, end));

    // One timed sample builds BOTH images of a pair. Production caching
    // builds only the new image each frame; this is not that per-frame cost.
    // Persistent vectors let OpenCV reuse level Mat allocations between pairs.
    start = std::chrono::steady_clock::now();
    const int previous_levels = cv::buildOpticalFlowPyramid(
        previous, runtime.previous_pyramid, win, kPyramidMaxLevel,
        config.derivative_pyramid, cv::BORDER_REFLECT_101,
        cv::BORDER_CONSTANT, false);
    const int current_levels = cv::buildOpticalFlowPyramid(
        current, runtime.current_pyramid, win, kPyramidMaxLevel,
        config.derivative_pyramid, cv::BORDER_REFLECT_101,
        cv::BORDER_CONSTANT, false);
    end = std::chrono::steady_clock::now();
    if (record)
        result.pyramid.samples.add(elapsedMilliseconds(start, end));

    if (record)
    {
        result.previous_pyramid_levels = previous_levels;
        result.current_pyramid_levels = current_levels;
        result.previous_pyramid_mats = runtime.previous_pyramid.size();
        result.current_pyramid_mats = runtime.current_pyramid.size();
    }

    // Clear both directions before checking for empty input so a cornerless
    // pair cannot reuse the preceding pair's flow or reach LK's point assert.
    runtime.forward_points.clear();
    runtime.forward_status.clear();
    runtime.forward_error.clear();
    runtime.reverse_points.clear();
    runtime.reverse_status.clear();
    runtime.reverse_error.clear();
    if (fixed_points.empty())
    {
        if (record)
        {
            ++result.lk_forward.samples.skipped_empty_points;
            ++result.lk_reverse.samples.skipped_empty_points;
            result.lk_forward.samples.valid_points.push_back(0);
            result.lk_reverse.samples.valid_points.push_back(0);
        }
        return;  // No timing sample: a skipped call must not look like zero-cost LK.
    }

    // Forward LK: previous -> current, maxLevel=3, no initial-flow flag.
    runtime.forward_points.reserve(fixed_points.size());
    runtime.forward_status.reserve(fixed_points.size());
    runtime.forward_error.reserve(fixed_points.size());
    start = std::chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(runtime.previous_pyramid, runtime.current_pyramid,
                             fixed_points, runtime.forward_points,
                             runtime.forward_status, runtime.forward_error,
                             win, kForwardMaxLevel, criteria, 0);
    end = std::chrono::steady_clock::now();
    const double forward_ms = elapsedMilliseconds(start, end);

    // Reverse LK: current -> previous, maxLevel=1.  The output is initialized
    // with the original fixed points (not the forward result), and
    // OPTFLOW_USE_INITIAL_FLOW is set, matching the tracker's reverse-check
    // convention.  prevPts are the forward-tracked points in the current frame.
    runtime.reverse_points.assign(fixed_points.begin(), fixed_points.end());
    runtime.reverse_status.reserve(fixed_points.size());
    runtime.reverse_error.reserve(fixed_points.size());
    start = std::chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(runtime.current_pyramid, runtime.previous_pyramid,
                             runtime.forward_points, runtime.reverse_points,
                             runtime.reverse_status, runtime.reverse_error,
                             win, kReverseMaxLevel, criteria,
                             cv::OPTFLOW_USE_INITIAL_FLOW);
    end = std::chrono::steady_clock::now();
    const double reverse_ms = elapsedMilliseconds(start, end);

    if (record)
    {
        result.lk_forward.samples.add(
            forward_ms, countValid(runtime.forward_status));
        result.lk_reverse.samples.add(
            reverse_ms, countValid(runtime.reverse_status));
    }
}

void runCase(const std::vector<cv::Mat> &images,
             const std::vector<std::vector<cv::Point2f>> &fixed_points,
             const CaseConfig &config,
             int rounds,
             int warmup_rounds,
             bool baseline,
             std::vector<BaselineSample> &baseline_samples,
             CaseResult &result)
{
    cv::setNumThreads(config.threads);
    result.config = config;
    result.actual_threads = cv::getNumThreads();

    CaseRuntime runtime;
    const std::size_t pair_count = images.size() - 1;

    // Warmup rounds are full passes over the image pairs.  They use the same
    // persistent buffers as the measured passes but do not contribute samples.
    for (int warmup = 0; warmup < warmup_rounds; ++warmup)
    {
        for (std::size_t pair = 0; pair < pair_count; ++pair)
        {
            runPair(runtime, images[pair], images[pair + 1],
                    fixed_points[pair], config, false, result);
        }
    }

    // Store the first case's outputs per (round, pair); all cases compare
    // against those outputs outside the timed regions, including empty pairs.
    for (int round = 0; round < rounds; ++round)
    {
        for (std::size_t pair = 0; pair < pair_count; ++pair)
        {
            runPair(runtime, images[pair], images[pair + 1],
                    fixed_points[pair], config, true, result);

            const std::size_t sample_index =
                static_cast<std::size_t>(round) * pair_count + pair;
            BaselineSample &sample = baseline_samples[sample_index];
            if (baseline)
            {
                sample.forward.points = runtime.forward_points;
                sample.forward.status = runtime.forward_status;
                sample.reverse.points = runtime.reverse_points;
                sample.reverse.status = runtime.reverse_status;
            }
            compareFlow(runtime.forward_points, runtime.forward_status,
                        sample.forward, result.lk_forward.baseline_check);
            compareFlow(runtime.reverse_points, runtime.reverse_status,
                        sample.reverse, result.lk_reverse.baseline_check);
        }
    }

    finalizeFlowCompare(result.lk_forward.baseline_check);
    finalizeFlowCompare(result.lk_reverse.baseline_check);
}

std::string jsonEscape(const std::string &value)
{
    std::ostringstream out;
    for (unsigned char c : value)
    {
        switch (c)
        {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20)
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(c) << std::dec << std::setfill(' ');
                else
                    out << static_cast<char>(c);
                break;
        }
    }
    return out.str();
}

std::string quote(const std::string &value)
{
    return "\"" + jsonEscape(value) + "\"";
}

void writeCountSummary(std::ostream &out, const CountSummary &summary)
{
    out << "{\"count\":" << summary.count
        << ",\"mean\":" << summary.mean
        << ",\"p50\":" << summary.p50
        << ",\"p95\":" << summary.p95
        << ",\"min\":" << summary.min
        << ",\"max\":" << summary.max << "}";
}

void writePointCompare(std::ostream &out, const PointCompare &compare)
{
    out << "{\"samples\":" << compare.samples
        << ",\"count_mismatch_samples\":" << compare.count_mismatch_samples
        << ",\"max_abs_count_delta\":" << compare.max_abs_count_delta
        << ",\"coordinate_comparisons\":" << compare.comparisons
        << ",\"max_coordinate_diff_px\":" << compare.max_coordinate_diff_px
        << ",\"mean_coordinate_diff_px\":" << compare.mean_coordinate_diff_px
        << "}";
}

void writeFlowCompare(std::ostream &out, const FlowCompare &compare)
{
    out << "{\"samples\":" << compare.samples
        << ",\"status_mismatch_total\":" << compare.status_mismatch_total
        << ",\"status_mismatch_samples\":" << compare.status_mismatch_samples
        << ",\"valid_count_delta_total\":" << compare.valid_count_delta_total
        << ",\"valid_count_delta_mean\":" << compare.valid_count_delta_mean
        << ",\"coordinate_comparisons\":" << compare.coordinate_comparisons
        << ",\"max_coordinate_diff_px\":" << compare.max_coordinate_diff_px
        << ",\"mean_coordinate_diff_px\":" << compare.mean_coordinate_diff_px
        << ",\"p95_coordinate_diff_px\":" << compare.p95_coordinate_diff_px
        << "}";
}

void writeStage(std::ostream &out,
                const StageResult &stage,
                bool has_valid_points,
                bool has_reference_check,
                bool has_baseline_check)
{
    const Summary timing = summarize(stage.samples.milliseconds);
    out << "{\"samples\":" << timing.count
        << ",\"skipped_empty_points\":" << stage.samples.skipped_empty_points;
    if (timing.count != 0)
        out << ",\"mean_ms\":" << timing.mean
            << ",\"p50_ms\":" << timing.p50
            << ",\"p95_ms\":" << timing.p95
            << ",\"min_ms\":" << timing.min
            << ",\"max_ms\":" << timing.max;
    else
        out << ",\"mean_ms\":null,\"p50_ms\":null,\"p95_ms\":null,\"min_ms\":null,\"max_ms\":null";
    out << ",\"valid_points\":";
    if (has_valid_points && !stage.samples.valid_points.empty())
        writeCountSummary(out, summarizeCounts(stage.samples.valid_points));
    else
        out << "null";

    if (has_reference_check)
    {
        out << ",\"reference_check\":";
        writePointCompare(out, stage.gftt_reference_check);
    }
    if (has_baseline_check)
    {
        out << ",\"baseline_check\":";
        writeFlowCompare(out, stage.baseline_check);
    }
    out << "}";
}

void writeJson(std::ostream &out, const BenchmarkResult &result)
{
    out << std::setprecision(12);
    out << "{\n";
    out << "  \"schema\":\"tracker_kernels_microbench_v1\",\n";
    out << "  \"opencv_version\":" << quote(CV_VERSION) << ",\n";
    out << "  \"opencl\":{\"requested\":" << (result.opencl.requested ? "true" : "false")
        << ",\"have_opencl\":" << (result.opencl.have_opencl ? "true" : "false")
        << ",\"context_initialized\":" << (result.opencl.context_initialized ? "true" : "false")
        << ",\"device_available\":" << (result.opencl.device_available ? "true" : "false")
        << ",\"runtime_enabled\":" << (result.opencl.runtime_enabled ? "true" : "false")
        << ",\"kernels_use_opencl\":false"
        << ",\"execution_backend\":\"cpu_mat\""
        << ",\"device_name\":" << quote(result.opencl.device_name)
        << ",\"device_vendor\":" << quote(result.opencl.device_vendor)
        << ",\"device_version\":" << quote(result.opencl.device_version)
        << ",\"device_type\":" << quote(result.opencl.device_type)
        << ",\"error\":" << quote(result.opencl.error) << "},\n";
    out << "  \"input\":{\"directory\":" << quote(result.input.directory)
        << ",\"images_available\":" << result.input.images_available
        << ",\"images_used\":" << result.input.images_used
        << ",\"truncated\":" << (result.input.truncated ? "true" : "false")
        << ",\"pair_count\":" << result.input.pair_count
        << ",\"width\":" << result.input.width
        << ",\"height\":" << result.input.height
        << ",\"first_image\":" << quote(result.input.first_image)
        << ",\"last_image\":" << quote(result.input.last_image) << "},\n";
    out << "  \"method\":{\"rounds\":" << result.options.rounds
        << ",\"warmup_rounds\":" << result.options.warmup_rounds
        << ",\"max_images\":" << result.options.max_images
        << ",\"baseline_case\":" << quote(result.baseline_case_name)
        << ",\"percentile\":\"linear interpolation over sorted samples\""
        << ",\"empty_fixed_points\":\"skip both LK calls; timing excludes skips; valid_points includes zeros\""
        << ",\"gftt\":{\"max_corners\":" << kGfttMaxCorners
        << ",\"quality_level\":" << kGfttQualityLevel
        << ",\"min_distance\":" << kGfttMinDistance
        << ",\"block_size\":" << kGfttBlockSize
        << ",\"use_harris\":" << (kGfttUseHarris ? "true" : "false")
        << ",\"k\":" << kGfttHarrisK << "}"
        << ",\"pyramid\":{\"win_size\":" << kKltWinSize
        << ",\"max_level\":" << kPyramidMaxLevel
        << ",\"try_reuse_input_image\":false"
        << ",\"images_built_per_sample\":2"
        << ",\"timing_scope\":\"previous_and_current_images_per_pair\""
        << ",\"production_cached_images_built_per_frame\":1}"
        << ",\"lk_forward\":{\"max_level\":" << kForwardMaxLevel
        << ",\"criteria\":\"COUNT+EPS," << kTermMaxCount << ","
        << kTermEpsilon << "\",\"flags\":0}"
        << ",\"lk_reverse\":{\"max_level\":" << kReverseMaxLevel
        << ",\"criteria\":\"COUNT+EPS," << kTermMaxCount << ","
        << kTermEpsilon << "\",\"initial\":\"original_fixed_points\""
        << ",\"flags\":\"OPTFLOW_USE_INITIAL_FLOW\"}"
        << ",\"auxiliary\":{\"corner_min_eigenval\":{\"block_size\":"
        << kCornerBlockSize << ",\"ksize\":" << kCornerKSize
        << "},\"dilate_3x3\":true}"
        << "},\n";
    out << "  \"cases\":[\n";
    for (std::size_t i = 0; i < result.cases.size(); ++i)
    {
        const CaseResult &case_result = result.cases[i];
        out << "    {\"name\":" << quote(case_result.config.name)
            << ",\"threads\":" << case_result.config.threads
            << ",\"actual_threads\":" << case_result.actual_threads
            << ",\"derivative_pyramid\":"
            << (case_result.config.derivative_pyramid ? "true" : "false")
            << ",\"previous_pyramid_levels\":" << case_result.previous_pyramid_levels
            << ",\"current_pyramid_levels\":" << case_result.current_pyramid_levels
            << ",\"previous_pyramid_mats\":" << case_result.previous_pyramid_mats
            << ",\"current_pyramid_mats\":" << case_result.current_pyramid_mats
            << ",\"stages\":{";

        out << "\"gftt\":";
        writeStage(out, case_result.gftt, true, true, false);
        out << ",\"pyramid\":";
        writeStage(out, case_result.pyramid, false, false, false);
        out << ",\"lk_forward\":";
        writeStage(out, case_result.lk_forward, true, false, true);
        out << ",\"lk_reverse\":";
        writeStage(out, case_result.lk_reverse, true, false, true);
        out << ",\"corner_min_eigenval\":";
        writeStage(out, case_result.corner_min_eigenval, false, false, false);
        out << ",\"dilate_3x3\":";
        writeStage(out, case_result.dilate_3x3, false, false, false);
        out << "}}";
        if (i + 1 < result.cases.size())
            out << ",";
        out << "\n";
    }
    out << "  ]\n}\n";
}

std::string csvEscape(const std::string &value)
{
    if (value.find_first_of(",\"\n\r") == std::string::npos)
        return value;
    std::string escaped = "\"";
    for (char c : value)
    {
        if (c == '"')
            escaped += "\"\"";
        else
            escaped += c;
    }
    escaped += "\"";
    return escaped;
}

void writeCsvStage(std::ostream &out,
                   const BenchmarkResult &result,
                   const CaseResult &case_result,
                   const std::string &stage_name,
                   const StageResult &stage,
                   bool has_valid_points,
                   bool has_reference_check,
                   bool has_baseline_check)
{
    const Summary timing = summarize(stage.samples.milliseconds);
    const CountSummary counts = summarizeCounts(stage.samples.valid_points);
    const PointCompare &point_compare = stage.gftt_reference_check;
    const FlowCompare &flow_compare = stage.baseline_check;

    out << csvEscape(case_result.config.name) << ','
        << case_result.config.threads << ','
        << (case_result.config.derivative_pyramid ? 1 : 0) << ','
        << stage_name << ','
        << timing.count << ',' << stage.samples.skipped_empty_points << ',';
    if (timing.count != 0)
        out << timing.mean << ',' << timing.p50 << ',' << timing.p95 << ','
            << timing.min << ',' << timing.max;
    else
        out << ",,,,";
    out << ',';
    if (has_valid_points)
        out << counts.mean << ',' << counts.p50 << ',' << counts.p95 << ','
            << counts.min << ',' << counts.max;
    else
        out << ",,,,";
    out << ',';
    if (has_reference_check)
        out << point_compare.count_mismatch_samples << ','
            << point_compare.max_abs_count_delta << ','
            << point_compare.max_coordinate_diff_px;
    else
        out << ",,";
    out << ',';
    if (has_baseline_check)
        out << flow_compare.status_mismatch_total << ','
            << flow_compare.status_mismatch_samples << ','
            << flow_compare.valid_count_delta_total << ','
            << flow_compare.valid_count_delta_mean << ','
            << flow_compare.coordinate_comparisons << ','
            << flow_compare.max_coordinate_diff_px << ','
            << flow_compare.p95_coordinate_diff_px;
    else
        out << ",,,,,,";
    out << ',' << result.input.images_used
        << ',' << result.input.pair_count
        << ',' << result.options.rounds
        << ',' << result.options.warmup_rounds
        << ',' << (result.opencl.requested ? 1 : 0)
        << ',' << (result.opencl.have_opencl ? 1 : 0)
        << ',' << (result.opencl.device_available ? 1 : 0)
        << ',' << (result.opencl.runtime_enabled ? 1 : 0)
        << ',' << csvEscape(result.opencl.device_name)
        << ",cpu_mat,0,2,previous_and_current_images_per_pair,1"
        << '\n';
}

void writeCsv(std::ostream &out, const BenchmarkResult &result)
{
    out << "case,threads,derivative_pyramid,stage,samples,skipped_empty_points,mean_ms,p50_ms,p95_ms,min_ms,max_ms,"
           "valid_points_mean,valid_points_p50,valid_points_p95,valid_points_min,valid_points_max,"
           "reference_count_mismatch_samples,reference_max_abs_count_delta,reference_max_coordinate_diff_px,"
           "status_mismatch_total,status_mismatch_samples,valid_count_delta_total,valid_count_delta_mean,"
           "coordinate_comparisons,max_coordinate_diff_px,coordinate_diff_p95_px,"
           "images_used,pair_count,rounds,warmup_rounds,opencl_requested,opencl_have,opencl_device_available,"
           "opencl_runtime_enabled,opencl_device,execution_backend,kernels_use_opencl,"
           "pyramid_images_built_per_sample,pyramid_timing_scope,production_cached_images_built_per_frame\n";
    for (const CaseResult &case_result : result.cases)
    {
        writeCsvStage(out, result, case_result, "gftt", case_result.gftt,
                      true, true, false);
        writeCsvStage(out, result, case_result, "pyramid", case_result.pyramid,
                      false, false, false);
        writeCsvStage(out, result, case_result, "lk_forward", case_result.lk_forward,
                      true, false, true);
        writeCsvStage(out, result, case_result, "lk_reverse", case_result.lk_reverse,
                      true, false, true);
        writeCsvStage(out, result, case_result, "corner_min_eigenval",
                      case_result.corner_min_eigenval, false, false, false);
        writeCsvStage(out, result, case_result, "dilate_3x3", case_result.dilate_3x3,
                      false, false, false);
    }
}

void printUsage(const char *program)
{
    std::cerr
        << "Usage: " << program << " [options] <image-directory>\n"
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

Options parseOptions(int argc, char **argv)
{
    Options options;
    bool have_directory = false;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        const auto requireValue = [&](const std::string &name) -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error(name + " requires a value");
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help")
        {
            printUsage(argv[0]);
            std::exit(0);
        }
        else if (arg == "--max-images")
            options.max_images = std::stoi(requireValue(arg));
        else if (arg == "--rounds")
            options.rounds = std::stoi(requireValue(arg));
        else if (arg == "--warmup-rounds")
            options.warmup_rounds = std::stoi(requireValue(arg));
        else if (arg == "--format")
            options.format = requireValue(arg);
        else if (arg == "--output")
            options.output_path = requireValue(arg);
        else if (arg == "--opencl")
            options.request_opencl = true;
        else if (!arg.empty() && arg[0] == '-')
            throw std::runtime_error("unknown option: " + arg);
        else if (!have_directory)
        {
            options.image_dir = arg;
            have_directory = true;
        }
        else
            throw std::runtime_error("unexpected extra positional argument: " + arg);
    }

    if (!have_directory)
        throw std::runtime_error("missing image directory");
    if (options.max_images < 2 || options.max_images > kMaxImages)
        throw std::runtime_error("--max-images must be in [2, 120]");
    if (options.rounds < 1)
        throw std::runtime_error("--rounds must be >= 1");
    if (options.warmup_rounds < 0)
        throw std::runtime_error("--warmup-rounds must be >= 0");
    if (options.format != "json" && options.format != "csv")
        throw std::runtime_error("--format must be json or csv");
    return options;
}

BenchmarkResult runBenchmark(const Options &options)
{
    BenchmarkResult result;
    result.options = options;
    result.opencl = inspectOpenCL(options.request_opencl);

    std::cerr << "execution_backend=cpu_mat; OpenCL runtime_enabled="
              << result.opencl.runtime_enabled << " (runtime flag only; no GPU kernels)\n"
              << "pyramid timing: two images per pair; production cache: one new image per frame\n";

    std::vector<fs::path> all_images = findImages(options.image_dir);
    result.input.directory = options.image_dir.string();
    result.input.images_available = all_images.size();
    if (all_images.size() > static_cast<std::size_t>(options.max_images))
    {
        all_images.resize(static_cast<std::size_t>(options.max_images));
        result.input.truncated = true;
    }
    if (all_images.size() < 2)
        throw std::runtime_error("need at least two grayscale PNG/PGM images");

    result.input.images_used = all_images.size();
    result.input.pair_count = all_images.size() - 1;
    result.input.first_image = all_images.front().filename().string();
    result.input.last_image = all_images.back().filename().string();

    std::cerr << "loading " << all_images.size() << " images from "
              << options.image_dir << "\n";
    std::vector<cv::Mat> images = loadImages(all_images);
    result.input.width = images.front().cols;
    result.input.height = images.front().rows;
    for (const cv::Mat &image : images)
    {
        if (image.size() != images.front().size())
            throw std::runtime_error("all benchmark images must have the same size");
    }

    std::cerr << "fixing GFTT points for " << result.input.pair_count
              << " image pairs\n";
    const std::vector<std::vector<cv::Point2f>> fixed_points =
        computeReferencePoints(images);

    std::vector<CaseConfig> configs = {
        {"threads1_image_only", 1, false},
        {"threads1_derivatives", 1, true},
        {"threads2_image_only", 2, false},
        {"threads2_derivatives", 2, true},
        {"threads4_image_only", 4, false},
        {"threads4_derivatives", 4, true},
    };
    result.baseline_case_name = configs.front().name;

    const std::size_t baseline_count =
        static_cast<std::size_t>(options.rounds) * result.input.pair_count;
    std::vector<BaselineSample> baseline_samples(baseline_count);
    result.cases.reserve(configs.size());

    for (std::size_t i = 0; i < configs.size(); ++i)
    {
        const bool baseline = i == 0;
        result.cases.emplace_back();
        CaseResult &case_result = result.cases.back();
        std::cerr << "running " << configs[i].name
                  << " (warmup=" << options.warmup_rounds
                  << ", measured=" << options.rounds << ")\n";
        runCase(images, fixed_points, configs[i], options.rounds,
                options.warmup_rounds, baseline, baseline_samples,
                case_result);
    }

    cv::setNumThreads(1);
    return result;
}

void writeOutput(const BenchmarkResult &result)
{
    std::ofstream file;
    std::ostream *out = &std::cout;
    if (!result.options.output_path.empty())
    {
        file.open(result.options.output_path, std::ios::out | std::ios::trunc);
        if (!file)
            throw std::runtime_error("failed to open output file: " +
                                     result.options.output_path);
        out = &file;
    }

    if (result.options.format == "csv")
        writeCsv(*out, result);
    else
        writeJson(*out, result);
    out->flush();
    if (!*out)
        throw std::runtime_error("failed to write benchmark output");
}

}  // namespace

int main(int argc, char **argv)
{
    try
    {
        const Options options = parseOptions(argc, argv);
        const BenchmarkResult result = runBenchmark(options);
        writeOutput(result);
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
