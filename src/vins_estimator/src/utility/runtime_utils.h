#pragma once

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace vins
{
// Sensor-time gate: playback speed and wall-clock scheduling do not affect it.
// The tracker-only first frame preserves the original 20 Hz -> 10 Hz cadence.
class FrameRateGate
{
public:
    explicit FrameRateGate(double rate = 0.0, bool skip_first = false)
    { reset(rate, skip_first); }

    void reset(double rate, bool skip_first = false)
    {
        if (!std::isfinite(rate) || rate < 0.0)
            throw std::invalid_argument("Frame rate must be finite and nonnegative");
        period_ = rate > 0.0 ? 1.0 / rate : 0.0;
        skip_first_ = skip_first;
        seen_ = accepted_ = false;
        last_seen_ = next_ = 0.0;
    }

    bool accept(double timestamp)
    {
        if (!std::isfinite(timestamp)) return false;
        if (seen_ && timestamp < last_seen_ - 1e-6)
            seen_ = accepted_ = false;
        const bool first = !seen_;
        seen_ = true;
        last_seen_ = timestamp;
        if (period_ == 0.0) return true;
        if (first && skip_first_) return false;
        if (!accepted_) {
            accepted_ = true;
            next_ = timestamp + period_;
            return true;
        }
        if (timestamp + 1e-6 < next_) return false;
        next_ += std::max(1.0, std::floor((timestamp - next_) / period_) + 1.0) * period_;
        return true;
    }
private:
    double period_, last_seen_, next_;
    bool skip_first_, seen_, accepted_;
};

// Zero capacity explicitly disables history; it never means unbounded storage.
template<class T>
class BoundedHistory
{
public:
    explicit BoundedHistory(size_t capacity = 0) : capacity_(capacity) {}
    void reset(size_t capacity) { capacity_ = capacity; items_.clear(); }
    void clear() { items_.clear(); }
    void push(const T &item)
    {
        if (!capacity_) return;
        if (items_.size() == capacity_) items_.pop_front();
        items_.push_back(item);
    }
    size_t size() const { return items_.size(); }
    const std::deque<T> &items() const { return items_; }
private:
    size_t capacity_;
    std::deque<T> items_;
};

// One submitting thread, fixed persistent workers. run() is a synchronous barrier
// and propagates worker exceptions after every worker has completed its job.
class WorkerPool
{
public:
    explicit WorkerPool(size_t count)
    {
        if (!count) throw std::invalid_argument("Worker pool cannot be empty");
        try {
            for (size_t i = 0; i < count; ++i)
                workers_.emplace_back([this, i] { work(i); });
        } catch (...) {
            stop();
            throw;
        }
    }
    ~WorkerPool() { stop(); }
    WorkerPool(const WorkerPool &) = delete;
    WorkerPool &operator=(const WorkerPool &) = delete;

    void run(std::function<void(size_t)> job)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        job_ = std::move(job);
        error_ = nullptr;
        remaining_ = workers_.size();
        ++generation_;
        ready_.notify_all();
        done_.wait(lock, [this] { return remaining_ == 0; });
        job_ = {};
        if (error_) std::rethrow_exception(error_);
    }
private:
    void stop()
    {
        { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
        ready_.notify_all();
        for (auto &thread : workers_) if (thread.joinable()) thread.join();
    }
    void work(size_t index)
    {
        size_t generation = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            ready_.wait(lock, [&] { return stopping_ || generation_ != generation; });
            if (stopping_) return;
            generation = generation_;
            lock.unlock();
            std::exception_ptr error;
            try { job_(index); } catch (...) { error = std::current_exception(); }
            lock.lock();
            if (error && !error_) error_ = error;
            if (--remaining_ == 0) done_.notify_one();
        }
    }
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable ready_, done_;
    std::function<void(size_t)> job_;
    std::exception_ptr error_;
    size_t generation_ = 0, remaining_ = 0;
    bool stopping_ = false;
};
}  // namespace vins
