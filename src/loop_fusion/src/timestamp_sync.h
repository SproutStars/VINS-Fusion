#pragma once

#include <builtin_interfaces/msg/time.hpp>

#include <cstdint>
#include <cstdlib>

namespace loop_fusion
{

// VINS used to carry image timestamps through a double. At EuRoC epoch values
// one double ULP is roughly 240 ns, so a pose stamp can differ from the exact
// bag image stamp by about 100 ns even though both describe the same frame.
// Keep the tolerance far below the image period (50 ms at 20 Hz) but well
// above that round-trip error.
constexpr int64_t kStampMatchToleranceNs = 2'000;

inline int64_t stampNs(const builtin_interfaces::msg::Time &stamp)
{
    return static_cast<int64_t>(stamp.sec) * 1'000'000'000LL + stamp.nanosec;
}

inline bool stampsMatch(int64_t lhs_ns, int64_t rhs_ns,
                        int64_t tolerance_ns = kStampMatchToleranceNs)
{
    return std::llabs(lhs_ns - rhs_ns) <= tolerance_ns;
}

}  // namespace loop_fusion
