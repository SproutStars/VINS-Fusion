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
constexpr int64_t K_STAMP_MATCH_TOLERANCE_NS = 2'000;

inline int64_t stampNs(const builtin_interfaces::msg::Time & _stamp)
{
  return static_cast<int64_t>(_stamp.sec) * 1'000'000'000LL + _stamp.nanosec;
}

inline bool stampsMatch(int64_t _lhs_ns, int64_t _rhs_ns, int64_t _tolerance_ns = K_STAMP_MATCH_TOLERANCE_NS)
{
  return std::llabs(_lhs_ns - _rhs_ns) <= _tolerance_ns;
}

} // namespace loop_fusion
