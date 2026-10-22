#pragma once

#include <ceres/ceres.h>

#include <cmath>
#include <string>

namespace vins
{
// Ceres reports a wall-clock limit as NO_CONVERGENCE. Keep the historical
// English-message check as a compatibility fallback, but also require the
// measured duration to be close to the configured budget when the message is
// absent or localized.
inline bool solverTimedOut(const ceres::Solver::Summary &summary,
                           double elapsed_seconds,
                           double max_solver_time_seconds)
{
    constexpr double kElapsedBudgetRatio = 0.95;
    const bool message_indicates_timeout =
        summary.message.find("Maximum solver time reached") != std::string::npos;
    if (!std::isfinite(elapsed_seconds) || !std::isfinite(max_solver_time_seconds) ||
        max_solver_time_seconds <= 0.0)
    {
        return message_indicates_timeout;
    }

    return message_indicates_timeout ||
           (summary.termination_type == ceres::NO_CONVERGENCE &&
            elapsed_seconds >= max_solver_time_seconds * kElapsedBudgetRatio);
}
}  // namespace vins
