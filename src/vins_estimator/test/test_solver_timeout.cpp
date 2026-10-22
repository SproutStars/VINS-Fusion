#include <gtest/gtest.h>

#include <ceres/ceres.h>

#include "estimator/solver_timeout.h"

TEST(SolverTimeout, UsesTerminationTypeAndElapsedBudgetWithoutMessage)
{
    ceres::Solver::Summary summary;
    summary.termination_type = ceres::NO_CONVERGENCE;

    EXPECT_TRUE(vins::solverTimedOut(summary, 0.96, 1.0));
    EXPECT_TRUE(vins::solverTimedOut(summary, 0.95, 1.0));
    EXPECT_FALSE(vins::solverTimedOut(summary, 0.94, 1.0));

    summary.message = "Ceres Solver Report: Terminating: Maximum number of iterations reached.";
    EXPECT_FALSE(vins::solverTimedOut(summary, 0.10, 1.0));
}

TEST(SolverTimeout, DoesNotTreatEarlyConvergenceAsTimeout)
{
    ceres::Solver::Summary summary;
    summary.termination_type = ceres::CONVERGENCE;
    summary.message = "Ceres Solver Report: Terminating: Function tolerance reached.";

    EXPECT_FALSE(vins::solverTimedOut(summary, 1.0, 1.0));
}

TEST(SolverTimeout, RetainsMessageCompatibilityFallback)
{
    ceres::Solver::Summary summary;
    summary.termination_type = ceres::CONVERGENCE;
    summary.message = "Maximum solver time reached";

    EXPECT_TRUE(vins::solverTimedOut(summary, 0.01, 1.0));
    EXPECT_TRUE(vins::solverTimedOut(summary, 0.01, 0.0));
}
