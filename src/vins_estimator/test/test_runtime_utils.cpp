#include <gtest/gtest.h>
#include "utility/runtime_utils.h"
#include <array>
#include <atomic>

TEST(FrameRateGate, PreservesEurocEverySecondFrame)
{
    vins::FrameRateGate gate(10.0, true);
    for (int i = 0; i < 2000; ++i)
        EXPECT_EQ(gate.accept(1413393212.2557605 + i * 0.05), i % 2 == 1) << i;
}
TEST(FrameRateGate, HandlesRatesJumpsAndRestart)
{
    vins::FrameRateGate gate(5.0);
    int accepted = 0;
    for (int i = 0; i < 100; ++i) accepted += gate.accept(i * 0.01);
    EXPECT_EQ(accepted, 5);
    EXPECT_TRUE(gate.accept(10));
    EXPECT_FALSE(gate.accept(10.01));
    EXPECT_TRUE(gate.accept(0));
    EXPECT_FALSE(gate.accept(std::numeric_limits<double>::quiet_NaN()));
    gate.reset(0);
    EXPECT_TRUE(gate.accept(0));
    EXPECT_TRUE(gate.accept(0.001));
    EXPECT_THROW(gate.reset(-1), std::invalid_argument);
}
TEST(BoundedHistory, KeepsLatestAndCanDisableOrReset)
{
    vins::BoundedHistory<int> history(3);
    for (int i = 0; i < 100; ++i) history.push(i);
    ASSERT_EQ(history.size(), 3u);
    EXPECT_EQ(history.items().front(), 97);
    EXPECT_EQ(history.items().back(), 99);
    history.clear(); EXPECT_EQ(history.size(), 0u);
    history.reset(0); history.push(1); EXPECT_EQ(history.size(), 0u);
    history.reset(1); history.push(2); history.push(3);
    ASSERT_EQ(history.size(), 1u); EXPECT_EQ(history.items().front(), 3);
}
TEST(WorkerPool, ReusesWorkersAndPropagatesExceptions)
{
    vins::WorkerPool pool(4);
    std::array<std::thread::id, 4> identities;
    std::array<int, 4> values{};
    pool.run([&](size_t i) { identities[i] = std::this_thread::get_id(); });
    for (int run = 0; run < 100; ++run) {
        pool.run([&](size_t i) {
            EXPECT_EQ(identities[i], std::this_thread::get_id());
            ++values[i];
        });
    }
    for (auto value : values) EXPECT_EQ(value, 100);
    EXPECT_THROW(pool.run([](size_t i) { if (i == 1) throw std::runtime_error("test"); }), std::runtime_error);
    pool.run([&](size_t i) { ++values[i]; });
    for (auto value : values) EXPECT_EQ(value, 101);
    EXPECT_THROW(vins::WorkerPool(0), std::invalid_argument);
}
