#include <gtest/gtest.h>
#include "estimator/feature_manager.h"

namespace
{
using Frame = std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>;
Frame observations(std::initializer_list<int> ids)
{
    Frame frame;
    for (int id : ids) {
        Eigen::Matrix<double, 7, 1> value;
        value << 0.1, 0.2, 1, 100, 200, 0, 0;
        frame[id].emplace_back(0, value);
    }
    return frame;
}
class FeatureIndex : public ::testing::Test
{
protected:
    Eigen::Matrix3d rotations[WINDOW_SIZE + 1];
    FeatureManager manager{rotations};
    void SetUp() override
    {
        for (auto &rotation : rotations) rotation.setIdentity();
        INIT_DEPTH = 5;
    }
    void add(int frame, std::initializer_list<int> ids)
    { manager.addFeatureCheckParallax(frame, observations(ids), 0); }
    void expectIds(std::initializer_list<int> expected)
    {
        std::vector<int> actual;
        for (const auto &feature : manager.feature) actual.push_back(feature.feature_id);
        EXPECT_EQ(actual, std::vector<int>(expected));
    }
};
}

TEST_F(FeatureIndex, PreservesInsertionAndSolverOrder)
{
    add(0, {7, 3});
    add(1, {1, 7, 3});
    expectIds({3, 7, 1});
    EXPECT_EQ(manager.feature.front().feature_per_frame.size(), 2u);
    EXPECT_EQ(manager.feature.back().start_frame, 1);
}
TEST_F(FeatureIndex, OutliersFailuresAndResetEraseIndexEntries)
{
    add(0, {1, 2, 3});
    std::set<int> outliers{2};
    manager.removeOutlier(outliers);
    manager.feature.front().solve_flag = 2;
    manager.removeFailures();
    expectIds({3});
    add(1, {1, 2, 3});
    expectIds({3, 1, 2});
    EXPECT_EQ(manager.feature.back().feature_per_frame.size(), 1u);
    manager.clearState();
    add(0, {1, 2, 3});
    expectIds({1, 2, 3});
}
TEST_F(FeatureIndex, RemoveBackAllowsReinsertion)
{
    add(0, {1, 2});
    add(1, {2});
    manager.removeBack();
    expectIds({2});
    add(1, {1, 2});
    expectIds({2, 1});
    EXPECT_EQ(manager.feature.front().feature_per_frame.size(), 2u);
}
TEST_F(FeatureIndex, DepthShiftErasesShortTracksAndKeepsLongTracks)
{
    add(0, {1, 2});
    add(1, {2});
    add(2, {2});
    for (auto &feature : manager.feature) feature.estimated_depth = 5;
    manager.removeBackShiftDepth(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(),
                                Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    expectIds({2});
    EXPECT_DOUBLE_EQ(manager.feature.front().estimated_depth, 5);
    add(2, {1, 2});
    expectIds({2, 1});
}
TEST_F(FeatureIndex, RemoveFrontErasesIndexAndAdjustsNewTracks)
{
    add(8, {1});
    add(9, {1, 2});
    add(10, {1, 3});
    manager.removeFront(10);
    expectIds({1, 3});
    EXPECT_EQ(manager.feature.back().start_frame, 9);
    add(10, {2});
    expectIds({1, 3, 2});
    EXPECT_EQ(manager.feature.back().start_frame, 10);
}
TEST_F(FeatureIndex, RepeatedEraseAndReinsertDoesNotLeaveStaleIterators)
{
    for (int iteration = 0; iteration < 500; ++iteration) {
        add(0, {1, 2, 3, 4});
        std::set<int> outliers{2, 4};
        manager.removeOutlier(outliers);
        manager.removeBack();
        EXPECT_TRUE(manager.feature.empty());
    }
}
