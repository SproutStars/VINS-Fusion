#include "estimator/feature_manager.h"
#include <gtest/gtest.h>

namespace
{
using Frame = std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>;
Frame observations(std::initializer_list<int> _ids)
{
  Frame frame;
  for (int id : _ids) {
    Eigen::Matrix<double, 7, 1> value;
    value << 0.1, 0.2, 1, 100, 200, 0, 0;
    frame[id].emplace_back(0, value);
  }
  return frame;
}
class FeatureIndex : public ::testing::Test {
protected:
  Eigen::Matrix3d rotations_[WINDOW_SIZE + 1];
  FeatureManager manager_{rotations_};
  void SetUp() override
  {
    for (auto & rotation : rotations_)
      rotation.setIdentity();
    INIT_DEPTH = 5;
  }
  void add(int _frame, std::initializer_list<int> _ids)
  {
    manager_.addFeatureCheckParallax(_frame, observations(_ids), 0);
  }
  void expectIds(std::initializer_list<int> _expected)
  {
    std::vector<int> actual;
    for (const auto & feature : manager_.feature_)
      actual.push_back(feature.feature_id_);
    EXPECT_EQ(actual, std::vector<int>(_expected));
  }
};
}  // namespace

TEST_F(FeatureIndex, PreservesInsertionAndSolverOrder)
{
  add(0, {7, 3});
  add(1, {1, 7, 3});
  expectIds({3, 7, 1});
  EXPECT_EQ(manager_.feature_.front().feature_per_frame_.size(), 2u);
  EXPECT_EQ(manager_.feature_.back().start_frame_, 1);
}
TEST_F(FeatureIndex, OutliersFailuresAndResetEraseIndexEntries)
{
  add(0, {1, 2, 3});
  std::set<int> outliers{2};
  manager_.removeOutlier(outliers);
  manager_.feature_.front().solve_flag_ = 2;
  manager_.removeFailures();
  expectIds({3});
  add(1, {1, 2, 3});
  expectIds({3, 1, 2});
  EXPECT_EQ(manager_.feature_.back().feature_per_frame_.size(), 1u);
  manager_.clearState();
  add(0, {1, 2, 3});
  expectIds({1, 2, 3});
}
TEST_F(FeatureIndex, RemoveBackAllowsReinsertion)
{
  add(0, {1, 2});
  add(1, {2});
  manager_.removeBack();
  expectIds({2});
  add(1, {1, 2});
  expectIds({2, 1});
  EXPECT_EQ(manager_.feature_.front().feature_per_frame_.size(), 2u);
}
TEST_F(FeatureIndex, DepthShiftErasesShortTracksAndKeepsLongTracks)
{
  add(0, {1, 2});
  add(1, {2});
  add(2, {2});
  for (auto & feature : manager_.feature_)
    feature.estimated_depth_ = 5;
  manager_.removeBackShiftDepth(
    Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
  expectIds({2});
  EXPECT_DOUBLE_EQ(manager_.feature_.front().estimated_depth_, 5);
  add(2, {1, 2});
  expectIds({2, 1});
}
TEST_F(FeatureIndex, RemoveFrontErasesIndexAndAdjustsNewTracks)
{
  add(8, {1});
  add(9, {1, 2});
  add(10, {1, 3});
  manager_.removeFront(10);
  expectIds({1, 3});
  EXPECT_EQ(manager_.feature_.back().start_frame_, 9);
  add(10, {2});
  expectIds({1, 3, 2});
  EXPECT_EQ(manager_.feature_.back().start_frame_, 10);
}
TEST_F(FeatureIndex, RepeatedEraseAndReinsertDoesNotLeaveStaleIterators)
{
  for (int iteration = 0; iteration < 500; ++iteration) {
    add(0, {1, 2, 3, 4});
    std::set<int> outliers{2, 4};
    manager_.removeOutlier(outliers);
    manager_.removeBack();
    EXPECT_TRUE(manager_.feature_.empty());
  }
}
