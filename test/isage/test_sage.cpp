#include "gtest/gtest.h"

#include "sage.hpp"

#include <vector>

TEST(SageTopic, create) {
  isage::wtm::SageTopic<std::vector<double> > vec(10, 0.0);
}
TEST(SageTopic, set) {
  isage::wtm::SageTopic<std::vector<double> > vec(10, 0.0);
  vec[5] = 10;
  ASSERT_EQ(10, vec[5]);
}
TEST(DenseSageTopic, create) {
  isage::wtm::DenseSageTopic vec(10, 0.0);
}
TEST(DenseSageTopic, set) {
  isage::wtm::DenseSageTopic vec(10, 0.0);
  vec[5] = 10;
  ASSERT_EQ(10, vec[5]);
}
