#include "gtest/gtest.h"

#include "dmc.hpp"

#include <vector>

TEST(mathops, log_add) {
  std::vector<double> vec(3, -1.7);
  ASSERT_NEAR(-0.6013877113, mathops::log_add(vec), 1E-6);
}

TEST(mathops, sample_uniform_log) {
  double x = -.56;
  mathops::sample_uniform_log(x);
}

TEST(mathops, log_add2) {
  std::vector<double> vec(5, -1.7);
  const double norm = -0.090562087565899;
  ASSERT_NEAR(norm, mathops::log_add(vec), 1E-6);
}
