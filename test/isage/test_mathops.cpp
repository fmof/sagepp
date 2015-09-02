#include "gtest/gtest.h"

#include "dmc.hpp"
#include "mathops.hpp"
#include "util.hpp"
#include "gsl/gsl_sf_log.h"

#include <set>
#include <vector>

TEST(mathops, log_add) {
  std::vector<double> vec(3, -1.7);
  ASSERT_NEAR(-0.6013877113, mathops::log_add(vec), 1E-6);
}

TEST(mathops, sample_uniform_log) {
  double x = -.56;
  mathops::sample_uniform_log(x);
}

TEST(mathops, sample_uniform_converted_range) {
  const int num = 3;
  double lower[num] = { -3.2, .4, 200 };
  double upper[num] = { 1.4,  .41, 1000};
  bool good[num] = {true, true, true};
  for(int i = 0; i < 1000; ++i) {
    for(int k = 0; k < num; ++k) {
      double x = mathops::sample_uniform(lower[k], upper[k]);
      if(x < lower[k] || x > upper[k]) {
	good[k] = false;
      }
    }    
  }
  for(int k = 0; k < num; ++k) {
    ASSERT_TRUE(good[k]);
  }
}

TEST(mathops, log_add2) {
  std::vector<double> vec(5, -1.7);
  const double norm = -0.090562087565899;
  ASSERT_NEAR(norm, mathops::log_add(vec), 1E-6);
}

TEST(mathops, log_sum_exp_vector) {
  std::vector<double> ft_weights;
  ft_weights.push_back(5);
  ft_weights.push_back(4);
  std::vector<double> terms = isage::util::exp(ft_weights);
  ASSERT_EQ(5, ft_weights[0]);
  ASSERT_EQ(4, ft_weights[1]);
  std::vector<double> expected_exp;
  expected_exp.push_back(148.413159102576);
  expected_exp.push_back(54.5981500331442);
  for(int i = 0; i < 2; ++i) {
    ASSERT_NEAR(expected_exp[i], terms[i], 1E-6);
  }
  const double expected = gsl_sf_log(terms[0] + terms[1]);
  ASSERT_NEAR(expected, mathops::log_sum_exp(ft_weights), 1E-6);
}
// TEST(mathops, log_sum_exp_set) {
//   std::set<double> ft_weights;
//   ft_weights.insert(5);
//   ft_weights.insert(4);
//   std::set<double> terms = isage::util::exp(ft_weights);
//   // std::set<double> expected_exp;
//   // expected_exp.push_back(148.413159102576);
//   // expected_exp.push_back(54.5981500331442);
//   // for(int i = 0; i < 2; ++i) {
//   //   ASSERT_NEAR(expected_exp[i], terms[i], 1E-6);
//   // }
//   // const double expected = gsl_sf_log(terms[0] + terms[1]);
//   // ASSERT_NEAR(expected, mathops::log_sum_exp(ft_weights), 1E-6);
// }

TEST(Util, exp_ptr) {
  std::vector<double> vec;
  vec.push_back(5);
  vec.push_back(4);
  std::vector<double> expected;
  expected.push_back(148.413159102576);
  expected.push_back(54.5981500331442);
  isage::util::exp(&vec);
  for(int i = 0; i < 2; ++i) {
    ASSERT_NEAR(expected[i], vec[i], 1E-6) << "(" << i << ") not equal";
  }
}

TEST(Util, exp_ref) {
  std::vector<double> vec;
  vec.push_back(5);
  vec.push_back(4);
  std::vector<double> expected;
  expected.push_back(148.413159102576);
  expected.push_back(54.5981500331442);
  std::vector<double> res = isage::util::exp(vec);
  for(int i = 0; i < 2; ++i) {
    ASSERT_NEAR(expected[i], res[i], 1E-6) << "(" << i << ") not equal";
  }
}
