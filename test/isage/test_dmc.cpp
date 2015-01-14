#include "gtest/gtest.h"

#include "dmc.hpp"
#include "logging.hpp"
#include "util.hpp"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_log.h>

#include <vector>

TEST(DMC, create_fixed) {
  dmc::dmc dmc_pr(10);
}
TEST(DMC, create_fixed_with_value) {
  dmc::dmc dmc_pr(10, 0.1);
}
TEST(DMC, change_with_op) {
  dmc::dmc dmc_pr(2, 0.1);
  dmc_pr.hyperparameter(0,10);
  ASSERT_EQ(10, dmc_pr.hyperparameter(0));
  ASSERT_EQ(0.1, dmc_pr.hyperparameter(1));
}
TEST(DMC_CAT, log_u_sample) {
  std::vector<double> vec(5, -1.7);
  const double r = -0.7;
  ASSERT_EQ(2,dmc::cat::log_u_sample_with_value(r, vec));
}
TEST(DMC_CAT, log_u_sample_specific) {
  std::vector<double> vec(10);
  vec[0] = -6.83176;
  vec[1] = -6.82535;
  vec[2] = -6.82535;
  vec[3] = -6.82535;
  vec[4] = -6.82535;
  vec[5] = -6.82535;
  vec[6] = -6.82467;
  vec[7] = -6.82398;
  vec[8] = -6.8233;
  vec[9] = -6.82193;
  const double selected_log_val = -4.522908033;
  ASSERT_NEAR(-4.52265, mathops::log_add(vec), 1E-6);
  ASSERT_EQ(9,dmc::cat::log_u_sample_with_value(selected_log_val, vec));
}

TEST(DMC_CAT, log_u_sample_boundary_just_below) {
  std::vector<double> vec(5, -1.7);
  const double norm = -0.090562087565899;
  ASSERT_EQ(4,dmc::cat::log_u_sample_with_value(norm, vec));
}
TEST(DMC_CAT, log_u_sample_boundary_just_above) {
  std::vector<double> vec(5, -1.7);
  //notice, that the third decimal point is 1 instead of 0
  const double norm = -0.091562087565899;
  ASSERT_EQ(4,dmc::cat::log_u_sample_with_value(norm, vec));
}

TEST(DMC_CAT, log_u_sample_boundary_just_above2) {
  std::vector<double> vec(5, -1.7);
  const double norm = -0.011562087565899;
  ASSERT_EQ(-1,dmc::cat::log_u_sample_with_value(norm, vec));
}

TEST(DMC_CAT, u_sample) {
  std::vector<double> vec;
  vec.push_back(.25);
  vec.push_back(.5);
  vec.push_back(.25);
  const double r = 0.7;
  ASSERT_EQ(1,dmc::cat::u_sample_with_value(r, vec));
}
TEST(DMC_CAT, u_sample1) {
  std::vector<double> vec;
  vec.push_back(.25);
  vec.push_back(.5);
  vec.push_back(.25);
  const double r = 0.9;
  ASSERT_EQ(2,dmc::cat::u_sample_with_value(r, vec));
}
TEST(DMC_CAT, u_sample2) {
  std::vector<double> vec;
  vec.push_back(.25);
  vec.push_back(.5);
  vec.push_back(.25);
  const double r = 1.0;
  ASSERT_EQ(2,dmc::cat::u_sample_with_value(r, vec));
}
TEST(DMC_CAT, u_sample3) {
  std::vector<double> vec;
  vec.push_back(.25);
  vec.push_back(.5);
  vec.push_back(.25);
  const double r = 1.1;
  ASSERT_EQ(-1,dmc::cat::u_sample_with_value(r, vec));
}

TEST(DMC, log_u_conditional_oracle) {
  dmc::dmc dmc(5, 0.0);
  ASSERT_EQ(gsl_sf_log(5.0/16.0), dmc.log_u_conditional_oracle(5, 16, 1));
  ASSERT_EQ(gsl_sf_log(15.0/136.0), dmc.log_u_conditional_oracle(5, 16, 2));
  ASSERT_NEAR(gsl_sf_log(7.897472976299696423020080281E-6), dmc.log_u_conditional_oracle(50, 124, 14), 1E-10);
}

TEST(DMC, log_u_conditional_gsl) {
  dmc::dmc dmc(5, 0.0);
  ASSERT_NEAR(gsl_sf_log(5.0/16.0), dmc.log_u_conditional_gsl(5, 16, 1), 1E-10);
  ASSERT_NEAR(gsl_sf_log(15.0/136.0), dmc.log_u_conditional_gsl(5, 16, 2), 1E-10);
  ASSERT_NEAR(gsl_sf_log(7.897472976299696423020080281E-6), dmc.log_u_conditional_gsl(50, 124, 14), 1E-10);
}

TEST(DMC, log_u_conditional_sterling) {
  dmc::dmc dmc(5, 0.0);
  ASSERT_NEAR(gsl_sf_log(5.0/16.0), dmc.log_u_conditional_oracle(5, 16, 1), 1E-10);
  ASSERT_NEAR(gsl_sf_log(15.0/136.0), dmc.log_u_conditional_oracle(5, 16, 2), 1E-10);
  ASSERT_NEAR(gsl_sf_log(7.897472976299696423020080281E-6), dmc.log_u_conditional_oracle(50, 124, 14), 1E-10);
}

TEST(DMC, Wallach1) {
  const gsl_rng_type *which_gsl_rng = gsl_rng_mt19937;
  gsl_rng *rnd_gen = gsl_rng_alloc(which_gsl_rng);
  const int support_size = 3;
  std::vector<double> gold({10.0, 1.0, 1.0});
  std::vector<double> alpha(support_size);
  for(int i = 0; i < support_size; ++i) {
    alpha[i] = gsl_rng_uniform(rnd_gen) * 20;
  }
  dmc::dmc my_dmc(alpha);
  const int num_inst = 10000;
  const int num_samples = 1000;
  std::vector< std::vector<int> > counts(num_inst, std::vector<int>(support_size));
  double *theta = (double*)isage::util::MALLOC(sizeof(double)*support_size);
  for(int i = 0; i < num_inst; ++i) {
    gsl_ran_dirichlet(rnd_gen, support_size, gold.data(), theta);
    gsl_ran_multinomial(rnd_gen, support_size, num_samples, (const double*)theta, (unsigned int*)counts[i].data());
  }
  free(theta);
  my_dmc.reestimate_hyperparameters_wallach1(counts, 5000);
  for(int k = 0; k < support_size; ++k) {
    EXPECT_NEAR(my_dmc.hyperparameter(k), gold[k], 0.5);
  }
}
