#include "gtest/gtest.h"

#include "dmc.hpp"
#include "logging.hpp"
#include "util.hpp"
#include "mathops.hpp"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_psi.h>

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

TEST(Dirichlet, hyperparameters_objective_check_gsl) {
  double val = 0.1;
  int size = 2;
  std::vector<double> alpha(size, val);
  typedef dmc::DirichletVariationalClosure ClosureType;
  std::vector<std::vector<double> > vparams = { {.4, .6}, {.7, .3} };
  ClosureType closure;
  closure.variational_params = &vparams;
  optimize::GSLVector x0(alpha);
  val = mathops::exp(val);
  double f = dmc::dirichlet::hyperparameters_variational_objective_gsl(x0.get(), &closure);
  double expected = 2 * (gsl_sf_lngamma(2*val) - 2*gsl_sf_lngamma(val)) + 
    ( (val - 1) * ( gsl_sf_psi(vparams[0][0]) - gsl_sf_psi( 1.0 ) ) ) +
    ( (val - 1) * ( gsl_sf_psi(vparams[0][1]) - gsl_sf_psi( 1.0 ) ) ) +
    ( (val - 1) * ( gsl_sf_psi(vparams[1][0]) - gsl_sf_psi( 1.0 ) ) ) +
    ( (val - 1) * ( gsl_sf_psi(vparams[1][1]) - gsl_sf_psi( 1.0 ) ) );
  ASSERT_NEAR(-expected, f, 1E-6);
}

TEST(Dirichlet, hyperparameters_grad_check_gsl) {
  double val = 0.1;
  const int size = 2;
  std::vector<double> alpha(size, val);
  typedef dmc::DirichletVariationalClosure ClosureType;
  std::vector<std::vector<double> > vparams = { {.4, .6}, {.7, .3} };
  ClosureType closure;
  closure.variational_params = &vparams;
  optimize::GSLVector x0(alpha);
  std::vector<double> alpha_pert(alpha);
  double var = 1E-8;
  alpha_pert[0] += var;
  optimize::GSLVector x1(alpha_pert);
  optimize::GSLVector grad(size);
  dmc::dirichlet::hyperparameters_variational_gradient_gsl(x0.get(), &closure, grad.get());
  std::vector<double> vgrad = grad.to_container<std::vector<double> >();
  val = mathops::exp(val);
  std::vector<double> expected_grad = {
    2 * ( gsl_sf_psi(2*val) - gsl_sf_psi(val) ) + gsl_sf_psi(.4) - gsl_sf_psi(1) + gsl_sf_psi(.7) - gsl_sf_psi(1),
    2 * ( gsl_sf_psi(2*val) - gsl_sf_psi(val) ) + gsl_sf_psi(.6) - gsl_sf_psi(1) + gsl_sf_psi(.3) - gsl_sf_psi(1)
  };
  isage::util::scalar_product(-1.0, &expected_grad);
  for(int i = 0; i < size; ++i) {
    ASSERT_NEAR(expected_grad[i], vgrad[i], 1E-5);
  }
}

TEST(Dirichlet, hyperparameters_grad_check_finite_differences_gsl) {
  double val = 0.1;
  const int size = 2;
  std::vector<double> alpha(size, val);
  typedef dmc::DirichletVariationalClosure ClosureType;
  std::vector<std::vector<double> > vparams = { {.4, .6}, {.7, .3} };
  ClosureType closure;
  closure.variational_params = &vparams;
  optimize::GSLVector x0(alpha);
  double f0 = dmc::dirichlet::hyperparameters_variational_objective_gsl(x0.get(), &closure);
  std::vector<double> alpha_pert(alpha);
  double pert = 1E-8;
  alpha_pert[0] += pert;
  optimize::GSLVector x1(alpha_pert);
  double f1 = dmc::dirichlet::hyperparameters_variational_objective_gsl(x1.get(), &closure);
  double est = (f1 - f0) / (mathops::exp(val + pert) - mathops::exp(val));
  optimize::GSLVector grad(size);
  dmc::dirichlet::hyperparameters_variational_gradient_gsl(x0.get(), &closure, grad.get());
  std::vector<double> vgrad = grad.to_container<std::vector<double> >();
  ASSERT_NEAR(vgrad[0], est, 1E-6);
}

TEST(Dirichlet, hyperparameters_objective_check) {
  double val = 0.1;
  int size = 2;
  std::vector<double> alpha(size, val);
  typedef dmc::DirichletVariationalClosure ClosureType;
  std::vector<std::vector<double> > vparams = { {.4, .6}, {.7, .3} };
  ClosureType closure;
  closure.variational_params = &vparams;
  double f = dmc::dirichlet::hyperparameters_variational_objective(alpha, &closure);
  double expected = 2 * (gsl_sf_lngamma(2*val) - 2*gsl_sf_lngamma(val)) + 
    ( (val - 1) * ( gsl_sf_psi(vparams[0][0]) - gsl_sf_psi( 1.0 ) ) ) +
    ( (val - 1) * ( gsl_sf_psi(vparams[0][1]) - gsl_sf_psi( 1.0 ) ) ) +
    ( (val - 1) * ( gsl_sf_psi(vparams[1][0]) - gsl_sf_psi( 1.0 ) ) ) +
    ( (val - 1) * ( gsl_sf_psi(vparams[1][1]) - gsl_sf_psi( 1.0 ) ) );
  ASSERT_NEAR(expected, f, 1E-6);
}

TEST(Dirichlet, hyperparameters_grad_check) {
  double val = 0.1;
  const int size = 2;
  std::vector<double> alpha(size, val);
  typedef dmc::DirichletVariationalClosure ClosureType;
  std::vector<std::vector<double> > vparams = { {.4, .6}, {.7, .3} };
  ClosureType closure;
  closure.variational_params = &vparams;
  std::vector<double> alpha_pert(alpha);
  double var = 1E-8;
  alpha_pert[0] += var;
  std::vector<double> grad;
  dmc::dirichlet::hyperparameters_variational_gradient(alpha, &closure, grad);
  std::vector<double> expected_grad = {
    2 * ( gsl_sf_psi(2*val) - gsl_sf_psi(val) ) + gsl_sf_psi(.4) - gsl_sf_psi(1) + gsl_sf_psi(.7) - gsl_sf_psi(1),
    2 * ( gsl_sf_psi(2*val) - gsl_sf_psi(val) ) + gsl_sf_psi(.6) - gsl_sf_psi(1) + gsl_sf_psi(.3) - gsl_sf_psi(1)
  };
  for(int i = 0; i < size; ++i) {
    ASSERT_NEAR(expected_grad[i], grad[i], 1E-5);
  }
}

TEST(Dirichlet, hyperparameters_grad_check_finite_differences) {
  double val = 0.1;
  const int size = 2;
  std::vector<double> alpha(size, val);
  typedef dmc::DirichletVariationalClosure ClosureType;
  std::vector<std::vector<double> > vparams = { {.4, .6}, {.7, .3} };
  ClosureType closure;
  closure.variational_params = &vparams;
  double f0 = dmc::dirichlet::hyperparameters_variational_objective(alpha, &closure);
  std::vector<double> alpha_pert(alpha);
  double pert = 1E-8;
  alpha_pert[0] += pert;
  double f1 = dmc::dirichlet::hyperparameters_variational_objective(alpha_pert, &closure);
  double est = (f1 - f0) / pert;
  std::vector<double> grad;
  dmc::dirichlet::hyperparameters_variational_gradient(alpha, &closure, grad);
  ASSERT_NEAR(grad[0], est, 1E-6);
}

TEST(Dirichlet, hyperparameters_lazy_hessian_check1) {
  std::vector<double> alpha = {0.1, 0.2};
  //const double tg_0_3 = 2*gsl_sf_psi_n(1, .3);
  // first, make sure it's computing what we think it's checking
  // std::vector< std::vector< double > > e_H = {
  //   {tg_0_3 - 2*gsl_sf_psi_n(1, .1), tg_0_3}, 
  //   {tg_0_3 , tg_0_3 - 2*gsl_sf_psi_n(1, .2)}
  // };
  std::vector< std::vector< double > > e_H = {
    {-178.375869209, 24.4907290922}, 
    {24.4907290922 , -28.044025318}
  };
  for(int i = 0; i < 2; ++i) {
    for(int j = 0; j < 2; ++j) {
      double h = 
	dmc::dirichlet::hyperparameters_variational_lazy_hessian(alpha,
								 i, j,
								 2);
      ASSERT_NEAR(e_H[i][j], h, 1E-6) << "Hessian off for i = " << i << " and j = " << j;
    }
  }
  // off-diagonal, common, element
  double common = dmc::dirichlet::hyperparameters_variational_lazy_hessian(alpha, 0, 1, 2);
  ASSERT_NEAR(24.4907290922, common, 1E-5);
  ASSERT_NEAR(-202.86659830, 
	      dmc::dirichlet::hyperparameters_variational_lazy_hessian(alpha, 0, 0, 2) - common, 
	      1E-5);
}
TEST(Dirichlet, hyperparameters_lazy_hessian_check2) {
  std::vector<double> alpha = {0.1 + 1E-8, 0.2};
  std::vector< std::vector< double > > e_H = {
    {-178.375830677, 24.4907275867}, 
    {24.4907275867 , -28.044026824}
  };
  for(int i = 0; i < 2; ++i) {
    for(int j = 0; j < 2; ++j) {
      double h = 
	dmc::dirichlet::hyperparameters_variational_lazy_hessian(alpha,
								 i, j,
								 2);
      ASSERT_NEAR(e_H[i][j], h, 1E-6) << "Hessian off for i = " << i << " and j = " << j;
    }
  }
}

TEST(Dirichlet, hyperparameters_lazy_hessian_check_finite_diffs) {
  std::vector<double> alpha = {0.1, 0.2};
  typedef dmc::DirichletVariationalClosure ClosureType;
  std::vector<std::vector<double> > vparams = { {.4, .6}, {.7, .3} };
  const int M = 2;
  ASSERT_EQ(M, vparams.size());
  ClosureType closure;
  closure.variational_params = &vparams;
  std::vector<double> grad;
  dmc::dirichlet::hyperparameters_variational_gradient(alpha, &closure, grad);
  const double pert = 1E-5;
  const double tol = 1E-1;
  std::vector< std::vector<double> > grad_pert(2, std::vector<double>());
  for(int k = 0; k < 2; ++k) { // which axis of alpha do we perturb?
    std::vector<double> alpha_pert(alpha);
    alpha_pert[k] += pert;
    dmc::dirichlet::hyperparameters_variational_gradient(alpha_pert, &closure, grad_pert[k]);
  }
  for(int i = 0; i < 2; ++i) {
    for(int j = 0; j < 2; ++j) {
      double h = 
	dmc::dirichlet::hyperparameters_variational_lazy_hessian(alpha,
								 i, j,
								 2);
      double fd1 = (grad_pert[i][j] - grad[i])/(2.0*pert);
      double fd2 = (grad_pert[j][i] - grad[j])/(2.0*pert);
      double fd = fd1 + fd2;
      EXPECT_NEAR(fd, h, tol) << "h_{" << i << ", " << j << "} = " << h << " doesn't match finite diff. approx. = " << fd << ", with perturbation = " << pert;
    }
  }
}

TEST(Dirichlet, optimize_hypers_variational_nr) {
  const int size = 2;
  std::vector<double> alpha = {.1, .2};
  ASSERT_EQ(2, alpha.size());
  std::vector<std::vector<double> > vparams = { {3.4, 1.2}, {0.9, 1.8} };
  typedef dmc::DirichletVariationalClosure ClosureType;
  ClosureType closure;
  closure.variational_params = &vparams;

  typedef std::vector<double> Vector;
  Vector grad(2);
  Vector hessian_diag(2);
  dmc::dirichlet::hyperparameters_variational_gradient(alpha, &closure, grad);
  ASSERT_NEAR(11.9468785386, grad[0], 1E-8);
  ASSERT_NEAR(1.358759101, grad[1], 1E-8);
  //polygamma(0, x+y)-2 polygamma(0, x)-1.89558, 2 polygamma(0, x+y)-2 polygamma(0, y)-2.21427
  ASSERT_NEAR(2*gsl_sf_psi(alpha[0] + alpha[1]) - 2*gsl_sf_psi(alpha[0]) - 1.89558,
	      grad[0], 1E-4);
  ASSERT_NEAR(2*gsl_sf_psi(alpha[0] + alpha[1]) - 2*gsl_sf_psi(alpha[1]) - 2.21427,
	      grad[1], 1E-4);
  // get the diagonal of the Hessian
  dmc::dirichlet::hyperparameters_variational_hessian_diag(alpha, 2, hessian_diag);
  // get the off-diagonal of the Hessian
  double hessian_odiag = dmc::dirichlet::hyperparameters_variational_lazy_hessian(alpha, 0, 1, 2);
  ASSERT_NEAR(24.4907290922, hessian_odiag, 1E-5);
  // substract off the common value
  isage::util::sum(-1*hessian_odiag, &hessian_diag);
  ASSERT_NEAR(-202.86659830,  hessian_diag[0], 1E-5);
  ASSERT_NEAR(-52.5347544101, hessian_diag[1], 1E-5);

  // Now test the optimization
  double f_init = dmc::dirichlet::hyperparameters_variational_objective(alpha, &closure);
  ASSERT_NEAR(-1.884514552765, f_init, 1E-8);
  std::vector<double> point = dmc::dirichlet::hyperparameters_variational_nr(alpha, vparams);
  bool eq = true;
  for(int i = 0; i < size; ++i) {
    eq &= std::abs(alpha[i] - point[i]) < 1E-8;
  }
  ASSERT_FALSE(eq);
  double f_end = dmc::dirichlet::hyperparameters_variational_objective(point, &closure);
  ASSERT_GT(f_end, f_init);
}

// TEST(DMC, optimize_hypers_variational_gsl_multimin) {
//   double val = 1;
//   const int size = 2;
//   std::vector<double> alpha(size, val);
//   dmc::dmc my_dmc(alpha);
//   std::vector<std::vector<double> > vparams = { {3.4, 1.2}, {0.9, 1.8} };
//   my_dmc.reestimate_hyperparameters_variational(vparams);
  
// }
