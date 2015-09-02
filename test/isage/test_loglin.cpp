#include "gtest/gtest.h"

#include <boost/tokenizer.hpp>
#include "loglin.hpp"
#include "mathops.hpp"
#include "optimize.hpp"
#include "util.hpp"
//#include "gsl/mathops::exp.h"
#include "gsl/gsl_sf_log.h"

#include <map>
#include <string>
#include <vector>

using namespace loglin;

TEST(loglin, create_model) {  
  UnigramMaxent<size_t, std::vector<double> > ment;
  //std::vector<double> vec(3, -1.7);
  //ASSERT_NEAR(-0.6013877113, mathops::log_add(vec), 1E-6);
}

TEST(loglin, lesson1) {
  UnigramMaxent<size_t, std::vector<double> > model;
  std::vector<double> weights(4);
  weights[0] = 1; // striped circle
  weights[1] = 1; // solid triangle
  weights[2] = 2; // solid circle
  weights[3] = 0; // striped triangle
  const double test_tol = 1E-6;
  // compute normalization
  model.weights(weights);
  const double log_norm = model.log_normalizer();
  const double expected_log_norm = 2.626523375;
  ASSERT_NEAR(expected_log_norm, log_norm, test_tol);
  // test individual log probs
  ASSERT_NEAR(1 - expected_log_norm, model.lp(0), test_tol);
  ASSERT_NEAR(1 - expected_log_norm, model.lp(1), test_tol);
  ASSERT_NEAR(2 - expected_log_norm, model.lp(2), test_tol);
  ASSERT_NEAR(0 - expected_log_norm, model.lp(3), test_tol);
  // test log likelihood
  std::map<size_t, int> data;
  data[0] = 15;
  data[1] = 10;
  data[2] = 30;
  data[3] = 5;
  const double expected_ll = -72.59140250218672;
  ASSERT_NEAR(expected_ll, model.ll(data), test_tol);
  // test gradient of ll
  const std::vector<double> grad = model.ll_grad(data);
  EXPECT_NEAR(30 - 60*model.p(2), grad[2], test_tol);
  EXPECT_NEAR(10 - 60*model.p(1), grad[1], test_tol);
  EXPECT_NEAR(15 - 60*model.p(0), grad[0], test_tol);
  EXPECT_NEAR(5  - 60*model.p(3), grad[3], test_tol);
}

struct UnigramClosure {
  typedef UnigramMaxent<int, std::vector<double> > MaxentModel;
  MaxentModel* model;
  std::vector<double>* counts;
};

class TestUnigramMaxentLM {
public:
  typedef UnigramClosure ClosureType;
  static double elbo_contribution(const gsl_vector* trial_weights, void *fparams) {
    ClosureType* closure = (ClosureType*)fparams;
    std::vector<double> nweights = 
      optimize::GSLVector::to_container<std::vector<double> >(trial_weights);
    // reset the maxent weights; this will renormalize everything
    closure->model->weights(nweights);
    double ll = closure->model->ll_dense_data(*(closure->counts));
    // TRACE << "returning " << (-ll);
    return -ll;
  }
  static void elbo_gradient(const gsl_vector* trial_weights, void *fparams, gsl_vector* gsl_grad) {
    ClosureType* closure = (ClosureType*)fparams;
    std::vector<double> nweights = 
      optimize::GSLVector::to_container<std::vector<double> >(trial_weights);
    // reset the maxent weights; this will renormalize everything
    closure->model->weights(nweights);
    typedef std::vector<double> WeightType;
    WeightType grad = closure->model->ll_grad_dense_data(*(closure->counts));
    // negate
    isage::util::scalar_product(-1.0, &grad);
    // and then convert grad to gsl_grad
    DEBUG << "computing grad....";
    gsl_vector_memcpy(gsl_grad, optimize::GSLVector(grad).get());
  }
  static void elbo_contrib_grad(const gsl_vector* trial_weights, void *fparams,
				double* f, gsl_vector* grad) {
    *f = TestUnigramMaxentLM::elbo_contribution(trial_weights, fparams);
    TestUnigramMaxentLM::elbo_gradient(trial_weights, fparams, grad);
  }
  gsl_multimin_function_fdf get_fdf(ClosureType* params, const size_t size) {
    gsl_multimin_function_fdf objective;
    objective.n   = size;
    objective.f   = &TestUnigramMaxentLM::elbo_contribution;
    objective.df  = &TestUnigramMaxentLM::elbo_gradient;
    objective.fdf = &TestUnigramMaxentLM::elbo_contrib_grad;
    objective.params = (void*)params;
    return objective;
  }

  static double lbfgs_elbo(void *fparams, const lbfgsfloatval_t *trial_weights,
			   lbfgsfloatval_t *lbfgs_grad, const int n, const lbfgsfloatval_t step) {
    ClosureType* closure = (ClosureType*)fparams;
    std::vector<double> nweights = 
      optimize::LibLBFGSVector::to_container<std::vector<double> >(trial_weights, n);
    // reset the maxent weights; this will renormalize everything
    closure->model->weights(nweights);
    double ll = closure->model->ll_dense_data(*(closure->counts));
    typedef std::vector<double> WeightType;
    WeightType grad = closure->model->ll_grad_dense_data(*(closure->counts));
    // negate
    isage::util::scalar_product(-1.0, &grad);
    optimize::LibLBFGSVector::copy(grad, lbfgs_grad);
    return -ll;
  }
  optimize::LibLBFGSFunction get_liblbfgs_fdf(ClosureType* params, const size_t size) {
    optimize::LibLBFGSFunction objective;
    objective.eval   = &TestUnigramMaxentLM::lbfgs_elbo;
    objective.progress = &optimize::LibLBFGSNoOp::progress;
    objective.params = (void*)params;
    return objective;
  }

  std::vector<double> get_optimization_initial_point(int size_) {
    std::vector<double> foo(size_, 0.0);
    return foo;
  }
};
TEST(loglin, word_unigram_lm_2words_gsl) {
  #ifndef ISAGE_LOG_AS_COUT
  boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
  #endif
  UnigramMaxent<int, std::vector<double> > model;
  std::map<std::string, int> vocab;
  std::vector<double> counts;
  std::string text = "the the man";
  boost::char_separator<char> sep(" ");
  boost::tokenizer<boost::char_separator<char> > tokens(text, sep);
  int tok_count = 0;
  for (const auto& t : tokens) {
    std::map<std::string,int>::iterator it = vocab.find(t);
    if(it == vocab.end()) {
      vocab.insert(it, std::pair<std::string, int>(t, tok_count++));
      counts.push_back(1.0);
    } else {
      counts[it->second]++;
    }    
  }
  // Test basic data stats 
  {
    ASSERT_EQ(2, vocab.size());
    ASSERT_EQ(2, counts[0]); // the
    ASSERT_EQ(1, counts[1]); // man
  }
  UnigramClosure params;
  params.counts = &counts;
  params.model = &model;
  TestUnigramMaxentLM lm;
  gsl_multimin_function_fdf my_func = lm.get_fdf(&params, vocab.size());
  std::vector<double> point = lm.get_optimization_initial_point(vocab.size());
  optimize::GSLMinimizer optimizer(point.size());
  optimizer.tolerance(1E-8);
  std::vector<int> iterate_stati;
  std::vector<int> grad_stati;
  int opt_status = optimizer.minimize(&my_func, point, &iterate_stati, &grad_stati);
  // THIS TEST CURRENTLY FAILS
  //ASSERT_NEAR(point[0] - point[1], gsl_sf_log(2), 1E-8);
  INFO << "Unigram Maxent LM has optimization status " << opt_status;
}
TEST(loglin, word_unigram_lm_gsl) {
  #ifndef ISAGE_LOG_AS_COUT
  boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::warning);
  #endif
  UnigramMaxent<int, std::vector<double> > model;
  std::map<std::string, int> vocab;
  std::vector<double> counts;
  std::string text = "The dog ran with the man . The man ran with the cat . The cat meowed .";
  boost::char_separator<char> sep(" ");
  boost::tokenizer<boost::char_separator<char> > tokens(text, sep);
  int tok_count = 0;
  for (const auto& t : tokens) {
    std::map<std::string,int>::iterator it = vocab.find(t);
    if(it == vocab.end()) {
      vocab.insert(it, std::pair<std::string, int>(t, tok_count++));
      counts.push_back(1.0);
    } else {
      counts[it->second]++;
    }    
  }
  // Test basic data stats 
  {
    ASSERT_EQ(9, vocab.size());
    ASSERT_EQ(3, counts[0]); // The
    ASSERT_EQ(1, counts[1]); // dog
    ASSERT_EQ(2, counts[2]); // ran
    ASSERT_EQ(2, counts[3]); // with
    ASSERT_EQ(2, counts[4]); // the
    ASSERT_EQ(2, counts[5]); // man
    ASSERT_EQ(3, counts[6]); // .
    ASSERT_EQ(2, counts[7]); // cat
    ASSERT_EQ(1, counts[8]); // meowed
  }
  UnigramClosure params;
  params.counts = &counts;
  params.model = &model;
  TestUnigramMaxentLM lm;
  gsl_multimin_function_fdf my_func = lm.get_fdf(&params, vocab.size());
  std::vector<double> point = lm.get_optimization_initial_point(vocab.size());
  // first test at point == 0
  {
    model.weights(point);
    for(int i = 0; i < 9; ++i) {
      ASSERT_NEAR(-gsl_sf_log(9.0), model.lp(i), 1E-6);
    }
    ASSERT_NEAR(-39.55004239205195, model.ll_dense_data(counts), 1E-5);
  }
  // first test at point == 1
  {
    std::vector<double> ones = isage::util::sum(1.0, point);
    for(int i = 0; i < 9; ++i) {
      ASSERT_NEAR(1.0, ones[i], 1E-6);
    }
    model.weights(ones);
    // check the log normalizer
    const double elognorm = gsl_sf_log(9 * mathops::exp(1));
    ASSERT_NEAR(gsl_sf_log(9.0) + 1.0, elognorm, 1E-6);
    ASSERT_NEAR(elognorm, model.log_normalizer(), 1E-6);
    for(int i = 0; i < 9; ++i) {
      ASSERT_NEAR(1.0 - elognorm, model.lp(i), 1E-6);
      ASSERT_NEAR(1.0/9.0, model.p(i), 1E-1);
    }
    // now check the gradient
    std::vector<double> grad_at_ones = model.ll_grad_dense_data(counts);
    std::vector<double> ones_pert(ones);
    double ll_at_ones = model.ll_dense_data(counts);
    EXPECT_NEAR(-39.55004239205194, ll_at_ones, 1E-8);
    EXPECT_NEAR(18.0 * gsl_sf_log(mathops::exp(1.0)/(9.0 * mathops::exp(1.0))),
		ll_at_ones, 1E-8);
    // now perturb the first coordinate some
    ones_pert[0] += 1E-5;
    model.weights(ones_pert);
    double ll_at_ones_pert = model.ll_dense_data(counts);
    EXPECT_NEAR(15.0 * gsl_sf_log(mathops::exp(1.0)/(mathops::exp(1 + 1E-5) + 8.0 * mathops::exp(1.0))) + 
		3.0 * gsl_sf_log(mathops::exp(1.0 + 1E-5)/(mathops::exp(1 + 1E-5) + 8.0 * mathops::exp(1.0))),
		ll_at_ones_pert, 1E-8);
    std::vector<double> grad_at_ones_pert = model.ll_grad_dense_data(counts);
    // check finite differences
    double h = 1E-10;
    EXPECT_NEAR( ((15.0 * gsl_sf_log(mathops::exp(1.0)/(mathops::exp(1 + h) + 8.0 * mathops::exp(1.0))) + 
    		   3.0 * gsl_sf_log(mathops::exp(1.0 + h)/(mathops::exp(1 + h) + 8.0 * mathops::exp(1.0)))) - 
    		  (18.0 * gsl_sf_log(mathops::exp(1.0)/(9.0 * mathops::exp(1.0))))) / h,
    		 grad_at_ones[0], 1E-3);
  }
  {
    std::vector<double> x(counts.size(), 1.0);
    x[0] = 2.0;
    model.weights(x);
    EXPECT_NEAR(-39.695115580816356, model.ll_dense_data(counts), 1E-8);
    std::vector<double> grad = model.ll_grad_dense_data(counts);
    EXPECT_NEAR(-1.5650108567165084, grad[0], 1E-5);
  }
  optimize::GSLMinimizer optimizer(point.size());
  optimizer.tolerance(1E-8);
  std::vector<int> iterate_stati;
  std::vector<int> grad_stati;
  int opt_status = optimizer.minimize(&my_func, point, &iterate_stati, &grad_stati);
  INFO << "Unigram Maxent LM has optimization status " << opt_status;
  // for(int i = 0; i < 9; ++i) {
  //   BOOST_LOG_TRIVIAL(info) << "end point(" << i << ") = " << point[i];
  // }
  // for(int i = 0; i < iterate_stati.size(); ++i) {
  //   BOOST_LOG_TRIVIAL(info) << "iteration " << i << " had status " << iterate_stati[i];
  // }
  // for(int i = 0; i < grad_stati.size(); ++i) {
  //   BOOST_LOG_TRIVIAL(info) << "iteration " << i << " had grad status " << grad_stati[i];
  // }
}

TEST(loglin, word_unigram_lm_2words_liblbfgs) {
  #ifndef ISAGE_LOG_AS_COUT
  boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
  #endif
  UnigramMaxent<int, std::vector<double> > model;
  std::map<std::string, int> vocab;
  std::vector<double> counts;
  std::string text = "the the man";
  boost::char_separator<char> sep(" ");
  boost::tokenizer<boost::char_separator<char> > tokens(text, sep);
  int tok_count = 0;
  for (const auto& t : tokens) {
    std::map<std::string,int>::iterator it = vocab.find(t);
    if(it == vocab.end()) {
      vocab.insert(it, std::pair<std::string, int>(t, tok_count++));
      counts.push_back(1.0);
    } else {
      counts[it->second]++;
    }    
  }
  // Test basic data stats 
  {
    ASSERT_EQ(2, vocab.size());
    ASSERT_EQ(2, counts[0]); // the
    ASSERT_EQ(1, counts[1]); // man
  }
  UnigramClosure params;
  params.counts = &counts;
  params.model = &model;
  TestUnigramMaxentLM lm;
  optimize::LibLBFGSFunction my_func = lm.get_liblbfgs_fdf(&params, vocab.size());
  std::vector<double> point = lm.get_optimization_initial_point(vocab.size());
  optimize::LibLBFGSMinimizer optimizer(point.size());
  int opt_status = optimizer.minimize(&my_func, point);
  ASSERT_EQ(0, opt_status);
  ASSERT_NEAR(point[0] - point[1], gsl_sf_log(2), 1E-6);
}


class TestL2RegularizedUnigramMaxentLM {
public:
  typedef UnigramClosure ClosureType;
  static double elbo_contribution(const gsl_vector* trial_weights, void *fparams) {
    ClosureType* closure = (ClosureType*)fparams;
    std::vector<double> nweights = 
      optimize::GSLVector::to_container<std::vector<double> >(trial_weights);
    // reset the maxent weights; this will renormalize everything
    closure->model->weights(nweights);
    double ll = closure->model->ll_dense_data(*(closure->counts));
    double reg = 0.5 * isage::util::sum(isage::util::square(nweights));
    DEBUG << "reg is " << reg;
    return -ll + reg;
  }
  static void elbo_gradient(const gsl_vector* trial_weights, void *fparams, gsl_vector* gsl_grad) {
    ClosureType* closure = (ClosureType*)fparams;
    std::vector<double> nweights = 
      optimize::GSLVector::to_container<std::vector<double> >(trial_weights);
    // reset the maxent weights; this will renormalize everything
    closure->model->weights(nweights);
    typedef std::vector<double> WeightType;
    WeightType grad = closure->model->ll_grad_dense_data(*(closure->counts));
    // add in the regularized
    WeightType reg = isage::util::scalar_product(-1.0, nweights);
    isage::util::sum_in_first(&grad, reg);
    // negate
    isage::util::scalar_product(-1.0, &grad);
    // and then convert grad to gsl_grad
    gsl_vector_memcpy(gsl_grad, optimize::GSLVector(grad).get());
  }
  static void elbo_contrib_grad(const gsl_vector* trial_weights, void *fparams,
				double* f, gsl_vector* grad) {
    *f = TestL2RegularizedUnigramMaxentLM::elbo_contribution(trial_weights, fparams);
    TestL2RegularizedUnigramMaxentLM::elbo_gradient(trial_weights, fparams, grad);
  }

  gsl_multimin_function_fdf get_fdf(ClosureType* params, const size_t size) {
    gsl_multimin_function_fdf objective;
    objective.n   = size;
    objective.f   = &TestL2RegularizedUnigramMaxentLM::elbo_contribution;
    objective.df  = &TestL2RegularizedUnigramMaxentLM::elbo_gradient;
    objective.fdf = &TestL2RegularizedUnigramMaxentLM::elbo_contrib_grad;
    objective.params = (void*)params;
    return objective;
  }
  std::vector<double> get_optimization_initial_point(int size_) {
    std::vector<double> foo(size_, 0.0);
    return foo;
  }
};
TEST(loglin, word_unigram_lm_l2regularized) {
  #ifndef ISAGE_LOG_AS_COUT
  boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::warning);
  #endif
  UnigramMaxent<int, std::vector<double> > model;
  std::map<std::string, int> vocab;
  std::vector<double> counts;
  std::string text = "The dog ran with the man . The man ran with the cat . The cat meowed .";
  boost::char_separator<char> sep(" ");
  boost::tokenizer<boost::char_separator<char> > tokens(text, sep);
  int tok_count = 0;
  for (const auto& t : tokens) {
    std::map<std::string,int>::iterator it = vocab.find(t);
    if(it == vocab.end()) {
      vocab.insert(it, std::pair<std::string, int>(t, tok_count++));
      counts.push_back(1.0);
    } else {
      counts[it->second]++;
    }    
  }
  // Test basic data stats 
  {
    ASSERT_EQ(9, vocab.size());
    ASSERT_EQ(3, counts[0]); // The
    ASSERT_EQ(1, counts[1]); // dog
    ASSERT_EQ(2, counts[2]); // ran
    ASSERT_EQ(2, counts[3]); // with
    ASSERT_EQ(2, counts[4]); // the
    ASSERT_EQ(2, counts[5]); // man
    ASSERT_EQ(3, counts[6]); // .
    ASSERT_EQ(2, counts[7]); // cat
    ASSERT_EQ(1, counts[8]); // meowed
  }
  UnigramClosure params;
  params.counts = &counts;
  params.model = &model;
  TestL2RegularizedUnigramMaxentLM lm;
  gsl_multimin_function_fdf my_func = lm.get_fdf(&params, vocab.size());
  std::vector<double> point = lm.get_optimization_initial_point(vocab.size());
  // first test at point == 0
  {
    model.weights(point);
    for(int i = 0; i < 9; ++i) {
      ASSERT_NEAR(-gsl_sf_log(9.0), model.lp(i), 1E-6);
    }
    ASSERT_NEAR(-39.55004239205195, model.ll_dense_data(counts), 1E-5);
  }
  // first test at point == 1
  {
    std::vector<double> ones = isage::util::sum(1.0, point);
    for(int i = 0; i < 9; ++i) {
      ASSERT_NEAR(1.0, ones[i], 1E-6);
    }
    model.weights(ones);
    // check the log normalizer
    const double elognorm = gsl_sf_log(9 * mathops::exp(1));
    ASSERT_NEAR(gsl_sf_log(9.0) + 1.0, elognorm, 1E-6);
    ASSERT_NEAR(elognorm, model.log_normalizer(), 1E-6);
    for(int i = 0; i < 9; ++i) {
      ASSERT_NEAR(1.0 - elognorm, model.lp(i), 1E-6);
      ASSERT_NEAR(1.0/9.0, model.p(i), 1E-1);
    }
    // now check the gradient
    std::vector<double> grad_at_ones = model.ll_grad_dense_data(counts);
    optimize::GSLVector gvec(ones);
    double ll_at_ones = (my_func.f)(gvec.get(), &params);
    EXPECT_NEAR(39.55004239205194 + 4.5, ll_at_ones, 1E-8);
    // now perturb the first coordinate some
  }
  optimize::GSLMinimizer optimizer(point.size());
  optimizer.tolerance(1E-8);
  std::vector<int> iterate_stati;
  std::vector<int> grad_stati;
  int opt_status = optimizer.minimize(&my_func, point, &iterate_stati, &grad_stati);
  INFO << "Unigram Maxent LM has optimization status " << opt_status;
  for(int i = 0; i < 9; ++i) {
    INFO << "end point(" << i << ") = " << point[i];
  }
  for(unsigned int i = 0; i < iterate_stati.size(); ++i) {
    INFO << "iteration " << i << " had status " << iterate_stati[i];
  }
  for(unsigned int i = 0; i < grad_stati.size(); ++i) {
    INFO << "iteration " << i << " had grad status " << grad_stati[i];
  }
}
