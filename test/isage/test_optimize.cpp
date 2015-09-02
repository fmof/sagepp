#include "gtest/gtest.h"

#include "boost/bind.hpp"
#include "gsl/gsl_multimin.h"
#include "logging.hpp"
#include "optimize.hpp"

#include <vector>

TEST(gsl_vector, to_container) {
  std::vector<int> vec;
  vec.push_back(5);
  vec.push_back(7);

  optimize::GSLVector gvec = optimize::GSLVector(vec);
  std::vector<int> res = gvec.to_container<std::vector<int> >();

  ASSERT_EQ(2, res.size());
  ASSERT_NEAR(5, res[0], 1E-6);
  ASSERT_NEAR(7, res[1], 1E-6);
}
TEST(gsl_vector, sum_std_vector) {
  std::vector<double> vec = {5.0, 2.0, 10.0};
  std::vector<double> vec1 = {1.0, 2.0, 9.0};
  optimize::GSLVector gvec(vec);
  std::vector<double> res = optimize::GSLVector::sum(gvec.get(), &vec1);
  ASSERT_NEAR(6.0, res[0], 1E-6);
  ASSERT_NEAR(4.0, res[1], 1E-6);
  ASSERT_NEAR(19.0, res[2], 1E-6);
}
TEST(gsl_vector, dist) {
  std::vector<double> vec0 = {0.0, 0.0};
  std::vector<double> vec1 = {1.0, 1.0};
  ASSERT_NEAR(sqrt(2.0), 
	      optimize::GSLVector::dist(optimize::GSLVector(vec0).get(),
					optimize::GSLVector(vec1).get()), 
	      1E-5);
}

/**
 * This computes the function
 *
 *     A*(x-a)^2 + B(y-b)^2 + C,
 * where
 *  fparams = [a, b, A, B, C]
 *
 */
double paraboloid2D(const gsl_vector* x, void *fparams){
  const double *p = (const double *)fparams;
  const double y = gsl_vector_get(x, 0);
  const double z = gsl_vector_get(x, 1);
  return p[2]*(y - p[0])*(y-p[0]) + p[3]*(z-p[1])*(z-p[1]) + p[4];
}
void paraboloid2DGrad(const gsl_vector* v, void *fparams, gsl_vector *grad) {
  const double *x = (const double*)(v->data);
  const double *p = (const double*)fparams;
  gsl_vector_set(grad, 0, 2*(x[0] - p[0])*p[2]);
  gsl_vector_set(grad, 1, 2*(x[1] - p[1])*p[3]);
}
void paraboloid2DFD(const gsl_vector* x, void * params, 
		    double * f, gsl_vector* g) {
  *f = paraboloid2D(x, params);
  paraboloid2DGrad(x, params, g);
}
TEST(gsl_minimizer, paraboloid_cstyle) {
  #ifndef ISAGE_LOG_AS_COUT
  boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
  #endif
  double par[5] = { 1.0, 2.0, 10.0, 20.0, 30.0 };
  gsl_multimin_function_fdf my_func;
  my_func.n = 2;  /* number of function components */
  my_func.f = &paraboloid2D;
  my_func.df = &paraboloid2DGrad;
  my_func.fdf = &paraboloid2DFD;
  my_func.params = (void *)par;

  std::vector<double> point = {5.0, 7.0};
  optimize::GSLMinimizer optimizer(2);
  int opt_status = optimizer.minimize(&my_func, point);
  ASSERT_EQ(opt_status, GSL_SUCCESS);
  ASSERT_NEAR(1.0,  point[0], 1E-6);
  ASSERT_NEAR(2.0,  point[1], 1E-6);
  ASSERT_NEAR(30.0, optimizer.value(), 1E-6);
}

class Paraboloid2D {
private:
  double par[5] = {1.0, 2.0, 10.0, 20.0, 30.0};
public:
  static double paraboloid2D(const gsl_vector* x, void *fparams){
    const double *p = (const double *)fparams;
    const double y = gsl_vector_get(x, 0);
    const double z = gsl_vector_get(x, 1);
    return p[2]*(y - p[0])*(y-p[0]) + p[3]*(z-p[1])*(z-p[1]) + p[4];
  }
  static void paraboloid2DGrad(const gsl_vector* v, void *fparams, gsl_vector *grad) {
    const double *x = (const double*)(v->data);
    const double *p = (const double*)fparams;
    gsl_vector_set(grad, 0, 2*(x[0] - p[0])*p[2]);
    gsl_vector_set(grad, 1, 2*(x[1] - p[1])*p[3]);
  }
  static void paraboloid2DFD(const gsl_vector* x, void * params, 
		      double * f, gsl_vector* g) {
    *f = paraboloid2D(x, params);
    paraboloid2DGrad(x, params, g);
  }
  gsl_multimin_function_fdf get() {
    gsl_multimin_function_fdf my_func;
    my_func.n = 2;  /* number of function components */
    my_func.f = &Paraboloid2D::paraboloid2D;
    my_func.df = &Paraboloid2D::paraboloid2DGrad;
    my_func.fdf = &Paraboloid2D::paraboloid2DFD;
    my_func.params = (void*)par;
    return my_func;
  }
};
TEST(gsl_minimizer, paraboloid_cppstyle) {
  #ifndef ISAGE_LOG_AS_COUT
  boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
  #endif
  Paraboloid2D paraboloid;  
  gsl_multimin_function_fdf my_func = paraboloid.get();
  std::vector<double> point = {5.0, 7.0};
  optimize::GSLMinimizer optimizer(2);
  int opt_status = optimizer.minimize(&my_func, point);
  ASSERT_EQ(opt_status, GSL_SUCCESS);
  ASSERT_NEAR(1.0,  point[0], 1E-6);
  ASSERT_NEAR(2.0,  point[1], 1E-6);
  ASSERT_NEAR(30.0, optimizer.value(), 1E-6);
}

class SimpleLibLBFGSMaxent {
private:
  typedef lbfgsfloatval_t flt;
  const int counts[2] = {2, 1};

public:
  SimpleLibLBFGSMaxent() {
  }

  static flt evaluate(void *instance, const flt *x, flt *g, const int n, const flt step) {
    //int *i_counts = (int*)instance;
    int i_counts[2] = {2, 1};
    flt z = 0.;
    for (int i = 0; i < n; ++i)
      z += exp(x[i]);

    flt lz = log(z);

    flt fx = 0.;
    for (int i = 0; i < n; ++i)
      fx -= i_counts[i] * (x[i] - lz);

    for (int i = 0; i < n; ++i)
      g[i] = -i_counts[i] + 3 * exp(x[i])/z;

    return fx;
  }

  static int progress(void *instance, const flt *x, const flt *g,
		      const flt fx, const flt xnorm,
		      const flt gnorm, const flt step,
		      int n, int k, int ls) {
    return 0;
  }

  optimize::LibLBFGSFunction objective() {
    optimize::LibLBFGSFunction func;
    func.eval = &SimpleLibLBFGSMaxent::evaluate;
    func.progress = &SimpleLibLBFGSMaxent::progress;
    func.params = (void*)(this->counts);
    return func;
  }
};
TEST(liblbfgs_minimizer, maxent2_cppstyle) {
  SimpleLibLBFGSMaxent maxent;  
  optimize::LibLBFGSFunction my_func = maxent.objective();
  std::vector<double> point = {0.0, 0.0};
  optimize::LibLBFGSMinimizer optimizer(2);
  int opt_status = optimizer.minimize(&my_func, point);
  ASSERT_EQ(0, opt_status);
  ASSERT_NEAR(0.34,  point[0], 1E-2);
  ASSERT_NEAR(-0.34,  point[1], 1E-2);
}

