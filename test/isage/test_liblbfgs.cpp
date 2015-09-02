#include "gtest/gtest.h"

#include <map>
#include <string>
#include <vector>

#include <stdio.h>
#include <math.h>
#include <lbfgs.h>

typedef lbfgsfloatval_t flt;

const int counts[2] = {2, 1};
flt evaluate(void *instance, const flt *x,
	     flt *g, const int n, const flt step) {
  flt z = 0.;
  for (int i = 0; i < n; ++i)
    z += exp(x[i]);

  flt lz = log(z);

  flt fx = 0.;
  for (int i = 0; i < n; ++i)
    fx -= counts[i] * (x[i] - lz);

  for (int i = 0; i < n; ++i)
    g[i] = -counts[i] + 3 * exp(x[i])/z;

  return fx;
}

int progress(void *instance,
	     const flt *x,
	     const flt *g,
	     const flt fx,
	     const flt xnorm,
	     const flt gnorm,
	     const flt step,
	     int n,
	     int k,
	     int ls) {
  return 0;
}


class SimpleLibLBFGSMaxent {
private:
  const int num_dim_;
  lbfgs_parameter_t params;
  const int counts[2] = {2, 1};

public:
  SimpleLibLBFGSMaxent() : num_dim_(2) {
    lbfgs_parameter_init(&params);
    params.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
  }

  int optimize(flt *x, flt *f_val = NULL) {
    flt fx;
    int ret = lbfgs(2, x, &fx, _evaluate,
		    _progress, this, &params);
    if(f_val != NULL) {
      *f_val = fx;
    }
    return ret;
  }
  static lbfgsfloatval_t _evaluate(void *instance,
				   const lbfgsfloatval_t *x,
				   lbfgsfloatval_t *g,
				   const int n,
				   const lbfgsfloatval_t step) {
    return reinterpret_cast<SimpleLibLBFGSMaxent*>(instance)->evaluate(x, g, n, step);
  }
  flt evaluate(const flt *x, flt *g, const int n, const flt step) {
    flt z = 0.;
    for (int i = 0; i < n; ++i)
      z += exp(x[i]);

    flt lz = log(z);

    flt fx = 0.;
    for (int i = 0; i < n; ++i)
      fx -= this->counts[i] * (x[i] - lz);

    for (int i = 0; i < n; ++i)
      g[i] = -this->counts[i] + 3 * exp(x[i])/z;

    return fx;
  }

  static int _progress(void *instance,
		       const lbfgsfloatval_t *x,
		       const lbfgsfloatval_t *g,
		       const lbfgsfloatval_t fx,
		       const lbfgsfloatval_t xnorm,
		       const lbfgsfloatval_t gnorm,
		       const lbfgsfloatval_t step,
		       int n,
		       int k,
		       int ls) {
    return reinterpret_cast<SimpleLibLBFGSMaxent*>(instance)->progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
  }
  int progress(const flt *x,
	       const flt *g,
	       const flt fx,
	       const flt xnorm,
	       const flt gnorm,
	       const flt step,
	       int n,
	       int k,
	       int ls) {
    return 0;
  }
};

TEST(liblbfgs, simple_maxent_lm_static) {
  int ret = 0;
  flt fx;
  flt *x = lbfgs_malloc(2);
  lbfgs_parameter_t param;

  ASSERT_TRUE(x != NULL);
  x[0] = 0.;
  x[1] = 0.;

  /* Initialize the parameters for the L-BFGS optimization. */
  lbfgs_parameter_init(&param);
  param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;

  /*
        Start the L-BFGS optimization; this will invoke the callback functions
        evaluate() and progress() when necessary.
  */
  //SimpleLibLBFGSMaxent obj;
  ret = lbfgs(2, x, &fx, evaluate,
	      progress, NULL, &param);

  /* Report the result. */
  ASSERT_EQ(0, ret);
  ASSERT_NEAR(0.34, x[0], 1E-2);
  ASSERT_NEAR(-0.34, x[1], 1E-2);
  lbfgs_free(x);
}

TEST(liblbfgs, simple_maxent_lm_class_member_fn) {
  flt *x = lbfgs_malloc(2);

  ASSERT_TRUE(x != NULL);
  x[0] = 0.;
  x[1] = 0.;

  SimpleLibLBFGSMaxent obj;
  int ret = obj.optimize(x);

  /* Report the result. */
  ASSERT_EQ(0, ret);
  ASSERT_NEAR(0.34, x[0], 1E-2);
  ASSERT_NEAR(-0.34, x[1], 1E-2);
  lbfgs_free(x);
}
