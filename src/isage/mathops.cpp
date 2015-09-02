#include "mathops.hpp"

#ifdef USE_GSL_EXP
#include <gsl/gsl_sf_exp.h>
#else
#include <cmath>
#endif

namespace mathops {
  const double NEGATIVE_INFINITY = - std::numeric_limits<double>::infinity();
  const gsl_rng_type *which_gsl_rng = gsl_rng_mt19937;
  gsl_rng *rnd_gen = gsl_rng_alloc(which_gsl_rng);

  const double exp(const double x) {
    double res = 
#ifdef USE_GSL_EXP
      gsl_sf_exp(x);
#else
    std::exp(x);
#endif
    return res;
  }
}
