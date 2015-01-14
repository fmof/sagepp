#include "mathops.hpp"

namespace mathops {
  const double NEGATIVE_INFINITY = - std::numeric_limits<double>::infinity();
  const gsl_rng_type *which_gsl_rng = gsl_rng_mt19937;
  gsl_rng *rnd_gen = gsl_rng_alloc(which_gsl_rng);
}
