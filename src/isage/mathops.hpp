/**
 * This provides a library for the
 * Dirichlet-Multinomial Compound 
 * distribution. 
 */

#ifndef ISAGE_MATH_OPS_H_
#define ISAGE_MATH_OPS_H_

#include "gsl/gsl_math.h"
#include <gsl/gsl_rng.h>
#include "gsl/gsl_sf_exp.h"
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_log.h>

#include "logging.hpp"
#include <iostream>

#include <limits>
#include <vector>

namespace mathops {
  extern const gsl_rng_type *which_gsl_rng;
  extern gsl_rng *rnd_gen;

  extern const double NEGATIVE_INFINITY;
  inline const double log_add(double lp, double lq) {
    const bool lpi = lp == NEGATIVE_INFINITY;
    const bool lqi = lq == NEGATIVE_INFINITY;
    if(!lpi & !lqi) {
      return (lq < lp) ? 
	(lp + gsl_log1p(gsl_sf_exp(lq - lp))) : 
	(lq + gsl_log1p(gsl_sf_exp(lp - lq)));
    } else if(lpi) { 
      return lq;
    } else if(lqi) {
      return lp;
    } else {
      return NEGATIVE_INFINITY;
    }
  }
  inline const double log_add(const std::vector<double>& log_probs) {
    double sum = mathops::NEGATIVE_INFINITY;
    for(double lp : log_probs) {
      sum = mathops::log_add(sum, lp);
    }
    return sum;
  }

  /**
   * Draw a number uniformly at random from the unnormalized log-prob
   * semiring. l is a **logarithmic** scaling constant.
   * 
   * 1. Draw r ~ Uniform(0, 1)
   * 2. Return x = log(r * exp(l)) = log(r) + l
   */
  inline const double sample_uniform_log(const double l) {
    const double r = gsl_rng_uniform_pos(rnd_gen);
    //really, this is log(r * exp(l)) = log(r) + log(exp(l)) + log(r) + l
    const double x = gsl_sf_log(r) + l;
    BOOST_LOG_TRIVIAL(trace) << "Generated number is " << r << " transformed to " << x;
    return x;
  }

  inline const double sample_uniform(const double l) {
    const double r = gsl_rng_uniform_pos(rnd_gen);
    const double x = r*l;
    BOOST_LOG_TRIVIAL(trace) << "Generated number is " << r << " transformed to " << x;
    return x;
  }
  inline const double add(const std::vector<double>& weights) {
    double sum = 0.0;
    for(double w : weights) {
      sum += w;
    }
    return sum;
  }

  inline const double sterling(const double value) {
    return (value * gsl_sf_log(value)) - value;
  }
}

#endif
