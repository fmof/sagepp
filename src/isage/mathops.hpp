/**
 * This provides a library for the
 * Dirichlet-Multinomial Compound 
 * distribution. 
 */

#ifndef ISAGE_MATH_OPS_H_
#define ISAGE_MATH_OPS_H_

#include <gsl/gsl_math.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_psi.h>

#include "logging.hpp"
#include "util.hpp"
#include <iostream>

#include <limits>
#include <vector>

namespace mathops {
  extern const gsl_rng_type *which_gsl_rng;
  extern gsl_rng *rnd_gen;

  extern const double NEGATIVE_INFINITY;

  const double exp(const double x);

  inline const double log_add(double lp, double lq) {
    const bool lpi = lp == NEGATIVE_INFINITY;
    const bool lqi = lq == NEGATIVE_INFINITY;
    if(!lpi & !lqi) {
      return (lq < lp) ? 
	(lp + gsl_log1p(exp(lq - lp))) : 
	(lq + gsl_log1p(exp(lp - lq)));
    } else if(lpi) { 
      return lq;
    } else if(lqi) {
      return lp;
    } else {
      return NEGATIVE_INFINITY;
    }
  }
  template <typename Container>
  inline const double log_add(const Container& log_probs) {
    double sum = mathops::NEGATIVE_INFINITY;
    for(double lp : log_probs) {
      sum = mathops::log_add(sum, lp);
    }
    return sum;
  }

  template <typename Container>
  inline const double log_sum_exp(const Container& log_weights) {
    typedef typename Container::value_type T;
    const T& max = isage::util::max(log_weights);
    Container nterms = isage::util::sum(-1 * max, log_weights);
    return (double)max + log_add(nterms);
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
    TRACE << "Generated number is " << r << " transformed to " << x;
    return x;
  }

  inline const double sample_uniform(const double l) {
    const double r = gsl_rng_uniform_pos(rnd_gen);
    const double x = r*l;
    TRACE << "Generated number is " << r << " transformed to " << x;
    return x;
  }
  inline const double sample_uniform(const double lower, const double upper) {
    const double r = gsl_rng_uniform_pos(rnd_gen);
    const double x = (upper - lower)*r + lower;
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

  inline std::vector<double> rng_uniform_vector(const size_t& size) {
    std::vector<double> vec(size);
    for(size_t i = 0; i < size; ++i) {
      vec[i] = gsl_rng_uniform_pos(rnd_gen);
    }
    return vec;
  }

  template <typename V>
  inline void add_uniform_noise(V* const x, const double scale = 1.0) {
    typename V::iterator it = x->begin();
    for(;it != x->end(); ++it) {
      *it += sample_uniform(scale);
    }
  };

  template <typename V>
  inline void add_uniform_noise(V* const x, const double lower,
				const double upper) {
    typename V::iterator it = x->begin();
    for(;it != x->end(); ++it) {
      *it += sample_uniform(lower, upper);
    }
  };

  inline double truncated_lngamma(const double x, const double min) {
    return gsl_sf_lngamma(x < min ? min : x);
  }

  inline void random_choose(void * dest, size_t k, void * src, 
			    size_t n, size_t size){
    gsl_ran_choose(rnd_gen, dest, k, src, n, size);
  }
  inline void random_choose(void * dest, size_t k, void * src, 
			    size_t n, size_t size, gsl_rng *rg){
    gsl_ran_choose(rg, dest, k, src, n, size);
  }
}

namespace isage { namespace util {
    template <typename V>
    inline void exp(V* const x) {
      //typedef typename V::value_type T;
      typename V::iterator it = x->begin();
      // if(std::is_const< decltype(it) >::value) {
      // 	for(;it != x->end(); ++it) {
      // 	  T temp = *it;
      // 	  x->erase(it);
      // 	  x->insert(it, gsl_sf_exp(temp));
      // 	}
      // } else {
      for(;it != x->end(); ++it) {
	*it = mathops::exp(*it);
      }
      //      }
    };
    template <typename V>
    inline void log(V* const x) {
      for(typename V::iterator it = x->begin();
	  it != x->end(); ++it) {
	*it = gsl_sf_log(*it);
      }
    };
    template <typename V>
    inline V exp(const V& x) {
      V res(x);
      exp(&res);
      return res;
    };
    template <typename V>
    inline V log(const V& x) {
      V res(x);
      log(&res);
      return res;
    };
  }
}

#endif
