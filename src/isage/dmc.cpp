#include "dmc.hpp"
#include "mathops.hpp"
#include "logging.hpp"
#include "util.hpp"

#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_psi.h>

#include <vector>

namespace dmc {
  /**
   * Compute the gradient of the log normalizer A(hypers):
   * 
   *   ∂A(hypers)
   *   ---------- = Ψ(hyper_i) - Ψ(\sum_j hyper_j)
   *       ∂i
   *
   * where Ψ(x) is the digamma function (derivative of the log Gamma).
   */
  std::vector<double> dmc::grad_log_partition() {
    std::vector<double> grad(hyperparameters_.size());
    const double digsum = gsl_sf_psi(hyperparameter_sum_);
    for(const auto& hyper_ : hyperparameters_) {
      grad.push_back(gsl_sf_psi(hyper_) - digsum);      
    }
    return grad;
  }

  /**
   * This implementation of Wallach's "Method 1" uses an iterative
   * fixed-point algorithm to reestimate the hyperparameters.
   */
  void dmc::reestimate_hyperparameters_wallach1(const std::vector<std::vector<int> >& counts,
						int num_iterations, double floor) {
    const std::vector< std::vector< int > > histogram = isage::util::histogram(counts);
    const std::vector<int> marginals = isage::util::marginals(counts);
    const std::map<int, int> marginal_histogram = isage::util::sparse_histogram(marginals);
    const int K = counts[0].size();
    const std::vector<int> column_max = isage::util::column_max<int>(counts);
    const int max_marginal = isage::util::max(marginals);
    for(int iter = 0; iter < num_iterations; ++iter) {
      double d = 0.0;
      double s = 0.0;
      for(int idx = 1; idx <= max_marginal; ++idx) {
	d += 1.0/(idx - 1 + hyperparameter_sum_);
	s += (marginal_histogram.find(idx) == marginal_histogram.end() ? 0 : marginal_histogram.at(idx))*d;
      }
      hyperparameter_sum_ = 0.0;
      for(int k = 0; k < K; ++k) {
	const std::vector<int>& k_histogram = histogram[k];
	const int c_max = column_max[k];
	const double h_k = hyperparameters_[k];
	d = 0.0;
	double sk = 0.0;
	for(int n = 1; n <= c_max; ++n) {
	  d += 1.0/(n - 1 + h_k);
	  sk += k_histogram[n]*d;
	}
	double y = h_k * sk / s;
	hyperparameters_[k] = (y > floor) ? y : floor;
	hyperparameter_sum_ += hyperparameters_[k];
      }
    }
  }

  /**
   * This implementation of Wallach's "Method 1" uses an iterative
   * fixed-point algorithm to reestimate the hyperparameters.
   */
  void mfgdmc::reestimate_hyperparameters_wallach1(const std::vector<std::vector<std::vector<int> > >& counts,
						   int num_iterations, double floor) {
    const std::vector< std::vector< int > > histogram = isage::util::mf_histogram(counts);
    const std::vector<int> marginals = isage::util::marginals(counts);
    const std::map<int, int> marginal_histogram = isage::util::sparse_histogram(marginals);
    const int K = counts[0][0].size();
    const std::vector<int> column_max = isage::util::column_max<int>(counts);
    const int max_marginal = isage::util::max(marginals);
    for(int iter = 0; iter < num_iterations; ++iter) {
      double d = 0.0;
      double s = 0.0;
      for(int idx = 1; idx <= max_marginal; ++idx) {
	d += 1.0/(idx - 1 + hyperparameter_sum_);
	s += (marginal_histogram.find(idx) == marginal_histogram.end() ? 0 : marginal_histogram.at(idx))*d;
      }
      hyperparameter_sum_ = 0.0;
      for(int k = 0; k < K; ++k) {
	const std::vector<int>& k_histogram = histogram[k];
	const int c_max = column_max[k];
	const double h_k = hyperparameters_[k];
	d = 0.0;
	double sk = 0.0;
	for(int n = 1; n <= c_max; ++n) {
	  d += 1.0/(n - 1 + h_k);
	  sk += k_histogram[n]*d;
	}
	double y = h_k * sk / s;
	hyperparameters_[k] = (y > floor) ? y : floor;
	hyperparameter_sum_ += hyperparameters_[k];
      }
    }
  }

  /**
   * Compute relevant ratio of, e.g.,
   * log(Gamma(x+y) / Gamma(x)) by expanding out
   *   Gamma(x+y) = \prod_{i = 0}^{y-1} (x + y - i) * Gamma(x),
   * dividing out Gamma(x), and taking logs.
   */
  const double dmc::log_u_conditional_oracle(const double base_value,
					     const double sum,
					     const int num_to_remove) {
    const double nsum = (double)(base_value + num_to_remove - 1);
    double numerator = 0.0;
    int i;
    // this handles the numerator
    for(i = 0; i < num_to_remove; i++) {
      numerator += gsl_sf_log(nsum - i);
    }
    // now handle the denom.
    double denominator = 0.0;
    double dsum = (double)(sum + num_to_remove - 1);
    for(i = 0; i < num_to_remove; i++) {
      denominator += gsl_sf_log(dsum - i);
    }
    return numerator - denominator;
  }

  /**
   * Compute relevant ratio of, e.g.,
   * log(Gamma(x+y) / Gamma(x)) by expanding out
   *   Gamma(x+y) = \prod_{i = 0}^{y-1} (x + y - i) * Gamma(x),
   * dividing out Gamma(x), and taking logs.
   */
  const double dmc::log_u_conditional_oracle(const size_t idx,
					     const int* const histogram,
					     const int sum,
					     const int num_to_remove) {
    return dmc::log_u_conditional_oracle(histogram[idx] + hyperparameters_[idx],
					 sum + hyperparameter_sum_, num_to_remove);
  }
  /**
   * Compute relevant ratio of, e.g.,
   * log(Gamma(x+y) / Gamma(x)) by expanding out
   *   Gamma(x+y) = \prod_{i = 0}^{y-1} (x + y - i) * Gamma(x),
   * dividing out Gamma(x), and taking logs.
   */
  const double dmc::log_u_conditional_oracle(const size_t idx,
					     const std::vector<int>& histogram,
					     const int sum,
					     const int num_to_remove) {
    return dmc::log_u_conditional_oracle(histogram[idx] + hyperparameters_[idx],
					 sum + hyperparameter_sum_, num_to_remove);
  }

  /**
   * Compute relevant ratio of, e.g., log(Gamma(histogram[idx]+y) / Gamma(x))
   * with GSL's gsl_sf_lngamma function.
   */
  const double dmc::log_u_conditional_gsl(const double base_value,
					  const double sum,
					  const int num_to_remove) {
    const double numerator = gsl_sf_lngamma((double)(base_value + num_to_remove)) - gsl_sf_lngamma((double)(base_value));
    const double denominator = gsl_sf_lngamma((double)(sum + num_to_remove)) - gsl_sf_lngamma((double)sum);
    return numerator - denominator;
  }

  /**
   * Compute relevant ratio of, e.g., log(Gamma(histogram[idx]+y) / Gamma(x))
   * with GSL's gsl_sf_lngamma function.
   */
  const double dmc::log_u_conditional_gsl(const size_t idx,
					  const int* const histogram,
					  const int sum,
					  const int num_to_remove) {
    return dmc::log_u_conditional_gsl(histogram[idx] + hyperparameters_[idx],
				      sum + hyperparameter_sum_, num_to_remove);
  }
  /**
   * Compute relevant ratio of, e.g., log(Gamma(histogram[idx]+y) / Gamma(x))
   * with GSL's gsl_sf_lngamma function.
   */
  const double dmc::log_u_conditional_gsl(const size_t idx,
					  const std::vector<int>& histogram,
					  const int sum,
					  const int num_to_remove) {
    return dmc::log_u_conditional_gsl(histogram[idx] + hyperparameters_[idx],
				      sum + hyperparameter_sum_, num_to_remove);
  }

  /**
   * Compute log(Gamma(x+y) / Gamma(x)) by using Sterling's approximation
   * for log(n) ~= n*log(n) - n.
   */
  const double dmc::log_u_conditional_sterling(const double base_value,
					       const double sum,
					       const int num_to_remove) {
    const double numerator = mathops::sterling((double)(base_value + num_to_remove)) - mathops::sterling((double)(base_value));
    const double denominator = mathops::sterling((double)(sum + num_to_remove)) - mathops::sterling((double)sum);
    return numerator - denominator;
  }

  /**
   * Compute log(Gamma(x+y) / Gamma(x)) by using Sterling's approximation
   * for log(n) ~= n*log(n) - n.
   */
  const double dmc::log_u_conditional_sterling(const size_t idx,
					       const int* const histogram,
					       const int sum,
					       const int num_to_remove) {
    return dmc::log_u_conditional_sterling(histogram[idx] + hyperparameters_[idx],
					   sum + hyperparameter_sum_, num_to_remove);
  }
  /**
   * Compute log(Gamma(x+y) / Gamma(x)) by using Sterling's approximation
   * for log(n) ~= n*log(n) - n.
   */
  const double dmc::log_u_conditional_sterling(const size_t idx,
					       const std::vector<int>& histogram,
					       const int sum,
					       const int num_to_remove) {
    return dmc::log_u_conditional_sterling(histogram[idx] + hyperparameters_[idx],
					   sum + hyperparameter_sum_, num_to_remove);
  }


  /**
   * This computes the log of the unnormalized conditional:
   *
   *            Gamma( x + c)
   *            -------------
   *               Gamma(x)
   *  log    -------------------
   *          Gamma(sum x_j + c)
   *            --------------
   *            Gamma(sum x_j)
   *
   * where x_i = histogram[idx], and sum = \sum_j x_j.
   *
   * By default, this calls
   *   log_u_conditional_oracle     if 1 <= num_to_remove <= 4
   *   log_u_conditional_gsl        if num_to_remove >= 5
   *
   * These can be changed by setting:
   *   - use_gsl(bool)
   *   - use_sterling(bool)
   */
  const double dmc::log_u_conditional(const size_t idx,
				      const int* const histogram,
				      const int sum,
				      const int num_to_remove) {
    if(num_to_remove <= 4 || (!use_gsl_ && !use_sterling_)) {
      return log_u_conditional_oracle(idx, histogram, sum, num_to_remove);
    } else {
      if(use_gsl_) {
	return log_u_conditional_gsl(idx, histogram, sum, num_to_remove);
      } else {
	// call sterling function
	return log_u_conditional_sterling(idx, histogram, sum, num_to_remove);
      }
    }
  }

  /**
   * This computes the log of the unnormalized conditional:
   *
   *            Gamma( x + c)
   *            -------------
   *               Gamma(x)
   *  log    -------------------
   *          Gamma(sum x_j + c)
   *            --------------
   *            Gamma(sum x_j)
   *
   * where x_i = histogram[idx], and sum = \sum_j x_j.
   *
   * By default, this calls
   *   log_u_conditional_oracle     if 1 <= num_to_remove <= 4
   *   log_u_conditional_gsl        if num_to_remove >= 5
   *
   * These can be changed by setting:
   *   - use_gsl(bool)
   *   - use_sterling(bool)
   */
  const double dmc::log_u_conditional(const size_t idx,
				      const std::vector<int>& histogram,
				      const int sum,
				      const int num_to_remove) {
    if(num_to_remove <= 4 || (!use_gsl_ && !use_sterling_)) {
      return log_u_conditional_oracle(idx, histogram, sum, num_to_remove);
    } else {
      if(use_gsl_) {
	return log_u_conditional_gsl(idx, histogram, sum, num_to_remove);
      } else {
	// call sterling function
	return log_u_conditional_sterling(idx, histogram, sum, num_to_remove);
      }
    }
  }

  ///////////////////////////////////////////////////////////

  const double gdmc::log_joint(const std::vector< std::vector<int> >& counts) {
    const int domain_size = this->size();
    double lgamma_hp_i = 0.0;
    double lgamma_num_i = 0.0;
    double lgamma_num_strata = 0.0;
    recompute_hyperparameter_sum();
    double log_prod = num_strata_ * gsl_sf_lngamma(hyperparameter_sum_);
    for(int i = 0; i < domain_size; ++i) {
      lgamma_hp_i += gsl_sf_lngamma(hyperparameters_[i]);
    }
    for(int strata = 0; strata < num_strata_; ++strata) {
      const std::vector<int> strata_counts = counts[strata];
      double strata_sum = 0.0;
      for(int dom_index = 0; dom_index < domain_size; ++dom_index) {
	double x = strata_counts[dom_index] + hyperparameters_[dom_index];
	strata_sum += x;
	lgamma_num_i += gsl_sf_lngamma(x);
      }
      lgamma_num_strata += gsl_sf_lngamma(strata_sum);
    }
    log_prod += lgamma_num_strata - lgamma_num_i - num_strata_*lgamma_hp_i;
    return log_prod;
  }

  const double mfgdmc::log_joint(const std::vector< std::vector< std::vector<int> > >& counts) {
    const int domain_size = this->size();
    double lgamma_hp_i = 0.0;
    double lgamma_num_i = 0.0;
    double lgamma_num_strata = 0.0;
    recompute_hyperparameter_sum();
    int num_computed_strata = num_entries();
    double log_prod = num_computed_strata * gsl_sf_lngamma(hyperparameter_sum_);
    for(int i = 0; i < domain_size; ++i) {
      lgamma_hp_i += gsl_sf_lngamma(hyperparameters_[i]);
    }
    for(int strata = 0; strata < num_strata_1_; ++strata) {
      for(int strata2 = 0; strata2 < num_strata_2_[strata]; ++strata2) {
	const std::vector<int> strata_counts = counts[strata][strata2];
	double strata_sum = 0.0;
	for(int dom_index = 0; dom_index < domain_size; ++dom_index) {
	  double x = strata_counts[dom_index] + hyperparameters_[dom_index];
	  strata_sum += x;
	  lgamma_num_i += gsl_sf_lngamma(x);
	}
	lgamma_num_strata += gsl_sf_lngamma(strata_sum);
      }
    }
    log_prod += lgamma_num_strata - lgamma_num_i - num_computed_strata*lgamma_hp_i;
    return log_prod;
  }
}
