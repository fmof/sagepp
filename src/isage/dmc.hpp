/**
 * This provides a library for the
 * Dirichlet-Multinomial Compound 
 * distribution. 
 */

#ifndef ISAGE_DMC_H_
#define ISAGE_DMC_H_

#include "logging.hpp"
#include "util.hpp"
#include "mathops.hpp"
#include <map>
#include <vector>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>

namespace dmc {
  struct UniformMultinomial {
  public:
    UniformMultinomial() {
    }
    std::vector<double> operator()(int support_size) {
      return std::vector<double>(support_size, 1.0/(double)support_size);
    }
    std::vector<double> operator()(const std::vector<double>& hyper) {
      return std::vector<double>(hyper.size(), 1.0/(double)hyper.size());
    }
  };
  struct UniformHyperSeedWeightedMultinomial {
  private:
    double uweight_;
    double weight_;
  public:
    UniformHyperSeedWeightedMultinomial(double unseeded_weight, double seeded_weight) : uweight_(unseeded_weight), weight_(seeded_weight) {
    }
    std::vector<double> operator()(int support_size) {
      return std::vector<double>(support_size, uweight_/(double)support_size);
    }
    std::vector<double> operator()(const std::vector<double>& hyper) {
      return isage::util::sum(hyper, std::vector<double>(hyper.size(), weight_/(double)hyper.size()));
    }
  };

  /**
   * A class representing a categorical distribution.
   */
  class cat {
  protected:
    const std::vector<double> probabilities_;
  public:
    cat(std::vector<double>& probs) : probabilities_(probs) {
    }
    inline const double& operator[](const size_t idx) const {
      return probabilities_[idx];
    }
    inline static int log_u_sample(std::vector<double> ulp) {
      const double sum = mathops::log_add(ulp);
      const double rand_draw = mathops::sample_uniform_log(sum);
      return log_u_sample_with_value(rand_draw, ulp);
    }
    inline static int log_u_sample_with_value(const double rand_draw, 
					      std::vector<double> ulp) {
      double run_sum = mathops::NEGATIVE_INFINITY;
      int container_idx = 0;
      for(std::vector<double>::iterator it = ulp.begin(); it != ulp.end(); ++it) {
	const double lp = *it;
	run_sum = mathops::log_add(run_sum, lp);
	if(rand_draw <= run_sum) return container_idx;
	++container_idx;
      }
      BOOST_LOG_TRIVIAL(debug) << "comparing run_sum = " << run_sum << " with rd " << rand_draw;
      if(fabs(rand_draw - run_sum) < 1E-8) { 
	return container_idx - 1;
      }
      BOOST_LOG_TRIVIAL(error) << "For log-probs with sum " << run_sum << " and random draw " << rand_draw << ", we got to the end of the array and did not find a result";
      for(auto x : ulp) {
	BOOST_LOG_TRIVIAL(error) << "\tvalue is " << x; 
      }
      return -1;
    }

    inline static int u_sample(double x1, double x2) {
      const double sum = x1 + x2;
      const double rand_draw = mathops::sample_uniform(sum);
      return (rand_draw <= x1) ? 0 : 1;
    }

    inline static int u_sample(std::vector<double> ulp) {
      const double sum = mathops::add(ulp);
      const double rand_draw = mathops::sample_uniform(sum);
      return u_sample_with_value(rand_draw, ulp);
    }
    inline static int u_sample_with_value(const double rand_draw, 
					  std::vector<double> ulp) {
      double run_sum = 0.0;
      int container_idx = 0;
      for(std::vector<double>::iterator it = ulp.begin(); it != ulp.end(); ++it) {
	const double w = *it;
	run_sum += w;
	if(rand_draw <= run_sum) return container_idx;
	++container_idx;
      }
      BOOST_LOG_TRIVIAL(debug) << "comparing run_sum = " << run_sum << " with rd " << rand_draw;
      if(fabs(rand_draw - run_sum) < 1E-8) { 
	return container_idx - 1;
      }
      BOOST_LOG_TRIVIAL(debug) << "For weights with sum " << run_sum << " and random draw " << rand_draw << ", we got to the end of the array and did not find a result";
      return -1;
    }
  };

  class dmc {
  private:
    bool use_gsl_ = true;
    bool use_sterling_ = false;
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
      ar & use_gsl_;
      ar & use_sterling_;
      ar & hyperparameters_;
      ar & hyperparameter_sum_;
    }
  protected:
    std::vector<double> hyperparameters_;
    double hyperparameter_sum_;
  public:
    inline const int size() {
      return hyperparameters_.size();
    }
    std::vector<double> hyperparameters() {
      return hyperparameters_;
    }
    dmc() {
    }
    dmc(int domain_size) : hyperparameters_(std::vector<double>(domain_size)) {
    }
    dmc(int domain_size, double value) : 
      hyperparameters_(std::vector<double>(domain_size, value)) {
      recompute_hyperparameter_sum();
    }
    dmc(int domain_size, const double* const values) {
      hyperparameters_ = std::vector<double>(domain_size);
      for(int i = 0; i < domain_size; i++) {
	hyperparameters_[i] = values[i];
      }
      recompute_hyperparameter_sum();
    }
    dmc(int domain_size, const std::vector<double>& values) : hyperparameters_(values) {
      recompute_hyperparameter_sum();
    }
    dmc(const std::vector<double>& values) : hyperparameters_(values) {
      recompute_hyperparameter_sum();
    }

    /**
     * Compute the gradient of the log normalizer A(hypers):
     * 
     *   ∂A(hypers)
     *   ---------- = Ψ(hyper_i) - Ψ(\sum_j hyper_j)
     *       ∂i
     *
     * where Ψ(x) is the digamma function (derivative of the log Gamma).
     */
    std::vector<double> grad_log_partition();

    /**
     * This implementation of Wallach's "Method 1" uses an iterative
     * fixed-point algorithm to reestimate the hyperparameters.
     */
    void reestimate_hyperparameters_wallach1(const std::vector<std::vector<int> >& counts,
					     int num_iterations = 200, double floor = 1E-9);


    inline void recompute_hyperparameter_sum() {
      hyperparameter_sum_ = 0.0;
      for(double x : hyperparameters_) {
	hyperparameter_sum_ += x;
      }
    }

    inline const double hyperparameter(const size_t component_idx) const {
      return hyperparameters_[component_idx];
    }

    inline void hyperparameter(const size_t component_idx, double value) {
      hyperparameters_[component_idx] = value;
    }

    /**
     * This computes the unnormalized, conditional distribution \propto
     * histogram[component_idx] + hyperparameter[component_idx]
     */
    inline const double u_conditional(const size_t component_idx, const int* const histogram) {
      return histogram[component_idx] + hyperparameters_[component_idx];
    }

    inline void use_gsl(bool b) {
      use_gsl_ = b;
      
    }
    inline void use_sterling(bool b) {
      use_sterling_ = b;
    }
    inline const bool use_gsl() {
      return use_gsl_;
    }
    inline const bool use_sterling() {
      return use_sterling_;
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
    const double log_u_conditional(const size_t idx,
				   const int* const histogram,
				   const int sum,
				   const int num_to_remove);

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
    const double log_u_conditional(const size_t idx,
				   const std::vector<int>& histogram,
				   const int sum,
				   const int num_to_remove);

    const double log_u_conditional_oracle(const size_t idx,
					  const int* const histogram,
					  const int sum,
					  const int num_to_remove);
    const double log_u_conditional_oracle(const size_t idx,
					  const std::vector<int>& histogram,
					  const int sum,
					  const int num_to_remove);
    const double log_u_conditional_oracle(const double value, const double sum,
					  const int num_to_remove);

    const double log_u_conditional_gsl(const size_t idx,
				       const int* const histogram,
				       const int sum,
				       const int num_to_remove);
    const double log_u_conditional_gsl(const size_t idx,
				       const std::vector<int>& histogram,
				       const int sum,
				       const int num_to_remove);
    const double log_u_conditional_gsl(const double value, const double sum,
				       const int num_to_remove);

    const double log_u_conditional_sterling(const size_t idx,
					    const std::vector<int>& histogram,
					    const int sum,
					    const int num_to_remove);
    const double log_u_conditional_sterling(const size_t idx,
					    const int* const histogram,
					    const int sum,
					    const int num_to_remove);
    const double log_u_conditional_sterling(const double value,  const double sum,
					    const int num_to_remove);    
  };

  /**
   * A "gated" DMC distribution, such as for the following generative story,
   * when an indicator variable z_i is given:
   *
   *   \phi_k ~ Dir(\beta)
   *   x_i | \phi, z_i ~ Cat(\phi_{z_i})
   *
   * This class is concerned with modeling p(X | \beta, Z).
   */
  class gdmc : public dmc {
  private:
    std::vector< std::vector<double> > collapsed_params_;
    int num_strata_;
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
      // serialize base class information
      ar & boost::serialization::base_object<dmc>(*this);
      ar & collapsed_params_;
      ar & num_strata_;
    }
  public:
    gdmc() {
    }
    gdmc(int domain_size, int num_strata) : dmc(domain_size),
					    num_strata_(num_strata) {
      for(int i = 0; i < num_strata_; i++) {
	std::vector<double> s(domain_size);
	collapsed_params_.push_back(s);
      }
    }
    gdmc(int domain_size, double value, int num_strata) : 
      dmc(domain_size, value), num_strata_(num_strata) {
      for(int i = 0; i < num_strata_; i++) {
	std::vector<double> s(domain_size);
	collapsed_params_.push_back(s);
      }
    }
    gdmc(int domain_size, const double* const values, int num_strata) : 
      dmc(domain_size, values), num_strata_(num_strata)  {
      for(int i = 0; i < num_strata_; i++) {
	std::vector<double> s(domain_size);
	collapsed_params_.push_back(s);
      }
    }
    gdmc(int domain_size, const std::vector<double>& values, int num_strata) : 
      dmc(domain_size, values), num_strata_(num_strata)  {
      for(int i = 0; i < num_strata_; i++) {
	std::vector<double> s(domain_size);
	collapsed_params_.push_back(s);
      }
    }

    inline std::vector< std::vector<double> > collapsed_params() {
      return collapsed_params_;
    }

    template <typename C> inline void reestimate_collapsed_parameters(const std::vector< std::vector<C> >& counts) {
      const int domain_size = this->size();
      for(int strata = 0; strata < num_strata_; ++strata) {
	const std::vector<C> strata_counts = counts[strata];
	if(domain_size > strata_counts.size()) {
	  BOOST_LOG_TRIVIAL(error) << "domain size = " << domain_size << " is greater than num columns: " << strata_counts.size();
	  throw 20;
	}
	double strata_sum = 0.0;
	for(int dom_index = 0; dom_index < domain_size; ++dom_index) {
	  strata_sum += strata_counts[dom_index] + hyperparameters_[dom_index];
	}
	for(int dom_index = domain_size - 1; dom_index >= 0; --dom_index) {
	  collapsed_params_[strata][dom_index] = (strata_counts[dom_index] + hyperparameters_[dom_index]) / strata_sum;
	}
      }
    }

    const double log_joint(const std::vector< std::vector<int> >& counts);
  };

  /**
   * A multi-faceted, "gated" DMC distribution, such as for the following generative story,
   * when an indicator variable z_i is given:
   *
   *   \phi_k ~ Dir(\beta)
   *   x_i | \phi, z_i ~ Cat(\phi_{z_i})
   *
   * This class is concerned with modeling p(X | \beta, Z).
   */
  class mfgdmc : public gdmc {
  private:
    std::vector< std::vector< std::vector<double> > > collapsed_params_;
    int num_strata_1_;
    std::vector<int> partial_sums_;
    std::vector<int> num_strata_2_;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
      // serialize base class information
      ar & boost::serialization::base_object<gdmc>(*this);
      ar & collapsed_params_;
      ar & num_strata_1_;
      ar & partial_sums_;
      ar & num_strata_2_;
    }
  public:
    mfgdmc() {
    }
    mfgdmc(int domain_size, const std::vector<double>& values,
	   int num_strata_1, std::vector<int> num_strata_2) : 
      gdmc(domain_size, values, num_strata_1), num_strata_1_(num_strata_1),
      num_strata_2_(num_strata_2)  {
      int ps = 0;
      for(int i = 0; i < num_strata_1_; i++) {
	partial_sums_.push_back(ps);
	ps += num_strata_2_[i];
	collapsed_params_.push_back(std::vector< std::vector<double> >(num_strata_2_[i]));
	for(int j = 0; j < num_strata_2_[i]; j++) {
	  collapsed_params_[i][j] = std::vector<double>(domain_size, 0.0);
	}
      }
    }

    /**
     * This implementation of Wallach's "Method 1" uses an iterative
     * fixed-point algorithm to reestimate the hyperparameters.
     */
    void reestimate_hyperparameters_wallach1(const std::vector<std::vector<std::vector<int> > >& counts,
					     int num_iterations = 200, double floor = 1E-9);


    virtual inline int num_entries() {
      int s = 0;
      for(int i : num_strata_2_) {
	s+=i;
      }
      return s;
    }
    virtual inline int strata_mapping(int row, int col) {
      return partial_sums_[row] + col;
    }

    inline std::vector< std::vector< std::vector< double > > > collapsed_params() {
      return collapsed_params_;
    }

    template <typename C> inline void reestimate_collapsed_parameters(const std::vector< std::vector< std::vector<C> > >& counts) {
      const int domain_size = this->size();
      for(int strata = 0; strata < num_strata_1_; ++strata) {
	for(int strata2 = 0; strata2 < num_strata_2_[strata]; ++strata2) {
	  const std::vector<C> strata_counts = counts[strata][strata2];
	  double strata_sum = 0.0;
	  for(int dom_index = 0; dom_index < domain_size; ++dom_index) {
	    strata_sum += strata_counts[dom_index] + hyperparameters_[dom_index];
	  }
	  for(int dom_index = domain_size - 1; dom_index >= 0; --dom_index) {
	    collapsed_params_[strata][strata2][dom_index] = (strata_counts[dom_index] + hyperparameters_[dom_index]) / strata_sum;
	  }
	}
      }
    }
    const double log_joint(const std::vector< std::vector< std::vector<int> > >& counts);
  };
}

#endif
