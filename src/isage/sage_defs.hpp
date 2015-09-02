#ifndef ISAGE_WTM_SAGE_DEFS_H_
#define ISAGE_WTM_SAGE_DEFS_H_

#include "dmc.hpp"
#include "loglin.hpp"
#include "mathops.hpp"
#include "optimize.hpp"
#include "util.hpp"
#include "wtm.hpp"

#include <fstream>
#include <iostream>
#include <ostream>
#include "stdlib.h"
#include <time.h>

// for pair
#include "map"
#include <cmath>
#include <limits>
#include <utility>
#include <unordered_set>
#include <thread>
#include "omp.h"
#include "lock.hpp"
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>

#ifndef SAGE_VERSION
#define SAGE_VERSION 1
#endif

namespace isage {
  namespace wtm {
    enum SageTopicRegularization {
      L2 = 0,
      IMPROPER = 1
    };

    std::istream& operator>>(std::istream& in, isage::wtm::SageTopicRegularization &how);
    double compute_sage_regularizer(double value, SageTopicRegularization how);
    double compute_grad_sage_regularizer(double value, SageTopicRegularization how);

    template <typename MaxentSupport, typename MaxentWeights>
    struct SageCClosure {
    public:
      typedef loglin::UnigramMaxent<MaxentSupport, MaxentWeights> MaxentModel;
      std::vector<double>* topic_counts;
      std::vector<double>* background;
      double marginal_count;
      MaxentModel* maxent;
      double regularizer_multiplier;
      SageTopicRegularization regularizer_type = SageTopicRegularization::L2;
    };

    struct SageCClosureIntVector : public SageCClosure<int, std::vector<double> > {
    };
    
    template <typename EtaType>
    class SageTopic {
    public:
      typedef std::vector<double> TauType;
      typedef int MaxentSupportType;
      typedef loglin::UnigramMaxent<MaxentSupportType, EtaType> MaxentModel;
      typedef SageCClosureIntVector ClosureType;
    protected:
      int support_size_;
      double tau_hyper_;
      std::vector<double>* background_;
      double log_partition_;
      //TauType tau_;
      EtaType* eta_;
      std::vector<double>* summed_weights_;
      MaxentModel maxent_;
      isage::wtm::SageTopicRegularization regularization_type_;
    private:
      friend class boost::serialization::access;
      // When the class Archive corresponds to an output archive, the
      // & operator is defined similar to <<.  Likewise, when the class Archive
      // is a type of input archive the & operator is defined similar to >>.
      template<class Archive>
      void serialize(Archive& ar, const unsigned int version) {
        ar & support_size_;
        ar & background_;
        ar & log_partition_;
        ar & eta_;
	ar & regularization_type_;
      } 
    public:
      typedef EtaType Eta;
      SageTopic<EtaType>() : background_(NULL), eta_(new EtaType()), 
			     summed_weights_(new std::vector<double>()),
			     regularization_type_(isage::wtm::SageTopicRegularization::L2) {
      }
      SageTopic<EtaType>(int support_size, double kappa, 
			 isage::wtm::SageTopicRegularization reg_type) : support_size_(support_size), 
        tau_hyper_(kappa), background_(NULL),
	regularization_type_(reg_type) { //, /*tau_(support_size), */ eta_(new EtaType(support_size,0.0) ) {
        eta_ = new EtaType(support_size_, 0.0);
        summed_weights_ = new std::vector<double>(support_size_, 0.0);
        maxent_.weights(eta_);
      }
      ~SageTopic() {
	delete eta_;
        delete summed_weights_;
      }
      double& operator[](const size_t& idx) {
	return eta_->operator[](idx);
      }
      void prepare() {
	if(summed_weights_->size() == 0) {
	  summed_weights_->resize(support_size_, 0.0);
	}
      }
      void push_to_maxent() {
	this->eta(*eta_);
      }
      MaxentModel* maxent_model() {
        return &maxent_;
      }
      std::vector<double>* background() {
        return background_;
      }
      void background(std::vector<double>* background) {
	background_ = background;
      }
      double log_partition() {
	return log_partition_;
      }
      void renormalize(const std::vector<double>& logterm_sums) {
	log_partition_ = mathops::log_sum_exp(logterm_sums);
      }
      void renormalize() {
	std::vector<double> logterm_sums = isage::util::sum(*background_, *eta_);
        renormalize(logterm_sums);
      }
      void eta(const EtaType& eta) {
        eta_->assign(eta.begin(), eta.end());
        const size_t size = eta_->size();
        for(size_t i = 0; i < size; ++i) {
          summed_weights_->operator[](i) = 
            background_->operator[](i) + eta_->operator[](i);
        }
        maxent_.weights(summed_weights_);
        renormalize();
      }
      EtaType eta() {
        return *eta_;
      }
      double l_probability(const MaxentSupportType& obj) {
        return maxent_.lp(obj);
      }
      double probability(const MaxentSupportType& obj) {
        return maxent_.p(obj);
      }

      template <typename OutputType> OutputType as() {
	std::vector<double> logterm_sums = isage::util::sum(*background_, *eta_);
        renormalize(logterm_sums);
        isage::util::sum(-1.0 * log_partition_, &logterm_sums);
        isage::util::exp(&logterm_sums);
        OutputType out(logterm_sums);
        return out;
      }

      template <typename OutputType> OutputType eta_as(bool normalize = true) {
        if(normalize) {
          double lp = mathops::log_sum_exp(*eta_);
          OutputType res = isage::util::sum(-1.0 * lp, *eta_);
          isage::util::exp(&res);
          return res;
        } else {
          OutputType res(*eta_);
          return res;
        }
      }

      static double elbo_contribution(const gsl_vector* trial_weights, void *fparams) __attribute__((deprecated)) {
        ClosureType* closure = (ClosureType*)fparams;
        // update the maxent model
        // add the trial_weights to the background
        std::vector<double> nweights = 
          optimize::GSLVector::sum(trial_weights, closure->background);
        // reset the maxent weights; this will renormalize everything
        closure->maxent->weights(nweights);
        double ll = closure->maxent->ll_dense_data(*(closure->topic_counts));
        // now compute the regularizer
        typedef std::vector<double> WeightType;
        const double reg_multiplier = closure->regularizer_multiplier;
        //double regularizer = reg_multiplier * isage::util::sum(isage::util::quartic(optimize::GSLVector::to_container<WeightType>(trial_weights)));
        double regularizer = reg_multiplier * isage::util::sum(isage::util::square(optimize::GSLVector::to_container<WeightType>(trial_weights)));
        ll += (-.5 * regularizer);
        // make sure to negate this
        return -ll;
      }
      static void elbo_gradient(const gsl_vector* trial_weights, void *fparams, gsl_vector* gsl_grad) __attribute__((deprecated)) {
        ClosureType* closure = (ClosureType*)fparams;
        // update the maxent model
        // add the trial_weights to the background
        // nweights = trial_weights + background
        std::vector<double> nweights = 
          optimize::GSLVector::sum(trial_weights, closure->background);
        // reset the maxent weights; this will renormalize everything
        closure->maxent->weights(nweights);
        //typedef (*(closure->maxent))::WeightType WeightType;
        typedef std::vector<double> WeightType;
        WeightType grad = closure->maxent->ll_grad_dense_data(*(closure->topic_counts));
        // subtract off regularization, which in this case happens to be trial_weights^3 (according to the SAGE paper, at least...)
        WeightType regularizer = optimize::GSLVector::to_container<WeightType>(trial_weights);
        //isage::util::cube(&regularizer);
        //const double reg_multiplier = closure->regularizer_multiplier * 2;
        const double reg_multiplier = closure->regularizer_multiplier ;
        isage::util::scalar_product(-1.0 * reg_multiplier, &regularizer);
        isage::util::sum_in_first(&grad, regularizer);
        // negate
        isage::util::scalar_product(-1.0, &grad);
        // and then convert grad to gsl_grad
        gsl_vector_memcpy(gsl_grad, optimize::GSLVector(grad).get());
      }
      static void elbo_contrib_grad(const gsl_vector* trial_weights, void *fparams,
                                    double* f, gsl_vector* grad) __attribute__((deprecated)) {
        *f = SageTopic<EtaType>::elbo_contribution(trial_weights, fparams);
        SageTopic<EtaType>::elbo_gradient(trial_weights, fparams, grad);
      }
      gsl_multimin_function_fdf get_fdf(ClosureType* params) __attribute__((deprecated)) {
        gsl_multimin_function_fdf objective;
        objective.n   = (size_t)support_size_;
        objective.f   = &SageTopic<EtaType>::elbo_contribution;
        objective.df  = &SageTopic<EtaType>::elbo_gradient;
        objective.fdf = &SageTopic<EtaType>::elbo_contrib_grad;
        objective.params = (void*)params;
        return objective;
      }

      static double liblbfgs_elbo(void *fparams, const lbfgsfloatval_t *trial_weights,
				  lbfgsfloatval_t *lbfgs_grad, const int n, const lbfgsfloatval_t step) {
        ClosureType* closure = (ClosureType*)fparams;
        // update the maxent model
        // add the trial_weights to the background
        std::vector<double> nweights = 
          optimize::LibLBFGSVector::sum(trial_weights, closure->background, n);
        // reset the maxent weights; this will renormalize everything
        closure->maxent->weights(nweights);
        typedef std::vector<double> WeightType;
	double ll = 0.0;
	{
	  ll = closure->maxent->ll_dense_data(*(closure->topic_counts));
	  // now compute the regularizer
	  const double reg_multiplier = closure->regularizer_multiplier;
	  double weight_sum = 0.0;
	  for(int i = 0; i < n; ++i) {
	    const auto val = trial_weights[i];
	    auto rval = compute_sage_regularizer(val, closure->regularizer_type);
	    weight_sum += rval;
	  }
	  double regularizer_s = reg_multiplier * weight_sum;
	  ll += (-.5 * regularizer_s);
	}

	// now compute the gradient
	{
	  WeightType grad = closure->maxent->ll_grad_dense_data(*(closure->topic_counts));
	  // subtract off regularization
	  const double reg_multiplier = closure->regularizer_multiplier * 0.5;
	  for(int i = 0; i < n; ++i) {
	    const lbfgsfloatval_t val = trial_weights[i];
	    const auto gval = compute_grad_sage_regularizer(val, closure->regularizer_type);
	    lbfgs_grad[i] = -grad[i] + reg_multiplier * gval;
	  }
	}
	return -ll;
      }
      optimize::LibLBFGSFunction get_liblbfgs_func(ClosureType* params) {
	optimize::LibLBFGSFunction objective;
        objective.eval   = &SageTopic<EtaType>::liblbfgs_elbo;
	objective.progress = &optimize::LibLBFGSNoOp::progress;
        objective.params = (void*)params;
        return objective;
      }

      int fit_topic(const std::vector<double>* counts, double reg_mult) {
        ClosureType params;
        params.topic_counts = const_cast< std::vector<double>* >(counts);
        params.background = this->background();
        params.regularizer_multiplier = reg_mult;
        params.maxent = this->maxent_model();
	params.regularizer_type = regularization_type_;
	optimize::LibLBFGSFunction my_func = this->get_liblbfgs_func(&params);
	Eta point = this->get_optimization_initial_point();
	optimize::LibLBFGSMinimizer optimizer(point.size());
        int opt_status = optimizer.minimize(&my_func, point);
	this->eta(point);
	return opt_status;
      }

      // This gets an initial point.
      // Following the original SAGE implementation, this initializes to the zero vector
      // (which makes sense, because in expectation, due to the sparsity-inducing prior, 
      // it should be zero).
      Eta get_optimization_initial_point() {
        Eta foo(support_size_, 0.0);
        return foo;
      }
      int support_size() {
	return support_size_;
      }
    };

    class DenseSageTopic : public SageTopic<std::vector<double> > {
    private:
      friend class boost::serialization::access;
      // When the class Archive corresponds to an output archive, the
      // & operator is defined similar to <<.  Likewise, when the class Archive
      // is a type of input archive the & operator is defined similar to >>.
      template<class Archive>
      void serialize(Archive& ar, const unsigned int version) {
        // serialize base class information
        ar & boost::serialization::base_object<SageTopic<std::vector<double> > >(*this);
      } 
    public:
      DenseSageTopic(int support_size, double kappa,
		     isage::wtm::SageTopicRegularization reg_type) : SageTopic<std::vector<double> >(support_size, kappa, reg_type) {
      }
    };

    enum TopicInitializerChoice {
      UNIFORM = 0,
      SUBSET = 1,
      BACKGROUND = 2
    };

    std::istream& operator>>(std::istream& in, isage::wtm::TopicInitializerChoice &how);

    struct SageInitializer {
    private:
      double inv_nt_;
      int num_words_;
      int nt_;
      int num_threads_;
      const TopicInitializerChoice fit_topic_how_;
      SageTopicRegularization reg_topic_how_;

      /**
       * Provide parameters eta over V elements for a SageTopic maxent model. 
       * This effectively returns "-background + signed random noise", 
       * in order to produce a SAGE topic 
       *   p(w | eta) \propto exp(background + eta)
       * that is near uniform.
       */
      std::vector<double> topic_uniform(const std::vector<double>& hyper,
					std::vector<double>* background,
					double num_types) {
	const double iv = 1.0/num_types;
        std::vector<double> vec( *background );
	isage::util::scalar_product(-1.0, &vec);
        mathops::add_uniform_noise(&vec, -iv, iv);
	return vec;
      }

      /**
       * Provide parameters eta over V elements for a SageTopic maxent model. 
       * This produces signed random noise around 0,
       * in order to produce a SAGE topic 
       *   p(w | eta) \propto exp(background + eta)
       * that is close to the background distribution. 
       * Effectively, this results in a maximum-likelihood topic 
       * **distribution**.
       */
      std::vector<double> topic_background(const std::vector<double>& hyper,
					   std::vector<double>* background,
					   double num_types) {
	const double iv = 1.0/num_types;
        std::vector<double> vec( (size_t)num_types, 0.0 );
        mathops::add_uniform_noise(&vec, -iv, iv);
	return vec;
      }

      /**
       * Provide parameters eta over V elements for a SageTopic maxent model. 
       * This randomly selects d' documents in the corpus (or all D if 
       * d' > D), where 
       *   d' = 10000/(average number of tokens per doc).
       * This fits a SAGE topic 
       *   p(w | eta) \propto exp(background + eta)
       * where the background parameters are fixed and shared across
       * all topics.
       */
      template <typename Corpus, typename Vocab>
      std::vector<double> topic_fit_subset(const std::vector<double>& hyper,
					   const Corpus* corpus,
					   const Vocab* vocab,
					   std::vector<double>* background) {
	const double avg_words_per_doc = (double)num_words_ / (double)corpus->num_docs();
        int init_num = (int)(10000.0 / avg_words_per_doc);
        if(init_num > corpus->num_docs()) {
          init_num = corpus->num_docs();
        }
        // pick init_num docs
        std::vector<int> chosen(init_num);
        {
          std::vector<int> all(corpus->num_docs());
          for(size_t i = 0; i < all.size(); ++i) {
            all[i] = i;
          }
	  gsl_rng *rg = gsl_rng_alloc(gsl_rng_mt19937);
          mathops::random_choose(chosen.data(), init_num,
                                 all.data(), corpus->num_docs(),
                                 sizeof(int), rg);
	  gsl_rng_free(rg);
        }
        std::vector<double> dense_counts(hyper.size(), 0.0);
        for(const auto& di : chosen) {
          for(const auto& pair : corpus->operator[](di).multinomial()) {
            dense_counts[vocab->index(pair.first)] += pair.second;
          }
        }
	typedef SageTopic<std::vector<double> > TopicType;
	TopicType topic(hyper.size(), 1.0, reg_topic_how_);
	topic.background(background);
	int opt_status = topic.fit_topic(&dense_counts, 1.0);
        INFO << "Maxent optimization in initialization of topic resulted in " << opt_status << " status";
	std::vector<double> point = topic.eta_as<std::vector<double> >(false);
        return point;
      }

    public:
      SageInitializer(int num_topics) : 
	num_words_(-1), nt_(num_topics), num_threads_(1),
	fit_topic_how_(TopicInitializerChoice::SUBSET),
	reg_topic_how_(SageTopicRegularization::L2) {
	inv_nt_ = 1.0/(double)nt_;
      }
      SageInitializer(int num_topics, int num_words) : 
        num_words_(num_words), nt_(num_topics), num_threads_(1),
	fit_topic_how_(TopicInitializerChoice::SUBSET),
	reg_topic_how_(SageTopicRegularization::L2) {
	inv_nt_ = 1.0/(double)nt_;
      }
      SageInitializer(int num_topics, int num_words, int num_threads) : 
        num_words_(num_words), nt_(num_topics), num_threads_(num_threads),
	fit_topic_how_(TopicInitializerChoice::SUBSET),
	reg_topic_how_(SageTopicRegularization::L2)  {
	inv_nt_ = 1.0/(double)nt_;
      }
      SageInitializer(int num_topics, int num_words, int num_threads,
		      TopicInitializerChoice fit_how, 
		      SageTopicRegularization reg_topic_how) : 
        num_words_(num_words), nt_(num_topics), num_threads_(num_threads),
	fit_topic_how_(fit_how), reg_topic_how_(reg_topic_how)  {
	inv_nt_ = 1.0/(double)nt_;
      }
      
      /**
       * Return the maximum number of threads that *may* be
       * used during initialization. Not all initializations are
       * multithreaded, so the number of threads actually used 
       * could be significantly less than the value returned here.
       */
      int num_threads() {
	return num_threads_;
      }

      /**
       * Provide multinomial parameters over K elements, were each
       * p_k \propto 1/K + U(-1/K, 1/K)
       */
      std::vector<double> assignment() {
	std::vector<double> vec(nt_, inv_nt_);
	mathops::add_uniform_noise(&vec, -inv_nt_, inv_nt_);
	double norm = isage::util::sum(vec);
	isage::util::scalar_product(1.0/norm, &vec);
	return vec;
      }

      /**
       * Provide parameters eta over V elements for a SageTopic maxent model. 
       * This randomly selects d' documents in the corpus (or all D if 
       * d' > D), where 
       *   d' = 10000/(average number of tokens per doc).
       * This fits a SAGE topic 
       *   p(w | eta) \propto exp(background + eta)
       * where the background parameters are fixed and shared across
       * all topics.
       */
      template <typename Corpus, typename Vocab>
      std::vector<double> topic(const std::vector<double>& hyper,
                                const Corpus* corpus,
                                const Vocab* vocab,
				std::vector<double>* background) {
	switch(fit_topic_how_) {
	case TopicInitializerChoice::UNIFORM:
	  return topic_uniform(hyper, background, (double)(vocab->num_words()));
	case TopicInitializerChoice::SUBSET:
	  return topic_fit_subset(hyper, corpus, vocab, background);
	case TopicInitializerChoice::BACKGROUND:
	  return topic_background(hyper, background, (double)(vocab->num_words()));
	default:
	  ERROR << "Invalid topic fit selection \"" << fit_topic_how_ << "\"";
	  throw 4;
	}
      }

      /**
       * Provide Dirichlet parameters gamma over K elements.
       * Given N words in a **document** and initial parameters alpha, 
       * each gamma_k is
       *   gamma_k = alpha_k + M/K + U(-M/K, M/K)
       * This is a better option for large corpora, vs
       * usage(const std::vector<double>&).
       */
      std::vector<double> usage(const std::vector<double>& hyper, const double N) {
	const double d = (double)N/(double)hyper.size();
        std::vector<double> vec(nt_, d);
        isage::util::sum_in_first(&vec, hyper);
        mathops::add_uniform_noise(&vec, -d, d);
	return vec;
      }

      /**
       * Provide Dirichlet parameters gamma over K elements.
       * Given M words in a corpus, and initial parameters alpha, 
       * each gamma_k is
       *   gamma_k = alpha_k + M/K + U(-M/K, M/K)
       * For large corpora, this results in parameters that yield
       * very flat Dirichlet draws. In such cases, it is better to use
       * usage(const std::vector<double>&, const int).
       */
      std::vector<double> usage(const std::vector<double>& hyper) {
	if(num_words_ <= 0) {
	  ERROR << "Attempting to call .usage with num_words in corpus == " << num_words_;
	  throw 3;
	}
	return usage(hyper, (double)num_words_);
      }
    };

    struct SageStrategy {
      SageStrategy() {
      }
      double em_frobenius_threshold = 1E-6;
      double eta_density_threshold = 1E-4;
      int num_learn_iters = 100;
      int num_e_iters = 25;
      int num_m_iters = 1;
      int num_e_threads = 1;
      int num_m_threads = 1;
      int hyper_update_min = 20;
      int hyper_update_iter = 5;
      int update_model_every = 5;
      int partial_restarts = 0;
      int num_learn_restart_iters = 25;
      int num_e_restart_iters = 25;
      int print_topics_every = 5;
      int print_topics_k = 10;
      int print_usage_every = 5;
      int em_verbosity = 1;
      bool heldout = false;
    };
  }
}

#endif
