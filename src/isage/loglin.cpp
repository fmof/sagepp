#include "loglin.hpp"
#include "logging.hpp"
#include "mathops.hpp"
#include <vector>

namespace loglin {
  template <typename SupportType, typename WeightType>
  UnigramMaxent<SupportType, WeightType>::UnigramMaxent(bool renormalize_when_set) : stale_log_normalizer_(true), renormalize_when_set_(renormalize_when_set), weights_(NULL), log_normalizer_(0.0) {
  }
  template <typename SupportType, typename WeightType>
  UnigramMaxent<SupportType, WeightType>::UnigramMaxent() : stale_log_normalizer_(true), renormalize_when_set_(true), weights_(NULL), log_normalizer_(0.0) {
  }

  template <typename SupportType, typename WeightType>
  void UnigramMaxent<SupportType, WeightType>::weights(WeightType* n_weights) {
    weights_ = n_weights;
    stale_log_normalizer_ = true;
    if(renormalize_when_set_) this->renormalize();
  }
  template <typename SupportType, typename WeightType>
  void UnigramMaxent<SupportType, WeightType>::weights(WeightType& n_weights) {
    weights_ = &n_weights;
    stale_log_normalizer_ = true;
    if(renormalize_when_set_) this->renormalize();
  }

  template <typename SupportType, typename WeightType>
  void UnigramMaxent<SupportType, WeightType>::renormalize() {
    log_normalizer_ = mathops::log_sum_exp(*weights_);
    stale_log_normalizer_ = false;
  }

  template <typename SupportType, typename WeightType>
  double UnigramMaxent<SupportType, WeightType>::log_normalizer() {
    if(stale_log_normalizer_) {
      renormalize();
    }
    return log_normalizer_;
  }

  template <typename SupportType, typename WeightType> 
  double UnigramMaxent<SupportType, WeightType>::lp(SupportType obj) {
    return weights_->operator[](obj) - log_normalizer();
  }
  template <typename SupportType, typename WeightType> 
  double UnigramMaxent<SupportType, WeightType>::p(SupportType obj) {
    return mathops::exp(lp(obj));
  }


  template <typename SupportType, typename WeightType> 
  template <typename SparseDataSet>
  double UnigramMaxent<SupportType, WeightType>::ll(const SparseDataSet& data) {
    renormalize();
    double ll = 0.0;
    for(const auto& obj : data) {
      ll += obj.second * lp(obj.first);
    }
    return ll;
  }

  template <typename SupportType, typename WeightType> 
  template <typename SparseDataSet>
  WeightType UnigramMaxent<SupportType, WeightType>::ll_grad(const SparseDataSet& data) {
    renormalize();
    WeightType grad;
    // note that we reference grad.end() in order to always have amortized O(1) insertion
    typename WeightType::iterator g_it = grad.end();
    double sum = 0.0;
    for(const auto& obj : data) {
      sum += obj.second;
    }
    for(const auto& obj : data) {
      double val = obj.second - (sum * p(obj.first));
      // these two separate statements are needed because initially, g_it == 0x0 (due to empty container)
      g_it = grad.emplace(g_it, val);
      // and update to point to the end (for O(1) insertion)
      ++g_it;
    }
    return grad;
  }

  template <typename SupportType, typename WeightType> 
  double UnigramMaxent<SupportType, WeightType>::ll_dense_data(const std::vector<double>& data) {
    renormalize();
    double ll = 0.0;
    const size_t data_size = data.size();
    for(size_t i = 0; i < data_size; ++i) {
      //BOOST_LOG_TRIVIAL(debug) << "obj " << i << ", data[i] = " << data[i] << ", lp(i) = " << lp(i);
      ll += data[i] * lp(i);
    }
    return ll;
  }

  template <typename SupportType, typename WeightType> 
  WeightType UnigramMaxent<SupportType, WeightType>::ll_grad_dense_data(const std::vector<double>& data) {
    renormalize();
    WeightType grad;
    // note that we reference grad.end() in order to always have amortized O(1) insertion
    typename WeightType::iterator g_it = grad.end();
    const size_t data_size = data.size();
    const double N = isage::util::sum(data);
    DEBUG << "Total number of instances = " << N;
    for(size_t i = 0; i < data_size; ++i) {
      double val = data[i] - N*p(i);
      // these two separate statements are needed because initially, g_it == 0x0 (due to empty container)
      g_it = grad.emplace(g_it, val);
      // and update to point to the end (for O(1) insertion)
      ++g_it;
    }
    return grad;
  }

  IntUnigramMaxentL2::IntUnigramMaxentL2() {
    model_ = new loglin::UnigramMaxent<int, std::vector<double> >();
  }
  IntUnigramMaxentL2::~IntUnigramMaxentL2() {
    delete model_;
  }

  double IntUnigramMaxentL2::int_unigram_ll(const gsl_vector* trial_weights, void *fparams) {
    IntUnigramMaxentClosure* closure = (IntUnigramMaxentClosure*)fparams;
    std::vector<double> nweights = 
      optimize::GSLVector::to_container<std::vector<double> >(trial_weights);
    closure->model->weights(nweights);
    double ll = closure->model->ll_dense_data(*(closure->counts));
    double reg = isage::util::sum(isage::util::square(nweights));
    reg *= closure->regularizer_strength / 2.0;
    return -ll + reg;
  }
  void IntUnigramMaxentL2::int_unigram_gradient(const gsl_vector* trial_weights, 
						void *fparams, gsl_vector* gsl_grad) {
    IntUnigramMaxentClosure* closure = (IntUnigramMaxentClosure*)fparams;
    std::vector<double> nweights = 
      optimize::GSLVector::to_container<std::vector<double> >(trial_weights);
    // reset the maxent weights; this will renormalize everything
    closure->model->weights(nweights);
    typedef std::vector<double> WeightType;
    WeightType grad = closure->model->ll_grad_dense_data(*(closure->counts));
    isage::util::linear_combination_in_first(&grad, nweights, -1.0, closure->regularizer_strength);
    // and then convert grad to gsl_grad
    gsl_vector_memcpy(gsl_grad, optimize::GSLVector(grad).get());
  }
  void IntUnigramMaxentL2::int_unigram_ll_grad(const gsl_vector* trial_weights, void *fparams,
					       double* f, gsl_vector* grad) {
    *f = int_unigram_ll(trial_weights, fparams);
    int_unigram_gradient(trial_weights, fparams, grad);
  }
  
  gsl_multimin_function_fdf IntUnigramMaxentL2::get_fdf(IntUnigramMaxentClosure* params, const size_t size) {
    gsl_multimin_function_fdf objective;
    objective.n   = size;
    objective.f   = &int_unigram_ll;
    objective.df  = &int_unigram_gradient;
    objective.fdf = &int_unigram_ll_grad;
    params->model = model_;
    objective.params = (void*)params;
    return objective;
  }


  template class UnigramMaxent<int, std::vector<double> >;
  template class UnigramMaxent<size_t, std::vector<double> >;

  template double UnigramMaxent<size_t, std::vector<double> >::ll(const std::map<size_t, int>&);
  template double UnigramMaxent<int, std::vector<double> >::ll(const std::map<int, int>&);
  template std::vector<double> UnigramMaxent<size_t, std::vector<double> >::ll_grad(const std::map<size_t, int>&);
  template std::vector<double> UnigramMaxent<int, std::vector<double> >::ll_grad(const std::map<int, int>&);

  // double ll_type_view(const std::vector<double>& point, 
  // 		      const std::vector<double>& feature_counts,
  // 		      const int num_instances, const double log_partition) {
  //   if(point.size() != feature_counts.size()) {
  //     BOOST_LOG_TRIVIAL(error) << "The parameter size (" << point.size() << " must be the same as the feature counts (" << feature_counts.size() << ")";
  //     throw 2;
  //   }
  //   const size_t size = point.size();
  //   double res = 0.0;
  //   // now take the dot product
  //   for(size_t i = 0; i < size; ++i) {
  //     res += point[i] * feature_counts[i];
  //   }
  //   // get the normalizer
  //   res -= (num_instances * log_partition);
  //   return res;
  // }

  // void ll_grad_type_view(const std::vector<double>& feature_counts,
  // 			 const std::vector<double>& occurrence_counts,
  // 			 const std::vector<double>& prob,
  // 			 std::vector<double>* grad) {
  //   if(point.size() != feature_counts.size()) {
  //     BOOST_LOG_TRIVIAL(error) << "The parameter size (" << point.size() << " must be the same as the feature counts (" << feature_counts.size() << ")";
  //     throw 2;
  //   }
  //   if(point.size() != grad->size()) {
  //     BOOST_LOG_TRIVIAL(error) << "The parameter size (" << point.size() << " must be the same as the gradient receptacle size (" << grad->size() << ")";
  //     throw 2;
  //   }
  //   const size_t size = point.size();
  //   for(size_t i = 0; i < size; ++i) {
  //     double f_i = feature_counts.at(i);
  //     double expected = occurrence_counts.at(i) * prob.at(i);
  //     grad->operator[](i) = f_i - f_i*expected;
  //   }
  // }
}
