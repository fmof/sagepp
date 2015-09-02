#ifndef ISAGE_LOGLIN_H_ 
#define ISAGE_LOGLIN_H_

#include <logging.hpp>
#include <optimize.hpp>
#include <util.hpp>
#include <vector>

namespace loglin {
  /**
   * Note that WeightType must be indexable by SupportType
   */
  template <typename SupportType, typename WeightType>
  class UnigramMaxent {
  private:
    bool stale_log_normalizer_;
    bool renormalize_when_set_;
  protected:
    WeightType* weights_;
    double log_normalizer_;
  public:
    UnigramMaxent();
    UnigramMaxent(bool renormalize_when_set);
    double lp(SupportType obj);
    double p(SupportType obj);
    double log_normalizer();

    template <typename SparseDataSet> double ll(const SparseDataSet& data);
    template <typename SparseDataSet> WeightType ll_grad(const SparseDataSet& data);
    double ll_dense_data(const std::vector<double>& data);
    WeightType ll_grad_dense_data(const std::vector<double>& data);
    void weights(WeightType& n_weights);
    void weights(WeightType* n_weights);
    void renormalize();
  };

  struct IntUnigramMaxentClosure {
    typedef loglin::UnigramMaxent<int, std::vector<double> > MaxentModel;
    MaxentModel* model;
    std::vector<double>* counts;
    double regularizer_strength = 0.0;
  };

  class IntUnigramMaxentL2 {
  private:
    loglin::UnigramMaxent<int, std::vector<double> >* model_;
  public:
    IntUnigramMaxentL2();
    ~IntUnigramMaxentL2();
    static double int_unigram_ll(const gsl_vector* trial_weights, void *fparams);
    static void int_unigram_gradient(const gsl_vector* trial_weights, 
				     void *fparams, gsl_vector* gsl_grad);
    static void int_unigram_ll_grad(const gsl_vector* trial_weights, void *fparams,
				    double* f, gsl_vector* grad);
    gsl_multimin_function_fdf get_fdf(IntUnigramMaxentClosure* params, const size_t size);
  };
}

#endif
