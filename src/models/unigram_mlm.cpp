#include "logging.hpp"
#include "loglin.hpp"
#include "optimize.hpp"
#include "wtm.hpp"

#include <gsl/gsl_rng.h>
#include <boost/program_options.hpp>

// for serialization
#include <fstream>
#include <iostream>
#include <type_traits>
// include headers that implement a archive in simple text format
#include <boost/algorithm/string.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <fstream>


namespace po = boost::program_options;

void init_logging() {
boost::log::core::get()->set_filter
(
 boost::log::trivial::severity >= boost::log::trivial::info
 );
}

template <typename Corpus>
int get_num_tokens(const Corpus& corpus) {
  int res = 0;
  for(const auto& doc : corpus) {
    res += doc.num_words();
  }
  return res;
}

struct UnigramClosure {
  typedef loglin::UnigramMaxent<int, std::vector<double> > MaxentModel;
  MaxentModel* model;
  std::vector<double>* counts;
};
class L2RegularizedUnigramMaxentLM {
public:
  typedef UnigramClosure ClosureType;
  static double elbo_contribution(const gsl_vector* trial_weights, void *fparams) {
    ClosureType* closure = (ClosureType*)fparams;
    std::vector<double> nweights = 
      optimize::GSLVector::to_container<std::vector<double> >(trial_weights);
    // reset the maxent weights; this will renormalize everything
    closure->model->weights(nweights);
    double ll = closure->model->ll_dense_data(*(closure->counts));
    double reg = 0.5 * isage::util::sum(isage::util::square(nweights));
    BOOST_LOG_TRIVIAL(debug) << "reg is " << reg;
    return -ll + reg;
  }
  static void elbo_gradient(const gsl_vector* trial_weights, void *fparams, gsl_vector* gsl_grad) {
    ClosureType* closure = (ClosureType*)fparams;
    std::vector<double> nweights = 
      optimize::GSLVector::to_container<std::vector<double> >(trial_weights);
    // reset the maxent weights; this will renormalize everything
    closure->model->weights(nweights);
    typedef std::vector<double> WeightType;
    WeightType grad = closure->model->ll_grad_dense_data(*(closure->counts));
    // add in the regularized
    WeightType reg = isage::util::scalar_product(-1.0, nweights);
    isage::util::sum_in_first(&grad, reg);
    // negate
    isage::util::scalar_product(-1.0, &grad);
    // and then convert grad to gsl_grad
    gsl_vector_memcpy(gsl_grad, optimize::GSLVector(grad).get());
  }
  static void elbo_contrib_grad(const gsl_vector* trial_weights, void *fparams,
				double* f, gsl_vector* grad) {
    *f = L2RegularizedUnigramMaxentLM::elbo_contribution(trial_weights, fparams);
    L2RegularizedUnigramMaxentLM::elbo_gradient(trial_weights, fparams, grad);
  }
  gsl_multimin_function_fdf get_fdf(ClosureType* params, const size_t size) {
    gsl_multimin_function_fdf objective;
    objective.n   = size;
    objective.f   = &L2RegularizedUnigramMaxentLM::elbo_contribution;
    objective.df  = &L2RegularizedUnigramMaxentLM::elbo_gradient;
    objective.fdf = &L2RegularizedUnigramMaxentLM::elbo_contrib_grad;
    objective.params = (void*)params;
    return objective;
  }

  static double lbfgs_elbo(void *fparams, const lbfgsfloatval_t *trial_weights,
			   lbfgsfloatval_t *lbfgs_grad, const int n, const lbfgsfloatval_t step) {
    ClosureType* closure = (ClosureType*)fparams;
    std::vector<double> nweights = 
      optimize::LibLBFGSVector::to_container<std::vector<double> >(trial_weights, n);
    // reset the maxent weights; this will renormalize everything
    closure->model->weights(nweights);
    double ll = closure->model->ll_dense_data(*(closure->counts));
    typedef std::vector<double> WeightType;
    WeightType grad = closure->model->ll_grad_dense_data(*(closure->counts));
    // negate
    isage::util::scalar_product(-1.0, &grad);
    optimize::LibLBFGSVector::copy(grad, lbfgs_grad);
    return -ll;
  }
  optimize::LibLBFGSFunction get_liblbfgs_fdf(ClosureType* params, const size_t size) {
    optimize::LibLBFGSFunction objective;
    objective.eval   = &L2RegularizedUnigramMaxentLM::lbfgs_elbo;
    objective.progress = &optimize::LibLBFGSNoOp::progress;
    objective.params = (void*)params;
    return objective;
  }

  std::vector<double> get_optimization_initial_point(int size_) {
    std::vector<double> foo(size_, 0.0);
    return foo;
  }
};


int main(int n_args, char** args) {
  init_logging();

  po::variables_map vm;
  {
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help", "produce help message")
      ("vocab", po::value< std::string >(),
       "vocab filepath (one word type per line)")
      ("train", po::value< std::string >(), 
       "input training path")
      ;

    po::store(po::parse_command_line(n_args, args, desc), vm);
    if (vm.count("help")) {
      BOOST_LOG_TRIVIAL(error) << desc << "\n";
      return 1;
    }
    po::notify(vm);
  }

  typedef std::string string;
  typedef string VocabType;
  typedef isage::wtm::Vocabulary< VocabType > SVocab;
  typedef double CountType;
  typedef isage::wtm::Document< VocabType, CountType > Doc;
  typedef isage::wtm::Corpus< Doc > Corpus;

  if(vm.count("train")) {
    BOOST_LOG_TRIVIAL(info) << "Going to read from " << vm["train"].as<std::string>();
    SVocab vocab = SVocab::from_file(vm["vocab"].as<std::string>(),
				     "__OOV__");
    
    Corpus corpus("train_corpus", vm["train"].as<std::string>(), vocab);
    std::vector<double> counts(vocab.num_words());
    for(const auto& doc : corpus) {
      for(const auto& mult : doc.multinomial()) {
	counts[vocab.index(mult.first)] += mult.second;
      }
    }
    loglin::UnigramMaxent<int, std::vector<double> > model;
    UnigramClosure params;
    params.counts = &counts;
    params.model = &model;
    L2RegularizedUnigramMaxentLM lm;
    optimize::LibLBFGSFunction my_func = lm.get_liblbfgs_fdf(&params, vocab.num_words());
    //gsl_multimin_function_fdf my_func = lm.get_fdf(&params, vocab.num_words());
    std::vector<double> point = lm.get_optimization_initial_point(vocab.num_words());
    optimize::LibLBFGSMinimizer optimizer(point.size());
    //optimize::GSLMinimizer optimizer(point.size());
    //std::vector<int> iterate_stati;
    //std::vector<int> grad_stati;
    int opt_status = optimizer.minimize(&my_func, point);
    INFO << "Unigram Maxent LM has optimization status " << opt_status;
    //BOOST_LOG_TRIVIAL(info) << "Unigram Maxent LM has optimization status " << opt_status << " after " << iterate_stati.size() << " minimization attempts and " << grad_stati.size() << " gradient status checks";
  }
  return 0;
}
