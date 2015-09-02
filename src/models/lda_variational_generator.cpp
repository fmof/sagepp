#include "logging.hpp"
#include "sage.hpp"
#include "wtm.hpp"
#include "wtm_variational.hpp"

#include <gsl/gsl_rng.h>
#include <boost/program_options.hpp>

// for serialization
#include <fstream>
#include <iostream>
#include <ostream>
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

namespace po = boost::program_options;

void init_logging() {
#ifndef ISAGE_LOG_AS_COUT
boost::log::core::get()->set_filter
(
boost::log::trivial::severity >= boost::log::trivial::info
 );
#endif
}

template <typename Corpus>
int get_num_tokens(const Corpus& corpus) {
  int res = 0;
  for(const auto& doc : corpus) {
    res += doc.num_words();
  }
  return res;
}

isage::wtm::SymmetricHyperparams get_shp(double base) {
  isage::wtm::SymmetricHyperparams shp = isage::wtm::SymmetricHyperparams();
  shp.h_word = base;
  shp.h_theta = base;
  return shp;
}

int main(int n_args, char** args) {
  init_logging();
  isage::util::print_pstats();

  int num_topics;
  int num_epochs_;

  isage::wtm::LDAVStrategy strategy;
  
  std::string output_usage_name;
  std::string heldout_output_usage_name;

  po::variables_map vm;
  {
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help", "produce help message")
      ("vocab-size", po::value< int >()->default_value(10),
       "number of vocab words (default: 10)")
      ("words-per-doc", po::value<int>()->default_value(10),
       "number of words per document (default: 10)")
      ("bias", po::value<double>()->default_value(.8),
       "Bernoulli parameter p for how to partition the vocab words. (default: 0.8)")
      ("num-docs", po::value<int>()->default_value(10),
       "number of documents to generate (default: 10)")
      //////////////////////////
      ("topics", po::value<int>(&num_topics)->default_value(10), 
       "number of topics to use")
      ("train-epochs", po::value<int>(&num_epochs_)->default_value(5),
       "Number of epochs to run")
      ("em-iterations", po::value<int>(&(strategy.num_learn_iters))->default_value(100), 
       "number of EM iterations to run")
      ("e-steps", po::value<int>(&(strategy.num_e_iters))->default_value(25), 
       "number of iterations to perform, per E-step")
      ("m-steps", po::value<int>(&(strategy.num_m_iters))->default_value(1), 
       "number of iterations to perform, per M-step")
      ("update-hypers", po::value<int>(&(strategy.hyper_update_iter))->default_value(-1),
       "how often to update the hyperparameters (default: -1 == never update)")
      ("update-model-interval", po::value<int>(&(strategy.update_model_every))->default_value(5), "update the model every [some] number of EM steps (default: 5)")
      ("print-topics-every", po::value<int>(&(strategy.print_topics_every))->default_value(5), "print topics every [some] number of EM steps (default: 5)")
      ("print-usage-every", po::value<int>(&(strategy.print_usage_every))->default_value(5), "print topic usage every [some] number of EM steps (default: 5)")
      ("top-k", po::value<int>(&(strategy.print_topics_k))->default_value(10), "number of words per topic to print (default: 10)")
      ("em-verbosity", po::value<int>(&(strategy.em_verbosity))->default_value(1),
       "how verbose should EM output be (default: 1; higher == more verbose)")
      ("eta-density-threshold", po::value<double>(&(strategy.eta_density_threshold))->default_value(1E-4),
       "the threshold t for counting the number of eta parameters are above t (default: 1E-4)")
      ////////////////////////////////
      ("topic-usage-file", po::value<std::string>(&output_usage_name)->default_value("-"), 
       "filename to write topic usage to (default: - (to console)")
      ("heldout-topic-usage-file", po::value<std::string>(&heldout_output_usage_name)->default_value("-"), 
       "filename to write heldout topic usage to (default: - (to console)")
      ("inferencer-serialization", po::value<std::string>(), "filename to serialize inference state to")
      ("serialized-inferencer", po::value<std::string>(), "filename to READ serialized inference state from")
      ////////////////////////////////
      ;

    po::store(po::parse_command_line(n_args, args, desc), vm);
    if (vm.count("help")) {
      ERROR << desc << "\n";
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
  typedef std::vector<double> TopicType;
  typedef isage::wtm::DiscreteLDA< VocabType, std::vector<double> > Model;
  typedef isage::wtm::DiscreteVariational< Doc, VocabType, TopicType > Variational;

  isage::util::SmartWriter usage_outer(output_usage_name);
  isage::util::SmartWriter assign_outer("assignments");
  
  Variational* var_inf = NULL;
  SVocab word_vocab("__OOV__");
  for(int wi = 1; wi <= vm["vocab-size"].as<int>(); ++wi) {
    word_vocab.make_word("word_" + std::to_string(wi));
  }

  Corpus corpus("train_corpus");
  corpus.generate(vm["num-docs"].as<int>(),
		  vm["words-per-doc"].as<int>(),
		  vm["bias"].as<double>(),
		  word_vocab
		  );
  int num_words_total = get_num_tokens(corpus);
  INFO << "Number of documents: " << corpus.num_docs();
  INFO << "Number of word tokens total: " << num_words_total;
  INFO << "Number of vocab types: " << word_vocab.num_words();

  isage::wtm::SymmetricHyperparams shp;
  shp.h_theta = 1.0/(double)num_topics;
  shp.h_word =  0.1; 
  INFO << "Creating model with " << num_topics << " topics";
  Model dm(num_topics, &shp, &word_vocab);
  INFO << "Done creating model.";
  var_inf = new Variational(&dm, &corpus, &word_vocab);
  isage::wtm::UniformHyperSeedWeightedInitializer initer(num_topics, corpus.num_docs(), (double)num_words_total/(double)corpus.num_docs());
  var_inf->init(initer);

  for(int epoch = 0; epoch < num_epochs_; ++epoch) {
    INFO << "Starting learning epoch " << epoch;
    var_inf->learn(strategy, epoch, usage_outer, assign_outer);
    INFO << "Done with inference in epoch " << epoch;
    // // create and open a character archive for output
    // if(vm.count("inferencer-serialization")) {
    //   std::string sfname = vm["inferencer-serialization"].as<std::string>() + 
    // 	".iteration" + std::to_string((1+epoch));	
    //   std::ofstream ofs(sfname, std::ios::out|std::ios::binary);
    //   boost::iostreams::filtering_streambuf<boost::iostreams::output> out;
    //   out.push(boost::iostreams::gzip_compressor());
    //   out.push(ofs);
    //   boost::archive::binary_oarchive oa(out);
    //   oa << (*var_inf);
    //   INFO << "see " << sfname << " for serialized inferencer";
    // }
    dm.print_topics(strategy.print_topics_k, word_vocab);
  }

  if(var_inf != NULL) {
    delete var_inf;
  }
  return 0;
}

