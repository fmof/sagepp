#include "concrete_util/io.h"
#include "logging.hpp"
#include "wtm.hpp"
#include "wtm_sampling.hpp"

#include <gsl/gsl_rng.h>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

// for serialization
#include <fstream>
#include <iostream>
// include headers that implement a archive in simple text format
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>

void init_logging() {
  boost::log::core::get()->set_filter
    (
     boost::log::trivial::severity >= boost::log::trivial::info
     );
}

isage::wtm::SymmetricHyperparams get_shp(double base) {
  isage::wtm::SymmetricHyperparams shp = isage::wtm::SymmetricHyperparams();
  shp.h_word = base;
  shp.h_theta = base;
  return shp;
}

template <typename Doc, typename VType>
isage::wtm::Corpus<Doc> get_corpus(const std::string& name, const std::string& fpath,
				   isage::wtm::Vocabulary<VType>& word_vocab, 
				   const isage::wtm::StopWordList *stops) { 
  isage::wtm::Corpus< Doc > corpus("my_corpus");
  int num_comms = 0;
  concrete::util::CommunicationSequence *concrete_reader;
  concrete::util::get_communication_sequence( fpath, concrete_reader );
  for(concrete_reader->begin(); concrete_reader->keep_reading(); 
      concrete_reader->operator++()) {
    concrete::Communication communication = *(*concrete_reader);
    isage::wtm::WordPruner< VType > wp(&word_vocab);
    if(stops != NULL) {
      wp.stopwords(*stops);
      wp.use_stopwords(true);
    }
    BOOST_LOG_TRIVIAL(trace) << communication.id;
    ++num_comms;
    Doc my_doc(communication, wp);
    corpus.add_document(my_doc);
  }
  delete concrete_reader;
  return corpus;
}


int main(int n_args, char** args) {
  init_logging();

  int num_topics;
  int num_iterations;
  int burnin;
  int num_heldout_iterations;
  int heldout_burnin;
  int num_epochs_;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("train", po::value< std::string >(), 
     "input training path")
    ("dev", po::value< std::string >(), "input development path")
    //////////////////////////
    ("topics", po::value<int>(&num_topics)->default_value(10), 
     "number of topics to use")
    ("train-epochs", po::value<int>(&num_epochs_)->default_value(5),
     "Number of epochs to run")
    ("iterations", po::value<int>(&num_iterations)->default_value(200), 
     "number of iterations, per epoch")
    ("burnin", po::value<int>(&burnin)->default_value(100), 
     "number of burnin iterations")
    ("heldout-iterations", po::value<int>(&num_heldout_iterations)->default_value(200), 
     "number of (heldout) iterations")
    ("heldout-burnin", po::value<int>(&heldout_burnin)->default_value(50), 
     "number of (heldout) burnin iterations")
    ("stoplist", po::value<std::string>(),
     "tab-separated file containing stopwords in the first column (all other columns ignored)")
    ("sampler-serialization", po::value<std::string>(), "filename to serialize sampler to")
    ("serialized-sampler", po::value<std::string>(), "filename to READ serialized sampler from")
    ////////////////////////////////
    ///////////////////////////////
    ("compute-dev-ll", po::bool_switch()->default_value(false),
     "compute log-likelihood on the heldout set")
    ("compute-dev-label-tsv", po::value<std::string>(),
     "label the heldout and print out the labeling as a TSV file")
    ("compute-coherence", po::value<std::vector<std::string> >()->multitoken(),
     "compute topic coherence on all the space-separated distributions. Known values are gov_obs, rel_obs, gov_kind, rel_kind, template, slot")
    ("coherence-file", po::value<std::string>(),
     "file to write out coherence results")
    ("coherence-M", po::value<int>(), "take top M for each distribution while computing coherences")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(n_args, args, desc), vm);
  po::notify(vm);    

  if (vm.count("help")) {
    BOOST_LOG_TRIVIAL(error) << desc << "\n";
    return 1;
  }

  typedef std::string string;
  typedef isage::wtm::Vocabulary<string> SVocab;
  typedef isage::wtm::Document< string > Doc;
  typedef isage::wtm::DiscreteLDA<string, std::vector<double> > Model;
  typedef isage::wtm::CollapsedGibbsDMC<Doc, string > Sampler;

  isage::wtm::StopWordList *word_stops = NULL;
  if(vm.count("stoplist")) {
    word_stops = new isage::wtm::StopWordList(vm["stoplist"].as<std::string>());
  }

  if(vm.count("train")) {
    BOOST_LOG_TRIVIAL(info) << "Going to read from " << vm["train"].as<std::string>();
    SVocab word_vocab("__WORD_OOV__");
    isage::wtm::Corpus<Doc> corpus = get_corpus<Doc, string>("train_corpus", vm["train"].as<std::string>(),
							     word_vocab, word_stops);

    isage::wtm::SymmetricHyperparams shp = get_shp(0.1);
    isage::wtm::SampleEveryIter sample_strat(num_iterations, burnin);

    Model dm(num_topics, &shp, &word_vocab);
    Sampler sampler(&dm, &corpus, &word_vocab);
    sampler.sampling_strategy(&sample_strat);
    sampler.init();

    isage::wtm::Corpus<Doc> heldout_corpus;
    if(vm.count("dev")) {
      word_vocab.allow_new_words(false);
      heldout_corpus = get_corpus<Doc, string>("dev_corpus", vm["dev"].as<std::string>(), 
						       word_vocab, word_stops);
    }
    for(int epoch = 0; epoch < num_epochs_; ++epoch) {
      sampler.learn(num_iterations*epoch);
      // sampler.transfer_learned_parameters();
      // dm.print_governors(10, gov_kind_vocab);
      BOOST_LOG_TRIVIAL(info) << "Done with sampling in epoch " << epoch;
      // create and open a character archive for output
      if(vm.count("sampler-serialization")) {
	BOOST_LOG_TRIVIAL(error) << "This is not yet done :(";
	// std::string sfname = vm["sampler-serialization"].as<std::string>() + ".iteration" + std::to_string((1+epoch)*num_iterations);	
	// std::ofstream ofs(sfname, std::ios::out|std::ios::binary);
	// boost::iostreams::filtering_streambuf<boost::iostreams::output> out;
	// out.push(boost::iostreams::gzip_compressor());
	// out.push(ofs);
	// boost::archive::binary_oarchive oa(out);
	// oa << sampler;
	// BOOST_LOG_TRIVIAL(info) << "see " << sfname << " for serialized sampler";
      }
      // and now run on held-out
      if(vm.count("dev")) {
	isage::wtm::SampleEveryIter sample_strat_heldout(num_heldout_iterations, heldout_burnin);
	sample_strat_heldout.heldout(true);
	Sampler heldout_sampler(&dm, &heldout_corpus, &word_vocab);
	heldout_sampler.sampling_strategy(&sample_strat_heldout);
	heldout_sampler.init();
	heldout_sampler.learn();
	BOOST_LOG_TRIVIAL(info) << "Done with sampling heldout";
	// BOOST_LOG_TRIVIAL(info) << "Training epoch " << epoch << ": Held-out marginalized ll: " << 
	//   dm.latent_and_kind_marginalized_loglikelihood(heldout_corpus, gov_vocab, rel_vocab,
	// 						heldout_sampler.doc_template_params());
      }
    }
  } // else if(vm.count("serialized-sampler")) {
  //   Sampler heldout_sampler;
  //   std::ifstream ifs(vm["serialized-sampler"].as<std::string>(), std::ios::in|std::ios::binary);
  //   assert(ifs.good());
  //   boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
  //   in.push(boost::iostreams::gzip_decompressor());
  //   in.push(ifs);
  //   boost::archive::binary_iarchive ia(in);
  //   ia >> heldout_sampler;

  //   isage::wtm::SampleWithKindsEveryIter sample_strat_heldout(num_heldout_iterations, heldout_burnin);
  //   sample_strat_heldout.heldout(true);
  //   heldout_sampler.sampling_strategy(&sample_strat_heldout);

  //   Model dm = heldout_sampler.reconstruct_model();
  //   heldout_sampler.model(&dm);
  //   // load corpus
  //   isage::wtm::Vocabulary<string> gkv = heldout_sampler.gov_kind_vocab();
  //   isage::wtm::Vocabulary<string> rkv = heldout_sampler.rel_kind_vocab();
  //   isage::wtm::Vocabulary<string> gv = heldout_sampler.gov_vocab();
  //   isage::wtm::Vocabulary<string> rv = heldout_sampler.rel_vocab();
  //   gkv.allow_new_words(false);
  //   rkv.allow_new_words(false);
  //   gv.allow_new_words(false);
  //   rv.allow_new_words(false);
  //   const bool run_with_latent_kinds = heldout_sampler.use_lexical();
  //   isage::wtm::Corpus<Doc> heldout_corpus =
  //     get_corpus<Doc, string, string>("dev_corpus", vm["dev"].as<std::string>(), 
  // 				      gkv, rkv, gv, rv,
  // 				      gov_lemma_stops,
  // 				      run_with_latent_kinds,
  // 				      lemma_mapper_ptr);
  //   //set corpus
  //   heldout_sampler.corpus(&heldout_corpus);

  //   bool need_to_sample = vm["compute-dev-ll"].as<bool>() | vm.count("compute-dev-label-tsv");
  //   if(need_to_sample) {
  //     heldout_sampler.latent_kinds(true);
  //     heldout_sampler.heldout_init();
  //     heldout_sampler.learn();
  //     heldout_sampler.transfer_learned_parameters();
  //     BOOST_LOG_TRIVIAL(info) << "Done with sampling heldout";
  //   }
  //   if(vm.count("compute-dev-label-tsv")) {
  //     heldout_sampler.print_labeling(vm["compute-dev-label-tsv"].as<std::string>());
  //   }
  //   if(vm["compute-dev-ll"].as<bool>()) {
  //     double heldout_ll =
  // 	dm.latent_and_kind_marginalized_loglikelihood(heldout_corpus, gv, rv,
  // 						      heldout_sampler.doc_template_params());
  //     BOOST_LOG_TRIVIAL(info) << "Held-out marginalized ll: " << heldout_ll;
  //   }
  //   if(vm.count("compute-coherence")) {
  //     if(!vm.count("coherence-file")) {
  // 	BOOST_LOG_TRIVIAL(error) << "You must supply a coherence file";
  // 	throw 5;
  //     }
  //     std::vector<std::string> which_coherences = vm["compute-coherence"].as<std::vector< std::string> >();
  //     std::ofstream myfile;
  //     myfile.open(vm["coherence-file"].as<std::string>());
  //     myfile << "which\tid\tcoherence\twhich_avg\n";
  //     for(const std::string& which : which_coherences) {
  // 	BOOST_LOG_TRIVIAL(error) << which;
  // 	std::vector<double> coherences;
  // 	if(which == "gov_kind") {
  // 	  coherences = 
  // 	    dm.compute_coherences< Doc, string>(vm["coherence-M"].as<int>(), which, heldout_corpus, gkv);
  // 	  int id = 0;
  // 	  double avg = narutil::average(coherences);
  // 	  for(const double coher : coherences) {
  // 	    myfile << which << "\t" << (id++) << "\t" << coher << "\t" << avg << "\n";
  // 	  }
  // 	}
  // 	if(which == "gov_obs") {
  // 	  coherences = 
  // 	    dm.compute_coherences< Doc, string>(vm["coherence-M"].as<int>(), which, heldout_corpus, gv);
  // 	  int id = 0;
  // 	  double avg = narutil::average(coherences);
  // 	  for(const double coher : coherences) {
  // 	    myfile << which << "\t" << (id++) << "\t" << coher << "\t" << avg << "\n";
  // 	  }
  // 	}
  // 	if(which == "gov_obs_kind_marginalized") {
  // 	  coherences = 
  // 	    dm.compute_coherences< Doc, string>(vm["coherence-M"].as<int>(), which, heldout_corpus, gv);
  // 	  int id = 0;
  // 	  double avg = narutil::average(coherences);
  // 	  for(const double coher : coherences) {
  // 	    myfile << which << "\t" << (id++) << "\t" << coher << "\t" << avg << "\n";
  // 	  }
  // 	}
  //     }
  //     myfile.close();
  //   }
  //   if(gov_lemma_stops != NULL) {
  //     delete gov_lemma_stops;
  //   }
  // }
  return 0;
}

// #include "concrete_util/io.h"
// #include "logging.hpp"
// #include "wtm.hpp"

// void init_logging()
// {
//   boost::log::core::get()->set_filter
//     (
//      boost::log::trivial::severity >= boost::log::trivial::info
//      );
// }

// int main() {
//   init_logging();
//   isage::wtm::SymmetricHyperparams shp = isage::wtm::SymmetricHyperparams();
//   shp.h_word = .1;
//   shp.h_theta = .1;
//   typedef std::string string;
//   isage::wtm::Vocabulary<string> vocabs("__OOV__");
//   typedef isage::wtm::Document< string > Doc;
//   isage::wtm::Corpus<Doc > corpus("my_corpus");

//   int num_comms = 0;
//   const isage::wtm::VerbPruner<string> word_pruner(&vocabs);
//   concrete::util::GZipCommunicationSequence concrete_reader("/tmp/rand10nyt.gz");
//   for(concrete_reader.begin(); concrete_reader.keep_reading(); ++concrete_reader) {
//     concrete::Communication communication = *concrete_reader;
//     BOOST_LOG_TRIVIAL(trace) << communication.id;
//     ++num_comms;
//     Doc my_doc(communication, word_pruner);
//     corpus.add_document(my_doc);
//   }

//   isage::wtm::SampleEveryIter sample_strat(1000,200);
//   isage::wtm::DiscreteLDA<string, std::vector<double> > dm(10, &shp, &vocabs);
//   isage::wtm::CollapsedGibbsDMC<isage::wtm::DiscreteLDA<string> , Doc, isage::wtm::Vocabulary<string> > sampler(&dm, &corpus, &vocabs);
//   sampler.sampling_strategy(&sample_strat);
//   sampler.init(corpus, vocabs);
//   sampler.learn(vocabs);
//   sampler.transfer_learned_parameters();
//   dm.print_topics(10, vocabs);
//   BOOST_LOG_TRIVIAL(info) << "Done with sampling";
//   return 0;
// }
