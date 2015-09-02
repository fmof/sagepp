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

int main(int n_args, char** args) {
  init_logging();
  isage::util::print_pstats();

  int num_topics;

  std::string output_usage_name;
  std::string output_topic_name;
  std::string assignment_usage_name;
  std::string heldout_output_usage_name;
  std::string background_output_name;

  po::variables_map vm;
  {
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help", "produce help message")
      ("corpus", po::value< std::string >(), 
       "input training path")
      ////////////////////////////////
      ("serialized-inferencer", po::value<std::string>(),
       "filename to READ serialized inference state from")
      ///////////////////////////////
      ("coherence-file", po::value<std::string>(),
       "file to write out coherence results")
      ("coherence-M", po::value<int>(), "take top M for each distribution while computing coherences")
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
  //typedef isage::wtm::DenseSageTopic TopicType;
  typedef std::vector<double> InnerTopicType;
  typedef isage::wtm::SageTopic< InnerTopicType > TopicType;
  typedef isage::wtm::DiscreteLDA< VocabType, std::vector<double> > Model;
  //typedef isage::wtm::DiscreteLDA< VocabType, TopicType > Model;
  SAGE_TYPEDEF_BOILER(Doc, VocabType, TopicType);

  Corpus heldout_corpus;
  int num_heldout_words = 0;

  Variational heldout_inferencer;
  SVocab word_vocab;

  std::ifstream ifs(vm["serialized-inferencer"].as<std::string>(),
		    std::ios::in|std::ios::binary);
  assert(ifs.good());
  boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
  in.push(boost::iostreams::gzip_decompressor());
  in.push(ifs);
  boost::archive::binary_iarchive ia(in);
  ia >> heldout_inferencer;
  word_vocab = *(heldout_inferencer.vocab());
  word_vocab.allow_new_words(false);
  heldout_corpus = 
    Corpus("coherence_corpus",
	   vm["corpus"].as<std::string>(), word_vocab);
  num_heldout_words = get_num_tokens(heldout_corpus);
  INFO << "Number of heldout documents: " << heldout_corpus.num_docs();
  INFO << "Number of heldout word tokens: " << num_heldout_words;

  num_topics = heldout_inferencer.num_topics();
  Model dm = heldout_inferencer.reconstruct_model();
  heldout_inferencer.model(&dm);
  heldout_inferencer.corpus(&heldout_corpus);
  isage::wtm::SageInitializer h_initer(num_topics, num_heldout_words);
  heldout_inferencer.reinit(h_initer);

  INFO << "Computing coherence";

  if(!vm.count("coherence-file")) {
    ERROR << "You must supply a coherence file";
    throw 5;
  }
  std::vector<InnerTopicType> topics = heldout_inferencer.get_topic_etas<InnerTopicType>();

  std::vector<double> coherences = 
    Model::compute_coherences< Doc >(vm["coherence-M"].as<int>(), heldout_corpus, word_vocab,
				     topics);
  int id = 0;
  double avg = isage::util::average(coherences);

  std::ofstream myfile;
  myfile.open(vm["coherence-file"].as<std::string>());
  myfile << "which\tid\tcoherence\twhich_avg\n";
  for(const double coher : coherences) {
    myfile << "words\t" << (id++) << "\t" << coher << "\t" << avg << "\n";
  }
  myfile.close();
  INFO << "Done computing coherence";
  INFO << "Average coherence: " << avg;
  return 0;
}
