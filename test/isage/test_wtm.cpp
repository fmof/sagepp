#include "gtest/gtest.h"

#include "concrete_util/io.h"
#include "logging.hpp"
#include "wtm.hpp"
#include "wtm_sampling.hpp"
#include "wtm_variational.hpp"

#include <vector>

#include <gsl/gsl_rng.h>

TEST(RandomInitializer, run_support10) {
  isage::wtm::RandomInitializer ti(10);
  int i = 0;
  for(; i < 1000; ++i) {
    int x = ti(0,0);
    ASSERT_GE(x, 0);
    ASSERT_LT(x, 10);
  }
  ASSERT_EQ(1000, i);
}

TEST(GSL_RNG, use_trivial) {
  const gsl_rng_type *gsl_rng_ = gsl_rng_mt19937;
  gsl_rng* rnd_gen = gsl_rng_alloc(gsl_rng_);
  int i = 0;
  for(; i < 1000; ++i) {
    int x = gsl_rng_uniform_int(rnd_gen, 1);
    ASSERT_GE(x, 0);
    ASSERT_LT(x, 1);
  }
  ASSERT_EQ(1000, i);
}
TEST(GSL_RNG, use_1000) {
  const gsl_rng_type *gsl_rng_ = gsl_rng_mt19937;
  gsl_rng* rnd_gen = gsl_rng_alloc(gsl_rng_);
  int i = 0;
  for(; i < 1000; ++i) {
    int x = gsl_rng_uniform_int(rnd_gen, 1000);
    ASSERT_GE(x, 0);
    ASSERT_LT(x, 1000);
  }
  ASSERT_EQ(1000, i);
}

TEST(DiscreteLDASampler, run_100) {
  boost::log::core::get()->set_filter
    (
     boost::log::trivial::severity >= boost::log::trivial::debug
     );
  int num_iterations = 100;
  int burnin = 10;
  int num_topics = 5;
  typedef std::string string;
  typedef isage::wtm::Vocabulary<string> SVocab;
  typedef isage::wtm::Document< string > Doc;
  typedef isage::wtm::DiscreteLDA<string, std::vector<double> > Model;
  typedef isage::wtm::CollapsedGibbsDMC<Doc, string > Sampler;

  SVocab word_vocab("__WORD_OOV__");
  isage::wtm::SymmetricHyperparams shp = isage::wtm::SymmetricHyperparams();
  shp.h_word = 0.1;
  shp.h_theta = 0.1;
  isage::wtm::SampleEveryIter sample_strat(num_iterations, burnin);

  isage::wtm::Corpus< Doc > corpus("my_corpus");
  int num_comms = 0;
  concrete::util::CommunicationSequence *concrete_reader;
  concrete::util::get_communication_sequence( "test/resources/NYT_ENG_19980113.0597.dir", concrete_reader );
  for(concrete_reader->begin(); concrete_reader->keep_reading(); 
      concrete_reader->operator++()) {
    concrete::Communication communication = *(*concrete_reader);
    isage::wtm::WordPruner< string > wp(&word_vocab);
    BOOST_LOG_TRIVIAL(trace) << communication.id;
    ++num_comms;
    Doc my_doc(communication, wp);
    corpus.add_document(my_doc);
  }
  delete concrete_reader;
  ASSERT_EQ(1, num_comms);
  ASSERT_EQ(1, corpus.num_docs());
  Model dm(num_topics, &shp, &word_vocab);
  Sampler sampler(&dm, &corpus, &word_vocab);
  sampler.sampling_strategy(&sample_strat);
  isage::wtm::RandomInitializer ri(num_topics);
  sampler.init(ri);
  sampler.learn();
}

TEST(DiscreteVariational, create) {
  boost::log::core::get()->set_filter
    (
     boost::log::trivial::severity >= boost::log::trivial::trace
     );
  int num_iterations = 100;
  int burnin = 10;
  int num_topics = 5;
  typedef std::string string;
  typedef isage::wtm::Vocabulary<string> SVocab;
  typedef isage::wtm::Document< string > Doc;
  typedef isage::wtm::DiscreteLDA<string, std::vector<double> > Model;
  typedef isage::wtm::DiscreteVariational<Doc, string, std::vector<double> > Variational;

  SVocab word_vocab("__WORD_OOV__");
  isage::wtm::SymmetricHyperparams shp = isage::wtm::SymmetricHyperparams();
  shp.h_word = 0.1;
  shp.h_theta = 0.1;
  isage::wtm::SampleEveryIter sample_strat(num_iterations, burnin);

  isage::wtm::Corpus< Doc > corpus("my_corpus");
  int num_comms = 0;
  concrete::util::CommunicationSequence *concrete_reader;
  int num_words_total = 0;
  concrete::util::get_communication_sequence( "test/resources/NYT_ENG_19980113.0597.dir", concrete_reader );
  for(concrete_reader->begin(); concrete_reader->keep_reading(); 
      concrete_reader->operator++()) {
    concrete::Communication communication = *(*concrete_reader);
    isage::wtm::WordPruner< string > wp(&word_vocab);
    BOOST_LOG_TRIVIAL(trace) << communication.id;
    ++num_comms;
    Doc my_doc(communication, wp);
    num_words_total += my_doc.num_words();
    corpus.add_document(my_doc);
  }
  delete concrete_reader;
  ASSERT_EQ(1, num_comms);
  ASSERT_EQ(1, corpus.num_docs());
  Model dm(num_topics, &shp, &word_vocab);

  Variational var_inf(&dm, &corpus, &word_vocab);
  isage::wtm::UniformHyperSeedWeightedInitializer uni_mult(num_topics, num_comms, (double)num_words_total);
  var_inf.init(uni_mult);
  //test the initialization
  std::vector<std::vector<std::vector<double> > > vap = var_inf.var_assignment_params();
  ASSERT_EQ(1, vap.size());
  const int nw = corpus[0].num_words();
  ASSERT_EQ(nw, vap[0].size());
  for(int i = 0; i < nw; ++i) {
    ASSERT_EQ(num_topics, vap[0][i].size());
    for(int j = 0; j < num_topics; ++j) {
      ASSERT_EQ(1.0/(double)num_topics, vap[0][i][j]);
    }
  }
  std::vector<std::vector<double> > vtp = var_inf.var_topic_params();
  ASSERT_EQ(num_topics, vtp.size());
  for(int i = 0; i < num_topics; ++i) {
    ASSERT_EQ(word_vocab.num_words(), vtp[i].size());
    for(int j = 0; j < word_vocab.num_words(); ++j) {
      ASSERT_EQ(shp.h_word + (double)num_words_total/(double)word_vocab.num_words(), vtp[i][j]);
    }
  }
  std::vector<std::vector<double> > vtup = var_inf.var_topic_usage_params();
  ASSERT_EQ(1, vtup.size());
  ASSERT_EQ(num_topics, vtup[0].size());
  for(int j = 0; j < num_topics; ++j) {
    ASSERT_EQ(shp.h_theta + (double)num_comms/(double)num_topics, vtup[0][j]);
  }

  var_inf.learn();
}
