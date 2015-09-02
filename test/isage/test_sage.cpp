#include "gtest/gtest.h"

#include "sage.hpp"
#include "wtm.hpp"

#include <algorithm>
#include <vector>

TEST(SageTopic, create) {
  isage::wtm::SageTopic<std::vector<double> > vec(10, 0.0, isage::wtm::SageTopicRegularization::L2);
}
TEST(SageTopic, set) {
  isage::wtm::SageTopic<std::vector<double> > vec(10, 0.0, isage::wtm::SageTopicRegularization::L2);
  vec[5] = 10;
  ASSERT_EQ(10, vec[5]);
}
TEST(SageTopic, eta_set) {
  typedef std::vector<double> Eta;
  typedef isage::wtm::SageTopic< Eta > Topic;
  Topic topic(2, 1.0, isage::wtm::SageTopicRegularization::L2);
  Eta neta(2, 0.0);
  neta[0] =  1.0;
  neta[1] = -0.5;
  // Set the background to be 1:
  Eta* background = new Eta(2, 1.0);
  topic.background(background);
  // Set eta
  topic.eta(neta);
  Eta stored_eta = topic.eta();
  // Test that the data are correct
  ASSERT_EQ(2, stored_eta.size());
  ASSERT_EQ(neta.size(), stored_eta.size());
  for(int i = 0; i < topic.support_size(); ++i) {
    ASSERT_NEAR(neta[i], stored_eta[i], 1E-5);
  }
  // test the log-normalizer
  ASSERT_NEAR(2.2014132779827524, topic.log_partition(), 1E-5);
  // test the probability estimate
  ASSERT_NEAR(0.8175744761936438, topic.probability(0), 1E-5);
  ASSERT_NEAR(0.18242552380635635, topic.probability(1), 1E-5);
  delete background;  
}
TEST(DenseSageTopic, create) {
  isage::wtm::DenseSageTopic vec(10, 0.0, isage::wtm::SageTopicRegularization::L2);
}
TEST(DenseSageTopic, set) {
  isage::wtm::DenseSageTopic vec(10, 0.0, isage::wtm::SageTopicRegularization::L2);
  vec[5] = 10;
  ASSERT_EQ(10, vec[5]);
}

TEST(SageInference, infer) {
  typedef std::string string;
  typedef string VocabType;
  typedef isage::wtm::Vocabulary< VocabType > Vocab;
  typedef int CountType;
  typedef isage::wtm::Document< VocabType, CountType > Doc;
  typedef isage::wtm::Corpus< Doc > Corpus;
  typedef isage::wtm::SageTopic<std::vector<double> > TopicType;
  typedef isage::wtm::DiscreteLDA< VocabType, std::vector<double> > Model;
  typedef isage::wtm::SageVariationalHighMem< Doc, VocabType, TopicType > Variational;

#ifndef ISAGE_LOG_AS_COUT
  boost::log::core::get()->set_filter
    (
     boost::log::trivial::severity >= boost::log::trivial::warning
     );
#endif

  Vocab vocab("OOV");
  vocab.make_word("a"); // 1
  vocab.make_word("b"); // 2
  vocab.make_word("c"); // 3
  vocab.make_word("d"); // 4
  vocab.make_word("z"); // 5
  vocab.make_word("y"); // 6
  vocab.make_word("x"); // 7
  vocab.make_word("w"); // 8
  Doc doc0 = 
    Doc::from_svm_light_string("1 1:3 2:1 3:1 4:3 5:1 8:1 # doc_0", vocab);
  Doc doc1 = 
    Doc::from_svm_light_string("1 1:1 3:1 5:1 6:4 7:2 8:1 # doc_1", vocab);
  Corpus corpus("train");  
  corpus.add_document(doc0);
  corpus.add_document(doc1);
  const int num_topics = 2;
  const int num_words_total = 20;
 
  isage::wtm::SymmetricHyperparams shp;
  shp.h_theta = 1.0/(double)num_topics;
  shp.h_word =  0.1; // unneeded, really
  Model dm(num_topics, &shp, &vocab);
  Variational var_inf(&dm, &corpus, &vocab,
		      isage::wtm::SageTopicRegularization::L2);
  isage::wtm::SageInitializer initer(num_topics, num_words_total);
  var_inf.alloc();

  std::vector<std::vector<int> > e_words = {
    {1, 2, 3, 4, 5, 8},
    {1, 3, 5, 6, 7, 8}
  };
  var_inf.words_in_docs(e_words);
  std::vector<std::vector<int> > e_word_counts = {
    {3, 1, 1, 3, 1, 1},
    {1, 1, 1, 4, 2, 1}
  };
  var_inf.word_type_counts(e_word_counts);
  // now set the background
  //const double log_norm = gsl_sf_log(1 + 0.0001);
  // std::vector<double> e_back = {
  //   gsl_sf_log(0.0001/20.0) - log_norm,
  //   -gsl_sf_log(5) - log_norm,
  //   -gsl_sf_log(20) - log_norm,
  //   -gsl_sf_log(10) - log_norm,
  //   -gsl_sf_log(20.0/3.0) - log_norm,
  //   -gsl_sf_log(10) - log_norm,
  //   -gsl_sf_log(5) - log_norm,
  //   -gsl_sf_log(10) - log_norm,
  //   -gsl_sf_log(10) - log_norm
  // };
  std::vector<double> e_back(9, 0.0);
  var_inf.background(&e_back);
  var_inf.propagate_background();
  // set the variational parameters
  for(int t = 0; t < num_topics; ++t) {
    std::vector<double> u(num_topics, 1.0/(double)num_topics);
    for(int x = 0; x < num_topics; ++x) {
      u[x] += ((t % 2 == 0) ? 1 : -1) * ((double)(1 + (x % 5))/50.0);
    }
    var_inf.var_topic_usage_params(t, u);
  }
  for(int di = 0; di < 2; ++di) {
    for(size_t i = 0; i < e_words[di].size(); ++i) {
      std::vector<double> a(num_topics, 1.0/(double)num_topics);
      for(int x = 0; x < num_topics; ++x) {
	a[x] += ((di % 2 == 0) ? 1 : -1) * ((double)(1 + (x % 5))/50.0);
      }
      var_inf.var_assignment_params(di, i, a);
    }
  }
  // now set the topics
  // std::vector<double> e_weights = {
  //   4.99943e-06, .2, 0.0499993, 0.0999991, 0.149999,
  //   0.0999991, .2, 0.0999991, 0.0999991
  // };
  std::vector<double> e_weights = {
    4.99943e-06, .2, 0.0499993, 0.03491, 0.149999,
    0.000991, .2, 0.0999991, 0.9991
  };
  std::vector<double> copy(e_weights);
  for(int t = 0; t < num_topics; ++t) {
    std::next_permutation(copy.begin(), copy.end());
    std::vector<double> c2(copy);
    if(t % 2 == 0) {
      isage::util::exp(&c2);
      isage::util::exp(&c2);
    }
    var_inf.topic(t)->eta(c2);
    std::next_permutation(copy.begin(), copy.end());
  }
  // std::vector<double> e_probs = isage::util::sum(e_weights, e_back);
  std::vector<double> e_probs(e_weights);
  isage::util::exp(&e_probs);
  double Z = isage::util::sum(e_probs);
  isage::util::scalar_product(1.0/Z, &e_probs);
  // for(int w = 0; w < 9; ++w) {
  //   ASSERT_NEAR(e_probs[w], var_inf.topic(0)->probability(w), 1E-5);
  //   ASSERT_NEAR(e_probs[w], var_inf.topic(1)->probability(w), 1E-5);
  //   ASSERT_NEAR(var_inf.topic(0)->probability(w), var_inf.topic(1)->probability(w), 1E-8);
  // }


  // std::vector<std::vector<double> > usages = var_inf.var_topic_usage_params();
  // isage::util::print_2d(usages);
  // std::cout << std::endl;

#ifndef ISAGE_LOG_AS_COUT
  boost::log::core::get()->set_filter
    (
     boost::log::trivial::severity >= boost::log::trivial::warning
     );
#endif
  isage::wtm::SageStrategy strategy;
  strategy.heldout = false;
  strategy.hyper_update_iter = 50;
  var_inf.learn(strategy);
#ifndef ISAGE_LOG_AS_COUT
  boost::log::core::get()->set_filter
    (
     boost::log::trivial::severity >= boost::log::trivial::info
     );
#endif
}

TEST(SageInference, infer_limmem) {
  typedef std::string string;
  typedef string VocabType;
  typedef isage::wtm::Vocabulary< VocabType > Vocab;
  typedef int CountType;
  typedef isage::wtm::Document< VocabType, CountType > Doc;
  typedef isage::wtm::Corpus< Doc > Corpus;
  typedef isage::wtm::SageTopic<std::vector<double> > TopicType;
  typedef isage::wtm::DiscreteLDA< VocabType, std::vector<double> > Model;
  typedef isage::wtm::SageVariationalLimMem< Doc, VocabType, TopicType > Variational;

#ifndef ISAGE_LOG_AS_COUT
  boost::log::core::get()->set_filter
    (
     boost::log::trivial::severity >= boost::log::trivial::warning
     );
#endif

  Vocab vocab("OOV");
  vocab.make_word("a"); // 1
  vocab.make_word("b"); // 2
  vocab.make_word("c"); // 3
  vocab.make_word("d"); // 4
  vocab.make_word("z"); // 5
  vocab.make_word("y"); // 6
  vocab.make_word("x"); // 7
  vocab.make_word("w"); // 8
  Doc doc0 = 
    Doc::from_svm_light_string("1 1:3 2:1 3:1 4:3 5:1 8:1 # doc_0", vocab);
  Doc doc1 = 
    Doc::from_svm_light_string("1 1:1 3:1 5:1 6:4 7:2 8:1 # doc_1", vocab);
  Corpus corpus("train");  
  corpus.add_document(doc0);
  corpus.add_document(doc1);
  const int num_topics = 2;
  const int num_words_total = 20;
 
  isage::wtm::SymmetricHyperparams shp;
  shp.h_theta = 1.0/(double)num_topics;
  shp.h_word =  0.1; // unneeded, really
  Model dm(num_topics, &shp, &vocab);
  Variational var_inf(&dm, &corpus, &vocab,
		      isage::wtm::SageTopicRegularization::L2);
  isage::wtm::SageInitializer initer(num_topics, num_words_total);
  var_inf.alloc();

  std::vector<std::vector<int> > e_words = {
    {1, 2, 3, 4, 5, 8},
    {1, 3, 5, 6, 7, 8}
  };
  var_inf.words_in_docs(e_words);
  std::vector<std::vector<int> > e_word_counts = {
    {3, 1, 1, 3, 1, 1},
    {1, 1, 1, 4, 2, 1}
  };
  var_inf.word_type_counts(e_word_counts);
  // now set the background
  //const double log_norm = gsl_sf_log(1 + 0.0001);
  // std::vector<double> e_back = {
  //   gsl_sf_log(0.0001/20.0) - log_norm,
  //   -gsl_sf_log(5) - log_norm,
  //   -gsl_sf_log(20) - log_norm,
  //   -gsl_sf_log(10) - log_norm,
  //   -gsl_sf_log(20.0/3.0) - log_norm,
  //   -gsl_sf_log(10) - log_norm,
  //   -gsl_sf_log(5) - log_norm,
  //   -gsl_sf_log(10) - log_norm,
  //   -gsl_sf_log(10) - log_norm
  // };
  std::vector<double> e_back(9, 0.0);
  var_inf.background(&e_back);
  var_inf.propagate_background();
  // set the variational parameters
  for(int t = 0; t < num_topics; ++t) {
    std::vector<double> u(num_topics, 1.0/(double)num_topics);
    for(int x = 0; x < num_topics; ++x) {
      u[x] += ((t % 2 == 0) ? 1 : -1) * ((double)(1 + (x % 5))/50.0);
    }
    var_inf.var_topic_usage_params(t, u);
  }
  for(int di = 0; di < 2; ++di) {
    for(size_t i = 0; i < e_words[di].size(); ++i) {
      std::vector<double> a(num_topics, 1.0/(double)num_topics);
      for(int x = 0; x < num_topics; ++x) {
	a[x] += ((di % 2 == 0) ? 1 : -1) * ((double)(1 + (x % 5))/50.0);
      }
    }
  }
  // now set the topics
  // std::vector<double> e_weights = {
  //   4.99943e-06, .2, 0.0499993, 0.0999991, 0.149999,
  //   0.0999991, .2, 0.0999991, 0.0999991
  // };
  std::vector<double> e_weights = {
    4.99943e-06, .2, 0.0499993, 0.03491, 0.149999,
    0.000991, .2, 0.0999991, 0.9991
  };
  std::vector<double> copy(e_weights);
  for(int t = 0; t < num_topics; ++t) {
    std::next_permutation(copy.begin(), copy.end());
    std::vector<double> c2(copy);
    if(t % 2 == 0) {
      isage::util::exp(&c2);
      isage::util::exp(&c2);
    }
    var_inf.topic(t)->eta(c2);
    std::next_permutation(copy.begin(), copy.end());
  }
  // std::vector<double> e_probs = isage::util::sum(e_weights, e_back);
  std::vector<double> e_probs(e_weights);
  isage::util::exp(&e_probs);
  double Z = isage::util::sum(e_probs);
  isage::util::scalar_product(1.0/Z, &e_probs);
  // for(int w = 0; w < 9; ++w) {
  //   ASSERT_NEAR(e_probs[w], var_inf.topic(0)->probability(w), 1E-5);
  //   ASSERT_NEAR(e_probs[w], var_inf.topic(1)->probability(w), 1E-5);
  //   ASSERT_NEAR(var_inf.topic(0)->probability(w), var_inf.topic(1)->probability(w), 1E-8);
  // }


  // std::vector<std::vector<double> > usages = var_inf.var_topic_usage_params();
  // isage::util::print_2d(usages);
  // std::cout << std::endl;

#ifndef ISAGE_LOG_AS_COUT
  boost::log::core::get()->set_filter
    (
     boost::log::trivial::severity >= boost::log::trivial::warning
     );
#endif
  isage::wtm::SageStrategy strategy;
  strategy.heldout = false;
  strategy.hyper_update_iter = 50;
  var_inf.learn(strategy);
#ifndef ISAGE_LOG_AS_COUT
  boost::log::core::get()->set_filter
    (
     boost::log::trivial::severity >= boost::log::trivial::info
     );
#endif
}


TEST(SageInference, infer_lim_mem_vs_full) {
  typedef std::string string;
  typedef string VocabType;
  typedef isage::wtm::Vocabulary< VocabType > Vocab;
  typedef int CountType;
  typedef isage::wtm::Document< VocabType, CountType > Doc;
  typedef isage::wtm::Corpus< Doc > Corpus;
  typedef isage::wtm::SageTopic<std::vector<double> > TopicType;
  typedef isage::wtm::DiscreteLDA< VocabType, std::vector<double> > Model;
  typedef isage::wtm::SageVariationalHighMem< Doc, VocabType, TopicType > Variational;
  typedef isage::wtm::SageVariationalLimMem< Doc, VocabType, TopicType > VariationalLim;

#ifndef ISAGE_LOG_AS_COUT
  boost::log::core::get()->set_filter
    (
     boost::log::trivial::severity >= boost::log::trivial::warning
     );
#endif

  Vocab vocab("OOV");
  vocab.make_word("a"); // 1
  vocab.make_word("b"); // 2
  vocab.make_word("c"); // 3
  vocab.make_word("d"); // 4
  vocab.make_word("z"); // 5
  vocab.make_word("y"); // 6
  vocab.make_word("x"); // 7
  vocab.make_word("w"); // 8
  Doc doc0 = 
    Doc::from_svm_light_string("1 1:3 2:1 3:1 4:3 5:1 8:1 # doc_0", vocab);
  Doc doc1 = 
    Doc::from_svm_light_string("1 1:1 3:1 5:1 6:4 7:2 8:1 # doc_1", vocab);
  Corpus corpus("train");  
  corpus.add_document(doc0);
  corpus.add_document(doc1);
  const int num_topics = 2;
  const int num_words_total = 20;
 
  isage::wtm::SymmetricHyperparams shp;
  shp.h_theta = 1.0/(double)num_topics;
  shp.h_word =  0.1; // unneeded, really
  Model dm(num_topics, &shp, &vocab);
  Variational var_inf(&dm, &corpus, &vocab,
		      isage::wtm::SageTopicRegularization::L2);
  VariationalLim var_inf_l(&dm, &corpus, &vocab,
			   isage::wtm::SageTopicRegularization::L2);
  isage::wtm::SageInitializer initer(num_topics, num_words_total);
  var_inf.alloc();
  var_inf_l.alloc();

  std::vector<std::vector<int> > e_words = {
    {1, 2, 3, 4, 5, 8},
    {1, 3, 5, 6, 7, 8}
  };
  var_inf.words_in_docs(e_words);
  var_inf_l.words_in_docs(e_words);
  std::vector<std::vector<int> > e_word_counts = {
    {3, 1, 1, 3, 1, 1},
    {1, 1, 1, 4, 2, 1}
  };
  var_inf.word_type_counts(e_word_counts);
  var_inf_l.word_type_counts(e_word_counts);
  std::vector<double> e_back(9, 0.0);
  var_inf.background(&e_back);
  var_inf.propagate_background();
  var_inf_l.background(&e_back);
  var_inf_l.propagate_background();
  // set the variational parameters
  for(int t = 0; t < num_topics; ++t) {
    std::vector<double> u(num_topics, 1.0/(double)num_topics);
    for(int x = 0; x < num_topics; ++x) {
      u[x] += ((t % 2 == 0) ? 1 : -1) * ((double)(1 + (x % 5))/50.0);
    }
    var_inf.var_topic_usage_params(t, u);
    var_inf_l.var_topic_usage_params(t, u);
  }
  for(int di = 0; di < 2; ++di) {
    for(size_t i = 0; i < e_words[di].size(); ++i) {
      std::vector<double> a(num_topics, 1.0/(double)num_topics);
      for(int x = 0; x < num_topics; ++x) {
	a[x] += ((di % 2 == 0) ? 1 : -1) * ((double)(1 + (x % 5))/50.0);
      }
      var_inf.var_assignment_params(di, i, a);
      // NOTE: var_inf_l doesn't have .var_assignment_params
    }
  }
  // now set the topics
  // std::vector<double> e_weights = {
  //   4.99943e-06, .2, 0.0499993, 0.0999991, 0.149999,
  //   0.0999991, .2, 0.0999991, 0.0999991
  // };
  std::vector<double> e_weights = {
    4.99943e-06, .2, 0.0499993, 0.03491, 0.149999,
    0.000991, .2, 0.0999991, 0.9991
  };
  std::vector<double> copy(e_weights);
  for(int t = 0; t < num_topics; ++t) {
    std::next_permutation(copy.begin(), copy.end());
    std::vector<double> c2(copy);
    if(t % 2 == 0) {
      isage::util::exp(&c2);
      isage::util::exp(&c2);
    }
    var_inf.topic(t)->eta(c2);
    var_inf_l.topic(t)->eta(c2);
    std::next_permutation(copy.begin(), copy.end());
  }
  std::vector<double> e_probs(e_weights);
  isage::util::exp(&e_probs);
  double Z = isage::util::sum(e_probs);
  isage::util::scalar_product(1.0/Z, &e_probs);

#ifndef ISAGE_LOG_AS_COUT
  boost::log::core::get()->set_filter
    (
     boost::log::trivial::severity >= boost::log::trivial::warning
     );
#endif
  isage::wtm::SageStrategy strategy;
  strategy.heldout = false;
  strategy.hyper_update_iter = -1;
  var_inf.learn(strategy);
  var_inf_l.learn(strategy);
  // Now check to make sure that the usage posteriors are the same
  {
    std::vector<std::vector<double> > full_usage = var_inf.var_topic_usage_params();
    std::vector<std::vector<double> > lim_usage = var_inf_l.var_topic_usage_params();
    const size_t full_size = full_usage.size();
    const size_t lim_size = lim_usage.size();
    ASSERT_EQ(full_size, lim_size);
    for(size_t i = 0; i < full_size; ++i) {
      const size_t full_i_size = full_usage[i].size();
      const size_t lim_i_size = lim_usage[i].size();
      ASSERT_EQ(full_i_size, lim_i_size);
      for(size_t j = 0; j < full_i_size; ++j) {
	ASSERT_NEAR(full_usage[i][j], lim_usage[i][j], 1E-6);
      }
    }
  }

  {
    std::vector<std::vector<double> > full_ecounts = var_inf.expected_counts();
    std::vector<std::vector<double> > lim_ecounts = var_inf_l.expected_counts();
    const size_t full_size = full_ecounts.size();
    const size_t lim_size = lim_ecounts.size();
    ASSERT_EQ(full_size, lim_size);
    for(size_t i = 0; i < full_size; ++i) {
      const size_t full_i_size = full_ecounts[i].size();
      const size_t lim_i_size = lim_ecounts[i].size();
      ASSERT_EQ(full_i_size, lim_i_size);
      for(size_t j = 0; j < full_i_size; ++j) {
	ASSERT_NEAR(full_ecounts[i][j], lim_ecounts[i][j], 1E-6);
      }
    }
  }
#ifndef ISAGE_LOG_AS_COUT
  boost::log::core::get()->set_filter
    (
     boost::log::trivial::severity >= boost::log::trivial::info
     );
#endif
}

TEST(SageInference, infer_lim_mem_single_multithread) {
  typedef std::string string;
  typedef string VocabType;
  typedef isage::wtm::Vocabulary< VocabType > Vocab;
  typedef int CountType;
  typedef isage::wtm::Document< VocabType, CountType > Doc;
  typedef isage::wtm::Corpus< Doc > Corpus;
  typedef isage::wtm::SageTopic<std::vector<double> > TopicType;
  typedef isage::wtm::DiscreteLDA< VocabType, std::vector<double> > Model;
  typedef isage::wtm::SageVariationalLimMem< Doc, VocabType, TopicType > VariationalLim;

#ifndef ISAGE_LOG_AS_COUT
  boost::log::core::get()->set_filter
    (
     boost::log::trivial::severity >= boost::log::trivial::warning
     );
#endif

  Vocab vocab("OOV");
  vocab.make_word("a"); // 1
  vocab.make_word("b"); // 2
  vocab.make_word("c"); // 3
  vocab.make_word("d"); // 4
  vocab.make_word("z"); // 5
  vocab.make_word("y"); // 6
  vocab.make_word("x"); // 7
  vocab.make_word("w"); // 8
  Doc doc0 = 
    Doc::from_svm_light_string("1 1:3 2:1 3:1 4:3 5:1 8:1 # doc_0", vocab);
  Doc doc1 = 
    Doc::from_svm_light_string("1 1:1 3:1 5:1 6:4 7:2 8:1 # doc_1", vocab);
  Corpus corpus("train");  
  corpus.add_document(doc0);
  corpus.add_document(doc1);
  const int num_topics = 2;
  const int num_words_total = 20;
 
  isage::wtm::SymmetricHyperparams shp;
  shp.h_theta = 1.0/(double)num_topics;
  shp.h_word =  0.1; // unneeded, really
  Model dm(num_topics, &shp, &vocab);
  const int num_tests = 50;
  std::vector<int> e_thread_nums = {1, 100};
  std::vector<int> m_thread_nums = {1, 2};
  for(int test_num = 0; test_num < num_tests; ++test_num) {
    for(int e_thread : e_thread_nums) {
      for(int m_thread : m_thread_nums) {
	DEBUG << "Run " << test_num << " with " << e_thread << " E-step threads and " << m_thread << " M-step threads.";
	VariationalLim var_inf(&dm, &corpus, &vocab,
			       isage::wtm::SageTopicRegularization::L2);
	VariationalLim var_inf_l(&dm, &corpus, &vocab,
				 isage::wtm::SageTopicRegularization::L2);
	isage::wtm::SageInitializer initer(num_topics, num_words_total);
	var_inf.alloc();
	var_inf_l.alloc();

	std::vector<std::vector<int> > e_words = {
	  {1, 2, 3, 4, 5, 8},
	  {1, 3, 5, 6, 7, 8}
	};
	var_inf.words_in_docs(e_words);
	var_inf_l.words_in_docs(e_words);
	std::vector<std::vector<int> > e_word_counts = {
	  {3, 1, 1, 3, 1, 1},
	  {1, 1, 1, 4, 2, 1}
	};
	var_inf.word_type_counts(e_word_counts);
	var_inf_l.word_type_counts(e_word_counts);
	std::vector<double> e_back(9, 0.0);
	var_inf.background(&e_back);
	var_inf.propagate_background();
	var_inf_l.background(&e_back);
	var_inf_l.propagate_background();
	// set the variational parameters
	for(int t = 0; t < num_topics; ++t) {
	  std::vector<double> u(num_topics, 1.0/(double)num_topics);
	  for(int x = 0; x < num_topics; ++x) {
	    u[x] += ((t % 2 == 0) ? 1 : -1) * ((double)(1 + (x % 5))/50.0);
	  }
	  var_inf.var_topic_usage_params(t, u);
	  var_inf_l.var_topic_usage_params(t, u);
	}
	for(int di = 0; di < 2; ++di) {
	  for(size_t i = 0; i < e_words[di].size(); ++i) {
	    std::vector<double> a(num_topics, 1.0/(double)num_topics);
	    for(int x = 0; x < num_topics; ++x) {
	      a[x] += ((di % 2 == 0) ? 1 : -1) * ((double)(1 + (x % 5))/50.0);
	    }
	  }
	}
	// now set the topics
	// std::vector<double> e_weights = {
	//   4.99943e-06, .2, 0.0499993, 0.0999991, 0.149999,
	//   0.0999991, .2, 0.0999991, 0.0999991
	// };
	std::vector<double> e_weights = {
	  4.99943e-06, .2, 0.0499993, 0.03491, 0.149999,
	  0.000991, .2, 0.0999991, 0.9991
	};
	std::vector<double> copy(e_weights);
	for(int t = 0; t < num_topics; ++t) {
	  std::next_permutation(copy.begin(), copy.end());
	  std::vector<double> c2(copy);
	  if(t % 2 == 0) {
	    isage::util::exp(&c2);
	    isage::util::exp(&c2);
	  }
	  var_inf.topic(t)->eta(c2);
	  var_inf_l.topic(t)->eta(c2);
	  std::next_permutation(copy.begin(), copy.end());
	}
	std::vector<double> e_probs(e_weights);
	isage::util::exp(&e_probs);
	double Z = isage::util::sum(e_probs);
	isage::util::scalar_product(1.0/Z, &e_probs);

#ifndef ISAGE_LOG_AS_COUT
	boost::log::core::get()->set_filter
	  (
	   boost::log::trivial::severity >= boost::log::trivial::warning
	   );
#endif
	isage::wtm::SageStrategy ut;
	ut.num_m_threads = 1;
	ut.num_e_threads = 1;
	ut.heldout = false;
	ut.hyper_update_iter = -1;
	var_inf.learn(ut);

	isage::wtm::SageStrategy mt;
	mt.num_m_threads = m_thread;
	mt.num_e_threads = e_thread;
	mt.heldout = false;
	mt.hyper_update_iter = -1;
	var_inf_l.learn(mt);
	// Now check to make sure that the usage posteriors are the same
	{
	  std::vector<std::vector<double> > full_usage = var_inf.var_topic_usage_params();
	  std::vector<std::vector<double> > lim_usage = var_inf_l.var_topic_usage_params();
	  const size_t full_size = full_usage.size();
	  const size_t lim_size = lim_usage.size();
	  ASSERT_EQ(full_size, lim_size);
	  for(size_t i = 0; i < full_size; ++i) {
	    const size_t full_i_size = full_usage[i].size();
	    const size_t lim_i_size = lim_usage[i].size();
	    ASSERT_EQ(full_i_size, lim_i_size);
	    for(size_t j = 0; j < full_i_size; ++j) {
	      ASSERT_NEAR(full_usage[i][j], lim_usage[i][j], 1E-6);
	    }
	  }
	}

	{
	  std::vector<std::vector<double> > full_ecounts = var_inf.expected_counts();
	  std::vector<std::vector<double> > lim_ecounts = var_inf_l.expected_counts();
	  const size_t full_size = full_ecounts.size();
	  const size_t lim_size = lim_ecounts.size();
	  ASSERT_EQ(full_size, lim_size);
	  for(size_t i = 0; i < full_size; ++i) {
	    const size_t full_i_size = full_ecounts[i].size();
	    const size_t lim_i_size = lim_ecounts[i].size();
	    ASSERT_EQ(full_i_size, lim_i_size);
	    for(size_t j = 0; j < full_i_size; ++j) {
	      ASSERT_NEAR(full_ecounts[i][j], lim_ecounts[i][j], 1E-6);
	    }
	  }
	}
#ifndef ISAGE_LOG_AS_COUT
	boost::log::core::get()->set_filter
	  (
	   boost::log::trivial::severity >= boost::log::trivial::info
	   );
#endif
      }
    }
  }
}
