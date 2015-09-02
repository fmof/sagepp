#include "gtest/gtest.h"

#include "logging.hpp"
#include "sage.hpp"
#include "wtm.hpp"
#include "wtm_sampling.hpp"
#include "wtm_variational.hpp"

#include <string>
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

TEST(Vocabulary, create_from_nl_file) {
  std::string path = "test/resources/vocab_list_simple.txt";
  typedef isage::wtm::Vocabulary<std::string> Vocab;
  Vocab vocab = Vocab::from_file(path, "__OOV__");
  std::vector<std::string> expected_words = {
    "__OOV__", "The", "dog", "ran", "with", "the",
    "man", ".", "cat", "meowed"
  };
  ASSERT_EQ(10, vocab.num_words());
  for(size_t i = 0; i < 10; ++i) {
    ASSERT_EQ(expected_words[i], vocab.word(i));
  }
  vocab.make_word("foo_bar");
  ASSERT_EQ(10, vocab.num_words());
}

TEST(Document, str_svm_light_file) {
  #ifndef ISAGE_LOG_AS_COUT
  boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
  #endif
  std::string path = "test/resources/vocab_list_simple.txt";
  typedef isage::wtm::Vocabulary<std::string> Vocab;
  Vocab vocab = Vocab::from_file(path, "__OOV__");
  typedef isage::wtm::Document<std::string, int> Doc;
  std::map<std::string, int> counts;
  counts["The"] = 3;
  counts["dog"] = 1;
  counts["ran"] = 2;
  counts["with"] = 2;
  counts["the"] = 2;
  counts["man"] = 2;
  counts["."] = 3;
  counts["cat"] = 2;
  counts["meowed"] = 1;
  {
    Doc doc =  
      Doc::from_svm_light_string("1 1:3 2:1 3:2 4:2 5:2 6:2 7:3 8:2 9:1 # doc_1",
				 vocab);
    ASSERT_EQ("doc_1", doc.id);
    typename Doc::Multinomial mult = doc.multinomial();
    ASSERT_EQ(9, mult.size());
    ASSERT_EQ(18, doc.num_words());
    ASSERT_TRUE(doc.type_view());
    for(const auto& pair : mult) {
      ASSERT_EQ(counts[pair.first], pair.second) << "Counts not equal for " << pair.first;
    }
  }
  {
    Doc doc = Doc::from_svm_light_string("1 1:3 2:1 3:2 4:2 5:2 6:2 7:3 8:2 9:1 # ",
					 vocab);
    ASSERT_EQ("UnknownDocId", doc.id);
    typename Doc::Multinomial mult = doc.multinomial();
    ASSERT_EQ(9, mult.size());
    ASSERT_EQ(18, doc.num_words());
    ASSERT_TRUE(doc.type_view());
    for(const auto& pair : mult) {
      ASSERT_EQ(counts[pair.first], pair.second) << "Counts not equal for " << pair.first;
    }
  } 
  {
    Doc doc = Doc::from_svm_light_string("1 1:3 2:1 3:2 4:2 5:2 6:2 7:3 8:2 9:1  ",
					 vocab);
    ASSERT_EQ("UnknownDocId", doc.id);
    typename Doc::Multinomial mult = doc.multinomial();
    ASSERT_EQ(9, mult.size());
    ASSERT_EQ(18, doc.num_words());
    ASSERT_TRUE(doc.type_view());
    for(const auto& pair : mult) {
      ASSERT_EQ(counts[pair.first], pair.second) << "Counts not equal for " << pair.first;
    }
  } 
}
TEST(Document, int_svm_light_file) {
  #ifndef ISAGE_LOG_AS_COUT
  boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
  #endif
  std::string path = "test/resources/vocab_list_simple.txt";
  typedef isage::wtm::Vocabulary<int> Vocab;
  Vocab vocab(0);
  for(int i = 0; i < 9; i++) vocab.make_word(i + 1);
  typedef isage::wtm::Document<int, int> Doc;
  {
    std::vector<int> counts = {0, 3, 1, 2, 2, 2, 2, 3, 2, 1};
    Doc doc = Doc::from_svm_light_string("1 1:3 2:1 3:2 4:2 5:2 6:2 7:3 8:2 9:1 # doc_1",
					 vocab);
    ASSERT_EQ("doc_1", doc.id);
    typename Doc::Multinomial mult = doc.multinomial();
    ASSERT_EQ(9, mult.size());
    ASSERT_EQ(18, doc.num_words());
    ASSERT_TRUE(doc.type_view());
    for(const auto& pair : mult) {
      ASSERT_EQ(counts[pair.first], pair.second) << "Counts not equal for " << pair.first;
    }
  }
}
TEST(Document, str_svm_light_file_fractional_values) {
  #ifndef ISAGE_LOG_AS_COUT
  boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
  #endif
  std::string path = "test/resources/vocab_list_simple.txt";
  typedef isage::wtm::Vocabulary<std::string> Vocab;
  Vocab vocab = Vocab::from_file(path, "__OOV__");
  typedef isage::wtm::Document<std::string, double> Doc;
  std::map<std::string, double> counts;
  counts["The"] = 3.9;
  counts["dog"] = 1;
  counts["ran"] = 2.1;
  counts["with"] = 2;
  counts["the"] = 2;
  counts["man"] = 2;
  counts["."] = 3;
  counts["cat"] = 2.5;
  counts["meowed"] = 1;
  {
    Doc doc = Doc::from_svm_light_string("1 1:3.9 2:1 3:2.1 4:2 5:2 6:2 7:3 8:2.5 9:1 # doc_1",
					 vocab);
    ASSERT_EQ("doc_1", doc.id);
    typename Doc::Multinomial mult = doc.multinomial();
    ASSERT_EQ(9, mult.size());
    ASSERT_EQ(19.5, doc.num_words());
    ASSERT_TRUE(doc.type_view());
    for(const auto& pair : mult) {
      ASSERT_EQ(counts[pair.first], pair.second) << "Counts not equal for " << pair.first;
    }
  }
}

TEST(Corpus, svm_light_file) {
  #ifndef ISAGE_LOG_AS_COUT
  boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
  #endif
  typedef isage::wtm::Vocabulary<std::string> Vocab;
  Vocab vocab = Vocab::from_file("test/resources/vocab_list_simple.txt", "__OOV__");
  typedef isage::wtm::Document<std::string> Doc;
  typedef isage::wtm::Corpus<Doc> Corpus;
  Corpus corpus("train", "test/resources/simple_svm_light.txt", vocab);
  ASSERT_EQ(3, corpus.num_docs());
  ASSERT_EQ("doc_1", corpus[0].id);
  ASSERT_EQ("doc_2", corpus[1].id);
  ASSERT_EQ("doc_3", corpus[2].id);
}

