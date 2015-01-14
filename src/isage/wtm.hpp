// Header for word-based topic models

#ifndef ISAGE_WTM_H_
#define ISAGE_WTM_H_

#include "concrete.hpp"
#include "dmc.hpp"
#include "mathops.hpp"
#include "util.hpp"

#include <fstream>
#include <iostream>
#include "stdlib.h"
#include <time.h>

// for pair
#include "map"
#include <utility>
#include <unordered_set>
#include <string>
#include <vector>

#include <fstream>
#include <boost/algorithm/string.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>

#include <gsl/gsl_rng.h>

namespace isage { 
  namespace wtm {
    class StopWordList {
    private:
      typedef std::string string;
      std::unordered_set<string> words;
      void init(const string& fpath, const string& delim) {
	std::ifstream ifile(fpath);
	std::string line;
	std::vector<string> strs;
	if(ifile.is_open()) {
	  while( getline(ifile,line) ) {
	    boost::split(strs, line , boost::is_any_of(delim));
	    string w = strs[0];
	    words.insert(w);
	    strs.clear();
	  }
	  ifile.close();
	} else {
	  BOOST_LOG_TRIVIAL(error) << "StopWordList was unable to open file " << fpath; 
	  throw 19;
	}
      };

    public:
      StopWordList() {
      }
      StopWordList(const string& fpath) {
	init(fpath, "\t");
      };
      inline const bool contains(const string& word) const {
	return words.find(word) != words.end();
      }
      inline const int num_words() const {
	return words.size();
      }
    };

    template <typename W>
    class Vocabulary {
    private:
      int oov_index_;
      std::unordered_map<W, int> word_idx;
      std::vector<W> words;
      bool allow_new_words_;
      friend class boost::serialization::access;
      // When the class Archive corresponds to an output archive, the
      // & operator is defined similar to <<.  Likewise, when the class Archive
      // is a type of input archive the & operator is defined similar to >>.
      template<class Archive>
      void serialize(Archive& ar, const unsigned int version) {
	ar & oov_index_;
	ar & word_idx;
	ar & words;
	ar & allow_new_words_;
      }
    public:
      Vocabulary<W>() : oov_index_(-1), allow_new_words_(true) {
      }
      Vocabulary<W>(const W& oov) : oov_index_(0), allow_new_words_(true) {
	words.push_back(oov);
	word_idx[oov] = 0;
      };
      Vocabulary<W>(const W& oov, bool allow_new) : oov_index_(0), allow_new_words_(allow_new) {
	words.push_back(oov);
	word_idx[oov] = 0;
      };
      int set_oov(const W& oov) {
	if(allow_new_words_) {
	  const int prev_size = words.size();
	  words.push_back(word);
	  word_idx[word] = prev_size;
	  oov_index_ = prev_size;
	  return 0;
	} else {
	  BOOST_LOG_TRIVIAL(error) << "Vocabulary does not allow new words -- NOT adding oov symbol " << oov;
	  return 1;
	}
      }
      void allow_new_words(bool b) {
	allow_new_words_ = b;
      }
      bool allow_new_words() {
	return allow_new_words_;
      }
      void make_word(const W& word) {
	if(allow_new_words_ && word_idx.find(word) == word_idx.end()) {
	  const int prev_size = words.size();
	  words.push_back(word);
	  word_idx[word] = prev_size;
	}    
      }
      inline const W& word(const size_t& idx) const {
	return words.at(idx);
      }
      inline const W& word(const size_t& idx) {
	return words[idx];
      }
      inline const int index(const W& word) {
	return (word_idx.find(word) == word_idx.end()) ? oov_index_ : word_idx[word];
      }
      inline const int index(const W& word) const {
	auto finder = word_idx.find(word);
	return (finder == word_idx.end()) ? oov_index_ : finder->second;
      }

      inline const int num_words() const {
	return words.size();
      }
    };

    template <typename W>
    class WordPruner {
    protected:
      Vocabulary<W>* const vocab_ptr_;
      StopWordList stopwords_;
      bool use_stopwords_;
      bool add_to_vocab_;
    public:
      WordPruner< W >(const concrete::Communication& comm) : use_stopwords_(false), add_to_vocab_(false) {
      }
      WordPruner< W >(Vocabulary<W>* const vp) : vocab_ptr_(vp), use_stopwords_(false), add_to_vocab_(true) {
      }
      virtual const W make_word_view(const std::string& word) const {
	return word;
      }
      virtual std::vector<W> prune(const concrete::Tokenization& tokenization) const {
	std::vector<W> words;
	for(concrete::Token token : tokenization.tokenList.tokenList) {
	  W word = this->make_word_view(token.text);
	  if(add_to_vocab_) {
	    this->vocab_ptr_->make_word(word);
	  }
	  words.push_back( word );
	}
	return words;
      }
      WordPruner<W>& use_stopwords(bool b) {
	use_stopwords_ = b;
	return *this;
      }
      WordPruner<W>& stopwords(const StopWordList& list_) {
	stopwords_ = list_;
	return *this;
      }
    };

    // template <typename W>
    // class VerbPruner : public WordPruner<W> {
    // public:
    //   VerbPruner< W >(const concrete::Communication& comm) : WordPruner<W>(comm) {
    //   }
    //   VerbPruner< W >(Vocabulary<W>* const vp) : WordPruner<W>(vp) {
    //   }
    //   virtual const W make_word_view(const std::string& word) const {
    //     return word;
    //   }
    //   virtual std::vector<W> prune(const concrete::Tokenization& tokenization) const {
    //     std::vector<W> words;
    //     const concrete::TokenTagging* pos = concrete::util::first_pos_tagging(tokenization, "Stanford");
    //     if(pos == NULL) {
    // 	return words;
    //     }
    //     std::vector<concrete::TaggedToken> pos_tags = pos->taggedTokenList;
    //     for(concrete::Token token : tokenization.tokenList.tokenList) {
    // 	if(pos_tags[token.tokenIndex].tag[0] != 'V') continue;
    // 	W word = this->make_word_view(token.text);
    // 	this->vocab_ptr_->make_word(word);
    // 	words.push_back( word );
    //     }
    //     return words;
    //   }
    // };

    template <typename W>
    class Document {
    private:
      std::vector< W > words_;
    public:
      const std::string id;
      Document< W >(const std::string id_) : id(id_) {
      }

      Document< W >(const concrete::Communication& communication,
		    const WordPruner< W >& word_pruner) : id(communication.id) {
	for(concrete::Section section : communication.sectionList) {
	  if(! section.__isset.sentenceList) continue;
	  for(concrete::Sentence sentence : section.sentenceList) {
	    if(! sentence.__isset.tokenization) continue;
	    concrete::Tokenization tokenization = sentence.tokenization;
	    std::vector< W > toks_to_add = word_pruner.prune(tokenization);
	    if(toks_to_add.size() > 0) {
	      words_.insert(words_.end(), toks_to_add.begin(), toks_to_add.end());
	    }
	  }
	}
	BOOST_LOG_TRIVIAL(info) << "Document " << this->id << " has " << num_words() << " words";
      }

      inline const int num_words() const {
	return words_.size();
      }

      inline void add_word(const W& word) {
	words_.push_back(word);
      }

      inline const W& operator[](const size_t idx) const {
	return words_[idx];
      }
    };

    template <typename D> 
    class Corpus {
    private:
      std::vector<D> documents;
      std::string name;
    
    public:
      Corpus<D>() {
      }
      Corpus<D>(std::string corpus_name) : name(corpus_name) {
      }

      void add_document(D& document) {
	documents.push_back(document);
      }

      inline const std::vector<D>& get_corpus() const {
	return documents;
      }
      inline const D& operator[](const size_t idx) const {
	return documents[idx];
      }
      inline const int num_docs() {
	return documents.size();
      }
    };

    class SymmetricHyperparams {
    public:
      double h_theta;
      double h_word;
    };

    //    template <int num_topics>
    struct RandomInitializer {
    private:
      const int num_topics;
      const gsl_rng_type *which_gsl_rng_;
      gsl_rng *rnd_gen_;
    public:
      RandomInitializer(int nt) : num_topics(nt), which_gsl_rng_(gsl_rng_mt19937), rnd_gen_(gsl_rng_alloc(which_gsl_rng_)) {
      }
      int operator()(int doc_id, int word_id) {
	return gsl_rng_uniform_int(rnd_gen_, num_topics);
      }
    };

    /**
     * Vanilla LDA: observations are generated by discrete 
     * distributions with Dirichlet priors.
     * W: observation type
     * T: topic type
     */
    template <typename W, typename T = std::vector<double> >
    class DiscreteLDA {
    private:
      typedef std::vector<double> hvector;
      // setting model parameters
      const int num_topics_;

      //// hyperparameters
      // how to use each template
      hvector hyper_theta_;
      // how often words are used
      hvector hyper_word_;

      //// priors
      // Each document has a prior probability of using
      // any particular template
      std::vector< std::vector<double> > prior_topic_;
      // Each template has a prior probability of generating 
      // any particular governor
      std::vector< T > prior_word_;

      bool h_word_set_ = false;

    public:
      DiscreteLDA<W, T>(int n_topic, SymmetricHyperparams* shp,
			Vocabulary<W> *vocabs) : num_topics_(n_topic), 
	hyper_theta_(hvector(num_topics_, shp->h_theta)), hyper_word_(hvector(vocabs->num_words(), shp->h_word)), h_word_set_(true) {
	// fill_hyper(hyper_theta_, shp->h_theta, num_topics_);
	// fill_hyper(hyper_word_, shp->h_word, vocabs->num_words());
      }
      ~DiscreteLDA() {
      }
      inline const int num_topics() {
	return num_topics_;
      }
      inline const hvector hyper_word() {
	return hyper_word_;
      }
      inline const hvector hyper_theta() {
	return hyper_theta_;
      }
    
      //transfer functions
      inline void prior_topic(const std::vector< std::vector< double > >& posterior_topic) {
	prior_topic_ = posterior_topic;
      }
      inline void prior_word(const std::vector< T >& posterior_word) {
	prior_word_ = posterior_word;
      }

      inline const std::vector< std::vector< double> >& prior_topic() {
	return prior_topic_;
      }
      inline const std::vector< T >& prior_word() {
	if(! h_word_set_) {
	  BOOST_LOG_TRIVIAL(error) << "Hyperparameters for topics are not set";
	}
	return prior_word_;
      }

      void hyper_word(const hvector& hw) {
	hyper_word_ = hw;
	h_word_set_ = true;
      }
      void hyper_theta(const hvector& ht) {
	hyper_theta_ = ht;
      }
      DiscreteLDA<W,T>& hyper_word(double num, double val) {
	hyper_word_ = hvector(num, val);
	h_word_set_ = true;
	return *this;
      }

      void print_topics(const int num_per, const Vocabulary<W>& vocab) {
	int topic_idx = 0;
	for(const T topic : prior_word_) {
	  BOOST_LOG_TRIVIAL(info) << "Topic " << topic_idx;
	  std::vector<size_t> sorted_topic = isage::util::sort_indices(topic, false);
	  for(size_t item_idx = 0; item_idx < num_per; ++item_idx) {
	    size_t which = sorted_topic[item_idx];
	    BOOST_LOG_TRIVIAL(info) << "\t" << topic[which] << "\t" << vocab.word(which);
	  }
	  ++topic_idx;
	}
      }
    };
  } 
}

#endif // ISAGE_WTM_H_
