// Header for word-based topic models

#ifndef ISAGE_WTM_H_
#define ISAGE_WTM_H_

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
#include <ostream>
#include <sstream>
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
	  ERROR << "StopWordList was unable to open file " << fpath; 
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
      /**
       * This constructs a vocab object from a provided list of words.
       * Each "word" must be on its own line (white-spaces are not allowed).
       * OOV is set to 0.
       */
      static Vocabulary<W> from_file(const std::string& vocab_list_path, const W& oov) {
	Vocabulary<W> vocab(oov);
	std::ifstream ifile(vocab_list_path);
	std::string line;
	if(ifile.is_open()) {
	  while( getline(ifile,line) ) {
	    boost::trim(line);
	    if(line.size()) {
	      vocab.make_word(line);
	    }
	  }
	  ifile.close();
	} else {
	  ERROR << "Vocabulary was unable to open file " << vocab_list_path; 
	  throw 19;
	}
	vocab.allow_new_words(false);
	return vocab;
      };
      int set_oov(const W& oov) {
	if(allow_new_words_) {
	  const int prev_size = words.size();
	  words.push_back(oov);
	  word_idx[oov] = prev_size;
	  oov_index_ = prev_size;
	  return 0;
	} else {
	  ERROR << "Vocabulary does not allow new words -- NOT adding oov symbol " << oov;
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
      //template <typename V>
      template <typename V, typename W2V>
      inline const V& word_as(const size_t& idx) const {
	return W2V(words.at(idx));
      }

      inline const int index(const W& word) {
	return (word_idx.find(word) == word_idx.end()) ? oov_index_ : word_idx[word];
      }
      inline const int index(const W& word) const {
	auto finder = word_idx.find(word);
	return (finder == word_idx.end()) ? oov_index_ : finder->second;
      }

      inline const size_t num_words() const {
	return words.size();
      }

      inline const std::vector<W>& word_list() const {
	return words;
      }
    };

    template <typename W, typename MCount = int>
    class Document {
    public:
      typedef std::map< W, MCount > Multinomial;
      typedef int Label;
      typedef MCount WordCountType;
      std::string id;
      Document<W, MCount>() : num_words_(0) {
      }
      Document< W, MCount >(const std::string id_) : id(id_), num_words_(0), label_(0) {
      }

      //template <typename V, typename V2W>
      static Document<W, MCount> from_svm_light_string(const std::string& svm_light_str,
						       const Vocabulary<W>& vocab) {
	std::list<std::string> strs;
	auto comment_it = svm_light_str.find('#');
	std::string not_comment_str;
	std::string comment_str;
	if(comment_it != std::string::npos) {
	  not_comment_str = svm_light_str.substr(0, comment_it);
	  comment_str = svm_light_str.substr(comment_it+1, svm_light_str.size());
	  boost::trim(comment_str);
	} else {
	  not_comment_str = svm_light_str;
	}
	boost::split(strs, not_comment_str, boost::is_any_of(" "));
	if(!strs.size()) {
	  ERROR << "could not process svm-light-like document " << svm_light_str;
	  throw 2;
	}
	std::list<std::string>::iterator it = strs.begin();
	Label label = isage::util::str_to_value<Label>(*it);
	++it;
	Document<W, MCount> doc;
	doc.type_view(true);
	doc.label(label);
	for(; it != strs.end(); ++it) {
	  DEBUG << (*it);
	  auto separator_it = it->find(':');
	  if(separator_it != std::string::npos) {
	    int ft = std::stoi(it->substr(0, separator_it));
	    MCount val = isage::util::str_num_value<MCount>(it->substr(separator_it+1, it->size()));
	    TRACE << "ft = " << ft << ", val = " << val;
	    W w_word = vocab.word(ft);
	    //W w_word = vocab.template word_as< W, V2W >(ft);
	    //W w_word = V2W(vocab.word(ft));
	    doc.multinomial_increase(w_word, val);
	  } 
	}
	if(comment_str.size()) {
	  doc.id = comment_str;
	} else {
	  doc.id = "UnknownDocId";
	}
	return doc;
      }

      inline const MCount num_words() const {
	return num_words_;
      }
      inline void add_word(const W& word) {
	multinomial_[word] += 1;
	num_words_++;
	if(!type_view_) {
	  words_.push_back(word);
	}
      }
      inline void multinomial_increase(const W& word, const MCount& count) {
	multinomial_[word] += count;
	num_words_ += count;
	if(!type_view_){
	  for(int i = 0; i < count; ++i) {
	    words_.push_back(word);
	  }
	}
      }
      inline const Multinomial& multinomial() const {
	return multinomial_;
      }
      inline const W& operator[](const size_t idx) const {
	if(type_view_) {
	  ERROR << "Cannot use operator[] when document is in type_view";
	  throw 2;
	}
	return words_[idx];
      }
      void type_view(bool b) {
	if(num_words_ > 0) {
	  ERROR << "Cannot set type_view after adding words";
	}
	type_view_ = b;
      }
      bool type_view() {
	return type_view_;
      }
      void label(Label label) {
	label_ = label;
      }
      Label label() {
	return label_;
      }

    private:
      std::vector< W > words_;
      Multinomial multinomial_;
      MCount num_words_;
      Label label_;
      bool type_view_;
    };

    template <typename D> 
    class Corpus {
    private:
      typedef std::vector<D> Container;
      Container documents;
      std::string name;
    
    public:
      typedef D document_type;
      Corpus<D>() {
      }
      Corpus<D>(std::string corpus_name) : name(corpus_name) {
      }
      /**
       * Read in an SVM-light style corpus: each document is on its own line
       * Note that features start 1
       *
       * <line> .=. <target> [SPACE] (<feature>:<value>[SPACE])+ #[SPACE]?<info>
       * <target> .=. +1 | -1 | 0 | <float> 
       * <feature> .=. <integer> | "qid"
       * <value> .=. <float>
       * <info> .=. <string>
       */
      //template <typename V, typename V2W> 
      template <typename W>
      inline Corpus< D >(const std::string& corpus_name, 
			 const std::string& svm_light_style_path,
			 const Vocabulary<W>& vocab) : name(corpus_name) {
      	std::ifstream ifile(svm_light_style_path);
      	std::string line;
      	std::list<std::string> strs;
      	if(ifile.is_open()) {
      	  while( getline(ifile,line) ) {
	    boost::trim(line);
	    if(!line.size()) continue;
	    //D doc = D::template from_svm_light_string<V,V2W>(line, vocab);
	    D doc = D::from_svm_light_string(line, vocab);
	    add_document(doc);
      	    strs.clear();
      	  }
      	  ifile.close();
      	} else {
      	  ERROR << "StopWordList was unable to open file " << svm_light_style_path; 
      	  throw 19;
      	}
      }

      template <typename W>
      void generate(const int num_docs, const int num_words_per_doc,
		    const double vocab_bias, const Vocabulary<W>& vocab) {
	// these partition the vocab: note that
	// larger_v_span + smaller_v_span = vocab size
	const std::vector<W> words = vocab.word_list();
	const int larger_v_span = words.size() * .5;
	const int smaller_v_span = words.size() - larger_v_span;
	// how many words do you take from each partition
	// these values are effectively swapped halfway through
	int larger_num =  num_words_per_doc * vocab_bias;
	int smaller_num = num_words_per_doc - larger_num;
	// further, construct receptacles for the values
	// as above, (pointers) to these are swapped halfway through
	unsigned int* const larger_receptacle = new unsigned int[larger_v_span];
	unsigned int* const smaller_receptacle = new unsigned int[smaller_v_span];
	// construct a uniform distribution for each partition
	std::vector<double> larger_p(larger_v_span, 1.0/(double)(larger_v_span));
	std::vector<double> smaller_p(smaller_v_span, 1.0/(double)(smaller_v_span));
	std::vector<double>* const left_draw = &larger_p;
	std::vector<double>* const right_draw = &smaller_p;
	int left_doc_draw = larger_num; 
	int right_doc_draw = smaller_num;
	// the "left half" (first half) of documents always draws from the 
	// "larger" vocab span, while the "right" half (second half) always
	// draws from the "smaller" vocab span
	const int left_span = larger_v_span; const int right_span = smaller_v_span;
	unsigned int* const left_receptacle = larger_receptacle;
	unsigned int* const right_receptacle = smaller_receptacle;
	for(int di = 0; di < num_docs; ++di) {
	  // at the halfway mark, swap the "pointers"
	  if(di == (int)(num_docs/2)) {
	    left_doc_draw = smaller_num; 
	    right_doc_draw = larger_num;
	  }
	  D doc("doc_" + std::to_string(di));
	  doc.type_view(true);
	  std::stringstream dstream;
	  dstream << doc.id << " :: ";
	  // first draw from the left half of the vocab
	  for(int i = 0; i < left_span; ++i) {
	    left_receptacle[i] = 0;
	  }
	  for(int i = 0; i < right_span; ++i) {
	    right_receptacle[i] = 0;
	  }
	  gsl_ran_multinomial(mathops::rnd_gen, left_span, left_doc_draw,
			      left_draw->data(), left_receptacle);
	  unsigned int lsum = 0;
	  for(int i = 0; i < left_span; ++i) {
	    lsum += left_receptacle[i];
	    if(left_receptacle[i] > 0) {
	      doc.multinomial_increase(words[i], left_receptacle[i]);
	      dstream << words[i] << ":" << left_receptacle[i] << " ";
	    }
	  }
	  // then do the same for the right half, but remember to add the offset
	  gsl_ran_multinomial(mathops::rnd_gen, right_span, right_doc_draw,
			      right_draw->data(), right_receptacle);
	  unsigned int rsum = 0;
	  for(int i = 0; i < right_span; ++i) {
	    rsum += right_receptacle[i];
	    if(right_receptacle[i] > 0) {
	      DEBUG << "i = " << i << ", right_span = " << right_span << " vs. right_doc_draw = " << right_doc_draw;
	      doc.multinomial_increase(words[i + left_span], right_receptacle[i]);
	      dstream << words[i+left_span] << ":" << right_receptacle[i] << " ";
	    }
	  }
	  DEBUG << dstream.str() << "; left sum = " << lsum << ", right sum = " << rsum;
	  this->add_document(doc);
	}
	delete[] larger_receptacle;
	delete[] smaller_receptacle;
      }

      typename Container::const_iterator begin() const {
	return documents.begin();
      }
      typename Container::const_iterator end() const {
	return documents.end();
      }

      void add_document(D& document) {
	documents.push_back(document);
      }

      inline const Container& get_corpus() const {
	return documents;
      }
      inline const D& operator[](const size_t idx) const {
	return documents[idx];
      }
      inline const int num_docs() {
	return documents.size();
      }
      inline const int num_docs() const {
	return documents.size();
      }

      template <typename W>
      inline std::map< std::pair<int, int>, int> word_doc_cooccur(const Vocabulary<W>& vocab) const {
	std::map< std::pair<int, int>, int> doc_occurs;
	for(const D& doc : documents) {
	  auto doc_verb_idf = doc.multinomial();
	  std::vector<int> seen;
	  for(const auto& entry : doc_verb_idf) {
	    seen.push_back(vocab.index(entry.first));
	  }
	  const int num_types_seen = seen.size();
	  for(int i = 0; i < num_types_seen; ++i) {
	    for(int j = 0; j < i; ++j) {
	      const int w_i = seen[i];
	      const int w_j = seen[j];
	      doc_occurs[ std::pair<int, int>(w_i, w_j) ] += 1;
	    }
	  }
	}
	return doc_occurs;
      }
      template <typename W>
      inline std::map<int, int> word_doc_occur(const Vocabulary<W>& vocab) const {
	std::map<int, int> doc_occurs;
	for(const D& doc : documents) {
	  auto doc_verb_idf = doc.multinomial();
	  for(const auto& entry : doc_verb_idf) {
	    doc_occurs[ vocab.index(entry.first) ] += 1;
	  }
	}
	return doc_occurs;
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
      DiscreteLDA<W, T>(int n_topic) : num_topics_(n_topic) {
      }
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
	  ERROR << "Hyperparameters for topics are not set";
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

      void print_usage(std::ostream & outter = std::cout, 
		       bool sorted = false) {
	for(const auto& doc_usage : prior_topic_) {
	  std::stringstream stream;
	  if(sorted) {
	    
	  } else {
	    for(const auto& p : doc_usage) {
	      stream << p << " ";
	    }
	    outter << stream.str();
	    outter << std::endl;
	  }
	}
      }

      void print_topics(const size_t num_per, const Vocabulary<W>& vocab, 
			const std::vector< T >& topics) {
	int topic_idx = 0;
	for(const T topic : topics) {
	  std::stringstream stream;
	  stream << "Topic " << topic_idx << " ::: ";
	  std::vector<size_t> sorted_topic = isage::util::sort_indices(topic, false);
	  for(size_t item_idx = 0; item_idx < num_per && item_idx < vocab.num_words(); ++item_idx) {
	    size_t which = sorted_topic[item_idx];
	    stream << vocab.word(which) << " [" << topic[which] << "] ";
	  }
	  INFO << stream.str();
	  ++topic_idx;
	}
      }

      void print_topics(std::ostream& stream, const Vocabulary<W>& vocab, 
			const std::vector< T >& topics) {
	for(const T topic : topics) {
	  for(const auto& val : topic) {
	    stream << val << " ";
	  }
	  stream << "\n";
	}
      }
      void print_topics(std::ostream& stream, const Vocabulary<W>& vocab) {
	print_topics(stream, vocab, prior_word_);
      }

      void print_topics(const int num_per, const Vocabulary<W>& vocab) {
	print_topics(num_per, vocab, prior_word_);
      }

      T* topic(const size_t& topic_id) {
	return &(prior_word_[topic_id]);
      }

      template <typename D>
      static std::vector<double> compute_coherences(const int M,
						    const Corpus<D>& corpus,
						    const Vocabulary<W>& vocab,
						    const std::vector< T > topics) {
	std::map<int, int> w_occur = 
	  corpus.template word_doc_occur< W >(vocab);
	std::map<std::pair<int, int>, int> w_cooc = 
	  corpus.template word_doc_cooccur< W >(vocab);
	return isage::util::compute_coherences(M, topics, w_occur, w_cooc);
      }

      template <typename D>
      std::vector<double> compute_coherences(const int M,
					     const Corpus<D>& corpus,
					     const Vocabulary<W>& vocab) {
	return DiscreteLDA<W, T>::compute_coherences<D>(M, corpus, vocab, prior_word_);
      }
    };
  } 
}

#endif // ISAGE_WTM_H_
