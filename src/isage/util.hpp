#ifndef ISAGE_UTIL_H_
#define ISAGE_UTIL_H_

#include "logging.hpp"
#include <gsl/gsl_sf_log.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <map>
#include <math.h>
#include <sstream>
#include <type_traits> // for std::is_const
#include <unordered_map>
#include <utility> // for std::pair
#include <vector>

namespace isage { namespace util {

    class SmartWriter {
    private:
      std::string base_;
      const bool console_;
      std::ofstream* f_ptr = NULL;
      std::string curr_file;
    public:
      SmartWriter();
      SmartWriter(const std::string& fname);
      ~SmartWriter();
      std::ostream& get();
      std::ostream& get(const int i);
      std::ostream& get(const std::string& suffix);
      std::string base_name();
      std::string name();
      bool to_file();
    };

    /**
     * Print various stats of the process
     */
    void print_pstats();

    template <typename T>
    std::vector<T> zeros(int size_) {
      std::vector<T> zero(size_, 0);
      return zero;
    }

    template <typename T>
    class VectorSortIndex {
    private:
      const std::vector<T>* vp_;
      int mult_;
    public:
      VectorSortIndex(const std::vector<T>* v, int mult) : vp_(v), mult_(mult) {
      }
      bool operator() (const size_t& i, const size_t& j) { 
	return (mult_ * (*vp_)[i]) < (mult_ * (*vp_)[j]);
      }
    };

    template <typename T>
    std::vector<size_t> sort_indices(const std::vector<T> &v, bool ascending) {
      const int mult = ascending ? 1 : -1;
      // initialize original index locations
      std::vector<size_t> idx(v.size());
      for(size_t i = 0; i != idx.size(); ++i) {
	idx[i] = i;
      }

      VectorSortIndex<T> vsi(&v, mult);
      // sort indexes based on comparing values in v
      std::sort(idx.begin(), idx.end(), vsi);

      return idx;
    }


    /**
     * Find the average in a vector.
     */
    template <typename T>
    inline T average(const std::vector< T >& counts) {
      T rsum = 0;
      for(const T& elem : counts) {
	rsum += elem;
      }
      return counts.size() ? (rsum / (T)counts.size()) : 0.0;
    };

    template <typename T>
    inline T dist(const std::vector<T>& x, const std::vector<T>& y) {
      if(x.size() != y.size()) {
	ERROR << "Sizes not equal: " << x.size() << " vs. " << y.size();
	throw 2;
      }
      double d = 0.0;
      for(size_t i = 0; i < (size_t)x.size(); ++i) {
	double diff = x[i] - y[i];
	d += (diff * diff);
      }
      return sqrt(d);
    }

    inline std::vector<double> compute_coherences(const int M,
						  const std::vector< std::vector<double> >& probs,
						  const std::map<int, int>& single_occur,
						  const std::map< std::pair<int, int>, int >& double_occur) {
      std::vector<double> coherences;
      const int num_dists = probs.size();
      for(int dist_idx = 0; dist_idx < num_dists; ++dist_idx) {
	const std::vector<double>& curr_probs = probs[dist_idx];
	const std::vector<size_t> sorted_topic = isage::util::sort_indices(curr_probs, false);
	double run_coher = 0.0;
	for(int m = 1; m < M; ++m) { 
	  const int word_at_m = sorted_topic[m];
	  for(int l = 0; l < m - 1; ++l) {
	    const int word_at_l = sorted_topic[l];
	    std::pair<int,int> p( word_at_m, word_at_l );
	    double num = 1.0;
	    if(! double_occur.count(p)) {
	      p.first = word_at_l;
	      p.second = word_at_m;
	      if( !double_occur.count(p) ){
		// do nothing
	      } else {
		double x = double_occur.at(p);
		num += x;
	      }
	    } else {
	      double x = double_occur.at(p);
	      num += x;
	    }
	    //double denom = (double)( single_occur.count(word_at_l) ? single_occur.at(word_at_l) : 1E-9);
	    double denom = (double)( single_occur.count(word_at_l) ? single_occur.at(word_at_l) : 0);
	    run_coher += gsl_sf_log( num / denom );
	  }
	}
	coherences.push_back(run_coher);
      }
      return coherences;
    }

    template <typename T>
    inline std::vector<T> sum(const std::vector< T >& x, const std::vector< T >& y) {
      if(x.size() != y.size()) {
	throw 1;
      }
      const size_t dim = x.size();
      std::vector<T> res(dim);
      for(size_t idx = 0; idx < dim; ++idx) {
	res[idx] = x[idx] + y[idx];
      }
      return res;
    };

    template <typename T>
    inline void sum(const T& x, std::vector< T >* y) {
      const size_t dim = y->size();
      for(size_t idx = 0; idx < dim; ++idx) {
	(*y)[idx] += x;
      }
    };

    /**
     * Compute x = ax + by, where x, y are vectors and a, b are 
     * scalars.
     */
    template <typename T>
    inline void linear_combination_in_first(std::vector< T >* const x, const std::vector< T >& y, const T a, const T b) {
      const size_t dim = x->size();
      if(dim != y.size()) {
    	throw 1;
      }
      for(size_t idx = 0; idx < dim; ++idx) {
	T val = a * (x->operator[](idx));
	x->operator[](idx) = val + (b * y[idx]);
      }
    };

    template <typename T>
    inline void sum_in_first(std::vector< T >* const x, const std::vector< T >& y) {
      const size_t dim = x->size();
      if(dim != y.size()) {
    	throw 1;
      }
      for(size_t idx = 0; idx < dim; ++idx) {
    	x->operator[](idx) += y[idx];
      }
    };

    template <typename T>
    inline std::vector<T> sum(const T& x, const std::vector< T >& y) {
      const size_t dim = y.size();
      std::vector<T> res(dim, x);
      sum_in_first(&res, y);
      return res;
    };

    template <typename T>
    inline void ensure_min(const T& val, std::vector< T >* const x) {
      const size_t dim = x->size();
      for(size_t idx = 0; idx < dim; ++idx) {
	const T& xval = (*x)[idx];
	if(xval < val) {
	  (*x)[idx] = val;
	}
      }
    };

    template <typename T>
    inline std::vector<T> column(const std::vector<std::vector<T> >& matrix, const size_t& col_idx) {
      std::vector<T> slice;
      for(const std::vector<T>& row : matrix) {
	slice.push_back(row[col_idx]);
      }
      return slice;
    }

    template <typename T>
    inline bool vectors_equal(const std::vector<T>& x, const std::vector<T>& y) {
      bool eq = true;
      if(x.size() != y.size()) {
	ERROR << "comparing containers of different sizes";
	return false;
      }
      const size_t size = x.size();
      for(size_t i = 0; i < size; ++i) {
	eq &= (x[i] == y[i]);
	if(!eq) {
	  return false;
	}
      }
      return true;
    }

    template <typename T>
    inline void product(const T& val, std::vector< T >* const x) {
      const int dim = x->size();
      for(int idx = 0; idx < dim; ++idx) {
    	(*x)[idx] *= val;
      }
    };

    template <typename V>
    inline void zero_out(V* const x) {
      typename V::iterator it = x->begin();
      for(;it != x->end(); ++it) {
	*it = 0.0;
      }
    };

    // template <typename V>
    // inline void exp(V* const x) {
    //   //typedef typename V::value_type T;
    //   typename V::iterator it = x->begin();
    //   // if(std::is_const< decltype(it) >::value) {
    //   // 	for(;it != x->end(); ++it) {
    //   // 	  T temp = *it;
    //   // 	  x->erase(it);
    //   // 	  x->insert(it, gsl_sf_exp(temp));
    //   // 	}
    //   // } else {
    //   for(;it != x->end(); ++it) {
    // 	*it = mathops::exp(*it);
    //   }
    //   //      }
    // };
    // template <typename V>
    // inline void log(V* const x) {
    //   for(typename V::iterator it = x->begin();
    // 	  it != x->end(); ++it) {
    // 	*it = gsl_sf_log(*it);
    //   }
    // };
    // template <typename V>
    // inline V exp(const V& x) {
    //   V res(x);
    //   exp(&res);
    //   return res;
    // };
    // template <typename V>
    // inline V log(const V& x) {
    //   V res(x);
    //   log(&res);
    //   return res;
    // };

    template <typename V>
    inline void square(V* const x) {
      typedef typename V::value_type T;
      typename V::iterator it = x->begin();
      for(;it != x->end(); ++it) {
	T val = *it;
	*it = val * val;
      }
    };
    template <typename V>
    inline V square(const V& x) {
      V res(x);
      square(&res);
      return res;
    };
    template <typename V>
    inline void cube(V* const x) {
      typedef typename V::value_type T;
      typename V::iterator it = x->begin();
      for(;it != x->end(); ++it) {
	T val = *it;
	*it = val * val * val;
      }
    };
    template <typename V>
    inline V cube(const V& x) {
      V res(x);
      cube(&res);
      return res;
    };
    template <typename V>
    inline void quartic(V* const x) {
      typedef typename V::value_type T;
      typename V::iterator it = x->begin();
      for(;it != x->end(); ++it) {
	const T val = *it;
	*it = val * val * val * val;
      }
    };
    template <typename V>
    inline V quartic(const V& x) {
      V res(x);
      quartic(&res);
      return res;
    };

    template <typename T, typename U = T>
    inline std::vector<T> scalar_product(const U& x, const std::vector< T >& y) {
      const size_t dim = y.size();
      std::vector<T> res(dim);
      for(size_t idx = 0; idx < dim; ++idx) {
	res[idx] = x * y[idx];
      }
      return res;
    };
    template <typename T, typename U = T>
    inline void scalar_product(const U& x, std::vector< T >* y) {
      const size_t dim = y->size();
      for(size_t idx = 0; idx < dim; ++idx) {
	y->operator[](idx) *= x;
      }
    };


    /**
     * Find the maximum in an M-length vector of elements.
     */
    template <typename V>
    inline typename V::value_type max(const V& counts) {
      typedef typename V::value_type T;
      T max_num = 0;
      for(const T& elem_count : counts) {
	if(elem_count > max_num) {
	  max_num = elem_count;
	}
      }
      return max_num;
    };

    /**
     * Find the maximum in an MxN matrix of elements.
     */
    template <typename T>
    inline T max(const std::vector< std::vector< T > >& counts) {
      T max_num = 0;
      for(const std::vector<T>& row_counts : counts) {
	T r_max = max(row_counts);
	if(r_max > max_num) {
	  max_num = r_max;
	}
      }
      return max_num;
    };

    /**
     * Find the maximum in an MxN matrix of elements.
     */
    template <typename T>
    inline T max(const std::vector<std::vector< std::vector< T > > >& counts) {
      T max_num = 0;
      for(const std::vector<std::vector<T> >& row_counts : counts) {
	T r_max = max<T>(row_counts);
	if(r_max > max_num) {
	  max_num = r_max;
	}
      }
      return max_num;
    };

    /**
     * Find the maximum in an MxN matrix of elements.
     */
    template <typename T>
    inline std::vector<T> column_max(const std::vector< std::vector< T > >& counts) {
      const int K = counts[0].size();
      std::vector<T> maxes(K);
      for(const std::vector<T>& row_counts : counts) {
	for(int k = 0; k < K; ++k) {
	  if(row_counts[k] > maxes[k]) {
	    maxes[k] = row_counts[k];
	  }
	}
      }
      return maxes;
    };

    /**
     * Find the maximum in an MxN matrix of elements.
     */
    template <typename T>
    inline std::vector<T> column_max(const std::vector<std::vector< std::vector< T > > >& counts) {
      const int K = counts[0][0].size();
      std::vector<T> maxes(K);
      for(const std::vector<std::vector<T> >& row_counts : counts) {
	std::vector<T> m = column_max<T>(row_counts);
	for(int k = 0; k < K; ++k) {
	  if(m[k] > maxes[k]) {
	    maxes[k] = m[k];
	  }
	}
      }
      return maxes;
    };

    /**
     * Given a count matrix of D x K, return a histogram
     * recording, for each k <= K, the number of times k appeared.
     */
    inline std::vector<std::vector<int> > histogram(const std::vector<std::vector< int > >& counts) {
      const int K = counts[0].size();
      const int num_rows = counts.size();
      const int matrix_max = isage::util::max(counts);
      std::vector< std::vector< int > > hist(K, std::vector<int>(matrix_max + 1, 0));
      for(int k = 0; k < K; ++k) {
	for(int i = 0; i < num_rows; ++i) {
	  ++hist[k][ counts[i][k] ];
	}
      }
      return hist;
    };

    /**
     * Given a count matrix of D x K, return a histogram
     * recording, for each k <= K, the number of times k appeared.
     */
    inline std::vector<std::vector<int> > mf_histogram(const std::vector< std::vector<std::vector< int > > >& counts) {
      const int K = counts[0][0].size();
      const int num_rows = counts.size();
      const int num_cols = counts[0].size();
      const int matrix_max = isage::util::max(counts);
      std::vector< std::vector< int > > hist(K, std::vector<int>(matrix_max + 1, 0));
      for(int k = 0; k < K; ++k) {
	for(int i = 0; i < num_rows; ++i) {
	  for(int j = 0; j < num_cols; ++j) {
	    ++hist[k][ counts[i][j][k] ];
	  }
	}
      }
      return hist;
    };

    template <typename V> 
    inline typename V::value_type sum(const V& counts) {
      typedef typename V::value_type T;
      T sum = 0;
      for(const T& x : counts) {
	sum += x;
      }
      return sum;
    }

    inline std::map<int,int> sparse_histogram(const std::vector< int >& counts) {
      std::map<int, int> hist_;
      for(const int x : counts) {
	++hist_[x];
      }
      return hist_;
    }

    inline std::vector<int> marginals(const std::vector<std::vector< int > >& counts) {
      const int num_rows = counts.size();
      std::vector<int> marginals_(num_rows, 0);
      int i = 0;
      for(const std::vector<int>& row_counts : counts) {
	marginals_[i++] = isage::util::sum(row_counts);
      }
      return marginals_;
    }

    inline std::vector<int> marginals(const std::vector<std::vector<std::vector< int > > >& counts) {
      std::vector<int> marginals_;
      for(const std::vector<std::vector<int> >& row_counts : counts) {
	for(const std::vector<int>& col_counts : row_counts) {
	  marginals_.push_back(isage::util::sum(col_counts));
	}
      }
      return marginals_;
    }

    /**
     * 
     */
    inline std::vector<int> marginal_histogram(const std::vector<std::vector< int > >& counts) {
      const std::vector<int> marginals_ = isage::util::marginals(counts);
      const int max_marginal = isage::util::max(marginals_);
      std::vector<int> lengths(max_marginal + 1, 0);
      for(const int marg : marginals_) {
	++lengths[marg];
      }
      return lengths;
    };

    /**
     * Find the Frobenius norm of a (dense) matrix. 
     * The matrix cannot be ragged.
     */
    template <typename T>
    inline double frobenius_norm(const std::vector< std::vector< T > >& matrix) {
      double fn = 0.0;
      for(const auto& row : matrix) {
	for(const auto& elem : row) {
	  fn += (elem * elem);
	}
      }
      return sqrt(fn);
    };


    template <typename T>
    class pair_hash { 
    public:
      const size_t operator()(const std::pair<T, T>& pair_) const {
	return std::hash<T>()(pair_.first) ^ std::hash<T>()(pair_.second);
      }
    };

    template <typename K, typename V > using pair_map =
      std::unordered_map< const std::pair< K, K >, V , isage::util::pair_hash<K> >;
    template <typename K> using pair_icount =
      std::unordered_map< const std::pair< K, K >, int, isage::util::pair_hash<K> >;

    template <typename V>
    inline void print_1d(const V& vec) {
      for(const auto& elem: vec) {
	std::cout << std::setprecision(8) << elem << " ";
	//	printf("%.8f ", elem);
      }
      //printf("\n");
      std::cout << std::endl;
    }
    template <typename T>
    inline void print_2d(const std::vector<std::vector<T> >& vec) {
      for(const std::vector<T>& rvec : vec) {
	for(const T& elem: rvec) {
	  std::cout << elem << ' ';
	}
	std::cout << std::endl;
      }
    }

    template <typename F>
    inline F str_num_value(const std::string& str) {
      F res;
      std::istringstream convert(str);
      if( ! (convert >> res)) {
	ERROR << "Could not parse string " << str << " into a numeric type";
	res = (F)0.0;
      }
      return res;
    }
    template <typename F>
    inline F str_to_value(const std::string& str) {
      F res;
      std::istringstream convert(str);
      if( ! (convert >> res)) {
	ERROR << "Could not parse string " << str << " into a numeric type";
	throw 2;
      }
      return res;
    }

    template <typename T>
    inline void check_positive(const std::vector< T >& vec) {
      int which = 0;
      for(const T& elem : vec) {
	if(elem <= 0) {
	  ERROR << "Element[" << which << "] = " << elem << " is not positive";
	}
	which++;
      }
    }

    template <typename T>
    inline void check_positive(const std::vector<std::vector<T> >& vec) {
      for(const std::vector<T>& rvec : vec) {
	check_positive<T>(rvec);
      }
    }

    template <typename T>
    inline void check_positive(const std::vector<std::vector<std::vector<T> > >& vec) {
      for(const std::vector< std::vector<T > >& rvec : vec) {
	check_positive<T>(rvec);
      }
    }

    inline void* MALLOC(size_t size) {
      void* ptr = malloc(size);
      if(ptr == NULL) {
	throw 5;
      }
      return ptr;
    }

    template <class T> inline void fill_array1d(T* array, T value,
						const int size) {
      DEBUG << "filling " << array << " with " << size << " copies of " << value;
      for(int i = 0; i < size; i++) {
	array[i] = value;
      }
    }

    template <typename X> inline void allocate_1d(X* &array, const int size) {
      DEBUG << "making 1D array @ " << &array << " of size " << size;
      array = (X*)MALLOC(sizeof(X) * size);
    }

    template <typename X> inline void allocate_2d(X** &array, const int num_rows, const int* num_cols) {
      DEBUG << "making 2D array @ " << &array << " of row size " << num_rows;
      //array = (X**)MALLOC(sizeof(X*) * num_rows);
      allocate_1d<X*>(array, num_rows);
      for(int i = 0; i < num_rows; i++) {
	DEBUG << "\trow " << i << ", num cols: " << num_cols[i];
	DEBUG << " @ " << array[i];
	DEBUG << *(array+i);
	allocate_1d<X>(array[i], num_cols[i]);
      }
      DEBUG << "done";
    }

    template <typename X> inline const X** transfer_2d(const X** into,
						       X** source) {
      const int r_len = sizeof(source)/sizeof(X*);
      into = (const X**)malloc(sizeof(X*) *  r_len);
      for(int r = 0; r < r_len; r++) {
	const int c_len = sizeof(into[r])/sizeof(X);
	into[r] = (const X*)malloc(sizeof(X) * c_len);
	for(int c = 0; c < c_len; c++) {
	  into[r][c] = source[r][c];
	}
      }
      return into;
    }
    template <typename X> inline const X*& transfer_1d(X* source) {
      X* into;
      const int r_len = sizeof(source)/sizeof(X);
      DEBUG << "For transfer: allocating array @ " << &into << " of size " << r_len;
      allocate_1d(into, r_len);
      for(int r = 0; r < r_len; r++) {
	into[r] = source[r];
      }
      return into;
    }

    template <typename X> inline X* make_1d(const int size, const X value) {
      X* into;
      allocate_1d(into, size);
      fill_array1d<X>(into, value, size);
      return into;
    }


    // Inline functions

    template <typename X> inline X& get_1d_ref(X arr) {
      return &arr;
    }

    /////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////

    template <typename T> class Vector {
    protected:
      std::vector< T > array;
    public:
      Vector<T>() {
      }
      T& operator[](std::size_t idx) {
	return array[idx];
      }
      void add_row() {
	array.push_back( T() );
      }
      void add_row(const int size, T val) {
	const int curr_row = array.size();
	array[curr_row].push_back(val);
      }
    };

    template <typename T> 
    class Vector2D {
    private:
      std::vector< std::vector<T> > array;
    public:
      Vector2D<T>() {
	DEBUG << "calling vector2d constructor";
      }
    
      // virtual ~Vector2D() {
      //   DEBUG << "calling vector2d destructor" << std::endl;
      // }

      std::vector<T>& operator[](std::size_t idx) {
	return array[idx];
      }

      void add_row() {
	std::vector<T>* vec = new std::vector<T>();
	array.push_back( *vec );
      }
      void add_row(const int size, T* arr) {
	const int curr_row = array.size();
	DEBUG << "Curr_row = " << curr_row;
	add_row();
	for(int i = 0; i < size; i++) {
	  array[curr_row].push_back(arr[i]);
	}
      }
    };
	
  }
}

#endif
