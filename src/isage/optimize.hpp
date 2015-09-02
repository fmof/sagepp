#ifndef ISAGE_OPTIMIZE_H_ 
#define ISAGE_OPTIMIZE_H_

#include "gsl/gsl_multimin.h"
#include <lbfgs.h>
#include <list>
#include <logging.hpp>
#include <ostream>
#include <vector>
#include "util.hpp"

#define ADD_OPTIMIZATION_STATUSES(mins, grads) << " after " << (mins) << " minimization attempts and " << (grads) << " gradient status checks"

namespace optimize {
  
  /// This represents an unsuccessful attempt at converting 
  /// C++ member function pointers to C-style function pointers.
  // template< typename F >
  // class GSLMultiminFDFWrapper : public gsl_multimin_function_fdf {
  // public:
  //   GSLMultiminFDFWrapper(const F& func) : _func(func) {
  //     fdf = &GSLMultiminFDFWrapper::invoke;
  //     params=this;
  //   }
  // private:
  //   const F& _func;
  //   static void invoke(const gsl_vector* x, void* y, double* z, gsl_vector* g) {
  //     return static_cast<GSLMultiminFDFWrapper*>(params)->_func(x,y,z,g);
  //   }
  // };

  class GSLVector {
  private:
    const size_t num_dim_;
    gsl_vector *vec_;
  public:
    template <typename Container> 
    inline explicit GSLVector(const Container& input) : num_dim_((const size_t)input.size()), 
							vec_(gsl_vector_alloc(num_dim_)) {
      size_t counter = 0;
      for(const auto& obj : input) {
	gsl_vector_set(vec_, counter++, obj);
      }
    };
    GSLVector(const int size);
    GSLVector(const gsl_vector* gvec);
    ~GSLVector();
    
    /**
     * IndexableContainer must have:
     * - an operator[](size_t idx) (preferably constant time!)
     * - a constructor IndexableContainer(int size)
     */
    template <typename IndexableContainer>
    inline static IndexableContainer sum(const gsl_vector* vec, 
					 const IndexableContainer* other) {
      const size_t vec_size = vec->size;
      if(other->size() > vec_size) {
	ERROR << "Cannot sum gsl_vector of size " << vec_size << " with container of size " << (other->size());
	throw 1;
      }
      IndexableContainer res(vec_size);
      for(size_t i = 0; i < vec_size; ++i) {
	res[i] = gsl_vector_get(vec, i) + other->operator[](i);
      }
      return res;
    }

    template <typename Container> static inline Container to_container(const gsl_vector* vec) {
      Container res;
      typename Container::iterator it = res.end();
      const size_t v_size = vec->size;
      for(size_t counter = 0; counter < v_size; ++counter) {
	it = res.insert(it, gsl_vector_get(vec, counter));
	++it;
      }
      return res;
    }
    template <typename Container> inline Container to_container() {
      Container res;
      typename Container::iterator it = res.end();
      for(size_t counter = 0; counter < num_dim_; ++counter) {
	it = res.insert(it, gsl_vector_get(vec_, counter));
	++it;
      }
      return res;
    }

    template <typename Container> inline void to_container(Container& container) {
      container.clear();
      typename Container::iterator it = container.end();
      for(size_t counter = 0; counter < num_dim_; ++counter) {
	it = container.insert(it, gsl_vector_get(vec_, counter));
	++it;
      }
    }
    
    static double dist(gsl_vector* x0, gsl_vector* x1);

    friend std::ostream& operator<< (std::ostream& stream, const GSLVector& vec);
    void update(gsl_vector* update);
    gsl_vector* get();
  };
  /**
   * Note that WeightType must be indexable by SupportType
   */
  class GSLMinimizer {
  private:
    const gsl_multimin_fdfminimizer_type *minimizer_type_;
    gsl_multimin_fdfminimizer *minimizer_;
    double gradient_tolerance_;
    double tolerance_;
    double step_size_;
    int num_dim_;  
    int num_steps_;
  public:
    GSLMinimizer(const int num_dim);
    ~GSLMinimizer();

    template <typename PointType = std::vector<double>,
	      typename Pushbackable = std::list<int> > 
    inline int minimize(gsl_multimin_function_fdf* func_ptr, 
			PointType& point, int num_steps,
			Pushbackable* iterate_statuses,
			Pushbackable* gradient_statuses) {
      // convert init point to gsl_vector
      GSLVector gsl_point(point);
      gsl_vector* raw_gvec = gsl_point.get();
      gsl_multimin_fdfminimizer_set(minimizer_, func_ptr, raw_gvec, 
				    step_size_, tolerance_);
      int status;
      int iter = 0;
      int minimize_attempts = 0, grad_checks = 0;
      do {
	DEBUG << "GSLMinimizer iteration " << iter;
	status = gsl_multimin_fdfminimizer_iterate(minimizer_);
	++minimize_attempts;
	DEBUG << "status after " << iter << "th iteration: " << status;
	iter++;
	if(iterate_statuses != NULL) {
	  iterate_statuses->push_back(status);
	}
	if (status) break;
	status = gsl_multimin_test_gradient(minimizer_->gradient, 
					    gradient_tolerance_);
	++grad_checks;
	if(gradient_statuses != NULL) {
	  gradient_statuses->push_back(status);
	}
      } while (status == GSL_CONTINUE && iter < num_steps);
      const double dist = optimize::GSLVector::dist(raw_gvec, minimizer_->x);
      INFO << "Optimization moved the point " << dist << " units away";
      gsl_point.update(minimizer_->x);
      gsl_point.to_container(point);
      switch(status) {
      case GSL_ENOPROG:
	WARN << "Optimization iteration is not making progress toward solution" ADD_OPTIMIZATION_STATUSES(minimize_attempts, grad_checks);
	break;
      case GSL_CONTINUE:
	WARN << "Optimization iteration has not converged" ADD_OPTIMIZATION_STATUSES(minimize_attempts, grad_checks);
	break;
      case GSL_SUCCESS:
	INFO << "Optimization successful" ADD_OPTIMIZATION_STATUSES(minimize_attempts, grad_checks);
	break;
      default:
	WARN << "Optimization resulted in GSL code " << status ADD_OPTIMIZATION_STATUSES(minimize_attempts, grad_checks);
	break;
      }
      return status;
    }

    template <typename PointType = std::vector<double>,
	      typename Pushbackable = std::list<int> > 
    inline int minimize(gsl_multimin_function_fdf* func_ptr, 
			PointType& point,
			Pushbackable* iterate_statuses,
			Pushbackable* gradient_statuses) {
      return minimize(func_ptr, point, num_steps_,
		      iterate_statuses, gradient_statuses);
    }

    template <typename PointType = std::vector<double> > 
    inline int minimize(gsl_multimin_function_fdf* func_ptr, 
			PointType& point) {
      return minimize<PointType, std::list<int> >(func_ptr, point, num_steps_, NULL, NULL);
    }
    void gradient_tolerance(double gt);
    void tolerance(double tol);
    void step_size(double ss);
    void num_steps(int num);
    double gradient_tolerance();
    double tolerance();
    double step_size();
    int num_steps();
    
    double value();
  };

  struct LibLBFGSFunction {
    lbfgs_evaluate_t eval;
    lbfgs_progress_t progress;
    void* params;
  };

  struct LibLBFGSNoOp {
    static int progress(void *instance, const lbfgsfloatval_t *x,
		 const lbfgsfloatval_t *g, const lbfgsfloatval_t fx,
		 const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, 
		 const lbfgsfloatval_t step,
		 int n, int k, int ls) {
      return 0;
    }
  };

  class LibLBFGSVector {
  private:
    const size_t num_dim_;
    lbfgsfloatval_t *vec_;
    void init_vec();
  public:
    template <typename Container> 
    inline explicit LibLBFGSVector(const Container& input) : num_dim_((const size_t)input.size()), vec_(NULL) {
      init_vec();
      size_t counter = 0;
      for(const auto& obj : input) {
	vec_[counter++] = obj;
      }
    };
    LibLBFGSVector(const int size);
    ~LibLBFGSVector();

    /**
     * IndexableContainer must have:
     * - an operator[](size_t idx) (preferably constant time!)
     * - a constructor IndexableContainer(int size)
     */
    template <typename IndexableContainer>
    inline static IndexableContainer sum(const lbfgsfloatval_t* vec, 
					 const IndexableContainer* other,
					 const size_t vec_size) {
      IndexableContainer res(vec_size);
      for(size_t i = 0; i < vec_size; ++i) {
	res[i] = vec[i] + other->operator[](i);
      }
      return res;
    }
    
    template <typename Container> static inline Container to_container(const lbfgsfloatval_t* vec, const size_t v_size) {
      Container res;
      typename Container::iterator it = res.end();
      for(size_t counter = 0; counter < v_size; ++counter) {
	it = res.insert(it, vec[counter]);
	++it;
      }
      return res;
    }
    template <typename Container> inline Container to_container() {
      Container res;
      typename Container::iterator it = res.end();
      for(size_t counter = 0; counter < num_dim_; ++counter) {
	it = res.insert(it, vec_[counter]);
	++it;
      }
      return res;
    }
    template <typename Container> inline void to_container(Container& container) {
      container.clear();
      typename Container::iterator it = container.end();
      for(size_t counter = 0; counter < num_dim_; ++counter) {
	it = container.insert(it, vec_[counter]);
	++it;
      }
    }
    
    template <typename Container> 
    static inline void copy(const Container& other, lbfgsfloatval_t* vec) {
      const size_t v_size = other.size();
      for(size_t counter = 0; counter < v_size; ++counter) {
	vec[counter] = other[counter];
      }
    }

    lbfgsfloatval_t* get();
  };

  class LibLBFGSMinimizer {
  private:
    const int num_dim_;
    lbfgs_parameter_t params_;
  public:
    LibLBFGSMinimizer(const int num_dim);
    
    template <typename PointType = std::vector<double> >
    int minimize(LibLBFGSFunction* func_ptr, PointType& point) {
      lbfgsfloatval_t func_val;
      LibLBFGSVector vec(point);
      lbfgsfloatval_t* raw_lbfgs_point = vec.get();
      int status = lbfgs(num_dim_, raw_lbfgs_point, &func_val,
			 func_ptr->eval, func_ptr->progress,
			 func_ptr->params, &params_);
      vec.to_container(point);
      return status;
    }
  };
}

#endif
