#include "optimize.hpp"
#include <math.h>
#include <cstring>

namespace optimize {
  GSLMinimizer::GSLMinimizer(const int num_dim) : minimizer_type_(gsl_multimin_fdfminimizer_vector_bfgs2), minimizer_(gsl_multimin_fdfminimizer_alloc(minimizer_type_, num_dim)), gradient_tolerance_(1E-3), tolerance_(1E-4), step_size_(1E-2), num_dim_(num_dim), num_steps_(100) {
  }
  GSLMinimizer::~GSLMinimizer() {
    gsl_multimin_fdfminimizer_free(minimizer_);
  }
  void GSLMinimizer::gradient_tolerance(double gt) {
    gradient_tolerance_ = gt;
  }
  void GSLMinimizer::tolerance(double tol) {
    tolerance_ = tol;
  }
  void GSLMinimizer::step_size(double ss) {
    step_size_ = ss;
  }
  void GSLMinimizer::num_steps(int num) {
    num_steps_ = num;
  }
  double GSLMinimizer::gradient_tolerance() {
    return gradient_tolerance_;
  }
  double GSLMinimizer::tolerance() {
    return tolerance_;
  }
  double GSLMinimizer::step_size() {
    return step_size_;
  }
  int GSLMinimizer::num_steps() {
    return num_steps_;
  }

  double GSLMinimizer::value() {
    return minimizer_->f;
  }

  GSLVector::GSLVector(const gsl_vector* gvec) : num_dim_(gvec->size), vec_(gsl_vector_alloc(num_dim_)) {
    gsl_vector_memcpy(vec_, const_cast<gsl_vector*>(gvec));
  }
  GSLVector::GSLVector(const int size) : num_dim_(size), vec_(gsl_vector_alloc(num_dim_)) {
  }
  GSLVector::~GSLVector() {
    gsl_vector_free(vec_);
  }
  std::ostream& operator<< (std::ostream& stream, const GSLVector& vec) {
    stream << "gsl_vector[";
    for(size_t i = 0; i < (size_t)vec.num_dim_; ++i) {
      stream << gsl_vector_get(vec.vec_, i);
      if(i+1 < (size_t)vec.num_dim_) {
	stream << ", ";
      }
    }
    stream << "]";
    return stream;
  }
  void GSLVector::update(gsl_vector* update) {
    gsl_vector_memcpy(vec_, const_cast<gsl_vector*>(update));
  }
  gsl_vector* GSLVector::get() {
    return vec_;
  }

  double GSLVector::dist(gsl_vector* x0, gsl_vector* x1) {
    double d = 0.0;
    for(size_t i = 0; i < (size_t)x0->size; ++i) {
      double diff = gsl_vector_get(x1, i) - gsl_vector_get(x0, i);
      d += (diff * diff);
    }
    return sqrt(d);
  }

  void LibLBFGSVector::init_vec() {
    vec_ = lbfgs_malloc(num_dim_);
    if(vec_ == NULL) {
      ERROR << "LibLBFGSVector cannot alloc enough memory (" << num_dim_ << ")";
      throw 4;
    }
  }

  LibLBFGSVector::LibLBFGSVector(const int size) : num_dim_(size) {
    init_vec();
  }
  LibLBFGSVector::~LibLBFGSVector() {
    if(vec_ != NULL) {
      lbfgs_free(vec_);
    }
  }
  lbfgsfloatval_t* LibLBFGSVector::get() {
    return vec_;
  }

  LibLBFGSMinimizer::LibLBFGSMinimizer(const int num_dim) : num_dim_(num_dim) {
    lbfgs_parameter_init(&params_);
    params_.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
  }
  
}
