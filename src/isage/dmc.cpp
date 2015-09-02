#include "dmc.hpp"
#include "mathops.hpp"
#include "logging.hpp"
#include "util.hpp"
#include "mathops.hpp"

#include <cmath>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_psi.h>

#include <vector>

namespace dmc {

  // double dirichlet::log_partition(const std::vector<double>& params, const unsigned int M) {
  //   const double psum = isage::util::sum(params);
  //   double individual_sum = 0.0;
  //   for(const auto& p : params) {
  //     individual_sum += gsl_sf_lngamma(p);
  //   }
  //   return (const double)(M) * (gsl_sf_lngamma(psum) - individual_sum);
  // }

  double dirichlet::hyperparameters_variational_objective(const std::vector<double>& trial, ClosureType* closure) {
    const double psum = isage::util::sum(trial);
    double individual_sum = 0.0;
    for(const auto& p : trial) {
      individual_sum += gsl_sf_lngamma(p);
    }
    const double M = closure->variational_params->size();
    const double trial_part = M * (gsl_sf_lngamma(psum) - individual_sum);
    double variational_part = 0.0;
    for(const auto& vec: *(closure->variational_params)) {
      std::vector<double> var_grad = dirichlet::grad_log_partition_static(vec);
      for(size_t i = 0; i < var_grad.size(); ++i) {
	const double x = ((trial[i] - 1.0)*var_grad[i]);
	variational_part += x;
      }
    }
    return trial_part + variational_part;
  }
  void dirichlet::hyperparameters_variational_gradient(const std::vector<double>& trial, ClosureType* closure, 
						       std::vector<double>& grad) {
    grad = dirichlet::grad_log_partition_static(trial);
    isage::util::scalar_product(-1.0, &grad);
    const double M = (const double)(closure->variational_params->size());
    isage::util::scalar_product(M, &grad);
    for(const auto& vec : *(closure->variational_params)) {
      std::vector<double> var_grad = dirichlet::grad_log_partition_static(vec);
      isage::util::sum_in_first(&grad, var_grad);
    }
  }
  double dirichlet::hyperparameters_variational_lazy_hessian(const std::vector<double>& trial_weights,
							     const int i, const int j, const double M) {
    const double sum = isage::util::sum(trial_weights);
    double h = M * gsl_sf_psi_n(1, sum);
    if(i == j) {
      h -= (M * gsl_sf_psi_n(1, trial_weights[i]));
    }
    return h;
  }

  void dirichlet::hyperparameters_variational_hessian_diag(const std::vector<double>& trial_weights,
							   const double M, std::vector<double>& diag) {
    if(diag.size() == 0) {
      ERROR << "Hessian diagonal receptacle must have non-zero size!";
      throw 3;
    }
    const size_t dim = diag.size();
    for(size_t i = 0; i < dim; ++i) {
      diag[i] = dirichlet::hyperparameters_variational_lazy_hessian(trial_weights, i, i, M);
    }
  }

  // std::vector<double> dirichlet::hyperparameters_variational_nr(const std::vector<double>& trial_weights,
  // 								ClosureType* closure,
  // 								const double thres,
  // 								const int max_iters) {
  //   std::vector<double> point(trial_weights);
  //   const size_t dim = point.size();
  //   const double M = (const double)(closure->variational_params->size());
  //   int iter = 0;
  //   std::vector<double> grad(dim);
  //   std::vector<double> hessian_diag(dim);
  //   double hessian_odiag = 0.0;
  //   double val = dirichlet::hyperparameters_variational_objective(point, closure);
  //   double oval = val;
  //   double diff = 1E9;
  //   do {
  //     // compute the gradient
  //     dirichlet::hyperparameters_variational_gradient(point, closure, grad);
  //     // get the diagonal of the Hessian
  //     dirichlet::hyperparameters_variational_hessian_diag(point, M, hessian_diag);
  //     // get the off-diagonal of the Hessian
  //     hessian_odiag = dirichlet::hyperparameters_variational_lazy_hessian(point, 0, 1, M);
  //     // substract off the common value
  //     isage::util::sum(-1*hessian_odiag, &hessian_diag);
  //     // compute "c"
  //     double c_num = 0.0, c_denom = 0.0;
  //     for(size_t i = 0; i < dim; ++i) {
  // 	c_num += grad[i] / hessian_diag[i];
  // 	c_denom += 1.0 / hessian_diag[i];
  //     }
  //     c_denom += 1.0 / hessian_odiag;
  //     const double c = c_num / c_denom;
  //     // compute the multiplier
  //     isage::util::scalar_product(-1.0 * c, &grad);
  //     for(size_t i = 0; i < dim; ++i) {
  // 	grad[i] /= hessian_diag[i];
  // 	INFO << "multiplier for point " << i << ", val of " << point[i]  << " is " << grad[i] ;
  //     }
  //     isage::util::linear_combination_in_first(&point, grad, 1.0, -1.0);
  //     isage::util::ensure_min(1E-8, &point);
  //     val = dirichlet::hyperparameters_variational_objective(point, closure);
  //     diff = val - oval;
  //     if(diff <= 0 && iter > 0) { // we're arching back, so stop
  //     	// undo it
  //     	isage::util::linear_combination_in_first(&point, grad, 1.0, 1.0);
  //     	WARN << " Breaking due to arcing back...";
  //     	break;
  //     }
  //     INFO << "Moving " << oval << " to " << val << " (diff = " << diff << ")";
  //     isage::util::print_1d(point);
  //     oval = val;
  //   } while(++iter < max_iters && std::abs(diff) > thres);
  //   return point;
  // }
  std::vector<double> dirichlet::hyperparameters_variational_nr(const std::vector<double>& trial_weights,
								ClosureType* closure,
								const double thres,
								const int max_iters) {
    std::vector<double> point(trial_weights);
    const size_t dim = point.size();
    const double M = (const double)(closure->variational_params->size());
    int iter = 0;
    std::vector<double> grad(dim);
    std::vector<double> hessian_diag(dim);
    double hessian_odiag = 0.0;
    double val = dirichlet::hyperparameters_variational_objective(point, closure);
    double oval = val;
    //double diff = 1E9;
    int decay = 0;
    double decay_fact = 0.8;
    int max_decay = 10;
    bool keep_going = true;
    //    do {
    while(keep_going) {
      // INFO << " Current value: " << val;
      // isage::util::print_1d(point);
      // compute the gradient
      dirichlet::hyperparameters_variational_gradient(point, closure, grad);
      //isage::util::print_1d(grad);
      // get the diagonal of the Hessian
      dirichlet::hyperparameters_variational_hessian_diag(point, M, hessian_diag);
      // get the off-diagonal of the Hessian
      hessian_odiag = dirichlet::hyperparameters_variational_lazy_hessian(point, 0, 1, M);
      // substract off the common value
      isage::util::sum(-1*hessian_odiag, &hessian_diag);
      // compute "c"
      double c_num = 0.0, c_denom = 0.0;
      for(size_t i = 0; i < dim; ++i) {
	c_num += grad[i] / hessian_diag[i];
	c_denom += 1.0 / hessian_diag[i];
      }
      c_denom += 1.0 / hessian_odiag;
      const double c = c_num / c_denom;
      // compute the multiplier
      isage::util::scalar_product(-1.0 * c, &grad);
      for(size_t i = 0; i < dim; ++i) {
	grad[i] /= hessian_diag[i];
	INFO << "multiplier for point " << i << ", val of " << point[i]  << " is " << grad[i] ;
      }

      std::vector<double> update(dim);
      // this loop is to prevent bad updates due to unstable Hessians
      // if all always went well, we'd only do one update
      while (true) {
	bool singular_h = false;
	for(size_t i = 0; i < dim; ++i) {
	double step = pow(decay_fact, decay) * grad[i];
	if (point[i] <= step) {
	  singular_h = true;
	  break;
	}
	update[i] = point[i] - step;
      }
	
      // if the Hessian became unstable, then we need to revert the changes
      // otherwise, we're good!
      if(singular_h) {
	decay++;
	update = point;
	if(decay > max_decay) {
	  break;
	}
      } else {
	break;
      }
    }

      // compute the alpha sum and check for alpha converge
      double alpha_sum = 0.0;
      keep_going = false;
      for(size_t i = 0; i < dim; ++i) {
	alpha_sum += update[i];
	if(abs((update[i] - point[i]) / point[i]) >= thres) {
	  keep_going = true;
	}
      }
      if(++iter >= max_iters) {
	keep_going = false;
      }
      if(decay > max_decay) {
	break;
      }
      for(size_t i = 0; i < dim; ++i) {
	point[i] = update[i];
      }

      // isage::util::linear_combination_in_first(&point, grad, 1.0, -1.0);
      // isage::util::ensure_min(1E-8, &point);
      val = dirichlet::hyperparameters_variational_objective(point, closure);
      //diff = val - oval;
      // if(diff <= 0 && iter > 0) { // we're arching back, so stop
      // 	// undo it
      // 	isage::util::linear_combination_in_first(&point, grad, 1.0, 1.0);
      // 	WARN << " Breaking due to arcing back...";
      // 	break;
      // }
      INFO << "Moving " << oval << " to " << val << " (diff = " << (val - oval) << ")";
      //isage::util::print_1d(point);
      oval = val;
    }
    //    } while(++iter < max_iters && keep_going); // && std::abs(diff) > thres);
    return point;
  }
  std::vector<double> dirichlet::hyperparameters_variational_nr(const std::vector<double>& trial_weights,
								const std::vector<std::vector<double> >& suff_stats,
								const double thres,
								const int max_iters) {
    ClosureType closure;
    closure.variational_params = const_cast<std::vector<std::vector<double> >* >(&suff_stats);
    return dirichlet::hyperparameters_variational_nr(trial_weights, &closure, thres,  max_iters);
  }

  // Below are three GSL-specific functions. Note that they epx the trial value
  // (since GSL optimization is only for unconstrained problems.
  double dirichlet::hyperparameters_variational_objective_gsl(const gsl_vector* gtrial, void *fparams) {
    ClosureType* closure = (ClosureType*)fparams;
    std::vector<double> trial = optimize::GSLVector::to_container<std::vector<double> >(gtrial);
    isage::util::exp(&trial);
    return -dirichlet::hyperparameters_variational_objective(trial, closure);
  }
  void dirichlet::hyperparameters_variational_gradient_gsl(const gsl_vector* gtrial, 
							   void *fparams, gsl_vector* gsl_grad) {
    ClosureType* closure = (ClosureType*)fparams;
    std::vector<double> trial = optimize::GSLVector::to_container<std::vector<double> >(gtrial);
    isage::util::exp(&trial);
    std::vector<double> grad;
    dirichlet::hyperparameters_variational_gradient(trial, closure, grad);
    isage::util::scalar_product(-1.0, &grad);
    gsl_vector_memcpy(gsl_grad, optimize::GSLVector(grad).get());
  }
  void dirichlet::hyperparameters_variational_obj_grad_gsl(const gsl_vector* trial, void *fparams,
							   double* f, gsl_vector* grad) {
    *f = dirichlet::hyperparameters_variational_objective_gsl(trial, fparams);
    dirichlet::hyperparameters_variational_gradient_gsl(trial, fparams, grad);
  }

  gsl_multimin_function_fdf dirichlet::get_fdf(ClosureType* params, const size_t size) {
    gsl_multimin_function_fdf objective;
    objective.n   = size;
    objective.f   = &dirichlet::hyperparameters_variational_objective_gsl;
    objective.df  = &dirichlet::hyperparameters_variational_gradient_gsl;
    objective.fdf = &dirichlet::hyperparameters_variational_obj_grad_gsl;
    objective.params = (void*)params;
    return objective;
  }

  /**
   * Compute the gradient of the log normalizer A(hypers):
   * 
   *   ∂A(hypers)
   *   ---------- = Ψ(hyper_i) - Ψ(\sum_j hyper_j)
   *       ∂i
   *
   * where Ψ(x) is the digamma function (derivative of the log Gamma).
   */
  std::vector<double> dirichlet::grad_log_partition_static(const std::vector<double>& hyperparameters) {
    std::vector<double> grad(hyperparameters.size());
    const double hsum = isage::util::sum(hyperparameters);
    const double digsum = gsl_sf_psi(hsum);
    size_t index = 0;
    for(const auto& hyper_ : hyperparameters) {
      grad[index++] = gsl_sf_psi(hyper_) - digsum;
    }
    return grad;
  }
  /**
   * Compute the gradient of the log normalizer A(hypers):
   * 
   *   ∂A(hypers)
   *   ---------- = Ψ(hyper_i) - Ψ(\sum_j hyper_j)
   *       ∂i
   *
   * where Ψ(x) is the digamma function (derivative of the log Gamma).
   */
  void dirichlet::grad_log_partition_static(const std::vector<double>& hyperparameters, std::vector<double>* grad) {
    if(grad == NULL) {
      ERROR << "grad pointer must not be null";
      throw 2;
    }
    if(grad->size() != hyperparameters.size()) {
      ERROR << "grad size must be equal to hyperparameter size";
      throw 2;
    }
    const double hsum = isage::util::sum(hyperparameters);
    const double digsum = gsl_sf_psi(hsum);
    size_t index = 0;
    for(const auto& hyper_ : hyperparameters) {
      grad->operator[](index++) = gsl_sf_psi(hyper_) - digsum;
    }
  }

  /**
   * Compute the gradient of the log normalizer A(hypers):
   * 
   *   ∂A(hypers)
   *   ---------- = Ψ(hyper_i) - Ψ(\sum_j hyper_j) 
   *       ∂i
   *
   * where Ψ(x) is the digamma function (derivative of the log Gamma).
   */
  std::vector<double> dmc::grad_log_partition() {
    std::vector<double> grad(hyperparameters_.size());
    const double digsum = gsl_sf_psi(hyperparameter_sum_);
    size_t index = 0;
    for(const auto& hyper_ : hyperparameters_) {
      grad[index++] = gsl_sf_psi(hyper_) - digsum;
    }
    return grad;
  }

  /**
   * This implementation of Wallach's "Method 1" uses an iterative
   * fixed-point algorithm to reestimate the hyperparameters.
   */
  void dmc::reestimate_hyperparameters_wallach1(const std::vector<std::vector<int> >& counts,
						int num_iterations, double floor) {
    const std::vector< std::vector< int > > histogram = isage::util::histogram(counts);
    const std::vector<int> marginals = isage::util::marginals(counts);
    const std::map<int, int> marginal_histogram = isage::util::sparse_histogram(marginals);
    const int K = counts[0].size();
    const std::vector<int> column_max = isage::util::column_max<int>(counts);
    const int max_marginal = isage::util::max(marginals);
    for(int iter = 0; iter < num_iterations; ++iter) {
      double d = 0.0;
      double s = 0.0;
      for(int idx = 1; idx <= max_marginal; ++idx) {
	d += 1.0/(idx - 1 + hyperparameter_sum_);
	s += (marginal_histogram.find(idx) == marginal_histogram.end() ? 0 : marginal_histogram.at(idx))*d;
      }
      hyperparameter_sum_ = 0.0;
      for(int k = 0; k < K; ++k) {
	const std::vector<int>& k_histogram = histogram[k];
	const int c_max = column_max[k];
	const double h_k = hyperparameters_[k];
	d = 0.0;
	double sk = 0.0;
	for(int n = 1; n <= c_max; ++n) {
	  d += 1.0/(n - 1 + h_k);
	  sk += k_histogram[n]*d;
	}
	double y = h_k * sk / s;
	hyperparameters_[k] = (y > floor) ? y : floor;
	hyperparameter_sum_ += hyperparameters_[k];
      }
    }
  }

  void dmc::reestimate_hyperparameters_variational(const std::vector< std::vector<double> >& var_params) {
    dirichlet::ClosureType closure;
    closure.variational_params = const_cast<std::vector<std::vector<double> >* >(&var_params);
    size_t hsize = hyperparameters_.size();
    gsl_multimin_function_fdf hparam_fdf = dirichlet::get_fdf(&closure, hsize);
    // get an initial point: we'll use (directly) the current hyperparameters
    optimize::GSLMinimizer optimizer(hsize);
    std::vector<double> copy(hyperparameters_);
    int opt_status = optimizer.minimize(&hparam_fdf, copy);
    INFO << "Hyperparameter optimization has optimization status " << opt_status;
    //isage::util::print_1d(copy);
    recompute_hyperparameter_sum();
  }

  // void dmc::reestimate_symmetric_hyperparameters_variational_newton(const std::vector< std::vector<double> >& var_params) {
  //   // double a, log_a, init_a = 100;
  //   // double f, df, d2f;
  //   // int iter = 0;

  //   // log_a = log(init_a);
  //   // do
  //   //   {
  //   //     iter++;
  //   //     a = exp(log_a);
  //   //     if (isnan(a))
  //   // 	  {
  //   //         init_a = init_a * 10;
  //   //         printf("warning : alpha is nan; new init = %5.5f\n", init_a);
  //   //         a = init_a;
  //   //         log_a = log(a);
  //   // 	  }
  //   //     f = alhood(a, ss, D, K);
  //   //     df = d_alhood(a, ss, D, K);
  //   //     d2f = d2_alhood(a, D, K);
  //   //     log_a = log_a - df/(d2f * a + df);
  //   //     printf("alpha maximization : %5.5f   %5.5f\n", f, df);
  //   //   }
  //   // while ((fabs(df) > NEWTON_THRESH) && (iter < MAX_ALPHA_ITER));
  //   // return(exp(log_a));

  //   dirichlet::ClosureType closure;
  //   closure.variational_params = const_cast<std::vector<std::vector<double> >* >(&var_params);
  //   size_t hsize = hyperparameters_.size();
  //   gsl_multimin_function_fdf hparam_fdf = dirichlet::get_fdf(&closure, hsize);
  //   // get an initial point: we'll use (directly) the current hyperparameters
  //   optimize::GSLMinimizer optimizer(hsize);
  //   int opt_status = optimizer.minimize(&hparam_fdf, hyperparameters_);
  //   BOOST_LOG_TRIVIAL(info) << "Hyperparameter optimization has optimization status " << opt_status;
  //   isage::util::print_1d(hyperparameters_);
  //   recompute_hyperparameter_sum();
  // }

  /**
   * This implementation of Wallach's "Method 1" uses an iterative
   * fixed-point algorithm to reestimate the hyperparameters.
   */
  void mfgdmc::reestimate_hyperparameters_wallach1(const std::vector<std::vector<std::vector<int> > >& counts,
						   int num_iterations, double floor) {
    const std::vector< std::vector< int > > histogram = isage::util::mf_histogram(counts);
    const std::vector<int> marginals = isage::util::marginals(counts);
    const std::map<int, int> marginal_histogram = isage::util::sparse_histogram(marginals);
    const int K = counts[0][0].size();
    const std::vector<int> column_max = isage::util::column_max<int>(counts);
    const int max_marginal = isage::util::max(marginals);
    for(int iter = 0; iter < num_iterations; ++iter) {
      double d = 0.0;
      double s = 0.0;
      for(int idx = 1; idx <= max_marginal; ++idx) {
	d += 1.0/(idx - 1 + hyperparameter_sum_);
	s += (marginal_histogram.find(idx) == marginal_histogram.end() ? 0 : marginal_histogram.at(idx))*d;
      }
      hyperparameter_sum_ = 0.0;
      for(int k = 0; k < K; ++k) {
	const std::vector<int>& k_histogram = histogram[k];
	const int c_max = column_max[k];
	const double h_k = hyperparameters_[k];
	d = 0.0;
	double sk = 0.0;
	for(int n = 1; n <= c_max; ++n) {
	  d += 1.0/(n - 1 + h_k);
	  sk += k_histogram[n]*d;
	}
	double y = h_k * sk / s;
	hyperparameters_[k] = (y > floor) ? y : floor;
	hyperparameter_sum_ += hyperparameters_[k];
      }
    }
  }

  /**
   * Compute relevant ratio of, e.g.,
   * log(Gamma(x+y) / Gamma(x)) by expanding out
   *   Gamma(x+y) = \prod_{i = 0}^{y-1} (x + y - i) * Gamma(x),
   * dividing out Gamma(x), and taking logs.
   */
  const double dmc::log_u_conditional_oracle(const double base_value,
					     const double sum,
					     const int num_to_remove) {
    const double nsum = (double)(base_value + num_to_remove - 1);
    double numerator = 0.0;
    int i;
    // this handles the numerator
    for(i = 0; i < num_to_remove; i++) {
      numerator += gsl_sf_log(nsum - i);
    }
    // now handle the denom.
    double denominator = 0.0;
    double dsum = (double)(sum + num_to_remove - 1);
    for(i = 0; i < num_to_remove; i++) {
      denominator += gsl_sf_log(dsum - i);
    }
    return numerator - denominator;
  }

  /**
   * Compute relevant ratio of, e.g.,
   * log(Gamma(x+y) / Gamma(x)) by expanding out
   *   Gamma(x+y) = \prod_{i = 0}^{y-1} (x + y - i) * Gamma(x),
   * dividing out Gamma(x), and taking logs.
   */
  const double dmc::log_u_conditional_oracle(const size_t idx,
					     const int* const histogram,
					     const int sum,
					     const int num_to_remove) {
    return dmc::log_u_conditional_oracle(histogram[idx] + hyperparameters_[idx],
					 sum + hyperparameter_sum_, num_to_remove);
  }
  /**
   * Compute relevant ratio of, e.g.,
   * log(Gamma(x+y) / Gamma(x)) by expanding out
   *   Gamma(x+y) = \prod_{i = 0}^{y-1} (x + y - i) * Gamma(x),
   * dividing out Gamma(x), and taking logs.
   */
  const double dmc::log_u_conditional_oracle(const size_t idx,
					     const std::vector<int>& histogram,
					     const int sum,
					     const int num_to_remove) {
    return dmc::log_u_conditional_oracle(histogram[idx] + hyperparameters_[idx],
					 sum + hyperparameter_sum_, num_to_remove);
  }

  /**
   * Compute relevant ratio of, e.g., log(Gamma(histogram[idx]+y) / Gamma(x))
   * with GSL's gsl_sf_lngamma function.
   */
  const double dmc::log_u_conditional_gsl(const double base_value,
					  const double sum,
					  const int num_to_remove) {
    const double numerator = gsl_sf_lngamma((double)(base_value + num_to_remove)) - gsl_sf_lngamma((double)(base_value));
    const double denominator = gsl_sf_lngamma((double)(sum + num_to_remove)) - gsl_sf_lngamma((double)sum);
    return numerator - denominator;
  }

  /**
   * Compute relevant ratio of, e.g., log(Gamma(histogram[idx]+y) / Gamma(x))
   * with GSL's gsl_sf_lngamma function.
   */
  const double dmc::log_u_conditional_gsl(const size_t idx,
					  const int* const histogram,
					  const int sum,
					  const int num_to_remove) {
    return dmc::log_u_conditional_gsl(histogram[idx] + hyperparameters_[idx],
				      sum + hyperparameter_sum_, num_to_remove);
  }
  /**
   * Compute relevant ratio of, e.g., log(Gamma(histogram[idx]+y) / Gamma(x))
   * with GSL's gsl_sf_lngamma function.
   */
  const double dmc::log_u_conditional_gsl(const size_t idx,
					  const std::vector<int>& histogram,
					  const int sum,
					  const int num_to_remove) {
    return dmc::log_u_conditional_gsl(histogram[idx] + hyperparameters_[idx],
				      sum + hyperparameter_sum_, num_to_remove);
  }

  /**
   * Compute log(Gamma(x+y) / Gamma(x)) by using Sterling's approximation
   * for log(n) ~= n*log(n) - n.
   */
  const double dmc::log_u_conditional_sterling(const double base_value,
					       const double sum,
					       const int num_to_remove) {
    const double numerator = mathops::sterling((double)(base_value + num_to_remove)) - mathops::sterling((double)(base_value));
    const double denominator = mathops::sterling((double)(sum + num_to_remove)) - mathops::sterling((double)sum);
    return numerator - denominator;
  }

  /**
   * Compute log(Gamma(x+y) / Gamma(x)) by using Sterling's approximation
   * for log(n) ~= n*log(n) - n.
   */
  const double dmc::log_u_conditional_sterling(const size_t idx,
					       const int* const histogram,
					       const int sum,
					       const int num_to_remove) {
    return dmc::log_u_conditional_sterling(histogram[idx] + hyperparameters_[idx],
					   sum + hyperparameter_sum_, num_to_remove);
  }
  /**
   * Compute log(Gamma(x+y) / Gamma(x)) by using Sterling's approximation
   * for log(n) ~= n*log(n) - n.
   */
  const double dmc::log_u_conditional_sterling(const size_t idx,
					       const std::vector<int>& histogram,
					       const int sum,
					       const int num_to_remove) {
    return dmc::log_u_conditional_sterling(histogram[idx] + hyperparameters_[idx],
					   sum + hyperparameter_sum_, num_to_remove);
  }


  /**
   * This computes the log of the unnormalized conditional:
   *
   *            Gamma( x + c)
   *            -------------
   *               Gamma(x)
   *  log    -------------------
   *          Gamma(sum x_j + c)
   *            --------------
   *            Gamma(sum x_j)
   *
   * where x_i = histogram[idx], and sum = \sum_j x_j.
   *
   * By default, this calls
   *   log_u_conditional_oracle     if 1 <= num_to_remove <= 4
   *   log_u_conditional_gsl        if num_to_remove >= 5
   *
   * These can be changed by setting:
   *   - use_gsl(bool)
   *   - use_sterling(bool)
   */
  const double dmc::log_u_conditional(const size_t idx,
				      const int* const histogram,
				      const int sum,
				      const int num_to_remove) {
    if(num_to_remove <= 4 || (!use_gsl_ && !use_sterling_)) {
      return log_u_conditional_oracle(idx, histogram, sum, num_to_remove);
    } else {
      if(use_gsl_) {
	return log_u_conditional_gsl(idx, histogram, sum, num_to_remove);
      } else {
	// call sterling function
	return log_u_conditional_sterling(idx, histogram, sum, num_to_remove);
      }
    }
  }

  /**
   * This computes the log of the unnormalized conditional:
   *
   *            Gamma( x + c)
   *            -------------
   *               Gamma(x)
   *  log    -------------------
   *          Gamma(sum x_j + c)
   *            --------------
   *            Gamma(sum x_j)
   *
   * where x_i = histogram[idx], and sum = \sum_j x_j.
   *
   * By default, this calls
   *   log_u_conditional_oracle     if 1 <= num_to_remove <= 4
   *   log_u_conditional_gsl        if num_to_remove >= 5
   *
   * These can be changed by setting:
   *   - use_gsl(bool)
   *   - use_sterling(bool)
   */
  const double dmc::log_u_conditional(const size_t idx,
				      const std::vector<int>& histogram,
				      const int sum,
				      const int num_to_remove) {
    if(num_to_remove <= 4 || (!use_gsl_ && !use_sterling_)) {
      return log_u_conditional_oracle(idx, histogram, sum, num_to_remove);
    } else {
      if(use_gsl_) {
	return log_u_conditional_gsl(idx, histogram, sum, num_to_remove);
      } else {
	// call sterling function
	return log_u_conditional_sterling(idx, histogram, sum, num_to_remove);
      }
    }
  }

  ///////////////////////////////////////////////////////////

  const double gdmc::log_joint(const std::vector< std::vector<int> >& counts) {
    const int domain_size = this->size();
    double lgamma_hp_i = 0.0;
    double lgamma_num_i = 0.0;
    double lgamma_num_strata = 0.0;
    recompute_hyperparameter_sum();
    double log_prod = num_strata_ * gsl_sf_lngamma(hyperparameter_sum_);
    for(int i = 0; i < domain_size; ++i) {
      lgamma_hp_i += gsl_sf_lngamma(hyperparameters_[i]);
    }
    for(int strata = 0; strata < num_strata_; ++strata) {
      const std::vector<int> strata_counts = counts[strata];
      double strata_sum = 0.0;
      for(int dom_index = 0; dom_index < domain_size; ++dom_index) {
	double x = strata_counts[dom_index] + hyperparameters_[dom_index];
	strata_sum += x;
	lgamma_num_i += gsl_sf_lngamma(x);
      }
      lgamma_num_strata += gsl_sf_lngamma(strata_sum);
    }
    log_prod += lgamma_num_strata - lgamma_num_i - num_strata_*lgamma_hp_i;
    return log_prod;
  }

  const double mfgdmc::log_joint(const std::vector< std::vector< std::vector<int> > >& counts) {
    const int domain_size = this->size();
    double lgamma_hp_i = 0.0;
    double lgamma_num_i = 0.0;
    double lgamma_num_strata = 0.0;
    recompute_hyperparameter_sum();
    int num_computed_strata = num_entries();
    double log_prod = num_computed_strata * gsl_sf_lngamma(hyperparameter_sum_);
    for(int i = 0; i < domain_size; ++i) {
      lgamma_hp_i += gsl_sf_lngamma(hyperparameters_[i]);
    }
    for(int strata = 0; strata < num_strata_1_; ++strata) {
      for(int strata2 = 0; strata2 < num_strata_2_[strata]; ++strata2) {
	const std::vector<int> strata_counts = counts[strata][strata2];
	double strata_sum = 0.0;
	for(int dom_index = 0; dom_index < domain_size; ++dom_index) {
	  double x = strata_counts[dom_index] + hyperparameters_[dom_index];
	  strata_sum += x;
	  lgamma_num_i += gsl_sf_lngamma(x);
	}
	lgamma_num_strata += gsl_sf_lngamma(strata_sum);
      }
    }
    log_prod += lgamma_num_strata - lgamma_num_i - num_computed_strata*lgamma_hp_i;
    return log_prod;
  }
}
