#include "gtest/gtest.h"

#include "gsl/gsl_multimin.h"

/**
 * This computes the function
 *
 *     A*(x-a)^2 + B(y-b)^2 + C,
 * where
 *  fparams = [a, b, A, B, C]
 *
 */
double paraboloid2D(const gsl_vector* x, void *fparams){
  const double *p = (const double *)fparams;
  const double y = gsl_vector_get(x, 0);
  const double z = gsl_vector_get(x, 1);
  return p[2]*(y - p[0])*(y-p[0]) + p[3]*(z-p[1])*(z-p[1]) + p[4];
}
void paraboloid2DGrad(const gsl_vector* v, void *fparams, gsl_vector *grad) {
  const double *x = (const double*)(v->data);
  const double *p = (const double*)fparams;
  gsl_vector_set(grad, 0, 2*(x[0] - p[0])*p[2]);
  gsl_vector_set(grad, 1, 2*(x[1] - p[1])*p[3]);
}
void paraboloid2DFD(const gsl_vector* x, void * params, 
		    double * f, gsl_vector* g) {
  *f = paraboloid2D(x, params);
  paraboloid2DGrad(x, params, g);
}

TEST(gsl_opt, paraboloid_cstyle) {
  /* Paraboloid center at (1,2), scale factors (10, 20), 
     minimum value 30 */
  size_t iter = 0;
  int status;
  /* Position of the minimum (1,2), scale factors 
     10,20, height 30. */
  double par[5] = { 1.0, 2.0, 10.0, 20.0, 30.0 };
  gsl_vector *x;
  gsl_multimin_function_fdf my_func;

  my_func.n = 2;  /* number of function components */
  my_func.f = &paraboloid2D;
  my_func.df = &paraboloid2DGrad;
  my_func.fdf = &paraboloid2DFD;
  my_func.params = (void *)par;

  /* Starting point, x = (5,7) */
  x = gsl_vector_alloc (2);
  gsl_vector_set (x, 0, 5.0);
  gsl_vector_set (x, 1, 7.0);

  const gsl_multimin_fdfminimizer_type *T = gsl_multimin_fdfminimizer_vector_bfgs2;
  gsl_multimin_fdfminimizer *s = gsl_multimin_fdfminimizer_alloc (T, 2);
  gsl_multimin_fdfminimizer_set(s, &my_func, x, 0.01, 1e-4);

  do {
      iter++;
      status = gsl_multimin_fdfminimizer_iterate(s);
      if (status)
        break;
      status = gsl_multimin_test_gradient(s->gradient, 1e-3);
  } while (status == GSL_CONTINUE && iter < 100);

  ASSERT_EQ(status, GSL_SUCCESS);
  ASSERT_EQ(2, iter);
  ASSERT_NEAR(1.0, gsl_vector_get(s->x, 0), 1E-6);
  ASSERT_NEAR(2.0, gsl_vector_get(s->x, 1), 1E-6);
  ASSERT_NEAR(30.0, s->f, 1E-6);

  gsl_multimin_fdfminimizer_free (s);
  gsl_vector_free (x);
}


