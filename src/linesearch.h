#pragma once

#include <EigenTypes.h>
#include "variables/mixed_variable.h"
#include "variables/displacement.h"
#include "simulation_state.h"
#include <iomanip>

namespace {

  // Cubic interopolation to find minimum alpha along an interval
  //    Cubic has form g(alpha) = a alpha^3 + b alpha^2 + f'(x0) alpha + f(x0)
  //    Solve for [a b] then use quadratic equation to find the minimum.
  //
  // f(x) = f(x + alpha*d) where d is the descent direction
  //
  // Params
  //    fx0   - f(x)
  //    gTd   - f'(x0)^T d
  //    fx1   - f(x + a1*d)
  //    fx2   - f(x + a2*d)
  //    a1    - newest step size in interval (alpha_1)
  //    a2    - previous step size in interval (alpha_2)
  template <typename Scalar>
  static inline Scalar cubic(Scalar fx0, Scalar gTd, Scalar fx1, Scalar fx2,
      Scalar a1, Scalar a2) {

    // Determinant of coefficient fitting matrix
    Scalar det = 1.0 / (a1*a1*a2*a2*(a1-a2));
    
    // Adjugate matrix (for 2x2 inverse)
    Eigen::Matrix<Scalar,2,2> A;
    A <<     a2*a2,   -a1*a1,
         -a2*a2*a2, a1*a1*a1;

    // Right hand side of interpolation system 
    Eigen::Matrix<Scalar,2,1> B;
    B(0) = fx1 - a1*gTd - fx0;
    B(1) = fx2 - a2*gTd - fx0;
    
    // Solve cubic coefficients c = [a b]
    Eigen::Matrix<Scalar,2,1> c = det * A * B; 

    // If the cubic coefficient is 0, use solution for quadratic
    // i.e. g(alpha) = b * alpha^2 + f'(x0) * alpha + f(x0)
    //      g'(alpha) = (2b * alpha + f'(x0))
    //      Set equal to zero and solve for alpha.
    if (std::abs(c(0)) < 1e-12) {
      return -gTd / (2.0*c(1));
    } else {
      // Otherwise use quadratic formula to find the minimum of the cubic
      Scalar d = std::sqrt(c(1)*c(1) - 3*c(0)*gTd);
      return (-c(1) + d) / (3.0 * c(0));
    }
  }
}

//Implementation of simple backtracking linesearch using bisection 
//Input:
//  x - initial point
//  d -  search direction
//  f - the optimization objective function f(x)
//  g - gradient of function at initial point
//  max_iterations - max line search iterations
//  alpha - max proportion of line search
//  c- sufficient decrease parameter
//  p - bisection ratio
//  Callback - callback that executes each line search iteration
//Output:
//  x - contains point to which linesearch converged, or final accepted point before termination
//  SolverExitStatus - exit code for solver 
namespace mfem {
  enum SolverExitStatus {
      CONVERGED,
      MAX_ITERATIONS_REACHED
  };

  template <int DIM, typename Scalar, typename EnergyFunc,
      typename Callback = decltype(default_linesearch_callback)>
  SolverExitStatus linesearch_backtracking(const SimState<DIM>& state,
      Scalar& alpha,  const EnergyFunc energy, Scalar c=1e-4, Scalar p=0.5,
      const Callback func = default_linesearch_callback) {
    
    unsigned int max_iterations = state.config_->ls_iters;

    // Compute gradient dot descent direction
    Scalar gTd = state.x_->gradient().dot(state.x_->delta());
    for (const auto& var : state.mixed_vars_) {
      gTd += var->gradient().dot(state.x_->delta())
           + var->gradient_mixed().dot(var->delta());
    }
    for (const auto& var : state.vars_) {
      gTd += var->gradient().dot(state.x_->delta());
    }

    Scalar fx0 = energy(0);
    Scalar fx_prev = fx0;
    Scalar alpha_prev = alpha;

    int iter = 0;
    while (iter < max_iterations) {

      Scalar fx = energy(alpha);

      // Armijo sufficient decrease condition
      Scalar fxn = fx0 + (alpha * c) * gTd;
      if (fx < fxn) {
        break;
      }

      // Scalar alpha_tmp = (iter == 0) ? (gTd / (2.0 * (fx0 + gTd - fx)))
      //     : cubic(fx0, gTd, fx, alpha, fx_prev, alpha_prev);
      // fx_prev = fx;
      // alpha_prev = alpha;
      // alpha = std::clamp(alpha_tmp, 0.1*alpha, 0.9*alpha );
      alpha = alpha * p;
      ++iter;
    }

    if(iter < max_iterations) {
      state.x_->value() += alpha * state.x_->delta();
      for (auto& var : state.mixed_vars_) {
        var->value() += alpha * var->delta();
      }
    }
    return (iter == max_iterations 
        ? SolverExitStatus::MAX_ITERATIONS_REACHED 
        : SolverExitStatus::CONVERGED);
  }
}
