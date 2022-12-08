#pragma once

#include "linear_solver.h"

namespace mfem {

  template <typename Solver, typename SystemMatrix, typename Scalar, int DIM>
  class EigenIterativeSolver : public LinearSolver<Scalar, DIM> {

    typedef LinearSolver<Scalar, DIM> Base;

  public:

    EigenIterativeSolver(SimState<DIM>* state) 
        : LinearSolver<Scalar,DIM>(state) {
      solver_.setMaxIterations(state->config_->max_iterative_solver_iters);
      solver_.setTolerance(state->config_->itr_tol);
    }

    void solve() override {
      system_matrix_.pre_solve(Base::state_);

      solver_.compute(system_matrix_.A());
      tmp_ = solver_.solve(system_matrix_.b());

      std::cout << "- CG iters: " << solver_.iterations() << std::endl;
      std::cout << "- CG error: " << solver_.error() << std::endl;
      system_matrix_.post_solve(Base::state_, tmp_);
    }

    Solver& eigen_solver() {
      return solver_;
    }

  private:

    SystemMatrix system_matrix_;
    Solver solver_;
    Eigen::VectorXx<Scalar> tmp_;
  };
}
