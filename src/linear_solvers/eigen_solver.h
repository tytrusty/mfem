#pragma once

#include "linear_solver.h"

namespace mfem {

  template <typename Solver, typename SystemMatrix, typename Scalar, int DIM>
  class EigenSolver : public LinearSolver<Scalar, DIM> {

    typedef LinearSolver<Scalar, DIM> Base;

  public:

    EigenSolver(SimState<DIM>* state) : LinearSolver<Scalar,DIM>(state),
        has_init_(false) {}

    void solve() override {
      system_matrix_.pre_solve(Base::state_);
      //if (!has_init_) { // Can't do this for collisions since pattern changes
        solver_.analyzePattern(system_matrix_.A());
        has_init_ = true;
      //}
      solver_.factorize(system_matrix_.A());
      if (solver_.info() != Eigen::Success) {
       std::cerr << "prefactor failed! " << std::endl;
       exit(1);
      }
      tmp_ = solver_.solve(system_matrix_.b());
      system_matrix_.post_solve(Base::state_, tmp_);

    }

  private:
    // Type to represent the left-hand-side system matrix for the linear solve.
    // Different SystemMatrix types vary in their assembly process and matrix
    // structure, necessitating this abstraction.
    SystemMatrix system_matrix_;
    
    Solver solver_;
    Eigen::VectorXx<Scalar> tmp_;
    bool has_init_;

  };


}
