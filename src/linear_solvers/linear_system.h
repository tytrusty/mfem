#pragma once

#include "EigenTypes.h"
#include "simulation_state.h"
#include "block_matrix.h"

namespace mfem {

  template<typename Scalar>
  class SystemMatrixPD {

  public:

    typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor> MatrixType;
    typedef Eigen::VectorXx<Scalar> VectorType;

    template<int DIM>
    void pre_solve(const SimState<DIM>* state) {
      // Add LHS and RHS from each variable
      lhs_ = state->x_->lhs();
      rhs_ = state->x_->rhs();

      for (auto& var : state->vars_) {
        lhs_ += var->lhs();
        rhs_ += var->rhs();
      }
      for (auto& var : state->mixed_vars_) {
        lhs_ += var->lhs();
        rhs_ += var->rhs();
      }
    }

    template<int DIM>
    void post_solve(const SimState<DIM>* state, const Eigen::VectorXd& dx) {

      state->x_->delta() = dx;
      for (auto& var : state->mixed_vars_) {
        var->solve(dx);
      }
    }

    const MatrixType& A() const {
      return lhs_;
    }

    const VectorType& b() const {
      return rhs_;
    }


  private:
    
    // linear system left hand side
    MatrixType lhs_; 

    // linear system right hand side
    VectorType rhs_;       
  };

  template<typename Scalar, int DIM>
  class SystemMatrixIndefinite {
  public:

    typedef Eigen::BlockMatrix<DIM> MatrixType;
    typedef Eigen::VectorXx<Scalar> VectorType;

    void pre_solve(const SimState<DIM>* state) {
      lhs_.attach_state(state);

      // Set rhs system
      rhs_.resize(lhs_.rows());

      rhs_.head(state->x_->size()) = -state->x_->gradient();
      //for (const auto& var : state->mixed_vars_) {
      //  rhs_.head(state->x_->size()) -= var->gradient();
      //}

      int curr_row = state->x_->size();
      for (const auto& var : state->mixed_vars_) {
        rhs_.segment(curr_row, var->size()) = -var->gradient_mixed();
        curr_row += var->size();
        rhs_.segment(curr_row, var->size_dual()) = -var->gradient_dual();
        curr_row += var->size_dual();
      }
      assert(rhs_.size() == curr_row);
    }

    void post_solve(const SimState<DIM>* state, const Eigen::VectorXd& dx) {

      state->x_->delta() = dx.head(state->x_->size());

      int curr_row = state->x_->size();
      for (auto& var : state->mixed_vars_) {
        var->delta() = dx.segment(curr_row, var->size());
        curr_row += var->size();
        var->lambda() = dx.segment(curr_row, var->size_dual());
        curr_row += var->size_dual();
      }
    }

    const MatrixType& A() const {
      return lhs_;
    }

    const VectorType& b() const {
      return rhs_;
    }


  private:
    
    // linear system left hand side
    MatrixType lhs_; 

    // linear system right hand side
    VectorType rhs_;       
// [M 0 B']
// [0 H C']
// [B C 0 ]
// [M 0 0  B' D'][x ]
// [0 H 0  C' 0 ][s ]
// [0 0 H  0  E'][d ]
// [B C 0  0  0 ][la]
// [D 0 E  0  0 ][ga]
  };

}
