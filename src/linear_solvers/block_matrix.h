#pragma once 

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include "simulation_state.h"

namespace Eigen {

  // Wrapped for sparse matrix to support mixed FEM KKT matrix
  template<int DIM>
  class BlockMatrix : public Eigen::EigenBase<BlockMatrix<DIM>> {
  
  public:
    // Required typedefs, constants, and method:
    typedef double Scalar;
    typedef double RealScalar;
    typedef int StorageIndex;
    enum {
      ColsAtCompileTime = Eigen::Dynamic,
      MaxColsAtCompileTime = Eigen::Dynamic,
      IsRowMajor = true
    };
   
    // Assuming square system
    Index rows() const { return state_->size(); }
    Index cols() const { return state_->size(); }
   
    template<typename Rhs>
    Eigen::Product<BlockMatrix,Rhs,Eigen::AliasFreeProduct> operator*(
        const Eigen::MatrixBase<Rhs>& x) const {
      return Eigen::Product<BlockMatrix,Rhs,Eigen::AliasFreeProduct>(
          *this, x.derived());
    }
   
    // Custom API:
    BlockMatrix() : state_(nullptr) {}
   
    const mfem::SimState<DIM>& state() const { return *state_; }

    void attach_state(const mfem::SimState<DIM>* state) {
      state_ = state;
    }
   
  private:
    const mfem::SimState<DIM>* state_;
  };
}

// Implementation of BlockMatrix * Eigen::DenseVector though a specialization
// of internal::generic_product_impl
namespace Eigen {
namespace internal {

  // BlockMatrix looks like a SparseMatrix, so let's inherits its traits:
  template<int DIM>
  struct traits<BlockMatrix<DIM>>
      :  public Eigen::internal::traits<Eigen::SparseMatrix<double>>
  {};

  // GEMV stands for matrix-vector
  template<int DIM, typename Rhs>
  struct generic_product_impl<BlockMatrix<DIM>, Rhs, SparseShape,
      DenseShape, GemvProduct>
      : generic_product_impl_base<BlockMatrix<DIM>,
      Rhs, generic_product_impl<BlockMatrix<DIM>,Rhs>>
  {

    typedef typename Product<BlockMatrix<DIM>,Rhs>::Scalar Scalar;
 
    template<typename Dest>
    static void scaleAndAddTo(Dest& dst, const BlockMatrix<DIM>& lhs,
        const Rhs& rhs, const Scalar& alpha) {

      // This method should implement "dst += alpha * lhs * rhs" inplace,
      // however, for iterative solvers, alpha is always equal to 1,
      // so let's not bother about it.
      assert(alpha==Scalar(1) && "scaling is not implemented");
      EIGEN_ONLY_USED_FOR_DEBUG(alpha);

      //std::cout << "HELLO?" << " dst norm: " << dst.norm() << std::endl;
      const mfem::SimState<DIM>& state = lhs.state();
      const VectorXd& dx = rhs.head(state.x_->size());

      state.x_->product_hessian(dx, dst.head(state.x_->size()));

      // For any non-mixed anergies just take product of delta-x and
      // the position-based hessian.
      for (const auto& var : state.vars_) {
        var->product_hessian(dx, dst.head(state.x_->size()));
      }

      // Multiply dual variables against jacobians of constraint energy
      int curr_row = state.x_->size();
      for (const auto& var : state.mixed_vars_) {
        curr_row += var->size();
        const VectorXd& la = rhs.segment(curr_row, var->size_dual());
        var->product_jacobian_x(la, dst.head(state.x_->size()), true);
        curr_row += var->size_dual();
      }

      // Accumulate products for product entries corresponding to
      // mixed variables & their dual variables.
      curr_row = state.x_->size();
      for (const auto& var : state.mixed_vars_) {

        const VectorXd& ds = rhs.segment(curr_row, var->size());
        const VectorXd& la = rhs.segment(curr_row + var->size(),
            var->size_dual());

        // Terms for mixed variable
        Ref<VectorXd> out = dst.segment(curr_row, var->size());
        var->product_hessian(ds, out);
        var->product_jacobian_mixed(la, out);
        curr_row += var->size();

        // Terms for dual variable
        Ref<VectorXd> out_la = dst.segment(curr_row, var->size_dual());
        var->product_jacobian_x(dx, out_la, false);
        var->product_jacobian_mixed(ds, out_la);
        curr_row += var->size_dual();
      }
    }
  };
}
}
