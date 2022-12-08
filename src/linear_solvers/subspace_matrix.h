#pragma once

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include <unsupported/Eigen/SparseExtra>
#include "simulation_state.h"

namespace Eigen {

  // Wrapped for sparse matrix to support mixed FEM KKT matrix
  template<int DIM> class SubspaceMatrix : public Eigen::EigenBase<SubspaceMatrix<DIM>> {
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

    typedef Eigen::VectorXx<Scalar> VectorType;
   
    // Assuming square system
    Index rows() const { return size(); }
    Index cols() const { return size(); }
   
    template<typename Rhs>
    Eigen::Product<SubspaceMatrix,Rhs,Eigen::AliasFreeProduct> operator*(
        const Eigen::MatrixBase<Rhs>& x) const {
      return Eigen::Product<SubspaceMatrix,Rhs,Eigen::AliasFreeProduct>(
          *this, x.derived());
    }

    void pre_solve(const mfem::SimState<DIM>* state) {
      attach_state(state);
      
      SparseMatrixdRowMajor M = state->mesh_->mass_matrix();
      VectorXd ones = VectorXd::Ones(M.cols());
      VectorXd lumped = M * ones;
      Msqrtinv_ = lumped.array().rsqrt();

      // Set rhs system
      rhs_.resize(size());

      VectorXd fp = Msqrtinv_ * -state->x_->gradient().array();

      int curr_row = 0;
      for (const auto& var : state->mixed_vars_) {
        Ref<VectorXd> out = rhs_.segment(curr_row, var->size());

        // c
        out = -var->gradient_dual();

        // - G * fp
        VectorXd tmp = -Msqrtinv_ * fp.array();
        var->product_jacobian_x(tmp, out, false);

        // G * G' * D^-1 * fy 

        // Cinv
        VectorXd C = VectorXd::Zero(var->size());
        VectorXd ones = VectorXd::Ones(var->size());
        var->product_jacobian_mixed(ones, C); // aliasing? who the fuck knows
        C = C.array().inverse();


        // fy = H^-1/2 bs
        // D^-1 = C^-1 H^1/2
        VectorXd fy = -C.array() * var->gradient_mixed().array();
        //var->product_hessian_sqrt(fy, fy);

        tmp = VectorXd::Zero(fp.size());
        var->product_jacobian_x(fy, tmp, true);
        tmp = tmp.array() * Msqrtinv_ * Msqrtinv_;
        var->product_jacobian_x(tmp, out, false);
        curr_row += var->size();
      }
      //saveMarket(rhs_, "rhs_sub.mkt");
      assert(rhs_.size() == curr_row);
    }

    void post_solve(const mfem::SimState<DIM>* state, const Eigen::VectorXd& dy)
    {

      state->x_->delta().setZero();

      VectorXd fp = Msqrtinv_ * -state->x_->gradient().array();

      int curr_row = 0;
      for (auto& var : state->mixed_vars_) {
        // Solve for s
        const VectorXd& y = dy.segment(curr_row, var->size());
        var->delta().setZero();
        var->lambda().setZero();
        var->product_hessian_sqrt_inv(y, var->delta());

        // Cinv
        VectorXd C = VectorXd::Zero(var->size());
        VectorXd ones = VectorXd::Ones(var->size());
        var->product_jacobian_mixed(ones, C);
        C = C.array().inverse();

        VectorXd gy = -var->gradient_mixed();
        VectorXd fy = VectorXd::Zero(gy.size()); 
        var->product_hessian_sqrt_inv(gy, fy);

        // Solve for lambda
        VectorXd tmp = -C.array() * (y - fy).array(); 
        var->product_hessian_sqrt(tmp, var->lambda());

        // Solve for x 
        tmp = VectorXd::Zero(fp.size());
        var->product_jacobian_x(var->lambda(),tmp,true);
        VectorXd p = -Msqrtinv_ * tmp.array() + fp.array();
        state->x_->delta() = Msqrtinv_ * p.array();
        curr_row += var->size();

        //VectorXd out(ones.size());
        //MatrixXd eye = MatrixXd::Identity(ones.size(), ones.size());
        //out = (*this) * eye.col(0);

        //saveMarket(out, "lump_sub.mkt");
        //saveMarket(tmp, "Gla_sub.mkt");
        //saveMarket(p, "p_sub.mkt");
        //saveMarket(state->x_->delta(), "x_sub.mkt");
        //saveMarket(var->delta(), "s_sub.mkt");
        //saveMarket(var->lambda(), "la_sub.mkt");
      }
    }

    const SubspaceMatrix<DIM>& A() const {
      return *this;
    }

    const VectorType& b() const {
      return rhs_;
    }
   
    // Custom API:
    SubspaceMatrix() : state_(nullptr) {}
   
    const mfem::SimState<DIM>& state() const { return *state_; }

    void attach_state(const mfem::SimState<DIM>* state) {
      state_ = state;
    }

    const ArrayXd& M() const {
      return Msqrtinv_;
    }
   
  private:

    int size() const { 
      int size = 0;
      for (const auto& var : state_->mixed_vars_) {
        size += var->size();
      }
      return size;
    }

    const mfem::SimState<DIM>* state_;

    ArrayXd Msqrtinv_;

    // linear system right hand side
    VectorType rhs_;       
  };


  template<int DIM>
  class SubspaceSystem {
  public:

    typedef SparseMatrix<double,RowMajor> MatrixType;

    void pre_solve(const mfem::SimState<DIM>* state) {
      attach_state(state);
      
      SparseMatrixdRowMajor M = state->mesh_->mass_matrix();
      VectorXd ones = VectorXd::Ones(M.cols());
      VectorXd lumped = M * ones;
      Msqrtinv_ = lumped.array().rsqrt();
      Minv_ = lumped.array().inverse();

      // Set rhs system
      int size = 0;
      for (const auto& var : state_->mixed_vars_) {
        size += var->size();
      }
      rhs_.resize(size);

      VectorXd fp = Msqrtinv_ * -state->x_->gradient().array();

      int curr_row = 0;
      for (const auto& var : state->mixed_vars_) {
        Ref<VectorXd> out = rhs_.segment(curr_row, var->size());

        // B - (n x (nele * N))
        SparseMatrix<double> B;
        var->jacobian_x(B);

        // C - (nele*N x nele*N)
        SparseMatrix<double> C;
        var->jacobian_mixed(C);

        // D^2 - (nele*N x nele*N)
        SparseMatrix<double> H;
        var->hessian_inv(H);

        // G = B * M^{-1/2}
        SparseMatrix<double> G = VectorXd(Msqrtinv_).asDiagonal() * B;
        lhs_ = G.transpose() * G + C*C*H;

        // fp = M^{-1/2} bx
        // G*fp = B * M^{-1} * bx  
        rhs_ = B.transpose() * (Minv_.asDiagonal() * -state->x_->gradient());

        // fy = H^{-1/2} by
        // D*fy = C * H^{-1} * by
        rhs_ -= C * H * var->gradient_mixed();
        rhs_ += var->gradient_dual();
        curr_row += var->size();
        //saveMarket(lhs_, "lhs2.mkt");
        //saveMarket(rhs_, "rhs2.mkt");
      }
      assert(rhs_.size() == curr_row);
    }

    void post_solve(const mfem::SimState<DIM>* state, const Eigen::VectorXd& dx)
    {
      state->x_->delta().setZero();

      VectorXd fp = Msqrtinv_ * -state->x_->gradient().array();

      int curr_row = 0;
      for (auto& var : state->mixed_vars_) {
        const VectorXd& la = dx.segment(curr_row, var->size());
        var->lambda() = la;

        // fp = M^{-1/2} bx , G = B * M^{-1/2}
        // p = fp - G' * la = M^{-1/2} (bx - B' * la)
        // dx = (M^{-1/2}p) = M^{-1} (bx - B'*la)
        var->product_jacobian_x(la,state->x_->delta(),true);
        state->x_->delta() = (Minv_.array() 
            * (-state->x_->gradient() - state->x_->delta()).array());
        
        // fy = H^{-1/2} by , D = C * H^{-1/2}
        // y = fy - D' * la = H^{-1/2} (bs - C * la)
        // ds = (H^{-1/2}p) = H^{-1} (bs - C*la)
        var->delta().setZero();
        VectorXd tmp = VectorXd::Zero(var->size());
        var->product_jacobian_mixed(la, tmp);
        tmp = -var->gradient_mixed() - tmp;
        var->product_hessian_inv(tmp, var->delta());
        curr_row += var->size();
        //saveMarket(state->x_->delta(), "x_sub.mkt");
        //saveMarket(var->delta(), "s_sub.mkt");
        //saveMarket(var->lambda(), "la_sub.mkt");
      }
    }

    const MatrixType& A() const {
      return lhs_;
    }

    const VectorXd& b() const {
      return rhs_;
    }
   
    SubspaceSystem() : state_(nullptr) {}
   
    const mfem::SimState<DIM>& state() const { return *state_; }

    void attach_state(const mfem::SimState<DIM>* state) {
      state_ = state;
    }

  private:

    const mfem::SimState<DIM>* state_;

    ArrayXd Msqrtinv_;

    // linear system right and left hand side
    MatrixType lhs_;
    VectorXd rhs_;       
    VectorXd Minv_;
  };
}

// Implementation of SubspaceMatrix * Eigen::DenseVector though a specialization
// of internal::generic_product_impl
namespace Eigen {
namespace internal {

  // SubspaceMatrix looks like a SparseMatrix, so let's inherits its traits:
  template<int DIM>
  struct traits<SubspaceMatrix<DIM>>
      :  public Eigen::internal::traits<Eigen::SparseMatrix<double>>
  {};

  // GEMV stands for matrix-vector
  template<int DIM, typename Rhs>
  struct generic_product_impl<SubspaceMatrix<DIM>, Rhs, SparseShape,
      DenseShape, GemvProduct>
      : generic_product_impl_base<SubspaceMatrix<DIM>,
      Rhs, generic_product_impl<SubspaceMatrix<DIM>,Rhs>>
  {

    typedef typename Product<SubspaceMatrix<DIM>,Rhs>::Scalar Scalar;
 
    template<typename Dest>
    static void scaleAndAddTo(Dest& dst, const SubspaceMatrix<DIM>& lhs,
        const Rhs& rhs, const Scalar& alpha) {
      assert(alpha==Scalar(1) && "scaling is not implemented");
      EIGEN_ONLY_USED_FOR_DEBUG(alpha);

      const mfem::SimState<DIM>& state = lhs.state();

      assert(state.vars_.size() == 0);

      int n = state.x_->size();

      // GGT D^-1 + D
      // Multiply dual variables against jacobians of constraint energy
      int curr_row = 0;
      for (const auto& var : state.mixed_vars_) {
        Ref<VectorXd> out = dst.segment(curr_row, var->size());
        const VectorXd& y = rhs.segment(curr_row, var->size());

        // C^-1
        VectorXd C = VectorXd::Zero(var->size());
        VectorXd ones = VectorXd::Ones(var->size());
        var->product_jacobian_mixed(ones, C);
        C = C.array().inverse();

        // D^-1 * y
        VectorXd tmpD = VectorXd::Zero(y.size());
        VectorXd y1 = C.array() * y.array();
        var->product_hessian_sqrt(y1, tmpD);
        
        // G * G' * (D^-1 * y)
        VectorXd tmp = VectorXd::Zero(n);
        var->product_jacobian_x(tmpD, tmp, true);
        tmp = tmp.array() * lhs.M() * lhs.M();
        var->product_jacobian_x(tmp, out, false);

        // + D * y 
        VectorXd y2 = VectorXd::Zero(var->size());
        var->product_hessian_sqrt_inv(y,y2);
        var->product_jacobian_mixed(y2,out); 
        //var->product_jacobian_mixed(y, y2); 
        //var->product_hessian_sqrt_inv(y2,out);
        curr_row += var->size();
      }
    }
  };
}
}
