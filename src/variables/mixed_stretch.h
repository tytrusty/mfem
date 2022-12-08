#pragma once

#include "mixed_variable.h"
#include "optimizers/optimizer_data.h"
#include "utils/sparse_utils.h"

namespace mfem {

  // Variable for DIMxDIM symmetric deformation stretch matrix
  // from polar decomposition of deformation gradient (F = RS) 
  template<int DIM>
  class MixedStretch : public MixedVariable<DIM> {

    typedef MixedVariable<DIM> Base;

  public:

    MixedStretch(std::shared_ptr<Mesh> mesh) : MixedVariable<DIM>(mesh)
    {}

    static std::string name() {
      return "mixed-stretch";
    }

    double energy(const Eigen::VectorXd& x, const Eigen::VectorXd& s) override;
    double constraint_value(const Eigen::VectorXd& x,
        const Eigen::VectorXd& s) override;
    void update(const Eigen::VectorXd& x, double dt) override;
    void reset() override;
    void post_solve() override;

    Eigen::VectorXd rhs() override;
    Eigen::VectorXd gradient() override;
    Eigen::VectorXd gradient_mixed() override;
    Eigen::VectorXd gradient_dual() override;

    const Eigen::SparseMatrix<double, Eigen::RowMajor>& lhs() override {
      return A_;
    }

    void solve(const Eigen::VectorXd& dx) override;

    Eigen::VectorXd& delta() override {
      return ds_;
    }

    Eigen::VectorXd& value() override {
      return s_;
    }

    Eigen::VectorXd& lambda() override {
      return la_;
    }

    int size() const override {
      return s_.size();
    }

    int size_dual() const override {
      return la_.size();
    }

    void evaluate_constraint(const Eigen::VectorXd& x,
        Eigen::VectorXd&) override;
    void hessian(Eigen::SparseMatrix<double>&) override;
    void hessian_inv(Eigen::SparseMatrix<double>&) override;
    void jacobian_x(Eigen::SparseMatrix<double>&) override;
    void jacobian_mixed(Eigen::SparseMatrix<double>&) override;

    void product_hessian(const Eigen::VectorXd& x,
        Eigen::Ref<Eigen::VectorXd> out) const override;
    void product_hessian_inv(const Eigen::VectorXd& x,
        Eigen::Ref<Eigen::VectorXd> out) const override; 
    void product_jacobian_x(const Eigen::VectorXd& x,
        Eigen::Ref<Eigen::VectorXd> out, bool transposed) const override;
    void product_jacobian_mixed(const Eigen::VectorXd& x,
        Eigen::Ref<Eigen::VectorXd> out) const override;
    void product_hessian_sqrt(const Eigen::VectorXd& x,
        Eigen::Ref<Eigen::VectorXd> out) const override;
    void product_hessian_sqrt_inv(const Eigen::VectorXd& x,
        Eigen::Ref<Eigen::VectorXd> out) const override;

  protected:

    void update_rotations(const Eigen::VectorXd& x);
    void update_derivatives(double dt);

  private:

    // Number of degrees of freedom per element
    // For DIM == 3 we have 6 DOFs per element, and
    // 3 DOFs for DIM == 2;
    static constexpr int N() {
      return DIM == 3 ? 6 : 3;
    }

    static constexpr int M() {
      return DIM * DIM;
    }

    // Matrix and vector data types
    using MatD  = Eigen::Matrix<double, DIM, DIM>; // 3x3 or 2x2
    using VecN  = Eigen::Vector<double, N()>;      // 6x1 or 3x1
    using VecM  = Eigen::Vector<double, M()>;      // 9x1
    using MatM  = Eigen::Matrix<double, M(), M()>; // 9x9
    using MatN  = Eigen::Matrix<double, N(), N()>; // 6x6 or 3x3
    using MatMN = Eigen::Matrix<double, M(), N()>; // 9x6 or 4x3

    // Nx1 vector reprenting identity matrix
    static constexpr VecN Ivec() {
      VecN I; 
      if constexpr (DIM == 3) {
        I << 1,1,1,0,0,0;
      } else {
        I << 1,1,0;
      }
      return I;
    }

    static constexpr MatN Sym() {
      MatN m; 
      if constexpr (DIM == 3) {
        m = (VecN() << 1,1,1,2,2,2).finished().asDiagonal();
      } else {
        m = (VecN() << 1,1,2).finished().asDiagonal();
      }
      return m;
    }

    static constexpr MatN Syminv() {
      MatN m; 
      if constexpr (DIM == 3) {
        m = (VecN() << 1,1,1,0.5,0.5,0.5).finished().asDiagonal();
      } else {
        m = (VecN() << 1,1,0.5).finished().asDiagonal();
      }
      return m;
    }

    using Base::mesh_;

    OptimizerData data_;      // Stores timing results
    int nelem_;               // number of elements
    Eigen::VectorXd s_;       // deformation variables
    Eigen::VectorXd ds_;      // deformation variables deltas
    Eigen::VectorXd la_;      // lagrange multipliers
    Eigen::VectorXd rhs_;     // RHS for schur complement system
    Eigen::VectorXd grad_;    // Gradient with respect to 's' variables
    Eigen::VectorXd grad_x_;  // Gradient with respect to 'x' variables
    Eigen::VectorXd grad_la_; // Gradient with respect to dual variables
    Eigen::VectorXd gl_;      // tmp var: g_\Lambda in the notes
    Eigen::VectorXd Jdx_;     // tmp var: Jacobian multiplied by dx
    std::vector<MatD> R_;     // per-element rotations
    std::vector<VecN> S_;     // per-element deformation
    std::vector<VecN> g_;     // per-element gradients
    std::vector<MatN> H_;     // per-element hessians
    std::vector<MatN> Hloc_;  // per-element hessians
    std::vector<MatN> Hinv_;  // per-element hessian inverse
    std::vector<MatMN> dSdF_; 
    std::vector<Eigen::MatrixXd> Aloc_;
    Eigen::SparseMatrix<double, Eigen::RowMajor> A_;
    std::shared_ptr<Assembler<double,DIM,-1>> assembler_;
  };
}
