#pragma once

#include "variable.h"
#include "optimizers/optimizer_data.h"
#include "time_integrators/implicit_integrator.h"
#include "utils/sparse_utils.h"
#include "mesh/mesh.h"

namespace mfem {

  class SimConfig;

  // Nodal displacement variable
  template<int DIM>
  class Displacement : public Variable<DIM> {

    typedef Variable<DIM> Base;

  public:

    Displacement(std::shared_ptr<Mesh> mesh,
          std::shared_ptr<SimConfig> config);

    double energy(const Eigen::VectorXd& s) override;
    void update(const Eigen::VectorXd& x, double dt) override;
    void reset() override;
    void post_solve() override;

    Eigen::VectorXd rhs() override;
    Eigen::VectorXd gradient() override;

    const Eigen::SparseMatrix<double, Eigen::RowMajor>& lhs() override {
      return lhs_;
    }

    Eigen::VectorXd& delta() override {
      return dx_;
    }

    Eigen::VectorXd& value() override {
      return x_;
    }

    const std::shared_ptr<ImplicitIntegrator> integrator() const {
      return integrator_;
    }

    // "Unproject" out of reduced space with dirichlet BCs removed
    void unproject(Eigen::VectorXd& x) const {
      const auto& P = mesh_->projection_matrix();
      assert(x.size() == P.rows());
      x = P.transpose() * x + b_;
    }

    int size() const override {
      return x_.size();
    }

    void product_hessian(const Eigen::VectorXd& x,
        Eigen::Ref<Eigen::VectorXd> out) const override {
      assert(out.size() == x.size());
      out += lhs_ * x;
    }

  private:

    // Number of degrees of freedom per element
    // For DIM == 3 we have 6 DOFs per element, and
    // 3 DOFs for DIM == 2;
    static constexpr int N() {
      return DIM == 3 ? 12 : 9;
    }

    static constexpr int M() {
      return DIM * DIM;
    }

    // Matrix and vector data types
    using VecD  = Eigen::Matrix<double, DIM, 1>;   // 3x1 or 2x1
    using MatD  = Eigen::Matrix<double, DIM, DIM>; // 3x3 or 2x2
    using VecN  = Eigen::Vector<double, N()>;      // 6x1 or 3x1
    using VecM  = Eigen::Vector<double, M()>;      // 9x1
    using MatN  = Eigen::Matrix<double, N(), N()>; // 6x6 or 3x3
    using MatMN = Eigen::Matrix<double, M(), N()>; // 9x6 or 4x3


    using Base::mesh_;

    OptimizerData data_;      // Stores timing results

    std::shared_ptr<SimConfig> config_;
    std::shared_ptr<ImplicitIntegrator> integrator_;

    Eigen::SparseMatrix<double, Eigen::RowMajor> lhs_;

    Eigen::VectorXd x_;       // displacement variables
    Eigen::VectorXd b_;       // dirichlet values
    Eigen::VectorXd dx_;      // displacement deltas
    Eigen::VectorXd rhs_;     // right-hand-side vector
    Eigen::VectorXd grad_;    // Gradient with respect to 's' variables
    Eigen::VectorXd f_ext_;   // body forces
  };
}
