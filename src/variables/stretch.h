#pragma once

#include "variable.h"
#include "optimizers/optimizer_data.h"
#include "time_integrators/implicit_integrator.h"
#include "utils/sparse_utils.h"

namespace mfem {

  class SimConfig;

  // Nodal displacement variable
  template<int DIM>
  class Stretch : public Variable<DIM> {

    typedef Variable<DIM> Base;

  public:

    Stretch(std::shared_ptr<Mesh> mesh);

    static std::string name() {
      return "stretch";
    }

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
      std::cerr << "stretch::delta() unused" << std::endl;
      return grad_;
    }

    Eigen::VectorXd& value() override {
      std::cerr << "stretch::value() unused" << std::endl;
      return grad_;
    }

    int size() const override {
      return 0;
    }

    void product_hessian(const Eigen::VectorXd& x,
        Eigen::Ref<Eigen::VectorXd> out) const override {}

  private:

    // TODO need to get size of mesh
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
    int nelem_;               // number of elements

    Eigen::SparseMatrix<double, Eigen::RowMajor> lhs_;
    Eigen::SparseMatrix<double, Eigen::RowMajor> K_;   // stiffness matrix

    Eigen::VectorXd grad_;               // Gradient with respect to x
    std::vector<Eigen::VectorXd> g_;     // per-element gradients
    std::vector<Eigen::MatrixXd> H_;     // per-element hessians
    std::vector<Eigen::MatrixXd> Aloc_;
    Eigen::SparseMatrix<double, Eigen::RowMajor> A_;
    std::shared_ptr<Assembler<double,DIM,-1>> assembler_;
    std::shared_ptr<VecAssembler<double,DIM,-1>> vec_assembler_;
  };
}
