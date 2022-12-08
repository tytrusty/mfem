#pragma once

#include "EigenTypes.h"
#include "variable.h"

namespace mfem {

  class Mesh;

  // Base class for mixed fem variables.
  template<int DIM>
  class MixedVariable : public Variable<DIM> {
  public:
    
    MixedVariable(std::shared_ptr<Mesh> mesh) : Variable<DIM>(mesh) {
    }

    // Evaluate the energy associated with the variable
    // x - Nodal displacements 
    // y - Mixed variable
    virtual double energy(const Eigen::VectorXd& x,
        const Eigen::VectorXd& y) = 0;

    // Evaluate the energy associated with the mixed variable constraint
    // x - Nodal displacements 
    // y - Mixed variable
    virtual double constraint_value(const Eigen::VectorXd& x,
        const Eigen::VectorXd& y) = 0;

    // Update the state given a new set of displacements
    // x  - Nodal displacements
    // dt - Timestep size
    virtual void update(const Eigen::VectorXd& x, double dt) = 0;

    // Gradient of the energy with respect to mixed variable
    virtual Eigen::VectorXd gradient_mixed() = 0;

    // Gradient of the energy with respect to dual variable
    virtual Eigen::VectorXd gradient_dual() = 0;

    // Given the solution for displacements, solve the updates of the
    // mixed variables
    // dx - nodal displacement deltas
    virtual void solve(const Eigen::VectorXd& dx) = 0;

    // Returns lagrange multipliers
    virtual Eigen::VectorXd& lambda() = 0;

    // Returns number of dual variables
    virtual int size_dual() const = 0;

    virtual void evaluate_constraint(const Eigen::VectorXd&,
        Eigen::VectorXd&) {}
    virtual void hessian(Eigen::SparseMatrix<double>&) {}
    virtual void hessian_inv(Eigen::SparseMatrix<double>&) {}
    virtual void jacobian_x(Eigen::SparseMatrix<double>&) {}
    virtual void jacobian_mixed(Eigen::SparseMatrix<double>&) {}

    // Matrix vector product with hessian of variable and a vector, x
    // Output written to "out" vector
    virtual void product_jacobian_x(const Eigen::VectorXd& x,
        Eigen::Ref<Eigen::VectorXd> out, bool transposed) const = 0;
    virtual void product_jacobian_mixed(const Eigen::VectorXd& x,
        Eigen::Ref<Eigen::VectorXd> out) const = 0;

    virtual void product_hessian_inv(const Eigen::VectorXd& x,
        Eigen::Ref<Eigen::VectorXd> out) const {
      std::cout << "product_hessian_inv unimplemented!" << std::endl;
    }

    virtual void product_hessian_sqrt(const Eigen::VectorXd& x,
        Eigen::Ref<Eigen::VectorXd> out) const { 
      std::cout << "product_hessian_sqrt unimplemented!" << std::endl;
    }
    virtual void product_hessian_sqrt_inv(const Eigen::VectorXd& x,
        Eigen::Ref<Eigen::VectorXd> out) const { 
      std::cout << "product_hessian_sqrt_inv unimplemented!" << std::endl;
    }
  };

}
