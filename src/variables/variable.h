#pragma once

#include "EigenTypes.h"
#include <memory>

namespace mfem {

  class Mesh;

  // Base class for fem degrees of freedom.
  template<int DIM>
  class Variable {
  public:

    Variable(std::shared_ptr<Mesh> mesh) : mesh_(mesh) {
    }

    virtual ~Variable() = default;

    // Evaluate the energy associated with the variable
    // x - variable value
    virtual double energy(const Eigen::VectorXd& x) { return 0.0; }

    // Update the state given a new set of displacements
    // x  - Nodal displacements
    // dt - Timestep size
    virtual void update(const Eigen::VectorXd& x, double dt) = 0;

    // Resets the state
    virtual void reset() = 0;

    // Pre-solve procedures
    virtual void pre_solve() {}

    // Pre-solve procedures
    virtual void post_solve() {}

    // Build and return the right-hand-side of schur-complement
    // reduced system of equations
    virtual Eigen::VectorXd rhs() = 0;

    // Gradient of the energy
    virtual Eigen::VectorXd gradient() = 0;

    // Left-hand-side of the system
    virtual const Eigen::SparseMatrix<double, Eigen::RowMajor>& lhs() = 0;

    // Returns the updates from the solves
    virtual Eigen::VectorXd& delta() = 0;

    // Returns variable values
    virtual Eigen::VectorXd& value() = 0;

    // Return number of degrees of freedom 
    virtual int size() const = 0;

    const std::shared_ptr<Mesh> mesh() const {
      return mesh_;
    }

    // Matrix vector product with hessian of variable and a vector, x
    // Output written to "out" vector
    virtual void product_hessian(const Eigen::VectorXd& x,
        Eigen::Ref<Eigen::VectorXd> out) const = 0;

  protected:

    std::shared_ptr<Mesh> mesh_;

  };

}
