#pragma once

#include "mesh/mesh.h"

namespace mfem {

  // Simulation Mesh for triangle mesh
  class Tri2DMesh : public Mesh {
  public:

    Tri2DMesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
        std::shared_ptr<MaterialModel> material);

    virtual void volumes(Eigen::VectorXd& vol) override;
    virtual void mass_matrix(Eigen::SparseMatrixdRowMajor& M,
        const Eigen::VectorXd& vols) override;
    void deformation_gradient(const Eigen::VectorXd& x,
        Eigen::VectorXd& F) override;

    void init_jacobian() override;

    bool fixed_jacobian() override { 
      return true;
    }
    const std::vector<Eigen::MatrixXd>& local_jacobians() override{
      return Jloc_;
    }

    // virtual Eigen::MatrixXd vertices() override;

  private:
    Eigen::MatrixXd dphidX_;

  };
}
