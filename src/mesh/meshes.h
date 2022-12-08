#pragma once

#include "mesh.h"

namespace mfem {

  // Mesh for a collection of disconnected meshes
  class Meshes : public Mesh {
  public:

    Meshes(const std::vector<std::shared_ptr<Mesh>>& meshes);

    virtual void init() override;
    void volumes(Eigen::VectorXd& vol) override;
    void mass_matrix(Eigen::SparseMatrixdRowMajor& M,
        const Eigen::VectorXd& vols) override;
    void deformation_gradient(const Eigen::VectorXd& x,
        Eigen::VectorXd& F) override;
    void init_jacobian() override;
    void init_bcs() override;
    void update_bcs(double dt) override;

    const std::vector<std::shared_ptr<Mesh>>& meshes() const {
      return meshes_;
    }

  protected:

    std::vector<std::shared_ptr<Mesh>> meshes_;

  };
}
