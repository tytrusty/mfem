#pragma once

#include "mesh.h"

namespace mfem {

  // Simulation Mesh for tetrahedral mesh
  class TetrahedralMesh : public Mesh {
  public:

    TetrahedralMesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
        const Eigen::VectorXi& material_ids,
        const std::vector<std::shared_ptr<MaterialModel>>& materials)
        : Mesh(V,T,material_ids,materials) {
    }

    TetrahedralMesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
        std::shared_ptr<MaterialModel> material)
        : Mesh(V,T,material) {
    }

    void volumes(Eigen::VectorXd& vol) override;
    void mass_matrix(Eigen::SparseMatrixdRowMajor& M,
        const Eigen::VectorXd& vols) override;
    void deformation_gradient(const Eigen::VectorXd& x,
        Eigen::VectorXd& F) override;
    void init_jacobian() override;

  };
}
