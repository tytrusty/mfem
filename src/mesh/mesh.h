#pragma once

#include <Eigen/Dense>
#include <EigenTypes.h>
#include <memory>
#include "boundary_conditions/boundary_condition.h"

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif


namespace Eigen {
  using SparseMatrixdRowMajor = Eigen::SparseMatrix<double, Eigen::RowMajor>;
}
namespace mfem {

  class MaterialConfig;
  class MaterialModel;

  struct Element {

    Element(std::shared_ptr<MaterialModel> material)
        : material_(material) {}
    std::shared_ptr<MaterialModel> material_;
  };
  
  enum class MatrixType {
    FULL,         // Full matrix
    PROJECT_ROWS, // Project BCs from row entries
    PROJECTED     // Project out both row and columns
  };

  enum class JacobianType {
    FULL,               // Full jacobian matrix
    PROJECTED_WEIGHTED, // Projected BCs out and volume weighted 
  };

  struct SkinningData {
    Eigen::MatrixXd V_;   // vertex positions    #V by 3
    Eigen::MatrixXd TC_;  // texture coordinates #TC by 2
    Eigen::MatrixXd N_;   // corner normals      #N by 3
    Eigen::MatrixXi F_;   // face indices        #F by 3
    Eigen::MatrixXi FTC_; // face indices into texture coordinates
    Eigen::MatrixXi FN_;  // face indices into vertex normals
    Eigen::SparseMatrixdRowMajor W_; // skinning weights
    bool empty_ = true;
  };

  // Class to maintain the state and perform physics updates on an object,
  // which has a particular discretization, material, and material config
  class Mesh {
  public:

    Mesh() = default;
    
    Mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
        const Eigen::VectorXi& material_ids,
        const std::vector<std::shared_ptr<MaterialModel>>& materials);

    Mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
        std::shared_ptr<MaterialModel> material);
    
    virtual void init();

    // Compute per-element volumes. Size of "vol" is reset
    // vol - nelem x 1 per-element volumes
    virtual void volumes(Eigen::VectorXd& vol) = 0;

    // Compute mass matrix
    // M    - sparse matrix
    // vol  - nelem x 1 per-element volumes
    virtual void mass_matrix(Eigen::SparseMatrixdRowMajor& M,
        const Eigen::VectorXd& vols) = 0;


    virtual bool fixed_jacobian() { return true; }

    // Defo gradients
    virtual void deformation_gradient(const Eigen::VectorXd& x,
        Eigen::VectorXd& F) = 0;

    // Computes dF/dq jacobian matrix where F is the vectorized deformation
    // gradient, and q is the vector of position configuration    
    virtual void init_jacobian() {};
    virtual void update_jacobian(const Eigen::VectorXd& x) {}
    
    template<JacobianType T = JacobianType::PROJECTED_WEIGHTED >
    const Eigen::SparseMatrixdRowMajor& jacobian() {
      if constexpr (T == JacobianType::PROJECTED_WEIGHTED) {
        return PJW_;
      } else {
        return J_;
      }
    }

    virtual const std::vector<Eigen::MatrixXd>& local_jacobians() {
      return Jloc_;
    }

    virtual const Eigen::VectorXd& volumes() const {
      return vols_;
    }

    virtual const Eigen::MatrixXd& vertices() const {
      return V_;
    }

    Eigen::SparseMatrixdRowMajor laplacian() {
      return P_ * (J_.transpose() * W_ * J_) * P_.transpose();
    }

    template<MatrixType T = MatrixType::PROJECTED>
    const Eigen::SparseMatrixdRowMajor& mass_matrix() {
      if constexpr (T == MatrixType::PROJECTED) {
        return PMP_;
      } else if constexpr (T == MatrixType::PROJECT_ROWS) {
        return PM_;
      } else {
        return M_;
      }
    }

    const Eigen::SparseMatrixdRowMajor& projection_matrix() {
      return P_;
    }

    // Update boundary conditions
    virtual void init_bcs();
    virtual void update_bcs(double dt) {
      bc_->step(V_, dt);
    }

  public:

    std::vector<int> free_map_;
    Eigen::VectorXi is_fixed_;
    BoundaryConditionConfig bc_config_;
    std::unique_ptr<BoundaryCondition> bc_;

    Eigen::MatrixXd V_;     // Current deformed vertices
    Eigen::MatrixXd Vref_;  // Reference vertex positions
    Eigen::MatrixXd Vinit_; // Initial (deformed or undeformed) vertices
    Eigen::MatrixXd initial_velocity_; // Temporary. Just expose integrator
    Eigen::MatrixXi T_;
    Eigen::MatrixXi F_;
    Eigen::VectorXi mat_ids_;

    SkinningData skinning_data_;

    std::shared_ptr<MaterialModel> material_;
    std::vector<Element> elements_;

  protected:

    // Weighted jacobian matrix with dirichlet BCs projected out
    Eigen::SparseMatrixdRowMajor PJW_;
    Eigen::SparseMatrixdRowMajor J_;   // Shape function jacobian
    Eigen::SparseMatrixdRowMajor PMP_; // Mass matrix (dirichlet BCs projected)
    Eigen::SparseMatrixdRowMajor PM_;  // Mass matrix (rows projected)
    Eigen::SparseMatrixdRowMajor M_;   // Mass matrix (full matrix)
    Eigen::SparseMatrixdRowMajor P_;   // pinning matrix (for dirichlet BCs) 
    Eigen::SparseMatrixd W_;           // weight matrix
    Eigen::VectorXd vols_;
    std::vector<Eigen::MatrixXd> Jloc_;
  };
}

// Add discretizations
#include "mesh/tet_mesh.h"
