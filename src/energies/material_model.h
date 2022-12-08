#pragma once

#include <EigenTypes.h>
#include <string>
#include <memory>

namespace mfem {

  class MaterialConfig;

  // Base pure virtual class for material models
  class MaterialModel {
  public:
    
    static std::string name() {
      return "base";
    }

    MaterialModel(const std::shared_ptr<MaterialConfig>& config) 
        : config_(config) {}

    virtual ~MaterialModel() = default;

    // TODO - Don't do this, use template specializations
    //        do MatrixBase and Dimension as parmeters

    // Computes psi, the strain energy density value.
    // S - 6x1 symmetric deformation
    virtual double energy(const Eigen::Vector6d& S) = 0; 

    // Gradient with respect to symmetric deformation, S
    // S - 6x1 symmetric deformation
    virtual Eigen::Vector6d gradient(const Eigen::Vector6d& S) = 0;

    // Hessian matrix for symmetric deformation
    // S - 6x1 symmetric deformation
    virtual Eigen::Matrix6d hessian(const Eigen::Vector6d& S,
        bool psd_fix = true) = 0;

    // Optional energy for non-mixed systems
    // Computes psi, the strain energy density value.
    // F - 9x1 deformation gradient flattened (column-major)
    virtual double energy(const Eigen::Vector9d& F) {
      std::cerr << "energy unimplemented for 9x1 input" << std::endl;
      return 0;
    }

    // Non-mixed gradient with respect to deformation gradient
    // F - 9x1 deformation gradient flattened (column-major)
    virtual Eigen::Vector9d gradient(const Eigen::Vector9d& F) {
      Eigen::Vector9d g;
      std::cerr << "gradient unimplemented for 9x1 input" << std::endl;
      return g;
    }

    // Non-mixed hessian 
    // F - 9x1 deformation gradient flattened (column-major)
    virtual Eigen::Matrix9d hessian(const Eigen::Vector9d& F) {
      std::cerr << "gradient unimplemented for 9x1 input" << std::endl;
      Eigen::Matrix9d H;
      return H;
    }

    // Optional mixed energy for shells
    // Computes psi, the strain energy density value.
    // F - 9x1 deformation gradient flattened (column-major)
    virtual double energy(const Eigen::Vector3d& s) {
      std::cerr << "energy unimplemented for 3x1 input" << std::endl;
      return 0;
    }

    // shell gradient with respect to deformation gradient
    // F - 9x1 deformation gradient flattened (column-major)
    virtual Eigen::Vector3d gradient(const Eigen::Vector3d& s) {
      Eigen::Vector3d g;
      std::cerr << "gradient unimplemented for 3x1 input" << std::endl;
      return g;
    }

    // mixed shell hessian 
    // F - 9x1 deformation gradient flattened (column-major)
    virtual Eigen::Matrix3d hessian(const Eigen::Vector3d& s) {
      std::cerr << "gradient unimplemented for 3x1 input" << std::endl;
      Eigen::Matrix3d H;
      return H;
    }

    // Computes psi, the strain energy density value.
    // F - 4x1 deformation gradient flattened (column-major)
    virtual double energy(const Eigen::Vector4d& F) {
      std::cerr << "energy unimplemented for 4x1 input" << std::endl;
      return 0;
    }

    // shell gradient with respect to deformation gradient
    // F - 9x1 deformation gradient flattened (column-major)
    virtual Eigen::Vector4d gradient(const Eigen::Vector4d& F) {
      Eigen::Vector4d g;
      std::cerr << "gradient unimplemented for 4x1 input" << std::endl;
      return g;
    }

    // mixed shell hessian 
    // F - 9x1 deformation gradient flattened (column-major)
    virtual Eigen::Matrix4d hessian(const Eigen::Vector4d& F) {
      std::cerr << "gradient unimplemented for 4x1 input" << std::endl;
      Eigen::Matrix4d H;
      return H;
    }

    const std::shared_ptr<MaterialConfig>& config() const {
      return config_;
    }

  protected:

    std::shared_ptr<MaterialConfig> config_;     

  };
}
