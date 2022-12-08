#pragma once

#include "energies/material_model.h"

namespace mfem {

  // Stable neohookean material model
  class StableNeohookean : public MaterialModel {
  public:
    
    static std::string name() {
      return "Stable-Neohookean";
    }

    StableNeohookean(const std::shared_ptr<MaterialConfig>& config)
        : MaterialModel(config) {
      std::cout << "Creating Stable-Neohookean" << std::endl;
    }

    double energy(const Eigen::Vector6d& S) override; 
    Eigen::Vector6d gradient(const Eigen::Vector6d& S) override; 
    Eigen::Matrix6d hessian(const Eigen::Vector6d& S,
        bool psd_fix = true) override;

    double energy(const Eigen::Vector9d& F) override;
    Eigen::Vector9d gradient(const Eigen::Vector9d& F) override;
    Eigen::Matrix9d hessian(const Eigen::Vector9d& F) override;

    double energy(const Eigen::Vector3d& s) override;
    Eigen::Vector3d gradient(const Eigen::Vector3d& s) override;
    Eigen::Matrix3d hessian(const Eigen::Vector3d& s) override;

    double energy(const Eigen::Vector4d& F) override;
    Eigen::Vector4d gradient(const Eigen::Vector4d& F) override;
    Eigen::Matrix4d hessian(const Eigen::Vector4d& F) override;
  };
}
