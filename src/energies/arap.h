#pragma once

#include "energies/material_model.h"

namespace mfem {

  // as-rigid-as-possible material model
  class ArapModel : public MaterialModel {
  public:
    
    static std::string name() {
      return "ARAP";
    }

    ArapModel(const std::shared_ptr<MaterialConfig>& config)
        : MaterialModel(config) {
      std::cout << "Creating ARAP energy" << std::endl;

    }

    double energy(const Eigen::Vector6d& S) override; 

    Eigen::Vector6d gradient(const Eigen::Vector6d& S) override; 

    Eigen::Matrix6d hessian_inv(const Eigen::Vector6d& S);
    Eigen::Matrix6d hessian(const Eigen::Vector6d& S,
        bool psd_fix = true) override;

    double energy(const Eigen::Vector3d& s) override;
    Eigen::Vector3d gradient(const Eigen::Vector3d& s) override;
    Eigen::Matrix3d hessian(const Eigen::Vector3d& s) override;    
  };


}
