#pragma once

#include "boundary_condition.h"

namespace mfem {

  // Pins both ends along specified axis, and squash or stretches along
  // this axis.
  class BendBC : public BoundaryCondition {
  public:
    BendBC(const Eigen::MatrixXd& V,
        const BoundaryConditionConfig& config)
        : BoundaryCondition(V, config) {}

    void init(Eigen::MatrixXd& V) override {
      is_fixed_ = Eigen::VectorXi::Zero(V.rows());
      group_centers_.resize(groups_.size());
      group_velocity_.resize(groups_.size());

      for (size_t i = 0; i < groups_.size(); ++i) {

        for (int j : groups_[i]) {
          is_fixed_(j) = 1;
        }

        group_centers_[i] = V.row(groups_[i].back());
        group_velocity_[i] = std::pow(-1.0, i) * config_.velocity * M_PI;
      }
      update_free_map();
    }

    void step(Eigen::MatrixXd& V, double dt) override {
      Eigen::MatrixXd dV = Eigen::MatrixXd::Zero(V.rows(), V.cols());
      int dim = V.cols();

      for (size_t i = 0; i < groups_.size(); ++i) {

        Eigen::MatrixXd R;
        double a = group_velocity_[i];

        if (V.cols() == 3) {
          Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();
          R = Eigen::AngleAxis<double>(a * dt, axis).toRotationMatrix();
        } else {
          R = Eigen::Rotation2D<double>(a * dt).toRotationMatrix();
        }

        const Eigen::RowVectorXd& o = group_centers_[i];

        for (int j : groups_[i]) {
          dV.row(j) = (R * (V.row(j) - o).transpose()).transpose() 
                    + o - V.row(j);
        }
      }
      V += dV;
    }

  private:
    std::vector<Eigen::RowVectorXd> group_centers_; // Center of rotation
    std::vector<double> group_velocity_;            // Angular velocities
  }; 

}