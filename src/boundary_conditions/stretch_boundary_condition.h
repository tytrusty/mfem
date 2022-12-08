#pragma once

#include "boundary_condition.h"

namespace mfem {

  // Pins both ends along specified axis, and squash or stretches along
  // this axis.
  class StretchBC : public BoundaryCondition {
  public:
    StretchBC(const Eigen::MatrixXd& V,
        const BoundaryConditionConfig& config)
        : BoundaryCondition(V, config) {}

    void init(Eigen::MatrixXd& V) override {

      is_fixed_ = Eigen::VectorXi::Zero(V.rows());
      group_velocity_.resize(groups_.size());
      for (size_t i = 0; i < groups_.size(); ++i) {

        for (int j : groups_[i]) {
          is_fixed_(j) = 1;
        }
        Eigen::RowVectorXd vel(V.cols());
        vel.setZero();
        vel(config_.axis) = std::pow(-1.0, i) * config_.velocity;
        group_velocity_[i] = vel;
      }
      update_free_map();
    }

    void step(Eigen::MatrixXd& V, double dt) override {
      Eigen::MatrixXd dV = Eigen::MatrixXd::Zero(V.rows(), V.cols());
      for (size_t i = 0; i < groups_.size(); ++i) {
        for (int j : groups_[i]) {
          dV.row(j) = group_velocity_[i] * dt; 
        }
      }
      V += dV;
    }

  private:
    std::vector<Eigen::RowVectorXd> group_velocity_;
  }; 

}