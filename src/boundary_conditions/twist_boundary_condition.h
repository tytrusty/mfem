#pragma once

#include "boundary_condition.h"

namespace mfem {

  // Pins both ends along specified axis, and twists along axis
  class TwistBC : public BoundaryCondition {
  public:
    TwistBC(const Eigen::MatrixXd& V,
        const BoundaryConditionConfig& config)
        : BoundaryCondition(V, config) {}

    void init(Eigen::MatrixXd& V) override {
      is_fixed_ = Eigen::VectorXi::Zero(V.rows());
      group_velocity_.resize(groups_.size());

      Eigen::RowVectorXd bmin = V.colwise().minCoeff();
      Eigen::RowVectorXd bmax = V.colwise().maxCoeff();
      center_ = 0.5 * (bmin + bmax);

      for (size_t i = 0; i < groups_.size(); ++i) {

        for (int j : groups_[i]) {
          is_fixed_(j) = 1;
        }
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
          Eigen::Vector3d axis = Eigen::Vector3d::UnitX();
          R = Eigen::AngleAxis<double>(a * dt, axis).toRotationMatrix();
        } else {
          R = Eigen::Rotation2D<double>(a * dt).toRotationMatrix();
        }

        for (int j : groups_[i]) {
          dV.row(j) = (R * (V.row(j) - center_).transpose()).transpose() 
                    + center_ - V.row(j);
        }
      }
      V += dV;
    }

  private:
    Eigen::RowVectorXd center_;
    std::vector<double> group_velocity_; // Angular velocities
  }; 

}