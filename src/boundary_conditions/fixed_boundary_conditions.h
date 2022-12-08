#pragma once

#include "boundary_condition.h"

namespace mfem {

  // Boundary condition (BC) with no fixed points. Basically a no-op used for
  // default BCs when none are specified.
  class NullBC : public BoundaryCondition {
  public:
    NullBC(const Eigen::MatrixXd& V, const BoundaryConditionConfig& config)
        : BoundaryCondition(V, config) {
      int N = V.rows();
      is_fixed_ = Eigen::VectorXi::Zero(N);
      free_map_ = Eigen::VectorXi::LinSpaced(N, 0, N-1);
    }
  };

  // BC with no fixed points, but scales the vertices of the mesh, inducing
  // an initial deformation.
  class ScaleBC : public NullBC {
  public:
    ScaleBC(const Eigen::MatrixXd& V, const BoundaryConditionConfig& config)
        : NullBC(V, config) {}

    void init(Eigen::MatrixXd& V) override {
      int N = V.cols();
      Eigen::RowVectorXd bmin = V.colwise().minCoeff();
      Eigen::RowVectorXd bmax = V.colwise().maxCoeff();
      Eigen::RowVectorXd offset = 0.5 * (bmin + bmax);
      V = (V.rowwise() - offset) * 1.5;
      V.rowwise() += offset;
    }
  };

  // BC with no fixed points, but randomizes the vertex positions.
  class RandomizeBC : public NullBC {
  public:
    RandomizeBC(const Eigen::MatrixXd& V,
        const BoundaryConditionConfig& config)
        : NullBC(V, config) {}

    void init(Eigen::MatrixXd& V) override {
      Eigen::RowVectorXd bmin = V.colwise().minCoeff();
      Eigen::RowVectorXd bmax = V.colwise().maxCoeff();
      Eigen::RowVectorXd offset = 0.5 * (bmin + bmax);
      Eigen::MatrixXd tmp = V;
      V.setRandom();
      V /= 2.0;
      offset(1) += (bmax(1) - bmin(1)) * 0.5;
      offset -= V.row(0);
      V.rowwise() += offset;
    }
  };

  // BC for pinning a single point.
  class OnePointBC : public BoundaryCondition {
  public:
    OnePointBC(const Eigen::MatrixXd& V,
        const BoundaryConditionConfig& config)
        : BoundaryCondition(V, config) {
      int N = V.rows();
      is_fixed_ = Eigen::VectorXi::Zero(N);
      is_fixed_(0) = 1;
      update_free_map();
    }
  };

  // Pins two points at opposing ends of the mesh.
  class HangBC : public BoundaryCondition {
  public:
    HangBC(const Eigen::MatrixXd& V, const BoundaryConditionConfig& config)
        : BoundaryCondition(V, config) {
      
      is_fixed_ = Eigen::VectorXi::Zero(V.rows());
      for (const auto& group : groups_) {
        is_fixed_(group.back()) = 1;
      }
      update_free_map();
    }
  };

  // Pins one end of the mesh
  class HangEndsBC : public BoundaryCondition {
  public:
    HangEndsBC(const Eigen::MatrixXd& V,
        const BoundaryConditionConfig& config)
        : BoundaryCondition(V, config) {
      is_fixed_ = Eigen::VectorXi::Zero(V.rows());

      for (int i : groups_[group_id_]) {
        is_fixed_(i) = 1;
      }
      update_free_map();
    }
  private:
    int group_id_ = 1;
  };
}
