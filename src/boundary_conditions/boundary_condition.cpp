#include "boundary_condition.h"

using namespace mfem;

void BoundaryCondition::init_boundary_groups(const Eigen::MatrixXd &V,
    std::vector<std::vector<int>> &bc_groups, double ratio, int axis) {

  // resize to match size
  Eigen::RowVectorXd bottomLeft = V.colwise().minCoeff();
  Eigen::RowVectorXd topRight = V.colwise().maxCoeff();
  Eigen::RowVectorXd range = topRight - bottomLeft;

  bc_groups.resize(2);
  for (int i = 0; i < V.rows(); i++) {
    if (V(i, axis) < bottomLeft[axis] + range[axis] * ratio) {
      bc_groups[0].emplace_back(i);
    } else if (V(i, axis) > topRight[axis] - range[axis] * ratio) {
      bc_groups[1].emplace_back(i);
    }
  }
}

void BoundaryCondition::update_free_map() {
  free_map_.resize(is_fixed_.size());
  int curr = 0;
  for (int i = 0; i < is_fixed_.size(); ++i) {
    if (is_fixed_(i) == 0) {
      free_map_(i) = curr++;
    } else {
      free_map_(i) = -1;
    }
  }  
}
