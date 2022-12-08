#pragma once

#include <EigenTypes.h>

Eigen::SparseMatrixd pinning_matrix(const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& T, const Eigen::VectorXi& to_pin, bool kkt=false);
