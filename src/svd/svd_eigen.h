#pragma once

#include <EigenTypes.h>

namespace mfem {

  template <typename Scalar, int D, typename DerivedF>
  void svd(const Eigen::MatrixBase<DerivedF>& F,
      Eigen::Matrix<Scalar,D,1>& sigma, Eigen::Matrix<double,D,D>& U,
      Eigen::Matrix<Scalar,D,D>& V, bool identify_flips = true) {
    
    Eigen::JacobiSVD<Eigen::Matrix<Scalar,D,D>> svd(F,
        Eigen::ComputeFullU | Eigen::ComputeFullV);
    sigma = svd.singularValues();
    U = svd.matrixU();
    V = svd.matrixV();

    if (identify_flips) {
      Eigen::Matrix<Scalar,D,D> J = Eigen::Matrix<Scalar,D,D>::Identity();
      J(D-1, D-1) = -1.0;
      if (U.determinant() < 0.0) {
        U = U * J;
        sigma[D-1] = -sigma[D-1];
      }
      if (V.determinant() < 0.0) {
        Eigen::Matrix<Scalar,D,D> Vt = V.transpose();
        Vt = J * Vt;
        V = Vt.transpose();
        sigma[D-1] = -sigma[D-1];
      }
    }        
  }
}
