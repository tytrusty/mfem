#include <Eigen/Dense>
#include <EigenTypes.h>

namespace Eigen {
  using Tensor222d = std::array<std::array<Eigen::Vector2d,2>, 2>;
  using Tensor2222d = std::array<std::array<Eigen::Matrix2d,2>, 2>;
}

//NOTE: Then tensor data structure stores arrays of matrix/vectors.
//      For a 3D tensor, such as dS, the entry dS[i][j] = dS/dF_ij, which is a vector
//      For a 4D tensors, such as dV, the entry dV[i][j] = dV/dF_ij which is a matrix
//Input:
//  F - this function computes the gradient of svd(F), where F is a 3x3 matrix. 
//Output (for the definition of the tensorial types, see EigenTypes.h):
//  dU - the 3x3x3x3 derivative of U wrt to F. dU[x][y][i][j] contains the derivative of U[x][y] wrt to F[i][j]
//  dV - the 3x3x3x3 derivative of U wrt to F. dV[x][y][i][j] contains the derivative of V[x][y] wrt to F[i][j]
//  dS - the 3x3x3 derivative of the singular values wrt to F. dS[x][i][j] contains the derivative of the x^{th} singlar value wrt to F[i][j]
void dsvd(Eigen::Tensor3333d &dU, Eigen::Tensor333d  &dS,
    Eigen::Tensor3333d &dV, Eigen::Ref<const Eigen::Matrix3d> F);

// 2D version
void dsvd(Eigen::Tensor2222d &dU, Eigen::Tensor222d  &dS,
    Eigen::Tensor2222d &dV, Eigen::Ref<const Eigen::Matrix2d> F);

void dsvd(Eigen::Ref<const Eigen::Matrix3d> F, 
    Eigen::Ref<const Eigen::Matrix3d> U,
    Eigen::Ref<const Eigen::Vector3d> S,
    Eigen::Ref<const Eigen::Matrix3d> Vt,
    std::array<Eigen::Matrix3d, 9>& dR_dF);    
