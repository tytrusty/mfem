#include "tet_mesh.h"
#include <igl/volume.h>
#include "linear_tetmesh_mass_matrix.h"
#include "linear_tet_mass_matrix.h"
#include "linear_tetmesh_dphi_dX.h"
#include "energies/material_model.h"
#include "config.h"

using namespace Eigen;
using namespace mfem;

namespace {

  // From dphi/dX, form jacobian dphi/dq where
  // q is an elements local set of vertices
  template <typename Scalar>
  void local_jacobian(Matrix<Scalar,9,12>& B, const Matrix<Scalar,4,3>& dX) {
    B  << dX(0,0), 0, 0, dX(1,0), 0, 0, dX(2,0), 0, 0, dX(3,0), 0, 0,
          0, dX(0,0), 0, 0, dX(1,0), 0, 0, dX(2,0), 0, 0, dX(3,0), 0,
          0, 0, dX(0,0), 0, 0, dX(1,0), 0, 0, dX(2,0), 0, 0, dX(3,0),
          dX(0,1), 0, 0, dX(1,1), 0, 0, dX(2,1), 0, 0, dX(3,1), 0, 0, 
          0, dX(0,1), 0, 0, dX(1,1), 0, 0, dX(2,1), 0, 0, dX(3,1), 0,
          0, 0, dX(0,1), 0, 0, dX(1,1), 0, 0, dX(2,1), 0, 0, dX(3,1),
          dX(0,2), 0, 0, dX(1,2), 0, 0, dX(2,2), 0, 0, dX(3,2), 0, 0, 
          0, dX(0,2), 0, 0, dX(1,2), 0, 0, dX(2,2), 0, 0, dX(3,2), 0,
          0, 0, dX(0,2), 0, 0, dX(1,2), 0, 0, dX(2,2), 0, 0, dX(3,2);
  }

}

void TetrahedralMesh::volumes(Eigen::VectorXd& vol) {
  igl::volume(Vref_, T_, vol);
  vol = vol.cwiseAbs();
}

void TetrahedralMesh::mass_matrix(SparseMatrixdRowMajor& M,
    const VectorXd& vols) {
  VectorXd densities = VectorXd::Constant(T_.rows(),
      material_->config()->density);
  sim::linear_tetmesh_mass_matrix(M, Vref_, T_, densities, vols);
}

void TetrahedralMesh::init_jacobian() {
  Jloc_.resize(T_.rows());

  MatrixXd dphidX;
  sim::linear_tetmesh_dphi_dX(dphidX, Vref_, T_);
  std::vector<Triplet<double>> trips;

  // #pragma omp parallel for
  for (int i = 0; i < T_.rows(); ++i) { 
    // Local block
    Matrix<double,9,12> B;
    Matrix<double, 4,3> dX = sim::unflatten<4,3>(dphidX.row(i));
    local_jacobian(B, dX);
    Jloc_[i] = B;

    // Inserting triplets
    for (int j = 0; j < 9; ++j) {

      // k-th vertex of the tetrahedra
      for (int k = 0; k < 4; ++k) {
        int vid = T_(i,k); // vertex index

        // x,y,z index for the k-th vertex
        for (int l = 0; l < 3; ++l) {
          double val = B(j,3*k+l);
          trips.push_back(Triplet<double>(9*i+j, 3*vid+l, val));
        }
      }
    }
  }
  J_.resize(9*T_.rows(), V_.size());
  J_.setFromTriplets(trips.begin(),trips.end());
}

void TetrahedralMesh::deformation_gradient(const VectorXd& x, VectorXd& F) {
  assert(x.size() == J_.cols());
  F = J_ * x;
}
