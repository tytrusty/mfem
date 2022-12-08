#include "pinning_matrix.h"

using namespace Eigen;

SparseMatrixd pinning_matrix(const MatrixXd& V, const MatrixXi& F,
    const VectorXi& to_pin, bool kkt) {

  typedef Triplet<double> T;
  std::vector<T> trips;

  int d = V.cols();
  int row =0;
  for (int i = 0; i < V.rows(); ++i) {
    if (!to_pin(i)) {
      for (int j = 0; j < d; ++j) {
        trips.push_back(T(row++, d*i + j, 1));
      }
    }
  }

  int n = V.size();
  if (kkt) {
    for (int i = 0; i < F.rows(); ++i) {
      for (int j = 0; j < d*d; ++j) {
        trips.push_back(T(row++, n + d*d*i+j, 1));
      }
    }
    n += d*d*F.rows();
  }

  SparseMatrixd A(row, n);
  A.setFromTriplets(trips.begin(), trips.end());
  return A;
}
