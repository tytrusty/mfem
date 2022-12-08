#pragma once

#include <EigenTypes.h>

namespace mfem {

  // Class for parallel assembly of FEM stiffness matrices
  // Each element's input is a block of size NxN composed of DIMxDIM sub-blocks
  // these sub-blocks are scattered to their global nodes positions and
  // summed with duplicates.
  template <typename Scalar, int DIM, int N>
  class Assembler {

  private:

    template<typename T>
    static void reorder(std::vector<T>& val, const std::vector<int>& indices) {
      std::vector<T> tmp(indices.size());

      for (size_t i = 0; i < indices.size(); ++i) {
        tmp[i] = val[indices[i]];
        //tmp[indices[i]] = val[i];
      }
      std::copy(tmp.begin(), tmp.end(), val.begin());
    }

    // Returns the size of the local blocks for assembly. If N is dynamic
    // M() returns -1
    static constexpr int M() {
      if (N == -1) {
        return -1;
      } else {
        return DIM * N;
      }
    }

    using MatM  = Eigen::Matrix<Scalar, M(), M()>;

  public:
    // Initialize assembler / analyze sparsity of system
    // None of this wonderfully optimized since we only have to do it once
    // E        - elements nelem x 4 for tetrahedra
    // free_map - |nnodes| maps node to its position in unpinned vector
    //            equals -1 if node is pinned
    Assembler(const Eigen::MatrixXi& E, const std::vector<int>& free_map);

    

    // Update entries of matrix using per-element blocks
    // blocks   - |nelem| N*DIM x N*DIM blocks to update assembly matrix
    void update_matrix(const std::vector<MatM>& blocks);

    // Element IDs, global, and local coordinates. Each of these vectors
    // is of the same size.
    std::vector<int> element_ids;
    std::vector<std::pair<int,int>> global_pairs;
    std::vector<std::pair<int,int>> local_pairs;

    int num_nodes; // number of unique pairs / blocks in matrix
    std::vector<int> multiplicity; // number of pairs to sum over for a node
    std::vector<int> row_offsets;
    std::vector<int> offsets;
    Eigen::SparseMatrix<Scalar, Eigen::RowMajor> A;

  };

  // Class for parallel assembly of FEM vectors
  // Scalar {double, float)}
  // DIM    {2, 3}
  // N      - Number of points per element (-1 if Dynamic)
  template <typename Scalar, int DIM, int N>
  class VecAssembler {
  public:

    // Initialize assembler / analyze sparsity of system
    // None of this wonderfully optimized since we only have to do it once
    // E        - elements nelem x 4 for tetrahedra
    // free_map - |nnodes| maps node to its position in unpinned vector
    //            equals -1 if node is pinned
    VecAssembler(const Eigen::MatrixXi& E,
        const std::vector<int>& free_map);

    // Returns the size of the local blocks for assembly. If N is dynamic
    // M() returns -1
    static constexpr int M() {
      if (N == -1) {
        return -1;
      } else {
        return DIM * N;
      }
    }

    // Assemble local products into vector
    // vecs   - |nnodes|xN*M x 1
    void assemble(const std::vector<Eigen::Matrix<Scalar,M(),1>>& vecs,
        Eigen::VectorXd& a);
  private:


    // Element IDs, global, and local coordinates. Each of these vectors
    // is of the same size.
    std::vector<int> element_ids;
    std::vector<int> global_vids;
    std::vector<int> local_vids;

    int num_nodes; // number of unique pairs / blocks in matrix
    int size_;     // size of the assembled vector
    std::vector<int> multiplicity; // number of pairs to sum over for a node
    std::vector<int> row_offsets;
    std::vector<int> offsets;
  };


  // Builds a block symmetric matrix of the form
  // P = [A B^T; B C] where C is block diagonal
  template <int N, int AOrdering, int BOrdering, int POrdering>
  void fill_block_matrix(const Eigen::SparseMatrix<double, AOrdering>& A,
      const Eigen::SparseMatrix<double, BOrdering>& B,
      const std::vector<Eigen::Matrix<double, N, N>>& C,
      Eigen::SparseMatrix<double,POrdering>& mat) {
    
    using namespace Eigen;

    mat.resize(A.rows()+B.rows(), A.rows()+B.rows());
    std::vector<Triplet<double>> trips;

    // Mass matrix terms
    for (int i = 0; i < A.outerSize(); ++i) {
      for (typename SparseMatrix<double,AOrdering>::InnerIterator it(A,i); it; ++it) {
        trips.push_back(Triplet<double>(it.row(),it.col(),it.value()));
      }
    }

    int offset = A.rows(); // offset for off diagonal blocks

    // Jacobian off-diagonal entries
    for (int i = 0; i < B.outerSize(); ++i) {
      for (typename SparseMatrix<double,BOrdering>::InnerIterator it(B, i); it; ++it) {
        trips.push_back(Triplet<double>(offset+it.row(),it.col(),it.value()));
        trips.push_back(Triplet<double>(it.col(),offset+it.row(),it.value()));
      }
    }

    // Compliance block entries
    for (int i = 0; i < C.size(); ++i) {
      
      int offset = A.rows() + i * N;

      for (int j = 0; j < N; ++j) {
        for (int k = 0; k < N; ++k) {
          trips.push_back(Triplet<double>(offset+j, offset+k, C[i](j,k)));
        }
      }

    }
    mat.setFromTriplets(trips.begin(), trips.end());
  }

  // Builds a block symmetric matrix of the form
  // P = [A 0; 0 C] where C is block diagonal
  template <int N, int Ordering>
  void fill_block_matrix(const Eigen::SparseMatrix<double,Ordering>& A,
      const std::vector<Eigen::Matrix<double, N, N>>& C,
      Eigen::SparseMatrix<double, Ordering>& mat) {
    
    using namespace Eigen;
    int m = N * C.size();
    mat.resize(A.rows()+m, A.rows()+m);
    std::vector<Triplet<double>> trips;

    // Mass matrix terms
    for (int i = 0; i < A.outerSize(); ++i) {
      for (typename SparseMatrix<double,Ordering>::InnerIterator it(A,i); it; ++it) {
        trips.push_back(Triplet<double>(it.row(),it.col(),it.value()));
      }
    }

    // Compliance block entries
    for (int i = 0; i < C.size(); ++i) {
      
      int offset = A.rows() + i * N;

      for (int j = 0; j < N; ++j) {
        for (int k = 0; k < N; ++k) {
          trips.push_back(Triplet<double>(offset+j, offset+k, C[i](j,k)));
        }
      }

    }
    mat.setFromTriplets(trips.begin(), trips.end());
  }

  template <int N>
  void fill_asym_block_matrix(const Eigen::SparseMatrixd& A,
      const Eigen::SparseMatrixd& B,
      const std::vector<Eigen::Matrix<double, N, N>>& C,
      Eigen::SparseMatrixd& mat) {
    
    using namespace Eigen;

    mat.resize(A.rows()+B.cols(), A.rows()+B.cols());
    std::vector<Triplet<double>> trips;

    // Mass matrix terms
    for (int i = 0; i < A.outerSize(); ++i) {
      for (SparseMatrixd::InnerIterator it(A,i); it; ++it) {
        trips.push_back(Triplet<double>(it.row(),it.col(),it.value()));
      }
    }

    int offset = A.rows(); // offset for off diagonal blocks

    // Jacobian off-diagonal entries
    for (int i = 0; i < B.outerSize(); ++i) {
      for (SparseMatrixd::InnerIterator it(B, i); it; ++it) {
        trips.push_back(Triplet<double>(it.row(),offset+it.col(),it.value()));
      }
    }

    // Compliance block entries
    for (int i = 0; i < C.size(); ++i) {
      
      int offset = A.rows() + i * N;

      for (int j = 0; j < N; ++j) {
        for (int k = 0; k < N; ++k) {
          trips.push_back(Triplet<double>(offset+j, offset+k, C[i](j,k)));
        }
      }

    }
    mat.setFromTriplets(trips.begin(), trips.end());
  }

  template <typename Scalar>
  void fill_block_matrix(const Eigen::SparseMatrixd& A,
      const Eigen::SparseMatrixd& B, Eigen::SparseMatrix<Scalar>& mat) {
    
    using namespace Eigen;
    mat.resize(A.rows()+B.rows(), A.cols()+B.cols());
    std::vector<Triplet<double>> trips;

    // Mass matrix terms
    for (int i = 0; i < A.outerSize(); ++i) {
      for (SparseMatrixd::InnerIterator it(A,i); it; ++it) {
        trips.push_back(Triplet<double>(it.row(),it.col(),it.value()));
      }
    }

    int offset = A.rows();
    for (int i = 0; i < B.outerSize(); ++i) {
      for (SparseMatrixd::InnerIterator it(B,i); it; ++it) {
        trips.push_back(Triplet<double>(offset + it.row(),
            offset + it.col(),it.value()));
      }
    }
    mat.setFromTriplets(trips.begin(), trips.end());
  }


  template <int R, int C>
  void init_block_diagonal(Eigen::SparseMatrixd& mat, int N) {
    mat.resize(R*N, C*N);
    mat.reserve(Eigen::VectorXi::Constant(C*N,R));

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < C; ++j) {
        for (int k = 0; k < R; ++k) {
          mat.insert(R*i + k, C*i + j) = 0;
        }
      }
    }
  }

  template <int R, int C>
  void update_block_diagonal(std::vector<Eigen::Matrix<double, R, C>> data,
      Eigen::SparseMatrixd& mat) {

    int N = data.size();
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
      
      int start = R*C*i;

      for (int j = 0; j < C; ++j) {
        for (int k = 0; k < R; ++k) {
          mat.valuePtr()[start + j*R + k] = data[i](k,j);
        }
      }
    }
  }


  // Build block diagonal matrix
  template <typename Scalar, int Ordering>
  void build_block_diagonal(Eigen::SparseMatrix<Scalar, Ordering>& B,
      const std::vector<Eigen::SparseMatrix<Scalar, Ordering>>& A) {
    
    using Iterator = typename Eigen::SparseMatrix<Scalar, Ordering>::InnerIterator;
    size_t nrows = 0, ncols = 0;

    for (size_t i = 0; i < A.size(); ++i) {
      assert(A[i].rows() > 0 && A[i].cols() > 0);
      nrows += A[i].rows();
      ncols += A[i].cols();
    }
    
    B.resize(nrows, ncols);

    // This could be parallelized easily
    std::vector<Eigen::Triplet<double>> trips;

    size_t row_offset = 0;
    size_t col_offset = 0;

    for (size_t i = 0; i < A.size(); ++i) { 
      for (int j = 0; j < A[i].outerSize(); ++j) {
        for (Iterator it(A[i],j); it; ++it) {
          trips.push_back(Eigen::Triplet<double>(row_offset + it.row(),
              col_offset + it.col(), it.value()));
        }
      }
      row_offset += A[i].rows();
      col_offset += A[i].cols();
    }
    B.setFromTriplets(trips.begin(), trips.end());
  }

}