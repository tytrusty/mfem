#include "linear_solver_factory.h"
#include "mesh/mesh.h"
#include "EigenTypes.h"
#include "linear_solvers/eigen_solver.h"
#include "linear_solvers/eigen_iterative_solver.h"
#include "linear_solvers/affine_pcg.h"
#include "linear_solvers/linear_system.h"
#include "linear_solvers/preconditioners.h"
#include "linear_solvers/subspace_matrix.h"
#include <unsupported/Eigen/IterativeSolvers>

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

using namespace mfem;
using namespace Eigen;

using Scalar = double;

template<int DIM>
LinearSolverFactory<DIM>::LinearSolverFactory() {

  // Register positive definite primal condensation solvers.
  register_pd_solvers();

  // Registering indefinite solvers
  register_indefinite_solvers();
}

template<int DIM>
void LinearSolverFactory<DIM>::register_pd_solvers() {
  // Sparse matrix type
  using SpMat = SystemMatrixPD<Scalar>::MatrixType; 
  
  // Eigen LLT
  using LLT = SimplicialLLT<SpMat>;
  this->register_type(LinearSolverType::SOLVER_EIGEN_LLT, "eigen-llt",
      [](SimState<DIM>* state)
      ->std::unique_ptr<LinearSolver<Scalar, DIM>>
      {return std::make_unique<EigenSolver<LLT, SystemMatrixPD<Scalar>,
        Scalar, DIM>>(state);});

  // Eigen LDLT
  using LDLT = SimplicialLDLT<SpMat>;
  this->register_type(LinearSolverType::SOLVER_EIGEN_LDLT, "eigen-ldlt",
      [](SimState<DIM>* state)
      ->std::unique_ptr<LinearSolver<Scalar, DIM>>
      {return std::make_unique<EigenSolver<LDLT, SystemMatrixPD<Scalar>,
        Scalar, DIM>>(state);});

  // Eigen LU
  using LU = SparseLU<SpMat>;
  this->register_type(LinearSolverType::SOLVER_EIGEN_LU, "eigen-lu",
      [](SimState<DIM>* state)
      ->std::unique_ptr<LinearSolver<Scalar, DIM>>
      {return std::make_unique<EigenSolver<LU, SystemMatrixPD<Scalar>,
        Scalar, DIM>>(state);});

  #if defined(SIM_USE_CHOLMOD)
  using CHOLMOD = CholmodSupernodalLLT<SpMat>;
  this->register_type(LinearSolverType::SOLVER_CHOLMOD, "cholmod",
      [](SimState<DIM>* state)
      ->std::unique_ptr<LinearSolver<Scalar, DIM>>
      {return std::make_unique<EigenSolver<CHOLMOD, SystemMatrixPD<Scalar>,
        Scalar, DIM>>(state);});
  #endif

  // Affine Body Dynamics initialized PCG with ARAP preconditioner
//   register_type(SolverType::SOLVER_AFFINE_PCG, "affine-pcg",
//       [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
//       ->std::unique_ptr<LinearSolver<Scalar, RowMajor>>
//       {return std::make_unique<AffinePCG<Scalar, RowMajor>>(mesh, config);});

  // Eigen Conjugate gradient with diagonal preconditioner
  using EIGEN_CG_DIAG = ConjugateGradient<SpMat, Lower|Upper>;
  this->register_type(LinearSolverType::SOLVER_EIGEN_CG_DIAG, "eigen-pcg-diag",
      [](SimState<DIM>* state)
      ->std::unique_ptr<LinearSolver<Scalar, DIM>>
      { 
        return std::make_unique<EigenIterativeSolver<
          EIGEN_CG_DIAG, SystemMatrixPD<Scalar>,
          Scalar, DIM>>(state);
      }
  );

  // Eigen Conjugate gradient with incomplete cholesky preconditioner
  using SOLVER_EIGEN_CG_IC = ConjugateGradient<SpMat, Lower|Upper,
      IncompleteCholesky<Scalar>>;
  this->register_type(LinearSolverType::SOLVER_EIGEN_CG_IC, "eigen-pcg-IC",
      [](SimState<DIM>* state)
      ->std::unique_ptr<LinearSolver<Scalar, DIM>>
      { 
        return std::make_unique<EigenIterativeSolver<
          SOLVER_EIGEN_CG_IC, SystemMatrixPD<Scalar>,
          Scalar, DIM>>(state);
      }
  );
}

template<int DIM>
void LinearSolverFactory<DIM>::register_indefinite_solvers() {

  // ADMM linear system solver
  // (basically using preconditioner as solver here)
  using SOLVER_ADMM = Eigen::ADMMPreconditioner<Scalar,DIM>;
  this->register_type(LinearSolverType::SOLVER_ADMM, "admm-solver",
      [](SimState<DIM>* state)->std::unique_ptr<LinearSolver<Scalar, DIM>>
      { 
        auto solver = std::make_unique<EigenIterativeSolver<SOLVER_ADMM,
            SystemMatrixIndefinite<Scalar,DIM>, Scalar, DIM>>(state);
        solver->eigen_solver().init(state);
        return solver;
      }
  );

  // Indefinite matrix type (with matrix-vec product implemented)
  using BlockMat = typename SystemMatrixIndefinite<Scalar,DIM>::MatrixType;
  
  // Minres solver with block diagonal approximate schur-complement
  // preconditioner.
  using SOLVER_MINRES_ID = MINRES<BlockMat, Lower|Upper,
        BlockDiagonalPreconditioner<Scalar, DIM>>;
  this->register_type(LinearSolverType::SOLVER_MINRES_ID,
      "minres-indefinite-block",
      [](SimState<DIM>* state)->std::unique_ptr<LinearSolver<Scalar, DIM>>
      { 
        auto solver = std::make_unique<EigenIterativeSolver<
          SOLVER_MINRES_ID, SystemMatrixIndefinite<Scalar,DIM>,
          Scalar, DIM>>(state);
        solver->eigen_solver().preconditioner().init(state);
        return solver;
      }
  );

  // Minres solver with ADMM preconditioner
  using SOLVER_MINRES_ADMM = MINRES<BlockMat,Lower|Upper,
        ADMMPreconditioner<Scalar, DIM>>;
  this->register_type(LinearSolverType::SOLVER_MINRES_ADMM,
      "minres-indefinite-admm",
      [](SimState<DIM>* state)->std::unique_ptr<LinearSolver<Scalar, DIM>>
      { 
        auto solver = std::make_unique<EigenIterativeSolver<
          SOLVER_MINRES_ADMM, SystemMatrixIndefinite<Scalar,DIM>,
          Scalar, DIM>>(state);
        solver->eigen_solver().preconditioner().init(state);
        return solver;
      }
  );
}

template class mfem::LinearSolverFactory<3>;
template class mfem::LinearSolverFactory<2>;
