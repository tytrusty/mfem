#pragma once

#include <EigenTypes.h>
#include <set>

namespace mfem {

  // Converts Young's modulus and Poisson's ratio to Lame parameters
  // E      - young's modulus (in pascals)
  // nu     - poisson's ratio
  // lambda - lame first parameter
  // mu     - lame second parameter
  constexpr void Enu_to_lame(double E, double nu, double& lambda, double& mu) {
    mu = E/(2.0*(1.0+nu));
    lambda = (E*nu)/((1.0+nu)*(1.0-2.0*nu));
  }

  enum VariableType {
    VAR_DISPLACEMENT,
    VAR_STRETCH,
    VAR_COLLISION,
    VAR_FRICTION,
    VAR_MIXED_STRETCH,
    VAR_MIXED_COLLISION
  };

  enum OptimizerType {
    OPTIMIZER_ALM,
    OPTIMIZER_ADMM,
    OPTIMIZER_SQP,
    OPTIMIZER_SQP_PD,
    OPTIMIZER_NEWTON,
    OPTIMIZER_SQP_BENDING,
  };

  enum TimeIntegratorType {
    TI_BDF1,
    TI_BDF2,
    TI_BDF3,
    TI_BDF4,
    TI_BDF5,
    TI_BDF6
  };

  enum LinearSolverType {
    SOLVER_EIGEN_LLT,
    SOLVER_EIGEN_LDLT,
    SOLVER_EIGEN_LU,
    SOLVER_CHOLMOD,
    SOLVER_AFFINE_PCG,
    SOLVER_EIGEN_CG_DIAG,
    SOLVER_EIGEN_CG_IC,
    SOLVER_EIGEN_GS,
    SOLVER_MINRES_ID,
    SOLVER_MINRES_ADMM,
    SOLVER_ADMM,
    SOLVER_SUBSPACE,
    SOLVER_AMGCL
  };

  enum MaterialModelType { 
    MATERIAL_SNH,   // Stable neohookean
    MATERIAL_NH,    // neohookean
    MATERIAL_COROT, // corotational
    MATERIAL_FCR,   // fixed-corotational
    MATERIAL_ARAP,  // as-rigid-as possible
    MATERIAL_FUNG   // exponential
  };

  enum BCScriptType {
    BC_NULL,
    BC_SCALEF,
    BC_HANG,
    BC_HANGENDS,
    BC_STRETCH,
    BC_SQUASH,
    BC_STRETCHNSQUASH,
    BC_BEND,
    BC_TWIST,
    BC_TWISTNSTRETCH,
    BC_TWISTNSNS,
    BC_TWISTNSNS_OLD,
    BC_RUBBERBANDPULL,
    BC_ONEPOINT,
    BC_RANDOM,
    BC_FALL,
  };
  
  
  // Global parameters for the simulation
  struct SimConfig {
    double h = 0.034; 
    float ext[3] = {0., -9.8, 0.};
    bool show_timing = true;
    bool show_data = true;
    bool save_substeps = false;
    int outer_steps = 5;
    int ls_iters = 20;

    double mu = 0.5;
    double espv = 1e-3;
    double kappa = 10.0;
    double max_kappa = 1e6;
    double constraint_tol = 1e-2;
    bool enable_ccd = false;
    double dhat = 1e-2;

    // update kappa and lambda if residual below this tolerance
    double update_zone_tol = 1e-1; 

    double newton_tol = 1e-10;
    double ls_tol = 1e-4;
    OptimizerType optimizer = OPTIMIZER_NEWTON;
    int max_iterative_solver_iters = 500;
    double itr_tol = 1e-4;
    LinearSolverType solver_type = SOLVER_EIGEN_LLT;
    TimeIntegratorType ti_type = TI_BDF1;
    std::set<VariableType> variables = {};
    std::set<VariableType> mixed_variables = {
      VAR_MIXED_STRETCH,
      VAR_MIXED_COLLISION
    };

  };

  // Simple config for material parameters for a single object
  struct MaterialConfig {
    MaterialModelType material_model = MATERIAL_SNH;
    double ym = 1e6;
    double pr = 0.45;
    double mu = ym/(2.0*(1.0+pr));
    double la = (ym*pr)/((1.0+pr)*(1.0-2.0*pr));
    double density = 1000.0;
    double thickness = 1e-3;
  };

  struct BoundaryConditionConfig {
    BCScriptType type = BCScriptType::BC_NULL;
    double ratio = 0.1;
    int axis = 0;
    double velocity = 0.1;
  };

}
