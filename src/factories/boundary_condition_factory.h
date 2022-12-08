#pragma once

#include "factory.h"
#include "mesh/mesh.h"
#include "boundary_conditions/boundary_condition.h"

namespace mfem {
  
  class BoundaryConditionFactory : public Factory<BCScriptType,
      BoundaryCondition, Eigen::MatrixXd, BoundaryConditionConfig> {
  public:
    BoundaryConditionFactory();
  };
}