#pragma once

#include "factory.h"
#include "config.h"
#include "optimizers/optimizer.h"

namespace mfem {

  class Mesh;

  template<int DIM>
  class OptimizerFactory : public Factory<OptimizerType,
      Optimizer<DIM>, SimState<DIM>&> {
  public:
    OptimizerFactory();
  };
}
