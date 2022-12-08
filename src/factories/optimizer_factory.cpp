#include "optimizer_factory.h"
#include "optimizers/optimizer.h"
#include "optimizers/newton_optimizer.h"
#include "mesh/mesh.h"

using namespace mfem;
using namespace Eigen;

template<int DIM>
OptimizerFactory<DIM>::OptimizerFactory() {
  // Newton's
  this->register_type(OptimizerType::OPTIMIZER_NEWTON,
      NewtonOptimizer<DIM>::name(),
      [](SimState<DIM>& state)->std::unique_ptr<Optimizer<DIM>>
      {return std::make_unique<NewtonOptimizer<DIM>>(state);});
}

template class mfem::OptimizerFactory<3>;
template class mfem::OptimizerFactory<2>;
