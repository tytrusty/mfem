#include "optimizer.h"
#include "mesh/mesh.h"
#include "time_integrators/BDF.h"

using namespace mfem;
using namespace Eigen;

template <int DIM>
void Optimizer<DIM>::reset() {
  state_.mesh_->init();
}

template class mfem::Optimizer<3>;
template class mfem::Optimizer<2>;
