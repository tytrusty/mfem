#include "variable_factory.h"
#include "variables/mixed_stretch.h"
#include "variables/stretch.h"
#include "mesh/mesh.h"

using namespace mfem;
using namespace Eigen;

template<int DIM>
MixedVariableFactory<DIM>::MixedVariableFactory() {
  // Volumetric stretch
  this->register_type(VariableType::VAR_MIXED_STRETCH,
      MixedStretch<DIM>::name(),
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<MixedVariable<DIM>>
      {return std::make_unique<MixedStretch<DIM>>(mesh);});
}

template<int DIM>
VariableFactory<DIM>::VariableFactory() {
  // Volumetric stretch
  this->register_type(VariableType::VAR_STRETCH,
      Stretch<DIM>::name(),
      [](std::shared_ptr<Mesh> mesh, std::shared_ptr<SimConfig> config)
      ->std::unique_ptr<Variable<DIM>>
      {return std::make_unique<Stretch<DIM>>(mesh);});
}

template class mfem::MixedVariableFactory<3>;
template class mfem::MixedVariableFactory<2>;
template class mfem::VariableFactory<3>;
template class mfem::VariableFactory<2>;
