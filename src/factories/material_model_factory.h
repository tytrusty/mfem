#pragma once

#include "factory.h"
#include "config.h"
#include "energies/material_model.h"

namespace mfem {

  class MaterialModelFactory : public Factory<MaterialModelType,
      MaterialModel, std::shared_ptr<MaterialConfig>> {
  public:
    MaterialModelFactory();
  };
}
