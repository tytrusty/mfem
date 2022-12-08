#include "energies/material_model.h"
#include "energies/neohookean.h"
#include "energies/corotational.h"
#include "energies/fixed_corotational.h"
#include "energies/arap.h"
#include "energies/stable_neohookean.h"
#include "energies/fung.h"
#include "factories/material_model_factory.h"

using namespace mfem;

MaterialModelFactory::MaterialModelFactory() {
  
  // ARAP
  register_type(MaterialModelType::MATERIAL_ARAP, ArapModel::name(),
      [](std::shared_ptr<MaterialConfig> config)
      ->std::unique_ptr<MaterialModel>
      {return std::make_unique<ArapModel>(config);});

  // Corotated Elasticity
  register_type(MaterialModelType::MATERIAL_COROT, Corotational::name(),
      [](std::shared_ptr<MaterialConfig> config)
      ->std::unique_ptr<MaterialModel>
      {return std::make_unique<Corotational>(config);});

  // Fixed Corotated Elasticity
  register_type(MaterialModelType::MATERIAL_FCR, FixedCorotational::name(),
      [](std::shared_ptr<MaterialConfig> config)
      ->std::unique_ptr<MaterialModel>
      {return std::make_unique<FixedCorotational>(config);});

  // Fung
  register_type(MaterialModelType::MATERIAL_FUNG, Fung::name(),
      [](std::shared_ptr<MaterialConfig> config)
      ->std::unique_ptr<MaterialModel>
      {return std::make_unique<Fung>(config);});   

  // Neohookean
  register_type(MaterialModelType::MATERIAL_NH, Neohookean::name(),
      [](std::shared_ptr<MaterialConfig> config)
      ->std::unique_ptr<MaterialModel>
      {return std::make_unique<Neohookean>(config);});   

  // Stable Neohookean
  register_type(MaterialModelType::MATERIAL_SNH, StableNeohookean::name(),
      [](std::shared_ptr<MaterialConfig> config)
      ->std::unique_ptr<MaterialModel>
      {return std::make_unique<StableNeohookean>(config);});   
}
