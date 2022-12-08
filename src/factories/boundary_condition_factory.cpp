#include "boundary_condition_factory.h"
#include "boundary_conditions/fixed_boundary_conditions.h"
#include "boundary_conditions/bend_boundary_condition.h"
#include "boundary_conditions/stretch_boundary_condition.h"
#include "boundary_conditions/twist_boundary_condition.h"

using namespace mfem;
using namespace Eigen;

BoundaryConditionFactory::BoundaryConditionFactory() {

  register_type(BCScriptType::BC_NULL, "null", 
      [](Eigen::MatrixXd V, BoundaryConditionConfig cfg)
      ->std::unique_ptr<BoundaryCondition>
      {return std::make_unique<NullBC>(V, cfg);});

  register_type(BCScriptType::BC_SCALEF, "scale", 
      [](Eigen::MatrixXd V, BoundaryConditionConfig cfg)
      ->std::unique_ptr<BoundaryCondition>
      {return std::make_unique<ScaleBC>(V, cfg);});

  register_type(BCScriptType::BC_RANDOM, "randomize", 
      [](Eigen::MatrixXd V, BoundaryConditionConfig cfg)
      ->std::unique_ptr<BoundaryCondition>
      {return std::make_unique<RandomizeBC>(V, cfg);});

  register_type(BCScriptType::BC_ONEPOINT, "onepoint", 
      [](Eigen::MatrixXd V, BoundaryConditionConfig cfg)
      ->std::unique_ptr<BoundaryCondition>
      {return std::make_unique<OnePointBC>(V, cfg);});

  register_type(BCScriptType::BC_HANG, "hang", 
      [](Eigen::MatrixXd V, BoundaryConditionConfig cfg)
      ->std::unique_ptr<BoundaryCondition>
      {return std::make_unique<HangBC>(V, cfg);});

  register_type(BCScriptType::BC_HANGENDS, "hangends", 
      [](Eigen::MatrixXd V, BoundaryConditionConfig cfg)
      ->std::unique_ptr<BoundaryCondition>
      {return std::make_unique<HangEndsBC>(V, cfg);});
  
  register_type(BCScriptType::BC_BEND, "bend", 
      [](Eigen::MatrixXd V, BoundaryConditionConfig cfg)
      ->std::unique_ptr<BoundaryCondition>
      {return std::make_unique<BendBC>(V, cfg);});

  register_type(BCScriptType::BC_STRETCH, "stretch", 
      [](Eigen::MatrixXd V, BoundaryConditionConfig cfg)
      ->std::unique_ptr<BoundaryCondition>
      {return std::make_unique<StretchBC>(V, cfg);});
  
  register_type(BCScriptType::BC_TWIST, "twist", 
      [](Eigen::MatrixXd V, BoundaryConditionConfig cfg)
      ->std::unique_ptr<BoundaryCondition>
      {return std::make_unique<TwistBC>(V, cfg);});  
}