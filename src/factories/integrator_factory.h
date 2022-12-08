#pragma once

#include "factory.h"
#include "config.h"
#include "time_integrators/implicit_integrator.h"

namespace mfem {
  class IntegratorFactory : public Factory<TimeIntegratorType,
      ImplicitIntegrator, Eigen::VectorXd, Eigen::VectorXd, double> {
  public:
    IntegratorFactory();
  };
}
