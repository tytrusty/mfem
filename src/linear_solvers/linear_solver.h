#pragma once

#include "EigenTypes.h"
#include "simulation_state.h"

namespace mfem {

  template <typename Scalar, int DIM>
  class LinearSolver {

  public:

    LinearSolver(SimState<DIM>* state) : state_(state) {}
    
    virtual void solve() = 0;

    virtual ~LinearSolver() = default;

  protected:
    SimState<DIM>* state_;
  };

}
