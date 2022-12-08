#include "newton_optimizer.h"

#include "linesearch.h"
#include "mesh/mesh.h"
#include "factories/linear_solver_factory.h"
#include <unsupported/Eigen/SparseExtra>

using namespace mfem;
using namespace Eigen;

template <int DIM> void NewtonOptimizer<DIM>::step() { state_.data_.clear();
  // Pre operations for variables
  state_.x_->pre_solve();
  for (auto& var : state_.mixed_vars_) {
    var->pre_solve();
  }
  for (auto& var : state_.vars_) {
    var->pre_solve();
  }

  int i = 0;
  double grad_norm;
  double E = 0, E_prev = 0, res = 0;
  do {

    // Update gradient and hessian
    update_system();

    // Solve linear system
    substep(grad_norm);

    double alpha = 1.0;

    auto energy_func = [&state = state_](double a) {
      double h2 = std::pow(state.x_->integrator()->dt(), 2);

      Eigen::VectorXd x0 = state.x_->value() + a * state.x_->delta();
      double val = state.x_->energy(x0);
      state.x_->unproject(x0);

      for (const auto& var : state.mixed_vars_) {
        const Eigen::VectorXd si = var->value() + a * var->delta();
        val += h2 * var->energy(x0, si) - var->constraint_value(x0, si);  
      }
      for (const auto& var : state.vars_) {
        val += h2 * var->energy(x0);  
      }
      return val;
    };

    // Record initial energies
    E = energy_func(0.0);
    res = std::abs((E - E_prev) / (E+1e-6));
    E_prev = E;

    // Linesearch on descent direction
    state_.data_.timer.start("LS");
    auto status = linesearch_backtracking(state_, alpha, energy_func,0.0,0.5);
    state_.data_.timer.stop("LS");

    // Record some data
    state_.data_.add(" Iteration", i+1);
    state_.data_.add("Energy", E);
    state_.data_.add("Energy res", res);
    state_.data_.add("Newton dec", grad_norm);
    state_.data_.add("alpha ", alpha);
    ++i;
    Base::callback(state_);

  } while (i < state_.config_->outer_steps
      && grad_norm > state_.config_->newton_tol
      && (res > 1e-12));

  if (state_.config_->show_data) {
    state_.data_.print_data(state_.config_->show_timing);
  }

  // Update dirichlet boundary conditions
  state_.mesh_->update_bcs(state_.config_->h);

  // Post solve update nodal and mixed variables
  state_.x_->post_solve();
  for (auto& var : state_.mixed_vars_) {
    var->post_solve();
  }
  for (auto& var : state_.vars_) {
    var->post_solve();
  }
}

template <int DIM>
void NewtonOptimizer<DIM>::update_system() {
  state_.data_.timer.start("update");

  // Get full configuration vector
  VectorXd x = state_.x_->value();
  state_.x_->unproject(x);

  if (!state_.mesh_->fixed_jacobian()) {
    state_.mesh_->update_jacobian(x);
  }

  for (auto& var : state_.vars_) {
    var->update(x, state_.x_->integrator()->dt());
  }
  for (auto& var : state_.mixed_vars_) {
    var->update(x, state_.x_->integrator()->dt());
  }
  state_.data_.timer.stop("update");
}

template <int DIM>
void NewtonOptimizer<DIM>::substep(double& decrement) {
  // Solve linear system
  state_.data_.timer.start("global");
  solver_->solve();
  state_.data_.timer.stop("global");

  // Use infinity norm of deltas as termination criteria
  decrement = state_.x_->delta().template lpNorm<Infinity>();
  for (auto& var : state_.mixed_vars_) {
    decrement = std::max(decrement, var->delta().template lpNorm<Infinity>());
  }
}

template <int DIM>
void NewtonOptimizer<DIM>::reset() {
  Optimizer<DIM>::reset();

  state_.x_->reset();
  for (auto& var : state_.vars_) {
    var->reset();
  }
  for (auto& var : state_.mixed_vars_) {
    var->reset();
  }

  LinearSolverFactory<DIM> solver_factory;
  solver_ = solver_factory.create(state_.config_->solver_type, &state_);
}

template class mfem::NewtonOptimizer<3>;
template class mfem::NewtonOptimizer<2>;
