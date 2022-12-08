#include "displacement.h"

#include "mesh/mesh.h"
#include "energies/material_model.h"
#include "svd/newton_procrustes.h"
#include "utils/pinning_matrix.h"
#include "config.h"
#include "factories/integrator_factory.h"
#include "time_integrators/implicit_integrator.h"

using namespace Eigen;
using namespace mfem;

template<int DIM>
Displacement<DIM>::Displacement(std::shared_ptr<Mesh> mesh,
    std::shared_ptr<SimConfig> config)
    : Variable<DIM>(mesh), config_(config) {
}

template<int DIM>
double Displacement<DIM>::energy(const VectorXd& x) {

  double h = integrator_->dt();
  const auto& P = mesh_->projection_matrix();
  VectorXd xt = P.transpose()*x + b_;
  VectorXd diff = xt - integrator_->x_tilde() - h*h*f_ext_;
  const auto& MM = mesh_->template mass_matrix<MatrixType::FULL>();
  double e = 0.5*diff.transpose()*MM*diff;
  return e;
}

template<int DIM>
void Displacement<DIM>::post_solve() {
  // TODO b_,/BCs should not be here, boundary conditions
  // probably should be owned by either optimizer or mesh
  #pragma omp parallel for
  for (int i = 0; i < mesh_->V_.rows(); ++i) {
    if (mesh_->is_fixed_(i)) {
      b_.segment<DIM>(DIM*i) = mesh_->V_.row(i).transpose();
    }
  }

  const auto& P = mesh_->projection_matrix();
  VectorXd x = P.transpose()*x_ + b_;
  integrator_->update(x);

  // Update mesh vertices
  MatrixXd V = Map<MatrixXd>(x.data(), mesh_->V_.cols(), mesh_->V_.rows());
  mesh_->V_ = V.transpose();
}

template<int DIM>
void Displacement<DIM>::update(const Eigen::VectorXd&, double) {}

template<int DIM>
VectorXd Displacement<DIM>::rhs() {
  data_.timer.start("RHS - x");
  rhs_ = -gradient();
  data_.timer.stop("RHS - x");
  return rhs_;
}

template<int DIM>
VectorXd Displacement<DIM>::gradient() {
  double h = integrator_->dt();
  const auto& P = mesh_->projection_matrix();
  VectorXd xt = P.transpose()*x_ + b_;
  VectorXd diff = xt - integrator_->x_tilde() - h*h*f_ext_;

  const auto& PM = mesh_->template mass_matrix<MatrixType::PROJECT_ROWS>();
  grad_ = PM * diff;
  return grad_;
}

template<int DIM>
void Displacement<DIM>::reset() {

  MatrixXd tmp = mesh_->V_.transpose();
  x_ = Map<VectorXd>(tmp.data(), mesh_->V_.size());

  tmp = mesh_->initial_velocity_.transpose();
  VectorXd v0 = Map<VectorXd>(tmp.data(), mesh_->V_.size());

  IntegratorFactory factory;
  integrator_ = factory.create(config_->ti_type, x_, v0, config_->h);  
  
  // Project out Dirichlet boundary conditions
  const auto& P = mesh_->projection_matrix();
  b_ = x_ - P.transpose()*P*x_;
  x_ = P * x_;
  dx_ = 0*x_;

  lhs_ = mesh_->template mass_matrix<MatrixType::PROJECTED>();

  // External gravity force
  VecD ext = Map<Matrix<float,DIM,1>>(config_->ext).template cast<double>();
  f_ext_ = ext.replicate(mesh_->V_.rows(),1);
}

template class mfem::Displacement<3>; // 3D
template class mfem::Displacement<2>; // 2D

