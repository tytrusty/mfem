#include "stretch.h"
#include "mesh/mesh.h"
#include "energies/material_model.h"
#include "config.h"

using namespace Eigen;
using namespace mfem;

template<int DIM>
Stretch<DIM>::Stretch(std::shared_ptr<Mesh> mesh)
    : Variable<DIM>(mesh) {
}

template<int DIM>
double Stretch<DIM>::energy(const VectorXd& x) {

  VectorXd def_grad;
  mesh_->deformation_gradient(x, def_grad);
  double e_psi = 0.0;
  #pragma omp parallel for reduction(+ : e_psi)
  for (int i = 0; i < nelem_; ++i) {
    double vol = mesh_->volumes()[i];
    const VecM& F = def_grad.segment<M()>(M()*i);
    e_psi += mesh_->elements_[i].material_->energy(F) * vol;
  }
  return e_psi;
}

template<int DIM>
void Stretch<DIM>::post_solve() {}

template<int DIM>
void Stretch<DIM>::update(const Eigen::VectorXd& x, double h) {

  double h2 = h*h;
  VectorXd def_grad;
  mesh_->deformation_gradient(x, def_grad);

  const std::vector<MatrixXd>& Jloc = mesh_->local_jacobians();

  // Computing per-element hessian and gradients
  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    const VecM& F = def_grad.segment<M()>(M()*i);
    double vol = mesh_->volumes()[i];
    
    H_[i] = (Jloc[i].transpose() * mesh_->elements_[i].material_->hessian(F)
        * Jloc[i]) * vol * h2;
    g_[i] = Jloc[i].transpose() * mesh_->elements_[i].material_->gradient(F)
        * vol * h2;
  }
  assembler_->update_matrix(H_);
  lhs_ = assembler_->A;

  data_.timer.start("RHS - stretch");
  VectorXd g;
  vec_assembler_->assemble(g_, g);
  grad_ = g;
  data_.timer.stop("RHS - stretch");
}

template<int DIM>
VectorXd Stretch<DIM>::rhs() {
  return -gradient();
}

template<int DIM>
VectorXd Stretch<DIM>::gradient() {
  return grad_;
}

template<int DIM>
void Stretch<DIM>::reset() {
  nelem_ = mesh_->T_.rows();

  H_.resize(nelem_);
  g_.resize(nelem_);
  Aloc_.resize(nelem_);
  assembler_ = std::make_shared<Assembler<double,DIM,-1>>(
      mesh_->T_, mesh_->free_map_);
  vec_assembler_ = std::make_shared<VecAssembler<double,DIM,-1>>(mesh_->T_,
      mesh_->free_map_);

  #pragma omp parallel for
  for (int i = 0; i < nelem_; ++i) {
    H_[i].setIdentity();
    g_[i].setZero();
  }
}

template class mfem::Stretch<3>; // 3D
template class mfem::Stretch<2>; // 2D

