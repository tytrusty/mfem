#include "mesh.h"
#include "energies/material_model.h"
#include "config.h"
#include "energies/stable_neohookean.h"
#include "utils/pinning_matrix.h"
#include "igl/boundary_facets.h"
#include "igl/oriented_facets.h"
#include "igl/edges.h"
#include "factories/boundary_condition_factory.h"

using namespace mfem;
using namespace Eigen;

Mesh::Mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
    const Eigen::VectorXi& material_ids,
    const std::vector<std::shared_ptr<MaterialModel>>& materials)
    : V_(V), Vref_(V), Vinit_(V), T_(T), mat_ids_(material_ids) {
  assert(materials.size() > 0);
  material_ = materials[0];

  for (Eigen::Index i = 0; i < T_.rows(); ++i) {
    elements_.push_back(Element(materials[material_ids(i)]));
  }

  igl::boundary_facets(T_, F_);
  initial_velocity_ = 0 * V_;
}


Mesh::Mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
    std::shared_ptr<MaterialModel> material)
    : V_(V), Vref_(V), Vinit_(V), T_(T), material_(material) {

  mat_ids_.resize(0);

  for (int i = 0; i < T_.rows(); ++i) {
    elements_.push_back(Element(material));
  }

  igl::boundary_facets(T_, F_);
  initial_velocity_ = 0 * V_;
}

void Mesh::init() {
  V_ = Vinit_;
  init_bcs();

  volumes(vols_);

  int M = std::pow(V_.cols(),2);

  // Initialize volume sparse matrix
  W_.resize(T_.rows()*M, T_.rows()*M);
  std::vector<Triplet<double>> trips;
  for (int i = 0; i < T_.rows(); ++i) {
    for (int j = 0; j < M; ++j) {
      trips.push_back(Triplet<double>(M*i+j, M*i+j,vols_[i]));
    }
  }
  W_.setFromTriplets(trips.begin(),trips.end());

  init_jacobian();
  PJW_ = P_ * J_.transpose() * W_;

  mass_matrix(M_, vols_);

  PM_ = P_ * M_;
  PMP_ = PM_ * P_.transpose();
  
  int dim = V_.cols();

  MatrixXi E, F;
  if (dim == 2) {
    igl::oriented_facets(T_, E);
  } else {
    igl::boundary_facets(T_, F);
    igl::edges(T_, E);
  }
}

void Mesh::init_bcs() {

  BoundaryConditionFactory factory;
  bc_ = factory.create(bc_config_.type, Vref_, bc_config_);
  bc_->init(V_);
  is_fixed_ = bc_->fixed();
  P_ = pinning_matrix(V_, T_, is_fixed_);
  free_map_.resize(is_fixed_.size(), -1);
  int curr = 0;
  for (int i = 0; i < is_fixed_.size(); ++i) {
    if (is_fixed_(i) == 0) {
      free_map_[i] = curr++;
    }
  }  
}
