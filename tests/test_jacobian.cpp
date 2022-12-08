#include "catch2/catch.hpp"
#include "test_common.h"
#include "svd/dsvd.h"
#include "igl/per_face_normals.h"
#include "igl/read_triangle_mesh.h"
using namespace Test;

TEST_CASE("Tet Jacobian - dF/dx") {

  App<> app;
  std::shared_ptr<MixedALMOptimizer> obj = app.sim;
  int n = obj->J_.cols();
  MatrixXd Jk = obj->J_.block(0,0,9,n);

  Vector9d vecF = Jk * (obj->P_.transpose() * obj->xt_ + obj->b_);
  Matrix3d F = Matrix3d(vecF.data());
  
  // function for finite differences
  auto E = [&](const VectorXd& x)-> VectorXd {
    Vector9d vecF = Jk * (obj->P_.transpose() * x + obj->b_);
    return vecF;
  };

  // Finite difference gradient
  MatrixXd fgrad;
  MatrixXd grad = obj->P_ * Jk.transpose();
  VectorXd qt = obj->xt_;
  finite_jacobian(qt, E, fgrad, SECOND);
  CHECK(compare_jacobian(grad.transpose(), fgrad));
}

TEST_CASE("Tri Jacobian - dF/dx") {
  std::shared_ptr<SimConfig> config;
  std::shared_ptr<MaterialConfig> material_config;
  std::shared_ptr<MaterialModel> material;
  std::shared_ptr<TriMesh> obj;
  std::string filename = "../models/triangle.obj";
  //std::string filename = "../models/tet.mesh";
  MatrixXd meshV, meshN;
  MatrixXi meshF;

  // Read the mesh
  igl::read_triangle_mesh(filename,meshV,meshF);

  double fac = meshV.maxCoeff();
  meshV.array() /= fac;
  igl::per_face_normals(meshV,meshF,meshN);

  // Initialize simulator
  config = std::make_shared<SimConfig>();
  config->show_data = false;
  config->show_timing = false;
  material_config = std::make_shared<MaterialConfig>();
  material = std::make_shared<StableNeohookean>(material_config);
  obj = std::make_shared<TriMesh>(meshV, meshF, meshN, material,
      material_config);
  std::cout << "4 " << std::endl;

  SparseMatrixdRowMajor J;
  VectorXd vols;
  obj->jacobian(J, vols, false);
  MatrixXd perturb(obj->V_);
  perturb.setRandom();
  obj->V_ += perturb * 10;

  MatrixXd tmp = obj->V_.transpose();
  VectorXd xt = Map<VectorXd>(tmp.data(), obj->V_.size());
  obj->init_jacobian();
  obj->update_jacobian(xt);
  std::cout << "6 " << std::endl;


  // function for finite differences
  auto E = [&](const VectorXd& x)-> VectorXd {

    // Update mesh vertices
    const MatrixXd V = Map<const MatrixXd>(x.data(), obj->V_.cols(), obj->V_.rows());
    obj->V_ = V.transpose();

    for(int i = 0; i < obj->T_.rows(); ++i) {
      Matrix<double, 9, 3> N;
      N << obj->N_(i,0), 0, 0,
          0, obj->N_(i,0), 0,
          0, 0, obj->N_(i,0),
          obj->N_(i,1), 0, 0,
          0, obj->N_(i,1), 0,
          0, 0, obj->N_(i,1),
          obj->N_(i,2), 0, 0,
          0, obj->N_(i,2), 0,
          0, 0, obj->N_(i,2);
      const RowVector3d v1 = obj->V_.row(obj->T_(i,1)) - obj->V_.row(obj->T_(i,0));
      const RowVector3d v2 = obj->V_.row(obj->T_(i,2)) - obj->V_.row(obj->T_(i,0));
      RowVector3d n = v1.cross(v2);
      double l = n.norm();
      n /= l;
      Vector9d vecF = J*x + N*n.transpose();
      // std::cout << "vecF: " << vecF << std::endl;
      return vecF;
    }
  };
  std::cout << "7 " << std::endl;

  // Finite difference gradient
  const Eigen::SparseMatrixdRowMajor& J2 = obj->Mesh::jacobian();
  MatrixXd fgrad;
  MatrixXd grad = J2;//(J + J2);//.transpose();
  finite_jacobian(xt, E, fgrad, SECOND);
  CHECK(compare_jacobian(grad, fgrad));


}