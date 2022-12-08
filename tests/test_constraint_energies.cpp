#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "test_common.h"
#include "svd/dsvd.h"
using namespace Test;

TEST_CASE("Constraint Energy Gradient - dEL/dL") {

  App<> app;
  std::shared_ptr<MixedALMOptimizer> obj = app.sim;
  VectorXd def_grad = obj->J_*(obj->P_.transpose()*obj->xt_+obj->b_);

  // Energy function for finite differences
  auto E = [&](const VectorXd& la)-> double {
    double h = app.config->h;
    double Ela = 0;

    for (int i = 0; i < obj->nelem_; ++i) {
      Matrix<double,9,6> W;
      Wmat(obj->R_[i],W);
      Vector9d diff = W*obj->s_.segment(6*i,6) - def_grad.segment(9*i,9);
      Ela += la.segment(9*i,9).dot(diff) * obj->vols_[i];
    }
    return Ela;
  };

  // Compute gradient
  VectorXd grad(9*obj->nelem_);
  for (int i = 0; i < obj->nelem_; ++i) {
    Matrix<double,9,6> W;
    Wmat(obj->R_[i],W);
    Vector9d diff = W*obj->s_.segment(6*i,6) - def_grad.segment(9*i,9);
    grad.segment(9*i,9) = obj->vols_[i] * diff;
  }

  // Finite difference gradient
  VectorXd fgrad;
  finite_gradient(obj->la_, E, fgrad, SECOND);

  CHECK(compare_gradient(grad, fgrad));
}

TEST_CASE("Constraint Energy Gradient - dEL/ds") {

  App<> app;
  std::shared_ptr<MixedALMOptimizer> obj = app.sim;

  VectorXd s(6*obj->nelem_);
  for (int i = 0; i < obj->nelem_; ++i) {
    s.segment(6*i,6) = obj->s_.segment(6*i,6);
  }

  VectorXd def_grad = obj->J_*(obj->P_.transpose()*obj->xt_+obj->b_);
  
  // Energy function for finite differences
  auto E = [&](const VectorXd& st)-> double {
    double h = app.config->h;
    double Ela = 0;

    for (int i = 0; i < obj->nelem_; ++i) {
      Matrix<double,9,6> W;
      Wmat(obj->R_[i],W);
      Vector9d diff = W*st.segment(6*i,6) - def_grad.segment(9*i,9);
      Ela += obj->la_.segment(9*i,9).dot(diff) * obj->vols_[i];
    }
    return Ela;
  };

  // Compute gradient
  VectorXd grad(6*obj->nelem_);
  for (int i = 0; i < obj->nelem_; ++i) {
    Matrix<double,9,6> W;
    Wmat(obj->R_[i],W);
    Vector6d g = W.transpose() * obj->la_.segment(9*i,9);
    grad.segment(6*i,6) = obj->vols_[i] * g;
  }

  // Finite difference gradient
  VectorXd fgrad;
  finite_gradient(s, E, fgrad, SECOND); 
  CHECK(compare_gradient(grad, fgrad));
}

TEST_CASE("Constraint Energy Gradient - dEL/dx") {

  App<> app;
  std::shared_ptr<MixedALMOptimizer> obj = app.sim;
  
  // Energy function for finite differences
  auto E = [&](const VectorXd& x)-> double {
    double h = app.config->h;
    double Ela = 0;
    VectorXd def_grad = obj->J_*(obj->P_.transpose()*x+obj->b_);

    for (int i = 0; i < obj->nelem_; ++i) {
      Matrix3d F = Map<Matrix3d>(def_grad.segment(9*i,9).data());
      JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
      Matrix3d R = svd.matrixU() *  svd.matrixV().transpose();
      Matrix<double,9,6> W;
      Wmat(R,W);

      Vector9d diff = W*obj->s_.segment(6*i,6) - def_grad.segment(9*i,9);
      Ela += obj->la_.segment(9*i,9).dot(diff) * obj->vols_[i];
    }
    return Ela;
  };

  // Compute gradient
  VectorXd grad = obj->P_ * (obj->Jw_.transpose() 
      * obj->WhatS_ - obj->Jw_.transpose()) * obj->la_;

  // Finite difference gradient
  VectorXd fgrad;
  finite_gradient(obj->xt_, E, fgrad, SECOND);
  CHECK(compare_gradient(grad, fgrad));
}

TEST_CASE("Constraint Energy Gradient - d2EL/dxds") {

  App<> app;
  std::shared_ptr<MixedALMOptimizer> obj = app.sim;

  VectorXd s(6*obj->nelem_);
  for (int i = 0; i < obj->nelem_; ++i) {
    obj->s_.segment(6*i,6).setRandom() * 10;

    s.segment(6*i,6) = obj->s_.segment(6*i,6);
  }

  obj->la_.setOnes();
  obj->la_ *= 10;

  // Compute jacobian
  obj->update_system();

  MatrixXd grad = obj->WhatL_.transpose() * obj->Jw_ * obj->P_.transpose();
  grad.transposeInPlace();

  
  // Energy function for finite differences
  auto Ex = [&](const VectorXd& x)-> double {
    double h = app.config->h;
    double Ela = 0;
    VectorXd def_grad = obj->J_*(obj->P_.transpose()*x+obj->b_);

    for (int i = 0; i < obj->nelem_; ++i) {
      Matrix3d F = Map<Matrix3d>(def_grad.segment(9*i,9).data());
      JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
      Matrix3d R = svd.matrixU() *  svd.matrixV().transpose();
      Matrix<double,9,6> W;
      Wmat(R,W);

      Vector9d diff = W*obj->s_.segment(6*i,6) - def_grad.segment(9*i,9);
      Ela += obj->la_.segment(9*i,9).dot(diff) * obj->vols_[i];
    }
    return Ela;
  };


  // Vecotr function for finite differences
  auto E = [&](const VectorXd& s)-> VectorXd {
    for (int i = 0; i < obj->nelem_; ++i) {
      obj->s_.segment(6*i,6) = s.segment(6*i,6);
    }
    // Compute gradient
    VectorXd g;
    VectorXd xt = obj->xt_;
    finite_gradient(xt,Ex,g,SIXTH,1e-4);
    return g;
  };

  // Finite difference gradient
  MatrixXd fgrad;
  finite_jacobian(s, E, fgrad, SIXTH, 1e-4);

  std::cout << "fgrad: \n" << fgrad << std::endl;
  std::cout << "grad: \n" << grad << std::endl;
  CHECK(compare_jacobian(grad, fgrad,1e-3));
}
