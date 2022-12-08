#include "catch2/catch.hpp"
#include "test_common.h"
#include "svd/dsvd.h"
#include "optimizers/mixed_sqp_optimizer.h"

using namespace Test;

// Tests for the symmetric constraint \sum_i^m la_i : (S(x) - s_i)

/*TEST_CASE("Symmetric Constraint Energy Gradient - dEL/dL") {

  App<MixedSQPOptimizer> app;
  std::shared_ptr<MixedSQPOptimizer> obj = app.sim;
  VectorXd def_grad = obj->J_*(obj->P_.transpose()*obj->x_+obj->b_);

  // Energy function for finite differences
  auto E = [&](const VectorXd& la)-> double {
    double Ela = 0;
    VectorXd e_L(obj->nelem_);

    for (int i = 0; i < obj->nelem_; ++i) {
      Matrix3d F = Map<Matrix3d>(def_grad.segment<9>(9*i).data());
      JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);

      Matrix3d S = svd.matrixV() * svd.singularValues().asDiagonal() 
          * svd.matrixV().transpose();
      Vector6d stmp; stmp << S(0,0), S(1,1), S(2,2), S(1,0), S(2,0), S(2,1);

      const Vector6d& si = obj->s_.segment<6>(6*i);
      Vector6d diff = Sym * (stmp - si);
      e_L(i) = la.segment<6>(6*i).dot(diff) * obj->vols_[i];
    }
    return e_L.sum();
  };

  // Compute gradient
  VectorXd grad(6*obj->nelem_);

  for (int i = 0; i < obj->nelem_; ++i) {
    const Vector6d& si = obj->s_.segment(6*i,6);
    grad.segment<6>(6*i) = obj->vols_[i]*Sym*(obj->S_[i] - si);
  }

  // Finite difference gradient
  VectorXd fgrad;
  finite_gradient(obj->la_, E, fgrad, SECOND);

  CHECK(compare_gradient(grad, fgrad));
}

TEST_CASE("Symmetric Constraint Energy Gradient - dEL/ds") {

  App<MixedSQPOptimizer> app;
  std::shared_ptr<MixedSQPOptimizer> obj = app.sim;
  VectorXd def_grad = obj->J_*(obj->P_.transpose()*obj->x_+obj->b_);
  obj->la_.setRandom();

  // Energy function for finite differences
  auto E = [&](const VectorXd& s)-> double {
    double Ela = 0;
    VectorXd e_L(obj->nelem_);

    for (int i = 0; i < obj->nelem_; ++i) {
      const Vector6d& si = s.segment<6>(6*i);
      Vector6d diff = Sym * (obj->S_[i] - si);
      e_L(i) = obj->la_.segment<6>(6*i).dot(diff) * obj->vols_[i];
    }
    return e_L.sum();
  };

  // Compute gradient
  VectorXd grad(6*obj->nelem_);

  for (int i = 0; i < obj->nelem_; ++i) {
    const Vector6d& la = obj->la_.segment<6>(6*i);
    grad.segment<6>(6*i) = -obj->vols_[i]*Sym*la;
  }

  // Finite difference gradient
  VectorXd fgrad;
  finite_gradient(obj->s_, E, fgrad, SECOND);
  CHECK(compare_gradient(grad, fgrad));
}

TEST_CASE("Symmetric Constraint Energy Gradient - dEL/dx") {

  App<MixedSQPOptimizer> app;
  std::shared_ptr<MixedSQPOptimizer> obj = app.sim;
  obj->la_.setRandom();

  // Energy function for finite differences
  auto E = [&](const VectorXd& x)-> double {
    double Ela = 0;
    VectorXd e_L(obj->nelem_);
    VectorXd def_grad = obj->J_*(obj->P_.transpose()*x+obj->b_);

    for (int i = 0; i < obj->nelem_; ++i) {

      Matrix3d F = Map<Matrix3d>(def_grad.segment<9>(9*i).data());
      JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);

      Matrix3d S = svd.matrixV() * svd.singularValues().asDiagonal() 
          * svd.matrixV().transpose();
      Vector6d stmp; stmp << S(0,0), S(1,1), S(2,2), S(1,0), S(2,0), S(2,1);

      const Vector6d& si = obj->s_.segment<6>(6*i);
      Vector6d diff = Sym * (stmp - si);
      e_L(i) = obj->la_.segment<6>(6*i).dot(diff) * obj->vols_[i];
    }
    return e_L.sum();
  };

  VectorXd grad = -obj->Gx_ * obj->la_;

  // Finite difference gradient
  VectorXd fgrad;
  finite_gradient(obj->x_, E, fgrad, SECOND);
  CHECK(compare_gradient(grad, fgrad));
}*/