#include "catch2/catch.hpp"
#include "test_common.h"
#include "svd/dsvd.h"
#include "svd/newton_procrustes.h"
using namespace Test;

/*
TEST_CASE("dsvd - dS/dF") {

  Matrix3d F;
  F << 1.0, 0.1, 0.2,
       0.1, 2.0, 0.4,
       0.3, 0.4, 0.5;

  JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
  // Matrix3d U = svd.matrixU();
  // Matrix3d V = svd.matrixV();
  // std::array<Matrix3d, 9> dRdF;
  // dsvd(F, U, svd.singularValues(), V, dRdF);



  Tensor3333d dU, dV;
  Tensor333d dS;
  dsvd(dU, dS, dV, F);

  Matrix3d V = svd.matrixV();
  Matrix3d S = svd.singularValues().asDiagonal();
  std::array<Matrix3d, 9> dS_dF;

  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      dS_dF[3*c + r] = dV[r][c]*S*V.transpose() + V*dS[r][c].asDiagonal()*V.transpose()
          + V*S*dV[r][c].transpose();
    }
  }

  Matrix<double, 9, 9> J;
  for (int i = 0; i < 9; ++i) {
    J.col(i) = Vector9d(dS_dF[i].data());
  }
  
  Matrix<double, 6, 9> Js;
  Js.row(0) = J.row(0);
  Js.row(1) = J.row(4);
  Js.row(2) = J.row(8);
  Js.row(3) = J.row(1);
  Js.row(4) = J.row(2);
  Js.row(5) = J.row(5);


  // Vecotr function for finite differences
  auto E = [&](const VectorXd& vecF)-> VectorXd {

    F = Matrix3d(vecF.data());
    JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
    Matrix3d S = svd.matrixV() * svd.singularValues().asDiagonal()
        * svd.matrixV().transpose();
    Vector6d vecS; vecS << S(0,0), S(1,1), S(2,2), S(1,0), S(2,0), S(2,1);
    return vecS;
  };

  // Finite difference gradient
  MatrixXd fgrad;
  VectorXd vecF = Vector9d(F.data());
  finite_jacobian(vecF, E, fgrad, SECOND);

  std::cout << "J: " << J << std::endl;
  std::cout << "fgrad: \n" << fgrad << std::endl;
  std::cout << "grad: \n" << Js << std::endl;
  CHECK(compare_jacobian(Js, fgrad));
}*/

TEST_CASE("dsvd - dR/dF") {

  Matrix3d F;
  F << 1.0, 0.1, 0.2,
       0.1, 2.0, 0.4,
       0.3, 0.4, 0.5;

  JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
  Matrix3d U = svd.matrixU();
  Matrix3d V = svd.matrixV();
  std::array<Matrix3d, 9> dRdF;
  dsvd(F, U, svd.singularValues(), V, dRdF);

  Matrix<double, 9, 9> J;
  for (int i = 0; i < 9; ++i) {
    J.row(i) = Vector9d(dRdF[i].data()).transpose();
  }

  // Vecotr function for finite differences
  auto E = [&](const VectorXd& vecF)-> VectorXd {

    F = Matrix3d(vecF.data());
    JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
    Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
    Vector9d vecR(R.data());
    return vecR;
  };

  // Finite difference gradient
  MatrixXd fgrad;
  VectorXd vecF = Vector9d(F.data());
  finite_jacobian(vecF, E, fgrad, SECOND);

  // std::cout << "fgrad: \n" << fgrad << std::endl;
  // std::cout << "grad: \n" << J << std::endl;
  CHECK(compare_jacobian(J, fgrad));
}

TEST_CASE("dsvd - dWs/dF") {

  Matrix3d F;
  F << 1.0, 0.1, 0.2,
       0.1, 2.0, 0.4,
       0.3, 0.4, 0.5;

  Vector6d s;
  s << 1.11, 1.2, 1.3, 0.2, 0.3, 0.4;

  JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
  Matrix3d U = svd.matrixU();
  Matrix3d V = svd.matrixV();
  std::array<Matrix3d, 9> dRdF;
  dsvd(F, U, svd.singularValues(), V, dRdF);

  Matrix<double, 9, 9> J;
  Matrix<double, 9, 6> What;

  for (int i = 0; i < 9; ++i) {
    Wmat(dRdF[i] , What);
    J.col(i) = (What * s);
  }

  // function for finite differences
  auto E = [&](const VectorXd& vecF)-> VectorXd {

    F = Matrix3d(vecF.data());
    JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
    Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
    Matrix<double, 9, 6> W;
    Wmat(R, W);
    return W*s;
  };

  // Finite difference gradient
  MatrixXd fgrad;
  VectorXd vecF = Vector9d(F.data());
  finite_jacobian(vecF, E, fgrad, SECOND);
  CHECK(compare_jacobian(J, fgrad));
}


TEST_CASE("dsvd - dWs/dq") {
  App<> app;
  std::shared_ptr<MixedALMOptimizer> obj = app.sim;
  int n = obj->J_.cols();
  MatrixXd Jk = obj->J_.block(0,0,9,n);
  obj->xt_ *= 10;

  Vector9d vecF = Jk * (obj->P_.transpose() * obj->xt_ + obj->b_);
  Matrix3d F = Matrix3d(vecF.data());
  
  Vector6d s;
  s << 1.11, 1.2, 1.3, 0.2, 0.3, 0.4;
  s *= 100;

  JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
  Matrix3d U = svd.matrixU();
  Matrix3d V = svd.matrixV();
  std::array<Matrix3d, 9> dRdF;
  dsvd(F, U, svd.singularValues(), V, dRdF);

  Matrix<double, 9, 9> Whats;
  Matrix<double, 9, 6> What;
  for (int i = 0; i < 9; ++i) {
    Wmat(dRdF[i] , What);
    Whats.row(i) = (What * s).transpose();
  }

  MatrixXd J = obj->P_ * Jk.transpose() * Whats;

  // function for finite differences
  auto E = [&](const VectorXd& x)-> VectorXd {
    Vector9d vecF = Jk * (obj->P_.transpose() * x + obj->b_);
    Matrix3d F = Matrix3d(vecF.data());
    JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
    Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
    Matrix<double, 9, 6> W;
    Wmat(R, W);
    return W*s;
  };

  // Finite difference gradient
  MatrixXd fgrad;
  VectorXd qt = obj->xt_;
  finite_jacobian(qt, E, fgrad, SIXTH, 1e-6);
  // std::cout << "frad: \n" << fgrad << std::endl;
  // std::cout << "grad: \n" << J.transpose() << std::endl;
  CHECK(compare_jacobian(J.transpose(), fgrad));
}

TEST_CASE("dRdF") {

  //generate a random rotation
  Eigen::Matrix3d F = 2.0*Eigen::Matrix3d::Random();
  Eigen::Matrix3d S = Eigen::Matrix3d::Random();
  Eigen::Matrix<double, 9,9> dRdF_fd;
  Eigen::Matrix<double, 3,3>  tmpR0, tmpR1;
  Eigen::Matrix<double, 9,9> dRdF; 
  double alpha = 1e-5;

  Eigen::Matrix3d R0 = Eigen::Matrix3d::Identity();
  
  Eigen::Matrix3d perturb;

  //Finite Difference approximation
  for(unsigned int ii=0; ii< 3; ++ii) {
    for(unsigned int jj=0; jj< 3; ++jj) {
      perturb.setZero();
      perturb(ii,jj) = alpha;
      tmpR0 = tmpR1 = R0;
      newton_procrustes(tmpR0, S, F+perturb);
      newton_procrustes(tmpR1, S, F-perturb);
      dRdF_fd.col(ii + 3*jj) = sim::flatten(((tmpR0 - tmpR1).array()/(2.*alpha)).matrix());
    }

  }
  
  newton_procrustes(R0, S, F, true, dRdF, 1e-8, 200);

  //error 
  double error = (dRdF_fd - dRdF).norm();

  std::cout<<"************* dRdF error: "<<error<<" ************* \n";
  CHECK(error <= alpha);
}