#include "catch2/catch.hpp"
#include "test_common.h"
#include "svd/dsvd.h"
using namespace Test;

// TEST_CASE("dsvd - dR/dF") {

//   Matrix3d F;
//   F << 1.0, 0.1, 0.2,
//        0.1, 2.0, 0.4,
//        0.3, 0.4, 0.5;

//   JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
//   Matrix3d U = svd.matrixU();
//   Matrix3d V = svd.matrixV();
//   std::array<Matrix3d, 9> dRdF;
//   dsvd(F, U, svd.singularValues(), V, dRdF);

//   Matrix<double, 9, 9> J;
//   for (int i = 0; i < 9; ++i) {
//     J.row(i) = Vector9d(dRdF[i].data()).transpose();
//   }

//   // Vecotr function for finite differences
//   auto E = [&](const VectorXd& vecF)-> VectorXd {

//     F = Matrix3d(vecF.data());
//     JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
//     Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
//     Vector9d vecR(R.data());
//     return vecR;
//   };

//   // Finite difference gradient
//   MatrixXd fgrad;
//   VectorXd vecF = Vector9d(F.data());
//   finite_jacobian(vecF, E, fgrad, SECOND);

//   // std::cout << "fgrad: \n" << fgrad << std::endl;
//   // std::cout << "grad: \n" << J << std::endl;
//   CHECK(compare_jacobian(J, fgrad));
// }

TEST_CASE("dsvd2") {

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
    J.row(i) = Vector9d(dRdF[i].data()).transpose();
  }

  // f = (RS-F) : (RS-F) = tr( (RS-F)(RS - F)^T) = tr( RS*(RS-F)^T - F*(RS-F)^T)
  // g = df/dR = S(RS-F)^T?
  // dg/DR = 

  Matrix3d S = V * svd.singularValues().asDiagonal() * V.transpose();
  Matrix3d R = U*V.transpose();

  Matrix9d dhdR;
  Matrix9d dhdF;
  Matrix3d I = Matrix3d::Identity();

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Matrix3d dhdR_ij = I.col(i) * I.row(j) * S * S;
      Matrix3d dhdF_ij = - I.col(i) * I.row(j) * S;
      
      dhdR.row(3*j + i) = Vector9d(dhdR_ij.data());
      dhdF.row(3*j + i) = Vector9d(dhdF_ij.data());
    }
  }

  // std::cout << "dhdR \n" << dhdR << std::endl;
  // std::cout << "dhdF \n" << dhdF << std::endl;
  // std::cout << "TEST: \n" << -dhdR.inverse() * dhdF << std::endl;
  // std::cout << "J:\n" << J << std::endl;
  //std::cout << "R*: " << U*V.transpose() << std::endl;
  //std::cout << "h(R) = S(R-F): " << S * (R*S - F) << std::endl;

  // // function for finite differences
  // auto E = [&](const VectorXd& vecF)-> VectorXd {

  //   F = Matrix3d(vecF.data());
  //   JacobiSVD<Matrix3d> svd(F, ComputeFullU | ComputeFullV);
  //   Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
  //   Matrix<double, 9, 6> W;
  //   Wmat(R, W);
  //   return W*s;
  // };

  // // Finite difference gradient
  // MatrixXd fgrad;
  // VectorXd vecF = Vector9d(F.data());
  // finite_jacobian(vecF, E, fgrad, SECOND);
  // CHECK(compare_jacobian(J, fgrad));
  // std::cout << "!!!!!!!!!!!!!!" << std::endl;
}