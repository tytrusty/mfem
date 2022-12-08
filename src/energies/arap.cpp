#include "energies/arap.h"
#include "config.h"

using namespace Eigen;
using namespace mfem;

double ArapModel::energy(const Vector6d& S) {
    
  double mu = config_->mu;
  double S1 = S(0);
  double S2 = S(1);
  double S3 = S(2);
  double S4 = S(3);
  double S5 = S(4);
  double S6 = S(5);
  return (mu*(pow(S1-1.0,2.0)+pow(S2-1.0,2.0)+pow(S3-1.0,2.0)+(S4*S4)*2.0
        +(S5*S5)*2.0+(S6*S6)*2.0))/2.0;
}

Vector6d ArapModel::gradient(const Vector6d& S) {
  
  double mu = config_->mu;
  double la = config_->la;
  double S1 = S(0);
  double S2 = S(1);
  double S3 = S(2);
  double S4 = S(3);
  double S5 = S(4);
  double S6 = S(5);
  Vector6d g;
  g(0) = (mu*(S1*2.0-2.0))/2.0;
  g(1) = (mu*(S2*2.0-2.0))/2.0;
  g(2) = (mu*(S3*2.0-2.0))/2.0;
  g(3) = S4*mu*2.0;
  g(4) = S5*mu*2.0;
  g(5) = S6*mu*2.0;
  return g;

}

Matrix6d ArapModel::hessian_inv(const Vector6d& S) {
  
  double mu = config_->mu;
  double S1 = S(0);
  double S2 = S(1);
  double S3 = S(2);
  double S4 = S(3);
  double S5 = S(4);
  double S6 = S(5);
  Matrix6d Hinv;
  Hinv.setZero();
  Hinv(0,0) = 1. / mu;
  Hinv(1,1) = 1. / mu;
  Hinv(2,2) = 1. / mu;
  Hinv(3,3) = 1. / (2.0 * mu);
  Hinv(4,4) = 1. / (2.0 * mu);
  Hinv(5,5) = 1. / (2.0 * mu);
  return Hinv;
}

Matrix6d ArapModel::hessian(const Vector6d& S, bool psd_fix) {
  double mu = config_->mu;
  Vector6d tmp; tmp << 1,1,1,2,2,2;
  return tmp.asDiagonal() * mu;
}

double ArapModel::energy(const Vector3d& s) {
  double mu = config_->mu;
  double S1 = s(0);
  double S2 = s(1);
  double S3 = s(2);
  return (mu*(pow(S1-1.0,2.0)+pow(S2-1.0,2.0)+(S3*S3)*2.0))/2.0;;
}

Vector3d ArapModel::gradient(const Vector3d& s) {
  double mu = config_->mu;
  double S1 = s(0);
  double S2 = s(1);
  double S3 = s(2);
  Vector3d g;
  g(0) = (mu*(S1*2.0-2.0))/2.0;
  g(1) = (mu*(S2*2.0-2.0))/2.0;
  g(2) = S3*mu*2.0;
  return g;
}

Matrix3d ArapModel::hessian(const Vector3d& s) {
  double mu = config_->mu;
  double S1 = s(0);
  double S2 = s(1);
  double S3 = s(2);
  Matrix3d H;
  H.setZero();
  H(0,0) = mu;
  H(1,1) = mu;
  H(2,2) = mu*2.0;
  return H;
}
