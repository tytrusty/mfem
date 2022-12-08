#include "energies/fung.h"
#include "simple_psd_fix.h"
#include "config.h"

using namespace Eigen;
using namespace mfem;

static const double c = 4;

double Fung::energy(const Vector6d& S) {
    
  double mu = config_->mu;
  double la = config_->la;

  double S1 = S(0);
  double S2 = S(1);
  double S3 = S(2);
  double S4 = S(3);
  double S5 = S(4);
  double S6 = S(5);

  return (mu*(exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0)-1.0))/2.0+(mu*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0+(la*pow((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0,2.0))/2.0;
}

Vector6d Fung::gradient(const Vector6d& S) {
  
  double mu = config_->mu;
  double la = config_->la;
  double S1 = S(0);
  double S2 = S(1);
  double S3 = S(2);
  double S4 = S(3);
  double S5 = S(4);
  double S6 = S(5);

  Vector6d g;
  g(0,0) = S1*mu-la*(S2*S3-S6*S6)*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)+(S1*c*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0))/2.0;
  g(1,0) = S2*mu-la*(S1*S3-S5*S5)*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)+(S2*c*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0))/2.0;
  g(2,0) = S3*mu-la*(S1*S2-S4*S4)*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)+(S3*c*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0))/2.0;
  g(3,0) = S4*mu*2.0+la*(S3*S4*2.0-S5*S6*2.0)*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)+S4*c*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0);
  g(4,0) = S5*mu*2.0+la*(S2*S5*2.0-S4*S6*2.0)*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)+S5*c*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0);
  g(5,0) = S6*mu*2.0+la*(S1*S6*2.0-S4*S5*2.0)*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)+S6*c*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0);
  return g;
}

Matrix6d Fung::hessian(const Vector6d& S, bool psd_fix) {
  Matrix6d H;
  H.setZero();
  double mu = config_->mu;
  double la = config_->la;
  double S1 = S(0);
  double S2 = S(1);
  double S3 = S(2);
  double S4 = S(3);
  double S5 = S(4);
  double S6 = S(5);
       
  H(0,0) = mu+la*pow(S2*S3-S6*S6,2.0)+(c*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0))/2.0+((S1*S1)*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0))/2.0;
  H(0,1) = -S3*la*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)+la*(S1*S3-S5*S5)*(S2*S3-S6*S6)+(S1*S2*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0))/2.0;
  H(0,2) = -S2*la*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)+la*(S1*S2-S4*S4)*(S2*S3-S6*S6)+(S1*S3*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0))/2.0;
  H(0,3) = -la*(S3*S4*2.0-S5*S6*2.0)*(S2*S3-S6*S6)+S1*S4*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0);
  H(0,4) = -la*(S2*S5*2.0-S4*S6*2.0)*(S2*S3-S6*S6)+S1*S5*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0);
  H(0,5) = S6*la*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0-la*(S1*S6*2.0-S4*S5*2.0)*(S2*S3-S6*S6)+S1*S6*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0);
  H(1,0) = -S3*la*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)+la*(S1*S3-S5*S5)*(S2*S3-S6*S6)+(S1*S2*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0))/2.0;
  H(1,1) = mu+la*pow(S1*S3-S5*S5,2.0)+(c*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0))/2.0+((S2*S2)*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0))/2.0;
  H(1,2) = -S1*la*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)+la*(S1*S2-S4*S4)*(S1*S3-S5*S5)+(S2*S3*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0))/2.0;
  H(1,3) = -la*(S3*S4*2.0-S5*S6*2.0)*(S1*S3-S5*S5)+S2*S4*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0);
  H(1,4) = S5*la*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0-la*(S2*S5*2.0-S4*S6*2.0)*(S1*S3-S5*S5)+S2*S5*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0);
  H(1,5) = -la*(S1*S6*2.0-S4*S5*2.0)*(S1*S3-S5*S5)+S2*S6*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0);
  H(2,0) = -S2*la*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)+la*(S1*S2-S4*S4)*(S2*S3-S6*S6)+(S1*S3*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0))/2.0;
  H(2,1) = -S1*la*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)+la*(S1*S2-S4*S4)*(S1*S3-S5*S5)+(S2*S3*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0))/2.0;
  H(2,2) = mu+la*pow(S1*S2-S4*S4,2.0)+(c*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0))/2.0+((S3*S3)*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0))/2.0;
  H(2,3) = S4*la*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0-la*(S3*S4*2.0-S5*S6*2.0)*(S1*S2-S4*S4)+S3*S4*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0);
  H(2,4) = -la*(S2*S5*2.0-S4*S6*2.0)*(S1*S2-S4*S4)+S3*S5*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0);
  H(2,5) = -la*(S1*S6*2.0-S4*S5*2.0)*(S1*S2-S4*S4)+S3*S6*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0);
  H(3,0) = -la*(S3*S4*2.0-S5*S6*2.0)*(S2*S3-S6*S6)+S1*S4*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0);
  H(3,1) = -la*(S3*S4*2.0-S5*S6*2.0)*(S1*S3-S5*S5)+S2*S4*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0);
  H(3,2) = S4*la*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0-la*(S3*S4*2.0-S5*S6*2.0)*(S1*S2-S4*S4)+S3*S4*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0);
  H(3,3) = mu*2.0+la*pow(S3*S4*2.0-S5*S6*2.0,2.0)+c*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0)+S3*la*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0+(S4*S4)*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0)*2.0;
  H(3,4) = S6*la*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*-2.0+la*(S2*S5*2.0-S4*S6*2.0)*(S3*S4*2.0-S5*S6*2.0)+S4*S5*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0)*2.0;
  H(3,5) = S5*la*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*-2.0+la*(S1*S6*2.0-S4*S5*2.0)*(S3*S4*2.0-S5*S6*2.0)+S4*S6*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0)*2.0;
  H(4,0) = -la*(S2*S5*2.0-S4*S6*2.0)*(S2*S3-S6*S6)+S1*S5*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0);
  H(4,1) = S5*la*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0-la*(S2*S5*2.0-S4*S6*2.0)*(S1*S3-S5*S5)+S2*S5*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0);
  H(4,2) = -la*(S2*S5*2.0-S4*S6*2.0)*(S1*S2-S4*S4)+S3*S5*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0);
  H(4,3) = S6*la*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*-2.0+la*(S2*S5*2.0-S4*S6*2.0)*(S3*S4*2.0-S5*S6*2.0)+S4*S5*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0)*2.0;
  H(4,4) = mu*2.0+la*pow(S2*S5*2.0-S4*S6*2.0,2.0)+c*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0)+S2*la*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0+(S5*S5)*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0)*2.0;
  H(4,5) = S4*la*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*-2.0+la*(S1*S6*2.0-S4*S5*2.0)*(S2*S5*2.0-S4*S6*2.0)+S5*S6*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0)*2.0;
  H(5,0) = S6*la*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0-la*(S1*S6*2.0-S4*S5*2.0)*(S2*S3-S6*S6)+S1*S6*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0);
  H(5,1) = -la*(S1*S6*2.0-S4*S5*2.0)*(S1*S3-S5*S5)+S2*S6*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0);
  H(5,2) = -la*(S1*S6*2.0-S4*S5*2.0)*(S1*S2-S4*S4)+S3*S6*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0);
  H(5,3) = S5*la*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*-2.0+la*(S1*S6*2.0-S4*S5*2.0)*(S3*S4*2.0-S5*S6*2.0)+S4*S6*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0)*2.0;
  H(5,4) = S4*la*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*-2.0+la*(S1*S6*2.0-S4*S5*2.0)*(S2*S5*2.0-S4*S6*2.0)+S5*S6*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0)*2.0;
  H(5,5) = mu*2.0+la*pow(S1*S6*2.0-S4*S5*2.0,2.0)+c*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0)+S1*la*((mu+c*mu)/la+S1*(S6*S6)+S2*(S5*S5)+S3*(S4*S4)-S1*S2*S3-S4*S5*S6*2.0+1.0)*2.0+(S6*S6)*(c*c)*mu*exp((c*(S1*S1+S2*S2+S3*S3+(S4*S4)*2.0+(S5*S5)*2.0+(S6*S6)*2.0-3.0))/2.0)*2.0;
  if(psd_fix) {
    sim::simple_psd_fix(H);
  }

  return H;
}

double Fung::energy(const Eigen::Vector9d& F)  { return 0.; }
Eigen::Vector9d Fung::gradient(const Eigen::Vector9d& F) {
  return Eigen::Vector9d::Zero();
}
Eigen::Matrix9d Fung::hessian(const Eigen::Vector9d& F) {
  return Eigen::Matrix9d::Identity();
}
