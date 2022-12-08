#pragma once

#include <EigenTypes.h>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <limits>
#include "optimizers/optimizer_data.h"

// Utility for workin with corotational
//preconditioned conjugate gradient
template<typename PreconditionerSolver, typename Scalar, int Ordering>
inline int pcg(Eigen::VectorXx<Scalar>& x,
    const Eigen::SparseMatrix<Scalar, Ordering> &A,
    const Eigen::VectorXx<Scalar> &b, Eigen::VectorXx<Scalar> &r,
    Eigen::VectorXx<Scalar> &z, Eigen::VectorXx<Scalar> &zm1, Eigen::VectorXx<Scalar> &p,
    Eigen::VectorXx<Scalar> &Ap, PreconditionerSolver &pre,
    Scalar tol = 1e-4, unsigned int num_itr = 500) {

    if(b.norm() < std::sqrt(std::numeric_limits<Scalar>::epsilon())) {
      x.setZero();
      return 0;
    }
  
    r = b - A * x;

   if (r.norm()/b.norm() < tol || (r.norm() < std::sqrt(std::numeric_limits<Scalar>::epsilon()))) {
          return 0;
    }

  mfem::Timer t;

  //t.start("setup");
  z = pre.solve(r);
  zm1 = z;
  p = z;
  Scalar rsold = r.dot(z);
  Scalar rsnew = 0.;
  Scalar alpha = 0.;
  Scalar beta = 0.;
  //t.stop("setup");

  //Eigen::VectorXd zm1;

  for(unsigned int i=0; i<num_itr; ++i) {
    //t.start("Ap");
    Ap = A * p;
    //t.stop("Ap");
    //t.start("rsnew");
    alpha = rsold / (p.dot(Ap));
    x = x + alpha * p;
    r = r - alpha * Ap;
    rsnew = r.dot(r);
    //t.stop("rsnew");
    
    if (r.norm()/b.norm() < tol || (r.norm() < std::numeric_limits<Scalar>::epsilon())) {
          return i;
    }
    
    zm1 = z;
    //t.start("precon");
    z = pre.solve(r);
    //t.stop("precon");
    
    //t.start("end");
    rsnew = r.dot(z-zm1);

    //HS beta
    //beta = rsnew/p.dot(z-zm1);
    //PRP
    beta = rsnew/rsold;

    //FR
    //beta =  r.dot(z)/rsold;

    p = z + beta * p;
    rsold = r.dot(z);
    //t.stop("end");
  }
  // t.print();
  return num_itr;
}

template<typename PreconditionerSolver, typename Scalar, int Ordering>
inline int pcr(Eigen::VectorXx<Scalar>& x,
    const Eigen::SparseMatrix<Scalar, Ordering> &A,
    const Eigen::VectorXx<Scalar> &b, Eigen::VectorXx<Scalar> &r,
    Eigen::VectorXx<Scalar> &z, Eigen::VectorXx<Scalar> &p,
    Eigen::VectorXx<Scalar> &Ap, PreconditionerSolver &pre,
    Scalar tol = 1e-4, unsigned int num_itr = 500) {

      Eigen::VectorXd Ar;
      Eigen::VectorXd Api;

      r = b-A*x;

      if (r.dot(r) < tol) {
        return 0; 
      }
      r = pre.solve(r);
      p = r;
      Ap = A*p;
      Ar = A*r;

      Scalar rold;

      for(unsigned int i=0; i<num_itr; ++i) {

        rold = r.dot(Ar);

        Api = pre.solve(Ap);
        Scalar alpha = rold/(Ap.dot(Api));

        x = x + alpha*p;

        
        r = r - alpha*Api;
        
        if (std::fabs((b-A*x).norm() - tol*b.norm()) < 1e-8) {
          return i;
        }

        Ar = A*r;
        Scalar beta = r.dot(Ar)/rold;

        p = r + beta*p;
        
        Ap = Ar + beta*Ap;

      }

}
//TODO pcg.tpp for tempalte implementation
//#include <pcg.cpp>
