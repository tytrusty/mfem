#pragma once

#include <Eigen/Dense>
#include <Eigen/QR>

//solve orthogonal procrustes problem using newton;s method 
//find rotation such that ||R*A - B||F is minimized

static Eigen::Matrix<double, 9,9> dRdFtmp; 

static const Eigen::Matrix3d cpx = [] { Eigen::Matrix3d tmp;
      tmp << 0.,0., 0.,
             0.,0., -1.,
             0.,1., 0.;
      return tmp;
}();

static const Eigen::Matrix3d cpy = [] { Eigen::Matrix3d tmp;
      tmp << 0.,0., 1.,
            0.,0., 0.,
            -1.,0., 0.;
      return tmp;
}();

static const Eigen::Matrix3d cpz = [] { Eigen::Matrix3d tmp;
      tmp << 0.,-1., 0.,
            1.,0., 0.,
            0.,0., 0.;
      return tmp;
}();

static const Eigen::Matrix3d cpxx = [] { Eigen::Matrix3d tmp;
      
      tmp = cpx*cpx;
      return tmp;
}();

static const Eigen::Matrix3d cpyy = [] { Eigen::Matrix3d tmp;
      
      tmp = cpy*cpy;
      return tmp;
}();

static const Eigen::Matrix3d cpzz = [] { Eigen::Matrix3d tmp;
      
      tmp = cpz*cpz;
      return tmp;
}();

static const Eigen::Matrix3d cpxy = [] { Eigen::Matrix3d tmp;
      
      tmp = 0.5*(cpx*cpy + cpy*cpx).array();
      return tmp;
}();

static const Eigen::Matrix3d cpxz = [] { Eigen::Matrix3d tmp;
      
      tmp = 0.5*(cpx*cpz + cpz*cpx).array();
      return tmp;
}();

static const Eigen::Matrix3d cpyz = [] { Eigen::Matrix3d tmp;
      
      tmp = 0.5*(cpy*cpz + cpz*cpy).array();
      return tmp;
}();


static const Eigen::Matrix<double, 9,3> Skew_To_Full = [] {

  Eigen::Matrix<double, 9,3> tmp;  

  tmp <<    0., 0., 0., //(0,0)
            0., 0., 1., //(1,0)
            0., -1., 0., //(2,0)
            0., 0., -1., //(0,1)
            0., 0., 0., //(1,1)
            1., 0., 0., //(2,1)
            0.,  1., 0., //(0,2)
            -1., 0.,  0., //(1,2)
            0., 0., 0.; //(2,2)

   return tmp;

}();

template<typename DerivedMat, typename DerivedVec>
void rodrigues(Eigen::MatrixBase<DerivedMat> &R, const Eigen::MatrixBase<DerivedVec> &omega) {
    
    using Scalar = typename DerivedVec::Scalar; 

    Scalar angle = omega.norm();
    //handle the singularity ... return identity for 0 angle of rotation
    if(std::fabs(angle) < 1e-8) {
        R.setIdentity();
        return;
    }

    Eigen::Matrix<Scalar, 3,1> axis = omega.normalized();
    Eigen::Matrix<Scalar, 3,3> K;
    
    K << 0., -axis(2), axis(1),
         axis(2), 0., -axis(0),
         -axis(1), axis(0), 0.;
    R = Eigen::Matrix3d::Identity() + std::sin(angle)*K + (1-std::cos(angle))*K*K;
}

template<typename DerivedR, typename DerivedA, typename DerivedB, typename DerivedDeriv = Eigen::Matrix<double, 9,9> >
void newton_procrustes(Eigen::MatrixBase<DerivedR> &R,  const Eigen::MatrixBase<DerivedA> &A, const Eigen::MatrixBase<DerivedB> &B, bool compute_gradients = false, Eigen::MatrixBase<DerivedDeriv> &dRdF = dRdFtmp, double tol = 1e-6, int max_iter = 100) {

    using Scalar = typename DerivedR::Scalar;

    //constant matrices needed for computing gradient and hessian
    Eigen::Matrix<Scalar,3,3> Y = R*A*B.transpose();
    Eigen::Matrix<Scalar, 3,1> g;
    Eigen::Matrix<Scalar, 3,3> H;
    Eigen::Matrix<Scalar, 3,1> omega;
    Eigen::Matrix<Scalar, 3,3> dR;

    Scalar E0, E1;

    //newton loop
    unsigned int itr = 0;

    do {

        //compute useful gradients here if needed
        g << -(cpx*Y).trace(), -(cpy*Y).trace(), -(cpz*Y).trace();

        //std::cout<<"GRADIENT: "<<g<<"\n";
        if(g.norm() < tol) {
            //converged to within tolerance
            break;
        }

        H << -(cpxx*Y).trace(), -(cpxy*Y).trace(), -(cpxz*Y).trace(),
              -(cpxy*Y).trace(), -(cpyy*Y).trace(), -(cpyz*Y).trace(),
              -(cpxz*Y).trace(), -(cpyz*Y).trace(), -(cpzz*Y).trace();


        omega = -H.colPivHouseholderQr().solve(g);
        E0 = -(R*Y).trace();
        E1 = E0 + 1.0;

        do {

            rodrigues(dR, omega);
            E1 = -(dR*Y).trace();

            omega.noalias() =  omega*0.6;
            
        }while(E1 > E0 && omega.norm() > tol);
        
        R = dR*R;
        Y = dR*Y;

        ++itr;
        
    }while(itr < max_iter);

    if(compute_gradients) {

          //compute dRdF gradient which is a  3x3x3x3 fourth order tensor
          //stored compactly as a 3x9 matrix.

          //update hessian to that at optimal point
          H << -(cpxx*Y).trace(), -(cpxy*Y).trace(), -(cpxz*Y).trace(),
               -(cpxy*Y).trace(), -(cpyy*Y).trace(), -(cpyz*Y).trace(),
               -(cpxz*Y).trace(), -(cpyz*Y).trace(), -(cpzz*Y).trace();

          if(H.norm() < 1e-6) {
                std::cout<<"H FAILURE\n";
                exit(1);
          }

          Eigen::Matrix<Scalar,3,9> dF; //F derivatives
          Y =  R*A;

          dF.row(0) = sim::flatten(-(cpx*Y));
          dF.row(1) = sim::flatten(-(cpy*Y));
          dF.row(2) = sim::flatten(-(cpz*Y));

          //gradients computed 

          //apply rotation
          dRdF = -sim::flatten_multiply_right<Eigen::Matrix<Scalar, 3,3>>(R)*Skew_To_Full*H.colPivHouseholderQr().solve(dF);
    }
    
}
