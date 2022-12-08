// This file is part of libigl, a simple c++ geometry processing library.
// 
// Copyright (C) 2013 Alec Jacobson <alecjacobson@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.
#include <cmath>
#include <algorithm>

#undef USE_SCALAR_IMPLEMENTATION
#define USE_SSE_IMPLEMENTATION
#undef USE_AVX_IMPLEMENTATION
#define COMPUTE_U_AS_MATRIX
#define COMPUTE_V_AS_MATRIX
#include "Singular_Value_Decomposition_Preamble.hpp"

// disable runtime asserts on xor eax,eax type of stuff (doesn't always work,
// disable explicitly in compiler settings)
#pragma runtime_checks( "u", off )  
template<typename T>
void svd3x3_sse(
  const Eigen::Matrix<T, 3*4, 3>& A, 
  Eigen::Matrix<T, 3*4, 3> &U, 
  Eigen::Matrix<T, 3*4, 1> &S, 
  Eigen::Matrix<T, 3*4, 3>&V)
{
  // this code assumes USE_SSE_IMPLEMENTATION is defined 
  float Ashuffle[9][4], Ushuffle[9][4], Vshuffle[9][4], Sshuffle[3][4];
  for (int i=0; i<3; i++)
  {
    for (int j=0; j<3; j++)
    {
      for (int k=0; k<4; k++)
      {
        Ashuffle[i + j*3][k] = A(i + 3*k, j);
      }
    }
  }

#include "Singular_Value_Decomposition_Kernel_Declarations.hpp"

  ENABLE_SSE_IMPLEMENTATION(Va11=_mm_loadu_ps(Ashuffle[0]);)
  ENABLE_SSE_IMPLEMENTATION(Va21=_mm_loadu_ps(Ashuffle[1]);)
  ENABLE_SSE_IMPLEMENTATION(Va31=_mm_loadu_ps(Ashuffle[2]);)
  ENABLE_SSE_IMPLEMENTATION(Va12=_mm_loadu_ps(Ashuffle[3]);)
  ENABLE_SSE_IMPLEMENTATION(Va22=_mm_loadu_ps(Ashuffle[4]);)
  ENABLE_SSE_IMPLEMENTATION(Va32=_mm_loadu_ps(Ashuffle[5]);)
  ENABLE_SSE_IMPLEMENTATION(Va13=_mm_loadu_ps(Ashuffle[6]);)
  ENABLE_SSE_IMPLEMENTATION(Va23=_mm_loadu_ps(Ashuffle[7]);)
  ENABLE_SSE_IMPLEMENTATION(Va33=_mm_loadu_ps(Ashuffle[8]);)

#include "Singular_Value_Decomposition_Main_Kernel_Body.hpp"

  ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Ushuffle[0],Vu11);)
  ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Ushuffle[1],Vu21);)
  ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Ushuffle[2],Vu31);)
  ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Ushuffle[3],Vu12);)
  ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Ushuffle[4],Vu22);)
  ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Ushuffle[5],Vu32);)
  ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Ushuffle[6],Vu13);)
  ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Ushuffle[7],Vu23);)
  ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Ushuffle[8],Vu33);)

  ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Vshuffle[0],Vv11);)
  ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Vshuffle[1],Vv21);)
  ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Vshuffle[2],Vv31);)
  ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Vshuffle[3],Vv12);)
  ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Vshuffle[4],Vv22);)
  ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Vshuffle[5],Vv32);)
  ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Vshuffle[6],Vv13);)
  ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Vshuffle[7],Vv23);)
  ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Vshuffle[8],Vv33);)

  ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Sshuffle[0],Va11);)
  ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Sshuffle[1],Va22);)
  ENABLE_SSE_IMPLEMENTATION(_mm_storeu_ps(Sshuffle[2],Va33);)

  for (int i=0; i<3; i++)
  {
    for (int j=0; j<3; j++)
    {
      for (int k=0; k<4; k++)
      {
        U(i + 3*k, j) = Ushuffle[i + j*3][k];
        V(i + 3*k, j) = Vshuffle[i + j*3][k];
      }
    }
  }

  for (int i=0; i<3; i++)
  {
    for (int k=0; k<4; k++)
    {
      S(i + 3*k, 0) = Sshuffle[i][k];
    }
  }
}


//#ifdef __SSE__
template<typename T>
void polar_svd3x3_sse(const Eigen::Matrix<T, 3*4, 3>& A, Eigen::Matrix<T, 3*4, 3> &R)
{
  // should be caught at compile time, but just to be 150% sure:
  assert(A.rows() == 3*4 && A.cols() == 3);

  Eigen::Matrix<T, 3*4, 3> U, Vt;
  Eigen::Matrix<T, 3*4, 1> S;
  svd3x3_sse(A, U, S, Vt);

  for (int k=0; k<4; k++)
  {
    R.block(3*k, 0, 3, 3) = U.block(3*k, 0, 3, 3) * Vt.block(3*k, 0, 3, 3).transpose();
  }
}

template<typename T>
void svd3x3_sse(const Eigen::VectorXx<T>& F, std::vector<Eigen::Matrix3f>& U,
    std::vector<Eigen::Vector3f>& sigma, std::vector<Eigen::Matrix3f>& V) {

  int nelem = F.size() / 9;
  int N = (nelem / 4) + int(nelem % 4 != 0);
  U.resize(nelem);
  sigma.resize(nelem);
  V.resize(nelem);

  #pragma omp parallel for 
  for (int ii = 0; ii < N; ++ii) {
    Eigen::Matrix<float,12,3> F4,U4,V4;
    Eigen::Matrix<float,12,1> S4;
    // SSE implementation operates on 4 matrices at a time, so assemble
    // 12 x 3 matrices
    for (int jj = 0; jj < 4; ++jj) {
      int i = ii*4 +jj;
      if (i >= nelem)
        break;
      Eigen::Matrix3x<T> f4 = Eigen::Map<const Eigen::Matrix3x<T>>(
          (F.template segment<9>(9*i)).data());
      F4.block(3*jj, 0, 3, 3) = f4.template cast<float>();
    }
 
    // Solve rotations
    svd3x3_sse(F4, U4, S4, V4);

    // Assign rotations to per-element matrices
    for (int jj = 0; jj < 4; jj++) {
      int i = ii*4 +jj;
      if (i >= nelem)
        break;
      U[i] = U4.block(3*jj, 0, 3, 3);
      V[i] = V4.block(3*jj, 0, 3, 3);
      sigma[i] = S4.segment(3*jj, 3); 
    }
  }
}

#pragma runtime_checks( "u", restore )
