#ifndef CUDA_LANCZOS_H
#define CUDA_LANCZOS_H

//*************************************************************
//
// Header file for the paralell cuda implementation Lanczos
// 
//*************************************************************

#include<iostream>
#include<vector>
#include<complex>
using namespace std;

/*
#include "cuda_runtime.h"
#include "cuComplex.h"
#include "cublas_v2.h"
*/

#include "sparsematrix.h"

//typedef ELLmatrix MyMatrix;

//--------------------------------------------------------------
// The Lanczos routine computes the Lanczos matrix
// Diagonalization of the Lanczosmatrix must be done separately
// in the routine LanczosDiag
//--------------------------------------------------------------

#include <qd/dd_real.h>
#include <qd/qd_real.h>
#include <qd/fpu.h>


struct dd_cmplx{dd_real x; dd_real y;};
struct qd_cmplx{qd_real x; qd_real y;};

#ifdef DOUBLEDOUBLE
#define realformat  dd_real
#define cmplxformat dd_cmplx
#define todouble to_double
#elif defined QUADDOUBLE
#define realformat  qd_real
#define cmplxformat qd_cmplx
#define todouble    to_double;
#else
#define realformat  double
#define cmplxformat cuda_cmplx
#define todouble    double
#endif

const int sizeofreal=sizeof(realformat);
const int sizeofcmplx=sizeof(cmplxformat);

#include "banddiag.h"

// This is the CUDA implementation of Lanczos
class Lanczos_impl
{
public:
  //  Lanczos_impl(MyMatrix& H_in,int M_in); // initializes Lanczos with random vector
  Lanczos_impl(MyMatrix& H_in,int M_in,int basestatenr); // initializes Lanczos with a basestate 
  Lanczos_impl(MyMatrix& H_in,int M_in, vector<MyField>& vec_in); // initializes Lanczos using the input vector
  ~Lanczos_impl();
  void DoLanczos();
  MyField* GetInitVector(){return &phi[0];}
  MyField* GetPhi(const int i){return &phi[i*Cols];}
  //  vector<MyField>& GetInitVector(){return *phi[0];}
  //  vector<MyField>& GetPhi(const int i){return *phi[i];}
  void DiagonalizeLanczosMatrix();
  vector<double>& GetLanczosa(){return lanczos_a;}
  vector<double>& GetLanczosb(){return lanczos_b;}  // we start numbering b from b0,b1 etc.
  vector<double>& GetEigenvalues(){return LM->Eigenvalues();}
  double* GetEigenvector(const int i){return LM->Eigenvector(i);}
  int GetNeigenvalues(){return Niterations;}
  int GetNLanczosIterations(){return Niterations;}
  double GetNormofInitVector(){return initnorm;} 
  MyField* GetLanczosEigenvectorinStateSpaceBasis(const int n);
private:
  MyMatrix& H;
  int N; //matrix with N columns
  const int M; //# of Lanzcos iterations
  int Mstop; 
  int Niterations;

  MyField* phi; 
  vector<double> lanczos_a; 
  vector<double> lanczos_b; 

  LanczosMatrix* LM;
  
  double initnorm;

  bool LanczosIteration(int m);
  void RandomInitialization();

  //  void Reorthogonalize(int m);
  void CheckOrthogonality(int n,void* d_z,const int m);
  void Reorthogonalize(int n,void* d_z,int m)

  double Normalize(int i);
  //  double GetNorm(const int n,cuda_cmplx* d_z);

  // CUDA specific routines:
  //  cublasHandle_t handle;
  //  cublasStatus_t status;
  //  cudaError_t err;

  //  status = cublasCreate(&handle);

  int Rows;
  int Cols;
  int Nc;
  int Ntot;

  // device pointers:
  void* d_m_val; // a pointer to a complex number generally
  int*  d_m_col; 
  void* d_phi;


  int threadsPerBlock;
  int blocksPerGrid;
  
  /*
  void PrintOutPhi(const int m);
  void PrintOutMatrix();
  */
  /*
  int UploadMatrixToDevice();
  int DownloadMatrixFromDevice();
  int UploadLanczosVectorsToDevice();
  int DownloadLanczosVectorsFromDevice();
  int FreeMemoryOnDevice();
  */
};

#endif // CUDA_LANCZOS_H



