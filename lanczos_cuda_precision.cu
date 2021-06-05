#ifndef LANCZOS_PRECISION_CUDA_CU
#define LANCZOS_PRECISION_CUDA_CU

//--------------------------------------------------------------
// The Lanczos routine computes the Lanczos matrix 
// Diagonalization of the Lanczosmatrix must be done separately
// in the routine LanczosDiag
//
// This file contains the high-precision parallell cuda implementation of Lanczos
//
//
//--------------------------------------------------------------

using namespace std;

#ifdef DEBUG
const bool WTRACE=true;
#else
const bool WTRACE=false;
#endif

#ifdef DEBUG2
const bool TRACE=true;
#else
const bool TRACE=false;
#endif

#include<iostream>
#include<iomanip>

const double TINYNUMBER=1.e-15; // numbers smaller than this are effectively considered 0.
const double LANCZOSEXITLIMIT=1.e-15;

#include "unistd.h"
#include "lanczos.h"
#include "rnddef.h"

#include "cuda_runtime.h"
#include <cublas_v2.h>
#include "cuComplex.h"

#include "cuda_global.h"
#include "cuda_precision.h"
#include "lanczos_cuda_precision.h" 

#include "lowlevelroutines.h"

#include "global.h"

//Lanczos::Lanczos(MyMatrix& H_in,int M_in){impl = new Lanczos_impl(H_in,M_in);} // initializes Lanczos with random vector
Lanczos::Lanczos(MyMatrix& H_in,int M_in,int basestatenr){impl = new Lanczos_impl(H_in,M_in,basestatenr);} // initializes Lanczos with a basestate 
Lanczos::Lanczos(MyMatrix& H_in,int M_in, vector<MyField>& vec_in){impl = new Lanczos_impl(H_in,M_in,vec_in);}  // initializes Lanczos using the input vector
Lanczos::~Lanczos(){delete impl;} 
void Lanczos::DoLanczos(){impl->DoLanczos();}
MyField* Lanczos::GetInitVector(){return impl->GetInitVector();}
MyField* Lanczos::GetPhi(const int i){return impl->GetPhi(i);}
void Lanczos::DiagonalizeLanczosMatrix(){impl->DiagonalizeLanczosMatrix();}
vector<double>& Lanczos::GetLanczosa(){return impl->GetLanczosa();}
vector<double>& Lanczos::GetLanczosb(){return impl->GetLanczosb();}  // we start numbering b from b0,b1 etc.
vector<double>& Lanczos::GetEigenvalues(){return impl->GetEigenvalues();}
double* Lanczos::GetEigenvector(const int i){return impl->GetEigenvector(i);}
int Lanczos::GetNeigenvalues(){return impl->GetNeigenvalues();}
int Lanczos::GetNLanczosIterations(){return impl->GetNLanczosIterations();}
double Lanczos::GetNormofInitVector(){return impl->GetNormofInitVector();} 
MyField* Lanczos::GetLanczosEigenvectorinStateSpaceBasis(const int n){return impl->GetLanczosEigenvectorinStateSpaceBasis(n);}








Lanczos_impl::Lanczos_impl(MyMatrix& H_in,int M_in,int basestatenr):
 H(H_in),N(H.Ncols()),M(min(N,M_in)),Mstop(M),Niterations(0),phi(NULL),
 lanczos_a(M),lanczos_b(M-1),LM(0),initnorm(1.),
 handle(NULL),status(cublasCreate(&handle)),err(cudaSuccess),
 Rows(H_in.Nrows()),Cols(H_in.Ncols()),Nc(H_in.NStoredCols()),Ntot(Rows*Nc),
 d_m_val(NULL),d_m_col(NULL),d_phi(NULL)
{
  if(WTRACE) cout << "Starting Lanczos, M=" << M << " with base state " << basestatenr << " Rows:" << Rows << " Cols:" << Cols << " Nc: " << Nc << " Ntot:" << Ntot << endl;
  phi = new MyField[(M+1)*Cols]; // allocate space for Lanczos vectors
  for(int i=0; i<(M+1)*Cols; i++){ phi[i]=MyField(0.,0.); }
  
  if(basestatenr >=0 && basestatenr < Cols)
    {
      if(TRACE) cout << "Setting base state: " << basestatenr << endl;
      phi[basestatenr]=complex<double>(1.,0.); // initialize the vector
      initnorm=1.;
    }
  else
    {     // Choose a random basestate if basestatenr is -1 or >#of states
      RandomInitialization(); // pick a random state
      if(TRACE)
	{
	  cout << "phi[0]:" << endl;
	  for(int i=0; i<Cols; i++) cout << setprecision(DEBUGPRECISION) << phi[i] << endl;
	}	
    }
  
  UploadMatrixToDevice(H,&d_m_val,&d_m_col);

  prec_lanczos_a = new realformat[M];
  prec_lanczos_b = new realformat[M]; // only the M-1 components will finally be used

  AllocateSpaceOnDevice((M+1)*Cols*sizeofcmplx,&d_phi);
  InitializeSpaceOnDevice((M+1)*Cols*sizeofcmplx,&d_phi);

  UploadToDeviceAndExpand(&phi[0],1,&d_phi); 

  Normalize(0); // normalize the random vector
  
  if(TRACE)
    {
      cout << "phi[0]:" << endl;
      InspectDeviceCmplx(&d_phi,Cols);
    }
  
  
  if(TRACE) 
    {
      cout << "the matrix is: " << H << endl;
    }
}

Lanczos_impl::Lanczos_impl(MyMatrix& H_in,int M_in, vector<MyField>& vec_in):
  H(H_in),N(H.Ncols()),M(min(N,M_in)),Mstop(M),Niterations(0),phi(NULL),
lanczos_a(M),lanczos_b(M-1),LM(0),initnorm(1.),
handle(NULL),status(cublasCreate(&handle)),err(cudaSuccess),
Rows(H_in.Nrows()),Cols(H_in.Ncols()),Nc(H_in.NStoredCols()),Ntot(Rows*Nc),
d_m_val(NULL),d_m_col(NULL),d_phi(NULL)
{
  if(WTRACE) cout << "Starting Lanczos with initial vector, M=" << M << endl;
  phi = new MyField[(M+1)*Cols]; // allocate space for Lanczos vectors
  for(int i=0; i<(M+1)*Cols; i++){ phi[i]=MyField(0.,0.); }
  
  for(int i=0; i<Cols; i++) phi[i]=vec_in[i]; // initialize the vector
  
  if(TRACE)
    {
      cout << "phi[0]:" << endl;
      for(int i=0; i<Cols; i++) cout << phi[i] << " ";
    }
  
  
  UploadMatrixToDevice(H,&d_m_val,&d_m_col);

  prec_lanczos_a = new realformat[M];
  prec_lanczos_b = new realformat[M]; // only the M-1 components will finally be used

  AllocateSpaceOnDevice((M+1)*Cols*sizeofcmplx,&d_phi);
  InitializeSpaceOnDevice((M+1)*Cols*sizeofcmplx,&d_phi);

  UploadToDeviceAndExpand(&phi[0],1,&d_phi); 

  
  //    if(TRACE){ cout << "phi[0] after upload" << endl; PrintOutPhi(0);}
  
  initnorm=Normalize(0); // record the normalization of the init vector, and normalize it
  if(TRACE) cout << "Init norm is: " << initnorm << endl;
  if(abs(initnorm) < TINYNUMBER){Mstop=0; Niterations=0;} // in case the starting vector is the zero vector
}


Lanczos_impl::~Lanczos_impl()
{
  if(WTRACE) cout << "Destroying class Lanczos" << endl;

  delete[] prec_lanczos_a;
  delete[] prec_lanczos_b;

  FreeMemoryOnDevice(&d_phi);

  delete[] phi;
  delete LM;
}


void Lanczos_impl::DoLanczos()
{
  if(TRACE) cout << "Starting DoLanczos (lanczos_precision_cuda.cu)" << endl;
  int m=0;
  bool continueiteration=true;
  while(continueiteration && m<Mstop){
    if(TRACE) cout << "Doing Lanczos step m:" << m << " M: " << M << endl;
    continueiteration=LanczosIteration(m++);
  } 
  Niterations=m; // gives the number of lanczos vectors
  if(TRACE){
    cout << "There are in all " << Niterations << " orthogonal Lanczos vectors" << endl;
    if(Niterations >0)
      {
	for(int i=0; i<Niterations-1; i++){
	  cout << "a[" << i << "]=" << lanczos_a[i] << " b[" << i << "]="<< lanczos_b[i] << endl;}
	cout << "a[" << Niterations-1 << "]=" << lanczos_a[Niterations-1] << endl;
      }
  }
  
  DownloadFromDeviceAndContract(&d_phi,(M+1)*Cols,phi);
  FreeMatrixOnDevice(&d_m_val,&d_m_col);
  FreeMemoryOnDevice(&d_phi);
}


double Lanczos_impl::Normalize(int i)
{
  if(WTRACE) cout << "in Lanczos_impl::Normalize" << endl;

  realformat norm=makereal(0.);
  void* norm_ptr=reinterpret_cast<void*>(&norm);

  void* d_v= &d_phi[i*Cols*sizeofcmplx];

  cuda_norm_cmplx(Cols,d_v,norm_ptr); 
  cuda_Zdinvscal(Cols,norm_ptr,d_v); 
  
  double norm_double= todouble(norm);

  if(TRACE) cout << "Norm is " << setprecision(DEBUGPRECISION) << norm_double << endl;
  return norm_double;
}

/*
double Lanczos_impl::GetNorm(const int n,cuda_cmplx* d_z)
{
  if(WTRACE) cout << "in Lanczos_impl::GetNorm" << endl;

  realformat norm=makereal(0.);
  void* norm_ptr=reinterpret_cast<void*>(&norm);
  void* d_v= &d_phi[i*Cols*sizeofcmplx];

  cuda_norm_cmplx(Cols,d_v,norm_ptr); 
  double norm_double= todouble(norm);

  return norm_double;
}
*/



void Lanczos_impl::RandomInitialization()
  {
  if(WTRACE) cout << "In RandomInitialization" << endl;
    MyField* phi0=&phi[0];
    for(int i=0; i<Cols; i++)
      {
	double r_real=2.*RAN.Get()-1.;
	double r_imag=2.*RAN.Get()-1.;

	MyField r=MyField(r_real,r_imag); // random start

	phi0[i]=r;
      }
  }


void Lanczos_impl::CheckOrthogonality(int n,void* d_z,const int m)
{
  cout << "Checking orthogonality:" << endl;

  for(int i=0; i<=m; i++)
    {
      cmplxformat q=cmplxformat(makereal(0.),makereal(0.));
      void* qp=reinterpret_cast<void*>(&q);
      void* d_phii=&d_phi[i*n*sizeofcmplx];
      cuda_cmplxdotproduct_cmplx(n,d_phii,d_z,qp); 
      
      cuda_cmplx qd=cuda_cmplx(todouble(q.x),todouble(q.y));

      if(cuCabs(qd) > 1.e-13)
	{
	  cout << "WARNING: Severe Loss of Orthogonality";
	  cout << "<phi[" << i << "]|z>=" << setprecision(DEBUGPRECISION) << qd << endl;
	  cout << endl;
	}
    }
}

void Lanczos_impl::Reorthogonalize(int n,void* d_z,int m)
{
  if(WTRACE) cout << "Starting Reorthogonalize" << endl;
  //  for(int j=0; j<MGSREPEATSTEPS; j++) // 2 for Modified Gram-Schmidt with reorthogo..
    for(int i=0; i<=m; i++)
      {
	cmplxformat* q;
	void* qp=reinterpret_cast<void*>(q)
	void* d_phii=&d_phi[i*n*sizeofcmplx];
	cuda_cmplxdotproduct_cmplx(n,d_phii,d_z,qp); // q=<phi_i|z> 
	cuda_cmplx_Zmaxpy(n,qp,d_z,d_phii); // |z>=|z>-q*|phi_i>; 
      }
  if(TRACE) cout << "Done Reorthogonalize" << endl; 
}


bool Lanczos_impl::LanczosIteration(const int m)
{
  if(WTRACE)  cout << "Starting Lanczos iteration " << m << endl;
  Niterations++;
  
  void* d_phim=&d_phi[m*Cols*sizeofcmplx]; //already normalized     
  void* d_z   =&d_phi[(m+1)*Cols*sizeofcmplx];                 
  
  int blockspergrid=(Rows+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
  
  cuda_mat_vec_multiply_cmplx(Rows,Nc,d_m_col ,d_m_val, d_phim, d_z); // |z> = H |phi_m>  
  err = cudaGetLastError();                                                  
  if(TRACE) cout << "end of cuda_kernel_mat_vec_multiply_cmplx, last error" << err << endl; 
  

  if(TRACE)
    {
      cout << "H * phi[" << m << "]" << endl;
      InspectDeviceCmplx(&d_z,Cols);
    }

    
  cuda_cmplx a=cmplx_0;
  realformat* am=&prec_lanczos_a[m]; // the pointer the the device lanczos value am
  void* amp = reinterpret_cast<void*>(am);
  if(TRACE) cout << "Launching ComplexDotProduct" << endl; 
  cuda_realdotproduct_cmplx(Cols,d_phim,d_z,amp); // a_m=<phi_m|z>, should be real 
  
  if(TRACE) cout << "lanczos_a[" << m << "]=" << todouble(prec_lanczos_a[m]) << endl;
  
  cuda_real_Zmaxpy(Cols,amp,d_phim,d_z); // |z>=|z>-a_m*|phi_m>  
  
  if(m>0)
    {
      void* bmm1=reinterpret_cast<void*>(&prec_lanczos_b[(m-1)];)
      void* d_phimm1=&d_phi[(m-1)*Cols*sizeofcmplx];
      
      cuda_real_Zmaxpy(Cols,bmm1,d_phimm1,d_z); // |z>=|z>-b_(m-1)*|phi_(m-1)>
    }
  
#ifdef REORTHOGONALIZE
  Reorthogonalize(Cols,d_z,m);
#else
  
#endif
  if(TRACE) cout << "After Reorthogonalization" << endl;
  if(TRACE) CheckOrthogonality(Cols,d_z,m);


  void* bm=reinterpret_cast<void*>(&prec_lanczos_b[m]);
  cuda_norm_cmplx(Cols,d_z,bm);

  if(m == M-1) // end of lanczos procedure               
    {
      if(TRACE) cout << "Truncating the Lanczos procedure, reached max matrix size, lanczos_b[M-1]= " << todouble(prec_lanczos_b[M-1]) << " --> 0" << endl;
      return false;
    }
  
  if(TRACE) cout << "lanczos_a[" << m << "]=" << todouble(prec_lanczos_a[m]) << endl;
  if(TRACE) cout << "lanczos_b[" << m << "]=" << todouble(prec_lanczos_b[m]) << endl;
  if(abs(prec_lanczos_b[m])<=LANCZOSEXITLIMIT){return false;} // stop the Lanczos procedure;            
  
  cuda_Zdinvscal(Cols,bm,d_z); // |z> -> |z>/beta

  if(TRACE){
    cout << "Lanczos vector phi[" << m+1 << "]: " << endl;
    InspectDevice(&d_z,Cols);
  }

  return true;
}



void Lanczos_impl::DiagonalizeLanczosMatrix()
{
  if(Niterations==0) return;
  delete LM;
  LM = new LanczosMatrix(Niterations,lanczos_a,lanczos_b);
  LM->Diagonalize();
}

MyField* Lanczos_impl::GetLanczosEigenvectorinStateSpaceBasis(const int n)
{
  MyField* phiM=&phi[M*Cols]; // the last one is used as storage
  double* nu=LM->Eigenvector(n);

  for(int i=0; i<N; i++){ phiM[i]=0;} // initialize to zero.
  for(int p=0; p<Niterations; p++)
    {
      MyField* phip=&phi[p*Cols];
      AddToVector(&N,phiM,phip,&nu[p]);	 
    }
  return phiM;
}

