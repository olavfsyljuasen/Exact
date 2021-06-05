//--------------------------------------------------------------
// The Lanczos routine computes the Lanczos matrix 
// Diagonalization of the Lanczosmatrix must be done separately
// in the routine LanczosDiag
//
// This file contains the standard serial implementation of Lanczos
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
//#include <cublas_v2.h>
#include "cuComplex.h"

#include "cuda_global.h"
#include "cuda_precision.h"

#include "lanczos_cuda.h" 

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
 lanczos_a(M),lanczos_b(M),LM(0),initnorm(1.),err(cudaSuccess),
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
  AllocateSpaceOnDevice(&d_phi,(M+1)*Cols);
  UploadToDeviceAndExpand(phi,(M+1)*Cols,&d_phi);

  Normalize(0); // normalize the random vector
  
  if(TRACE)
    {
      cout << "phi[0]:" << endl;
      InspectDevice(&d_phi,Cols);
    }
  
  
  if(TRACE) 
    {
      cout << "the matrix is: " << H << endl;
    }
}

Lanczos_impl::Lanczos_impl(MyMatrix& H_in,int M_in, vector<MyField>& vec_in):
  H(H_in),N(H.Ncols()),M(min(N,M_in)),Mstop(M),Niterations(0),phi(NULL),
lanczos_a(M),lanczos_b(M),LM(0),initnorm(1.),err(cudaSuccess),
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
  AllocateSpaceOnDevice(&d_phi,(M+1)*Cols);
  UploadToDeviceAndExpand(phi,(M+1)*Cols,&d_phi);
  
  
  //    if(TRACE){ cout << "phi[0] after upload" << endl; PrintOutPhi(0);}
  
  initnorm=Normalize(0); // record the normalization of the init vector, and normalize it
  if(TRACE) cout << "Init norm is: " << initnorm << endl;
  if(abs(initnorm) < TINYNUMBER){Mstop=0; Niterations=0;} // in case the starting vector is the zero vector
}


Lanczos_impl::~Lanczos_impl()
{
  if(WTRACE) cout << "Destroying class Lanczos" << endl;
  //  cublasDestroy(handle);


  delete[] phi;
  delete LM;
}


void Lanczos_impl::DoLanczos()
{
  if(TRACE) cout << "Starting DoLanczos (lanczos_cuda.cu)" << endl;
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
  //  FreeMatrixOnDevice(&d_m_val,&d_m_col);

  FreeMemoryOnDevice(&d_phi);

  FreeMemoryOnDevice(&d_m_col);
  FreeMemoryOnDevice(&d_m_val);


}


double Lanczos_impl::Normalize(int i)
{
  if(WTRACE) cout << "in Lanczos_impl::Normalize" << endl;
  double norm=0.;
  /*
  cuda_Dznrm2(Cols,&d_phi[i*Cols],&norm);
  //  cublasDznrm2(handle,Cols,&d_phi[i*Cols],1,&norm);
  if(TRACE) cout << "Norm is " << setprecision(DEBUGPRECISION) << norm << endl;
  double factor=1./norm;
  cuda_Zdscal(Cols,&factor,&d_phi[i*Cols]);
  //  cublasZdscal(handle,Cols,&factor,&d_phi[i*Cols],1);
  */
  cudap_Normalize(Cols,&d_phi[i*Cols],&norm);
  if(TRACE) cout << "Norm is " << setprecision(DEBUGPRECISION) << norm << endl;
  return norm;
}

/*
double Lanczos_impl::GetNorm(const int n,cuda_cmplx* d_z)
{
  if(WTRACE) cout << "in Lanczos_impl::GetNorm" << endl;
  double norm=0.;
  cuda_Dznrm2(n,d_z,&norm);
  //cublasDznrm2(handle,n,d_z,1,&norm);
  return norm;
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


void Lanczos_impl::CheckOrthogonality(int n,cmplxformat* d_z,const int m)
{
  cout << "Checking orthogonality:" << endl;
  for(int i=0; i<=m; i++)
    {
      //      cuda_cmplx q=cmplx_0;
      cmplxformat qp;
      cmplxformat* d_phii=&d_phi[i*n];
      cudap_cmplxdotproduct_cmplx(n,d_phii,d_z,&qp);  // q=<phi_i|z> 
      /* 
     cuda_cmplx* d_phii=&d_phi[i*n];
      cuda_CmplexDotProductCmplx(n,d_phii,d_z,&q);
      //      cublasZdotc(handle,n,d_phii,1,d_z,1,&q); // q=<z|phi_i> 
      */

      cuda_cmplx q;
      q.x= todouble(qp.x);
      q.y= todouble(qp.y); 

      if(cuCabs(q) > 1.e-13)
	{
	  cout << "WARNING: Severe Loss of Orthogonality";
	  cout << "<phi[" << i << "]|z>=" << setprecision(DEBUGPRECISION) << q;
	  cout << endl;
	}
    }
}


void Lanczos_impl::Reorthogonalize(int n,cmplxformat* d_z,int m)
{
  if(WTRACE) cout << "In Reorthogonalize" << endl;
  for(int j=0; j<MGSREPEATSTEPS; j++) // 2 for Modified Gram-Schmidt with reorthogo..
    for(int i=0; i<=m; i++)
      {
	cmplxformat q;
	const cmplxformat* d_phii=&d_phi[i*n];
	const cmplxformat* d_zconst= d_z;
	cudap_cmplxdotproduct_cmplx(n,d_phii,d_zconst,&q);  // q=<phi_i|z> 
	cudap_cmplx_Zmaxpy(n,&q,d_z,d_phii); // |z>=|z>-q*|phi_i>; 

	//	cuda_ComplexDotProduct(n,d_phii,d_z,&q); // q=<phi_i|z> 
	//cublasZdotc(handle,n,d_phii,1,d_z,1,&q); // q=<phi_i|z> 
	//cuda_cmplx negq=cuCsub(cmplx_0,q);
	//cuda_Zaxpy(n,&negq,d_z,d_phii); // |z>=|z>-q*|phi_i>; 
	//cublasZaxpy(handle,n,&negq, d_z,1,d_phii,1); // |z>=|z>-q*|phi_i>; 

      }
  if(TRACE) cout << "Done Reorthogonalize" << endl; 
}


bool Lanczos_impl::LanczosIteration(const int m)
{
  if(WTRACE)  cout << "Starting Lanczos iteration " << m << endl;
  Niterations++;
  
  
  cmplxformat* d_phim=&d_phi[m*Cols]; //already normalized     
  cmplxformat* d_z   =&d_phi[(m+1)*Cols]; 
  
  
  // if(TRACE) cout << "Launching cuda_mat_vec_multiply_cmplx with " << blocksPerGrid << " blocks and " << threadsPerBlock  << " threads per block" << endl; 
  //cuda_mat_vec_multiply_cmplx<<<blocksPerGrid,threadsPerBlock>>>(Rows,Nc,d_m_col ,d_m_val, d_phim, d_z); // |z> = H |phi_m>  
  cudap_mat_vec_multiply_cmplx(Rows,Nc,d_m_col ,d_m_val, d_phim, d_z); // |z> = H |phi_m>  
  //cuda_mat_vec_multiply_cmplx_dd<<<blocksPerGrid,threadsPerBlock>>>(Rows,Nc,d_m_col ,d_m_val, d_phim, d_z); // |z> = H |phi_m>  
  
  err = cudaGetLastError();						     
  if(TRACE) cout << "end of cuda_mat_vec_multiply_cmplx, last error" << err << endl; 
  
  if(TRACE)
    {
      cout << "H * phi[" << m << "]" << endl;
      InspectDevice(&d_z,Cols);
    }
  


  if(TRACE) cout << "Launching ComplexDotProduct" << endl; 
  cudap_realdotproduct_cmplx(Cols,d_phim,d_z,&planczos_am); // a_m=<phi_m|z>, should be real 

  cudap_real_Zmaxpy(Cols,&planczos_am,d_z,d_phim);  // |z>=|z>-a_m*|phi_m>  

  lanczos_a[m]=todouble(planczos_am);

  if(TRACE) cout << "lanczos_a[" << m << "]=" << lanczos_a[m] << endl;

  /*
  //  cublasZdotc(handle,Cols,d_phim,1,d_z,1,&a);



  cuda_cmplx negam=make_cuDoubleComplex(-lanczos_a[m],0.);
  
  cuda_Zaxpy(Cols,&negam,d_phim,d_z); // |z>=|z>-a_m*|phi_m>  
  //cublasZaxpy(handle,Cols,&negam,d_phim,1,d_z,1); // |z>=|z>-a_m*|phi_m>  
  */

  if(m>0)
    {
      /*
      cuda_cmplx negbmm1=make_cuDoubleComplex(-lanczos_b[m-1],0.);
      cuda_cmplx* d_phimm1=&d_phi[(m-1)*Cols];
      */
      cmplxformat* d_phimm1=&d_phi[(m-1)*Cols]; //already normalized     
      realformat* bmm1_ptr= &planczos_bm; // point to the previous lanczos_bm     

      cudap_real_Zmaxpy(Cols,bmm1_ptr,d_z,d_phimm1); // |z>=|z>-b_(m-1)*|phi_(m-1)>
      //cublasZaxpy(handle,Cols,&negbmm1,d_phimm1,1,d_z,1);//|z>=|z>-b_(m-1)*|phi_(m-1)>

    }
  
#ifdef REORTHOGONALIZE
  Reorthogonalize(Cols,d_z,m);
#else

#endif
  if(TRACE) cout << "After Reorthogonalization" << endl;
  if(TRACE) CheckOrthogonality(Cols,d_z,m);

  realformat* bm_ptr= &planczos_bm;      
  
  cudap_norm_cmplx(Cols,d_z,bm_ptr); 

  lanczos_b[m]=todouble(planczos_bm);

  if(m == M-1) // end of lanczos procedure               
    {
      if(TRACE) cout << "Truncating the Lanczos procedure, reached max matrix size, lanczos_b[M-1]= " << lanczos_b[m] << " --> 0" << endl;
      return false;
    }
  
  if(TRACE) cout << "lanczos_b[" << m << "]=" << lanczos_b[m] << endl;
  
  if(abs(lanczos_b[m])<=LANCZOSEXITLIMIT){return false;} // stop the Lanczos procedure;            
 
  cudap_Zdinvscal(Cols,bm_ptr,d_z); // |z> -> |z>/bm

  //   double factor=1./beta;  
  //  cuda_Zdscal(Cols,&factor,d_z); // |z> -> |z>/beta
  //cublasZdscal(handle,Cols,&factor,d_z,1); // |z> -> |z>/beta


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

