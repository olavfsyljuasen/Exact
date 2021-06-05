#include<iostream>
#include<vector>
#include<iostream>
#include<complex>
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

#include "banddiag.h"



//This routine uses Lapack to diagonalize the Lanczos symmetric triband matrix
//
//Both eigenvalues and eigenvectors are obtained

//#ifdef MACOSXBLAS
//#include<accelerate/accelerate.h>
//#endif

//
//#ifdef OPENBLAS
//#include<OpenBLAS/cblas.h>
//#endif

#ifdef LAPACKE
namespace mylapacke{
#define LAPACK_DISABLE_NAN_CHECK
#define LAPACK_COMPLEX_CUSTOM
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#include<lapacke.h>
}
#elif MKL_LAPACKE
#define LAPACK_DISABLE_NAN_CHECK
#define LAPACK_COMPLEX_CUSTOM
#include "mkl_lapacke.h"
#endif



/*
extern "C" {

#ifdef OPENBLAS
void dsteqr_(char* COMPZ,int* N,double* D,double* E,double* Z,int* LDZ,double* WORK,int* INFO ); //QR method
#endif
}
*/

LanczosMatrix::LanczosMatrix(int N_in,vector<double>& diag_in,vector<double>& offd_in):N(N_in),diag(diag_in),offd(offd_in),evec(N*N)
{
}

void LanczosMatrix::Diagonalize()
{
  if(WTRACE) cout << "in banddiag.C: Diagonalizing Lanczos matrix: " << endl;
 
  if(TRACE)
    {
      for(int i=0; i<N-1; i++)
          cout << diag[i] << " " << offd[i] << endl;
      cout << diag[N-1] << endl;
    }


  char compz='I'; // also include eigenvectors
  vector<double> Work(4*N);
  int info=0;



#ifdef LAPACKE
  //  LAPACKE_dsteqr(&COMPZ,&N,&diag[0],&offd[0],&evec[0],&N,&Work[0],&info);
  mylapacke::LAPACKE_dsteqr_work(LAPACK_COL_MAJOR,compz,N,&diag[0], &offd[0],&evec[0], N,&Work[0]);

#elif MKL_LAPACKE
  LAPACKE_dsteqr(LAPACK_COL_MAJOR,compz,N,&diag[0], &offd[0],&evec[0], N);
#endif
  

  if(TRACE)
    {
      cout << "Eigenvalues and eigenvectors: " << endl;
      for(int i=0; i<N; i++)
	{
	  cout << diag[i] << " : ";
	  for(int j=0; j<N; j++) cout << evec[i*N+j] << " ";
	  cout << endl;
	}
    }

  /* 
#ifdef OPENBLAS
  dsteqr_(&compz,&N,&diag[0],&offd[0],&evec[0],&N,&Work[0],&info);
  //dsteqr_(&COMPZ,&N,&diag[0],&offd[0],&evec[0],&N,&Work[0],&info);
#endif

#ifdef MACOSXBLAS
  dsteqr_(&COMPZ,&N,&diag[0],&offd[0],&evec[0],&N,&Work[0],&info);
#endif
  */

  if(TRACE)
    {
      cout << "Eigenvalues and eigenvectors: " << endl;
      for(int i=0; i<N; i++)
        {
          cout << diag[i] << " : ";
          for(int j=0; j<N; j++) cout << evec[i*N+j] << " ";
          cout << endl;
        }
    }


}
