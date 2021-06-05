#ifndef FULLDIAGONALIZATION_C
#define FULLDIAGONALIZATION_C


#include<iostream>
#include<vector>
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


#include "sparsematrix.h"
#include "global.h"

#include "fulldiagonalization.h"


#ifdef LAPACKE
namespace mylapacke{
#define LAPACK_DISABLE_NAN_CHECK
#define LAPACK_COMPLEX_CUSTOM
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#include<lapacke.h>
}
#elif  MKL_LAPACKE
//namespace mylapacke{
#define LAPACK_DISABLE_NAN_CHECK
#define LAPACK_COMPLEX_CUSTOM
//#define lapack_complex_float std::complex<float>
//#define lapack_complex_double std::complex<double>
#include "mkl_lapacke.h"
//}
#endif

/*
#ifdef OPENBLAS
extern "C" 
{
  //  void zheev_(char*,char*, int*, complex<double>* A, int*, double*, complex<double>*,int*, double*, int*)
  void zheev_(char*,char*, int*, double* A, int*, double*,double*,int*, double*, int*);
}
#endif
*/


#ifdef MACOSXBLAS
#include<accelerate/accelerate.h>
#endif



FullDiagonalization::FullDiagonalization(MyMatrix& H):JOBS('V'),UPLO('U'),N(H.Ncols()),LDA(N),A(N*N),B(N*N),W(N),LWORK(N*N),WORK(LWORK),RWORK(3*N-2),INFO(0)
{
  // While we like to keep the row-major convention on matrices
  // it is convenient here to convert the matrix to a column-major order and do the Lapack diagonalization in Fortran convention
  // with column major order. This lets us omit transposes in lapacke and also a final transpose at the end.

  if(TRACE){

    cout << "Matrix in memory before conversion:" << endl;
    
    cout << H << endl;

    for(int j=0; j<min(H.Nrows(),10); j++)
      {
	for(int i=0; i<min(H.Ncols(),10); i++)
	  {
	    cout << H(j,i) << " ";
	  }
	cout << endl;
      }
  }
  
  H.ConvertToDenseMatrix(A,false); // false:convert to column-major order, no conversion needed on output.
 
  if(TRACE){
    cout << "Matrix in memory before diag:" << endl;
    for(int i=0; i<A.size(); i++) cout << A[i] << " ";
    cout << endl;
  }

  /*
#ifdef OPENBLAS
  H.ConvertToDenseMatrix(A,false); 
  zheev_(&JOBS,&UPLO,&N,reinterpret_cast<double*>(&A[0]),&LDA, &W[0],reinterpret_cast<double*>(&WORK[0]),&LWORK, &RWORK[0],&INFO);
#endif
  */

#ifdef LAPACKE
  mylapacke::LAPACKE_zheev_work(LAPACK_COL_MAJOR, JOBS, UPLO,
  			N, &A[0],LDA, &W[0],&WORK[0],LWORK,&RWORK[0]);
#elif MKL_LAPACKE
  
  LAPACKE_zheev(LAPACK_COL_MAJOR, JOBS, UPLO,N, reinterpret_cast<lapack_complex_double*>(&A[0]),LDA, &W[0]);
#endif

  if(TRACE){
    cout << "Matrix in memory after diag:" << endl;
    for(int i=0; i<A.size(); i++) cout << A[i] << " ";
    cout << endl;
  }
  if(TRACE)
    for(int i=0; i<N; i++)
      {
	cout << "Eigenvalue #" << i << " = " << W[i] << endl;
	cout << "Eigenvector: ";
	
	MyField* start=GetEigenvector(i);       
	for(int j=0; j<N; j++)
	  cout << start[j] << " ";
	cout << endl;
      }
  
}


vector<double>& FullDiagonalization::GetEigenvalues(){return W;}
MyField* FullDiagonalization::GetEigenvector(const int i){return &A[i*N];}
int FullDiagonalization::GetNeigenvalues(){return N;}




#endif  //FULLDIAGONALIZATION_C
