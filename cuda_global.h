#ifndef CUDA_GLOBAL_H
#define CUDA_GLOBAL_H

#include <complex>
#include <iostream>
using namespace std;

#include "cuda_runtime.h"
#include <cublas_v2.h>
#include "cuComplex.h"

#include "sparsematrix.h"
#include "global.h"

typedef cuDoubleComplex cuda_cmplx; 

ostream& operator<<(ostream& os,cuda_cmplx a);

const cuda_cmplx cmplx_0 = make_cuDoubleComplex(0.,0.);
const cuda_cmplx cmplx_1 = make_cuDoubleComplex(1.,0.);

#define THREADS_PER_BLOCK 256
#define BLOCKS 32

const int NTHREADSPERBLOCK=THREADS_PER_BLOCK;

//const bool REORTHOGONALIZE=true;

int AllocateSpaceOnDevice(cuda_cmplx** d_phi_ptr,const int N);
int AllocateSpaceOnDevice(double** d_phi_ptr,const int N);
//int AllocateSpaceOnDevice(void** d_phi_ptr,const int N);

int InitializeSpaceOnDevice(const int N,void** d_phi_ptr);
int UploadToDevice(const void* phi,const int N,cuda_cmplx** d_phi_ptr);
int UploadToDevice(const cuda_cmplx* phi,const int N,cuda_cmplx** d_phi_ptr);
int UploadToDevice(const double* phi,const int N,double** d_phi_ptr);
int UploadToDevice(const void* phi,const int N,void** d_phi_ptr);

//int DownloadFromDevice(cuda_cmplx** d_phi_ptr,const int N,MyField* phi)
int DownloadFromDevice(cuda_cmplx** d_phi_ptr,const int N,void *phi);
int DownloadFromDevice(double** d_phi_ptr,const int N,void *phi);
int DownloadFromDevice(void** d_phi_ptr,const int N,void *phi);


int FreeMemoryOnDevice(cuda_cmplx** d_phi);
int FreeMemoryOnDevice(double** d_phi);
int FreeMemoryOnDevice(int** d_phi);

int UploadMatrixToDevice(MyMatrix& H,cuda_cmplx** d_m_val,int** d_m_col);
int FreeMatrixOnDevice(cuda_cmplx** d_m_val_ptr,int** d_m_col_ptr);



#endif // CUDA_GLOBAL_H
