#ifndef CUDA_ROUTINES_CU
#define CUDA_ROUTINES_CU

//routines that calls cuda_kernels, i.e. driver-routines for cuda calls
//routines with prefix cu as in cuRoutine can be called directly from host code.

#include "cuda_global.h"

int cuda_mat_vec_multiply_cmplx(
const int M,const int Nc,const int* indices ,
const cuda_cmplx* data , const cuda_cmplx* x, cuda_cmplx* y);


int cuda_real_Zmaxpy(const int N,const double* q,cuda_cmplx* d_x,const cuda_cmplx* d_y); 
int cuda_cmplx_Zmaxpy(const int N,const cuda_cmplx* q,cuda_cmplx* d_x,const cuda_cmplx* d_y); 

int cuda_realdotproduct_real(const int N,const double* d_x,const double* d_y,double* ans); 
int cuda_realdotproduct_cmplx(const int N,const cuda_cmplx* d_x,const cuda_cmplx* d_y,double* ans); 
int cuda_cmplxdotproduct_cmplx(const int N,const cuda_cmplx* d_x,const cuda_cmplx* d_y,cuda_cmplx* ans);

int cuRealDotProductReal(const int N,const double* x,const double* y,double* res);
int cuRealDotProductCmplx(const int N,const void* x,const void* y,double* res);
int cuCmplxDotProductCmplx(int N,void* x,void* y,void* res);

int cuda_norm_cmplx(const int N,const cuda_cmplx* d_x,double* ans); 

int cuda_Zdinvscal(int N,double* q,cuda_cmplx* d_x);
int cuda_Zdscal(int N,double* q,cuda_cmplx* d_x);
int cuda_Normalize(const int N,cuda_cmplx* d_x,double* ans); 


#endif // CUDA_ROUTINES_H
