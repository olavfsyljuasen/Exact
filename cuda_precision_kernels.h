#ifndef CUDA_PRECISION_KERNELS_H
#define CUDA_PRECISION_KERNELS_H

using namespace std;

#include "cuda_global.h"


__global__ void cudap_kernel_mat_vec_multiply_cmplx(const int M, const int Nc,const int* indices , const cuda_cmplx* data , const void* x, void* y); 

__global__ void cudap_kernel_real_Zmaxpy(const int N,const void* q,void* x,const void* y);

__global__ void cudap_kernel_cmplx_Zmaxpy(const int N,const void* q,void* x,const void* y); 

__global__ void cudap_kernel_realdotproduct_real(const int N,const void* x,const void* y,void* ans); 
__global__ void cudap_kernel_realdotproduct_cmplx(const int N,const void* x,const void* y,void* ans); 
__global__ void cudap_kernel_cmplxdotproduct_cmplx(const int N,const void* x,const void* y,void* ans); 

__global__ void cudap_kernel_Zdscal(const int N,const void* q,void* x); 

__global__ void cudap_kernel_make_precision_cmplx(const int N,const void* x,void* xx); 
__global__ void cudap_kernel_make_precision_real(const int N,const double* x,void* xx); 
__global__ void cudap_kernel_make_cmplx(const int N,const void* xx,void* x); 
__global__ void cudap_kernel_make_double(const int N,const void* xx,double* x); 


#endif // CUDA_PRECISION_KERNELS_H
