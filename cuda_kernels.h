#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include "cuda_global.h"

__global__ void cuda_kernel_mat_vec_multiply_cmplx(
const int M, const int Nc,const int* indices , 
const cuda_cmplx* data , const cuda_cmplx* x, cuda_cmplx* y); 

__global__ void cuda_kernel_real_Zmaxpy(const int N,const double* q,cuda_cmplx* x,const cuda_cmplx* y); 

__global__ void cuda_kernel_cmplx_Zmaxpy(const int N,const cuda_cmplx* q,cuda_cmplx* x,const cuda_cmplx* y); 

__global__ void cuda_kernel_realdotproduct_real(const int N,const double* x,const double* y,double* ans); 

__global__ void cuda_kernel_realdotproduct_cmplx(const int N,const cuda_cmplx* x,const cuda_cmplx* y,double* ans); 

__global__ void cuda_kernel_cmplxdotproduct_cmplx(const int N,const cuda_cmplx* x,const cuda_cmplx* y,cuda_cmplx* ans);

__global__ void cuda_kernel_Zdscal(const int N,const double* q,cuda_cmplx* x); 


#endif // CUDA_KERNELS_H
