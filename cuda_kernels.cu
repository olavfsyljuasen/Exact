#ifndef CUDA_KERNELS_CU
#define CUDA_KERNELS_CU

#include "cuda_kernels.h"

// This is the CUDA matrix vector multiply routine


__global__ void cuda_kernel_mat_vec_multiply_cmplx(
const int M, const int Nc,const int* indices , 
const cuda_cmplx* data , const cuda_cmplx* x, cuda_cmplx* y) 
{
  int row = blockDim.x * blockIdx.x + threadIdx.x ;
  
  if ( row < M )
    {
      cuda_cmplx dot;
      dot.x=0.;
      dot.y=0.;

      for ( int c = 0; c < Nc ; c ++)
	{
	  int indx=c*M+row;

	  double val_r = data[indx].x;
	  double val_i = data[indx].y;
	
	  int col = indices[indx];
	  double x_r = x[col].x;
	  double x_i = x[col].y;
	
	  dot.x = dot.x + val_r*x_r -val_i*x_i;
	  dot.y = dot.y + val_r*x_i +val_i*x_r;
	}
      y[row].x = y[row].x + dot.x;
      y[row].y = y[row].y + dot.y;
    }
}

// computes: x= x-q*y; for a real q and complex x and y
__global__ void cuda_kernel_real_Zmaxpy(const int N,const double* q,cuda_cmplx* x,const cuda_cmplx* y) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < N)
    {
      double qval=*q;

      x[i].x = x[i].x - qval*y[i].x;
      x[i].y = x[i].y - qval*y[i].y; 
    }
}

// computes: x= x-q*y; for a cmplx q and complex x and y
__global__ void cuda_kernel_cmplx_Zmaxpy(const int N,const cuda_cmplx* q,cuda_cmplx* x,const cuda_cmplx* y) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < N)
    {
      double qr = (*q).x;
      double qi = (*q).y;

      double xr=x[i].x;
      double xi=x[i].y;

      double yr=y[i].x;
      double yi=y[i].y;

      x[i].x = xr - qr*yr + qi*yi;
      x[i].y = xi - qr*yi - qi*yr; 
    }
}


//this routine computes the real part of a dotproduct between two real vectors
__global__ void cuda_kernel_realdotproduct_real(const int N,const double* x,const double* y,double* ans) 
{
  __shared__ double cache[THREADS_PER_BLOCK]; 

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  double r = 0.;

  while(i < N)
    {
      r = r + x[i]*y[i];

      i += blockDim.x * gridDim.x;
    }
  cache[threadIdx.x] = r;

  __syncthreads();


  i = blockDim.x / 2;
  while (i != 0)
    {
      if (threadIdx.x < i)
	{
	  cache[threadIdx.x] =cache[threadIdx.x]+cache[threadIdx.x + i];
	}
      __syncthreads();
      i /= 2;
    }

  if (threadIdx.x == 0)
    {
      ans[blockIdx.x] = cache[0];
    }  
}



//this routine computes the real part of a dotproduct between two complex vectors
__global__ void cuda_kernel_realdotproduct_cmplx(const int N,const cuda_cmplx* x,const cuda_cmplx* y,double* ans) 
{
  __shared__ double cache[THREADS_PER_BLOCK]; 

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  double r = 0.;

  while(i < N)
    {
      r = r + x[i].x*y[i].x + x[i].y*y[i].y;

      i += blockDim.x * gridDim.x;
    }
  cache[threadIdx.x] = r;

  __syncthreads();


  i = blockDim.x / 2;
  while (i != 0)
    {
      if (threadIdx.x < i)
	{
	  cache[threadIdx.x] =cache[threadIdx.x]+cache[threadIdx.x + i];
	}
      __syncthreads();
      i /= 2;
    }

  if (threadIdx.x == 0)
    {
      ans[blockIdx.x] = cache[0];
    }  
}


//this routine computes the dotproduct between two complex vectors
__global__ void cuda_kernel_cmplxdotproduct_cmplx(const int N,const cuda_cmplx* x,const cuda_cmplx* y,cuda_cmplx* ans) 
{
  __shared__ cuda_cmplx cache[THREADS_PER_BLOCK]; 

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  cuda_cmplx r;
  r.x=0.;
  r.y=0.;


  while(i < N)
    {
      cuda_cmplx u=x[i];
      cuda_cmplx v=y[i];

      r.x = r.x + u.x*v.x + u.y*v.y; 
      r.y = r.y - u.y*v.x + u.x*v.y;

      i += blockDim.x * gridDim.x;
    }
  cache[threadIdx.x] = r;

  __syncthreads();


  i = blockDim.x / 2;
  while (i != 0)
    {
      if (threadIdx.x < i)
	{
	  cache[threadIdx.x].x =cache[threadIdx.x].x+cache[threadIdx.x + i].x;
	  cache[threadIdx.x].y =cache[threadIdx.x].y+cache[threadIdx.x + i].y;
	}
      __syncthreads();
      i /= 2;
    }

  if (threadIdx.x == 0)
    {
      ans[blockIdx.x] = cache[0];
    }  
}


__global__ void cuda_kernel_Zdscal(const int N,const double* q,cuda_cmplx* x) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < N)
    {
      double xr=x[i].x;
      double xi=x[i].y;

      double qval= *q;

      x[i].x = xr*qval;
      x[i].y = xi*qval; 
    }
}


#endif // CUDA_KERNELS_CU
