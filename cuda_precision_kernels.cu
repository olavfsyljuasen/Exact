#ifndef CUDA_PRECISION_KERNELS_CU
#define CUDA_PRECISION_KERNELS_CU

#include "cuda_precision_kernels.h"
#include "gqd.cu"

struct gdd_cmplx
{
  gdd_real x; 
  gdd_real y; 
  __device__ gdd_cmplx(){}; 
  __device__ gdd_cmplx(double xd,double yd){x=make_dd(xd); y=make_dd(yd);}
};
struct gqd_cmplx{gqd_real x; gqd_real y; gqd_cmplx(){}; gqd_cmplx(double xd,double yd){x=make_qd(xd); y=make_qd(yd);}};

#ifdef DOUBLEDOUBLE
#define realformat  gdd_real
#define cmplxformat gdd_cmplx
#define todouble    to_double
#define makereal    make_dd
#elif defined QUADDOUBLE
#define realformat  gqd_real
#define cmplxformat gqd_cmplx
#define todouble    to_double
#define makereal    make_qd
#else
#define realformat  double
#define cmplxformat cuda_cmplx
#define todouble    double
#define makereal    double
#endif


// This is the CUDA matrix vector multiply routine


__global__ void cudap_kernel_mat_vec_multiply_cmplx(
const int M, const int Nc,const int* indices , 
const cuda_cmplx* data , const void* x, void* y) 
{
  int row = blockDim.x * blockIdx.x + threadIdx.x ;
  
  if ( row < M )
    {
      cmplxformat dot=cmplxformat(0.,0.);
      const cmplxformat* xp=reinterpret_cast<const cmplxformat*>(x);
      cmplxformat* yp=reinterpret_cast<cmplxformat*>(y);

      for ( int c = 0; c < Nc ; c ++)
	{
	  int indx=c*M+row;

	  realformat val_r = makereal(data[indx].x);
	  realformat val_i = makereal(data[indx].y);
	
	  int col = indices[indx];
	  realformat x_r = xp[col].x;
	  realformat x_i = xp[col].y;
	
	  dot.x = dot.x + val_r*x_r -val_i*x_i;
	  dot.y = dot.y + val_r*x_i +val_i*x_r;
	}
      yp[row].x = yp[row].x + dot.x;
      yp[row].y = yp[row].y + dot.y;
    }
}

// computes: x= x-q*y; for a real q and complex x and y
__global__ void cudap_kernel_real_Zmaxpy(const int N,const void* q,void* x,const void* y) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < N)
    {
      cmplxformat* xp=reinterpret_cast<cmplxformat*>(x);
      const cmplxformat* yp=reinterpret_cast<const cmplxformat*>(y);
      
      const realformat* qp =reinterpret_cast<const realformat*>(q);
      realformat qval=*qp;

      realformat xr=xp[i].x;
      realformat xi=xp[i].y;

      realformat yr=yp[i].x;
      realformat yi=yp[i].y;

      xp[i].x = xr - qval*yr;
      xp[i].y = xi - qval*yi; 
    }
}

// computes: x= x-q*y; for a cmplx q and complex x and y
__global__ void cudap_kernel_cmplx_Zmaxpy(const int N,const void* q,void* x,const void* y) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < N)
    {
      cmplxformat* xp=reinterpret_cast<cmplxformat*>(x);
      const cmplxformat* yp=reinterpret_cast<const cmplxformat*>(y);
      
      const cmplxformat* qp =reinterpret_cast<const cmplxformat*>(q);

      realformat qr = (*qp).x;
      realformat qi = (*qp).y;

      realformat xr=xp[i].x;
      realformat xi=xp[i].y;

      realformat yr=yp[i].x;
      realformat yi=yp[i].y;

      xp[i].x = xr - qr*yr + qi*yi;
      xp[i].y = xi - qr*yi - qi*yr; 
    }
}

//this routine computes the real part of a dotproduct between two real vectors
__global__ void cudap_kernel_realdotproduct_real(const int N,const void* x,const void* y,void* ans) 
{
  __shared__ realformat cache[THREADS_PER_BLOCK]; 

  realformat* ansp=reinterpret_cast<realformat*>(ans); 

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  realformat r = makereal(0.);

  const realformat* xp=reinterpret_cast<const realformat*>(x);
  const realformat* yp=reinterpret_cast<const realformat*>(y);

  while(i < N)
    {
      r = r + xp[i]*yp[i];

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
      ansp[blockIdx.x] = cache[0];
    }  
}



//this routine computes the real part of a dotproduct between two complex vectors
__global__ void cudap_kernel_realdotproduct_cmplx(const int N,const void* x,const void* y,void* ans) 
{
  __shared__ realformat cache[THREADS_PER_BLOCK]; 

  realformat* ansp=reinterpret_cast<realformat*>(ans); 

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  realformat r = makereal(0.);

  const cmplxformat* xp=reinterpret_cast<const cmplxformat*>(x);
  const cmplxformat* yp=reinterpret_cast<const cmplxformat*>(y);

  while(i < N)
    {
      cmplxformat u=xp[i];
      cmplxformat v=yp[i];

      r = r + u.x*v.x; 
      r = r + u.y*v.y;

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
      ansp[blockIdx.x] = cache[0];
    }  
}


//this routine computes the dotproduct between two complex vectors
__global__ void cudap_kernel_cmplxdotproduct_cmplx(const int N,const void* x,const void* y,void* ans) 
{
  __shared__ cmplxformat cache[THREADS_PER_BLOCK]; 

  cmplxformat* ansp=reinterpret_cast<cmplxformat*>(ans); 

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  cmplxformat r = cmplxformat(0.,0.);

  const cmplxformat* xp=reinterpret_cast<const cmplxformat*>(x);
  const cmplxformat* yp=reinterpret_cast<const cmplxformat*>(y);

  while(i < N)
    {
      cmplxformat u=xp[i];
      cmplxformat v=yp[i];

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
      ansp[blockIdx.x] = cache[0];
    }  
}


__global__ void cudap_kernel_Zdscal(const int N,const void* q,void* x) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < N)
    {
      cmplxformat* xp = reinterpret_cast<cmplxformat*>(x);
    
      realformat xr=xp[i].x;
      realformat xi=xp[i].y;

      const realformat* qp = reinterpret_cast<const realformat*>(q);
      realformat qval= *qp;

      xp[i].x = xr*qval;
      xp[i].y = xi*qval; 
    }
}

// takes an array of cuda_cmplx and makes it into an array with higher precision
__global__ void cudap_kernel_make_precision_cmplx(const int N,const void* x,void* xx) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < N)
    {
      cmplxformat* xxp=reinterpret_cast<cmplxformat*>(xx);
      const cuda_cmplx* xp=reinterpret_cast<const cuda_cmplx*>(x);

      xxp[i].x=makereal(xp[i].x);
      xxp[i].y=makereal(xp[i].y);
    }
}

// takes an array of doubles and makes it into an array with higher precision
__global__ void cudap_kernel_make_precision_real(const int N,const double* x,void* xx) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < N)
    {
      realformat* xxp=reinterpret_cast<realformat*>(xx);
      xxp[i]=makereal(x[i]);
    }
}

// takes an array of cmplxformat and make it into an array of cuda_cmplx
__global__ void cudap_kernel_make_cmplx(const int N,const void* xx,void* x) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < N)
    {
      const cmplxformat* xxp=reinterpret_cast<const cmplxformat*>(xx);
      cuda_cmplx* xp=reinterpret_cast<cuda_cmplx*>(x);

      xp[i].x=todouble(xxp[i].x);
      xp[i].y=todouble(xxp[i].y);
    }
}

// takes an array of realformat and make it into an array of doubles
__global__ void cudap_kernel_make_double(const int N,const void* xx,double* x) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < N)
    {
      const realformat* xxp=reinterpret_cast<const realformat*>(xx);
      x[i]=todouble(xxp[i]);
    }
}





#endif // CUDA_PRECISION_KERNELS_CU
