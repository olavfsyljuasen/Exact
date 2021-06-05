#ifndef CUDA_ROUTINES_CU
#define CUDA_ROUTINES_CU

//routines that calls cuda_kernels, i.e. driver-routines for cuda calls
//routines with prefix cu as in cuRoutine can be called directly from host code.


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

#include "cuda_global.h"
#include "cuda_routines.h"
#include "cuda_kernels.h"

int cuda_mat_vec_multiply_cmplx(
const int M,const int Nc,const int* indices ,
const cuda_cmplx* data , const cuda_cmplx* x, cuda_cmplx* y)
{
  if(WTRACE) cout << "Starting cuda_mat_vec_multiply_cmplx" << endl;
  int b = (M + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  
  cuda_kernel_mat_vec_multiply_cmplx<<<b, THREADS_PER_BLOCK>>>(M,Nc,indices,data,x,y);

  if(WTRACE) cout << "Done with cuda_mat_vec_multiply_cmplx" << endl;
  return cudaSuccess;
}

int cuda_real_Zmaxpy(const int N,const double* q,cuda_cmplx* d_x,const cuda_cmplx* d_y) 
{
  if(WTRACE) cout << "Starting cuda_real_Zmaxpy" << endl;

  double* d_q=NULL;
  AllocateSpaceOnDevice(&d_q,1);
  UploadToDevice(q,1,&d_q);

  int b = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  cuda_kernel_real_Zmaxpy<<<b, THREADS_PER_BLOCK>>>(N,d_q,d_x,d_y);

  FreeMemoryOnDevice(&d_q);

  if(WTRACE) cout << "Done with cuda_real_Zmaxpy" << endl;
  return cudaSuccess;
}

int cuda_cmplx_Zmaxpy(const int N,const cuda_cmplx* q,cuda_cmplx* d_x,const cuda_cmplx* d_y) 
{
  if(WTRACE) cout << "Starting cuda_cmplx_Zmaxpy" << endl;

  cuda_cmplx* d_q=NULL;
  AllocateSpaceOnDevice(&d_q,1);
  UploadToDevice(q,1,&d_q);

  int b = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  cuda_kernel_cmplx_Zmaxpy<<<b, THREADS_PER_BLOCK>>>(N,d_q,d_x,d_y);

  FreeMemoryOnDevice(&d_q);

  if(WTRACE) cout << "Done with cuda_cmplx_Zmaxpy" << endl;
  return cudaSuccess;
}


int cuda_realdotproduct_real(const int N,const double* d_x,const double* d_y,double* ans) 
{

  if(WTRACE) cout << "Starting cuda_realdotproduct_cmplx" << endl;
  cudaError_t err;

  double c[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ c[i]=double(0.);}
  double* d_c=NULL;

  AllocateSpaceOnDevice(&d_c,BLOCKS);
  UploadToDevice(c,BLOCKS,&d_c);

  cuda_kernel_realdotproduct_real<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, d_y, d_c); 

  err = cudaGetLastError();
  if(err != cudaSuccess)
    {
      fprintf(stderr, "Failed to execute (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  DownloadFromDevice(&d_c,BLOCKS,c);
  FreeMemoryOnDevice(&d_c);

  // sum the contribution from all the blocks:
  for(int i = 1; i < BLOCKS; i++){c[0] += c[i];}

  *ans=c[0];

  if(WTRACE) cout << "Done with cuda_realdotproduct_cmplx" << endl;
  return cudaSuccess;
}



int cuda_realdotproduct_cmplx(const int N,const cuda_cmplx* d_x,const cuda_cmplx* d_y,double* ans) 
{

  if(WTRACE) cout << "Starting cuda_realdotproduct_cmplx" << endl;
  cudaError_t err;

  double c[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ c[i]=double(0.);}
  double* d_c=NULL;

  AllocateSpaceOnDevice(&d_c,BLOCKS);
  UploadToDevice(c,BLOCKS,&d_c);

  cuda_kernel_realdotproduct_cmplx<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, d_y, d_c); 

  err = cudaGetLastError();
  if(err != cudaSuccess)
    {
      fprintf(stderr, "Failed to execute (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  DownloadFromDevice(&d_c,BLOCKS,c);
  FreeMemoryOnDevice(&d_c);

  // sum the contribution from all the blocks:
  for(int i = 1; i < BLOCKS; i++){c[0] += c[i];}

  *ans=c[0];

  if(WTRACE) cout << "Done with cuda_realdotproduct_cmplx" << endl;
  return cudaSuccess;
}


int cuda_cmplxdotproduct_cmplx(const int N,const cuda_cmplx* d_x,const cuda_cmplx* d_y,cuda_cmplx* ans) 
{
  if(WTRACE) cout << "Starting cuda_cmplxdotproduct_cmplx" << endl;
  cudaError_t err;
  
  cuda_cmplx c[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ c[i].x=0.; c[i].y=0.;}
  cuda_cmplx* d_c=NULL;
  
  AllocateSpaceOnDevice(&d_c,BLOCKS);
  UploadToDevice(c,BLOCKS,&d_c);

  cuda_kernel_cmplxdotproduct_cmplx<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, d_y, d_c); 

  err = cudaGetLastError();
  if(err != cudaSuccess)
    {
      fprintf(stderr, "Failed to execute (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  DownloadFromDevice(&d_c,BLOCKS,c);
  FreeMemoryOnDevice(&d_c);

  // sum the contribution from all the blocks:
  for(int i = 1; i < BLOCKS; i++){c[0].x += c[i].x; c[0].y += c[i].y;}

  *ans=c[0];

  if(WTRACE) cout << "Done with cuda_cmplxdotproduct_cmplx" << endl;
  return cudaSuccess;
}

// computes the dot product between two real vectors
int cuRealDotProductReal(const int N,const double* x,const double* y,double* res)
{
  double* d_x=NULL;
  AllocateSpaceOnDevice(&d_x,N);
  UploadToDevice(x,N,&d_x);

  double* d_y=NULL;
  AllocateSpaceOnDevice(&d_y,N);
  UploadToDevice(y,N,&d_y);

  double rp =double(0.);

  cuda_realdotproduct_real(N,d_x,d_y,&rp); 

  *res=rp;

  FreeMemoryOnDevice(&d_x);
  FreeMemoryOnDevice(&d_y);
  
  return 0;
}

// computes the real part of the dot product between two complex vectors
int cuRealDotProductCmplx(const int N,const void* x,const void* y,double* res)
{
  cuda_cmplx* d_x=NULL;
  AllocateSpaceOnDevice(&d_x,N);
  UploadToDevice(reinterpret_cast<const cuda_cmplx*>(x),N,&d_x);  

  cuda_cmplx* d_y=NULL;
  AllocateSpaceOnDevice(&d_y,N);
  UploadToDevice(reinterpret_cast<const cuda_cmplx*>(y),N,&d_y);  

  double rp =double(0.);

  cuda_realdotproduct_cmplx(N,d_x,d_y,&rp); 

  *res=rp;

  FreeMemoryOnDevice(&d_x);
  FreeMemoryOnDevice(&d_y);

  return 0;
}


// computes the dot product between two complex vectors
int cuCmplxDotProductCmplx(int N,void* x,void* y,void* res)
{
  cuda_cmplx* d_x=NULL;
  AllocateSpaceOnDevice(&d_x,N);
  UploadToDevice(reinterpret_cast<cuda_cmplx*>(x),N,&d_x);  

  cuda_cmplx* d_y=NULL;
  AllocateSpaceOnDevice(&d_y,N);
  UploadToDevice(reinterpret_cast<cuda_cmplx*>(y),N,&d_y);  

  cuda_cmplx rp;

  cuda_cmplxdotproduct_cmplx(N,d_x,d_y,&rp); 

  cuda_cmplx* res_ptr=reinterpret_cast<cuda_cmplx*>(res);

  (*res_ptr).x=rp.x;
  (*res_ptr).y=rp.y;

  FreeMemoryOnDevice(&d_x);
  FreeMemoryOnDevice(&d_y);
  
  return 0;
}


int cuda_norm_cmplx(const int N,const cuda_cmplx* d_x,double* ans) 
{

  if(WTRACE) cout << "Starting cuda_norm_cmplx" << endl;
  cudaError_t err;

  double c[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ c[i]=double(0.);}
  double* d_c=NULL;

  AllocateSpaceOnDevice(&d_c,BLOCKS);
  UploadToDevice(&c[0],BLOCKS,&d_c);

  cuda_kernel_realdotproduct_cmplx<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, d_x, d_c); 

  err = cudaGetLastError();
  if(err != cudaSuccess)
    {
      fprintf(stderr, "Failed to execute (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  DownloadFromDevice(&d_c,BLOCKS,c);
  FreeMemoryOnDevice(&d_c);

  // sum the contribution from all the blocks:
  for(int i = 1; i < BLOCKS; i++){c[0] += c[i];}

  *ans=sqrt(c[0]);

  if(WTRACE) cout << "Done with cuda_norm_cmplx" << endl;
  return cudaSuccess;
}

// divide the complex vector x by  a real number: x=x/q
int cuda_Zdinvscal(int N,double* q,cuda_cmplx* d_x)
{
  if(WTRACE) cout << "Starting cuda_Zdinvscal " << endl;

  double s= 1./(*q); // s= 1/q taking the inverse
  
  double* d_s=NULL;
  AllocateSpaceOnDevice(&d_s,1);
  UploadToDevice(&s,1,&d_s);

  int b = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  cuda_kernel_Zdscal<<<b, THREADS_PER_BLOCK>>>(N,d_s,d_x);

  FreeMemoryOnDevice(&d_s);

  if(WTRACE) cout << "Done with cuda_Zdinvscal" << endl;
  return cudaSuccess;
}


// multiply the complex vector x by  a real number: x=x*q
int cuda_Zdscal(int N,double* q,cuda_cmplx* d_x)
{
  if(WTRACE) cout << "Starting cuda_Zdscal " << endl;

  double* d_s=NULL;
  AllocateSpaceOnDevice(&d_s,1);
  UploadToDevice(q,1,&d_s);

  int b = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  cuda_kernel_Zdscal<<<b, THREADS_PER_BLOCK>>>(N,d_s,d_x);

  FreeMemoryOnDevice(&d_s);

  if(WTRACE) cout << "Done with cuda_Zdscal" << endl;
  return cudaSuccess;
}


int cuda_Normalize(const int N,cuda_cmplx* d_x,double* ans) 
{
  if(WTRACE) cout << "Starting cuda_Normalize" << endl;
  double norm=0.;
  cuda_norm_cmplx(N,d_x,&norm); 
  cuda_Zdinvscal(N,&norm,d_x);
  *ans=norm;
  return cudaSuccess;  
}


#endif // CUDA_ROUTINES_CU
