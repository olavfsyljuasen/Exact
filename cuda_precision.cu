#ifndef CUDA_PRECISION_CU
#define CUDA_PRECISION_CU

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
#include "cuda_precision.h"
#include "cuda_precision_kernels.h"

//#include "precision_types.h"


int cudap_mat_vec_multiply_cmplx(
const int M,const int Nc,const int* indices ,
const void* data , const cmplxformat* x, cmplxformat* y)
{
  if(WTRACE) cout << "Starting cudap_mat_vec_multiply_cmplx" << endl;
  int b = (M + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  
  const cuda_cmplx* datap=reinterpret_cast<const cuda_cmplx*>(data);
  const void* xp=reinterpret_cast<const void*>(x);
  void* yp=reinterpret_cast<void*>(y);

  cudap_kernel_mat_vec_multiply_cmplx<<<b, THREADS_PER_BLOCK>>>(M,Nc,indices,datap,xp,yp);

  if(WTRACE) cout << "Done with cudap_mat_vec_multiply_cmplx" << endl;
  return cudaSuccess;
}

int cudap_real_Zmaxpy(const int N,const realformat* q,cmplxformat* d_x,const cmplxformat* d_y) 
{
  if(WTRACE) cout << "Starting cudap_real_Zmaxpy" << endl;

  realformat* d_q=NULL;
  AllocateSpaceOnDevice(&d_q,1);
  UploadToDevice(q,1,&d_q);

  int b = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  cudap_kernel_real_Zmaxpy<<<b, THREADS_PER_BLOCK>>>(N,d_q,d_x,d_y);

  FreeMemoryOnDevice(&d_q);

  if(WTRACE) cout << "Done with cudap_real_Zmaxpy" << endl;
  return cudaSuccess;
}

int cudap_cmplx_Zmaxpy(const int N,const cmplxformat* q,cmplxformat* d_x,const cmplxformat* d_y) 
{
  if(WTRACE) cout << "Starting cudap_cmplx_Zmaxpy" << endl;

  cmplxformat* d_q=NULL;
  AllocateSpaceOnDevice(&d_q,1);
  UploadToDevice(q,1,&d_q);

  int b = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  cudap_kernel_cmplx_Zmaxpy<<<b, THREADS_PER_BLOCK>>>(N,d_q,d_x,d_y);

  FreeMemoryOnDevice(&d_q);

  if(WTRACE) cout << "Done with cuda_cmplx_Zmaxpy" << endl;
  return cudaSuccess;
}


int cudap_realdotproduct_real(const int N,const realformat* d_x,const realformat* d_y,realformat* ans) 
{

  if(WTRACE) cout << "Starting cudap_realdotproduct_cmplx" << endl;
  cudaError_t err;

  realformat c[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ c[i]=realformat(0.);}
  realformat* d_c=NULL;

  AllocateSpaceOnDevice(&d_c,BLOCKS);
  UploadToDevice(c,BLOCKS,&d_c);

  cudap_kernel_realdotproduct_real<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, d_y, d_c); 

  err = cudaGetLastError();
  if(err != cudaSuccess)
    {
      fprintf(stderr, "Failed to execute (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  DownloadFromDevice(&d_c,BLOCKS,c);
  FreeMemoryOnDevice(&d_c);

  // sum the contribution from all the blocks:
  unsigned int old_cw;
  fpu_fix_start(&old_cw);
  for(int i = 1; i < BLOCKS; i++){c[0] += c[i];}
  fpu_fix_end(&old_cw); 

  *ans=c[0];

  if(WTRACE) cout << "Done with cuda_realdotproduct_cmplx" << endl;
  return cudaSuccess;
}



int cudap_realdotproduct_cmplx(const int N,const cmplxformat* d_x,const cmplxformat* d_y,realformat* ans) 
{

  if(WTRACE) cout << "Starting cudap_realdotproduct_cmplx" << endl;
  cudaError_t err;

  realformat c[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ c[i]=realformat(0.);}
  realformat* d_c=NULL;

  AllocateSpaceOnDevice(&d_c,BLOCKS);
  UploadToDevice(c,BLOCKS,&d_c);

  cudap_kernel_realdotproduct_cmplx<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, d_y, d_c); 

  err = cudaGetLastError();
  if(err != cudaSuccess)
    {
      fprintf(stderr, "Failed to execute (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  DownloadFromDevice(&d_c,BLOCKS,c);
  FreeMemoryOnDevice(&d_c);

  // sum the contribution from all the blocks:
  unsigned int old_cw;
  fpu_fix_start(&old_cw);
  for(int i = 1; i < BLOCKS; i++){c[0] += c[i];}
  fpu_fix_end(&old_cw); 


  *ans=c[0];

  if(WTRACE) cout << "Done with cudap_realdotproduct_cmplx" << endl;
  return cudaSuccess;
}


int cudap_cmplxdotproduct_cmplx(const int N,const cmplxformat* d_x,const cmplxformat* d_y,cmplxformat* ans) 
{
  if(WTRACE) cout << "Starting cudap_cmplxdotproduct_cmplx" << endl;
  cudaError_t err;
  
  cmplxformat c[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ c[i]=cmplxformat(0.,0.);}
  cmplxformat* d_c=NULL;
  
  AllocateSpaceOnDevice(&d_c,BLOCKS);
  UploadToDevice(c,BLOCKS,&d_c);

  cudap_kernel_cmplxdotproduct_cmplx<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, d_y, d_c); 

  err = cudaGetLastError();
  if(err != cudaSuccess)
    {
      fprintf(stderr, "Failed to execute (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  DownloadFromDevice(&d_c,BLOCKS,c);
  FreeMemoryOnDevice(&d_c);

  // sum the contribution from all the blocks:
  unsigned int old_cw;
  fpu_fix_start(&old_cw);
  for(int i = 1; i < BLOCKS; i++){c[0].x += c[i].x; c[0].y += c[i].y;}
  fpu_fix_end(&old_cw); 

  *ans=c[0];

  if(WTRACE) cout << "Done with cudap_cmplxdotproduct_cmplx" << endl;
  return cudaSuccess;
}

// computes the dot product between two real vectors
int cupRealDotProductReal(const int N,const double* x,const double* y,double* res)
{
  realformat* d_x=NULL;
  AllocateSpaceOnDevice(&d_x,N);
  UploadToDeviceAndExpand(x,N,&d_x);

  realformat* d_y=NULL;
  AllocateSpaceOnDevice(&d_y,N);
  UploadToDeviceAndExpand(y,N,&d_y);

  realformat rp =realformat(0.);

  cudap_realdotproduct_real(N,d_x,d_y,&rp); 

  *res=todouble(rp);

  FreeMemoryOnDevice(&d_x);
  FreeMemoryOnDevice(&d_y);
  
  return 0;
}

// computes the real part of the dot product between two complex vectors
int cupRealDotProductCmplx(const int N,const void* x,const void* y,double* res)
{
  cmplxformat* d_x=NULL;
  AllocateSpaceOnDevice(&d_x,N);
  UploadToDeviceAndExpand(x,N,&d_x);  

  cmplxformat* d_y=NULL;
  AllocateSpaceOnDevice(&d_y,N);
  UploadToDeviceAndExpand(y,N,&d_y);  

  realformat rp =realformat(0.);

  cudap_realdotproduct_cmplx(N,d_x,d_y,&rp); 

  *res=todouble(rp);

  FreeMemoryOnDevice(&d_x);
  FreeMemoryOnDevice(&d_y);

  return 0;
}

/*
int cuConvertToDouble(void* p,double* d)
{
  realformat pval=reinterpret_cast<realformat>(*p);
  *d = todouble(pval);
}
*/

// computes the dot product between two complex vectors
int cupCmplxDotProductCmplx(int N,void* x,void* y,void* res)
{
  cmplxformat* d_x=NULL;
  AllocateSpaceOnDevice(&d_x,N);
  UploadToDeviceAndExpand(x,N,&d_x);  

  cmplxformat* d_y=NULL;
  AllocateSpaceOnDevice(&d_y,N);
  UploadToDeviceAndExpand(y,N,&d_y);  

  cmplxformat rp =cmplxformat(0.,0.);

  cudap_cmplxdotproduct_cmplx(N,d_x,d_y,&rp); 

  cuda_cmplx* res_ptr=reinterpret_cast<cuda_cmplx*>(res);

  (*res_ptr).x=todouble(rp.x);
  (*res_ptr).y=todouble(rp.y);

  FreeMemoryOnDevice(&d_x);
  FreeMemoryOnDevice(&d_y);
  
  return 0;
}


int cupNormCmplx(int N,void* x,double* res)
{
  cmplxformat* d_x=NULL;
  AllocateSpaceOnDevice(&d_x,N);
  UploadToDeviceAndExpand(x,N,&d_x);  


  realformat res_prec=realformat(0.);;

  cudap_norm_cmplx(N,d_x,&res_prec); 


  (*res)=todouble(res_prec);

  FreeMemoryOnDevice(&d_x);
  
  return 0;
}

int cupNormalize(int N,void* x,double* res)
{
  cmplxformat* d_x=NULL;
  AllocateSpaceOnDevice(&d_x,N);
  UploadToDeviceAndExpand(x,N,&d_x);  


  realformat res_prec=realformat(0.);;

  cudap_norm_cmplx(N,d_x,&res_prec); 
  cudap_Zdinvscal(N,&res_prec,d_x);

  DownloadFromDeviceAndContract(&d_x,N,x);

  (*res)=todouble(res_prec);

  FreeMemoryOnDevice(&d_x);
  
  return 0;
}


int cudap_norm_cmplx(const int N,const cmplxformat* d_x,void* ans) 
{

  if(WTRACE) cout << "Starting cudap_norm_cmplx" << endl;
  cudaError_t err;

  realformat c[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ c[i]=realformat(0.);}
  realformat* d_c=NULL;

  AllocateSpaceOnDevice(&d_c,BLOCKS);
  UploadToDevice(&c[0],BLOCKS,&d_c);

  cudap_kernel_realdotproduct_cmplx<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, d_x, d_c); 

  err = cudaGetLastError();
  if(err != cudaSuccess)
    {
      fprintf(stderr, "Failed to execute (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  DownloadFromDevice(&d_c,BLOCKS,c);
  FreeMemoryOnDevice(&d_c);

  // sum the contribution from all the blocks:
  unsigned int old_cw;
  fpu_fix_start(&old_cw);
  for(int i = 1; i < BLOCKS; i++){c[0] += c[i];}
  fpu_fix_end(&old_cw); 

  realformat* ansp = reinterpret_cast<realformat*>(ans);
  *ansp=sqrt(c[0]);

  if(WTRACE) cout << "Done with cudap_norm_cmplx" << endl;
  return cudaSuccess;
}

// divide the complex vector x by  a real number: x=x/q
int cudap_Zdinvscal(int N,realformat* q,cmplxformat* d_x)
{
  if(WTRACE) cout << "Starting cudap_Zdinvscal " << endl;

  realformat s= inv(*q); // s= 1/q taking the inverse
  
  realformat* d_s=NULL;
  AllocateSpaceOnDevice(&d_s,1);
  UploadToDevice(&s,1,&d_s);

  int b = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  cudap_kernel_Zdscal<<<b, THREADS_PER_BLOCK>>>(N,d_s,d_x);

  FreeMemoryOnDevice(&d_s);

  if(WTRACE) cout << "Done with cudap_Zdinvscal" << endl;
  return cudaSuccess;
}


// multiply the complex vector x by  a real number: x=x*q
int cudap_Zdscal(int N,realformat* q,cmplxformat* d_x)
{
  if(WTRACE) cout << "Starting cudap_Zdscal " << endl;

  realformat* d_s=NULL;
  AllocateSpaceOnDevice(&d_s,1);
  UploadToDevice(q,1,&d_s);

  int b = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  cudap_kernel_Zdscal<<<b, THREADS_PER_BLOCK>>>(N,d_s,d_x);

  FreeMemoryOnDevice(&d_s);

  if(WTRACE) cout << "Done with cudap_Zdscal" << endl;
  return cudaSuccess;
}


int cudap_Normalize(const int N,cmplxformat* d_x,double* ans) 
{
  if(WTRACE) cout << "Starting cudap_Normalize" << endl;
  realformat norm=makereal(0.);
  cudap_norm_cmplx(N,d_x,&norm); 
  cudap_Zdinvscal(N,&norm,d_x);
  *ans=todouble(norm);
  return cudaSuccess;  
}


int AllocateSpaceOnDevice(cmplxformat** d_p,const int N)
{
  if(WTRACE) cout << "In AllocateSpaceOnDevice (cmplxformat)" << endl;	
  if(TRACE) cout << "Allocating " << N*sizeof(cmplxformat) << " bytes on device" << endl;

  if (cudaMalloc(d_p, N*sizeof(cmplxformat)) != cudaSuccess)
    {
      fprintf(stderr, "!device memory allocatioNerror (AllocateSpaceOnDevice)\n");
      return EXIT_FAILURE;
    }
  if(WTRACE) cout << "Done with AllocateSpaceOnDevice (cmplxformat)" << endl;	
  return cudaSuccess;
}

int AllocateSpaceOnDevice(realformat** d_p,const int N)
{
  if(WTRACE) cout << "In AllocateSpaceOnDevice (realformat)" << endl;	
  if(TRACE) cout << "Allocating " << N*sizeof(realformat) << " bytes on device" << endl;

  if (cudaMalloc(d_p, N*sizeof(realformat)) != cudaSuccess)
    {
      fprintf(stderr, "!!!! device memory allocatioNerror (AllocateSpaceOnDevice)\n");
      return EXIT_FAILURE;
    }
    
  if(WTRACE) cout << "Done with AllocateSpaceOnDevice (realformat)" << endl;	
  return cudaSuccess;
}

int UploadToDevice(const cmplxformat* phi,const int N,cmplxformat** d_phi_ptr)
{
  if(WTRACE) cout << "In UploadToDevice (cmplxformat)" << endl;	
  cudaError_t err;

  err = cudaMemcpy(*d_phi_ptr, phi, N*sizeof(cmplxformat), cudaMemcpyHostToDevice);

  if(err != cudaSuccess )
    {
      fprintf(stderr, "Failed to copy from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  if(WTRACE) cout << "Done with UploadToDevice (cmplxformat)" << endl;	
  return cudaSuccess;
}

int UploadToDevice(const realformat* phi,const int N,realformat** d_phi_ptr)
{
  if(WTRACE) cout << "In UploadToDevice (realformat)" << endl;	
  cudaError_t err;

  err = cudaMemcpy(*d_phi_ptr, phi, N*sizeof(realformat), cudaMemcpyHostToDevice);

  if(err != cudaSuccess )
    {
      fprintf(stderr, "Failed to copy from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  if(WTRACE) cout << "Done with UploadToDevice (realformat)" << endl;	
  return cudaSuccess;
}


int DownloadFromDevice(cmplxformat **d_phi_ptr,const int N,cmplxformat* phi)
{
  if(WTRACE) cout << "In DownloadFromDevice" << endl;	
  cudaError_t err;

  err = cudaMemcpy(phi,*d_phi_ptr,N*sizeof(cmplxformat),cudaMemcpyDeviceToHost);

  if(err != cudaSuccess )
    {
      fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  return cudaSuccess;
}

int DownloadFromDevice(realformat **d_phi_ptr,const int N,realformat* phi)
{
  if(WTRACE) cout << "In DownloadFromDevice" << endl;	
  cudaError_t err;

  err = cudaMemcpy(phi,*d_phi_ptr, N*sizeof(realformat), cudaMemcpyDeviceToHost);

  if(err != cudaSuccess )
    {
      fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  return cudaSuccess;
}

int FreeMemoryOnDevice(cmplxformat** d_phi)
{
  if(WTRACE) cout << "In FreeMemoryOnDevice" << endl;	
  cudaError_t err;

  err = cudaFree(*d_phi);
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to free memory on device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  if(WTRACE) cout << "Successfully freed Memory On Device (cmplxformat)" << endl;	
  return cudaSuccess;
}

int FreeMemoryOnDevice(realformat** d_phi)
{
  if(WTRACE) cout << "In FreeMemoryOnDevice" << endl;	
  cudaError_t err;

  err = cudaFree(*d_phi);
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to free memory on device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  if(WTRACE) cout << "Successfully freed Memory On Device (realformat)" << endl;	
  return cudaSuccess;
}



int UploadToDeviceAndExpand(const void* x,const int N,cmplxformat** d_xx)
{
  if(WTRACE) cout << "In UploadToDeviceAndExpand (cmplxformat)" << endl;	
  /*
  const cuda_cmplx* pptr=reinterpret_cast<const cuda_cmplx*>(p);
  cmplxformat cp[N];
  for(int i=0; i<N; i++){ cp[i].x=pptr[i].x; cp[i].y=pptr[i].y;}
  UploadToDevice(cp,N,d_p);
  */
  cuda_cmplx* d_x=NULL;
  AllocateSpaceOnDevice(&d_x,N);
  UploadToDevice(reinterpret_cast<const cuda_cmplx*>(x),N,&d_x);

  int b = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  cudap_kernel_make_precision_cmplx<<<b, THREADS_PER_BLOCK>>>(N,d_x,*d_xx);  

  FreeMemoryOnDevice(&d_x);

  if(WTRACE) cout << "Done with UploadToDeviceAndExpand (cmplxformat)" << endl;	
  return cudaSuccess;
}

int UploadToDeviceAndExpand(const double* x,const int N,realformat** d_xx)
{
  if(WTRACE) cout << "In UploadToDeviceAndExpand (realformat)" << endl;	
  /*
  realformat cp[N];
  for(int i=0; i<N; i++){ cp[i]=realformat(p[i]);}
  UploadToDevice(cp,N,d_p);
  */
  double* d_x=NULL;
  AllocateSpaceOnDevice(&d_x,N);
  UploadToDevice(x,N,&d_x);

  int b = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  cudap_kernel_make_precision_real<<<b, THREADS_PER_BLOCK>>>(N,d_x,*d_xx);  

  FreeMemoryOnDevice(&d_x);

  if(WTRACE) cout << "Done with UploadToDeviceAndExpand (realformat)" << endl;	
  return cudaSuccess;
}

int DownloadFromDeviceAndContract(cmplxformat** d_xx,int N,void* x)
{
  if(WTRACE) cout << "In DownloadFromDeviceAndContract (cmplxformat)" << endl;	
  /*
  cuda_cmplx* pptr=reinterpret_cast<cuda_cmplx*>(p);
  cmplxformat cp[N];
  DownloadFromDevice(d_p,N,cp);
  for(int i=0; i<N; i++){ pptr[i]=make_cuDoubleComplex(todouble(cp[i].x),todouble(cp[i].y));}
  */

  cuda_cmplx* d_x=NULL;
  AllocateSpaceOnDevice(&d_x,N);
  
  int b = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  cudap_kernel_make_cmplx<<<b, THREADS_PER_BLOCK>>>(N,*d_xx,d_x);  

  DownloadFromDevice(&d_x,N,reinterpret_cast<cuda_cmplx*>(x));
  FreeMemoryOnDevice(&d_x);

  if(WTRACE) cout << "Done with DownloadFromDeviceAndContract (cmplxformat)" << endl;	
  return cudaSuccess;
}

int DownloadFromDeviceAndContract(realformat** d_xx,int N,double* x)
{
  if(WTRACE) cout << "In DownloadFromDeviceAndContract (realformat)" << endl;	
  /*
  realformat cp[N];
  DownloadFromDevice(d_p,N,cp);

  for(int i=0; i<N; i++){ p[i]=todouble(cp[i]);}
  */
  double* d_x=NULL;
  AllocateSpaceOnDevice(&d_x,N);
  
  int b = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  cudap_kernel_make_double<<<b, THREADS_PER_BLOCK>>>(N,*d_xx,d_x);  

  DownloadFromDevice(&d_x,N,x);
  FreeMemoryOnDevice(&d_x);

  if(WTRACE) cout << "Done with DownloadFromDeviceAndContract (realformat)" << endl;	
  return cudaSuccess;
}

int InspectDevice(cmplxformat** d_p,int N)
{
  if(WTRACE) cout << "In InspectDevice (cmplxformat)" << endl;	
  cuda_cmplx cp[N];
  DownloadFromDeviceAndContract(d_p,N,cp);

  for(int i=0; i<N; i++){cout << cp[i] << endl;}
  if(WTRACE) cout << "Done with InspectDevice (cmplxformat)" << endl;	
  return cudaSuccess;
}

int InspectDevice(realformat** d_p,int N)
{
  if(WTRACE) cout << "In InspectDevice (realformat)" << endl;	
  double cp[N];
  DownloadFromDeviceAndContract(d_p,N,cp);

  for(int i=0; i<N; i++){cout << cp[i] << endl;}
  if(WTRACE) cout << "Done with InspectDevice (realformat)" << endl;	
  return cudaSuccess;
}


#endif // CUDA_PRECISION_CU
