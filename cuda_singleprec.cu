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


int cuda_mat_vec_multiply_cmplx(
const int M,const int Nc,const int* indices ,
const void* data , const cmplxformat* x, cmplxformat* y)
{
  if(WTRACE) cout << "Starting cuda_mat_vec_multiply_cmplx" << endl;
  int b = (M + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  
  const cuda_cmplx* datap=reinterpret_cast<const cuda_cmplx*>(data);
  const void* xp=reinterpret_cast<const void*>(x);
  void* yp=reinterpret_cast<void*>(y);

  cuda_kernel_mat_vec_multiply_cmplx<<<b, THREADS_PER_BLOCK>>>(M,Nc,indices,datap,xp,yp);

  if(WTRACE) cout << "Done with cuda_mat_vec_multiply_cmplx" << endl;
  return cudaSuccess;
}

int cuda_real_Zmaxpy(const int N,const realformat* q,cmplxformat* d_x,const cmplxformat* d_y) 
{
  if(WTRACE) cout << "Starting cuda_real_Zmaxpy" << endl;

  realformat* d_q=NULL;
  AllocateSpaceOnDevice(&d_q,1);
  UploadToDevice(q,1,&d_q);

  int b = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  cuda_kernel_real_Zmaxpy<<<b, THREADS_PER_BLOCK>>>(N,d_q,d_x,d_y);

  FreeMemoryOnDevice(&d_q);

  if(WTRACE) cout << "Done with cuda_real_Zmaxpy" << endl;
  return cudaSuccess;
}

int cuda_cmplx_Zmaxpy(const int N,const realformat* q,void* d_x,const void* d_y) 
{
  if(WTRACE) cout << "Starting cuda_cmplx_Zmaxpy" << endl;

  realformat* d_q=NULL;
  AllocateSpaceOnDevice(&d_q,1);
  UploadToDevice(q,1,&d_q);

  int b = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  cuda_kernel_cmplx_Zmaxpy<<<b, THREADS_PER_BLOCK>>>(N,d_q,d_x,d_y);

  FreeMemoryOnDevice(&d_q);

  if(WTRACE) cout << "Done with cuda_cmplx_Zmaxpy" << endl;
  return cudaSuccess;
}


int cuda_realdotproduct_real(const int N,const realformat* d_x,const realformat* d_y,realformat* ans) 
{

  if(WTRACE) cout << "Starting cuda_realdotproduct_cmplx" << endl;
  cudaError_t err;

  realformat c[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ c[i]=realformat(0.);}
  realformat* d_c=NULL;

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
  unsigned int old_cw;
  fpu_fix_start(&old_cw);
  for(int i = 1; i < BLOCKS; i++){c[0] += c[i];}
  fpu_fix_end(&old_cw); 

  *ans=c[0];

  if(WTRACE) cout << "Done with cuda_realdotproduct_cmplx" << endl;
  return cudaSuccess;
}



int cuda_realdotproduct_cmplx(const int N,const cmplxformat* d_x,const cmplxformat* d_y,realformat* ans) 
{

  if(WTRACE) cout << "Starting cuda_realdotproduct_cmplx" << endl;
  cudaError_t err;

  realformat c[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ c[i]=realformat(0.);}
  realformat* d_c=NULL;

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
  unsigned int old_cw;
  fpu_fix_start(&old_cw);
  for(int i = 1; i < BLOCKS; i++){c[0] += c[i];}
  fpu_fix_end(&old_cw); 


  *ans=c[0];

  if(WTRACE) cout << "Done with cuda_realdotproduct_cmplx" << endl;
  return cudaSuccess;
}


int cuda_cmplxdotproduct_cmplx(const int N,const cmplxformat* d_x,const cmplxformat* d_y,cmplxformat* ans) 
{
  if(WTRACE) cout << "Starting cuda_cmplxdotproduct_cmplx" << endl;
  cudaError_t err;
  
  cmplxformat c[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ c[i]=cmplxformat(0.,0.);}
  cmplxformat* d_c=NULL;
  
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
  unsigned int old_cw;
  fpu_fix_start(&old_cw);
  for(int i = 1; i < BLOCKS; i++){c[0].x += c[i].x; c[0].y += c[i].y;}
  fpu_fix_end(&old_cw); 

  *ans=c[0];

  if(WTRACE) cout << "Done with cuda_cmplxdotproduct_cmplx" << endl;
  return cudaSuccess;
}

// computes the dot product between two real vectors
int cuRealDotProductReal(const int N,const double* x,const double* y,double* res)
{
  realformat* d_x=NULL;
  AllocateSpaceOnDevice(&d_x,N);
  UploadToDeviceAndExpand(x,N,&d_x);

  realformat* d_y=NULL;
  AllocateSpaceOnDevice(&d_y,N);
  UploadToDeviceAndExpand(y,N,&d_y);

  realformat rp =realformat(0.);

  cuda_realdotproduct_real(N,d_x,d_y,&rp); 

  *res=todouble(rp);

  FreeMemoryOnDevice(&d_x);
  FreeMemoryOnDevice(&d_y);
  
  return 0;
}

// computes the real part of the dot product between two complex vectors
int cuRealDotProductCmplx(const int N,const void* x,const void* y,double* res)
{
  cmplxformat* d_x=NULL;
  AllocateSpaceOnDevice(&d_x,N);
  UploadToDeviceAndExpand(x,N,&d_x);  

  cmplxformat* d_y=NULL;
  AllocateSpaceOnDevice(&d_y,N);
  UploadToDeviceAndExpand(y,N,&d_y);  

  realformat rp =realformat(0.);

  cuda_realdotproduct_cmplx(N,d_x,d_y,&rp); 

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
int cuCmplxDotProductCmplx(int N,void* x,void* y,void* res)
{
  cmplxformat* d_x=NULL;
  AllocateSpaceOnDevice(&d_x,N);
  UploadToDeviceAndExpand(x,N,&d_x);  

  cmplxformat* d_y=NULL;
  AllocateSpaceOnDevice(&d_y,N);
  UploadToDeviceAndExpand(y,N,&d_y);  

  cmplxformat rp =cmplxformat(0.,0.);

  cuda_cmplxdotproduct_cmplx(N,d_x,d_y,&rp); 

  cuda_cmplx* res_ptr=reinterpret_cast<cuda_cmplx*>(res);

  (*res_ptr).x=todouble(rp.x);
  (*res_ptr).y=todouble(rp.y);

  FreeMemoryOnDevice(&d_x);
  FreeMemoryOnDevice(&d_y);
  
  return 0;
}


int cuda_norm_cmplx(const int N,const cmplxformat* d_x,void* ans) 
{

  if(WTRACE) cout << "Starting cuda_norm_cmplx" << endl;
  cudaError_t err;

  realformat c[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ c[i]=realformat(0.);}
  realformat* d_c=NULL;

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
  unsigned int old_cw;
  fpu_fix_start(&old_cw);
  for(int i = 1; i < BLOCKS; i++){c[0] += c[i];}
  fpu_fix_end(&old_cw); 

  realformat* ansp = reinterpret_cast<realformat*>(ans);
  *ansp=sqrt(c[0]);

  if(WTRACE) cout << "Done with cuda_norm_cmplx" << endl;
  return cudaSuccess;
}

// divide the complex vector x by  a real number: x=x/q
int cuda_Zdinvscal(int N,realformat* q,cmplxformat* d_x)
{
  if(WTRACE) cout << "Starting cuda_Zdinvscal " << endl;

  realformat s= inv(*q); // s= 1/q taking the inverse
  
  realformat* d_s=NULL;
  AllocateSpaceOnDevice(&d_s,1);
  UploadToDevice(&s,1,&d_s);

  int b = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  cuda_kernel_Zdscal<<<b, THREADS_PER_BLOCK>>>(N,d_s,d_x);

  FreeMemoryOnDevice(&d_s);

  if(WTRACE) cout << "Done with cuda_Zdinvscal" << endl;
  return cudaSuccess;
}


// multiply the complex vector x by  a real number: x=x*q
int cuda_Zdscal(int N,realformat* q,cmplxformat* d_x)
{
  if(WTRACE) cout << "Starting cuda_Zdscal " << endl;

  realformat* d_s=NULL;
  AllocateSpaceOnDevice(&d_s,1);
  UploadToDevice(q,1,&d_s);

  int b = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  cuda_kernel_Zdscal<<<b, THREADS_PER_BLOCK>>>(N,d_s,d_x);

  FreeMemoryOnDevice(&d_s);

  if(WTRACE) cout << "Done with cuda_Zdscal" << endl;
  return cudaSuccess;
}


int cuda_Normalize(const int N,cmplxformat* d_x,double* ans) 
{
  if(WTRACE) cout << "Starting cuda_Normalize" << endl;
  realformat norm=makereal(0.);
  cuda_norm_cmplx(N,d_x,&norm); 
  cuda_Zdinvscal(N,&norm,d_x);
  *ans=todouble(norm);
  return cudaSuccess;  
}


int AllocateSpaceOnDevice(cmplxformat** d_p,const int N)
{
  if(WTRACE) cout << "In AllocateSpaceOnDevice" << endl;	

  if (cudaMalloc(d_p, N*sizeof(cmplxformat)) != cudaSuccess)
    {
      fprintf(stderr, "!device memory allocatioNerror (AllocateSpaceOnDevice)\n");
      return EXIT_FAILURE;
    }
  return cudaSuccess;
}

int AllocateSpaceOnDevice(realformat** d_p,const int N)
{
  if(WTRACE) cout << "In AllocateSpaceOnDevice" << endl;	

  if (cudaMalloc(d_p, N*sizeof(realformat)) != cudaSuccess)
    {
      fprintf(stderr, "!!!! device memory allocatioNerror (AllocateSpaceOnDevice)\n");
      return EXIT_FAILURE;
    }
  return cudaSuccess;
}

int UploadToDevice(const cmplxformat* phi,const int N,cmplxformat** d_phi_ptr)
{
  if(WTRACE) cout << "In UploadToDevice" << endl;	
  cudaError_t err;

  err = cudaMemcpy(*d_phi_ptr, phi, N*sizeof(cmplxformat), cudaMemcpyHostToDevice);

  if(err != cudaSuccess )
    {
      fprintf(stderr, "Failed to copy from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  return cudaSuccess;
}

int UploadToDevice(const realformat* phi,const int N,realformat** d_phi_ptr)
{
  if(WTRACE) cout << "In UploadToDevice" << endl;	
  cudaError_t err;

  err = cudaMemcpy(*d_phi_ptr, phi, N*sizeof(realformat), cudaMemcpyHostToDevice);

  if(err != cudaSuccess )
    {
      fprintf(stderr, "Failed to copy from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
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
  return cudaSuccess;
}



int UploadToDeviceAndExpand(const void* p,const int N,cmplxformat** d_p)
{
  const cuda_cmplx* pptr=reinterpret_cast<const cuda_cmplx*>(p);
  cmplxformat cp[N];
  for(int i=0; i<N; i++){ cp[i]=cmplxformat(pptr[i].x,pptr[i].y);}
  UploadToDevice(cp,N,d_p);
  return cudaSuccess;
}

int UploadToDeviceAndExpand(const double* p,const int N,realformat** d_p)
{
  realformat cp[N];
  for(int i=0; i<N; i++){ cp[i]=realformat(p[i]);}
  UploadToDevice(cp,N,d_p);
  return cudaSuccess;
}

int DownloadFromDeviceAndContract(cmplxformat** d_p,int N,void* p)
{
  cuda_cmplx* pptr=reinterpret_cast<cuda_cmplx*>(p);
  cmplxformat cp[N];
  DownloadFromDevice(d_p,N,cp);

  for(int i=0; i<N; i++){ pptr[i]=make_cuDoubleComplex(todouble(cp[i].x),todouble(cp[i].y));}
  return cudaSuccess;
}

int DownloadFromDeviceAndContract(realformat** d_p,int N,double* p)
{
  realformat cp[N];
  DownloadFromDevice(d_p,N,cp);

  for(int i=0; i<N; i++){ p[i]=todouble(cp[i]);}
  return cudaSuccess;
}

int InspectDevice(cmplxformat** d_p,int N)
{
  cuda_cmplx cp[N];
  DownloadFromDeviceAndContract(d_p,N,cp);

  for(int i=0; i<N; i++){cout << cp[i] << endl;}
  return cudaSuccess;
}

int InspectDevice(realformat** d_p,int N)
{
  double cp[N];
  DownloadFromDeviceAndContract(d_p,N,cp);

  for(int i=0; i<N; i++){cout << cp[i] << endl;}
  return cudaSuccess;
}


/*
//*****************************************************************

int cuda_Zaxpy(int N,cuda_cmplx* q,cuda_cmplx* x,cuda_cmplx* y)
{
  cudaError_t err;
  if(WTRACE) cout << "In cuda_Zaxpy" << endl;

  cuda_cmplx* d_q=NULL;
  AllocateSpaceOnDevice(1,&d_q);

  UploadToDevice(q,1,&d_q);

  int  blockspergrid = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  if(WTRACE) cout << "Launcing cuda_Zaxpy with " << blockspergrid << " blocks" << endl;
#ifdef DOUBLEDOUBLE
  cudakernel_Zaxpy_dd<<<blockspergrid, THREADS_PER_BLOCK>>>(N,d_q,x,y); 
#elif QUADDOUBLE
  cudakernel_Zaxpy_qd<<<blockspergrid, THREADS_PER_BLOCK>>>(N,d_q,x,y); 
#else
  cudakernel_Zaxpy<<<blockspergrid, THREADS_PER_BLOCK>>>(N,d_q,x,y); 
#endif      

  err = cudaGetLastError();
  if(err != cudaSuccess)
    {
      fprintf(stderr, "Failed to execute cuda_Zaxpy (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  FreeMemoryOnDevice(&d_q);

  if(WTRACE) cout << "Done with cuda_Zaxpy" << endl;
  return cudaSuccess;
}


int cuZaxpy(int N,void* q,void* x,void* y)
{
  if(WTRACE) cout << "Starting Zaxpy " << endl;
  cuda_cmplx* d_x=NULL;
  AllocateSpaceOnDevice(N,&d_x);
  
  cuda_cmplx* x_ptr=reinterpret_cast<cuda_cmplx*>(x);
  UploadToDevice(x_ptr,N,&d_x);

  cuda_cmplx* d_y=NULL;
  AllocateSpaceOnDevice(N,&d_y);
  
  cuda_cmplx* y_ptr=reinterpret_cast<cuda_cmplx*>(y);
  UploadToDevice(y_ptr,N,&d_y);

  cuda_cmplx* q_ptr=reinterpret_cast<cuda_cmplx*>(q);

  cuda_Zaxpy(N,q_ptr,d_x,d_y);

  DownloadFromDevice(&d_x,N,x_ptr);

  FreeMemoryOnDevice(&d_x);
  FreeMemoryOnDevice(&d_y);
  
  if(WTRACE) cout << "Done with Zaxpy " << endl;

  return cudaSuccess;
}




int cuda_ComplexDotProduct(int N,cuda_cmplx* x,cuda_cmplx* y,cuda_cmplx* res)
{
  if(WTRACE) cout << "Starting cuda_ComplexDotProduct" << endl;
  cudaError_t err;

#ifdef DOUBLEDOUBLE
  dd_real cr[BLOCKS];
  dd_real ci[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ cr[i]=dd_real(0.); ci[i]=dd_real(0.);}
  void* d_cr=NULL;
  void* d_ci=NULL;
  int sizeofdouble=sizeof(dd_real);
#elif QUADDOUBLE
  qd_real cr[BLOCKS];
  qd_real ci[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ cr[i]=qd_real(0.); ci[i]=qd_real(0.);}
  void* d_cr=NULL;
  void* d_ci=NULL;
  int sizeofdouble=sizeof(qd_real);
#else
  double cr[BLOCKS];
  double ci[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ cr[i]=0.; ci[i]=0.;}
  void* d_cr=NULL;
  void* d_ci=NULL;
  int sizeofdouble=sizeof(double);
#endif

  err = cudaMalloc(&d_cr, sizeofdouble * BLOCKS);
  err = cudaMalloc(&d_ci, sizeofdouble * BLOCKS);
  if(err !=cudaSuccess)
    {
      fprintf(stderr, "!!!! device memory allocatioNerror (ComplexDotProduct)\n");
      return EXIT_FAILURE;
    }

  err=cudaMemcpy(d_cr,&cr[0], sizeofdouble*BLOCKS, cudaMemcpyHostToDevice);
  err=cudaMemcpy(d_ci,&ci[0], sizeofdouble*BLOCKS, cudaMemcpyHostToDevice);
  if(err != cudaSuccess) 
    {
      fprintf(stderr, "Failed to copy from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
#ifdef DOUBLEDOUBLE
  cudakernel_dotproduct_cmplx_dd<<<BLOCKS, THREADS_PER_BLOCK>>>(N, x, y, d_cr, d_ci); 
#elif QUADDOUBLE
  cudakernel_dotproduct_cmplx_qd<<<BLOCKS, THREADS_PER_BLOCK>>>(N, x, y, d_cr, d_ci); 
#else
  cudakernel_dotproduct_cmplx<<<BLOCKS, THREADS_PER_BLOCK>>>(N, x, y, d_cr, d_ci); 
#endif  

  err = cudaGetLastError();
  if(err != cudaSuccess)
    {
      fprintf(stderr, "Failed to execute (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  err=cudaMemcpy(&cr[0],d_cr,sizeofdouble*BLOCKS, cudaMemcpyDeviceToHost);
  err=cudaMemcpy(&ci[0],d_ci,sizeofdouble*BLOCKS, cudaMemcpyDeviceToHost);
  if(err != cudaSuccess) 
    {
      fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  err=cudaFree(d_cr);
  err=cudaFree(d_ci);
  if ( err != cudaSuccess)
    {
      fprintf(stderr, "Failed to free memory on device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  unsigned int old_cw;
  fpu_fix_start(&old_cw);

  for(int i = 1; i < BLOCKS; i++)
    { 
      cr[0] += cr[i];
      ci[0] += ci[i];
    }

#if defined DOUBLEDOUBLE || defined QUADDOUBLE  
  *res=make_cuDoubleComplex(to_double(cr[0]),to_double(ci[0]));
#else
  *res=make_cuDoubleComplex(cr[0],ci[0]);
#endif

  fpu_fix_end(&old_cw); 
  if(WTRACE) cout << "Done with cuda_ComplexDotProduct" << endl;
  return cudaSuccess;
} 



int cuComplexDotProduct(int N,void* x,void* y,void* res)
{
  cuda_cmplx* d_x=NULL;
  AllocateSpaceOnDevice(N,&d_x);
  
  cuda_cmplx* x_ptr=reinterpret_cast<cuda_cmplx*>(x);
  UploadToDevice(x_ptr,N,&d_x);

  cuda_cmplx* d_y=NULL;
  AllocateSpaceOnDevice(N,&d_y);
  
  cuda_cmplx* y_ptr=reinterpret_cast<cuda_cmplx*>(y);
  UploadToDevice(y_ptr,N,&d_y);

  cuda_cmplx* res_ptr=reinterpret_cast<cuda_cmplx*>(res);
  
  cuda_ComplexDotProduct(N,d_x,d_y,res_ptr);

  FreeMemoryOnDevice(&d_x);
  FreeMemoryOnDevice(&d_y);
  
  return 0;
}


//***********************************************************************
// cuda norm routines

int cuda_Dznrm2(int N,cuda_cmplx* d_x,double* res)
{
  if(WTRACE) cout << "Starting cuda_Dznrm2" << endl;
  cudaError_t err;

#ifdef DOUBLEDOUBLE
  dd_real c[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ c[i]=dd_real(0.);}
  void* d_c=NULL;
  int sizeofdouble=sizeof(dd_real);
#elif QUADDOUBLE
  qd_real c[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ c[i]=qd_real(0.);}
  void* d_c=NULL;
  int sizeofdouble=sizeof(qd_real);
#else
  double c[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ c[i]=0.;}
  void* d_c=NULL;
  int sizeofdouble=sizeof(double);
#endif

  err = cudaMalloc(&d_c, sizeofdouble * BLOCKS);
  if(err !=cudaSuccess)
    {
      fprintf(stderr, "!!!! device memory allocatioNerror (ComplexDotProduct)\n");
      return EXIT_FAILURE;
    }

  err=cudaMemcpy(d_c,&c[0], sizeofdouble*BLOCKS, cudaMemcpyHostToDevice);
  if(err != cudaSuccess) 
    {
      fprintf(stderr, "Failed to copy from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
#ifdef DOUBLEDOUBLE
  cudakernel_norm_cmplx_dd<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, d_c); 
#elif QUADDOUBLE
  cudakernel_norm_cmplx_qd<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, d_c); 
#else
  cudakernel_norm_cmplx<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, d_c); 
#endif  

  err = cudaGetLastError();
  if(err != cudaSuccess)
    {
      fprintf(stderr, "Failed to execute (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  err=cudaMemcpy(&c[0],d_c,sizeofdouble*BLOCKS, cudaMemcpyDeviceToHost);
  if(err != cudaSuccess) 
    {
      fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  err=cudaFree(d_c);
  if ( err != cudaSuccess)
    {
      fprintf(stderr, "Failed to free memory on device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  unsigned int old_cw;
  fpu_fix_start(&old_cw);

  for(int i = 1; i < BLOCKS; i++)
    { 
      c[0] += c[i];
    }

#if defined DOUBLEDOUBLE || defined QUADDOUBLE 
  *res=to_double(sqrt(c[0])); 
#else
  *res=sqrt(c[0]);
#endif

  fpu_fix_end(&old_cw);
  if(WTRACE) cout << "Done with cuda_Dznrm2" << endl;
  return cudaSuccess;
} 



int cuDznrm2(int N,void* x,double* res)
{
  cuda_cmplx* d_x=NULL;
  AllocateSpaceOnDevice(N,&d_x);
  
  cuda_cmplx* x_ptr=reinterpret_cast<cuda_cmplx*>(x);
  UploadToDevice(x_ptr,N,&d_x);

  cuda_Dznrm2(N,d_x,res);

  FreeMemoryOnDevice(&d_x);
  
  return 0;
}

//*************************************************************************************
// Normalize a complex vector

int cuda_Normalize(int N,cuda_cmplx* d_x,double* res)
{
  if(WTRACE) cout << "Starting cuda_Dznrm2" << endl;
  cudaError_t err;

#ifdef DOUBLEDOUBLE
  dd_real c[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ c[i]=dd_real(0.);}
  void* d_c=NULL;
  dd_real norm=dd_real(0.);
  dd_real invnorm=dd_real(0.);
  int sizeofdouble=sizeof(dd_real);
#elif QUADDOUBLE
  qd_real c[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ c[i]=qd_real(0.);}
  void* d_c=NULL;
  qd_real norm=qd_real(0.);
  qd_real invnorm=qd_real(0.);
  int sizeofdouble=sizeof(qd_real);
#else
  double c[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ c[i]=0.;}
  void* d_c=NULL;
  double norm=0.;
  double invnorm=0.;
  int sizeofdouble=sizeof(double);
#endif

  err = cudaMalloc(&d_c, sizeofdouble * BLOCKS);
  if(err !=cudaSuccess)
    {
      fprintf(stderr, "!!!! device memory allocatioNerror (ComplexDotProduct)\n");
      return EXIT_FAILURE;
    }

  err=cudaMemcpy(d_c,&c[0], sizeofdouble*BLOCKS, cudaMemcpyHostToDevice);
  if(err != cudaSuccess) 
    {
      fprintf(stderr, "Failed to copy from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
#ifdef DOUBLEDOUBLE
  cudakernel_norm_cmplx_dd<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, d_c); 
#elif QUADDOUBLE
  cudakernel_norm_cmplx_qd<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, d_c); 
#else
  cudakernel_norm_cmplx<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, d_c); 
#endif  

  err = cudaGetLastError();
  if(err != cudaSuccess)
    {
      fprintf(stderr, "Failed to execute (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  err=cudaMemcpy(&c[0],d_c,sizeofdouble*BLOCKS, cudaMemcpyDeviceToHost);
  if(err != cudaSuccess) 
    {
      fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  unsigned int old_cw;
  fpu_fix_start(&old_cw);

  for(int i = 1; i < BLOCKS; i++)
    { 
      c[0] += c[i];
    }

  norm=sqrt(c[0]);

#if defined DOUBLEDOUBLE || defined QUADDOUBLE
  *res=to_double(norm);
#else
  *res=norm;
#endif
  
  // use c[0] to store the inverse norm and then multiply by c[0]
  c[0]=inv(norm);

  fpu_fix_end(&old_cw);

  
  err=cudaMemcpy(d_c,&c[0], sizeofdouble*1, cudaMemcpyHostToDevice);
  if(err != cudaSuccess) 
    {
      fprintf(stderr, "Failed to copy from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  int  blockspergrid = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
#ifdef DOUBLEDOUBLE
  cudakernel_Zddscal_dd<<<blockspergrid, THREADS_PER_BLOCK>>>(N,d_c,d_x); 
#elif QUADDOUBLE
  cudakernel_Zqdscal_qd<<<blockspergrid, THREADS_PER_BLOCK>>>(N,d_c,d_x); 
#else
  cudakernel_Zdscal<<<blockspergrid, THREADS_PER_BLOCK>>>(N,d_c,d_x); 
#endif      

  err=cudaFree(d_c);
  if ( err != cudaSuccess)
    {
      fprintf(stderr, "Failed to free memory on device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  
  if(WTRACE) cout << "Done with cuda_norm_driver" << endl;
  return cudaSuccess;
} 


int cuNormalize(int N,void* x,double* res)
{
  cuda_cmplx* d_x=NULL;
  AllocateSpaceOnDevice(N,&d_x);
  
  cuda_cmplx* x_ptr=reinterpret_cast<cuda_cmplx*>(x);
  UploadToDevice(x_ptr,N,&d_x);

  cuda_Normalize(N,d_x,res);

  DownloadFromDevice(&d_x,N,x_ptr);

  FreeMemoryOnDevice(&d_x);
  
  return 0;
}


//***********************  the double dot product: *************************************


int cuda_DoubleDotProduct(int N,double* d_x,double* d_y,double* res)
{
  cudaError_t err;

#ifdef DOUBLEDOUBLE
  dd_real c[BLOCKS];
  for(int i=0; i<BLOCKS; i++) c[i]=dd_real(0.);
  void* d_c=NULL;
  int sizeofdouble=sizeof(dd_real);
#elif QUADDOUBLE
  qd_real c[BLOCKS];
  for(int i=0; i<BLOCKS; i++) c[i]=qd_real(0.);
  void* d_c=NULL;
  int sizeofdouble=sizeof(qd_real);
#else
  double c[BLOCKS];
  for(int i=0; i<BLOCKS; i++) c[i]=0.;
  void* d_c=NULL;
  int sizeofdouble=sizeof(double);
#endif

  err = cudaMalloc(&d_c, sizeofdouble * BLOCKS);
  if(err !=cudaSuccess)
    {
      fprintf(stderr, "!!!! device memory allocatioNerror (DoubleDotProduct)\n");
      return EXIT_FAILURE;
    }

  err=cudaMemcpy(d_c,&c[0], sizeofdouble*BLOCKS, cudaMemcpyHostToDevice);
  if(err != cudaSuccess) 
    {
      fprintf(stderr, "Failed to copy from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

#ifdef DOUBLEDOUBLE
  cudakernel_dotproduct_double_dd<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, d_y, d_c); 
#elif QUADDOUBLE
  cudakernel_dotproduct_double_qd<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, d_y, d_c); 
#else
  cudakernel_dotproduct_double<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, d_y, d_c); 
#endif

  err = cudaGetLastError();
  if(err != cudaSuccess)
    {
      fprintf(stderr, "Failed to execute cuda_dotproduct_double_dd (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  err=cudaMemcpy(&c[0],d_c,sizeofdouble*BLOCKS, cudaMemcpyDeviceToHost);
  if(err != cudaSuccess) 
    {
      fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  err=cudaFree(d_c);
  if ( err != cudaSuccess)
    {
      fprintf(stderr, "Failed to free memory d_c on device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  unsigned int old_cw;
  fpu_fix_start(&old_cw);

  for(int i = 1; i < BLOCKS; i++){ c[0] += c[i];}

#if defined DOUBLEDOUBLE || defined QUADDOUBLE    
  *res=to_double(c[0]);
#else
  *res=c[0];
#endif

  fpu_fix_end(&old_cw);
  return cudaSuccess;
} 
  

int cuDoubleDotProduct(int N,double* x,double* y,double* res)
{
  double* d_x=NULL;
  AllocateSpaceOnDevice(N,&d_x);
  
  UploadToDevice(x,N,&d_x);

  double* d_y=NULL;
  AllocateSpaceOnDevice(N,&d_y);
  
  UploadToDevice(y,N,&d_y);

  cuda_DoubleDotProduct(N,d_x,d_y,res);

  FreeMemoryOnDevice(&d_x);
  FreeMemoryOnDevice(&d_y);
  
  return cudaSuccess;
}
*/


#endif // CUDA_PRECISION_DRIVERS_CU
