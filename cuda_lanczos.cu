#ifndef CUDA_LANCZOS_CU
#define CUDA_LANCZOS_CU

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

#include <qd/dd_real.h>
#include <qd/qd_real.h>
#include <qd/fpu.h>


struct dd_cmplx{dd_real x; dd_real y;};
struct qd_cmplx{qd_real x; qd_real y;};

#ifdef DOUBLEDOUBLE
#define realformat  dd_real
#define cmplxformat dd_cmplx
#define todouble to_double
#elif defined QUADDOUBLE
#define realformat  qd_real
#define cmplxformat qd_cmplx
#define todouble    to_double;
#else
#define realformat  double
#define cmplxformat cuda_cmplx
#define todouble    double
#endif

const int sizeofreal=sizeof(realformat);
const int sizeofcmplx=sizeof(cmplxformat);


class cudaLanczos
{
public:
  cudaLanczos():
  bool Doiteration(int m);

private:
  int Cols;
  int Rows;
  int Nc;

  cuda_cmplx* d_m_val; // device pointers refering to the matrix     
  int*        d_m_col; 

  int M;

  // ptrs to host variables, supplied in the initialization
  double* lanczos_a;
  double* lanczos_b;
  cuda_cmplx* phi;

  // variables having extended precision

  realformat* prec_lanczos_a;   
  realformat* prec_lanczos_b;

  // device variables:
  void*       d_phi;
};

cudaLanczos::cudaLanczos()
{
  prec_lanczos_a = new realformat[M];
  prec_lanczos_b = new realformat[M]; // only the M-1 components will finally be used

  AllocateSpaceOnDevice((M+1)*Cols*sizeofcmplx,&d_phi);
  InitializeSpaceOnDevice((M+1)*Cols*sizeofcmplx,&d_phi);
}

cudaLanczos::~cudaLanczos()
{
  delete[] prec_lanczos_a;
  delete[] prec_lanczos_b;

  FreeMemoryOnDevice(&d_phi);
}

cudaLanczos::InitializePhi0()
{
  cmplxformat* tempphi[Cols];
  
  for(int i=0; i<Cols; i++){tempphi[i]=cmplxformat(phi[i].x,phi[i].y);}
  
  void* tempphi_ptr=reinterpret_cast<void*>(tempphi);
  UploadToDevice(tempphi_ptr,Cols*sizeofcmplx,&d_phi);   
}

// take the lanczos precision data and transdfer it to doubles
cudaLanczos::TransferLanczos(const int m)
{

  for(int i=0; i<m; i++){lanczos_a[i]=todouble(prec_lanczos_a[i]);}
  for(int i=0; i<m; i++){lanczos_b[i]=todouble(prec_lanczos_b[i]);}
}

cudaLanczos::DownloadPhis(const int m)
{
  cmplxformat* temp[m*Cols];
  void* temp_ptr=reinterpret_cast<void*>(temp);

  DownloadFromDevice(&d_phi,m*Cols*sizeofcmplx,temp_ptr);   
  for(int i=0; i<m; i++){phi[i*Cols]=cuda_cmplx(todouble(temp[i].x),todouble(temp[i].y));}
}


void cudaLanczos::Reorthogonalize(const int m)
{
  if(WTRACE) cout << "Starting Reorthogonalize" << endl;
  //  for(int j=0; j<MGSREPEATSTEPS; j++) // 2 for Modified Gram-Schmidt with reorthogo..
    for(int i=0; i<=m; i++)
      {
	cmplxformat* q;
	void* qp=reinterpret_cast<void*>(q)
	void* d_phii=&d_phi[i*n*sizeofcmplx];
	cuda_cmplxdotproduct_cmplx(n,d_phii,d_z,qp); // q=<phi_i|z> 
	cuda_cmplx_Zmaxpy(n,qp,d_z,d_phii); // |z>=|z>-q*|phi_i>; 
      }
  if(TRACE) cout << "Done Reorthogonalize" << endl; 
}


 
bool cudaLanczos::Doiteration(const int m)
{
  void* d_phim=&d_phi[m*Cols*sizeofcmplx]; //already normalized     
  void* d_z   =&d_phi[(m+1)*Cols*sizeofcmplx];                 
  
  int blockspergrid=(Rows+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;

  cuda_mat_vec_multiply_cmplx(Rows,Nc,d_m_col ,d_m_val, d_phim, d_z); // |z> = H |phi_m>  
  err = cudaGetLastError();                                                  
  if(TRACE) cout << "end of cuda_kernel_mat_vec_multiply_cmplx, last error" << err << endl; 

  /*  
  if(TRACE)
    {
      cout << "H * phi[" << m << "]" << endl;
      InspectDevice(&d_z,Cols);
    }
  */
  
  cuda_cmplx a=cmplx_0;
  realformat* am=&prec_lanczos_a[m]; // the pointer the the device lanczos value am
  void* amp = reinterpret_cast<void*>(am);
  if(TRACE) cout << "Launching ComplexDotProduct" << endl; 
  cuda_realdotproduct_cmplx(Cols,d_phim,d_z,amp); // a_m=<phi_m|z>, should be real 
  
  if(TRACE) cout << "lanczos_a[" << m << "]=" << todouble(prec_lanczos_a[m]) << endl;
  
  cuda_real_Zmaxpy(Cols,amp,d_phim,d_z); // |z>=|z>-a_m*|phi_m>  
  
  if(m>0)
    {
      void* bmm1=reinterpret_cast<void*>(&prec_lanczos_b[(m-1)];)
      void* d_phimm1=&d_phi[(m-1)*Cols*sizeofcmplx];
      
      cuda_real_Zmaxpy(Cols,bmm1,d_phimm1,d_z); // |z>=|z>-b_(m-1)*|phi_(m-1)>
    }
  
#ifdef REORTHOGONALIZE
  Reorthogonalize(Cols,d_z,m);
#else
  
#endif
  if(TRACE) cout << "After Reorthogonalization" << endl;
  if(TRACE) CheckOrthogonality(Cols,d_z,m);


  void* bm=reinterpret_cast<void*>(&prec_lanczos_b[m]);
  cuda_norm_cmplx(Cols,d_z,bm);

  if(m == M-1) // end of lanczos procedure               
    {
      if(TRACE) cout << "Truncating the Lanczos procedure, reached max matrix size, lanczos_b[M-1]= " << todouble(prec_lanczos_b[M-1]) << " --> 0" << endl;
      return false;
    }
  
  if(TRACE) cout << "lanczos_a[" << m << "]=" << todouble(prec_lanczos_a[m]) << endl;
  if(TRACE) cout << "lanczos_b[" << m << "]=" << todouble(prec_lanczos_b[m]) << endl;
  if(abs(prec_lanczos_b[m])<=LANCZOSEXITLIMIT){return false;} // stop the Lanczos procedure;            
  
  cuda_Zdinvscal(Cols,bm,d_z); // |z> -> |z>/beta

  if(TRACE){
    cout << "Lanczos vector phi[" << m+1 << "]: " << endl;
    InspectDevice(&d_z,Cols);
  }
}










/*


//***********************************************************************
// cuda norm routines

int cuda_Dznrm2(int N,cuda_cmplx* d_x,double* res)
{
  if(WTRACE) cout << "Starting cuda_Dznrm2" << endl;
  cudaError_t err;

#ifdef DOUBLEDOUBLE
  dd_real c[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ c[i]=dd_real(0.);}
  void* dev_c=NULL;
  int sizeofdouble=sizeof(dd_real);
#elif QUADDOUBLE
  qd_real c[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ c[i]=qd_real(0.);}
  void* dev_c=NULL;
  int sizeofdouble=sizeof(qd_real);
#else
  double c[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ c[i]=0.;}
  void* dev_c=NULL;
  int sizeofdouble=sizeof(double);
#endif

  err = cudaMalloc(&dev_c, sizeofdouble * BLOCKS);
  if(err !=cudaSuccess)
    {
      fprintf(stderr, "!!!! device memory allocatioNerror (ComplexDotProduct)\n");
      return EXIT_FAILURE;
    }

  err=cudaMemcpy(dev_c,&c[0], sizeofdouble*BLOCKS, cudaMemcpyHostToDevice);
  if(err != cudaSuccess) 
    {
      fprintf(stderr, "Failed to copy from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
#ifdef DOUBLEDOUBLE
  cudakernel_norm_cmplx_dd<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, dev_c); 
#elif QUADDOUBLE
  cudakernel_norm_cmplx_qd<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, dev_c); 
#else
  cudakernel_norm_cmplx<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, dev_c); 
#endif  

  err = cudaGetLastError();
  if(err != cudaSuccess)
    {
      fprintf(stderr, "Failed to execute (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  err=cudaMemcpy(&c[0],dev_c,sizeofdouble*BLOCKS, cudaMemcpyDeviceToHost);
  if(err != cudaSuccess) 
    {
      fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  err=cudaFree(dev_c);
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
  void* dev_c=NULL;
  dd_real norm=dd_real(0.);
  dd_real invnorm=dd_real(0.);
  int sizeofdouble=sizeof(dd_real);
#elif QUADDOUBLE
  qd_real c[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ c[i]=qd_real(0.);}
  void* dev_c=NULL;
  qd_real norm=qd_real(0.);
  qd_real invnorm=qd_real(0.);
  int sizeofdouble=sizeof(qd_real);
#else
  double c[BLOCKS];
  for(int i=0; i<BLOCKS; i++){ c[i]=0.;}
  void* dev_c=NULL;
  double norm=0.;
  double invnorm=0.;
  int sizeofdouble=sizeof(double);
#endif

  err = cudaMalloc(&dev_c, sizeofdouble * BLOCKS);
  if(err !=cudaSuccess)
    {
      fprintf(stderr, "!!!! device memory allocatioNerror (ComplexDotProduct)\n");
      return EXIT_FAILURE;
    }

  err=cudaMemcpy(dev_c,&c[0], sizeofdouble*BLOCKS, cudaMemcpyHostToDevice);
  if(err != cudaSuccess) 
    {
      fprintf(stderr, "Failed to copy from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
#ifdef DOUBLEDOUBLE
  cudakernel_norm_cmplx_dd<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, dev_c); 
#elif QUADDOUBLE
  cudakernel_norm_cmplx_qd<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, dev_c); 
#else
  cudakernel_norm_cmplx<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, dev_c); 
#endif  

  err = cudaGetLastError();
  if(err != cudaSuccess)
    {
      fprintf(stderr, "Failed to execute (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  err=cudaMemcpy(&c[0],dev_c,sizeofdouble*BLOCKS, cudaMemcpyDeviceToHost);
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

  
  err=cudaMemcpy(dev_c,&c[0], sizeofdouble*1, cudaMemcpyHostToDevice);
  if(err != cudaSuccess) 
    {
      fprintf(stderr, "Failed to copy from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  int  blockspergrid = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
#ifdef DOUBLEDOUBLE
  cudakernel_Zddscal_dd<<<blockspergrid, THREADS_PER_BLOCK>>>(N,dev_c,d_x); 
#elif QUADDOUBLE
  cudakernel_Zqdscal_qd<<<blockspergrid, THREADS_PER_BLOCK>>>(N,dev_c,d_x); 
#else
  cudakernel_Zdscal<<<blockspergrid, THREADS_PER_BLOCK>>>(N,dev_c,d_x); 
#endif      

  err=cudaFree(dev_c);
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
  void* dev_c=NULL;
  int sizeofdouble=sizeof(dd_real);
#elif QUADDOUBLE
  qd_real c[BLOCKS];
  for(int i=0; i<BLOCKS; i++) c[i]=qd_real(0.);
  void* dev_c=NULL;
  int sizeofdouble=sizeof(qd_real);
#else
  double c[BLOCKS];
  for(int i=0; i<BLOCKS; i++) c[i]=0.;
  void* dev_c=NULL;
  int sizeofdouble=sizeof(double);
#endif

  err = cudaMalloc(&dev_c, sizeofdouble * BLOCKS);
  if(err !=cudaSuccess)
    {
      fprintf(stderr, "!!!! device memory allocatioNerror (DoubleDotProduct)\n");
      return EXIT_FAILURE;
    }

  err=cudaMemcpy(dev_c,&c[0], sizeofdouble*BLOCKS, cudaMemcpyHostToDevice);
  if(err != cudaSuccess) 
    {
      fprintf(stderr, "Failed to copy from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

#ifdef DOUBLEDOUBLE
  cudakernel_dotproduct_double_dd<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, d_y, dev_c); 
#elif QUADDOUBLE
  cudakernel_dotproduct_double_qd<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, d_y, dev_c); 
#else
  cudakernel_dotproduct_double<<<BLOCKS, THREADS_PER_BLOCK>>>(N, d_x, d_y, dev_c); 
#endif

  err = cudaGetLastError();
  if(err != cudaSuccess)
    {
      fprintf(stderr, "Failed to execute cuda_dotproduct_double_dd (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  err=cudaMemcpy(&c[0],dev_c,sizeofdouble*BLOCKS, cudaMemcpyDeviceToHost);
  if(err != cudaSuccess) 
    {
      fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  err=cudaFree(dev_c);
  if ( err != cudaSuccess)
    {
      fprintf(stderr, "Failed to free memory dev_c on device (error code %s)!\n", cudaGetErrorString(err));
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

#endif // CUDA_LANCZOS_CU
