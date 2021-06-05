#ifndef CUDA_GLOBAL_CU
#define CUDA_GLOBAL_CU

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

ostream& operator<<(ostream& os,cuda_cmplx a){os << "(" << cuCreal(a) << "," << cuCimag(a) << ")"; return os;}


int AllocateSpaceOnDevice(cuda_cmplx** d_phi_ptr,const int N)
{
  if(WTRACE) cout << "In AllocateSpaceOnDevice (cuda_cmplx)" << endl;	
  if(TRACE) cout << "Allocating " << N*sizeof(cuda_cmplx) << " bytes on device" << endl;

  if (cudaMalloc(d_phi_ptr, N*sizeof(cuda_cmplx)) != cudaSuccess)
    {
      fprintf(stderr, "!!!! device memory allocatioNerror (AllocateSpaceOnDevice)\n");
      return EXIT_FAILURE;
    }
  if(WTRACE) cout << "Done with AllocateSpaceOnDevice (cuda_cmplx)" << endl;	
  return cudaSuccess;
}

int AllocateSpaceOnDevice(double** d_phi_ptr,const int N)
{
  if(WTRACE) cout << "In AllocateSpaceOnDevice (double)" << endl;	
  if(TRACE) cout << "Allocating " << N*sizeof(double) << " bytes on device" << endl;

  if (cudaMalloc(d_phi_ptr, N*sizeof(double)) != cudaSuccess)
    {
      fprintf(stderr, "!!!! device memory allocatioNerror (AllocateSpaceOnDevice)\n");
      return EXIT_FAILURE;
    }
  if(WTRACE) cout << "Done with AllocateSpaceOnDevice (double)" << endl;	
  return cudaSuccess;
}

/*
int AllocateSpaceOnDevice(void** d_phi_ptr,const int N)
{
  if(WTRACE) cout << "In AllocateSpaceOnDevice (void)" << endl;	

  if (cudaMalloc(d_phi_ptr, N) != cudaSuccess)
    {
      fprintf(stderr, "!!!! device memory allocatioNerror (AllocateSpaceOnDevice)\n");
      return EXIT_FAILURE;
    }

  return cudaSuccess;
}
*/

int InitializeSpaceOnDevice(const int N,void** d_phi_ptr)
{
  if(WTRACE) cout << "In InitializeSpaceOnDevice (void)" << endl;	

  if (cudaMemset(d_phi_ptr,0,N) != cudaSuccess)
    {
      fprintf(stderr, "!!!! device memory Memset error (InitializeSpaceOnDevice)\n");
      return EXIT_FAILURE;
    }
  return cudaSuccess;
}


int UploadToDevice(void* phi,const int N,cuda_cmplx** d_phi_ptr)
{
  if(WTRACE) cout << "In UploadToDevice (cuda_cmplx)" << endl;	
  cudaError_t err;

  cuda_cmplx* phi_ptr=reinterpret_cast<cuda_cmplx*>(phi);

  // upload the matrix to the GPU
  err = cudaMemcpy(*d_phi_ptr, phi_ptr, N*sizeof(cuda_cmplx), cudaMemcpyHostToDevice);

  if(err != cudaSuccess )
    {
      fprintf(stderr, "Failed to copy from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  return cudaSuccess;
}


int UploadToDevice(const cuda_cmplx* phi,const int N,cuda_cmplx** d_phi_ptr)
{
  if(WTRACE) cout << "In UploadToDevice" << endl;	
  cudaError_t err;

  err = cudaMemcpy(*d_phi_ptr, phi, N*sizeof(cuda_cmplx), cudaMemcpyHostToDevice);

  if(err != cudaSuccess )
    {
      fprintf(stderr, "Failed to copy from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  return cudaSuccess;
}

int UploadToDevice(const double* phi,const int N,double** d_phi_ptr)
{
  if(WTRACE) cout << "In UploadToDevice (double)" << endl;	
  cudaError_t err;

  err = cudaMemcpy(*d_phi_ptr, phi, N*sizeof(double), cudaMemcpyHostToDevice);

  if(err != cudaSuccess )
    {
      fprintf(stderr, "Failed to copy from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  return cudaSuccess;
}

int UploadToDevice(const void* phi,const int N,void** d_phi_ptr)
{
  if(WTRACE) cout << "In UploadToDevice (void)" << endl;	
  cudaError_t err;

  err = cudaMemcpy(*d_phi_ptr, phi, N, cudaMemcpyHostToDevice);

  if(err != cudaSuccess )
    {
      fprintf(stderr, "Failed to copy from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  return cudaSuccess;
}


//int DownloadFromDevice(cuda_cmplx** d_phi_ptr,const int N,MyField* phi)
int DownloadFromDevice(cuda_cmplx **d_phi_ptr,const int N,void *phi)
{
  if(WTRACE) cout << "In DownloadFromDevice" << endl;	
  cudaError_t err;

  cuda_cmplx *phi_ptr=reinterpret_cast<cuda_cmplx*>(phi);

  err = cudaMemcpy(phi_ptr,*d_phi_ptr, N*sizeof(cuda_cmplx), cudaMemcpyDeviceToHost);

  if(err != cudaSuccess )
    {
      fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  return cudaSuccess;
}

int DownloadFromDevice(double** d_phi_ptr,const int N,void *phi)
{
  if(WTRACE) cout << "In DownloadFromDevice (double)" << endl;	
  cudaError_t err;

  double* phi_ptr=reinterpret_cast<double*>(phi);

  err = cudaMemcpy(phi_ptr,*d_phi_ptr, N*sizeof(double), cudaMemcpyDeviceToHost);

  if(err != cudaSuccess )
    {
      fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  return cudaSuccess;
}

int DownloadFromDevice(void **d_phi_ptr,const int N,void *phi)
{
  if(WTRACE) cout << "In DownloadFromDevice (void)" << endl;	
  cudaError_t err;

  err = cudaMemcpy(phi,*d_phi_ptr, N, cudaMemcpyDeviceToHost);

  if(err != cudaSuccess )
    {
      fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  return cudaSuccess;
}

/*

int InspectDevice(cuda_cmplx** d_phi_ptr,const int N)
{
  if(WTRACE) cout << "In InspectDevice" << endl;	
  cudaError_t err;

  cuda_cmplx temp[N];
  cuda_cmplx* temp_ptr=&temp[0];

  // download 
  err = cudaMemcpy(temp_ptr,*d_phi_ptr, N*sizeof(cuda_cmplx), cudaMemcpyDeviceToHost);

  if(err != cudaSuccess )
    {
      fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  for(int i=0; i<N; i++)
    {
      cout << setprecision(DEBUGPRECISION) << temp[i] << endl;
    }
  cout << endl;

  return cudaSuccess;
}

int InspectDevice(double** d_phi_ptr,const int N)
{
  if(WTRACE) cout << "In InspectDevice" << endl;	
  cudaError_t err;

  double temp[N];
  double* temp_ptr=&temp[0];

  // download 
  err = cudaMemcpy(temp_ptr,*d_phi_ptr, N*sizeof(double), cudaMemcpyDeviceToHost);

  if(err != cudaSuccess )
    {
      fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  for(int i=0; i<N; i++){cout << setprecision(DEBUGPRECISION) << temp[i] << endl;}
  cout << endl;

  return cudaSuccess;
}
*/



int FreeMemoryOnDevice(cuda_cmplx** d_phi)
{
  if(WTRACE) cout << "In FreeMemoryOnDevice (cuda_cmplx)" << endl;	
  cudaError_t err;

  err = cudaFree(*d_phi);
    
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to free memory on device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  if(WTRACE) cout << "Successfully freed memory (cuda_cmplx)" << endl;
  return cudaSuccess;
}

int FreeMemoryOnDevice(double** d_phi)
{
  if(WTRACE) cout << "In FreeMemoryOnDevice (double)" << endl;	
  cudaError_t err;

  err = cudaFree(*d_phi);
    
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to free memory on device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  if(WTRACE) cout << "Successfully freed memory (double)" << endl;
  return cudaSuccess;
}

int FreeMemoryOnDevice(int** d_phi)
{
  if(WTRACE) cout << "In FreeMemoryOnDevice (int)" << endl;	
  cudaError_t err;

  err = cudaFree(*d_phi);
    
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to free memory on device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  if(WTRACE) cout << "Successfully freed memory (int)" << endl;
  return cudaSuccess;
}

/*
int FreeMemoryOnDevice(void** d_phi)
{
  if(WTRACE) cout << "In FreeMemoryOnDevice (double)" << endl;	
  cudaError_t err;

  err = cudaFree(*d_phi);
    
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to free memory on device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  return cudaSuccess;
}
*/


int UploadMatrixToDevice(MyMatrix& H,cuda_cmplx** d_m_val,int** d_m_col)
{
  if(WTRACE) cout << "In UploadMatrixToDevice" << endl;	
  cudaError_t err;
  cuda_cmplx* m_val=reinterpret_cast<cuda_cmplx*>(H.GetValStartPtr());
  int*     m_col=H.GetColStartPtr();

  int Ntot=H.Nrows()*H.NStoredCols();

  if (cudaMalloc(d_m_val, Ntot*sizeof(m_val[0])) != cudaSuccess)
    {
      fprintf(stderr, "!!!! device memory allocatioNerror (allocate d_m_val)\n");
      return EXIT_FAILURE;
    }

  if (cudaMalloc(d_m_col, Ntot*sizeof(m_col[0])) != cudaSuccess)
    {
      fprintf(stderr, "!!!! device memory allocatioNerror (allocate d_m_col)\n");
      return EXIT_FAILURE;
    }

  // upload the matrix to the GPU
  err = cudaMemcpy(*d_m_val, m_val, Ntot*sizeof(m_val[0]), cudaMemcpyHostToDevice);

  if(err != cudaSuccess )
    {
      fprintf(stderr, "Failed to copy vector m_val from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  err = cudaMemcpy(*d_m_col, m_col, Ntot*sizeof(m_col[0]), cudaMemcpyHostToDevice);

  if(err != cudaSuccess )
    {
      fprintf(stderr, "Failed to copy vector m_col from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  return cudaSuccess;
}


int FreeMatrixOnDevice(cuda_cmplx** d_m_val_ptr,int** d_m_col_ptr)
{
  if(WTRACE) cout << "In FreeMatrixOnDevice" << endl;	
  cudaError_t err;

  err = cudaFree(*d_m_val_ptr);

  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to free Matrix value memory on device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  if(WTRACE) cout << "Freed d_m_val" << endl;	

  err = cudaFree(*d_m_col_ptr);
    
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to free Matrix column memory on device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  if(WTRACE) cout << "Successfully freed matrix on device" << endl;	
  return cudaSuccess;
}


#endif // CUDA_GLOBAL_CU
