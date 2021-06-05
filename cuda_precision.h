#ifndef CUDA_PRECISION_H
#define CUDA_PRECISION_H

using namespace std;

#include "precision_types.h"


int cudap_mat_vec_multiply_cmplx(
const int M,const int Nc,const int* indices ,
const void* data , const cmplxformat* x, cmplxformat* y);

int cudap_real_Zmaxpy(const int N,const realformat* q,cmplxformat* d_x,const cmplxformat* d_y); 
int cudap_cmplx_Zmaxpy(const int N,const cmplxformat* q,cmplxformat* d_x,const cmplxformat* d_y); 

int cudap_realdotproduct_real(const int N,const realformat* d_x,const realformat* d_y,realformat* ans); 
int cudap_realdotproduct_cmplx(const int N,const cmplxformat* d_x,const cmplxformat* d_y,realformat* ans); 
int cudap_cmplxdotproduct_cmplx(const int N,const cmplxformat* d_x,const cmplxformat* d_y,cmplxformat* ans); 

int cudap_norm_cmplx(const int N,const cmplxformat* d_x,void* ans); 
int cudap_Zdinvscal(int N,realformat* q,cmplxformat* d_x);
int cudap_Zdscal(int N,realformat* q,cmplxformat* d_x);
int cudap_Normalize(const int N,cmplxformat* d_x,double* ans); 

int cupRealDotProductReal(const int N,const double* x,const double* y,double* res);
int cupRealDotProductCmplx(const int N,const void* x,const void* y,double* res);
int cupCmplxDotProductCmplx(int N,void* x,void* y,void* res);
int cupNormCmplx(int N,void* x,double* res);
int cupNormalize(int N,void* x,double* res);

int AllocateSpaceOnDevice(cmplxformat** d_p,const int N);
int AllocateSpaceOnDevice(realformat** d_p,const int N);

int UploadToDevice(const cmplxformat* phi,const int N,cmplxformat** d_phi_ptr);
int UploadToDevice(const realformat* phi,const int N,realformat** d_phi_ptr);

int DownloadFromDevice(cmplxformat **d_phi_ptr,const int N,cmplxformat* phi);
int DownloadFromDevice(realformat **d_phi_ptr,const int N,realformat* phi);

int FreeMemoryOnDevice(cmplxformat** d_phi);
int FreeMemoryOnDevice(realformat** d_phi);

int UploadToDeviceAndExpand(const void* p,const int N,cmplxformat** d_p);
int UploadToDeviceAndExpand(const double* p,const int N,realformat** d_p);

int DownloadFromDeviceAndContract(cmplxformat** d_p,int N,void* p);
int DownloadFromDeviceAndContract(realformat** d_p,int N,double* p);

int InspectDevice(cmplxformat** d_p,int N);
int InspectDevice(realformat** d_p,int N);

#endif // CUDA_PRECISION_H
