#include<complex>
#include<vector>
#include<iostream>
#include<iomanip>
using namespace std;

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

#ifdef OPENBLAS
#include<OpenBLAS/cblas.h>
#endif


#ifdef MACOSXBLAS
#include<Accelerate/Accelerate.h>
#endif

#ifdef MKL_BLAS
//#include "mkl.h"
#include "mkl_cblas.h"
#endif



// The extern C can be taken out when using cblas.h
/*
extern "C"
{
#ifdef OPENBLAS
  double ddot_(int*,double*,int*,double*,int*); // real dotproduct !!!consider dsdot also
  //  complex<double> zdotc_(int*,complex<double>*,int*,complex<double>*,int*); // complex dotprod
  double* zdotc_(int*,complex<double>*,int*,complex<double>*,int*); // complex dotprod
  double dznrm2_(int*,complex<double>*,int*); // Norm of vector
  void   zdscal_(int*,double*,complex<double>*,int*); // multiply vector with real scalar
  void   zscal_(int*,complex<double>*,complex<double>*,int*); // multiply vector with complex scalar
  void   zaxpy_(int*,complex<double>*,complex<double>*,int*,complex<double>*,int*); // add to vector
#endif
}

*/



double ScalarProduct(int* N,double* v1,double* v2)
{
  double res;
  
#ifdef OPENBLAS
  int stride=1;
  res=cblas_ddot(*N,v1,stride,v2,stride);
#elif MACOSXBLAS
  int stride=1;
  res=cblas_ddot(*N,v1,stride,v2,stride);
#elif MKL_BLAS
  int stride=1;
  res=cblas_ddot(*N,v1,stride,v2,stride);
#else
  res=0.;
  for(int i=0; i<*N; i++){res+=v1[i]*v2[i];}
#endif

  return res;
}


double ScalarProduct(vector<double>& v1,vector<double>& v2)
{
  int N=v1.size(); return ScalarProduct(&N,&v1[0],&v2[0]);
}



complex<double> ScalarProduct(int* N,complex<double>* v1,complex<double>* v2)
{
  //  if(TRACE) cout << "Calculating the complex scalar product" << endl;
  complex<double> res=complex<double>(0.,0.);
  int stride=1;

  //  if(TRACE) cout << "N: " << *N << " *v1: " << v1 << " *v2: " << v2 << " *res: " << &res << endl;

  
#ifdef OPENBLAS
  double* v1d=reinterpret_cast<double*>(v1);
  double* v2d=reinterpret_cast<double*>(v2);
  openblas_complex_double* res2=reinterpret_cast<openblas_complex_double*>(&res);
  cblas_zdotc_sub(*N,v1d,stride,v2d,stride,res2);
#elif MACOSXBLAS
  cblas_zdotc_sub(*N,v1,stride,v2,stride,&res);
#elif MKL_BLAS
  void* v1d=reinterpret_cast<void*>(v1);
  void* v2d=reinterpret_cast<void*>(v2);
  void* resd=reinterpret_cast<void*>(&res);
  cblas_zdotc_sub(*N,v1d,stride,v2d,stride,resd);
#else
  for(int i=0; i<*N; i++){ res+=conj(v1[i])*v2[i]; }
#endif

  //  if(TRACE) cout << "Returning with scalar product" << res << endl;
  return res;
}

complex<double> ScalarProduct(vector<complex<double> >& v1,vector<complex<double> >& v2)
{
  int N=v1.size(); return ScalarProduct(&N,&v1[0],&v2[0]);
}

double GetNorm(int* N,complex<double>* v)
{
  int stride=1;
#ifdef OPENBLAS
  double* vd=reinterpret_cast<double*>(v);
  double Norm=cblas_dznrm2(*N,vd,stride); 
#elif MACOSXBLAS
  double Norm=cblas_dznrm2(*N,v,stride);
#elif MKL_BLAS
  void* vd=reinterpret_cast<void*>(v);
  double Norm=cblas_dznrm2(*N,vd,stride);
#else
  MyField Norm_squared=ScalarProduct(N,v,v);
  double Norm=sqrt(Norm_squared.real());
#endif
  return Norm;
}


double Normalize(int* N,complex<double>* v)
  {
    int stride=1;
#ifdef OPENBLAS
    double* vd=reinterpret_cast<double*>(v);
    double Norm=cblas_dznrm2(*N,vd,stride); 
    double factor=1./Norm;
    cblas_zdscal(*N,factor,vd,stride); // actually normalize
#elif MACOSXBLAS
    double Norm=cblas_dznrm2(*N,v,stride);
    double factor=1./Norm;
    cblas_zdscal(*N,factor,v,stride); // actually normalize
#elif MKL_BLAS
    void* vd=reinterpret_cast<void*>(v);
    double Norm=cblas_dznrm2(*N,vd,stride);
    double factor=1./Norm;
    cblas_zdscal(*N,factor,vd,stride); // actually normalize
#else
    MyField Norm_squared=ScalarProduct(N,v,v);
    double Norm=sqrt(Norm_squared.real());
    double factor = 1./Norm;
    for(int i=0; i<*N; i++){ v[i]*=factor;}
#endif
    if(TRACE) cout << "Norm is " << setprecision(DEBUGPRECISION) << Norm << endl;
    return Norm;
  }


double Normalize(vector<complex<double> >& v)
{
  int N=v.size(); return Normalize(&N,&v[0]);
}



void AddToVector(int* N,complex<double>* v1,complex<double>* v0,complex<double>* a)
{
    int stride=1;
  // performs the routine v1=v1+a*v0;
#ifdef OPENBLAS
    double* v0d=reinterpret_cast<double*>(v0);
    double* v1d=reinterpret_cast<double*>(v1);
    double* ad=reinterpret_cast<double*>(a);
    cblas_zaxpy(*N,ad,v0d,stride,v1d,stride); 
#elif MACOSXBLAS
    cblas_zaxpy(*N,a,v0,stride,v1,stride); 
#elif MKL_BLAS
    void* v0d=reinterpret_cast<void*>(v0);
    void* v1d=reinterpret_cast<void*>(v1);
    void* ad=reinterpret_cast<void*>(a);
    cblas_zaxpy(*N,ad,v0d,stride,v1d,stride); 
#else
    for(int i=0; i<*N; i++){v1[i]+=(*a)*v0[i];}
#endif
}

void AddToVector(int* N,complex<double>* v1,complex<double>* v0,double* a)
{
  int stride=1;

  complex<double> cmplxa=complex<double>(*a,0.); // convert to complex
  AddToVector(N,v1,v0,&cmplxa);
}


void ScaleVector(int* N,complex<double>* v,complex<double>* factor)
{
  // v=factor*v;

    int stride=1;
#ifdef OPENBLAS
    double* vd=reinterpret_cast<double*>(v);
    double* factord =reinterpret_cast<double*>(factor);
    cblas_zscal(*N,factord,vd,stride); // scale the vector
#elif MACOSXBLAS
    cblas_zscal(*N,factor,v,stride); // scale the vector
#elif MKL_BLAS
    void* vd=reinterpret_cast<void*>(v);
    void* factord =reinterpret_cast<void*>(factor);
    cblas_zscal(*N,factord,vd,stride); // scale the vector
#else
    for(int i=0; i<*N; i++){v[i]*=(*factor);} // scale the vector
#endif
}

void ScaleVector(int* N,complex<double>* v,double* factor)
{
  // v=factor*v;

    int stride=1;
#ifdef OPENBLAS
    double* vd=reinterpret_cast<double*>(v);
    cblas_zdscal(*N,*factor,vd,stride); // scale the vector
#elif MACOSXBLAS
    cblas_zdscal(*N,*factor,v,stride); // scale the vector
#elif MKL_BLAS
    void* vd=reinterpret_cast<void*>(v);
    cblas_zdscal(*N,*factor,vd,stride); // scale the vector
#else
    for(int i=0; i<*N; i++){v[i]*=(*factor);}  // scale the vector
#endif
}



