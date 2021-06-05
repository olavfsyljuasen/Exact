#include<iostream>
#include<iomanip>
#include<string>
#include<complex>
#include <stdio.h>
#include <bitset>
using namespace std;

#include "rnddef.h"

#include <qd/dd_real.h>
#include <qd/qd_real.h>
#include <qd/fpu.h>

#include "cuda_precision.h"


const int NPREC=100;

typedef struct {
  unsigned int mantissa_low:32;     
  unsigned int mantissa_high:20;
  unsigned int exponent:11;        
  unsigned int sign:1;
} tDoubleStruct;


typedef struct {     
  unsigned int fraction:23;
  unsigned int exponent:8;        
  unsigned int sign:1;
} tFloatStruct;

ostream& operator<<(ostream& os,tDoubleStruct b)
{
  os << bitset<1>(b.sign) << " " 
     << bitset<11>(b.exponent) << " " 
     << bitset<20>(b.mantissa_high) << " "
     << bitset<32>(b.mantissa_low);
  return os;
}

typedef union{
  double d;
  unsigned char c[8];
} cdouble;


int main()
{
  const int Niter=1000; 
  const int N = 1000;

  int diffcounter=0;
  
  for(int r=0; r<Niter; r++)
    {
      complex<double> a[N];
      complex<double> b[N];

      for(int i=0; i<N; i++)
	{
	  // random initializtion
	  a[i].real()=RAN.Get();
	  b[i].real()=RAN.Get();
	  a[i].imag()=RAN.Get();
	  b[i].imag()=RAN.Get();
	}




      // use cuda:
      complex<double> rcuda=complex<double>(0.,0.);
      
      void* rcuda_ptr=reinterpret_cast<void*>(&rcuda);
      cupCmplxDotProductCmplx(N,&a[0],&b[0],rcuda_ptr);
      /*
#ifdef DOUBLEDOUBLE
      cout << "with cuda doubledouble add:" << endl;
#elif QUADDOUBLE
      cout << "with cuda quaddouble add:" << endl;
#else
      cout << "with cuda add:" << endl;
#endif
      cout << setprecision(NPREC) << rcuda << "\n" <<  *reinterpret_cast<tDoubleStruct*>(&rcuda) << endl; 
      */
      // ensure that 80-bits arithmetic is not there
      
      unsigned int old_cw;
      fpu_fix_start(&old_cw);
      
#ifdef DOUBLEDOUBLE      
      dd_real aqdr[N];
      dd_real aqdi[N];
      for(int i=0; i<N; i++){aqdr[i]=dd_real(a[i].real()); aqdi[i]=dd_real(a[i].imag());}
      dd_real bqdr[N];
      dd_real bqdi[N];
      for(int i=0; i<N; i++){bqdr[i]=dd_real(b[i].real()); bqdi[i]=dd_real(b[i].imag());}
      
      dd_real rqdr= dd_real(0.);
      dd_real rqdi= dd_real(0.);
#elif QUADDOUBLE
      qd_real aqdr[N];
      qd_real aqdi[N];
      for(int i=0; i<N; i++){aqdr[i]=qd_real(a[i].real()); aqdi[i]=qd_real(a[i].imag());}
      qd_real bqdr[N];
      qd_real bqdi[N];
      for(int i=0; i<N; i++){bqdr[i]=qd_real(b[i].real()); bqdi[i]=qd_real(b[i].imag());}
      
      qd_real rqdr= qd_real(0.);
      qd_real rqdi= qd_real(0.);
#endif
      for(int i=0; i<N; i++)
	{ 
	  rqdr=rqdr+aqdr[i]*bqdr[i]+aqdi[i]*bqdi[i];
	  rqdi=rqdi+aqdr[i]*bqdi[i]-aqdi[i]*bqdr[i];
	}
      
      complex<double> rq=complex<double>(to_double(rqdr),to_double(rqdi));
      /*
      cout << "with double-double precision:" << endl;
      cout << setprecision(NPREC) << rq << "\n" <<  *reinterpret_cast<tDoubleStruct*>(&rq) << endl; 
      */
      fpu_fix_end(&old_cw); 


      cout << "********* iteration " << r << " ******* " << endl; 

      complex<double> diff=rcuda-rq;
      if(diff !=complex<double>(0.,0.)){
	diffcounter++; 
	cout << "WARNING: Difference: " << diff << endl;

#ifdef DOUBLEDOUBLE
	cout << "with doubledouble add:" << endl;
#elif QUADDOUBLE
	cout << "with quaddouble add:" << endl;
#else
	cout << "with double add:" << endl;
#endif
	double rcudar=rcuda.real();
	double rcudai=rcuda.imag();

	double rqr=rq.real();
	double rqi=rq.imag();

	cout << "real part: " << endl;

	cout << "cuda:" << endl;
	cout << setprecision(NPREC) << rcudar << "\n" <<  *reinterpret_cast<tDoubleStruct*>(&rcudar) << endl; 

	cout << "using qd library:" << endl;
	cout << setprecision(NPREC) << rqr << "\n" <<  *reinterpret_cast<tDoubleStruct*>(&rqr) << endl; 

	cout << "imaginary part: " << endl;

	cout << "cuda:" << endl;
	cout << setprecision(NPREC) << rcudai << "\n" <<  *reinterpret_cast<tDoubleStruct*>(&rcudai) << endl; 

	cout << "using qd library:" << endl;
	cout << setprecision(NPREC) << rqi << "\n" <<  *reinterpret_cast<tDoubleStruct*>(&rqi) << endl; 
      }
    }
  cout << "The # of differences: " << diffcounter << endl;
}
