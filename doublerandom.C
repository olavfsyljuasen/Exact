#include<iostream>
#include<iomanip>
#include<string>
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
      double a[N];
      double b[N];

      for(int i=0; i<N; i++)
	{
	  // random initializtion
	  a[i]=RAN.Get();
	  b[i]=RAN.Get();
	}

      // use cuda:
      double rcuda=0.;
      double* a_ptr=&a[0];
      double* b_ptr=&b[0];
      
      cupRealDotProductReal(N,a_ptr,b_ptr,&rcuda);
      
      unsigned int old_cw;
      fpu_fix_start(&old_cw);
      
#ifdef DOUBLEDOUBLE      
      dd_real aqd[N];
      for(int i=0; i<N; i++){aqd[i]=dd_real(a[i]);}
      dd_real bqd[N];
      for(int i=0; i<N; i++){bqd[i]=dd_real(b[i]);}
      
      dd_real rqd= dd_real(0.);
#elif QUADDOUBLE
      qd_real aqd[N];
      for(int i=0; i<N; i++){aqd[i]=qd_real(a[i]);}
      qd_real bqd[N];
      for(int i=0; i<N; i++){bqd[i]=qd_real(b[i]);}
      
      qd_real rqd= dd_real(0.);
#else
      double aqd[N];
      for(int i=0; i<N; i++){aqd[i]=a[i];}
      double bqd[N];
      for(int i=0; i<N; i++){bqd[i]=b[i];}
      
      double rqd=0.;
#endif
      
      for(int i=0; i<N; i++){ rqd=rqd+aqd[i]*bqd[i];}
      
#if defined DOUBLEDOUBLE || defined QUADDOUBLE
      double rq=to_double(rqd);
#else
      double rq=rqd;
#endif

      fpu_fix_end(&old_cw); 

      double diff=rcuda-rq;


      cout << "********* iteration " << r << " ******* " << endl; 

      if(diff !=0.){
	diffcounter++; 
	cout << "WARNING: Difference: " << diff << endl;

#ifdef DOUBLEDOUBLE
	cout << "with cuda doubledouble add:" << endl;
#elif QUADDOUBLE
	cout << "with cuda quaddouble add:" << endl;
#else
	cout << "with cuda add:" << endl;
#endif
	cout << setprecision(NPREC) << rcuda << "\n" <<  *reinterpret_cast<tDoubleStruct*>(&rcuda) << endl; 

	cout << "with double-double precision:" << endl;
	cout << setprecision(NPREC) << rq << "\n" <<  *reinterpret_cast<tDoubleStruct*>(&rq) << endl; 
      }
    }
  cout << "The # of differences: " << diffcounter << endl;
}
