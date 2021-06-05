#include<iostream>
#include<iomanip>
#include<string>
#include <stdio.h>
#include <bitset>


#include <qd/dd_real.h>
#include <qd/qd_real.h>
#include <qd/fpu.h>

#include "cuda_precision.h"

using namespace std;

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

  // initialize the double numbers using characters

  unsigned char ain[4][8]={52,115,234,125,143,133,254,63,
			   196,106,11,10,101,43,233,191,
			   225,233,110,138,63,93,242,63,
			   25,70,72,93,78,188,238,63};
  
  unsigned char bin[4][8]={140,161,229,135,45,240,237,191,
			   16,252,185,200,226,32,230,191,
			   244,223,65,187,92,151,251,63,
			   16,23,254,228,85,182,230,191};
  
  /*
  unsigned char ain[4][8]={1,0,0,0,0,0,248,64,
			   0,0,0,0,0,0,0,0,
			   0,0,0,0,0,0,0,0,
			   0,0,0,0,0,0,0,0};

  unsigned char bin[4][8]={255,255,255,16,0,0,240,63,
			   0,0,0,0,0,0,0,0,
			   0,0,0,0,0,0,0,0,
			   0,0,0,0,0,0,0,0};

  */
  cdouble c_a[4];
  cdouble c_b[4];
  
  double a[4];
  double b[4];
  
  for(int j=0; j<4 ; j++)
    {
      for(int i=0; i<8; i++){c_a[j].c[i]=ain[j][i]; c_b[j].c[i]=bin[j][i];}
      a[j]=c_a[j].d; b[j]=c_b[j].d;
    }	      
  
  cout << "a : " << endl;
  for(int j=0; j<4 ; j++)
    {

      cout << setprecision(NPREC) << a[j] << " : " 
	   <<  *reinterpret_cast<tDoubleStruct*>(&a[j]) << endl;
    }
  
  cout << "b : " << endl;
  for(int j=0; j<4 ; j++)
    {
      cout << setprecision(NPREC) << b[j] << " : " 
	   <<  *reinterpret_cast<tDoubleStruct*>(&b[j]) << endl;
    }
  
  // computation:


  double rl1=0.; for(int i=0; i<4; i++){rl1 += a[i]*b[i]+0.;}
  //  double rl=a[0]*a[0];

  cout << "loop addition: " << endl;
  //  cout << setprecision(NPREC) << rl1 << " " <<  *reinterpret_cast<tDoubleStruct*>(&rl1) << endl;
  cout << setprecision(NPREC) << rl1 << endl;

  double rb= ((a[0]*b[0]+a[1]*b[1])+a[2]*b[2])+a[3]*b[3];
  cout << "loop addition on one line:" << endl;
  cout << setprecision(NPREC) << rb << "\n" <<  *reinterpret_cast<tDoubleStruct*>(&rb) << endl; 

  double rc= a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3];  
  cout << "all-on one line addition:" << endl;
  cout << setprecision(NPREC) << rc << "\n" <<  *reinterpret_cast<tDoubleStruct*>(&rc) << endl; 
  
  double rt = (a[0]*b[0]+a[1]*b[1])+(a[2]*b[2]+a[3]*b[3]); 
  cout << "tree addition:" << endl;
  cout << setprecision(NPREC) << rt << "\n" <<  *reinterpret_cast<tDoubleStruct*>(&rt) << endl;

  double rp= (a[0]*b[0]+a[2]*b[2])+(a[1]*b[1]+a[3]*b[3]); 
  cout << "permuted tree addition:" << endl;
  cout << setprecision(NPREC) << rc << "\n" <<  *reinterpret_cast<tDoubleStruct*>(&rp) << endl; 



  // use cuda:
  double rcuda=0.;
  double* a_ptr=&a[0];
  double* b_ptr=&b[0];
  
  cupRealDotProductReal(4,a_ptr,b_ptr,&rcuda);

#ifdef DOUBLEDOUBLE
  cout << "with cuda doubledouble add:" << endl;
#elif QUADDOUBLE
  cout << "with cuda quaddouble add:" << endl;
#else
  cout << "with cuda add:" << endl;
#endif
  cout << setprecision(NPREC) << rcuda << "\n" <<  *reinterpret_cast<tDoubleStruct*>(&rcuda) << endl; 

  // ensure that 80-bits arithmetic is not there

  unsigned int old_cw;
  fpu_fix_start(&old_cw);

#ifdef DOUBLEDOUBLE
  dd_real aqd[4];
  for(int i=0; i<4; i++){aqd[i]=dd_real(a[i]);}
  dd_real bqd[4];
  for(int i=0; i<4; i++){bqd[i]=dd_real(b[i]);}
  dd_real rqd;
#elif QUADDOUBLE
  qd_real aqd[4];
  for(int i=0; i<4; i++){aqd[i]=qd_real(a[i]);}
  qd_real bqd[4];
  for(int i=0; i<4; i++){bqd[i]=qd_real(b[i]);}
  qd_real rqd;
#else
  double aqd[4];
  for(int i=0; i<4; i++){aqd[i]=a[i];}
  double bqd[4];
  for(int i=0; i<4; i++){bqd[i]=b[i];}
  double rqd;
#endif

  rqd= aqd[0]*bqd[0]+aqd[1]*bqd[1]+aqd[2]*bqd[2]+aqd[3]*bqd[3];

#if defined DOUBLEDOUBLE || defined QUADDOUBLE
  double rq=to_double(rqd);
#else
  double rq=rqd;
#endif

  cout << "with double-double precision:" << endl;
  cout << setprecision(NPREC) << rq << "\n" <<  *reinterpret_cast<tDoubleStruct*>(&rq) << endl; 
  
  fpu_fix_end(&old_cw); 


  cout << "Exact result from Mathematica:" << endl;
  //  cout << "0.05676694142698115610397570616101072896295363295438396079189459497715830593733699060976505279541015625" << endl;
  
  double ex=0.05676694142698115610397570616101072896295363295438396079189459497715830593733699060976505279541015625;
  cout << setprecision(NPREC) << ex << "\n" <<  *reinterpret_cast<tDoubleStruct*>(&ex) << endl;
  
}




