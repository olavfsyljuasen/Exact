#include<iostream>
#include<iomanip>
#include<string>
#include <stdio.h>
#include <bitset>
#include<complex>


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

  unsigned char cin[4][8]={225,233,110,138,63,93,242,63,
			   196,106,11,10,101,43,233,191,
			   52,115,234,125,143,133,254,63,
			   25,70,72,93,78,188,238,63};
  
  unsigned char din[4][8]={16,252,185,200,226,32,230,191,
			   16,23,254,228,85,182,230,191,			   
			   140,161,229,135,45,240,237,191,			   
			   244,223,65,187,92,151,251,63};
			   
  
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
  complex<double> a[4];
  complex<double> b[4];

  cdouble c_a[4];
  cdouble c_b[4];
  cdouble c_c[4];
  cdouble c_d[4];
  

  
  for(int j=0; j<4 ; j++)
    {
      for(int i=0; i<8; i++)
	{
	  c_a[j].c[i]=ain[j][i]; 
	  c_b[j].c[i]=bin[j][i];
	  c_c[j].c[i]=cin[j][i]; 
	  c_d[j].c[i]=din[j][i];
	}

      a[j].real()=c_a[j].d;  a[j].imag()=c_d[j].d; 
      b[j].real()=c_b[j].d;  b[j].imag()=c_c[j].d; 
    }	      

  for(int j=0; j<4 ; j++)
    {
      cout << setprecision(NPREC) << a[j] << "," << endl; 
    }

  cout << endl;
  for(int j=0; j<4 ; j++)
    {
      cout << setprecision(NPREC) << b[j]  << "," << endl; 
    }

 
  // computation:


  complex<double> rl1=complex<double>(0.,0.); for(int i=0; i<4; i++){rl1 += conj(a[i])*b[i];}
  //  double rl=a[0]*a[0];

  cout << "loop addition: " << endl;
  cout << setprecision(NPREC) << rl1
       << "\n(" <<  *reinterpret_cast<tDoubleStruct*>(&rl1.real()) 
       << "," << *reinterpret_cast<tDoubleStruct*>(&rl1.imag()) << ")" << endl;

  complex<double> rb= ((conj(a[0])*b[0]+conj(a[1])*b[1])+conj(a[2])*b[2])+conj(a[3])*b[3];
  cout << "loop addition on one line:" << endl;
  cout << setprecision(NPREC) << rb 
       << "\n(" <<  *reinterpret_cast<tDoubleStruct*>(&rb.real()) 
       << "," << *reinterpret_cast<tDoubleStruct*>(&rb.imag()) << ")" << endl;

complex<double> rc= conj(a[0])*b[0]+conj(a[1])*b[1]+conj(a[2])*b[2]+conj(a[3])*b[3];  
  cout << "all-on one line addition:" << endl;
  cout << setprecision(NPREC) << rc 
       << "\n(" <<  *reinterpret_cast<tDoubleStruct*>(&rc.real()) 
       << "," << *reinterpret_cast<tDoubleStruct*>(&rc.imag()) << ")" << endl;

complex<double> rt = (conj(a[0])*b[0]+conj(a[1])*b[1])+(conj(a[2])*b[2]+conj(a[3])*b[3]); 
  cout << "tree addition:" << endl;
  cout << setprecision(NPREC) << rt
       << "\n(" <<  *reinterpret_cast<tDoubleStruct*>(&rt.real()) 
       << "," << *reinterpret_cast<tDoubleStruct*>(&rt.imag()) << ")" << endl;


complex<double> rp= (conj(a[0])*b[0]+conj(a[2])*b[2])+(conj(a[1])*b[1]+conj(a[3])*b[3]); 
cout << "permuted tree addition:" << endl;
cout << setprecision(NPREC) << rp
			       << "\n(" <<  *reinterpret_cast<tDoubleStruct*>(&rp.real()) 
	<< "," << *reinterpret_cast<tDoubleStruct*>(&rp.imag()) << ")" << endl;
      

  // use cuda:
 complex<double> rcuda=complex<double>(0.,0.);
  
 cupCmplxDotProductCmplx(4,&a[0],&b[0],&rcuda);
#ifdef DOUBLEDOUBLE 
 cout << "with cuda double-double add:" << endl;
#elif QUADDOUBLE
 cout << "with cuda quad-double add:" << endl;
#endif
 double rcudar=rcuda.real();
 double rcudai=rcuda.imag();

 cout << setprecision(NPREC) << rcuda
      << "\n(" <<  *reinterpret_cast<tDoubleStruct*>(&rcudar) 
      << "," << *reinterpret_cast<tDoubleStruct*>(&rcudai) << ")" << endl;
 
  // ensure that 80-bits arithmetic is not there
  unsigned int old_cw;
  fpu_fix_start(&old_cw);

#ifdef DOUBLEDOUBLE
  dd_real aqd_r[4];
  dd_real aqd_i[4];
  for(int i=0; i<4; i++){aqd_r[i]=dd_real(a[i].real()); aqd_i[i]=dd_real(a[i].imag());}
  dd_real bqd_r[4];
  dd_real bqd_i[4];
  for(int i=0; i<4; i++){bqd_r[i]=dd_real(b[i].real()); bqd_i[i]=dd_real(b[i].imag());}

  dd_real rqd_r=dd_real(0.);
  dd_real rqd_i=dd_real(0.);
#elif QUADDOUBLE
  qd_real aqd_r[4];
  qd_real aqd_i[4];
  for(int i=0; i<4; i++){aqd_r[i]=qd_real(a[i].real()); aqd_i[i]=qd_real(a[i].imag());}
  qd_real bqd_r[4];
  qd_real bqd_i[4];
  for(int i=0; i<4; i++){bqd_r[i]=qd_real(b[i].real()); bqd_i[i]=qd_real(b[i].imag());}

  qd_real rqd_r=qd_real(0.);
  qd_real rqd_i=qd_real(0.);
#else
  double aqd_r[4];
  double aqd_i[4];
  for(int i=0; i<4; i++){aqd_r[i]=a[i].real(); aqd_i[i]=a[i].imag();}
  double bqd_r[4];
  double bqd_i[4];
  for(int i=0; i<4; i++){bqd_r[i]=b[i].real(); bqd_i[i]=b[i].imag();}

  double rqd_r=0.;
  double rqd_i=0.;
#endif

  for(int i=0; i<4; i++)
    {
      rqd_r += (aqd_r[i]*bqd_r[i] +aqd_i[i]*bqd_i[i]);
      rqd_i += (aqd_r[i]*bqd_i[i] -aqd_i[i]*bqd_r[i]);
    }
 
#if defined DOUBLEDOUBLE || defined QUADDOUBLE
  complex<double> rq=complex<double>(to_double(rqd_r),to_double(rqd_i));
#else
  complex<double> rq=complex<double>(rqd_r,rqd_i));
#endif
  cout << "with double-double precision:" << endl;
  cout << setprecision(NPREC) << rq 
       << "\n(" <<  *reinterpret_cast<tDoubleStruct*>(&rq.real())
       << "," << *reinterpret_cast<tDoubleStruct*>(&rq.imag()) << ")" << endl;
  
  fpu_fix_end(&old_cw); 

  

  cout << "Exact result from Mathematica:" << endl;
/*
complex<double> ex1=complex<double>(-2.578394175362822422900803053404825344984911612311634143733865554004580644686939194798469543457031250,1.542527326076506160842781164955092073202660040558017418588243087340661219286630512215197086334228516); 
  cout << setprecision(NPREC) << ex1
       << "\n(" <<  *reinterpret_cast<tDoubleStruct*>(&ex1.real()) 
       << "," << *reinterpret_cast<tDoubleStruct*>(&ex1.imag()) << ")" << endl;
*/

  complex<double> ex=complex<double>(-0.3070546396189546020286552234522465097575886804157143225425352466018136254888304392807185649871826171875,7.6196861207298429880377363264937364957732460596802295769322286114223763409114553724066354334354400634765625);

  cout << setprecision(NPREC) << ex
       << "\n(" <<  *reinterpret_cast<tDoubleStruct*>(&ex.real()) 
       << "," << *reinterpret_cast<tDoubleStruct*>(&ex.imag()) << ")" << endl;



  
  
}




