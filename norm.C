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

const int NPREC=40;

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

  const int N=14;
  complex<double> a[N];
  a[0] = complex<double>(0.9943696162663400173187255859375,0.44064897857606410980224609375);
  a[1] = complex<double>(0.8651147224009037017822265625,-0.9997712378390133380889892578125);
  a[2] = complex<double>(-0.7437511044554412364959716796875,-0.3953348645009100437164306640625);
  a[3] = complex<double>(0.9980810307897627353668212890625,-0.7064882148988544940948486328125);
  a[4] = complex<double>(-0.5278220474720001220703125,-0.815322808921337127685546875);
  a[5] = complex<double>(-0.2068385477177798748016357421875,-0.62747957743704319000244140625);
  a[6] = complex<double>(-0.2241785195656120777130126953125,-0.3088785498403012752532958984375);
  a[7] = complex<double>(0.33949208073318004608154296875,-0.206465062685310840606689453125);
  a[8] = complex<double>(0.8710781452246010303497314453125,0.077633463777601718902587890625);
  a[9] = complex<double>(0.69262183643877506256103515625,-0.161610961891710758209228515625);
  a[10]= complex<double>(-0.3734529740177094936370849609375,0.370439001359045505523681640625);
  a[11]= complex<double>(0.04909632541239261627197265625,-0.5910955020226538181304931640625);
  a[12]= complex<double>(-0.11309421248733997344970703125,0.756234874017536640167236328125);
  a[13]= complex<double>(-0.54084555990993976593017578125,-0.9452248080633580684661865234375);


  // Ordinary precsion calculation      
  
  double od=0.;
  for(int i=0; i<N; i++){ od+= a[i].real()*a[i].real()+a[i].imag()*a[i].imag();}
  od =sqrt(od);
  
  cout << "Norm: " << setprecision(NPREC) << od
       << "\n(" <<  *reinterpret_cast<tDoubleStruct*>(&od) << endl; 

  // Cuda calculation
  
  double cudares=0.;

  cupNormCmplx(N,&a[0],&cudares);


  cout << "Norm from cuda: " << setprecision(NPREC) << cudares
       << "\n(" <<  *reinterpret_cast<tDoubleStruct*>(&cudares) << endl; 


  // Extended precsion calculation      
  unsigned int old_cw;
  fpu_fix_start(&old_cw);
  
#ifdef DOUBLEDOUBLE      
  dd_real aqdr[N];
  dd_real aqdi[N];
  for(int i=0; i<N; i++){aqdr[i]=dd_real(a[i].real()); aqdi[i]=dd_real(a[i].imag());}
  
  dd_real rqdr= dd_real(0.);
#elif QUADDOUBLE
  qd_real aqdr[N];
  qd_real aqdi[N];
  for(int i=0; i<N; i++){aqdr[i]=qd_real(a[i].real()); aqdi[i]=qd_real(a[i].imag());}
  
  qd_real rqdr= qd_real(0.);
#endif
  for(int i=0; i<N; i++)
    { 
      rqdr=rqdr+aqdr[i]*aqdr[i]+aqdi[i]*aqdi[i];
    }
  double rq=to_double(sqrt(rqdr));
  
  fpu_fix_end(&old_cw);
  
  cout << "Norm from qd: " << setprecision(NPREC) << rq
       << "\n(" <<  *reinterpret_cast<tDoubleStruct*>(&rq) << endl; 

  cout << "Exact result from Mathematica:" << endl;
  
  double exact=3.238649270339105369093780924088956328564407003104323067290473602074308116323412499812262237899676137;
  cout << "Mathematica: " << setprecision(NPREC) << exact
       << "\n(" <<  *reinterpret_cast<tDoubleStruct*>(&exact) << endl; 


  cout << "Normalizing the initial vector gives:" << endl;

  complex<double> nod= a[0]/od;
  cout << setprecision(NPREC) << nod
       << "\n(" <<  *reinterpret_cast<tDoubleStruct*>(&nod.real()) 
       << ",\n" << *reinterpret_cast<tDoubleStruct*>(&nod.imag()) << ")" << endl;


  complex<double> b[N];
  for(int i=0; i<N; i++){ b[i]=a[i];}

  double cudanorm=0.;
  cupNormalize(N,&b[0],&cudanorm);

  cout << "Norm from cuda: " << setprecision(NPREC) << cudanorm
       << "\n(" <<  *reinterpret_cast<tDoubleStruct*>(&cudanorm) << endl; 

  cout << "The first element is" << endl;
  cout << setprecision(NPREC) << b[0]
       << "\n(" <<  *reinterpret_cast<tDoubleStruct*>(&b[0].real()) 
       << ",\n" << *reinterpret_cast<tDoubleStruct*>(&b[0].imag()) << ")" << endl;




  complex<double> ex;

  cout << "Exact result from Mathematica:" << endl;

  ex=complex<double>(0.307032201780288406842629385799681139290685038108995515432031158996683008195009176210759922593244079,0.1360594932621171433981435912347341429435828639607626447338576798288305208402713045524814671799806946);
/*
  ex[1]=complex<double>(0.267122077813112863250028076262508310134495774110991466421332380173895210729955523265584369042066838,0.308700064250653325255561734898477987335596132455187214401289605100799550790205198080199069535308434);

 ex[2]=complex<double>(-0.229648548630141163958422877527293112880150504590262999474474508987035389831207866932443148527979608,0.1220678225708324969962166815084571100420003297961093424235538234072121776331516574532951143420065716);
*/
// for(int i=0; i<1; i++)
  {
    cout << setprecision(NPREC) << ex
	 << "\n(" <<  *reinterpret_cast<tDoubleStruct*>(&ex.real()) 
	 << "," << *reinterpret_cast<tDoubleStruct*>(&ex.imag()) << ")" << endl;
  }

}




