#ifndef BITMANIP_H
#define BITMANIP_H

#include<iostream>

using namespace std;

typedef long int pstate; 

const pstate NOSTATE=-1; // -1 is reserved for No allowed state 
const pstate pstate1=1; // define the unit 1 to be of the type pstate
const pstate pstate0=0; // define 0 to be of the type pstate


const pstate nlowerbits[64]={
0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383,
32767, 65535, 131071, 262143, 524287, 1048575, 2097151, 4194303, 
8388607, 16777215, 33554431, 67108863, 134217727, 268435455, 
536870911, 1073741823, 2147483647, 4294967295, 8589934591, 
17179869183, 34359738367, 68719476735, 137438953471, 274877906943, 
549755813887, 1099511627775, 2199023255551, 4398046511103,
8796093022207, 17592186044415, 35184372088831, 70368744177663,
140737488355327, 281474976710655, 562949953421311, 1125899906842623,
2251799813685247, 4503599627370495, 9007199254740991,
18014398509481983, 36028797018963967, 72057594037927935,
144115188075855871, 288230376151711743, 576460752303423487,
1152921504606846975, 2305843009213693951, 4611686018427387903,9223372036854775807}; // the numerical values of the n lowest bits

const pstate MAXSTATE=nlowerbits[63]; // 



/*
void setbitinplace(pstate& s,int k){s=s|(pstate1<<k); return s;}
void clrbitinplace(pstate& s,int k){s=s&~(pstate1<<k);}
void flpbitinplace(pstate& s,int k){s=s^(pstate1<<k);}
void setbitsinplace(pstate& s,int k,int n){pstate b=pstate1<<k; for(int i=0; i<n; i++){s=s|b;  b<<=1;}}
void clrbitsinplace(pstate& s,int k,int n){pstate b=pstate1<<k; for(int i=0; i<n; i++){s=s&~b; b<<=1;}}
void flpbitsinplace(pstate& s,int k,int n){pstate b=pstate1<<k; for(int i=0; i<n; i++){s=s^b;  b<<=1;}}
*/

pstate setbit(pstate s,int k,int cycle){return s|(pstate1<<k);}
pstate clrbit(pstate s,int k,int cycle){return s&~(pstate1<<k);}
pstate flpbit(pstate s,int k,int cycle){return s^(pstate1<<k);}
pstate setbits(pstate s,int k,int n,int cycle){pstate b=pstate1<<k; for(int i=0; i<n; i++){s=s|b;  b<<=1;} return s;}
pstate clrbits(pstate s,int k,int n,int cycle){pstate b=pstate1<<k; for(int i=0; i<n; i++){s=s&~b; b<<=1;} return s;}
pstate flpbits(pstate s,int k,int n,int cycle){pstate b=pstate1<<k; for(int i=0; i<n; i++){s=s^b;  b<<=1;} return s;}

pstate invert(pstate s,const int cycle){pstate b=pstate1<<0; for(int i=0; i<cycle; i++){s=s^b;  b<<=1;} return s;}

bool isbitset(const pstate s,const int k,const int cycle){return (s&(pstate1<<k))>>k==pstate1;}

int maxbitset(pstate s,int cycle)
{
  int max=-1; // returns -1 if no bits are set
  for(int i=0; s!=0; s>>=1){max++;}
  return max;
}

int bitcount(pstate s,int cycle)
{
  int count=0;
  for(int i=0; s!=0; s>>=1){if(s & pstate1) count++;}
  return count;
}

pstate cyclicshiftright(pstate s,int cycle)
{
  pstate mask=(s & pstate1) << (cycle-1);
  pstate t=s>>1;
  t=t|mask;
  return t;
}

pstate cyclicshiftnright(pstate s,int n,int cycle)
{
  pstate mask=(s & nlowerbits[n]) << (cycle-n);
  pstate t=s>>n;
  t=t|mask;
  return t;
}

pstate cyclicshiftleft(pstate s,int cycle)
{
  pstate mask=( s & (pstate1 << (cycle-1)) ) >> (cycle-1);
  pstate t=s<<1;
  t=t|mask;
  t=t&nlowerbits[cycle];
  return t;
}

pstate cyclicshiftnleft(pstate s,int n,int cycle)
{
  pstate mask=(s & (nlowerbits[n] << (cycle-n)) ) >> (cycle-n);
  pstate t=s<<n;
  t=t|mask;
  t=t&nlowerbits[cycle];
  return t;
}

pstate reflect(pstate s,int cycle)
{
  pstate t=pstate0;
  for(int i=0; s!=0 && i<cycle; i++,s>>=1){t =t | ((s& pstate1)<<(cycle-i-1));}   
  return t;
}


string bitstring(pstate s,int cycle)
{
  string str=(s==pstate0 ? "0":"");
  for(int i=0; (s!=0 || i<cycle); s>>=1, i++){if(s & pstate1){str='1'+str;}else{str='0'+str;}}
  return str;
}

/*
string bitstring(pstate s,int z)
{
  string str=(s==pstate0 ? "0":"");
  for(int i=0; (s!=0 || i<z); s>>=1, i++){if(s & pstate1){str='1'+str;}else{str='0'+str;}}
  return str;
}
*/

int Nbitwalls(pstate s,int cycle)
{
  return bitcount(cyclicshiftright(s,cycle)^s,cycle);
}

/*
pstate Splussi(const pstate s,const int i)
{
  if(s==NOSTATE || isbitset(s,i)){return NOSTATE;}
  return setbit(s,i);
}

pstate Sminusi(const pstate s,const int i)
{
  if(s==NOSTATE || !isbitset(s,i)){return NOSTATE;}
  return clrbit(s,i);
}

double Szi(const pstate s,const int i)
{
  return (isbitset(s,i) ? 0.5:-0.5); 
}
*/

pstate FlipSpinUp(const pstate s,const int i,const int cycle)
{
  if(s==NOSTATE || isbitset(s,i,cycle)){return NOSTATE;}
  return setbit(s,i,cycle);
}

pstate FlipSpinDown(const pstate s,const int i,const int cycle)
{
  if(s==NOSTATE || !isbitset(s,i,cycle)){return NOSTATE;}
  return clrbit(s,i,cycle);
}

int SpinDirection(const pstate s,const int i,const int cycle)
{
  return (isbitset(s,i,cycle) ? 1:-1); 
}


#endif // BITMANIP_H
/*
int main()
{
  pstate b=(pstate1<<2);   


  cout << "b: " << b << " b<<1: " << (b<<1) << endl;
  pstate c=maxbitset(b);
  cout << "b:" << b << ",maxbitset: " << maxbitset(b) << ",bitcount:" << bitcount(b) << ",bitstring:" << bitstring(b) << endl;
  cout << "b:" << b << ",maxbitset: " << maxbitset(b) << ",bitcount:" << bitcount(b) << ",bitstring:" << bitstring(b) << endl;
  b=clrbits(b,5,2);
  cout << "b:" << b << ",maxbitset: " << maxbitset(b) << ",bitcount:" << bitcount(b) << ",bitstring:" << bitstring(b) << endl;
  b=setbits(b,0,2);
  cout << "b:" << b << ",maxbitset: " << maxbitset(b) << ",bitcount:" << bitcount(b) << ",bitstring:" << bitstring(b) << endl;
  b=pstate0;
  cout << "b:" << b << ",maxbitset: " << maxbitset(b) << ",bitcount:" << bitcount(b) << ",bitstring:" << bitstring(b) << endl;
  b=flpbits(b,0,7);
  cout << "b:" << b << ",maxbitset: " << maxbitset(b) << ",bitcount:" << bitcount(b) << ",bitstring:" << bitstring(b) << endl;



 
}

*/
