#ifndef TWODOMAINSTATESMANIP_H
#define TWODOMAINSTATESMANIP_H

#include<iostream>

using namespace std;


struct twodomainstate
{
twodomainstate(short int j_in=0,short int l_in=0):j(j_in),l(l_in){}
  short int j;
  short int l;
};

typedef twodomainstate pstate; 

bool operator<(pstate a,pstate b)
{
  return( a.l<b.l ? true: (a.l==b.l && a.j<b.j ? true: false));
} 

bool operator<=(pstate a,pstate b)
{
  return( a.l<b.l ? true: (a.l==b.l && a.j<=b.j ? true: false));
} 

bool operator==(pstate a,pstate b)
{
  return( a.l==b.l && a.j==b.j);
} 

ostream& operator<<(ostream& os, pstate a){return os << "|" << a.j << "," << a.l << ">";} 


const pstate NOSTATE=pstate(-1,-1); // -1 is reserved for No allowed state 
const pstate pstate0=pstate( 0, 0); // define 0 to be of the type pstate

//const pstate MAXSTATE=pstate; // something

/*
pstate setbit(pstate s,int k){return s|(pstate1<<k);}
pstate clrbit(pstate s,int k){return s&~(pstate1<<k);}
pstate flpbit(pstate s,int k){return s^(pstate1<<k);}
pstate setbits(pstate s,int k,int n){pstate b=pstate1<<k; for(int i=0; i<n; i++){s=s|b;  b<<=1;} return s;}
pstate clrbits(pstate s,int k,int n){pstate b=pstate1<<k; for(int i=0; i<n; i++){s=s&~b; b<<=1;} return s;}
pstate flpbits(pstate s,int k,int n){pstate b=pstate1<<k; for(int i=0; i<n; i++){s=s^b;  b<<=1;} return s;}

bool isbitset(const pstate s,const int k){return (s&(pstate1<<k))>>k==pstate1;}

*/

pstate setbits(const pstate s,const int k,const int n,const int N)
{
  pstate t=NOSTATE;
  if(s.l==0){ t.j=k; t.l=n;} 
  if(k==(s.j+s.l)%N && s.l+n <= N){ t.l=s.l+n; t.j=(t.l==N ? 0:s.j);}
  if((k+n)%N==s.j && s.l+n <= N){ t.l=s.l+n; t.j=(t.l==N ? 0:k);}
  return t;
}

bool isbitset(const pstate s,const int k,const int N)
{
  short int b=s.j;             // first up-spin in domain
  short int e=(s.j+s.l-1+N)%N; // last  up-spin in domain  

  return(b<e && k>=b && k<=e || (b>e && (k<=e || k>=b)) ? true:false);
}

int maxbitset(const pstate s,const int N)
{
  if(s.l==0){ return -1;}
  if( (s.j+s.l-1)%N > s.j ){return (s.j+s.l-1)%N;}
  return N-1; // if wrap around or all spins up
}


int bitcount(const pstate s,const int N){return s.l;}


pstate flpbits(pstate s,int k,int n,int N)
{
  if(s.l==0){ s.j=k; s.l=n; return s;}
  if(s.j=k && s.l==n){ s.j=0; s.l=0; return s;}
  if(s.l==N){ s.j=(k+n)%N; s.l=n; return s;}
  if(s.j=(k+n)%N && s.l==n){ s.j=0; s.l=N; return s;}

  return NOSTATE;
}

pstate invert(const pstate s,const int N)
{
  pstate t=s;
  t.j = (s.j+s.l)%N; // also works for |0,0> and |0,N>
  t.l = N-s.l;

  return t;
}

pstate cyclicshiftright(const pstate s,const int N)
{
  pstate t=s;
  if(t.l !=0 && t.l !=N){t.j= (t.j-1+N)%N;}
  return t;
}

pstate cyclicshiftnright(const pstate s,const int n,const int N)
{
  pstate t=s;
  if(t.l !=0 && t.l !=N){t.j= (t.j-n+N)%N;}
  return t;
}

pstate cyclicshiftleft(const pstate s,const int N)
{
  pstate t=s;
  if(t.l !=0 && t.l !=N){t.j= (t.j+1)%N;}
  return t;
}

pstate cyclicshiftnleft(const pstate s,const int n,const int N)
{
  pstate t=s;
  if(t.l !=0 && t.l !=N){t.j= (t.j+n)%N;}
  return t;
}

pstate reflect(const pstate s,const int N)
{
  pstate t=s;
  if(t.l !=0 && t.l !=N){t.j= (N-t.j-t.l+N)%N;}
  return t;
}

string bitstring(const pstate s,const int N)
{
  string str;

  if(s.l==0){for(int i=0; i<N; i++){str=str+'0';} return str;} // all down
  if(s.l==N){for(int i=0; i<N; i++){str=str+'1';} return str;} // all up

  short int b=s.j;             // first up-spin in domain
  short int e=(s.j+s.l-1+N)%N; // last  up-spin in domain

  if(b<=e)
    { // normal sequence:
      for(int i=0; i<b; i++){str='0'+str;} 
      for(int i=b; i<=e; i++){str='1'+str;} 
      for(int i=e+1; i<N; i++){str='0'+str;} 
    }
  else
    { // wrap-around
      for(int i=0; i<=e; i++){str='1'+str;} 
      for(int i=e+1; i<b; i++){str='0'+str;} 
      for(int i=b; i<N; i++){str='1'+str;} 
    }
  return str;    
}

int Nbitwalls(const pstate s,const int N)
{
  return (s.l==0 || s.l==N ? 0 : 2);
}

pstate FlipSpinUp(const pstate s,const int i,const int N)
{
  pstate t=NOSTATE; // the default return value;
  if(s.l==0){t.j=i; t.l=1; return t;} // all spins down 
  if(s.l==N || s==NOSTATE){return t;} // all spins up or NOSTATE
  if(i==(s.j-1+N)%N){t.l=s.l+1; t.j=(t.l==N ? 0:i  ); return t;}
  if(i==(s.j+s.l)%N){t.l=s.l+1; t.j=(t.l==N ? 0:s.j); return t;}
  return t;
}

pstate FlipSpinDown(const pstate s,const int i,const int N)
{
  pstate t=NOSTATE; // the default return value;
  if(s.l==N){t.j=(i+1)%N; t.l=N-1; return t;} // all spins up
  if(s.l==0 || s==NOSTATE){return t;} // all spins down or NOSTATE
  if(i==s.j          ){t.l=s.l-1; t.j=(t.l==0 ? 0:(s.j+1)%N); return t;}
  if(i==(s.j+s.l-1)%N){t.l=s.l-1; t.j=(t.l==0 ? 0: s.j     ); return t;}
  return t;
}

int SpinDirection(const pstate s,const int i,const int N)
{
  if(s.l==0) return -1;
  if(s.l==N) return +1;

  short int b=s.j;  
  short int e=(s.j+s.l-1+N)%N;

  return(b<=e && i>=b && i<=e || (b>e && (i<=e || i>=b)) ? +1:-1);
}


#endif
