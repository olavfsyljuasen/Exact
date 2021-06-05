#include<iostream>
#include<vector>
using namespace std;

#include "lowlevelroutines.h"

int main()
{
  int N=3;
  vector<double> a(N);
  vector<double> b(N);
  a[0]=1.;   a[1]=2.;   a[2]=2.;
  b[0]=1.;   b[1]=0.;   b[2]=4.;

  double res=ScalarProduct(&N,&a[0],&b[0]);

  cout << "res: " << res << endl;

  vector<complex<double> > c(N);
  vector<complex<double> > d(N);
  c[0]=complex<double>(0,1);   c[1]=complex<double>(1,0);   c[2]=complex<double>(0,1);
  d[0]=complex<double>(0,-1);   d[1]=complex<double>(1,0);   d[2]=complex<double>(0,1);

  complex<double> res2=ScalarProduct(&N,&c[0],&d[0]);
  cout << "res2: " << res2 << endl;

  double norm= Normalize(&N,&c[0]);
  cout << "Normalize c: " << norm << endl;
  
  cout << "c: " << endl;

  for(int i=0; i<N; i++) cout << c[i] << " ";
  cout << endl;

  double norm2= Normalize(&N,&c[0]);
  cout << "Normalize c: " << norm2 << endl;

}

