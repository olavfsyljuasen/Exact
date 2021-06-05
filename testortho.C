#include <iostream>
#include <complex>
using namespace std;

#include "lowlevelroutines.h"

main()
{
  int N=10;
  vector<complex<double> > v0(N); 
  vector<complex<double> > z(N); 

  for(int i=0; i<N; i++)
    {
      v0[i]=complex<double>(1.,0.);
      z[i]=complex<double>(0.,0.);
    }

  z[0]=complex<double>(10.,0.);

  complex<double> before=ScalarProduct(&N,&v0[0],&z[0]); // q=<z|phi_i>
  complex<double> negq=-before;
  AddToVector(&N,&z[0],&v0[0],&negq); // |z> = |z> -q*|phi_i>;
  complex<double> after=ScalarProduct(&N,&v0[0],&z[0]);
  cout << "<z|v0>=" << before << " negq:" << negq << " <z|v1>=" << after << endl;

  for(int i=0; i<N; i++)
    {
      cout << "V0[" << i << "]= " << v0[i] << endl;
    }

  for(int i=0; i<N; i++)
    {
      cout << "z[" << i << "]= " << z[i] << endl;
    }

}
