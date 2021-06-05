#include<iostream>
#include<complex>
using namespace std;

#include "cuda_global.h"

main()
{
  double a=1;
  double b=2;

  complex<double> c=complex<double>(a,b);

  cout << " c before: " << c << endl;


  ComplexSquare(&c);

  cout << " c " << c << endl;
}
  


