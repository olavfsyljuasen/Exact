#include<iostream>

using namespace std;

#include "bitmanip.h"

int main()
{
  pstate b=(pstate1<<0);   

  cout << "b:" << b << ",bitstring:" << bitstring(b) << endl;

  pstate c=cyclicshiftleft(b,8);
  cout << "b:" << b << ",cyclicshiftleft: " << c << ",bitstring:" << bitstring(c) << endl;

  pstate d=cyclicshiftnleft(c,7,8);
  cout << "b:" << b << ",cyclicshift3left: " << d << ",bitstring:" << bitstring(d) << endl;



 
}


