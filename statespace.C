#include<iostream>
#include <fstream>
using namespace std;

const bool TRACE=true;
//const bool WRITEOUTALLSTATES=true;

ofstream logfile("out.log");

#include "statespace.h"

main()
{
  const int N=8;
  const int Ndw=8;
  StateSpace myspace(N,Ndw);
  
  int s=myspace.Nsectors();
  for(int i=0; i<s; i++)
    {
      cout << "sector: " << i << endl;
      cout << *myspace.sectors[i] << endl;
    }


}
