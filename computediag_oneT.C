#include<iostream>
#include<vector>
#include<string>
#include<fstream>
#include<cstdlib>
using namespace std;

const bool TRACE=false;

#include "onetemperature.h"


void error(const char* p1,const char* p2)
{
  cerr << p1 << " " << p2 << endl;
  exit(1);
}


int main(int argc,char* argv[])
{

  // les inn datafil
  ifstream infile(argv[1]);
  if(!infile) error("cannot open input file",argv[1]);

  string Zfilename=string(argv[2]);

  OneTemperature onet(Zfilename);
  onet.FindTdependence(infile);
  infile.close();
}
