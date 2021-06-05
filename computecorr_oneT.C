#include<iostream>
#include<vector>
#include<string>
#include<sstream>
#include<fstream>
#include<cstdlib>
using namespace std;

//****************************************************
//****    To compute correlation functions from exact data
//******************************************************
//usage:
//
//computecorr_oneT.exec filename Zfilename > outfile
//
//

#ifdef DEBUG
const bool TRACE=true;
#else
const bool TRACE=false;
#endif

#include "onetemperature.h"

void error(const char* p1,const char* p2)
{
  cerr << p1 << " " << p2 << endl;
  exit(1);
}


int main(int argc,char* argv[])
{
  if(TRACE) cout << argc << " " << argv[1] << endl;

  char* DATAFILENAME=argv[1];

  ifstream infile(DATAFILENAME);
  if(!infile) error("cannot open input file",DATAFILENAME);

  string Zfilename=string(argv[2]);

  CorrelationData corr(Zfilename);
  corr.Compute(infile);
  infile.close();
}
