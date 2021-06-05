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
//computecorr.exec beta wmin wmax Nbins filename > outfile
//
//

const bool TRACE=false;

#include "alltemperatures.h"

void error(const char* p1,const char* p2)
{
  cerr << p1 << " " << p2 << endl;
  exit(1);
}


int main(int argc,char* argv[])
{
  if(TRACE) cout << argc << " " << argv[1] << endl;

  // smoothing broadness
  double beta=1; // default
  double wmin=-1.; 
  double wmax= 8.; 
  int    Nbins=100; 
  double sigma=0.3;

  string a1str=argv[1]; istringstream a1sst(a1str); a1sst >> beta;
  string a2str=argv[2]; istringstream a2sst(a2str); a2sst >> wmin;
  string a3str=argv[3]; istringstream a3sst(a3str); a3sst >> wmax;
  string a4str=argv[4]; istringstream a4sst(a4str); a4sst >> Nbins;
  string a5str=argv[5]; istringstream a5sst(a5str); a5sst >> sigma;

  char* DATAFILENAME=argv[6];

  ifstream infile(DATAFILENAME);
  if(!infile) error("cannot open input file",DATAFILENAME);

  FrequencyBins fb(Nbins,wmin,wmax,sigma);

  CorrelationData corr(beta,fb);
  corr.Compute(infile);
  infile.close();
}
