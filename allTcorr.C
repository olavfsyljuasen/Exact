#include<iostream>
#include<vector>
#include<string>
#include<fstream>
#include<cstdlib>
using namespace std;

#include "alltemperatures.h"


void error(const char* p1,const char* p2)
{
  cerr << p1 << " " << p2 << endl;
  exit(1);
}


int main(int argc,char* argv[])
{

  // les inn betalist
  ifstream betafile("betalist");
  if(!betafile) error("cannot open input file","betalist");
  double beta;
  vector<double> betalist;
  while(betafile)
    { 
      betafile >> beta;
      if(betafile) betalist.push_back(beta);
    } 
  betafile.close();

  int Nbins=100;
  double wmin=0.;
  double wmax=10.;

  FrequencyBins fb(Nbins,wmin,wmax);


  // les inn datafil
  ifstream infile(argv[1]);
  if(!infile) error("cannot open input file",argv[1]);

  AllCorrTemperatures allt(betalist);
  allt.FindTdependence(infile);
  infile.close();
}
