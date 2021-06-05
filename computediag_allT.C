#include<iostream>
#include<vector>
#include<string>
#include<fstream>
#include<cstdlib>
using namespace std;

const bool TRACE=false;

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
  storedreal beta;
  vector<storedreal> betalist;
  while(betafile)
    { 
      betafile >> beta;
      if(betafile) betalist.push_back(beta);
    } 
  betafile.close();

  // les inn datafil
  ifstream infile(argv[1]);
  if(!infile) error("cannot open input file",argv[1]);

  string Zfilename=string(argv[2]);

  AllTemperatures allt(betalist,Zfilename);
  allt.FindTdependence(infile);
  infile.close();
}
