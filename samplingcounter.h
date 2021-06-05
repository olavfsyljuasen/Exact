#ifndef SAMPLINGCOUNTER_H
#define SAMPLINGCOUNTER_H

#include<iostream>
#include<fstream>
#include<string>
#include<cstdio>
using namespace std;

#include "statespace.h"

class SamplingCounter
{
  friend ostream& operator<<(ostream& os,SamplingCounter& t)
  {for(int i=0; i<t.Nsectors; i++){os << i << " " << t.counter[i] <<endl;} return os;}
 public:
  SamplingCounter(StateSpace& s,double fraction_in,double sector_in);
  ~SamplingCounter();
  void Save();
  bool Load();
  bool MoreSectorsToSample(int&,int&);
  void Decrement(const int);
  string filename;
  int Nsectors;
  vector<int> counter;
};

SamplingCounter::SamplingCounter(StateSpace& s,double fraction_in,double sector_in):filename("sectorstosample"),Nsectors(s.Nsectors()),counter(Nsectors)
{
  bool isloaded=Load();
  if(!isloaded)
    { // generate the sampling list 

      int xsector=static_cast<int>(sector_in); // if -1 do all sectors
      
      for(int i=0; i<Nsectors; i++)
	{
	  if(xsector == -1 || i == xsector)
	    {      
#ifdef FULLDIAGONALIZATION
	      counter[i]=1;
#else
	      int Nstates=s.sectors[i]->Nstates;
#ifdef USEBASISSTATESASINITFORSMALLSECTORS
	      counter[i]= (Nstates < SMALLSECTORLIMIT ? Nstates: static_cast<int>(Nstates*fraction_in));
#else
	      counter[i]=static_cast<int>(Nstates*fraction_in);
	      if(counter[i]==0) counter[i]=1; // do not accept 0 as smallest value
#endif
#endif
	    }
	  else
	    counter[i]=0;
	}

      Save();
    }
}

SamplingCounter::~SamplingCounter()
{
  bool isdone=true;
  for(int i=0; i<Nsectors; i++){if(counter[i]!=0){isdone=false; break;}}
  if(isdone){ remove(filename.c_str());} // rm file
}

bool SamplingCounter::MoreSectorsToSample(int& nts,int& ctr)
{
  bool isloaded=Load();
  if(!isloaded){ cout << "Cannot Load SamplingCounter" << endl;}
  int sec=0;
  while( sec<Nsectors)
    {
      while(counter[sec]>0)
	{
	  //	  counter[sec]--; 
	  ctr=counter[sec]-1; // the -1 is to ensure that counter takes the states in range
	  nts=sec; 
	  //	  Save();
	  return true;
	}
      sec++;
    } 
  return false;
}


void SamplingCounter::Decrement(const int sec){counter[sec]--; Save();} 


bool SamplingCounter::Load()
{
  ifstream ifile(filename.c_str());
  if(!ifile){return false;}
  for(int i=0; i<Nsectors; i++)
    {
      int k;
      ifile >> k; 
      ifile >> counter[k];
    }
  return true;
}


void SamplingCounter::Save()
{
  ofstream ofile(filename.c_str());
  for(int i=0; i<Nsectors; i++) 
    ofile << i << " " << counter[i] << endl;
  ofile.close();
}

#endif //SAMPLINGCOUNTER_H
