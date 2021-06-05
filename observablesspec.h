#ifndef OBSERVABLESSPEC_H
#define OBSERVABLESSPEC_H

#include<fstream>
#include<iostream>
#include<string>
#include<vector>
using namespace std;

const string FILENAME="obs_spec.dat";

enum specid{sBETA,sESUB,sNWBINS,sWMIN,sWMAX,sSIGMA};
const int Nspecs=6;

class ObservablesSpec
{
  friend ostream& operator<<(ostream& os,ObservablesSpec& r)
  {
    os << " Beta: "      << r.GetBETA() 
       << " Esub: "   << r.GetESUB() 
       << " Nwbins: " << r.GetNWBINS() 
       << " Nwmin: "  << r.GetWMIN() 
       << " Nwmax: "  << r.GetWMAX() 
       << " Smooth: " << r.GetSIGMA() << endl; 
    return os;
  }
  
 public:
  ObservablesSpec(double* specin);
  ObservablesSpec(void);
  ~ObservablesSpec(){if(TRACE) cout << "removing observablesspec\n";}
  double GetBETA(){return specs[sBETA];}
  double GetESUB(){return specs[sESUB];}
  double GetNWBINS(){return specs[sNWBINS];}
  double GetWMIN(){return specs[sWMIN];}
  double GetWMAX(){return specs[sWMAX];}
  double GetSIGMA(){return specs[sSIGMA];}
 private:
  string file;
  vector<double> specs; 
  void SetSpec(double* specin){for(int i=0; i<Nspecs; i++){specs[i]=specin[i];}}  
};


/*
ObservablesSpec::ObservablesSpec(double* specin):file(FILENAME),specs(Nspecs)
{
  ifstream infile(file.c_str());
  if(infile) 
    { // file exists read it
      for(int i=0; i<Nspecs; i++){infile >> specs[i];}
    }
  else
    {
      SetSpec(specin);
      // file does not exist, write it:
      ofstream outfile(file.c_str());
      for(int i=0; i<Nspecs; i++){outfile << specs[i] << " ";}
      outfile << endl;
      outfile.close();
    }
  infile.close();
}


ObservablesSpec::ObservablesSpec(void):file(FILENAME),specs(Nspecs)
{
  ifstream infile(file.c_str());
  if(infile) 
    { // file exists read it
      for(int i=0; i<Nspecs; i++){infile >> specs[i];}
    }
  else
    {
      cout << "Error no " << FILENAME << " exists!, cannot make observable specs" << endl;
      exit(1);
    }
  infile.close();
}


*/

#endif // OBSERVABLESSPEC_H
