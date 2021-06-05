#ifndef OBSERVABLESSPEC_C
#define OBSERVABLESSPEC_C

#ifdef DEBUG
const bool WTRACE=true;
#else
const bool WTRACE=false;
#endif


#ifdef DEBUG2
const bool TRACE=true;
#else
const bool TRACE=false;
#endif

#include "stdlib.h"
#include "observablesspec.h"


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




#endif // OBSERVABLESSPEC_C
