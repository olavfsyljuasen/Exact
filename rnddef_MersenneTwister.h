#ifndef RNDDEF_MERSENNETWISTER_H
#define RNDDEF_MERSENNETWISTER_H

#include<fstream>
#include<string>
#include "Mersenne/MersenneTwister.h"

// the class name is MTRand

const string RANDOMSEED="randomseed";

unsigned long int GetRandomSeed(){
  ifstream file(RANDOMSEED.c_str());
  unsigned long int seed;
  file >> seed; return seed;
}

class RandomNumberGenerator{
public:
  RandomNumberGenerator(unsigned long int seed=21312512): 
    generator(GetRandomSeed())
  {
    ofstream file(RANDOMSEED.c_str());
    file << GetNewSeed() << endl;
  }
  double operator()(){return generator.randExc();}
  double Get(){return generator.randExc();}
  double HighPrecision(){return generator.rand53();} // for better resolution
  double Ran(){return generator.randExc();}
  unsigned long int GetNewSeed(){return generator.randInt();}
 private:
  unsigned long seed;
  
  RANDOMNUMBERGENERATOR generator;
};

// declare the global random number generator
RandomNumberGenerator RAN(GetRandomSeed());



#endif // RNDDEF_MERSENNETWISTER_H
