#ifndef RNDDEF_H
#define RNDDEF_H

// This file decides which rnadom number gneerators to use
// it consist of a name of a generator +  aheader file.

// the mathpack routines random number generators:
//#define RANDOMNUMBERGENERATOR Ran013 // for a list see rnddef_mathpack.h
//#define RANDOMNUMBERGENERATOR Ran19937 // for a list of others see rnddef.h
//#include "rnddef_Mathpack.h" // the random number generator RAN

// The Mersenne-Twister random number routine:
#define RANDOMNUMBERGENERATOR MTRand
#include "rnddef_MersenneTwister.h"

#endif

