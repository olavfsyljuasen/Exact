#ifndef INTTOSTRING_C
#define INTTOSTRING_C

#include <string>
using namespace std;

#include "inttostring.h"

// a routine that converts integers to strings
const int MAXDIGITS = 20;
const int ASCII0 = 48;
string int2string(int a)
{
  int digit[MAXDIGITS];
  int d=0;
  string s("");
  if (a == 0){ s+=ASCII0; return s;}
  while( a > 0){ digit[d++] = a-10*(a/10); a /=10 ;}
  for(int i=d-1; i>=0; i--) s+= digit[i]+ASCII0;
  return s;
}

string int2string(unsigned int a)
{
  int digit[MAXDIGITS];
  int d=0;
  string s("");
  if (a == 0){ s+=ASCII0; return s;}
  while( a > 0){ digit[d++] = a-10*(a/10); a /=10 ;}
  for(int i=d-1; i>=0; i--) s+= digit[i]+ASCII0;
  return s;
}


// this routine also handles leading zeros
string int2string(int a,int nsiffer)
{
  int digit[MAXDIGITS];
  string s("");
  for(int d=0; d<MAXDIGITS; d++){ digit[d] = a-10*(a/10); a /=10 ;}
  for(int i=nsiffer-1; i>=0; i--) s+= digit[i]+ASCII0;
  return s;
}

#endif // INTTOSTRING_C
