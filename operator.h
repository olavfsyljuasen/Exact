#ifndef OPERATOR_H
#define OPERATOR_H

//****************************************************
//
// Base class for operators. 
// Each operator should be declared in separate files
// and inherit properties from the class Operator
//
//****************************************************

//#include<fstream>
#include<iostream>
#include<complex>
#include<vector>
#include<string>
using namespace std;

//#include "inttostring.h"
#include "sparsematrix.h"
#include "global.h"

//#include "statespace.h"


class Operator
{
public:
  Operator(string name_in,int Nsectors_in);
  ~Operator();
  int GetOutSector(const int i);
  MyMatrix* GetSector(const int i);
  void Free();
  void Save(const int i);
  bool Load(const int i);
  bool Exists(const int i);
  virtual void GenerateSector(const int i){}
  string name;
  const int Nsectors;
  vector<int> Osector;
  MyMatrix* matrixptr;
};

#endif /* OPERATOR_H */



