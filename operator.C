#ifndef OPERATOR_C
#define OPERATOR_C

//****************************************************
//
// Base class for operators. 
// Each operator should be declared in separate files
// and inherit properties from the class Operator
//
//****************************************************

#include<fstream>
#include<iostream>
#include<complex>
#include<vector>
#include<string>
using namespace std;

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



#include "sparsematrix.h"
#include "operator.h"
#include "inttostring.h"
#include "global.h"

#ifdef ELLMATRIX
const string extension=".ELL";
#else
const string extension=".CSR";
#endif

Operator::Operator(string name_in,int Nsectors_in):name(name_in),Nsectors(Nsectors_in),Osector(Nsectors_in),matrixptr(NULL){}

Operator::~Operator()
{
  if(TRACE) cout << "Destroying base class Operator" << endl;
  delete matrixptr;
}

int Operator::GetOutSector(const int i){return Osector[i];}

MyMatrix* Operator::GetSector(const int i)
{
  if(!Exists(i)) GenerateSector(i); 
  Load(i); 
  return matrixptr;
}

void Operator::Free(){delete matrixptr; matrixptr=NULL;}


void Operator::Save(const int i)
{
  string filename=name+"_"+int2string(i)+extension;
  if(WTRACE) cout << "Saving " << filename << endl;
  ofstream ofile(filename.c_str());
  matrixptr->Write(ofile);
  ofile.close();
}

bool Operator::Load(const int i)
{
  string filename=name+"_"+int2string(i)+extension;
  if(WTRACE) cout << "Loading " << filename << endl;
  ifstream ifile(filename.c_str());
  delete matrixptr;
  matrixptr=new MyMatrix(ifile);
  if(matrixptr->Nelements()==0){ delete matrixptr; matrixptr=0; return false;} // check if file is empty.
  return true;
}

bool Operator::Exists(const int i)
{
  string filename=name+"_"+int2string(i)+extension;
  ifstream ifile(filename.c_str());
  return (ifile ? true : false);
}

#endif /* OPERATOR_C */



