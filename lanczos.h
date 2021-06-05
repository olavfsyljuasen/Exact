#ifndef LANCZOS_H
#define LANCZOS_H

//******************************************************************
//
// This inlcude file forms the interface to the Lanczos routine
// Its implementation-specific details are in the class Lanczos_impl
// 
//******************************************************************


#include<iostream>
#include<vector>
#include<complex>
using namespace std;

#include "sparsematrix.h"
#include "global.h"

//--------------------------------------------------------------
// The Lanczos routine computes the Lanczos matrix
// Diagonalization of the Lanczosmatrix must be done separately
// in the routine LanczosDiag
//--------------------------------------------------------------

class Lanczos_impl;
class Lanczos
{
public:
  //  Lanczos(MyMatrix& H_in,int M_in); // initializes Lanczos with random vector
  Lanczos(MyMatrix& H_in,int M_in,int basestatenr); // initializes Lanczos with a basestate 
  Lanczos(MyMatrix& H_in,int M_in, vector<MyField>& vec_in); // initializes Lanczos using the input vector
  ~Lanczos(); 
  void DoLanczos();
  MyField* GetInitVector();
  MyField* GetPhi(const int i);
  void DiagonalizeLanczosMatrix();
  vector<double>& GetLanczosa();
  vector<double>& GetLanczosb();
  vector<double>& GetEigenvalues();
  double* GetEigenvector(const int i);
  int GetNeigenvalues();
  int GetNLanczosIterations();
  double GetNormofInitVector();
  MyField* GetLanczosEigenvectorinStateSpaceBasis(const int n);
private:
  Lanczos_impl* impl; // specific implementation of Lanczos
};

#endif // LANCZOS_H



