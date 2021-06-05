#ifndef FULLDIAGONALIZATION_H
#define FULLDIAGONALIZATION_H


#include<iostream>
#include<vector>
#include<complex>
using namespace std;

#include "sparsematrix.h"
#include "global.h"

class FullDiagonalization
{
 public:
  FullDiagonalization(MyMatrix& H);
  vector<double>& GetEigenvalues();
  MyField* GetEigenvector(const int i);
  int GetNeigenvalues();
 private:
  char JOBS;
  char UPLO; // upper triangular matrix is stored after diag.
  int N;      // size of matrix
  int LDA; 
  vector<MyField> A; // the input matrix
  vector<MyField> B; // the input matrix
  vector<double> W; // will contain the eigenvalues
  int LWORK;
  vector<MyField> WORK;
  vector<double> RWORK;
  int INFO;
};



#endif  //FULLDIAGONALIZATION_H
