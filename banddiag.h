#ifndef BANDDIAG_H
#define BANDDIAG_H

#include<iostream>
#include<vector>
using namespace std;


class LanczosMatrix
{
public:
  LanczosMatrix(int N_in,vector<double>& diag_in,vector<double>& offd_in);
  void Diagonalize();
  double Eigenvalue(const int i){return diag[i];} 
  vector<double>& Eigenvalues(){return diag;}
  double* Eigenvector(const int i){return &evec[i*N];} 
private:
  int N;
  vector<double> diag;
  vector<double> offd;
  vector<double> evec; // contains eigenvectors on exit. 
};

#endif // BANDDIAG_H
