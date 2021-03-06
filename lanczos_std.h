#ifndef LANCZOS_STD_H
#define LANCZOS_STD_H

//*************************************************************
//
// Header file for the standard implementation of Lanczos
// (as opposed to a parallell implementation)
// 
//*************************************************************

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

#include "banddiag.h"

// This is the standard implementation of Lanczos
class Lanczos_impl
{
public:
  //  Lanczos_impl(MyMatrix& H_in,int M_in); // initializes Lanczos with random vector
  Lanczos_impl(MyMatrix& H_in,int M_in,int basestatenr); // initializes Lanczos with a basestate 
  Lanczos_impl(MyMatrix& H_in,int M_in, vector<MyField>& vec_in); // initializes Lanczos using the input vector
  ~Lanczos_impl();
  void DoLanczos();
  MyField* GetInitVector(){return &phi[0];}
  MyField* GetPhi(const int i){return &phi[i*N];}
  void DiagonalizeLanczosMatrix();
  vector<double>& GetLanczosa(){return lanczos_a;}
  vector<double>& GetLanczosb(){return lanczos_b;}  // we start numbering b from b0,b1 etc.
  vector<double>& GetEigenvalues(){return LM->Eigenvalues();}
  double* GetEigenvector(const int i){return LM->Eigenvector(i);}
  int GetNeigenvalues(){return Niterations;}
  int GetNLanczosIterations(){return Niterations;}
  double GetNormofInitVector(){return initnorm;} 
  MyField* GetLanczosEigenvectorinStateSpaceBasis(const int n);
private:
  MyMatrix& H;
  int N; //matrix with N columns
  const int M; //# of Lanzcos iterations
  int Mstop;
  int Niterations;

  MyField* phi; 
  vector<double> lanczos_a; 
  vector<double> lanczos_b; 

  LanczosMatrix* LM;
  
  double initnorm;

  bool LanczosIteration(int m);
  void RandomInitialization();

  //  void ConstructGamma1(MyField* v1,MyField* v0,double& a0);
  //  void ConstructGammam(MyField* v_mp1,MyField* v_m,MyField* v_mm1,double& am,double& Nm);
  //  void Reorthogonalize(int m);
  void Reorthogonalize(const int N,MyField* z,const int m);
  void CheckOrthogonality(const int N,MyField* z,const int m);

};

#endif // LANCZOS_STD_H



