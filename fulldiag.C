#include<iostream>
#include<vector>
using namespace std;

#include "fulldiagonalization.h"

#include "sparsematrix.h"

int main()
{
  const int N=3;
  COOmatrix M(N,N);
  M(0,0)=1;
  M(0,1)=complex<double>(0.,0.5);
  M(1,0)=complex<double>(0.,-0.5);
  M(1,1)=1;

  CSRmatrix A(M);

  for(int j=0; j <N; j++)
    {
      for(int i=0; i<N; i++)
	cout << A(j,i) << " ";
      cout << endl;
    }



  FullDiagonalization fd(A);


  for(int g=0; g<N; g++)
    {
      MyField* v=fd.GetEigenvector(g);
      vector<double>& evals=fd.GetEigenvalues();
      cout << "Eigenvector: " << g << " Eigenvalue: " << evals[g] << endl;
      for(int i=0; i<N; i++) cout << v[i] << " ";
      cout << endl;
    }


}
