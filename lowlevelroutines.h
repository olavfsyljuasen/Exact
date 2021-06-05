#ifndef LOWLEVELROUTINES_H
#define LOWLEVELROUTINES_H

#include<complex>
#include<vector>
using namespace std;

double ScalarProduct(int* N,double* v1,double* v2);
double ScalarProduct(vector<double>& v1,vector<double>& v2);

complex<double> ScalarProduct(int* N,complex<double>* v1,complex<double>* v2);
complex<double> ScalarProduct(vector<complex<double> >& v1,vector<complex<double> >& v2);

double GetNorm(int* N,complex<double>* v);

double Normalize(int* N,complex<double>* v);
double Normalize(vector<complex<double> >& v);

void AddToVector(int* N,complex<double>* v1,complex<double>* v0,complex<double>* a);
void AddToVector(int* N,complex<double>* v1,complex<double>* v0,double* a);

void ScaleVector(int* N,complex<double>* v,complex<double>* factor);
void ScaleVector(int* N,complex<double>* v,double* factor);

#endif //LOWLEVELROUTINES_H
