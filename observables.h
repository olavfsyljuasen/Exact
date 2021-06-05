#ifndef OBSERVABLES_H
#define OBSERVABLES_H

const double MAXMINENERGY=100000.;

#include "operator.h"

#ifdef FULLDIAGONALIZATION
#include "fulldiagonalization.h"
#else
#include "lanczos.h"
#endif

/*
#include "alltemperatures.h"


#include "lowlevelroutines.h"
*/

class BasicObservables_impl;
class BasicObservables
{
 public:
#ifdef FULLDIAGONALIZATION
  BasicObservables(int,int,FullDiagonalization&,double);
#else
  BasicObservables(int,int,Lanczos&,double);
#endif
  ~BasicObservables();
 private:
  BasicObservables_impl* impl;
};


class SingleOperatorObservable_impl;
class SingleOperatorObservable
{
 public:
  SingleOperatorObservable(Operator& Aop_in,string filename_in);
  ~SingleOperatorObservable();
#ifdef FULLDIAGONALIZATION
  void Calculate(int sector,int nstates,FullDiagonalization& mydiag);
#else
  void Calculate(int sector,int nstates,Lanczos& mydiag);
#endif
  Operator& GetOperator();
 private:
  SingleOperatorObservable_impl* impl;
};


//**********************************************************************************
#ifdef CORRELATIONFUNCTION

class CorrelationFunctionObservable_impl;
class CorrelationFunctionObservable
{
 public:
  CorrelationFunctionObservable(Operator& Aop_in,string filename_in);
  ~CorrelationFunctionObservable();
#ifdef FULLDIAGONALIZATION
  void Calculate(int sector,int nstates,FullDiagonalization& mydiag1,FullDiagonalization& mydiag2);
#else
  void Calculate(int sector,int nstates,Lanczos& mydiag1,Lanczos& mydiag2);
#endif
  bool CalculateInitVector(int is1,MyField* vinit,MyField* avinit);
  Operator& GetOperator();
 private:
  CorrelationFunctionObservable_impl* impl;
};

#endif //CORRELATIONFUNCTION
  
#endif //OBSERVABLES_H
    
