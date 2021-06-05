#ifndef OBSERVABLES_STD_H
#define OBSERVABLES_STD_H

#include<fstream>
using namespace std;


#include "operator.h"

#ifdef FULLDIAGONALIZATION
#include "fulldiagonalization.h"
#else
#include "lanczos.h"
#endif

//#include "alltemperatures.h"
//#include "lowlevelroutines.h"

class BasicObservables_impl
{
 public:
#ifdef FULLDIAGONALIZATION
  BasicObservables_impl(int,int,FullDiagonalization&,double);
#else
  BasicObservables_impl(int,int,Lanczos&,double);
#endif
};


class SingleOperatorObservable_impl
{
 public:
  SingleOperatorObservable_impl(Operator& Aop_in,string filename_in);
#ifdef FULLDIAGONALIZATION
  void Calculate(int sector,int nstates,FullDiagonalization& mydiag);
#else
  void Calculate(int sector,int nstates,Lanczos& mydiag);
#endif
  Operator& GetOperator();
 private:
  Operator& Aop;
  string filename;
};



#ifdef CORRELATIONFUNCTION

class CorrelationFunctionObservable_impl
{
 public:
  CorrelationFunctionObservable_impl(Operator& Aop_in,string filename_in);
#ifdef FULLDIAGONALIZATION
  void Calculate(int sector,int nstates,FullDiagonalization& mydiag1,FullDiagonalization& mydiag2);
#else
  void Calculate(int sector,int nstates,Lanczos& mydiag1,Lanczos& mydiag2);
#endif
  bool CalculateInitVector(int is1,MyField* vinit,MyField* avinit);
  Operator& GetOperator();
 private:
  Operator& Aop;
  string filename;
};

  
#endif // CORRELATIONFUNCTION
  
  
#endif //OBSERVABLES_STD_H
    
