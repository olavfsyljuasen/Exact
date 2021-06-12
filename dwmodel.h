#ifndef DWMODEL_H
#define DWMODEL_H

using namespace std;

//const bool TRACE=false;

enum parameterlist{LINEID,JX,JY,JZ,J2X,J2Y,J2Z,HX,HZ,C,NSITES,NDW,Q,SECTOR,MLANCZOS,SAMPLEFRAC,BETA,ESUB,NWBINS,WMIN,WMAX,SIGMA};
const int NPARAMS=22;

#include "inttostring.h"
#include "observables.h"
#include "statespace.h"
#include "xyzhamiltonian1d.h"
#include "spluss.h"
#include "sx.h"
#include "sy.h"
#include "sz.h"
#include "projectionoperator.h"
#include "hash.h"

class DWmodel
{
 public:
  DWmodel(double* par_in);
  ~DWmodel();
  int Nsectors(){return Nsec;}
  MyMatrix* GetHamiltonianSector(const int s){return Hop.GetSector(s);}
  void ReleaseHamiltonian(){Hop.Free();} // free memory
  int GetOutSector(const int s){return Aop.GetOutSector(s);}
  StateSpace& GetStateSpace(){return myspace;}
  vector<SingleOperatorObservable*>& GetSingleOperatorObservables(){return operatorlist;}
#ifdef CORRELATIONFUNCTION
  CorrelationFunctionObservable* GetCorrelationFunctionObservable(){return cfptr;}
#endif
 private:
  double* par;
  string hashkey;
  const int N;
  StateSpace myspace;
  const int Nsec;
  XYZhamiltonian1d Hop;
#ifdef SXSX
  Sx Aop; // <SxSx> corr. function
#elif defined SYSY
  Sy Aop; // <SySy> corr. function
#elif defined SZSZ
  Sz Aop; // <SzSz> correlation function operator
#else
  Spluss Aop; // <S-S+> default;
#endif
  Sz Szop;
#ifdef CALCULATE_PROJECTIONOPS
  Pop Pu1;
  Pop Pu2;
  Pop Pu3;

  Pop Pd1;
  Pop Pd2;
  Pop Pd3;

  Pop Pud;
  Pop Pdu;

  Pop Puud;
  Pop Pudd;
  Pop Pudu;
  Pop Pdud;

  Pop Puudd;

  Pop Pud1u;
  Pop Pud2u;
  Pop Pud3u;
  Pop Pud4u;
  Pop Pud5u;
  Pop Pud6u;
  Pop Pud7u;
  Pop Pud8u;
  Pop Pud9u;

  Pop Puud1u;
  Pop Puud2u;
  Pop Puud3u;
  Pop Puud4u;
  Pop Puud5u;
  Pop Puud6u;
  Pop Puud7u;
  Pop Puud8u;
  Pop Puud9u;

  Pop Puud1uu;
  Pop Puud2uu;
  Pop Puud3uu;
  Pop Puud4uu;
  Pop Puud5uu;
  Pop Puud6uu;
  Pop Puud7uu;
  Pop Puud8uu;
  Pop Puud9uu;
#endif
  vector<SingleOperatorObservable*> operatorlist;
#ifdef CORRELATIONFUNCTION
  CorrelationFunctionObservable* cfptr;
#endif
};




DWmodel::DWmodel(double* par_in):
par(par_in),hashkey(int2string(myhash(par,12*sizeof(double)))),
  N(par[NSITES]),myspace(N,par[NDW]),Nsec(myspace.Nsectors()),
  Hop("op_H"+hashkey,par[JX],par[JY],par[JZ],par[J2X],par[J2Y],par[J2Z],par[HX],par[HZ],par[C],myspace),
  Aop("op_Sx"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),myspace),
  Szop("op_Sz"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),myspace)
#ifdef CALCULATE_PROJECTIONOPS
  ,operatorlist(42),
  Pu1(   "op_Pu1" +int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),   1,1,myspace), // u
  Pu2(   "op_Pu2" +int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),   3,2,myspace), // uu
  Pu3(   "op_Pu3" +int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),   7,3,myspace), // uuu

  Pd1(   "op_Pd1" +int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),   0,1,myspace), // d
  Pd2(   "op_Pd2" +int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),   0,2,myspace), // dd
  Pd3(   "op_Pd3" +int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),   0,3,myspace), // ddd

  Pud(  "op_Pud"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),      2,2,myspace), // ud 
  Pdu(  "op_Pdu"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),      1,2,myspace), // du 

  Puud( "op_Puud"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),    6,3,myspace), // uud 
  Pudd( "op_Pudd"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),    4,3,myspace), // udd 
  Pudu( "op_Pudu"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),    5,3,myspace), // udu 
  Pdud( "op_Pdud"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),    2,3,myspace), // dud 

  Puudd( "op_Puudd"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]), 12,4,myspace), // uudd 

  Pud1u("op_Pud1u"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),   5,3,myspace), // udu 
  Pud2u("op_Pud2u"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),   9,4,myspace), // uddu
  Pud3u("op_Pud3u"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),  17,5,myspace), // udddu
  Pud4u("op_Pud4u"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),  33,6,myspace), // uddddu
  Pud5u("op_Pud5u"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),  65,7,myspace), // udddddu
  Pud6u("op_Pud6u"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]), 129,8,myspace), // uddddddu
  Pud7u("op_Pud7u"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]), 257,9,myspace), // udddddddu
  Pud8u("op_Pud8u"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]), 513,10,myspace),// uddddddddu
  Pud9u("op_Pud9u"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),1025,11,myspace), // udddddddddu

  Puud1u("op_Puud1u"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),  13,4,myspace), // uudu 
  Puud2u("op_Puud2u"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),  25,5,myspace), // uuddu
  Puud3u("op_Puud3u"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),  49,6,myspace), // uudddu
  Puud4u("op_Puud4u"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),  97,7,myspace), // uuddddu
  Puud5u("op_Puud5u"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]), 193,8,myspace), // uudddddu
  Puud6u("op_Puud6u"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]), 385,9,myspace), // uuddddddu
  Puud7u("op_Puud7u"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]), 769,10,myspace), //uudddddddu
  Puud8u("op_Puud8u"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),1537,11,myspace),// uuddddddddu
  Puud9u("op_Puud9u"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),3073,12,myspace), // uudddddddddu

  Puud1uu("op_Puud1uu"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),  27,5,myspace), // uuduu 
  Puud2uu("op_Puud2uu"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),  51,6,myspace), // uudduu
  Puud3uu("op_Puud3uu"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),  99,7,myspace), // uuddduu
  Puud4uu("op_Puud4uu"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]), 195,8,myspace), // uudddduu
  Puud5uu("op_Puud5uu"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]), 387,9,myspace), // uuddddduu
  Puud6uu("op_Puud6uu"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]), 771,10,myspace), // uudddddduu
  Puud7uu("op_Puud7uu"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),1539,11,myspace), //uuddddddduu
  Puud8uu("op_Puud8uu"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),3075,12,myspace),// uudddddddduu
  Puud9uu("op_Puud9uu"+int2string(static_cast<int>(par[Q]))+"_"+hashkey,static_cast<int>(par[Q]),6147,13,myspace) // uuddddddddduu
#else
  ,operatorlist(2)
#endif
{
  if(TRACE) cout << "Constructing DWmodel " << hashkey << endl;
  logfile << "Hashkey for these parameters is: " << hashkey << endl;

  operatorlist[ 0]=new SingleOperatorObservable(Aop,"sx.dat");
  operatorlist[ 1]=new SingleOperatorObservable(Szop,"sz.dat");
#ifdef CALCULATE_PROJECTIONOPS
  operatorlist[ 2]=new SingleOperatorObservable(Pu1,"pu1.dat");
  operatorlist[ 3]=new SingleOperatorObservable(Pu2,"pu2.dat");
  operatorlist[ 4]=new SingleOperatorObservable(Pu3,"pu3.dat");

  operatorlist[ 5]=new SingleOperatorObservable(Pd1,"pd1.dat");
  operatorlist[ 6]=new SingleOperatorObservable(Pd2,"pd2.dat");
  operatorlist[ 7]=new SingleOperatorObservable(Pd3,"pd3.dat");

  operatorlist[ 8]=new SingleOperatorObservable(Pud,"pud.dat");
  operatorlist[ 9]=new SingleOperatorObservable(Pdu,"pdu.dat");

  operatorlist[10]=new SingleOperatorObservable(Puud,"puud.dat");
  operatorlist[11]=new SingleOperatorObservable(Pudd,"pudd.dat");
  operatorlist[12]=new SingleOperatorObservable(Pudu,"pudu.dat");
  operatorlist[13]=new SingleOperatorObservable(Pdud,"pdud.dat");

  operatorlist[14]=new SingleOperatorObservable(Puudd,"puudd.dat");

  operatorlist[15]=new SingleOperatorObservable(Pud1u,"pud1u.dat");
  operatorlist[16]=new SingleOperatorObservable(Pud2u,"pud2u.dat");
  operatorlist[17]=new SingleOperatorObservable(Pud3u,"pud3u.dat");
  operatorlist[18]=new SingleOperatorObservable(Pud4u,"pud4u.dat");
  operatorlist[19]=new SingleOperatorObservable(Pud5u,"pud5u.dat");
  operatorlist[20]=new SingleOperatorObservable(Pud6u,"pud6u.dat");
  operatorlist[21]=new SingleOperatorObservable(Pud7u,"pud7u.dat");
  operatorlist[22]=new SingleOperatorObservable(Pud8u,"pud8u.dat");
  operatorlist[23]=new SingleOperatorObservable(Pud9u,"pud9u.dat");

  operatorlist[24]=new SingleOperatorObservable(Puud1u,"puud1u.dat");
  operatorlist[25]=new SingleOperatorObservable(Puud2u,"puud2u.dat");
  operatorlist[26]=new SingleOperatorObservable(Puud3u,"puud3u.dat");
  operatorlist[27]=new SingleOperatorObservable(Puud4u,"puud4u.dat");
  operatorlist[28]=new SingleOperatorObservable(Puud5u,"puud5u.dat");
  operatorlist[29]=new SingleOperatorObservable(Puud6u,"puud6u.dat");
  operatorlist[30]=new SingleOperatorObservable(Puud7u,"puud7u.dat");
  operatorlist[31]=new SingleOperatorObservable(Puud8u,"puud8u.dat");
  operatorlist[32]=new SingleOperatorObservable(Puud9u,"puud9u.dat");

  operatorlist[33]=new SingleOperatorObservable(Puud1uu,"puud1uu.dat");
  operatorlist[34]=new SingleOperatorObservable(Puud2uu,"puud2uu.dat");
  operatorlist[35]=new SingleOperatorObservable(Puud3uu,"puud3uu.dat");
  operatorlist[36]=new SingleOperatorObservable(Puud4uu,"puud4uu.dat");
  operatorlist[37]=new SingleOperatorObservable(Puud5uu,"puud5uu.dat");
  operatorlist[38]=new SingleOperatorObservable(Puud6uu,"puud6uu.dat");
  operatorlist[39]=new SingleOperatorObservable(Puud7uu,"puud7uu.dat");
  operatorlist[40]=new SingleOperatorObservable(Puud8uu,"puud8uu.dat");
  operatorlist[41]=new SingleOperatorObservable(Puud9uu,"puud9uu.dat");
#endif

#ifdef CORRELATIONFUNCTION
  cfptr = new CorrelationFunctionObservable(Aop,"corr.dat");
  logfile << "Correlation function is " <<
#ifdef SXSX    
    "S^x(" << static_cast<int>(par[Q]) << ")S^x(" << static_cast<int>(par[Q]) << ")>" << endl;
#elif defined SYSY
    "S^y(" << static_cast<int>(par[Q]) << ")S^y(" << static_cast<int>(par[Q]) << ")>" << endl;
#elif defined SXSX
    "S^z(" << static_cast<int>(par[Q]) << ")S^z(" << static_cast<int>(par[Q]) << ")>" << endl;
#else
    "S^-(" << static_cast<int>(par[Q]) << ")S^+(" << static_cast<int>(par[Q]) << ")>" << endl;
#endif
#endif

  if(TRACE) cout << "Done with constructing DWmodel" << endl;
}

DWmodel::~DWmodel()
{
  if(TRACE) cout << "Destroying class DWmodel" << endl;
  for(int i=0; i<operatorlist.size(); i++){delete operatorlist[i];}
#ifdef CORRELATIONFUNCTION
  delete cfptr;
#endif
}


#endif // DWMODEL_H
