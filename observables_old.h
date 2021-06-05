#ifndef OBSERVABLES_H
#define OBSERVABLES_H

#include<fstream>
using namespace std;

const double MAXMINENERGY=100000.;

#include "alltemperatures.h"
#include "fulldiagonalization.h"
#include "operator.h"
#include "lowlevelroutines.h"

class BasicObservables
{
 public:
#ifdef FULLDIAGONALIZATION
  BasicObservables(int,int,FullDiagonalization&,double);
#else
  BasicObservables(int,int,Lanczos&,double);
#endif
};


#ifdef FULLDIAGONALIZATION
BasicObservables::BasicObservables(int sector,int nstates,FullDiagonalization& mydiag,double beta)
#else
BasicObservables::BasicObservables(int sector,int nstates,Lanczos& mydiag,double beta)
#endif
{
  double Z=0;
  double Energy=0.;
  double MinEnergy=MAXMINENERGY;
  int Neigenvalues=mydiag.GetNeigenvalues();

  vector<double>& eval=mydiag.GetEigenvalues();


  //  DataLine Zline(sector,Neigenvalues);
  DataLine Zline(sector,nstates);
  for(int i=0; i<Neigenvalues; i++)
    {
#ifdef FULLDIAGONALIZATION
      double factor=1.;
#else
      double* evec=mydiag.GetEigenvector(i);
      double factor=evec[0]*evec[0];
#endif
      if(eval[i] < MinEnergy){ MinEnergy=eval[i];}
      Zline.Insert(eval[i],factor);
    }
  
  ofstream Zfile("Z.dat",ios::app);
  Zline.Write(Zfile);
  Zfile.close();
  /*
  if(MinEnergy < -1) logfile << "WARNING: ";
  logfile << "Minimum energy found: " << MinEnergy << endl;
  */

  // Energy:
  //  DataLine Eline(sector,Neigenvalues);
  DataLine Eline(sector,nstates);
  for(int i=0; i<Neigenvalues; i++)
    {
      //      Eline.energy[i]=eval[i];

#ifdef FULLDIAGONALIZATION
      double factor=eval[i];
#else
      double* evec=mydiag.GetEigenvector(i);
      double factor=evec[0]*evec[0]*eval[i];
#endif
      Eline.Insert(eval[i],factor);
    }
  
  ofstream Efile("energy.dat",ios::app);
  Eline.Write(Efile);
  Efile.close();

  // Energy Squared:
  //DataLine Eline2(sector,Neigenvalues);
  DataLine Eline2(sector,nstates);
  for(int i=0; i<Neigenvalues; i++)
    {
      //      Eline2.energy[i]=eval[i];

#ifdef FULLDIAGONALIZATION
      double factor=eval[i]*eval[i];
#else
      double* evec=mydiag.GetEigenvector(i);
      double factor=evec[0]*evec[0]*eval[i]*eval[i];
#endif
      Eline2.Insert(eval[i],factor);
    }
  
  ofstream Efile2("energy2.dat",ios::app);
  Eline2.Write(Efile2);
  Efile2.close();
}


class SingleOperatorObservable
{
 public:
 SingleOperatorObservable(Operator& Aop_in,string filename_in):Aop(Aop_in),filename(filename_in){}
#ifdef FULLDIAGONALIZATION
  void Calculate(int sector,int nstates,FullDiagonalization& mydiag);
#else
  void Calculate(int sector,int nstates,Lanczos& mydiag);
#endif
 private:
  Operator& Aop;
  string filename;
};


#ifdef FULLDIAGONALIZATION
void SingleOperatorObservable::Calculate(int sector,int nstates,FullDiagonalization& mydiag)
#else
void SingleOperatorObservable::Calculate(int sector,int nstates,Lanczos& mydiag)
#endif
{
  if(TRACE) cout << "in SingleOperatorObservable Calculate " << filename << " sector " << sector << endl; 

  int Neigenvalues=mydiag.GetNeigenvalues();

  vector<double>& eval=mydiag.GetEigenvalues();

  if(Aop.GetOutSector(sector)!=sector)
    {
      if(TRACE) cout << "operator is off-diagonal, returning" << endl;
      return;
    }

  MyMatrix* Aptr=Aop.GetSector(sector);

  if(Aptr==0)
    {
      if(TRACE) cout << "operator action is 0, returning" << endl;
      return;
    }

  MyMatrix& A=*Aptr;
  // A:
  //
  //    DataLine Aline(sector,Neigenvalues);
  DataLine Aline(sector,nstates);

#ifdef FULLDIAGONALIZATION
  for(int i=0; i<Neigenvalues; i++)
    {
      //      Aline.energy[i]=eval[i];
      
      MyField* evec=mydiag.GetEigenvector(i);
      vector<MyField> ev(Neigenvalues);
      vector<MyField> evec_star=A*evec;
      double factor=ScalarProduct(&Neigenvalues,&evec_star[0],&evec[0]).real();
      Aline.Insert(eval[i],factor);
    }
#else
  vector<MyField> Aphi0=A*mydiag.GetPhi(0);
  int Nelements=Aphi0.size();
  for(int i=0; i<Neigenvalues; i++)
    {
      //      Aline.energy[i]=eval[i];
      double* evec=mydiag.GetEigenvector(i);
      double nui0=evec[0];
      /*
      complex<double> msum=complex<double>(0.,0.);
      for(int j=0; j<Neigenvalues; j++)
	{
	  vector<MyField> phij=mydiag.GetPhi(j);
	  double jthcomp=*(evec+j);
	  msum += jthcomp*ScalarProduct(phij,Aphi0);
	}
      */
      MyField* psi_i=mydiag.GetLanczosEigenvectorinStateSpaceBasis(i);
      MyField msum=ScalarProduct(&Nelements,psi_i,&Aphi0[0]);
	
      //      Aline.factor[i]=(nui0*msum).real(); // CHECK THIS
      double factor=(nui0*msum).real(); // CHECK THIS
      Aline.Insert(eval[i],factor);
    }
#endif // FULLDIAGONALIZATION  

  ofstream Afile(filename.c_str(),ios::app);
  Aline.Write(Afile);
  Afile.close();

  Aop.Free();
}


//**********************************************************************************
#ifdef CORRELATIONFUNCTION

class CorrelationFunctionObservable
{
 public:
 CorrelationFunctionObservable(Operator& Aop_in,string filename_in):Aop(Aop_in),filename(filename_in){}
#ifdef FULLDIAGONALIZATION
  void Calculate(int sector,int nstates,FullDiagonalization& mydiag1,FullDiagonalization& mydiag2);
#else
  void Calculate(int sector,int nstates,Lanczos& mydiag1,Lanczos& mydiag2);
#endif
  Operator& Aop;
 private:
  string filename;
};

#ifdef FULLDIAGONALIZATION
void CorrelationFunctionObservable::Calculate(int sector,int nstates,FullDiagonalization& mydiag1,FullDiagonalization& mydiag2)
{
  if(TRACE) cout << "in CorrelationFunctionObservable Calculate " << filename << " sector " << sector << " nstates:" << nstates << endl; 
  
  CorrelationDataLine Aline(sector,1);
  
  int Neigenvalues1=mydiag1.GetNeigenvalues();
  int Neigenvalues2=mydiag2.GetNeigenvalues();
  
  if(TRACE) cout << "Nevals1: " << Neigenvalues1 << " Nevals2: " << Neigenvalues2 << endl;
  
  MyMatrix* Aptr=Aop.GetSector(sector);
  
  if(Aptr!=0 && Neigenvalues1>0 && Neigenvalues2>0)
    {
      vector<double>& eval1=mydiag1.GetEigenvalues();
      vector<double>& eval2=mydiag2.GetEigenvalues();
      
      MyMatrix& A=*Aptr;
      int G=A.Nrows();
      
      for(int n=0; n<Neigenvalues1; n++)
	{
	  MyField* evec_n=mydiag1.GetEigenvector(n);
	  vector<MyField> Aevec_n=A*evec_n;
	  
	  for(int m=0; m<Neigenvalues2; m++)
	    {
	      MyField* evec_m=mydiag2.GetEigenvector(m);
	      MyField matrixelem=ScalarProduct(&G,evec_m,&Aevec_n[0]);
	      
	      Aline.Insert(eval1[n],eval2[m]-eval1[n],norm(matrixelem)); // store data abs^2
	    }
	}
    }
  
  ofstream Afile(filename.c_str(),ios::app);
  Aline.Write(Afile);
  Afile.close();
}
#else
void CorrelationFunctionObservable::Calculate(int sector,int nstates,Lanczos& mydiag1,Lanczos& mydiag2)
{
  if(TRACE) cout << "in CorrelationFunctionObservable Calculate Lanczos " << filename << " sector " << sector << " nstates: " << nstates << endl; 
  
  CorrelationDataLine Aline(sector,nstates);
  
  int Neigenvalues1=mydiag1.GetNeigenvalues();
  int Neigenvalues2=mydiag2.GetNeigenvalues();
  
  if(TRACE) cout << "Nevals1: " << Neigenvalues1 << " Nevals2: " << Neigenvalues2 << endl;
  
  MyMatrix* Aptr=Aop.GetSector(sector);
  
  if(TRACE) cout << "Aptr: " << Aptr << endl;

  if(Aptr!=0 && Neigenvalues1>0 && Neigenvalues2>0) // else just write out a zero dataline
    {

      vector<double>& eval1=mydiag1.GetEigenvalues();
      vector<double>& eval2=mydiag2.GetEigenvalues();

      MyMatrix& A=*Aptr;
      int G=A.Nrows();
      
      for(int n=0; n<Neigenvalues1; n++)
	{
	  double* evec1_n=mydiag1.GetEigenvector(n);
	  
	  if(TRACE){
	    cout << "n: " << n << " evec1_n: ";
	    for(int i=0; i<Neigenvalues1; i++){cout << evec1_n[i] << " ";}
	    cout << endl;
	  }
	  
	  for(int m=0; m<Neigenvalues2; m++)
	    {
	      double* evec2_m=mydiag2.GetEigenvector(m);
	      
	      if(TRACE){
		cout << "m:" << m << " evec2_m: ";
		for(int i=0; i<Neigenvalues2; i++){cout << evec2_m[i] << " ";}
		cout << endl;
	      }
	      

	      int Nstates2=A.Nrows();
	      vector<MyField> Apsin(Nstates2);
	      A.TimesVector(Apsin,mydiag1.GetLanczosEigenvectorinStateSpaceBasis(n));


	      MyField* psim  = mydiag2.GetLanczosEigenvectorinStateSpaceBasis(m);
	      int nstates2=Apsin.size();
	      MyField matrixelem=ScalarProduct(&nstates2,psim,&Apsin[0]);
	      MyField melem=evec1_n[0]*evec2_m[0]*conj(matrixelem)*mydiag2.GetNormofInitVector();

	      if(TRACE)
		{
		  cout << "Apsin: " << endl;
		  for(int i=0; i<Apsin.size(); i++) cout << Apsin[i] << " ";
		  cout << endl;
		  

		  cout << "matrixelem: " << matrixelem << " melem: " << melem << endl;
		}
	      Aline.Insert(eval1[n],eval2[m]-eval1[n],melem.real()); // store data
	    }
	}
    }
  ofstream Afile(filename.c_str(),ios::app);
  Aline.Write(Afile);
  Afile.close();
  if(TRACE) cout << "Done with CorrelationFunctionObservable" << endl;

}
  
#endif // FULLDIAGONALIZATION
  
#endif // CORRELATIONFUNCTION
  
  
#endif //OBSERVABLES_H
    
