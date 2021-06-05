#include<iostream>
#include<fstream>
using namespace std;


const bool TRACE=true;
//const bool WRITEOUTALLSTATES=false;


ofstream logfile("log.txt",ios::app);



#include "timer.h"
//#include "inttostring.h"
#include "samplingcounter.h"

//#include "rnddef.h" // the random number generator

#include "sparsematrix.h"

typedef ELLmatrix MyMatrix;

#ifdef FULLDIAGONALIZATION
#include "fulldiagonalization.h"
#else
#include "lanczos.h"
#endif

#include "observables.h"

#ifdef DOMAINWALLMODEL
#include "dwmodel.h" // definitions of model goes here.
#endif

#include "RunParameter.h"




int main()
{
  if(TRACE) cout << "Executing main()" << endl;
  Timer mytimer;  
  logfile << "\nStarting at: " << mytimer << endl;
  mytimer.Start();
  
  RunParameters rp("read.in");
  logfile << "Parameters: " << rp;
  
  double* parameters= rp.GetPars(1); // only one line
  if(TRACE) cout << "defining model" << endl;

#ifdef DOMAINWALLMODEL
  DWmodel mymodel(parameters);
#else
  cout << "ERROR: No model defined" << endl;
  exit(1);
#endif

  if(TRACE) cout << "---DONE DEFINING MODEL---" << endl;


  int M=static_cast<int>(parameters[MLANCZOS]);
  

  SamplingCounter scounter(mymodel.GetStateSpace(),parameters[SAMPLEFRAC]);
  
#ifdef FULLDIAGONALIZATION
  logfile << "Doing Full diagonalization" << endl;
#else
  logfile << "Doing Lanczos with M=" << M 
	  << " SAMPLEFRAC=" << parameters[SAMPLEFRAC] << endl;
#endif
  
  logfile << "Sample counter: " << endl;
  logfile << scounter << endl;
  
  
  
  int is1=0;
  int counter=0;
  while(scounter.MoreSectorsToSample(is1,counter))
    {
      Timer runtimer; runtimer.Start();
      logfile << "Doing sector: " << is1 << " counter: " << counter;
      if(TRACE) cout << "Doing sector: " << is1 << " counter: " << counter << endl;
      
      MyMatrix* Hptr=mymodel.GetHamiltonianSector(is1);
      if(Hptr==0){continue;}
      MyMatrix& H=*Hptr;
      
      const int Nstates=H.Ncols();
      logfile << "  Nstates: " << Nstates;
      
      if(TRACE)
	{
	  cout << "The matrix for sector " << is1 << " is: " << endl;
	  cout << H << endl;
	  
	  for(int j=0; j<H.Nrows(); j++)
	    {
	      for(int i=0; i<H.Ncols(); i++)
		{
		  cout << H(j,i) << " ";
		}
	      cout << endl;
	    }
	}



#ifdef FULLDIAGONALIZATION
      FullDiagonalization mydiag(H);
      
      BasicObservables(is1,1,mydiag,10);
      mymodel.ReleaseHamiltonian();

      vector<SingleOperatorObservable*>& obslist=mymodel.GetSingleOperatorObservables();
      for(int i=0; i<obslist.size(); i++)
	{
	  obslist[i]->Calculate(is1,1,mydiag);
	}

#ifdef CORRELATIONFUNCTION
      if(TRACE) cout << "Calculating correlation function" << endl;
      CorrelationFunctionObservable* cfptr=mymodel.GetCorrelationFunctionObservable();
      int is2=cfptr->GetOperator().GetOutSector(is1);

      if(TRACE) cout << is1 << " -> " << is2 << endl; 

      if(is2 != is1)
	{
	  MyMatrix* Hptr2=mymodel.GetHamiltonianSector(is2);
	  if(Hptr2==0){continue;}
	  MyMatrix& H2=*Hptr2;

	  if(TRACE){
	    cout << "The matrix for sector " << is2 << " is: " << endl;
	    cout << H2 << endl;
	    
	    for(int j=0; j<H2.Nrows(); j++)
	      {
		for(int i=0; i<H2.Ncols(); i++)
		  {
		    cout << H2(j,i) << " ";
		  }
		cout << endl;
	      }
	  }
	  
	  FullDiagonalization mydiag2(H2);
	  
	  cfptr->Calculate(is1,1,mydiag,mydiag2);
	}
      else
	{ cfptr->Calculate(is1,1,mydiag,mydiag);}
#endif

#else // LANCZOS
      Lanczos mylanczos(H,M,counter); // start with basis state nr

      //      vector<MyField>& v_init=mylanczos.GetInitVector();
      MyField* v_init=mylanczos.GetInitVector();

      mylanczos.DoLanczos();

      // diagonalize Lanczos matrix
      mylanczos.DiagonalizeLanczosMatrix();
      if(TRACE) cout << "Eigenvalues: " << endl;
      
      vector<double> evals=mylanczos.GetEigenvalues();

      if(TRACE){
	for(int i=0; i<mylanczos.GetNLanczosIterations(); i++){ cout << evals[i] << " ";}
	cout << endl;
      }


      if(TRACE){
	for(int i=0; i<mylanczos.GetNLanczosIterations(); i++)
	  { 
	    cout << "eigenvalue: " << evals[i] << ":" << endl;
	    cout << "eigenvector:" << endl;
	    double* eigenveci=mylanczos.GetEigenvector(i);
	    for(int j=0; j<mylanczos.GetNLanczosIterations(); j++)
	      {cout << eigenveci[j] << " ";}
	    cout << endl;
	  }
      }

      mymodel.ReleaseHamiltonian(); // free the Hamiltonian matrix

      //Observables:
      //Partition function and energy
      BasicObservables(is1,Nstates,mylanczos,10);
      
      //Single Operator observables
      vector<SingleOperatorObservable*>& obslist=mymodel.GetSingleOperatorObservables();
      for(int i=0; i<obslist.size(); i++)
	{
	  obslist[i]->Calculate(is1,Nstates,mylanczos);
	}
      
#ifdef CORRELATIONFUNCTION
      //For dynamic quantities
      CorrelationFunctionObservable* cfptr=mymodel.GetCorrelationFunctionObservable();
      
      MyMatrix* Aptr=cfptr->GetOperator().GetSector(is1); 
      if(Aptr==0){if(TRACE) cout << "A has no action on sector " << is1 << " ignoring contribution" << endl; continue;}
      
      MyMatrix& A=*Aptr;
      
      if(TRACE)
	{
	  cout << "A: " << endl;
	  cout << A << endl;
	  
	  for(int j=0; j<A.Nrows(); j++)
	    {
	      for(int i=0; i<A.Ncols(); i++)
		{
		  cout << A(j,i) << " ";
		}
	      cout << endl;
	    }
	}
      //      vector<MyField> av_init=A*v_init;
      const int Nstates2=A.Nrows();
      vector<MyField> av_init(Nstates2);
      A.TimesVector(av_init,v_init);
      //      int dummy=Normalize(av_init); // normalize the second vector
      
      if(TRACE)
	{	  
	  cout << "A*v_init: " << endl;
	  for(int i=0; i<av_init.size(); i++) cout << av_init[i] << " ";
	  cout << endl;
	}
      //      mymodel.ReleaseSecondSector(); 
      cfptr->GetOperator().Free();
      
      
      int is2=cfptr->GetOperator().GetOutSector(is1);
      if(TRACE) cout << "A: " << is1 << " -> " << is2 << endl;
      
      if(TRACE) cout << "Getting H for the 2nd Lanczos, the H operates on sector " << is2 << endl;
      
      Hptr=mymodel.GetHamiltonianSector(is2);
      if(TRACE) cout << "Hptr: " << Hptr << endl;
      if(Hptr==0){cout << "H2=0 WHAT TO DO!!!" << endl; exit(1);}
      MyMatrix& H2=*Hptr;
      
      if(TRACE){
	cout << "The matrix for sector " << is2 << " is: " << endl;
	cout << H2 << endl;
	
	for(int j=0; j<H2.Nrows(); j++)
	  {
	    for(int i=0; i<H2.Ncols(); i++)
	      {
		cout << H2(j,i) << " ";
	      }
	    cout << endl;
	  }
      }
      
      Lanczos mylanczos2(H2,M,av_init); 
      mylanczos2.DoLanczos();

      if(TRACE) cout << "done with 2. lanczos" << endl;
      mylanczos2.DiagonalizeLanczosMatrix();
      if(TRACE) cout << "done with 2. Lanczos diagonalization" << endl;

      mymodel.ReleaseHamiltonian(); // free the Hamiltonian matrix
      
      if(TRACE) cout << "Calculating observable" << endl;
      cfptr->Calculate(is1,Nstates,mylanczos,mylanczos2);

  
      // end of dynamic quantities
#endif // CORRELATIONFUNCTION
      
#endif // FULLDIAGONALIZATION/LANCZOS
      
      runtimer.Stop();
      double iteratortime = runtimer.GetTimeElapsed();
      logfile << " Time pr. iteration: " << iteratortime;
      if(iteratortime !=0.)
	logfile << " s. #iter in 1 hour: " 
		<< static_cast<int>(3600/iteratortime) 
		<< " in 24 hours: " 
		<< static_cast<int>(86400/iteratortime);
      logfile << endl;
    }
  mytimer.Stop();
  double time_spent = mytimer.GetTimeElapsed();
  logfile << "Time elapsed:  " << time_spent << " sec. :" 
          << time_spent/3600. << " hours." << endl;
  logfile << "Ending at: " << mytimer;
  logfile << "---Done--- " << endl;
  logfile.close();
  
  ofstream terminationsign("end.exec");
  terminationsign << "execution ended at " << mytimer;
  terminationsign.close();
  
  cout << "program ended \n";
  return 0;
}
