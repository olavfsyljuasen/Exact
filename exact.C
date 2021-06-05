#include<iostream>
#include<fstream>
using namespace std;

#ifdef DEBUG
const bool WTRACE=true;
#else
const bool WTRACE=false;
#endif


#ifdef DEBUG2
const bool TRACE=true;
#else
const bool TRACE=false;
#endif

//const bool WRITEOUTALLSTATES=false;


ofstream logfile("log.txt",ios::app);



#include "timer.h"
//#include "inttostring.h"
#include "samplingcounter.h"

//#include "rnddef.h" // the random number generator

#include "sparsematrix.h"

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

#include "global.h" // global typedefs


#include "observablesspec.h" // a file containing T,ESUB,Frequencygrid, for use with ONETEMPERATURE


const int LOGFILETICKLER=10; // only write every 10. line to logfile


// for use with the PBS system when signal is received, process
// exits gracefully.
#include <stdlib.h>
#include <signal.h>
void sighandler(int a)
{
  logfile << "process killed. Exiting gracefully." << endl;
  exit(0);
}




int main()
{
  // for use with the PBS system, when time runs out
  // the exit signal SIGTERM is sent to the program
  // which is handled so that it exits gracefully:

  signal(SIGTERM, sighandler);

  if(WTRACE) cout << "Executing main()" << endl;
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

  if(WTRACE) cout << "---DONE DEFINING MODEL---" << endl;

  // Write new spec file for observables when it does not exist from before  
  double spec[6]={parameters[BETA],parameters[ESUB],parameters[NWBINS],parameters[WMIN],parameters[WMAX],parameters[SIGMA]};

  ObservablesSpec ObsSpec(&spec[0]);

  int M=static_cast<int>(parameters[MLANCZOS]);
  

  SamplingCounter scounter(mymodel.GetStateSpace(),parameters[SAMPLEFRAC],parameters[SECTOR]);
  
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
      if(counter%LOGFILETICKLER==0) logfile << "Doing sector: " << is1 << " counter: " << counter;
      if(WTRACE) cout << "*****  Doing sector: " << is1 << " counter: " << counter << endl;

      MyMatrix* Hptr=mymodel.GetHamiltonianSector(is1);
      if(Hptr==0){scounter.Decrement(is1); continue;}
      MyMatrix& H=*Hptr;
      
      const int Nstates=H.Ncols();
      //      logfile << "  Nstates: " << Nstates;
      
      if(TRACE)
	{
	  cout << "The Hamiltonian for sector " << is1 << " is: " << endl;
	  cout << H << endl;
	  
	  for(int j=0; j<min(H.Nrows(),10); j++)
	    {
	      for(int i=0; i<min(H.Ncols(),10); i++)
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
      CorrelationFunctionObservable* cfptr=mymodel.GetCorrelationFunctionObservable();
      int is2=cfptr->GetOperator().GetOutSector(is1);

      if(TRACE) cout << is1 << " -> " << is2 << endl; 

      if(is2 != is1)
	{
	  MyMatrix* Hptr2=mymodel.GetHamiltonianSector(is2);
	  if(Hptr2==0){scounter.Decrement(is1); continue;}
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
#ifdef USEBASISSTATESASINITFORSMALLSECTORS
      Lanczos mylanczos(H,M,counter); // start with basis state nr
#else
      Lanczos mylanczos(H,M,-1); // start with random state
#endif
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
      if(WTRACE) cout << "***Recording BasicObservables" << endl;
      BasicObservables(is1,Nstates,mylanczos,10);
      
      //Single Operator observables
      if(WTRACE) cout << "***Recording Single Operator observables" << endl;
      vector<SingleOperatorObservable*>& obslist=mymodel.GetSingleOperatorObservables();

      for(int i=0; i<obslist.size(); i++)
	{
	  obslist[i]->Calculate(is1,Nstates,mylanczos);
	}
      
#ifdef CORRELATIONFUNCTION
      if(WTRACE) cout << "***Recording CORRELATIONFUNCTION" << endl;
      //For dynamic quantities
      CorrelationFunctionObservable* cfptr=mymodel.GetCorrelationFunctionObservable();

      int is2=cfptr->GetOperator().GetOutSector(is1);
      if(TRACE) cout << "A: " << is1 << " -> " << is2 << endl;

      const int Nstates2=mymodel.GetStateSpace().sectors[is2]->Nstates;
      if(TRACE) cout << "Nstates: " << Nstates << " -> " << Nstates2 << endl;


      vector<MyField> Av_init(Nstates2);

      bool Afinite=cfptr->CalculateInitVector(is1,&v_init[0],&Av_init[0]);

      if(TRACE) cout << "Freeing cfptr" << endl;
      cfptr->GetOperator().Free();

      if(TRACE)
	{
	  cout << "A*v_init:" << endl;
	  for(int j=0; j<Nstates2; j++) cout << Av_init[j] << " ";
	  cout << endl;
	}

      if(!Afinite){scounter.Decrement(is1); continue;} // matrix is zero, start over.
      
      if(TRACE) cout << "Getting H for the 2nd Lanczos, the H operates on sector " << is2 << endl;
      
      Hptr=mymodel.GetHamiltonianSector(is2);
      if(TRACE) cout << "Hptr: " << Hptr << endl;
      if(Hptr==0){cout << "H2=0 WHAT TO DO!!!" << endl; exit(1);}
      MyMatrix& H2=*Hptr;
      
      if(TRACE){
	cout << "The matrix for sector " << is2 << " is: " << endl;
	cout << H2 << endl;
	
	for(int j=0; j<min(H2.Nrows(),10); j++)
	  {
	    for(int i=0; i<min(H2.Ncols(),10); i++)
	      {
		cout << H2(j,i) << " ";
	      }
	    cout << endl;
	  }
      }
      
      if(WTRACE) cout << "***Doing the 2. Lanczos" << endl;
      Lanczos mylanczos2(H2,M,Av_init); 
      mylanczos2.DoLanczos();

      if(TRACE) cout << "done with 2. lanczos" << endl;
      mylanczos2.DiagonalizeLanczosMatrix();
      if(TRACE) cout << "done with 2. Lanczos diagonalization" << endl;

      mymodel.ReleaseHamiltonian(); // free the Hamiltonian matrix
      
      if(WTRACE) cout << "***Calculating correlation function" << endl;
      cfptr->Calculate(is1,Nstates,mylanczos,mylanczos2);
  
      // end of dynamic quantities
#endif // CORRELATIONFUNCTION
      
#endif // FULLDIAGONALIZATION/LANCZOS

      scounter.Decrement(is1); // register completion and save

      runtimer.Stop();
      double iteratortime = runtimer.GetTimeElapsed();
      if(counter%LOGFILETICKLER==0)
	{
	  logfile << " Time pr. iteration: " << iteratortime;
	  if(iteratortime !=0.)
	    logfile << " s. #iter in 1 hour: " 
		    << static_cast<int>(3600/iteratortime) 
		    << " in 24 hours: " 
		    << static_cast<int>(86400/iteratortime);
	  logfile << endl;
	}
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
