#ifndef OBSERVABLES_STD_C
#define OBSERVABLES_STD_C

#include<fstream>
#include<complex>
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

#include "cuda_runtime.h"
#include <cublas_v2.h>
#include "cuComplex.h"

#include "cuda_global.h"
#include "cuda_routines.h"
#include "sparsematrix.h"
#include "observables.h"
#include "observables_cuda.h"
#ifdef ONETEMPERATURE
#include "onetemperature.h"
#else
#include "alltemperatures.h" // the default
#endif
#include "lowlevelroutines.h"
#include "global.h"




#ifdef FULLDIAGONALIZATION
BasicObservables::BasicObservables(int sector,int nstates,FullDiagonalization& mydiag,double beta){impl = new BasicObservables_impl(sector,nstates,mydiag,beta);}
#else
BasicObservables::BasicObservables(int sector,int nstates,Lanczos& mydiag,double beta){impl = new BasicObservables_impl(sector,nstates,mydiag,beta);}
#endif
BasicObservables::~BasicObservables(){delete impl;}



#ifdef FULLDIAGONALIZATION
BasicObservables_impl::BasicObservables_impl(int sector,int nstates,FullDiagonalization& mydiag,double beta)
#else
BasicObservables_impl::BasicObservables_impl(int sector,int nstates,Lanczos& mydiag,double beta)
#endif
{
  if(WTRACE) cout << "Starting BasicObservables_imp (observables_cuda.cu)" << endl;

  double MinEnergy=MAXMINENERGY;
  int Neigenvalues=mydiag.GetNeigenvalues();

  vector<double>& eval=mydiag.GetEigenvalues();

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


  // Energy:

  DataLine Eline(sector,nstates);
  for(int i=0; i<Neigenvalues; i++)
    {
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
  DataLine Eline2(sector,nstates);
  for(int i=0; i<Neigenvalues; i++)
    {
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



SingleOperatorObservable::SingleOperatorObservable(Operator& Aop_in,string filename_in){impl=new SingleOperatorObservable_impl(Aop_in,filename_in);}

#ifdef FULLDIAGONALIZATION
void SingleOperatorObservable::Calculate(int sector,int nstates,FullDiagonalization& mydiag){impl->Calculate(sector,nstates,mydiag);}
#else
void SingleOperatorObservable::Calculate(int sector,int nstates,Lanczos& mydiag){impl->Calculate(sector,nstates,mydiag);}
#endif

SingleOperatorObservable::~SingleOperatorObservable(){delete impl;}

Operator& SingleOperatorObservable::GetOperator(){return impl->GetOperator();}


SingleOperatorObservable_impl::SingleOperatorObservable_impl(Operator& Aop_in,string filename_in):Aop(Aop_in),filename(filename_in),handle(NULL),status(cublasCreate(&handle)){}

SingleOperatorObservable_impl::~SingleOperatorObservable_impl(){cublasDestroy(handle);}

#ifdef FULLDIAGONALIZATION
void SingleOperatorObservable_impl::Calculate(int sector,int nstates,FullDiagonalization& mydiag)
#else
void SingleOperatorObservable_impl::Calculate(int sector,int nstates,Lanczos& mydiag)
#endif
{
  if(WTRACE) cout << "in SingleOperatorObservable Calculate " << filename << " sector " << sector << endl; 

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

  DataLine Aline(sector,nstates);

  MyMatrix& A=*Aptr;
  int Rows=A.Nrows();
  int Cols=A.Ncols();
  int Nc=A.NStoredCols();

  if(WTRACE) cout << "Rows: " << Rows << " Cols: " << Cols << " Nc " << Nc << endl;

  /*  
  int threadsPerBlock= NTHREADSPERBLOCK;
  int blocksPerGrid  = (Rows + threadsPerBlock - 1)/threadsPerBlock;
  */

  // Upload A
  cuda_cmplx* d_m_val=NULL;
  int* d_m_col=NULL;
  UploadMatrixToDevice(A,&d_m_val,&d_m_col);
  
#ifdef FULLDIAGONALIZATION

  MyField* phi_ptr= mydiag.GetEigenvector(0);
  // uploading eigenvectors to device:
  cuda_cmplx* d_phi=NULL;
  AllocateSpaceOnDevice(&d_phi,Cols*Neigenvalues);
  UploadToDevice(phi_ptr,Cols*Neigenvalues,&d_phi);

  // allocate space for the resulting vector A phi
  cuda_cmplx* d_Aphi=NULL;
  AllocateSpaceOnDevice(&d_Aphi,Rows*Neigenvalues);
  MyField* Aphi=new MyField[Rows*Neigenvalues];
  UploadToDevice(Aphi,Rows*Neigenvalues,&d_Aphi);
  
  // carry out multiplication A phi_i for all eigenvectors i
  for(int i=0; i<Neigenvalues; i++)
    {
      //      if(WTRACE) cout << "Launching cuda_mat_vec_multiply_cmplx with " << blocksPerGrid << " blocks and " << threadsPerBlock  << " threads per block" << endl; 
      //      cuda_mat_vec_multiply_cmplx<<<blocksPerGrid,threadsPerBlock>>>(Rows,Nc,d_m_col ,d_m_val, &d_phi[i*Cols], &d_Aphi[i*Cols]); // Aphi1=A*phi1;
      cuda_mat_vec_multiply_cmplx(Rows,Nc,d_m_col ,d_m_val, &d_phi[i*Cols], &d_Aphi[i*Cols]); // Aphi1=A*phi1;
      err = cudaGetLastError();						     
      if(TRACE) cout << "end of cuda_mat_vec_multiply_cmplx, last error" << err << endl; 
      
    }
  
  
  // then finally calculate all matrix elements
  
  // allocate answer array
  cuda_cmplx* phiAphi=new cuda_cmplx[Neigenvalues];
  for(int p=0; p<Neigenvalues; p++)
    {
      cublasZdotc(handle,Rows,&d_phi[p*Cols],1,&d_Aphi[p*Cols],1,&phiAphi[p]);
    }
  
  MyField* phiAphi_complex=reinterpret_cast<MyField*>(phiAphi);
  
  for(int p=0; p<Neigenvalues; p++)
    {
      Aline.Insert(eval[p],phiAphi_complex[p].real());
    }
  
  FreeMatrixOnDevice(&d_m_val,&d_m_col);
  FreeMemoryOnDevice(&d_Aphi);
  FreeMemoryOnDevice(&d_phi);
  delete[] Aphi;
  delete[] phi2Aphi1;
  
#else
  
  // uploading lanczos vectors to device:
  cuda_cmplx* d_phi=NULL;
  AllocateSpaceOnDevice(&d_phi,Cols*Neigenvalues);
  UploadToDevice(reinterpret_cast<cuda_cmplx*>(mydiag.GetInitVector()),Cols*Neigenvalues,&d_phi);
  
  // allocate space for the resulting vector A phi_1
  cuda_cmplx* d_Aphi=NULL;
  AllocateSpaceOnDevice(&d_Aphi,Rows);
  MyField* Aphi=new MyField[Rows];
  UploadToDevice(reinterpret_cast<cuda_cmplx*>(Aphi),Rows,&d_Aphi);

  // carry out the matrix multiplication A phi1
  //  if(WTRACE) cout << "Launching cuda_mat_vec_multiply_cmplx with " << blocksPerGrid << " blocks and " << threadsPerBlock  << " threads per block" << endl; 
  cuda_mat_vec_multiply_cmplx(Rows,Nc,d_m_col ,d_m_val, &d_phi[0], &d_Aphi[0]); // Aphi=A*phi;
  cudaError_t err = cudaGetLastError();						     
  if(WTRACE) cout << "end of cuda_mat_vec_multiply_cmplx, last error" << err << endl; 

  //  cudaThreadSynchronize();
  
  // allocate answer array
  cuda_cmplx* phi2Aphi1=new cuda_cmplx[Neigenvalues];
  
  if(TRACE) cout << "Starting on computing the matrix elements" << endl;    
  for( int p=0; p<Neigenvalues; p++)
    {
      // scalarproduct 
      cublasZdotc(handle,Rows,&d_phi[p*Cols],1,&d_Aphi[0],1,&phi2Aphi1[p]);
    }

  delete[] Aphi;
  FreeMatrixOnDevice(&d_m_val,&d_m_col);
  FreeMemoryOnDevice(&d_Aphi);
  FreeMemoryOnDevice(&d_phi);
  
  
  for(int i=0; i<Neigenvalues; i++)
    {
	  //      Aline.energy[i]=eval[i];
      double* evec=mydiag.GetEigenvector(i);
      double nui0=evec[0];
      
      MyField* psi_i=reinterpret_cast<MyField*>(&phi2Aphi1[i]);
      
      double factor=(nui0*(*psi_i)).real(); // CHECK THIS, I think I miss a factor nu something
      Aline.Insert(eval[i],factor);
    }

  delete[] phi2Aphi1;
  
#endif // FULLDIAGONALIZATION  

  ofstream Afile(filename.c_str(),ios::app);
  Aline.Write(Afile);
  Afile.close();

  Aop.Free();
}

Operator& SingleOperatorObservable_impl::GetOperator(){return Aop;}


//**********************************************************************************
#ifdef CORRELATIONFUNCTION

CorrelationFunctionObservable::CorrelationFunctionObservable(Operator& Aop_in,string filename_in)
{impl=new CorrelationFunctionObservable_impl(Aop_in,filename_in);}

CorrelationFunctionObservable::~CorrelationFunctionObservable(){delete impl;}


#ifdef FULLDIAGONALIZATION

void CorrelationFunctionObservable::Calculate(int sector,int nstates,FullDiagonalization& mydiag1,FullDiagonalization& mydiag2)
{impl->Calculate(sector,nstates,mydiag1,mydiag2);}

#else

void CorrelationFunctionObservable::Calculate(int sector,int nstates,Lanczos& mydiag1,Lanczos& mydiag2){impl->Calculate(sector,nstates,mydiag1,mydiag2);}

#endif

bool CorrelationFunctionObservable::CalculateInitVector(int sector,MyField* vinit,MyField* avinit){return impl->CalculateInitVector(sector,vinit,avinit);}

Operator& CorrelationFunctionObservable::GetOperator(){return impl->GetOperator();}


CorrelationFunctionObservable_impl::CorrelationFunctionObservable_impl(Operator& Aop_in,string filename_in):Aop(Aop_in),filename(filename_in),handle(NULL),status(cublasCreate(&handle)){}

CorrelationFunctionObservable_impl::~CorrelationFunctionObservable_impl(){cublasDestroy(handle);}

Operator& CorrelationFunctionObservable_impl::GetOperator(){return Aop;}

#ifdef FULLDIAGONALIZATION
void CorrelationFunctionObservable_impl::Calculate(int sector,int nstates,FullDiagonalization& mydiag1,FullDiagonalization& mydiag2)
{
  if(WTRACE) cout << "in CorrelationFunctionObservable Calculate " << filename << " sector " << sector << " nstates:" << nstates << endl; 
  
  CorrelationDataLine Aline(sector,1);
  
  int Neigenvalues1=mydiag1.GetNeigenvalues();
  int Neigenvalues2=mydiag2.GetNeigenvalues();
  
  if(TRACE) cout << "Nevals1: " << Neigenvalues1 << " Nevals2: " << Neigenvalues2 << endl;
  
  MyMatrix* Aptr=Aop.GetSector(sector);
  
  if(Aptr!=0 && Neigenvalues1>0 && Neigenvalues2>0)
    {
      // compute the matrix <phi2_m | A | phi1_n>

      MyField* phi1_ptr= mydiag1.GetEigenvector(0);
      MyField* phi2_ptr= mydiag2.GetEigenvector(0);


      MyMatrix& A=*Aptr;
      int Rows=A.Nrows();
      int Cols=A.Ncols();
      int Nc=A.NStoredCols();

      int threadsPerBlock= NTHREADSPERBLOCK;
      int blocksPerGrid  = (Rows + threadsPerBlock - 1)/threadsPerBlock;

	
      // Upload A
      cuda_cmplx* d_m_val=NULL;
      int* d_m_col=NULL;
      UploadMatrixToDevice(A,&d_m_val,&d_m_col);

      // uploading lanczos vectors to device:
      cuda_cmplx* d_phi1=NULL;
      AllocateSpaceOnDevice(&d_phi1,Cols*Neigenvalues1);
      UploadToDevice(phi1_ptr,Cols*Neigenvalues1,&d_phi1);

      // allocate space for the resulting vectors
      cuda_cmplx* d_Aphi1=NULL;
      AllocateSpaceOnDevice(&d_Aphi1,Rows*Neigenvalues1);
      MyField* Aphi1=new MyField[Rows*Neigenvalues1];
      UploadToDevice(Aphi1,Rows*Neigenvalues1,&d_Aphi1);


      // carry out the matrix multiplication A phi1
      for(int i=0; i<Neigenvalues1; i++)
	{
	  //	if(TRACE) cout << "Launching cuda_mat_vec_multiply_cmplx with " << blocksPerGrid << " blocks and " << threadsPerBlock  << " threads per block" << endl; 
	  //cuda_mat_vec_multiply_cmplx<<<blocksPerGrid,threadsPerBlock>>>(Rows,Nc,d_m_col ,d_m_val, &d_phi1[i*Cols], &d_Aphi1[i*Cols]); // Aphi1=A*phi1;
	  cuda_mat_vec_multiply_cmplx(Rows,Nc,d_m_col ,d_m_val, &d_phi1[i*Cols], &d_Aphi1[i*Cols]); // Aphi1=A*phi1;
   	err = cudaGetLastError();						     
	if(TRACE) cout << "end of cuda_mat_vec_multiply_cmplx, last error" << err << endl; 

	}

      FreeMatrixOnDevice(&d_m_val,&d_m_col);
      FreeMemoryOnDevice(&d_phi1);

      // upload the second set of lanczos vectors to device:
      cuda_cmplx* d_phi2=NULL;
      AllocateSpaceOnDevice(&d_phi2,Rows*Neigenvalues2);
      UploadToDevice(phi2_ptr,Rows*Neigenvalues2,&d_phi2);


      // allocate answer array
      cuda_cmplx* phi2Aphi1=new cuda_cmplx[Neigenvalues1*Neigenvalues2];

      if(TRACE) cout << "Starting on computing the matrix" << endl;    
      for( int k=0; k<Neigenvalues2; k++)
	for( int p=0; p<Neigenvalues1; p++)
	  {
	    // scalarproduct 
	    int indx=k*Neigenvalues1+p;
	    cublasZdotc(handle,Rows,&d_phi2[k*Rows],1,&d_Aphi1[p*Rows],1,&phi2Aphi1[indx]);
	  }

      FreeMemoryOnDevice(&d_Aphi1);
      FreeMemoryOnDevice(&d_phi2);

      vector<double>& eval1=mydiag1.GetEigenvalues();
      vector<double>& eval2=mydiag2.GetEigenvalues();
      
      for( int k=0; k<Neigenvalues2; k++)
	for( int p=0; p<Neigenvalues1; p++)
	  {
	    int indx=k*Neigenvalues1+p;
	    MyField* matrixelem_ptr=reinterpret_cast<MyField*>(&phi2Aphi1[indx]);
	    Aline.Insert(eval1[n],eval2[m]-eval1[n],norm(*matrixelem_ptr)); // store data abs^2
	  }
      delete[] Aphi1;
      delete[] phi2Aphi1;
    }
  
  ofstream Afile(filename.c_str(),ios::app);
  Aline.Write(Afile);
  Afile.close();
}
#else

void CorrelationFunctionObservable_impl::Calculate(int sector,int nstates,Lanczos& mydiag1,Lanczos& mydiag2)
{
  if(WTRACE) cout << "in CorrelationFunctionObservable Calculate Lanczos " << filename << " sector " << sector << " nstates: " << nstates << endl; 
  
  CorrelationDataLine Aline(sector,nstates);
  
  int Neigenvalues1=mydiag1.GetNeigenvalues();
  int Neigenvalues2=mydiag2.GetNeigenvalues();
  
  if(TRACE) cout << "Nevals1: " << Neigenvalues1 << " Nevals2: " << Neigenvalues2 << endl;
  
  MyMatrix* Aptr=Aop.GetSector(sector);
  
  if(TRACE) cout << "Aptr: " << Aptr << endl;

  if(Aptr!=0 && Neigenvalues1>0 && Neigenvalues2>0) // else just write out a zero dataline
    {
      // compute the matrix <phi2 | A | phi1>

      MyMatrix& A=*Aptr;
      int Rows=A.Nrows();
      int Cols=A.Ncols();
      int Nc=A.NStoredCols();

      /*
      const int threadsPerBlock= NTHREADSPERBLOCK;
      const int blocksPerGrid  = (Rows + threadsPerBlock - 1)/threadsPerBlock;
      */	

      // Upload A
      cuda_cmplx* d_m_val=NULL;
      int* d_m_col=NULL;
      UploadMatrixToDevice(A,&d_m_val,&d_m_col);

      // uploading lanczos vectors to device:
      cuda_cmplx* d_phi1=NULL;
      AllocateSpaceOnDevice(&d_phi1,Cols*Neigenvalues1);
      UploadToDevice(reinterpret_cast<cuda_cmplx*>(mydiag1.GetInitVector()),Cols*Neigenvalues1,&d_phi1);

      // allocate space for the resulting vectors
      cuda_cmplx* d_Aphi1=NULL;
      AllocateSpaceOnDevice(&d_Aphi1,Rows*Neigenvalues1);
      MyField* Aphi1=new MyField[Rows*Neigenvalues1];
      UploadToDevice(reinterpret_cast<cuda_cmplx*>(Aphi1),Rows*Neigenvalues1,&d_Aphi1);


      // carry out the matrix multiplication A phi1
      for(int i=0; i<Neigenvalues1; i++)
	{
	  //	if(TRACE) cout << "Launching cuda_mat_vec_multiply_cmplx with " << blocksPerGrid << " blocks and " << threadsPerBlock  << " threads per block" << endl; 
	  //	cuda_mat_vec_multiply_cmplx<<<blocksPerGrid,threadsPerBlock>>>(Rows,Nc,d_m_col ,d_m_val, &d_phi1[i*Cols], &d_Aphi1[i*Rows]); // Aphi1=A*phi1;
	cuda_mat_vec_multiply_cmplx(Rows,Nc,d_m_col ,d_m_val, &d_phi1[i*Cols], &d_Aphi1[i*Rows]); // Aphi1=A*phi1;
   	cudaError_t err = cudaGetLastError();						     
	if(TRACE) cout << "end of cuda_mat_vec_multiply_cmplx, last error" << err << endl; 

	}

      FreeMatrixOnDevice(&d_m_val,&d_m_col);
      FreeMemoryOnDevice(&d_phi1);

      // upload the second set of lanczos vectors to device:
      cuda_cmplx* d_phi2=NULL;
      AllocateSpaceOnDevice(&d_phi2,Rows*Neigenvalues2);
      UploadToDevice(reinterpret_cast<cuda_cmplx*>(mydiag2.GetInitVector()),Rows*Neigenvalues2,&d_phi2);



      // allocate answer array
      cuda_cmplx* phi2Aphi1=new cuda_cmplx[Neigenvalues1*Neigenvalues2];

      if(TRACE) cout << "Starting on computing the matrix" << endl;    
      for( int k=0; k<Neigenvalues2; k++)
	for( int p=0; p<Neigenvalues1; p++)
	  {
	    // scalarproduct 
	    int indx=k*Neigenvalues1+p;
	    cublasZdotc(handle,Rows,&d_phi2[k*Rows],1,&d_Aphi1[p*Rows],1,&phi2Aphi1[indx]);
	  }

      FreeMemoryOnDevice(&d_Aphi1);
      FreeMemoryOnDevice(&d_phi2);


      vector<double>& eval1=mydiag1.GetEigenvalues();
      vector<double>& eval2=mydiag2.GetEigenvalues();


      
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
	      

	      MyField sum=0;
	      for(int k=0; k<Neigenvalues2; k++)
		for(int p=0; p<Neigenvalues1; p++)
		  {
		    int indx=k*Neigenvalues1+p;
		    complex<double>* cptr=reinterpret_cast<complex<double>*>(&phi2Aphi1[indx]);
		    sum+=evec1_n[p]*evec2_m[k]*conj(*cptr);
		  }

	      MyField melem=evec1_n[0]*evec2_m[0]*sum*mydiag2.GetNormofInitVector();

	      if(TRACE)
		{
		  cout << " melem: " << melem << endl;
		}
	      Aline.Insert(eval1[n],eval2[m]-eval1[n],melem.real()); // store data
	    }
	}
      delete[] Aphi1;
      delete[] phi2Aphi1;
    }
  ofstream Afile(filename.c_str(),ios::app);
  Aline.Write(Afile);
  Afile.close();
  if(TRACE) cout << "Done with CorrelationFunctionObservable" << endl;

}
#endif // FULLDIAGONALIZATION

// Routine for making the multiplicatiojn A*v_init
bool CorrelationFunctionObservable_impl::CalculateInitVector(int sector,MyField* vinit,MyField* Avinit)
{
  if(WTRACE) cout << "in CalculateInitVector " << endl;

  MyMatrix* Aptr=Aop.GetSector(sector);

  if(Aptr == NULL) return false;
  
  MyMatrix& A=*Aptr;

  int Rows=A.Nrows();
  int Cols=A.Ncols();
  int Nc=A.NStoredCols();

  if(TRACE)
    {
      cout << "A: " << endl;
      cout << A << endl;

      cout << "Rows:" << Rows << " Cols:" << Cols << " Nc:" << Nc << endl;
      
      for(int j=0; j<min(A.Nrows(),10); j++)
	{
	  for(int i=0; i<min(A.Ncols(),10); i++)
	    {
	      cout << A(j,i) << " ";
	    }
	  cout << endl;
	}
    }


  if(TRACE)
    {
      cout << "v_init: " << endl;
      for(int i=0; i<Cols; i++) cout << vinit[i] << " ";
      cout << endl;
    }
  

  /*
  const int threadsPerBlock= NTHREADSPERBLOCK;
  const int blocksPerGrid  = (Rows + threadsPerBlock - 1)/threadsPerBlock;
  */

  // Upload A
  cuda_cmplx* d_m_val=NULL;
  int* d_m_col=NULL;
  UploadMatrixToDevice(A,&d_m_val,&d_m_col);
  
  // uploading init vector to device:
  cuda_cmplx* d_vinit=NULL;
  cuda_cmplx* vinit_ptr=reinterpret_cast<cuda_cmplx*>(vinit);
  AllocateSpaceOnDevice(&d_vinit,Cols);
  UploadToDevice(vinit_ptr,Cols,&d_vinit);
  
  // allocate space for the resulting vectors
  cuda_cmplx* d_Avinit=NULL;
  cuda_cmplx* Avinit_ptr=reinterpret_cast<cuda_cmplx*>(Avinit);
  AllocateSpaceOnDevice(&d_Avinit,Rows);
  UploadToDevice(Avinit_ptr,Rows,&d_Avinit); // ensure initialization is correct
  // carry out the matrix multiplication A vinit
  
  //  if(WTRACE) cout << "Launching cuda_mat_vec_multiply_cmplx with " << blocksPerGrid << " blocks and " << threadsPerBlock  << " threads per block" << endl; 
  //  cuda_mat_vec_multiply_cmplx<<<blocksPerGrid,threadsPerBlock>>>(Rows,Nc,d_m_col ,d_m_val, d_vinit, d_Avinit); // Aphi1=A*phi1;
  cuda_mat_vec_multiply_cmplx(Rows,Nc,d_m_col ,d_m_val, d_vinit, d_Avinit); // Aphi1=A*phi1;
  cudaError_t err = cudaGetLastError();						     
  if(TRACE) cout << "end of cuda_mat_vec_multiply_cmplx, last error" << err << endl; 
  
  DownloadFromDevice(&d_Avinit,Rows,Avinit_ptr);

  if(TRACE)
    {
      cout << "Av_init: " << endl;
      for(int i=0; i<Rows; i++) cout << Avinit[i] << " ";
      cout << endl;
    }

  FreeMatrixOnDevice(&d_m_val,&d_m_col);
  FreeMemoryOnDevice(&d_vinit);
  FreeMemoryOnDevice(&d_Avinit);  

  return true;
}

  
#endif // CORRELATIONFUNCTION

  
#endif //OBSERVABLES_CUDA_CU
   
