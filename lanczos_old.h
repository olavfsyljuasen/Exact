#ifndef LANCZOS_H
#define LANCZOS_H

#include<iostream>
#include<vector>
#include<complex>
using namespace std;

const double TINYNUMBER=1e-15;

typedef CSRmatrix MyMatrix;

//--------------------------------------------------------------
// The Lanczos routine computes the Lanczos matrix
// Diagonalization of the Lanczosmatrix must be done separately
// in the routine LanczosDiag
//--------------------------------------------------------------


complex<double> t=complex<double>(1,0);
#include "lowlevelroutines.h"
complex<double> t2=complex<double>(1,0);
#include "banddiag.h"


class Lanczos
{
public:
  Lanczos(MyMatrix& H_in,int M_in); // initializes Lanczos with random vector
  Lanczos(MyMatrix& H_in,int M_in,int basestatenr); // initializes Lanczos with a basestate 
  Lanczos(MyMatrix& H_in,int M_in, vector<MyField>& vec_in); // initializes Lanczos using the input vector
  ~Lanczos(); 
  void DoLanczos();
  vector<MyField>& GetInitVector(){return *phi[0];}
  vector<MyField>& GetPhi(const int i){return *phi[i];}
  void DiagonalizeLanczosMatrix();
  vector<double>& GetLanczosa(){return lanczos_a;}
  vector<double>& GetLanczosb(){return lanczos_b;}  // we start numbering b from b0,b1 etc.
  vector<double>& GetEigenvalues(){return LM->Eigenvalues();}
  double* GetEigenvector(const int i){return LM->Eigenvector(i);}
  int GetNeigenvalues(){return Niterations;}
  int GetNLanczosIterations(){return Niterations;}
  double GetNormofInitVector(){return initnorm;} 
  vector<MyField>& GetLanczosEigenvectorinStateSpaceBasis(const int n);
private:
  MyMatrix& H;
  int N; //matrix with N columns
  int M; //# of Lanzcos iterations
  int Niterations;

  vector<vector<MyField>*> phi; 
  vector<double> lanczos_a; 
  vector<double> lanczos_b; 

  LanczosMatrix* LM;
  
  double initnorm;

  bool LanczosIteration(int m);
  void RandomInitialization();

  void ConstructGamma1(vector<MyField>& v1,vector<MyField>& v0,double& a0);
  void ConstructGammam(vector<MyField>& v_mp1,vector<MyField>& v_m,vector<MyField>& v_mm1,double& am,double& Nm);
  void Reorthogonalize(int m);
};


Lanczos::Lanczos(MyMatrix& H_in,int M_in):H(H_in),N(H.Ncols()),M(min(N,M_in)),Niterations(0),phi(M+1),lanczos_a(M),lanczos_b(M-1),LM(0),initnorm(1.)
  {
    if(TRACE) cout << "Starting Lanczos, M=" << M << endl;
    for(int i=0; i<M+1; i++) phi[i]=new vector<MyField>(N); // allocate space for Lanczos vectors
    RandomInitialization();
  }

Lanczos::Lanczos(MyMatrix& H_in,int M_in,int basestatenr):H(H_in),N(H.Ncols()),M(min(N,M_in)),Niterations(0),phi(M+1),lanczos_a(M),lanczos_b(M-1),LM(0),initnorm(1.)
  {
    if(TRACE) cout << "Starting Lanczos, M=" << M << " with base state" << basestatenr << endl;
    for(int i=0; i<M+1; i++) phi[i]=new vector<MyField>(N); // allocate space for Lanczos vectors
    vector<MyField>& v=*(phi[0]);

    v[basestatenr]=complex<double>(1.,0.); // initialize the vector
    initnorm=1.;
  }

Lanczos::Lanczos(MyMatrix& H_in,int M_in, vector<MyField>& vec_in):H(H_in),N(H.Ncols()),M(min(N,M_in)),Niterations(0),phi(M+1),lanczos_a(M),lanczos_b(M-1),LM(0),initnorm(1.)
  {
    if(TRACE) cout << "Starting Lanczos with initial vector, M=" << M << endl;
    for(int i=0; i<M+1; i++) phi[i]=new vector<MyField>(N); // allocate space for Lanczos vectors
    vector<MyField>& v=*(phi[0]);
    for(int i=0; i<N; i++) v[i]=vec_in[i]; // initialize the vector

    initnorm=Normalize(&N,&v[0]); // record the normalization of the init vector
    if(TRACE) cout << "Init norm is: " << initnorm << endl;
    if(abs(initnorm) < TINYNUMBER){M=0; Niterations=0;} // in case the starting vector is the zero vector
  }

Lanczos::~Lanczos()
{
  if(TRACE) cout << "Destroying class Lanczos" << endl;
  for(int i=0; i<phi.size(); i++){ delete phi[i];}
  delete LM;
}

void Lanczos::DoLanczos()
{
  int m=0;
  bool continueiteration=true;
  while(continueiteration && m<M){
    if(TRACE) cout << "Doing Lanczos step m:" << m << " M: " << M << endl;
    continueiteration=LanczosIteration(m++);
  } 
  Niterations=m; // gives the number of lanczos vectors
  if(TRACE){
    cout << "There are in all " << Niterations << " orthogonal Lanczos vectors" << endl;
    if(Niterations >0)
      {
	for(int i=0; i<Niterations-1; i++){
	  cout << "a[" << i << "]=" << lanczos_a[i] << " b[" << i << "]="<< lanczos_b[i] << endl;}
	cout << "a[" << Niterations-1 << "]=" << lanczos_a[Niterations-1] << endl;
      }
  }
}


void Lanczos::ConstructGamma1(vector<MyField>& v1,vector<MyField>& v0,double& a0)
  {
    double nega0=-a0;
    AddToVector(&N,&v1[0],&v0[0],&nega0); // v1 = v1 -a0*v0;
  }

void Lanczos::ConstructGammam(vector<MyField>& v_mp1,vector<MyField>& v_m,vector<MyField>& v_mm1,double& am,double& bmm1)
  {
    // computes v_mp1 -> v_mp1-am*v_m-bmm1*v_mm1
    
    MyField negam=-am;
    AddToVector(&N,&v_mp1[0],&v_m[0],&negam); // v_mp1=v_mp1 - am*v_m;
    MyField negbmm1=-bmm1;
    AddToVector(&N,&v_mp1[0],&v_mm1[0],&negbmm1); // v_mp1=v_mp1 - bmm1*v_mm1;
  }

void Lanczos::RandomInitialization()
  {
    vector<MyField>& v=*(phi[0]);
    for(int i=0; i<N; i++)
      {
	//	MyField r=std::complex<double>(rand(),rand()); // random start
	//	MyField r=std::complex<double>(RAN.Get(),RAN.Get()); // random start
	MyField r=std::complex<double>(2.*RAN.Get()-1.,2.*RAN.Get()-1.); // random start
	//MyField r=complex<double>(1.,1.); // random start
	v[i]=r;
      }
    double initnorm=Normalize(&N,&v[0]); // record the normalization of the random vector
    if(TRACE)
      {
	cout << "Initial vector: ";
	for(int i=0; i<N; i++){cout << v[i] << " ";}
	cout << endl;
	cout << "Norm= " << ScalarProduct(&N,&v[0],&v[0]) << endl;
      }
  }

void Lanczos::Reorthogonalize(const int m)
  {
    vector<MyField> vm=*(phi[m]);
    for(int i=0; i<m; i++)
      {
	vector<MyField> vi=*(phi[i]);
	MyField q=ScalarProduct(vm,vi);
	MyField q0=1./(1.-q*q);
	MyField q1=q0*q;
	ScaleVector(&N,&vm[0],&q0); // vm = q0*vm;
	MyField negq1=-q1;  
	AddToVector(&N,&vm[0],&vi[0],&negq1); // vm = vm -q1*vi;
      }
  }

bool Lanczos::LanczosIteration(const int m)
  {
    if(TRACE)  cout << "Starting Lanczos iteration " << m << endl;
    Niterations++;
    if(m==0)
      { // first element
	vector<MyField>& v0=*(phi[0]); //already normalized
	vector<MyField>& v1=*(phi[1]);
	//	v1=H*v0;
	H.TimesVector(v1,v0);
	lanczos_a[0]=ScalarProduct(v0,v1).real();

	if(M>1) // if not the last element
	  {
	    ConstructGamma1(v1,v0,lanczos_a[0]);
	    if(REORTHOGONALIZE) Reorthogonalize(1);
	    lanczos_b[0]=Normalize(v1);
	  }
      }
    /*
    else if(m==M-1 && M>1)
      { // last element
	if(TRACE) cout << "Last lanczos iteration" << endl;
	vector<MyField>& v_mm1=*(phi[m-1]); //already normalized
	vector<MyField>& v_m  =*(phi[m]);
	v_m=H*v_mm1;
	lanczos_a[m]=ScalarProduct(v_mm1,v_m).real();
	ConstructGamma1(v_m,v_mm1,lanczos_a[m]);
      }
    */
    else
      {
	vector<MyField>& v_mm1=*(phi[m-1]); //already normalized
	vector<MyField>& v_m  =*(phi[m  ]); //already normalized
	vector<MyField>& v_mp1=*(phi[m+1]);

	//	v_mp1=H*v_m;
	H.TimesVector(v_mp1,v_m);
	lanczos_a[m]=ScalarProduct(v_m,v_mp1).real();

	if(m < M-1) // not last element
	  {
	    ConstructGammam(v_mp1,v_m,v_mm1,lanczos_a[m],lanczos_b[m-1]);	    
	    if(REORTHOGONALIZE) Reorthogonalize(m+1);
	    lanczos_b[m]=Normalize(v_mp1);
	  }
      }


    if(TRACE){ 
      cout << "Lanczos vector after iteration #: " << m << endl;
      vector<MyField>& v=*(phi[m]);
      for(int i=0; i<N; i++){cout << v[i] << " ";}
      cout << endl;
      cout << "Norm= " << sqrt(ScalarProduct(v,v)) << endl;
      cout << "a[" << m << "]=" << lanczos_a[m];
      if(m<M-1) cout << " b[" << m << "]=" << lanczos_b[m];
      cout << endl;
    }

   
    if(m<M-1 && abs(lanczos_b[m])<=TINYNUMBER){return false;} // stop the Lanczos procedure; 
    return true;
  }

void Lanczos::DiagonalizeLanczosMatrix()
{
  if(Niterations==0) return;
  delete LM;
  LM = new LanczosMatrix(Niterations,lanczos_a,lanczos_b);
  LM->Diagonalize();
}

vector<MyField>& Lanczos::GetLanczosEigenvectorinStateSpaceBasis(const int n)
{
  vector<MyField>& v=*(phi[M]); // the last one is used as storage
  double* nu=LM->Eigenvector(n);

  for(int i=0; i<N; i++){ v[i]=0;} // initialize to zero.
  for(int p=0; p<Niterations; p++)
    {
      vector<MyField>& vp=*(phi[p]);
      AddToVector(&N,&v[0],&vp[0],&nu[p]);	 
    }
  return v;
}


#endif // LANCZOS_H



