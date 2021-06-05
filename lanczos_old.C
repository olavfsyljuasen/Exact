#ifndef LANCZOS_H
#define LANCZOS_H

#include<iostream>
#include<vector>
using namespace std;


typedef CSRmatrix MyMatrix;

//--------------------------------------------------------------
// The Lanczos routine computes the Lanczos matrix
// Diagonalization of the Lanczosmatrix must be done separately.
//--------------------------------------------------------------

class Lanczos
{
public:
  Lanczos(MyMatrix& H_in,int M_in); // initializes Lanczos with random vector
  Lanczos(MyMatrix& H_in,int M_in, vector<MyField>& vec_in); // initializes Lanczos using the input vector
  void LanczosIteration(int m);
  vector<MyField>& GetInitvector(){return phi[0];}
  vector<MyField>& GetLanczosa(){return lanczos_a;}
  vector<MyField>& GetLanczosN(){return lanczos_N;}
private:
  MyMatrix& H;
  const int N; //matrix with N columns
  const int M; //# of Lanzcos iterations

  vector<vector<MyField>*> phi; 
  vector<MyField> lanczos_a; 
  vector<MyField> lanczos_N; 

  void RandomInitialization();

  MyField ScalarProduct(vector<MyField>& v1,vector<MyField>& v2);
  double Normalize(vector<MyField>& v);
  void ConstructGamma1(vector<MyField>& v1,const vector<MyField>& v0,const MyField& a0);
  void ConstructGammam(vector<MyField>& v_mp1,const vector<MyField>& v_m,const vector<MyField>& v_mm1,const MyField& am,const MyField& Nm);
  void Reorthogonalize(int m);
};


Lanczos::Lanczos(MyMatrix& H_in,int M_in):H(H_in),M(M_in),N(H.Ncols()),phi(M+1),lanczos_a(M),lanczos_N(M+1)
  {
    cout << "Starting Lanczos" << endl;
    for(int i=0; i<M; i++) phi[i]=new vector<MyField>(N); // allocate space for Lanczos vectors
    RandomInitialization();
  }

Lanczos::Lanczos(MyMatrix& H_in,int M_in, vector<MyField>& vec_in):H(H_in),M(M_in),N(H.Ncols()),phi(M+1),lanczos_a(M),lanczos_N(M+1)
  {
    cout << "Starting Lanczos" << endl;
    for(int i=0; i<M; i++) phi[i]=new vector<MyField>(N); // allocate space for Lanczos vectors
    for(int i=0; i<N; i++) phi[0][i]=vec_in[i]; // initialize the vector
  }


MyField Lanczos::ScalarProduct(vector<MyField>& v1,vector<MyField>& v2)
  {
    MyField s;
    for(int i=0; i<N; i++){ s+= conj(v1[i])*v2[i];}
    return s;
  }

double Lanczos::Normalize(vector<MyField>& v)
  {
    MyField Nsq=ScalarProduct(v,v);
    double Norm=sqrt(Nsq.real());
    double factor = 1./Norm;
    for(int i=0; i<N; i++){ v[i]*=factor;}
    return Norm;
  }

void Lanczos::ConstructGamma1(vector<MyField>& v1,const vector<MyField>& v0,const MyField& a0)
  {
    for(int i=0; i<N; i++){ v1[i]-=a0*v0[i];}
  }

void Lanczos::ConstructGammam(vector<MyField>& v_mp1,const vector<MyField>& v_m,const vector<MyField>& v_mm1,const MyField& am,const MyField& Nm)
  {
    for(int i=0; i<N; i++){ v_mp1[i]+=(-am*v_m[i]-Nm*v_mm1[i]);}
  }

void Lanczos::RandomInitialization()
  {
    vector<MyField>& v=*(phi[0]);
    for(int i=0; i<N; i++)
      {
	MyField r=complex<double>(rand(),rand()); // random start
	//	MyField r=complex<double>(1.,1.); // random start
	v[i]=r;
      }
    double dummy=Normalize(v); // do not record the normalization of the random vector
    cout << "Initial vector: ";
    for(int i=0; i<N; i++){cout << v[i] << " ";}
    cout << endl;
    cout << "Norm= " << ScalarProduct(v,v) << endl;
  }

void Lanczos::Reorthogonalize(int m)
  {
    for(int i=1; i<m; i++)
      {
	vector<MyField> vm=*(phi[m]);
	vector<MyField> vi=*(phi[i]);
	MyField q=ScalarProduct(vm,vi);
	MyField q0=1./(1.-q*q);
	MyField q1=q*q0;
	for(int i=0; i<N; i++){vm[i]=q0*vm[i]-q1*vi[i];}
      }
  }

void Lanczos::LanczosIteration(int m)
  {
    cout << "Starting Lanczos iteration " << m << endl;
    if(m==0)
      {
	vector<MyField>& v0=*(phi[0]); //already normalized
	vector<MyField>& v1=*(phi[1]);
	v1=H*v0;
	lanczos_a[0]=ScalarProduct(v0,v1);
	ConstructGamma1(v1,v0,lanczos_a[0]);
	Reorthogonalize(1);
	lanczos_N[1]=Normalize(v1);

      }
    else
      {
	vector<MyField>& v_mm1=*(phi[m-1]); //already normalized
	vector<MyField>& v_m  =*(phi[m  ]); //already normalized
	vector<MyField>& v_mp1=*(phi[m+1]);

	v_mp1=H*v_m;
	lanczos_a[m]=ScalarProduct(v_m,v_mp1);
	ConstructGammam(v_mp1,v_m,v_mm1,lanczos_a[m],lanczos_N[m]);
	Reorthogonalize(m+1);
	lanczos_N[m+1]=Normalize(v_mp1);
      }
    cout << "vector: " << m+1 << endl;
    vector<MyField>& v=*(phi[m+1]);
    for(int i=0; i<N; i++){cout << v[i] << " ";}
    cout << endl;
    cout << "Norm= " << ScalarProduct(v,v) << endl;
  }


#endif // LANCZOS_H



