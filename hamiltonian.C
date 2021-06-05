#include<iostream>
#include<complex>
#include<vector>
using namespace std;

#include "sparsematrix.h"
#include "statespace.h"


class XYZmodel
{
public:
  XYZmodel(int N_in,double Jx_in,double Jy_in,double Jz_in,double hx_in,double hz_in);
  void ConstructHamiltonian(Sector& sec);
private:
  const int N;
  const double Jx;
  const double Jy;
  const double Jz;
  const double hx;
  const double hz;
  const double Ja;
  const double Jp;
};

XYZmodel::XYZmodel(int N_in,double Jx_in,double Jy_in,double Jz_in,double hx_in,double hz_in):
  N(N_in),Jx(Jx_in),Jy(Jy_in),Jz(Jz_in),hx(hx_in),hz(hz_in),Ja((Jx_in-Jy_in)/4.),Jp((Jx_in+Jy_in)/4.){}

void XYZmodel::ConstructHamiltonian(Sector& sec)
{
  cout << "Constructing Hamiltonian for sector # " << sec.id << endl;

  const int Nstates=sec.Nstates;
  
  COOmatrix H(Nstates,Nstates);

  for(int i=0; i<Nstates; i++)
    {
      Stateref rs=sec.state[i];

      pstate s=rs.s;
      double in=rs.norm;
	
      pstate news;
      int j;
      double jn;

      for(int r=0; r<N; r++)
	{
	  // go thru all terms in the Hamiltonian
	  
	  //pair operators: neighboring sites
	  int r0=r;
	  int r1=(r!=N-1 ? r+1:0);
	  double k=sec.qnumbers[MOMENTUM]*2.*PI/N;
	  double g=exp(-k*r);
	  
	  // S+S-
	  news = Splussi(Sminusi(s,r0),r1);
	  j    = sec.FindIndx(news);
	  jn   = sec.state[j].norm;
	  H(j,i)+=g*sqrt(jn/in)*(Jx+Jy)/4. ;
	  
	  // S-S+
	  news = Sminusi(Splussi(s,r0),r1);
	  j    = sec.FindIndx(news);
	  jn   = sec.state[j].norm; 
	  H(j,i)+=g*sqrt(jn/in)*(Jx+Jy)/4.;
	  
	  // S+S+
	  news = Splussi(Splussi(s,r0),r1);
	  j    = sec.FindIndx(news);
	  jn   = sec.state[j].norm; 
	  H(j,i)+=g*sqrt(jn/in)*(Jx-Jy)/4.;
	  
	  // S-S-
	  news = Sminusi(Sminusi(s,r0),r1);
	  j    = sec.FindIndx(news);
	  jn   = sec.state[j].norm; 
	  H(j,i)+=g*sqrt(jn/in)*(Jx-Jy)/4.;

	  // SzSz
	  H(i,i)+=Jz*Szi(s,r0)*Szi(s,r1);
	  
	  // S+
	  news = Splussi(s,r0);
	  j    = sec.FindIndx(news);
	  jn   = sec.state[j].norm; 
	  H(j,i)+=g*sqrt(jn/in)*hx/2.;
	  
	  // S-
	  news = Sminusi(s,r0);
	  j    = sec.FindIndx(news);
	  jn   = sec.state[j].norm; 
	  H(j,i)+=g*sqrt(jn/in)*hx/2.;
	  
	  // Sz
	  H(i,i)+=hz*Szi(s,r0);
	}      
    }
  
  H.DeleteValues(0.);
  cout << H << endl;

  cout << "Done constructing the Hamiltonian for sector # " << sec.id << endl;
}


int main()
{
  cout << "Starting" << endl;
  int N=4;
  StateSpace myspace(N,2);
  XYZmodel xxxmodel(N,1.,0.,0.,0.,0.);
  
  for(int i=0; i<myspace.Nsectors(); i++)
    {
      xxxmodel.ConstructHamiltonian(*(myspace.sectors[i]));
    }
  cout << "The end" << endl;
}



