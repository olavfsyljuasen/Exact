#ifndef XYZHAMILTONIAN1D_h
#define XYZHAMILTONIAN1D_h

#include<iostream>
#include<complex>
#include<vector>
#include "stdio.h" // for remove file command
using namespace std;

#include "sparsematrix.h"
#include "statespace.h"
#include "operator.h"

class XYZhamiltonian1d: public Operator
{
public:
  XYZhamiltonian1d(string name_in,double Jx_in,double Jy_in,double Jz_in,double hx_in,double hz_in,double Jz2_in, double C_in,StateSpace& statespace_in);
  void GenerateSector(const int i);
private:
  string name;
  StateSpace& statespace;  
  const int N;
  const double Jx;
  const double Jy;
  const double Jz;
  const double hx;
  const double hz;
  const double Jz2;
  const double C;

};

XYZhamiltonian1d::XYZhamiltonian1d(string name_in,double Jx_in,double Jy_in,double Jz_in,double hx_in,double hz_in,double Jz2_in,double C_in,StateSpace& statespace_in):
name(name_in),statespace(statespace_in),N(statespace_in.GetN()),
  Jx(Jx_in),Jy(Jy_in),Jz(Jz_in),hx(hx_in),hz(hz_in),Jz2(Jz2_in),C(C_in),
  Operator(name_in,statespace_in.Nsectors())
{
  if(WTRACE){ cout << "Starting XYZhamiltonian1d" << endl;}
  
  // first set up the sector actions
  const int Nsectors=statespace.Nsectors();
  for(int is1=0; is1<Nsectors; is1++)
    {
      Operator::Osector[is1]=is1; // the Hamiltonian is diagonal in sectors
    }
}


void XYZhamiltonian1d::GenerateSector(const int is1) 
{ 
  // construct the operator for sector is1

  if(WTRACE) cout << "*** Constructing Hamiltonian for sector " << is1 << endl;
  logfile << "\n*** Constructing Hamiltonian for sector " << is1 << endl;
  
  Sector sec=*statespace.sectors[is1];
  const int Nstates=sec.Nstates;
  
  COOmatrix H(Nstates,Nstates);
  
  
  //make table of complex exponentials
  double k=sec.qnumbers[MOMENTUM]*2.*PI/N;
  if(TRACE)  cout << "k=" << k << endl;
  vector<MyField> g(N);
  for(int r=0; r<N; r++){g[r]=complex<double>(cos(k*r),-sin(k*r));}
  

  int startstate=-1;
  const int savetickler=10000;   // how often to save the matrix
  const string tempfilename=name+"_"+int2string(is1)+".COOtemp";
  ifstream infile(tempfilename.c_str());
  if(infile)
    {
      if(TRACE) cout << "Reading temporary operator " << tempfilename << endl; 
      logfile << "Reading temporary operator " << tempfilename << endl;
      infile.read((char*) &startstate,sizeof(startstate));
      H.Read(infile);
    }
  infile.close();

  
  if(TRACE) cout << "Starting construction on state " << startstate+1 << endl;
  logfile << "Starting construction on state: " << startstate+1 << endl;
  for(int i=startstate+1; i<Nstates; i++)
    {

      Stateref rs=sec.state[i];
      
      pstate s=rs.s;
      double in=rs.norm; 
      
      pstate news;
      int j;
      double jn;
      
      pstate newr; // the new reference state
      int ri=0;    // the number of translations needed to get the new ref.state
      
      if(TRACE) cout << "Action on state: " << bitstring(s,N) << endl;
      
      for(int r=0; r<N; r++)
	{
	  // go thru all terms in the Hamiltonian
	  
	  //pair operators: neighboring sites
	  int r0=r;
	  //	  int r1=(r!=N-1 ? r+1:0);
	  int r1=(r+1)%N;
	  int r2=(r+2)%N;
	  
	  if(TRACE) cout << "r0=" << r0 << " r1=" << r1 << endl;
	  
	  
	  // S+S-
	  news = statespace.Spluss(statespace.Sminus(s,r0),r1);
	  newr = statespace.GetRefState(news,ri);
	  j    = sec.FindIndx(newr);
	  if(j<Nstates)
	    {
	      jn   = sec.state[j].norm;
	      H.AddTo(j,i,g[ri]*sqrt(jn/in)*(Jx+Jy)/4.);
	      if(TRACE){
		cout << "S+S-: news: " << bitstring(news,N) << " newr: " << bitstring(newr,N) 
		     << " j:" << j << " jn:" << jn << " ri: " << ri 
		     << " g[ri]=" << g[ri] << endl;
		cout << "H(" << j << "," << i << ")=" << H(j,i) << endl;
	      }
	    }
	  // S-S+
	  news = statespace.Sminus(statespace.Spluss(s,r0),r1);
	  newr = statespace.GetRefState(news,ri);
	  j    = sec.FindIndx(newr);
	  if(j<Nstates)
	    {
	      jn   = sec.state[j].norm; 
	      H.AddTo(j,i,g[ri]*sqrt(jn/in)*(Jx+Jy)/4.);
	      if(TRACE){
		cout << "S-S+: news: " << bitstring(news,N) << " newr: " << bitstring(newr,N) 
		     << " j:" << j << " jn:" << jn << " ri: " << ri 
		     << " g[ri]=" << g[ri] << endl;
		cout << "H(" << j << "," << i << ")=" << H(j,i) << endl;
	      }  
	    }
	  
	  // S+S+
	  news = statespace.Spluss(statespace.Spluss(s,r0),r1);
	  newr = statespace.GetRefState(news,ri);
	  j    = sec.FindIndx(newr);
	  if(j<Nstates)
	    {
	      jn   = sec.state[j].norm; 
	      H.AddTo(j,i,g[ri]*sqrt(jn/in)*(Jx-Jy)/4.);
	      
	      if(TRACE){
		cout << "S+S+: news: " << bitstring(news,N) << " newr: " << bitstring(newr,N) 
		     << " j:" << j << " jn:" << jn << " ri: " << ri 
		     << " g[ri]=" << g[ri] << endl;
		cout << "H(" << j << "," << i << ")=" << H(j,i) << endl;
	      }
	    }
	  // S-S-
	  news = statespace.Sminus(statespace.Sminus(s,r0),r1);
	  newr = statespace.GetRefState(news,ri);
	  j    = sec.FindIndx(newr);
	  if(j<Nstates)
	    {
	      jn   = sec.state[j].norm; 
	      H.AddTo(j,i,g[ri]*sqrt(jn/in)*(Jx-Jy)/4.);
	      if(TRACE){
		cout << "S-S-: news: " << bitstring(news,N) << " newr: " << bitstring(newr,N) 
		     << " j:" << j << " jn:" << jn << " ri: " << ri 
		     << " g[ri]=" << g[ri] << endl;
		cout << "H(" << j << "," << i << ")=" << H(j,i) << endl;
	      }
	    }
	  
	  // SzSz
	  H.AddTo(i,i,Jz*statespace.Sz(s,r0)*statespace.Sz(s,r1));
	  
	  if(TRACE){
	    cout << "SzSz: news: " << bitstring(s,N) 
		 << " newr: " << bitstring(s,N) << endl;
	    cout << "H(" << i << "," << i << ")=" << H(i,i) << endl;
	  }
	  
	  
	  // S+
	  news = statespace.Spluss(s,r0);
	  newr = statespace.GetRefState(news,ri);
	  j    = sec.FindIndx(newr);
	  if(j<Nstates)
	    {
	      jn   = sec.state[j].norm; 
	      H.AddTo(j,i,g[ri]*sqrt(jn/in)*(-hx/2.));
	      if(TRACE){
		cout << "S+: news: " << news << " " << bitstring(news,N) << " newr: " << bitstring(newr,N) 
		     << " j:" << j << " jn:" << jn << " ri: " << ri 
		     << " g[ri]=" << g[ri] << endl;
		cout << "H(" << j << "," << i << ")=" << H(j,i) << endl;
	      }
	    }
	  
	  // S-
	  news = statespace.Sminus(s,r0);
	  newr = statespace.GetRefState(news,ri);
	  j    = sec.FindIndx(newr);
	  if(j<Nstates)
	    {
	      jn   = sec.state[j].norm; 
	      H.AddTo(j,i,g[ri]*sqrt(jn/in)*(-hx/2.));
	      if(TRACE){
		cout << "S-: news: " << bitstring(news,N) << " newr: " << bitstring(newr,N) 
		     << " j:" << j << " jn:" << jn << " ri: " << ri 
		     << " g[ri]=" << g[ri] << endl;
		cout << "H(" << j << "," << i << ")=" << H(j,i) << endl;
	      }
	    }
	  
	  // Sz
	  H.AddTo(i,i,-hz*statespace.Sz(s,r0));  // if hz >0 the spins will point in the positive z-direction
	  
	  if(TRACE){
	    cout << "Sz: news: " << bitstring(s,N) 
		 << " newr: " << bitstring(s,N) << endl;
	    cout << "H(" << i << "," << i << ")=" << H(i,i) << endl;
	  }

	  // Sb next-nearest neighbor
	  H.AddTo(i,i,Jz2*statespace.Sz(s,r0)*statespace.Sz(s,r2));
	  
	  if(TRACE){
	    cout << "Jz2: news: " << bitstring(s,N) 
		 << " newr: " << bitstring(s,N) << endl;
	    cout << "H(" << i << "," << i << ")=" << H(i,i) << endl;
	  }

	  
	  // Constant
	  H.AddTo(i,i,C);
	  
	  if(TRACE){
	    cout << "C: news: " << bitstring(s,N) 
		 << " newr: " << bitstring(s,N) << endl;
	    cout << "H(" << i << "," << i << ")=" << H(i,i) << endl;
	  }
	  
	  if(TRACE) cout << "The Hamiltonian is now: " << H << endl;
	}

      // Save temporary file      
      if(i%savetickler==0 && i!=0)
	{
	  if(TRACE) cout << "Writing temporary operator for the first " << i << " states to disk" << endl; 
	  ofstream outfile(tempfilename.c_str());
	  outfile.write((char*) &i,sizeof(i)); // write the number of the last state
	  H.Write(outfile);
	  outfile.close();
	}
    }

  
  //  H.SetSmallImagValuesToZero(1e-16);
  H.DeleteTinyValues(1e-16);
  if(TRACE){
    cout << "-----H in sector " << sec.id << " is ----- " << endl;
    cout << H << endl;
    cout << "---------------" << endl;
  }
  if(TRACE)
    cout << "The matrix H is " << (H.IsReal() ? "real": "complex") << endl;
  

  matrixptr=new MyMatrix(H); // convert to the sparse format
  if(TRACE){
    cout << "The matrix in MyMatrix format is:" << endl;
    cout << *matrixptr << endl;
  }

  if(TRACE) cout << "Removing temp file " << tempfilename << endl;
  remove(tempfilename.c_str()); // remove the temporary file on disk

  Operator::Save(is1); // save the operator
  Operator::Free();

  if(WTRACE)
    cout << "Done constructing the Hamiltonian for sector # " << sec.id << endl;

  logfile << "Done constructing the Hamiltonian for sector # " << sec.id << endl;
  
  if(TRACE) cout << "----------" << endl;
}


#endif // XYZHAMILTONIAN1D_H
