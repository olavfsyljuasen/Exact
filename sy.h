#ifndef SY_H
#define SY_H

#include<iostream>
#include<complex>
#include<vector>
#include<string>
using namespace std;

//#include "inttostring.h"

#include "sparsematrix.h"
#include "statespace.h"
#include "operator.h"


class Sy : public Operator
{
 public:
  Sy(string name_in,int q_in,StateSpace& statespace_in);
  void GenerateSector(const int i);
 private:
  string name;
  int q; // momentum
  StateSpace& statespace;
  int N; // system size
};


Sy::Sy(string name_in,int q_in,StateSpace& statespace_in):name(name_in),q(q_in),statespace(statespace_in),N(statespace_in.GetN()),Operator(name_in,statespace_in.Nsectors())
{
  if(WTRACE) cout << "Constructing the operator Sy(" << q << ")" << endl;

  // first set up the sector actions
  const int Nsectors=statespace.Nsectors();
  for(int is1=0; is1<Nsectors; is1++)
    {
      Sector sec1=*statespace.sectors[is1];

      vector<int> newqnumbers(3);      

      //find the out sector
      int k1=sec1.qnumbers[MOMENTUM];
      int k2=(k1+q)%N;     // Sy increases the momentum by q

      newqnumbers[MOMENTUM]=k2;

#ifdef SPINPARITY_CONSERVED
      int sp1=sec1.qnumbers[SPINPARITY];
      int sp2=(sp1+1)%2;   // Sy changes the parity
      newqnumbers[SPINPARITY]=sp2; 
#endif

      // Magnetization is not conserved
     
#ifdef MAGNETIZATION_CONSERVED
      cout << "in sy.h: Sy does not conserve magnetization!" << endl;
      exit(1);
      //      int ma1=sec1.qnumbers[MAGNETIZATION];
      //      int ma2=ma1+1;        
      //      newqnumbers[MAGNETIZATION]=ma2;
#endif
      
      int is2=statespace.FindSector(newqnumbers);
      Operator::Osector[is1]=is2;
    }
  
}


void Sy::GenerateSector(const int is1)
{
  // construct the operator for sector is1
  if(WTRACE) cout << "\n***Constructing Sy(" << q << ") operator for sector " << is1 << endl;
  logfile << "\n***Constructing Sy(" << q << ") operator for sector " << is1 << endl;
  //do not construct operator if it exists on disk
  //if(Operator::Exists(is1)){continue;}
      
  Sector sec1=*statespace.sectors[is1];
  const int Nstates1=sec1.Nstates;
  
  int is2=Operator::Osector[is1];
  Sector sec2=*statespace.sectors[is2];
  const int Nstates2=sec2.Nstates;
  
  COOmatrix A(Nstates2,Nstates1);
  if(TRACE) cout << "Nstates: " << Nstates1 << " >> " << Nstates2 << endl;
  
  //make table of complex exponentials
      // the exponential e^{-i(k+q)r}
  double k2=sec2.qnumbers[MOMENTUM]*2.*PI/N; // k+q
  if(TRACE) cout << "k+q=" << k2 << endl;
  vector<MyField> g(N);
  for(int r=0; r<N; r++){g[r]=complex<double>(cos(k2*r),-sin(k2*r));}
  
  // the exponential e^{-iqr}
  double qv=q*2.*PI/N; // q
  if(TRACE) cout << "q=" << qv << endl;
  vector<MyField> f(N);
  for(int r=0; r<N; r++){f[r]=complex<double>(cos(qv*r),-sin(qv*r));}
  

  int startstate=-1;
  const int savetickler=10000;   // how often to save the matrix                              
  const string tempfilename=name+"_"+int2string(is1)+".COOtemp";
  ifstream infile(tempfilename.c_str());
  if(infile)
    {
      if(TRACE) cout << "Reading temporary operator " << tempfilename << endl;
      logfile << "Reading temporary operator " << tempfilename << endl;

      infile.read((char*) &startstate,sizeof(startstate));
      A.Read(infile);
    }
  infile.close();


  const complex<double> imagunit=complex<double>(0.,1.);


  if(TRACE) cout << "Starting construction on state " << startstate+1 << endl;
  logfile << "Starting construction on state " << startstate+1 << endl;

  for(int i=startstate+1; i<Nstates1; i++)
    {
      Stateref rs=sec1.state[i];
      
      pstate s=rs.s;
      double in=rs.norm;
      
      if(TRACE) cout << "Action on state: " << bitstring(s,N) << endl;
      
      pstate news;
      int j;
      double jn;
      
      pstate newr; // the new reference state
      int ri=0;   // # of translations needed to get new ref. state
      
      
      for(int r=0; r<N; r++)
	{
	  if(TRACE) cout << "r=" << r << endl;
	  // go thru all terms in the operatr
	  news = statespace.Spluss(s,r);
	  newr = statespace.GetRefState(news,ri);
	  j    = sec2.FindIndx(newr);
	  if(j<Nstates2)
	    {
	      jn   = sec2.state[j].norm; 
	      A.AddTo(j,i,-imagunit*0.5*f[r]*g[ri]*sqrt(jn/in));
	      
	      
	      if(TRACE)
		{ cout << "S+: news: " << bitstring(news,N) << " newr: " << bitstring(newr,N) 
		       << " j:" << j << " in:" << in << " jn:" << jn << " ri: " << ri 
		       << " g[ri]=" << g[ri] << " f[r]=" << f[r] << endl;
		  cout << "A(" << j << "," << i << ")=" << A(j,i) << endl;
		}
	    }
	  news = statespace.Sminus(s,r);
	  newr = statespace.GetRefState(news,ri);
	  j    = sec2.FindIndx(newr);
	  if(j<Nstates2)
	    {
	      jn   = sec2.state[j].norm; 
	      A.AddTo(j,i,imagunit*0.5*f[r]*g[ri]*sqrt(jn/in));
	      
	      
	      if(TRACE)
		{ cout << "S-: news: " << bitstring(news,N) << " newr: " << bitstring(newr,N) 
		       << " j:" << j << " in:" << in << " jn:" << jn << " ri: " << ri 
		       << " g[ri]=" << g[ri] << " f[r]=" << f[r] << endl;
		  cout << "A(" << j << "," << i << ")=" << A(j,i) << endl;
		}
	    }
	}

      // Save temporary file                                                                 
      if(i%savetickler==0 && i!=0)
        {
          if(TRACE) cout << "Writing temporary operator for the first " << i << " states to \
disk" << endl;
          ofstream outfile(tempfilename.c_str());
          outfile.write((char*) &i,sizeof(i)); // write the number of the last state         
          A.Write(outfile);
          outfile.close();
        }
      
    }
  //  A.SetSmallImagValuesToZero(1e-15);
  A.DeleteTinyValues(1e-15);
  
  if(TRACE){
    cout << "------Sy in sector " << is1 << " is------" << endl;
    cout << A << endl;
    cout << "---------------" << endl;
  }
  
  if(TRACE) cout << "The matrix Sy is " << (A.IsReal() ? "real": "complex") << endl;
  if(TRACE) cout << "Done constructing the operator for sector # " << sec1.id << endl;
  
  matrixptr=new MyMatrix(A); // convert to the ELL format
  if(TRACE){
    cout << "The matrix in ELL format is:" << endl;
    cout << *matrixptr;
  }

  if(TRACE) cout << "Removing temp file " << tempfilename << endl;
  remove(tempfilename.c_str()); // remove the temporary file on disk   

  Operator::Save(is1); // save the operator
  Operator::Free();


  if(WTRACE) cout << "Done constructing the Sy(" << q << ") for sector # " << sec1.id << endl;
  logfile << "Done constructing the Sy(" << q << ") for sector # " << sec1.id << endl;

  if(TRACE) cout << "----------" << endl;
}

#endif /* SY_H */


