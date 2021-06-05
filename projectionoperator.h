#ifndef PROJECTIONOPERATOR_H
#define PROJECTIONOPERATOR_H

#include<iostream>
#include<complex>
#include<vector>
#include<string>
using namespace std;

//#include "inttostring.h"

#include "sparsematrix.h"
#include "statespace.h"
#include "operator.h"



class Pop : public Operator
{
 public:
  Pop(string name_in,int q_in,unsigned long bitstring_in,int nbits_in,StateSpace& statespace_in);
  void GenerateSector(const int i);
 private:
  string name;
  int q; // momentum
  unsigned long bitstring; // pattern of spins
  int nbits;   // the number of spins in the pattern
  StateSpace& statespace;
  int N; // system size
};


Pop::Pop(string name_in,int q_in,unsigned long bitstring_in,int nbits_in,StateSpace& statespace_in):name(name_in),q(q_in),bitstring(bitstring_in),nbits(nbits_in),statespace(statespace_in),N(statespace_in.GetN()),Operator(name_in,statespace_in.Nsectors())
{
  if(WTRACE) cout << "Constructing the Pop(" << q << ") operator for bitstring " << bitstring << " nbits="<< nbits << endl;
  // first set up the sector actions
  const int Nsectors=statespace.Nsectors();
  for(int is1=0; is1<Nsectors; is1++)
    {
      Sector sec1=*statespace.sectors[is1];

      vector<int> newqnumbers(3);      

      //find the out sector
      int k1=sec1.qnumbers[MOMENTUM];
      int k2=(k1+q)%N;     // Pop increases the momentum by q

      newqnumbers[MOMENTUM]=k2;

#ifdef SPINPARITY_CONSERVED
      int sp1=sec1.qnumbers[SPINPARITY];
      int sp2=sp1;   // Sz does not change the parity
      newqnumbers[SPINPARITY]=sp2; 
#endif

#ifdef MAGNETIZATION_CONSERVED
      int ma1=sec1.qnumbers[MAGNETIZATION];
      int ma2=ma1;        // Sz does not change the magnetization
      newqnumbers[MAGNETIZATION]=ma2;
#endif
      
      int is2=statespace.FindSector(newqnumbers);
      Operator::Osector[is1]=is2;
    }
  
}

void Pop::GenerateSector(const int is1)
{

  // construct the operator for each sector

  //do not construct operator if it exists on disk
  // if(Operator::Exists(is1)){continue;}
  
  if(WTRACE) cout << "Constructing Pop(" << q << ") for sector: " << is1 << endl;
  logfile << "\nConstructing Pop(" << q << ") for sector: " << is1 << " bitstring:" << bitstring << " nbits=" << nbits << endl;
  
  // convert the bitpatterin into a string of doubles.
  vector<double> pattern(nbits);
  for(int i=0; i<nbits; i++)
    pattern[i]=( (bitstring & 1<<i)==0 ? -0.5 : 0.5);
      

  Sector sec1=*statespace.sectors[is1];
  const int Nstates1=sec1.Nstates;
  
  int is2=Operator::Osector[is1];
  Sector sec2=*statespace.sectors[is2];
  const int Nstates2=sec2.Nstates;
  
  COOmatrix A(Nstates2,Nstates1);
  
  if(TRACE) cout << "Nstates: " << Nstates1 << " >> " << Nstates2 << endl;
  
  //make table of complex exponentials
  double qv=q*2.*PI/N;
  vector<MyField> g(N);
  for(int r=0; r<N; r++){g[r]=complex<double>(cos(qv*r),-sin(qv*r));}


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

  if(TRACE) cout << "Starting construction on state " << startstate+1 << endl;
  logfile << "Starting construction on state " << startstate+1 << endl;
  
  for(int i=startstate+1; i<Nstates1; i++)
    {
      Stateref rs=sec1.state[i];
      
      pstate s=rs.s;
      double in=rs.norm;
      
      pstate news;
      int j;
      double jn;
      
      int ri=0; 
      pstate newr; // the new reference state
      
      for(int r=0; r<N; r++)
	{
	  bool keepchecking=true;
	  int b=0;
	  while(keepchecking && b<nbits)
	    {
	      if(statespace.Sz(s,(r+b)%N) != pattern[b]){keepchecking=false;}
	      b++;
	    }

	  if(keepchecking)
	    {
	      j    = sec2.FindIndx(s); 
	      if(j<Nstates2)
		{
		  jn   = sec2.state[j].norm; 
		  A.AddTo(j,i,g[r]*sqrt(jn/in)*1.);
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
  A.DeleteTinyValues(1e-16);
  if(TRACE){
    cout << "-----Sz(" << is1 << ") -------" << endl;
    cout << A << endl;
    cout << "---------------" << endl;
    
    cout << "The matrix A is " << (A.IsReal() ? "real": "complex") << endl;
  }
  matrixptr=new MyMatrix(A); // convert to the sparse format
  if(TRACE) cout << *matrixptr << endl;

  if(TRACE) cout << "Removing temp file " << tempfilename << endl;
  remove(tempfilename.c_str()); // remove the temporary file on disk      

  Operator::Save(is1); // save the operator
  Operator::Free();

  if(WTRACE) cout << "Done constructing Pop(" << q << ") for sector # " << sec1.id << endl;
  logfile << "Done constructing Pop(" << q << ") for sector # " << sec1.id << endl;


  if(TRACE) cout << "----------" << endl;
}
#endif /* PROJECTIONOPERATOR_H */


