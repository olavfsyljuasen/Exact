#ifndef STATESPACE_H
#define STATESPACE_H

#include<iostream>
#include<vector>
#include<algorithm>
#include<cstdlib>
#include "math.h"

#if defined BITREPRESENTATION
#include "bitmanip.h"
#elif defined DOMAINREPRESENTATION
#include "twodomainstatesmanip.h"
#endif

using namespace std;

const double PI=M_PI;

#ifdef DEBUG2
const bool WRITEOUTALLSTATES=true;
#else
const bool WRITEOUTALLSTATES=false;
#endif

const pstate zerostate=pstate0;

enum conslaws{MOMENTUM,SPINPARITY,MAGNETIZATION};
const int MAXCONSLAWS=3;

#ifdef SPINPARITY_CONSERVED
const bool cons_spinparity=true;
#else
const bool cons_spinparity=false;
#endif

#ifdef MAGNETIZATION_CONSERVED
const bool cons_magnetization=true;
#else
const bool cons_magnetization=false;
#endif



class Stateref
{
  friend ostream& operator<<(ostream& os,const Stateref& r){os << "(" << bitstring(r.s,8)
 << "," << r.norm << ")"; return os;}
public:
  Stateref(pstate a,double b):s(a),norm(b){}
  pstate s;
  int norm;
  friend bool operator==(const Stateref& l,const pstate& p){return l.s==p;}
  friend bool operator==(const Stateref& l,const Stateref& r){return l.s==r.s;}
  friend bool operator<(const Stateref& l,const pstate& p){return l.s<p;}
  friend bool operator<(const Stateref& l,const Stateref& r){return l.s<r.s;}
};


class Sector
{
  friend ostream& operator<<(ostream& os,const Sector& r)
  { 
    os << r.id << " ("; for(int i=0; i<MAXCONSLAWS; i++){ os << r.qnumbers[i] << " ";} os << ")";
    os << " #states:" << r.Nstates << " "; 
    if(TRACE && WRITEOUTALLSTATES)
      for(int i=0; i<r.Nstates; i++){ os << r.state[i] << " ";} 
    return os;
  }
  friend bool operator==(const Sector& l,const vector<int>& p)
  {
    bool retval=true;
    for(int i=0; i<l.qnumbers.size(); i++){retval &= (l.qnumbers[i]==p[i]);} 
    return retval;
  }
  friend bool operator==(const Sector* l,const vector<int>& p)
  {
    bool retval=true;
    for(int i=0; i<l->qnumbers.size(); i++){retval &= (l->qnumbers[i]==p[i]);} 
    return retval;
  }
public:
 Sector(vector<int> qvals,vector<Stateref> sf,int id_in):qnumbers(qvals),Nstates(sf.size()),state(sf),id(id_in){}
  int FindIndx(const pstate);
  const int id;
  const int Nstates;
  vector<int> qnumbers;
  vector<Stateref> state;
};


int Sector::FindIndx(const pstate s)
{
  vector<Stateref>::iterator iter=find(state.begin(),state.end(),s);
  return iter-state.begin();
}


class StateSpace
{
public:
  StateSpace(const int,const int);
  ~StateSpace();
  
  vector<pstate> InsertDomains(const vector<pstate>);
  vector<pstate> Negate(const vector<pstate>);
  int Nstates(){return list.size();}

  int Nsectors(){return Ntotsectors;}
  vector<Sector*> sectors;
  int FindSector(const vector<int>& v);

  int CalculateNorm(const pstate,vector<int>);

  pstate GetRefState(const pstate& s,int& nt);
  pstate GetRefState(const pstate& s){int dummy; return GetRefState(s,dummy);} 
  int GetN(){return N;}

  pstate Spluss(pstate s,int i){return FlipSpinUp(s,i,N);}
  pstate Sminus(pstate s,int i){return FlipSpinDown(s,i,N);}
  double Sz(pstate s, int i){return 0.5*SpinDirection(s,i,N);}

private:
  const int N;
  const int Ndw;
  const pstate zerostate;
  const pstate maxstate;
  vector<pstate> list; 
  vector<int> Nconsvalues; // the number of values for each conserved quantity
  int Ntotsectors;  // the total number of state sectors

  //pstate Translate(pstate s){return cyclicshiftright(s,N);}
  //pstate Translaten(pstate s,int n){return cyclicshiftnright(s,n,N);}
  pstate Translate(pstate s){return cyclicshiftleft(s,N);}
  pstate Translaten(pstate s,int n){return cyclicshiftnleft(s,n,N);}
  pstate Spaceinversion(pstate s,int n){return (n==0 ? s : reflect(s,N));}
  pstate Spininversion(pstate s,int n){return  (n==0 ? s : invert(s,N));}

  int Nspinups(pstate s){return bitcount(s,N);}
  int Ndwalls(pstate s){return Nbitwalls(s,N);}
  int SpinParity(pstate s){return Nspinups(s)%2;}

};

StateSpace::StateSpace(const int n_in,const int ndw_in):N(n_in),Ndw(ndw_in),zerostate(pstate0),maxstate(invert(pstate0,n_in)),Nconsvalues(MAXCONSLAWS), Ntotsectors(1)
{
  if(TRACE) cout << "Starting StateSpace" << endl;

  logfile << "Constructing domain-wall statespace for N=" << N << ", NDW=" << Ndw << endl; 
  //  if(N > 8*sizeof(pstate)-1){ cout << "ERROR: Exiting because system size exceeds bit size of pstate" << endl; exit(1);}  

  list.push_back(zerostate);   // the zero state
  vector<pstate> oldlist=list;
  for(int d=0; d<Ndw/2; d++)  // go thru all domain insertions and add to list
    {
      vector<pstate> newlist=InsertDomains(oldlist);
      list.insert(list.end(),newlist.begin(),newlist.end()); // put the result into list
      oldlist=newlist;
    }

  vector<pstate> negstates=Negate(list); // also include the spin-reversed states 
  list.insert(list.end(),negstates.begin(),negstates.end());

  //  sort(list.begin(),list.end());


  /*
  cout << "Total number of states : " << list.size() << endl; 

  for(int i=0; i<list.size(); i++)
    {
      cout << bitstring(list[i],2) << endl;
    }

  */


  /*
  for(int item=0; item<list.size(); item++)
    {
      pstate b=list[item];
      cout << "b: " << bitstring(b,N) << " Ndwalls: " << Ndwalls(b) << endl;
    }
  */

  // Group the states into sectors
  // Set up the different sectors: 
  Nconsvalues[MOMENTUM     ]= N;  // for systems with conserved moementum
  Nconsvalues[MAGNETIZATION]=( cons_magnetization  ? 2*N-1:1);   // The number of values for each conservation law
  Nconsvalues[SPINPARITY   ]=( cons_spinparity     ? 2:1);

  for(int i=0; i<MAXCONSLAWS; i++) Ntotsectors*=Nconsvalues[i]; // calc. the total number of sectors

  logfile << "Ntotsectors: " << Ntotsectors << endl;
  // Set up the states in each sector
  int sectorid=0;
  vector<int> cval(MAXCONSLAWS);
  for(cval[0]=0; cval[0]<Nconsvalues[0]; cval[0]++){
    for(cval[1]=0; cval[1]<Nconsvalues[1]; cval[1]++){
      for(cval[2]=0; cval[2]<Nconsvalues[2]; cval[2]++){
	vector<Stateref> slist; // sector statelist
	// for each sector go thru list of domainwall states
	for(int i=0; i<list.size(); i++)
	  {
	    pstate s=list[i];
	    // only consider states with correct magnetization and spin parity
	    if(cons_magnetization && Nspinups(s) != cval[MAGNETIZATION]) continue;
	    if(cons_spinparity    && SpinParity(s) != cval[SPINPARITY]) continue;
	    
	    pstate state=GetRefState(s);
	    vector<Stateref>::iterator siter=find(slist.begin(),slist.end(),state);
	    if(siter != slist.end()){continue; } // state found continue
	    else  // state not found, insert new state if it has nonzero norm
	      {
		double norm=CalculateNorm(state,cval);
		//		cout << "state: " << state << " " << bitstring(state,2) << " norm:" << norm << endl;
		if(norm !=0)
		  {
		    Stateref newstate(state,norm);
		    slist.insert(lower_bound(slist.begin(),slist.end(),state),newstate);
		  }
	      }
	  }
	Sector* sec=new Sector(cval,slist,sectorid++); // make sector pointer
	sectors.push_back(sec); // put it onto the sector list.

	logfile << "Sector " << *sec << endl;
      }
    }
  }



 
  if(WRITEOUTALLSTATES)
    {
      ofstream ofile("allstates.dat");
      
      if(TRACE) cout << "Writing out all the states to the file allstates.dat" << endl;
      for(int i=0; i<sectors.size(); i++)
	{
	  ofile << "sector " << i << ":" << endl;
	  ofile << *sectors[i] << endl;
	}
      ofile.close();
    }

  if(TRACE) cout << "Done constructing StateSpace" << endl;      
}

StateSpace::~StateSpace()
{
  if(TRACE) cout << "Destroying class StateSpace" << endl;
  for(int i=0; i<sectors.size(); i++){delete sectors[i];}
}


pstate StateSpace::GetRefState(const pstate& s,int& nt)
{
  if(s == NOSTATE) return NOSTATE;
  // do all symmetry transformations and return the state with minimum refnumber
  // nt contains the number of translations needed to transform the old state into the ref state. 
  pstate rstate=maxstate; // the highest possible state
  pstate tstate=s;

  for(int i=0; i<N; i++)
    {
      tstate=Translaten(s,i); 
      if(tstate < rstate){rstate=tstate; nt=i;}
    }  
  return rstate;
}



int StateSpace::CalculateNorm(const pstate s,vector<int> svals)
{
  // find the first occurence of the original state
  int r=0;
  while(r<N)
    {
      pstate t=Translaten(s,++r);
      //      cout << "s: " << bitstring(s,2) << " translated " << r << " steps: " << bitstring(t,2) << endl; 
      if(t==s) break;
    }

  if( svals[MOMENTUM]%(N/r) !=0) return 0;   // check if periodicity is compatible with momentum. if not return 0

  return (N*N)/r;
}

vector<pstate> StateSpace::InsertDomains(const vector<pstate> inlist)
{
  vector<pstate> tlist;
  for(int item=0; item<inlist.size(); item++)
    {
      const pstate instate=inlist[item];
      const int k=maxbitset(instate,N);
      for(int m=2; m< N-k+1; m++)
	for(int l=1; l<N+1-k-m; l++){ tlist.push_back(setbits(instate,k+m,l,N));}
	    //      for(int l=1; l<N-k-1; l++){ tlist.push_back(setbits(instate,k+2,l));}
    }
  return tlist;
}

vector<pstate> StateSpace::Negate(const vector<pstate> inlist)
{
  vector<pstate> tlist;
  for(int item=0; item<inlist.size(); item++)
    {
      tlist.push_back(invert(inlist[item],N));
    }
  return tlist;
}

int StateSpace::FindSector(const vector<int>& v)
{
  vector<Sector*>::iterator iter=find(sectors.begin(),sectors.end(),v);
  return iter-sectors.begin();
}
    



#endif /* STATESPACE_H */
