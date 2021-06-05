#ifndef ALLTEMPERATURES_H
#define ALLTEMPERATURES_H

#include<iostream>
#include<vector>
#include<fstream>
#include<algorithm>
#include "math.h"
#include "inttostring.h"

#ifdef STOREDATAASBINARIES
const bool USEBINARYSTORAGEFORMAT=true;
#else
const bool USEBINARYSTORAGEFORMAT=false; // default is text
#endif

// format of stored number: 
#ifdef STOREDATAASFLOAT
typedef float storedreal;  
#else 
typedef double storedreal; // default is double
#endif

const storedreal EXPONENTUNDERFLOW = -80.;
const storedreal MINIMUMFACTORTOSTORE= 1.e-7;

class DataLine
{
 public:
  DataLine(int,int,int);
  int sector;
  int nstates;
  vector<storedreal> energy; // can be exchanged with float 
  vector<storedreal> factor; // can be exchanged with float 
  void Write(ofstream&);
  void Read(ifstream&);
  storedreal Compute(storedreal&,storedreal&);
  storedreal GetEnergyMin(){return (energy.size() >0 ? *min_element(energy.begin(),energy.end()):999999999.);} 
  void Insert(const storedreal a,const storedreal b)
  {if(b>MINIMUMFACTORTOSTORE || b < -MINIMUMFACTORTOSTORE){energy.push_back(a); factor.push_back(b);}}
};

DataLine::DataLine(int sector_in=0,int nstates_in=0,int N=0):sector(sector_in),nstates(nstates_in),energy(N),factor(N){}


void DataLine::Write(ofstream& outfile)
{
  int N=energy.size();
  if(USEBINARYSTORAGEFORMAT)
    {
      outfile.write( (char*) &sector,sizeof(sector));
      outfile.write( (char*) &nstates,sizeof(nstates));
      outfile.write( (char*) &N,sizeof(N));
      outfile.write( (char*) &energy[0],N*sizeof(energy[0]));
      outfile.write( (char*) &factor[0],N*sizeof(factor[0]));
    }
  else
    {
      outfile << sector << " " << nstates << " " << N << " ";
      for(int i=0; i<N; i++){outfile << energy[i] << " " << factor[i] << " ";}
      outfile << endl;
    }
}


void DataLine::Read(ifstream& infile)
{
  int N=0;

  if(USEBINARYSTORAGEFORMAT)
    {
      infile.read( (char*) &sector,sizeof(sector));
      infile.read( (char*) &nstates,sizeof(nstates));
      infile.read( (char*) &N,sizeof(N));
      energy.resize(N);
      factor.resize(N);
      infile.read( (char*) &energy[0],N*sizeof(energy[0]));
      infile.read( (char*) &factor[0],N*sizeof(factor[0]));
    }
  else
    {
      infile >> sector; 
      infile >> nstates; 
      infile >> N; 
      energy.resize(N);
      factor.resize(N);
      for(int i=0; i<N; i++){infile >> energy[i]; infile>>factor[i];}
    }
}

storedreal DataLine::Compute(storedreal& beta,storedreal& esub)
{
  //  cout << "Computing" << endl;
  storedreal p=0;
  for(int i=0; i<energy.size(); i++)
    {
      storedreal exponent=-beta*(energy[i]-esub); // subtract esub to avoid big numbers
      if(exponent > EXPONENTUNDERFLOW) // only add it if not too small
	{
	  p+= factor[i]*exp(exponent);
	}
    }
  return nstates*p;
}

//**************************************************

class FrequencyBins
{
 public:
  FrequencyBins(const int N_in,const storedreal wmin_in,const storedreal wmax_in,storedreal sigma_in);
  int GetBin(const storedreal w){return static_cast<int>((w-wmin)/dw +0.5);}
  storedreal GetCenterFreq(const int i){return center_w[i];}
  storedreal GetDW(){return dw;}
  const int N;
  int GetNsmooth(){return Nsmooth;}
  storedreal* GetSmoothPtr(){return &smoother[0];}
 private:
  const storedreal wmin;
  const storedreal wmax;
  const storedreal dw;
  vector<storedreal> center_w;
  // for the smoothing function
  const storedreal sigma;      // std.dev
  const int Nsmooth;
  vector<storedreal> smoother;
};

FrequencyBins::FrequencyBins(const int N_in,const storedreal wmin_in,const storedreal wmax_in,storedreal sigma_in=0):N(N_in),wmin(wmin_in),wmax(wmax_in),dw((wmax-wmin)/N),center_w(N_in),sigma(sigma_in)
,Nsmooth(6*sigma/dw +1),smoother(Nsmooth)
{
  if(TRACE) cout << "Creating FrequencyBins" << endl;
  for(int i=0; i<N; i++){center_w[i]=wmin+dw*i;}
  if(TRACE){
    cout << "Center frequencies are:" << endl;
    for(int i=0; i<N; i++){cout << center_w[i] << " ";}
    cout << endl;
  }



  //  cerr << "Nsmooth=" << Nsmooth << endl;
  if(Nsmooth > 1)
    {
      if(TRACE) cout << "Set up smoothing function" << endl;
      storedreal factor = 1./(sqrt(2.*M_PI)*sigma);
      for(int i=0; i<Nsmooth; i++)
	{
	  storedreal x=i*dw;
	  smoother[i]=exp(-x*x/(2.*sigma*sigma))*factor; 
	  //	  cerr << "smoother[" << i << "]=" << smoother[i] << endl;
	}
    }
  else
    smoother[0]=1./dw;
}


//***************************************************

class CorrelationDataLine
{
 public:
  CorrelationDataLine(int,int,int);
  int sector;
  int nstates;
  vector<storedreal> energy;  // can be replaced with float 
  vector<storedreal> omega;   // can be replaced with float 
  vector<storedreal> factor;  // can be replaced with float 
  void Write(ofstream&);
  void Read(ifstream&);
  void Compute(storedreal&,storedreal&,storedreal*,FrequencyBins&);
  void Insert(const storedreal a,const storedreal b,const storedreal c)
  {
    if(c>MINIMUMFACTORTOSTORE || c<-MINIMUMFACTORTOSTORE)
      {energy.push_back(a); omega.push_back(b); factor.push_back(c);}
  }
};

CorrelationDataLine::CorrelationDataLine(int sector_in=0,int nstates_in=0,int N=0):sector(sector_in),nstates(nstates_in),energy(N),omega(N),factor(N){}


void CorrelationDataLine::Write(ofstream& outfile)
{
  int N=energy.size();

  if(USEBINARYSTORAGEFORMAT)
    {
      outfile.write( (char*) &sector,sizeof(sector));
      outfile.write( (char*) &nstates,sizeof(nstates));
      outfile.write( (char*) &N,sizeof(N));
      outfile.write( (char*) &energy[0],N*sizeof(energy[0]));
      outfile.write( (char*) &omega[0],N*sizeof(omega[0]));
      outfile.write( (char*) &factor[0],N*sizeof(factor[0]));
    }
  else
    {
      outfile << sector << " " << nstates << " " << N << " ";
      for(int i=0; i<N; i++){outfile << energy[i] << " " << omega[i] << " " << factor[i] << " ";}
      outfile << endl;
    }
}


void CorrelationDataLine::Read(ifstream& infile)
{
  int N=0;

  if(USEBINARYSTORAGEFORMAT)
    {
      infile.read( (char*) &sector,sizeof(sector));
      infile.read( (char*) &nstates,sizeof(nstates));
      infile.read( (char*) &N,sizeof(N));
      energy.resize(N);
      omega.resize(N);
      factor.resize(N);
      infile.read( (char*) &energy[0],N*sizeof(energy[0]));
      infile.read( (char*) &omega[0],N*sizeof(omega[0]));
      infile.read( (char*) &factor[0],N*sizeof(factor[0]));
    }
  else
    {
      infile >> sector; 
      infile >> nstates; 
      infile >> N; 
      energy.resize(N);
      omega.resize(N);
      factor.resize(N);
      for(int i=0; i<N; i++){infile >> energy[i]; infile>> omega[i]; infile>>factor[i];}
    }
}

void CorrelationDataLine::Compute(storedreal& beta,storedreal& esub,storedreal* dataarray,FrequencyBins& fb)
{
  storedreal dw=fb.GetDW();

  int Nsmooth=fb.GetNsmooth(); 
  storedreal* sptr=fb.GetSmoothPtr();

  for(int i=0; i<energy.size(); i++)
    {
      storedreal w=omega[i]; 
      int bin=fb.GetBin(w);

      storedreal exponent=-beta*(energy[i]-esub);
      storedreal weight =( exponent > EXPONENTUNDERFLOW ? nstates*factor[i]*exp(exponent): 0.);
      
      if(bin >=0 && bin < fb.N) dataarray[bin] += weight*(*sptr); 

      for(int s=1; s<Nsmooth; s++)
	{
	  storedreal dataval = *(sptr+s)*weight;
	  if(bin+s >=0 && bin+s < fb.N){ dataarray[bin+s]+= dataval;}
	  if(bin-s >=0 && bin-s < fb.N){ dataarray[bin-s]+= dataval;}
	}
    }
}


//********************************************************


const int NSECTORSMAX=60;

class AllTemperatures
{
 public:
  AllTemperatures(vector<storedreal>& beta_in,string Zfilename);
 void FindTdependence(string filename);
 void FindTdependence(ifstream& infile);
 int Nsectors;
 const int NT;             // # of temperatures
 vector<storedreal> beta;      // list of inverse T
 vector<storedreal> Zlist;     // Storing Z
 vector<storedreal> Diaglist; 
 vector<int> scounter;
 storedreal FindEnergyMin(string filename);
 void CollectData(vector<storedreal>& data,string filename);
 void CollectData(vector<storedreal>& data,ifstream& infile);
 void DividebyZ(vector<storedreal>& data){
   for(int i=0; i<NT; i++){data[i]/=Zlist[i];}}
 void Write(vector<storedreal>& data,string filename);
 void Write(vector<storedreal>& data);
 storedreal esub;
};

AllTemperatures::AllTemperatures(vector<storedreal>& beta_in,string Zfilename="Z.dat"):Nsectors(NSECTORSMAX),NT(beta_in.size()),beta(beta_in),Zlist(NT),Diaglist(NT),scounter(NSECTORSMAX),esub(0.)
{
  // Read in the partition function
  double energy_min=FindEnergyMin("Z.dat");
  //  cout << "energy_min=" << energy_min << endl;
  esub=energy_min;

  CollectData(Zlist,Zfilename);
  Write(Zlist,Zfilename+".T");
}


void AllTemperatures::FindTdependence(string filename)
{
  for(int i=0; i<NT; i++){Diaglist[i]=0.;}

  CollectData(Diaglist,filename);
  DividebyZ(Diaglist);
  string newfilename=filename+".T";
  Write(Diaglist,newfilename);
}


void AllTemperatures::FindTdependence(ifstream& infile)
{
  for(int i=0; i<NT; i++){Diaglist[i]=0.;}

  CollectData(Diaglist,infile);
  DividebyZ(Diaglist);
  Write(Diaglist);
}


void AllTemperatures::Write(vector<storedreal>& data,string filename)
{
  ofstream outfile(filename.c_str());
  for(int i=0; i<NT; i++)
    outfile << beta[i] << " " << data[i] << endl;
  outfile.close();
}


void AllTemperatures::Write(vector<storedreal>& data)
{
  for(int i=0; i<NT; i++)
    cout << beta[i] << " " << data[i] << endl;
}


void AllTemperatures::CollectData(vector<storedreal>& data,string filename)
{
  if(TRACE) cout << "Collecting data from " << filename << endl;
  ifstream infile(filename.c_str());
  CollectData(data,infile);
  infile.close();
  if(TRACE) cout << "Done Collecting data from " << filename << endl;
}

void AllTemperatures::CollectData(vector<storedreal>& data,ifstream& infile)
{
  if(TRACE) cout << "Collecting data" << endl;
  DataLine datal;
  storedreal temp[NT*Nsectors];
  for(int i=0; i<NT*Nsectors; i++){temp[i]=0.;}
  int sectormax=0;

  for(int i=0; i<Nsectors; i++){scounter[i]=0.;}

  while(infile)
    {
      datal.Read(infile);
      if(!infile) continue;
      int s=datal.sector;
      if(s >=NSECTORSMAX){ cerr << "WARNING sector >= NSECTORMAX" << endl;}
      if(s > sectormax){sectormax=s;}
      scounter[s]++;

      storedreal* start=&temp[s*NT];
      for(int i=0; i<NT; i++){start[i]+=datal.Compute(beta[i],esub);}
    }
  
  if(TRACE) cout << "finalize" << endl;
  for(int i=0; i<NT; i++)
    {
      for(int s=0; s<=sectormax; s++)
	{
	  if(scounter[s]==0) continue;
	  storedreal* start=&temp[s*NT];
	  data[i]+=start[i]/(1.*scounter[s]); // Average over sector samples
	}
    }
}

storedreal AllTemperatures::FindEnergyMin(string filename)
{
  storedreal energy_min=99999999999;
  if(TRACE) cout << "Finding the minimal energy" << endl;
  DataLine datal;

  ifstream infile(filename.c_str());

  while(infile)
    {
      datal.Read(infile);
      if(!infile) continue;

      if(datal.GetEnergyMin() < energy_min){energy_min=datal.GetEnergyMin();}
    }
  return energy_min;
}

//*********************************************************

//********************************************************


//const int NSECTORSMAX=500;

class CorrelationData
{
 public:
  CorrelationData(storedreal beta_in,FrequencyBins& fb_in,string Zfilename);
  void Compute(string filename);
  void Compute(ifstream& infile);
  storedreal beta;
  int Nsectors;
  FrequencyBins& fb;
  const int Nwbins;
  storedreal Z;
  vector<storedreal> corrdata; 
  vector<int> scounter;
  void CollectZ(storedreal&,string filename);
  void CollectZ(storedreal&,ifstream& infile);
  storedreal FindEnergyMin(string);
  void CollectData(vector<storedreal>& data,string filename);
  void CollectData(vector<storedreal>& data,ifstream& infile);
  void DividebyZ(vector<storedreal>& data)
  {for(int i=0; i<Nwbins; i++){data[i]/=Z;}}
  void Write(vector<storedreal>& data,string filename);
  void Write(vector<storedreal>& data);
  storedreal esub;
};

CorrelationData::CorrelationData(storedreal beta_in,FrequencyBins& fb_in,string Zfilename="Z.dat"):beta(beta_in),Nsectors(NSECTORSMAX),fb(fb_in),Nwbins(fb.N),Z(0.),corrdata(Nwbins),scounter(NSECTORSMAX),esub(0.)
{
  // Read in the partition function
  double energy_min=FindEnergyMin("Z.dat");
  //  cout << "energy_min=" << energy_min << endl;
  esub=energy_min; // subtract the minimum energy so as not to get to large exponentials.

  CollectZ(Z,Zfilename);
}


void CorrelationData::Compute(string filename)
{
  for(int i=0; i<Nwbins; i++){corrdata[i]=0.;}

  CollectData(corrdata,filename);
  DividebyZ(corrdata);
  string newfilename=filename+".T";
  Write(corrdata,newfilename);
}


void CorrelationData::Compute(ifstream& infile)
{
  for(int i=0; i<Nwbins; i++){corrdata[i]=0.;}

  CollectData(corrdata,infile);
  DividebyZ(corrdata);
  Write(corrdata);
}


void CorrelationData::Write(vector<storedreal>& data,string filename)
{
  string newfilename=filename+".corr";
  ofstream outfile(newfilename.c_str());
  
  for(int i=0; i<Nwbins; i++)
    outfile << fb.GetCenterFreq(i) << " " << data[i] << endl;
  outfile.close();
}

void CorrelationData::Write(vector<storedreal>& data)
{
  for(int i=0; i<Nwbins; i++)
    cout << fb.GetCenterFreq(i) << " " << corrdata[i] << endl;
}

void CorrelationData::CollectData(vector<storedreal>& data,string filename)
{
  if(TRACE) cout << "Collecting data from " << filename << endl;
  ifstream infile(filename.c_str());
  CollectData(data,infile);
  infile.close();
}

void CorrelationData::CollectData(vector<storedreal>& data,ifstream& infile)
{
  CorrelationDataLine datal;
  storedreal temp[Nsectors*Nwbins];
  for(int i=0; i<Nsectors*Nwbins; i++){temp[i]=0.;}
  int sectormax=0;
  
  for(int i=0; i<Nsectors; i++){scounter[i]=0.;}
  
  while(infile)
    {
      datal.Read(infile);
      if(!infile) continue;

      int s=datal.sector;
      if(s >=NSECTORSMAX){ cerr << "WARNING sector >= NSECTORMAX, Not counted!" << endl; continue;}
      if(s > sectormax){sectormax=s;}
      scounter[s]++;
      storedreal* start=&temp[s*Nwbins];
      datal.Compute(beta,esub,start,fb);
    }
  

  for(int s=0; s<=sectormax; s++)
    {
      if(scounter[s]==0) continue;
      storedreal* start=&temp[s*Nwbins];
      for(int j=0; j<Nwbins; j++){
	data[j]+=start[j]/(1.*scounter[s]);
      }
    }
}


void CorrelationData::CollectZ(storedreal& data,string filename)
{
  if(TRACE) cout << "Collecting data from " << filename << endl;
  ifstream infile(filename.c_str());
  CollectZ(data,infile);
  infile.close();
}

void CorrelationData::CollectZ(storedreal& data,ifstream& infile)
{
  DataLine datal;
  storedreal temp[Nsectors];
  for(int i=0; i<Nsectors; i++){temp[i]=0.;}
  int sectormax=0;

  for(int i=0; i<Nsectors; i++){scounter[i]=0.;}

  while(infile)
    {
      datal.Read(infile);
      if(!infile) continue;

      int s=datal.sector;
      if(s >=NSECTORSMAX){ cerr << "WARNING sector >= NSECTORMAX" << endl; continue;}
      if(s > sectormax){sectormax=s;}
      scounter[s]++;
      temp[s]+=datal.Compute(beta,esub);
    }
  
  for(int s=0; s<=sectormax; s++)
    {
      if(scounter[s]==0) continue;
      data+=temp[s]/(1.*scounter[s]); // Average over sector samples
    }
}


storedreal CorrelationData::FindEnergyMin(string filename)
{
  storedreal energy_min=99999999999;
  if(TRACE) cout << "Finding the minimal energy" << endl;
  DataLine datal;

  ifstream infile(filename.c_str());

  while(infile)
    {
      datal.Read(infile);
      if(!infile) continue;

      if(datal.GetEnergyMin() < energy_min){energy_min=datal.GetEnergyMin();}
    }
  return energy_min;
}



#endif // ALLTEMPERATURES_H

