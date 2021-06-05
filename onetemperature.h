#ifndef ONETEMPERATURE_H
#define ONETEMPERATURE_H

#include<iostream>
#include<vector>
#include<fstream>
#include<algorithm>
#include "math.h"
#include "inttostring.h"
#include "observablesspec.h"

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
  storedreal factor; // can be exchanged with float 
  void Write(ofstream&);
  void Read(ifstream&);
  storedreal Compute(storedreal&,storedreal&);
  storedreal GetEnergyMin(){return min_energy;}
  storedreal min_energy;
  void Insert(const storedreal a,const storedreal b)
  {
    if(a < min_energy){ min_energy = a;}
    if(b>MINIMUMFACTORTOSTORE || b < -MINIMUMFACTORTOSTORE)
      {
	storedreal exponent=-beta*(a-Esub);
        factor += ( exponent > EXPONENTUNDERFLOW ? nstates*b*exp(exponent): 0.);
      }
  }
 private:
  double beta;
  double Esub;
};

DataLine::DataLine(int sector_in=0,int nstates_in=0,int N=0):sector(sector_in),nstates(nstates_in),min_energy(9999999999.),factor(0.)
{
  // read the temperature and esub:
  ObservablesSpec obsspec; 
  beta=obsspec.GetBETA(); 
  Esub=obsspec.GetESUB();
}



void DataLine::Write(ofstream& outfile)
{
  if(USEBINARYSTORAGEFORMAT)
    {
      outfile.write( (char*) &sector,sizeof(sector));
      outfile.write( (char*) &nstates,sizeof(nstates));
      outfile.write( (char*) &factor,sizeof(factor));
    }
  else
    {
      outfile << sector << " " << nstates << " " << factor << endl;
    }
}


void DataLine::Read(ifstream& infile)
{
  if(USEBINARYSTORAGEFORMAT)
    {
      infile.read( (char*) &sector,sizeof(sector));
      infile.read( (char*) &nstates,sizeof(nstates));
      infile.read( (char*) &factor,sizeof(factor));
    }
  else
    {
      infile >> sector; 
      infile >> nstates; 
      infile >> factor;
    }
}

storedreal DataLine::Compute(storedreal& beta,storedreal& esub)
{
  /*
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
  */
  //  return nstates*factor;
  return 0.;
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

FrequencyBins::FrequencyBins(const int N_in,const storedreal wmin_in,const storedreal wmax_in,storedreal sigma_in=0):N(N_in),wmin(wmin_in),wmax(wmax_in),dw((wmax-wmin)/N),center_w(N_in),sigma(sigma_in),Nsmooth(6*sigma/dw +1),smoother(Nsmooth)
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
  ~CorrelationDataLine(){delete fb;}
  int sector;
  int nstates;
  int Nwbins;
  vector<storedreal> dataarray;  // can be replaced with float 
  void Write(ofstream&);
  void Read(ifstream&);
  //  void Compute(storedreal&,storedreal&,storedreal*,FrequencyBins&);
  void Insert(const storedreal erg,const storedreal w,const storedreal factor)
  {
    if(factor>MINIMUMFACTORTOSTORE || factor<-MINIMUMFACTORTOSTORE)
      { 
	int bin=fb->GetBin(w);
	
	storedreal exponent=-beta*(erg-esub);
	storedreal weight =( exponent > EXPONENTUNDERFLOW ? nstates*factor*exp(exponent): 0.);
	
	if(bin >=0 && bin < Nwbins) dataarray[bin] += weight*(*sptr); 

	for(int s=1; s<Nsmooth; s++)
	  {
	    storedreal dataval = *(sptr+s)*weight;
	    if(bin+s >=0 && bin+s < Nwbins){ dataarray[bin+s]+= dataval;}
	    if(bin-s >=0 && bin-s < Nwbins){ dataarray[bin-s]+= dataval;}
	  }
      }
  }

 private:
  double beta;
  double esub;

  FrequencyBins* fb;
  storedreal dw;
  int Nsmooth; 
  storedreal* sptr;
};

CorrelationDataLine::CorrelationDataLine(int sector_in=0,int nstates_in=0,int N=0):sector(sector_in),nstates(nstates_in),dataarray(0)
{
  // read the observables specs
  ObservablesSpec ospec;

  beta=ospec.GetBETA();
  esub=ospec.GetESUB();

  Nwbins=ospec.GetNWBINS();
  dataarray.resize(Nwbins);

  fb=new FrequencyBins(ospec.GetNWBINS(),ospec.GetWMIN(),ospec.GetWMAX(),ospec.GetSIGMA());

  dw=fb->GetDW();

  Nsmooth=fb->GetNsmooth(); 
  sptr=fb->GetSmoothPtr();

}


void CorrelationDataLine::Write(ofstream& outfile)
{
  if(USEBINARYSTORAGEFORMAT)
    {
      outfile.write( (char*) &sector,sizeof(sector));
      outfile.write( (char*) &nstates,sizeof(nstates));
      outfile.write( (char*) &Nwbins,sizeof(Nwbins));
      outfile.write( (char*) &dataarray[0],Nwbins*sizeof(dataarray[0]));
    }
  else
    {
      outfile << sector << " " << nstates << " " << Nwbins << " ";
      for(int i=0; i<Nwbins; i++){outfile << dataarray[i] << " ";}
      outfile << endl;
    }
}


void CorrelationDataLine::Read(ifstream& infile)
{
  int Nwbins=0;

  if(USEBINARYSTORAGEFORMAT)
    {
      infile.read( (char*) &sector,sizeof(sector));
      infile.read( (char*) &nstates,sizeof(nstates));
      infile.read( (char*) &Nwbins,sizeof(Nwbins));
      dataarray.resize(Nwbins);
      infile.read( (char*) &dataarray[0],Nwbins*sizeof(dataarray[0]));
    }
  else
    {
      infile >> sector; 
      infile >> nstates; 
      infile >> Nwbins; 
      dataarray.resize(Nwbins);
      for(int i=0; i<Nwbins; i++){infile >> dataarray[i];}
    }
}

/*
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
*/

//********************************************************


const int NSECTORSMAX=1000;

class OneTemperature
{
 public:
 OneTemperature(string Zfilename);
 void FindTdependence(string filename);
 void FindTdependence(ifstream& infile);
 int Nsectors;
 storedreal Z;     // Storing Z
 storedreal Diag; 
 vector<int> scounter;
 // storedreal FindEnergyMin(string filename);
 void CollectData(storedreal& data,string filename);
 void CollectData(storedreal& data,ifstream& infile);
 void DividebyZ(storedreal& data){data/=Z;}
 void Write(storedreal& data,string filename);
 void Write(storedreal& data);
};

OneTemperature::OneTemperature(string Zfilename="Z.dat"):Nsectors(NSECTORSMAX),Z(0.),Diag(0.),scounter(NSECTORSMAX)
{

  CollectData(Z,Zfilename);
  Write(Z,Zfilename+".T");
}


void OneTemperature::FindTdependence(string filename)
{
  Diag=0;

  CollectData(Diag,filename);
  DividebyZ(Diag);
  string newfilename=filename+".T";
  Write(Diag,newfilename);
}


void OneTemperature::FindTdependence(ifstream& infile)
{
  Diag=0;

  CollectData(Diag,infile);
  DividebyZ(Diag);
  Write(Diag);
}


void OneTemperature::Write(storedreal& data,string filename)
{
  ofstream outfile(filename.c_str());
  outfile << data << endl;
  outfile.close();
}


void OneTemperature::Write(storedreal& data)
{
    cout << data << endl;
}


void OneTemperature::CollectData(storedreal& data,string filename)
{
  if(TRACE) cout << "Collecting data from " << filename << endl;
  ifstream infile(filename.c_str());
  CollectData(data,infile);
  infile.close();
  if(TRACE) cout << "Done Collecting data from " << filename << endl;
}

void OneTemperature::CollectData(storedreal& data,ifstream& infile)
{
  if(TRACE) cout << "Collecting data" << endl;
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
      if(s >=NSECTORSMAX){ cerr << "WARNING sector >= NSECTORMAX" << endl;}
      if(s > sectormax){sectormax=s;}
      scounter[s]++;

      temp[s]+=datal.factor;
    }
  
  if(TRACE) cout << "finalize" << endl;
  
  for(int s=0; s<=sectormax; s++)
    {
      if(scounter[s]==0) continue;
      data+=temp[s]/(1.*scounter[s]); // Average over sector samples
    }
  
}

/*
storedreal OneTemperature::FindEnergyMin(string filename)
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
*/

//*********************************************************

//********************************************************


//const int NSECTORSMAX=500;

class CorrelationData
{
 public:
  CorrelationData(string Zfilename); // specifiy partiion function file
  ~CorrelationData(){delete fb;}
  void Compute(string filename);
  void Compute(ifstream& infile);
  int Nsectors;
  int Nwbins;
  storedreal Z;
  vector<storedreal> corrdata; 
  vector<int> scounter;
  void CollectZ(storedreal&,string filename);
  void CollectZ(storedreal&,ifstream& infile);
  void CollectData(vector<storedreal>& data,string filename);
  void CollectData(vector<storedreal>& data,ifstream& infile);
  void DividebyZ(vector<storedreal>& data)
  {for(int i=0; i<Nwbins; i++){data[i]/=Z;}}
  void Write(vector<storedreal>& data,string filename);
  void Write(vector<storedreal>& data);
  storedreal esub;
  FrequencyBins* fb;
};

CorrelationData::CorrelationData(string Zfilename="Z.dat"):Nsectors(NSECTORSMAX),fb(0),Z(0.),corrdata(0),scounter(NSECTORSMAX)
{
  // read the observables specs
  ObservablesSpec ospec;

  Nwbins=ospec.GetNWBINS();
  fb=new FrequencyBins(ospec.GetNWBINS(),ospec.GetWMIN(),ospec.GetWMAX(),ospec.GetSIGMA());

  corrdata.resize(Nwbins);

  // Read in the partition function
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
    outfile << fb->GetCenterFreq(i) << " " << data[i] << endl;
  outfile.close();
}

void CorrelationData::Write(vector<storedreal>& data)
{
  for(int i=0; i<Nwbins; i++)
    cout << fb->GetCenterFreq(i) << " " << corrdata[i] << endl;
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
      for(int j=0; j<Nwbins; j++){start[j]+=datal.dataarray[j];}
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
      temp[s]+=datal.factor;
    }
  
  for(int s=0; s<=sectormax; s++)
    {
      if(scounter[s]==0) continue;
      data+=temp[s]/(1.*scounter[s]); // Average over sector samples
    }
}

/*
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
*/


#endif // ONETEMPERATURE_H

