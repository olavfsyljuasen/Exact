#include<complex>
#include<iostream>
#include<vector>
#include<algorithm>
#include<fstream>
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

const double TINYVALUE=1.e-16; // the lower cutoff on finite values

#include "sparsematrix.h"

#include "global.h"
//COOmatrix is sparse density matrix which is easy to construct
//it is simply a list of Tuples which has indices and a value.

// The matrix can come from file
COOmatrix::COOmatrix(istream& is):M(0),N(0),list(0),dummy(0){Read(is);}


MyField& COOmatrix::operator()(const int r,const int c)
{
  if( r>=M || c>=N || r<0 || c<0){return dummy;} //write to dummy if one writes outside matrix dim.
  Tuple t(r,c);
  vector<Tuple>::iterator iter=find(list.begin(),list.end(),t);
  if(iter != list.end())
    { // found element
      return (*iter).a; 
    }
  else
    {
      // insert element
      iter=lower_bound(list.begin(),list.end(),t);
      iter=list.insert(iter,t); // insert return iterator to inserted element
      return (*iter).a;
    }
}  


// dedicated AddTo routine 
void COOmatrix::AddTo(const int r,const int c,MyField a_in)
{
  if( r>=M || c>=N || r<0 || c<0){return;} //return if one writes outside matrix dim.


  // eliminate very small imaginary parts:
  if(abs(a_in.imag()) < TINYVALUE){ a_in=complex<double>(a_in.real(),0.);}
  // insert only if value is bigger than a tiny value: 
  if(abs(a_in) > TINYVALUE)
    {
      Tuple t(r,c);
      vector<Tuple>::iterator iter=find(list.begin(),list.end(),t);
      if(iter == list.end())
	{
	  // element was not found, insert new element
	  iter=lower_bound(list.begin(),list.end(),t);
	  iter=list.insert(iter,t); // insert return iterator to inserted element
	}
      // add to value:
      (*iter).a += a_in;
    }
}  



MyField COOmatrix::Get(const int r,const int c)
{
  if( r>=M || c>=N || r<0 || c<0){return dummy;} //write to dummy if one writes outside matrix dim.
  Tuple t(r,c);
  vector<Tuple>::iterator iter=find(list.begin(),list.end(),t);
  if(iter != list.end())
    { // found element
      return (*iter).a; 
    }
  else
    return 0.;
}


void COOmatrix::Read(istream& is)
{
  is.read( (char*) &M,sizeof(M));
  is.read( (char*) &N,sizeof(N));
  int Nelem=0;
  is.read( (char*) &Nelem,sizeof(Nelem));
  list.resize(Nelem);
  is.read( (char*) &list[0],Nelem*sizeof(list[0]));
  is.read( (char*) &dummy,sizeof(dummy));
}

ostream& COOmatrix::Write(ostream& os)
{
  os.write( (char*) &M,sizeof(M));
  os.write( (char*) &N,sizeof(N));
  int Nelem=list.size();
  os.write( (char*) &Nelem,sizeof(Nelem));
  os.write( (char*) &list[0],Nelem*sizeof(list[0]));
  os.write( (char*) &dummy,sizeof(dummy));
  return os;
}


//*****************************************************************



CSRmatrix::CSRmatrix(istream& is):M(0),N(0),Ntot(0),col(0),rp(0),list(0)
#ifdef SMALLNUMBEROFENTRYVALUES
  ,entryvalue(0)
#endif
{
  Read(is);  // The matrix can also come from file
}

CSRmatrix::CSRmatrix(COOmatrix& Min):M(Min.Nrows()),N(Min.Ncols()),Ntot(Min.Nelements()),col(Ntot),rp(M),list(Ntot)
#ifdef SMALLNUMBEROFENTRYVALUES
  ,Nvals(0),entryvalue(0)
#endif
{
  int old_row_indx=-1;
  int indx=0;
  
  for(int i=0; i<Min.Nelements(); i++)
    {
      int row_indx=Min.item(i).row;
      int col_indx=Min.item(i).col;
      MyField value=Min.item(i).a;

     col[i]=col_indx;
#ifdef SMALLNUMBEROFENTRYVALUES
 
      unsigned short int k=0;
      bool found=false;
      while(k<entryvalue.size())
	{ if(entryvalue[k++] == value){found=true; break;}}
      list[i]=k;
      if(!found){entryvalue.push_back(value); Nvals++;}
#else
      list[i]=value;
#endif
      if(old_row_indx != row_indx)
	{
	  for(int j=old_row_indx+1; j<M; j++){rp[j]=indx;} 
	  old_row_indx=row_indx;
	}
      indx++;
    }
  for(int j=old_row_indx+1; j<M; j++){rp[j]=indx;} 
  Ntot=indx;
}


void CSRmatrix::Read(istream& is)
{
  is.read( (char*) &M,sizeof(M));
  is.read( (char*) &N,sizeof(N));
  is.read( (char*) &Ntot,sizeof(Ntot));
  col.resize(Ntot);
  is.read( (char*) &col[0],Ntot*sizeof(col[0]));
  rp.resize(M);
  is.read( (char*) &rp[0],M*sizeof(rp[0]));
  list.resize(Ntot); 
  is.read( (char*) &list[0],Ntot*sizeof(list[0]));
#ifdef SMALLNUMBEROFENTRYVALUES
  is.read( (char*) &Nvals,sizeof(Nvals));
  entryvalue.resize(Nvals);
  is.read( (char*) &entryvalue[0],Nvals*sizeof(entryvalue[0]));
#endif  
}

ostream& CSRmatrix::Write(ostream& os)
{
  os.write( (char*) &M,sizeof(M));
  os.write( (char*) &N,sizeof(N));
  os.write( (char*) &Ntot,sizeof(Ntot));
  os.write( (char*) &col[0],Ntot*sizeof(col[0]));
  os.write( (char*) &rp[0],M*sizeof(rp[0]));
  os.write( (char*) &list[0],Ntot*sizeof(list[0]));
#ifdef SMALLNUMBEROFENTRYVALUES
  os.write( (char*) &Nvals,sizeof(Nvals));
  os.write( (char*) &entryvalue[0],Nvals*sizeof(entryvalue[0]));
#endif
  return os;
}


vector<MyField> CSRmatrix::operator*(vector<MyField>& v)
{
  if(WTRACE)  cout << "in multiplication with vector<MyField>" << endl;
  vector<MyField> svec(M); // initialize answer vector
  for(int r=0; r<M; r++)
    {
      int start=rp[r];
      int stopp=(r != M-1 ? rp[r+1]: Ntot);
      int k=start;
      while(k<stopp){

#ifdef SMALLNUMBEROFENTRYVALUES
	svec[r]+=entryvalue[list[k]]*v[col[k]];
#else
	svec[r]+=list[k]*v[col[k]];
#endif
	k++;
      }
    }
  return svec;
}

vector<MyField> CSRmatrix::operator*(MyField* vptr)
{
  if(TRACE)  cout << "in multiplication" << endl;
  vector<MyField> svec(M); // initialize answer vector
  for(int r=0; r<M; r++)
    {
      int start=rp[r];
      int stopp=(r != M-1 ? rp[r+1]: Ntot);
      int k=start;
      while(k<stopp){

#ifdef SMALLNUMBEROFENTRYVALUES
	svec[r]+=entryvalue[list[k]]*vptr[col[k]];
#else
	svec[r]+=list[k]*vptr[col[k]];
#endif
	k++;
      }
    }
  return svec;
}

void CSRmatrix::TimesVector(vector<MyField>& vout,vector<MyField>& vin)
{
  if(WTRACE)  cout << "in multiplication with vector<MyField>" << endl;
  for(int r=0; r<M; r++)
    {
      int start=rp[r];
      int stopp=(r != M-1 ? rp[r+1]: Ntot);
      int k=start;
      while(k<stopp){

#ifdef SMALLNUMBEROFENTRYVALUES
	vout[r]+=entryvalue[list[k]]*vin[col[k]];
#else
	vout[r]+=list[k]*vin[col[k]];
#endif
	k++;
      }
    }
}

void CSRmatrix::TimesVector(vector<MyField>& vout,MyField* vin)
{
  if(WTRACE)  cout << "in multiplication with vector<MyField>" << endl;
  for(int r=0; r<M; r++)
    {
      int start=rp[r];
      int stopp=(r != M-1 ? rp[r+1]: Ntot);
      int k=start;
      while(k<stopp){

#ifdef SMALLNUMBEROFENTRYVALUES
	vout[r]+=entryvalue[list[k]]*vin[col[k]];
#else
	vout[r]+=list[k]*vin[col[k]];
#endif
	k++;
      }
    }
}

void CSRmatrix::TimesVector(MyField* vout,MyField* vin)
{
  if(WTRACE)  cout << "in multiplication with vector<MyField>" << endl;
  for(int r=0; r<M; r++)
    {
      int start=rp[r];
      int stopp=(r != M-1 ? rp[r+1]: Ntot);
      int k=start;
      while(k<stopp){

#ifdef SMALLNUMBEROFENTRYVALUES
	vout[r]+=entryvalue[list[k]]*vin[col[k]];
#else
	vout[r]+=list[k]*vin[col[k]];
#endif
	k++;
      }
    }
}


MyField CSRmatrix::operator()(const int r,const int c) // read-only routine, inefficient, only for testing purposes. 
{
  Tuple t(r,c);
  int start=rp[r];
  int stopp=(r != M-1 ? rp[r+1]: Ntot);
  for(int i=start; i<stopp; i++)
    {
#ifdef SMALLNUMBEROFENTRYVALUES
      if( col[i]==c ){ return entryvalue[list[i]];}
#else
      if( col[i]==c ){ return list[i];}
#endif
    }
  return 0.; // not found
}


void CSRmatrix::ConvertToDenseMatrix(vector<MyField>& A,bool rowmajororder=true)
{
  if(rowmajororder)
    { // c format
      for(int r=0; r<M; r++)
	{
	  int start=rp[r];
	  int stopp=(r != M-1 ? rp[r+1]: Ntot);
	  for(int i=start; i<stopp; i++){ A[r*N+col[i]]=list[i];}
	}
    }
  else
    { // fortran format:
      for(int r=0; r<M; r++)
	{
	  int start=rp[r];
	  int stopp=(r != M-1 ? rp[r+1]: Ntot);
	  for(int i=start; i<stopp; i++){ A[M*col[i]+r]=list[i];}
	}
    }

}

//******************************************************************************************

ELLmatrix::ELLmatrix(istream& is):M(0),N(0),Nc(0),Ntot(0),col(0),list(0)
#ifdef SMALLNUMBEROFENTRYVALUES
  ,entryvalue(0)
#endif
{
  Read(is);  // The matrix can also come from file
}

ELLmatrix::ELLmatrix(COOmatrix& Min,int Nc_in):M(Min.Nrows()),N(Min.Ncols()),Nc(Nc_in),Ntot(M*Nc),col(Ntot),list(Ntot)
#ifdef SMALLNUMBEROFENTRYVALUES
  ,Nvals(0),entryvalue(0)
#endif
{
  if(Nc==0) // go thru COO matrix to find the maximum number of columns
    { 
      vector<int> histogram(M);
      for(int i=0; i<M; i++){histogram[i]=0;}

      for(int i=0; i<Min.Nelements(); i++)
	{
	  int row_indx=Min.item(i).row;
	  histogram[row_indx]++;
	}

      Nc=*max_element(histogram.begin(),histogram.end());
      Ntot=Nc*M;

      col.resize(Ntot);
      list.resize(Ntot);
      
      
      if(TRACE)
	{
	  cout << "Histogram of #of non-zero columns for each row" << endl;
	  for(int r=0; r<M; r++)
	    {
	      cout << histogram[r] << " ";
	      for(int i=0; i<histogram[r]; i++) cout << "*";
	      cout << endl;
	    }
	  cout << "Max: " << Nc << " Ntot: " << Ntot << " storage penalty factor: " << M*Nc/(1.*Min.Nelements()) <<  endl;
	}

      //      logfile << "ELLmatrix: Rows=" << M << " Nc (stored cols)=" << Nc << " storage penalty factor: " << M*Nc/(1.*Min.Nelements()) <<  endl;
    }

  vector<int> counter(M);
  for(int i=0; i<M; i++){counter[i]=0;}
  
  for(int i=0; i<Min.Nelements(); i++)
    {
      int row_indx=Min.item(i).row;
      int col_indx=Min.item(i).col;
      MyField value=Min.item(i).a;

      int index=counter[row_indx]*M+row_indx; // store the matrices internally as Column major order. 
      col[index]=col_indx;
      
#ifdef SMALLNUMBEROFENTRYVALUES	  
      unsigned short int k=0;
      bool found=false;
      while(k<entryvalue.size())
	{ if(entryvalue[k++] == value){found=true; break;}}
      list[index]=k;
      if(!found){entryvalue.push_back(value); Nvals++;}
#else
      list[index]=value;
#endif
      counter[row_indx]++;
    }
  if(TRACE) cout << " Constructed ELL " << endl;
}


void ELLmatrix::Read(istream& is)
{
  is.read( (char*) &M,sizeof(M));
  is.read( (char*) &N,sizeof(N));
  is.read( (char*) &Nc,sizeof(Nc));
  is.read( (char*) &Ntot,sizeof(Ntot));
  col.resize(Ntot);
  is.read( (char*) &col[0],Ntot*sizeof(col[0]));
  list.resize(Ntot); 
  is.read( (char*) &list[0],Ntot*sizeof(list[0]));
#ifdef SMALLNUMBEROFENTRYVALUES
  is.read( (char*) &Nvals,sizeof(Nvals));
  entryvalue.resize(Nvals);
  is.read( (char*) &entryvalue[0],Nvals*sizeof(entryvalue[0]));
#endif  
}

ostream& ELLmatrix::Write(ostream& os)
{
  os.write( (char*) &M,sizeof(M));
  os.write( (char*) &N,sizeof(N));
  os.write( (char*) &Nc,sizeof(Nc));
  os.write( (char*) &Ntot,sizeof(Ntot));
  os.write( (char*) &col[0],Ntot*sizeof(col[0]));
  os.write( (char*) &list[0],Ntot*sizeof(list[0]));
#ifdef SMALLNUMBEROFENTRYVALUES
  os.write( (char*) &Nvals,sizeof(Nvals));
  os.write( (char*) &entryvalue[0],Nvals*sizeof(entryvalue[0]));
#endif
  return os;
}


vector<MyField> ELLmatrix::operator*(vector<MyField>& v)
{
  if(WTRACE)  cout << "in multiplication with vector<MyField>" << endl;
  vector<MyField> svec(M); // initialize answer vector
  for(int c=0; c<Nc; c++)
    for(int r=0; r<M; r++)
      {
	int index=c*M+r;
	
#ifdef SMALLNUMBEROFENTRYVALUES
	  svec[r]+=entryvalue[list[index]]*v[col[index]];
#else
	  svec[r]+=list[index]*v[col[index]];
#endif
      }
  return svec;
}

vector<MyField> ELLmatrix::operator*(MyField* vptr)
{
  if(WTRACE)  cout << "in multiplication with MyField*" << endl;
  vector<MyField> svec(M); // initialize answer vector
  for(int c=0; c<Nc; c++)
    for(int r=0; r<M; r++)
      {
	int index=c*M+r;
	
#ifdef SMALLNUMBEROFENTRYVALUES
	  svec[r]+=entryvalue[list[index]]*vptr[col[index]];
#else
	  svec[r]+=list[index]*vptr[col[index]];
#endif
      }
  return svec;
}


void ELLmatrix::TimesVector(vector<MyField>& vout,vector<MyField>& vin)
{
  if(WTRACE)  cout << "in multiplication with vector<MyField>" << endl;
  for(int c=0; c<Nc; c++)
    for(int r=0; r<M; r++)
      {
	int index=c*M+r;
	
#ifdef SMALLNUMBEROFENTRYVALUES
	  vout[r]+=entryvalue[list[index]]*vin[col[index]];
#else
	  vout[r]+=list[index]*vin[col[index]];
#endif
      }
}

void ELLmatrix::TimesVector(vector<MyField>& vout,MyField* vin)
{
  if(WTRACE)  cout << "in multiplication with vector<MyField>" << endl;
  for(int c=0; c<Nc; c++)
    for(int r=0; r<M; r++)
      {
	int index=c*M+r;
	
#ifdef SMALLNUMBEROFENTRYVALUES
	  vout[r]+=entryvalue[list[index]]*vin[col[index]];
#else
	  vout[r]+=list[index]*vin[col[index]];
#endif
      }
}

void ELLmatrix::TimesVector(MyField* vout,MyField* vin)
{
  if(WTRACE)  cout << "in multiplication with MyField*" << endl;
  for(int c=0; c<Nc; c++)
    for(int r=0; r<M; r++)
      {
	int index=c*M+r;
	
#ifdef SMALLNUMBEROFENTRYVALUES
	  vout[r]+=entryvalue[list[index]]*vin[col[index]];
#else
	  vout[r]+=list[index]*vin[col[index]];
#endif
      }
}


MyField ELLmatrix::operator()(const int r,const int c) // read-only routine, inefficient, only for testing purposes. 
{
  for(int cl=0; cl<Nc; cl++)
    {
      int index=cl*M+r;
#ifdef SMALLNUMBEROFENTRYVALUES
      if( col[index]==c ){ return entryvalue[list[index]];}
#else
      if( col[index]==c ){ return list[index];}
#endif
    }
  return 0.; // not found.
}


void ELLmatrix::ConvertToDenseMatrix(vector<MyField>& A,bool rowmajororder=true)
{
  if(WTRACE) cout << "in ConvertToDenseMatrix" << endl;
  for(int cl=0; cl<Nc; cl++)
    for(int r=0; r<M; r++)
      {
	int index=cl*M+r;
	MyField entry=list[index];
	if(entry==MyField(0.)) continue;
	if(rowmajororder)
	  A[r*N+col[index]]=entry; // C format
	else
	  A[col[index]*M+r]=entry; // Fortran format
      }
}



