#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H

#include<complex>
#include<iostream>
#include<vector>
#include<algorithm>
#include<fstream>
using namespace std;

typedef complex<double> MyField;
//typedef complex<float> MyField;
//typedef double MyField;

//COOmatrix is sparse density matrix which is easy to construct
//it is simply a list of Tuples which has indices and a value.

class Tuple
{
  friend ostream& operator<<(ostream& os,Tuple& t)
  {os << "(" << t.row << "," << t.col << ":" << t.a << ")"; return os;} 
 public:
  Tuple(int r,int c,MyField ain=0):row(r),col(c),a(ain){}
  int row;
  int col;
  MyField a;
  friend bool operator==(const Tuple& l, const Tuple& r){return ( l.row==r.row && l.col==r.col);}
  friend bool operator<(const Tuple& l, const Tuple& r)
  { return ( l.row < r.row ? true : (l.row==r.row ? l.col<r.col : false));}
  friend bool operator==(const Tuple& l, const MyField& value){return l.a==value;}
  friend bool operator<(const Tuple& l, const MyField& value){return abs(l.a) < abs(value);}
};

class SmallValue
{
 public:
 SmallValue(MyField maxval_in):maxval(abs(maxval_in)){}
  bool operator()(Tuple t){return abs(t.a)<maxval;}
 private:
  double maxval;
};
      

class COOmatrix
{
  friend ostream& operator<<(ostream& os,COOmatrix& c)
  {for(int i=0; i<c.list.size(); i++){os << c.list[i] << " ";} return os;}
 public:
 COOmatrix(int M_in,int N_in):M(M_in),N(N_in){}
  int Nrows(){return M;}
  int Ncols(){return N;}
  int Nelements(){return list.size();}
  Tuple& item(const int i){return list[i];}
  MyField& operator()(const int,const int);
  void DeleteValues(MyField v)
  {
    vector<Tuple>::iterator iter=remove(list.begin(), list.end(),v);
    list.erase(iter,list.end());
  }
  void DeleteTinyValues(double limit)
  {
    vector<Tuple>::iterator iter=remove_if(list.begin(), list.end(),SmallValue(limit));
    list.erase(iter,list.end());
  }
  void SetSmallImagValuesToZero(double limit)
  {
    for(int i=0; i<list.size(); i++){ 
      if(abs(list[i].a.imag()) < limit){ list[i].a=complex<double>(list[i].a.real(),0.);}}
  }
  bool IsReal()
  {
    for(int i=0; i<list.size(); i++)
      { if(list[i].a.imag() != 0.){return false;}}
    return true;
  }
private:
  const int M; //rows
  const int N; //cols
  vector<Tuple> list;
  MyField dummy; // 
};

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


//CSRmatrix is a sparse matrix which is efficient in use for multiplying with vectors
//
//Entry class contains the colum number and the value
/*
class Entry
{
  friend ostream& operator<<(ostream& os,Entry& t)
  {os << "(" << t.col << "," << t.a << ")"; return os;} 
 public:
#ifdef SMALLNUMBEROFENTRYVALUES
  Entry(int c=0,unsigned short int a_in=0):col(c),a(ain){}
  int col;
  unsigned short int a;
#else
 Entry(int c=0,MyField a_in=0.):col(c),a(a_in){}
  int col;
  MyField a;
#endif
};


class Entry
{
  friend ostream& operator<<(ostream& os,Entry& t)
   {os << "(" << t.col << "," << t.a << ")"; return os;} 
 public:
  Entry(int c=0,MyField a_in=0):col(c),a(a_in){}
  int col;
  MyField a;
};
*/

/*
struct Entry
{
  MyField a;
  int col;
};
*/

class CSRmatrix
{
  friend ostream& operator<<(ostream& os,CSRmatrix& m)
  {m.PrintList(os) << endl; m.Printrp(os) << endl; return os;}
 public:
  CSRmatrix(COOmatrix& Min);        // The matrix is constructed from a COOmatrix
  CSRmatrix(istream& is);           // The matrix can also come from file
  vector<MyField> operator*(vector<MyField>& v); // Multiplying a dense vector
  vector<MyField> operator*(MyField* vptr); // Multiplying a dense vector
  void TimesVector(vector<MyField>& vout,vector<MyField>& vin);// x dense vector
  MyField operator()(const int,const int); // read-only routine, inefficient. 
  
  int Nrows(){return M;}
  int Ncols(){return N;}
  int Nelements(){return Ntot;}
  
  void ConvertToDenseMatrix(vector<MyField>&,bool );
  ostream& PrintList(ostream& os){for(int i=0; i<list.size(); i++){os << " (" << col[i] << "," << list[i] << ")";} return os;} 
  ostream& Printrp(ostream& os){for(int i=0; i<rp.size(); i++){os << rp[i] << " ";} return os;} 
  
  ostream& Write(ostream& os);
  void     Read(istream& is);
  
 private:
  int M; //rows
  int N; //cols
  int Ntot; // Total number of elements;
  vector<int> col;
  vector<int> rp;
#ifdef SMALLNUMBEROFENTRYVALUES
  vector<unsigned short int> list;
  int Nvals;
  vector<MyField> entryvalue;
#else
  vector<MyField> list;
#endif

};

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
  if(TRACE)  cout << "in multiplication with vector<MyField>" << endl;
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
  if(TRACE)  cout << "in multiplication with vector<MyField>" << endl;
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

//--------------------------------------------------------------------------------------------

/*
int main()
{
  cout << "Starting" << endl;

  short int f=0;
  cout << "sizeof short:" << sizeof(f) << endl; 

  int d1=6;
  int d2=4;

  COOmatrix A(d1,d2);
  A(0,0)=1;
  A(0,1)=2.;
  A(0,2)=3.;
  A(0,3)=4;
  A(1,0)=2;
  A(2,0)=3;
  A(3,0)=4;
  A(3,3)=4;

  cout << A << endl;
  A.DeleteValues(0.);
  cout << A << endl;

  CSRmatrix B(A);
  
  

  for(int j=0; j<d1; j++)
    {
    for(int i=0; i<d2; i++)
      {
	cout << B(j,i) << " ";
      }
    cout << endl;
    }

  cout << "sizeof(B): " << sizeof(B) << endl;

  string filename="test.dat";
  ofstream file(filename.c_str());

  B.Write(file);

  file.close();
  
  ifstream filein(filename.c_str());

  CSRmatrix C(filein);

  //
  cout << "The matrix C: " << endl;

  for(int j=0; j<d1; j++)
    {
    for(int i=0; i<d2; i++)
      {
	cout << C(j,i) << " ";
      }
    cout << endl;
    }

  vector<MyField> v(d2);
  v[0]=5;
  v[1]=6;
  v[2]=100;
  v[3]=700;
  cout << endl;
  vector<MyField> Answer(d1);
  Answer=C*v;
  for(int i=0; i<d1 ;i++) cout << Answer[i] << endl;

  cout << "Ended" << endl;
}

*/


#endif /* SPARSEMATRIX_H */
