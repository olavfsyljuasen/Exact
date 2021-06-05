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
  Tuple():row(0),col(0),a(0){}
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
 COOmatrix(int M_in,int N_in):M(M_in),N(N_in),list(0),dummy(0){}
  COOmatrix(istream& is);
  int Nrows(){return M;}
  int Ncols(){return N;}
  int Nelements(){return list.size();}
  Tuple& item(const int i){return list[i];}
  MyField Get(const int,const int); // read value
  MyField& operator()(const int,const int); // assign value
  void AddTo(const int r,const int c,MyField a_in); // addto value
  ostream& Write(ostream& os);
  void     Read(istream& is);

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
  int M; //rows
  int N; //cols
  vector<Tuple> list;
  MyField dummy; // 
};


//*****************************************************************

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
  void TimesVector(vector<MyField>& vout,MyField* vin);// x dense vector
  void TimesVector(MyField* vout,MyField* vin);// x dense vector
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

//**********************************************************************

class ELLmatrix
{
  friend ostream& operator<<(ostream& os,ELLmatrix& m)
  {m.PrintList(os) << endl; return os;}
 public:
  ELLmatrix(COOmatrix& Min,int Nc_in=0);    // The matrix is constructed from a COOmatrix
  ELLmatrix(istream& is);           // The matrix can also come from file
  vector<MyField> operator*(vector<MyField>& v); // Multiplying a dense vector
  vector<MyField> operator*(MyField* vptr); // Multiplying a dense vector
  void TimesVector(vector<MyField>& vout,vector<MyField>& vin);// x dense vector
  void TimesVector(vector<MyField>& vout,MyField* vin);// x dense vector
  void TimesVector(MyField* vout,MyField* vin);// x dense vector
  MyField operator()(const int,const int); // read-only routine, inefficient. 

  MyField* GetValStartPtr(){return &list[0];}
  int* GetColStartPtr(){return &col[0];}
  
  int Nrows(){return M;}
  int Ncols(){return N;}
  int NStoredCols(){return Nc;} // the number of stored columns
  int Nelements(){return Ntot;}
  
  void ConvertToDenseMatrix(vector<MyField>&,bool );
  ostream& PrintList(ostream& os)
    {
      for(int i=0; i<list.size(); i++){os << list[i] << " ";}
      os << "Column list: " << endl;
      for(int i=0; i<col.size(); i++){os << col[i] << " ";}
      return os;
    }
 
  ostream& Write(ostream& os);
  void     Read(istream& is);
  
 private:
  int M; //rows
  int N; //cols
  int Nc; // the maximum number of columns stored
  int Ntot; // Total number of elements including zeros;
  vector<int> col;
#ifdef SMALLNUMBEROFENTRYVALUES
  vector<unsigned short int> list;
  int Nvals;
  vector<MyField> entryvalue;
#else
  vector<MyField> list;
#endif
};


#endif
