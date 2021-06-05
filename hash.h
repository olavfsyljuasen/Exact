#include<iostream>
#include<string>
using namespace std;

unsigned sax_hash ( unsigned char* key, int len )
{
  unsigned char *p = key;
  unsigned h = 0;
  int i;
  
  for ( i = 0; i < len; i++ )
    h ^= ( h << 5 ) + ( h >> 2 ) + p[i];
  
  return h;
}

unsigned fnv_hash ( unsigned char* key, int len )
{
  unsigned char *p = key;
  unsigned h = 2166136261;
  int i;
  
  for ( i = 0; i < len; i++ )
    h = ( h * 16777619 ) ^ p[i];
  
  return h;
}

unsigned oat_hash ( unsigned char* key, int len )
{
  unsigned char *p = key;
  unsigned h = 0;
  int i;
  
  for ( i = 0; i < len; i++ ) {
    h += p[i];
    h += ( h << 10 );
    h ^= ( h >> 6 );
  }
  
  h += ( h << 3 );
  h ^= ( h >> 11 );
  h += ( h << 15 );
  
  return h;
}


unsigned myhash(double* dptr,int length)
{
  return fnv_hash(reinterpret_cast<unsigned char*>(dptr),length);
}

/*
int main()
{
  double t[5]={1.01,0.1,0.2,3.,4.1};
  int c=sizeof(t);
  double* dptr=&t[0];
  cout << myhash(dptr,c) << endl;


}
*/
