#ifndef GLOBAL_H
#define GLOBAL_H

#ifdef ELLMATRIX
typedef ELLmatrix MyMatrix;
#else
typedef CSRmatrix MyMatrix;
#endif

typedef complex<double> MyField;


#endif // GLOBAL_H
