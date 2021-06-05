#ifndef PRECISION_TYPES_H
#define PRECISION_TYPES_H


#include <qd/dd_real.h>
#include <qd/qd_real.h>
#include <qd/fpu.h>

struct sd_cmplx{double x; double y; sd_cmplx(double xd=0.,double yd=0.){x=xd; y=yd;}};
struct dd_cmplx{dd_real x; dd_real y; dd_cmplx(double xd=0.,double yd=0.){x=dd_real(xd); y=dd_real(yd);}};
struct qd_cmplx{qd_real x; qd_real y; qd_cmplx(double xd=0.,double yd=0.){x=qd_real(xd); y=qd_real(yd);}};

#ifdef DOUBLEDOUBLE
#define realformat  dd_real
#define cmplxformat dd_cmplx
#define todouble to_double
#define makereal    dd_real
#elif defined QUADDOUBLE
#define realformat  qd_real
#define cmplxformat qd_cmplx
#define todouble    to_double
#define makereal    qd_real
#else
#define realformat  double
#define cmplxformat sd_cmplx
#define todouble    double
#define makereal    double
#endif

/*
const int sizeofreal=sizeof(realformat);
const int sizeofcmplx=sizeof(cmplxformat);
*/

#endif //PRECISION_TYPES_H
