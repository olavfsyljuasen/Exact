#ifndef TIMER_H
#define TIMER_H

#include<iostream>
#include<time.h>
#include<cstdlib>
using namespace std;


class Timer{
  friend ostream&
    operator<<(ostream& os, Timer timer)
    { time_t t;
      if( time(&t) == time_t(-1)){exit(1);}
      os << ctime(&t); return os;
    }  
 public:
  Timer(){if(TRACE) cout << "starting timer\n";}
  ~Timer(){if(TRACE) cout << "deleting timer\n";}
  void Start(){t1=time(0);}
  void Stop(){t2=time(0);}
  double GetTimeElapsed(){return difftime(t2,t1);}
 private:
  time_t t1;
  time_t t2;
};

#endif // TIMER_H
