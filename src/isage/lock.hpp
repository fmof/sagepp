#ifndef ISAGE_LOCK_H_
#define ISAGE_LOCK_H_

namespace isage {

#ifdef _OPENMP
#include <omp.h>

  struct Mutex {
    Mutex();
    ~Mutex();
    void lock();
    void unlock();
   
    Mutex(const Mutex& );
    Mutex& operator= (const Mutex& );
  private:
    omp_lock_t lock_;
  };

#else

  struct Mutex {
    void lock();
    void unlock();
  };

#endif
 
  struct Lock {
    explicit Lock(Mutex& m);
    ~Lock();
    void unlock();
    void relock();
  private:
    Mutex& mut_;
    bool locked_;
    void operator=(const Lock&);
    Lock(const Lock&);
  };
}

#endif
