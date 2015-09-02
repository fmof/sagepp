#include "lock.hpp"

namespace isage {

#ifdef _OPENMP
#include <omp.h>

  Mutex::Mutex() { 
     omp_init_lock(&lock_); 
   }
  Mutex::~Mutex() { 
     omp_destroy_lock(&lock_); 
   }
  void Mutex::lock() { 
     omp_set_lock(&lock_); 
   }
  void Mutex::unlock() { 
     omp_unset_lock(&lock_); 
   }
   
  Mutex::Mutex(const Mutex& ) { 
     omp_init_lock(&lock_); 
   }
  Mutex& Mutex::operator=(const Mutex& ) { 
     return *this; 
   }

 #else
  void Mutex::lock() {
  }
  void Mutex::unlock() {
  }
 #endif
 
  Lock::Lock(Mutex& m) : mut_(m), locked_(true) { 
     mut_.lock(); 
   }
  Lock::~Lock() { 
     unlock(); 
   }
  void Lock::unlock() { 
     if(!locked_) return; 
     locked_=false; 
     mut_.unlock(); 
   }
  void Lock::relock() { 
    if(locked_) 
      return; 
    mut_.lock(); 
    locked_=true; 
  }
}
