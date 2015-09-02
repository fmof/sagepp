#include "util.hpp"
#include "version.hpp"

#include <fstream>
#include <iostream>
#include <ostream>

#include <sys/types.h>
#include <unistd.h>

namespace isage { namespace util {

    SmartWriter::SmartWriter() : console_(true), f_ptr(NULL), curr_file("stdout") {
    }
    SmartWriter::SmartWriter(const std::string& fname) : 
      base_(fname), console_(fname == "-"), f_ptr(NULL) {
      if(console_) curr_file = "stdout";
    }

    SmartWriter::~SmartWriter() {
      if(!console_ && f_ptr != NULL) {
	f_ptr->close();
	delete f_ptr;
      }
    }

    std::ostream& SmartWriter::get(const std::string& suffix) {
      if(!console_) {
	const std::string& f = suffix.size() > 0 ? (base_ + "." + suffix) : base_;
	if(f_ptr != NULL) {
	  INFO << "Closing previously opened file " << curr_file << " and opening the new one " << f;
	  f_ptr->close();
	  delete f_ptr;
	}
	curr_file = f;
	f_ptr = new std::ofstream(f, std::ofstream::out);
	return *f_ptr;
      } else {
	return std::cout;
      }    
    }
    std::ostream& SmartWriter::get() {
      return get("");
    }
    std::ostream& SmartWriter::get(const int i) {
      return get(std::to_string(i));
    }
    std::string SmartWriter::base_name() {
      return base_;
    }
    std::string SmartWriter::name() {
      return curr_file;
    }

    bool SmartWriter::to_file() {
      return !console_ && this->name() != "/dev/null";
    }

    /**
     * Print various stats of the process
     */
    void print_pstats() {
      pid_t pid = getpid();
      pid_t ppid = getppid();
      INFO << "PROCESS ID: " << pid;
      INFO << "Parent Process ID: " << ppid;
      INFO << "iSage Library built from: " << isage::ISAGE_GIT_SHA;
      INFO << "iSage Library built one: " << isage::ISAGE_BUILD_DATE;
    }

  }
}
