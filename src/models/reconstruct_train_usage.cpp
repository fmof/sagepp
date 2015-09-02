#include "logging.hpp"
#include "sage.hpp"
#include "wtm.hpp"
#include "wtm_variational.hpp"

#include <gsl/gsl_rng.h>
#include <boost/program_options.hpp>

// for serialization
#include <fstream>
#include <iostream>
#include <ostream>
#include <type_traits>
// include headers that implement a archive in simple text format
#include <boost/algorithm/string.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>

namespace po = boost::program_options;

void init_logging() {
#ifndef ISAGE_LOG_AS_COUT
boost::log::core::get()->set_filter
(
boost::log::trivial::severity >= boost::log::trivial::info
 );
#endif
}

int main(int n_args, char** args) {
  init_logging();
  isage::util::print_pstats();

  po::variables_map vm;
  {
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help", "produce help message")
      ("serialized-inferencer", po::value<std::string>()->required(), 
       "filename to READ serialized inference state from")
      ("topic-usage-file", po::value<std::string>()->required()->default_value("-"), 
       "filename to write topic usage to (default: - (to console)")
      ;

    po::store(po::parse_command_line(n_args, args, desc), vm);
    if (vm.count("help")) {
      ERROR << desc << "\n";
      return 1;
    }
    po::notify(vm);
  }

  typedef std::string string;
  typedef string VocabType;
  typedef double CountType;
  typedef isage::wtm::Document< VocabType, CountType > Doc;
  typedef isage::wtm::SageTopic<std::vector<double> > TopicType;
  SAGE_TYPEDEF_BOILER(Doc, VocabType, TopicType);

  Variational heldout_inferencer;
  std::ifstream ifs(vm["serialized-inferencer"].as<std::string>(), std::ios::in|std::ios::binary);
  assert(ifs.good());
  boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
  in.push(boost::iostreams::gzip_decompressor());
  in.push(ifs);
  boost::archive::binary_iarchive ia(in);
  ia >> heldout_inferencer;

  std::vector<std::vector<double> > usage_estimates;
  heldout_inferencer.get_usage_estimates(&usage_estimates);
  isage::util::SmartWriter usage_outer(vm["topic-usage-file"].as<std::string>());
  std::ostream& outter = usage_outer.get();
  INFO << "Writing topic usage output to " << usage_outer.name();
  // get an outstream
  for(const auto& doc_usage : usage_estimates) {
    std::stringstream stream;
    for(const auto& p : doc_usage) {
      stream << p << " ";
    }
    outter << stream.str();
    outter << std::endl;
  }
  return 0;
}
