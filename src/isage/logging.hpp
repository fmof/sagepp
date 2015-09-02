#ifndef ISAGE_LOGGING_H_
#define ISAGE_LOGGING_H_

namespace logging {  
  void init();
}

#ifdef ISAGE_LOG_AS_COUT
#include <iostream>

#define ADD_FILE_LINE_FUNC << "(" << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << ") "

#ifndef DEBUG
#include "boost/iostreams/stream.hpp"
#include "boost/iostreams/device/null.hpp"
static boost::iostreams::stream< boost::iostreams::null_sink > null_stream( ( boost::iostreams::null_sink() ) );
#define TRACE null_stream
#define DEBUG null_stream
#else
#define TRACE std::cout << std::endl << "TRACE " ADD_FILE_LINE_FUNC
#define DEBUG std::cout << std::endl << "DEBUG " ADD_FILE_LINE_FUNC
#endif

#define INFO std::cout << std::endl << "INFO " ADD_FILE_LINE_FUNC
#define WARN std::cout << std::endl << "WARN  " ADD_FILE_LINE_FUNC
#define ERROR std::cout << std::endl << "ERROR "  ADD_FILE_LINE_FUNC

#else

#include <boost/log/core.hpp>

#include <boost/log/sources/basic_logger.hpp>
#include <boost/log/sources/global_logger_storage.hpp>
// #include <boost/log/sources/record_ostream.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup.hpp>
#include <boost/log/sources/severity_feature.hpp>
#include <boost/log/sources/severity_logger.hpp>

// #include <boost/log/trivial.hpp>

// #include <boost/log/utility/setup/console.hpp>
// #include <boost/log/utility/setup/file.hpp>
// #include <boost/log/utility/setup/common_attributes.hpp>


#define ADD_FILE_LINE_FUNC << "[" << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << "] "

#define TRACE BOOST_LOG_SEV(my_logger::get(), boost::log::trivial::trace) 
#define DEBUG BOOST_LOG_SEV(my_logger::get(), boost::log::trivial::debug) 
#define INFO BOOST_LOG_SEV(my_logger::get(), boost::log::trivial::info) 
#define WARN BOOST_LOG_SEV(my_logger::get(), boost::log::trivial::warning) 
#define ERROR BOOST_LOG_SEV(my_logger::get(), boost::log::trivial::error) 

//Narrow-char thread-safe logger.
typedef boost::log::sources::severity_logger_mt< boost::log::trivial::severity_level > logger_t;

//declares a global logger with a custom initialization
// behind the scenes, this macro defines a struct
BOOST_LOG_GLOBAL_LOGGER(my_logger, logger_t)

// #define TRACE_TRIVIAL BOOST_LOG_TRIVIAL(trace) ADD_FILE_LINE_FUNC
// #define DEBUG_TRIVIAL BOOST_LOG_TRIVIAL(debug) ADD_FILE_LINE_FUNC
// #define INFO_TRIVIAL BOOST_LOG_TRIVIAL(info) ADD_FILE_LINE_FUNC
// #define WARN_TRIVIAL BOOST_LOG_TRIVIAL(warning) ADD_FILE_LINE_FUNC
// #define ERROR_TRIVIAL BOOST_LOG_TRIVIAL(error) ADD_FILE_LINE_FUNC

// #define TRACE_CUSTOM BOOST_LOG_TRIVIAL(trace) 
// #define DEBUG_CUSTOM BOOST_LOG_TRIVIAL(debug) 
// #define INFO_CUSTOM BOOST_LOG_TRIVIAL(info) 
// #define WARN_CUSTOM BOOST_LOG_TRIVIAL(warning) 
// #define ERROR_CUSTOM BOOST_LOG_TRIVIAL(error) 

// #define GET_MACRO(_0, _1, _2, NAME, ...) NAME

// #define TRACE(...) GET_MACRO(_0, ##__VA_ARGS__, TRACE_TRIVIAL, TRACE_CUSTOM)(__VA_ARGS__)
// #define DEBUG(...) GET_MACRO(_0, ##__VA_ARGS__, TRACE_TRIVIAL, TRACE_CUSTOM)(__VA_ARGS__)
// #define INFO(...) GET_MACRO(_0, ##__VA_ARGS__, TRACE_TRIVIAL, TRACE_CUSTOM)(__VA_ARGS__)
// #define WARN(...) GET_MACRO(_0, ##__VA_ARGS__, TRACE_TRIVIAL, TRACE_CUSTOM)(__VA_ARGS__)
// #define ERROR(...) GET_MACRO(_0, ##__VA_ARGS__, TRACE_TRIVIAL, TRACE_CUSTOM)(__VA_ARGS__)
#endif

#endif
