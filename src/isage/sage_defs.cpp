#include "sage_defs.hpp"

namespace isage {
  namespace wtm {
    std::istream& operator>>(std::istream& in, isage::wtm::SageTopicRegularization &how) {
      std::string token;
      in >> token;
      if (token == "L2")
	how = isage::wtm::SageTopicRegularization::L2;
      else if (token == "IMPROPER" )
	how = isage::wtm::SageTopicRegularization::IMPROPER;
      else {
	ERROR << "Unknown regularization choice " << token;
	throw 5;
      }
      //else throw boost::program_options::validation_error("Invalid unit");
      return in;
    }

    double compute_sage_regularizer(double value, SageTopicRegularization how) {
      double res = 0.0;
      switch(how) {
      case SageTopicRegularization::L2:
	res = value * value;
	break;
      case SageTopicRegularization::IMPROPER:
	res = value * value * value * value;
	break;
      default:
	ERROR << "Unknown regularization type " << how;
	throw 4;
      }
      return res;
    }
    double compute_grad_sage_regularizer(double value, SageTopicRegularization how) {
      double res = 0.0;
      switch(how) {
      case SageTopicRegularization::L2:
	res = 2.0 * value;
	break;
      case SageTopicRegularization::IMPROPER:
	// in this case this happens to be trial_weights^3 (according to the SAGE paper, at least...)
	// we need to multiply by 2 to account for .5 * d/dx(x^4) = 2x^3
	res = 4.0 * value * value * value;
	break;
      default:
	ERROR << "Unknown regularization type " << how;
	throw 4;
      }
      return res;
    }

    std::istream& operator>>(std::istream& in, isage::wtm::TopicInitializerChoice &how) {
      std::string token;
      in >> token;
      if (token == "UNIFORM")
	how = isage::wtm::TopicInitializerChoice::UNIFORM;
      else if (token == "SUBSET")
	how = isage::wtm::TopicInitializerChoice::SUBSET;
      else if (token == "BACKGROUND")
	how = isage::wtm::TopicInitializerChoice::BACKGROUND;
      else {
	ERROR << "Unknown initialization choice " << token;
	throw 5;
      }
      //else throw boost::program_options::validation_error("Invalid unit");
      return in;
    }

  }
}
