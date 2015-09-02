// Header for word-based topic models

#ifndef ISAGE_WTM_SAGE_H_
#define ISAGE_WTM_SAGE_H_

#include "sage_defs.hpp"
#include "sage_limmem.hpp"
#include "sage_bloated.hpp"

#ifdef LIM_MEM_VARIATIONAL
#define SAGE_TYPEDEF_BOILER(Doc, VocabType, TopicType) \
  typedef isage::wtm::SageVariationalLimMem< Doc, VocabType, TopicType > Variational
#else
#define SAGE_TYPEDEF_BOILER(Doc, VocabType, TopicType) \
  typedef isage::wtm::SageVariationalHighMem< Doc, VocabType, TopicType > Variational
#endif


#endif
