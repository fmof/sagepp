/**
 * A libnar-specific utility library for Concrete.
 */

#ifndef ISAGE_CONCRETE_H_
#define ISAGE_CONCRETE_H_

#include "concrete/communication_types.h"
#include "concrete/entities_types.h"
#include "concrete/situations_types.h"
#include "concrete/structure_types.h"
#include "concrete/uuid_types.h"

#include "util.hpp"

#include <list>
#include <map>
#include <vector>
#include <unordered_map>

namespace concrete {
}

namespace concrete { namespace util {
  struct uuid_hash { 
    const size_t operator()(const concrete::UUID& uuid) const {
      return std::hash<std::string>()(uuid.uuidString);
    }
  };

  template <typename T > using uuid_map = std::unordered_map< const concrete::UUID, T , concrete::util::uuid_hash >;

  template <typename T>
  inline const T* const first_set_with_name(const std::vector<T>& obj_of_interest,
					    const std::string& tool_name) {
    for(typename std::vector< T >::const_iterator it = obj_of_interest.begin();
	it != obj_of_interest.end(); ++it) {
      if(it->metadata.tool.find(tool_name) != std::string::npos) return &(*it);  
    }
    return NULL;
  };
  template <typename T>
  inline const T* const first_set_with_name(const std::vector<T>& obj_of_interest,
					    const std::string& tool_name,
					    const std::string& type) {
    for(typename std::vector< T >::const_iterator it = obj_of_interest.begin();
	it != obj_of_interest.end(); ++it) {
      if(it->metadata.tool.find(tool_name) != std::string::npos &&
	 it->taggingType == type) return &(*it);  
    }
    return NULL;
  };

  const concrete::TokenTagging* const first_pos_tagging(const concrete::Tokenization& tokenization,
							const std::string& tool_name);
  const concrete::TokenTagging* const first_ner_tagging(const concrete::Tokenization& tokenization,
							const std::string& tool_name);
  const concrete::TokenTagging* const first_lemma_tagging(const concrete::Tokenization& tokenization,
							  const std::string& tool_name);
  const concrete::DependencyParse* const first_dependency_parse(const concrete::Tokenization& tokenization,
								const std::string& tool_name);
  const concrete::EntitySet* const first_entity_set(const concrete::Communication& comm,
						    const std::string& tool_name);

  const uuid_map<concrete::EntityMention> mention_id_to_mention(const concrete::Communication& comm, const std::string& tool_name);
  const uuid_map<concrete::Tokenization> mention_id_to_tokenization(const concrete::Communication& comm, const std::string& tool_name);
  const uuid_map< std::list< concrete::SituationMention > > tokenization_id_to_situation_mention(const concrete::Communication& comm, const std::string& tool_name);
  uuid_map<concrete::Tokenization> tokenization_id_to_tokenization(const concrete::Communication& comm);

}
}
#endif
