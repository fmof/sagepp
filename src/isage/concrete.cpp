#include "concrete.hpp"
#include <list>
#include <vector>
#include <unordered_map>

#include "logging.hpp"

namespace concrete { namespace util {

  const concrete::TokenTagging* const first_pos_tagging(const concrete::Tokenization& tokenization,
							const std::string& tool_name) {
    return concrete::util::first_set_with_name<concrete::TokenTagging>(tokenization.tokenTaggingList, tool_name, "POS");
  }
  const concrete::TokenTagging* const first_ner_tagging(const concrete::Tokenization& tokenization,
							const std::string& tool_name) {
    return concrete::util::first_set_with_name<concrete::TokenTagging>(tokenization.tokenTaggingList, tool_name, "NER");
  }
  const concrete::TokenTagging* const first_lemma_tagging(const concrete::Tokenization& tokenization,
							  const std::string& tool_name) {
    return concrete::util::first_set_with_name<concrete::TokenTagging>(tokenization.tokenTaggingList, tool_name, "LEMMA");
  }

  const concrete::DependencyParse* const first_dependency_parse(const concrete::Tokenization& tokenization,
								const std::string& tool_name) {
    return concrete::util::first_set_with_name<concrete::DependencyParse>(tokenization.dependencyParseList, tool_name);
  }

  const concrete::EntitySet* const first_entity_set(const concrete::Communication& comm,
						    const std::string& tool_name) {
    return concrete::util::first_set_with_name<concrete::EntitySet>(comm.entitySetList, tool_name);
  }

  uuid_map<concrete::Tokenization> tokenization_id_to_tokenization(const concrete::Communication& comm) {
    uuid_map<concrete::Tokenization> tutt;
    int num_tok = 0;
    if(!comm.__isset.sectionList) return tutt;
    for(const concrete::Section& section : comm.sectionList) {
      if(!section.__isset.sentenceList) continue;
      //for(const concrete::Sentence& sentence : section.sentenceList) {
      for(std::vector<concrete::Sentence>::const_iterator sit = section.sentenceList.begin();
	  sit != section.sentenceList.end(); ++sit) {
	if(!sit->__isset.tokenization) continue;
	const concrete::Tokenization* tptr = &(sit->tokenization);
	//BOOST_LOG_TRIVIAL(debug) << "Storing tokenization idx = " << num_tok << " with ID " << tptr->uuid.uuidString;
	tutt[tptr->uuid] = *tptr;
	//BOOST_LOG_TRIVIAL(debug) << "checking: " << tutt[tptr->uuid].uuid.uuidString;
	//BOOST_LOG_TRIVIAL(debug) << "...loaded";
	++num_tok;
      }
    }
    return tutt;
  }

  const uuid_map<concrete::EntityMention> mention_id_to_mention(const concrete::Communication& comm, const std::string& tool_name) {
    const concrete::EntityMentionSet* ems =
      concrete::util::first_set_with_name<concrete::EntityMentionSet>(comm.entityMentionSetList,
								     tool_name);
    // select the theory corresponding to toolname
    uuid_map<concrete::EntityMention> mitm;
    if(!ems) return mitm;
    for(concrete::EntityMention em : ems->mentionList) {
      mitm[em.uuid] = em;
    }
    return mitm;
  }

  const uuid_map< std::list< concrete::SituationMention > > tokenization_id_to_situation_mention(const concrete::Communication& comm, const std::string& tool_name) {
    const concrete::SituationMentionSet* sms =
      concrete::util::first_set_with_name<concrete::SituationMentionSet>(comm.situationMentionSetList,
									tool_name);
    uuid_map< std::list< concrete::SituationMention> > tism;
    if(!sms) {
      BOOST_LOG_TRIVIAL(warning) << "Did not find any SituationMentionSets in communication " << comm.id << " with name containing " << tool_name;
      return tism;
    }
    for(const concrete::SituationMention sm : sms->mentionList) {
      // if(tism.find(sm.tokens.tokenizationId) == tism.end()) {
      // 	tism[sm.tokens.tokenizationId] = std::list<concrete::SituationMention>();
      // }
      tism[sm.tokens.tokenizationId].push_back(sm);
    }
    return tism;
  }

  const uuid_map<concrete::Tokenization> mention_id_to_tokenization(const concrete::Communication& comm, const std::string& tool_name) {
    const concrete::EntityMentionSet* ems =
      concrete::util::first_set_with_name<concrete::EntityMentionSet>(comm.entityMentionSetList,
								     tool_name);
    uuid_map<concrete::Tokenization> mitt;
    if(!ems) {
      BOOST_LOG_TRIVIAL(warning) << "Did not find any EntityMentionSets in communication " << comm.id << " with name containing " << tool_name;
      return mitt;
    }
    uuid_map<concrete::Tokenization> tid_to_tok =
      concrete::util::tokenization_id_to_tokenization(comm);
    for(const concrete::EntityMention em : ems->mentionList) {
      concrete::Tokenization tp = tid_to_tok.at(em.tokens.tokenizationId);
      mitt[em.uuid] = tp;
    }
    return mitt;
  }

}
}
