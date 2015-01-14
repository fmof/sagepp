#include "gtest/gtest.h"

#include "concrete.hpp"

#include "concrete_util/uuid_util.h"
#include "concrete_util/io.h"

#include "logging.hpp"

#include <unordered_map>

TEST(UUIDString, create) {
  concrete::util::uuid_factory uuid_maker;
  BOOST_LOG_TRIVIAL(debug) << "Generated UUID is " << uuid_maker.get_uuid();
}

TEST(Communication, readBinaryThroughGen) {
  concrete::Communication communication;
  concrete::util::concrete_io concrete_reader;
  concrete_reader.deserialize<concrete::util::TBinaryProtocol, concrete::Communication>(&communication, "test/resources/AFP_ENG_19940531.0390.tbinary.concrete");
  ASSERT_EQ("AFP_ENG_19940531.0390", communication.id);
}

TEST(Communication, readCompactThroughGen) {
  concrete::Communication communication;
  concrete::util::concrete_io concrete_reader;
  concrete_reader.deserialize<concrete::util::TCompactProtocol, concrete::Communication>(&communication, "test/resources/AFP_ENG_19940531.0390.tcompact.concrete");
  ASSERT_EQ("AFP_ENG_19940531.0390", communication.id);
}

TEST(Communication, readTBinaryProtocol) {
  concrete::Communication communication;
  concrete::util::concrete_io concrete_reader;
  const char *name = "test/resources/AFP_ENG_19940531.0390.tbinary.concrete";
  concrete_reader.deserialize_binary<concrete::Communication>(&communication, name);
  ASSERT_EQ("AFP_ENG_19940531.0390", communication.id);
}

TEST(Communication, readTCompactProtocol) {
  concrete::Communication communication;
  concrete::util::concrete_io concrete_reader;
  const char *name = "test/resources/AFP_ENG_19940531.0390.tcompact.concrete";
  concrete_reader.deserialize_compact<concrete::Communication>(&communication, name);
  ASSERT_EQ("AFP_ENG_19940531.0390", communication.id);
}

TEST(EntitySet, find) {
  concrete::Communication communication;
  concrete::util::concrete_io concrete_reader;
  const char *name = "test/resources/AFP_ENG_19940531.0390.tcompact.concrete";
  concrete_reader.deserialize_compact<concrete::Communication>(&communication, name);
  const concrete::EntitySet* es = concrete::util::first_entity_set(communication,
								  "Stanford");
  ASSERT_TRUE(es != NULL);
}

TEST(ConcreteUtil, first_set_with_name_mention_from_compact_AFP_EN_19940531_0390) {
  concrete::Communication comm;
  concrete::util::concrete_io concrete_reader;
  const char *name = "test/resources/AFP_ENG_19940531.0390.tcompact.concrete";
  BOOST_LOG_TRIVIAL(debug) << "preparing to read " << name;
  concrete_reader.deserialize_compact<concrete::Communication>(&comm, name);
  BOOST_LOG_TRIVIAL(debug) << "successfully loaded communication from " << name;
  ASSERT_EQ("AFP_ENG_19940531.0390", comm.id);
  const concrete::EntityMentionSet* ems =
    concrete::util::first_set_with_name<concrete::EntityMentionSet>(comm.entityMentionSetList,
								   "Stanford");
  BOOST_LOG_TRIVIAL(debug) << "Creating mention id to tokenization mapping";
  ASSERT_TRUE(ems != NULL);
  ASSERT_EQ(50, ems->mentionList.size());
}

TEST(ConcreteUtil, iterate_sentences) {
  concrete::Communication comm;
  concrete::util::concrete_io concrete_reader;
  const char *name = "test/resources/AFP_ENG_19940531.0390.tcompact.concrete";
  BOOST_LOG_TRIVIAL(debug) << "preparing to read " << name;
  concrete_reader.deserialize_compact<concrete::Communication>(&comm, name);
  BOOST_LOG_TRIVIAL(debug) << "successfully loaded communication from " << name;
  ASSERT_EQ("AFP_ENG_19940531.0390", comm.id);
  int num_toks = 0;
  for(concrete::Section section : comm.sectionList) {
    if(!section.__isset.sentenceList) continue;
    for(concrete::Sentence sentence : section.sentenceList) {
      ASSERT_TRUE(sentence.__isset.tokenization);
      ++num_toks;
    }
  }
  ASSERT_EQ(9, num_toks);
}

TEST(ConcreteUtil, uuid_hash) {
  concrete::util::uuid_factory uf;
  concrete::UUID uuid;
  uuid.__set_uuidString(uf.get_uuid());
  const size_t uuid_hash = concrete::util::uuid_hash()(uuid);
  BOOST_LOG_TRIVIAL(info) << "Created hash of UUID[" << uuid.uuidString << "] = " << uuid_hash;
}

TEST(ConcreteUtil, uuid_map_no_alias) {
  std::unordered_map< const concrete::UUID, int, concrete::util::uuid_hash > tutt;
  concrete::util::uuid_factory uf;
  concrete::UUID uuid;
  uuid.__set_uuidString(uf.get_uuid());
  BOOST_LOG_TRIVIAL(debug) << "Attempting to add element with value " << uuid.uuidString << " to the map";
  tutt[uuid] = 101;
  ASSERT_EQ(1, tutt.size());
  ASSERT_EQ(101, tutt[uuid]);
}


TEST(ConcreteUtil, uuid_map) {
  concrete::util::uuid_map<int> tutt;
  concrete::util::uuid_factory uf;
  concrete::UUID uuid;
  uuid.__set_uuidString(uf.get_uuid());
  tutt[uuid] = 101;
  ASSERT_EQ(1, tutt.size());
  ASSERT_EQ(101, tutt[uuid]);
}


TEST(ConcreteUtil, tok_id_to_tok_map_from_compact_AFP_EN_19940531_0390_comm) {
  concrete::Communication comm;
  concrete::util::concrete_io concrete_reader;
  const char *name = "test/resources/AFP_ENG_19940531.0390.tcompact.concrete";
  concrete_reader.deserialize_compact<concrete::Communication>(&comm, name);
  ASSERT_EQ("AFP_ENG_19940531.0390", comm.id);
  const concrete::util::uuid_map<concrete::Tokenization> tok_id_to_tptr =
    concrete::util::tokenization_id_to_tokenization(comm);
  // check to make sure all UUID strings are the same
  for(auto& item : tok_id_to_tptr) {
    ASSERT_EQ(item.first.uuidString, item.second.uuid.uuidString);
  };
  ASSERT_EQ(9, tok_id_to_tptr.size());
}


TEST(ConcreteUtil, mention_id_to_tokenization_map_from_compact_AFP_EN_19940531_0390_comm) {
  concrete::Communication comm;
  concrete::util::concrete_io concrete_reader;
  const char *name = "test/resources/AFP_ENG_19940531.0390.tcompact.concrete";
  concrete_reader.deserialize_compact<concrete::Communication>(&comm, name);
  ASSERT_EQ("AFP_ENG_19940531.0390", comm.id);
  const concrete::util::uuid_map<concrete::Tokenization> mention_id_to_tokenization =
    concrete::util::mention_id_to_tokenization(comm, "Stanford");
  ASSERT_EQ(50, mention_id_to_tokenization.size());
}
