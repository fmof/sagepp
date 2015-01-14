
CONFIG_FILE := Makefile.config
include $(CONFIG_FILE)

#default target
help:


BUILD_DIR_LINK := $(BUILD_DIR)
RELEASE_BUILD_DIR := $(BUILD_DIR)_release
DEBUG_BUILD_DIR := $(BUILD_DIR)_debug

DEBUG ?= 0
# If we want to debug the generated C++, then 
# explicitly set BUILD=debug
BUILD ?= release
RUNNER ?= $(SHELL)
ifeq ($(DEBUG), 1)
	BUILD_DIR := $(DEBUG_BUILD_DIR)
	OTHER_BUILD_DIR := $(RELEASE_BUILD_DIR)
	BUILD = debug
	RUNNER = gdb
else
	BUILD_DIR := $(RELEASE_BUILD_DIR)
	OTHER_BUILD_DIR := $(DEBUG_BUILD_DIR)
	RUNNER += -c 
endif

TOP_LEVEL_SRC = src
SRC_SUB_DIRS=$(shell find $(TOP_LEVEL_SRC) -mindepth 1 -type d)

BUILD_WHERE=$(BUILD_DIR)/build
INCLUDE_WHERE=$(BUILD_DIR)/include
LIB_WHERE=$(BUILD_DIR)/lib
EXC_WHERE=$(BUILD_DIR)/exec

ALL_BUILD_DIRS := $(sort $(BUILD_DIR) $(BUILD_WHERE) \
	$(INCLUDE_WHERE) $(LIB_WHERE) $(EXC_WHERE) \
	$(foreach build,$(BUILD_WHERE) $(INCLUDE_WHERE), \
		$(subst $(TOP_LEVEL_SRC)/,$(build)/,$(SRC_SUB_DIRS))))

# The target shared library and static library name
NAME := $(LIB_WHERE)/lib$(PROJECT).so
STATIC_NAME := $(LIB_WHERE)/lib$(PROJECT).a

# We're always going to compile to C++ 2011 standard
CXX_BUILD_FLAGS_BASE = -std=c++11 -Wall
CXX_BUILD_FLAGS_debug   = -g -O0
CXX_BUILD_FLAGS_release = -O3

# set the basic compile options
CXXFLAGS += $(CXX_BUILD_FLAGS_BASE) $(CXX_BUILD_FLAGS_$(BUILD)) 
# add thrift
CXXFLAGS += -I$(THRIFT_INCLUDE_DIR) -L$(THRIFT_LIB_DIR) $(CXX_THRIFT_FLAGS)
# add concrete
CXXFLAGS += -I$(CONCRETE_INCLUDE_DIR) -L$(CONCRETE_LIB_DIR)
CXXFLAGS += -I$(CONCRETE_UTIL_INCLUDE_DIR) -L$(CONCRETE_UTIL_LIB_DIR)
# add Boost
CXXFLAGS += $(CXX_BOOST_FLAGS) -I$(BOOST_INCLUDE_DIR) -L$(BOOST_LIB_DIR) 
# add GSL
CXXFLAGS += -I$(GSL_INCLUDE_DIR) -L$(GSL_LIB_DIR)

# for linking
LITTLE_L_FLAGS := $(foreach library,$(LITTLE_Ls),-l$(library))
# The -fPIC is needed to generate position independent code, which
# is required for creating a shared library. The position independent
# code makes the generated machine code use relative rather than 
# absolute addresses.
CXX_SHARED_FLAGS := -shared -fPIC
CXX_LINKING_FLAGS := $(LITTLE_L_FLAGS)


################################################################

.PHONY: isage clean env_set help test test-clean list-models

##################################################
##################################################
######## GENERATED/BASIC CONCRETE TARGETS ########
##################################################
##################################################

##############################
# Get all source files
##############################

ISAGE_SRC_DIR = $(TOP_LEVEL_SRC)/isage
ISAGE_SRC_ = $(shell find $(ISAGE_SRC_DIR) -type f -name '*.cpp')
ISAGE_H_ = $(shell find $(ISAGE_SRC_DIR) -type f -name '*.hpp')
ISAGE_OBJ_ = $(patsubst %.cpp,%.o,$(subst $(TOP_LEVEL_SRC)/,$(BUILD_WHERE)/,$(ISAGE_SRC_)))
ISAGE_DEPS_ = $(patsubst %.cpp,%.d,$(subst $(TOP_LEVEL_SRC)/,$(BUILD_WHERE)/,$(ISAGE_SRC_)))

.PRECIOUS: $(ISAGE_DEPS_)

# use an order-only prerequisite on $(LIB_WHERE) to prevent 
# directory creation timestamps from messing with the build
$(BUILD_WHERE)/isage/%.d: $(ISAGE_SRC_DIR)/%.cpp | $(ALL_BUILD_DIRS)
#	@echo "Could not find: $@"
	@set -e; rm -f $@; \
	$(CXX) -MM $(CXX_SHARED_FLAGS) $(CXXFLAGS) -I$(ISAGE_SRC_DIR) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,$(BUILD_WHERE)/isage/\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$
	$(call verify_exist,$@)

$(BUILD_WHERE)/isage/%.o: $(BUILD_WHERE)/isage/%.d $(ISAGE_SRC_DIR)/%.cpp $(ISAGE_H_) | $(ALL_BUILD_DIRS) 
	$(CXX) $(CXX_SHARED_FLAGS) $(CXXFLAGS) -I$(ISAGE_SRC_DIR) -c $(ISAGE_SRC_DIR)/$*.cpp -o $@

isage: $(NAME) $(STATIC_NAME)

# use an order-only prerequisite on $(LIB_WHERE) to prevent 
# directory creation timestamps from messing with the build
$(NAME): $(ISAGE_OBJ_) | $(ALL_BUILD_DIRS)
	$(CXX) $(CXXFLAGS) $(CXX_SHARED_FLAGS) -o $@ $(ISAGE_OBJ_) $(CXX_LINKING_FLAGS)
	@ echo
$(STATIC_NAME): $(ISAGE_OBJ_) | $(ALL_BUILD_DIRS)
	ar rcs $@ $(ISAGE_OBJ_)
	@ echo

$(ALL_BUILD_DIRS):
	@ mkdir -p $@

##################################################
##################################################
################ MODEL TARGETS ###################
##################################################
##################################################

CONC_TEST_FLAGS += -I$(CONCRETE_INCLUDE_DIR) -L$(CONCRETE_LIB_DIR)
CONC_TEST_FLAGS += -I$(CONCRETE_UTIL_INCLUDE_DIR) -L$(CONCRETE_UTIL_LIB_DIR)

MODEL_SRC_DIRS = $(TOP_LEVEL_SRC)/models
MODEL_CPP_FILES_ = $(foreach subdir,$(MODEL_SRC_DIRS),$(wildcard $(subdir)/*.cpp))
MODEL_OBJ_FILES_ = $(subst $(TOP_LEVEL_SRC)/,$(EXC_WHERE)/,$(MODEL_CPP_FILES_:.cpp=.o))
UNIT_MODELS_NO_FULL = $(filter-out $(MODEL_SRC_DIRS),$(subst $(TOP_LEVEL_SRC)/,,$(MODEL_CPP_FILES_:.cpp=)))
UNIT_MODELS = $(filter-out $(MODEL_SRC_DIRS),$(subst $(TOP_LEVEL_SRC)/,$(EXC_WHERE)/,$(MODEL_CPP_FILES_:.cpp=)))

EXC_DIRS = $(foreach srcsubdir,$(MODEL_SRC_DIRS), \
				$(subst $(TOP_LEVEL_SRC)/,$(EXC_WHERE)/,$(srcsubdir)))

MODEL_DEPS_ = $(patsubst %.cpp,%.d,$(subst $(TOP_LEVEL_SRC)/,$(EXC_WHERE)/,$(MODEL_CPP_FILES_)))

.PRECIOUS: $(MODEL_DEPS_)

$(EXC_WHERE)/models/%.d: $(MODEL_SRC_DIRS)/%.cpp | $(ALL_BUILD_DIRS)
	@if [ ! -d $(@D) ]; then mkdir $(@D); fi
	@set -e; rm -f $@; \
	$(CXX) -MM $(CXX_SHARED_FLAGS) $(CXXFLAGS) -I$(ISAGE_SRC_DIR) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,$(EXC_WHERE)/models/\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$
	$(call verify_exist,$@)

$(EXC_WHERE)/models/%.o : $(MODEL_SRC_DIRS)/%.cpp $(EXC_WHERE)/models/%.d
	@echo "Trying to compile and assemble model driver $<"
	@$(ECHO) "$(STATUS_COLOR)Trying to compile and assemble model driver $(OK_COLOR)$(EXC_WHERE)/models/$*$(NO_COLOR)" 
	$(CXX) $(CXXFLAGS) $(CONC_TEST_FLAGS) -I$(ISAGE_SRC_DIR) -c $< -o $@

$(UNIT_MODELS) : $(EXC_DIRS)/% : $(EXC_WHERE)/models/%.o $(NAME) | $(@D)
	@if [ ! -d $(@D) ]; then mkdir $(@D); fi
	echo $^
	$(CXX) $(CXXFLAGS) -L$(LIB_WHERE) -I$(ISAGE_SRC_DIR) -o $@ -lthrift $^ -lisage -lthrift $(CONCRETE_SO) $(CONCRETE_UTIL_LIB_DIR)/concrete_util.so $(THRIFT_LIB_DIR)/libthrift.so $(CXX_LINKING_FLAGS)
#	$(CXX) $(CXXFLAGS) -L$(LIB_WHERE) -I$(ISAGE_SRC_DIR) -o $@ $^ -lisage $(CXX_LINKING_FLAGS)
	$(call verify_exist,$@)
	@$(ECHO) "Executable $(STATUS_COLOR)$@$(NO_COLOR) ready to run"
	@$(ECHO) ""

$(UNIT_MODELS_NO_FULL): % : $(EXC_WHERE)/%

list-models:
	@$(ECHO) "Known model driver targets:"
	$(foreach m,$(UNIT_MODELS_NO_FULL),@$(ECHO) "\t$(OK_COLOR)$(m)$(NO_COLOR)")

##################################################
##################################################
################ CLEAN TARGETS ###################
##################################################
##################################################

clean: test-clean
	find . -type f -name '*~' -delete
	rm -rf $(BUILD_WHERE) $(INCLUDE_WHERE) $(LIB_WHERE) $(BUILD_DIR) $(EXC_WHERE)

TEST_DIR = test
TEST_SRC_DIR_PREFIX = $(TEST_DIR)
#TEST_SRC_DIRS = $(foreach srcsubdir,$(SRC_SUB_DIRS), \
#			$(subst $(TOP_LEVEL_SRC)/,$(TEST_DIR)/,$(srcsubdir)))
TEST_SRC_DIRS = $(TEST_DIR)/isage
TEST_EXC_DIR = $(TEST_DIR)/exc
TEST_BUILD_DIR = $(TEST_DIR)/build
TEST_LIB_DIR = $(TEST_DIR)/lib

TEST_EXC_DIRS = $(foreach srcsubdir,$(TEST_SRC_DIRS), \
			$(subst $(TEST_SRC_DIR_PREFIX)/,$(TEST_EXC_DIR)/,$(srcsubdir)))
TEST_BUILD_DIRS = $(foreach srcsubdir,$(TEST_SRC_DIRS), \
			$(subst $(TEST_SRC_DIR_PREFIX)/,$(TEST_BUILD_DIR)/,$(srcsubdir)))
TEST_LIB_DIRS = $(foreach srcsubdir,$(TEST_SRC_DIRS), \
			$(subst $(TEST_SRC_DIR_PREFIX)/,$(TEST_LIB_DIR)/,$(srcsubdir)))

ALL_TEST_BUILD_DIRS = $(TEST_EXC_DIRS) $(TEST_BUILD_DIRS) $(TEST_LIB_DIRS)

test-clean:
	rm -fr $(TESTS) $(ALL_TEST_BUILD_DIRS)

##################################################
##################################################
################# TEST TARGETS ###################
##################################################
##################################################

GTEST_INCLUDE := -isystem ${GTEST_DIR}/include -I$(GTEST_DIR)/include -I$(GTEST_DIR)
GTEST_LIB := -L$(GTEST_DIR)/lib
GTEST_LIB_LINKING_FLAGS := -lpthread
GTEST_FLAGS += $(GTEST_INCLUDE) $(GTEST_LIB) -pthread $(GTEST_LIB_LINKING_FLAGS)

GTEST_HEADERS_ = $(GTEST_DIR)/include/gtest/*.h \
                 $(GTEST_DIR)/include/gtest/internal/*.h
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS_)

#CXXFLAGS = $(PASSED_CXX) -isystem $(GTEST_DIR)/include -I$(GTEST_DIR)/include -I$(GTEST_DIR)
#CXXFLAGS += $(TEST_INCLUDE) $(TEST_LIB) $(TEST_FLAGS)

AR=ar -rv

TEST_CPP_FILES_ = $(foreach subdir,$(TEST_SRC_DIRS),$(wildcard $(subdir)/*.cpp))
TEST_DEPS_ = $(subst $(TEST_SRC_DIR_PREFIX)/,$(TEST_BUILD_DIR)/,$(TEST_CPP_FILES_:.cpp=.d))
TEST_OBJ_FILES_ = $(subst $(TEST_SRC_DIR_PREFIX)/,$(TEST_BUILD_DIR)/,$(TEST_CPP_FILES_:.cpp=.o))
UNIT_TESTS_NO_FULL = $(filter-out $(ALL_TEST_BUILD_DIRS),$(subst $(TEST_SRC_DIR_PREFIX)/,,$(TEST_CPP_FILES_:.cpp=)))
UNIT_TESTS = $(filter-out $(ALL_TEST_BUILD_DIRS),$(subst $(TEST_SRC_DIR_PREFIX)/,$(TEST_EXC_DIR)/,$(TEST_CPP_FILES_:.cpp=)))


# For simplicity and to avoid depending on Google Test's
# implementation details, the dependencies specified below are
# conservative and not optimized.  This is fine as Google Test
# compiles fast and for ordinary users its source rarely changes.
$(TEST_LIB_DIR)/gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(GTEST_FLAGS) $(CXXFLAGS) -c $(GTEST_DIR)/src/gtest-all.cc -o $@ $(CXX_LINKING_FLAGS)
$(TEST_LIB_DIR)/gtest.a : $(TEST_LIB_DIR)/gtest-all.o
	$(AR) $@ $^

$(TEST_LIB_DIR)/gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(GTEST_FLAGS) $(CXXFLAGS) -c $(GTEST_DIR)/src/gtest_main.cc -o $@ $(CXX_LINKING_FLAGS)
$(TEST_LIB_DIR)/gtest_main.a : $(TEST_LIB_DIR)/gtest-all.o $(TEST_LIB_DIR)/gtest_main.o
	$(AR) $@ $^

$(TEST_BUILD_DIRS)/%.d : $(TEST_SRC_DIRS)/%.cpp 
#	@$(ECHO) "Could not find: $@"
	@if [ ! -d $(@D) ]; then mkdir $(@D); fi
	@set -e; rm -f $@; \
	$(CXX) -MM $(CXXFLAGS) $(GTEST_FLAGS) $(CONC_TEST_FLAGS) -I$(subst $(TEST_BUILD_DIR)/,$(TOP_LEVEL_SRC)/,$(@D)) $^ > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,$(@D)/models/\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$
	$(call verify_exist,$@)

$(TEST_BUILD_DIRS)/%.o : $(TEST_SRC_DIRS)/%.cpp $(GTEST_HEADERS_)
	@$(ECHO) "$(STATUS_COLOR)Trying to compile and assemble unit test $(OK_COLOR) $(TEST_BUILD_DIRS)/$* $(NO_COLOR)" 
	$(CXX) $(CXXFLAGS) $(GTEST_FLAGS) $(CONC_TEST_FLAGS) -I$(subst $(TEST_BUILD_DIR)/,$(TOP_LEVEL_SRC)/,$(@D)) -c $< -o $@

$(UNIT_TESTS) : $(TEST_EXC_DIR)/% : $(TEST_BUILD_DIR)/%.o $(TEST_LIB_DIR)/gtest_main.a $(NAME) | $(@D)
	@if [ ! -d $(@D) ]; then mkdir -p $(@D); fi
# unfortunately, some libraries must be repeated (hence the multiple -l{thrift,gsl,gslcblas,m}
	$(CXX) $(CXXFLAGS) $(GTEST_FLAGS) -L$(LIB_WHERE) -o $@ -lthrift $< -lgsl -lgslcblas -lm $(filter-out $<,$^) $(GTEST_LIB_LINKING_FLAGS) -lisage -lthrift $(CONCRETE_SO) $(THRIFT_LIB_DIR)/libthrift.so $(CONCRETE_UTIL_LIB_DIR)/concrete_util.so  $(CXX_LINKING_FLAGS)
	$(call verify_exist,$(TEST_EXC_DIR)/$*)
	@$(ECHO) "Running unit test $(STATUS_COLOR)$(TEST_EXC_DIR)/$*$(NO_COLOR)"
	$(RUNNER) $@ 
	$(eval RAN_TEST_$(TEST_EXC_DIR)/$*=1)
	@$(ECHO) ""

.PHONY: $(UNIT_TESTS_NO_FULL)
$(UNIT_TESTS_NO_FULL): % : $(TEST_EXC_DIR)/% | $(ALL_TEST_BUILD_DIRS)
	@if [ -z $(RAN_TEST_$<) ]; then $(RUNNER) $<; fi
list-tests:
	@$(ECHO) "Known unit tests:"
	@$(ECHO) $(foreach m,$(UNIT_TESTS_NO_FULL), "\t$(OK_COLOR)$(m)$(NO_COLOR)\n")

$(ALL_TEST_BUILD_DIRS):
	mkdir -p $@

test: $(NAME) $(ALL_TEST_BUILD_DIRS) $(UNIT_TESTS)

help:
	@echo '--------------------------------------------------------------------------------'
	@echo 'isage makefile:'
	@echo '    Known targets:'
	@echo '     - isage:       Compile the main isage library.'
	@echo '     - test:        Run all schemapp tests'
	@echo '     - list-tests:  Find which tests can be run individually. For a test, e.g., libnar/test_foo, run "make libnar/test_foo"'
	@$(ECHO) $(foreach m,$(UNIT_TESTS_NO_FULL), "\t\t* $(OK_COLOR)$(m)$(NO_COLOR)\n")
	@echo '     - list-models'
	@$(ECHO) $(foreach m,$(UNIT_MODELS_NO_FULL),"\t\t* $(OK_COLOR)$(m)$(NO_COLOR)\n")
	@echo '--------------------------------------------------------------------------------'

-include $(ISAGE_DEPS_)
-include $(MODEL_DEPS_)
-include $(TEST_DEPS_)
