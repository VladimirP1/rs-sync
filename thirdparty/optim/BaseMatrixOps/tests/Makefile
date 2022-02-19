# BMO Test build

ifeq ($(CXX),)
	CXX = g++
endif

ifeq ($(BMO_CXX_STD),)
	ifneq ($(BMO_ENABLE_EIGEN_WRAPPERS),)
		CXX_STD = -std=c++14
	else
		CXX_STD = -std=c++11
	endif
else
	CXX_STD=$(BMO_CXX_STD)
endif

OPT_FLAGS = -g -O0 -Wall -Wextra

# source directories
SDIR = .
HEADER_DIR = $(SDIR)/../include

ifneq ($(BMO_ENABLE_EIGEN_WRAPPERS),)
	OPT_FLAGS := $(OPT_FLAGS) -DBMO_ENABLE_EIGEN_WRAPPERS
	HEADER_DIR := $(HEADER_DIR) -I$(EIGEN_INCLUDE_PATH)
else ifneq ($(BMO_ENABLE_ARMA_WRAPPERS),)
	OPT_FLAGS := $(OPT_FLAGS) -DBMO_ENABLE_ARMA_WRAPPERS
	HEADER_DIR := $(HEADER_DIR) -I$(ARMA_INCLUDE_PATH)
endif

SOURCES_TESTS= $(wildcard $(SDIR)/*.cpp)
OBJECTS_TESTS= $(SOURCES_TESTS:.cpp=.test)

all: $(OBJECTS_TESTS)

$(SDIR)/%.test: $(SDIR)/%.cpp
	$(CXX) $(CXX_STD) $(OPT_FLAGS) -I$(HEADER_DIR) $< -o $@ $(LIBS)

# cleanup
.PHONY: clean
clean:
	@rm -rf *.test *.gcov *.gcno *.gcda *.dSYM

# individual make calls

BMO_MAKE_CALL= $(CXX) $(CXX_STD) $(OPT_FLAGS) -I$(HEADER_DIR) $@.cpp -o $@.test $(LIBS)

log_vec:
	$(BMO_MAKE_CALL)

rand:
	$(BMO_MAKE_CALL)
