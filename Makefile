
INSTALL_ROOT ?= $(HOME)
INCLUDES := -I"include" -I"$(INSTALL_ROOT)/include"
LIBS := -L"$(INSTALL_ROOT)/lib"

CC := g++
CFLAGS := -std=c++11 -Wall
LDFLAGS := -lm -lutils -lOpenCL -lisaOpenCL -lAstroData -lDedispersion -lIntegration -lSNR

ifdef DEBUG
	CFLAGS += -O0 -g3
else
	CFLAGS += -O3 -g0
endif

ifdef LOFAR
	CFLAGS += -DHAVE_HDF5
	INCLUDES += -I"$(HDF5INCLUDE)"
	LIBS += -L"$(HDF5DIR)"
	LDFLAGS += -lhdf5 -lhdf5_cpp -lz
endif
ifdef PSRDADA
	CFLAGS += -DHAVE_PSRDADA
	LDFLAGS += -lpsrdada -lcudart
endif
ifdef OPENMP
	CFLAGS += -fopenmp
endif

all: bin/CommandLine.o bin/Trigger.o bin/TransientSearch

bin/CommandLine.o: include/CommandLine.hpp src/CommandLine.cpp
	-@mkdir -p bin
	$(CC) -o bin/CommandLine.o -c src/CommandLine.cpp $(INCLUDES) $(CFLAGS)

bin/Trigger.o: include/Trigger.hpp src/Trigger.cpp
	-@mkdir -p bin
	$(CC) -o bin/Trigger.o -c src/Trigger.cpp $(INCLUDES) $(CFLAGS)

bin/TransientSearch: include/configuration.hpp src/TransientSearch.cpp
	-@mkdir -p bin
	$(CC) -o bin/amber src/TransientSearch.cpp bin/CommandLine.o bin/Trigger.o $(INCLUDES) $(LIBS) $(LDFLAGS) $(CFLAGS)

clean:
	-@rm bin/*

install: all
	-@mkdir -p $(INSTALL_ROOT)/bin
	-@cp bin/amber $(INSTALL_ROOT)/bin
