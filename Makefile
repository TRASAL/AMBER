
# https://github.com/isazi/utils
UTILS := $(HOME)/src/utils
# https://github.com/isazi/OpenCL
OPENCL := $(HOME)/src/OpenCL
# https://github.com/isazi/AstroData
ASTRODATA := $(HOME)/src/AstroData
# https://github.com/isazi/Dedispersion
DEDISPERSION := $(HOME)/src/Dedispersion
# https://github.com/isazi/SNR
SNR := $(HOME)/src/SNR
# http://psrdada.sourceforge.net/
PSRDADA  := $(HOME)/src/psrdada
# Boost
BOOST := $(HOME)/src/boost

INCLUDES := -I"include" -I"$(ASTRODATA)/include" -I"$(UTILS)/include" -I"$(DEDISPERSION)/include" -I"$(SNR)/include" -I"$(PSRDADA)/src/"
CL_INCLUDES := $(INCLUDES) -I"$(OPENCL)/include"
CL_LIBS := -L"$(OPENCL_LIB)"
BOOST_LIBS := -L"$(BOOST)/lib"

CFLAGS := -std=c++11 -Wall
ifneq ($(debug), 1)
	CFLAGS += -O3 -g0
else
	CFLAGS += -O0 -g3
endif
ifeq ($(openmp), 1)
	CFLAGS += -fopenmp
endif

LDFLAGS := -lm
CL_LDFLAGS := $(LDFLAGS) -lOpenCL
HDF5_LDFLAGS := -lhdf5 -lhdf5_cpp
BOOST_LDFLAGS := -lboost_mpi -lboost_serialization

CC := g++
MPI := mpicxx

# Dependencies
KERNELS := $(DEDISPERSION)/bin/Shifts.o $(DEDISPERSION)/bin/Dedispersion.o $(SNR)/bin/SNR.o
DEPS := $(ASTRODATA)/bin/Observation.o $(ASTRODATA)/bin/Platform.o $(UTILS)/bin/ArgumentList.o $(UTILS)/bin/Timer.o $(UTILS)/bin/utils.o
CL_DEPS := $(DEPS) $(OPENCL)/bin/Exceptions.o $(OPENCL)/bin/InitializeOpenCL.o $(OPENCL)/bin/Kernel.o 
DADA_DEPS := $(PSRDADA)/src/dada_hdu.o $(PSRDADA)/src/ipcbuf.o $(PSRDADA)/src/ipcio.o $(PSRDADA)/src/ipcutil.o $(PSRDADA)/src/ascii_header.o $(PSRDADA)/src/multilog.o


all: bin/TransientSearch

bin/TransientSearch: $(CL_DEPS) $(DADA_DEPS) $(KERNELS) $(ASTRODATA)/include/ReadData.hpp $(ASTRODATA)/include/Generator.hpp include/configuration.hpp src/TransientSearch.cpp
	$(MPI) -o bin/TransientSearch src/TransientSearch.cpp $(KERNELS) $(CL_DEPS) $(DADA_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(HDF5_LDFLAGS) $(CL_LDFLAGS) $(CFLAGS)

clean:
	-@rm bin/*

