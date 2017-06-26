
SOURCE_ROOT ?= $(HOME)

# https://github.com/isazi/utils
UTILS := $(SOURCE_ROOT)/utils
# https://github.com/isazi/OpenCL
OPENCL := $(SOURCE_ROOT)/OpenCL
# https://github.com/isazi/AstroData
ASTRODATA := $(SOURCE_ROOT)/AstroData
# https://github.com/isazi/Dedispersion
DEDISPERSION := $(SOURCE_ROOT)/Dedispersion
# https://github.com/isazi/Integration
INTEGRATION := $(SOURCE_ROOT)/Integration
# https://github.com/isazi/SNR
SNR := $(SOURCE_ROOT)/SNR

# HDF5
HDF5_INCLUDE ?= -I/usr/include
HDF5_LDFLAGS ?= -L/usr/lib
HDF5_LIBS ?= -lhdf5 -lhdf5_cpp -lz

INCLUDES := -I"include" -I"$(ASTRODATA)/include" -I"$(UTILS)/include" -I"$(DEDISPERSION)/include" -I"$(INTEGRATION)/include" -I"$(SNR)/include" $(HDF5_INCLUDE) -I"$(PSRDADA)/src/"
CL_INCLUDES := $(INCLUDES) -I"$(OPENCL)/include"

CFLAGS := -std=c++11 -Wall
ifdef DEBUG
	CFLAGS += -O0 -g3
else
	CFLAGS += -O3 -g0
endif
ifdef OPENMP
	CFLAGS += -fopenmp
endif
ifdef PSRDADA
CFLAGS += -DHAVE_PSRDADA
DADA_DEPS := $(PSRDADA)/src/dada_hdu.o $(PSRDADA)/src/ipcbuf.o $(PSRDADA)/src/ipcio.o $(PSRDADA)/src/ipcutil.o $(PSRDADA)/src/ascii_header.o $(PSRDADA)/src/multilog.o $(PSRDADA)/src/tmutil.o
else
DADA_DEPS := 
endif

LDFLAGS := -lm
CL_LDFLAGS := $(LDFLAGS) -lOpenCL

CC := g++

# Dependencies
KERNELS := $(DEDISPERSION)/bin/Shifts.o $(DEDISPERSION)/bin/Dedispersion.o $(INTEGRATION)/bin/Integration.o $(SNR)/bin/SNR.o
DEPS := $(ASTRODATA)/bin/Observation.o $(ASTRODATA)/bin/Platform.o $(ASTRODATA)/bin/ReadData.o $(UTILS)/bin/ArgumentList.o $(UTILS)/bin/Timer.o $(UTILS)/bin/utils.o
CL_DEPS := $(DEPS) $(OPENCL)/bin/Exceptions.o $(OPENCL)/bin/InitializeOpenCL.o $(OPENCL)/bin/Kernel.o


all: bin/BeamDriver.o bin/Trigger.o bin/TransientSearch

bin/BeamDriver.o: include/BeamDriver.hpp src/BeamDriver.cpp
	-@mkdir -p bin
	$(CC) -o bin/BeamDriver.o -c src/BeamDriver.cpp $(INCLUDES) $(CFLAGS)

bin/Trigger.o: include/Trigger.hpp src/Trigger.cpp
	-@mkdir -p bin
	$(CC) -o bin/Trigger.o -c src/Trigger.cpp $(INCLUDES) $(CFLAGS)

bin/TransientSearch: $(CL_DEPS) $(DADA_DEPS) $(KERNELS) bin/BeamDriver.o bin/Trigger.o $(ASTRODATA)/include/ReadData.hpp $(ASTRODATA)/include/Generator.hpp include/configuration.hpp src/TransientSearch.cpp
	-@mkdir -p bin
	$(CC) -o bin/TransientSearch src/TransientSearch.cpp bin/BeamDriver.o bin/Trigger.o $(KERNELS) $(CL_DEPS) $(DADA_DEPS) $(HDF5_INCLUDE) $(CL_INCLUDES) $(CL_LIBS) $(HDF5_LDFLAGS) $(HDF5_LIBS) $(CL_LDFLAGS) $(CFLAGS)

clean:
	-@rm bin/*

