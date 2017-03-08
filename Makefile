
SOURCE_ROOT ?= $(HOME)

# https://github.com/isazi/utils
UTILS := $(SOURCE_ROOT)/src/utils
# https://github.com/isazi/OpenCL
OPENCL := $(SOURCE_ROOT)/src/OpenCL
# https://github.com/isazi/AstroData
ASTRODATA := $(SOURCE_ROOT)/src/AstroData
# https://github.com/isazi/Dedispersion
DEDISPERSION := $(SOURCE_ROOT)/src/Dedispersion
# https://github.com/isazi/Integration
INTEGRATION := $(SOURCE_ROOT)/src/Integration
# https://github.com/isazi/SNR
SNR := $(SOURCE_ROOT)/src/SNR
# HDF5
HDF5 := $(SOURCE_ROOT)/src/HDF5
# http://psrdada.sourceforge.net/
PSRDADA  := $(SOURCE_ROOT)/src/psrdada

INCLUDES := -I"include" -I"$(ASTRODATA)/include" -I"$(UTILS)/include" -I"$(DEDISPERSION)/include" -I"$(INTEGRATION)/include" -I"$(SNR)/include" -I"$(HDF5)/include" -I"$(PSRDADA)/src/"
CL_INCLUDES := $(INCLUDES) -I"$(OPENCL)/include"
CL_LIBS := -L"$(OPENCL_LIB)"
HDF5_LIBS := -L"$(HDF5)/lib"

CFLAGS := -std=c++11 -Wall
ifneq ($(DEBUG), 1)
	CFLAGS += -O3 -g0
else
	CFLAGS += -O0 -g3
endif
ifeq ($(openmp), 1)
	CFLAGS += -fopenmp
endif

LDFLAGS := -lm
CL_LDFLAGS := $(LDFLAGS) -lOpenCL
HDF5_LDFLAGS := -lhdf5 -lhdf5_cpp -lz

CC := g++

# Dependencies
KERNELS := $(DEDISPERSION)/bin/Shifts.o $(DEDISPERSION)/bin/Dedispersion.o $(INTEGRATION)/bin/Integration.o $(SNR)/bin/SNR.o
DEPS := $(ASTRODATA)/bin/Observation.o $(ASTRODATA)/bin/Platform.o $(ASTRODATA)/bin/ReadData.o $(UTILS)/bin/ArgumentList.o $(UTILS)/bin/Timer.o $(UTILS)/bin/utils.o
CL_DEPS := $(DEPS) $(OPENCL)/bin/Exceptions.o $(OPENCL)/bin/InitializeOpenCL.o $(OPENCL)/bin/Kernel.o
DADA_DEPS := $(PSRDADA)/src/dada_hdu.o $(PSRDADA)/src/ipcbuf.o $(PSRDADA)/src/ipcio.o $(PSRDADA)/src/ipcutil.o $(PSRDADA)/src/ascii_header.o $(PSRDADA)/src/multilog.o $(PSRDADA)/src/tmutil.o


all: bin/BeamDriver.o bin/TransientSearch bin/printTimeSeries

bin/BeamDriver.o: include/BeamDriver.hpp src/BeamDriver.cpp
	$(CC) -o bin/BeamDriver.o -c src/BeamDriver.cpp $(INCLUDES) $(CFLAGS)

bin/TransientSearch: $(CL_DEPS) $(DADA_DEPS) $(KERNELS) bin/BeamDriver.o $(ASTRODATA)/include/ReadData.hpp $(ASTRODATA)/include/Generator.hpp include/configuration.hpp src/TransientSearch.cpp
	$(CC) -o bin/TransientSearch src/TransientSearch.cpp bin/BeamDriver.o $(KERNELS) $(CL_DEPS) $(DADA_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(HDF5_LIBS) $(CL_LDFLAGS) $(HDF5_LDFLAGS) $(CFLAGS)

bin/printTimeSeries: $(DEPS) $(DADA_DEPS) $(DEDISPERSION)/bin/Shifts.o $(ASTRODATA)/include/ReadData.hpp $(ASTRODATA)/bin/ReadData.o include/configuration.hpp src/printTimeSeries.cpp
	$(CC) -o bin/printTimeSeries src/printTimeSeries.cpp $(DEPS) $(DEDISPERSION)/bin/Shifts.o $(DADA_DEPS) $(CL_INCLUDES) -I"$(PSRDADA)/src" -I"$(HDF5)/include" $(HDF5_LIBS) $(CL_LDFLAGS) $(HDF5_LDFLAGS) $(CFLAGS)

clean:
	-@rm bin/*

