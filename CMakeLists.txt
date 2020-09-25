cmake_minimum_required(VERSION 3.8)
project(AMBER)

# Version
set(PROJECT_VERSION_MAJOR 3)
set(PROJECT_VERSION_MINOR 0)
set(PROJECT_VERSION_PATCH 1)

# Includes
include(GNUInstallDirs)

# CMake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
set(TARGET_LINK_LIBRARIES isa_utils isa_opencl astrodata rfim dedispersion integration snr OpenCL)
if($ENV{OPENMP})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()
if($ENV{LOFAR})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_HDF5")
  set(TARGET_LINK_LIBRARIES ${TARGET_LINK_LIBRARIES} hdf5 hdf5_cpp z)
endif()
if($ENV{PSRDADA})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_PSRDADA")
  set(TARGET_LINK_LIBRARIES ${TARGET_LINK_LIBRARIES} psrdada cudart)
endif()

# amber
set(AMBER_HEADER
  include/CommandLine.hpp
  include/configuration.hpp
  include/DataTypes.hpp
  include/Kernels.hpp
  include/Memory.hpp
  include/Trigger.hpp
  include/Pipeline.hpp)
set(AMBER_SOURCE
  src/AMBER.cpp
  src/CommandLine.cpp
  src/Kernels.cpp
  src/Memory.cpp
  src/Trigger.cpp
  src/Pipeline.cpp)
add_executable(amber
  ${AMBER_HEADER}
  ${AMBER_SOURCE}
)
target_include_directories(amber PRIVATE include)
target_link_libraries(amber PRIVATE ${TARGET_LINK_LIBRARIES})

install(TARGETS amber
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
