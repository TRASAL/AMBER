cmake_minimum_required(VERSION 3.10)
project(AMBER VERSION 2.1)
include(GNUInstallDirs)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
set(TARGET_LINK_LIBRARIES isa_utils isa_opencl astrodata dedispersion integration snr OpenCL)
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
add_executable(amber
  src/CommandLine.cpp
  src/TransientSearch.cpp
  src/Trigger.cpp
)
target_include_directories(amber PRIVATE include)
target_link_libraries(amber PRIVATE ${TARGET_LINK_LIBRARIES})

install(TARGETS amber
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
