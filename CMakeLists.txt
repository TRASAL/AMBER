cmake_minimum_required(VERSION 3.10)
project(AMBER VERSION 2.1)
include(GNUInstallDirs)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g0 -march=native -mtune=native")

add_executable(amber
  src/CommandLine.cpp
  src/TransientSearch.cpp
  src/Trigger.cpp
)
target_include_directories(amber PRIVATE include)
target_link_libraries(amber dedispersion integration snr isa_utils isa_opencl astrodata OpenCL)
if($ENV{LOFAR})
  target_link_libraries(amber hdf5 hdf5_cpp z)
endif()
if($ENV{PSRDADA})
  target_link_libraries(amber psrdada cudart)
endif()

install(TARGETS amber
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
