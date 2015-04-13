
#include <Shifts.hpp>
#include <Dedispersion.hpp>
#include <SNR.hpp>

#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

// Types for the data
typedef float dataType;
const std::string dataName("float");

// DEBUG mode, prints to screen some useful information
const bool DEBUG = true;

// SYNC mode, OpenCL queue operations
const bool SYNC = true;

#endif // CONFIGURATION_HPP

