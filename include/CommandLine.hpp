// Copyright 2017 Netherlands Institute for Radio Astronomy (ASTRON)
// Copyright 2017 Netherlands eScience Center
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "configuration.hpp"

#pragma once

struct Options {
  // Debug mode
  bool debug;
  // Print messages to standard output
  bool print;
  // Use subband dedispersion
  bool subbandDedispersion;
  // Compact the triggered events in time and DM dimensions
  bool compactResults;
  // Threshold for triggering
  float threshold;
};

struct DeviceOptions {
  // OpenCL platform ID
  unsigned int platformID;
  // OpenCL device ID
  unsigned int deviceID;
  // OpenCL device name
  std::string deviceName;
  // Padding of OpenCL devices
  AstroData::paddingConf padding;
};

struct DataOptions {
  // Use LOFAR file as input
  bool dataLOFAR;
  // Use SIGPROC file as input
  bool dataSIGPROC;
  // Use PSRDADA buffer as input
  bool dataPSRDADA;
  // Limit the number of batches processed from a LOFAR file
  bool limit;
  // Size (in bytes) of the SIGPROC file header
  unsigned int headerSizeSIGPROC;
  // Name of the input file
  std::string dataFile;
  // Name of the LOFAR header file
  std::string headerFile;
  // Basename for the output files
  std::string outputFile;
  // Name of the file containing the zapped channels
  std::string channelsFile;
  // Name of the file containing the integration steps
  std::string integrationFile;
#ifdef HAVE_PSRDADA
  // PSRDADA buffer key
  key_t dadaKey;
#endif // HAVE_PSRDADA
};

struct Configurations {
  // Configuration of single step dedispersion kernel
  Dedispersion::tunedDedispersionConf dedispersionParameters;
  // Configuration of subband dedispersion kernel, step one
  Dedispersion::tunedDedispersionConf dedispersionStepOneParameters;
  // Configuration of subband dedispersion kernel, step two
  Dedispersion::tunedDedispersionConf dedispersionStepTwoParameters;
  // Configuration of integration kernel
  Integration::tunedIntegrationConf integrationParameters;
  // Configuration of SNR kernel
  SNR::tunedSNRConf snrParameters;
};

struct GeneratorOptions {
  // Use random numbers in generated data
  bool random;
  // Width of random generated pulse
  unsigned int width;
  // DM of random generated pulse
  float DM;
};

// Function to process the command line options
void processCommandLineOptions(isa::utils::ArgumentList & argumentList, Options & options, DeviceOptions & deviceOptions, DataOptions & dataOptions, Configurations & configurations, GeneratorOptions & generatorOptions, AstroData::Observation & observation);
// Function to print the usage message
void usage(const std::string & program);
