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


#pragma once

struct Options {
  // Debug mode
  bool debug = false;
  // Print messages to standard output
  bool print = false;
  // Use subband dedispersion
  bool subbandDedispersion = false;
  // Avoid merging batches of dispersed data into contiguous memory
  bool splitBatchesDedispersion = false;
  // Compact the triggered events in time and DM dimensions
  bool compactResults = false;
  // Threshold for triggering
  float threshold = 0.0f;
};

struct DeviceOptions {
  // OpenCL synchronized operations
  bool synchronized = false;
  // OpenCL platform ID
  unsigned int platformID = 0;
  // OpenCL device ID
  unsigned int deviceID = 0;
  // OpenCL device name
  std::string deviceName{};
  // Padding of OpenCL devices
  AstroData::paddingConf padding{};
};

struct DataOptions {
  // Use LOFAR file as input
  bool dataLOFAR = false;
  // Use SIGPROC file as input
  bool dataSIGPROC = false;
  // Use PSRDADA buffer as input
  bool dataPSRDADA = false;
  // Limit the number of batches processed from a LOFAR file
  bool limit = false;
  // Size (in bytes) of the SIGPROC file header
  unsigned int headerSizeSIGPROC = 0;
  // Name of the input file
  std::string dataFile{};
  // Name of the LOFAR header file
  std::string headerFile{};
  // Basename for the output files
  std::string outputFile{};
  // Name of the file containing the zapped channels
  std::string channelsFile{};
  // Name of the file containing the integration steps
  std::string integrationFile{};
#ifdef HAVE_PSRDADA
  // PSRDADA buffer key
  key_t dadaKey = 0;
#endif // HAVE_PSRDADA
};

struct GeneratorOptions {
  // Use random numbers in generated data
  bool random = false;
  // Width of random generated pulse
  unsigned int width = 0;
  // DM of random generated pulse
  float DM = 0.0f;
};

struct HostMemory {
  // Input data
  std::vector<std::vector<std::vector<inputDataType> *> *> input;
  // Zapped channels
  std::vector<unsigned int> zappedChannels;
  // Integration steps
  std::set<unsigned int> integrationSteps;
  // Map to create synthesized beams
  std::vector<unsigned int> beamMapping;
  // Dispersed data
  std::vector<inputDataType> dispersedData;
  // Subbanded data
  std::vector<outputDataType> subbandedData;
  // Dedispersed data
  std::vector<outputDataType> dedispersedData;
  // Integrated data
  std::vector<outputDataType> integratedData;
  // SNR data
  std::vector<float> snrData;
  // Index of the sample with highest SNR value
  std::vector<unsigned int> snrSamples;
  // Shifts single step dedispersion
  std::vector<float> * shiftsSingleStep;
  // Shifts step one subbanding dedispersion
  std::vector<float> * shiftsStepOne;
  // Shifts step two subbanding dedispersion
  std::vector<float> * shiftsStepTwo;
#ifdef HAVE_PSRDADA
  // PSRDADA ring buffer
  dada_hdu_t * ringBuffer = 0;
  // Input data
  std::vector<std::vector<inputDataType> *> inputDADA;
#endif // HAVE_PSRDADA
};

struct DeviceMemory {
  // Shifts single step dedispersion
  cl::Buffer shiftsSingleStep;
  // Shifts step one subbanding dedispersion
  cl::Buffer shiftsStepOne;
  // Shifts step two subbanding dedispersion
  cl::Buffer shiftsStepTwo;
  // Zapped channels
  cl::Buffer zappedChannels;
  // Map to create synthesized beams
  cl::Buffer beamMapping;
  // Dispersed dada
  cl::Buffer dispersedData;
  // Subbanded data
  cl::Buffer subbandedData;
  // Dedispersed data
  cl::Buffer dedispersedData;
  // Integrated data
  cl::Buffer integratedData;
  // SNR data
  cl::Buffer snrData;
  // Index of the sample with highest SNR value
  cl::Buffer snrSamples;
};

struct KernelConfigurations {
  // Configuration of single step dedispersion kernel
  Dedispersion::tunedDedispersionConf dedispersionSingleStepParameters;
  // Configuration of subband dedispersion kernel, step one
  Dedispersion::tunedDedispersionConf dedispersionStepOneParameters;
  // Configuration of subband dedispersion kernel, step two
  Dedispersion::tunedDedispersionConf dedispersionStepTwoParameters;
  // Configuration of integration kernel
  Integration::tunedIntegrationConf integrationParameters;
  // Configuration of SNR kernel
  SNR::tunedSNRConf snrParameters;
};

struct Kernels {
  // Single step dedispersion kernel
  cl::Kernel * dedispersionSingleStep = 0;
  // Step one subbanding dedispersion kernel
  cl::Kernel * dedispersionStepOne = 0;
  // Step two subbanding dedispersion kernel
  cl::Kernel * dedispersionStepTwo = 0;
  // Integration kernels, one for each integration step
  std::vector<cl::Kernel *> integration;
  // SNR kernels, one for the original data and one for each integration step
  std::vector<cl::Kernel *> snr;
};

struct KernelRunTimeConfigurations {
  // Global NDrange for single step dedispersion
  cl::NDRange dedispersionSingleStepGlobal;
  // Local NDRange for single step dedispersion
  cl::NDRange dedispersionSingleStepLocal;
  // Global NDRange for subbanding dedispersion step one
  cl::NDRange dedispersionStepOneGlobal;
  // Local NDRange for subbanding dedispersion step one
  cl::NDRange dedispersionStepOneLocal;
  // Global NDRange for subbanding dedispersion step two
  cl::NDRange dedispersionStepTwoGlobal;
  // Local NDRange for subbanding dedispersion step two
  cl::NDRange dedispersionStepTwoLocal;
  // Global NDRange for integration
  std::vector<cl::NDRange> integrationGlobal;
  // Local NDRange for integration
  std::vector<cl::NDRange> integrationLocal;
  // Global NDRange for SNR
  std::vector<cl::NDRange> snrGlobal;
  // Local NDRange for SNR
  std::vector<cl::NDRange> snrLocal;
};

struct Timers {
  isa::utils::Timer inputLoad;
  isa::utils::Timer search;
  isa::utils::Timer inputHandling;
  isa::utils::Timer inputCopy;
  isa::utils::Timer dedispersionSingleStep;
  isa::utils::Timer dedispersionStepOne;
  isa::utils::Timer dedispersionStepTwo;
  isa::utils::Timer integration;
  isa::utils::Timer snr;
  isa::utils::Timer outputCopy;
  isa::utils::Timer trigger;
};

struct TriggeredEvent {
  unsigned int beam = 0;
  unsigned int sample = 0;
  unsigned int integration = 0;
  float DM = 0.0f;
  float SNR = 0.0f;
};

struct CompactedEvent : TriggeredEvent {
  unsigned int compactedIntegration = 1;
  unsigned int compactedDMs = 1;
};

using TriggeredEvents = std::vector<std::map<unsigned int, std::vector<TriggeredEvent>>>;
using CompactedEvents = std::vector<std::vector<CompactedEvent>>;

struct OpenCLRunTime {
  cl::Context * context = 0;
  std::vector<cl::Platform> * platforms = 0;
  std::vector<cl::Device> * devices = 0;
  std::vector<std::vector<cl::CommandQueue>> * queues = 0;
};

