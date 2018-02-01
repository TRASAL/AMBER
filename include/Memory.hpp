// Copyright 2018 Netherlands Institute for Radio Astronomy (ASTRON)
// Copyright 2018 Netherlands eScience Center
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

#include "CommandLine.hpp"

#pragma once

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
  dada_hdu_t * ringBuffer;
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

// Load input files
void loadInput(AstroData::Observation & observation, const DeviceOptions & deviceOptions, const DataOptions & dataOptions, HostMemory & hostMemory, Timers & timers);
// Allocate host memory
void allocateHostMemory(AstroData::Observation & observation, const Options & options, const DeviceOptions & deviceOptions, HostMemory & hostMemory);
// Allocate device memory
void allocateDeviceMemory(const cl::Context * clContext, const std::vector<std::vector<cl::CommandQueue>> * clQueues, const Options & options, const HostMemory & hostMemory, DeviceMemory & deviceMemory);
