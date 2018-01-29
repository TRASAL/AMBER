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

#include "CommandLine.hpp"

#pragma once

struct Kernels {
  // Single step dedispersion kernel
  cl::Kernel * dedispersion;
  // Step one subbanding dedispersion kernel
  cl::Kernel * dedispersionStepOne;
  // Step two subbanding dedispersion kernel
  cl::Kernel * dedispersionStepTwo;
  // Integration kernels, one for each integration step
  std::vector<cl::Kernel *> integration;
  // SNR kernels, one for the original data and one for each integration step
  std::vector<cl::Kernel *> snr;
};

// Function to generate all necessary OpenCL kernels
void generateOpenCLKernels(const AstroData::Observation & observation, const Options & options, const DeviceOptions & deviceOptions, const Configurations & configurations, Kernels & kernels);
