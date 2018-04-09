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

#pragma once

/**
 * @brief Execute the pipeline.
 *
 * @param openclRunTime
 * @param observation
 * @param options
 * @param deviceOptions
 * @param dataOptions
 * @param timers
 * @param kernels
 * @param kernelConfigurations
 * @param kernelRunTimeConfigurations
 * @param hostMemory
 * @param deviceMemory
 */
void pipeline(OpenCLRunTime & openclRunTime, AstroData::Observation & observation,
              Options & options, DeviceOptions & deviceOptions, DataOptions & dataOptions,
              Timers & timers, Kernels & kernels, KernelConfigurations & kernelConfigurations,
              KernelRunTimeConfigurations & kernelRunTimeConfigurations, HostMemory & hostMemory,
              DeviceMemory & deviceMemory);

/**
 * @brief Prepare input data for transfer to OpenCL devices.
 *
 * @param batch
 * @param observation
 * @param options
 * @param deviceOptions
 * @param dataOptions
 * @param timers
 * @param hostMemory
 * @param deviceMemory
 *
 * @return status code
 */
int inputHandling(unsigned int batch, AstroData::Observation & observation, Options & options,
                   DeviceOptions & deviceOptions, DataOptions & dataOptions, Timers & timers,
                   HostMemory & hostMemory, DeviceMemory & deviceMemory);

/**
 *
 * @param batch
 * @param openclRunTime
 * @param observation
 * @param options
 * @param deviceOptions
 * @param timers
 * @param hostMemory
 * @param deviceMemory
 * @return
 */
int copyInputToDevice(unsigned int batch, OpenCLRunTime & openclRunTime, AstroData::Observation & observation,
                      Options & options, DeviceOptions & deviceOptions, Timers & timers, HostMemory & hostMemory,
                      DeviceMemory & deviceMemory);
