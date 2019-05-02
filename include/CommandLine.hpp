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
#include "DataTypes.hpp"

#pragma once

/**
 ** @brief Process the command line options.
 **
 ** @param argumentList List of command line arguments.
 ** @param options AMBER options.
 ** @param deviceOptions Device specific options.
 ** @param dataOptions Data specific options.
 ** @param hostMemoryDumpFiles Files for dumping intermediate data products for debugging.
 ** @param kernelConfigurations The configuration of the different kernels.
 ** @param generatorOptions Data generation options.
 ** @param observation Object containing the observation.
 */
void processCommandLineOptions(isa::utils::ArgumentList & argumentList, Options & options, DeviceOptions & deviceOptions, DataOptions & dataOptions, HostMemoryDumpFiles & hostMemoryDumpFiles, KernelConfigurations & kernelConfigurations, GeneratorOptions & generatorOptions, AstroData::Observation & observation);
/**
 ** @brief Print the usage message.
 **
 ** @param program The executable name.
 */
void usage(const std::string & program);
