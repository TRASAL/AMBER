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

// Function to process the command line options
void processCommandLineOptions(const isa::utils::ArgumentList & argumentList, const Options & options, const DeviceOptions & deviceOptions, const DataOptions & dataOptions, const KernelConfigurations & kernelConfigurations, const GeneratorOptions & generatorOptions, const AstroData::Observation & observation);
// Function to print the usage message
void usage(const std::string & program);
