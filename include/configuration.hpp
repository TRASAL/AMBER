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

#include <ArgumentList.hpp>
#include <utils.hpp>
#include <Timer.hpp>
#include <Observation.hpp>
#include <Platform.hpp>
#include <ReadData.hpp>
#include <SynthesizedBeams.hpp>
#include <Generator.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>

#include <Shifts.hpp>
#include <Dedispersion.hpp>
#include <Integration.hpp>
#include <SNR.hpp>

#pragma once

// Types for the data
typedef std::uint8_t inputDataType;
const std::uint8_t inputBits = 8;
const std::string inputDataName("uchar");
typedef float intermediateDataType;
const std::string intermediateDataName("float");
typedef float outputDataType;
const std::string outputDataName("float");

// DEBUG mode, prints to screen some useful information
const bool DEBUG = true;

// SYNC mode, OpenCL queue operations
const bool SYNC = true;

