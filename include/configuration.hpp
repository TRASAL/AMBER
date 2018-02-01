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

#include <iostream>
#include <string>
#include <exception>
#include <fstream>
#include <vector>
#include <set>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iterator>

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

#include "DataTypes.hpp"

#pragma once

// Input data type
typedef std::uint8_t inputDataType;
// Bits of the input data type
const std::uint8_t inputBits = 8;
// OpenCL data type for input data
const std::string inputDataName("uchar");
// Intermediate data type used in dedispersion
typedef float intermediateDataType;
// OpenCL data type for intermediate data
const std::string intermediateDataName("float");
// Output data type
typedef float outputDataType;
// OpenCL data type for output data
const std::string outputDataName("float");

// SYNC mode, OpenCL queue operations
const bool SYNC = true;

