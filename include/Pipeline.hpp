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
#include "DataTypes.hpp"

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
 * @param kernelRunTimeConfigurations
 * @param hostMemory
 * @param deviceMemory
 */
void pipeline(const isa::OpenCL::OpenCLRunTime &openclRunTime, const AstroData::Observation &observation, const Options &options, const DeviceOptions &deviceOptions, const DataOptions &dataOptions, Timers &timers, const Kernels &kernels, const KernelRunTimeConfigurations &kernelRunTimeConfigurations, HostMemory &hostMemory, const DeviceMemory &deviceMemory, HostMemoryDumpFiles &hostMemoryDumpFiles);

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
int inputHandling(const unsigned int batch, const AstroData::Observation &observation, const Options &options, const DeviceOptions &deviceOptions, const DataOptions &dataOptions, Timers &timers, HostMemory &hostMemory, const DeviceMemory &deviceMemory);

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
int copyInputToDevice(const unsigned int batch, const isa::OpenCL::OpenCLRunTime &openclRunTime, const AstroData::Observation &observation, const Options &options, const DeviceOptions &deviceOptions, Timers &timers, HostMemory &hostMemory, const DeviceMemory &deviceMemory);

/**
 ** @brief Radio Frequency Interference mitigation.
 * 
 * @param batch
 * @param syncPoint
 * @param openclRunTime
 * @param options
 * @param deviceOptions
 * @param timers
 * @param kernels
 * @param kernelRunTimeConfigurations
 * @param hostMemory
 */
int rfim(const unsigned int batch, cl::Event &syncPoint, const isa::OpenCL::OpenCLRunTime &openclRunTime, const Options &options, const DeviceOptions &deviceOptions, Timers &timers, const Kernels &kernels, const KernelRunTimeConfigurations &kernelRunTimeConfigurations, HostMemory &hostMemory);

/**
 * @brief Downsampling input data in the time dimension.
 * 
 * @param batch
 * @param syncPoint
 * @param openclRunTime
 * @param observation
 * @param options
 * @param deviceOptions
 * @param timers
 * @param kernels
 * @param kernelRunTimeConfigurations
 * @param hostMemory
 * @param deviceMemory
 * @param hostMemoryDumpFiles
 */
int downsampling(const unsigned int batch, cl::Event &syncPoint, const isa::OpenCL::OpenCLRunTime &openclRunTime, const AstroData::Observation &observation, const Options &options, const DeviceOptions &deviceOptions, Timers &timers, const Kernels &kernels, const KernelRunTimeConfigurations &kernelRunTimeConfigurations, HostMemory &hostMemory, const DeviceMemory &deviceMemory, HostMemoryDumpFiles &hostMemoryDumpFiles);

/**
 * @brief Dedispersion step.
 * 
 * @param batch
 * @param firstSynthesizedBeam
 * @param syncPoint
 * @param openclRunTime
 * @param observation
 * @param options
 * @param deviceOptions
 * @param timers
 * @param kernels
 * @param kernelRunTimeConfigurations
 * @param hostMemory
 * @param deviceMemory
 * @param hostMemoryDumpFiles
 */
int dedispersion(const unsigned int batch, const unsigned int firstSynthesizedBeam, cl::Event &syncPoint, const isa::OpenCL::OpenCLRunTime &openclRunTime, const AstroData::Observation &observation, const Options &options, const DeviceOptions &deviceOptions, Timers &timers, const Kernels &kernels, const KernelRunTimeConfigurations &kernelRunTimeConfigurations, HostMemory &hostMemory, const DeviceMemory &deviceMemory, HostMemoryDumpFiles &hostMemoryDumpFiles);

/**
 * @brief Compute the SNR of dedispersed time series.
 * 
 * @param batch
 * @param firstSynthesizedBeam
 * @param syncPoint
 * @param openclRunTime
 * @param observation
 * @param options
 * @param deviceOptions
 * @param timers
 * @param kernels
 * @param kernelRunTimeConfigurations
 * @param hostMemory
 * @param deviceMemory
 * @param hostMemoryDumpFiles
 * @param triggeredEvents
 */
int dedispersionSNR(const unsigned int batch, const unsigned int firstSynthesizedBeam, cl::Event &syncPoint, const isa::OpenCL::OpenCLRunTime &openclRunTime, const AstroData::Observation &observation, const Options &options, const DeviceOptions &deviceOptions, Timers &timers, const Kernels &kernels, const KernelRunTimeConfigurations &kernelRunTimeConfigurations, HostMemory &hostMemory, const DeviceMemory &deviceMemory, HostMemoryDumpFiles &hostMemoryDumpFiles, TriggeredEvents &triggeredEvents);

/**
 * @brief Search dedispersed time series for pulses of different widths.
 * 
 * @param batch
 * @param firstSynthesizedBeam
 * @param stepNumber
 * @param step
 * @param syncPoint
 * @param openclRunTime
 * @param observation
 * @param options
 * @param deviceOptions
 * @param timers
 * @param kernels
 * @param kernelRunTimeConfigurations
 * @param hostMemory
 * @param deviceMemory
 * @param hostMemoryDumpFiles
 * @param triggeredEvents
 */
int pulseWidthSearch(const unsigned int batch, const unsigned int firstSynthesizedBeam, const unsigned int stepNumber, const unsigned int step, cl::Event &syncPoint, const isa::OpenCL::OpenCLRunTime &openclRunTime, const AstroData::Observation &observation, const Options &options, const DeviceOptions &deviceOptions, Timers &timers, const Kernels &kernels, const KernelRunTimeConfigurations &kernelRunTimeConfigurations, HostMemory &hostMemory, const DeviceMemory &deviceMemory, HostMemoryDumpFiles &hostMemoryDumpFiles, TriggeredEvents &triggeredEvents);

/**
 * @brief Write the results to a trigger file.
 * 
 * @param batch
 * @param firstSynthesizedBeam
 * @param observation
 * @param options
 * @param dataOptions
 * @param timers
 * @param triggeredEvents
 * @param compactedEvents
 * @param outputTrigger
 */
int printResults(const unsigned int batch, const unsigned int firstSynthesizedBeam, const AstroData::Observation &observation, const Options &options, const DataOptions &dataOptions, Timers &timers, TriggeredEvents &triggeredEvents, CompactedEvents &compactedEvents, std::ofstream &outputTrigger);

/**
 * @brief Close output files and buffers.
 * 
 * @param options
 * @param dataOptions
 * @param hostMemory
 * @param hostMemoryDumpFiles
 * @param outputTrigger
 */
void clean(const Options &options, const DataOptions &dataOptions, HostMemory &hostMemory, HostMemoryDumpFiles &hostMemoryDumpFiles, std::ofstream &outputTrigger);