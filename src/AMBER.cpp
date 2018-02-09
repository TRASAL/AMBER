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

#include <configuration.hpp>
#include <CommandLine.hpp>
#include <Kernels.hpp>
#include <Memory.hpp>
#include <Pipeline.hpp>

int main(int argc, char * argv[]) {
  // Command line options
  Options options {};
  DeviceOptions deviceOptions {};
  DataOptions dataOptions {};
  GeneratorOptions generatorOptions {};
  // Memory
  HostMemory hostMemory {};
  DeviceMemory deviceMemory {};
  // OpenCL kernels
  OpenCLRunTime openclRunTime {};
  KernelConfigurations kernelConfigurations {};
  Kernels kernels {};
  KernelRunTimeConfigurations kernelRunTimeConfigurations {};
  // Timers
  Timers timers {};
  // Observation
  AstroData::Observation observation;

  // Process command line arguments
  isa::utils::ArgumentList args(argc, argv);
  try {
    processCommandLineOptions(args, options, deviceOptions, dataOptions, kernelConfigurations, generatorOptions,
                              observation);
  } catch ( std::exception & err ) {
    return 1;
  }

  // Load or generate input data
  try {
    hostMemory.zappedChannels.resize(observation.getNrChannels(deviceOptions.padding.at(deviceOptions.deviceName)
                                                                   / sizeof(unsigned int)));
    try {
      AstroData::readZappedChannels(observation, dataOptions.channelsFile, hostMemory.zappedChannels);
      AstroData::readIntegrationSteps(observation, dataOptions.integrationFile, hostMemory.integrationSteps);
    } catch ( AstroData::FileError & err ) {
      std::cerr << err.what() << std::endl;
      throw;
    }
    hostMemory.input.resize(observation.getNrBeams());
    if ( dataOptions.dataLOFAR || dataOptions.dataSIGPROC || dataOptions.dataPSRDADA ) {
      loadInput(observation, deviceOptions, dataOptions, hostMemory, timers);
    } else {
      for ( unsigned int beam = 0; beam < observation.getNrBeams(); beam++ ) {
        // TODO: if there are multiple synthesized beams, the generated data should take this into account
        hostMemory.input.at(beam) = new std::vector<std::vector<inputDataType> *>(observation.getNrBatches());
        AstroData::generateSinglePulse(generatorOptions.width, generatorOptions.DM, observation,
                                       deviceOptions.padding.at(deviceOptions.deviceName),
                                       *(hostMemory.input.at(beam)), inputBits, generatorOptions.random);
      }
    }
  } catch ( std::exception & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  // Print message with observation and search information
  if ( options.print) {
    std::cout << "Device: " << deviceOptions.deviceName << "(" + std::to_string(deviceOptions.platformID) +  ", ";
    std::cout << std::to_string(deviceOptions.deviceID) + ")" << std::endl;
    std::cout << "Padding: " << deviceOptions.padding[deviceOptions.deviceName] << " bytes" << std::endl;
    std::cout << std::endl;
    std::cout << "Beams: " << observation.getNrBeams() << std::endl;
    std::cout << "Synthesized Beams: " << observation.getNrSynthesizedBeams() << std::endl;
    std::cout << "Batches: " << observation.getNrBatches() << std::endl;
    std::cout << "Samples: " << observation.getNrSamplesPerBatch() << std::endl;
    std::cout << "Sampling time: " << observation.getSamplingTime() << std::endl;
    std::cout << "Frequency range: " << observation.getMinFreq() << " MHz, " << observation.getMaxFreq() << " MHz";
    std::cout << std::endl;
    std::cout << "Subbands: " << observation.getNrSubbands() << " (" << observation.getSubbandBandwidth() << " MHz)";
    std::cout << std::endl;
    std::cout << "Channels: " << observation.getNrChannels() << " (" << observation.getChannelBandwidth() << " MHz)";
    std::cout << std::endl;
    std::cout << "Zapped Channels: " << observation.getNrZappedChannels() << std::endl;
    std::cout << "Integration steps: " << hostMemory.integrationSteps.size() << std::endl;
    if ( options.subbandDedispersion ) {
      std::cout << "Subbanding DMs: " << observation.getNrDMs(true) << " (" << observation.getFirstDM(true) << ", ";
      std::cout << observation.getFirstDM(true) + ((observation.getNrDMs(true) - 1) * observation.getDMStep(true));
      std::cout << ")" << std::endl;
    }
    std::cout << "DMs: " << observation.getNrDMs() << " (" << observation.getFirstDM() << ", ";
    std::cout << observation.getFirstDM() + ((observation.getNrDMs() - 1) * observation.getDMStep()) << ")";
    std::cout << std::endl;
    std::cout << std::endl;
  }

  // Initialize OpenCL
  openclRunTime.context = new cl::Context();
  openclRunTime.platforms = new std::vector<cl::Platform>();
  openclRunTime.devices = new std::vector<cl::Device>();
  openclRunTime.queues = new std::vector<std::vector<cl::CommandQueue>>();
  try {
    isa::OpenCL::initializeOpenCL(deviceOptions.platformID, 1, openclRunTime.platforms, openclRunTime.context,
                                  openclRunTime.devices, openclRunTime.queues);
  } catch ( isa::OpenCL::OpenCLError & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  // Memory allocation
  allocateHostMemory(observation, options, deviceOptions, kernelConfigurations, hostMemory);
  if ( observation.getNrDelayBatches() > observation.getNrBatches() ) {
    std::cerr << "Not enough input batches for the specified search." << std::endl;
    return 1;
  }
  try {
    allocateDeviceMemory(openclRunTime, options, deviceOptions, hostMemory, deviceMemory);
  } catch ( cl::Error & err ) {
    std::cerr << "Memory error: " << err.what() << " " << err.err() << std::endl;
    return 1;
  }

  // Generate OpenCL kernels
  try {
    generateOpenCLKernels(openclRunTime, observation, options, deviceOptions, kernelConfigurations, hostMemory,
                          deviceMemory, kernels);
  } catch ( std::exception & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  // Generate run time configurations for the OpenCL kernels
  generateOpenCLRunTimeConfigurations(observation, options, deviceOptions, kernelConfigurations, hostMemory,
                                      kernelRunTimeConfigurations);

  // Search loop
  pipeline(openclRunTime, observation, options, deviceOptions, dataOptions, timers, kernels, kernelConfigurations,
           kernelRunTimeConfigurations, hostMemory, deviceMemory);

  // Store performance statistics before shutting down
  std::ofstream outputStats;
  outputStats.open(dataOptions.outputFile + ".stats");
  outputStats << std::fixed << std::setprecision(6);
  outputStats << "# nrDMs" << std::endl;
  if ( options.subbandDedispersion ) {
    outputStats << observation.getNrDMs(true) * observation.getNrDMs() << std::endl;
  } else {
    outputStats << observation.getNrDMs() << std::endl;
  }
  outputStats << "# timers.inputLoad" << std::endl;
  outputStats << timers.inputLoad.getTotalTime() << std::endl;
  outputStats << "# timers.search" << std::endl;
  outputStats << timers.search.getTotalTime() << std::endl;
  outputStats << "# inputHandlingTotal inputHandlingAvg err" << std::endl;
  outputStats << timers.inputHandling.getTotalTime() << " " << timers.inputHandling.getAverageTime() << " ";
  outputStats << timers.inputHandling.getStandardDeviation() << std::endl;
  outputStats << "# inputCopyTotal inputCopyAvg err" << std::endl;
  outputStats << timers.inputCopy.getTotalTime() << " " << timers.inputCopy.getAverageTime() << " ";
  outputStats << timers.inputCopy.getStandardDeviation() << std::endl;
  if ( ! options.subbandDedispersion ) {
    outputStats << "# dedispersionSingleStepTotal dedispersionSingleStepAvg err" << std::endl;
    outputStats << timers.dedispersionSingleStep.getTotalTime() << " ";
    outputStats << timers.dedispersionSingleStep.getAverageTime() << " ";
    outputStats << timers.dedispersionSingleStep.getStandardDeviation() << std::endl;
  } else {
    outputStats << "# dedispersionStepOneTotal dedispersionStepOneAvg err" << std::endl;
    outputStats << timers.dedispersionStepOne.getTotalTime() << " ";
    outputStats << timers.dedispersionStepOne.getAverageTime() << " ";
    outputStats << timers.dedispersionStepOne.getStandardDeviation() << std::endl;
    outputStats << "# dedispersionStepTwoTotal dedispersionStepTwoAvg err" << std::endl;
    outputStats << timers.dedispersionStepTwo.getTotalTime() << " ";
    outputStats << timers.dedispersionStepTwo.getAverageTime() << " ";
    outputStats << timers.dedispersionStepTwo.getStandardDeviation() << std::endl;
  }
  outputStats << "# integrationTotal integrationAvg err" << std::endl;
  outputStats << timers.integration.getTotalTime() << " " << timers.integration.getAverageTime() << " ";
  outputStats << timers.integration.getStandardDeviation() << std::endl;
  outputStats << "# snrTotal snrAvg err" << std::endl;
  outputStats << timers.snr.getTotalTime() << " " << timers.snr.getAverageTime() << " ";
  outputStats << timers.snr.getStandardDeviation() << std::endl;
  outputStats << "# outputCopyTotal outputCopyAvg err" << std::endl;
  outputStats << timers.outputCopy.getTotalTime() << " " << timers.outputCopy.getAverageTime() << " ";
  outputStats << timers.outputCopy.getStandardDeviation() << std::endl;
  outputStats << "# triggerTimeTotal triggerTimeAvg err" << std::endl;
  outputStats << timers.trigger.getTotalTime() << " " << timers.trigger.getAverageTime() << " ";
  outputStats << timers.trigger.getStandardDeviation() << std::endl;
  outputStats.close();

  return 0;
}

