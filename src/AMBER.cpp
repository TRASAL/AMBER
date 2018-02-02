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
#include <Trigger.hpp>

int main(int argc, char * argv[]) {
  // Command line options
  Options options;
  DeviceOptions deviceOptions;
  DataOptions dataOptions;
  GeneratorOptions generatorOptions;
  // Memory
  HostMemory hostMemory;
  DeviceMemory deviceMemory;
  // OpenCL kernels
  OpenCLRunTime openclRunTime;
  KernelConfigurations kernelConfigurations;
  Kernels kernels;
  KernelRunTimeConfigurations kernelRunTimeConfigurations;
  // Timers
  Timers timers;
  // Observation
  AstroData::Observation observation;

  // Process command line arguments
  isa::utils::ArgumentList args(argc, argv);
  try {
    processCommandLineOptions(args, options, deviceOptions, dataOptions, kernelConfigurations, generatorOptions, observation);
  } catch ( std::exception & err ) {
    return 1;
  }

  // Load or generate input data
  try {
    if ( dataOptions.dataLOFAR || dataOptions.dataSIGPROC || dataOptions.dataPSRDADA ) {
      loadInput(observation, deviceOptions, dataOptions, hostMemory, timers);
    } else {
      for ( unsigned int beam = 0; beam < observation.getNrBeams(); beam++ ) {
        // TODO: if there are multiple synthesized beams, the generated data should take this into account
        hostMemory.input.at(beam) = new std::vector<std::vector<inputDataType> *>(observation.getNrBatches());
        AstroData::generateSinglePulse(generatorOptions.width, generatorOptions.DM, observation, deviceOptions.padding.at(deviceOptions.deviceName), *(hostMemory.input.at(beam)), inputBits, generatorOptions.random);
      }
    }
  } catch ( std::exception & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  // Print message with observation and search information
  if ( options.print) {
    std::cout << "Device: " << deviceOptions.deviceName << "(" + std::to_string(deviceOptions.platformID) +  ", " + std::to_string(deviceOptions.deviceID) + ")" << std::endl;
    std::cout << "Padding: " << deviceOptions.padding[deviceOptions.deviceName] << " bytes" << std::endl;
    std::cout << std::endl;
    std::cout << "Beams: " << observation.getNrBeams() << std::endl;
    std::cout << "Synthesized Beams: " << observation.getNrSynthesizedBeams() << std::endl;
    std::cout << "Batches: " << observation.getNrBatches() << std::endl;
    std::cout << "Samples: " << observation.getNrSamplesPerBatch() << std::endl;
    std::cout << "Sampling time: " << observation.getSamplingTime() << std::endl;
    std::cout << "Frequency range: " << observation.getMinFreq() << " MHz, " << observation.getMaxFreq() << " MHz" << std::endl;
    std::cout << "Subbands: " << observation.getNrSubbands() << " (" << observation.getSubbandBandwidth() << " MHz)" << std::endl;
    std::cout << "Channels: " << observation.getNrChannels() << " (" << observation.getChannelBandwidth() << " MHz)" << std::endl;
    std::cout << "Zapped Channels: " << observation.getNrZappedChannels() << std::endl;
    std::cout << "Integration steps: " << hostMemory.integrationSteps.size() << std::endl;
    if ( options.subbandDedispersion ) {
      std::cout << "Subbanding DMs: " << observation.getNrDMs(true) << " (" << observation.getFirstDM(true) << ", " << observation.getFirstDM(true) + ((observation.getNrDMs(true) - 1) * observation.getDMStep(true)) << ")" << std::endl;
    }
    std::cout << "DMs: " << observation.getNrDMs() << " (" << observation.getFirstDM() << ", " << observation.getFirstDM() + ((observation.getNrDMs() - 1) * observation.getDMStep()) << ")" << std::endl;
    std::cout << std::endl;
  }

  // Initialize OpenCL
  openclRunTime.context = new cl::Context();
  openclRunTime.platforms = new std::vector<cl::Platform>();
  openclRunTime.devices = new std::vector<cl::Device>();
  openclRunTime.queues = new std::vector<std::vector<cl::CommandQueue>>();
  try {
    isa::OpenCL::initializeOpenCL(deviceOptions.platformID, 1, openclRunTime.platforms, openclRunTime.context, openclRunTime.devices, openclRunTime.queues);
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
    generateOpenCLKernels(openclRunTime, observation, options, deviceOptions, kernelConfigurations, hostMemory, deviceMemory, kernels);
  } catch ( std::exception & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }
  if ( ! options.subbandDedispersion ) {
    if ( kernelConfigurations.dedispersionSingleStepParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getSplitBatches() ) {
      // TODO: add support for splitBatches
    } else {
      kernels.dedispersionSingleStep->setArg(0, deviceMemory.dispersedData);
      kernels.dedispersionSingleStep->setArg(1, deviceMemory.dedispersedData);
      kernels.dedispersionSingleStep->setArg(2, deviceMemory.beamMapping);
      kernels.dedispersionSingleStep->setArg(3, deviceMemory.zappedChannels);
      kernels.dedispersionSingleStep->setArg(4, deviceMemory.shiftsSingleStep);
    }
  } else {
    if ( kernelConfigurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getSplitBatches() ) {
      // TODO: add support for splitBatches
    } else {
      kernels.dedispersionStepOne->setArg(0, deviceMemory.dispersedData);
      kernels.dedispersionStepOne->setArg(1, deviceMemory.subbandedData);
      kernels.dedispersionStepOne->setArg(2, deviceMemory.zappedChannels);
      kernels.dedispersionStepOne->setArg(3, deviceMemory.shiftsStepOne);
    }
    kernels.dedispersionStepTwo->setArg(0, deviceMemory.subbandedData);
    kernels.dedispersionStepTwo->setArg(1, deviceMemory.dedispersedData);
    kernels.dedispersionStepTwo->setArg(2, deviceMemory.beamMapping);
    kernels.dedispersionStepTwo->setArg(3, deviceMemory.shiftsStepTwo);
  }

  // Generate run time configurations for the OpenCL kernels
  generateOpenCLRunTimeConfigurations(observation, options, deviceOptions, kernelConfigurations, hostMemory, kernelRunTimeConfigurations);

  // Search loop
  std::ofstream output;
  bool errorDetected = false;
  cl::Event syncPoint;

  timers.search.start();
  output.open(dataOptions.outputFile + ".trigger");
  if ( options.compactResults ) {
    output << "# beam batch sample integration_step compacted_integration_steps time DM compacted_DMs SNR" << std::endl;
  } else {
    output << "# beam batch sample integration_step time DM SNR" << std::endl;
  }
  for ( unsigned int batch = 0; batch < observation.getNrBatches(); batch++ ) {
    TriggeredEvents triggeredEvents(observation.getNrSynthesizedBeams());
    CompactedEvents compactedEvents(observation.getNrSynthesizedBeams());

    // Load the input
    timers.inputHandling.start();
    if ( !dataOptions.dataPSRDADA ) {
      // If there are not enough available batches, computation is complete
      if ( options.subbandDedispersion ) {
        if ( batch == observation.getNrBatches() - observation.getNrDelayBatches(true) ) {
          break;
        }
      } else {
        if ( batch == observation.getNrBatches() - observation.getNrDelayBatches() ) {
          break;
        }
      }
      // If there are enough batches, prepare them for transfer to device
      for ( unsigned int beam = 0; beam < observation.getNrBeams(); beam++ ) {
        if ( options.subbandDedispersion ) {
          if ( !kernelConfigurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getSplitBatches() ) {
            for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ ) {
              for ( unsigned int chunk = 0; chunk < observation.getNrDelayBatches(true) - 1; chunk++ ) {
                if ( inputBits >= 8 ) {
                  memcpy(reinterpret_cast< void * >(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (chunk * observation.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&((hostMemory.input.at(beam)->at(batch + chunk))->at(channel * observation.getNrSamplesPerBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))))), observation.getNrSamplesPerBatch() * sizeof(inputDataType));
                } else {
                  memcpy(reinterpret_cast< void * >(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (chunk * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&((hostMemory.input.at(beam)->at(batch + chunk))->at(channel * isa::utils::pad(observation.getNrSamplesPerBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))))), (observation.getNrSamplesPerBatch() / (8 / inputBits)) * sizeof(inputDataType));
                }
              }
              if ( inputBits >= 8 ) {
                memcpy(reinterpret_cast< void * >(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + ((observation.getNrDelayBatches(true) - 1) * observation.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&((hostMemory.input.at(beam)->at(batch + (observation.getNrDelayBatches(true) - 1)))->at(channel * observation.getNrSamplesPerBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))))), (observation.getNrSamplesPerDispersedBatch(true) % observation.getNrSamplesPerBatch()) * sizeof(inputDataType));
              } else {
                memcpy(reinterpret_cast< void * >(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + ((observation.getNrDelayBatches(true) - 1) * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&((hostMemory.input.at(beam)->at(batch + (observation.getNrDelayBatches(true) - 1)))->at(channel * isa::utils::pad(observation.getNrSamplesPerBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))))), ((observation.getNrSamplesPerDispersedBatch(true) % observation.getNrSamplesPerBatch()) / (8 / inputBits)) * sizeof(inputDataType));
              }
            }
          }
        } else {
          if ( !kernelConfigurations.dedispersionSingleStepParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getSplitBatches() ) {
            for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ ) {
              for ( unsigned int chunk = 0; chunk < observation.getNrDelayBatches() - 1; chunk++ ) {
                if ( inputBits >= 8 ) {
                  memcpy(reinterpret_cast< void * >(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (chunk * observation.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&((hostMemory.input.at(beam)->at(batch + chunk))->at(channel * observation.getNrSamplesPerBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))))), observation.getNrSamplesPerBatch() * sizeof(inputDataType));
                } else {
                  memcpy(reinterpret_cast< void * >(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (chunk * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&((hostMemory.input.at(beam)->at(batch + chunk))->at(channel * isa::utils::pad(observation.getNrSamplesPerBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))))), (observation.getNrSamplesPerBatch() / (8 / inputBits)) * sizeof(inputDataType));
                }
              }
              if ( inputBits >= 8 ) {
                memcpy(reinterpret_cast< void * >(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + ((observation.getNrDelayBatches() - 1) * observation.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&((hostMemory.input.at(beam)->at(batch + (observation.getNrDelayBatches() - 1)))->at(channel * observation.getNrSamplesPerBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))))), (observation.getNrSamplesPerDispersedBatch() % observation.getNrSamplesPerBatch()) * sizeof(inputDataType));
              } else {
                memcpy(reinterpret_cast< void * >(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + ((observation.getNrDelayBatches() - 1) * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&((hostMemory.input.at(beam)->at(batch + (observation.getNrDelayBatches() - 1)))->at(channel * isa::utils::pad(observation.getNrSamplesPerBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))))), ((observation.getNrSamplesPerDispersedBatch() % observation.getNrSamplesPerBatch()) / (8 / inputBits)) * sizeof(inputDataType));
              }
            }
          }
        }
      }
    } else {
#ifdef HAVE_PSRDADA
      try {
        if ( ipcbuf_eod(reinterpret_cast< ipcbuf_t * >(hostMemory.ringBuffer->data_block)) ) {
          errorDetected = true;
          break;
        }
        if ( options.subbandDedispersion ) {
          AstroData::readPSRDADA(*hostMemory.ringBuffer, hostMemory.inputDADA.at(batch % observation.getNrDelayBatches(true)));
        } else {
          AstroData::readPSRDADA(*hostMemory.ringBuffer, hostMemory.inputDADA.at(batch % observation.getNrDelayBatches()));
        }
      } catch ( AstroData::RingBufferError & err ) {
        std::cerr << "Error: " << err.what() << std::endl;
        return 1;
      }
      // If there are enough data buffered, proceed with the computation
      // Otherwise, move to the next iteration of the search loop
      if ( options.subbandDedispersion ) {
        if ( batch < observation.getNrDelayBatches(true) - 1 ) {
          continue;
        }
      } else {
        if ( batch < observation.getNrDelayBatches() - 1 ) {
          continue;
        }
      }
      for ( unsigned int beam = 0; beam < observation.getNrBeams(); beam++ ) {
        if ( options.subbandDedispersion ) {
          if ( !kernelConfigurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getSplitBatches() ) {
            for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ ) {
              for ( unsigned int chunk = batch - (observation.getNrDelayBatches(true) - 1); chunk < batch; chunk++ ) {
                // Full batches
                if ( inputBits >= 8 ) {
                  memcpy(reinterpret_cast< void * >(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + ((chunk - (batch - (observation.getNrDelayBatches(true) - 1))) * observation.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&(hostMemory.inputDADA.at(chunk % observation.getNrDelayBatches(true))->at((beam * observation.getNrChannels() * observation.getNrSamplesPerBatch()) + (channel * observation.getNrSamplesPerBatch())))), observation.getNrSamplesPerBatch() * sizeof(inputDataType));
                } else {
                  memcpy(reinterpret_cast< void * >(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + ((chunk - (batch - (observation.getNrDelayBatches(true) - 1))) * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&(hostMemory.inputDADA.at(chunk % observation.getNrDelayBatches(true))->at((beam * observation.getNrChannels() * (observation.getNrSamplesPerBatch() / (8 / inputBits))) + (channel * (observation.getNrSamplesPerBatch() / (8 / inputBits)))))), (observation.getNrSamplesPerBatch() / (8 / inputBits)) * sizeof(inputDataType));
                }
              }
              // Remainder (part of current batch)
              if ( inputBits >= 8 ) {
                memcpy(reinterpret_cast< void * >(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + ((observation.getNrDelayBatches(true) - 1) * observation.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&(hostMemory.inputDADA.at(batch % observation.getNrDelayBatches(true))->at((beam * observation.getNrChannels() * observation.getNrSamplesPerBatch()) + (channel * observation.getNrSamplesPerBatch())))), (observation.getNrSamplesPerDispersedBatch(true) % observation.getNrSamplesPerBatch()) * sizeof(inputDataType));
              } else {
                memcpy(reinterpret_cast< void * >(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + ((observation.getNrDelayBatches(true) - 1) * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&(hostMemory.inputDADA.at(batch % observation.getNrDelayBatches(true))->at((beam * observation.getNrChannels() * (observation.getNrSamplesPerBatch() / (8 / inputBits))) + (channel * (observation.getNrSamplesPerBatch() / (8 / inputBits)))))), ((observation.getNrSamplesPerDispersedBatch(true) % observation.getNrSamplesPerBatch()) / (8 / inputBits)) * sizeof(inputDataType));
              }
            }
          }
        } else {
          if ( !kernelConfigurations.dedispersionSingleStepParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getSplitBatches() ) {
            for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ ) {
              for ( unsigned int chunk = batch - (observation.getNrDelayBatches() - 1); chunk < batch; chunk++ ) {
                if ( inputBits >= 8 ) {
                  memcpy(reinterpret_cast< void * >(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + ((chunk - (batch - (observation.getNrDelayBatches() - 1))) * observation.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&(hostMemory.inputDADA.at(chunk % observation.getNrDelayBatches())->at((beam * observation.getNrChannels() * observation.getNrSamplesPerBatch()) + (channel * observation.getNrSamplesPerBatch())))), observation.getNrSamplesPerBatch() * sizeof(inputDataType));
                } else {
                  memcpy(reinterpret_cast< void * >(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + ((chunk - (batch - (observation.getNrDelayBatches() - 1))) * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&(hostMemory.inputDADA.at(chunk % observation.getNrDelayBatches())->at((beam * observation.getNrChannels() * (observation.getNrSamplesPerBatch() / (8 / inputBits))) + (channel * (observation.getNrSamplesPerBatch() / (8 / inputBits)))))), (observation.getNrSamplesPerBatch() / (8 / inputBits)) * sizeof(inputDataType));
                }
              }
              if ( inputBits >= 8 ) {
                memcpy(reinterpret_cast< void * >(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + ((observation.getNrDelayBatches() - 1) * observation.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&(hostMemory.inputDADA.at(batch % observation.getNrDelayBatches())->at((beam * observation.getNrChannels() * observation.getNrSamplesPerBatch()) + (channel * observation.getNrSamplesPerBatch())))), (observation.getNrSamplesPerDispersedBatch() % observation.getNrSamplesPerBatch()) * sizeof(inputDataType));
              } else {
                memcpy(reinterpret_cast< void * >(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + ((observation.getNrDelayBatches() - 1) * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&(hostMemory.inputDADA.at(batch % observation.getNrDelayBatches())->at((beam * observation.getNrChannels() * (observation.getNrSamplesPerBatch() / (8 / inputBits))) + (channel * (observation.getNrSamplesPerBatch() / (8 / inputBits)))))), ((observation.getNrSamplesPerDispersedBatch() % observation.getNrSamplesPerBatch()) / (8 / inputBits)) * sizeof(inputDataType));
              }
            }
          }
        }
      }
#endif // HAVE_PSRDADA
    }
    timers.inputHandling.stop();
    // Copy input from host to device
    try {
      if ( SYNC ) {
        timers.inputCopy.start();
        if ( options.subbandDedispersion ) {
          if ( kernelConfigurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getSplitBatches() ) {
            // TODO: add support for splitBatches
          } else {
            openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueWriteBuffer(deviceMemory.dispersedData, CL_TRUE, 0, hostMemory.dispersedData.size() * sizeof(inputDataType), reinterpret_cast< void * >(hostMemory.dispersedData.data()), 0, &syncPoint);
          }
        } else {
          if ( kernelConfigurations.dedispersionSingleStepParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getSplitBatches() ) {
            // TODO: add support for splitBatches
          } else {
            openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueWriteBuffer(deviceMemory.dispersedData, CL_TRUE, 0, hostMemory.dispersedData.size() * sizeof(inputDataType), reinterpret_cast< void * >(hostMemory.dispersedData.data()), 0, &syncPoint);
          }
        }
        syncPoint.wait();
        timers.inputCopy.stop();
      } else {
        if ( options.subbandDedispersion ) {
          if ( kernelConfigurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getSplitBatches() ) {
            // TODO: add support for splitBatches
          } else {
            openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueWriteBuffer(deviceMemory.dispersedData, CL_FALSE, 0, hostMemory.dispersedData.size() * sizeof(inputDataType), reinterpret_cast< void * >(hostMemory.dispersedData.data()));
          }
        } else {
          if ( kernelConfigurations.dedispersionSingleStepParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getSplitBatches() ) {
            // TODO: add support for splitBatches
          } else {
            openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueWriteBuffer(deviceMemory.dispersedData, CL_FALSE, 0, hostMemory.dispersedData.size() * sizeof(inputDataType), reinterpret_cast< void * >(hostMemory.dispersedData.data()));
          }
        }
      }
      if ( options.debug ) {
        // TODO: add support for splitBatches
        std::cerr << "dispersedData" << std::endl;
        if ( options.subbandDedispersion ) {
          if ( inputBits >= 8 ) {
            for ( unsigned int beam = 0; beam < observation.getNrBeams(); beam++ ) {
              std::cerr << "Beam: " << beam << std::endl;
              for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ ) {
                for ( unsigned int sample = 0; sample < observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType)); sample++ ) {
                  std::cerr << static_cast< float >(hostMemory.dispersedData.at((beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + sample)) << " ";
                }
                std::cerr << std::endl;
              }
              std::cerr << std::endl;
            }
          } else {
            // TODO: add support for input data less than 8 bit
          }
        } else {
          if ( inputBits >= 8 ) {
            for ( unsigned int beam = 0; beam < observation.getNrBeams(); beam++ ) {
              std::cerr << "Beam: " << beam << std::endl;
              for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ ) {
                for ( unsigned int sample = 0; sample < observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType)); sample++ ) {
                  std::cerr << static_cast< float >(hostMemory.dispersedData.at((beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + sample)) << " ";
                }
                std::cerr << std::endl;
              }
              std::cerr << std::endl;
            }
          } else {
            // TODO: add support for input data less than 8 bit
          }
        }
      }
    } catch ( cl::Error & err ) {
      std::cerr << "Input copy error -- Batch: " << std::to_string(batch) << ", " << err.what() << " " << err.err() << std::endl;
      errorDetected = true;
    }
    if ( options.subbandDedispersion ) {
      if ( kernelConfigurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getSplitBatches() && (batch < observation.getNrDelayBatches()) ) {
        // Not enough batches in the buffer to start the search
        continue;
      }
    } else {
      if ( kernelConfigurations.dedispersionSingleStepParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getSplitBatches() && (batch < observation.getNrDelayBatches()) ) {
        // Not enough batches in the buffer to start the search
        continue;
      }
    }

    // Dedispersion
    if ( options.subbandDedispersion ) {
      if ( kernelConfigurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getSplitBatches() ) {
        // TODO: add support for splitBatches
      }
      if ( SYNC ) {
        try {
          timers.dedispersionStepOne.start();
          openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueNDRangeKernel(*(kernels.dedispersionStepOne), cl::NullRange, kernelRunTimeConfigurations.dedispersionStepOneGlobal, kernelRunTimeConfigurations.dedispersionStepOneLocal, 0, &syncPoint);
          syncPoint.wait();
          timers.dedispersionStepOne.stop();
        } catch ( cl::Error & err ) {
          std::cerr << "Dedispersion Step One error -- Batch: " << std::to_string(batch) << ", " << err.what() << " " << err.err() << std::endl;
          errorDetected = true;
        }
        try {
          timers.dedispersionStepTwo.start();
          openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueNDRangeKernel(*(kernels.dedispersionStepTwo), cl::NullRange, kernelRunTimeConfigurations.dedispersionStepTwoGlobal, kernelRunTimeConfigurations.dedispersionStepTwoLocal, 0, &syncPoint);
          syncPoint.wait();
          timers.dedispersionStepTwo.stop();
        } catch ( cl::Error & err ) {
          std::cerr << "Dedispersion Step Two error -- Batch: " << std::to_string(batch) << ", " << err.what() << " " << err.err() << std::endl;
          errorDetected = true;
        }
      } else {
        try {
          openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueNDRangeKernel(*(kernels.dedispersionStepOne), cl::NullRange, kernelRunTimeConfigurations.dedispersionStepOneGlobal, kernelRunTimeConfigurations.dedispersionStepOneLocal, 0, 0);
        } catch ( cl::Error & err ) {
          std::cerr << "Dedispersion Step One error -- Batch: " << std::to_string(batch) << ", " << err.what() << " " << err.err() << std::endl;
          errorDetected = true;
        }
        try {
          openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueNDRangeKernel(*(kernels.dedispersionStepTwo), cl::NullRange, kernelRunTimeConfigurations.dedispersionStepTwoGlobal, kernelRunTimeConfigurations.dedispersionStepTwoLocal, 0, 0);
        } catch ( cl::Error & err ) {
          std::cerr << "Dedispersion Step Two error -- Batch: " << std::to_string(batch) << ", " << err.what() << " " << err.err() << std::endl;
          errorDetected = true;
        }
      }
    } else {
      try {
          if ( kernelConfigurations.dedispersionSingleStepParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getSplitBatches() ) {
            // TODO: add support for splitBatches
          }
          if ( SYNC ) {
            timers.dedispersionSingleStep.start();
            openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueNDRangeKernel(*(kernels.dedispersionSingleStep), cl::NullRange, kernelRunTimeConfigurations.dedispersionSingleStepGlobal, kernelRunTimeConfigurations.dedispersionSingleStepLocal, 0, &syncPoint);
            syncPoint.wait();
            timers.dedispersionSingleStep.stop();
          } else {
            openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueNDRangeKernel(*(kernels.dedispersionSingleStep), cl::NullRange, kernelRunTimeConfigurations.dedispersionSingleStepGlobal, kernelRunTimeConfigurations.dedispersionSingleStepLocal);
          }
      } catch ( cl::Error & err ) {
        std::cerr << "Dedispersion error -- Batch: " << std::to_string(batch) << ", " << err.what() << " " << err.err() << std::endl;
        errorDetected = true;
      }
    }
    if ( options.debug ) {
      if ( options.subbandDedispersion ) {
        try {
          openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueReadBuffer(deviceMemory.subbandedData, CL_TRUE, 0, hostMemory.subbandedData.size() * sizeof(outputDataType), reinterpret_cast< void * >(hostMemory.subbandedData.data()), 0, &syncPoint);
          syncPoint.wait();
          openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueReadBuffer(deviceMemory.dedispersedData, CL_TRUE, 0, hostMemory.dedispersedData.size() * sizeof(outputDataType), reinterpret_cast< void * >(hostMemory.dedispersedData.data()), 0, &syncPoint);
          syncPoint.wait();
          std::cerr << "subbandedData" << std::endl;
          for ( unsigned int beam = 0; beam < observation.getNrBeams(); beam++ ) {
            std::cerr << "Beam: " << beam << std::endl;
            for ( unsigned int dm = 0; dm < observation.getNrDMs(true); dm++ ) {
              std::cerr << "Subbanding DM: " << dm << std::endl;
              for ( unsigned int subband = 0; subband < observation.getNrSubbands(); subband++ ) {
                for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(true); sample++ ) {
                  std::cerr << hostMemory.subbandedData.at((beam * observation.getNrDMs(true) * observation.getNrSubbands() * observation.getNrSamplesPerBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + (dm * observation.getNrSubbands() * observation.getNrSamplesPerBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + (subband * observation.getNrSamplesPerBatch(true, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType))) + sample) << " ";
                }
                std::cerr << std::endl;
              }
              std::cerr << std::endl;
            }
            std::cerr << std::endl;
          }
          std::cerr << "dedispersedData" << std::endl;
          for ( unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++ ) {
            std::cerr << "sBeam: " << sBeam << std::endl;
            for ( unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++ ) {
              for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
                std::cerr << "DM: " << (subbandingDM * observation.getNrDMs()) + dm << std::endl;
                for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++ ) {
                  std::cerr << hostMemory.dedispersedData.at((sBeam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + (dm * observation.getNrSamplesPerBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + sample) << " ";
                }
                std::cerr << std::endl;
              }
              std::cerr << std::endl;
            }
            std::cerr << std::endl;
          }
        } catch ( cl::Error & err) {
          std::cerr << "Impossible to read deviceMemory.subbandedData and deviceMemory.dedispersedData: " << err.what() << " " << err.err() << std::endl;
          errorDetected = true;
        }
      } else {
        try {
          openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueReadBuffer(deviceMemory.dedispersedData, CL_TRUE, 0, hostMemory.dedispersedData.size() * sizeof(outputDataType), reinterpret_cast< void * >(hostMemory.dedispersedData.data()), 0, &syncPoint);
          syncPoint.wait();
          std::cerr << "dedispersedData" << std::endl;
          for ( unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++ ) {
            std::cerr << "sBeam: " << sBeam << std::endl;
            for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
              std::cerr << "DM: " << dm << std::endl;
              for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++ ) {
                std::cerr << hostMemory.dedispersedData.at((sBeam * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + (dm * observation.getNrSamplesPerBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + sample) << " ";
              }
              std::cerr << std::endl;
            }
            std::cerr << std::endl;
          }
        } catch ( cl::Error & err ) {
          std::cerr << "Impossible to read deviceMemory.dedispersedData: " << err.what() << " " << err.err() << std::endl;
          errorDetected = true;
        }
      }
    }

    // SNR of dedispersed data
    try {
      if ( SYNC ) {
        timers.snr.start();
        openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueNDRangeKernel(*kernels.snr[hostMemory.integrationSteps.size()], cl::NullRange, kernelRunTimeConfigurations.snrGlobal[hostMemory.integrationSteps.size()], kernelRunTimeConfigurations.snrLocal[hostMemory.integrationSteps.size()], 0, &syncPoint);
        syncPoint.wait();
        timers.snr.stop();
        timers.outputCopy.start();
        openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueReadBuffer(deviceMemory.snrData, CL_TRUE, 0, hostMemory.snrData.size() * sizeof(float), reinterpret_cast< void * >(hostMemory.snrData.data()), 0, &syncPoint);
        syncPoint.wait();
        openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueReadBuffer(deviceMemory.snrSamples, CL_TRUE, 0, hostMemory.snrSamples.size() * sizeof(unsigned int), reinterpret_cast< void * >(hostMemory.snrSamples.data()), 0, &syncPoint);
        syncPoint.wait();
        timers.outputCopy.stop();
      } else {
        openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueNDRangeKernel(*kernels.snr[hostMemory.integrationSteps.size()], cl::NullRange, kernelRunTimeConfigurations.snrGlobal[hostMemory.integrationSteps.size()], kernelRunTimeConfigurations.snrLocal[hostMemory.integrationSteps.size()]);
        openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueReadBuffer(deviceMemory.snrData, CL_FALSE, 0, hostMemory.snrData.size() * sizeof(float), reinterpret_cast< void * >(hostMemory.snrData.data()));
        openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueReadBuffer(deviceMemory.snrSamples, CL_FALSE, 0, hostMemory.snrSamples.size() * sizeof(unsigned int), reinterpret_cast< void * >(hostMemory.snrSamples.data()));
        openclRunTime.queues->at(deviceOptions.deviceID)[0].finish();
      }
    } catch ( cl::Error & err ) {
      std::cerr << "SNR dedispersed data error -- Batch: " << std::to_string(batch) << ", " << err.what() << " " << err.err() << std::endl;
      errorDetected = true;
    }
    timers.trigger.start();
    trigger(options.subbandDedispersion, deviceOptions.padding[deviceOptions.deviceName], 0, options.threshold, observation, hostMemory.snrData, hostMemory.snrSamples, triggeredEvents);
    timers.trigger.stop();
    if ( options.debug ) {
      if ( options.subbandDedispersion ) {
        std::cerr << "hostMemory.snrData" << std::endl;
        for ( unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++ ) {
          std::cerr << "sBeam: " << sBeam << std::endl;
          for ( unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++ ) {
            for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
              std::cerr << hostMemory.snrData.at((sBeam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding[deviceOptions.deviceName] / sizeof(float))) + (subbandingDM * observation.getNrDMs()) + dm) << " ";
            }
          }
          std::cerr << std::endl;
        }
      } else {
        std::cerr << "hostMemory.snrData" << std::endl;
        for ( unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++ ) {
          std::cerr << "sBeam: " << sBeam << std::endl;
          for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
            std::cerr << hostMemory.snrData.at((sBeam * observation.getNrDMs(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(float))) + dm) << " ";
          }
          std::cerr << std::endl;
        }
      }
      std::cerr << std::endl;
    }

    // Integration and SNR loop
    for ( unsigned int stepNumber = 0; stepNumber < hostMemory.integrationSteps.size(); stepNumber++ ) {
      auto step = hostMemory.integrationSteps.begin();

      std::advance(step, stepNumber);
      try {
        if ( SYNC ) {
          timers.integration.start();
          openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueNDRangeKernel(*kernels.integration[stepNumber], cl::NullRange, kernelRunTimeConfigurations.integrationGlobal[stepNumber], kernelRunTimeConfigurations.integrationLocal[stepNumber], 0, &syncPoint);
          syncPoint.wait();
          timers.integration.stop();
          timers.snr.start();
          openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueNDRangeKernel(*kernels.snr[stepNumber], cl::NullRange, kernelRunTimeConfigurations.snrGlobal[stepNumber], kernelRunTimeConfigurations.snrLocal[stepNumber], 0, &syncPoint);
          syncPoint.wait();
          timers.snr.stop();
          timers.outputCopy.start();
          openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueReadBuffer(deviceMemory.snrData, CL_TRUE, 0, hostMemory.snrData.size() * sizeof(float), reinterpret_cast< void * >(hostMemory.snrData.data()), 0, &syncPoint);
          syncPoint.wait();
          openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueReadBuffer(deviceMemory.snrSamples, CL_TRUE, 0, hostMemory.snrSamples.size() * sizeof(unsigned int), reinterpret_cast< void * >(hostMemory.snrSamples.data()), 0, &syncPoint);
          syncPoint.wait();
          timers.outputCopy.stop();
        } else {
          openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueNDRangeKernel(*kernels.integration[stepNumber], cl::NullRange, kernelRunTimeConfigurations.integrationGlobal[stepNumber], kernelRunTimeConfigurations.integrationLocal[stepNumber]);
          openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueNDRangeKernel(*kernels.snr[stepNumber], cl::NullRange, kernelRunTimeConfigurations.snrGlobal[stepNumber], kernelRunTimeConfigurations.snrLocal[stepNumber]);
          openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueReadBuffer(deviceMemory.snrData, CL_FALSE, 0, hostMemory.snrData.size() * sizeof(float), reinterpret_cast< void * >(hostMemory.snrData.data()));
          openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueReadBuffer(deviceMemory.snrSamples, CL_FALSE, 0, hostMemory.snrSamples.size() * sizeof(unsigned int), reinterpret_cast< void * >(hostMemory.snrSamples.data()));
          openclRunTime.queues->at(deviceOptions.deviceID)[0].finish();
        }
      } catch ( cl::Error & err ) {
        std::cerr << "SNR integration loop error -- Batch: " << std::to_string(batch) << ", Step: " << std::to_string(*step) << ", " << err.what() << " " << err.err() << std::endl;
        errorDetected = true;
      }
      if ( options.debug ) {
        try {
          openclRunTime.queues->at(deviceOptions.deviceID)[0].enqueueReadBuffer(deviceMemory.integratedData, CL_TRUE, 0, hostMemory.integratedData.size() * sizeof(outputDataType), reinterpret_cast< void * >(hostMemory.integratedData.data()), 0, &syncPoint);
          syncPoint.wait();
        } catch ( cl::Error & err ) {
          std::cerr << "Impossible to read deviceMemory.integratedData: " << err.what() << " " << err.err() << std::endl;
          errorDetected = true;
        }
        std::cerr << "integratedData" << std::endl;
        if ( options.subbandDedispersion ) {
          for ( unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++ ) {
            std::cerr << "sBeam: " << sBeam << std::endl;
            for ( unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++ ) {
              for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
                std::cerr << "DM: " << (subbandingDM * observation.getNrDMs()) + dm << std::endl;
                for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch() / *step; sample++ ) {
                  std::cerr << hostMemory.integratedData.at((sBeam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / *(hostMemory.integrationSteps.begin()), deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / *(hostMemory.integrationSteps.begin()), deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / *(hostMemory.integrationSteps.begin()), deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + sample) << " ";
                }
                std::cerr << std::endl;
              }
            }
            std::cerr << std::endl;
          }
        } else {
          for ( unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++ ) {
            std::cerr << "sBeam: " << sBeam << std::endl;
            for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
              std::cerr << "DM: " << dm << std::endl;
              for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch() / *step; sample++ ) {
                std::cerr << hostMemory.integratedData.at((sBeam * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / *(hostMemory.integrationSteps.begin()), deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / *(hostMemory.integrationSteps.begin()), deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + sample) << " ";
              }
              std::cerr << std::endl;
            }
            std::cerr << std::endl;
          }
        }
      }
      timers.trigger.start();
      trigger(options.subbandDedispersion, deviceOptions.padding[deviceOptions.deviceName], *step, options.threshold, observation, hostMemory.snrData, hostMemory.snrSamples, triggeredEvents);
      timers.trigger.stop();
      if ( options.debug ) {
        if ( options.subbandDedispersion ) {
          std::cerr << "hostMemory.snrData" << std::endl;
          for ( unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++ ) {
            std::cerr << "sBeam: " << sBeam << std::endl;
            for ( unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++ ) {
              for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
                std::cerr << hostMemory.snrData.at((sBeam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding[deviceOptions.deviceName] / sizeof(float))) + (subbandingDM * observation.getNrDMs()) + dm) << " ";
              }
            }
            std::cerr << std::endl;
          }
        } else {
          std::cerr << "hostMemory.snrData" << std::endl;
          for ( unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++ ) {
            std::cerr << "sBeam: " << sBeam << std::endl;
            for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
              std::cerr << hostMemory.snrData.at((sBeam * observation.getNrDMs(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(float))) + dm) << " ";
            }
            std::cerr << std::endl;
          }
        }
        std::cerr << std::endl;
      }
    }
    if ( errorDetected ) {
      output.close();
#ifdef HAVE_PSRDADA
      if ( dataOptions.dataPSRDADA ) {
        if ( dada_hdu_unlock_read(hostMemory.ringBuffer) != 0 ) {
          std::cerr << "Impossible to unlock the PSRDADA ringbuffer for reading the header." << std::endl;
        }
        dada_hdu_disconnect(hostMemory.ringBuffer);
      }
#endif // HAVE_PSRDADA
      return 1;
    }
    // Print and compact results
    timers.trigger.start();
    if ( options.compactResults ) {
      compact(observation, triggeredEvents, compactedEvents);
      for ( auto beamEvents = compactedEvents.begin(); beamEvents != compactedEvents.end(); ++beamEvents ) {
        for ( auto event = beamEvents->begin(); event != beamEvents->end(); ++event ) {
          unsigned int integration = 0;
          unsigned int delay = 0;
          float firstDM = 0.0f;

          if ( event->integration == 0 ) {
            integration = 1;
          } else {
            integration = event->integration;
          }
          if ( options.subbandDedispersion ) {
            delay = observation.getNrDelayBatches(true) - 1;
            firstDM = observation.getFirstDM(true);
          } else {
            delay = observation.getNrDelayBatches() - 1;
            firstDM = observation.getFirstDM();
          }
          if ( dataOptions.dataPSRDADA ) {
#ifdef HAVE_PSRDADA
            output << event->beam << " " << (batch - delay) << " " << event->sample  << " " << integration << " " << event->compactedIntegration << " " << (((batch - delay) * observation.getNrSamplesPerBatch()) + (event->sample * integration)) * observation.getSamplingTime() << " " << firstDM + (event->DM * observation.getDMStep()) << " " << event->compactedDMs << " " << event->SNR << std::endl;
#endif // HAVE_PSRDADA
          } else {
            output << event->beam << " " << batch << " " << event->sample  << " " << integration << " " << event->compactedIntegration << " " << ((batch * observation.getNrSamplesPerBatch()) + (event->sample * integration)) * observation.getSamplingTime() << " " << firstDM + (event->DM * observation.getDMStep()) << " " << event->compactedDMs << " " << event->SNR << std::endl;
          }
        }
      }
    } else {
      for ( auto beamEvents = triggeredEvents.begin(); beamEvents != triggeredEvents.end(); ++beamEvents ) {
        for ( auto dmEvents = beamEvents->begin(); dmEvents != beamEvents->end(); ++dmEvents) {
          for ( auto event = dmEvents->second.begin(); event != dmEvents->second.end(); ++event ) {
            unsigned int integration = 0;
            unsigned int delay = 0;
            float firstDM = 0.0f;

            if ( event->integration == 0 ) {
              integration = 1;
            } else {
              integration = event->integration;
            }
            if ( options.subbandDedispersion ) {
              delay = observation.getNrDelayBatches(true) - 1;
              firstDM = observation.getFirstDM(true);
            } else {
              delay = observation.getNrDelayBatches() - 1;
              firstDM = observation.getFirstDM();
            }
            if ( dataOptions.dataPSRDADA ) {
#ifdef HAVE_PSRDADA
              output << event->beam << " " << (batch - delay) << " " << event->sample  << " " << integration << " " << (((batch - delay) * observation.getNrSamplesPerBatch()) + (event->sample * integration)) * observation.getSamplingTime() << " " << firstDM + (event->DM * observation.getDMStep()) << " " << event->SNR << std::endl;
#endif // HAVE_PSRDADA
            } else {
              output << event->beam << " " << batch << " " << event->sample  << " " << integration << " " << ((batch * observation.getNrSamplesPerBatch()) + (event->sample * integration)) * observation.getSamplingTime() << " " << firstDM + (event->DM * observation.getDMStep()) << " " << event->SNR << std::endl;
            }
          }
        }
      }
    }
    timers.trigger.stop();
  }
#ifdef HAVE_PSRDADA
  if ( dataOptions.dataPSRDADA ) {
    if ( dada_hdu_unlock_read(hostMemory.ringBuffer) != 0 ) {
      std::cerr << "Impossible to unlock the PSRDADA ringbuffer for reading the header." << std::endl;
    }
    dada_hdu_disconnect(hostMemory.ringBuffer);
  }
#endif // HAVE_PSRDADA
  output.close();
  timers.search.stop();

  // Store statistics before shutting down
  output.open(dataOptions.outputFile + ".stats");
  output << std::fixed << std::setprecision(6);
  output << "# nrDMs" << std::endl;
  if ( options.subbandDedispersion ) {
    output << observation.getNrDMs(true) * observation.getNrDMs() << std::endl;
  } else {
    output << observation.getNrDMs() << std::endl;
  }
  output << "# timers.inputLoad" << std::endl;
  output << timers.inputLoad.getTotalTime() << std::endl;
  output << "# timers.search" << std::endl;
  output << timers.search.getTotalTime() << std::endl;
  output << "# inputHandlingTotal inputHandlingAvg err" << std::endl;
  output << timers.inputHandling.getTotalTime() << " " << timers.inputHandling.getAverageTime() << " " << timers.inputHandling.getStandardDeviation() << std::endl;
  output << "# inputCopyTotal inputCopyAvg err" << std::endl;
  output << timers.inputCopy.getTotalTime() << " " << timers.inputCopy.getAverageTime() << " " << timers.inputCopy.getStandardDeviation() << std::endl;
  if ( ! options.subbandDedispersion ) {
    output << "# dedispersionSingleStepTotal dedispersionSingleStepAvg err" << std::endl;
    output << timers.dedispersionSingleStep.getTotalTime() << " " << timers.dedispersionSingleStep.getAverageTime() << " " << timers.dedispersionSingleStep.getStandardDeviation() << std::endl;
  } else {
    output << "# dedispersionStepOneTotal dedispersionStepOneAvg err" << std::endl;
    output << timers.dedispersionStepOne.getTotalTime() << " " << timers.dedispersionStepOne.getAverageTime() << " " << timers.dedispersionStepOne.getStandardDeviation() << std::endl;
    output << "# dedispersionStepTwoTotal dedispersionStepTwoAvg err" << std::endl;
    output << timers.dedispersionStepTwo.getTotalTime() << " " << timers.dedispersionStepTwo.getAverageTime() << " " << timers.dedispersionStepTwo.getStandardDeviation() << std::endl;
  }
  output << "# integrationTotal integrationAvg err" << std::endl;
  output << timers.integration.getTotalTime() << " " << timers.integration.getAverageTime() << " " << timers.integration.getStandardDeviation() << std::endl;
  output << "# snrTotal snrAvg err" << std::endl;
  output << timers.snr.getTotalTime() << " " << timers.snr.getAverageTime() << " " << timers.snr.getStandardDeviation() << std::endl;
  output << "# outputCopyTotal outputCopyAvg err" << std::endl;
  output << timers.outputCopy.getTotalTime() << " " << timers.outputCopy.getAverageTime() << " " << timers.outputCopy.getStandardDeviation() << std::endl;
  output << "# triggerTimeTotal triggerTimeAvg err" << std::endl;
  output << timers.trigger.getTotalTime() << " " << timers.trigger.getAverageTime() << " " << timers.trigger.getStandardDeviation() << std::endl;
  output.close();

  return 0;
}

