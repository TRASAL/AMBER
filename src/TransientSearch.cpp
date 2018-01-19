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

#include <configuration.hpp>
#include <CommandLine.hpp>
#include <Trigger.hpp>


int main(int argc, char * argv[]) {
  Options options;
  DeviceOptions deviceOptions;
  DataOptions dataOptions;
  Configurations configurations;
  GeneratorOptions generatorOptions;
  AstroData::Observation observation;

  // Process command line arguments
  isa::utils::ArgumentList args(argc, argv);
  try {
    processCommandLineOptions(args, options, deviceOptions, dataOptions, configurations, generatorOptions, observation);
  } catch ( std::exception & err ) {
    return 1;
  }

  // Load observation data
  isa::utils::Timer loadTime;
  std::vector< std::vector< std::vector< inputDataType > * > * > input(observation.getNrBeams());
  std::vector< std::vector< inputDataType > * > inputDADA;
  std::set< unsigned int > integrationSteps;
  if ( dataOptions.dataLOFAR ) {
#ifdef HAVE_HDF5
    input[0] = new std::vector< std::vector< inputDataType > * >(observation.getNrBatches());
    loadTime.start();
    if ( dataOptions.limit ) {
      AstroData::readLOFAR(dataOptions.headerFile, dataOptions.dataFile, observation, deviceOptions.padding[deviceOptions.deviceName], *(input[0]), observation.getNrBatches());
    } else {
      AstroData::readLOFAR(dataOptions.headerFile, dataOptions.dataFile, observation, deviceOptions.padding[deviceOptions.deviceName], *(input[0]));
    }
    loadTime.stop();
#endif // HAVE_HDF5
  } else if ( dataOptions.dataSIGPROC ) {
    input[0] = new std::vector< std::vector< inputDataType > * >(observation.getNrBatches());
    loadTime.start();
    AstroData::readSIGPROC(observation, deviceOptions.padding[deviceOptions.deviceName], inputBits, dataOptions.headerSizeSIGPROC, dataOptions.dataFile, *(input[0]));
    loadTime.stop();
  } else if ( dataOptions.dataPSRDADA ) {
#ifdef HAVE_PSRDADA
    dataOptions.ringBuffer = dada_hdu_create(0);
    dada_hdu_set_key(dataOptions.ringBuffer, dataOptions.dadaKey);
    if ( dada_hdu_connect(dataOptions.ringBuffer) != 0 ) {
      std::cerr << "Impossible to connect to PSRDADA ringbuffer." << std::endl;
    }
    if ( dada_hdu_lock_read(dataOptions.ringBuffer) != 0 ) {
      std::cerr << "Impossible to lock the PSRDADA ringbuffer for reading the header." << std::endl;
    }
    try {
      AstroData::readPSRDADAHeader(observation, *dataOptions.ringBuffer);
    } catch ( AstroData::RingBufferError & err ) {
      std::cerr << "Error: " << err.what() << std::endl;
      return -1;
    }
#endif // HAVE_PSRDADA
  } else {
    for ( unsigned int beam = 0; beam < observation.getNrBeams(); beam++ ) {
      // TODO: if there are multiple synthesized beams, the generated data should take this into account
      input[beam] = new std::vector< std::vector< inputDataType > * >(observation.getNrBatches());
      AstroData::generateSinglePulse(generatorOptions.width, generatorOptions.DM, observation, deviceOptions.padding[deviceOptions.deviceName], *(input[beam]), inputBits, generatorOptions.random);
    }
  }
  std::vector<unsigned int> zappedChannels(observation.getNrChannels(deviceOptions.padding[deviceOptions.deviceName] / sizeof(unsigned int)));
  try {
    AstroData::readZappedChannels(observation, dataOptions.channelsFile, zappedChannels);
    AstroData::readIntegrationSteps(observation, dataOptions.integrationFile, integrationSteps);
  } catch ( AstroData::FileError & err ) {
    std::cerr << err.what() << std::endl;
  }
  if ( DEBUG ) {
    std::cout << "Device: " << deviceOptions.deviceName << std::endl;
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
    std::cout << "Integration steps: " << integrationSteps.size() << std::endl;
    if ( options.subbandDedispersion ) {
      std::cout << "Subbanding DMs: " << observation.getNrDMs(true) << " (" << observation.getFirstDM(true) << ", " << observation.getFirstDM(true) + ((observation.getNrDMs(true) - 1) * observation.getDMStep(true)) << ")" << std::endl;
    }
    std::cout << "DMs: " << observation.getNrDMs() << " (" << observation.getFirstDM() << ", " << observation.getFirstDM() + ((observation.getNrDMs() - 1) * observation.getDMStep()) << ")" << std::endl;
    std::cout << std::endl;
    if ( (dataOptions.dataLOFAR || dataOptions.dataSIGPROC) ) {
      std::cout << "Time to load the input: " << std::fixed << std::setprecision(6) << loadTime.getTotalTime() << " seconds." << std::endl;
      std::cout << std::endl;
    }
  }

  // Initialize OpenCL
  cl::Context * clContext = new cl::Context();
  std::vector< cl::Platform > * clPlatforms = new std::vector< cl::Platform >();
  std::vector< cl::Device > * clDevices = new std::vector< cl::Device >();
  std::vector< std::vector< cl::CommandQueue > > * clQueues = new std::vector< std::vector < cl::CommandQueue > >();

  try {
    isa::OpenCL::initializeOpenCL(deviceOptions.platformID, 1, clPlatforms, clContext, clDevices, clQueues);
  } catch ( isa::OpenCL::OpenCLError & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  // Host memory allocation
  std::vector< float > * shiftsStepOne = Dedispersion::getShifts(observation, deviceOptions.padding[deviceOptions.deviceName]);
  std::vector< float > * shiftsStepTwo = Dedispersion::getShiftsStepTwo(observation, deviceOptions.padding[deviceOptions.deviceName]);
  if ( DEBUG ) {
    if ( options.print ) {
      std::cerr << "shiftsStepOne" << std::endl;
      for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ ) {
        std::cerr << shiftsStepOne->at(channel) << " ";
      }
      std::cerr << std::endl;
      if ( options.subbandDedispersion ) {
        std::cerr << "shiftsStepTwo" << std::endl;
        for ( unsigned int subband = 0; subband < observation.getNrSubbands(); subband++ ) {
          std::cerr << shiftsStepTwo->at(subband) << " ";
        }
        std::cerr << std::endl;
      }
      std::cerr << std::endl;
    }
  }
  if ( options.subbandDedispersion ) {
    observation.setNrSamplesPerBatch(observation.getNrSamplesPerBatch() + static_cast< unsigned int >(shiftsStepTwo->at(0) * (observation.getFirstDM() + ((observation.getNrDMs() - 1) * observation.getDMStep()))), true);
    observation.setNrSamplesPerDispersedBatch(observation.getNrSamplesPerBatch(true) + static_cast< unsigned int >(shiftsStepOne->at(0) * (observation.getFirstDM(true) + ((observation.getNrDMs(true) - 1) * observation.getDMStep(true)))), true);
    observation.setNrDelayBatches(static_cast< unsigned int >(std::ceil(static_cast< double >(observation.getNrSamplesPerDispersedBatch(true)) / observation.getNrSamplesPerBatch())), true);
  } else {
    observation.setNrSamplesPerDispersedBatch(observation.getNrSamplesPerBatch() + static_cast< unsigned int >(shiftsStepOne->at(0) * (observation.getFirstDM() + ((observation.getNrDMs() - 1) * observation.getDMStep()))));
    observation.setNrDelayBatches(static_cast< unsigned int >(std::ceil(static_cast< double >(observation.getNrSamplesPerDispersedBatch()) / observation.getNrSamplesPerBatch())));
  }
  std::vector<unsigned int> beamMapping;
  std::vector< inputDataType > dispersedData;
  std::vector< outputDataType > subbandedData;
  std::vector< outputDataType > dedispersedData;
  std::vector< outputDataType > integratedData;
  std::vector< float > snrData;
  std::vector< unsigned int > snrSamples;

  if ( options.subbandDedispersion ) {
#ifdef HAVE_PSRDADA
    if ( dataOptions.dataPSRDADA ) {
      inputDADA.resize(observation.getNrDelayBatches(true));
      for ( unsigned int batch = 0; batch < observation.getNrDelayBatches(true); batch++ ) {
        if ( inputBits >= 8 ) {
          inputDADA.at(batch) = new std::vector<inputDataType>(observation.getNrBeams() * observation.getNrChannels() * observation.getNrSamplesPerBatch());
        } else {
          inputDADA.at(batch) = new std::vector<inputDataType>(observation.getNrBeams() * observation.getNrChannels() * (observation.getNrSamplesPerBatch() / (8 / inputBits)));
        }
      }
    }
#endif // HAVE_PSRDADA
    if ( configurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getSplitBatches() ) {
      // TODO: add support for splitBatches
    } else {
      if ( inputBits >= 8 ) {
        dispersedData.resize(observation.getNrBeams() * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType)));
      } else {
        dispersedData.resize(observation.getNrBeams() * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType)));
      }
    }
    beamMapping.resize(observation.getNrSynthesizedBeams() * observation.getNrSubbands(deviceOptions.padding[deviceOptions.deviceName] / sizeof(unsigned int)));
    subbandedData.resize(observation.getNrBeams() * observation.getNrDMs(true) * observation.getNrSubbands() * observation.getNrSamplesPerBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType)));
    dedispersedData.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType)));
    integratedData.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / *(integrationSteps.begin()), deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType)));
    snrData.resize(observation.getNrSynthesizedBeams() * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding[deviceOptions.deviceName] / sizeof(float)));
    snrSamples.resize(observation.getNrSynthesizedBeams() * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding[deviceOptions.deviceName] / sizeof(unsigned int)));
  } else {
#ifdef HAVE_PSRDADA
    if ( dataOptions.dataPSRDADA ) {
      inputDADA.resize(observation.getNrDelayBatches());
      for ( unsigned int batch = 0; batch < observation.getNrDelayBatches(); batch++ ) {
        if ( inputBits >= 8 ) {
          inputDADA.at(batch) = new std::vector<inputDataType>(observation.getNrBeams() * observation.getNrChannels() * observation.getNrSamplesPerBatch());
        } else {
          inputDADA.at(batch) = new std::vector<inputDataType>(observation.getNrBeams() * observation.getNrChannels() * (observation.getNrSamplesPerBatch() / (8 / inputBits)));
        }
      }
    }
#endif // HAVE_PSRDADA
    if ( configurations.dedispersionParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getSplitBatches() ) {
      // TODO: add support for splitBatches
    } else {
      if ( inputBits >= 8 ) {
        dispersedData.resize(observation.getNrBeams() * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType)));
      } else {
        dispersedData.resize(observation.getNrBeams() * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType)));
      }
    }
    beamMapping.resize(observation.getNrSynthesizedBeams() * observation.getNrChannels(deviceOptions.padding[deviceOptions.deviceName] / sizeof(unsigned int)));
    dedispersedData.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType)));
    integratedData.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / *(integrationSteps.begin()), deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType)));
    snrData.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(float)));
    snrSamples.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(unsigned int)));
  }
  AstroData::generateBeamMapping(observation, beamMapping, deviceOptions.padding[deviceOptions.deviceName], options.subbandDedispersion);

  if ( observation.getNrDelayBatches() > observation.getNrBatches() ) {
    std::cerr << "Not enough input batches for the search." << std::endl;
    return 1;
  }

  // Device memory allocation and data transfers
  cl::Buffer shiftsStepOne_d;
  cl::Buffer shiftsStepTwo_d;
  cl::Buffer zappedChannels_d;
  cl::Buffer beamMapping_d;
  cl::Buffer dispersedData_d;
  cl::Buffer subbandedData_d;
  cl::Buffer dedispersedData_d;
  cl::Buffer integratedData_d;
  cl::Buffer snrData_d;
  cl::Buffer snrSamples_d;

  try {
    shiftsStepOne_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, shiftsStepOne->size() * sizeof(float), 0, 0);
    zappedChannels_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, zappedChannels.size() * sizeof(unsigned int), 0, 0);
    beamMapping_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, beamMapping.size() * sizeof(unsigned int), 0, 0);
    dispersedData_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, dispersedData.size() * sizeof(inputDataType), 0, 0);
    dedispersedData_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, dedispersedData.size() * sizeof(outputDataType), 0, 0);
    integratedData_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, integratedData.size() * sizeof(outputDataType), 0, 0);
    snrData_d = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, snrData.size() * sizeof(float), 0, 0);
    snrSamples_d = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, snrSamples.size() * sizeof(unsigned int), 0, 0);
    if ( options.subbandDedispersion ) {
      shiftsStepTwo_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, shiftsStepTwo->size() * sizeof(float), 0, 0);
      subbandedData_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, subbandedData.size() * sizeof(outputDataType), 0, 0);
    }
    clQueues->at(deviceOptions.deviceID)[0].enqueueWriteBuffer(shiftsStepOne_d, CL_FALSE, 0, shiftsStepOne->size() * sizeof(float), reinterpret_cast< void * >(shiftsStepOne->data()));
    if ( options.subbandDedispersion ) {
      clQueues->at(deviceOptions.deviceID)[0].enqueueWriteBuffer(shiftsStepTwo_d, CL_FALSE, 0, shiftsStepTwo->size() * sizeof(float), reinterpret_cast< void * >(shiftsStepTwo->data()));
    }
    clQueues->at(deviceOptions.deviceID)[0].enqueueWriteBuffer(beamMapping_d, CL_FALSE, 0, beamMapping.size() * sizeof(unsigned int), reinterpret_cast< void * >(beamMapping.data()));
    clQueues->at(deviceOptions.deviceID)[0].enqueueWriteBuffer(zappedChannels_d, CL_FALSE, 0, zappedChannels.size() * sizeof(unsigned int), reinterpret_cast< void * >(zappedChannels.data()));
    clQueues->at(deviceOptions.deviceID)[0].finish();
  } catch ( cl::Error & err ) {
    std::cerr << "Memory error: " << err.what() << " " << err.err() << std::endl;
    return 1;
  }

  // Generate OpenCL kernels
  std::string * code;
  cl::Kernel * dedispersionK, * dedispersionStepOneK, * dedispersionStepTwoK;
  std::vector< cl::Kernel * > integrationDMsSamplesK(integrationSteps.size() + 1), snrDMsSamplesK(integrationSteps.size() + 1);

  if ( options.subbandDedispersion ) {
    code = Dedispersion::getSubbandDedispersionStepOneOpenCL< inputDataType, outputDataType >(*(configurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))), deviceOptions.padding[deviceOptions.deviceName], inputBits, inputDataName, intermediateDataName, outputDataName, observation, *shiftsStepOne);
    try {
      dedispersionStepOneK = isa::OpenCL::compile("dedispersionStepOne", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(deviceOptions.deviceID));
      if ( configurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getSplitBatches() ) {
        // TODO: add support for splitBatches
      } else {
        dedispersionStepOneK->setArg(0, dispersedData_d);
        dedispersionStepOneK->setArg(1, subbandedData_d);
        dedispersionStepOneK->setArg(2, zappedChannels_d);
        dedispersionStepOneK->setArg(3, shiftsStepOne_d);
      }
    } catch ( isa::OpenCL::OpenCLError & err ) {
      std::cerr << err.what() << std::endl;
      return 1;
    }
    delete code;
    code = Dedispersion::getSubbandDedispersionStepTwoOpenCL< outputDataType >(*(configurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())), deviceOptions.padding[deviceOptions.deviceName], outputDataName, observation, *shiftsStepTwo);
    try {
      dedispersionStepTwoK = isa::OpenCL::compile("dedispersionStepTwo", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(deviceOptions.deviceID));
      dedispersionStepTwoK->setArg(0, subbandedData_d);
      dedispersionStepTwoK->setArg(1, dedispersedData_d);
      dedispersionStepTwoK->setArg(2, beamMapping_d);
      dedispersionStepTwoK->setArg(3, shiftsStepTwo_d);
    } catch ( isa::OpenCL::OpenCLError & err ) {
      std::cerr << err.what() << std::endl;
      return 1;
    }
    delete code;
  } else {
    code = Dedispersion::getDedispersionOpenCL< inputDataType, outputDataType >(*(configurations.dedispersionParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())), deviceOptions.padding[deviceOptions.deviceName], inputBits, inputDataName, intermediateDataName, outputDataName, observation, *shiftsStepOne);
    try {
      dedispersionK = isa::OpenCL::compile("dedispersion", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(deviceOptions.deviceID));
      if ( configurations.dedispersionParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getSplitBatches() ) {
        // TODO: add support for splitBatches
      } else {
        dedispersionK->setArg(0, dispersedData_d);
        dedispersionK->setArg(1, dedispersedData_d);
        dedispersionK->setArg(2, beamMapping_d);
        dedispersionK->setArg(3, zappedChannels_d);
        dedispersionK->setArg(4, shiftsStepOne_d);
      }
    } catch ( isa::OpenCL::OpenCLError & err ) {
      std::cerr << err.what() << std::endl;
      return 1;
    }
    delete code;
  }
  if ( options.subbandDedispersion ) {
    code = SNR::getSNRDMsSamplesOpenCL< outputDataType >(*(configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch())), outputDataName, observation, observation.getNrSamplesPerBatch(), deviceOptions.padding[deviceOptions.deviceName]);
  } else {
    code = SNR::getSNRDMsSamplesOpenCL< outputDataType >(*(configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch())), outputDataName, observation, observation.getNrSamplesPerBatch(), deviceOptions.padding[deviceOptions.deviceName]);
  }
  try {
    snrDMsSamplesK[integrationSteps.size()] = isa::OpenCL::compile("snrDMsSamples" + std::to_string(observation.getNrSamplesPerBatch()), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(deviceOptions.deviceID));
    snrDMsSamplesK[integrationSteps.size()]->setArg(0, dedispersedData_d);
    snrDMsSamplesK[integrationSteps.size()]->setArg(1, snrData_d);
    snrDMsSamplesK[integrationSteps.size()]->setArg(2, snrSamples_d);
  } catch ( isa::OpenCL::OpenCLError & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }
  delete code;
  for ( unsigned int stepNumber = 0; stepNumber < integrationSteps.size(); stepNumber++ ) {
    auto step = integrationSteps.begin();

    std::advance(step, stepNumber);
    if ( options.subbandDedispersion ) {
      code = Integration::getIntegrationDMsSamplesOpenCL< outputDataType >(*(configurations.integrationParameters[deviceOptions.deviceName]->at(observation.getNrDMs(true) * observation.getNrDMs())->at(*step)), observation, outputDataName, *step, deviceOptions.padding[deviceOptions.deviceName]);
    } else {
      code = Integration::getIntegrationDMsSamplesOpenCL< outputDataType >(*(configurations.integrationParameters[deviceOptions.deviceName]->at(observation.getNrDMs())->at(*step)), observation, outputDataName, *step, deviceOptions.padding[deviceOptions.deviceName]);
    }
    try {
      if ( *step > 1 ) {
        integrationDMsSamplesK[stepNumber] = isa::OpenCL::compile("integrationDMsSamples" + std::to_string(*step), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(deviceOptions.deviceID));
        integrationDMsSamplesK[stepNumber]->setArg(0, dedispersedData_d);
        integrationDMsSamplesK[stepNumber]->setArg(1, integratedData_d);
      }
    } catch ( isa::OpenCL::OpenCLError & err ) {
      std::cerr << err.what() << std::endl;
      return 1;
    }
    delete code;
    if ( options.subbandDedispersion ) {
      code = SNR::getSNRDMsSamplesOpenCL< outputDataType >(*(configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)), outputDataName, observation, observation.getNrSamplesPerBatch() / *step, deviceOptions.padding[deviceOptions.deviceName]);
    } else {
      code = SNR::getSNRDMsSamplesOpenCL< outputDataType >(*(configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)), outputDataName, observation, observation.getNrSamplesPerBatch() / *step, deviceOptions.padding[deviceOptions.deviceName]);
    }
    try {
      snrDMsSamplesK[stepNumber] = isa::OpenCL::compile("snrDMsSamples" + std::to_string(observation.getNrSamplesPerBatch() / *step), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(deviceOptions.deviceID));
      snrDMsSamplesK[stepNumber]->setArg(0, integratedData_d);
      snrDMsSamplesK[stepNumber]->setArg(1, snrData_d);
      snrDMsSamplesK[stepNumber]->setArg(2, snrSamples_d);
    } catch ( isa::OpenCL::OpenCLError & err ) {
      std::cerr << err.what() << std::endl;
      return 1;
    }
    delete code;
  }

  // Set execution parameters
  cl::NDRange dedispersionGlobal, dedispersionLocal;
  cl::NDRange dedispersionStepOneGlobal, dedispersionStepOneLocal;
  cl::NDRange dedispersionStepTwoGlobal, dedispersionStepTwoLocal;
  if ( options.subbandDedispersion ) {
    dedispersionStepOneGlobal = cl::NDRange(isa::utils::pad(observation.getNrSamplesPerBatch(true) / configurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getNrItemsD0(), configurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getNrThreadsD0()), observation.getNrDMs(true) / configurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getNrItemsD1(), observation.getNrBeams() * observation.getNrSubbands());
    dedispersionStepOneLocal = cl::NDRange(configurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getNrThreadsD0(), configurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getNrThreadsD1(), 1);
    if ( DEBUG ) {
      std::cout << "DedispersionStepOne" << std::endl;
        std::cout << "Global: " << isa::utils::pad(observation.getNrSamplesPerBatch(true) / configurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getNrItemsD0(), configurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getNrThreadsD0()) << ", " << observation.getNrDMs(true) / configurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getNrItemsD1() << ", " << observation.getNrBeams() * observation.getNrSubbands() << std::endl;
        std::cout << "Local: " << configurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getNrThreadsD0() << ", " << configurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getNrThreadsD1() << ", 1" << std::endl;
        std::cout << "Parameters: ";
        std::cout << configurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->print() << std::endl;
        std::cout << std::endl;
    }
    dedispersionStepTwoGlobal = cl::NDRange(isa::utils::pad(observation.getNrSamplesPerBatch(true) / configurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrItemsD0(), configurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrThreadsD0()), observation.getNrDMs() / configurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrItemsD1(), observation.getNrSynthesizedBeams() * observation.getNrDMs(true));
    dedispersionStepTwoLocal = cl::NDRange(configurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrThreadsD0(), configurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrThreadsD1(), 1);
    if ( DEBUG ) {
      std::cout << "DedispersionStepTwo" << std::endl;
        std::cout << "Global: " << isa::utils::pad(observation.getNrSamplesPerBatch(true) / configurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrItemsD0(), configurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrThreadsD0()) << ", " << observation.getNrDMs() / configurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrItemsD1() << ", " << observation.getNrSynthesizedBeams() * observation.getNrDMs(true) << std::endl;
        std::cout << "Local: " << configurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrThreadsD0() << ", " << configurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrThreadsD1() << ", 1" << std::endl;
        std::cout << "Parameters: ";
        std::cout << configurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->print() << std::endl;
        std::cout << std::endl;
    }
  } else {
    dedispersionGlobal = cl::NDRange(isa::utils::pad(observation.getNrSamplesPerBatch() / configurations.dedispersionParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrItemsD0(), configurations.dedispersionParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrThreadsD0()), observation.getNrDMs() / configurations.dedispersionParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrItemsD1(), observation.getNrSynthesizedBeams());
    dedispersionLocal = cl::NDRange(configurations.dedispersionParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrThreadsD0(), configurations.dedispersionParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrThreadsD1(), 1);
    if ( DEBUG ) {
      std::cout << "Dedispersion" << std::endl;
      std::cout << "Global: " << isa::utils::pad(observation.getNrSamplesPerBatch() / configurations.dedispersionParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrItemsD0(), configurations.dedispersionParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrThreadsD0()) << ", " << observation.getNrDMs() / configurations.dedispersionParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrItemsD1() << ", " << observation.getNrSynthesizedBeams() << std::endl;
      std::cout << "Local: " << configurations.dedispersionParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrThreadsD0() << ", " << configurations.dedispersionParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrThreadsD1() << ", 1" << std::endl;
      std::cout << "Parameters: ";
      std::cout << configurations.dedispersionParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->print() << std::endl;
      std::cout << std::endl;
    }
  }
  std::vector< cl::NDRange > integrationGlobal(integrationSteps.size());
  std::vector< cl::NDRange > integrationLocal(integrationSteps.size());
  std::vector< cl::NDRange > snrDMsSamplesGlobal(integrationSteps.size() + 1);
  std::vector< cl::NDRange > snrDMsSamplesLocal(integrationSteps.size() + 1);
  if ( options.subbandDedispersion ) {
    snrDMsSamplesGlobal[integrationSteps.size()] = cl::NDRange(configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch())->getNrThreadsD0(), observation.getNrDMs(true) * observation.getNrDMs(), observation.getNrSynthesizedBeams());
    snrDMsSamplesLocal[integrationSteps.size()] = cl::NDRange(configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch())->getNrThreadsD0(), 1, 1);
    if ( DEBUG ) {
      std::cout << "SNRDMsSamples (" + std::to_string(observation.getNrSamplesPerBatch()) + ")" << std::endl;
      std::cout << "Global: " << configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch())->getNrThreadsD0() << ", " << observation.getNrDMs(true) * observation.getNrDMs() << " " << observation.getNrSynthesizedBeams() << std::endl;
      std::cout << "Local: " << configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch())->getNrThreadsD0() << ", 1, 1" << std::endl;
      std::cout << "Parameters: ";
      std::cout << configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch())->print() << std::endl;
      std::cout << std::endl;
    }
  } else {
    snrDMsSamplesGlobal[integrationSteps.size()] = cl::NDRange(configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch())->getNrThreadsD0(), observation.getNrDMs(), observation.getNrSynthesizedBeams());
    snrDMsSamplesLocal[integrationSteps.size()] = cl::NDRange(configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch())->getNrThreadsD0(), 1, 1);
    if ( DEBUG ) {
      std::cout << "SNRDMsSamples (" + std::to_string(observation.getNrSamplesPerBatch()) + ")" << std::endl;
      std::cout << "Global: " << configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch())->getNrThreadsD0() << ", " << observation.getNrDMs() << ", " << observation.getNrSynthesizedBeams() << std::endl;
      std::cout << "Local: " << configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch())->getNrThreadsD0() << ", 1, 1" << std::endl;
      std::cout << "Parameters: ";
      std::cout << configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch())->print() << std::endl;
      std::cout << std::endl;
    }
  }
  for ( unsigned int stepNumber = 0; stepNumber < integrationSteps.size(); stepNumber++ ) {
    auto step = integrationSteps.begin();

    std::advance(step, stepNumber);
    if ( options.subbandDedispersion ) {
      integrationGlobal[stepNumber] = cl::NDRange(configurations.integrationParameters[deviceOptions.deviceName]->at(observation.getNrDMs(true) * observation.getNrDMs())->at(*step)->getNrThreadsD0() * ((observation.getNrSamplesPerBatch() / *step) / configurations.integrationParameters[deviceOptions.deviceName]->at(observation.getNrDMs(true) * observation.getNrDMs())->at(*step)->getNrItemsD0()), observation.getNrDMs(true) * observation.getNrDMs(), observation.getNrSynthesizedBeams());
      integrationLocal[stepNumber] = cl::NDRange(configurations.integrationParameters[deviceOptions.deviceName]->at(observation.getNrDMs(true) * observation.getNrDMs())->at(*step)->getNrThreadsD0(), 1, 1);
      if ( DEBUG ) {
        std::cout << "integrationDMsSamples (" + std::to_string(*step) + ")" << std::endl;
        std::cout << "Global: " << configurations.integrationParameters[deviceOptions.deviceName]->at(observation.getNrDMs(true) * observation.getNrDMs())->at(*step)->getNrThreadsD0() * ((observation.getNrSamplesPerBatch() / *step) / configurations.integrationParameters[deviceOptions.deviceName]->at(observation.getNrDMs(true) * observation.getNrDMs())->at(*step)->getNrItemsD0()) << ", " << observation.getNrDMs(true) * observation.getNrDMs() << ", " << observation.getNrSynthesizedBeams() << std::endl;
        std::cout << "Local: " << configurations.integrationParameters[deviceOptions.deviceName]->at(observation.getNrDMs(true) * observation.getNrDMs())->at(*step)->getNrThreadsD0() << ", 1, 1" << std::endl;
        std::cout << "Parameters: ";
        std::cout << configurations.integrationParameters[deviceOptions.deviceName]->at(observation.getNrDMs(true) * observation.getNrDMs())->at(*step)->print() << std::endl;
        std::cout << std::endl;
      }
      snrDMsSamplesGlobal[stepNumber] = cl::NDRange(configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)->getNrThreadsD0(), observation.getNrDMs(true) * observation.getNrDMs(), observation.getNrSynthesizedBeams());
      snrDMsSamplesLocal[stepNumber] = cl::NDRange(configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)->getNrThreadsD0(), 1, 1);
      if ( DEBUG ) {
        std::cout << "SNRDMsSamples (" + std::to_string(observation.getNrSamplesPerBatch() / *step) + ")" << std::endl;
        std::cout << "Global: " << configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)->getNrThreadsD0() << ", " << observation.getNrDMs(true) * observation.getNrDMs() << " " << observation.getNrSynthesizedBeams() << std::endl;
        std::cout << "Local: " << configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)->getNrThreadsD0() << ", 1, 1" << std::endl;
        std::cout << "Parameters: ";
        std::cout << configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)->print() << std::endl;
        std::cout << std::endl;
      }
    } else {
      integrationGlobal[stepNumber] = cl::NDRange(configurations.integrationParameters[deviceOptions.deviceName]->at(observation.getNrDMs())->at(*step)->getNrThreadsD0() * ((observation.getNrSamplesPerBatch() / *step) / configurations.integrationParameters[deviceOptions.deviceName]->at(observation.getNrDMs())->at(*step)->getNrItemsD0()), observation.getNrDMs(), observation.getNrSynthesizedBeams());
      integrationLocal[stepNumber] = cl::NDRange(configurations.integrationParameters[deviceOptions.deviceName]->at(observation.getNrDMs())->at(*step)->getNrThreadsD0(), 1, 1);
      if ( DEBUG ) {
        std::cout << "integrationDMsSamples (" + std::to_string(*step) + ")" << std::endl;
        std::cout << "Global: " << configurations.integrationParameters[deviceOptions.deviceName]->at(observation.getNrDMs())->at(*step)->getNrThreadsD0() * ((observation.getNrSamplesPerBatch() / *step) / configurations.integrationParameters[deviceOptions.deviceName]->at(observation.getNrDMs())->at(*step)->getNrItemsD0()) << ", " << observation.getNrDMs() << ", " << observation.getNrSynthesizedBeams() << std::endl;
        std::cout << "Local: " << configurations.integrationParameters[deviceOptions.deviceName]->at(observation.getNrDMs())->at(*step)->getNrThreadsD0() << ", 1, 1" << std::endl;
        std::cout << "Parameters: ";
        std::cout << configurations.integrationParameters[deviceOptions.deviceName]->at(observation.getNrDMs())->at(*step)->print() << std::endl;
        std::cout << std::endl;
      }
      snrDMsSamplesGlobal[stepNumber] = cl::NDRange(configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)->getNrThreadsD0(), observation.getNrDMs(), observation.getNrSynthesizedBeams());
      snrDMsSamplesLocal[stepNumber] = cl::NDRange(configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)->getNrThreadsD0(), 1, 1);
      if ( DEBUG ) {
        std::cout << "SNRDMsSamples (" + std::to_string(observation.getNrSamplesPerBatch() / *step) + ")" << std::endl;
        std::cout << "Global: " << configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)->getNrThreadsD0() << ", " << observation.getNrDMs() << " " << observation.getNrSynthesizedBeams() << std::endl;
        std::cout << "Local: " << configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)->getNrThreadsD0() << ", 1, 1" << std::endl;
        std::cout << "Parameters: ";
        std::cout << configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)->print() << std::endl;
        std::cout << std::endl;
      }
    }
  }

  // Search loop
  std::ofstream output;
  bool errorDetected = false;
  cl::Event syncPoint;
  isa::utils::Timer searchTimer, inputHandlingTimer, inputCopyTimer, dedispersionTimer, dedispersionStepOneTimer, dedispersionStepTwoTimer, integrationTimer, snrDMsSamplesTimer, outputCopyTimer, triggerTimer;

  searchTimer.start();
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
    inputHandlingTimer.start();
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
          if ( !configurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getSplitBatches() ) {
            for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ ) {
              for ( unsigned int chunk = 0; chunk < observation.getNrDelayBatches(true) - 1; chunk++ ) {
                if ( inputBits >= 8 ) {
                  memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (chunk * observation.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&((input[beam]->at(batch + chunk))->at(channel * observation.getNrSamplesPerBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))))), observation.getNrSamplesPerBatch() * sizeof(inputDataType));
                } else {
                  memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (chunk * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&((input[beam]->at(batch + chunk))->at(channel * isa::utils::pad(observation.getNrSamplesPerBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))))), (observation.getNrSamplesPerBatch() / (8 / inputBits)) * sizeof(inputDataType));
                }
              }
              if ( inputBits >= 8 ) {
                memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + ((observation.getNrDelayBatches(true) - 1) * observation.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&((input[beam]->at(batch + (observation.getNrDelayBatches(true) - 1)))->at(channel * observation.getNrSamplesPerBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))))), (observation.getNrSamplesPerDispersedBatch(true) % observation.getNrSamplesPerBatch()) * sizeof(inputDataType));
              } else {
                memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + ((observation.getNrDelayBatches(true) - 1) * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&((input[beam]->at(batch + (observation.getNrDelayBatches(true) - 1)))->at(channel * isa::utils::pad(observation.getNrSamplesPerBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))))), ((observation.getNrSamplesPerDispersedBatch(true) % observation.getNrSamplesPerBatch()) / (8 / inputBits)) * sizeof(inputDataType));
              }
            }
          }
        } else {
          if ( !configurations.dedispersionParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getSplitBatches() ) {
            for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ ) {
              for ( unsigned int chunk = 0; chunk < observation.getNrDelayBatches() - 1; chunk++ ) {
                if ( inputBits >= 8 ) {
                  memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (chunk * observation.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&((input[beam]->at(batch + chunk))->at(channel * observation.getNrSamplesPerBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))))), observation.getNrSamplesPerBatch() * sizeof(inputDataType));
                } else {
                  memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (chunk * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&((input[beam]->at(batch + chunk))->at(channel * isa::utils::pad(observation.getNrSamplesPerBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))))), (observation.getNrSamplesPerBatch() / (8 / inputBits)) * sizeof(inputDataType));
                }
              }
              if ( inputBits >= 8 ) {
                memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + ((observation.getNrDelayBatches() - 1) * observation.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&((input[beam]->at(batch + (observation.getNrDelayBatches() - 1)))->at(channel * observation.getNrSamplesPerBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))))), (observation.getNrSamplesPerDispersedBatch() % observation.getNrSamplesPerBatch()) * sizeof(inputDataType));
              } else {
                memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + ((observation.getNrDelayBatches() - 1) * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&((input[beam]->at(batch + (observation.getNrDelayBatches() - 1)))->at(channel * isa::utils::pad(observation.getNrSamplesPerBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))))), ((observation.getNrSamplesPerDispersedBatch() % observation.getNrSamplesPerBatch()) / (8 / inputBits)) * sizeof(inputDataType));
              }
            }
          }
        }
      }
    } else {
#ifdef HAVE_PSRDADA
      try {
        if ( ipcbuf_eod(reinterpret_cast< ipcbuf_t * >(dataOptions.ringBuffer->data_block)) ) {
          errorDetected = true;
          break;
        }
        if ( options.subbandDedispersion ) {
          AstroData::readPSRDADA(*dataOptions.ringBuffer, inputDADA.at(batch % observation.getNrDelayBatches(true)));
        } else {
          AstroData::readPSRDADA(*dataOptions.ringBuffer, inputDADA.at(batch % observation.getNrDelayBatches()));
        }
      } catch ( AstroData::RingBufferError & err ) {
        std::cerr << "Error: " << err.what() << std::endl;
        return -1;
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
          if ( !configurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getSplitBatches() ) {
            for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ ) {
              for ( unsigned int chunk = batch - (observation.getNrDelayBatches(true) - 1); chunk < batch; chunk++ ) {
                // Full batches
                if ( inputBits >= 8 ) {
                  memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + ((chunk - (batch - (observation.getNrDelayBatches(true) - 1))) * observation.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&(inputDADA.at(chunk % observation.getNrDelayBatches(true))->at((beam * observation.getNrChannels() * observation.getNrSamplesPerBatch()) + (channel * observation.getNrSamplesPerBatch())))), observation.getNrSamplesPerBatch() * sizeof(inputDataType));
                } else {
                  memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + ((chunk - (batch - (observation.getNrDelayBatches(true) - 1))) * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&(inputDADA.at(chunk % observation.getNrDelayBatches(true))->at((beam * observation.getNrChannels() * (observation.getNrSamplesPerBatch() / (8 / inputBits))) + (channel * (observation.getNrSamplesPerBatch() / (8 / inputBits)))))), (observation.getNrSamplesPerBatch() / (8 / inputBits)) * sizeof(inputDataType));
                }
              }
              // Remainder (part of current batch)
              if ( inputBits >= 8 ) {
                memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + ((observation.getNrDelayBatches(true) - 1) * observation.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&(inputDADA.at(batch % observation.getNrDelayBatches(true))->at((beam * observation.getNrChannels() * observation.getNrSamplesPerBatch()) + (channel * observation.getNrSamplesPerBatch())))), (observation.getNrSamplesPerDispersedBatch(true) % observation.getNrSamplesPerBatch()) * sizeof(inputDataType));
              } else {
                memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + ((observation.getNrDelayBatches(true) - 1) * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&(inputDADA.at(batch % observation.getNrDelayBatches(true))->at((beam * observation.getNrChannels() * (observation.getNrSamplesPerBatch() / (8 / inputBits))) + (channel * (observation.getNrSamplesPerBatch() / (8 / inputBits)))))), ((observation.getNrSamplesPerDispersedBatch(true) % observation.getNrSamplesPerBatch()) / (8 / inputBits)) * sizeof(inputDataType));
              }
            }
          }
        } else {
          if ( !configurations.dedispersionParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getSplitBatches() ) {
            for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ ) {
              for ( unsigned int chunk = batch - (observation.getNrDelayBatches() - 1); chunk < batch; chunk++ ) {
                if ( inputBits >= 8 ) {
                  memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + ((chunk - (batch - (observation.getNrDelayBatches() - 1))) * observation.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&(inputDADA.at(chunk % observation.getNrDelayBatches())->at((beam * observation.getNrChannels() * observation.getNrSamplesPerBatch()) + (channel * observation.getNrSamplesPerBatch())))), observation.getNrSamplesPerBatch() * sizeof(inputDataType));
                } else {
                  memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + ((chunk - (batch - (observation.getNrDelayBatches() - 1))) * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&(inputDADA.at(chunk % observation.getNrDelayBatches())->at((beam * observation.getNrChannels() * (observation.getNrSamplesPerBatch() / (8 / inputBits))) + (channel * (observation.getNrSamplesPerBatch() / (8 / inputBits)))))), (observation.getNrSamplesPerBatch() / (8 / inputBits)) * sizeof(inputDataType));
                }
              }
              if ( inputBits >= 8 ) {
                memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + ((observation.getNrDelayBatches() - 1) * observation.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&(inputDADA.at(batch % observation.getNrDelayBatches())->at((beam * observation.getNrChannels() * observation.getNrSamplesPerBatch()) + (channel * observation.getNrSamplesPerBatch())))), (observation.getNrSamplesPerDispersedBatch() % observation.getNrSamplesPerBatch()) * sizeof(inputDataType));
              } else {
                memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + ((observation.getNrDelayBatches() - 1) * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&(inputDADA.at(batch % observation.getNrDelayBatches())->at((beam * observation.getNrChannels() * (observation.getNrSamplesPerBatch() / (8 / inputBits))) + (channel * (observation.getNrSamplesPerBatch() / (8 / inputBits)))))), ((observation.getNrSamplesPerDispersedBatch() % observation.getNrSamplesPerBatch()) / (8 / inputBits)) * sizeof(inputDataType));
              }
            }
          }
        }
      }
#endif // HAVE_PSRDADA
    }
    inputHandlingTimer.stop();
    // Copy input from host to device
    try {
      if ( SYNC ) {
        inputCopyTimer.start();
        if ( options.subbandDedispersion ) {
          if ( configurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getSplitBatches() ) {
            // TODO: add support for splitBatches
          } else {
            clQueues->at(deviceOptions.deviceID)[0].enqueueWriteBuffer(dispersedData_d, CL_TRUE, 0, dispersedData.size() * sizeof(inputDataType), reinterpret_cast< void * >(dispersedData.data()), 0, &syncPoint);
          }
        } else {
          if ( configurations.dedispersionParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getSplitBatches() ) {
            // TODO: add support for splitBatches
          } else {
            clQueues->at(deviceOptions.deviceID)[0].enqueueWriteBuffer(dispersedData_d, CL_TRUE, 0, dispersedData.size() * sizeof(inputDataType), reinterpret_cast< void * >(dispersedData.data()), 0, &syncPoint);
          }
        }
        syncPoint.wait();
        inputCopyTimer.stop();
      } else {
        if ( options.subbandDedispersion ) {
          if ( configurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getSplitBatches() ) {
            // TODO: add support for splitBatches
          } else {
            clQueues->at(deviceOptions.deviceID)[0].enqueueWriteBuffer(dispersedData_d, CL_FALSE, 0, dispersedData.size() * sizeof(inputDataType), reinterpret_cast< void * >(dispersedData.data()));
          }
        } else {
          if ( configurations.dedispersionParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getSplitBatches() ) {
            // TODO: add support for splitBatches
          } else {
            clQueues->at(deviceOptions.deviceID)[0].enqueueWriteBuffer(dispersedData_d, CL_FALSE, 0, dispersedData.size() * sizeof(inputDataType), reinterpret_cast< void * >(dispersedData.data()));
          }
        }
      }
      if ( DEBUG ) {
        if ( options.print ) {
          // TODO: add support for splitBatches
          std::cerr << "dispersedData" << std::endl;
          if ( options.subbandDedispersion ) {
            if ( inputBits >= 8 ) {
              for ( unsigned int beam = 0; beam < observation.getNrBeams(); beam++ ) {
                std::cerr << "Beam: " << beam << std::endl;
                for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ ) {
                  for ( unsigned int sample = 0; sample < observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType)); sample++ ) {
                    std::cerr << static_cast< float >(dispersedData[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + sample]) << " ";
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
                    std::cerr << static_cast< float >(dispersedData[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType))) + sample]) << " ";
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
      }
    } catch ( cl::Error & err ) {
      std::cerr << "Input copy error -- Batch: " << std::to_string(batch) << ", " << err.what() << " " << err.err() << std::endl;
      errorDetected = true;
    }
    if ( options.subbandDedispersion ) {
      if ( configurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getSplitBatches() && (batch < observation.getNrDelayBatches()) ) {
        // Not enough batches in the buffer to start the search
        continue;
      }
    } else {
      if ( configurations.dedispersionParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getSplitBatches() && (batch < observation.getNrDelayBatches()) ) {
        // Not enough batches in the buffer to start the search
        continue;
      }
    }

    // Dedispersion
    if ( options.subbandDedispersion ) {
      if ( configurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getSplitBatches() ) {
        // TODO: add support for splitBatches
      }
      if ( SYNC ) {
        try {
          dedispersionTimer.start();
          dedispersionStepOneTimer.start();
          clQueues->at(deviceOptions.deviceID)[0].enqueueNDRangeKernel(*dedispersionStepOneK, cl::NullRange, dedispersionStepOneGlobal, dedispersionStepOneLocal, 0, &syncPoint);
          syncPoint.wait();
          dedispersionStepOneTimer.stop();
        } catch ( cl::Error & err ) {
          std::cerr << "Dedispersion Step One error -- Batch: " << std::to_string(batch) << ", " << err.what() << " " << err.err() << std::endl;
          errorDetected = true;
        }
        try {
          dedispersionStepTwoTimer.start();
          clQueues->at(deviceOptions.deviceID)[0].enqueueNDRangeKernel(*dedispersionStepTwoK, cl::NullRange, dedispersionStepTwoGlobal, dedispersionStepTwoLocal, 0, &syncPoint);
          syncPoint.wait();
          dedispersionStepTwoTimer.stop();
          dedispersionTimer.stop();
        } catch ( cl::Error & err ) {
          std::cerr << "Dedispersion Step Two error -- Batch: " << std::to_string(batch) << ", " << err.what() << " " << err.err() << std::endl;
          errorDetected = true;
        }
      } else {
        try {
          clQueues->at(deviceOptions.deviceID)[0].enqueueNDRangeKernel(*dedispersionStepOneK, cl::NullRange, dedispersionStepOneGlobal, dedispersionStepOneLocal, 0, 0);
        } catch ( cl::Error & err ) {
          std::cerr << "Dedispersion Step One error -- Batch: " << std::to_string(batch) << ", " << err.what() << " " << err.err() << std::endl;
          errorDetected = true;
        }
        try {
          clQueues->at(deviceOptions.deviceID)[0].enqueueNDRangeKernel(*dedispersionStepTwoK, cl::NullRange, dedispersionStepTwoGlobal, dedispersionStepTwoLocal, 0, 0);
        } catch ( cl::Error & err ) {
          std::cerr << "Dedispersion Step Two error -- Batch: " << std::to_string(batch) << ", " << err.what() << " " << err.err() << std::endl;
          errorDetected = true;
        }
      }
    } else {
      try {
          if ( configurations.dedispersionParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getSplitBatches() ) {
            // TODO: add support for splitBatches
          }
          if ( SYNC ) {
            dedispersionTimer.start();
            clQueues->at(deviceOptions.deviceID)[0].enqueueNDRangeKernel(*dedispersionK, cl::NullRange, dedispersionGlobal, dedispersionLocal, 0, &syncPoint);
            syncPoint.wait();
            dedispersionTimer.stop();
          } else {
            clQueues->at(deviceOptions.deviceID)[0].enqueueNDRangeKernel(*dedispersionK, cl::NullRange, dedispersionGlobal, dedispersionLocal);
          }
      } catch ( cl::Error & err ) {
        std::cerr << "Dedispersion error -- Batch: " << std::to_string(batch) << ", " << err.what() << " " << err.err() << std::endl;
        errorDetected = true;
      }
    }
    if ( DEBUG ) {
      if ( options.print ) {
        if ( options.subbandDedispersion ) {
          try {
            clQueues->at(deviceOptions.deviceID)[0].enqueueReadBuffer(subbandedData_d, CL_TRUE, 0, subbandedData.size() * sizeof(outputDataType), reinterpret_cast< void * >(subbandedData.data()), 0, &syncPoint);
            syncPoint.wait();
            clQueues->at(deviceOptions.deviceID)[0].enqueueReadBuffer(dedispersedData_d, CL_TRUE, 0, dedispersedData.size() * sizeof(outputDataType), reinterpret_cast< void * >(dedispersedData.data()), 0, &syncPoint);
            syncPoint.wait();
            std::cerr << "subbandedData" << std::endl;
            for ( unsigned int beam = 0; beam < observation.getNrBeams(); beam++ ) {
              std::cerr << "Beam: " << beam << std::endl;
              for ( unsigned int dm = 0; dm < observation.getNrDMs(true); dm++ ) {
                std::cerr << "Subbanding DM: " << dm << std::endl;
                for ( unsigned int subband = 0; subband < observation.getNrSubbands(); subband++ ) {
                  for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(true); sample++ ) {
                    std::cerr << subbandedData[(beam * observation.getNrDMs(true) * observation.getNrSubbands() * observation.getNrSamplesPerBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + (dm * observation.getNrSubbands() * observation.getNrSamplesPerBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + (subband * observation.getNrSamplesPerBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + sample] << " ";
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
                    std::cerr << dedispersedData[(sBeam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + (dm * observation.getNrSamplesPerBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + sample] << " ";
                  }
                  std::cerr << std::endl;
                }
                std::cerr << std::endl;
              }
              std::cerr << std::endl;
            }
          } catch ( cl::Error & err) {
            std::cerr << "Impossible to read subbandedData_d and dedispersedData_d: " << err.what() << " " << err.err() << std::endl;
            errorDetected = true;
          }
        } else {
          try {
            clQueues->at(deviceOptions.deviceID)[0].enqueueReadBuffer(dedispersedData_d, CL_TRUE, 0, dedispersedData.size() * sizeof(outputDataType), reinterpret_cast< void * >(dedispersedData.data()), 0, &syncPoint);
            syncPoint.wait();
            std::cerr << "dedispersedData" << std::endl;
            for ( unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++ ) {
              std::cerr << "sBeam: " << sBeam << std::endl;
              for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
                std::cerr << "DM: " << dm << std::endl;
                for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++ ) {
                  std::cerr << dedispersedData[(sBeam * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + (dm * observation.getNrSamplesPerBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + sample] << " ";
                }
                std::cerr << std::endl;
              }
              std::cerr << std::endl;
            }
          } catch ( cl::Error & err ) {
            std::cerr << "Impossible to read dedispersedData_d: " << err.what() << " " << err.err() << std::endl;
            errorDetected = true;
          }
        }
      }
    }

    // SNR of dedispersed data
    try {
      if ( SYNC ) {
        snrDMsSamplesTimer.start();
        clQueues->at(deviceOptions.deviceID)[0].enqueueNDRangeKernel(*snrDMsSamplesK[integrationSteps.size()], cl::NullRange, snrDMsSamplesGlobal[integrationSteps.size()], snrDMsSamplesLocal[integrationSteps.size()], 0, &syncPoint);
        syncPoint.wait();
        snrDMsSamplesTimer.stop();
        outputCopyTimer.start();
        clQueues->at(deviceOptions.deviceID)[0].enqueueReadBuffer(snrData_d, CL_TRUE, 0, snrData.size() * sizeof(float), reinterpret_cast< void * >(snrData.data()), 0, &syncPoint);
        syncPoint.wait();
        clQueues->at(deviceOptions.deviceID)[0].enqueueReadBuffer(snrSamples_d, CL_TRUE, 0, snrSamples.size() * sizeof(unsigned int), reinterpret_cast< void * >(snrSamples.data()), 0, &syncPoint);
        syncPoint.wait();
        outputCopyTimer.stop();
      } else {
        clQueues->at(deviceOptions.deviceID)[0].enqueueNDRangeKernel(*snrDMsSamplesK[integrationSteps.size()], cl::NullRange, snrDMsSamplesGlobal[integrationSteps.size()], snrDMsSamplesLocal[integrationSteps.size()]);
        clQueues->at(deviceOptions.deviceID)[0].enqueueReadBuffer(snrData_d, CL_FALSE, 0, snrData.size() * sizeof(float), reinterpret_cast< void * >(snrData.data()));
        clQueues->at(deviceOptions.deviceID)[0].enqueueReadBuffer(snrSamples_d, CL_FALSE, 0, snrSamples.size() * sizeof(unsigned int), reinterpret_cast< void * >(snrSamples.data()));
        clQueues->at(deviceOptions.deviceID)[0].finish();
      }
    } catch ( cl::Error & err ) {
      std::cerr << "SNR dedispersed data error -- Batch: " << std::to_string(batch) << ", " << err.what() << " " << err.err() << std::endl;
      errorDetected = true;
    }
    triggerTimer.start();
    trigger(options.subbandDedispersion, deviceOptions.padding[deviceOptions.deviceName], 0, options.threshold, observation, snrData, snrSamples, triggeredEvents);
    triggerTimer.stop();
    if ( DEBUG ) {
      if ( options.print ) {
        if ( options.subbandDedispersion ) {
          std::cerr << "snrData" << std::endl;
          for ( unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++ ) {
            std::cerr << "sBeam: " << sBeam << std::endl;
            for ( unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++ ) {
              for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
                std::cerr << snrData[(sBeam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding[deviceOptions.deviceName] / sizeof(float))) + (subbandingDM * observation.getNrDMs()) + dm] << " ";
              }
            }
            std::cerr << std::endl;
          }
        } else {
          std::cerr << "snrData" << std::endl;
          for ( unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++ ) {
            std::cerr << "sBeam: " << sBeam << std::endl;
            for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
              std::cerr << snrData[(sBeam * observation.getNrDMs(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(float))) + dm] << " ";
            }
            std::cerr << std::endl;
          }
        }
        std::cerr << std::endl;
      }
    }

    // Integration and SNR loop
    for ( unsigned int stepNumber = 0; stepNumber < integrationSteps.size(); stepNumber++ ) {
      auto step = integrationSteps.begin();

      std::advance(step, stepNumber);
      try {
        if ( SYNC ) {
          integrationTimer.start();
          clQueues->at(deviceOptions.deviceID)[0].enqueueNDRangeKernel(*integrationDMsSamplesK[stepNumber], cl::NullRange, integrationGlobal[stepNumber], integrationLocal[stepNumber], 0, &syncPoint);
          syncPoint.wait();
          integrationTimer.stop();
          snrDMsSamplesTimer.start();
          clQueues->at(deviceOptions.deviceID)[0].enqueueNDRangeKernel(*snrDMsSamplesK[stepNumber], cl::NullRange, snrDMsSamplesGlobal[stepNumber], snrDMsSamplesLocal[stepNumber], 0, &syncPoint);
          syncPoint.wait();
          snrDMsSamplesTimer.stop();
          outputCopyTimer.start();
          clQueues->at(deviceOptions.deviceID)[0].enqueueReadBuffer(snrData_d, CL_TRUE, 0, snrData.size() * sizeof(float), reinterpret_cast< void * >(snrData.data()), 0, &syncPoint);
          syncPoint.wait();
          clQueues->at(deviceOptions.deviceID)[0].enqueueReadBuffer(snrSamples_d, CL_TRUE, 0, snrSamples.size() * sizeof(unsigned int), reinterpret_cast< void * >(snrSamples.data()), 0, &syncPoint);
          syncPoint.wait();
          outputCopyTimer.stop();
        } else {
          clQueues->at(deviceOptions.deviceID)[0].enqueueNDRangeKernel(*integrationDMsSamplesK[stepNumber], cl::NullRange, integrationGlobal[stepNumber], integrationLocal[stepNumber]);
          clQueues->at(deviceOptions.deviceID)[0].enqueueNDRangeKernel(*snrDMsSamplesK[stepNumber], cl::NullRange, snrDMsSamplesGlobal[stepNumber], snrDMsSamplesLocal[stepNumber]);
          clQueues->at(deviceOptions.deviceID)[0].enqueueReadBuffer(snrData_d, CL_FALSE, 0, snrData.size() * sizeof(float), reinterpret_cast< void * >(snrData.data()));
          clQueues->at(deviceOptions.deviceID)[0].enqueueReadBuffer(snrSamples_d, CL_FALSE, 0, snrSamples.size() * sizeof(unsigned int), reinterpret_cast< void * >(snrSamples.data()));
          clQueues->at(deviceOptions.deviceID)[0].finish();
        }
      } catch ( cl::Error & err ) {
        std::cerr << "SNR integration loop error -- Batch: " << std::to_string(batch) << ", Step: " << std::to_string(*step) << ", " << err.what() << " " << err.err() << std::endl;
        errorDetected = true;
      }
      if ( DEBUG ) {
        if ( options.print ) {
          try {
            clQueues->at(deviceOptions.deviceID)[0].enqueueReadBuffer(integratedData_d, CL_TRUE, 0, integratedData.size() * sizeof(outputDataType), reinterpret_cast< void * >(integratedData.data()), 0, &syncPoint);
            syncPoint.wait();
          } catch ( cl::Error & err ) {
            std::cerr << "Impossible to read integratedData_d: " << err.what() << " " << err.err() << std::endl;
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
                    std::cerr << integratedData[(sBeam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / *(integrationSteps.begin()), deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / *(integrationSteps.begin()), deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / *(integrationSteps.begin()), deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + sample] << " ";
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
                  std::cerr << integratedData[(sBeam * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / *(integrationSteps.begin()), deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / *(integrationSteps.begin()), deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType))) + sample] << " ";
                }
                std::cerr << std::endl;
              }
              std::cerr << std::endl;
            }
          }
        }
      }
      triggerTimer.start();
      trigger(options.subbandDedispersion, deviceOptions.padding[deviceOptions.deviceName], *step, options.threshold, observation, snrData, snrSamples, triggeredEvents);
      triggerTimer.stop();
      if ( DEBUG ) {
        if ( options.print ) {
          if ( options.subbandDedispersion ) {
            std::cerr << "snrData" << std::endl;
            for ( unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++ ) {
              std::cerr << "sBeam: " << sBeam << std::endl;
              for ( unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++ ) {
                for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
                  std::cerr << snrData[(sBeam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding[deviceOptions.deviceName] / sizeof(float))) + (subbandingDM * observation.getNrDMs()) + dm] << " ";
                }
              }
              std::cerr << std::endl;
            }
          } else {
            std::cerr << "snrData" << std::endl;
            for ( unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++ ) {
              std::cerr << "sBeam: " << sBeam << std::endl;
              for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
                std::cerr << snrData[(sBeam * observation.getNrDMs(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(float))) + dm] << " ";
              }
              std::cerr << std::endl;
            }
          }
          std::cerr << std::endl;
        }
      }
    }
    if ( errorDetected ) {
      output.close();
#ifdef HAVE_PSRDADA
      if ( dataOptions.dataPSRDADA ) {
        if ( dada_hdu_unlock_read(dataOptions.ringBuffer) != 0 ) {
          std::cerr << "Impossible to unlock the PSRDADA ringbuffer for reading the header." << std::endl;
        }
        dada_hdu_disconnect(dataOptions.ringBuffer);
      }
#endif // HAVE_PSRDADA
      return 1;
    }
    // Print and compact results
    triggerTimer.start();
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
    triggerTimer.stop();
  }
#ifdef HAVE_PSRDADA
  if ( dataOptions.dataPSRDADA ) {
    if ( dada_hdu_unlock_read(dataOptions.ringBuffer) != 0 ) {
      std::cerr << "Impossible to unlock the PSRDADA ringbuffer for reading the header." << std::endl;
    }
    dada_hdu_disconnect(dataOptions.ringBuffer);
  }
#endif // HAVE_PSRDADA
  output.close();
  searchTimer.stop();

  // Store statistics before shutting down
  output.open(dataOptions.outputFile + ".stats");
  output << std::fixed << std::setprecision(6);
  output << "# nrDMs" << std::endl;
  if ( options.subbandDedispersion ) {
    output << observation.getNrDMs(true) * observation.getNrDMs() << std::endl;
  } else {
    output << observation.getNrDMs() << std::endl;
  }
  output << "# searchTimer" << std::endl;
  output << searchTimer.getTotalTime() << std::endl;
  output << "# inputHandlingTotal inputHandlingAvg err" << std::endl;
  output << inputHandlingTimer.getTotalTime() << " " << inputHandlingTimer.getAverageTime() << " " << inputHandlingTimer.getStandardDeviation() << std::endl;
  output << "# inputCopyTotal inputCopyAvg err" << std::endl;
  output << inputCopyTimer.getTotalTime() << " " << inputCopyTimer.getAverageTime() << " " << inputCopyTimer.getStandardDeviation() << std::endl;
  output << "# dedispersionTotal dedispersionAvg err" << std::endl;
  output << dedispersionTimer.getTotalTime() << " " << dedispersionTimer.getAverageTime() << " " << dedispersionTimer.getStandardDeviation() << std::endl;
  output << "# dedispersionStepOneTotal dedispersionStepOneAvg err" << std::endl;
  output << dedispersionStepOneTimer.getTotalTime() << " " << dedispersionStepOneTimer.getAverageTime() << " " << dedispersionStepOneTimer.getStandardDeviation() << std::endl;
  output << "# dedispersionStepTwoTotal dedispersionStepTwoAvg err" << std::endl;
  output << dedispersionStepTwoTimer.getTotalTime() << " " << dedispersionStepTwoTimer.getAverageTime() << " " << dedispersionStepTwoTimer.getStandardDeviation() << std::endl;
  output << "# integrationTotal integrationAvg err" << std::endl;
  output << integrationTimer.getTotalTime() << " " << integrationTimer.getAverageTime() << " " << integrationTimer.getStandardDeviation() << std::endl;
  output << "# snrDMsSamplesTotal snrDMsSamplesAvg err" << std::endl;
  output << snrDMsSamplesTimer.getTotalTime() << " " << snrDMsSamplesTimer.getAverageTime() << " " << snrDMsSamplesTimer.getStandardDeviation() << std::endl;
  output << "# outputCopyTotal outputCopyAvg err" << std::endl;
  output << outputCopyTimer.getTotalTime() << " " << outputCopyTimer.getAverageTime() << " " << outputCopyTimer.getStandardDeviation() << std::endl;
  output << "# triggerTimeTotal triggerTimeAvg err" << std::endl;
  output << triggerTimer.getTotalTime() << " " << triggerTimer.getAverageTime() << " " << triggerTimer.getStandardDeviation() << std::endl;
  output.close();

  return 0;
}

