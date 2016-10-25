// Copyright 2015 Alessio Sclocco <a.sclocco@vu.nl>
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
#include <BeamDriver.hpp>

void trigger(const bool compactResults, bool subbandDedispersion, const unsigned int padding, const unsigned int second, const unsigned int integration, const float threshold, const AstroData::Observation & obs, isa::utils::Timer & timer, const std::vector< float > & snrData, std::ofstream & output);


int main(int argc, char * argv[]) {
  bool print = false;
  bool dataLOFAR = false;
  bool dataSIGPROC = false;
  bool dataPSRDada = false;
  bool subbandDedispersion = false;
  bool limit = false;
  bool compactResults = false;
  uint8_t inputBits = 0;
  unsigned int clPlatformID = 0;
  unsigned int clDeviceID = 0;
  unsigned int bytesToSkip = 0;
  float threshold = 0.0f;
  std::string deviceName;
  std::string dataFile;
  std::string headerFile;
  std::string outputFile;
  std::string channelsFile;
  std::string integrationFile;
  std::ofstream output;
  isa::utils::ArgumentList args(argc, argv);
  // Fake single pulse
  bool random = false;
  unsigned int width = 0;
  float DM = 0.0f;
  // Observation object
  AstroData::Observation obs;
  // Configurations
  AstroData::paddingConf padding;
  PulsarSearch::tunedDedispersionConf dedispersionParameters;
  PulsarSearch::tunedDedispersionConf dedispersionStepOneParameters;
  PulsarSearch::tunedDedispersionConf dedispersionStepTwoParameters;
  PulsarSearch::tunedIntegrationConf integrationParameters;
  PulsarSearch::tunedSNRConf snrParameters;
  // PSRDada
  key_t dadaKey;
  dada_hdu_t * ringBuffer;

  try {
    clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
    clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    deviceName = args.getSwitchArgument< std::string >("-device_name");

    subbandDedispersion = args.getSwitch("-subband_dedispersion");

    AstroData::readPaddingConf(padding, args.getSwitchArgument< std::string >("-padding_file"));
    channelsFile = args.getSwitchArgument< std::string >("-zapped_channels");
    integrationFile = args.getSwitchArgument< std::string >("-integration_steps");
    if ( !subbandDedispersion ) {
      PulsarSearch::readTunedDedispersionConf(dedispersionParameters, args.getSwitchArgument< std::string >("-dedispersion_file"));
    } else {
      PulsarSearch::readTunedDedispersionConf(dedispersionStepOneParameters, args.getSwitchArgument< std::string >("-dedispersion_step_one_file"));
      PulsarSearch::readTunedDedispersionConf(dedispersionStepTwoParameters, args.getSwitchArgument< std::string >("-dedispersion_step_two_file"));
    }
    PulsarSearch::readTunedIntegrationConf(integrationParameters, args.getSwitchArgument< std::string >("-integration_file"));
    PulsarSearch::readTunedSNRConf(snrParameters, args.getSwitchArgument< std::string >("-snr_file"));

    compactResults = args.getSwitch("-compact_results");
    print = args.getSwitch("-print");

    dataLOFAR = args.getSwitch("-lofar");
    dataSIGPROC = args.getSwitch("-sigproc");
    dataPSRDada = args.getSwitch("-dada");
    if ( !((((!(dataLOFAR && dataSIGPROC) && dataPSRDada) || (!(dataLOFAR && dataPSRDada) && dataSIGPROC)) || (!(dataSIGPROC && dataPSRDada) && dataLOFAR)) || ((!dataLOFAR && !dataSIGPROC) && !dataPSRDada)) ) {
      std::cerr << "-lofar -sigproc and -dada are mutually exclusive." << std::endl;
      throw std::exception();
    } else if ( dataLOFAR ) {
      obs.setNrBeams(1);
      headerFile = args.getSwitchArgument< std::string >("-header");
      dataFile = args.getSwitchArgument< std::string >("-data");
      limit = args.getSwitch("-limit");
      if ( limit ) {
        obs.setNrSeconds(args.getSwitchArgument< unsigned int >("-seconds"));
      }
    } else if ( dataSIGPROC ) {
      obs.setNrBeams(1);
      bytesToSkip = args.getSwitchArgument< unsigned int >("-header");
      dataFile = args.getSwitchArgument< std::string >("-data");
      obs.setNrSeconds(args.getSwitchArgument< unsigned int >("-seconds"));
      obs.setFrequencyRange(1, args.getSwitchArgument< unsigned int >("-channels"), args.getSwitchArgument< float >("-min_freq"), args.getSwitchArgument< float >("-channel_bandwidth"));
      obs.setNrSamplesPerBatch(args.getSwitchArgument< unsigned int >("-samples"));
    } else if ( dataPSRDada ) {
      dadaKey = args.getSwitchArgument< key_t >("-dada_key");
      obs.setNrBeams(args.getSwitchArgument< unsigned int >("-beams"));
      obs.setNrSeconds(args.getSwitchArgument< unsigned int >("-seconds"));
    } else {
      obs.setNrBeams(args.getSwitchArgument< unsigned int >("-beams"));
      obs.setNrSyntheticBeams(args.getSwitchArgument< unsigned int >("-synthetic_beams"));
      obs.setNrSeconds(args.getSwitchArgument< unsigned int >("-seconds"));
      if ( subbandDedispersion ) {
        obs.setFrequencyRange(args.getSwitchArgument< unsigned int >("-subbands"), args.getSwitchArgument< unsigned int >("-channels"), args.getSwitchArgument< float >("-min_freq"), args.getSwitchArgument< float >("-channel_bandwidth"));
      } else {
        obs.setFrequencyRange(1, args.getSwitchArgument< unsigned int >("-channels"), args.getSwitchArgument< float >("-min_freq"), args.getSwitchArgument< float >("-channel_bandwidth"));
      }
      obs.setNrSamplesPerBatch(args.getSwitchArgument< unsigned int >("-samples"));
      random = args.getSwitch("-random");
      width = args.getSwitchArgument< unsigned int >("-width");
      DM = args.getSwitchArgument< float >("-dm");
    }
    inputBits = args.getSwitchArgument< unsigned int >("-input_bits");
    outputFile = args.getSwitchArgument< std::string >("-output");
    if ( subbandDedispersion ) {
      obs.setDMSubbandingRange(args.getSwitchArgument< unsigned int >("-subbanding_dms"), args.getSwitchArgument< unsigned int >("-subbanding_dm_first"), args.getSwitchArgument< unsigned int >("-subbanding_dm_step"));
    }
    obs.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), args.getSwitchArgument< unsigned int >("-dm_first"), args.getSwitchArgument< unsigned int >("-dm_step"));
    threshold = args.getSwitchArgument< float >("-threshold");
  } catch ( isa::utils::EmptyCommandLine & err ) {
    std::cerr <<  args.getName() << " -opencl_platform ... -opencl_device ... -device_name ... -padding_file ... -zapped_channels ... -integration_steps ... -integration_file ... -snr_file ... [-subband_dedispersion] [-print] [-compact_results] [-lofar] [-sigproc] [-dada] -input_bits ... -output ... -dms ... -dm_first ... -dm_step ... -threshold ..."<< std::endl;
    std::cerr << "\tDedispersion: -dedispersion_file ..." << std::endl;
    std::cerr << "\tSubband Dedispersion: -subband_dedispersion -dedispersion_step_one_file ... -dedispersion_step_two_file ... -subbands ... -subbanding_dms ... -subbanding_dm_first ... -subbanding_dm_step ..." << std::endl;
    std::cerr << "\t -lofar -header ... -data ... [-limit]" << std::endl;
    std::cerr << "\t\t -limit -seconds ..." << std::endl;
    std::cerr << "\t -sigproc -header ... -data ... -seconds ... -channels ... -min_freq ... -channel_bandwidth ... -samples ..." << std::endl;
    std::cerr << "\t -dada -dada_key ... -beams ... -seconds ..." << std::endl;
    std::cerr << "\t [-random] -width ... -dm ... -beams ... -synthetic_beams ... -seconds ... -channels ... -min_freq ... -channel_bandwidth ... -samples ..." << std::endl;
    return 1;
  } catch ( std::exception & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  // Load observation data
  isa::utils::Timer loadTime;
  std::vector< std::vector< std::vector< inputDataType > * > * > input(obs.getNrBeams());
  std::vector< uint8_t > zappedChannels(obs.getNrPaddedChannels(padding[deviceName] / sizeof(uint8_t)));
  std::set< unsigned int > integrationSteps;
  if ( dataLOFAR ) {
    input[0] = new std::vector< std::vector< inputDataType > * >(obs.getNrSeconds());
    loadTime.start();
    if ( limit ) {
      AstroData::readLOFAR(headerFile, dataFile, obs, padding[deviceName], *(input[0]), obs.getNrSeconds());
    } else {
      AstroData::readLOFAR(headerFile, dataFile, obs, padding[deviceName], *(input[0]));
    }
    loadTime.stop();
  } else if ( dataSIGPROC ) {
    input[0] = new std::vector< std::vector< inputDataType > * >(obs.getNrSeconds());
    loadTime.start();
    AstroData::readSIGPROC(obs, padding[deviceName], inputBits, bytesToSkip, dataFile, *(input[0]));
    loadTime.stop();
  } else if ( dataPSRDada ) {
    ringBuffer = dada_hdu_create(0);
    dada_hdu_set_key(ringBuffer, dadaKey);
    dada_hdu_connect(ringBuffer);
    dada_hdu_lock_read(ringBuffer);
  } else {
    for ( unsigned int beam = 0; beam < obs.getNrBeams(); beam++ ) {
      // TODO: if there are multiple synthetic beams, the generated data should take this into account
      input[beam] = new std::vector< std::vector< inputDataType > * >(obs.getNrSeconds());
      AstroData::generateSinglePulse(width, DM, obs, padding[deviceName], *(input[beam]), inputBits, random);
    }
  }
  AstroData::readZappedChannels(obs, channelsFile, zappedChannels);
  AstroData::readIntegrationSteps(obs, integrationFile, integrationSteps);
  if ( DEBUG ) {
    std::cout << "Device: " << deviceName << std::endl;
    std::cout << "Padding: " << padding[deviceName] << " bytes" << std::endl;
    std::cout << std::endl;
    std::cout << "Beams: " << obs.getNrBeams() << std::endl;
    std::cout << "Synthetic Beams: " << obs.getNrSyntheticBeams() << std::endl;
    std::cout << "Seconds: " << obs.getNrSeconds() << std::endl;
    std::cout << "Samples: " << obs.getNrSamplesPerBatch() << std::endl;
    std::cout << "Frequency range: " << obs.getMinFreq() << " MHz, " << obs.getMaxFreq() << " MHz" << std::endl;
    std::cout << "Subbands: " << obs.getNrSubbands() << " (" << obs.getSubbandBandwidth() << " MHz)" << std::endl;
    std::cout << "Channels: " << obs.getNrChannels() << " (" << obs.getChannelBandwidth() << " MHz)" << std::endl;
    std::cout << "Zapped Channels: " << obs.getNrZappedChannels() << std::endl;
    std::cout << "Integration steps: " << integrationSteps.size() << std::endl;
    if ( subbandDedispersion ) {
      std::cout << "Subbanding DMs: " << obs.getNrDMsSubbanding() << " (" << obs.getFirstDMSubbanding() << ", " << obs.getFirstDMSubbanding() + ((obs.getNrDMsSubbanding() - 1) * obs.getDMSubbandingStep()) << ")" << std::endl;
    }
    std::cout << "DMs: " << obs.getNrDMs() << " (" << obs.getFirstDM() << ", " << obs.getFirstDM() + ((obs.getNrDMs() - 1) * obs.getDMStep()) << ")" << std::endl;
    std::cout << std::endl;
    if ( (dataLOFAR || dataSIGPROC) ) {
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
    isa::OpenCL::initializeOpenCL(clPlatformID, obs.getNrBeams(), clPlatforms, clContext, clDevices, clQueues);
  } catch ( isa::OpenCL::OpenCLError & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  // Host memory allocation
  std::vector< float > * shiftsStepOne = PulsarSearch::getShifts(obs, padding[deviceName]);
  std::vector< float > * shiftsStepTwo = PulsarSearch::getSubbandStepTwoShifts(obs, padding[deviceName]);
  if ( subbandDedispersion ) {
    obs.setNrSamplesPerBatchSubbanding(obs.getNrSamplesPerBatch() + static_cast< unsigned int >(shiftsStepTwo->at(0) * (obs.getFirstDM() + ((obs.getNrDMs() - 1) * obs.getDMStep()))));
    obs.setNrSamplesPerSubbandingDispersedChannel(obs.getNrSamplesPerBatchSubbanding() + static_cast< unsigned int >(shiftsStepOne->at(0) * (obs.getFirstDMSubbanding() + ((obs.getNrDMsSubbanding() - 1) * obs.getDMSubbandingStep()))));
    obs.setNrDelaySeconds(static_cast< unsigned int >(std::ceil(static_cast< double >(obs.getNrSamplesPerSubbandingDispersedChannel()) / obs.getNrSamplesPerBatch())));
  } else {
    obs.setNrSamplesPerDispersedChannel(obs.getNrSamplesPerBatch() + static_cast< unsigned int >(shiftsStepOne->at(0) * (obs.getFirstDM() + ((obs.getNrDMs() - 1) * obs.getDMStep()))));
    obs.setNrDelaySeconds(static_cast< unsigned int >(std::ceil(static_cast< double >(obs.getNrSamplesPerDispersedChannel()) / obs.getNrSamplesPerBatch())));
  }
  std::vector< uint8_t > beamDriver;
  std::vector< inputDataType > dispersedData;
  std::vector< outputDataType > subbandedData;
  std::vector< outputDataType > dedispersedData;
  std::vector< outputDataType > integratedData;
  std::vector< float > snrData;

  if ( subbandDedispersion ) {
    if ( dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getSplitSeconds() ) {
      // TODO: add support for splitSeconds
    } else {
      if ( inputBits >= 8 ) {
        dispersedData.resize(obs.getNrBeams() * obs.getNrChannels() * obs.getNrSamplesPerPaddedSubbandingDispersedChannel(padding[deviceName] / sizeof(inputDataType)));
      } else {
        dispersedData.resize(obs.getNrBeams() * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerSubbandingDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType)));
      }
    }
    beamDriver.resize(obs.getNrSyntheticBeams() * obs.getNrPaddedSubbands(padding[deviceName] / sizeof(uint8_t)));
    subbandedData.resize(obs.getNrBeams() * obs.getNrDMsSubbanding() * obs.getNrSubbands() * obs.getNrSamplesPerPaddedBatchSubbanding(padding[deviceName] / sizeof(outputDataType)));
    dedispersedData.resize(obs.getNrSyntheticBeams() * obs.getNrDMsSubbanding() * obs.getNrDMs() * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(outputDataType)));
    integratedData.resize(obs.getNrSyntheticBeams() * obs.getNrDMsSubbanding() * obs.getNrDMs() * isa::utils::pad(obs.getNrSamplesPerBatch() / *(integrationSteps.begin()), padding[deviceName] / sizeof(outputDataType)));
    snrData.resize(obs.getNrSyntheticBeams() * isa::utils::pad(obs.getNrDMsSubbanding() * obs.getNrDMs(), padding[deviceName] / sizeof(float)));
  } else {
    if ( dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitSeconds() ) {
      // TODO: add support for splitSeconds
    } else {
      if ( inputBits >= 8 ) {
        dispersedData.resize(obs.getNrBeams() * obs.getNrChannels() * obs.getNrSamplesPerPaddedDispersedChannel(padding[deviceName] / sizeof(inputDataType)));
      } else {
        dispersedData.resize(obs.getNrBeams() * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType)));
      }
    }
    beamDriver.resize(obs.getNrSyntheticBeams() * obs.getNrPaddedChannels(padding[deviceName] / sizeof(uint8_t)));
    dedispersedData.resize(obs.getNrSyntheticBeams() * obs.getNrDMs() * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(outputDataType)));
    integratedData.resize(obs.getNrSyntheticBeams() * obs.getNrDMs() * isa::utils::pad(obs.getNrSamplesPerBatch() / *(integrationSteps.begin()), padding[deviceName] / sizeof(outputDataType)));
    snrData.resize(obs.getNrSyntheticBeams() * isa::utils::pad(obs.getNrDMs(), padding[deviceName] / sizeof(float)));
  }
  generateBeamDriver(subbandDedispersion, obs, beamDriver, padding[deviceName]);

  if ( obs.getNrDelaySeconds() > obs.getNrSeconds() ) {
    std::cerr << "Not enough input seconds for the search." << std::endl;
    return 1;
  }

  // Device memory allocation and data transfers
  cl::Buffer shiftsStepOne_d;
  cl::Buffer shiftsStepTwo_d;
  cl::Buffer zappedChannels_d;
  cl::Buffer beamDriver_d;
  cl::Buffer dispersedData_d;
  cl::Buffer subbandedData_d;
  cl::Buffer dedispersedData_d;
  cl::Buffer integratedData_d;
  cl::Buffer snrData_d;

  try {
    shiftsStepOne_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, shiftsStepOne->size() * sizeof(float), 0, 0);
    zappedChannels_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, zappedChannels.size() * sizeof(uint8_t), 0, 0);
    beamDriver_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, beamDriver.size() * sizeof(uint8_t), 0, 0);
    dispersedData_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, dispersedData.size() * sizeof(inputDataType), 0, 0);
    dedispersedData_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, dedispersedData.size() * sizeof(outputDataType), 0, 0);
    integratedData_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, integratedData.size() * sizeof(outputDataType), 0, 0);
    snrData_d = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, snrData.size() * sizeof(float), 0, 0);
    if ( subbandDedispersion ) {
      shiftsStepTwo_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, shiftsStepTwo->size() * sizeof(float), 0, 0);
      subbandedData_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, subbandedData.size() * sizeof(outputDataType), 0, 0);
    }
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(shiftsStepOne_d, CL_FALSE, 0, shiftsStepOne->size() * sizeof(float), reinterpret_cast< void * >(shiftsStepOne->data()));
    if ( subbandDedispersion ) {
      clQueues->at(clDeviceID)[0].enqueueWriteBuffer(shiftsStepTwo_d, CL_FALSE, 0, shiftsStepTwo->size() * sizeof(float), reinterpret_cast< void * >(shiftsStepTwo->data()));
    }
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(beamDriver_d, CL_FALSE, 0, beamDriver.size() * sizeof(uint8_t), reinterpret_cast< void * >(beamDriver.data()));
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(zappedChannels_d, CL_FALSE, 0, zappedChannels.size() * sizeof(uint8_t), reinterpret_cast< void * >(zappedChannels.data()));
    clQueues->at(clDeviceID)[0].finish();
  } catch ( cl::Error & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  // Generate OpenCL kernels
  std::string * code;
  cl::Kernel * dedispersionK, * dedispersionStepOneK, * dedispersionStepTwoK;
  std::vector< cl::Kernel * > integrationDMsSamplesK(integrationSteps.size() + 1), snrDMsSamplesK(integrationSteps.size() + 1);

  if ( subbandDedispersion ) {
    code = PulsarSearch::getSubbandDedispersionStepOneOpenCL< inputDataType, outputDataType >(*(dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())), padding[deviceName], inputBits, inputDataName, intermediateDataName, outputDataName, obs, *shiftsStepOne);
    try {
      dedispersionStepOneK = isa::OpenCL::compile("dedispersionStepOne", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
      if ( dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMs())->getSplitSeconds() ) {
        // TODO: add support for splitSeconds
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
    code = PulsarSearch::getSubbandDedispersionStepTwoOpenCL< inputDataType >(*(dedispersionStepTwoParameters.at(deviceName)->at(obs.getNrDMs())), padding[deviceName], inputDataName, obs, *shiftsStepTwo);
    try {
      dedispersionStepTwoK = isa::OpenCL::compile("dedispersionStepTwo", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
      dedispersionStepTwoK->setArg(0, subbandedData_d);
      dedispersionStepTwoK->setArg(1, dedispersedData_d);
      dedispersionStepTwoK->setArg(2, beamDriver_d);
      dedispersionStepTwoK->setArg(3, shiftsStepTwo_d);
    } catch ( isa::OpenCL::OpenCLError & err ) {
      std::cerr << err.what() << std::endl;
      return 1;
    }
    delete code;
  } else {
    code = PulsarSearch::getDedispersionOpenCL< inputDataType, outputDataType >(*(dedispersionParameters.at(deviceName)->at(obs.getNrDMs())), padding[deviceName], inputBits, inputDataName, intermediateDataName, outputDataName, obs, *shiftsStepOne, zappedChannels);
    try {
      dedispersionK = isa::OpenCL::compile("dedispersion", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
      if ( dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitSeconds() ) {
        // TODO: add support for splitSeconds
      } else {
        dedispersionK->setArg(0, dispersedData_d);
        dedispersionK->setArg(1, dedispersedData_d);
        dedispersionK->setArg(2, zappedChannels_d);
        dedispersionK->setArg(3, shiftsStepOne_d);
      }
    } catch ( isa::OpenCL::OpenCLError & err ) {
      std::cerr << err.what() << std::endl;
      return 1;
    }
    delete code;
  }
  if ( subbandDedispersion ) {
    code = PulsarSearch::getSNRDMsSamplesOpenCL< outputDataType >(*(snrParameters.at(deviceName)->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(obs.getNrSamplesPerBatch())), outputDataName, obs, obs.getNrSamplesPerBatch(), padding[deviceName]);
  } else {
    code = PulsarSearch::getSNRDMsSamplesOpenCL< outputDataType >(*(snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerBatch())), outputDataName, obs, obs.getNrSamplesPerBatch(), padding[deviceName]);
  }
  try {
    snrDMsSamplesK[integrationSteps.size()] = isa::OpenCL::compile("snrDMsSamples" + std::to_string(obs.getNrSamplesPerBatch()), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
    snrDMsSamplesK[integrationSteps.size()]->setArg(0, dedispersedData_d);
    snrDMsSamplesK[integrationSteps.size()]->setArg(1, snrData_d);
  } catch ( isa::OpenCL::OpenCLError & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }
  delete code;
  for ( unsigned int stepNumber = 0; stepNumber < integrationSteps.size(); stepNumber++ ) {
    auto step = integrationSteps.begin();

    std::advance(step, stepNumber);
    if ( subbandDedispersion ) {
      code = PulsarSearch::getIntegrationDMsSamplesOpenCL< outputDataType >(*(integrationParameters[deviceName]->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(*step)), obs, outputDataName, *step, padding[deviceName]);
    } else {
      code = PulsarSearch::getIntegrationDMsSamplesOpenCL< outputDataType >(*(integrationParameters[deviceName]->at(obs.getNrDMs())->at(*step)), obs, outputDataName, *step, padding[deviceName]);
    }
    try {
      if ( *step > 1 ) {
        integrationDMsSamplesK[stepNumber] = isa::OpenCL::compile("integrationDMsSamples" + std::to_string(*step), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
        integrationDMsSamplesK[stepNumber]->setArg(0, dedispersedData_d);
        integrationDMsSamplesK[stepNumber]->setArg(1, integratedData_d);
      }
    } catch ( isa::OpenCL::OpenCLError & err ) {
      std::cerr << err.what() << std::endl;
      return 1;
    }
    delete code;
    if ( subbandDedispersion ) {
      code = PulsarSearch::getSNRDMsSamplesOpenCL< outputDataType >(*(snrParameters.at(deviceName)->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(obs.getNrSamplesPerBatch() / *step)), outputDataName, obs, obs.getNrSamplesPerBatch() / *step, padding[deviceName]);
    } else {
      code = PulsarSearch::getSNRDMsSamplesOpenCL< outputDataType >(*(snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerBatch() / *step)), outputDataName, obs, obs.getNrSamplesPerBatch() / *step, padding[deviceName]);
    }
    try {
      snrDMsSamplesK[stepNumber] = isa::OpenCL::compile("snrDMsSamples" + std::to_string(obs.getNrSamplesPerBatch() / *step), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
      snrDMsSamplesK[stepNumber]->setArg(0, integratedData_d);
      snrDMsSamplesK[stepNumber]->setArg(1, snrData_d);
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
  if ( subbandDedispersion ) {
    dedispersionStepOneGlobal = cl::NDRange(isa::utils::pad(obs.getNrSamplesPerBatchSubbanding() / dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getNrItemsD0(), dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getNrThreadsD0()), obs.getNrDMsSubbanding() / dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getNrItemsD1(), obs.getNrBeams() * obs.getNrSubbands());
    dedispersionStepOneLocal = cl::NDRange(dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getNrThreadsD0(), dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getNrThreadsD1(), 1);
    if ( DEBUG ) {
      std::cout << "DedispersionStepOne" << std::endl;
        std::cout << "Global: " << isa::utils::pad(obs.getNrSamplesPerBatchSubbanding() / dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getNrItemsD0(), dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getNrThreadsD0()) << ", " << obs.getNrDMsSubbanding() / dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getNrItemsD1() << ", " << obs.getNrBeams() * obs.getNrSubbands() << std::endl;
        std::cout << "Local: " << dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getNrThreadsD0() << ", " << dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getNrThreadsD1() << ", 1" << std::endl;
        std::cout << "Parameters: ";
        std::cout << dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->print() << std::endl;
        std::cout << std::endl;
    }
    dedispersionStepTwoGlobal = cl::NDRange(isa::utils::pad(obs.getNrSamplesPerBatchSubbanding() / dedispersionStepTwoParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getNrItemsD0(), dedispersionStepTwoParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getNrThreadsD0()), obs.getNrDMsSubbanding() / dedispersionStepTwoParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getNrItemsD1(), obs.getNrBeams() * obs.getNrSubbands());
    dedispersionStepTwoLocal = cl::NDRange(dedispersionStepTwoParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getNrThreadsD0(), dedispersionStepTwoParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getNrThreadsD1(), 1);
    if ( DEBUG ) {
      std::cout << "DedispersionStepTwo" << std::endl;
        std::cout << "Global: " << isa::utils::pad(obs.getNrSamplesPerBatch() / dedispersionStepTwoParameters.at(deviceName)->at(obs.getNrDMs())->getNrItemsD0(), dedispersionStepTwoParameters.at(deviceName)->at(obs.getNrDMs())->getNrThreadsD0()) << ", " << obs.getNrDMs() / dedispersionStepTwoParameters.at(deviceName)->at(obs.getNrDMs())->getNrItemsD1() << ", " << obs.getNrSyntheticBeams() * obs.getNrDMsSubbanding() << std::endl;
        std::cout << "Local: " << dedispersionStepTwoParameters.at(deviceName)->at(obs.getNrDMs())->getNrThreadsD0() << ", " << dedispersionStepTwoParameters.at(deviceName)->at(obs.getNrDMs())->getNrThreadsD1() << ", 1" << std::endl;
        std::cout << "Parameters: ";
        std::cout << dedispersionStepTwoParameters.at(deviceName)->at(obs.getNrDMs())->print() << std::endl;
        std::cout << std::endl;
    }
  } else {
    dedispersionGlobal = cl::NDRange(isa::utils::pad(obs.getNrSamplesPerBatch() / dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrItemsD0(), dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrThreadsD0()), obs.getNrDMs() / dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrItemsD1(), obs.getNrSyntheticBeams());
    dedispersionLocal = cl::NDRange(dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrThreadsD0(), dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrThreadsD1(), 1);
    if ( DEBUG ) {
      std::cout << "Dedispersion" << std::endl;
      std::cout << "Global: " << isa::utils::pad(obs.getNrSamplesPerBatch() / dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrItemsD0(), dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrThreadsD0()) << ", " << obs.getNrDMs() / dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrItemsD1() << ", " << obs.getNrSyntheticBeams() << std::endl;
      std::cout << "Local: " << dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrThreadsD0() << ", " << dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrThreadsD1() << ", 1" << std::endl;
      std::cout << "Parameters: ";
      std::cout << dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->print() << std::endl;
      std::cout << std::endl;
    }
  }
  std::vector< cl::NDRange > integrationGlobal(integrationSteps.size());
  std::vector< cl::NDRange > integrationLocal(integrationSteps.size());
  std::vector< cl::NDRange > snrDMsSamplesGlobal(integrationSteps.size() + 1);
  std::vector< cl::NDRange > snrDMsSamplesLocal(integrationSteps.size() + 1);
  if ( subbandDedispersion ) {
    snrDMsSamplesGlobal[integrationSteps.size()] = cl::NDRange(snrParameters.at(deviceName)->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(obs.getNrSamplesPerBatch())->getNrThreadsD0(), obs.getNrDMsSubbanding() * obs.getNrDMs(), obs.getNrSyntheticBeams());
    snrDMsSamplesLocal[integrationSteps.size()] = cl::NDRange(snrParameters.at(deviceName)->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(obs.getNrSamplesPerBatch())->getNrThreadsD0(), 1, 1);
    if ( DEBUG ) {
      std::cout << "SNRDMsSamples (" + std::to_string(obs.getNrSamplesPerBatch()) + ")" << std::endl;
      std::cout << "Global: " << snrParameters.at(deviceName)->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(obs.getNrSamplesPerBatch())->getNrThreadsD0() << ", " << obs.getNrDMs() << std::endl;
      std::cout << "Local: " << snrParameters.at(deviceName)->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(obs.getNrSamplesPerBatch())->getNrThreadsD0() << ", 1" << std::endl;
      std::cout << "Parameters: ";
      std::cout << snrParameters.at(deviceName)->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(obs.getNrSamplesPerBatch())->print() << std::endl;
      std::cout << std::endl;
    }
  } else {
    snrDMsSamplesGlobal[integrationSteps.size()] = cl::NDRange(snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerBatch())->getNrThreadsD0(), obs.getNrDMs(), obs.getNrSyntheticBeams());
    snrDMsSamplesLocal[integrationSteps.size()] = cl::NDRange(snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerBatch())->getNrThreadsD0(), 1, 1);
    if ( DEBUG ) {
      std::cout << "SNRDMsSamples (" + std::to_string(obs.getNrSamplesPerBatch()) + ")" << std::endl;
      std::cout << "Global: " << snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerBatch())->getNrThreadsD0() << ", " << obs.getNrDMs() << std::endl;
      std::cout << "Local: " << snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerBatch())->getNrThreadsD0() << ", 1" << std::endl;
      std::cout << "Parameters: ";
      std::cout << snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerBatch())->print() << std::endl;
      std::cout << std::endl;
    }
  }
  for ( unsigned int stepNumber = 0; stepNumber < integrationSteps.size(); stepNumber++ ) {
    auto step = integrationSteps.begin();

    std::advance(step, stepNumber);
    if ( subbandDedispersion ) {
      integrationGlobal[stepNumber] = cl::NDRange(integrationParameters[deviceName]->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(*step)->getNrThreadsD0() * ((obs.getNrSamplesPerBatch() / *step) / integrationParameters[deviceName]->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(*step)->getNrItemsD0()), obs.getNrDMsSubbanding() * obs.getNrDMs(), obs.getNrSyntheticBeams());
      integrationLocal[stepNumber] = cl::NDRange(integrationParameters[deviceName]->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(*step)->getNrThreadsD0(), 1, 1);
      if ( DEBUG ) {
        std::cout << "integrationDMsSamples (" + std::to_string(*step) + ")" << std::endl;
        std::cout << "Global: " << integrationParameters[deviceName]->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(*step)->getNrThreadsD0() * ((obs.getNrSamplesPerBatch() / *step) / integrationParameters[deviceName]->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(*step)->getNrItemsD0()) << ", " << obs.getNrDMsSubbanding() * obs.getNrDMs() << std::endl;
        std::cout << "Local: " << integrationParameters[deviceName]->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(*step)->getNrThreadsD0() << ", 1" << std::endl;
        std::cout << "Parameters: ";
        std::cout << integrationParameters[deviceName]->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(*step)->print() << std::endl;
        std::cout << std::endl;
      }
      snrDMsSamplesGlobal[stepNumber] = cl::NDRange(snrParameters.at(deviceName)->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(obs.getNrSamplesPerBatch() / *step)->getNrThreadsD0(), obs.getNrDMsSubbanding() * obs.getNrDMs(), obs.getNrSyntheticBeams());
      snrDMsSamplesLocal[stepNumber] = cl::NDRange(snrParameters.at(deviceName)->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(obs.getNrSamplesPerBatch() / *step)->getNrThreadsD0(), 1, 1);
      if ( DEBUG ) {
        std::cout << "SNRDMsSamples (" + std::to_string(obs.getNrSamplesPerBatch() / *step) + ")" << std::endl;
        std::cout << "Global: " << snrParameters.at(deviceName)->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(obs.getNrSamplesPerBatch() / *step)->getNrThreadsD0() << ", " << obs.getNrDMsSubbanding() * obs.getNrDMs() << std::endl;
        std::cout << "Local: " << snrParameters.at(deviceName)->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(obs.getNrSamplesPerBatch() / *step)->getNrThreadsD0() << ", 1" << std::endl;
        std::cout << "Parameters: ";
        std::cout << snrParameters.at(deviceName)->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(obs.getNrSamplesPerBatch() / *step)->print() << std::endl;
        std::cout << std::endl;
      }
    } else {
      integrationGlobal[stepNumber] = cl::NDRange(integrationParameters[deviceName]->at(obs.getNrDMs())->at(*step)->getNrThreadsD0() * ((obs.getNrSamplesPerBatch() / *step) / integrationParameters[deviceName]->at(obs.getNrDMs())->at(*step)->getNrItemsD0()), obs.getNrDMs(), obs.getNrSyntheticBeams());
      integrationLocal[stepNumber] = cl::NDRange(integrationParameters[deviceName]->at(obs.getNrDMs())->at(*step)->getNrThreadsD0(), 1, 1);
      if ( DEBUG ) {
        std::cout << "integrationDMsSamples (" + std::to_string(*step) + ")" << std::endl;
        std::cout << "Global: " << integrationParameters[deviceName]->at(obs.getNrDMs())->at(*step)->getNrThreadsD0() * ((obs.getNrSamplesPerBatch() / *step) / integrationParameters[deviceName]->at(obs.getNrDMs())->at(*step)->getNrItemsD0()) << ", " << obs.getNrDMs() << std::endl;
        std::cout << "Local: " << integrationParameters[deviceName]->at(obs.getNrDMs())->at(*step)->getNrThreadsD0() << ", 1" << std::endl;
        std::cout << "Parameters: ";
        std::cout << integrationParameters[deviceName]->at(obs.getNrDMs())->at(*step)->print() << std::endl;
        std::cout << std::endl;
      }
      snrDMsSamplesGlobal[stepNumber] = cl::NDRange(snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerBatch() / *step)->getNrThreadsD0(), obs.getNrDMs(), obs.getNrSyntheticBeams());
      snrDMsSamplesLocal[stepNumber] = cl::NDRange(snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerBatch() / *step)->getNrThreadsD0(), 1, 1);
      if ( DEBUG ) {
        std::cout << "SNRDMsSamples (" + std::to_string(obs.getNrSamplesPerBatch() / *step) + ")" << std::endl;
        std::cout << "Global: " << snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerBatch() / *step)->getNrThreadsD0() << ", " << obs.getNrDMs() << std::endl;
        std::cout << "Local: " << snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerBatch() / *step)->getNrThreadsD0() << ", 1" << std::endl;
        std::cout << "Parameters: ";
        std::cout << snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerBatch() / *step)->print() << std::endl;
        std::cout << std::endl;
      }
    }
  }

  // Search loop
  bool errorDetected = false;
  cl::Event syncPoint;
  isa::utils::Timer searchTimer, inputHandlingTimer, inputCopyTimer, dedispersionTimer, dedispersionStepOneTimer, dedispersionStepTwoTimer, integrationTimer, snrDMsSamplesTimer, outputCopyTimer, triggerTimer;

  searchTimer.start();
  output.open(outputFile + ".trigger");
  output << "# second beam integration_step DM SNR" << std::endl;
  for ( unsigned int second = 0; second < obs.getNrSeconds() - obs.getNrDelaySeconds(); second++ ) {
    // Load the input
    inputHandlingTimer.start();
    for ( unsigned int beam = 0; beam < obs.getNrBeams(); beam++ ) {
      if ( subbandDedispersion ) {
        if ( !dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getSplitSeconds() ) {
          for ( unsigned int channel = 0; channel < obs.getNrChannels(); channel++ ) {
            for ( unsigned int chunk = 0; chunk < obs.getNrDelaySeconds() - 1; chunk++ ) {
              if ( inputBits >= 8 ) {
                memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * obs.getNrChannels() * obs.getNrSamplesPerPaddedSubbandingDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + (channel * obs.getNrSamplesPerPaddedSubbandingDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + (chunk * obs.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&((input[beam]->at(second + chunk))->at(channel * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(inputDataType))))), obs.getNrSamplesPerBatch() * sizeof(inputDataType));
              } else {
                memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerSubbandingDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(obs.getNrSamplesPerSubbandingDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + (chunk * (obs.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&((input[beam]->at(second + chunk))->at(channel * isa::utils::pad(obs.getNrSamplesPerBatch() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))))), (obs.getNrSamplesPerBatch() / (8 / inputBits)) * sizeof(inputDataType));
              }
            }
            if ( inputBits >= 8 ) {
              memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * obs.getNrChannels() * obs.getNrSamplesPerPaddedSubbandingDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + (channel * obs.getNrSamplesPerPaddedSubbandingDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + ((obs.getNrDelaySeconds() - 1) * obs.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&((input[beam]->at(second + (obs.getNrDelaySeconds() - 1)))->at(channel * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(inputDataType))))), (obs.getNrSamplesPerSubbandingDispersedChannel() % obs.getNrSamplesPerBatch()) * sizeof(inputDataType));
            } else {
              memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerSubbandingDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(obs.getNrSamplesPerSubbandingDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + ((obs.getNrDelaySeconds() - 1) * (obs.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&((input[beam]->at(second + (obs.getNrDelaySeconds() - 1)))->at(channel * isa::utils::pad(obs.getNrSamplesPerBatch() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))))), ((obs.getNrSamplesPerSubbandingDispersedChannel() % obs.getNrSamplesPerBatch()) / (8 / inputBits)) * sizeof(inputDataType));
            }
          }
        }
      } else {
        if ( !dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitSeconds() ) {
          for ( unsigned int channel = 0; channel < obs.getNrChannels(); channel++ ) {
            for ( unsigned int chunk = 0; chunk < obs.getNrDelaySeconds() - 1; chunk++ ) {
              if ( inputBits >= 8 ) {
                memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * obs.getNrChannels() * obs.getNrSamplesPerPaddedDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + (channel * obs.getNrSamplesPerPaddedDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + (chunk * obs.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&((input[beam]->at(second + chunk))->at(channel * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(inputDataType))))), obs.getNrSamplesPerBatch() * sizeof(inputDataType));
              } else {
                memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(obs.getNrSamplesPerDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + (chunk * (obs.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&((input[beam]->at(second + chunk))->at(channel * isa::utils::pad(obs.getNrSamplesPerBatch() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))))), (obs.getNrSamplesPerBatch() / (8 / inputBits)) * sizeof(inputDataType));
              }
            }
            if ( inputBits >= 8 ) {
              memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * obs.getNrChannels() * obs.getNrSamplesPerPaddedDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + (channel * obs.getNrSamplesPerPaddedDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + ((obs.getNrDelaySeconds() - 1) * obs.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&((input[beam]->at(second + (obs.getNrDelaySeconds() - 1)))->at(channel * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(inputDataType))))), (obs.getNrSamplesPerDispersedChannel() % obs.getNrSamplesPerBatch()) * sizeof(inputDataType));
            } else {
              memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(obs.getNrSamplesPerDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + ((obs.getNrDelaySeconds() - 1) * (obs.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&((input[beam]->at(second + (obs.getNrDelaySeconds() - 1)))->at(channel * isa::utils::pad(obs.getNrSamplesPerBatch() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))))), ((obs.getNrSamplesPerDispersedChannel() % obs.getNrSamplesPerBatch()) / (8 / inputBits)) * sizeof(inputDataType));
            }
          }
        }
      }
    }
    inputHandlingTimer.stop();
    try {
      if ( SYNC ) {
        inputCopyTimer.start();
        if ( subbandDedispersion ) {
          if ( dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getSplitSeconds() ) {
            // TODO: add support for splitSeconds
          } else {
            clQueues->at(clDeviceID)[0].enqueueWriteBuffer(dispersedData_d, CL_TRUE, 0, dispersedData.size() * sizeof(inputDataType), reinterpret_cast< void * >(dispersedData.data()), 0, &syncPoint);
          }
        } else {
          if ( dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitSeconds() ) {
            // TODO: add support for splitSeconds
          } else {
            clQueues->at(clDeviceID)[0].enqueueWriteBuffer(dispersedData_d, CL_TRUE, 0, dispersedData.size() * sizeof(inputDataType), reinterpret_cast< void * >(dispersedData.data()), 0, &syncPoint);
          }
        }
        syncPoint.wait();
        inputCopyTimer.stop();
      } else {
        if ( subbandDedispersion ) {
          if ( dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getSplitSeconds() ) {
            // TODO: add support for splitSeconds
          } else {
            clQueues->at(clDeviceID)[0].enqueueWriteBuffer(dispersedData_d, CL_FALSE, 0, dispersedData.size() * sizeof(inputDataType), reinterpret_cast< void * >(dispersedData.data()));
          }
        } else {
          if ( dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitSeconds() ) {
            // TODO: add support for splitSeconds
          } else {
            clQueues->at(clDeviceID)[0].enqueueWriteBuffer(dispersedData_d, CL_FALSE, 0, dispersedData.size() * sizeof(inputDataType), reinterpret_cast< void * >(dispersedData.data()));
          }
        }
      }
      if ( DEBUG ) {
        if ( print ) {
          // TODO: add support for printing dispersedData to std::cerr
        }
      }
    } catch ( cl::Error & err ) {
      std::cerr << "Input copy error -- Second: " << std::to_string(second) << ", " << err.what() << " " << err.err() << std::endl;
      errorDetected = true;
    }
    if ( subbandDedispersion ) {
      if ( dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getSplitSeconds() && (second < obs.getNrDelaySeconds()) ) {
        // Not enough seconds in the buffer to start the search
        continue;
      }
    } else {
      if ( dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitSeconds() && (second < obs.getNrDelaySeconds()) ) {
        // Not enough seconds in the buffer to start the search
        continue;
      }
    }

    // Dedispersion
    try {
      if ( subbandDedispersion ) {
        if ( dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getSplitSeconds() ) {
          // TODO: add support for splitSeconds
        }
        if ( SYNC ) {
          dedispersionTimer.start();
          dedispersionStepOneTimer.start();
          clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*dedispersionStepOneK, cl::NullRange, dedispersionStepOneGlobal, dedispersionStepOneLocal, 0, &syncPoint);
          syncPoint.wait();
          dedispersionStepOneTimer.stop();
          dedispersionStepTwoTimer.start();
          clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*dedispersionStepTwoK, cl::NullRange, dedispersionStepTwoGlobal, dedispersionStepTwoLocal, 0, &syncPoint);
          syncPoint.wait();
          dedispersionStepTwoTimer.stop();
          dedispersionTimer.stop();
        } else {
          clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*dedispersionStepOneK, cl::NullRange, dedispersionStepOneGlobal, dedispersionStepOneLocal, 0, 0);
          clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*dedispersionStepTwoK, cl::NullRange, dedispersionStepTwoGlobal, dedispersionStepTwoLocal, 0, 0);
        }
      } else {
        if ( dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitSeconds() ) {
          // TODO: add support for splitSeconds
        }
        if ( SYNC ) {
          dedispersionTimer.start();
          clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*dedispersionK, cl::NullRange, dedispersionGlobal, dedispersionLocal, 0, &syncPoint);
          syncPoint.wait();
          dedispersionTimer.stop();
        } else {
          clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*dedispersionK, cl::NullRange, dedispersionGlobal, dedispersionLocal);
        }
      }
    } catch ( cl::Error & err ) {
      std::cerr << "Dedispersion error -- Second: " << std::to_string(second) << ", " << err.what() << " " << err.err() << std::endl;
      errorDetected = true;
    }
    if ( DEBUG ) {
      if ( print ) {
        // TODO: add support for printing dispersedData to std::cerr
      }
    }

    // SNR of dedispersed data
    try {
      if ( SYNC ) {
        snrDMsSamplesTimer.start();
        clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*snrDMsSamplesK[integrationSteps.size()], cl::NullRange, snrDMsSamplesGlobal[integrationSteps.size()], snrDMsSamplesLocal[integrationSteps.size()], 0, &syncPoint);
        syncPoint.wait();
        snrDMsSamplesTimer.stop();
        outputCopyTimer.start();
        clQueues->at(clDeviceID)[0].enqueueReadBuffer(snrData_d, CL_TRUE, 0, snrData.size() * sizeof(float), reinterpret_cast< void * >(snrData.data()), 0, &syncPoint);
        syncPoint.wait();
        outputCopyTimer.stop();
      } else {
        clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*snrDMsSamplesK[integrationSteps.size()], cl::NullRange, snrDMsSamplesGlobal[integrationSteps.size()], snrDMsSamplesLocal[integrationSteps.size()]);
        clQueues->at(clDeviceID)[0].enqueueReadBuffer(snrData_d, CL_FALSE, 0, snrData.size() * sizeof(float), reinterpret_cast< void * >(snrData.data()));
        clQueues->at(clDeviceID)[0].finish();
      }
    } catch ( cl::Error & err ) {
      std::cerr << "SNR dedispersed data error -- Second: " << std::to_string(second) << ", " << err.what() << " " << err.err() << std::endl;
      errorDetected = true;
    }
    trigger(compactResults, subbandDedispersion, padding[deviceName], second, 0, threshold, obs, triggerTimer, snrData, output);
    if ( DEBUG ) {
      if ( print ) {
        // TODO: add support for printing dispersedData to std::cerr
      }
    }

    // Integration and SNR loop
    for ( unsigned int stepNumber = 0; stepNumber < integrationSteps.size(); stepNumber++ ) {
      auto step = integrationSteps.begin();

      std::advance(step, stepNumber);
      try {
        if ( SYNC ) {
          integrationTimer.start();
          clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*integrationDMsSamplesK[stepNumber], cl::NullRange, integrationGlobal[stepNumber], integrationLocal[stepNumber], 0, &syncPoint);
          syncPoint.wait();
          integrationTimer.stop();
          snrDMsSamplesTimer.start();
          clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*snrDMsSamplesK[stepNumber], cl::NullRange, snrDMsSamplesGlobal[stepNumber], snrDMsSamplesLocal[stepNumber], 0, &syncPoint);
          syncPoint.wait();
          snrDMsSamplesTimer.stop();
          outputCopyTimer.start();
          clQueues->at(clDeviceID)[0].enqueueReadBuffer(snrData_d, CL_TRUE, 0, snrData.size() * sizeof(float), reinterpret_cast< void * >(snrData.data()), 0, &syncPoint);
          syncPoint.wait();
          outputCopyTimer.stop();
        } else {
          clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*integrationDMsSamplesK[stepNumber], cl::NullRange, integrationGlobal[stepNumber], integrationLocal[stepNumber]);
          clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*snrDMsSamplesK[stepNumber], cl::NullRange, snrDMsSamplesGlobal[stepNumber], snrDMsSamplesLocal[stepNumber]);
          clQueues->at(clDeviceID)[0].enqueueReadBuffer(snrData_d, CL_FALSE, 0, snrData.size() * sizeof(float), reinterpret_cast< void * >(snrData.data()));
          clQueues->at(clDeviceID)[0].finish();
        }
      } catch ( cl::Error & err ) {
        std::cerr << "SNR integration loop error -- Second: " << std::to_string(second) << ", Step: " << std::to_string(*step) << ", " << err.what() << " " << err.err() << std::endl;
        errorDetected = true;
      }
      if ( DEBUG ) {
        if ( print ) {
          // TODO: add support for printing dispersedData to std::cerr
        }
      }
      trigger(compactResults, subbandDedispersion, padding[deviceName], second, *step, threshold, obs, triggerTimer, snrData, output);
      if ( DEBUG ) {
        if ( print ) {
          // TODO: add support for printing dispersedData to std::cerr
        }
      }
    }
    if ( errorDetected ) {
      output.close();
      return 1;
    }
  }
  if ( dataPSRDada ) {
    dada_hdu_unlock_read(ringBuffer);
    dada_hdu_disconnect(ringBuffer);
  }
  output.close();
  searchTimer.stop();

  // Store statistics before shutting down
  output.open(outputFile + ".stats");
  output << std::fixed << std::setprecision(6);
  output << "# nrDMs" << std::endl;
  output << obs.getNrDMs() << std::endl;
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

void trigger(const bool compactResults, bool subbandDedispersion, const unsigned int padding, const unsigned int second, const unsigned int integration, const float threshold, const AstroData::Observation & obs, isa::utils::Timer & timer, const std::vector< float > & snrData, std::ofstream & output) {
  bool previous = false;
  unsigned int nrDMs = 0;
  float firstDM = 0.0f;

  if ( subbandDedispersion ) {
    nrDMs = obs.getNrDMsSubbanding() * obs.getNrDMs();
    firstDM = obs.getFirstDMSubbanding();
  } else {
    nrDMs = obs.getNrDMs();
    firstDM = obs.getFirstDM();
  }
  timer.start();
  for ( unsigned int beam = 0; beam < obs.getNrSyntheticBeams(); beam++ ) {
    unsigned int maxDM = 0;
    double maxSNR = 0.0;

    for ( unsigned int dm = 0; dm < nrDMs; dm++ ) {
      if ( compactResults ) {
        if ( snrData[(beam * isa::utils::pad(nrDMs, padding / sizeof(float))) + dm] >= threshold ) {
          if ( previous ) {
            if ( snrData[(beam * isa::utils::pad(nrDMs, padding / sizeof(float))) + dm] > maxSNR ) {
              maxDM = dm;
              maxSNR = snrData[(beam * isa::utils::pad(nrDMs, padding / sizeof(float))) + dm];
            }
          } else {
            previous = true;
            maxDM = dm;
            maxSNR = snrData[(beam * isa::utils::pad(nrDMs, padding / sizeof(float))) + dm];
          }
        } else if ( previous ) {
          output << second << " " << beam << " " << firstDM + (maxDM * obs.getDMStep()) << " " << maxSNR << std::endl;
          previous = false;
          maxDM = 0;
          maxSNR = 0;
        }
      } else {
        if ( snrData[(beam * isa::utils::pad(nrDMs, padding / sizeof(float))) + dm] >= threshold ) {
          output << second << " " << beam << " " << integration << " " << firstDM + (dm * obs.getDMStep()) << " " << snrData[(beam * isa::utils::pad(nrDMs, padding / sizeof(float))) + dm] << std::endl;
        }
      }
    }
    if ( previous ) {
      output << second << " " << beam << " " << integration << " " << firstDM + (maxDM * obs.getDMStep()) << " " << maxSNR << std::endl;
    }
  }
  timer.stop();
}

