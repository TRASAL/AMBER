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
#include <Trigger.hpp>


int main(int argc, char * argv[]) {
  bool print = false;
  bool dataLOFAR = false;
  bool dataSIGPROC = false;
  bool dataPSRDADA = false;
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
#ifdef HAVE_PSRDADA
  // PSRDADA
  key_t dadaKey;
  dada_hdu_t * ringBuffer;
#endif

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
    dataPSRDADA = args.getSwitch("-dada");
#ifdef HAVE_PSRDADA
#else
    if (dataPSRDADA) {
      std::cerr << "Not compiled with PSRDADA support" << std::endl;
      throw std::exception();
    };
#endif
    if ( !((((!(dataLOFAR && dataSIGPROC) && dataPSRDADA) || (!(dataLOFAR && dataPSRDADA) && dataSIGPROC)) || (!(dataSIGPROC && dataPSRDADA) && dataLOFAR)) || ((!dataLOFAR && !dataSIGPROC) && !dataPSRDADA)) ) {
      std::cerr << "-lofar -sigproc and -dada are mutually exclusive." << std::endl;
      throw std::exception();
    } else if ( dataLOFAR ) {
      obs.setNrBeams(1);
      headerFile = args.getSwitchArgument< std::string >("-header");
      dataFile = args.getSwitchArgument< std::string >("-data");
      limit = args.getSwitch("-limit");
      if ( limit ) {
        obs.setNrBatches(args.getSwitchArgument< unsigned int >("-batches"));
      }
    } else if ( dataSIGPROC ) {
      obs.setNrBeams(1);
      obs.setNrSynthesizedBeams(1);
      bytesToSkip = args.getSwitchArgument< unsigned int >("-header");
      dataFile = args.getSwitchArgument< std::string >("-data");
      obs.setNrBatches(args.getSwitchArgument< unsigned int >("-batches"));
      obs.setFrequencyRange(1, args.getSwitchArgument< unsigned int >("-channels"), args.getSwitchArgument< float >("-min_freq"), args.getSwitchArgument< float >("-channel_bandwidth"));
      obs.setNrSamplesPerBatch(args.getSwitchArgument< unsigned int >("-samples"));
      obs.setSamplingTime(args.getSwitchArgument< float >("-sampling_time"));
#ifdef HAVE_PSRDADA
    } else if ( dataPSRDADA ) {
      std::string temp = args.getSwitchArgument< std::string >("-dada_key");
      dadaKey = std::stoi("0x" + temp, 0, 16);
      obs.setNrBeams(args.getSwitchArgument< unsigned int >("-beams"));
      obs.setNrSynthesizedBeams(args.getSwitchArgument< unsigned int >("-synthesized_beams"));
      obs.setNrBatches(args.getSwitchArgument< unsigned int >("-batches"));
#endif
    } else {
      obs.setNrBeams(args.getSwitchArgument< unsigned int >("-beams"));
      obs.setNrSynthesizedBeams(args.getSwitchArgument< unsigned int >("-synthesized_beams"));
      obs.setNrBatches(args.getSwitchArgument< unsigned int >("-batches"));
      if ( subbandDedispersion ) {
        obs.setFrequencyRange(args.getSwitchArgument< unsigned int >("-subbands"), args.getSwitchArgument< unsigned int >("-channels"), args.getSwitchArgument< float >("-min_freq"), args.getSwitchArgument< float >("-channel_bandwidth"));
      } else {
        obs.setFrequencyRange(1, args.getSwitchArgument< unsigned int >("-channels"), args.getSwitchArgument< float >("-min_freq"), args.getSwitchArgument< float >("-channel_bandwidth"));
      }
      obs.setNrSamplesPerBatch(args.getSwitchArgument< unsigned int >("-samples"));
      obs.setSamplingTime(args.getSwitchArgument< float >("-sampling_time"));
      random = args.getSwitch("-random");
      width = args.getSwitchArgument< unsigned int >("-width");
      DM = args.getSwitchArgument< float >("-dm");
    }
    inputBits = args.getSwitchArgument< unsigned int >("-input_bits");
    outputFile = args.getSwitchArgument< std::string >("-output");
    if ( subbandDedispersion ) {
      obs.setDMSubbandingRange(args.getSwitchArgument< unsigned int >("-subbanding_dms"), args.getSwitchArgument< float >("-subbanding_dm_first"), args.getSwitchArgument< float >("-subbanding_dm_step"));
    }
    obs.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), args.getSwitchArgument< float >("-dm_first"), args.getSwitchArgument< float >("-dm_step"));
    threshold = args.getSwitchArgument< float >("-threshold");
  } catch ( isa::utils::EmptyCommandLine & err ) {
    std::cerr <<  args.getName() << " -opencl_platform ... -opencl_device ... -device_name ... -padding_file ... -zapped_channels ... -integration_steps ... -integration_file ... -snr_file ... [-subband_dedispersion] [-print] [-compact_results] [-lofar] [-sigproc] [-dada] -input_bits ... -output ... -dms ... -dm_first ... -dm_step ... -threshold ..."<< std::endl;
    std::cerr << "\tDedispersion: -dedispersion_file ..." << std::endl;
    std::cerr << "\tSubband Dedispersion: -subband_dedispersion -dedispersion_step_one_file ... -dedispersion_step_two_file ... -subbands ... -subbanding_dms ... -subbanding_dm_first ... -subbanding_dm_step ..." << std::endl;
    std::cerr << "\t -lofar -header ... -data ... [-limit]" << std::endl;
    std::cerr << "\t\t -limit -batches ..." << std::endl;
    std::cerr << "\t -sigproc -header ... -data ... -batches ... -channels ... -min_freq ... -channel_bandwidth ... -samples ... -sampling_time ..." << std::endl;
    std::cerr << "\t -dada -dada_key ... -beams ... -synthesized_beams ... -batches ..." << std::endl;
    std::cerr << "\t [-random] -width ... -dm ... -beams ... -synthesized_beams ... -batches ... -channels ... -min_freq ... -channel_bandwidth ... -samples ... -sampling_time ..." << std::endl;
    return 1;
  } catch ( std::exception & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  // Load observation data
  isa::utils::Timer loadTime;
  std::vector< std::vector< std::vector< inputDataType > * > * > input(obs.getNrBeams());
  std::vector< std::vector< inputDataType > * > inputDADA;
  std::set< unsigned int > integrationSteps;
  if ( dataLOFAR ) {
    input[0] = new std::vector< std::vector< inputDataType > * >(obs.getNrBatches());
    loadTime.start();
    if ( limit ) {
      AstroData::readLOFAR(headerFile, dataFile, obs, padding[deviceName], *(input[0]), obs.getNrBatches());
    } else {
      AstroData::readLOFAR(headerFile, dataFile, obs, padding[deviceName], *(input[0]));
    }
    loadTime.stop();
  } else if ( dataSIGPROC ) {
    input[0] = new std::vector< std::vector< inputDataType > * >(obs.getNrBatches());
    loadTime.start();
    AstroData::readSIGPROC(obs, padding[deviceName], inputBits, bytesToSkip, dataFile, *(input[0]));
    loadTime.stop();
#ifdef HAVE_PSRDADA
  } else if ( dataPSRDADA ) {
    ringBuffer = dada_hdu_create(0);
    dada_hdu_set_key(ringBuffer, dadaKey);
    if ( dada_hdu_connect(ringBuffer) != 0 ) {
      std::cerr << "Impossible to connect to PSRDADA ringbuffer." << std::endl;
    }
    if ( dada_hdu_lock_read(ringBuffer) != 0 ) {
      std::cerr << "Impossible to lock the PSRDADA ringbuffer for reading the header." << std::endl;
    }
    try {
      AstroData::readPSRDADAHeader(obs, *ringBuffer);
    } catch ( AstroData::RingBufferError & err ) {
      std::cerr << "Error: " << err.what() << std::endl;
      return -1;
    }
#endif
  } else {
    for ( unsigned int beam = 0; beam < obs.getNrBeams(); beam++ ) {
      // TODO: if there are multiple synthesized beams, the generated data should take this into account
      input[beam] = new std::vector< std::vector< inputDataType > * >(obs.getNrBatches());
      AstroData::generateSinglePulse(width, DM, obs, padding[deviceName], *(input[beam]), inputBits, random);
    }
  }
  std::vector< uint8_t > zappedChannels(obs.getNrPaddedChannels(padding[deviceName] / sizeof(uint8_t)));
  try {
    AstroData::readZappedChannels(obs, channelsFile, zappedChannels);
    AstroData::readIntegrationSteps(obs, integrationFile, integrationSteps);
  } catch ( AstroData::FileError & err ) {
    std::cerr << err.what() << std::endl;
  }
  if ( DEBUG ) {
    std::cout << "Device: " << deviceName << std::endl;
    std::cout << "Padding: " << padding[deviceName] << " bytes" << std::endl;
    std::cout << std::endl;
    std::cout << "Beams: " << obs.getNrBeams() << std::endl;
    std::cout << "Synthesized Beams: " << obs.getNrSynthesizedBeams() << std::endl;
    std::cout << "Batches: " << obs.getNrBatches() << std::endl;
    std::cout << "Samples: " << obs.getNrSamplesPerBatch() << std::endl;
    std::cout << "Sampling time: " << obs.getSamplingTime() << std::endl;
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
    isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);
  } catch ( isa::OpenCL::OpenCLError & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  // Host memory allocation
  std::vector< float > * shiftsStepOne = PulsarSearch::getShifts(obs, padding[deviceName]);
  std::vector< float > * shiftsStepTwo = PulsarSearch::getShiftsStepTwo(obs, padding[deviceName]);
  if ( DEBUG ) {
    if ( print ) {
      std::cerr << "shiftsStepOne" << std::endl;
      for ( unsigned int channel = 0; channel < obs.getNrChannels(); channel++ ) {
        std::cerr << shiftsStepOne->at(channel) << " ";
      }
      std::cerr << std::endl;
      if ( subbandDedispersion ) {
        std::cerr << "shiftsStepTwo" << std::endl;
        for ( unsigned int subband = 0; subband < obs.getNrSubbands(); subband++ ) {
          std::cerr << shiftsStepTwo->at(subband) << " ";
        }
        std::cerr << std::endl;
      }
      std::cerr << std::endl;
    }
  }
  if ( subbandDedispersion ) {
    obs.setNrSamplesPerBatchSubbanding(obs.getNrSamplesPerBatch() + static_cast< unsigned int >(shiftsStepTwo->at(0) * (obs.getFirstDM() + ((obs.getNrDMs() - 1) * obs.getDMStep()))));
    obs.setNrSamplesPerSubbandingDispersedChannel(obs.getNrSamplesPerBatchSubbanding() + static_cast< unsigned int >(shiftsStepOne->at(0) * (obs.getFirstDMSubbanding() + ((obs.getNrDMsSubbanding() - 1) * obs.getDMSubbandingStep()))));
    obs.setNrDelayBatchesSubbanding(static_cast< unsigned int >(std::ceil(static_cast< double >(obs.getNrSamplesPerSubbandingDispersedChannel()) / obs.getNrSamplesPerBatch())));
  } else {
    obs.setNrSamplesPerDispersedChannel(obs.getNrSamplesPerBatch() + static_cast< unsigned int >(shiftsStepOne->at(0) * (obs.getFirstDM() + ((obs.getNrDMs() - 1) * obs.getDMStep()))));
    obs.setNrDelayBatches(static_cast< unsigned int >(std::ceil(static_cast< double >(obs.getNrSamplesPerDispersedChannel()) / obs.getNrSamplesPerBatch())));
  }
  std::vector< uint8_t > beamDriver;
  std::vector< inputDataType > dispersedData;
  std::vector< outputDataType > subbandedData;
  std::vector< outputDataType > dedispersedData;
  std::vector< outputDataType > integratedData;
  std::vector< float > snrData;
  std::vector< unsigned int > snrSamples;

  if ( subbandDedispersion ) {
#ifdef HAVE_PSRDADA
    if ( dataPSRDADA ) {
      inputDADA.resize(obs.getNrDelayBatchesSubbanding());
      for ( unsigned int batch = 0; batch < obs.getNrDelayBatchesSubbanding(); batch++ ) {
        if ( inputBits >= 8 ) {
          inputDADA.at(batch) = new std::vector< inputDataType >(obs.getNrBeams() * obs.getNrChannels() * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(inputDataType)));
        } else {
          inputDADA.at(batch) = new std::vector< inputDataType >(obs.getNrBeams() * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerBatch() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType)));
        }
      }
    }
#endif
    if ( dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getSplitBatches() ) {
      // TODO: add support for splitBatches
    } else {
      if ( inputBits >= 8 ) {
        dispersedData.resize(obs.getNrBeams() * obs.getNrChannels() * obs.getNrSamplesPerPaddedSubbandingDispersedChannel(padding[deviceName] / sizeof(inputDataType)));
      } else {
        dispersedData.resize(obs.getNrBeams() * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerSubbandingDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType)));
      }
    }
    beamDriver.resize(obs.getNrSynthesizedBeams() * obs.getNrPaddedSubbands(padding[deviceName] / sizeof(uint8_t)));
    subbandedData.resize(obs.getNrBeams() * obs.getNrDMsSubbanding() * obs.getNrSubbands() * obs.getNrSamplesPerPaddedBatchSubbanding(padding[deviceName] / sizeof(outputDataType)));
    dedispersedData.resize(obs.getNrSynthesizedBeams() * obs.getNrDMsSubbanding() * obs.getNrDMs() * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(outputDataType)));
    integratedData.resize(obs.getNrSynthesizedBeams() * obs.getNrDMsSubbanding() * obs.getNrDMs() * isa::utils::pad(obs.getNrSamplesPerBatch() / *(integrationSteps.begin()), padding[deviceName] / sizeof(outputDataType)));
    snrData.resize(obs.getNrSynthesizedBeams() * isa::utils::pad(obs.getNrDMsSubbanding() * obs.getNrDMs(), padding[deviceName] / sizeof(float)));
    snrSamples.resize(obs.getNrSynthesizedBeams() * isa::utils::pad(obs.getNrDMsSubbanding() * obs.getNrDMs(), padding[deviceName] / sizeof(unsigned int)));
  } else {
#ifdef HAVE_PSRDADA
    if ( dataPSRDADA ) {
      inputDADA.resize(obs.getNrDelayBatches());
      for ( unsigned int batch = 0; batch < obs.getNrDelayBatches(); batch++ ) {
        if ( inputBits >= 8 ) {
          inputDADA.at(batch) = new std::vector< inputDataType >(obs.getNrBeams() * obs.getNrChannels() * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(inputDataType)));
        } else {
          inputDADA.at(batch) = new std::vector< inputDataType >(obs.getNrBeams() * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerBatch() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType)));
        }
      }
    }
#endif
    if ( dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitBatches() ) {
      // TODO: add support for splitBatches
    } else {
      if ( inputBits >= 8 ) {
        dispersedData.resize(obs.getNrBeams() * obs.getNrChannels() * obs.getNrSamplesPerPaddedDispersedChannel(padding[deviceName] / sizeof(inputDataType)));
      } else {
        dispersedData.resize(obs.getNrBeams() * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType)));
      }
    }
    beamDriver.resize(obs.getNrSynthesizedBeams() * obs.getNrPaddedChannels(padding[deviceName] / sizeof(uint8_t)));
    dedispersedData.resize(obs.getNrSynthesizedBeams() * obs.getNrDMs() * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(outputDataType)));
    integratedData.resize(obs.getNrSynthesizedBeams() * obs.getNrDMs() * isa::utils::pad(obs.getNrSamplesPerBatch() / *(integrationSteps.begin()), padding[deviceName] / sizeof(outputDataType)));
    snrData.resize(obs.getNrSynthesizedBeams() * obs.getNrPaddedDMs(padding[deviceName] / sizeof(float)));
    snrSamples.resize(obs.getNrSynthesizedBeams() * obs.getNrPaddedDMs(padding[deviceName] / sizeof(unsigned int)));
  }
  generateBeamDriver(subbandDedispersion, obs, beamDriver, padding[deviceName]);

  if ( obs.getNrDelayBatches() > obs.getNrBatches() ) {
    std::cerr << "Not enough input batches for the search." << std::endl;
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
  cl::Buffer snrSamples_d;

  try {
    shiftsStepOne_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, shiftsStepOne->size() * sizeof(float), 0, 0);
    zappedChannels_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, zappedChannels.size() * sizeof(uint8_t), 0, 0);
    beamDriver_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, beamDriver.size() * sizeof(uint8_t), 0, 0);
    dispersedData_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, dispersedData.size() * sizeof(inputDataType), 0, 0);
    dedispersedData_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, dedispersedData.size() * sizeof(outputDataType), 0, 0);
    integratedData_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, integratedData.size() * sizeof(outputDataType), 0, 0);
    snrData_d = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, snrData.size() * sizeof(float), 0, 0);
    snrSamples_d = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, snrSamples.size() * sizeof(unsigned int), 0, 0);
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
    std::cerr << "Memory error: " << err.what() << " " << err.err() << std::endl;
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
      if ( dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getSplitBatches() ) {
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
    code = PulsarSearch::getSubbandDedispersionStepTwoOpenCL< outputDataType >(*(dedispersionStepTwoParameters.at(deviceName)->at(obs.getNrDMs())), padding[deviceName], outputDataName, obs, *shiftsStepTwo);
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
    code = PulsarSearch::getDedispersionOpenCL< inputDataType, outputDataType >(*(dedispersionParameters.at(deviceName)->at(obs.getNrDMs())), padding[deviceName], inputBits, inputDataName, intermediateDataName, outputDataName, obs, *shiftsStepOne);
    try {
      dedispersionK = isa::OpenCL::compile("dedispersion", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
      if ( dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitBatches() ) {
        // TODO: add support for splitBatches
      } else {
        dedispersionK->setArg(0, dispersedData_d);
        dedispersionK->setArg(1, dedispersedData_d);
        dedispersionK->setArg(2, beamDriver_d);
        dedispersionK->setArg(3, zappedChannels_d);
        dedispersionK->setArg(4, shiftsStepOne_d);
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
    snrDMsSamplesK[integrationSteps.size()]->setArg(2, snrSamples_d);
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
    dedispersionStepTwoGlobal = cl::NDRange(isa::utils::pad(obs.getNrSamplesPerBatchSubbanding() / dedispersionStepTwoParameters.at(deviceName)->at(obs.getNrDMs())->getNrItemsD0(), dedispersionStepTwoParameters.at(deviceName)->at(obs.getNrDMs())->getNrThreadsD0()), obs.getNrDMs() / dedispersionStepTwoParameters.at(deviceName)->at(obs.getNrDMs())->getNrItemsD1(), obs.getNrSynthesizedBeams() * obs.getNrDMsSubbanding());
    dedispersionStepTwoLocal = cl::NDRange(dedispersionStepTwoParameters.at(deviceName)->at(obs.getNrDMs())->getNrThreadsD0(), dedispersionStepTwoParameters.at(deviceName)->at(obs.getNrDMs())->getNrThreadsD1(), 1);
    if ( DEBUG ) {
      std::cout << "DedispersionStepTwo" << std::endl;
        std::cout << "Global: " << isa::utils::pad(obs.getNrSamplesPerBatchSubbanding() / dedispersionStepTwoParameters.at(deviceName)->at(obs.getNrDMs())->getNrItemsD0(), dedispersionStepTwoParameters.at(deviceName)->at(obs.getNrDMs())->getNrThreadsD0()) << ", " << obs.getNrDMs() / dedispersionStepTwoParameters.at(deviceName)->at(obs.getNrDMs())->getNrItemsD1() << ", " << obs.getNrSynthesizedBeams() * obs.getNrDMsSubbanding() << std::endl;
        std::cout << "Local: " << dedispersionStepTwoParameters.at(deviceName)->at(obs.getNrDMs())->getNrThreadsD0() << ", " << dedispersionStepTwoParameters.at(deviceName)->at(obs.getNrDMs())->getNrThreadsD1() << ", 1" << std::endl;
        std::cout << "Parameters: ";
        std::cout << dedispersionStepTwoParameters.at(deviceName)->at(obs.getNrDMs())->print() << std::endl;
        std::cout << std::endl;
    }
  } else {
    dedispersionGlobal = cl::NDRange(isa::utils::pad(obs.getNrSamplesPerBatch() / dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrItemsD0(), dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrThreadsD0()), obs.getNrDMs() / dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrItemsD1(), obs.getNrSynthesizedBeams());
    dedispersionLocal = cl::NDRange(dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrThreadsD0(), dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrThreadsD1(), 1);
    if ( DEBUG ) {
      std::cout << "Dedispersion" << std::endl;
      std::cout << "Global: " << isa::utils::pad(obs.getNrSamplesPerBatch() / dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrItemsD0(), dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrThreadsD0()) << ", " << obs.getNrDMs() / dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrItemsD1() << ", " << obs.getNrSynthesizedBeams() << std::endl;
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
    snrDMsSamplesGlobal[integrationSteps.size()] = cl::NDRange(snrParameters.at(deviceName)->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(obs.getNrSamplesPerBatch())->getNrThreadsD0(), obs.getNrDMsSubbanding() * obs.getNrDMs(), obs.getNrSynthesizedBeams());
    snrDMsSamplesLocal[integrationSteps.size()] = cl::NDRange(snrParameters.at(deviceName)->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(obs.getNrSamplesPerBatch())->getNrThreadsD0(), 1, 1);
    if ( DEBUG ) {
      std::cout << "SNRDMsSamples (" + std::to_string(obs.getNrSamplesPerBatch()) + ")" << std::endl;
      std::cout << "Global: " << snrParameters.at(deviceName)->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(obs.getNrSamplesPerBatch())->getNrThreadsD0() << ", " << obs.getNrDMsSubbanding() * obs.getNrDMs() << " " << obs.getNrSynthesizedBeams() << std::endl;
      std::cout << "Local: " << snrParameters.at(deviceName)->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(obs.getNrSamplesPerBatch())->getNrThreadsD0() << ", 1, 1" << std::endl;
      std::cout << "Parameters: ";
      std::cout << snrParameters.at(deviceName)->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(obs.getNrSamplesPerBatch())->print() << std::endl;
      std::cout << std::endl;
    }
  } else {
    snrDMsSamplesGlobal[integrationSteps.size()] = cl::NDRange(snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerBatch())->getNrThreadsD0(), obs.getNrDMs(), obs.getNrSynthesizedBeams());
    snrDMsSamplesLocal[integrationSteps.size()] = cl::NDRange(snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerBatch())->getNrThreadsD0(), 1, 1);
    if ( DEBUG ) {
      std::cout << "SNRDMsSamples (" + std::to_string(obs.getNrSamplesPerBatch()) + ")" << std::endl;
      std::cout << "Global: " << snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerBatch())->getNrThreadsD0() << ", " << obs.getNrDMs() << ", " << obs.getNrSynthesizedBeams() << std::endl;
      std::cout << "Local: " << snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerBatch())->getNrThreadsD0() << ", 1, 1" << std::endl;
      std::cout << "Parameters: ";
      std::cout << snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerBatch())->print() << std::endl;
      std::cout << std::endl;
    }
  }
  for ( unsigned int stepNumber = 0; stepNumber < integrationSteps.size(); stepNumber++ ) {
    auto step = integrationSteps.begin();

    std::advance(step, stepNumber);
    if ( subbandDedispersion ) {
      integrationGlobal[stepNumber] = cl::NDRange(integrationParameters[deviceName]->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(*step)->getNrThreadsD0() * ((obs.getNrSamplesPerBatch() / *step) / integrationParameters[deviceName]->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(*step)->getNrItemsD0()), obs.getNrDMsSubbanding() * obs.getNrDMs(), obs.getNrSynthesizedBeams());
      integrationLocal[stepNumber] = cl::NDRange(integrationParameters[deviceName]->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(*step)->getNrThreadsD0(), 1, 1);
      if ( DEBUG ) {
        std::cout << "integrationDMsSamples (" + std::to_string(*step) + ")" << std::endl;
        std::cout << "Global: " << integrationParameters[deviceName]->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(*step)->getNrThreadsD0() * ((obs.getNrSamplesPerBatch() / *step) / integrationParameters[deviceName]->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(*step)->getNrItemsD0()) << ", " << obs.getNrDMsSubbanding() * obs.getNrDMs() << ", " << obs.getNrSynthesizedBeams() << std::endl;
        std::cout << "Local: " << integrationParameters[deviceName]->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(*step)->getNrThreadsD0() << ", 1, 1" << std::endl;
        std::cout << "Parameters: ";
        std::cout << integrationParameters[deviceName]->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(*step)->print() << std::endl;
        std::cout << std::endl;
      }
      snrDMsSamplesGlobal[stepNumber] = cl::NDRange(snrParameters.at(deviceName)->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(obs.getNrSamplesPerBatch() / *step)->getNrThreadsD0(), obs.getNrDMsSubbanding() * obs.getNrDMs(), obs.getNrSynthesizedBeams());
      snrDMsSamplesLocal[stepNumber] = cl::NDRange(snrParameters.at(deviceName)->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(obs.getNrSamplesPerBatch() / *step)->getNrThreadsD0(), 1, 1);
      if ( DEBUG ) {
        std::cout << "SNRDMsSamples (" + std::to_string(obs.getNrSamplesPerBatch() / *step) + ")" << std::endl;
        std::cout << "Global: " << snrParameters.at(deviceName)->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(obs.getNrSamplesPerBatch() / *step)->getNrThreadsD0() << ", " << obs.getNrDMsSubbanding() * obs.getNrDMs() << " " << obs.getNrSynthesizedBeams() << std::endl;
        std::cout << "Local: " << snrParameters.at(deviceName)->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(obs.getNrSamplesPerBatch() / *step)->getNrThreadsD0() << ", 1, 1" << std::endl;
        std::cout << "Parameters: ";
        std::cout << snrParameters.at(deviceName)->at(obs.getNrDMsSubbanding() * obs.getNrDMs())->at(obs.getNrSamplesPerBatch() / *step)->print() << std::endl;
        std::cout << std::endl;
      }
    } else {
      integrationGlobal[stepNumber] = cl::NDRange(integrationParameters[deviceName]->at(obs.getNrDMs())->at(*step)->getNrThreadsD0() * ((obs.getNrSamplesPerBatch() / *step) / integrationParameters[deviceName]->at(obs.getNrDMs())->at(*step)->getNrItemsD0()), obs.getNrDMs(), obs.getNrSynthesizedBeams());
      integrationLocal[stepNumber] = cl::NDRange(integrationParameters[deviceName]->at(obs.getNrDMs())->at(*step)->getNrThreadsD0(), 1, 1);
      if ( DEBUG ) {
        std::cout << "integrationDMsSamples (" + std::to_string(*step) + ")" << std::endl;
        std::cout << "Global: " << integrationParameters[deviceName]->at(obs.getNrDMs())->at(*step)->getNrThreadsD0() * ((obs.getNrSamplesPerBatch() / *step) / integrationParameters[deviceName]->at(obs.getNrDMs())->at(*step)->getNrItemsD0()) << ", " << obs.getNrDMs() << ", " << obs.getNrSynthesizedBeams() << std::endl;
        std::cout << "Local: " << integrationParameters[deviceName]->at(obs.getNrDMs())->at(*step)->getNrThreadsD0() << ", 1, 1" << std::endl;
        std::cout << "Parameters: ";
        std::cout << integrationParameters[deviceName]->at(obs.getNrDMs())->at(*step)->print() << std::endl;
        std::cout << std::endl;
      }
      snrDMsSamplesGlobal[stepNumber] = cl::NDRange(snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerBatch() / *step)->getNrThreadsD0(), obs.getNrDMs(), obs.getNrSynthesizedBeams());
      snrDMsSamplesLocal[stepNumber] = cl::NDRange(snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerBatch() / *step)->getNrThreadsD0(), 1, 1);
      if ( DEBUG ) {
        std::cout << "SNRDMsSamples (" + std::to_string(obs.getNrSamplesPerBatch() / *step) + ")" << std::endl;
        std::cout << "Global: " << snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerBatch() / *step)->getNrThreadsD0() << ", " << obs.getNrDMs() << " " << obs.getNrSynthesizedBeams() << std::endl;
        std::cout << "Local: " << snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerBatch() / *step)->getNrThreadsD0() << ", 1, 1" << std::endl;
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
  if ( compactResults ) {
    output << "# beam batch sample integration_step compacted_integration_steps time DM compacted_DMs SNR" << std::endl;
  } else {
    output << "# beam batch sample integration_step time DM SNR" << std::endl;
  }
  for ( unsigned int batch = 0; batch < obs.getNrBatches(); batch++ ) {
    triggeredEvents_t triggeredEvents(obs.getNrSynthesizedBeams());
    compactedEvents_t compactedEvents(obs.getNrSynthesizedBeams());

    // Load the input
    inputHandlingTimer.start();
    if ( !dataPSRDADA ) {
      // If there are not enough available batches, computation is complete
      if ( subbandDedispersion ) {
        if ( batch == obs.getNrBatches() - obs.getNrDelayBatchesSubbanding() ) {
          break;
        }
      } else {
        if ( batch == obs.getNrBatches() - obs.getNrDelayBatches() ) {
          break;
        }
      }
      // If there are enough batches, prepare them for transfer to device
      for ( unsigned int beam = 0; beam < obs.getNrBeams(); beam++ ) {
        if ( subbandDedispersion ) {
          if ( !dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getSplitBatches() ) {
            for ( unsigned int channel = 0; channel < obs.getNrChannels(); channel++ ) {
              for ( unsigned int chunk = 0; chunk < obs.getNrDelayBatchesSubbanding() - 1; chunk++ ) {
                if ( inputBits >= 8 ) {
                  memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * obs.getNrChannels() * obs.getNrSamplesPerPaddedSubbandingDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + (channel * obs.getNrSamplesPerPaddedSubbandingDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + (chunk * obs.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&((input[beam]->at(batch + chunk))->at(channel * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(inputDataType))))), obs.getNrSamplesPerBatch() * sizeof(inputDataType));
                } else {
                  memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerSubbandingDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(obs.getNrSamplesPerSubbandingDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + (chunk * (obs.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&((input[beam]->at(batch + chunk))->at(channel * isa::utils::pad(obs.getNrSamplesPerBatch() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))))), (obs.getNrSamplesPerBatch() / (8 / inputBits)) * sizeof(inputDataType));
                }
              }
              if ( inputBits >= 8 ) {
                memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * obs.getNrChannels() * obs.getNrSamplesPerPaddedSubbandingDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + (channel * obs.getNrSamplesPerPaddedSubbandingDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + ((obs.getNrDelayBatchesSubbanding() - 1) * obs.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&((input[beam]->at(batch + (obs.getNrDelayBatchesSubbanding() - 1)))->at(channel * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(inputDataType))))), (obs.getNrSamplesPerSubbandingDispersedChannel() % obs.getNrSamplesPerBatch()) * sizeof(inputDataType));
              } else {
                memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerSubbandingDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(obs.getNrSamplesPerSubbandingDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + ((obs.getNrDelayBatchesSubbanding() - 1) * (obs.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&((input[beam]->at(batch + (obs.getNrDelayBatchesSubbanding() - 1)))->at(channel * isa::utils::pad(obs.getNrSamplesPerBatch() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))))), ((obs.getNrSamplesPerSubbandingDispersedChannel() % obs.getNrSamplesPerBatch()) / (8 / inputBits)) * sizeof(inputDataType));
              }
            }
          }
        } else {
          if ( !dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitBatches() ) {
            for ( unsigned int channel = 0; channel < obs.getNrChannels(); channel++ ) {
              for ( unsigned int chunk = 0; chunk < obs.getNrDelayBatches() - 1; chunk++ ) {
                if ( inputBits >= 8 ) {
                  memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * obs.getNrChannels() * obs.getNrSamplesPerPaddedDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + (channel * obs.getNrSamplesPerPaddedDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + (chunk * obs.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&((input[beam]->at(batch + chunk))->at(channel * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(inputDataType))))), obs.getNrSamplesPerBatch() * sizeof(inputDataType));
                } else {
                  memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(obs.getNrSamplesPerDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + (chunk * (obs.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&((input[beam]->at(batch + chunk))->at(channel * isa::utils::pad(obs.getNrSamplesPerBatch() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))))), (obs.getNrSamplesPerBatch() / (8 / inputBits)) * sizeof(inputDataType));
                }
              }
              if ( inputBits >= 8 ) {
                memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * obs.getNrChannels() * obs.getNrSamplesPerPaddedDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + (channel * obs.getNrSamplesPerPaddedDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + ((obs.getNrDelayBatches() - 1) * obs.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&((input[beam]->at(batch + (obs.getNrDelayBatches() - 1)))->at(channel * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(inputDataType))))), (obs.getNrSamplesPerDispersedChannel() % obs.getNrSamplesPerBatch()) * sizeof(inputDataType));
              } else {
                memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(obs.getNrSamplesPerDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + ((obs.getNrDelayBatches() - 1) * (obs.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&((input[beam]->at(batch + (obs.getNrDelayBatches() - 1)))->at(channel * isa::utils::pad(obs.getNrSamplesPerBatch() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))))), ((obs.getNrSamplesPerDispersedChannel() % obs.getNrSamplesPerBatch()) / (8 / inputBits)) * sizeof(inputDataType));
              }
            }
          }
        }
      }
    } else {
#ifdef HAVE_PSRDADA
      try {
        if ( ipcbuf_eod(reinterpret_cast< ipcbuf_t * >(ringBuffer->data_block)) ) {
          errorDetected = true;
          break;
        }
        if ( subbandDedispersion ) {
          AstroData::readPSRDADA(*ringBuffer, inputDADA.at(batch % obs.getNrDelayBatchesSubbanding()));
        } else {
          AstroData::readPSRDADA(*ringBuffer, inputDADA.at(batch % obs.getNrDelayBatches()));
        }
      } catch ( AstroData::RingBufferError & err ) {
        std::cerr << "Error: " << err.what() << std::endl;
        return -1;
      }
      // If there are enough data buffered, proceed with the computation
      // Otherwise, move to the next iteration of the search loop
      if ( subbandDedispersion ) {
        if ( batch < obs.getNrDelayBatchesSubbanding() - 1 ) {
          continue;
        }
      } else {
        if ( batch < obs.getNrDelayBatches() - 1 ) {
          continue;
        }
      }
      for ( unsigned int beam = 0; beam < obs.getNrBeams(); beam++ ) {
        if ( subbandDedispersion ) {
          if ( !dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getSplitBatches() ) {
            for ( unsigned int channel = 0; channel < obs.getNrChannels(); channel++ ) {
              for ( unsigned int chunk = batch - (obs.getNrDelayBatchesSubbanding() - 1); chunk < batch; chunk++ ) {
                // Full batches
                if ( inputBits >= 8 ) {
                  memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * obs.getNrChannels() * obs.getNrSamplesPerPaddedSubbandingDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + (channel * obs.getNrSamplesPerPaddedSubbandingDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + ((chunk - (batch - (obs.getNrDelayBatchesSubbanding() - 1))) * obs.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&(inputDADA.at(chunk % obs.getNrDelayBatchesSubbanding())->at((beam * obs.getNrChannels() * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(inputDataType))) + (channel * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(inputDataType)))))), obs.getNrSamplesPerBatch() * sizeof(inputDataType));
                } else {
                  memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerSubbandingDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(obs.getNrSamplesPerSubbandingDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + ((chunk - (batch - (obs.getNrDelayBatchesSubbanding() - 1))) * (obs.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&(inputDADA.at(chunk % obs.getNrDelayBatchesSubbanding())->at((beam * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerBatch() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(obs.getNrSamplesPerBatch() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType)))))), (obs.getNrSamplesPerBatch() / (8 / inputBits)) * sizeof(inputDataType));
                }
              }
              // Remainder (part of current batch)
              if ( inputBits >= 8 ) {
                memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * obs.getNrChannels() * obs.getNrSamplesPerPaddedSubbandingDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + (channel * obs.getNrSamplesPerPaddedSubbandingDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + ((obs.getNrDelayBatchesSubbanding() - 1) * obs.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&(inputDADA.at(batch % obs.getNrDelayBatchesSubbanding())->at((beam * obs.getNrChannels() * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(inputDataType))) + (channel * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(inputDataType)))))), (obs.getNrSamplesPerSubbandingDispersedChannel() % obs.getNrSamplesPerBatch()) * sizeof(inputDataType));
              } else {
                memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerSubbandingDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(obs.getNrSamplesPerSubbandingDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + ((obs.getNrDelayBatchesSubbanding() - 1) * (obs.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&(inputDADA.at(batch % obs.getNrDelayBatchesSubbanding())->at((beam * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerBatch() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(obs.getNrSamplesPerBatch() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType)))))), ((obs.getNrSamplesPerSubbandingDispersedChannel() % obs.getNrSamplesPerBatch()) / (8 / inputBits)) * sizeof(inputDataType));
              }
            }
          }
        } else {
          if ( !dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitBatches() ) {
            for ( unsigned int channel = 0; channel < obs.getNrChannels(); channel++ ) {
              for ( unsigned int chunk = batch - (obs.getNrDelayBatches() - 1); chunk < batch; chunk++ ) {
                if ( inputBits >= 8 ) {
                  memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * obs.getNrChannels() * obs.getNrSamplesPerPaddedDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + (channel * obs.getNrSamplesPerPaddedDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + ((chunk - (batch - (obs.getNrDelayBatches() - 1))) * obs.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&(inputDADA.at(chunk % obs.getNrDelayBatches())->at((beam * obs.getNrChannels() * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(inputDataType))) + (channel * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(inputDataType)))))), obs.getNrSamplesPerBatch() * sizeof(inputDataType));
                } else {
                  memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(obs.getNrSamplesPerDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + ((chunk - (batch - (obs.getNrDelayBatches() - 1))) * (obs.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&(inputDADA.at(chunk % obs.getNrDelayBatches())->at((beam * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerBatch() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(obs.getNrSamplesPerBatch() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType)))))), (obs.getNrSamplesPerBatch() / (8 / inputBits)) * sizeof(inputDataType));
                }
              }
              if ( inputBits >= 8 ) {
                memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * obs.getNrChannels() * obs.getNrSamplesPerPaddedDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + (channel * obs.getNrSamplesPerPaddedDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + ((obs.getNrDelayBatches() - 1) * obs.getNrSamplesPerBatch())])), reinterpret_cast< void * >(&(inputDADA.at(batch % obs.getNrDelayBatches())->at((beam * obs.getNrChannels() * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(inputDataType))) + (channel * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(inputDataType)))))), (obs.getNrSamplesPerDispersedChannel() % obs.getNrSamplesPerBatch()) * sizeof(inputDataType));
              } else {
                memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(beam * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(obs.getNrSamplesPerDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + ((obs.getNrDelayBatches() - 1) * (obs.getNrSamplesPerBatch() / (8 / inputBits)))])), reinterpret_cast< void * >(&(inputDADA.at(batch % obs.getNrDelayBatches())->at((beam * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerBatch() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + (channel * isa::utils::pad(obs.getNrSamplesPerBatch() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType)))))), ((obs.getNrSamplesPerDispersedChannel() % obs.getNrSamplesPerBatch()) / (8 / inputBits)) * sizeof(inputDataType));
              }
            }
          }
        }
      }
#endif
    }
    inputHandlingTimer.stop();
    // Copy input from host to device
    try {
      if ( SYNC ) {
        inputCopyTimer.start();
        if ( subbandDedispersion ) {
          if ( dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getSplitBatches() ) {
            // TODO: add support for splitBatches
          } else {
            clQueues->at(clDeviceID)[0].enqueueWriteBuffer(dispersedData_d, CL_TRUE, 0, dispersedData.size() * sizeof(inputDataType), reinterpret_cast< void * >(dispersedData.data()), 0, &syncPoint);
          }
        } else {
          if ( dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitBatches() ) {
            // TODO: add support for splitBatches
          } else {
            clQueues->at(clDeviceID)[0].enqueueWriteBuffer(dispersedData_d, CL_TRUE, 0, dispersedData.size() * sizeof(inputDataType), reinterpret_cast< void * >(dispersedData.data()), 0, &syncPoint);
          }
        }
        syncPoint.wait();
        inputCopyTimer.stop();
      } else {
        if ( subbandDedispersion ) {
          if ( dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getSplitBatches() ) {
            // TODO: add support for splitBatches
          } else {
            clQueues->at(clDeviceID)[0].enqueueWriteBuffer(dispersedData_d, CL_FALSE, 0, dispersedData.size() * sizeof(inputDataType), reinterpret_cast< void * >(dispersedData.data()));
          }
        } else {
          if ( dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitBatches() ) {
            // TODO: add support for splitBatches
          } else {
            clQueues->at(clDeviceID)[0].enqueueWriteBuffer(dispersedData_d, CL_FALSE, 0, dispersedData.size() * sizeof(inputDataType), reinterpret_cast< void * >(dispersedData.data()));
          }
        }
      }
      if ( DEBUG ) {
        if ( print ) {
          // TODO: add support for splitBatches
          std::cerr << "dispersedData" << std::endl;
          if ( subbandDedispersion ) {
            if ( inputBits >= 8 ) {
              for ( unsigned int beam = 0; beam < obs.getNrBeams(); beam++ ) {
                std::cerr << "Beam: " << beam << std::endl;
                for ( unsigned int channel = 0; channel < obs.getNrChannels(); channel++ ) {
                  for ( unsigned int sample = 0; sample < obs.getNrSamplesPerPaddedSubbandingDispersedChannel(padding[deviceName] / sizeof(inputDataType)); sample++ ) {
                    std::cerr << static_cast< float >(dispersedData[(beam * obs.getNrChannels() * obs.getNrSamplesPerPaddedSubbandingDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + (channel * obs.getNrSamplesPerPaddedSubbandingDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + sample]) << " ";
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
              for ( unsigned int beam = 0; beam < obs.getNrBeams(); beam++ ) {
                std::cerr << "Beam: " << beam << std::endl;
                for ( unsigned int channel = 0; channel < obs.getNrChannels(); channel++ ) {
                  for ( unsigned int sample = 0; sample < obs.getNrSamplesPerPaddedDispersedChannel(padding[deviceName] / sizeof(inputDataType)); sample++ ) {
                    std::cerr << static_cast< float >(dispersedData[(beam * obs.getNrChannels() * obs.getNrSamplesPerPaddedDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + (channel * obs.getNrSamplesPerPaddedDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + sample]) << " ";
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
    if ( subbandDedispersion ) {
      if ( dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getSplitBatches() && (batch < obs.getNrDelayBatches()) ) {
        // Not enough batches in the buffer to start the search
        continue;
      }
    } else {
      if ( dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitBatches() && (batch < obs.getNrDelayBatches()) ) {
        // Not enough batches in the buffer to start the search
        continue;
      }
    }

    // Dedispersion
    if ( subbandDedispersion ) {
      if ( dedispersionStepOneParameters.at(deviceName)->at(obs.getNrDMsSubbanding())->getSplitBatches() ) {
        // TODO: add support for splitBatches
      }
      if ( SYNC ) {
        try {
          dedispersionTimer.start();
          dedispersionStepOneTimer.start();
          clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*dedispersionStepOneK, cl::NullRange, dedispersionStepOneGlobal, dedispersionStepOneLocal, 0, &syncPoint);
          syncPoint.wait();
          dedispersionStepOneTimer.stop();
        } catch ( cl::Error & err ) {
          std::cerr << "Dedispersion Step One error -- Batch: " << std::to_string(batch) << ", " << err.what() << " " << err.err() << std::endl;
          errorDetected = true;
        }
        try {
          dedispersionStepTwoTimer.start();
          clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*dedispersionStepTwoK, cl::NullRange, dedispersionStepTwoGlobal, dedispersionStepTwoLocal, 0, &syncPoint);
          syncPoint.wait();
          dedispersionStepTwoTimer.stop();
          dedispersionTimer.stop();
        } catch ( cl::Error & err ) {
          std::cerr << "Dedispersion Step Two error -- Batch: " << std::to_string(batch) << ", " << err.what() << " " << err.err() << std::endl;
          errorDetected = true;
        }
      } else {
        try {
          clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*dedispersionStepOneK, cl::NullRange, dedispersionStepOneGlobal, dedispersionStepOneLocal, 0, 0);
        } catch ( cl::Error & err ) {
          std::cerr << "Dedispersion Step One error -- Batch: " << std::to_string(batch) << ", " << err.what() << " " << err.err() << std::endl;
          errorDetected = true;
        }
        try {
          clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*dedispersionStepTwoK, cl::NullRange, dedispersionStepTwoGlobal, dedispersionStepTwoLocal, 0, 0);
        } catch ( cl::Error & err ) {
          std::cerr << "Dedispersion Step Two error -- Batch: " << std::to_string(batch) << ", " << err.what() << " " << err.err() << std::endl;
          errorDetected = true;
        }
      }
    } else {
      try {
          if ( dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitBatches() ) {
            // TODO: add support for splitBatches
          }
          if ( SYNC ) {
            dedispersionTimer.start();
            clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*dedispersionK, cl::NullRange, dedispersionGlobal, dedispersionLocal, 0, &syncPoint);
            syncPoint.wait();
            dedispersionTimer.stop();
          } else {
            clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*dedispersionK, cl::NullRange, dedispersionGlobal, dedispersionLocal);
          }
      } catch ( cl::Error & err ) {
        std::cerr << "Dedispersion error -- Batch: " << std::to_string(batch) << ", " << err.what() << " " << err.err() << std::endl;
        errorDetected = true;
      }
    }
    if ( DEBUG ) {
      if ( print ) {
        if ( subbandDedispersion ) {
          try {
            clQueues->at(clDeviceID)[0].enqueueReadBuffer(subbandedData_d, CL_TRUE, 0, subbandedData.size() * sizeof(outputDataType), reinterpret_cast< void * >(subbandedData.data()), 0, &syncPoint);
            syncPoint.wait();
            clQueues->at(clDeviceID)[0].enqueueReadBuffer(dedispersedData_d, CL_TRUE, 0, dedispersedData.size() * sizeof(outputDataType), reinterpret_cast< void * >(dedispersedData.data()), 0, &syncPoint);
            syncPoint.wait();
            std::cerr << "subbandedData" << std::endl;
            for ( unsigned int beam = 0; beam < obs.getNrBeams(); beam++ ) {
              std::cerr << "Beam: " << beam << std::endl;
              for ( unsigned int dm = 0; dm < obs.getNrDMsSubbanding(); dm++ ) {
                std::cerr << "Subbanding DM: " << dm << std::endl;
                for ( unsigned int subband = 0; subband < obs.getNrSubbands(); subband++ ) {
                  for ( unsigned int sample = 0; sample < obs.getNrSamplesPerBatchSubbanding(); sample++ ) {
                    std::cerr << subbandedData[(beam * obs.getNrDMsSubbanding() * obs.getNrSubbands() * obs.getNrSamplesPerPaddedBatchSubbanding(padding[deviceName] / sizeof(outputDataType))) + (dm * obs.getNrSubbands() * obs.getNrSamplesPerPaddedBatchSubbanding(padding[deviceName] / sizeof(outputDataType))) + (subband * obs.getNrSamplesPerPaddedBatchSubbanding(padding[deviceName] / sizeof(outputDataType))) + sample] << " ";
                  }
                  std::cerr << std::endl;
                }
                std::cerr << std::endl;
              }
              std::cerr << std::endl;
            }
            std::cerr << "dedispersedData" << std::endl;
            for ( unsigned int sBeam = 0; sBeam < obs.getNrSynthesizedBeams(); sBeam++ ) {
              std::cerr << "sBeam: " << sBeam << std::endl;
              for ( unsigned int subbandingDM = 0; subbandingDM < obs.getNrDMsSubbanding(); subbandingDM++ ) {
                for ( unsigned int dm = 0; dm < obs.getNrDMs(); dm++ ) {
                  std::cerr << "DM: " << (subbandingDM * obs.getNrDMs()) + dm << std::endl;
                  for ( unsigned int sample = 0; sample < obs.getNrSamplesPerBatch(); sample++ ) {
                    std::cerr << dedispersedData[(sBeam * obs.getNrDMsSubbanding() * obs.getNrDMs() * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(outputDataType))) + (subbandingDM * obs.getNrDMs() * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(outputDataType))) + (dm * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(outputDataType))) + sample] << " ";
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
            clQueues->at(clDeviceID)[0].enqueueReadBuffer(dedispersedData_d, CL_TRUE, 0, dedispersedData.size() * sizeof(outputDataType), reinterpret_cast< void * >(dedispersedData.data()), 0, &syncPoint);
            syncPoint.wait();
            std::cerr << "dedispersedData" << std::endl;
            for ( unsigned int sBeam = 0; sBeam < obs.getNrSynthesizedBeams(); sBeam++ ) {
              std::cerr << "sBeam: " << sBeam << std::endl;
              for ( unsigned int dm = 0; dm < obs.getNrDMs(); dm++ ) {
                std::cerr << "DM: " << dm << std::endl;
                for ( unsigned int sample = 0; sample < obs.getNrSamplesPerBatch(); sample++ ) {
                  std::cerr << dedispersedData[(sBeam * obs.getNrDMs() * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(outputDataType))) + (dm * obs.getNrSamplesPerPaddedBatch(padding[deviceName] / sizeof(outputDataType))) + sample] << " ";
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
        clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*snrDMsSamplesK[integrationSteps.size()], cl::NullRange, snrDMsSamplesGlobal[integrationSteps.size()], snrDMsSamplesLocal[integrationSteps.size()], 0, &syncPoint);
        syncPoint.wait();
        snrDMsSamplesTimer.stop();
        outputCopyTimer.start();
        clQueues->at(clDeviceID)[0].enqueueReadBuffer(snrData_d, CL_TRUE, 0, snrData.size() * sizeof(float), reinterpret_cast< void * >(snrData.data()), 0, &syncPoint);
        syncPoint.wait();
        clQueues->at(clDeviceID)[0].enqueueReadBuffer(snrSamples_d, CL_TRUE, 0, snrSamples.size() * sizeof(unsigned int), reinterpret_cast< void * >(snrSamples.data()), 0, &syncPoint);
        syncPoint.wait();
        outputCopyTimer.stop();
      } else {
        clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*snrDMsSamplesK[integrationSteps.size()], cl::NullRange, snrDMsSamplesGlobal[integrationSteps.size()], snrDMsSamplesLocal[integrationSteps.size()]);
        clQueues->at(clDeviceID)[0].enqueueReadBuffer(snrData_d, CL_FALSE, 0, snrData.size() * sizeof(float), reinterpret_cast< void * >(snrData.data()));
        clQueues->at(clDeviceID)[0].enqueueReadBuffer(snrSamples_d, CL_FALSE, 0, snrSamples.size() * sizeof(unsigned int), reinterpret_cast< void * >(snrSamples.data()));
        clQueues->at(clDeviceID)[0].finish();
      }
    } catch ( cl::Error & err ) {
      std::cerr << "SNR dedispersed data error -- Batch: " << std::to_string(batch) << ", " << err.what() << " " << err.err() << std::endl;
      errorDetected = true;
    }
    triggerTimer.start();
    trigger(subbandDedispersion, padding[deviceName], 0, threshold, obs, snrData, snrSamples, triggeredEvents);
    triggerTimer.stop();
    if ( DEBUG ) {
      if ( print ) {
        if ( subbandDedispersion ) {
          std::cerr << "snrData" << std::endl;
          for ( unsigned int sBeam = 0; sBeam < obs.getNrSynthesizedBeams(); sBeam++ ) {
            std::cerr << "sBeam: " << sBeam << std::endl;
            for ( unsigned int subbandingDM = 0; subbandingDM < obs.getNrDMsSubbanding(); subbandingDM++ ) {
              for ( unsigned int dm = 0; dm < obs.getNrDMs(); dm++ ) {
                std::cerr << snrData[(sBeam * isa::utils::pad(obs.getNrDMsSubbanding() * obs.getNrDMs(), padding[deviceName] / sizeof(float))) + (subbandingDM * obs.getNrDMs()) + dm] << " ";
              }
            }
            std::cerr << std::endl;
          }
        } else {
          std::cerr << "snrData" << std::endl;
          for ( unsigned int sBeam = 0; sBeam < obs.getNrSynthesizedBeams(); sBeam++ ) {
            std::cerr << "sBeam: " << sBeam << std::endl;
            for ( unsigned int dm = 0; dm < obs.getNrDMs(); dm++ ) {
              std::cerr << snrData[(sBeam * obs.getNrPaddedDMs(padding[deviceName] / sizeof(float))) + dm] << " ";
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
          clQueues->at(clDeviceID)[0].enqueueReadBuffer(snrSamples_d, CL_TRUE, 0, snrSamples.size() * sizeof(unsigned int), reinterpret_cast< void * >(snrSamples.data()), 0, &syncPoint);
          syncPoint.wait();
          outputCopyTimer.stop();
        } else {
          clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*integrationDMsSamplesK[stepNumber], cl::NullRange, integrationGlobal[stepNumber], integrationLocal[stepNumber]);
          clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*snrDMsSamplesK[stepNumber], cl::NullRange, snrDMsSamplesGlobal[stepNumber], snrDMsSamplesLocal[stepNumber]);
          clQueues->at(clDeviceID)[0].enqueueReadBuffer(snrData_d, CL_FALSE, 0, snrData.size() * sizeof(float), reinterpret_cast< void * >(snrData.data()));
          clQueues->at(clDeviceID)[0].enqueueReadBuffer(snrSamples_d, CL_FALSE, 0, snrSamples.size() * sizeof(unsigned int), reinterpret_cast< void * >(snrSamples.data()));
          clQueues->at(clDeviceID)[0].finish();
        }
      } catch ( cl::Error & err ) {
        std::cerr << "SNR integration loop error -- Batch: " << std::to_string(batch) << ", Step: " << std::to_string(*step) << ", " << err.what() << " " << err.err() << std::endl;
        errorDetected = true;
      }
      if ( DEBUG ) {
        if ( print ) {
          try {
            clQueues->at(clDeviceID)[0].enqueueReadBuffer(integratedData_d, CL_TRUE, 0, integratedData.size() * sizeof(outputDataType), reinterpret_cast< void * >(integratedData.data()), 0, &syncPoint);
            syncPoint.wait();
          } catch ( cl::Error & err ) {
            std::cerr << "Impossible to read integratedData_d: " << err.what() << " " << err.err() << std::endl;
            errorDetected = true;
          }
          std::cerr << "integratedData" << std::endl;
          if ( subbandDedispersion ) {
            for ( unsigned int sBeam = 0; sBeam < obs.getNrSynthesizedBeams(); sBeam++ ) {
              std::cerr << "sBeam: " << sBeam << std::endl;
              for ( unsigned int subbandingDM = 0; subbandingDM < obs.getNrDMsSubbanding(); subbandingDM++ ) {
                for ( unsigned int dm = 0; dm < obs.getNrDMs(); dm++ ) {
                  std::cerr << "DM: " << (subbandingDM * obs.getNrDMs()) + dm << std::endl;
                  for ( unsigned int sample = 0; sample < obs.getNrSamplesPerBatch() / *step; sample++ ) {
                    std::cerr << integratedData[(sBeam * obs.getNrDMsSubbanding() * obs.getNrDMs() * isa::utils::pad(obs.getNrSamplesPerBatch() / *(integrationSteps.begin()), padding[deviceName] / sizeof(outputDataType))) + (subbandingDM * obs.getNrDMs() * isa::utils::pad(obs.getNrSamplesPerBatch() / *(integrationSteps.begin()), padding[deviceName] / sizeof(outputDataType))) + (dm * isa::utils::pad(obs.getNrSamplesPerBatch() / *(integrationSteps.begin()), padding[deviceName] / sizeof(outputDataType))) + sample] << " ";
                  }
                  std::cerr << std::endl;
                }
              }
              std::cerr << std::endl;
            }
          } else {
            for ( unsigned int sBeam = 0; sBeam < obs.getNrSynthesizedBeams(); sBeam++ ) {
              std::cerr << "sBeam: " << sBeam << std::endl;
              for ( unsigned int dm = 0; dm < obs.getNrDMs(); dm++ ) {
                std::cerr << "DM: " << dm << std::endl;
                for ( unsigned int sample = 0; sample < obs.getNrSamplesPerBatch() / *step; sample++ ) {
                  std::cerr << integratedData[(sBeam * obs.getNrDMs() * isa::utils::pad(obs.getNrSamplesPerBatch() / *(integrationSteps.begin()), padding[deviceName] / sizeof(outputDataType))) + (dm * isa::utils::pad(obs.getNrSamplesPerBatch() / *(integrationSteps.begin()), padding[deviceName] / sizeof(outputDataType))) + sample] << " ";
                }
                std::cerr << std::endl;
              }
              std::cerr << std::endl;
            }
          }
        }
      }
      triggerTimer.start();
      trigger(subbandDedispersion, padding[deviceName], *step, threshold, obs, snrData, snrSamples, triggeredEvents);
      triggerTimer.stop();
      if ( DEBUG ) {
        if ( print ) {
          if ( subbandDedispersion ) {
            std::cerr << "snrData" << std::endl;
            for ( unsigned int sBeam = 0; sBeam < obs.getNrSynthesizedBeams(); sBeam++ ) {
              std::cerr << "sBeam: " << sBeam << std::endl;
              for ( unsigned int subbandingDM = 0; subbandingDM < obs.getNrDMsSubbanding(); subbandingDM++ ) {
                for ( unsigned int dm = 0; dm < obs.getNrDMs(); dm++ ) {
                  std::cerr << snrData[(sBeam * isa::utils::pad(obs.getNrDMsSubbanding() * obs.getNrDMs(), padding[deviceName] / sizeof(float))) + (subbandingDM * obs.getNrDMs()) + dm] << " ";
                }
              }
              std::cerr << std::endl;
            }
          } else {
            std::cerr << "snrData" << std::endl;
            for ( unsigned int sBeam = 0; sBeam < obs.getNrSynthesizedBeams(); sBeam++ ) {
              std::cerr << "sBeam: " << sBeam << std::endl;
              for ( unsigned int dm = 0; dm < obs.getNrDMs(); dm++ ) {
                std::cerr << snrData[(sBeam * obs.getNrPaddedDMs(padding[deviceName] / sizeof(float))) + dm] << " ";
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
      if ( dataPSRDADA ) {
        if ( dada_hdu_unlock_read(ringBuffer) != 0 ) {
          std::cerr << "Impossible to unlock the PSRDADA ringbuffer for reading the header." << std::endl;
        }
        dada_hdu_disconnect(ringBuffer);
      }
#endif
      return 1;
    }
    // Print and compact results
    triggerTimer.start();
    if ( compactResults ) {
      compact(obs, triggeredEvents, compactedEvents);
      for ( auto beamEvents = compactedEvents.begin(); beamEvents != compactedEvents.end(); ++beamEvents ) {
        for ( auto event = beamEvents->begin(); event != beamEvents->end(); ++event ) {
          unsigned int integration = 0;
          float firstDM = 0.0f;

          if ( event->integration == 0 ) {
            integration = 1;
          } else {
            integration = event->integration;
          }
          if ( subbandDedispersion ) {
            firstDM = obs.getFirstDMSubbanding();
          } else {
            firstDM = obs.getFirstDM();
          }
          output << event->beam << " " << batch << " " << event->sample  << " " << integration << " " << event->compactedIntegration << " " << ((batch * obs.getNrSamplesPerBatch()) + (event->sample * integration)) * obs.getSamplingTime() << " " << firstDM + (event->DM * obs.getDMStep()) << " " << event->compactedDMs << " " << event->SNR << std::endl;
        }
      }
    } else {
      for ( auto beamEvents = triggeredEvents.begin(); beamEvents != triggeredEvents.end(); ++beamEvents ) {
        for ( auto dmEvents = beamEvents->begin(); dmEvents != beamEvents->end(); ++dmEvents) {
          for ( auto event = dmEvents->second.begin(); event != dmEvents->second.end(); ++event ) {
            unsigned int integration = 0;
            float firstDM = 0.0f;

            if ( event->integration == 0 ) {
              integration = 1;
            } else {
              integration = event->integration;
            }
            if ( subbandDedispersion ) {
              firstDM = obs.getFirstDMSubbanding();
            } else {
              firstDM = obs.getFirstDM();
            }
            output << event->beam << " " << batch << " " << event->sample  << " " << integration << " " << ((batch * obs.getNrSamplesPerBatch()) + (event->sample * integration)) * obs.getSamplingTime() << " " << firstDM + (event->DM * obs.getDMStep()) << " " << event->SNR << std::endl;
          }
        }
      }
    }
    triggerTimer.stop();
  }
#ifdef HAVE_PSRDADA
  if ( dataPSRDADA ) {
    if ( dada_hdu_unlock_read(ringBuffer) != 0 ) {
      std::cerr << "Impossible to unlock the PSRDADA ringbuffer for reading the header." << std::endl;
    }
    dada_hdu_disconnect(ringBuffer);
  }
#endif
  output.close();
  searchTimer.stop();

  // Store statistics before shutting down
  output.open(outputFile + ".stats");
  output << std::fixed << std::setprecision(6);
  output << "# nrDMs" << std::endl;
  if ( subbandDedispersion ) {
    output << obs.getNrDMsSubbanding() * obs.getNrDMs() << std::endl;
  } else {
    output << obs.getNrDMs() << std::endl;
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

