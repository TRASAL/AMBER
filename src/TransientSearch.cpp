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

// TODO: PSRDada multibeam

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
#include <boost/mpi.hpp>

#include <configuration.hpp>

void trigger(const bool compactResults, const unsigned int second, const unsigned int integration, const float threshold, const AstroData::Observation & obs, isa::utils::Timer & timer, boost::mpi::communicator & workers, const std::vector< float > & snrData, std::ofstream & output);


int main(int argc, char * argv[]) {
  bool print = false;
	bool dataLOFAR = false;
	bool dataSIGPROC = false;
  bool dataPSRDada = false;
  bool limit = false;
  bool compactResults = false;
  uint8_t inputBits = 0;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	unsigned int bytesToSkip = 0;
  unsigned int nrThreads = 0;
  float threshold = 0.0f;
	std::string deviceName;
	std::string dataFile;
	std::string headerFile;
	std::string outputFile;
  std::string channelsFile;
  std::string integrationFile;
  std::vector< std::ofstream > output;
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
  PulsarSearch::tunedIntegrationConf integrationParameters;
  PulsarSearch::tunedSNRConf snrParameters;
  // PSRDada
  key_t dadaKey;
  dada_hdu_t * ringBuffer;
  // MPI
  boost::mpi::environment envMPI(argc, argv);
  boost::mpi::communicator workers;

	try {
		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
		deviceName = args.getSwitchArgument< std::string >("-device_name");

    AstroData::readPaddingConf(padding, args.getSwitchArgument< std::string >("-padding_file"));
    channelsFile = args.getSwitchArgument< std::string >("-zapped_channels");
    integrationFile = args.getSwitchArgument< std::string >("-integration_steps");
    PulsarSearch::readTunedDedispersionConf(dedispersionParameters, args.getSwitchArgument< std::string >("-dedispersion_file"));
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
      obs.setFrequencyRange(args.getSwitchArgument< unsigned int >("-channels"), args.getSwitchArgument< float >("-min_freq"), args.getSwitchArgument< float >("-channel_bandwidth"));
			obs.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
    } else if ( dataPSRDada ) {
      dadaKey = args.getSwitchArgument< key_t >("-dada_key");
      obs.setNrBeams(args.getSwitchArgument< unsigned int >("-beams"));
      obs.setNrSeconds(args.getSwitchArgument< unsigned int >("-seconds"));
		} else {
      obs.setNrBeams(args.getSwitchArgument< unsigned int >("-beams"));
      obs.setNrSeconds(args.getSwitchArgument< unsigned int >("-seconds"));
      obs.setFrequencyRange(args.getSwitchArgument< unsigned int >("-channels"), args.getSwitchArgument< float >("-min_freq"), args.getSwitchArgument< float >("-channel_bandwidth"));
      obs.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
      random = args.getSwitch("-random");
      width = args.getSwitchArgument< unsigned int >("-width");
      DM = args.getSwitchArgument< float >("-dm");
		}
    inputBits = args.getSwitchArgument< unsigned int >("-input_bits");
		outputFile = args.getSwitchArgument< std::string >("-output");
    unsigned int tempUInts[3] = {args.getSwitchArgument< unsigned int >("-dm_node"), 0, 0};
    float tempFloats[2] = {args.getSwitchArgument< float >("-dm_first"), args.getSwitchArgument< float >("-dm_step")};
    obs.setDMRange(tempUInts[0], tempFloats[0] + (workers.rank() * tempUInts[0] * tempFloats[1]), tempFloats[1]);
    threshold = args.getSwitchArgument< float >("-threshold");
	} catch ( isa::utils::EmptyCommandLine & err ) {
    std::cerr <<  args.getName() << " -opencl_platform ... -opencl_device ... -device_name ... -padding_file ... -zapped_channels ... -integration_steps ... -dedispersion_file ... -integration_file ... -snr_file ... [-print] [-compact_results] [-lofar] [-sigproc] [-dada] -input_bits ... -output ... -dm_node ... -dm_first ... -dm_step ... -threshold ..."<< std::endl;
    std::cerr << "\t -lofar -header ... -data ... [-limit]" << std::endl;
    std::cerr << "\t\t -limit -seconds ..." << std::endl;
    std::cerr << "\t -sigproc -header ... -data ... -seconds ... -channels ... -min_freq ... -channel_bandwidth ... -samples ..." << std::endl;
    std::cerr << "\t -dada -dada_key ... -beams ... -seconds ..." << std::endl;
    std::cerr << "\t [-random] -width ... -dm ... -beams ... -seconds ... -channels ... -min_freq ... -channel_bandwidth ... -samples ..." << std::endl;
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
      input[beam] = new std::vector< std::vector< inputDataType > * >(obs.getNrSeconds());
      AstroData::generateSinglePulse(width, DM, obs, padding[deviceName], *(input[beam]), inputBits, random);
    }
  }
  AstroData::readZappedChannels(obs, channelsFile, zappedChannels);
  AstroData::readIntegrationSteps(obs, integrationFile, integrationSteps);
	if ( DEBUG && workers.rank() == 0 ) {
    std::cout << "Device: " << deviceName << std::endl;
    std::cout << "Padding: " << padding[deviceName] << " bytes" << std::endl;
    std::cout << std::endl;
    std::cout << "Beams: " << obs.getNrBeams() << std::endl;
    std::cout << "Seconds: " << obs.getNrSeconds() << std::endl;
    std::cout << "Samples: " << obs.getNrSamplesPerSecond() << std::endl;
    std::cout << "Frequency range: " << obs.getMinFreq() << " MHz, " << obs.getMaxFreq() << " MHz" << std::endl;
    std::cout << "Channels: " << obs.getNrChannels() << " (" << obs.getChannelBandwidth() << " MHz)" << std::endl;
    std::cout << "Zapped Channels: " << obs.getNrZappedChannels() << std::endl;
    std::cout << "Integration steps: " << integrationSteps.size() << std::endl;
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
  std::vector< float > * shifts = PulsarSearch::getShifts(obs, padding[deviceName]);
  obs.setNrSamplesPerDispersedChannel(obs.getNrSamplesPerSecond() + static_cast< unsigned int >(shifts->at(0) * (obs.getFirstDM() + ((obs.getNrDMs() - 1) * obs.getDMStep()))));
  obs.setNrDelaySeconds(static_cast< unsigned int >(std::ceil(static_cast< double >(obs.getNrSamplesPerDispersedChannel()) / obs.getNrSamplesPerSecond())));
  std::vector< std::vector< inputDataType > > dispersedData(obs.getNrBeams());
  std::vector< std::vector< outputDataType > > dedispersedData(obs.getNrBeams());
  std::vector< std::vector< outputDataType > > integratedData(obs.getNrBeams());
  std::vector< std::vector< float > > snrData(obs.getNrBeams());

  for ( unsigned int beam = 0; beam < obs.getNrBeams(); beam++ ) {
    if ( !dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitSeconds() ) {
      if ( inputBits >= 8 ) {
        dispersedData[beam] = std::vector< inputDataType >(obs.getNrChannels() * obs.getNrSamplesPerPaddedDispersedChannel(padding[deviceName] / sizeof(inputDataType)));
      } else {
        dispersedData[beam] = std::vector< inputDataType >(obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType)));
      }
    }
    dedispersedData[beam] = std::vector< outputDataType >(obs.getNrDMs() * obs.getNrSamplesPerPaddedSecond(padding[deviceName] / sizeof(outputDataType)));
    integratedData[beam] = std::vector< outputDataType >(obs.getNrDMs() * isa::utils::pad(obs.getNrSamplesPerSecond() / *(--(integrationSteps.end())), padding[deviceName] / sizeof(outputDataType)));
    snrData[beam] = std::vector< float >(obs.getNrPaddedDMs(padding[deviceName] / sizeof(float)));
  }

  if ( obs.getNrDelaySeconds() >obs.getNrSeconds() ) {
    std::cerr << "Not enough seconds in input." << std::endl;
    return 1;
  }

  // Device memory allocation and data transfers
  cl::Buffer shifts_d;
  cl::Buffer zappedChannels_d;
  std::vector< std::vector< cl::Buffer > > dispersedData_d(obs.getNrBeams());
  std::vector< cl::Buffer > dedispersedData_d(obs.getNrBeams()), integratedData_d(obs.getNrBeams()), snrData_d(obs.getNrBeams());

  try {
    shifts_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, shifts->size() * sizeof(float), 0, 0);
    zappedChannels_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, zappedChannels.size() * sizeof(uint8_t), 0, 0);
    for ( unsigned int beam = 0; beam < obs.getNrBeams(); beam++ ) {
      dispersedData_d[beam] = std::vector< cl::Buffer >(obs.getNrDelaySeconds() + 1);
      if ( dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitSeconds() ) {
        if ( inputBits >= 8 ) {
          dispersedData_d[beam][obs.getNrDelaySeconds()] = cl::Buffer(*clContext, CL_MEM_READ_ONLY, obs.getNrDelaySeconds() * obs.getNrChannels() * obs.getNrSamplesPerPaddedSecond(padding[deviceName] / sizeof(inputDataType)) * sizeof(inputDataType), 0, 0);
        } else {
          dispersedData_d[beam][obs.getNrDelaySeconds()] = cl::Buffer(*clContext, CL_MEM_READ_ONLY, obs.getNrDelaySeconds() * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerSecond() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType)) * sizeof(inputDataType), 0, 0);
        }
        for ( unsigned int second = 0; second < obs.getNrDelaySeconds(); second++ ) {
          cl_buffer_region offsets;
          cl_int err = 0;

          if ( inputBits >= 8 ) {
            offsets.origin = second * obs.getNrChannels() * obs.getNrSamplesPerPaddedSecond(padding[deviceName] / sizeof(inputDataType)) * sizeof(inputDataType);
            offsets.size = obs.getNrChannels() * obs.getNrSamplesPerPaddedSecond(padding[deviceName] / sizeof(inputDataType)) * sizeof(inputDataType);
          } else {
            offsets.origin = second * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerSecond() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType)) * sizeof(inputDataType);
            offsets.size = obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerSecond() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType)) * sizeof(inputDataType);
          }
          dispersedData_d[beam][second] = dispersedData_d[beam][obs.getNrDelaySeconds()].createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, reinterpret_cast< void * >(&offsets), &err);
          if ( err != CL_SUCCESS ) {
            std::cerr << "Error allocating sub buffers." << std::endl;
            return 1;
          }
        }
      } else {
        dispersedData_d[beam][obs.getNrDelaySeconds()] = cl::Buffer(*clContext, CL_MEM_READ_ONLY, dispersedData[beam].size() * sizeof(inputDataType), 0, 0);
      }
      dedispersedData_d[beam] = cl::Buffer(*clContext, CL_MEM_READ_WRITE, dedispersedData[beam].size() * sizeof(outputDataType), 0, 0);
      integratedData_d[beam] = cl::Buffer(*clContext, CL_MEM_READ_WRITE, integratedData[beam].size() * sizeof(outputDataType), 0, 0);
      snrData_d[beam] = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, snrData[beam].size() * sizeof(float), 0, 0);
    }
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(shifts_d, CL_FALSE, 0, shifts->size() * sizeof(float), reinterpret_cast< void * >(shifts->data()));
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(zappedChannels_d, CL_FALSE, 0, zappedChannels.size() * sizeof(uint8_t), reinterpret_cast< void * >(zappedChannels.data()));
  } catch ( cl::Error & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

	if ( DEBUG && workers.rank() == 0 ) {
    std::cout << std::fixed << std::setprecision(3);
    if ( dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitSeconds() ) {
      if ( inputBits >= 8 ) {
        std::cout << "dispersedData: " << isa::utils::giga(static_cast< double >(obs.getNrBeams()) * obs.getNrDelaySeconds() * obs.getNrChannels() * obs.getNrSamplesPerPaddedSecond(padding[deviceName] / sizeof(inputDataType)) * sizeof(inputDataType)) << " GB" << std::endl;
      } else {
        std::cout << "dispersedData: " << isa::utils::giga(static_cast< double >(obs.getNrBeams()) * obs.getNrDelaySeconds() * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerSecond() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType)) * sizeof(inputDataType)) << " GB" << std::endl;
      }
    } else {
      std::cout << "dispersedData: " << isa::utils::giga(static_cast< double >(obs.getNrBeams()) * dispersedData[0].size() * sizeof(inputDataType)) << " GB" << std::endl;
    }
    std::cout << "snrData: " << isa::utils::giga(static_cast< double >(obs.getNrBeams()) * snrData[0].size() * sizeof(float)) << " GB" << std::endl;
    std::cout << "shifts: " << isa::utils::giga(static_cast< double >(shifts->size()) * sizeof(float)) << " GB" << std::endl;
    std::cout << "zappedChannels: " << isa::utils::giga(static_cast< double >(zappedChannels.size()) * sizeof(uint8_t)) << " GB" << std::endl;
    std::cout << "dedispersedData: " << isa::utils::giga(static_cast< double >(obs.getNrBeams()) * obs.getNrDMs() * obs.getNrSamplesPerPaddedSecond(padding[deviceName] / sizeof(outputDataType)) * sizeof(outputDataType)) << " GB" << std::endl;
    std::cout << "integratedData: " << isa::utils::giga(static_cast< double >(obs.getNrBeams()) * obs.getNrDMs() * isa::utils::pad(obs.getNrSamplesPerSecond() / *(--(integrationSteps.end())), padding[deviceName] / sizeof(outputDataType))) << " GB" << std::endl;
    std::cout << std::endl;
	}

	// Generate OpenCL kernels
  std::string * code;
  std::vector< cl::Kernel * > dedispersionK(obs.getNrBeams());
  std::vector< std::vector< cl::Kernel * > > integrationDMsSamplesK(obs.getNrBeams()), snrDMsSamplesK(obs.getNrBeams());

  code = PulsarSearch::getDedispersionOpenCL< inputDataType, outputDataType >(*(dedispersionParameters.at(deviceName)->at(obs.getNrDMs())), padding[deviceName], inputBits, inputDataName, intermediateDataName, outputDataName, obs, *shifts, zappedChannels);
	try {
    for ( unsigned int beam = 0; beam < obs.getNrBeams(); beam++ ) {
      dedispersionK[beam] = isa::OpenCL::compile("dedispersion", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
      if ( dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitSeconds() ) {
        dedispersionK[beam]->setArg(1, dispersedData_d[beam][obs.getNrDelaySeconds()]);
        dedispersionK[beam]->setArg(2, dedispersedData_d[beam]);
        dedispersionK[beam]->setArg(3, shifts_d);
        dedispersionK[beam]->setArg(4, zappedChannels_d);
      } else {
        dedispersionK[beam]->setArg(0, dispersedData_d[beam][obs.getNrDelaySeconds()]);
        dedispersionK[beam]->setArg(1, dedispersedData_d[beam]);
        dedispersionK[beam]->setArg(2, shifts_d);
        dedispersionK[beam]->setArg(3, zappedChannels_d);
      }
    }
	} catch ( isa::OpenCL::OpenCLError & err ) {
    std::cerr << err.what() << std::endl;
		return 1;
	}
  delete shifts;
  delete code;
  code = PulsarSearch::getSNRDMsSamplesOpenCL< outputDataType >(*(snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerSecond())), outputDataName, obs.getNrSamplesPerSecond(), padding[deviceName]);
  try {
    for ( unsigned int beam = 0; beam < obs.getNrBeams(); beam++ ) {
      snrDMsSamplesK[beam] = std::vector< cl::Kernel * >(integrationSteps.size() + 1);
      snrDMsSamplesK[beam][integrationSteps.size()] = isa::OpenCL::compile("snrDMsSamples" + isa::utils::toString(obs.getNrSamplesPerSecond()), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
      snrDMsSamplesK[beam][integrationSteps.size()]->setArg(0, dedispersedData_d[beam]);
      snrDMsSamplesK[beam][integrationSteps.size()]->setArg(1, snrData_d[beam]);
    }
  } catch ( isa::OpenCL::OpenCLError & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }
  delete code;
  for ( unsigned int beam = 0; beam < obs.getNrBeams(); beam++ ) {
    integrationDMsSamplesK[beam] = std::vector< cl::Kernel * >(integrationSteps.size());
    for ( unsigned int stepNumber = 0; stepNumber < integrationSteps.size(); stepNumber++ ) {
      auto step = integrationSteps.begin();

      for ( unsigned int i = 0; i < stepNumber; i++ ) {
        ++step;
      }
      code = PulsarSearch::getIntegrationDMsSamplesOpenCL< outputDataType >(*(integrationParameters[deviceName]->at(obs.getNrSamplesPerSecond())->at(*step)), obs.getNrSamplesPerSecond(), outputDataName, *step, padding[deviceName]);
      try {
        if ( *step > 1 ) {
          integrationDMsSamplesK[beam][stepNumber] = isa::OpenCL::compile("integrationDMsSamples" + isa::utils::toString(*step), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
          integrationDMsSamplesK[beam][stepNumber]->setArg(0, dedispersedData_d[beam]);
          integrationDMsSamplesK[beam][stepNumber]->setArg(1, integratedData_d[beam]);
        }
      } catch ( isa::OpenCL::OpenCLError & err ) {
        std::cerr << err.what() << std::endl;
        return 1;
      }
      delete code;
      code = PulsarSearch::getSNRDMsSamplesOpenCL< outputDataType >(*(snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerSecond() / *step)), outputDataName, obs.getNrSamplesPerSecond() / *step, padding[deviceName]);
      try {
        snrDMsSamplesK[beam][stepNumber] = isa::OpenCL::compile("snrDMsSamples" + isa::utils::toString(obs.getNrSamplesPerSecond() / *step), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
        snrDMsSamplesK[beam][stepNumber]->setArg(0, integratedData_d[beam]);
        snrDMsSamplesK[beam][stepNumber]->setArg(1, snrData_d[beam]);
      } catch ( isa::OpenCL::OpenCLError & err ) {
        std::cerr << err.what() << std::endl;
        return 1;
      }
      delete code;
    }
  }

  // Set execution parameters
  nrThreads = obs.getNrSamplesPerSecond() / dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrItemsD0();
  cl::NDRange dedispersionGlobal(nrThreads, obs.getNrDMs() / dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrItemsD1());
  cl::NDRange dedispersionLocal(dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrThreadsD0(), dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrThreadsD1());
  if ( DEBUG && workers.rank() == 0 ) {
    std::cout << "Dedispersion" << std::endl;
    std::cout << "Global: " << nrThreads << ", " << obs.getNrDMs() / dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrItemsD1() << std::endl;
    std::cout << "Local: " << dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrThreadsD0() << ", " << dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getNrThreadsD1() << std::endl;
    std::cout << "Parameters: ";
    std::cout << dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->print() << std::endl;
    std::cout << std::endl;
  }
  std::vector< cl::NDRange > integrationGlobal(integrationSteps.size());
  std::vector< cl::NDRange > integrationLocal(integrationSteps.size());
  std::vector< cl::NDRange > snrDMsSamplesGlobal(integrationSteps.size());
  std::vector< cl::NDRange > snrDMsSamplesLocal(integrationSteps.size());
  nrThreads = snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerSecond())->getNrThreadsD0();
  snrDMsSamplesGlobal[integrationSteps.size()] = cl::NDRange(nrThreads, obs.getNrDMs());
  snrDMsSamplesLocal[integrationSteps.size()] = cl::NDRange(nrThreads, 1);
  if ( DEBUG && workers.rank() == 0 ) {
    std::cout << "SNRDMsSamples (" + isa::utils::toString(obs.getNrSamplesPerSecond()) + ")" << std::endl;
    std::cout << "Global: " << nrThreads << ", " << obs.getNrDMs() << std::endl;
    std::cout << "Local: " << nrThreads << ", 1" << std::endl;
    std::cout << "Parameters: ";
    std::cout << snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerSecond())->print() << std::endl;
    std::cout << std::endl;
  }
  for ( unsigned int stepNumber = 0; stepNumber < integrationSteps.size(); stepNumber++ ) {
    auto step = integrationSteps.begin();

    for ( unsigned int i = 0; i < stepNumber; i++ ) {
      ++step;
    }
    nrThreads = integrationParameters[deviceName]->at(obs.getNrSamplesPerSecond())->at(*step)->getNrThreadsD0() * ((obs.getNrSamplesPerSecond() / *step) / integrationParameters[deviceName]->at(obs.getNrSamplesPerSecond())->at(*step)->getNrItemsD0());
    integrationGlobal[stepNumber] = cl::NDRange(nrThreads, obs.getNrDMs());
    integrationLocal[stepNumber] = cl::NDRange(integrationParameters[deviceName]->at(obs.getNrSamplesPerSecond())->at(*step)->getNrThreadsD0(), 1);
    if ( DEBUG && workers.rank() == 0 ) {
      std::cout << "integrationDMsSamples (" + isa::utils::toString(*step) + ")" << std::endl;
      std::cout << "Global: " << nrThreads << ", " << obs.getNrDMs() << std::endl;
      std::cout << "Local: " << integrationParameters[deviceName]->at(obs.getNrSamplesPerSecond())->at(*step)->getNrThreadsD0() << ", 1" << std::endl;
      std::cout << "Parameters: ";
      std::cout << integrationParameters[deviceName]->at(obs.getNrSamplesPerSecond())->at(*step)->print() << std::endl;
      std::cout << std::endl;
    }
    nrThreads = snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerSecond() / *step)->getNrThreadsD0();
    snrDMsSamplesGlobal[stepNumber] = cl::NDRange(nrThreads, obs.getNrDMs());
    snrDMsSamplesLocal[stepNumber] = cl::NDRange(nrThreads, 1);
    if ( DEBUG && workers.rank() == 0 ) {
      std::cout << "SNRDMsSamples (" + isa::utils::toString(obs.getNrSamplesPerSecond() / *step) + ")" << std::endl;
      std::cout << "Global: " << nrThreads << ", " << obs.getNrDMs() << std::endl;
      std::cout << "Local: " << nrThreads << ", 1" << std::endl;
      std::cout << "Parameters: ";
      std::cout << snrParameters.at(deviceName)->at(obs.getNrDMs())->at(obs.getNrSamplesPerSecond() / *step)->print() << std::endl;
      std::cout << std::endl;
    }
  }

	// Search loop
  isa::utils::Timer nodeTime;
  std::vector< cl::Event > syncPoint(obs.getNrBeams());
  std::vector< isa::utils::Timer > searchTime(obs.getNrBeams()), inputHandlingTime(obs.getNrBeams()), inputCopyTime(obs.getNrBeams()), dedispTime(obs.getNrBeams()), integrationTime(obs.getNrBeams()), snrDMsSamplesTime(obs.getNrBeams()), outputCopyTime(obs.getNrBeams()), triggerTime(obs.getNrBeams());

  output = std::vector< std::ofstream >(obs.getNrBeams());
  for ( unsigned int beam = 0; beam < obs.getNrBeams(); beam++ ) {
    output[beam].open(outputFile + "_" + isa::utils::toString(workers.rank()) + "_B" + isa::utils::toString(beam) + ".trigger");
    output[beam] << "# second integration DM SNR" << std::endl;
  }
  workers.barrier();
  nodeTime.start();
  for ( unsigned int second = 0; second < obs.getNrSeconds() - obs.getNrDelaySeconds(); second++ ) {
    #pragma omp parallel for schedule(static, 1)
    for ( unsigned int beam = 0; beam < obs.getNrBeams(); beam++ ) {
      searchTime[beam].start();
      // Load the input
      if ( !dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitSeconds() ) {
        inputHandlingTime[beam].start();
        for ( unsigned int channel = 0; channel < obs.getNrChannels(); channel++ ) {
          for ( unsigned int chunk = 0; chunk < obs.getNrDelaySeconds() - 1; chunk++ ) {
            if ( inputBits >= 8 ) {
              memcpy(reinterpret_cast< void * >(&(dispersedData[beam].data()[(channel * obs.getNrSamplesPerPaddedDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + (chunk * obs.getNrSamplesPerSecond())])), reinterpret_cast< void * >(&((input[beam]->at(second + chunk))->at(channel * obs.getNrSamplesPerPaddedSecond(padding[deviceName] / sizeof(inputDataType))))), obs.getNrSamplesPerSecond() * sizeof(inputDataType));
            } else {
              memcpy(reinterpret_cast< void * >(&(dispersedData[beam].data()[(channel * isa::utils::pad(obs.getNrSamplesPerDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + (chunk * (obs.getNrSamplesPerSecond() / (8 / inputBits)))])), reinterpret_cast< void * >(&((input[beam]->at(second + chunk))->at(channel * isa::utils::pad(obs.getNrSamplesPerSecond() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))))), (obs.getNrSamplesPerSecond() / (8 / inputBits)) * sizeof(inputDataType));
            }
          }
          if ( obs.getNrSamplesPerDispersedChannel() % obs.getNrSamplesPerSecond() == 0 ) {
            if ( inputBits >= 8 ) {
              memcpy(reinterpret_cast< void * >(&(dispersedData[beam].data()[(channel * obs.getNrSamplesPerPaddedDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + ((obs.getNrDelaySeconds() - 1) * obs.getNrSamplesPerSecond())])), reinterpret_cast< void * >(&((input[beam]->at(second + (obs.getNrDelaySeconds() - 1)))->at(channel * obs.getNrSamplesPerPaddedSecond(padding[deviceName] / sizeof(inputDataType))))), obs.getNrSamplesPerDispersedChannel() * sizeof(inputDataType));
            } else {
              memcpy(reinterpret_cast< void * >(&(dispersedData[beam].data()[(channel * isa::utils::pad(obs.getNrSamplesPerDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + ((obs.getNrDelaySeconds() - 1) * (obs.getNrSamplesPerSecond() / (8 / inputBits)))])), reinterpret_cast< void * >(&((input[beam]->at(second + (obs.getNrDelaySeconds() - 1)))->at(channel * isa::utils::pad(obs.getNrSamplesPerSecond() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))))), (obs.getNrSamplesPerDispersedChannel() / (8 / inputBits)) * sizeof(inputDataType));
            }
          } else {
            if ( inputBits >= 8 ) {
              memcpy(reinterpret_cast< void * >(&(dispersedData[beam].data()[(channel * obs.getNrSamplesPerPaddedDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + ((obs.getNrDelaySeconds() - 1) * obs.getNrSamplesPerSecond())])), reinterpret_cast< void * >(&((input[beam]->at(second + (obs.getNrDelaySeconds() - 1)))->at(channel * obs.getNrSamplesPerPaddedSecond(padding[deviceName] / sizeof(inputDataType))))), (obs.getNrSamplesPerDispersedChannel() % obs.getNrSamplesPerSecond()) * sizeof(inputDataType));
            } else {
              memcpy(reinterpret_cast< void * >(&(dispersedData[beam].data()[(channel * isa::utils::pad(obs.getNrSamplesPerDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + ((obs.getNrDelaySeconds() - 1) * (obs.getNrSamplesPerSecond() / (8 / inputBits)))])), reinterpret_cast< void * >(&((input[beam]->at(second + (obs.getNrDelaySeconds() - 1)))->at(channel * isa::utils::pad(obs.getNrSamplesPerSecond() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))))), ((obs.getNrSamplesPerDispersedChannel() % obs.getNrSamplesPerSecond()) / (8 / inputBits)) * sizeof(inputDataType));
            }
          }
        }
        inputHandlingTime[beam].stop();
      }
      try {
        if ( SYNC ) {
          inputCopyTime[beam].start();
          if ( dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitSeconds() ) {
            if ( inputBits >= 8 ) {
              clQueues->at(clDeviceID)[beam].enqueueWriteBuffer(dispersedData_d[beam][second % obs.getNrDelaySeconds()], CL_TRUE, 0, obs.getNrChannels() * obs.getNrSamplesPerPaddedSecond(padding[deviceName] / sizeof(inputDataType)) * sizeof(inputDataType), reinterpret_cast< void * >(input[beam]->at(second)->data()), 0, &syncPoint[beam]);
            } else {
              clQueues->at(clDeviceID)[beam].enqueueWriteBuffer(dispersedData_d[beam][second % obs.getNrDelaySeconds()], CL_TRUE, 0, obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerSecond() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType)) * sizeof(inputDataType), reinterpret_cast< void * >(input[beam]->at(second)->data()), 0, &syncPoint[beam]);
            }
          } else {
            clQueues->at(clDeviceID)[beam].enqueueWriteBuffer(dispersedData_d[beam][obs.getNrDelaySeconds()], CL_TRUE, 0, dispersedData[beam].size() * sizeof(inputDataType), reinterpret_cast< void * >(dispersedData[beam].data()), 0, &syncPoint[beam]);
          }
          syncPoint[beam].wait();
          inputCopyTime[beam].stop();
        } else {
          if ( dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitSeconds() ) {
            if ( inputBits >= 8 ) {
              clQueues->at(clDeviceID)[beam].enqueueWriteBuffer(dispersedData_d[beam][second % obs.getNrDelaySeconds()], CL_FALSE, 0, obs.getNrChannels() * obs.getNrSamplesPerPaddedSecond(padding[deviceName] / sizeof(inputDataType)) * sizeof(inputDataType), reinterpret_cast< void * >(input[beam]->at(second)->data()));
            } else {
              clQueues->at(clDeviceID)[beam].enqueueWriteBuffer(dispersedData_d[beam][second % obs.getNrDelaySeconds()], CL_FALSE, 0, obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerSecond() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType)) * sizeof(inputDataType), reinterpret_cast< void * >(input[beam]->at(second)->data()));
            }
          } else {
            clQueues->at(clDeviceID)[beam].enqueueWriteBuffer(dispersedData_d[beam][obs.getNrDelaySeconds()], CL_FALSE, 0, dispersedData[beam].size() * sizeof(inputDataType), reinterpret_cast< void * >(dispersedData[beam].data()));
          }
        }
        if ( DEBUG && workers.rank() == 0 ) {
          if ( print ) {
            // TODO: add support for splitSeconds
            std::cout << std::fixed << std::setprecision(3);
            if ( dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitSeconds() ) {
            } else {
              for ( unsigned int channel = 0; channel < obs.getNrChannels(); channel++ ) {
                std::cout << channel << " : ";
                for ( unsigned int sample = 0; sample < obs.getNrSamplesPerDispersedChannel(); sample++ ) {
                  if ( inputBits > 8 ) {
                    std::cout << dispersedData[beam][(channel * obs.getNrSamplesPerPaddedDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + sample] << " ";
                  } else if ( inputBits == 8 ) {
                    std::cout << static_cast< float >(dispersedData[beam][(channel * obs.getNrSamplesPerPaddedDispersedChannel(padding[deviceName] / sizeof(inputDataType))) + sample]) << " ";
                  } else {
                    unsigned int byte = sample / (8 / inputBits);
                    char value = 0;
                    uint8_t firstBit = (sample % (8 / inputBits)) * inputBits;
                    char buffer = dispersedData[beam][(channel * isa::utils::pad(obs.getNrSamplesPerDispersedChannel() / (8 / inputBits), padding[deviceName] / sizeof(inputDataType))) + byte];

                    for ( uint8_t bit = 0; bit < inputBits; bit++ ) {
                      isa::utils::setBit(value, isa::utils::getBit(buffer, firstBit + bit), bit);
                    }
                    if ( inputDataName == "char" ) {
                      for ( uint8_t bit = inputBits; bit < 8; bit++ ) {
                        isa::utils::setBit(value, isa::utils::getBit(buffer, firstBit + (inputBits - 1)), bit);
                      }
                    }
                    std::cout << static_cast< float >(value) << " ";
                  }
                }
                std::cout << std::endl;
              }
            }
            std::cout << std::endl;
          }
        }
      } catch ( cl::Error & err ) {
        std::cerr << "Beam: " << isa::utils::toString(beam) << ", Second: " << isa::utils::toString(second) << ", " << err.what() << " " << err.err() << std::endl;
      }
      if ( dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitSeconds() && (second < obs.getNrDelaySeconds()) ) {
        // Not enough seconds in the buffer
        continue;
      }

      // Dedispersion
      try {
        if ( dedispersionParameters.at(deviceName)->at(obs.getNrDMs())->getSplitSeconds() ) {
          dedispersionK[beam]->setArg(0, second % obs.getNrDelaySeconds());
        }
        if ( SYNC ) {
          dedispTime[beam].start();
          clQueues->at(clDeviceID)[beam].enqueueNDRangeKernel(*dedispersionK[beam], cl::NullRange, dedispersionGlobal, dedispersionLocal, 0, &syncPoint[beam]);
          syncPoint[beam].wait();
          dedispTime[beam].stop();
        } else {
          clQueues->at(clDeviceID)[beam].enqueueNDRangeKernel(*dedispersionK[beam], cl::NullRange, dedispersionGlobal, dedispersionLocal);
        }
        if ( DEBUG && workers.rank() == 0 ) {
          if ( print ) {
            clQueues->at(clDeviceID)[beam].enqueueReadBuffer(dedispersedData_d[beam], CL_TRUE, 0, dedispersedData[beam].size() * sizeof(outputDataType), reinterpret_cast< void * >(dedispersedData[beam].data()));
            std::cout << std::fixed << std::setprecision(3);
            for ( unsigned int dm = 0; dm < obs.getNrDMs(); dm++ ) {
              std::cout << dm << " : ";
              for ( unsigned int sample = 0; sample < obs.getNrSamplesPerSecond(); sample++ ) {
                std::cout << dedispersedData[beam][(dm * obs.getNrSamplesPerPaddedSecond(padding[deviceName] / sizeof(outputDataType))) + sample] << " ";
              }
              std::cout << std::endl;
            }
            std::cout << std::endl;
          }
        }
        // SNR of dedispersed data
        if ( SYNC ) {
          snrDMsSamplesTime[beam].start();
          clQueues->at(clDeviceID)[beam].enqueueNDRangeKernel(*snrDMsSamplesK[beam][integrationSteps.size()], cl::NullRange, snrDMsSamplesGlobal[integrationSteps.size()], snrDMsSamplesLocal[integrationSteps.size()], 0, &syncPoint[beam]);
          syncPoint[beam].wait();
          snrDMsSamplesTime[beam].stop();
          outputCopyTime[beam].start();
          clQueues->at(clDeviceID)[beam].enqueueReadBuffer(snrData_d[beam], CL_TRUE, 0, snrData[beam].size() * sizeof(float), reinterpret_cast< void * >(snrData[beam].data()), 0, &syncPoint[beam]);
          syncPoint[beam].wait();
          outputCopyTime[beam].stop();
        } else {
          clQueues->at(clDeviceID)[beam].enqueueNDRangeKernel(*snrDMsSamplesK[beam][integrationSteps.size()], cl::NullRange, snrDMsSamplesGlobal[integrationSteps.size()], snrDMsSamplesLocal[integrationSteps.size()]);
          clQueues->at(clDeviceID)[beam].enqueueReadBuffer(snrData_d[beam], CL_FALSE, 0, snrData[beam].size() * sizeof(float), reinterpret_cast< void * >(snrData[beam].data()));
          clQueues->at(clDeviceID)[beam].finish();
        }
        trigger(compactResults, second, 0, threshold, obs, triggerTime[beam], workers, snrData[beam], output[beam]);
        if ( DEBUG && workers.rank() == 0 ) {
          if ( print ) {
            std::cout << std::fixed << std::setprecision(6);
            for ( unsigned int dm = 0; dm < obs.getNrDMs(); dm++ ) {
              std::cout << dm << ": " << snrData[beam][dm] << std::endl;
            }
            std::cout << std::endl;
          }
        }
        // SNR of integrated dedispersed data
        for ( unsigned int stepNumber = 0; stepNumber < integrationSteps.size(); stepNumber++ ) {
          auto step = integrationSteps.begin();

          for ( unsigned int i = 0; i < stepNumber; i++ ) {
            ++step;
          }
          if ( SYNC ) {
            integrationTime[beam].start();
            clQueues->at(clDeviceID)[beam].enqueueNDRangeKernel(*integrationDMsSamplesK[beam][stepNumber], cl::NullRange, integrationGlobal[stepNumber], integrationLocal[stepNumber], 0, &syncPoint[beam]);
            syncPoint[beam].wait();
            integrationTime[beam].stop();
            snrDMsSamplesTime[beam].start();
            clQueues->at(clDeviceID)[beam].enqueueNDRangeKernel(*snrDMsSamplesK[beam][stepNumber], cl::NullRange, snrDMsSamplesGlobal[stepNumber], snrDMsSamplesLocal[stepNumber], 0, &syncPoint[beam]);
            syncPoint[beam].wait();
            snrDMsSamplesTime[beam].stop();
            outputCopyTime[beam].start();
            clQueues->at(clDeviceID)[beam].enqueueReadBuffer(snrData_d[beam], CL_TRUE, 0, snrData[beam].size() * sizeof(float), reinterpret_cast< void * >(snrData[beam].data()), 0, &syncPoint[beam]);
            syncPoint[beam].wait();
            outputCopyTime[beam].stop();
          } else {
            clQueues->at(clDeviceID)[beam].enqueueNDRangeKernel(*integrationDMsSamplesK[beam][stepNumber], cl::NullRange, integrationGlobal[stepNumber], integrationLocal[stepNumber]);
            clQueues->at(clDeviceID)[beam].enqueueNDRangeKernel(*snrDMsSamplesK[beam][stepNumber], cl::NullRange, snrDMsSamplesGlobal[stepNumber], snrDMsSamplesLocal[stepNumber]);
            clQueues->at(clDeviceID)[beam].enqueueReadBuffer(snrData_d[beam], CL_FALSE, 0, snrData[beam].size() * sizeof(float), reinterpret_cast< void * >(snrData[beam].data()));
            clQueues->at(clDeviceID)[beam].finish();
          }
          if ( DEBUG && workers.rank() == 0 ) {
            if ( print ) {
              clQueues->at(clDeviceID)[beam].enqueueReadBuffer(integratedData_d[beam], CL_TRUE, 0, integratedData[beam].size() * sizeof(outputDataType), reinterpret_cast< void * >(integratedData[beam].data()));
              std::cout << std::fixed << std::setprecision(3);
              for ( unsigned int dm = 0; dm < obs.getNrDMs(); dm++ ) {
                std::cout << dm << " : ";
                for ( unsigned int sample = 0; sample < obs.getNrSamplesPerSecond() / *step; sample++ ) {
                  std::cout << integratedData[beam][(dm * isa::utils::pad(obs.getNrSamplesPerSecond() / *step, padding[deviceName] / sizeof(outputDataType))) + sample] << " ";
                }
                std::cout << std::endl;
              }
              std::cout << std::endl;
            }
          }
          trigger(compactResults, second, *step, threshold, obs, triggerTime[beam], workers, snrData[beam], output[beam]);
          if ( DEBUG && workers.rank() == 0 ) {
            if ( print ) {
              std::cout << std::fixed << std::setprecision(6);
              for ( unsigned int dm = 0; dm < obs.getNrDMs(); dm++ ) {
                std::cout << dm << ": " << snrData[beam][dm] << std::endl;
              }
              std::cout << std::endl;
            }
          }
        }
      } catch ( cl::Error & err ) {
        std::cerr << "Beam: " << isa::utils::toString(beam) << ", Second: " << isa::utils::toString(second) << ", " << err.what() << " " << err.err() << std::endl;
      }
      searchTime[beam].stop();
    }
  }
  nodeTime.stop();
  workers.barrier();

  if ( dataPSRDada ) {
    dada_hdu_unlock_read(ringBuffer);
    dada_hdu_disconnect(ringBuffer);
  }

  for ( unsigned int beam = 0; beam < obs.getNrBeams(); beam++ ) {
    output[beam].close();
    // Store statistics before shutting down
    output[beam].open(outputFile + "_" + isa::utils::toString(workers.rank()) + "_B" + isa::utils::toString(beam) + ".stats");
    output[beam] << "# nrDMs nodeTime searchTime inputHandlingTotal inputHandlingAvg err inputCopyTotal inputCopyAvg err dedispersionTotal dedispersionAvg err integrationTotal integrationAvg err snrDMsSamplesTotal snrDMsSamplesAvg err outputCopyTotal outputCopyAvg err triggerTimeTotal triggerTimeAvg err" << std::endl;
    output[beam] << std::fixed << std::setprecision(6);
    output[beam] << obs.getNrDMs() << " ";
    output[beam] << nodeTime.getTotalTime() << " ";
    output[beam] << searchTime[beam].getTotalTime() << " ";
    output[beam] << inputHandlingTime[beam].getTotalTime() << " " << inputHandlingTime[beam].getAverageTime() << " " << inputHandlingTime[beam].getStandardDeviation() << " ";
    output[beam] << inputCopyTime[beam].getTotalTime() << " " << inputCopyTime[beam].getAverageTime() << " " << inputCopyTime[beam].getStandardDeviation() << " ";
    output[beam] << dedispTime[beam].getTotalTime() << " " << dedispTime[beam].getAverageTime() << " " << dedispTime[beam].getStandardDeviation() << " ";
    output[beam] << integrationTime[beam].getTotalTime() << " " << integrationTime[beam].getAverageTime() << " " << integrationTime[beam].getStandardDeviation() << " ";
    output[beam] << snrDMsSamplesTime[beam].getTotalTime() << " " << snrDMsSamplesTime[beam].getAverageTime() << " " << snrDMsSamplesTime[beam].getStandardDeviation() << " ";
    output[beam] << outputCopyTime[beam].getTotalTime() << " " << outputCopyTime[beam].getAverageTime() << " " << outputCopyTime[beam].getStandardDeviation() << " ";
    output[beam] << triggerTime[beam].getTotalTime() << " " << triggerTime[beam].getAverageTime() << " " << triggerTime[beam].getStandardDeviation() << " ";
    output[beam] << std::endl;
    output[beam].close();
  }

	return 0;
}

void trigger(const bool compactResults, const unsigned int second, const unsigned int integration, const float threshold, const AstroData::Observation & obs, isa::utils::Timer & timer, boost::mpi::communicator & workers, const std::vector< float > & snrData, std::ofstream & output) {
  bool previous = false;
  unsigned int maxDM = 0;
  double maxSNR = 0.0;

  timer.start();
  for ( unsigned int dm = 0; dm < obs.getNrDMs(); dm++ ) {
    if ( compactResults ) {
      if ( snrData[dm] >= threshold ) {
        if ( previous ) {
          if ( snrData[dm] > maxSNR ) {
            maxDM = dm;
            maxSNR = snrData[dm];
          }
        } else {
          previous = true;
          maxDM = dm;
          maxSNR = snrData[dm];
        }
      } else if ( previous ) {
        output << second << " " << obs.getFirstDM() + (((workers.rank() * obs.getNrDMs()) + maxDM) * obs.getDMStep()) << " " << maxSNR << std::endl;
        previous = false;
        maxDM = 0;
        maxSNR = 0;
      }
    } else {
      if ( snrData[dm] >= threshold ) {
        output << second << " " << integration << " " << obs.getFirstDM() + (((workers.rank() * obs.getNrDMs()) + dm) * obs.getDMStep()) << " " << snrData[dm] << std::endl;
      }
    }
  }
  if ( previous ) {
    output << second << " " << integration << " " << obs.getFirstDM() + (((workers.rank() * obs.getNrDMs()) + maxDM) * obs.getDMStep()) << " " << maxSNR << std::endl;
  }
  timer.stop();
}

