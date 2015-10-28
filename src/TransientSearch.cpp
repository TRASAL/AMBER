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
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <boost/mpi.hpp>

#include <configuration.hpp>

#include <ArgumentList.hpp>
#include <utils.hpp>
#include <Timer.hpp>
#include <Observation.hpp>
#include <Platform.hpp>
#include <ReadData.hpp>
#include <Generator.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>


int main(int argc, char * argv[]) {
  bool print = false;
	bool dataLOFAR = false;
	bool dataSIGPROC = false;
  bool dataPSRDada = false;
  bool limit = false;
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
  PulsarSearch::tunedSNRDedispersedConf snrDParameters;
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
    PulsarSearch::readTunedDedispersionConf(dedispersionParameters, args.getSwitchArgument< std::string >("-dedispersion_file"));
    PulsarSearch::readTunedSNRDedispersedConf(snrDParameters, args.getSwitchArgument< std::string >("-snr_file"));

    print = args.getSwitch("-print");
		obs.setPadding(padding[deviceName]);

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
    std::cerr <<  args.getName() << " -opencl_platform ... -opencl_device ... -device_name ... -padding_file ... -dedispersion_file ... -snr_file ... [-print] [-lofar] [-sigproc] [-dada] -input_bits ... -output ... -dm_node ... -dm_first ... -dm_step ... -threshold ..."<< std::endl;
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
	if ( dataLOFAR ) {
    input[0] = new std::vector< std::vector< inputDataType > * >(obs.getNrSeconds());
    loadTime.start();
    if ( limit ) {
      AstroData::readLOFAR(headerFile, dataFile, obs, *(input[0]), obs.getNrSeconds());
    } else {
      AstroData::readLOFAR(headerFile, dataFile, obs, *(input[0]));
    }
    loadTime.stop();
	} else if ( dataSIGPROC ) {
    input[0] = new std::vector< std::vector< inputDataType > * >(obs.getNrSeconds());
    loadTime.start();
    AstroData::readSIGPROC(obs, inputBits, bytesToSkip, dataFile, *(input[0]));
    loadTime.stop();
  } else if ( dataPSRDada ) {
    ringBuffer = dada_hdu_create(0);
    dada_hdu_set_key(ringBuffer, dadaKey);
    dada_hdu_connect(ringBuffer);
    dada_hdu_lock_read(ringBuffer);
	} else {
    for ( unsigned int beam = 0; beam < obs.getNrBeams(); beam++ ) {
      input[beam] = new std::vector< std::vector< inputDataType > * >(obs.getNrSeconds());
      AstroData::generateSinglePulse(width, DM, obs, *(input[beam]), inputBits, random);
    }
  }
	if ( DEBUG && workers.rank() == 0 ) {
    std::cout << "Device: " << deviceName << std::endl;
    std::cout << "Padding: " << padding[deviceName] << std::endl;
    std::cout << std::endl;
    std::cout << "Beams: " << obs.getNrBeams() << std::endl;
    std::cout << "Seconds: " << obs.getNrSeconds() << std::endl;
    std::cout << "Samples: " << obs.getNrSamplesPerSecond() << std::endl;
    std::cout << "Frequency range: " << obs.getMinFreq() << " MHz, " << obs.getMaxFreq() << " MHz" << std::endl;
    std::cout << "Channels: " << obs.getNrChannels() << " (" << obs.getChannelBandwidth() << " MHz)" << std::endl;
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
  std::vector< float > * shifts = PulsarSearch::getShifts(obs);
  obs.setNrSamplesPerDispersedChannel(obs.getNrSamplesPerSecond() + static_cast< unsigned int >(shifts->at(0) * (obs.getFirstDM() + ((obs.getNrDMs() - 1) * obs.getDMStep()))));
  obs.setNrDelaySeconds(static_cast< unsigned int >(std::ceil(obs.getNrSamplesPerDispersedChannel() / obs.getNrSamplesPerSecond())));
  std::vector< std::vector< inputDataType > > dispersedData(obs.getNrBeams());
  std::vector< std::vector< outputDataType > > dedispersedData(obs.getNrBeams());
  std::vector< std::vector< float > > snrData(obs.getNrBeams());

  for ( unsigned int beam = 0; beam < obs.getNrBeams(); beam++ ) {
    if ( !dedispersionParameters[deviceName][obs.getNrDMs()].getSplitSeconds() ) {
      if ( inputBits >= 8 ) {
        dispersedData[beam] = std::vector< inputDataType >(obs.getNrChannels() * obs.getNrSamplesPerDispersedChannel());
      } else {
        dispersedData[beam] = std::vector< inputDataType >(obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerDispersedChannel() / (8 / inputBits), obs.getPadding()));
      }
    }
    dedispersedData[beam] = std::vector< outputDataType >(obs.getNrDMs() * obs.getNrSamplesPerPaddedSecond());
    snrData[beam] = std::vector< float >(obs.getNrPaddedDMs());
  }

  // Device memory allocation and data transfers
  cl::Buffer shifts_d;
  std::vector< std::vector< cl::Buffer > > dispersedData_d(obs.getNrBeams());
  std::vector< cl::Buffer > dedispersedData_d(obs.getNrBeams()), snrData_d(obs.getNrBeams());

  try {
    shifts_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, shifts->size() * sizeof(float), 0, 0);
    for ( unsigned int beam = 0; beam < obs.getNrBeams(); beam++ ) {
      dispersedData_d[beam] = std::vector< cl::Buffer >(obs.getNrDelaySeconds() + 1);
      if ( dedispersionParameters[deviceName][obs.getNrDMs()].getSplitSeconds() ) {
        if ( inputBits >= 8 ) {
          dispersedData_d[beam][obs.getNrDelaySeconds()] = cl::Buffer(*clContext, CL_MEM_READ_ONLY, obs.getNrDelaySeconds() * obs.getNrChannels() * obs.getNrSamplesPerPaddedSecond() * sizeof(inputDataType), 0, 0);
        } else {
          dispersedData_d[beam][obs.getNrDelaySeconds()] = cl::Buffer(*clContext, CL_MEM_READ_ONLY, obs.getNrDelaySeconds() * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerSecond() / (8 / inputBits), obs.getPadding()) * sizeof(inputDataType), 0, 0);
        }
        for ( unsigned int second = 0; second < obs.getNrDelaySeconds(); second++ ) {
          cl_buffer_region offsets;
          cl_int err = 0;

          if ( inputBits >= 8 ) {
            offsets.origin = second * obs.getNrChannels() * obs.getNrSamplesPerPaddedSecond() * sizeof(inputDataType);
            offsets.size = obs.getNrChannels() * obs.getNrSamplesPerPaddedSecond() * sizeof(inputDataType);
          } else {
            offsets.origin = second * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerSecond() / (8 / inputBits), obs.getPadding()) * sizeof(inputDataType);
            offsets.size = obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerSecond() / (8 / inputBits), obs.getPadding()) * sizeof(inputDataType);
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
      snrData_d[beam] = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, snrData[beam].size() * sizeof(float), 0, 0);
    }
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(shifts_d, CL_TRUE, 0, shifts->size() * sizeof(float), reinterpret_cast< void * >(shifts->data()));
  } catch ( cl::Error & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

	if ( DEBUG && workers.rank() == 0 ) {
		double hostMemory = 0.0;
		double deviceMemory = 0.0;

    if ( dedispersionParameters[deviceName][obs.getNrDMs()].getSplitSeconds() ) {
      if ( inputBits >= 8 ) {
        deviceMemory += obs.getNrBeams() * obs.getNrDelaySeconds() * obs.getNrChannels() * obs.getNrSamplesPerPaddedSecond() * sizeof(inputDataType);
      } else {
        deviceMemory += obs.getNrBeams() * obs.getNrDelaySeconds() * obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerSecond() / (8 / inputBits), obs.getPadding()) * sizeof(inputDataType);
      }
    } else {
      hostMemory += obs.getNrBeams() * dispersedData[0].size() * sizeof(inputDataType);
    }
    hostMemory += obs.getNrBeams() * snrData[0].size() * sizeof(float);
    deviceMemory += hostMemory;
    deviceMemory += shifts->size() * sizeof(float);
    deviceMemory += obs.getNrBeams() * obs.getNrDMs() * obs.getNrSamplesPerPaddedSecond() * sizeof(outputDataType);

		std::cout << "Allocated host memory: " << std::fixed << std::setprecision(3) << isa::utils::giga(hostMemory) << " GB." << std::endl;
		std::cout << "Allocated device memory: " << std::fixed << std::setprecision(3) << isa::utils::giga(deviceMemory) << "GB." << std::endl;
    std::cout << std::endl;
	}

	// Generate OpenCL kernels
  std::string * code;
  std::vector< cl::Kernel * > dedispersionK(obs.getNrBeams()), snrDedispersedK(obs.getNrBeams());

  code = PulsarSearch::getDedispersionOpenCL(dedispersionParameters[deviceName][obs.getNrDMs()], inputBits, inputDataName, intermediateDataName, outputDataName, obs, *shifts);
	try {
    for ( unsigned int beam = 0; beam < obs.getNrBeams(); beam++ ) {
      dedispersionK[beam] = isa::OpenCL::compile("dedispersion", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
      if ( dedispersionParameters[deviceName][obs.getNrDMs()].getSplitSeconds() ) {
        dedispersionK[beam]->setArg(1, dispersedData_d[beam][obs.getNrDelaySeconds()]);
        dedispersionK[beam]->setArg(2, dedispersedData_d[beam]);
        dedispersionK[beam]->setArg(3, shifts_d);
      } else {
        dedispersionK[beam]->setArg(0, dispersedData_d[beam][obs.getNrDelaySeconds()]);
        dedispersionK[beam]->setArg(1, dedispersedData_d[beam]);
        dedispersionK[beam]->setArg(2, shifts_d);
      }
    }
	} catch ( isa::OpenCL::OpenCLError & err ) {
    std::cerr << err.what() << std::endl;
		return 1;
	}
  delete shifts;
  delete code;
  code = PulsarSearch::getSNRDedispersedOpenCL(snrDParameters[deviceName][obs.getNrDMs()], outputDataName, obs);
  try {
    for ( unsigned int beam = 0; beam < obs.getNrBeams(); beam++ ) {
      snrDedispersedK[beam] = isa::OpenCL::compile("snrDedispersed", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
      snrDedispersedK[beam]->setArg(0, dedispersedData_d[beam]);
      snrDedispersedK[beam]->setArg(1, snrData_d[beam]);
    }
  } catch ( isa::OpenCL::OpenCLError & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }
  delete code;

  // Set execution parameters
  // TODO: Avoid overpadding of threads
  if ( obs.getNrSamplesPerSecond() % (dedispersionParameters[deviceName][obs.getNrDMs()].getNrSamplesPerBlock() * dedispersionParameters[deviceName][obs.getNrDMs()].getNrSamplesPerThread()) == 0 ) {
    nrThreads = obs.getNrSamplesPerSecond() / dedispersionParameters[deviceName][obs.getNrDMs()].getNrSamplesPerThread();
  } else if ( obs.getNrSamplesPerPaddedSecond() % (dedispersionParameters[deviceName][obs.getNrDMs()].getNrSamplesPerBlock() * dedispersionParameters[deviceName][obs.getNrDMs()].getNrSamplesPerThread()) == 0 ) {
    nrThreads = obs.getNrSamplesPerPaddedSecond() / dedispersionParameters[deviceName][obs.getNrDMs()].getNrSamplesPerThread();
  } else {
    nrThreads = isa::utils::pad(obs.getNrSamplesPerSecond() / dedispersionParameters[deviceName][obs.getNrDMs()].getNrSamplesPerThread(), dedispersionParameters[deviceName][obs.getNrDMs()].getNrSamplesPerBlock());
  }
  cl::NDRange dedispersionGlobal(nrThreads, obs.getNrDMs() / dedispersionParameters[deviceName][obs.getNrDMs()].getNrDMsPerThread());
  cl::NDRange dedispersionLocal(dedispersionParameters[deviceName][obs.getNrDMs()].getNrSamplesPerBlock(), dedispersionParameters[deviceName][obs.getNrDMs()].getNrDMsPerBlock());
  if ( DEBUG && workers.rank() == 0 ) {
    std::cout << "Dedispersion" << std::endl;
    std::cout << "Global: " << nrThreads << ", " << obs.getNrDMs() / dedispersionParameters[deviceName][obs.getNrDMs()].getNrDMsPerThread() << std::endl;
    std::cout << "Local: " << dedispersionParameters[deviceName][obs.getNrDMs()].getNrSamplesPerBlock() << ", " << dedispersionParameters[deviceName][obs.getNrDMs()].getNrDMsPerBlock() << std::endl;
    std::cout << "Parameters: ";
    std::cout << dedispersionParameters[deviceName][obs.getNrDMs()].print() << std::endl;
    std::cout << std::endl;
  }
  nrThreads = snrDParameters[deviceName][obs.getNrDMs()].getNrSamplesPerBlock();
  cl::NDRange snrDedispersedGlobal(nrThreads, obs.getNrDMs());
  cl::NDRange snrDedispersedLocal(snrDParameters[deviceName][obs.getNrDMs()].getNrSamplesPerBlock(), 1);
  if ( DEBUG && workers.rank() == 0 ) {
    std::cout << "SNRDedispersed" << std::endl;
    std::cout << "Global: " << nrThreads << ", " << obs.getNrDMs() << std::endl;
    std::cout << "Local: " << snrDParameters[deviceName][obs.getNrDMs()].getNrSamplesPerBlock() << ", 1" << std::endl;
    std::cout << "Parameters: ";
    std::cout << snrDParameters[deviceName][obs.getNrDMs()].print() << std::endl;
    std::cout << std::endl;
  }

	// Search loop
  isa::utils::Timer nodeTime;
  std::vector< cl::Event > syncPoint(obs.getNrBeams());
  std::vector< isa::utils::Timer > searchTime(obs.getNrBeams()), inputHandlingTime(obs.getNrBeams()), inputCopyTime(obs.getNrBeams()), dedispTime(obs.getNrBeams()), snrDedispersedTime(obs.getNrBeams()), outputCopyTime(obs.getNrBeams()), triggerTime(obs.getNrBeams());

  output = std::vector< std::ofstream >(obs.getNrBeams());
  for ( unsigned int beam = 0; beam < obs.getNrBeams(); beam++ ) {
    output[beam].open(outputFile + "_" + isa::utils::toString(workers.rank()) + "_B" + isa::utils::toString(beam) + ".trigger");
    output[beam] << "# second DM SNR" << std::endl;
  }
  workers.barrier();
  nodeTime.start();
  for ( unsigned int second = 0; second < obs.getNrSeconds() - obs.getNrDelaySeconds(); second++ ) {
    #pragma omp parallel for schedule(static, 1)
    for ( unsigned int beam = 0; beam < obs.getNrBeams(); beam++ ) {
      searchTime[beam].start();
      // Load the input
      if ( !dedispersionParameters[deviceName][obs.getNrDMs()].getSplitSeconds() ) {
        inputHandlingTime[beam].start();
        for ( unsigned int channel = 0; channel < obs.getNrChannels(); channel++ ) {
          for ( unsigned int chunk = 0; chunk < obs.getNrDelaySeconds(); chunk++ ) {
            if ( inputBits >= 8 ) {
              memcpy(reinterpret_cast< void * >(&(dispersedData[beam].data()[(channel * obs.getNrSamplesPerDispersedChannel()) + (chunk * obs.getNrSamplesPerSecond())])), reinterpret_cast< void * >(&((input[beam]->at(second + chunk))->at(channel * obs.getNrSamplesPerPaddedSecond()))), obs.getNrSamplesPerSecond() * sizeof(inputDataType));
            } else {
              memcpy(reinterpret_cast< void * >(&(dispersedData[beam].data()[(channel * isa::utils::pad(obs.getNrSamplesPerDispersedChannel() / (8 / inputBits), obs.getPadding())) + (chunk * (obs.getNrSamplesPerSecond() / (8 / inputBits)))])), reinterpret_cast< void * >(&((input[beam]->at(second + chunk))->at(channel * isa::utils::pad(obs.getNrSamplesPerSecond() / (8 / inputBits), obs.getPadding())))), (obs.getNrSamplesPerSecond() / (8 / inputBits)) * sizeof(inputDataType));
            }
          }
          if ( obs.getNrSamplesPerDispersedChannel() % obs.getNrSamplesPerSecond() != 0 ) {
            if ( inputBits >= 8 ) {
              memcpy(reinterpret_cast< void * >(&(dispersedData[beam].data()[(channel * obs.getNrSamplesPerDispersedChannel()) + (obs.getNrDelaySeconds() * obs.getNrSamplesPerSecond())])), reinterpret_cast< void * >(&((input[beam]->at(second + obs.getNrDelaySeconds()))->at(channel * obs.getNrSamplesPerPaddedSecond()))), (obs.getNrSamplesPerDispersedChannel() % obs.getNrSamplesPerSecond()) * sizeof(inputDataType));
            } else {
              memcpy(reinterpret_cast< void * >(&(dispersedData[beam].data()[(channel * isa::utils::pad(obs.getNrSamplesPerDispersedChannel() / (8 / inputBits), obs.getPadding())) + (obs.getNrDelaySeconds() * (obs.getNrSamplesPerSecond() / (8 / inputBits)))])), reinterpret_cast< void * >(&((input[beam]->at(second + obs.getNrDelaySeconds()))->at(channel * isa::utils::pad(obs.getNrSamplesPerSecond() / (8 / inputBits), obs.getPadding())))), ((obs.getNrSamplesPerDispersedChannel() % obs.getNrSamplesPerSecond()) / (8 / inputBits)) * sizeof(inputDataType));
            }
          }
        }
        inputHandlingTime[beam].stop();
      }
      try {
        if ( SYNC ) {
          inputCopyTime[beam].start();
          if ( dedispersionParameters[deviceName][obs.getNrDMs()].getSplitSeconds() ) {
            if ( inputBits >= 8 ) {
              clQueues->at(clDeviceID)[beam].enqueueWriteBuffer(dispersedData_d[beam][second % obs.getNrDelaySeconds()], CL_TRUE, 0, obs.getNrChannels() * obs.getNrSamplesPerPaddedSecond() * sizeof(inputDataType), reinterpret_cast< void * >(input[beam]->at(second)->data()), 0, &syncPoint[beam]);
            } else {
              clQueues->at(clDeviceID)[beam].enqueueWriteBuffer(dispersedData_d[beam][second % obs.getNrDelaySeconds()], CL_TRUE, 0, obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerSecond() / (8 / inputBits), obs.getPadding()) * sizeof(inputDataType), reinterpret_cast< void * >(input[beam]->at(second)->data()), 0, &syncPoint[beam]);
            }
          } else {
            clQueues->at(clDeviceID)[beam].enqueueWriteBuffer(dispersedData_d[beam][obs.getNrDelaySeconds()], CL_TRUE, 0, dispersedData[beam].size() * sizeof(inputDataType), reinterpret_cast< void * >(dispersedData[beam].data()), 0, &syncPoint[beam]);
          }
          syncPoint[beam].wait();
          inputCopyTime[beam].stop();
        } else {
          if ( dedispersionParameters[deviceName][obs.getNrDMs()].getSplitSeconds() ) {
            if ( inputBits >= 8 ) {
              clQueues->at(clDeviceID)[beam].enqueueWriteBuffer(dispersedData_d[beam][second % obs.getNrDelaySeconds()], CL_FALSE, 0, obs.getNrChannels() * obs.getNrSamplesPerPaddedSecond() * sizeof(inputDataType), reinterpret_cast< void * >(input[beam]->at(second)->data()));
            } else {
              clQueues->at(clDeviceID)[beam].enqueueWriteBuffer(dispersedData_d[beam][second % obs.getNrDelaySeconds()], CL_FALSE, 0, obs.getNrChannels() * isa::utils::pad(obs.getNrSamplesPerSecond() / (8 / inputBits), obs.getPadding()) * sizeof(inputDataType), reinterpret_cast< void * >(input[beam]->at(second)->data()));
            }
          } else {
            clQueues->at(clDeviceID)[beam].enqueueWriteBuffer(dispersedData_d[beam][obs.getNrDelaySeconds()], CL_FALSE, 0, dispersedData[beam].size() * sizeof(inputDataType), reinterpret_cast< void * >(dispersedData[beam].data()));
          }
        }
        if ( DEBUG && workers.rank() == 0 ) {
          if ( print ) {
            // TODO: add support for splitSeconds
            std::cout << std::fixed << std::setprecision(3);
            if ( dedispersionParameters[deviceName][obs.getNrDMs()].getSplitSeconds() ) {
            } else {
              for ( unsigned int channel = 0; channel < obs.getNrChannels(); channel++ ) {
                std::cout << channel << " : ";
                for ( unsigned int sample = 0; sample < obs.getNrSamplesPerDispersedChannel(); sample++ ) {
                  if ( inputBits >= 8 ) {
                    std::cout << dispersedData[beam][(channel * obs.getNrSamplesPerDispersedChannel()) + sample] << " ";
                  } else {
                    uint8_t value = 0;
                    inputDataType buffer = dispersedData[beam][(channel * isa::utils::pad(obs.getNrSamplesPerDispersedChannel() / (8 / inputBits), obs.getPadding())) + (sample / (8 / inputBits))];

                    for ( uint8_t bit = 0; bit < inputBits; bit++ ) {
                      isa::utils::setBit(value, isa::utils::getBit(buffer, (sample % (8 / inputBits)) + bit), bit);
                    }
                    std::cout << static_cast< unsigned int >(value) << " ";
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
      if ( dedispersionParameters[deviceName][obs.getNrDMs()].getSplitSeconds() && (second < obs.getNrDelaySeconds()) ) {
        // Not enough seconds in the buffer
        continue;
      }

      // Run the kernels
      try {
        if ( dedispersionParameters[deviceName][obs.getNrDMs()].getSplitSeconds() ) {
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
                std::cout << dedispersedData[beam][(dm * obs.getNrSamplesPerPaddedSecond()) + sample] << " ";
              }
              std::cout << std::endl;
            }
            std::cout << std::endl;
          }
        }
        if ( SYNC ) {
          snrDedispersedTime[beam].start();
          clQueues->at(clDeviceID)[beam].enqueueNDRangeKernel(*snrDedispersedK[beam], cl::NullRange, snrDedispersedGlobal, snrDedispersedLocal, 0, &syncPoint[beam]);
          syncPoint[beam].wait();
          snrDedispersedTime[beam].stop();
          outputCopyTime[beam].start();
          clQueues->at(clDeviceID)[beam].enqueueReadBuffer(snrData_d[beam], CL_TRUE, 0, snrData[beam].size() * sizeof(float), reinterpret_cast< void * >(snrData[beam].data()), 0, &syncPoint[beam]);
          syncPoint[beam].wait();
          outputCopyTime[beam].stop();
        } else {
          clQueues->at(clDeviceID)[beam].enqueueNDRangeKernel(*snrDedispersedK[beam], cl::NullRange, snrDedispersedGlobal, snrDedispersedLocal);
          clQueues->at(clDeviceID)[beam].enqueueReadBuffer(snrData_d[beam], CL_FALSE, 0, snrData[beam].size() * sizeof(float), reinterpret_cast< void * >(snrData[beam].data()));
          clQueues->at(clDeviceID)[beam].finish();
        }
        // Triggering
        triggerTime[beam].start();
        bool previous = false;
        unsigned int maxDM = 0;
        double maxSNR = 0.0;

        for ( unsigned int dm = 0; dm < obs.getNrDMs(); dm++ ) {
          if ( snrData[beam][dm] >= threshold ) {
            if ( !previous || snrData[beam][dm] > maxSNR ) {
              previous = true;
              maxDM = dm;
              maxSNR = snrData[beam][dm];
            }
          } else if ( previous ) {
            output[beam] << second << " " << obs.getFirstDM() + (((workers.rank() * obs.getNrDMs()) + maxDM) * obs.getDMStep()) << " " << maxSNR << std::endl;
            previous = false;
            maxDM = 0;
            maxSNR = 0.0;
          }
        }
        triggerTime[beam].stop();
        if ( DEBUG && workers.rank() == 0 ) {
          if ( print ) {
            std::cout << std::fixed << std::setprecision(6);
            for ( unsigned int dm = 0; dm < obs.getNrDMs(); dm++ ) {
              std::cout << dm << ": " << snrData[beam][dm] << std::endl;
            }
            std::cout << std::endl;
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
    output[beam] << "# nrDMs nodeTime searchTime inputHandlingTotal inputHandlingAvg err inputCopyTotal inputCopyAvg err dedispersionTotal dedispersionAvg err snrDedispersedTotal snrDedispersedAvg err outputCopyTotal outputCopyAvg err triggerTimeTotal triggerTimeAvg err" << std::endl;
    output[beam] << std::fixed << std::setprecision(6);
    output[beam] << obs.getNrDMs() << " ";
    output[beam] << nodeTime.getTotalTime() << " ";
    output[beam] << searchTime[beam].getTotalTime() << " ";
    output[beam] << inputHandlingTime[beam].getTotalTime() << " " << inputHandlingTime[beam].getAverageTime() << " " << inputHandlingTime[beam].getStandardDeviation() << " ";
    output[beam] << inputCopyTime[beam].getTotalTime() << " " << inputCopyTime[beam].getAverageTime() << " " << inputCopyTime[beam].getStandardDeviation() << " ";
    output[beam] << dedispTime[beam].getTotalTime() << " " << dedispTime[beam].getAverageTime() << " " << dedispTime[beam].getStandardDeviation() << " ";
    output[beam] << snrDedispersedTime[beam].getTotalTime() << " " << snrDedispersedTime[beam].getAverageTime() << " " << snrDedispersedTime[beam].getStandardDeviation() << " ";
    output[beam] << outputCopyTime[beam].getTotalTime() << " " << outputCopyTime[beam].getAverageTime() << " " << outputCopyTime[beam].getStandardDeviation() << " ";
    output[beam] << triggerTime[beam].getTotalTime() << " " << triggerTime[beam].getAverageTime() << " " << triggerTime[beam].getStandardDeviation() << " ";
    output[beam] << std::endl;
    output[beam].close();
  }

	return 0;
}

