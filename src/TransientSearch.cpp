// Copyright 2013 Alessio Sclocco <a.sclocco@vu.nl>
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
#include <iomanip>
#include <algorithm>
#include <cmath>
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

#include <Shifts.hpp>
#include <Dedispersion.hpp>
#include <SNR.hpp>


int main(int argc, char * argv[]) {
  bool print = false;
  bool noData = false;
  bool random = false;
	bool dataLOFAR = false;
	bool dataSIGPROC = false;
  bool limit = false;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	unsigned int bytesToSkip = 0;
  unsigned int secondsToBuffer = 0;
  unsigned int nrThreads = 0;
  unsigned int remainingSamples = 0;
  float threshold = 0.0f;
	std::string deviceName;
	std::string dataFile;
	std::string headerFile;
	std::string outputFile;
	std::ofstream output;
  isa::utils::ArgumentList args(argc, argv);
	// Observation object
  AstroData::Observation obs;
  // Fake pulsar
  unsigned int width = 0;
  float DM = 0;
  // Configurations
  AstroData::paddingConf padding;
  AstroData::vectorWidthConf vectorWidth;
  PulsarSearch::tunedDedispersionConf dedispersionParameters;
  PulsarSearch::tunedSNRDedispersedConf snrDParameters;

	// Initialize MPI
	boost::mpi::environment envMPI(argc, argv);
  boost::mpi::communicator world;

	try {
		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
		deviceName = args.getSwitchArgument< std::string >("-device_name");

    AstroData::readPaddingConf(padding, args.getSwitchArgument< std::string >("-padding_file"));
    AstroData::readVectorWidthConf(vectorWidth, args.getSwitchArgument< std::string >("-vector_file"));
    PulsarSearch::readTunedDedispersionConf(dedispersionParameters, args.getSwitchArgument< std::string >("-dedispersion_file"));
    PulsarSearch::readTunedSNRDedispersedConf(snrDParameters, args.getSwitchArgument< std::string >("-snr_file"));

    print = args.getSwitch("-print");
		obs.setPadding(padding[deviceName]);

		dataLOFAR = args.getSwitch("-lofar");
		dataSIGPROC = args.getSwitch("-sigproc");
		if ( dataLOFAR && dataSIGPROC ) {
			std::cerr << "-lofar and -sigproc are mutually exclusive." << std::endl;
			throw std::exception();
		} else if ( dataLOFAR ) {
			headerFile = args.getSwitchArgument< std::string >("-header");
			dataFile = args.getSwitchArgument< std::string >("-data");
      limit = args.getSwitch("-limit");
      if ( limit ) {
        obs.setNrSeconds(args.getSwitchArgument< unsigned int >("-seconds"));
      }
		} else if ( dataSIGPROC ) {
			bytesToSkip = args.getSwitchArgument< unsigned int >("-header");
			dataFile = args.getSwitchArgument< std::string >("-data");
			obs.setNrSeconds(args.getSwitchArgument< unsigned int >("-seconds"));
      obs.setFrequencyRange(args.getSwitchArgument< unsigned int >("-channels"), args.getSwitchArgument< float >("-min_freq"), args.getSwitchArgument< float >("-channel_bandwidth"));
			obs.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
		} else {
      noData = args.getSwitch("-no_data");
      if ( !noData ) {
        random = args.getSwitch("-random");
        width = args.getSwitchArgument< unsigned int >("-width");
        DM = args.getSwitchArgument< float >("-dm");
      }
      obs.setNrSeconds(args.getSwitchArgument< unsigned int >("-seconds"));
      obs.setFrequencyRange(args.getSwitchArgument< unsigned int >("-channels"), args.getSwitchArgument< float >("-min_freq"), args.getSwitchArgument< float >("-channel_bandwidth"));
      obs.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
		}
		outputFile = args.getSwitchArgument< std::string >("-output");
    unsigned int tempUInts[3] = {args.getSwitchArgument< unsigned int >("-dm_node"), 0, 0};
    float tempFloats[2] = {args.getSwitchArgument< float >("-dm_first"), args.getSwitchArgument< float >("-dm_step")};
    obs.setDMRange(tempUInts[0], tempFloats[0] + (world.rank() * tempUInts[0] * tempFloats[1]), tempFloats[1]);
    threshold = args.getSwitchArgument< float >("-threshold");
	} catch ( isa::utils::EmptyCommandLine & err ) {
    std::cerr <<  args.getName() << " -opencl_platform ... -opencl_device ... -device_name ... -padding_file ... -vector_file ... -dedispersion_file ... -snr_file ... [-print] [-lofar] [-sigproc] -output ... -dm_node ... -dm_first ... -dm_step ... -threshold ..."<< std::endl;
    std::cerr << "\t -lofar -header ... -data ... [-limit]" << std::endl;
    std::cerr << "\t\t -limit -seconds ..." << std::endl;
    std::cerr << "\t -sigproc -header ... -data ... -seconds ... -channels ... -min_freq ... -channel_bandwidth ... -samples ..." << std::endl;
    std::cerr << "\t [-random] -width ... -dm ... -seconds ... -channels ... -min_freq ... -channel_bandwidth ... -samples ..." << std::endl;
    std::cerr << "\t -no_data -seconds ... -channels ... -min_freq ... -channel_bandwidth ... -samples ..." << std::endl;
    return 1;
  } catch ( std::exception & err ) {
		std::cerr << err.what() << std::endl;
		return 1;
	}

	// Load observation data
  isa::utils::Timer loadTime;
	std::vector< std::vector< dataType > * > * input = new std::vector< std::vector< dataType > * >(obs.getNrSeconds());
	if ( dataLOFAR ) {
    loadTime.start();
    if ( limit ) {
      AstroData::readLOFAR(headerFile, dataFile, obs, *input, obs.getNrSeconds());
    } else {
      AstroData::readLOFAR(headerFile, dataFile, obs, *input);
    }
    loadTime.stop();
	} else if ( dataSIGPROC ) {
    loadTime.start();
		input->resize(obs.getNrSeconds());
    AstroData::readSIGPROC(obs, bytesToSkip, dataFile, *input);
    loadTime.stop();
	} else {
    if ( noData ) {
      input->at(0) = new std::vector< dataType >(obs.getNrChannels() * obs.getNrSamplesPerPaddedSecond());
      std::fill(input->at(0)->begin(), input->at(0)->end(), 42);
      for ( unsigned int second = 1; second < obs.getNrSeconds(); second++ ) {
        input->at(second) = input->at(0);
      }
    } else {
      AstroData::generatePulsar(period, width, DM, obs, *input, random);
    }
  }
	if ( DEBUG && world.rank() == 0 ) {
    std::cout << "Device: " << deviceName << std::endl;
    std::cout << "Padding: " << padding[deviceName] << std::endl;
    std::cout << "Vector: " << vectorWidth[deviceName] << std::endl;
    std::cout << std::endl;
    std::cout << "Seconds: " << obs.getNrSeconds() << std::endl;
    std::cout << "Samples: " << obs.getNrSamplesPerSecond() << std::endl;
    std::cout << "Frequency range: " << obs.getMinFreq() << " MHz, " << obs.getMaxFreq() << " MHz" << std::endl;
    std::cout << "Channels: " << obs.getNrChannels() << " (" << obs.getChannelBandwidth() << " MHz)" << std::endl;
    std::cout << std::endl;
		std::cout << "Time to load the input: " << std::fixed << std::setprecision(6) << loadTime.getTotalTime() << " seconds." << std::endl;
    std::cout << std::endl;
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
  std::vector< float > * shifts = PulsarSearch::getShifts(obs);
  obs.setNrSamplesPerDispersedChannel(obs.getNrSamplesPerSecond() + static_cast< unsigned int >(shifts->at(0) * (obs.getFirstDM() + ((obs.getNrDMs() - 1) * obs.getDMStep()))));
  secondsToBuffer = obs.getNrSamplesPerDispersedChannel() / obs.getNrSamplesPerSecond();
  remainingSamples = obs.getNrSamplesPerDispersedChannel() % obs.getNrSamplesPerSecond();
  std::vector< dataType > dispersedData(obs.getNrChannels() * obs.getNrSamplesPerDispersedChannel());
  std::vector< dataType > dedispersedData(obs.getNrDMs() * obs.getNrSamplesPerPaddedSecond());
  std::vector< float > snrData(obs.getNrPaddedDMs());

  // Device memory allocation and data transfers
  cl::Buffer shifts_d, dispersedData_d, dedispersedData_d, snrData_d;

  try {
    shifts_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, shifts->size() * sizeof(float), 0, 0);
    dispersedData_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, dispersedData.size() * sizeof(dataType), 0, 0);
    dedispersedData_d = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, obs.getNrDMs() * obs.getNrSamplesPerPaddedSecond() * sizeof(dataType), 0, 0);
    snrData_d = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, obs.getNrPaddedDMs() * sizeof(float), 0, 0);

    // shifts_d
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(shifts_d, CL_TRUE, 0, shifts->size() * sizeof(float), reinterpret_cast< void * >(shifts->data()));
  } catch ( cl::Error & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

	if ( DEBUG && world.rank() == 0 ) {
		double hostMemory = 0.0;
		double deviceMemory = 0.0;

    hostMemory += dispersedData.size() * sizeof(dataType);
    hostMemory += snrData.size() * sizeof(float);
    deviceMemory += hostMemory;
    deviceMemory += shifts->size() * sizeof(float);
    deviceMemory += obs.getNrDMs() * obs.getNrSamplesPerPaddedSecond() * sizeof(dataType);

		std::cout << "Allocated host memory: " << std::fixed << std::setprecision(3) << isa::utils::giga(hostMemory) << " GB." << std::endl;
		std::cout << "Allocated device memory: " << std::fixed << std::setprecision(3) << isa::utils::giga(deviceMemory) << "GB." << std::endl;
    std::cout << std::endl;
	}

	// Generate OpenCL kernels
  std::string * code;
  cl::Kernel * dedispersionK, * snrDedispersedK;

  code = PulsarSearch::getDedispersionOpenCL(dedispersionParameters[deviceName][obs.getNrDMs()], dataName, obs, *shifts);
	try {
    dedispersionK = isa::OpenCL::compile("dedispersion", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
	} catch ( isa::OpenCL::OpenCLError & err ) {
    std::cerr << err.what() << std::endl;
		return 1;
	}
  delete shifts;
  delete code;
  code = PulsarSearch::getSNRDedispersedOpenCL(snrDParameters[deviceName][obs.getNrDMs()], dataName, obs);
  try {
    snrDedispersedK = isa::OpenCL::compile("snrDedispersed", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
  } catch ( isa::OpenCL::OpenCLError & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }
  delete code;

  // Set execution parameters
  if ( obs.getNrSamplesPerSecond() % (dedispersionParameters[deviceName][obs.getNrDMs()].getNrSamplesPerBlock() * dedispersionParameters[deviceName][obs.getNrDMs()].getNrSamplesPerThread()) == 0 ) {
    nrThreads = obs.getNrSamplesPerSecond() / dedispersionParameters[deviceName][obs.getNrDMs()].getNrSamplesPerThread();
  } else {
    nrThreads = obs.getNrSamplesPerPaddedSecond() / dedispersionParameters[deviceName][obs.getNrDMs()].getNrSamplesPerThread();
  }
  cl::NDRange dedispersionGlobal(nrThreads, obs.getNrDMs() / dedispersionParameters[deviceName][obs.getNrDMs()].getNrDMsPerThread());
  cl::NDRange dedispersionLocal(dedispersionParameters[deviceName][obs.getNrDMs()].getNrSamplesPerBlock(), dedispersionParameters[deviceName][obs.getNrDMs()].getNrDMsPerBlock());
  if ( DEBUG && world.rank() == 0 ) {
    std::cout << "Dedispersion" << std::endl;
    std::cout << "Global: " << nrThreads << ", " << obs.getNrDMs() / dedispersionParameters[deviceName][obs.getNrDMs()].getNrDMsPerThread() << std::endl;
    std::cout << "Local: " << dedispersionParameters[deviceName][obs.getNrDMs()].getNrSamplesPerBlock() << ", " << dedispersionParameters[deviceName][obs.getNrDMs()].getNrDMsPerBlock() << std::endl;
    std::cout << "Parameters: ";
    std::cout << dedispersionParameters[deviceName][obs.getNrDMs()].print() << std::endl;
    std::cout << std::endl;
  }
  if ( obs.getNrSamplesPerSecond() % (snrDParameters[deviceName][obs.getNrDMs()].getNrSamplesPerBlock()) == 0 ) {
    nrThreads = obs.getNrSamplesPerSecond();
  } else {
    nrThreads = obs.getNrSamplesPerPaddedSecond();
  }
  cl::NDRange snrDedispersedGlobal(nrThreads);
  cl::NDRange snrDedispersedLocal(snrDParameters[deviceName][obs.getNrDMs()].getNrSamplesPerBlock());
  if ( DEBUG && world.rank() == 0 ) {
    std::cout << "SNRDedispersed" << std::endl;
    std::cout << "Global: " << nrThreads << std::endl;
    std::cout << "Local: " << snrDParameters[deviceName][obs.getNrDMs()].getNrSamplesPerBlock() << std::endl;
    std::cout << "Parameters: ";
    std::cout << snrDParameters[deviceName][obs.getNrDMs()].print() << std::endl;
    std::cout << std::endl;
  }

  dedispersionK->setArg(0, dispersedData_d);
  dedispersionK->setArg(1, dedispersedData_d);
  dedispersionK->setArg(2, shifts_d);
  snrDedispersedK->setArg(0, dedispersedData_d);
  snrDedispersedK->setArg(1, snrData_d);

	// Search loop
  cl::Event syncPoint;
  isa::utils::Timer searchTime;
  isa::utils::Timer inputHandlingTime;
  isa::utils::Timer inputCopyTime;
  isa::utils::Timer dedispTime;
  isa::utils::Timer snrDedispersedTime;
  isa::utils::Timer outputCopyTime;
  isa::utils::Timer triggerTime;

  world.barrier();
  searchTime.start();
  output.sync_with_stdio(false);
  output.open(outputFile + "_" + isa::utils::toString(world.rank()) + ".trigger");
  output << "# DM SNR" << std::endl;
	for ( unsigned int second = 0; second < obs.getNrSeconds() - secondsToBuffer; second++ ) {
		// Load the input
    inputHandlingTime.start();
		for ( unsigned int channel = 0; channel < obs.getNrChannels(); channel++ ) {
			for ( unsigned int chunk = 0; chunk < secondsToBuffer; chunk++ ) {
        memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(channel * obs.getNrSamplesPerDispersedChannel()) + (chunk * obs.getNrSamplesPerSecond())])), reinterpret_cast< void * >(&((input->at(second + chunk))->at(channel * obs.getNrSamplesPerPaddedSecond()))), obs.getNrSamplesPerSecond() * sizeof(dataType));
			}
      memcpy(reinterpret_cast< void * >(&(dispersedData.data()[(channel * obs.getNrSamplesPerDispersedChannel()) + (secondsToBuffer * obs.getNrSamplesPerSecond())])), reinterpret_cast< void * >(&((input->at(second + secondsToBuffer))->at(channel * obs.getNrSamplesPerPaddedSecond()))), remainingSamples * sizeof(dataType));
		}
    try {
      inputCopyTime.start();
      clQueues->at(clDeviceID)[0].enqueueWriteBuffer(dispersedData_d, CL_TRUE, 0, dispersedData.size() * sizeof(dataType), reinterpret_cast< void * >(dispersedData.data()), 0, &syncPoint);
      syncPoint.wait();
      inputCopyTime.stop();
      if ( DEBUG ) {
        if ( print && world.rank() == 0 ) {
          std::cout << std::fixed << std::setprecision(3);
          for ( unsigned int channel = 0; channel < obs.getNrChannels(); channel++ ) {
            std::cout << channel << " : ";
            for ( unsigned int sample = 0; sample < obs.getNrSamplesPerDispersedChannel(); sample++ ) {
              std::cout << dispersedData[(channel * obs.getNrSamplesPerDispersedChannel()) + sample] << " ";
            }
            std::cout << std::endl;
          }
          std::cout << std::endl;
        }
      }
    } catch ( cl::Error & err ) {
      std::cerr << err.what() << std::endl;
      return 1;
    }
    inputHandlingTime.stop();

		// Run the kernels
		try {
      dedispTime.start();
      clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*dedispersionK, cl::NullRange, dedispersionGlobal, dedispersionLocal, 0, &syncPoint);
      syncPoint.wait();
      dedispTime.stop();
      if ( DEBUG ) {
        if ( print && world.rank() == 0 ) {
          clQueues->at(clDeviceID)[0].enqueueReadBuffer(dedispersedData_d, CL_TRUE, 0, dedispersedData.size() * sizeof(dataType), reinterpret_cast< void * >(dedispersedData.data()));
          std::cout << std::fixed << std::setprecision(3);
          for ( unsigned int dm = 0; dm < obs.getNrDMs(); dm++ ) {
            std::cout << dm << " : ";
            for ( unsigned int sample = 0; sample < obs.getNrSamplesPerSecond(); sample++ ) {
              std::cout << dedispersedData[(dm * obs.getNrSamplesPerPaddedSecond()) + sample] << " ";
            }
            std::cout << std::endl;
          }
          std::cout << std::endl;
        }
      }
      snrDedispersedTime.start();
      clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*snrDedispersedK, cl::NullRange, snrDedispersedGlobal, snrDedispersedLocal, 0, &syncPoint);
      syncPoint.wait();
      snrDedispersedTime.stop();
      outputCopyTime.start();
      clQueues->at(clDeviceID)[0].enqueueReadBuffer(snrData_d, CL_TRUE, 0, snrData.size() * sizeof(float), reinterpret_cast< void * >(snrData.data()), 0, &syncPoint);
      syncPoint.wait();
      outputCopyTime.stop();
      triggerTime.start();
      for ( unsigned int dm = 0; dm < obs.getNrDMs(); dm++ ) {
        if ( snrData[dm] >= threshold ) {
          output << obs.getFirstDM() + (((world.rank() * obs.getNrDMs()) + dm) * obs.getDMStep())  << " " << sndData[dm] << std::endl;
        }
      }
      triggerTime.stop();
      if ( DEBUG ) {
        if ( print && world.rank() == 0 ) {
          std::cout << std::fixed << std::setprecision(6);
          for ( unsigned int dm = 0; dm < obs.getNrDMs(); dm++ ) {
            std::cout << dm << ": " << snrData[dm] << std::endl;
          }
          std::cout << std::endl;
        }
      }
		} catch ( cl::Error & err ) {
			std::cerr << err.what() <<" "  << err.err() << std::endl;
			return 1;
		}
	}
  output.close();
  world.barrier();
  searchTime.stop();

  // Store statistics
	output.open(outputFile + "_" + isa::utils::toString(world.rank()) + ".stats");
  output << "# nrDMs searchTime inputHandlingTotal inputHandlingAvg err inputCopyTotal inputCopyAvg err dedispersionTotal dedispersionAvg err snrDedispersedTotal snrDedispersedAvg err outputCopyTotal outputCopyAvg err triggerTimeTotal triggerTimeAvg err" << std::endl;
  output << std::fixed << std::setprecision(6);
  output << obs.getNrDMs() << " ";
  output << searchTime.getTotalTime() << " ";
  output << inputHandlingTime.getTotalTime() << " " << inputHandlingTime.getAverageTime() << " " << inputHandlingTime.getStandardDeviation() << " ";
  output << inputCopyTime.getTotalTime() << " " << inputCopyTime.getAverageTime() << " " << inputCopyTime.getStandardDeviation() << " ";
  output << dedispTime.getTotalTime() << " " << dedispTime.getAverageTime() << " " << dedispTime.getStandardDeviation() << " ";
  output << snrDedispersedTime.getTotalTime() << " " << snrDedispersedTime.getAverageTime() << " " << snrDedispersedTime.getStandardDeviation() << " ";
  output << outputCopyTime.getTotalTime() << " " << outputCopyTime.getAverageTime() << " " << outputCopyTime.getStandardDeviation() << " ";
  output << triggerTime.getTotalTime() << " " << triggerTime.getAverageTime() << " " << triggerTime.getStandardDeviation() << " ";
  output << std::endl;
  output.close();

	return 0;
}

