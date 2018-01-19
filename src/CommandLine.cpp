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

#include <CommandLine.hpp>


void processCommandLineOptions(isa::utils::ArgumentList & argumentList, Options & options, DeviceOptions & deviceOptions, DataOptions & dataOptions, Configurations & configurations, GeneratorOptions & generatorOptions, AstroData::Observation & observation) {
  try {
    options.print = argumentList.getSwitch("-print");
    options.subbandDedispersion = argumentList.getSwitch("-subband_dedispersion");
    options.compactResults = argumentList.getSwitch("-compact_results");
    options.threshold = argumentList.getSwitchArgument<float>("-threshold");
    deviceOptions.platformID = argumentList.getSwitchArgument<unsigned int>("-opencl_platform");
    deviceOptions.deviceID = argumentList.getSwitchArgument<unsigned int>("-opencl_device");
    deviceOptions.deviceName = argumentList.getSwitchArgument<std::string>("-device_name");
    AstroData::readPaddingConf(deviceOptions.padding, argumentList.getSwitchArgument<std::string>("-padding_file"));
    dataOptions.dataLOFAR = argumentList.getSwitch("-lofar");
#ifndef HAVE_HDF5
    if (dataOptions.dataLOFAR) {
      std::cerr << "Not compiled with HDF5 support." << std::endl;
      throw std::exception();
    };
#endif // HAVE_HDF5
    dataOptions.dataPSRDADA = argumentList.getSwitch("-dada");
#ifndef HAVE_PSRDADA
    if (dataOptions.dataPSRDADA) {
      std::cerr << "Not compiled with PSRDADA support." << std::endl;
      throw std::exception();
    };
#endif // HAVE_PSRDADA
    dataOptions.dataSIGPROC = argumentList.getSwitch("-sigproc");
    if ( !((((!(dataOptions.dataLOFAR && dataOptions.dataSIGPROC) && dataOptions.dataPSRDADA) || (!(dataOptions.dataLOFAR && dataOptions.dataPSRDADA) && dataOptions.dataSIGPROC)) || (!(dataOptions.dataSIGPROC && dataOptions.dataPSRDADA) && dataOptions.dataLOFAR)) || ((!dataOptions.dataLOFAR && !dataOptions.dataSIGPROC) && !dataOptions.dataPSRDADA)) ) {
      std::cerr << "-lofar -sigproc and -dada are mutually exclusive." << std::endl;
      throw std::exception();
    }
    dataOptions.channelsFile = argumentList.getSwitchArgument<std::string>("-zapped_channels");
    dataOptions.integrationFile = argumentList.getSwitchArgument<std::string>("-integration_steps");
    if ( !options.subbandDedispersion ) {
      Dedispersion::readTunedDedispersionConf(configurations.dedispersionParameters, argumentList.getSwitchArgument<std::string>("-dedispersion_file"));
    } else {
      Dedispersion::readTunedDedispersionConf(configurations.dedispersionStepOneParameters, argumentList.getSwitchArgument<std::string>("-dedispersion_step_one_file"));
      Dedispersion::readTunedDedispersionConf(configurations.dedispersionStepTwoParameters, argumentList.getSwitchArgument<std::string>("-dedispersion_step_two_file"));
    }
    Integration::readTunedIntegrationConf(configurations.integrationParameters, argumentList.getSwitchArgument<std::string>("-integration_file"));
    SNR::readTunedSNRConf(configurations.snrParameters, argumentList.getSwitchArgument<std::string>("-snr_file"));
    if ( dataOptions.dataLOFAR ) {
      observation.setNrBeams(1);
      observation.setNrSynthesizedBeams(1);
      dataOptions.headerFile = argumentList.getSwitchArgument<std::string>("-header");
      dataOptions.dataFile = argumentList.getSwitchArgument<std::string>("-data");
      dataOptions.limit = argumentList.getSwitch("-limit");
      if ( dataOptions.limit ) {
        observation.setNrBatches(argumentList.getSwitchArgument<unsigned int>("-batches"));
      }
    } else if ( dataOptions.dataSIGPROC ) {
      observation.setNrBeams(1);
      observation.setNrSynthesizedBeams(1);
      dataOptions.headerSizeSIGPROC = argumentList.getSwitchArgument<unsigned int>("-header");
      dataOptions.dataFile = argumentList.getSwitchArgument<std::string>("-data");
      observation.setNrBatches(argumentList.getSwitchArgument<unsigned int>("-batches"));
      if ( options.subbandDedispersion ) {
        observation.setFrequencyRange(argumentList.getSwitchArgument<unsigned int>("-subbands"), argumentList.getSwitchArgument<unsigned int>("-channels"), argumentList.getSwitchArgument<float>("-min_freq"), argumentList.getSwitchArgument<float>("-channel_bandwidth"));
      } else {
        observation.setFrequencyRange(1, argumentList.getSwitchArgument<unsigned int>("-channels"), argumentList.getSwitchArgument<float>("-min_freq"), argumentList.getSwitchArgument<float>("-channel_bandwidth"));
      }
      observation.setNrSamplesPerBatch(argumentList.getSwitchArgument<unsigned int>("-samples"));
      observation.setSamplingTime(argumentList.getSwitchArgument<float>("-sampling_time"));
    } else if ( dataOptions.dataPSRDADA ) {
#ifdef HAVE_PSRDADA
      dataOptions.dadaKey = std::stoi("0x" + argumentList.getSwitchArgument<std::string>("-dada_key"), 0, 16);
      if ( options.subbandDedispersion ) {
        observation.setFrequencyRange(argumentList.getSwitchArgument<unsigned int>("-subbands"), 1, 1.0f, 1.0f);
      } else {
        observation.setFrequencyRange(1, 1, 1.0f, 1.0f);
      }
      observation.setNrBeams(argumentList.getSwitchArgument<unsigned int>("-beams"));
      observation.setNrSynthesizedBeams(argumentList.getSwitchArgument<unsigned int>("-synthesized_beams"));
      observation.setNrBatches(argumentList.getSwitchArgument<unsigned int>("-batches"));
#endif // HAVE_PSRDADA
    } else {
      observation.setNrBeams(argumentList.getSwitchArgument<unsigned int>("-beams"));
      observation.setNrSynthesizedBeams(argumentList.getSwitchArgument<unsigned int>("-synthesized_beams"));
      observation.setNrBatches(argumentList.getSwitchArgument<unsigned int>("-batches"));
      if ( options.subbandDedispersion ) {
        observation.setFrequencyRange(argumentList.getSwitchArgument<unsigned int>("-subbands"), argumentList.getSwitchArgument<unsigned int>("-channels"), argumentList.getSwitchArgument<float>("-min_freq"), argumentList.getSwitchArgument<float>("-channel_bandwidth"));
      } else {
        observation.setFrequencyRange(1, argumentList.getSwitchArgument<unsigned int>("-channels"), argumentList.getSwitchArgument<float>("-min_freq"), argumentList.getSwitchArgument<float>("-channel_bandwidth"));
      }
      observation.setNrSamplesPerBatch(argumentList.getSwitchArgument<unsigned int>("-samples"));
      observation.setSamplingTime(argumentList.getSwitchArgument<float>("-sampling_time"));
      generatorOptions.random = argumentList.getSwitch("-random");
      generatorOptions.width = argumentList.getSwitchArgument<unsigned int>("-width");
      generatorOptions.DM = argumentList.getSwitchArgument<float>("-dm");
    }
    dataOptions.outputFile = argumentList.getSwitchArgument<std::string>("-output");
    if ( options.subbandDedispersion ) {
      observation.setDMRange(argumentList.getSwitchArgument<unsigned int>("-subbanding_dms"), argumentList.getSwitchArgument<float>("-subbanding_dm_first"), argumentList.getSwitchArgument<float>("-subbanding_dm_step"), true);
    }
    observation.setDMRange(argumentList.getSwitchArgument<unsigned int>("-dms"), argumentList.getSwitchArgument<float>("-dm_first"), argumentList.getSwitchArgument<float>("-dm_step"));
  } catch ( isa::utils::EmptyCommandLine & err ) {
    usage(argumentList.getName());
    throw;
  } catch ( isa::utils::SwitchNotFound & err ) {
    std::cerr << err.what() << std::endl;
    throw;
  } catch ( std::exception & err ) {
    std::cerr << "Unknown error: " << err.what() << std::endl;
    throw;
  }
}

void usage(const std::string & program) {
    std::cerr << program << " -opencl_platform ... -opencl_device ... -device_name ... -padding_file ... -zapped_channels ... -integration_steps ... -integration_file ... -snr_file ... [-subband_dedispersion] [-print] [-compact_results] -output ... -dms ... -dm_first ... -dm_step ... -threshold ... [-sigproc]";
#ifdef HAVE_HDF5
    std::cerr << " [-lofar]";
#endif // HAVE_HDF5
#ifdef HAVE_PSRDADA
    std::cerr << " [-dada]";
#endif // HAVE_PSRDADA
    std::cerr << std::endl;
    std::cerr << "\tDedispersion: -dedispersion_file ..." << std::endl;
    std::cerr << "\tSubband Dedispersion: -subband_dedispersion -dedispersion_step_one_file ... -dedispersion_step_two_file ... -subbands ... -subbanding_dms ... -subbanding_dm_first ... -subbanding_dm_step ..." << std::endl;
#ifdef HAVE_HDF5
    std::cerr << "\tLOFAR: -lofar -header ... -data ... [-limit]" << std::endl;
    std::cerr << "\t\t -limit -batches ..." << std::endl;
#endif // HAVE_HDF5
    std::cerr << "\tSIGPROC: -sigproc -header ... -data ... -batches ... -channels ... -min_freq ... -channel_bandwidth ... -samples ... -sampling_time ..." << std::endl;
#ifdef HAVE_PSRDADA
    std::cerr << "\tPSRDADA: -dada -dada_key ... -beams ... -synthesized_beams ... -batches ..." << std::endl;
#endif // HAVE_PSRDADA
    std::cerr << "\t [-random] -width ... -dm ... -beams ... -synthesized_beams ... -batches ... -channels ... -min_freq ... -channel_bandwidth ... -samples ... -sampling_time ..." << std::endl;
}
