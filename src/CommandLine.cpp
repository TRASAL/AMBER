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

void processCommandLineOptions(isa::utils::ArgumentList &argumentList, Options &options, DeviceOptions &deviceOptions, DataOptions &dataOptions, KernelConfigurations &kernelConfigurations, GeneratorOptions &generatorOptions, AstroData::Observation &observation)
{
    try
    {
        options.debug = argumentList.getSwitch("-debug");
        options.print = argumentList.getSwitch("-print");
        options.dataDump = argumentList.getSwitch("-data_dump");
        options.splitBatchesDedispersion = argumentList.getSwitch("-splitbatches_dedispersion");
        options.subbandDedispersion = argumentList.getSwitch("-subband_dedispersion");
        options.compactResults = argumentList.getSwitch("-compact_results");
        options.threshold = argumentList.getSwitchArgument<float>("-threshold");
        deviceOptions.platformID = argumentList.getSwitchArgument<unsigned int>("-opencl_platform");
        deviceOptions.deviceID = argumentList.getSwitchArgument<unsigned int>("-opencl_device");
        deviceOptions.deviceName = argumentList.getSwitchArgument<std::string>("-device_name");
        deviceOptions.synchronized = argumentList.getSwitch("-sync");
        AstroData::readPaddingConf(deviceOptions.padding, argumentList.getSwitchArgument<std::string>("-padding_file"));
        if (argumentList.getSwitch("-snr_standard"))
        {
            options.snrMode = SNRMode::Standard;
        }
        else if (argumentList.getSwitch("-snr_momad"))
        {
            options.snrMode = SNRMode::Momad;
        }
        else
        {
            // Default option right now is to use the standard mode.
            options.snrMode = SNRMode::Standard;
        }
        dataOptions.dataLOFAR = argumentList.getSwitch("-lofar");
#ifndef HAVE_HDF5
        if (dataOptions.dataLOFAR)
        {
            std::cerr << "Not compiled with HDF5 support." << std::endl;
            throw std::exception();
        };
#endif // HAVE_HDF5
        dataOptions.dataPSRDADA = argumentList.getSwitch("-dada");
#ifndef HAVE_PSRDADA
        if (dataOptions.dataPSRDADA)
        {
            std::cerr << "Not compiled with PSRDADA support." << std::endl;
            throw std::exception();
        };
#endif // HAVE_PSRDADA
        dataOptions.dataSIGPROC = argumentList.getSwitch("-sigproc");
        if (!((((!(dataOptions.dataLOFAR && dataOptions.dataSIGPROC) && dataOptions.dataPSRDADA) || (!(dataOptions.dataLOFAR && dataOptions.dataPSRDADA) && dataOptions.dataSIGPROC)) || (!(dataOptions.dataSIGPROC && dataOptions.dataPSRDADA) && dataOptions.dataLOFAR)) || ((!dataOptions.dataLOFAR && !dataOptions.dataSIGPROC) && !dataOptions.dataPSRDADA)))
        {
            std::cerr << "-lofar -sigproc and -dada are mutually exclusive." << std::endl;
            throw std::exception();
        }
        dataOptions.channelsFile = argumentList.getSwitchArgument<std::string>("-zapped_channels");
        dataOptions.integrationFile = argumentList.getSwitchArgument<std::string>("-integration_steps");
        if (!options.subbandDedispersion)
        {
            Dedispersion::readTunedDedispersionConf(kernelConfigurations.dedispersionSingleStepParameters, argumentList.getSwitchArgument<std::string>("-dedispersion_file"));
        }
        else
        {
            Dedispersion::readTunedDedispersionConf(kernelConfigurations.dedispersionStepOneParameters, argumentList.getSwitchArgument<std::string>("-dedispersion_stepone_file"));
            Dedispersion::readTunedDedispersionConf(kernelConfigurations.dedispersionStepTwoParameters, argumentList.getSwitchArgument<std::string>("-dedispersion_steptwo_file"));
        }
        Integration::readTunedIntegrationConf(kernelConfigurations.integrationParameters, argumentList.getSwitchArgument<std::string>("-integration_file"));
        if (options.snrMode == SNRMode::Standard)
        {
            SNR::readTunedSNRConf(kernelConfigurations.snrParameters, argumentList.getSwitchArgument<std::string>("-snr_file"));
        }
        else if (options.snrMode == SNRMode::Momad)
        {
            SNR::readTunedSNRConf(kernelConfigurations.maxParameters, argumentList.getSwitchArgument<std::string>("-max_file"));
            SNR::readTunedSNRConf(kernelConfigurations.medianOfMediansStepOneParameters, argumentList.getSwitchArgument<std::string>("-mom_stepone_file"));
            SNR::readTunedSNRConf(kernelConfigurations.medianOfMediansStepTwoParameters, argumentList.getSwitchArgument<std::string>("-mom_steptwo_file"));
            SNR::readTunedSNRConf(kernelConfigurations.medianOfMediansAbsoluteDeviationParameters, argumentList.getSwitchArgument<std::string>("-momad_file"));
        }
        if (dataOptions.dataLOFAR)
        {
#ifdef HAVE_HDF5
            observation.setNrBeams(1);
            observation.setNrSynthesizedBeams(1);
            dataOptions.headerFile = argumentList.getSwitchArgument<std::string>("-header");
            dataOptions.dataFile = argumentList.getSwitchArgument<std::string>("-data");
            dataOptions.limit = argumentList.getSwitch("-limit");
            if (dataOptions.limit)
            {
                observation.setNrBatches(argumentList.getSwitchArgument<unsigned int>("-batches"));
            }
#endif // HAVE_HDF5
        }
        else if (dataOptions.dataSIGPROC)
        {
            observation.setNrBeams(1);
            observation.setNrSynthesizedBeams(1);
            dataOptions.streamingMode = argumentList.getSwitch("-stream");
            dataOptions.headerSizeSIGPROC = argumentList.getSwitchArgument<unsigned int>("-header");
            dataOptions.dataFile = argumentList.getSwitchArgument<std::string>("-data");
            observation.setNrBatches(argumentList.getSwitchArgument<unsigned int>("-batches"));
            if (options.subbandDedispersion)
            {
                observation.setFrequencyRange(argumentList.getSwitchArgument<unsigned int>("-subbands"),
                                              argumentList.getSwitchArgument<unsigned int>("-channels"),
                                              argumentList.getSwitchArgument<float>("-min_freq"),
                                              argumentList.getSwitchArgument<float>("-channel_bandwidth"));
            }
            else
            {
                observation.setFrequencyRange(1, argumentList.getSwitchArgument<unsigned int>("-channels"),
                                              argumentList.getSwitchArgument<float>("-min_freq"),
                                              argumentList.getSwitchArgument<float>("-channel_bandwidth"));
            }
            observation.setNrSamplesPerBatch(argumentList.getSwitchArgument<unsigned int>("-samples"));
            observation.setSamplingTime(argumentList.getSwitchArgument<float>("-sampling_time"));
        }
        else if (dataOptions.dataPSRDADA)
        {
#ifdef HAVE_PSRDADA
            dataOptions.dadaKey = std::stoi("0x" + argumentList.getSwitchArgument<std::string>("-dada_key"), 0, 16);
            if (options.subbandDedispersion)
            {
                observation.setFrequencyRange(argumentList.getSwitchArgument<unsigned int>("-subbands"), 1, 1.0f, 1.0f);
            }
            else
            {
                observation.setFrequencyRange(1, 1, 1.0f, 1.0f);
            }
            observation.setNrBeams(argumentList.getSwitchArgument<unsigned int>("-beams"));
            observation.setNrSynthesizedBeams(argumentList.getSwitchArgument<unsigned int>("-synthesized_beams"));
            observation.setNrBatches(argumentList.getSwitchArgument<unsigned int>("-batches"));
#endif // HAVE_PSRDADA
        }
        else
        {
            observation.setNrBeams(argumentList.getSwitchArgument<unsigned int>("-beams"));
            observation.setNrSynthesizedBeams(argumentList.getSwitchArgument<unsigned int>("-synthesized_beams"));
            observation.setNrBatches(argumentList.getSwitchArgument<unsigned int>("-batches"));
            if (options.subbandDedispersion)
            {
                observation.setFrequencyRange(argumentList.getSwitchArgument<unsigned int>("-subbands"),
                                              argumentList.getSwitchArgument<unsigned int>("-channels"),
                                              argumentList.getSwitchArgument<float>("-min_freq"),
                                              argumentList.getSwitchArgument<float>("-channel_bandwidth"));
            }
            else
            {
                observation.setFrequencyRange(1, argumentList.getSwitchArgument<unsigned int>("-channels"),
                                              argumentList.getSwitchArgument<float>("-min_freq"),
                                              argumentList.getSwitchArgument<float>("-channel_bandwidth"));
            }
            observation.setNrSamplesPerBatch(argumentList.getSwitchArgument<unsigned int>("-samples"));
            observation.setSamplingTime(argumentList.getSwitchArgument<float>("-sampling_time"));
            generatorOptions.random = argumentList.getSwitch("-random");
            generatorOptions.width = argumentList.getSwitchArgument<unsigned int>("-width");
            generatorOptions.DM = argumentList.getSwitchArgument<float>("-dm");
        }
        dataOptions.outputFile = argumentList.getSwitchArgument<std::string>("-output");
        if (options.subbandDedispersion)
        {
            observation.setDMRange(argumentList.getSwitchArgument<unsigned int>("-subbanding_dms"),
                                   argumentList.getSwitchArgument<float>("-subbanding_dm_first"),
                                   argumentList.getSwitchArgument<float>("-subbanding_dm_step"), true);
        }
        observation.setDMRange(argumentList.getSwitchArgument<unsigned int>("-dms"),
                               argumentList.getSwitchArgument<float>("-dm_first"),
                               argumentList.getSwitchArgument<float>("-dm_step"));
    }
    catch (isa::utils::EmptyCommandLine &err)
    {
        usage(argumentList.getName());
        throw;
    }
    catch (isa::utils::SwitchNotFound &err)
    {
        std::cerr << err.what() << std::endl;
        throw;
    }
    catch (std::exception &err)
    {
        std::cerr << err.what() << std::endl;
        throw;
    }
}

void usage(const std::string &program)
{
    std::cerr << program << " [-debug] [-print] [-data_dump] -opencl_platform ... -opencl_device ... -device_name ... [-sync]";
    std::cerr << "  -padding_file ... -zapped_channels ... -integration_steps ... -integration_file ...";
    std::cerr << " [-splitbatches_dedispersion] [-subband_dedispersion] [-snr_standard | -snr_momad]";
    std::cerr << " [-compact_results] -output ... -dms ... -dm_first ... -dm_step ... -threshold ... [-sigproc]";
#ifdef HAVE_HDF5
    std::cerr << " [-lofar]";
#endif // HAVE_HDF5
#ifdef HAVE_PSRDADA
    std::cerr << " [-dada]";
#endif // HAVE_PSRDADA
    std::cerr << std::endl;
    std::cerr << "\tDedispersion: -dedispersion_file ..." << std::endl;
    std::cerr << "\tSubband Dedispersion: -subband_dedispersion -dedispersion_stepone_file ...";
    std::cerr << "-dedispersion_steptwo_file ... -subbands ... -subbanding_dms ... -subbanding_dm_first ...";
    std::cerr << "-subbanding_dm_step ..." << std::endl;
    std::cerr << "\tStandard SNR: -snr_file" << std::endl;
    std::cerr << "\tMOMAD SNR: -max_file ... -mom_stepone_file ... -mom_steptwo_file ... -momad_file ..." << std::endl;
#ifdef HAVE_HDF5
    std::cerr << "\tLOFAR: -lofar -header ... -data ... [-limit]" << std::endl;
    std::cerr << "\t\t -limit -batches ..." << std::endl;
#endif // HAVE_HDF5
    std::cerr << "\tSIGPROC: -sigproc [-stream] -header ... -data ... -batches ... -channels ... -min_freq ...";
    std::cerr << "-channel_bandwidth ... -samples ... -sampling_time ..." << std::endl;
#ifdef HAVE_PSRDADA
    std::cerr << "\tPSRDADA: -dada -dada_key ... -beams ... -synthesized_beams ... -batches ..." << std::endl;
#endif // HAVE_PSRDADA
    std::cerr << "\t [-random] -width ... -dm ... -beams ... -synthesized_beams ... -batches ... -channels ...";
    std::cerr << "-min_freq ... -channel_bandwidth ... -samples ... -sampling_time ..." << std::endl;
}
