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

void processCommandLineOptions(isa::utils::ArgumentList &argumentList, Options &options, DeviceOptions &deviceOptions, DataOptions &dataOptions, HostMemoryDumpFiles & hostMemoryDumpFiles, KernelConfigurations &kernelConfigurations, GeneratorOptions &generatorOptions, AstroData::Observation &observation)
{
    try
    {
        options.debug = argumentList.getSwitch("-debug");
        options.print = argumentList.getSwitch("-print");
        options.dataDump = argumentList.getSwitch("-data_dump");
        if (options.dataDump)
        {
            hostMemoryDumpFiles.dumpFilesPrefix = argumentList.getSwitchArgument<std::string>("-dump_prefix");
        }
        options.rfimOptions.enable = argumentList.getSwitch("-rfim");
        options.splitBatchesDedispersion = argumentList.getSwitch("-splitbatches_dedispersion");
        options.subbandDedispersion = argumentList.getSwitch("-subband_dedispersion");
        options.compactResults = argumentList.getSwitch("-compact_results");
        options.threshold = argumentList.getSwitchArgument<float>("-threshold");
        deviceOptions.platformID = argumentList.getSwitchArgument<unsigned int>("-opencl_platform");
        deviceOptions.deviceID = argumentList.getSwitchArgument<unsigned int>("-opencl_device");
        deviceOptions.deviceName = argumentList.getSwitchArgument<std::string>("-device_name");
        deviceOptions.synchronized = argumentList.getSwitch("-sync");
        AstroData::readPaddingConf(deviceOptions.padding, argumentList.getSwitchArgument<std::string>("-padding_file"));
        if ( options.rfimOptions.enable )
        {
            options.rfimOptions.timeDomainSigmaCut = argumentList.getSwitch("-time_domain_sigma_cut");
            if ( options.rfimOptions.timeDomainSigmaCut )
            {
                options.rfimOptions.timeDomainSigmaCutStepsFile = argumentList.getSwitchArgument<std::string>("-time_domain_sigma_cut_steps");
                RFIm::readRFImConfig(kernelConfigurations.timeDomainSigmaCutParameters, argumentList.getSwitchArgument<std::string>("-time_domain_sigma_cut_configuration"));
            }
            options.rfimOptions.frequencyDomainSigmaCut = argumentList.getSwitch("-frequency_domain_sigma_cut");
            if ( options.rfimOptions.frequencyDomainSigmaCut )
            {
                options.rfimOptions.nrBins = argumentList.getSwitchArgument<unsigned int>("-nr_bins");
                options.rfimOptions.frequencyDomainSigmaCutStepsFile = argumentList.getSwitchArgument<std::string>("-frequency_domain_sigma_cut_steps");
                RFIm::readRFImConfig(kernelConfigurations.frequencyDomainSigmaCutParameters, argumentList.getSwitchArgument<std::string>("-frequency_domain_sigma_cut_configuration"));
            }
        }
        if (argumentList.getSwitch("-snr_standard"))
        {
            options.snrMode = SNRMode::Standard;
        }
        else if ( argumentList.getSwitch("-snr_sc") )
        {
            options.snrMode = SNRMode::SigmaCut;
        }
        else if (argumentList.getSwitch("-snr_momad"))
        {
            options.snrMode = SNRMode::Momad;
        }
        else if ( argumentList.getSwitch("-snr_mom_sigmacut") )
        {
            options.snrMode = SNRMode::MomSigmaCut;
        }
        else
        {
            // Default option right now is to use the standard mode.
            options.snrMode = SNRMode::Standard;
        }
        if ( inputBits >= 8 )
        {
            options.downsampling = argumentList.getSwitch("-downsampling");
            if ( options.downsampling )
            {
                observation.setDownsampling(argumentList.getSwitchArgument<unsigned int>("-downsampling_factor"));
                Integration::readTunedIntegrationConf(kernelConfigurations.downsamplingParameters, argumentList.getSwitchArgument<std::string>("-downsampling_configuration"));
            }
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
        try
        {
            dataOptions.synthesizedBeamsFile = argumentList.getSwitchArgument<std::string>("-synthesized_beams_file");
        }
        catch ( isa::utils::SwitchNotFound & err )
        {
            // If not specified, do nothing.
        }
        try
        {
            dataOptions.integrationFile = argumentList.getSwitchArgument<std::string>("-integration_steps");
        }
        catch ( isa::utils::SwitchNotFound & err )
        {
            // If not specified, do nothing.
        }
        if (!options.subbandDedispersion)
        {
            Dedispersion::readTunedDedispersionConf(kernelConfigurations.dedispersionSingleStepParameters, argumentList.getSwitchArgument<std::string>("-dedispersion_file"));
        }
        else
        {
            Dedispersion::readTunedDedispersionConf(kernelConfigurations.dedispersionStepOneParameters, argumentList.getSwitchArgument<std::string>("-dedispersion_stepone_file"));
            Dedispersion::readTunedDedispersionConf(kernelConfigurations.dedispersionStepTwoParameters, argumentList.getSwitchArgument<std::string>("-dedispersion_steptwo_file"));
        }
        if ( dataOptions.integrationFile.size() > 1 )
        {
            Integration::readTunedIntegrationConf(kernelConfigurations.integrationParameters, argumentList.getSwitchArgument<std::string>("-integration_file"));
        }
        if (options.snrMode == SNRMode::Standard)
        {
            SNR::readTunedSNRConf(kernelConfigurations.snrParameters, argumentList.getSwitchArgument<std::string>("-snr_file"));
        }
        else if ( options.snrMode == SNRMode::SigmaCut )
        {
            SNR::readTunedSNRConf(kernelConfigurations.snrParameters, argumentList.getSwitchArgument<std::string>("-snr_file"));
            options.nSigma = argumentList.getSwitchArgument<float>("-nsigma");
            try
            {
                options.sigmaCorrectionFactor = argumentList.getSwitchArgument<float>("-correction_factor");
            }
            catch ( isa::utils::SwitchNotFound &err )
            {
                options.sigmaCorrectionFactor = 1.0f;
            }
        }
        else if (options.snrMode == SNRMode::Momad)
        {
            SNR::readTunedSNRConf(kernelConfigurations.maxParameters, argumentList.getSwitchArgument<std::string>("-max_file"));
            SNR::readTunedSNRConf(kernelConfigurations.medianOfMediansStepOneParameters, argumentList.getSwitchArgument<std::string>("-mom_stepone_file"));
            SNR::readTunedSNRConf(kernelConfigurations.medianOfMediansStepTwoParameters, argumentList.getSwitchArgument<std::string>("-mom_steptwo_file"));
            SNR::readTunedSNRConf(kernelConfigurations.medianOfMediansAbsoluteDeviationParameters, argumentList.getSwitchArgument<std::string>("-momad_file"));
        }
        else if (options.snrMode == SNRMode::MomSigmaCut)
        {
            SNR::readTunedSNRConf(kernelConfigurations.maxStdSigmaCutParameters, argumentList.getSwitchArgument<std::string>("-max_std_file"));
            SNR::readTunedSNRConf(kernelConfigurations.medianOfMediansStepOneParameters, argumentList.getSwitchArgument<std::string>("-mom_stepone_file"));
            SNR::readTunedSNRConf(kernelConfigurations.medianOfMediansStepTwoParameters, argumentList.getSwitchArgument<std::string>("-mom_steptwo_file"));
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
            dataOptions.dataFile = argumentList.getSwitchArgument<std::string>("-data");
            try
            {
                dataOptions.headerSizeSIGPROC = argumentList.getSwitchArgument<unsigned int>("-header");
            }
            catch( const isa::utils::SwitchNotFound &err )
            {
                dataOptions.headerSizeSIGPROC = AstroData::getSIGPROCHeaderSize(dataOptions.dataFile);
            }
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
            try
            {
                options.nrSynthesizedBeamsPerChunk = argumentList.getSwitchArgument<unsigned int>("-synthesized_beams_chunk");
            }
            catch ( isa::utils::SwitchNotFound & err )
            {
                options.nrSynthesizedBeamsPerChunk = observation.getNrSynthesizedBeams();
            }
            observation.setNrBatches(argumentList.getSwitchArgument<unsigned int>("-batches"));
#endif // HAVE_PSRDADA
        }
        else
        {
            observation.setNrBeams(argumentList.getSwitchArgument<unsigned int>("-beams"));
            observation.setNrSynthesizedBeams(argumentList.getSwitchArgument<unsigned int>("-synthesized_beams"));
            try
            {
                options.nrSynthesizedBeamsPerChunk = argumentList.getSwitchArgument<unsigned int>("-synthesized_beams_chunk");
            }
            catch ( isa::utils::SwitchNotFound & err )
            {
                options.nrSynthesizedBeamsPerChunk = observation.getNrSynthesizedBeams();
            }
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
    std::cerr << program << " [-debug] [-print] [-data_dump] -opencl_platform <int> -opencl_device <int> -device_name <string> [-sync]";
    std::cerr << " [-rfim]";
    std::cerr << " [-splitbatches_dedispersion] [-subband_dedispersion] [-snr_standard | -snr_sc | -snr_momad | -snr_mom_sigmacut] [-downsampling]";
    std::cerr << " -padding_file <string> -zapped_channels <string> [-synthesized_beams_file <string>] [-integration_steps <string>]";
    std::cerr << " [-compact_results] -output <string> -dms <int> -dm_first <float> -dm_step <float> -threshold <float>";
    std::cerr << " [-sigproc]";
#ifdef HAVE_HDF5
    std::cerr << " [-lofar]";
#endif // HAVE_HDF5
#ifdef HAVE_PSRDADA
    std::cerr << " [-dada]";
#endif // HAVE_PSRDADA
    std::cerr << std::endl;
    std::cerr << "\tData dump: -dump_prefix <string>" << std::endl;
    std::cerr << "\tRFIm: [-time_domain_sigma_cut] [-frequency_domain_sigma_cut]" << std::endl;
    std::cerr << "\t\tTime domain sigma cut: -time_domain_sigma_cut_steps <string> -time_domain_sigma_cut_configuration <string>" << std::endl;
    std::cerr << "\t\tFrequency domain sigma cut: -nr_bins <int> -frequency_domain_sigma_cut_steps <string> -frequency_domain_sigma_cut_configuration <string>" << std::endl;
    std::cerr << "\tDownsampling: -downsampling_factor <int> -downsampling_configuration <string>" << std::endl;
    std::cerr << "\tDedispersion: -dedispersion_file <string>" << std::endl;
    std::cerr << "\tSubband Dedispersion: -subband_dedispersion -dedispersion_stepone_file <string>" << std::endl;
    std::cerr << "\t\t-dedispersion_steptwo_file <string> -subbands <int> -subbanding_dms <int> -subbanding_dm_first <float>" << std::endl;
    std::cerr << "\t\t-subbanding_dm_step <float>" << std::endl;
    std::cerr << "\tIntegration Steps: -integration_file <string>" << std::endl;
    std::cerr << "\tStandard SNR: -snr_file <string>" << std::endl;
    std::cerr << "\tSNR with Sigma Cut: -snr_file <string> -nsigma <float> -correction_factor <float>" << std::endl;
    std::cerr << "\tMOMAD SNR: -max_file <string> -mom_stepone_file <string> -mom_steptwo_file <string> -momad_file <string>" << std::endl;
    std::cerr << "\tMOM Sigma Cut SNR: -max_std_file <string> -mom_stepone_file <string> -mom_steptwo_file <string>" << std::endl;
    std::cerr << std::endl;
#ifdef HAVE_HDF5
    std::cerr << "\tLOFAR: -lofar -header <string> -data <string> [-limit]" << std::endl;
    std::cerr << "\t\t-limit -batches <int>" << std::endl;
#endif // HAVE_HDF5
    std::cerr << "\tSIGPROC: -sigproc [-stream] [-header <int>] -data <string> -batches <int> -channels <int> -min_freq <float>" << std::endl;
    std::cerr << "\t\t-channel_bandwidth <float> -samples <int> -sampling_time <float>" << std::endl;
#ifdef HAVE_PSRDADA
    std::cerr << "\tPSRDADA: -dada -dada_key <string> -beams <int> -synthesized_beams <int> [-synthesized_beams_chunk <int>] -batches <int>" << std::endl;
#endif // HAVE_PSRDADA
    std::cerr << "\tTest data: [-random] -width <int> -dm <float> -beams <int> -synthesized_beams <int> [-synthesized_beams_chunk <int>] -batches <int> -channels <int>" << std::endl;
    std::cerr << "\t\t-min_freq <float> -channel_bandwidth <float> -samples <int> -sampling_time <float>" << std::endl;
}
