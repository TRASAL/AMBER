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
#include <DataTypes.hpp>
#include <CommandLine.hpp>
#include <Kernels.hpp>
#include <Memory.hpp>
#include <Pipeline.hpp>

int main(int argc, char *argv[])
{
    // Command line options
    Options options{};
    DeviceOptions deviceOptions{};
    DataOptions dataOptions{};
    GeneratorOptions generatorOptions{};
    // Memory
    HostMemory hostMemory{};
    DeviceMemory deviceMemory{};
    HostMemoryDumpFiles hostMemoryDumpFiles{};
    // OpenCL kernels
    isa::OpenCL::OpenCLRunTime openclRunTime{};
    KernelConfigurations kernelConfigurations{};
    Kernels kernels{};
    KernelRunTimeConfigurations kernelRunTimeConfigurations{};
    // Timers
    Timers timers{};
    // Observation
    AstroData::Observation observation;

    // Process command line arguments
    isa::utils::ArgumentList args(argc, argv);
    try
    {
        processCommandLineOptions(args, options, deviceOptions, dataOptions, hostMemoryDumpFiles, kernelConfigurations, generatorOptions, observation);
    }
    catch (std::exception &err)
    {
        return 1;
    }

    // Load or generate input data
    try
    {
        hostMemory.input.resize(observation.getNrBeams());
        if (dataOptions.dataLOFAR || dataOptions.dataSIGPROC || dataOptions.dataPSRDADA)
        {
            loadInput(observation, deviceOptions, dataOptions, hostMemory, timers);
        }
        else
        {
            for (unsigned int beam = 0; beam < observation.getNrBeams(); beam++)
            {
                // TODO: if there are multiple synthesized beams, the generated data should take this into account
                hostMemory.input.at(beam) = new std::vector<std::vector<inputDataType> *>(observation.getNrBatches());
                AstroData::generateSinglePulse(generatorOptions.width, generatorOptions.DM, observation, deviceOptions.padding.at(deviceOptions.deviceName), *(hostMemory.input.at(beam)), inputBits, generatorOptions.random);
            }
        }
        try
        {
            hostMemory.zappedChannels.resize(observation.getNrChannels(deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(unsigned int)));
        }
        catch (std::out_of_range &err)
        {
            std::cerr << "No padding specified for " << deviceOptions.deviceName << "." << std::endl;
            return 1;
        }
        try
        {
            AstroData::readZappedChannels(observation, dataOptions.channelsFile, hostMemory.zappedChannels);
            AstroData::readIntegrationSteps(observation, dataOptions.integrationFile, hostMemory.integrationSteps);
        }
        catch (AstroData::FileError &err)
        {
            std::cerr << err.what() << std::endl;
            return 1;
        }
    }
    catch (std::exception &err)
    {
        std::cerr << err.what() << std::endl;
        return 1;
    }
    // Compute the amount of samples necessary for dedispersion
    if ( options.subbandDedispersion )
    {
        float maxShiftStepOne = 0;
        float maxShiftStepTwo = 0;
        float inverseHighFreq = 1.0f / std::pow(observation.getSubbandMaxFreq(), 2.0f);
        float inverseFreq = 1.0f / std::pow(observation.getSubbandMinFreq(), 2.0f);
        maxShiftStepTwo = 4148.808f * (inverseFreq - inverseHighFreq) * observation.getNrSamplesPerBatch();
        maxShiftStepTwo /= observation.getNrSamplesPerBatch() * observation.getSamplingTime();
        observation.setNrSamplesPerBatch(static_cast<unsigned int>(std::ceil(observation.getNrSamplesPerBatch() + (maxShiftStepTwo * (observation.getFirstDM() + ((observation.getNrDMs() - 1) * observation.getDMStep()))))), true);
        inverseHighFreq = 1.0f / std::pow(observation.getMaxFreq(), 2.0f);
        inverseFreq = 1.0f / std::pow(observation.getMinFreq(), 2.0f);
        maxShiftStepOne = 4148.808f * (inverseFreq - inverseHighFreq) * observation.getNrSamplesPerBatch();
        maxShiftStepOne /= observation.getNrSamplesPerBatch() * observation.getSamplingTime();
        observation.setNrSamplesPerDispersedBatch(static_cast<unsigned int>(std::ceil(observation.getNrSamplesPerBatch(true) + (maxShiftStepOne * (observation.getFirstDM(true) + ((observation.getNrDMs(true) - 1) * observation.getDMStep(true)))))), true);
        if ( options.downsampling )
        {
            observation.setNrSamplesPerBatch(observation.getNrSamplesPerBatch(true, observation.getDownsampling()), true);
            observation.setNrSamplesPerDispersedBatch(observation.getNrSamplesPerDispersedBatch(true, observation.getDownsampling()), true);
        }
        observation.setNrDelayBatches(static_cast<unsigned int>(std::ceil(static_cast<double>(observation.getNrSamplesPerDispersedBatch(true)) / observation.getNrSamplesPerBatch(true))), true);
    }
    else
    {
        float maxShiftSingleStep = 0;
        float inverseHighFreq = 1.0f / std::pow(observation.getMaxFreq(), 2.0f);
        float inverseFreq = 1.0f / std::pow(observation.getMinFreq(), 2.0f);
        maxShiftSingleStep = 4148.808f * (inverseFreq - inverseHighFreq) * observation.getNrSamplesPerBatch();
        maxShiftSingleStep /= observation.getNrSamplesPerBatch() * observation.getSamplingTime();
        observation.setNrSamplesPerDispersedBatch(static_cast<unsigned int>(std::ceil(observation.getNrSamplesPerBatch() + (maxShiftSingleStep * (observation.getFirstDM() + ((observation.getNrDMs() - 1) * observation.getDMStep()))))));
        if ( options.downsampling )
        {
            observation.setNrSamplesPerDispersedBatch(observation.getNrSamplesPerDispersedBatch(false, observation.getDownsampling()));
        }
        observation.setNrDelayBatches(static_cast<unsigned int>(std::ceil(static_cast<double>(observation.getNrSamplesPerDispersedBatch()) / observation.getNrSamplesPerBatch())));
    }
    // Computing dedispersion shifts and mapping of synthesized beams
    if ( options.subbandDedispersion )
    {
        hostMemory.shiftsStepOne = Dedispersion::getShifts(observation, deviceOptions.padding.at(deviceOptions.deviceName));
        hostMemory.shiftsStepTwo = Dedispersion::getShiftsStepTwo(observation, deviceOptions.padding.at(deviceOptions.deviceName));
        hostMemory.beamMapping.resize(observation.getNrSynthesizedBeams() * observation.getNrSubbands(deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(unsigned int)));
    }
    else
    {
        hostMemory.shiftsSingleStep = Dedispersion::getShifts(observation, deviceOptions.padding.at(deviceOptions.deviceName));
        hostMemory.beamMapping.resize(observation.getNrSynthesizedBeams() * observation.getNrChannels(deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(unsigned int)));
    }
    AstroData::generateBeamMapping(observation, hostMemory.beamMapping, deviceOptions.padding.at(deviceOptions.deviceName), options.subbandDedispersion);
    // Print message with observation and search information
    if (options.print)
    {
        std::cout << "Device: " << deviceOptions.deviceName << " (" + std::to_string(deviceOptions.platformID) + ", ";
        std::cout << std::to_string(deviceOptions.deviceID) + ")" << std::endl;
        std::cout << "Padding: " << deviceOptions.padding[deviceOptions.deviceName] << " bytes" << std::endl;
        std::cout << std::endl;
        std::cout << "Beams: " << observation.getNrBeams() << std::endl;
        std::cout << "Synthesized Beams: " << observation.getNrSynthesizedBeams() << std::endl;
        std::cout << "Batches: " << observation.getNrBatches() << std::endl;
        if ( options.subbandDedispersion )
        {
            std::cout << "Max Delay (in Batches): " << observation.getNrDelayBatches(true) << std::endl;
            std::cout << "Samples per Batch (input): " << observation.getNrSamplesPerBatch() << std::endl;
            std::cout << "Samples per Dispersed Batch: " << observation.getNrSamplesPerDispersedBatch(true) / observation.getDownsampling() << std::endl;
            std::cout << "Samples per Batch (subband): " << observation.getNrSamplesPerBatch(true) / observation.getDownsampling() << std::endl;
            std::cout << "Samples per Batch (dedispersed): " << observation.getNrSamplesPerBatch() / observation.getDownsampling() << std::endl;
        }
        else
        {
            std::cout << "Max Delay (in Batches): " << observation.getNrDelayBatches() << std::endl;
            std::cout << "Samples per Batch (input): " << observation.getNrSamplesPerBatch() << std::endl;
            std::cout << "Samples per Dispersed Batch: " << observation.getNrSamplesPerDispersedBatch() / observation.getDownsampling() << std::endl;
        }
        std::cout << "Sampling time: " << observation.getSamplingTime() * observation.getDownsampling() << std::endl;
        if ( options.downsampling )
        {
            std::cout << "Downsampling factor: " << observation.getDownsampling() << std::endl;
        }
        std::cout << "Frequency range: " << observation.getMinFreq() << " MHz, " << observation.getMaxFreq() << " MHz";
        std::cout << std::endl;
        std::cout << "Subbands: " << observation.getNrSubbands() << " (" << observation.getSubbandBandwidth() << " MHz)";
        std::cout << std::endl;
        std::cout << "Channels: " << observation.getNrChannels() << " (" << observation.getChannelBandwidth() << " MHz)";
        std::cout << std::endl;
        std::cout << "Zapped Channels: " << observation.getNrZappedChannels() << std::endl;
        std::cout << "Integration steps: " << hostMemory.integrationSteps.size() << std::endl;
        if (options.subbandDedispersion)
        {
            std::cout << "Subbanding DMs: " << observation.getNrDMs(true) << " (" << observation.getFirstDM(true) << ", ";
            std::cout << observation.getFirstDM(true) + ((observation.getNrDMs(true) - 1) * observation.getDMStep(true));
            std::cout << ")" << std::endl;
        }
        std::cout << "DMs: " << observation.getNrDMs() << " (" << observation.getFirstDM() << ", ";
        std::cout << observation.getFirstDM() + ((observation.getNrDMs() - 1) * observation.getDMStep()) << ")";
        std::cout << std::endl;
        std::cout << std::endl;
    }

    // Initialize OpenCL
    isa::OpenCL::OpenCLRunTime openCLRunTime;
    try
    {
        isa::OpenCL::initializeOpenCL(deviceOptions.platformID, 1, openclRunTime);
    }
    catch (isa::OpenCL::OpenCLError &err)
    {
        std::cerr << err.what() << std::endl;
        return 1;
    }

    // Memory allocation
    allocateHostMemory(observation, options, deviceOptions, dataOptions, kernelConfigurations, hostMemory);
    if (observation.getNrDelayBatches() > observation.getNrBatches())
    {
        std::cerr << "Not enough input batches for the specified search." << std::endl;
        return 1;
    }
    try
    {
        allocateDeviceMemory(observation, openclRunTime, options, deviceOptions, hostMemory, deviceMemory);
    }
    catch (cl::Error &err)
    {
        std::cerr << "Memory error: " << err.what() << " " << err.err() << std::endl;
        return 1;
    }
    if ( options.debug )
    {
        std::uint64_t hostMemorySize = 0;
        hostMemorySize += hostMemory.zappedChannels.size() * sizeof(unsigned int);
        hostMemorySize += hostMemory.beamMapping.size() * sizeof(unsigned int);
        hostMemorySize += hostMemory.dispersedData.size() * sizeof(inputDataType);
        hostMemorySize += hostMemory.subbandedData.size() * sizeof(outputDataType);
        hostMemorySize += hostMemory.dedispersedData.size() * sizeof(outputDataType);
        hostMemorySize += hostMemory.integratedData.size() * sizeof(outputDataType);
        if ( options.snrMode == SNRMode::Standard )
        {
            hostMemorySize += hostMemory.snrData.size() * sizeof(float);
            hostMemorySize += hostMemory.snrSamples.size() * sizeof(unsigned int);
        }
        else if ( options.snrMode == SNRMode::Momad )
        {
            hostMemorySize += hostMemory.maxValues.size() * sizeof(outputDataType);
            hostMemorySize += hostMemory.maxIndices.size() * sizeof(unsigned int);
            hostMemorySize += hostMemory.stdevs.size() * sizeof(outputDataType);
            hostMemorySize += hostMemory.medianOfMediansStepOne.size() * sizeof(outputDataType);
            hostMemorySize += hostMemory.medianOfMedians.size() * sizeof(outputDataType);
            hostMemorySize += hostMemory.medianOfMediansAbsoluteDeviation.size() * sizeof(outputDataType);
        }
        if ( options.subbandDedispersion )
        {
            hostMemorySize += hostMemory.shiftsStepOne->size() * sizeof(float);
            hostMemorySize += hostMemory.shiftsStepTwo->size() * sizeof(float);
        }
        else
        {
            hostMemorySize += hostMemory.shiftsSingleStep->size() * sizeof(float);
        }
        std::cout << "Allocated memory: " << isa::utils::giga(hostMemorySize) << " GB" << std::endl;
        std::cout << std::endl;
    }

    // Generate OpenCL kernels
    try
    {
        generateOpenCLKernels(openclRunTime, observation, options, deviceOptions, kernelConfigurations, hostMemory, deviceMemory, kernels);
    }
    catch (std::exception &err)
    {
        std::cerr << "OpenCL code generation error: " << err.what() << std::endl;
        return 1;
    }

    // Generate run time configurations for the OpenCL kernels
    generateOpenCLRunTimeConfigurations(observation, options, deviceOptions, kernelConfigurations, hostMemory, kernelRunTimeConfigurations);

    // Search loop
    pipeline(openclRunTime, observation, options, deviceOptions, dataOptions, timers, kernels, kernelRunTimeConfigurations, hostMemory, deviceMemory, hostMemoryDumpFiles);

    // Store performance statistics before shutting down
    std::ofstream outputStats;
    outputStats.open(dataOptions.outputFile + ".stats");
    outputStats << std::fixed << std::setprecision(6);
    outputStats << "# nrDMs" << std::endl;
    if (options.subbandDedispersion)
    {
        outputStats << observation.getNrDMs(true) * observation.getNrDMs() << std::endl;
    }
    else
    {
        outputStats << observation.getNrDMs() << std::endl;
    }
    outputStats << "# timers.inputLoad" << std::endl;
    outputStats << timers.inputLoad.getTotalTime() << std::endl;
    outputStats << "# timers.search" << std::endl;
    outputStats << timers.search.getTotalTime() << std::endl;
    outputStats << "# inputHandlingTotal inputHandlingAvg err" << std::endl;
    outputStats << timers.inputHandling.getTotalTime() << " " << timers.inputHandling.getAverageTime() << " ";
    outputStats << timers.inputHandling.getStandardDeviation() << std::endl;
    outputStats << "# inputCopyTotal inputCopyAvg err" << std::endl;
    outputStats << timers.inputCopy.getTotalTime() << " " << timers.inputCopy.getAverageTime() << " ";
    outputStats << timers.inputCopy.getStandardDeviation() << std::endl;
    if ( options.downsampling )
    {
        outputStats << "# downsamplingTotal downsamplingAvg err" << std::endl;
        outputStats << timers.downsampling.getTotalTime() << " " << timers.downsampling.getAverageTime() << " ";
        outputStats << timers.downsampling.getStandardDeviation() << std::endl;
    }
    if (!options.subbandDedispersion)
    {
        outputStats << "# dedispersionSingleStepTotal dedispersionSingleStepAvg err" << std::endl;
        outputStats << timers.dedispersionSingleStep.getTotalTime() << " ";
        outputStats << timers.dedispersionSingleStep.getAverageTime() << " ";
        outputStats << timers.dedispersionSingleStep.getStandardDeviation() << std::endl;
    }
    else
    {
        outputStats << "# dedispersionStepOneTotal dedispersionStepOneAvg err" << std::endl;
        outputStats << timers.dedispersionStepOne.getTotalTime() << " ";
        outputStats << timers.dedispersionStepOne.getAverageTime() << " ";
        outputStats << timers.dedispersionStepOne.getStandardDeviation() << std::endl;
        outputStats << "# dedispersionStepTwoTotal dedispersionStepTwoAvg err" << std::endl;
        outputStats << timers.dedispersionStepTwo.getTotalTime() << " ";
        outputStats << timers.dedispersionStepTwo.getAverageTime() << " ";
        outputStats << timers.dedispersionStepTwo.getStandardDeviation() << std::endl;
    }
    outputStats << "# integrationTotal integrationAvg err" << std::endl;
    outputStats << timers.integration.getTotalTime() << " " << timers.integration.getAverageTime() << " ";
    outputStats << timers.integration.getStandardDeviation() << std::endl;
    if (options.snrMode == SNRMode::Standard)
    {
        outputStats << "# snrTotal snrAvg err" << std::endl;
        outputStats << timers.snr.getTotalTime() << " " << timers.snr.getAverageTime() << " ";
        outputStats << timers.snr.getStandardDeviation() << std::endl;
    }
    else if (options.snrMode == SNRMode::Momad)
    {
        outputStats << "# maxTotal maxAvg err" << std::endl;
        outputStats << timers.max.getTotalTime() << " " << timers.max.getAverageTime() << " ";
        outputStats << timers.max.getStandardDeviation() << std::endl;
        outputStats << "# medianOfMediansStepOneTotal medianOfMediansStepOneAvg err" << std::endl;
        outputStats << timers.medianOfMediansStepOne.getTotalTime() << " " << timers.medianOfMediansStepOne.getAverageTime() << " ";
        outputStats << timers.medianOfMediansStepOne.getStandardDeviation() << std::endl;
        outputStats << "# medianOfMediansStepTwoTotal medianOfMediansStepTwoAvg err" << std::endl;
        outputStats << timers.medianOfMediansStepTwo.getTotalTime() << " " << timers.medianOfMediansStepTwo.getAverageTime() << " ";
        outputStats << timers.medianOfMediansStepTwo.getStandardDeviation() << std::endl;
        outputStats << "# medianOfMediansAbsoluteDeviationStepOneTotal medianOfMediansAbsoluteDeviationStepOneAvg err" << std::endl;
        outputStats << timers.medianOfMediansAbsoluteDeviationStepOne.getTotalTime() << " " << timers.medianOfMediansAbsoluteDeviationStepOne.getAverageTime() << " ";
        outputStats << timers.medianOfMediansAbsoluteDeviationStepOne.getStandardDeviation() << std::endl;
        outputStats << "# medianOfMediansAbsoluteDeviationStepTwoTotal medianOfMediansAbsoluteDeviationStepTwoAvg err" << std::endl;
        outputStats << timers.medianOfMediansAbsoluteDeviationStepTwo.getTotalTime() << " " << timers.medianOfMediansAbsoluteDeviationStepTwo.getAverageTime() << " ";
        outputStats << timers.medianOfMediansAbsoluteDeviationStepTwo.getStandardDeviation() << std::endl;
    }
    else if (options.snrMode == SNRMode::MomSigmaCut)
    {
        outputStats << "# maxStdSigmaCutTotal maxAvg err" << std::endl;
        outputStats << timers.max.getTotalTime() << " " << timers.max.getAverageTime() << " ";
        outputStats << timers.max.getStandardDeviation() << std::endl;
        outputStats << "# medianOfMediansStepOneTotal medianOfMediansStepOneAvg err" << std::endl;
        outputStats << timers.medianOfMediansStepOne.getTotalTime() << " " << timers.medianOfMediansStepOne.getAverageTime() << " ";
        outputStats << timers.medianOfMediansStepOne.getStandardDeviation() << std::endl;
        outputStats << "# medianOfMediansStepTwoTotal medianOfMediansStepTwoAvg err" << std::endl;
        outputStats << timers.medianOfMediansStepTwo.getTotalTime() << " " << timers.medianOfMediansStepTwo.getAverageTime() << " ";
        outputStats << timers.medianOfMediansStepTwo.getStandardDeviation() << std::endl;
    }
    outputStats << "# outputCopyTotal outputCopyAvg err" << std::endl;
    outputStats << timers.outputCopy.getTotalTime() << " " << timers.outputCopy.getAverageTime() << " ";
    outputStats << timers.outputCopy.getStandardDeviation() << std::endl;
    outputStats << "# triggerTimeTotal triggerTimeAvg err" << std::endl;
    outputStats << timers.trigger.getTotalTime() << " " << timers.trigger.getAverageTime() << " ";
    outputStats << timers.trigger.getStandardDeviation() << std::endl;
    outputStats.close();

    return 0;
}
