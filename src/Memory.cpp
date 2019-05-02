// Copyright 2018 Netherlands Institute for Radio Astronomy (ASTRON)
// Copyright 2018 Netherlands eScience Center
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

#include <Memory.hpp>

void loadInput(AstroData::Observation &observation, const DeviceOptions &deviceOptions, const DataOptions &dataOptions, HostMemory &hostMemory, Timers &timers)
{
    if (dataOptions.dataLOFAR)
    {
#ifdef HAVE_HDF5
        hostMemory.input.at(0) = new std::vector<std::vector<inputDataType> *>(observation.getNrBatches());
        timers.inputLoad.start();
        if (dataOptions.limit)
        {
            AstroData::readLOFAR(dataOptions.headerFile, dataOptions.dataFile, observation, deviceOptions.padding.at(deviceOptions.deviceName), *(hostMemory.input.at(0)), observation.getNrBatches());
        }
        else
        {
            AstroData::readLOFAR(dataOptions.headerFile, dataOptions.dataFile, observation, deviceOptions.padding.at(deviceOptions.deviceName), *(hostMemory.input.at(0)));
        }
        timers.inputLoad.stop();
#endif // HAVE_HDF5
    }
    else if (dataOptions.dataSIGPROC && !dataOptions.streamingMode)
    {
        hostMemory.input.at(0) = new std::vector<std::vector<inputDataType> *>(observation.getNrBatches());
        timers.inputLoad.start();
        AstroData::readSIGPROC(observation, deviceOptions.padding.at(deviceOptions.deviceName), inputBits, dataOptions.headerSizeSIGPROC, dataOptions.dataFile, *(hostMemory.input.at(0)));
        timers.inputLoad.stop();
    }
    else if (dataOptions.dataPSRDADA)
    {
#ifdef HAVE_PSRDADA
        hostMemory.ringBuffer = dada_hdu_create(0);
        dada_hdu_set_key(hostMemory.ringBuffer, dataOptions.dadaKey);
        if (dada_hdu_connect(hostMemory.ringBuffer) != 0)
        {
            throw AstroData::RingBufferError("ERROR: impossible to connect to PSRDADA ringbuffer \"" + std::to_string(dataOptions.dadaKey) + "\".");
        }
        if (dada_hdu_lock_read(hostMemory.ringBuffer) != 0)
        {
            throw AstroData::RingBufferError("ERROR: impossible to lock the PSRDADA ringbuffer for reading the header.");
        }
        timers.inputLoad.start();
        AstroData::readPSRDADAHeader(observation, *hostMemory.ringBuffer);
        timers.inputLoad.stop();
#endif // HAVE_PSRDADA
    }
}

void allocateHostMemory(AstroData::Observation &observation, const Options &options, const DeviceOptions &deviceOptions, const DataOptions &dataOptions, const KernelConfigurations &kernelConfigurations, HostMemory &hostMemory)
{
    if (!options.subbandDedispersion)
    {
        if (dataOptions.dataPSRDADA || dataOptions.streamingMode)
        {
            hostMemory.inputStream.resize(observation.getNrDelayBatches());
            for (unsigned int batch = 0; batch < observation.getNrDelayBatches(); batch++)
            {
                if (inputBits >= 8)
                {
                    hostMemory.inputStream.at(batch) = new std::vector<inputDataType>(observation.getNrBeams() * observation.getNrChannels() * observation.getNrSamplesPerBatch());
                }
                else
                {
                    hostMemory.inputStream.at(batch) = new std::vector<inputDataType>(observation.getNrBeams() * observation.getNrChannels() * (observation.getNrSamplesPerBatch() / (8 / inputBits)));
                }
            }
        }
        if (kernelConfigurations.dedispersionSingleStepParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getSplitBatches())
        {
            // TODO: add support for splitBatches
        }
        else
        {
            if (inputBits >= 8)
            {
                hostMemory.dispersedData.resize(observation.getNrBeams() * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(), deviceOptions.padding.at(deviceOptions.deviceName)));
            }
            else
            {
                hostMemory.dispersedData.resize(observation.getNrBeams() * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType)));
            }
        }
        hostMemory.dedispersedData.resize(options.nrSynthesizedBeamsPerChunk * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / observation.getDownsampling(), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType)));
        if (hostMemory.integrationSteps.size() > 0)
        {
            hostMemory.integratedData.resize(options.nrSynthesizedBeamsPerChunk * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / observation.getDownsampling() / *(hostMemory.integrationSteps.begin()), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType)));
        }
        if ( options.snrMode == SNRMode::Standard || options.snrMode == SNRMode::SigmaCut )
        {
            hostMemory.snrData.resize(options.nrSynthesizedBeamsPerChunk * observation.getNrDMs(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(float)));
            hostMemory.snrSamples.resize(options.nrSynthesizedBeamsPerChunk * observation.getNrDMs(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(unsigned int)));
        }
        else if (options.snrMode == SNRMode::Momad || options.snrMode == SNRMode::MomSigmaCut)
        {
            hostMemory.maxValues.resize(options.nrSynthesizedBeamsPerChunk * observation.getNrDMs(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType)));
            hostMemory.maxIndices.resize(options.nrSynthesizedBeamsPerChunk * observation.getNrDMs(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(unsigned int)));
            if (options.snrMode == SNRMode::MomSigmaCut)
            {
              hostMemory.stdevs.resize(options.nrSynthesizedBeamsPerChunk * observation.getNrDMs(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType)));
            }
            hostMemory.medianOfMedians.resize(options.nrSynthesizedBeamsPerChunk * observation.getNrDMs(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType)));
            if (options.snrMode == SNRMode::Momad)
            {
              hostMemory.medianOfMediansAbsoluteDeviation.resize(options.nrSynthesizedBeamsPerChunk * observation.getNrDMs(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType)));
            }
        }
    }
    else
    {
        if (dataOptions.dataPSRDADA || dataOptions.streamingMode)
        {
            hostMemory.inputStream.resize(observation.getNrDelayBatches(true));
            for (unsigned int batch = 0; batch < observation.getNrDelayBatches(true); batch++)
            {
                if (inputBits >= 8)
                {
                    hostMemory.inputStream.at(batch) = new std::vector<inputDataType>(observation.getNrBeams() * observation.getNrChannels() * observation.getNrSamplesPerBatch());
                }
                else
                {
                    hostMemory.inputStream.at(batch) = new std::vector<inputDataType>(observation.getNrBeams() * observation.getNrChannels() * (observation.getNrSamplesPerBatch() / (8 / inputBits)));
                }
            }
        }
        if (kernelConfigurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getSplitBatches())
        {
            // TODO: add support for splitBatches
        }
        else
        {
            if (inputBits >= 8)
            {
                hostMemory.dispersedData.resize(observation.getNrBeams() * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType)));
            }
            else
            {
                hostMemory.dispersedData.resize(observation.getNrBeams() * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType)));
            }
        }

        hostMemory.subbandedData.resize(observation.getNrBeams() * observation.getNrDMs(true) * observation.getNrSubbands() * isa::utils::pad(observation.getNrSamplesPerBatch(true) / observation.getDownsampling(), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType)));
        hostMemory.dedispersedData.resize(options.nrSynthesizedBeamsPerChunk * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / observation.getDownsampling(), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType)));
        if (hostMemory.integrationSteps.size() > 0)
        {
            hostMemory.integratedData.resize(options.nrSynthesizedBeamsPerChunk * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / observation.getDownsampling() / *(hostMemory.integrationSteps.begin()), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType)));
        }
        if ( options.snrMode == SNRMode::Standard || options.snrMode == SNRMode::SigmaCut )
        {
            hostMemory.snrData.resize(options.nrSynthesizedBeamsPerChunk * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(float)));
            hostMemory.snrSamples.resize(options.nrSynthesizedBeamsPerChunk * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(unsigned int)));
        }
        else if (options.snrMode == SNRMode::Momad || options.snrMode == SNRMode::MomSigmaCut)
        {
            hostMemory.maxValues.resize(options.nrSynthesizedBeamsPerChunk * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType)));
            hostMemory.maxIndices.resize(options.nrSynthesizedBeamsPerChunk * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(unsigned int)));
            if (options.snrMode == SNRMode::MomSigmaCut)
            {
              hostMemory.stdevs.resize(options.nrSynthesizedBeamsPerChunk * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType)));
            }
            hostMemory.medianOfMedians.resize(options.nrSynthesizedBeamsPerChunk * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType)));
            if (options.snrMode == SNRMode::Momad)
            {
              hostMemory.medianOfMediansAbsoluteDeviation.resize(options.nrSynthesizedBeamsPerChunk * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType)));
            }
        }
    }
}

void allocateDeviceMemory(const AstroData::Observation &observation, const isa::OpenCL::OpenCLRunTime &openclRunTime, const Options &options, const DeviceOptions &deviceOptions, const HostMemory &hostMemory, DeviceMemory &deviceMemory)
{
    if (!options.subbandDedispersion)
    {
        deviceMemory.shiftsSingleStep = cl::Buffer(*openclRunTime.context, CL_MEM_READ_ONLY, hostMemory.shiftsSingleStep->size() * sizeof(float));
        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueWriteBuffer(deviceMemory.shiftsSingleStep, CL_FALSE, 0, hostMemory.shiftsSingleStep->size() * sizeof(float), reinterpret_cast<void *>(hostMemory.shiftsSingleStep->data()));
    }
    else
    {
        deviceMemory.shiftsStepOne = cl::Buffer(*openclRunTime.context, CL_MEM_READ_ONLY, hostMemory.shiftsStepOne->size() * sizeof(float));
        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueWriteBuffer(deviceMemory.shiftsStepOne, CL_FALSE, 0, hostMemory.shiftsStepOne->size() * sizeof(float), reinterpret_cast<void *>(hostMemory.shiftsStepOne->data()));
        deviceMemory.shiftsStepTwo = cl::Buffer(*openclRunTime.context, CL_MEM_READ_ONLY, hostMemory.shiftsStepTwo->size() * sizeof(float));
        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueWriteBuffer(deviceMemory.shiftsStepTwo, CL_FALSE, 0, hostMemory.shiftsStepTwo->size() * sizeof(float), reinterpret_cast<void *>(hostMemory.shiftsStepTwo->data()));
        deviceMemory.subbandedData = cl::Buffer(*openclRunTime.context, CL_MEM_READ_WRITE, hostMemory.subbandedData.size() * sizeof(outputDataType));
    }
    deviceMemory.zappedChannels = cl::Buffer(*openclRunTime.context, CL_MEM_READ_ONLY, hostMemory.zappedChannels.size() * sizeof(unsigned int));
    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueWriteBuffer(deviceMemory.zappedChannels, CL_FALSE, 0, hostMemory.zappedChannels.size() * sizeof(unsigned int), reinterpret_cast<const void *>(hostMemory.zappedChannels.data()));
    deviceMemory.beamMapping = cl::Buffer(*openclRunTime.context, CL_MEM_READ_ONLY, hostMemory.beamMapping.size() * sizeof(unsigned int));
    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueWriteBuffer(deviceMemory.beamMapping, CL_FALSE, 0, hostMemory.beamMapping.size() * sizeof(unsigned int), reinterpret_cast<const void *>(hostMemory.beamMapping.data()));
    if ( options.downsampling )
    {
        deviceMemory.dispersedData = cl::Buffer(*openclRunTime.context, CL_MEM_READ_WRITE, hostMemory.dispersedData.size() * sizeof(inputDataType));
    }
    else
    {
        deviceMemory.dispersedData = cl::Buffer(*openclRunTime.context, CL_MEM_READ_ONLY, hostMemory.dispersedData.size() * sizeof(inputDataType));
    }
    deviceMemory.dedispersedData = cl::Buffer(*openclRunTime.context, CL_MEM_READ_WRITE, hostMemory.dedispersedData.size() * sizeof(outputDataType));
    if (hostMemory.integrationSteps.size() > 0)
    {
        deviceMemory.integratedData = cl::Buffer(*openclRunTime.context, CL_MEM_READ_WRITE, hostMemory.integratedData.size() * sizeof(outputDataType));
    }
    if ( options.snrMode == SNRMode::Standard || options.snrMode == SNRMode::SigmaCut )
    {
        deviceMemory.snrData = cl::Buffer(*openclRunTime.context, CL_MEM_WRITE_ONLY, hostMemory.snrData.size() * sizeof(float));
        deviceMemory.snrSamples = cl::Buffer(*openclRunTime.context, CL_MEM_WRITE_ONLY, hostMemory.snrSamples.size() * sizeof(unsigned int));
    }
    else if (options.snrMode == SNRMode::Momad || options.snrMode == SNRMode::MomSigmaCut)
    {
        deviceMemory.maxValues = cl::Buffer(*openclRunTime.context, CL_MEM_WRITE_ONLY, hostMemory.maxValues.size() * sizeof(outputDataType));
        deviceMemory.maxIndices = cl::Buffer(*openclRunTime.context, CL_MEM_WRITE_ONLY, hostMemory.maxIndices.size() * sizeof(unsigned int));
        if (options.snrMode == SNRMode::MomSigmaCut)
        {
          deviceMemory.stdevs = cl::Buffer(*openclRunTime.context, CL_MEM_WRITE_ONLY, hostMemory.stdevs.size() * sizeof(outputDataType));
        }
        if (!options.subbandDedispersion)
        {
            deviceMemory.medianOfMediansStepOne = cl::Buffer(*openclRunTime.context, CL_MEM_READ_WRITE, options.nrSynthesizedBeamsPerChunk * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / observation.getDownsampling() / options.medianStepSize, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType)) * sizeof(outputDataType));
            deviceMemory.medianOfMediansStepTwo = cl::Buffer(*openclRunTime.context, CL_MEM_READ_WRITE, options.nrSynthesizedBeamsPerChunk * observation.getNrDMs(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType)) * sizeof(outputDataType));
        }
        else
        {
            deviceMemory.medianOfMediansStepOne = cl::Buffer(*openclRunTime.context, CL_MEM_READ_WRITE, options.nrSynthesizedBeamsPerChunk * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / observation.getDownsampling() / options.medianStepSize, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType)) * sizeof(outputDataType));
            deviceMemory.medianOfMediansStepTwo = cl::Buffer(*openclRunTime.context, CL_MEM_READ_WRITE, options.nrSynthesizedBeamsPerChunk * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType)) * sizeof(outputDataType));
        }
    }
    openclRunTime.queues->at(deviceOptions.deviceID).at(0).finish();
}
