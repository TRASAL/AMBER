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

#include <Pipeline.hpp>
#include <Trigger.hpp>

void pipeline(const OpenCLRunTime &openclRunTime, const AstroData::Observation &observation, const Options &options, const DeviceOptions &deviceOptions, const DataOptions &dataOptions, Timers &timers, const Kernels &kernels, const KernelConfigurations &kernelConfigurations, const KernelRunTimeConfigurations &kernelRunTimeConfigurations, HostMemory &hostMemory, const DeviceMemory &deviceMemory, HostMemoryDumpFiles &hostMemoryDumpFiles)
{
    bool errorDetected = false;
    int status = 0;
    std::ofstream outputTrigger;
    cl::Event syncPoint;

    if (options.dataDump)
    {
        if (options.subbandDedispersion)
        {
            hostMemoryDumpFiles.subbandedData.open(hostMemoryDumpFiles.dumpFilesPrefix + "subbandedData.dump");
            if (options.snrMode == SNRMode::Momad)
            {
                hostMemory.medianOfMediansStepOne.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / options.medianStepSize, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType)));
            }
        }
        else
        {
            if (options.snrMode == SNRMode::Momad)
            {
                hostMemory.medianOfMediansStepOne.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / options.medianStepSize, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType)));
            }
        }
        hostMemoryDumpFiles.dedispersedData.open(hostMemoryDumpFiles.dumpFilesPrefix + "dedispersedData.dump");
        hostMemoryDumpFiles.integratedData.open(hostMemoryDumpFiles.dumpFilesPrefix + "integratedData.dump");
        if (options.snrMode == SNRMode::Standard)
        {
            hostMemoryDumpFiles.snrData.open(hostMemoryDumpFiles.dumpFilesPrefix + "snrData.dump");
            hostMemoryDumpFiles.snrSamplesData.open(hostMemoryDumpFiles.dumpFilesPrefix + "snrSamplesData.dump");
        }
        else if (options.snrMode == SNRMode::Momad)
        {
            hostMemoryDumpFiles.maxValuesData.open(hostMemoryDumpFiles.dumpFilesPrefix + "maxValuesData.dump");
            hostMemoryDumpFiles.maxIndicesData.open(hostMemoryDumpFiles.dumpFilesPrefix + "maxIndicesData.dump");
            hostMemoryDumpFiles.medianOfMediansStepOneData.open(hostMemoryDumpFiles.dumpFilesPrefix + "medianOfMediansStepOneData.dump");
            hostMemoryDumpFiles.medianOfMediansData.open(hostMemoryDumpFiles.dumpFilesPrefix + "medianOfMediansData.dump");
            hostMemoryDumpFiles.medianOfMediansAbsoluteDeviationData.open(hostMemoryDumpFiles.dumpFilesPrefix + "medianOfMediansAbsoluteDeviation.dump");
        }
    }
    timers.search.start();
    outputTrigger.open(dataOptions.outputFile + ".trigger");
    if (!outputTrigger)
    {
        std::cerr << "Impossible to open " + dataOptions.outputFile + "." << std::endl;
        throw std::exception();
    }
    if (options.compactResults)
    {
        outputTrigger << "# beam batch sample integration_step compacted_integration_steps time DM compacted_DMs SNR";
        outputTrigger << std::endl;
    }
    else
    {
        outputTrigger << "# beam batch sample integration_step time DM SNR" << std::endl;
    }
    for (unsigned int batch = 0; batch < observation.getNrBatches(); batch++)
    {
        TriggeredEvents triggeredEvents(observation.getNrSynthesizedBeams());
        CompactedEvents compactedEvents(observation.getNrSynthesizedBeams());

        status = inputHandling(batch, observation, options, deviceOptions, dataOptions, timers, hostMemory, deviceMemory);
        if (status == 1)
        {
            // Not enough data for this iteration, move to next.
            continue;
        }
        else if (status == -1)
        {
            // Not enough batches remaining, exit the main loop.
            break;
        }
        status = copyInputToDevice(batch, openclRunTime, observation, options, deviceOptions, timers, hostMemory, deviceMemory);
        if (status != 0)
        {
            break;
        }
        if (options.splitBatchesDedispersion && (batch < observation.getNrDelayBatches()))
        {
            // Not enough batches in the buffer to start the search
            continue;
        }

        if (options.dataDump)
        {
            if (options.subbandDedispersion)
            {
                hostMemoryDumpFiles.subbandedData << "# Batch: " << batch << std::endl;
            }
            hostMemoryDumpFiles.dedispersedData << "# Batch: " << batch << std::endl;
            hostMemoryDumpFiles.integratedData << "# Batch: " << batch << std::endl;
            if (options.snrMode == SNRMode::Standard)
            {
                hostMemoryDumpFiles.snrData << "# Batch: " << batch << std::endl;
                hostMemoryDumpFiles.snrSamplesData << "# Batch: " << batch << std::endl;
            }
            else if (options.snrMode == SNRMode::Momad)
            {
                hostMemoryDumpFiles.maxValuesData << "# Batch: " << batch << std::endl;
                hostMemoryDumpFiles.maxIndicesData << "# Batch: " << batch << std::endl;
                hostMemoryDumpFiles.medianOfMediansStepOneData << "# Batch: " << batch << std::endl;
                hostMemoryDumpFiles.medianOfMediansData << "# Batch: " << batch << std::endl;
                hostMemoryDumpFiles.medianOfMediansAbsoluteDeviationData << "# Batch: " << batch << std::endl;
            }
        }
        // Dedispersion
        if (options.subbandDedispersion)
        {
            if (options.splitBatchesDedispersion)
            {
                // TODO: implement or remove splitBatches mode
            }
            if (deviceOptions.synchronized)
            {
                try
                {
                    timers.dedispersionStepOne.start();
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*(kernels.dedispersionStepOne), cl::NullRange, kernelRunTimeConfigurations.dedispersionStepOneGlobal, kernelRunTimeConfigurations.dedispersionStepOneLocal, nullptr, &syncPoint);
                    syncPoint.wait();
                    timers.dedispersionStepOne.stop();
                }
                catch (cl::Error &err)
                {
                    std::cerr << "Dedispersion Step One error -- Batch: " << std::to_string(batch) << ", " << err.what() << " ";
                    std::cerr << err.err() << std::endl;
                    errorDetected = true;
                }
                try
                {
                    timers.dedispersionStepTwo.start();
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*(kernels.dedispersionStepTwo), cl::NullRange, kernelRunTimeConfigurations.dedispersionStepTwoGlobal, kernelRunTimeConfigurations.dedispersionStepTwoLocal, nullptr, &syncPoint);
                    syncPoint.wait();
                    timers.dedispersionStepTwo.stop();
                }
                catch (cl::Error &err)
                {
                    std::cerr << "Dedispersion Step Two error -- Batch: " << std::to_string(batch) << ", " << err.what() << " ";
                    std::cerr << err.err() << std::endl;
                    errorDetected = true;
                }
            }
            else
            {
                try
                {
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*(kernels.dedispersionStepOne), cl::NullRange, kernelRunTimeConfigurations.dedispersionStepOneGlobal, kernelRunTimeConfigurations.dedispersionStepOneLocal, nullptr, nullptr);
                }
                catch (cl::Error &err)
                {
                    std::cerr << "Dedispersion Step One error -- Batch: " << std::to_string(batch) << ", " << err.what() << " ";
                    std::cerr << err.err() << std::endl;
                    errorDetected = true;
                }
                try
                {
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*(kernels.dedispersionStepTwo), cl::NullRange, kernelRunTimeConfigurations.dedispersionStepTwoGlobal, kernelRunTimeConfigurations.dedispersionStepTwoLocal, nullptr, nullptr);
                }
                catch (cl::Error &err)
                {
                    std::cerr << "Dedispersion Step Two error -- Batch: " << std::to_string(batch) << ", " << err.what() << " ";
                    std::cerr << err.err() << std::endl;
                    errorDetected = true;
                }
            }
        }
        else
        {
            try
            {
                if (options.splitBatchesDedispersion)
                {
                    // TODO: implement or remove splitBatches mode
                }
                if (deviceOptions.synchronized)
                {
                    timers.dedispersionSingleStep.start();
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*(kernels.dedispersionSingleStep), cl::NullRange, kernelRunTimeConfigurations.dedispersionSingleStepGlobal, kernelRunTimeConfigurations.dedispersionSingleStepLocal, nullptr, &syncPoint);
                    syncPoint.wait();
                    timers.dedispersionSingleStep.stop();
                }
                else
                {
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*(kernels.dedispersionSingleStep), cl::NullRange, kernelRunTimeConfigurations.dedispersionSingleStepGlobal, kernelRunTimeConfigurations.dedispersionSingleStepLocal);
                }
            }
            catch (cl::Error &err)
            {
                std::cerr << "Dedispersion error -- Batch: " << std::to_string(batch) << ", " << err.what() << " " << err.err();
                std::cerr << std::endl;
                errorDetected = true;
            }
        }
        if (options.dataDump)
        {
            if (options.subbandDedispersion)
            {
                try
                {
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.subbandedData, CL_TRUE, 0, hostMemory.subbandedData.size() * sizeof(outputDataType), reinterpret_cast<void *>(hostMemory.subbandedData.data()), nullptr, &syncPoint);
                    syncPoint.wait();
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.dedispersedData, CL_TRUE, 0, hostMemory.dedispersedData.size() * sizeof(outputDataType), reinterpret_cast<void *>(hostMemory.dedispersedData.data()), nullptr, &syncPoint);
                    syncPoint.wait();
                    for (unsigned int beam = 0; beam < observation.getNrBeams(); beam++)
                    {
                        hostMemoryDumpFiles.subbandedData << "# Beam: " << beam << std::endl;
                        for (unsigned int dm = 0; dm < observation.getNrDMs(true); dm++)
                        {
                            hostMemoryDumpFiles.subbandedData << "# Subbanding DM: " << dm << std::endl;
                            for (unsigned int subband = 0; subband < observation.getNrSubbands(); subband++)
                            {
                                hostMemoryDumpFiles.subbandedData << "# Subband: " << subband << std::endl;
                                for (unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(true); sample++)
                                {
                                    hostMemoryDumpFiles.subbandedData << hostMemory.subbandedData
                                                                             .at((beam * observation.getNrDMs(true) * observation.getNrSubbands() * observation.getNrSamplesPerBatch(true, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType))) + (dm * observation.getNrSubbands() * observation.getNrSamplesPerBatch(true, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType))) +
                                                                                 (subband * observation.getNrSamplesPerBatch(true,
                                                                                                                             deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType))) +
                                                                                 sample)
                                                                      << std::endl;
                                }
                                hostMemoryDumpFiles.subbandedData << std::endl << std::endl;
                            }
                            hostMemoryDumpFiles.subbandedData << std::endl;
                        }
                        hostMemoryDumpFiles.subbandedData << std::endl;
                    }
                    for (unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++)
                    {
                        hostMemoryDumpFiles.dedispersedData << "# Synthesized Beam: " << sBeam << std::endl;
                        for (unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++)
                        {
                            for (unsigned int dm = 0; dm < observation.getNrDMs(); dm++)
                            {
                                hostMemoryDumpFiles.dedispersedData << "# DM: " << (subbandingDM * observation.getNrDMs()) + dm << std::endl;
                                for (unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++)
                                {
                                    hostMemoryDumpFiles.dedispersedData << hostMemory.dedispersedData
                                                                               .at((sBeam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType))) +
                                                                                   (dm * observation.getNrSamplesPerBatch(false,
                                                                                                                          deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType))) +
                                                                                   sample)
                                                                        << std::endl;
                                }
                                hostMemoryDumpFiles.dedispersedData << std::endl;
                            }
                            hostMemoryDumpFiles.dedispersedData << std::endl;
                        }
                        hostMemoryDumpFiles.dedispersedData << std::endl;
                    }
                }
                catch (cl::Error &err)
                {
                    std::cerr << "Impossible to read deviceMemory.subbandedData and deviceMemory.dedispersedData: ";
                    std::cerr << err.what() << " " << err.err() << std::endl;
                    errorDetected = true;
                }
            }
            else
            {
                try
                {
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.dedispersedData, CL_TRUE, 0, hostMemory.dedispersedData.size() * sizeof(outputDataType), reinterpret_cast<void *>(hostMemory.dedispersedData.data()), nullptr, &syncPoint);
                    syncPoint.wait();
                    for (unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++)
                    {
                        hostMemoryDumpFiles.dedispersedData << "# Synthesized Beam: " << sBeam << std::endl;
                        for (unsigned int dm = 0; dm < observation.getNrDMs(); dm++)
                        {
                            hostMemoryDumpFiles.dedispersedData << "# DM: " << dm << std::endl;
                            for (unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++)
                            {
                                hostMemoryDumpFiles.dedispersedData << hostMemory.dedispersedData
                                                                           .at((sBeam * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType))) + (dm * observation.getNrSamplesPerBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType))) + sample)
                                                                    << std::endl;
                            }
                            hostMemoryDumpFiles.dedispersedData << std::endl << std::endl;
                        }
                        hostMemoryDumpFiles.dedispersedData << std::endl;
                    }
                }
                catch (cl::Error &err)
                {
                    std::cerr << "Impossible to read deviceMemory.dedispersedData: " << err.what() << " " << err.err();
                    std::cerr << std::endl;
                    errorDetected = true;
                }
            }
        }

        // SNR of dedispersed data
        if (options.snrMode == SNRMode::Standard)
        {
            try
            {
                if (deviceOptions.synchronized)
                {
                    timers.snr.start();
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.snr[hostMemory.integrationSteps.size()], cl::NullRange, kernelRunTimeConfigurations.snrGlobal[hostMemory.integrationSteps.size()], kernelRunTimeConfigurations.snrLocal[hostMemory.integrationSteps.size()], nullptr, &syncPoint);
                    syncPoint.wait();
                    timers.snr.stop();
                    timers.outputCopy.start();
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.snrData, CL_TRUE, 0, hostMemory.snrData.size() * sizeof(float), reinterpret_cast<void *>(hostMemory.snrData.data()), nullptr, &syncPoint);
                    syncPoint.wait();
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.snrSamples, CL_TRUE, 0, hostMemory.snrSamples.size() * sizeof(unsigned int), reinterpret_cast<void *>(hostMemory.snrSamples.data()), nullptr, &syncPoint);
                    syncPoint.wait();
                    timers.outputCopy.stop();
                }
                else
                {
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.snr[hostMemory.integrationSteps.size()], cl::NullRange, kernelRunTimeConfigurations.snrGlobal[hostMemory.integrationSteps.size()], kernelRunTimeConfigurations.snrLocal[hostMemory.integrationSteps.size()]);
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.snrData, CL_FALSE, 0, hostMemory.snrData.size() * sizeof(float), reinterpret_cast<void *>(hostMemory.snrData.data()));
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.snrSamples, CL_FALSE, 0, hostMemory.snrSamples.size() * sizeof(unsigned int), reinterpret_cast<void *>(hostMemory.snrSamples.data()));
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).finish();
                }
            }
            catch (cl::Error &err)
            {
                std::cerr << "SNR dedispersed data error -- Batch: " << std::to_string(batch) << ", " << err.what() << " ";
                std::cerr << err.err() << std::endl;
                errorDetected = true;
            }
            timers.trigger.start();
            trigger(options, deviceOptions.padding.at(deviceOptions.deviceName), 0, observation, hostMemory, triggeredEvents);
            timers.trigger.stop();
            if (options.dataDump)
            {
                if (options.subbandDedispersion)
                {
                    for (unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++)
                    {
                        hostMemoryDumpFiles.snrData << "# Synthesized Beam: " << sBeam << std::endl;
                        hostMemoryDumpFiles.snrSamplesData << "# Synthesized Beam: " << sBeam << std::endl;
                        for (unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++)
                        {
                            for (unsigned int dm = 0; dm < observation.getNrDMs(); dm++)
                            {
                                hostMemoryDumpFiles.snrData << hostMemory.snrData.at((sBeam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(float))) + (subbandingDM * observation.getNrDMs()) + dm);
                                hostMemoryDumpFiles.snrData << std::endl;
                                hostMemoryDumpFiles.snrSamplesData << hostMemory.snrSamples.at((sBeam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(unsigned int))) + (subbandingDM * observation.getNrDMs()) + dm);
                                hostMemoryDumpFiles.snrSamplesData << std::endl;
                            }
                        }
                        hostMemoryDumpFiles.snrData << std::endl << std::endl;
                        hostMemoryDumpFiles.snrSamplesData << std::endl << std::endl;
                    }
                }
                else
                {
                    for (unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++)
                    {
                        hostMemoryDumpFiles.snrData << "# Synthesized Beam: " << sBeam << std::endl;
                        hostMemoryDumpFiles.snrSamplesData << "# Synthesized Beam: " << sBeam << std::endl;
                        for (unsigned int dm = 0; dm < observation.getNrDMs(); dm++)
                        {
                            hostMemoryDumpFiles.snrData << hostMemory.snrData.at((sBeam * observation.getNrDMs(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(float))) + dm);
                            hostMemoryDumpFiles.snrData << std::endl;
                            hostMemoryDumpFiles.snrSamplesData << hostMemory.snrSamples.at((sBeam * observation.getNrDMs(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(unsigned int))) + dm);
                            hostMemoryDumpFiles.snrSamplesData << std::endl;
                        }
                        hostMemoryDumpFiles.snrData << std::endl << std::endl;
                        hostMemoryDumpFiles.snrSamplesData << std::endl << std::endl;
                    }
                }
                hostMemoryDumpFiles.snrData << std::endl;
                hostMemoryDumpFiles.snrSamplesData << std::endl;
            }
        }
        else if (options.snrMode == SNRMode::Momad)
        {
            try
            {
                if (deviceOptions.synchronized)
                {
                    // Max
                    timers.max.start();
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.max[hostMemory.integrationSteps.size()], cl::NullRange, kernelRunTimeConfigurations.maxGlobal[hostMemory.integrationSteps.size()], kernelRunTimeConfigurations.maxLocal[hostMemory.integrationSteps.size()], nullptr, &syncPoint);
                    syncPoint.wait();
                    timers.max.stop();
                    // Transfer of max values to host
                    timers.outputCopy.start();
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.maxValues, CL_TRUE, 0, hostMemory.maxValues.size() * sizeof(outputDataType), reinterpret_cast<void *>(hostMemory.maxValues.data()), nullptr, &syncPoint);
                    syncPoint.wait();
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.maxIndices, CL_TRUE, 0, hostMemory.maxIndices.size() * sizeof(unsigned int), reinterpret_cast<void *>(hostMemory.maxIndices.data()), nullptr, &syncPoint);
                    syncPoint.wait();
                    timers.outputCopy.stop();
                    // Median of medians first step
                    timers.medianOfMediansStepOne.start();
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.medianOfMediansStepOne[hostMemory.integrationSteps.size()], cl::NullRange, kernelRunTimeConfigurations.medianOfMediansStepOneGlobal[hostMemory.integrationSteps.size()], kernelRunTimeConfigurations.medianOfMediansStepOneLocal[hostMemory.integrationSteps.size()], nullptr, &syncPoint);
                    syncPoint.wait();
                    timers.medianOfMediansStepOne.stop();
                    // Median of medians second step
                    timers.medianOfMediansStepTwo.start();
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.medianOfMediansStepTwo[hostMemory.integrationSteps.size()], cl::NullRange, kernelRunTimeConfigurations.medianOfMediansStepTwoGlobal[hostMemory.integrationSteps.size()], kernelRunTimeConfigurations.medianOfMediansStepTwoLocal[hostMemory.integrationSteps.size()], nullptr, &syncPoint);
                    syncPoint.wait();
                    timers.medianOfMediansStepTwo.stop();
                    // Trasfer of medians of medians to host
                    timers.outputCopy.start();
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.medianOfMediansStepTwo, CL_TRUE, 0, hostMemory.medianOfMedians.size() * sizeof(outputDataType), reinterpret_cast<void *>(hostMemory.medianOfMedians.data()), nullptr, &syncPoint);
                    syncPoint.wait();
                    timers.outputCopy.stop();
                    // Median of medians absolute deviation first step
                    timers.medianOfMediansAbsoluteDeviationStepOne.start();
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.medianOfMediansAbsoluteDeviation[hostMemory.integrationSteps.size()], cl::NullRange, kernelRunTimeConfigurations.medianOfMediansAbsoluteDeviationGlobal[hostMemory.integrationSteps.size()], kernelRunTimeConfigurations.medianOfMediansAbsoluteDeviationLocal[hostMemory.integrationSteps.size()], nullptr, &syncPoint);
                    syncPoint.wait();
                    timers.medianOfMediansAbsoluteDeviationStepOne.stop();
                    // Median of medians absolute deviation second step
                    timers.medianOfMediansAbsoluteDeviationStepTwo.start();
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.medianOfMediansStepTwo[hostMemory.integrationSteps.size()], cl::NullRange, kernelRunTimeConfigurations.medianOfMediansStepTwoGlobal[hostMemory.integrationSteps.size()], kernelRunTimeConfigurations.medianOfMediansStepTwoLocal[hostMemory.integrationSteps.size()], nullptr, &syncPoint);
                    syncPoint.wait();
                    timers.medianOfMediansAbsoluteDeviationStepTwo.stop();
                    // Transfers of median of medians absolute deviation to host
                    timers.outputCopy.start();
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.medianOfMediansStepTwo, CL_TRUE, 0, hostMemory.medianOfMediansAbsoluteDeviation.size() * sizeof(outputDataType), reinterpret_cast<void *>(hostMemory.medianOfMediansAbsoluteDeviation.data()), nullptr, &syncPoint);
                    syncPoint.wait();
                    timers.outputCopy.stop();
                }
                else
                {
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.max[hostMemory.integrationSteps.size()], cl::NullRange, kernelRunTimeConfigurations.maxGlobal[hostMemory.integrationSteps.size()], kernelRunTimeConfigurations.maxLocal[hostMemory.integrationSteps.size()]);
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.maxValues, CL_FALSE, 0, hostMemory.maxValues.size() * sizeof(outputDataType), reinterpret_cast<void *>(hostMemory.maxValues.data()));
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.maxIndices, CL_FALSE, 0, hostMemory.maxIndices.size() * sizeof(unsigned int), reinterpret_cast<void *>(hostMemory.maxIndices.data()));
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.medianOfMediansStepOne[hostMemory.integrationSteps.size()], cl::NullRange, kernelRunTimeConfigurations.medianOfMediansStepOneGlobal[hostMemory.integrationSteps.size()], kernelRunTimeConfigurations.medianOfMediansStepOneLocal[hostMemory.integrationSteps.size()]);
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.medianOfMediansStepTwo[hostMemory.integrationSteps.size()], cl::NullRange, kernelRunTimeConfigurations.medianOfMediansStepTwoGlobal[hostMemory.integrationSteps.size()], kernelRunTimeConfigurations.medianOfMediansStepTwoLocal[hostMemory.integrationSteps.size()]);
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.medianOfMediansStepTwo, CL_FALSE, 0, hostMemory.medianOfMedians.size() * sizeof(outputDataType), reinterpret_cast<void *>(hostMemory.medianOfMedians.data()));
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.medianOfMediansAbsoluteDeviation[hostMemory.integrationSteps.size()], cl::NullRange, kernelRunTimeConfigurations.medianOfMediansAbsoluteDeviationGlobal[hostMemory.integrationSteps.size()], kernelRunTimeConfigurations.medianOfMediansAbsoluteDeviationLocal[hostMemory.integrationSteps.size()]);
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.medianOfMediansStepTwo[hostMemory.integrationSteps.size()], cl::NullRange, kernelRunTimeConfigurations.medianOfMediansStepTwoGlobal[hostMemory.integrationSteps.size()], kernelRunTimeConfigurations.medianOfMediansStepTwoLocal[hostMemory.integrationSteps.size()]);
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.medianOfMediansStepTwo, CL_FALSE, 0, hostMemory.medianOfMediansAbsoluteDeviation.size() * sizeof(outputDataType), reinterpret_cast<void *>(hostMemory.medianOfMediansAbsoluteDeviation.data()));
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).finish();
                }
            }
            catch (cl::Error &err)
            {
                std::cerr << "MOMAD dedispersed data error -- Batch: " << std::to_string(batch) << ", " << err.what() << " ";
                std::cerr << err.err() << std::endl;
                errorDetected = true;
            }
            timers.trigger.start();
            trigger(options, deviceOptions.padding.at(deviceOptions.deviceName), 0, observation, hostMemory, triggeredEvents);
            timers.trigger.stop();
            if (options.dataDump)
            {
                try
                {
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.medianOfMediansStepOne, CL_TRUE, 0, hostMemory.medianOfMediansStepOne.size() * sizeof(outputDataType), reinterpret_cast<void *>(hostMemory.medianOfMediansStepOne.data()), nullptr, &syncPoint);
                    syncPoint.wait();
                }
                catch (cl::Error &err)
                {
                    std::cerr << "Impossible to read deviceMemory.medianOfMediansStepOne: " << err.what() << " " << err.err();
                    std::cerr << std::endl;
                    errorDetected = true;
                }
                if (options.subbandDedispersion)
                {
                    for (unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++)
                    {
                        hostMemoryDumpFiles.maxValuesData << "# Synthesized Beam: " << sBeam << std::endl;
                        hostMemoryDumpFiles.maxIndicesData << "# Synthesized Beam: " << sBeam << std::endl;
                        hostMemoryDumpFiles.medianOfMediansStepOneData << "# Synthesized Beam: " << sBeam << std::endl;
                        hostMemoryDumpFiles.medianOfMediansData << "# Synthesized Beam: " << sBeam << std::endl;
                        hostMemoryDumpFiles.medianOfMediansAbsoluteDeviationData << "# Synthesized Beam: " << sBeam << std::endl;
                        for (unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++)
                        {
                            for (unsigned int dm = 0; dm < observation.getNrDMs(); dm++)
                            {
                                hostMemoryDumpFiles.maxValuesData << hostMemory.maxValues.at((sBeam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(float))) + (subbandingDM * observation.getNrDMs()) + dm);
                                hostMemoryDumpFiles.maxValuesData << std::endl;
                                hostMemoryDumpFiles.maxIndicesData << hostMemory.maxIndices.at((sBeam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(unsigned int))) + (subbandingDM * observation.getNrDMs()) + dm);
                                hostMemoryDumpFiles.maxIndicesData << std::endl;
                                hostMemoryDumpFiles.medianOfMediansData << hostMemory.medianOfMedians.at((sBeam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(float))) + (subbandingDM * observation.getNrDMs()) + dm);
                                hostMemoryDumpFiles.medianOfMediansData << std::endl;
                                hostMemoryDumpFiles.medianOfMediansAbsoluteDeviationData << hostMemory.medianOfMediansAbsoluteDeviation.at((sBeam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(float))) + (subbandingDM * observation.getNrDMs()) + dm);
                                hostMemoryDumpFiles.medianOfMediansAbsoluteDeviationData << std::endl;
                                hostMemoryDumpFiles.medianOfMediansStepOneData << "# DM: " << (subbandingDM * observation.getNrDMs()) + dm << std::endl;
                                for (unsigned int sample = 0; sample < observation.getNrSamplesPerBatch() / options.medianStepSize; sample++ )
                                {
                                    hostMemoryDumpFiles.medianOfMediansStepOneData << hostMemory.medianOfMediansStepOne.at((sBeam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / options.medianStepSize, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType))) + (((subbandingDM * observation.getNrDMs()) + dm) * isa::utils::pad(observation.getNrSamplesPerBatch() / options.medianStepSize, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType))) + sample) << std::endl;
                                }
                                hostMemoryDumpFiles.medianOfMediansStepOneData << std::endl << std::endl;
                            }
                        }
                        hostMemoryDumpFiles.maxValuesData << std::endl << std::endl;
                        hostMemoryDumpFiles.maxIndicesData << std::endl << std::endl;
                        hostMemoryDumpFiles.medianOfMediansData << std::endl << std::endl;
                        hostMemoryDumpFiles.medianOfMediansAbsoluteDeviationData << std::endl << std::endl;
                    }
                }
                else
                {
                    for (unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++)
                    {
                        hostMemoryDumpFiles.maxValuesData << "# Synthesized Beam: " << sBeam << std::endl;
                        hostMemoryDumpFiles.maxIndicesData << "# Synthesized Beam: " << sBeam << std::endl;
                        hostMemoryDumpFiles.medianOfMediansStepOneData << "# Synthesized Beam: " << sBeam << std::endl;
                        hostMemoryDumpFiles.medianOfMediansData << "# Synthesized Beam: " << sBeam << std::endl;
                        hostMemoryDumpFiles.medianOfMediansAbsoluteDeviationData << "# Synthesized Beam: " << sBeam << std::endl;
                        for (unsigned int dm = 0; dm < observation.getNrDMs(); dm++)
                        {
                            hostMemoryDumpFiles.maxValuesData << hostMemory.maxValues.at((sBeam * observation.getNrDMs(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(float))) + dm);
                            hostMemoryDumpFiles.maxValuesData << std::endl;
                            hostMemoryDumpFiles.maxIndicesData << hostMemory.maxIndices.at((sBeam * observation.getNrDMs(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(float))) + dm);
                            hostMemoryDumpFiles.maxIndicesData << std::endl;
                            hostMemoryDumpFiles.medianOfMediansData << hostMemory.medianOfMedians.at((sBeam * observation.getNrDMs(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(float))) + dm);
                            hostMemoryDumpFiles.medianOfMediansData << std::endl;
                            hostMemoryDumpFiles.medianOfMediansAbsoluteDeviationData << hostMemory.medianOfMediansAbsoluteDeviation.at((sBeam * observation.getNrDMs(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(float))) + dm);
                            hostMemoryDumpFiles.medianOfMediansAbsoluteDeviationData << std::endl;
                            hostMemoryDumpFiles.medianOfMediansStepOneData << "# DM: " << dm << std::endl;
                            for (unsigned int sample = 0; sample < observation.getNrSamplesPerBatch() / options.medianStepSize; sample++ )
                            {
                                hostMemoryDumpFiles.medianOfMediansStepOneData << hostMemory.medianOfMediansStepOne.at((sBeam * observation.getNrDMs()) * isa::utils::pad(observation.getNrSamplesPerBatch() / options.medianStepSize, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType))) + ((dm * isa::utils::pad(observation.getNrSamplesPerBatch() / options.medianStepSize, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType))) + sample) << std::endl;
                            }
                            hostMemoryDumpFiles.medianOfMediansStepOneData << std::endl << std::endl;
                        }
                        hostMemoryDumpFiles.maxValuesData << std::endl << std::endl;
                        hostMemoryDumpFiles.maxIndicesData << std::endl << std::endl;
                        hostMemoryDumpFiles.medianOfMediansData << std::endl << std::endl;
                        hostMemoryDumpFiles.medianOfMediansAbsoluteDeviationData << std::endl << std::endl;
                    }
                }
            }
        }

        // Integration and SNR loop
        for (unsigned int stepNumber = 0; stepNumber < hostMemory.integrationSteps.size(); stepNumber++)
        {
            auto step = hostMemory.integrationSteps.begin();

            std::advance(step, stepNumber);
            try
            {
                if (deviceOptions.synchronized)
                {
                    timers.integration.start();
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.integration[stepNumber], cl::NullRange, kernelRunTimeConfigurations.integrationGlobal[stepNumber], kernelRunTimeConfigurations.integrationLocal[stepNumber], nullptr, &syncPoint);
                    syncPoint.wait();
                    timers.integration.stop();
                    if (options.snrMode == SNRMode::Standard)
                    {
                        timers.snr.start();
                        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.snr[stepNumber], cl::NullRange, kernelRunTimeConfigurations.snrGlobal[stepNumber], kernelRunTimeConfigurations.snrLocal[stepNumber], nullptr, &syncPoint);
                        syncPoint.wait();
                        timers.snr.stop();
                        timers.outputCopy.start();
                        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.snrData, CL_TRUE, 0, hostMemory.snrData.size() * sizeof(float), reinterpret_cast<void *>(hostMemory.snrData.data()), nullptr, &syncPoint);
                        syncPoint.wait();
                        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.snrSamples, CL_TRUE, 0, hostMemory.snrSamples.size() * sizeof(unsigned int), reinterpret_cast<void *>(hostMemory.snrSamples.data()), nullptr, &syncPoint);
                        syncPoint.wait();
                        timers.outputCopy.stop();
                    }
                    else if (options.snrMode == SNRMode::Momad)
                    {
                        // Max
                        timers.max.start();
                        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.max[stepNumber], cl::NullRange, kernelRunTimeConfigurations.maxGlobal[stepNumber], kernelRunTimeConfigurations.maxLocal[stepNumber], nullptr, &syncPoint);
                        syncPoint.wait();
                        timers.max.stop();
                        // Transfer of max values to host
                        timers.outputCopy.start();
                        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.maxValues, CL_TRUE, 0, hostMemory.maxValues.size() * sizeof(outputDataType), reinterpret_cast<void *>(hostMemory.maxValues.data()), nullptr, &syncPoint);
                        syncPoint.wait();
                        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.maxIndices, CL_TRUE, 0, hostMemory.maxIndices.size() * sizeof(unsigned int), reinterpret_cast<void *>(hostMemory.maxIndices.data()), nullptr, &syncPoint);
                        syncPoint.wait();
                        timers.outputCopy.stop();
                        // Median of medians first step
                        timers.medianOfMediansStepOne.start();
                        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.medianOfMediansStepOne[stepNumber], cl::NullRange, kernelRunTimeConfigurations.medianOfMediansStepOneGlobal[stepNumber], kernelRunTimeConfigurations.medianOfMediansStepOneLocal[stepNumber], nullptr, &syncPoint);
                        syncPoint.wait();
                        timers.medianOfMediansStepOne.stop();
                        // Median of medians second step
                        timers.medianOfMediansStepTwo.start();
                        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.medianOfMediansStepTwo[stepNumber], cl::NullRange, kernelRunTimeConfigurations.medianOfMediansStepTwoGlobal[stepNumber], kernelRunTimeConfigurations.medianOfMediansStepTwoLocal[stepNumber], nullptr, &syncPoint);
                        syncPoint.wait();
                        timers.medianOfMediansStepTwo.stop();
                        // Trasfer of medians of medians to host
                        timers.outputCopy.start();
                        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.medianOfMediansStepTwo, CL_TRUE, 0, hostMemory.medianOfMedians.size() * sizeof(outputDataType), reinterpret_cast<void *>(hostMemory.medianOfMedians.data()), nullptr, &syncPoint);
                        syncPoint.wait();
                        timers.outputCopy.stop();
                        // Median of medians absolute deviation first step
                        timers.medianOfMediansAbsoluteDeviationStepOne.start();
                        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.medianOfMediansAbsoluteDeviation[stepNumber], cl::NullRange, kernelRunTimeConfigurations.medianOfMediansAbsoluteDeviationGlobal[stepNumber], kernelRunTimeConfigurations.medianOfMediansAbsoluteDeviationLocal[stepNumber], nullptr, &syncPoint);
                        syncPoint.wait();
                        timers.medianOfMediansAbsoluteDeviationStepOne.stop();
                        // Median of medians absolute deviation second step
                        timers.medianOfMediansAbsoluteDeviationStepTwo.start();
                        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.medianOfMediansStepTwo[stepNumber], cl::NullRange, kernelRunTimeConfigurations.medianOfMediansStepTwoGlobal[stepNumber], kernelRunTimeConfigurations.medianOfMediansStepTwoLocal[stepNumber], nullptr, &syncPoint);
                        syncPoint.wait();
                        timers.medianOfMediansAbsoluteDeviationStepTwo.stop();
                        // Transfers of median of medians absolute deviation to host
                        timers.outputCopy.start();
                        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.medianOfMediansStepTwo, CL_TRUE, 0, hostMemory.medianOfMediansAbsoluteDeviation.size() * sizeof(outputDataType), reinterpret_cast<void *>(hostMemory.medianOfMediansAbsoluteDeviation.data()), nullptr, &syncPoint);
                        syncPoint.wait();
                        timers.outputCopy.stop();
                    }
                }
                else
                {
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.integration[stepNumber], cl::NullRange, kernelRunTimeConfigurations.integrationGlobal[stepNumber], kernelRunTimeConfigurations.integrationLocal[stepNumber]);
                    if (options.snrMode == SNRMode::Standard)
                    {
                        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.snr[stepNumber], cl::NullRange, kernelRunTimeConfigurations.snrGlobal[stepNumber], kernelRunTimeConfigurations.snrLocal[stepNumber]);
                        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.snrData, CL_FALSE, 0, hostMemory.snrData.size() * sizeof(float), reinterpret_cast<void *>(hostMemory.snrData.data()));
                        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.snrSamples, CL_FALSE, 0, hostMemory.snrSamples.size() * sizeof(unsigned int), reinterpret_cast<void *>(hostMemory.snrSamples.data()));
                    }
                    else if (options.snrMode == SNRMode::Momad)
                    {
                        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.max[stepNumber], cl::NullRange, kernelRunTimeConfigurations.maxGlobal[stepNumber], kernelRunTimeConfigurations.maxLocal[stepNumber]);
                        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.maxValues, CL_FALSE, 0, hostMemory.maxValues.size() * sizeof(outputDataType), reinterpret_cast<void *>(hostMemory.maxValues.data()));
                        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.maxIndices, CL_FALSE, 0, hostMemory.maxIndices.size() * sizeof(unsigned int), reinterpret_cast<void *>(hostMemory.maxIndices.data()));
                        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.medianOfMediansStepOne[stepNumber], cl::NullRange, kernelRunTimeConfigurations.medianOfMediansStepOneGlobal[stepNumber], kernelRunTimeConfigurations.medianOfMediansStepOneLocal[stepNumber]);
                        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.medianOfMediansStepTwo[stepNumber], cl::NullRange, kernelRunTimeConfigurations.medianOfMediansStepTwoGlobal[stepNumber], kernelRunTimeConfigurations.medianOfMediansStepTwoLocal[stepNumber]);
                        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.medianOfMediansStepTwo, CL_FALSE, 0, hostMemory.medianOfMedians.size() * sizeof(outputDataType), reinterpret_cast<void *>(hostMemory.medianOfMedians.data()));
                        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.medianOfMediansAbsoluteDeviation[stepNumber], cl::NullRange, kernelRunTimeConfigurations.medianOfMediansAbsoluteDeviationGlobal[stepNumber], kernelRunTimeConfigurations.medianOfMediansAbsoluteDeviationLocal[stepNumber]);
                        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueNDRangeKernel(*kernels.medianOfMediansStepTwo[stepNumber], cl::NullRange, kernelRunTimeConfigurations.medianOfMediansStepTwoGlobal[stepNumber], kernelRunTimeConfigurations.medianOfMediansStepTwoLocal[stepNumber]);
                        openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.medianOfMediansStepTwo, CL_FALSE, 0, hostMemory.medianOfMediansAbsoluteDeviation.size() * sizeof(outputDataType), reinterpret_cast<void *>(hostMemory.medianOfMediansAbsoluteDeviation.data()));
                    }
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).finish();
                }
            }
            catch (cl::Error &err)
            {
                std::cerr << "SNR integration loop error -- Batch: " << std::to_string(batch) << ", Step: ";
                std::cerr << std::to_string(*step) << ", " << err.what() << " " << err.err() << std::endl;
                errorDetected = true;
            }
            timers.trigger.start();
            trigger(options, deviceOptions.padding.at(deviceOptions.deviceName), *step, observation, hostMemory, triggeredEvents);
            timers.trigger.stop();
            if (options.dataDump)
            {
                try
                {
                    openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueReadBuffer(deviceMemory.integratedData, CL_TRUE, 0, hostMemory.integratedData.size() * sizeof(outputDataType), reinterpret_cast<void *>(hostMemory.integratedData.data()), nullptr, &syncPoint);
                    syncPoint.wait();
                }
                catch (cl::Error &err)
                {
                    std::cerr << "Impossible to read deviceMemory.integratedData: " << err.what() << " " << err.err();
                    std::cerr << std::endl;
                    errorDetected = true;
                }
                hostMemoryDumpFiles.integratedData << "# Integration: " << *step << std::endl;
                if (options.snrMode == SNRMode::Standard)
                {
                    hostMemoryDumpFiles.snrData << "# Integration: " << *step << std::endl;
                    hostMemoryDumpFiles.snrSamplesData << "# Integration: " << *step << std::endl;
                }
                else if (options.snrMode == SNRMode::Momad)
                {
                    hostMemoryDumpFiles.maxValuesData << "# Integration: " << *step << std::endl;
                    hostMemoryDumpFiles.maxIndicesData << "# Integration: " << *step << std::endl;
                    hostMemoryDumpFiles.medianOfMediansStepOneData << "# Integration: " << *step << std::endl;
                    hostMemoryDumpFiles.medianOfMediansData << "# Integration: " << *step << std::endl;
                    hostMemoryDumpFiles.medianOfMediansAbsoluteDeviationData << "# Integration: " << *step << std::endl;
                }
                if (options.subbandDedispersion)
                {
                    for (unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++)
                    {
                        hostMemoryDumpFiles.integratedData << "# Synthesized Beam: " << sBeam << std::endl;
                        if (options.snrMode == SNRMode::Standard)
                        {
                            hostMemoryDumpFiles.snrData << "# Synthesized Beam: " << sBeam << std::endl;
                            hostMemoryDumpFiles.snrSamplesData << "# Synthesized Beam: " << sBeam << std::endl;
                        }
                        else if (options.snrMode == SNRMode::Momad)
                        {
                            hostMemoryDumpFiles.maxValuesData << "# Synthesized Beam: " << sBeam << std::endl;
                            hostMemoryDumpFiles.maxIndicesData << "# Synthesized Beam: " << sBeam << std::endl;
                            hostMemoryDumpFiles.medianOfMediansStepOneData << "# Synthesized Beam: " << sBeam << std::endl;
                            hostMemoryDumpFiles.medianOfMediansData << "# Synthesized Beam: " << sBeam << std::endl;
                            hostMemoryDumpFiles.medianOfMediansAbsoluteDeviationData << "# Synthesized Beam: " << sBeam << std::endl;
                        }
                        for (unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++)
                        {
                            for (unsigned int dm = 0; dm < observation.getNrDMs(); dm++)
                            {
                                hostMemoryDumpFiles.integratedData << "# DM: " << (subbandingDM * observation.getNrDMs()) + dm << std::endl;
                                for (unsigned int sample = 0; sample < observation.getNrSamplesPerBatch() / *step; sample++)
                                {
                                    hostMemoryDumpFiles.integratedData << hostMemory.integratedData.at((sBeam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / *(hostMemory.integrationSteps.begin()), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / *(hostMemory.integrationSteps.begin()), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / *(hostMemory.integrationSteps.begin()), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType))) + sample) << std::endl;
                                }
                                hostMemoryDumpFiles.integratedData << std::endl << std::endl;
                                if (options.snrMode == SNRMode::Standard)
                                {
                                    hostMemoryDumpFiles.snrData << hostMemory.snrData.at((sBeam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(float))) + (subbandingDM * observation.getNrDMs()) + dm);
                                    hostMemoryDumpFiles.snrData << std::endl;
                                    hostMemoryDumpFiles.snrSamplesData << hostMemory.snrSamples.at((sBeam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(unsigned int))) + (subbandingDM * observation.getNrDMs()) + dm);
                                    hostMemoryDumpFiles.snrSamplesData << std::endl;
                                }
                                else if (options.snrMode == SNRMode::Momad)
                                {
                                    hostMemoryDumpFiles.maxValuesData << hostMemory.maxValues.at((sBeam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(float))) + (subbandingDM * observation.getNrDMs()) + dm);
                                    hostMemoryDumpFiles.maxValuesData << std::endl;
                                    hostMemoryDumpFiles.maxIndicesData << hostMemory.maxIndices.at((sBeam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(unsigned int))) + (subbandingDM * observation.getNrDMs()) + dm);
                                    hostMemoryDumpFiles.maxIndicesData << std::endl;
                                    hostMemoryDumpFiles.medianOfMediansData << hostMemory.medianOfMedians.at((sBeam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(float))) + (subbandingDM * observation.getNrDMs()) + dm);
                                    hostMemoryDumpFiles.medianOfMediansData << std::endl;
                                    hostMemoryDumpFiles.medianOfMediansAbsoluteDeviationData << hostMemory.medianOfMediansAbsoluteDeviation.at((sBeam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(float))) + (subbandingDM * observation.getNrDMs()) + dm);
                                    hostMemoryDumpFiles.medianOfMediansAbsoluteDeviationData << std::endl;
                                    hostMemoryDumpFiles.medianOfMediansStepOneData << "# DM: " << (subbandingDM * observation.getNrDMs()) + dm << std::endl;
                                    for (unsigned int sample = 0; sample < observation.getNrSamplesPerBatch() / *step / options.medianStepSize; sample++)
                                    {
                                        hostMemoryDumpFiles.medianOfMediansStepOneData << hostMemory.medianOfMediansStepOne.at((sBeam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / options.medianStepSize, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType))) + (((subbandingDM * observation.getNrDMs()) + dm) * isa::utils::pad(observation.getNrSamplesPerBatch() / options.medianStepSize, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType))) + sample) << std::endl;
                                    }
                                    hostMemoryDumpFiles.medianOfMediansStepOneData << std::endl << std::endl;
                                }
                            }
                        }
                        if (options.snrMode == SNRMode::Standard)
                        {
                            hostMemoryDumpFiles.snrData << std::endl << std::endl;
                            hostMemoryDumpFiles.snrSamplesData << std::endl << std::endl;
                        }
                        else if (options.snrMode == SNRMode::Momad)
                        {
                            hostMemoryDumpFiles.maxValuesData << std::endl << std::endl;
                            hostMemoryDumpFiles.maxIndicesData << std::endl << std::endl;
                            hostMemoryDumpFiles.medianOfMediansData << std::endl << std::endl;
                            hostMemoryDumpFiles.medianOfMediansAbsoluteDeviationData << std::endl << std::endl;
                        }
                    }
                }
                else
                {
                    for (unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++)
                    {
                        hostMemoryDumpFiles.integratedData << "# Synthesized Beam: " << sBeam << std::endl;
                        if (options.snrMode == SNRMode::Standard)
                        {
                            hostMemoryDumpFiles.snrData << "# Synthesized Beam: " << sBeam << std::endl;
                            hostMemoryDumpFiles.snrSamplesData << "# Synthesized Beam: " << sBeam << std::endl;
                        }
                        else if (options.snrMode == SNRMode::Momad)
                        {
                            hostMemoryDumpFiles.maxValuesData << "# Synthesized Beam: " << sBeam << std::endl;
                            hostMemoryDumpFiles.maxIndicesData << "# Synthesized Beam: " << sBeam << std::endl;
                            hostMemoryDumpFiles.medianOfMediansStepOneData << "# Synthesized Beam: " << sBeam << std::endl;
                            hostMemoryDumpFiles.medianOfMediansData << "# Synthesized Beam: " << sBeam << std::endl;
                            hostMemoryDumpFiles.medianOfMediansAbsoluteDeviationData << "# Synthesized Beam: " << sBeam << std::endl;
                        }
                        for (unsigned int dm = 0; dm < observation.getNrDMs(); dm++)
                        {
                            hostMemoryDumpFiles.integratedData << "# DM: " << dm << std::endl;
                            for (unsigned int sample = 0; sample < observation.getNrSamplesPerBatch() / *step; sample++)
                            {
                                std::cerr << hostMemory.integratedData.at((sBeam * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / *(hostMemory.integrationSteps.begin()), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / *(hostMemory.integrationSteps.begin()), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType))) + sample) << std::endl;
                            }
                            hostMemoryDumpFiles.integratedData << std::endl << std::endl;
                            if (options.snrMode == SNRMode::Standard)
                            {
                                hostMemoryDumpFiles.snrData << hostMemory.snrData.at((sBeam * observation.getNrDMs(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(float))) + dm);
                                hostMemoryDumpFiles.snrData << std::endl;
                                hostMemoryDumpFiles.snrSamplesData << hostMemory.snrSamples.at((sBeam * observation.getNrDMs(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(unsigned int))) + dm);
                                hostMemoryDumpFiles.snrSamplesData << std::endl;
                            }
                            else if (options.snrMode == SNRMode::Momad)
                            {
                                hostMemoryDumpFiles.maxValuesData << hostMemory.maxValues.at((sBeam * observation.getNrDMs(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(float))) + dm);
                                hostMemoryDumpFiles.maxValuesData << std::endl;
                                hostMemoryDumpFiles.maxIndicesData << hostMemory.maxIndices.at((sBeam * observation.getNrDMs(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(float))) + dm);
                                hostMemoryDumpFiles.maxIndicesData << std::endl;
                                hostMemoryDumpFiles.medianOfMediansData << hostMemory.medianOfMedians.at((sBeam * observation.getNrDMs(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(float))) + dm);
                                hostMemoryDumpFiles.medianOfMediansData << std::endl;
                                hostMemoryDumpFiles.medianOfMediansAbsoluteDeviationData << hostMemory.medianOfMediansAbsoluteDeviation.at((sBeam * observation.getNrDMs(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(float))) + dm);
                                hostMemoryDumpFiles.medianOfMediansAbsoluteDeviationData << std::endl;
                                hostMemoryDumpFiles.medianOfMediansStepOneData << "# DM: " << dm << std::endl;
                                for (unsigned int sample = 0; sample < observation.getNrSamplesPerBatch() / options.medianStepSize; sample++ )
                                {
                                    hostMemoryDumpFiles.medianOfMediansStepOneData << hostMemory.medianOfMediansStepOne.at((sBeam * observation.getNrDMs()) * isa::utils::pad(observation.getNrSamplesPerBatch() / options.medianStepSize, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType))) + ((dm * isa::utils::pad(observation.getNrSamplesPerBatch() / options.medianStepSize, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType))) + sample) << std::endl;
                                }
                                hostMemoryDumpFiles.medianOfMediansStepOneData << std::endl << std::endl;
                            }
                        }
                        if (options.snrMode == SNRMode::Standard)
                        {
                            hostMemoryDumpFiles.snrData << std::endl << std::endl;
                            hostMemoryDumpFiles.snrSamplesData << std::endl << std::endl;
                        }
                        else if (options.snrMode == SNRMode::Momad)
                        {
                            hostMemoryDumpFiles.maxValuesData << std::endl << std::endl;
                            hostMemoryDumpFiles.maxIndicesData << std::endl << std::endl;
                            hostMemoryDumpFiles.medianOfMediansData << std::endl << std::endl;
                            hostMemoryDumpFiles.medianOfMediansAbsoluteDeviationData << std::endl << std::endl;
                        }
                    }
                }
            }
        }
        if (errorDetected)
        {
            outputTrigger.close();
#ifdef HAVE_PSRDADA
            if (dataOptions.dataPSRDADA)
            {
                if (dada_hdu_unlock_read(hostMemory.ringBuffer) != 0)
                {
                    std::cerr << "Impossible to unlock the PSRDADA ringbuffer for reading the header." << std::endl;
                }
                dada_hdu_disconnect(hostMemory.ringBuffer);
            }
#endif // HAVE_PSRDADA
            return;
        }
        // Print and compact results
        timers.trigger.start();
        if (options.compactResults)
        {
            compact(observation, triggeredEvents, compactedEvents);
            for (auto &compactedEvent : compactedEvents)
            {
                for (auto &event : compactedEvent)
                {
                    unsigned int integration = 0;
                    float firstDM;
                    unsigned int delay = 0;

                    if (event.integration == 0)
                    {
                        integration = 1;
                    }
                    else
                    {
                        integration = event.integration;
                    }
                    if (options.subbandDedispersion)
                    {
                        if (dataOptions.dataPSRDADA || dataOptions.streamingMode)
                        {
                            delay = observation.getNrDelayBatches(true) - 1;
                        }
                        firstDM = observation.getFirstDM(true);
                    }
                    else
                    {
                        if (dataOptions.dataPSRDADA || dataOptions.streamingMode)
                        {
                            delay = observation.getNrDelayBatches() - 1;
                        }
                        firstDM = observation.getFirstDM();
                    }
                    if (dataOptions.dataPSRDADA || dataOptions.streamingMode)
                    {
                        outputTrigger << event.beam << " " << (batch - delay) << " " << event.sample << " " << integration;
                        outputTrigger << " " << event.compactedIntegration << " ";
                        outputTrigger << (((batch - delay) * observation.getNrSamplesPerBatch()) + (event.sample * integration)) * observation.getSamplingTime() << " " << firstDM + (event.DM * observation.getDMStep()) << " ";
                        outputTrigger << event.compactedDMs << " " << event.SNR << std::endl;
                    }
                    else
                    {
                        outputTrigger << event.beam << " " << batch << " " << event.sample << " " << integration << " ";
                        outputTrigger << event.compactedIntegration << " " << ((batch * observation.getNrSamplesPerBatch()) + (event.sample * integration)) * observation.getSamplingTime() << " ";
                        outputTrigger << firstDM + (event.DM * observation.getDMStep()) << " " << event.compactedDMs << " ";
                        outputTrigger << event.SNR << std::endl;
                    }
                }
            }
        }
        else
        {
            for (auto &triggeredEvent : triggeredEvents)
            {
                for (auto &dmEvents : triggeredEvent)
                {
                    for (auto &event : dmEvents.second)
                    {
                        unsigned int integration = 0;
                        float firstDM;
                        unsigned int delay = 0;

                        if (event.integration == 0)
                        {
                            integration = 1;
                        }
                        else
                        {
                            integration = event.integration;
                        }
                        if (options.subbandDedispersion)
                        {
                            if (dataOptions.dataPSRDADA || dataOptions.streamingMode)
                            {
                                delay = observation.getNrDelayBatches(true) - 1;
                            }
                            firstDM = observation.getFirstDM(true);
                        }
                        else
                        {
                            if (dataOptions.dataPSRDADA || dataOptions.streamingMode)
                            {
                                delay = observation.getNrDelayBatches() - 1;
                            }
                            firstDM = observation.getFirstDM();
                        }
                        if (dataOptions.dataPSRDADA || dataOptions.streamingMode)
                        {
                            outputTrigger << event.beam << " " << (batch - delay) << " " << event.sample << " " << integration;
                            outputTrigger << " " << (((batch - delay) * observation.getNrSamplesPerBatch()) + (event.sample * integration)) * observation.getSamplingTime() << " ";
                            outputTrigger << firstDM + (event.DM * observation.getDMStep()) << " " << event.SNR << std::endl;
                        }
                        else
                        {
                            outputTrigger << event.beam << " " << batch << " " << event.sample << " " << integration << " ";
                            outputTrigger << ((batch * observation.getNrSamplesPerBatch()) + (event.sample * integration)) * observation.getSamplingTime() << " " << firstDM + (event.DM * observation.getDMStep()) << " ";
                            outputTrigger << event.SNR << std::endl;
                        }
                    }
                }
            }
        }
        timers.trigger.stop();
    }
#ifdef HAVE_PSRDADA
    if (dataOptions.dataPSRDADA)
    {
        if (dada_hdu_unlock_read(hostMemory.ringBuffer) != 0)
        {
            std::cerr << "Impossible to unlock the PSRDADA ringbuffer for reading the header." << std::endl;
        }
        dada_hdu_disconnect(hostMemory.ringBuffer);
    }
#endif // HAVE_PSRDADA
    outputTrigger.close();
    timers.search.stop();
    if (options.dataDump)
    {
        if (options.subbandDedispersion)
        {
            hostMemoryDumpFiles.subbandedData.close();
        }
        hostMemoryDumpFiles.dedispersedData.close();
        hostMemoryDumpFiles.integratedData.close();
        if (options.snrMode == SNRMode::Standard)
        {
            hostMemoryDumpFiles.snrData.close();
            hostMemoryDumpFiles.snrSamplesData.close();
        }
        else if (options.snrMode == SNRMode::Momad)
        {
            hostMemoryDumpFiles.maxValuesData.close();
            hostMemoryDumpFiles.maxIndicesData.close();
            hostMemoryDumpFiles.medianOfMediansStepOneData.close();
            hostMemoryDumpFiles.medianOfMediansData.close();
            hostMemoryDumpFiles.medianOfMediansAbsoluteDeviationData.close();
        }
    }
}

int inputHandling(const unsigned int batch, const AstroData::Observation &observation, const Options &options, const DeviceOptions &deviceOptions, const DataOptions &dataOptions, Timers &timers, HostMemory &hostMemory, const DeviceMemory &deviceMemory)
{
    // Load the input
    timers.inputHandling.start();
    if (!dataOptions.dataPSRDADA && !dataOptions.streamingMode)
    {
        // If there are not enough available batches, computation is complete
        if (options.subbandDedispersion)
        {
            if (batch == observation.getNrBatches() - observation.getNrDelayBatches(true))
            {
                return -1;
            }
        }
        else
        {
            if (batch == observation.getNrBatches() - observation.getNrDelayBatches())
            {
                return -1;
            }
        }
        // If there are enough batches, prepare them for transfer to device
        for (unsigned int beam = 0; beam < observation.getNrBeams(); beam++)
        {
            if (options.subbandDedispersion)
            {
                if (options.splitBatchesDedispersion)
                {
                    // TODO: implement or remove splitBatches mode
                }
                else
                {
                    for (unsigned int channel = 0; channel < observation.getNrChannels(); channel++)
                    {
                        for (unsigned int chunk = 0; chunk < observation.getNrDelayBatches(true) - 1; chunk++)
                        {
                            if (inputBits >= 8)
                            {
                                memcpy(
                                    reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (chunk * observation.getNrSamplesPerBatch())])),
                                    reinterpret_cast<void *>(&((hostMemory.input.at(beam)->at(batch + chunk))->at(channel * observation.getNrSamplesPerBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))))),
                                    observation.getNrSamplesPerBatch() * sizeof(inputDataType));
                            }
                            else
                            {
                                memcpy(
                                    reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (chunk * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])),
                                    reinterpret_cast<void *>(&((hostMemory.input.at(beam)->at(batch + chunk))->at(channel * isa::utils::pad(observation.getNrSamplesPerBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))))),
                                    (observation.getNrSamplesPerBatch() / (8 / inputBits)) * sizeof(inputDataType));
                            }
                        }
                        if (inputBits >= 8)
                        {
                            memcpy(
                                reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + ((observation.getNrDelayBatches(true) - 1) * observation.getNrSamplesPerBatch())])),
                                reinterpret_cast<void *>(&((hostMemory.input.at(beam)->at(batch + (observation.getNrDelayBatches(true) - 1)))->at(channel * observation.getNrSamplesPerBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))))),
                                (observation.getNrSamplesPerDispersedBatch(true) % observation.getNrSamplesPerBatch()) * sizeof(inputDataType));
                        }
                        else
                        {
                            memcpy(
                                reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + ((observation.getNrDelayBatches(true) - 1) * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])),
                                reinterpret_cast<void *>(&((hostMemory.input.at(beam)->at(batch + (observation.getNrDelayBatches(true) - 1)))->at(channel * isa::utils::pad(observation.getNrSamplesPerBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))))),
                                ((observation.getNrSamplesPerDispersedBatch(true) % observation.getNrSamplesPerBatch()) / (8 / inputBits)) * sizeof(inputDataType));
                        }
                    }
                }
            }
            else
            {
                if (options.splitBatchesDedispersion)
                {
                    // TODO: implement or remove splitBatches mode
                }
                else
                {
                    for (unsigned int channel = 0; channel < observation.getNrChannels(); channel++)
                    {
                        for (unsigned int chunk = 0; chunk < observation.getNrDelayBatches() - 1; chunk++)
                        {
                            if (inputBits >= 8)
                            {
                                memcpy(
                                    reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (chunk * observation.getNrSamplesPerBatch())])),
                                    reinterpret_cast<void *>(&((hostMemory.input.at(beam)->at(batch + chunk))->at(channel * observation.getNrSamplesPerBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))))),
                                    observation.getNrSamplesPerBatch() * sizeof(inputDataType));
                            }
                            else
                            {
                                memcpy(
                                    reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (chunk * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])),
                                    reinterpret_cast<void *>(&((hostMemory.input.at(beam)->at(batch + chunk))->at(channel * isa::utils::pad(observation.getNrSamplesPerBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))))),
                                    (observation.getNrSamplesPerBatch() / (8 / inputBits)) * sizeof(inputDataType));
                            }
                        }
                        if (inputBits >= 8)
                        {
                            memcpy(
                                reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + ((observation.getNrDelayBatches() - 1) * observation.getNrSamplesPerBatch())])),
                                reinterpret_cast<void *>(&((hostMemory.input.at(beam)->at(batch + (observation.getNrDelayBatches() - 1)))->at(channel * observation.getNrSamplesPerBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))))),
                                (observation.getNrSamplesPerDispersedBatch() % observation.getNrSamplesPerBatch()) * sizeof(inputDataType));
                        }
                        else
                        {
                            memcpy(
                                reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + ((observation.getNrDelayBatches() - 1) * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])),
                                reinterpret_cast<void *>(&((hostMemory.input.at(beam)->at(batch + (observation.getNrDelayBatches() - 1)))->at(channel * isa::utils::pad(observation.getNrSamplesPerBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))))),
                                ((observation.getNrSamplesPerDispersedBatch() % observation.getNrSamplesPerBatch()) / (8 / inputBits)) * sizeof(inputDataType));
                        }
                    }
                }
            }
        }
    }
    else
    {
        if (dataOptions.dataPSRDADA)
        {
#ifdef HAVE_PSRDADA
            try
            {
                if (ipcbuf_eod(reinterpret_cast<ipcbuf_t *>(hostMemory.ringBuffer->data_block)))
                {
                    return -1;
                }
                if (options.subbandDedispersion)
                {
                    AstroData::readPSRDADA(*hostMemory.ringBuffer,
                                           hostMemory.inputStream.at(batch % observation.getNrDelayBatches(true)));
                }
                else
                {
                    AstroData::readPSRDADA(*hostMemory.ringBuffer,
                                           hostMemory.inputStream.at(batch % observation.getNrDelayBatches()));
                }
            }
            catch (AstroData::RingBufferError &err)
            {
                std::cerr << "Error: " << err.what() << std::endl;
                throw std::exception();
            }
#endif // HAVE_PSRDADA
        }
        else if (dataOptions.streamingMode)
        {
            if (options.subbandDedispersion)
            {
                AstroData::readSIGPROC(observation, deviceOptions.padding.at(deviceOptions.deviceName), inputBits, dataOptions.headerSizeSIGPROC, dataOptions.dataFile, hostMemory.inputStream.at(batch % observation.getNrDelayBatches(true)), batch);
            }
            else
            {
                AstroData::readSIGPROC(observation, deviceOptions.padding.at(deviceOptions.deviceName), inputBits, dataOptions.headerSizeSIGPROC, dataOptions.dataFile, hostMemory.inputStream.at(batch % observation.getNrDelayBatches()), batch);
            }
        }
        // If there are enough data buffered, proceed with the computation
        // Otherwise, move to the next iteration of the search loop
        if (options.subbandDedispersion)
        {
            if (batch < observation.getNrDelayBatches(true) - 1)
            {
                return 1;
            }
        }
        else
        {
            if (batch < observation.getNrDelayBatches() - 1)
            {
                return 1;
            }
        }
        for (unsigned int beam = 0; beam < observation.getNrBeams(); beam++)
        {
            if (options.subbandDedispersion)
            {
                if (options.splitBatchesDedispersion)
                {
                    // TODO: implement or remove splitBatches mode
                }
                else
                {
                    for (unsigned int channel = 0; channel < observation.getNrChannels(); channel++)
                    {
                        for (unsigned int chunk = batch - (observation.getNrDelayBatches(true) - 1); chunk < batch; chunk++)
                        {
                            // Full batches
                            if (inputBits >= 8)
                            {
                                memcpy(
                                    reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + ((chunk - (batch - (observation.getNrDelayBatches(true) - 1))) * observation.getNrSamplesPerBatch())])),
                                    reinterpret_cast<void *>(&(hostMemory.inputStream.at(chunk % observation.getNrDelayBatches(true))->at((beam * observation.getNrChannels() * observation.getNrSamplesPerBatch()) + (channel * observation.getNrSamplesPerBatch())))),
                                    observation.getNrSamplesPerBatch() * sizeof(inputDataType));
                            }
                            else
                            {
                                memcpy(
                                    reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + ((chunk - (batch - (observation.getNrDelayBatches(true) - 1))) * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])),
                                    reinterpret_cast<void *>(&(hostMemory.inputStream.at(chunk % observation.getNrDelayBatches(true))->at((beam * observation.getNrChannels() * (observation.getNrSamplesPerBatch() / (8 / inputBits))) + (channel * (observation.getNrSamplesPerBatch() / (8 / inputBits)))))),
                                    (observation.getNrSamplesPerBatch() / (8 / inputBits)) * sizeof(inputDataType));
                            }
                        }
                        // Remainder (part of current batch)
                        if (inputBits >= 8)
                        {
                            memcpy(
                                reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + ((observation.getNrDelayBatches(true) - 1) * observation.getNrSamplesPerBatch())])),
                                reinterpret_cast<void *>(&(hostMemory.inputStream.at(batch % observation.getNrDelayBatches(true))->at((beam * observation.getNrChannels() * observation.getNrSamplesPerBatch()) + (channel * observation.getNrSamplesPerBatch())))),
                                (observation.getNrSamplesPerDispersedBatch(true) % observation.getNrSamplesPerBatch()) * sizeof(inputDataType));
                        }
                        else
                        {
                            memcpy(
                                reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + ((observation.getNrDelayBatches(true) - 1) * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])),
                                reinterpret_cast<void *>(&(hostMemory.inputStream.at(batch % observation.getNrDelayBatches(true))->at((beam * observation.getNrChannels() * (observation.getNrSamplesPerBatch() / (8 / inputBits))) + (channel * (observation.getNrSamplesPerBatch() / (8 / inputBits)))))),
                                ((observation.getNrSamplesPerDispersedBatch(true) % observation.getNrSamplesPerBatch()) / (8 / inputBits)) * sizeof(inputDataType));
                        }
                    }
                }
            }
            else
            {
                if (options.splitBatchesDedispersion)
                {
                    // TODO: implement or remove splitBatches mode
                }
                else
                {
                    for (unsigned int channel = 0; channel < observation.getNrChannels(); channel++)
                    {
                        for (unsigned int chunk = batch - (observation.getNrDelayBatches() - 1); chunk < batch; chunk++)
                        {
                            if (inputBits >= 8)
                            {
                                memcpy(
                                    reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + ((chunk - (batch - (observation.getNrDelayBatches() - 1))) * observation.getNrSamplesPerBatch())])),
                                    reinterpret_cast<void *>(&(hostMemory.inputStream.at(chunk % observation.getNrDelayBatches())->at((beam * observation.getNrChannels() * observation.getNrSamplesPerBatch()) + (channel * observation.getNrSamplesPerBatch())))),
                                    observation.getNrSamplesPerBatch() * sizeof(inputDataType));
                            }
                            else
                            {
                                memcpy(
                                    reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + ((chunk - (batch - (observation.getNrDelayBatches() - 1))) * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])),
                                    reinterpret_cast<void *>(&(hostMemory.inputStream.at(chunk % observation.getNrDelayBatches())->at((beam * observation.getNrChannels() * (observation.getNrSamplesPerBatch() / (8 / inputBits))) + (channel * (observation.getNrSamplesPerBatch() / (8 / inputBits)))))), (observation.getNrSamplesPerBatch() / (8 / inputBits)) * sizeof(inputDataType));
                            }
                        }
                        if (inputBits >= 8)
                        {
                            memcpy(
                                reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + ((observation.getNrDelayBatches() - 1) * observation.getNrSamplesPerBatch())])),
                                reinterpret_cast<void *>(&(hostMemory.inputStream.at(batch % observation.getNrDelayBatches())->at((beam * observation.getNrChannels() * observation.getNrSamplesPerBatch()) + (channel * observation.getNrSamplesPerBatch())))),
                                (observation.getNrSamplesPerDispersedBatch() % observation.getNrSamplesPerBatch()) * sizeof(inputDataType));
                        }
                        else
                        {
                            memcpy(
                                reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + ((observation.getNrDelayBatches() - 1) * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])),
                                reinterpret_cast<void *>(&(hostMemory.inputStream.at(batch % observation.getNrDelayBatches())->at((beam * observation.getNrChannels() * (observation.getNrSamplesPerBatch() / (8 / inputBits))) + (channel * (observation.getNrSamplesPerBatch() / (8 / inputBits)))))),
                                ((observation.getNrSamplesPerDispersedBatch() % observation.getNrSamplesPerBatch()) / (8 / inputBits)) * sizeof(inputDataType));
                        }
                    }
                }
            }
        }
    }
    timers.inputHandling.stop();
    return 0;
}

int copyInputToDevice(const unsigned int batch, const OpenCLRunTime &openclRunTime, const AstroData::Observation &observation, const Options &options, const DeviceOptions &deviceOptions, Timers &timers, HostMemory &hostMemory, const DeviceMemory &deviceMemory)
{
    cl::Event syncPoint;

    // Copy input from host to device
    try
    {
        if (deviceOptions.synchronized)
        {
            timers.inputCopy.start();
            if (options.splitBatchesDedispersion)
            {
                // TODO: implement or remove splitBatches mode
            }
            else
            {
                openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueWriteBuffer(deviceMemory.dispersedData, CL_TRUE, 0, hostMemory.dispersedData.size() * sizeof(inputDataType), reinterpret_cast<void *>(hostMemory.dispersedData.data()), nullptr, &syncPoint);
            }
            syncPoint.wait();
            timers.inputCopy.stop();
        }
        else
        {
            if (options.splitBatchesDedispersion)
            {
                // TODO: implement or remove splitBatches mode
            }
            else
            {
                openclRunTime.queues->at(deviceOptions.deviceID).at(0).enqueueWriteBuffer(deviceMemory.dispersedData, CL_FALSE, 0, hostMemory.dispersedData.size() * sizeof(inputDataType), reinterpret_cast<void *>(hostMemory.dispersedData.data()));
            }
        }
    }
    catch (cl::Error &err)
    {
        std::cerr << "Input copy error -- Batch: " << std::to_string(batch) << ", " << err.what() << " " << err.err();
        std::cerr << std::endl;
        throw std::exception();
    }
    return 0;
}
