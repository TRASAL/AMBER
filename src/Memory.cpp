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

void loadInput(AstroData::Observation & observation, const DeviceOptions & deviceOptions, const DataOptions & dataOptions, HostMemory & hostMemory, Timers & timers) {
  hostMemory.zappedChannels.resize(observation.getNrChannels(deviceOptions.padding[deviceOptions.deviceName] / sizeof(unsigned int)));
  try {
    AstroData::readZappedChannels(observation, dataOptions.channelsFile, hostMemory.zappedChannels);
    AstroData::readIntegrationSteps(observation, dataOptions.integrationFile, hostMemory.integrationSteps);
  } catch ( AstroData::FileError & err ) {
    std::cerr << err.what() << std::endl;
    throw;
  }
  hostMemory.input.resize(observation.getNrBeams());
  if ( dataOptions.dataLOFAR ) {
#ifdef HAVE_HDF5
    hostMemory.input.at(0) = new std::vector<std::vector<inputDataType> *>(observation.getNrBatches());
    timers.inputLoad.start();
    if ( dataOptions.limit ) {
      AstroData::readLOFAR(dataOptions.headerFile, dataOptions.dataFile, observation, deviceOptions.padding[deviceOptions.deviceName], *(hostMemory.input.at(0)), observation.getNrBatches());
    } else {
      AstroData::readLOFAR(dataOptions.headerFile, dataOptions.dataFile, observation, deviceOptions.padding[deviceOptions.deviceName], *(hostMemory.input.at(0)));
    }
    timers.inputLoad.stop();
#endif // HAVE_HDF5
  } else if ( dataOptions.dataSIGPROC ) {
    hostMemory.input.at(0) = new std::vector<std::vector<inputDataType> *>(observation.getNrBatches());
    timers.inputLoad.start();
    AstroData::readSIGPROC(observation, deviceOptions.padding[deviceOptions.deviceName], inputBits, dataOptions.headerSizeSIGPROC, dataOptions.dataFile, *(hostMemory.input.at(0)));
    timers.inputLoad.stop();
  } else if ( dataOptions.dataPSRDADA ) {
#ifdef HAVE_PSRDADA
    hostMemory.ringBuffer = dada_hdu_create(0);
    dada_hdu_set_key(hostMemory.ringBuffer, dataOptions.dadaKey);
    if ( dada_hdu_connect(hostMemory.ringBuffer) != 0 ) {
      throw AstroData::RingBufferError("ERROR: impossible to connect to PSRDADA ringbuffer \"" + std::to_string(dataOptions.dadaKey) + "\".");
    }
    if ( dada_hdu_lock_read(hostMemory.ringBuffer) != 0 ) {
      throw AstroData::RingBufferError("ERROR: impossible to lock the PSRDADA ringbuffer for reading the header.");
    }
    timers.inputLoad.start();
    AstroData::readPSRDADAHeader(observation, *hostMemory.ringBuffer);
    timers.inputLoad.stop();
#endif // HAVE_PSRDADA
  } else {
    for ( unsigned int beam = 0; beam < observation.getNrBeams(); beam++ ) {
      // TODO: if there are multiple synthesized beams, the generated data should take this into account
      hostMemory.input.at(beam) = new std::vector<std::vector<inputDataType> *>(observation.getNrBatches());
      AstroData::generateSinglePulse(generatorOptions.width, generatorOptions.DM, observation, deviceOptions.padding[deviceOptions.deviceName], *(hostMemory.input.at(beam)), inputBits, generatorOptions.random);
    }
  }
}

void allocateHostMemory(AstroData::Observation & observation, const Options & options, const DeviceOptions & deviceOptions, HostMemory & hostMemory) {
  if ( !options.subbandDedispersion ) {
    hostMemory.shiftsSingleStep = Dedispersion::getShifts(observation, deviceOptions.padding[deviceOptions.deviceName]);
    if ( options.debug ) {
      std::cerr << "shiftsSingleStep" << std::endl;
      for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ ) {
        std::cerr << hostMemory.shiftsSingleStep->at(channel) << " ";
      }
      std::cerr << std::endl << std::endl;
    }
    observation.setNrSamplesPerDispersedBatch(observation.getNrSamplesPerBatch() + static_cast<unsigned int>(hostMemory.shiftsSingleStep->at(0) * (observation.getFirstDM() + ((observation.getNrDMs() - 1) * observation.getDMStep()))));
    observation.setNrDelayBatches(static_cast<unsigned int>(std::ceil(static_cast<double>(observation.getNrSamplesPerDispersedBatch()) / observation.getNrSamplesPerBatch())));
#ifdef HAVE_PSRDADA
    if ( dataOptions.dataPSRDADA ) {
      hostMemory.inputDADA.resize(observation.getNrDelayBatches());
      for ( unsigned int batch = 0; batch < observation.getNrDelayBatches(); batch++ ) {
        if ( inputBits >= 8 ) {
          hostMemory.inputDADA.at(batch) = new std::vector<inputDataType>(observation.getNrBeams() * observation.getNrChannels() * observation.getNrSamplesPerBatch());
        } else {
          hostMemory.inputDADA.at(batch) = new std::vector<inputDataType>(observation.getNrBeams() * observation.getNrChannels() * (observation.getNrSamplesPerBatch() / (8 / inputBits)));
        }
      }
    }
#endif // HAVE_PSRDADA
    if ( configurations.dedispersionParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getSplitBatches() ) {
      // TODO: add support for splitBatches
    } else {
      if ( inputBits >= 8 ) {
        hostMemory.dispersedData.resize(observation.getNrBeams() * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType)));
      } else {
        hostMemory.dispersedData.resize(observation.getNrBeams() * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType)));
      }
    }
    hostMemory.beamMapping.resize(observation.getNrSynthesizedBeams() * observation.getNrChannels(deviceOptions.padding[deviceOptions.deviceName] / sizeof(unsigned int)));
    hostMemory.dedispersedData.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType)));
    hostMemory.integratedData.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / *(integrationSteps.begin()), deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType)));
    hostMemory.snrData.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(float)));
    hostMemory.snrSamples.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(unsigned int)));
  } else {
    hostMemory.shiftsStepOne = Dedispersion::getShifts(observation, deviceOptions.padding[deviceOptions.deviceName]);
    hostMemory.shiftsStepTwo = Dedispersion::getShiftsStepTwo(observation, deviceOptions.padding[deviceOptions.deviceName]);
    if ( options.debug ) {
      std::cerr << "shiftsStepOne" << std::endl;
      for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ ) {
        std::cerr << hostMemory.shiftsStepOne->at(channel) << " ";
      }
      std::cerr << std::endl << "shiftsStepTwo" << std::endl;
      for ( unsigned int subband = 0; subband < observation.getNrSubbands(); subband++ ) {
        std::cerr << hostMemory.shiftsStepTwo->at(subband) << " ";
      }
      std::cerr << std::endl << std::endl;
    }
    observation.setNrSamplesPerBatch(observation.getNrSamplesPerBatch() + static_cast<unsigned int>(hostMemory.shiftsStepTwo->at(0) * (observation.getFirstDM() + ((observation.getNrDMs() - 1) * observation.getDMStep()))), true);
    observation.setNrSamplesPerDispersedBatch(observation.getNrSamplesPerBatch(true) + static_cast<unsigned int>(hostMemory.shiftsStepOne->at(0) * (observation.getFirstDM(true) + ((observation.getNrDMs(true) - 1) * observation.getDMStep(true)))), true);
    observation.setNrDelayBatches(static_cast<unsigned int>(std::ceil(static_cast<double>(observation.getNrSamplesPerDispersedBatch(true)) / observation.getNrSamplesPerBatch())), true);
#ifdef HAVE_PSRDADA
    if ( dataOptions.dataPSRDADA ) {
      hostMemory.inputDADA.resize(observation.getNrDelayBatches(true));
      for ( unsigned int batch = 0; batch < observation.getNrDelayBatches(true); batch++ ) {
        if ( inputBits >= 8 ) {
          hostMemory.inputDADA.at(batch) = new std::vector<inputDataType>(observation.getNrBeams() * observation.getNrChannels() * observation.getNrSamplesPerBatch());
        } else {
          hostMemory.inputDADA.at(batch) = new std::vector<inputDataType>(observation.getNrBeams() * observation.getNrChannels() * (observation.getNrSamplesPerBatch() / (8 / inputBits)));
        }
      }
    }
#endif // HAVE_PSRDADA
    if ( configurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getSplitBatches() ) {
      // TODO: add support for splitBatches
    } else {
      if ( inputBits >= 8 ) {
        hostMemory.dispersedData.resize(observation.getNrBeams() * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType)));
      } else {
        hostMemory.dispersedData.resize(observation.getNrBeams() * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding[deviceOptions.deviceName] / sizeof(inputDataType)));
      }
    }
    hostMemory.beamMapping.resize(observation.getNrSynthesizedBeams() * observation.getNrSubbands(deviceOptions.padding[deviceOptions.deviceName] / sizeof(unsigned int)));
    hostMemory.subbandedData.resize(observation.getNrBeams() * observation.getNrDMs(true) * observation.getNrSubbands() * observation.getNrSamplesPerBatch(true, deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType)));
    hostMemory.dedispersedData.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType)));
    hostMemory.integratedData.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / *(integrationSteps.begin()), deviceOptions.padding[deviceOptions.deviceName] / sizeof(outputDataType)));
   hostMemory.snrData.resize(observation.getNrSynthesizedBeams() * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding[deviceOptions.deviceName] / sizeof(float)));
    hostMemory.snrSamples.resize(observation.getNrSynthesizedBeams() * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), deviceOptions.padding[deviceOptions.deviceName] / sizeof(unsigned int)));
  }
  AstroData::generateBeamMapping(observation, hostMemory.beamMapping, deviceOptions.padding[deviceOptions.deviceName], options.subbandDedispersion);
}

void allocateDeviceMemory(const cl::Context * clContext, const std::vector<std::vector<cl::CommandQueue>> * clQueues, const Options & options, const DeviceOptions & deviceOptions, const HostMemory & hostMemory, DeviceMemory & deviceMemory) {
  if ( !options.subbandDedispersion ) {
    deviceMemory.shiftsSingleStep = cl::Buffer(*clContext, CL_MEM_READ_ONLY, hostMemory.shiftsSingleStep->size() * sizeof(float), 0, 0);
    clQueues->at(deviceOptions.deviceID)[0].enqueueWriteBuffer(deviceMemory.shiftsSingleStep, CL_FALSE, 0, hostMemory.shiftsSingleStep->size() * sizeof(float), reinterpret_cast<void *>(hostMemory.shiftsSingleStep->data()));
  } else {
    deviceMemory.shiftsStepOne = cl::Buffer(*clContext, CL_MEM_READ_ONLY, hostMemory.shiftsStepOne->size() * sizeof(float), 0, 0);
    clQueues->at(deviceOptions.deviceID)[0].enqueueWriteBuffer(deviceMemory.shiftsStepOne, CL_FALSE, 0, hostMemory.shiftsStepOne->size() * sizeof(float), reinterpret_cast<void *>(hostMemory.shiftsStepOne->data()));
    deviceMemory.shiftsStepTwo = cl::Buffer(*clContext, CL_MEM_READ_ONLY, hostMemory.shiftsStepTwo->size() * sizeof(float), 0, 0);
    clQueues->at(deviceOptions.deviceID)[0].enqueueWriteBuffer(deviceMemory.shiftsStepTwo, CL_FALSE, 0, hostMemory.shiftsStepTwo->size() * sizeof(float), reinterpret_cast<void *>(hostMemory.shiftsStepTwo->data()));
    deviceMemory.subbandedData = cl::Buffer(*clContext, CL_MEM_READ_WRITE, hostMemory.subbandedData.size() * sizeof(outputDataType), 0, 0);
  }
  deviceMemory.zappedChannels = cl::Buffer(*clContext, CL_MEM_READ_ONLY, hostMemory.zappedChannels.size() * sizeof(unsigned int), 0, 0);
  clQueues->at(deviceOptions.deviceID)[0].enqueueWriteBuffer(deviceMemory.zappedChannels, CL_FALSE, 0, hostMemory.zappedChannels.size() * sizeof(unsigned int), reinterpret_cast< void * >(hostMemory.zappedChannels.data()));
  deviceMemory.beamMapping = cl::Buffer(*clContext, CL_MEM_READ_ONLY, hostMemory.beamMapping.size() * sizeof(unsigned int), 0, 0);
  clQueues->at(deviceOptions.deviceID)[0].enqueueWriteBuffer(deviceMemory.beamMapping, CL_FALSE, 0, hostMemory.beamMapping.size() * sizeof(unsigned int), reinterpret_cast<void *>(hostMemory.beamMapping.data()));
  deviceMemory.dispersedData = cl::Buffer(*clContext, CL_MEM_READ_ONLY, hostMemory.dispersedData.size() * sizeof(inputDataType), 0, 0);
  deviceMemory.dedispersedData = cl::Buffer(*clContext, CL_MEM_READ_WRITE, hostMemory.dedispersedData.size() * sizeof(outputDataType), 0, 0);
  deviceMemory.integratedData = cl::Buffer(*clContext, CL_MEM_READ_WRITE, hostMemory.integratedData.size() * sizeof(outputDataType), 0, 0);
  deviceMemory.snrData = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, hostMemory.snrData.size() * sizeof(float), 0, 0);
  deviceMemory.snrSamples = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, hostMemory.snrSamples.size() * sizeof(unsigned int), 0, 0);
  clQueues->at(deviceOptions.deviceID)[0].finish();
}
