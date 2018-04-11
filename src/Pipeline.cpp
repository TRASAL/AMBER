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

void pipeline(const OpenCLRunTime & openclRunTime, const AstroData::Observation & observation, const Options & options,
              const DeviceOptions & deviceOptions, const DataOptions & dataOptions, Timers & timers,
              const Kernels & kernels, const KernelConfigurations & kernelConfigurations,
              const KernelRunTimeConfigurations & kernelRunTimeConfigurations, HostMemory & hostMemory,
              DeviceMemory & deviceMemory) {
  bool errorDetected = false;
  int status = 0;
  std::ofstream outputTrigger;
  cl::Event syncPoint;

  timers.search.start();
  outputTrigger.open(dataOptions.outputFile + ".trigger");
  if ( !outputTrigger ) {
    std::cerr << "Impossible to open " + dataOptions.outputFile + "." << std::endl;
    throw std::exception();
  }
  if ( options.compactResults ) {
    outputTrigger << "# beam batch sample integration_step compacted_integration_steps time DM compacted_DMs SNR";
    outputTrigger << std::endl;
  } else {
    outputTrigger << "# beam batch sample integration_step time DM SNR" << std::endl;
  }
  for ( unsigned int batch = 0; batch < observation.getNrBatches(); batch++ ) {
    TriggeredEvents triggeredEvents(observation.getNrSynthesizedBeams());
    CompactedEvents compactedEvents(observation.getNrSynthesizedBeams());

    status = inputHandling(batch, observation, options, deviceOptions, dataOptions, timers, hostMemory, deviceMemory);
    if ( status == 1 ) {
      // Not enough data for this iteration, move to next.
      continue;
    } else if ( status == -1 ) {
      // Not enough batches remaining, exit the main loop.
      break;
    }
    status = copyInputToDevice(batch, openclRunTime, observation, options, deviceOptions, timers, hostMemory, deviceMemory);
    if ( status != 0 ) {
      break;
    }
    if ( options.splitBatchesDedispersion && (batch < observation.getNrDelayBatches()) ) {
      // Not enough batches in the buffer to start the search
      continue;
    }

    // Dedispersion
    if ( options.subbandDedispersion ) {
      if ( options.splitBatchesDedispersion ) {
        // TODO: implement or remove splitBatches mode
      }
      if ( deviceOptions.synchronized ) {
        try {
          timers.dedispersionStepOne.start();
          openclRunTime.queues->at(deviceOptions.deviceID).at(0)
            .enqueueNDRangeKernel(*(kernels.dedispersionStepOne), cl::NullRange,
                                  kernelRunTimeConfigurations.dedispersionStepOneGlobal,
                                  kernelRunTimeConfigurations.dedispersionStepOneLocal, nullptr, &syncPoint);
          syncPoint.wait();
          timers.dedispersionStepOne.stop();
        } catch ( cl::Error & err ) {
          std::cerr << "Dedispersion Step One error -- Batch: " << std::to_string(batch) << ", " << err.what() << " ";
          std::cerr << err.err() << std::endl;
          errorDetected = true;
        }
        try {
          timers.dedispersionStepTwo.start();
          openclRunTime.queues->at(deviceOptions.deviceID).at(0)
            .enqueueNDRangeKernel(*(kernels.dedispersionStepTwo), cl::NullRange,
                                  kernelRunTimeConfigurations.dedispersionStepTwoGlobal,
                                  kernelRunTimeConfigurations.dedispersionStepTwoLocal, nullptr, &syncPoint);
          syncPoint.wait();
          timers.dedispersionStepTwo.stop();
        } catch ( cl::Error & err ) {
          std::cerr << "Dedispersion Step Two error -- Batch: " << std::to_string(batch) << ", " << err.what() << " ";
          std::cerr << err.err() << std::endl;
          errorDetected = true;
        }
      } else {
        try {
          openclRunTime.queues->at(deviceOptions.deviceID).at(0)
            .enqueueNDRangeKernel(*(kernels.dedispersionStepOne), cl::NullRange,
                                  kernelRunTimeConfigurations.dedispersionStepOneGlobal,
                                  kernelRunTimeConfigurations.dedispersionStepOneLocal, nullptr, nullptr);
        } catch ( cl::Error & err ) {
          std::cerr << "Dedispersion Step One error -- Batch: " << std::to_string(batch) << ", " << err.what() << " ";
          std::cerr << err.err() << std::endl;
          errorDetected = true;
        }
        try {
          openclRunTime.queues->at(deviceOptions.deviceID).at(0)
            .enqueueNDRangeKernel(*(kernels.dedispersionStepTwo), cl::NullRange,
                                  kernelRunTimeConfigurations.dedispersionStepTwoGlobal,
                                  kernelRunTimeConfigurations.dedispersionStepTwoLocal, nullptr, nullptr);
        } catch ( cl::Error & err ) {
          std::cerr << "Dedispersion Step Two error -- Batch: " << std::to_string(batch) << ", " << err.what() << " ";
          std::cerr << err.err() << std::endl;
          errorDetected = true;
        }
      }
    } else {
      try {
        if ( options.splitBatchesDedispersion ) {
          // TODO: implement or remove splitBatches mode
        }
        if ( deviceOptions.synchronized ) {
          timers.dedispersionSingleStep.start();
          openclRunTime.queues->at(deviceOptions.deviceID).at(0)
            .enqueueNDRangeKernel(*(kernels.dedispersionSingleStep), cl::NullRange,
                                  kernelRunTimeConfigurations.dedispersionSingleStepGlobal,
                                  kernelRunTimeConfigurations.dedispersionSingleStepLocal, nullptr, &syncPoint);
          syncPoint.wait();
          timers.dedispersionSingleStep.stop();
        } else {
          openclRunTime.queues->at(deviceOptions.deviceID).at(0)
            .enqueueNDRangeKernel(*(kernels.dedispersionSingleStep), cl::NullRange,
                                  kernelRunTimeConfigurations.dedispersionSingleStepGlobal,
                                  kernelRunTimeConfigurations.dedispersionSingleStepLocal);
        }
      } catch ( cl::Error & err ) {
        std::cerr << "Dedispersion error -- Batch: " << std::to_string(batch) << ", " << err.what() << " " << err.err();
        std::cerr << std::endl;
        errorDetected = true;
      }
    }
    if ( options.debug ) {
      if ( options.subbandDedispersion ) {
        try {
          openclRunTime.queues->at(deviceOptions.deviceID).at(0)
            .enqueueReadBuffer(deviceMemory.subbandedData, CL_TRUE, 0,
                               hostMemory.subbandedData.size() * sizeof(outputDataType),
                               reinterpret_cast<void *>(hostMemory.subbandedData.data()), nullptr, &syncPoint);
          syncPoint.wait();
          openclRunTime.queues->at(deviceOptions.deviceID).at(0)
            .enqueueReadBuffer(deviceMemory.dedispersedData, CL_TRUE, 0,
                               hostMemory.dedispersedData.size() * sizeof(outputDataType),
                               reinterpret_cast<void *>(hostMemory.dedispersedData.data()), nullptr, &syncPoint);
          syncPoint.wait();
          std::cerr << "subbandedData" << std::endl;
          for ( unsigned int beam = 0; beam < observation.getNrBeams(); beam++ ) {
            std::cerr << "Beam: " << beam << std::endl;
            for ( unsigned int dm = 0; dm < observation.getNrDMs(true); dm++ ) {
              std::cerr << "Subbanding DM: " << dm << std::endl;
              for ( unsigned int subband = 0; subband < observation.getNrSubbands(); subband++ ) {
                for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(true); sample++ ) {
                  std::cerr << hostMemory.subbandedData
                    .at((beam * observation.getNrDMs(true) * observation.getNrSubbands()
                         * observation.getNrSamplesPerBatch(true, deviceOptions.padding.at(deviceOptions.deviceName)
                                                                  / sizeof(outputDataType))) + (dm * observation.getNrSubbands()
                                                                                                * observation.getNrSamplesPerBatch(true, deviceOptions.padding.at(deviceOptions.deviceName)
                                                                                                                                         / sizeof(outputDataType))) +
                        (subband * observation.getNrSamplesPerBatch(true,
                                                                    deviceOptions.padding.at(deviceOptions.deviceName)
                                                                    / sizeof(outputDataType))) + sample) << " ";
                }
                std::cerr << std::endl;
              }
              std::cerr << std::endl;
            }
            std::cerr << std::endl;
          }
          std::cerr << "dedispersedData" << std::endl;
          for ( unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++ ) {
            std::cerr << "sBeam: " << sBeam << std::endl;
            for ( unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++ ) {
              for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
                std::cerr << "DM: " << (subbandingDM * observation.getNrDMs()) + dm << std::endl;
                for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++ ) {
                  std::cerr << hostMemory.dedispersedData
                    .at((sBeam * observation.getNrDMs(true) * observation.getNrDMs()
                         * observation.getNrSamplesPerBatch(false, deviceOptions.padding.at(deviceOptions.deviceName)
                                                                   / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs()
                                                                                                 * observation.getNrSamplesPerBatch(false, deviceOptions.padding.at(deviceOptions.deviceName)
                                                                                                                                           / sizeof(outputDataType))) +
                        (dm * observation.getNrSamplesPerBatch(false,
                                                               deviceOptions.padding.at(deviceOptions.deviceName)
                                                               / sizeof(outputDataType))) + sample) << " ";
                }
                std::cerr << std::endl;
              }
              std::cerr << std::endl;
            }
            std::cerr << std::endl;
          }
        } catch ( cl::Error & err) {
          std::cerr << "Impossible to read deviceMemory.subbandedData and deviceMemory.dedispersedData: ";
          std::cerr << err.what() << " " << err.err() << std::endl;
          errorDetected = true;
        }
      } else {
        try {
          openclRunTime.queues->at(deviceOptions.deviceID).at(0)
            .enqueueReadBuffer(deviceMemory.dedispersedData, CL_TRUE, 0,
                               hostMemory.dedispersedData.size() * sizeof(outputDataType),
                               reinterpret_cast<void *>(hostMemory.dedispersedData.data()), nullptr, &syncPoint);
          syncPoint.wait();
          std::cerr << "dedispersedData" << std::endl;
          for ( unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++ ) {
            std::cerr << "sBeam: " << sBeam << std::endl;
            for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
              std::cerr << "DM: " << dm << std::endl;
              for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++ ) {
                std::cerr << hostMemory.dedispersedData
                  .at((sBeam * observation.getNrDMs()
                       * observation.getNrSamplesPerBatch(false, deviceOptions.padding.at(deviceOptions.deviceName)
                                                                 / sizeof(outputDataType)))
                      + (dm * observation.getNrSamplesPerBatch(false,
                                                               deviceOptions.padding.at(deviceOptions.deviceName)
                                                               / sizeof(outputDataType))) + sample) << " ";
              }
              std::cerr << std::endl;
            }
            std::cerr << std::endl;
          }
        } catch ( cl::Error & err ) {
          std::cerr << "Impossible to read deviceMemory.dedispersedData: " << err.what() << " " << err.err();
          std::cerr << std::endl;
          errorDetected = true;
        }
      }
    }

    // SNR of dedispersed data
    try {
      if ( deviceOptions.synchronized ) {
        timers.snr.start();
        openclRunTime.queues->at(deviceOptions.deviceID).at(0)
          .enqueueNDRangeKernel(*kernels.snr[hostMemory.integrationSteps.size()], cl::NullRange,
                                kernelRunTimeConfigurations.snrGlobal[hostMemory.integrationSteps.size()],
                                kernelRunTimeConfigurations.snrLocal[hostMemory.integrationSteps.size()], nullptr,
                                &syncPoint);
        syncPoint.wait();
        timers.snr.stop();
        timers.outputCopy.start();
        openclRunTime.queues->at(deviceOptions.deviceID).at(0)
          .enqueueReadBuffer(deviceMemory.snrData, CL_TRUE, 0, hostMemory.snrData.size() * sizeof(float),
                             reinterpret_cast<void *>(hostMemory.snrData.data()), nullptr, &syncPoint);
        syncPoint.wait();
        openclRunTime.queues->at(deviceOptions.deviceID).at(0)
          .enqueueReadBuffer(deviceMemory.snrSamples, CL_TRUE, 0, hostMemory.snrSamples.size() * sizeof(unsigned int),
                             reinterpret_cast<void *>(hostMemory.snrSamples.data()), nullptr, &syncPoint);
        syncPoint.wait();
        timers.outputCopy.stop();
      } else {
        openclRunTime.queues->at(deviceOptions.deviceID).at(0)
          .enqueueNDRangeKernel(*kernels.snr[hostMemory.integrationSteps.size()], cl::NullRange,
                                kernelRunTimeConfigurations.snrGlobal[hostMemory.integrationSteps.size()],
                                kernelRunTimeConfigurations.snrLocal[hostMemory.integrationSteps.size()]);
        openclRunTime.queues->at(deviceOptions.deviceID).at(0)
          .enqueueReadBuffer(deviceMemory.snrData, CL_FALSE, 0, hostMemory.snrData.size() * sizeof(float),
                             reinterpret_cast<void *>(hostMemory.snrData.data()));
        openclRunTime.queues->at(deviceOptions.deviceID).at(0)
          .enqueueReadBuffer(deviceMemory.snrSamples, CL_FALSE, 0,
                             hostMemory.snrSamples.size() * sizeof(unsigned int),
                             reinterpret_cast<void *>(hostMemory.snrSamples.data()));
        openclRunTime.queues->at(deviceOptions.deviceID).at(0).finish();
      }
    } catch ( cl::Error & err ) {
      std::cerr << "SNR dedispersed data error -- Batch: " << std::to_string(batch) << ", " << err.what() << " ";
      std::cerr << err.err() << std::endl;
      errorDetected = true;
    }
    timers.trigger.start();
    trigger(options.subbandDedispersion, deviceOptions.padding.at(deviceOptions.deviceName), 0, options.threshold,
            observation, hostMemory.snrData, hostMemory.snrSamples, triggeredEvents);
    timers.trigger.stop();
    if ( options.debug ) {
      if ( options.subbandDedispersion ) {
        std::cerr << "hostMemory.snrData" << std::endl;
        for ( unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++ ) {
          std::cerr << "sBeam: " << sBeam << std::endl;
          for ( unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++ ) {
            for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
              std::cerr << hostMemory.snrData
                .at((sBeam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(),
                                             deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(float)))
                    + (subbandingDM * observation.getNrDMs()) + dm) << " ";
            }
          }
          std::cerr << std::endl;
        }
      } else {
        std::cerr << "hostMemory.snrData" << std::endl;
        for ( unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++ ) {
          std::cerr << "sBeam: " << sBeam << std::endl;
          for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
            std::cerr << hostMemory.snrData
              .at((sBeam * observation.getNrDMs(false, deviceOptions.padding.at(deviceOptions.deviceName)
                                                       / sizeof(float))) + dm) << " ";
          }
          std::cerr << std::endl;
        }
      }
      std::cerr << std::endl;
    }

    // Integration and SNR loop
    for ( unsigned int stepNumber = 0; stepNumber < hostMemory.integrationSteps.size(); stepNumber++ ) {
      auto step = hostMemory.integrationSteps.begin();

      std::advance(step, stepNumber);
      try {
        if ( deviceOptions.synchronized ) {
          timers.integration.start();
          openclRunTime.queues->at(deviceOptions.deviceID).at(0)
            .enqueueNDRangeKernel(*kernels.integration[stepNumber], cl::NullRange,
                                  kernelRunTimeConfigurations.integrationGlobal[stepNumber],
                                  kernelRunTimeConfigurations.integrationLocal[stepNumber], nullptr, &syncPoint);
          syncPoint.wait();
          timers.integration.stop();
          timers.snr.start();
          openclRunTime.queues->at(deviceOptions.deviceID).at(0)
            .enqueueNDRangeKernel(*kernels.snr[stepNumber], cl::NullRange,
                                  kernelRunTimeConfigurations.snrGlobal[stepNumber],
                                  kernelRunTimeConfigurations.snrLocal[stepNumber], nullptr, &syncPoint);
          syncPoint.wait();
          timers.snr.stop();
          timers.outputCopy.start();
          openclRunTime.queues->at(deviceOptions.deviceID).at(0)
            .enqueueReadBuffer(deviceMemory.snrData, CL_TRUE, 0,
                               hostMemory.snrData.size() * sizeof(float),
                               reinterpret_cast<void *>(hostMemory.snrData.data()), nullptr, &syncPoint);
          syncPoint.wait();
          openclRunTime.queues->at(deviceOptions.deviceID).at(0)
            .enqueueReadBuffer(deviceMemory.snrSamples, CL_TRUE, 0,
                               hostMemory.snrSamples.size() * sizeof(unsigned int),
                               reinterpret_cast<void *>(hostMemory.snrSamples.data()), nullptr, &syncPoint);
          syncPoint.wait();
          timers.outputCopy.stop();
        } else {
          openclRunTime.queues->at(deviceOptions.deviceID).at(0)
            .enqueueNDRangeKernel(*kernels.integration[stepNumber], cl::NullRange,
                                  kernelRunTimeConfigurations.integrationGlobal[stepNumber],
                                  kernelRunTimeConfigurations.integrationLocal[stepNumber]);
          openclRunTime.queues->at(deviceOptions.deviceID).at(0)
            .enqueueNDRangeKernel(*kernels.snr[stepNumber], cl::NullRange,
                                  kernelRunTimeConfigurations.snrGlobal[stepNumber],
                                  kernelRunTimeConfigurations.snrLocal[stepNumber]);
          openclRunTime.queues->at(deviceOptions.deviceID).at(0)
            .enqueueReadBuffer(deviceMemory.snrData, CL_FALSE, 0,
                               hostMemory.snrData.size() * sizeof(float),
                               reinterpret_cast<void *>(hostMemory.snrData.data()));
          openclRunTime.queues->at(deviceOptions.deviceID).at(0)
            .enqueueReadBuffer(deviceMemory.snrSamples, CL_FALSE, 0,
                               hostMemory.snrSamples.size() * sizeof(unsigned int),
                               reinterpret_cast<void *>(hostMemory.snrSamples.data()));
          openclRunTime.queues->at(deviceOptions.deviceID).at(0).finish();
        }
      } catch ( cl::Error & err ) {
        std::cerr << "SNR integration loop error -- Batch: " << std::to_string(batch) << ", Step: ";
        std::cerr << std::to_string(*step) << ", " << err.what() << " " << err.err() << std::endl;
        errorDetected = true;
      }
      if ( options.debug ) {
        try {
          openclRunTime.queues->at(deviceOptions.deviceID).at(0)
            .enqueueReadBuffer(deviceMemory.integratedData, CL_TRUE, 0,
                               hostMemory.integratedData.size() * sizeof(outputDataType),
                               reinterpret_cast<void *>(hostMemory.integratedData.data()), nullptr, &syncPoint);
          syncPoint.wait();
        } catch ( cl::Error & err ) {
          std::cerr << "Impossible to read deviceMemory.integratedData: " << err.what() << " " << err.err();
          std::cerr << std::endl;
          errorDetected = true;
        }
        std::cerr << "integratedData" << std::endl;
        if ( options.subbandDedispersion ) {
          for ( unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++ ) {
            std::cerr << "sBeam: " << sBeam << std::endl;
            for ( unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++ ) {
              for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
                std::cerr << "DM: " << (subbandingDM * observation.getNrDMs()) + dm << std::endl;
                for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch() / *step; sample++ ) {
                  std::cerr << hostMemory.integratedData.at((sBeam * observation.getNrDMs(true)
                                                             * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() /
                                                                                                        *(hostMemory.integrationSteps.begin()), deviceOptions.padding.at(deviceOptions.deviceName)
                                                                                                                                                / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs() *
                                                                                                                                                                              isa::utils::pad(observation.getNrSamplesPerBatch() / *(hostMemory.integrationSteps.begin()),
                                                                                                                                                                                              deviceOptions.padding.at(deviceOptions.deviceName)
                                                                                                                                                                                              / sizeof(outputDataType))) + (dm
                                                                                                                                                                                                                            * isa::utils::pad(observation.getNrSamplesPerBatch() / *(hostMemory.integrationSteps.begin()),
                                                                                                                                                                                                                                              deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(outputDataType)))
                                                            + sample) << " ";
                }
                std::cerr << std::endl;
              }
            }
            std::cerr << std::endl;
          }
        } else {
          for ( unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++ ) {
            std::cerr << "sBeam: " << sBeam << std::endl;
            for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
              std::cerr << "DM: " << dm << std::endl;
              for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch() / *step; sample++ ) {
                std::cerr << hostMemory.integratedData
                  .at((sBeam * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() /
                                                                        *(hostMemory.integrationSteps.begin()), deviceOptions.padding.at(deviceOptions.deviceName)
                                                                                                                / sizeof(outputDataType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() /
                                                                                                                                                                   *(hostMemory.integrationSteps.begin()), deviceOptions.padding.at(deviceOptions.deviceName)
                                                                                                                                                                                                           / sizeof(outputDataType))) + sample) << " ";
              }
              std::cerr << std::endl;
            }
            std::cerr << std::endl;
          }
        }
      }
      timers.trigger.start();
      trigger(options.subbandDedispersion, deviceOptions.padding.at(deviceOptions.deviceName), *step, options.threshold,
              observation, hostMemory.snrData, hostMemory.snrSamples, triggeredEvents);
      timers.trigger.stop();
      if ( options.debug ) {
        if ( options.subbandDedispersion ) {
          std::cerr << "hostMemory.snrData" << std::endl;
          for ( unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++ ) {
            std::cerr << "sBeam: " << sBeam << std::endl;
            for ( unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++ ) {
              for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
                std::cerr << hostMemory.snrData
                  .at((sBeam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(),
                                               deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(float)))
                      + (subbandingDM * observation.getNrDMs()) + dm) << " ";
              }
            }
            std::cerr << std::endl;
          }
        } else {
          std::cerr << "hostMemory.snrData" << std::endl;
          for ( unsigned int sBeam = 0; sBeam < observation.getNrSynthesizedBeams(); sBeam++ ) {
            std::cerr << "sBeam: " << sBeam << std::endl;
            for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
              std::cerr << hostMemory.snrData
                .at((sBeam * observation.getNrDMs(false, deviceOptions.padding.at(deviceOptions.deviceName)
                                                         / sizeof(float))) + dm) << " ";
            }
            std::cerr << std::endl;
          }
        }
        std::cerr << std::endl;
      }
    }
    if ( errorDetected ) {
      outputTrigger.close();
#ifdef HAVE_PSRDADA
      if ( dataOptions.dataPSRDADA ) {
        if ( dada_hdu_unlock_read(hostMemory.ringBuffer) != 0 ) {
          std::cerr << "Impossible to unlock the PSRDADA ringbuffer for reading the header." << std::endl;
        }
        dada_hdu_disconnect(hostMemory.ringBuffer);
      }
#endif // HAVE_PSRDADA
      return;
    }
    // Print and compact results
    timers.trigger.start();
    if ( options.compactResults ) {
      compact(observation, triggeredEvents, compactedEvents);
      for ( auto & compactedEvent : compactedEvents ) {
        for ( auto & event : compactedEvent ) {
          unsigned int integration = 0;
          float firstDM;
#ifdef HAVE_PSRDADA
          unsigned int delay = 0;
#endif // HAVE_PSRDADA

          if ( event.integration == 0 ) {
            integration = 1;
          } else {
            integration = event.integration;
          }
          if ( options.subbandDedispersion ) {
#ifdef HAVE_PSRDADA
            delay = observation.getNrDelayBatches(true) - 1;
#endif // HAVE_PSRADA
            firstDM = observation.getFirstDM(true);
          } else {
#ifdef HAVE_PSRDADA
            delay = observation.getNrDelayBatches() - 1;
#endif // HAVE_PSRDADA
            firstDM = observation.getFirstDM();
          }
          if ( dataOptions.dataPSRDADA ) {
#ifdef HAVE_PSRDADA
            outputTrigger << event.beam << " " << (batch - delay) << " " << event.sample  << " " << integration;
            outputTrigger << " " << event.compactedIntegration << " ";
            outputTrigger << (((batch - delay) * observation.getNrSamplesPerBatch()) + (event.sample * integration))
                * observation.getSamplingTime() << " " << firstDM + (event.DM * observation.getDMStep()) << " ";
            outputTrigger << event.compactedDMs << " " << event.SNR << std::endl;
#endif // HAVE_PSRDADA
          } else {
            outputTrigger << event.beam << " " << batch << " " << event.sample  << " " << integration << " ";
            outputTrigger << event.compactedIntegration << " " << ((batch * observation.getNrSamplesPerBatch())
                                                                   + (event.sample * integration)) * observation.getSamplingTime() << " ";
            outputTrigger << firstDM + (event.DM * observation.getDMStep()) << " " << event.compactedDMs << " ";
            outputTrigger << event.SNR << std::endl;
          }
        }
      }
    } else {
      for ( auto & triggeredEvent : triggeredEvents ) {
        for ( auto & dmEvents : triggeredEvent ) {
          for ( auto & event : dmEvents.second ) {
            unsigned int integration = 0;
            float firstDM;
#ifdef HAVE_PSRDADA
            unsigned int delay = 0;
#endif // HAVE_PSRDADA

            if (event.integration == 0 ) {
              integration = 1;
            } else {
              integration = event.integration;
            }
            if ( options.subbandDedispersion ) {
#ifdef HAVE_PSRDADA
              delay = observation.getNrDelayBatches(true) - 1;
#endif // HAVE_PSRDADA
              firstDM = observation.getFirstDM(true);
            } else {
#ifdef HAVE_PSRDADA
              delay = observation.getNrDelayBatches() - 1;
#endif // HAVE_PSRDADA
              firstDM = observation.getFirstDM();
            }
            if ( dataOptions.dataPSRDADA ) {
#ifdef HAVE_PSRDADA
              outputTrigger << event.beam << " " << (batch - delay) << " " << event.sample  << " " << integration;
              outputTrigger << " " << (((batch - delay) * observation.getNrSamplesPerBatch())
                  + (event.sample * integration)) * observation.getSamplingTime() << " ";
              outputTrigger << firstDM + (event.DM * observation.getDMStep()) << " " << event.SNR << std::endl;
#endif // HAVE_PSRDADA
            } else {
              outputTrigger << event.beam << " " << batch << " " << event.sample  << " " << integration << " ";
              outputTrigger << ((batch * observation.getNrSamplesPerBatch()) + (event.sample * integration))
                               * observation.getSamplingTime() << " " << firstDM + (event.DM * observation.getDMStep()) << " ";
              outputTrigger << event.SNR << std::endl;
            }
          }
        }
      }
    }
    timers.trigger.stop();
  }
#ifdef HAVE_PSRDADA
  if ( dataOptions.dataPSRDADA ) {
    if ( dada_hdu_unlock_read(hostMemory.ringBuffer) != 0 ) {
      std::cerr << "Impossible to unlock the PSRDADA ringbuffer for reading the header." << std::endl;
    }
    dada_hdu_disconnect(hostMemory.ringBuffer);
  }
#endif // HAVE_PSRDADA
  outputTrigger.close();
  timers.search.stop();

}

int inputHandling(const unsigned int batch, const AstroData::Observation & observation, const Options & options,
                   const DeviceOptions & deviceOptions, const DataOptions & dataOptions, Timers & timers,
                   HostMemory & hostMemory, DeviceMemory & deviceMemory) {
  // Load the input
  timers.inputHandling.start();
  if ( !dataOptions.dataPSRDADA || !dataOptions.streamingMode ) {
    // If there are not enough available batches, computation is complete
    if ( options.subbandDedispersion ) {
      if ( batch == observation.getNrBatches() - observation.getNrDelayBatches(true) ) {
        return -1;
      }
    } else {
      if ( batch == observation.getNrBatches() - observation.getNrDelayBatches() ) {
        return -1;
      }
    }
    // If there are enough batches, prepare them for transfer to device
    for ( unsigned int beam = 0; beam < observation.getNrBeams(); beam++ ) {
      if ( options.subbandDedispersion ) {
        if ( options.splitBatchesDedispersion ) {
          // TODO: implement or remove splitBatches mode
        } else {
          for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ ) {
            for ( unsigned int chunk = 0; chunk < observation.getNrDelayBatches(true) - 1; chunk++ ) {
              if ( inputBits >= 8 ) {
                memcpy(
                  reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (chunk * observation.getNrSamplesPerBatch())])),
                  reinterpret_cast<void *>(&((hostMemory.input.at(beam)->at(batch + chunk))->at(channel * observation.getNrSamplesPerBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))))),
                  observation.getNrSamplesPerBatch() * sizeof(inputDataType)
                );
              } else {
                memcpy(
                  reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (chunk * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])),
                  reinterpret_cast<void *>(&((hostMemory.input.at(beam)->at(batch + chunk))->at(channel * isa::utils::pad(observation.getNrSamplesPerBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))))),
                  (observation.getNrSamplesPerBatch() / (8 / inputBits)) * sizeof(inputDataType)
                );
              }
            }
            if ( inputBits >= 8 ) {
              memcpy(
                reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + ((observation.getNrDelayBatches(true) - 1) * observation.getNrSamplesPerBatch())])),
                reinterpret_cast<void *>(&((hostMemory.input.at(beam)->at(batch + (observation.getNrDelayBatches(true) - 1)))->at(channel * observation.getNrSamplesPerBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))))),
                (observation.getNrSamplesPerDispersedBatch(true) % observation.getNrSamplesPerBatch()) * sizeof(inputDataType)
              );
            } else {
              memcpy(
                reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + ((observation.getNrDelayBatches(true) - 1) * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])),
                reinterpret_cast<void *>(&((hostMemory.input.at(beam)->at(batch + (observation.getNrDelayBatches(true) - 1)))->at(channel * isa::utils::pad(observation.getNrSamplesPerBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))))),
                ((observation.getNrSamplesPerDispersedBatch(true) % observation.getNrSamplesPerBatch()) / (8 / inputBits)) * sizeof(inputDataType)
              );
            }
          }
        }
      } else {
        if ( options.splitBatchesDedispersion ) {
          // TODO: implement or remove splitBatches mode
        } else {
          for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ ) {
            for ( unsigned int chunk = 0; chunk < observation.getNrDelayBatches() - 1; chunk++ ) {
              if ( inputBits >= 8 ) {
                memcpy(
                  reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (chunk * observation.getNrSamplesPerBatch())])),
                  reinterpret_cast<void *>(&((hostMemory.input.at(beam)->at(batch + chunk))->at(channel * observation.getNrSamplesPerBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))))),
                  observation.getNrSamplesPerBatch() * sizeof(inputDataType)
                );
              } else {
                memcpy(
                  reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (chunk * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])),
                  reinterpret_cast<void *>(&((hostMemory.input.at(beam)->at(batch + chunk))->at(channel * isa::utils::pad(observation.getNrSamplesPerBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))))),
                  (observation.getNrSamplesPerBatch() / (8 / inputBits)) * sizeof(inputDataType)
                );
              }
            }
            if ( inputBits >= 8 ) {
              memcpy(
                reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + ((observation.getNrDelayBatches() - 1) * observation.getNrSamplesPerBatch())])),
                reinterpret_cast<void *>(&((hostMemory.input.at(beam)->at(batch + (observation.getNrDelayBatches() - 1)))->at(channel * observation.getNrSamplesPerBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))))),
                (observation.getNrSamplesPerDispersedBatch() % observation.getNrSamplesPerBatch()) * sizeof(inputDataType)
              );
            } else {
              memcpy(
                reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + ((observation.getNrDelayBatches() - 1) * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])),
                reinterpret_cast<void *>(&((hostMemory.input.at(beam)->at(batch + (observation.getNrDelayBatches() - 1)))->at(channel * isa::utils::pad(observation.getNrSamplesPerBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))))),
                ((observation.getNrSamplesPerDispersedBatch() % observation.getNrSamplesPerBatch()) / (8 / inputBits)) * sizeof(inputDataType)
              );
            }
          }
        }
      }
    }
  } else {
    if ( dataOptions.dataPSRDADA ) {
#ifdef HAVE_PSRDADA
      try {
        if ( ipcbuf_eod(reinterpret_cast< ipcbuf_t * >(hostMemory.ringBuffer->data_block)) ) {
          return -1;
        }
        if ( options.subbandDedispersion ) {
          AstroData::readPSRDADA(*hostMemory.ringBuffer,
                                 hostMemory.inputStream.at(batch % observation.getNrDelayBatches(true)));
        } else {
          AstroData::readPSRDADA(*hostMemory.ringBuffer,
                                 hostMemory.inputStream.at(batch % observation.getNrDelayBatches()));
        }
      } catch ( AstroData::RingBufferError & err ) {
        std::cerr << "Error: " << err.what() << std::endl;
        throw std::exception();
      }
#endif // HAVE_PSRDADA
    } else if ( dataOptions.streamingMode ) {
      readSIGPROC(observation, deviceOptions.padding.at(deviceOptions.deviceName), inputBits, dataOptions.headerSizeSIGPROC, dataOptions.dataFile, hostMemory.inputStream.at(batch % observation.getNrDelayBatches()), batch);
    }
    // If there are enough data buffered, proceed with the computation
    // Otherwise, move to the next iteration of the search loop
    if ( options.subbandDedispersion ) {
      if ( batch < observation.getNrDelayBatches(true) - 1 ) {
        return 1;
      }
    } else {
      if ( batch < observation.getNrDelayBatches() - 1 ) {
        return 1;
      }
    }
    for ( unsigned int beam = 0; beam < observation.getNrBeams(); beam++ ) {
      if ( options.subbandDedispersion ) {
      if ( options.splitBatchesDedispersion ) {
        // TODO: implement or remove splitBatches mode
      } else {
          for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ ) {
            for ( unsigned int chunk = batch - (observation.getNrDelayBatches(true) - 1); chunk < batch; chunk++ ) {
              // Full batches
              if ( inputBits >= 8 ) {
                memcpy(
                  reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + ((chunk - (batch - (observation.getNrDelayBatches(true) - 1))) * observation.getNrSamplesPerBatch())])),
                  reinterpret_cast<void *>(&(hostMemory.inputStream.at(chunk % observation.getNrDelayBatches(true))->at((beam * observation.getNrChannels() * observation.getNrSamplesPerBatch()) + (channel * observation.getNrSamplesPerBatch())))),
                  observation.getNrSamplesPerBatch() * sizeof(inputDataType)
                );
              } else {
                memcpy(
                  reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + ((chunk - (batch - (observation.getNrDelayBatches(true) - 1))) * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])),
                  reinterpret_cast<void *>(&(hostMemory.inputStream.at(chunk % observation.getNrDelayBatches(true))->at((beam * observation.getNrChannels() * (observation.getNrSamplesPerBatch() / (8 / inputBits))) + (channel * (observation.getNrSamplesPerBatch() / (8 / inputBits)))))),
                  (observation.getNrSamplesPerBatch() / (8 / inputBits)) * sizeof(inputDataType)
                );
              }
            }
            // Remainder (part of current batch)
            if ( inputBits >= 8 ) {
              memcpy(
                reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(true, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + ((observation.getNrDelayBatches(true) - 1) * observation.getNrSamplesPerBatch())])),
                reinterpret_cast<void *>(&(hostMemory.inputStream.at(batch % observation.getNrDelayBatches(true))->at((beam * observation.getNrChannels() * observation.getNrSamplesPerBatch()) + (channel * observation.getNrSamplesPerBatch())))),
                (observation.getNrSamplesPerDispersedBatch(true) % observation.getNrSamplesPerBatch()) * sizeof(inputDataType)
              );
            } else {
              memcpy(
                reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch(true) / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + ((observation.getNrDelayBatches(true) - 1) * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])),
                reinterpret_cast<void *>(&(hostMemory.inputStream.at(batch % observation.getNrDelayBatches(true))->at((beam * observation.getNrChannels() * (observation.getNrSamplesPerBatch() / (8 / inputBits))) + (channel * (observation.getNrSamplesPerBatch() / (8 / inputBits)))))),
                ((observation.getNrSamplesPerDispersedBatch(true) % observation.getNrSamplesPerBatch()) / (8 / inputBits)) * sizeof(inputDataType)
              );
            }
          }
        }
      } else {
      if ( options.splitBatchesDedispersion ) {
        // TODO: implement or remove splitBatches mode
      } else {
          for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ ) {
            for ( unsigned int chunk = batch - (observation.getNrDelayBatches() - 1); chunk < batch; chunk++ ) {
              if ( inputBits >= 8 ) {
                memcpy(
                  reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + ((chunk - (batch - (observation.getNrDelayBatches() - 1))) * observation.getNrSamplesPerBatch())])),
                  reinterpret_cast<void *>(&(hostMemory.inputStream.at(chunk % observation.getNrDelayBatches())->at((beam * observation.getNrChannels() * observation.getNrSamplesPerBatch()) + (channel * observation.getNrSamplesPerBatch())))),
                  observation.getNrSamplesPerBatch() * sizeof(inputDataType)
                );
              } else {
                memcpy(
                  reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + ((chunk - (batch - (observation.getNrDelayBatches() - 1))) * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])),
                  reinterpret_cast<void *>(&(hostMemory.inputStream.at(chunk % observation.getNrDelayBatches())->at((beam * observation.getNrChannels() * (observation.getNrSamplesPerBatch() / (8 / inputBits))) + (channel * (observation.getNrSamplesPerBatch() / (8 / inputBits)))))), (observation.getNrSamplesPerBatch() / (8 / inputBits)) * sizeof(inputDataType)
                );
              }
            }
            if ( inputBits >= 8 ) {
              memcpy(
                reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * observation.getNrSamplesPerDispersedBatch(false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + ((observation.getNrDelayBatches() - 1) * observation.getNrSamplesPerBatch())])),
                reinterpret_cast<void *>(&(hostMemory.inputStream.at(batch % observation.getNrDelayBatches())->at((beam * observation.getNrChannels() * observation.getNrSamplesPerBatch()) + (channel * observation.getNrSamplesPerBatch())))),
                (observation.getNrSamplesPerDispersedBatch() % observation.getNrSamplesPerBatch()) * sizeof(inputDataType)
              );
            } else {
              memcpy(
                reinterpret_cast<void *>(&(hostMemory.dispersedData.data()[(beam * observation.getNrChannels() * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel * isa::utils::pad(observation.getNrSamplesPerDispersedBatch() / (8 / inputBits), deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + ((observation.getNrDelayBatches() - 1) * (observation.getNrSamplesPerBatch() / (8 / inputBits)))])),
                reinterpret_cast<void *>(&(hostMemory.inputStream.at(batch % observation.getNrDelayBatches())->at((beam * observation.getNrChannels() * (observation.getNrSamplesPerBatch() / (8 / inputBits))) + (channel * (observation.getNrSamplesPerBatch() / (8 / inputBits)))))),
                ((observation.getNrSamplesPerDispersedBatch() % observation.getNrSamplesPerBatch()) / (8 / inputBits)) * sizeof(inputDataType)
              );
            }
          }
        }
      }
    }
  }
  timers.inputHandling.stop();
}

int copyInputToDevice(const unsigned int batch, const OpenCLRunTime & openclRunTime,
                      const AstroData::Observation & observation, const Options & options,
                      const DeviceOptions & deviceOptions, Timers & timers, HostMemory & hostMemory,
                      DeviceMemory & deviceMemory) {
  cl::Event syncPoint;

  // Copy input from host to device
  try {
    if ( deviceOptions.synchronized ) {
      timers.inputCopy.start();
      if ( options.splitBatchesDedispersion ) {
        // TODO: implement or remove splitBatches mode
      } else {
        openclRunTime.queues->at(deviceOptions.deviceID).at(0)
          .enqueueWriteBuffer(deviceMemory.dispersedData, CL_TRUE, 0,
                              hostMemory.dispersedData.size() * sizeof(inputDataType),
                              reinterpret_cast<void *>(hostMemory.dispersedData.data()), nullptr, &syncPoint);
      }
      syncPoint.wait();
      timers.inputCopy.stop();
    } else {
      if ( options.splitBatchesDedispersion ) {
        // TODO: implement or remove splitBatches mode
      } else {
        openclRunTime.queues->at(deviceOptions.deviceID).at(0)
          .enqueueWriteBuffer(deviceMemory.dispersedData, CL_FALSE, 0,
                              hostMemory.dispersedData.size() * sizeof(inputDataType),
                              reinterpret_cast<void *>(hostMemory.dispersedData.data()));
      }
    }
    if ( options.debug ) {
      // TODO: implement or remove splitBatches mode
      std::cerr << "dispersedData" << std::endl;
      if ( options.subbandDedispersion ) {
        if ( inputBits >= 8 ) {
          for ( unsigned int beam = 0; beam < observation.getNrBeams(); beam++ ) {
            std::cerr << "Beam: " << beam << std::endl;
            for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ ) {
              for ( unsigned int sample = 0;
                    sample < observation.getNrSamplesPerDispersedBatch(
                      true, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType));
                    sample++ ) {
                std::cerr << static_cast<float>(hostMemory.dispersedData.at(
                  (beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(
                    true, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel
                                                                                                          * observation.getNrSamplesPerDispersedBatch(
                    true, deviceOptions.padding.at(deviceOptions.deviceName)
                          / sizeof(inputDataType))) + sample)) << " ";
              }
              std::cerr << std::endl;
            }
            std::cerr << std::endl;
          }
        } else {
          // TODO: add support for input data less than 8 bit
        }
      } else {
        if ( inputBits >= 8 ) {
          for ( unsigned int beam = 0; beam < observation.getNrBeams(); beam++ ) {
            std::cerr << "Beam: " << beam << std::endl;
            for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ ) {
              for ( unsigned int sample = 0;
                    sample < observation.getNrSamplesPerDispersedBatch(
                      false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType));
                    sample++ ) {
                std::cerr << static_cast<float>(hostMemory.dispersedData.at(
                  (beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(
                    false, deviceOptions.padding.at(deviceOptions.deviceName) / sizeof(inputDataType))) + (channel
                                                                                                           * observation.getNrSamplesPerDispersedBatch(
                    false, deviceOptions.padding.at(deviceOptions.deviceName)
                           / sizeof(inputDataType))) + sample)) << " ";
              }
              std::cerr << std::endl;
            }
            std::cerr << std::endl;
          }
        } else {
          // TODO: add support for input data less than 8 bit
        }
      }
    }
  } catch ( cl::Error & err ) {
    std::cerr << "Input copy error -- Batch: " << std::to_string(batch) << ", " << err.what() << " " << err.err();
    std::cerr << std::endl;
    throw std::exception();
  }
  return 0;
}
