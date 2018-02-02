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

#include <Kernels.hpp>

void generateOpenCLKernels(const OpenCLRunTime & openclRunTime, const AstroData::Observation & observation, const Options & options, const DeviceOptions & deviceOptions, const KernelConfigurations & kernelConfigurations, const HostMemory & hostMemory, Kernels & kernels) {
  std::string * code;

  if ( ! options.subbandDedispersion ) {
    code = Dedispersion::getDedispersionOpenCL<inputDataType, outputDataType>(*(kernelConfigurations.dedispersionSingleStepParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())), deviceOptions.padding.at(deviceOptions.deviceName), inputBits, inputDataName, intermediateDataName, outputDataName, observation, *shiftsStepOne);
    try {
      kernels.dedispersionSingleStep = isa::OpenCL::compile("dedispersion", *code, "-cl-mad-enable -Werror", *openclRunTime.context, openclRunTime.devices->at(deviceOptions.deviceID));
    } catch ( isa::OpenCL::OpenCLError & err ) {
      std::cerr << err.what() << std::endl;
      throw;
    }
    delete code;
  } else {
    code = Dedispersion::getSubbandDedispersionStepOneOpenCL<inputDataType, outputDataType>(*(kernelConfigurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))), deviceOptions.padding.at(deviceOptions.deviceName), inputBits, inputDataName, intermediateDataName, outputDataName, observation, *shiftsStepOne);
    try {
      kernels.dedispersionStepOne = isa::OpenCL::compile("dedispersionStepOne", *code, "-cl-mad-enable -Werror", *openclRunTime.context, openclRunTime.devices->at(deviceOptions.deviceID));
    } catch ( isa::OpenCL::OpenCLError & err ) {
      std::cerr << err.what() << std::endl;
      throw;
    }
    delete code;
    code = Dedispersion::getSubbandDedispersionStepTwoOpenCL<outputDataType>(*(kernelConfigurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())), deviceOptions.padding.at(deviceOptions.deviceName), outputDataName, observation, *shiftsStepTwo);
    try {
      kernels.dedispersionStepTwo = isa::OpenCL::compile("dedispersionStepTwo", *code, "-cl-mad-enable -Werror", *openclRunTime.context, openclRunTime.devices->at(deviceOptions.deviceID));
    } catch ( isa::OpenCL::OpenCLError & err ) {
      std::cerr << err.what() << std::endl;
      throw;
    }
    delete code;
  }
  if ( ! options.subbandDedispersion ) {
    code = SNR::getSNRDMsSamplesOpenCL< outputDataType >(*(kernelConfigurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch())), outputDataName, observation, observation.getNrSamplesPerBatch(), deviceOptions.padding.at(deviceOptions.deviceName));
  } else {
    code = SNR::getSNRDMsSamplesOpenCL< outputDataType >(*(kernelConfigurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch())), outputDataName, observation, observation.getNrSamplesPerBatch(), deviceOptions.padding.at(deviceOptions.deviceName));
  }
  try {
    kernels.snr.at(hostMemory.integrationSteps.size()) = isa::OpenCL::compile("snrDMsSamples" + std::to_string(observation.getNrSamplesPerBatch()), *code, "-cl-mad-enable -Werror", *openclRunTime.context, openclRunTime.devices->at(deviceOptions.deviceID));
    kernels.snr.at(hostMemory.integrationSteps.size())->setArg(0, deviceMemory.dedispersedData);
    kernels.snr.at(hostMemory.integrationSteps.size())->setArg(1, deviceMemory.snrData);
    kernels.snr.at(hostMemory.integrationSteps.size())->setArg(2, deviceMemory.snrSamples);
  } catch ( isa::OpenCL::OpenCLError & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }
  delete code;
  for ( unsigned int stepNumber = 0; stepNumber < hostMemory.integrationSteps.size(); stepNumber++ ) {
    auto step = hostMemory.integrationSteps.begin();

    std::advance(step, stepNumber);
    if ( options.subbandDedispersion ) {
      code = Integration::getIntegrationDMsSamplesOpenCL< outputDataType >(*(kernelConfigurations.integrationParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(*step)), observation, outputDataName, *step, deviceOptions.padding.at(deviceOptions.deviceName));
    } else {
      code = Integration::getIntegrationDMsSamplesOpenCL< outputDataType >(*(kernelConfigurations.integrationParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(*step)), observation, outputDataName, *step, deviceOptions.padding.at(deviceOptions.deviceName));
    }
    try {
      if ( *step > 1 ) {
        kernels.integration.at(stepNumber) = isa::OpenCL::compile("integrationDMsSamples" + std::to_string(*step), *code, "-cl-mad-enable -Werror", *openclRunTime.context, openclRunTime.devices->at(deviceOptions.deviceID));
        kernels.integration.at(stepNumber)->setArg(0, deviceMemory.dedispersedData);
        kernels.integration.at(stepNumber)->setArg(1, deviceMemory.integratedData);
      }
    } catch ( isa::OpenCL::OpenCLError & err ) {
      std::cerr << err.what() << std::endl;
      return 1;
    }
    delete code;
    if ( options.subbandDedispersion ) {
      code = SNR::getSNRDMsSamplesOpenCL< outputDataType >(*(kernelConfigurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)), outputDataName, observation, observation.getNrSamplesPerBatch() / *step, deviceOptions.padding.at(deviceOptions.deviceName));
    } else {
      code = SNR::getSNRDMsSamplesOpenCL< outputDataType >(*(kernelConfigurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)), outputDataName, observation, observation.getNrSamplesPerBatch() / *step, deviceOptions.padding.at(deviceOptions.deviceName));
    }
    try {
      kernels.snr.at(stepNumber) = isa::OpenCL::compile("snrDMsSamples" + std::to_string(observation.getNrSamplesPerBatch() / *step), *code, "-cl-mad-enable -Werror", *openclRunTime.context, openclRunTime.devices->at(deviceOptions.deviceID));
      kernels.snr.at(stepNumber)->setArg(0, deviceMemory.integratedData);
      kernels.snr.at(stepNumber)->setArg(1, deviceMemory.snrData);
      kernels.snr.at(stepNumber)->setArg(2, deviceMemory.snrSamples);
    } catch ( isa::OpenCL::OpenCLError & err ) {
      std::cerr << err.what() << std::endl;
      return 1;
    }
    delete code;
  }
}

void generateOpenCLRunTimeConfigurations(const AstroData::Observation & observation, const Options & options, const DeviceOptions & deviceOptions, const KernelConfigurations & kernelConfigurations, const HostMemory & hostMemory, KernelRunTimeConfigurations & kernelRunTimeConfigurations) {
  if ( ! options.subbandDedispersion ) {
    kernelRunTimeConfigurations.dedispersionSingleStepGlobal = cl::NDRange(isa::utils::pad(observation.getNrSamplesPerBatch() / kernelConfigurations.dedispersionSingleStepParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrItemsD0(), kernelConfigurations.dedispersionSingleStepParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrThreadsD0()), observation.getNrDMs() / kernelConfigurations.dedispersionSingleStepParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrItemsD1(), observation.getNrSynthesizedBeams());
    kernelRunTimeConfigurations.dedispersionSingleStepLocal = cl::NDRange(kernelConfigurations.dedispersionSingleStepParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrThreadsD0(), kernelConfigurations.dedispersionSingleStepParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrThreadsD1(), 1);
    if ( options.debug ) {
      std::cout << "DedispersionSingleStep" << std::endl;
      std::cout << "Global: " << isa::utils::pad(observation.getNrSamplesPerBatch() / kernelConfigurations.dedispersionSingleStepParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrItemsD0(), kernelConfigurations.dedispersionSingleStepParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrThreadsD0()) << ", " << observation.getNrDMs() / kernelConfigurations.dedispersionSingleStepParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrItemsD1() << ", " << observation.getNrSynthesizedBeams() << std::endl;
      std::cout << "Local: " << kernelConfigurations.dedispersionSingleStepParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrThreadsD0() << ", " << kernelConfigurations.dedispersionSingleStepParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrThreadsD1() << ", 1" << std::endl;
      std::cout << "Parameters: ";
      std::cout << kernelConfigurations.dedispersionSingleStepParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->print() << std::endl;
      std::cout << std::endl;
    }
  } else {
    kernelRunTimeConfigurations.dedispersionStepOneGlobal = cl::NDRange(isa::utils::pad(observation.getNrSamplesPerBatch(true) / kernelConfigurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getNrItemsD0(), kernelConfigurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getNrThreadsD0()), observation.getNrDMs(true) / kernelConfigurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getNrItemsD1(), observation.getNrBeams() * observation.getNrSubbands());
    kernelRunTimeConfigurations.dedispersionStepOneLocal = cl::NDRange(kernelConfigurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getNrThreadsD0(), kernelConfigurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getNrThreadsD1(), 1);
    if ( options.debug ) {
      std::cout << "DedispersionStepOne" << std::endl;
        std::cout << "Global: " << isa::utils::pad(observation.getNrSamplesPerBatch(true) / kernelConfigurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getNrItemsD0(), kernelConfigurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getNrThreadsD0()) << ", " << observation.getNrDMs(true) / kernelConfigurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getNrItemsD1() << ", " << observation.getNrBeams() * observation.getNrSubbands() << std::endl;
        std::cout << "Local: " << kernelConfigurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getNrThreadsD0() << ", " << kernelConfigurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->getNrThreadsD1() << ", 1" << std::endl;
        std::cout << "Parameters: ";
        std::cout << kernelConfigurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))->print() << std::endl;
        std::cout << std::endl;
    }
    kernelRunTimeConfigurations.dedispersionStepTwoGlobal = cl::NDRange(isa::utils::pad(observation.getNrSamplesPerBatch(true) / kernelConfigurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrItemsD0(), kernelConfigurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrThreadsD0()), observation.getNrDMs() / kernelConfigurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrItemsD1(), observation.getNrSynthesizedBeams() * observation.getNrDMs(true));
    kernelRunTimeConfigurations.dedispersionStepTwoLocal = cl::NDRange(kernelConfigurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrThreadsD0(), kernelConfigurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrThreadsD1(), 1);
    if ( options.debug ) {
      std::cout << "DedispersionStepTwo" << std::endl;
        std::cout << "Global: " << isa::utils::pad(observation.getNrSamplesPerBatch(true) / kernelConfigurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrItemsD0(), kernelConfigurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrThreadsD0()) << ", " << observation.getNrDMs() / kernelConfigurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrItemsD1() << ", " << observation.getNrSynthesizedBeams() * observation.getNrDMs(true) << std::endl;
        std::cout << "Local: " << kernelConfigurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrThreadsD0() << ", " << kernelConfigurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->getNrThreadsD1() << ", 1" << std::endl;
        std::cout << "Parameters: ";
        std::cout << kernelConfigurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->print() << std::endl;
        std::cout << std::endl;
    }
  }
  kernelRunTimeConfigurations.integrationGlobal.resize(hostMemory.integrationSteps.size());
  kernelRunTimeConfigurations.integrationLocal.resize(hostMemory.integrationSteps.size());
  kernelRunTimeConfigurations.snrGlobal.resize(hostMemory.integrationSteps.size() + 1);
  kernelRunTimeConfigurations.snrLocal.resize(hostMemory.integrationSteps.size() + 1);
  if ( ! options.subbandDedispersion ) {
    kernelRunTimeConfigurations.snrGlobal.at(hostMemory.integrationSteps.size()) = cl::NDRange(kernelConfigurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch())->getNrThreadsD0(), observation.getNrDMs(), observation.getNrSynthesizedBeams());
    kernelRunTimeConfigurations.snrLocal.at(hostMemory.integrationSteps.size()) = cl::NDRange(kernelConfigurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch())->getNrThreadsD0(), 1, 1);
    if ( options.debug ) {
      std::cout << "SNR (" + std::to_string(observation.getNrSamplesPerBatch()) + ")" << std::endl;
      std::cout << "Global: " << kernelConfigurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch())->getNrThreadsD0() << ", " << observation.getNrDMs() << ", " << observation.getNrSynthesizedBeams() << std::endl;
      std::cout << "Local: " << kernelConfigurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch())->getNrThreadsD0() << ", 1, 1" << std::endl;
      std::cout << "Parameters: ";
      std::cout << kernelConfigurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch())->print() << std::endl;
      std::cout << std::endl;
    }
  } else {
    kernelRunTimeConfigurations.snrGlobal.at(hostMemory.integrationSteps.size()) = cl::NDRange(kernelConfigurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch())->getNrThreadsD0(), observation.getNrDMs(true) * observation.getNrDMs(), observation.getNrSynthesizedBeams());
    kernelRunTimeConfigurations.snrLocal.at(hostMemory.integrationSteps.size()) = cl::NDRange(kernelConfigurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch())->getNrThreadsD0(), 1, 1);
    if ( options.debug ) {
      std::cout << "SNR (" + std::to_string(observation.getNrSamplesPerBatch()) + ")" << std::endl;
      std::cout << "Global: " << kernelConfigurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch())->getNrThreadsD0() << ", " << observation.getNrDMs(true) * observation.getNrDMs() << " " << observation.getNrSynthesizedBeams() << std::endl;
      std::cout << "Local: " << kernelConfigurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch())->getNrThreadsD0() << ", 1, 1" << std::endl;
      std::cout << "Parameters: ";
      std::cout << kernelConfigurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch())->print() << std::endl;
      std::cout << std::endl;
    }
  }
  for ( unsigned int stepNumber = 0; stepNumber < hostMemory.integrationSteps.size(); stepNumber++ ) {
    auto step = hostMemory.integrationSteps.begin();

    std::advance(step, stepNumber);
    if ( ! options.subbandDedispersion ) {
      kernelRunTimeConfigurations.integrationGlobal.at(stepNumber) = cl::NDRange(kernelConfigurations.integrationParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(*step)->getNrThreadsD0() * ((observation.getNrSamplesPerBatch() / *step) / kernelConfigurations.integrationParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(*step)->getNrItemsD0()), observation.getNrDMs(), observation.getNrSynthesizedBeams());
      kernelRunTimeConfigurations.integrationLocal.at(stepNumber) = cl::NDRange(kernelConfigurations.integrationParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(*step)->getNrThreadsD0(), 1, 1);
      if ( options.debug ) {
        std::cout << "integration (" + std::to_string(*step) + ")" << std::endl;
        std::cout << "Global: " << kernelConfigurations.integrationParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(*step)->getNrThreadsD0() * ((observation.getNrSamplesPerBatch() / *step) / kernelConfigurations.integrationParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(*step)->getNrItemsD0()) << ", " << observation.getNrDMs() << ", " << observation.getNrSynthesizedBeams() << std::endl;
        std::cout << "Local: " << kernelConfigurations.integrationParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(*step)->getNrThreadsD0() << ", 1, 1" << std::endl;
        std::cout << "Parameters: ";
        std::cout << kernelConfigurations.integrationParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(*step)->print() << std::endl;
        std::cout << std::endl;
      }
      kernelRunTimeConfigurations.snrGlobal.at(stepNumber) = cl::NDRange(kernelConfigurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)->getNrThreadsD0(), observation.getNrDMs(), observation.getNrSynthesizedBeams());
      kernelRunTimeConfigurations.snrLocal.at(stepNumber) = cl::NDRange(kernelConfigurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)->getNrThreadsD0(), 1, 1);
      if ( options.debug ) {
        std::cout << "SNR (" + std::to_string(observation.getNrSamplesPerBatch() / *step) + ")" << std::endl;
        std::cout << "Global: " << kernelConfigurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)->getNrThreadsD0() << ", " << observation.getNrDMs() << " " << observation.getNrSynthesizedBeams() << std::endl;
        std::cout << "Local: " << kernelConfigurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)->getNrThreadsD0() << ", 1, 1" << std::endl;
        std::cout << "Parameters: ";
        std::cout << kernelConfigurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)->print() << std::endl;
        std::cout << std::endl;
      }
    } else {
      kernelRunTimeConfigurations.integrationGlobal.at(stepNumber) = cl::NDRange(kernelConfigurations.integrationParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(*step)->getNrThreadsD0() * ((observation.getNrSamplesPerBatch() / *step) / kernelConfigurations.integrationParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(*step)->getNrItemsD0()), observation.getNrDMs(true) * observation.getNrDMs(), observation.getNrSynthesizedBeams());
      kernelRunTimeConfigurations.integrationLocal.at(stepNumber) = cl::NDRange(kernelConfigurations.integrationParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(*step)->getNrThreadsD0(), 1, 1);
      if ( options.debug ) {
        std::cout << "integration (" + std::to_string(*step) + ")" << std::endl;
        std::cout << "Global: " << kernelConfigurations.integrationParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(*step)->getNrThreadsD0() * ((observation.getNrSamplesPerBatch() / *step) / kernelConfigurations.integrationParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(*step)->getNrItemsD0()) << ", " << observation.getNrDMs(true) * observation.getNrDMs() << ", " << observation.getNrSynthesizedBeams() << std::endl;
        std::cout << "Local: " << kernelConfigurations.integrationParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(*step)->getNrThreadsD0() << ", 1, 1" << std::endl;
        std::cout << "Parameters: ";
        std::cout << kernelConfigurations.integrationParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(*step)->print() << std::endl;
        std::cout << std::endl;
      }
      kernelRunTimeConfigurations.snrGlobal.at(stepNumber) = cl::NDRange(kernelConfigurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)->getNrThreadsD0(), observation.getNrDMs(true) * observation.getNrDMs(), observation.getNrSynthesizedBeams());
      kernelRunTimeConfigurations.snrLocal.at(stepNumber) = cl::NDRange(kernelConfigurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)->getNrThreadsD0(), 1, 1);
      if ( options.debug ) {
        std::cout << "SNR (" + std::to_string(observation.getNrSamplesPerBatch() / *step) + ")" << std::endl;
        std::cout << "Global: " << kernelConfigurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)->getNrThreadsD0() << ", " << observation.getNrDMs(true) * observation.getNrDMs() << " " << observation.getNrSynthesizedBeams() << std::endl;
        std::cout << "Local: " << kernelConfigurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)->getNrThreadsD0() << ", 1, 1" << std::endl;
        std::cout << "Parameters: ";
        std::cout << kernelConfigurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)->print() << std::endl;
        std::cout << std::endl;
      }
    }
  }
}
