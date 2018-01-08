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

#include <KernelGenerator.hpp>

void generateOpenCLKernels(const AstroData::Observation & observation, const Options & options, const DeviceOptions & deviceOptions, const Configurations & configurations, Kernels & kernels) {
  std::string * code;

  if ( options.subbandDedispersion ) {
    code = Dedispersion::getSubbandDedispersionStepOneOpenCL<inputDataType, outputDataType>(*(configurations.dedispersionStepOneParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true))), deviceOptions.padding[deviceOptions.deviceName], inputBits, inputDataName, intermediateDataName, outputDataName, observation, *shiftsStepOne);
    try {
      kernels.dedispersionStepOne = isa::OpenCL::compile("dedispersionStepOne", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(deviceOptions.deviceID));
    } catch ( isa::OpenCL::OpenCLError & err ) {
      std::cerr << err.what() << std::endl;
      throw;
    }
    delete code;
    code = Dedispersion::getSubbandDedispersionStepTwoOpenCL<outputDataType>(*(configurations.dedispersionStepTwoParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())), deviceOptions.padding[deviceOptions.deviceName], outputDataName, observation, *shiftsStepTwo);
    try {
      kernels.dedispersionStepTwo = isa::OpenCL::compile("dedispersionStepTwo", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(deviceOptions.deviceID));
    } catch ( isa::OpenCL::OpenCLError & err ) {
      std::cerr << err.what() << std::endl;
      throw;
    }
    delete code;
  } else {
    code = Dedispersion::getDedispersionOpenCL<inputDataType, outputDataType>(*(configurations.dedispersionParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())), deviceOptions.padding[deviceOptions.deviceName], inputBits, inputDataName, intermediateDataName, outputDataName, observation, *shiftsStepOne);
    try {
      kernels.dedispersion = isa::OpenCL::compile("dedispersion", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(deviceOptions.deviceID));
    } catch ( isa::OpenCL::OpenCLError & err ) {
      std::cerr << err.what() << std::endl;
      throw;
    }
    delete code;
  }
  if ( options.subbandDedispersion ) {
    code = SNR::getSNRDMsSamplesOpenCL< outputDataType >(*(configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch())), outputDataName, observation, observation.getNrSamplesPerBatch(), deviceOptions.padding[deviceOptions.deviceName]);
  } else {
    code = SNR::getSNRDMsSamplesOpenCL< outputDataType >(*(configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch())), outputDataName, observation, observation.getNrSamplesPerBatch(), deviceOptions.padding[deviceOptions.deviceName]);
  }
  try {
    snrDMsSamplesK[integrationSteps.size()] = isa::OpenCL::compile("snrDMsSamples" + std::to_string(observation.getNrSamplesPerBatch()), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(deviceOptions.deviceID));
    snrDMsSamplesK[integrationSteps.size()]->setArg(0, dedispersedData_d);
    snrDMsSamplesK[integrationSteps.size()]->setArg(1, snrData_d);
    snrDMsSamplesK[integrationSteps.size()]->setArg(2, snrSamples_d);
  } catch ( isa::OpenCL::OpenCLError & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }
  delete code;
  for ( unsigned int stepNumber = 0; stepNumber < integrationSteps.size(); stepNumber++ ) {
    auto step = integrationSteps.begin();

    std::advance(step, stepNumber);
    if ( options.subbandDedispersion ) {
      code = Integration::getIntegrationDMsSamplesOpenCL< outputDataType >(*(configurations.integrationParameters[deviceOptions.deviceName]->at(observation.getNrDMs(true) * observation.getNrDMs())->at(*step)), observation, outputDataName, *step, deviceOptions.padding[deviceOptions.deviceName]);
    } else {
      code = Integration::getIntegrationDMsSamplesOpenCL< outputDataType >(*(configurations.integrationParameters[deviceOptions.deviceName]->at(observation.getNrDMs())->at(*step)), observation, outputDataName, *step, deviceOptions.padding[deviceOptions.deviceName]);
    }
    try {
      if ( *step > 1 ) {
        integrationDMsSamplesK[stepNumber] = isa::OpenCL::compile("integrationDMsSamples" + std::to_string(*step), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(deviceOptions.deviceID));
        integrationDMsSamplesK[stepNumber]->setArg(0, dedispersedData_d);
        integrationDMsSamplesK[stepNumber]->setArg(1, integratedData_d);
      }
    } catch ( isa::OpenCL::OpenCLError & err ) {
      std::cerr << err.what() << std::endl;
      return 1;
    }
    delete code;
    if ( options.subbandDedispersion ) {
      code = SNR::getSNRDMsSamplesOpenCL< outputDataType >(*(configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs(true) * observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)), outputDataName, observation, observation.getNrSamplesPerBatch() / *step, deviceOptions.padding[deviceOptions.deviceName]);
    } else {
      code = SNR::getSNRDMsSamplesOpenCL< outputDataType >(*(configurations.snrParameters.at(deviceOptions.deviceName)->at(observation.getNrDMs())->at(observation.getNrSamplesPerBatch() / *step)), outputDataName, observation, observation.getNrSamplesPerBatch() / *step, deviceOptions.padding[deviceOptions.deviceName]);
    }
    try {
      snrDMsSamplesK[stepNumber] = isa::OpenCL::compile("snrDMsSamples" + std::to_string(observation.getNrSamplesPerBatch() / *step), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(deviceOptions.deviceID));
      snrDMsSamplesK[stepNumber]->setArg(0, integratedData_d);
      snrDMsSamplesK[stepNumber]->setArg(1, snrData_d);
      snrDMsSamplesK[stepNumber]->setArg(2, snrSamples_d);
    } catch ( isa::OpenCL::OpenCLError & err ) {
      std::cerr << err.what() << std::endl;
      return 1;
    }
    delete code;
  }
}
