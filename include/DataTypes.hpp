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

#pragma once

struct Options
{
    // Debug mode
    bool debug = false;
    // Print messages to standard output
    bool print = false;
    // Use subband dedispersion
    bool subbandDedispersion = false;
    // Avoid merging batches of dispersed data into contiguous memory
    bool splitBatchesDedispersion = false;
    // Compact the triggered events in time and DM dimensions
    bool compactResults = false;
    // Threshold for triggering
    float threshold = 0.0f;
    // SNR mode
    SNRMode snrMode;
    // Step size for median of medians (MOMAD mode)
    unsigned int medianStepSize = 5;
};

struct DeviceOptions
{
    // OpenCL synchronized operations
    bool synchronized = false;
    // OpenCL platform ID
    unsigned int platformID = 0;
    // OpenCL device ID
    unsigned int deviceID = 0;
    // OpenCL device name
    std::string deviceName{};
    // Padding of OpenCL devices
    AstroData::paddingConf padding{};
};

struct DataOptions
{
    // Use LOFAR file as input
    bool dataLOFAR = false;
    // Use SIGPROC file as input
    bool dataSIGPROC = false;
    // Use PSRDADA buffer as input
    bool dataPSRDADA = false;
    // SIGPROC streaming mode
    bool streamingMode = false;
    // Size (in bytes) of the SIGPROC file header
    unsigned int headerSizeSIGPROC = 0;
    // Name of the input file
    std::string dataFile{};
    // Basename for the output files
    std::string outputFile{};
    // Name of the file containing the zapped channels
    std::string channelsFile{};
    // Name of the file containing the integration steps
    std::string integrationFile{};
#ifdef HAVE_HDF5
    // Limit the number of batches processed from a LOFAR file
    bool limit = false;
    // Name of the LOFAR header file
    std::string headerFile{};
#endif // HAVE_HDF5
#ifdef HAVE_PSRDADA
    // PSRDADA buffer key
    key_t dadaKey = 0;
#endif // HAVE_PSRDADA
};

struct GeneratorOptions
{
    // Use random numbers in generated data
    bool random = false;
    // Width of random generated pulse
    unsigned int width = 0;
    // DM of random generated pulse
    float DM = 0.0f;
};

struct HostMemory
{
    // Input data when reading a whole input file
    std::vector<std::vector<std::vector<inputDataType> *> *> input;
    // Input data when in streaming mode
    std::vector<std::vector<inputDataType> *> inputStream;
    // Zapped channels
    std::vector<unsigned int> zappedChannels;
    // Integration steps
    std::set<unsigned int> integrationSteps;
    // Map to create synthesized beams
    std::vector<unsigned int> beamMapping;
    // Dispersed data
    std::vector<inputDataType> dispersedData;
    // Subbanded data
    std::vector<outputDataType> subbandedData;
    // Dedispersed data
    std::vector<outputDataType> dedispersedData;
    // Integrated data
    std::vector<outputDataType> integratedData;
    // SNR data (SNR mode)
    std::vector<float> snrData;
    // Index of the sample with highest SNR value (SNR mode)
    std::vector<unsigned int> snrSamples;
    // Value of max sample (MOMAD mode)
    std::vector<outputDataType> maxValues;
    // Index of max sample (MOMAD MODE)
    std::vector<unsigned int> maxIndices;
    // Medians of medians (MOMAD mode)
    std::vector<outputDataType> mediansOfMedians;
    // Medians of medians absolute deviation (MOMAD mode)
    std::vector<outputDataType> medianOfMediansAbsoluteDeviation;
    // Shifts single step dedispersion
    std::vector<float> *shiftsSingleStep = nullptr;
    // Shifts step one subbanding dedispersion
    std::vector<float> *shiftsStepOne = nullptr;
    // Shifts step two subbanding dedispersion
    std::vector<float> *shiftsStepTwo = nullptr;
#ifdef HAVE_PSRDADA
    // PSRDADA ring buffer
    dada_hdu_t *ringBuffer = nullptr;
#endif // HAVE_PSRDADA
};

struct DeviceMemory
{
    // Shifts single step dedispersion
    cl::Buffer shiftsSingleStep;
    // Shifts step one subbanding dedispersion
    cl::Buffer shiftsStepOne;
    // Shifts step two subbanding dedispersion
    cl::Buffer shiftsStepTwo;
    // Zapped channels
    cl::Buffer zappedChannels;
    // Map to create synthesized beams
    cl::Buffer beamMapping;
    // Dispersed dada
    cl::Buffer dispersedData;
    // Subbanded data
    cl::Buffer subbandedData;
    // Dedispersed data
    cl::Buffer dedispersedData;
    // Integrated data
    cl::Buffer integratedData;
    // SNR data (SNR mode)
    cl::Buffer snrData;
    // Index of the sample with highest SNR value (SNR mode)
    cl::Buffer snrSamples;
    // Value of max sample (MOMAD mode)
    cl::Buffer maxValues;
    // Index of max sample (MOMAD mode)
    cl::Buffer maxIndices;
    // Median of medians first step (MOMAD mode)
    cl::Buffer medianOfMediansStepOne;
    // Median of medians second step (MOMAD mode)
    cl::Buffer medianOfMediansStepTwo;
};

struct KernelConfigurations
{
    // Configuration of single step dedispersion kernel
    Dedispersion::tunedDedispersionConf dedispersionSingleStepParameters;
    // Configuration of subband dedispersion kernel, step one
    Dedispersion::tunedDedispersionConf dedispersionStepOneParameters;
    // Configuration of subband dedispersion kernel, step two
    Dedispersion::tunedDedispersionConf dedispersionStepTwoParameters;
    // Configuration of integration kernel
    Integration::tunedIntegrationConf integrationParameters;
    // Configuration of SNR kernel (SNR mode)
    SNR::tunedSNRConf snrParameters;
    // Configuration of max kernel (MOMAD mode)
    SNR::tunedSNRConf maxParameters;
    // Configuration of median of medians first step (MOMAD mode)
    SNR::tunedSNRConf medianOfMediansStepOneParameters;
    // Configuration of median of medians second step (MOMAD mode)
    SNR::tunedSNRConf medianOfMediansStepTwoParameters;
    // Configuration of median of medians absolute deviation (MOMAD mode)
    SNR::tunedSNRConf medianOfMediansAbsoluteDeviationParameters;
};

struct Kernels
{
    // Single step dedispersion kernel
    cl::Kernel *dedispersionSingleStep = nullptr;
    // Step one subbanding dedispersion kernel
    cl::Kernel *dedispersionStepOne = nullptr;
    // Step two subbanding dedispersion kernel
    cl::Kernel *dedispersionStepTwo = nullptr;
    // Integration kernels, one for each integration step
    std::vector<cl::Kernel *> integration;
    // SNR kernels, one for the original data and one for each integration step (SNR mode)
    std::vector<cl::Kernel *> snr;
    // Max kernels, one for the original data and one for each integration step (MOMAD mode)
    std::vector<cl::Kernel *> max;
    // Median of medians first step, one for the original data and one for each integration step (MOMAD mode)
    std::vector<cl::Kernel *> medianOfMediansStepOne;
    // Median of medians second step, one for the original data and one for each integration step (MOMAD mode)
    std::vector<cl::Kernel *> medianOfMediansStepTwo;
    // Median of medians absolute deviation, one for the original data and one for each integration step (MOMAD mode)
    std::vector<cl::Kernel *> medianOfMediansAbsoluteDeviation;
};

struct KernelRunTimeConfigurations
{
    // Global NDrange for single step dedispersion
    cl::NDRange dedispersionSingleStepGlobal;
    // Local NDRange for single step dedispersion
    cl::NDRange dedispersionSingleStepLocal;
    // Global NDRange for subbanding dedispersion step one
    cl::NDRange dedispersionStepOneGlobal;
    // Local NDRange for subbanding dedispersion step one
    cl::NDRange dedispersionStepOneLocal;
    // Global NDRange for subbanding dedispersion step two
    cl::NDRange dedispersionStepTwoGlobal;
    // Local NDRange for subbanding dedispersion step two
    cl::NDRange dedispersionStepTwoLocal;
    // Global NDRange for integration
    std::vector<cl::NDRange> integrationGlobal;
    // Local NDRange for integration
    std::vector<cl::NDRange> integrationLocal;
    // Global NDRange for SNR (SNR mode)
    std::vector<cl::NDRange> snrGlobal;
    // Local NDRange for SNR (SNR mode)
    std::vector<cl::NDRange> snrLocal;
    // Global NDRange for max (MOMAD mode)
    std::vector<cl::NDRange> maxGlobal;
    // Local NDRange for max (MOMAD mode)
    std::vector<cl::NDRange> maxLocal;
    // Global NDRange for median of medians first step (MOMAD mode)
    std::vector<cl::NDRange> medianOfMediansStepOneGlobal;
    // Local NDRange for median of medians first step (MOMAD mode)
    std::vector<cl::NDRange> medianOfMediansStepOneLocal;
    // Global NDRange for median of medians second step (MOMAD mode)
    std::vector<cl::NDRange> medianOfMediansStepTwoGlobal;
    // Local NDRange for median of medians second step (MOMAD mode)
    std::vector<cl::NDRange> medianOfMediansStepTwoLocal;
    // Global NDRange for median of medians absolute deviation (MOMAD mode)
    std::vector<cl::NDRange> medianOfMediansAbsoluteDeviationGlobal;
    // Local NDRange for median of medians absolute deviation (MOMAD mode)
    std::vector<cl::NDRange> medianOfMediansAbsoluteDeviationLocal;
};

struct Timers
{
    isa::utils::Timer inputLoad;
    isa::utils::Timer search;
    isa::utils::Timer inputHandling;
    isa::utils::Timer inputCopy;
    isa::utils::Timer dedispersionSingleStep;
    isa::utils::Timer dedispersionStepOne;
    isa::utils::Timer dedispersionStepTwo;
    isa::utils::Timer integration;
    isa::utils::Timer snr;
    isa::utils::Timer max;
    isa::utils::Timer medianOfMediansStepOne;
    isa::utils::Timer medianOfMediansStepTwo;
    isa::utils::Timer medianOfMediansAbsoluteDeviationStepOne;
    isa::utils::Timer medianOfMediansAbsoluteDeviationStepTwo;
    isa::utils::Timer outputCopy;
    isa::utils::Timer trigger;
};

struct TriggeredEvent
{
    unsigned int beam = 0;
    unsigned int sample = 0;
    unsigned int integration = 0;
    float DM = 0.0f;
    float SNR = 0.0f;
};

struct CompactedEvent : TriggeredEvent
{
    unsigned int compactedIntegration = 1;
    unsigned int compactedDMs = 1;
};

using TriggeredEvents = std::vector<std::map<unsigned int, std::vector<TriggeredEvent>>>;
using CompactedEvents = std::vector<std::vector<CompactedEvent>>;

struct OpenCLRunTime
{
    cl::Context *context = nullptr;
    std::vector<cl::Platform> *platforms = nullptr;
    std::vector<cl::Device> *devices = nullptr;
    std::vector<std::vector<cl::CommandQueue>> *queues = nullptr;
};

enum SNRMode
{
    Standard,
    Momad
};
