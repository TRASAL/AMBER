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

typedef enum
{
    Standard,
    Momad,
    MomSigmaCut
} SNRMode;

struct Options
{
    // Debug mode
    bool debug = false;
    // Print messages to standard output
    bool print = false;
    /**
     ** @brief Enable RFI mitigation
     */
    bool rfim = false;
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
    /**
    ** @brief Sigma cut (MomSigmaCut)
    */
    float nSigma = 3.0f;
    /**
    ** @brief Correction Factor (MomSigmaCut)
    **
    ** Correction factor to compute SNR when the standard deviation is computed with a sigma cut.
    ** nSigma : 1.25 = sigmaCorrectionFactor : 1.544
    ** nSigma : 1.50 = sigmaCorrectionFactor : 1.350
    ** nSigma : 1.75 = sigmaCorrectionFactor : 1.222
    ** nSigma : 1.00 = sigmaCorrectionFactor : 1.853
    ** nSigma : 2.00 = sigmaCorrectionFactor : 1.136
    ** nSigma : 2.25 = sigmaCorrectionFactor : 1.082
    ** nSigma : 2.50 = sigmaCorrectionFactor : 1.049
    ** nSigma : 2.75 = sigmaCorrectionFactor : 1.027
    ** nSigma : 3.00 = sigmaCorrectionFactor : 1.014
    ** nSigma : 3.25 = sigmaCorrectionFactor : 1.006
    ** nSigma : 3.50 = sigmaCorrectionFactor : 1.003
    ** nSigma : 3.75 = sigmaCorrectionFactor : 1.001
    ** nSigma : 4.00 = sigmaCorrectionFactor : 1.001
    */
    float sigmaCorrectionFactor = 1.014f;
    /**
     ** @brief Data dumping mode.
     ** In this mode, all data are dumped to disk. Only intended for debug purpose.
     */
    bool dataDump = false;
    /**
     ** @brief Enable downsampling before dedispersion.
     */
    bool downsampling = false;
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
    // Value of standard deviation samples (MomSigmaCut mode)
    std::vector<outputDataType> stdevs;
    /**
     ** @brief Host storage for the first step of the median of medians; only used for debugging.
     */
    std::vector<outputDataType> medianOfMediansStepOne;
    // Medians of medians (MOMAD mode)
    std::vector<outputDataType> medianOfMedians;
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

struct HostMemoryDumpFiles
{
    /**
     ** @brief Prefix for the dump files.
     */
    std::string dumpFilesPrefix{};
    /**
     ** @brief File where to dump downsampled data.
     */
    std::ofstream downsampledData;
    /**
     ** @brief File where to dump subbanded data.
     */
    std::ofstream subbandedData;
    /**
     ** @brief File where to dump dedispersed data.
     */
    std::ofstream dedispersedData;
    /**
     ** @brief File where to dump integrated data.
     */
    std::ofstream integratedData;
    /**
     ** @brief File where to dump SNR data (SNR mode).
     */
    std::ofstream snrData;
    /**
     ** @brief File where to dump max SNR sample indices (SNR mode) data.
     */
    std::ofstream snrSamplesData;
    /**
     ** @brief File where to dump max values (MOMAD mode) data.
     */
    std::ofstream maxValuesData;
    /**
     ** @brief File where to dump max indices (MOMAD mode) data.
     */
    std::ofstream maxIndicesData;
    /**
     ** @brief File where to dump standard deviation values (MomSigmaCut mode) data.
     */
    std::ofstream stdevsData;
    /**
     ** @brief File where to dump the first step of the median of medians (MOMAD mode) data.
     */
    std::ofstream medianOfMediansStepOneData;
    /**
     ** @brief File where to dump median of medians (MOMAD mode) data.
     */
    std::ofstream medianOfMediansData;
    /**
     ** @brief File where to dump median of medians absolute deviation (MOMAD mode) data.
     */
    std::ofstream medianOfMediansAbsoluteDeviationData;
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
    // Values of standard deviation
    cl::Buffer stdevs;
    // Median of medians first step (MOMAD mode)
    cl::Buffer medianOfMediansStepOne;
    // Median of medians second step (MOMAD mode)
    cl::Buffer medianOfMediansStepTwo;
};

struct KernelConfigurations
{
    /**
     ** @brief Configuration for the time domain sigma cut (RFIm)
     */
    RFIm::RFImConfigurations timeDomainSigmaCutParameters;
    // Configuration of downsampling kernel
    Integration::tunedIntegrationConf downsamplingParameters;
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
    // Configuration of max and standard deviation kernel (MomSigmaCut mode)
    SNR::tunedSNRConf maxStdSigmaCutParameters;
};

struct Kernels
{
    /**
     ** @brief Time domain sigma cut kernels, one for each sigma value (RFIm)
     */
    std::vector<cl::Kernel *> timeDomainSigmaCut;
    // Downsampling kernel
    cl::Kernel *downsampling = nullptr;
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
    // Max and standard deviation kernels, one for the original data and one for each integration step (MomSigmaCut mode)
    std::vector<cl::Kernel *> maxStdSigmaCut;
};

struct KernelRunTimeConfigurations
{
    /**
     ** @brief Global NDRange for time domain sigma cut (RFIm).
     */
    std::vector<cl::NDRange> timeDomainSigmaCutGlobal;
    /**
     ** @brief Local NDRange for time domain sigma cut (RFIm).
     */
    std::vector<cl::NDRange> timeDomainSigmaCutLocal;
    // Global NDRange for downsampling
    cl::NDRange downsamplingGlobal;
    // Local NDRange for downsampling
    cl::NDRange downsamplingLocal;
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
    // Global NDRange for max and standard deviation (MomSigmaCut mode)
    std::vector<cl::NDRange> maxStdSigmaCutGlobal;
    // Local NDRange for max and standard deviation (MomSigmaCut mode)
    std::vector<cl::NDRange> maxStdSigmaCutLocal;
};

struct Timers
{
    isa::utils::Timer inputLoad;
    isa::utils::Timer search;
    isa::utils::Timer inputHandling;
    isa::utils::Timer inputCopy;
    /**
     ** @brief Timer for time domain sigma cut.
     */
    isa::utils::Timer timeDomainSigmaCut;
    // Timer for downsampling
    isa::utils::Timer downsampling;
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
    unsigned int DM = 0;
    float SNR = 0.0f;
};

struct CompactedEvent : TriggeredEvent
{
    unsigned int compactedIntegration = 1;
    unsigned int compactedDMs = 1;
};

using TriggeredEvents = std::vector<std::map<unsigned int, std::vector<TriggeredEvent>>>;
using CompactedEvents = std::vector<std::vector<CompactedEvent>>;
