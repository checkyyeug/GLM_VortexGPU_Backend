#include "core/dsp/dsd_processor.hpp"
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <cstring>

#ifdef VORTEX_ENABLE_CUDA
#include "core/gpu/cuda_kernels.hpp"
#endif

namespace vortex::core::dsp {

DSDProcessor::ProcessingState::~ProcessingState() {
    // Cleanup GPU memory
    if (gpuInputBuffer) {
#ifdef VORTEX_ENABLE_CUDA
        cudaFree(gpuInputBuffer);
#endif
    }
    if (gpuOutputBuffer) {
#ifdef VORTEX_ENABLE_CUDA
        cudaFree(gpuOutputBuffer);
#endif
    }
    if (gpuCoefficients) {
#ifdef VORTEX_ENABLE_CUDA
        cudaFree(gpuCoefficients);
#endif
    }
    if (gpuWorkBuffer) {
#ifdef VORTEX_ENABLE_CUDA
        cudaFree(gpuWorkBuffer);
#endif
    }
}

DSDProcessor::ProcessingState::ProcessingState(ProcessingState&& other) noexcept
    : inputBuffer(std::move(other.inputBuffer))
    , outputBuffer(std::move(other.outputBuffer))
    , filterCoeffs(std::move(other.filterCoeffs))
    , delayLine(std::move(other.delayLine))
    , fftBuffer(std::move(other.fftBuffer))
    , gpuInputBuffer(other.gpuInputBuffer)
    , gpuOutputBuffer(other.gpuOutputBuffer)
    , gpuCoefficients(other.gpuCoefficients)
    , gpuWorkBuffer(other.gpuWorkBuffer)
    , delayIndex(other.delayIndex)
    , gpuInitialized(other.gpuInitialized) {

    // Reset other to prevent double-free
    other.gpuInputBuffer = nullptr;
    other.gpuOutputBuffer = nullptr;
    other.gpuCoefficients = nullptr;
    other.gpuWorkBuffer = nullptr;
    other.gpuInitialized = false;
}

DSDProcessor::ProcessingState& DSDProcessor::ProcessingState::operator=(ProcessingState&& other) noexcept {
    if (this != &other) {
        // Cleanup existing resources
        this->~ProcessingState();

        // Move resources
        inputBuffer = std::move(other.inputBuffer);
        outputBuffer = std::move(other.outputBuffer);
        filterCoeffs = std::move(other.filterCoeffs);
        delayLine = std::move(other.delayLine);
        fftBuffer = std::move(other.fftBuffer);
        gpuInputBuffer = other.gpuInputBuffer;
        gpuOutputBuffer = other.gpuOutputBuffer;
        gpuCoefficients = other.gpuCoefficients;
        gpuWorkBuffer = other.gpuWorkBuffer;
        delayIndex = other.delayIndex;
        gpuInitialized = other.gpuInitialized;

        // Reset other
        other.gpuInputBuffer = nullptr;
        other.gpuOutputBuffer = nullptr;
        other.gpuCoefficients = nullptr;
        other.gpuWorkBuffer = nullptr;
        other.gpuInitialized = false;
    }
    return *this;
}

DSDProcessor::DSDProcessor(const Config& config)
    : config_(config)
    , initialized_(false)
    , totalSamplesProcessed_(0)
    , totalProcessingTime_(0.0)
    , gpuFallbacks_(0) {
}

DSDProcessor::~DSDProcessor() {
    shutdown();
}

bool DSDProcessor::initialize() {
    if (initialized_) {
        return true;
    }

    Logger::info("Initializing DSD1024 processor ({} Hz -> {} Hz, {} channels)",
                 config_.inputSampleRate, config_.outputSampleRate, config_.channels);

    try {
        // Calculate buffer sizes
        const uint32_t decimationRatio = config_.inputSampleRate / config_.outputSampleRate;
        const uint32_t inputBlockSize = config_.blockSize * decimationRatio;
        const uint32_t outputBlockSize = config_.blockSize;

        // Allocate input and output buffers
        state_.inputBuffer.resize(inputBlockSize * config_.channels);
        state_.outputBuffer.resize(outputBlockSize * config_.channels * 2); // Double for safety

        // Initialize filter coefficients
        initializeFilterCoefficients();

        // Initialize delay line for FIR filter
        state_.delayLine.resize(config_.filterOrder * config_.channels);
        state_.delayIndex = 0;

        // Initialize FFT workspace
        state_.fftBuffer.resize(std::max(inputBlockSize, outputBlockSize));

        // Initialize GPU if requested
        if (config_.gpuAcceleration) {
            if (!initializeGPU()) {
                Logger::warn("GPU initialization failed, falling back to CPU processing");
                config_.gpuAcceleration = false;
            }
        }

        initialized_ = true;
        Logger::info("DSD1024 processor initialized successfully (GPU: {})",
                     config_.gpuAcceleration ? "enabled" : "disabled");
        return true;

    } catch (const std::exception& e) {
        Logger::error("Failed to initialize DSD1024 processor: {}", e.what());
        return false;
    }
}

void DSDProcessor::shutdown() {
    if (!initialized_) {
        return;
    }

    Logger::info("Shutting down DSD1024 processor");

    // Cleanup GPU resources
    cleanupGPU();

    // Clear buffers
    state_.inputBuffer.clear();
    state_.outputBuffer.clear();
    state_.filterCoeffs.clear();
    state_.delayLine.clear();
    state_.fftBuffer.clear();

    initialized_ = false;

    // Log final statistics
    if (totalSamplesProcessed_ > 0) {
        double avgLatency = totalProcessingTime_ / totalSamplesProcessed_;
        Logger::info("DSD1024 processor stats: {} samples processed, avg latency: {:.3f}μs, GPU fallbacks: {}",
                     totalSamplesProcessed_, avgLatency, gpuFallbacks_);
    }
}

uint32_t DSDProcessor::processDSD1024(const uint8_t* input,
                                     uint32_t inputSamples,
                                     float* output,
                                     uint32_t maxOutputSamples) {
    if (!initialized_) {
        Logger::error("DSD processor not initialized");
        return 0;
    }

    if (!input || !output) {
        Logger::error("Invalid input/output pointers");
        return 0;
    }

    const auto startTime = std::chrono::high_resolution_clock::now();

    uint32_t outputSamples = 0;

    try {
        // Use GPU or CPU processing based on availability
        if (config_.gpuAcceleration && state_.gpuInitialized) {
            outputSamples = processGPU(input, inputSamples, output, maxOutputSamples);
            if (outputSamples == 0) {
                Logger::warn("GPU processing failed, falling back to CPU");
                gpuFallbacks_++;
                outputSamples = processCPU(input, inputSamples, output, maxOutputSamples);
            }
        } else {
            outputSamples = processCPU(input, inputSamples, output, maxOutputSamples);
        }

        // Apply post-processing effects
        if (outputSamples > 0) {
            const uint32_t totalOutputSamples = outputSamples * config_.channels;

            if (config_.noiseShaping) {
                applyNoiseShaping(output, totalOutputSamples);
            }

            if (config_.enableDynamicRange) {
                applyDynamicRangeCompression(output, totalOutputSamples);
            }

            if (config_.enableDithering) {
                applyDithering(output, totalOutputSamples);
            }

            // Apply gain normalization
            if (config_.gainNormalization != 1.0f) {
                for (uint32_t i = 0; i < totalOutputSamples; ++i) {
                    output[i] *= config_.gainNormalization;
                }
            }
        }

    } catch (const std::exception& e) {
        Logger::error("Exception during DSD1024 processing: {}", e.what());
        return 0;
    }

    // Update performance metrics
    const auto endTime = std::chrono::high_resolution_clock::now();
    const auto processingTime = std::chrono::duration_cast<std::chrono::microseconds>(
        endTime - startTime).count();

    totalSamplesProcessed_ += outputSamples;
    totalProcessingTime_ += static_cast<double>(processingTime);

    logPerformance(processingTime, outputSamples);

    return outputSamples;
}

uint32_t DSDProcessor::processRealTime(const uint8_t* input,
                                       size_t inputSize,
                                       float* output,
                                       size_t outputCapacity) {
    if (!initialized_) {
        return 0;
    }

    // Convert bytes to DSD samples (8 samples per byte)
    const uint32_t inputSamples = static_cast<uint32_t>(inputSize * 8);
    const uint32_t maxOutputSamples = static_cast<uint32_t>(outputCapacity / config_.channels);

    // Process with optimized block size for real-time performance
    return processDSD1024(input, inputSamples, output, maxOutputSamples);
}

void DSDProcessor::reset() {
    if (!initialized_) {
        return;
    }

    // Clear delay line
    std::fill(state_.delayLine.begin(), state_.delayLine.end(), 0.0f);
    state_.delayIndex = 0;

    // Clear buffers
    std::fill(state_.inputBuffer.begin(), state_.inputBuffer.end(), 0.0f);
    std::fill(state_.outputBuffer.begin(), state_.outputBuffer.end(), 0.0f);

    Logger::debug("DSD1024 processor reset");
}

bool DSDProcessor::updateConfig(const Config& newConfig) {
    if (initialized_) {
        Logger::warn("Cannot update configuration while processor is running");
        return false;
    }

    config_ = newConfig;
    Logger::info("DSD1024 processor configuration updated");
    return true;
}

std::string DSDProcessor::getProcessingStats() const {
    if (totalSamplesProcessed_ == 0) {
        return "{\"status\":\"no_processing_data\"}";
    }

    const double avgLatency = totalProcessingTime_ / totalSamplesProcessed_;
    const double samplesPerSecond = totalSamplesProcessed_ / (totalProcessingTime_ / 1000000.0);
    const double utilization = (samplesPerSecond / config_.outputSampleRate) * 100.0;

    char buffer[512];
    snprintf(buffer, sizeof(buffer),
        "{"
        "\"total_samples_processed\":%llu,"
        "\"average_latency_us\":%.3f,"
        "\"samples_per_second\":%.1f,"
        "\"utilization_percent\":%.2f,"
        "\"gpu_fallbacks\":%u,"
        "\"gpu_accelerated\":%s"
        "}",
        static_cast<unsigned long long>(totalSamplesProcessed_),
        avgLatency,
        samplesPerSecond,
        utilization,
        gpuFallbacks_,
        isGPUAccelerated() ? "true" : "false"
    );

    return std::string(buffer);
}

uint64_t DSDProcessor::getEstimatedLatency() const {
    if (!initialized_) {
        return 0;
    }

    // Base latency estimate based on processing pipeline
    const uint32_t decimationRatio = config_.inputSampleRate / config_.outputSampleRate;
    const uint32_t pipelineStages = 4; // DSD conversion, filtering, downsampling, post-processing
    const uint64_t baseLatency = pipelineStages * config_.blockSize * 1000000ULL / config_.outputSampleRate;

    // Add GPU transfer overhead if applicable
    if (isGPUAccelerated()) {
        return baseLatency + 100; // 100μs GPU transfer overhead
    }

    return baseLatency + 50; // 50μs CPU processing overhead
}

void DSDProcessor::initializeFilterCoefficients() {
    const uint32_t filterLength = config_.filterOrder;
    state_.filterCoeffs.resize(filterLength);

    // Design a low-pass FIR filter using Kaiser window
    const float cutoffFreq = 0.5f / static_cast<float>(config_.inputSampleRate / config_.outputSampleRate);
    const float beta = 8.0f; // Kaiser window parameter

    for (uint32_t i = 0; i < filterLength; ++i) {
        const int32_t n = static_cast<int32_t>(i) - static_cast<int32_t>(filterLength) / 2;

        if (n == 0) {
            state_.filterCoeffs[i] = 2.0f * cutoffFreq;
        } else {
            state_.filterCoeffs[i] = std::sin(2.0f * M_PI * cutoffFreq * n) / (M_PI * n);
        }

        // Apply Kaiser window (simplified version without Bessel function)
        const float alpha = filterLength / 2.0f;
        const float value = 1.0f - std::pow((static_cast<float>(n) / alpha), 2.0f);
        if (value > 0.0f) {
            // Use Hamming window as approximation for Kaiser window
            const float window = 0.54f - 0.46f * std::cos(2.0f * M_PI * i / (filterLength - 1));
            state_.filterCoeffs[i] *= window;
        } else {
            state_.filterCoeffs[i] = 0.0f;
        }
    }

    // Normalize filter coefficients
    float sum = 0.0f;
    for (float coeff : state_.filterCoeffs) {
        sum += coeff;
    }

    for (float& coeff : state_.filterCoeffs) {
        coeff /= sum;
    }

    Logger::debug("DSD1024 FIR filter initialized ({} taps, cutoff: {:.3f})",
                 filterLength, cutoffFreq);
}

bool DSDProcessor::initializeGPU() {
#ifdef VORTEX_ENABLE_CUDA
    try {
        // Check CUDA availability
        int deviceCount = 0;
        cudaError_t result = cudaGetDeviceCount(&deviceCount);
        if (result != cudaSuccess || deviceCount == 0) {
            Logger::warn("No CUDA devices available");
            return false;
        }

        // Allocate GPU memory buffers
        const size_t inputBufferSize = state_.inputBuffer.size() * sizeof(float);
        const size_t outputBufferSize = state_.outputBuffer.size() * sizeof(float);
        const size_t coeffSize = state_.filterCoeffs.size() * sizeof(float);

        result = cudaMalloc(&state_.gpuInputBuffer, inputBufferSize);
        if (result != cudaSuccess) return false;

        result = cudaMalloc(&state_.gpuOutputBuffer, outputBufferSize);
        if (result != cudaSuccess) {
            cudaFree(state_.gpuInputBuffer);
            return false;
        }

        result = cudaMalloc(&state_.gpuCoefficients, coeffSize);
        if (result != cudaSuccess) {
            cudaFree(state_.gpuInputBuffer);
            cudaFree(state_.gpuOutputBuffer);
            return false;
        }

        result = cudaMalloc(&state_.gpuWorkBuffer, std::max(inputBufferSize, outputBufferSize));
        if (result != cudaSuccess) {
            cudaFree(state_.gpuInputBuffer);
            cudaFree(state_.gpuOutputBuffer);
            cudaFree(state_.gpuCoefficients);
            return false;
        }

        // Copy filter coefficients to GPU
        result = cudaMemcpy(state_.gpuCoefficients, state_.filterCoeffs.data(),
                           coeffSize, cudaMemcpyHostToDevice);
        if (result != cudaSuccess) {
            cleanupGPU();
            return false;
        }

        state_.gpuInitialized = true;
        Logger::info("GPU acceleration initialized successfully");
        return true;

    } catch (const std::exception& e) {
        Logger::error("GPU initialization failed: {}", e.what());
        cleanupGPU();
        return false;
    }
#else
    Logger::warn("CUDA support not compiled in");
    return false;
#endif
}

void DSDProcessor::cleanupGPU() {
#ifdef VORTEX_ENABLE_CUDA
    if (state_.gpuInputBuffer) {
        cudaFree(state_.gpuInputBuffer);
        state_.gpuInputBuffer = nullptr;
    }
    if (state_.gpuOutputBuffer) {
        cudaFree(state_.gpuOutputBuffer);
        state_.gpuOutputBuffer = nullptr;
    }
    if (state_.gpuCoefficients) {
        cudaFree(state_.gpuCoefficients);
        state_.gpuCoefficients = nullptr;
    }
    if (state_.gpuWorkBuffer) {
        cudaFree(state_.gpuWorkBuffer);
        state_.gpuWorkBuffer = nullptr;
    }
#endif
    state_.gpuInitialized = false;
}

uint32_t DSDProcessor::processCPU(const uint8_t* input,
                                 uint32_t inputSamples,
                                 float* output,
                                 uint32_t maxOutputSamples) {
    const uint32_t decimationRatio = config_.inputSampleRate / config_.outputSampleRate;
    const uint32_t maxInputSamples = maxOutputSamples * decimationRatio;
    const uint32_t samplesToProcess = std::min(inputSamples, maxInputSamples);

    // Convert DSD bits to float
    convertDSDBitsToFloat(input, samplesToProcess, state_.inputBuffer.data());

    // Apply FIR filtering
    applyFIRFilter(state_.inputBuffer.data(), state_.outputBuffer.data(), samplesToProcess);

    // Downsample to target sample rate
    return downsample(state_.outputBuffer.data(), samplesToProcess, output, decimationRatio);
}

uint32_t DSDProcessor::processGPU(const uint8_t* input,
                                 uint32_t inputSamples,
                                 float* output,
                                 uint32_t maxOutputSamples) {
#ifdef VORTEX_ENABLE_CUDA
    try {
        // TODO: Implement GPU processing using CUDA kernels
        // For now, fall back to CPU processing
        return processCPU(input, inputSamples, output, maxOutputSamples);

    } catch (const std::exception& e) {
        Logger::error("GPU processing failed: {}", e.what());
        return 0;
    }
#else
    return processCPU(input, inputSamples, output, maxOutputSamples);
#endif
}

void DSDProcessor::applyNoiseShaping(float* data, uint32_t samples) {
    // Simple 1st order noise shaping filter
    static float prevError = 0.0f;

    for (uint32_t i = 0; i < samples; ++i) {
        const float current = data[i];
        const float quantized = std::round(current * 8388607.0f) / 8388607.0f; // 24-bit quantization
        const float error = current - quantized;

        data[i] = quantized + prevError * 0.5f; // 1st order feedback
        prevError = error;
    }
}

void DSDProcessor::applyDynamicRangeCompression(float* data, uint32_t samples) {
    const float thresholdLinear = std::pow(10.0f, config_.threshold / 20.0f);
    const float ratioInv = 1.0f / config_.compressionRatio;

    for (uint32_t i = 0; i < samples; ++i) {
        const float absSample = std::abs(data[i]);

        if (absSample > thresholdLinear) {
            const float aboveThreshold = absSample / thresholdLinear;
            const float compressedAbove = std::pow(aboveThreshold, ratioInv);
            const float compressedSample = thresholdLinear * compressedAbove;

            data[i] = (data[i] < 0.0f) ? -compressedSample : compressedSample;
        }
    }
}

void DSDProcessor::applyDithering(float* data, uint32_t samples) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

    const float ditherAmplitude = 1.0f / (1 << config_.ditherBits);

    for (uint32_t i = 0; i < samples; ++i) {
        data[i] += dis(gen) * ditherAmplitude;
    }
}

void DSDProcessor::convertDSDBitsToFloat(const uint8_t* dsdData, uint32_t numBits, float* output) {
    for (uint32_t bitIndex = 0; bitIndex < numBits; ++bitIndex) {
        const uint32_t byteIndex = bitIndex / 8;
        const uint32_t bitOffset = 7 - (bitIndex % 8);
        const bool dsdBit = (dsdData[byteIndex] >> bitOffset) & 1;

        output[bitIndex] = dsdBit ? 1.0f : -1.0f;
    }
}

void DSDProcessor::applyFIRFilter(const float* input, float* output, uint32_t samples) {
    const uint32_t filterLength = static_cast<uint32_t>(state_.filterCoeffs.size());

    for (uint32_t i = 0; i < samples; ++i) {
        float sum = 0.0f;

        for (uint32_t j = 0; j < filterLength; ++j) {
            const int32_t index = static_cast<int32_t>(i) - static_cast<int32_t>(j);

            if (index >= 0 && index < static_cast<int32_t>(samples)) {
                sum += input[index] * state_.filterCoeffs[j];
            } else if (index < 0) {
                // Use delay line for boundary conditions
                const uint32_t delayIdx = (state_.delayIndex + index + filterLength) % filterLength;
                sum += state_.delayLine[delayIdx] * state_.filterCoeffs[j];
            }
        }

        output[i] = sum;

        // Update delay line
        state_.delayLine[state_.delayIndex] = input[i];
        state_.delayIndex = (state_.delayIndex + 1) % filterLength;
    }
}

uint32_t DSDProcessor::downsample(const float* input,
                                 uint32_t inputSamples,
                                 float* output,
                                 uint32_t decimationRatio) {
    const uint32_t outputSamples = inputSamples / decimationRatio;

    for (uint32_t i = 0; i < outputSamples; ++i) {
        const uint32_t inputIndex = i * decimationRatio;

        // Simple averaging decimation
        float sum = 0.0f;
        for (uint32_t j = 0; j < decimationRatio; ++j) {
            if (inputIndex + j < inputSamples) {
                sum += input[inputIndex + j];
            }
        }

        output[i] = sum / static_cast<float>(decimationRatio);
    }

    return outputSamples;
}

void DSDProcessor::logPerformance(uint64_t processingTime, uint32_t samples) const {
    const double latency = static_cast<double>(processingTime) / samples;

    // Log performance warnings
    if (latency > 100.0) { // >100μs per sample
        Logger::warn("High DSD processing latency detected: {:.3f}μs per sample", latency);
    } else if (latency > 50.0) { // >50μs per sample
        Logger::debug("Elevated DSD processing latency: {:.3f}μs per sample", latency);
    }

    // Log performance info periodically
    static uint32_t logCounter = 0;
    if (++logCounter >= 10000) { // Log every 10,000 calls
        Logger::info("DSD processing performance: {:.3f}μs per sample, GPU: {}",
                     latency, isGPUAccelerated() ? "enabled" : "disabled");
        logCounter = 0;
    }
}

} // namespace vortex::core::dsp