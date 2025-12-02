#include "core/dsp/waveform_processor.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>

#ifdef VORTEX_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace vortex::core::dsp {

WaveformProcessor::WaveformProcessor()
    : initialized_(false)
    , ringBufferSize_(0)
    , ringBufferWritePos_(0)
    , ringBufferReadPos_(0)
    , gpuInputBuffer_(nullptr)
    , gpuOutputBuffer_(nullptr)
    , gpuInitialized_(false)
    , totalFramesProcessed_(0)
    , totalProcessingTime_(0.0)
    , gpuFramesProcessed_(0)
    , cpuFramesProcessed_(0) {
    ringBufferReadPos_.store(0);
}

WaveformProcessor::~WaveformProcessor() {
    shutdown();
}

bool WaveformProcessor::initialize(const Config& config) {
    if (initialized_) {
        Logger::warn("WaveformProcessor already initialized");
        return true;
    }

    // Validate configuration
    if (config.sampleRate <= 0.0f || config.bufferSize == 0 ||
        config.channels <= 0 || config.waveformLength == 0) {
        Logger::error("Invalid waveform processor configuration");
        return false;
    }

    config_ = config;

    // Calculate required ring buffer size based on window duration
    size_t windowSamples = static_cast<size_t>(config_.windowDuration * config_.sampleRate);
    ringBufferSize_ = windowSamples * config_.channels;

    Logger::info("Initializing WaveformProcessor: {} samples window, {} output samples, {} channels",
                 ringBufferSize_ / config_.channels, config_.waveformLength, config_.channels);

    try {
        // Initialize ring buffer
        if (!initializeRingBuffer()) {
            Logger::error("Failed to initialize ring buffer");
            return false;
        }

        // Initialize waveform buffers
        if (!initializeWaveformBuffers()) {
            Logger::error("Failed to initialize waveform buffers");
            return false;
        }

        // Initialize peak tracking
        if (!initializePeakTracking()) {
            Logger::error("Failed to initialize peak tracking");
            return false;
        }

        // Initialize smoothing filters
        if (!initializeSmoothingFilters()) {
            Logger::error("Failed to initialize smoothing filters");
            return false;
        }

        // Initialize GPU resources if requested
        if (config_.processingMode == ProcessingMode::GPU || config_.processingMode == ProcessingMode::Auto) {
            if (!initializeGPUResources()) {
                if (config_.processingMode == ProcessingMode::GPU) {
                    Logger::error("GPU initialization failed but GPU mode was requested");
                    return false;
                } else {
                    Logger::warn("GPU initialization failed, falling back to CPU processing");
                    config_.processingMode = ProcessingMode::CPU;
                }
            }
        }

        initialized_ = true;
        Logger::info("WaveformProcessor initialized successfully ({} processing)",
                     config_.processingMode == ProcessingMode::GPU ? "GPU" : "CPU");
        return true;

    } catch (const std::exception& e) {
        Logger::error("Exception during waveform processor initialization: {}", e.what());
        return false;
    }
}

bool WaveformProcessor::initialize(float sampleRate, size_t bufferSize, int channels) {
    Config config;
    config.sampleRate = sampleRate;
    config.bufferSize = bufferSize;
    config.channels = channels;
    config.waveformLength = 512;
    config.windowDuration = 0.1f;
    return initialize(config);
}

void WaveformProcessor::shutdown() {
    if (!initialized_) {
        return;
    }

    Logger::info("Shutting down WaveformProcessor");

    // Cleanup GPU resources
    cleanupGPUResources();

    // Clear buffers
    ringBuffer_.clear();
    waveformBuffer_.clear();
    peakBuffer_.clear();
    rmsBuffer_.clear();
    smoothedBuffer_.clear();
    currentPeaks_.clear();
    peakHoldCounters_.clear();
    maxAmplitudes_.clear();
    smoothedValues_.clear();

    initialized_ = false;

    // Log final statistics
    if (totalFramesProcessed_ > 0) {
        double avgProcessingTime = totalProcessingTime_ / totalFramesProcessed_;
        Logger::info("WaveformProcessor stats: {} frames processed, avg time: {:.3f}μs, GPU frames: {}, CPU frames: {}",
                     totalFramesProcessed_.load(), avgProcessingTime,
                     gpuFramesProcessed_.load(), cpuFramesProcessed_.load());
    }
}

std::vector<WaveformProcessor::WaveformData> WaveformProcessor::processAudio(const float* audioData, size_t numSamples) {
    std::vector<WaveformData> result;

    if (!processAudio(audioData, numSamples, result)) {
        result.resize(config_.channels);
        for (int ch = 0; ch < config_.channels; ++ch) {
            result[ch].samples.resize(config_.waveformLength, 0.0f);
            result[ch].peaks.resize(config_.waveformLength, 0.0f);
            result[ch].rms.resize(config_.waveformLength, 0.0f);
            result[ch].isValid = false;
        }
    }

    return result;
}

bool WaveformProcessor::processAudio(const float* audioData, size_t numSamples,
                                     std::vector<WaveformData>& outputWaveform) {
    if (!initialized_) {
        Logger::error("WaveformProcessor not initialized");
        return false;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    outputWaveform.resize(config_.channels);

    bool success = false;

    try {
        if (config_.processingMode == ProcessingMode::GPU && gpuInitialized_) {
            success = processWaveformGPU(audioData, numSamples, outputWaveform);
        } else {
            success = processWaveformCPU(audioData, numSamples, outputWaveform);
        }

    } catch (const std::exception& e) {
        Logger::error("Exception during waveform processing: {}", e.what());
        success = false;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto processingTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

    updatePerformanceStats(static_cast<double>(processingTime), config_.processingMode == ProcessingMode::GPU && gpuInitialized_);

    return success;
}

std::vector<WaveformProcessor::WaveformData> WaveformProcessor::getCurrentWaveform() {
    std::vector<WaveformData> result;
    result.resize(config_.channels);

    if (!initialized_) {
        return result;
    }

    for (int ch = 0; ch < config_.channels; ++ch) {
        WaveformData& data = result[ch];
        data.samples.resize(config_.waveformLength);
        data.peaks.resize(config_.waveformLength);
        data.rms.resize(config_.waveformLength);
        data.isValid = true;
        data.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

        // Copy current waveform data from output buffer
        for (size_t i = 0; i < config_.waveformLength; ++i) {
            size_t bufferIndex = ch * config_.waveformLength + i;
            if (bufferIndex < waveformBuffer_.size()) {
                data.samples[i] = waveformBuffer_[bufferIndex];
                data.peaks[i] = peakBuffer_[bufferIndex];
                data.rms[i] = rmsBuffer_[bufferIndex];
            } else {
                data.samples[i] = 0.0f;
                data.peaks[i] = 0.0f;
                data.rms[i] = 0.0f;
            }
        }

        data.maxAmplitude = maxAmplitudes_[ch];
        data.minAmplitude = 0.0f; // Could track minimum if needed
    }

    return result;
}

bool WaveformProcessor::processWaveformCPU(const float* audioData, size_t numSamples,
                                              std::vector<WaveformData>& outputWaveform) {
    // Add new audio to ring buffer
    addToRingBuffer(audioData, numSamples);

    // Extract audio for waveform analysis
    std::vector<float> analysisBuffer;
    size_t windowSamples = static_cast<size_t>(config_.windowDuration * config_.sampleRate) * config_.channels;

    if (!extractFromRingBuffer(analysisBuffer, windowSamples)) {
        // Not enough data yet
        return false;
    }

    // Process each channel
    for (int ch = 0; ch < config_.channels; ++ch) {
        WaveformData& waveformData = outputWaveform[ch];
        waveformData.samples.resize(config_.waveformLength);
        waveformData.peaks.resize(config_.waveformLength);
        waveformData.rms.resize(config_.waveformLength);
        waveformData.isValid = true;
        waveformData.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

        // Deinterleave channel data
        std::vector<float> channelData(windowSamples / config_.channels);
        for (size_t i = 0; i < channelData.size(); ++i) {
            channelData[i] = analysisBuffer[i * config_.channels + ch];
        }

        // Calculate waveform based on display mode
        switch (config_.displayMode) {
            case DisplayMode::Peaks:
                calculatePeaks(channelData.data(), channelData.size(), waveformData.samples.data());
                break;

            case DisplayMode::RMS:
                calculateRMS(channelData.data(), channelData.size(), waveformData.samples.data());
                break;

            case DisplayMode::Average:
                // Simple moving average
                for (size_t i = 0; i < config_.waveformLength; ++i) {
                    size_t startIdx = i * channelData.size() / config_.waveformLength;
                    size_t endIdx = (i + 1) * channelData.size() / config_.waveformLength;
                    endIdx = std::min(endIdx, channelData.size());

                    float sum = 0.0f;
                    for (size_t j = startIdx; j < endIdx; ++j) {
                        sum += std::abs(channelData[j]);
                    }
                    waveformData.samples[i] = (endIdx > startIdx) ? sum / (endIdx - startIdx) : 0.0f;
                }
                break;

            case DisplayMode::Instantaneous:
                // Downsample to target length
                downsampleWaveform(channelData.data(), channelData.size(),
                                  waveformData.samples.data(), config_.waveformLength);
                break;

            case DisplayMode::Envelope:
                // Simplified envelope follower
                {
                    float attack = 0.01f;
                    float release = 0.1f;
                    float envelope = 0.0f;

                    for (size_t i = 0; i < config_.waveformLength; ++i) {
                        size_t inputIdx = i * channelData.size() / config_.waveformLength;
                        if (inputIdx < channelData.size()) {
                            float input = std::abs(channelData[inputIdx]);
                            if (input > envelope) {
                                envelope = envelope + (input - envelope) * attack;
                            } else {
                                envelope = envelope + (input - envelope) * release;
                            }
                            waveformData.samples[i] = envelope;
                        }
                    }
                }
                break;
        }

        // Calculate peaks and RMS if enabled
        if (config_.enablePeakDetection) {
            calculatePeaks(channelData.data(), channelData.size(), waveformData.peaks.data());
        } else {
            std::fill(waveformData.peaks.begin(), waveformData.peaks.end(), 0.0f);
        }

        if (config_.enableRMS) {
            calculateRMS(channelData.data(), channelData.size(), waveformData.rms.data());
        } else {
            std::fill(waveformData.rms.begin(), waveformData.rms.end(), 0.0f);
        }

        // Apply smoothing if enabled
        if (config_.enableSmoothing) {
            applySmoothing(waveformData.samples.data(), smoothedBuffer_.data(),
                           config_.waveformLength);
            std::copy(smoothedBuffer_.data(), smoothedBuffer_.data() + config_.waveformLength,
                      waveformData.samples.data());
        }

        // Apply compression if enabled
        if (config_.enableCompression) {
            compressWaveform(waveformData.samples.data(), config_.waveformLength);
        }

        // Normalize if enabled
        if (config_.normalizeOutput) {
            normalizeWaveform(waveformData.samples.data(), config_.waveformLength);
        }

        // Apply time scaling
        if (config_.timeScale == TimeScale::Logarithmic) {
            std::vector<float> tempBuffer(config_.waveformLength);
            std::copy(waveformData.samples.data(),
                      waveformData.samples.data() + config_.waveformLength,
                      tempBuffer.data());
            applyLogarithmicTimeScale(tempBuffer.data(), config_.waveformLength,
                                       waveformData.samples.data(), config_.waveformLength);
        }

        // Update output buffers
        for (size_t i = 0; i < config_.waveformLength; ++i) {
            size_t bufferIndex = ch * config_.waveformLength + i;
            if (bufferIndex < waveformBuffer_.size()) {
                waveformBuffer_[bufferIndex] = waveformData.samples[i];
                peakBuffer_[bufferIndex] = waveformData.peaks[i];
                rmsBuffer_[bufferIndex] = waveformData.rms[i];
            }
        }

        // Track amplitude
        auto maxIt = std::max_element(waveformData.samples.begin(), waveformData.samples.end());
        if (maxIt != waveformData.samples.end()) {
            waveformData.maxAmplitude = *maxIt;
            maxAmplitudes_[ch] = std::max(maxAmplitudes_[ch], *maxIt);
        } else {
            waveformData.maxAmplitude = 0.0f;
        }

        waveformData.minAmplitude = 0.0f; // Could track minimum if needed
    }

    return true;
}

bool WaveformProcessor::processWaveformGPU(const float* audioData, size_t numSamples,
                                             std::vector<WaveformData>& outputWaveform) {
#ifdef VORTEX_ENABLE_CUDA
    if (!gpuInitialized_) {
        return processWaveformCPU(audioData, numSamples, outputWaveform);
    }

    // GPU implementation would go here
    // For now, fallback to CPU
    return processWaveformCPU(audioData, numSamples, outputWaveform);
#else
    return processWaveformCPU(audioData, numSamples, outputWaveform);
#endif
}

bool WaveformProcessor::initializeRingBuffer() {
    ringBuffer_.resize(ringBufferSize_);
    std::fill(ringBuffer_.begin(), ringBuffer_.end(), 0.0f);
    ringBufferWritePos_ = 0;
    return true;
}

bool WaveformProcessor::initializeWaveformBuffers() {
    size_t totalWaveformSize = config_.waveformLength * config_.channels;
    waveformBuffer_.resize(totalWaveformSize);
    peakBuffer_.resize(totalWaveformSize);
    rmsBuffer_.resize(totalWaveformSize);
    smoothedBuffer_.resize(config_.waveformLength);

    std::fill(waveformBuffer_.begin(), waveformBuffer_.end(), 0.0f);
    std::fill(peakBuffer_.begin(), peakBuffer_.end(), 0.0f);
    std::fill(rmsBuffer_.begin(), rmsBuffer_.end(), 0.0f);
    std::fill(smoothedBuffer_.begin(), smoothedBuffer_.end(), 0.0f);

    return true;
}

bool WaveformProcessor::initializePeakTracking() {
    currentPeaks_.resize(config_.channels);
    peakHoldCounters_.resize(config_.channels);
    maxAmplitudes_.resize(config_.channels);

    std::fill(currentPeaks_.begin(), currentPeaks_.end(), 0.0f);
    std::fill(peakHoldCounters_.begin(), peakHoldCounters_.end(), 0);
    std::fill(maxAmplitudes_.begin(), maxAmplitudes_.end(), 0.0f);

    return true;
}

bool WaveformProcessor::initializeSmoothingFilters() {
    smoothedValues_.resize(config_.waveformLength);
    std::fill(smoothedValues_.begin(), smoothedValues_.end(), 0.0f);

    return true;
}

bool WaveformProcessor::initializeGPUResources() {
#ifdef VORTEX_ENABLE_CUDA
    try {
        // Allocate GPU memory buffers
        cudaError_t result;

        size_t inputSize = config_.bufferSize * config_.channels * sizeof(float);
        size_t outputSize = config_.waveformLength * config_.channels * sizeof(float);

        result = cudaMalloc(&gpuInputBuffer_, inputSize);
        if (result != CUDA_SUCCESS) {
            Logger::error("Failed to allocate GPU input buffer: {}", result);
            return false;
        }

        result = cudaMalloc(&gpuOutputBuffer_, outputSize);
        if (result != CUDA_SUCCESS) {
            Logger::error("Failed to allocate GPU output buffer: {}", result);
            cleanupGPUResources();
            return false;
        }

        gpuInitialized_ = true;
        Logger::info("GPU resources initialized successfully");
        return true;

    } catch (const std::exception& e) {
        Logger::error("Exception during GPU initialization: {}", e.what());
        return false;
    }
#else
    Logger::warn("GPU support not compiled in");
    return false;
#endif
}

void WaveformProcessor::cleanupGPUResources() {
#ifdef VORTEX_ENABLE_CUDA
    if (gpuInputBuffer_) {
        cudaFree(gpuInputBuffer_);
        gpuInputBuffer_ = nullptr;
    }

    if (gpuOutputBuffer_) {
        cudaFree(gpuOutputBuffer_);
        gpuOutputBuffer_ = nullptr;
    }
#endif

    gpuInitialized_ = false;
}

void WaveformProcessor::addToRingBuffer(const float* audioData, size_t numSamples) {
    for (size_t i = 0; i < numSamples * config_.channels; ++i) {
        ringBuffer_[ringBufferWritePos_] = audioData[i];
        ringBufferWritePos_ = (ringBufferWritePos_ + 1) % ringBufferSize_;
    }
}

bool WaveformProcessor::extractFromRingBuffer(std::vector<float>& output, size_t numSamples) {
    size_t readPos = ringBufferReadPos_.load();

    if (ringBufferWritePos_ >= readPos) {
        size_t available = ringBufferWritePos_ - readPos;
        if (available < numSamples) {
            return false; // Not enough data
        }
    } else {
        // Buffer wrapped around
        size_t available1 = ringBufferSize_ - readPos;
        size_t available2 = ringBufferWritePos_;
        if (available1 + available2 < numSamples) {
            return false; // Not enough data
        }
    }

    output.resize(numSamples);

    for (size_t i = 0; i < numSamples; ++i) {
        output[i] = ringBuffer_[(readPos + i) % ringBufferSize_];
    }

    ringBufferReadPos_.store((readPos + numSamples) % ringBufferSize_);
    return true;
}

void WaveformProcessor::calculatePeaks(const float* audioData, size_t numSamples, float* peaks) {
    const size_t blockSize = numSamples / config_.waveformLength;
    uint32_t peakHoldSamples = static_cast<uint32_t>(config_.peakHoldTime * config_.sampleRate);

    for (size_t i = 0; i < config_.waveformLength; ++i) {
        size_t startIdx = i * blockSize;
        size_t endIdx = std::min(startIdx + blockSize, numSamples);

        float maxVal = 0.0f;
        for (size_t j = startIdx; j < endIdx; ++j) {
            float absVal = std::abs(audioData[j]);
            if (absVal > maxVal) {
                maxVal = absVal;
            }
        }

        // Peak hold logic
        if (maxVal > currentPeaks_[0]) { // Simplified - would track per channel
            currentPeaks_[0] = maxVal;
            peakHoldCounters_[0] = 0;
        } else {
            peakHoldCounters_[0]++;
            if (peakHoldCounters_[0] > peakHoldSamples) {
                currentPeaks_[0] *= config_.decayRate;
            }
        }

        peaks[i] = currentPeaks_[0];
    }
}

void WaveformProcessor::calculateRMS(const float* audioData, size_t numSamples, float* rms) {
    const size_t blockSize = numSamples / config_.waveformLength;

    for (size_t i = 0; i < config_.waveformLength; ++i) {
        size_t startIdx = i * blockSize;
        size_t endIdx = std::min(startIdx + blockSize, numSamples);

        float sum = 0.0f;
        for (size_t j = startIdx; j < endIdx; ++j) {
            sum += audioData[j] * audioData[j];
        }

        size_t count = endIdx - startIdx;
        rms[i] = count > 0 ? std::sqrt(sum / count) : 0.0f;
    }
}

void WaveformProcessor::detectZeroCrossings(const float* audioData, size_t numSamples, std::vector<bool>& crossings) {
    const size_t blockSize = numSamples / config_.waveformLength;
    crossings.resize(config_.waveformLength);

    for (size_t i = 0; i < config_.waveformLength; ++i) {
        size_t startIdx = i * blockSize;
        size_t endIdx = std::min(startIdx + blockSize, numSamples);

        if (startIdx < numSamples && endIdx < numSamples) {
            int crossingCount = 0;
            bool wasPositive = audioData[startIdx] >= 0.0f;

            for (size_t j = startIdx + 1; j < endIdx; ++j) {
                bool isPositive = audioData[j] >= 0.0f;
                if (wasPositive != isPositive) {
                    crossingCount++;
                    wasPositive = isPositive;
                }
            }

            crossings[i] = (crossingCount > 0);
        } else {
            crossings[i] = false;
        }
    }
}

void WaveformProcessor::applySmoothing(float* input, float* output, size_t length) {
    float alpha = config_.smoothingFactor;

    for (size_t i = 0; i < length; ++i) {
        smoothedValues_[i] = smoothedValues_[i] + alpha * (input[i] - smoothedValues_[i]);
        output[i] = smoothedValues_[i];
    }
}

void WaveformProcessor::downsampleWaveform(const float* input, size_t inputLength,
                                            float* output, size_t outputLength) {
    float ratio = static_cast<float>(inputLength) / outputLength;

    for (size_t i = 0; i < outputLength; ++i) {
        size_t inputIndex = static_cast<size_t>(i * ratio);
        output[i] = (inputIndex < inputLength) ? input[inputIndex] : 0.0f;
    }
}

void WaveformProcessor::compressWaveform(float* data, size_t length) {
    float ratio = config_.compressionRatio;

    for (size_t i = 0; i < length; ++i) {
        if (data[i] > 0.0f) {
            data[i] = std::log10(data[i] + 1.0f) / std::log10(ratio + 1.0f) * ratio;
        }
    }
}

void WaveformProcessor::normalizeWaveform(float* data, size_t length) {
    float maxVal = *std::max_element(data, data + length);

    if (maxVal > 0.0f) {
        float scale = 1.0f / maxVal;
        for (size_t i = 0; i < length; ++i) {
            data[i] *= scale;
        }
    }
}

void WaveformProcessor::applyLinearTimeScale(const float* input, size_t inputLength,
                                             float* output, size_t outputLength) {
    downsampleWaveform(input, inputLength, output, outputLength);
}

void WaveformProcessor::applyLogarithmicTimeScale(const float* input, size_t inputLength,
                                                  float* output, size_t outputLength) {
    for (size_t i = 0; i < outputLength; ++i) {
        float logPos = std::pow(static_cast<float>(i) / (outputLength - 1), 2.0f);
        size_t inputIndex = static_cast<size_t>(logPos * (inputLength - 1));
        output[i] = (inputIndex < inputLength) ? input[inputIndex] : 0.0f;
    }
}

void WaveformProcessor::updatePerformanceStats(double processingTimeMs, bool usedGPU) const {
    totalFramesProcessed_++;
    totalProcessingTime_ += processingTimeMs;

    if (usedGPU) {
        gpuFramesProcessed_++;
    } else {
        cpuFramesProcessed_++;
    }
}

// Setter and getter methods
void WaveformProcessor::setDisplayMode(DisplayMode mode) {
    config_.displayMode = mode;
}

WaveformProcessor::DisplayMode WaveformProcessor::getDisplayMode() const {
    return config_.displayMode;
}

void WaveformProcessor::setTimeScale(TimeScale scale) {
    config_.timeScale = scale;
}

WaveformProcessor::TimeScale WaveformProcessor::getTimeScale() const {
    return config_.timeScale;
}

void WaveformProcessor::setProcessingMode(ProcessingMode mode) {
    config_.processingMode = mode;
}

WaveformProcessor::ProcessingMode WaveformProcessor::getProcessingMode() const {
    return config_.processingMode;
}

bool WaveformProcessor::isGPUAvailable() const {
    return gpuInitialized_;
}

const WaveformProcessor::Config& WaveformProcessor::getConfig() const {
    return config_;
}

std::string WaveformProcessor::getPerformanceStats() const {
    double avgProcessingTime = totalFramesProcessed_ > 0 ?
                             totalProcessingTime_ / totalFramesProcessed_ : 0.0;

    char buffer[512];
    snprintf(buffer, sizeof(buffer),
        "{"
        "\"total_frames_processed\":%llu,"
        "\"average_processing_time_us\":%.3f,"
        "\"gpu_frames_processed\":%llu,"
        "\"cpu_frames_processed\":%llu,"
        "\"gpu_utilization_percent\":%.2f,"
        "\"processing_mode\":\"%s\""
        "}",
        static_cast<unsigned long long>(totalFramesProcessed_.load()),
        avgProcessingTime,
        static_cast<unsigned long long>(gpuFramesProcessed_.load()),
        static_cast<unsigned long long>(cpuFramesProcessed_.load()),
        totalFramesProcessed_ > 0 ?
            (static_cast<double>(gpuFramesProcessed_.load()) / totalFramesProcessed_ * 100.0) : 0.0,
        config_.processingMode == ProcessingMode::GPU ? "GPU" : "CPU"
    );

    return std::string(buffer);
}

void WaveformProcessor::reset() {
    if (!initialized_) {
        return;
    }

    // Reset ring buffer
    std::fill(ringBuffer_.begin(), ringBuffer_.end(), 0.0f);
    ringBufferWritePos_ = 0;
    ringBufferReadPos_.store(0);

    // Reset waveform buffers
    std::fill(waveformBuffer_.begin(), waveformBuffer_.end(), 0.0f);
    std::fill(peakBuffer_.begin(), peakBuffer_.end(), 0.0f);
    std::fill(rmsBuffer_.begin(), rmsBuffer_.end(), 0.0f);

    // Reset peak tracking
    std::fill(currentPeaks_.begin(), currentPeaks_.end(), 0.0f);
    std::fill(peakHoldCounters_.begin(), peakHoldCounters_.end(), 0);
    std::fill(maxAmplitudes_.begin(), maxAmplitudes_.end(), 0.0f);

    // Reset smoothing filters
    std::fill(smoothedValues_.begin(), smoothedValues_.end(), 0.0f);

    // Reset performance counters
    totalFramesProcessed_ = 0;
    totalProcessingTime_ = 0.0;
    gpuFramesProcessed_ = 0;
    cpuFramesProcessed_ = 0;

    Logger::debug("WaveformProcessor reset");
}

bool WaveformProcessor::isInitialized() const {
    return initialized_;
}

void WaveformProcessor::setWaveformLength(size_t length) {
    config_.waveformLength = length;
    if (initialized_) {
        initializeWaveformBuffers();
        initializeSmoothingFilters();
    }
}

size_t WaveformProcessor::getWaveformLength() const {
    return config_.waveformLength;
}

void WaveformProcessor::setPeakHoldTime(float holdTime) {
    config_.peakHoldTime = holdTime;
}

float WaveformProcessor::getPeakHoldTime() const {
    return config_.peakHoldTime;
}

void WaveformProcessor::setPeakDetectionEnabled(bool enabled) {
    config_.enablePeakDetection = enabled;
}

bool WaveformProcessor::isPeakDetectionEnabled() const {
    return config_.enablePeakDetection;
}

} // namespace vortex::core::dsp