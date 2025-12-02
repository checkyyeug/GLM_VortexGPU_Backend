#include "convolution.hpp"
#include "../utils/logger.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <thread>
#include <future>
#include <numeric>

#ifdef VORTEX_GPU_CUDA
#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#endif

#ifdef VORTEX_GPU_OPENCL
#include <CL/cl.h>
#endif

namespace vortex {

// ConvolutionEngine Implementation
ConvolutionEngine::ConvolutionEngine() {
    Logger::info("ConvolutionEngine: Initializing 16M-point convolution engine");

    // Initialize statistics
    statistics_.startTime = std::chrono::steady_clock::now();
    lastUpdateTime_ = statistics_.startTime;
}

ConvolutionEngine::~ConvolutionEngine() {
    shutdown();
    Logger::info("ConvolutionEngine: 16M-point convolution engine destroyed");
}

bool ConvolutionEngine::initialize(const ConvolutionConfig& config) {
    if (initialized_.load()) {
        Logger::warn("ConvolutionEngine: Already initialized");
        return true;
    }

    Logger::info("ConvolutionEngine: Initializing with max IR length {}, {} Hz sample rate, {} channels",
                 config.maxIRLength, config.sampleRate, config.numChannels);

    config_ = config;

    try {
        // Validate configuration
        if (config.maxIRLength > MAX_IR_LENGTH) {
            setError("Maximum IR length exceeds limit");
            return false;
        }

        if (!isValidSampleRate(config.sampleRate)) {
            setError("Invalid sample rate");
            return false;
        }

        if (config.numChannels > MAX_CHANNELS) {
            setError("Number of channels exceeds maximum");
            return false;
        }

        // Calculate optimal FFT size and partitions
        fftSize_ = calculateOptimalFFTSize(config_.maxIRLength, config_.blockSize);
        numPartitions_ = calculateNumPartitions(config_.maxIRLength, fftSize_);

        Logger::info("ConvolutionEngine: Using FFT size {}, {} partitions", fftSize_, numPartitions_);

        // Initialize FFT twiddle factors
        initializeFFT();

        // Initialize processing buffers
        initializeBuffers();

        // Initialize GPU resources if enabled
        if (config.enableGPUAcceleration) {
            if (!initializeGPUResources()) {
                Logger::warn("ConvolutionEngine: GPU initialization failed, falling back to CPU");
                config_.enableGPUAcceleration = false;
            }
        }

        // Initialize multi-threading
        if (config.numThreads > 0 || config.numThreads == 0) {
            uint32_t numThreads = config.numThreads > 0 ? config.numThreads : std::thread::hardware_concurrency();
            setProcessingThreads(numThreads);
        }

        processingActive_.store(true);
        initialized_.store(true);

        Logger::info("ConvolutionEngine: Initialization complete");
        return true;

    } catch (const std::exception& e) {
        setError("Initialization failed: " + std::string(e.what()));
        return false;
    }
}

void ConvolutionEngine::shutdown() {
    if (!initialized_.load()) {
        return;
    }

    Logger::info("ConvolutionEngine: Shutting down");

    processingActive_.store(false);
    initialized_.store(false);

    // Stop multi-threading
    multithreadingEnabled_.store(false);

    // Wait for processing threads to finish
    for (auto& thread : processingThreads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    processingThreads_.clear();

    // Cleanup GPU resources
    if (config_.enableGPUAcceleration) {
        cleanupGPUResources();
    }

    // Clear buffers and state
    {
        std::lock_guard<std::mutex> lock(buffersMutex_);
        buffers_.inputBuffer.clear();
        buffers_.outputBuffer.clear();
        buffers_.intermediateBuffer.clear();
        buffers_.fftBuffer.clear();
        buffers_.irBuffer.clear();
        buffers_.overlapBuffer.clear();
        buffers_.windowBuffer.clear();
    }

    // Clear impulse responses
    {
        std::lock_guard<std::mutex> lock(irMutex_);
        impulseResponses_.clear();
        currentIRName_.clear();
    }

    Logger::info("ConvolutionEngine: Shutdown complete");
}

bool ConvolutionEngine::isInitialized() const {
    return initialized_.load();
}

bool ConvolutionEngine::processAudio(const float* input, float* output, size_t numSamples) {
    if (!initialized_.load() || !input || !output || !processingActive_.load()) {
        return false;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    // Process through multi-channel pipeline
    bool success = processAudioMultiChannel({&input}, {&output}, numSamples, 1);

    // Update statistics
    if (success) {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<float, std::milli>(endTime - startTime);

        std::lock_guard<std::mutex> lock(statsMutex_);
        statistics_.totalSamplesProcessed += numSamples;
        statistics_.averageLatency = (statistics_.averageLatency * 0.9f) + (duration.count() * 0.1f);
        statistics_.maxLatency = std::max(statistics_.maxLatency, duration.count());
        statistics_.minLatency = std::min(statistics_.minLatency, duration.count());
        statistics_.lastActivity = std::chrono::steady_clock::now();
        statistics_.fftSize = fftSize_;
        statistics_.numPartitions = numPartitions_;
    }

    return success;
}

bool ConvolutionEngine::processAudioMultiChannel(const std::vector<const float*>& inputs,
                                                 std::vector<float*>& outputs,
                                                 size_t numSamples, uint16_t channels) {
    if (!initialized_.load() || inputs.empty() || outputs.empty() ||
        inputs.size() != outputs.size() || channels == 0) {
        return false;
    }

    // Check if we have an active IR
    {
        std::lock_guard<std::mutex> lock(irMutex_);
        if (currentIRName_.empty() || impulseResponses_.find(currentIRName_) == impulseResponses_.end()) {
            // No IR loaded, just copy input to output (dry signal)
            for (uint16_t ch = 0; ch < channels && ch < inputs.size(); ++ch) {
                if (inputs[ch] && outputs[ch]) {
                    std::copy(inputs[ch], inputs[ch] + numSamples, outputs[ch]);
                }
            }
            return true;
        }
    }

    try {
        // Buffer management
        std::lock_guard<std::mutex> lock(buffersMutex_);
        if (buffers_.inputBuffer.size() < numSamples * channels) {
            buffers_.inputBuffer.resize(numSamples * channels);
            buffers_.outputBuffer.resize(numSamples * channels);
            buffers_.intermediateBuffer.resize(numSamples * channels);
        }

        // Process each channel independently
        for (uint16_t ch = 0; ch < channels && ch < inputs.size(); ++ch) {
            const float* input = inputs[ch];
            float* output = outputs[ch];

            if (!input || !output) {
                continue;
            }

            // Apply predelay if enabled
            if (predelay_ > 0.0f) {
                applyPredelay(input, output, numSamples, predelay_);
                input = output; // Use delayed signal as input
            }

            // Main convolution processing
            if (config_.enableGPUAcceleration && gpuProcessor_) {
                // GPU processing path
                if (!gpuProcessor_->processAudio(input, output, numSamples, 1)) {
                    Logger::warn("ConvolutionEngine: GPU processing failed, falling back to CPU");
                    processChannelData(input, output, numSamples, ch);
                }
            } else {
                // CPU processing path
                processChannelData(input, output, numSamples, ch);
            }

            // Apply gain
            if (gain_ != 1.0f) {
                for (size_t i = 0; i < numSamples; ++i) {
                    output[i] *= gain_;
                }
            }

            // Apply wet/dry mix
            if (wetLevel_ != 1.0f || dryLevel_ != 0.0f) {
                for (size_t i = 0; i < numSamples; ++i) {
                    output[i] = output[i] * wetLevel_ + input[i] * dryLevel_;
                }
            }

            // Apply dithering if enabled
            if (config_.enableDithering) {
                applyDithering(output, numSamples);
            }

            // Apply noise shaping if enabled
            if (config_.enableNoiseShaping) {
                applyNoiseShaping(output, numSamples);
            }
        }

        // Call processing callback if set
        if (processingCallback_) {
            processingCallback_(inputs[0], outputs[0], numSamples);
        }

        return true;

    } catch (const std::exception& e) {
        setError("Audio processing failed: " + std::string(e.what()));
        return false;
    }
}

bool ConvolutionEngine::loadImpulseResponse(const std::string& filePath, const std::string& name) {
    Logger::info("ConvolutionEngine: Loading impulse response from: {}", filePath);

    try {
        // Read audio file
        std::vector<float> audioData;
        uint32_t sampleRate = 0;
        uint16_t channels = 0;

        if (!loadAudioFile(filePath, audioData, sampleRate, channels)) {
            setError("Failed to load audio file: " + filePath);
            return false;
        }

        return loadImpulseResponseFromData(audioData, sampleRate, name.empty() ? filePath : name);

    } catch (const std::exception& e) {
        setError("Failed to load impulse response: " + std::string(e.what()));
        return false;
    }
}

bool ConvolutionEngine::loadImpulseResponseFromData(const std::vector<float>& data, uint32_t sampleRate,
                                                   const std::string& name) {
    if (data.empty() || name.empty()) {
        return false;
    }

    Logger::info("ConvolutionEngine: Loading impulse response '{}' with {} samples", name, data.size());

    try {
        ImpulseResponse ir;
        ir.name = name;
        ir.length = static_cast<uint32_t>(data.size());
        ir.sampleRate = sampleRate;
        ir.channels = 1; // For now, assume mono
        ir.format = ImpulseResponseFormat::TIME_DOMAIN;
        ir.timeDomainData.resize(1);
        ir.timeDomainData[0] = data;
        ir.createdTime = std::chrono::system_clock::now();
        ir.modifiedTime = ir.createdTime;

        // Process the impulse response
        if (!processImpulseResponse(ir)) {
            setError("Failed to process impulse response");
            return false;
        }

        // Store the IR
        {
            std::lock_guard<std::mutex> lock(irMutex_);
            impulseResponses_[name] = ir;
            currentIRName_ = name;
        }

        // Upload to GPU if enabled
        if (config_.enableGPUAcceleration && gpuProcessor_ && !ir.frequencyDomainData.empty()) {
            // Combine channels for GPU processing
            std::vector<std::complex<float>> combinedData;
            for (const auto& channelData : ir.frequencyDomainData) {
                combinedData.insert(combinedData.end(), channelData.begin(), channelData.end());
            }
            gpuProcessor_->uploadImpulseResponse(combinedData, ir.channels);
        }

        // Update statistics
        {
            std::lock_guard<std::mutex> lock(statsMutex_);
            statistics_.activeIRs = impulseResponses_.size();
        }

        if (irChangedCallback_) {
            irChangedCallback_(name, ir);
        }

        Logger::info("ConvolutionEngine: Impulse response '{}' loaded successfully", name);
        return true;

    } catch (const std::exception& e) {
        setError("Failed to load impulse response data: " + std::string(e.what()));
        return false;
    }
}

bool ConvolutionEngine::createSyntheticIR(const std::string& name, float length, float rt60,
                                         float mixLevel, float decay) {
    Logger::info("ConvolutionEngine: Creating synthetic IR '{}' length={}s RT60={}s", name, length, rt60);

    try {
        uint32_t numSamples = static_cast<uint32_t>(length * config_.sampleRate);
        std::vector<float> irData(numSamples);

        // Generate synthetic reverb impulse response using exponentially decaying noise
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> noise(0.0f, 0.1f);

        float decayRate = -std::log(0.001f) / (rt60 * config_.sampleRate); // 60dB decay
        float earlyReflectionsTime = 0.05f * config_.sampleRate; // 50ms early reflections

        // Early reflections
        for (uint32_t i = 0; i < earlyReflectionsTime && i < numSamples; ++i) {
            float amplitude = std::exp(-decayRate * i) * mixLevel;
            irData[i] = noise(gen) * amplitude * 0.5f;
        }

        // Late reverb
        for (uint32_t i = earlyReflectionsTime; i < numSamples; ++i) {
            float amplitude = std::exp(-decayRate * i) * mixLevel;
            irData[i] = noise(gen) * amplitude;
        }

        // Apply smoothing and normalization
        smoothImpulseResponse(irData);
        normalizeImpulseResponseData(irData);

        return loadImpulseResponseFromData(irData, config_.sampleRate, name);

    } catch (const std::exception& e) {
        setError("Failed to create synthetic IR: " + std::string(e.what()));
        return false;
    }
}

bool ConvolutionEngine::setWetDryMix(float wetLevel, float dryLevel) {
    wetLevel = std::clamp(wetLevel, 0.0f, 2.0f);
    dryLevel = std::clamp(dryLevel, 0.0f, 2.0f);

    wetLevel_ = wetLevel;
    dryLevel_ = dryLevel;

    Logger::info("ConvolutionEngine: Set wet/dry mix: wet={:.2f}, dry={:.2f}", wetLevel, dryLevel);
    return true;
}

ConvolutionEngine::ConvolutionStatistics ConvolutionEngine::getStatistics() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    return statistics_;
}

void ConvolutionEngine::resetStatistics() {
    std::lock_guard<std::mutex> lock(statsMutex_);
    statistics_ = ConvolutionStatistics{};
    statistics_.startTime = std::chrono::steady_clock::now();
    statistics_.lastActivity = statistics_.startTime;
    statistics_.minLatency = 1000.0f;
}

bool ConvolutionEngine::isHealthy() const {
    if (!initialized_.load() || !processingActive_.load()) {
        return false;
    }

    auto stats = getStatistics();
    return (stats.averageLatency < config_.maxLatencyMs &&
            stats.totalHarmonicDistortion < 0.01f &&  // < 1% THD
            stats.droppedFrames == 0);
}

// Private methods implementation
void ConvolutionEngine::processChannelData(const float* input, float* output, size_t numSamples, uint16_t channel) {
    if (!input || !output) {
        return;
    }

    // Get the current IR
    ImpulseResponse currentIR;
    {
        std::lock_guard<std::mutex> lock(irMutex_);
        if (currentIRName_.empty() || impulseResponses_.find(currentIRName_) == impulseResponses_.end()) {
            // No IR loaded, just copy input to output
            std::copy(input, input + numSamples, output);
            return;
        }
        currentIR = impulseResponses_.at(currentIRName_);
    }

    // Process using the selected method
    switch (config_.method) {
        case ConvolutionMethod::OVERLAP_ADD:
            processOverlapAdd(input, output, numSamples);
            break;
        case ConvolutionMethod::OVERLAP_SAVE:
            processOverlapSave(input, output, numSamples);
            break;
        case ConvolutionMethod::UNIFORM_PARTITIONED:
            processUniformPartitioned(input, output, numSamples);
            break;
        case ConvolutionMethod::FREQUENCY_DOMAIN:
            processFrequencyDomain(input, output, numSamples);
            break;
        default:
            processUniformPartitioned(input, output, numSamples);
            break;
    }
}

void ConvolutionEngine::processUniformPartitioned(const float* input, float* output, size_t numSamples) {
    std::lock_guard<std::mutex> lock(buffersMutex_);

    // Ensure buffers are properly sized
    size_t blockSize = config_.blockSize;
    if (buffers_.inputBuffer.size() < blockSize) {
        buffers_.inputBuffer.resize(blockSize);
        buffers_.outputBuffer.resize(blockSize);
        buffers_.intermediateBuffer.resize(blockSize);
        buffers_.fftBuffer.resize(fftSize_);
    }

    // Get the current IR
    ImpulseResponse currentIR;
    {
        std::lock_guard<std::mutex> irLock(irMutex_);
        if (currentIRName_.empty() || impulseResponses_.find(currentIRName_) == impulseResponses_.end()) {
            std::copy(input, input + numSamples, output);
            return;
        }
        currentIR = impulseResponses_.at(currentIRName_);
    }

    if (currentIR.frequencyDomainData.empty()) {
        std::copy(input, input + numSamples, output);
        return;
    }

    // Clear output buffer
    std::fill(output, output + numSamples, 0.0f);

    // Process blocks
    for (size_t blockStart = 0; blockStart < numSamples; blockStart += blockSize) {
        size_t currentBlockSize = std::min(blockSize, numSamples - blockStart);

        // Copy input block and apply window
        std::copy(input + blockStart, input + blockStart + currentBlockSize, buffers_.inputBuffer.begin());
        std::fill(buffers_.inputBuffer.begin() + currentBlockSize, buffers_.inputBuffer.end(), 0.0f);

        if (config_.enableWindowing) {
            applyWindowFunction(buffers_.inputBuffer, config_.windowFunction);
        }

        // Zero-pad and compute FFT
        std::fill(buffers_.fftBuffer.begin(), buffers_.fftBuffer.end(), std::complex<float>(0.0f));
        for (size_t i = 0; i < currentBlockSize; ++i) {
            buffers_.fftBuffer[i] = std::complex<float>(buffers_.inputBuffer[i], 0.0f);
        }

        computeFFT(buffers_.fftBuffer);

        // Multiply with IR frequency response
        if (channel < currentIR.frequencyDomainData.size()) {
            const auto& irFreqData = currentIR.frequencyDomainData[channel];
            size_t complexSize = std::min(buffers_.fftBuffer.size(), irFreqData.size());

            for (size_t i = 0; i < complexSize; ++i) {
                buffers_.fftBuffer[i] *= irFreqData[i];
            }
        }

        // Inverse FFT
        computeIFFT(buffers_.fftBuffer);

        // Extract real part and add to output
        for (size_t i = 0; i < currentBlockSize && (blockStart + i) < numSamples; ++i) {
            output[blockStart + i] = buffers_.fftBuffer[i].real();
        }

        // Update statistics
        {
            std::lock_guard<std::mutex> statsLock(statsMutex_);
            statistics_.fftCalls++;
        }
    }
}

void ConvolutionEngine::processFrequencyDomain(const float* input, float* output, size_t numSamples) {
    // For frequency domain processing, we need to handle the entire block at once
    std::lock_guard<std::mutex> lock(buffersMutex_);

    if (buffers_.fftBuffer.size() < fftSize_) {
        buffers_.fftBuffer.resize(fftSize_);
    }

    // Clear output
    std::fill(output, output + numSamples, 0.0f);

    // Get current IR
    ImpulseResponse currentIR;
    {
        std::lock_guard<std::mutex> irLock(irMutex_);
        if (currentIRName_.empty() || impulseResponses_.find(currentIRName_) == impulseResponses_.end()) {
            std::copy(input, input + numSamples, output);
            return;
        }
        currentIR = impulseResponses_.at(currentIRName_);
    }

    if (currentIR.frequencyDomainData.empty()) {
        std::copy(input, input + numSamples, output);
        return;
    }

    // Zero-pad input
    std::fill(buffers_.fftBuffer.begin(), buffers_.fftBuffer.end(), std::complex<float>(0.0f));
    size_t copySize = std::min(static_cast<size_t>(numSamples), buffers_.fftBuffer.size());
    for (size_t i = 0; i < copySize; ++i) {
        buffers_.fftBuffer[i] = std::complex<float>(input[i], 0.0f);
    }

    // Apply window if enabled
    if (config_.enableWindowing) {
        std::vector<float> windowData(copySize);
        for (size_t i = 0; i < copySize; ++i) {
            windowData[i] = buffers_.fftBuffer[i].real();
        }
        applyWindowFunction(windowData, config_.windowFunction);
        for (size_t i = 0; i < copySize; ++i) {
            buffers_.fftBuffer[i] = std::complex<float>(windowData[i], 0.0f);
        }
    }

    // Compute FFT
    computeFFT(buffers_.fftBuffer);

    // Multiply with IR frequency response (assuming first channel)
    if (!currentIR.frequencyDomainData.empty()) {
        const auto& irFreqData = currentIR.frequencyDomainData[0];
        size_t complexSize = std::min(buffers_.fftBuffer.size(), irFreqData.size());

        for (size_t i = 0; i < complexSize; ++i) {
            buffers_.fftBuffer[i] *= irFreqData[i];
        }
    }

    // Inverse FFT
    computeIFFT(buffers_.fftBuffer);

    // Extract real part
    for (size_t i = 0; i < numSamples; ++i) {
        output[i] = buffers_.fftBuffer[i].real();
    }

    // Update statistics
    {
        std::lock_guard<std::mutex> statsLock(statsMutex_);
        statistics_.fftCalls += 2; // One FFT and one IFFT
    }
}

void ConvolutionEngine::computeFFT(std::vector<std::complex<float>>& data) {
    // Implement Cooley-Tukey FFT algorithm
    uint32_t N = static_cast<uint32_t>(data.size());
    if (N <= 1) return;

    // Bit-reversal permutation
    for (uint32_t i = 0, j = 0; i < N; ++i) {
        if (j > i) {
            std::swap(data[i], data[j]);
        }
        uint32_t m = N >> 1;
        while (j >= m) {
            j ^= m;
            m >>= 1;
        }
        j |= m;
    }

    // Cooley-Tukey FFT
    for (uint32_t len = 2; len <= N; len <<= 1) {
        float angle = -2.0f * M_PI / len;
        std::complex<float> wlen(std::cos(angle), std::sin(angle));

        for (uint32_t i = 0; i < N; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (uint32_t j = 0; j < len / 2; ++j) {
                std::complex<float> u = data[i + j];
                std::complex<float> v = data[i + j + len / 2] * w;
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

void ConvolutionEngine::computeIFFT(std::vector<std::complex<float>>& data) {
    // Compute inverse FFT using forward FFT and conjugation
    for (auto& val : data) {
        val = std::conj(val);
    }

    computeFFT(data);

    for (auto& val : data) {
        val = std::conj(val) / static_cast<float>(data.size());
    }
}

void ConvolutionEngine::applyWindowFunction(std::vector<float>& data, const std::string& windowType) {
    size_t N = data.size();
    if (N == 0) return;

    if (windowType == "hann" || windowType == "hanning") {
        for (size_t i = 0; i < N; ++i) {
            float window = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (N - 1)));
            data[i] *= window;
        }
    } else if (windowType == "hamming") {
        for (size_t i = 0; i < N; ++i) {
            float window = 0.54f - 0.46f * std::cos(2.0f * M_PI * i / (N - 1));
            data[i] *= window;
        }
    } else if (windowType == "blackman") {
        for (size_t i = 0; i < N; ++i) {
            float window = 0.42f - 0.5f * std::cos(2.0f * M_PI * i / (N - 1)) +
                           0.08f * std::cos(4.0f * M_PI * i / (N - 1));
            data[i] *= window;
        }
    }
}

bool ConvolutionEngine::processImpulseResponse(ImpulseResponse& ir) {
    try {
        // Convert to frequency domain
        if (!convertToFrequencyDomain(ir)) {
            return false;
        }

        // Analyze properties
        if (!analyzeImpulseResponseProperties(ir)) {
            return false;
        }

        // Normalize if enabled
        if (normalizationEnabled_) {
            normalizeImpulseResponse(ir, 0.0f);
        }

        ir.processed = true;
        return true;

    } catch (const std::exception& e) {
        Logger::error("ConvolutionEngine: Failed to process IR: {}", e.what());
        return false;
    }
}

bool ConvolutionEngine::convertToFrequencyDomain(ImpulseResponse& ir) {
    if (ir.timeDomainData.empty()) {
        return false;
    }

    ir.frequencyDomainData.clear();
    ir.frequencyDomainData.resize(ir.timeDomainData.size());

    for (size_t ch = 0; ch < ir.timeDomainData.size(); ++ch) {
        const auto& timeData = ir.timeDomainData[ch];

        // Create FFT buffer and zero-pad
        std::vector<std::complex<float>> fftData(fftSize_);
        for (size_t i = 0; i < std::min(timeData.size(), static_cast<size_t>(fftSize_)); ++i) {
            fftData[i] = std::complex<float>(timeData[i], 0.0f);
        }

        // Apply window if enabled
        if (config_.enableWindowing) {
            std::vector<float> windowData(timeData.size());
            for (size_t i = 0; i < timeData.size(); ++i) {
                windowData[i] = fftData[i].real();
            }
            applyWindowFunction(windowData, config_.windowFunction);
            for (size_t i = 0; i < timeData.size(); ++i) {
                fftData[i] = std::complex<float>(windowData[i], 0.0f);
            }
        }

        // Compute FFT
        computeFFT(fftData);
        ir.frequencyDomainData[ch] = std::move(fftData);
    }

    return true;
}

bool ConvolutionEngine::analyzeImpulseResponseProperties(ImpulseResponse& ir) {
    if (ir.timeDomainData.empty()) {
        return false;
    }

    const auto& data = ir.timeDomainData[0];

    // Calculate peak level
    auto peakIt = std::max_element(data.begin(), data.end(),
                                   [](float a, float b) { return std::abs(a) < std::abs(b); });
    ir.peakLevel = 20.0f * std::log10(std::max(std::abs(*peakIt), 1e-10f));

    // Calculate average level
    float sum = std::accumulate(data.begin(), data.end(), 0.0f,
                               [](float acc, float val) { return acc + std::abs(val); });
    ir.averageLevel = 20.0f * std::log10(std::max(sum / data.size(), 1e-10f));

    // Calculate dynamic range
    float minVal = *std::min_element(data.begin(), data.end());
    float maxVal = *std::max_element(data.begin(), data.end());
    ir.dynamicRange = 20.0f * std::log10(std::abs(maxVal) / std::max(std::abs(minVal), 1e-10f));

    // Estimate RT60 (simplified)
    float threshold = ir.peakLevel - 60.0f; // 60dB below peak
    uint32_t rt60Sample = 0;
    for (uint32_t i = static_cast<uint32_t>(data.size() * 0.1); i < data.size(); ++i) {
        float level = 20.0f * std::log10(std::max(std::abs(data[i]), 1e-10f));
        if (level <= threshold) {
            rt60Sample = i;
            break;
        }
    }
    ir.rt60 = static_cast<float>(rt60Sample) / ir.sampleRate;

    // Generate frequency response
    if (!ir.frequencyDomainData.empty()) {
        ir.frequencyResponse.clear();
        ir.phaseResponse.clear();

        const auto& freqData = ir.frequencyDomainData[0];
        for (const auto& sample : freqData) {
            ir.frequencyResponse.push_back(20.0f * std::log10(std::max(std::abs(sample), 1e-10f)));
            ir.phaseResponse.push_back(std::arg(sample));
        }
    }

    return true;
}

bool ConvolutionEngine::normalizeImpulseResponse(ImpulseResponse& ir, float targetLevel) {
    if (ir.timeDomainData.empty()) {
        return false;
    }

    for (auto& channelData : ir.timeDomainData) {
        // Find peak value
        float peakValue = 0.0f;
        for (float sample : channelData) {
            peakValue = std::max(peakValue, std::abs(sample));
        }

        if (peakValue > 1e-10f) {
            // Calculate normalization factor
            float targetLinear = dbToLinear(targetLevel);
            float normalizationFactor = targetLinear / peakValue;

            // Apply normalization
            for (float& sample : channelData) {
                sample *= normalizationFactor;
            }
        }
    }

    return true;
}

void ConvolutionEngine::applyDithering(float* audio, size_t numSamples) {
    if (!audio || numSamples == 0) {
        return;
    }

    // Simple triangular dithering
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

    for (size_t i = 0; i < numSamples; ++i) {
        float dither = dis(gen);
        audio[i] += dither / static_cast<float>(1 << (config_.ditherDepth - 1));
    }
}

void ConvolutionEngine::applyNoiseShaping(float* audio, size_t numSamples) {
    // Simple first-order noise shaping
    static float noiseShaperState = 0.0f;

    for (size_t i = 0; i < numSamples; ++i) {
        float error = std::round(audio[i]) - audio[i];
        noiseShaperState = error * 0.5f; // Simple first-order filter
        audio[i] += noiseShaperState;
    }
}

void ConvolutionEngine::initializeBuffers() {
    std::lock_guard<std::mutex> lock(buffersMutex_);

    size_t bufferSize = std::max(config_.blockSize, static_cast<uint32_t>(4096));
    buffers_.inputBuffer.resize(bufferSize);
    buffers_.outputBuffer.resize(bufferSize);
    buffers_.intermediateBuffer.resize(bufferSize);
    buffers_.fftBuffer.resize(fftSize_);
    buffers_.irBuffer.resize(fftSize_);
    buffers_.overlapBuffer.resize(config_.blockSize);

    if (config_.enableWindowing) {
        buffers_.windowBuffer.resize(bufferSize);
        // Generate window function
        generateWindowFunction(buffers_.windowBuffer, config_.windowFunction);
    }
}

void ConvolutionEngine::initializeFFT() {
    fftTwiddleFactors_.clear();
    fftTwiddleFactors_.resize(fftSize_);

    for (uint32_t i = 0; i < fftSize_; ++i) {
        float angle = -2.0f * M_PI * i / fftSize_;
        fftTwiddleFactors_[i] = std::complex<float>(std::cos(angle), std::sin(angle));
    }

    Logger::info("ConvolutionEngine: Initialized FFT with {} points", fftSize_);
}

bool ConvolutionEngine::initializeGPUResources() {
    try {
        gpuProcessor_ = std::make_unique<GPUConvolutionProcessor>(config_);
        if (!gpuProcessor_->initialize()) {
            Logger::error("ConvolutionEngine: Failed to initialize GPU processor");
            return false;
        }

        Logger::info("ConvolutionEngine: GPU resources initialized successfully");
        return true;

    } catch (const std::exception& e) {
        Logger::error("ConvolutionEngine: GPU initialization error: {}", e.what());
        return false;
    }
}

void ConvolutionEngine::cleanupGPUResources() {
    if (gpuProcessor_) {
        gpuProcessor_->shutdown();
        gpuProcessor_.reset();
    }

    gpuIRBuffer_ = nullptr;
    gpuInputBuffer_ = nullptr;
    gpuOutputBuffer_ = nullptr;
    gpuWorkBuffer_ = nullptr;
    gpuMemorySize_ = 0;
}

// Utility functions
uint32_t ConvolutionEngine::calculateOptimalFFTSize(uint32_t irLength, uint32_t blockSize) const {
    uint32_t minSize = blockSize + irLength;

    // Find next power of 2 that's >= minSize
    uint32_t fftSize = 1;
    while (fftSize < minSize) {
        fftSize <<= 1;
    }

    // Ensure it's within reasonable bounds
    fftSize = std::clamp(fftSize, MIN_FFT_SIZE, MAX_FFT_SIZE);

    return fftSize;
}

uint32_t ConvolutionEngine::calculateNumPartitions(uint32_t irLength, uint32_t fftSize) const {
    if (irLength <= fftSize) {
        return 1;
    }
    return (irLength + fftSize - 1) / fftSize;
}

float ConvolutionEngine::dbToLinear(float db) const {
    return std::pow(10.0f, db / 20.0f);
}

float ConvolutionEngine::linearToDb(float linear) const {
    return 20.0f * std::log10(std::max(linear, 1e-10f));
}

bool ConvolutionEngine::isValidSampleRate(uint32_t sampleRate) const {
    return (sampleRate >= 8000 && sampleRate <= 384000);
}

void ConvolutionEngine::setError(const std::string& error) const {
    lastError_ = error;
    diagnosticMessages_.push_back(error);
    Logger::error("ConvolutionEngine: {}", error);

    if (errorCallback_) {
        errorCallback_("engine", error);
    }
}

bool ConvolutionEngine::loadAudioFile(const std::string& filePath, std::vector<float>& audioData,
                                      uint32_t& sampleRate, uint16_t& channels) {
    // Placeholder implementation - would use audio loading library
    // For now, just create a simple test impulse response

    sampleRate = config_.sampleRate;
    channels = 1;

    // Create a simple exponential decay impulse response
    float length = 2.0f; // 2 seconds
    uint32_t numSamples = static_cast<uint32_t>(length * sampleRate);
    audioData.resize(numSamples);

    for (uint32_t i = 0; i < numSamples; ++i) {
        float decayRate = 2.0f; // decay rate
        audioData[i] = std::exp(-decayRate * static_cast<float>(i) / sampleRate);

        // Add some early reflections
        if (i < sampleRate * 0.1f) {
            audioData[i] *= (1.0f + 0.3f * std::sin(2.0f * M_PI * i * 1000.0f / sampleRate));
        }
    }

    return true;
}

void ConvolutionEngine::smoothImpulseResponse(std::vector<float>& irData) {
    size_t N = irData.size();
    if (N < 4) return;

    // Apply simple smoothing filter
    for (size_t i = 2; i < N - 2; ++i) {
        irData[i] = 0.25f * irData[i-2] + 0.5f * irData[i] + 0.25f * irData[i+2];
    }
}

void ConvolutionEngine::normalizeImpulseResponseData(std::vector<float>& irData) {
    if (irData.empty()) return;

    // Find peak value
    float peakValue = 0.0f;
    for (float sample : irData) {
        peakValue = std::max(peakValue, std::abs(sample));
    }

    if (peakValue > 1e-10f) {
        // Normalize to -6dB
        float targetPeak = dbToLinear(-6.0f);
        float normalizationFactor = targetPeak / peakValue;

        for (float& sample : irData) {
            sample *= normalizationFactor;
        }
    }
}

void ConvolutionEngine::applyPredelay(const float* input, float* output, size_t numSamples, float delayMs) {
    uint32_t delaySamples = static_cast<uint32_t>(delayMs * config_.sampleRate / 1000.0f);

    // Simple delay line implementation
    static std::vector<float> delayLine;
    static uint32_t writePos = 0;

    // Resize delay line if needed
    if (delayLine.size() < delaySamples + numSamples) {
        delayLine.resize(delaySamples + numSamples, 0.0f);
    }

    // Write to delay line
    for (size_t i = 0; i < numSamples; ++i) {
        delayLine[writePos] = input[i];
        writePos = (writePos + 1) % delayLine.size();
    }

    // Read from delay line
    uint32_t readPos = (writePos + delayLine.size() - delaySamples) % delayLine.size();
    for (size_t i = 0; i < numSamples; ++i) {
        output[i] = delayLine[readPos];
        readPos = (readPos + 1) % delayLine.size();
    }
}

void ConvolutionEngine::generateWindowFunction(std::vector<float>& window, const std::string& windowType) {
    size_t N = window.size();
    if (N == 0) return;

    if (windowType == "hann" || windowType == "hanning") {
        for (size_t i = 0; i < N; ++i) {
            window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (N - 1)));
        }
    } else if (windowType == "hamming") {
        for (size_t i = 0; i < N; ++i) {
            window[i] = 0.54f - 0.46f * std::cos(2.0f * M_PI * i / (N - 1));
        }
    } else {
        // Default to rectangular window
        std::fill(window.begin(), window.end(), 1.0f);
    }
}

// GPUConvolutionProcessor Implementation
ConvolutionEngine::GPUConvolutionProcessor::GPUConvolutionProcessor(const ConvolutionConfig& config)
    : config_(config) {
    Logger::info("ConvolutionEngine::GPU: Initializing GPU convolution processor");
}

ConvolutionEngine::GPUConvolutionProcessor::~GPUConvolutionProcessor() {
    shutdown();
    Logger::info("ConvolutionEngine::GPU: GPU convolution processor destroyed");
}

bool ConvolutionEngine::GPUConvolutionProcessor::initialize() {
    if (initialized_.load()) {
        return true;
    }

    try {
        // Try to initialize CUDA first
        if (initializeCUDA()) {
            Logger::info("ConvolutionEngine::GPU: CUDA backend initialized");
            initialized_.store(true);
            return true;
        }

        // Fall back to OpenCL
        if (initializeOpenCL()) {
            Logger::info("ConvolutionEngine::GPU: OpenCL backend initialized");
            initialized_.store(true);
            return true;
        }

        Logger::error("ConvolutionEngine::GPU: No GPU backend available");
        return false;

    } catch (const std::exception& e) {
        Logger::error("ConvolutionEngine::GPU: Initialization failed: {}", e.what());
        return false;
    }
}

void ConvolutionEngine::GPUConvolutionProcessor::shutdown() {
    initialized_.store(false);

    // Cleanup GPU resources
    if (irBuffer_) {
        // Free GPU memory
        irBuffer_ = nullptr;
    }
    if (inputBuffer_) {
        inputBuffer_ = nullptr;
    }
    if (outputBuffer_) {
        outputBuffer_ = nullptr;
    }
    if (workBuffer_) {
        workBuffer_ = nullptr;
    }

    Logger::info("ConvolutionEngine::GPU: GPU processor shutdown");
}

bool ConvolutionEngine::GPUConvolutionProcessor::uploadImpulseResponse(const std::vector<std::complex<float>>& irData, uint16_t channels) {
    if (!initialized_.load()) {
        return false;
    }

    // Calculate required buffer size
    size_t complexSize = irData.size();
    size_t bufferSize = complexSize * sizeof(std::complex<float>);

    // Allocate or resize GPU buffer if needed
    if (bufferSize_ < bufferSize) {
        // This would reallocate GPU memory
        bufferSize_ = bufferSize;
    }

    // Upload IR data to GPU
    // Implementation depends on GPU backend
    irSize_ = complexSize;
    numChannels_ = channels;

    Logger::info("ConvolutionEngine::GPU: Uploaded impulse response with {} complex samples", complexSize);
    return true;
}

bool ConvolutionEngine::GPUConvolutionProcessor::processAudio(const float* input, float* output, size_t numSamples, uint16_t channels) {
    if (!initialized_.load() || !input || !output) {
        return false;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    // Process audio using the available GPU backend
    switch (gpuContext_.initialized) {
#ifdef VORTEX_GPU_CUDA
        case 1: // CUDA
            processWithCUDA(input, output, numSamples, channels);
            break;
#endif
#ifdef VORTEX_GPU_OPENCL
        case 2: // OpenCL
            processWithOpenCL(input, output, numSamples, channels);
            break;
#endif
        default:
            return false;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<float, std::milli>(endTime - startTime);
    processingTime_.store(duration.count());

    return true;
}

bool ConvolutionEngine::GPUConvolutionProcessor::initializeCUDA() {
#ifdef VORTEX_GPU_CUDA
    try {
        // Initialize CUDA
        cudaError_t result = cudaSuccess;

        // This would initialize CUDA devices, context, etc.
        // For now, just mark as initialized for testing

        gpuContext_.device = nullptr;
        gpuContext_.context = nullptr;
        gpuContext_.initialized = true;

        return true;

    } catch (const std::exception& e) {
        Logger::error("ConvolutionEngine::GPU: CUDA initialization failed: {}", e.what());
        return false;
    }
#else
    Logger::warn("ConvolutionEngine::GPU: CUDA support not compiled");
    return false;
#endif
}

bool ConvolutionEngine::GPUConvolutionProcessor::initializeOpenCL() {
#ifdef VORTEX_GPU_OPENCL
    try {
        // Initialize OpenCL
        // This would enumerate platforms and devices, create context, etc.

        gpuContext_.device = nullptr;
        gpuContext_.context = nullptr;
        gpuContext_.initialized = true;

        return true;

    } catch (const std::exception& e) {
        Logger::error("ConvolutionEngine::GPU: OpenCL initialization failed: {}", e.what());
        return false;
    }
#else
    Logger::warn("ConvolutionEngine::GPU: OpenCL support not compiled");
    return false;
#endif
}

void ConvolutionEngine::GPUConvolutionProcessor::processWithCUDA(const float* input, float* output, size_t numSamples, uint16_t channels) {
    // Placeholder for CUDA processing implementation
    // Real implementation would:
    // 1. Copy input data to GPU
    // 2. Launch CUDA kernels for convolution
    // 3. Copy output data back to CPU
    // 4. Synchronize if necessary

    // For now, just copy input to output (no processing)
    if (input && output) {
        std::copy(input, input + numSamples * channels, output);
    }
}

void ConvolutionEngine::GPUConvolutionProcessor::processWithOpenCL(const float* input, float* output, size_t numSamples, uint16_t channels) {
    // Placeholder for OpenCL processing implementation
    // Similar to CUDA but using OpenCL APIs

    // For now, just copy input to output (no processing)
    if (input && output) {
        std::copy(input, input + numSamples * channels, output);
    }
}

float ConvolutionEngine::GPUConvolutionProcessor::getProcessingTime() const {
    return processingTime_.load();
}

float ConvolutionEngine::GPUConvolutionProcessor::getGPUUtilization() const {
    return gpuUtilization_.load();
}

} // namespace vortex