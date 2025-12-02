#include "equalizer.hpp"
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

#ifdef VORTEX_GPU_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#endif

#ifdef VORTEX_GPU_OPENCL
#include <CL/cl.h>
#endif

namespace vortex {

// Equalizer Implementation
Equalizer::Equalizer() {
    Logger::info("Equalizer: Initializing 512-band graphic equalizer");

    // Initialize statistics
    statistics_.startTime = std::chrono::steady_clock::now();
    lastUpdateTime_ = statistics_.startTime;
}

Equalizer::~Equalizer() {
    shutdown();
    Logger::info("Equalizer: 512-band graphic equalizer destroyed");
}

bool Equalizer::initialize(const EqualizerConfig& config) {
    if (initialized_.load()) {
        Logger::warn("Equalizer: Already initialized");
        return true;
    }

    Logger::info("Equalizer: Initializing with {} bands, {} Hz sample rate, {} channels",
                 config.numBands, config.sampleRate, config.numChannels);

    config_ = config;

    try {
        // Validate configuration
        if (config.numBands > MAX_BANDS) {
            setError("Number of bands exceeds maximum");
            return false;
        }

        if (config.sampleRate < 8000 || config.sampleRate > 384000) {
            setError("Invalid sample rate");
            return false;
        }

        // Initialize frequency scale
        initializeFrequencyScale();

        // Initialize filter bands
        initializeFilterBands();

        // Initialize processing buffers
        initializeBuffers();

        // Initialize GPU resources if enabled
        if (config.enableGPUAcceleration) {
            if (!initializeGPUResources()) {
                Logger::warn("Equalizer: GPU initialization failed, falling back to CPU");
                config_.enableGPUAcceleration = false;
            }
        }

        // Initialize multi-threading
        if (config.enableMultiThreading) {
            uint32_t numThreads = config.numThreads > 0 ? config.numThreads : std::thread::hardware_concurrency();
            setProcessingThreads(numThreads);
        }

        // Load default flat preset
        loadFlatPreset();

        initialized_.store(true);
        Logger::info("Equalizer: Initialization complete");
        return true;

    } catch (const std::exception& e) {
        setError("Initialization failed: " + std::string(e.what()));
        return false;
    }
}

void Equalizer::shutdown() {
    if (!initialized_.load()) {
        return;
    }

    Logger::info("Equalizer: Shutting down");

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
        buffers_.lookaheadBuffer.clear();
        buffers_.fftBuffer.clear();
        buffers_.spectrumBuffer.clear();
    }

    // Clear filter states
    {
        std::lock_guard<std::mutex> lock(bandsMutex_);
        for (auto& band : bands_) {
            std::fill(band.delayLine.begin(), band.delayLine.end(), 0.0f);
            std::fill(band.history.begin(), band.history.end(), 0.0f);
        }
    }

    Logger::info("Equalizer: Shutdown complete");
}

bool Equalizer::isInitialized() const {
    return initialized_.load();
}

bool Equalizer::processAudio(const float* input, float* output, size_t numSamples) {
    if (!initialized_.load() || !input || !output) {
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
    }

    return success;
}

bool Equalizer::processAudioMultiChannel(const std::vector<const float*>& inputs,
                                         std::vector<float*>& outputs,
                                         size_t numSamples, uint16_t channels) {
    if (!initialized_.load() || inputs.empty() || outputs.empty() ||
        inputs.size() != outputs.size() || channels == 0) {
        return false;
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

            if (config_.enableGPUAcceleration && gpuProcessor_) {
                // GPU processing path
                if (!gpuProcessor_->processAudio(input, output, numSamples, 1)) {
                    Logger::warn("Equalizer: GPU processing failed, falling back to CPU");
                    // Fallback to CPU processing
                    processChannelData(input, output, numSamples, ch);
                }
            } else {
                // CPU processing path
                if (multithreadingEnabled_.load() && channels > 1) {
                    processChannelThread(input, output, numSamples, ch, 1);
                } else {
                    processChannelData(input, output, numSamples, ch);
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

bool Equalizer::setBandGain(uint32_t bandIndex, float gain) {
    if (!isValidBandIndex(bandIndex)) {
        return false;
    }

    gain = std::clamp(gain, MIN_GAIN, MAX_GAIN);

    std::lock_guard<std::mutex> lock(bandsMutex_);
    bands_[bandIndex].gain = gain;

    // Recompute filter coefficients
    computeBiquadCoefficients(bands_[bandIndex]);

    // Update GPU if enabled
    if (config_.enableGPUAcceleration && gpuProcessor_) {
        updateGPUBand(bandIndex);
    }

    // Notify callback
    if (bandChangedCallback_) {
        bandChangedCallback_(bandIndex, bands_[bandIndex]);
    }

    return true;
}

bool Equalizer::setBandParameters(uint32_t bandIndex, float frequency, float gain, float Q,
                                   FilterType type, FilterSlope slope) {
    if (!isValidBandIndex(bandIndex)) {
        return false;
    }

    frequency = std::clamp(frequency, MIN_FREQUENCY, MAX_FREQUENCY);
    gain = std::clamp(gain, MIN_GAIN, MAX_GAIN);
    Q = std::clamp(Q, MIN_Q, MAX_Q);

    std::lock_guard<std::mutex> lock(bandsMutex_);
    FilterBand& band = bands_[bandIndex];
    band.frequency = frequency;
    band.gain = gain;
    band.Q = Q;
    band.type = type;
    band.slope = slope;

    // Recompute filter coefficients
    computeBiquadCoefficients(band);

    // Update GPU if enabled
    if (config_.enableGPUAcceleration && gpuProcessor_) {
        updateGPUBand(bandIndex);
    }

    // Notify callback
    if (bandChangedCallback_) {
        bandChangedCallback_(bandIndex, band);
    }

    return true;
}

Equalizer::FilterBand Equalizer::getBand(uint32_t bandIndex) const {
    std::lock_guard<std::mutex> lock(bandsMutex_);
    if (isValidBandIndex(bandIndex)) {
        return bands_[bandIndex];
    }
    return FilterBand{};
}

std::vector<Equalizer::FilterBand> Equalizer::getAllBands() const {
    std::lock_guard<std::mutex> lock(bandsMutex_);
    return bands_;
}

std::vector<float> Equalizer::getFrequencyResponse(uint32_t numPoints) const {
    std::vector<float> response(numPoints);
    const float minLog = std::log10(MIN_FREQUENCY);
    const float maxLog = std::log10(MAX_FREQUENCY);
    const float logRange = maxLog - minLog;

    for (uint32_t i = 0; i < numPoints; ++i) {
        float logFreq = minLog + (static_cast<float>(i) / static_cast<float>(numPoints - 1)) * logRange;
        float frequency = std::pow(10.0f, logFreq);
        response[i] = getFrequencyResponseAtFrequency(frequency);
    }

    return response;
}

float Equalizer::getFrequencyResponseAtFrequency(float frequency) const {
    std::lock_guard<std::mutex> lock(bandsMutex_);
    float response = 0.0f;

    for (const auto& band : bands_) {
        if (band.enabled && !band.bypassed) {
            response += computeFrequencyResponse(frequency, band);
        }
    }

    return response;
}

bool Equalizer::savePreset(const std::string& name, const std::string& description) {
    if (name.empty()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(bandsMutex_);

    Preset preset;
    preset.name = name;
    preset.description = description;
    preset.author = "VortexGPU";
    preset.version = "1.0.0";
    preset.createdTime = std::chrono::system_clock::now();
    preset.modifiedTime = preset.createdTime;

    // Store band settings
    preset.bandGains.clear();
    preset.bandSettings = bands_;

    for (const auto& band : bands_) {
        preset.bandGains.push_back(band.gain);
    }

    // Save preset
    {
        std::lock_guard<std::mutex> presetLock(presetsMutex_);
        presets_[name] = preset;
        currentPreset_ = name;
    }

    // Auto-save to file if enabled
    if (config_.autoSavePresets) {
        exportPreset(name, config_.presetDirectory + name + ".preset");
    }

    if (presetChangedCallback_) {
        presetChangedCallback_(name, preset);
    }

    Logger::info("Equalizer: Saved preset '{}'", name);
    return true;
}

bool Equalizer::loadPreset(const std::string& name) {
    std::lock_guard<std::mutex> presetLock(presetsMutex_);
    auto it = presets_.find(name);
    if (it == presets_.end()) {
        return false;
    }

    const Preset& preset = it->second;

    // Apply preset to bands
    {
        std::lock_guard<std::mutex> bandLock(bandsMutex_);
        if (preset.bandSettings.size() == bands_.size()) {
            // Load complete band settings
            bands_ = preset.bandSettings;
        } else if (preset.bandGains.size() == bands_.size()) {
            // Load only gains
            for (size_t i = 0; i < preset.bandGains.size() && i < bands_.size(); ++i) {
                bands_[i].gain = preset.bandGains[i];
                computeBiquadCoefficients(bands_[i]);
            }
        }

        // Update GPU if enabled
        if (config_.enableGPUAcceleration && gpuProcessor_) {
            updateGPUFilters();
        }
    }

    currentPreset_ = name;

    if (presetChangedCallback_) {
        presetChangedCallback_(name, preset);
    }

    Logger::info("Equalizer: Loaded preset '{}'", name);
    return true;
}

bool Equalizer::loadFlatPreset() {
    Logger::info("Equalizer: Loading flat preset");

    std::lock_guard<std::mutex> lock(bandsMutex_);
    for (auto& band : bands_) {
        band.gain = 0.0f;
        band.type = FilterType::BELL;
        band.slope = FilterSlope::SLOPE_24_DB;
        band.Q = 1.0f;
        band.enabled = true;
        band.bypassed = false;
        computeBiquadCoefficients(band);
    }

    if (config_.enableGPUAcceleration && gpuProcessor_) {
        updateGPUFilters();
    }

    currentPreset_ = "Flat";

    if (presetChangedCallback_) {
        Preset flatPreset;
        flatPreset.name = "Flat";
        flatPreset.description = "Flat frequency response";
        presetChangedCallback_("Flat", flatPreset);
    }

    return true;
}

bool Equalizer::loadVocalPreset() {
    Logger::info("Equalizer: Loading vocal preset");

    std::lock_guard<std::mutex> lock(bandsMutex_);
    for (auto& band : bands_) {
        if (band.frequency < 200) {
            band.gain = -6.0f;  // Reduce bass
        } else if (band.frequency >= 1000 && band.frequency <= 4000) {
            band.gain = 3.0f;   // Boost vocal range
        } else if (band.frequency > 8000) {
            band.gain = -3.0f;  // Reduce harsh highs
        } else {
            band.gain = 0.0f;
        }
        computeBiquadCoefficients(band);
    }

    if (config_.enableGPUAcceleration && gpuProcessor_) {
        updateGPUFilters();
    }

    currentPreset_ = "Vocal";
    return true;
}

bool Equalizer::loadBassBoostPreset() {
    Logger::info("Equalizer: Loading bass boost preset");

    std::lock_guard<std::mutex> lock(bandsMutex_);
    for (auto& band : bands_) {
        if (band.frequency < 200) {
            band.gain = 8.0f;   // Strong bass boost
        } else if (band.frequency < 500) {
            band.gain = 4.0f;   // Moderate bass boost
        } else {
            band.gain = 0.0f;
        }
        computeBiquadCoefficients(band);
    }

    if (config_.enableGPUAcceleration && gpuProcessor_) {
        updateGPUFilters();
    }

    currentPreset_ = "Bass Boost";
    return true;
}

Equalizer::EqualizerStatistics Equalizer::getStatistics() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    EqualizerStatistics stats = statistics_;

    // Calculate current values
    stats.currentLevel = currentLevel_.load();
    stats.peakLevel = peakLevel_.load();
    stats.averageLevel = averageLevel_.load();
    stats.activeBands = std::count_if(bands_.begin(), bands_.end(),
                                       [](const FilterBand& band) { return band.enabled && !band.bypassed; });

    return stats;
}

void Equalizer::resetStatistics() {
    std::lock_guard<std::mutex> lock(statsMutex_);
    statistics_ = EqualizerStatistics{};
    statistics_.startTime = std::chrono::steady_clock::now();
    statistics_.lastActivity = statistics_.startTime;
    statistics_.minLatency = 1000.0f;
}

bool Equalizer::isHealthy() const {
    if (!initialized_.load()) {
        return false;
    }

    auto stats = getStatistics();
    return (stats.averageLatency < config_.maxLatencyMs &&
            stats.totalHarmonicDistortion < 0.01f &&  // < 1% THD
            stats.droppedFrames == 0);
}

// Private methods implementation
void Equalizer::initializeFilterBands() {
    bands_.clear();
    bands_.resize(config_.numBands);

    for (uint32_t i = 0; i < config_.numBands; ++i) {
        FilterBand& band = bands_[i];
        band.bandIndex = i;
        band.frequency = indexToFrequency(i);
        band.gain = 0.0f;
        band.Q = 1.0f;
        band.type = FilterType::BELL;
        band.slope = FilterSlope::SLOPE_24_DB;
        band.enabled = true;
        band.bypassed = false;

        // Initialize delay line and history
        band.delayLine.fill(0.0f);
        band.history.fill(0.0f);

        // Compute initial coefficients
        computeBiquadCoefficients(band);
    }

    Logger::info("Equalizer: Initialized {} filter bands", config_.numBands);
}

void Equalizer::initializeFrequencyScale() {
    frequencyScale_.clear();
    frequencyScale_.resize(config_.numBands);

    switch (config_.frequencyScale) {
        case FrequencyScale::LOGARITHMIC:
            computeFrequencyScale();
            break;
        case FrequencyScale::LINEAR:
            for (uint32_t i = 0; i < config_.numBands; ++i) {
                frequencyScale_[i] = MIN_FREQUENCY + (MAX_FREQUENCY - MIN_FREQUENCY) * i / (config_.numBands - 1);
            }
            break;
        case FrequencyScale::OCTAVE:
            for (uint32_t i = 0; i < config_.numBands; ++i) {
                float octave = std::log2(MIN_FREQUENCY) + i * std::log2(2.0f) / 3.0f;  // 1/3 octave
                frequencyScale_[i] = std::exp2(octave);
            }
            break;
        default:
            computeFrequencyScale();
            break;
    }
}

void Equalizer::computeFrequencyScale() {
    const float minLog = std::log10(MIN_FREQUENCY);
    const float maxLog = std::log10(MAX_FREQUENCY);
    const float logRange = maxLog - minLog;

    for (uint32_t i = 0; i < config_.numBands; ++i) {
        float logFreq = minLog + (static_cast<float>(i) / static_cast<float>(config_.numBands - 1)) * logRange;
        frequencyScale_[i] = std::pow(10.0f, logFreq);
    }
}

void Equalizer::computeBiquadCoefficients(FilterBand& band) {
    const float fs = static_cast<float>(config_.sampleRate);
    const float w0 = 2.0f * M_PI * band.frequency / fs;
    const float cosw0 = std::cos(w0);
    const float sinw0 = std::sin(w0);
    const float alpha = sinw0 / (2.0f * band.Q);
    const float A = dbToLinear(band.gain);

    float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f;
    float a0 = 1.0f, a1 = 0.0f, a2 = 0.0f;

    switch (band.type) {
        case FilterType::BELL: {
            b0 = 1.0f + alpha * A;
            b1 = -2.0f * cosw0;
            b2 = 1.0f - alpha * A;
            a0 = 1.0f + alpha;
            a1 = -2.0f * cosw0;
            a2 = 1.0f - alpha;
            break;
        }
        case FilterType::LOW_SHELF: {
            b0 = A * ((A + 1.0f) + (A - 1.0f) * cosw0 + 2.0f * std::sqrt(A) * alpha);
            b1 = -2.0f * A * ((A - 1.0f) + (A + 1.0f) * cosw0);
            b2 = A * ((A + 1.0f) + (A - 1.0f) * cosw0 - 2.0f * std::sqrt(A) * alpha);
            a0 = (A + 1.0f) + (A - 1.0f) * cosw0 + 2.0f * std::sqrt(A) * alpha;
            a1 = -2.0f * ((A - 1.0f) + (A + 1.0f) * cosw0);
            a2 = (A + 1.0f) + (A - 1.0f) * cosw0 - 2.0f * std::sqrt(A) * alpha;
            break;
        }
        case FilterType::HIGH_SHELF: {
            b0 = A * ((A + 1.0f) - (A - 1.0f) * cosw0 + 2.0f * std::sqrt(A) * alpha);
            b1 = 2.0f * A * ((A - 1.0f) - (A + 1.0f) * cosw0);
            b2 = A * ((A + 1.0f) - (A - 1.0f) * cosw0 - 2.0f * std::sqrt(A) * alpha);
            a0 = (A + 1.0f) - (A - 1.0f) * cosw0 + 2.0f * std::sqrt(A) * alpha;
            a1 = 2.0f * ((A - 1.0f) - (A + 1.0f) * cosw0);
            a2 = (A + 1.0f) - (A - 1.0f) * cosw0 - 2.0f * std::sqrt(A) * alpha;
            break;
        }
        default:
            // Default to bell filter
            b0 = 1.0f + alpha * A;
            b1 = -2.0f * cosw0;
            b2 = 1.0f - alpha * A;
            a0 = 1.0f + alpha;
            a1 = -2.0f * cosw0;
            a2 = 1.0f - alpha;
            break;
    }

    // Normalize by a0
    band.bCoefficients[0] = b0 / a0;
    band.bCoefficients[1] = b1 / a0;
    band.bCoefficients[2] = b2 / a0;
    band.aCoefficients[0] = 1.0f;
    band.aCoefficients[1] = a1 / a0;
    band.aCoefficients[2] = a2 / a0;
}

void Equalizer::processChannelData(const float* input, float* output, size_t numSamples, uint16_t channel) {
    if (!input || !output) {
        return;
    }

    std::lock_guard<std::mutex> lock(bandsMutex_);

    // Copy input to output initially
    std::copy(input, input + numSamples, output);

    // Process each filter band
    for (auto& band : bands_) {
        if (band.enabled && !band.bypassed) {
            processBiquadSection(output, output, numSamples, band);
        }
    }

    // Update level statistics
    updateLevelStatistics(output, numSamples);
}

void Equalizer::processBiquadSection(const float* input, float* output, size_t numSamples,
                                     FilterBand& band) {
    const float b0 = band.bCoefficients[0];
    const float b1 = band.bCoefficients[1];
    const float b2 = band.bCoefficients[2];
    const float a1 = band.aCoefficients[1];
    const float a2 = band.aCoefficients[2];

    float x1 = band.history[0];
    float x2 = band.history[1];
    float y1 = band.history[2];
    float y2 = band.history[3];

    for (size_t n = 0; n < numSamples; ++n) {
        float x0 = input[n];
        float y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;

        output[n] = y0;

        // Update delay line
        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;
    }

    // Store delay line state
    band.history[0] = x1;
    band.history[1] = x2;
    band.history[2] = y1;
    band.history[3] = y2;
}

float Equalizer::computeFrequencyResponse(float frequency, const FilterBand& band) const {
    const float fs = static_cast<float>(config_.sampleRate);
    const float w = 2.0f * M_PI * frequency / fs;
    const float cosw = std::cos(w);
    const float sinw = std::sin(w);

    const float b0 = band.bCoefficients[0];
    const float b1 = band.bCoefficients[1];
    const float b2 = band.bCoefficients[2];
    const float a1 = band.aCoefficients[1];
    const float a2 = band.aCoefficients[2];

    // Compute complex frequency response
    std::complex<float> numerator = b0 + b1 * std::exp(-std::complex<float>(0, w)) +
                                   b2 * std::exp(-std::complex<float>(0, 2*w));
    std::complex<float> denominator = 1.0f + a1 * std::exp(-std::complex<float>(0, w)) +
                                      a2 * std::exp(-std::complex<float>(0, 2*w));

    std::complex<float> response = numerator / denominator;
    float magnitude = std::abs(response);

    return linearToDb(magnitude);
}

void Equalizer::updateLevelStatistics(const float* audio, size_t numSamples) {
    if (!audio || numSamples == 0) {
        return;
    }

    float currentSum = 0.0f;
    float peak = 0.0f;

    for (size_t i = 0; i < numSamples; ++i) {
        float sample = std::abs(audio[i]);
        currentSum += sample;
        peak = std::max(peak, sample);
    }

    float current = currentSum / static_cast<float>(numSamples);

    // Update statistics with exponential smoothing
    currentLevel_.store(currentLevel_.load() * 0.9f + current * 0.1f);
    peakLevel_.store(std::max(peakLevel_.load(), peak));
    averageLevel_.store(averageLevel_.load() * 0.995f + current * 0.005f);
}

void Equalizer::initializeBuffers() {
    std::lock_guard<std::mutex> lock(buffersMutex_);

    size_t bufferSize = std::max(config_.bufferSize, static_cast<uint32_t>(4096));
    buffers_.inputBuffer.resize(bufferSize);
    buffers_.outputBuffer.resize(bufferSize);
    buffers_.intermediateBuffer.resize(bufferSize);
    buffers_.lookaheadBuffer.resize(config_.lookaheadSamples);
    buffers_.fftBuffer.resize(config_.spectrumSize);
    buffers_.spectrumBuffer.resize(config_.spectrumSize / 2 + 1);
}

bool Equalizer::initializeGPUResources() {
    try {
        gpuProcessor_ = std::make_unique<GPUProcessor>(config_);
        if (!gpuProcessor_->initialize()) {
            Logger::error("Equalizer: Failed to initialize GPU processor");
            return false;
        }

        // Upload initial filter coefficients
        if (!gpuProcessor_->uploadFilterCoefficients(bands_)) {
            Logger::error("Equalizer: Failed to upload filter coefficients to GPU");
            return false;
        }

        Logger::info("Equalizer: GPU resources initialized successfully");
        return true;

    } catch (const std::exception& e) {
        Logger::error("Equalizer: GPU initialization error: {}", e.what());
        return false;
    }
}

void Equalizer::cleanupGPUResources() {
    if (gpuProcessor_) {
        gpuProcessor_->shutdown();
        gpuProcessor_.reset();
    }

    gpuFilterCoefficients_ = nullptr;
    gpuFilterStates_ = nullptr;
    gpuProcessingBuffer_ = nullptr;
    gpuMemorySize_ = 0;
}

bool Equalizer::updateGPUBand(uint32_t bandIndex) {
    if (!gpuProcessor_ || !isValidBandIndex(bandIndex)) {
        return false;
    }

    // This would update a single band on the GPU
    // Implementation depends on GPU backend
    return gpuProcessor_->uploadFilterCoefficients(bands_);
}

bool Equalizer::updateGPUFilters() {
    if (!gpuProcessor_) {
        return false;
    }

    return gpuProcessor_->uploadFilterCoefficients(bands_);
}

// Utility methods
float Equalizer::dbToLinear(float db) const {
    return std::pow(10.0f, db / 20.0f);
}

float Equalizer::linearToDb(float linear) const {
    return 20.0f * std::log10(std::max(linear, 1e-10f));
}

float Equalizer::indexToFrequency(uint32_t index) const {
    if (index >= frequencyScale_.size()) {
        return 1000.0f; // Default to 1kHz
    }
    return frequencyScale_[index];
}

bool Equalizer::isValidBandIndex(uint32_t index) const {
    return index < bands_.size();
}

void Equalizer::setError(const std::string& error) const {
    lastError_ = error;
    Logger::error("Equalizer: {}", error);
}

void Equalizer::applyDithering(float* audio, size_t numSamples) {
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

// GPUProcessor Implementation
Equalizer::GPUProcessor::GPUProcessor(const EqualizerConfig& config) : config_(config) {
    Logger::info("Equalizer::GPU: Initializing GPU processor");
}

Equalizer::GPUProcessor::~GPUProcessor() {
    shutdown();
    Logger::info("Equalizer::GPU: GPU processor destroyed");
}

bool Equalizer::GPUProcessor::initialize() {
    if (initialized_.load()) {
        return true;
    }

    try {
        // Try to initialize CUDA first
        if (initializeCUDA()) {
            Logger::info("Equalizer::GPU: CUDA backend initialized");
            initialized_.store(true);
            return true;
        }

        // Fall back to OpenCL
        if (initializeOpenCL()) {
            Logger::info("Equalizer::GPU: OpenCL backend initialized");
            initialized_.store(true);
            return true;
        }

        // Fall back to Vulkan
        if (initializeVulkan()) {
            Logger::info("Equalizer::GPU: Vulkan backend initialized");
            initialized_.store(true);
            return true;
        }

        Logger::error("Equalizer::GPU: No GPU backend available");
        return false;

    } catch (const std::exception& e) {
        Logger::error("Equalizer::GPU: Initialization failed: {}", e.what());
        return false;
    }
}

void Equalizer::GPUProcessor::shutdown() {
    initialized_.store(false);

    // Cleanup GPU resources
    if (coefficientBuffer_) {
        // Free GPU memory
        coefficientBuffer_ = nullptr;
    }
    if (stateBuffer_) {
        stateBuffer_ = nullptr;
    }
    if (inputBuffer_) {
        inputBuffer_ = nullptr;
    }
    if (outputBuffer_) {
        outputBuffer_ = nullptr;
    }

    Logger::info("Equalizer::GPU: GPU processor shutdown");
}

bool Equalizer::GPUProcessor::isInitialized() const {
    return initialized_.load();
}

bool Equalizer::GPUProcessor::uploadFilterCoefficients(const std::vector<FilterBand>& bands) {
    if (!initialized_.load()) {
        return false;
    }

    // This would upload filter coefficients to GPU memory
    // Implementation depends on the GPU backend being used
    return true;
}

bool Equalizer::GPUProcessor::processAudio(const float* input, float* output, size_t numSamples, uint16_t channels) {
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
            // Fallback to CPU processing
            return false;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<float, std::milli>(endTime - startTime);
    processingTime_.store(duration.count());

    return true;
}

bool Equalizer::GPUProcessor::initializeCUDA() {
#ifdef VORTEX_GPU_CUDA
    try {
        // Initialize CUDA context and resources
        // This is a placeholder - real implementation would:
        // 1. Initialize CUDA
        // 2. Create context and command queue
        // 3. Allocate GPU memory for coefficients, states, and buffers
        // 4. Compile CUDA kernels for filter processing

        gpuContext_.device = nullptr; // CUDA device handle
        gpuContext_.context = nullptr; // CUDA context
        gpuContext_.initialized = true;

        return true;

    } catch (const std::exception& e) {
        Logger::error("Equalizer::GPU: CUDA initialization failed: {}", e.what());
        return false;
    }
#else
    Logger::warn("Equalizer::GPU: CUDA support not compiled");
    return false;
#endif
}

bool Equalizer::GPUProcessor::initializeOpenCL() {
#ifdef VORTEX_GPU_OPENCL
    try {
        // Initialize OpenCL context and resources
        // This is a placeholder - real implementation would:
        // 1. Enumerate OpenCL platforms and devices
        // 2. Create context and command queue
        // 3. Build OpenCL kernels for filter processing
        // 4. Allocate OpenCL buffers

        gpuContext_.device = nullptr; // OpenCL device
        gpuContext_.context = nullptr; // OpenCL context
        gpuContext_.initialized = true;

        return true;

    } catch (const std::exception& e) {
        Logger::error("Equalizer::GPU: OpenCL initialization failed: {}", e.what());
        return false;
    }
#else
    Logger::warn("Equalizer::GPU: OpenCL support not compiled");
    return false;
#endif
}

bool Equalizer::GPUProcessor::initializeVulkan() {
    // Vulkan implementation would go here
    Logger::warn("Equalizer::GPU: Vulkan support not implemented");
    return false;
}

void Equalizer::GPUProcessor::processWithCUDA(const float* input, float* output, size_t numSamples, uint16_t channels) {
    // Placeholder for CUDA processing implementation
    // Real implementation would:
    // 1. Copy input data to GPU
    // 2. Launch CUDA kernels for filter processing
    // 3. Copy output data back to CPU
    // 4. Synchronize if necessary

    // For now, just copy input to output (no processing)
    std::copy(input, input + numSamples * channels, output);
}

void Equalizer::GPUProcessor::processWithOpenCL(const float* input, float* output, size_t numSamples, uint16_t channels) {
    // Placeholder for OpenCL processing implementation
    // Similar to CUDA but using OpenCL APIs

    // For now, just copy input to output (no processing)
    std::copy(input, input + numSamples * channels, output);
}

} // namespace vortex