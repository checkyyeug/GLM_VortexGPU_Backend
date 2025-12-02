#include "core/dsp/vu_meter.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>

#ifdef VORTEX_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace vortex::core::dsp {

VUMeter::VUMeter()
    : initialized_(false)
    , historyEnabled_(false)
    , gpuAudioBuffer_(nullptr)
    , gpuLevelsBuffer_(nullptr)
    , gpuInitialized_(false)
    , totalFramesProcessed_(0)
    , totalProcessingTime_(0.0)
    , gpuFramesProcessed_(0)
    , cpuFramesProcessed_(0) {

    // Initialize default configuration
    config_ = Config{};

    // Initialize level arrays
    const size_t maxChannels = 8;
    currentLevels_.resize(maxChannels, 0.0f);
    peakLevels_.resize(maxChannels, 0.0f);
    smoothedLevels_.resize(maxChannels, 0.0f);
    attackCoefficients_.resize(maxChannels, 0.0f);
    releaseCoefficients_.resize(maxChannels, 0.0f);
    peakHoldCounters_.resize(maxChannels, 0);
    peakHoldValues_.resize(maxChannels, 0.0f);
}

VUMeter::~VUMeter() {
    shutdown();
}

bool VUMeter::initialize(const Config& config) {
    if (initialized_) {
        Logger::warn("VUMeter already initialized");
        return true;
    }

    // Validate configuration
    if (config.sampleRate <= 0.0f || config.channels <= 0) {
        Logger::error("Invalid VU meter configuration");
        return false;
    }

    config_ = config;

    Logger::info("Initializing VUMeter: {} Hz, {} channels, {} meter type",
                 config_.sampleRate, config_.channels,
                 static_cast<int>(config_.meterType));

    try {
        // Initialize level state
        if (!initializeLevelState()) {
            Logger::error("Failed to initialize level state");
            return false;
        }

        // Initialize ballistics
        if (!initializeBallistics()) {
            Logger::error("Failed to initialize ballistics");
            return false;
        }

        // Initialize history tracking
        if (!initializeHistory()) {
            Logger::error("Failed to initialize history tracking");
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
        Logger::info("VUMeter initialized successfully ({} processing)",
                     config_.processingMode == ProcessingMode::GPU ? "GPU" : "CPU");
        return true;

    } catch (const std::exception& e) {
        Logger::error("Exception during VU meter initialization: {}", e.what());
        return false;
    }
}

bool VUMeter::initialize(float sampleRate, size_t bufferSize, int channels) {
    Config config;
    config.sampleRate = sampleRate;
    config.channels = channels;
    config.meterType = MeterType::RMS;
    config.referenceLevel = ReferenceLevel::dBFS_20;
    config.attackTime = 0.001f;
    config.releaseTime = 0.1f;
    return initialize(config);
}

void VUMeter::shutdown() {
    if (!initialized_) {
        return;
    }

    Logger::info("Shutting down VUMeter");

    // Cleanup GPU resources
    cleanupGPUResources();

    // Clear buffers
    currentLevels_.clear();
    peakLevels_.clear();
    smoothedLevels_.clear();
    attackCoefficients_.clear();
    releaseCoefficients_.clear();
    peakHoldCounters_.clear();
    peakHoldValues_.clear();

    // Clear history
    levelHistory_.leftHistory.clear();
    levelHistory_.rightHistory.clear();
    levelHistory_.monoHistory.clear();
    levelHistory_.peakHistory.clear();
    levelHistory_.timestamps.clear();

    initialized_ = false;

    // Log final statistics
    if (totalFramesProcessed_ > 0) {
        double avgProcessingTime = totalProcessingTime_ / totalFramesProcessed_;
        Logger::info("VUMeter stats: {} frames processed, avg time: {:.3f}μs, GPU frames: {}, CPU frames: {}",
                     totalFramesProcessed_.load(), avgProcessingTime,
                     gpuFramesProcessed_.load(), cpuFramesProcessed_.load());
    }
}

VUMeter::VUReading VUMeter::processAudio(const float* audioData, size_t numSamples) {
    VUReading reading;

    if (!processAudio(audioData, numSamples, reading)) {
        // Return default reading on error
        reading.leftLevel = -60.0f;
        reading.rightLevel = -60.0f;
        reading.monoLevel = -60.0f;
        reading.peakLevel = -60.0f;
        reading.averageLevel = -60.0f;
        reading.loudnessLUFS = -23.0f;  // EBU R128 standard
        reading.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        reading.isValid = false;
        reading.isClipping = false;
        reading.stereoBalance = 0.0f;
        reading.dynamicRange = 0.0f;
    }

    return reading;
}

bool VUMeter::processAudio(const float* audioData, size_t numSamples, VUMeter& reading) {
    if (!initialized_) {
        Logger::error("VUMeter not initialized");
        return false;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    bool success = false;

    try {
        if (config_.processingMode == ProcessingMode::GPU && gpuInitialized_) {
            success = processAudioGPU(audioData, numSamples, reading);
        } else {
            success = processAudioCPU(audioData, numSamples, reading);
        }

    } catch (const std::exception& e) {
        Logger::error("Exception during VU meter processing: {}", e.what());
        success = false;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto processingTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

    updatePerformanceStats(static_cast<double>(processingTime), config_.processingMode == ProcessingMode::GPU && gpuInitialized_);

    return success;
}

VUMeter::VUReading VUMeter::getCurrentReading() const {
    VUReading reading;

    if (!initialized_) {
        reading.leftLevel = -60.0f;
        reading.rightLevel = -60.0f;
        reading.monoLevel = -60.0f;
        reading.peakLevel = -60.0f;
        reading.averageLevel = -60.0f;
        reading.loudnessLUFS = -23.0f;
        reading.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        reading.isValid = false;
        reading.isClipping = false;
        reading.stereoBalance = 0.0f;
        reading.dynamicRange = 0.0f;
        return reading;
    }

    reading.leftLevel = currentLevels_[0];
    reading.rightLevel = (config_.channels >= 2) ? currentLevels_[1] : currentLevels_[0];
    reading.monoLevel = (reading.leftLevel + reading.rightLevel) / 2.0f;
    reading.peakLevel = peakLevels_[0];
    reading.averageLevel = smoothedLevels_[0];
    reading.loudnessLUFS = -23.0f; // Would need LUFS calculation
    reading.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    reading.isValid = true;
    reading.isClipping = isClipping(currentLevels_.data(), config_.channels, 0.0f);

    // Calculate stereo metrics
    calculateStereoMetrics(currentLevels_.data(), reading.monoLevel, reading.stereoBalance);
    calculateDynamicRange(reading, reading.dynamicRange);

    return reading;
}

std::vector<float> VUMeter::getCurrentLevels() const {
    std::vector<float> levels;

    if (!initialized_) {
        return levels;
    }

    levels.push_back(currentLevels_[0]); // Left
    if (config_.channels >= 2) {
        levels.push_back(currentLevels_[1]); // Right
    }
    levels.push_back((currentLevels_[0] + ((config_.channels >= 2) ? currentLevels_[1] : currentLevels_[0])) / 2.0f); // Mono
    levels.push_back(peakLevels_[0]); // Peak

    return levels;
}

bool VUMeter::processAudioCPU(const float* audioData, size_t numSamples, VUReading& reading) {
    // Calculate input levels based on meter type
    std::vector<float> inputLevels(config_.channels);

    switch (config_.meterType) {
        case MeterType::Peak:
            calculatePeakLevels(audioData, numSamples, inputLevels.data());
            break;

        case MeterType::RMS:
            calculateRMSLevels(audioData, numSamples, inputLevels.data());
            break;

        case MeterType::VU:
            calculateVULevels(audioData, numSamples, inputLevels.data());
            break;

        case MeterType::KSystem:
            calculateKSystemLevels(audioData, numSamples, inputLevels.data());
            break;

        case MeterType::LUFS:
            calculateLUFSLevels(audioData, numSamples, inputLevels.data());
            break;

        case MeterType::Digital:
            calculateDigitalLevels(audioData, numSamples, inputLevels.data());
            break;

        case MeterType::PPM:
            calculatePPMLevels(audioData, numSamples, inputLevels.data());
            break;
    }

    // Apply ballistics
    std::vector<float> processedLevels(config_.channels);
    applyBallistics(processedLevels.data(), inputLevels.data(), config_.channels);

    // Update peak hold
    updatePeakHold(peakLevels_.data(), processedLevels.data(), config_.channels);

    // Convert to display levels (apply reference offset and scaling)
    for (int ch = 0; ch < config_.channels; ++ch) {
        currentLevels_[ch] = levelToDisplay(processedLevels[ch]);
    }

    // Fill reading structure
    reading.leftLevel = currentLevels_[0];
    reading.rightLevel = (config_.channels >= 2) ? currentLevels_[1] : currentLevels_[0];
    reading.monoLevel = (reading.leftLevel + reading.rightLevel) / 2.0f;
    reading.peakLevel = levelToDisplay(peakLevels_[0]);
    reading.averageLevel = levelToDisplay(smoothedLevels_[0]);
    reading.loudnessLUFS = -23.0f; // Placeholder
    reading.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    reading.isValid = true;
    reading.isClipping = isClipping(currentLevels_.data(), config_.channels, 0.0f);

    // Calculate stereo metrics
    calculateStereoMetrics(currentLevels_.data(), reading.monoLevel, reading.stereoBalance);
    calculateDynamicRange(reading, reading.dynamicRange);

    // Add to history
    if (config_.enableHistory) {
        addToHistory(reading);
    }

    return true;
}

bool VUMeter::processAudioGPU(const float* audioData, size_t numSamples, VUReading& reading) {
#ifdef VORTEX_ENABLE_CUDA
    if (!gpuInitialized_) {
        return processAudioCPU(audioData, numSamples, reading);
    }

    // GPU implementation would go here
    // For now, fallback to CPU
    return processAudioCPU(audioData, numSamples, reading);
#else
    return processAudioCPU(audioData, numSamples, reading);
#endif
}

void VUMeter::calculatePeakLevels(const float* audioData, size_t numSamples, float* levels) {
    for (int ch = 0; ch < config_.channels; ++ch) {
        float peak = 0.0f;

        for (size_t i = ch; i < numSamples * config_.channels; i += config_.channels) {
            float absVal = std::abs(audioData[i]);
            if (absVal > peak) {
                peak = absVal;
            }
        }

        levels[ch] = peak;
    }
}

void VUMeter::calculateRMSLevels(const float* audioData, size_t numSamples, float* levels) {
    for (int ch = 0; ch < config_.channels; ++ch) {
        float sum = 0.0f;

        for (size_t i = ch; i < numSamples * config_.channels; i += config_.channels) {
            sum += audioData[i] * audioData[i];
        }

        float rms = (numSamples > 0) ? std::sqrt(sum / (numSamples / config_.channels)) : 0.0f;
        levels[ch] = rms;
    }
}

void VUMeter::calculateVULevels(const float* audioData, size_t numSamples, float* levels) {
    // VU meters have 300ms integration time and specific ballistics
    float integrationConstant = 1.0f - std::exp(-1.0f / (0.3f * config_.sampleRate));

    for (int ch = 0; ch < config_.channels; ++ch) {
        float level = 0.0f;

        for (size_t i = ch; i < numSamples * config_.channels; i += config_.channels) {
            float absVal = std::abs(audioData[i]);
            level = level * integrationConstant + (1.0f - integrationConstant) * absVal;
        }

        // Convert to dBFS (0 dBFS = 1.0)
        levels[ch] = (level > 0.0f) ? 20.0f * std::log10(level) : -60.0f;
    }
}

void VUMeter::calculateKSystemLevels(const float* audioData, size_t numSamples, float* levels) {
    // Calculate K-system levels based on reference level
    float referenceOffset = getReferenceOffset();

    for (int ch = 0; ch < config_.channels; ++ch) {
        float level = 0.0f;

        for (size_t i = ch; i < numSamples * config_.channels; i += config_.channels) {
            float absVal = std::abs(audioData[i]);
            if (absVal > level) {
                level = absVal;
            }
        }

        // Convert to dBFS, then apply K-system reference
        float dBFS = (level > 0.0f) ? 20.0f * std::log10(level) : -60.0f;
        levels[ch] = dBFS + referenceOffset;
    }
}

void VUMeter::calculateLUFSLevels(const float* audioData, size_t numSamples, float* levels) {
    // Simplified LUFS calculation (EBU R128)
    // Full implementation would use gating and frequency weighting
    for (int ch = 0; ch < config_.channels; ++ch) {
        float sum = 0.0f;

        for (size_t i = ch; i < numSamples * config_.channels; i += config_.channels) {
            sum += audioData[i] * audioData[i];
        }

        float rms = (numSamples > 0) ? std::sqrt(sum / (numSamples / config_.channels)) : 0.0f;
        // LUFS conversion (simplified)
        float loudness = (rms > 0.0f) ? -0.691f + 10.0f * std::log10(rms) : -60.0f;
        levels[ch] = loudness + 23.0f; // LUFS = -0.691 + 10*log10(rms) + 23.0
    }
}

void VUMeter::calculateDigitalLevels(const float* audioData, size_t numSamples, float* levels) {
    // Digital peak meter (instantaneous)
    for (int ch = 0; ch < config_.channels; ++ch) {
        float peak = 0.0f;

        for (size_t i = ch; i < numSamples * config_.channels; i += config_.channels) {
            float absVal = std::abs(audioData[i]);
            if (absVal > peak) {
                peak = absVal;
            }
        }

        // Convert to dBFS
        levels[ch] = (peak > 0.0f) ? 20.0f * std::log10(peak) : -60.0f;
    }
}

void VUMeter::calculatePPMLevels(const float* audioData, size_t numSamples, float* levels) {
    // PPM (Program Peak Meter) - BBC standard with specific attack/release
    float attackTime = 0.01f;  // 10ms attack
    float releaseTime = 2.5f;   // 2.5s release
    float sampleRate = config_.sampleRate;

    float attackConstant = std::exp(-1.0f / (attackTime * sampleRate));
    float releaseConstant = std::exp(-1.0f / (releaseTime * sampleRate));

    for (int ch = 0; ch < config_.channels; ++ch) {
        float level = 0.0f;

        for (size_t i = ch; i < numSamples * config_.channels; i += config_.channels) {
            float absVal = std::abs(audioData[i]);

            if (absVal > level) {
                level = level + (absVal - level) * attackConstant;
            } else {
                level = level + (absVal - level) * releaseConstant;
            }
        }

        levels[ch] = (level > 0.0f) ? 20.0f * std::log10(level) : -60.0f;
    }
}

void VUMeter::applyBallistics(float* currentLevels, const float* inputLevels, size_t channels) {
    for (size_t i = 0; i < channels; ++i) {
        // Apply attack/release ballistics
        float diff = inputLevels[i] - currentLevels[i];
        if (diff > 0) {
            currentLevels[i] += diff * attackCoefficients_[i];
        } else {
            currentLevels[i] += diff * releaseCoefficients_[i];
        }
    }
}

void VUMeter::updatePeakHold(float* peakLevels, const float* inputLevels, size_t channels) {
    uint32_t holdSamples = static_cast<uint32_t>(config_.holdTime * config_.sampleRate);

    for (size_t i = 0; i < channels; ++i) {
        if (inputLevels[i] > peakLevels_[i]) {
            peakLevels_[i] = inputLevels[i];
            peakHoldCounters_[i] = 0;
        } else {
            if (peakHoldCounters_[i] < holdSamples) {
                peakHoldCounters_[i]++;
            } else {
                peakLevels_[i] *= 0.999f; // Slow decay
                peakHoldCounters_[i]++;
            }
        }
    }
}

void VUMeter::calculateStereoMetrics(const float* levels, float& monoLevel, float& balance) {
    if (config_.channels < 2) {
        monoLevel = levels[0];
        balance = 0.0f;
        return;
    }

    float leftLevel = levels[0];
    float rightLevel = levels[1];

    // Mono level (average)
    monoLevel = (leftLevel + rightLevel) / 2.0f;

    // Stereo balance (-1 to +1, where 0 = center)
    if (std::abs(leftLevel + rightLevel) > 0.001f) {
        balance = (rightLevel - leftLevel) / (std::abs(leftLevel) + std::abs(rightLevel));
    } else {
        balance = 0.0f;
    }
}

void VUMeter::addToHistory(const VUReading& reading) {
    if (!config_.enableHistory) {
        return;
    }

    // Add current reading to history at current index
    size_t index = levelHistory_.currentIndex;

    // Resize vectors if needed
    if (levelHistory_.leftHistory.size() < config_.historySize) {
        levelHistory_.leftHistory.resize(config_.historySize);
        levelHistory_.rightHistory.resize(config_.historySize);
        levelHistory_.monoHistory.resize(config_.historySize);
        levelHistory_.peakHistory.resize(config_.historySize);
        levelHistory_.timestamps.resize(config_.historySize);
    }

    levelHistory_.leftHistory[index] = reading.leftLevel;
    levelHistory_.rightHistory[index] = reading.rightLevel;
    levelHistory_.monoHistory[index] = reading.monoLevel;
    levelHistory_.peakHistory[index] = reading.peakLevel;
    levelHistory_.timestamps[index] = reading.timestamp;

    // Update index and wrap around
    index++;
    if (index >= config_.historySize) {
        index = 0;
        levelHistory_.isFull = true;
    }
    levelHistory_.currentIndex = index;
}

void VUMeter::calculateDynamicRange(const VUReading& reading, float& dynamicRange) {
    // Simplified dynamic range calculation
    float minLevel = std::min({reading.leftLevel, reading.rightLevel, -60.0f});
    float maxLevel = std::max({reading.leftLevel, reading.rightLevel, reading.peakLevel});
    dynamicRange = maxLevel - minLevel;
}

bool VUMeter::initializeLevelState() {
    currentLevels_.resize(config_.channels);
    peakLevels_.resize(config_.channels);
    smoothedLevels_.resize(config_.channels);
    attackCoefficients_.resize(config_.channels);
    releaseCoefficients_.resize(config_.channels);
    peakHoldCounters_.resize(config_.channels);
    peakHoldValues_.resize(config_.channels);

    // Initialize to minimum level
    std::fill(currentLevels_.begin(), currentLevels_.end(), -60.0f);
    std::fill(peakLevels_.begin(), peakLevels_.end(), -60.0f);
    std::fill(smoothedLevels_.begin(), smoothedLevels_.end(), -60.0f);
    std::fill(peakHoldCounters_.begin(), peakHoldCounters_.end(), 0);
    std::fill(peakHoldValues_.begin(), peakHoldValues_.end(), 0.0f);

    // Calculate ballistics coefficients
    for (int ch = 0; ch < config_.channels; ++ch) {
        attackCoefficients_[ch] = std::exp(-1.0f / (config_.attackTime * config_.sampleRate));
        releaseCoefficients_[ch] = std::exp(-1.0f / (config_.releaseTime * config_.sampleRate));
    }

    return true;
}

bool VUMeter::initializeBallistics() {
    return true; // Handled in initializeLevelState
}

bool VUMeter::initializeHistory() {
    if (!config_.enableHistory) {
        return true;
    }

    levelHistory_.leftHistory.reserve(config_.historySize);
    levelHistory_.rightHistory.reserve(config_.historySize);
    levelHistory_.monoHistory.reserve(config_.historySize);
    levelHistory_.peakHistory.reserve(config_.historySize);
    levelHistory_.timestamps.reserve(config_.historySize);

    levelHistory_.currentIndex = 0;
    levelHistory_.isFull = false;

    return true;
}

bool VUMeter::initializeGPUResources() {
#ifdef VORTEX_ENABLE_CUDA
    try {
        size_t audioSize = config_.bufferSize * config_.channels * sizeof(float);
        size_t levelsSize = config_.channels * sizeof(float);

        // Allocate GPU memory buffers
        cudaError_t result;

        result = cudaMalloc(&gpuAudioBuffer_, audioSize);
        if (result != CUDA_SUCCESS) {
            Logger::error("Failed to allocate GPU audio buffer: {}", result);
            return false;
        }

        result = cudaMalloc(&gpuLevelsBuffer_, levelsSize);
        if (result != CUDA_SUCCESS) {
            Logger::error("Failed to allocate GPU levels buffer: {}", result);
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

void VUMeter::cleanupGPUResources() {
#ifdef VORTEX_ENABLE_CUDA
    if (gpuAudioBuffer_) {
        cudaFree(gpuAudioBuffer_);
        gpuAudioBuffer_ = nullptr;
    }

    if (gpuLevelsBuffer_) {
        cudaFree(gpuLevelsBuffer_);
        gpuLevelsBuffer_ = nullptr;
    }
#endif

    gpuInitialized_ = false;
}

float VUMeter::getReferenceOffset() const {
    switch (config_.referenceLevel) {
        case ReferenceLevel::dBFS_0:   return 0.0f;
        case ReferenceLevel::dBFS_18:  return -18.0f;
        case ReferenceLevel::dBFS_20: return -20.0f;
        case ReferenceLevel::dBFS_24: return -24.0f;
        case ReferenceLevel::VU_0:   return -4.0f;  // 0 VU = +4 dBu = +2.18 dBFS
        case ReferenceLevel::K20:    return -20.0f; // K-20: -20 dBFS = 0 K-unit
        case ReferenceLevel::K14:    return -14.0f; // K-14: -14 dBFS = 0 K-unit
        case ReferenceLevel::K12:    return -12.0f; // K-12: -12 dBFS = 0 K-unit
        default: return 0.0f;
    }
}

float VUMeter::levelToDisplay(float level) const {
    return level + getReferenceOffset();
}

float VUMeter::displayToLevel(float display) const {
    return display - getReferenceOffset();
}

void VUMeter::updatePerformanceStats(double processingTimeMs, bool usedGPU) const {
    totalFramesProcessed_++;
    totalProcessingTime_ += processingTimeMs;

    if (usedGPU) {
        gpuFramesProcessed_++;
    } else {
        cpuFramesProcessed_++;
    }
}

// Setter and getter methods
void VUMeter::setMeterType(MeterType type) {
    config_.meterType = type;
}

VUMeter::MeterType VUMeter::getMeterType() const {
    return config_.meterType;
}

void VUMeter::setReferenceLevel(ReferenceLevel level) {
    config_.referenceLevel = level;
}

VUMeter::ReferenceLevel VUMeter::getReferenceLevel() const {
    return config_.referenceLevel;
}

void VUMeter::setProcessingMode(ProcessingMode mode) {
    config_.processingMode = mode;
}

VUMeter::ProcessingMode VUMeter::getProcessingMode() const {
    return config_.processingMode;
}

bool VUMeter::isGPUAvailable() const {
    return gpuInitialized_;
}

const VUMeter::Config& VUMeter::getConfig() const {
    return config_;
}

std::string VUMeter::getPerformanceStats() const {
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

void VUMeter::reset() {
    if (!initialized_) {
        return;
    }

    // Reset level state
    std::fill(currentLevels_.begin(), currentLevels_.end(), -60.0f);
    std::fill(peakLevels_.begin(), peakLevels_.end(), -60.0f);
    std::fill(smoothedLevels_.begin(), smoothedLevels_.end(), -60.0f);
    std::fill(peakHoldCounters_.begin(), peakHoldCounters_.end(), 0);
    std::fill(peakHoldValues_.begin(), peakHoldValues_.end(), 0.0f);

    // Clear history
    levelHistory_.leftHistory.clear();
    levelHistory_.rightHistory.clear();
    levelHistory_.monoHistory.clear();
    levelHistory_.peakHistory.clear();
    levelHistory_.timestamps.clear();
    levelHistory_.currentIndex = 0;
    levelHistory_.isFull = false;

    // Reset performance counters
    totalFramesProcessed_ = 0;
    totalProcessingTime_ = 0.0;
    gpuFramesProcessed_ = 0;
    cpuFramesProcessed_ = 0;

    Logger::debug("VUMeter reset");
}

void VUMeter::resetPeaks() {
    std::fill(peakLevels_.begin(), peakLevels_.end(), -60.0f);
    std::fill(peakHoldCounters_.begin(), peakHoldCounters_.end(), 0);
    std::fill(peakHoldValues_.begin(), peakHoldValues_.end(), 0.0f);
}

bool VUMeter::isInitialized() const {
    return initialized_;
}

void VUMeter::setAttackTime(float time) {
    config_.attackTime = time;
    if (initialized_) {
        // Recalculate attack coefficients
        for (int ch = 0; ch < config_.channels; ++ch) {
            attackCoefficients_[ch] = std::exp(-1.0f / (config_.attackTime * config_.sampleRate));
        }
    }
}

float VUMeter::getAttackTime() const {
    return config_.attackTime;
}

void VUMeter::setReleaseTime(float time) {
    config_.releaseTime = time;
    if (initialized_) {
        // Recalculate release coefficients
        for (int ch = 0; ch < config_.channels; ++ch) {
            releaseCoefficients_[ch] = std::exp(-1.0f / (config_.releaseTime * config_.sampleRate));
        }
    }
}

float VUMeter::getReleaseTime() const {
    return config_.releaseTime;
}

void VUMeter::setPeakHoldTime(float time) {
    config_.holdTime = time;
    // Peak hold counters are reset on next audio processing
}

float VUMeter::getPeakHoldTime() const {
    return config_.holdTime;
}

void VUMeter::setStereoLinkEnabled(bool enabled) {
    config_.enableStereoLink = enabled;
}

bool VUMeter::isStereoLinkEnabled() const {
    return config_.enableStereoLink;
}

void VUMeter::setTruePeakEnabled(bool enabled) {
    config_.enableTruePeak = enabled;
}

bool VUMeter::isTruePeakEnabled() const {
    return config_.enableTruePeak;
}

const VUMeter::LevelHistory& VUMeter::getLevelHistory() const {
    return levelHistory_;
}

void VUMeter::setHistoryEnabled(bool enabled) {
    config_.enableHistory = enabled;
    if (!enabled) {
        levelHistory_.leftHistory.clear();
        levelHistory_.rightHistory.clear();
        levelHistory_.monoHistory.clear();
        levelHistory_.peakHistory.clear();
        levelHistory_.timestamps.clear();
        levelHistory_.currentIndex = 0;
        levelHistory_.isFull = false;
    }
}

bool VUMeter::isHistoryEnabled() const {
    return config_.enableHistory;
}

// Helper function implementations
float VUMeter::fastExp(float x) const {
    // Fast approximation of exp(x)
    // Using polynomial approximation
    if (x < -87.0f) return 0.0f;
    if (x > 88.0f) return std::exp(x);

    // Polynomial approximation
    float result = 1.0f + x * (1.0f + x * (0.5f + x * (0.1666667f + x * 0.0416667f)));
    return result;
}

float VUMeter::fastLog(float x) const {
    if (x <= 0.0f) return -60.0f;

    // Fast log10 approximation
    return 20.0f * std::log(x);
}

float VUMeter::fastSqrt(float x) const {
    if (x <= 0.0f) return 0.0f;

    // Fast sqrt approximation using bit manipulation
    float xhalf = 0.5f * x;
    int i = *(int*)&xhalf;
    i = 0x5f375a86 - (i >> 1);
    return xhalf * *(float*)&i;
}

bool VUMeter::isClipping(const float* levels, size_t channels, float threshold) const {
    for (size_t i = 0; i < channels; ++i) {
        if (levels[i] >= threshold) {
            return true;
        }
    }
    return false;
}

} // namespace vortex::core::dsp