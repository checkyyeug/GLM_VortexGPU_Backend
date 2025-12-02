#pragma once

#include <vector>
#include <memory>
#include <atomic>
#include <chrono>
#include "system/logger.hpp"

namespace vortex::core::dsp {

/**
 * High-performance VU meter processor for real-time audio level monitoring
 *
 * Features:
 * - Real-time level monitoring at 1000+ Hz update rate
 * - Multiple VU meter types (Peak, RMS, VU, K-System, LUFS)
 * - Configurable attack and release times
 * - Peak hold functionality
 * - Stereo and multi-channel support
 * - Logarithmic scaling for dBFS
 * - Reference level calibration
 * - GPU acceleration for high sample rates
 * - Ballistics matching professional standards
 */
class VUMeter {
public:
    /**
     * VU meter type
     */
    enum class MeterType {
        Peak,           // Peak level meter (fast attack, slow release)
        RMS,            // RMS level meter (smoothed average)
        VU,             // Traditional VU meter (300ms integration)
        KSystem,        // K-System meter (K-20, K-14, K-12)
        LUFS,           // Loudness units (EBU R128)
        Digital,        // Digital peak meter (instantaneous)
        PPM             // Program Peak Meter (BBC standard)
    };

    /**
     * Reference level standards
     */
    enum class ReferenceLevel {
        dBFS_0,         // 0 dBFS (full scale digital)
        dBFS_18,        // -18 dBFS (professional digital)
        dBFS_20,        // -20 dBFS (common broadcast standard)
        dBFS_24,        // -24 dBFS (EBU R128 standard)
        VU_0,           // 0 VU (+4 dBu)
        K20,            // K-20 system (-20 dBFS = 0 K-unit)
        K14,            // K-14 system (-14 dBFS = 0 K-unit)
        K12             // K-12 system (-12 dBFS = 0 K-unit)
    };

    /**
     * Processing mode
     */
    enum class ProcessingMode {
        CPU,
        GPU,
        Auto
    };

    /**
     * Configuration structure
     */
    struct Config {
        MeterType meterType = MeterType::RMS;
        ReferenceLevel referenceLevel = ReferenceLevel::dBFS_20;
        ProcessingMode processingMode = ProcessingMode::Auto;
        float sampleRate = 44100.0f;
        int channels = 2;
        float attackTime = 0.001f;        // Attack time in seconds
        float releaseTime = 0.1f;         // Release time in seconds
        float holdTime = 0.0f;            // Peak hold time (0 = disabled)
        bool enableStereoLink = true;      // Link stereo channels
        bool enableTruePeak = false;       // Enable true peak detection
        float integrationTime = 0.3f;     // Integration time for VU meters
        float windowSize = 0.0f;           // Window size for LUFS
        bool enableHistory = false;        // Enable level history
        size_t historySize = 1000;         // History buffer size
        float ballisticsIntegration = 0.0f; // Custom integration time
        float minLevel = -60.0f;          // Minimum display level
        float maxLevel = 0.0f;            // Maximum display level
    };

    /**
     * VU meter reading structure
     */
    struct VUReading {
        float leftLevel;                  // Left channel level in dBFS
        float rightLevel;                 // Right channel level in dBFS
        float monoLevel;                  // Mono (L+R)/2 level in dBFS
        float peakLevel;                  // Peak level in dBFS
        float averageLevel;               // Average level in dBFS
        float loudnessLUFS;              // LUFS loudness (if applicable)
        uint64_t timestamp;              // Timestamp of reading
        bool isValid;                     // Reading validity
        bool isClipping;                 // Clipping detection
        float stereoBalance;             // Stereo balance (-1 to +1)
        float dynamicRange;              // Dynamic range in dB
    };

    /**
     * Level history structure
     */
    struct LevelHistory {
        std::vector<float> leftHistory;
        std::vector<float> rightHistory;
        std::vector<float> monoHistory;
        std::vector<float> peakHistory;
        std::vector<uint64_t> timestamps;
        size_t currentIndex;
        bool isFull;
    };

    /**
     * Constructor
     */
    VUMeter();

    /**
     * Destructor
     */
    ~VUMeter();

    /**
     * Initialize the VU meter
     * @param config Configuration parameters
     * @return true if initialization successful
     */
    bool initialize(const Config& config);

    /**
     * Initialize with legacy parameters
     * @param sampleRate Sample rate in Hz
     * @param bufferSize Audio buffer size
     * @param channels Number of channels
     * @return true if initialization successful
     */
    bool initialize(float sampleRate, size_t bufferSize, int channels);

    /**
     * Shutdown and cleanup resources
     */
    void shutdown();

    /**
     * Process audio data and return current VU meter readings
     * @param audioData Input audio data (interleaved)
     * @param numSamples Number of samples per channel
     * @return VU meter reading
     */
    VUReading processAudio(const float* audioData, size_t numSamples);

    /**
     * Process audio data with existing state
     * @param audioData Input audio data
     * @param numSamples Number of samples
     * @param reading Output VU meter reading
     * @return true if processing successful
     */
    bool processAudio(const float* audioData, size_t numSamples, VUReading& reading);

    /**
     * Get current VU meter reading without processing new audio
     * @return Current VU meter reading
     */
    VUReading getCurrentReading() const;

    /**
     * Get current levels in different units
     * @return Vector of current levels [left, right, mono, peak] in dBFS
     */
    std::vector<float> getCurrentLevels() const;

    /**
     * Set meter type
     * @param type VU meter type
     */
    void setMeterType(MeterType type);

    /**
     * Get current meter type
     */
    MeterType getMeterType() const;

    /**
     * Set reference level
     * @param level Reference level standard
     */
    void setReferenceLevel(ReferenceLevel level);

    /**
     * Get current reference level
     */
    ReferenceLevel getReferenceLevel() const;

    /**
     * Set processing mode
     * @param mode Processing mode
     */
    void setProcessingMode(ProcessingMode mode);

    /**
     * Get current processing mode
     */
    ProcessingMode getProcessingMode() const;

    /**
     * Check if GPU acceleration is available
     */
    bool isGPUAvailable() const;

    /**
     * Get current configuration
     */
    const Config& getConfig() const;

    /**
     * Get performance statistics
     */
    std::string getPerformanceStats() const;

    /**
     * Reset level history and internal state
     */
    void reset();

    /**
     * Reset peaks only
     */
    void resetPeaks();

    /**
     * Check if processor is initialized
     */
    bool isInitialized() const;

    /**
     * Set attack time
     * @param time Attack time in seconds
     */
    void setAttackTime(float time);

    /**
     * Get current attack time
     */
    float getAttackTime() const;

    /**
     * Set release time
     * @param time Release time in seconds
     */
    void setReleaseTime(float time);

    /**
     * Get current release time
     */
    float getReleaseTime() const;

    /**
     * Set peak hold time
     * @param time Peak hold time in seconds
     */
    void setPeakHoldTime(float time);

    /**
     * Get current peak hold time
     */
    float getPeakHoldTime() const;

    /**
     * Enable/disable stereo linking
     * @param enabled Whether to link stereo channels
     */
    void setStereoLinkEnabled(bool enabled);

    /**
     * Check if stereo linking is enabled
     */
    bool isStereoLinkEnabled() const;

    /**
     * Enable/disable true peak detection
     * @param enabled Whether to enable true peak detection
     */
    void setTruePeakEnabled(bool enabled);

    /**
     * Check if true peak detection is enabled
     */
    bool isTruePeakEnabled() const;

    /**
     * Get level history
     * @return Level history structure
     */
    const LevelHistory& getLevelHistory() const;

    /**
     * Enable/disable level history
     * @param enabled Whether to record history
     */
    void setHistoryEnabled(bool enabled);

    /**
     * Check if level history is enabled
     */
    bool isHistoryEnabled() const;

private:
    Config config_;
    bool initialized_;

    // Level state
    std::vector<float> currentLevels_;
    std::vector<float> peakLevels_;
    std::vector<float> smoothedLevels_;
    std::vector<float> attackCoefficients_;
    std::vector<float> releaseCoefficients_;

    // Peak hold state
    std::vector<uint32_t> peakHoldCounters_;
    std::vector<float> peakHoldValues_;

    // History tracking
    LevelHistory levelHistory_;
    bool historyEnabled_;

    // GPU resources
    void* gpuAudioBuffer_;
    void* gpuLevelsBuffer_;
    bool gpuInitialized_;

    // Performance tracking
    mutable std::atomic<uint64_t> totalFramesProcessed_;
    mutable std::atomic<double> totalProcessingTime_;
    mutable std::atomic<uint64_t> gpuFramesProcessed_;
    mutable std::atomic<uint64_t> cpuFramesProcessed_;

    // Internal methods
    bool initializeLevelState();
    bool initializeBallistics();
    bool initializeHistory();
    bool initializeGPUResources();
    void cleanupGPUResources();

    void processAudioCPU(const float* audioData, size_t numSamples, VUReading& reading);
    bool processAudioGPU(const float* audioData, size_t numSamples, VUReading& reading);

    void calculatePeakLevels(const float* audioData, size_t numSamples, float* levels);
    void calculateRMSLevels(const float* audioData, size_t numSamples, float* levels);
    void calculateVULevels(const float* audioData, size_t numSamples, float* levels);
    void calculateKSystemLevels(const float* audioData, size_t numSamples, float* levels);
    void calculateLUFSLevels(const float* audioData, size_t numSamples, float* levels);
    void calculateDigitalLevels(const *audioData, size_t numSamples, float* levels);
    void calculatePPMLevels(const float* audioData, size_t numSamples, float* levels);

    void applyBallistics(float* currentLevels, const float* inputLevels, size_t channels);
    void updatePeakHold(float* peakLevels, const float* inputLevels, size_t channels);
    void calculateStereoMetrics(const float* levels, float& monoLevel, float& balance);

    float getReferenceOffset() const;
    float levelToDisplay(float level) const;
    float displayToLevel(float display) const;

    void addToHistory(const VUReading& reading);
    void calculateDynamicRange(const VUReading& reading, float& dynamicRange);

    void updatePerformanceStats(double processingTimeMs, bool usedGPU) const;

    // Helper functions
    float fastExp(float x) const;
    float fastLog(float x) const;
    float fastSqrt(float x) const;
    bool isClipping(const float* levels, size_t channels, float threshold) const;
};

} // namespace vortex::core::dsp