#pragma once

#include <vector>
#include <memory>
#include <atomic>
#include <chrono>
#include "system/logger.hpp"

namespace vortex::core::dsp {

/**
 * High-performance waveform processor for real-time audio visualization
 *
 * Features:
 * - Real-time waveform analysis at 60+ FPS for visualization
 * - GPU acceleration for high-sample-rate audio
 * - Configurable buffer sizes and downsampling
 * - Peak detection and RMS calculation
 * - Zero-crossing detection
 * - Multi-channel support
 * - Efficient ring buffer management
 * - Configurable resolution and scaling
 */
class WaveformProcessor {
public:
    /**
     * Waveform display mode
     */
    enum class DisplayMode {
        Peaks,          // Peak values over time window
        RMS,            // RMS values over time window
        Average,        // Average values
        Instantaneous   // Instantaneous samples
        Envelope        // Envelope follower output
    };

    /**
     * Time scale mode
     */
    enum class TimeScale {
        Linear,         // Linear time axis
        Logarithmic     // Logarithmic time compression for older samples
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
        size_t bufferSize = 1024;            // Input buffer size
        size_t waveformLength = 512;        // Output waveform length (pixels)
        float sampleRate = 44100.0f;        // Sample rate in Hz
        int channels = 2;                   // Number of channels
        DisplayMode displayMode = DisplayMode::Peaks;
        TimeScale timeScale = TimeScale::Linear;
        ProcessingMode processingMode = ProcessingMode::Auto;
        float windowDuration = 0.1f;        // Duration of waveform window in seconds
        float decayRate = 0.95f;            // Peak decay rate (0.0 to 1.0)
        bool enablePeakDetection = true;    // Enable peak detection
        bool enableRMS = true;              // Enable RMS calculation
        bool enableZeroCrossing = false;    // Enable zero-crossing detection
        bool normalizeOutput = true;        // Normalize to 0.0-1.0 range
        bool enableSmoothing = true;        // Enable smoothing filter
        float smoothingFactor = 0.1f;       // Smoothing filter coefficient
        float peakHoldTime = 0.05f;         // Peak hold time in seconds
        bool enableCompression = false;     // Enable logarithmic compression
        float compressionRatio = 20.0f;     // Compression ratio
    };

    /**
     * Waveform data structure
     */
    struct WaveformData {
        std::vector<float> samples;        // Waveform samples
        std::vector<float> peaks;          // Peak values (optional)
        std::vector<float> rms;            // RMS values (optional)
        std::vector<bool> zeroCrossings;   // Zero-crossing markers (optional)
        float maxAmplitude;                // Maximum amplitude in window
        float minAmplitude;                // Minimum amplitude in window
        uint64_t timestamp;                // Timestamp of data
        bool isValid;                       // Data validity flag
    };

    /**
     * Constructor
     */
    WaveformProcessor();

    /**
     * Destructor
     */
    ~WaveformProcessor();

    /**
     * Initialize the waveform processor
     * @param config Configuration parameters
     * @return true if initialization successful
     */
    bool initialize(const Config& config);

    /**
     * Initialize with legacy parameters
     * @param sampleRate Sample rate in Hz
     * @param bufferSize Input buffer size
     * @param channels Number of channels
     * @return true if initialization successful
     */
    bool initialize(float sampleRate, size_t bufferSize, int channels);

    /**
     * Shutdown and cleanup resources
     */
    void shutdown();

    /**
     * Process audio data and return waveform data
     * @param audioData Input audio data (interleaved)
     * @param numSamples Number of samples per channel
     * @return Vector of waveform data (one per channel)
     */
    std::vector<WaveformData> processAudio(const float* audioData, size_t numSamples);

    /**
     * Process audio data with existing ring buffer
     * @param audioData Input audio data
     * @param numSamples Number of samples
     * @param outputWaveform Output waveform data
     * @return true if processing successful
     */
    bool processAudio(const float* audioData, size_t numSamples,
                     std::vector<WaveformData>& outputWaveform);

    /**
     * Get current waveform data without processing new audio
     * @return Vector of current waveform data
     */
    std::vector<WaveformData> getCurrentWaveform();

    /**
     * Set display mode
     * @param mode Waveform display mode
     */
    void setDisplayMode(DisplayMode mode);

    /**
     * Get current display mode
     */
    DisplayMode getDisplayMode() const;

    /**
     * Set time scale mode
     * @param scale Time scale mode
     */
    void setTimeScale(TimeScale scale);

    /**
     * Get current time scale mode
     */
    TimeScale getTimeScale() const;

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
     * Reset internal state and buffers
     */
    void reset();

    /**
     * Check if processor is initialized
     */
    bool isInitialized() const;

    /**
     * Set waveform length (number of samples in output)
     * @param length Waveform length in samples
     */
    void setWaveformLength(size_t length);

    /**
     * Get current waveform length
     */
    size_t getWaveformLength() const;

    /**
     * Set peak hold time
     * @param holdTime Peak hold time in seconds
     */
    void setPeakHoldTime(float holdTime);

    /**
     * Get current peak hold time
     */
    float getPeakHoldTime() const;

    /**
     * Enable/disable peak detection
     * @param enabled Whether to enable peak detection
     */
    void setPeakDetectionEnabled(bool enabled);

    /**
     * Check if peak detection is enabled
     */
    bool isPeakDetectionEnabled() const;

private:
    Config config_;
    bool initialized_;

    // Ring buffer for input audio
    std::vector<float> ringBuffer_;
    size_t ringBufferSize_;
    size_t ringBufferWritePos_;
    std::atomic<size_t> ringBufferReadPos_;

    // Waveform output buffers
    std::vector<float> waveformBuffer_;
    std::vector<float> peakBuffer_;
    std::vector<float> rmsBuffer_;
    std::vector<float> smoothedBuffer_;

    // Peak tracking
    std::vector<float> currentPeaks_;
    std::vector<uint32_t> peakHoldCounters_;
    std::vector<float> maxAmplitudes_;

    // Smoothing filter state
    std::vector<float> smoothedValues_;

    // GPU resources
    void* gpuInputBuffer_;
    void* gpuOutputBuffer_;
    bool gpuInitialized_;

    // Performance tracking
    mutable std::atomic<uint64_t> totalFramesProcessed_;
    mutable std::atomic<double> totalProcessingTime_;
    mutable std::atomic<uint64_t> gpuFramesProcessed_;
    mutable std::atomic<uint64_t> cpuFramesProcessed_;

    // Internal methods
    bool initializeRingBuffer();
    bool initializeWaveformBuffers();
    bool initializePeakTracking();
    bool initializeSmoothingFilters();
    bool initializeGPUResources();
    void cleanupGPUResources();

    void addToRingBuffer(const float* audioData, size_t numSamples);
    bool extractFromRingBuffer(std::vector<float>& output, size_t numSamples);

    void processWaveformCPU(const float* audioData, size_t numSamples,
                            std::vector<WaveformData>& outputWaveform);
    bool processWaveformGPU(const float* audioData, size_t numSamples,
                           std::vector<WaveformData>& outputWaveform);

    void calculatePeaks(const float* audioData, size_t numSamples, float* peaks);
    void calculateRMS(const float* audioData, size_t numSamples, float* rms);
    void detectZeroCrossings(const float* audioData, size_t numSamples, std::vector<bool>& crossings);
    void applySmoothing(float* input, float* output, size_t length);

    void downsampleWaveform(const float* input, size_t inputLength,
                           float* output, size_t outputLength);
    void compressWaveform(float* data, size_t length);
    void normalizeWaveform(float* data, size_t length);

    void updatePerformanceStats(double processingTimeMs, bool usedGPU) const;

    // Time scaling functions
    void applyLinearTimeScale(const float* input, size_t inputLength,
                              float* output, size_t outputLength);
    void applyLogarithmicTimeScale(const float* input, size_t inputLength,
                                   float* output, size_t outputLength);
};

} // namespace vortex::core::dsp