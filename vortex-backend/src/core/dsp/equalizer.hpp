#pragma once

#include "../audio_types.hpp"
#include "../network_types.hpp"

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <complex>
#include <thread>
#include <mutex>
#include <atomic>
#include <array>

namespace vortex {

/**
 * @brief Professional 512-band graphic equalizer with GPU acceleration
 *
 * Implements a high-precision 512-band graphic equalizer optimized for professional
 * audio processing and real-time performance. Supports logarithmic and linear
 * frequency distributions with GPU-accelerated processing for <1ms latency.
 *
 * Features:
 * - 512 frequency bands spanning 20Hz - 20kHz
 * - 8-octave bands per octave for precise control
 * - GPU-accelerated FIR filter design and processing
 * - Real-time coefficient updates with smooth transitions
 * - Support for multiple filter types (Bell, Shelf, Pass filters)
 * - Professional presets and custom impulse response loading
 * - Automatic filter interpolation and smoothing
 * - Multi-channel independent processing
 * - Phase-linear and minimum-phase modes
 * - Advanced dithering and noise shaping
 * - Real-time spectrum analysis feedback
 */
class Equalizer {
public:
    enum class FilterType {
        BELL,                   // Parametric bell filter
        LOW_SHELF,             // Low shelf filter
        HIGH_SHELF,            // High shelf filter
        LOW_PASS,              // Low pass filter
        HIGH_PASS,             // High pass filter
        BAND_PASS,             // Band pass filter
        NOTCH,                 // Notch filter
        ALL_PASS               // All pass filter (phase correction)
    };

    enum class FilterSlope {
        SLOPE_6_DB,            // 6 dB/octave (1st order)
        SLOPE_12_DB,           // 12 dB/octave (2nd order)
        SLOPE_18_DB,           // 18 dB/octave (3rd order)
        SLOPE_24_DB,           // 24 dB/octave (4th order)
        SLOPE_30_DB,           // 30 dB/octave (5th order)
        SLOPE_36_DB,           // 36 dB/octave (6th order)
        SLOPE_48_DB,           // 48 dB/octave (8th order)
        SLOPE_72_DB            // 72 dB/octave (12th order)
    };

    enum class FrequencyScale {
        LOGARITHMIC,           // Logarithmic frequency distribution
        LINEAR,                // Linear frequency distribution
        OCTAVE,                // Octave-based distribution
        THIRD_OCTAVE,          // 1/3 octave bands
        SIXTH_OCTAVE,          // 1/6 octave bands
        CUSTOM                 // Custom frequency mapping
    };

    enum class ProcessingMode {
        REAL_TIME,             // Optimized for <1ms latency
        HIGH_QUALITY,          // Optimized for accuracy
        POWERSAVING,           // Optimized for efficiency
        ADAPTIVE               // Automatically adjusts based on load
    };

    enum class PhaseMode {
        PHASE_LINEAR,          // Phase-linear processing
        MINIMUM_PHASE,         // Minimum-phase filters
        LINEAR_PHASE_MINIMUM,  // Hybrid approach
        PHASE_CORRECTED        // Phase correction applied
    };

    struct FilterBand {
        uint32_t bandIndex;                    // Band number (0-511)
        float frequency;                      // Center frequency in Hz
        float gain;                          // Gain in dB (-60 to +20)
        float Q;                             // Quality factor (0.1 to 100)
        FilterType type = FilterType::BELL;   // Filter type
        FilterSlope slope = FilterSlope::SLOPE_24_DB; // Slope
        bool enabled = true;                 // Band enabled state
        bool bypassed = false;               // Bypass state
        float frequencyResponse = 0.0f;      // Current frequency response

        // Filter coefficients (computed)
        std::array<float, 6> bCoefficients;   // Numerator coefficients
        std::array<float, 6> aCoefficients;   // Denominator coefficients

        // Processing state
        std::array<float, 4> delayLine;       // Delay line for biquad processing
        std::array<float, 4> history;         // Filter history

        // GPU data
        void* gpuCoefficients = nullptr;      // GPU coefficient buffer
        void* gpuState = nullptr;             // GPU state buffer
    };

    struct EqualizerConfig {
        // Basic configuration
        uint32_t numBands = 512;              // Number of frequency bands
        uint32_t sampleRate = 96000;          // Sample rate in Hz
        uint16_t numChannels = 2;             // Number of audio channels
        uint32_t bufferSize = 1024;           // Processing buffer size
        FrequencyScale frequencyScale = FrequencyScale::LOGARITHMIC;

        // Processing options
        ProcessingMode mode = ProcessingMode::REAL_TIME;
        PhaseMode phaseMode = PhaseMode::MINIMUM_PHASE;
        bool enableGPUAcceleration = true;
        bool enableMultiThreading = true;
        uint32_t numThreads = 0;              // 0 = auto-detect

        // Quality settings
        bool enableHighPrecision = false;     // Use double precision
        bool enableDithering = true;
        bool enableNoiseShaping = false;
        uint32_t ditherDepth = 24;            // Bits for dithering

        // Performance settings
        float maxLatencyMs = 1.0f;            // Maximum allowed latency
        bool enableLatencyMonitoring = true;
        bool enablePerformanceCounters = true;

        // Smoothing and interpolation
        bool enableSmoothing = true;
        float smoothingTime = 10.0f;          // Smoothing time in ms
        float interpolationFactor = 0.1f;     // Interpolation factor

        // Preset management
        bool enablePresets = true;
        std::string presetDirectory = "presets/";
        bool autoSavePresets = false;

        // Real-time analysis
        bool enableSpectrumAnalysis = true;
        uint32_t spectrumSize = 65536;        // FFT size for analysis
        bool enablePhaseAnalysis = false;

        // Advanced options
        bool enableLookahead = false;         // Lookahead processing
        uint32_t lookaheadSamples = 0;        // Lookahead buffer size
        bool enableCrossfeed = false;         // Crossfeed processing
        float crossfeedAmount = 0.0f;         // Crossfeed amount (0-1)
    };

    struct Preset {
        std::string name;
        std::string description;
        std::string author;
        std::string version;
        std::vector<float> bandGains;          // 512 band gains in dB
        std::vector<FilterBand> bandSettings;  // Complete band settings
        std::map<std::string, std::string> metadata;
        std::chrono::system_clock::time_point createdTime;
        std::chrono::system_clock::time_point modifiedTime;
    };

    struct EqualizerStatistics {
        // Performance metrics
        float averageLatency = 0.0f;          // Average processing latency
        float maxLatency = 0.0f;              // Maximum latency
        float minLatency = 1000.0f;           // Minimum latency
        uint32_t totalSamplesProcessed = 0;   // Total samples processed
        uint32_t droppedFrames = 0;           // Number of dropped frames

        // CPU/GPU usage
        float cpuUsage = 0.0f;                // CPU usage percentage
        float gpuUsage = 0.0f;                // GPU usage percentage
        float memoryUsage = 0.0f;             // Memory usage in MB
        uint32_t activeBands = 0;             // Number of active bands

        // Processing quality
        float signalToNoiseRatio = 0.0f;      // SNR in dB
        float totalHarmonicDistortion = 0.0f; // THD percentage
        float intermodulationDistortion = 0.0f; // IMD percentage

        // Quality metrics
        float processingThroughput = 0.0f;    // Samples per second
        float qualityScore = 0.0f;            // Overall quality score (0-100)
        uint32_t coefficientUpdates = 0;      // Number of coefficient updates

        // Timing
        std::chrono::steady_clock::time_point startTime;
        std::chrono::steady_clock::time_point lastActivity;
        std::chrono::steady_clock::time_point lastCoefficientUpdate;
    };

    Equalizer();
    ~Equalizer();

    // Initialization
    bool initialize(const EqualizerConfig& config);
    void shutdown();
    bool isInitialized() const;

    // Configuration
    void setConfiguration(const EqualizerConfig& config);
    EqualizerConfig getConfiguration() const;

    // Main processing
    bool processAudio(const float* input, float* output, size_t numSamples);
    bool processAudioInterleaved(const float* input, float* output, size_t numSamples, uint16_t channels);
    bool processAudioMultiChannel(const std::vector<const float*>& inputs, std::vector<float*>& outputs,
                                   size_t numSamples, uint16_t channels);

    // Band control
    bool setBandGain(uint32_t bandIndex, float gain);
    bool setBandParameters(uint32_t bandIndex, float frequency, float gain, float Q,
                           FilterType type, FilterSlope slope);
    bool enableBand(uint32_t bandIndex, bool enabled);
    bool bypassBand(uint32_t bandIndex, bool bypassed);
    FilterBand getBand(uint32_t bandIndex) const;
    std::vector<FilterBand> getAllBands() const;
    bool resetBand(uint32_t bandIndex);
    bool resetAllBands();

    // Frequency response
    std::vector<float> getFrequencyResponse(uint32_t numPoints = 2048) const;
    float getFrequencyResponseAtFrequency(float frequency) const;
    std::vector<std::complex<float>> getComplexFrequencyResponse(uint32_t numPoints = 2048) const;
    std::vector<float> getPhaseResponse(uint32_t numPoints = 2048) const;
    float getPhaseResponseAtFrequency(float frequency) const;

    // Impulse response
    std::vector<float> getImpulseResponse(uint32_t numSamples = 65536) const;
    bool setImpulseResponse(const std::vector<float>& impulseResponse);
    bool loadImpulseResponseFromFile(const std::string& filePath);
    bool saveImpulseResponseToFile(const std::string& filePath) const;

    // Preset management
    bool savePreset(const std::string& name, const std::string& description = "");
    bool loadPreset(const std::string& name);
    bool deletePreset(const std::string& name);
    std::vector<Preset> getAvailablePresets() const;
    bool exportPreset(const std::string& name, const std::string& filePath) const;
    bool importPreset(const std::string& filePath, const std::string& name = "");

    // Built-in presets
    bool loadFlatPreset();
    bool loadVocalPreset();
    bool loadBassBoostPreset();
    bool loadTrebleBoostPreset();
    bool loadLoudnessPreset();
    bool loadAudiophilePreset();
    bool loadStudioPreset();
    bool loadLivePreset();
    bool loadHeadphonePreset();
    bool loadCarPreset();

    // Real-time analysis
    std::vector<float> analyzeSpectrum(const float* audio, size_t numSamples) const;
    std::vector<float> getCurrentSpectrum() const;
    float getCurrentLevel() const;
    float getPeakLevel() const;
    float getAverageLevel() const;

    // Advanced features
    bool enableLookaheadProcessing(bool enable, uint32_t lookaheadSamples = 0);
    bool setCrossfeed(float amount);
    bool setDithering(bool enable, uint32_t depth = 24);
    bool setNoiseShaping(bool enable, uint32_t order = 3);

    // Filter design
    bool designBandFilter(FilterBand& band);
    bool updateFilterCoefficients(FilterBand& band);
    bool smoothParameterChanges(uint32_t bandIndex, float targetGain, float targetQ);

    // GPU acceleration
    bool enableGPUAcceleration(bool enable);
    bool isGPUAccelerationEnabled() const;
    bool updateGPUFilters();
    bool syncGPUState();

    // Multi-threading
    bool setProcessingThreads(uint32_t numThreads);
    uint32_t getProcessingThreads() const;

    // Statistics and monitoring
    EqualizerStatistics getStatistics() const;
    void resetStatistics();
    bool isHealthy() const;

    // Calibration and testing
    bool performCalibration();
    bool generateTestTone(float frequency, float amplitude, float duration);
    bool measureFrequencyResponse(std::vector<float>& frequencies, std::vector<float>& responses);

    // Callbacks
    using PresetChangedCallback = std::function<void(const std::string&, const Preset&)>;
    using BandChangedCallback = std::function<void(uint32_t, const FilterBand&)>;
    using ProcessingCallback = std::function<void(const float*, float*, size_t)>;

    void setPresetChangedCallback(PresetChangedCallback callback);
    void setBandChangedCallback(BandChangedCallback callback);
    void setProcessingCallback(ProcessingCallback callback);

private:
    // Core processing
    void initializeFilterBands();
    void initializeFrequencyScale();
    void computeFrequencyScale();
    bool initializeGPUResources();
    void cleanupGPUResources();

    // Filter design algorithms
    void computeBiquadCoefficients(FilterBand& band);
    void computeFIRCoefficients(FilterBand& band);
    void computeIIRCoefficients(FilterBand& band);
    float computeFrequencyResponse(float frequency, const FilterBand& band) const;

    // Processing functions
    void processBiquadSection(const float* input, float* output, size_t numSamples,
                             FilterBand& band);
    void processFIRSection(const float* input, float* output, size_t numSamples,
                          FilterBand& band);
    void processChannelData(const float* input, float* output, size_t numSamples, uint16_t channel);
    void applyDithering(float* audio, size_t numSamples);
    void applyNoiseShaping(float* audio, size_t numSamples);

    // Smoothing and interpolation
    void smoothParameterChanges();
    void interpolateCoefficients(uint32_t bandIndex, const std::array<float, 6>& targetCoefficients);
    float smoothParameterValue(float current, float target, float factor);

    // GPU processing
    class GPUProcessor;
    std::unique_ptr<GPUProcessor> gpuProcessor_;
    bool updateGPUBand(uint32_t bandIndex);
    bool processWithGPU(const float* input, float* output, size_t numSamples);

    // Multi-threading
    void processThreaded(const float* input, float* output, size_t numSamples);
    void processChannelThread(const float* input, float* output, size_t numSamples,
                             uint16_t startChannel, uint16_t numChannels);
    std::vector<std::thread> processingThreads_;
    std::atomic<bool> multithreadingEnabled_{false};

    // Memory management
    struct ProcessingBuffers {
        std::vector<float> inputBuffer;
        std::vector<float> outputBuffer;
        std::vector<float> intermediateBuffer;
        std::vector<float> lookaheadBuffer;
        std::vector<std::complex<float>> fftBuffer;
        std::vector<float> spectrumBuffer;
    };

    ProcessingBuffers buffers_;
    mutable std::mutex buffersMutex_;

    // State
    EqualizerConfig config_;
    std::atomic<bool> initialized_{false};
    std::vector<FilterBand> bands_;
    std::vector<float> frequencyScale_;
    mutable std::mutex bandsMutex_;

    // Presets
    std::map<std::string, Preset> presets_;
    std::string currentPreset_;
    mutable std::mutex presetsMutex_;

    // Statistics
    mutable std::mutex statsMutex_;
    EqualizerStatistics statistics_;
    std::chrono::steady_clock::time_point lastUpdateTime_;

    // Processing state
    std::atomic<float> currentLevel_{0.0f};
    std::atomic<float> peakLevel_{0.0f};
    std::atomic<float> averageLevel_{0.0f};

    // GPU resources
    void* gpuFilterCoefficients_ = nullptr;
    void* gpuFilterStates_ = nullptr;
    void* gpuProcessingBuffer_ = nullptr;
    size_t gpuMemorySize_ = 0;

    // Callbacks
    PresetChangedCallback presetChangedCallback_;
    BandChangedCallback bandChangedCallback_;
    ProcessingCallback processingCallback_;

    // Utility functions
    float frequencyToIndex(float frequency) const;
    float indexToFrequency(uint32_t index) const;
    bool isValidBandIndex(uint32_t index) const;
    float dbToLinear(float db) const;
    float linearToDb(float linear) const;
    float frequencyToOctave(float frequency) const;
    float octaveToFrequency(float octave) const;

    // Error handling
    void setError(const std::string& error) const;
    std::string getLastError() const;
    mutable std::string lastError_;

    // Constants
    static constexpr uint32_t MAX_BANDS = 512;
    static constexpr float MIN_FREQUENCY = 20.0f;
    static constexpr float MAX_FREQUENCY = 20000.0f;
    static constexpr float MIN_GAIN = -60.0f;
    static constexpr float MAX_GAIN = 20.0f;
    static constexpr float MIN_Q = 0.1f;
    static constexpr float MAX_Q = 100.0f;
};

/**
 * @brief GPU-accelerated equalizer processor
 *
 * Implements the GPU processing pipeline for the 512-band equalizer,
 * utilizing CUDA, OpenCL, or Vulkan for parallel filter processing.
 */
class Equalizer::GPUProcessor {
public:
    GPUProcessor(const EqualizerConfig& config);
    ~GPUProcessor();

    bool initialize();
    void shutdown();
    bool isInitialized() const;

    bool uploadFilterCoefficients(const std::vector<FilterBand>& bands);
    bool uploadFilterStates(const std::vector<FilterBand>& bands);
    bool processAudio(const float* input, float* output, size_t numSamples, uint16_t channels);

    bool downloadFilterStates(std::vector<FilterBand>& bands);
    bool synchronize();
    float getProcessingTime() const;

private:
    struct GPUContext {
        void* device = nullptr;
        void* context = nullptr;
        void* commandQueue = nullptr;
        void* memoryManager = nullptr;
        bool initialized = false;
    };

    bool initializeCUDA();
    bool initializeOpenCL();
    bool initializeVulkan();

    void processWithCUDA(const float* input, float* output, size_t numSamples, uint16_t channels);
    void processWithOpenCL(const float* input, float* output, size_t numSamples, uint16_t channels);
    void processWithVulkan(const float* input, float* output, size_t numSamples, uint16_t channels);

    EqualizerConfig config_;
    GPUContext gpuContext_;
    std::atomic<bool> initialized_{false};

    void* coefficientBuffer_ = nullptr;
    void* stateBuffer_ = nullptr;
    void* inputBuffer_ = nullptr;
    void* outputBuffer_ = nullptr;
    size_t bufferSize_ = 0;

    std::atomic<float> processingTime_{0.0f};
};

} // namespace vortex