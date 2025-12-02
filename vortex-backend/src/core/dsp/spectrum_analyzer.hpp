#pragma once

#include "audio_types.hpp"
#include "network_types.hpp"

#include <memory>
#include <vector>
#include <complex>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>

namespace vortex {

/**
 * @brief High-performance real-time spectrum analyzer with GPU acceleration
 *
 * Provides professional-grade spectrum analysis optimized for real-time audio visualization
 * with 2048-point FFT analysis at 60fps, GPU acceleration, and adaptive quality scaling.
 *
 * Features:
 * - 2048-point FFT with configurable windowing
 * - Multiple window functions (Hann, Hamming, Blackman, Kaiser)
 * - Logarithmic and linear frequency scales
 * - GPU-accelerated processing (CUDA/OpenCL/Vulkan)
 * - Real-time peak hold and averaging
 * - 1/3 octave band analysis
 * - Spectral centroid and bandwidth calculation
 * - Harmonic analysis and peak detection
 * - Adaptive quality for different update rates
 */
class SpectrumAnalyzer {
public:
    enum class WindowFunction {
        RECTANGULAR,
        HANN,
        HAMMING,
        BLACKMAN,
        BLACKMAN_HARRIS,
        KAISER,
        NUTTALL
    };

    enum class FrequencyScale {
        LINEAR,
        LOGARITHMIC,
        MEL,
        BARK,
        ERB
    };

    enum class QualityMode {
        FAST,           // Single-precision, basic filtering
        BALANCED,       // Double-precision, good filtering
        HIGH,           // Double-precision, advanced filtering
        REFERENCE      // Double-precision, minimal leakage, oversampling
    };

    enum class ProcessingMode {
        REAL_TIME,     // Optimized for <16ms latency
        HIGH_QUALITY,  // Optimized for accuracy
        BATCH          // Optimized for offline processing
        ADAPTIVE       // Automatically adjusts based on load
    };

    struct SpectrumConfig {
        uint32_t fftSize = 2048;              // FFT size (power of 2)
        uint32_t outputSize = 1025;           // Number of frequency bins (fftSize/2 + 1)
        WindowFunction windowFunction = WindowFunction::HANN;
        FrequencyScale frequencyScale = FrequencyScale::LOGARITHMIC;
        float minFrequency = 20.0f;           // Hz
        float maxFrequency = 20000.0f;        // Hz
        QualityMode quality = QualityMode::BALANCED;
        ProcessingMode mode = ProcessingMode::REAL_TIME;
        uint32_t updateRate = 60;              // Updates per second
        bool enablePeakHold = true;
        float peakHoldTime = 2.0f;             // seconds
        bool enableAveraging = true;
        float averagingTime = 0.5f;           // seconds
        bool enableSmoothing = true;
        float smoothingFactor = 0.8f;          // Exponential smoothing
        bool enableNormalization = true;
        bool enableGPUAcceleration = true;
        uint32_t numChannels = 2;
    };

    SpectrumAnalyzer();
    ~SpectrumAnalyzer();

    // Initialization
    bool initialize(const SpectrumConfig& config);
    void shutdown();
    bool isInitialized() const;

    // Configuration
    void setConfiguration(const SpectrumConfig& config);
    SpectrumConfig getConfiguration() const;

    // Processing
    bool processAudio(const float* inputBuffer, size_t numSamples, uint32_t sampleRate,
                      std::vector<float>& outputSpectrum);
    bool processAudioStereo(const float* leftChannel, const float* rightChannel, size_t numSamples,
                           uint32_t sampleRate, std::vector<float>& outputSpectrum);

    // Real-time processing with callback
    using SpectrumCallback = std::function<void(const std::vector<float>& spectrum,
                                                const SpectrumData& metadata)>;
    void setSpectrumCallback(SpectrumCallback callback);
    void startRealTimeProcessing(uint32_t sampleRate);
    void stopRealTimeProcessing();
    bool isRealTimeProcessingActive() const;

    // Analysis results
    struct SpectrumMetrics {
        std::vector<float> spectrum;              // Frequency bins
        float spectralCentroid = 0.0f;            // Hz
        float spectralBandwidth = 0.0f;           // Hz
        float spectralFlux = 0.0f;                // Rate of spectral change
        float spectralRolloff = 0.0f;             // Frequency below which 85% of energy exists
        float zeroCrossingRate = 0.0f;          // Rate of sign changes
        float harmonicPeakDetection = 0.0f;       // Harmonic confidence
        float noiseLevel = 0.0f;                // Estimated noise floor
        float dynamicRange = 0.0f;              // Peak to noise ratio
        std::vector<float> octaveBands;          // 1/3 octave band energies
        std::vector<float> melBands;             // Mel frequency bands
        uint64_t processingTimeUs = 0;           // Processing time in microseconds
        float gpuUtilization = 0.0f;            // GPU utilization percentage
        std::chrono::steady_clock::time_point timestamp;
    };

    SpectrumMetrics getMetrics() const;
    SpectrumData getSpectrumData() const;

    // Advanced features
    void setPeakHoldTime(float seconds);
    void setAveragingTime(float seconds);
    void setSmoothingFactor(float factor);
    void setFrequencyRange(float minFreq, float maxFreq);

    // Harmonic analysis
    struct HarmonicInfo {
        std::vector<float> harmonicFrequencies;    // Fundamental and harmonics
        std::vector<float> harmonicAmplitudes;     // Relative amplitudes
        float fundamentalFrequency = 0.0f;          // Hz
        float harmonicRatio = 0.0f;                // Average harmonic ratio
        std::vector<float> inharmonicities;        // Deviation from harmonic series
        bool hasStrongFundamental = false;
        float pitchConfidence = 0.0f;               // 0-1
    };

    HarmonicInfo analyzeHarmonics(const std::vector<float>& spectrum, uint32_t sampleRate);
    std::vector<float> detectPeaks(const std::vector<float>& spectrum, float threshold = 0.1f);
    float estimatePitch(const std::vector<float>& spectrum, uint32_t sampleRate);

    // GPU acceleration
    bool enableGPUAcceleration(bool enable);
    bool isGPUAccelerationEnabled() const;
    bool switchGPUBackend(GPUBackend backend);
    std::vector<GPUBackend> getAvailableGPUBackends() const;

    // Quality and performance
    void setQualityMode(QualityMode mode);
    void setProcessingMode(ProcessingMode mode);
    void setUpdateRate(uint32_t updatesPerSecond);

    // Performance monitoring
    struct PerformanceMetrics {
        float averageLatencyMs = 0.0f;
        float maxLatencyMs = 0.0f;
        float minLatencyMs = 0.0f;
        float cpuUsage = 0.0f;
        float gpuUsage = 0.0f;
        uint32_t droppedFrames = 0;
        uint32_t processedFrames = 0;
        float processingThroughput = 0.0f;     // spectra per second
        float qualityScore = 0.0f;              // 0-100
    };

    PerformanceMetrics getPerformanceMetrics() const;
    void resetPerformanceMetrics();

protected:
    // Core FFT processing
    bool initializeFFT();
    void shutdownFFT();

    // GPU processing pipeline
    struct GPUProcessingContext {
        void* deviceContext = nullptr;
        void* computeShader = nullptr;
        void* memoryManager = nullptr;
        std::vector<float*> deviceBuffers;
        void* windowFunctionBuffer = nullptr;
        bool isInitialized = false;
    };

    bool initializeGPUProcessing();
    void shutdownGPUProcessing();
    bool processWithGPU(const float* input, size_t numSamples, std::vector<float>& output);
    bool processWithCPU(const float* input, size_t numSamples, std::vector<float>& output);

    // Window functions
    void generateWindowFunction(std::vector<float>& window);
    void applyWindowFunction(const std::vector<float>& input, const std::vector<float>& window,
                             std::vector<float>& output);

    // FFT implementation
    void computeFFT(const std::vector<std::complex<float>>& input, std::vector<std::complex<float>>& output);
    void computeIFFT(const std::vector<std::complex<float>>& input, std::vector<std::complex<float>>& output);
    void computePowerSpectrum(const std::vector<std::complex<float>>& fftData, std::vector<float>& magnitude);

    // Frequency scaling
    void convertToLogScale(const std::vector<float>& linearSpectrum, std::vector<float>& logSpectrum);
    void convertToMelScale(const std::vector<float>& linearSpectrum, uint32_t sampleRate,
                           std::vector<float>& melSpectrum);
    void convertToBarkScale(const std::vector<float>& linearSpectrum, uint32_t sampleRate,
                           std::vector<float>& barkSpectrum);
    void convertToERBScale(const std::vector<float>& linearSpectrum, uint32_t sampleRate,
                          std::vector<float>& erbSpectrum);

    // Peak hold and averaging
    void updatePeakHold(const std::vector<float>& currentSpectrum);
    void updateAveraging(const std::vector<float>& currentSpectrum);
    void updateSmoothing(const std::vector<float>& currentSpectrum);

    // Real-time processing thread
    void realTimeProcessingThread();
    void submitAudioFrame(const std::vector<float>& audioFrame, uint32_t sampleRate);

    // Analysis functions
    float calculateSpectralCentroid(const std::vector<float>& spectrum);
    float calculateSpectralBandwidth(const std::vector<float>& spectrum);
    float calculateSpectralRolloff(const std::vector<float>& spectrum);
    float calculateZeroCrossingRate(const std::vector<float>& data);
    std::vector<float> calculateOctaveBands(const std::vector<float>& spectrum, uint32_t sampleRate);
    std::vector<float> calculateMelBands(const std::vector<float>& spectrum, uint32_t sampleRate);

    // Error handling
    void setError(const std::string& error);
    std::string getLastError() const;

private:
    // Configuration
    SpectrumConfig config_;

    // FFT data
    std::vector<std::complex<float>> fftBuffer_;
    std::vector<float> windowBuffer_;
    std::vector<float> magnitudeBuffer_;
    std::vector<float> phaseBuffer_;
    std::vector<float> outputSpectrum_;

    // Peak hold and averaging
    std::vector<float> peakHoldSpectrum_;
    std::vector<std::chrono::steady_clock::time_point> peakHoldTime_;
    std::vector<float> averagedSpectrum_;
    std::vector<float> smoothedSpectrum_;
    std::vector<float> smoothedHistory_;

    // Frequency mapping
    std::vector<float> frequencyMap_;
    std::vector<float> octaveBandFreqs_;
    std::vector<std::pair<uint32_t, uint32_t>> octaveBandRanges_;
    std::vector<float> melFilterBank_;
    std::vector<float> barkFilterBank_;

    // Real-time processing
    std::unique_ptr<std::thread> realTimeThread_;
    std::atomic<bool> realTimeProcessingActive_{false};
    std::queue<std::vector<float>> audioFrameQueue_;
    std::mutex frameQueueMutex_;
    std::condition_variable frameQueueCondition_;
    SpectrumCallback spectrumCallback_;
    uint32_t currentSampleRate_ = 0;

    // GPU processing
    GPUProcessingContext gpuContext_;
    bool gpuEnabled_ = false;
    GPUBackend currentGPUBackend_ = GPUBackend::NONE;

    // Performance tracking
    mutable PerformanceMetrics performanceMetrics_;
    std::chrono::steady_clock::time_point lastProcessingTime_;

    // Thread safety
    mutable std::mutex spectrumMutex_;
    mutable std::mutex metricsMutex_;

    // FFT implementation (FFTW or custom)
    void* fftPlan_ = nullptr;
    void* ifftPlan_ = nullptr;
    bool fftwInitialized_ = false;

    // Constants
    static constexpr uint32_t MAX_FFT_SIZE = 65536;    // 64K points
    static constexpr uint32_t MIN_FFT_SIZE = 64;       // 64 points
    static constexpr float LOG_FREQUENCY_MIN = 20.0f;
    static constexpr float LOG_FREQUENCY_MAX = 20000.0f;
    static constexpr uint32_t MEL_BANDS = 40;
    static constexpr float MEL_MIN_FREQ = 133.3334f;
    static constexpr float MEL_MAX_FREQ = 6855.4976f;
    static constexpr uint32_t BARK_BANDS = 24;
    static constexpr float OCTAVE_BANDS = 24;
};

/**
 * @brief GPU-accelerated spectrum analyzer implementation
 */
class GPUSpectrumAnalyzer {
public:
    struct GPUKernelConfig {
        uint32_t blockSize = 1024;            // Number of samples per block
        uint32_t fftSize = 2048;              // FFT size
        uint32_t numThreads = 128;            // Number of GPU threads
        bool useSharedMemory = true;
        bool enableAsyncCopy = true;
        float threadBlockSize = 256.0f;       // Samples per thread
    };

    GPUSpectrumAnalyzer();
    ~GPUSpectrumAnalyzer();

    bool initialize(const GPUKernelConfig& config, GPUBackend backend);
    void shutdown();
    bool isInitialized() const;

    bool processBatch(const std::vector<float*>& inputBuffers, const std::vector<size_t>& bufferSizes,
                      std::vector<std::vector<float>>& outputSpectra);
    bool processSingle(const float* input, size_t numSamples, std::vector<float>& output);

private:
    struct GPUKernel {
        void* kernel = nullptr;
        void* context = nullptr;
        void* stream = nullptr;
        size_t kernelSize = 0;
        bool isCompiled = false;
    };

    GPUKernel fftKernel_;
    GPUKernel windowKernel_;
    GPUKernel magnitudeKernel_;
    GPUKernel scalingKernel_;

    GPUKernelConfig config_;
    bool initialized_ = false;
    GPUBackend backend_;
};

} // namespace vortex