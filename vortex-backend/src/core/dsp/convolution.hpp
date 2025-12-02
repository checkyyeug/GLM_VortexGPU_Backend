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
#include <condition_variable>
#include <array>

namespace vortex {

/**
 * @brief Massive 16 million-point convolution engine for professional audio processing
 *
 * Implements an ultra-high-performance convolution processor capable of handling
 * impulse responses up to 16 million points (over 5 minutes at 48kHz), designed for
 * professional audio production, acoustic simulation, and advanced reverb processing.
 *
 * Features:
 * - 16 million-point convolution capacity (16777216 points)
 * - Multi-block FFT processing with overlap-add/save methods
 * - GPU-accelerated convolution using CUDA, OpenCL, and Vulkan
 * - Real-time processing with configurable latency modes
 * - Multi-channel independent processing (up to 32 channels)
 * - Dynamic impulse response loading and hot-swapping
 * - Automatic partitioning optimization for different IR lengths
 * - Advanced dithering and noise shaping for high-quality output
 * - Zero-latency monitoring mode for professional applications
 * - Impulse response analysis and visualization tools
 * - Automatic gain control and normalization
 */
class ConvolutionEngine {
public:
    enum class ProcessingMode {
        REAL_TIME,             // Optimized for minimum latency (<10ms)
        HIGH_QUALITY,          // Optimized for maximum quality
        POWERSAVING,           // Optimized for efficiency
        ADAPTIVE,              // Automatically adjusts based on load
        OFFLINE               // Optimized for batch processing
    };

    enum class LatencyMode {
        ZERO_LATENCY,         // Minimum possible latency
        LOW_LATENCY,          // Low latency (10-50ms)
        BALANCED,             // Balanced latency/quality (50-100ms)
        HIGH_LATENCY,         // High latency for maximum quality (100ms+)
        ULTRA_LOW_LATENCY     // Ultra-low latency (<5ms)
    };

    enum class ConvolutionMethod {
        OVERLAP_ADD,          // Overlap-add method
        OVERLAP_SAVE,         // Overlap-save method
        UNIFORM_PARTITIONED,   // Uniform partitioned convolution
        NON_UNIFORM_PARTITIONED, // Non-uniform partitioned convolution
        HYBRID,               // Hybrid method
        FREQUENCY_DOMAIN      // Pure frequency domain processing
    };

    enum class ImpulseResponseFormat {
        TIME_DOMAIN,           // Time domain impulse response
        FREQUENCY_DOMAIN,      // Frequency domain response
        MINIMUM_PHASE,         // Minimum-phase IR
        LINEAR_PHASE,          // Linear-phase IR
        MULTI_BAND,            // Multi-band IR
        AMBISONICS             // Ambisonics IR
    };

    struct ConvolutionConfig {
        // Basic configuration
        uint32_t maxIRLength = 16777216;      // Maximum IR length (16M points)
        uint32_t sampleRate = 48000;           // Sample rate in Hz
        uint16_t numChannels = 2;              // Number of audio channels
        uint32_t blockSize = 1024;             // Processing block size
        ProcessingMode mode = ProcessingMode::REAL_TIME;
        LatencyMode latencyMode = LatencyMode::BALANCED;
        ConvolutionMethod method = ConvolutionMethod::UNIFORM_PARTITIONED;

        // FFT configuration
        uint32_t fftSize = 65536;              // FFT size (will be auto-optimized)
        uint32_t numPartitions = 0;            // 0 = auto-calculate
        bool enableZeroPadding = true;
        bool enableWindowing = true;
        std::string windowFunction = "hann";

        // GPU configuration
        bool enableGPUAcceleration = true;
        bool enableMultiGPU = false;
        std::vector<std::string> preferredGPUs;
        bool enableUnifiedMemory = false;
        bool enableGPUDirect = false;

        // Performance settings
        float maxLatencyMs = 100.0f;           // Maximum allowed latency
        float targetLatencyMs = 50.0f;         // Target latency
        bool enableAdaptiveLatency = true;
        uint32_t cpuAffinityMask = 0;          // 0 = auto-detect
        uint32_t numThreads = 0;              // 0 = auto-detect

        // Quality settings
        bool enableHighPrecision = false;     // Use double precision
        bool enableDithering = true;
        bool enableNoiseShaping = false;
        uint32_t ditherDepth = 24;            // Bits for dithering
        bool enableNormalization = true;
        bool enableAutoGainControl = false;

        // Memory management
        uint32_t maxMemoryUsage = 2048;       // Maximum memory usage in MB
        bool enableMemoryPooling = true;
        bool enableStreamingMode = false;     // Stream IRs from disk
        uint32_t streamChunkSize = 1024;      // Chunk size for streaming

        // Advanced options
        bool enableWetDryMix = true;
        float defaultWetLevel = 1.0f;         // Wet signal level
        float defaultDryLevel = 0.0f;         // Dry signal level
        bool enablePredelay = false;
        float defaultPredelay = 0.0f;         // Predelay in ms

        // Monitoring and analysis
        bool enableIRAnalysis = true;
        bool enableRealTimeAnalysis = false;
        bool enableSpectrumDisplay = false;
        uint32_t analysisBlockSize = 8192;
    };

    struct ImpulseResponse {
        std::string name;
        std::string description;
        std::string filePath;
        uint32_t length = 0;                 // IR length in samples
        uint16_t channels = 1;              // Number of channels
        uint32_t sampleRate = 48000;        // Sample rate
        ImpulseResponseFormat format = ImpulseResponseFormat::TIME_DOMAIN;

        // IR data
        std::vector<std::vector<float>> timeDomainData;     // Time domain samples per channel
        std::vector<std::vector<std::complex<float>>> frequencyDomainData; // Frequency domain data

        // Analysis data
        float peakLevel = 0.0f;             // Peak level in dB
        float averageLevel = 0.0f;           // Average level in dB
        float dynamicRange = 0.0f;           // Dynamic range in dB
        float rt60 = 0.0f;                  // RT60 reverberation time
        std::vector<float> frequencyResponse; // Frequency response
        std::vector<float> phaseResponse;     // Phase response

        // Metadata
        std::map<std::string, std::string> metadata;
        std::chrono::system_clock::time_point createdTime;
        std::chrono::system_clock::time_point modifiedTime;

        // Processing state
        bool loaded = false;
        bool processed = false;
        bool active = false;
    };

    struct ConvolutionStatistics {
        // Performance metrics
        float averageLatency = 0.0f;          // Average processing latency
        float maxLatency = 0.0f;              // Maximum latency
        float minLatency = 1000.0f;           // Minimum latency
        uint64_t totalSamplesProcessed = 0;   // Total samples processed
        uint64_t droppedFrames = 0;           // Number of dropped frames

        // Processing statistics
        float cpuUsage = 0.0f;                // CPU usage percentage
        float gpuUsage = 0.0f;                // GPU usage percentage
        float memoryUsage = 0.0f;             // Memory usage in MB
        uint32_t activeIRs = 0;               // Number of active IRs

        // Quality metrics
        float signalToNoiseRatio = 0.0f;      // SNR in dB
        float totalHarmonicDistortion = 0.0f; // THD percentage
        float processingThroughput = 0.0f;    // Samples per second
        float qualityScore = 0.0f;            // Overall quality score (0-100)

        // Convolution-specific metrics
        uint32_t fftCalls = 0;                // Number of FFT calls
        uint32_t fftSize = 0;                 // Current FFT size
        uint32_t numPartitions = 0;           // Number of partitions
        float partitionEfficiency = 0.0f;     // Partition efficiency
        float convolutionEfficiency = 0.0f;   // Convolution efficiency

        // Timing
        std::chrono::steady_clock::time_point startTime;
        std::chrono::steady_clock::time_point lastActivity;
    };

    ConvolutionEngine();
    ~ConvolutionEngine();

    // Initialization
    bool initialize(const ConvolutionConfig& config);
    void shutdown();
    bool isInitialized() const;

    // Configuration
    void setConfiguration(const ConvolutionConfig& config);
    ConvolutionConfig getConfiguration() const;

    // Main processing
    bool processAudio(const float* input, float* output, size_t numSamples);
    bool processAudioMultiChannel(const std::vector<const float*>& inputs,
                                   std::vector<float*>& outputs,
                                   size_t numSamples, uint16_t channels);
    bool processAudioWithDryWet(const float* input, float* output, size_t numSamples,
                                float wetLevel = 1.0f, float dryLevel = 0.0f);

    // Impulse response management
    bool loadImpulseResponse(const std::string& filePath, const std::string& name = "");
    bool loadImpulseResponseFromData(const std::vector<float>& data, uint32_t sampleRate,
                                     const std::string& name = "");
    bool unloadImpulseResponse(const std::string& name);
    bool setActiveImpulseResponse(const std::string& name);
    std::string getActiveImpulseResponse() const;
    std::vector<std::string> getAvailableImpulseResponses() const;
    ImpulseResponse getImpulseResponse(const std::string& name) const;

    // Impulse response creation and analysis
    bool createSyntheticIR(const std::string& name, float length, float rt60 = 1.0f,
                          float mixLevel = 0.5f, float decay = 0.7f);
    bool createReverseIR(const std::string& name, const std::string& sourceIR);
    bool createMultiBandIR(const std::string& name, const std::vector<std::string>& bandIRs,
                           const std::vector<uint32_t>& crossoverFreqs);
    ImpulseResponse analyzeImpulseResponse(const std::vector<float>& data, uint32_t sampleRate);

    // Real-time control
    bool setWetDryMix(float wetLevel, float dryLevel);
    bool setPredelay(float delayMs);
    bool setGain(float gain);
    bool setNormalization(bool enabled, float targetLevel = 0.0f);

    // Processing control
    bool setProcessingMode(ProcessingMode mode);
    bool setLatencyMode(LatencyMode mode);
    bool setConvolutionMethod(ConvolutionMethod method);
    bool setBlockSize(uint32_t blockSize);

    // Analysis and monitoring
    std::vector<float> analyzeCurrentIR() const;
    std::vector<float> analyzeFrequencyResponse() const;
    std::vector<float> analyzePhaseResponse() const;
    float getEstimatedRT60() const;
    float getEstimatedDynamicRange() const;
    float getCurrentSNR() const;

    // GPU management
    bool enableGPUAcceleration(bool enable);
    bool isGPUAccelerationEnabled() const;
    std::vector<std::string> getAvailableGPUs() const;
    bool setPreferredGPU(const std::string& gpuName);
    float getGPUUtilization() const;

    // Memory management
    bool setMemoryLimit(uint32_t maxMemoryMB);
    uint32_t getMemoryUsage() const;
    uint32_t getMaxMemoryUsage() const;
    bool enableMemoryPooling(bool enable);

    // Advanced features
    bool enableZeroLatencyMonitoring(bool enable);
    bool enableStreamingMode(bool enable, uint32_t chunkSize = 1024);
    bool enableAdaptiveProcessing(bool enable);
    bool enableMultiChannelProcessing(bool enable, uint16_t channels);

    // Preset management
    bool savePreset(const std::string& name, const std::string& description = "");
    bool loadPreset(const std::string& name);
    bool deletePreset(const std::string& name);
    std::vector<std::string> getAvailablePresets() const;

    // Export and import
    bool exportImpulseResponse(const std::string& name, const std::string& filePath) const;
    bool exportConfiguration(const std::string& filePath) const;
    bool importConfiguration(const std::string& filePath);

    // Statistics and monitoring
    ConvolutionStatistics getStatistics() const;
    void resetStatistics();
    bool isHealthy() const;
    std::vector<std::string> getDiagnosticMessages() const;

    // Callbacks
    using IRChangedCallback = std::function<void(const std::string&, const ImpulseResponse&)>;
    using ProcessingCallback = std::function<void(const float*, float*, size_t)>;
    using ErrorCallback = std::function<void(const std::string&, const std::string&)>;

    void setIRChangedCallback(IRChangedCallback callback);
    void setProcessingCallback(ProcessingCallback callback);
    void setErrorCallback(ErrorCallback callback);

private:
    // Core processing
    void initializePartitions();
    void initializeFFT();
    void initializeGPUResources();
    void cleanupGPUResources();

    // Convolution methods
    void processOverlapAdd(const float* input, float* output, size_t numSamples);
    void processOverlapSave(const float* input, float* output, size_t numSamples);
    void processUniformPartitioned(const float* input, float* output, size_t numSamples);
    void processNonUniformPartitioned(const float* input, float* output, size_t numSamples);
    void processFrequencyDomain(const float* input, float* output, size_t numSamples);

    // FFT operations
    void computeFFT(std::vector<std::complex<float>>& data);
    void computeIFFT(std::vector<std::complex<float>>& data);
    void computeRealFFT(const std::vector<float>& input, std::vector<std::complex<float>>& output);
    void computeRealIFFT(const std::vector<std::complex<float>>& input, std::vector<float>& output);
    void applyWindowFunction(std::vector<float>& data, const std::string& windowType);

    // IR processing
    bool processImpulseResponse(ImpulseResponse& ir);
    bool convertToFrequencyDomain(ImpulseResponse& ir);
    bool normalizeImpulseResponse(ImpulseResponse& ir, float targetLevel);
    bool analyzeImpulseResponseProperties(ImpulseResponse& ir);
    bool partitionImpulseResponse(const ImpulseResponse& ir, std::vector<std::vector<std::complex<float>>>& partitions);

    // GPU processing
    class GPUConvolutionProcessor;
    std::unique_ptr<GPUConvolutionProcessor> gpuProcessor_;
    bool processWithGPU(const float* input, float* output, size_t numSamples);
    bool uploadIRToGPU(const std::vector<std::complex<float>>& irData);
    bool syncGPUState();

    // Memory management
    struct ProcessingBuffers {
        std::vector<float> inputBuffer;
        std::vector<float> outputBuffer;
        std::vector<float> intermediateBuffer;
        std::vector<std::complex<float>> fftBuffer;
        std::vector<std::complex<float>> irBuffer;
        std::vector<float> overlapBuffer;
        std::vector<float> windowBuffer;
    };

    ProcessingBuffers buffers_;
    mutable std::mutex buffersMutex_;

    // Multi-threading
    void processThreaded(const float* input, float* output, size_t numSamples, uint16_t channels);
    void processChannelThread(const float* input, float* output, size_t numSamples,
                             uint16_t channel);
    std::vector<std::thread> processingThreads_;
    std::atomic<bool> multithreadingEnabled_{false};

    // State management
    ConvolutionConfig config_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> processingActive_{false};
    std::string currentIRName_;
    std::map<std::string, ImpulseResponse> impulseResponses_;
    mutable std::mutex irMutex_;

    // Processing parameters
    float wetLevel_ = 1.0f;
    float dryLevel_ = 0.0f;
    float gain_ = 1.0f;
    float predelay_ = 0.0f;
    bool normalizationEnabled_ = true;
    bool zeroLatencyMonitoring_ = false;

    // FFT configuration
    std::vector<std::complex<float>> fftTwiddleFactors_;
    std::vector<float> fftWindow_;
    uint32_t fftSize_ = 0;
    uint32_t numPartitions_ = 0;

    // Statistics
    mutable std::mutex statsMutex_;
    ConvolutionStatistics statistics_;
    std::chrono::steady_clock::time_point lastUpdateTime_;

    // GPU resources
    void* gpuIRBuffer_ = nullptr;
    void* gpuInputBuffer_ = nullptr;
    void* gpuOutputBuffer_ = nullptr;
    void* gpuWorkBuffer_ = nullptr;
    size_t gpuMemorySize_ = 0;

    // Callbacks
    IRChangedCallback irChangedCallback_;
    ProcessingCallback processingCallback_;
    ErrorCallback errorCallback_;

    // Utility functions
    uint32_t calculateOptimalFFTSize(uint32_t irLength, uint32_t blockSize) const;
    uint32_t calculateNumPartitions(uint32_t irLength, uint32_t fftSize) const;
    float estimateLatency(uint32_t fftSize, uint32_t blockSize) const;
    bool isValidSampleRate(uint32_t sampleRate) const;
    float dbToLinear(float db) const;
    float linearToDb(float linear) const;

    // Error handling
    void setError(const std::string& error) const;
    std::string getLastError() const;
    mutable std::string lastError_;
    std::vector<std::string> diagnosticMessages_;

    // Constants
    static constexpr uint32_t MAX_IR_LENGTH = 16777216;      // 16 million points
    static constexpr uint32_t MIN_FFT_SIZE = 64;
    static constexpr uint32_t MAX_FFT_SIZE = 134217728;    // 128M points
    static constexpr float MAX_LATENCY_MS = 1000.0f;        // 1 second maximum latency
    static constexpr uint32_t MAX_CHANNELS = 32;
};

/**
 * @brief GPU-accelerated convolution processor
 *
 * Implements the GPU processing pipeline for massive convolution operations,
 * utilizing CUDA, OpenCL, or Vulkan for parallel processing of ultra-large
 * impulse responses.
 */
class ConvolutionEngine::GPUConvolutionProcessor {
public:
    GPUConvolutionProcessor(const ConvolutionConfig& config);
    ~GPUConvolutionProcessor();

    bool initialize();
    void shutdown();
    bool isInitialized() const;

    bool uploadImpulseResponse(const std::vector<std::complex<float>>& irData, uint32_t channels);
    bool processAudio(const float* input, float* output, size_t numSamples, uint16_t channels);
    bool synchronize();
    float getProcessingTime() const;
    float getGPUUtilization() const;

private:
    struct GPUContext {
        void* device = nullptr;
        void* context = nullptr;
        void* commandQueue = nullptr;
        void* memoryManager = nullptr;
        void* cuBLASHandle = nullptr;
        void* cuFFTHandle = nullptr;
        bool initialized = false;
    };

    bool initializeCUDA();
    bool initializeOpenCL();
    bool initializeVulkan();

    void processWithCUDA(const float* input, float* output, size_t numSamples, uint16_t channels);
    void processWithOpenCL(const float* input, float* output, size_t numSamples, uint16_t channels);
    void processWithVulkan(const float* input, float* output, size_t numSamples, uint16_t channels);

    bool setupCUDAFFTPattern();
    bool setupOpenCLKernel();
    bool setupVulkanComputeShader();

    ConvolutionConfig config_;
    GPUContext gpuContext_;
    std::atomic<bool> initialized_{false};

    // GPU buffers
    void* irBuffer_ = nullptr;
    void* inputBuffer_ = nullptr;
    void* outputBuffer_ = nullptr;
    void* workBuffer_ = nullptr;
    void* fftPlan_ = nullptr;
    void* ifftPlan_ = nullptr;

    size_t bufferSize_ = 0;
    size_t irSize_ = 0;
    uint16_t numChannels_ = 0;

    std::atomic<float> processingTime_{0.0f};
    std::atomic<float> gpuUtilization_{0.0f};

    // CUDA-specific
    struct CUDAContext {
        cufftHandle fftHandle = nullptr;
        cufftHandle ifftHandle = nullptr;
        cublasHandle_t blasHandle = nullptr;
        float* deviceInput = nullptr;
        float* deviceOutput = nullptr;
        cuComplex* deviceComplex = nullptr;
    };

    // OpenCL-specific
    struct OpenCLContext {
        cl_context context = nullptr;
        cl_device_id device = nullptr;
        cl_command_queue queue = nullptr;
        cl_program program = nullptr;
        cl_kernel convolutionKernel = nullptr;
        cl_mem inputBuffer = nullptr;
        cl_mem outputBuffer = nullptr;
        cl_mem irBuffer = nullptr;
    };

    // Vulkan-specific
    struct VulkanContext {
        void* instance = nullptr;
        void* device = nullptr;
        void* queue = nullptr;
        void* descriptorPool = nullptr;
        void* computePipeline = nullptr;
        void* inputBuffer = nullptr;
        void* outputBuffer = nullptr;
        void* irBuffer = nullptr;
    };
};

} // namespace vortex