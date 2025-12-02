#pragma once

#include <memory>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include "system/logger.hpp"

namespace vortex::core::dsp {

/**
 * DSD1024 processor for ultra-high-resolution Direct Stream Digital audio
 * Supports processing at 45.1584 MHz sample rate with GPU acceleration
 * Implements advanced multi-stage filtering and decimation for PCM conversion
 * Optimized for real-time processing with sub-millisecond latency
 * Features automatic gain control, noise shaping, and dynamic range optimization
 */
class DSDProcessor {
public:
    /**
     * DSD processing configuration
     */
    struct Config {
        uint32_t inputSampleRate = 45158400;    // DSD1024: 45.1584 MHz
        uint32_t outputSampleRate = 352800;      // Target: 352.8 kHz (8:1 decimation)
        uint32_t channels = 2;                   // Stereo
        uint32_t blockSize = 8192;               // Processing block size
        bool gpuAcceleration = true;             // Use GPU when available
        bool noiseShaping = true;                // Apply noise shaping
        float gainNormalization = 0.707f;        // -3dB gain
        uint32_t filterOrder = 8;                // FIR filter order

        // Advanced processing options
        bool enableDynamicRange = true;          // Dynamic range compression
        float compressionRatio = 2.0f;           // Compression ratio
        float threshold = -24.0f;                // Compression threshold in dB
        bool enableDithering = true;             // Add dithering noise
        uint32_t ditherBits = 24;                // Dithering bit depth
    };

    /**
     * DSD processing state and buffers
     */
    struct ProcessingState {
        std::vector<float> inputBuffer;          // DSD input samples
        std::vector<float> outputBuffer;         // PCM output samples
        std::vector<float> filterCoeffs;         // FIR filter coefficients
        std::vector<float> delayLine;            // Filter delay line
        std::vector<std::complex<float>> fftBuffer; // FFT workspace

        // GPU memory handles
        void* gpuInputBuffer = nullptr;
        void* gpuOutputBuffer = nullptr;
        void* gpuCoefficients = nullptr;
        void* gpuWorkBuffer = nullptr;

        size_t delayIndex = 0;
        bool gpuInitialized = false;

        ProcessingState() = default;
        ~ProcessingState();

        // Disable copying for GPU memory safety
        ProcessingState(const ProcessingState&) = delete;
        ProcessingState& operator=(const ProcessingState&) = delete;

        ProcessingState(ProcessingState&& other) noexcept;
        ProcessingState& operator=(ProcessingState&& other) noexcept;
    };

    /**
     * Constructor
     * @param config Processing configuration
     */
    explicit DSDProcessor(const Config& config = Config{});

    /**
     * Destructor
     */
    ~DSDProcessor();

    /**
     * Initialize the DSD processor
     * Allocates buffers, initializes GPU resources, and prepares filters
     * @return true if initialization successful, false otherwise
     */
    bool initialize();

    /**
     * Shutdown the DSD processor
     * Releases GPU resources and cleans up buffers
     */
    void shutdown();

    /**
     * Process DSD1024 audio data to PCM
     * Performs multi-stage decimation, filtering, and format conversion
     * @param input Pointer to DSD input data (1-bit samples interleaved)
     * @param inputSamples Number of input DSD samples per channel
     * @param output Pointer to output buffer for PCM data
     * @param maxOutputSamples Maximum number of output samples to generate
     * @return Number of output samples generated, or 0 on error
     */
    uint32_t processDSD1024(const uint8_t* input,
                           uint32_t inputSamples,
                           float* output,
                           uint32_t maxOutputSamples);

    /**
     * Process DSD data with real-time constraints
     * Optimized for low-latency audio processing
     * @param input Input DSD data
     * @param inputSize Size of input data in bytes
     * @param output Output buffer for PCM data
     * @param outputCapacity Capacity of output buffer in samples
     * @return Number of samples processed, or 0 on error
     */
    uint32_t processRealTime(const uint8_t* input,
                            size_t inputSize,
                            float* output,
                            size_t outputCapacity);

    /**
     * Reset processing state
     * Clears filters and delay lines for clean processing start
     */
    void reset();

    /**
     * Update processing configuration
     * @param newConfig New configuration parameters
     * @return true if configuration update successful, false otherwise
     */
    bool updateConfig(const Config& newConfig);

    /**
     * Get current processing statistics
     * @return Processing statistics as JSON string
     */
    std::string getProcessingStats() const;

    /**
     * Check if GPU acceleration is available
     * @return true if GPU processing is enabled and available
     */
    bool isGPUAccelerated() const { return state_.gpuInitialized && config_.gpuAcceleration; }

    /**
     * Get estimated processing latency
     * @return Processing latency in microseconds
     */
    uint64_t getEstimatedLatency() const;

    /**
     * Get current configuration
     * @return Current processing configuration
     */
    const Config& getConfig() const { return config_; }

private:
    Config config_;
    ProcessingState state_;
    bool initialized_;

    // Performance metrics
    mutable uint64_t totalSamplesProcessed_;
    mutable double totalProcessingTime_;
    mutable uint32_t gpuFallbacks_;

    /**
     * Initialize FIR filter coefficients
     * Creates low-pass filter for DSD to PCM conversion
     */
    void initializeFilterCoefficients();

    /**
     * Initialize GPU resources
     * Allocates GPU memory and prepares kernels
     * @return true if GPU initialization successful, false otherwise
     */
    bool initializeGPU();

    /**
     * Cleanup GPU resources
     * Releases GPU memory and contexts
     */
    void cleanupGPU();

    /**
     * Process using CPU implementation
     * Fallback CPU processing when GPU is not available
     * @param input Input DSD data
     * @param inputSamples Number of input samples
     * @param output Output buffer
     * @param maxOutputSamples Maximum output samples
     * @return Number of output samples generated
     */
    uint32_t processCPU(const uint8_t* input,
                       uint32_t inputSamples,
                       float* output,
                       uint32_t maxOutputSamples);

    /**
     * Process using GPU implementation
     * Accelerated processing using CUDA/OpenCL
     * @param input Input DSD data
     * @param inputSamples Number of input samples
     * @param output Output buffer
     * @param maxOutputSamples Maximum output samples
     * @return Number of output samples generated
     */
    uint32_t processGPU(const uint8_t* input,
                       uint32_t inputSamples,
                       float* output,
                       uint32_t maxOutputSamples);

    /**
     * Apply noise shaping filter
     * Reduces quantization noise in output audio
     * @param data Audio data to process
     * @param samples Number of samples
     */
    void applyNoiseShaping(float* data, uint32_t samples);

    /**
     * Apply dynamic range compression
     * Reduces dynamic range for better perceived loudness
     * @param data Audio data to process
     * @param samples Number of samples
     */
    void applyDynamicRangeCompression(float* data, uint32_t samples);

    /**
     * Apply dithering
     * Adds low-level noise to reduce quantization artifacts
     * @param data Audio data to process
     * @param samples Number of samples
     */
    void applyDithering(float* data, uint32_t samples);

    /**
     * Convert DSD bits to float samples
     * Converts 1-bit DSD to float representation
     * @param dsdData DSD byte data
     * @param numBits Number of bits to convert
     * @param output Output float buffer
     */
    void convertDSDBitsToFloat(const uint8_t* dsdData, uint32_t numBits, float* output);

    /**
     * Apply FIR filter
     * Low-pass filtering for anti-aliasing
     * @param input Input samples
     * @param output Output samples
     * @param samples Number of samples to process
     */
    void applyFIRFilter(const float* input, float* output, uint32_t samples);

    /**
     * Downsample audio data
     * Decimation from high sample rate to target rate
     * @param input High-rate input samples
     * @param inputSamples Number of input samples
     * @param output Output buffer
     * @param decimationRatio Downsample ratio
     * @return Number of output samples
     */
    uint32_t downsample(const float* input,
                       uint32_t inputSamples,
                       float* output,
                       uint32_t decimationRatio);

    /**
     * Log processing performance
     * @param processingTime Time taken for processing in microseconds
     * @param samplesProcessed Number of samples processed
     */
    void logPerformance(uint64_t processingTime, uint32_t samplesProcessed) const;
};

} // namespace vortex::core::dsp