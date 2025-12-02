#pragma once

#include <vector>
#include <complex>
#include <memory>
#include <atomic>
#include <functional>
#include <cstring>
#include "system/logger.hpp"

namespace vortex::core::dsp {

/**
 * High-performance spectrum analyzer with GPU acceleration
 *
 * Features:
 * - Real-time FFT analysis at 60+ FPS for visualization
 * - GPU acceleration via CUDA/OpenCL/Vulkan compute shaders
 * - Multiple window functions (Hanning, Hamming, Blackman, etc.)
 * - Configurable frequency scaling (linear/logarithmic)
 * - Amplitude scaling (linear/dB/square root)
 * - Overlap-add processing for smooth transitions
 * - Multi-channel support
 * - Memory-efficient processing pipeline
 */
class SpectrumAnalyzer {
public:
    /**
     * Window function types
     */
    enum class WindowType {
        Rectangular,
        Hanning,
        Hamming,
        Blackman,
        BlackmanHarris,
        Kaiser,
        FlatTop
    };

    /**
     * Frequency scaling modes
     */
    enum class FrequencyScale {
        Linear,
        Logarithmic,
        Mel,
        Octave
    };

    /**
     * Amplitude scaling modes
     */
    enum class AmplitudeScale {
        Linear,
        Decibel,
        SquareRoot
    };

    /**
     * Processing mode
     */
    enum class ProcessingMode {
        CPU,
        GPU,
        Auto  // Automatically choose best available
    };

    /**
     * Configuration structure
     */
    struct Config {
        size_t fftSize = 2048;              // FFT size (must be power of 2)
        size_t hopSize = 512;               // Hop size for overlap processing
        WindowType windowType = WindowType::Hanning;
        FrequencyScale frequencyScale = FrequencyScale::Logarithmic;
        AmplitudeScale amplitudeScale = AmplitudeScale::Decibel;
        ProcessingMode processingMode = ProcessingMode::Auto;
        size_t numFrequencyBands = 256;     // Output frequency bands for log scale
        float sampleRate = 44100.0f;
        int channels = 2;
        bool enableWindowing = true;
        bool enableOverlap = true;
        bool normalizeOutput = true;
        float kaiserBeta = 8.0f;            // Kaiser window parameter
    };

    /**
     * Constructor
     */
    SpectrumAnalyzer();

    /**
     * Destructor
     */
    ~SpectrumAnalyzer();

    /**
     * Initialize the spectrum analyzer
     * @param config Configuration parameters
     * @return true if initialization successful
     */
    bool initialize(const Config& config);

    /**
     * Initialize with legacy parameters
     * @param sampleRate Sample rate in Hz
     * @param fftSize FFT size (must be power of 2)
     * @param channels Number of audio channels
     * @return true if initialization successful
     */
    bool initialize(float sampleRate, size_t fftSize, int channels);

    /**
     * Shutdown and cleanup resources
     */
    void shutdown();

    /**
     * Process audio data and return spectrum analysis
     * @param audioData Input audio data (interleaved)
     * @param numSamples Number of samples per channel
     * @return Vector of spectrum data (one vector per channel)
     */
    std::vector<std::vector<float>> processAudio(const float* audioData, size_t numSamples);

    /**
     * Process audio data with existing overlap buffer
     * @param audioData Input audio data
     * @param numSamples Number of samples
     * @param outputSpectrum Output spectrum data
     * @return true if processing successful
     */
    bool processAudio(const float* audioData, size_t numSamples,
                     std::vector<std::vector<float>>& outputSpectrum);

    /**
     * Set window function type
     * @param windowType Window function to use
     */
    void setWindowType(WindowType windowType);

    /**
     * Get current window type
     */
    WindowType getWindowType() const;

    /**
     * Set frequency scaling mode
     * @param frequencyScale Frequency scaling mode
     */
    void setFrequencyScale(FrequencyScale frequencyScale);

    /**
     * Get current frequency scaling mode
     */
    FrequencyScale getFrequencyScale() const;

    /**
     * Set amplitude scaling mode
     * @param amplitudeScale Amplitude scaling mode
     */
    void setAmplitudeScale(AmplitudeScale amplitudeScale);

    /**
     * Get current amplitude scaling mode
     */
    AmplitudeScale getAmplitudeScale() const;

    /**
     * Set processing mode (CPU/GPU/Auto)
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
     * Get frequency resolution in Hz
     */
    double getFrequencyResolution() const;

    /**
     * Get frequency band centers for logarithmic scaling
     */
    std::vector<double> getFrequencyBandCenters() const;

    /**
     * Get performance statistics
     */
    std::string getPerformanceStats() const;

    /**
     * Reset internal state and buffers
     */
    void reset();

    /**
     * Check if analyzer is initialized
     */
    bool isInitialized() const;

    /**
     * Set overlap ratio (0.0 to 0.875)
     * @param ratio Overlap ratio
     */
    void setOverlapRatio(float ratio);

    /**
     * Get current overlap ratio
     */
    float getOverlapRatio() const;

    /**
     * Enable/disable windowing
     * @param enabled Whether to apply window function
     */
    void setWindowingEnabled(bool enabled);

    /**
     * Check if windowing is enabled
     */
    bool isWindowingEnabled() const;

private:
    Config config_;
    bool initialized_;

    // Processing buffers
    std::vector<float> window_;
    std::vector<float> inputBuffer_;
    std::vector<std::complex<float>> fftBuffer_;
    std::vector<float> magnitudeBuffer_;
    std::vector<float> outputBuffer_;

    // Overlap processing
    std::vector<float> overlapBuffer_;
    size_t overlapSize_;
    size_t writePos_;
    size_t readPos_;

    // Frequency scaling
    std::vector<float> frequencyMapping_;
    std::vector<double> frequencyBandCenters_;
    bool frequencyMappingDirty_;

    // Performance tracking
    mutable std::atomic<uint64_t> totalFramesProcessed_;
    mutable std::atomic<double> totalProcessingTime_;
    mutable std::atomic<uint64_t> gpuFramesProcessed_;
    mutable std::atomic<uint64_t> cpuFramesProcessed_;

    // GPU resources
    void* gpuFFTPlan_;
    void* gpuInputBuffer_;
    void* gpuOutputBuffer_;
    void* gpuWindowBuffer_;
    bool gpuInitialized_;

    // Internal methods
    bool initializeWindow();
    bool initializeOverlapBuffer();
    bool initializeFrequencyMapping();
    bool initializeGPUResources();
    void cleanupGPUResources();

    void applyWindow(float* data, size_t length);
    void computeFFT(const float* input, std::complex<float>* output, size_t length);
    void computeMagnitude(const std::complex<float>* fftData, float* magnitude, size_t length);
    void scaleAmplitude(float* data, size_t length);
    void mapFrequencyBands(const float* input, float* output, size_t inputLength, size_t outputLength);

    bool processAudioCPU(const float* audioData, size_t numSamples,
                         std::vector<std::vector<float>>& outputSpectrum);
    bool processAudioGPU(const float* audioData, size_t numSamples,
                        std::vector<std::vector<float>>& outputSpectrum);

    void updatePerformanceStats(double processingTimeMs, bool usedGPU) const;

    // Window function generation
    void generateRectangularWindow(float* window, size_t length);
    void generateHanningWindow(float* window, size_t length);
    void generateHammingWindow(float* window, size_t length);
    void generateBlackmanWindow(float* window, size_t length);
    void generateBlackmanHarrisWindow(float* window, size_t length);
    void generateKaiserWindow(float* window, size_t length, float beta);
    void generateFlatTopWindow(float* window, size_t length);

    // Frequency band calculation
    void calculateLogarithmicBands();
    void calculateMelBands();
    void calculateOctaveBands();

    // FFT implementation fallbacks
    void computeFFTCPU(const float* input, std::complex<float>* output, size_t length);
    void computeFFTGPU(const float* input, std::complex<float>* output, size_t length);

    // Utility methods
    bool isPowerOfTwo(size_t n) const;
    size_t nextPowerOfTwo(size_t n) const;
    float hanningWindow(float n, size_t N) const;
    float kaiserBessel(float x) const;
};

} // namespace vortex::core::dsp