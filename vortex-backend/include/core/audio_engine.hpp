#pragma once

#include <memory>
#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <complex>
#include <cmath>

namespace vortex {

enum class AudioFormat {
    PCM = 0,
    FLAC = 1,
    WAV = 2,
    MP3 = 3,
    AAC = 4,
    OGG = 5,
    OPUS = 6,
    DSD64 = 7,
    DSD128 = 8,
    DSD256 = 9,
    DSD512 = 10,
    DSD1024 = 11
};

struct AudioMetadata {
    std::string title;
    std::string artist;
    std::string album;
    std::string genre;
    std::string year;

    AudioFormat format = AudioFormat::PCM;
    int sampleRate = 44100;
    int channels = 2;
    int bitDepth = 16;
    int bitrate = 0;

    size_t numSamples = 0;
    double durationSeconds = 0.0;

    // Additional metadata
    std::map<std::string, std::string> customMetadata;
};

class AudioBufferManager;
class ProcessingChain;
class GPUProcessor;

/**
 * @brief Core audio engine for Vortex GPU Audio Backend
 *
 * This class provides the main audio processing functionality with GPU acceleration
 * support. It handles audio device management, file loading, real-time processing,
 * and spectrum analysis.
 */
class AudioEngine {
public:
    AudioEngine();
    ~AudioEngine();

    // Initialization and shutdown
    bool initialize(double sampleRate = 44100.0, int bufferSize = 512);
    void shutdown();

    // GPU acceleration
    bool enableGPUAcceleration(const std::string& backend = "CUDA");
    bool isGPUBackendAvailable(const std::string& backend) const;

    // Audio processing
    void processBuffer(const float* input, float* output, size_t numSamples);

    // File operations
    bool loadAudioFile(const std::string& filePath);
    std::vector<float> processAudioSegment(size_t startSample, size_t numSamples);

    // Analysis
    std::vector<float> getCurrentSpectrum(size_t fftSize = 2048);

    // Getters
    double getSampleRate() const;
    int getBufferSize() const;
    int getChannels() const;
    bool isInitialized() const;
    bool isGPUEnabled() const;
    const AudioMetadata& getCurrentMetadata() const;
    size_t getCurrentAudioLength() const;

private:
    // Core configuration
    double sampleRate_;
    int bufferSize_;
    int channels_;
    int bitDepth_;
    bool isInitialized_;
    bool gpuEnabled_;
    std::string gpuBackend_;

    // Components
    std::unique_ptr<AudioBufferManager> bufferManager_;
    std::unique_ptr<ProcessingChain> processingChain_;
    std::unique_ptr<GPUProcessor> gpuProcessor_;

    // JUCE components
    std::unique_ptr<juce::AudioDeviceManager> deviceManager_;
    std::unique_ptr<juce::AudioFormatManager> formatManager_;

    // Current audio data
    AudioMetadata currentAudioMetadata_;
    std::vector<float> currentAudioData_;

    // Processing thread
    std::unique_ptr<std::thread> processingThread_;
    std::atomic<bool> stopProcessing_{false};

    // Private methods
    bool initializeAudioDevice();
    bool checkCUDAAvailability() const;
    bool checkOpenCLAvailability() const;
    bool checkVulkanAvailability() const;
    void processingLoop();

    // Audio analysis
    std::vector<float> computeSpectrum(const std::vector<float>& audio);
    static void fft(std::vector<std::complex<float>>& data);
};

} // namespace vortex