#include "audio_engine.hpp"
#include "system/logger.hpp"
#include "system/config_manager.hpp"

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_dsp/juce_dsp.h>

namespace vortex {

AudioEngine::AudioEngine()
    : sampleRate_(44100.0)
    , bufferSize_(512)
    , channels_(2)
    , bitDepth_(24)
    , isInitialized_(false)
    , gpuEnabled_(false)
    , processingChain_(std::make_unique<ProcessingChain>())
    , bufferManager_(std::make_unique<AudioBufferManager>())
    , deviceManager_(std::make_unique<juce::AudioDeviceManager>())
    , formatManager_(std::make_unique<juce::AudioFormatManager>())
{
    Logger::info("AudioEngine constructor started");

    // Register audio formats
    formatManager_->registerBasicFormats();

    // Configure default audio device setup
    auto setup = deviceManager_->getAudioDeviceSetup();
    setup.sampleRate = sampleRate_;
    setup.bufferSize = bufferSize_;
    setup.inputChannels = channels_;
    setup.outputChannels = channels_;

    Logger::info("AudioEngine constructor completed");
}

AudioEngine::~AudioEngine() {
    shutdown();
    Logger::info("AudioEngine destroyed");
}

bool AudioEngine::initialize(double sampleRate, int bufferSize) {
    if (isInitialized_) {
        Logger::warning("AudioEngine already initialized");
        return true;
    }

    Logger::info("Initializing AudioEngine with sampleRate: {}, bufferSize: {}", sampleRate, bufferSize);

    try {
        sampleRate_ = sampleRate;
        bufferSize_ = bufferSize;

        // Initialize audio device manager
        if (!initializeAudioDevice()) {
            Logger::error("Failed to initialize audio device");
            return false;
        }

        // Initialize buffer manager
        if (!bufferManager_->initialize(sampleRate, bufferSize, channels_, bitDepth_)) {
            Logger::error("Failed to initialize buffer manager");
            return false;
        }

        // Initialize processing chain
        if (!processingChain_->initialize(sampleRate, bufferSize, channels_)) {
            Logger::error("Failed to initialize processing chain");
            return false;
        }

        // Start audio processing thread
        processingThread_ = std::make_unique<std::thread>(&AudioEngine::processingLoop, this);

        isInitialized_ = true;
        Logger::info("AudioEngine initialized successfully");
        return true;

    } catch (const std::exception& e) {
        Logger::error("AudioEngine initialization failed: {}", e.what());
        return false;
    }
}

void AudioEngine::shutdown() {
    if (!isInitialized_) {
        return;
    }

    Logger::info("Shutting down AudioEngine");

    // Stop processing thread
    if (processingThread_ && processingThread_->joinable()) {
        stopProcessing_.store(true);
        processingThread_->join();
        processingThread_.reset();
    }

    // Shutdown components in reverse order
    if (processingChain_) {
        processingChain_->shutdown();
    }

    if (bufferManager_) {
        bufferManager_->shutdown();
    }

    if (deviceManager_) {
        deviceManager_->closeAudioDevice();
    }

    isInitialized_ = false;
    Logger::info("AudioEngine shutdown completed");
}

bool AudioEngine::initializeAudioDevice() {
    Logger::info("Initializing audio device");

    // Set up audio device type
    std::vector<std::string> deviceTypes = {
        juce::AudioIODeviceType::getTypeForWaveOut(),  // Windows WaveOut
        juce::AudioIODeviceType::getTypeForDirectSound(),  // Windows DirectSound
        juce::AudioIODeviceType::getTypeForASIO(),  // ASIO
        juce::AudioIODeviceType::getTypeForCoreAudio(),  // macOS CoreAudio
        juce::AudioIODeviceType::getTypeForALSA(),  // Linux ALSA
        juce::AudioIODeviceType::getTypeForJack(),  // JACK
        juce::AudioIODeviceType::getTypeForPulseAudio()  // Linux PulseAudio
    };

    auto setup = deviceManager_->getAudioDeviceSetup();
    setup.sampleRate = sampleRate_;
    setup.bufferSize = bufferSize_;
    setup.inputChannels = channels_;
    setup.outputChannels = channels_;
    setup.useDefaultInputChannels = false;
    setup.useDefaultOutputChannels = true;

    // Try to initialize with available device types
    for (const auto& deviceType : deviceTypes) {
        Logger::debug("Trying device type: {}", deviceType);

        auto* type = deviceManager_->availableDeviceTypes.find(deviceType);
        if (type != nullptr) {
            deviceManager_->setCurrentAudioDeviceType(deviceType, true);

            juce::String error = deviceManager_->initialise(
                channels_,  // input channels
                channels_,  // output channels
                nullptr,    // no XML
                true,       // select default device
                nullptr     // no preferred setup
            );

            if (error.isEmpty()) {
                Logger::info("Audio device initialized with type: {}", deviceType);
                return true;
            } else {
                Logger::warning("Failed to initialize with {}: {}", deviceType, error.toStdString());
                deviceManager_->closeAudioDevice();
            }
        }
    }

    Logger::error("Failed to initialize any audio device");
    return false;
}

bool AudioEngine::enableGPUAcceleration(const std::string& backend) {
    if (!isInitialized_) {
        Logger::error("AudioEngine not initialized");
        return false;
    }

    Logger::info("Enabling GPU acceleration with backend: {}", backend);

    try {
        // Check GPU backend availability
        if (!isGPUBackendAvailable(backend)) {
            Logger::error("GPU backend {} is not available", backend);
            return false;
        }

        // Create GPU processor
        gpuProcessor_ = std::make_unique<GPUProcessor>();

        if (!gpuProcessor_->initialize(backend, sampleRate_, bufferSize_, channels_)) {
            Logger::error("Failed to initialize GPU processor");
            gpuProcessor_.reset();
            return false;
        }

        // Configure processing chain for GPU acceleration
        processingChain_->enableGPUAcceleration(gpuProcessor_.get());

        gpuEnabled_ = true;
        gpuBackend_ = backend;

        Logger::info("GPU acceleration enabled successfully with backend: {}", backend);
        return true;

    } catch (const std::exception& e) {
        Logger::error("Failed to enable GPU acceleration: {}", e.what());
        gpuProcessor_.reset();
        return false;
    }
}

bool AudioEngine::isGPUBackendAvailable(const std::string& backend) const {
    if (backend == "CUDA") {
        return checkCUDAAvailability();
    } else if (backend == "OpenCL") {
        return checkOpenCLAvailability();
    } else if (backend == "Vulkan") {
        return checkVulkanAvailability();
    }
    return false;
}

bool AudioEngine::checkCUDAAvailability() const {
#ifdef VORTEX_ENABLE_CUDA
    try {
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        return error == cudaSuccess && deviceCount > 0;
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

bool AudioEngine::checkOpenCLAvailability() const {
#ifdef VORTEX_ENABLE_OPENCL
    try {
        cl_uint numPlatforms;
        cl_int error = clGetPlatformIDs(0, nullptr, &numPlatforms);
        return error == CL_SUCCESS && numPlatforms > 0;
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

bool AudioEngine::checkVulkanAvailability() const {
#ifdef VORTEX_ENABLE_VULKAN
    try {
        vk::ApplicationInfo appInfo;
        vk::InstanceCreateInfo createInfo;
        vk::Instance instance = vk::createInstance(createInfo);
        instance.destroy();
        return true;
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

void AudioEngine::processBuffer(const float* input, float* output, size_t numSamples) {
    if (!isInitialized_) {
        Logger::error("AudioEngine not initialized for processing");
        return;
    }

    try {
        // Get processing buffer from manager
        auto buffer = bufferManager_->getProcessingBuffer(numSamples);

        // Copy input to processing buffer
        std::copy(input, input + numSamples, buffer.data());

        // Apply processing chain
        if (gpuEnabled_ && gpuProcessor_) {
            processingChain_->processGPU(buffer.data(), numSamples, gpuProcessor_.get());
        } else {
            processingChain_->processCPU(buffer.data(), numSamples);
        }

        // Copy result to output
        std::copy(buffer.data(), buffer.data() + numSamples, output);

        // Return buffer to manager
        bufferManager_->returnProcessingBuffer(std::move(buffer));

    } catch (const std::exception& e) {
        Logger::error("Audio processing failed: {}", e.what());
        // Fallback: copy input to output
        std::copy(input, input + numSamples, output);
    }
}

bool AudioEngine::loadAudioFile(const std::string& filePath) {
    if (!isInitialized_) {
        Logger::error("AudioEngine not initialized");
        return false;
    }

    Logger::info("Loading audio file: {}", filePath);

    try {
        juce::File file(filePath);
        if (!file.existsAsFile()) {
            Logger::error("Audio file does not exist: {}", filePath);
            return false;
        }

        auto* reader = formatManager_->createReaderFor(file);
        if (reader == nullptr) {
            Logger::error("Failed to create audio reader for: {}", filePath);
            return false;
        }

        // Extract metadata
        AudioMetadata metadata;
        metadata.title = reader->getMetadataValue("title").toStdString();
        metadata.artist = reader->getMetadataValue("artist").toStdString();
        metadata.album = reader->getMetadataValue("album").toStdString();
        metadata.sampleRate = static_cast<int>(reader->sampleRate);
        metadata.channels = reader->numChannels;
        metadata.bitDepth = reader->bitsPerSample;
        metadata.numSamples = static_cast<size_t>(reader->lengthInSamples);

        // Determine audio format
        std::string fileExtension = file.getFileExtension().toLowerCase().toStdString();
        if (fileExtension == ".wav") {
            metadata.format = AudioFormat::WAV;
        } else if (fileExtension == ".flac") {
            metadata.format = AudioFormat::FLAC;
        } else if (fileExtension == ".mp3") {
            metadata.format = AudioFormat::MP3;
        } else if (fileExtension == ".aac") {
            metadata.format = AudioFormat::AAC;
        } else if (fileExtension == ".ogg") {
            metadata.format = AudioFormat::OGG;
        } else if (fileExtension == ".opus") {
            metadata.format = AudioFormat::OPUS;
        } else {
            metadata.format = AudioFormat::PCM;
        }

        // Calculate duration
        if (metadata.sampleRate > 0) {
            metadata.durationSeconds = static_cast<double>(metadata.numSamples) / metadata.sampleRate;
        }

        // Store metadata
        currentAudioMetadata_ = metadata;

        // Load audio data
        size_t totalSamples = metadata.numSamples * metadata.channels;
        currentAudioData_.resize(totalSamples);

        juce::AudioBuffer<float> audioBuffer(metadata.channels, static_cast<int>(metadata.numSamples));
        reader->read(&audioBuffer, 0, static_cast<int>(metadata.numSamples), 0, true, true);

        // Interleave channels
        for (size_t sample = 0; sample < metadata.numSamples; ++sample) {
            for (int channel = 0; channel < metadata.channels; ++channel) {
                currentAudioData_[sample * metadata.channels + channel] =
                    audioBuffer.getSample(channel, static_cast<int>(sample));
            }
        }

        delete reader; // Clean up

        Logger::info("Audio file loaded successfully: {} ({} samples, {} channels, {} Hz)",
                     filePath, metadata.numSamples, metadata.channels, metadata.sampleRate);

        return true;

    } catch (const std::exception& e) {
        Logger::error("Failed to load audio file {}: {}", filePath, e.what());
        return false;
    }
}

std::vector<float> AudioEngine::processAudioSegment(size_t startSample, size_t numSamples) {
    std::vector<float> output(numSamples * channels_);

    if (!isInitialized_ || currentAudioData_.empty()) {
        Logger::warning("No audio data available for processing");
        return output;
    }

    try {
        // Check bounds
        size_t availableSamples = currentAudioData_.size() / channels_;
        if (startSample + numSamples > availableSamples) {
            numSamples = availableSamples - startSample;
            output.resize(numSamples * channels_);
        }

        if (numSamples == 0) {
            return output;
        }

        // Extract segment
        std::vector<float> input(numSamples * channels_);
        std::copy(currentAudioData_.begin() + startSample * channels_,
                  currentAudioData_.begin() + (startSample + numSamples) * channels_,
                  input.begin());

        // Process segment
        processBuffer(input.data(), output.data(), numSamples * channels_);

        return output;

    } catch (const std::exception& e) {
        Logger::error("Failed to process audio segment: {}", e.what());
        return output;
    }
}

std::vector<float> AudioEngine::getCurrentSpectrum(size_t fftSize) {
    if (!isInitialized_ || currentAudioData_.empty()) {
        return std::vector<float>(fftSize / 2, 0.0f);
    }

    try {
        // Get recent audio samples for spectrum analysis
        size_t windowSize = std::min(fftSize, currentAudioData_.size());
        std::vector<float> window(windowSize);

        size_t startSample = (currentAudioData_.size() / channels_ > windowSize) ?
                           (currentAudioData_.size() / channels_ - windowSize) : 0;

        // Copy window (use first channel for spectrum analysis)
        for (size_t i = 0; i < windowSize; ++i) {
            window[i] = currentAudioData_[(startSample + i) * channels_];
        }

        // Apply window function (Hanning)
        for (size_t i = 0; i < windowSize; ++i) {
            window[i] *= 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (windowSize - 1)));
        }

        // Compute FFT
        return computeSpectrum(window);

    } catch (const std::exception& e) {
        Logger::error("Failed to compute spectrum: {}", e.what());
        return std::vector<float>(fftSize / 2, 0.0f);
    }
}

std::vector<float> AudioEngine::computeSpectrum(const std::vector<float>& audio) {
    size_t n = audio.size();
    size_t fftSize = 1;
    while (fftSize < n) fftSize <<= 1;

    std::vector<std::complex<float>> fftData(fftSize);

    // Copy and zero-pad
    for (size_t i = 0; i < n; ++i) {
        fftData[i] = std::complex<float>(audio[i], 0.0f);
    }
    for (size_t i = n; i < fftSize; ++i) {
        fftData[i] = std::complex<float>(0.0f, 0.0f);
    }

    // Apply FFT (Cooley-Tukey algorithm)
    fft(fftData);

    // Compute magnitude spectrum
    std::vector<float> spectrum(fftSize / 2);
    for (size_t i = 0; i < fftSize / 2; ++i) {
        spectrum[i] = std::abs(fftData[i]) / fftSize;
    }

    return spectrum;
}

void AudioEngine::fft(std::vector<std::complex<float>>& data) {
    size_t n = data.size();

    // Bit reversal
    for (size_t i = 0, j = 0; i < n; ++i) {
        if (i < j) {
            std::swap(data[i], data[j]);
        }
        size_t m = n >> 1;
        while (j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // Cooley-Tukey FFT
    for (size_t len = 2; len <= n; len <<= 1) {
        float angle = -2.0f * M_PI / len;
        std::complex<float> wlen(std::cos(angle), std::sin(angle));

        for (size_t i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (size_t j = 0; j < len / 2; ++j) {
                std::complex<float> u = data[i + j];
                std::complex<float> v = data[i + j + len / 2] * w;
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

void AudioEngine::processingLoop() {
    Logger::info("Audio processing loop started");

    while (!stopProcessing_.load()) {
        try {
            // This would be the real-time audio processing loop
            // In a real implementation, this would be driven by audio callbacks
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

        } catch (const std::exception& e) {
            Logger::error("Exception in processing loop: {}", e.what());
        }
    }

    Logger::info("Audio processing loop stopped");
}

// Getters
double AudioEngine::getSampleRate() const { return sampleRate_; }
int AudioEngine::getBufferSize() const { return bufferSize_; }
int AudioEngine::getChannels() const { return channels_; }
bool AudioEngine::isInitialized() const { return isInitialized_; }
bool AudioEngine::isGPUEnabled() const { return gpuEnabled_; }
const AudioMetadata& AudioEngine::getCurrentMetadata() const { return currentAudioMetadata_; }
size_t AudioEngine::getCurrentAudioLength() const { return currentAudioData_.size() / channels_; }

} // namespace vortex