#include "core/dsp/spectrum_analyzer.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>

#ifdef VORTEX_ENABLE_CUDA
#include <cuda_runtime.h>
#include <cufft.h>
#endif

#ifdef VORTEX_ENABLE_OPENCL
#include <CL/cl.h>
#endif

namespace vortex::core::dsp {

SpectrumAnalyzer::SpectrumAnalyzer()
    : initialized_(false)
    , overlapSize_(0)
    , writePos_(0)
    , readPos_(0)
    , frequencyMappingDirty_(false)
    , totalFramesProcessed_(0)
    , totalProcessingTime_(0.0)
    , gpuFramesProcessed_(0)
    , cpuFramesProcessed_(0)
    , gpuFFTPlan_(nullptr)
    , gpuInputBuffer_(nullptr)
    , gpuOutputBuffer_(nullptr)
    , gpuWindowBuffer_(nullptr)
    , gpuInitialized_(false) {
}

SpectrumAnalyzer::~SpectrumAnalyzer() {
    shutdown();
}

bool SpectrumAnalyzer::initialize(const Config& config) {
    if (initialized_) {
        Logger::warn("SpectrumAnalyzer already initialized");
        return true;
    }

    // Validate configuration
    if (config.sampleRate <= 0.0f || config.fftSize == 0 ||
        config.channels <= 0 || !isPowerOfTwo(config.fftSize)) {
        Logger::error("Invalid spectrum analyzer configuration");
        return false;
    }

    config_ = config;
    overlapSize_ = static_cast<size_t>(config_.fftSize * config_.hopSize / config_.fftSize);

    Logger::info("Initializing SpectrumAnalyzer: {} Hz FFT, {} channels, {} Hz sample rate",
                 config_.fftSize, config_.channels, config_.sampleRate);

    try {
        // Initialize window function
        if (!initializeWindow()) {
            Logger::error("Failed to initialize window function");
            return false;
        }

        // Initialize processing buffers
        inputBuffer_.resize(config_.fftSize * config_.channels);
        fftBuffer_.resize(config_.fftSize);
        magnitudeBuffer_.resize(config_.fftSize / 2 + 1);
        outputBuffer_.resize(config_.numFrequencyBands);

        // Initialize overlap processing
        if (!initializeOverlapBuffer()) {
            Logger::error("Failed to initialize overlap buffer");
            return false;
        }

        // Initialize frequency mapping
        if (!initializeFrequencyMapping()) {
            Logger::error("Failed to initialize frequency mapping");
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
        Logger::info("SpectrumAnalyzer initialized successfully ({} processing)",
                     config_.processingMode == ProcessingMode::GPU ? "GPU" : "CPU");
        return true;

    } catch (const std::exception& e) {
        Logger::error("Exception during spectrum analyzer initialization: {}", e.what());
        return false;
    }
}

bool SpectrumAnalyzer::initialize(float sampleRate, size_t fftSize, int channels) {
    Config config;
    config.sampleRate = sampleRate;
    config.fftSize = fftSize;
    config.channels = channels;
    config.hopSize = fftSize / 4; // 75% overlap
    return initialize(config);
}

void SpectrumAnalyzer::shutdown() {
    if (!initialized_) {
        return;
    }

    Logger::info("Shutting down SpectrumAnalyzer");

    // Cleanup GPU resources
    cleanupGPUResources();

    // Clear buffers
    window_.clear();
    inputBuffer_.clear();
    fftBuffer_.clear();
    magnitudeBuffer_.clear();
    outputBuffer_.clear();
    overlapBuffer_.clear();
    frequencyMapping_.clear();
    frequencyBandCenters_.clear();

    initialized_ = false;

    // Log final statistics
    if (totalFramesProcessed_ > 0) {
        double avgProcessingTime = totalProcessingTime_ / totalFramesProcessed_;
        Logger::info("SpectrumAnalyzer stats: {} frames processed, avg time: {:.3f}μs, GPU frames: {}, CPU frames: {}",
                     totalFramesProcessed_.load(), avgProcessingTime,
                     gpuFramesProcessed_.load(), cpuFramesProcessed_.load());
    }
}

std::vector<std::vector<float>> SpectrumAnalyzer::processAudio(const float* audioData, size_t numSamples) {
    std::vector<std::vector<float>> result;

    if (!processAudio(audioData, numSamples, result)) {
        result.resize(config_.channels);
        for (int ch = 0; ch < config_.channels; ++ch) {
            result[ch].resize(config_.numFrequencyBands, 0.0f);
        }
    }

    return result;
}

bool SpectrumAnalyzer::processAudio(const float* audioData, size_t numSamples,
                                   std::vector<std::vector<float>>& outputSpectrum) {
    if (!initialized_) {
        Logger::error("SpectrumAnalyzer not initialized");
        return false;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    outputSpectrum.resize(config_.channels);
    for (int ch = 0; ch < config_.channels; ++ch) {
        outputSpectrum[ch].resize(config_.numFrequencyBands, 0.0f);
    }

    bool success = false;

    try {
        if (config_.processingMode == ProcessingMode::GPU && gpuInitialized_) {
            success = processAudioGPU(audioData, numSamples, outputSpectrum);
        } else {
            success = processAudioCPU(audioData, numSamples, outputSpectrum);
        }

    } catch (const std::exception& e) {
        Logger::error("Exception during spectrum analysis: {}", e.what());
        success = false;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto processingTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

    updatePerformanceStats(static_cast<double>(processingTime), config_.processingMode == ProcessingMode::GPU && gpuInitialized_);

    return success;
}

bool SpectrumAnalyzer::processAudioCPU(const float* audioData, size_t numSamples,
                                        std::vector<std::vector<float>>& outputSpectrum) {
    // Add audio to overlap buffer
    size_t samplesToProcess = 0;

    if (config_.enableOverlap) {
        // Copy new audio to overlap buffer
        size_t samplesAvailable = overlapBuffer_.size() / config_.channels;
        size_t newSamples = std::min(numSamples, overlapBuffer_.size() - samplesAvailable);

        for (size_t i = 0; i < newSamples; ++i) {
            for (int ch = 0; ch < config_.channels; ++ch) {
                overlapBuffer_[(samplesAvailable + i) * config_.channels + ch] =
                    audioData[i * config_.channels + ch];
            }
        }

        samplesToProcess = (samplesAvailable + newSamples) / overlapSize_;

        // Process overlapped windows
        for (size_t window = 0; window < samplesToProcess; ++window) {
            size_t windowStart = window * overlapSize_;

            // Copy window data to input buffer for each channel
            for (int ch = 0; ch < config_.channels; ++ch) {
                for (size_t i = 0; i < config_.fftSize; ++i) {
                    inputBuffer_[i] = overlapBuffer_[(windowStart + i) * config_.channels + ch];
                }

                // Apply window if enabled
                if (config_.enableWindowing) {
                    applyWindow(inputBuffer_.data(), config_.fftSize);
                }

                // Compute FFT
                computeFFT(inputBuffer_.data(), fftBuffer_.data(), config_.fftSize);

                // Compute magnitude spectrum
                computeMagnitude(fftBuffer_.data(), magnitudeBuffer_.data(), config_.fftSize / 2 + 1);

                // Scale amplitude
                scaleAmplitude(magnitudeBuffer_.data(), config_.fftSize / 2 + 1);

                // Map to frequency bands
                mapFrequencyBands(magnitudeBuffer_.data(), outputSpectrum[ch].data(),
                                 config_.fftSize / 2 + 1, config_.numFrequencyBands);
            }
        }

        // Shift overlap buffer for next processing
        size_t remainingSamples = (samplesAvailable + newSamples) - samplesToProcess * overlapSize_;
        for (size_t i = 0; i < remainingSamples; ++i) {
            for (int ch = 0; ch < config_.channels; ++ch) {
                overlapBuffer_[i * config_.channels + ch] =
                    overlapBuffer_[(samplesToProcess * overlapSize_ + i) * config_.channels + ch];
            }
        }
    } else {
        // Non-overlapped processing
        size_t numWindows = numSamples / config_.fftSize;
        for (size_t window = 0; window < numWindows; ++window) {
            for (int ch = 0; ch < config_.channels; ++ch) {
                // Deinterleave channel data
                for (size_t i = 0; i < config_.fftSize; ++i) {
                    inputBuffer_[i] = audioData[(window * config_.fftSize + i) * config_.channels + ch];
                }

                // Apply window if enabled
                if (config_.enableWindowing) {
                    applyWindow(inputBuffer_.data(), config_.fftSize);
                }

                // Compute FFT
                computeFFT(inputBuffer_.data(), fftBuffer_.data(), config_.fftSize);

                // Compute magnitude spectrum
                computeMagnitude(fftBuffer_.data(), magnitudeBuffer_.data(), config_.fftSize / 2 + 1);

                // Scale amplitude
                scaleAmplitude(magnitudeBuffer_.data(), config_.fftSize / 2 + 1);

                // Map to frequency bands
                mapFrequencyBands(magnitudeBuffer_.data(), outputSpectrum[ch].data(),
                                 config_.fftSize / 2 + 1, config_.numFrequencyBands);
            }
        }
    }

    return true;
}

bool SpectrumAnalyzer::processAudioGPU(const float* audioData, size_t numSamples,
                                        std::vector<std::vector<float>>& outputSpectrum) {
#ifdef VORTEX_ENABLE_CUDA
    if (!gpuInitialized_) {
        return processAudioCPU(audioData, numSamples, outputSpectrum);
    }

    // GPU implementation would go here
    // For now, fallback to CPU
    return processAudioCPU(audioData, numSamples, outputSpectrum);
#else
    return processAudioCPU(audioData, numSamples, outputSpectrum);
#endif
}

bool SpectrumAnalyzer::initializeWindow() {
    window_.resize(config_.fftSize);

    switch (config_.windowType) {
        case WindowType::Rectangular:
            generateRectangularWindow(window_.data(), config_.fftSize);
            break;
        case WindowType::Hanning:
            generateHanningWindow(window_.data(), config_.fftSize);
            break;
        case WindowType::Hamming:
            generateHammingWindow(window_.data(), config_.fftSize);
            break;
        case WindowType::Blackman:
            generateBlackmanWindow(window_.data(), config_.fftSize);
            break;
        case WindowType::BlackmanHarris:
            generateBlackmanHarrisWindow(window_.data(), config_.fftSize);
            break;
        case WindowType::Kaiser:
            generateKaiserWindow(window_.data(), config_.fftSize, config_.kaiserBeta);
            break;
        case WindowType::FlatTop:
            generateFlatTopWindow(window_.data(), config_.fftSize);
            break;
    }

    return true;
}

bool SpectrumAnalyzer::initializeOverlapBuffer() {
    if (!config_.enableOverlap) {
        return true;
    }

    // Buffer should hold at least one complete FFT window plus overlap
    size_t bufferSamples = config_.fftSize + overlapSize_;
    overlapBuffer_.resize(bufferSamples * config_.channels);
    std::fill(overlapBuffer_.begin(), overlapBuffer_.end(), 0.0f);

    return true;
}

bool SpectrumAnalyzer::initializeFrequencyMapping() {
    frequencyMapping_.clear();
    frequencyBandCenters_.clear();

    switch (config_.frequencyScale) {
        case FrequencyScale::Linear:
            // No mapping needed for linear scale
            break;

        case FrequencyScale::Logarithmic:
            calculateLogarithmicBands();
            break;

        case FrequencyScale::Mel:
            calculateMelBands();
            break;

        case FrequencyScale::Octave:
            calculateOctaveBands();
            break;
    }

    frequencyMappingDirty_ = false;
    return true;
}

bool SpectrumAnalyzer::initializeGPUResources() {
#ifdef VORTEX_ENABLE_CUDA
    try {
        // Allocate GPU memory
        cudaError_t result;

        // FFT plan
        cufftHandle plan;
        result = cufftPlan1d(&plan, static_cast<int>(config_.fftSize), CUFFT_R2C, 1);
        if (result != CUDA_SUCCESS) {
            Logger::error("Failed to create CUDA FFT plan: {}", result);
            return false;
        }

        gpuFFTPlan_ = plan;

        // Input buffer
        result = cudaMalloc(&gpuInputBuffer_, config_.fftSize * sizeof(float));
        if (result != CUDA_SUCCESS) {
            Logger::error("Failed to allocate GPU input buffer: {}", result);
            cleanupGPUResources();
            return false;
        }

        // Output buffer (complex)
        result = cudaMalloc(&gpuOutputBuffer_, (config_.fftSize / 2 + 1) * sizeof(cufftComplex));
        if (result != CUDA_SUCCESS) {
            Logger::error("Failed to allocate GPU output buffer: {}", result);
            cleanupGPUResources();
            return false;
        }

        // Window buffer
        result = cudaMalloc(&gpuWindowBuffer_, config_.fftSize * sizeof(float));
        if (result != CUDA_SUCCESS) {
            Logger::error("Failed to allocate GPU window buffer: {}", result);
            cleanupGPUResources();
            return false;
        }

        // Copy window to GPU
        result = cudaMemcpy(gpuWindowBuffer_, window_.data(),
                          config_.fftSize * sizeof(float), cudaMemcpyHostToDevice);
        if (result != CUDA_SUCCESS) {
            Logger::error("Failed to copy window to GPU: {}", result);
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

void SpectrumAnalyzer::cleanupGPUResources() {
#ifdef VORTEX_ENABLE_CUDA
    if (gpuFFTPlan_) {
        cufftDestroy(static_cast<cufftHandle>(gpuFFTPlan_));
        gpuFFTPlan_ = nullptr;
    }

    if (gpuInputBuffer_) {
        cudaFree(gpuInputBuffer_);
        gpuInputBuffer_ = nullptr;
    }

    if (gpuOutputBuffer_) {
        cudaFree(gpuOutputBuffer_);
        gpuOutputBuffer_ = nullptr;
    }

    if (gpuWindowBuffer_) {
        cudaFree(gpuWindowBuffer_);
        gpuWindowBuffer_ = nullptr;
    }
#endif

    gpuInitialized_ = false;
}

void SpectrumAnalyzer::applyWindow(float* data, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        data[i] *= window_[i];
    }
}

void SpectrumAnalyzer::computeFFT(const float* input, std::complex<float>* output, size_t length) {
    computeFFTCPU(input, output, length);
}

void SpectrumAnalyzer::computeFFTCPU(const float* input, std::complex<float>* output, size_t length) {
    // Simplified FFT implementation - in production would use optimized library like FFTW
    // For now, use a basic implementation for demonstration

    std::vector<std::complex<float>> temp(length);
    for (size_t i = 0; i < length; ++i) {
        temp[i] = std::complex<float>(input[i], 0.0f);
    }

    // Basic DFT implementation (slow but functional)
    for (size_t k = 0; k < length / 2 + 1; ++k) {
        std::complex<float> sum(0.0f, 0.0f);
        for (size_t n = 0; n < length; ++n) {
            float angle = -2.0f * M_PI * k * n / length;
            sum += temp[n] * std::complex<float>(std::cos(angle), std::sin(angle));
        }
        output[k] = sum;
    }
}

void SpectrumAnalyzer::computeMagnitude(const std::complex<float>* fftData, float* magnitude, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        magnitude[i] = std::abs(fftData[i]);
    }
}

void SpectrumAnalyzer::scaleAmplitude(float* data, size_t length) {
    switch (config_.amplitudeScale) {
        case AmplitudeScale::Linear:
            if (config_.normalizeOutput) {
                // Normalize to 0.0 - 1.0 range
                float maxValue = *std::max_element(data, data + length);
                if (maxValue > 0.0f) {
                    for (size_t i = 0; i < length; ++i) {
                        data[i] /= maxValue;
                    }
                }
            }
            break;

        case AmplitudeScale::Decibel:
            for (size_t i = 0; i < length; ++i) {
                // Convert to dB, avoid log(0)
                data[i] = data[i] > 1e-10f ? 20.0f * std::log10(data[i]) : -100.0f;
            }
            break;

        case AmplitudeScale::SquareRoot:
            for (size_t i = 0; i < length; ++i) {
                data[i] = std::sqrt(data[i]);
            }
            break;
    }
}

void SpectrumAnalyzer::mapFrequencyBands(const float* input, float* output,
                                          size_t inputLength, size_t outputLength) {
    if (config_.frequencyScale == FrequencyScale::Linear) {
        // Simple downsampling for linear scale
        if (inputLength <= outputLength) {
            std::copy(input, input + inputLength, output);
            // Pad with zeros if needed
            std::fill(output + inputLength, output + outputLength, 0.0f);
        } else {
            // Downsample
            float ratio = static_cast<float>(inputLength) / outputLength;
            for (size_t i = 0; i < outputLength; ++i) {
                size_t inputIndex = static_cast<size_t>(i * ratio);
                output[i] = (inputIndex < inputLength) ? input[inputIndex] : 0.0f;
            }
        }
    } else {
        // Use pre-computed frequency mapping
        if (frequencyMapping_.size() == outputLength) {
            for (size_t i = 0; i < outputLength; ++i) {
                size_t startBin = static_cast<size_t>(frequencyMapping_[i]);
                size_t endBin = (i < outputLength - 1) ?
                               static_cast<size_t>(frequencyMapping_[i + 1]) : inputLength;

                if (startBin < inputLength) {
                    // Average energy in the band
                    float sum = 0.0f;
                    size_t count = 0;
                    for (size_t j = startBin; j < std::min(endBin, inputLength); ++j) {
                        sum += input[j] * input[j]; // Energy
                        ++count;
                    }
                    output[i] = count > 0 ? std::sqrt(sum / count) : 0.0f;
                } else {
                    output[i] = 0.0f;
                }
            }
        } else {
            // Fallback: simple copy and pad
            std::copy(input, input + std::min(inputLength, outputLength), output);
            std::fill(output + std::min(inputLength, outputLength), output + outputLength, 0.0f);
        }
    }
}

void SpectrumAnalyzer::calculateLogarithmicBands() {
    // Calculate logarithmic frequency bands
    double minFreq = 20.0; // Hz
    double maxFreq = config_.sampleRate / 2.0; // Nyquist frequency
    double freqResolution = config_.sampleRate / config_.fftSize;

    frequencyMapping_.resize(config_.numFrequencyBands);
    frequencyBandCenters_.resize(config_.numFrequencyBands);

    // Logarithmic spacing
    double logMin = std::log10(minFreq);
    double logMax = std::log10(maxFreq);
    double logStep = (logMax - logMin) / config_.numFrequencyBands;

    for (size_t i = 0; i < config_.numFrequencyBands; ++i) {
        double centerFreq = std::pow(10.0, logMin + i * logStep);
        frequencyBandCenters_[i] = centerFreq;
        frequencyMapping_[i] = static_cast<float>(centerFreq / freqResolution);
    }
}

void SpectrumAnalyzer::calculateMelBands() {
    // Calculate Mel frequency bands
    double minMel = 2595.0 * std::log10(1.0 + 20.0 / 700.0);
    double maxMel = 2595.0 * std::log10(1.0 + (config_.sampleRate / 2.0) / 700.0);
    double melStep = (maxMel - minMel) / config_.numFrequencyBands;

    frequencyMapping_.resize(config_.numFrequencyBands);
    frequencyBandCenters_.resize(config_.numFrequencyBands);

    double freqResolution = config_.sampleRate / config_.fftSize;

    for (size_t i = 0; i < config_.numFrequencyBands; ++i) {
        double mel = minMel + i * melStep;
        double freq = 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
        frequencyBandCenters_[i] = freq;
        frequencyMapping_[i] = static_cast<float>(freq / freqResolution);
    }
}

void SpectrumAnalyzer::calculateOctaveBands() {
    // Calculate octave frequency bands
    std::vector<double> octaveFreqs;
    double currentFreq = 31.25; // Start at ~31 Hz

    while (currentFreq <= config_.sampleRate / 2.0) {
        octaveFreqs.push_back(currentFreq);
        currentFreq *= 2.0; // Next octave
    }

    frequencyMapping_.resize(config_.numFrequencyBands);
    frequencyBandCenters_.resize(config_.numFrequencyBands);

    double freqResolution = config_.sampleRate / config_.fftSize;

    // Distribute bands logarithmically
    double logStart = std::log2(octaveFreqs.front());
    double logEnd = std::log2(octaveFreqs.back());
    double logStep = (logEnd - logStart) / config_.numFrequencyBands;

    for (size_t i = 0; i < config_.numFrequencyBands; ++i) {
        double freq = std::pow(2.0, logStart + i * logStep);
        frequencyBandCenters_[i] = freq;
        frequencyMapping_[i] = static_cast<float>(freq / freqResolution);
    }
}

// Window function implementations
void SpectrumAnalyzer::generateRectangularWindow(float* window, size_t length) {
    std::fill(window, window + length, 1.0f);
}

void SpectrumAnalyzer::generateHanningWindow(float* window, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (length - 1)));
    }
}

void SpectrumAnalyzer::generateHammingWindow(float* window, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        window[i] = 0.54f - 0.46f * std::cos(2.0f * M_PI * i / (length - 1));
    }
}

void SpectrumAnalyzer::generateBlackmanWindow(float* window, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        window[i] = 0.42f - 0.5f * std::cos(2.0f * M_PI * i / (length - 1)) +
                   0.08f * std::cos(4.0f * M_PI * i / (length - 1));
    }
}

void SpectrumAnalyzer::generateBlackmanHarrisWindow(float* window, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        float n = static_cast<float>(i);
        float N = static_cast<float>(length - 1);
        window[i] = 0.35875f - 0.48829f * std::cos(2.0f * M_PI * n / N) +
                   0.14128f * std::cos(4.0f * M_PI * n / N) -
                   0.01168f * std::cos(6.0f * M_PI * n / N);
    }
}

void SpectrumAnalyzer::generateKaiserWindow(float* window, size_t length, float beta) {
    float i0_beta = kaiserBessel(beta);
    for (size_t i = 0; i < length; ++i) {
        float n = static_cast<float>(i);
        float N = static_cast<float>(length - 1);
        float alpha = (2.0f * n / N) - 1.0f;
        window[i] = kaiserBessel(beta * std::sqrt(1.0f - alpha * alpha)) / i0_beta;
    }
}

void SpectrumAnalyzer::generateFlatTopWindow(float* window, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        float n = static_cast<float>(i);
        float N = static_cast<float>(length - 1);
        window[i] = 1.0f - 1.93f * std::cos(2.0f * M_PI * n / N) +
                   1.29f * std::cos(4.0f * M_PI * n / N) -
                   0.388f * std::cos(6.0f * M_PI * n / N) +
                   0.032f * std::cos(8.0f * M_PI * n / N);
    }
}

float SpectrumAnalyzer::kaiserBessel(float x) {
    // Simplified Bessel function implementation
    if (std::abs(x) < 1e-10f) {
        return 1.0f;
    }

    float sum = 1.0f;
    float term = 1.0f;
    float k = 1.0f;

    for (int i = 1; i < 20; ++i) {
        term *= (x * x) / (4.0f * k * k);
        sum += term;
        k += 1.0f;

        if (term < 1e-10f) {
            break;
        }
    }

    return sum;
}

// Setter and getter methods
void SpectrumAnalyzer::setWindowType(WindowType windowType) {
    config_.windowType = windowType;
    if (initialized_) {
        initializeWindow();
    }
}

SpectrumAnalyzer::WindowType SpectrumAnalyzer::getWindowType() const {
    return config_.windowType;
}

void SpectrumAnalyzer::setFrequencyScale(FrequencyScale frequencyScale) {
    config_.frequencyScale = frequencyScale;
    frequencyMappingDirty_ = true;
}

SpectrumAnalyzer::FrequencyScale SpectrumAnalyzer::getFrequencyScale() const {
    return config_.frequencyScale;
}

void SpectrumAnalyzer::setAmplitudeScale(AmplitudeScale amplitudeScale) {
    config_.amplitudeScale = amplitudeScale;
}

SpectrumAnalyzer::AmplitudeScale SpectrumAnalyzer::getAmplitudeScale() const {
    return config_.amplitudeScale;
}

void SpectrumAnalyzer::setProcessingMode(ProcessingMode mode) {
    config_.processingMode = mode;
}

SpectrumAnalyzer::ProcessingMode SpectrumAnalyzer::getProcessingMode() const {
    return config_.processingMode;
}

bool SpectrumAnalyzer::isGPUAvailable() const {
    return gpuInitialized_;
}

const SpectrumAnalyzer::Config& SpectrumAnalyzer::getConfig() const {
    return config_;
}

double SpectrumAnalyzer::getFrequencyResolution() const {
    return static_cast<double>(config_.sampleRate) / config_.fftSize;
}

std::vector<double> SpectrumAnalyzer::getFrequencyBandCenters() const {
    return frequencyBandCenters_;
}

std::string SpectrumAnalyzer::getPerformanceStats() const {
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

void SpectrumAnalyzer::reset() {
    if (!initialized_) {
        return;
    }

    std::fill(overlapBuffer_.begin(), overlapBuffer_.end(), 0.0f);
    writePos_ = 0;
    readPos_ = 0;

    // Reset performance counters
    totalFramesProcessed_ = 0;
    totalProcessingTime_ = 0.0;
    gpuFramesProcessed_ = 0;
    cpuFramesProcessed_ = 0;

    Logger::debug("SpectrumAnalyzer reset");
}

bool SpectrumAnalyzer::isInitialized() const {
    return initialized_;
}

void SpectrumAnalyzer::setOverlapRatio(float ratio) {
    config_.hopSize = static_cast<size_t>(config_.fftSize * (1.0f - ratio));
    if (initialized_) {
        initializeOverlapBuffer();
    }
}

float SpectrumAnalyzer::getOverlapRatio() const {
    return 1.0f - (static_cast<float>(config_.hopSize) / config_.fftSize);
}

void SpectrumAnalyzer::setWindowingEnabled(bool enabled) {
    config_.enableWindowing = enabled;
}

bool SpectrumAnalyzer::isWindowingEnabled() const {
    return config_.enableWindowing;
}

void SpectrumAnalyzer::updatePerformanceStats(double processingTimeMs, bool usedGPU) const {
    totalFramesProcessed_++;
    totalProcessingTime_ += processingTimeMs;

    if (usedGPU) {
        gpuFramesProcessed_++;
    } else {
        cpuFramesProcessed_++;
    }
}

bool SpectrumAnalyzer::isPowerOfTwo(size_t n) const {
    return (n > 0) && ((n & (n - 1)) == 0);
}

size_t SpectrumAnalyzer::nextPowerOfTwo(size_t n) const {
    if (isPowerOfTwo(n)) {
        return n;
    }

    size_t power = 1;
    while (power < n) {
        power <<= 1;
    }
    return power;
}

} // namespace vortex::core::dsp