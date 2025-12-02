#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <complex>
#include <random>
#include <chrono>
#include <fstream>

#include "../../src/core/dsp/convolution.hpp"
#include "../../src/core/audio_engine.hpp"

using namespace vortex;
using namespace testing;

class ConvolutionTest : public ::testing::Test {
protected:
    void SetUp() override {
        convolution_ = std::make_unique<ConvolutionEngine>();

        // Initialize with standard audio parameters
        ConvolutionEngine::Config config;
        config.sampleRate = 48000;
        config.numChannels = 2;
        config.blockSize = 512;
        config.maxImpulseLength = 65536; // 64K samples for testing
        config.enableGPUAcceleration = false; // Use CPU for testing
        config.fftMethod = ConvolutionEngine::FFTMethod::AUTO;

        ASSERT_TRUE(convolution_->initialize(config));

        // Generate test signals
        generateTestSignal();
        generateTestImpulseResponse();
    }

    void TearDown() override {
        convolution_.reset();
    }

    void generateTestSignal() {
        testSignal_.resize(blockSize_);
        referenceSignal_.resize(blockSize_);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

        // Generate white noise
        for (size_t i = 0; i < blockSize_; ++i) {
            testSignal_[i] = dis(gen);
        }

        // Copy reference
        referenceSignal_ = testSignal_;
    }

    void generateTestImpulseResponse() {
        // Generate a simple test impulse response
        impulseResponse_.resize(1024);

        // Create a simple reverb-like impulse response
        for (size_t i = 0; i < impulseResponse_.size(); ++i) {
            float decay = std::exp(-static_cast<float>(i) / 1000.0f);
            float noise = (std::rand() % 1000 - 500) / 10000.0f;
            impulseResponse_[i] = decay * noise;
        }

        // Add some direct signal at the beginning
        if (impulseResponse_.size() > 0) {
            impulseResponse_[0] = 1.0f; // Direct impulse
        }

        // Normalize impulse response
        float maxMagnitude = 0.001f;
        for (float sample : impulseResponse_) {
            maxMagnitude = std::max(maxMagnitude, std::abs(sample));
        }

        for (float& sample : impulseResponse_) {
            sample /= maxMagnitude;
        }
    }

    void generateSineWaveImpulse(float frequency, float amplitude = 0.5f) {
        sineImpulse_.resize(2048);

        for (size_t i = 0; i < sineImpulse_.size(); ++i) {
            float time = static_cast<float>(i) / sampleRate_;
            float decay = std::exp(-static_cast<float>(i) / 2000.0f);
            sineImpulse_[i] = amplitude * decay * std::sin(2.0f * M_PI * frequency * time);
        }
    }

    bool compareSignals(const std::vector<float>& signal1, const std::vector<float>& signal2, float tolerance = 1e-3f) {
        if (signal1.size() != signal2.size()) {
            return false;
        }

        for (size_t i = 0; i < signal1.size(); ++i) {
            if (std::abs(signal1[i] - signal2[i]) > tolerance) {
                return false;
            }
        }

        return true;
    }

    float calculateSignalEnergy(const std::vector<float>& signal) {
        float energy = 0.0f;
        for (float sample : signal) {
            energy += sample * sample;
        }
        return energy;
    }

    std::unique_ptr<ConvolutionEngine> convolution_;
    std::vector<float> testSignal_;
    std::vector<float> referenceSignal_;
    std::vector<float> outputSignal_;
    std::vector<float> impulseResponse_;
    std::vector<float> sineImpulse_;

    static constexpr uint32_t sampleRate_ = 48000;
    static constexpr size_t blockSize_ = 512;
};

// Initialization tests
TEST_F(ConvolutionTest, InitializeWithValidConfig) {
    ConvolutionEngine::Config config;
    config.sampleRate = 44100;
    config.numChannels = 2;
    config.blockSize = 256;
    config.maxImpulseLength = 32768;
    config.enableGPUAcceleration = false;

    ConvolutionEngine conv;
    EXPECT_TRUE(conv.initialize(config));
    EXPECT_TRUE(conv.isInitialized());
}

TEST_F(ConvolutionTest, InitializeWithInvalidConfig) {
    ConvolutionEngine::Config config;
    config.sampleRate = 0;  // Invalid
    config.numChannels = 2;
    config.blockSize = 256;

    ConvolutionEngine conv;
    EXPECT_FALSE(conv.initialize(config));
    EXPECT_FALSE(conv.isInitialized());
}

TEST_F(ConvolutionTest, GetConfiguration) {
    auto config = convolution_->getConfiguration();
    EXPECT_EQ(config.sampleRate, sampleRate_);
    EXPECT_EQ(config.numChannels, 2);
    EXPECT_EQ(config.blockSize, blockSize_);
}

// Impulse response loading tests
TEST_F(ConvolutionTest, LoadImpulseResponseFromMemory) {
    EXPECT_TRUE(convolution_->loadImpulseResponse(impulseResponse_.data(), impulseResponse_.size()));
    EXPECT_TRUE(convolution_->hasImpulseResponse());
}

TEST_F(ConvolutionTest, LoadImpulseResponseFromFile) {
    // Create a temporary impulse response file
    std::string tempFile = "test_impulse.wav";

    // Simple WAV header (44 bytes) + PCM data
    std::vector<uint8_t> wavFile;
    wavFile.resize(44 + impulseResponse_.size() * sizeof(float));

    // Write WAV header
    // RIFF header
    wavFile[0] = 'R'; wavFile[1] = 'I'; wavFile[2] = 'F'; wavFile[3] = 'F';
    uint32_t fileSize = 36 + impulseResponse_.size() * sizeof(float);
    std::memcpy(&wavFile[4], &fileSize, 4);
    wavFile[8] = 'W'; wavFile[9] = 'A'; wavFile[10] = 'V'; wavFile[11] = 'E';

    // fmt chunk
    wavFile[12] = 'f'; wavFile[13] = 'm'; wavFile[14] = 't'; wavFile[15] = ' ';
    uint32_t fmtChunkSize = 16;
    std::memcpy(&wavFile[16], &fmtChunkSize, 4);
    uint16_t audioFormat = 3; // IEEE float
    std::memcpy(&wavFile[20], &audioFormat, 2);
    uint16_t numChannels = 1;
    std::memcpy(&wavFile[22], &numChannels, 2);
    uint32_t sampleRate = 48000;
    std::memcpy(&wavFile[24], &sampleRate, 4);
    uint32_t byteRate = sampleRate * sizeof(float);
    std::memcpy(&wavFile[28], &byteRate, 4);
    uint16_t blockAlign = sizeof(float);
    std::memcpy(&wavFile[32], &blockAlign, 2);
    uint16_t bitsPerSample = 32;
    std::memcpy(&wavFile[34], &bitsPerSample, 2);

    // data chunk
    wavFile[36] = 'd'; wavFile[37] = 'a'; wavFile[38] = 't'; wavFile[39] = 'a';
    uint32_t dataSize = impulseResponse_.size() * sizeof(float);
    std::memcpy(&wavFile[40], &dataSize, 4);

    // Write PCM data
    std::memcpy(&wavFile[44], impulseResponse_.data(), dataSize);

    // Write file
    std::ofstream file(tempFile, std::ios::binary);
    file.write(reinterpret_cast<const char*>(wavFile.data()), wavFile.size());
    file.close();

    EXPECT_TRUE(convolution_->loadImpulseResponseFromFile(tempFile));
    EXPECT_TRUE(convolution_->hasImpulseResponse());

    // Clean up
    std::remove(tempFile.c_str());
}

TEST_F(ConvolutionTest, LoadInvalidImpulseResponse) {
    // Test with null pointer
    EXPECT_FALSE(convolution_->loadImpulseResponse(nullptr, 1024));

    // Test with zero length
    EXPECT_FALSE(convolution_->loadImpulseResponse(impulseResponse_.data(), 0));

    // Test with excessive length
    std::vector<float> hugeIR(10000000); // 10M samples
    EXPECT_FALSE(convolution_->loadImpulseResponse(hugeIR.data(), hugeIR.size()));
}

TEST_F(ConvolutionTest, LoadImpulseResponseMultiChannel) {
    std::vector<float> stereoIR(impulseResponse_.size() * 2);

    // Copy mono IR to both channels
    for (size_t i = 0; i < impulseResponse_.size(); ++i) {
        stereoIR[i * 2] = impulseResponse_[i];     // Left channel
        stereoIR[i * 2 + 1] = impulseResponse_[i]; // Right channel
    }

    EXPECT_TRUE(convolution_->loadImpulseResponse(stereoIR.data(), impulseResponse_.size(), 2));
}

// Processing tests
TEST_F(ConvolutionTest, ProcessAudioWithoutIR) {
    outputSignal_.resize(blockSize_);

    // Should fail without impulse response loaded
    EXPECT_FALSE(convolution_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2));
}

TEST_F(ConvolutionTest, ProcessAudioWithIR) {
    outputSignal_.resize(blockSize_);

    convolution_->loadImpulseResponse(impulseResponse_.data(), impulseResponse_.size());

    EXPECT_TRUE(convolution_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2));

    // Output should be different from input
    EXPECT_FALSE(compareSignals(testSignal_, outputSignal_, 1e-6f));

    // Energy should change (convolution effect)
    float inputEnergy = calculateSignalEnergy(testSignal_);
    float outputEnergy = calculateSignalEnergy(outputSignal_);
    EXPECT_NE(inputEnergy, outputEnergy);
}

TEST_F(ConvolutionTest, ProcessWithIdentityIR) {
    outputSignal_.resize(blockSize_);

    // Create identity impulse response
    std::vector<float> identityIR(1, 1.0f);
    convolution_->loadImpulseResponse(identityIR.data(), identityIR.size());

    EXPECT_TRUE(convolution_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2));

    // With identity IR, output should match input (after initial delay)
    EXPECT_TRUE(compareSignals(testSignal_, outputSignal_, 1e-4f));
}

TEST_F(ConvolutionTest, ProcessMultiChannel) {
    outputSignal_.resize(blockSize_);

    // Load stereo impulse response
    std::vector<float> stereoIR(impulseResponse_.size() * 2);
    for (size_t i = 0; i < impulseResponse_.size(); ++i) {
        stereoIR[i * 2] = impulseResponse_[i];     // Left channel
        stereoIR[i * 2 + 1] = impulseResponse_[i] * 0.8f; // Right channel (slightly different)
    }

    convolution_->loadImpulseResponse(stereoIR.data(), impulseResponse_.size(), 2);

    std::vector<const float*> inputs(2);
    std::vector<float*> outputs(2);
    std::vector<float> inputBuffer[2];
    std::vector<float> outputBuffer[2];

    for (int ch = 0; ch < 2; ++ch) {
        inputBuffer[ch].resize(blockSize_);
        outputBuffer[ch].resize(blockSize_);
        inputs[ch] = inputBuffer[ch].data();
        outputs[ch] = outputBuffer[ch].data();

        // Generate test signal for each channel
        for (size_t i = 0; i < blockSize_; ++i) {
            inputBuffer[ch][i] = testSignal_[i] * (ch + 1) * 0.5f;
        }
    }

    EXPECT_TRUE(convolution_->processAudioMultiChannel(inputs, outputs, blockSize_, 2));
}

// Bypass tests
TEST_F(ConvolutionTest, BypassMode) {
    outputSignal_.resize(blockSize_);

    convolution_->loadImpulseResponse(impulseResponse_.data(), impulseResponse_.size());
    convolution_->setBypass(true);

    EXPECT_TRUE(convolution_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2));

    // In bypass mode, output should match input
    EXPECT_TRUE(compareSignals(testSignal_, outputSignal_, 1e-6f));

    convolution_->setBypass(false);
    EXPECT_FALSE(convolution_->isBypassed());

    convolution_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2);

    // Should be different when not bypassed
    EXPECT_FALSE(compareSignals(testSignal_, outputSignal_, 1e-6f));
}

// Wet/Dry mix tests
TEST_F(ConvolutionTest, WetDryMix) {
    outputSignal_.resize(blockSize_);

    convolution_->loadImpulseResponse(impulseResponse_.data(), impulseResponse_.size());

    // Test 50% wet / 50% dry
    convolution_->setWetDryMix(0.5f);
    EXPECT_FLOAT_EQ(convolution_->getWetDryMix(), 0.5f);

    EXPECT_TRUE(convolution_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2));

    // Output should be partially processed
    // (This is a basic test - real verification would require more precise analysis)
}

TEST_F(ConvolutionTest, WetDryMixEdgeCases) {
    // Test invalid mix values
    convolution_->setWetDryMix(-0.5f); // Should clamp to 0.0f
    EXPECT_FLOAT_EQ(convolution_->getWetDryMix(), 0.0f);

    convolution_->setWetDryMix(1.5f); // Should clamp to 1.0f
    EXPECT_FLOAT_EQ(convolution_->getWetDryMix(), 1.0f);

    convolution_->setWetDryMix(0.0f); // Fully dry
    EXPECT_FLOAT_EQ(convolution_->getWetDryMix(), 0.0f);

    convolution_->setWetDryMix(1.0f); // Fully wet
    EXPECT_FLOAT_EQ(convolution_->getWetDryMix(), 1.0f);
}

// Latency tests
TEST_F(ConvolutionTest, GetLatency) {
    // Test without impulse response
    EXPECT_EQ(convolution_->getLatency(), 0);

    // Load impulse response and check latency
    convolution_->loadImpulseResponse(impulseResponse_.data(), impulseResponse_.size());
    uint32_t latency = convolution_->getLatency();

    // Latency should be reasonable (depends on implementation)
    EXPECT_LT(latency, blockSize_ * 2); // Should be less than 2 blocks
}

TEST_F(ConvolutionTest, ProcessWithLatency) {
    outputSignal_.resize(blockSize_);

    convolution_->loadImpulseResponse(impulseResponse_.data(), impulseResponse_.size());

    // Process multiple blocks to handle latency
    for (int i = 0; i < 10; ++i) {
        convolution_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2);
    }

    // Should have processed audio after several blocks
    float outputEnergy = calculateSignalEnergy(outputSignal_);
    EXPECT_GT(outputEnergy, 0.0f);
}

// FFT method tests
TEST_F(ConvolutionTest, SetFFTMethod) {
    EXPECT_TRUE(convolution_->setFFTMethod(ConvolutionEngine::FFTMethod::AUTO));
    EXPECT_EQ(convolution_->getFFTMethod(), ConvolutionEngine::FFTMethod::AUTO);

    EXPECT_TRUE(convolution_->setFFTMethod(ConvolutionEngine::FFTMethod::FFTW));
    EXPECT_EQ(convolution_->getFFTMethod(), ConvolutionEngine::FFTMethod::FFTW);

    EXPECT_TRUE(convolution_->setFFTMethod(ConvolutionEngine::FFTMethod::KISS));
    EXPECT_EQ(convolution_->getFFTMethod(), ConvolutionEngine::FFTMethod::KISS);

    EXPECT_TRUE(convolution_->setFFTMethod(ConvolutionEngine::FFTMethod::OOURA));
    EXPECT_EQ(convolution_->getFFTMethod(), ConvolutionEngine::FFTMethod::OOURA);
}

// Performance tests
TEST_F(ConvolutionTest, PerformanceBasicProcessing) {
    outputSignal_.resize(blockSize_);
    convolution_->loadImpulseResponse(impulseResponse_.data(), impulseResponse_.size());

    const int numIterations = 100;
    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numIterations; ++i) {
        convolution_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

    // Should process reasonably fast for real-time use
    EXPECT_LT(duration.count(), 50000); // Less than 50ms for 100 iterations
    EXPECT_LT(duration.count() / numIterations, 500); // Less than 500 microseconds per block
}

TEST_F(ConvolutionTest, PerformanceLargeImpulseResponse) {
    // Test with large impulse response
    std::vector<float> largeIR(32768); // 32K samples
    generateTestImpulseResponse();

    // Create reverb-like tail
    for (size_t i = 0; i < largeIR.size(); ++i) {
        float decay = std::exp(-static_cast<float>(i) / 8000.0f);
        float noise = (std::rand() % 1000 - 500) / 20000.0f;
        largeIR[i] = decay * noise;
    }

    convolution_->loadImpulseResponse(largeIR.data(), largeIR.size());

    outputSignal_.resize(blockSize_);

    const int numIterations = 10;
    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numIterations; ++i) {
        convolution_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

    // Even with large IR, should still be reasonably fast
    EXPECT_LT(duration.count(), 20000); // Less than 20ms for 10 iterations
}

// GPU acceleration tests
TEST_F(ConvolutionTest, GPUAcceleration) {
    // Test with GPU acceleration
    ConvolutionEngine::Config gpuConfig;
    gpuConfig.sampleRate = 48000;
    gpuConfig.numChannels = 2;
    gpuConfig.blockSize = 512;
    gpuConfig.maxImpulseLength = 65536;
    gpuConfig.enableGPUAcceleration = true;

    ConvolutionEngine gpuConv;
    if (gpuConv.initialize(gpuConfig)) {
        EXPECT_TRUE(gpuConv.isGPUAccelerated());

        gpuConv.loadImpulseResponse(impulseResponse_.data(), impulseResponse_.size());

        outputSignal_.resize(blockSize_);
        EXPECT_TRUE(gpuConv.processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2));
    } else {
        // GPU not available, test should pass anyway
        SUCCEED() << "GPU acceleration not available on this system";
    }
}

// Statistics tests
TEST_F(ConvolutionTest, GetStatistics) {
    auto stats = convolution_->getStatistics();

    EXPECT_EQ(stats.totalSamplesProcessed, 0);
    EXPECT_EQ(stats.totalProcessingTime, 0);
    EXPECT_EQ(stats.averageProcessingTime, 0);
    EXPECT_EQ(stats.maxProcessingTime, 0);
    EXPECT_EQ(stats.impulseLength, 0);
    EXPECT_FALSE(stats.gpuAccelerated);

    // Load impulse response and process
    convolution_->loadImpulseResponse(impulseResponse_.data(), impulseResponse_.size());
    outputSignal_.resize(blockSize_);
    convolution_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2);

    stats = convolution_->getStatistics();
    EXPECT_EQ(stats.totalSamplesProcessed, blockSize_);
    EXPECT_GT(stats.totalProcessingTime, 0);
    EXPECT_GT(stats.averageProcessingTime, 0);
    EXPECT_GT(stats.maxProcessingTime, 0);
    EXPECT_EQ(stats.impulseLength, impulseResponse_.size());
}

TEST_F(ConvolutionTest, ResetStatistics) {
    // Process some audio first
    convolution_->loadImpulseResponse(impulseResponse_.data(), impulseResponse_.size());
    outputSignal_.resize(blockSize_);
    convolution_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2);

    auto stats = convolution_->getStatistics();
    EXPECT_GT(stats.totalSamplesProcessed, 0);

    // Reset and verify
    convolution_->resetStatistics();
    stats = convolution_->getStatistics();
    EXPECT_EQ(stats.totalSamplesProcessed, 0);
    EXPECT_EQ(stats.totalProcessingTime, 0);
}

// Memory tests
TEST_F(ConvolutionTest, MemoryUsage) {
    size_t initialMemory = convolution_->getMemoryUsage();
    EXPECT_GT(initialMemory, 0);

    // Load impulse response
    convolution_->loadImpulseResponse(impulseResponse_.data(), impulseResponse_.size());

    size_t currentMemory = convolution_->getMemoryUsage();
    EXPECT_GT(currentMemory, initialMemory);

    // Memory should increase approximately by IR size + FFT buffers
    size_t expectedIncrease = impulseResponse_.size() * sizeof(float) * 2; // IR + FFT buffers
    EXPECT_GE(currentMemory - initialMemory, expectedIncrease * 0.8f); // Allow some overhead
}

// Edge cases and error handling
TEST_F(ConvolutionTest, ProcessWithNullPointers) {
    outputSignal_.resize(blockSize_);
    convolution_->loadImpulseResponse(impulseResponse_.data(), impulseResponse_.size());

    EXPECT_FALSE(convolution_->processAudio(nullptr, outputSignal_.data(), blockSize_, 2));
    EXPECT_FALSE(convolution_->processAudio(testSignal_.data(), nullptr, blockSize_, 2));
    EXPECT_FALSE(convolution_->processAudio(testSignal_.data(), outputSignal_.data(), 0, 2));
}

TEST_F(ConvolutionTest, ProcessWithInvalidBlockSize) {
    outputSignal_.resize(blockSize_);
    convolution_->loadImpulseResponse(impulseResponse_.data(), impulseResponse_.size());

    // Block size should match configuration
    EXPECT_FALSE(convolution_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_ + 1, 2));
    EXPECT_FALSE(convolution_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_ - 1, 2));
}

TEST_F(ConvolutionTest, ProcessWithInvalidChannels) {
    outputSignal_.resize(blockSize_);
    convolution_->loadImpulseResponse(impulseResponse_.data(), impulseResponse_.size());

    // Channel count should match configuration
    EXPECT_FALSE(convolution_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 1));
    EXPECT_FALSE(convolution_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 4));
}

// Real-time mode tests
TEST_F(ConvolutionTest, SetRealTimeMode) {
    convolution_->setRealTimeMode(true);
    EXPECT_TRUE(convolution_->isRealTimeMode());

    convolution_->setRealTimeMode(false);
    EXPECT_FALSE(convolution_->isRealTimeMode());
}

TEST_F(ConvolutionTest, RealTimeModeProcessing) {
    convolution_->setRealTimeMode(true);
    convolution_->loadImpulseResponse(impulseResponse_.data(), impulseResponse_.size());

    outputSignal_.resize(blockSize_);
    EXPECT_TRUE(convolution_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2));

    auto stats = convolution_->getStatistics();
    // Real-time mode should have reasonable processing time
    EXPECT_LT(stats.averageProcessingTime, 1000); // Less than 1 millisecond
}

// Thread safety tests (basic)
TEST_F(ConvolutionTest, BasicThreadSafety) {
    // This is a basic thread safety test
    // More comprehensive testing would require proper synchronization setup

    outputSignal_.resize(blockSize_);
    convolution_->loadImpulseResponse(impulseResponse_.data(), impulseResponse_.size());

    // Test concurrent access (basic smoke test)
    std::thread processThread([&]() {
        for (int i = 0; i < 50; ++i) {
            convolution_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    std::thread settingsThread([&]() {
        for (int i = 0; i < 50; ++i) {
            convolution_->setWetDryMix(0.3f + (i % 5) * 0.1f);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    processThread.join();
    settingsThread.join();

    // If we get here without crashing, basic thread safety is working
    SUCCEED();
}

// Impulse response validation tests
TEST_F(ConvolutionTest, ValidateImpulseResponse) {
    // Test with NaN values
    std::vector<float> nanIR(100);
    nanIR[50] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(convolution_->loadImpulseResponse(nanIR.data(), nanIR.size()));

    // Test with infinite values
    std::vector<float> infIR(100);
    infIR[50] = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(convolution_->loadImpulseResponse(infIR.data(), infIR.size()));

    // Test with normal values (should work)
    std::vector<float> normalIR(100, 0.1f);
    EXPECT_TRUE(convolution_->loadImpulseResponse(normalIR.data(), normalIR.size()));
}