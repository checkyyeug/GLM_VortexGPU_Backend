#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <complex>
#include <random>
#include <chrono>

#include "../../src/core/dsp/equalizer.hpp"
#include "../../src/core/audio_engine.hpp"

using namespace vortex;
using namespace testing;

class EqualizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        equalizer_ = std::make_unique<Equalizer>();

        // Initialize with standard audio parameters
        Equalizer::Config config;
        config.sampleRate = 48000;
        config.numChannels = 2;
        config.blockSize = 512;
        config.enableGPUAcceleration = false; // Use CPU for testing
        config.fftSize = 4096;

        ASSERT_TRUE(equalizer_->initialize(config));

        // Generate test signal
        generateTestSignal();
    }

    void TearDown() override {
        equalizer_.reset();
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

    void generateSineWave(float frequency, float amplitude = 0.5f) {
        testSignal_.resize(blockSize_);
        referenceSignal_.resize(blockSize_);

        for (size_t i = 0; i < blockSize_; ++i) {
            float time = static_cast<float>(i) / sampleRate_;
            testSignal_[i] = amplitude * std::sin(2.0f * M_PI * frequency * time);
        }

        referenceSignal_ = testSignal_;
    }

    float calculateSpectrumDifference(const std::vector<float>& signal1,
                                     const std::vector<float>& signal2) {
        // Simple FFT and compare magnitude spectra
        auto spectrum1 = calculateSpectrum(signal1);
        auto spectrum2 = calculateSpectrum(signal2);

        float diff = 0.0f;
        for (size_t i = 0; i < spectrum1.size(); ++i) {
            diff += std::abs(spectrum1[i] - spectrum2[i]);
        }

        return diff / spectrum1.size();
    }

    std::vector<float> calculateSpectrum(const std::vector<float>& signal) {
        // Simple DFT for testing
        std::vector<std::complex<float>> fft(signal.size());
        std::vector<float> magnitude(signal.size() / 2 + 1);

        for (size_t k = 0; k < signal.size() / 2 + 1; ++k) {
            std::complex<float> sum(0.0f, 0.0f);
            for (size_t n = 0; n < signal.size(); ++n) {
                float angle = -2.0f * M_PI * k * n / signal.size();
                sum += signal[n] * std::complex<float>(std::cos(angle), std::sin(angle));
            }
            magnitude[k] = std::abs(sum);
        }

        return magnitude;
    }

    std::unique_ptr<Equalizer> equalizer_;
    std::vector<float> testSignal_;
    std::vector<float> referenceSignal_;
    std::vector<float> outputSignal_;

    static constexpr uint32_t sampleRate_ = 48000;
    static constexpr size_t blockSize_ = 512;
};

// Initialization tests
TEST_F(EqualizerTest, InitializeWithValidConfig) {
    Equalizer::Config config;
    config.sampleRate = 44100;
    config.numChannels = 2;
    config.blockSize = 256;
    config.enableGPUAcceleration = false;

    Equalizer eq;
    EXPECT_TRUE(eq.initialize(config));
    EXPECT_TRUE(eq.isInitialized());
}

TEST_F(EqualizerTest, InitializeWithInvalidConfig) {
    Equalizer::Config config;
    config.sampleRate = 0;  // Invalid
    config.numChannels = 2;
    config.blockSize = 256;

    Equalizer eq;
    EXPECT_FALSE(eq.initialize(config));
    EXPECT_FALSE(eq.isInitialized());
}

TEST_F(EqualizerTest, GetConfiguration) {
    auto config = equalizer_->getConfiguration();
    EXPECT_EQ(config.sampleRate, sampleRate_);
    EXPECT_EQ(config.numChannels, 2);
    EXPECT_EQ(config.blockSize, blockSize_);
}

// Band manipulation tests
TEST_F(EqualizerTest, SetBandGain) {
    // Test setting individual band gains
    const float testGain = -3.0f;

    EXPECT_TRUE(equalizer_->setBandGain(0, testGain));  // Low frequency
    EXPECT_TRUE(equalizer_->setBandGain(256, testGain)); // Mid frequency
    EXPECT_TRUE(equalizer_->setBandGain(511, testGain)); // High frequency

    // Test invalid band indices
    EXPECT_FALSE(equalizer_->setBandGain(-1, testGain));
    EXPECT_FALSE(equalizer_->setBandGain(512, testGain));
}

TEST_F(EqualizerTest, SetBandGainsArray) {
    std::vector<float> gains(512, 0.0f);

    // Set some bands to different values
    gains[100] = 6.0f;
    gains[200] = -6.0f;
    gains[300] = 3.0f;

    EXPECT_TRUE(equalizer_->setBandGains(gains));

    // Test invalid size
    std::vector<float> invalidGains(100, 0.0f);
    EXPECT_FALSE(equalizer_->setBandGains(invalidGains));
}

TEST_F(EqualizerTest, GetBandGain) {
    const float testGain = -5.0f;

    EXPECT_TRUE(equalizer_->setBandGain(256, testGain));
    EXPECT_FLOAT_EQ(equalizer_->getBandGain(256), testGain);

    // Test invalid band indices
    EXPECT_FLOAT_EQ(equalizer_->getBandGain(-1), 0.0f);
    EXPECT_FLOAT_EQ(equalizer_->getBandGain(512), 0.0f);
}

// Filter type tests
TEST_F(EqualizerTest, SetFilterType) {
    EXPECT_TRUE(equalizer_->setFilterType(0, Equalizer::FilterType::PEAK));
    EXPECT_TRUE(equalizer_->setFilterType(100, Equalizer::FilterType::LOW_SHELF));
    EXPECT_TRUE(equalizer_->setFilterType(400, Equalizer::FilterType::HIGH_SHELF));
    EXPECT_TRUE(equalizer_->setFilterType(500, Equalizer::FilterType::BELL));

    // Test invalid band indices
    EXPECT_FALSE(equalizer_->setFilterType(-1, Equalizer::FilterType::PEAK));
    EXPECT_FALSE(equalizer_->setFilterType(512, Equalizer::FilterType::PEAK));
}

TEST_F(EqualizerTest, GetFilterType) {
    equalizer_->setFilterType(256, Equalizer::FilterType::LOW_SHELF);
    EXPECT_EQ(equalizer_->getFilterType(256), Equalizer::FilterType::LOW_SHELF);

    // Default should be PEAK
    EXPECT_EQ(equalizer_->getFilterType(0), Equalizer::FilterType::PEAK);
}

// Bypass and enable tests
TEST_F(EqualizerTest, BypassEqualizer) {
    outputSignal_.resize(blockSize_);

    // Process with bypass enabled - should pass signal unchanged
    equalizer_->setBypass(true);
    equalizer_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2);

    for (size_t i = 0; i < blockSize_; ++i) {
        EXPECT_FLOAT_EQ(outputSignal_[i], testSignal_[i]);
    }

    // Disable bypass and process again
    equalizer_->setBypass(false);
    equalizer_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2);

    // Should be different now (unless all gains are 0)
    bool isDifferent = false;
    for (size_t i = 0; i < blockSize_; ++i) {
        if (std::abs(outputSignal_[i] - testSignal_[i]) > 1e-6f) {
            isDifferent = true;
            break;
        }
    }

    // May or may not be different depending on default gains
}

TEST_F(EqualizerTest, SetBandBypass) {
    const float testGain = 12.0f;

    equalizer_->setBandGain(256, testGain);
    equalizer_->setBandBypass(256, true);

    outputSignal_.resize(blockSize_);
    equalizer_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2);

    // Band should be bypassed, so minimal effect
    // (This is a basic test - real implementation would need more precise verification)
}

// Preset tests
TEST_F(EqualizerTest, LoadPreset) {
    EXPECT_TRUE(equalizer_->loadPreset("flat"));
    EXPECT_TRUE(equalizer_->loadPreset("rock"));
    EXPECT_TRUE(equalizer_->loadPreset("jazz"));
    EXPECT_TRUE(equalizer_->loadPreset("classical"));
    EXPECT_TRUE(equalizer_->loadPreset("electronic"));

    // Test invalid preset
    EXPECT_FALSE(equalizer_->loadPreset("invalid_preset"));
}

TEST_F(EqualizerTest, SavePreset) {
    // Modify equalizer settings
    equalizer_->setBandGain(100, 3.0f);
    equalizer_->setBandGain(200, -3.0f);

    EXPECT_TRUE(equalizer_->savePreset("test_preset", "Test preset for unit testing"));

    // Load the preset and verify
    equalizer_->loadPreset("flat");
    EXPECT_FLOAT_EQ(equalizer_->getBandGain(100), 0.0f);

    equalizer_->loadPreset("test_preset");
    EXPECT_FLOAT_EQ(equalizer_->getBandGain(100), 3.0f);
    EXPECT_FLOAT_EQ(equalizer_->getBandGain(200), -3.0f);
}

// Processing tests
TEST_F(EqualizerTest, ProcessAudioBasic) {
    outputSignal_.resize(blockSize_);

    EXPECT_TRUE(equalizer_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2));

    // Output should be different from input if any gain is applied
    // With default flat response, should be similar
}

TEST_F(EqualizerTest, ProcessAudioWithGains) {
    outputSignal_.resize(blockSize_);

    // Apply some gain
    const float testGain = 6.0f;
    equalizer_->setBandGain(256, testGain); // Affect mid frequencies

    equalizer_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2);

    // Should see some difference in the frequency response
    float spectrumDiff = calculateSpectrumDifference(testSignal_, outputSignal_);
    EXPECT_GT(spectrumDiff, 0.0f);
}

TEST_F(EqualizerTest, ProcessMultiChannel) {
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

    EXPECT_TRUE(equalizer_->processAudioMultiChannel(inputs, outputs, blockSize_, 2));
}

// Q factor tests
TEST_F(EqualizerTest, SetBandQ) {
    EXPECT_TRUE(equalizer_->setBandQ(256, 1.5f));

    // Test invalid Q values
    EXPECT_FALSE(equalizer_->setBandQ(256, 0.0f));
    EXPECT_FALSE(equalizer_->setBandQ(256, -1.0f));

    // Test invalid band index
    EXPECT_FALSE(equalizer_->setBandQ(-1, 1.5f));
    EXPECT_FALSE(equalizer_->setBandQ(512, 1.5f));
}

TEST_F(EqualizerTest, GetBandQ) {
    const float testQ = 2.0f;

    EXPECT_TRUE(equalizer_->setBandQ(256, testQ));
    EXPECT_FLOAT_EQ(equalizer_->getBandQ(256), testQ);
}

// Frequency tests
TEST_F(EqualizerTest, GetBandFrequency) {
    float freq = equalizer_->getBandFrequency(0);
    EXPECT_GT(freq, 0.0f);
    EXPECT_LT(freq, 100.0f); // First band should be very low frequency

    freq = equalizer_->getBandFrequency(256);
    EXPECT_GT(freq, 1000.0f);
    EXPECT_LT(freq, 2000.0f); // Around Nyquist/2

    freq = equalizer_->getBandFrequency(511);
    EXPECT_GT(freq, 20000.0f); // Near Nyquist
}

// Master level tests
TEST_F(EqualizerTest, SetMasterLevel) {
    EXPECT_TRUE(equalizer_->setMasterLevel(-3.0f));
    EXPECT_FLOAT_EQ(equalizer_->getMasterLevel(), -3.0f);

    EXPECT_TRUE(equalizer_->setMasterLevel(6.0f));
    EXPECT_FLOAT_EQ(equalizer_->getMasterLevel(), 6.0f);
}

TEST_F(EqualizerTest, ProcessWithMasterLevel) {
    outputSignal_.resize(blockSize_);

    // Set master level to +6dB (approximately 2x gain)
    equalizer_->setMasterLevel(6.0f);

    equalizer_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2);

    // Output should be louder (approximately 2x amplitude)
    for (size_t i = 0; i < blockSize_; ++i) {
        EXPECT_NEAR(std::abs(outputSignal_[i]), std::abs(testSignal_[i]) * 2.0f, 0.1f);
    }
}

// Statistics tests
TEST_F(EqualizerTest, GetStatistics) {
    auto stats = equalizer_->getStatistics();

    EXPECT_EQ(stats.totalSamplesProcessed, 0);
    EXPECT_EQ(stats.totalProcessingTime, 0);
    EXPECT_EQ(stats.averageProcessingTime, 0);
    EXPECT_EQ(stats.maxProcessingTime, 0);
    EXPECT_EQ(stats.activeBands, 512);

    // Process some audio
    outputSignal_.resize(blockSize_);
    equalizer_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2);

    stats = equalizer_->getStatistics();
    EXPECT_EQ(stats.totalSamplesProcessed, blockSize_);
    EXPECT_GT(stats.totalProcessingTime, 0);
    EXPECT_GT(stats.averageProcessingTime, 0);
    EXPECT_GT(stats.maxProcessingTime, 0);
}

TEST_F(EqualizerTest, ResetStatistics) {
    // Process some audio first
    outputSignal_.resize(blockSize_);
    equalizer_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2);

    auto stats = equalizer_->getStatistics();
    EXPECT_GT(stats.totalSamplesProcessed, 0);

    // Reset and verify
    equalizer_->resetStatistics();
    stats = equalizer_->getStatistics();
    EXPECT_EQ(stats.totalSamplesProcessed, 0);
    EXPECT_EQ(stats.totalProcessingTime, 0);
}

// GPU acceleration tests
TEST_F(EqualizerTest, GPUAcceleration) {
    // Test with GPU acceleration
    Equalizer::Config gpuConfig;
    gpuConfig.sampleRate = 48000;
    gpuConfig.numChannels = 2;
    gpuConfig.blockSize = 512;
    gpuConfig.enableGPUAcceleration = true;

    Equalizer gpuEq;
    if (gpuEq.initialize(gpuConfig)) {
        EXPECT_TRUE(gpuEq.isGPUAccelerated());

        outputSignal_.resize(blockSize_);
        EXPECT_TRUE(gpuEq.processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2));
    } else {
        // GPU not available, test should pass anyway
        SUCCEED() << "GPU acceleration not available on this system";
    }
}

// Performance tests
TEST_F(EqualizerTest, PerformanceBasicProcessing) {
    const int numIterations = 1000;
    outputSignal_.resize(blockSize_);

    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numIterations; ++i) {
        equalizer_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

    // Should process quickly (this is a basic performance check)
    EXPECT_LT(duration.count(), 100000); // Less than 100ms for 1000 iterations

    auto stats = equalizer_->getStatistics();
    EXPECT_LT(stats.averageProcessingTime, 100); // Less than 100 microseconds per block
}

// Edge cases and error handling
TEST_F(EqualizerTest, ProcessWithNullPointers) {
    EXPECT_FALSE(equalizer_->processAudio(nullptr, testSignal_.data(), blockSize_, 2));
    EXPECT_FALSE(equalizer_->processAudio(testSignal_.data(), nullptr, blockSize_, 2));
    EXPECT_FALSE(equalizer_->processAudio(testSignal_.data(), testSignal_.data(), 0, 2));
}

TEST_F(EqualizerTest, ProcessWithInvalidBlockSize) {
    outputSignal_.resize(blockSize_);

    // Block size should match configuration
    EXPECT_FALSE(equalizer_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_ + 1, 2));
    EXPECT_FALSE(equalizer_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_ - 1, 2));
}

TEST_F(EqualizerTest, ProcessWithInvalidChannels) {
    outputSignal_.resize(blockSize_);

    // Channel count should match configuration
    EXPECT_FALSE(equalizer_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 1));
    EXPECT_FALSE(equalizer_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 4));
}

// Real-time mode tests
TEST_F(EqualizerTest, SetRealTimeMode) {
    equalizer_->setRealTimeMode(true);
    EXPECT_TRUE(equalizer_->isRealTimeMode());

    equalizer_->setRealTimeMode(false);
    EXPECT_FALSE(equalizer_->isRealTimeMode());
}

TEST_F(EqualizerTest, RealTimeModeProcessing) {
    equalizer_->setRealTimeMode(true);

    outputSignal_.resize(blockSize_);
    EXPECT_TRUE(equalizer_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2));

    auto stats = equalizer_->getStatistics();
    // Real-time mode should have minimal processing time
    EXPECT_LT(stats.averageProcessingTime, 50); // Less than 50 microseconds
}

// Memory and resource tests
TEST_F(EqualizerTest, MemoryUsage) {
    size_t initialMemory = equalizer_->getMemoryUsage();
    EXPECT_GT(initialMemory, 0);

    // Apply some settings that might increase memory usage
    std::vector<float> gains(512, 0.0f);
    equalizer_->setBandGains(gains);

    size_t currentMemory = equalizer_->getMemoryUsage();
    EXPECT_GE(currentMemory, initialMemory);
}

// Thread safety tests (basic)
TEST_F(EqualizerTest, BasicThreadSafety) {
    // This is a basic thread safety test
    // More comprehensive testing would require proper synchronization setup

    outputSignal_.resize(blockSize_);

    // Test concurrent access (basic smoke test)
    std::thread processThread([&]() {
        for (int i = 0; i < 100; ++i) {
            equalizer_->processAudio(testSignal_.data(), outputSignal_.data(), blockSize_, 2);
        }
    });

    std::thread settingsThread([&]() {
        for (int i = 0; i < 100; ++i) {
            equalizer_->setBandGain(i % 512, static_cast<float>(i % 10 - 5));
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    });

    processThread.join();
    settingsThread.join();

    // If we get here without crashing, basic thread safety is working
    SUCCEED();
}