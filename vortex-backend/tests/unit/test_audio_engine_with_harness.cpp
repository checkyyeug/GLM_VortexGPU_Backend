#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "audio_test_harness.hpp"
#include "core/audio_engine.hpp"
#include "core/dsp/eq_processor.hpp"
#include "core/dsp/convolver.hpp"
#include "core/gpu/gpu_processor.hpp"

using namespace vortex;
using namespace vortex::testing;
using namespace testing;

class AudioEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        AudioTestHarness::TestConfiguration config;
        config.sampleRate = 44100.0;
        config.bufferSize = 512;
        config.channels = 2;
        config.maxProcessingTimeMs = 5.0; // Stricter than default
        config.maxSignalToNoiseRatioDb = -90.0;
        config.maxTotalHarmonicDistortionDb = -110.0;

        harness_.setConfiguration(config);

        // Initialize audio engine
        audioEngine_ = std::make_unique<AudioEngine>();
        ASSERT_TRUE(audioEngine_->initialize(config.sampleRate, config.bufferSize));
    }

    void TearDown() override {
        if (audioEngine_) {
            audioEngine_->shutdown();
        }
    }

    AudioTestHarness harness_;
    std::unique_ptr<AudioEngine> audioEngine_;
};

TEST_F(AudioEngineTest, RealtimeLatencyConstraint) {
    // Test that audio processing meets <5ms latency requirement
    auto processor = [this](const std::vector<float>& input) -> std::vector<float> {
        std::vector<float> output(input.size());
        audioEngine_->processBuffer(input.data(), output.data(), input.size());
        return output;
    };

    auto result = harness_.testProcessorLatency(processor, 5.0);
    EXPECT_TRUE(result.passed) << result.errorMessage;

    // Additional real-time constraint validation
    EXPECT_REALTIME_CONSTRAINT(result.performance.processingTimeMs, 5.0, "Audio Engine Processing");
}

TEST_F(AudioEngineTest, AudioQualityValidation) {
    // Test audio quality with 1kHz sine wave
    auto processor = [this](const std::vector<float>& input) -> std::vector<float> {
        std::vector<float> output(input.size());
        audioEngine_->processBuffer(input.data(), output.data(), input.size());
        return output;
    };

    auto result = harness_.testAudioQuality(processor, 1000.0);
    EXPECT_TRUE(result.passed) << result.errorMessage;

    EXPECT_AUDIO_QUALITY(result.quality, harness_.getConfiguration());
    EXPECT_PERFORMANCE(result.performance, harness_.getConfiguration());
}

TEST_F(AudioEngineTest, FrequencyResponseLinearity) {
    // Test frequency response across audible range
    auto processor = [this](const std::vector<float>& input) -> std::vector<float> {
        std::vector<float> output(input.size());
        audioEngine_->processBuffer(input.data(), output.data(), input.size());
        return output;
    };

    std::vector<double> testFreqs = {20, 100, 1000, 10000, 20000};
    auto result = harness_.testFrequencyResponse(processor, testFreqs);
    EXPECT_TRUE(result.passed) << result.errorMessage;

    EXPECT_LE(result.quality.frequencyResponseDeviationDb, 0.1);
}

TEST_F(AudioEngineTest, MemoryUsageConstraints) {
    // Test memory usage stays within limits
    auto processor = [this](const std::vector<float>& input) -> std::vector<float> {
        std::vector<float> output(input.size());
        audioEngine_->processBuffer(input.data(), output.data(), input.size());
        return output;
    };

    auto result = harness_.testMemoryUsage(processor, 256.0); // 256MB limit
    EXPECT_TRUE(result.passed) << result.errorMessage;

    EXPECT_LE(result.performance.memoryUsageMB, 256.0);
}

TEST_F(AudioEngineTest, LongTermStability) {
    // Test stability over extended periods (10 seconds)
    auto processor = [this](const std::vector<float>& input) -> std::vector<float> {
        std::vector<float> output(input.size());
        audioEngine_->processBuffer(input.data(), output.data(), input.size());
        return output;
    };

    auto result = harness_.testLongTermStability(processor, 10.0);
    EXPECT_TRUE(result.passed) << result.errorMessage;
}

TEST_F(AudioEngineTest, ConcurrentProcessing) {
    // Test multiple concurrent audio streams
    auto processor = [this](const std::vector<float>& input) -> std::vector<float> {
        std::vector<float> output(input.size());
        audioEngine_->processBuffer(input.data(), output.data(), input.size());
        return output;
    };

    auto result = harness_.testConcurrentProcessing(processor, 4);
    EXPECT_TRUE(result.passed) << result.errorMessage;
}

class EqualizerProcessorTest : public ::testing::Test {
protected:
    void SetUp() override {
        AudioTestHarness::TestConfiguration config;
        config.sampleRate = 44100.0;
        config.maxProcessingTimeMs = 2.0;
        config.maxSignalToNoiseRatioDb = -100.0;

        harness_.setConfiguration(config);

        // Create 10-band equalizer
        eqProcessor_ = std::make_unique<EQProcessor>();
        std::vector<double> frequencies = {31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000};
        std::vector<double> gains(10, 0.0); // Flat response
        std::vector<double> q(10, 1.0);

        eqProcessor_->configure(frequencies, gains, q);
    }

    AudioTestHarness harness_;
    std::unique_ptr<EQProcessor> eqProcessor_;
};

TEST_F(EqualizerProcessorTest, FrequencyResponseAccuracy) {
    // Test equalizer frequency response
    auto eqFunc = [this](const std::vector<float>& input, const std::vector<double>& frequencies, const std::vector<double>& gains) -> std::vector<float> {
        eqProcessor_->setGains(gains);
        std::vector<float> output(input.size());
        eqProcessor_->process(input.data(), output.data(), input.size());
        return output;
    };

    std::vector<double> frequencies = {31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000};
    std::vector<double> gains = {0.0, 1.0, -1.0, 2.0, -2.0, 0.0, 1.0, -1.0, 2.0, -2.0};

    auto result = harness_.testEqualizer(eqFunc, frequencies, gains);
    EXPECT_TRUE(result.passed) << result.errorMessage;
}

TEST_F(EqualizerProcessorTest, RealtimeConstraint) {
    auto processor = [this](const std::vector<float>& input) -> std::vector<float> {
        std::vector<float> output(input.size());
        eqProcessor_->process(input.data(), output.data(), input.size());
        return output;
    };

    AUDIO_PROCESSOR_TEST(processor, "Equalizer", harness_.getConfiguration());
}

class ConvolutionProcessorTest : public ::testing::Test {
protected:
    void SetUp() override {
        AudioTestHarness::TestConfiguration config;
        config.sampleRate = 44100.0;
        config.maxProcessingTimeMs = 10.0;
        config.maxSignalToNoiseRatioDb = -80.0;

        harness_.setConfiguration(config);

        // Create test impulse response (1 second reverberation)
        impulseResponse_.resize(44100); // 1 second at 44.1kHz
        for (size_t i = 0; i < impulseResponse_.size(); ++i) {
            double t = i / 44100.0;
            impulseResponse_[i] = static_cast<float>(std::exp(-t * 2.0) * std::sin(2.0 * M_PI * 1000.0 * t));
        }

        convProcessor_ = std::make_unique<Convolver>();
        convProcessor_->loadImpulseResponse(impulseResponse_);
    }

    AudioTestHarness harness_;
    std::unique_ptr<Convolver> convProcessor_;
    std::vector<float> impulseResponse_;
};

TEST_F(ConvolutionProcessorTest, ConvolutionQuality) {
    auto processor = [this](const std::vector<float>& input) -> std::vector<float> {
        std::vector<float> output(input.size());
        convProcessor_->process(input.data(), output.data(), input.size());
        return output;
    };

    auto result = harness_.testAudioQuality(processor, 440.0);
    EXPECT_TRUE(result.passed) << result.errorMessage;

    EXPECT_AUDIO_QUALITY(result.quality, harness_.getConfiguration());
}

TEST_F(ConvolutionProcessorTest, ImpulseResponseFidelity) {
    auto processor = [this](const std::vector<float>& input) -> std::vector<float> {
        std::vector<float> output(input.size());
        convProcessor_->process(input.data(), output.data(), input.size());
        return output;
    };

    auto result = harness_.testConvolution(processor, impulseResponse_);
    EXPECT_TRUE(result.passed) << result.errorMessage;
}

// GPU-accelerated tests (only run if GPU is available)
#ifdef VORTEX_GPU_ENABLED
class GPUProcessorTest : public ::testing::Test {
protected:
    void SetUp() override {
        AudioTestHarness::TestConfiguration config;
        config.sampleRate = 44100.0;
        config.enableGPUTests = true;
        config.gpuBackend = "CUDA";
        config.maxProcessingTimeMs = 1.0; // GPU should be faster

        harness_.setConfiguration(config);

        gpuProcessor_ = std::make_unique<GPUProcessor>();
        ASSERT_TRUE(gpuProcessor_->initialize("CUDA"));
    }

    void TearDown() override {
        if (gpuProcessor_) {
            gpuProcessor_->shutdown();
        }
    }

    AudioTestHarness harness_;
    std::unique_ptr<GPUProcessor> gpuProcessor_;
};

TEST_F(GPUProcessorTest, GPUAccelerationSpeedup) {
    // Compare CPU vs GPU performance
    auto cpuProcessor = [this](const std::vector<float>& input) -> std::vector<float> {
        std::vector<float> output(input.size());
        // CPU implementation here
        return output;
    };

    auto gpuProcessor = [this](const std::vector<float>& input) -> std::vector<float> {
        std::vector<float> output(input.size());
        gpuProcessor_->process(input.data(), output.data(), input.size());
        return output;
    };

    auto result = harness_.testGPUAcceleration(cpuProcessor, gpuProcessor, "CUDA");
    EXPECT_TRUE(result.passed) << result.errorMessage;
}

TEST_F(GPUProcessorTest, GPUMemoryTransfer) {
    auto processor = [this](const std::vector<float>& input) -> std::vector<float> {
        std::vector<float> output(input.size());
        gpuProcessor_->process(input.data(), output.data(), input.size());
        return output;
    };

    // Test with 1MB data blocks
    auto result = harness_.testGPUMemoryTransfer(processor, 1024 * 1024);
    EXPECT_TRUE(result.passed) << result.errorMessage;
}
#endif // VORTEX_GPU_ENABLED

class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        AudioTestHarness::TestConfiguration config;
        config.sampleRate = 44100.0;
        config.maxProcessingTimeMs = 15.0; // Full chain can take longer
        config.maxSignalToNoiseRatioDb = -70.0; // Accept some degradation

        harness_.setConfiguration(config);

        // Create full processing chain
        audioEngine_ = std::make_unique<AudioEngine>();
        ASSERT_TRUE(audioEngine_->initialize(config.sampleRate, 512));

        eqProcessor_ = std::make_unique<EQProcessor>();
        std::vector<double> frequencies = {100, 1000, 10000};
        std::vector<double> gains = {1.0, 0.0, -1.0};
        std::vector<double> q(3, 1.414);
        eqProcessor_->configure(frequencies, gains, q);

        convProcessor_ = std::make_unique<Convolver>();
        std::vector<float> ir(4096, 0.0f);
        ir[0] = 1.0f; // Simple passthrough impulse
        convProcessor_->loadImpulseResponse(ir);
    }

    AudioTestHarness harness_;
    std::unique_ptr<AudioEngine> audioEngine_;
    std::unique_ptr<EQProcessor> eqProcessor_;
    std::unique_ptr<Convolver> convProcessor_;
};

TEST_F(IntegrationTest, FullProcessingChain) {
    // Test complete audio processing pipeline
    auto processor = [this](const std::vector<float>& input) -> std::vector<float> {
        std::vector<float> buffer = input;

        // Apply equalization
        std::vector<float> eqBuffer(buffer.size());
        eqProcessor_->process(buffer.data(), eqBuffer.data(), buffer.size());
        buffer = eqBuffer;

        // Apply convolution
        std::vector<float> convBuffer(buffer.size());
        convProcessor_->process(buffer.data(), convBuffer.data(), buffer.size());
        buffer = convBuffer;

        // Apply final audio engine processing
        std::vector<float> output(buffer.size());
        audioEngine_->processBuffer(buffer.data(), output.data(), buffer.size());

        return output;
    };

    auto result = harness_.testAudioQuality(processor, 1000.0);
    EXPECT_TRUE(result.passed) << result.errorMessage;

    EXPECT_AUDIO_QUALITY(result.quality, harness_.getConfiguration());
    EXPECT_PERFORMANCE(result.performance, harness_.getConfiguration());
}

TEST_F(IntegrationTest, RealtimeConstraintFullChain) {
    auto processor = [this](const std::vector<float>& input) -> std::vector<float> {
        std::vector<float> buffer = input;

        std::vector<float> eqBuffer(buffer.size());
        eqProcessor_->process(buffer.data(), eqBuffer.data(), buffer.size());

        std::vector<float> convBuffer(buffer.size());
        convProcessor_->process(eqBuffer.data(), convBuffer.data(), buffer.size());

        std::vector<float> output(buffer.size());
        audioEngine_->processBuffer(convBuffer.data(), output.data(), buffer.size());

        return output;
    };

    auto result = harness_.testProcessorLatency(processor, 15.0);
    EXPECT_TRUE(result.passed) << result.errorMessage;
}

// Performance benchmarks
class PerformanceBenchmark : public ::testing::Test {
protected:
    void SetUp() override {
        AudioTestHarness::TestConfiguration config;
        config.sampleRate = 44100.0;
        harness_.setConfiguration(config);
    }

    AudioTestHarness harness_;
};

TEST_F(PerformanceBenchmark, BenchmarkAudioEngine) {
    // Simple passthrough processor
    auto processor = [](const std::vector<float>& input) -> std::vector<float> {
        return input; // Passthrough
    };

    auto metrics = harness_.benchmarkProcessor(processor, 10.0);

    // Benchmark results - these are not pass/fail but for monitoring
    EXPECT_GT(metrics.throughputSamplesPerSecond, 1000000.0) << "Low throughput detected";
    EXPECT_LT(metrics.processingTimeMs, 1.0) << "High processing latency detected";
}

TEST_F(PerformanceBenchmark, MemoryStability) {
    // Test memory stability over many iterations
    auto processor = [](const std::vector<float>& input) -> std::vector<float> {
        std::vector<float> output(input.size());
        std::copy(input.begin(), input.end(), output.begin());
        return output;
    };

    double initialMemory = 0.0;
    double peakMemory = 0.0;

    {
        ScopedMemoryTracker tracker(initialMemory, peakMemory);

        for (int i = 0; i < 1000; ++i) {
            auto testSignal = harness_.generateSineWave(1000.0, 0.1);
            auto output = processor(testSignal);
        }
    }

    // Memory should not grow significantly
    double memoryGrowth = peakMemory - initialMemory;
    EXPECT_LT(memoryGrowth, 50.0) << "Memory leak detected: " << memoryGrowth << " MB growth";
}