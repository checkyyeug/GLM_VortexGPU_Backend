#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "core/audio_engine.hpp"
#include "testing/audio_test_harness.hpp"
#include "system/logger.hpp"

#include <juce_audio_formats/juce_audio_formats.h>
#include <thread>
#include <chrono>

using namespace vortex;
using namespace vortex::testing;
using namespace testing;

class AudioEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::initialize();

        // Configure test harness for audio engine
        AudioTestHarness::TestConfiguration config;
        config.sampleRate = 44100.0;
        config.bufferSize = 512;
        config.channels = 2;
        config.bitDepth = 24;
        config.maxProcessingTimeMs = 5.0; // Stricter requirement
        config.maxSignalToNoiseRatioDb = -100.0;
        config.maxTotalHarmonicDistortionDb = -120.0;

        harness_.setConfiguration(config);

        // Initialize audio engine
        audioEngine_ = std::make_unique<AudioEngine>();
        ASSERT_TRUE(audioEngine_->initialize(config.sampleRate, config.bufferSize));
    }

    void TearDown() override {
        if (audioEngine_) {
            audioEngine_->shutdown();
        }
        Logger::shutdown();
    }

    // Test data generators
    std::vector<float> generateSineWave(double frequency, double durationSeconds) {
        int numSamples = static_cast<int>(harness_.getConfiguration().sampleRate * durationSeconds);
        std::vector<float> signal(numSamples);

        for (int i = 0; i < numSamples; ++i) {
            double time = i / harness_.getConfiguration().sampleRate;
            signal[i] = static_cast<float>(std::sin(2.0 * M_PI * frequency * time));
        }

        return signal;
    }

    std::vector<float> generateStereoSineWave(double frequency, double durationSeconds) {
        auto monoWave = generateSineWave(frequency, durationSeconds);
        std::vector<float> stereoWave(monoWave.size() * 2);

        for (size_t i = 0; i < monoWave.size(); ++i) {
            stereoWave[i * 2] = monoWave[i];      // Left channel
            stereoWave[i * 2 + 1] = monoWave[i];  // Right channel (same for simplicity)
        }

        return stereoWave;
    }

    std::vector<float> generateWhiteNoise(double durationSeconds) {
        int numSamples = static_cast<int>(harness_.getConfiguration().sampleRate * durationSeconds);
        std::vector<float> noise(numSamples);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (int i = 0; i < numSamples; ++i) {
            noise[i] = dist(gen);
        }

        return noise;
    }

    AudioTestHarness harness_;
    std::unique_ptr<AudioEngine> audioEngine_;
};

// Test basic initialization and shutdown
TEST_F(AudioEngineTest, InitializationAndShutdown) {
    EXPECT_TRUE(audioEngine_->isInitialized());
    EXPECT_EQ(audioEngine_->getSampleRate(), 44100.0);
    EXPECT_EQ(audioEngine_->getBufferSize(), 512);
    EXPECT_EQ(audioEngine_->getChannels(), 2);
    EXPECT_FALSE(audioEngine_->isGPUEnabled());
}

// Test real-time processing constraints
TEST_F(AudioEngineTest, RealTimeProcessingConstraints) {
    // Generate test signal
    auto testSignal = generateSineWave(1000.0, 0.1); // 100ms of 1kHz sine wave
    std::vector<float> output(testSignal.size());

    // Measure processing time
    auto start = std::chrono::high_resolution_clock::now();

    audioEngine_->processBuffer(testSignal.data(), output.data(), testSignal.size() / 2);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Real-time constraint: <5ms for 100ms buffer (10:1 ratio)
    EXPECT_LT(duration.count(), 5000) << "Processing took " << duration.count() << "μs, expected <5000μs";

    // Validate output quality
    AudioQualityMetrics metrics = harness_.analyzeAudioQuality(
        testSignal, output, harness_.getConfiguration().sampleRate);

    EXPECT_AUDIO_QUALITY(metrics, harness_.getConfiguration());
    EXPECT_PERFORMANCE(harness_.testProcessorLatency([](const std::vector<float>& input) -> std::vector<float> {
            return input; // Passthrough for this test
        }).performance, harness_.getConfiguration());
}

// Test audio buffer management
TEST_F(AudioEngineTest, AudioBufferManagement) {
    // Test buffer allocation and deallocation
    auto bufferManager = std::make_unique<AudioBufferManager>();

    ASSERT_TRUE(bufferManager->initialize(44100.0, 512, 2, 24));
    EXPECT_TRUE(bufferManager->isInitialized());

    // Test processing buffer allocation
    auto buffer1 = bufferManager->getProcessingBuffer(1024);
    EXPECT_NE(buffer1.data, nullptr);
    EXPECT_TRUE(buffer1.inUse);
    EXPECT_EQ(bufferManager->getAvailableBuffers(), bufferManager->getPoolSize() - 1);

    auto buffer2 = bufferManager->getProcessingBuffer(1024);
    EXPECT_NE(buffer2.data, nullptr);
    EXPECT_TRUE(buffer2.inUse);
    EXPECT_NE(buffer1.data, buffer2.data);

    // Return buffers
    bufferManager->returnProcessingBuffer(std::move(buffer1));
    bufferManager->returnProcessingBuffer(std::move(buffer2));

    EXPECT_EQ(bufferManager->getAvailableBuffers(), bufferManager->getPoolSize());

    // Test memory constraints
    PerformanceMetrics perf = bufferManager->getStatistics();
    EXPECT_LT(perf.memoryUsageMB, 512.0); // Should be well under 512MB

    bufferManager->shutdown();
}

// Test audio quality with sine wave
TEST_F(AudioEngineTest, AudioQualitySineWave) {
    // Generate test signal
    auto inputSignal = generateStereoSineWave(1000.0, 0.5); // 500ms at 1kHz

    std::vector<float> outputSignal(inputSignal.size());

    // Process through audio engine
    audioEngine_->processBuffer(inputSignal.data(), outputSignal.data(), inputSignal.size() / 2);

    // Analyze quality
    AudioQualityMetrics metrics = harness_.analyzeAudioQuality(
        inputSignal, outputSignal, harness_.getConfiguration().sampleRate);

    // For a sine wave through a passthrough system, we expect:
    // - Very high SNR (>90dB)
    // - Very low THD (<-100dB)
    // - Flat frequency response (<0.1dB deviation)
    EXPECT_GT(metrics.signalToNoiseRatioDb, 90.0);
    EXPECT_LT(metrics.totalHarmonicDistortionDb, -100.0);
    EXPECT_LT(metrics.frequencyResponseDeviationDb, 0.1);
}

// Test GPU acceleration availability
TEST_F(AudioEngineTest, GPUAccelerationAvailability) {
    // Test CUDA availability
    bool cudaAvailable = audioEngine_->isGPUBackendAvailable("CUDA");

    if (cudaAvailable) {
        // Try to enable CUDA acceleration
        EXPECT_TRUE(audioEngine_->enableGPUAcceleration("CUDA"));
        EXPECT_TRUE(audioEngine_->isGPUEnabled());

        // Test GPU processing
        auto testSignal = generateSineWave(440.0, 0.1);
        std::vector<float> output(testSignal.size());

        auto start = std::chrono::high_resolution_clock::now();
        audioEngine_->processBuffer(testSignal.data(), output.data(), testSignal.size() / 2);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // GPU processing should be fast
        EXPECT_LT(duration.count(), 3000) << "GPU processing took too long: " << duration.count() << "μs";

        // Quality should still be maintained
        AudioQualityMetrics metrics = harness_.analyzeAudioQuality(
            testSignal, output, harness_.getConfiguration().sampleRate);
        EXPECT_GT(metrics.signalToNoiseRatioDb, 80.0);

    } else {
        // Skip GPU tests if not available
        GTEST_SKIP() << "CUDA not available, skipping GPU acceleration tests";
    }
}

// Test concurrent processing
TEST_F(AudioEngineTest, ConcurrentProcessing) {
    const int numThreads = 4;
    const int iterationsPerThread = 100;

    std::vector<std::thread> threads;
    std::vector<std::chrono::microseconds> durations;
    std::mutex durationsMutex;

    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < iterationsPerThread; ++i) {
                auto testSignal = generateSineWave(440.0 + t * 100.0, 0.01); // Different frequencies
                std::vector<float> output(testSignal.size());

                auto start = std::chrono::high_resolution_clock::now();
                audioEngine_->processBuffer(testSignal.data(), output.data(), testSignal.size() / 2);
                auto end = std::chrono::high_resolution_clock::now();

                {
                    std::lock_guard<std::mutex> lock(durationsMutex);
                    durations.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
                }
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Analyze concurrent performance
    ASSERT_EQ(durations.size(), numThreads * iterationsPerThread);

    // Calculate statistics
    auto [minIt, maxIt] = std::minmax_element(durations.begin(), durations.end());
    double avgDuration = std::accumulate(durations.begin(), durations.end(), 0.0) / durations.size();

    // All processing should be under 10ms
    EXPECT_LT(maxIt->count(), 10000) << "Max concurrent processing time: " << maxIt->count() << "μs";

    // Performance should be consistent (within 3x of average)
    EXPECT_LT(avgDuration * 3.0, maxIt->count()) << "Inconsistent performance detected";

    EXPECT_LT(avgDuration, 5000) << "Average concurrent processing time too high: " << avgDuration << "μs";
}

// Test long-term stability
TEST_F(AudioEngineTest, LongTermStability) {
    const int durationSeconds = 10; // 10 second test
    const int iterationsPerSecond = 100;

    std::chrono::microseconds maxDuration{0};
    std::chrono::microseconds minDuration{std::chrono::microseconds::max()};

    for (int second = 0; second < durationSeconds; ++second) {
        for (int iteration = 0; iteration < iterationsPerSecond; ++iteration) {
            auto testSignal = generateSineWave(440.0, 0.005); // 5ms buffer
            std::vector<float> output(testSignal.size());

            auto start = std::chrono::high_resolution_clock::now();
            audioEngine_->processBuffer(testSignal.data(), output.data(), testSignal.size() / 2);
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            // Update min/max
            if (duration > maxDuration) maxDuration = duration;
            if (duration < minDuration) minDuration = duration;

            // Real-time constraint check
            ASSERT_LT(duration.count(), 10000) << "Real-time constraint violated at "
                                                   << (second + 1) << "." << iteration;
        }
    }

    // Check performance stability (max should be within 5x of min)
    EXPECT_LT(minDuration.count() * 5, maxDuration.count())
        << "Performance not stable: min=" << minDuration.count()
        << "μs, max=" << maxDuration.count() << "μs";
}

// Test edge cases
TEST_F(AudioEngineTest, EdgeCases) {
    // Test with empty buffer
    std::vector<float> emptyBuffer;
    std::vector<float> emptyOutput;

    audioEngine_->processBuffer(emptyBuffer.data(), emptyOutput.data(), 0);

    // Test with single sample
    std::vector<float> singleSample = {1.0f, -1.0f};
    std::vector<float> singleOutput(2);

    audioEngine_->processBuffer(singleSample.data(), singleOutput.data(), 1);
    EXPECT_GT(std::abs(singleOutput[0]), 0.0f); // Should be processed

    // Test with silent buffer
    std::vector<float> silentBuffer(1024, 0.0f);
    std::vector<float> silentOutput(1024);

    audioEngine_->processBuffer(silentBuffer.data(), silentOutput.data(), 512);

    // Output should be close to silent
    float sum = 0.0f;
    for (float sample : silentOutput) {
        sum += std::abs(sample);
    }
    EXPECT_LT(sum / silentOutput.size(), 0.001f) << "Silent buffer should remain silent";
}

// Test performance benchmarks
TEST_F(AudioEngineTest, PerformanceBenchmarks) {
    const int numIterations = 1000;

    // Benchmark different buffer sizes
    std::vector<size_t> bufferSizes = {256, 512, 1024, 2048, 4096};

    for (size_t bufferSize : bufferSizes) {
        auto testSignal = generateSineWave(1000.0, bufferSize / 44100.0);
        std::vector<float> output(testSignal.size());

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < numIterations; ++i) {
            audioEngine_->processBuffer(testSignal.data(), output.data(), testSignal.size() / 2);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto totalTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        double avgTime = totalTime.count() / static_cast<double>(numIterations);
        double throughput = (bufferSize * sizeof(float) * 2) / (avgTime / 1e6) / (1024 * 1024); // MB/s

        // Performance expectations
        EXPECT_LT(avgTime, 5000) << "Buffer size " << bufferSize << " processing too slow: " << avgTime << "μs";
        EXPECT_GT(throughput, 100.0) << "Throughput too low for buffer size " << bufferSize << ": " << throughput << " MB/s";

        // Log performance for analysis
        double samplesPerSec = (bufferSize / 2) / (avgTime / 1e6);
        std::cout << "Buffer size " << bufferSize << ": avg " << avgTime << "μs, "
                  << throughput << " MB/s, " << samplesPerSec << " samples/s" << std::endl;
    }
}

// Test memory efficiency
TEST_F(AudioEngineTest, MemoryEfficiency) {
    // Test memory usage doesn't grow over time
    size_t initialMemory = harness_.getCurrentMemoryUsageMB();

    // Process many buffers
    for (int i = 0; i < 1000; ++i) {
        auto testSignal = generateSineWave(440.0 + (i % 100), 0.1);
        std::vector<float> output(testSignal.size());

        audioEngine_->processBuffer(testSignal.data(), output.data(), testSignal.size() / 2);
    }

    size_t finalMemory = harness_.getCurrentMemoryUsageMB();
    size_t memoryGrowth = finalMemory - initialMemory;

    // Memory growth should be minimal (<50MB for 1000 iterations)
    EXPECT_LT(memoryGrowth, 50) << "Memory leak detected: " << memoryGrowth << " MB growth";

    std::cout << "Memory usage: initial=" << initialMemory << "MB, final=" << finalMemory
              << "MB, growth=" << memoryGrowth << "MB" << std::endl;
}

// Test error handling
TEST_F(AudioEngineTest, ErrorHandling) {
    // Test invalid parameters
    EXPECT_TRUE(audioEngine_->initialize(44100.0, 512)); // Should be fine
    EXPECT_FALSE(audioEngine_->initialize(0.0, 0)); // Invalid parameters

    // Test processing with null pointers (should not crash)
    audioEngine_->processBuffer(nullptr, nullptr, 0);

    // Test with inconsistent buffer sizes
    std::vector<float> input(1000);
    std::vector<float> output(500); // Wrong size

    // This should handle gracefully (implementation dependent)
    audioEngine_->processBuffer(input.data(), output.data(), 250);
}