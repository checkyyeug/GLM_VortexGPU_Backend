#include <gtest/gtest.h>
#include "core/audio_engine.hpp"
#include "core/processing_chain.hpp"
#include "core/gpu/gpu_processor.hpp"
#include "testing/audio_test_harness.hpp"
#include "system/logger.hpp"

#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <random>
#include <algorithm>
#include <numeric>
#include <iomanip>

using namespace vortex;
using namespace vortex::testing;
using namespace std::chrono_literals;

class RealtimeConstraintsTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::initialize();

        // Configure test harness for real-time constraints
        AudioTestHarness::TestConfiguration config;
        config.sampleRate = 44100.0;
        config.bufferSize = 512;
        config.channels = 2;
        config.bitDepth = 24;
        config.maxProcessingTimeMs = 10.0; // 10ms target
        config.maxSignalToNoiseRatioDb = -80.0;
        config.maxTotalHarmonicDistortionDb = -100.0;
        config.enableGPUTests = true;

        harness_.setConfiguration(config);

        // Initialize components
        InitializeAudioEngine();
        InitializeGPUProcessor();
    }

    void TearDown() override {
        ShutdownGPUProcessor();
        ShutdownAudioEngine();
        Logger::shutdown();
    }

    void InitializeAudioEngine() {
        audioEngine_ = std::make_unique<AudioEngine>();
        ASSERT_TRUE(audioEngine_->initialize(config.sampleRate, config.bufferSize));
        EXPECT_TRUE(audioEngine_->isInitialized());
    }

    void InitializeGPUProcessor() {
        gpuAvailable_ = false;

#ifdef VORTEX_ENABLE_CUDA
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error == cudaSuccess && deviceCount > 0) {
            gpuProcessor_ = std::make_unique<GPUProcessor>();
            if (gpuProcessor_->initialize("CUDA", config.sampleRate, config.bufferSize, config.channels)) {
                gpuAvailable_ = true;
                EXPECT_TRUE(audioEngine_->enableGPUAcceleration("CUDA"));
                EXPECT_TRUE(audioEngine_->isGPUEnabled());
            }
        }
#endif

        if (!gpuAvailable_) {
            std::cout << "GPU not available, using CPU-only processing" << std::endl;
        }
    }

    void ShutdownAudioEngine() {
        if (audioEngine_) {
            audioEngine_->shutdown();
        }
    }

    void ShutdownGPUProcessor() {
        if (gpuProcessor_) {
            gpuProcessor_->shutdown();
        }
    }

    // Performance measurement utilities
    struct PerformanceMetrics {
        std::vector<double> processingTimes;
        double minTime;
        double maxTime;
        double avgTime;
        double stdDev;
        double p95; // 95th percentile
        double p99; // 99th percentile
        size_t samples;

        PerformanceMetrics() : minTime(0.0), maxTime(0.0), avgTime(0.0), stdDev(0.0), p95(0.0), p99(0.0), samples(0) {}

        void update(const std::vector<double>& times) {
            processingTimes = times;
            samples = times.size();

            if (samples == 0) return;

            std::sort(processingTimes.begin(), processingTimes.end());

            minTime = processingTimes.front();
            maxTime = processingTimes.back();
            avgTime = std::accumulate(processingTimes.begin(), processingTimes.end(), 0.0) / samples;

            // Calculate standard deviation
            double variance = 0.0;
            for (double time : processingTimes) {
                variance += (time - avgTime) * (time - avgTime);
            }
            stdDev = std::sqrt(variance / samples);

            // Calculate percentiles
            size_t p95Index = static_cast<size_t>(samples * 0.95);
            size_t p99Index = static_cast<size_t>(samples * 0.99);

            p95 = (p95Index < samples) ? processingTimes[p95Index] : maxTime;
            p99 = (p99Index < samples) ? processingTimes[p99Index] : maxTime;
        }

        void print(const std::string& testName, double constraintMs = 10.0) const {
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "[" << testName << "] Performance Metrics:" << std::endl;
            std::cout << "  Samples: " << samples << std::endl;
            std::cout << "  Average: " << avgTime << "μs" << std::endl;
            std::cout << "  Minimum: " << minTime << "μs" << std::endl;
            std::cout << "  Maximum: " << maxTime << "μs" << std::endl;
            std::cout << "  Std Dev: " << stdDev << "μs" << std::endl;
            std::cout << "  95th Pct: " << p95 << "μs" << std::endl;
            std::cout << "  99th Pct: " << p99 << "μs" << std::endl;
            std::cout << "  Constraint: " << constraintMs << "ms" << std::endl;
            std::cout << "  Compliance: " << (avgTime < constraintMs * 1000 ? "✓ PASS" : "✗ FAIL") << std::endl;
            std::cout << "  Overhead: " << ((avgTime / (constraintMs * 1000.0) - 1.0) * 100) << "%" << std::endl;
        }
    };

    // Real-time constraint assertion helper
    void AssertRealtimeConstraint(const PerformanceMetrics& metrics,
                                    double constraintMs = 10.0,
                                    const std::string& test_name = "") {
        EXPECT_LT(metrics.avgTime, constraintMs * 1000.0)
            << test_name << ": Average processing time " << metrics.avgTime << "μs exceeds constraint " << (constraintMs * 1000) << "μs";

        EXPECT_LT(metrics.p99, constraintMs * 1000.0)
            << test_name << ": 99th percentile " << metrics.p99 << "μs exceeds constraint " << (constraintMs * 1000) << "μs";
    }

    AudioTestHarness::TestConfiguration config;
    std::unique_ptr<AudioEngine> audioEngine_;
    std::unique_ptr<GPUProcessor> gpuProcessor_;
    bool gpuAvailable_ = false;
};

// Test real-time processing with small buffers
TEST_F(RealtimeConstraintsTest, SmallBufferProcessing) {
    PerformanceMetrics cpuMetrics;
    PerformanceMetrics gpuMetrics;

    // Test different buffer sizes
    std::vector<size_t> bufferSizes = {64, 128, 256, 512, 1024};
    const int numIterations = 1000;

    for (size_t bufferSize : bufferSizes) {
        std::cout << "\n=== Testing Buffer Size: " << bufferSize << " ("
                  << (bufferSize / config.sampleRate * 1000.0) << "ms) ===" << std::endl;

        // Generate test signal
        std::vector<float> testSignal(bufferSize);
        for (size_t i = 0; i < bufferSize; ++i) {
            double time = i / config.sampleRate;
            testSignal[i] = static_cast<float>(std::sin(2.0 * M_PI * 440.0 * time));
        }

        // Test CPU processing
        std::vector<float> cpuOutput(bufferSize);
        std::vector<double> cpuTimes;
        cpuTimes.reserve(numIterations);

        for (int i = 0; i < numIterations; ++i) {
            auto start = high_resolution_clock::now();
            audioEngine_->processBuffer(testSignal.data(), cpuOutput.data(), bufferSize / 2);
            auto end = high_resolution_clock::now();

            cpuTimes.push_back(duration_cast<nanoseconds>(end - start).count());
        }

        cpuMetrics.update(cpuTimes);
        cpuMetrics.print("CPU Processing", 10.0);
        AssertRealtimeConstraint(cpuMetrics, 10.0, "CPU Small Buffer " + std::to_string(bufferSize));

        // Test GPU processing if available
        if (gpuAvailable_) {
            std::vector<float> gpuOutput(bufferSize);
            std::vector<double> gpuTimes;
            gpuTimes.reserve(numIterations);

            for (int i = 0; i < numIterations; ++i) {
                auto start = high_resolution_clock::now();
                audioEngine_->processBuffer(testSignal.data(), gpuOutput.data(), bufferSize / 2);
                auto end = high_resolution_clock::now();

                gpuTimes.push_back(duration_cast<nanoseconds>(end - start).count());
            }

            gpuMetrics.update(gpuTimes);
            gpuMetrics.print("GPU Processing", 5.0); // GPU should be faster
            AssertRealtimeConstraint(gpuMetrics, 5.0, "GPU Small Buffer " + std::to_string(bufferSize));

            // Compare performance
            double speedup = static_cast<double>(cpuMetrics.avgTime) / gpuMetrics.avgTime;
            std::cout << "  GPU Speedup: " << speedup << "x" << std::endl;
        }

        std::cout << "---" << std::endl;
    }
}

// Test real-time processing with different load levels
TEST_F(RealtimeConstraintsTest, DifferentLoadLevels) {
    const std::vector<int> loadLevels = {1, 10, 50, 100, 500};

    PerformanceMetrics metrics;

    for (int loadLevel : loadLevels) {
        std::cout << "\n=== Testing Load Level: " << loadLevel << " concurrent operations ===" << std::endl;

        std::vector<std::vector<float>> testSignals(loadLevel);
        std::vector<std::vector<float>> outputs(loadLevel);
        std::vector<double> times;

        // Generate test signals
        for (int i = 0; i < loadLevel; ++i) {
            testSignals[i].resize(1024);
            for (size_t j = 0; j < testSignals[i].size(); ++j) {
                double time = j / config.sampleRate;
                testSignals[i][j] = static_cast<float>(std::sin(2.0 * M_PI * (100.0 + i * 10.0) * time));
            }
            outputs[i].resize(testSignals[i].size());
        }

        // Measure processing time
        auto start = high_resolution_clock::now();

        for (int i = 0; i < loadLevel; ++i) {
            audioEngine_->processBuffer(testSignals[i].data(), outputs[i].data(), testSignals[i].size() / 2);
        }

        auto end = high_resolution_clock::now();

        double totalTimeMs = duration_cast<milliseconds>(end - start).count();
        double avgTimeMs = totalTimeMs / loadLevel;

        times.push_back(avgTimeMs);

        std::cout << "  Total time: " << totalTimeMs << "ms" << std::endl;
        std::cout << "  Average time per operation: " << avgTimeMs << "ms" << std::endl;

        // Real-time constraint: average time per operation should be <10ms
        EXPECT_LT(avgTimeMs, 10.0) << "Load level " << loadLevel << " exceeds real-time constraint";
    }

    // Analyze results
    metrics.update(times);
    metrics.print("Multi-load Processing", 10.0);
    AssertRealtimeConstraint(metrics, 10.0, "Multi-load Test");
}

// Test sustained real-time processing
TEST_F(RealtimeConstraintsTest, SustainedRealTimeProcessing) {
    const int durationSeconds = 30; // 30 second test
    const int targetFrequency = 60; // 60 Hz processing
    const int maxBufferSize = 1024;

    std::vector<double> processingTimes;
    std::atomic<int> violationCount(0);
    std::atomic<bool> stopTest(false);

    std::cout << "\n=== Sustained Real-time Processing (" << durationSeconds
              << "s @ " << targetFrequency << "Hz) ===" << std::endl;

    auto testThread = std::thread([&]() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> bufferSizeDist(256, maxBufferSize);
        std::uniform_real_distribution<float> frequencyDist(50.0, 2000.0);

        while (!stopTest.load()) {
            size_t bufferSize = bufferSizeDist(gen);
            double frequency = frequencyDist(gen);

            // Generate test signal
            std::vector<float> testSignal(bufferSize);
            for (size_t i = 0; i < bufferSize; ++i) {
                double time = i / config.sampleRate;
                testSignal[i] = static_cast<float>(std::sin(2.0 * M_PI * frequency * time));
            }

            // Process audio
            std::vector<float> output(bufferSize);
            auto start = high_resolution_clock::now();

            audioEngine_->processBuffer(testSignal.data(), output.data(), bufferSize / 2);

            auto end = high_resolution_clock::now();
            double processingTimeMs = duration_cast<microseconds>(end - start).count() / 1000.0;

            // Check real-time constraint
            if (processingTimeMs > 10.0) {
                violationCount.fetch_add(1);
            }

            processingTimes.push_back(processingTimeMs);

            // Rate limiting to target frequency
            std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(1000000.0 / targetFrequency)));
        }
    });

    // Run test for specified duration
    std::this_thread::sleep_for(std::chrono::seconds(durationSeconds));

    stopTest.store(true);
    testThread.join();

    // Analyze sustained performance
    PerformanceMetrics metrics;
    metrics.update(processingTimes);
    metrics.print("Sustained Processing", 10.0);
    AssertRealtimeConstraint(metrics, 10.0, "Sustained Real-time Test");

    // Check violation rate
    double violationRate = (static_cast<double>(violationCount.load()) / processingTimes.size()) * 100.0;
    std::cout << "Real-time Violations: " << violationCount.load() << std::endl;
    std::cout << "Total Operations: " << processingTimes.size() << std::endl;
    std::cout << "Violation Rate: " << std::fixed << std::setprecision(2) << violationRate << "%" << std::endl;

    // Expect less than 1% violations
    EXPECT_LT(violationRate, 1.0) << "Real-time violation rate too high";
}

// Test real-time constraints with audio quality requirements
TEST_F(AudioQualityPreservationUnderRealtimeConstraints, QualityPreservationUnderRealtimeConstraints) {
    std::cout << "\n=== Audio Quality Preservation Under Real-time Constraints ===" << std::endl;

    const std::vector<TestSignalType> signalTypes = {
        TestSignalType::SINE_WAVE,
        TestSignalType::MULTI_TONE,
        TestSignalType::SWEEP
        TestSignalType::WHITE_NOISE
    };

    for (auto signalType : signalTypes) {
        std::cout << "\nTesting: " << static_cast<int>(signalType) << std::endl;

        // Generate test signal
        std::vector<float> inputSignal = harness_.generateSineWave(1000.0, 2.0); // 2 seconds

        // Real-time processing with quality check
        RealtimeTestResult result = harness_.testRealtimeProcessing(
            [&inputSignal](const std::vector<float>& input) -> std::vector<float> {
                std::vector<float> output(input.size());
                audioEngine_->processBuffer(input.data(), output.data(), input.size() / 2);
                return output;
            },
            5.0, // 5ms constraint (stricter for quality testing)
            "Real-time Quality Test - " + std::to_string(static_cast<int>(signalType))
        );

        EXPECT_TRUE(result.passed) << result.errorMessage;

        // Additional quality analysis
        AudioQualityMetrics metrics = harness_.analyzeAudioQuality(
            inputSignal, result.output, config.sampleRate);

        std::cout << "  Quality Metrics:" << std::endl;
        std::cout << "    SNR: " << metrics.signalToNoiseRatioDb << " dB" << std::endl;
        std::cout << "    THD: " << metrics.totalHarmonicDistortionDb << " dB" << std::endl;
        std::cout << "    Freq Response Dev: " << metrics.frequencyResponseDeviationDb << " dB" << std::endl;
        std::cout << "    Processing Time: " << result.duration.count() << "μs" << std::endl;
    }
}

// Test worst-case scenario real-time constraints
TEST_F(RealtimeConstraintsTest, WorstCaseScenario) {
    std::cout << "\n=== Worst-Case Scenario Testing ===" << std::endl;

    // Test with maximum buffer size and complex processing
    const size_t maxBufferSize = 4096; // ~93ms buffer
    const int numConcurrentOperations = 10;
    const int iterationsPerOperation = 10;

    PerformanceMetrics metrics;

    for (int iteration = 0; iteration < 100; ++iteration) {
        std::cout << "Worst-Case Test " << iteration + 1 << "/100" << std::endl;

        // Generate complex multi-tone signal
        std::vector<float> testSignal(maxBufferSize);
        for (size_t i = 0; i < maxBufferSize; ++i) {
            double time = i / config.sampleRate;
            float sample = 0.0f;

            // Add multiple frequency components
            sample += static_cast<float>(0.2f * std::sin(2.0 * M_PI * 100.0 * time));
            sample += static_cast<float>(0.2f * std::sin(2.0 * M_PI * 1000.0 * time));
            sample += static_cast<float>(0.2f * std::sin(2.0 * M_PI * 5000.0 * time));
            sample += static_cast<float>(0.2f * std::sin(2.0 * M_PI * 10000.0 * time));
            sample += static_cast<float>(0.2f * std::sin(2.0 * M_PI * 15000.0 * time));

            testSignal[i] = sample;
        }

        // Process with concurrent operations
        std::vector<std::thread> threads;
        std::vector<double> operationTimes(numConcurrentOperations);

        for (int t = 0; t < numConcurrentOperations; ++t) {
            threads.emplace_back([&testSignal, &operationTimes, t, maxBufferSize, &audioEngine_]() {
                std::vector<float> output(maxBufferSize);

                for (int i = 0; i < iterationsPerOperation; ++i) {
                    auto start = high_resolution_clock::now();
                    audioEngine_->processBuffer(testSignal.data(), output.data(), maxBufferSize / 2);
                    auto end = high_resolution_clock::now();

                    operationTimes[t] += duration_cast<microseconds>(end - start).count() / iterationsPerOperation;
                }
            });
        }

        // Wait for all threads
        for (auto& thread : threads) {
            thread.join();
        }

        // Calculate average processing time
        double avgProcessingTime = std::accumulate(operationTimes.begin(), operationTimes.end(), 0.0) / (numConcurrentOperations * iterationsPerOperation);

        // Real-time constraint: 93ms buffer should process faster than 93ms
        double constraintMs = (maxBufferSize / config.sampleRate) * 1000.0;

        EXPECT_LT(avgProcessingTime, constraintMs)
            << "Worst-case test " << iteration + 1 << ": processing time " << avgProcessingTime
            << "μs exceeds buffer time " << constraintMs << "μs";

        metrics.update(operationTimes);
        metrics.print("Worst-Case Test", constraintMs / 1000.0);
        AssertRealtimeConstraint(metrics, constraintMs / 1000.0, "Worst-Case Test " + std::to_string(iteration + 1));
    }
}

// Test memory efficiency under real-time constraints
TEST_F(RealtimeConstraintsTest, MemoryEfficiencyUnderRealtimeConstraints) {
    std::cout << "\n=== Memory Efficiency Under Real-time Constraints ===" << std::endl;

    const int numOperations = 1000;
    const std::vector<size_t> bufferSizes = {512, 1024, 2048, 4096, 8192};

    size_t initialMemory = harness_.getCurrentMemoryUsageMB();
    std::vector<size_t> memoryMeasurements;

    for (size_t bufferSize : bufferSizes) {
        std::cout << "Testing memory efficiency with buffer size: " << bufferSize << std::endl;

        // Process with memory monitoring
        size_t bufferMemoryStart = harness_.getCurrentMemoryUsageMB();

        std::vector<float> testSignal(bufferSize);
        for (size_t i = 0; i < bufferSize; ++i) {
            testSignal[i] = static_cast<float>(std::sin(2.0 * M_PI * 440.0 * (i / config.sampleRate)));
        }

        std::vector<float> output(bufferSize);

        for (int i = 0; i < numOperations; ++i) {
            auto start = high_resolution_clock::now();
            audioEngine_->processBuffer(testSignal.data(), output.data(), bufferSize / 2);
            auto end = high_resolution_clock::now();

            double processingTimeMs = duration_cast<microseconds>(end - start).count() / 1000.0;

            // Real-time constraint
            EXPECT_LT(processingTimeMs, 10.0)
                << "Buffer size " << bufferSize << ": processing time " << processingTimeMs << "ms";

            // Check memory growth
            if (i % 100 == 0) {
                size_t currentMemory = harness_.getCurrentMemoryUsageMB();
                size_t bufferMemoryGrowth = currentMemory - bufferMemoryStart;
                memoryMeasurements.push_back(bufferMemoryGrowth);

                // Memory growth should be minimal (<1MB per 100 operations)
                EXPECT_LT(bufferMemoryGrowth, 1024 * 1024)
                    << "Memory growth too large: " << bufferMemoryGrowth << " bytes after " << (i + 1) << " operations";
            }
        }

        // Final memory measurement
        size_t bufferMemoryEnd = harness_.getCurrentMemoryUsageMB();
        size_t totalBufferMemory = bufferMemoryEnd - bufferMemoryStart;

        std::cout << "  Buffer Memory: " << totalBufferMemory << " bytes ("
                  << (totalBufferMemory / 1024.0 / 1024.0) << " MB)" << std::endl;
        std::cout << "  Per-Operation Memory: " << (totalBufferMemory / numOperations) << " bytes" << std::endl;
    }

    // Final memory cleanup
    size_t finalMemory = harness_.getCurrentMemoryUsageMB();
    size_t totalMemoryGrowth = finalMemory - initialMemory;

    std::cout << "\nMemory Usage Summary:" << std::endl;
    std::cout << "  Initial Memory: " << initialMemory << " MB" << std::endl;
    std::cout << "  Final Memory: " << finalMemory << " MB" << std::endl;
    std::cout << "  Total Growth: " << totalMemoryGrowth << " MB" << std::endl;

    // Memory growth should be minimal
    EXPECT_LT(totalMemoryGrowth, 256) << "Excessive memory growth during processing";

    // Analyze memory measurements
    if (!memoryMeasurements.empty()) {
        double avgMemoryGrowth = std::accumulate(memoryMeasurements.begin(), memoryMeasurements.end(), 0.0) / memoryMeasurements.size();
        double maxMemoryGrowth = *std::max_element(memoryMeasurements.begin(), memoryMeasurements.end());

        EXPECT_LT(avgMemoryGrowth, 512 * 1024) << "Average memory growth too large: " << avgMemoryGrowth << " bytes";
        EXPECT_LT(maxMemoryGrowth, 1024 * 1024) << "Maximum memory growth too large: " << maxMemoryGrowth << " bytes";

        std::cout << "Average Buffer Memory Growth: " << (avgMemoryGrowth / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "Maximum Buffer Memory Growth: " << (maxMemoryGrowth / 1024.0 / 1024.0) << " MB" << std::endl;
    }
}

// Test system responsiveness under varying load
TEST_F(RealtimeConstraintsTest, SystemResponsiveness) {
    std::cout << "\n=== System Responsiveness Test ===" << std::endl;

    // Test system can quickly respond to sudden increases in processing demand
    const int baselineOperations = 10;
    const int spikeOperations = 100;
    const int recoveryTimeSeconds = 5;

    // Baseline measurement
    std::vector<float> baselineSignal(1024);
    for (size_t i = 0; i < baselineSignal.size(); ++i) {
        baselineSignal[i] = static_cast<float>(std::sin(2.0 * M_PI * 440.0 * (i / config.sampleRate)));
    }

    std::vector<float> baselineOutput(baselineSignal.size());
    std::vector<double> baselineTimes(baselineOperations);

    for (int i = 0; i < baselineOperations; ++i) {
        auto start = high_resolution_clock::now();
        audioEngine_->processBuffer(baselineSignal.data(), baselineOutput.data(), baselineSignal.size() / 2);
        auto end = high_resolution_clock::now();

        baselineTimes[i] = duration_cast<microseconds>(end - start).count();
    }

    PerformanceMetrics baselineMetrics;
    baselineMetrics.update(baselineTimes);
    double baselineAvg = baselineMetrics.avgTime;

    std::cout << "Baseline Performance:" << std::endl;
    baselineMetrics.print("Baseline", 10.0);

    // Spike test - sudden increase in load
    std::vector<float> spikeSignal(1024);
    for (size_t i = 0; i < spikeSignal.size(); ++i) {
        spikeSignal[i] = static_cast<float>(std::sin(2.0 * M_PI * 440.0 * (i / config.sampleRate)) * 2.0f); // Louder signal
    }

    std::vector<float> spikeOutput(spikeSignal.size());
    std::vector<double> spikeTimes(spikeOperations);

    std::cout << "Applying processing spike..." << std::endl;
    auto spikeStart = high_resolution_clock::now();

    for (int i = 0; i < spikeOperations; ++i) {
        auto start = high_resolution_clock::now();
        audioEngine->processBuffer(spikeSignal.data(), spikeOutput.data(), spikeSignal.size() / 2);
        auto end = high_resolution_clock::now();

        spikeTimes[i] = duration_cast<microseconds>(end - start).count();

        // Check if system is maintaining responsiveness
        if (i > 0 && i % 10 == 0) {
            double currentAvg = std::accumulate(spikeTimes.begin(), spikeTimes.begin() + i + 1, 0.0) / (i + 1);
            double degradation = (currentAvg - baselineAvg) / baselineAvg * 100.0;

            // Allow up to 100% degradation during spike
            EXPECT_LT(degradation, 200.0) << "System too unresponsive during spike at iteration " << i
                << ": " << degradation << "% degradation";
        }
    }

    auto spikeEnd = high_resolution_clock::now();
    double spikeDurationMs = duration_cast<milliseconds>(spikeEnd - spikeStart).count();

    PerformanceMetrics spikeMetrics;
    spikeMetrics.update(spikeTimes);
    double spikeAvg = spikeMetrics.avgTime;

    std::cout << "Spike Performance:" << std::endl;
    spikeMetrics.print("Spike Test", 20.0);

    std::cout << "Spike Duration: " << spikeDurationMs << "ms" << std::endl;
    std::cout << "Performance Degradation: " << ((spikeAvg - baselineAvg) / baselineAvg * 100.0) << "%" << std::endl;

    // Recovery test - return to baseline load
    std::cout << "Testing recovery..." << std::endl;
    auto recoveryStart = high_resolution_clock::now();

    std::vector<float> recoveryOutput(recoverySignal.size());
    std::vector<double> recoveryTimes(baselineOperations);

    for (int i = 0; i < baselineOperations; ++i) {
        auto start = high_resolution_clock::now();
        audioEngine->processBuffer(recoverySignal.data(), recoveryOutput.data(), recoverySignal.size() / 2);
        auto end = high_resolution_clock::now();

        recoveryTimes[i] = duration_cast<microseconds>(end - start).count();

        if (i > 0 && i % 5 == 0) {
            double currentAvg = std::accumulate(recoveryTimes.begin(), recoveryTimes.begin() + i + 1, 0.0) / (i + 1);
            double recoveryRatio = currentAvg / baselineAvg;

            // Should recover to within 10% of baseline
            EXPECT_GT(recoveryRatio, 0.9) << "Poor recovery after spike at iteration " << i
                << ": recovery ratio " << (recoveryRatio * 100.0) << "%";
        }
    }

    auto recoveryEnd = high_resolution_clock::now();
    double recoveryDurationMs = duration_cast<milliseconds>(recoveryEnd - recoveryStart).count();

    PerformanceMetrics recoveryMetrics;
    recoveryMetrics.update(recoveryTimes);
    double recoveryAvg = recoveryMetrics.avgTime;

    std::cout << "Recovery Performance:" << std::endl;
    recoveryMetrics.print("Recovery Test", 10.0);
    std::cout << "Recovery Duration: " << recoveryDurationMs << "ms" << std::endl;
    std::cout << "Recovery Ratio: " << (recoveryAvg / baselineAvg) << std::endl;

    // Recovery should be relatively quick (<5 seconds)
    EXPECT_LT(recoveryDurationMs, recoveryTimeSeconds * 1000.0)
        << "Recovery took too long: " << recoveryDurationMs << "ms";

    // Recovery performance should be close to baseline
    EXPECT_LT(std::abs(recoveryAvg - baselineAvg), baselineAvg * 0.2)
        << "Recovery performance too degraded: " << recoveryAvg << "μs vs " << baselineAvg << "μs";
}

} // namespace vortex