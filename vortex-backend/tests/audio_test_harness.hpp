#pragma once

#include <vector>
#include <complex>
#include <memory>
#include <functional>
#include <chrono>
#include <string>
#include <map>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_dsp/juce_dsp.h>

namespace vortex::testing {

/**
 * @brief Audio processing test harness for TDD-driven audio algorithm testing
 *
 * This test harness provides comprehensive testing capabilities for audio processing
 * components with real-time constraints validation and audio quality measurements.
 */
class AudioTestHarness {
public:
    struct TestConfiguration {
        // Audio parameters
        double sampleRate = 44100.0;
        int bufferSize = 512;
        int channels = 2;
        int bitDepth = 24;

        // Real-time constraints
        double maxProcessingTimeMs = 10.0;
        double maxMemoryUsageMB = 512.0;

        // Quality thresholds
        double maxSignalToNoiseRatioDb = -80.0;
        double maxTotalHarmonicDistortionDb = -100.0;
        double maxFrequencyResponseDeviationDb = 0.1;
        double maxPhaseDeviationDeg = 1.0;

        // Test parameters
        int numTestFrequencies = 10;
        double minTestFrequency = 20.0;
        double maxTestFrequency = 20000.0;

        // GPU test flags
        bool enableGPUTests = false;
        std::string gpuBackend = "CUDA"; // CUDA, OpenCL, Vulkan
    };

    struct AudioQualityMetrics {
        double signalToNoiseRatioDb = 0.0;
        double totalHarmonicDistortionDb = 0.0;
        double frequencyResponseDeviationDb = 0.0;
        double phaseDeviationDeg = 0.0;
        double peakSignalDb = 0.0;
        double rmsSignalDb = 0.0;
        double dynamicRangeDb = 0.0;
    };

    struct PerformanceMetrics {
        double processingTimeMs = 0.0;
        double memoryUsageMB = 0.0;
        double cpuUtilization = 0.0;
        double gpuUtilization = 0.0;
        double throughputSamplesPerSecond = 0.0;
    };

    struct TestResult {
        bool passed = false;
        std::string testName;
        std::string errorMessage;
        AudioQualityMetrics quality;
        PerformanceMetrics performance;
        std::chrono::microseconds duration{0};
    };

    using AudioProcessor = std::function<std::vector<float>(const std::vector<float>&)>;

public:
    AudioTestHarness();
    explicit AudioTestHarness(const TestConfiguration& config);
    ~AudioTestHarness();

    // Configuration
    void setConfiguration(const TestConfiguration& config) { config_ = config; }
    const TestConfiguration& getConfiguration() const { return config_; }

    // Test signal generation
    std::vector<float> generateSineWave(double frequency, double durationSec, double amplitude = 1.0) const;
    std::vector<float> generateWhiteNoise(double durationSec, double amplitude = 1.0) const;
    std::vector<float> generatePinkNoise(double durationSec, double amplitude = 1.0) const;
    std::vector<float> generateSweep(double startFreq, double endFreq, double durationSec, double amplitude = 1.0) const;
    std::vector<float> generateImpulseTrain(double frequency, double durationSec, double amplitude = 1.0) const;
    std::vector<float> generateMultiTone(const std::vector<double>& frequencies, double durationSec, double amplitude = 1.0) const;

    // Audio analysis tools
    AudioQualityMetrics analyzeAudioQuality(const std::vector<float>& input, const std::vector<float>& output, double sampleRate) const;
    std::vector<std::complex<double>> computeFFT(const std::vector<float>& audio, double sampleRate) const;
    std::vector<double> computeSpectrum(const std::vector<float>& audio, double sampleRate) const;
    double computeTHD(const std::vector<float>& audio, double fundamentalFrequency, double sampleRate) const;
    double computeSNR(const std::vector<float>& signal, const std::vector<float>& noise) const;
    std::vector<double> computeFrequencyResponse(const std::vector<float>& input, const std::vector<float>& output, double sampleRate) const;

    // Test execution
    TestResult testProcessorLatency(AudioProcessor processor, double targetLatencyMs = 10.0) const;
    TestResult testAudioQuality(AudioProcessor processor, double testFrequency = 1000.0) const;
    TestResult testFrequencyResponse(AudioProcessor processor, const std::vector<double>& testFrequencies = {}) const;
    TestResult testDynamicRange(AudioProcessor processor, double inputAmplitude = 1.0) const;
    TestResult testPhaseLinearity(AudioProcessor processor, const std::vector<double>& testFrequencies = {}) const;
    TestResult testMemoryUsage(AudioProcessor processor, double maxMemoryMB = 512.0) const;
    TestResult testConcurrentProcessing(AudioProcessor processor, int numConcurrentStreams = 4) const;
    TestResult testLongTermStability(AudioProcessor processor, double durationSec = 60.0) const;

    // GPU-specific tests
    TestResult testGPUAcceleration(AudioProcessor cpuProcessor, AudioProcessor gpuProcessor, const std::string& backend) const;
    TestResult testGPUMemoryTransfer(AudioProcessor processor, size_t dataSize = 1024*1024) const;
    TestResult testGPUConcurrency(AudioProcessor processor, int numConcurrentKernels = 8) const;

    // Specialized audio processing tests
    TestResult testEqualizer(std::function<std::vector<float>(const std::vector<float>&, const std::vector<double>&, const std::vector<double>&)> eq,
                             const std::vector<double>& frequencies, const std::vector<double>& gains) const;
    TestResult testConvolution(AudioProcessor processor, const std::vector<float>& impulseResponse) const;
    TestResult testResampling(AudioProcessor processor, double inputSampleRate, double outputSampleRate) const;
    TestResult testDSDProcessing(AudioProcessor processor, int dsdRate) const;

    // Real-time constraint validation
    bool validateRealtimeConstraint(std::function<void()> operation, double maxTimeMs) const;
    bool validateMemoryConstraint(std::function<void()> operation, double maxMemoryMB) const;

    // Test data export/import
    bool exportTestData(const std::vector<float>& audio, const std::string& filename, double sampleRate) const;
    std::vector<float> importTestData(const std::string& filename) const;
    void exportTestResults(const std::vector<TestResult>& results, const std::string& filename) const;

    // Benchmarking
    PerformanceMetrics benchmarkProcessor(AudioProcessor processor, double durationSec = 10.0) const;
    std::map<std::string, PerformanceMetrics> benchmarkMultipleProcessors(
        const std::map<std::string, AudioProcessor>& processors, double durationSec = 10.0) const;

    // Assertion helpers for testing frameworks
    static void assertAudioQuality(const AudioQualityMetrics& metrics, const TestConfiguration& config, const std::string& testName = "");
    static void assertPerformance(const PerformanceMetrics& metrics, const TestConfiguration& config, const std::string& testName = "");
    static void assertRealtimeConstraint(double actualTimeMs, double maxTimeMs, const std::string& operationName = "");

private:
    TestConfiguration config_;
    mutable std::mt19937 rng_;

    // Helper methods
    std::vector<double> generateFrequencyVector(double minFreq, double maxFreq, int numFrequencies, bool logarithmic = true) const;
    std::vector<std::complex<double>> fft(const std::vector<float>& input) const;
    double dbToLinear(double db) const { return std::pow(10.0, db / 20.0); }
    double linearToDb(double linear) const { return 20.0 * std::log10(std::abs(linear) + 1e-10); }
    double getCurrentMemoryUsageMB() const;
    double getCPUTimeMs() const;

    // Template test generators
    template<typename Func>
    TestResult runTimedTest(Func&& testFunc, const std::string& testName) const;

    // Audio file format support
    juce::AudioFormatManager formatManager_;
};

/**
 * @brief RAII timer for measuring execution time
 */
class ScopedTimer {
public:
    explicit ScopedTimer(std::chrono::microseconds& result) : result_(result), start_(std::chrono::high_resolution_clock::now()) {}
    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        result_ = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
    }

private:
    std::chrono::microseconds& result_;
    std::chrono::high_resolution_clock::time_point start_;
};

/**
 * @brief RAII memory tracker for measuring memory usage
 */
class ScopedMemoryTracker {
public:
    explicit ScopedMemoryTracker(double& initialMemory, double& peakMemory)
        : initialMemory_(initialMemory), peakMemory_(peakMemory) {
        initialMemory_ = getCurrentMemoryUsageMB();
        peakMemory_ = initialMemory_;
    }

    ~ScopedMemoryTracker() {
        double current = getCurrentMemoryUsageMB();
        peakMemory_ = std::max(peakMemory_, current);
    }

private:
    double& initialMemory_;
    double& peakMemory_;
    static double getCurrentMemoryUsageMB();
};

/**
 * @brief Convenience macros for audio testing
 */
#define EXPECT_AUDIO_QUALITY(metrics, config) \
    vortex::testing::AudioTestHarness::assertAudioQuality(metrics, config, ::testing::UnitTest::GetInstance()->current_test_info()->name())

#define EXPECT_PERFORMANCE(metrics, config) \
    vortex::testing::AudioTestHarness::assertPerformance(metrics, config, ::testing::UnitTest::GetInstance()->current_test_info()->name())

#define EXPECT_REALTIME_CONSTRAINT(actual_ms, max_ms, operation) \
    vortex::testing::AudioTestHarness::assertRealtimeConstraint(actual_ms, max_ms, operation)

#define AUDIO_PROCESSOR_TEST(processor, test_name, config) \
    EXPECT_AUDIO_QUALITY(harness.testAudioQuality(processor).quality, config) << " in " << test_name; \
    EXPECT_PERFORMANCE(harness.testProcessorLatency(processor).performance, config) << " in " << test_name

} // namespace vortex::testing