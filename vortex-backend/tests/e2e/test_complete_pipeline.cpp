#include <gtest/gtest.h>
#include "core/audio_engine.hpp"
#include "core/fileio/audio_file_loader.hpp"
#include "core/fileio/format_detector.hpp"
#include "core/gpu/gpu_processor.hpp"
#include "testing/audio_test_harness.hpp"
#include "system/logger.hpp"

#include <thread>
#include <chrono>
#include <fstream>
#include <random>

using namespace vortex;
using namespace vortex::testing;
using namespace std::chrono_literals;

class CompletePipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::initialize();

        // Configure test harness
        AudioTestHarness::TestConfiguration config;
        config.sampleRate = 44100.0;
        config.bufferSize = 1024;
        config.channels = 2;
        config.bitDepth = 24;
        config.maxProcessingTimeMs = 10.0; // 10ms real-time constraint
        config.maxSignalToNoiseRatioDb = -80.0;
        config.maxTotalHarmonicDistortionDb = -100.0;
        config.enableGPUTests = true;

        harness_.setConfiguration(config);

        // Initialize components
        InitializeAudioEngine();
        InitializeFileLoader();
        InitializeGPUProcessor();
    }

    void TearDown() override {
        ShutdownGPUProcessor();
        ShutdownFileLoader();
        ShutdownAudioEngine();
        Logger::shutdown();
    }

    void InitializeAudioEngine() {
        audioEngine_ = std::make_unique<AudioEngine>();
        ASSERT_TRUE(audioEngine_->initialize(config.sampleRate, config.bufferSize));
        EXPECT_TRUE(audioEngine_->isInitialized());
    }

    void InitializeFileLoader() {
        fileLoader_ = std::make_unique<AudioFileLoader>();
        fileLoader_->setCacheSize(100 * 1024 * 1024); // 100MB cache
        fileLoader_->enableCache(true);
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
            std::cout << "GPU not available, tests will run CPU-only" << std::endl;
        }
    }

    void ShutdownAudioEngine() {
        if (audioEngine_) {
            audioEngine_->shutdown();
        }
    }

    void ShutdownFileLoader() {
        if (fileLoader_) {
            fileLoader_->clearCache();
        }
    }

    void ShutdownGPUProcessor() {
        if (gpuProcessor_) {
            gpuProcessor_->shutdown();
        }
    }

    // Create test audio file
    std::string CreateTestAudioFile(const std::string& filename, double durationSeconds = 1.0) {
        std::vector<float> audioData;
        int numSamples = static_cast<int>(config.sampleRate * durationSeconds);
        audioData.resize(numSamples);

        // Generate multi-tone test signal
        std::vector<double> frequencies = {100.0, 440.0, 1000.0, 5000.0}; // Low, Mid, High, Ultra-high

        for (int sample = 0; sample < numSamples; ++sample) {
            double time = sample / config.sampleRate;
            float sampleValue = 0.0f;

            for (double freq : frequencies) {
                sampleValue += static_cast<float>(0.25f * std::sin(2.0 * M_PI * freq * time));
            }

            // Apply gentle envelope
            double envelope = 0.5 * (1.0 - std::cos(2.0 * M_PI * time / durationSeconds));
            sampleValue *= static_cast<float>(envelope);

            audioData[sample] = sampleValue;
        }

        // Write as WAV file
        juce::WavAudioFormat format;
        juce::File file(filename);
        std::unique_ptr<juce::AudioFormatWriter> writer;

        writer.reset(format.createWriterFor(new juce::FileOutputStream(file), 44100.0, 2, 24));

        if (writer != nullptr) {
            juce::AudioBuffer<float> buffer(2, numSamples);
            for (int channel = 0; channel < 2; ++channel) {
                buffer.copyFrom(0, channel, numSamples, audioData.data(), 1, 2);
            }

            writer->writeFromAudioBuffer(buffer, 0, numSamples);
        }

        return filename;
    }

    // Generate comprehensive test signal
    std::vector<float> GenerateTestSignal(TestSignalType type, double durationSeconds = 2.0) {
        int numSamples = static_cast<int>(config.sampleRate * durationSeconds);
        std::vector<float> signal(numSamples);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> noiseDist(-0.1f, 0.1f); // Small noise floor

        switch (type) {
            case TestSignalType::SINE_WAVE: {
                for (int i = 0; i < numSamples; ++i) {
                    double time = i / config.sampleRate;
                    signal[i] = static_cast<float>(std::sin(2.0 * M_PI * 440.0 * time)) + noiseDist(gen);
                }
                break;
            }

            case TestSignalType::MULTI_TONE: {
                std::vector<double> frequencies = {100.0, 440.0, 1000.0, 3520.0, 8000.0};
                for (int i = 0; i < numSamples; ++i) {
                    double time = i / config.sampleRate;
                    float sampleValue = 0.0f;
                    for (double freq : frequencies) {
                        sampleValue += static_cast<float>(std::sin(2.0 * M_PI * freq * time));
                    }
                    signal[i] = (sampleValue / frequencies.size()) + noiseDist(gen);
                }
                break;
            }

            case TestSignalType::WHITE_NOISE: {
                std::uniform_real_distribution<float> signalDist(-1.0f, 1.0f);
                for (int i = 0; i < numSamples; ++i) {
                    signal[i] = signalDist(gen);
                }
                break;
            }

            case TestSignalType::PINK_NOISE: {
                // Simple pink noise approximation
                for (int i = 0; i < numSamples; ++i) {
                    double value = 0.0;
                    double weight = 0.0;
                    for (int octave = 0; octave < 6; ++octave) {
                        double factor = 1.0 / (1 << octave);
                        if (i % (1 << octave) == 0) {
                            double noise = (gen() % 2000000) / 1000000.0 - 1.0; // -1 to 1
                            value += noise * factor;
                            weight += factor;
                        }
                    }
                    signal[i] = static_cast<float>(value / weight) + noiseDist(gen);
                }
                break;
            }

            case TestSignalType::SWEEP: {
                for (int i = 0; i < numSamples; ++i) {
                    double time = i / config.sampleRate;
                    double frequency = 20.0 * std::pow(20000.0 / 20.0, time / durationSeconds);
                    signal[i] = static_cast<float>(std::sin(2.0 * M_PI * frequency * time)) + noiseDist(gen);
                }
                break;
            }
        }

        return signal;
    }

    enum class TestSignalType {
        SINE_WAVE,
        MULTI_TONE,
        WHITE_NOISE,
        PINK_NOISE,
        SWEEP
    };

    AudioTestHarness::TestConfiguration config;
    std::unique_ptr<AudioEngine> audioEngine_;
    std::unique_ptr<AudioFileLoader> fileLoader_;
    std::unique_ptr<GPUProcessor> gpuProcessor_;
    bool gpuAvailable_ = false;
};

// Test complete audio file loading and processing pipeline
TEST_F(CompletePipelineTest, AudioFileProcessingPipeline) {
    const std::string testFile = "test_complete_pipeline.wav";

    // Create test audio file
    CreateTestAudioFile(testFile, 2.0);

    // Load audio file
    auto start = high_resolution_clock::now();
    LoadResult loadResult = fileLoader_->loadAudioFile(testFile);
    auto end = high_resolution_clock::now();

    ASSERT_TRUE(loadResult.success) << "Failed to load audio file: " << loadResult.error;
    ASSERT_FALSE(loadResult.fromCache) << "First load should not be from cache";
    ASSERT_EQ(loadResult.metadata.format, AudioFormat::WAV);
    EXPECT_GT(loadResult.audioData.numSamples, 0);

    auto loadTime = duration_cast<milliseconds>(end - start);

    std::cout << "Audio file loaded successfully:" << std::endl;
    std::cout << "  Duration: " << loadResult.audioData.durationSeconds << " seconds" << std::endl;
    std::cout << "  Channels: " << loadResult.audioData.channels << std::endl;
    std::cout << "  Sample Rate: " << loadResult.audioData.sampleRate << " Hz" << std::endl;
    std::cout << "  Load Time: " << loadTime.count() << " ms" << std::endl;

    // Verify audio quality
    auto reloadedResult = fileLoader_->loadAudioFile(testFile);
    ASSERT_TRUE(reloadedResult.success);
    EXPECT_TRUE(reloadedResult.fromCache) << "Second load should be from cache";

    // Compare loaded data
    ASSERT_EQ(loadResult.audioData.data.size(), reloadedResult.audioData.data.size());

    float maxDifference = 0.0f;
    for (size_t i = 0; i < loadResult.audioData.data.size(); ++i) {
        float diff = std::abs(loadResult.audioData.data[i] - reloadedResult.audioData.data[i]);
        maxDifference = std::max(maxDifference, diff);
    }

    EXPECT_LT(maxDifference, 1e-6f) << "Cached data should match original data exactly";

    // Load audio data into engine
    ASSERT_TRUE(audioEngine_->loadAudioFile(testFile));

    // Process through audio engine
    std::vector<float> processedSignal(loadResult.audioData.data.size());
    audioEngine_->processBuffer(loadResult.audioData.data(), processedSignal.data(),
                              loadResult.audioData.numSamples);

    // Verify processing didn't crash and produced reasonable output
    EXPECT_FALSE(processedSignal.empty());

    // Basic quality check
    float sum = 0.0f;
    for (float sample : processedSignal) {
        sum += std::abs(sample);
    }
    float rms = sum / processedSignal.size();

    EXPECT_GT(rms, 0.0001f) << "Output should have some audio content";
    EXPECT_LT(rms, 10.0f) << "Output should not be clipped";

    std::cout << "Pipeline processing completed successfully" << std::endl;
}

// Test real-time performance constraints
TEST_F(CompletePipelineTest, RealTimePerformanceConstraints) {
    const int numIterations = 1000;
    const double maxProcessingTimeMs = 10.0; // 10ms constraint

    // Generate test signal
    auto testSignal = GenerateTestSignal(TestSignalType::MULTI_TONE, 0.1); // 100ms buffer
    std::vector<float> outputSignal(testSignal.size());

    // Measure processing time with all components
    std::vector<double> processingTimes;
    processingTimes.reserve(numIterations);

    for (int i = 0; i < numIterations; ++i) {
        auto start = high_resolution_clock::now();

        // Process through complete pipeline
        audioEngine_->processBuffer(testSignal.data(), outputSignal.data(), testSignal.size() / 2);

        auto end = high_resolution_clock::now();
        double processingTimeMs = duration_cast<microseconds>(end - start).count() / 1000.0;

        processingTimes.push_back(processingTimeMs);

        // Real-time constraint check
        EXPECT_LT(processingTimeMs, maxProcessingTimeMs)
            << "Real-time constraint violated: " << processingTimeMs << "ms > " << maxProcessingTimeMs << "ms (iteration " << i << ")";
    }

    // Analyze performance statistics
    auto [minIt, maxIt] = std::minmax_element(processingTimes.begin(), processingTimes.end());
    double avgTime = std::accumulate(processingTimes.begin(), processingTimes.end(), 0.0) / processingTimes.size();

    // Calculate standard deviation
    double variance = 0.0;
    for (double time : processingTimes) {
        variance += (time - avgTime) * (time - avgTime);
    }
    double stdDev = std::sqrt(variance / processingTimes.size());

    // Performance expectations
    EXPECT_LT(avgTime, maxProcessingTimeMs * 0.8) << "Average processing time should be well below constraint";
    EXPECT_LT(maxIt->count(), maxProcessingTimeMs) << "Maximum processing time should not exceed constraint";
    EXPECT_LT(stdDev, maxProcessingTimeMs * 0.2) << "Processing time should be consistent";

    std::cout << "Real-time performance analysis:" << std::endl;
    std::cout << "  Average: " << avgTime << "ms" << std::endl;
    std::cout << "  Minimum: " << minIt->count() << "ms" << std::endl;
    std::cout << "  Maximum: " << maxIt->count() << "ms" << std::endl;
    std::cout << "  Std Dev: " << stdDev << "ms" << std::endl;
    std::cout << "  Constraint: " << maxProcessingTimeMs << "ms" << std::endl;
}

// Test GPU acceleration vs CPU processing
TEST_F(CompletePipelineTest, GPUAccelerationComparison) {
    if (!gpuAvailable_) {
        GTEST_SKIP() << "GPU not available, skipping GPU acceleration tests";
    }

    // Generate test signals of different sizes
    std::vector<size_t> signalSizes = {
        1024,    // 23ms
        4096,    // 93ms
        16384,   // 371ms
        65536,   // 1.49s
        262144   // 5.95s
    };

    for (size_t signalSize : signalSizes) {
        auto testSignal = GenerateTestSignal(TestSignalType::SINE_WAVE, signalSize / config.sampleRate);
        std::vector<float> gpuOutput(testSignal.size());
        std::vector<float> cpuOutput(testSignal.size());

        // Test GPU processing
        auto gpuStart = high_resolution_clock::now();
        audioEngine_->processBuffer(testSignal.data(), gpuOutput.data(), testSignal.size() / 2);
        auto gpuEnd = high_resolution_clock::now();

        // Disable GPU for CPU comparison
        audioEngine_->disableGPUAcceleration();

        // Test CPU processing
        auto cpuStart = high_resolution_clock::now();
        audioEngine_->processBuffer(testSignal.data(), cpuOutput.data(), testSignal.size() / 2);
        auto cpuEnd = high_resolution_clock::now();

        // Re-enable GPU
        audioEngine_->enableGPUAcceleration("CUDA");

        auto gpuTime = duration_cast<microseconds>(gpuEnd - gpuStart);
        auto cpuTime = duration_cast<microseconds>(cpuEnd - cpuStart);

        double speedup = static_cast<double>(cpuTime.count()) / gpuTime.count();

        // Verify results are similar (allow small numerical differences)
        float maxDifference = 0.0f;
        for (size_t i = 0; i < testSignal.size(); ++i) {
            float diff = std::abs(gpuOutput[i] - cpuOutput[i]);
            maxDifference = std::max(maxDifference, diff);
        }

        EXPECT_LT(maxDifference, 1e-4f) << "GPU and CPU results differ too much for size " << signalSize;

        // For larger buffers, GPU should be faster
        if (signalSize >= 16384) { // >= 371ms buffer
            EXPECT_GT(speedup, 1.0) << "GPU should be faster for size " << signalSize
                << " (GPU: " << gpuTime.count() << "μs, CPU: " << cpuTime.count() << "μs)";
        }

        std::cout << "Signal Size: " << signalSize << " samples (" << (signalSize / config.sampleRate) << "s)" << std::endl;
        std::cout << "  GPU Time: " << gpuTime.count() << "μs" << std::endl;
        std::cout << "  CPU Time: " << cpuTime.count() << "μs" << std::endl;
        std::cout << "  Speedup: " << speedup << "x" << std::endl;
    }
}

// Test memory usage and efficiency
TEST_F(CompletePipelineTest, MemoryUsageEfficiency) {
    const int numFiles = 50;
    const double durationSeconds = 2.0;

    size_t initialMemory = harness_.getCurrentMemoryUsageMB();

    // Process multiple audio files sequentially
    std::vector<std::string> testFiles;
    for (int i = 0; i < numFiles; ++i) {
        std::string filename = "memory_test_" + std::to_string(i) + ".wav";
        CreateTestAudioFile(filename, durationSeconds);
        testFiles.push_back(filename);

        // Load and process
        auto loadResult = fileLoader_->loadAudioFile(filename);
        ASSERT_TRUE(loadResult.success);

        audioEngine_->loadAudioFile(filename);

        // Process audio data
        std::vector<float> output(loadResult.audioData.data.size());
        audioEngine_->processBuffer(loadResult.audioData.data(), output.data(),
                                  loadResult.audioData.numSamples);

        // Check memory usage every 10 files
        if ((i + 1) % 10 == 0) {
            size_t currentMemory = harness_.getCurrentMemoryUsageMB();
            size_t memoryGrowth = currentMemory - initialMemory;

            // Memory growth should be reasonable (<200MB for 50 files)
            EXPECT_LT(memoryGrowth, 200) << "Memory leak detected after " << (i + 1) << " files: " << memoryGrowth << " MB";
        }
    }

    size_t finalMemory = harness_.getCurrentMemoryUsageMB();
    size_t totalMemoryGrowth = finalMemory - initialMemory;

    // Cleanup test files
    for (const auto& file : testFiles) {
        std::remove(file.c_str());
    }

    // Total memory growth should be minimal (<300MB for 50 files)
    EXPECT_LT(totalMemoryGrowth, 300) << "Excessive memory growth: " << totalMemoryGrowth << " MB";

    // Clear cache and check memory cleanup
    fileLoader_->clearCache();
    audioEngine_->shutdown();
    audioEngine_.reset(new AudioEngine());
    audioEngine_->initialize(config.sampleRate, config.bufferSize);

    size_t cleanupMemory = harness_.getCurrentMemoryUsageMB();
    size_t memoryAfterCleanup = initialMemory > cleanupMemory ? (initialMemory - cleanupMemory) : 0;

    EXPECT_LT(memoryAfterCleanup, 50) << "Memory not properly cleaned up: " << memoryAfterCleanup << " MB";

    std::cout << "Memory Usage Analysis:" << std::endl;
    std::cout << "  Initial Memory: " << initialMemory << " MB" << std::endl;
    std::cout << "  Final Memory: " << finalMemory << " MB" << std::endl;
    std::cout << "  Total Growth: " << totalMemoryGrowth << " MB" << std::endl;
    std::cout << "  After Cleanup: " << cleanupMemory << " MB" << std::endl;
}

// Test audio quality preservation through processing pipeline
TEST_F(CompletePipelineTest, AudioQualityPreservation) {
    // Generate test signals
    std::vector<std::pair<std::string, std::vector<float>>> testSignals;

    testSignals.emplace_back("Sine Wave", GenerateTestSignal(TestSignalType::SINE_WAVE, 1.0));
    testSignals.emplace_back("Multi Tone", GenerateTestSignal(TestSignalType::MULTI_TONE, 1.0));
    testSignals.emplace_back("White Noise", GenerateTestSignal(TestSignalType::WHITE_NOISE, 1.0));
    testSignals.emplace_back("Pink Noise", GenerateTestSignal(TestSignalType::PINK_NOISE, 1.0));
    testSignals.emplace_back("Frequency Sweep", GenerateTestSignal(TestSignalType::SWEEP, 1.0));

    for (const auto& [signalName, inputSignal] : testSignals) {
        std::cout << "Testing audio quality for: " << signalName << std::endl;

        // Create WAV file
        std::string filename = "quality_test_" + signalName + ".wav";
        CreateTestAudioFile(filename, 1.0);

        // Load and process
        auto loadResult = fileLoader_->loadAudioFile(filename);
        ASSERT_TRUE(loadResult.success);

        audioEngine_->loadAudioFile(filename);

        std::vector<float> outputSignal(inputSignal.size());
        audioEngine_->processBuffer(loadResult.audioData.data(), outputSignal.data(),
                                  loadResult.audioData.numSamples);

        // Analyze audio quality
        AudioQualityMetrics metrics = harness_.analyzeAudioQuality(
            inputSignal, outputSignal, config.sampleRate);

        // Quality expectations
        EXPECT_GT(metrics.signalToNoiseRatioDb, 70.0) << signalName << ": SNR too low";
        EXPECT_LT(metrics.totalHarmonicDistortionDb, -60.0) << signalName << ": THD too high";
        EXPECT_LT(metrics.frequencyResponseDeviationDb, 0.5) << signalName << ": Frequency response deviation too high";

        // Report results
        std::cout << "  Signal-to-Noise Ratio: " << metrics.signalToNoiseRatioDb << " dB" << std::endl;
        std::cout << "  Total Harmonic Distortion: " << metrics.totalHarmonicDistortionDb << " dB" << std::endl;
        std::cout << "  Frequency Response Deviation: " << metrics.frequencyResponseDeviationDb << " dB" << std::endl;
        std::cout << "  Peak Signal: " << metrics.peakSignalDb << " dB" << std::endl;
        std::cout << "  RMS Signal: " << metrics.rmsSignalDb << " dB" << std::endl;

        // Cleanup
        std::remove(filename.c_str());
    }
}

// Test concurrent processing and thread safety
TEST_F(CompletePipelineTest, ConcurrentProcessing) {
    const int numThreads = 4;
    const int iterationsPerThread = 100;

    // Create test signal
    auto testSignal = GenerateTestSignal(TestSignalType::MULTI_TONE, 2.0);

    std::vector<std::thread> threads;
    std::vector<std::vector<double>> threadTimes(numThreads);
    std::vector<std::mutex> timeMutexes(numThreads);

    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t, &testSignal]() {
            for (int i = 0; i < iterationsPerThread; ++i) {
                std::vector<float> output(testSignal.size());

                auto start = high_resolution_clock::now();

                // Load and process audio file
                std::string filename = "concurrent_test_" + std::to_string(t) + "_" + std::to_string(i) + ".wav";
                CreateTestAudioFile(filename, 0.01); // 10ms file

                auto loadResult = fileLoader_->loadAudioFile(filename);
                ASSERT_TRUE(loadResult.success);

                audioEngine_->loadAudioFile(filename);

                audioEngine_->processBuffer(loadResult.audioData.data(), output.data(),
                                          loadResult.audioData.numSamples);

                auto end = high_resolution_clock::now();

                double processingTimeMs = duration_cast<microseconds>(end - start).count() / 1000.0;

                {
                    std::lock_guard<std::mutex> lock(timeMutexes[t]);
                    threadTimes[t].push_back(processingTimeMs);
                }

                // Real-time constraint
                EXPECT_LT(processingTimeMs, 10.0)
                    << "Real-time constraint violated in thread " << t << ", iteration " << i;

                // Cleanup
                std::remove(filename.c_str());
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Analyze concurrent performance
    size_t totalOperations = numThreads * iterationsPerThread;
    std::vector<double> allTimes;

    for (const auto& threadTimes : threadTimes) {
        allTimes.insert(allTimes.end(), threadTimes.begin(), threadTimes.end());
    }

    ASSERT_EQ(allTimes.size(), totalOperations);

    auto [minTime, maxTime] = std::minmax_element(allTimes.begin(), allTimes.end());
    double avgTime = std::accumulate(allTimes.begin(), allTimes.end(), 0.0) / allTimes.size();

    // Concurrent processing should maintain performance
    EXPECT_LT(avgTime, 5.0) << "Average concurrent processing time too high";
    EXPECT_LT(maxTime->count(), 10.0) << "Maximum concurrent processing time too high";
    EXPECT_LT((maxTime->count() - minTime->count()) / avgTime, 5.0) << "Processing time variance too high";

    std::cout << "Concurrent Processing Analysis:" << std::std::endl;
    std::cout << "  Total Operations: " << totalOperations << std::endl;
    std::cout << "  Average Time: " << avgTime << "ms" << std::endl;
    std::cout << "  Minimum Time: " << minTime->count() << "ms" << std::endl;
    std::cout << "  Maximum Time: " << maxTime->count() << "ms" << std::endl;
}

// Test system stability under load
TEST_F(CompletePipelineTest, SystemStabilityUnderLoad) {
    const int stressTestDurationSeconds = 30;
    const int operationsPerSecond = 50;

    auto startTest = high_resolution_clock::now();
    int totalOperations = 0;
    int failedOperations = 0;

    while (duration_cast<seconds>(high_resolution_clock::now() - startTest).count() < stressTestDurationSeconds) {
        std::vector<std::thread> concurrentThreads(4);

        // Launch concurrent operations
        for (int t = 0; t < 4; ++t) {
            concurrentThreads.emplace_back([&, t, &totalOperations, &failedOperations]() {
                for (int i = 0; i < operationsPerSecond / 4; ++i) {
                    try {
                        // Create unique test signal
                        auto testSignal = GenerateTestSignal(TestSignalType::SINE_WAVE, 0.02);

                        // Process audio
                        std::vector<float> output(testSignal.size());
                        audioEngine_->processBuffer(testSignal.data(), output.data(), testSignal.size() / 2);

                        totalOperations++;

                        // Validate output
                        float sum = 0.0f;
                        for (float sample : output) {
                            sum += std::abs(sample);
                        }

                        if (sum == 0.0f || std::isnan(sum) || std::isinf(sum)) {
                            failedOperations++;
                        }

                    } catch (const std::exception& e) {
                        failedOperations++;
                        std::cout << "Exception in stress test: " << e.what() << std::endl;
                    }
                }
            });
        }

        // Wait for batch of operations
        for (auto& thread : concurrentThreads) {
            thread.join();
        }

        // Small delay to prevent system overload
        std::this_thread::sleep_for(10ms);
    }

    double testDuration = duration_cast<seconds>(high_resolution_clock::now() - startTest).count();
    double operationsPerSecond = totalOperations / testDuration;
    double failureRate = (static_cast<double>(failedOperations) / totalOperations) * 100.0;

    std::cout << "System Stability Test Results:" << std::std::endl;
    std::cout << "  Test Duration: " << testDuration << " seconds" << std::endl;
    std::cout << "  Total Operations: " << totalOperations << std::endl;
    std::cout << "  Operations/Second: " << operationsPerSecond << std::endl;
    std::cout << "  Failed Operations: " << failedOperations << std::endl;
    std::cout << "  Failure Rate: " << failureRate << "%" << std::endl;

    // Stability expectations
    EXPECT_GT(totalOperations, 1000) << "Should process at least 1000 operations";
    EXPECT_LT(failureRate, 1.0) << "Failure rate should be less than 1%";
    EXPECT_GT(operationsPerSecond, 100.0) << "Should achieve >100 operations/second";
}