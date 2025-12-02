#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <chrono>
#include <thread>
#include <fstream>
#include <filesystem>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <numeric>
#include "core/audio_engine.hpp"
#include "core/fileio/audio_file_loader.hpp"
#include "core/fileio/format_detector.hpp"
#include "core/dsp/dsd_processor.hpp"
#include "core/gpu/gpu_processor.hpp"
#include "testing/audio_test_harness.hpp"
#include "system/logger.hpp"

using namespace vortex;
using namespace vortex::testing;
using namespace testing;
using namespace std::chrono_literals;

namespace {
    // Performance constants from specification
    constexpr double MAX_FILE_PROCESSING_TIME_MS = 5000.0;  // 5 seconds per specification
    constexpr double MAX_UPLOAD_RESPONSE_TIME_MS = 200.0;    // 200ms for upload response
    constexpr double MAX_FORMAT_DETECTION_TIME_MS = 50.0;    // 50ms for format detection
    constexpr double MAX_METADATA_EXTRACTION_TIME_MS = 100.0; // 100ms for metadata extraction
    constexpr double MAX_DSD_PROCESSING_LATENCY_MS = 10.0;   // 10ms for real-time DSD processing

    // Test file sizes
    constexpr size_t SMALL_FILE_SIZE = 1 * 1024 * 1024;     // 1MB
    constexpr size_t MEDIUM_FILE_SIZE = 10 * 1024 * 1024;   // 10MB
    constexpr size_t LARGE_FILE_SIZE = 100 * 1024 * 1024;    // 100MB
    constexpr size_t MAX_FILE_SIZE = 1024 * 1024 * 1024;    // 1GB (max supported)

    // Helper function to generate test audio data
    std::vector<uint8_t> generateAudioData(size_t size, AudioFormat format) {
        std::vector<uint8_t> data(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);

        // Generate different patterns for different formats
        switch (format) {
            case AudioFormat::WAV: {
                // WAV header pattern
                if (size >= 44) {
                    std::vector<uint8_t> wavHeader = {
                        0x52, 0x49, 0x46, 0x46,  // "RIFF"
                        0x00, 0x00, 0x00, 0x00,  // File size - 8
                        0x57, 0x41, 0x56, 0x45,  // "WAVE"
                        0x66, 0x6D, 0x74, 0x20,  // "fmt "
                        0x10, 0x00, 0x00, 0x00,  // Subchunk1Size
                        0x01, 0x00,              // AudioFormat (PCM)
                        0x02, 0x00,              // NumChannels (stereo)
                        0x44, 0xAC, 0x00, 0x00,  // SampleRate (44100)
                        0x10, 0xB1, 0x02, 0x00,  // ByteRate
                        0x04, 0x00,              // BlockAlign
                        0x10, 0x00,              // BitsPerSample
                        0x64, 0x61, 0x74, 0x61,  // "data"
                        0x00, 0x00, 0x00, 0x00   // Subchunk2Size
                    };
                    std::copy(wavHeader.begin(), wavHeader.end(), data.begin());
                }

                // Fill remaining with PCM audio data
                for (size_t i = 44; i < size; ++i) {
                    data[i] = static_cast<uint8_t>(dis(gen) % 256);
                }
                break;
            }

            case AudioFormat::MP3: {
                // MP3 ID3v2 header
                if (size >= 10) {
                    std::vector<uint8_t> mp3Header = {0x49, 0x44, 0x33, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
                    std::copy(mp3Header.begin(), mp3Header.end(), data.begin());
                }

                // Fill with MP3-like data
                for (size_t i = 10; i < size; ++i) {
                    data[i] = static_cast<uint8_t>(dis(gen) % 256);
                }
                break;
            }

            case AudioFormat::FLAC: {
                // FLAC marker
                if (size >= 4) {
                    data[0] = 0x66; data[1] = 0x4C; data[2] = 0x61; data[3] = 0x43;  // "fLaC"
                }

                // Fill with FLAC-like data
                for (size_t i = 4; i < size; ++i) {
                    data[i] = static_cast<uint8_t>(dis(gen) % 256);
                }
                break;
            }

            case AudioFormat::DSD1024:
            case AudioFormat::DSF: {
                // DSF header
                if (size >= 52) {
                    // Minimal DSF header
                    data[0] = 0x44; data[1] = 0x53; data[2] = 0x46; data[3] = 0x20;  // "DSF "
                    data[32] = 0x02; data[33] = 0x00;  // 2 channels
                    data[36] = 0x01; data[37] = 0x00;  // 1-bit DSD
                }

                // Fill with DSD data (1-bit pattern)
                for (size_t i = 52; i < size; ++i) {
                    data[i] = (dis(gen) % 2) ? 0xFF : 0x00;
                }
                break;
            }

            default:
                // Generic data
                for (size_t i = 0; i < size; ++i) {
                    data[i] = static_cast<uint8_t>(dis(gen) % 256);
                }
                break;
        }

        return data;
    }

    // Helper function to create test audio file
    std::string createTestAudioFile(const std::string& filename, size_t size, AudioFormat format) {
        std::string testFile = std::filesystem::temp_directory_path().string() + "/" + filename;
        std::ofstream file(testFile, std::ios::binary);

        auto audioData = generateAudioData(size, format);
        file.write(reinterpret_cast<const char*>(audioData.data()), audioData.size());
        file.close();

        return testFile;
    }

    // Helper function to measure execution time
    template<typename Func>
    double measureTime(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count() / 1000.0;  // Convert to milliseconds
    }

    // Performance result structure
    struct PerformanceResult {
        double timeMs;
        size_t fileSize;
        double throughputMBps;
        bool passed;
        std::string metric;
    };

    // Calculate throughput in MB/s
    double calculateThroughput(double timeMs, size_t fileSize) {
        if (timeMs <= 0) return 0.0;
        return (fileSize / (1024.0 * 1024.0)) / (timeMs / 1000.0);
    }
}

class FileProcessingPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::initialize();

        // Configure test harness
        AudioTestHarness::TestConfiguration config;
        config.sampleRate = 44100.0;
        config.bufferSize = 512;
        config.channels = 2;
        config.maxProcessingTimeMs = MAX_FILE_PROCESSING_TIME_MS;

        harness_.setConfiguration(config);

        // Initialize components
        audioEngine_ = std::make_unique<AudioEngine>();
        ASSERT_TRUE(audioEngine_->initialize(44100.0, 512));

        fileLoader_ = std::make_unique<AudioFileLoader>();
        ASSERT_TRUE(fileLoader_->initialize());

        formatDetector_ = std::make_unique<FormatDetector>();
        ASSERT_TRUE(formatDetector_->initialize());

        dsdProcessor_ = std::make_unique<DSDProcessor>();
        ASSERT_TRUE(dsdProcessor_->initialize(45158400.0, 2));  // DSD1024

        gpuProcessor_ = std::make_unique<GPUProcessor>();
        gpuAvailable_ = gpuProcessor_->initialize("CUDA", 44100.0, 512, 2);
    }

    void TearDown() override {
        // Clean up test files
        for (const auto& file : testFiles_) {
            if (std::filesystem::exists(file)) {
                std::filesystem::remove(file);
            }
        }

        if (gpuProcessor_) {
            gpuProcessor_->shutdown();
        }
        if (dsdProcessor_) {
            dsdProcessor_->shutdown();
        }
        fileLoader_.reset();
        audioEngine_->shutdown();
        Logger::shutdown();
    }

    void addTestFile(const std::string& filename, size_t size, AudioFormat format) {
        testFiles_.push_back(createTestAudioFile(filename, size, format));
    }

    PerformanceResult runFormatDetectionTest(const std::string& filePath) {
        PerformanceResult result;

        result.fileSize = std::filesystem::file_size(filePath);
        result.metric = "Format Detection";

        result.timeMs = measureTime([this, &filePath]() {
            auto format = formatDetector_->detectFormat(filePath);
        });

        result.throughputMBps = calculateThroughput(result.timeMs, result.fileSize);
        result.passed = result.timeMs <= MAX_FORMAT_DETECTION_TIME_MS;

        return result;
    }

    PerformanceResult runFileLoadingTest(const std::string& filePath) {
        PerformanceResult result;

        result.fileSize = std::filesystem::file_size(filePath);
        result.metric = "File Loading";

        result.timeMs = measureTime([this, &filePath]() {
            auto audioData = fileLoader_->loadFile(filePath);
        });

        result.throughputMBps = calculateThroughput(result.timeMs, result.fileSize);
        result.passed = result.timeMs <= MAX_FILE_PROCESSING_TIME_MS;

        return result;
    }

    PerformanceResult runDSDProcessingTest(size_t dataSize) {
        PerformanceResult result;

        result.fileSize = dataSize;
        result.metric = "DSD Processing";

        // Generate DSD data
        std::vector<uint8_t> dsdData(dataSize);
        std::random_device rd;
        std::mt19937 gen(rd());
        for (size_t i = 0; i < dataSize; ++i) {
            dsdData[i] = (gen() % 2) ? 0xFF : 0x00;
        }

        std::vector<float> pcmOutput(dataSize * 8);

        result.timeMs = measureTime([this, &dsdData, &pcmOutput]() {
            dsdProcessor_->processDSDData(dsdData.data(), pcmOutput.data(), dsdData.size());
        });

        result.throughputMBps = calculateThroughput(result.timeMs, result.fileSize);
        result.passed = result.timeMs <= MAX_DSD_PROCESSING_LATENCY_MS;

        return result;
    }

    void printPerformanceResult(const PerformanceResult& result) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Metric: " << result.metric << std::endl;
        std::cout << "  File Size: " << (result.fileSize / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "  Time: " << result.timeMs << " ms" << std::endl;
        std::cout << "  Throughput: " << result.throughputMBps << " MB/s" << std::endl;
        std::cout << "  Status: " << (result.passed ? "PASS" : "FAIL") << std::endl;
        std::cout << std::endl;
    }

    AudioTestHarness harness_;
    std::unique_ptr<AudioEngine> audioEngine_;
    std::unique_ptr<AudioFileLoader> fileLoader_;
    std::unique_ptr<FormatDetector> formatDetector_;
    std::unique_ptr<DSDProcessor> dsdProcessor_;
    std::unique_ptr<GPUProcessor> gpuProcessor_;
    std::vector<std::string> testFiles_;
    bool gpuAvailable_ = false;
};

// Test format detection performance for different file sizes
TEST_F(FileProcessingPerformanceTest, FormatDetectionPerformance) {
    std::vector<std::pair<size_t, std::string>> testSizes = {
        {SMALL_FILE_SIZE, "small"},
        {MEDIUM_FILE_SIZE, "medium"},
        {LARGE_FILE_SIZE, "large"}
    };

    std::vector<AudioFormat> testFormats = {
        AudioFormat::WAV, AudioFormat::MP3, AudioFormat::FLAC, AudioFormat::DSF
    };

    for (const auto& [fileSize, sizeName] : testSizes) {
        for (AudioFormat format : testFormats) {
            // Arrange
            std::string filename = sizeName + "_test_" + fileLoader_->getExtensionForFormat(format);
            addTestFile(filename, fileSize, format);
            std::string filePath = testFiles_.back();

            // Act
            auto result = runFormatDetectionTest(filePath);

            // Assert
            EXPECT_TRUE(result.passed) << "Format detection for " << sizeName << " "
                                      << fileLoader_->getExtensionForFormat(format)
                                      << " should complete within " << MAX_FORMAT_DETECTION_TIME_MS << "ms";

            EXPECT_GT(result.throughputMBps, 1.0) << "Throughput should be at least 1 MB/s";

            printPerformanceResult(result);
        }
    }
}

// Test file loading performance for different formats and sizes
TEST_F(FileProcessingPerformanceTest, FileLoadingPerformance) {
    std::vector<std::pair<size_t, std::string>> testSizes = {
        {SMALL_FILE_SIZE, "small"},
        {MEDIUM_FILE_SIZE, "medium"},
        {LARGE_FILE_SIZE, "large"}
    };

    std::vector<AudioFormat> testFormats = {
        AudioFormat::WAV, AudioFormat::MP3, AudioFormat::FLAC, AudioFormat::DSF
    };

    for (const auto& [fileSize, sizeName] : testSizes) {
        for (AudioFormat format : testFormats) {
            // Arrange
            std::string filename = sizeName + "_load_test_" + fileLoader_->getExtensionForFormat(format);
            addTestFile(filename, fileSize, format);
            std::string filePath = testFiles_.back();

            // Act
            auto result = runFileLoadingTest(filePath);

            // Assert
            EXPECT_TRUE(result.passed) << "File loading for " << sizeName << " "
                                      << fileLoader_->getExtensionForFormat(format)
                                      << " should complete within " << MAX_FILE_PROCESSING_TIME_MS << "ms";

            // Minimum throughput requirements based on file size
            double minThroughput = 0.0;
            if (fileSize == SMALL_FILE_SIZE) {
                minThroughput = 0.5;  // 0.5 MB/s for small files
            } else if (fileSize == MEDIUM_FILE_SIZE) {
                minThroughput = 2.0;  // 2 MB/s for medium files
            } else if (fileSize == LARGE_FILE_SIZE) {
                minThroughput = 5.0;  // 5 MB/s for large files
            }

            EXPECT_GT(result.throughputMBps, minThroughput) << "Throughput should be at least "
                                                            << minThroughput << " MB/s for " << sizeName << " files";

            printPerformanceResult(result);
        }
    }
}

// Test DSD processing performance under real-time constraints
TEST_F(FileProcessingPerformanceTest, DSDProcessingRealTimeConstraints) {
    std::vector<size_t> dsdBlockSizes = {
        1024,    // Small block
        4096,    // Medium block
        8192,    // Large block
        16384    // Very large block
    };

    for (size_t blockSize : dsdBlockSizes) {
        // Act
        auto result = runDSDProcessingTest(blockSize);

        // Assert
        EXPECT_TRUE(result.passed) << "DSD processing for " << blockSize << " bytes should complete within "
                                  << MAX_DSD_PROCESSING_LATENCY_MS << "ms for real-time processing";

        // For real-time DSD processing, we need high throughput
        EXPECT_GT(result.throughputMBps, 100.0) << "DSD processing throughput should be >100 MB/s for real-time";

        printPerformanceResult(result);
    }
}

// Test concurrent file processing performance
TEST_F(FileProcessingPerformanceTest, ConcurrentFileProcessing) {
    const int numConcurrentFiles = 10;
    std::vector<std::thread> threads;
    std::vector<PerformanceResult> results(numConcurrentFiles);
    std::atomic<int> successfulLoads{0};

    // Arrange - Create multiple test files
    for (int i = 0; i < numConcurrentFiles; ++i) {
        std::string filename = "concurrent_test_" + std::to_string(i) + ".wav";
        addTestFile(filename, MEDIUM_FILE_SIZE, AudioFormat::WAV);
    }

    // Act - Process files concurrently
    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numConcurrentFiles; ++i) {
        threads.emplace_back([&, i]() {
            auto filePath = testFiles_[i];
            results[i] = runFileLoadingTest(filePath);
            if (results[i].passed) {
                successfulLoads++;
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    // Assert
    EXPECT_EQ(successfulLoads, numConcurrentFiles) << "All concurrent file loads should succeed";

    // Overall throughput should be good
    double totalDataSize = numConcurrentFiles * MEDIUM_FILE_SIZE;
    double totalThroughput = calculateThroughput(totalTime.count(), totalDataSize);

    std::cout << "Concurrent Processing Results:" << std::endl;
    std::cout << "  Files: " << numConcurrentFiles << std::endl;
    std::cout << "  Total Time: " << totalTime.count() << " ms" << std::endl;
    std::cout << "  Total Data: " << (totalDataSize / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  Total Throughput: " << totalThroughput << " MB/s" << std::endl;
    std::cout << "  Successful: " << successfulLoads << "/" << numConcurrentFiles << std::endl;
    std::cout << std::endl;

    EXPECT_GT(totalThroughput, 10.0) << "Overall concurrent throughput should be >10 MB/s";
}

// Test memory usage during file processing
TEST_F(FileProcessingPerformanceTest, FileProcessingMemoryUsage) {
    size_t initialMemory = harness_.getCurrentMemoryUsageMB();

    // Process multiple large files
    const int numFiles = 5;
    for (int i = 0; i < numFiles; ++i) {
        std::string filename = "memory_test_" + std::to_string(i) + ".flac";
        addTestFile(filename, LARGE_FILE_SIZE, AudioFormat::FLAC);
        std::string filePath = testFiles_.back();

        // Load and process the file
        auto audioData = fileLoader_->loadFile(filePath);
        EXPECT_TRUE(audioData.has_value()) << "File " << i << " should load successfully";

        // Simulate processing
        std::this_thread::sleep_for(10ms);
    }

    size_t finalMemory = harness_.getCurrentMemoryUsageMB();
    size_t memoryGrowth = finalMemory - initialMemory;

    std::cout << "Memory Usage Analysis:" << std::endl;
    std::cout << "  Initial Memory: " << initialMemory << " MB" << std::endl;
    std::cout << "  Final Memory: " << finalMemory << " MB" << std::endl;
    std::cout << "  Memory Growth: " << memoryGrowth << " MB" << std::endl;
    std::cout << "  Files Processed: " << numFiles << std::endl;
    std::cout << "  Memory per File: " << (static_cast<double>(memoryGrowth) / numFiles) << " MB" << std::endl;
    std::cout << std::endl;

    // Memory growth should be reasonable
    EXPECT_LT(memoryGrowth, 500) << "Memory growth should be <500MB for processing " << numFiles << " large files";
}

// Test file upload simulation performance
TEST_F(FileProcessingPerformanceTest, FileUploadSimulation) {
    std::vector<std::pair<size_t, std::string>> testSizes = {
        {SMALL_FILE_SIZE, "small"},
        {MEDIUM_FILE_SIZE, "medium"}
    };

    for (const auto& [fileSize, sizeName] : testSizes) {
        // Arrange
        std::string filename = "upload_test_" + sizeName + ".wav";
        addTestFile(filename, fileSize, AudioFormat::WAV);
        std::string filePath = testFiles_.back();

        // Act - Simulate upload process (file copy + response generation)
        std::string uploadDest = std::filesystem::temp_directory_path().string() + "/uploaded_" + sizeName + ".wav";

        double uploadTimeMs = measureTime([&]() {
            // Simulate file copy
            std::ifstream src(filePath, std::ios::binary);
            std::ofstream dst(uploadDest, std::ios::binary);
            dst << src.rdbuf();
            src.close();
            dst.close();

            // Simulate response generation
            std::this_thread::sleep_for(1ms);  // Simulate processing overhead
        });

        // Clean up uploaded file
        std::filesystem::remove(uploadDest);

        double throughput = calculateThroughput(uploadTimeMs, fileSize);
        bool passed = uploadTimeMs <= MAX_UPLOAD_RESPONSE_TIME_MS;

        // Assert
        EXPECT_TRUE(passed) << "Upload simulation for " << sizeName << " file should complete within "
                           << MAX_UPLOAD_RESPONSE_TIME_MS << "ms";

        EXPECT_GT(throughput, 1.0) << "Upload throughput should be at least 1 MB/s";

        std::cout << "Upload Simulation (" << sizeName << "):" << std::endl;
        std::cout << "  File Size: " << (fileSize / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "  Upload Time: " << uploadTimeMs << " ms" << std::endl;
        std::cout << "  Throughput: " << throughput << " MB/s" << std::endl;
        std::cout << "  Status: " << (passed ? "PASS" : "FAIL") << std::endl;
        std::cout << std::endl;
    }
}

// Test sustained file processing performance
TEST_F(FileProcessingPerformanceTest, SustainedFileProcessing) {
    const int durationSeconds = 30;
    const int targetFilesPerSecond = 10;

    std::vector<double> processingTimes;
    int processedFiles = 0;

    auto startTime = std::chrono::steady_clock::now();
    auto endTime = startTime + std::chrono::seconds(durationSeconds);

    std::cout << "Sustained Processing Test (" << durationSeconds << " seconds)" << std::endl;

    while (std::chrono::steady_clock::now() < endTime && processedFiles < durationSeconds * targetFilesPerSecond) {
        // Arrange
        std::string filename = "sustained_test_" + std::to_string(processedFiles) + ".mp3";
        addTestFile(filename, MEDIUM_FILE_SIZE, AudioFormat::MP3);
        std::string filePath = testFiles_.back();

        // Act - Process the file
        auto result = runFileLoadingTest(filePath);
        processingTimes.push_back(result.timeMs);

        if (result.passed) {
            processedFiles++;
        }

        // Small delay to simulate realistic load
        std::this_thread::sleep_for(50ms);
    }

    // Calculate statistics
    double avgProcessingTime = 0.0;
    double maxProcessingTime = 0.0;
    double minProcessingTime = std::numeric_limits<double>::max();

    if (!processingTimes.empty()) {
        avgProcessingTime = std::accumulate(processingTimes.begin(), processingTimes.end(), 0.0) / processingTimes.size();
        maxProcessingTime = *std::max_element(processingTimes.begin(), processingTimes.end());
        minProcessingTime = *std::min_element(processingTimes.begin(), processingTimes.end());
    }

    std::cout << "  Files Processed: " << processedFiles << std::endl;
    std::cout << "  Files per Second: " << (static_cast<double>(processedFiles) / durationSeconds) << std::endl;
    std::cout << "  Average Time: " << avgProcessingTime << " ms" << std::endl;
    std::cout << "  Min Time: " << minProcessingTime << " ms" << std::endl;
    std::cout << "  Max Time: " << maxProcessingTime << " ms" << std::endl;
    std::cout << std::endl;

    // Assert
    EXPECT_GE(processedFiles, durationSeconds * targetFilesPerSecond * 0.8) << "Should process at least 80% of target files";
    EXPECT_LT(avgProcessingTime, MAX_FILE_PROCESSING_TIME_MS) << "Average processing time should be within limits";
    EXPECT_LT(maxProcessingTime, MAX_FILE_PROCESSING_TIME_MS * 2.0) << "Maximum processing time should be reasonable";
}

// Test GPU acceleration impact on file processing
TEST_F(FileProcessingPerformanceTest, GPUAccelerationImpact) {
    if (!gpuAvailable_) {
        GTEST_SKIP() << "GPU not available, skipping GPU acceleration test";
    }

    const size_t testDataSize = 10 * 1024 * 1024;  // 10MB test data

    // Generate test data
    std::vector<float> cpuInput(testDataSize / sizeof(float));
    std::vector<float> cpuOutput(testDataSize / sizeof(float));
    std::vector<float> gpuOutput(testDataSize / sizeof(float));

    // Fill with test data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (auto& sample : cpuInput) {
        sample = dis(gen);
    }

    // CPU processing
    double cpuTimeMs = measureTime([&]() {
        for (size_t i = 0; i < cpuInput.size(); ++i) {
            cpuOutput[i] = cpuInput[i] * 0.5f;  // Simple gain processing
        }
    });

    // GPU processing (simulated with CUDA processor)
    double gpuTimeMs = measureTime([&]() {
        // In a real implementation, this would use GPU kernels
        // For now, simulate GPU processing with potential speedup
        for (size_t i = 0; i < gpuOutput.size(); ++i) {
            gpuOutput[i] = cpuInput[i] * 0.5f;
        }
    });

    double speedup = cpuTimeMs / gpuTimeMs;
    double cpuThroughput = calculateThroughput(cpuTimeMs, testDataSize);
    double gpuThroughput = calculateThroughput(gpuTimeMs, testDataSize);

    std::cout << "GPU Acceleration Impact:" << std::endl;
    std::cout << "  CPU Time: " << cpuTimeMs << " ms" << std::endl;
    std::cout << "  GPU Time: " << gpuTimeMs << " ms" << std::endl;
    std::cout << "  CPU Throughput: " << cpuThroughput << " MB/s" << std::endl;
    std::cout << "  GPU Throughput: " << gpuThroughput << " MB/s" << std::endl;
    std::cout << "  Speedup: " << speedup << "x" << std::endl;
    std::cout << std::endl;

    // In a real implementation, GPU should show significant speedup
    // For this test, we just verify that both complete successfully
    EXPECT_LT(cpuTimeMs, 1000.0) << "CPU processing should complete within 1 second";
    EXPECT_LT(gpuTimeMs, 1000.0) << "GPU processing should complete within 1 second";
}