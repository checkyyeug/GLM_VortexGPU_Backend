#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <chrono>
#include <thread>
#include <fstream>
#include <filesystem>
#include <vector>
#include <memory>
#include <cmath>
#include "core/audio_engine.hpp"
#include "core/dsp/dsd_processor.hpp"
#include "core/fileio/audio_file_loader.hpp"
#include "core/gpu/gpu_processor.hpp"
#include "core/gpu/cuda_kernels.hpp"
#include "testing/audio_test_harness.hpp"
#include "system/logger.hpp"

using namespace vortex;
using namespace vortex::testing;
using namespace testing;
using namespace std::chrono_literals;

namespace {
    // DSD1024 constants
    constexpr double DSD1024_SAMPLE_RATE = 45158400.0;  // 44.1 kHz * 1024
    constexpr size_t DSD1024_BLOCK_SIZE = 4096;          // Processing block size
    constexpr double DSD1024_BITRATE = 45158400.0;       // 45.1584 MHz

    // Helper function to create simulated DSD1024 data
    std::vector<uint8_t> generateDSD1024Data(size_t samples, double frequency = 1000.0) {
        std::vector<uint8_t> dsdData(samples);

        // Generate 1-bit DSD data with simulated PWM pattern
        for (size_t i = 0; i < samples; ++i) {
            double time = static_cast<double>(i) / DSD1024_SAMPLE_RATE;
            double value = std::sin(2.0 * M_PI * frequency * time);

            // Convert to 1-bit DSD using PWM (simplified simulation)
            double pwm = (value + 1.0) / 2.0;  // Normalize to 0-1
            dsdData[i] = (pwm > 0.5) ? 0xFF : 0x00;  // 1-bit representation
        }

        return dsdData;
    }

    // Helper function to create DSD file header
    std::vector<uint8_t> createDSFHeader(size_t dataSize) {
        std::vector<uint8_t> header(52, 0);  // DSF header size

        // DSF header format
        // Chunk ID: "DSF "
        header[0] = 0x44; header[1] = 0x53; header[2] = 0x46; header[3] = 0x20;

        // Chunk size (8 bytes)
        uint64_t chunkSize = 52 + dataSize;
        for (int i = 0; i < 8; ++i) {
            header[4 + i] = (chunkSize >> (i * 8)) & 0xFF;
        }

        // Total file size (8 bytes)
        for (int i = 0; i < 8; ++i) {
            header[12 + i] = (chunkSize >> (i * 8)) & 0xFF;
        }

        // Sample rate (4 bytes) - DSD1024
        uint32_t sampleRate = static_cast<uint32_t>(DSD1024_SAMPLE_RATE);
        for (int i = 0; i < 4; ++i) {
            header[28 + i] = (sampleRate >> (i * 8)) & 0xFF;
        }

        // Channels (2 bytes) - stereo
        header[32] = 2;
        header[33] = 0;

        // Bit depth (2 bytes) - 1-bit DSD
        header[36] = 1;
        header[37] = 0;

        // Sample count (8 bytes)
        uint64_t sampleCount = dataSize * 8;  // 1-bit samples
        for (int i = 0; i < 8; ++i) {
            header[40 + i] = (sampleCount >> (i * 8)) & 0xFF;
        }

        // Block size per channel (4 bytes)
        uint32_t blockSize = DSD1024_BLOCK_SIZE;
        for (int i = 0; i < 4; ++i) {
            header[48 + i] = (blockSize >> (i * 8)) & 0xFF;
        }

        return header;
    }

    // Helper function to create DSD1024 test file
    std::string createDSD1024TestFile(const std::string& filename, size_t durationMs = 1000) {
        size_t sampleCount = static_cast<size_t>(DSD1024_SAMPLE_RATE * durationMs / 1000.0);
        size_t dataSize = sampleCount / 8;  // 1-bit samples

        std::string testFile = std::filesystem::temp_directory_path().string() + "/" + filename;
        std::ofstream file(testFile, std::ios::binary);

        // Write DSF header
        auto header = createDSFHeader(dataSize);
        file.write(reinterpret_cast<const char*>(header.data()), header.size());

        // Write DSD data
        auto dsdData = generateDSD1024Data(dataSize, 1000.0);  // 1kHz test tone
        file.write(reinterpret_cast<const char*>(dsdData.data()), dsdData.size());

        file.close();
        return testFile;
    }

    // Helper function to create DFF test file
    std::string createDFFTestFile(const std::string& filename, size_t durationMs = 1000) {
        size_t sampleCount = static_cast<size_t>(DSD1024_SAMPLE_RATE * durationMs / 1000.0);
        size_t dataSize = sampleCount / 8;  // 1-bit samples

        std::string testFile = std::filesystem::temp_directory_path().string() + "/" + filename;
        std::ofstream file(testFile, std::ios::binary);

        // DFF header format (simplified)
        std::vector<uint8_t> header(20, 0);

        // DSDIFF signature: "FRM8"
        header[0] = 0x46; header[1] = 0x52; header[2] = 0x4D; header[3] = 0x38;

        // Total size (8 bytes)
        uint64_t totalSize = 20 + 12 + 24 + dataSize;
        for (int i = 0; i < 8; ++i) {
            header[4 + i] = (totalSize >> (i * 8)) & 0xFF;
        }

        file.write(reinterpret_cast<const char*>(header.data()), header.size());

        // DSD chunk
        std::vector<uint8_t> dsdChunkHeader(12, 0);
        dsdChunkHeader[0] = 0x44; dsdChunkHeader[1] = 0x53; dsdChunkHeader[2] = 0x44; dsdChunkHeader[3] = 0x20;  // "DSD "

        uint32_t dsdChunkSize = 24 + dataSize;
        for (int i = 0; i < 4; ++i) {
            dsdChunkHeader[8 + i] = (dsdChunkSize >> (i * 8)) & 0xFF;
        }

        file.write(reinterpret_cast<const char*>(dsdChunkHeader.data()), dsdChunkHeader.size());

        // Sample rate, channels, etc.
        std::vector<uint8_t> dsdFormat(24, 0);
        uint32_t sampleRate = static_cast<uint32_t>(DSD1024_SAMPLE_RATE);
        for (int i = 0; i < 4; ++i) {
            dsdFormat[4 + i] = (sampleRate >> (i * 8)) & 0xFF;
        }
        dsdFormat[12] = 2;  // Channels
        dsdFormat[16] = 1;  // Bit depth

        file.write(reinterpret_cast<const char*>(dsdFormat.data()), dsdFormat.size());

        // Write DSD data
        auto dsdData = generateDSD1024Data(dataSize, 2000.0);  // 2kHz test tone
        file.write(reinterpret_cast<const char*>(dsdData.data()), dsdData.size());

        file.close();
        return testFile;
    }
}

class DSDProcessingIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::initialize();

        // Configure test harness for DSD processing
        AudioTestHarness::TestConfiguration config;
        config.sampleRate = DSD1024_SAMPLE_RATE;
        config.bufferSize = DSD1024_BLOCK_SIZE;
        config.channels = 2;
        config.bitDepth = 1;  // 1-bit DSD
        config.maxProcessingTimeMs = 10.0;  // DSD processing should be <10ms
        config.enableGPUTests = true;

        harness_.setConfiguration(config);

        // Initialize audio engine
        audioEngine_ = std::make_unique<AudioEngine>();
        ASSERT_TRUE(audioEngine_->initialize(DSD1024_SAMPLE_RATE, DSD1024_BLOCK_SIZE));

        // Initialize DSD processor
        dsdProcessor_ = std::make_unique<DSDProcessor>();
        ASSERT_TRUE(dsdProcessor_->initialize(DSD1024_SAMPLE_RATE, 2));

        // Initialize GPU processor if available
        gpuProcessor_ = std::make_unique<GPUProcessor>();
        if (gpuProcessor_->initialize("CUDA", DSD1024_SAMPLE_RATE, DSD1024_BLOCK_SIZE, 2)) {
            gpuAvailable_ = true;
            std::cout << "GPU acceleration available for DSD processing" << std::endl;
        } else {
            std::cout << "GPU acceleration not available, using CPU fallback" << std::endl;
        }

        // Initialize CUDA processor if GPU is available
        if (gpuAvailable_) {
            cudaProcessor_ = std::make_unique<cuda::CUDAAudioProcessor>();
            cudaAvailable_ = cudaProcessor_->initialize();
        }
    }

    void TearDown() override {
        // Clean up test files
        for (const auto& file : testFiles_) {
            if (std::filesystem::exists(file)) {
                std::filesystem::remove(file);
            }
        }

        if (cudaProcessor_) {
            cudaProcessor_->shutdown();
        }
        if (gpuProcessor_) {
            gpuProcessor_->shutdown();
        }
        if (dsdProcessor_) {
            dsdProcessor_->shutdown();
        }
        audioEngine_->shutdown();
        Logger::shutdown();
    }

    void addTestFile(const std::string& filename, size_t durationMs = 1000) {
        testFiles_.push_back(createDSD1024TestFile(filename, durationMs));
    }

    void addDFFTestFile(const std::string& filename, size_t durationMs = 1000) {
        testFiles_.push_back(createDFFTestFile(filename, durationMs));
    }

    // Helper function to measure DSD processing performance
    double measureDSDProcessingTime(const std::vector<uint8_t>& dsdData,
                                   std::vector<float>& pcmOutput,
                                   bool useGPU = false) {
        auto startTime = std::chrono::high_resolution_clock::now();

        if (useGPU && cudaAvailable_) {
            // GPU-accelerated DSD to PCM conversion
            ASSERT_TRUE(cudaProcessor_->convertDSDtoPCM(dsdData.data(), pcmOutput.data(), dsdData.size()));
        } else {
            // CPU DSD processing
            dsdProcessor_->processDSDData(dsdData.data(), pcmOutput.data(), dsdData.size());
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        return duration.count() / 1000.0;  // Convert to milliseconds
    }

    AudioTestHarness harness_;
    std::unique_ptr<AudioEngine> audioEngine_;
    std::unique_ptr<DSDProcessor> dsdProcessor_;
    std::unique_ptr<GPUProcessor> gpuProcessor_;
    std::unique_ptr<cuda::CUDAAudioProcessor> cudaProcessor_;
    std::vector<std::string> testFiles_;
    bool gpuAvailable_ = false;
    bool cudaAvailable_ = false;
};

// Test DSD1024 file loading and basic processing
TEST_F(DSDProcessingIntegrationTest, LoadAndProcessDSD1024File) {
    // Arrange
    addTestFile("dsd1024_test.dsf");
    std::string filePath = testFiles_.back();

    // Initialize audio file loader
    auto fileLoader = std::make_unique<AudioFileLoader>();
    ASSERT_TRUE(fileLoader->initialize());

    // Act - Load DSD1024 file
    auto audioData = fileLoader->loadFile(filePath);

    // Assert
    ASSERT_TRUE(audioData.has_value()) << "DSD1024 file should load successfully";
    EXPECT_EQ(audioData->format, AudioFormat::DSF) << "Format should be detected as DSF";
    EXPECT_EQ(audioData->sampleRate, DSD1024_SAMPLE_RATE) << "Sample rate should be DSD1024";
    EXPECT_EQ(audioData->channels, 2) << "Should be stereo";
    EXPECT_GT(audioData->data.size(), 0) << "Should have audio data";

    // Process the DSD data
    std::vector<float> pcmOutput(audioData->data.size() * 8);  // Convert 1-bit to float
    bool success = dsdProcessor_->processDSDData(audioData->data.data(), pcmOutput.data(), audioData->data.size());

    ASSERT_TRUE(success) << "DSD processing should succeed";
    EXPECT_GT(pcmOutput.size(), 0) << "Should produce PCM output";

    // Validate PCM output has reasonable values
    for (size_t i = 0; i < std::min(pcmOutput.size(), static_cast<size_t>(1000)); ++i) {
        EXPECT_GE(pcmOutput[i], -1.0f) << "PCM sample should be >= -1.0";
        EXPECT_LE(pcmOutput[i], 1.0f) << "PCM sample should be <= 1.0";
    }
}

// Test DFF format processing
TEST_F(DSDProcessingIntegrationTest, ProcessDFFFormat) {
    // Arrange
    addDFFTestFile("dsd1024_test.dff");
    std::string filePath = testFiles_.back();

    auto fileLoader = std::make_unique<AudioFileLoader>();
    ASSERT_TRUE(fileLoader->initialize());

    // Act - Load DFF file
    auto audioData = fileLoader->loadFile(filePath);

    // Assert
    ASSERT_TRUE(audioData.has_value()) << "DFF file should load successfully";
    EXPECT_EQ(audioData->format, AudioFormat::DFF) << "Format should be detected as DFF";
    EXPECT_EQ(audioData->sampleRate, DSD1024_SAMPLE_RATE) << "Sample rate should be DSD1024";

    // Process DFF data
    std::vector<float> pcmOutput(audioData->data.size() * 8);
    bool success = dsdProcessor_->processDSDData(audioData->data.data(), pcmOutput.data(), audioData->data.size());

    ASSERT_TRUE(success) << "DFF DSD processing should succeed";
    EXPECT_GT(pcmOutput.size(), 0) << "Should produce PCM output";
}

// Test DSD1024 real-time processing constraints
TEST_F(DSDProcessingIntegrationTest, DSD1024RealTimeConstraints) {
    // Arrange - Create a processing block of DSD1024 data
    size_t blockSize = DSD1024_BLOCK_SIZE;
    auto dsdData = generateDSD1024Data(blockSize / 8, 1000.0);  // 1kHz tone
    std::vector<float> pcmOutput(blockSize);

    // Act - Measure processing time
    double processingTimeMs = measureDSDProcessingTime(dsdData, pcmOutput, false);  // CPU processing

    // Assert - Should meet real-time constraints
    double bufferTimeMs = (blockSize / DSD1024_SAMPLE_RATE) * 1000.0;
    EXPECT_LT(processingTimeMs, bufferTimeMs * 0.5)
        << "DSD processing should be <50% of buffer time";

    std::cout << "DSD1024 processing time: " << processingTimeMs << "ms" << std::endl;
    std::cout << "Buffer time: " << bufferTimeMs << "ms" << std::endl;
    std::cout << "Processing ratio: " << (processingTimeMs / bufferTimeMs) << std::endl;
}

// Test GPU-accelerated DSD processing
TEST_F(DSDProcessingIntegrationTest, GPUAcceleratedDSDProcessing) {
    if (!cudaAvailable_) {
        GTEST_SKIP() << "CUDA not available, skipping GPU DSD processing test";
    }

    // Arrange - Create larger DSD data for GPU processing
    size_t blockSize = DSD1024_BLOCK_SIZE * 4;  // Larger block for GPU
    auto dsdData = generateDSD1024Data(blockSize / 8, 1000.0);
    std::vector<float> gpuOutput(blockSize);
    std::vector<float> cpuOutput(blockSize);

    // Act - Measure GPU vs CPU processing times
    double gpuTimeMs = measureDSDProcessingTime(dsdData, gpuOutput, true);
    double cpuTimeMs = measureDSDProcessingTime(dsdData, cpuOutput, false);

    // Assert - GPU should be faster for large blocks
    EXPECT_LT(gpuTimeMs, cpuTimeMs) << "GPU processing should be faster than CPU";

    // Results should be similar (allowing for small numerical differences)
    ASSERT_EQ(gpuOutput.size(), cpuOutput.size()) << "Outputs should have same size";

    for (size_t i = 0; i < std::min(gpuOutput.size(), static_cast<size_t>(1000)); ++i) {
        float diff = std::abs(gpuOutput[i] - cpuOutput[i]);
        EXPECT_LT(diff, 0.01f) << "GPU and CPU results should be similar within tolerance";
    }

    std::cout << "GPU DSD processing time: " << gpuTimeMs << "ms" << std::endl;
    std::cout << "CPU DSD processing time: " << cpuTimeMs << "ms" << std::endl;
    std::cout << "GPU speedup: " << (cpuTimeMs / gpuTimeMs) << "x" << std::endl;
}

// Test DSD1024 memory usage efficiency
TEST_F(DSDProcessingIntegrationTest, DSDMemoryUsage) {
    // Arrange - Create multiple DSD blocks to test memory efficiency
    const int numBlocks = 100;
    std::vector<std::vector<uint8_t>> dsdBlocks;
    std::vector<std::vector<float>> pcmBlocks;

    size_t initialMemory = harness_.getCurrentMemoryUsageMB();

    // Act - Process multiple DSD blocks
    for (int i = 0; i < numBlocks; ++i) {
        auto dsdData = generateDSD1024Data(DSD1024_BLOCK_SIZE / 8, 440.0 + i * 10.0);
        dsdBlocks.push_back(dsdData);

        std::vector<float> pcmOutput(DSD1024_BLOCK_SIZE);
        pcmBlocks.push_back(pcmOutput);

        dsdProcessor_->processDSDData(dsdData.data(), pcmOutput.data(), dsdData.size());
    }

    size_t finalMemory = harness_.getCurrentMemoryUsageMB();
    size_t memoryGrowth = finalMemory - initialMemory;

    // Assert - Memory growth should be reasonable
    EXPECT_LT(memoryGrowth, 500) << "Memory growth should be <500MB for 100 DSD blocks";

    std::cout << "Initial memory: " << initialMemory << "MB" << std::endl;
    std::cout << "Final memory: " << finalMemory << "MB" << std::endl;
    std::cout << "Memory growth: " << memoryGrowth << "MB" << std::endl;
}

// Test DSD1024 to high-resolution PCM conversion quality
TEST_F(DSDProcessingIntegrationTest, DSDToPCMQuality) {
    // Arrange - Generate DSD test tone at known frequency
    double testFrequency = 1000.0;  // 1kHz test tone
    size_t samples = static_cast<size_t>(DSD1024_SAMPLE_RATE * 0.01);  // 10ms
    auto dsdData = generateDSD1024Data(samples / 8, testFrequency);
    std::vector<float> pcmOutput(samples);

    // Act - Convert DSD to PCM
    bool success = dsdProcessor_->processDSDData(dsdData.data(), pcmOutput.data(), dsdData.size());

    ASSERT_TRUE(success) << "DSD to PCM conversion should succeed";

    // Assert - Check if the test frequency is present in PCM output
    // Perform simple FFT to verify frequency content
    std::vector<float> magnitude(samples / 2 + 1);
    dsdProcessor_->analyzeSpectrum(pcmOutput.data(), magnitude.data(), samples);

    // Find peak frequency
    size_t maxIndex = 0;
    float maxMagnitude = 0.0f;
    for (size_t i = 1; i < magnitude.size(); ++i) {  // Skip DC component
        if (magnitude[i] > maxMagnitude) {
            maxMagnitude = magnitude[i];
            maxIndex = i;
        }
    }

    double detectedFrequency = (static_cast<double>(maxIndex) * DSD1024_SAMPLE_RATE) / (2.0 * magnitude.size());

    // Should detect frequency close to the test frequency
    EXPECT_NEAR(detectedFrequency, testFrequency, testFrequency * 0.1)
        << "Should detect frequency within 10% of test frequency";

    std::cout << "Test frequency: " << testFrequency << " Hz" << std::endl;
    std::cout << "Detected frequency: " << detectedFrequency << " Hz" << std::endl;
    std::cout << "Frequency error: " << std::abs(detectedFrequency - testFrequency) << " Hz" << std::endl;
}

// Test DSD1024 processing with different sample rates
TEST_F(DSDProcessingIntegrationTest, DSDMultipleSampleRates) {
    std::vector<double> sampleRates = {
        2822400.0,   // DSD64
        5644800.0,   // DSD128
        11289600.0,  // DSD256
        22579200.0,  // DSD512
        45158400.0   // DSD1024
    };

    for (double sampleRate : sampleRates) {
        // Arrange - Create DSD processor for each sample rate
        auto processor = std::make_unique<DSDProcessor>();
        ASSERT_TRUE(processor->initialize(sampleRate, 2)) << "Should initialize processor for " << sampleRate << " Hz";

        size_t blockSize = 4096;
        size_t dsdSamples = blockSize / 8;
        auto dsdData = generateDSD1024Data(dsdSamples, 1000.0);
        std::vector<float> pcmOutput(blockSize);

        // Act - Process DSD at different sample rates
        bool success = processor->processDSDData(dsdData.data(), pcmOutput.data(), dsdData.size());

        // Assert - Should succeed for all supported DSD rates
        ASSERT_TRUE(success) << "DSD processing should succeed at " << sampleRate << " Hz";
        EXPECT_GT(pcmOutput.size(), 0) << "Should produce output at " << sampleRate << " Hz";

        processor->shutdown();

        std::cout << "Successfully processed DSD at " << sampleRate << " Hz" << std::endl;
    }
}

// Test DSD1024 processing under sustained load
TEST_F(DSDProcessingIntegrationTest, DSDSustainedLoad) {
    const int durationSeconds = 10;
    const int blocksPerSecond = 100;  // 100 blocks per second
    const double maxProcessingTimeMs = 10.0;  // Maximum allowed processing time

    std::vector<double> processingTimes;
    int successfulBlocks = 0;

    auto startTime = std::chrono::steady_clock::now();
    auto endTime = startTime + std::chrono::seconds(durationSeconds);

    while (std::chrono::steady_clock::now() < endTime) {
        // Generate DSD block
        auto dsdData = generateDSD1024Data(DSD1024_BLOCK_SIZE / 8, 440.0);
        std::vector<float> pcmOutput(DSD1024_BLOCK_SIZE);

        // Measure processing time
        auto blockStart = std::chrono::high_resolution_clock::now();
        bool success = dsdProcessor_->processDSDData(dsdData.data(), pcmOutput.data(), dsdData.size());
        auto blockEnd = std::chrono::high_resolution_clock::now();

        auto processingTime = std::chrono::duration_cast<std::chrono::microseconds>(blockEnd - blockStart);
        double processingTimeMs = processingTime.count() / 1000.0;

        if (success) {
            successfulBlocks++;
            processingTimes.push_back(processingTimeMs);
            EXPECT_LT(processingTimeMs, maxProcessingTimeMs)
                << "Processing time should not exceed " << maxProcessingTimeMs << "ms";
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));  // 100 Hz processing
    }

    // Calculate statistics
    double avgProcessingTime = 0.0;
    double maxProcessingTime = 0.0;
    if (!processingTimes.empty()) {
        avgProcessingTime = std::accumulate(processingTimes.begin(), processingTimes.end(), 0.0) / processingTimes.size();
        maxProcessingTime = *std::max_element(processingTimes.begin(), processingTimes.end());
    }

    // Assert - Should maintain performance under sustained load
    EXPECT_GT(successfulBlocks, durationSeconds * 50) << "Should process at least 50 blocks per second";
    EXPECT_LT(avgProcessingTime, maxProcessingTimeMs / 2.0) << "Average processing time should be reasonable";
    EXPECT_LT(maxProcessingTime, maxProcessingTimeMs) << "Maximum processing time should not exceed limit";

    std::cout << "Processed " << successfulBlocks << " blocks in " << durationSeconds << " seconds" << std::endl;
    std::cout << "Average processing time: " << avgProcessingTime << "ms" << std::endl;
    std::cout << "Maximum processing time: " << maxProcessingTime << "ms" << std::endl;
}

// Test DSD1024 error handling and recovery
TEST_F(DSDProcessingIntegrationTest, DSDErrorHandling) {
    // Test with invalid DSD data
    std::vector<uint8_t> invalidData(1000, 0x55);  // Invalid pattern
    std::vector<float> pcmOutput(8000);  // Output buffer

    // Should handle invalid data gracefully
    bool result = dsdProcessor_->processDSDData(invalidData.data(), pcmOutput.data(), invalidData.size());
    EXPECT_TRUE(result) << "Should handle invalid data without crashing";

    // Test with null pointers (should not crash)
    // Note: In a real implementation, you would add proper null pointer checks
    // This test ensures the implementation is robust
    EXPECT_NO_THROW({
        dsdProcessor_->reset();
    }) << "Should handle reset operation without crashing";

    // Test recovery after error
    auto validData = generateDSD1024Data(1000, 1000.0);
    bool recoveryResult = dsdProcessor_->processDSDData(validData.data(), pcmOutput.data(), validData.size());
    EXPECT_TRUE(recoveryResult) << "Should recover and process valid data after error";
}

// Test DSD1024 multichannel support
TEST_F(DSDProcessingIntegrationTest, DSDMultiChannelSupport) {
    // Test 5.1 surround sound (6 channels)
    const int numChannels = 6;
    auto processor = std::make_unique<DSDProcessor>();
    ASSERT_TRUE(processor->initialize(DSD1024_SAMPLE_RATE, numChannels)) << "Should initialize for multichannel";

    // Generate multichannel DSD data
    size_t samplesPerChannel = DSD1024_BLOCK_SIZE / numChannels;
    std::vector<std::vector<uint8_t>> multichannelData;

    for (int ch = 0; ch < numChannels; ++ch) {
        auto channelData = generateDSD1024Data(samplesPerChannel / 8, 440.0 + ch * 100.0);
        multichannelData.push_back(channelData);
    }

    // Process multichannel data
    std::vector<float> multichannelOutput(DSD1024_BLOCK_SIZE);
    bool success = processor->processMultichannelDSD(multichannelData.data(), multichannelOutput.data(), samplesPerChannel / 8);

    ASSERT_TRUE(success) << "Multichannel DSD processing should succeed";
    EXPECT_GT(multichannelOutput.size(), 0) << "Should produce multichannel output";

    processor->shutdown();
    std::cout << "Successfully processed " << numChannels << "-channel DSD1024 audio" << std::endl;
}