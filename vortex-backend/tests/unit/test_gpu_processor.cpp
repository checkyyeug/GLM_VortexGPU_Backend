#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "core/gpu/gpu_processor.hpp"
#include "system/logger.hpp"

using namespace vortex;
using ::testing::_;
using ::testing::Return;
using ::testing::DoAll;
using ::testing::SetArgReferee;

class GPUProcessorTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::initialize("test_gpu_processor.log", Logger::Level::Debug, false);
    }

    void TearDown() override {
        Logger::shutdown();
    }

    // Test helper functions
    bool hasCUDAGPU() {
        return GPUProcessor::isCUDAAvailable();
    }

    bool hasOpenCLGPU() {
        return GPUProcessor::isOpenCLAvailable();
    }

    bool hasVulkanGPU() {
        return GPUProcessor::isVulkanAvailable();
    }
};

// Test GPU processor availability
TEST_F(GPUProcessorTest, TestGPUAvailability) {
    // Test that we can query GPU availability without crashing
    bool cudaAvailable = GPUProcessor::isCUDAAvailable();
    bool openclAvailable = GPUProcessor::isOpenCLAvailable();
    bool vulkanAvailable = GPUProcessor::isVulkanAvailable();

    // Log the availability for debugging
    LOG_INFO("CUDA available: {}", cudaAvailable);
    LOG_INFO("OpenCL available: {}", openclAvailable);
    LOG_INFO("Vulkan available: {}", vulkanAvailable);

    // At least one backend should be available in most test environments
    EXPECT_TRUE(cudaAvailable || openclAvailable || vulkanAvailable);
}

// Test GPU processor factory
TEST_F(GPUProcessorTest, TestGPUProcessorFactory) {
    auto availableBackends = GPUProcessorFactory::getAvailableBackends();

    EXPECT_FALSE(availableBackends.empty());

    // Try to create processors for each available backend
    for (const auto& backend : availableBackends) {
        auto processor = GPUProcessorFactory::create(backend);
        EXPECT_NE(processor, nullptr);
        EXPECT_EQ(processor->getCurrentBackend(), backend);

        // Should be able to shutdown without crashing
        processor->shutdown();
    }
}

// Test GPU processor initialization with CUDA
TEST_F(GPUProcessorTest, TestCUDAInitialization) {
    if (!hasCUDAGPU()) {
        GTEST_SKIP() << "CUDA GPU not available, skipping test";
    }

    auto processor = GPUProcessorFactory::create(GPUBackend::CUDA);
    ASSERT_NE(processor, nullptr);

    // Test initialization
    const uint32_t sampleRate = 44100;
    const uint32_t bufferSize = 512;
    const uint16_t channels = 2;

    EXPECT_TRUE(processor->initialize(GPUBackend::CUDA, sampleRate, bufferSize, channels));
    EXPECT_TRUE(processor->isInitialized());
    EXPECT_EQ(processor->getCurrentBackend(), GPUBackend::CUDA);

    // Test device enumeration
    auto devices = processor->getAvailableDevices();
    EXPECT_FALSE(devices.empty());

    if (!devices.empty()) {
        EXPECT_TRUE(processor->selectDevice(devices[0]));
        EXPECT_EQ(processor->getCurrentDevice(), devices[0]);
    }

    // Test memory operations
    const size_t testSize = 1024;
    EXPECT_TRUE(processor->allocateAudioBuffer(testSize, "test_buffer"));

    GPUStatus status = processor->getStatus();
    EXPECT_GE(status.utilization, 0.0f);
    EXPECT_LE(status.utilization, 100.0f);

    uint64_t memoryUsage = processor->getMemoryUsage();
    uint64_t totalMemory = processor->getTotalMemory();
    EXPECT_LE(memoryUsage, totalMemory);

    // Test cleanup
    EXPECT_TRUE(processor->deallocateBuffer("test_buffer"));
    processor->shutdown();
    EXPECT_FALSE(processor->isInitialized());
}

// Test GPU processor initialization with OpenCL
TEST_F(GPUProcessorTest, TestOpenCLInitialization) {
    if (!hasOpenCLGPU()) {
        GTEST_SKIP() << "OpenCL GPU not available, skipping test";
    }

    auto processor = GPUProcessorFactory::create(GPUBackend::OPENCL);
    ASSERT_NE(processor, nullptr);

    const uint32_t sampleRate = 44100;
    const uint32_t bufferSize = 512;
    const uint16_t channels = 2;

    EXPECT_TRUE(processor->initialize(GPUBackend::OPENCL, sampleRate, bufferSize, channels));
    EXPECT_TRUE(processor->isInitialized());
    EXPECT_EQ(processor->getCurrentBackend(), GPUBackend::OPENCL);

    auto devices = processor->getAvailableDevices();
    EXPECT_FALSE(devices.empty());

    GPUStatus status = processor->getStatus();
    EXPECT_GE(status.utilization, 0.0f);
    EXPECT_LE(status.utilization, 100.0f);

    processor->shutdown();
    EXPECT_FALSE(processor->isInitialized());
}

// Test GPU processor initialization with Vulkan
TEST_F(GPUProcessorTest, TestVulkanInitialization) {
    if (!hasVulkanGPU()) {
        GTEST_SKIP() << "Vulkan GPU not available, skipping test";
    }

    auto processor = GPUProcessorFactory::create(GPUBackend::VULKAN);
    ASSERT_NE(processor, nullptr);

    const uint32_t sampleRate = 44100;
    const uint32_t bufferSize = 512;
    const uint16_t channels = 2;

    EXPECT_TRUE(processor->initialize(GPUBackend::VULKAN, sampleRate, bufferSize, channels));
    EXPECT_TRUE(processor->isInitialized());
    EXPECT_EQ(processor->getCurrentBackend(), GPUBackend::VULKAN);

    auto devices = processor->getAvailableDevices();
    EXPECT_FALSE(devices.empty());

    processor->shutdown();
    EXPECT_FALSE(processor->isInitialized());
}

// Test GPU audio processing
TEST_F(GPUProcessorTest, TestAudioProcessing) {
    if (!hasCUDAGPU() && !hasOpenCLGPU()) {
        GTEST_SKIP() << "No GPU backend available, skipping test";
    }

    // Try CUDA first, then OpenCL
    std::vector<GPUBackend> backendsToTest;
    if (hasCUDAGPU()) backendsToTest.push_back(GPUBackend::CUDA);
    if (hasOpenCLGPU()) backendsToTest.push_back(GPUBackend::OPENCL);

    for (auto backend : backendsToTest) {
        auto processor = GPUProcessorFactory::create(backend);
        ASSERT_NE(processor, nullptr);

        const uint32_t sampleRate = 44100;
        const uint32_t bufferSize = 512;
        const uint16_t channels = 2;

        EXPECT_TRUE(processor->initialize(backend, sampleRate, bufferSize, channels));
        EXPECT_TRUE(processor->isInitialized());

        // Test basic audio processing
        const size_t numSamples = bufferSize * channels;
        std::vector<float> input(numSamples);
        std::vector<float> output(numSamples);

        // Generate test signal (sine wave)
        for (size_t i = 0; i < numSamples; i += channels) {
            float phase = 2.0f * M_PI * 440.0f * i / (sampleRate * channels);
            input[i] = std::sin(phase);     // Left channel
            input[i+1] = std::sin(phase);   // Right channel
        }

        // Process audio
        EXPECT_TRUE(processor->processAudio(input.data(), output.data(), numSamples));

        // Output should be different from input (processed signal)
        bool different = false;
        for (size_t i = 0; i < numSamples && !different; ++i) {
            if (std::abs(input[i] - output[i]) > 1e-6f) {
                different = true;
            }
        }
        EXPECT_TRUE(different) << "Output should be processed and different from input";

        // Check for NaN or infinite values
        for (size_t i = 0; i < numSamples; ++i) {
            EXPECT_FALSE(std::isnan(output[i])) << "Output contains NaN at index " << i;
            EXPECT_FALSE(std::isinf(output[i])) << "Output contains infinite values at index " << i;
            EXPECT_TRUE(std::isfinite(output[i])) << "Output is not finite at index " << i;
        }

        processor->shutdown();
    }
}

// Test GPU spectrum analysis
TEST_F(GPUProcessorTest, TestSpectrumAnalysis) {
    if (!hasCUDAGPU()) {
        GTEST_SKIP() << "CUDA GPU not available, skipping test";
    }

    auto processor = GPUProcessorFactory::create(GPUBackend::CUDA);
    ASSERT_NE(processor, nullptr);

    const uint32_t sampleRate = 44100;
    const uint32_t bufferSize = 2048; // FFT size
    const uint16_t channels = 2;

    EXPECT_TRUE(processor->initialize(GPUBackend::CUDA, sampleRate, bufferSize, channels));
    EXPECT_TRUE(processor->isInitialized());

    const size_t fftSize = 2048;
    std::vector<float> input(fftSize);
    std::vector<float> spectrum(fftSize / 2);

    // Generate test signal (440 Hz sine wave)
    for (size_t i = 0; i < fftSize; ++i) {
        float phase = 2.0f * M_PI * 440.0f * i / sampleRate;
        input[i] = std::sin(phase);
    }

    EXPECT_TRUE(processor->processAudioSpectrum(input.data(), spectrum.data(), fftSize));

    // Spectrum should have peak at 440 Hz frequency bin
    size_t bin440 = static_cast<size_t>(440.0f * fftSize / sampleRate);
    EXPECT_LT(bin440, spectrum.size());

    // Find peak in spectrum
    auto maxIt = std::max_element(spectrum.begin(), spectrum.end());
    size_t peakBin = std::distance(spectrum.begin(), maxIt);

    // Peak should be near 440 Hz bin (within ±5 bins)
    EXPECT_NEAR(static_cast<int>(peakBin), static_cast<int>(bin440), 5);

    // Spectrum values should be positive
    for (size_t i = 0; i < spectrum.size(); ++i) {
        EXPECT_GE(spectrum[i], 0.0f) << "Spectrum value at bin " << i << " should be non-negative";
        EXPECT_TRUE(std::isfinite(spectrum[i])) << "Spectrum value at bin " << i << " should be finite";
    }

    processor->shutdown();
}

// Test GPU convolution
TEST_F(GPUProcessorTest, TestConvolution) {
    if (!hasCUDAGPU()) {
        GTEST_SKIP() << "CUDA GPU not available, skipping test";
    }

    auto processor = GPUProcessorFactory::create(GPUBackend::CUDA);
    ASSERT_NE(processor, nullptr);

    const uint32_t sampleRate = 44100;
    const uint32_t bufferSize = 512;
    const uint16_t channels = 2;

    EXPECT_TRUE(processor->initialize(GPUBackend::CUDA, sampleRate, bufferSize, channels));
    EXPECT_TRUE(processor->isInitialized());

    const size_t inputSamples = 512;
    const size_t impulseSamples = 128;
    const size_t outputSamples = inputSamples + impulseSamples - 1;

    std::vector<float> input(inputSamples);
    std::vector<float> impulse(impulseSamples);
    std::vector<float> output(outputSamples);

    // Generate test signals
    for (size_t i = 0; i < inputSamples; ++i) {
        input[i] = std::sin(2.0f * M_PI * 1000.0f * i / sampleRate);
    }

    // Create simple impulse response (echo)
    impulse[0] = 1.0f;
    impulse[impulseSamples / 2] = 0.5f;

    EXPECT_TRUE(processor->processAudioConvolution(
        input.data(), impulse.data(), output.data(),
        inputSamples, impulseSamples));

    // Check output validity
    for (size_t i = 0; i < outputSamples; ++i) {
        EXPECT_FALSE(std::isnan(output[i])) << "Convolution output contains NaN at index " << i;
        EXPECT_FALSE(std::isinf(output[i])) << "Convolution output contains infinite values at index " << i;
        EXPECT_TRUE(std::isfinite(output[i])) << "Convolution output is not finite at index " << i;
    }

    // Output should not be all zeros
    bool nonZero = false;
    for (size_t i = 0; i < outputSamples && !nonZero; ++i) {
        if (std::abs(output[i]) > 1e-10f) {
            nonZero = true;
        }
    }
    EXPECT_TRUE(nonZero) << "Convolution output should not be all zeros";

    processor->shutdown();
}

// Test GPU equalization
TEST_F(GPUProcessorTest, TestEqualization) {
    if (!hasCUDAGPU()) {
        GTEST_SKIP() << "CUDA GPU not available, skipping test";
    }

    auto processor = GPUProcessorFactory::create(GPUBackend::CUDA);
    ASSERT_NE(processor, nullptr);

    const uint32_t sampleRate = 44100;
    const uint32_t bufferSize = 512;
    const uint16_t channels = 2;

    EXPECT_TRUE(processor->initialize(GPUBackend::CUDA, sampleRate, bufferSize, channels));
    EXPECT_TRUE(processor->isInitialized());

    const size_t numSamples = bufferSize * channels;
    const size_t numBands = 10;

    std::vector<float> input(numSamples);
    std::vector<float> output(numSamples);
    std::vector<float> frequencies(numBands);
    std::vector<float> gains(numBands);
    std::vector<float> qValues(numBands, 1.0f);

    // Generate test signal (white noise)
    std::srand(42);
    for (size_t i = 0; i < numSamples; ++i) {
        input[i] = (std::rand() / static_cast<float>(RAND_MAX) - 0.5f) * 0.1f;
    }

    // Create EQ bands (logarithmic frequency distribution)
    for (size_t i = 0; i < numBands; ++i) {
        frequencies[i] = 100.0f * std::pow(10.0f, i * 3.0f / numBands); // 100Hz to 10kHz
        gains[i] = (i % 2 == 0) ? 3.0f : -3.0f; // Alternate boost/cut
    }

    EXPECT_TRUE(processor->processAudioEqualization(
        input.data(), output.data(), numSamples,
        frequencies, gains));

    // Check output validity
    for (size_t i = 0; i < numSamples; ++i) {
        EXPECT_FALSE(std::isnan(output[i])) << "Equalizer output contains NaN at index " << i;
        EXPECT_FALSE(std::isinf(output[i])) << "Equalizer output contains infinite values at index " << i;
        EXPECT_TRUE(std::isfinite(output[i])) << "Equalizer output is not finite at index " << i;
    }

    // Output should be different from input
    bool different = false;
    for (size_t i = 0; i < numSamples && !different; ++i) {
        if (std::abs(input[i] - output[i]) > 1e-6f) {
            different = true;
        }
    }
    EXPECT_TRUE(different) << "Equalizer output should be different from input";

    processor->shutdown();
}

// Test GPU memory management
TEST_F(GPUProcessorTest, TestMemoryManagement) {
    if (!hasCUDAGPU()) {
        GTEST_SKIP() << "CUDA GPU not available, skipping test";
    }

    auto processor = GPUProcessorFactory::create(GPUBackend::CUDA);
    ASSERT_NE(processor, nullptr);

    const uint32_t sampleRate = 44100;
    const uint32_t bufferSize = 512;
    const uint16_t channels = 2;

    EXPECT_TRUE(processor->initialize(GPUBackend::CUDA, sampleRate, bufferSize, channels));
    EXPECT_TRUE(processor->isInitialized());

    // Test buffer allocation and deallocation
    std::vector<std::string> bufferNames;
    const size_t numBuffers = 10;
    const size_t bufferSizeBytes = 1024 * 1024; // 1MB per buffer

    // Allocate multiple buffers
    for (size_t i = 0; i < numBuffers; ++i) {
        std::string bufferName = "test_buffer_" + std::to_string(i);
        EXPECT_TRUE(processor->allocateAudioBuffer(bufferSizeBytes, bufferName));
        bufferNames.push_back(bufferName);
    }

    // Check memory usage increased
    uint64_t memoryUsage = processor->getMemoryUsage();
    uint64_t totalMemory = processor->getTotalMemory();
    EXPECT_GT(memoryUsage, bufferSizeBytes * numBuffers);
    EXPECT_LT(memoryUsage, totalMemory);

    // Test memory copying
    if (!bufferNames.empty()) {
        std::vector<float> hostData(bufferSizeBytes / sizeof(float));
        for (size_t i = 0; i < hostData.size(); ++i) {
            hostData[i] = static_cast<float>(i);
        }

        EXPECT_TRUE(processor->copyToDevice(hostData.data(), bufferNames[0], bufferSizeBytes));

        std::vector<float> retrievedData(hostData.size());
        EXPECT_TRUE(processor->copyFromDevice(bufferNames[0], retrievedData.data(), bufferSizeBytes));

        // Verify data integrity
        for (size_t i = 0; i < hostData.size(); ++i) {
            EXPECT_FLOAT_EQ(hostData[i], retrievedData[i]) << "Data mismatch at index " << i;
        }
    }

    // Deallocate all buffers
    for (const auto& bufferName : bufferNames) {
        EXPECT_TRUE(processor->deallocateBuffer(bufferName));
    }

    // Check memory usage decreased
    uint64_t finalMemoryUsage = processor->getMemoryUsage();
    EXPECT_LT(finalMemoryUsage, memoryUsage);

    processor->shutdown();
}

// Test GPU performance optimization
TEST_F(GPUProcessorTest, TestPerformanceOptimization) {
    if (!hasCUDAGPU()) {
        GTEST_SKIP() << "CUDA GPU not available, skipping test";
    }

    auto processor = GPUProcessorFactory::create(GPUBackend::CUDA);
    ASSERT_NE(processor, nullptr);

    const uint32_t sampleRate = 44100;
    const uint32_t bufferSize = 512;
    const uint16_t channels = 2;

    EXPECT_TRUE(processor->initialize(GPUBackend::CUDA, sampleRate, bufferSize, channels));
    EXPECT_TRUE(processor->isInitialized());

    // Test real-time optimization
    EXPECT_TRUE(processor->optimizeForRealtime());

    // Test utilization target setting
    EXPECT_TRUE(processor->setUtilizationTarget(85.0f));

    // Test multi-GPU enable/disable
    EXPECT_TRUE(processor->enableMultiGPU(false));
    EXPECT_TRUE(processor->enableMultiGPU(true));

    processor->shutdown();
}

// Test error handling
TEST_F(GPUProcessorTest, TestErrorHandling) {
    auto processor = GPUProcessorFactory::create(GPUBackend::CUDA);
    ASSERT_NE(processor, nullptr);

    // Test operations on uninitialized processor
    EXPECT_FALSE(processor->isInitialized());

    std::vector<float> input(1024);
    std::vector<float> output(1024);

    EXPECT_FALSE(processor->processAudio(input.data(), output.data(), 1024));
    EXPECT_FALSE(processor->allocateAudioBuffer(1024, "test"));

    // Test invalid parameters
    const uint32_t sampleRate = 44100;
    const uint32_t bufferSize = 512;
    const uint16_t channels = 2;

    // Initialize with valid parameters first
    if (GPUProcessor::isCUDAAvailable()) {
        EXPECT_TRUE(processor->initialize(GPUBackend::CUDA, sampleRate, bufferSize, channels));
        EXPECT_TRUE(processor->isInitialized());

        // Test operations with invalid parameters
        EXPECT_FALSE(processor->processAudio(nullptr, output.data(), 1024));
        EXPECT_FALSE(processor->processAudio(input.data(), nullptr, 1024));
        EXPECT_FALSE(processor->processAudio(input.data(), output.data(), 0));

        // Test invalid buffer operations
        EXPECT_FALSE(processor->allocateAudioBuffer(0, "zero_size"));
        EXPECT_FALSE(processor->allocateAudioBuffer(1024, "")); // Empty name
        EXPECT_FALSE(processor->deallocateBuffer("nonexistent_buffer"));
        EXPECT_FALSE(processor->copyToDevice(nullptr, "test", 1024));
        EXPECT_FALSE(processor->copyFromDevice("test", nullptr, 1024));

        processor->shutdown();
    }
}