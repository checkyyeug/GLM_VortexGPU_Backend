#include <gtest/gtest.h>
#include "core/gpu/gpu_processor.hpp"
#include "core/gpu/cuda_kernels.hpp"
#include "core/audio_buffer_manager.hpp"
#include "testing/audio_test_harness.hpp"
#include "system/logger.hpp"

#include <vector>
#include <chrono>
#include <random>

#ifdef VORTEX_ENABLE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#endif

using namespace vortex;
using namespace vortex::testing;
using namespace std::chrono_literals;

class GPUAccelerationTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::initialize();

        // Check if GPU is available
        gpuAvailable_ = checkGPUAvailability();
        if (!gpuAvailable_) {
            GTEST_SKIP() << "GPU not available, skipping GPU acceleration tests";
        }

        // Initialize test harness
        AudioTestHarness::TestConfiguration config;
        config.sampleRate = 44100.0;
        config.bufferSize = 1024;
        config.channels = 2;
        config.maxProcessingTimeMs = 2.0; // GPU should be faster
        config.enableGPUTests = true;

        harness_.setConfiguration(config);

        // Initialize GPU processor
        gpuProcessor_ = std::make_unique<GPUProcessor>();
        ASSERT_TRUE(gpuProcessor_->initialize("CUDA", config.sampleRate, config.bufferSize, config.channels));

        // Initialize CUDA processor
        cudaProcessor_ = std::make_unique<cuda::CUDAAudioProcessor>();
        ASSERT_TRUE(cudaProcessor_->initialize());
    }

    void TearDown() override {
        if (cudaProcessor_) {
            cudaProcessor_->shutdown();
        }
        if (gpuProcessor_) {
            gpuProcessor_->shutdown();
        }
        Logger::shutdown();
    }

    bool checkGPUAvailability() {
#ifdef VORTEX_ENABLE_CUDA
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        return error == cudaSuccess && deviceCount > 0;
#else
        return false;
#endif
    }

    // Generate test signals
    std::vector<float> generateSineWave(double frequency, double durationSeconds, double amplitude = 1.0) {
        int numSamples = static_cast<int>(44100.0 * durationSeconds);
        std::vector<float> signal(numSamples);

        for (int i = 0; i < numSamples; ++i) {
            double time = i / 44100.0;
            signal[i] = static_cast<float>(amplitude * std::sin(2.0 * M_PI * frequency * time));
        }

        return signal;
    }

    std::vector<float> generateStereoSignal(const std::vector<float>& mono) {
        std::vector<float> stereo(mono.size() * 2);
        for (size_t i = 0; i < mono.size(); ++i) {
            stereo[i * 2] = mono[i];      // Left channel
            stereo[i * 2 + 1] = mono[i] * 0.5f;  // Right channel (different level)
        }
        return stereo;
    }

    std::vector<float> generateImpulseResponse(size_t length, double sampleRate = 44100.0) {
        std::vector<float> ir(length);

        // Create a simple impulse response (reverb tail)
        for (size_t i = 0; i < length; ++i) {
            double t = i / sampleRate;
            // Exponential decay with early reflection
            if (i == 0) {
                ir[i] = 1.0f; // Direct sound
            } else if (i == static_cast<size_t>(sampleRate * 0.03)) { // 30ms early reflection
                ir[i] = 0.6f;
            } else {
                ir[i] = static_cast<float>(std::exp(-t * 2.0) * std::sin(2.0 * M_PI * 1000.0 * t) * 0.1f);
            }
        }

        return ir;
    }

    AudioTestHarness harness_;
    std::unique_ptr<GPUProcessor> gpuProcessor_;
    std::unique_ptr<cuda::CUDAAudioProcessor> cudaProcessor_;
    bool gpuAvailable_ = false;
};

// Test GPU processor initialization
TEST_F(GPUAccelerationTest, GPUProcessorInitialization) {
    ASSERT_TRUE(gpuProcessor_->isInitialized());
    EXPECT_EQ(gpuProcessor_->getBackend(), "CUDA");

    // Query device information
    size_t deviceMemory = gpuProcessor_->getDeviceMemorySize();
    EXPECT_GT(deviceMemory, 1024 * 1024 * 1024); // At least 1GB

    size_t availableMemory = gpuProcessor_->getAvailableMemory();
    EXPECT_GT(availableMemory, deviceMemory / 2); // At least 50% available

    double utilization = gpuProcessor_->getUtilization();
    EXPECT_GE(utilization, 0.0);
    EXPECT_LE(utilization, 1.0);

    std::cout << "GPU Device Memory: " << deviceMemory / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Available Memory: " << availableMemory / (1024 * 1024) << " MB" << std::endl;
    std::cout << "GPU Utilization: " << (utilization * 100.0) << "%" << std::endl;
}

// Test CUDA gain processing
TEST_F(GPUAccelerationTest, CUDAGainProcessing) {
    // Generate test signal
    auto inputSignal = generateSineWave(1000.0, 1.0); // 1 second at 1kHz
    float gain = 0.5f; // -6dB

    std::vector<float> gpuOutput(inputSignal.size());
    std::vector<float> cpuOutput(inputSignal.size());

    // Process on GPU
    auto gpuStart = high_resolution_clock::now();
    ASSERT_TRUE(cudaProcessor_->applyGain(inputSignal.data(), gpuOutput.data(), gain, inputSignal.size()));
    auto gpuEnd = high_resolution_clock::now();

    // Process on CPU for comparison
    auto cpuStart = high_resolution_clock::now();
    std::transform(inputSignal.begin(), inputSignal.end(), cpuOutput.begin(),
                   [gain](float sample) { return sample * gain; });
    auto cpuEnd = high_resolution_clock::now();

    auto gpuTime = duration_cast<microseconds>(gpuEnd - gpuStart);
    auto cpuTime = duration_cast<microseconds>(cpuEnd - cpuStart);

    // Verify results match CPU implementation
    float maxDifference = 0.0f;
    for (size_t i = 0; i < inputSignal.size(); ++i) {
        float diff = std::abs(gpuOutput[i] - cpuOutput[i]);
        maxDifference = std::max(maxDifference, diff);
    }

    EXPECT_LT(maxDifference, 1e-6f) << "GPU and CPU results differ by " << maxDifference;

    // GPU should be faster for large buffers
    if (inputSignal.size() > 10000) {
        EXPECT_LT(gpuTime.count(), cpuTime.count())
            << "GPU processing (" << gpuTime.count() << "μs) should be faster than CPU (" << cpuTime.count() << "μs)";
    }

    std::cout << "GPU gain processing: " << gpuTime.count() << "μs" << std::endl;
    std::cout << "CPU gain processing: " << cpuTime.count() << "μs" << std::endl;
    std::cout << "Speedup: " << (static_cast<double>(cpuTime.count()) / gpuTime.count()) << "x" << std::endl;
}

// Test CUDA convolution
TEST_F(GPUAccelerationTest, CUDAConvolution) {
    // Generate test signals
    auto inputSignal = generateSineWave(440.0, 2.0); // 2 seconds at A4
    auto impulseResponse = generateImpulseResponse(8192); // 8192 samples

    std::vector<float> gpuOutput(inputSignal.size() + impulseResponse.size() - 1);
    std::vector<float> cpuOutput(gpuOutput.size());

    // Process on GPU
    auto gpuStart = high_resolution_clock::now();
    ASSERT_TRUE(cudaProcessor_->convolve(inputSignal.data(), impulseResponse.data(),
                                           gpuOutput.data(), inputSignal.size(), impulseResponse.size()));
    auto gpuEnd = high_resolution_clock::now();

    // Simple CPU convolution for comparison (slower but for validation)
    auto cpuStart = high_resolution_clock::now();
    for (size_t i = 0; i < gpuOutput.size(); ++i) {
        cpuOutput[i] = 0.0f;
        for (size_t j = 0; j < impulseResponse.size() && j <= i && j < inputSignal.size(); ++j) {
            cpuOutput[i] += inputSignal[i - j] * impulseResponse[j];
        }
    }
    auto cpuEnd = high_resolution_clock::now();

    auto gpuTime = duration_cast<microseconds>(gpuEnd - gpuStart);
    auto cpuTime = duration_cast<microseconds>(cpuEnd - cpuStart);

    // Verify results (allow small numerical differences)
    float maxDifference = 0.0f;
    float sumDifference = 0.0f;
    for (size_t i = 0; i < gpuOutput.size(); ++i) {
        float diff = std::abs(gpuOutput[i] - cpuOutput[i]);
        maxDifference = std::max(maxDifference, diff);
        sumDifference += diff;
    }

    float avgDifference = sumDifference / gpuOutput.size();

    EXPECT_LT(avgDifference, 1e-4f) << "Average difference too large: " << avgDifference;
    EXPECT_LT(maxDifference, 1e-3f) << "Maximum difference too large: " << maxDifference;

    // GPU should be significantly faster for convolution
    EXPECT_LT(gpuTime.count(), cpuTime.count() / 2)
        << "GPU convolution should be at least 2x faster than CPU";

    std::cout << "GPU convolution: " << gpuTime.count() << "μs" << std::endl;
    std::cout << "CPU convolution: " << cpuTime.count() << "μs" << std::endl;
    std::cout << "Speedup: " << (static_cast<double>(cpuTime.count()) / gpuTime.count()) << "x" << std::endl;
}

// Test CUDA FFT processing
TEST_F(GPUAccelerationTest, CUDAFFTProcessing) {
    // Generate test signal
    auto inputSignal = generateSineWave(1000.0, 0.1); // 100ms at 1kHz
    size_t fftSize = 2048;

    std::vector<float> gpuMagnitude(fftSize / 2 + 1);
    std::vector<float> cpuMagnitude(fftSize / 2 + 1);

    // Pad input to FFT size
    std::vector<float> paddedInput(fftSize, 0.0f);
    size_t copySize = std::min(inputSignal.size(), fftSize);
    std::copy(inputSignal.begin(), inputSignal.begin() + copySize, paddedInput.begin());

    // Process on GPU
    auto gpuStart = high_resolution_clock::now();
    ASSERT_TRUE(cudaProcessor_->computeFFT(paddedInput.data(), gpuMagnitude.data(), fftSize));
    auto gpuEnd = high_resolution_clock::now();

    // Process on CPU using reference implementation (JUCE would be used in practice)
    auto cpuStart = high_resolution_clock::now();
    // Simple FFT magnitude calculation (simplified reference)
    for (size_t i = 0; i < fftSize / 2 + 1; ++i) {
        cpuMagnitude[i] = std::abs(paddedInput[i]); // Simplified - real FFT
    }
    auto cpuEnd = high_resolution_clock::now();

    auto gpuTime = duration_cast<microseconds>(gpuEnd - gpuStart);
    auto cpuTime = duration_cast<microseconds>(cpuEnd - cpuStart);

    // Verify FFT properties
    EXPECT_GT(gpuMagnitude.size(), 0);
    EXPECT_EQ(gpuMagnitude.size(), fftSize / 2 + 1);

    // Look for peak at 1kHz
    size_t binWidth = 44100 / fftSize;
    size_t expectedBin = static_cast<size_t>(1000.0 / binWidth);

    bool foundPeak = false;
    for (size_t i = std::max(static_cast<size_t>(0), expectedBin - 2);
         i < std::min(gpuMagnitude.size(), expectedBin + 3); ++i) {
        if (gpuMagnitude[i] > 0.1f) { // Threshold
            foundPeak = true;
            break;
        }
    }
    EXPECT_TRUE(foundPeak) << "FFT should show peak at 1kHz";

    // GPU FFT should be faster
    EXPECT_LT(gpuTime.count(), cpuTime.count())
        << "GPU FFT should be faster than CPU";

    std::cout << "GPU FFT: " << gpuTime.count() << "μs" << std::endl;
    std::cout << "CPU FFT: " << cpuTime.count() << "μs" << std::endl;
}

// Test GPU dynamics processing
TEST_F(GPUAccelerationTest, CUDADynamicsProcessing) {
    // Generate test signal with various amplitudes
    auto inputSignal = generateSineWave(440.0, 1.0);

    // Add dynamic range to test compressor
    for (size_t i = 0; i < inputSignal.size(); ++i) {
        double time = i / 44100.0;
        float envelope = static_cast<float>(0.5 + 0.5 * std::sin(2.0 * M_PI * 2.0 * time));
        inputSignal[i] *= envelope;
    }

    std::vector<float> gpuOutput(inputSignal.size());
    std::vector<float> cpuOutput(inputSignal.size());

    // Dynamics parameters
    float threshold = 0.1f;
    float ratio = 4.0f;
    float attackTime = 0.01f;
    float releaseTime = 0.1f;
    float sampleRate = 44100.0f;

    // Process on GPU
    auto gpuStart = high_resolution_clock::now();
    ASSERT_TRUE(cudaProcessor_->applyDynamics(inputSignal.data(), gpuOutput.data(),
                                            threshold, ratio, attackTime, releaseTime,
                                            sampleRate, inputSignal.size()));
    auto gpuEnd = high_resolution_clock::now();

    // Simple CPU dynamics processing for comparison
    auto cpuStart = high_resolution_clock::now();
    float envelope = 0.0f;
    float attackCoeff = std::exp(-1.0f / (attackTime * sampleRate));
    float releaseCoeff = std::exp(-1.0f / (releaseTime * sampleRate));

    for (size_t i = 0; i < inputSignal.size(); ++i) {
        float inputLevel = std::abs(inputSignal[i]);
        float targetLevel = (inputLevel > threshold) ? threshold : inputLevel;
        envelope = targetLevel + (envelope - targetLevel) *
                   ((inputLevel > threshold) ? attackCoeff : releaseCoeff);

        float gain = (envelope > threshold) ?
                    (threshold + (envelope - threshold) / ratio) / envelope : 1.0f;

        cpuOutput[i] = inputSignal[i] * gain;
    }
    auto cpuEnd = high_resolution_clock::now();

    auto gpuTime = duration_cast<microseconds>(gpuEnd - gpuStart);
    auto cpuTime = duration_cast<microseconds>(cpuEnd - cpuStart);

    // Verify dynamics processing (output should have reduced dynamic range)
    float inputRMS = 0.0f, outputRMS = 0.0f;
    for (size_t i = 0; i < inputSignal.size(); ++i) {
        inputRMS += inputSignal[i] * inputSignal[i];
        outputRMS += gpuOutput[i] * gpuOutput[i];
    }
    inputRMS = std::sqrt(inputRMS / inputSignal.size());
    outputRMS = std::sqrt(outputRMS / gpuOutput.size());

    // Output RMS should be lower than input RMS due to compression
    EXPECT_LT(outputRMS, inputRMS);

    // GPU should be faster for large buffers
    if (inputSignal.size() > 10000) {
        EXPECT_LT(gpuTime.count(), cpuTime.count())
            << "GPU dynamics should be faster than CPU";
    }

    std::cout << "Input RMS: " << inputRMS << std::endl;
    std::cout << "Output RMS: " << outputRMS << std::endl;
    std::cout << "GPU dynamics: " << gpuTime.count() << "μs" << std::endl;
    std::cout << "CPU dynamics: " << cpuTime.count() << "μs" << std::endl;
}

// Test GPU memory management
TEST_F(GPUAccelerationTest, GPUMemoryManagement) {
    // Test memory allocation and deallocation
    auto memoryManager = gpuProcessor_->getMemoryManager();
    ASSERT_NE(memoryManager, nullptr);

    // Allocate various sized buffers
    std::vector<void*> allocations;
    std::vector<size_t> sizes = {1024, 4096, 16384, 65536, 262144, 1048576}; // 1KB to 1MB

    for (size_t size : sizes) {
        void* ptr = memoryManager->allocate(size);
        EXPECT_NE(ptr, nullptr) << "Failed to allocate " << size << " bytes";
        allocations.push_back(ptr);
    }

    // Check memory usage
    size_t usedMemory = memoryManager->getUsedMemory();
    EXPECT_GT(usedMemory, 0);

    size_t totalMemory = memoryManager->getTotalMemory();
    EXPECT_GE(totalMemory, usedMemory);

    double utilization = memoryManager->getUtilization();
    EXPECT_GE(utilization, 0.0);
    EXPECT_LE(utilization, 1.0);

    std::cout << "GPU Memory Usage: " << usedMemory / (1024 * 1024) << " MB" << std::endl;
    std::cout << "GPU Total Memory: " << totalMemory / (1024 * 1024) << " MB" << std::endl;
    std::cout << "GPU Memory Utilization: " << (utilization * 100.0) << "%" << std::endl;

    // Deallocate memory
    for (void* ptr : allocations) {
        memoryManager->deallocate(ptr);
    }

    // Memory should be freed
    size_t finalUsedMemory = memoryManager->getUsedMemory();
    EXPECT_LT(finalUsedMemory, usedMemory);
}

// Test GPU vs CPU performance comparison
TEST_F(GPUAccelerationTest, GPUvsCPUPerformanceComparison) {
    const int numIterations = 100;
    std::vector<double> speedupRatios;

    // Test different buffer sizes
    std::vector<size_t> bufferSizes = {1024, 4096, 16384, 65536}; // 1KB to 64KB

    for (size_t bufferSize : bufferSizes) {
        auto testSignal = generateSineWave(1000.0, bufferSize / 44100.0);
        std::vector<float> gpuOutput(testSignal.size());
        std::vector<float> cpuOutput(testSignal.size());

        // GPU timing
        auto gpuStart = high_resolution_clock::now();
        for (int i = 0; i < numIterations; ++i) {
            cudaProcessor_->applyGain(testSignal.data(), gpuOutput.data(), 0.5f, testSignal.size());
        }
        auto gpuEnd = high_resolution_clock::now();

        // CPU timing
        auto cpuStart = high_resolution_clock::now();
        for (int i = 0; i < numIterations; ++i) {
            std::transform(testSignal.begin(), testSignal.end(), cpuOutput.begin(),
                           [](float s) { return s * 0.5f; });
        }
        auto cpuEnd = high_resolution_clock::now();

        auto gpuTime = duration_cast<microseconds>(gpuEnd - gpuStart);
        auto cpuTime = duration_cast<microseconds>(cpuEnd - cpuStart);

        double speedup = static_cast<double>(cpuTime.count()) / gpuTime.count();
        speedupRatios.push_back(speedup);

        std::cout << "Buffer Size: " << bufferSize
                  << ", GPU: " << gpuTime.count() << "μs"
                  << ", CPU: " << cpuTime.count() << "μs"
                  << ", Speedup: " << speedup << "x" << std::endl;

        // For larger buffers, GPU should show significant speedup
        if (bufferSize >= 4096) {
            EXPECT_GT(speedup, 1.0) << "GPU should be faster for buffer size " << bufferSize;
        }
    }

    // Calculate average speedup
    double avgSpeedup = std::accumulate(speedupRatios.begin(), speedupRatios.end(), 0.0) / speedupRatios.size();
    std::cout << "Average GPU Speedup: " << avgSpeedup << "x" << std::endl;

    // GPU should provide overall speedup
    EXPECT_GT(avgSpeedup, 1.0) << "GPU should provide overall performance benefit";
}

// Test real-time constraints with GPU
TEST_F(GPUAccelerationTest, GPURealTimeConstraints) {
    // Test with real-time buffer sizes (512-4096 samples at 44.1kHz)
    std::vector<size_t> realTimeBufferSizes = {512, 1024, 2048, 4096};

    for (size_t bufferSize : realTimeBufferSizes) {
        double bufferTimeMs = (bufferSize / 44100.0) * 1000.0;
        double maxProcessingTimeMs = bufferTimeMs * 0.5; // Should process faster than half buffer time

        auto testSignal = generateSineWave(1000.0, bufferTimeMs / 1000.0);
        std::vector<float> output(testSignal.size());

        auto start = high_resolution_clock::now();
        cudaProcessor_->applyGain(testSignal.data(), output.data(), 1.0f, testSignal.size());
        auto end = high_resolution_clock::now();

        auto processingTime = duration_cast<microseconds>(end - start).count() / 1000.0;

        EXPECT_LT(processingTime, maxProcessingTimeMs)
            << "GPU processing time " << processingTime << "ms exceeds limit " << maxProcessingTimeMs
            << "ms for buffer size " << bufferSize;

        std::cout << "Buffer: " << bufferSize
                  << ", Buffer Time: " << bufferTimeMs << "ms"
                  << ", GPU Processing: " << processingTime << "ms"
                  << ", Limit: " << maxProcessingTimeMs << "ms" << std::endl;
    }
}