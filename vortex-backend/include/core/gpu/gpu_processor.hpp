#pragma once

#include <string>
#include <vector>
#include <memory>
#include <atomic>

#ifdef VORTEX_ENABLE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#endif

#ifdef VORTEX_ENABLE_OPENCL
#include <CL/cl.h>
#endif

#ifdef VORTEX_ENABLE_VULKAN
#include <vulkan/vulkan.hpp>
#endif

namespace vortex {

class GPUMemoryManager;

/**
 * @brief GPU processor interface for audio processing acceleration
 *
 * This class provides a unified interface for GPU-accelerated audio processing
 * across multiple GPU backends (CUDA, OpenCL, Vulkan). It handles device management,
 * memory allocation, stream synchronization, and provides high-level audio
 * processing functions optimized for GPU execution.
 */
class GPUProcessor {
public:
    GPUProcessor();
    ~GPUProcessor();

    // Initialization
    bool initialize(const std::string& backend, double sampleRate, int bufferSize, int channels);
    void shutdown();

    // Backend support
    bool isBackendSupported(const std::string& backend) const;

    // Processing methods
    void beginProcessing(size_t numSamples, int channels);
    void endProcessing();

    bool processAudio(float* audioData, size_t numSamples, int channels);
    bool processFFT(float* input, float* output, size_t fftSize);
    bool processConvolution(float* input, float* output, float* ir,
                           size_t inputSize, size_t irSize, size_t outputSize);
    bool processEqualizer(float* audio, const float* frequencies,
                         const float* gains, const float* qValues,
                         size_t numBands, size_t numSamples, int channels);

    // Memory management
    void* allocateGPUMemory(size_t size);
    void deallocateGPUMemory(void* ptr);
    bool copyToGPU(const void* hostPtr, void* devicePtr, size_t size);
    bool copyFromGPU(const void* devicePtr, void* hostPtr, size_t size);

    // Device information
    void queryDeviceInfo();
    size_t getDeviceMemorySize() const;
    size_t getAvailableMemory() const;
    std::string getDeviceName() const;

    // Getters
    bool isInitialized() const;
    const std::string& getBackend() const;
    GPUMemoryManager* getMemoryManager() const;

private:
    // Core state
    bool initialized_;
    std::string currentBackend_;
    double sampleRate_;
    int bufferSize_;
    int channels_;
    int deviceId_;

    // Memory management
    std::unique_ptr<GPUMemoryManager> memoryManager_;

    // Processing streams
    std::vector<void*> streams_;
    int streamCount_;
    std::atomic<int> currentStream_;

    // Backend-specific handles
#ifdef VORTEX_ENABLE_CUDA
    cublasHandle_t cublasHandle_ = nullptr;
    cufftHandle cufftPlan_ = nullptr;
#endif

#ifdef VORTEX_ENABLE_OPENCL
    cl_platform_id clPlatform_ = nullptr;
    cl_device_id clDevice_ = nullptr;
    cl_context clContext_ = nullptr;
    cl_command_queue clCommandQueue_ = nullptr;
#endif

#ifdef VORTEX_ENABLE_VULKAN
    vk::Instance vulkanInstance_;
    vk::PhysicalDevice vulkanPhysicalDevice_;
    vk::Device vulkanDevice_;
    vk::Queue vulkanQueue_;
    vk::CommandPool vulkanCommandPool_;
#endif

    // Backend initialization
    bool initializeBackend(const std::string& backend);
    void shutdownBackend();

    // CUDA backend
    bool checkCUDASupport() const;
    bool initializeCUDABackend();
    void shutdownCUDABackend();

    // OpenCL backend
    bool checkOpenCLSupport() const;
    bool initializeOpenCLBackend();
    void shutdownOpenCLBackend();

    // Vulkan backend
    bool checkVulkanSupport() const;
    bool initializeVulkanBackend();
    void shutdownVulkanBackend();

    // Stream management
    bool initializeStreams();
    void shutdownStreams();
    void* createStream();
    void destroyStream(void* stream);

    // Kernel initialization
    bool initializeKernels();
    void shutdownKernels();
    bool initializeFFTKernels();
    bool initializeAudioKernels();
    bool initializeConvolutionKernels();

    // Backend-specific processing
#ifdef VORTEX_ENABLE_CUDA
    bool processFFTCUDA(float* input, float* output, size_t fftSize);
    bool processAudioCUDA(float* audio, size_t numSamples, int channels);
    bool processConvolutionCUDA(float* input, float* output, float* ir,
                               size_t inputSize, size_t irSize, size_t outputSize);
    bool processEqualizerCUDA(float* audio, const float* frequencies,
                             const float* gains, const float* qValues,
                             size_t numBands, size_t numSamples, int channels);
    void queryCUDADeviceInfo();
#endif

#ifdef VORTEX_ENABLE_OPENCL
    bool processFFTOpenCL(float* input, float* output, size_t fftSize);
    bool processAudioOpenCL(float* audio, size_t numSamples, int channels);
    bool processConvolutionOpenCL(float* input, float* output, float* ir,
                                 size_t inputSize, size_t irSize, size_t outputSize);
    bool processEqualizerOpenCL(float* audio, const float* frequencies,
                               const float* gains, const float* qValues,
                               size_t numBands, size_t numSamples, int channels);
    void queryOpenCLDeviceInfo();
#endif

#ifdef VORTEX_ENABLE_VULKAN
    bool processFFTVulkan(float* input, float* output, size_t fftSize);
    bool processAudioVulkan(float* audio, size_t numSamples, int channels);
    bool processConvolutionVulkan(float* input, float* output, float* ir,
                                 size_t inputSize, size_t irSize, size_t outputSize);
    bool processEqualizerVulkan(float* audio, const float* frequencies,
                               const float* gains, const float* qValues,
                               size_t numBands, size_t numSamples, int channels);
    void queryVulkanDeviceInfo();
#endif

    // Helper methods
    void* getCurrentStream() const;
    bool synchronizeStream(void* stream);
    bool synchronizeAllStreams();
};

} // namespace vortex