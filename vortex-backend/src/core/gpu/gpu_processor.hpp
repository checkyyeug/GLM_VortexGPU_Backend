#pragma once

#include "audio_types.hpp"
#include "network_types.hpp"

#include <memory>
#include <vector>
#include <string>
#include <mutex>
#include <atomic>

namespace vortex {

/**
 * @brief GPU processor base class for hardware-accelerated audio processing
 *
 * This class provides the interface for GPU-accelerated audio processing
 * supporting CUDA, OpenCL, and Vulkan backends with <10ms latency
 * requirements and >80% GPU utilization targets.
 */
class GPUProcessor {
public:
    GPUProcessor();
    virtual ~GPUProcessor();

    // Lifecycle management
    virtual bool initialize(GPUBackend backend, uint32_t sampleRate, uint32_t bufferSize, uint16_t channels) = 0;
    virtual void shutdown() = 0;
    virtual bool isInitialized() const = 0;

    // Backend management
    virtual GPUBackend getCurrentBackend() const = 0;
    virtual std::vector<GPUBackend> getSupportedBackends() const = 0;
    virtual bool switchBackend(GPUBackend newBackend) = 0;

    // Device management
    virtual std::vector<std::string> getAvailableDevices() const = 0;
    virtual bool selectDevice(const std::string& deviceId) = 0;
    virtual std::string getCurrentDevice() const = 0;

    // Audio processing operations
    virtual bool processAudio(float* input, float* output, size_t numSamples) = 0;
    virtual bool processAudioSpectrum(float* input, float* spectrum, size_t numSamples) = 0;
    virtual bool processAudioConvolution(float* input, float* impulseResponse, float* output,
                                         size_t inputSamples, size_t impulseSamples) = 0;
    virtual bool processAudioEqualization(float* input, float* output, size_t numSamples,
                                          const std::vector<float>& frequencies,
                                          const std::vector<float>& gains) = 0;

    // Memory management
    virtual bool allocateAudioBuffer(size_t size, const std::string& name) = 0;
    virtual bool deallocateBuffer(const std::string& name) = 0;
    virtual bool copyToDevice(const float* hostData, const std::string& deviceBuffer, size_t size) = 0;
    virtual bool copyFromDevice(const std::string& deviceBuffer, float* hostData, size_t size) = 0;

    // Status and monitoring
    virtual GPUStatus getStatus() const = 0;
    virtual uint64_t getMemoryUsage() const = 0;
    virtual uint64_t getTotalMemory() const = 0;
    virtual float getUtilization() const = 0;
    virtual float getTemperature() const = 0;

    // Performance optimization
    virtual bool optimizeForRealtime() = 0;
    virtual bool setUtilizationTarget(float targetPercentage) = 0;
    virtual bool enableMultiGPU(bool enable) = 0;

    // Shader management
    virtual bool loadComputeShader(const std::string& shaderPath, const std::string& shaderName) = 0;
    virtual bool executeShader(const std::string& shaderName, const std::vector<void*>& parameters) = 0;

    // Utility functions
    static bool isCUDAAvailable();
    static bool isOpenCLAvailable();
    static bool isVulkanAvailable();

protected:
    // Common GPU operations
    virtual bool initializeDevice() = 0;
    virtual void shutdownDevice() = 0;
    virtual bool validateDevice() = 0;

    // Memory helpers
    struct GPUBuffer {
        std::string name;
        void* devicePtr = nullptr;
        size_t size = 0;
        bool isAllocated = false;
    };

    std::map<std::string, GPUBuffer> m_buffers;
    mutable std::mutex m_bufferMutex;

    // Device state
    GPUBackend m_currentBackend = GPUBackend::NONE;
    std::string m_currentDeviceId;
    bool m_initialized = false;
    bool m_multiGPUEnabled = false;
    float m_utilizationTarget = 80.0f;

    // Performance tracking
    std::atomic<float> m_currentUtilization{0.0f};
    std::atomic<uint64_t> m_memoryUsed{0};
    std::atomic<float> m_currentTemperature{0.0f};
};

/**
 * @brief CUDA implementation of GPU processor
 */
class CudaProcessor : public GPUProcessor {
public:
    CudaProcessor();
    ~CudaProcessor() override;

    bool initialize(GPUBackend backend, uint32_t sampleRate, uint32_t bufferSize, uint16_t channels) override;
    void shutdown() override;
    bool isInitialized() const override;

    GPUBackend getCurrentBackend() const override { return GPUBackend::CUDA; }
    std::vector<GPUBackend> getSupportedBackends() const override;
    bool switchBackend(GPUBackend newBackend) override;

    std::vector<std::string> getAvailableDevices() const override;
    bool selectDevice(const std::string& deviceId) override;
    std::string getCurrentDevice() const override;

    bool processAudio(float* input, float* output, size_t numSamples) override;
    bool processAudioSpectrum(float* input, float* spectrum, size_t numSamples) override;
    bool processAudioConvolution(float* input, float* impulseResponse, float* output,
                                size_t inputSamples, size_t impulseSamples) override;
    bool processAudioEqualization(float* input, float* output, size_t numSamples,
                                 const std::vector<float>& frequencies,
                                 const std::vector<float>& gains) override;

    bool allocateAudioBuffer(size_t size, const std::string& name) override;
    bool deallocateBuffer(const std::string& name) override;
    bool copyToDevice(const float* hostData, const std::string& deviceBuffer, size_t size) override;
    bool copyFromDevice(const std::string& deviceBuffer, float* hostData, size_t size) override;

    GPUStatus getStatus() const override;
    uint64_t getMemoryUsage() const override;
    uint64_t getTotalMemory() const override;
    float getUtilization() const override;
    float getTemperature() const override;

    bool optimizeForRealtime() override;
    bool setUtilizationTarget(float targetPercentage) override;
    bool enableMultiGPU(bool enable) override;

    bool loadComputeShader(const std::string& shaderPath, const std::string& shaderName) override;
    bool executeShader(const std::string& shaderName, const std::vector<void*>& parameters) override;

private:
    bool initializeCudaDevice();
    void shutdownCudaDevice();
    bool validateCudaDevice();

    // CUDA device properties
    int m_cudaDeviceId = -1;
    size_t m_totalDeviceMemory = 0;
    std::string m_deviceName;

    // CUDA streams for concurrent processing
    std::vector<cudaStream_t> m_streams;
    static constexpr size_t NUM_STREAMS = 4;

    // CUDA kernels (would be implemented in .cu files)
    bool launchAudioProcessingKernel(const float* input, float* output, size_t numSamples);
    bool launchSpectrumKernel(const float* input, float* spectrum, size_t numSamples);
    bool launchConvolutionKernel(const float* input, const float* impulse, float* output,
                                 size_t inputSamples, size_t impulseSamples);
    bool launchEqualizerKernel(const float* input, float* output, size_t numSamples,
                              const float* frequencies, const float* gains, size_t numBands);

    // Performance optimization
    bool m_pinnedMemoryEnabled = false;
    bool m_peerToPeerEnabled = false;
};

/**
 * @brief OpenCL implementation of GPU processor
 */
class OpenCLProcessor : public GPUProcessor {
public:
    OpenCLProcessor();
    ~OpenCLProcessor() override;

    bool initialize(GPUBackend backend, uint32_t sampleRate, uint32_t bufferSize, uint16_t channels) override;
    void shutdown() override;
    bool isInitialized() const override;

    GPUBackend getCurrentBackend() const override { return GPUBackend::OPENCL; }
    std::vector<GPUBackend> getSupportedBackends() const override;
    bool switchBackend(GPUBackend newBackend) override;

    std::vector<std::string> getAvailableDevices() const override;
    bool selectDevice(const std::string& deviceId) override;
    std::string getCurrentDevice() const override;

    bool processAudio(float* input, float* output, size_t numSamples) override;
    bool processAudioSpectrum(float* input, float* spectrum, size_t numSamples) override;
    bool processAudioConvolution(float* input, float* impulseResponse, float* output,
                                size_t inputSamples, size_t impulseSamples) override;
    bool processAudioEqualization(float* input, float* output, size_t numSamples,
                                 const std::vector<float>& frequencies,
                                 const std::vector<float>& gains) override;

    bool allocateAudioBuffer(size_t size, const std::string& name) override;
    bool deallocateBuffer(const std::string& name) override;
    bool copyToDevice(const float* hostData, const std::string& deviceBuffer, size_t size) override;
    bool copyFromDevice(const std::string& deviceBuffer, float* hostData, size_t size) override;

    GPUStatus getStatus() const override;
    uint64_t getMemoryUsage() const override;
    uint64_t getTotalMemory() const override;
    float getUtilization() const override;
    float getTemperature() const override;

    bool optimizeForRealtime() override;
    bool setUtilizationTarget(float targetPercentage) override;
    bool enableMultiGPU(bool enable) override;

    bool loadComputeShader(const std::string& shaderPath, const std::string& shaderName) override;
    bool executeShader(const std::string& shaderName, const std::vector<void*>& parameters) override;

private:
    bool initializeOpenCLDevice();
    void shutdownOpenCLDevice();
    bool validateOpenCLDevice();

    // OpenCL context and resources
    cl_platform_id m_platform = nullptr;
    cl_device_id m_device = nullptr;
    cl_context m_context = nullptr;
    cl_command_queue m_queue = nullptr;
    cl_program m_program = nullptr;

    // OpenCL kernels
    std::map<std::string, cl_kernel> m_kernels;

    // Device properties
    std::string m_deviceName;
    size_t m_totalDeviceMemory = 0;
    cl_uint m_computeUnits = 0;
};

/**
 * @brief Vulkan implementation of GPU processor
 */
class VulkanProcessor : public GPUProcessor {
public:
    VulkanProcessor();
    ~VulkanProcessor() override;

    bool initialize(GPUBackend backend, uint32_t sampleRate, uint32_t bufferSize, uint16_t channels) override;
    void shutdown() override;
    bool isInitialized() const override;

    GPUBackend getCurrentBackend() const override { return GPUBackend::VULKAN; }
    std::vector<GPUBackend> getSupportedBackends() const override;
    bool switchBackend(GPUBackend newBackend) override;

    std::vector<std::string> getAvailableDevices() const override;
    bool selectDevice(const std::string& deviceId) override;
    std::string getCurrentDevice() const override;

    bool processAudio(float* input, float* output, size_t numSamples) override;
    bool processAudioSpectrum(float* input, float* spectrum, size_t numSamples) override;
    bool processAudioConvolution(float* input, float* impulseResponse, float* output,
                                size_t inputSamples, size_t impulseSamples) override;
    bool processAudioEqualization(float* input, float* output, size_t numSamples,
                                 const std::vector<float>& frequencies,
                                 const std::vector<float>& gains) override;

    bool allocateAudioBuffer(size_t size, const std::string& name) override;
    bool deallocateBuffer(const std::string& name) override;
    bool copyToDevice(const float* hostData, const std::string& deviceBuffer, size_t size) override;
    bool copyFromDevice(const std::string& deviceBuffer, float* hostData, size_t size) override;

    GPUStatus getStatus() const override;
    uint64_t getMemoryUsage() const override;
    uint64_t getTotalMemory() const override;
    float getUtilization() const override;
    float getTemperature() const override;

    bool optimizeForRealtime() override;
    bool setUtilizationTarget(float targetPercentage) override;
    bool enableMultiGPU(bool enable) override;

    bool loadComputeShader(const std::string& shaderPath, const std::string& shaderName) override;
    bool executeShader(const std::string& shaderName, const std::vector<void*>& parameters) override;

private:
    bool initializeVulkanDevice();
    void shutdownVulkanDevice();
    bool validateVulkanDevice();

    // Vulkan resources
    vk::Instance m_instance;
    vk::PhysicalDevice m_physicalDevice;
    vk::Device m_device;
    vk::Queue m_computeQueue;
    vk::CommandPool m_commandPool;
    vk::DescriptorPool m_descriptorPool;

    // Device properties
    std::string m_deviceName;
    size_t m_totalDeviceMemory = 0;
    uint32_t m_computeQueueFamily = 0;

    // Vulkan compute pipelines
    std::map<std::string, vk::Pipeline> m_pipelines;
    std::map<std::string, vk::PipelineLayout> m_pipelineLayouts;
};

/**
 * @brief Factory for creating appropriate GPU processor
 */
class GPUProcessorFactory {
public:
    static std::unique_ptr<GPUProcessor> create(GPUBackend backend);
    static std::vector<GPUBackend> getAvailableBackends();
    static GPUBackend getRecommendedBackend();
};

} // namespace vortex