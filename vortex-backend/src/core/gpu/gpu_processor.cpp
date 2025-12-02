#include "gpu_processor.hpp"
#include "system/logger.hpp"

#ifdef VORTEX_ENABLE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#endif

#ifdef VORTEX_ENABLE_OPENCL
#include <CL/cl.h>
#endif

#ifdef VORTEX_ENABLE_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace vortex {

GPUProcessor::GPUProcessor()
    : initialized_(false)
    , currentBackend_("NONE")
    , deviceId_(0)
    , memoryManager_(nullptr)
    , streamCount_(4)
    , currentStream_(0)
{
    Logger::info("GPUProcessor constructor");
}

GPUProcessor::~GPUProcessor() {
    shutdown();
    Logger::info("GPUProcessor destroyed");
}

bool GPUProcessor::initialize(const std::string& backend, double sampleRate, int bufferSize, int channels) {
    if (initialized_) {
        Logger::warning("GPUProcessor already initialized");
        return true;
    }

    Logger::info("Initializing GPU processor with backend: {}", backend);

    try {
        // Validate backend
        if (!isBackendSupported(backend)) {
            Logger::error("GPU backend '{}' is not supported", backend);
            return false;
        }

        currentBackend_ = backend;
        sampleRate_ = sampleRate;
        bufferSize_ = bufferSize;
        channels_ = channels;

        // Initialize specific backend
        if (!initializeBackend(backend)) {
            Logger::error("Failed to initialize GPU backend: {}", backend);
            return false;
        }

        // Initialize memory manager
        memoryManager_ = std::make_unique<GPUMemoryManager>();
        if (!memoryManager_->initialize(backend, 1024 * 1024 * 1024)) { // 1GB default
            Logger::error("Failed to initialize GPU memory manager");
            return false;
        }

        // Initialize processing streams
        if (!initializeStreams()) {
            Logger::error("Failed to initialize GPU processing streams");
            return false;
        }

        // Initialize kernels
        if (!initializeKernels()) {
            Logger::error("Failed to initialize GPU kernels");
            return false;
        }

        // Query and log device information
        queryDeviceInfo();

        initialized_ = true;
        Logger::info("GPU processor initialized successfully with backend: {}", backend);
        return true;

    } catch (const std::exception& e) {
        Logger::error("GPU processor initialization failed: {}", e.what());
        return false;
    }
}

void GPUProcessor::shutdown() {
    if (!initialized_) {
        return;
    }

    Logger::info("Shutting down GPU processor");

    // Shutdown components in reverse order
    shutdownKernels();
    shutdownStreams();

    if (memoryManager_) {
        memoryManager_->shutdown();
        memoryManager_.reset();
    }

    shutdownBackend();

    currentBackend_ = "NONE";
    initialized_ = false;

    Logger::info("GPU processor shutdown completed");
}

bool GPUProcessor::isBackendSupported(const std::string& backend) const {
    if (backend == "CUDA") {
        return checkCUDASupport();
    } else if (backend == "OpenCL") {
        return checkOpenCLSupport();
    } else if (backend == "Vulkan") {
        return checkVulkanSupport();
    }
    return false;
}

bool GPUProcessor::checkCUDASupport() const {
#ifdef VORTEX_ENABLE_CUDA
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return error == cudaSuccess && deviceCount > 0;
#else
    return false;
#endif
}

bool GPUProcessor::checkOpenCLSupport() const {
#ifdef VORTEX_ENABLE_OPENCL
    cl_uint numPlatforms = 0;
    cl_int error = clGetPlatformIDs(0, nullptr, &numPlatforms);
    return error == CL_SUCCESS && numPlatforms > 0;
#else
    return false;
#endif
}

bool GPUProcessor::checkVulkanSupport() const {
#ifdef VORTEX_ENABLE_VULKAN
    try {
        vk::ApplicationInfo appInfo;
        vk::InstanceCreateInfo createInfo;
        vk::Instance instance = vk::createInstance(createInfo);
        instance.destroy();
        return true;
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

bool GPUProcessor::initializeBackend(const std::string& backend) {
    if (backend == "CUDA") {
        return initializeCUDABackend();
    } else if (backend == "OpenCL") {
        return initializeOpenCLBackend();
    } else if (backend == "Vulkan") {
        return initializeVulkanBackend();
    }
    return false;
}

#ifdef VORTEX_ENABLE_CUDA
bool GPUProcessor::initializeCUDABackend() {
    try {
        // Select CUDA device
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error != cudaSuccess || deviceCount == 0) {
            Logger::error("No CUDA devices found");
            return false;
        }

        // Use first available device (could be made configurable)
        deviceId_ = 0;
        error = cudaSetDevice(deviceId_);
        if (error != cudaSuccess) {
            Logger::error("Failed to set CUDA device: {}", cudaGetErrorString(error));
            return false;
        }

        // Initialize cuBLAS
        cublasStatus_t cublasError = cublasCreate(&cublasHandle_);
        if (cublasError != CUBLAS_STATUS_SUCCESS) {
            Logger::error("Failed to create cuBLAS handle");
            return false;
        }

        // Initialize cuFFT
        cufftResult fftError = cufftCreate(&cufftPlan_);
        if (fftError != CUFFT_SUCCESS) {
            Logger::error("Failed to create cuFFT handle");
            return false;
        }

        Logger::info("CUDA backend initialized successfully");
        return true;

    } catch (const std::exception& e) {
        Logger::error("CUDA backend initialization failed: {}", e.what());
        return false;
    }
}

void GPUProcessor::shutdownCUDABackend() {
#ifdef VORTEX_ENABLE_CUDA
    if (cufftPlan_) {
        cufftDestroy(cufftPlan_);
        cufftPlan_ = nullptr;
    }

    if (cublasHandle_) {
        cublasDestroy(cublasHandle_);
        cublasHandle_ = nullptr;
    }

    cudaDeviceReset();
#endif
}
#endif

#ifdef VORTEX_ENABLE_OPENCL
bool GPUProcessor::initializeOpenCLBackend() {
    try {
        // Get OpenCL platforms
        cl_uint numPlatforms = 0;
        cl_int error = clGetPlatformIDs(0, nullptr, &numPlatforms);
        if (error != CL_SUCCESS || numPlatforms == 0) {
            Logger::error("No OpenCL platforms found");
            return false;
        }

        std::vector<cl_platform_id> platforms(numPlatforms);
        error = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
        if (error != CL_SUCCESS) {
            Logger::error("Failed to get OpenCL platforms");
            return false;
        }

        // Use first platform
        clPlatform_ = platforms[0];

        // Get devices
        cl_uint numDevices = 0;
        error = clGetDeviceIDs(clPlatform_, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
        if (error != CL_SUCCESS || numDevices == 0) {
            Logger::error("No OpenCL GPU devices found");
            return false;
        }

        std::vector<cl_device_id> devices(numDevices);
        error = clGetDeviceIDs(clPlatform_, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
        if (error != CL_SUCCESS) {
            Logger::error("Failed to get OpenCL devices");
            return false;
        }

        // Use first device
        clDevice_ = devices[0];

        // Create context
        cl_context_properties props[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)clPlatform_,
            0
        };
        clContext_ = clCreateContext(props, 1, &clDevice_, nullptr, nullptr, &error);
        if (error != CL_SUCCESS) {
            Logger::error("Failed to create OpenCL context");
            return false;
        }

        // Create command queue
        clCommandQueue_ = clCreateCommandQueue(clContext_, clDevice_, 0, &error);
        if (error != CL_SUCCESS) {
            Logger::error("Failed to create OpenCL command queue");
            return false;
        }

        Logger::info("OpenCL backend initialized successfully");
        return true;

    } catch (const std::exception& e) {
        Logger::error("OpenCL backend initialization failed: {}", e.what());
        return false;
    }
}

void GPUProcessor::shutdownOpenCLBackend() {
#ifdef VORTEX_ENABLE_OPENCL
    if (clCommandQueue_) {
        clReleaseCommandQueue(clCommandQueue_);
        clCommandQueue_ = nullptr;
    }

    if (clContext_) {
        clReleaseContext(clContext_);
        clContext_ = nullptr;
    }
#endif
}
#endif

#ifdef VORTEX_ENABLE_VULKAN
bool GPUProcessor::initializeVulkanBackend() {
    try {
        // Create Vulkan instance
        vk::ApplicationInfo appInfo;
        appInfo.apiVersion = VK_API_VERSION_1_3;

        vk::InstanceCreateInfo createInfo;
        createInfo.pApplicationInfo = &appInfo;

        vulkanInstance_ = vk::createInstance(createInfo);

        // Get physical devices
        auto physicalDevices = vulkanInstance_.enumeratePhysicalDevices();
        if (physicalDevices.empty()) {
            Logger::error("No Vulkan physical devices found");
            return false;
        }

        // Use first device
        vulkanPhysicalDevice_ = physicalDevices[0];

        // Create logical device
        float queuePriority = 1.0f;
        vk::DeviceQueueCreateInfo queueCreateInfo;
        queueCreateInfo.queueFamilyIndex = 0; // Find appropriate queue family
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        vk::DeviceCreateInfo deviceCreateInfo;
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

        vulkanDevice_ = vulkanPhysicalDevice_.createDevice(deviceCreateInfo);

        // Get queue
        vulkanQueue_ = vulkanDevice_.getQueue(0, 0);

        // Create command pool
        vk::CommandPoolCreateInfo poolCreateInfo;
        poolCreateInfo.queueFamilyIndex = 0;
        vulkanCommandPool_ = vulkanDevice_.createCommandPool(poolCreateInfo);

        Logger::info("Vulkan backend initialized successfully");
        return true;

    } catch (const std::exception& e) {
        Logger::error("Vulkan backend initialization failed: {}", e.what());
        return false;
    }
}

void GPUProcessor::shutdownVulkanBackend() {
#ifdef VORTEX_ENABLE_VULKAN
    if (vulkanDevice_) {
        if (vulkanCommandPool_) {
            vulkanDevice_.destroyCommandPool(vulkanCommandPool_);
            vulkanCommandPool_ = nullptr;
        }
        vulkanDevice_.destroy();
        vulkanDevice_ = nullptr;
    }

    if (vulkanInstance_) {
        vulkanInstance_.destroy();
        vulkanInstance_ = nullptr;
    }
#endif
}
#endif

void GPUProcessor::shutdownBackend() {
    if (currentBackend_ == "CUDA") {
        shutdownCUDABackend();
    } else if (currentBackend_ == "OpenCL") {
        shutdownOpenCLBackend();
    } else if (currentBackend_ == "Vulkan") {
        shutdownVulkanBackend();
    }
}

bool GPUProcessor::initializeStreams() {
    try {
        streams_.clear();
        streams_.reserve(streamCount_);

        for (int i = 0; i < streamCount_; ++i) {
            streams_.push_back(createStream());
        }

        Logger::debug("Created {} GPU processing streams", streamCount_);
        return true;

    } catch (const std::exception& e) {
        Logger::error("Failed to initialize GPU streams: {}", e.what());
        return false;
    }
}

void GPUProcessor::shutdownStreams() {
    for (auto& stream : streams_) {
        destroyStream(stream);
    }
    streams_.clear();
}

bool GPUProcessor::initializeKernels() {
    try {
        // Initialize FFT kernels
        if (!initializeFFTKernels()) {
            return false;
        }

        // Initialize audio processing kernels
        if (!initializeAudioKernels()) {
            return false;
        }

        // Initialize convolution kernels
        if (!initializeConvolutionKernels()) {
            return false;
        }

        Logger::info("GPU kernels initialized successfully");
        return true;

    } catch (const std::exception& e) {
        Logger::error("Failed to initialize GPU kernels: {}", e.what());
        return false;
    }
}

void GPUProcessor::shutdownKernels() {
    // Cleanup would go here
    Logger::debug("GPU kernels shutdown");
}

void* GPUProcessor::createStream() {
    if (currentBackend_ == "CUDA") {
#ifdef VORTEX_ENABLE_CUDA
        cudaStream_t stream;
        cudaError_t error = cudaStreamCreate(&stream);
        return (error == cudaSuccess) ? stream : nullptr;
#endif
    }
    return nullptr;
}

void GPUProcessor::destroyStream(void* stream) {
    if (stream && currentBackend_ == "CUDA") {
#ifdef VORTEX_ENABLE_CUDA
        cudaStreamDestroy(static_cast<cudaStream_t>(stream));
#endif
    }
}

void GPUProcessor::beginProcessing(size_t numSamples, int channels) {
    currentStream_ = (currentStream_ + 1) % streamCount_;
    Logger::debug("Beginning GPU processing: {} samples, {} channels", numSamples, channels);
}

void GPUProcessor::endProcessing() {
    if (currentBackend_ == "CUDA") {
#ifdef VORTEX_ENABLE_CUDA
        cudaStreamSynchronize(static_cast<cudaStream_t>(streams_[currentStream_]));
#endif
    }
    Logger::debug("GPU processing completed");
}

bool GPUProcessor::processAudio(float* audioData, size_t numSamples, int channels) {
    if (!initialized_) {
        Logger::error("GPU processor not initialized");
        return false;
    }

    try {
        // Allocate GPU memory
        size_t dataSize = numSamples * channels * sizeof(float);
        void* gpuBuffer = memoryManager_->allocate(dataSize);
        if (!gpuBuffer) {
            Logger::error("Failed to allocate GPU memory for audio processing");
            return false;
        }

        // Copy data to GPU
        if (!copyToGPU(audioData, gpuBuffer, dataSize)) {
            memoryManager_->deallocate(gpuBuffer);
            return false;
        }

        // Process on GPU
        if (!processAudioGPU(gpuBuffer, numSamples, channels)) {
            memoryManager_->deallocate(gpuBuffer);
            return false;
        }

        // Copy results back to CPU
        if (!copyFromGPU(gpuBuffer, audioData, dataSize)) {
            memoryManager_->deallocate(gpuBuffer);
            return false;
        }

        // Cleanup
        memoryManager_->deallocate(gpuBuffer);

        return true;

    } catch (const std::exception& e) {
        Logger::error("GPU audio processing failed: {}", e.what());
        return false;
    }
}

bool GPUProcessor::processFFT(float* input, float* output, size_t fftSize) {
    if (!initialized_) {
        Logger::error("GPU processor not initialized");
        return false;
    }

    try {
        // Implementation depends on backend
        if (currentBackend_ == "CUDA") {
            return processFFTCUDA(input, output, fftSize);
        } else if (currentBackend_ == "OpenCL") {
            return processFFTOpenCL(input, output, fftSize);
        } else if (currentBackend_ == "Vulkan") {
            return processFFTVulkan(input, output, fftSize);
        }

        return false;

    } catch (const std::exception& e) {
        Logger::error("GPU FFT processing failed: {}", e.what());
        return false;
    }
}

bool GPUProcessor::processConvolution(float* input, float* output, float* ir,
                                    size_t inputSize, size_t irSize, size_t outputSize) {
    if (!initialized_) {
        Logger::error("GPU processor not initialized");
        return false;
    }

    try {
        // Implementation depends on backend
        if (currentBackend_ == "CUDA") {
            return processConvolutionCUDA(input, output, ir, inputSize, irSize, outputSize);
        } else if (currentBackend_ == "OpenCL") {
            return processConvolutionOpenCL(input, output, ir, inputSize, irSize, outputSize);
        } else if (currentBackend_ == "Vulkan") {
            return processConvolutionVulkan(input, output, ir, inputSize, irSize, outputSize);
        }

        return false;

    } catch (const std::exception& e) {
        Logger::error("GPU convolution processing failed: {}", e.what());
        return false;
    }
}

bool GPUProcessor::processEqualizer(float* audio, const float* frequencies,
                                   const float* gains, const float* qValues,
                                   size_t numBands, size_t numSamples, int channels) {
    if (!initialized_) {
        Logger::error("GPU processor not initialized");
        return false;
    }

    try {
        // Implementation depends on backend
        if (currentBackend_ == "CUDA") {
            return processEqualizerCUDA(audio, frequencies, gains, qValues,
                                       numBands, numSamples, channels);
        } else if (currentBackend_ == "OpenCL") {
            return processEqualizerOpenCL(audio, frequencies, gains, qValues,
                                         numBands, numSamples, channels);
        } else if (currentBackend_ == "Vulkan") {
            return processEqualizerVulkan(audio, frequencies, gains, qValues,
                                         numBands, numSamples, channels);
        }

        return false;

    } catch (const std::exception& e) {
        Logger::error("GPU equalizer processing failed: {}", e.what());
        return false;
    }
}

void GPUProcessor::queryDeviceInfo() {
    Logger::info("Querying GPU device information");

    if (currentBackend_ == "CUDA") {
        queryCUDADeviceInfo();
    } else if (currentBackend_ == "OpenCL") {
        queryOpenCLDeviceInfo();
    } else if (currentBackend_ == "Vulkan") {
        queryVulkanDeviceInfo();
    }
}

#ifdef VORTEX_ENABLE_CUDA
void GPUProcessor::queryCUDADeviceInfo() {
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, deviceId_);
    if (error == cudaSuccess) {
        Logger::info("CUDA Device: {} (Compute {}.{}), {} MB memory",
                     prop.name, prop.major, prop.minor,
                     prop.totalGlobalMem / (1024 * 1024));
    }
}
#endif

#ifdef VORTEX_ENABLE_OPENCL
void GPUProcessor::queryOpenCLDeviceInfo() {
    cl_char name[256] = {0};
    cl_int error = clGetDeviceInfo(clDevice_, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    if (error == CL_SUCCESS) {
        Logger::info("OpenCL Device: {}", name);
    }
}
#endif

#ifdef VORTEX_ENABLE_VULKAN
void GPUProcessor::queryVulkanDeviceInfo() {
    auto properties = vulkanPhysicalDevice_.getProperties();
    Logger::info("Vulkan Device: {} (API Version {}.{}.{})",
                 properties.deviceName,
                 VK_VERSION_MAJOR(properties.apiVersion),
                 VK_VERSION_MINOR(properties.apiVersion),
                 VK_VERSION_PATCH(properties.apiVersion));
}
#endif

// Getters
bool GPUProcessor::isInitialized() const { return initialized_; }
const std::string& GPUProcessor::getBackend() const { return currentBackend_; }
GPUMemoryManager* GPUProcessor::getMemoryManager() const { return memoryManager_.get(); }

} // namespace vortex