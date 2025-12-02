#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <atomic>
#include <memory>

namespace vortex {

/**
 * @brief GPU memory manager with pooling and defragmentation
 *
 * This class provides efficient GPU memory management with automatic memory
 * pooling, defragmentation, and statistics tracking. It supports multiple GPU
 * backends and provides a unified interface for memory allocation/deallocation
 * optimized for audio processing workloads.
 */
class GPUMemoryManager {
public:
    struct Statistics {
        size_t totalMemorySize = 0;
        size_t usedMemory = 0;
        size_t availableMemory = 0;
        double utilization = 0.0;

        size_t totalAllocations = 0;
        size_t totalDeallocations = 0;
        size_t activeBlocks = 0;

        size_t poolAllocations = 0;
        size_t blockSplits = 0;
        size_t defragmentations = 0;
        size_t allocationFailures = 0;
    };

    struct MemoryBlock {
        void* pointer = nullptr;
        size_t size = 0;
        bool inUse = false;
        std::chrono::steady_clock::time_point timestamp;
    };

    struct MemoryPool {
        void* pointer = nullptr;
        size_t size = 0;
        bool used = false;
        std::chrono::steady_clock::time_point timestamp;
    };

    GPUMemoryManager();
    ~GPUMemoryManager();

    // Initialization
    bool initialize(const std::string& backend, size_t poolSize);
    void shutdown();

    // Memory allocation
    void* allocate(size_t size);
    void deallocate(void* ptr);

    // Statistics and monitoring
    void printStatistics() const;
    Statistics getStatistics() const;
    void resetStatistics();

    // Memory information
    size_t getTotalMemory() const;
    size_t getUsedMemory() const;
    size_t getAvailableMemory() const;
    double getUtilization() const;

    // Configuration
    void setMaxBlockSize(size_t size) { maxBlockSize_ = size; }
    void setMinBlockSize(size_t size) { minBlockSize_ = size; }
    void setAlignment(size_t alignment) { alignment_ = alignment; }
    void setFragmentationThreshold(float threshold) { fragmentationThreshold_ = threshold; }

private:
    // Core state
    bool initialized_;
    std::string backend_;

    // Memory pools
    size_t totalMemorySize_;
    size_t usedMemory_;
    size_t maxBlockSize_;
    size_t minBlockSize_;
    size_t alignment_;
    float fragmentationThreshold_;

    // Memory management
    mutable std::mutex memoryMutex_;
    std::vector<MemoryBlock> allocatedBlocks_;
    std::vector<MemoryPool> memoryPools_;

    // Statistics
    mutable std::mutex statsMutex_;
    Statistics stats_;

    // Backend-specific initialization
    bool initializeBackendResources();
    void cleanupBackendResources();
    bool allocateInitialPool();

    // Memory management algorithms
    MemoryBlock* findFreeBlock(size_t size);
    MemoryBlock* splitBlock(size_t size);
    bool defragment();
    bool fragmentationTooHigh() const;

    // Low-level memory allocation
    void* allocateGPUMemory(size_t size);
    void deallocateGPUMemory(void* ptr);

    // Debugging and statistics
    void printStatsStatistics() const;

    // Backend-specific implementations
#ifdef VORTEX_ENABLE_CUDA
    void* allocateCUDAMemory(size_t size);
    void deallocateCUDAMemory(void* ptr);
    bool initializeCUDAResources();
    void cleanupCUDAResources();
#endif

#ifdef VORTEX_ENABLE_OPENCL
    void* allocateOpenCLMemory(size_t size);
    void deallocateOpenCLMemory(void* ptr);
    bool initializeOpenCLResources();
    void cleanupOpenCLResources();
#endif

#ifdef VORTEX_ENABLE_VULKAN
    void* allocateVulkanMemory(size_t size);
    void deallocateVulkanMemory(void* ptr);
    bool initializeVulkanResources();
    void cleanupVulkanResources();
#endif
};

} // namespace vortex