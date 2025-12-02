#pragma once

#include "audio_types.hpp"
#include "network_types.hpp"

#include <memory>
#include <vector>
#include <string>
#include <mutex>
#include <atomic>
#include <unordered_map>

namespace vortex {

/**
 * @brief GPU memory manager for efficient audio buffer management
 *
 * This class manages GPU memory allocation, deallocation, and optimization
 * for real-time audio processing with sub-10ms latency requirements.
 */
class MemoryManager {
public:
    MemoryManager();
    ~MemoryManager();

    // Lifecycle
    bool initialize(GPUBackend backend, uint64_t poolSize);
    void shutdown();
    bool isInitialized() const;

    // Memory allocation
    void* allocate(size_t size, const std::string& name = "");
    void deallocate(void* ptr);
    void* reallocate(void* ptr, size_t newSize);

    // Memory pool management
    bool createPool(const std::string& poolName, size_t poolSize, size_t blockSize);
    void destroyPool(const std::string& poolName);
    void* allocateFromPool(const std::string& poolName);
    void deallocateToPool(const std::string& poolName, void* ptr);

    // Memory operations
    bool copyToDevice(const void* hostPtr, void* devicePtr, size_t size);
    bool copyFromDevice(const void* devicePtr, void* hostPtr, size_t size);
    bool copyDeviceToDevice(const void* srcPtr, void* dstPtr, size_t size);
    bool memsetDevice(void* devicePtr, int value, size_t size);

    // Memory optimization
    bool optimizeForRealtime();
    defragmentMemory();
    bool prefetchMemory(const std::vector<void*>& pointers);
    bool flushMemory();

    // Memory monitoring
    uint64_t getTotalMemory() const;
    uint64_t getUsedMemory() const;
    uint64_t getFreeMemory() const;
    float getUtilization() const;

    // Memory statistics
    struct MemoryStats {
        uint64_t totalAllocations = 0;
        uint64_t totalDeallocations = 0;
        uint64_t peakUsage = 0;
        uint64_t currentUsage = 0;
        uint32_t fragmentationLevel = 0;
        float allocationEfficiency = 0.0f;
    };

    MemoryStats getStatistics() const;
    void resetStatistics();

    // Memory mapping for zero-copy
    bool* allocatePinnedHost(size_t size);
    void deallocatePinnedHost(void* ptr);
    bool* mapDeviceMemory(void* devicePtr, size_t size);
    void unmapDeviceMemory(void* mappedPtr);

    // Buffer management
    struct AudioBuffer {
        void* devicePtr = nullptr;
        void* hostPtr = nullptr;
        size_t size = 0;
        bool isPinned = false;
        bool isMapped = false;
        std::string name;
    };

    std::shared_ptr<AudioBuffer> createAudioBuffer(size_t size, const std::string& name = "");
    bool destroyAudioBuffer(const std::string& name);
    std::shared_ptr<AudioBuffer> getAudioBuffer(const std::string& name);

    // Advanced features
    bool enableMemoryOvercommit(bool enable);
    bool setMemoryPressureThreshold(float threshold);
    bool garbageCollect();

protected:
    // Backend-specific implementations
    bool initializeCUDA(uint64_t poolSize);
    bool initializeOpenCL(uint64_t poolSize);
    bool initializeVulkan(uint64_t poolSize);

    void shutdownCUDA();
    void shutdownOpenCL();
    void shutdownVulkan();

    // Memory pool implementation
    struct MemoryPool {
        std::string name;
        void* basePtr = nullptr;
        size_t totalSize = 0;
        size_t blockSize = 0;
        std::vector<bool> allocationMap;
        std::vector<void*> freeList;
        mutable std::mutex mutex;
    };

    std::unordered_map<std::string, std::unique_ptr<MemoryPool>> m_pools;

    // Memory tracking
    struct Allocation {
        void* ptr;
        size_t size;
        std::string name;
        std::chrono::steady_clock::time_point timestamp;
    };

    std::unordered_map<void*, Allocation> m_allocations;
    mutable std::mutex m_allocationMutex;

    // State
    GPUBackend m_backend = GPUBackend::NONE;
    bool m_initialized = false;
    uint64_t m_totalPoolSize = 0;
    std::atomic<uint64_t> m_usedMemory{0};

    // Configuration
    bool m_overcommitEnabled = false;
    float m_pressureThreshold = 0.9f;
    bool m_realtimeOptimized = false;

    // Statistics
    mutable std::mutex m_statsMutex;
    MemoryStats m_stats;

private:
    // Internal helpers
    void* allocateFromSystem(size_t size);
    void deallocateToSystem(void* ptr);
    bool isPointerValid(void* ptr) const;
    void updateStatistics(size_t allocatedSize, bool deallocation = false);

    // Alignment utilities
    static constexpr size_t MEMORY_ALIGNMENT = 64; // 64-byte alignment for GPU
    static size_t alignSize(size_t size);
};

/**
 * @brief Factory for creating appropriate memory manager
 */
class MemoryManagerFactory {
public:
    static std::unique_ptr<MemoryManager> create(GPUBackend backend, uint64_t poolSize);
    static std::vector<GPUBackend> getSupportedBackends();
};

} // namespace vortex