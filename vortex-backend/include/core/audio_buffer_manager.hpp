#pragma once

#include <memory>
#include <vector>
#include <mutex>
#include <chrono>
#include <atomic>
#include <map>

namespace vortex {

struct AudioBuffer {
    void* data = nullptr;
    size_t size = 0;
    bool inUse = false;
    std::chrono::high_resolution_clock::time_point allocationTime;
    std::chrono::high_resolution_clock::time_point lastAccessTime;
};

/**
 * @brief High-performance audio buffer manager with memory pooling
 *
 * This class provides efficient memory management for audio processing buffers
 * with aligned memory allocation, object pooling, and real-time safety.
 * It supports both temporary and reusable buffers with automatic cleanup.
 */
class AudioBufferManager {
public:
    struct Statistics {
        size_t totalBuffers = 0;
        size_t activeBuffers = 0;
        size_t bufferPoolHits = 0;
        size_t newAllocations = 0;
        size_t bufferReuses = 0;
        size_t temporaryBuffers = 0;
        size_t allocationFailures = 0;
        size_t bufferReturns = 0;
    };

    AudioBufferManager();
    ~AudioBufferManager();

    // Initialization
    bool initialize(double sampleRate = 44100.0, int bufferSize = 512, int channels = 2, int bitDepth = 24);
    void shutdown();

    // Buffer management
    AudioBuffer getProcessingBuffer(size_t numSamples);
    void returnProcessingBuffer(AudioBuffer&& buffer);
    std::unique_ptr<float[]> createTemporaryBuffer(size_t numSamples);
    void destroyTemporaryBuffer(std::unique_ptr<float[]>& buffer);

    // Memory information
    size_t getTotalMemoryUsed() const;
    size_t getAvailableBuffers() const;
    Statistics getStatistics() const;
    void resetStatistics();

    // Pool optimization
    void optimizeBufferPool();

    // Configuration
    void setPoolSize(size_t size);
    void setMaxPoolSize(size_t size);
    void setAlignment(size_t alignment);

    // Getters
    size_t getPoolSize() const;
    size_t getMaxPoolSize() const;
    size_t getAlignment() const;
    size_t getAlignedBufferSize() const;
    bool isInitialized() const;

private:
    // Core configuration
    bool initialized_;
    double sampleRate_;
    int bufferSize_;
    int channels_;
    int bitDepth_;

    // Buffer pool settings
    size_t poolSize_;
    size_t maxPoolSize_;
    size_t alignment_;
    size_t alignedBufferSize_;

    // Memory tracking
    std::atomic<size_t> totalMemoryUsed_;

    // Buffer pool
    mutable std::mutex poolMutex_;
    std::vector<AudioBuffer> bufferPool_;
    std::map<void*, AudioBuffer> allocatedBuffers_;

    // Statistics
    mutable std::mutex statsMutex_;
    Statistics stats_;

    // Private methods
    bool initializeBufferPool();
    void* allocateAlignedBuffer(size_t size);
    void deallocateAlignedBuffer(void* ptr);
};

} // namespace vortex