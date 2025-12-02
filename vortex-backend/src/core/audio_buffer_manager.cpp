#include "audio_buffer_manager.hpp"
#include "system/logger.hpp"

#include <algorithm>
#include <cstring>

namespace vortex {

AudioBufferManager::AudioBufferManager()
    : initialized_(false)
    , sampleRate_(44100.0)
    , bufferSize_(512)
    , channels_(2)
    , bitDepth_(24)
    , poolSize_(32)
    , maxPoolSize_(256)
    , alignment_(64)
    , totalMemoryUsed_(0)
{
    Logger::info("AudioBufferManager constructor");
}

AudioBufferManager::~AudioBufferManager() {
    shutdown();
    Logger::info("AudioBufferManager destroyed");
}

bool AudioBufferManager::initialize(double sampleRate, int bufferSize, int channels, int bitDepth) {
    if (initialized_) {
        Logger::warning("AudioBufferManager already initialized");
        return true;
    }

    Logger::info("Initializing AudioBufferManager: sr={}, bs={}, ch={}, depth={}",
                 sampleRate, bufferSize, channels, bitDepth);

    try {
        sampleRate_ = sampleRate;
        bufferSize_ = bufferSize;
        channels_ = channels;
        bitDepth_ = bitDepth;

        // Calculate aligned buffer size
        size_t baseSize = bufferSize_ * channels_ * sizeof(float);
        alignedBufferSize_ = ((baseSize + alignment_ - 1) / alignment_) * alignment_;

        // Pre-allocate buffer pool
        if (!initializeBufferPool()) {
            Logger::error("Failed to initialize buffer pool");
            return false;
        }

        // Initialize statistics
        resetStatistics();

        initialized_ = true;
        Logger::info("AudioBufferManager initialized successfully (pool size: {}, buffer size: {} bytes)",
                     poolSize_, alignedBufferSize_);

        return true;

    } catch (const std::exception& e) {
        Logger::error("AudioBufferManager initialization failed: {}", e.what());
        return false;
    }
}

void AudioBufferManager::shutdown() {
    if (!initialized_) {
        return;
    }

    Logger::info("Shutting down AudioBufferManager");

    std::lock_guard<std::mutex> lock(poolMutex_);

    // Clear buffer pool
    for (auto& buffer : bufferPool_) {
        deallocateAlignedBuffer(buffer.data);
    }
    bufferPool_.clear();

    // Clear allocated buffers tracking
    for (auto& pair : allocatedBuffers_) {
        deallocateAlignedBuffer(pair.second.data);
    }
    allocatedBuffers_.clear();

    totalMemoryUsed_ = 0;
    initialized_ = false;

    Logger::info("AudioBufferManager shutdown completed");
}

bool AudioBufferManager::initializeBufferPool() {
    std::lock_guard<std::mutex> lock(poolMutex_);

    try {
        bufferPool_.reserve(poolSize_);

        for (size_t i = 0; i < poolSize_; ++i) {
            void* memory = allocateAlignedBuffer(alignedBufferSize_);
            if (!memory) {
                Logger::error("Failed to allocate buffer {} of size {}", i, alignedBufferSize_);
                return false;
            }

            AudioBuffer buffer;
            buffer.data = memory;
            buffer.size = alignedBufferSize_;
            buffer.inUse = false;
            buffer.allocationTime = std::chrono::high_resolution_clock::now();

            bufferPool_.push_back(buffer);
            totalMemoryUsed_ += alignedBufferSize_;
        }

        Logger::info("Buffer pool initialized with {} buffers ({} MB total)",
                     poolSize_, (totalMemoryUsed_ / (1024 * 1024)));

        return true;

    } catch (const std::exception& e) {
        Logger::error("Buffer pool initialization failed: {}", e.what());
        return false;
    }
}

AudioBuffer AudioBufferManager::getProcessingBuffer(size_t numSamples) {
    if (!initialized_) {
        Logger::error("AudioBufferManager not initialized");
        return AudioBuffer{nullptr, 0, false};
    }

    size_t requiredSize = numSamples * channels_ * sizeof(float);
    requiredSize = ((requiredSize + alignment_ - 1) / alignment_) * alignment_;

    std::lock_guard<std::mutex> lock(poolMutex_);

    // First, try to find a free buffer in the pool
    for (auto& buffer : bufferPool_) {
        if (!buffer.inUse && buffer.size >= requiredSize) {
            buffer.inUse = true;
            buffer.allocationTime = std::chrono::high_resolution_clock::now();
            buffer.lastAccessTime = buffer.allocationTime;

            stats_.bufferPoolHits++;
            stats_.activeBuffers++;

            Logger::debug("Buffer allocated from pool (size: {} bytes)", buffer.size);
            return buffer;
        }
    }

    // No suitable buffer in pool, try to allocate new one
    if (bufferPool_.size() < maxPoolSize_) {
        void* memory = allocateAlignedBuffer(requiredSize);
        if (memory) {
            AudioBuffer buffer;
            buffer.data = memory;
            buffer.size = requiredSize;
            buffer.inUse = true;
            buffer.allocationTime = std::chrono::high_resolution_clock::now();
            buffer.lastAccessTime = buffer.allocationTime;

            bufferPool_.push_back(buffer);
            totalMemoryUsed_ += requiredSize;

            stats_.newAllocations++;
            stats_.activeBuffers++;

            Logger::debug("New buffer allocated (size: {} bytes)", requiredSize);
            return buffer;
        }
    }

    // Pool is full and can't allocate more, reuse oldest buffer
    auto oldestIt = std::min_element(bufferPool_.begin(), bufferPool_.end(),
        [](const AudioBuffer& a, const AudioBuffer& b) {
            return a.allocationTime < b.allocationTime;
        });

    if (oldestIt != bufferPool_.end() && oldestIt->size >= requiredSize) {
        Logger::warning("Reusing oldest buffer from pool");
        oldestIt->inUse = true;
        oldestIt->allocationTime = std::chrono::high_resolution_clock::now();
        oldestIt->lastAccessTime = oldestIt->allocationTime;

        stats_.bufferReuses++;
        stats_.activeBuffers++;

        return *oldestIt;
    }

    // All attempts failed
    Logger::error("Failed to allocate processing buffer of size {} bytes", requiredSize);
    stats_.allocationFailures++;

    return AudioBuffer{nullptr, 0, false};
}

void AudioBufferManager::returnProcessingBuffer(AudioBuffer&& buffer) {
    if (!initialized_ || buffer.data == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(poolMutex_);

    // Mark buffer as available
    buffer.inUse = false;
    buffer.lastAccessTime = std::chrono::high_resolution_clock::now();

    stats_.activeBuffers--;
    stats_.bufferReturns++;

    Logger::debug("Buffer returned to pool");
}

std::unique_ptr<float[]> AudioBufferManager::createTemporaryBuffer(size_t numSamples) {
    if (!initialized_) {
        Logger::error("AudioBufferManager not initialized");
        return nullptr;
    }

    size_t requiredSize = numSamples * sizeof(float);
    size_t alignedSize = ((requiredSize + alignment_ - 1) / alignment_) * alignment_;

    void* memory = allocateAlignedBuffer(alignedSize);
    if (!memory) {
        Logger::error("Failed to allocate temporary buffer of size {} bytes", alignedSize);
        return nullptr;
    }

    // Track allocation
    std::lock_guard<std::mutex> lock(poolMutex_);
    allocatedBuffers_[memory] = AudioBuffer{memory, alignedSize, true};
    totalMemoryUsed_ += alignedSize;

    stats_.temporaryBuffers++;
    stats_.activeBuffers++;

    Logger::debug("Temporary buffer allocated (size: {} bytes)", alignedSize);

    return std::unique_ptr<float[]>(static_cast<float*>(memory));
}

void AudioBufferManager::destroyTemporaryBuffer(std::unique_ptr<float[]>& buffer) {
    if (!buffer || !initialized_) {
        return;
    }

    void* memory = buffer.get();

    std::lock_guard<std::mutex> lock(poolMutex_);

    auto it = allocatedBuffers_.find(memory);
    if (it != allocatedBuffers_.end()) {
        deallocateAlignedBuffer(it->second.data);
        totalMemoryUsed_ -= it->second.size;
        allocatedBuffers_.erase(it);

        stats_.activeBuffers--;
        Logger::debug("Temporary buffer destroyed");
    }

    buffer.release();
}

size_t AudioBufferManager::getTotalMemoryUsed() const {
    std::lock_guard<std::mutex> lock(poolMutex_);
    return totalMemoryUsed_;
}

size_t AudioBufferManager::getAvailableBuffers() const {
    std::lock_guard<std::mutex> lock(poolMutex_);

    size_t available = 0;
    for (const auto& buffer : bufferPool_) {
        if (!buffer.inUse) {
            available++;
        }
    }
    return available;
}

AudioBufferManager::Statistics AudioBufferManager::getStatistics() const {
    std::lock_guard<std::mutex> lock(poolMutex_);
    return stats_;
}

void AudioBufferManager::resetStatistics() {
    std::lock_guard<std::mutex> lock(poolMutex_);
    stats_ = Statistics{};
    stats_.totalBuffers = bufferPool_.size();
}

void AudioBufferManager::optimizeBufferPool() {
    if (!initialized_) {
        return;
    }

    std::lock_guard<std::mutex> lock(poolMutex_);

    auto now = std::chrono::high_resolution_clock::now();
    auto threshold = std::chrono::minutes(5); // 5 minutes threshold

    // Remove unused buffers older than threshold
    auto it = std::remove_if(bufferPool_.begin(), bufferPool_.end(),
        [now, threshold](const AudioBuffer& buffer) {
            if (!buffer.inUse) {
                auto age = now - buffer.lastAccessTime;
                if (age > threshold && bufferPool_.size() > poolSize_) {
                    deallocateAlignedBuffer(buffer.data);
                    totalMemoryUsed_ -= buffer.size;
                    return true;
                }
            }
            return false;
        });

    size_t removed = std::distance(it, bufferPool_.end());
    bufferPool_.erase(it, bufferPool_.end());

    if (removed > 0) {
        Logger::info("Optimized buffer pool: removed {} unused buffers", removed);
    }
}

void* AudioBufferManager::allocateAlignedBuffer(size_t size) {
#if defined(_WIN32)
    return _aligned_malloc(size, alignment_);
#elif defined(__linux__) || defined(__APPLE__)
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment_, size) != 0) {
        return nullptr;
    }
    return ptr;
#else
    // Fallback to aligned allocation
    size_t totalSize = size + alignment_ - 1;
    void* raw = std::malloc(totalSize);
    if (!raw) {
        return nullptr;
    }

    uintptr_t addr = reinterpret_cast<uintptr_t>(raw);
    uintptr_t aligned = (addr + alignment_ - 1) & ~(alignment_ - 1);
    reinterpret_cast<void**>(aligned)[-1] = raw; // Store original pointer for free

    return reinterpret_cast<void*>(aligned);
#endif
}

void AudioBufferManager::deallocateAlignedBuffer(void* ptr) {
    if (!ptr) {
        return;
    }

#if defined(_WIN32)
    _aligned_free(ptr);
#elif defined(__linux__) || defined(__APPLE__)
    std::free(ptr);
#else
    // Fallback aligned deallocation
    void* raw = reinterpret_cast<void**>(ptr)[-1];
    std::free(raw);
#endif
}

// Configuration methods
void AudioBufferManager::setPoolSize(size_t size) {
    if (size > maxPoolSize_) {
        Logger::warning("Pool size {} exceeds maximum {}", size, maxPoolSize_);
        return;
    }
    poolSize_ = size;
}

void AudioBufferManager::setMaxPoolSize(size_t size) {
    if (size < poolSize_) {
        Logger::warning("Max pool size {} is less than current pool size {}", size, poolSize_);
        return;
    }
    maxPoolSize_ = size;
}

void AudioBufferManager::setAlignment(size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        Logger::error("Invalid alignment size: {}", alignment);
        return;
    }
    alignment_ = alignment;
}

// Getters
size_t AudioBufferManager::getPoolSize() const { return poolSize_; }
size_t AudioBufferManager::getMaxPoolSize() const { return maxPoolSize_; }
size_t AudioBufferManager::getAlignment() const { return alignment_; }
size_t AudioBufferManager::getAlignedBufferSize() const { return alignedBufferSize_; }
bool AudioBufferManager::isInitialized() const { return initialized_; }

} // namespace vortex