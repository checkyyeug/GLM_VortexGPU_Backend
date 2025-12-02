#include "memory_manager.hpp"
#include "system/logger.hpp"

#ifdef VORTEX_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef VORTEX_ENABLE_OPENCL
#include <CL/cl.h>
#endif

#include <algorithm>
#include <chrono>

namespace vortex {

GPUMemoryManager::GPUMemoryManager()
    : initialized_(false)
    , totalMemorySize_(0)
    , usedMemory_(0)
    , maxBlockSize_(64 * 1024 * 1024) // 64MB
    , minBlockSize_(1024) // 1KB
    , alignment_(256)
    , fragmentationThreshold_(0.1f) // 10%
{
    Logger::info("GPUMemoryManager constructor");
}

GPUMemoryManager::~GPUMemoryManager() {
    shutdown();
    Logger::info("GPUMemoryManager destroyed");
}

bool GPUMemoryManager::initialize(const std::string& backend, size_t poolSize) {
    if (initialized_) {
        Logger::warning("GPUMemoryManager already initialized");
        return true;
    }

    Logger::info("Initializing GPU memory manager with backend: {}, pool size: {} MB",
                 backend, poolSize / (1024 * 1024));

    try {
        backend_ = backend;
        totalMemorySize_ = poolSize;
        usedMemory_ = 0;

        // Initialize backend-specific resources
        if (!initializeBackendResources()) {
            Logger::error("Failed to initialize backend resources");
            return false;
        }

        // Allocate initial memory pool
        if (!allocateInitialPool()) {
            Logger::error("Failed to allocate initial memory pool");
            return false;
        }

        // Initialize statistics
        resetStatistics();

        initialized_ = true;
        Logger::info("GPU memory manager initialized successfully ({} MB pool)",
                     totalMemorySize_ / (1024 * 1024));

        return true;

    } catch (const std::exception& e) {
        Logger::error("GPU memory manager initialization failed: {}", e.what());
        return false;
    }
}

void GPUMemoryManager::shutdown() {
    if (!initialized_) {
        return;
    }

    Logger::info("Shutting down GPU memory manager");

    // Print final statistics
    printStatistics();

    // Free all allocated memory blocks
    std::lock_guard<std::mutex> lock(memoryMutex_);
    for (auto& block : allocatedBlocks_) {
        deallocateGPUMemory(block.pointer);
    }
    allocatedBlocks_.clear();

    // Free memory pool
    for (auto& pool : memoryPools_) {
        deallocateGPUMemory(pool.pointer);
    }
    memoryPools_.clear();

    // Cleanup backend resources
    cleanupBackendResources();

    totalMemorySize_ = 0;
    usedMemory_ = 0;
    initialized_ = false;

    Logger::info("GPU memory manager shutdown completed");
}

bool GPUMemoryManager::initializeBackendResources() {
    if (backend_ == "CUDA") {
        return initializeCUDAResources();
    } else if (backend_ == "OpenCL") {
        return initializeOpenCLResources();
    } else if (backend_ == "Vulkan") {
        return initializeVulkanResources();
    }
    return false;
}

void GPUMemoryManager::cleanupBackendResources() {
    if (backend_ == "CUDA") {
        cleanupCUDAResources();
    } else if (backend_ == "OpenCL") {
        cleanupOpenCLResources();
    } else if (backend_ == "Vulkan") {
        cleanupVulkanResources();
    }
}

bool GPUMemoryManager::allocateInitialPool() {
    try {
        // Allocate large memory pool
        void* poolPtr = allocateGPUMemory(totalMemorySize_);
        if (!poolPtr) {
            Logger::error("Failed to allocate initial memory pool of {} bytes", totalMemorySize_);
            return false;
        }

        // Create initial pool entry
        MemoryPool pool;
        pool.pointer = poolPtr;
        pool.size = totalMemorySize_;
        pool.used = false;
        pool.timestamp = std::chrono::steady_clock::now();

        memoryPools_.push_back(pool);

        Logger::debug("Initial memory pool allocated: {} bytes", totalMemorySize_);
        return true;

    } catch (const std::exception& e) {
        Logger::error("Initial pool allocation failed: {}", e.what());
        return false;
    }
}

void* GPUMemoryManager::allocate(size_t size) {
    if (!initialized_) {
        Logger::error("Memory manager not initialized");
        return nullptr;
    }

    // Align size to alignment boundary
    size_t alignedSize = ((size + alignment_ - 1) / alignment_) * alignment_;

    std::lock_guard<std::mutex> lock(memoryMutex_);

    // Update statistics
    stats_.totalAllocations++;

    // Try to find suitable free block
    MemoryBlock* block = findFreeBlock(alignedSize);
    if (block) {
        block->inUse = true;
        block->timestamp = std::chrono::steady_clock::now();
        usedMemory_ += block->size;

        stats_.poolAllocations++;
        stats_.activeBlocks++;

        Logger::debug("Memory allocated from pool: {} bytes", alignedSize);
        return block->pointer;
    }

    // Try to split a larger block
    block = splitBlock(alignedSize);
    if (block) {
        block->inUse = true;
        block->timestamp = std::chrono::steady_clock::now();
        usedMemory_ += block->size;

        stats_.blockSplits++;
        stats_.activeBlocks++;

        Logger::debug("Memory allocated from split block: {} bytes", alignedSize);
        return block->pointer;
    }

    // Try to defragment and retry
    if (defragment()) {
        block = findFreeBlock(alignedSize);
        if (block) {
            block->inUse = true;
            block->timestamp = std::chrono::steady_clock::now();
            usedMemory_ += block->size;

            stats_.defragmentations++;
            stats_.activeBlocks++;

            Logger::debug("Memory allocated after defragmentation: {} bytes", alignedSize);
            return block->pointer;
        }
    }

    // All allocation attempts failed
    stats_.allocationFailures++;
    Logger::error("Failed to allocate {} bytes from GPU memory", alignedSize);
    return nullptr;
}

void GPUMemoryManager::deallocate(void* ptr) {
    if (!ptr || !initialized_) {
        return;
    }

    std::lock_guard<std::mutex> lock(memoryMutex_);

    // Find and mark block as free
    for (auto& block : allocatedBlocks_) {
        if (block.pointer == ptr && block.inUse) {
            block.inUse = false;
            block.timestamp = std::chrono::steady_clock::now();
            usedMemory_ -= block.size;

            stats_.totalDeallocations++;
            stats_.activeBlocks--;

            Logger::debug("Memory deallocated: {} bytes", block.size);
            return;
        }
    }

    Logger::warning("Attempted to deallocate unknown pointer: {}", ptr);
}

GPUMemoryManager::MemoryBlock* GPUMemoryManager::findFreeBlock(size_t size) {
    // Search for exact or slightly larger free block
    MemoryBlock* bestBlock = nullptr;
    size_t bestSize = SIZE_MAX;

    for (auto& block : allocatedBlocks_) {
        if (!block.inUse && block.size >= size && block.size < bestSize) {
            bestBlock = &block;
            bestSize = block.size;

            // Exact match found
            if (block.size == size) {
                break;
            }
        }
    }

    return bestBlock;
}

GPUMemoryManager::MemoryBlock* GPUMemoryManager::splitBlock(size_t size) {
    // Find a larger block that can be split
    for (auto& block : allocatedBlocks_) {
        if (!block.inUse && block.size > size + minBlockSize_) {
            // Create new block for the remainder
            MemoryBlock newBlock;
            newBlock.pointer = static_cast<uint8_t*>(block.pointer) + size;
            newBlock.size = block.size - size;
            newBlock.inUse = false;
            newBlock.timestamp = std::chrono::steady_clock::now();

            // Resize original block
            block.size = size;

            // Add new block to list
            allocatedBlocks_.push_back(newBlock);

            return &block;
        }
    }

    return nullptr;
}

bool GPUMemoryManager::defragment() {
    if (fragmentationTooHigh()) {
        Logger::info("Fragmentation too high, performing defragmentation");

        // Simple defragmentation: move all free blocks to the end
        std::vector<MemoryBlock> usedBlocks;
        std::vector<MemoryBlock> freeBlocks;

        for (const auto& block : allocatedBlocks_) {
            if (block.inUse) {
                usedBlocks.push_back(block);
            } else {
                freeBlocks.push_back(block);
            }
        }

        // Rebuild allocated blocks list
        allocatedBlocks_.clear();

        // Add used blocks first
        size_t offset = 0;
        for (const auto& block : usedBlocks) {
            MemoryBlock newBlock = block;
            newBlock.pointer = static_cast<uint8_t*>(memoryPools_[0].pointer) + offset;
            offset += newBlock.size;
            allocatedBlocks_.push_back(newBlock);
        }

        // Add remaining space as one free block
        size_t remainingSpace = memoryPools_[0].size - offset;
        if (remainingSpace >= minBlockSize_) {
            MemoryBlock freeBlock;
            freeBlock.pointer = static_cast<uint8_t*>(memoryPools_[0].pointer) + offset;
            freeBlock.size = remainingSpace;
            freeBlock.inUse = false;
            freeBlock.timestamp = std::chrono::steady_clock::now();
            allocatedBlocks_.push_back(freeBlock);
        }

        return true;
    }

    return false;
}

bool GPUMemoryManager::fragmentationTooHigh() const {
    if (allocatedBlocks_.empty()) {
        return false;
    }

    size_t freeBlocks = 0;
    size_t totalFreeSize = 0;

    for (const auto& block : allocatedBlocks_) {
        if (!block.inUse) {
            freeBlocks++;
            totalFreeSize += block.size;
        }
    }

    // Calculate fragmentation ratio
    size_t totalAllocated = totalMemorySize_ - usedMemory_;
    if (totalAllocated == 0) {
        return false;
    }

    float fragmentation = 1.0f - (static_cast<float>(totalFreeSize) / totalAllocated);

    return (freeBlocks > 1 && fragmentation > fragmentationThreshold_);
}

void GPUMemoryManager::printStatsStatistics() const {
    std::lock_guard<std::mutex> lock(memoryMutex_);

    Logger::info("=== GPU Memory Manager Statistics ===");
    Logger::info("Backend: {}", backend_);
    Logger::info("Pool Size: {} MB", totalMemorySize_ / (1024 * 1024));
    Logger::info("Used Memory: {} MB", usedMemory_ / (1024 * 1024));
    Logger::info("Available Memory: {} MB", (totalMemorySize_ - usedMemory_) / (1024 * 1024));
    Logger::info("Utilization: {:.1f}%", (static_cast<double>(usedMemory_) / totalMemorySize_) * 100.0);
    Logger::info("Total Allocations: {}", stats_.totalAllocations);
    Logger::info("Total Deallocations: {}", stats_.totalDeallocations);
    Logger::info("Active Blocks: {}", stats_.activeBlocks);
    Logger::info("Pool Allocations: {}", stats_.poolAllocations);
    Logger::info("Block Splits: {}", stats_.blockSplits);
    Logger::info("Defragmentations: {}", stats_.defragmentations);
    Logger::info("Allocation Failures: {}", stats_.allocationFailures);
    Logger::info("=======================================");
}

void GPUMemoryManager::printStatistics() const {
    if (initialized_) {
        printStatsStatistics();
    }
}

GPUMemoryManager::Statistics GPUMemoryManager::getStatistics() const {
    std::lock_guard<std::mutex> lock(memoryMutex_);
    Statistics stats = stats_;
    stats.totalMemorySize = totalMemorySize_;
    stats.usedMemory = usedMemory_;
    stats.availableMemory = totalMemorySize_ - usedMemory_;
    stats.utilization = static_cast<double>(usedMemory_) / totalMemorySize_;
    return stats;
}

void GPUMemoryManager::resetStatistics() {
    std::lock_guard<std::mutex> lock(memoryMutex_);
    stats_ = Statistics{};
}

size_t GPUMemoryManager::getTotalMemory() const {
    std::lock_guard<std::mutex> lock(memoryMutex_);
    return totalMemorySize_;
}

size_t GPUMemoryManager::getUsedMemory() const {
    std::lock_guard<std::mutex> lock(memoryMutex_);
    return usedMemory_;
}

size_t GPUMemoryManager::getAvailableMemory() const {
    std::lock_guard<std::mutex> lock(memoryMutex_);
    return totalMemorySize_ - usedMemory_;
}

double GPUMemoryManager::getUtilization() const {
    std::lock_guard<std::mutex> lock(memoryMutex_);
    return totalMemorySize_ > 0 ? (static_cast<double>(usedMemory_) / totalMemorySize_) : 0.0;
}

void* GPUMemoryManager::allocateGPUMemory(size_t size) {
    if (backend_ == "CUDA") {
        return allocateCUDAMemory(size);
    } else if (backend_ == "OpenCL") {
        return allocateOpenCLMemory(size);
    } else if (backend_ == "Vulkan") {
        return allocateVulkanMemory(size);
    }
    return nullptr;
}

void GPUMemoryManager::deallocateGPUMemory(void* ptr) {
    if (!ptr) return;

    if (backend_ == "CUDA") {
        deallocateCUDAMemory(ptr);
    } else if (backend_ == "OpenCL") {
        deallocateOpenCLMemory(ptr);
    } else if (backend_ == "Vulkan") {
        deallocateVulkanMemory(ptr);
    }
}

#ifdef VORTEX_ENABLE_CUDA
void* GPUMemoryManager::allocateCUDAMemory(size_t size) {
    void* ptr = nullptr;
    cudaError_t error = cudaMalloc(&ptr, size);
    if (error != cudaSuccess) {
        Logger::error("CUDA memory allocation failed: {}", cudaGetErrorString(error));
        return nullptr;
    }
    return ptr;
}

void GPUMemoryManager::deallocateCUDAMemory(void* ptr) {
    if (ptr) {
        cudaError_t error = cudaFree(ptr);
        if (error != cudaSuccess) {
            Logger::error("CUDA memory deallocation failed: {}", cudaGetErrorString(error));
        }
    }
}

bool GPUMemoryManager::initializeCUDAResources() {
    // No additional CUDA resources needed for basic memory management
    return true;
}

void GPUMemoryManager::cleanupCUDAResources() {
    // CUDA cleanup handled by cudaFree calls
}
#endif

#ifdef VORTEX_ENABLE_OPENCL
void* GPUMemoryManager::allocateOpenCLMemory(size_t size) {
    // OpenCL memory allocation would require a context
    // For now, return nullptr as placeholder
    return nullptr;
}

void GPUMemoryManager::deallocateOpenCLMemory(void* ptr) {
    // OpenCL memory deallocation would require a context
}

bool GPUMemoryManager::initializeOpenCLResources() {
    // OpenCL memory manager would need access to CL context
    return true;
}

void GPUMemoryManager::cleanupOpenCLResources() {
    // OpenCL cleanup
}
#endif

#ifdef VORTEX_ENABLE_VULKAN
void* GPUMemoryManager::allocateVulkanMemory(size_t size) {
    // Vulkan memory allocation would require a device and memory allocator
    return nullptr;
}

void GPUMemoryManager::deallocateVulkanMemory(void* ptr) {
    // Vulkan memory deallocation
}

bool GPUMemoryManager::initializeVulkanResources() {
    // Vulkan memory manager setup
    return true;
}

void GPUMemoryManager::cleanupVulkanResources() {
    // Vulkan cleanup
}
#endif

} // namespace vortex