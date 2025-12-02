#pragma once

#include "format_detector.hpp"
#include <memory>
#include <vector>
#include <future>
#include <chrono>
#include <mutex>
#include <map>
#include <queue>
#include <atomic>
#include <random>

namespace vortex {

struct AudioData {
    std::vector<float> data;
    size_t channels = 0;
    size_t numSamples = 0;
    double sampleRate = 0.0;
    int bitDepth = 0;
    double durationSeconds = 0.0;
};

struct LoadOptions {
    // Basic loading options
    int targetChannels = 0;        // 0 = original
    double targetSampleRate = 0.0; // 0 = original
    int targetBitDepth = 0;        // 0 = original
    size_t maxSamples = 0;         // 0 = no limit

    // Processing options
    float gain = 1.0f;
    bool normalize = false;
    bool bypassCache = false;

    // Channel mapping (empty = no mapping)
    std::vector<int> channelMapping;
};

struct LoadResult {
    bool success = false;
    std::string filePath;
    std::string error;
    AudioData audioData;
    AudioMetadata metadata;
    bool fromCache = false;
};

/**
 * @brief High-performance multi-format audio file loader with caching
 *
 * This class provides comprehensive audio file loading capabilities supporting
 * multiple formats with resampling, channel mapping, normalization, and caching.
 * It's optimized for both single-file and batch loading operations with
 * asynchronous support and memory-efficient caching.
 */
class AudioFileLoader {
public:
    AudioFileLoader();
    ~AudioFileLoader();

    // Single file loading
    LoadResult loadAudioFile(const std::string& filePath, const LoadOptions& options = LoadOptions{});
    std::future<LoadResult> loadAudioFileAsync(const std::string& filePath, const LoadOptions& options = LoadOptions{});

    // Batch loading
    std::vector<std::future<LoadResult>> loadMultipleFilesAsync(
        const std::vector<std::string>& filePaths,
        const LoadOptions& options = LoadOptions{});

    // Cache management
    void enableCache(bool enabled);
    void setCacheSize(size_t maxSizeBytes);
    void clearCache();
    size_t getCacheSize() const;
    size_t getCacheEntryCount() const;

    // Configuration
    void setMaxConcurrentLoads(int maxLoads) { maxConcurrentLoads_ = maxLoads; }
    int getMaxConcurrentLoads() const { return maxConcurrentLoads_; }

    // Format support
    bool isFormatSupported(AudioFormat format) const;
    std::vector<AudioFormat> getSupportedFormats() const;

private:
    struct CacheEntry {
        AudioData audioData;
        AudioMetadata metadata;
        std::chrono::steady_clock::time_point timestamp;
        size_t estimatedSize;
    };

    // Core components
    juce::AudioFormatManager formatManager_;

    // Configuration
    int maxConcurrentLoads_;
    bool cacheEnabled_;
    size_t maxCacheSize_;
    size_t currentCacheSize_;

    // Cache storage
    mutable std::mutex cacheMutex_;
    std::map<std::string, CacheEntry> audioCache_;

    // Loading methods
    bool loadWithJUCE(const juce::File& file, LoadResult& result, const LoadOptions& options);
    bool postProcessAudioData(LoadResult& result, const LoadOptions& options);

    // Audio processing
    void normalizeAudioData(AudioData& audioData);
    void applyDithering(AudioData& audioData, int targetBitDepth);
    void applyChannelMapping(AudioData& audioData, const std::vector<int>& mapping);

    // Cache methods
    void cacheAudioData(const std::string& filePath, const LoadResult& result);
    std::unique_ptr<CacheEntry> getCachedData(const std::string& filePath);
};

} // namespace vortex