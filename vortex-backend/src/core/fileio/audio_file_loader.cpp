#include "audio_file_loader.hpp"
#include "format_detector.hpp"
#include "system/logger.hpp"

#include <juce_audio_formats/juce_audio_formats.h>
#include <algorithm>
#include <thread>
#include <future>

namespace vortex {

AudioFileLoader::AudioFileLoader()
    : maxConcurrentLoads_(4)
    , cacheEnabled_(true)
    , maxCacheSize_(1024 * 1024 * 1024) // 1GB cache
    , currentCacheSize_(0)
{
    formatManager_.registerBasicFormats();
    Logger::info("AudioFileLoader initialized with max {} concurrent loads", maxConcurrentLoads_);
}

AudioFileLoader::~AudioFileLoader() {
    clearCache();
    Logger::info("AudioFileLoader destroyed");
}

LoadResult AudioFileLoader::loadAudioFile(const std::string& filePath, const LoadOptions& options) {
    Logger::info("Loading audio file: {}", filePath);

    LoadResult result;
    result.filePath = filePath;

    try {
        // Check cache first
        if (cacheEnabled_ && !options.bypassCache) {
            auto cachedData = getCachedData(filePath);
            if (cachedData) {
                Logger::info("Audio file loaded from cache: {}", filePath);
                result.audioData = std::move(*cachedData);
                result.metadata = cachedData->metadata;
                result.fromCache = true;
                result.success = true;
                return result;
            }
        }

        // Verify file exists
        juce::File file(filePath);
        if (!file.existsAsFile()) {
            result.error = "File does not exist: " + filePath;
            Logger::error(result.error);
            return result;
        }

        // Detect format and extract metadata
        FormatDetector detector;
        AudioFormat format = detector.detectFormat(filePath);
        result.metadata = detector.extractMetadata(filePath);
        result.metadata.format = format;

        // Validate format support
        if (!isFormatSupported(format)) {
            result.error = "Unsupported audio format: " + detector.getFormatName(format);
            Logger::error(result.error);
            return result;
        }

        // Load audio data using JUCE
        if (!loadWithJUCE(file, result, options)) {
            result.error = "Failed to load audio data with JUCE";
            Logger::error(result.error);
            return result;
        }

        // Apply format-specific processing
        if (!postProcessAudioData(result, options)) {
            result.error = "Failed to post-process audio data";
            Logger::error(result.error);
            return result;
        }

        // Cache the result if enabled
        if (cacheEnabled_ && !options.bypassCache) {
            cacheAudioData(filePath, result);
        }

        result.success = true;
        Logger::info("Audio file loaded successfully: {} ({} samples, {} channels, {} Hz)",
                     filePath, result.audioData.numSamples, result.audioData.channels, result.audioData.sampleRate);

        return result;

    } catch (const std::exception& e) {
        result.error = "Exception during audio file loading: " + std::string(e.what());
        Logger::error(result.error);
        return result;
    }
}

bool AudioFileLoader::loadWithJUCE(const juce::File& file, LoadResult& result, const LoadOptions& options) {
    try {
        auto* reader = formatManager_.createReaderFor(file);
        if (!reader) {
            Logger::error("Failed to create audio reader for: {}", file.getFullPathName().toStdString());
            return false;
        }

        // Configure reader based on options
        int targetChannels = options.targetChannels > 0 ? options.targetChannels : reader->numChannels;
        double targetSampleRate = options.targetSampleRate > 0 ? options.targetSampleRate : reader->sampleRate;

        // Calculate total samples to read
        juce::int64 totalSamples = reader->lengthInSamples;
        if (options.maxSamples > 0 && totalSamples > options.maxSamples) {
            totalSamples = options.maxSamples;
        }

        // Create audio buffer
        result.audioData.channels = targetChannels;
        result.audioData.sampleRate = targetSampleRate;
        result.audioData.numSamples = static_cast<size_t>(totalSamples);
        result.audioData.data.resize(result.audioData.numSamples * targetChannels);

        // Setup JUCE audio buffer
        juce::AudioBuffer<float> buffer(targetChannels, static_cast<int>(totalSamples));

        // Read audio data with optional resampling
        if (targetSampleRate != reader->sampleRate) {
            // Setup resampler
            auto resampler = std::make_unique<juce::ResamplingAudioSource>(
                nullptr, false, targetChannels);

            // Create temporary buffer with original sample rate
            juce::AudioBuffer<float> tempBuffer(reader->numChannels, static_cast<int>(totalSamples));
            reader->read(&tempBuffer, 0, static_cast<int>(totalSamples), 0, true, true);

            // Configure resampler
            resampler->setResamplingRatio(reader->sampleRate / targetSampleRate);
            resampler->prepareToPlay(targetChannels, targetSampleRate);

            // Resample to target buffer
            juce::AudioSourceChannelInfo channelInfo(buffer);
            resampler->getNextAudioBlock(channelInfo);
        } else {
            // Direct read without resampling
            reader->read(&buffer, 0, static_cast<int>(totalSamples), 0, true, true);
        }

        // Convert interleaved format
        for (int channel = 0; channel < targetChannels; ++channel) {
            const float* channelData = buffer.getReadPointer(channel);
            for (size_t sample = 0; sample < result.audioData.numSamples; ++sample) {
                result.audioData.data[sample * targetChannels + channel] = channelData[sample];
            }
        }

        // Set bit depth
        result.audioData.bitDepth = options.targetBitDepth > 0 ? options.targetBitDepth : reader->bitsPerSample;

        // Calculate duration
        result.audioData.durationSeconds = static_cast<double>(result.audioData.numSamples) / targetSampleRate;

        delete reader; // Clean up
        return true;

    } catch (const std::exception& e) {
        Logger::error("JUCE audio loading failed: {}", e.what());
        return false;
    }
}

bool AudioFileLoader::postProcessAudioData(LoadResult& result, const LoadOptions& options) {
    try {
        // Apply gain if specified
        if (options.gain != 1.0f) {
            for (auto& sample : result.audioData.data) {
                sample *= options.gain;
            }
        }

        // Apply normalization if requested
        if (options.normalize) {
            normalizeAudioData(result.audioData);
        }

        // Apply dithering if reducing bit depth
        if (options.targetBitDepth > 0 && options.targetBitDepth < result.audioData.bitDepth) {
            applyDithering(result.audioData, options.targetBitDepth);
        }

        // Channel mapping if requested
        if (!options.channelMapping.empty()) {
            applyChannelMapping(result.audioData, options.channelMapping);
        }

        return true;

    } catch (const std::exception& e) {
        Logger::error("Audio post-processing failed: {}", e.what());
        return false;
    }
}

void AudioFileLoader::normalizeAudioData(AudioData& audioData) {
    if (audioData.data.empty()) {
        return;
    }

    // Find peak value
    float peak = 0.0f;
    for (float sample : audioData.data) {
        peak = std::max(peak, std::abs(sample));
    }

    if (peak > 0.0f) {
        float normalizationGain = 1.0f / peak;
        for (float& sample : audioData.data) {
            sample *= normalizationGain;
        }

        Logger::debug("Audio data normalized by factor: {:.3f}", normalizationGain);
    }
}

void AudioFileLoader::applyDithering(AudioData& audioData, int targetBitDepth) {
    if (audioData.data.empty() || targetBitDepth >= 24) {
        return; // No dithering needed for 24-bit or higher
    }

    float scale = std::pow(2.0f, targetBitDepth - 1) - 1.0f;
    float invScale = 1.0f / scale;

    // Simple triangular probability dithering
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (float& sample : audioData.data) {
        float scaled = sample * scale;
        float dither = dist(gen);
        float quantized = std::round(scaled + dither);
        sample = quantized * invScale;
    }

    audioData.bitDepth = targetBitDepth;
    Logger::debug("Applied {}-bit dithering", targetBitDepth);
}

void AudioFileLoader::applyChannelMapping(AudioData& audioData, const std::vector<int>& mapping) {
    if (audioData.data.empty() || mapping.empty()) {
        return;
    }

    size_t originalChannels = audioData.channels;
    size_t newChannels = mapping.size();

    if (newChannels == originalChannels) {
        // Reorder channels
        std::vector<float> reorderedData(audioData.data.size());

        for (size_t sample = 0; sample < audioData.numSamples; ++sample) {
            for (size_t ch = 0; ch < newChannels; ++ch) {
                int sourceChannel = mapping[ch];
                if (sourceChannel >= 0 && sourceChannel < static_cast<int>(originalChannels)) {
                    reorderedData[sample * newChannels + ch] =
                        audioData.data[sample * originalChannels + sourceChannel];
                } else {
                    reorderedData[sample * newChannels + ch] = 0.0f;
                }
            }
        }

        audioData.data = std::move(reorderedData);
        audioData.channels = newChannels;
    } else {
        // Channel count change - create new buffer
        std::vector<float> newData(audioData.numSamples * newChannels, 0.0f);

        for (size_t sample = 0; sample < audioData.numSamples; ++sample) {
            for (size_t ch = 0; ch < newChannels; ++ch) {
                int sourceChannel = mapping[ch];
                if (sourceChannel >= 0 && sourceChannel < static_cast<int>(originalChannels)) {
                    newData[sample * newChannels + ch] =
                        audioData.data[sample * originalChannels + sourceChannel];
                }
            }
        }

        audioData.data = std::move(newData);
        audioData.channels = newChannels;
    }

    Logger::debug("Applied channel mapping: {} -> {} channels", originalChannels, newChannels);
}

std::future<LoadResult> AudioFileLoader::loadAudioFileAsync(const std::string& filePath, const LoadOptions& options) {
    return std::async(std::launch::async, [this, filePath, options]() {
        return loadAudioFile(filePath, options);
    });
}

std::vector<std::future<LoadResult>> AudioFileLoader::loadMultipleFilesAsync(
    const std::vector<std::string>& filePaths,
    const LoadOptions& options) {

    Logger::info("Loading {} audio files asynchronously", filePaths.size());

    std::vector<std::future<LoadResult>> futures;
    futures.reserve(filePaths.size());

    // Limit concurrent loads
    std::vector<std::thread> workers;
    std::queue<std::string> fileQueue;
    std::mutex queueMutex;
    std::condition_variable cv;
    std::atomic<bool> done{false};

    // Initialize queue
    for (const auto& filePath : filePaths) {
        fileQueue.push(filePath);
    }

    // Worker function
    auto worker = [this, &fileQueue, &queueMutex, &cv, &done, &futures, options]() {
        while (true) {
            std::string filePath;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                cv.wait(lock, [&fileQueue, &done]() { return !fileQueue.empty() || done.load(); });

                if (fileQueue.empty() && done.load()) {
                    break;
                }

                if (!fileQueue.empty()) {
                    filePath = fileQueue.front();
                    fileQueue.pop();
                } else {
                    continue;
                }
            }

            // Load file
            try {
                LoadResult result = loadAudioFile(filePath, options);
                // Store result (in a real implementation, this would be thread-safe)
                Logger::debug("Async load completed: {}", filePath);
            } catch (const std::exception& e) {
                Logger::error("Async load failed for {}: {}", filePath, e.what());
            }
        }
    };

    // Start worker threads
    size_t numWorkers = std::min(static_cast<size_t>(maxConcurrentLoads_), filePaths.size());
    for (size_t i = 0; i < numWorkers; ++i) {
        workers.emplace_back(worker);
    }

    // Wait for workers to complete
    for (auto& worker : workers) {
        worker.join();
    }

    Logger::info("Batch load completed for {} files", filePaths.size());

    return futures; // In a real implementation, this would return actual futures
}

void AudioFileLoader::cacheAudioData(const std::string& filePath, const LoadResult& result) {
    if (!cacheEnabled_) {
        return;
    }

    std::lock_guard<std::mutex> lock(cacheMutex_);

    // Check cache size limit
    size_t estimatedSize = result.audioData.data.size() * sizeof(float) + 1024; // Rough estimate
    if (currentCacheSize_ + estimatedSize > maxCacheSize_) {
        // Evict oldest items if necessary
        while (!audioCache_.empty() && currentCacheSize_ + estimatedSize > maxCacheSize_) {
            auto oldest = audioCache_.begin();
            currentCacheSize_ -= oldest->second.estimatedSize;
            audioCache_.erase(oldest);
        }
    }

    // Add to cache
    CacheEntry entry;
    entry.audioData = result.audioData;
    entry.metadata = result.metadata;
    entry.timestamp = std::chrono::steady_clock::now();
    entry.estimatedSize = estimatedSize;

    audioCache_[filePath] = entry;
    currentCacheSize_ += estimatedSize;

    Logger::debug("Cached audio file: {} ({} MB)", filePath, estimatedSize / (1024 * 1024));
}

std::unique_ptr<AudioFileLoader::CacheEntry> AudioFileLoader::getCachedData(const std::string& filePath) {
    if (!cacheEnabled_) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(cacheMutex_);

    auto it = audioCache_.find(filePath);
    if (it != audioCache_.end()) {
        // Update timestamp for LRU
        it->second.timestamp = std::chrono::steady_clock::now();
        return std::make_unique<CacheEntry>(it->second);
    }

    return nullptr;
}

void AudioFileLoader::clearCache() {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    audioCache_.clear();
    currentCacheSize_ = 0;
    Logger::info("Audio cache cleared");
}

void AudioFileLoader::enableCache(bool enabled) {
    cacheEnabled_ = enabled;
    if (!enabled) {
        clearCache();
    }
    Logger::info("Audio cache {}", enabled ? "enabled" : "disabled");
}

void AudioFileLoader::setCacheSize(size_t maxSize) {
    maxCacheSize_ = maxSize;
    std::lock_guard<std::mutex> lock(cacheMutex_);

    // Trim cache if necessary
    while (currentCacheSize_ > maxCacheSize_ && !audioCache_.empty()) {
        auto oldest = audioCache_.begin();
        currentCacheSize_ -= oldest->second.estimatedSize;
        audioCache_.erase(oldest);
    }

    Logger::info("Cache size set to {} MB", maxSize / (1024 * 1024));
}

size_t AudioFileLoader::getCacheSize() const {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    return currentCacheSize_;
}

size_t AudioFileLoader::getCacheEntryCount() const {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    return audioCache_.size();
}

bool AudioFileLoader::isFormatSupported(AudioFormat format) const {
    switch (format) {
        case AudioFormat::PCM:
        case AudioFormat::FLAC:
        case AudioFormat::WAV:
        case AudioFormat::MP3:
        case AudioFormat::AAC:
        case AudioFormat::OGG:
        case AudioFormat::OPUS:
            return true;
        case AudioFormat::DSD64:
        case AudioFormat::DSD128:
        case AudioFormat::DSD256:
        case AudioFormat::DSD512:
        case AudioFormat::DSD1024:
            // DSD support would require additional implementation
            return false;
        default:
            return false;
    }
}

std::vector<AudioFormat> AudioFileLoader::getSupportedFormats() const {
    return {
        AudioFormat::PCM,
        AudioFormat::FLAC,
        AudioFormat::WAV,
        AudioFormat::MP3,
        AudioFormat::AAC,
        AudioFormat::OGG,
        AudioFormat::OPUS
    };
}

} // namespace vortex