#pragma once

#include "audio_types.hpp"
#include "network_types.hpp"

#include <memory>
#include <string>
#include <vector>
#include <fstream>

namespace vortex {

/**
 * @brief Base class for audio file decoders
 *
 * This abstract class defines the interface for all audio format decoders,
 * providing support for lossless and lossy formats from MP3 to DSD1024.
 */
class AudioFileDecoder {
public:
    AudioFileDecoder() = default;
    virtual ~AudioFileDecoder() = default;

    // Decoder identification
    virtual std::string getFormatName() const = 0;
    virtual std::vector<std::string> getSupportedExtensions() const = 0;
    virtual bool canHandleFile(const std::string& filePath) const = 0;

    // File operations
    virtual bool loadFile(const std::string& filePath) = 0;
    virtual void close() = 0;
    virtual bool isLoaded() const = 0;

    // Audio data access
    virtual AudioMetadata getMetadata() const = 0;
    virtual std::vector<float> getAudioData(uint64_t startSample = 0, uint64_t numSamples = 0) = 0;
    virtual uint64_t getTotalSamples() const = 0;
    virtual uint32_t getSampleRate() const = 0;
    virtual uint16_t getChannels() const = 0;
    virtual uint16_t getBitDepth() const = 0;

    // Streaming support
    virtual bool enableStreaming(uint32_t bufferSize = 4096) = 0;
    virtual std::vector<float> readNextSamples() = 0;
    virtual bool isEndOfFile() const = 0;
    virtual void seek(uint64_t samplePosition) = 0;

    // Format-specific features
    virtual bool supportsGaplessPlayback() const { return false; }
    virtual bool supportsMetadataEditing() const { return false; }
    virtual bool supportsCueSheets() const { return false; }

    // Error handling
    virtual std::string getLastError() const = 0;
    virtual bool hasErrors() const = 0;

protected:
    std::string filePath_;
    bool loaded_ = false;
    std::string lastError_;
};

/**
 * @brief Main audio file loader with format auto-detection
 */
class AudioFileLoader {
public:
    AudioFileLoader();
    ~AudioFileLoader();

    // File loading
    bool load(const std::string& filePath);
    void close();
    bool isLoaded() const;

    // File information
    std::string getFileName() const;
    std::string getFilePath() const;
    std::string getFileExtension() const;
    uint64_t getFileSize() const;

    // Audio data access
    AudioMetadata getMetadata() const;
    std::vector<float> getAudioData(uint64_t startSample = 0, uint64_t numSamples = 0);
    std::vector<float> getEntireAudioData();

    // Format information
    AudioFormat getFormat() const;
    std::string getFormatName() const;
    std::string getCodec() const;

    // Audio properties
    uint64_t getTotalSamples() const;
    uint32_t getSampleRate() const;
    uint16_t getChannels() const;
    uint16_t getBitDepth() const;
    Duration getDuration() const;

    // Streaming support
    bool enableStreaming(uint32_t bufferSize = 4096);
    std::vector<float> readNextSamples();
    bool isEndOfFile() const;
    void seek(uint64_t samplePosition);

    // Validation
    bool isValidAudioFile(const std::string& filePath) const;
    static bool isExtensionSupported(const std::string& extension);

    // Error handling
    std::string getLastError() const;
    bool hasErrors() const;

    // Decoder management
    static void registerDecoder(std::unique_ptr<AudioFileDecoder> decoder);
    static std::vector<AudioFormat> getSupportedFormats();

private:
    std::unique_ptr<AudioFileDecoder> createDecoderForFile(const std::string& filePath);
    void updateFileProperties();

    std::string filePath_;
    std::string fileName_;
    std::string extension_;
    uint64_t fileSize_ = 0;

    std::unique_ptr<AudioFileDecoder> currentDecoder_;
    bool streamingEnabled_ = false;

    // Static decoder registry
    static std::vector<std::unique_ptr<AudioFileDecoder>> registeredDecoders_;
    static bool decodersInitialized_;
};

/**
 * @brief High-resolution audio decoder for DSD1024 and professional formats
 */
class HighResolutionDecoder : public AudioFileDecoder {
public:
    HighResolutionDecoder();
    ~HighResolutionDecoder() override;

    bool loadFile(const std::string& filePath) override;
    void close() override;
    bool isLoaded() const override;

    AudioMetadata getMetadata() const override;
    std::vector<float> getAudioData(uint64_t startSample = 0, uint64_t numSamples = 0) override;
    uint64_t getTotalSamples() const override;
    uint32_t getSampleRate() const override;
    uint16_t getChannels() const override;
    uint16_t getBitDepth() const override;

    bool enableStreaming(uint32_t bufferSize = 4096) override;
    std::vector<float> readNextSamples() override;
    bool isEndOfFile() const override;
    void seek(uint64_t samplePosition) override;

    std::string getLastError() const override;
    bool hasErrors() const override;

protected:
    // DSD processing
    virtual bool processDSDBitstream(const uint8_t* dsdData, size_t dsdSize,
                                    std::vector<float>& pcmData) = 0;
    virtual bool validateDSDFormat(const std::string& filePath) = 0;

    // High-resolution processing
    void applyDSDLowPassFilter(std::vector<float>& audioData);
    void applyNoiseShaping(std::vector<float>& audioData, uint16_t targetBitDepth);

    std::ifstream fileStream_;
    AudioMetadata metadata_;
    std::vector<float> audioCache_;
    bool streamingMode_ = false;
    uint32_t streamingBufferSize_ = 4096;
    uint64_t currentSamplePosition_ = 0;
    std::string lastError_;
};

} // namespace vortex