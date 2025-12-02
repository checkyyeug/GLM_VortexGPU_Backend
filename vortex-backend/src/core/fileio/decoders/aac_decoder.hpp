#pragma once

#include <string>
#include <vector>
#include <optional>
#include <fstream>
#include <chrono>
#include "core/fileio/audio_file_loader.hpp"
#include "system/logger.hpp"

// Forward declarations for FAAD2
typedef void* NeAACDecHandle;
typedef struct NeAACDecConfiguration_t* NeAACDecConfigurationPtr;
typedef struct NeAACDecFrameInfo_t NeAACDecFrameInfo;

namespace vortex::core::fileio {

/**
 * AAC (Advanced Audio Coding) decoder
 * Supports AAC decoding in various container formats (ADTS, raw AAC)
 * Uses FAAD2 library for high-quality AAC decoding
 * Extracts ID3 metadata and audio properties
 */
class AACDecoder {
public:
    AACDecoder();
    ~AACDecoder();

    /**
     * Initialize the AAC decoder
     * @return true if initialization successful, false otherwise
     */
    bool initialize();

    /**
     * Shutdown the AAC decoder and clean up resources
     */
    void shutdown();

    /**
     * Check if the file can be decoded as AAC
     * @param filePath Path to the audio file
     * @return true if file is supported AAC format, false otherwise
     */
    bool canDecode(const std::string& filePath) const;

    /**
     * Decode AAC file to PCM audio data
     * @param filePath Path to the AAC file
     * @return AudioData if decoding successful, nullopt otherwise
     */
    std::optional<AudioData> decode(const std::string& filePath);

    /**
     * Extract metadata from AAC file
     * @param filePath Path to the AAC file
     * @param metadata Metadata structure to fill
     * @return true if metadata extraction successful, false otherwise
     */
    bool extractMetadata(const std::string& filePath, AudioMetadata& metadata);

private:
    bool initialized_;
    NeAACDecHandle decoderHandle_;

    /**
     * Check if data buffer contains AAC format signature
     * @param data Data buffer to check
     * @param size Size of data buffer
     * @return true if AAC format detected, false otherwise
     */
    bool isAACFormat(const uint8_t* data, size_t size) const;

    /**
     * Extract ID3 metadata from AAC file
     * @param data File data buffer
     * @param size Size of data buffer
     * @param metadata Metadata structure to fill
     */
    void extractID3Metadata(const uint8_t* data, size_t size, AudioMetadata& metadata);

    /**
     * Parse ID3v2 frames and extract metadata
     * @param data Frame data buffer
     * @param size Size of frame data
     * @param metadata Metadata structure to fill
     */
    void parseID3v2Frames(const uint8_t* data, size_t size, AudioMetadata& metadata);
};

} // namespace vortex::core::fileio