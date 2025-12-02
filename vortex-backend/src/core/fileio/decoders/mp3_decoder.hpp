#pragma once

#include <string>
#include <vector>
#include <optional>
#include <fstream>
#include <chrono>
#include "core/fileio/audio_file_loader.hpp"
#include "system/logger.hpp"

// Forward declaration for mpg123
typedef struct mpg123_handle_struct mpg123_handle;

namespace vortex::core::fileio {

/**
 * MP3 audio file decoder using mpg123 library
 * Supports ID3v1, ID3v2 tags and all MP3 variants
 */
class MP3Decoder {
public:
    MP3Decoder();
    ~MP3Decoder();

    /**
     * Initialize the MP3 decoder
     * @return true if initialization successful, false otherwise
     */
    bool initialize();

    /**
     * Shutdown the MP3 decoder and clean up resources
     */
    void shutdown();

    /**
     * Check if the file can be decoded as MP3
     * @param filePath Path to the audio file
     * @return true if file is supported MP3 format, false otherwise
     */
    bool canDecode(const std::string& filePath) const;

    /**
     * Decode MP3 file to PCM audio data
     * @param filePath Path to the MP3 file
     * @return AudioData if decoding successful, nullopt otherwise
     */
    std::optional<AudioData> decode(const std::string& filePath);

    /**
     * Extract metadata from MP3 file
     * @param filePath Path to the MP3 file
     * @param metadata Metadata structure to fill
     * @return true if metadata extraction successful, false otherwise
     */
    bool extractMetadata(const std::string& filePath, AudioMetadata& metadata);

private:
    mpg123_handle* handle_;
    bool initialized_;

    /**
     * Check if data buffer contains MP3 format signature
     * @param data Data buffer to check
     * @param size Size of data buffer
     * @return true if MP3 format detected, false otherwise
     */
    bool isMP3Format(const uint8_t* data, size_t size) const;

    /**
     * Extract ID3v2 tag metadata from file
     * @param file File stream positioned at beginning
     * @param metadata Metadata structure to fill
     */
    void extractID3v2Metadata(std::ifstream& file, AudioMetadata& metadata);

    /**
     * Parse ID3v2 frame data and extract metadata
     * @param data Frame data buffer
     * @param size Size of frame data
     * @param metadata Metadata structure to fill
     */
    void parseID3v2Frames(const uint8_t* data, size_t size, AudioMetadata& metadata);
};

} // namespace vortex::core::fileio