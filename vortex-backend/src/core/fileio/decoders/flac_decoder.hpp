#pragma once

#include <string>
#include <vector>
#include <optional>
#include <fstream>
#include <chrono>
#include "core/fileio/audio_file_loader.hpp"
#include "system/logger.hpp"

namespace vortex::core::fileio {

/**
 * FLAC audio file decoder
 * Supports lossless FLAC decoding with full metadata extraction
 * Uses libFLAC for high-quality decoding
 */
class FLACDecoder {
public:
    FLACDecoder();
    ~FLACDecoder();

    /**
     * Initialize the FLAC decoder
     * @return true if initialization successful, false otherwise
     */
    bool initialize();

    /**
     * Shutdown the FLAC decoder and clean up resources
     */
    void shutdown();

    /**
     * Check if the file can be decoded as FLAC
     * @param filePath Path to the audio file
     * @return true if file is supported FLAC format, false otherwise
     */
    bool canDecode(const std::string& filePath) const;

    /**
     * Decode FLAC file to PCM audio data
     * @param filePath Path to the FLAC file
     * @return AudioData if decoding successful, nullopt otherwise
     */
    std::optional<AudioData> decode(const std::string& filePath);

    /**
     * Extract metadata from FLAC file
     * @param filePath Path to the FLAC file
     * @param metadata Metadata structure to fill
     * @return true if metadata extraction successful, false otherwise
     */
    bool extractMetadata(const std::string& filePath, AudioMetadata& metadata);

private:
    bool initialized_;

    // Forward declaration for implementation class
    class FLACDecoderImpl;
};

} // namespace vortex::core::fileio