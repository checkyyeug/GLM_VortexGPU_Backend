#pragma once

#include <string>
#include <vector>
#include <optional>
#include <fstream>
#include <chrono>
#include <cstdint>
#include "core/fileio/audio_file_loader.hpp"
#include "system/logger.hpp"

// Forward declarations for libogg and libvorbis
struct ogg_page;
struct ogg_packet;
struct ogg_stream_state;
struct ogg_sync_state;
struct vorbis_info;
struct vorbis_comment;
struct vorbis_dsp_state;
struct vorbis_block;

namespace vortex::core::fileio {

/**
 * OGG/Vorbis audio decoder
 * Supports OGG container with Vorbis audio codec
 * Uses libogg and libvorbis for high-quality decoding
 * Extracts Vorbis comment metadata
 */
class OGGDecoder {
public:
    OGGDecoder();
    ~OGGDecoder();

    /**
     * Initialize the OGG decoder
     * @return true if initialization successful, false otherwise
     */
    bool initialize();

    /**
     * Shutdown the OGG decoder and clean up resources
     */
    void shutdown();

    /**
     * Check if the file can be decoded as OGG
     * @param filePath Path to the audio file
     * @return true if file is supported OGG format, false otherwise
     */
    bool canDecode(const std::string& filePath) const;

    /**
     * Decode OGG file to PCM audio data
     * @param filePath Path to the OGG file
     * @return AudioData if decoding successful, nullopt otherwise
     */
    std::optional<AudioData> decode(const std::string& filePath);

    /**
     * Extract metadata from OGG file
     * @param filePath Path to the OGG file
     * @param metadata Metadata structure to fill
     * @return true if metadata extraction successful, false otherwise
     */
    bool extractMetadata(const std::string& filePath, AudioMetadata& metadata);

private:
    bool initialized_;

    // OGG/Vorbis decoder state
    ogg_sync_state oggSyncState_;
    vorbis_info vorbisInfo_;
    vorbis_comment vorbisComment_;
    vorbis_dsp_state vorbisDspState_;
    vorbis_block vorbisBlock_;

    // File reading context structure
    struct FileReadContext {
        const uint8_t* data;
        size_t size;
        size_t position;

        FileReadContext() : data(nullptr), size(0), position(0) {}
    };

    /**
     * Process OGG/Vorbis headers (identification, comment, setup)
     * @param context File reading context
     * @return true if headers processed successfully, false otherwise
     */
    bool processOGGHeaders(FileReadContext& context);

    /**
     * Decode OGG/Vorbis audio data
     * @param context File reading context
     * @param audioData Audio data structure to fill
     * @return true if decoding successful, false otherwise
     */
    bool decodeOGGData(FileReadContext& context, AudioData& audioData);

    /**
     * Read OGG page from file data
     * @param context File reading context
     * @param page Output OGG page structure
     * @return true if page read successfully, false otherwise
     */
    bool readOGGPage(FileReadContext& context, ogg_page& page);

    /**
     * Extract Vorbis comment metadata
     * @param metadata Metadata structure to fill
     */
    void extractVorbisComments(AudioMetadata& metadata);

    /**
     * Cleanup decoder state
     */
    void cleanupDecoder();
};

} // namespace vortex::core::fileio