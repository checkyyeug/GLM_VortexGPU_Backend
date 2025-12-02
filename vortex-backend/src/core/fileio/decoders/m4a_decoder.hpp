#pragma once

#include <string>
#include <vector>
#include <optional>
#include <fstream>
#include <chrono>
#include <cstdint>
#include <memory>
#include "core/fileio/audio_file_loader.hpp"
#include "system/logger.hpp"

namespace vortex::core::fileio {

// Forward declarations
class AACDecoder;
class ALACDecoder;

/**
 * M4A (MPEG-4 Audio) container decoder
 * Supports M4A files containing AAC or ALAC audio
 * Acts as a container format handler that delegates to specific decoders
 * Extracts iTunes metadata and audio properties
 */
class M4ADecoder {
public:
    M4ADecoder();
    ~M4ADecoder();

    /**
     * Initialize the M4A decoder
     * @return true if initialization successful, false otherwise
     */
    bool initialize();

    /**
     * Shutdown the M4A decoder and clean up resources
     */
    void shutdown();

    /**
     * Check if the file can be decoded as M4A
     * @param filePath Path to the audio file
     * @return true if file is supported M4A format, false otherwise
     */
    bool canDecode(const std::string& filePath) const;

    /**
     * Decode M4A file to PCM audio data
     * @param filePath Path to the M4A file
     * @return AudioData if decoding successful, nullopt otherwise
     */
    std::optional<AudioData> decode(const std::string& filePath);

    /**
     * Extract metadata from M4A file
     * @param filePath Path to the M4A file
     * @param metadata Metadata structure to fill
     * @return true if metadata extraction successful, false otherwise
     */
    bool extractMetadata(const std::string& filePath, AudioMetadata& metadata);

private:
    bool initialized_;
    std::unique_ptr<AACDecoder> aacDecoder_;
    std::unique_ptr<ALACDecoder> alacDecoder_;

    // Audio codec enumeration
    enum class AudioCodec {
        UNKNOWN,
        AAC,
        ALAC
    };

    // M4A container information structure
    struct M4AContainerInfo {
        AudioCodec audioCodec;
        bool hasAudioTrack;

        M4AContainerInfo() : audioCodec(AudioCodec::UNKNOWN), hasAudioTrack(false) {}
    };

    /**
     * Check if data buffer contains M4A format signature
     * @param data Data buffer to check
     * @param size Size of data buffer
     * @return true if M4A format detected, false otherwise
     */
    bool isM4AFormat(const uint8_t* data, size_t size) const;

    // MP4 container parsing functions
    bool parseM4AContainer(const std::string& filePath, M4AContainerInfo& containerInfo);
    bool parseMovieAtom(std::ifstream& file, uint64_t atomSize, M4AContainerInfo& containerInfo);
    bool parseTrackAtom(std::ifstream& file, uint64_t atomSize, M4AContainerInfo& containerInfo);
    bool parseMediaAtom(std::ifstream& file, uint64_t atomSize, M4AContainerInfo& containerInfo);
    bool parseHandlerAtom(std::ifstream& file, uint64_t atomSize, M4AContainerInfo& containerInfo);
    bool parseMediaInfoAtom(std::ifstream& file, uint64_t atomSize, M4AContainerInfo& containerInfo);
    bool parseSampleTableAtom(std::ifstream& file, uint64_t atomSize, M4AContainerInfo& containerInfo);
    bool parseSampleDescriptionAtom(std::ifstream& file, uint64_t atomSize, M4AContainerInfo& containerInfo);

    // iTunes metadata extraction
    void extractM4AMetadata(const std::string& filePath, AudioMetadata& metadata);
    void parseiTunesMetadataList(std::ifstream& file, uint64_t atomSize, AudioMetadata& metadata);
    void readMetadataText(std::ifstream& file, uint64_t size, std::optional<std::string>& field);
    void readMetadataYear(std::ifstream& file, uint64_t size, std::optional<uint16_t>& field);
    void readMetadataTrack(std::ifstream& file, uint64_t size, std::optional<uint16_t>& field);
};

} // namespace vortex::core::fileio