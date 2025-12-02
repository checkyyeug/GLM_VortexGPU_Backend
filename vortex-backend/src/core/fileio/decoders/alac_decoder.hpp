#pragma once

#include <string>
#include <vector>
#include <optional>
#include <fstream>
#include <chrono>
#include <cstdint>
#include "core/fileio/audio_file_loader.hpp"
#include "system/logger.hpp"

namespace vortex::core::fileio {

/**
 * ALAC (Apple Lossless Audio Codec) decoder
 * Supports lossless ALAC decoding within MP4 containers
 * Extracts iTunes metadata and audio properties
 */
class ALACDecoder {
public:
    ALACDecoder();
    ~ALACDecoder();

    /**
     * Initialize the ALAC decoder
     * @return true if initialization successful, false otherwise
     */
    bool initialize();

    /**
     * Shutdown the ALAC decoder and clean up resources
     */
    void shutdown();

    /**
     * Check if the file can be decoded as ALAC
     * @param filePath Path to the audio file
     * @return true if file is supported ALAC format, false otherwise
     */
    bool canDecode(const std::string& filePath) const;

    /**
     * Decode ALAC file to PCM audio data
     * @param filePath Path to the ALAC file
     * @return AudioData if decoding successful, nullopt otherwise
     */
    std::optional<AudioData> decode(const std::string& filePath);

    /**
     * Extract metadata from ALAC file
     * @param filePath Path to the ALAC file
     * @param metadata Metadata structure to fill
     * @return true if metadata extraction successful, false otherwise
     */
    bool extractMetadata(const std::string& filePath, AudioMetadata& metadata);

private:
    bool initialized_;

    // ALAC configuration structure
    struct ALACConfig {
        uint32_t sampleRate;
        uint16_t channels;
        uint8_t bitDepth;
        uint32_t framesPerPacket;

        ALACConfig() : sampleRate(44100), channels(2), bitDepth(16), framesPerPacket(4096) {}
    };

    // MP4 container parsing structures
    struct MP4Container {
        bool hasALACTrack;
        ALACConfig alacConfig;
        uint64_t audioDataOffset;
        uint64_t audioDataSize;

        MP4Container() : hasALACTrack(false), audioDataOffset(0), audioDataSize(0) {}
    };

    /**
     * Check if data buffer contains ALAC format signature
     * @param data Data buffer to check
     * @param size Size of data buffer
     * @return true if ALAC format detected, false otherwise
     */
    bool isALACFormat(const uint8_t* data, size_t size) const;

    // MP4 container parsing functions
    bool parseMP4Container(std::ifstream& file, MP4Container& container);
    bool parseMovieAtom(std::ifstream& file, uint64_t atomSize, MP4Container& container);
    bool parseTrackAtom(std::ifstream& file, uint64_t atomSize, MP4Container& container);
    bool parseMediaAtom(std::ifstream& file, uint64_t atomSize, MP4Container& container);
    bool parseMediaInfoAtom(std::ifstream& file, uint64_t atomSize, MP4Container& container);
    bool parseSampleTableAtom(std::ifstream& file, uint64_t atomSize, MP4Container& container);
    bool parseSampleDescriptionAtom(std::ifstream& file, uint64_t atomSize, MP4Container& container);
    bool parseALACConfig(std::ifstream& file, ALACConfig& config);

    /**
     * Decode ALAC compressed data to PCM
     * @param file Input file stream
     * @param container MP4 container information
     * @param alac ALAC decoder handle
     * @param pcmData Output buffer for float samples
     * @param totalSamples Total samples to decode
     * @return true if decoding successful, false otherwise
     */
    bool decodeALACData(std::ifstream& file, const MP4Container& container,
                        void* alac, float* pcmData, uint32_t totalSamples);

    // iTunes metadata extraction
    void extractiTunesMetadata(std::ifstream& file, AudioMetadata& metadata);
    void parseiTunesMetadataList(std::ifstream& file, uint64_t atomSize, AudioMetadata& metadata);
    void readMetadataText(std::ifstream& file, uint64_t size, std::optional<std::string>& field);
    void readMetadataYear(std::ifstream& file, uint64_t size, std::optional<uint16_t>& field);
    void readMetadataTrack(std::ifstream& file, uint64_t size, std::optional<uint16_t>& field);
};

} // namespace vortex::core::fileio