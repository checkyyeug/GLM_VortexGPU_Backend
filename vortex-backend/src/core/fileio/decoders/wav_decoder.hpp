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
 * WAV audio file decoder
 * Supports PCM 8-bit, 16-bit, 24-bit, 32-bit integer and 32-bit float formats
 * Extracts RIFF INFO metadata
 */
class WAVDecoder {
public:
    WAVDecoder();
    ~WAVDecoder();

    /**
     * Initialize the WAV decoder
     * @return true if initialization successful, false otherwise
     */
    bool initialize();

    /**
     * Shutdown the WAV decoder and clean up resources
     */
    void shutdown();

    /**
     * Check if the file can be decoded as WAV
     * @param filePath Path to the audio file
     * @return true if file is supported WAV format, false otherwise
     */
    bool canDecode(const std::string& filePath) const;

    /**
     * Decode WAV file to PCM audio data
     * @param filePath Path to the WAV file
     * @return AudioData if decoding successful, nullopt otherwise
     */
    std::optional<AudioData> decode(const std::string& filePath);

    /**
     * Extract metadata from WAV file
     * @param filePath Path to the WAV file
     * @param metadata Metadata structure to fill
     * @return true if metadata extraction successful, false otherwise
     */
    bool extractMetadata(const std::string& filePath, AudioMetadata& metadata);

private:
    bool initialized_;

    // WAV header structures
    struct ChunkHeader {
        char id[4];
        uint32_t size;
    };

    struct WAVHeader {
        char riffId[4];
        uint32_t fileSize;
        char waveId[4];
        uint16_t audioFormat;
        uint16_t numChannels;
        uint32_t sampleRate;
        uint32_t byteRate;
        uint16_t blockAlign;
        uint16_t bitsPerSample;
        uint32_t dataSize;
        uint64_t dataOffset;
    };

    /**
     * Parse WAV header from file
     * @param file Input file stream
     * @param header WAV header structure to fill
     * @return true if parsing successful, false otherwise
     */
    bool parseWAVHeader(std::ifstream& file, WAVHeader& header);

    /**
     * Decode WAV audio data to float samples
     * @param file Input file stream positioned at data
     * @param header Parsed WAV header
     * @param pcmData Output buffer for float samples
     * @param totalSamples Number of samples to decode
     * @return true if decoding successful, false otherwise
     */
    bool decodeWAVData(std::ifstream& file, const WAVHeader& header,
                       float* pcmData, uint32_t totalSamples);

    // Bit depth specific decoding functions
    bool decodeWAV8Bit(std::ifstream& file, float* pcmData, uint32_t totalSamples);
    bool decodeWAV16Bit(std::ifstream& file, float* pcmData, uint32_t totalSamples);
    bool decodeWAV24Bit(std::ifstream& file, float* pcmData, uint32_t totalSamples);
    bool decodeWAV32BitInt(std::ifstream& file, float* pcmData, uint32_t totalSamples);
    bool decodeWAV32BitFloat(std::ifstream& file, float* pcmData, uint32_t totalSamples);

    /**
     * Extract INFO metadata from RIFF LIST chunk
     * @param file Input file stream
     * @param metadata Metadata structure to fill
     */
    void extractInfoMetadata(std::ifstream& file, AudioMetadata& metadata);
};

} // namespace vortex::core::fileio