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
 * DSD128 audio decoder (Direct Stream Digital at 5.6448 MHz)
 * High-resolution DSD format supporting both DSDIFF (.dff) and DSF (.dsf) containers
 * Uses multi-stage filtering for high-quality 128:1 downsampling to PCM
 * Extracts DSD-specific metadata including artist, title, genre, and technical properties
 * Optimized for ultra-high-resolution audio processing with GPU acceleration support
 */
class DSD128Decoder {
public:
    DSD128Decoder();
    ~DSD128Decoder();

    /**
     * Initialize the DSD128 decoder
     * @return true if initialization successful, false otherwise
     */
    bool initialize();

    /**
     * Shutdown the DSD128 decoder and clean up resources
     */
    void shutdown();

    /**
     * Check if the file can be decoded as DSD128
     * @param filePath Path to the audio file
     * @return true if file is supported DSD128 format, false otherwise
     */
    bool canDecode(const std::string& filePath) const;

    /**
     * Decode DSD128 file to PCM audio data
     * @param filePath Path to the DSD128 file
     * @return AudioData if decoding successful, nullopt otherwise
     */
    std::optional<AudioData> decode(const std::string& filePath);

    /**
     * Extract metadata from DSD128 file
     * @param filePath Path to the DSD128 file
     * @param metadata Metadata structure to fill
     * @return true if metadata extraction successful, false otherwise
     */
    bool extractMetadata(const std::string& filePath, AudioMetadata& metadata);

private:
    bool initialized_;

    // DSD file information structure
    struct DSDFileInfo {
        double sampleRate;
        uint16_t channels;
        uint8_t bitDepth;
        uint64_t totalFrames;
        uint64_t dataOffset;
        uint64_t dataSize;

        DSDFileInfo() : sampleRate(0), channels(2), bitDepth(1),
                      totalFrames(0), dataOffset(0), dataSize(0) {}
    };

    /**
     * Check if data buffer contains DSD128 format signature
     * @param data Data buffer to check
     * @param size Size of data buffer
     * @return true if DSD128 format detected, false otherwise
     */
    bool isDSD128Format(const uint8_t* data, size_t size) const;

    /**
     * Parse DSD file (either DSDIFF or DSF) and extract format information
     * @param file Input file stream
     * @param fileInfo DSD file information structure to fill
     * @return true if parsing successful, false otherwise
     */
    bool parseDSDFile(std::ifstream& file, DSDFileInfo& fileInfo);

    /**
     * Parse DSDIFF (.dff) file format
     * @param file Input file stream
     * @param fileInfo DSD file information structure to fill
     * @return true if parsing successful, false otherwise
     */
    bool parseDSDIFFFile(std::ifstream& file, DSDFileInfo& fileInfo);

    /**
     * Parse DSF (.dsf) file format
     * @param file Input file stream
     * @param fileInfo DSD file information structure to fill
     * @return true if parsing successful, false otherwise
     */
    bool parseDSFFile(std::ifstream& file, DSDFileInfo& fileInfo);

    /**
     * Decode DSD 1-bit data to PCM with advanced multi-stage filtering
     * @param file Input file stream positioned at audio data
     * @param fileInfo DSD file information
     * @param pcmData Output buffer for PCM samples
     * @param totalSamples Total samples to decode
     * @param downsampleRatio Downsampling ratio (typically 128:1 for DSD128)
     * @return true if decoding successful, false otherwise
     */
    bool decodeDSDToPCM(std::ifstream& file, const DSDFileInfo& fileInfo,
                        float* pcmData, uint64_t totalSamples, int downsampleRatio);

    /**
     * Extract DSD128-specific metadata
     * @param file Input file stream
     * @param metadata Metadata structure to fill
     */
    void extractDSDMetadata(std::ifstream& file, AudioMetadata& metadata);

    /**
     * Extract metadata from DSDIFF format
     * @param file Input file stream
     * @param metadata Metadata structure to fill
     */
    void extractDSDIFFMetadata(std::ifstream& file, AudioMetadata& metadata);

    /**
     * Extract metadata from DSF format
     * @param file Input file stream
     * @param metadata Metadata structure to fill
     */
    void extractDSFMetadata(std::ifstream& file, AudioMetadata& metadata);

    // Helper function for big-endian conversion
    template<typename T>
    T ntoh(T value) const {
        if constexpr (sizeof(T) == 2) {
            return static_cast<T>(((value & 0xFF) << 8) | ((value >> 8) & 0xFF));
        } else if constexpr (sizeof(T) == 4) {
            return static_cast<T>(((value & 0xFF) << 24) |
                                  ((value & 0xFF00) << 8) |
                                  ((value >> 8) & 0xFF00) |
                                  ((value >> 24) & 0xFF));
        } else if constexpr (sizeof(T) == 8) {
            return static_cast<T>(((value & 0xFFULL) << 56) |
                                  ((value & 0xFF00ULL) << 40) |
                                  ((value & 0xFF0000ULL) << 24) |
                                  ((value & 0xFF000000ULL) << 8) |
                                  ((value >> 8) & 0xFF000000ULL) |
                                  ((value >> 24) & 0xFF0000ULL) |
                                  ((value >> 40) & 0xFF00ULL) |
                                  ((value >> 56) & 0xFFULL));
        }
        return value;
    }

    size_t fileSize;  // For metadata extraction
};

} // namespace vortex::core::fileio