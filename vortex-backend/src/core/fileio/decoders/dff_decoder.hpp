#pragma once

#include <string>
#include <vector>
#include <optional>
#include <fstream>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include "core/fileio/audio_file_loader.hpp"
#include "system/logger.hpp"

namespace vortex::core::fileio {

// Forward declarations for DSD decoders
class DSD64Decoder;
class DSD128Decoder;
class DSD256Decoder;
class DSD512Decoder;
class DSD1024Decoder;

/**
 * DFF (DSDIFF) audio decoder container
 * Handles DSDIFF (.dff) container format for DSD audio files
 * Supports DSD64 through DSD1024 formats with automatic sample rate detection
 * Delegates to appropriate DSD decoder based on format detection
 * Extracts DSDIFF metadata including artist, title, genre, and technical properties
 * Optimized for high-resolution audio processing with GPU acceleration support
 */
class DFFDecoder {
public:
    DFFDecoder();
    ~DFFDecoder();

    /**
     * Initialize the DFF decoder
     * @return true if initialization successful, false otherwise
     */
    bool initialize();

    /**
     * Shutdown the DFF decoder and clean up resources
     */
    void shutdown();

    /**
     * Check if the file can be decoded as DFF
     * @param filePath Path to the audio file
     * @return true if file is supported DFF format, false otherwise
     */
    bool canDecode(const std::string& filePath) const;

    /**
     * Decode DFF file to PCM audio data
     * @param filePath Path to the DFF file
     * @return AudioData if decoding successful, nullopt otherwise
     */
    std::optional<AudioData> decode(const std::string& filePath);

    /**
     * Extract metadata from DFF file
     * @param filePath Path to the DFF file
     * @param metadata Metadata structure to fill
     * @return true if metadata extraction successful, false otherwise
     */
    bool extractMetadata(const std::string& filePath, AudioMetadata& metadata);

private:
    bool initialized_;

    // DSD file information structure
    struct DSDIFFHeader {
        char signature[4];      // "FRM8"
        uint64_t fileSize;      // Total file size
        char format[4];         // "DSD "

        // Format chunk information
        uint32_t sampleRate;
        uint16_t channels;
        uint8_t bitDepth;

        // Data chunk information
        uint64_t totalSamples;
        uint64_t dataOffset;
        uint64_t dataSize;

        // Compression type
        uint8_t compressionType;

        DSDIFFHeader() : fileSize(0), sampleRate(0), channels(2), bitDepth(1),
                        totalSamples(0), dataOffset(0), dataSize(0), compressionType(0) {
            std::memset(signature, 0, 4);
            std::memset(format, 0, 4);
        }
    };

    /**
     * Check if data buffer contains DFF format signature
     * @param data Data buffer to check
     * @param size Size of data buffer
     * @return true if DFF format detected, false otherwise
     */
    bool isDFFFormat(const uint8_t* data, size_t size) const;

    /**
     * Parse DSDIFF file header and extract format information
     * @param file Input file stream
     * @param header DSDIFF header structure to fill
     * @return true if parsing successful, false otherwise
     */
    bool parseDSDIFFHeader(std::ifstream& file, DSDIFFHeader& header);

    /**
     * Create a temporary DSD file from DFF data for delegation
     * @param dsdData Raw DSD data from DFF file
     * @param header DSDIFF header information
     * @return Path to temporary file
     */
    std::string createTempDSDFile(const std::vector<uint8_t>& dsdData, const DSDIFFHeader& header);

    /**
     * Extract DSDFF-specific metadata chunks
     * @param file Input file stream
     * @param metadata Metadata structure to fill
     */
    void extractDSDIFFMetadata(std::ifstream& file, AudioMetadata& metadata);

    /**
     * Extract ID3v2 metadata from DSDIFF file
     * @param data ID3v2 data buffer
     * @param size Size of ID3v2 data
     * @param metadata Metadata structure to fill
     */
    void extractID3v2Metadata(const uint8_t* data, size_t size, AudioMetadata& metadata);

    /**
     * Get DSD format name based on sample rate
     * @param sampleRate Sample rate in Hz
     * @return String representation of DSD format
     */
    std::string getDSDFormatName(double sampleRate) const;

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
};

} // namespace vortex::core::fileio