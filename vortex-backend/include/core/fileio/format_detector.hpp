#pragma once

#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace vortex {

enum class AudioFormat {
    PCM = 0,
    FLAC = 1,
    WAV = 2,
    MP3 = 3,
    AAC = 4,
    OGG = 5,
    OPUS = 6,
    DSD64 = 7,
    DSD128 = 8,
    DSD256 = 9,
    DSD512 = 10,
    DSD1024 = 11
};

struct AudioMetadata {
    // Basic information
    std::string filePath;
    std::string fileName;
    std::string directory;
    std::string fileExtension;
    size_t fileSizeBytes = 0;

    // Audio properties
    AudioFormat format = AudioFormat::PCM;
    int sampleRate = 44100;
    int channels = 2;
    int bitDepth = 16;
    size_t numSamples = 0;
    double durationSeconds = 0.0;
    int bitrate = 0;

    // Standard metadata
    std::string title;
    std::string artist;
    std::string album;
    std::string albumArtist;
    std::string genre;
    std::string year;
    std::string trackNumber;
    std::string discNumber;
    std::string comment;

    // Extended metadata
    std::string composer;
    std::string lyricist;
    std::string conductor;
    std::string encodedBy;
    std::string encodedWith;
    std::string copyright;

    // Additional metadata for different formats
    std::map<std::string, std::string> customMetadata;
};

/**
 * @brief Audio format detection and metadata extraction system
 *
 * This class provides comprehensive audio format detection using magic numbers,
 * JUCE audio format support, and custom format parsers. It extracts detailed
 * metadata from various audio formats including lossless (FLAC, WAV), lossy
 * (MP3, AAC, OGG, Opus), and high-resolution DSD formats.
 */
class FormatDetector {
public:
    FormatDetector();
    ~FormatDetector();

    // Format detection
    AudioFormat detectFormat(const std::string& filePath);
    AudioFormat detectFormatByMagicNumbers(const std::string& filePath);

    // Metadata extraction
    AudioMetadata extractMetadata(const std::string& filePath);
    std::vector<AudioMetadata> extractBatchMetadata(const std::vector<std::string>& filePaths);

    // Format utilities
    std::string getFormatName(AudioFormat format) const;
    bool isFormatSupported(AudioFormat format) const;
    std::vector<AudioFormat> getSupportedFormats() const;

private:
    // JUCE format manager
    juce::AudioFormatManager formatManager_;

    // Magic number database
    std::map<std::string, std::vector<std::vector<uint8_t>>> magicNumbers_;

    // Initialization
    void initializeMagicNumbers();

    // Format detection helpers
    bool checkSignature(const std::vector<uint8_t>& data, const std::vector<uint8_t>& signature);
    bool checkDSDFormat(const std::vector<uint8_t>& data);
    AudioFormat convertJuceFormat(const std::string& juceFormatName);
    AudioFormat convertFormatName(const std::string& formatName);

    // Metadata extraction helpers
    void extractFormatSpecificMetadata(AudioMetadata& metadata,
                                      const std::string& filePath,
                                      const std::string& formatName);
    void extractMP3Metadata(AudioMetadata& metadata, const std::string& filePath);
    void extractFLACMetadata(AudioMetadata& metadata, const std::string& filePath);
    void extractOggMetadata(AudioMetadata& metadata, const std::string& filePath);
    void extractWAVMetadata(AudioMetadata& metadata, const std::string& filePath);
    void extractFileMetadata(AudioMetadata& metadata, const std::string& filePath);
};

} // namespace vortex