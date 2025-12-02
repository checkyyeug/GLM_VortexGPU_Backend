#pragma once

#include "audio_types.hpp"
#include "network_types.hpp"

#include <string>
#include <vector>
#include <array>
#include <fstream>
#include <memory>

namespace vortex {

/**
 * @brief Audio format detection using file signatures and extensions
 *
 * This class provides accurate format detection for all supported audio formats
 * including lossless, lossy, and high-resolution DSD formats up to DSD1024.
 */
class FormatDetector {
public:
    FormatDetector();
    ~FormatDetector() = default;

    // Main detection methods
    AudioFormat detectFormat(const std::string& filePath) const;
    AudioFormat detectFromSignature(const std::string& filePath) const;
    AudioFormat detectFromExtension(const std::string& filePath) const;

    // Format validation
    bool isValidFormat(const std::string& filePath, AudioFormat expectedFormat = AudioFormat::UNKNOWN) const;
    bool isSupportedFormat(const std::string& filePath) const;

    // Format information
    std::string getFormatName(AudioFormat format) const;
    std::vector<std::string> getFormatExtensions(AudioFormat format) const;
    std::vector<AudioFormat> getSupportedFormats() const;

    // High-resolution format detection
    bool isHighResolution(const std::string& filePath) const;
    bool isDSDFormat(const std::string& filePath) const;
    uint32_t getDSDBitRate(const std::string& filePath) const;

    // Lossless/lossy detection
    bool isLossless(const std::string& filePath) const;
    bool isLossy(const std::string& filePath) const;

    // Quality assessment
    enum class QualityLevel {
        LOW,      // MP3 < 128kbps, AAC < 96kbps
        MEDIUM,   // MP3 128-192kbps, AAC 96-128kbps
        HIGH,     // MP3 192-320kbps, AAC 128-256kbps
        LOSSLESS, // FLAC, ALAC, WAV (16-bit)
        HI_RES,   // WAV/AIFF (24-bit), high-bitrate lossless
        DSD       // DSD64-DSD1024
    };

    QualityLevel assessQuality(const std::string& filePath) const;
    std::string getQualityDescription(QualityLevel quality) const;

    // Technical specifications
    struct FormatSpecs {
        AudioFormat format;
        std::vector<std::string> extensions;
        std::vector<uint8_t> signature;
        size_t signatureOffset = 0;
        bool isLossless = false;
        bool isHighResolution = false;
        uint32_t maxSampleRate = 0;
        uint16_t maxBitDepth = 0;
        uint32_t maxBitrate = 0;
        std::string mimeType;
        std::string description;
    };

    std::vector<FormatSpecs> getAllFormatSpecs() const;
    FormatSpecs getFormatSpecs(AudioFormat format) const;

    // Error handling
    std::string getLastError() const;
    bool hasErrors() const;

private:
    // File signature analysis
    bool readFileSignature(const std::string& filePath, std::vector<uint8_t>& signature, size_t maxBytes = 32) const;
    bool matchesSignature(const std::vector<uint8_t>& fileData, const std::vector<uint8_t>& pattern, size_t offset = 0) const;

    // Format-specific detection
    AudioFormat detectWAV(const std::vector<uint8_t>& signature) const;
    AudioFormat detectFLAC(const std::vector<uint8_t>& signature) const;
    AudioFormat detectMP3(const std::vector<uint8_t>& signature) const;
    AudioFormat detectAAC(const std::vector<uint8_t>& signature) const;
    AudioFormat detectOGG(const std::vector<uint8_t>& signature) const;
    AudioFormat detectDSD(const std::vector<uint8_t>& signature) const;
    AudioFormat detectALAC(const std::vector<uint8_t>& signature) const;
    AudioFormat detectAIFF(const std::vector<uint8_t>& signature) const;

    // DSD format detection
    bool detectDSF(const std::vector<uint8_t>& signature) const;
    bool detectDFF(const std::vector<uint8_t>& signature) const;
    uint32_t extractDSDBitRate(const std::vector<uint8_t>& signature) const;

    // Container format detection
    bool detectMP4Container(const std::vector<uint8_t>& signature) const;
    bool detectMatroskaContainer(const std::vector<uint8_t>& signature) const;

    // Utility methods
    std::string getFileExtension(const std::string& filePath) const;
    std::string toLower(const std::string& str) const;
    bool fileExists(const std::string& filePath) const;
    size_t getFileSize(const std::string& filePath) const;

    // Format registry
    void initializeFormatRegistry();
    std::map<AudioFormat, FormatSpecs> formatRegistry_;
    std::map<std::string, AudioFormat> extensionMap_;
    std::map<std::vector<uint8_t>, AudioFormat> signatureMap_;

    mutable std::string lastError_;
    bool registryInitialized_ = false;
};

/**
 * @brief Format conversion utilities
 */
class FormatConverter {
public:
    // Conversion capabilities
    static bool canConvert(AudioFormat sourceFormat, AudioFormat targetFormat);
    static std::vector<AudioFormat> getConversionTargets(AudioFormat sourceFormat);

    // Conversion quality levels
    enum class ConversionQuality {
        FAST,         // Prioritize speed over quality
        BALANCED,     // Balance speed and quality
        HIGH,         // Prioritize quality
        LOSSLESS      // Lossless conversion when possible
    };

    // Conversion options
    struct ConversionOptions {
        ConversionQuality quality = ConversionQuality::BALANCED;
        uint32_t targetSampleRate = 0;    // 0 = keep original
        uint16_t targetBitDepth = 0;      // 0 = keep original
        uint16_t targetChannels = 0;      // 0 = keep original
        uint32_t targetBitrate = 0;       // For lossy formats
        bool preserveMetadata = true;
        bool preserveArtwork = true;
        std::string outputPath;
    };

    // Conversion validation
    static bool validateConversionOptions(const ConversionOptions& options);
    static std::vector<std::string> getConversionWarnings(const ConversionOptions& options);
};

/**
 * @brief Audio format validator and quality checker
 */
class AudioFormatValidator {
public:
    // Validation results
    struct ValidationResult {
        bool isValid = true;
        AudioFormat detectedFormat = AudioFormat::UNKNOWN;
        QualityLevel quality = QualityLevel::LOW;
        std::vector<std::string> warnings;
        std::vector<std::string> errors;
        bool isCorrupted = false;
        bool hasEncryption = false;
        bool hasDRM = false;
    };

    // Validation methods
    static ValidationResult validateFile(const std::string& filePath);
    static ValidationResult validateStream(const std::vector<uint8_t>& audioData, AudioFormat format);

    // Quality assessment
    static QualityLevel assessAudioQuality(const std::string& filePath);
    static bool meetsMinimumQuality(const std::string& filePath, QualityLevel minimumLevel);

    // Corruption detection
    static bool detectCorruption(const std::string& filePath);
    static std::vector<size_t> findCorruptedSections(const std::string& filePath, size_t chunkSize = 1024);

    // Metadata validation
    static bool validateMetadata(const std::string& filePath);
    static std::vector<std::string> getMetadataWarnings(const std::string& filePath);

private:
    // Internal validation helpers
    static bool validateWAVFile(const std::string& filePath);
    static bool validateFLACFile(const std::string& filePath);
    static bool validateMP3File(const std::string& filePath);
    static bool validateDSDFile(const std::string& filePath);

    static bool checkAudioConsistency(const std::vector<float>& audioData);
    static bool detectTruncation(const std::string& filePath);
};

} // namespace vortex