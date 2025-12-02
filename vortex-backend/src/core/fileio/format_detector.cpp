#include "format_detector.hpp"
#include "system/logger.hpp"

#include <juce_audio_formats/juce_audio_formats.h>
#include <algorithm>
#include <fstream>
#include <cstring>

namespace vortex {

FormatDetector::FormatDetector() {
    formatManager_.registerBasicFormats();
    initializeMagicNumbers();
    Logger::info("FormatDetector initialized");
}

FormatDetector::~FormatDetector() {
    Logger::info("FormatDetector destroyed");
}

void FormatDetector::initializeMagicNumbers() {
    // WAV format signatures
    magicNumbers_["WAV"] = {
        {0x52, 0x49, 0x46, 0x46}, // "RIFF"
        {0x57, 0x41, 0x56, 0x45}  // "WAVE"
    };

    // FLAC format signature
    magicNumbers_["FLAC"] = {
        {0x66, 0x4C, 0x61, 0x43}  // "fLaC"
    };

    // MP3 format signatures (ID3v2 tag or sync bytes)
    magicNumbers_["MP3"] = {
        {0x49, 0x44, 0x33},        // "ID3"
        {0xFF, 0xFB},              // MPEG-1 Layer 3
        {0xFF, 0xF3},              // MPEG-2 Layer 3
        {0xFF, 0xF2}               // MPEG-2.5 Layer 3
    };

    // AAC format signatures
    magicNumbers_["AAC"] = {
        {0xFF, 0xF1},              // ADTS header
        {0xFF, 0xF9},              // ADTS header
        {0x4D, 0x34, 0x20},       // M4A/MP4 "M4 "
        {0x66, 0x74, 0x79, 0x70}  // M4A/MP4 "ftyp"
    };

    // OGG Vorbis format signature
    magicNumbers_["OGG"] = {
        {0x4F, 0x67, 0x67, 0x53}  // "OggS"
    };

    // Opus format signature
    magicNumbers_["OPUS"] = {
        {0x4F, 0x67, 0x67, 0x53}  // "OggS" (same container as OGG)
    };

    // DSD format signatures
    magicNumbers_["DSD"] = {
        {0x44, 0x53, 0x44, 0x01}, // "DSD\1" (DSF)
        {0x46, 0x52, 0x4D, 0x38}  // "FRM8" (DFF)
    };
}

AudioFormat FormatDetector::detectFormat(const std::string& filePath) {
    Logger::info("Detecting format for file: {}", filePath);

    try {
        // First try JUCE format detection
        juce::File file(filePath);
        if (file.existsAsFile()) {
            auto* reader = formatManager_.createReaderFor(file);
            if (reader) {
                std::string formatName = reader->getFormatName().toStdString();
                AudioFormat format = convertJuceFormat(formatName);
                Logger::info("JUCE detected format: {} -> {}", formatName, static_cast<int>(format));
                delete reader;
                return format;
            }
        }

        // Fallback to magic number detection
        return detectFormatByMagicNumbers(filePath);

    } catch (const std::exception& e) {
        Logger::error("Format detection failed for {}: {}", filePath, e.what());
        return AudioFormat::PCM;
    }
}

AudioFormat FormatDetector::detectFormatByMagicNumbers(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        Logger::error("Cannot open file for magic number detection: {}", filePath);
        return AudioFormat::PCM;
    }

    // Read first 512 bytes for magic number detection
    std::vector<uint8_t> header(512);
    file.read(reinterpret_cast<char*>(header.data()), header.size());
    size_t bytesRead = static_cast<size_t>(file.gcount());
    header.resize(bytesRead);

    // Check each format signature
    for (const auto& pair : magicNumbers_) {
        const std::string& formatName = pair.first;
        const std::vector<std::vector<uint8_t>>& signatures = pair.second;

        for (const auto& signature : signatures) {
            if (checkSignature(header, signature)) {
                AudioFormat format = convertFormatName(formatName);
                Logger::info("Magic number detection: {} -> {}", formatName, static_cast<int>(format));
                return format;
            }
        }
    }

    // Additional format-specific checks
    if (checkDSDFormat(header)) {
        Logger::info("DSD format detected by custom logic");
        return AudioFormat::DSD64; // Default DSD format
    }

    Logger::warning("Unknown format detected, defaulting to PCM");
    return AudioFormat::PCM;
}

bool FormatDetector::checkSignature(const std::vector<uint8_t>& data, const std::vector<uint8_t>& signature) {
    if (data.size() < signature.size()) {
        return false;
    }

    return std::memcmp(data.data(), signature.data(), signature.size()) == 0;
}

bool FormatDetector::checkDSDFormat(const std::vector<uint8_t>& data) {
    // Check for DSF format
    if (data.size() >= 4 && std::memcmp(data.data(), "DSD\1", 4) == 0) {
        return true;
    }

    // Check for DFF format
    if (data.size() >= 4 && std::memcmp(data.data(), "FRM8", 4) == 0) {
        return true;
    }

    return false;
}

AudioFormat FormatDetector::convertJuceFormat(const std::string& juceFormatName) {
    if (juceFormatName == "WAV file") {
        return AudioFormat::WAV;
    } else if (juceFormatName == "FLAC file") {
        return AudioFormat::FLAC;
    } else if (juceFormatName == "MP3 file") {
        return AudioFormat::MP3;
    } else if (juceFormatName == "AAC file" || juceFormatName == "M4A/MP4 file") {
        return AudioFormat::AAC;
    } else if (juceFormatName == "Ogg Vorbis file") {
        return AudioFormat::OGG;
    } else if (juceFormatName == "Opus file") {
        return AudioFormat::OPUS;
    }

    return AudioFormat::PCM;
}

AudioFormat FormatDetector::convertFormatName(const std::string& formatName) {
    if (formatName == "WAV") {
        return AudioFormat::WAV;
    } else if (formatName == "FLAC") {
        return AudioFormat::FLAC;
    } else if (formatName == "MP3") {
        return AudioFormat::MP3;
    } else if (formatName == "AAC") {
        return AudioFormat::AAC;
    } else if (formatName == "OGG") {
        return AudioFormat::OGG;
    } else if (formatName == "OPUS") {
        return AudioFormat::OPUS;
    } else if (formatName == "DSD") {
        return AudioFormat::DSD64; // Default to DSD64
    }

    return AudioFormat::PCM;
}

AudioMetadata FormatDetector::extractMetadata(const std::string& filePath) {
    Logger::info("Extracting metadata from: {}", filePath);

    AudioMetadata metadata;
    metadata.filePath = filePath;

    try {
        juce::File file(filePath);
        if (!file.existsAsFile()) {
            Logger::error("File does not exist: {}", filePath);
            return metadata;
        }

        auto* reader = formatManager_.createReaderFor(file);
        if (!reader) {
            Logger::error("Cannot create audio reader for: {}", filePath);
            return metadata;
        }

        // Basic audio properties
        metadata.sampleRate = static_cast<int>(reader->sampleRate);
        metadata.channels = reader->numChannels;
        metadata.bitDepth = reader->bitsPerSample;
        metadata.numSamples = static_cast<size_t>(reader->lengthInSamples);
        metadata.durationSeconds = static_cast<double>(metadata.numSamples) / metadata.sampleRate;

        // Extract format
        std::string formatName = reader->getFormatName().toStdString();
        metadata.format = convertJuceFormat(formatName);

        // Extract standard metadata
        metadata.title = reader->getMetadataValue("title").toStdString();
        metadata.artist = reader->getMetadataValue("artist").toStdString();
        metadata.album = reader->getMetadataValue("album").toStdString();
        metadata.albumArtist = reader->getMetadataValue("albumArtist").toStdString();
        metadata.genre = reader->getMetadataValue("genre").toStdString();
        metadata.year = reader->getMetadataValue("year").toStdString();
        metadata.trackNumber = reader->getMetadataValue("trackNumber").toStdString();
        metadata.discNumber = reader->getMetadataValue("discNumber").toStdString();
        metadata.comment = reader->getMetadataValue("comment").toStdString();

        // Extract additional metadata
        metadata.composer = reader->getMetadataValue("composer").toStdString();
        metadata.lyricist = reader->getMetadataValue("lyricist").toStdString();
        metadata.conductor = reader->getMetadataValue("conductor").toStdString();
        metadata.encodedBy = reader->getMetadataValue("encodedBy").toStdString();
        metadata.copyright = reader->getMetadataValue("copyright").toStdString();

        // Extract technical metadata
        std::string bitrateStr = reader->getMetadataValue("bitRate").toStdString();
        if (!bitrateStr.empty()) {
            metadata.bitrate = std::stoi(bitrateStr);
        }

        std::string encodedWithStr = reader->getMetadataValue("encodedWith").toStdString();
        if (!encodedWithStr.empty()) {
            metadata.encodedWith = encodedWithStr;
        }

        // Format-specific metadata extraction
        extractFormatSpecificMetadata(metadata, filePath, formatName);

        // Extract file metadata
        extractFileMetadata(metadata, filePath);

        delete reader; // Clean up

        Logger::info("Metadata extracted successfully for: {} ({}, {}Hz, {}ch, {}bits)",
                     metadata.title.empty() ? "Unknown Title" : metadata.title,
                     getFormatName(metadata.format),
                     metadata.sampleRate,
                     metadata.channels,
                     metadata.bitDepth);

        return metadata;

    } catch (const std::exception& e) {
        Logger::error("Metadata extraction failed for {}: {}", filePath, e.what());
        return metadata;
    }
}

void FormatDetector::extractFormatSpecificMetadata(AudioMetadata& metadata,
                                                   const std::string& filePath,
                                                   const std::string& formatName) {
    try {
        if (formatName == "MP3 file") {
            extractMP3Metadata(metadata, filePath);
        } else if (formatName == "FLAC file") {
            extractFLACMetadata(metadata, filePath);
        } else if (formatName == "Ogg Vorbis file") {
            extractOggMetadata(metadata, filePath);
        } else if (formatName == "WAV file") {
            extractWAVMetadata(metadata, filePath);
        }
    } catch (const std::exception& e) {
        Logger::warning("Format-specific metadata extraction failed: {}", e.what());
    }
}

void FormatDetector::extractMP3Metadata(AudioMetadata& metadata, const std::string& filePath) {
    // MP3-specific metadata extraction would go here
    // For now, we rely on JUCE's ID3 tag extraction
    Logger::debug("Extracting MP3-specific metadata");
}

void FormatDetector::extractFLACMetadata(AudioMetadata& metadata, const std::string& filePath) {
    // FLAC-specific metadata extraction would go here
    Logger::debug("Extracting FLAC-specific metadata");

    // Check for DSD in FLAC (FLAC-DSD)
    std::ifstream file(filePath, std::ios::binary);
    if (file.is_open()) {
        // Look for DSD marker in FLAC stream
        // This is simplified - real implementation would parse FLAC metadata blocks
        file.seekg(0, std::ios::end);
        size_t fileSize = static_cast<size_t>(file.tellg());
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(4096);
        while (file.read(buffer.data(), buffer.size())) {
            size_t bytesRead = static_cast<size_t>(file.gcount());
            // Search for DSD markers (simplified)
            for (size_t i = 0; i < bytesRead - 3; ++i) {
                if (std::memcmp(&buffer[i], "DSD", 3) == 0) {
                    metadata.format = AudioFormat::DSD64; // FLAC-DSD
                    Logger::info("FLAC-DSD format detected");
                    return;
                }
            }
        }
    }
}

void FormatDetector::extractOggMetadata(AudioMetadata& metadata, const std::string& filePath) {
    // Ogg Vorbis/Opus-specific metadata extraction
    Logger::debug("Extracting Ogg-specific metadata");

    // Differentiate between Ogg Vorbis and Opus
    std::ifstream file(filePath, std::ios::binary);
    if (file.is_open()) {
        file.seekg(28); // Skip Ogg header
        char signature[8] = {0};
        file.read(signature, 7);

        if (std::strcmp(signature, "\x01vorbis") == 0) {
            metadata.format = AudioFormat::OGG;
            Logger::info("Ogg Vorbis format confirmed");
        } else if (std::strcmp(signature, "OpusHead") == 0) {
            metadata.format = AudioFormat::OPUS;
            Logger::info("Opus format confirmed");
        }
    }
}

void FormatDetector::extractWAVMetadata(AudioMetadata& metadata, const std::string& filePath) {
    // WAV-specific metadata extraction
    Logger::debug("Extracting WAV-specific metadata");

    std::ifstream file(filePath, std::ios::binary);
    if (file.is_open()) {
        // Check for DSD in WAV (DSD-WAV)
        file.seekg(12); // Skip RIFF header
        char chunkId[5] = {0};
        file.read(chunkId, 4);

        if (std::strcmp(chunkId, "ds64") == 0) {
            metadata.format = AudioFormat::DSD64;
            Logger::info("DSD-WAV format detected");
        }
    }
}

void FormatDetector::extractFileMetadata(AudioMetadata& metadata, const std::string& filePath) {
    try {
        std::ifstream file(filePath, std::ios::binary | std::ios::ate);
        if (file.is_open()) {
            metadata.fileSizeBytes = static_cast<size_t>(file.tellg());

            // Calculate bitrate if not available
            if (metadata.bitrate == 0 && metadata.durationSeconds > 0) {
                metadata.bitrate = static_cast<int>((metadata.fileSizeBytes * 8) / metadata.durationSeconds);
            }

            // Extract file path components
            size_t lastSlash = filePath.find_last_of("/\\");
            if (lastSlash != std::string::npos) {
                metadata.fileName = filePath.substr(lastSlash + 1);
                metadata.directory = filePath.substr(0, lastSlash);
            } else {
                metadata.fileName = filePath;
                metadata.directory = "";
            }

            // Extract file extension
            size_t lastDot = metadata.fileName.find_last_of('.');
            if (lastDot != std::string::npos) {
                metadata.fileExtension = metadata.fileName.substr(lastDot + 1);
                std::transform(metadata.fileExtension.begin(), metadata.fileExtension.end(),
                             metadata.fileExtension.begin(), ::tolower);
            }
        }
    } catch (const std::exception& e) {
        Logger::warning("File metadata extraction failed: {}", e.what());
    }
}

std::vector<AudioMetadata> FormatDetector::extractBatchMetadata(const std::vector<std::string>& filePaths) {
    Logger::info("Extracting metadata for {} files", filePaths.size());

    std::vector<AudioMetadata> metadataList;
    metadataList.reserve(filePaths.size());

    for (const auto& filePath : filePaths) {
        try {
            auto metadata = extractMetadata(filePath);
            metadataList.push_back(std::move(metadata));
        } catch (const std::exception& e) {
            Logger::error("Failed to extract metadata from {}: {}", filePath, e.what());
            // Add empty metadata to maintain index consistency
            AudioMetadata emptyMetadata;
            emptyMetadata.filePath = filePath;
            metadataList.push_back(std::move(emptyMetadata));
        }
    }

    Logger::info("Batch metadata extraction completed for {} files", metadataList.size());
    return metadataList;
}

std::string FormatDetector::getFormatName(AudioFormat format) const {
    switch (format) {
        case AudioFormat::PCM: return "PCM";
        case AudioFormat::FLAC: return "FLAC";
        case AudioFormat::WAV: return "WAV";
        case AudioFormat::MP3: return "MP3";
        case AudioFormat::AAC: return "AAC";
        case AudioFormat::OGG: return "OGG";
        case AudioFormat::OPUS: return "OPUS";
        case AudioFormat::DSD64: return "DSD64";
        case AudioFormat::DSD128: return "DSD128";
        case AudioFormat::DSD256: return "DSD256";
        case AudioFormat::DSD512: return "DSD512";
        case AudioFormat::DSD1024: return "DSD1024";
        default: return "Unknown";
    }
}

bool FormatDetector::isFormatSupported(AudioFormat format) const {
    switch (format) {
        case AudioFormat::PCM:
        case AudioFormat::FLAC:
        case AudioFormat::WAV:
        case AudioFormat::MP3:
        case AudioFormat::AAC:
        case AudioFormat::OGG:
        case AudioFormat::OPUS:
        case AudioFormat::DSD64:
        case AudioFormat::DSD128:
        case AudioFormat::DSD256:
        case AudioFormat::DSD512:
        case AudioFormat::DSD1024:
            return true;
        default:
            return false;
    }
}

std::vector<AudioFormat> FormatDetector::getSupportedFormats() const {
    return {
        AudioFormat::PCM,
        AudioFormat::FLAC,
        AudioFormat::WAV,
        AudioFormat::MP3,
        AudioFormat::AAC,
        AudioFormat::OGG,
        AudioFormat::OPUS,
        AudioFormat::DSD64,
        AudioFormat::DSD128,
        AudioFormat::DSD256,
        AudioFormat::DSD512,
        AudioFormat::DSD1024
    };
}

} // namespace vortex