#include "alac_decoder.hpp"
#include <cstring>
#include <algorithm>
#include <stdexcept>

#ifdef VORTEX_ENABLE_ALAC
#include <alac.h>
#endif

namespace vortex::core::fileio {

ALACDecoder::ALACDecoder() : initialized_(false) {}

ALACDecoder::~ALACDecoder() {
    shutdown();
}

bool ALACDecoder::initialize() {
    if (initialized_) {
        return true;
    }

#ifndef VORTEX_ENABLE_ALAC
    Logger::error("ALAC support not enabled in build");
    return false;
#else
    initialized_ = true;
    Logger::info("ALAC decoder initialized successfully");
    return true;
#endif
}

void ALACDecoder::shutdown() {
    if (!initialized_) {
        return;
    }

    initialized_ = false;
    Logger::info("ALAC decoder shutdown");
}

bool ALACDecoder::canDecode(const std::string& filePath) const {
    if (!initialized_) {
        return false;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Read potential MP4 container header
    std::vector<uint8_t> header(1024);
    file.read(reinterpret_cast<char*>(header.data()), header.size());
    size_t bytesRead = file.gcount();

    file.close();

    if (bytesRead < 12) {
        return false;
    }

    return isALACFormat(header.data(), bytesRead);
}

std::optional<AudioData> ALACDecoder::decode(const std::string& filePath) {
    if (!initialized_) {
        Logger::error("ALAC decoder not initialized");
        return std::nullopt;
    }

#ifndef VORTEX_ENABLE_ALAC
    Logger::error("ALAC support not enabled in build");
    return std::nullopt;
#else

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        Logger::error("Cannot open ALAC file: {}", filePath);
        return std::nullopt;
    }

    try {
        Logger::info("Decoding ALAC file: {}", filePath);

        // Parse MP4 container to find ALAC audio track
        MP4Container container;
        if (!parseMP4Container(file, container)) {
            Logger::error("Failed to parse MP4 container for ALAC: {}", filePath);
            return std::nullopt;
        }

        if (!container.hasALACTrack) {
            Logger::error("No ALAC track found in file: {}", filePath);
            return std::nullopt;
        }

        // Initialize ALAC decoder
        alac_decoder_t* alac = alac_create(container.alacConfig);
        if (!alac) {
            Logger::error("Failed to create ALAC decoder");
            return std::nullopt;
        }

        // Create audio data structure
        AudioData audioData;
        audioData.sampleRate = static_cast<double>(container.alacConfig.sampleRate);
        audioData.channels = static_cast<uint16_t>(container.alacConfig.channels);
        audioData.bitDepth = static_cast<uint16_t>(container.alacConfig.bitDepth);
        audioData.format = AudioFormat::ALAC;

        // Calculate total samples and allocate buffer
        uint32_t totalSamples = (container.audioDataSize * 8) /
                               (container.alacConfig.bitDepth * container.alacConfig.channels);
        size_t pcmSize = totalSamples * sizeof(float);
        audioData.data.resize(pcmSize);
        float* pcmData = reinterpret_cast<float*>(audioData.data.data());

        // Decode ALAC data
        if (!decodeALACData(file, container, alac, pcmData, totalSamples)) {
            Logger::error("Failed to decode ALAC audio data");
            alac_destroy(alac);
            return std::nullopt;
        }

        alac_destroy(alac);

        Logger::info("ALAC decoded successfully: {} samples, {} channels, {:.2f} seconds",
                    totalSamples / audioData.channels, audioData.channels,
                    static_cast<double>(totalSamples / audioData.channels) / audioData.sampleRate);

        return audioData;

    } catch (const std::exception& e) {
        Logger::error("Exception during ALAC decoding: {}", e.what());
        return std::nullopt;
    }

#endif // VORTEX_ENABLE_ALAC
}

bool ALACDecoder::extractMetadata(const std::string& filePath, AudioMetadata& metadata) {
    if (!initialized_) {
        return false;
    }

#ifndef VORTEX_ENABLE_ALAC
    Logger::error("ALAC support not enabled in build");
    return false;
#else

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    try {
        // Initialize metadata structure
        metadata.format = AudioFormat::ALAC;
        metadata.codec = "ALAC";

        // Parse MP4 container
        MP4Container container;
        if (!parseMP4Container(file, container)) {
            return false;
        }

        if (!container.hasALACTrack) {
            return false;
        }

        // Extract technical metadata
        metadata.sampleRate = static_cast<uint32_t>(container.alacConfig.sampleRate);
        metadata.channels = static_cast<uint16_t>(container.alacConfig.channels);
        metadata.bitDepth = static_cast<uint16_t>(container.alacConfig.bitDepth);

        // Calculate duration
        uint32_t totalSamples = (container.audioDataSize * 8) /
                               (container.alacConfig.bitDepth * container.alacConfig.channels);
        metadata.duration = std::chrono::duration<double>(
            static_cast<double>(totalSamples / container.alacConfig.channels) /
            container.alacConfig.sampleRate);

        // Calculate bitrate (ALAC is lossless, so bitrate varies)
        metadata.bitrate = static_cast<uint32_t>(
            container.alacConfig.sampleRate *
            container.alacConfig.channels *
            container.alacConfig.bitDepth);

        // Extract iTunes metadata
        extractiTunesMetadata(file, metadata);

        return true;

    } catch (const std::exception& e) {
        Logger::error("Exception during ALAC metadata extraction: {}", e.what());
        return false;
    }

#endif // VORTEX_ENABLE_ALAC
}

bool ALACDecoder::isALACFormat(const uint8_t* data, size_t size) const {
    if (size < 12) {
        return false;
    }

    // Check for MP4 container signature
    if (data[4] == 'f' && data[5] == 't' && data[6] == 'y' && data[7] == 'p') {
        // Look for ALAC specific atom in the header
        for (size_t i = 0; i < size - 4; ++i) {
            if (data[i] == 'a' && data[i+1] == 'l' && data[i+2] == 'a' && data[i+3] == 'c') {
                return true;
            }
        }
    }

    return false;
}

bool ALACDecoder::parseMP4Container(std::ifstream& file, MP4Container& container) {
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    container.hasALACTrack = false;
    container.audioDataSize = 0;
    container.audioDataOffset = 0;

    while (file.tellg() < static_cast<std::streamsize>(fileSize)) {
        uint32_t atomSize, atomType;
        file.read(reinterpret_cast<char*>(&atomSize), 4);
        file.read(reinterpret_cast<char*>(&atomType), 4);

        if (file.eof()) {
            break;
        }

        // Convert from big-endian
        atomSize = ntohl(atomSize);
        atomType = ntohl(atomType);

        uint64_t atomSize64 = atomSize;
        if (atomSize == 1) {
            // 64-bit size
            file.read(reinterpret_cast<char*>(&atomSize64), 8);
            atomSize64 = be64toh(atomSize64);
        }

        if (atomSize64 == 0) {
            // Atom extends to end of file
            atomSize64 = fileSize - file.tellg() + 8;
        }

        // Save current position for atom processing
        auto atomStart = file.tellg();

        // Process different atom types
        switch (atomType) {
            case 0x6d6f6f76:  // 'moov'
                parseMovieAtom(file, atomSize64 - 8, container);
                break;

            case 0x6d646174:  // 'mdat'
                if (!container.hasALACTrack) {
                    container.audioDataOffset = file.tellg();
                    container.audioDataSize = atomSize64 - 8;
                }
                break;

            default:
                // Skip unknown atoms
                break;
        }

        // Move to next atom
        file.seekg(atomStart + static_cast<std::streamoff>(atomSize64 - 8));
    }

    return container.hasALACTrack;
}

bool ALACDecoder::parseMovieAtom(std::ifstream& file, uint64_t atomSize, MP4Container& container) {
    auto atomEnd = file.tellg() + static_cast<std::streamoff>(atomSize);

    while (file.tellg() < atomEnd) {
        uint32_t subAtomSize, subAtomType;
        file.read(reinterpret_cast<char*>(&subAtomSize), 4);
        file.read(reinterpret_cast<char*>(&subAtomType), 4);

        if (file.eof()) {
            break;
        }

        subAtomSize = ntohl(subAtomSize);
        subAtomType = ntohl(subAtomType);

        auto subAtomStart = file.tellg();

        switch (subAtomType) {
            case 0x7472616b:  // 'trak'
                parseTrackAtom(file, subAtomSize - 8, container);
                break;

            default:
                // Skip other sub-atoms
                break;
        }

        file.seekg(subAtomStart + static_cast<std::streamoff>(subAtomSize - 8));
    }

    return true;
}

bool ALACDecoder::parseTrackAtom(std::ifstream& file, uint64_t atomSize, MP4Container& container) {
    auto atomEnd = file.tellg() + static_cast<std::streamoff>(atomSize);

    while (file.tellg() < atomEnd) {
        uint32_t subAtomSize, subAtomType;
        file.read(reinterpret_cast<char*>(&subAtomSize), 4);
        file.read(reinterpret_cast<char*>(&subAtomType), 4);

        if (file.eof()) {
            break;
        }

        subAtomSize = ntohl(subAtomSize);
        subAtomType = ntohl(subAtomType);

        auto subAtomStart = file.tellg();

        switch (subAtomType) {
            case 0x6d646961:  // 'mdia'
                parseMediaAtom(file, subAtomSize - 8, container);
                break;

            default:
                break;
        }

        file.seekg(subAtomStart + static_cast<std::streamoff>(subAtomSize - 8));
    }

    return true;
}

bool ALACDecoder::parseMediaAtom(std::ifstream& file, uint64_t atomSize, MP4Container& container) {
    auto atomEnd = file.tellg() + static_cast<std::streamoff>(atomSize);

    while (file.tellg() < atomEnd) {
        uint32_t subAtomSize, subAtomType;
        file.read(reinterpret_cast<char*>(&subAtomSize), 4);
        file.read(reinterpret_cast<char*>(&subAtomType), 4);

        if (file.eof()) {
            break;
        }

        subAtomSize = ntohl(subAtomSize);
        subAtomType = ntohl(subAtomType);

        auto subAtomStart = file.tellg();

        switch (subAtomType) {
            case 0x6d696e66:  // 'minf'
                parseMediaInfoAtom(file, subAtomSize - 8, container);
                break;

            default:
                break;
        }

        file.seekg(subAtomStart + static_cast<std::streamoff>(subAtomSize - 8));
    }

    return true;
}

bool ALACDecoder::parseMediaInfoAtom(std::ifstream& file, uint64_t atomSize, MP4Container& container) {
    auto atomEnd = file.tellg() + static_cast<std::streamoff>(atomSize);

    while (file.tellg() < atomEnd) {
        uint32_t subAtomSize, subAtomType;
        file.read(reinterpret_cast<char*>(&subAtomSize), 4);
        file.read(reinterpret_cast<char*>(&subAtomType), 4);

        if (file.eof()) {
            break;
        }

        subAtomSize = ntohl(subAtomSize);
        subAtomType = ntohl(subAtomType);

        auto subAtomStart = file.tellg();

        switch (subAtomType) {
            case 0x7374626c:  // 'stbl'
                parseSampleTableAtom(file, subAtomSize - 8, container);
                break;

            default:
                break;
        }

        file.seekg(subAtomStart + static_cast<std::streamoff>(subAtomSize - 8));
    }

    return true;
}

bool ALACDecoder::parseSampleTableAtom(std::ifstream& file, uint64_t atomSize, MP4Container& container) {
    auto atomEnd = file.tellg() + static_cast<std::streamoff>(atomSize);

    while (file.tellg() < atomEnd) {
        uint32_t subAtomSize, subAtomType;
        file.read(reinterpret_cast<char*>(&subAtomSize), 4);
        file.read(reinterpret_cast<char*>(&subAtomType), 4);

        if (file.eof()) {
            break;
        }

        subAtomSize = ntohl(subAtomSize);
        subAtomType = ntohl(subAtomType);

        auto subAtomStart = file.tellg();

        switch (subAtomType) {
            case 0x73747364:  // 'stsd'
                parseSampleDescriptionAtom(file, subAtomSize - 8, container);
                break;

            default:
                break;
        }

        file.seekg(subAtomStart + static_cast<std::streamoff>(subAtomSize - 8));
    }

    return true;
}

bool ALACDecoder::parseSampleDescriptionAtom(std::ifstream& file, uint64_t atomSize, MP4Container& container) {
    uint32_t version, flags, entryCount;
    file.read(reinterpret_cast<char*>(&version), 4);
    file.read(reinterpret_cast<char*>(&flags), 4);
    file.read(reinterpret_cast<char*>(&entryCount), 4);

    entryCount = ntohl(entryCount);

    for (uint32_t i = 0; i < entryCount; ++i) {
        uint32_t entrySize, dataFormat;
        file.read(reinterpret_cast<char*>(&entrySize), 4);
        file.read(reinterpret_cast<char*>(&dataFormat), 4);

        entrySize = ntohl(entrySize);
        dataFormat = ntohl(dataFormat);

        if (dataFormat == 0x616c6163) {  // 'alac'
            // Found ALAC sample description
            container.hasALACTrack = true;

            // Skip reserved fields
            file.seekg(6, std::ios::cur);

            // Read ALAC specific configuration
            uint16_t dataReferenceIndex;
            file.read(reinterpret_cast<char*>(&dataReferenceIndex), 2);

            // Skip to ALAC specific configuration
            file.seekg(16, std::ios::cur);

            // Parse ALAC configuration
            parseALACConfig(file, container.alacConfig);
            break;
        } else {
            // Skip non-ALAC entries
            file.seekg(entrySize - 8, std::ios::cur);
        }
    }

    return container.hasALACTrack;
}

bool ALACDecoder::parseALACConfig(std::ifstream& file, ALACConfig& config) {
    // ALAC magic cookie format
    uint32_t formatFlags;
    file.read(reinterpret_cast<char*>(&formatFlags), 4);
    formatFlags = ntohl(formatFlags);

    config.sampleRate = (formatFlags >> 16) & 0xFFFF;
    config.bitDepth = (formatFlags >> 8) & 0xFF;

    file.seekg(4, std::ios::cur);  // Skip reserved

    uint32_t framesPerPacket;
    file.read(reinterpret_cast<char*>(&framesPerPacket), 4);
    framesPerPacket = ntohl(framesPerPacket);

    uint32_t channels;
    file.read(reinterpret_cast<char*>(&channels), 4);
    channels = ntohl(channels);

    config.channels = static_cast<uint16_t>(channels);

    Logger::debug("ALAC Config: {} Hz, {} channels, {} bits",
                 config.sampleRate, config.channels, config.bitDepth);

    return true;
}

bool ALACDecoder::decodeALACData(std::ifstream& file, const MP4Container& container,
                                  alac_decoder_t* alac, float* pcmData, uint32_t totalSamples) {
    // Seek to audio data
    file.seekg(container.audioDataOffset);

    // Read compressed ALAC data
    std::vector<uint8_t> compressedData(container.audioDataSize);
    file.read(reinterpret_cast<char*>(compressedData.data()), container.audioDataSize);

    if (file.gcount() != static_cast<std::streamsize>(container.audioDataSize)) {
        Logger::error("Failed to read compressed ALAC data");
        return false;
    }

    // Decode ALAC data
    uint32_t decodedSamples = 0;
    uint8_t* inputPtr = compressedData.data();
    float* outputPtr = pcmData;
    size_t inputSize = compressedData.size();

    while (inputSize > 0 && decodedSamples < totalSamples) {
        uint32_t inputUsed = 0;
        uint32_t outputSamples = 0;

        if (!alac_decode(alac, inputPtr, inputSize, &inputUsed,
                        outputPtr, totalSamples - decodedSamples, &outputSamples)) {
            Logger::error("ALAC decode failed");
            return false;
        }

        inputPtr += inputUsed;
        inputSize -= inputUsed;
        outputPtr += outputSamples * container.alacConfig.channels;
        decodedSamples += outputSamples;
    }

    return decodedSamples > 0;
}

void ALACDecoder::extractiTunesMetadata(std::ifstream& file, AudioMetadata& metadata) {
    // This is a simplified implementation
    // In practice, you would parse the iTunes/ilst atom for complete metadata
    // For now, we'll look for common metadata atoms in the file

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    while (file.tellg() < static_cast<std::streamsize>(fileSize)) {
        uint32_t atomSize, atomType;
        file.read(reinterpret_cast<char*>(&atomSize), 4);
        file.read(reinterpret_cast<char*>(&atomType), 4);

        if (file.eof()) {
            break;
        }

        atomSize = ntohl(atomSize);
        atomType = ntohl(atomType);

        // Look for iTunes metadata atoms
        if (atomType == 0x696c7374) {  // 'ilst' - iTunes metadata list
            parseiTunesMetadataList(file, atomSize - 8, metadata);
            break;
        } else if (atomSize == 0) {
            break;
        } else {
            file.seekg(atomSize - 8, std::ios::cur);
        }
    }
}

void ALACDecoder::parseiTunesMetadataList(std::ifstream& file, uint64_t atomSize, AudioMetadata& metadata) {
    auto atomEnd = file.tellg() + static_cast<std::streamoff>(atomSize);

    while (file.tellg() < atomEnd) {
        uint32_t itemSize, itemType;
        file.read(reinterpret_cast<char*>(&itemSize), 4);
        file.read(reinterpret_cast<char*>(&itemType), 4);

        if (file.eof()) {
            break;
        }

        itemSize = ntohl(itemSize);
        itemType = ntohl(itemType);

        if (itemSize < 8) {
            break;
        }

        auto itemStart = file.tellg();
        uint64_t remainingSize = itemSize - 8;

        // Parse different metadata item types
        switch (itemType) {
            case 0xa9746974:  // '©tit' - Title
            case 0x7469746c:  // 'titl' - Title
                readMetadataText(file, remainingSize, metadata.title);
                break;

            case 0xa9617274:  // '©art' - Artist
            case 0x61727473:  // 'arts' - Artist
                readMetadataText(file, remainingSize, metadata.artist);
                break;

            case 0xa9616c62:  // '©alb' - Album
            case 0x616c6275:  // 'albu' - Album
                readMetadataText(file, remainingSize, metadata.album);
                break;

            case 0xa9646179:  // '©day' - Year
                readMetadataYear(file, remainingSize, metadata.year);
                break;

            case 0xa967656e:  // '©gen' - Genre
            case 0x67656e72:  // 'genr' - Genre
                readMetadataText(file, remainingSize, metadata.genre);
                break;

            case 0x74726b6e:  // 'trkn' - Track number
                readMetadataTrack(file, remainingSize, metadata.track);
                break;

            default:
                // Skip unknown metadata items
                break;
        }

        file.seekg(itemStart + static_cast<std::streamoff>(remainingSize));
    }
}

void ALACDecoder::readMetadataText(std::ifstream& file, uint64_t size, std::optional<std::string>& field) {
    if (size < 8) return;

    // Skip header (usually 8 bytes for text metadata)
    file.seekg(8, std::ios::cur);
    size -= 8;

    // Read text data
    std::vector<char> textData(size);
    file.read(textData.data(), size);

    if (file.gcount() > 0) {
        field = std::string(textData.data(), file.gcount());
    }
}

void ALACDecoder::readMetadataYear(std::ifstream& file, uint64_t size, std::optional<uint16_t>& field) {
    if (size < 8) return;

    // Skip header
    file.seekg(8, std::ios::cur);
    size -= 8;

    // Read year text
    std::vector<char> yearText(size);
    file.read(yearText.data(), size);

    if (file.gcount() >= 4) {
        std::string yearStr(yearText.data(), 4);
        try {
            field = static_cast<uint16_t>(std::stoi(yearStr));
        } catch (...) {
            // Invalid year format
        }
    }
}

void ALACDecoder::readMetadataTrack(std::ifstream& file, uint64_t size, std::optional<uint16_t>& field) {
    if (size < 8) return;

    // Skip header
    file.seekg(8, std::ios::cur);
    size -= 8;

    // Read track number (usually stored as 2 pairs of 16-bit integers)
    if (size >= 4) {
        uint16_t trackNumber, totalTracks;
        file.read(reinterpret_cast<char*>(&trackNumber), 2);
        file.read(reinterpret_cast<char*>(&totalTracks), 2);

        trackNumber = ntohs(trackNumber);
        if (trackNumber > 0) {
            field = trackNumber;
        }
    }
}

} // namespace vortex::core::fileio