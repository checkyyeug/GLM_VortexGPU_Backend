#include "m4a_decoder.hpp"
#include "aac_decoder.hpp"
#include "alac_decoder.hpp"
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace vortex::core::fileio {

M4ADecoder::M4ADecoder() : initialized_(false) {}

M4ADecoder::~M4ADecoder() {
    shutdown();
}

bool M4ADecoder::initialize() {
    if (initialized_) {
        return true;
    }

    // Initialize sub-decoders
    aacDecoder_ = std::make_unique<AACDecoder>();
    alacDecoder_ = std::make_unique<ALACDecoder>();

    if (!aacDecoder_->initialize()) {
        Logger::error("Failed to initialize AAC decoder for M4A support");
        return false;
    }

    if (!alacDecoder_->initialize()) {
        Logger::error("Failed to initialize ALAC decoder for M4A support");
        return false;
    }

    initialized_ = true;
    Logger::info("M4A decoder initialized successfully");
    return true;
}

void M4ADecoder::shutdown() {
    if (!initialized_) {
        return;
    }

    if (aacDecoder_) {
        aacDecoder_->shutdown();
        aacDecoder_.reset();
    }

    if (alacDecoder_) {
        alacDecoder_->shutdown();
        alacDecoder_.reset();
    }

    initialized_ = false;
    Logger::info("M4A decoder shutdown");
}

bool M4ADecoder::canDecode(const std::string& filePath) const {
    if (!initialized_) {
        return false;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Read MP4 container header
    std::vector<uint8_t> header(1024);
    file.read(reinterpret_cast<char*>(header.data()), header.size());
    size_t bytesRead = file.gcount();

    file.close();

    if (bytesRead < 12) {
        return false;
    }

    return isM4AFormat(header.data(), bytesRead);
}

std::optional<AudioData> M4ADecoder::decode(const std::string& filePath) {
    if (!initialized_) {
        Logger::error("M4A decoder not initialized");
        return std::nullopt;
    }

    Logger::info("Decoding M4A file: {}", filePath);

    // Determine the audio codec in the M4A container
    M4AContainerInfo containerInfo;
    if (!parseM4AContainer(filePath, containerInfo)) {
        Logger::error("Failed to parse M4A container: {}", filePath);
        return std::nullopt;
    }

    // Delegate to appropriate decoder based on codec
    switch (containerInfo.audioCodec) {
        case AudioCodec::AAC:
            if (aacDecoder_->canDecode(filePath)) {
                return aacDecoder_->decode(filePath);
            } else {
                Logger::error("AAC decoder cannot handle this M4A file: {}", filePath);
                return std::nullopt;
            }

        case AudioCodec::ALAC:
            if (alacDecoder_->canDecode(filePath)) {
                return alacDecoder_->decode(filePath);
            } else {
                Logger::error("ALAC decoder cannot handle this M4A file: {}", filePath);
                return std::nullopt;
            }

        default:
            Logger::error("Unsupported audio codec in M4A file: {}", filePath);
            return std::nullopt;
    }
}

bool M4ADecoder::extractMetadata(const std::string& filePath, AudioMetadata& metadata) {
    if (!initialized_) {
        return false;
    }

    // Initialize metadata structure
    metadata.format = AudioFormat::M4A;
    metadata.codec = "M4A";

    // Parse M4A container to get container info
    M4AContainerInfo containerInfo;
    if (!parseM4AContainer(filePath, containerInfo)) {
        return false;
    }

    // Extract container-level metadata
    extractM4AMetadata(filePath, metadata);

    // Delegate metadata extraction to appropriate decoder
    switch (containerInfo.audioCodec) {
        case AudioCodec::AAC:
            if (!aacDecoder_->extractMetadata(filePath, metadata)) {
                Logger::warn("Failed to extract AAC metadata from M4A file: {}", filePath);
            }
            metadata.codec = "AAC";
            break;

        case AudioCodec::ALAC:
            if (!alacDecoder_->extractMetadata(filePath, metadata)) {
                Logger::warn("Failed to extract ALAC metadata from M4A file: {}", filePath);
            }
            metadata.codec = "ALAC";
            break;

        default:
            metadata.codec = "Unknown";
            break;
    }

    return true;
}

bool M4ADecoder::isM4AFormat(const uint8_t* data, size_t size) const {
    if (size < 12) {
        return false;
    }

    // Check for MP4 container signature
    if (data[4] == 'f' && data[5] == 't' && data[6] == 'y' && data[7] == 'p') {
        // This is an MP4 file, check for audio-related atoms
        // Look for 'moov' and audio track atoms
        for (size_t i = 0; i < size - 4; ++i) {
            if (data[i] == 'm' && data[i+1] == 'o' && data[i+2] == 'o' && data[i+3] == 'v') {
                return true;  // Found movie atom
            }
            if (data[i] == 'm' && data[i+1] == 'd' && data[i+2] == 'i' && data[i+3] == 'a') {
                return true;  // Found media atom
            }
            if (data[i] == 'm' && data[i+1] == 'p' && data[i+2] == '4' && data[i+3] == 'a') {
                return true;  // Found M4A specific atom
            }
        }
    }

    return false;
}

bool M4ADecoder::parseM4AContainer(const std::string& filePath, M4AContainerInfo& containerInfo) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    containerInfo.audioCodec = AudioCodec::UNKNOWN;
    containerInfo.hasAudioTrack = false;

    // Parse MP4 atoms to find audio codec
    while (file.tellg() < static_cast<std::streamoff>(fileSize)) {
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

        switch (atomType) {
            case 0x6d6f6f76:  // 'moov'
                parseMovieAtom(file, atomSize64 - 8, containerInfo);
                break;

            default:
                // Skip unknown atoms
                break;
        }

        // Move to next atom
        file.seekg(atomStart + static_cast<std::streamoff>(atomSize64 - 8));
    }

    return containerInfo.hasAudioTrack;
}

bool M4ADecoder::parseMovieAtom(std::ifstream& file, uint64_t atomSize, M4AContainerInfo& containerInfo) {
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
                parseTrackAtom(file, subAtomSize - 8, containerInfo);
                break;

            default:
                // Skip other sub-atoms
                break;
        }

        file.seekg(subAtomStart + static_cast<std::streamoff>(subAtomSize - 8));
    }

    return true;
}

bool M4ADecoder::parseTrackAtom(std::ifstream& file, uint64_t atomSize, M4AContainerInfo& containerInfo) {
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
                parseMediaAtom(file, subAtomSize - 8, containerInfo);
                break;

            default:
                break;
        }

        file.seekg(subAtomStart + static_cast<std::streamoff>(subAtomSize - 8));
    }

    return true;
}

bool M4ADecoder::parseMediaAtom(std::ifstream& file, uint64_t atomSize, M4AContainerInfo& containerInfo) {
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
            case 0x68646c72:  // 'hdlr'
                parseHandlerAtom(file, subAtomSize - 8, containerInfo);
                break;

            case 0x6d696e66:  // 'minf'
                parseMediaInfoAtom(file, subAtomSize - 8, containerInfo);
                break;

            default:
                break;
        }

        file.seekg(subAtomStart + static_cast<std::streamoff>(subAtomSize - 8));
    }

    return true;
}

bool M4ADecoder::parseHandlerAtom(std::ifstream& file, uint64_t atomSize, M4AContainerInfo& containerInfo) {
    if (atomSize < 24) {
        return false;
    }

    // Skip version, flags, pre_defined
    file.seekg(8, std::ios::cur);

    // Read handler type
    uint32_t handlerType;
    file.read(reinterpret_cast<char*>(&handlerType), 4);
    handlerType = ntohl(handlerType);

    // Check if this is an audio handler
    if (handlerType == 0x736f756e) {  // 'soun'
        containerInfo.hasAudioTrack = true;
    }

    return true;
}

bool M4ADecoder::parseMediaInfoAtom(std::ifstream& file, uint64_t atomSize, M4AContainerInfo& containerInfo) {
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
                parseSampleTableAtom(file, subAtomSize - 8, containerInfo);
                break;

            default:
                break;
        }

        file.seekg(subAtomStart + static_cast<std::streamoff>(subAtomSize - 8));
    }

    return true;
}

bool M4ADecoder::parseSampleTableAtom(std::ifstream& file, uint64_t atomSize, M4AContainerInfo& containerInfo) {
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
                parseSampleDescriptionAtom(file, subAtomSize - 8, containerInfo);
                break;

            default:
                break;
        }

        file.seekg(subAtomStart + static_cast<std::streamoff>(subAtomSize - 8));
    }

    return true;
}

bool M4ADecoder::parseSampleDescriptionAtom(std::ifstream& file, uint64_t atomSize, M4AContainerInfo& containerInfo) {
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

        // Check audio codec format
        if (dataFormat == 0x6d703461) {      // 'mp4a' - AAC
            containerInfo.audioCodec = AudioCodec::AAC;
            Logger::debug("Found AAC audio codec in M4A container");
        } else if (dataFormat == 0x616c6163) { // 'alac' - ALAC
            containerInfo.audioCodec = AudioCodec::ALAC;
            Logger::debug("Found ALAC audio codec in M4A container");
        }

        // Skip rest of entry
        file.seekg(entrySize - 8, std::ios::cur);
    }

    return true;
}

void M4ADecoder::extractM4AMetadata(const std::string& filePath, AudioMetadata& metadata) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return;
    }

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Look for iTunes metadata atoms
    while (file.tellg() < static_cast<std::streamoff>(fileSize)) {
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

void M4ADecoder::parseiTunesMetadataList(std::ifstream& file, uint64_t atomSize, AudioMetadata& metadata) {
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

void M4ADecoder::readMetadataText(std::ifstream& file, uint64_t size, std::optional<std::string>& field) {
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

void M4ADecoder::readMetadataYear(std::ifstream& file, uint64_t size, std::optional<uint16_t>& field) {
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

void M4ADecoder::readMetadataTrack(std::ifstream& file, uint64_t size, std::optional<uint16_t>& field) {
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