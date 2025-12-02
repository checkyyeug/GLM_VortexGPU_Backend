#include "aac_decoder.hpp"
#include <cstring>
#include <algorithm>
#include <stdexcept>

#ifdef VORTEX_ENABLE_AAC
#include <aacdecoder.h>
#include <neaacdec.h>
#endif

namespace vortex::core::fileio {

AACDecoder::AACDecoder() : initialized_(false), decoderHandle_(nullptr) {}

AACDecoder::~AACDecoder() {
    shutdown();
}

bool AACDecoder::initialize() {
    if (initialized_) {
        return true;
    }

#ifndef VORTEX_ENABLE_AAC
    Logger::error("AAC support not enabled in build");
    return false;
#else

    // Initialize FAAD2 AAC decoder
    decoderHandle_ = NeAACDecOpen();
    if (!decoderHandle_) {
        Logger::error("Failed to initialize AAC decoder");
        return false;
    }

    // Configure decoder
    NeAACDecConfigurationPtr config = NeAACDecGetCurrentConfiguration(decoderHandle_);
    if (!config) {
        Logger::error("Failed to get AAC decoder configuration");
        NeAACDecClose(decoderHandle_);
        decoderHandle_ = nullptr;
        return false;
    }

    // Configure for float output
    config->outputFormat = FAAD_FMT_FLOAT;
    config->defSampleRate = 44100;
    config->defObjectType = LC;

    unsigned long cap = NeAACDecSetConfiguration(decoderHandle_, config);
    if (cap == 0) {
        Logger::error("Failed to set AAC decoder configuration");
        NeAACDecClose(decoderHandle_);
        decoderHandle_ = nullptr;
        return false;
    }

    initialized_ = true;
    Logger::info("AAC decoder initialized successfully");
    return true;

#endif // VORTEX_ENABLE_AAC
}

void AACDecoder::shutdown() {
    if (!initialized_) {
        return;
    }

    if (decoderHandle_) {
        NeAACDecClose(decoderHandle_);
        decoderHandle_ = nullptr;
    }

    initialized_ = false;
    Logger::info("AAC decoder shutdown");
}

bool AACDecoder::canDecode(const std::string& filePath) const {
    if (!initialized_) {
        return false;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Read potential AAC signature
    std::vector<uint8_t> header(1024);
    file.read(reinterpret_cast<char*>(header.data()), header.size());
    size_t bytesRead = file.gcount();

    file.close();

    if (bytesRead < 2) {
        return false;
    }

    return isAACFormat(header.data(), bytesRead);
}

std::optional<AudioData> AACDecoder::decode(const std::string& filePath) {
    if (!initialized_) {
        Logger::error("AAC decoder not initialized");
        return std::nullopt;
    }

#ifndef VORTEX_ENABLE_AAC
    Logger::error("AAC support not enabled in build");
    return std::nullopt;
#else

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        Logger::error("Cannot open AAC file: {}", filePath);
        return std::nullopt;
    }

    try {
        Logger::info("Decoding AAC file: {}", filePath);

        // Read entire file
        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<uint8_t> aacData(fileSize);
        file.read(reinterpret_cast<char*>(aacData.data()), fileSize);

        if (file.gcount() != static_cast<std::streamsize>(fileSize)) {
            Logger::error("Failed to read AAC file data");
            return std::nullopt;
        }

        // Initialize decoder with first frame
        unsigned long sampleRate = 0;
        unsigned char channels = 0;
        int result = NeAACDecInit(decoderHandle_, aacData.data(), fileSize,
                                 &sampleRate, &channels);

        if (result < 0) {
            Logger::error("Failed to initialize AAC decoder: {}", result);
            return std::nullopt;
        }

        // Create audio data structure
        AudioData audioData;
        audioData.sampleRate = static_cast<double>(sampleRate);
        audioData.channels = static_cast<uint16_t>(channels);
        audioData.bitDepth = 32;  // Float output
        audioData.format = AudioFormat::AAC;

        // Calculate estimated total samples and allocate buffer
        // This is approximate since AAC is variable bitrate
        uint64_t estimatedFrames = fileSize / 1024;  // Rough estimate
        uint64_t estimatedSamples = estimatedFrames * 1024;  // 1024 samples per frame typical
        size_t bufferSize = estimatedSamples * channels * sizeof(float);
        audioData.data.resize(bufferSize);
        float* pcmData = reinterpret_cast<float*>(audioData.data.data());

        // Decode AAC frames
        size_t decodedSamples = 0;
        size_t inputOffset = 0;

        while (inputOffset < fileSize) {
            void* inputBuffer = aacData.data() + inputOffset;
            size_t inputSize = fileSize - inputOffset;

            NeAACDecFrameInfo frameInfo;
            void* outputBuffer = NeAACDecDecode(decoderHandle_, &frameInfo,
                                              static_cast<unsigned char*>(inputBuffer),
                                              static_cast<unsigned long>(inputSize));

            if (frameInfo.error != 0) {
                Logger::warn("AAC decoding error in frame: {}", frameInfo.error);
                break;
            }

            if (frameInfo.samples > 0) {
                // Check if we need to resize output buffer
                size_t requiredSize = (decodedSamples + frameInfo.samples) * sizeof(float);
                if (requiredSize > audioData.data.size()) {
                    audioData.data.resize(requiredSize * 2);  // Double buffer size
                    pcmData = reinterpret_cast<float*>(audioData.data.data()) + decodedSamples;
                }

                // Copy decoded samples
                float* decodedFloats = static_cast<float*>(outputBuffer);
                std::copy(decodedFloats, decodedFloats + frameInfo.samples, pcmData);
                decodedSamples += frameInfo.samples;
                pcmData += frameInfo.samples;
            }

            // Advance input position
            if (frameInfo.bytesconsumed > 0) {
                inputOffset += frameInfo.bytesconsumed;
            } else {
                // If no bytes consumed, advance by 1 to avoid infinite loop
                inputOffset++;
            }

            // Break if we've decoded enough frames
            if (inputOffset >= fileSize) {
                break;
            }
        }

        // Resize to actual decoded size
        audioData.data.resize(decodedSamples * sizeof(float));

        Logger::info("AAC decoded successfully: {} samples, {} channels, {:.2f} seconds",
                    decodedSamples / audioData.channels, audioData.channels,
                    static_cast<double>(decodedSamples / audioData.channels) / audioData.sampleRate);

        return audioData;

    } catch (const std::exception& e) {
        Logger::error("Exception during AAC decoding: {}", e.what());
        return std::nullopt;
    }

#endif // VORTEX_ENABLE_AAC
}

bool AACDecoder::extractMetadata(const std::string& filePath, AudioMetadata& metadata) {
    if (!initialized_) {
        return false;
    }

#ifndef VORTEX_ENABLE_AAC
    Logger::error("AAC support not enabled in build");
    return false;
#else

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    try {
        // Initialize metadata structure
        metadata.format = AudioFormat::AAC;
        metadata.codec = "AAC";

        // Read file
        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<uint8_t> aacData(fileSize);
        file.read(reinterpret_cast<char*>(aacData.data()), fileSize);

        // Initialize decoder to get format info
        unsigned long sampleRate = 0;
        unsigned char channels = 0;
        int result = NeAACDecInit(decoderHandle_, aacData.data(), fileSize,
                                 &sampleRate, &channels);

        if (result >= 0) {
            // Extract technical metadata
            metadata.sampleRate = static_cast<uint32_t>(sampleRate);
            metadata.channels = static_cast<uint16_t>(channels);
            metadata.bitDepth = 32;  // Float output

            // Estimate duration (this is approximate for AAC)
            uint64_t estimatedFrames = fileSize / 1024;
            uint64_t estimatedSamples = estimatedFrames * 1024;
            metadata.duration = std::chrono::duration<double>(
                static_cast<double>(estimatedSamples / channels) / sampleRate);

            // Estimate bitrate (AAC is variable bitrate)
            metadata.bitrate = static_cast<uint32_t>(
                (fileSize * 8.0) / metadata.duration.count() / 1000.0);
        }

        // Look for ID3 tags (AAC files can have ID3v2 tags)
        extractID3Metadata(aacData.data(), fileSize, metadata);

        return true;

    } catch (const std::exception& e) {
        Logger::error("Exception during AAC metadata extraction: {}", e.what());
        return false;
    }

#endif // VORTEX_ENABLE_AAC
}

bool AACDecoder::isAACFormat(const uint8_t* data, size_t size) const {
    if (size < 2) {
        return false;
    }

    // Check for AAC sync word (12 bits: 0xFFF)
    uint16_t sync = (data[0] << 8) | data[1];
    if ((sync & 0xFFF0) == 0xFFF0) {
        // Validate AAC header structure
        if (size >= 3) {
            // Check for valid MPEG version and layer
            uint8_t version = (data[1] >> 3) & 0x03;
            uint8_t layer = (data[1] >> 1) & 0x03;

            // MPEG-4 or MPEG-2 with layer 0 indicates AAC
            if ((version == 0 && layer == 0) || (version == 2 && layer == 0)) {
                return true;
            }
        }
    }

    // Check for ADTS header (most common AAC format)
    if (size >= 7) {
        // ADTS sync word (12 bits = 0xFFF)
        uint16_t adtsSync = (data[0] << 4) | ((data[1] >> 4) & 0x0F);
        if (adtsSync == 0xFFF) {
            // Check ADTS layer (must be 0)
            uint8_t layer = (data[1] >> 1) & 0x03;
            if (layer == 0) {
                // Check MPEG version (0 = MPEG-4, 1 = MPEG-2)
                uint8_t mpegVersion = (data[1] >> 3) & 0x01;
                // Check protection absent
                uint8_t protectionAbsent = data[1] & 0x01;

                // If these checks pass, it's likely ADTS AAC
                return true;
            }
        }
    }

    return false;
}

void AACDecoder::extractID3Metadata(const uint8_t* data, size_t size, AudioMetadata& metadata) {
    // Check for ID3v2 tag
    if (size >= 10 && data[0] == 'I' && data[1] == 'D' && data[2] == '3') {
        uint8_t version = data[3];
        uint8_t flags = data[5];

        // Calculate tag size (synchsafe integer)
        uint32_t tagSize = (data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9];

        if (tagSize > 0 && tagSize < size - 10) {
            const uint8_t* tagData = data + 10;
            parseID3v2Frames(tagData, tagSize, metadata);
        }
    }
}

void AACDecoder::parseID3v2Frames(const uint8_t* data, size_t size, AudioMetadata& metadata) {
    size_t offset = 0;

    while (offset + 10 <= size) {
        // Read frame header
        char frameId[5] = {0};
        std::memcpy(frameId, data + offset, 4);

        uint32_t frameSize = (data[offset + 4] << 24) | (data[offset + 5] << 16) |
                            (data[offset + 6] << 8) | data[offset + 7];

        uint16_t flags = (data[offset + 8] << 8) | data[offset + 9];

        offset += 10;

        if (frameSize == 0 || offset + frameSize > size) {
            break;  // Invalid frame
        }

        // Process common frame IDs
        std::string frameStr(frameId);
        std::string frameValue;

        // Skip encoding byte (usually present in text frames)
        size_t dataOffset = offset;
        if (frameSize > 0 && (data[dataOffset] == 0 || data[dataOffset] == 1)) {
            dataOffset++;
        }

        if (frameSize > 1) {
            frameValue.assign(reinterpret_cast<const char*>(data + dataOffset), frameSize - 1);
        }

        // Map frame IDs to metadata fields
        if (frameStr == "TIT2" && !frameValue.empty()) {
            metadata.title = frameValue;
        } else if (frameStr == "TPE1" && !frameValue.empty()) {
            metadata.artist = frameValue;
        } else if (frameStr == "TALB" && !frameValue.empty()) {
            metadata.album = frameValue;
        } else if (frameStr == "TDRC" && !frameValue.empty()) {
            // Extract year from date frame
            if (frameValue.size() >= 4) {
                try {
                    metadata.year = static_cast<uint16_t>(std::stoi(frameValue.substr(0, 4)));
                } catch (...) {
                    // Invalid year format
                }
            }
        } else if (frameStr == "TCON" && !frameValue.empty()) {
            metadata.genre = frameValue;
        } else if (frameStr == "TRCK" && !frameValue.empty()) {
            try {
                metadata.track = static_cast<uint16_t>(std::stoi(frameValue));
            } catch (...) {
                // Invalid track number
            }
        }

        offset += frameSize;
    }
}

} // namespace vortex::core::fileio