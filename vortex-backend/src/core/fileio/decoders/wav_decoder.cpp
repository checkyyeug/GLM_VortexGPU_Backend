#include "wav_decoder.hpp"
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace vortex::core::fileio {

WAVDecoder::WAVDecoder() : initialized_(false) {}

WAVDecoder::~WAVDecoder() {
    shutdown();
}

bool WAVDecoder::initialize() {
    if (initialized_) {
        return true;
    }

    // WAV decoder doesn't require external library initialization
    initialized_ = true;
    Logger::info("WAV decoder initialized successfully");
    return true;
}

void WAVDecoder::shutdown() {
    if (!initialized_) {
        return;
    }

    initialized_ = false;
    Logger::info("WAV decoder shutdown");
}

bool WAVDecoder::canDecode(const std::string& filePath) const {
    if (!initialized_) {
        return false;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Read RIFF header
    uint8_t header[12];
    file.read(reinterpret_cast<char*>(header), 12);
    size_t bytesRead = file.gcount();

    file.close();

    if (bytesRead < 12) {
        return false;
    }

    // Check RIFF and WAVE signatures
    return (header[0] == 'R' && header[1] == 'I' && header[2] == 'F' && header[3] == 'F' &&
            header[8] == 'W' && header[9] == 'A' && header[10] == 'V' && header[11] == 'E');
}

std::optional<AudioData> WAVDecoder::decode(const std::string& filePath) {
    if (!initialized_) {
        Logger::error("WAV decoder not initialized");
        return std::nullopt;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        Logger::error("Cannot open WAV file: {}", filePath);
        return std::nullopt;
    }

    try {
        // Read and parse WAV header
        WAVHeader header;
        if (!parseWAVHeader(file, header)) {
            Logger::error("Failed to parse WAV header: {}", filePath);
            return std::nullopt;
        }

        // Validate format
        if (header.audioFormat != 1) {  // Only support PCM
            Logger::error("Unsupported WAV format: {} (only PCM supported)", header.audioFormat);
            return std::nullopt;
        }

        // Create audio data structure
        AudioData audioData;
        audioData.sampleRate = static_cast<double>(header.sampleRate);
        audioData.channels = static_cast<uint16_t>(header.numChannels);
        audioData.bitDepth = static_cast<uint16_t>(header.bitsPerSample);
        audioData.format = AudioFormat::WAV;

        // Calculate total samples
        uint32_t bytesPerSample = (header.bitsPerSample / 8);
        uint32_t totalSamples = header.dataSize / bytesPerSample;
        uint32_t totalFrames = totalSamples / header.numChannels;

        // Allocate buffer for decoded data
        size_t pcmSize = totalSamples * sizeof(float);
        audioData.data.resize(pcmSize);
        float* pcmData = reinterpret_cast<float*>(audioData.data.data());

        // Read audio data
        if (!decodeWAVData(file, header, pcmData, totalSamples)) {
            Logger::error("Failed to decode WAV data: {}", filePath);
            return std::nullopt;
        }

        Logger::info("WAV decoded successfully: {} samples, {} frames, {:.2f} seconds",
                    totalSamples, totalFrames, static_cast<double>(totalFrames) / header.sampleRate);

        return audioData;

    } catch (const std::exception& e) {
        Logger::error("Exception during WAV decoding: {}", e.what());
        return std::nullopt;
    }
}

bool WAVDecoder::extractMetadata(const std::string& filePath, AudioMetadata& metadata) {
    if (!initialized_) {
        return false;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Initialize metadata structure
    metadata.format = AudioFormat::WAV;
    metadata.codec = "PCM";

    try {
        // Parse WAV header
        WAVHeader header;
        if (!parseWAVHeader(file, header)) {
            return false;
        }

        // Extract technical metadata
        metadata.sampleRate = static_cast<uint32_t>(header.sampleRate);
        metadata.channels = static_cast<uint16_t>(header.numChannels);
        metadata.bitDepth = static_cast<uint16_t>(header.bitsPerSample);

        // Calculate duration
        uint32_t bytesPerSample = (header.bitsPerSample / 8);
        uint32_t totalSamples = header.dataSize / bytesPerSample;
        uint32_t totalFrames = totalSamples / header.numChannels;

        metadata.duration = std::chrono::duration<double>(
            static_cast<double>(totalFrames) / header.sampleRate);

        // Calculate bitrate
        metadata.bitrate = metadata.sampleRate * metadata.channels * metadata.bitDepth;

        // Try to extract INFO metadata chunks if present
        extractInfoMetadata(file, metadata);

        return true;

    } catch (const std::exception& e) {
        Logger::error("Exception during WAV metadata extraction: {}", e.what());
        return false;
    }
}

bool WAVDecoder::parseWAVHeader(std::ifstream& file, WAVHeader& header) {
    // Read RIFF header
    file.read(reinterpret_cast<char*>(&header.riffId), 4);
    file.read(reinterpret_cast<char*>(&header.fileSize), 4);
    file.read(reinterpret_cast<char*>(&header.waveId), 4);

    // Validate RIFF signature
    if (std::strncmp(header.riffId, "RIFF", 4) != 0 ||
        std::strncmp(header.waveId, "WAVE", 4) != 0) {
        Logger::error("Invalid RIFF/WAVE signature");
        return false;
    }

    // Parse chunks
    while (file.good()) {
        ChunkHeader chunkHeader;
        file.read(reinterpret_cast<char*>(&chunkHeader.id), 4);
        file.read(reinterpret_cast<char*>(&chunkHeader.size), 4);

        if (file.eof()) {
            break;
        }

        if (std::strncmp(chunkHeader.id, "fmt ", 4) == 0) {
            // Format chunk
            file.read(reinterpret_cast<char*>(&header.audioFormat), 2);
            file.read(reinterpret_cast<char*>(&header.numChannels), 2);
            file.read(reinterpret_cast<char*>(&header.sampleRate), 4);
            file.read(reinterpret_cast<char*>(&header.byteRate), 4);
            file.read(reinterpret_cast<char*>(&header.blockAlign), 2);
            file.read(reinterpret_cast<char*>(&header.bitsPerSample), 2);

            // Skip any extra format bytes
            if (chunkHeader.size > 16) {
                file.seekg(chunkHeader.size - 16, std::ios::cur);
            }
        } else if (std::strncmp(chunkHeader.id, "data", 4) == 0) {
            // Data chunk
            header.dataSize = chunkHeader.size;
            header.dataOffset = file.tellg();

            // Don't read the data here, just remember position
            file.seekg(chunkHeader.size, std::ios::cur);
        } else {
            // Skip other chunks (LIST, INFO, etc.)
            file.seekg(chunkHeader.size, std::ios::cur);
        }

        // Ensure we're on even byte boundary (WAV chunks are word-aligned)
        if (chunkHeader.size % 2 != 0) {
            file.seekg(1, std::ios::cur);
        }
    }

    // Validate required fields
    if (header.audioFormat == 0 || header.numChannels == 0 ||
        header.sampleRate == 0 || header.bitsPerSample == 0) {
        Logger::error("Invalid WAV format parameters");
        return false;
    }

    // Return to data position for reading
    file.seekg(header.dataOffset);
    return true;
}

bool WAVDecoder::decodeWAVData(std::ifstream& file, const WAVHeader& header,
                               float* pcmData, uint32_t totalSamples) {
    // Seek to data position
    file.seekg(header.dataOffset);

    switch (header.bitsPerSample) {
        case 8:
            return decodeWAV8Bit(file, pcmData, totalSamples);
        case 16:
            return decodeWAV16Bit(file, pcmData, totalSamples);
        case 24:
            return decodeWAV24Bit(file, pcmData, totalSamples);
        case 32:
            if (header.audioFormat == 3) {  // IEEE float
                return decodeWAV32BitFloat(file, pcmData, totalSamples);
            } else {
                return decodeWAV32BitInt(file, pcmData, totalSamples);
            }
        default:
            Logger::error("Unsupported WAV bit depth: {}", header.bitsPerSample);
            return false;
    }
}

bool WAVDecoder::decodeWAV8Bit(std::ifstream& file, float* pcmData, uint32_t totalSamples) {
    std::vector<uint8_t> buffer(totalSamples);
    file.read(reinterpret_cast<char*>(buffer.data()), totalSamples);

    if (file.gcount() != totalSamples) {
        Logger::error("Failed to read 8-bit WAV data");
        return false;
    }

    // Convert 8-bit unsigned to float (-1.0 to 1.0)
    for (uint32_t i = 0; i < totalSamples; ++i) {
        pcmData[i] = (static_cast<float>(buffer[i]) - 128.0f) / 128.0f;
    }

    return true;
}

bool WAVDecoder::decodeWAV16Bit(std::ifstream& file, float* pcmData, uint32_t totalSamples) {
    std::vector<int16_t> buffer(totalSamples);
    file.read(reinterpret_cast<char*>(buffer.data()), totalSamples * sizeof(int16_t));

    if (file.gcount() != static_cast<std::streamsize>(totalSamples * sizeof(int16_t))) {
        Logger::error("Failed to read 16-bit WAV data");
        return false;
    }

    // Convert 16-bit signed to float (-1.0 to 1.0)
    for (uint32_t i = 0; i < totalSamples; ++i) {
        pcmData[i] = static_cast<float>(buffer[i]) / 32768.0f;
    }

    return true;
}

bool WAVDecoder::decodeWAV24Bit(std::ifstream& file, float* pcmData, uint32_t totalSamples) {
    std::vector<uint8_t> buffer(totalSamples * 3);
    file.read(reinterpret_cast<char*>(buffer.data()), totalSamples * 3);

    if (file.gcount() != static_cast<std::streamsize>(totalSamples * 3)) {
        Logger::error("Failed to read 24-bit WAV data");
        return false;
    }

    // Convert 24-bit signed to float (-1.0 to 1.0)
    for (uint32_t i = 0; i < totalSamples; ++i) {
        int32_t sample = (static_cast<int32_t>(buffer[i * 3]) << 8) |
                        (static_cast<int32_t>(buffer[i * 3 + 1]) << 16) |
                        (static_cast<int32_t>(buffer[i * 3 + 2]) << 24);

        // Sign extend
        sample >>= 8;

        pcmData[i] = static_cast<float>(sample) / 8388608.0f;  // 2^23
    }

    return true;
}

bool WAVDecoder::decodeWAV32BitInt(std::ifstream& file, float* pcmData, uint32_t totalSamples) {
    std::vector<int32_t> buffer(totalSamples);
    file.read(reinterpret_cast<char*>(buffer.data()), totalSamples * sizeof(int32_t));

    if (file.gcount() != static_cast<std::streamsize>(totalSamples * sizeof(int32_t))) {
        Logger::error("Failed to read 32-bit integer WAV data");
        return false;
    }

    // Convert 32-bit signed to float (-1.0 to 1.0)
    for (uint32_t i = 0; i < totalSamples; ++i) {
        pcmData[i] = static_cast<float>(buffer[i]) / 2147483648.0f;  // 2^31
    }

    return true;
}

bool WAVDecoder::decodeWAV32BitFloat(std::ifstream& file, float* pcmData, uint32_t totalSamples) {
    file.read(reinterpret_cast<char*>(pcmData), totalSamples * sizeof(float));

    if (file.gcount() != static_cast<std::streamsize>(totalSamples * sizeof(float))) {
        Logger::error("Failed to read 32-bit float WAV data");
        return false;
    }

    return true;
}

void WAVDecoder::extractInfoMetadata(std::ifstream& file, AudioMetadata& metadata) {
    // Save current position
    auto currentPos = file.tellg();

    // Search for LIST INFO chunk
    file.seekg(12);  // Skip RIFF header

    while (file.good() && !file.eof()) {
        ChunkHeader chunkHeader;
        file.read(reinterpret_cast<char*>(&chunkHeader.id), 4);
        file.read(reinterpret_cast<char*>(&chunkHeader.size), 4);

        if (file.eof()) {
            break;
        }

        if (std::strncmp(chunkHeader.id, "LIST", 4) == 0) {
            // Read list type
            char listType[5] = {0};
            file.read(listType, 4);

            if (std::strncmp(listType, "INFO", 4) == 0) {
                // Parse INFO sub-chunks
                size_t remainingSize = chunkHeader.size - 4;
                while (remainingSize >= 8) {
                    char subChunkId[5] = {0};
                    file.read(subChunkId, 4);
                    uint32_t subChunkSize;
                    file.read(reinterpret_cast<char*>(&subChunkSize), 4);

                    if (subChunkSize == 0 || subChunkSize > remainingSize - 8) {
                        break;
                    }

                    // Read sub-chunk data
                    std::vector<char> subChunkData(subChunkSize);
                    file.read(subChunkData.data(), subChunkSize);
                    std::string subChunkValue(subChunkData.data(),
                                            strnlen(subChunkData.data(), subChunkSize));

                    // Map INFO chunk IDs to metadata fields
                    if (std::strncmp(subChunkId, "INAM", 4) == 0) {
                        metadata.title = subChunkValue;
                    } else if (std::strncmp(subChunkId, "IART", 4) == 0) {
                        metadata.artist = subChunkValue;
                    } else if (std::strncmp(subChunkId, "IPRD", 4) == 0) {
                        metadata.album = subChunkValue;
                    } else if (std::strncmp(subChunkId, "ICRD", 4) == 0) {
                        try {
                            metadata.year = static_cast<uint16_t>(std::stoi(subChunkValue));
                        } catch (...) {
                            // Invalid year format
                        }
                    } else if (std::strncmp(subChunkId, "IGNR", 4) == 0) {
                        metadata.genre = subChunkValue;
                    } else if (std::strncmp(subChunkId, "ITRK", 4) == 0) {
                        try {
                            metadata.track = static_cast<uint16_t>(std::stoi(subChunkValue));
                        } catch (...) {
                            // Invalid track number
                        }
                    }

                    remainingSize -= 8 + subChunkSize;
                    // Ensure word alignment
                    if (subChunkSize % 2 != 0) {
                        file.seekg(1, std::ios::cur);
                        remainingSize--;
                    }
                }
                break;  // Found INFO chunk
            } else {
                // Skip non-INFO LIST chunk
                file.seekg(chunkHeader.size - 4, std::ios::cur);
            }
        } else {
            // Skip other chunks
            file.seekg(chunkHeader.size, std::ios::cur);
        }

        // Ensure word alignment
        if (chunkHeader.size % 2 != 0) {
            file.seekg(1, std::ios::cur);
        }
    }

    // Restore position
    file.seekg(currentPos);
}

} // namespace vortex::core::fileio