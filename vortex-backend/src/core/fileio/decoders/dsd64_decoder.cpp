#include "dsd64_decoder.hpp"
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace vortex::core::fileio {

DSD64Decoder::DSD64Decoder() : initialized_(false) {}

DSD64Decoder::~DSD64Decoder() {
    shutdown();
}

bool DSD64Decoder::initialize() {
    if (initialized_) {
        return true;
    }

    initialized_ = true;
    Logger::info("DSD64 decoder initialized successfully");
    return true;
}

void DSD64Decoder::shutdown() {
    if (!initialized_) {
        return;
    }

    initialized_ = false;
    Logger::info("DSD64 decoder shutdown");
}

bool DSD64Decoder::canDecode(const std::string& filePath) const {
    if (!initialized_) {
        return false;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Read potential DSD signature
    std::vector<uint8_t> header(1024);
    file.read(reinterpret_cast<char*>(header.data()), header.size());
    size_t bytesRead = file.gcount();

    file.close();

    if (bytesRead < 4) {
        return false;
    }

    return isDSD64Format(header.data(), bytesRead);
}

std::optional<AudioData> DSD64Decoder::decode(const std::string& filePath) {
    if (!initialized_) {
        Logger::error("DSD64 decoder not initialized");
        return std::nullopt;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        Logger::error("Cannot open DSD64 file: {}", filePath);
        return std::nullopt;
    }

    try {
        Logger::info("Decoding DSD64 file: {}", filePath);

        // Parse DSD file to determine format and properties
        DSDFileInfo fileInfo;
        if (!parseDSDFile(file, fileInfo)) {
            Logger::error("Failed to parse DSD64 file: {}", filePath);
            return std::nullopt;
        }

        // Verify this is DSD64 (2.8224 MHz)
        if (fileInfo.sampleRate != 2822400.0) {
            Logger::error("File is not DSD64 (sample rate: {} Hz)", fileInfo.sampleRate);
            return std::nullopt;
        }

        // Create audio data structure
        AudioData audioData;
        audioData.sampleRate = fileInfo.sampleRate;
        audioData.channels = fileInfo.channels;
        audioData.bitDepth = 1;  // 1-bit DSD
        audioData.format = AudioFormat::DSD64;

        // Allocate buffer for decoded PCM data (converted from DSD)
        uint64_t totalSamples = static_cast<uint64_t>(fileInfo.totalFrames * fileInfo.channels);
        size_t pcmSize = totalSamples * sizeof(float);
        audioData.data.resize(pcmSize);
        float* pcmData = reinterpret_cast<float*>(audioData.data.data());

        // Decode DSD to PCM
        if (!decodeDSDToPCM(file, fileInfo, pcmData, totalSamples)) {
            Logger::error("Failed to decode DSD64 to PCM");
            return std::nullopt;
        }

        Logger::info("DSD64 decoded successfully: {} samples, {} channels, {:.2f} seconds",
                    totalSamples / audioData.channels, audioData.channels,
                    static_cast<double>(totalSamples / audioData.channels) / audioData.sampleRate);

        return audioData;

    } catch (const std::exception& e) {
        Logger::error("Exception during DSD64 decoding: {}", e.what());
        return std::nullopt;
    }
}

bool DSD64Decoder::extractMetadata(const std::string& filePath, AudioMetadata& metadata) {
    if (!initialized_) {
        return false;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    try {
        // Initialize metadata structure
        metadata.format = AudioFormat::DSD64;
        metadata.codec = "DSD64";

        // Parse DSD file
        DSDFileInfo fileInfo;
        if (!parseDSDFile(file, fileInfo)) {
            return false;
        }

        // Extract technical metadata
        metadata.sampleRate = static_cast<uint32_t>(fileInfo.sampleRate);
        metadata.channels = static_cast<uint16_t>(fileInfo.channels);
        metadata.bitDepth = 1;  // 1-bit DSD

        // Calculate duration
        metadata.duration = std::chrono::duration<double>(
            static_cast<double>(fileInfo.totalFrames) / fileInfo.sampleRate);

        // Calculate bitrate (DSD64 = 2.8224 MHz per channel)
        metadata.bitrate = static_cast<uint32_t>(
            fileInfo.sampleRate * fileInfo.channels / 1000.0);

        // Extract DSD-specific metadata
        extractDSDMetadata(file, metadata);

        return true;

    } catch (const std::exception& e) {
        Logger::error("Exception during DSD64 metadata extraction: {}", e.what());
        return false;
    }
}

bool DSD64Decoder::isDSD64Format(const uint8_t* data, size_t size) const {
    if (size < 4) {
        return false;
    }

    // Check for DSDIFF format (DFF)
    if (size >= 4 && data[0] == 'F' && data[1] == 'R' && data[2] == 'M' && data[3] == '8') {
        return true;
    }

    // Check for DSF format
    if (size >= 4 && data[0] == 'D' && data[1] == 'S' && data[2] == 'F' && data[3] == ' ') {
        return true;
    }

    return false;
}

bool DSD64Decoder::parseDSDFile(std::ifstream& file, DSDFileInfo& fileInfo) {
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read first 4 bytes to determine format
    uint8_t signature[4];
    file.read(reinterpret_cast<char*>(signature), 4);

    if (file.gcount() < 4) {
        return false;
    }

    // Check for DSDIFF (DFF) format
    if (signature[0] == 'F' && signature[1] == 'R' && signature[2] == 'M' && signature[3] == '8') {
        return parseDSDIFFFile(file, fileInfo);
    }

    // Check for DSF format
    if (signature[0] == 'D' && signature[1] == 'S' && signature[2] == 'F' && signature[3] == ' ') {
        return parseDSFFile(file, fileInfo);
    }

    return false;
}

bool DSD64Decoder::parseDSDIFFFile(std::ifstream& file, DSDFileInfo& fileInfo) {
    // DSDIFF file format parsing
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    fileInfo.totalFrames = 0;
    fileInfo.sampleRate = 0;
    fileInfo.channels = 2;
    fileInfo.bitDepth = 1;

    // Read DSDIFF header
    uint8_t dsdHeader[28];
    file.read(reinterpret_cast<char*>(dsdHeader), 28);

    if (file.gcount() < 28) {
        return false;
    }

    // Parse DSD chunks
    while (file.tellg() < static_cast<std::streamoff>(fileSize)) {
        uint32_t chunkId, chunkSize;
        file.read(reinterpret_cast<char*>(&chunkId), 4);
        file.read(reinterpret_cast<char*>(&chunkSize), 4);

        if (file.eof()) {
            break;
        }

        chunkSize = ntohl(chunkSize);

        char chunkIdStr[5] = {0};
        std::memcpy(chunkIdStr, &chunkId, 4);

        if (std::strcmp(chunkIdStr, "fmt ") == 0) {
            // Format chunk
            if (chunkSize >= 40) {
                uint8_t formatData[40];
                file.read(reinterpret_cast<char*>(formatData), 40);

                // Extract sample rate
                fileInfo.sampleRate = (static_cast<uint32_t>(formatData[8]) << 24) |
                                     (static_cast<uint32_t>(formatData[9]) << 16) |
                                     (static_cast<uint32_t>(formatData[10]) << 8) |
                                     static_cast<uint32_t>(formatData[11]);

                // Extract channels
                fileInfo.channels = (static_cast<uint16_t>(formatData[16]) << 8) |
                                   static_cast<uint16_t>(formatData[17]);

                Logger::debug("DSDIFF format: {} Hz, {} channels", fileInfo.sampleRate, fileInfo.channels);
            }
        } else if (std::strcmp(chunkIdStr, "data") == 0) {
            // Data chunk
            fileInfo.dataOffset = file.tellg();
            fileInfo.dataSize = chunkSize;

            // Calculate total frames (1-bit samples per channel)
            if (fileInfo.channels > 0) {
                fileInfo.totalFrames = (chunkSize * 8) / fileInfo.channels;
            }

            Logger::debug("DSDIFF data: {} bytes, {} frames", chunkSize, fileInfo.totalFrames);
            break;  // Found data chunk, no need to read further
        } else {
            // Skip unknown chunks
            file.seekg(chunkSize, std::ios::cur);
        }
    }

    return fileInfo.totalFrames > 0 && fileInfo.sampleRate > 0;
}

bool DSD64Decoder::parseDSFFile(std::ifstream& file, DSDFileInfo& fileInfo) {
    // DSF file format parsing
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read DSF header
    uint8_t dsfHeader[52];
    file.read(reinterpret_cast<char*>(dsfHeader), 52);

    if (file.gcount() < 52) {
        return false;
    }

    // Extract DSD format information from DSF header
    fileInfo.sampleRate = (static_cast<uint32_t>(dsfHeader[28]) << 24) |
                         (static_cast<uint32_t>(dsfHeader[29]) << 16) |
                         (static_cast<uint32_t>(dsfHeader[30]) << 8) |
                         static_cast<uint32_t>(dsfHeader[31]);

    fileInfo.channels = (static_cast<uint16_t>(dsfHeader[32]) << 8) |
                       static_cast<uint16_t>(dsfHeader[33]);

    fileInfo.bitDepth = (static_cast<uint16_t>(dsfHeader[36]) << 8) |
                       static_cast<uint16_t>(dsfHeader[37]);

    // Calculate total samples
    uint64_t sampleCount = (static_cast<uint64_t>(dsfHeader[40]) << 56) |
                          (static_cast<uint64_t>(dsfHeader[41]) << 48) |
                          (static_cast<uint64_t>(dsfHeader[42]) << 40) |
                          (static_cast<uint64_t>(dsfHeader[43]) << 32) |
                          (static_cast<uint64_t>(dsfHeader[44]) << 24) |
                          (static_cast<uint64_t>(dsfHeader[45]) << 16) |
                          (static_cast<uint64_t>(dsfHeader[46]) << 8) |
                          static_cast<uint64_t>(dsfHeader[47]);

    fileInfo.totalFrames = sampleCount / fileInfo.channels;
    fileInfo.dataOffset = 52;  // After header
    fileInfo.dataSize = fileSize - 52;

    Logger::debug("DSF format: {} Hz, {} channels, {} total frames",
                 fileInfo.sampleRate, fileInfo.channels, fileInfo.totalFrames);

    return fileInfo.totalFrames > 0 && fileInfo.sampleRate > 0;
}

bool DSD64Decoder::decodeDSDToPCM(std::ifstream& file, const DSDFileInfo& fileInfo,
                                  float* pcmData, uint64_t totalSamples) {
    // Seek to audio data
    file.seekg(fileInfo.dataOffset);

    // DSD64 has a very high sample rate (2.8224 MHz), so we need to downsample
    // for practical PCM output. We'll use a downsampling ratio of 64:1
    // to get a standard high-quality sample rate.
    const int downsampleRatio = 64;
    const double targetSampleRate = fileInfo.sampleRate / downsampleRatio;

    Logger::debug("DSD64 downsample ratio: {} ({} Hz -> {} Hz)",
                 downsampleRatio, fileInfo.sampleRate, targetSampleRate);

    // Read DSD data in chunks and convert to PCM
    const size_t chunkSize = 65536;  // 64KB chunks
    std::vector<uint8_t> dsdData(chunkSize);
    uint64_t samplesProcessed = 0;
    uint64_t pcmIndex = 0;

    // DSD to PCM conversion buffer
    std::vector<float> dsdBuffer(downsampleRatio * fileInfo.channels);
    std::vector<float> accumulators(fileInfo.channels, 0.0f);
    int dsdAccumulatorIndex = 0;

    while (samplesProcessed < totalSamples && file.tellg() < fileInfo.dataOffset + static_cast<std::streamoff>(fileInfo.dataSize)) {
        size_t bytesToRead = std::min(chunkSize, fileInfo.dataSize - (file.tellg() - fileInfo.dataOffset));
        file.read(reinterpret_cast<char*>(dsdData.data()), bytesToRead);
        size_t bytesRead = file.gcount();

        if (bytesRead == 0) {
            break;
        }

        // Process DSD bytes
        for (size_t byteIndex = 0; byteIndex < bytesRead; ++byteIndex) {
            uint8_t dsdByte = dsdData[byteIndex];

            // Process each bit in the byte (8 DSD samples)
            for (int bitIndex = 0; bitIndex < 8; ++bitIndex) {
                if (samplesProcessed >= totalSamples) {
                    break;
                }

                // Extract bit
                bool dsdBit = (dsdByte >> (7 - bitIndex)) & 1;

                // Accumulate DSD samples for downsampling
                for (uint16_t channel = 0; channel < fileInfo.channels; ++channel) {
                    accumulators[channel] += dsdBit ? 1.0f : -1.0f;
                }

                samplesProcessed++;

                // Check if we have enough samples for downsampling
                if (++dsdAccumulatorIndex >= downsampleRatio) {
                    // Output downsampled PCM values
                    for (uint16_t channel = 0; channel < fileInfo.channels; ++channel) {
                        if (pcmIndex < totalSamples) {
                            // Convert accumulated DSD to PCM and normalize
                            pcmData[pcmIndex] = accumulators[channel] / downsampleRatio;
                            pcmIndex++;
                        }
                        accumulators[channel] = 0.0f;
                    }
                    dsdAccumulatorIndex = 0;
                }
            }

            if (samplesProcessed >= totalSamples) {
                break;
            }
        }
    }

    // Flush remaining accumulator if any
    if (dsdAccumulatorIndex > 0 && pcmIndex < totalSamples) {
        for (uint16_t channel = 0; channel < fileInfo.channels; ++channel) {
            if (pcmIndex < totalSamples) {
                pcmData[pcmIndex] = accumulators[channel] / dsdAccumulatorIndex;
                pcmIndex++;
            }
        }
    }

    // Update audio data info with actual downsampled rate
    if (pcmIndex > 0) {
        Logger::debug("DSD64 to PCM conversion: {} DSD samples -> {} PCM samples ({}x downsampling)",
                     samplesProcessed, pcmIndex, samplesProcessed / pcmIndex);
    }

    return pcmIndex > 0;
}

void DSD64Decoder::extractDSDMetadata(std::ifstream& file, AudioMetadata& metadata) {
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read first 4 bytes to determine format
    uint8_t signature[4];
    file.read(reinterpret_cast<char*>(signature), 4);

    file.seekg(0, std::ios::beg);

    // Extract format-specific metadata
    if (signature[0] == 'F' && signature[1] == 'R' && signature[2] == 'M' && signature[3] == '8') {
        extractDSDIFFMetadata(file, metadata);
    } else if (signature[0] == 'D' && signature[1] == 'S' && signature[2] == 'F' && signature[3] == ' ') {
        extractDSFMetadata(file, metadata);
    }
}

void DSD64Decoder::extractDSDIFFMetadata(std::ifstream& file, AudioMetadata& metadata) {
    // Look for DSDIFF metadata chunks
    while (file.tellg() < static_cast<std::streamoff>(fileSize)) {
        uint32_t chunkId, chunkSize;
        file.read(reinterpret_cast<char*>(&chunkId), 4);
        file.read(reinterpret_cast<char*>(&chunkSize), 4);

        if (file.eof()) {
            break;
        }

        chunkSize = ntohl(chunkSize);

        char chunkIdStr[5] = {0};
        std::memcpy(chunkIdStr, &chunkId, 4);

        // Look for metadata chunks
        if (std::strcmp(chunkIdStr, "DITI") == 0) {
            // DSDIFF title chunk
            std::vector<char> titleData(chunkSize);
            file.read(titleData.data(), chunkSize);
            metadata.title = std::string(titleData.data(), chunkSize);
        } else if (std::strcmp(chunkIdStr, "DIAR") == 0) {
            // DSDIFF artist chunk
            std::vector<char> artistData(chunkSize);
            file.read(artistData.data(), chunkSize);
            metadata.artist = std::string(artistData.data(), chunkSize);
        } else {
            // Skip other chunks
            file.seekg(chunkSize, std::ios::cur);
        }
    }
}

void DSD64Decoder::extractDSFMetadata(std::ifstream& file, AudioMetadata& metadata) {
    // DSF format typically doesn't have extensive metadata
    // Could add metadata extraction from ID3 tags if present
    // For now, just log the format info
    Logger::debug("DSF metadata extracted for DSD64 format");
}

} // namespace vortex::core::fileio