#include "mp3_decoder.hpp"
#include <mpg123.h>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace vortex::core::fileio {

MP3Decoder::MP3Decoder() : handle_(nullptr), initialized_(false) {}

MP3Decoder::~MP3Decoder() {
    shutdown();
}

bool MP3Decoder::initialize() {
    if (initialized_) {
        return true;
    }

    // Initialize mpg123 library
    int result = mpg123_init();
    if (result != MPG123_OK) {
        Logger::error("Failed to initialize mpg123 library: {}", mpg123_plain_strerror(result));
        return false;
    }

    // Create mpg123 handle
    handle_ = mpg123_new(nullptr, &result);
    if (handle_ == nullptr) {
        Logger::error("Failed to create mpg123 handle: {}", mpg123_plain_strerror(result));
        mpg123_exit();
        return false;
    }

    // Enable all supported formats
    result = mpg123_format_all(handle_);
    if (result != MPG123_OK) {
        Logger::error("Failed to set mpg123 formats: {}", mpg123_plain_strerror(result));
        mpg123_delete(handle_);
        mpg123_exit();
        return false;
    }

    initialized_ = true;
    Logger::info("MP3 decoder initialized successfully");
    return true;
}

void MP3Decoder::shutdown() {
    if (!initialized_) {
        return;
    }

    if (handle_) {
        mpg123_delete(handle_);
        handle_ = nullptr;
    }

    mpg123_exit();
    initialized_ = false;
    Logger::info("MP3 decoder shutdown");
}

bool MP3Decoder::canDecode(const std::string& filePath) const {
    if (!initialized_) {
        return false;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Read header to check for MP3 signature
    std::vector<uint8_t> header(1024);
    file.read(reinterpret_cast<char*>(header.data()), header.size());
    size_t bytesRead = file.gcount();

    file.close();

    return isMP3Format(header.data(), bytesRead);
}

std::optional<AudioData> MP3Decoder::decode(const std::string& filePath) {
    if (!initialized_) {
        Logger::error("MP3 decoder not initialized");
        return std::nullopt;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        Logger::error("Cannot open MP3 file: {}", filePath);
        return std::nullopt;
    }

    // Get file size
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    if (fileSize == 0) {
        Logger::error("Empty MP3 file: {}", filePath);
        return std::nullopt;
    }

    Logger::info("Decoding MP3 file: {} ({} bytes)", filePath, fileSize);

    // Open file with mpg123
    int result = mpg123_open_handle(handle_, &file);
    if (result != MPG123_OK) {
        Logger::error("Failed to open MP3 file with mpg123: {}", mpg123_plain_strerror(result));
        return std::nullopt;
    }

    try {
        // Get audio format information
        long rate = 0;
        int channels = 0;
        int encoding = 0;
        result = mpg123_getformat(handle_, &rate, &channels, &encoding);

        if (result != MPG123_OK) {
            Logger::error("Failed to get MP3 format: {}", mpg123_plain_strerror(result));
            mpg123_close(handle_);
            return std::nullopt;
        }

        // Validate format
        if (channels > 2 || rate > 96000) {
            Logger::warn("Unsupported MP3 format: {} Hz, {} channels", rate, channels);
        }

        // Determine output format (always convert to 32-bit float)
        AudioData audioData;
        audioData.sampleRate = static_cast<double>(rate);
        audioData.channels = static_cast<uint16_t>(channels);
        audioData.bitDepth = 32;
        audioData.format = AudioFormat::MP3;

        // Calculate PCM data size
        off_t totalSamples = mpg123_length(handle_);
        if (totalSamples == MPG123_ERR) {
            // If we can't get the exact length, estimate based on file size
            // Typical MP3 bitrate is ~128 kbps per channel
            double estimatedSeconds = (fileSize * 8.0) / (128000.0 * channels);
            totalSamples = static_cast<off_t>(estimatedSeconds * rate * channels);
        }

        // Reserve space for audio data
        size_t pcmSize = totalSamples * sizeof(float);
        audioData.data.resize(pcmSize);
        float* pcmData = reinterpret_cast<float*>(audioData.data.data());

        // Decode MP3 to PCM
        size_t decodedSamples = 0;
        size_t frameCount = 0;
        const size_t bufferSize = 4096;  // Process in 4KB chunks
        unsigned char buffer[bufferSize];
        size_t done = 0;

        while (true) {
            result = mpg123_read(handle_, buffer, bufferSize, &done);

            if (done > 0) {
                // Convert to float32
                size_t samplesToConvert = done / sizeof(int16_t);
                int16_t* intData = reinterpret_cast<int16_t*>(buffer);

                for (size_t i = 0; i < samplesToConvert && decodedSamples + i < totalSamples; ++i) {
                    pcmData[decodedSamples + i] = static_cast<float>(intData[i]) / 32768.0f;
                }

                decodedSamples += samplesToConvert;
                frameCount++;
            }

            if (result == MPG123_DONE) {
                break;
            }

            if (result != MPG123_OK) {
                Logger::warn("MP3 decoding warning: {}", mpg123_plain_strerror(result));
            }
        }

        // Resize audio data to actual decoded size
        size_t actualDataSize = decodedSamples * sizeof(float);
        audioData.data.resize(actualDataSize);

        Logger::info("MP3 decoded successfully: {} samples, {} frames, {:.2f} seconds",
                    decodedSamples, frameCount, static_cast<double>(decodedSamples) / (rate * channels));

        mpg123_close(handle_);
        return audioData;

    } catch (const std::exception& e) {
        Logger::error("Exception during MP3 decoding: {}", e.what());
        mpg123_close(handle_);
        return std::nullopt;
    }
}

bool MP3Decoder::extractMetadata(const std::string& filePath, AudioMetadata& metadata) {
    if (!initialized_) {
        return false;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Initialize metadata structure
    metadata.format = AudioFormat::MP3;
    metadata.codec = "MP3";

    // Extract ID3v2 tag if present
    extractID3v2Metadata(file, metadata);

    // Extract basic technical metadata from the MP3 stream
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Open with mpg123 to get technical info
    int result = mpg123_open_handle(handle_, &file);
    if (result == MPG123_OK) {
        long rate = 0;
        int channels = 0;
        int encoding = 0;

        if (mpg123_getformat(handle_, &rate, &channels, &encoding) == MPG123_OK) {
            metadata.sampleRate = static_cast<uint32_t>(rate);
            metadata.channels = static_cast<uint16_t>(channels);
            metadata.bitDepth = 16;  // MP3 is typically 16-bit equivalent
        }

        off_t totalSamples = mpg123_length(handle_);
        if (totalSamples != MPG123_ERR) {
            metadata.duration = std::chrono::duration<double>(
                static_cast<double>(totalSamples) / (rate * channels));
        }

        // Estimate bitrate (MP3 doesn't provide exact bitrate easily)
        double durationSeconds = metadata.duration.count();
        if (durationSeconds > 0) {
            metadata.bitrate = static_cast<uint32_t>((fileSize * 8.0) / durationSeconds / 1000.0);
        }

        mpg123_close(handle_);
    }

    return true;
}

bool MP3Decoder::isMP3Format(const uint8_t* data, size_t size) {
    if (size < 3) {
        return false;
    }

    // Check for ID3v2 tag
    if (size >= 10 && data[0] == 'I' && data[1] == 'D' && data[2] == '3') {
        return true;
    }

    // Check for MPEG sync pattern (11 consecutive set bits)
    if ((data[0] == 0xFF) && ((data[1] & 0xE0) == 0xE0)) {
        return true;
    }

    // Check for ID3v1 tag (at end of file)
    // For initial detection, we only check the beginning
    // The end tag would be checked during full file processing

    return false;
}

void MP3Decoder::extractID3v2Metadata(std::ifstream& file, AudioMetadata& metadata) {
    // Save current position
    auto currentPos = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read potential ID3v2 header
    uint8_t header[10];
    file.read(reinterpret_cast<char*>(header), 10);

    if (file.gcount() < 10) {
        file.seekg(currentPos);
        return;
    }

    // Check for ID3v2 signature
    if (header[0] != 'I' || header[1] != 'D' || header[2] != '3') {
        file.seekg(currentPos);
        return;
    }

    // Parse ID3v2 header
    uint8_t version = header[3];
    uint8_t flags = header[5];

    // Calculate tag size (synchsafe integer)
    uint32_t tagSize = (header[6] << 21) | (header[7] << 14) | (header[8] << 7) | header[9];

    Logger::debug("Found ID3v2.{} tag, size: {} bytes", version, tagSize);

    if (tagSize > 0 && tagSize < 10 * 1024 * 1024) {  // Reasonable size limit
        // Read tag data
        std::vector<uint8_t> tagData(tagSize);
        file.read(reinterpret_cast<char*>(tagData.data()), tagSize);

        if (file.gcount() == tagSize) {
            parseID3v2Frames(tagData.data(), tagSize, metadata);
        }
    }

    // Restore position
    file.seekg(currentPos);
}

void MP3Decoder::parseID3v2Frames(const uint8_t* data, size_t size, AudioMetadata& metadata) {
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