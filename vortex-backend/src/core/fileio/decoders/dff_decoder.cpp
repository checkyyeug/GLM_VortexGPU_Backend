#include "dff_decoder.hpp"
#include "dsd64_decoder.hpp"
#include "dsd128_decoder.hpp"
#include "dsd256_decoder.hpp"
#include "dsd512_decoder.hpp"
#include "dsd1024_decoder.hpp"
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace vortex::core::fileio {

DFFDecoder::DFFDecoder() : initialized_(false) {}

DFFDecoder::~DFFDecoder() {
    shutdown();
}

bool DFFDecoder::initialize() {
    if (initialized_) {
        return true;
    }

    initialized_ = true;
    Logger::info("DFF decoder initialized successfully");
    return true;
}

void DFFDecoder::shutdown() {
    if (!initialized_) {
        return;
    }

    initialized_ = false;
    Logger::info("DFF decoder shutdown");
}

bool DFFDecoder::canDecode(const std::string& filePath) const {
    if (!initialized_) {
        return false;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Read DFF signature
    uint8_t signature[12];
    file.read(reinterpret_cast<char*>(signature), 12);
    size_t bytesRead = file.gcount();

    file.close();

    if (bytesRead < 12) {
        return false;
    }

    // Check for DSDIFF signature "FRM8" followed by format "DSD "
    return (signature[0] == 'F' && signature[1] == 'R' && signature[2] == 'M' && signature[3] == '8' &&
            signature[8] == 'D' && signature[9] == 'S' && signature[10] == 'D' && signature[11] == ' ');
}

std::optional<AudioData> DFFDecoder::decode(const std::string& filePath) {
    if (!initialized_) {
        Logger::error("DFF decoder not initialized");
        return std::nullopt;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        Logger::error("Cannot open DFF file: {}", filePath);
        return std::nullopt;
    }

    try {
        Logger::info("Decoding DFF file: {}", filePath);

        // Parse DSDIFF header
        DSDIFFHeader header;
        if (!parseDSDIFFHeader(file, header)) {
            Logger::error("Failed to parse DSDIFF header: {}", filePath);
            return std::nullopt;
        }

        // Determine appropriate DSD decoder based on sample rate
        std::unique_ptr<ISDDecoder> dsdDecoder;
        if (header.sampleRate == 2822400.0) {
            // DSD64
            dsdDecoder = std::make_unique<DSD64Decoder>();
        } else if (header.sampleRate == 5644800.0) {
            // DSD128
            dsdDecoder = std::make_unique<DSD128Decoder>();
        } else if (header.sampleRate == 11289600.0) {
            // DSD256
            dsdDecoder = std::make_unique<DSD256Decoder>();
        } else if (header.sampleRate == 22579200.0) {
            // DSD512
            dsdDecoder = std::make_unique<DSD512Decoder>();
        } else if (header.sampleRate == 45158400.0) {
            // DSD1024
            dsdDecoder = std::make_unique<DSD1024Decoder>();
        } else {
            Logger::error("Unsupported DSDIFF sample rate: {} Hz", header.sampleRate);
            return std::nullopt;
        }

        // Initialize the appropriate decoder
        if (!dsdDecoder->initialize()) {
            Logger::error("Failed to initialize DSD decoder for sample rate: {} Hz", header.sampleRate);
            return std::nullopt;
        }

        // Create audio data structure with DFF-specific format
        AudioData audioData;
        audioData.sampleRate = header.sampleRate;
        audioData.channels = header.channels;
        audioData.bitDepth = 1;  // 1-bit DSD
        audioData.format = AudioFormat::DFF;

        // Read DSDIFF audio data
        std::vector<uint8_t> dsdData(header.dataSize);
        file.seekg(header.dataOffset);
        file.read(reinterpret_cast<char*>(dsdData.data()), header.dataSize);

        if (file.gcount() != static_cast<std::streamsize>(header.dataSize)) {
            Logger::error("Failed to read DSDIFF audio data");
            return std::nullopt;
        }

        // Use the appropriate DSD decoder to process the data
        // Since we have a different container (DFF) but same data format as other DSD files,
        // we'll create a temporary file and delegate to the specific decoder
        std::string tempFile = createTempDSDFile(dsdData, header);

        auto result = dsdDecoder->decode(tempFile);

        // Clean up temporary file
        std::remove(tempFile.c_str());

        if (!result) {
            Logger::error("Failed to decode DSDIFF audio data with {} decoder",
                         getDSDFormatName(header.sampleRate));
            return std::nullopt;
        }

        // Copy the decoded data but maintain DFF format information
        audioData.data = std::move(result->data);
        audioData.format = AudioFormat::DFF;  // Preserve DFF format

        Logger::info("DFF decoded successfully: {} samples, {} channels, {:.2f} seconds ({} format)",
                    audioData.data.size() / (sizeof(float) * audioData.channels),
                    audioData.channels,
                    static_cast<double>(audioData.data.size() / (sizeof(float) * audioData.channels)) / audioData.sampleRate,
                    getDSDFormatName(header.sampleRate));

        return audioData;

    } catch (const std::exception& e) {
        Logger::error("Exception during DFF decoding: {}", e.what());
        return std::nullopt;
    }
}

bool DFFDecoder::extractMetadata(const std::string& filePath, AudioMetadata& metadata) {
    if (!initialized_) {
        return false;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    try {
        // Initialize metadata structure
        metadata.format = AudioFormat::DFF;
        metadata.codec = "DFF";

        // Parse DSDIFF header
        DSDIFFHeader header;
        if (!parseDSDIFFHeader(file, header)) {
            return false;
        }

        // Extract technical metadata
        metadata.sampleRate = static_cast<uint32_t>(header.sampleRate);
        metadata.channels = static_cast<uint16_t>(header.channels);
        metadata.bitDepth = 1;  // 1-bit DSD

        // Calculate duration
        uint64_t totalSamples = static_cast<uint64_t>(header.totalSamples);
        metadata.duration = std::chrono::duration<double>(
            static_cast<double>(totalSamples / header.channels) / header.sampleRate);

        // Calculate bitrate
        metadata.bitrate = static_cast<uint32_t>(
            header.sampleRate * header.channels / 1000.0);

        // Determine DSD format name
        metadata.codec = getDSDFormatName(header.sampleRate);

        // Extract DSDIFF-specific metadata
        extractDSDIFFMetadata(file, metadata);

        return true;

    } catch (const std::chrono::system_error& e) {
        Logger::error("File system error during DFF metadata extraction: {}", e.what());
        return false;
    } catch (const std::exception& e) {
        Logger::error("Exception during DFF metadata extraction: {}", e.what());
        return false;
    }
}

bool DFFDecoder::isDFFFormat(const uint8_t* data, size_t size) const {
    return (size >= 12 &&
            data[0] == 'F' && data[1] == 'R' && data[2] == 'M' && data[3] == '8' &&
            data[8] == 'D' && data[9] == 'S' && data[10] == 'D' && data[11] == ' ');
}

bool DFFDecoder::parseDSDIFFHeader(std::ifstream& file, DSDIFFHeader& header) {
    // Read DSDIFF header
    uint8_t dsdHeader[12];
    file.read(reinterpret_cast<char*>(dsdHeader), 12);

    if (file.gcount() < 12) {
        return false;
    }

    // Verify DSDIFF signature
    if (dsdHeader[0] != 'F' || dsdHeader[1] != 'R' || dsdHeader[2] != 'M' || dsdHeader[3] != '8') {
        Logger::error("Invalid DSDIFF signature");
        return false;
    }

    // Extract file size
    header.fileSize = (static_cast<uint64_t>(dsdHeader[4]) << 56) |
                      (static_cast<uint64_t>(dsdHeader[5]) << 48) |
                      (static_cast<uint64_t>(dsdHeader[6]) << 40) |
                      (static_cast<uint64_t>(dsdHeader[7]) << 32) |
                      (static_cast<uint64_t>(dsdHeader[8]) << 24) |
                      (static_cast<uint64_t>(dsdHeader[9]) << 16) |
                      (static_cast<uint64_t>(dsdHeader[10]) << 8) |
                      static_cast<uint64_t>(dsdHeader[11]);

    // Verify DSD format
    if (dsdHeader[8] != 'D' || dsdHeader[9] != 'S' || dsdHeader[10] != 'D' || dsdHeader[11] != ' ') {
        Logger::error("Invalid DSD format in DSDIFF file");
        return false;
    }

    // Parse DSDIFF chunks to find format and data chunks
    file.seekg(12);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(12);

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
                header.sampleRate = (static_cast<uint32_t>(formatData[8]) << 24) |
                                   (static_cast<uint32_t>(formatData[9]) << 16) |
                                   (static_cast<uint32_t>(formatData[10]) << 8) |
                                   static_cast<uint32_t>(formatData[11]);

                // Extract channels
                header.channels = (static_cast<uint16_t>(formatData[16]) << 8) |
                                 static_cast<uint16_t>(formatData[17]);

                // Extract bit depth
                header.bitDepth = (static_cast<uint16_t>(formatData[20]) << 8) |
                                static_cast<uint16_t>(formatData[21]);

                // Extract compression type
                header.compressionType = formatData[36];
            } else {
                // Skip smaller format chunk
                file.seekg(chunkSize, std::ios::cur);
            }
        } else if (std::strcmp(chunkIdStr, "data") == 0) {
            // Data chunk
            header.dataOffset = file.tellg();
            header.dataSize = chunkSize;

            // Calculate total samples (1-bit samples per channel)
            if (header.channels > 0) {
                header.totalSamples = (chunkSize * 8) / header.channels;
            }

            Logger::debug("DSDIFF data: {} bytes, {} samples", chunkSize, header.totalSamples);
            break;  // Found data chunk, no need to read further
        } else {
            // Skip unknown chunks
            file.seekg(chunkSize, std::ios::cur);
        }
    }

    Logger::debug("DSDIFF Header: {} Hz, {} channels, {} bit depth, {} total samples, {} compression",
                 header.sampleRate, header.channels, header.bitDepth,
                 header.totalSamples, static_cast<int>(header.compressionType));

    return header.dataSize > 0 && header.sampleRate > 0;
}

std::string DFFDecoder::createTempDSDFile(const std::vector<uint8_t>& dsdData, const DSDIFFHeader& header) {
    // Create a temporary file path
    std::string tempDir = std::filesystem::temp_directory_path().string();
    std::string tempFile = tempDir + "/vortex_dsd_temp_" +
                          std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) +
                          ".dsd";

    // Write DSD data to temporary file
    std::ofstream tempStream(tempFile, std::ios::binary);
    if (!tempStream.is_open()) {
        throw std::runtime_error("Failed to create temporary DSD file");
    }

    // Write DSDIFF header
    tempStream.write("FRM8", 4);  // Signature

    // Write file size
    uint64_t fileSize = 12 + 40 + 12 + dsdData.size();  // Approximate size
    tempStream.put(static_cast<uint8_t>((fileSize >> 56) & 0xFF));
    tempStream.put(static_cast<uint8_t>((fileSize >> 48) & 0xFF));
    tempStream.put(static_cast<uint8_t>((fileSize >> 40) & 0xFF));
    tempStream.put(static_cast<uint8_t>((fileSize >> 32) & 0xFF));
    tempStream.put(static_cast<uint8_t>((fileSize >> 24) & 0xFF));
    tempStream.put(static_cast<uint8_t>((fileSize >> 16) & 0xFF));
    tempStream.put(static_cast<uint8_t>((fileSize >> 8) & 0xFF));
    tempStream.put(static_cast<uint8_t>(fileSize & 0xFF));

    tempStream.write("DSD ", 4);  // Format

    // Write format chunk
    tempStream.write("fmt ", 4);  // Chunk ID
    uint32_t formatSize = 40;
    tempStream.put(static_cast<uint8_t>((formatSize >> 24) & 0xFF));
    tempStream.put(static_cast<uint8_t>((formatSize >> 16) & 0xFF));
    tempStream.put(static_cast<uint8_t>((formatSize >> 8) & 0xFF));
    tempStream.put(static_cast<uint8_t>(formatSize & 0xFF));

    // Format data
    uint8_t formatData[40] = {0};

    // Sample rate
    formatData[8] = static_cast<uint8_t>((header.sampleRate >> 24) & 0xFF);
    formatData[9] = static_cast<uint8_t>((header.sampleRate >> 16) & 0xFF);
    formatData[10] = static_cast<uint8_t>((header.sampleRate >> 8) & 0xFF);
    formatData[11] = static_cast<uint8_t>(header.sampleRate & 0xFF);

    // Channels
    formatData[16] = static_cast<uint8_t>((header.channels >> 8) & 0xFF);
    formatData[17] = static_cast<uint8_t>(header.channels & 0xFF);

    // Bit depth
    formatData[20] = static_cast<uint8_t>((header.bitDepth >> 8) & 0xFF);
    formatData[21] = static_cast<uint8_t>(header.bitDepth & 0xFF);

    // Write format data
    tempStream.write(reinterpret_cast<char*>(formatData), 40);

    // Write data chunk
    tempStream.write("data", 4);  // Chunk ID
    tempStream.put(static_cast<uint8_t>((dsdData.size() >> 24) & 0xFF));
    tempStream.put(static_cast<uint8_t>((dsdData.size() >> 16) & 0xFF));
    tempStream.put(static_cast<uint8_t>((dsdData.size() >> 8) & 0xFF));
    tempStream.put(static_cast<uint8_t>(dsdData.size() & 0xFF));

    // Write actual DSD data
    tempStream.write(reinterpret_cast<const char*>(dsdData.data()), dsdData.size());

    tempStream.close();
    return tempFile;
}

std::string DFFDecoder::getDSDFormatName(double sampleRate) const {
    if (sampleRate >= 45000000.0) {
        return "DSD1024";
    } else if (sampleRate >= 22000000.0) {
        return "DSD512";
    } else if (sampleRate >= 11000000.0) {
        return "DSD256";
    } else if (sampleRate >= 5500000.0) {
        return "DSD128";
    } else if (sampleRate >= 2800000.0) {
        return "DSD64";
    } else {
        return "DSD";
    }
}

void DFFDecoder::extractDSDIFFMetadata(std::ifstream& file, AudioMetadata& metadata) {
    // Go back to beginning and look for metadata chunks
    file.seekg(12);

    // Get file size
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(12);

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
            // Title chunk
            std::vector<char> titleData(chunkSize);
            file.read(titleData.data(), chunkSize);
            metadata.title = std::string(titleData.data(), chunkSize);
            Logger::debug("DSDIFF title: {}", metadata.title);
        } else if (std::strcmp(chunkIdStr, "DIAR") == 0) {
            // Artist chunk
            std::vector<char> artistData(chunkSize);
            file.read(artistData.data(), chunkSize);
            metadata.artist = std::string(artistData.data(), chunkSize);
            Logger::debug("DSDIFF artist: {}", metadata.artist);
        } else if (std::strcmp(chunkIdStr, "DIGN") == 0) {
            // Genre chunk
            std::vector<char> genreData(chunkSize);
            file.read(genreData.data(), chunkSize);
            metadata.genre = std::string(genreData.data(), chunkSize);
            Logger::debug("DSDIFF genre: {}", metadata.genre);
        } else if (std::strcmp(chunkIdStr, "IDYT") == 0) {
            // Year chunk
            std::vector<char> yearData(chunkSize);
            file.read(yearData.data(), chunkSize);
            try {
                if (chunkSize >= 4) {
                    metadata.year = static_cast<uint16_t>(std::stoi(std::string(yearData.data(), 4)));
                    Logger::debug("DSDIFF year: {}", metadata.year);
                }
            } catch (...) {
                // Invalid year format
            }
        } else if (std::strcmp(chunkIdStr, "DTCO") == 0) {
            // Comment chunk
            std::vector<char> commentData(chunkSize);
            file.read(commentData.data(), chunkSize);
            metadata.comment = std::string(commentData.data(), chunkSize);
            Logger::debug("DSDIFF comment: {}", metadata.comment);
        } else if (std::strcmp(chunkIdStr, "DIEN") == 0) {
            // Encoder chunk
            std::vector<char> encoderData(chunkSize);
            file.read(encoderData.data(), chunkSize);
            Logger::debug("DSDIFF encoder: {}", std::string(encoderData.data(), chunkSize));
        } else {
            // Skip other chunks
            file.seekg(chunkSize, std::ios::cur);
        }
    }
}

void DFFDecoder::extractID3v2Metadata(const uint8_t* data, size_t size, AudioMetadata& metadata) {
    // DSDIFF can contain ID3v2 metadata at the end of the file
    // This function would handle ID3v2 tag parsing if needed
    // For now, DSDIFF-specific chunks are handled in extractDSDIFFMetadata
    Logger::debug("ID3v2 metadata extraction not yet implemented for DSDIFF");
}

} // namespace vortex::core::fileio