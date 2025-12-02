#include "dsf_decoder.hpp"
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace vortex::core::fileio {

DSFDecoder::DSFDecoder() : initialized_(false) {}

DSFDecoder::~DSFDecoder() {
    shutdown();
}

bool DSFDecoder::initialize() {
    if (initialized_) {
        return true;
    }

    initialized_ = true;
    Logger::info("DSF decoder initialized successfully");
    return true;
}

void DSFDecoder::shutdown() {
    if (!initialized_) {
        return;
    }

    initialized_ = false;
    Logger::info("DSF decoder shutdown");
}

bool DSFDecoder::canDecode(const std::string& filePath) const {
    if (!initialized_) {
        return false;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Read DSF signature
    uint8_t signature[4];
    file.read(reinterpret_cast<char*>(signature), 4);
    size_t bytesRead = file.gcount();

    file.close();

    if (bytesRead < 4) {
        return false;
    }

    // Check for DSF signature "DSF "
    return (signature[0] == 'D' && signature[1] == 'S' && signature[2] == 'F' && signature[3] == ' ');
}

std::optional<AudioData> DSFDecoder::decode(const std::string& filePath) {
    if (!initialized_) {
        Logger::error("DSF decoder not initialized");
        return std::nullopt;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        Logger::error("Cannot open DSF file: {}", filePath);
        return std::nullopt;
    }

    try {
        Logger::info("Decoding DSF file: {}", filePath);

        // Parse DSF header
        DSFHeader header;
        if (!parseDSFHeader(file, header)) {
            Logger::error("Failed to parse DSF header: {}", filePath);
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
            Logger::error("Unsupported DSF sample rate: {} Hz", header.sampleRate);
            return std::nullopt;
        }

        // Initialize the appropriate decoder
        if (!dsdDecoder->initialize()) {
            Logger::error("Failed to initialize DSD decoder for sample rate: {} Hz", header.sampleRate);
            return std::nullopt;
        }

        // Create audio data structure with DSF-specific format
        AudioData audioData;
        audioData.sampleRate = header.sampleRate;
        audioData.channels = header.channels;
        audioData.bitDepth = 1;  // 1-bit DSD
        audioData.format = AudioFormat::DSF;

        // Read DSF audio data
        std::vector<uint8_t> dsdData(header.dataSize);
        file.seekg(header.dataOffset);
        file.read(reinterpret_cast<char*>(dsdData.data()), header.dataSize);

        if (file.gcount() != static_cast<std::streamsize>(header.dataSize)) {
            Logger::error("Failed to read DSF audio data");
            return std::nullopt;
        }

        // Use the appropriate DSD decoder to process the data
        // Since we have a different container (DSF) but same data format as other DSD files,
        // we'll create a temporary file and delegate to the specific decoder
        std::string tempFile = createTempDSDFile(dsdData, header);

        auto result = dsdDecoder->decode(tempFile);

        // Clean up temporary file
        std::remove(tempFile.c_str());

        if (!result) {
            Logger::error("Failed to decode DSF audio data with {} decoder",
                         getDSDFormatName(header.sampleRate));
            return std::nullopt;
        }

        // Copy the decoded data but maintain DSF format information
        audioData.data = std::move(result->data);
        audioData.format = AudioFormat::DSF;  // Preserve DSF format

        Logger::info("DSF decoded successfully: {} samples, {} channels, {:.2f} seconds ({} format)",
                    audioData.data.size() / (sizeof(float) * audioData.channels),
                    audioData.channels,
                    static_cast<double>(audioData.data.size() / (sizeof(float) * audioData.channels)) / audioData.sampleRate,
                    getDSDFormatName(header.sampleRate));

        return audioData;

    } catch (const std::exception& e) {
        Logger::error("Exception during DSF decoding: {}", e.what());
        return std::nullopt;
    }
}

bool DSFDecoder::extractMetadata(const std::string& filePath, AudioMetadata& metadata) {
    if (!initialized_) {
        return false;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    try {
        // Initialize metadata structure
        metadata.format = AudioFormat::DSF;
        metadata.codec = "DSF";

        // Parse DSF header
        DSFHeader header;
        if (!parseDSFHeader(file, header)) {
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

        // Extract DSF-specific metadata
        extractDSFMetadata(file, metadata);

        return true;

    } catch (const std::chrono::system_error& e) {
        Logger::error("File system error during DSF metadata extraction: {}", e.what());
        return false;
    } catch (const std::exception& e) {
        Logger::error("Exception during DSF metadata extraction: {}", e.what());
        return false;
    }
}

bool DSFDecoder::isDSFFormat(const uint8_t* data, size_t size) const {
    return (size >= 4 && data[0] == 'D' && data[1] == 'S' && data[2] == 'F' && data[3] == ' ');
}

bool DSFDecoder::parseDSFHeader(std::ifstream& file, DSFHeader& header) {
    // Read DSF header
    uint8_t dsfHeader[52];
    file.read(reinterpret_cast<char*>(dsfHeader), 52);

    if (file.gcount() < 52) {
        return false;
    }

    // Verify DSF signature
    if (dsfHeader[0] != 'D' || dsfHeader[1] != 'S' || dsfHeader[2] != 'F' || dsfHeader[3] != ' ') {
        Logger::error("Invalid DSF signature");
        return false;
    }

    // Verify DSF version (should be 1.0)
    uint16_t version = (static_cast<uint16_t>(dsfHeader[4]) << 8) | static_cast<uint16_t>(dsfHeader[5]);
    if (version != 1) {
        Logger::warn("Unsupported DSF version: {}.{}. Expected 1.0", version >> 8, version & 0xFF);
    }

    // Extract DSD format information from DSF header
    header.sampleRate = (static_cast<uint64_t>(dsfHeader[28]) << 24) |
                       (static_cast<uint64_t>(dsfHeader[29]) << 16) |
                       (static_cast<uint64_t>(dsfHeader[30]) << 8) |
                       static_cast<uint64_t>(dsfHeader[31]);

    header.channels = (static_cast<uint16_t>(dsfHeader[32]) << 8) |
                     static_cast<uint16_t>(dsfHeader[33]);

    header.bitDepth = (static_cast<uint16_t>(dsfHeader[36]) << 8) |
                     static_cast<uint16_t>(dsfHeader[37]);

    header.blockSizePerChannel = (static_cast<uint32_t>(dsfHeader[48]) << 24) |
                                   (static_cast<uint32_t>(dsfHeader[49]) << 16) |
                                   (static_cast<uint32_t>(dsfHeader[50]) << 8) |
                                   static_cast<uint32_t>(dsfHeader[51]);

    // Calculate total samples
    header.totalSamples = (static_cast<uint64_t>(dsfHeader[40]) << 56) |
                          (static_cast<uint64_t>(dsfHeader[41]) << 48) |
                          (static_cast<uint64_t>(dsfHeader[42]) << 40) |
                          (static_cast<uint64_t>(dsfHeader[43]) << 32) |
                          (static_cast<uint64_t>(dsfHeader[44]) << 24) |
                          (static_cast<uint64_t>(dsfHeader[45]) << 16) |
                          (static_cast<uint64_t>(dsfHeader[46]) << 8) |
                          static_cast<uint64_t>(dsfHeader[47]);

    // Find data chunk
    file.seekg(52);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::end);
    fileSize = file.tellg();

    // Parse DSF chunks to find data chunk
    while (file.tellg() < static_cast<std::streamoff>(fileSize)) {
        uint32_t chunkSize;
        char chunkId[5] = {0};

        file.read(reinterpret_cast<char*>(chunkId), 4);
        file.read(reinterpret_cast<char*>(&chunkSize), 4);

        if (file.eof()) {
            break;
        }

        chunkSize = ntohl(chunkSize);

        if (std::strcmp(chunkId, "data") == 0) {
            // Found data chunk
            header.dataOffset = file.tellg();
            header.dataSize = chunkSize;
            break;
        } else {
            // Skip unknown chunks
            file.seekg(chunkSize, std::ios::cur);
        }
    }

    Logger::debug("DSF Header: {} Hz, {} channels, {} bit depth, {} total samples, {}x{} bytes/chunk",
                 header.sampleRate, header.channels, header.bitDepth,
                 header.totalSamples, header.blockSizePerChannel, header.blockSizePerChannel);

    return header.dataSize > 0;
}

std::string DSFDecoder::createTempDSDFile(const std::vector<uint8_t>& dsdData, const DSFHeader& header) {
    // Create a temporary file path
    std::string tempDir = std::filesystem::temp_directory_path().string();
    std::string tempFile = tempDir + "/vortex_dsd_temp_" +
                          std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) +
                          ".dsd";

    // Write DSF header
    std::ofstream tempStream(tempFile, std::ios::binary);
    if (!tempStream.is_open()) {
        throw std::runtime_error("Failed to create temporary DSF file");
    }

    // Write standard DSF header
    tempStream.write("DSD ", 4);  // Signature
    tempStream.put(0x00); tempStream.put(0x01);  // Version 1.0

    // Skip reserved bytes (12 bytes)
    tempStream.write("\0\0\0\0\0\0\0\0\0\0\0\0", 12);

    // File size (header + data)
    uint64_t fileSize = 52 + dsdData.size();
    tempStream.put(static_cast<uint8_t>((fileSize >> 56) & 0xFF));
    tempStream.put(static_cast<uint8_t>((fileSize >> 48) & 0xFF));
    tempStream.put(static_cast<uint8_t>((fileSize >> 40) & 0xFF));
    tempStream.put(static_cast<uint8_t>((fileSize >> 32) & 0xFF));
    tempStream.put(static_cast<uint8_t>((fileSize >> 24) & 0xFF));
    tempStream.put(static_cast<uint8_t>((fileSize >> 16) & 0xFF));
    tempStream.put(static_cast<uint8_t>((fileSize >> 8) & 0xFF));
    tempStream.put(static_cast<uint8_t>(fileSize & 0xFF));

    // Total file size (same as file size)
    tempStream.put(static_cast<uint8_t>((fileSize >> 56) & 0xFF));
    tempStream.put(static_cast<uint8_t>((fileSize >> 48) & 0xFF));
    tempStream.put(static_cast<uint8_t>((fileSize >> 40) & 0xFF));
    tempStream.put(static_cast<uint8_t>((fileSize >> 32) & 0xFF));
    tempStream.put(static_cast<uint8_t>((fileSize >> 24) & 0xFF));
    tempStream.put(static_cast<uint8_t>((fileSize >> 16) & 0xFF));
    tempStream.put(static_cast<uint8_t>((fileSize >> 8) & 0xFF));
    tempStream.put(static_cast<uint8_t>(fileSize & 0xFF));

    // Sample rate
    tempStream.put(static_cast<uint8_t>((header.sampleRate >> 24) & 0xFF));
    tempStream.put(static_cast<uint8_t>((header.sampleRate >> 16) & 0xFF));
    tempStream.put(static_cast<uint8_t>((header.sampleRate >> 8) & 0xFF));
    tempStream.put(static_cast<uint8_t>(header.sampleRate & 0xFF));

    // Channels
    tempStream.put(static_cast<uint8_t>((header.channels >> 8) & 0xFF));
    tempStream.put(static_cast<uint8_t>(header.channels & 0xFF));

    // Bit depth
    tempStream.put(static_cast<uint8_t>((header.bitDepth >> 8) & 0xFF));
    tempStream.put(static_cast<uint8_t>(header.bitDepth & 0xFF));

    // Channel mask (all channels)
    tempStream.put(0x01);
    if (header.channels >= 2) tempStream.put(0x01);
    if (header.channels >= 3) tempStream.put(0x01);
    tempStream.put(0x00); tempStream.put(0x00);
    tempStream.put(0x00); tempStream.put(0x00);

    // Reserved (4 bytes)
    tempStream.write("\0\0\0\0", 4);

    // Block size per channel
    tempStream.put(static_cast<uint8_t>((header.blockSizePerChannel >> 24) & 0xFF));
    tempStream.put(static_cast<uint8_t>((header.blockSizePerChannel >> 16) & 0xFF));
    tempStream.put(static_cast<uint8_t>((header.blockSizePerChannel >> 8) & 0xFF));
    tempStream.put(static_cast<uint8_t>(header.blockSizePerChannel & 0xFF));

    // Reserved (4 bytes)
    tempStream.write("\0\0\0\0", 4);

    // Total samples
    tempStream.put(static_cast<uint8_t>((header.totalSamples >> 56) & 0xFF));
    tempStream.put(static_cast<uint8_t>((header.totalSamples >> 48) & 0xFF));
    tempStream.put(static_cast<uint8_t>((header.totalSamples >> 40) & 0xFF));
    tempStream.put(static_cast<uint8_t>((header.totalSamples >> 32) & 0xFF));
    tempStream.put(static_cast<uint8_t>((header.totalSamples >> 24) & 0xFF));
    tempStream.put(static_cast<uint8_t>((header.totalSamples >> 16) & 0xFF));
    tempStream.put(static_cast<uint8_t>((header.totalSamples >> 8) & 0xFF));
    tempStream.put(static_cast<uint8_t>(header.totalSamples & 0xFF));

    // Write ID3v2 tag if present (skip for now, just write placeholder)
    tempStream.write("ID3 ", 4);  // ID3 chunk signature
    tempStream.put(0x00); tempStream.put(0x00); tempStream.put(0x00); tempStream.put(0x00);
    tempStream.put(0x00); tempStream.put(0x00); tempStream.put(0x00); tempStream.put(0x00);

    // Skip ID3 size (8 bytes) and data (0 bytes for now)
    tempStream.write("\0\0\0\0\0\0\0\0\0", 8);

    // Write data chunk
    tempStream.write("data", 4);  // Data chunk signature
    tempStream.put(static_cast<uint8_t>((dsdData.size() >> 24) & 0xFF));
    tempStream.put(static_cast<uint8_t>((dsdData.size() >> 16) & 0xFF));
    tempStream.put(static_cast<uint8_t>((dsdData.size() >> 8) & 0xFF));
    tempStream.put(static_cast<uint8_t>(dsdData.size() & 0xFF));

    // Write actual DSD data
    tempStream.write(reinterpret_cast<const char*>(dsdData.data()), dsdData.size());

    tempStream.close();
    return tempFile;
}

std::string DSFDecoder::getDSDFormatName(double sampleRate) const {
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

void DSFDecoder::extractDSFMetadata(std::ifstream& file, AudioMetadata& metadata) {
    // DSF format has limited metadata compared to ID3
    // Look for ID3v2 tag if present
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Look for ID3v2 tag (if present)
    while (file.tellg() < static_cast<std::streamoff>(fileSize)) {
        uint32_t chunkSize;
        char chunkId[5] = {0};

        file.read(reinterpret_cast<char*>(chunkId), 4);
        file.read(reinterpret_cast<char*>(&chunkSize), 4);

        if (file.eof()) {
            break;
        }

        chunkSize = ntohl(chunkSize);

        if (std::strcmp(chunkId, "ID3 ") == 0) {
            // Found ID3v2 tag
            size_t id3Size = chunkSize - 8;
            if (id3Size > 0 && id3Size < fileSize - (file.tellg() - 8)) {
                std::vector<uint8_t> id3Data(id3Size);
                file.read(reinterpret_cast<char*>(id3Data.data()), id3Size);

                // Extract ID3v2 metadata
                extractID3v2Metadata(id3Data.data(), id3Size, metadata);
            }
            break;
        } else {
            // Skip other chunks
            file.seekg(chunkSize, std::ios::cur);
        }
    }
}

void DSFDecoder::extractID3v2Metadata(const uint8_t* data, size_t size, AudioMetadata& metadata) {
    if (size < 10) {
        return;
    }

    // Parse ID3v2 frames
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