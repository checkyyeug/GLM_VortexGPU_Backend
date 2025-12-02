#include "binary_protocol.hpp"
#include "system/logger.hpp"

#include <zlib.h>
#include <cstring>
#include <chrono>

namespace vortex {

BinaryProtocol::BinaryProtocol()
    : maxMessageSize_(16 * 1024 * 1024) // 16MB
    , compressionEnabled_(true)
    , compressionLevel_(6)
    , checksumEnabled_(true)
{
    Logger::info("BinaryProtocol initialized");
}

BinaryProtocol::~BinaryProtocol() {
    Logger::info("BinaryProtocol destroyed");
}

std::vector<uint8_t> BinaryProtocol::serializeMessage(const network_message::NetworkMessage& message) {
    try {
        // Convert protocol buffer to bytes
        size_t size = message.ByteSizeLong();
        std::vector<uint8_t> protobufData(size);
        if (!message.SerializeToArray(protobufData.data(), size)) {
            Logger::error("Failed to serialize protocol buffer message");
            return {};
        }

        // Apply compression if enabled
        std::vector<uint8_t> payloadData;
        if (compressionEnabled_) {
            payloadData = compressData(protobufData);
        } else {
            payloadData = std::move(protobufData);
        }

        // Create binary message
        BinaryMessage binaryMessage;
        binaryMessage.header.version = PROTOCOL_VERSION;
        binaryMessage.header.type = static_cast<uint16_t>(message.message_type());
        binaryMessage.header.flags = 0;

        if (compressionEnabled_) {
            binaryMessage.header.flags |= FLAG_COMPRESSED;
        }

        binaryMessage.header.payloadSize = static_cast<uint32_t>(payloadData.size());
        binaryMessage.header.timestamp = getCurrentTimestamp();
        binaryMessage.payload = std::move(payloadData);

        // Generate checksum if enabled
        if (checksumEnabled_) {
            binaryMessage.header.checksum = calculateChecksum(binaryMessage);
            binaryMessage.header.flags |= FLAG_CHECKSUM;
        }

        // Serialize binary message
        return serializeBinaryMessage(binaryMessage);

    } catch (const std::exception& e) {
        Logger::error("Message serialization failed: {}", e.what());
        return {};
    }
}

std::optional<network_message::NetworkMessage> BinaryProtocol::deserializeMessage(const std::vector<uint8_t>& data) {
    try {
        if (data.empty()) {
            Logger::error("Empty data provided for deserialization");
            return std::nullopt;
        }

        // Deserialize binary message
        BinaryMessage binaryMessage = deserializeBinaryMessage(data);

        // Validate header
        if (!validateHeader(binaryMessage.header)) {
            Logger::error("Invalid message header");
            return std::nullopt;
        }

        // Verify checksum if present
        if ((binaryMessage.header.flags & FLAG_CHECKSUM) && checksumEnabled_) {
            uint32_t calculatedChecksum = calculateChecksum(binaryMessage);
            if (calculatedChecksum != binaryMessage.header.checksum) {
                Logger::error("Checksum validation failed");
                return std::nullopt;
            }
        }

        // Decompress payload if needed
        std::vector<uint8_t> protobufData;
        if (binaryMessage.header.flags & FLAG_COMPRESSED) {
            protobufData = decompressData(binaryMessage.payload);
        } else {
            protobufData = binaryMessage.payload;
        }

        // Parse protocol buffer
        network_message::NetworkMessage message;
        if (!message.ParseFromArray(protobufData.data(), protobufData.size())) {
            Logger::error("Failed to parse protocol buffer message");
            return std::nullopt;
        }

        return message;

    } catch (const std::exception& e) {
        Logger::error("Message deserialization failed: {}", e.what());
        return std::nullopt;
    }
}

std::vector<NetworkChunk> BinaryProtocol::createChunks(const std::vector<uint8_t>& data, size_t chunkSize) {
    std::vector<NetworkChunk> chunks;

    if (data.empty()) {
        Logger::warning("Empty data provided for chunking");
        return chunks;
    }

    size_t totalChunks = (data.size() + chunkSize - 1) / chunkSize;
    chunks.reserve(totalChunks);

    Logger::info("Creating {} chunks from {} bytes of data", totalChunks, data.size());

    for (size_t i = 0; i < totalChunks; ++i) {
        NetworkChunk chunk;
        chunk.chunkId = 1; // Can be made configurable
        chunk.totalChunks = static_cast<uint32_t>(totalChunks);
        chunk.sequenceNumber = static_cast<uint32_t>(i);

        size_t offset = i * chunkSize;
        size_t chunkDataSize = std::min(chunkSize, data.size() - offset);

        chunk.data.resize(chunkDataSize);
        std::copy(data.begin() + offset,
                  data.begin() + offset + chunkDataSize,
                  chunk.data.begin());

        // Calculate chunk checksum
        if (checksumEnabled_) {
            chunk.checksum = calculateCRC32(chunk.data);
        }

        chunks.push_back(std::move(chunk));
    }

    Logger::info("Created {} chunks successfully", chunks.size());
    return chunks;
}

std::vector<uint8_t> BinaryProtocol::reconstructChunks(const std::vector<NetworkChunk>& chunks) {
    if (chunks.empty()) {
        Logger::error("No chunks provided for reconstruction");
        return {};
    }

    try {
        // Validate all chunks have the same chunk ID and total count
        uint32_t chunkId = chunks[0].chunkId;
        uint32_t totalChunks = chunks[0].totalChunks;

        if (chunks.size() != totalChunks) {
            Logger::error("Incomplete chunk sequence: {}/{} chunks received",
                         chunks.size(), totalChunks);
            return {};
        }

        for (const auto& chunk : chunks) {
            if (chunk.chunkId != chunkId || chunk.totalChunks != totalChunks) {
                Logger::error("Inconsistent chunk metadata");
                return {};
            }
        }

        // Verify checksums
        if (checksumEnabled_) {
            for (const auto& chunk : chunks) {
                uint32_t calculatedChecksum = calculateCRC32(chunk.data);
                if (calculatedChecksum != chunk.checksum) {
                    Logger::error("Chunk checksum validation failed for chunk {}",
                                 chunk.sequenceNumber);
                    return {};
                }
            }
        }

        // Sort chunks by sequence number
        std::vector<NetworkChunk> sortedChunks = chunks;
        std::sort(sortedChunks.begin(), sortedChunks.end(),
                  [](const NetworkChunk& a, const NetworkChunk& b) {
                      return a.sequenceNumber < b.sequenceNumber;
                  });

        // Reconstruct data
        size_t totalSize = 0;
        for (const auto& chunk : sortedChunks) {
            totalSize += chunk.data.size();
        }

        std::vector<uint8_t> reconstructedData(totalSize);
        size_t offset = 0;

        for (const auto& chunk : sortedChunks) {
            std::copy(chunk.data.begin(), chunk.data.end(),
                      reconstructedData.begin() + offset);
            offset += chunk.data.size();
        }

        Logger::info("Successfully reconstructed {} bytes from {} chunks",
                     totalSize, chunks.size());
        return reconstructedData;

    } catch (const std::exception& e) {
        Logger::error("Chunk reconstruction failed: {}", e.what());
        return {};
    }
}

bool BinaryProtocol::validateProcessingChain(const ProcessingChain& chain) const {
    try {
        // Basic validation
        if (chain.steps_size() == 0) {
            Logger::error("Processing chain has no steps");
            return false;
        }

        if (chain.name().empty()) {
            Logger::error("Processing chain has no name");
            return false;
        }

        // Validate each step
        std::set<int> usedOrders;
        for (const auto& step : chain.steps()) {
            // Check step ID
            if (step.id().empty()) {
                Logger::error("Processing step has no ID");
                return false;
            }

            // Check step type
            if (step.step_type() == 0) {
                Logger::error("Processing step has invalid type");
                return false;
            }

            // Check order uniqueness
            if (usedOrders.count(step.order()) > 0) {
                Logger::error("Duplicate order {} in processing chain", step.order());
                return false;
            }
            usedOrders.insert(step.order());

            // Validate step parameters based on type
            if (!validateStepParameters(step)) {
                return false;
            }
        }

        Logger::debug("Processing chain validation passed: {}", chain.name());
        return true;

    } catch (const std::exception& e) {
        Logger::error("Processing chain validation failed: {}", e.what());
        return false;
    }
}

bool BinaryProtocol::validateStepParameters(const ProcessingStep& step) const {
    switch (step.step_type()) {
        case ProcessingType::EQUALIZER: {
            // Validate equalizer parameters
            auto it = step.parameters().find("bands");
            if (it == step.parameters().end()) {
                Logger::error("Equalizer step missing 'bands' parameter");
                return false;
            }

            int bands = std::stoi(it->second);
            if (bands < 1 || bands > 512) {
                Logger::error("Invalid number of equalizer bands: {}", bands);
                return false;
            }

            // Check frequency array
            auto freqIt = step.parameters().find("frequencies");
            if (freqIt == step.parameters().end()) {
                Logger::error("Equalizer step missing 'frequencies' parameter");
                return false;
            }

            // Check gain array
            auto gainIt = step.parameters().find("gains");
            if (gainIt == step.parameters().end()) {
                Logger::error("Equalizer step missing 'gains' parameter");
                return false;
            }

            break;
        }

        case ProcessingType::CONVOLUTION: {
            // Validate convolution parameters
            auto it = step.parameters().find("impulse_file");
            if (it == step.parameters().end() || it->second.empty()) {
                Logger::error("Convolution step missing or empty 'impulse_file' parameter");
                return false;
            }

            auto lengthIt = step.parameters().find("length");
            if (lengthIt != step.parameters().end()) {
                size_t length = std::stoull(lengthIt->second);
                if (length > 16777216) { // 16M points maximum
                    Logger::error("Convolution length exceeds maximum: {}", length);
                    return false;
                }
            }

            break;
        }

        case ProcessingType::GAIN: {
            // Validate gain parameters
            auto it = step.parameters().find("gain_db");
            if (it == step.parameters().end()) {
                Logger::error("Gain step missing 'gain_db' parameter");
                return false;
            }

            float gain = std::stof(it->second);
            if (gain < -60.0f || gain > 24.0f) {
                Logger::error("Gain value out of range: {}", gain);
                return false;
            }

            break;
        }

        case ProcessingType::RESAMPLER: {
            // Validate resampler parameters
            auto rateIt = step.parameters().find("output_rate");
            if (rateIt == step.parameters().end()) {
                Logger::error("Resampler step missing 'output_rate' parameter");
                return false;
            }

            int outputRate = std::stoi(rateIt->second);
            if (outputRate < 8000 || outputRate > 768000) {
                Logger::error("Invalid output sample rate: {}", outputRate);
                return false;
            }

            break;
        }

        default:
            // Unknown step types are allowed for extensibility
            Logger::debug("Unknown processing step type: {}", step.step_type());
            break;
    }

    return true;
}

std::vector<uint8_t> BinaryProtocol::serializeBinaryMessage(const BinaryMessage& message) {
    size_t totalSize = sizeof(MessageHeader) + message.payload.size();
    std::vector<uint8_t> data(totalSize);

    size_t offset = 0;

    // Serialize header
    memcpy(&data[offset], &message.header.version, sizeof(message.header.version));
    offset += sizeof(message.header.version);

    memcpy(&data[offset], &message.header.type, sizeof(message.header.type));
    offset += sizeof(message.header.type);

    memcpy(&data[offset], &message.header.flags, sizeof(message.header.flags));
    offset += sizeof(message.header.flags);

    memcpy(&data[offset], &message.header.payloadSize, sizeof(message.header.payloadSize));
    offset += sizeof(message.header.payloadSize);

    memcpy(&data[offset], &message.header.timestamp, sizeof(message.header.timestamp));
    offset += sizeof(message.header.timestamp);

    memcpy(&data[offset], &message.header.checksum, sizeof(message.header.checksum));
    offset += sizeof(message.header.checksum);

    // Serialize payload
    if (!message.payload.empty()) {
        memcpy(&data[offset], message.payload.data(), message.payload.size());
    }

    return data;
}

BinaryMessage BinaryProtocol::deserializeBinaryMessage(const std::vector<uint8_t>& data) {
    BinaryMessage message;

    if (data.size() < sizeof(MessageHeader)) {
        throw std::runtime_error("Insufficient data for message header");
    }

    size_t offset = 0;

    // Deserialize header
    memcpy(&message.header.version, &data[offset], sizeof(message.header.version));
    offset += sizeof(message.header.version);

    memcpy(&message.header.type, &data[offset], sizeof(message.header.type));
    offset += sizeof(message.header.type);

    memcpy(&message.header.flags, &data[offset], sizeof(message.header.flags));
    offset += sizeof(message.header.flags);

    memcpy(&message.header.payloadSize, &data[offset], sizeof(message.header.payloadSize));
    offset += sizeof(message.header.payloadSize);

    memcpy(&message.header.timestamp, &data[offset], sizeof(message.header.timestamp));
    offset += sizeof(message.header.timestamp);

    memcpy(&message.header.checksum, &data[offset], sizeof(message.header.checksum));
    offset += sizeof(message.header.checksum);

    // Deserialize payload
    if (message.header.payloadSize > 0) {
        if (data.size() < sizeof(MessageHeader) + message.header.payloadSize) {
            throw std::runtime_error("Insufficient data for message payload");
        }

        message.payload.resize(message.header.payloadSize);
        memcpy(message.payload.data(), &data[offset], message.header.payloadSize);
    }

    return message;
}

bool BinaryProtocol::validateHeader(const MessageHeader& header) const {
    // Check version compatibility
    if (header.version != PROTOCOL_VERSION) {
        Logger::error("Protocol version mismatch: expected {}, got {}",
                     PROTOCOL_VERSION, header.version);
        return false;
    }

    // Check message type
    if (header.type == 0) {
        Logger::error("Invalid message type: 0");
        return false;
    }

    // Check payload size limits
    if (header.payloadSize > maxMessageSize_) {
        Logger::error("Payload size exceeds maximum: {} > {}",
                     header.payloadSize, maxMessageSize_);
        return false;
    }

    // Check timestamp (not too old, not too far in future)
    uint64_t currentTime = getCurrentTimestamp();
    const uint64_t MAX_TIME_DIFF = 3600 * 1000000; // 1 hour in microseconds

    if (header.timestamp > currentTime + MAX_TIME_DIFF ||
        (currentTime > header.timestamp && currentTime - header.timestamp > MAX_TIME_DIFF)) {
        Logger::error("Message timestamp out of acceptable range");
        return false;
    }

    return true;
}

uint32_t BinaryProtocol::calculateChecksum(const BinaryMessage& message) {
    // Simple CRC32 checksum over header and payload
    uint32_t checksum = 0xFFFFFFFF;

    // Include header fields (except checksum field itself)
    const uint8_t* headerData = reinterpret_cast<const uint8_t*>(&message.header);
    size_t headerSize = offsetof(MessageHeader, checksum);

    for (size_t i = 0; i < headerSize; ++i) {
        checksum = updateCRC32(checksum, headerData[i]);
    }

    // Include payload
    for (uint8_t byte : message.payload) {
        checksum = updateCRC32(checksum, byte);
    }

    return ~checksum; // Final CRC32 value
}

uint32_t BinaryProtocol::calculateCRC32(const std::vector<uint8_t>& data) {
    uint32_t crc = 0xFFFFFFFF;
    for (uint8_t byte : data) {
        crc = updateCRC32(crc, byte);
    }
    return ~crc;
}

uint32_t BinaryProtocol::updateCRC32(uint32_t crc, uint8_t byte) {
    static const uint32_t crcTable[256] = {
        // Pre-computed CRC32 table (would be fully populated in production)
        0x00000000, 0x77073096, 0xee0e612c, 0x990951ba,
        // ... (full table would be included)
    };

    return (crc >> 8) ^ crcTable[(crc ^ byte) & 0xFF];
}

std::vector<uint8_t> BinaryProtocol::compressData(const std::vector<uint8_t>& data) {
    if (data.empty()) {
        return {};
    }

    uLongf compressedSize = compressBound(data.size());
    std::vector<uint8_t> compressed(compressedSize);

    int result = compress2(compressed.data(), &compressedSize,
                           data.data(), data.size(), compressionLevel_);

    if (result != Z_OK) {
        Logger::error("Compression failed with error: {}", result);
        return data; // Return uncompressed data on failure
    }

    compressed.resize(compressedSize);
    Logger::debug("Compressed data: {} -> {} bytes", data.size(), compressedSize);
    return compressed;
}

std::vector<uint8_t> BinaryProtocol::decompressData(const std::vector<uint8_t>& compressed) {
    if (compressed.empty()) {
        return {};
    }

    // Start with a buffer 4x the compressed size
    size_t decompressedSize = compressed.size() * 4;
    std::vector<uint8_t> decompressed(decompressedSize);

    int result;
    do {
        if (decompressedSize > maxMessageSize_) {
            Logger::error("Decompressed data exceeds maximum size");
            return {};
        }

        result = uncompress(decompressed.data(), &decompressedSize,
                           compressed.data(), compressed.size());

        if (result == Z_BUF_ERROR) {
            // Buffer too small, double it and try again
            decompressedSize *= 2;
            decompressed.resize(decompressedSize);
        } else if (result != Z_OK) {
            Logger::error("Decompression failed with error: {}", result);
            return {};
        }
    } while (result == Z_BUF_ERROR);

    decompressed.resize(decompressedSize);
    Logger::debug("Decompressed data: {} -> {} bytes", compressed.size(), decompressedSize);
    return decompressed;
}

uint64_t BinaryProtocol::getCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

// Configuration methods
void BinaryProtocol::setMaxMessageSize(size_t size) { maxMessageSize_ = size; }
void BinaryProtocol::setCompressionEnabled(bool enabled) { compressionEnabled_ = enabled; }
void BinaryProtocol::setCompressionLevel(int level) { compressionLevel_ = level; }
void BinaryProtocol::setChecksumEnabled(bool enabled) { checksumEnabled_ = enabled; }

// Getters
size_t BinaryProtocol::getMaxMessageSize() const { return maxMessageSize_; }
bool BinaryProtocol::isCompressionEnabled() const { return compressionEnabled_; }
int BinaryProtocol::getCompressionLevel() const { return compressionLevel_; }
bool BinaryProtocol::isChecksumEnabled() const { return checksumEnabled_; }

} // namespace vortex