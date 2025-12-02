#pragma once

#include <vector>
#include <optional>
#include <string>
#include <set>
#include <cstdint>
#include <zlib.h>

#include "vortex_api.hpp"

namespace vortex {

constexpr uint8_t PROTOCOL_VERSION = 1;
constexpr uint16_t FLAG_COMPRESSED = 0x0001;
constexpr uint16_t FLAG_CHECKSUM = 0x0002;
constexpr uint16_t FLAG_ENCRYPTED = 0x0004;

#pragma pack(push, 1)
struct MessageHeader {
    uint8_t version;
    uint16_t type;
    uint16_t flags;
    uint32_t payloadSize;
    uint64_t timestamp;
    uint32_t checksum;
};
#pragma pack(pop)

struct BinaryMessage {
    MessageHeader header;
    std::vector<uint8_t> payload;
};

struct NetworkChunk {
    uint32_t chunkId;
    uint32_t totalChunks;
    uint32_t sequenceNumber;
    std::vector<uint8_t> data;
    uint32_t checksum;
};

/**
 * @brief Binary protocol for high-performance audio network communication
 *
 * This class implements a binary protocol for efficient network communication
 * of audio data and control messages. It features compression, chunking,
 * checksums, and validation optimized for real-time audio applications.
 */
class BinaryProtocol {
public:
    BinaryProtocol();
    ~BinaryProtocol();

    // Message serialization
    std::vector<uint8_t> serializeMessage(const network_message::NetworkMessage& message);
    std::optional<network_message::NetworkMessage> deserializeMessage(const std::vector<uint8_t>& data);

    // Chunking for large messages
    std::vector<NetworkChunk> createChunks(const std::vector<uint8_t>& data, size_t chunkSize);
    std::vector<uint8_t> reconstructChunks(const std::vector<NetworkChunk>& chunks);

    // Validation
    bool validateProcessingChain(const ProcessingChain& chain) const;

    // Configuration
    void setMaxMessageSize(size_t size);
    void setCompressionEnabled(bool enabled);
    void setCompressionLevel(int level);
    void setChecksumEnabled(bool enabled);

    // Getters
    size_t getMaxMessageSize() const;
    bool isCompressionEnabled() const;
    int getCompressionLevel() const;
    bool isChecksumEnabled() const;

private:
    // Configuration
    size_t maxMessageSize_;
    bool compressionEnabled_;
    int compressionLevel_;
    bool checksumEnabled_;

    // Internal serialization
    std::vector<uint8_t> serializeBinaryMessage(const BinaryMessage& message);
    BinaryMessage deserializeBinaryMessage(const std::vector<uint8_t>& data);

    // Validation
    bool validateHeader(const MessageHeader& header) const;
    bool validateStepParameters(const ProcessingStep& step) const;

    // Checksum calculation
    uint32_t calculateChecksum(const BinaryMessage& message);
    uint32_t calculateCRC32(const std::vector<uint8_t>& data);
    uint32_t updateCRC32(uint32_t crc, uint8_t byte);

    // Compression
    std::vector<uint8_t> compressData(const std::vector<uint8_t>& data);
    std::vector<uint8_t> decompressData(const std::vector<uint8_t>& compressed);

    // Utility
    uint64_t getCurrentTimestamp() const;
};

} // namespace vortex