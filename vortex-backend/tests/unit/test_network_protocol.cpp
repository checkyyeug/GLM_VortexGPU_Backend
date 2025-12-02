#include <gtest/gtest.h>
#include "network/protocol/binary_protocol.hpp"
#include "system/logger.hpp"

#include <vector>
#include <string>
#include <random>
#include <chrono>

using namespace vortex;
using namespace std::chrono_literals;

class NetworkProtocolTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::initialize();
        protocol_ = std::make_unique<BinaryProtocol>();
    }

    void TearDown() override {
        Logger::shutdown();
    }

    // Helper function to generate test audio data
    std::vector<uint8_t> generateAudioData(size_t numSamples, int sampleRate = 44100) {
        std::vector<uint8_t> audioData(numSamples * sizeof(int16_t));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int16_t> dist(-32768, 32767);

        for (size_t i = 0; i < numSamples; ++i) {
            int16_t sample = dist(gen);
            std::memcpy(&audioData[i * sizeof(int16_t)], &sample, sizeof(int16_t));
        }

        return audioData;
    }

    // Helper function to create test network message
    network_message::NetworkMessage createTestAudioMessage(const std::string& filename) {
        network_message::NetworkMessage message;
        message.set_message_type(network_message::message_type::AUDIO_UPLOAD);
        message.set_channel("upload");

        // Create audio metadata
        auto* metadata = message.mutable_audio_metadata();
        metadata->set_title("Test Audio");
        metadata->set_artist("Test Artist");
        metadata->set_album("Test Album");
        metadata->set_format(network_message::AudioFormat::WAV);
        metadata->set_sample_rate(44100);
        metadata->set_channels(2);
        metadata->set_bit_depth(16);
        metadata->set_file_name(filename);

        // Create audio data
        auto audioData = generateAudioData(44100); // 1 second of audio
        auto* audio_content = message.mutable_audio_content();
        audio_content->set_data(std::string(reinterpret_cast<char*>(audioData.data()), audioData.size()));

        return message;
    }

    // Helper function to create test spectrum data
    network_message::NetworkMessage createTestSpectrumMessage() {
        network_message::NetworkMessage message;
        message.set_message_type(network_message::message_type::SPECTRUM_DATA);
        message.set_channel("visualization");

        auto* spectrum = message.mutable_spectrum_data();
        spectrum->set_sample_rate(44100);
        spectrum->set_fft_size(2048);

        // Generate frequency bins (0-20kHz logarithmic)
        for (int i = 0; i < 1024; ++i) {
            double freq = 20.0 * std::pow(20000.0 / 20.0, i / 1023.0);
            spectrum->add_frequency_bins(freq);
            spectrum->add_magnitudes(std::exp(-i / 100.0)); // Exponential decay
        }

        spectrum->set_timestamp(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

        return message;
    }

    // Helper function to create test processing chain
    network_message::NetworkMessage createTestProcessingChain() {
        network_message::NetworkMessage message;
        message.set_message_type(network_message::message_type::PROCESSING_CHAIN_UPDATE);
        message.set_channel("processing");

        auto* chain = message.mutable_processing_chain();
        chain->set_id("test_chain_001");
        chain->set_name("Test Processing Chain");
        chain->set_enabled(true);

        // Add equalizer step
        auto* eq_step = chain->add_steps();
        eq_step->set_id("eq_step_001");
        eq_step->set_type(network_message::ProcessingType::EQUALIZER);
        eq_step->set_enabled(true);
        eq_step->set_order(0);

        (*eq_step->mutable_parameters())["bands"] = "10";
        (*eq_step->mutable_parameters())["frequencies"] = "[31.5,63,125,250,500,1000,2000,4000,8000,16000]";
        (*eq_step->mutable_parameters())["gains"] = "[0.0,1.0,-1.0,2.0,-2.0,0.0,1.0,-1.0,2.0,-2.0]";

        // Add convolution step
        auto* conv_step = chain->add_steps();
        conv_step->set_id("conv_step_001");
        conv_step->set_type(network_message::ProcessingType::CONVOLUTION);
        conv_step->set_enabled(true);
        conv_step->set_order(1);

        (*conv_step->mutable_parameters())["impulse_file"] = "/test/ir.wav";
        (*conv_step->mutable_parameters())["length"] = "1048576"; // 1M points

        return message;
    }

    std::unique_ptr<BinaryProtocol> protocol_;
};

// Test basic message serialization and deserialization
TEST_F(NetworkProtocolTest, BasicMessageSerialization) {
    // Create test message
    network_message::NetworkMessage originalMessage = createTestAudioMessage("test.wav");

    // Serialize
    std::vector<uint8_t> serializedData = protocol_->serializeMessage(originalMessage);
    ASSERT_FALSE(serializedData.empty());

    // Deserialize
    auto deserializedMessage = protocol_->deserializeMessage(serializedData);
    ASSERT_TRUE(deserializedMessage.has_value());

    // Verify message content
    const auto& deserialized = deserializedMessage.value();
    EXPECT_EQ(deserialized.message_type(), network_message::message_type::AUDIO_UPLOAD);
    EXPECT_EQ(deserialized.channel(), "upload");
    EXPECT_TRUE(deserialized.has_audio_metadata());
    EXPECT_TRUE(deserialized.has_audio_content());

    const auto& metadata = deserialized.audio_metadata();
    EXPECT_EQ(metadata.title(), "Test Audio");
    EXPECT_EQ(metadata.artist(), "Test Artist");
    EXPECT_EQ(metadata.format(), network_message::AudioFormat::WAV);
    EXPECT_EQ(metadata.sample_rate(), 44100);
    EXPECT_EQ(metadata.channels(), 2);
    EXPECT_EQ(metadata.bit_depth(), 16);

    const auto& content = deserialized.audio_content();
    EXPECT_EQ(content.data().size(), 44100 * sizeof(int16_t));
}

// Test message compression
TEST_F(NetworkProtocolTest, MessageCompression) {
    // Enable compression
    protocol_->setCompressionEnabled(true);
    protocol_->setCompressionLevel(6);

    network_message::NetworkMessage message = createTestSpectrumMessage();

    // Serialize with compression
    std::vector<uint8_t> compressedData = protocol_->serializeMessage(message);

    // Disable compression
    protocol_->setCompressionEnabled(false);

    // Serialize without compression
    std::vector<uint8_t> uncompressedData = protocol_->serializeMessage(message);

    // Compressed data should be smaller for this spectrum message
    EXPECT_LT(compressedData.size(), uncompressedData.size());

    // Both should deserialize correctly
    auto compressedResult = protocol_->deserializeMessage(compressedData);
    auto uncompressedResult = protocol_->deserializeMessage(uncompressedData);

    ASSERT_TRUE(compressedResult.has_value());
    ASSERT_TRUE(uncompressedResult.has_value());

    // Results should be identical
    const auto& compressedMsg = compressedResult.value();
    const auto& uncompressedMsg = uncompressedResult.value();

    EXPECT_EQ(compressedMsg.message_type(), uncompressedMsg.message_type());
    EXPECT_EQ(compressedMsg.channel(), uncompressedMsg.channel());

    ASSERT_TRUE(compressedMsg.has_spectrum_data());
    ASSERT_TRUE(uncompressedMsg.has_spectrum_data());

    const auto& compressedSpec = compressedMsg.spectrum_data();
    const auto& uncompressedSpec = uncompressedMsg.spectrum_data();

    EXPECT_EQ(compressedSpec.sample_rate(), uncompressedSpec.sample_rate());
    EXPECT_EQ(compressedSpec.fft_size(), uncompressedSpec.fft_size());
    EXPECT_EQ(compressedSpec.frequency_bins_size(), uncompressedSpec.frequency_bins_size());
    EXPECT_EQ(compressedSpec.magnitudes_size(), uncompressedSpec.magnitudes_size());

    std::cout << "Uncompressed size: " << uncompressedData.size() << " bytes" << std::endl;
    std::cout << "Compressed size: " << compressedData.size() << " bytes" << std::endl;
    std::cout << "Compression ratio: " << (static_cast<double>(uncompressedData.size()) / compressedData.size()) << "x" << std::endl;
}

// Test message chunking
TEST_F(NetworkProtocolTest, MessageChunking) {
    // Create large message
    network_message::NetworkMessage message = createTestAudioMessage("large_test.wav");

    // Add large audio content (1MB)
    auto largeAudioData = generateAudioData(262144); // ~1MB of audio
    auto* audio_content = message.mutable_audio_content();
    audio_content->set_data(std::string(reinterpret_cast<char*>(largeAudioData.data()), largeAudioData.size()));

    // Serialize
    std::vector<uint8_t> serializedData = protocol_->serializeMessage(message);
    EXPECT_GT(serializedData.size(), 1024 * 1024); // Should be > 1MB

    // Create chunks
    size_t chunkSize = 65536; // 64KB chunks
    auto chunks = protocol_->createChunks(serializedData, chunkSize);

    EXPECT_GT(chunks.size(), 1); // Should create multiple chunks

    // Verify chunk properties
    uint32_t expectedTotalChunks = (serializedData.size() + chunkSize - 1) / chunkSize;
    EXPECT_EQ(chunks[0].total_chunks, expectedTotalChunks);

    // Verify all chunks have unique sequence numbers
    std::set<uint32_t> sequenceNumbers;
    for (const auto& chunk : chunks) {
        EXPECT_EQ(chunk.total_chunks, expectedTotalChunks);
        EXPECT_EQ(chunk.chunk_id, chunks[0].chunk_id);
        EXPECT_FALSE(sequenceNumbers.contains(chunk.sequence_number));
        sequenceNumbers.insert(chunk.sequence_number);
        EXPECT_FALSE(chunk.data.empty());
    }

    // Verify checksums
    protocol_->setChecksumEnabled(true);
    auto newChunks = protocol_->createChunks(serializedData, chunkSize);

    for (const auto& chunk : newChunks) {
        EXPECT_GT(chunk.checksum, 0);
    }

    std::cout << "Total chunks: " << chunks.size() << std::endl;
    std::cout << "Chunk size: " << chunkSize << " bytes" << std::endl;
}

// Test chunk reconstruction
TEST_F(NetworkProtocolTest, ChunkReconstruction) {
    // Create test data
    std::vector<uint8_t> originalData;
    for (int i = 0; i < 100000; ++i) {
        originalData.push_back(static_cast<uint8_t>(i % 256));
    }

    // Create chunks
    size_t chunkSize = 8192;
    auto chunks = protocol_->createChunks(originalData, chunkSize);

    // Reconstruct
    std::vector<uint8_t> reconstructedData = protocol_->reconstructChunks(chunks);

    // Verify reconstruction
    EXPECT_EQ(reconstructedData.size(), originalData.size());
    EXPECT_EQ(reconstructedData, originalData);

    // Test with missing chunks (should fail)
    std::vector<NetworkChunk> incompleteChunks(chunks.begin(), chunks.end() - 1);
    auto incompleteResult = protocol_->reconstructChunks(incompleteChunks);
    EXPECT_TRUE(incompleteResult.empty());
}

// Test processing chain validation
TEST_F(NetworkProtocolTest, ProcessingChainValidation) {
    // Test valid processing chain
    network_message::NetworkMessage message = createTestProcessingChain();

    // Should validate successfully
    EXPECT_TRUE(protocol_->validateProcessingChain(message.processing_chain()));

    // Test invalid processing chain (empty name)
    network_message::NetworkMessage invalidMessage;
    invalidMessage.set_message_type(network_message::message_type::PROCESSING_CHAIN_UPDATE);
    auto* chain = invalidMessage.mutable_processing_chain();
    chain->set_id("invalid_chain");
    chain->set_name(""); // Empty name
    chain->set_enabled(true);

    EXPECT_FALSE(protocol_->validateProcessingChain(invalidMessage.processing_chain()));

    // Test invalid processing chain (no steps)
    chain->set_name("no_steps_chain");
    EXPECT_FALSE(protocol_->validateProcessingChain(invalidMessage.processing_chain()));

    // Test processing chain with invalid step
    auto* invalidStep = chain->add_steps();
    invalidStep->set_id("invalid_step");
    invalidStep->set_type(0); // Invalid type
    invalidStep->set_enabled(true);
    invalidStep->set_order(0);

    EXPECT_FALSE(protocol_->validateProcessingChain(invalidMessage.processing_chain()));
}

// Test processing chain parameter validation
TEST_F(NetworkProtocolTest, ProcessingChainParameterValidation) {
    // Test equalizer with invalid parameters
    network_message::NetworkMessage message;
    message.set_message_type(network_message::message_type::PROCESSING_CHAIN_UPDATE);
    auto* chain = message.mutable_processing_chain();
    chain->set_id("eq_test");
    chain->set_name("Equalizer Test");
    chain->set_enabled(true);

    auto* eqStep = chain->add_steps();
    eqStep->set_id("eq_step");
    eqStep->set_type(network_message::ProcessingType::EQUALIZER);
    eqStep->set_enabled(true);
    eqStep->set_order(0);

    // Test invalid number of bands
    (*eqStep->mutable_parameters())["bands"] = "0"; // Zero bands
    EXPECT_FALSE(protocol_->validateProcessingChain(*chain));

    // Test too many bands
    (*eqStep->mutable_parameters())["bands"] = "600"; // Too many bands
    EXPECT_FALSE(protocol_->validateProcessingChain(*chain));

    // Test valid number of bands
    (*eqStep->mutable_parameters())["bands"] = "10";
    (*eqStep->mutable_parameters())["frequencies"] = "[100,1000,10000]";
    (*eqStep->mutable_parameters())["gains"] = "[0.0,2.0,-1.0]";
    EXPECT_TRUE(protocol_->validateProcessingChain(*chain));

    // Test convolution with invalid parameters
    network_message::NetworkMessage convMessage;
    auto* convChain = convMessage.mutable_processing_chain();
    convChain->set_id("conv_test");
    convChain->set_name("Convolution Test");
    convChain->set_enabled(true);

    auto* convStep = convChain->add_steps();
    convStep->set_id("conv_step");
    convStep->set_type(network_message::ProcessingType::CONVOLUTION);
    convStep->set_enabled(true);
    convStep->set_order(0);

    // Test missing impulse file
    EXPECT_FALSE(protocol_->validateProcessingChain(*convChain));

    // Test valid impulse file
    (*convStep->mutable_parameters())["impulse_file"] = "/test/ir.wav";
    EXPECT_TRUE(protocol_->validateProcessingChain(*convChain));

    // Test too long impulse response
    (*convStep->mutable_parameters())["length"] = "20000000"; // >16M points
    EXPECT_FALSE(protocol_->validateProcessingChain(*convChain));
}

// Test performance with large messages
TEST_F(NetworkProtocolTest, PerformanceWithLargeMessages) {
    // Create large audio messages
    std::vector<size_t> messageSizes = {1024, 10240, 102400, 1024000, 10240000}; // 1KB to 10MB

    for (size_t messageSize : messageSizes) {
        network_message::NetworkMessage message;
        message.set_message_type(network_message::message_type::AUDIO_UPLOAD);
        message.set_channel("performance_test");

        // Add metadata
        auto* metadata = message.mutable_audio_metadata();
        metadata->set_title("Performance Test");
        metadata->set_sample_rate(44100);
        metadata->set_channels(2);
        metadata->set_bit_depth(16);

        // Add audio content
        auto audioData = generateAudioData(messageSize / sizeof(int16_t));
        auto* audio_content = message.mutable_audio_content();
        audio_content->set_data(std::string(reinterpret_cast<char*>(audioData.data()), audioData.size()));

        // Measure serialization time
        auto start = std::chrono::high_resolution_clock::now();
        auto serializedData = protocol_->serializeMessage(message);
        auto end = std::chrono::high_resolution_clock::now();

        auto serializeTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Measure deserialization time
        start = std::chrono::high_resolution_clock::now();
        auto deserializedMessage = protocol_->deserializeMessage(serializedData);
        end = std::chrono::high_resolution_clock::now();

        auto deserializeTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        ASSERT_TRUE(deserializedMessage.has_value());

        // Verify correctness
        const auto& result = deserializedMessage.value();
        EXPECT_EQ(result.message_type(), network_message::message_type::AUDIO_UPLOAD);
        EXPECT_EQ(result.audio_metadata().title(), "Performance Test");
        EXPECT_EQ(result.audio_content().data().size(), audioData.size());

        // Performance expectations
        EXPECT_LT(serializeTime.count(), 10000) << "Serialization too slow for " << messageSize << " bytes";
        EXPECT_LT(deserializeTime.count(), 5000) << "Deserialization too slow for " << messageSize << " bytes";

        std::cout << "Message size: " << messageSize
                  << ", Serialize: " << serializeTime.count() << "μs"
                  << ", Deserialize: " << deserializeTime.count() << "μs"
                  << ", Total: " << (serializeTime + deserializeTime).count() << "μs" << std::endl;
    }
}

// Test concurrent operations
TEST_F(NetworkProtocolTest, ConcurrentOperations) {
    const int numThreads = 8;
    const int operationsPerThread = 100;

    std::vector<std::thread> threads;
    std::vector<std::chrono::microseconds> serializationTimes;
    std::vector<std::chrono::microseconds> deserializationTimes;
    std::mutex timesMutex;

    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < operationsPerThread; ++i) {
                // Create unique message for each operation
                network_message::NetworkMessage message = createTestAudioMessage(
                    "concurrent_test_" + std::to_string(t) + "_" + std::to_string(i) + ".wav");

                // Serialize
                auto serializeStart = std::chrono::high_resolution_clock::now();
                auto serializedData = protocol_->serializeMessage(message);
                auto serializeEnd = std::chrono::high_resolution_clock::now();

                // Deserialize
                auto deserializeStart = std::chrono::high_resolution_clock::now();
                auto deserializedMessage = protocol_->deserializeMessage(serializedData);
                auto deserializeEnd = std::chrono::high_resolution_clock::now();

                // Verify success
                ASSERT_TRUE(deserializedMessage.has_value());

                // Record times
                auto serializeTime = std::chrono::duration_cast<std::chrono::microseconds>(
                    serializeEnd - serializeStart);
                auto deserializeTime = std::chrono::duration_cast<std::chrono::microseconds>(
                    deserializeEnd - deserializeStart);

                {
                    std::lock_guard<std::mutex> lock(timesMutex);
                    serializationTimes.push_back(serializeTime);
                    deserializationTimes.push_back(deserializeTime);
                }
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Analyze performance
    ASSERT_EQ(serializationTimes.size(), numThreads * operationsPerThread);
    ASSERT_EQ(deserializationTimes.size(), numThreads * operationsPerThread);

    auto [minSerialTime, maxSerialTime] = std::minmax_element(serializationTimes.begin(), serializationTimes.end());
    auto [minDeserialTime, maxDeserialTime] = std::minmax_element(deserializationTimes.begin(), deserializationTimes.end());

    double avgSerialTime = std::accumulate(serializationTimes.begin(), serializationTimes.end(), 0.0) / serializationTimes.size();
    double avgDeserialTime = std::accumulate(deserializationTimes.begin(), deserializationTimes.end(), 0.0) / deserializationTimes.size();

    std::cout << "Concurrent operations: " << (numThreads * operationsPerThread) << std::endl;
    std::cout << "Serialization - Min: " << minSerialTime->count() << "μs, Max: " << maxSerialTime->count()
              << "μs, Avg: " << avgSerialTime << "μs" << std::endl;
    std::cout << "Deserialization - Min: " << minDeserialTime->count() << "μs, Max: " << maxDeserialTime->count()
              << "μs, Avg: " << avgDeserialTime << "μs" << std::endl;

    // Check for reasonable performance consistency
    EXPECT_LT(maxSerialTime->count(), minSerialTime->count() * 10)
        << "Serialization performance too inconsistent";
    EXPECT_LT(maxDeserialTime->count(), minDeserialTime->count() * 10)
        << "Deserialization performance too inconsistent";
}

// Test error handling
TEST_F(NetworkProtocolTest, ErrorHandling) {
    // Test empty data
    auto emptyResult = protocol_->deserializeMessage({});
    EXPECT_FALSE(emptyResult.has_value());

    // Test invalid data (random bytes)
    std::vector<uint8_t> invalidData(1000);
    for (size_t i = 0; i < invalidData.size(); ++i) {
        invalidData[i] = static_cast<uint8_t>(rand() % 256);
    }

    auto invalidResult = protocol_->deserializeMessage(invalidData);
    EXPECT_FALSE(invalidResult.has_value());

    // Test corrupted message
    network_message::NetworkMessage validMessage = createTestAudioMessage("test.wav");
    auto serializedData = protocol_->serializeMessage(validMessage);

    // Corrupt header
    if (serializedData.size() > 10) {
        serializedData[5] = 0xFF; // Corrupt a byte in header
        auto corruptedResult = protocol_->deserializeMessage(serializedData);
        EXPECT_FALSE(corruptedResult.has_value());
    }

    // Test with checksum enabled
    protocol_->setChecksumEnabled(true);
    auto checksumData = protocol_->serializeMessage(validMessage);

    // Corrupt payload
    if (checksumData.size() > 50) {
        checksumData[checksumData.size() - 1] ^= 0xFF; // Flip a bit
        auto checksumCorruptResult = protocol_->deserializeMessage(checksumData);
        EXPECT_FALSE(checksumCorruptResult.has_value());
    }
}

// Test protocol configuration
TEST_F(NetworkProtocolTest, ProtocolConfiguration) {
    // Test default configuration
    EXPECT_TRUE(protocol_->isCompressionEnabled());
    EXPECT_TRUE(protocol_->isChecksumEnabled());
    EXPECT_EQ(protocol_->getCompressionLevel(), 6);
    EXPECT_EQ(protocol_->getMaxMessageSize(), 16 * 1024 * 1024); // 16MB default

    // Test configuration changes
    protocol_->setCompressionEnabled(false);
    EXPECT_FALSE(protocol_->isCompressionEnabled());

    protocol_->setCompressionLevel(9);
    EXPECT_EQ(protocol_->getCompressionLevel(), 9);

    protocol_->setChecksumEnabled(false);
    EXPECT_FALSE(protocol_->isChecksumEnabled());

    protocol_->setMaxMessageSize(32 * 1024 * 1024); // 32MB
    EXPECT_EQ(protocol_->getMaxMessageSize(), 32 * 1024 * 1024);
}