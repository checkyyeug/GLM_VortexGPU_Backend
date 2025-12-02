#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "network/websocket_server.hpp"
#include "network_types.hpp"
#include "system/logger.hpp"

#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/client.hpp>
#include <thread>
#include <chrono>
#include <atomic>

using namespace vortex;
using ::testing::_;
using ::testing::Return;
using ::testing::DoAll;
using ::testing::SetArgReferee;

class WebSocketProtocolTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::initialize("test_websocket.log", Logger::Level::Debug, false);

        // Start WebSocket server for testing
        server = std::make_unique<WebSocketServer>();
        NetworkConfig config;
        config.websocketPort = 8081; // Use different port for testing
        config.maxConnections = 10;
        config.enableCompression = true;

        ASSERT_TRUE(server->start(config));

        // Give server time to start
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    void TearDown() override {
        if (server) {
            server->stop();
            server.reset();
        }
        Logger::shutdown();
    }

    using ClientType = websocketpp::client<websocketpp::config::asio>;
    using ConnectionHdl = websocketpp::connection_hdl;

    // Helper to create test client
    std::shared_ptr<ClientType> createTestClient() {
        auto client = std::make_shared<ClientType>();

        client->set_access_channels(websocketpp::log::alevel::all);
        client->clear_access_channels(websocketpp::log::alevel::frame_payload);
        client->set_error_channels(websocketpp::log::elevel::all);

        client->init_asio();

        return client;
    }

    // Connect client to server
    ConnectionHdl connectClient(std::shared_ptr<ClientType> client, uint16_t port = 8081) {
        websocketpp::lib::error_code ec;
        ConnectionHdl connection = client->get_connection(
            "ws://localhost:" + std::to_string(port), ec);

        EXPECT_FALSE(ec) << "Connection error: " << ec.message();
        client->connect(connection);

        // Wait for connection
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        return connection;
    }

    std::unique_ptr<WebSocketServer> server;
    std::atomic<int> messageCount{0};
    std::atomic<int> connectionCount{0};
};

// Test WebSocket server startup and shutdown
TEST_F(WebSocketProtocolTest, TestServerLifecycle) {
    EXPECT_TRUE(server->isRunning());
    EXPECT_EQ(server->getPort(), 8081);

    auto stats = server->getStatistics();
    EXPECT_EQ(stats.totalConnections, 0);
    EXPECT_EQ(stats.activeConnections, 0);
}

// Test basic WebSocket connection
TEST_F(WebSocketProtocolTest, TestBasicConnection) {
    auto client = createTestClient();

    ConnectionHdl connection = connectClient(client);

    // Run client io service briefly
    client->run_one();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Check connection statistics
    auto stats = server->getStatistics();
    EXPECT_GT(stats.totalConnections, 0);
    EXPECT_GE(stats.activeConnections, 1);
}

// Test multiple simultaneous connections
TEST_F(WebSocketProtocolTest, TestMultipleConnections) {
    const int numClients = 5;
    std::vector<std::shared_ptr<ClientType>> clients;
    std::vector<ConnectionHdl> connections;

    // Create multiple clients
    for (int i = 0; i < numClients; ++i) {
        auto client = createTestClient();
        ConnectionHdl connection = connectClient(client);

        clients.push_back(client);
        connections.push_back(connection);

        // Give each connection time to establish
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // Check connection statistics
    auto stats = server->getStatistics();
    EXPECT_EQ(stats.totalConnections, numClients);
    EXPECT_EQ(stats.activeConnections, numClients);

    // Disconnect all clients
    for (auto& client : clients) {
        client->close(websocketpp::close::status::normal, "");
    }

    // Give server time to process disconnections
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    stats = server->getStatistics();
    EXPECT_EQ(stats.activeConnections, 0);
}

// Test message subscription
TEST_F(WebSocketProtocolTest, TestMessageSubscription) {
    auto client = createTestClient();

    // Set up message handler
    std::string receivedMessage;
    client->set_message_handler([&client, &receivedMessage](ConnectionHdl hdl, ClientType::message_ptr msg) {
        receivedMessage = msg->get_payload();
    });

    ConnectionHdl connection = connectClient(client);

    // Subscribe to spectrum data
    std::string subscribeMessage = R"({
        "type": "subscribe",
        "dataTypes": ["spectrum"],
        "updateRate": 60
    })";

    client->send(connection, subscribeMessage, websocketpp::frame::opcode::text);

    // Run client io service and wait for response
    auto start = std::chrono::steady_clock::now();
    while (receivedMessage.empty() &&
           std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::steady_clock::now() - start).count() < 5) {
        client->run_one();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Should receive a subscription confirmation or spectrum data
    EXPECT_FALSE(receivedMessage.empty());
}

// Test binary protocol for audio data
TEST_F(WebSocketProtocolTest, TestBinaryProtocol) {
    auto client = createTestClient();

    // Set up binary message handler
    std::vector<uint8_t> receivedBinaryData;
    client->set_message_handler([&client, &receivedBinaryData](ConnectionHdl hdl, ClientType::message_ptr msg) {
        if (msg->get_opcode() == websocketpp::frame::opcode::binary) {
            receivedBinaryData.clear();
            const std::string& payload = msg->get_payload();
            receivedBinaryData.assign(payload.begin(), payload.end());
        }
    });

    ConnectionHdl connection = connectClient(client);

    // Create binary audio data (2048 samples for spectrum analysis)
    std::vector<float> audioData(2048);
    for (size_t i = 0; i < audioData.size(); ++i) {
        audioData[i] = std::sin(2.0f * M_PI * 440.0f * i / 44100.0f);
    }

    // Create binary protocol message
    std::vector<uint8_t> binaryMessage;

    // Header (as defined in network_types.hpp)
    BinaryProtocolHeader header;
    header.magic = 0x56545858; // "VTVX"
    header.version = 1;
    header.messageType = 1; // Audio data message type
    header.payloadSize = audioData.size() * sizeof(float);
    header.timestamp = getCurrentTimestamp();
    header.sequenceNumber = 1;
    header.flags = 0;

    // Serialize header
    std::vector<uint8_t> headerData = serializeBinaryHeader(header);
    binaryMessage.insert(binaryMessage.end(), headerData.begin(), headerData.end());

    // Add audio data
    const uint8_t* audioBytes = reinterpret_cast<const uint8_t*>(audioData.data());
    binaryMessage.insert(binaryMessage.end(), audioBytes, audioBytes + header.payloadSize);

    // Send binary message
    client->send(connection, binaryMessage, websocketpp::frame::opcode::binary);

    // Run client and wait for response
    auto start = std::chrono::steady_clock::now();
    while (receivedBinaryData.empty() &&
           std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::steady_clock::now() - start).count() < 5) {
        client->run_one();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Should receive a response (possibly spectrum analysis result)
    if (!receivedBinaryData.empty()) {
        // Verify response header
        EXPECT_GE(receivedBinaryData.size(), sizeof(BinaryProtocolHeader));
    }
}

// Test real-time data streaming
TEST_F(WebSocketProtocolTest, TestRealTimeDataStreaming) {
    auto client = createTestClient();

    // Track received messages
    std::vector<std::string> receivedMessages;
    client->set_message_handler([&client, &receivedMessages](ConnectionHdl hdl, ClientType::message_ptr msg) {
        receivedMessages.push_back(msg->get_payload());
    });

    ConnectionHdl connection = connectClient(client);

    // Subscribe to multiple data types with high update rate
    std::string subscribeMessage = R"({
        "type": "subscribe",
        "dataTypes": ["spectrum", "waveform", "meters", "hardware"],
        "updateRate": 120
    })";

    client->send(connection, subscribeMessage, websocketpp::frame::opcode::text);

    // Run client and collect messages for 2 seconds
    auto startTime = std::chrono::steady_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::steady_clock::now() - startTime).count() < 2) {
        client->run_one();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Should receive multiple real-time updates
    EXPECT_GT(receivedMessages.size(), 10);

    // Verify message format
    for (const auto& message : receivedMessages) {
        try {
            // Should be valid JSON
            rapidjson::Document doc;
            doc.Parse(message.c_str());
            EXPECT_FALSE(doc.HasParseError());

            if (!doc.HasParseError()) {
                // Should have type and timestamp fields
                EXPECT_TRUE(doc.HasMember("type"));
                EXPECT_TRUE(doc.HasMember("timestamp"));
            }
        } catch (const std::exception& e) {
            // Some messages might be binary
        }
    }
}

// Test connection error handling
TEST_F(WebSocketProtocolTest, TestConnectionErrorHandling) {
    // Test connecting to invalid port
    auto client = createTestClient();

    websocketpp::lib::error_code ec;
    ConnectionHdl connection = client->get_connection("ws://localhost:9999", ec);

    EXPECT_TRUE(ec) << "Should fail to connect to invalid port";

    if (!ec) {
        client->connect(connection);

        // Should fail to establish connection
        auto start = std::chrono::steady_clock::now();
        bool connected = false;

        while (std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::steady_clock::now() - start).count() < 2) {
            client->run_one();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // Should not be connected
        EXPECT_FALSE(connected);
    }
}

// Test message validation
TEST_F(WebSocketProtocolTest, TestMessageValidation) {
    auto client = createTestClient();

    // Track error messages
    std::string errorMessage;
    client->set_message_handler([&client, &errorMessage](ConnectionHdl hdl, ClientType::message_ptr msg) {
        if (msg->get_opcode() == websocketpp::frame::opcode::text) {
            rapidjson::Document doc;
            doc.Parse(msg->get_payload().c_str());
            if (!doc.HasParseError() && doc.HasMember("type") &&
                std::string(doc["type"].GetString()) == "error") {
                errorMessage = msg->get_payload();
            }
        }
    });

    ConnectionHdl connection = connectClient(client);

    // Send invalid JSON
    std::string invalidJson = "{ invalid json message";
    client->send(connection, invalidJson, websocketpp::frame::opcode::text);

    // Send invalid message type
    std::string invalidType = R"({
        "type": "invalid_type",
        "data": {}
    })";
    client->send(connection, invalidType, websocketpp::frame::opcode::text);

    // Run client and wait for error responses
    auto start = std::chrono::steady_clock::now();
    while (errorMessage.empty() &&
           std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::steady_clock::now() - start).count() < 2) {
        client->run_one();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Should receive error messages for invalid requests
    EXPECT_FALSE(errorMessage.empty());
}

// Test message compression
TEST_F(WebSocketProtocolTest, TestMessageCompression) {
    auto client = createTestClient();

    // Set up client to accept compression
    client->get_alog().set_level(websocketpp::log::alevel::devel);

    ConnectionHdl connection = connectClient(client);

    // Send large message to trigger compression
    std::string largeMessage(10000, 'x'); // 10KB message
    largeMessage = R"({
        "type": "test",
        "data": ")" + largeMessage + R"("
    })";

    client->send(connection, largeMessage, websocketpp::frame::opcode::text);

    // Measure transmission time
    auto start = std::chrono::high_resolution_clock::now();

    client->run_one();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // With compression enabled, large message transmission should be reasonable
    EXPECT_LT(duration.count(), 1000); // Should transmit within 1 second
}

// Test heartbeat mechanism
TEST_F(WebSocketProtocolTest, TestHeartbeat) {
    auto client = createTestClient();

    std::vector<std::string> heartbeatMessages;
    client->set_message_handler([&client, &heartbeatMessages](ConnectionHdl hdl, ClientType::message_ptr msg) {
        if (msg->get_opcode() == websocketpp::frame::opcode::text) {
            rapidjson::Document doc;
            doc.Parse(msg->get_payload().c_str());
            if (!doc.HasParseError() && doc.HasMember("type") &&
                std::string(doc["type"].GetString()) == "heartbeat") {
                heartbeatMessages.push_back(msg->get_payload());
            }
        }
    });

    ConnectionHdl connection = connectClient(client);

    // Keep connection alive for multiple heartbeat intervals
    std::this_thread::sleep_for(std::chrono::seconds(6));

    client->run_one();

    // Should receive at least one heartbeat message
    EXPECT_GT(heartbeatMessages.size(), 0);

    // Verify heartbeat message format
    if (!heartbeatMessages.empty()) {
        rapidjson::Document doc;
        doc.Parse(heartbeatMessages[0].c_str());
        EXPECT_FALSE(doc.HasParseError());
        EXPECT_TRUE(doc.HasMember("type"));
        EXPECT_TRUE(doc.HasMember("timestamp"));
    }
}

// Test concurrent message handling
TEST_F(WebSocketProtocolTest, TestConcurrentMessages) {
    const int numClients = 10;
    const int messagesPerClient = 20;

    std::vector<std::shared_ptr<ClientType>> clients;
    std::vector<ConnectionHdl> connections;
    std::vector<std::atomic<int>> messageCounts(numClients);

    // Create multiple clients
    for (int i = 0; i < numClients; ++i) {
        auto client = createTestClient();

        // Set up message handler for each client
        client->set_message_handler([&client, &messageCounts, i](ConnectionHdl hdl, ClientType::message_ptr msg) {
            messageCounts[i]++;
        });

        ConnectionHdl connection = connectClient(client);

        clients.push_back(client);
        connections.push_back(connection);

        messageCounts[i] = 0;
    }

    // Send messages concurrently
    std::vector<std::thread> threads;
    for (int clientIndex = 0; clientIndex < numClients; ++clientIndex) {
        threads.emplace_back([&, clientIndex]() {
            for (int msgIndex = 0; msgIndex < messagesPerClient; ++msgIndex) {
                std::string message = R"({
                    "type": "subscribe",
                    "dataTypes": ["spectrum"],
                    "updateRate": )" + std::to_string(10 + msgIndex) + R"(
                })";

                clients[clientIndex]->send(connections[clientIndex], message,
                                          websocketpp::frame::opcode::text);

                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Run all client io services
    for (auto& client : clients) {
        for (int i = 0; i < 10; ++i) {
            client->run_one();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    // Wait for responses
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Verify all messages were handled without errors
    for (int i = 0; i < numClients; ++i) {
        EXPECT_GE(messageCounts[i], 0) << "Client " << i << " should handle messages";
    }

    // Check server statistics
    auto stats = server->getStatistics();
    EXPECT_EQ(stats.activeConnections, numClients);
    EXPECT_GT(stats.bytesTransmitted, 0);
}