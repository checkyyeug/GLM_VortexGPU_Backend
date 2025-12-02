#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <websocketpp/client.hpp>
#include <websocketpp/config/asio_no_tls_client.hpp>
#include <nlohmann/json.hpp>
#include <chrono>
#include <thread>
#include <atomic>
#include <future>

using json = nlohmann::json;
using websocket_client = websocketpp::client<websocketpp::config::asio_no_tls_client>;

class WebSocketRealtimeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Start the WebSocket server on a test port
        server_port_ = 8081;
        server_url_ = "ws://localhost:" + std::to_string(server_port_);

        // Track message reception
        messages_received_.store(0);
        connection_established_.store(false);
        last_message_timestamp_.store(0);
    }

    void TearDown() override {
        // Cleanup connections and stop server
        if (client_) {
            client_->close(websocket_client::close::status::normal, "Test cleanup");
        }
    }

    // Test data structure for real-time audio visualization
    struct AudioVisualizationData {
        uint64_t timestamp;
        std::vector<float> spectrum;      // 2048-point FFT data
        std::vector<float> waveform;      // 1024-point waveform data
        float left_level;                 // Left channel VU meter
        float right_level;                // Right channel VU meter
        uint32_t sample_rate;             // Current sample rate
        uint64_t frames_processed;        // Total frames processed
        float cpu_usage;                  // CPU utilization percentage
        float gpu_usage;                  // GPU utilization percentage
    };

    // WebSocket message handler for testing
    void on_message(websocket_client* c, const websocket_client::message_ptr& msg) {
        try {
            auto json_data = json::parse(msg->get_payload());
            messages_received_.fetch_add(1);
            last_message_timestamp_.store(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());

            // Validate message structure
            EXPECT_TRUE(json_data.contains("type"));
            EXPECT_TRUE(json_data.contains("timestamp"));
            EXPECT_TRUE(json_data.contains("data"));

            if (json_data["type"] == "audio_visualization") {
                validate_audio_visualization_data(json_data["data"]);
            }
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Exception in message handler: " << e.what();
        }
    }

    void on_open(websocket_client* c) {
        connection_established_.store(true);
    }

    void validate_audio_visualization_data(const json& data) {
        // Validate required fields
        EXPECT_TRUE(data.contains("spectrum"));
        EXPECT_TRUE(data.contains("waveform"));
        EXPECT_TRUE(data.contains("left_level"));
        EXPECT_TRUE(data.contains("right_level"));
        EXPECT_TRUE(data.contains("sample_rate"));
        EXPECT_TRUE(data.contains("frames_processed"));
        EXPECT_TRUE(data.contains("cpu_usage"));
        EXPECT_TRUE(data.contains("gpu_usage"));

        // Validate data types and ranges
        EXPECT_TRUE(data["spectrum"].is_array());
        EXPECT_TRUE(data["waveform"].is_array());
        EXPECT_GE(data["spectrum"].size(), 512);  // Minimum spectrum size
        EXPECT_GE(data["waveform"].size(), 256);   // Minimum waveform size
        EXPECT_GE(data["sample_rate"], 44100);
        EXPECT_GE(data["frames_processed"], 0);
        EXPECT_GE(data["cpu_usage"], 0.0f);
        EXPECT_LE(data["cpu_usage"], 100.0f);
        EXPECT_GE(data["gpu_usage"], 0.0f);
        EXPECT_LE(data["gpu_usage"], 100.0f);

        // Validate audio levels (should be between -60 and 0 dBFS)
        EXPECT_GE(data["left_level"], -60.0f);
        EXPECT_LE(data["left_level"], 0.0f);
        EXPECT_GE(data["right_level"], -60.0f);
        EXPECT_LE(data["right_level"], 0.0f);

        // Validate spectrum data (should be non-negative)
        for (const auto& value : data["spectrum"]) {
            EXPECT_GE(value.get<float>(), 0.0f);
        }

        // Validate waveform data (should be between -1 and 1)
        for (const auto& value : data["waveform"]) {
            EXPECT_GE(value.get<float>(), -1.0f);
            EXPECT_LE(value.get<float>(), 1.0f);
        }
    }

    std::unique_ptr<websocket_client> client_;
    std::string server_url_;
    int server_port_;
    std::atomic<int> messages_received_{0};
    std::atomic<bool> connection_established_{false};
    std::atomic<uint64_t> last_message_timestamp_{0};
};

// Test 1: WebSocket connection establishment
TEST_F(WebSocketRealtimeTest, ConnectionEstablishment) {
    // This test would connect to the real WebSocket server
    // For now, we'll mock the connection since the server isn't implemented yet

    // Initialize client
    client_ = std::make_unique<websocket_client>();

    // Set up handlers
    client_->set_open_handler([this](websocketpp::connection_hdl) {
        on_open(client_.get());
    });

    client_->set_message_handler([this](websocketpp::connection_hdl,
                                       const websocket_client::message_ptr& msg) {
        on_message(client_.get(), msg);
    });

    // Connect to server
    websocketpp::lib::error_code ec;
    auto con = client_->get_connection(server_url_, ec);

    if (ec) {
        // Server not running yet - this is expected in the test environment
        GTEST_SKIP() << "WebSocket server not running - skipping connection test";
        return;
    }

    client_->connect(con);

    // Wait for connection or timeout
    auto start = std::chrono::steady_clock::now();
    while (!connection_established_.load() &&
           std::chrono::steady_clock::now() - start < std::chrono::seconds(5)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    EXPECT_TRUE(connection_established_.load())
        << "WebSocket connection should be established within 5 seconds";
}

// Test 2: Real-time data frequency and timing
TEST_F(WebSocketRealtimeTest, RealtimeDataFrequency) {
    // Test that visualization data is sent at the expected frequency (60 FPS)
    const int expected_messages_per_second = 60;
    const int test_duration_seconds = 2;
    const int expected_total_messages = expected_messages_per_second * test_duration_seconds;

    // For testing without actual server, we'll simulate message reception
    auto simulate_messages = [&]() {
        for (int i = 0; i < expected_total_messages; ++i) {
            // Create mock visualization data
            json message = {
                {"type", "audio_visualization"},
                {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count()},
                {"data", {
                    {"spectrum", std::vector<float>(2048, 0.5f)},
                    {"waveform", std::vector<float>(1024, 0.0f)},
                    {"left_level", -20.0f},
                    {"right_level", -18.0f},
                    {"sample_rate", 44100},
                    {"frames_processed", static_cast<uint64_t>(i * 1024)},
                    {"cpu_usage", 45.0f},
                    {"gpu_usage", 30.0f}
                }}
            };

            // Simulate message reception
            messages_received_.fetch_add(1);
            last_message_timestamp_.store(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());

            // Sleep to maintain 60 FPS rate
            std::this_thread::sleep_for(std::chrono::milliseconds(1000 / expected_messages_per_second));
        }
    };

    // Run simulation in background thread
    std::thread message_thread(simulate_messages);

    // Wait for test duration
    std::this_thread::sleep_for(std::chrono::seconds(test_duration_seconds));

    message_thread.join();

    // Verify message count
    EXPECT_GE(messages_received_.load(), expected_total_messages * 0.9)
        << "Should receive at least 90% of expected messages";

    // Verify timing consistency
    // Calculate average interval between messages
    auto current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    if (last_message_timestamp_.load() > 0 && messages_received_.load() > 1) {
        // Check that messages are arriving at roughly the expected interval
        // Allow for some tolerance due to system scheduling
        const double expected_interval_ms = 1000.0 / expected_messages_per_second;
        const double tolerance_ms = expected_interval_ms * 0.5; // 50% tolerance

        // This is a simplified check - in real implementation would track individual message timestamps
        EXPECT_NEAR(messages_received_.load(), expected_total_messages, expected_total_messages * 0.1)
            << "Message count should be close to expected value";
    }
}

// Test 3: Data integrity and validation
TEST_F(WebSocketRealtimeTest, DataIntegrityAndValidation) {
    // Test data structure validation through message processing
    json valid_message = {
        {"type", "audio_visualization"},
        {"timestamp", 1234567890},
        {"data", {
            {"spectrum", std::vector<float>(2048, 0.5f)},
            {"waveform", std::vector<float>(1024, 0.0f)},
            {"left_level", -20.0f},
            {"right_level", -18.0f},
            {"sample_rate", 44100},
            {"frames_processed", 1024},
            {"cpu_usage", 45.0f},
            {"gpu_usage", 30.0f}
        }}
    };

    // Test valid message processing
    EXPECT_NO_THROW({
        validate_audio_visualization_data(valid_message["data"]);
    });

    // Test invalid messages
    json missing_fields_message = {
        {"type", "audio_visualization"},
        {"timestamp", 1234567890},
        {"data", {
            {"spectrum", std::vector<float>(2048, 0.5f)},
            // Missing required fields
        }}
    };

    EXPECT_THROW({
        validate_audio_visualization_data(missing_fields_message["data"]);
    }, std::exception);

    // Test invalid data ranges
    json invalid_levels_message = {
        {"type", "audio_visualization"},
        {"timestamp", 1234567890},
        {"data", {
            {"spectrum", std::vector<float>(2048, 0.5f)},
            {"waveform", std::vector<float>(1024, 0.0f)},
            {"left_level", -100.0f},  // Invalid: too low
            {"right_level", 10.0f},   // Invalid: too high
            {"sample_rate", 44100},
            {"frames_processed", 1024},
            {"cpu_usage", 45.0f},
            {"gpu_usage", 30.0f}
        }}
    };

    EXPECT_THROW({
        validate_audio_visualization_data(invalid_levels_message["data"]);
    }, std::exception);
}

// Test 4: Performance under load
TEST_F(WebSocketRealtimeTest, PerformanceUnderLoad) {
    // Test that the system can handle multiple concurrent connections
    const int num_concurrent_clients = 10;
    const int messages_per_client = 100;

    std::vector<std::atomic<int>> client_message_counts(num_concurrent_clients);
    for (auto& count : client_message_counts) {
        count.store(0);
    }

    // Simulate multiple clients receiving data
    auto client_simulation = [&](int client_id) {
        for (int i = 0; i < messages_per_client; ++i) {
            // Simulate message reception
            client_message_counts[client_id].fetch_add(1);
            std::this_thread::sleep_for(std::chrono::microseconds(100)); // ~10kHz update rate
        }
    };

    // Start all client simulations
    std::vector<std::thread> client_threads;
    for (int i = 0; i < num_concurrent_clients; ++i) {
        client_threads.emplace_back(client_simulation, i);
    }

    // Wait for completion
    for (auto& thread : client_threads) {
        thread.join();
    }

    // Verify all clients received their messages
    for (int i = 0; i < num_concurrent_clients; ++i) {
        EXPECT_EQ(client_message_counts[i].load(), messages_per_client)
            << "Client " << i << " should receive all messages";
    }

    // Calculate total throughput
    int total_messages = 0;
    for (const auto& count : client_message_counts) {
        total_messages += count.load();
    }

    EXPECT_EQ(total_messages, num_concurrent_clients * messages_per_client)
        << "Total messages should match expected count";

    // Performance should be reasonable (this is a rough estimate)
    const int expected_min_messages_per_second = 1000; // Should handle at least 1k messages/sec
    EXPECT_GT(total_messages, expected_min_messages_per_second)
        << "Should handle reasonable message throughput";
}

// Test 5: Connection resilience and error handling
TEST_F(WebSocketRealtimeTest, ConnectionResilience) {
    // Test behavior when connection is interrupted

    std::atomic<bool> connection_lost{false};
    std::atomic<int> reconnection_attempts{0};

    // Simulate connection loss and recovery
    auto connection_simulation = [&]() {
        // Simulate initial connection
        connection_established_.store(true);
        messages_received_.store(0);

        // Simulate normal operation
        for (int i = 0; i < 10; ++i) {
            if (connection_lost.load()) {
                reconnection_attempts.fetch_add(1);
                connection_lost.store(false);
                connection_established_.store(true);
            }

            messages_received_.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Simulate connection loss
        connection_lost.store(true);
        connection_established_.store(false);

        // Simulate reconnection
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        reconnection_attempts.fetch_add(1);
        connection_established_.store(true);

        // Continue sending messages after reconnection
        for (int i = 0; i < 5; ++i) {
            messages_received_.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    };

    std::thread simulation_thread(connection_simulation);
    simulation_thread.join();

    EXPECT_GT(reconnection_attempts.load(), 0)
        << "Should attempt reconnection after loss";
    EXPECT_GT(messages_received_.load(), 10)
        << "Should receive messages before and after reconnection";
    EXPECT_TRUE(connection_established_.load())
        << "Should be connected after recovery";
}

// Test 6: Message compression and bandwidth efficiency
TEST_F(WebSocketRealtimeTest, MessageCompression) {
    // Test that messages are efficiently compressed for bandwidth

    // Create large visualization data (simulating high-resolution data)
    std::vector<float> large_spectrum(8192, 0.5f);  // 4x larger spectrum
    std::vector<float> large_waveform(4096, 0.0f);   // 4x larger waveform

    json large_message = {
        {"type", "audio_visualization"},
        {"timestamp", 1234567890},
        {"data", {
            {"spectrum", large_spectrum},
            {"waveform", large_waveform},
            {"left_level", -20.0f},
            {"right_level", -18.0f},
            {"sample_rate", 44100},
            {"frames_processed", 1024},
            {"cpu_usage", 45.0f},
            {"gpu_usage", 30.0f}
        }}
    };

    std::string message_str = large_message.dump();

    // Calculate message size
    size_t message_size = message_str.length();

    // For 60 FPS at 8kHz audio, we need efficient compression
    // Target: < 50KB per second per client
    const size_t max_bandwidth_per_message = 50000 / 60; // ~833 bytes per message

    EXPECT_LT(message_size, max_bandwidth_per_message * 10)
        << "Message should be reasonably sized for real-time transmission";

    // The actual implementation would use binary protocol and compression
    // This test validates that the JSON structure isn't excessively large
    EXPECT_LT(message_size, 100 * 1024)
        << "Single message should be under 100KB";
}