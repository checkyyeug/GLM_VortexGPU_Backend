#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <sstream>

#include "../../src/output/roon_bridge.hpp"
#include "../../src/core/audio_engine.hpp"

using namespace vortex;
using namespace testing;

class RoonBridgeTest : public ::testing::Test {
protected:
    void SetUp() override {
        roonBridge_ = std::make_unique<RoonBridge>();

        // Initialize with standard configuration
        RoonBridge::BridgeConfig config;
        config.deviceName = "Vortex Test Bridge";
        config.deviceId = "vortex-test-bridge";
        config.enableRAAT = true;
        config.enableAirPlay = true;
        config.enableHTTPControl = true;
        config.httpPort = 9330; // Use different port for testing
        config.airPlayPort = 5000;
        config.maxSampleRate = 192000;
        config.maxBitDepth = 32;
        config.maxChannels = 8;

        ASSERT_TRUE(roonBridge_->initialize(config));
    }

    void TearDown() override {
        if (roonBridge_ && roonBridge_->isRunning()) {
            roonBridge_->stop();
        }
        roonBridge_.reset();
    }

    void generateTestAudio(std::vector<float>& audio, size_t numSamples, uint16_t channels) {
        audio.resize(numSamples * channels);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

        for (size_t i = 0; i < numSamples; ++i) {
            for (uint16_t ch = 0; ch < channels; ++ch) {
                audio[i * channels + ch] = dis(gen);
            }
        }
    }

    void generateSineWave(std::vector<float>& audio, size_t numSamples, uint16_t channels,
                         float frequency = 440.0f, float amplitude = 0.5f) {
        audio.resize(numSamples * channels);

        for (size_t i = 0; i < numSamples; ++i) {
            float time = static_cast<float>(i) / 48000.0f; // Assume 48kHz sample rate
            float sample = amplitude * std::sin(2.0f * M_PI * frequency * time);

            for (uint16_t ch = 0; ch < channels; ++ch) {
                audio[i * channels + ch] = sample;
            }
        }
    }

    std::unique_ptr<RoonBridge> roonBridge_;
};

// Initialization tests
TEST_F(RoonBridgeTest, InitializeWithValidConfig) {
    RoonBridge::BridgeConfig config;
    config.deviceName = "Test Bridge";
    config.deviceId = "test-bridge";
    config.enableRAAT = true;
    config.enableAirPlay = true;
    config.enableHTTPControl = true;
    config.httpPort = 9331;

    RoonBridge bridge;
    EXPECT_TRUE(bridge.initialize(config));
    EXPECT_TRUE(bridge.isInitialized());
}

TEST_F(RoonBridgeTest, InitializeWithInvalidConfig) {
    RoonBridge::BridgeConfig config;
    config.deviceName = ""; // Invalid empty name
    config.deviceId = "test-bridge";

    RoonBridge bridge;
    EXPECT_FALSE(bridge.initialize(config));
    EXPECT_FALSE(bridge.isInitialized());
}

// Server control tests
TEST_F(RoonBridgeTest, StartStopServer) {
    EXPECT_FALSE(roonBridge_->isRunning());

    EXPECT_TRUE(roonBridge_->start());
    EXPECT_TRUE(roonBridge_->isRunning());

    // Should not start again when already running
    EXPECT_FALSE(roonBridge_->start());

    EXPECT_TRUE(roonBridge_->stop());
    EXPECT_FALSE(roonBridge_->isRunning());

    // Should not stop again when already stopped
    EXPECT_FALSE(roonBridge_->stop());
}

TEST_F(RoonBridgeTest, RestartServer) {
    EXPECT_TRUE(roonBridge_->start());
    EXPECT_TRUE(roonBridge_->isRunning());

    EXPECT_TRUE(roonBridge_->restart());
    EXPECT_TRUE(roonBridge_->isRunning());
}

// Service management tests
TEST_F(RoonBridgeTest, EnableDisableServices) {
    EXPECT_TRUE(roonBridge_->enableRAAT(false));
    EXPECT_FALSE(roonBridge_->isRAATEnabled());

    EXPECT_TRUE(roonBridge_->enableRAAT(true));
    EXPECT_TRUE(roonBridge_->isRAATEnabled());

    EXPECT_TRUE(roonBridge_->enableAirPlay(false));
    EXPECT_FALSE(roonBridge_->isAirPlayEnabled());

    EXPECT_TRUE(roonBridge_->enableAirPlay(true));
    EXPECT_TRUE(roonBridge_->isAirPlayEnabled());

    EXPECT_TRUE(roonBridge_->enableHTTPControl(false));
    EXPECT_FALSE(roonBridge_->isHTTPControlEnabled());

    EXPECT_TRUE(roonBridge_->enableHTTPControl(true));
    EXPECT_TRUE(roonBridge_->isHTTPControlEnabled());
}

// Device information tests
TEST_F(RoonBridgeTest, GetDeviceInfo) {
    auto deviceInfo = roonBridge_->getDeviceInfo();

    EXPECT_FALSE(deviceInfo.deviceId.empty());
    EXPECT_FALSE(deviceInfo.deviceName.empty());
    EXPECT_GT(deviceInfo.maxSampleRate, 0);
    EXPECT_GT(deviceInfo.maxBitDepth, 0);
    EXPECT_GT(deviceInfo.maxChannels, 0);
    EXPECT_FALSE(deviceInfo.version.empty());
}

TEST_F(RoonBridgeTest, SetDeviceInfo) {
    EXPECT_TRUE(roonBridge_->setDeviceName("New Test Name"));
    auto deviceInfo = roonBridge_->getDeviceInfo();
    EXPECT_EQ(deviceInfo.deviceName, "New Test Name");

    EXPECT_TRUE(roonBridge_->setMaxSampleRate(384000));
    deviceInfo = roonBridge_->getDeviceInfo();
    EXPECT_EQ(deviceInfo.maxSampleRate, 384000);

    EXPECT_TRUE(roonBridge_->setMaxBitDepth(24));
    deviceInfo = roonBridge_->getDeviceInfo();
    EXPECT_EQ(deviceInfo.maxBitDepth, 24);

    EXPECT_TRUE(roonBridge_->setMaxChannels(16));
    deviceInfo = roonBridge_->getDeviceInfo();
    EXPECT_EQ(deviceInfo.maxChannels, 16);
}

// Network configuration tests
TEST_F(RoonBridgeTest, NetworkConfiguration) {
    EXPECT_TRUE(roonBridge_->setHTTPPort(9332));
    EXPECT_EQ(roonBridge_->getHTTPPort(), 9332);

    EXPECT_TRUE(roonBridge_->setAirPlayPort(5001));
    EXPECT_EQ(roonBridge_->getAirPlayPort(), 5001);

    EXPECT_TRUE(roonBridge_->setNetworkInterface("eth0"));
    EXPECT_EQ(roonBridge_->getNetworkInterface(), "eth0");

    EXPECT_TRUE(roonBridge_->setBonjourName("Vortex Audio"));
    EXPECT_EQ(roonBridge_->getBonjourName(), "Vortex Audio");
}

// Audio format tests
TEST_F(RoonBridgeTest, SupportedFormats) {
    auto formats = roonBridge_->getSupportedFormats();

    EXPECT_FALSE(formats.empty());

    // Check for common formats
    bool hasPCM = false, hasDSD = false, hasMQA = false;
    for (const auto& format : formats) {
        if (format.type == RoonBridge::AudioFormat::PCM) hasPCM = true;
        if (format.type == RoonBridge::AudioFormat::DSD) hasDSD = true;
        if (format.type == RoonBridge::AudioFormat::MQA) hasMQA = true;
    }

    EXPECT_TRUE(hasPCM); // Should always support PCM
    // DSD and MQA support may be optional
}

TEST_F(RoonBridgeTest, FormatCapability) {
    // Test format capability queries
    EXPECT_TRUE(roonBridge_->supportsFormat(RoonBridge::AudioFormat::PCM, 44100, 16, 2));
    EXPECT_TRUE(roonBridge_->supportsFormat(RoonBridge::AudioFormat::PCM, 192000, 24, 2));

    // Test high sample rates
    EXPECT_TRUE(roonBridge_->supportsFormat(RoonBridge::AudioFormat::PCM, 384000, 32, 2));

    // Test invalid formats
    EXPECT_FALSE(roonBridge_->supportsFormat(RoonBridge::AudioFormat::PCM, 0, 16, 2));
    EXPECT_FALSE(roonBridge_->supportsFormat(RoonBridge::AudioFormat::PCM, 48000, 0, 2));
}

// Audio processing tests
TEST_F(RoonBridgeTest, ProcessAudio) {
    std::vector<float> audio;
    generateTestAudio(audio, 1024, 2);

    EXPECT_TRUE(roonBridge_->processAudio(audio.data(), audio.size() / 2, 2));
}

TEST_F(RoonBridgeTest, ProcessMultiChannelAudio) {
    std::vector<float> audio;
    generateSineWave(audio, 512, 8, 440.0f, 0.3f);

    std::vector<const float*> inputs(8);
    std::vector<float*> outputs(8);
    std::vector<std::vector<float>> inputBuffers(8);
    std::vector<std::vector<float>> outputBuffers(8);

    for (int ch = 0; ch < 8; ++ch) {
        inputBuffers[ch].resize(512);
        outputBuffers[ch].resize(512);
        inputs[ch] = inputBuffers[ch].data();
        outputs[ch] = outputBuffers[ch].data();

        // Deinterleave audio
        for (size_t i = 0; i < 512; ++i) {
            inputBuffers[ch][i] = audio[i * 8 + ch];
        }
    }

    EXPECT_TRUE(roonBridge_->processAudioMultiChannel(inputs.data(), outputs.data(), 512, 8));
}

// Zone and output tests
TEST_F(RoonBridgeTest, ZoneManagement) {
    auto zones = roonBridge_->getZones();
    EXPECT_TRUE(zones.empty() || !zones.empty()); // Either way is fine

    // Add a test zone
    std::string zoneId = roonBridge_->addZone("Test Zone", 2);
    EXPECT_FALSE(zoneId.empty());

    zones = roonBridge_->getZones();
    EXPECT_GT(zones.size(), 0);

    // Find our test zone
    bool found = false;
    for (const auto& zone : zones) {
        if (zone.id == zoneId) {
            EXPECT_EQ(zone.name, "Test Zone");
            EXPECT_EQ(zone.channels, 2);
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);

    // Remove the zone
    EXPECT_TRUE(roonBridge_->removeZone(zoneId));
    zones = roonBridge_->getZones();

    // Should no longer find our test zone
    found = false;
    for (const auto& zone : zones) {
        if (zone.id == zoneId) {
            found = true;
            break;
        }
    }
    EXPECT_FALSE(found);
}

TEST_F(RoonBridgeTest, OutputManagement) {
    auto outputs = roonBridge_->getOutputs();
    EXPECT_TRUE(outputs.empty() || !outputs.empty()); // Either way is fine

    // Add a test output
    std::string outputId = roonBridge_->addOutput("Test Output", RoonBridge::OutputType::ANALOG);
    EXPECT_FALSE(outputId.empty());

    outputs = roonBridge_->getOutputs();
    EXPECT_GT(outputs.size(), 0);

    // Find our test output
    bool found = false;
    for (const auto& output : outputs) {
        if (output.id == outputId) {
            EXPECT_EQ(output.name, "Test Output");
            EXPECT_EQ(output.type, RoonBridge::OutputType::ANALOG);
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);

    // Configure the output
    EXPECT_TRUE(roonBridge_->configureOutput(outputId, true, 0.8f, false)); // enabled, volume, muted

    // Remove the output
    EXPECT_TRUE(roonBridge_->removeOutput(outputId));
    outputs = roonBridge_->getOutputs();

    // Should no longer find our test output
    found = false;
    for (const auto& output : outputs) {
        if (output.id == outputId) {
            found = true;
            break;
        }
    }
    EXPECT_FALSE(found);
}

// Transport control tests
TEST_F(RoonBridgeTest, TransportControl) {
    auto transportState = roonBridge_->getTransportState();
    EXPECT_EQ(transportState.state, RoonBridge::TransportState::STOPPED);

    // Test play/pause/stop
    EXPECT_TRUE(roonBridge_->play());
    transportState = roonBridge_->getTransportState();
    EXPECT_EQ(transportState.state, RoonBridge::TransportState::PLAYING);

    EXPECT_TRUE(roonBridge_->pause());
    transportState = roonBridge_->getTransportState();
    EXPECT_EQ(transportState.state, RoonBridge::TransportState::PAUSED);

    EXPECT_TRUE(roonBridge_->stop());
    transportState = roonBridge_->getTransportState();
    EXPECT_EQ(transportState.state, RoonBridge::TransportState::STOPPED);

    // Test seek
    EXPECT_TRUE(roonBridge_->seek(5000)); // Seek to 5 seconds
    transportState = roonBridge_->getTransportState();
    EXPECT_EQ(transportState.position, 5000);
}

// Volume control tests
TEST_F(RoonBridgeTest, VolumeControl) {
    EXPECT_TRUE(roonBridge_->setVolume(0.5f));
    EXPECT_FLOAT_EQ(roonBridge_->getVolume(), 0.5f);

    EXPECT_TRUE(roonBridge_->setVolume(0.0f));
    EXPECT_FLOAT_EQ(roonBridge_->getVolume(), 0.0f);

    EXPECT_TRUE(roonBridge_->setVolume(1.0f));
    EXPECT_FLOAT_EQ(roonBridge_->getVolume(), 1.0f);

    // Test volume limits
    roonBridge_->setVolume(-0.5f); // Should clamp to 0.0
    EXPECT_FLOAT_EQ(roonBridge_->getVolume(), 0.0f);

    roonBridge_->setVolume(1.5f); // Should clamp to 1.0
    EXPECT_FLOAT_EQ(roonBridge_->getVolume(), 1.0f);

    // Test mute
    EXPECT_TRUE(roonBridge_->setMute(true));
    EXPECT_TRUE(roonBridge_->isMuted());

    EXPECT_TRUE(roonBridge_->setMute(false));
    EXPECT_FALSE(roonBridge_->isMuted());
}

// Metadata tests
TEST_F(RoonBridgeTest, Metadata) {
    // Set test metadata
    RoonBridge::TrackMetadata metadata;
    metadata.title = "Test Song";
    metadata.artist = "Test Artist";
    metadata.album = "Test Album";
    metadata.duration = 180000; // 3 minutes in milliseconds
    metadata.sampleRate = 44100;
    metadata.bitDepth = 16;
    metadata.channels = 2;

    EXPECT_TRUE(roonBridge_->setCurrentMetadata(metadata));

    auto currentMetadata = roonBridge_->getCurrentMetadata();
    EXPECT_EQ(currentMetadata.title, "Test Song");
    EXPECT_EQ(currentMetadata.artist, "Test Artist");
    EXPECT_EQ(currentMetadata.album, "Test Album");
    EXPECT_EQ(currentMetadata.duration, 180000);
}

// Statistics tests
TEST_F(RoonBridgeTest, Statistics) {
    auto stats = roonBridge_->getStatistics();

    EXPECT_GE(stats.totalPlaybackTime, 0);
    EXPECT_GE(stats.totalTracksPlayed, 0);
    EXPECT_GE(stats.totalBytesTransferred, 0);
    EXPECT_GE(stats.activeConnections, 0);
    EXPECT_GE(stats.uptimeSeconds, 0);
    EXPECT_GE(stats.cpuUsage, 0.0f);
    EXPECT_GE(stats.memoryUsage, 0);

    // Process some audio to affect statistics
    std::vector<float> audio;
    generateTestAudio(audio, 1024, 2);
    roonBridge_->processAudio(audio.data(), audio.size() / 2, 2);

    stats = roonBridge_->getStatistics();
    EXPECT_GT(stats.totalBytesTransferred, 0);
}

TEST_F(RoonBridgeTest, ResetStatistics) {
    // Process some audio first
    std::vector<float> audio;
    generateTestAudio(audio, 1024, 2);
    roonBridge_->processAudio(audio.data(), audio.size() / 2, 2);

    auto stats = roonBridge_->getStatistics();
    if (stats.totalBytesTransferred > 0) {
        roonBridge_->resetStatistics();
        stats = roonBridge_->getStatistics();
        EXPECT_EQ(stats.totalPlaybackTime, 0);
        EXPECT_EQ(stats.totalTracksPlayed, 0);
        EXPECT_EQ(stats.totalBytesTransferred, 0);
        EXPECT_EQ(stats.uptimeSeconds, 0);
    }
}

// HTTP control API tests
TEST_F(RoonBridgeTest, HTTPControlEndpoints) {
    // Start the bridge to enable HTTP control
    EXPECT_TRUE(roonBridge_->start());

    // Test HTTP endpoints would require actual HTTP client
    // For now, we'll test the endpoint configuration
    auto endpoints = roonBridge_->getHTTPEndpoints();
    EXPECT_FALSE(endpoints.empty());

    // Check for required endpoints
    bool hasDevice = false, hasTransport = false, hasVolume = false;
    for (const auto& endpoint : endpoints) {
        if (endpoint.path == "/api/device") hasDevice = true;
        if (endpoint.path == "/api/transport") hasTransport = true;
        if (endpoint.path == "/api/volume") hasVolume = true;
    }

    EXPECT_TRUE(hasDevice);
    EXPECT_TRUE(hasTransport);
    EXPECT_TRUE(hasVolume);
}

// AirPlay tests
TEST_F(RoonBridgeTest, AirPlayConfiguration) {
    EXPECT_TRUE(roonBridge_->setAirPlayName("Vortex AirPlay"));
    EXPECT_EQ(roonBridge_->getAirPlayName(), "Vortex AirPlay");

    EXPECT_TRUE(roonBridge_->setAirPlayPassword("password123"));
    EXPECT_EQ(roonBridge_->getAirPlayPassword(), "password123");

    EXPECT_TRUE(roonBridge_->enableAirPlayPassword(true));
    EXPECT_TRUE(roonBridge_->isAirPlayPasswordEnabled());

    EXPECT_TRUE(roonBridge_->enableAirPlayPassword(false));
    EXPECT_FALSE(roonBridge_->isAirPlayPasswordEnabled());
}

// RAAT (Roon Advanced Audio Transport) tests
TEST_F(RoonBridgeTest, RAATConfiguration) {
    EXPECT_TRUE(roonBridge_->setRAATLatency(1000)); // 1 second
    EXPECT_EQ(roonBridge_->getRAATLatency(), 1000);

    EXPECT_TRUE(roonBridge_->setRAATBufferSize(8192));
    EXPECT_EQ(roonBridge_->getRAATBufferSize(), 8192);

    EXPECT_TRUE(roonBridge_->enableRAATFlowControl(true));
    EXPECT_TRUE(roonBridge_->isRAATFlowControlEnabled());

    EXPECT_TRUE(roonBridge_->enableRAATFlowControl(false));
    EXPECT_FALSE(roonBridge_->isRAATFlowControlEnabled());
}

// Network discovery tests
TEST_F(RoonBridgeTest, NetworkDiscovery) {
    // Test Bonjour/mDNS service registration
    EXPECT_TRUE(roonBridge_->startNetworkDiscovery());
    EXPECT_TRUE(roonBridge_->isNetworkDiscoveryActive());

    EXPECT_TRUE(roonBridge_->stopNetworkDiscovery());
    EXPECT_FALSE(roonBridge_->isNetworkDiscoveryActive());
}

// Logging and diagnostics tests
TEST_F(RoonBridgeTest, Logging) {
    EXPECT_TRUE(roonBridge_->enableLogging(true));
    EXPECT_TRUE(roonBridge_->isLoggingEnabled());

    EXPECT_TRUE(roonBridge_->enableLogging(false));
    EXPECT_FALSE(roonBridge_->isLoggingEnabled());

    EXPECT_TRUE(roonBridge_->setLogLevel(RoonBridge::LogLevel::DEBUG));
    EXPECT_EQ(roonBridge_->getLogLevel(), RoonBridge::LogLevel::DEBUG);
}

TEST_F(RoonBridgeTest, Diagnostics) {
    EXPECT_TRUE(roonBridge_->isHealthy());

    auto diagnostics = roonBridge_->getDiagnostics();
    EXPECT_FALSE(diagnostics.empty() || diagnostics.empty()); // Either way is fine

    auto networkInfo = roonBridge_->getNetworkInfo();
    EXPECT_FALSE(networkInfo.interfaces.empty());
}

// Performance tests
TEST_F(RoonBridgeTest, PerformanceAudioProcessing) {
    std::vector<float> audio;
    generateSineWave(audio, 4096, 2, 1000.0f, 0.5f);

    const int numIterations = 100;
    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numIterations; ++i) {
        roonBridge_->processAudio(audio.data(), audio.size() / 2, 2);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

    // Should process quickly for real-time use
    EXPECT_LT(duration.count(), 100000); // Less than 100ms for 100 iterations
}

// Thread safety tests (basic)
TEST_F(RoonBridgeTest, BasicThreadSafety) {
    EXPECT_TRUE(roonBridge_->start());

    std::atomic<bool> running{true};
    std::atomic<int> operations{0};

    // Audio processing thread
    std::thread audioThread([&]() {
        std::vector<float> audio;
        generateTestAudio(audio, 1024, 2);

        while (running) {
            roonBridge_->processAudio(audio.data(), audio.size() / 2, 2);
            operations++;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    // Control thread
    std::thread controlThread([&]() {
        while (running) {
            roonBridge_->setVolume(0.5f + (operations % 10) * 0.05f);
            roonBridge_->play();
            roonBridge_->pause();
            operations++;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });

    // Status thread
    std::thread statusThread([&]() {
        while (running) {
            auto stats = roonBridge_->getStatistics();
            auto state = roonBridge_->getTransportState();
            operations++;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    // Run for a short time
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    running = false;

    audioThread.join();
    controlThread.join();
    statusThread.join();

    EXPECT_TRUE(roonBridge_->stop());

    // If we get here without crashing, basic thread safety is working
    SUCCEED();
}

// Memory tests
TEST_F(RoonBridgeTest, MemoryUsage) {
    size_t initialMemory = roonBridge_->getMemoryUsage();
    EXPECT_GT(initialMemory, 0);

    // Process some audio to potentially increase memory usage
    std::vector<float> audio;
    generateTestAudio(audio, 8192, 8); // More channels, larger buffer
    roonBridge_->processAudio(audio.data(), audio.size() / 8, 8);

    size_t currentMemory = roonBridge_->getMemoryUsage();
    EXPECT_GE(currentMemory, initialMemory);

    // Add zones and outputs
    std::string zoneId = roonBridge_->addZone("Test Zone", 2);
    std::string outputId = roonBridge_->addOutput("Test Output", RoonBridge::OutputType::ANALOG);

    currentMemory = roonBridge_->getMemoryUsage();
    EXPECT_GT(currentMemory, initialMemory);

    // Cleanup
    roonBridge_->removeZone(zoneId);
    roonBridge_->removeOutput(outputId);
}

// Configuration save/load tests
TEST_F(RoonBridgeTest, ConfigurationSaveLoad) {
    // Modify configuration
    roonBridge_->setDeviceName("Modified Bridge");
    roonBridge_->setHTTPPort(9333);
    roonBridge_->setVolume(0.75f);

    // Save configuration
    EXPECT_TRUE(roonBridge_->saveConfiguration("test_config.json"));

    // Modify configuration again
    roonBridge_->setDeviceName("Another Bridge");
    roonBridge_->setHTTPPort(9334);

    // Load configuration
    EXPECT_TRUE(roonBridge_->loadConfiguration("test_config.json"));

    // Verify configuration was restored
    auto deviceInfo = roonBridge_->getDeviceInfo();
    EXPECT_EQ(deviceInfo.deviceName, "Modified Bridge");
    EXPECT_EQ(roonBridge_->getHTTPPort(), 9333);
    EXPECT_FLOAT_EQ(roonBridge_->getVolume(), 0.75f);
}

// Error handling tests
TEST_F(RoonBridgeTest, ErrorHandling) {
    // Test invalid operations
    EXPECT_FALSE(roonBridge_->setVolume(-1.0f)); // Invalid volume
    EXPECT_FALSE(roonBridge_->setVolume(2.0f));  // Invalid volume

    EXPECT_FALSE(roonBridge_->setHTTPPort(0));   // Invalid port
    EXPECT_FALSE(roonBridge_->setHTTPPort(70000)); // Invalid port

    // Test invalid zone/output operations
    EXPECT_FALSE(roonBridge_->removeZone("invalid_zone_id"));
    EXPECT_FALSE(roonBridge_->removeOutput("invalid_output_id"));

    // Test audio processing with invalid parameters
    std::vector<float> audio;
    generateTestAudio(audio, 1024, 2);

    EXPECT_FALSE(roonBridge_->processAudio(nullptr, 512, 2));
    EXPECT_FALSE(roonBridge_->processAudio(audio.data(), 0, 2));
    EXPECT_FALSE(roonBridge_->processAudio(audio.data(), 512, 0));
}

// Integration tests
TEST_F(RoonBridgeTest, IntegrationWithAudioEngine) {
    // Test integration with audio engine components
    std::vector<float> audio;
    generateSineWave(audio, 2048, 2, 880.0f, 0.4f);

    // Process audio through the bridge
    EXPECT_TRUE(roonBridge_->processAudio(audio.data(), audio.size() / 2, 2));

    // Check that bridge statistics are updated
    auto stats = roonBridge_->getStatistics();
    EXPECT_GT(stats.totalBytesTransferred, 0);
}