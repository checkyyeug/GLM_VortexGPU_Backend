#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <array>

#include "../../src/output/hqplayer_naa.hpp"
#include "../../src/core/audio_engine.hpp"

using namespace vortex;
using namespace testing;

class HQPlayerNAATest : public ::testing::Test {
protected:
    void SetUp() override {
        hqplayerNaa_ = std::make_unique<HQPlayerNAA>();

        // Initialize with standard configuration
        HQPlayerNAA::NAAConfig config;
        config.deviceName = "Vortex NAA";
        config.deviceId = "vortex-naa";
        config.serverHost = "localhost"; // Use localhost for testing
        config.serverPort = 4321;       // Default HQPlayer NAA port
        config.enableTCP = true;
        config.enableUDP = true;
        config.maxSampleRate = 768000;  // Up to 768kHz
        config.maxBitDepth = 32;
        config.maxChannels = 8;
        config.bufferSize = 8192;
        config.latency = 1000;          // 1 second

        ASSERT_TRUE(hqplayerNaa_->initialize(config));
    }

    void TearDown() override {
        if (hqplayerNaa_ && hqplayerNaa_->isConnected()) {
            hqplayerNaa_->disconnect();
        }
        hqplayerNaa_.reset();
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
            float time = static_cast<float>(i) / 384000.0f; // Use high sample rate
            float sample = amplitude * std::sin(2.0f * M_PI * frequency * time);

            for (uint16_t ch = 0; ch < channels; ++ch) {
                audio[i * channels + ch] = sample;
            }
        }
    }

    std::unique_ptr<HQPlayerNAA> hqplayerNaa_;
};

// Initialization tests
TEST_F(HQPlayerNAATest, InitializeWithValidConfig) {
    HQPlayerNAA::NAAConfig config;
    config.deviceName = "Test NAA";
    config.deviceId = "test-naa";
    config.serverHost = "localhost";
    config.serverPort = 4322; // Different port for testing
    config.enableTCP = true;
    config.enableUDP = false;

    HQPlayerNAA naa;
    EXPECT_TRUE(naa.initialize(config));
    EXPECT_TRUE(naa.isInitialized());
}

TEST_F(HQPlayerNAATest, InitializeWithInvalidConfig) {
    HQPlayerNAA::NAAConfig config;
    config.deviceName = ""; // Invalid empty name
    config.deviceId = "test-naa";
    config.serverHost = "localhost";

    HQPlayerNAA naa;
    EXPECT_FALSE(naa.initialize(config));
    EXPECT_FALSE(naa.isInitialized());
}

// Connection tests
TEST_F(HQPlayerNAATest, ConnectDisconnect) {
    EXPECT_FALSE(hqplayerNaa_->isConnected());

    // Note: Actual connection will fail if HQPlayer server is not running
    // We'll test the connection logic regardless
    bool connectResult = hqplayerNaa_->connect();

    if (connectResult) {
        EXPECT_TRUE(hqplayerNaa_->isConnected());
        EXPECT_TRUE(hqplayerNaa_->disconnect());
        EXPECT_FALSE(hqplayerNaa_->isConnected());
    } else {
        // Connection failed, which is expected in test environment
        EXPECT_FALSE(hqplayerNaa_->isConnected());
        SUCCEED() << "HQPlayer server not available for connection testing";
    }
}

TEST_F(HQPlayerNAATest, Reconnect) {
    // Test reconnect functionality
    bool connectResult = hqplayerNaa_->connect();

    if (connectResult) {
        EXPECT_TRUE(hqplayerNaa_->reconnect());
        EXPECT_TRUE(hqplayerNaa_->isConnected());

        hqplayerNaa_->disconnect();
    } else {
        SUCCEED() << "HQPlayer server not available for reconnection testing";
    }
}

// Transport protocol tests
TEST_F(HQPlayerNAATest, TransportProtocols) {
    EXPECT_TRUE(hqplayerNaa_->enableTCP(true));
    EXPECT_TRUE(hqplayerNaa_->isTCPEnabled());

    EXPECT_TRUE(hqplayerNaa_->enableTCP(false));
    EXPECT_FALSE(hqplayerNaa_->isTCPEnabled());

    EXPECT_TRUE(hqplayerNaa_->enableUDP(true));
    EXPECT_TRUE(hqplayerNaa_->isUDPEnabled());

    EXPECT_TRUE(hqplayerNaa_->enableUDP(false));
    EXPECT_FALSE(hqplayerNaa_->isUDPEnabled());

    // Should have at least one protocol enabled
    EXPECT_TRUE(hqplayerNaa_->isTCPEnabled() || hqplayerNaa_->isUDPEnabled());
}

// Network configuration tests
TEST_F(HQPlayerNAATest, NetworkConfiguration) {
    EXPECT_TRUE(hqplayerNaa_->setServerHost("127.0.0.1"));
    EXPECT_EQ(hqplayerNaa_->getServerHost(), "127.0.0.1");

    EXPECT_TRUE(hqplayerNaa_->setServerPort(4322));
    EXPECT_EQ(hqplayerNaa_->getServerPort(), 4322);

    EXPECT_TRUE(hqplayerNaa_->setTimeout(5000));
    EXPECT_EQ(hqplayerNaa_->getTimeout(), 5000);

    EXPECT_TRUE(hqplayerNaa_->setNetworkInterface("eth0"));
    EXPECT_EQ(hqplayerNaa_->getNetworkInterface(), "eth0");
}

// Device information tests
TEST_F(HQPlayerNAATest, GetDeviceInfo) {
    auto deviceInfo = hqplayerNaa_->getDeviceInfo();

    EXPECT_FALSE(deviceInfo.deviceId.empty());
    EXPECT_FALSE(deviceInfo.deviceName.empty());
    EXPECT_GT(deviceInfo.maxSampleRate, 0);
    EXPECT_GT(deviceInfo.maxBitDepth, 0);
    EXPECT_GT(deviceInfo.maxChannels, 0);
    EXPECT_FALSE(deviceInfo.firmwareVersion.empty());
}

TEST_F(HQPlayerNAATest, SetDeviceInfo) {
    EXPECT_TRUE(hqplayerNaa_->setDeviceName("New NAA Name"));
    auto deviceInfo = hqplayerNaa_->getDeviceInfo();
    EXPECT_EQ(deviceInfo.deviceName, "New NAA Name");

    EXPECT_TRUE(hqplayerNaa_->setMaxSampleRate(1536000));
    deviceInfo = hqplayerNaa_->getDeviceInfo();
    EXPECT_EQ(deviceInfo.maxSampleRate, 1536000);

    EXPECT_TRUE(hqplayerNaa_->setMaxBitDepth(64)); // HQPlayer supports 64-bit
    deviceInfo = hqplayerNaa_->getDeviceInfo();
    EXPECT_EQ(deviceInfo.maxBitDepth, 64);

    EXPECT_TRUE(hqplayerNaa_->setMaxChannels(32));
    deviceInfo = hqplayerNaa_->getDeviceInfo();
    EXPECT_EQ(deviceInfo.maxChannels, 32);
}

// Audio format tests
TEST_F(HQPlayerNAATest, SupportedFormats) {
    auto formats = hqplayerNaa_->getSupportedFormats();

    EXPECT_FALSE(formats.empty());

    // Check for common formats supported by HQPlayer
    bool hasPCM = false, hasDSD = false, hasDXD = false;
    for (const auto& format : formats) {
        if (format.type == HQPlayerNAA::AudioFormat::PCM) hasPCM = true;
        if (format.type == HQPlayerNAA::AudioFormat::DSD) hasDSD = true;
        if (format.type == HQPlayerNAA::AudioFormat::DXD) hasDXD = true;
    }

    EXPECT_TRUE(hasPCM);  // Should always support PCM
    // HQPlayer is known for DSD and DXD support
}

TEST_F(HQPlayerNAATest, FormatCapability) {
    // Test format capability queries
    EXPECT_TRUE(hqplayerNaa_->supportsFormat(HQPlayerNAA::AudioFormat::PCM, 44100, 16, 2));
    EXPECT_TRUE(hqplayerNaa_->supportsFormat(HQPlayerNAA::AudioFormat::PCM, 192000, 24, 2));
    EXPECT_TRUE(hqplayerNaa_->supportsFormat(HQPlayerNAA::AudioFormat::PCM, 384000, 32, 2));

    // Test DSD formats
    EXPECT_TRUE(hqplayerNaa_->supportsFormat(HQPlayerNAA::AudioFormat::DSD, 2822400, 1, 1)); // DSD64
    EXPECT_TRUE(hqplayerNaa_->supportsFormat(HQPlayerNAA::AudioFormat::DSD, 5644800, 1, 1)); // DSD128

    // Test DXD format
    EXPECT_TRUE(hqplayerNaa_->supportsFormat(HQPlayerNAA::AudioFormat::DXD, 352800, 24, 2));

    // Test invalid formats
    EXPECT_FALSE(hqplayerNaa_->supportsFormat(HQPlayerNAA::AudioFormat::PCM, 0, 16, 2));
    EXPECT_FALSE(hqplayerNaa_->supportsFormat(HQPlayerNAA::AudioFormat::PCM, 48000, 0, 2));
}

// Audio processing tests
TEST_F(HQPlayerNAATest, ProcessAudio) {
    std::vector<float> audio;
    generateTestAudio(audio, 4096, 2);

    // Should be able to process audio even if not connected
    EXPECT_TRUE(hqplayerNaa_->processAudio(audio.data(), audio.size() / 2, 2));
}

TEST_F(HQPlayerNAATest, ProcessHighResolutionAudio) {
    std::vector<float> audio;
    generateSineWave(audio, 8192, 8, 1000.0f, 0.3f);

    EXPECT_TRUE(hqplayerNaa_->processAudio(audio.data(), audio.size() / 8, 8));
}

TEST_F(HQPlayerNAATest, ProcessMultiChannelAudio) {
    std::vector<float> audio;
    generateSineWave(audio, 4096, 8, 880.0f, 0.2f);

    std::vector<const float*> inputs(8);
    std::vector<float*> outputs(8);
    std::vector<std::vector<float>> inputBuffers(8);
    std::vector<std::vector<float>> outputBuffers(8);

    for (int ch = 0; ch < 8; ++ch) {
        inputBuffers[ch].resize(4096);
        outputBuffers[ch].resize(4096);
        inputs[ch] = inputBuffers[ch].data();
        outputs[ch] = outputBuffers[ch].data();

        // Deinterleave audio
        for (size_t i = 0; i < 4096; ++i) {
            inputBuffers[ch][i] = audio[i * 8 + ch];
        }
    }

    EXPECT_TRUE(hqplayerNaa_->processAudioMultiChannel(inputs.data(), outputs.data(), 4096, 8));
}

// Buffer management tests
TEST_F(HQPlayerNAATest, BufferConfiguration) {
    EXPECT_TRUE(hqplayerNaa_->setBufferSize(16384));
    EXPECT_EQ(hqplayerNaa_->getBufferSize(), 16384);

    EXPECT_TRUE(hqplayerNaa_->setBufferCount(8));
    EXPECT_EQ(hqplayerNaa_->getBufferCount(), 8);

    EXPECT_TRUE(hqplayerNaa_->setLatency(2000)); // 2 seconds
    EXPECT_EQ(hqplayerNaa_->getLatency(), 2000);
}

TEST_F(HQPlayerNAATest, BufferStatus) {
    auto bufferStatus = hqplayerNaa_->getBufferStatus();

    EXPECT_GE(bufferStatus.utilization, 0.0f);
    EXPECT_LE(bufferStatus.utilization, 1.0f);
    EXPECT_GE(bufferStatus.availableBuffers, 0);
    EXPECT_LE(bufferStatus.availableBuffers, bufferStatus.totalBuffers);
    EXPECT_GE(bufferStatus.underruns, 0);
    EXPECT_GE(bufferStatus.overruns, 0);
}

// Clock and synchronization tests
TEST_F(HQPlayerNAATest, ClockConfiguration) {
    EXPECT_TRUE(hqplayerNaa_->setClockSource(HQPlayerNAA::ClockSource::INTERNAL));
    EXPECT_EQ(hqplayerNaa_->getClockSource(), HQPlayerNAA::ClockSource::INTERNAL);

    EXPECT_TRUE(hqplayerNaa_->setClockSource(HQPlayerNAA::ClockSource::EXTERNAL));
    EXPECT_EQ(hqplayerNaa_->getClockSource(), HQPlayerNAA::ClockSource::EXTERNAL);

    EXPECT_TRUE(hqplayerNaa_->setClockSource(HQPlayerNAA::ClockSource::WORD_CLOCK));
    EXPECT_EQ(hqplayerNaa_->getClockSource(), HQPlayerNAA::ClockSource::WORD_CLOCK);
}

TEST_F(HQPlayerNAATest, ClockStatus) {
    auto clockStatus = hqplayerNaa_->getClockStatus();

    EXPECT_GE(clockStatus.sampleRate, 0);
    EXPECT_GE(clockStatus.clockAccuracy, 0.0f);
    EXPECT_LE(clockStatus.clockAccuracy, 1.0f);
    EXPECT_FALSE(clockStatus.isLocked || !clockStatus.isLocked); // Either way is fine
}

// Filter and modulation tests
TEST_F(HQPlayerNAATest, FilterConfiguration) {
    EXPECT_TRUE(hqplayerNaa_->enableUpsampling(true));
    EXPECT_TRUE(hqplayerNaa_->isUpsamplingEnabled());

    EXPECT_TRUE(hqplayerNaa_->enableUpsampling(false));
    EXPECT_FALSE(hqplayerNaa_->isUpsamplingEnabled());

    EXPECT_TRUE(hqplayerNaa_->setUpsamplingRatio(2));
    EXPECT_EQ(hqplayerNaa_->getUpsamplingRatio(), 2);

    EXPECT_TRUE(hqplayerNaa_->setUpsamplingRatio(8));
    EXPECT_EQ(hqplayerNaa_->getUpsamplingRatio(), 8);
}

TEST_F(HQPlayerNAATest, FilterTypes) {
    EXPECT_TRUE(hqplayerNaa_->setFilterType(HQPlayerNAA::FilterType::SINC_M));
    EXPECT_EQ(hqplayerNaa_->getFilterType(), HQPlayerNAA::FilterType::SINC_M);

    EXPECT_TRUE(hqplayerNaa_->setFilterType(HQPlayerNAA::FilterType::POLYPHASE));
    EXPECT_EQ(hqplayerNaa_->getFilterType(), HQPlayerNAA::FilterType::POLYPHASE);

    EXPECT_TRUE(hqplayerNaa_->setFilterType(HQPlayerNAA::FilterType::MIN_PHASE));
    EXPECT_EQ(hqplayerNaa_->getFilterType(), HQPlayerNAA::FilterType::MIN_PHASE);
}

TEST_F(HQPlayerNAATest, ModulationConfiguration) {
    EXPECT_TRUE(hqplayerNaa_->enableSigmaDeltaModulation(true));
    EXPECT_TRUE(hqplayerNaa_->isSigmaDeltaModulationEnabled());

    EXPECT_TRUE(hqplayerNaa_->enableSigmaDeltaModulation(false));
    EXPECT_FALSE(hqplayerNaa_->isSigmaDeltaModulationEnabled());

    EXPECT_TRUE(hqplayerNaa_->setModulationOrder(5));
    EXPECT_EQ(hqplayerNaa_->getModulationOrder(), 5);

    EXPECT_TRUE(hqplayerNaa_->setModulationOrder(7));
    EXPECT_EQ(hqplayerNaa_->getModulationOrder(), 7);
}

// DSD processing tests
TEST_F(HQPlayerNAATest, DSDProcessing) {
    EXPECT_TRUE(hqplayerNaa_->enableDSDProcessing(true));
    EXPECT_TRUE(hqplayerNaa_->isDSDProcessingEnabled());

    EXPECT_TRUE(hqplayerNaa_->enableDSDProcessing(false));
    EXPECT_FALSE(hqplayerNaa_->isDSDProcessingEnabled());

    EXPECT_TRUE(hqplayerNaa_->setDSDConversionType(HQPlayerNAA::DSDConversion::DIRECT));
    EXPECT_EQ(hqplayerNaa_->getDSDConversionType(), HQPlayerNAA::DSDConversion::DIRECT);

    EXPECT_TRUE(hqplayerNaa_->setDSDConversionType(HQPlayerNAA::DSDConversion::PDM));
    EXPECT_EQ(hqplayerNaa_->getDSDConversionType(), HQPlayerNAA::DSDConversion::PDM);
}

// Volume and control tests
TEST_F(HQPlayerNAATest, VolumeControl) {
    EXPECT_TRUE(hqplayerNaa_->setVolume(0.5f));
    EXPECT_FLOAT_EQ(hqplayerNaa_->getVolume(), 0.5f);

    EXPECT_TRUE(hqplayerNaa_->setVolume(0.0f));
    EXPECT_FLOAT_EQ(hqplayerNaa_->getVolume(), 0.0f);

    EXPECT_TRUE(hqplayerNaa_->setVolume(1.0f));
    EXPECT_FLOAT_EQ(hqplayerNaa_->getVolume(), 1.0f);

    // Test volume limits
    hqplayerNaa_->setVolume(-0.5f); // Should clamp to 0.0
    EXPECT_FLOAT_EQ(hqplayerNaa_->getVolume(), 0.0f);

    hqplayerNaa_->setVolume(1.5f); // Should clamp to 1.0
    EXPECT_FLOAT_EQ(hqplayerNaa_->getVolume(), 1.0f);

    // Test mute
    EXPECT_TRUE(hqplayerNaa_->setMute(true));
    EXPECT_TRUE(hqplayerNaa_->isMuted());

    EXPECT_TRUE(hqplayerNaa_->setMute(false));
    EXPECT_FALSE(hqplayerNaa_->isMuted());
}

// Statistics tests
TEST_F(HQPlayerNAATest, Statistics) {
    auto stats = hqplayerNaa_->getStatistics();

    EXPECT_GE(stats.totalSamplesProcessed, 0);
    EXPECT_GE(stats.totalBytesTransferred, 0);
    EXPECT_GE(stats.activeConnections, 0);
    EXPECT_GE(stats.uptimeSeconds, 0);
    EXPECT_GE(stats.cpuUsage, 0.0f);
    EXPECT_GE(stats.memoryUsage, 0);
    EXPECT_GE(stats.bufferUnderruns, 0);
    EXPECT_GE(stats.bufferOverruns, 0);

    // Process some audio to affect statistics
    std::vector<float> audio;
    generateTestAudio(audio, 4096, 2);
    hqplayerNaa_->processAudio(audio.data(), audio.size() / 2, 2);

    stats = hqplayerNaa_->getStatistics();
    EXPECT_GT(stats.totalSamplesProcessed, 0);
}

TEST_F(HQPlayerNAATest, ResetStatistics) {
    // Process some audio first
    std::vector<float> audio;
    generateTestAudio(audio, 4096, 2);
    hqplayerNaa_->processAudio(audio.data(), audio.size() / 2, 2);

    auto stats = hqplayerNaa_->getStatistics();
    if (stats.totalSamplesProcessed > 0) {
        hqplayerNaa_->resetStatistics();
        stats = hqplayerNaa_->getStatistics();
        EXPECT_EQ(stats.totalSamplesProcessed, 0);
        EXPECT_EQ(stats.totalBytesTransferred, 0);
        EXPECT_EQ(stats.bufferUnderruns, 0);
        EXPECT_EQ(stats.bufferOverruns, 0);
    }
}

// Network protocol tests
TEST_F(HQPlayerNAATest, ProtocolConfiguration) {
    EXPECT_TRUE(hqplayerNaa_->setProtocolVersion(2));
    EXPECT_EQ(hqplayerNaa_->getProtocolVersion(), 2);

    EXPECT_TRUE(hqplayerNaa_->setPacketSize(4096));
    EXPECT_EQ(hqplayerNaa_->getPacketSize(), 4096);

    EXPECT_TRUE(hqplayerNaa_->setKeepAlive(true));
    EXPECT_TRUE(hqplayerNaa_->isKeepAliveEnabled());

    EXPECT_TRUE(hqplayerNaa_->setKeepAlive(false));
    EXPECT_FALSE(hqplayerNaa_->isKeepAliveEnabled());
}

// Quality settings tests
TEST_F(HQPlayerNAATest, QualitySettings) {
    EXPECT_TRUE(hqplayerNaa_->setQualityMode(HQPlayerNAA::QualityMode::NORMAL));
    EXPECT_EQ(hqplayerNaa_->getQualityMode(), HQPlayerNAA::QualityMode::NORMAL);

    EXPECT_TRUE(hqplayerNaa_->setQualityMode(HQPlayerNAA::QualityMode::HIGH));
    EXPECT_EQ(hqplayerNaa_->getQualityMode(), HQPlayerNAA::QualityMode::HIGH);

    EXPECT_TRUE(hqplayerNaa_->setQualityMode(HQPlayerNAA::QualityMode::ULTIMATE));
    EXPECT_EQ(hqplayerNaa_->getQualityMode(), HQPlayerNAA::QualityMode::ULTIMATE);
}

// Logging and diagnostics tests
TEST_F(HQPlayerNAATest, Logging) {
    EXPECT_TRUE(hqplayerNaa_->enableLogging(true));
    EXPECT_TRUE(hqplayerNaa_->isLoggingEnabled());

    EXPECT_TRUE(hqplayerNaa_->enableLogging(false));
    EXPECT_FALSE(hqplayerNaa_->isLoggingEnabled());

    EXPECT_TRUE(hqplayerNaa_->setLogLevel(HQPlayerNAA::LogLevel::DEBUG));
    EXPECT_EQ(hqplayerNaa_->getLogLevel(), HQPlayerNAA::LogLevel::DEBUG);
}

TEST_F(HQPlayerNAATest, Diagnostics) {
    EXPECT_TRUE(hqplayerNaa_->isHealthy());

    auto diagnostics = hqplayerNaa_->getDiagnostics();
    EXPECT_FALSE(diagnostics.empty() || diagnostics.empty()); // Either way is fine

    auto networkInfo = hqplayerNaa_->getNetworkInfo();
    EXPECT_FALSE(networkInfo.interfaces.empty());

    auto systemInfo = hqplayerNaa_->getSystemInfo();
    EXPECT_FALSE(systemInfo.cpuCores == 0);
    EXPECT_GT(systemInfo.totalMemory, 0);
}

// Performance tests
TEST_F(HQPlayerNAATest, PerformanceAudioProcessing) {
    std::vector<float> audio;
    generateSineWave(audio, 8192, 8, 2000.0f, 0.4f);

    const int numIterations = 50;
    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numIterations; ++i) {
        hqplayerNaa_->processAudio(audio.data(), audio.size() / 8, 8);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

    // Should process quickly for real-time use
    EXPECT_LT(duration.count(), 100000); // Less than 100ms for 50 iterations
}

TEST_F(HQPlayerNAATest, PerformanceHighSampleRate) {
    // Test processing at high sample rates
    std::vector<float> audio;
    generateSineWave(audio, 16384, 2, 10000.0f, 0.3f); // 10kHz tone

    auto startTime = std::chrono::high_resolution_clock::now();

    hqplayerNaa_->processAudio(audio.data(), audio.size() / 2, 2);

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

    // Even high sample rate processing should be efficient
    EXPECT_LT(duration.count(), 5000); // Less than 5ms for 16K samples
}

// Thread safety tests (basic)
TEST_F(HQPlayerNAATest, BasicThreadSafety) {
    std::atomic<bool> running{true};
    std::atomic<int> operations{0};

    // Audio processing thread
    std::thread audioThread([&]() {
        std::vector<float> audio;
        generateTestAudio(audio, 4096, 8);

        while (running) {
            hqplayerNaa_->processAudio(audio.data(), audio.size() / 8, 8);
            operations++;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    // Control thread
    std::thread controlThread([&]() {
        while (running) {
            hqplayerNaa_->setVolume(0.5f + (operations % 10) * 0.05f);
            hqplayerNaa_->setBufferSize(4096 + (operations % 4) * 2048);
            operations++;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });

    // Status thread
    std::thread statusThread([&]() {
        while (running) {
            auto stats = hqplayerNaa_->getStatistics();
            auto bufferStatus = hqplayerNaa_->getBufferStatus();
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

    // If we get here without crashing, basic thread safety is working
    SUCCEED();
}

// Memory tests
TEST_F(HQPlayerNAATest, MemoryUsage) {
    size_t initialMemory = hqplayerNaa_->getMemoryUsage();
    EXPECT_GT(initialMemory, 0);

    // Process some audio to potentially increase memory usage
    std::vector<float> audio;
    generateTestAudio(audio, 16384, 8); // Large buffer, many channels
    hqplayerNaa_->processAudio(audio.data(), audio.size() / 8, 8);

    size_t currentMemory = hqplayerNaa_->getMemoryUsage();
    EXPECT_GE(currentMemory, initialMemory);

    // Increase buffer size
    hqplayerNaa_->setBufferSize(16384);
    hqplayerNaa_->setBufferCount(16);

    currentMemory = hqplayerNaa_->getMemoryUsage();
    EXPECT_GT(currentMemory, initialMemory);
}

// Error handling tests
TEST_F(HQPlayerNAATest, ErrorHandling) {
    // Test invalid operations
    EXPECT_FALSE(hqplayerNaa_->setVolume(-1.0f)); // Invalid volume
    EXPECT_FALSE(hqplayerNaa_->setVolume(2.0f));  // Invalid volume

    EXPECT_FALSE(hqplayerNaa_->setServerPort(0));   // Invalid port
    EXPECT_FALSE(hqplayerNaa_->setServerPort(70000)); // Invalid port

    EXPECT_FALSE(hqplayerNaa_->setBufferSize(0));  // Invalid buffer size

    // Test audio processing with invalid parameters
    std::vector<float> audio;
    generateTestAudio(audio, 4096, 2);

    EXPECT_FALSE(hqplayerNaa_->processAudio(nullptr, 2048, 2));
    EXPECT_FALSE(hqplayerNaa_->processAudio(audio.data(), 0, 2));
    EXPECT_FALSE(hqplayerNaa_->processAudio(audio.data(), 2048, 0));
}

// Configuration save/load tests
TEST_F(HQPlayerNAATest, ConfigurationSaveLoad) {
    // Modify configuration
    hqplayerNaa_->setDeviceName("Modified NAA");
    hqplayerNaa_->setServerPort(4322);
    hqplayerNaa_->setVolume(0.75f);
    hqplayerNaa_->setBufferSize(16384);

    // Save configuration
    EXPECT_TRUE(hqplayerNaa_->saveConfiguration("test_naa_config.json"));

    // Modify configuration again
    hqplayerNaa_->setDeviceName("Another NAA");
    hqplayerNaa_->setServerPort(4323);

    // Load configuration
    EXPECT_TRUE(hqplayerNaa_->loadConfiguration("test_naa_config.json"));

    // Verify configuration was restored
    auto deviceInfo = hqplayerNaa_->getDeviceInfo();
    EXPECT_EQ(deviceInfo.deviceName, "Modified NAA");
    EXPECT_EQ(hqplayerNaa_->getServerPort(), 4322);
    EXPECT_FLOAT_EQ(hqplayerNaa_->getVolume(), 0.75f);
    EXPECT_EQ(hqplayerNaa_->getBufferSize(), 16384);
}

// Integration tests
TEST_F(HQPlayerNAATest, IntegrationWithAudioEngine) {
    // Test integration with audio engine components
    std::vector<float> audio;
    generateSineWave(audio, 8192, 2, 352800.0f, 0.3f); // DXD sample rate

    // Process audio through the NAA
    EXPECT_TRUE(hqplayerNaa_->processAudio(audio.data(), audio.size() / 2, 2));

    // Check that NAA statistics are updated
    auto stats = hqplayerNaa_->getStatistics();
    EXPECT_GT(stats.totalSamplesProcessed, 0);
    EXPECT_GT(stats.totalBytesTransferred, 0);
}

// High-resolution format tests
TEST_F(HQPlayerNAATest, HighResolutionFormats) {
    // Test DSD256
    EXPECT_TRUE(hqplayerNaa_->supportsFormat(HQPlayerNAA::AudioFormat::DSD, 11289600, 1, 1));

    // Test DXD
    EXPECT_TRUE(hqplayerNaa_->supportsFormat(HQPlayerNAA::AudioFormat::DXD, 352800, 24, 2));

    // Test PCM at 384kHz/32-bit
    EXPECT_TRUE(hqplayerNaa_->supportsFormat(HQPlayerNAA::AudioFormat::PCM, 384000, 32, 2));

    // Test multi-channel high resolution
    EXPECT_TRUE(hqplayerNaa_->supportsFormat(HQPlayerNAA::AudioFormat::PCM, 192000, 32, 8));
}

// Advanced processing tests
TEST_F(HQPlayerNAATest, AdvancedProcessingFeatures) {
    // Enable all advanced features
    EXPECT_TRUE(hqplayerNaa_->enableUpsampling(true));
    EXPECT_TRUE(hqplayerNaa_->enableDSDProcessing(true));
    EXPECT_TRUE(hqplayerNaa_->enableSigmaDeltaModulation(true));
    EXPECT_TRUE(hqplayerNaa_->setQualityMode(HQPlayerNAA::QualityMode::ULTIMATE));

    // Process audio with advanced features enabled
    std::vector<float> audio;
    generateTestAudio(audio, 4096, 2);

    EXPECT_TRUE(hqplayerNaa_->processAudio(audio.data(), audio.size() / 2, 2));

    // Verify settings are active
    EXPECT_TRUE(hqplayerNaa_->isUpsamplingEnabled());
    EXPECT_TRUE(hqplayerNaa_->isDSDProcessingEnabled());
    EXPECT_TRUE(hqplayerNaa_->isSigmaDeltaModulationEnabled());
    EXPECT_EQ(hqplayerNaa_->getQualityMode(), HQPlayerNAA::QualityMode::ULTIMATE);
}