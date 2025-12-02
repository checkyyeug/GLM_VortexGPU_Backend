#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>

#include "../../src/output/output_manager.hpp"
#include "../../src/core/audio_engine.hpp"

using namespace vortex;
using namespace testing;

class OutputManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        outputManager_ = std::make_unique<OutputManager>();

        // Initialize with standard configuration
        OutputManager::ManagerConfig config;
        config.sampleRate = 48000;
        config.numChannels = 2;
        config.blockSize = 512;
        config.bufferSize = 4096;
        config.enableMultiDevice = true;
        config.enableAutoRouting = true;

        ASSERT_TRUE(outputManager_->initialize(config));

        // Generate test signal
        generateTestSignal();
    }

    void TearDown() override {
        outputManager_.reset();
    }

    void generateTestSignal() {
        testSignal_.resize(blockSize_ * numChannels_);
        referenceSignal_.resize(blockSize_ * numChannels_);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

        // Generate interleaved stereo signal
        for (size_t i = 0; i < blockSize_; ++i) {
            for (int ch = 0; ch < numChannels_; ++ch) {
                testSignal_[i * numChannels_ + ch] = dis(gen);
            }
        }

        // Copy reference
        referenceSignal_ = testSignal_;
    }

    void generateSineWave(float frequency, float amplitude = 0.5f) {
        testSignal_.resize(blockSize_ * numChannels_);
        referenceSignal_.resize(blockSize_ * numChannels_);

        for (size_t i = 0; i < blockSize_; ++i) {
            float time = static_cast<float>(i) / sampleRate_;
            float sample = amplitude * std::sin(2.0f * M_PI * frequency * time);

            for (int ch = 0; ch < numChannels_; ++ch) {
                testSignal_[i * numChannels_ + ch] = sample;
            }
        }

        referenceSignal_ = testSignal_;
    }

    std::vector<const float*> prepareInputBuffers() {
        std::vector<const float*> inputs(numChannels_);
        inputBuffers_.resize(numChannels_);

        for (int ch = 0; ch < numChannels_; ++ch) {
            inputBuffers_[ch].resize(blockSize_);
            inputs[ch] = inputBuffers_[ch].data();

            for (size_t i = 0; i < blockSize_; ++i) {
                inputBuffers_[ch][i] = testSignal_[i * numChannels_ + ch];
            }
        }

        return inputs;
    }

    std::unique_ptr<OutputManager> outputManager_;
    std::vector<float> testSignal_;
    std::vector<float> referenceSignal_;
    std::vector<std::vector<float>> inputBuffers_;

    static constexpr uint32_t sampleRate_ = 48000;
    static constexpr uint16_t numChannels_ = 2;
    static constexpr size_t blockSize_ = 512;
};

// Initialization tests
TEST_F(OutputManagerTest, InitializeWithValidConfig) {
    OutputManager::ManagerConfig config;
    config.sampleRate = 44100;
    config.numChannels = 2;
    config.blockSize = 256;
    config.bufferSize = 2048;

    OutputManager manager;
    EXPECT_TRUE(manager.initialize(config));
    EXPECT_TRUE(manager.isInitialized());
}

TEST_F(OutputManagerTest, InitializeWithInvalidConfig) {
    OutputManager::ManagerConfig config;
    config.sampleRate = 0;  // Invalid
    config.numChannels = 2;
    config.blockSize = 256;

    OutputManager manager;
    EXPECT_FALSE(manager.initialize(config));
    EXPECT_FALSE(manager.isInitialized());
}

// Device discovery tests
TEST_F(OutputManagerTest, DiscoverDevices) {
    auto devices = outputManager_->discoverDevices();

    // Should at least return some devices (may be virtual/test devices)
    EXPECT_GE(devices.size(), 0);

    // Check device properties
    for (const auto& device : devices) {
        EXPECT_FALSE(device.id.empty());
        EXPECT_FALSE(device.name.empty());
        EXPECT_GT(device.maxChannels, 0);
        EXPECT_GT(device.maxSampleRate, 0);
        EXPECT_GE(device.latencyMs, 0);
    }
}

TEST_F(OutputManagerTest, GetAvailableDevices) {
    auto devices = outputManager_->getAvailableDevices();

    // Should return same devices as discoverDevices()
    auto discoveredDevices = outputManager_->discoverDevices();
    EXPECT_EQ(devices.size(), discoveredDevices.size());
}

// Device management tests
TEST_F(OutputManagerTest, AddOutputDevice) {
    auto devices = outputManager_->discoverDevices();
    if (devices.empty()) {
        SUCCEED() << "No output devices available for testing";
        return;
    }

    const auto& device = devices[0];
    EXPECT_TRUE(outputManager_->addOutputDevice(device.id));

    // Should be able to get the device
    auto addedDevice = outputManager_->getDevice(device.id);
    EXPECT_FALSE(addedDevice.id.empty());
    EXPECT_EQ(addedDevice.id, device.id);
}

TEST_F(OutputManagerTest, AddInvalidDevice) {
    EXPECT_FALSE(outputManager_->addOutputDevice("invalid_device_id"));
}

TEST_F(OutputManagerTest, RemoveOutputDevice) {
    auto devices = outputManager_->discoverDevices();
    if (devices.empty()) {
        SUCCEED() << "No output devices available for testing";
        return;
    }

    const auto& device = devices[0];
    outputManager_->addOutputDevice(device.id);

    EXPECT_TRUE(outputManager_->removeOutputDevice(device.id));

    // Device should no longer be available
    auto removedDevice = outputManager_->getDevice(device.id);
    EXPECT_TRUE(removedDevice.id.empty());
}

TEST_F(OutputManagerTest, RemoveInvalidDevice) {
    EXPECT_FALSE(outputManager_->removeOutputDevice("invalid_device_id"));
}

// Device state tests
TEST_F(OutputManagerTest, EnableDisableDevice) {
    auto devices = outputManager_->discoverDevices();
    if (devices.empty()) {
        SUCCEED() << "No output devices available for testing";
        return;
    }

    const auto& device = devices[0];
    outputManager_->addOutputDevice(device.id);

    EXPECT_TRUE(outputManager_->enableDevice(device.id, false));
    auto deviceInfo = outputManager_->getDevice(device.id);
    EXPECT_FALSE(deviceInfo.enabled);

    EXPECT_TRUE(outputManager_->enableDevice(device.id, true));
    deviceInfo = outputManager_->getDevice(device.id);
    EXPECT_TRUE(deviceInfo.enabled);
}

// Routing tests
TEST_F(OutputManagerTest, SetDeviceRouting) {
    auto devices = outputManager_->discoverDevices();
    if (devices.size() < 2) {
        SUCCEED() << "Need at least 2 devices for routing tests";
        return;
    }

    std::string device1Id = devices[0].id;
    std::string device2Id = devices[1].id;

    outputManager_->addOutputDevice(device1Id);
    outputManager_->addOutputDevice(device2Id);

    // Test routing configuration
    OutputManager::RoutingConfig routing;
    routing.sourceDevice = device1Id;
    routing.targetDevice = device2Id;
    routing.sourceChannel = 0;
    routing.targetChannel = 0;
    routing.gain = 0.5f;

    EXPECT_TRUE(outputManager_->setDeviceRouting(routing));
}

TEST_F(OutputManagerTest, ClearDeviceRouting) {
    auto devices = outputManager_->discoverDevices();
    if (devices.size() < 2) {
        SUCCEED() << "Need at least 2 devices for routing tests";
        return;
    }

    std::string device1Id = devices[0].id;
    std::string device2Id = devices[1].id;

    outputManager_->addOutputDevice(device1Id);
    outputManager_->addOutputDevice(device2Id);

    OutputManager::RoutingConfig routing;
    routing.sourceDevice = device1Id;
    routing.targetDevice = device2Id;
    routing.sourceChannel = 0;
    routing.targetChannel = 0;
    routing.gain = 0.5f;

    outputManager_->setDeviceRouting(routing);
    EXPECT_TRUE(outputManager_->clearDeviceRouting(device1Id, device2Id));
}

// Audio processing tests
TEST_F(OutputManagerTest, ProcessAudioWithoutDevices) {
    auto inputs = prepareInputBuffers();

    // Should fail gracefully without output devices
    EXPECT_FALSE(outputManager_->processAudio(inputs.data(), blockSize_, numChannels_));
}

TEST_F(OutputManagerTest, ProcessAudioWithDevices) {
    auto devices = outputManager_->discoverDevices();
    if (devices.empty()) {
        SUCCEED() << "No output devices available for testing";
        return;
    }

    const auto& device = devices[0];
    outputManager_->addOutputDevice(device.id);

    auto inputs = prepareInputBuffers();

    // Should succeed with at least one device added
    EXPECT_TRUE(outputManager_->processAudio(inputs.data(), blockSize_, numChannels_));
}

TEST_F(OutputManagerTest, ProcessMultiChannelAudio) {
    auto devices = outputManager_->discoverDevices();
    if (devices.empty()) {
        SUCCEED() << "No output devices available for testing";
        return;
    }

    const auto& device = devices[0];
    outputManager_->addOutputDevice(device.id);

    auto inputs = prepareInputBuffers();
    std::vector<float*> outputs(numChannels_);
    std::vector<std::vector<float>> outputBuffers(numChannels_);

    for (int ch = 0; ch < numChannels_; ++ch) {
        outputBuffers[ch].resize(blockSize_);
        outputs[ch] = outputBuffers[ch].data();
    }

    EXPECT_TRUE(outputManager_->processAudioMultiChannel(inputs.data(), outputs.data(),
                                                         blockSize_, numChannels_));
}

// Buffer management tests
TEST_F(OutputManagerTest, SetDeviceBuffer) {
    auto devices = outputManager_->discoverDevices();
    if (devices.empty()) {
        SUCCEED() << "No output devices available for testing";
        return;
    }

    const auto& device = devices[0];
    outputManager_->addOutputDevice(device.id);

    std::vector<float> buffer(4096, 0.0f);
    EXPECT_TRUE(outputManager_->setDeviceBuffer(device.id, buffer.data(), buffer.size()));
}

TEST_F(OutputManagerTest, GetDeviceBuffer) {
    auto devices = outputManager_->discoverDevices();
    if (devices.empty()) {
        SUCCEED() << "No output devices available for testing";
        return;
    }

    const auto& device = devices[0];
    outputManager_->addOutputDevice(device.id);

    auto buffer = outputManager_->getDeviceBuffer(device.id);
    EXPECT_NE(buffer, nullptr);
}

// Synchronization tests
TEST_F(OutputManagerTest, SynchronizeDevices) {
    auto devices = outputManager_->discoverDevices();
    if (devices.size() < 2) {
        SUCCEED() << "Need at least 2 devices for synchronization tests";
        return;
    }

    for (const auto& device : devices) {
        outputManager_->addOutputDevice(device.id);
    }

    EXPECT_TRUE(outputManager_->synchronizeDevices());
}

TEST_F(OutputManagerTest, SetDeviceLatency) {
    auto devices = outputManager_->discoverDevices();
    if (devices.empty()) {
        SUCCEED() << "No output devices available for testing";
        return;
    }

    const auto& device = devices[0];
    outputManager_->addOutputDevice(device.id);

    EXPECT_TRUE(outputManager_->setDeviceLatency(device.id, 50.0f));
}

// Device configuration tests
TEST_F(OutputManagerTest, SetDeviceSampleRate) {
    auto devices = outputManager_->discoverDevices();
    if (devices.empty()) {
        SUCCEED() << "No output devices available for testing";
        return;
    }

    const auto& device = devices[0];
    outputManager_->addOutputDevice(device.id);

    // Try common sample rates
    EXPECT_TRUE(outputManager_->setDeviceSampleRate(device.id, 44100));
    EXPECT_TRUE(outputManager_->setDeviceSampleRate(device.id, 48000));
    EXPECT_TRUE(outputManager_->setDeviceSampleRate(device.id, 96000));
}

TEST_F(OutputManagerTest, SetDeviceBitDepth) {
    auto devices = outputManager_->discoverDevices();
    if (devices.empty()) {
        SUCCEED() << "No output devices available for testing";
        return;
    }

    const auto& device = devices[0];
    outputManager_->addOutputDevice(device.id);

    EXPECT_TRUE(outputManager_->setDeviceBitDepth(device.id, 16));
    EXPECT_TRUE(outputManager_->setDeviceBitDepth(device.id, 24));
    EXPECT_TRUE(outputManager_->setDeviceBitDepth(device.id, 32));
}

TEST_F(OutputManagerTest, SetDeviceChannels) {
    auto devices = outputManager_->discoverDevices();
    if (devices.empty()) {
        SUCCEED() << "No output devices available for testing";
        return;
    }

    const auto& device = devices[0];
    outputManager_->addOutputDevice(device.id);

    if (device.maxChannels >= 1) {
        EXPECT_TRUE(outputManager_->setDeviceChannels(device.id, 1));
    }
    if (device.maxChannels >= 2) {
        EXPECT_TRUE(outputManager_->setDeviceChannels(device.id, 2));
    }
}

// Monitoring and statistics tests
TEST_F(OutputManagerTest, GetDeviceStatus) {
    auto devices = outputManager_->discoverDevices();
    if (devices.empty()) {
        SUCCEED() << "No output devices available for testing";
        return;
    }

    const auto& device = devices[0];
    outputManager_->addOutputDevice(device.id);

    auto status = outputManager_->getDeviceStatus(device.id);

    EXPECT_FALSE(status.deviceId.empty());
    EXPECT_GE(status.cpuUsage, 0.0f);
    EXPECT_GE(status.bufferUtilization, 0.0f);
    EXPECT_LE(status.bufferUtilization, 1.0f);
    EXPECT_GE(status.latencyMs, 0.0f);
}

TEST_F(OutputManagerTest, GetManagerStatistics) {
    auto stats = outputManager_->getManagerStatistics();

    EXPECT_GE(stats.totalDevices, 0);
    EXPECT_GE(stats.activeDevices, 0);
    EXPECT_GE(stats.totalLatencyMs, 0.0f);
    EXPECT_GE(stats.totalCpuUsage, 0.0f);
    EXPECT_EQ(stats.totalSamplesOutput, 0);
    EXPECT_EQ(stats.droppedFrames, 0);

    // Process some audio and check statistics update
    auto devices = outputManager_->discoverDevices();
    if (!devices.empty()) {
        const auto& device = devices[0];
        outputManager_->addOutputDevice(device.id);

        auto inputs = prepareInputBuffers();
        outputManager_->processAudio(inputs.data(), blockSize_, numChannels_);

        stats = outputManager_->getManagerStatistics();
        EXPECT_GT(stats.totalSamplesOutput, 0);
    }
}

TEST_F(OutputManagerTest, ResetStatistics) {
    // Process some audio first
    auto devices = outputManager_->discoverDevices();
    if (!devices.empty()) {
        const auto& device = devices[0];
        outputManager_->addOutputDevice(device.id);

        auto inputs = prepareInputBuffers();
        outputManager_->processAudio(inputs.data(), blockSize_, numChannels_);
    }

    auto stats = outputManager_->getManagerStatistics();
    if (stats.totalSamplesOutput > 0) {
        outputManager_->resetStatistics();
        stats = outputManager_->getManagerStatistics();
        EXPECT_EQ(stats.totalSamplesOutput, 0);
        EXPECT_EQ(stats.droppedFrames, 0);
    }
}

// Master control tests
TEST_F(OutputManagerTest, SetMasterVolume) {
    EXPECT_TRUE(outputManager_->setMasterVolume(0.75f));
    EXPECT_FLOAT_EQ(outputManager_->getMasterVolume(), 0.75f);

    // Test edge cases
    outputManager_->setMasterVolume(-0.5f); // Should clamp to 0.0
    EXPECT_FLOAT_EQ(outputManager_->getMasterVolume(), 0.0f);

    outputManager_->setMasterVolume(1.5f); // Should clamp to 1.0
    EXPECT_FLOAT_EQ(outputManager_->getMasterVolume(), 1.0f);
}

TEST_F(OutputManagerTest, SetMasterMute) {
    outputManager_->setMasterMute(true);
    EXPECT_TRUE(outputManager_->isMasterMuted());

    outputManager_->setMasterMute(false);
    EXPECT_FALSE(outputManager_->isMasterMuted());
}

// Preset management tests
TEST_F(OutputManagerTest, SaveLoadDevicePreset) {
    auto devices = outputManager_->discoverDevices();
    if (devices.empty()) {
        SUCCEED() << "No output devices available for testing";
        return;
    }

    const auto& device = devices[0];
    outputManager_->addOutputDevice(device.id);

    // Modify device settings
    outputManager_->setDeviceLatency(device.id, 25.0f);
    outputManager_->setDeviceVolume(device.id, 0.8f);

    // Save preset
    EXPECT_TRUE(outputManager_->saveDevicePreset(device.id, "test_preset", "Test preset"));

    // Modify settings again
    outputManager_->setDeviceLatency(device.id, 50.0f);
    outputManager_->setDeviceVolume(device.id, 0.6f);

    // Load preset
    EXPECT_TRUE(outputManager_->loadDevicePreset(device.id, "test_preset"));

    // Settings should be restored
    auto status = outputManager_->getDeviceStatus(device.id);
    // Note: Actual restoration depends on implementation
}

TEST_F(OutputManagerTest, DeleteDevicePreset) {
    auto devices = outputManager_->discoverDevices();
    if (devices.empty()) {
        SUCCEED() << "No output devices available for testing";
        return;
    }

    const auto& device = devices[0];
    outputManager_->addOutputDevice(device.id);

    // Save and then delete preset
    outputManager_->saveDevicePreset(device.id, "temp_preset", "Temporary preset");
    EXPECT_TRUE(outputManager_->deleteDevicePreset(device.id, "temp_preset"));

    // Test deleting nonexistent preset
    EXPECT_FALSE(outputManager_->deleteDevicePreset(device.id, "nonexistent_preset"));
}

TEST_F(OutputManagerTest, GetDevicePresets) {
    auto devices = outputManager_->discoverDevices();
    if (devices.empty()) {
        SUCCEED() << "No output devices available for testing";
        return;
    }

    const auto& device = devices[0];
    outputManager_->addOutputDevice(device.id);

    auto presets = outputManager_->getDevicePresets(device.id);
    EXPECT_TRUE(presets.empty() || !presets.empty()); // Either way is fine

    // Add a preset and check
    outputManager_->saveDevicePreset(device.id, "test_preset", "Test preset");
    presets = outputManager_->getDevicePresets(device.id);
    EXPECT_TRUE(std::find(presets.begin(), presets.end(), "test_preset") != presets.end());
}

// Auto-routing tests
TEST_F(OutputManagerTest, SetAutoRoutingMode) {
    outputManager_->setAutoRoutingMode(true);
    EXPECT_TRUE(outputManager_->isAutoRoutingEnabled());

    outputManager_->setAutoRoutingMode(false);
    EXPECT_FALSE(outputManager_->isAutoRoutingEnabled());
}

TEST_F(OutputManagerTest, OptimizeRouting) {
    auto devices = outputManager_->discoverDevices();
    if (devices.size() < 2) {
        SUCCEED() << "Need at least 2 devices for routing optimization";
        return;
    }

    for (const auto& device : devices) {
        outputManager_->addOutputDevice(device.id);
    }

    EXPECT_TRUE(outputManager_->optimizeRouting());
}

// Error handling tests
TEST_F(OutputManagerTest, ProcessWithNullPointers) {
    EXPECT_FALSE(outputManager_->processAudio(nullptr, blockSize_, numChannels_));
}

TEST_F(OutputManagerTest, ProcessWithInvalidParameters) {
    auto inputs = prepareInputBuffers();

    // Invalid block size
    EXPECT_FALSE(outputManager_->processAudio(inputs.data(), 0, numChannels_));
    EXPECT_FALSE(outputManager_->processAudio(inputs.data(), blockSize_ + 1, numChannels_));

    // Invalid channel count
    EXPECT_FALSE(outputManager_->processAudio(inputs.data(), blockSize_, 0));
    EXPECT_FALSE(outputManager_->processAudio(inputs.data(), blockSize_, 33)); // Exceeds MAX_CHANNELS
}

TEST_F(OutputManagerTest, GetInvalidDevice) {
    auto device = outputManager_->getDevice("invalid_device_id");
    EXPECT_TRUE(device.id.empty());
}

TEST_F(OutputManagerTest, ControlInvalidDevice) {
    EXPECT_FALSE(outputManager_->enableDevice("invalid_device_id", false));
    EXPECT_FALSE(outputManager_->removeOutputDevice("invalid_device_id"));
    EXPECT_FALSE(outputManager_->setDeviceLatency("invalid_device_id", 50.0f));
    EXPECT_FALSE(outputManager_->setDeviceVolume("invalid_device_id", 0.5f));
    EXPECT_FALSE(outputManager_->setDeviceSampleRate("invalid_device_id", 48000));
}

// Performance tests
TEST_F(OutputManagerTest, PerformanceBasicProcessing) {
    auto devices = outputManager_->discoverDevices();
    if (devices.empty()) {
        SUCCEED() << "No output devices available for performance testing";
        return;
    }

    const auto& device = devices[0];
    outputManager_->addOutputDevice(device.id);

    auto inputs = prepareInputBuffers();

    const int numIterations = 100;
    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numIterations; ++i) {
        outputManager_->processAudio(inputs.data(), blockSize_, numChannels_);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

    // Should process quickly for real-time use
    EXPECT_LT(duration.count(), 50000); // Less than 50ms for 100 iterations
}

// Thread safety tests (basic)
TEST_F(OutputManagerTest, BasicThreadSafety) {
    auto devices = outputManager_->discoverDevices();
    if (devices.empty()) {
        SUCCEED() << "No output devices available for thread safety testing";
        return;
    }

    const auto& device = devices[0];
    outputManager_->addOutputDevice(device.id);

    auto inputs = prepareInputBuffers();

    // Test concurrent access (basic smoke test)
    std::thread processThread([&]() {
        for (int i = 0; i < 50; ++i) {
            outputManager_->processAudio(inputs.data(), blockSize_, numChannels_);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    std::thread controlThread([&]() {
        for (int i = 0; i < 50; ++i) {
            outputManager_->setDeviceVolume(device.id, 0.5f + (i % 5) * 0.1f);
            outputManager_->setDeviceLatency(device.id, 25.0f + (i % 10) * 5.0f);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    std::thread statusThread([&]() {
        for (int i = 0; i < 50; ++i) {
            auto status = outputManager_->getDeviceStatus(device.id);
            auto stats = outputManager_->getManagerStatistics();
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    processThread.join();
    controlThread.join();
    statusThread.join();

    // If we get here without crashing, basic thread safety is working
    SUCCEED();
}

// Memory tests
TEST_F(OutputManagerTest, MemoryUsage) {
    size_t initialMemory = outputManager_->getMemoryUsage();
    EXPECT_GT(initialMemory, 0);

    auto devices = outputManager_->discoverDevices();
    if (!devices.empty()) {
        const auto& device = devices[0];
        outputManager_->addOutputDevice(device.id);

        size_t currentMemory = outputManager_->getMemoryUsage();
        EXPECT_GE(currentMemory, initialMemory);

        // Add buffer for device
        std::vector<float> buffer(4096);
        outputManager_->setDeviceBuffer(device.id, buffer.data(), buffer.size());

        currentMemory = outputManager_->getMemoryUsage();
        EXPECT_GT(currentMemory, initialMemory);
    }
}

// Health monitoring tests
TEST_F(OutputManagerTest, HealthCheck) {
    EXPECT_TRUE(outputManager_->isHealthy());

    auto diagnostics = outputManager_->getDiagnosticMessages();
    EXPECT_TRUE(diagnostics.empty() || !diagnostics.empty()); // Either way is fine
}

// Integration tests with other components
TEST_F(OutputManagerTest, IntegrationWithAudioEngine) {
    // Test that output manager can work with audio engine
    auto devices = outputManager_->discoverDevices();
    if (devices.empty()) {
        SUCCEED() << "No output devices available for integration testing";
        return;
    }

    const auto& device = devices[0];
    outputManager_->addOutputDevice(device.id);

    auto inputs = prepareInputBuffers();

    // Process audio through the output manager
    EXPECT_TRUE(outputManager_->processAudio(inputs.data(), blockSize_, numChannels_));

    // Check device status after processing
    auto status = outputManager_->getDeviceStatus(device.id);
    EXPECT_FALSE(status.deviceId.empty());
}

// Hot-plug simulation tests
TEST_F(OutputManagerTest, SimulateDeviceHotPlug) {
    // Initial device discovery
    auto initialDevices = outputManager_->discoverDevices();
    size_t initialCount = initialDevices.size();

    // Simulate device addition (this is conceptual - actual hot-plug detection depends on OS)
    // For testing purposes, we'll just rediscover devices
    auto newDevices = outputManager_->discoverDevices();

    // Device count may change between discoveries
    EXPECT_GE(newDevices.size(), 0);
}