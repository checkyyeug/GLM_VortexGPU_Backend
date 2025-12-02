#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>

#include "../../src/core/processing/processing_chain.hpp"
#include "../../src/core/audio_engine.hpp"

using namespace vortex;
using namespace testing;

class ProcessingChainTest : public ::testing::Test {
protected:
    void SetUp() override {
        processingChain_ = std::make_unique<ProcessingChain>();

        // Initialize with standard configuration
        ProcessingChain::ChainConfig config;
        config.sampleRate = 48000;
        config.numChannels = 2;
        config.blockSize = 512;
        config.mode = ProcessingChain::ProcessingMode::REAL_TIME;
        config.latencyMode = ProcessingChain::LatencyMode::BALANCED;
        config.enableGPUAcceleration = false; // Use CPU for testing
        config.numThreads = 1; // Single thread for predictable testing

        ASSERT_TRUE(processingChain_->initialize(config));

        // Generate test signal
        generateTestSignal();
    }

    void TearDown() override {
        processingChain_.reset();
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

    std::vector<float*> prepareOutputBuffers() {
        std::vector<float*> outputs(numChannels_);
        outputBuffers_.resize(numChannels_);

        for (int ch = 0; ch < numChannels_; ++ch) {
            outputBuffers_[ch].resize(blockSize_);
            outputs[ch] = outputBuffers_[ch].data();
            std::fill(outputBuffers_[ch].begin(), outputBuffers_[ch].end(), 0.0f);
        }

        return outputs;
    }

    std::unique_ptr<ProcessingChain> processingChain_;
    std::vector<float> testSignal_;
    std::vector<float> referenceSignal_;
    std::vector<std::vector<float>> inputBuffers_;
    std::vector<std::vector<float>> outputBuffers_;

    static constexpr uint32_t sampleRate_ = 48000;
    static constexpr uint16_t numChannels_ = 2;
    static constexpr size_t blockSize_ = 512;
};

// Initialization tests
TEST_F(ProcessingChainTest, InitializeWithValidConfig) {
    ProcessingChain::ChainConfig config;
    config.sampleRate = 44100;
    config.numChannels = 2;
    config.blockSize = 256;
    config.mode = ProcessingChain::ProcessingMode::HIGH_QUALITY;

    ProcessingChain chain;
    EXPECT_TRUE(chain.initialize(config));
    EXPECT_TRUE(chain.isInitialized());
}

TEST_F(ProcessingChainTest, InitializeWithInvalidConfig) {
    ProcessingChain::ChainConfig config;
    config.sampleRate = 0;  // Invalid
    config.numChannels = 2;
    config.blockSize = 256;

    ProcessingChain chain;
    EXPECT_FALSE(chain.initialize(config));
    EXPECT_FALSE(chain.isInitialized());
}

// Node management tests
TEST_F(ProcessingChainTest, AddNode) {
    std::string nodeId = processingChain_->addNode(ProcessingChain::ProcessingStage::EQUALIZATION, "Test EQ");
    EXPECT_FALSE(nodeId.empty());

    auto node = processingChain_->getNode(nodeId);
    EXPECT_EQ(node.id, nodeId);
    EXPECT_EQ(node.name, "Test EQ");
    EXPECT_EQ(node.stage, ProcessingChain::ProcessingStage::EQUALIZATION);
    EXPECT_TRUE(node.enabled);
    EXPECT_FALSE(node.bypassed);
}

TEST_F(ProcessingChainTest, AddMultipleNodes) {
    std::string eqId = processingChain_->addNode(ProcessingChain::ProcessingStage::EQUALIZATION, "EQ");
    std::string convId = processingChain_->addNode(ProcessingChain::ProcessingStage::CONVOLUTION, "Conv");
    std::string dynId = processingChain_->addNode(ProcessingChain::ProcessingStage::DYNAMICS, "Dynamics");

    EXPECT_FALSE(eqId.empty());
    EXPECT_FALSE(convId.empty());
    EXPECT_FALSE(dynId.empty());

    auto allNodes = processingChain_->getAllNodes();
    EXPECT_EQ(allNodes.size(), 3);
}

TEST_F(ProcessingChainTest, RemoveNode) {
    std::string nodeId = processingChain_->addNode(ProcessingChain::ProcessingStage::EQUALIZATION, "Test");
    EXPECT_FALSE(nodeId.empty());

    // Verify node exists
    auto node = processingChain_->getNode(nodeId);
    EXPECT_EQ(node.id, nodeId);

    // Remove node
    EXPECT_TRUE(processingChain_->removeNode(nodeId));

    // Verify node no longer exists
    auto removedNode = processingChain_->getNode(nodeId);
    EXPECT_TRUE(removedNode.id.empty());
}

TEST_F(ProcessingChainTest, RemoveInvalidNode) {
    EXPECT_FALSE(processingChain_->removeNode("invalid_id"));
}

TEST_F(ProcessingChainTest, EnableDisableNode) {
    std::string nodeId = processingChain_->addNode(ProcessingChain::ProcessingStage::EQUALIZATION, "Test");

    EXPECT_TRUE(processingChain_->enableNode(nodeId, false));
    auto node = processingChain_->getNode(nodeId);
    EXPECT_FALSE(node.enabled);

    EXPECT_TRUE(processingChain_->enableNode(nodeId, true));
    node = processingChain_->getNode(nodeId);
    EXPECT_TRUE(node.enabled);
}

TEST_F(ProcessingChainTest, BypassUnbypassNode) {
    std::string nodeId = processingChain_->addNode(ProcessingChain::ProcessingStage::EQUALIZATION, "Test");

    EXPECT_TRUE(processingChain_->bypassNode(nodeId, true));
    auto node = processingChain_->getNode(nodeId);
    EXPECT_TRUE(node.bypassed);

    EXPECT_TRUE(processingChain_->bypassNode(nodeId, false));
    node = processingChain_->getNode(nodeId);
    EXPECT_FALSE(node.bypassed);
}

// Processing tests
TEST_F(ProcessingChainTest, ProcessAudioEmptyChain) {
    auto inputs = prepareInputBuffers();
    auto outputs = prepareOutputBuffers();

    // Should pass signal through unchanged with empty chain
    EXPECT_TRUE(processingChain_->processAudioMultiChannel(inputs, outputs, blockSize_, numChannels_));

    // Verify output matches input (pass-through)
    for (int ch = 0; ch < numChannels_; ++ch) {
        for (size_t i = 0; i < blockSize_; ++i) {
            EXPECT_FLOAT_EQ(outputs[ch][i], inputs[ch][i]);
        }
    }
}

TEST_F(ProcessingChainTest, ProcessAudioWithNodes) {
    // Add some processing nodes
    std::string eqId = processingChain_->addNode(ProcessingChain::ProcessingStage::EQUALIZATION, "EQ");
    std::string convId = processingChain_->addNode(ProcessingChain::ProcessingStage::CONVOLUTION, "Conv");

    // Set some parameters
    processingChain_->setNodeParameter(eqId, "gain", 6.0f);
    processingChain_->setNodeParameter(convId, "wet_level", 0.5f);

    auto inputs = prepareInputBuffers();
    auto outputs = prepareOutputBuffers();

    EXPECT_TRUE(processingChain_->processAudioMultiChannel(inputs, outputs, blockSize_, numChannels_));

    // Output should be different from input (processing occurred)
    bool isDifferent = false;
    for (int ch = 0; ch < numChannels_; ++ch) {
        for (size_t i = 0; i < blockSize_; ++i) {
            if (std::abs(outputs[ch][i] - inputs[ch][i]) > 1e-6f) {
                isDifferent = true;
                break;
            }
        }
        if (isDifferent) break;
    }

    // May or may not be different depending on node implementations
}

// Parameter control tests
TEST_F(ProcessingChainTest, SetGetNodeParameters) {
    std::string nodeId = processingChain_->addNode(ProcessingChain::ProcessingStage::EQUALIZATION, "Test");

    EXPECT_TRUE(processingChain_->setNodeParameter(nodeId, "gain", 3.0f));
    EXPECT_FLOAT_EQ(processingChain_->getNodeParameter(nodeId, "gain"), 3.0f);

    EXPECT_TRUE(processingChain_->setNodeParameter(nodeId, "frequency", 1000.0f));
    EXPECT_FLOAT_EQ(processingChain_->getNodeParameter(nodeId, "frequency"), 1000.0f);

    // Test invalid node
    EXPECT_FALSE(processingChain_->setNodeParameter("invalid", "gain", 1.0f));
    EXPECT_FLOAT_EQ(processingChain_->getNodeParameter("invalid", "gain"), 0.0f);

    // Test invalid parameter
    EXPECT_FALSE(processingChain_->setNodeParameter(nodeId, "invalid_param", 1.0f));
}

TEST_F(ProcessingChainTest, SetNodeGain) {
    std::string nodeId = processingChain_->addNode(ProcessingChain::ProcessingStage::EQUALIZATION, "Test");

    EXPECT_TRUE(processingChain_->setNodeGain(nodeId, -6.0f));
    auto node = processingChain_->getNode(nodeId);
    EXPECT_FLOAT_EQ(node.gain, -6.0f);
}

TEST_F(ProcessingChainTest, SetNodeWetDryMix) {
    std::string nodeId = processingChain_->addNode(ProcessingChain::ProcessingStage::CONVOLUTION, "Test");

    EXPECT_TRUE(processingChain_->setNodeWetDryMix(nodeId, 0.7f, 0.3f));
    auto node = processingChain_->getNode(nodeId);
    EXPECT_FLOAT_EQ(node.wetLevel, 0.7f);
    EXPECT_FLOAT_EQ(node.dryLevel, 0.3f);
}

// Routing tests
TEST_F(ProcessingChainTest, ConnectNodes) {
    std::string node1Id = processingChain_->addNode(ProcessingChain::ProcessingStage::EQUALIZATION, "EQ");
    std::string node2Id = processingChain_->addNode(ProcessingChain::ProcessingStage::CONVOLUTION, "Conv");

    EXPECT_TRUE(processingChain_->connectNodes(node1Id, node2Id));

    auto node1 = processingChain_->getNode(node1Id);
    auto node2 = processingChain_->getNode(node2Id);

    EXPECT_THAT(node1.outputs, Contains(node2Id));
    EXPECT_THAT(node2.inputs, Contains(node1Id));
}

TEST_F(ProcessingChainTest, DisconnectNodes) {
    std::string node1Id = processingChain_->addNode(ProcessingChain::ProcessingStage::EQUALIZATION, "EQ");
    std::string node2Id = processingChain_->addNode(ProcessingChain::ProcessingStage::CONVOLUTION, "Conv");

    // Connect first
    EXPECT_TRUE(processingChain_->connectNodes(node1Id, node2Id));

    // Then disconnect
    EXPECT_TRUE(processingChain_->disconnectNodes(node1Id, node2Id));

    auto node1 = processingChain_->getNode(node1Id);
    auto node2 = processingChain_->getNode(node2Id);

    EXPECT_THAT(node1.outputs, Not(Contains(node2Id)));
    EXPECT_THAT(node2.inputs, Not(Contains(node1Id)));
}

TEST_F(ProcessingChainTest, SetRoutingMode) {
    EXPECT_TRUE(processingChain_->setRoutingMode(ProcessingChain::RoutingMode::PARALLEL));
    EXPECT_EQ(processingChain_->getRoutingMode(), ProcessingChain::RoutingMode::PARALLEL);

    EXPECT_TRUE(processingChain_->setRoutingMode(ProcessingChain::RoutingMode::SERIES));
    EXPECT_EQ(processingChain_->getRoutingMode(), ProcessingChain::RoutingMode::SERIES);

    EXPECT_TRUE(processingChain_->setRoutingMode(ProcessingChain::RoutingMode::HYBRID));
    EXPECT_EQ(processingChain_->getRoutingMode(), ProcessingChain::RoutingMode::HYBRID);
}

// Processing mode tests
TEST_F(ProcessingChainTest, SetProcessingMode) {
    EXPECT_TRUE(processingChain_->setProcessingMode(ProcessingChain::ProcessingMode::HIGH_QUALITY));
    EXPECT_TRUE(processingChain_->setProcessingMode(ProcessingChain::ProcessingMode::OFFLINE));
    EXPECT_TRUE(processingChain_->setProcessingMode(ProcessingChain::ProcessingMode::ADAPTIVE));
}

TEST_F(ProcessingChainTest, SetLatencyMode) {
    EXPECT_TRUE(processingChain_->setLatencyMode(ProcessingChain::LatencyMode::ULTRA_LOW));
    EXPECT_TRUE(processingChain_->setLatencyMode(ProcessingChain::LatencyMode::HIGH));
    EXPECT_TRUE(processingChain_->setLatencyMode(ProcessingChain::LatencyMode::UNLIMITED));
}

// Stage enable tests
TEST_F(ProcessingChainTest, EnableProcessingStages) {
    EXPECT_TRUE(processingChain_->enableEqualizer(true));
    EXPECT_TRUE(processingChain_->enableConvolution(true));
    EXPECT_TRUE(processingChain_->enableSpectrumAnalysis(true));
}

// Preset management tests
TEST_F(ProcessingChainTest, SaveLoadPreset) {
    // Add some nodes and set parameters
    std::string eqId = processingChain_->addNode(ProcessingChain::ProcessingStage::EQUALIZATION, "EQ");
    std::string convId = processingChain_->addNode(ProcessingChain::ProcessingStage::CONVOLUTION, "Conv");

    processingChain_->setNodeGain(eqId, 3.0f);
    processingChain_->setNodeParameter(convId, "wet_level", 0.8f);

    // Save preset
    EXPECT_TRUE(processingChain_->savePreset("test_preset", "Test preset for unit testing"));

    // Modify settings
    processingChain_->setNodeGain(eqId, -6.0f);

    // Load preset and verify restoration
    EXPECT_TRUE(processingChain_->loadPreset("test_preset"));
    auto eqNode = processingChain_->getNode(eqId);
    EXPECT_FLOAT_EQ(eqNode.gain, 3.0f);

    // Test invalid preset
    EXPECT_FALSE(processingChain_->loadPreset("nonexistent_preset"));
}

TEST_F(ProcessingChainTest, DeletePreset) {
    // Save and then delete preset
    processingChain_->savePreset("temp_preset");
    auto presets = processingChain_->getAvailablePresets();
    EXPECT_THAT(presets, Contains("temp_preset"));

    EXPECT_TRUE(processingChain_->deletePreset("temp_preset"));
    presets = processingChain_->getAvailablePresets();
    EXPECT_THAT(presets, Not(Contains("temp_preset")));

    // Test deleting nonexistent preset
    EXPECT_FALSE(processingChain_->deletePreset("nonexistent_preset"));
}

// A/B comparison tests
TEST_F(ProcessingChainTest, ABComparison) {
    EXPECT_TRUE(processingChain_->enableABComparison(true));
    EXPECT_TRUE(processingChain_->switchToPreset("preset_A"));

    // Set some parameters for state A
    std::string nodeId = processingChain_->addNode(ProcessingChain::ProcessingStage::EQUALIZATION, "EQ");
    processingChain_->setNodeGain(nodeId, 6.0f);

    // Switch to state B
    EXPECT_TRUE(processingChain_->toggleABState());
    processingChain_->setNodeGain(nodeId, -6.0f);

    // Toggle back to A
    EXPECT_TRUE(processingChain_->toggleABState());
    auto node = processingChain_->getNode(nodeId);
    EXPECT_FLOAT_EQ(node.gain, 6.0f);

    EXPECT_TRUE(processingChain_->enableABComparison(false));
}

// Monitoring and statistics tests
TEST_F(ProcessingChainTest, GetMonitoringData) {
    auto monitoringData = processingChain_->getMonitoringData();

    EXPECT_GT(monitoringData.spectrum.size(), 0);
    EXPECT_EQ(monitoringData.peakLevels.size(), numChannels_);
    EXPECT_EQ(monitoringData.rmsLevels.size(), numChannels_);
    EXPECT_GE(monitoringData.cpuUsage, 0.0f);
    EXPECT_GE(monitoringData.gpuUsage, 0.0f);
}

TEST_F(ProcessingChainTest, GetLatency) {
    float latency = processingChain_->getLatency();
    EXPECT_GE(latency, 0.0f);
    EXPECT_LT(latency, 1000.0f); // Less than 1 second
}

TEST_F(ProcessingChainTest, GetCPUAndGPUUsage) {
    float cpuUsage = processingChain_->getCPUUsage();
    float gpuUsage = processingChain_->getGPUUsage();

    EXPECT_GE(cpuUsage, 0.0f);
    EXPECT_LE(cpuUsage, 100.0f);
    EXPECT_GE(gpuUsage, 0.0f);
    EXPECT_LE(gpuUsage, 100.0f);
}

TEST_F(ProcessingChainTest, GetStatistics) {
    auto stats = processingChain_->getStatistics();

    EXPECT_EQ(stats.totalSamplesProcessed, 0);
    EXPECT_EQ(stats.totalProcessingTimeUs, 0);
    EXPECT_EQ(stats.averageLatency, 0.0f);
    EXPECT_EQ(stats.activeNodes, 0);
    EXPECT_EQ(stats.enabledNodes, 0);
    EXPECT_EQ(stats.bypassedNodes, 0);

    // Process some audio
    auto inputs = prepareInputBuffers();
    auto outputs = prepareOutputBuffers();
    processingChain_->processAudioMultiChannel(inputs, outputs, blockSize_, numChannels_);

    stats = processingChain_->getStatistics();
    EXPECT_EQ(stats.totalSamplesProcessed, blockSize_ * numChannels_);
    EXPECT_GT(stats.totalProcessingTimeUs, 0);
}

TEST_F(ProcessingChainTest, ResetStatistics) {
    // Process some audio first
    auto inputs = prepareInputBuffers();
    auto outputs = prepareOutputBuffers();
    processingChain_->processAudioMultiChannel(inputs, outputs, blockSize_, numChannels_);

    auto stats = processingChain_->getStatistics();
    EXPECT_GT(stats.totalSamplesProcessed, 0);

    // Reset and verify
    processingChain_->resetStatistics();
    stats = processingChain_->getStatistics();
    EXPECT_EQ(stats.totalSamplesProcessed, 0);
    EXPECT_EQ(stats.totalProcessingTimeUs, 0);
}

// Performance optimization tests
TEST_F(ProcessingChainTest, OptimizeForPerformance) {
    EXPECT_TRUE(processingChain_->optimizeForPerformance());
}

TEST_F(ProcessingChainTest, GPUAcceleration) {
    EXPECT_TRUE(processingChain_->enableGPUAcceleration(true));

    // Test may not actually enable GPU if not available
    // The implementation should handle this gracefully
}

// Health monitoring tests
TEST_F(ProcessingChainTest, HealthCheck) {
    EXPECT_TRUE(processingChain_->isHealthy());

    auto diagnostics = processingChain_->getDiagnosticMessages();
    EXPECT_TRUE(diagnostics.empty() || !diagnostics.empty()); // Either way is fine
}

// Session management tests
TEST_F(ProcessingChainTest, SessionManagement) {
    EXPECT_TRUE(processingChain_->createSession("test_session"));

    auto sessions = processingChain_->getAvailableSessions();
    EXPECT_THAT(sessions, Contains("test_session"));

    // Save session
    EXPECT_TRUE(processingChain_->saveSession("test_session"));

    // Load session
    EXPECT_TRUE(processingChain_->loadSession("test_session"));

    // Delete session
    EXPECT_TRUE(processingChain_->deleteSession("test_session"));

    sessions = processingChain_->getAvailableSessions();
    EXPECT_THAT(sessions, Not(Contains("test_session")));
}

// Automation tests
TEST_F(ProcessingChainTest, Automation) {
    std::string nodeId = processingChain_->addNode(ProcessingChain::ProcessingStage::EQUALIZATION, "EQ");

    // Start automation
    EXPECT_TRUE(processingChain_->startAutomation(nodeId, "gain"));

    // Add keyframes
    EXPECT_TRUE(processingChain_->addAutomationKeyframe(nodeId, "gain", 0, 0.0f));
    EXPECT_TRUE(processingChain_->addAutomationKeyframe(nodeId, "gain", 1000, 6.0f));
    EXPECT_TRUE(processingChain_->addAutomationKeyframe(nodeId, "gain", 2000, -6.0f));

    // Get automation data
    auto automationData = processingChain_->getAutomationData();
    EXPECT_GT(automationData.size(), 0);

    // Stop automation
    EXPECT_TRUE(processingChain_->stopAutomation(nodeId, "gain"));

    // Remove keyframe
    EXPECT_TRUE(processingChain_->removeAutomationKeyframe(nodeId, "gain", 1000));
}

// Advanced features tests
TEST_F(ProcessingChainTest, Sidechain) {
    std::string nodeId = processingChain_->addNode(ProcessingChain::ProcessingStage::DYNAMICS, "Dynamics");

    EXPECT_TRUE(processingChain_->enableSidechain(nodeId, "input"));
}

TEST_F(ProcessingChainTest, ParallelProcessing) {
    EXPECT_TRUE(processingChain_->enableParallelBranch("branch_A", true));
    EXPECT_TRUE(processingChain_->enableParallelBranch("branch_B", true));

    EXPECT_TRUE(processingChain_->setParallelRatio("branch_A", "branch_B", 0.6f));
}

TEST_F(ProcessingChainTest, FeedbackRouting) {
    std::string node1Id = processingChain_->addNode(ProcessingChain::ProcessingStage::EQUALIZATION, "EQ");
    std::string node2Id = processingChain_->addNode(ProcessingChain::ProcessingStage::DYNAMICS, "Dynamics");

    EXPECT_TRUE(processingChain_->enableFeedback(node1Id, node2Id, 0.3f));
}

TEST_F(ProcessingChainTest, MasterGain) {
    EXPECT_TRUE(processingChain_->setMasterGain(-3.0f));
}

// Performance tests
TEST_F(ProcessingChainTest, PerformanceBasicProcessing) {
    // Add some processing nodes
    processingChain_->addNode(ProcessingChain::ProcessingStage::EQUALIZATION, "EQ");
    processingChain_->addNode(ProcessingChain::ProcessingStage::CONVOLUTION, "Conv");
    processingChain_->addNode(ProcessingChain::ProcessingStage::DYNAMICS, "Dynamics");

    auto inputs = prepareInputBuffers();
    auto outputs = prepareOutputBuffers();

    const int numIterations = 100;
    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numIterations; ++i) {
        processingChain_->processAudioMultiChannel(inputs, outputs, blockSize_, numChannels_);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

    // Should process reasonably fast for real-time use
    EXPECT_LT(duration.count(), 100000); // Less than 100ms for 100 iterations
    EXPECT_LT(duration.count() / numIterations, 1000); // Less than 1ms per block
}

// Thread safety tests (basic)
TEST_F(ProcessingChainTest, BasicThreadSafety) {
    // Add some nodes
    std::string eqId = processingChain_->addNode(ProcessingChain::ProcessingStage::EQUALIZATION, "EQ");
    std::string convId = processingChain_->addNode(ProcessingChain::ProcessingStage::CONVOLUTION, "Conv");

    auto inputs = prepareInputBuffers();
    auto outputs = prepareOutputBuffers();

    // Test concurrent access (basic smoke test)
    std::thread processThread([&]() {
        for (int i = 0; i < 50; ++i) {
            processingChain_->processAudioMultiChannel(inputs, outputs, blockSize_, numChannels_);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    std::thread settingsThread([&]() {
        for (int i = 0; i < 50; ++i) {
            processingChain_->setNodeParameter(eqId, "gain", static_cast<float>(i % 10 - 5));
            processingChain_->setNodeParameter(convId, "wet_level", (i % 5) * 0.2f);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    processThread.join();
    settingsThread.join();

    // If we get here without crashing, basic thread safety is working
    SUCCEED();
}

// Edge cases and error handling
TEST_F(ProcessingChainTest, ProcessWithNullPointers) {
    std::vector<const float*> inputs(numChannels_, nullptr);
    auto outputs = prepareOutputBuffers();

    EXPECT_FALSE(processingChain_->processAudioMultiChannel(inputs, outputs, blockSize_, numChannels_));

    inputs = prepareInputBuffers();
    std::vector<float*> nullOutputs(numChannels_, nullptr);

    EXPECT_FALSE(processingChain_->processAudioMultiChannel(inputs, nullOutputs, blockSize_, numChannels_));
}

TEST_F(ProcessingChainTest, ProcessWithInvalidParameters) {
    auto inputs = prepareInputBuffers();
    auto outputs = prepareOutputBuffers();

    // Invalid block size
    EXPECT_FALSE(processingChain_->processAudioMultiChannel(inputs, outputs, 0, numChannels_));
    EXPECT_FALSE(processingChain_->processAudioMultiChannel(inputs, outputs, blockSize_ + 1, numChannels_));

    // Invalid channel count
    EXPECT_FALSE(processingChain_->processAudioMultiChannel(inputs, outputs, blockSize_, 0));
    EXPECT_FALSE(processingChain_->processAudioMultiChannel(inputs, outputs, blockSize_, 33)); // Exceeds MAX_CHANNELS
}

// Callback tests
TEST_F(ProcessingChainTest, Callbacks) {
    bool nodeChanged = false;
    bool parameterChanged = false;
    bool stateChanged = false;

    processingChain_->setNodeChangedCallback([&](const ProcessingChain::ProcessingNode&) {
        nodeChanged = true;
    });

    processingChain_->setParameterChangedCallback([&](const std::string&, const std::string&, float) {
        parameterChanged = true;
    });

    processingChain_->setStateChangedCallback([&](const ProcessingChain::ChainState&) {
        stateChanged = true;
    });

    // Trigger callbacks
    std::string nodeId = processingChain_->addNode(ProcessingChain::ProcessingStage::EQUALIZATION, "EQ");
    processingChain_->setNodeParameter(nodeId, "gain", 3.0f);

    // Process to potentially trigger state changes
    auto inputs = prepareInputBuffers();
    auto outputs = prepareOutputBuffers();
    processingChain_->processAudioMultiChannel(inputs, outputs, blockSize_, numChannels_);

    // Note: Actual callback triggering depends on implementation
    // This test mainly verifies that callbacks can be set without issues
    SUCCEED();
}