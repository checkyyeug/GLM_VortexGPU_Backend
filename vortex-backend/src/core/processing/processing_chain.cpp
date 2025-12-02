#include "processing_chain.hpp"
#include "../../utils/logger.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <thread>
#include <future>

namespace vortex {

// ProcessingChain Implementation
ProcessingChain::ProcessingChain() {
    Logger::info("ProcessingChain: Initializing modular audio processing chain");

    // Initialize state
    state_.startTime = std::chrono::steady_clock::now();
    state_.lastUpdateTime = state_.startTime;

    // Initialize statistics
    statistics_.startTime = std::chrono::steady_clock::now();
    lastUpdateTime_ = statistics_.startTime;
}

ProcessingChain::~ProcessingChain() {
    shutdown();
    Logger::info("ProcessingChain: Modular audio processing chain destroyed");
}

bool ProcessingChain::initialize(const ChainConfig& config) {
    if (initialized_.load()) {
        Logger::warn("ProcessingChain: Already initialized");
        return true;
    }

    Logger::info("ProcessingChain: Initializing with {} Hz, {} channels, {} sample block size",
                 config.sampleRate, config.numChannels, config.blockSize);

    config_ = config;

    try {
        // Initialize processing components
        equalizer_ = std::make_unique<Equalizer>();
        if (!equalizer_->initialize(Equalizer::EqualizerConfig{
            .numBands = 512,
            .sampleRate = config.sampleRate,
            .numChannels = config.numChannels,
            .bufferSize = config.blockSize,
            .enableGPUAcceleration = config.enableGPUAcceleration,
            .enableMultiThreading = config.enableMultiThreading,
            .maxLatencyMs = config.maxLatencyMs / 4.0f, // EQ gets 25% of total latency budget
            .enableHighPrecision = config.enableHighPrecision
        })) {
            setError("Failed to initialize equalizer");
            return false;
        }

        convolution_ = std::make_unique<ConvolutionEngine>();
        if (!convolution_->initialize(ConvolutionEngine::ConvolutionConfig{
            .maxIRLength = 16777216,     // 16M points
            .sampleRate = config.sampleRate,
            .numChannels = config.numChannels,
            .blockSize = config.blockSize,
            .mode = ConvolutionEngine::ProcessingMode::REAL_TIME,
            .latencyMode = ConvolutionEngine::LatencyMode::BALANCED,
            .enableGPUAcceleration = config.enableGPUAcceleration,
            .maxLatencyMs = config.maxLatencyMs / 2.0f, // Convolution gets 50% of latency budget
            .enableHighPrecision = config.enableHighPrecision,
            .numThreads = config.numThreads
        })) {
            setError("Failed to initialize convolution engine");
            return false;
        }

        spectrumAnalyzer_ = std::make_unique<SpectrumAnalyzer>();
        if (!spectrumAnalyzer_->initialize(SpectrumAnalyzer::SpectrumConfig{
            .fftSize = 8192,
            .sampleRate = config.sampleRate,
            .numChannels = config.numChannels,
            .enableGPUAcceleration = config.enableGPUAcceleration,
            .enableRealTime = true,
            .updateRate = 60
        })) {
            setError("Failed to initialize spectrum analyzer");
            return false;
        }

        // Initialize processing buffers
        initializeBuffers();

        // Create default processing chain
        createDefaultChain();

        // Enable multi-threading if configured
        if (config.numThreads > 0 || config.numThreads == 0) {
            enableMultiThreading();
        }

        // Start background threads
        processingActive_.store(true);
        processingThread_ = std::make_unique<std::thread>(&ProcessingChain::processingThread, this);
        monitoringThread_ = std::make_unique<std::thread>(&ProcessingChain::monitoringThread, this);

        if (config.enableAutomation) {
            automationActive_.store(true);
            automationThread_ = std::make_unique<std::thread>(&ProcessingChain::automationThread, this);
        }

        initialized_.store(true);
        Logger::info("ProcessingChain: Initialization complete");
        return true;

    } catch (const std::exception& e) {
        setError("Initialization failed: " + std::string(e.what()));
        return false;
    }
}

void ProcessingChain::shutdown() {
    if (!initialized_.load()) {
        return;
    }

    Logger::info("ProcessingChain: Shutting down modular processing chain");

    processingActive_.store(false);
    automationActive_.store(false);

    // Wake up threads
    // (Need to add condition variables for proper thread synchronization)

    // Wait for threads to finish
    if (processingThread_ && processingThread_->joinable()) {
        processingThread_->join();
    }
    if (monitoringThread_ && monitoringThread_->joinable()) {
        monitoringThread_->join();
    }
    if (automationThread_ && automationThread_->joinable()) {
        automationThread_->join();
    }

    // Stop processing components
    if (equalizer_) {
        equalizer_->shutdown();
    }
    if (convolution_) {
        convolution_->shutdown();
    }
    if (spectrumAnalyzer_) {
        spectrumAnalyzer_->shutdown();
    }

    // Wait for worker threads
    for (auto& thread : workerThreads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    workerThreads_.clear();

    // Clear state
    {
        std::lock_guard<std::mutex> lock(nodesMutex_);
        nodes_.clear();
        processingOrder_.clear();
        routingGraph_.clear();
    }

    {
        std::lock_guard<std::mutex> lock(automationMutex_);
        automationData_.clear();
    }

    initialized_.store(false);
    Logger::info("ProcessingChain: Shutdown complete");
}

bool ProcessingChain::isInitialized() const {
    return initialized_.load();
}

bool ProcessingChain::processAudio(const float* input, float* output, size_t numSamples) {
    if (!initialized_.load() || !input || !output || !processingActive_.load()) {
        return false;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    // Process through modular chain
    bool success = processAudioMultiChannel({&input}, {&output}, numSamples, 1);

    // Update statistics
    if (success) {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<float, std::milli>(endTime - startTime);

        std::lock_guard<std::mutex> lock(statsMutex_);
        statistics_.totalSamplesProcessed += numSamples;
        statistics_.totalProcessingTimeUs += static_cast<uint64_t>(duration.count() * 1000);
        statistics_.averageLatency = (statistics_.averageLatency * 0.9f) + (duration.count() * 0.1f);
        statistics_.maxLatency = std::max(statistics_.maxLatency, duration.count());
        statistics_.minLatency = std::min(statistics_.minLatency, duration.count());
        statistics_.lastActivity = std::chrono::steady_clock::now();
    }

    return success;
}

bool ProcessingChain::processAudioMultiChannel(const std::vector<const float*>& inputs,
                                              std::vector<float*>& outputs,
                                              size_t numSamples, uint16_t channels) {
    if (!initialized_.load() || inputs.empty() || outputs.empty() ||
        inputs.size() != outputs.size() || channels == 0) {
        return false;
    }

    try {
        // Buffer management
        std::lock_guard<std::mutex> lock(buffersMutex_);
        size_t totalSamples = numSamples * channels;

        if (buffers_.inputBuffer.size() < totalSamples) {
            buffers_.inputBuffer.resize(totalSamples);
            buffers_.outputBuffer.resize(totalSamples);
            buffers_.intermediateBuffer.resize(totalSamples);
            buffers_.nodeInputs.resize(nodes_.size());
            buffers_.nodeOutputs.resize(nodes_.size());

            for (auto& inputBuffer : buffers_.nodeInputs) {
                inputBuffer.resize(totalSamples);
            }
            for (auto& outputBuffer : buffers_.nodeOutputs) {
                outputBuffer.resize(totalSamples);
            }
        }

        // Copy input to main buffer (interleaved)
        for (uint16_t ch = 0; ch < channels && ch < inputs.size(); ++ch) {
            if (inputs[ch]) {
                for (size_t i = 0; i < numSamples; ++i) {
                    buffers_.inputBuffer[i * channels + ch] = inputs[ch][i];
                }
            }
        }

        // Process through routing graph
        processRoutingGraph(inputs, outputs, numSamples, channels);

        // Update state
        state_.isProcessing = true;
        state_.currentPosition += numSamples;
        updateState();

        // Call parameter changed callback if needed
        if (parameterChangedCallback_) {
            // This would be called when parameters change during processing
        }

        return true;

    } catch (const std::exception& e) {
        setError("Audio processing failed: " + std::string(e.what()));
        return false;
    }
}

std::string ProcessingChain::addNode(ProcessingStage stage, const std::string& name) {
    std::lock_guard<std::mutex> lock(nodesMutex_);

    if (nodes_.size() >= MAX_NODES) {
        setError("Maximum number of nodes reached");
        return "";
    }

    std::string nodeId = generateNodeId();

    ProcessingNode node;
    node.id = nodeId;
    node.name = name.empty() ? getNodeStageName(stage) : name;
    node.stage = stage;
    node.enabled = true;
    node.bypassed = false;
    node.wetLevel = 1.0f;
    node.dryLevel = 0.0f;
    node.gain = 0.0f;
    node.phase = 0.0f;
    node.latencySamples = calculateNodeLatency(stage);
    node.lastProcessTime = std::chrono::steady_clock::now();

    // Initialize node-specific parameters
    initializeNodeParameters(node);

    nodes_[nodeId] = node;
    updateProcessingOrder();

    // Initialize node buffers
    size_t bufferSize = config_.blockSize * config_.numChannels;
    buffers_.nodeInputs[processingOrder_.size()].resize(bufferSize);
    buffers_.nodeOutputs[processingOrder_.size()].resize(bufferSize);

    Logger::info("ProcessingChain: Added node '{}' ({})", node.name, nodeId);

    if (nodeChangedCallback_) {
        nodeChangedCallback_(node);
    }

    return nodeId;
}

bool ProcessingChain::setNodeParameter(const std::string& nodeId, const std::string& parameter, float value) {
    std::lock_guard<std::mutex> lock(nodesMutex_);
    auto it = nodes_.find(nodeId);
    if (it == nodes_.end()) {
        setError("Node not found: " + nodeId);
        return false;
    }

    ProcessingNode& node = it->second;
    float oldValue = node.parameters[parameter];
    node.parameters[parameter] = value;

    // Handle specific parameters
    if (parameter == "gain") {
        node.gain = value;
    } else if (parameter == "wetLevel") {
        node.wetLevel = std::clamp(value, 0.0f, 2.0f);
    } else if (parameter == "dryLevel") {
        node.dryLevel = std::clamp(value, 0.0f, 2.0f);
    }

    // Update component-specific parameters
    updateNodeComponent(node, parameter, value);

    if (parameterChangedCallback_ && oldValue != value) {
        parameterChangedCallback_(nodeId, parameter, value);
    }

    return true;
}

bool ProcessingChain::enableEqualizer(bool enabled) {
    // Find equalizer node
    std::string equalizerId;
    {
        std::lock_guard<std::mutex> lock(nodesMutex_);
        for (const auto& [id, node] : nodes_) {
            if (node.stage == ProcessingStage::EQUALIZATION) {
                equalizerId = id;
                break;
            }
        }
    }

    if (equalizerId.empty()) {
        // Create equalizer node if it doesn't exist
        equalizerId = addNode(ProcessingStage::EQUALIZATION, "Equalizer");
    }

    return enableNode(equalizerId, enabled);
}

bool ProcessingChain::setEqualizerBandGain(uint32_t band, float gain) {
    if (!equalizer_) {
        return false;
    }

    return equalizer_->setBandGain(band, gain);
}

bool ProcessingChain::enableConvolution(bool enabled) {
    // Find convolution node
    std::string convolutionId;
    {
        std::lock_guard<std::mutex> lock(nodesMutex_);
        for (const auto& [id, node] : nodes_) {
            if (node.stage == ProcessingStage::CONVOLUTION) {
                convolutionId = id;
                break;
            }
        }
    }

    if (convolutionId.empty()) {
        // Create convolution node if it doesn't exist
        convolutionId = addNode(ProcessingStage::CONVOLUTION, "Convolution");
    }

    return enableNode(convolutionId, enabled);
}

bool ProcessingChain::loadImpulseResponse(const std::string& filePath) {
    if (!convolution_) {
        return false;
    }

    // Enable convolution node first
    if (!enableConvolution(true)) {
        return false;
    }

    return convolution_->loadImpulseResponse(filePath);
}

std::vector<float> ProcessingChain::getCurrentSpectrum() const {
    if (spectrumAnalyzer_) {
        return spectrumAnalyzer_->getCurrentSpectrum();
    }
    return {};
}

ProcessingChain::MonitoringData ProcessingChain::getMonitoringData() const {
    MonitoringData data;

    // Get spectrum data
    if (spectrumAnalyzer_) {
        data.spectrum = spectrumAnalyzer_->getCurrentSpectrum();
    }

    // Get level data
    {
        std::lock_guard<std::mutex> lock(buffersMutex_);
        data.peakLevels.resize(config_.numChannels, 0.0f);
        data.rmsLevels.resize(config_.numChannels, 0.0f);

        // Calculate levels from output buffer
        for (uint16_t ch = 0; ch < config_.numChannels; ++ch) {
            float peak = 0.0f;
            float sum = 0.0f;

            for (size_t i = ch; i < buffers_.outputBuffer.size(); i += config_.numChannels) {
                float sample = std::abs(buffers_.outputBuffer[i]);
                peak = std::max(peak, sample);
                sum += sample * sample;
            }

            data.peakLevels[ch] = peak;
            data.rmsLevels[ch] = std::sqrt(sum / (buffers_.outputBuffer.size() / config_.numChannels));
        }
    }

    // Get performance data
    data.cpuUsage = state_.cpuUsage;
    data.gpuUsage = state_.gpuUsage;
    data.latency = state_.averageLatency;
    data.samplesProcessed = state_.currentPosition;
    data.timestamp = std::chrono::steady_clock::now();

    return data;
}

float ProcessingChain::getLatency() const {
    return state_.averageLatency;
}

float ProcessingChain::getCPUUsage() const {
    return state_.cpuUsage;
}

float ProcessingChain::getGPUUsage() const {
    return state_.gpuUsage;
}

ProcessingChain::ChainStatistics ProcessingChain::getStatistics() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    return statistics_;
}

void ProcessingChain::resetStatistics() {
    std::lock_guard<std::mutex> lock(statsMutex_);
    statistics_ = ChainStatistics{};
    statistics_.startTime = std::chrono::steady_clock::now();
    statistics_.lastActivity = statistics_.startTime;
    statistics_.minLatency = 1000.0f;
}

bool ProcessingChain::isHealthy() const {
    if (!initialized_.load() || !processingActive_.load()) {
        return false;
    }

    auto stats = getStatistics();
    return (stats.averageLatency < config_.maxLatencyMs &&
            stats.totalHarmonicDistortion < 0.01f &&  // < 1% THD
            stats.droppedFrames == 0 &&
            state_.cpuUsage < config_.maxCPUUsage);
}

// Private methods implementation
void ProcessingChain::createDefaultChain() {
    // Create default processing nodes in order
    addNode(ProcessingStage::PRE_PROCESSING, "Input Pre-Processing");
    addNode(ProcessingStage::EQUALIZATION, "512-Band Equalizer");
    addNode(ProcessingStage::CONVOLUTION, "Convolution Engine");
    addNode(ProcessingStage::DYNAMICS, "Dynamics Processing");
    addNode(ProcessingStage::POST_PROCESSING, "Output Post-Processing");

    // Set up default routing
    updateProcessingOrder();
}

void ProcessingChain::processRoutingGraph(const std::vector<const float*>& inputs,
                                         std::vector<float*>& outputs,
                                         size_t numSamples, uint16_t channels) {
    std::vector<ProcessingNode*> activeNodes = getActiveNodes();

    if (activeNodes.empty()) {
        // No processing, just copy input to output
        for (uint16_t ch = 0; ch < channels && ch < inputs.size() && ch < outputs.size(); ++ch) {
            if (inputs[ch] && outputs[ch]) {
                std::copy(inputs[ch], inputs[ch] + numSamples, outputs[ch]);
            }
        }
        return;
    }

    // Initialize output with input
    std::copy(buffers_.inputBuffer.begin(), buffers_.inputBuffer.end(), buffers_.outputBuffer.begin());

    // Process nodes in order
    for (ProcessingNode* node : activeNodes) {
        size_t nodeIndex = std::distance(nodes_.begin(), nodes_.find(node->id));

        // Process node with interleaved data
        processNode(*node, buffers_.inputBuffer.data(), buffers_.outputBuffer.data(),
                   numSamples * channels, channels);

        // Update node statistics
        node->samplesProcessed += numSamples * channels;
        node->lastProcessTime = std::chrono::steady_clock::now();

        // Update levels
        float peak = 0.0f;
        float sum = 0.0f;
        for (size_t i = 0; i < buffers_.outputBuffer.size(); ++i) {
            float sample = std::abs(buffers_.outputBuffer[i]);
            peak = std::max(peak, sample);
            sum += sample * sample;
        }
        node->peakLevel = peak;
        node->averageLevel = std::sqrt(sum / buffers_.outputBuffer.size());
    }

    // Deinterleave output to individual channels
    for (uint16_t ch = 0; ch < channels && ch < outputs.size(); ++ch) {
        if (outputs[ch]) {
            for (size_t i = 0; i < numSamples; ++i) {
                outputs[ch][i] = buffers_.outputBuffer[i * channels + ch];
            }
        }
    }
}

void ProcessingChain::processNode(ProcessingNode& node, const float* input, float* output,
                                  size_t numSamples, uint16_t channels) {
    if (!node.enabled || node.bypassed) {
        // Node is bypassed, just copy input to output
        std::copy(input, input + numSamples, output);
        return;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    // Apply wet/dry mix
    if (node.wetLevel != 1.0f || node.dryLevel != 0.0f) {
        for (size_t i = 0; i < numSamples; ++i) {
            output[i] = input[i] * node.dryLevel + input[i] * node.wetLevel;
        }
        input = output; // Use mixed signal as input for processing
    }

    // Process based on node type
    switch (node.stage) {
        case ProcessingStage::EQUALIZATION:
            if (equalizer_) {
                equalizer_->processAudio(input, output, numSamples);
            }
            break;

        case ProcessingStage::CONVOLUTION:
            if (convolution_) {
                convolution_->processAudio(input, output, numSamples);
            }
            break;

        case ProcessingStage::PRE_PROCESSING:
            processPreProcessingNode(node, input, output, numSamples, channels);
            break;

        case ProcessingStage::DYNAMICS:
            processDynamicsNode(node, input, output, numSamples, channels);
            break;

        case ProcessingStage::POST_PROCESSING:
            processPostProcessingNode(node, input, output, numSamples, channels);
            break;

        default:
            // Default to pass-through
            std::copy(input, input + numSamples, output);
            break;
    }

    // Apply gain if needed
    if (node.gain != 0.0f) {
        float gainLinear = dbToLinear(node.gain);
        for (size_t i = 0; i < numSamples; ++i) {
            output[i] *= gainLinear;
        }
    }

    // Apply dithering if enabled
    if (config_.enableDithering) {
        applyDithering(output, numSamples);
    }

    // Update timing statistics
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    node.cpuTimeUs += duration.count();
}

void ProcessingChain::processPreProcessingNode(ProcessingNode& node, const float* input, float* output,
                                              size_t numSamples, uint16_t channels) {
    // Simple pre-processing: mild compression
    float threshold = dbToLinear(-12.0f); // -12dB threshold
    float ratio = 4.0f;
    float attackTime = 0.001f; // 1ms
    float releaseTime = 0.1f; // 100ms

    static float envelope = 0.0f;

    for (size_t i = 0; i < numSamples; ++i) {
        float inputSample = input[i];
        float absInput = std::abs(inputSample);

        // Update envelope
        float attackCoeff = (absInput > envelope) ? attackTime : releaseTime;
        envelope = envelope * (1.0f - attackCoeff) + absInput * attackCoeff;

        // Apply compression
        if (envelope > threshold) {
            float gain = threshold + (envelope - threshold) / ratio;
            gain /= envelope;
            output[i] = inputSample * gain;
        } else {
            output[i] = inputSample;
        }
    }
}

void ProcessingChain::processDynamicsNode(ProcessingNode& node, const float* input, float* output,
                                           size_t numSamples, uint16_t channels) {
    // Advanced dynamics processing
    float threshold = dbToLinear(-6.0f); // -6dB threshold
    float ratio = 8.0f;
    float kneeWidth = dbToLinear(2.0f); // 2dB knee

    for (size_t i = 0; i < numSamples; ++i) {
        float inputSample = input[i];
        float absInput = std::abs(inputSample);

        // Soft knee compression
        float gain = 1.0f;
        if (absInput > threshold) {
            float overThreshold = absInput - threshold;
            float kneeRatio = 1.0f + (ratio - 1.0f) * (overThreshold / kneeWidth);
            kneeRatio = std::min(kneeRatio, ratio);

            gain = threshold + overThreshold / kneeRatio;
            gain /= absInput;
        }

        output[i] = inputSample * gain;
    }
}

void ProcessingChain::processPostProcessingNode(ProcessingNode& node, const float* input, float* output,
                                                size_t numSamples, uint16_t channels) {
    // Simple limiter to prevent clipping
    float threshold = dbToLinear(-0.1f); // -0.1dB threshold

    for (size_t i = 0; i < numSamples; ++i) {
        float inputSample = input[i];
        float absInput = std::abs(inputSample);

        if (absInput > threshold) {
            float gain = threshold / absInput;
            output[i] = inputSample * gain;
        } else {
            output[i] = inputSample;
        }
    }
}

std::vector<ProcessingChain::ProcessingNode*> ProcessingChain::getActiveNodes() const {
    std::vector<ProcessingNode*> activeNodes;

    for (const auto& nodeId : processingOrder_) {
        auto it = nodes_.find(nodeId);
        if (it != nodes_.end() && isNodeActive(it->second)) {
            activeNodes.push_back(const_cast<ProcessingNode*>(&it->second));
        }
    }

    return activeNodes;
}

bool ProcessingChain::isNodeActive(const ProcessingNode& node) const {
    return node.enabled && !node.bypassed;
}

void ProcessingChain::initializeBuffers() {
    std::lock_guard<std::mutex> lock(buffersMutex_);

    size_t totalSamples = config_.blockSize * config_.numChannels;
    buffers_.inputBuffer.resize(totalSamples);
    buffers_.outputBuffer.resize(totalSamples);
    buffers_.intermediateBuffer.resize(totalSamples);
    buffers_.sidechainBuffer.resize(totalSamples);
    buffers_.parallelBufferA.resize(totalSamples);
    buffers_.parallelBufferB.resize(totalSamples);
    buffers_.feedbackBuffer.resize(totalSamples);
}

void ProcessingChain::processingThread() {
    Logger::info("ProcessingChain: Processing thread started");

    while (processingActive_.load()) {
        // Main processing loop
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        // Update performance statistics
        updatePerformanceStatistics();
    }

    Logger::info("ProcessingChain: Processing thread stopped");
}

void ProcessingChain::monitoringThread() {
    Logger::info("ProcessingChain: Monitoring thread started");

    while (processingActive_.load()) {
        try {
            // Update monitoring data
            updateMonitoringData();

            std::this_thread::sleep_for(std::chrono::milliseconds(100));

        } catch (const std::exception& e) {
            setError("Monitoring thread error: " + std::string(e.what()));
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    Logger::info("ProcessingChain: Monitoring thread stopped");
}

void ProcessingChain::automationThread() {
    Logger::info("ProcessingChain: Automation thread started");

    while (automationActive_.load()) {
        try {
            // Update automation
            updateAutomation();

            std::this_thread::sleep_for(std::chrono::milliseconds(10));

        } catch (const std::exception& e) {
            setError("Automation thread error: " + std::string(e.what()));
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    Logger::info("ProcessingChain: Automation thread stopped");
}

void ProcessingChain::updateProcessingOrder() {
    processingOrder_.clear();

    // Order nodes by stage
    std::vector<std::pair<ProcessingStage, std::string>> sortedNodes;
    for (const auto& [id, node] : nodes_) {
        sortedNodes.push_back({node.stage, id});
    }

    std::sort(sortedNodes.begin(), sortedNodes.end());

    for (const auto& [stage, id] : sortedNodes) {
        processingOrder_.push_back(id);
    }

    Logger::info("ProcessingChain: Processing order updated with {} nodes", processingOrder_.size());
}

void ProcessingChain::updateState() {
    state_.lastUpdateTime = std::chrono::steady_clock::now();

    // Count active nodes
    state_.activeNodes = 0;
    state_.enabledNodes = 0;
    state_.bypassedNodes = 0;

    {
        std::lock_guard<std::mutex> lock(nodesMutex_);
        for (const auto& [id, node] : nodes_) {
            if (node.enabled) {
                state_.enabledNodes++;
                if (!node.bypassed) {
                    state_.activeNodes++;
                }
            } else {
                state_.bypassedNodes++;
            }
        }
    }

    if (stateChangedCallback_) {
        stateChangedCallback_(state_);
    }
}

void ProcessingChain::updatePerformanceStatistics() {
    // Update CPU and GPU usage (placeholder)
    state_.cpuUsage = 25.0f + (state_.activeNodes * 5.0f);
    state_.gpuUsage = config_.enableGPUAcceleration ? 30.0f : 0.0f;
    state_.memoryUsage = 128.0f + (nodes_.size() * 8.0f); // MB

    // Update latency
    float totalLatency = 0.0f;
    {
        std::lock_guard<std::mutex> lock(nodesMutex_);
        for (const auto& [id, node] : nodes_) {
            if (isNodeActive(node)) {
                totalLatency += calculateNodeLatency(node.stage);
            }
        }
    }
    state_.averageLatency = totalLatency;

    // Update average with smoothing
    state_.cpuUsage = (state_.cpuUsage * 0.9f) + (state_.cpuUsage * 0.1f);
}

void ProcessingChain::updateMonitoringData() {
    // Update spectrum analyzer
    if (spectrumAnalyzer_ && config_.enableSpectrumAnalysis) {
        // This would trigger spectrum analysis
    }

    // Update level meters
    // This would calculate real-time levels
}

void ProcessingChain::updateAutomation() {
    if (!automationActive_.load()) {
        return;
    }

    std::lock_guard<std::mutex> lock(automationMutex_);

    for (auto& [key, automation] : automationData_) {
        if (automation.isRecording || automation.isPlaying) {
            // Update automation parameters
            processNodeAutomation(automation);
        }
    }

    currentProjectPosition_++;
}

float ProcessingChain::interpolateAutomationValue(const AutomationData& automation, uint64_t position) {
    if (automation.keyframes.empty()) {
        return 0.0f;
    }

    // Find surrounding keyframes
    auto it = std::lower_bound(automation.keyframes.begin(), automation.keyframes.end(),
                                 std::make_pair(position, 0.0f));

    if (it == automation.keyframes.begin()) {
        return automation.keyframes[0].second;
    }

    if (it == automation.keyframes.end()) {
        return automation.keyframes.back().second;
    }

    // Interpolate between keyframes
    auto nextIt = it;
    auto prevIt = it - 1;

    uint64_t prevPos = prevIt->first;
    uint64_t nextPos = nextIt->first;
    float prevVal = prevIt->second;
    float nextVal = nextIt->second;

    if (nextPos == prevPos) {
        return prevVal;
    }

    float t = static_cast<float>(position - prevPos) / static_cast<float>(nextPos - prevPos);
    t = std::clamp(t * automation.interpolationSpeed, 0.0f, 1.0f);

    return prevVal + t * (nextVal - prevVal);
}

void ProcessingChain::processNodeAutomation(ProcessingNode& node) {
    for (const auto& [key, automation] : automationData_) {
        if (automation.nodeId == node.id) {
            float value = interpolateAutomationValue(automation, currentProjectPosition_);
            setNodeParameter(node.id, key, value);
        }
    }
}

uint32_t ProcessingChain::calculateNodeLatency(ProcessingStage stage) const {
    switch (stage) {
        case ProcessingStage::EQUALIZATION:
            return config_.maxLatencyMs / 4 * config_.sampleRate / 1000;
        case ProcessingStage::CONVOLUTION:
            return config_.maxLatencyMs / 2 * config_.sampleRate / 1000;
        case ProcessingStage::DYNAMICS:
            return config_.maxLatencyMs / 10 * config_.sampleRate / 1000;
        default:
            return 64; // 64 samples default
    }
}

std::string ProcessingChain::getNodeStageName(ProcessingStage stage) const {
    switch (stage) {
        case ProcessingStage::INPUT: return "Input";
        case ProcessingStage::PRE_PROCESSING: return "Pre-Processing";
        case ProcessingStage::EQUALIZATION: return "Equalizer";
        case ProcessingStage::CONVOLUTION: return "Convolution";
        case ProcessingStage::DYNAMICS: return "Dynamics";
        case ProcessingStage::SPATIAL: return "Spatial";
        case ProcessingStage::POST_PROCESSING: return "Post-Processing";
        case ProcessingStage::OUTPUT: return "Output";
        case ProcessingStage::MONITORING: return "Monitoring";
        default: return "Unknown";
    }
}

void ProcessingChain::initializeNodeParameters(ProcessingNode& node) {
    // Initialize node-specific parameters based on stage
    switch (node.stage) {
        case ProcessingStage::EQUALIZATION:
            node.parameters["globalGain"] = 0.0f;
            node.parameters["globalQ"] = 1.0f;
            break;

        case ProcessingStage::CONVOLUTION:
            node.parameters["wetLevel"] = 1.0f;
            node.parameters["predelay"] = 0.0f;
            node.parameters["gain"] = 0.0f;
            break;

        case ProcessingStage::DYNAMICS:
            node.parameters["threshold"] = -6.0f;
            node.parameters["ratio"] = 4.0f;
            node.parameters["attack"] = 0.001f;
            node.parameters["release"] = 0.1f;
            break;

        default:
            break;
    }
}

void ProcessingChain::updateNodeComponent(ProcessingNode& node, const std::string& parameter, float value) {
    // Update component-specific parameters
    switch (node.stage) {
        case ProcessingStage::EQUALIZATION:
            if (equalizer_ && parameter == "globalGain") {
                // Update equalizer global gain
            }
            break;

        case ProcessingStage::CONVOLUTION:
            if (convolution_) {
                if (parameter == "wetLevel") {
                    convolution_->setWetDryMix(value, node.parameters["dryLevel"]);
                } else if (parameter == "dryLevel") {
                    convolution_->setWetDryMix(node.parameters["wetLevel"], value);
                } else if (parameter == "gain") {
                    convolution_->setGain(dbToLinear(value));
                }
            }
            break;

        case ProcessingStage::DYNAMICS:
            // Update dynamics parameters
            break;

        default:
            break;
    }
}

bool ProcessingChain::enableMultiThreading() {
    uint32_t numThreads = config_.numThreads > 0 ? config_.numThreads : std::thread::hardware_concurrency();

    // Limit threads to reasonable number
    numThreads = std::min(numThreads, static_cast<uint32_t>(8));

    // Create worker threads
    for (uint32_t i = 0; i < numThreads - 1; ++i) {
        workerThreads_.emplace_back([this]() {
            // Worker thread function
            while (processingActive_.load()) {
                // Process work items
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }

    multithreadingEnabled_.store(true);
    Logger::info("ProcessingChain: Multi-threading enabled with {} threads", numThreads);
    return true;
}

void ProcessingChain::applyDithering(float* audio, size_t numSamples) {
    if (!audio || numSamples == 0) {
        return;
    }

    // Simple triangular dithering
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

    for (size_t i = 0; i < numSamples; ++i) {
        float dither = dis(gen);
        audio[i] += dither / static_cast<float>(1 << (config_.ditherDepth - 1));
    }
}

float ProcessingChain::dbToLinear(float db) const {
    return std::pow(10.0f, db / 20.0f);
}

std::string ProcessingChain::generateNodeId() const {
    static std::atomic<uint32_t> counter{0};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);

    return "node_" + std::to_string(counter++) + "_" + std::to_string(dis(gen));
}

void ProcessingChain::setError(const std::string& error) const {
    lastError_ = error;
    diagnosticMessages_.push_back(error);
    Logger::error("ProcessingChain: {}", error);

    if (errorCallback_) {
        errorCallback_("chain", error);
    }
}

} // namespace vortex