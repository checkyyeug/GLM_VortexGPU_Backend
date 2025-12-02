#pragma once

#include "../../audio_types.hpp"
#include "../../network_types.hpp"
#include "../dsp/equalizer.hpp"
#include "../dsp/convolution.hpp"
#include "../dsp/spectrum_analyzer.hpp"

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <functional>

namespace vortex {

/**
 * @brief Advanced audio processing chain with modular architecture
 *
 * Implements a comprehensive, modular audio processing pipeline that integrates
 * all audio processing components into a unified system. Provides real-time
 * processing with sub-millisecond latency, GPU acceleration, and advanced
 * routing capabilities for professional audio applications.
 *
 * Processing Chain Architecture:
 * - Input Stage: Audio input handling and routing
 * - Pre-Processing: Dynamic range compression, limiting
 * - Equalization: 512-band graphic EQ with multiple filter types
 * - Convolution: 16M-point convolution for impulse responses
 * - Dynamics: Advanced dynamic range processing
 * - Spatial Processing: Multi-channel spatial audio processing
 * - Post-Processing: Final limiter, dithering, output stage
 * - Monitoring: Real-time analysis and metering
 *
 * Features:
 * - Modular plugin architecture for flexible routing
 * - GPU-accelerated processing with automatic fallback
 * - Real-time parameter automation and MIDI control
 * - Multi-channel support (up to 32 channels)
 * - Session management with undo/redo functionality
 * - A/B comparison and parallel processing
 * - Advanced metering and analysis tools
 * - Low-latency monitoring with zero-latency option
 */
class ProcessingChain {
public:
    enum class ProcessingStage {
        INPUT,                  // Audio input and routing
        PRE_PROCESSING,         // Pre-processing (compression, limiting)
        EQUALIZATION,          // 512-band equalizer
        CONVOLUTION,           // 16M-point convolution
        DYNAMICS,              // Advanced dynamics processing
        SPATIAL,               // Spatial processing
        POST_PROCESSING,        // Post-processing (limiter, dithering)
        OUTPUT,                // Output stage
        MONITORING             // Analysis and metering
    };

    enum class ProcessingMode {
        REAL_TIME,             // Real-time processing with minimal latency
        HIGH_QUALITY,          // High-quality processing with higher latency
        OFFLINE,              // Offline/batch processing mode
        ADAPTIVE               // Automatically adjusts based on load
    };

    enum class LatencyMode {
        ULTRA_LOW,             // <1ms latency
        LOW,                   // 1-10ms latency
        BALANCED,              // 10-50ms latency
        HIGH,                  // 50-100ms latency
        UNLIMITED              // Maximum quality regardless of latency
    };

    enum class RoutingMode {
        SERIES,                // Sequential processing
        PARALLEL,              // Parallel processing branches
        FEEDBACK,              // Feedback routing
        HYBRID                 // Combination of modes
    };

    struct ProcessingNode {
        std::string id;
        std::string name;
        ProcessingStage stage;
        bool enabled = true;
        bool bypassed = false;
        std::map<std::string, float> parameters;
        std::vector<std::string> inputs;     // Input node IDs
        std::vector<std::string> outputs;    // Output node IDs

        // Processing state
        float wetLevel = 1.0f;              // Wet signal level
        float dryLevel = 0.0f;              // Dry signal level
        float gain = 0.0f;                  // Gain in dB
        float phase = 0.0f;                 // Phase shift in degrees

        // Timing
        uint32_t latencySamples = 0;        // Processing latency
        std::chrono::steady_clock::time_point lastProcessTime;

        // Statistics
        uint64_t samplesProcessed = 0;
        uint64_t cpuTimeUs = 0;
        uint64_t gpuTimeUs = 0;
        float peakLevel = 0.0f;
        float averageLevel = 0.0f;

        // Plugin-specific data
        std::shared_ptr<void> pluginData;
    };

    struct ChainConfig {
        // Basic configuration
        uint32_t sampleRate = 48000;
        uint16_t numChannels = 2;
        uint32_t blockSize = 512;
        ProcessingMode mode = ProcessingMode::REAL_TIME;
        LatencyMode latencyMode = LatencyMode::BALANCED;

        // Audio quality
        bool enableHighPrecision = false;
        bool enableDithering = true;
        bool enableNoiseShaping = false;
        uint32_t ditherDepth = 24;

        // GPU configuration
        bool enableGPUAcceleration = true;
        bool enableMultiGPU = false;
        std::vector<std::string> preferredGPUs;

        // Performance
        float maxLatencyMs = 50.0f;
        uint32_t maxCPUUsage = 80;          // Maximum CPU usage percentage
        bool enableLoadBalancing = true;
        uint32_t numThreads = 0;          // 0 = auto-detect

        // Advanced features
        bool enableAutomation = true;
        bool enableMIDIControl = true;
        bool enableSidechain = false;
        bool enableParallelProcessing = true;
        bool enableUndoRedo = true;
        uint32_t maxUndoLevels = 32;

        // Monitoring
        bool enableSpectrumAnalysis = true;
        bool enablePhaseAnalysis = false;
        bool enableStereoAnalysis = true;
        bool enableMetering = true;
        uint32_t analysisBlockSize = 4096;

        // Session management
        bool autoSave = true;
        std::string autoSavePath = "sessions/";
        uint32_t autoSaveInterval = 300;    // seconds
    };

    struct ChainState {
        bool isProcessing = false;
        bool isRecording = false;
        bool isPlaying = false;
        uint64_t currentPosition = 0;
        float cpuUsage = 0.0f;
        float gpuUsage = 0.0f;
        float memoryUsage = 0.0f;
        float averageLatency = 0.0f;
        uint32_t activeNodes = 0;
        std::chrono::steady_clock::time_point lastUpdateTime;
    };

    struct AutomationData {
        std::string parameterId;
        std::string nodeId;
        std::vector<std::pair<uint64_t, float>> keyframes; // (position, value)
        bool isRecording = false;
        bool isPlaying = false;
        uint64_t currentPosition = 0;
        float interpolationSpeed = 0.1f;     // 0.0 = instant, 1.0 = smooth
    };

    ProcessingChain();
    ~ProcessingChain();

    // Initialization
    bool initialize(const ChainConfig& config);
    void shutdown();
    bool isInitialized() const;

    // Main processing
    bool processAudio(const float* input, float* output, size_t numSamples);
    bool processAudioMultiChannel(const std::vector<const float*>& inputs,
                                   std::vector<float*>& outputs,
                                   size_t numSamples, uint16_t channels);

    // Node management
    std::string addNode(ProcessingStage stage, const std::string& name);
    bool removeNode(const std::string& nodeId);
    bool enableNode(const std::string& nodeId, bool enabled);
    bool bypassNode(const std::string& nodeId, bool bypassed);
    ProcessingNode getNode(const std::string& nodeId) const;
    std::vector<ProcessingNode> getAllNodes() const;

    // Routing
    bool connectNodes(const std::string& sourceId, const std::string& destId);
    bool disconnectNodes(const std::string& sourceId, const std::string& destId);
    bool setRoutingMode(RoutingMode mode);
    RoutingMode getRoutingMode() const;

    // Processing stages
    bool enableEqualizer(bool enabled);
    bool loadEqualizerPreset(const std::string& presetName);
    bool setEqualizerBandGain(uint32_t band, float gain);
    bool enableConvolution(bool enabled);
    bool loadImpulseResponse(const std::string& filePath);
    bool enableSpectrumAnalysis(bool enabled);
    std::vector<float> getCurrentSpectrum() const;

    // Real-time control
    bool setNodeParameter(const std::string& nodeId, const std::string& parameter, float value);
    float getNodeParameter(const std::string& nodeId, const std::string& parameter) const;
    bool setNodeGain(const std::string& nodeId, float gain);
    bool setNodeWetDryMix(const std::string& nodeId, float wetLevel, float dryLevel);

    // Automation
    bool startAutomation(const std::string& nodeId, const std::string& parameter);
    bool stopAutomation(const std::string& nodeId, const std::string& parameter);
    bool addAutomationKeyframe(const std::string& nodeId, const std::string& parameter,
                               uint64_t position, float value);
    bool removeAutomationKeyframe(const std::string& nodeId, const std::string& parameter,
                                  uint64_t position);
    std::vector<AutomationData> getAutomationData() const;

    // Session management
    bool saveSession(const std::string& sessionPath);
    bool loadSession(const std::string& sessionPath);
    bool createSession(const std::string& name);
    bool deleteSession(const std::string& name);
    std::vector<std::string> getAvailableSessions() const;

    // Comparison and parallel processing
    bool enableABComparison(bool enabled);
    bool switchToPreset(const std::string& presetName);
    bool toggleABState();
    bool enableParallelBranch(const std::string& branchId, bool enabled);

    // Monitoring and analysis
    struct MonitoringData {
        std::vector<float> spectrum;
        std::vector<float> phase;
        std::vector<float> peakLevels;
        std::vector<float> rmsLevels;
        std::vector<float> dynamicRange;
        float cpuUsage = 0.0f;
        float gpuUsage = 0.0f;
        float latency = 0.0f;
        uint64_t samplesProcessed = 0;
        std::chrono::steady_clock::time_point timestamp;
    };

    MonitoringData getMonitoringData() const;
    float getLatency() const;
    float getCPUUsage() const;
    float getGPUUsage() const;

    // Preset management
    bool savePreset(const std::string& name, const std::string& description = "");
    bool loadPreset(const std::string& name);
    bool deletePreset(const std::string& name);
    std::vector<std::string> getAvailablePresets() const;

    // Performance optimization
    bool setProcessingMode(ProcessingMode mode);
    bool setLatencyMode(LatencyMode mode);
    bool enableGPUAcceleration(bool enable);
    bool optimizeForPerformance();

    // Advanced features
    bool enableSidechain(const std::string& nodeId, const std::string& sidechainSource);
    bool setParallelRatio(const std::string& branchA, const std::string& branchB, float ratio);
    bool enableFeedback(const std::string& sourceId, const std::string& destId, float amount);
    bool setMasterGain(float gain);

    // Statistics and monitoring
    struct ChainStatistics {
        uint64_t totalSamplesProcessed = 0;
        uint64_t totalProcessingTimeUs = 0;
        float averageLatency = 0.0f;
        float maxLatency = 0.0f;
        float minLatency = 1000.0f;
        float cpuUsage = 0.0f;
        float gpuUsage = 0.0f;
        uint32_t activeNodes = 0;
        uint32_t enabledNodes = 0;
        uint32_t bypassedNodes = 0;
        uint64_t droppedFrames = 0;
        float signalToNoiseRatio = 0.0f;
        float totalHarmonicDistortion = 0.0f;
        std::chrono::steady_clock::time_point startTime;
        std::chrono::steady_clock::time_point lastActivity;
    };

    ChainStatistics getStatistics() const;
    void resetStatistics();

    // Health monitoring
    bool isHealthy() const;
    std::vector<std::string> getDiagnosticMessages() const;

    // Callbacks
    using NodeChangedCallback = std::function<void(const ProcessingNode&)>;
    using ParameterChangedCallback = std::function<void(const std::string&, const std::string&, float)>;
    using StateChangedCallback = std::function<void(const ChainState&)>;
    using ErrorCallback = std::function<void(const std::string&, const std::string&)>;

    void setNodeChangedCallback(NodeChangedCallback callback);
    void setParameterChangedCallback(ParameterChangedCallback callback);
    void setStateChangedCallback(StateChangedCallback callback);
    void setErrorCallback(ErrorCallback callback);

private:
    // Core processing
    void processNode(ProcessingNode& node, const float* input, float* output,
                      size_t numSamples, uint16_t channels);
    void processEqualizerNode(ProcessingNode& node, const float* input, float* output,
                               size_t numSamples, uint16_t channels);
    void processConvolutionNode(ProcessingNode& node, const float* input, float* output,
                                size_t numSamples, uint16_t channels);
    void processDynamicsNode(ProcessingNode& node, const float* input, float* output,
                             size_t numSamples, uint16_t channels);

    // Routing
    void processRoutingGraph(const std::vector<const float*>& inputs,
                              std::vector<float*>& outputs,
                              size_t numSamples, uint16_t channels);
    std::vector<ProcessingNode*> getActiveNodes() const;
    bool validateRouting() const;

    // Automation
    void updateAutomation();
    float interpolateAutomationValue(const AutomationData& automation, uint64_t position);
    void processNodeAutomation(ProcessingNode& node);

    // Multi-threading
    void processingThread();
    void monitoringThread();
    void automationThread();
    bool enableMultiThreading();
    void processNodesParallel(const std::vector<ProcessingNode*>& nodes,
                              const std::vector<const float*>& inputs,
                              std::vector<float*>& outputs,
                              size_t numSamples, uint16_t channels);

    // Memory management
    struct ProcessingBuffers {
        std::vector<float> inputBuffer;
        std::vector<float> outputBuffer;
        std::vector<float> intermediateBuffer;
        std::vector<float> sidechainBuffer;
        std::vector<float> parallelBufferA;
        std::vector<float> parallelBufferB;
        std::vector<float> feedbackBuffer;
        std::vector<std::vector<float>> nodeInputs;
        std::vector<std::vector<float>> nodeOutputs;
    };

    ProcessingBuffers buffers_;
    mutable std::mutex buffersMutex_;

    // State management
    ChainConfig config_;
    ChainState state_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> processingActive_{false};

    // Node registry
    std::map<std::string, ProcessingNode> nodes_;
    std::vector<std::string> processingOrder_;
    std::map<std::string, std::vector<std::string>> routingGraph_;
    RoutingMode routingMode_ = RoutingMode::SERIES;
    mutable std::mutex nodesMutex_;

    // Processing components
    std::unique_ptr<Equalizer> equalizer_;
    std::unique_ptr<ConvolutionEngine> convolution_;
    std::unique_ptr<SpectrumAnalyzer> spectrumAnalyzer_;

    // Automation
    std::map<std::string, AutomationData> automationData_;
    std::atomic<bool> automationActive_{false};
    uint64_t currentProjectPosition_ = 0;
    mutable std::mutex automationMutex_;

    // A/B comparison
    bool abComparisonEnabled_ = false;
    bool abState_ = false; // false = A, true = B
    std::map<std::string, std::map<std::string, float>> presetA_;
    std::map<std::string, std::map<std::string, float>> presetB_;

    // Session management
    std::string currentSessionPath_;
    std::map<std::string, std::string> sessions_; // name -> path
    mutable std::mutex sessionMutex_;

    // Parallel processing
    std::map<std::string, bool> parallelBranches_;
    std::map<std::pair<std::string, std::string>, float> parallelRatios_;

    // Feedback routing
    std::map<std::pair<std::string, std::string>, float> feedbackAmounts_;

    // Statistics
    mutable std::mutex statsMutex_;
    ChainStatistics statistics_;
    std::chrono::steady_clock::time_point lastUpdateTime_;

    // Threads
    std::unique_ptr<std::thread> processingThread_;
    std::unique_ptr<std::thread> monitoringThread_;
    std::unique_ptr<std::thread> automationThread_;
    std::vector<std::thread> workerThreads_;
    std::atomic<bool> multithreadingEnabled_{false};

    // Callbacks
    NodeChangedCallback nodeChangedCallback_;
    ParameterChangedCallback parameterChangedCallback_;
    StateChangedCallback stateChangedCallback_;
    ErrorCallback errorCallback_;

    // Error handling
    mutable std::string lastError_;
    std::vector<std::string> diagnosticMessages_;
    void setError(const std::string& error) const;

    // Utility functions
    std::string generateNodeId() const;
    bool validateNodeId(const std::string& nodeId) const;
    void updateProcessingOrder();
    void updateState();
    bool isNodeActive(const ProcessingNode& node) const;
    float dbToLinear(float db) const;
    float linearToDb(float linear) const;

    // Constants
    static constexpr uint32_t MAX_NODES = 128;
    static constexpr uint32_t MAX_CHANNELS = 32;
    static constexpr float MAX_LATENCY_MS = 1000.0f;
    static constexpr uint32_t UNDO_MAX_LEVELS = 32;
};

} // namespace vortex