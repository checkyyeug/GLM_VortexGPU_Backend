#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <array>
#include <chrono>
#include <functional>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <queue>
#include <deque>
#include "core/audio/multi_channel_engine.hpp"

namespace vortex {
namespace core {
namespace audio {

/**
 * Low-latency Audio Routing System
 * Provides ultra-low latency audio signal routing with deterministic performance
 * Optimized for real-time audio applications with sub-millisecond latency requirements
 */

enum class RoutingNode {
    INPUT,              ///< Input source node
    OUTPUT,             ///< Output destination node
    MIXER,              ///< Mixer/Summing node
    SPLITTER,           ///< Splitter node (one to many)
    MERGER,             ///< Merger node (many to one)
    PROCESSOR,          ///< Audio processor node
    DELAY,              ///< Delay line node
    GAIN,               ///< Gain node
    MUTE,               ///< Mute node
    PAN,                ///< Pan node
    FADE,               ///< Fade/Crossfade node
    SWITCH,             ///< Switch node
    MATRIX,             ///< Matrix mixer node
    BUS,                ///< Audio bus node
    GROUP,              ///< Channel group node
    AUX_SEND,           ///< Auxiliary send node
    AUX_RETURN,         ///< Auxiliary return node
    INSERT,             ///< Insert point node
    SIDECHAIN,          ///< Sidechain input node
    CUSTOM              ///< Custom processing node
};

enum class RoutingMode {
    PASS_THROUGH,       ///< Direct pass-through
    MIX_ADD,            ///< Mix by addition
    MIX_MULTIPLY,       ///< Mix by multiplication
    MIX_AVERAGE,        ///< Mix by averaging
    CROSSFADE,          ///< Crossfade between sources
    SWITCHED,           ///< Switch between sources
    MATRIX,             ///< Matrix routing
    PARALLEL,           ///< Parallel processing
    SERIES,             ///< Series processing
    FEEDBACK,           ///< Feedback routing
    DYNAMIC,            ///< Dynamic routing based on conditions
    SCHEDULED,          ///< Time-scheduled routing
    AUTOMATED           ///< Parameter-automated routing
};

enum class LatencyMode {
    MINIMUM,            ///< Minimum possible latency
    LOW,                ///< Low latency
    BALANCED,           ///< Balanced latency/buffering
    SAFE,               ///< Safe with extra buffering
    CUSTOM              ///< Custom latency settings
};

enum class BufferMode {
    LOCK_FREE,          ///< Lock-free ring buffer
    ATOMIC,             ///< Atomic operations
    MEMORY_POOL,        ///< Pre-allocated memory pool
    STACK_ALLOCATED,    ///< Stack allocated buffers
    CUSTOM_ALLOCATOR,   ///< Custom allocator
    ZERO_COPY           ///< Zero-copy buffer passing
};

enum class RoutingPriority {
    CRITICAL,           ///< Critical path (highest priority)
    HIGH,               ///< High priority
    NORMAL,             ///< Normal priority
    LOW,                ///< Low priority
    BACKGROUND          ///< Background processing
};

struct RoutingPath {
    uint32_t source_node_id = 0;           ///< Source node ID
    uint32_t destination_node_id = 0;      ///< Destination node ID
    float gain = 1.0f;                     ///< Path gain
    float pan = 0.0f;                      ///< Pan (-1.0 to 1.0)
    bool muted = false;                    ///< Mute state
    bool enabled = true;                   ///< Path enabled state
    int32_t delay_samples = 0;             ///< Delay in samples
    RoutingMode mode = RoutingMode::PASS_THROUGH; ///< Routing mode
    RoutingPriority priority = RoutingPriority::NORMAL; ///< Processing priority
    uint32_t buffer_mask = UINT32_MAX;     ///< Buffer mask for multi-bus routing
    std::string name;                      ///< Path name
    std::vector<std::string> tags;         ///< Path tags
    uint64_t creation_time = 0;            ///< Creation timestamp
    uint32_t process_count = 0;            ///< Process call count
    double cpu_usage_percent = 0.0;       ///< CPU usage for this path
};

struct AudioNode {
    uint32_t id = 0;                       ///< Unique node ID
    RoutingNode type = RoutingNode::INPUT;  ///< Node type
    std::string name;                      ///< Node name
    std::string description;               ///< Node description
    int input_channels = 0;                ///< Number of input channels
    int output_channels = 0;               ///< Number of output channels
    int max_inputs = 1;                    ///< Maximum input connections
    int max_outputs = 1;                   ///< Maximum output connections
    LatencyMode latency_mode = LatencyMode::BALANCED; ///< Latency mode
    BufferMode buffer_mode = BufferMode::LOCK_FREE;   ///< Buffer mode
    RoutingPriority priority = RoutingPriority::NORMAL; ///< Node priority
    bool is_active = true;                 ///< Node active state
    bool supports_bypass = true;            ///< Bypass support
    bool bypassed = false;                 ///< Bypass state
    uint64_t process_latency_samples = 0;  ///< Processing latency in samples
    std::vector<float> parameters;         ///< Node parameters
    std::vector<std::string> parameter_names; ///< Parameter names
    std::chrono::steady_clock::time_point last_process_time;
};

struct RoutingMetrics {
    uint64_t total_paths = 0;              ///< Total number of paths
    uint64_t active_paths = 0;             ///< Active paths count
    uint64_t active_nodes = 0;             ///< Active nodes count
    double average_latency_ms = 0.0;       ///< Average routing latency
    double peak_latency_ms = 0.0;          ///< Peak routing latency
    double cpu_usage_percent = 0.0;        ///< Total CPU usage
    double memory_usage_mb = 0.0;          ///< Memory usage in MB
    uint64_t buffer_underruns = 0;         ///< Buffer underrun count
    uint64_t buffer_overruns = 0;          ///< Buffer overrun count
    uint64_t xruns_count = 0;              ///< Total XRUN count
    double max_throughput_samples_per_sec = 0.0; ///< Maximum throughput
    uint32_t dropped_frames = 0;           ///< Dropped frames count
    bool real_time_stable = true;          ///< Real-time stability
    std::chrono::steady_clock::time_point last_update;
};

struct RoutingConfig {
    int sample_rate = 48000;               ///< Sample rate
    int buffer_size = 256;                 ///< Buffer size
    int max_channels = 64;                 ///< Maximum channels
    int max_nodes = 1024;                  ///< Maximum nodes
    int max_paths = 4096;                  ///< Maximum paths
    LatencyMode default_latency = LatencyMode::LOW; ///< Default latency mode
    BufferMode default_buffer = BufferMode::LOCK_FREE; ///< Default buffer mode
    double target_latency_ms = 2.0;        ///< Target latency in milliseconds
    double max_acceptable_latency_ms = 10.0; ///< Maximum acceptable latency
    int cpu_affinity_core = -1;            ///< CPU core affinity (-1 = any)
    int thread_priority = 0;               ///< Thread priority (-10 to 10)
    bool enable_deterministic = true;      ///< Enable deterministic processing
    bool enable_zero_copy = false;         ///< Enable zero-copy where possible
    size_t memory_pool_size = 0;           ///< Memory pool size (0 = auto)
    bool enable_profiling = false;         ///< Enable performance profiling
    std::chrono::microseconds max_process_time{500}; ///< Max process time per frame
};

class LockFreeRingBuffer {
public:
    LockFreeRingBuffer(size_t size);
    ~LockFreeRingBuffer();

    bool write(const float* data, size_t samples);
    bool read(float* data, size_t samples);
    size_t available() const;
    size_t space() const;
    void clear();
    size_t size() const { return size_; }

private:
    alignas(64) std::atomic<size_t> read_pos_{0};
    alignas(64) std::atomic<size_t> write_pos_{0};
    size_t size_;
    size_t mask_;
    std::unique_ptr<float[]> buffer_;
};

class AudioProcessorNode {
public:
    AudioProcessorNode(uint32_t id, RoutingNode type, int input_channels, int output_channels);
    virtual ~AudioProcessorNode() = default;

    virtual bool initialize(const RoutingConfig& config) = 0;
    virtual void process(const float** inputs, float** outputs, size_t samples) = 0;
    virtual void reset() = 0;
    virtual void setParameter(int index, float value) = 0;
    virtual float getParameter(int index) const = 0;

    uint32_t getId() const { return id_; }
    RoutingNode getType() const { return type_; }
    int getInputChannels() const { return input_channels_; }
    int getOutputChannels() const { return output_channels_; }
    void setActive(bool active) { is_active_ = active; }
    bool isActive() const { return is_active_; }
    void setBypassed(bool bypassed) { bypassed_ = bypassed; }
    bool isBypassed() const { return bypassed_; }

protected:
    uint32_t id_;
    RoutingNode type_;
    int input_channels_;
    int output_channels_;
    bool is_active_;
    bool bypassed_;
    std::vector<float> parameters_;
    RoutingConfig config_;
};

class GainNode : public AudioProcessorNode {
public:
    GainNode(uint32_t id, int channels);
    bool initialize(const RoutingConfig& config) override;
    void process(const float** inputs, float** outputs, size_t samples) override;
    void reset() override;
    void setParameter(int index, float value) override;
    float getParameter(int index) const override;

    void setGain(float gain) { gain_ = gain; }
    float getGain() const { return gain_; }

private:
    float gain_ = 1.0f;
    float target_gain_ = 1.0f;
    bool ramping_ = false;
};

class MixerNode : public AudioProcessorNode {
public:
    MixerNode(uint32_t id, int input_channels, int output_channels);
    bool initialize(const RoutingConfig& config) override;
    void process(const float** inputs, float** outputs, size_t samples) override;
    void reset() override;
    void setParameter(int index, float value) override;
    float getParameter(int index) const override;

    void setInputGain(int input, float gain);
    void setOutputGain(int output, float gain);
    void setPan(int output, float pan);

private:
    std::vector<float> input_gains_;
    std::vector<float> output_gains_;
    std::vector<float> output_pans_;
};

class DelayNode : public AudioProcessorNode {
public:
    DelayNode(uint32_t id, int channels, int max_delay_samples);
    bool initialize(const RoutingConfig& config) override;
    void process(const float** inputs, float** outputs, size_t samples) override;
    void reset() override;
    void setParameter(int index, float value) override;
    float getParameter(int index) const override;

    void setDelay(float delay_seconds);
    float getDelay() const;

private:
    std::vector<std::unique_ptr<LockFreeRingBuffer>> delay_lines_;
    int max_delay_samples_;
    float delay_seconds_ = 0.0f;
    int delay_samples_ = 0;
};

class PanNode : public AudioProcessorNode {
public:
    PanNode(uint32_t id, int input_channels, int output_channels);
    bool initialize(const RoutingConfig& config) override;
    void process(const float** inputs, float** outputs, size_t samples) override;
    void reset() override;
    void setParameter(int index, float value) override;
    float getParameter(int index) const override;

    void setPan(float pan);
    float getPan() const;

private:
    float pan_ = 0.0f;
    float left_gain_ = 1.0f;
    float right_gain_ = 1.0f;
    void updateGains();
};

class FadeNode : public AudioProcessorNode {
public:
    FadeNode(uint32_t id, int channels);
    bool initialize(const RoutingConfig& config) override;
    void process(const float** inputs, float** outputs, size_t samples) override;
    void reset() override;
    void setParameter(int index, float value) override;
    float getParameter(int index) const override;

    void startFade(float target_gain, float duration_seconds);
    bool isFading() const { return fading_; }

private:
    float current_gain_ = 1.0f;
    float target_gain_ = 1.0f;
    float fade_duration_samples_ = 0;
    int samples_remaining_ = 0;
    float gain_step_ = 0.0f;
    bool fading_ = false;
};

class MatrixMixerNode : public AudioProcessorNode {
public:
    MatrixMixerNode(uint32_t id, int inputs, int outputs);
    bool initialize(const RoutingConfig& config) override;
    void process(const float** inputs, float** outputs, size_t samples) override;
    void reset() override;
    void setParameter(int index, float value) override;
    float getParameter(int index) const override;

    void setMatrixGain(int input, int output, float gain);
    float getMatrixGain(int input, int output) const;
    void clearMatrix();

private:
    std::vector<std::vector<float>> matrix_;
};

using RoutingCallback = std::function<void(const RoutingPath& path)>;
using NodeCallback = std::function<void(const AudioNode& node)>;
using MetricsCallback = std::function<void(const RoutingMetrics& metrics)>;

/**
 * Low-Latency Audio Router
 * Core audio routing system with ultra-low latency performance
 */
class AudioRouter {
public:
    AudioRouter();
    ~AudioRouter();

    /**
     * Initialize audio router
     * @param config Router configuration
     * @return True if initialization successful
     */
    bool initialize(const RoutingConfig& config);

    /**
     * Shutdown router and cleanup resources
     */
    void shutdown();

    /**
     * Process audio frame through routing network
     * @param input_buffers Input audio buffers
     * @param output_buffers Output audio buffers
     * @param num_samples Number of samples to process
     * @return True if processing successful
     */
    bool processAudioFrame(const float** input_buffers, float** output_buffers, size_t num_samples);

    /**
     * Add audio node to routing network
     * @param node Node configuration
     * @return True if node added successfully
     */
    bool addNode(const AudioNode& node);

    /**
     * Remove audio node from routing network
     * @param node_id Node ID to remove
     * @return True if node removed successfully
     */
    bool removeNode(uint32_t node_id);

    /**
     * Get audio node information
     * @param node_id Node ID
     * @return Node information if found
     */
    std::optional<AudioNode> getNode(uint32_t node_id) const;

    /**
     * Add routing path between nodes
     * @param path Routing path configuration
     * @return True if path added successfully
     */
    bool addPath(const RoutingPath& path);

    /**
     * Remove routing path
     * @param path_id Path ID (combination of source and destination)
     * @return True if path removed successfully
     */
    bool removePath(uint32_t path_id);

    /**
     * Get routing path information
     * @param source_id Source node ID
     * @param destination_id Destination node ID
     * @return Path information if found
     */
    std::optional<RoutingPath> getPath(uint32_t source_id, uint32_t destination_id) const;

    /**
     * Update path parameters
     * @param path_id Path ID
     * @param gain Path gain
     * @param pan Path pan
     * @param muted Mute state
     * @return True if update successful
     */
    bool updatePath(uint32_t path_id, float gain = -1.0f, float pan = -999.0f, bool muted = -1);

    /**
     * Enable/disable path
     * @param source_id Source node ID
     * @param destination_id Destination node ID
     * @param enabled Enable state
     * @return True if update successful
     */
    bool setPathEnabled(uint32_t source_id, uint32_t destination_id, bool enabled);

    /**
     * Set path delay
     * @param source_id Source node ID
     * @param destination_id Destination node ID
     * @param delay_samples Delay in samples
     * @return True if update successful
     */
    bool setPathDelay(uint32_t source_id, uint32_t destination_id, int32_t delay_samples);

    /**
     * Create and add a processor node
     * @param type Node type
     * @param name Node name
     * @param input_channels Number of input channels
     * @param output_channels Number of output channels
     * @return Node ID if successful, 0 otherwise
     */
    uint32_t createProcessorNode(RoutingNode type, const std::string& name,
                                int input_channels, int output_channels);

    /**
     * Connect two nodes with a path
     * @param source_id Source node ID
     * @param destination_id Destination node ID
     * @param gain Path gain
     * @param path_name Optional path name
     * @return True if connection successful
     */
    bool connectNodes(uint32_t source_id, uint32_t destination_id,
                     float gain = 1.0f, const std::string& path_name = "");

    /**
     * Disconnect two nodes
     * @param source_id Source node ID
     * @param destination_id Destination node ID
     * @return True if disconnection successful
     */
    bool disconnectNodes(uint32_t source_id, uint32_t destination_id);

    /**
     * Get all active paths
     * @return List of active paths
     */
    std::vector<RoutingPath> getActivePaths() const;

    /**
     * Get all nodes
     * @return List of all nodes
     */
    std::vector<AudioNode> getAllNodes() const;

    /**
     * Get nodes by type
     * @param type Node type
     * @return List of nodes of specified type
     */
    std::vector<AudioNode> getNodesByType(RoutingNode type) const;

    /**
     * Find path by name
     * @param name Path name
     * @return Path if found
     */
    std::optional<RoutingPath> findPathByName(const std::string& name) const;

    /**
     * Find node by name
     * @param name Node name
     * @return Node if found
     */
    std::optional<AudioNode> findNodeByName(const std::string& name) const;

    /**
     * Set routing matrix for matrix mixer node
     * @param node_id Node ID
     * @param matrix Gain matrix
     * @return True if matrix set successfully
     */
    bool setRoutingMatrix(uint32_t node_id, const std::vector<std::vector<float>>& matrix);

    /**
     * Get routing matrix from matrix mixer node
     * @param node_id Node ID
     * @return Gain matrix if node is a matrix mixer
     */
    std::optional<std::vector<std::vector<float>>> getRoutingMatrix(uint32_t node_id) const;

    /**
     * Start fade on all paths from a node
     * @param node_id Node ID
     * @param target_gain Target gain
     * @param duration_seconds Fade duration
     * @return True if fade started successfully
     */
    bool startNodeFade(uint32_t node_id, float target_gain, float duration_seconds);

    /**
     * Start crossfade between two nodes
     * @param source_id Source node ID
     * @param destination_id Destination node ID
     * @param duration_seconds Crossfade duration
     * @return True if crossfade started successfully
     */
    bool startCrossfade(uint32_t source_id, uint32_t destination_id, float duration_seconds);

    /**
     * Get routing metrics
     * @return Current routing metrics
     */
    RoutingMetrics getMetrics() const;

    /**
     * Register routing callback
     * @param callback Path change callback
     */
    void setRoutingCallback(RoutingCallback callback);

    /**
     * Register node callback
     * @param callback Node change callback
     */
    void setNodeCallback(NodeCallback callback);

    /**
     * Register metrics callback
     * @param callback Metrics update callback
     */
    void setMetricsCallback(MetricsCallback callback);

    /**
     * Reset entire routing system
     */
    void reset();

    /**
     * Clear all paths and nodes
     */
    void clear();

    /**
     * Validate routing configuration
     * @return True if configuration is valid
     */
    bool validateConfiguration() const;

    /**
     * Optimize routing for performance
     * @return True if optimization successful
     */
    bool optimizeRouting();

    /**
     * Export routing configuration
     * @return JSON string with routing configuration
     */
    std::string exportConfiguration() const;

    /**
     * Import routing configuration
     * @param config_json JSON configuration string
     * @return True if import successful
     */
    bool importConfiguration(const std::string& config_json);

private:
    struct PathInternal {
        RoutingPath path;
        float current_gain = 1.0f;
        float target_gain = 1.0f;
        bool gain_ramping = false;
        int32_t current_delay = 0;
        int32_t target_delay = 0;
        std::unique_ptr<LockFreeRingBuffer> delay_line;
        uint64_t process_count = 0;
        std::chrono::high_resolution_clock::time_point last_process;
    };

    struct NodeInternal {
        std::unique_ptr<AudioProcessorNode> processor;
        std::vector<const float*> input_buffers;
        std::vector<float*> output_buffers;
        std::vector<float> temp_buffer;
        AudioNode info;
        bool process_order_assigned = false;
        int process_order = 0;
    };

    // Core state
    bool initialized_ = false;
    RoutingConfig config_;
    mutable std::mutex mutex_;

    // Nodes and paths
    std::unordered_map<uint32_t, std::unique_ptr<NodeInternal>> nodes_;
    std::unordered_map<uint64_t, std::unique_ptr<PathInternal>> paths_; // key = (source << 32) | dest
    std::atomic<uint32_t> next_node_id_{1};

    // Processing state
    std::vector<std::vector<float>> process_buffers_;
    std::vector<const float*> input_pointers_;
    std::vector<float*> output_pointers_;
    std::vector<NodeInternal*> process_order_;
    bool process_order_valid_ = false;

    // Metrics and callbacks
    mutable std::mutex metrics_mutex_;
    RoutingMetrics metrics_;
    RoutingCallback routing_callback_;
    NodeCallback node_callback_;
    MetricsCallback metrics_callback_;
    std::chrono::steady_clock::time_point last_metrics_update_;
    uint64_t total_frames_processed_ = 0;

    // Performance monitoring
    std::atomic<uint64_t> buffer_underruns_{0};
    std::atomic<uint64_t> buffer_overruns_{0};
    std::atomic<uint32_t> dropped_frames_{0};
    std::chrono::high_resolution_clock::time_point start_time_;

    // Internal methods
    bool validateNode(const AudioNode& node) const;
    bool validatePath(const RoutingPath& path) const;
    void updateProcessOrder();
    void processNodeGraph(const float** inputs, float** outputs, size_t samples);
    void updateMetrics();
    void updatePathGains(float* buffer, size_t samples, PathInternal& path);
    void updatePathDelay(const float* input, float* output, size_t samples, PathInternal& path);
    void mixPathAudio(const float* input, float* output, size_t samples, PathInternal& path);
    uint64_t generatePathId(uint32_t source_id, uint32_t destination_id) const;
    std::unique_ptr<AudioProcessorNode> createProcessor(RoutingNode type, uint32_t id,
                                                       int input_channels, int output_channels);
    void optimizeProcessOrder();
    void detectCycleDependencies();
    bool isPathActive(uint64_t path_id) const;
    void clearNodeBuffers();
    void allocateBuffers();
    void deallocateBuffers();
};

/**
 * Audio Router Factory
 * Creates pre-configured routers for common use cases
 */
class AudioRouterFactory {
public:
    /**
     * Create router for mixing console
     * @param input_channels Number of input channels
     * @param output_channels Number of output channels
     * @param sample_rate Sample rate
     * @param buffer_size Buffer size
     * @return Configured router
     */
    static std::unique_ptr<AudioRouter> createMixingConsole(int input_channels, int output_channels,
                                                           int sample_rate = 48000, int buffer_size = 256);

    /**
     * Create router for DAW-style routing
     * @param tracks Number of audio tracks
     * @param buses Number of mixer buses
     * @param sample_rate Sample rate
     * @param buffer_size Buffer size
     * @return Configured router
     */
    static std::unique_ptr<AudioRouter> createDAWRouting(int tracks, int buses,
                                                        int sample_rate = 48000, int buffer_size = 256);

    /**
     * Create router for live sound mixing
     * @param inputs Number of inputs
     * @param outputs Number of outputs
     * @param matrix_size Matrix mixer size
     * @param sample_rate Sample rate
     * @param buffer_size Buffer size
     * @return Configured router
     */
    static std::unique_ptr<AudioRouter> createLiveSoundMixer(int inputs, int outputs, int matrix_size,
                                                            int sample_rate = 48000, int buffer_size = 64);

    /**
     * Create router for broadcast mixing
     * @param channels Number of audio channels
     * @param aux_sends Number of auxiliary sends
     * @param sample_rate Sample rate
     * @param buffer_size Buffer size
     * @return Configured router
     */
    static std::unique_ptr<AudioRouter> createBroadcastMixer(int channels, int aux_sends,
                                                            int sample_rate = 48000, int buffer_size = 512);

    /**
     * Create router for minimal latency
     * @param channels Number of channels
     * @param target_latency_ms Target latency in milliseconds
     * @param sample_rate Sample rate
     * @return Configured router
     */
    static std::unique_ptr<AudioRouter> createMinimalLatencyRouter(int channels, double target_latency_ms,
                                                                   int sample_rate = 48000);

private:
    static RoutingConfig createOptimalConfig(int channels, int sample_rate, int buffer_size,
                                            LatencyMode latency = LatencyMode::LOW);
};

// Utility functions
namespace audio_routing_utils {

    // Routing utilities
    uint64_t calculatePathHash(uint32_t source, uint32_t destination);
    bool isNodeConnected(uint32_t node_id, const std::vector<RoutingPath>& paths, bool as_source = true);
    std::vector<uint32_t> findConnectedNodes(uint32_t node_id, const std::vector<RoutingPath>& paths, bool as_source = true);
    bool hasCycle(uint32_t start_node, const std::vector<RoutingPath>& paths);
    std::vector<uint32_t> topologicalSort(const std::vector<RoutingPath>& paths, const std::vector<uint32_t>& nodes);

    // Audio utilities
    void applyGain(float* buffer, size_t samples, float gain);
    void applyPan(const float* input, float* output, size_t samples, float pan, int channels);
    void crossfade(const float* input1, const float* input2, float* output, size_t samples, float mix);
    void linearFade(float* buffer, size_t samples, float start_gain, float end_gain);
    float calculateRMS(const float* buffer, size_t samples);
    float calculatePeak(const float* buffer, size_t samples);

    // Performance utilities
    double calculateLatency(int buffer_size, int sample_rate);
    int calculateOptimalBufferSize(int sample_rate, double target_latency_ms);
    size_t calculateMemoryUsage(int channels, int buffer_size, int nodes, int paths);
    bool isLatencyAcceptable(double actual_latency_ms, double target_latency_ms);

    // Configuration utilities
    RoutingConfig createLowLatencyConfig(int sample_rate, int target_channels, double target_latency_ms);
    RoutingConfig createBalancedConfig(int sample_rate, int target_channels);
    RoutingConfig createSafeConfig(int sample_rate, int target_channels);
}

} // namespace audio
} // namespace core
} // namespace vortex