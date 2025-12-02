#pragma once

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <functional>
#include <unordered_map>
#include <vector>
#include <string>

#include "core/dsp/spectrum_analyzer.hpp"
#include "core/dsp/waveform_processor.hpp"
#include "core/dsp/vu_meter.hpp"
#include "system/logger.hpp"

// Forward declarations for Rust FFI
extern "C" {
    typedef struct WebSocketServer WebSocketServer;
    typedef struct SpectrumData SpectrumData;
    typedef struct AudioLevels AudioLevels;
    typedef struct WaveformData WaveformData;
}

namespace vortex::network {

/**
 * Real-time audio visualization data streaming
 *
 * This component bridges the audio processing pipeline (spectrum analyzer,
 * waveform processor, VU meter) with the WebSocket server for real-time
 * visualization data streaming to clients.
 *
 * Features:
 * - 60+ FPS streaming performance target
 * - Configurable update rates per client
 * - Efficient data serialization and compression
 * - Automatic subscription management
 * - Memory-efficient circular buffers
 * - Thread-safe operations
 * - Performance monitoring and metrics
 */

struct StreamConfig {
    uint32_t target_fps = 60;              // Target frames per second
    uint32_t max_subscribers = 1000;       // Maximum concurrent subscribers
    size_t buffer_size = 1024;             // Audio buffer size
    float sample_rate = 44100.0f;          // Audio sample rate
    uint16_t channels = 2;                 // Number of audio channels
    bool enable_compression = true;        // Enable data compression
    bool enable_metrics = true;            // Enable performance metrics
    uint32_t spectrum_fft_size = 2048;     // FFT size for spectrum analysis
    uint32_t waveform_length = 512;        // Waveform output length
    float update_interval_ms = 16.67f;     // Update interval (60 FPS)
    uint32_t max_queue_size = 100;         // Maximum queued updates per subscriber
};

struct StreamMetrics {
    std::atomic<uint64_t> total_frames_sent{0};
    std::atomic<uint64_t> total_bytes_sent{0};
    std::atomic<uint64_t> dropped_frames{0};
    std::atomic<double> avg_latency_ms{0.0};
    std::atomic<double> peak_latency_ms{0.0};
    std::atomic<double> min_fps{0.0};
    std::atomic<double> max_fps{0.0};
    std::atomic<double> avg_fps{0.0};
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point last_frame_time;
};

struct VisualizationData {
    std::vector<float> spectrum_magnitudes;    // Spectrum analyzer output
    std::vector<float> waveform_peaks;         // Waveform peaks
    std::vector<float> waveform_rms;           // Waveform RMS
    std::vector<float> vu_levels;              // VU meter levels
    uint64_t timestamp;                        // Microsecond timestamp
    float processing_time_ms;                  // Processing time for this frame
    bool is_valid;                            // Data validity flag
};

struct SubscriberInfo {
    std::string subscriber_id;
    std::string connection_id;
    std::vector<std::string> subscriptions;   // "spectrum", "waveform", "levels"
    float update_frequency;                    // Updates per second
    std::chrono::steady_clock::time_point last_update;
    std::queue<VisualizationData> pending_updates;
    std::mutex queue_mutex;
    std::atomic<uint64_t> frames_sent{0};
    std::atomic<uint64_t> frames_dropped{0};
};

class RealtimeStreamer {
public:
    RealtimeStreamer();
    ~RealtimeStreamer();

    // Initialization and lifecycle
    bool initialize(const StreamConfig& config);
    void shutdown();
    bool is_initialized() const { return initialized_; }

    // Audio processing integration
    void set_audio_sources(std::shared_ptr<core::dsp::SpectrumAnalyzer> spectrum_analyzer,
                          std::shared_ptr<core::dsp::WaveformProcessor> waveform_processor,
                          std::shared_ptr<core::dsp::VUMeter> vu_meter);

    // WebSocket server integration
    void set_websocket_server(WebSocketServer* server);

    // Streaming control
    bool start_streaming();
    void stop_streaming();
    bool is_streaming() const { return streaming_.load(); }

    // Subscriber management
    bool add_subscriber(const std::string& subscriber_id,
                       const std::string& connection_id,
                       const std::vector<std::string>& subscriptions,
                       float update_frequency = 60.0f);

    bool remove_subscriber(const std::string& subscriber_id);
    bool update_subscription(const std::string& subscriber_id,
                           const std::vector<std::string>& subscriptions,
                           float update_frequency);

    // Data processing
    bool process_audio_frame(const float* audio_data, size_t num_samples);
    void force_update();

    // Configuration
    void update_config(const StreamConfig& config);
    const StreamConfig& get_config() const { return config_; }

    // Metrics and monitoring
    StreamMetrics get_metrics() const;
    std::string get_performance_report() const;
    void reset_metrics();

    // WebSocket message creation
    std::vector<uint8_t> create_spectrum_message(const VisualizationData& data) const;
    std::vector<uint8_t> create_waveform_message(const VisualizationData& data) const;
    std::vector<uint8_t> create_levels_message(const VisualizationData& data) const;

private:
    // Core processing threads
    void audio_processing_thread();
    void streaming_thread();
    void metrics_thread();

    // Data collection and processing
    bool collect_visualization_data(VisualizationData& data);
    void process_subscriber_updates();

    // Message serialization
    std::vector<uint8_t> serialize_visualization_data(const VisualizationData& data,
                                                     const std::string& data_type) const;

    // Compression utilities
    std::vector<uint8_t> compress_data(const std::vector<uint8_t>& data) const;
    std::vector<uint8_t> decompress_data(const std::vector<uint8_t>& data) const;

    // Subscriber management
    void cleanup_inactive_subscribers();
    bool should_send_update(const SubscriberInfo& subscriber) const;
    void enqueue_update_for_subscriber(const std::string& subscriber_id,
                                     const VisualizationData& data);

    // WebSocket communication
    void send_to_websocket(const std::string& connection_id,
                          const std::string& channel,
                          const std::vector<uint8_t>& data);

    // Performance monitoring
    void update_frame_metrics(double processing_time_ms);
    void update_latency_metrics(double latency_ms);

    // Configuration
    StreamConfig config_;

    // State
    std::atomic<bool> initialized_{false};
    std::atomic<bool> streaming_{false};
    std::atomic<bool> shutdown_requested_{false};

    // Audio processing components
    std::shared_ptr<core::dsp::SpectrumAnalyzer> spectrum_analyzer_;
    std::shared_ptr<core::dsp::WaveformProcessor> waveform_processor_;
    std::shared_ptr<core::dsp::VUMeter> vu_meter_;

    // WebSocket server
    WebSocketServer* websocket_server_;

    // Threading
    std::thread audio_thread_;
    std::thread streaming_thread_;
    std::thread metrics_thread_;
    std::mutex audio_mutex_;
    std::mutex subscriber_mutex_;
    std::condition_variable audio_cv_;
    std::condition_variable streaming_cv_;

    // Audio buffer
    std::vector<float> audio_buffer_;
    std::atomic<bool> new_audio_data_{false};

    // Subscribers
    std::unordered_map<std::string, std::unique_ptr<SubscriberInfo>> subscribers_;
    std::unordered_map<std::string, std::vector<std::string>> connection_subscribers_;

    // Data buffers
    std::queue<VisualizationData> data_queue_;
    std::mutex queue_mutex_;
    static constexpr size_t MAX_QUEUE_SIZE = 100;

    // Metrics
    mutable std::mutex metrics_mutex_;
    StreamMetrics metrics_;

    // Performance tracking
    std::chrono::steady_clock::time_point last_metrics_update_;
    uint64_t frame_count_ = 0;
    double cumulative_processing_time_ = 0.0;
};

/**
 * Factory for creating and managing realtime streamer instances
 */
class RealtimeStreamerFactory {
public:
    static std::unique_ptr<RealtimeStreamer> create_streamer(const StreamConfig& config = StreamConfig{});
    static std::shared_ptr<RealtimeStreamer> get_shared_streamer(const StreamConfig& config = StreamConfig{});
    static void shutdown_all_streamers();

private:
    static std::unordered_map<std::string, std::weak_ptr<RealtimeStreamer>> streamers_;
    static std::mutex streamers_mutex_;
};

/**
 * Utility functions for real-time streaming
 */
namespace streaming_utils {
    // Frequency conversion utilities
    float frequency_to_mel(float frequency_hz);
    float frequency_to_bark(float frequency_hz);
    std::vector<float> generate_logarithmic_frequencies(float min_freq, float max_freq, size_t num_bins);

    // Data compression utilities
    std::vector<float> compress_dynamic_range(const std::vector<float>& data, float ratio = 20.0f);
    std::vector<float> apply_logarithmic_scale(const std::vector<float>& data);
    std::vector<uint8_t> delta_encode(const std::vector<float>& data, float precision = 0.001f);

    // Performance utilities
    double calculate_framerate(const std::vector<std::chrono::steady_clock::time_point>& timestamps);
    std::string format_bandwidth(uint64_t bytes_per_second);
    std::string format_latency(double latency_ms);

    // Validation utilities
    bool validate_spectrum_data(const std::vector<float>& magnitudes, size_t expected_size);
    bool validate_waveform_data(const std::vector<float>& waveform, size_t expected_size);
    bool validate_audio_levels(const std::vector<float>& levels, size_t expected_channels);
}

} // namespace vortex::network