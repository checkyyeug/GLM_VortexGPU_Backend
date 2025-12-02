#include "network/realtime_streaming.hpp"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <chrono>
#include <sstream>
#include <iomanip>

// Include compression libraries
#ifdef VORTEX_ENABLE_ZLIB
#include <zlib.h>
#endif

namespace vortex::network {

// RealtimeStreamer implementation
RealtimeStreamer::RealtimeStreamer()
    : websocket_server_(nullptr) {
    Logger::info("RealtimeStreamer: Creating instance");
}

RealtimeStreamer::~RealtimeStreamer() {
    shutdown();
}

bool RealtimeStreamer::initialize(const StreamConfig& config) {
    if (initialized_.load()) {
        Logger::warn("RealtimeStreamer already initialized");
        return true;
    }

    config_ = config;

    Logger::info("RealtimeStreamer: Initializing with target FPS: {}, sample rate: {:.0f}Hz",
                 config_.target_fps, config_.sample_rate);

    try {
        // Initialize audio buffer
        audio_buffer_.resize(config_.buffer_size * config_.channels, 0.0f);

        // Initialize metrics
        metrics_.start_time = std::chrono::steady_clock::now();
        metrics_.last_frame_time = metrics_.start_time;
        last_metrics_update_ = metrics_.start_time;

        initialized_ = true;
        Logger::info("RealtimeStreamer: Initialization complete");
        return true;

    } catch (const std::exception& e) {
        Logger::error("RealtimeStreamer: Exception during initialization: {}", e.what());
        return false;
    }
}

void RealtimeStreamer::shutdown() {
    if (!initialized_.load()) {
        return;
    }

    Logger::info("RealtimeStreamer: Shutting down");

    // Stop streaming
    stop_streaming();

    // Signal shutdown
    shutdown_requested_.store(true);
    audio_cv_.notify_all();
    streaming_cv_.notify_all();

    // Wait for threads to finish
    if (audio_thread_.joinable()) {
        audio_thread_.join();
    }
    if (streaming_thread_.joinable()) {
        streaming_thread_.join();
    }
    if (metrics_thread_.joinable()) {
        metrics_thread_.join();
    }

    // Clear subscribers
    {
        std::lock_guard<std::mutex> lock(subscriber_mutex_);
        subscribers_.clear();
        connection_subscribers_.clear();
    }

    // Clear data queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        std::queue<VisualizationData> empty;
        data_queue_.swap(empty);
    }

    // Log final metrics
    if (config_.enable_metrics) {
        Logger::info("RealtimeStreamer final metrics:\n{}", get_performance_report());
    }

    initialized_ = false;
    shutdown_requested_.store(false);
    Logger::info("RealtimeStreamer: Shutdown complete");
}

void RealtimeStreamer::set_audio_sources(std::shared_ptr<core::dsp::SpectrumAnalyzer> spectrum_analyzer,
                                        std::shared_ptr<core::dsp::WaveformProcessor> waveform_processor,
                                        std::shared_ptr<core::dsp::VUMeter> vu_meter) {
    spectrum_analyzer_ = spectrum_analyzer;
    waveform_processor_ = waveform_processor;
    vu_meter_ = vu_meter;

    Logger::info("RealtimeStreamer: Audio sources set - Spectrum: {}, Waveform: {}, VU: {}",
                 spectrum_analyzer ? "Yes" : "No",
                 waveform_processor ? "Yes" : "No",
                 vu_meter ? "Yes" : "No");
}

void RealtimeStreamer::set_websocket_server(WebSocketServer* server) {
    websocket_server_ = server;
    Logger::info("RealtimeStreamer: WebSocket server {}", server ? "set" : "cleared");
}

bool RealtimeStreamer::start_streaming() {
    if (!initialized_.load()) {
        Logger::error("RealtimeStreamer: Cannot start - not initialized");
        return false;
    }

    if (streaming_.load()) {
        Logger::warn("RealtimeStreamer: Already streaming");
        return true;
    }

    Logger::info("RealtimeStreamer: Starting streaming with {:.1f}ms update interval",
                 config_.update_interval_ms);

    try {
        // Reset shutdown flag
        shutdown_requested_.store(false);

        // Start audio processing thread
        audio_thread_ = std::thread(&RealtimeStreamer::audio_processing_thread, this);

        // Start streaming thread
        streaming_thread_ = std::thread(&RealtimeStreamer::streaming_thread, this);

        // Start metrics thread if enabled
        if (config_.enable_metrics) {
            metrics_thread_ = std::thread(&RealtimeStreamer::metrics_thread, this);
        }

        streaming_.store(true);
        Logger::info("RealtimeStreamer: Streaming started successfully");
        return true;

    } catch (const std::exception& e) {
        Logger::error("RealtimeStreamer: Exception starting streaming: {}", e.what());
        return false;
    }
}

void RealtimeStreamer::stop_streaming() {
    if (!streaming_.load()) {
        return;
    }

    Logger::info("RealtimeStreamer: Stopping streaming");

    streaming_.store(false);
    shutdown_requested_.store(true);

    // Wake up threads
    audio_cv_.notify_all();
    streaming_cv_.notify_all();

    // Wait for threads to finish
    if (audio_thread_.joinable()) {
        audio_thread_.join();
    }
    if (streaming_thread_.joinable()) {
        streaming_thread_.join();
    }
    if (metrics_thread_.joinable()) {
        metrics_thread_.join();
    }

    Logger::info("RealtimeStreamer: Streaming stopped");
}

bool RealtimeStreamer::add_subscriber(const std::string& subscriber_id,
                                     const std::string& connection_id,
                                     const std::vector<std::string>& subscriptions,
                                     float update_frequency) {
    if (!initialized_.load()) {
        Logger::error("RealtimeStreamer: Cannot add subscriber - not initialized");
        return false;
    }

    std::lock_guard<std::mutex> lock(subscriber_mutex_);

    // Check if subscriber already exists
    if (subscribers_.find(subscriber_id) != subscribers_.end()) {
        Logger::warn("RealtimeStreamer: Subscriber {} already exists", subscriber_id);
        return false;
    }

    // Validate subscriptions
    for (const auto& sub : subscriptions) {
        if (sub != "spectrum" && sub != "waveform" && sub != "levels") {
            Logger::error("RealtimeStreamer: Invalid subscription type: {}", sub);
            return false;
        }
    }

    // Validate update frequency
    if (update_frequency <= 0.0f || update_frequency > 1000.0f) {
        Logger::error("RealtimeStreamer: Invalid update frequency: {:.1f}", update_frequency);
        return false;
    }

    // Create subscriber
    auto subscriber = std::make_unique<SubscriberInfo>();
    subscriber->subscriber_id = subscriber_id;
    subscriber->connection_id = connection_id;
    subscriber->subscriptions = subscriptions;
    subscriber->update_frequency = update_frequency;
    subscriber->last_update = std::chrono::steady_clock::now();

    // Add to subscribers map
    subscribers_[subscriber_id] = std::move(subscriber);

    // Add to connection mapping
    connection_subscribers_[connection_id].push_back(subscriber_id);

    Logger::info("RealtimeStreamer: Added subscriber {} for connection {} with subscriptions: [{}]",
                 subscriber_id, connection_id, fmt::join(subscriptions, ", "));

    return true;
}

bool RealtimeStreamer::remove_subscriber(const std::string& subscriber_id) {
    std::lock_guard<std::mutex> lock(subscriber_mutex_);

    auto it = subscribers_.find(subscriber_id);
    if (it == subscribers_.end()) {
        Logger::warn("RealtimeStreamer: Subscriber {} not found", subscriber_id);
        return false;
    }

    std::string connection_id = it->second->connection_id;

    // Remove from connection mapping
    auto conn_it = connection_subscribers_.find(connection_id);
    if (conn_it != connection_subscribers_.end()) {
        auto& subs = conn_it->second;
        subs.erase(std::remove(subs.begin(), subs.end(), subscriber_id), subs.end());
        if (subs.empty()) {
            connection_subscribers_.erase(conn_it);
        }
    }

    // Remove subscriber
    subscribers_.erase(it);

    Logger::info("RealtimeStreamer: Removed subscriber {} for connection {}", subscriber_id, connection_id);
    return true;
}

bool RealtimeStreamer::update_subscription(const std::string& subscriber_id,
                                         const std::vector<std::string>& subscriptions,
                                         float update_frequency) {
    std::lock_guard<std::mutex> lock(subscriber_mutex_);

    auto it = subscribers_.find(subscriber_id);
    if (it == subscribers_.end()) {
        Logger::warn("RealtimeStreamer: Subscriber {} not found for update", subscriber_id);
        return false;
    }

    // Validate subscriptions
    for (const auto& sub : subscriptions) {
        if (sub != "spectrum" && sub != "waveform" && sub != "levels") {
            Logger::error("RealtimeStreamer: Invalid subscription type: {}", sub);
            return false;
        }
    }

    // Validate update frequency
    if (update_frequency <= 0.0f || update_frequency > 1000.0f) {
        Logger::error("RealtimeStreamer: Invalid update frequency: {:.1f}", update_frequency);
        return false;
    }

    // Update subscriber
    it->second->subscriptions = subscriptions;
    it->second->update_frequency = update_frequency;
    it->second->last_update = std::chrono::steady_clock::now();

    Logger::info("RealtimeStreamer: Updated subscriber {} with subscriptions: [{}], frequency: {:.1f}Hz",
                 subscriber_id, fmt::join(subscriptions, ", "), update_frequency);

    return true;
}

bool RealtimeStreamer::process_audio_frame(const float* audio_data, size_t num_samples) {
    if (!initialized_.load() || !audio_data || num_samples == 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(audio_mutex_);

    // Copy audio data to buffer
    size_t samples_to_copy = std::min(num_samples, audio_buffer_.size() / config_.channels);
    std::memcpy(audio_buffer_.data(), audio_data,
                samples_to_copy * config_.channels * sizeof(float));

    new_audio_data_.store(true);
    audio_cv_.notify_one();

    return true;
}

void RealtimeStreamer::force_update() {
    if (!initialized_.load()) {
        return;
    }

    new_audio_data_.store(true);
    audio_cv_.notify_one();
}

void RealtimeStreamer::audio_processing_thread() {
    Logger::info("RealtimeStreamer: Audio processing thread started");

    while (!shutdown_requested_.load()) {
        std::unique_lock<std::mutex> lock(audio_mutex_);

        // Wait for new audio data
        audio_cv_.wait(lock, [this] { return new_audio_data_.load() || shutdown_requested_.load(); });

        if (shutdown_requested_.load()) {
            break;
        }

        if (!new_audio_data_.load()) {
            continue;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        // Collect visualization data
        VisualizationData data{};
        if (collect_visualization_data(data)) {
            // Add to queue
            {
                std::lock_guard<std::mutex> queue_lock(queue_mutex_);
                if (data_queue_.size() < MAX_QUEUE_SIZE) {
                    data_queue_.push(data);
                } else {
                    // Queue full, drop oldest frame
                    data_queue_.pop();
                    data_queue_.push(data);
                    metrics_.dropped_frames.fetch_add(1);
                }
            }

            // Notify streaming thread
            streaming_cv_.notify_one();
        }

        new_audio_data_.store(false);

        // Update performance metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        update_frame_metrics(duration.count() / 1000.0);
    }

    Logger::info("RealtimeStreamer: Audio processing thread stopped");
}

void RealtimeStreamer::streaming_thread() {
    Logger::info("RealtimeStreamer: Streaming thread started");

    while (!shutdown_requested_.load()) {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        // Wait for new data
        streaming_cv_.wait(lock, [this] {
            return !data_queue_.empty() || shutdown_requested_.load();
        });

        if (shutdown_requested_.load()) {
            break;
        }

        if (data_queue_.empty()) {
            continue;
        }

        // Get latest data
        VisualizationData data = data_queue_.front();
        data_queue_.pop();
        lock.unlock();

        // Process subscriber updates
        process_subscriber_updates();
    }

    Logger::info("RealtimeStreamer: Streaming thread stopped");
}

void RealtimeStreamer::metrics_thread() {
    Logger::info("RealtimeStreamer: Metrics thread started");

    const auto metrics_interval = std::chrono::seconds(5);

    while (!shutdown_requested_.load()) {
        std::this_thread::sleep_for(metrics_interval);

        if (shutdown_requested_.load()) {
            break;
        }

        update_latency_metrics(0.0); // Will calculate from subscriber stats
        cleanup_inactive_subscribers();
    }

    Logger::info("RealtimeStreamer: Metrics thread stopped");
}

bool RealtimeStreamer::collect_visualization_data(VisualizationData& data) {
    auto start_time = std::chrono::high_resolution_clock::now();

    data.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        start_time.time_since_epoch()).count();
    data.is_valid = false;

    try {
        // Collect spectrum data
        if (spectrum_analyzer_ && spectrum_analyzer_->isInitialized()) {
            auto spectrum_data = spectrum_analyzer_->processAudio(audio_buffer_.data(),
                                                               audio_buffer_.size() / config_.channels);
            if (!spectrum_data.empty()) {
                data.spectrum_magnitudes = spectrum_data[0]; // Use first channel
                Logger::debug("Collected spectrum data: {} bins", data.spectrum_magnitudes.size());
            }
        }

        // Collect waveform data
        if (waveform_processor_ && waveform_processor_->isInitialized()) {
            auto waveform_data = waveform_processor_->processAudio(audio_buffer_.data(),
                                                                 audio_buffer_.size() / config_.channels);
            if (!waveform_data.empty()) {
                data.waveform_peaks = waveform_data[0].samples;
                if (!waveform_data[0].peaks.empty()) {
                    data.waveform_rms = waveform_data[0].peaks;
                }
                Logger::debug("Collected waveform data: {} samples", data.waveform_peaks.size());
            }
        }

        // Collect VU meter data
        if (vu_meter_ && vu_meter_->isInitialized()) {
            auto levels = vu_meter_->getCurrentLevels();
            if (!levels.empty()) {
                data.vu_levels = levels;
                Logger::debug("Collected VU data: {} channels", data.vu_levels.size());
            }
        }

        data.is_valid = true;

    } catch (const std::exception& e) {
        Logger::error("RealtimeStreamer: Exception collecting data: {}", e.what());
        return false;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    data.processing_time_ms = duration.count() / 1000.0;

    Logger::debug("Collected visualization data in {:.3f}ms", data.processing_time_ms);
    return true;
}

void RealtimeStreamer::process_subscriber_updates() {
    std::lock_guard<std::mutex> lock(subscriber_mutex_);

    auto current_time = std::chrono::steady_clock::now();

    for (auto& [subscriber_id, subscriber] : subscribers_) {
        // Check if this subscriber needs an update
        if (!should_send_update(*subscriber)) {
            continue;
        }

        // Get latest data from queue
        VisualizationData data{};
        bool has_data = false;
        {
            std::lock_guard<std::mutex> queue_lock(queue_mutex_);
            if (!data_queue_.empty()) {
                data = data_queue_.back(); // Get most recent data
                has_data = true;
            }
        }

        if (!has_data) {
            continue;
        }

        // Send updates for each subscription
        for (const auto& subscription : subscriber->subscriptions) {
            std::vector<uint8_t> message_data;

            if (subscription == "spectrum") {
                message_data = create_spectrum_message(data);
            } else if (subscription == "waveform") {
                message_data = create_waveform_message(data);
            } else if (subscription == "levels") {
                message_data = create_levels_message(data);
            }

            if (!message_data.empty()) {
                send_to_websocket(subscriber->connection_id, subscription, message_data);
                metrics_.total_frames_sent.fetch_add(1);
                metrics_.total_bytes_sent.fetch_add(message_data.size());
                subscriber->frames_sent.fetch_add(1);
            }
        }

        subscriber->last_update = current_time;
    }
}

std::vector<uint8_t> RealtimeStreamer::create_spectrum_message(const VisualizationData& data) const {
    if (data.spectrum_magnitudes.empty()) {
        return {};
    }

    // Create binary message format
    std::vector<uint8_t> message;

    // Header
    message.push_back(0x01); // Version
    message.push_back(0x03); // Data message type
    message.push_back('S');  // Spectrum identifier
    message.push_back('P');  // Spectrum identifier

    // Timestamp (8 bytes)
    uint64_t timestamp = data.timestamp;
    message.insert(message.end(), reinterpret_cast<uint8_t*>(&timestamp),
                   reinterpret_cast<uint8_t*>(&timestamp) + 8);

    // Sample rate (4 bytes)
    float sample_rate = config_.sample_rate;
    message.insert(message.end(), reinterpret_cast<uint8_t*>(&sample_rate),
                   reinterpret_cast<uint8_t*>(&sample_rate) + 4);

    // FFT size (4 bytes)
    uint32_t fft_size = config_.spectrum_fft_size;
    message.insert(message.end(), reinterpret_cast<uint8_t*>(&fft_size),
                   reinterpret_cast<uint8_t*>(&fft_size) + 4);

    // Number of bins (4 bytes)
    uint32_t num_bins = static_cast<uint32_t>(data.spectrum_magnitudes.size());
    message.insert(message.end(), reinterpret_cast<uint8_t*>(&num_bins),
                   reinterpret_cast<uint8_t*>(&num_bins) + 4);

    // Magnitude data
    for (float magnitude : data.spectrum_magnitudes) {
        message.insert(message.end(), reinterpret_cast<uint8_t*>(&magnitude),
                       reinterpret_cast<uint8_t*>(&magnitude) + 4);
    }

    // Apply compression if enabled
    if (config_.enable_compression) {
        return compress_data(message);
    }

    return message;
}

std::vector<uint8_t> RealtimeStreamer::create_waveform_message(const VisualizationData& data) const {
    if (data.waveform_peaks.empty()) {
        return {};
    }

    std::vector<uint8_t> message;

    // Header
    message.push_back(0x01); // Version
    message.push_back(0x03); // Data message type
    message.push_back('W');  // Waveform identifier
    message.push_back('F');  // Waveform identifier

    // Timestamp (8 bytes)
    uint64_t timestamp = data.timestamp;
    message.insert(message.end(), reinterpret_cast<uint8_t*>(&timestamp),
                   reinterpret_cast<uint8_t*>(&timestamp) + 8);

    // Number of samples (4 bytes)
    uint32_t num_samples = static_cast<uint32_t>(data.waveform_peaks.size());
    message.insert(message.end(), reinterpret_cast<uint8_t*>(&num_samples),
                   reinterpret_cast<uint8_t*>(&num_samples) + 4);

    // Peak data
    for (float peak : data.waveform_peaks) {
        message.insert(message.end(), reinterpret_cast<uint8_t*>(&peak),
                       reinterpret_cast<uint8_t*>(&peak) + 4);
    }

    // RMS data (if available)
    uint32_t has_rms = data.waveform_rms.empty() ? 0 : 1;
    message.insert(message.end(), reinterpret_cast<uint8_t*>(&has_rms),
                   reinterpret_cast<uint8_t*>(&has_rms) + 4);

    if (has_rms) {
        for (float rms : data.waveform_rms) {
            message.insert(message.end(), reinterpret_cast<uint8_t*>(&rms),
                           reinterpret_cast<uint8_t*>(&rms) + 4);
        }
    }

    // Apply compression if enabled
    if (config_.enable_compression) {
        return compress_data(message);
    }

    return message;
}

std::vector<uint8_t> RealtimeStreamer::create_levels_message(const VisualizationData& data) const {
    if (data.vu_levels.empty()) {
        return {};
    }

    std::vector<uint8_t> message;

    // Header
    message.push_back(0x01); // Version
    message.push_back(0x03); // Data message type
    message.push_back('L');  // Levels identifier
    message.push_back('V');  // Levels identifier

    // Timestamp (8 bytes)
    uint64_t timestamp = data.timestamp;
    message.insert(message.end(), reinterpret_cast<uint8_t*>(&timestamp),
                   reinterpret_cast<uint8_t*>(&timestamp) + 8);

    // Number of channels (2 bytes)
    uint16_t num_channels = static_cast<uint16_t>(data.vu_levels.size());
    message.insert(message.end(), reinterpret_cast<uint8_t*>(&num_channels),
                   reinterpret_cast<uint8_t*>(&num_channels) + 2);

    // Level data (4 bytes per channel)
    for (float level : data.vu_levels) {
        message.insert(message.end(), reinterpret_cast<uint8_t*>(&level),
                       reinterpret_cast<uint8_t*>(&level) + 4);
    }

    // Apply compression if enabled
    if (config_.enable_compression) {
        return compress_data(message);
    }

    return message;
}

std::vector<uint8_t> RealtimeStreamer::compress_data(const std::vector<uint8_t>& data) const {
#ifdef VORTEX_ENABLE_ZLIB
    // Simple zlib compression
    uLongf compressed_size = compressBound(data.size());
    std::vector<uint8_t> compressed(compressed_size);

    if (compress(compressed.data(), &compressed_size,
                 data.data(), data.size()) == Z_OK) {
        compressed.resize(compressed_size);
        return compressed;
    }
#endif

    // Fallback to original data
    return data;
}

bool RealtimeStreamer::should_send_update(const SubscriberInfo& subscriber) const {
    auto current_time = std::chrono::steady_clock::now();
    auto time_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time - subscriber.last_update).count();

    double update_interval_ms = 1000.0 / subscriber.update_frequency;
    return time_since_last >= update_interval_ms;
}

void RealtimeStreamer::send_to_websocket(const std::string& connection_id,
                                        const std::string& channel,
                                        const std::vector<uint8_t>& data) {
    if (!websocket_server_) {
        Logger::warn("RealtimeStreamer: WebSocket server not available");
        return;
    }

    // This would interface with the Rust WebSocket server
    // For now, just log the action
    Logger::debug("RealtimeStreamer: Sending {} bytes to connection {} on channel {}",
                 data.size(), connection_id, channel);

    // TODO: Implement actual WebSocket communication with Rust FFI
}

void RealtimeStreamer::cleanup_inactive_subscribers() {
    std::lock_guard<std::mutex> lock(subscriber_mutex_);

    auto current_time = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::minutes(5);

    auto it = subscribers_.begin();
    while (it != subscribers_.end()) {
        auto& [subscriber_id, subscriber] = *it;

        if (current_time - subscriber->last_update > timeout) {
            Logger::info("RealtimeStreamer: Removing inactive subscriber {}", subscriber_id);

            // Remove from connection mapping
            auto conn_it = connection_subscribers_.find(subscriber->connection_id);
            if (conn_it != connection_subscribers_.end()) {
                auto& subs = conn_it->second;
                subs.erase(std::remove(subs.begin(), subs.end(), subscriber_id), subs.end());
                if (subs.empty()) {
                    connection_subscribers_.erase(conn_it);
                }
            }

            it = subscribers_.erase(it);
        } else {
            ++it;
        }
    }
}

void RealtimeStreamer::update_frame_metrics(double processing_time_ms) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    frame_count_++;
    cumulative_processing_time_ += processing_time_ms;

    auto current_time = std::chrono::steady_clock::now();
    auto time_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time - metrics_.last_frame_time).count();

    if (time_since_last > 0) {
        double current_fps = 1000.0 / time_since_last;

        if (frame_count_ == 1) {
            metrics_.min_fps = current_fps;
            metrics_.max_fps = current_fps;
        } else {
            metrics_.min_fps = std::min(metrics_.min_fps.load(), current_fps);
            metrics_.max_fps = std::max(metrics_.max_fps.load(), current_fps);
        }

        metrics_.avg_fps = (metrics_.avg_fps * (frame_count_ - 1) + current_fps) / frame_count_;
    }

    metrics_.last_frame_time = current_time;
}

void RealtimeStreamer::update_latency_metrics(double latency_ms) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    metrics_.avg_latency_ms = latency_ms;
    metrics_.peak_latency_ms = std::max(metrics_.peak_latency_ms.load(), latency_ms);
}

StreamMetrics RealtimeStreamer::get_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

std::string RealtimeStreamer::get_performance_report() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    std::lock_guard<std::mutex> sub_lock(subscriber_mutex_);

    std::ostringstream report;
    report << "=== RealtimeStreamer Performance Report ===\n";
    report << "Configuration:\n";
    report << "  Target FPS: " << config_.target_fps << "\n";
    report << "  Sample Rate: " << config_.sample_rate << " Hz\n";
    report << "  Channels: " << config_.channels << "\n";
    report << "  Buffer Size: " << config_.buffer_size << "\n\n";

    report << "Metrics:\n";
    report << "  Frames Sent: " << metrics_.total_frames_sent.load() << "\n";
    report << "  Bytes Sent: " << streaming_utils::format_bandwidth(metrics_.total_bytes_sent.load()) << "\n";
    report << "  Dropped Frames: " << metrics_.dropped_frames.load() << "\n";
    report << "  Average FPS: " << std::fixed << std::setprecision(1) << metrics_.avg_fps.load() << "\n";
    report << "  Min/Max FPS: " << metrics_.min_fps.load() << "/" << metrics_.max_fps.load() << "\n";
    report << "  Average Latency: " << streaming_utils::format_latency(metrics_.avg_latency_ms.load()) << "\n";
    report << "  Peak Latency: " << streaming_utils::format_latency(metrics_.peak_latency_ms.load()) << "\n\n";

    report << "Subscribers:\n";
    report << "  Total: " << subscribers_.size() << "\n";
    report << "  Connections: " << connection_subscribers_.size() << "\n";

    for (const auto& [conn_id, subs] : connection_subscribers_) {
        report << "  Connection " << conn_id << ": " << subs.size() << " subscribers\n";
    }

    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - metrics_.start_time).count();
    report << "  Uptime: " << uptime << " seconds\n";

    return report.str();
}

void RealtimeStreamer::reset_metrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    metrics_.total_frames_sent.store(0);
    metrics_.total_bytes_sent.store(0);
    metrics_.dropped_frames.store(0);
    metrics_.avg_latency_ms.store(0.0);
    metrics_.peak_latency_ms.store(0.0);
    metrics_.min_fps.store(0.0);
    metrics_.max_fps.store(0.0);
    metrics_.avg_fps.store(0.0);
    metrics_.start_time = std::chrono::steady_clock::now();
    frame_count_ = 0;
    cumulative_processing_time_ = 0.0;
}

// Factory implementation
std::unordered_map<std::string, std::weak_ptr<RealtimeStreamer>> RealtimeStreamerFactory::streamers_;
std::mutex RealtimeStreamerFactory::streamers_mutex_;

std::unique_ptr<RealtimeStreamer> RealtimeStreamerFactory::create_streamer(const StreamConfig& config) {
    auto streamer = std::make_unique<RealtimeStreamer>();
    if (streamer->initialize(config)) {
        return streamer;
    }
    return nullptr;
}

std::shared_ptr<RealtimeStreamer> RealtimeStreamerFactory::get_shared_streamer(const StreamConfig& config) {
    std::lock_guard<std::mutex> lock(streamers_mutex_);

    std::string key = std::to_string(config.target_fps) + "_" +
                     std::to_string(config.sample_rate) + "_" +
                     std::to_string(config.channels);

    auto it = streamers_.find(key);
    if (it != streamers_.end()) {
        auto shared = it->second.lock();
        if (shared) {
            return shared;
        }
        streamers_.erase(it);
    }

    auto shared = std::shared_ptr<RealtimeStreamer>(create_streamer(config));
    if (shared) {
        streamers_[key] = shared;
    }

    return shared;
}

void RealtimeStreamerFactory::shutdown_all_streamers() {
    std::lock_guard<std::mutex> lock(streamers_mutex_);

    for (auto& [key, weak_streamer] : streamers_) {
        if (auto streamer = weak_streamer.lock()) {
            streamer->shutdown();
        }
    }

    streamers_.clear();
}

// Utility functions implementation
namespace streaming_utils {

float frequency_to_mel(float frequency_hz) {
    return 2595.0f * std::log10(1.0f + frequency_hz / 700.0f);
}

float frequency_to_bark(float frequency_hz) {
    return 13.0f * std::atan(0.00076f * frequency_hz) +
           3.5f * std::atan((frequency_hz / 7500.0f) * (frequency_hz / 7500.0f));
}

std::vector<float> generate_logarithmic_frequencies(float min_freq, float max_freq, size_t num_bins) {
    std::vector<float> frequencies;
    frequencies.reserve(num_bins);

    float log_min = std::log(min_freq);
    float log_max = std::log(max_freq);
    float log_range = log_max - log_min;

    for (size_t i = 0; i < num_bins; ++i) {
        float log_freq = log_min + (log_range * i) / (num_bins - 1);
        frequencies.push_back(std::exp(log_freq));
    }

    return frequencies;
}

std::vector<float> compress_dynamic_range(const std::vector<float>& data, float ratio) {
    std::vector<float> compressed;
    compressed.reserve(data.size());

    float threshold = 0.7f; // Compression threshold

    for (float sample : data) {
        float abs_sample = std::abs(sample);
        if (abs_sample > threshold) {
            float compressed_level = threshold + (abs_sample - threshold) / ratio;
            float sign = (sample >= 0.0f) ? 1.0f : -1.0f;
            compressed.push_back(sign * compressed_level);
        } else {
            compressed.push_back(sample);
        }
    }

    return compressed;
}

std::vector<float> apply_logarithmic_scale(const std::vector<float>& data) {
    std::vector<float> log_data;
    log_data.reserve(data.size());

    for (float value : data) {
        float abs_value = std::abs(value);
        if (abs_value > 1e-6f) {
            float log_value = std::log10(abs_value) / 6.0f; // Scale to [0,1]
            log_value = std::clamp(log_value, 0.0f, 1.0f);
            float sign = (value >= 0.0f) ? 1.0f : -1.0f;
            log_data.push_back(sign * log_value);
        } else {
            log_data.push_back(0.0f);
        }
    }

    return log_data;
}

std::vector<uint8_t> delta_encode(const std::vector<float>& data, float precision) {
    std::vector<uint8_t> encoded;
    encoded.reserve(data.size());

    if (data.empty()) {
        return encoded;
    }

    // First value as absolute
    int32_t first_value = static_cast<int32_t>(data[0] / precision);
    encoded.push_back(static_cast<uint8_t>((first_value >> 24) & 0xFF));
    encoded.push_back(static_cast<uint8_t>((first_value >> 16) & 0xFF));
    encoded.push_back(static_cast<uint8_t>((first_value >> 8) & 0xFF));
    encoded.push_back(static_cast<uint8_t>(first_value & 0xFF));

    // Delta encode remaining values
    int32_t prev_value = first_value;
    for (size_t i = 1; i < data.size(); ++i) {
        int32_t current_value = static_cast<int32_t>(data[i] / precision);
        int32_t delta = current_value - prev_value;

        // Store delta as signed byte (if range fits)
        if (delta >= -128 && delta <= 127) {
            encoded.push_back(static_cast<uint8_t>(delta & 0xFF));
        } else {
            // Fallback to 4-byte encoding
            encoded.push_back(0xFF); // Escape byte
            encoded.push_back(static_cast<uint8_t>((delta >> 24) & 0xFF));
            encoded.push_back(static_cast<uint8_t>((delta >> 16) & 0xFF));
            encoded.push_back(static_cast<uint8_t>((delta >> 8) & 0xFF));
            encoded.push_back(static_cast<uint8_t>(delta & 0xFF));
        }

        prev_value = current_value;
    }

    return encoded;
}

double calculate_framerate(const std::vector<std::chrono::steady_clock::time_point>& timestamps) {
    if (timestamps.size() < 2) {
        return 0.0;
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        timestamps.back() - timestamps.front()).count();

    return duration > 0 ? (timestamps.size() - 1) * 1000.0 / duration : 0.0;
}

std::string format_bandwidth(uint64_t bytes_per_second) {
    const char* units[] = {"B/s", "KB/s", "MB/s", "GB/s"};
    int unit = 0;
    double bytes = static_cast<double>(bytes_per_second);

    while (bytes >= 1024.0 && unit < 3) {
        bytes /= 1024.0;
        unit++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << bytes << " " << units[unit];
    return oss.str();
}

std::string format_latency(double latency_ms) {
    std::ostringstream oss;
    if (latency_ms < 1.0) {
        oss << std::fixed << std::setprecision(1) << (latency_ms * 1000.0) << " μs";
    } else {
        oss << std::fixed << std::setprecision(1) << latency_ms << " ms";
    }
    return oss.str();
}

bool validate_spectrum_data(const std::vector<float>& magnitudes, size_t expected_size) {
    if (magnitudes.size() != expected_size) {
        return false;
    }

    for (float magnitude : magnitudes) {
        if (!std::isfinite(magnitude) || magnitude < -120.0f || magnitude > 0.0f) {
            return false;
        }
    }

    return true;
}

bool validate_waveform_data(const std::vector<float>& waveform, size_t expected_size) {
    if (waveform.size() != expected_size) {
        return false;
    }

    for (float sample : waveform) {
        if (!std::isfinite(sample) || sample < -1.0f || sample > 1.0f) {
            return false;
        }
    }

    return true;
}

bool validate_audio_levels(const std::vector<float>& levels, size_t expected_channels) {
    if (levels.size() != expected_channels) {
        return false;
    }

    for (float level : levels) {
        if (!std::isfinite(level) || level < -60.0f || level > 0.0f) {
            return false;
        }
    }

    return true;
}

} // namespace streaming_utils

} // namespace vortex::network