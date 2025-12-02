#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <chrono>
#include <functional>
#include <queue>
#include <atomic>
#include <condition_variable>

#include "network/realtime_streaming.hpp"
#include "system/logger.hpp"

namespace vortex::network {

/**
 * Subscription types for audio visualization data
 */
enum class SubscriptionType : uint8_t {
    UNKNOWN = 0,
    SPECTRUM = 1,
    WAVEFORM = 2,
    VU_LEVELS = 3,
    PEAKS = 4,
    ZERO_CROSSINGS = 5,
    ENVELOPE = 6,
    PHASE = 7,
    STEREO_IMAGE = 8,
    CORRELATION = 9,
    HISTOGRAM = 10,
    WATERFALL = 11,
    SPECTROGRAM = 12,
    METADATA = 13,
    PERFORMANCE_STATS = 14,
    ALL = 255
};

/**
 * Subscription quality of service levels
 */
enum class SubscriptionQoS : uint8_t {
    BEST_EFFORT = 0,      // No guarantees, may drop frames
    LOW_LATENCY = 1,       // Prioritize speed over completeness
    RELIABLE = 2,          // Guaranteed delivery, may increase latency
    HIGH_QUALITY = 3       // Maximum quality, highest resource usage
};

/**
 * Subscription status
 */
enum class SubscriptionStatus : uint8_t {
    INACTIVE = 0,          // Not subscribed
    PENDING = 1,           // Waiting for activation
    ACTIVE = 2,            // Currently active
    PAUSED = 3,            // Temporarily paused
    ERROR = 4,             // Error state
    TERMINATED = 5         // Terminated
};

/**
 * Subscription parameters
 */
struct SubscriptionParameters {
    float update_frequency = 60.0f;          // Updates per second
    SubscriptionQoS qos = SubscriptionQoS::BEST_EFFORT;
    bool enable_compression = true;
    uint8_t compression_level = 6;
    bool enable_checksum = true;
    uint32_t max_message_size = 64 * 1024;   // 64KB
    uint32_t buffer_size = 1000;              // Buffer size
    float max_latency_ms = 100.0f;           // Maximum acceptable latency
    bool enable_delta_encoding = false;
    std::string custom_data_filter;           // JSON filter string
    std::unordered_map<std::string, std::string> custom_properties;

    // Spectrum-specific parameters
    uint32_t spectrum_fft_size = 2048;
    std::string spectrum_window_type = "hanning";
    std::string spectrum_frequency_scale = "logarithmic";
    float spectrum_min_frequency = 20.0f;
    float spectrum_max_frequency = 20000.0f;

    // Waveform-specific parameters
    uint32_t waveform_length = 512;
    std::string waveform_display_mode = "peaks";
    float waveform_window_duration = 0.1f;

    // VU-specific parameters
    std::string vu_meter_type = "rms";
    std::string vu_reference_level = "dBFS_20";
    float vu_attack_time_ms = 1.0f;
    float vu_release_time_ms = 100.0f;
    float vu_peak_hold_time_ms = 500.0f;
};

/**
 * Individual subscription
 */
struct Subscription {
    std::string subscription_id;
    std::string connection_id;
    SubscriptionType type;
    SubscriptionStatus status;
    SubscriptionParameters parameters;

    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point last_update;
    std::chrono::steady_clock::time_point last_sent;

    // Statistics
    std::atomic<uint64_t> messages_sent{0};
    std::atomic<uint64_t> messages_dropped{0};
    std::atomic<uint64_t> bytes_sent{0};
    std::atomic<double> avg_latency_ms{0.0};
    std::atomic<double> max_latency_ms{0.0};

    // Rate limiting
    std::chrono::steady_clock::time_point rate_limit_reset;
    std::atomic<uint32_t> messages_in_window{0};

    // Priority
    uint8_t priority = 0; // Higher number = higher priority

    Subscription() = default;
    Subscription(const std::string& id, const std::string& conn_id, SubscriptionType t)
        : subscription_id(id), connection_id(conn_id), type(t), status(SubscriptionStatus::PENDING),
          created_at(std::chrono::steady_clock::now()),
          last_update(std::chrono::steady_clock::now()),
          last_sent(std::chrono::steady_clock::time_point::min()),
          rate_limit_reset(std::chrono::steady_clock::now()) {}
};

/**
 * Connection summary for client
 */
struct ConnectionSummary {
    std::string connection_id;
    std::string client_type;
    std::string ip_address;
    std::chrono::steady_clock::time_point connected_at;
    std::chrono::steady_clock::time_point last_activity;

    // Subscription statistics
    std::atomic<uint32_t> active_subscriptions{0};
    std::atomic<uint32_t> pending_subscriptions{0};
    std::atomic<uint64_t> total_messages_sent{0};
    std::atomic<uint64_t> total_bytes_sent{0};

    // Performance metrics
    std::atomic<double> avg_latency_ms{0.0};
    std::atomic<float> bandwidth_mbps{0.0f};

    ConnectionSummary(const std::string& id) : connection_id(id),
        connected_at(std::chrono::steady_clock::now()),
        last_activity(std::chrono::steady_clock::now()) {}
};

/**
 * Subscription manager for WebSocket clients
 *
 * This component handles subscription management for real-time audio
 * visualization data streaming. It provides:
 * - Dynamic subscription creation and management
 * - Quality of service control
 * - Rate limiting and throttling
 * - Client connection tracking
 * - Performance monitoring and metrics
 * - Automatic cleanup of inactive connections
 * - Subscription priority handling
 * - Custom filtering and routing
 */
class SubscriptionManager {
public:
    SubscriptionManager();
    ~SubscriptionManager();

    // Configuration
    struct Config {
        uint32_t max_subscriptions = 10000;       // Maximum total subscriptions
        uint32_t max_subscriptions_per_connection = 50;
        float default_update_frequency = 60.0f;   // Default updates per second
        SubscriptionQoS default_qos = SubscriptionQoS::BEST_EFFORT;
        uint32_t max_message_queue_size = 1000;   // Per-subscription queue size
        float connection_timeout_seconds = 300.0f; // 5 minutes
        float cleanup_interval_seconds = 60.0f;    // Cleanup interval
        bool enable_metrics = true;
        bool enable_compression = true;
        uint32_t rate_limit_messages_per_second = 1000;
        float max_latency_ms = 100.0f;            // Maximum acceptable latency
        bool enable_subscription_prioritization = true;
        uint8_t max_priority_levels = 10;
    };

    // Lifecycle
    bool initialize(const Config& config);
    void shutdown();
    bool is_initialized() const { return initialized_; }

    // Connection management
    bool add_connection(const std::string& connection_id,
                       const std::string& client_type = "unknown",
                       const std::string& ip_address = "");

    bool remove_connection(const std::string& connection_id);
    bool update_connection_activity(const std::string& connection_id);
    bool has_connection(const std::string& connection_id) const;

    // Subscription management
    std::string add_subscription(const std::string& connection_id,
                                SubscriptionType type,
                                const SubscriptionParameters& params = SubscriptionParameters{});

    bool remove_subscription(const std::string& subscription_id);
    bool update_subscription(const std::string& subscription_id,
                            const SubscriptionParameters& params);

    bool pause_subscription(const std::string& subscription_id);
    bool resume_subscription(const std::string& subscription_id);
    bool set_subscription_priority(const std::string& subscription_id, uint8_t priority);

    // Subscription queries
    std::shared_ptr<Subscription> get_subscription(const std::string& subscription_id) const;
    std::vector<std::shared_ptr<Subscription>> get_subscriptions_for_connection(const std::string& connection_id) const;
    std::vector<std::shared_ptr<Subscription>> get_subscriptions_by_type(SubscriptionType type) const;
    std::vector<std::shared_ptr<Subscription>> get_active_subscriptions() const;

    // Batch operations
    std::vector<std::string> add_subscriptions(const std::string& connection_id,
                                              const std::vector<SubscriptionType>& types,
                                              const std::vector<SubscriptionParameters>& params);

    bool remove_subscriptions_for_connection(const std::string& connection_id);
    bool pause_subscriptions_for_connection(const std::string& connection_id);
    bool resume_subscriptions_for_connection(const std::string& connection_id);

    // Rate limiting and QoS
    bool should_send_message(const std::shared_ptr<Subscription>& subscription) const;
    void update_subscription_metrics(const std::string& subscription_id,
                                   double latency_ms,
                                   size_t message_size);

    // Connection and subscription statistics
    std::shared_ptr<ConnectionSummary> get_connection_summary(const std::string& connection_id) const;
    std::vector<std::shared_ptr<ConnectionSummary>> get_all_connections() const;

    // System-wide statistics
    struct Statistics {
        uint32_t total_connections = 0;
        uint32_t active_connections = 0;
        uint32_t total_subscriptions = 0;
        uint32_t active_subscriptions = 0;
        uint64_t total_messages_sent = 0;
        uint64_t total_messages_dropped = 0;
        uint64_t total_bytes_sent = 0;
        double avg_latency_ms = 0.0;
        double peak_latency_ms = 0.0;
        float total_bandwidth_mbps = 0.0f;
        std::chrono::steady_clock::time_point start_time;
    };

    Statistics get_statistics() const;
    void reset_statistics();

    // Maintenance
    void cleanup_inactive_connections();
    void enforce_rate_limits();
    void update_qos_levels();

    // Event callbacks
    using SubscriptionEventCallback = std::function<void(const std::shared_ptr<Subscription>&)>;
    using ConnectionEventCallback = std::function<void(const std::shared_ptr<ConnectionSummary>&)>;

    void set_subscription_added_callback(SubscriptionEventCallback callback);
    void set_subscription_removed_callback(SubscriptionEventCallback callback);
    void set_connection_added_callback(ConnectionEventCallback callback);
    void set_connection_removed_callback(ConnectionEventCallback callback);

    // Configuration access
    const Config& get_config() const { return config_; }
    void update_config(const Config& config);

    // Utility functions
    std::string subscription_type_to_string(SubscriptionType type) const;
    std::string qos_to_string(SubscriptionQoS qos) const;
    std::string status_to_string(SubscriptionStatus status) const;
    SubscriptionType string_to_subscription_type(const std::string& type) const;
    SubscriptionQoS string_to_qos(const std::string& qos) const;

    // Export/Import
    std::string export_subscriptions() const;
    bool import_subscriptions(const std::string& json_data);

    // Diagnostics
    std::string get_diagnostics_report() const;
    bool validate_subscription_parameters(const SubscriptionParameters& params) const;

private:
    // Internal data structures
    mutable std::shared_mutex subscriptions_mutex_;
    std::unordered_map<std::string, std::shared_ptr<Subscription>> subscriptions_;

    mutable std::shared_mutex connections_mutex_;
    std::unordered_map<std::string, std::shared_ptr<ConnectionSummary>> connections_;

    // Type and connection mappings for efficient queries
    mutable std::shared_mutex mappings_mutex_;
    std::unordered_map<SubscriptionType, std::unordered_set<std::string>> type_to_subscriptions_;
    std::unordered_map<std::string, std::unordered_set<std::string>> connection_to_subscriptions_;

    // Priority queue for message sending order
    mutable std::mutex priority_mutex_;
    std::vector<std::vector<std::string>> priority_queues_;

    // Configuration
    Config config_;
    std::atomic<bool> initialized_{false};

    // Statistics
    mutable std::mutex stats_mutex_;
    Statistics statistics_;

    // Cleanup thread
    std::thread cleanup_thread_;
    std::atomic<bool> cleanup_thread_running_{false};

    // Event callbacks
    mutable std::mutex callbacks_mutex_;
    SubscriptionEventCallback subscription_added_callback_;
    SubscriptionEventCallback subscription_removed_callback_;
    ConnectionEventCallback connection_added_callback_;
    ConnectionEventCallback connection_removed_callback_;

    // Internal helper methods
    std::string generate_subscription_id() const;
    void update_type_mapping(SubscriptionType type, const std::string& subscription_id, bool add);
    void update_connection_mapping(const std::string& connection_id, const std::string& subscription_id, bool add);
    void update_priority_queue(const std::shared_ptr<Subscription>& subscription, bool add);
    void update_statistics();
    void cleanup_thread_func();

    // Validation helpers
    bool is_connection_limit_reached(const std::string& connection_id) const;
    bool is_subscription_limit_reached(const std::string& connection_id) const;
    bool validate_connection_id(const std::string& connection_id) const;
    bool validate_subscription_id(const std::string& subscription_id) const;

    // Rate limiting helpers
    bool check_rate_limit(const std::shared_ptr<Subscription>& subscription) const;
    void reset_rate_limit_window(const std::shared_ptr<Subscription>& subscription);

    // QoS helpers
    void apply_qos_settings(std::shared_ptr<Subscription>& subscription);
    bool meets_qos_requirements(const std::shared_ptr<Subscription>& subscription,
                               double current_latency_ms) const;

    // Priority helpers
    std::vector<std::string> get_subscriptions_by_priority() const;
    uint8_t calculate_effective_priority(const std::shared_ptr<Subscription>& subscription) const;

    // Metrics helpers
    void update_connection_metrics(const std::string& connection_id,
                                 double latency_ms,
                                 size_t message_size);
    void update_global_statistics();
};

/**
 * Factory for creating subscription managers with common configurations
 */
class SubscriptionManagerFactory {
public:
    static std::unique_ptr<SubscriptionManager> create_default();
    static std::unique_ptr<SubscriptionManager> create_high_performance();
    static std::unique_ptr<SubscriptionManager> create_reliable();
    static std::unique_ptr<SubscriptionManager> create_low_latency();
};

/**
 * Utility functions for subscription management
 */
namespace subscription_utils {
    // Parameter validation
    bool validate_frequency(float frequency);
    bool validate_qos_level(SubscriptionQoS qos);
    bool validate_message_size(uint32_t size);

    // JSON conversion
    std::string subscription_parameters_to_json(const SubscriptionParameters& params);
    SubscriptionParameters json_to_subscription_parameters(const std::string& json);

    // Performance calculations
    double calculate_bandwidth_utilization(uint32_t messages_per_second, size_t avg_message_size);
    double calculate_estimated_latency(uint32_t queue_depth, float processing_time_ms);
    float calculate_qos_priority_score(const SubscriptionParameters& params, SubscriptionType type);

    // Subscription filtering
    bool matches_filter(const std::shared_ptr<Subscription>& subscription,
                      const std::string& filter_criteria);

    // Connection classification
    std::string classify_connection_type(const std::string& user_agent,
                                       const std::vector<SubscriptionType>& subscriptions);
}

} // namespace vortex::network