#include "network/subscription_manager.hpp"
#include <algorithm>
#include <random>
#include <sstream>
#include <iomanip>
#include <regex>
#include <unordered_set>

#ifdef VORTEX_ENABLE_NLOHMANN_JSON
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#endif

namespace vortex::network {

// SubscriptionManager implementation
SubscriptionManager::SubscriptionManager() {
    Logger::info("SubscriptionManager: Creating instance");
    statistics_.start_time = std::chrono::steady_clock::now();

    // Initialize priority queues
    priority_queues_.resize(config_.max_priority_levels);
}

SubscriptionManager::~SubscriptionManager() {
    shutdown();
}

bool SubscriptionManager::initialize(const Config& config) {
    if (initialized_.load()) {
        Logger::warn("SubscriptionManager already initialized");
        return true;
    }

    config_ = config;

    Logger::info("SubscriptionManager: Initializing with max_subscriptions={}, max_connections_per_subscription={}",
                 config_.max_subscriptions, config_.max_subscriptions_per_connection);

    try {
        // Resize priority queues based on max priority levels
        priority_queues_.resize(config_.max_priority_levels);
        for (auto& queue : priority_queues_) {
            queue.clear();
        }

        // Start cleanup thread
        cleanup_thread_running_.store(true);
        cleanup_thread_ = std::thread(&SubscriptionManager::cleanup_thread_func, this);

        initialized_.store(true);
        Logger::info("SubscriptionManager initialized successfully");
        return true;

    } catch (const std::exception& e) {
        Logger::error("SubscriptionManager: Exception during initialization: {}", e.what());
        return false;
    }
}

void SubscriptionManager::shutdown() {
    if (!initialized_.load()) {
        return;
    }

    Logger::info("SubscriptionManager: Shutting down");

    // Stop cleanup thread
    cleanup_thread_running_.store(false);
    if (cleanup_thread_.joinable()) {
        cleanup_thread_.join();
    }

    // Clear all data structures
    {
        std::unique_lock<std::shared_mutex> lock(subscriptions_mutex_);
        subscriptions_.clear();
    }

    {
        std::unique_lock<std::shared_mutex> lock(connections_mutex_);
        connections_.clear();
    }

    {
        std::unique_lock<std::shared_mutex> lock(mappings_mutex_);
        type_to_subscriptions_.clear();
        connection_to_subscriptions_.clear();
    }

    {
        std::unique_lock<std::mutex> lock(priority_mutex_);
        for (auto& queue : priority_queues_) {
            queue.clear();
        }
    }

    initialized_.store(false);
    Logger::info("SubscriptionManager shutdown complete");
}

bool SubscriptionManager::add_connection(const std::string& connection_id,
                                         const std::string& client_type,
                                         const std::string& ip_address) {
    if (!initialized_.load()) {
        Logger::error("SubscriptionManager: Cannot add connection - not initialized");
        return false;
    }

    if (!validate_connection_id(connection_id)) {
        Logger::error("SubscriptionManager: Invalid connection ID: {}", connection_id);
        return false;
    }

    std::unique_lock<std::shared_mutex> lock(connections_mutex_);

    // Check if connection already exists
    if (connections_.find(connection_id) != connections_.end()) {
        Logger::warn("SubscriptionManager: Connection {} already exists", connection_id);
        return false;
    }

    // Check if connection limit would be exceeded
    if (connections_.size() >= config_.max_subscriptions) {
        Logger::warn("SubscriptionManager: Connection limit reached, rejecting: {}", connection_id);
        return false;
    }

    // Create new connection summary
    auto connection = std::make_shared<ConnectionSummary>(connection_id);
    connection->client_type = client_type;
    connection->ip_address = ip_address;

    connections_[connection_id] = connection;

    // Update statistics
    update_statistics();

    Logger::info("SubscriptionManager: Added connection {} (client: {}, ip: {})",
                 connection_id, client_type, ip_address);

    // Call connection added callback
    {
        std::unique_lock<std::mutex> callback_lock(callbacks_mutex_);
        if (connection_added_callback_) {
            connection_added_callback_(connection);
        }
    }

    return true;
}

bool SubscriptionManager::remove_connection(const std::string& connection_id) {
    std::unique_lock<std::shared_mutex> lock(connections_mutex_);

    auto it = connections_.find(connection_id);
    if (it == connections_.end()) {
        Logger::warn("SubscriptionManager: Connection {} not found", connection_id);
        return false;
    }

    auto connection = it->second;

    // Remove all subscriptions for this connection
    remove_subscriptions_for_connection(connection_id);

    // Remove connection
    connections_.erase(it);

    // Update statistics
    update_statistics();

    Logger::info("SubscriptionManager: Removed connection {}", connection_id);

    // Call connection removed callback
    {
        std::unique_lock<std::mutex> callback_lock(callbacks_mutex_);
        if (connection_removed_callback_) {
            connection_removed_callback_(connection);
        }
    }

    return true;
}

bool SubscriptionManager::update_connection_activity(const std::string& connection_id) {
    std::shared_lock<std::shared_mutex> lock(connections_mutex_);

    auto it = connections_.find(connection_id);
    if (it == connections_.end()) {
        return false;
    }

    it->second->last_activity = std::chrono::steady_clock::now();
    return true;
}

bool SubscriptionManager::has_connection(const std::string& connection_id) const {
    std::shared_lock<std::shared_mutex> lock(connections_mutex_);
    return connections_.find(connection_id) != connections_.end();
}

std::string SubscriptionManager::add_subscription(const std::string& connection_id,
                                                  SubscriptionType type,
                                                  const SubscriptionParameters& params) {
    if (!initialized_.load()) {
        Logger::error("SubscriptionManager: Cannot add subscription - not initialized");
        return "";
    }

    if (!validate_subscription_parameters(params)) {
        Logger::error("SubscriptionManager: Invalid subscription parameters");
        return "";
    }

    if (!has_connection(connection_id)) {
        Logger::error("SubscriptionManager: Connection {} not found", connection_id);
        return "";
    }

    if (is_subscription_limit_reached(connection_id)) {
        Logger::warn("SubscriptionManager: Subscription limit reached for connection {}", connection_id);
        return "";
    }

    // Generate unique subscription ID
    std::string subscription_id = generate_subscription_id();

    // Create new subscription
    auto subscription = std::make_shared<Subscription>(subscription_id, connection_id, type);
    subscription->parameters = params;

    // Apply QoS settings
    apply_qos_settings(subscription);

    // Add to subscriptions map
    {
        std::unique_lock<std::shared_mutex> lock(subscriptions_mutex_);
        subscriptions_[subscription_id] = subscription;
    }

    // Update mappings
    update_type_mapping(type, subscription_id, true);
    update_connection_mapping(connection_id, subscription_id, true);
    update_priority_queue(subscription, true);

    // Update connection subscription count
    {
        std::shared_lock<std::shared_mutex> conn_lock(connections_mutex_);
        auto conn_it = connections_.find(connection_id);
        if (conn_it != connections_.end()) {
            conn_it->second->active_subscriptions.fetch_add(1);
        }
    }

    // Update statistics
    update_statistics();

    Logger::info("SubscriptionManager: Added subscription {} for connection {} (type: {})",
                 subscription_id, connection_id, subscription_type_to_string(type));

    // Call subscription added callback
    {
        std::unique_lock<std::mutex> callback_lock(callbacks_mutex_);
        if (subscription_added_callback_) {
            subscription_added_callback_(subscription);
        }
    }

    return subscription_id;
}

bool SubscriptionManager::remove_subscription(const std::string& subscription_id) {
    std::shared_lock<std::shared_mutex> sub_lock(subscriptions_mutex_);

    auto it = subscriptions_.find(subscription_id);
    if (it == subscriptions_.end()) {
        Logger::warn("SubscriptionManager: Subscription {} not found", subscription_id);
        return false;
    }

    auto subscription = it->second;
    std::string connection_id = subscription->connection_id;
    SubscriptionType type = subscription->type;

    // Remove from priority queue
    update_priority_queue(subscription, false);

    sub_lock.unlock();

    // Update mappings
    update_type_mapping(type, subscription_id, false);
    update_connection_mapping(connection_id, subscription_id, false);

    // Remove from subscriptions map
    {
        std::unique_lock<std::shared_mutex> lock(subscriptions_mutex_);
        subscriptions_.erase(subscription_id);
    }

    // Update connection subscription count
    {
        std::shared_lock<std::shared_mutex> conn_lock(connections_mutex_);
        auto conn_it = connections_.find(connection_id);
        if (conn_it != connections_.end()) {
            conn_it->second->active_subscriptions.fetch_sub(1);
        }
    }

    // Update statistics
    update_statistics();

    Logger::info("SubscriptionManager: Removed subscription {} for connection {}",
                 subscription_id, connection_id);

    // Call subscription removed callback
    {
        std::unique_lock<std::mutex> callback_lock(callbacks_mutex_);
        if (subscription_removed_callback_) {
            subscription_removed_callback_(subscription);
        }
    }

    return true;
}

bool SubscriptionManager::update_subscription(const std::string& subscription_id,
                                             const SubscriptionParameters& params) {
    if (!validate_subscription_parameters(params)) {
        Logger::error("SubscriptionManager: Invalid subscription parameters");
        return false;
    }

    std::shared_lock<std::shared_mutex> lock(subscriptions_mutex_);

    auto it = subscriptions_.find(subscription_id);
    if (it == subscriptions_.end()) {
        Logger::warn("SubscriptionManager: Subscription {} not found", subscription_id);
        return false;
    }

    auto subscription = it->second;
    lock.unlock();

    // Update parameters
    subscription->parameters = params;

    // Reapply QoS settings
    apply_qos_settings(subscription);

    Logger::info("SubscriptionManager: Updated subscription {}", subscription_id);
    return true;
}

bool SubscriptionManager::pause_subscription(const std::string& subscription_id) {
    std::shared_lock<std::shared_mutex> lock(subscriptions_mutex_);

    auto it = subscriptions_.find(subscription_id);
    if (it == subscriptions_.end()) {
        return false;
    }

    it->second->status = SubscriptionStatus::PAUSED;
    Logger::info("SubscriptionManager: Paused subscription {}", subscription_id);
    return true;
}

bool SubscriptionManager::resume_subscription(const std::string& subscription_id) {
    std::shared_lock<std::shared_mutex> lock(subscriptions_mutex_);

    auto it = subscriptions_.find(subscription_id);
    if (it == subscriptions_.end()) {
        return false;
    }

    it->second->status = SubscriptionStatus::ACTIVE;
    it->second->last_update = std::chrono::steady_clock::now();
    Logger::info("SubscriptionManager: Resumed subscription {}", subscription_id);
    return true;
}

bool SubscriptionManager::set_subscription_priority(const std::string& subscription_id, uint8_t priority) {
    if (priority >= config_.max_priority_levels) {
        Logger::error("SubscriptionManager: Priority {} exceeds maximum {}", priority, config_.max_priority_levels);
        return false;
    }

    std::shared_lock<std::shared_mutex> lock(subscriptions_mutex_);

    auto it = subscriptions_.find(subscription_id);
    if (it == subscriptions_.end()) {
        return false;
    }

    auto subscription = it->second;
    lock.unlock();

    // Remove from old priority queue
    update_priority_queue(subscription, false);

    // Update priority
    subscription->priority = priority;

    // Add to new priority queue
    update_priority_queue(subscription, true);

    Logger::info("SubscriptionManager: Set priority {} for subscription {}", priority, subscription_id);
    return true;
}

std::shared_ptr<Subscription> SubscriptionManager::get_subscription(const std::string& subscription_id) const {
    std::shared_lock<std::shared_mutex> lock(subscriptions_mutex_);
    auto it = subscriptions_.find(subscription_id);
    return (it != subscriptions_.end()) ? it->second : nullptr;
}

std::vector<std::shared_ptr<Subscription>> SubscriptionManager::get_subscriptions_for_connection(
    const std::string& connection_id) const {

    std::vector<std::shared_ptr<Subscription>> result;

    std::shared_lock<std::shared_mutex> lock(mappings_mutex_);
    auto it = connection_to_subscriptions_.find(connection_id);
    if (it != connection_to_subscriptions_.end()) {
        std::shared_lock<std::shared_mutex> sub_lock(subscriptions_mutex_);
        for (const auto& sub_id : it->second) {
            auto sub_it = subscriptions_.find(sub_id);
            if (sub_it != subscriptions_.end()) {
                result.push_back(sub_it->second);
            }
        }
    }

    return result;
}

std::vector<std::shared_ptr<Subscription>> SubscriptionManager::get_subscriptions_by_type(
    SubscriptionType type) const {

    std::vector<std::shared_ptr<Subscription>> result;

    std::shared_lock<std::shared_mutex> lock(mappings_mutex_);
    auto it = type_to_subscriptions_.find(type);
    if (it != type_to_subscriptions_.end()) {
        std::shared_lock<std::shared_mutex> sub_lock(subscriptions_mutex_);
        for (const auto& sub_id : it->second) {
            auto sub_it = subscriptions_.find(sub_id);
            if (sub_it != subscriptions_.end()) {
                result.push_back(sub_it->second);
            }
        }
    }

    return result;
}

std::vector<std::shared_ptr<Subscription>> SubscriptionManager::get_active_subscriptions() const {
    std::vector<std::shared_ptr<Subscription>> result;

    std::shared_lock<std::shared_mutex> lock(subscriptions_mutex_);
    for (const auto& [sub_id, subscription] : subscriptions_) {
        if (subscription->status == SubscriptionStatus::ACTIVE) {
            result.push_back(subscription);
        }
    }

    return result;
}

bool SubscriptionManager::should_send_message(const std::shared_ptr<Subscription>& subscription) const {
    if (!subscription || subscription->status != SubscriptionStatus::ACTIVE) {
        return false;
    }

    // Check rate limiting
    if (!check_rate_limit(subscription)) {
        subscription->messages_dropped.fetch_add(1);
        return false;
    }

    // Check update frequency
    auto current_time = std::chrono::steady_clock::now();
    auto time_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time - subscription->last_sent).count();

    double update_interval_ms = 1000.0 / subscription->parameters.update_frequency;
    return time_since_last >= update_interval_ms;
}

void SubscriptionManager::update_subscription_metrics(const std::string& subscription_id,
                                                     double latency_ms,
                                                     size_t message_size) {
    std::shared_lock<std::shared_mutex> lock(subscriptions_mutex_);

    auto it = subscriptions_.find(subscription_id);
    if (it == subscriptions_.end()) {
        return;
    }

    auto subscription = it->second;

    // Update subscription metrics
    subscription->messages_sent.fetch_add(1);
    subscription->bytes_sent.fetch_add(message_size);
    subscription->avg_latency_ms.store(
        (subscription->avg_latency_ms.load() + latency_ms) / 2.0);

    double current_max = subscription->max_latency_ms.load();
    while (latency_ms > current_max &&
           !subscription->max_latency_ms.compare_exchange_weak(current_max, latency_ms)) {
        // Retry if another thread updated the max value
    }

    subscription->last_sent = std::chrono::steady_clock::now();

    // Update connection metrics
    update_connection_metrics(subscription->connection_id, latency_ms, message_size);
}

std::shared_ptr<ConnectionSummary> SubscriptionManager::get_connection_summary(
    const std::string& connection_id) const {

    std::shared_lock<std::shared_mutex> lock(connections_mutex_);
    auto it = connections_.find(connection_id);
    return (it != connections_.end()) ? it->second : nullptr;
}

std::vector<std::shared_ptr<ConnectionSummary>> SubscriptionManager::get_all_connections() const {
    std::vector<std::shared_ptr<ConnectionSummary>> result;

    std::shared_lock<std::shared_mutex> lock(connections_mutex_);
    result.reserve(connections_.size());
    for (const auto& [id, connection] : connections_) {
        result.push_back(connection);
    }

    return result;
}

SubscriptionManager::Statistics SubscriptionManager::get_statistics() const {
    std::unique_lock<std::mutex> lock(stats_mutex_);
    return statistics_;
}

void SubscriptionManager::reset_statistics() {
    std::unique_lock<std::mutex> lock(stats_mutex_);
    statistics_ = Statistics{};
    statistics_.start_time = std::chrono::steady_clock::now();
}

void SubscriptionManager::cleanup_inactive_connections() {
    auto current_time = std::chrono::steady_clock::now();
    auto timeout_duration = std::chrono::seconds(static_cast<long>(config_.connection_timeout_seconds));

    std::vector<std::string> connections_to_remove;

    // Find inactive connections
    {
        std::shared_lock<std::shared_mutex> lock(connections_mutex_);
        for (const auto& [id, connection] : connections_) {
            if (current_time - connection->last_activity > timeout_duration) {
                connections_to_remove.push_back(id);
            }
        }
    }

    // Remove inactive connections
    for (const auto& id : connections_to_remove) {
        Logger::info("SubscriptionManager: Removing inactive connection {}", id);
        remove_connection(id);
    }
}

void SubscriptionManager::enforce_rate_limits() {
    std::shared_lock<std::shared_mutex> lock(subscriptions_mutex_);

    auto current_time = std::chrono::steady_clock::now();
    uint32_t max_messages = config_.rate_limit_messages_per_second;

    for (auto& [sub_id, subscription] : subscriptions_) {
        auto time_since_reset = std::chrono::duration_cast<std::chrono::seconds>(
            current_time - subscription->rate_limit_reset).count();

        // Reset window if more than 1 second has passed
        if (time_since_reset >= 1) {
            reset_rate_limit_window(subscription);
        }

        // Check if limit exceeded
        if (subscription->messages_in_window.load() >= max_messages) {
            subscription->status = SubscriptionStatus::PAUSED;
        } else if (subscription->status == SubscriptionStatus::PAUSED &&
                   subscription->messages_in_window.load() < max_messages * 0.8) {
            // Resume if dropped below 80% of limit
            subscription->status = SubscriptionStatus::ACTIVE;
        }
    }
}

void SubscriptionManager::update_qos_levels() {
    std::shared_lock<std::shared_mutex> lock(subscriptions_mutex_);

    for (auto& [sub_id, subscription] : subscriptions_) {
        apply_qos_settings(subscription);
    }
}

void SubscriptionManager::set_subscription_added_callback(SubscriptionEventCallback callback) {
    std::unique_lock<std::mutex> lock(callbacks_mutex_);
    subscription_added_callback_ = std::move(callback);
}

void SubscriptionManager::set_subscription_removed_callback(SubscriptionEventCallback callback) {
    std::unique_lock<std::mutex> lock(callbacks_mutex_);
    subscription_removed_callback_ = std::move(callback);
}

void SubscriptionManager::set_connection_added_callback(ConnectionEventCallback callback) {
    std::unique_lock<std::mutex> lock(callbacks_mutex_);
    connection_added_callback_ = std::move(callback);
}

void SubscriptionManager::set_connection_removed_callback(ConnectionEventCallback callback) {
    std::unique_lock<std::mutex> lock(callbacks_mutex_);
    connection_removed_callback_ = std::move(callback);
}

std::string SubscriptionManager::subscription_type_to_string(SubscriptionType type) const {
    switch (type) {
        case SubscriptionType::SPECTRUM: return "spectrum";
        case SubscriptionType::WAVEFORM: return "waveform";
        case SubscriptionType::VU_LEVELS: return "vu_levels";
        case SubscriptionType::PEAKS: return "peaks";
        case SubscriptionType::ZERO_CROSSINGS: return "zero_crossings";
        case SubscriptionType::ENVELOPE: return "envelope";
        case SubscriptionType::PHASE: return "phase";
        case SubscriptionType::STEREO_IMAGE: return "stereo_image";
        case SubscriptionType::CORRELATION: return "correlation";
        case SubscriptionType::HISTOGRAM: return "histogram";
        case SubscriptionType::WATERFALL: return "waterfall";
        case SubscriptionType::SPECTROGRAM: return "spectrogram";
        case SubscriptionType::METADATA: return "metadata";
        case SubscriptionType::PERFORMANCE_STATS: return "performance_stats";
        case SubscriptionType::ALL: return "all";
        default: return "unknown";
    }
}

std::string SubscriptionManager::qos_to_string(SubscriptionQoS qos) const {
    switch (qos) {
        case SubscriptionQoS::BEST_EFFORT: return "best_effort";
        case SubscriptionQoS::LOW_LATENCY: return "low_latency";
        case SubscriptionQoS::RELIABLE: return "reliable";
        case SubscriptionQoS::HIGH_QUALITY: return "high_quality";
        default: return "unknown";
    }
}

std::string SubscriptionManager::status_to_string(SubscriptionStatus status) const {
    switch (status) {
        case SubscriptionStatus::INACTIVE: return "inactive";
        case SubscriptionStatus::PENDING: return "pending";
        case SubscriptionStatus::ACTIVE: return "active";
        case SubscriptionStatus::PAUSED: return "paused";
        case SubscriptionStatus::ERROR: return "error";
        case SubscriptionStatus::TERMINATED: return "terminated";
        default: return "unknown";
    }
}

SubscriptionType SubscriptionManager::string_to_subscription_type(const std::string& type) const {
    std::string lower_type = type;
    std::transform(lower_type.begin(), lower_type.end(), lower_type.begin(), ::tolower);

    if (lower_type == "spectrum") return SubscriptionType::SPECTRUM;
    if (lower_type == "waveform") return SubscriptionType::WAVEFORM;
    if (lower_type == "vu_levels" || lower_type == "vu") return SubscriptionType::VU_LEVELS;
    if (lower_type == "peaks") return SubscriptionType::PEAKS;
    if (lower_type == "zero_crossings") return SubscriptionType::ZERO_CROSSINGS;
    if (lower_type == "envelope") return SubscriptionType::ENVELOPE;
    if (lower_type == "phase") return SubscriptionType::PHASE;
    if (lower_type == "stereo_image") return SubscriptionType::STEREO_IMAGE;
    if (lower_type == "correlation") return SubscriptionType::CORRELATION;
    if (lower_type == "histogram") return SubscriptionType::HISTOGRAM;
    if (lower_type == "waterfall") return SubscriptionType::WATERFALL;
    if (lower_type == "spectrogram") return SubscriptionType::SPECTROGRAM;
    if (lower_type == "metadata") return SubscriptionType::METADATA;
    if (lower_type == "performance_stats") return SubscriptionType::PERFORMANCE_STATS;
    if (lower_type == "all") return SubscriptionType::ALL;

    return SubscriptionType::UNKNOWN;
}

SubscriptionQoS SubscriptionManager::string_to_qos(const std::string& qos) const {
    std::string lower_qos = qos;
    std::transform(lower_qos.begin(), lower_qos.end(), lower_qos.begin(), ::tolower);

    if (lower_qos == "best_effort") return SubscriptionQoS::BEST_EFFORT;
    if (lower_qos == "low_latency") return SubscriptionQoS::LOW_LATENCY;
    if (lower_qos == "reliable") return SubscriptionQoS::RELIABLE;
    if (lower_qos == "high_quality") return SubscriptionQoS::HIGH_QUALITY;

    return SubscriptionQoS::BEST_EFFORT; // Default
}

std::string SubscriptionManager::get_diagnostics_report() const {
    std::ostringstream report;

    report << "=== SubscriptionManager Diagnostics Report ===\n";
    report << "Initialized: " << (initialized_.load() ? "Yes" : "No") << "\n";
    report << "Max Subscriptions: " << config_.max_subscriptions << "\n";
    report << "Max per Connection: " << config_.max_subscriptions_per_connection << "\n";
    report << "Default Update Frequency: " << config_.default_update_frequency << " Hz\n";
    report << "Cleanup Interval: " << config_.cleanup_interval_seconds << " seconds\n\n";

    auto stats = get_statistics();
    report << "Current Statistics:\n";
    report << "  Total Connections: " << stats.total_connections << "\n";
    report << "  Active Connections: " << stats.active_connections << "\n";
    report << "  Total Subscriptions: " << stats.total_subscriptions << "\n";
    report << "  Active Subscriptions: " << stats.active_subscriptions << "\n";
    report << "  Messages Sent: " << stats.total_messages_sent << "\n";
    report << "  Messages Dropped: " << stats.total_messages_dropped << "\n";
    report << "  Bytes Sent: " << stats.total_bytes_sent << "\n";
    report << "  Average Latency: " << std::fixed << std::setprecision(2) << stats.avg_latency_ms << " ms\n";
    report << "  Peak Latency: " << stats.peak_latency_ms << " ms\n";
    report << "  Total Bandwidth: " << std::setprecision(1) << stats.total_bandwidth_mbps << " Mbps\n";

    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - stats.start_time).count();
    report << "  Uptime: " << uptime << " seconds\n";

    return report.str();
}

bool SubscriptionManager::validate_subscription_parameters(const SubscriptionParameters& params) const {
    // Validate update frequency
    if (params.update_frequency < 0.1f || params.update_frequency > 1000.0f) {
        Logger::error("Invalid update frequency: {}", params.update_frequency);
        return false;
    }

    // Validate message size
    if (params.max_message_size == 0 || params.max_message_size > 10 * 1024 * 1024) {
        Logger::error("Invalid max message size: {}", params.max_message_size);
        return false;
    }

    // Validate buffer size
    if (params.buffer_size == 0 || params.buffer_size > 100000) {
        Logger::error("Invalid buffer size: {}", params.buffer_size);
        return false;
    }

    // Validate latency threshold
    if (params.max_latency_ms < 0.0f || params.max_latency_ms > 10000.0f) {
        Logger::error("Invalid max latency: {}", params.max_latency_ms);
        return false;
    }

    // Validate compression level
    if (params.compression_level > 9) {
        Logger::error("Invalid compression level: {}", params.compression_level);
        return false;
    }

    // Validate spectrum-specific parameters
    if (params.spectrum_fft_size == 0 || (params.spectrum_fft_size & (params.spectrum_fft_size - 1)) != 0) {
        Logger::error("Invalid spectrum FFT size: {}", params.spectrum_fft_size);
        return false;
    }

    if (params.spectrum_min_frequency < 0.0f || params.spectrum_min_frequency >= params.spectrum_max_frequency) {
        Logger::error("Invalid spectrum frequency range: {} to {}",
                     params.spectrum_min_frequency, params.spectrum_max_frequency);
        return false;
    }

    // Validate waveform-specific parameters
    if (params.waveform_length == 0 || params.waveform_length > 10000) {
        Logger::error("Invalid waveform length: {}", params.waveform_length);
        return false;
    }

    if (params.waveform_window_duration <= 0.0f || params.waveform_window_duration > 10.0f) {
        Logger::error("Invalid waveform window duration: {}", params.waveform_window_duration);
        return false;
    }

    // Validate VU-specific parameters
    if (params.vu_attack_time_ms < 0.0f || params.vu_attack_time_ms > 100.0f) {
        Logger::error("Invalid VU attack time: {}", params.vu_attack_time_ms);
        return false;
    }

    if (params.vu_release_time_ms < 0.0f || params.vu_release_time_ms > 10000.0f) {
        Logger::error("Invalid VU release time: {}", params.vu_release_time_ms);
        return false;
    }

    return true;
}

// Private helper methods

std::string SubscriptionManager::generate_subscription_id() const {
    static std::atomic<uint64_t> counter{0};
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    static thread_local std::uniform_int_distribution<uint32_t> dis(0, 0xFFFFFFFF);

    uint64_t timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    uint64_t count = counter.fetch_add(1);
    uint32_t random = dis(gen);

    std::ostringstream oss;
    oss << "sub_" << std::hex << timestamp << "_" << count << "_" << random;
    return oss.str();
}

void SubscriptionManager::update_type_mapping(SubscriptionType type,
                                             const std::string& subscription_id,
                                             bool add) {
    std::unique_lock<std::shared_mutex> lock(mappings_mutex_);

    auto& subscription_set = type_to_subscriptions_[type];

    if (add) {
        subscription_set.insert(subscription_id);
    } else {
        subscription_set.erase(subscription_id);
        if (subscription_set.empty()) {
            type_to_subscriptions_.erase(type);
        }
    }
}

void SubscriptionManager::update_connection_mapping(const std::string& connection_id,
                                                   const std::string& subscription_id,
                                                   bool add) {
    std::unique_lock<std::shared_mutex> lock(mappings_mutex_);

    auto& subscription_set = connection_to_subscriptions_[connection_id];

    if (add) {
        subscription_set.insert(subscription_id);
    } else {
        subscription_set.erase(subscription_id);
        if (subscription_set.empty()) {
            connection_to_subscriptions_.erase(connection_id);
        }
    }
}

void SubscriptionManager::update_priority_queue(const std::shared_ptr<Subscription>& subscription,
                                               bool add) {
    if (subscription->priority >= priority_queues_.size()) {
        Logger::warn("Subscription priority {} exceeds queue size", subscription->priority);
        return;
    }

    std::unique_lock<std::mutex> lock(priority_mutex_);
    auto& queue = priority_queues_[subscription->priority];

    if (add) {
        queue.push_back(subscription->subscription_id);
    } else {
        auto it = std::find(queue.begin(), queue.end(), subscription->subscription_id);
        if (it != queue.end()) {
            queue.erase(it);
        }
    }
}

void SubscriptionManager::update_statistics() {
    std::unique_lock<std::mutex> lock(stats_mutex_);

    statistics_.total_connections = connections_.size();

    // Count active connections and subscriptions
    auto current_time = std::chrono::steady_clock::now();
    auto timeout_duration = std::chrono::seconds(static_cast<long>(config_.connection_timeout_seconds));

    statistics_.active_connections = 0;
    statistics_.total_subscriptions = 0;
    statistics_.active_subscriptions = 0;

    for (const auto& [id, connection] : connections_) {
        if (current_time - connection->last_activity < timeout_duration) {
            statistics_.active_connections++;
            statistics_.total_subscriptions += connection->active_subscriptions.load();
        }
    }

    for (const auto& [id, subscription] : subscriptions_) {
        if (subscription->status == SubscriptionStatus::ACTIVE) {
            statistics_.active_subscriptions++;
        }
    }

    update_global_statistics();
}

void SubscriptionManager::cleanup_thread_func() {
    Logger::info("SubscriptionManager: Cleanup thread started");

    const auto cleanup_interval = std::chrono::seconds(static_cast<long>(config_.cleanup_interval_seconds));

    while (cleanup_thread_running_.load()) {
        std::this_thread::sleep_for(cleanup_interval);

        if (!cleanup_thread_running_.load()) {
            break;
        }

        try {
            cleanup_inactive_connections();
            enforce_rate_limits();
            update_qos_levels();
        } catch (const std::exception& e) {
            Logger::error("SubscriptionManager: Exception in cleanup thread: {}", e.what());
        }
    }

    Logger::info("SubscriptionManager: Cleanup thread stopped");
}

bool SubscriptionManager::is_connection_limit_reached(const std::string& connection_id) const {
    std::shared_lock<std::shared_mutex> lock(mappings_mutex_);
    auto it = connection_to_subscriptions_.find(connection_id);
    return it != connection_to_subscriptions_.end() &&
           it->second.size() >= config_.max_subscriptions_per_connection;
}

bool SubscriptionManager::validate_connection_id(const std::string& connection_id) const {
    // Connection ID should be non-empty and contain only valid characters
    if (connection_id.empty() || connection_id.length() > 256) {
        return false;
    }

    // Allow alphanumeric characters, underscore, hyphen, and colon
    static std::regex valid_id_pattern("^[a-zA-Z0-9_:-]+$");
    return std::regex_match(connection_id, valid_id_pattern);
}

bool SubscriptionManager::validate_subscription_id(const std::string& subscription_id) const {
    // Similar validation as connection ID
    if (subscription_id.empty() || subscription_id.length() > 256) {
        return false;
    }

    static std::regex valid_id_pattern("^[a-zA-Z0-9_:-]+$");
    return std::regex_match(subscription_id, valid_id_pattern);
}

bool SubscriptionManager::check_rate_limit(const std::shared_ptr<Subscription>& subscription) const {
    auto current_time = std::chrono::steady_clock::now();
    auto time_since_reset = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time - subscription->rate_limit_reset).count();

    // Reset window if more than 1 second has passed
    if (time_since_reset >= 1000) {
        return true; // Will be reset by calling function
    }

    return subscription->messages_in_window.load() < config_.rate_limit_messages_per_second;
}

void SubscriptionManager::reset_rate_limit_window(const std::shared_ptr<Subscription>& subscription) {
    subscription->rate_limit_reset = std::chrono::steady_clock::now();
    subscription->messages_in_window.store(0);
}

void SubscriptionManager::apply_qos_settings(std::shared_ptr<Subscription>& subscription) {
    // Apply QoS-specific adjustments to subscription parameters
    switch (subscription->parameters.qos) {
        case SubscriptionQoS::LOW_LATENCY:
            // Optimize for speed
            subscription->parameters.enable_compression = false;
            subscription->parameters.enable_checksum = false;
            subscription->parameters.buffer_size = std::min(subscription->parameters.buffer_size, 100u);
            subscription->parameters.max_latency_ms = 10.0f;
            break;

        case SubscriptionQoS::RELIABLE:
            // Optimize for reliability
            subscription->parameters.enable_compression = true;
            subscription->parameters.enable_checksum = true;
            subscription->parameters.buffer_size = std::max(subscription->parameters.buffer_size, 500u);
            subscription->parameters.max_latency_ms = 500.0f;
            break;

        case SubscriptionQoS::HIGH_QUALITY:
            // Optimize for quality
            subscription->parameters.enable_compression = false; // Keep full precision
            subscription->parameters.enable_checksum = true;
            subscription->parameters.buffer_size = 1000u;
            subscription->parameters.max_latency_ms = 1000.0f;
            break;

        case SubscriptionQoS::BEST_EFFORT:
        default:
            // Use defaults
            break;
    }

    // Update subscription status
    subscription->status = SubscriptionStatus::ACTIVE;
}

void SubscriptionManager::update_connection_metrics(const std::string& connection_id,
                                                   double latency_ms,
                                                   size_t message_size) {
    std::shared_lock<std::shared_mutex> lock(connections_mutex_);
    auto it = connections_.find(connection_id);
    if (it != connections_.end()) {
        auto connection = it->second;
        connection->total_messages_sent.fetch_add(1);
        connection->total_bytes_sent.fetch_add(message_size);
        connection->avg_latency_ms.store(
            (connection->avg_latency_ms.load() + latency_ms) / 2.0);
    }
}

void SubscriptionManager::update_global_statistics() {
    // Calculate global metrics
    uint64_t total_messages_sent = 0;
    uint64_t total_bytes_sent = 0;
    double total_latency = 0.0;
    uint64_t latency_count = 0;

    for (const auto& [id, connection] : connections_) {
        total_messages_sent += connection->total_messages_sent.load();
        total_bytes_sent += connection->total_bytes_sent.load();
        total_latency += connection->avg_latency_ms.load();
        latency_count++;
    }

    statistics_.total_messages_sent = total_messages_sent;
    statistics_.total_bytes_sent = total_bytes_sent;

    if (latency_count > 0) {
        statistics_.avg_latency_ms = total_latency / latency_count;
    }

    // Calculate bandwidth (bytes per second over the last second)
    auto current_time = std::chrono::steady_clock::now();
    static auto last_bandwidth_calculation = current_time;
    static uint64_t last_bytes_sent = 0;

    auto time_diff = std::chrono::duration_cast<std::chrono::seconds>(
        current_time - last_bandwidth_calculation).count();

    if (time_diff >= 1) {
        uint64_t bytes_diff = statistics_.total_bytes_sent - last_bytes_sent;
        double bandwidth_mbps = (bytes_diff * 8.0) / (time_diff * 1024.0 * 1024.0);
        statistics_.total_bandwidth_mbps = bandwidth_mbps;

        last_bandwidth_calculation = current_time;
        last_bytes_sent = statistics_.total_bytes_sent;
    }
}

// Factory implementations
std::unique_ptr<SubscriptionManager> SubscriptionManagerFactory::create_default() {
    auto manager = std::make_unique<SubscriptionManager>();
    SubscriptionManager::Config config;
    manager->initialize(config);
    return manager;
}

std::unique_ptr<SubscriptionManager> SubscriptionManagerFactory::create_high_performance() {
    auto manager = std::make_unique<SubscriptionManager>();
    SubscriptionManager::Config config;
    config.default_qos = SubscriptionQoS::LOW_LATENCY;
    config.enable_compression = false;
    config.max_message_queue_size = 500;
    manager->initialize(config);
    return manager;
}

std::unique_ptr<SubscriptionManager> SubscriptionManagerFactory::create_reliable() {
    auto manager = std::make_unique<SubscriptionManager>();
    SubscriptionManager::Config config;
    config.default_qos = SubscriptionQoS::RELIABLE;
    config.enable_compression = true;
    config.max_message_queue_size = 2000;
    manager->initialize(config);
    return manager;
}

std::unique_ptr<SubscriptionManager> SubscriptionManagerFactory::create_low_latency() {
    auto manager = std::make_unique<SubscriptionManager>();
    SubscriptionManager::Config config;
    config.default_qos = SubscriptionQoS::LOW_LATENCY;
    config.enable_compression = false;
    config.max_latency_ms = 10.0f;
    config.max_message_queue_size = 100;
    manager->initialize(config);
    return manager;
}

} // namespace vortex::network