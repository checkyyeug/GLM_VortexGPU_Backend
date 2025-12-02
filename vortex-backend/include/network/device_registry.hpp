#pragma once

#include "network/audio_discovery.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <condition_variable>
#include <fstream>
#include <sstream>

namespace VortexGPU {
namespace Network {

// Device registration status
enum class DeviceRegistrationStatus {
    REGISTERED,
    UNREGISTERED,
    PENDING_REGISTRATION,
    REGISTRATION_FAILED,
    REGISTRATION_EXPIRED,
    TEMPORARY_UNAVAILABLE
};

// Device class types
enum class DeviceClass {
    INPUT_DEVICE,
    OUTPUT_DEVICE,
    INPUT_OUTPUT_DEVICE,
    NETWORK_INTERFACE,
    MIXER,
    PROCESSOR,
    CONVERTER,
    BRIDGE,
    CONTROLLER,
    MONITOR,
    RECORDER,
    PLAYER,
    UNKNOWN
};

// Device priority levels
enum class DevicePriority {
    CRITICAL,
    HIGH,
    NORMAL,
    LOW,
    BACKGROUND
};

// Device health status
enum class DeviceHealth {
    HEALTHY,
    WARNING,
    DEGRADED,
    CRITICAL,
    OFFLINE,
    UNKNOWN
};

// Device configuration template
struct DeviceConfigurationTemplate {
    std::string template_id;
    std::string template_name;
    std::string template_version;
    std::string device_class;
    std::string manufacturer_pattern;
    std::string model_pattern;

    // Default configuration values
    uint32_t default_sample_rate = 48000;
    uint32_t default_buffer_size = 512;
    uint32_t default_channels = 2;
    double default_latency_ms = 10.0;

    // Capabilities bitmask
    uint64_t required_capabilities = 0;
    uint64_t optional_capabilities = 0;

    // Connection preferences
    std::vector<ConnectionType> preferred_connections;
    std::vector<DiscoveryProtocol> preferred_protocols;

    // Quality requirements
    double max_acceptable_latency_ms = 100.0;
    uint32_t min_required_throughput_mbps = 100;
    double max_acceptable_packet_loss = 1.0;

    // Auto-configuration settings
    bool auto_configure = true;
    bool auto_connect = false;
    bool auto_monitor = true;
    bool persist_configuration = true;

    // Custom properties
    std::map<std::string, std::string> custom_properties;

    // Template metadata
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point last_updated;
    std::string created_by;
    std::string description;
    std::vector<std::string> tags;
};

// Extended device information with registration data
struct RegistryDeviceInfo : public AudioDeviceInfo {
    // Registration information
    DeviceRegistrationStatus registration_status = DeviceRegistrationStatus::UNREGISTERED;
    std::chrono::system_clock::time_point registration_time;
    std::chrono::system_clock::time_point last_registration_attempt;
    std::chrono::seconds registration_ttl = std::chrono::hours(24);
    uint32_t registration_attempts = 0;
    std::string registration_token;
    std::string registration_source;

    // Device classification
    DeviceClass device_class = DeviceClass::UNKNOWN;
    DevicePriority priority = DevicePriority::NORMAL;
    DeviceHealth health_status = DeviceHealth::UNKNOWN;

    // Configuration management
    std::string configuration_id;
    std::string configuration_version;
    std::string configuration_source;
    std::map<std::string, std::string> configuration_properties;
    std::chrono::system_clock::time_point last_configuration_update;
    bool configuration_dirty = false;

    // Performance metrics
    double average_response_time_ms = 0.0;
    uint64_t total_requests = 0;
    uint64_t successful_requests = 0;
    uint64_t failed_requests = 0;
    std::chrono::system_clock::time_point last_successful_request;
    std::chrono::system_clock::time_point last_failed_request;

    // Health monitoring
    double cpu_utilization_percent = 0.0;
    double memory_utilization_percent = 0.0;
    double temperature_celsius = 0.0;
    double power_consumption_watts = 0.0;
    uint64_t uptime_seconds = 0;
    uint32_t error_count = 0;
    uint32_t warning_count = 0;
    std::vector<std::string> recent_errors;
    std::chrono::system_clock::time_point last_error_time;

    // Backup and redundancy
    std::vector<std::string> backup_devices;
    std::string primary_device_for;
    std::vector<std::string> failover_devices;
    bool is_backup_device = false;
    bool is_primary_device = false;

    // Grouping and organization
    std::vector<std::string> groups;
    std::vector<std::string> tags;
    std::string location;
    std::string owner;
    std::string department;
    std::string project;

    // Security and access
    std::vector<std::string> allowed_users;
    std::vector<std::string> allowed_groups;
    std::string access_level;
    bool requires_authentication = false;
    std::string authentication_method;

    // Lifecycle management
    bool auto_remove = true;
    std::chrono::seconds removal_timeout = std::chrono::hours(72);
    bool persistent = false;
    std::string persistence_file;

    // Metadata
    std::map<std::string, std::string> metadata;
    std::chrono::system_clock::time_point metadata_last_updated;

    // Comparison operators for sorting and searching
    bool operator<(const RegistryDeviceInfo& other) const {
        if (priority != other.priority) {
            return priority < other.priority;
        }
        return device_name < other.device_name;
    }

    // Health assessment methods
    bool isHealthy() const {
        return health_status == DeviceHealth::HEALTHY;
    }

    bool isAvailable() const {
        return status == DeviceStatus::ONLINE && isHealthy();
    }

    double getSuccessRate() const {
        return total_requests > 0 ?
            static_cast<double>(successful_requests) / total_requests * 100.0 : 100.0;
    }
};

// Registry statistics
struct RegistryStatistics {
    uint32_t total_devices = 0;
    uint32_t registered_devices = 0;
    uint32_t online_devices = 0;
    uint32_t offline_devices = 0;
    uint32_t healthy_devices = 0;
    uint32_t unhealthy_devices = 0;

    uint32_t input_devices = 0;
    uint32_t output_devices = 0;
    uint32_t input_output_devices = 0;

    uint32_t high_priority_devices = 0;
    uint32_t critical_priority_devices = 0;

    double average_response_time_ms = 0.0;
    double average_cpu_utilization = 0.0;
    double average_memory_utilization = 0.0;

    uint64_t total_requests = 0;
    uint64_t successful_requests = 0;
    uint64_t failed_requests = 0;

    std::chrono::system_clock::time_point last_update;
    std::chrono::seconds uptime;
};

// Registry configuration
struct RegistryConfig {
    // Storage configuration
    std::string storage_directory = "./device_registry";
    std::string database_file = "device_registry.db";
    std::string backup_directory = "./backups";
    bool enable_persistence = true;
    bool enable_backups = true;
    std::chrono::seconds backup_interval = std::chrono::hours(1);
    uint32_t max_backup_files = 24;

    // Registration configuration
    std::chrono::seconds default_registration_ttl = std::chrono::hours(24);
    std::chrono::seconds registration_retry_interval = std::chrono::minutes(5);
    uint32_t max_registration_attempts = 3;
    bool auto_register_discovered_devices = true;
    bool require_approval_for_new_devices = false;

    // Health monitoring configuration
    std::chrono::seconds health_check_interval = std::chrono::seconds(30);
    std::chrono::seconds device_timeout = std::chrono::minutes(5);
    std::chrono::seconds metrics_retention_period = std::chrono::days(7);
    bool enable_performance_monitoring = true;
    bool enable_health_monitoring = true;
    bool enable_predictive_maintenance = true;

    // Cleanup configuration
    std::chrono::seconds cleanup_interval = std::chrono::hours(1);
    std::chrono::seconds offline_device_removal_delay = std::chrono::hours(24);
    std::chrono::seconds stale_data_removal_delay = std::chrono::days(30);
    bool enable_automatic_cleanup = true;
    bool preserve_persistent_devices = true;

    // Security configuration
    bool enable_authentication = false;
    bool enable_authorization = false;
    std::string authentication_method = "token";
    std::string default_access_level = "read";
    bool log_access_attempts = true;

    // Performance configuration
    uint32_t max_concurrent_operations = 100;
    std::chrono::seconds operation_timeout = std::chrono::seconds(30);
    bool enable_caching = true;
    size_t cache_size_limit = 10000;
    std::chrono::seconds cache_ttl = std::chrono::minutes(5);

    // Event configuration
    bool enable_event_logging = true;
    bool enable_event_notifications = false;
    std::vector<std::string> event_subscribers;
    std::string event_log_file = "registry_events.log";
    size_t max_event_log_size = 100 * 1024 * 1024; // 100MB
    uint32_t max_event_log_files = 10;

    // Integration configuration
    bool integrate_with_discovery = true;
    bool integrate_with_monitoring = true;
    bool integrate_with_configuration = true;
    std::vector<std::string> external_integrations;
};

// Registry event types
enum class RegistryEventType {
    DEVICE_REGISTERED,
    DEVICE_UNREGISTERED,
    DEVICE_UPDATED,
    DEVICE_HEALTH_CHANGED,
    DEVICE_CONFIG_CHANGED,
    DEVICE_PRIORITY_CHANGED,
    DEVICE_GROUP_CHANGED,
    DEVICE_CONNECTED,
    DEVICE_DISCONNECTED,
    CONFIGURATION_TEMPLATE_CREATED,
    CONFIGURATION_TEMPLATE_UPDATED,
    CONFIGURATION_TEMPLATE_DELETED,
    REGISTRY_BACKUP_CREATED,
    REGISTRY_RESTORED,
    REGISTRY_CLEANUP_PERFORMED,
    ERROR_OCCURRED,
    WARNING_ISSUED
};

// Registry event
struct RegistryEvent {
    RegistryEventType type;
    std::string device_id;
    std::string device_name;
    std::string message;
    std::string details;
    std::chrono::system_clock::time_point timestamp;
    std::string source;
    std::string severity;

    // Event metadata
    std::map<std::string, std::string> metadata;
    std::vector<std::string> tags;
};

// Device registry query filters
struct DeviceQuery {
    std::vector<DeviceClass> device_classes;
    std::vector<DevicePriority> priorities;
    std::vector<DeviceHealth> health_statuses;
    std::vector<DeviceRegistrationStatus> registration_statuses;
    std::vector<std::string> manufacturers;
    std::vector<std::string> models;
    std::vector<ConnectionType> connection_types;
    std::vector<DeviceCapability> capabilities;
    std::vector<std::string> groups;
    std::vector<std::string> tags;
    std::vector<std::string> locations;

    // Numeric filters
    std::pair<double, double> latency_range = {0.0, 1000.0};
    std::pair<uint32_t, uint32_t> channel_range = {0, 1024};
    std::pair<double, double> cpu_utilization_range = {0.0, 100.0};
    std::pair<double, double> memory_utilization_range = {0.0, 100.0};
    std::pair<double, double> success_rate_range = {0.0, 100.0};

    // Text search
    std::string search_query;
    bool case_sensitive = false;
    bool exact_match = false;

    // Status filters
    bool online_only = false;
    bool healthy_only = false;
    bool registered_only = false;
    bool available_only = false;

    // Time-based filters
    std::optional<std::chrono::system_clock::time_point> registered_after;
    std::optional<std::chrono::system_clock::time_point> registered_before;
    std::optional<std::chrono::system_clock::time_point> last_seen_after;
    std::optional<std::chrono::system_clock::time_point> last_seen_before;

    // Sorting and pagination
    std::string sort_by = "device_name";
    bool sort_ascending = true;
    size_t offset = 0;
    size_t limit = 100;
};

// Forward declarations
class DeviceRegistry;
class RegistryStorage;
class RegistryCache;
class RegistryHealthMonitor;

// Main network audio device registry
class NetworkAudioDeviceRegistry {
public:
    NetworkAudioDeviceRegistry();
    ~NetworkAudioDeviceRegistry();

    // Initialization and lifecycle
    bool initialize(const RegistryConfig& config = {});
    void shutdown();
    bool isInitialized() const { return initialized_; }
    void reload();
    void save();

    // Device registration management
    bool registerDevice(const RegistryDeviceInfo& device);
    bool unregisterDevice(const std::string& device_id);
    bool updateDevice(const std::string& device_id, const RegistryDeviceInfo& updated_info);
    RegistryDeviceInfo getDevice(const std::string& device_id) const;
    bool isDeviceRegistered(const std::string& device_id) const;

    // Bulk operations
    std::vector<std::string> registerDevices(const std::vector<RegistryDeviceInfo>& devices);
    bool unregisterDevices(const std::vector<std::string>& device_ids);
    std::vector<RegistryDeviceInfo> updateDevices(const std::vector<RegistryDeviceInfo>& devices);

    // Device queries and searching
    std::vector<RegistryDeviceInfo> queryDevices(const DeviceQuery& query) const;
    std::vector<RegistryDeviceInfo> getAllDevices() const;
    std::vector<RegistryDeviceInfo> getDevicesByClass(DeviceClass device_class) const;
    std::vector<RegistryDeviceInfo> getDevicesByPriority(DevicePriority priority) const;
    std::vector<RegistryDeviceInfo> getDevicesByHealth(DeviceHealth health) const;
    std::vector<RegistryDeviceInfo> getAvailableDevices() const;
    std::vector<RegistryDeviceInfo> getHealthyDevices() const;
    std::vector<RegistryDeviceInfo> getDevicesByGroup(const std::string& group) const;
    std::vector<RegistryDeviceInfo> searchDevices(const std::string& query) const;

    // Device grouping and tagging
    bool addDeviceToGroup(const std::string& device_id, const std::string& group);
    bool removeDeviceFromGroup(const std::string& device_id, const std::string& group);
    bool addDeviceTag(const std::string& device_id, const std::string& tag);
    bool removeDeviceTag(const std::string& device_id, const std::string& tag);
    std::vector<std::string> getDeviceGroups(const std::string& device_id) const;
    std::vector<std::string> getDeviceTags(const std::string& device_id) const;
    std::vector<std::string> getAllGroups() const;
    std::vector<std::string> getAllTags() const;

    // Device priority management
    bool setDevicePriority(const std::string& device_id, DevicePriority priority);
    DevicePriority getDevicePriority(const std::string& device_id) const;
    std::vector<RegistryDeviceInfo> getDevicesByPriorityOrder() const;

    // Health monitoring
    void updateDeviceHealth(const std::string& device_id, DeviceHealth health, const std::string& message = "");
    void recordDeviceError(const std::string& device_id, const std::string& error);
    void updateDeviceMetrics(const std::string& device_id, double cpu_util, double memory_util,
                            double response_time, bool request_successful = true);
    std::vector<RegistryDeviceInfo> getUnhealthyDevices() const;
    std::vector<RegistryDeviceInfo> getDevicesNeedingAttention() const;

    // Configuration templates
    bool createConfigurationTemplate(const DeviceConfigurationTemplate& template_info);
    bool updateConfigurationTemplate(const std::string& template_id, const DeviceConfigurationTemplate& template_info);
    bool deleteConfigurationTemplate(const std::string& template_id);
    std::vector<DeviceConfigurationTemplate> getConfigurationTemplates() const;
    DeviceConfigurationTemplate getConfigurationTemplate(const std::string& template_id) const;
    bool applyConfigurationTemplate(const std::string& device_id, const std::string& template_id);
    std::string getBestMatchingTemplate(const std::string& device_id) const;

    // Backup and restore
    bool createBackup(const std::string& backup_name = "");
    bool restoreFromBackup(const std::string& backup_name);
    std::vector<std::string> getAvailableBackups() const;
    bool deleteBackup(const std::string& backup_name);
    bool scheduleBackup(const std::chrono::seconds& interval);
    void stopScheduledBackups();

    // Statistics and reporting
    RegistryStatistics getStatistics() const;
    std::vector<RegistryEvent> getRecentEvents(size_t max_events = 100) const;
    std::string generateHealthReport() const;
    std::string generateDeviceReport(const std::string& device_id) const;
    std::string generateRegistryReport() const;
    void exportToJSON(const std::string& filename) const;
    void exportToCSV(const std::string& filename) const;

    // Event handling
    void setEventCallback(std::function<void(const RegistryEvent&)> callback);
    void publishEvent(const RegistryEvent& event);
    std::vector<std::string> subscribeToEvents(const std::vector<RegistryEventType>& event_types);
    void unsubscribeFromEvents(const std::vector<std::string>& subscription_ids);

    // Maintenance and cleanup
    void performMaintenance();
    void cleanupStaleDevices();
    void cleanupOldEvents();
    void rebuildIndexes();
    void validateRegistry();
    void optimizeRegistry();

    // Configuration
    void updateConfig(const RegistryConfig& config);
    RegistryConfig getConfig() const { return config_; }

    // Integration with discovery service
    void setDiscoveryService(std::shared_ptr<AudioNetworkDiscovery> discovery_service);
    void syncWithDiscoveryService();
    bool autoRegisterDiscoveredDevices();

private:
    // Core components
    std::unique_ptr<RegistryStorage> storage_;
    std::unique_ptr<RegistryCache> cache_;
    std::unique_ptr<RegistryHealthMonitor> health_monitor_;

    // Configuration and state
    RegistryConfig config_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> running_{false};

    // Device registry storage
    mutable std::mutex registry_mutex_;
    std::unordered_map<std::string, RegistryDeviceInfo> device_registry_;
    std::unordered_map<std::string, DeviceConfigurationTemplate> configuration_templates_;

    // Indexes for efficient querying
    std::unordered_map<DeviceClass, std::unordered_set<std::string>> class_index_;
    std::unordered_map<DevicePriority, std::unordered_set<std::string>> priority_index_;
    std::unordered_map<DeviceHealth, std::unordered_set<std::string>> health_index_;
    std::unordered_map<std::string, std::unordered_set<std::string>> group_index_;
    std::unordered_map<std::string, std::unordered_set<std::string>> tag_index_;
    std::unordered_map<std::string, std::unordered_set<std::string>> manufacturer_index_;

    // Event system
    mutable std::mutex events_mutex_;
    std::queue<RegistryEvent> event_queue_;
    std::vector<RegistryEvent> event_history_;
    std::function<void(const RegistryEvent&)> event_callback_;
    std::unordered_map<std::string, std::vector<RegistryEventType>> event_subscriptions_;
    std::unordered_map<std::string, std::string> subscription_ids_;

    // Background threads
    std::thread maintenance_thread_;
    std::thread health_monitor_thread_;
    std::thread backup_thread_;
    std::thread event_processing_thread_;
    std::atomic<bool> scheduled_backup_enabled_{false};
    std::chrono::seconds backup_interval_{std::chrono::hours(1)};

    // Statistics
    mutable std::mutex stats_mutex_;
    RegistryStatistics statistics_;
    std::chrono::system_clock::time_point registry_start_time_;

    // Integration services
    std::shared_ptr<AudioNetworkDiscovery> discovery_service_;

    // Internal methods
    void maintenanceThread();
    void healthMonitorThread();
    void backupThread();
    void eventProcessingThread();

    bool loadFromStorage();
    bool saveToStorage();
    bool createBackupFile(const std::string& backup_path);
    bool restoreFromBackupFile(const std::string& backup_path);

    void addToIndexes(const RegistryDeviceInfo& device);
    void removeFromIndexes(const RegistryDeviceInfo& device);
    void updateIndexes(const RegistryDeviceInfo& old_device, const RegistryDeviceInfo& new_device);

    std::string generateDeviceId(const RegistryDeviceInfo& device) const;
    std::string generateRegistrationToken() const;
    bool validateDevice(const RegistryDeviceInfo& device) const;
    bool validateConfigurationTemplate(const DeviceConfigurationTemplate& template_info) const;

    void updateStatistics();
    void recordEvent(const RegistryEvent& event);
    void notifyEventSubscribers(const RegistryEvent& event);

    bool matchesQuery(const RegistryDeviceInfo& device, const DeviceQuery& query) const;
    std::vector<RegistryDeviceInfo> sortDevices(const std::vector<RegistryDeviceInfo>& devices,
                                                 const std::string& sort_by, bool ascending) const;

    void checkDeviceRegistrations();
    void refreshDeviceInformation();
    void performDeviceHealthChecks();

    // Utility methods
    std::string deviceClassToString(DeviceClass device_class) const;
    std::string priorityToString(DevicePriority priority) const;
    std::string healthToString(DeviceHealth health) const;
    DeviceClass stringToDeviceClass(const std::string& str) const;
    DevicePriority stringToPriority(const std::string& str) const;
    DeviceHealth stringToHealth(const std::string& str) const;
};

// Registry storage interface
class RegistryStorage {
public:
    virtual ~RegistryStorage() = default;
    virtual bool initialize(const RegistryConfig& config) = 0;
    virtual void shutdown() = 0;
    virtual bool saveDevice(const RegistryDeviceInfo& device) = 0;
    virtual bool loadDevice(const std::string& device_id, RegistryDeviceInfo& device) = 0;
    virtual bool deleteDevice(const std::string& device_id) = 0;
    virtual bool saveTemplate(const DeviceConfigurationTemplate& template_info) = 0;
    virtual bool loadTemplate(const std::string& template_id, DeviceConfigurationTemplate& template_info) = 0;
    virtual bool deleteTemplate(const std::string& template_id) = 0;
    virtual std::vector<RegistryDeviceInfo> loadAllDevices() = 0;
    virtual std::vector<DeviceConfigurationTemplate> loadAllTemplates() = 0;
    virtual bool createBackup(const std::string& backup_path) = 0;
    virtual bool restoreFromBackup(const std::string& backup_path) = 0;
    virtual bool cleanup() = 0;
};

// Registry cache implementation
class RegistryCache {
public:
    RegistryCache(size_t max_size = 10000, std::chrono::seconds ttl = std::chrono::minutes(5));
    ~RegistryCache();

    bool get(const std::string& key, RegistryDeviceInfo& device) const;
    void put(const std::string& key, const RegistryDeviceInfo& device);
    void invalidate(const std::string& key);
    void clear();
    size_t size() const;
    void setMaxSize(size_t max_size);
    void setTTL(std::chrono::seconds ttl);

private:
    mutable std::mutex cache_mutex_;
    size_t max_size_;
    std::chrono::seconds ttl_;
    std::unordered_map<std::string, std::pair<RegistryDeviceInfo, std::chrono::steady_clock::time_point>> cache_;
    std::list<std::string> lru_list_;
    std::unordered_map<std::string, std::list<std::string>::iterator> lru_map_;

    void evictOldest();
    void updateLRU(const std::string& key);
    bool isExpired(const std::chrono::steady_clock::time_point& timestamp) const;
};

// Health monitor implementation
class RegistryHealthMonitor {
public:
    RegistryHealthMonitor(const RegistryConfig& config);
    ~RegistryHealthMonitor();

    void start();
    void stop();
    void addDevice(const std::string& device_id);
    void removeDevice(const std::string& device_id);
    void updateDeviceMetrics(const std::string& device_id, double cpu_util, double memory_util,
                            double response_time, bool request_successful);
    void recordDeviceError(const std::string& device_id, const std::string& error);
    DeviceHealth getDeviceHealth(const std::string& device_id) const;
    std::vector<std::string> getUnhealthyDevices() const;

    void setHealthCallback(std::function<void(const std::string&, DeviceHealth)> callback);

private:
    struct DeviceHealthData {
        std::deque<double> cpu_utilization_history;
        std::deque<double> memory_utilization_history;
        std::deque<double> response_time_history;
        std::deque<bool> request_success_history;
        std::vector<std::string> recent_errors;
        std::chrono::steady_clock::time_point last_error_time;
        uint32_t consecutive_failures = 0;
        DeviceHealth current_health = DeviceHealth::UNKNOWN;
        std::chrono::steady_clock::time_point last_health_update;
    };

    std::atomic<bool> running_{false};
    RegistryConfig config_;
    std::thread monitor_thread_;
    std::unordered_map<std::string, DeviceHealthData> health_data_;
    mutable std::shared_mutex health_data_mutex_;
    std::function<void(const std::string&, DeviceHealth)> health_callback_;

    void monitorThread();
    void updateDeviceHealth(const std::string& device_id);
    DeviceHealth calculateHealth(const DeviceHealthData& data) const;
    void cleanupOldData();
};

// Utility functions
namespace RegistryUtils {
    std::string deviceClassToString(DeviceClass device_class);
    std::string priorityToString(DevicePriority priority);
    std::string healthToString(DeviceHealth health);
    std::string registrationStatusToString(DeviceRegistrationStatus status);
    std::string eventTypeToString(RegistryEventType type);

    DeviceClass stringToDeviceClass(const std::string& str);
    DevicePriority stringToPriority(const std::string& str);
    DeviceHealth stringToHealth(const std::string& str);
    DeviceRegistrationStatus stringToRegistrationStatus(const std::string& str);
    RegistryEventType stringToEventType(const std::string& str);

    std::string generateDeviceId(const RegistryDeviceInfo& device);
    std::string generateRegistrationToken();
    std::string generateUUID();

    bool isValidDeviceId(const std::string& device_id);
    bool isValidDeviceName(const std::string& device_name);
    bool isValidIpAddress(const std::string& ip_address);
    bool isValidMacAddress(const std::string& mac_address);

    std::string escapeJsonString(const std::string& str);
    std::string formatTimestamp(const std::chrono::system_clock::time_point& timestamp);
    std::string formatDuration(const std::chrono::seconds& duration);

    double calculateSuccessRate(uint64_t successful, uint64_t total);
    double calculateAverage(const std::deque<double>& values);
    std::string calculateHash(const std::string& data);
}

} // namespace Network
} // namespace VortexGPU