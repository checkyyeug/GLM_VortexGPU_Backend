#pragma once

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
#include <queue>
#include <condition_variable>

namespace VortexGPU {
namespace Network {

// Network discovery protocols
enum class DiscoveryProtocol {
    MDNS,           // Multicast DNS (Bonjour/Avahi)
    UPnP,           // Universal Plug and Play
    WS_DISCOVERY,   // Web Services Discovery
    DHT,            // Distributed Hash Table
    CUSTOM          // Custom protocol
};

// Device capabilities
enum class DeviceCapability {
    AUDIO_INPUT,
    AUDIO_OUTPUT,
    MULTICHANNEL,
    LOW_LATENCY,
    NETWORK_STREAMING,
    HARDWARE_ACCELERATION,
    VR_AUDIO,
    SPATIAL_AUDIO,
    PROFESSIONAL_AUDIO,
    MIDI_SUPPORT,
    CONTROL_SURFACE,
    EFFECTS_PROCESSING,
    MIXING_CONSOLE,
    RECORDING,
    PLAYBACK,
    DUPLEX,
    CLOCK_SYNC,
    WORD_CLOCK,
    AES3,
    ADAT,
    MADI,
    DANTE,
    RAVENNA,
    AES67,
    NDI,
    SRT
};

// Device connection type
enum class ConnectionType {
    WIRED_ETHERNET,
    WIRELESS_WIFI,
    WIRELESS_BLUETOOTH,
    USB,
    THUNDERBOLT,
    FIREWIRE,
    PCI,
    PCIe,
    INTERNAL,
    OPTICAL,
    COAXIAL
};

// Device status
enum class DeviceStatus {
    ONLINE,
    OFFLINE,
    BUSY,
    ERROR,
    MAINTENANCE,
    UNKNOWN
};

// Device information
struct AudioDeviceInfo {
    std::string device_id;
    std::string device_name;
    std::string manufacturer;
    std::string model;
    std::string firmware_version;
    std::string hardware_version;
    std::string serial_number;
    std::string mac_address;
    std::vector<std::string> ip_addresses;
    uint16_t control_port = 0;
    uint16_t streaming_port = 0;
    std::vector<uint16_t> additional_ports;

    // Audio capabilities
    std::vector<DeviceCapability> capabilities;
    uint32_t max_channels = 0;
    uint32_t supported_sample_rates = 0;  // Bitmask of supported rates
    uint32_t supported_bit_depths = 0;    // Bitmask of supported depths
    uint32_t max_buffer_size = 0;
    uint32_t min_buffer_size = 0;
    uint32_t preferred_buffer_size = 0;

    // Performance specifications
    double min_latency_ms = 0.0;
    double max_latency_ms = 0.0;
    double typical_latency_ms = 0.0;
    uint32_t max_throughput_mbps = 0;
    double cpu_utilization_percent = 0.0;
    double memory_utilization_percent = 0.0;

    // Network capabilities
    std::vector<DiscoveryProtocol> supported_protocols;
    std::vector<ConnectionType> connection_types;
    bool supports_ipv4 = true;
    bool supports_ipv6 = false;
    bool supports_multicast = false;
    bool supports_encryption = false;

    // Status information
    DeviceStatus status = DeviceStatus::UNKNOWN;
    std::chrono::steady_clock::time_point last_seen;
    std::chrono::steady_clock::time_point first_seen;
    uint32_t connection_count = 0;
    std::string status_message;

    // Quality metrics
    double signal_strength_dbm = -100.0;
    double packet_loss_percent = 0.0;
    double jitter_ms = 0.0;
    uint64_t bytes_received = 0;
    uint64_t bytes_sent = 0;

    // Custom properties
    std::map<std::string, std::string> custom_properties;
    std::vector<std::string> supported_formats;
    std::vector<std::string> supported_codecs;

    // Service information
    std::vector<std::string> services;
    std::string service_type;
    std::string service_domain;
    int32_t service_priority = 0;
    std::string location;

    bool operator==(const AudioDeviceInfo& other) const {
        return device_id == other.device_id;
    }

    bool operator<(const AudioDeviceInfo& other) const {
        return device_name < other.device_name;
    }
};

// Network statistics
struct NetworkStatistics {
    uint64_t total_packets_sent = 0;
    uint64_t total_packets_received = 0;
    uint64_t total_bytes_sent = 0;
    uint64_t total_bytes_received = 0;
    uint32_t active_connections = 0;
    uint32_t failed_connections = 0;
    double average_latency_ms = 0.0;
    double packet_loss_percent = 0.0;
    std::chrono::steady_clock::time_point last_update;
};

// Discovery event types
enum class DiscoveryEventType {
    DEVICE_DISCOVERED,
    DEVICE_LOST,
    DEVICE_UPDATED,
    DEVICE_CONNECTED,
    DEVICE_DISCONNECTED,
    SERVICE_DISCOVERED,
    SERVICE_LOST,
    NETWORK_ERROR
};

// Discovery event
struct DiscoveryEvent {
    DiscoveryEventType type;
    std::string device_id;
    AudioDeviceInfo device_info;
    std::string message;
    std::chrono::steady_clock::time_point timestamp;
};

// Discovery configuration
struct DiscoveryConfig {
    bool enable_mdns = true;
    bool enable_upnp = true;
    bool enable_ws_discovery = false;
    bool enable_dht = false;
    bool enable_ipv6 = false;
    bool passive_discovery = false;

    std::chrono::seconds scan_interval = std::chrono::seconds(30);
    std::chrono::seconds device_timeout = std::chrono::seconds(300);
    std::chrono::seconds connection_timeout = std::chrono::seconds(10);
    uint32_t max_concurrent_scans = 4;
    uint32_t discovery_port = 0;  // 0 = auto-assign

    std::vector<std::string> service_types;
    std::vector<std::string> preferred_manufacturers;
    std::vector<DeviceCapability> required_capabilities;
    std::vector<std::string> excluded_devices;
    std::vector<std::string> network_interfaces;

    std::string custom_discovery_domain = "local.";
    std::string device_type_filter = "audio";

    bool detailed_scanning = true;
    bool continuous_monitoring = true;
    bool auto_connect = false;
    bool track_performance = true;

    // Network preferences
    bool prefer_wired = true;
    bool prefer_low_latency = true;
    double max_acceptable_latency_ms = 100.0;
    uint32_t min_required_channels = 2;
    uint32_t max_acceptable_packet_loss = 5;
};

// Forward declarations
class MDNSDiscovery;
class UPnPDiscovery;
class WSDiscoveryService;
class DHTNetwork;

// Main audio network discovery service
class AudioNetworkDiscovery {
public:
    AudioNetworkDiscovery();
    ~AudioNetworkDiscovery();

    // Initialization and lifecycle
    bool initialize(const DiscoveryConfig& config = {});
    void shutdown();
    bool isInitialized() const { return initialized_; }

    // Discovery control
    void startDiscovery();
    void stopDiscovery();
    void pauseDiscovery();
    void resumeDiscovery();
    void forceScan();
    bool isDiscovering() const { return discovering_; }

    // Device filtering and searching
    std::vector<AudioDeviceInfo> getDiscoveredDevices() const;
    std::vector<AudioDeviceInfo> getDevicesByCapability(DeviceCapability capability) const;
    std::vector<AudioDeviceInfo> getDevicesByManufacturer(const std::string& manufacturer) const;
    std::vector<AudioDeviceInfo> getDevicesByConnectionType(ConnectionType type) const;
    std::vector<AudioDeviceInfo> getAvailableDevices() const;
    AudioDeviceInfo getDeviceById(const std::string& device_id) const;
    std::vector<AudioDeviceInfo> searchDevices(const std::string& query) const;

    // Device connectivity
    bool connectToDevice(const std::string& device_id);
    bool disconnectFromDevice(const std::string& device_id);
    bool isDeviceConnected(const std::string& device_id) const;
    std::vector<std::string> getConnectedDevices() const;

    // Device monitoring
    void startMonitoring(const std::string& device_id);
    void stopMonitoring(const std::string& device_id);
    void updateDeviceStatus(const std::string& device_id, const AudioDeviceInfo& info);
    AudioDeviceInfo getLatestDeviceInfo(const std::string& device_id) const;

    // Event handling
    void setDiscoveryCallback(std::function<void(const DiscoveryEvent&)> callback);
    void setDeviceStatusCallback(std::function<void(const std::string&, DeviceStatus)> callback);
    void setConnectionCallback(std::function<void(const std::string&, bool)> callback);

    // Statistics and diagnostics
    NetworkStatistics getNetworkStatistics() const;
    std::vector<DiscoveryEvent> getRecentEvents(size_t max_events = 100) const;
    std::string getDiagnosticInfo() const;
    void resetStatistics();

    // Configuration
    void updateConfig(const DiscoveryConfig& config);
    DiscoveryConfig getConfig() const { return config_; }
    void addServiceType(const std::string& service_type);
    void removeServiceType(const std::string& service_type);
    void setRequiredCapability(DeviceCapability capability);
    void removeRequiredCapability(DeviceCapability capability);

    // Network interface management
    std::vector<std::string> getAvailableInterfaces() const;
    void setPreferredInterface(const std::string& interface_name);
    std::string getPreferredInterface() const;

    // Quality assessment
    double calculateDeviceQuality(const AudioDeviceInfo& device) const;
    std::vector<AudioDeviceInfo> getDevicesByQuality() const;
    void setQualityWeights(double latency_weight = 0.3, double throughput_weight = 0.3,
                          double reliability_weight = 0.4);

private:
    // Core components
    std::unique_ptr<MDNSDiscovery> mdns_discovery_;
    std::unique_ptr<UPnPDiscovery> upnp_discovery_;
    std::unique_ptr<WSDiscoveryService> ws_discovery_;
    std::unique_ptr<DHTNetwork> dht_network_;

    // Configuration and state
    DiscoveryConfig config_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> discovering_{false};
    std::atomic<bool> paused_{false};

    // Device registry
    mutable std::mutex devices_mutex_;
    std::unordered_map<std::string, AudioDeviceInfo> discovered_devices_;
    std::unordered_set<std::string> connected_devices_;
    std::unordered_set<std::string> monitored_devices_;

    // Event system
    mutable std::mutex events_mutex_;
    std::queue<DiscoveryEvent> event_queue_;
    std::vector<DiscoveryEvent> event_history_;
    std::function<void(const DiscoveryEvent&)> discovery_callback_;
    std::function<void(const std::string&, DeviceStatus)> status_callback_;
    std::function<void(const std::string&, bool)> connection_callback_;

    // Background threads
    std::thread discovery_thread_;
    std::thread monitoring_thread_;
    std::thread event_processing_thread_;
    std::atomic<bool> running_{false};

    // Statistics
    mutable std::mutex stats_mutex_;
    NetworkStatistics network_stats_;
    std::chrono::steady_clock::time_point stats_start_time_;

    // Quality assessment
    double latency_weight_ = 0.3;
    double throughput_weight_ = 0.3;
    double reliability_weight_ = 0.4;

    // Internal methods
    void discoveryThread();
    void monitoringThread();
    void eventProcessingThread();

    void processDiscoveredDevice(const AudioDeviceInfo& device);
    void processLostDevice(const std::string& device_id);
    void processDeviceUpdate(const AudioDeviceInfo& device);

    bool matchesFilters(const AudioDeviceInfo& device) const;
    bool hasRequiredCapabilities(const AudioDeviceInfo& device) const;
    bool isPreferredDevice(const AudioDeviceInfo& device) const;

    void addEvent(const DiscoveryEvent& event);
    void updateNetworkStatistics();
    void cleanupStaleDevices();

    void initializeDiscoveryProtocols();
    void shutdownDiscoveryProtocols();

    // Protocol-specific handlers
    void onMDNSDeviceDiscovered(const AudioDeviceInfo& device);
    void onUPnPDeviceDiscovered(const AudioDeviceInfo& device);
    void onWSDeviceDiscovered(const AudioDeviceInfo& device);
    void onDHTDeviceDiscovered(const AudioDeviceInfo& device);

    // Device communication
    bool probeDevice(const AudioDeviceInfo& device);
    bool establishConnection(const std::string& device_id);
    void terminateConnection(const std::string& device_id);

    // Utility methods
    std::string generateDeviceId(const AudioDeviceInfo& device) const;
    void updateDeviceInfo(AudioDeviceInfo& existing, const AudioDeviceInfo& updated) const;
    double calculateSignalQuality(const AudioDeviceInfo& device) const;
    bool isDeviceResponsive(const AudioDeviceInfo& device) const;
};

// MDNS discovery implementation
class MDNSDiscovery {
public:
    MDNSDiscovery();
    ~MDNSDiscovery();

    bool initialize(const DiscoveryConfig& config);
    void shutdown();
    void startDiscovery();
    void stopDiscovery();

    void setDeviceCallback(std::function<void(const AudioDeviceInfo&)> callback);

private:
    // MDNS-specific implementation would go here
    std::atomic<bool> running_{false};
    std::thread mdns_thread_;
    std::function<void(const AudioDeviceInfo&)> device_callback_;
    DiscoveryConfig config_;
};

// UPnP discovery implementation
class UPnPDiscovery {
public:
    UPnPDiscovery();
    ~UPnPDiscovery();

    bool initialize(const DiscoveryConfig& config);
    void shutdown();
    void startDiscovery();
    void stopDiscovery();

    void setDeviceCallback(std::function<void(const AudioDeviceInfo&)> callback);

private:
    // UPnP-specific implementation would go here
    std::atomic<bool> running_{false};
    std::thread upnp_thread_;
    std::function<void(const AudioDeviceInfo&)> device_callback_;
    DiscoveryConfig config_;
};

// Web Services Discovery implementation
class WSDiscoveryService {
public:
    WSDiscoveryService();
    ~WSDiscoveryService();

    bool initialize(const DiscoveryConfig& config);
    void shutdown();
    void startDiscovery();
    void stopDiscovery();

    void setDeviceCallback(std::function<void(const AudioDeviceInfo&)> callback);

private:
    // WS-Discovery specific implementation would go here
    std::atomic<bool> running_{false};
    std::thread ws_thread_;
    std::function<void(const AudioDeviceInfo&)> device_callback_;
    DiscoveryConfig config_;
};

// Distributed Hash Table network
class DHTNetwork {
public:
    DHTNetwork();
    ~DHTNetwork();

    bool initialize(const DiscoveryConfig& config);
    void shutdown();
    void startDiscovery();
    void stopDiscovery();

    void setDeviceCallback(std::function<void(const AudioDeviceInfo&)> callback);

private:
    // DHT-specific implementation would go here
    std::atomic<bool> running_{false};
    std::thread dht_thread_;
    std::function<void(const AudioDeviceInfo&)> device_callback_;
    DiscoveryConfig config_;
};

// Utility functions
namespace DiscoveryUtils {
    std::string capabilityToString(DeviceCapability capability);
    std::string connectionTypeToString(ConnectionType type);
    std::string discoveryProtocolToString(DiscoveryProtocol protocol);
    std::string deviceStatusToString(DeviceStatus status);

    DeviceCapability stringToCapability(const std::string& str);
    ConnectionType stringToConnectionType(const std::string& str);
    DiscoveryProtocol stringToDiscoveryProtocol(const std::string& str);
    DeviceStatus stringToDeviceStatus(const std::string& str);

    std::vector<std::string> parseSampleRates(uint32_t rate_mask);
    std::vector<uint16_t> parseBitDepths(uint32_t depth_mask);

    bool isCompatibleProtocol(DiscoveryProtocol protocol, ConnectionType connection);
    double calculateNetworkQuality(const NetworkStatistics& stats);
    std::string generateDeviceFingerprint(const AudioDeviceInfo& device);
}

} // namespace Network
} // namespace VortexGPU