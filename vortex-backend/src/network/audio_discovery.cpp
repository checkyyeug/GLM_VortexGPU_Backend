#include "network/audio_discovery.hpp"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <random>
#include <regex>

namespace VortexGPU {
namespace Network {

// ============================================================================
// AudioNetworkDiscovery Implementation
// ============================================================================

AudioNetworkDiscovery::AudioNetworkDiscovery() {
    stats_start_time_ = std::chrono::steady_clock::now();
}

AudioNetworkDiscovery::~AudioNetworkDiscovery() {
    shutdown();
}

bool AudioNetworkDiscovery::initialize(const DiscoveryConfig& config) {
    if (initialized_.load()) {
        return true;
    }

    config_ = config;

    try {
        // Initialize discovery protocols
        initializeDiscoveryProtocols();

        running_.store(true);

        // Start background threads
        discovery_thread_ = std::thread(&AudioNetworkDiscovery::discoveryThread, this);
        monitoring_thread_ = std::thread(&AudioNetworkDiscovery::monitoringThread, this);
        event_processing_thread_ = std::thread(&AudioNetworkDiscovery::eventProcessingThread, this);

        initialized_.store(true);

        std::cout << "AudioNetworkDiscovery initialized with protocols: ";
        if (config_.enable_mdns) std::cout << "MDNS ";
        if (config_.enable_upnp) std::cout << "UPnP ";
        if (config_.enable_ws_discovery) std::cout << "WS-Discovery ";
        if (config_.enable_dht) std::cout << "DHT ";
        std::cout << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize AudioNetworkDiscovery: " << e.what() << std::endl;
        shutdown();
        return false;
    }
}

void AudioNetworkDiscovery::shutdown() {
    if (!initialized_.load()) {
        return;
    }

    running_.store(false);
    discovering_.store(false);

    // Wait for threads to finish
    if (discovery_thread_.joinable()) {
        discovery_thread_.join();
    }

    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }

    if (event_processing_thread_.joinable()) {
        event_processing_thread_.join();
    }

    // Shutdown discovery protocols
    shutdownDiscoveryProtocols();

    // Clear device registry
    {
        std::lock_guard<std::mutex> lock(devices_mutex_);
        discovered_devices_.clear();
        connected_devices_.clear();
        monitored_devices_.clear();
    }

    // Clear events
    {
        std::lock_guard<std::mutex> lock(events_mutex_);
        event_queue_ = {};
        event_history_.clear();
    }

    initialized_.store(false);

    std::cout << "AudioNetworkDiscovery shut down" << std::endl;
}

void AudioNetworkDiscovery::startDiscovery() {
    if (!initialized_.load()) {
        return;
    }

    discovering_.store(true);

    // Start protocol-specific discovery
    if (config_.enable_mdns && mdns_discovery_) {
        mdns_discovery_->startDiscovery();
    }

    if (config_.enable_upnp && upnp_discovery_) {
        upnp_discovery_->startDiscovery();
    }

    if (config_.enable_ws_discovery && ws_discovery_) {
        ws_discovery_->startDiscovery();
    }

    if (config_.enable_dht && dht_network_) {
        dht_network_->startDiscovery();
    }

    std::cout << "Started network audio discovery" << std::endl;
}

void AudioNetworkDiscovery::stopDiscovery() {
    discovering_.store(false);

    // Stop protocol-specific discovery
    if (mdns_discovery_) {
        mdns_discovery_->stopDiscovery();
    }

    if (upnp_discovery_) {
        upnp_discovery_->stopDiscovery();
    }

    if (ws_discovery_) {
        ws_discovery_->stopDiscovery();
    }

    if (dht_network_) {
        dht_network_->stopDiscovery();
    }

    std::cout << "Stopped network audio discovery" << std::endl;
}

void AudioNetworkDiscovery::pauseDiscovery() {
    paused_.store(true);
}

void AudioNetworkDiscovery::resumeDiscovery() {
    paused_.store(false);
}

void AudioNetworkDiscovery::forceScan() {
    if (!discovering_.load()) {
        startDiscovery();
    }

    // Trigger immediate scans on all protocols
    if (mdns_discovery_) {
        // Force mDNS rescan
    }

    if (upnp_discovery_) {
        // Force UPnP rescan
    }

    std::cout << "Forced network discovery scan" << std::endl;
}

std::vector<AudioDeviceInfo> AudioNetworkDiscovery::getDiscoveredDevices() const {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    std::vector<AudioDeviceInfo> devices;
    devices.reserve(discovered_devices_.size());

    for (const auto& pair : discovered_devices_) {
        devices.push_back(pair.second);
    }

    return devices;
}

std::vector<AudioDeviceInfo> AudioNetworkDiscovery::getDevicesByCapability(DeviceCapability capability) const {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    std::vector<AudioDeviceInfo> devices;

    for (const auto& pair : discovered_devices_) {
        const AudioDeviceInfo& device = pair.second;
        if (std::find(device.capabilities.begin(), device.capabilities.end(), capability) != device.capabilities.end()) {
            devices.push_back(device);
        }
    }

    return devices;
}

std::vector<AudioDeviceInfo> AudioNetworkDiscovery::getDevicesByManufacturer(const std::string& manufacturer) const {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    std::vector<AudioDeviceInfo> devices;

    for (const auto& pair : discovered_devices_) {
        const AudioDeviceInfo& device = pair.second;
        if (device.manufacturer == manufacturer) {
            devices.push_back(device);
        }
    }

    return devices;
}

std::vector<AudioDeviceInfo> AudioNetworkDiscovery::getDevicesByConnectionType(ConnectionType type) const {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    std::vector<AudioDeviceInfo> devices;

    for (const auto& pair : discovered_devices_) {
        const AudioDeviceInfo& device = pair.second;
        if (std::find(device.connection_types.begin(), device.connection_types.end(), type) != device.connection_types.end()) {
            devices.push_back(device);
        }
    }

    return devices;
}

std::vector<AudioDeviceInfo> AudioNetworkDiscovery::getAvailableDevices() const {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    std::vector<AudioDeviceInfo> devices;
    auto now = std::chrono::steady_clock::now();

    for (const auto& pair : discovered_devices_) {
        const AudioDeviceInfo& device = pair.second;
        if (device.status == DeviceStatus::ONLINE &&
            std::chrono::duration_cast<std::chrono::seconds>(now - device.last_seen).count() < config_.device_timeout.count()) {
            devices.push_back(device);
        }
    }

    return devices;
}

AudioDeviceInfo AudioNetworkDiscovery::getDeviceById(const std::string& device_id) const {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    auto it = discovered_devices_.find(device_id);
    if (it != discovered_devices_.end()) {
        return it->second;
    }

    return AudioDeviceInfo{};
}

std::vector<AudioDeviceInfo> AudioNetworkDiscovery::searchDevices(const std::string& query) const {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    std::vector<AudioDeviceInfo> devices;
    std::string lower_query = query;
    std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(), ::tolower);

    for (const auto& pair : discovered_devices_) {
        const AudioDeviceInfo& device = pair.second;

        // Search in name, manufacturer, and model
        std::string search_text = device.device_name + " " + device.manufacturer + " " + device.model;
        std::transform(search_text.begin(), search_text.end(), search_text.begin(), ::tolower);

        if (search_text.find(lower_query) != std::string::npos) {
            devices.push_back(device);
        }
    }

    return devices;
}

bool AudioNetworkDiscovery::connectToDevice(const std::string& device_id) {
    if (!device_id.empty()) {
        return establishConnection(device_id);
    }
    return false;
}

bool AudioNetworkDiscovery::disconnectFromDevice(const std::string& device_id) {
    if (!device_id.empty()) {
        terminateConnection(device_id);
        return true;
    }
    return false;
}

bool AudioNetworkDiscovery::isDeviceConnected(const std::string& device_id) const {
    std::lock_guard<std::mutex> lock(devices_mutex_);
    return connected_devices_.find(device_id) != connected_devices_.end();
}

std::vector<std::string> AudioNetworkDiscovery::getConnectedDevices() const {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    std::vector<std::string> connected;
    connected.reserve(connected_devices_.size());

    for (const auto& device_id : connected_devices_) {
        connected.push_back(device_id);
    }

    return connected;
}

void AudioNetworkDiscovery::startMonitoring(const std::string& device_id) {
    std::lock_guard<std::mutex> lock(devices_mutex_);
    monitored_devices_.insert(device_id);
}

void AudioNetworkDiscovery::stopMonitoring(const std::string& device_id) {
    std::lock_guard<std::mutex> lock(devices_mutex_);
    monitored_devices_.erase(device_id);
}

void AudioNetworkDiscovery::updateDeviceStatus(const std::string& device_id, const AudioDeviceInfo& info) {
    processDeviceUpdate(info);
}

AudioDeviceInfo AudioNetworkDiscovery::getLatestDeviceInfo(const std::string& device_id) const {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    auto it = discovered_devices_.find(device_id);
    if (it != discovered_devices_.end()) {
        return it->second;
    }

    return AudioDeviceInfo{};
}

void AudioNetworkDiscovery::setDiscoveryCallback(std::function<void(const DiscoveryEvent&)> callback) {
    discovery_callback_ = callback;
}

void AudioNetworkDiscovery::setDeviceStatusCallback(std::function<void(const std::string&, DeviceStatus)> callback) {
    status_callback_ = callback;
}

void AudioNetworkDiscovery::setConnectionCallback(std::function<void(const std::string&, bool)> callback) {
    connection_callback_ = callback;
}

NetworkStatistics AudioNetworkDiscovery::getNetworkStatistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return network_stats_;
}

std::vector<DiscoveryEvent> AudioNetworkDiscovery::getRecentEvents(size_t max_events) const {
    std::lock_guard<std::mutex> lock(events_mutex_);

    std::vector<DiscoveryEvent> events;
    size_t start_index = event_history_.size() > max_events ? event_history_.size() - max_events : 0;

    for (size_t i = start_index; i < event_history_.size(); ++i) {
        events.push_back(event_history_[i]);
    }

    return events;
}

std::string AudioNetworkDiscovery::getDiagnosticInfo() const {
    std::ostringstream oss;

    oss << "=== Audio Network Discovery Diagnostics ===\n";
    oss << "Initialized: " << (initialized_.load() ? "Yes" : "No") << "\n";
    oss << "Discovering: " << (discovering_.load() ? "Yes" : "No") << "\n";
    oss << "Paused: " << (paused_.load() ? "Yes" : "No") << "\n";

    {
        std::lock_guard<std::mutex> lock(devices_mutex_);
        oss << "Discovered devices: " << discovered_devices_.size() << "\n";
        oss << "Connected devices: " << connected_devices_.size() << "\n";
        oss << "Monitored devices: " << monitored_devices_.size() << "\n";
    }

    {
        std::lock_guard<std::mutex> lock(events_mutex_);
        oss << "Event queue size: " << event_queue_.size() << "\n";
        oss << "Event history size: " << event_history_.size() << "\n";
    }

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        oss << "Total packets sent: " << network_stats_.total_packets_sent << "\n";
        oss << "Total packets received: " << network_stats_.total_packets_received << "\n";
        oss << "Active connections: " << network_stats_.active_connections << "\n";
        oss << "Failed connections: " << network_stats_.failed_connections << "\n";
        oss << "Average latency: " << std::fixed << std::setprecision(2) << network_stats_.average_latency_ms << " ms\n";
    }

    return oss.str();
}

void AudioNetworkDiscovery::resetStatistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    network_stats_ = NetworkStatistics{};
    stats_start_time_ = std::chrono::steady_clock::now();
}

void AudioNetworkDiscovery::updateConfig(const DiscoveryConfig& config) {
    config_ = config;

    // Reinitialize discovery protocols if needed
    if (initialized_.load()) {
        initializeDiscoveryProtocols();
    }
}

void AudioNetworkDiscovery::addServiceType(const std::string& service_type) {
    config_.service_types.push_back(service_type);
}

void AudioNetworkDiscovery::removeServiceType(const std::string& service_type) {
    auto it = std::find(config_.service_types.begin(), config_.service_types.end(), service_type);
    if (it != config_.service_types.end()) {
        config_.service_types.erase(it);
    }
}

void AudioNetworkDiscovery::setRequiredCapability(DeviceCapability capability) {
    if (std::find(config_.required_capabilities.begin(), config_.required_capabilities.end(), capability)
        == config_.required_capabilities.end()) {
        config_.required_capabilities.push_back(capability);
    }
}

void AudioNetworkDiscovery::removeRequiredCapability(DeviceCapability capability) {
    auto it = std::find(config_.required_capabilities.begin(), config_.required_capabilities.end(), capability);
    if (it != config_.required_capabilities.end()) {
        config_.required_capabilities.erase(it);
    }
}

std::vector<std::string> AudioNetworkDiscovery::getAvailableInterfaces() const {
    // This would typically query the system for network interfaces
    // For now, return common interface names
    return {"eth0", "wlan0", "en0", "en1"};
}

void AudioNetworkDiscovery::setPreferredInterface(const std::string& interface_name) {
    config_.network_interfaces.clear();
    config_.network_interfaces.push_back(interface_name);
}

std::string AudioNetworkDiscovery::getPreferredInterface() const {
    if (!config_.network_interfaces.empty()) {
        return config_.network_interfaces[0];
    }
    return "";
}

double AudioNetworkDiscovery::calculateDeviceQuality(const AudioDeviceInfo& device) const {
    double quality = 0.0;

    // Latency component (lower is better)
    double latency_score = 100.0;
    if (device.typical_latency_ms > 0) {
        latency_score = std::max(0.0, 100.0 - (device.typical_latency_ms / 10.0));
    }
    quality += latency_score * latency_weight_;

    // Throughput component (higher is better)
    double throughput_score = std::min(100.0, device.max_throughput_mbps / 100.0);
    quality += throughput_score * throughput_weight_;

    // Reliability component (lower packet loss is better)
    double reliability_score = 100.0;
    if (device.packet_loss_percent > 0) {
        reliability_score = std::max(0.0, 100.0 - (device.packet_loss_percent * 10.0));
    }
    quality += reliability_score * reliability_weight_;

    return std::min(100.0, quality);
}

std::vector<AudioDeviceInfo> AudioNetworkDiscovery::getDevicesByQuality() const {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    std::vector<AudioDeviceInfo> devices;
    for (const auto& pair : discovered_devices_) {
        devices.push_back(pair.second);
    }

    std::sort(devices.begin(), devices.end(), [this](const AudioDeviceInfo& a, const AudioDeviceInfo& b) {
        return calculateDeviceQuality(a) > calculateDeviceQuality(b);
    });

    return devices;
}

void AudioNetworkDiscovery::setQualityWeights(double latency_weight, double throughput_weight, double reliability_weight) {
    double total = latency_weight + throughput_weight + reliability_weight;
    if (total > 0) {
        latency_weight_ = latency_weight / total;
        throughput_weight_ = throughput_weight / total;
        reliability_weight_ = reliability_weight / total;
    }
}

void AudioNetworkDiscovery::discoveryThread() {
    auto last_scan = std::chrono::steady_clock::now();

    while (running_.load()) {
        try {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_scan);

            if (discovering_.load() && !paused_.load() && elapsed >= config_.scan_interval) {
                // Perform periodic scan
                cleanupStaleDevices();
                updateNetworkStatistics();
                last_scan = now;
            }

            std::this_thread::sleep_for(std::chrono::seconds(1));

        } catch (const std::exception& e) {
            std::cerr << "Discovery thread error: " << e.what() << std::endl;
        }
    }
}

void AudioNetworkDiscovery::monitoringThread() {
    while (running_.load()) {
        try {
            std::vector<std::string> devices_to_monitor;

            {
                std::lock_guard<std::mutex> lock(devices_mutex_);
                devices_to_monitor.assign(monitored_devices_.begin(), monitored_devices_.end());
            }

            for (const auto& device_id : devices_to_monitor) {
                auto device_info = getDeviceById(device_id);
                if (!device_info.device_id.empty()) {
                    // Check if device is still responsive
                    if (isDeviceResponsive(device_info)) {
                        device_info.last_seen = std::chrono::steady_clock::now();
                        processDeviceUpdate(device_info);
                    } else {
                        device_info.status = DeviceStatus::OFFLINE;
                        processLostDevice(device_id);
                    }
                }
            }

            std::this_thread::sleep_for(std::chrono::seconds(10));

        } catch (const std::exception& e) {
            std::cerr << "Monitoring thread error: " << e.what() << std::endl;
        }
    }
}

void AudioNetworkDiscovery::eventProcessingThread() {
    while (running_.load()) {
        try {
            std::queue<DiscoveryEvent> events_to_process;

            {
                std::lock_guard<std::mutex> lock(events_mutex_);
                events_to_process.swap(event_queue_);
            }

            while (!events_to_process.empty()) {
                DiscoveryEvent event = events_to_process.front();
                events_to_process.pop();

                // Add to history
                {
                    std::lock_guard<std::mutex> lock(events_mutex_);
                    event_history_.push_back(event);
                    if (event_history_.size() > 1000) {
                        event_history_.erase(event_history_.begin());
                    }
                }

                // Call callback if set
                if (discovery_callback_) {
                    discovery_callback_(event);
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));

        } catch (const std::exception& e) {
            std::cerr << "Event processing thread error: " << e.what() << std::endl;
        }
    }
}

void AudioNetworkDiscovery::processDiscoveredDevice(const AudioDeviceInfo& device) {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    if (!matchesFilters(device)) {
        return;
    }

    std::string device_id = generateDeviceId(device);
    auto it = discovered_devices_.find(device_id);

    if (it == discovered_devices_.end()) {
        // New device discovered
        AudioDeviceInfo new_device = device;
        new_device.device_id = device_id;
        new_device.first_seen = std::chrono::steady_clock::now();
        new_device.last_seen = new_device.first_seen;
        new_device.status = DeviceStatus::ONLINE;

        discovered_devices_[device_id] = new_device;

        DiscoveryEvent event;
        event.type = DiscoveryEventType::DEVICE_DISCOVERED;
        event.device_id = device_id;
        event.device_info = new_device;
        event.timestamp = new_device.first_seen;
        event.message = "New audio device discovered: " + new_device.device_name;

        addEvent(event);
        std::cout << "Discovered audio device: " << new_device.device_name << std::endl;

    } else {
        // Update existing device
        updateDeviceInfo(it->second, device);
        it->second.last_seen = std::chrono::steady_clock::now();

        DiscoveryEvent event;
        event.type = DiscoveryEventType::DEVICE_UPDATED;
        event.device_id = device_id;
        event.device_info = it->second;
        event.timestamp = it->second.last_seen;
        event.message = "Audio device updated: " + it->second.device_name;

        addEvent(event);
    }
}

void AudioNetworkDiscovery::processLostDevice(const std::string& device_id) {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    auto it = discovered_devices_.find(device_id);
    if (it != discovered_devices_.end()) {
        it->second.status = DeviceStatus::OFFLINE;

        DiscoveryEvent event;
        event.type = DiscoveryEventType::DEVICE_LOST;
        event.device_id = device_id;
        event.device_info = it->second;
        event.timestamp = std::chrono::steady_clock::now();
        event.message = "Audio device lost: " + it->second.device_name;

        addEvent(event);
        std::cout << "Lost audio device: " << it->second.device_name << std::endl;

        // Remove from connected devices
        connected_devices_.erase(device_id);
        monitored_devices_.erase(device_id);

        // Call callbacks
        if (status_callback_) {
            status_callback_(device_id, DeviceStatus::OFFLINE);
        }

        if (connection_callback_) {
            connection_callback_(device_id, false);
        }
    }
}

void AudioNetworkDiscovery::processDeviceUpdate(const AudioDeviceInfo& device) {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    std::string device_id = device.device_id;
    auto it = discovered_devices_.find(device_id);

    if (it != discovered_devices_.end()) {
        DeviceStatus old_status = it->second.status;
        updateDeviceInfo(it->second, device);
        it->second.last_seen = std::chrono::steady_clock::now();

        // Check if status changed
        if (old_status != it->second.status && status_callback_) {
            status_callback_(device_id, it->second.status);
        }

        DiscoveryEvent event;
        event.type = DiscoveryEventType::DEVICE_UPDATED;
        event.device_id = device_id;
        event.device_info = it->second;
        event.timestamp = it->second.last_seen;
        event.message = "Audio device status updated: " + it->second.device_name;

        addEvent(event);
    }
}

bool AudioNetworkDiscovery::matchesFilters(const AudioDeviceInfo& device) const {
    // Check excluded devices
    for (const auto& excluded : config_.excluded_devices) {
        if (device.device_name.find(excluded) != std::string::npos ||
            device.manufacturer.find(excluded) != std::string::npos ||
            device.model.find(excluded) != std::string::npos) {
            return false;
        }
    }

    // Check required capabilities
    if (!hasRequiredCapabilities(device)) {
        return false;
    }

    // Check channel requirements
    if (config_.min_required_channels > 0 && device.max_channels < config_.min_required_channels) {
        return false;
    }

    // Check latency requirements
    if (config_.max_acceptable_latency_ms > 0 && device.min_latency_ms > config_.max_acceptable_latency_ms) {
        return false;
    }

    return true;
}

bool AudioNetworkDiscovery::hasRequiredCapabilities(const AudioDeviceInfo& device) const {
    for (auto required_capability : config_.required_capabilities) {
        bool found = false;
        for (auto device_capability : device.capabilities) {
            if (device_capability == required_capability) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

bool AudioNetworkDiscovery::isPreferredDevice(const AudioDeviceInfo& device) const {
    // Check if manufacturer is preferred
    for (const auto& preferred : config_.preferred_manufacturers) {
        if (device.manufacturer == preferred) {
            return true;
        }
    }

    // Check wired preference
    if (config_.prefer_wired) {
        for (auto connection_type : device.connection_types) {
            if (connection_type == ConnectionType::WIRED_ETHERNET ||
                connection_type == ConnectionType::USB ||
                connection_type == ConnectionType::THUNDERBOLT ||
                connection_type == ConnectionType::PCIe) {
                return true;
            }
        }
    }

    return false;
}

void AudioNetworkDiscovery::addEvent(const DiscoveryEvent& event) {
    std::lock_guard<std::mutex> lock(events_mutex_);
    event_queue_.push(event);
}

void AudioNetworkDiscovery::updateNetworkStatistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    network_stats_.last_update_time = std::chrono::steady_clock::now();

    {
        std::lock_guard<std::mutex> devices_lock(devices_mutex_);
        network_stats_.active_connections = connected_devices_.size();
    }

    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        network_stats_.last_update_time - stats_start_time_);

    if (duration.count() > 0) {
        // Calculate averages
        uint64_t total_packets = network_stats_.total_packets_sent + network_stats_.total_packets_received;
        if (total_packets > 0) {
            // Average latency calculation would be based on actual measurements
            network_stats_.average_latency_ms = 15.0; // Placeholder
        }
    }
}

void AudioNetworkDiscovery::cleanupStaleDevices() {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    auto now = std::chrono::steady_clock::now();
    auto timeout_threshold = config_.device_timeout;

    auto it = discovered_devices_.begin();
    while (it != discovered_devices_.end()) {
        auto inactive_time = std::chrono::duration_cast<std::chrono::seconds>(now - it->second.last_seen);

        if (inactive_time > timeout_threshold && it->second.status == DeviceStatus::ONLINE) {
            processLostDevice(it->first);
            it = discovered_devices_.erase(it);
        } else {
            ++it;
        }
    }
}

void AudioNetworkDiscovery::initializeDiscoveryProtocols() {
    // Initialize mDNS discovery
    if (config_.enable_mdns && !mdns_discovery_) {
        mdns_discovery_ = std::make_unique<MDNSDiscovery>();
        mdns_discovery_->initialize(config_);
        mdns_discovery_->setDeviceCallback([this](const AudioDeviceInfo& device) {
            onMDNSDeviceDiscovered(device);
        });
    }

    // Initialize UPnP discovery
    if (config_.enable_upnp && !upnp_discovery_) {
        upnp_discovery_ = std::make_unique<UPnPDiscovery>();
        upnp_discovery_->initialize(config_);
        upnp_discovery_->setDeviceCallback([this](const AudioDeviceInfo& device) {
            onUPnPDeviceDiscovered(device);
        });
    }

    // Initialize WS-Discovery
    if (config_.enable_ws_discovery && !ws_discovery_) {
        ws_discovery_ = std::make_unique<WSDiscoveryService>();
        ws_discovery_->initialize(config_);
        ws_discovery_->setDeviceCallback([this](const AudioDeviceInfo& device) {
            onWSDeviceDiscovered(device);
        });
    }

    // Initialize DHT network
    if (config_.enable_dht && !dht_network_) {
        dht_network_ = std::make_unique<DHTNetwork>();
        dht_network_->initialize(config_);
        dht_network_->setDeviceCallback([this](const AudioDeviceInfo& device) {
            onDHTDeviceDiscovered(device);
        });
    }
}

void AudioNetworkDiscovery::shutdownDiscoveryProtocols() {
    if (mdns_discovery_) {
        mdns_discovery_->shutdown();
        mdns_discovery_.reset();
    }

    if (upnp_discovery_) {
        upnp_discovery_->shutdown();
        upnp_discovery_.reset();
    }

    if (ws_discovery_) {
        ws_discovery_->shutdown();
        ws_discovery_.reset();
    }

    if (dht_network_) {
        dht_network_->shutdown();
        dht_network_.reset();
    }
}

void AudioNetworkDiscovery::onMDNSDeviceDiscovered(const AudioDeviceInfo& device) {
    processDiscoveredDevice(device);
}

void AudioNetworkDiscovery::onUPnPDeviceDiscovered(const AudioDeviceInfo& device) {
    processDiscoveredDevice(device);
}

void AudioNetworkDiscovery::onWSDeviceDiscovered(const AudioDeviceInfo& device) {
    processDiscoveredDevice(device);
}

void AudioNetworkDiscovery::onDHTDeviceDiscovered(const AudioDeviceInfo& device) {
    processDiscoveredDevice(device);
}

bool AudioNetworkDiscovery::probeDevice(const AudioDeviceInfo& device) {
    // In a real implementation, this would send a probe packet to the device
    // and wait for a response to verify it's actually accessible
    return device.status == DeviceStatus::ONLINE;
}

bool AudioNetworkDiscovery::establishConnection(const std::string& device_id) {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    auto it = discovered_devices_.find(device_id);
    if (it == discovered_devices_.end()) {
        return false;
    }

    if (connected_devices_.find(device_id) != connected_devices_.end()) {
        return true; // Already connected
    }

    // Attempt to establish connection
    if (probeDevice(it->second)) {
        connected_devices_.insert(device_id);
        it->second.connection_count++;

        DiscoveryEvent event;
        event.type = DiscoveryEventType::DEVICE_CONNECTED;
        event.device_id = device_id;
        event.device_info = it->second;
        event.timestamp = std::chrono::steady_clock::now();
        event.message = "Connected to audio device: " + it->second.device_name;

        addEvent(event);

        if (connection_callback_) {
            connection_callback_(device_id, true);
        }

        std::cout << "Connected to audio device: " << it->second.device_name << std::endl;
        return true;
    } else {
        {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            network_stats_.failed_connections++;
        }
        return false;
    }
}

void AudioNetworkDiscovery::terminateConnection(const std::string& device_id) {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    if (connected_devices_.erase(device_id) > 0) {
        auto it = discovered_devices_.find(device_id);
        if (it != discovered_devices_.end()) {
            DiscoveryEvent event;
            event.type = DiscoveryEventType::DEVICE_DISCONNECTED;
            event.device_id = device_id;
            event.device_info = it->second;
            event.timestamp = std::chrono::steady_clock::now();
            event.message = "Disconnected from audio device: " + it->second.device_name;

            addEvent(event);
            std::cout << "Disconnected from audio device: " << it->second.device_name << std::endl;
        }

        if (connection_callback_) {
            connection_callback_(device_id, false);
        }
    }
}

std::string AudioNetworkDiscovery::generateDeviceId(const AudioDeviceInfo& device) const {
    std::ostringstream oss;
    oss << device.manufacturer << "_" << device.model << "_" << device.serial_number;

    std::string id = oss.str();
    std::replace(id.begin(), id.end(), ' ', '_');
    std::replace(id.begin(), id.end(), ':', '_');

    return id;
}

void AudioNetworkDiscovery::updateDeviceInfo(AudioDeviceInfo& existing, const AudioDeviceInfo& updated) const {
    // Update only non-empty fields from updated info
    if (!updated.manufacturer.empty()) existing.manufacturer = updated.manufacturer;
    if (!updated.model.empty()) existing.model = updated.model;
    if (!updated.firmware_version.empty()) existing.firmware_version = updated.firmware_version;
    if (!updated.mac_address.empty()) existing.mac_address = updated.mac_address;
    if (updated.control_port > 0) existing.control_port = updated.control_port;
    if (updated.streaming_port > 0) existing.streaming_port = updated.streaming_port;
    if (!updated.capabilities.empty()) existing.capabilities = updated.capabilities;
    if (updated.max_channels > 0) existing.max_channels = updated.max_channels;
    if (updated.supported_sample_rates > 0) existing.supported_sample_rates = updated.supported_sample_rates;
    if (updated.min_latency_ms > 0) existing.min_latency_ms = updated.min_latency_ms;
    if (updated.max_throughput_mbps > 0) existing.max_throughput_mbps = updated.max_throughput_mbps;

    // Update status
    existing.status = updated.status;
    existing.last_seen = std::chrono::steady_clock::now();
}

double AudioNetworkDiscovery::calculateSignalQuality(const AudioDeviceInfo& device) const {
    double quality = 0.0;

    // Signal strength component
    if (device.signal_strength_dbm > -50) {
        quality += 40.0;
    } else if (device.signal_strength_dbm > -70) {
        quality += 30.0;
    } else if (device.signal_strength_dbm > -85) {
        quality += 20.0;
    } else {
        quality += 10.0;
    }

    // Packet loss component
    if (device.packet_loss_percent < 1.0) {
        quality += 30.0;
    } else if (device.packet_loss_percent < 3.0) {
        quality += 20.0;
    } else if (device.packet_loss_percent < 5.0) {
        quality += 10.0;
    }

    // Jitter component
    if (device.jitter_ms < 5.0) {
        quality += 30.0;
    } else if (device.jitter_ms < 10.0) {
        quality += 20.0;
    } else if (device.jitter_ms < 20.0) {
        quality += 10.0;
    }

    return std::min(100.0, quality);
}

bool AudioNetworkDiscovery::isDeviceResponsive(const AudioDeviceInfo& device) const {
    // Check if the device has been seen recently
    auto now = std::chrono::steady_clock::now();
    auto time_since_last_seen = std::chrono::duration_cast<std::chrono::seconds>(now - device.last_seen);

    return time_since_last_seen.count() < 60; // Consider responsive if seen within last minute
}

// ============================================================================
// MDNSDiscovery Implementation
// ============================================================================

MDNSDiscovery::MDNSDiscovery() {
}

MDNSDiscovery::~MDNSDiscovery() {
    shutdown();
}

bool MDNSDiscovery::initialize(const DiscoveryConfig& config) {
    config_ = config;
    // Initialize mDNS discovery components
    return true;
}

void MDNSDiscovery::shutdown() {
    running_.store(false);
    if (mdns_thread_.joinable()) {
        mdns_thread_.join();
    }
}

void MDNSDiscovery::startDiscovery() {
    if (running_.load()) {
        return;
    }

    running_.store(true);
    mdns_thread_ = std::thread([this]() {
        // mDNS discovery loop
        while (running_.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    });
}

void MDNSDiscovery::stopDiscovery() {
    running_.store(false);
    if (mdns_thread_.joinable()) {
        mdns_thread_.join();
    }
}

void MDNSDiscovery::setDeviceCallback(std::function<void(const AudioDeviceInfo&)> callback) {
    device_callback_ = callback;
}

// ============================================================================
// UPnPDiscovery Implementation
// ============================================================================

UPnPDiscovery::UPnPDiscovery() {
}

UPnPDiscovery::~UPnPDiscovery() {
    shutdown();
}

bool UPnPDiscovery::initialize(const DiscoveryConfig& config) {
    config_ = config;
    // Initialize UPnP discovery components
    return true;
}

void UPnPDiscovery::shutdown() {
    running_.store(false);
    if (upnp_thread_.joinable()) {
        upnp_thread_.join();
    }
}

void UPnPDiscovery::startDiscovery() {
    if (running_.load()) {
        return;
    }

    running_.store(true);
    upnp_thread_ = std::thread([this]() {
        // UPnP discovery loop
        while (running_.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }
    });
}

void UPnPDiscovery::stopDiscovery() {
    running_.store(false);
    if (upnp_thread_.joinable()) {
        upnp_thread_.join();
    }
}

void UPnPDiscovery::setDeviceCallback(std::function<void(const AudioDeviceInfo&)> callback) {
    device_callback_ = callback;
}

// ============================================================================
// WSDiscoveryService Implementation
// ============================================================================

WSDiscoveryService::WSDiscoveryService() {
}

WSDiscoveryService::~WSDiscoveryService() {
    shutdown();
}

bool WSDiscoveryService::initialize(const DiscoveryConfig& config) {
    config_ = config;
    // Initialize WS-Discovery components
    return true;
}

void WSDiscoveryService::shutdown() {
    running_.store(false);
    if (ws_thread_.joinable()) {
        ws_thread_.join();
    }
}

void WSDiscoveryService::startDiscovery() {
    if (running_.load()) {
        return;
    }

    running_.store(true);
    ws_thread_ = std::thread([this]() {
        // WS-Discovery loop
        while (running_.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(15));
        }
    });
}

void WSDiscoveryService::stopDiscovery() {
    running_.store(false);
    if (ws_thread_.joinable()) {
        ws_thread_.join();
    }
}

void WSDiscoveryService::setDeviceCallback(std::function<void(const AudioDeviceInfo&)> callback) {
    device_callback_ = callback;
}

// ============================================================================
// DHTNetwork Implementation
// ============================================================================

DHTNetwork::DHTNetwork() {
}

DHTNetwork::~DHTNetwork() {
    shutdown();
}

bool DHTNetwork::initialize(const DiscoveryConfig& config) {
    config_ = config;
    // Initialize DHT network components
    return true;
}

void DHTNetwork::shutdown() {
    running_.store(false);
    if (dht_thread_.joinable()) {
        dht_thread_.join();
    }
}

void DHTNetwork::startDiscovery() {
    if (running_.load()) {
        return;
    }

    running_.store(true);
    dht_thread_ = std::thread([this]() {
        // DHT discovery loop
        while (running_.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(20));
        }
    });
}

void DHTNetwork::stopDiscovery() {
    running_.store(false);
    if (dht_thread_.joinable()) {
        dht_thread_.join();
    }
}

void DHTNetwork::setDeviceCallback(std::function<void(const AudioDeviceInfo&)> callback) {
    device_callback_ = callback;
}

// ============================================================================
// DiscoveryUtils Implementation
// ============================================================================

namespace DiscoveryUtils {

std::string capabilityToString(DeviceCapability capability) {
    switch (capability) {
        case DeviceCapability::AUDIO_INPUT: return "Audio Input";
        case DeviceCapability::AUDIO_OUTPUT: return "Audio Output";
        case DeviceCapability::MULTICHANNEL: return "Multi-channel";
        case DeviceCapability::LOW_LATENCY: return "Low Latency";
        case DeviceCapability::NETWORK_STREAMING: return "Network Streaming";
        case DeviceCapability::HARDWARE_ACCELERATION: return "Hardware Acceleration";
        case DeviceCapability::VR_AUDIO: return "VR Audio";
        case DeviceCapability::SPATIAL_AUDIO: return "Spatial Audio";
        case DeviceCapability::PROFESSIONAL_AUDIO: return "Professional Audio";
        case DeviceCapability::MIDI_SUPPORT: return "MIDI Support";
        case DeviceCapability::CONTROL_SURFACE: return "Control Surface";
        case DeviceCapability::EFFECTS_PROCESSING: return "Effects Processing";
        case DeviceCapability::MIXING_CONSOLE: return "Mixing Console";
        case DeviceCapability::RECORDING: return "Recording";
        case DeviceCapability::PLAYBACK: return "Playback";
        case DeviceCapability::DUPLEX: return "Duplex";
        case DeviceCapability::CLOCK_SYNC: return "Clock Sync";
        case DeviceCapability::WORD_CLOCK: return "Word Clock";
        case DeviceCapability::AES3: return "AES3";
        case DeviceCapability::ADAT: return "ADAT";
        case DeviceCapability::MADI: return "MADI";
        case DeviceCapability::DANTE: return "Dante";
        case DeviceCapability::RAVENNA: return "RAVENNA";
        case DeviceCapability::AES67: return "AES67";
        case DeviceCapability::NDI: return "NDI";
        case DeviceCapability::SRT: return "SRT";
        default: return "Unknown";
    }
}

std::string connectionTypeToString(ConnectionType type) {
    switch (type) {
        case ConnectionType::WIRED_ETHERNET: return "Wired Ethernet";
        case ConnectionType::WIRELESS_WIFI: return "Wireless WiFi";
        case ConnectionType::WIRELESS_BLUETOOTH: return "Wireless Bluetooth";
        case ConnectionType::USB: return "USB";
        case ConnectionType::THUNDERBOLT: return "Thunderbolt";
        case ConnectionType::FIREWIRE: return "FireWire";
        case ConnectionType::PCI: return "PCI";
        case ConnectionType::PCIe: return "PCIe";
        case ConnectionType::INTERNAL: return "Internal";
        case ConnectionType::OPTICAL: return "Optical";
        case ConnectionType::COAXIAL: return "Coaxial";
        default: return "Unknown";
    }
}

std::string discoveryProtocolToString(DiscoveryProtocol protocol) {
    switch (protocol) {
        case DiscoveryProtocol::MDNS: return "mDNS";
        case DiscoveryProtocol::UPnP: return "UPnP";
        case DiscoveryProtocol::WS_DISCOVERY: return "WS-Discovery";
        case DiscoveryProtocol::DHT: return "DHT";
        case DiscoveryProtocol::CUSTOM: return "Custom";
        default: return "Unknown";
    }
}

std::string deviceStatusToString(DeviceStatus status) {
    switch (status) {
        case DeviceStatus::ONLINE: return "Online";
        case DeviceStatus::OFFLINE: return "Offline";
        case DeviceStatus::BUSY: return "Busy";
        case DeviceStatus::ERROR: return "Error";
        case DeviceStatus::MAINTENANCE: return "Maintenance";
        case DeviceStatus::UNKNOWN: return "Unknown";
        default: return "Unknown";
    }
}

DeviceCapability stringToCapability(const std::string& str) {
    // This would typically use a map for O(1) lookup
    if (str == "Audio Input" || str == "AUDIO_INPUT") return DeviceCapability::AUDIO_INPUT;
    if (str == "Audio Output" || str == "AUDIO_OUTPUT") return DeviceCapability::AUDIO_OUTPUT;
    if (str == "Multi-channel" || str == "MULTICHANNEL") return DeviceCapability::MULTICHANNEL;
    if (str == "Low Latency" || str == "LOW_LATENCY") return DeviceCapability::LOW_LATENCY;
    if (str == "Network Streaming" || str == "NETWORK_STREAMING") return DeviceCapability::NETWORK_STREAMING;
    // ... add more mappings as needed
    return DeviceCapability::AUDIO_INPUT; // Default
}

ConnectionType stringToConnectionType(const std::string& str) {
    if (str == "Wired Ethernet" || str == "WIRED_ETHERNET") return ConnectionType::WIRED_ETHERNET;
    if (str == "Wireless WiFi" || str == "WIRELESS_WIFI") return ConnectionType::WIRELESS_WIFI;
    if (str == "USB") return ConnectionType::USB;
    // ... add more mappings as needed
    return ConnectionType::WIRED_ETHERNET; // Default
}

DiscoveryProtocol stringToDiscoveryProtocol(const std::string& str) {
    if (str == "mDNS" || str == "MDNS") return DiscoveryProtocol::MDNS;
    if (str == "UPnP") return DiscoveryProtocol::UPnP;
    if (str == "WS-Discovery" || str == "WS_DISCOVERY") return DiscoveryProtocol::WS_DISCOVERY;
    if (str == "DHT") return DiscoveryProtocol::DHT;
    if (str == "Custom") return DiscoveryProtocol::CUSTOM;
    return DiscoveryProtocol::MDNS; // Default
}

DeviceStatus stringToDeviceStatus(const std::string& str) {
    if (str == "Online" || str == "ONLINE") return DeviceStatus::ONLINE;
    if (str == "Offline" || str == "OFFLINE") return DeviceStatus::OFFLINE;
    if (str == "Busy" || str == "BUSY") return DeviceStatus::BUSY;
    if (str == "Error" || str == "ERROR") return DeviceStatus::ERROR;
    if (str == "Maintenance" || str == "MAINTENANCE") return DeviceStatus::MAINTENANCE;
    if (str == "Unknown" || str == "UNKNOWN") return DeviceStatus::UNKNOWN;
    return DeviceStatus::UNKNOWN; // Default
}

std::vector<std::string> parseSampleRates(uint32_t rate_mask) {
    std::vector<std::string> rates;

    // Common sample rates and their bit positions
    std::vector<std::pair<uint32_t, std::string>> common_rates = {
        {1 << 0, "8000"},
        {1 << 1, "11025"},
        {1 << 2, "16000"},
        {1 << 3, "22050"},
        {1 << 4, "32000"},
        {1 << 5, "44100"},
        {1 << 6, "48000"},
        {1 << 7, "88200"},
        {1 << 8, "96000"},
        {1 << 9, "176400"},
        {1 << 10, "192000"},
        {1 << 11, "352800"},
        {1 << 12, "384000"},
        {1 << 13, "705600"},
        {1 << 14, "768000"},
        {1 << 15, "1411200"},
        {1 << 16, "1536000"},
        {1 << 17, "2822400"},
        {1 << 18, "3072000"},
        {1 << 19, "5644800"},
        {1 << 20, "6144000"},
        {1 << 21, "11289600"},
        {1 << 22, "12288000"}
    };

    for (const auto& rate_pair : common_rates) {
        if (rate_mask & rate_pair.first) {
            rates.push_back(rate_pair.second);
        }
    }

    return rates;
}

std::vector<uint16_t> parseBitDepths(uint32_t depth_mask) {
    std::vector<uint16_t> depths;

    // Common bit depths and their bit positions
    if (depth_mask & (1 << 0)) depths.push_back(8);
    if (depth_mask & (1 << 1)) depths.push_back(16);
    if (depth_mask & (1 << 2)) depths.push_back(20);
    if (depth_mask & (1 << 3)) depths.push_back(24);
    if (depth_mask & (1 << 4)) depths.push_back(32);

    return depths;
}

bool isCompatibleProtocol(DiscoveryProtocol protocol, ConnectionType connection) {
    switch (protocol) {
        case DiscoveryProtocol::MDNS:
        case DiscoveryProtocol::UPnP:
            return connection == ConnectionType::WIRED_ETHERNET ||
                   connection == ConnectionType::WIRELESS_WIFI;

        case DiscoveryProtocol::WS_DISCOVERY:
            return connection == ConnectionType::WIRED_ETHERNET;

        case DiscoveryProtocol::DHT:
            return connection == ConnectionType::WIRED_ETHERNET ||
                   connection == ConnectionType::WIRELESS_WIFI;

        case DiscoveryProtocol::CUSTOM:
            return true; // Custom protocols can work with any connection

        default:
            return false;
    }
}

double calculateNetworkQuality(const NetworkStatistics& stats) {
    double quality = 100.0;

    // Penalize high packet loss
    if (stats.packet_loss_percent > 0) {
        quality -= stats.packet_loss_percent * 2.0;
    }

    // Penalize high latency
    if (stats.average_latency_ms > 20) {
        quality -= (stats.average_latency_ms - 20) * 0.5;
    }

    // Penalize failed connections
    if (stats.failed_connections > 0) {
        double failure_rate = static_cast<double>(stats.failed_connections) /
                             (stats.active_connections + stats.failed_connections);
        quality -= failure_rate * 50.0;
    }

    return std::max(0.0, std::min(100.0, quality));
}

std::string generateDeviceFingerprint(const AudioDeviceInfo& device) {
    std::ostringstream oss;
    oss << device.manufacturer << ":" << device.model << ":" << device.serial_number;
    return oss.str();
}

} // namespace DiscoveryUtils

} // namespace Network
} // namespace VortexGPU