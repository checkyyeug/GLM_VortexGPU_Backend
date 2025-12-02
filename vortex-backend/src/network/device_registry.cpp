#include "network/device_registry.hpp"
#include <algorithm>
#include <random>
#include <regex>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <fstream>

namespace VortexGPU {
namespace Network {

// ============================================================================
// NetworkAudioDeviceRegistry Implementation
// ============================================================================

NetworkAudioDeviceRegistry::NetworkAudioDeviceRegistry() {
    registry_start_time_ = std::chrono::system_clock::now();
}

NetworkAudioDeviceRegistry::~NetworkAudioDeviceRegistry() {
    shutdown();
}

bool NetworkAudioDeviceRegistry::initialize(const RegistryConfig& config) {
    if (initialized_.load()) {
        return true;
    }

    config_ = config;

    try {
        // Create storage directory if it doesn't exist
        std::filesystem::create_directories(config_.storage_directory);
        std::filesystem::create_directories(config_.backup_directory);

        // Initialize components
        storage_ = std::make_unique<RegistryStorage>();
        cache_ = std::make_unique<RegistryCache>(config_.cache_size_limit, config_.cache_ttl);
        health_monitor_ = std::make_unique<RegistryHealthMonitor>(config_);

        // Initialize storage
        if (!storage_->initialize(config_)) {
            std::cerr << "Failed to initialize registry storage" << std::endl;
            return false;
        }

        // Load existing data
        if (!loadFromStorage()) {
            std::cerr << "Failed to load registry data from storage" << std::endl;
        }

        running_.store(true);

        // Start background threads
        maintenance_thread_ = std::thread(&NetworkAudioDeviceRegistry::maintenanceThread, this);
        health_monitor_thread_ = std::thread(&NetworkAudioDeviceRegistry::healthMonitorThread, this);
        event_processing_thread_ = std::thread(&NetworkAudioDeviceRegistry::eventProcessingThread, this);

        // Start scheduled backups if enabled
        if (config_.enable_backups) {
            scheduleBackup(config_.backup_interval);
        }

        initialized_.store(true);

        std::cout << "NetworkAudioDeviceRegistry initialized with "
                  << device_registry_.size() << " devices" << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize NetworkAudioDeviceRegistry: " << e.what() << std::endl;
        shutdown();
        return false;
    }
}

void NetworkAudioDeviceRegistry::shutdown() {
    if (!initialized_.load()) {
        return;
    }

    running_.store(false);
    scheduled_backup_enabled_.store(false);

    // Wait for threads to finish
    if (maintenance_thread_.joinable()) {
        maintenance_thread_.join();
    }

    if (health_monitor_thread_.joinable()) {
        health_monitor_thread_.join();
    }

    if (backup_thread_.joinable()) {
        backup_thread_.join();
    }

    if (event_processing_thread_.joinable()) {
        event_processing_thread_.join();
    }

    // Save final state
    if (config_.enable_persistence) {
        saveToStorage();
    }

    // Cleanup components
    device_registry_.clear();
    configuration_templates_.clear();
    class_index_.clear();
    priority_index_.clear();
    health_index_.clear();
    group_index_.clear();
    tag_index_.clear();

    if (health_monitor_) {
        health_monitor_->stop();
        health_monitor_.reset();
    }

    if (storage_) {
        storage_->shutdown();
        storage_.reset();
    }

    if (cache_) {
        cache_->clear();
        cache_.reset();
    }

    initialized_.store(false);

    std::cout << "NetworkAudioDeviceRegistry shut down" << std::endl;
}

void NetworkAudioDeviceRegistry::reload() {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    loadFromStorage();
}

void NetworkAudioDeviceRegistry::save() {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    saveToStorage();
}

bool NetworkAudioDeviceRegistry::registerDevice(const RegistryDeviceInfo& device) {
    if (!validateDevice(device)) {
        return false;
    }

    std::string device_id = generateDeviceId(device);
    RegistryDeviceInfo device_to_register = device;
    device_to_register.device_id = device_id;
    device_to_register.registration_status = DeviceRegistrationStatus::REGISTERED;
    device_to_register.registration_time = std::chrono::system_clock::now();
    device_to_register.registration_token = generateRegistrationToken();
    device_to_register.registration_attempts = 1;
    device_to_register.last_registration_attempt = device_to_register.registration_time;

    std::lock_guard<std::mutex> lock(registry_mutex_);

    // Check if device already exists
    auto it = device_registry_.find(device_id);
    bool is_new_device = (it == device_registry_.end());

    if (is_new_device) {
        device_registry_[device_id] = device_to_register;
        addToIndexes(device_to_register);

        // Save to storage
        if (storage_) {
            storage_->saveDevice(device_to_register);
        }

        // Publish event
        RegistryEvent event;
        event.type = RegistryEventType::DEVICE_REGISTERED;
        event.device_id = device_id;
        event.device_name = device_to_register.device_name;
        event.message = "Device registered: " + device_to_register.device_name;
        event.timestamp = std::chrono::system_clock::now();
        event.source = "DeviceRegistry";
        event.severity = "info";
        recordEvent(event);

        std::cout << "Registered device: " << device_to_register.device_name << " (ID: " << device_id << ")" << std::endl;
    } else {
        // Update existing device
        updateDevice(device_id, device_to_register);
    }

    // Update cache
    if (cache_) {
        cache_->put(device_id, device_to_register);
    }

    updateStatistics();
    return true;
}

bool NetworkAudioDeviceRegistry::unregisterDevice(const std::string& device_id) {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto it = device_registry_.find(device_id);
    if (it == device_registry_.end()) {
        return false;
    }

    const RegistryDeviceInfo& device = it->second;

    // Remove from indexes
    removeFromIndexes(device);

    // Remove from storage
    if (storage_) {
        storage_->deleteDevice(device_id);
    }

    // Remove from cache
    if (cache_) {
        cache_->invalidate(device_id);
    }

    // Remove from health monitor
    if (health_monitor_) {
        health_monitor_->removeDevice(device_id);
    }

    // Record event
    RegistryEvent event;
    event.type = RegistryEventType::DEVICE_UNREGISTERED;
    event.device_id = device_id;
    event.device_name = device.device_name;
    event.message = "Device unregistered: " + device.device_name;
    event.timestamp = std::chrono::system_clock::now();
    event.source = "DeviceRegistry";
    event.severity = "info";
    recordEvent(event);

    std::cout << "Unregistered device: " << device.device_name << " (ID: " << device_id << ")" << std::endl;

    device_registry_.erase(it);
    updateStatistics();
    return true;
}

bool NetworkAudioDeviceRegistry::updateDevice(const std::string& device_id, const RegistryDeviceInfo& updated_info) {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto it = device_registry_.find(device_id);
    if (it == device_registry_.end()) {
        return false;
    }

    RegistryDeviceInfo& existing_device = it->second;
    RegistryDeviceInfo old_device = existing_device;

    // Update device information
    existing_device = updated_info;
    existing_device.device_id = device_id;
    existing_device.last_registration_attempt = std::chrono::system_clock::now();
    existing_device.registration_attempts++;

    // Update indexes
    updateIndexes(old_device, existing_device);

    // Save to storage
    if (storage_) {
        storage_->saveDevice(existing_device);
    }

    // Update cache
    if (cache_) {
        cache_->put(device_id, existing_device);
    }

    // Record event
    RegistryEvent event;
    event.type = RegistryEventType::DEVICE_UPDATED;
    event.device_id = device_id;
    event.device_name = existing_device.device_name;
    event.message = "Device updated: " + existing_device.device_name;
    event.timestamp = std::chrono::system_clock::now();
    event.source = "DeviceRegistry";
    event.severity = "info";
    recordEvent(event);

    updateStatistics();
    return true;
}

RegistryDeviceInfo NetworkAudioDeviceRegistry::getDevice(const std::string& device_id) const {
    // Try cache first
    if (cache_) {
        RegistryDeviceInfo device;
        if (cache_->get(device_id, device)) {
            return device;
        }
    }

    // Try registry
    std::lock_guard<std::mutex> lock(registry_mutex_);
    auto it = device_registry_.find(device_id);
    if (it != device_registry_.end()) {
        // Update cache
        if (cache_) {
            cache_->put(device_id, it->second);
        }
        return it->second;
    }

    return RegistryDeviceInfo{};
}

bool NetworkAudioDeviceRegistry::isDeviceRegistered(const std::string& device_id) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    return device_registry_.find(device_id) != device_registry_.end();
}

std::vector<std::string> NetworkAudioDeviceRegistry::registerDevices(const std::vector<RegistryDeviceInfo>& devices) {
    std::vector<std::string> registered_ids;

    for (const auto& device : devices) {
        if (registerDevice(device)) {
            registered_ids.push_back(device.device_id);
        }
    }

    return registered_ids;
}

bool NetworkAudioDeviceRegistry::unregisterDevices(const std::vector<std::string>& device_ids) {
    bool all_success = true;

    for (const auto& device_id : device_ids) {
        if (!unregisterDevice(device_id)) {
            all_success = false;
        }
    }

    return all_success;
}

std::vector<RegistryDeviceInfo> NetworkAudioDeviceRegistry::updateDevices(const std::vector<RegistryDeviceInfo>& devices) {
    std::vector<RegistryDeviceInfo> updated_devices;

    for (const auto& device : devices) {
        if (updateDevice(device.device_id, device)) {
            updated_devices.push_back(device);
        }
    }

    return updated_devices;
}

std::vector<RegistryDeviceInfo> NetworkAudioDeviceRegistry::queryDevices(const DeviceQuery& query) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    std::vector<RegistryDeviceInfo> results;

    for (const auto& pair : device_registry_) {
        const RegistryDeviceInfo& device = pair.second;

        if (matchesQuery(device, query)) {
            results.push_back(device);
        }
    }

    // Apply sorting
    results = sortDevices(results, query.sort_by, query.sort_ascending);

    // Apply pagination
    if (query.offset >= results.size()) {
        return {};
    }

    size_t end_index = std::min(query.offset + query.limit, results.size());
    return std::vector<RegistryDeviceInfo>(results.begin() + query.offset, results.begin() + end_index);
}

std::vector<RegistryDeviceInfo> NetworkAudioDeviceRegistry::getAllDevices() const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    std::vector<RegistryDeviceInfo> devices;
    devices.reserve(device_registry_.size());

    for (const auto& pair : device_registry_) {
        devices.push_back(pair.second);
    }

    return devices;
}

std::vector<RegistryDeviceInfo> NetworkAudioDeviceRegistry::getDevicesByClass(DeviceClass device_class) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    std::vector<RegistryDeviceInfo> devices;
    auto it = class_index_.find(device_class);

    if (it != class_index_.end()) {
        devices.reserve(it->second.size());

        for (const auto& device_id : it->second) {
            auto device_it = device_registry_.find(device_id);
            if (device_it != device_registry_.end()) {
                devices.push_back(device_it->second);
            }
        }
    }

    return devices;
}

std::vector<RegistryDeviceInfo> NetworkAudioDeviceRegistry::getDevicesByPriority(DevicePriority priority) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    std::vector<RegistryDeviceInfo> devices;
    auto it = priority_index_.find(priority);

    if (it != priority_index_.end()) {
        devices.reserve(it->second.size());

        for (const auto& device_id : it->second) {
            auto device_it = device_registry_.find(device_id);
            if (device_it != device_registry_.end()) {
                devices.push_back(device_it->second);
            }
        }
    }

    return devices;
}

std::vector<RegistryDeviceInfo> NetworkAudioDeviceRegistry::getDevicesByHealth(DeviceHealth health) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    std::vector<RegistryDeviceInfo> devices;
    auto it = health_index_.find(health);

    if (it != health_index_.end()) {
        devices.reserve(it->second.size());

        for (const auto& device_id : it->second) {
            auto device_it = device_registry_.find(device_id);
            if (device_it != device_registry_.end()) {
                devices.push_back(device_it->second);
            }
        }
    }

    return devices;
}

std::vector<RegistryDeviceInfo> NetworkAudioDeviceRegistry::getAvailableDevices() const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    std::vector<RegistryDeviceInfo> devices;

    for (const auto& pair : device_registry_) {
        const RegistryDeviceInfo& device = pair.second;
        if (device.isAvailable()) {
            devices.push_back(device);
        }
    }

    return devices;
}

std::vector<RegistryDeviceInfo> NetworkAudioDeviceRegistry::getHealthyDevices() const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    std::vector<RegistryDeviceInfo> devices;

    for (const auto& pair : device_registry_) {
        const RegistryDeviceInfo& device = pair.second;
        if (device.isHealthy() && device.status == DeviceStatus::ONLINE) {
            devices.push_back(device);
        }
    }

    return devices;
}

std::vector<RegistryDeviceInfo> NetworkAudioDeviceRegistry::getDevicesByGroup(const std::string& group) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    std::vector<RegistryDeviceInfo> devices;
    auto it = group_index_.find(group);

    if (it != group_index_.end()) {
        devices.reserve(it->second.size());

        for (const auto& device_id : it->second) {
            auto device_it = device_registry_.find(device_id);
            if (device_it != device_registry_.end()) {
                devices.push_back(device_it->second);
            }
        }
    }

    return devices;
}

std::vector<RegistryDeviceInfo> NetworkAudioDeviceRegistry::searchDevices(const std::string& query) const {
    DeviceQuery device_query;
    device_query.search_query = query;
    return queryDevices(device_query);
}

bool NetworkAudioDeviceRegistry::addDeviceToGroup(const std::string& device_id, const std::string& group) {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto it = device_registry_.find(device_id);
    if (it == device_registry_.end()) {
        return false;
    }

    RegistryDeviceInfo& device = it->second;

    // Add to device's groups if not already present
    if (std::find(device.groups.begin(), device.groups.end(), group) == device.groups.end()) {
        device.groups.push_back(group);

        // Update group index
        group_index_[group].insert(device_id);

        // Update storage
        if (storage_) {
            storage_->saveDevice(device);
        }

        // Update cache
        if (cache_) {
            cache_->put(device_id, device);
        }

        // Record event
        RegistryEvent event;
        event.type = RegistryEventType::DEVICE_GROUP_CHANGED;
        event.device_id = device_id;
        event.device_name = device.device_name;
        event.message = "Device added to group: " + group;
        event.timestamp = std::chrono::system_clock::now();
        event.source = "DeviceRegistry";
        event.severity = "info";
        recordEvent(event);
    }

    return true;
}

bool NetworkAudioDeviceRegistry::removeDeviceFromGroup(const std::string& device_id, const std::string& group) {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto it = device_registry_.find(device_id);
    if (it == device_registry_.end()) {
        return false;
    }

    RegistryDeviceInfo& device = it->second;

    // Remove from device's groups
    auto group_it = std::find(device.groups.begin(), device.groups.end(), group);
    if (group_it != device.groups.end()) {
        device.groups.erase(group_it);

        // Update group index
        auto group_index_it = group_index_.find(group);
        if (group_index_it != group_index_.end()) {
            group_index_it->second.erase(device_id);
            if (group_index_it->second.empty()) {
                group_index_.erase(group_index_it);
            }
        }

        // Update storage
        if (storage_) {
            storage_->saveDevice(device);
        }

        // Update cache
        if (cache_) {
            cache_->put(device_id, device);
        }

        // Record event
        RegistryEvent event;
        event.type = RegistryEventType::DEVICE_GROUP_CHANGED;
        event.device_id = device_id;
        event.device_name = device.device_name;
        event.message = "Device removed from group: " + group;
        event.timestamp = std::chrono::system_clock::now();
        event.source = "DeviceRegistry";
        event.severity = "info";
        recordEvent(event);
    }

    return true;
}

bool NetworkAudioDeviceRegistry::addDeviceTag(const std::string& device_id, const std::string& tag) {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto it = device_registry_.find(device_id);
    if (it == device_registry_.end()) {
        return false;
    }

    RegistryDeviceInfo& device = it->second;

    // Add tag if not already present
    if (std::find(device.tags.begin(), device.tags.end(), tag) == device.tags.end()) {
        device.tags.push_back(tag);

        // Update tag index
        tag_index_[tag].insert(device_id);

        // Update storage
        if (storage_) {
            storage_->saveDevice(device);
        }

        // Update cache
        if (cache_) {
            cache_->put(device_id, device);
        }
    }

    return true;
}

bool NetworkAudioDeviceRegistry::removeDeviceTag(const std::string& device_id, const std::string& tag) {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto it = device_registry_.find(device_id);
    if (it == device_registry_.end()) {
        return false;
    }

    RegistryDeviceInfo& device = it->second;

    // Remove tag
    auto tag_it = std::find(device.tags.begin(), device.tags.end(), tag);
    if (tag_it != device.tags.end()) {
        device.tags.erase(tag_it);

        // Update tag index
        auto tag_index_it = tag_index_.find(tag);
        if (tag_index_it != tag_index_.end()) {
            tag_index_it->second.erase(device_id);
            if (tag_index_it->second.empty()) {
                tag_index_.erase(tag_index_it);
            }
        }

        // Update storage
        if (storage_) {
            storage_->saveDevice(device);
        }

        // Update cache
        if (cache_) {
            cache_->put(device_id, device);
        }
    }

    return true;
}

std::vector<std::string> NetworkAudioDeviceRegistry::getDeviceGroups(const std::string& device_id) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto it = device_registry_.find(device_id);
    if (it != device_registry_.end()) {
        return it->second.groups;
    }

    return {};
}

std::vector<std::string> NetworkAudioDeviceRegistry::getDeviceTags(const std::string& device_id) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto it = device_registry_.find(device_id);
    if (it != device_registry_.end()) {
        return it->second.tags;
    }

    return {};
}

std::vector<std::string> NetworkAudioDeviceRegistry::getAllGroups() const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    std::vector<std::string> groups;
    groups.reserve(group_index_.size());

    for (const auto& pair : group_index_) {
        groups.push_back(pair.first);
    }

    return groups;
}

std::vector<std::string> NetworkAudioDeviceRegistry::getAllTags() const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    std::vector<std::string> tags;
    tags.reserve(tag_index_.size());

    for (const auto& pair : tag_index_) {
        tags.push_back(pair.first);
    }

    return tags;
}

bool NetworkAudioDeviceRegistry::setDevicePriority(const std::string& device_id, DevicePriority priority) {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto it = device_registry_.find(device_id);
    if (it == device_registry_.end()) {
        return false;
    }

    RegistryDeviceInfo& device = it->second;
    DevicePriority old_priority = device.priority;
    device.priority = priority;

    // Update priority index
    if (old_priority != priority) {
        auto old_index_it = priority_index_.find(old_priority);
        if (old_index_it != priority_index_.end()) {
            old_index_it->second.erase(device_id);
            if (old_index_it->second.empty()) {
                priority_index_.erase(old_index_it);
            }
        }

        priority_index_[priority].insert(device_id);
    }

    // Update storage
    if (storage_) {
        storage_->saveDevice(device);
    }

    // Update cache
    if (cache_) {
        cache_->put(device_id, device);
    }

    // Record event
    RegistryEvent event;
    event.type = RegistryEventType::DEVICE_PRIORITY_CHANGED;
    event.device_id = device_id;
    event.device_name = device.device_name;
    event.message = "Device priority changed: " + priorityToString(old_priority) + " -> " + priorityToString(priority);
    event.timestamp = std::chrono::system_clock::now();
    event.source = "DeviceRegistry";
    event.severity = "info";
    recordEvent(event);

    return true;
}

DevicePriority NetworkAudioDeviceRegistry::getDevicePriority(const std::string& device_id) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto it = device_registry_.find(device_id);
    if (it != device_registry_.end()) {
        return it->second.priority;
    }

    return DevicePriority::NORMAL;
}

std::vector<RegistryDeviceInfo> NetworkAudioDeviceRegistry::getDevicesByPriorityOrder() const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    std::vector<RegistryDeviceInfo> devices;
    devices.reserve(device_registry_.size());

    for (const auto& pair : device_registry_) {
        devices.push_back(pair.second);
    }

    std::sort(devices.begin(), devices.end(),
              [](const RegistryDeviceInfo& a, const RegistryDeviceInfo& b) {
                  return a.priority < b.priority; // Lower enum value = higher priority
              });

    return devices;
}

void NetworkAudioDeviceRegistry::updateDeviceHealth(const std::string& device_id, DeviceHealth health, const std::string& message) {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto it = device_registry_.find(device_id);
    if (it == device_registry_.end()) {
        return;
    }

    RegistryDeviceInfo& device = it->second;
    DeviceHealth old_health = device.health_status;
    device.health_status = health;

    // Update health index
    if (old_health != health) {
        auto old_index_it = health_index_.find(old_health);
        if (old_index_it != health_index_.end()) {
            old_index_it->second.erase(device_id);
            if (old_index_it->second.empty()) {
                health_index_.erase(old_index_it);
            }
        }

        health_index_[health].insert(device_id);

        // Record event
        RegistryEvent event;
        event.type = RegistryEventType::DEVICE_HEALTH_CHANGED;
        event.device_id = device_id;
        event.device_name = device.device_name;
        event.message = "Device health changed: " + healthToString(old_health) + " -> " + healthToString(health);
        if (!message.empty()) {
            event.message += " (" + message + ")";
        }
        event.timestamp = std::chrono::system_clock::now();
        event.source = "DeviceRegistry";
        event.severity = (health == DeviceHealth::HEALTHY) ? "info" : "warning";
        recordEvent(event);
    }

    // Update storage
    if (storage_) {
        storage_->saveDevice(device);
    }

    // Update cache
    if (cache_) {
        cache_->put(device_id, device);
    }

    // Update health monitor
    if (health_monitor_) {
        health_monitor_->updateDeviceMetrics(device_id, device.cpu_utilization_percent,
                                            device.memory_utilization_percent,
                                            device.average_response_time_ms, true);
    }
}

void NetworkAudioDeviceRegistry::recordDeviceError(const std::string& device_id, const std::string& error) {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto it = device_registry_.find(device_id);
    if (it == device_registry_.end()) {
        return;
    }

    RegistryDeviceInfo& device = it->second;
    device.error_count++;
    device.last_error_time = std::chrono::system_clock::now();

    // Add to recent errors (keep last 10)
    device.recent_errors.push_back(error);
    if (device.recent_errors.size() > 10) {
        device.recent_errors.erase(device.recent_errors.begin());
    }

    // Update health status if error count is high
    if (device.error_count > 5 && device.health_status != DeviceHealth::CRITICAL) {
        updateDeviceHealth(device_id, DeviceHealth::DEGRADED, "High error rate: " + std::to_string(device.error_count));
    }

    // Update health monitor
    if (health_monitor_) {
        health_monitor_->recordDeviceError(device_id, error);
    }

    // Record event
    RegistryEvent event;
    event.type = RegistryEventType::ERROR_OCCURRED;
    event.device_id = device_id;
    event.device_name = device.device_name;
    event.message = "Device error: " + error;
    event.timestamp = std::chrono::system_clock::now();
    event.source = "DeviceRegistry";
    event.severity = "error";
    recordEvent(event);
}

void NetworkAudioDeviceRegistry::updateDeviceMetrics(const std::string& device_id, double cpu_util, double memory_util,
                                                    double response_time, bool request_successful) {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto it = device_registry_.find(device_id);
    if (it == device_registry_.end()) {
        return;
    }

    RegistryDeviceInfo& device = it->second;
    device.cpu_utilization_percent = cpu_util;
    device.memory_utilization_percent = memory_util;
    device.average_response_time_ms = response_time;
    device.total_requests++;

    if (request_successful) {
        device.successful_requests++;
        device.last_successful_request = std::chrono::system_clock::now();
    } else {
        device.failed_requests++;
        device.last_failed_request = std::chrono::system_clock::now();
    }

    // Update health monitor
    if (health_monitor_) {
        health_monitor_->updateDeviceMetrics(device_id, cpu_util, memory_util, response_time, request_successful);
    }
}

std::vector<RegistryDeviceInfo> NetworkAudioDeviceRegistry::getUnhealthyDevices() const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    std::vector<RegistryDeviceInfo> devices;

    for (const auto& pair : device_registry_) {
        const RegistryDeviceInfo& device = pair.second;
        if (!device.isHealthy() || device.status != DeviceStatus::ONLINE) {
            devices.push_back(device);
        }
    }

    return devices;
}

std::vector<RegistryDeviceInfo> NetworkAudioDeviceRegistry::getDevicesNeedingAttention() const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    std::vector<RegistryDeviceInfo> devices;

    for (const auto& pair : device_registry_) {
        const RegistryDeviceInfo& device = pair.second;

        // Check if device needs attention
        bool needs_attention = false;

        if (device.health_status == DeviceHealth::WARNING ||
            device.health_status == DeviceHealth::DEGRADED ||
            device.health_status == DeviceHealth::CRITICAL) {
            needs_attention = true;
        }

        if (device.error_count > 3) {
            needs_attention = true;
        }

        if (device.getSuccessRate() < 90.0) {
            needs_attention = true;
        }

        if (device.cpu_utilization_percent > 90.0 || device.memory_utilization_percent > 90.0) {
            needs_attention = true;
        }

        if (needs_attention) {
            devices.push_back(device);
        }
    }

    return devices;
}

bool NetworkAudioDeviceRegistry::createConfigurationTemplate(const DeviceConfigurationTemplate& template_info) {
    if (!validateConfigurationTemplate(template_info)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(registry_mutex_);

    DeviceConfigurationTemplate new_template = template_info;
    new_template.created_at = std::chrono::system_clock::now();
    new_template.last_updated = new_template.created_at;

    configuration_templates_[new_template.template_id] = new_template;

    // Save to storage
    if (storage_) {
        storage_->saveTemplate(new_template);
    }

    // Record event
    RegistryEvent event;
    event.type = RegistryEventType::CONFIGURATION_TEMPLATE_CREATED;
    event.message = "Configuration template created: " + new_template.template_name;
    event.timestamp = std::chrono::system_clock::now();
    event.source = "DeviceRegistry";
    event.severity = "info";
    recordEvent(event);

    return true;
}

bool NetworkAudioDeviceRegistry::updateConfigurationTemplate(const std::string& template_id, const DeviceConfigurationTemplate& template_info) {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto it = configuration_templates_.find(template_id);
    if (it == configuration_templates_.end()) {
        return false;
    }

    DeviceConfigurationTemplate& existing_template = it->second;
    existing_template = template_info;
    existing_template.template_id = template_id;
    existing_template.last_updated = std::chrono::system_clock::now();

    // Save to storage
    if (storage_) {
        storage_->saveTemplate(existing_template);
    }

    // Record event
    RegistryEvent event;
    event.type = RegistryEventType::CONFIGURATION_TEMPLATE_UPDATED;
    event.message = "Configuration template updated: " + existing_template.template_name;
    event.timestamp = std::chrono::system_clock::now();
    event.source = "DeviceRegistry";
    event.severity = "info";
    recordEvent(event);

    return true;
}

bool NetworkAudioDeviceRegistry::deleteConfigurationTemplate(const std::string& template_id) {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto it = configuration_templates_.find(template_id);
    if (it == configuration_templates_.end()) {
        return false;
    }

    std::string template_name = it->second.template_name;
    configuration_templates_.erase(it);

    // Delete from storage
    if (storage_) {
        storage_->deleteTemplate(template_id);
    }

    // Record event
    RegistryEvent event;
    event.type = RegistryEventType::CONFIGURATION_TEMPLATE_DELETED;
    event.message = "Configuration template deleted: " + template_name;
    event.timestamp = std::chrono::system_clock::now();
    event.source = "DeviceRegistry";
    event.severity = "info";
    recordEvent(event);

    return true;
}

std::vector<DeviceConfigurationTemplate> NetworkAudioDeviceRegistry::getConfigurationTemplates() const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    std::vector<DeviceConfigurationTemplate> templates;
    templates.reserve(configuration_templates_.size());

    for (const auto& pair : configuration_templates_) {
        templates.push_back(pair.second);
    }

    return templates;
}

DeviceConfigurationTemplate NetworkAudioDeviceRegistry::getConfigurationTemplate(const std::string& template_id) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto it = configuration_templates_.find(template_id);
    if (it != configuration_templates_.end()) {
        return it->second;
    }

    return DeviceConfigurationTemplate{};
}

bool NetworkAudioDeviceRegistry::applyConfigurationTemplate(const std::string& device_id, const std::string& template_id) {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto device_it = device_registry_.find(device_id);
    auto template_it = configuration_templates_.find(template_id);

    if (device_it == device_registry_.end() || template_it == configuration_templates_.end()) {
        return false;
    }

    const DeviceConfigurationTemplate& template_info = template_it->second;
    RegistryDeviceInfo& device = device_it->second;

    // Apply template configuration
    device.sample_rate = template_info.default_sample_rate;
    device.buffer_size = template_info.default_buffer_size;
    device.channels = template_info.default_channels;
    device.min_latency_ms = template_info.max_acceptable_latency_ms / 2.0;
    device.configuration_id = template_info.template_id;
    device.configuration_version = template_info.template_version;
    device.configuration_source = "template";
    device.last_configuration_update = std::chrono::system_clock::now();
    device.configuration_dirty = false;

    // Apply custom properties
    for (const auto& prop : template_info.custom_properties) {
        device.configuration_properties[prop.first] = prop.second;
    }

    // Save to storage
    if (storage_) {
        storage_->saveDevice(device);
    }

    // Update cache
    if (cache_) {
        cache_->put(device_id, device);
    }

    // Record event
    RegistryEvent event;
    event.type = RegistryEventType::DEVICE_CONFIG_CHANGED;
    event.device_id = device_id;
    event.device_name = device.device_name;
    event.message = "Applied configuration template: " + template_info.template_name;
    event.timestamp = std::chrono::system_clock::now();
    event.source = "DeviceRegistry";
    event.severity = "info";
    recordEvent(event);

    return true;
}

std::string NetworkAudioDeviceRegistry::getBestMatchingTemplate(const std::string& device_id) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto device_it = device_registry_.find(device_id);
    if (device_it == device_registry_.end()) {
        return "";
    }

    const RegistryDeviceInfo& device = device_it->second;
    std::string best_template_id;
    int best_score = -1;

    for (const auto& pair : configuration_templates_) {
        const DeviceConfigurationTemplate& template_info = pair.second;
        int score = 0;

        // Check device class match
        if (!template_info.device_class.empty() &&
            deviceClassToString(device.device_class).find(template_info.device_class) != std::string::npos) {
            score += 10;
        }

        // Check manufacturer match
        if (!template_info.manufacturer_pattern.empty()) {
            std::regex manufacturer_regex(template_info.manufacturer_pattern, std::regex_constants::icase);
            if (std::regex_search(device.manufacturer, manufacturer_regex)) {
                score += 8;
            }
        }

        // Check model match
        if (!template_info.model_pattern.empty()) {
            std::regex model_regex(template_info.model_pattern, std::regex_constants::icase);
            if (std::regex_search(device.model, model_regex)) {
                score += 6;
            }
        }

        // Check capabilities
        uint64_t device_capabilities_mask = 0;
        for (auto capability : device.capabilities) {
            device_capabilities_mask |= (1ULL << static_cast<int>(capability));
        }

        if ((device_capabilities_mask & template_info.required_capabilities) == template_info.required_capabilities) {
            score += 5;
        }

        if (score > best_score) {
            best_score = score;
            best_template_id = template_info.template_id;
        }
    }

    return best_template_id;
}

bool NetworkAudioDeviceRegistry::createBackup(const std::string& backup_name) {
    std::string backup_path;

    if (backup_name.empty()) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::ostringstream oss;
        oss << "registry_backup_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
        backup_path = config_.backup_directory + "/" + oss.str() + ".json";
    } else {
        backup_path = config_.backup_directory + "/" + backup_name + ".json";
    }

    bool success = createBackupFile(backup_path);

    if (success) {
        RegistryEvent event;
        event.type = RegistryEventType::REGISTRY_BACKUP_CREATED;
        event.message = "Registry backup created: " + backup_path;
        event.timestamp = std::chrono::system_clock::now();
        event.source = "DeviceRegistry";
        event.severity = "info";
        recordEvent(event);
    }

    return success;
}

bool NetworkAudioDeviceRegistry::restoreFromBackup(const std::string& backup_name) {
    std::string backup_path = config_.backup_directory + "/" + backup_name + ".json";

    if (!std::filesystem::exists(backup_path)) {
        return false;
    }

    // Clear current registry
    std::lock_guard<std::mutex> lock(registry_mutex_);
    device_registry_.clear();
    class_index_.clear();
    priority_index_.clear();
    health_index_.clear();
    group_index_.clear();
    tag_index_.clear();

    bool success = restoreFromBackupFile(backup_path);

    if (success) {
        // Rebuild indexes
        for (const auto& pair : device_registry_) {
            addToIndexes(pair.second);
        }

        // Save to storage
        if (storage_) {
            for (const auto& pair : device_registry_) {
                storage_->saveDevice(pair.second);
            }
        }

        RegistryEvent event;
        event.type = RegistryEventType::REGISTRY_RESTORED;
        event.message = "Registry restored from backup: " + backup_name;
        event.timestamp = std::chrono::system_clock::now();
        event.source = "DeviceRegistry";
        event.severity = "info";
        recordEvent(event);
    }

    return success;
}

std::vector<std::string> NetworkAudioDeviceRegistry::getAvailableBackups() const {
    std::vector<std::string> backups;

    if (!std::filesystem::exists(config_.backup_directory)) {
        return backups;
    }

    for (const auto& entry : std::filesystem::directory_iterator(config_.backup_directory)) {
        if (entry.path().extension() == ".json") {
            std::string filename = entry.path().filename().string();
            // Remove .json extension
            backups.push_back(filename.substr(0, filename.length() - 5));
        }
    }

    std::sort(backups.rbegin(), backups.rend()); // Sort by date (newest first)
    return backups;
}

bool NetworkAudioDeviceRegistry::deleteBackup(const std::string& backup_name) {
    std::string backup_path = config_.backup_directory + "/" + backup_name + ".json";

    if (std::filesystem::exists(backup_path)) {
        return std::filesystem::remove(backup_path);
    }

    return false;
}

bool NetworkAudioDeviceRegistry::scheduleBackup(const std::chrono::seconds& interval) {
    backup_interval_ = interval;
    scheduled_backup_enabled_.store(true);

    if (backup_thread_.joinable()) {
        backup_thread_.join();
    }

    backup_thread_ = std::thread(&NetworkAudioDeviceRegistry::backupThread, this);

    return true;
}

void NetworkAudioDeviceRegistry::stopScheduledBackups() {
    scheduled_backup_enabled_.store(false);
    if (backup_thread_.joinable()) {
        backup_thread_.join();
    }
}

RegistryStatistics NetworkAudioDeviceRegistry::getStatistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return statistics_;
}

std::vector<RegistryEvent> NetworkAudioDeviceRegistry::getRecentEvents(size_t max_events) const {
    std::lock_guard<std::mutex> lock(events_mutex_);

    std::vector<RegistryEvent> events;
    size_t start_index = event_history_.size() > max_events ? event_history_.size() - max_events : 0;

    for (size_t i = start_index; i < event_history_.size(); ++i) {
        events.push_back(event_history_[i]);
    }

    return events;
}

std::string NetworkAudioDeviceRegistry::generateHealthReport() const {
    std::ostringstream oss;

    std::lock_guard<std::mutex> lock(registry_mutex_);
    RegistryStatistics stats = getStatistics();

    oss << "=== Device Registry Health Report ===\n";
    oss << "Generated at: " << formatTimestamp(std::chrono::system_clock::now()) << "\n\n";

    oss << "Summary Statistics:\n";
    oss << "  Total devices: " << stats.total_devices << "\n";
    oss << "  Registered devices: " << stats.registered_devices << "\n";
    oss << "  Online devices: " << stats.online_devices << "\n";
    oss << "  Offline devices: " << stats.offline_devices << "\n";
    oss << "  Healthy devices: " << stats.healthy_devices << "\n";
    oss << "  Unhealthy devices: " << stats.unhealthy_devices << "\n\n";

    oss << "Device Categories:\n";
    oss << "  Input devices: " << stats.input_devices << "\n";
    oss << "  Output devices: " << stats.output_devices << "\n";
    oss << "  Input/Output devices: " << stats.input_output_devices << "\n\n";

    oss << "Priority Distribution:\n";
    oss << "  Critical priority: " << stats.critical_priority_devices << "\n";
    oss << "  High priority: " << stats.high_priority_devices << "\n\n";

    oss << "Performance Metrics:\n";
    oss << "  Average response time: " << std::fixed << std::setprecision(2) << stats.average_response_time_ms << " ms\n";
    oss << "  Average CPU utilization: " << std::fixed << std::setprecision(1) << stats.average_cpu_utilization << "%\n";
    oss << "  Average memory utilization: " << std::fixed << std::setprecision(1) << stats.average_memory_utilization << "%\n";
    oss << "  Request success rate: " << std::fixed << std::setprecision(1)
        << ((double)stats.successful_requests / stats.total_requests * 100.0) << "%\n\n";

    // Unhealthy devices section
    auto unhealthy_devices = getUnhealthyDevices();
    if (!unhealthy_devices.empty()) {
        oss << "Unhealthy Devices (" << unhealthy_devices.size() << "):\n";
        for (const auto& device : unhealthy_devices) {
            oss << "  - " << device.device_name << " (" << device.device_id << ")\n";
            oss << "    Status: " << healthToString(device.health_status) << "\n";
            oss << "    Error count: " << device.error_count << "\n";
            oss << "    Success rate: " << std::fixed << std::setprecision(1) << device.getSuccessRate() << "%\n";
        }
        oss << "\n";
    }

    // Devices needing attention
    auto attention_devices = getDevicesNeedingAttention();
    if (!attention_devices.empty()) {
        oss << "Devices Needing Attention (" << attention_devices.size() << "):\n";
        for (const auto& device : attention_devices) {
            oss << "  - " << device.device_name << " (" << device.device_id << ")\n";
        }
        oss << "\n";
    }

    return oss.str();
}

std::string NetworkAudioDeviceRegistry::generateDeviceReport(const std::string& device_id) const {
    std::ostringstream oss;

    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto it = device_registry_.find(device_id);
    if (it == device_registry_.end()) {
        oss << "Device not found: " << device_id << "\n";
        return oss.str();
    }

    const RegistryDeviceInfo& device = it->second;

    oss << "=== Device Report ===\n";
    oss << "Generated at: " << formatTimestamp(std::chrono::system_clock::now()) << "\n\n";

    oss << "Basic Information:\n";
    oss << "  Device ID: " << device.device_id << "\n";
    oss << "  Name: " << device.device_name << "\n";
    oss << "  Manufacturer: " << device.manufacturer << "\n";
    oss << "  Model: " << device.model << "\n";
    oss << "  Serial Number: " << device.serial_number << "\n";
    oss << "  Firmware Version: " << device.firmware_version << "\n\n";

    oss << "Classification:\n";
    oss << "  Class: " << deviceClassToString(device.device_class) << "\n";
    oss << "  Priority: " << priorityToString(device.priority) << "\n";
    oss << "  Health Status: " << healthToString(device.health_status) << "\n";
    oss << "  Registration Status: " << registrationStatusToString(device.registration_status) << "\n\n";

    oss << "Network Information:\n";
    oss << "  IP Addresses: ";
    for (size_t i = 0; i < device.ip_addresses.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << device.ip_addresses[i];
    }
    oss << "\n";
    oss << "  MAC Address: " << device.mac_address << "\n";
    oss << "  Control Port: " << device.control_port << "\n";
    oss << "  Streaming Port: " << device.streaming_port << "\n\n";

    oss << "Audio Capabilities:\n";
    oss << "  Max Channels: " << device.max_channels << "\n";
    oss << "  Sample Rate: " << device.sample_rate << " Hz\n";
    oss << "  Buffer Size: " << device.buffer_size << "\n";
    oss << "  Latency Range: " << device.min_latency_ms << " - " << device.max_latency_ms << " ms\n\n";

    oss << "Performance Metrics:\n";
    oss << "  Total Requests: " << device.total_requests << "\n";
    oss << "  Successful Requests: " << device.successful_requests << "\n";
    oss << "  Failed Requests: " << device.failed_requests << "\n";
    oss << "  Success Rate: " << std::fixed << std::setprecision(1) << device.getSuccessRate() << "%\n";
    oss << "  Average Response Time: " << std::fixed << std::setprecision(2) << device.average_response_time_ms << " ms\n";
    oss << "  CPU Utilization: " << std::fixed << std::setprecision(1) << device.cpu_utilization_percent << "%\n";
    oss << "  Memory Utilization: " << std::fixed << std::setprecision(1) << device.memory_utilization_percent << "%\n\n";

    oss << "Organization:\n";
    oss << "  Groups: ";
    for (size_t i = 0; i < device.groups.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << device.groups[i];
    }
    oss << "\n";
    oss << "  Tags: ";
    for (size_t i = 0; i < device.tags.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << device.tags[i];
    }
    oss << "\n";
    oss << "  Location: " << device.location << "\n";
    oss << "  Owner: " << device.owner << "\n\n";

    if (!device.recent_errors.empty()) {
        oss << "Recent Errors (" << device.recent_errors.size() << "):\n";
        for (const auto& error : device.recent_errors) {
            oss << "  - " << error << "\n";
        }
        oss << "\n";
    }

    oss << "Timestamps:\n";
    oss << "  First Seen: " << formatTimestamp(device.first_seen) << "\n";
    oss << "  Last Seen: " << formatTimestamp(device.last_seen) << "\n";
    oss << "  Registration Time: " << formatTimestamp(device.registration_time) << "\n";
    oss << "  Last Config Update: " << formatTimestamp(device.last_configuration_update) << "\n";

    return oss.str();
}

std::string NetworkAudioDeviceRegistry::generateRegistryReport() const {
    std::ostringstream oss;

    std::lock_guard<std::mutex> lock(registry_mutex_);

    oss << "=== Device Registry Full Report ===\n";
    oss << "Generated at: " << formatTimestamp(std::chrono::system_clock::now()) << "\n\n";

    // Summary section
    RegistryStatistics stats = getStatistics();
    oss << generateHealthReport();

    // All devices section
    oss << "All Registered Devices (" << device_registry_.size() << "):\n";
    oss << "=========================\n\n";

    for (const auto& pair : device_registry_) {
        const RegistryDeviceInfo& device = pair.second;
        oss << "Device: " << device.device_name << " (" << device.device_id << ")\n";
        oss << "  Manufacturer: " << device.manufacturer << "\n";
        oss << "  Model: " << device.model << "\n";
        oss << "  Class: " << deviceClassToString(device.device_class) << "\n";
        oss << "  Status: " << deviceStatusToString(device.status) << "\n";
        oss << "  Health: " << healthToString(device.health_status) << "\n";
        oss << "  Priority: " << priorityToString(device.priority) << "\n";
        oss << "  Channels: " << device.max_channels << "\n";
        oss << "  Sample Rate: " << device.sample_rate << " Hz\n";
        oss << "  Success Rate: " << std::fixed << std::setprecision(1) << device.getSuccessRate() << "%\n";
        oss << "  Average Response Time: " << std::fixed << std::setprecision(2) << device.average_response_time_ms << " ms\n";
        oss << "  Error Count: " << device.error_count << "\n";

        if (!device.groups.empty()) {
            oss << "  Groups: ";
            for (size_t i = 0; i < device.groups.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << device.groups[i];
            }
            oss << "\n";
        }

        if (!device.tags.empty()) {
            oss << "  Tags: ";
            for (size_t i = 0; i < device.tags.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << device.tags[i];
            }
            oss << "\n";
        }

        oss << "  Last Seen: " << formatTimestamp(device.last_seen) << "\n";
        oss << "\n";
    }

    // Configuration templates section
    if (!configuration_templates_.empty()) {
        oss << "Configuration Templates (" << configuration_templates_.size() << "):\n";
        oss << "=============================\n\n";

        for (const auto& pair : configuration_templates_) {
            const DeviceConfigurationTemplate& template_info = pair.second;
            oss << "Template: " << template_info.template_name << " (" << template_info.template_id << ")\n";
            oss << "  Version: " << template_info.template_version << "\n";
            oss << "  Device Class: " << template_info.device_class << "\n";
            oss << "  Default Sample Rate: " << template_info.default_sample_rate << " Hz\n";
            oss << "  Default Buffer Size: " << template_info.default_buffer_size << "\n";
            oss << "  Default Channels: " << template_info.default_channels << "\n";
            oss << "  Max Acceptable Latency: " << template_info.max_acceptable_latency_ms << " ms\n";
            oss << "  Auto-Configure: " << (template_info.auto_configure ? "Yes" : "No") << "\n";
            oss << "  Created: " << formatTimestamp(template_info.created_at) << "\n";
            oss << "  Last Updated: " << formatTimestamp(template_info.last_updated) << "\n";
            oss << "\n";
        }
    }

    return oss.str();
}

void NetworkAudioDeviceRegistry::exportToJSON(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return;
    }

    // This would implement full JSON export
    // For now, just write a simple representation
    file << "{\n";
    file << "  \"registry_info\": {\n";
    file << "    \"exported_at\": \"" << formatTimestamp(std::chrono::system_clock::now()) << "\",\n";
    file << "    \"total_devices\": " << device_registry_.size() << ",\n";
    file << "    \"total_templates\": " << configuration_templates_.size() << "\n";
    file << "  },\n";
    file << "  \"devices\": [\n";

    bool first = true;
    for (const auto& pair : device_registry_) {
        if (!first) file << ",\n";
        first = false;

        const RegistryDeviceInfo& device = pair.second;
        file << "    {\n";
        file << "      \"id\": \"" << device.device_id << "\",\n";
        file << "      \"name\": \"" << escapeJsonString(device.device_name) << "\",\n";
        file << "      \"manufacturer\": \"" << escapeJsonString(device.manufacturer) << "\",\n";
        file << "      \"model\": \"" << escapeJsonString(device.model) << "\",\n";
        file << "      \"class\": \"" << deviceClassToString(device.device_class) << "\",\n";
        file << "      \"priority\": \"" << priorityToString(device.priority) << "\",\n";
        file << "      \"health\": \"" << healthToString(device.health_status) << "\",\n";
        file << "      \"status\": \"" << deviceStatusToString(device.status) << "\",\n";
        file << "      \"channels\": " << device.max_channels << ",\n";
        file << "      \"sample_rate\": " << device.sample_rate << ",\n";
        file << "      \"success_rate\": " << std::fixed << std::setprecision(1) << device.getSuccessRate() << ",\n";
        file << "      \"error_count\": " << device.error_count << ",\n";
        file << "      \"last_seen\": \"" << formatTimestamp(device.last_seen) << "\"\n";
        file << "    }";
    }

    file << "\n  ]\n";
    file << "}\n";

    file.close();
}

void NetworkAudioDeviceRegistry::exportToCSV(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return;
    }

    // CSV header
    file << "Device ID,Device Name,Manufacturer,Model,Class,Priority,Health,Status,";
    file << "Channels,Sample Rate,Success Rate (%),Error Count,Last Seen\n";

    // Data rows
    for (const auto& pair : device_registry_) {
        const RegistryDeviceInfo& device = pair.second;

        file << device.device_id << ",";
        file << "\"" << device.device_name << "\",";
        file << "\"" << device.manufacturer << "\",";
        file << "\"" << device.model << "\",";
        file << deviceClassToString(device.device_class) << ",";
        file << priorityToString(device.priority) << ",";
        file << healthToString(device.health_status) << ",";
        file << deviceStatusToString(device.status) << ",";
        file << device.max_channels << ",";
        file << device.sample_rate << ",";
        file << std::fixed << std::setprecision(1) << device.getSuccessRate() << ",";
        file << device.error_count << ",";
        file << formatTimestamp(device.last_seen) << "\n";
    }

    file.close();
}

void NetworkAudioDeviceRegistry::setEventCallback(std::function<void(const RegistryEvent&)> callback) {
    event_callback_ = callback;
}

void NetworkAudioDeviceRegistry::publishEvent(const RegistryEvent& event) {
    recordEvent(event);
}

std::vector<std::string> NetworkAudioDeviceRegistry::subscribeToEvents(const std::vector<RegistryEventType>& event_types) {
    std::vector<std::string> subscription_ids;

    for (auto event_type : event_types) {
        std::string subscription_id = generateUUID();
        subscription_ids.push_back(subscription_id);
        subscription_ids_[subscription_id] = subscription_id;
        event_subscriptions_[subscription_id] = {event_type};
    }

    return subscription_ids;
}

void NetworkAudioDeviceRegistry::unsubscribeFromEvents(const std::vector<std::string>& subscription_ids) {
    for (const auto& subscription_id : subscription_ids) {
        event_subscriptions_.erase(subscription_id);
        subscription_ids_.erase(subscription_id);
    }
}

void NetworkAudioDeviceRegistry::performMaintenance() {
    if (config_.enable_automatic_cleanup) {
        cleanupStaleDevices();
        cleanupOldEvents();
    }

    if (config_.enable_health_monitoring) {
        performDeviceHealthChecks();
    }

    rebuildIndexes();
    updateStatistics();
}

void NetworkAudioDeviceRegistry::cleanupStaleDevices() {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto now = std::chrono::system_clock::now();
    auto removal_threshold = config_.offline_device_removal_delay;

    auto it = device_registry_.begin();
    while (it != device_registry_.end()) {
        const RegistryDeviceInfo& device = it->second;

        if (device.auto_remove && !device.persistent) {
            auto offline_time = std::chrono::duration_cast<std::chrono::seconds>(now - device.last_seen);

            if (device.status == DeviceStatus::OFFLINE && offline_time > removal_threshold) {
                std::string device_id = it->first;
                std::string device_name = it->second.device_name;

                // Remove from indexes
                removeFromIndexes(it->second);

                // Remove from storage
                if (storage_) {
                    storage_->deleteDevice(device_id);
                }

                // Remove from cache
                if (cache_) {
                    cache_->invalidate(device_id);
                }

                // Record event
                RegistryEvent event;
                event.type = RegistryEventType::REGISTRY_CLEANUP_PERFORMED;
                event.device_id = device_id;
                event.device_name = device_name;
                event.message = "Stale device removed: " + device_name;
                event.timestamp = now;
                event.source = "DeviceRegistry";
                event.severity = "info";
                recordEvent(event);

                it = device_registry_.erase(it);
                std::cout << "Removed stale device: " << device_name << std::endl;
            } else {
                ++it;
            }
        } else {
            ++it;
        }
    }
}

void NetworkAudioDeviceRegistry::cleanupOldEvents() {
    std::lock_guard<std::mutex> lock(events_mutex_);

    auto now = std::chrono::system_clock::now();
    auto cutoff_time = now - config_.metrics_retention_period;

    auto it = event_history_.begin();
    while (it != event_history_.end()) {
        if (it->timestamp < cutoff_time) {
            it = event_history_.erase(it);
        } else {
            ++it;
        }
    }
}

void NetworkAudioDeviceRegistry::rebuildIndexes() {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    // Clear existing indexes
    class_index_.clear();
    priority_index_.clear();
    health_index_.clear();
    group_index_.clear();
    tag_index_.clear();
    manufacturer_index_.clear();

    // Rebuild indexes
    for (const auto& pair : device_registry_) {
        addToIndexes(pair.second);
    }
}

void NetworkAudioDeviceRegistry::validateRegistry() {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    size_t validation_errors = 0;

    for (const auto& pair : device_registry_) {
        const RegistryDeviceInfo& device = pair.second;

        // Validate device ID
        if (!isValidDeviceId(device.device_id)) {
            validation_errors++;
            std::cerr << "Invalid device ID: " << device.device_id << std::endl;
        }

        // Validate device name
        if (!isValidDeviceName(device.device_name)) {
            validation_errors++;
            std::cerr << "Invalid device name: " << device.device_name << std::endl;
        }

        // Validate IP addresses
        for (const auto& ip : device.ip_addresses) {
            if (!isValidIpAddress(ip)) {
                validation_errors++;
                std::cerr << "Invalid IP address: " << ip << " for device " << device.device_id << std::endl;
            }
        }

        // Validate MAC address
        if (!isValidMacAddress(device.mac_address)) {
            validation_errors++;
            std::cerr << "Invalid MAC address: " << device.mac_address << " for device " << device.device_id << std::endl;
        }
    }

    if (validation_errors > 0) {
        std::cout << "Registry validation completed with " << validation_errors << " errors" << std::endl;
    } else {
        std::cout << "Registry validation completed successfully" << std::endl;
    }
}

void NetworkAudioDeviceRegistry::optimizeRegistry() {
    // Optimize indexes
    rebuildIndexes();

    // Clean up cache
    if (cache_) {
        cache_->clear();
    }

    // Perform storage optimization
    if (storage_) {
        storage_->cleanup();
    }

    // Update statistics
    updateStatistics();

    std::cout << "Registry optimization completed" << std::endl;
}

void NetworkAudioDeviceRegistry::updateConfig(const RegistryConfig& config) {
    config_ = config;

    // Update cache settings if needed
    if (cache_) {
        cache_->setMaxSize(config_.cache_size_limit);
        cache_->setTTL(config_.cache_ttl);
    }

    // Update health monitor settings
    if (health_monitor_) {
        // Health monitor would need a method to update its config
    }
}

void NetworkAudioDeviceRegistry::setDiscoveryService(std::shared_ptr<AudioNetworkDiscovery> discovery_service) {
    discovery_service_ = discovery_service;
}

void NetworkAudioDeviceRegistry::syncWithDiscoveryService() {
    if (!discovery_service_) {
        return;
    }

    auto discovered_devices = discovery_service_->getDiscoveredDevices();

    for (const auto& discovered_device : discovered_devices) {
        if (!isDeviceRegistered(discovered_device.device_id)) {
            // Auto-register discovered devices if enabled
            if (config_.auto_register_discovered_devices) {
                RegistryDeviceInfo registry_device;
                static_cast<AudioDeviceInfo&>(registry_device) = discovered_device;

                // Set default registry-specific properties
                registry_device.device_class = DeviceClass::UNKNOWN;
                registry_device.priority = DevicePriority::NORMAL;
                registry_device.health_status = DeviceHealth::HEALTHY;
                registry_device.registration_status = DeviceRegistrationStatus::PENDING_REGISTRATION;

                registerDevice(registry_device);
            }
        }
    }
}

bool NetworkAudioDeviceRegistry::autoRegisterDiscoveredDevices() {
    if (!discovery_service_) {
        return false;
    }

    auto discovered_devices = discovery_service_->getDiscoveredDevices();
    std::vector<RegistryDeviceInfo> devices_to_register;

    for (const auto& discovered_device : discovered_devices) {
        if (!isDeviceRegistered(discovered_device.device_id) && matchesFilters(discovered_device)) {
            RegistryDeviceInfo registry_device;
            static_cast<AudioDeviceInfo&>(registry_device) = discovered_device;

            // Set default registry-specific properties
            registry_device.device_class = DeviceClass::UNKNOWN;
            registry_device.priority = DevicePriority::NORMAL;
            registry_device.health_status = DeviceHealth::HEALTHY;
            registry_device.registration_status = DeviceRegistrationStatus::PENDING_REGISTRATION;

            devices_to_register.push_back(registry_device);
        }
    }

    std::vector<std::string> registered_ids = registerDevices(devices_to_register);
    return !registered_ids.empty();
}

// Private methods implementation

void NetworkAudioDeviceRegistry::maintenanceThread() {
    auto last_maintenance = std::chrono::steady_clock::now();

    while (running_.load()) {
        try {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_maintenance);

            if (elapsed >= config_.cleanup_interval) {
                performMaintenance();
                last_maintenance = now;
            }

            std::this_thread::sleep_for(std::chrono::minutes(1));

        } catch (const std::exception& e) {
            std::cerr << "Maintenance thread error: " << e.what() << std::endl;
        }
    }
}

void NetworkAudioDeviceRegistry::healthMonitorThread() {
    while (running_.load()) {
        try {
            performDeviceHealthChecks();
            std::this_thread::sleep_for(config_.health_check_interval);

        } catch (const std::exception& e) {
            std::cerr << "Health monitor thread error: " << e.what() << std::endl;
        }
    }
}

void NetworkAudioDeviceRegistry::backupThread() {
    while (scheduled_backup_enabled_.load()) {
        try {
            createBackup();
            std::this_thread::sleep_for(backup_interval_);

        } catch (const std::exception& e) {
            std::cerr << "Backup thread error: " << e.what() << std::endl;
        }
    }
}

void NetworkAudioDeviceRegistry::eventProcessingThread() {
    while (running_.load()) {
        try {
            std::queue<RegistryEvent> events_to_process;

            {
                std::lock_guard<std::mutex> lock(events_mutex_);
                events_to_process.swap(event_queue_);
            }

            while (!events_to_process.empty()) {
                RegistryEvent event = events_to_process.front();
                events_to_process.pop();

                // Add to history
                {
                    std::lock_guard<std::mutex> lock(events_mutex_);
                    event_history_.push_back(event);
                    if (event_history_.size() > 10000) {
                        event_history_.erase(event_history_.begin());
                    }
                }

                // Call callback if set
                if (event_callback_) {
                    event_callback_(event);
                }

                // Notify subscribers
                notifyEventSubscribers(event);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));

        } catch (const std::exception& e) {
            std::cerr << "Event processing thread error: " << e.what() << std::endl;
        }
    }
}

bool NetworkAudioDeviceRegistry::loadFromStorage() {
    if (!storage_) {
        return false;
    }

    // Load devices
    auto devices = storage_->loadAllDevices();
    for (const auto& device : devices) {
        device_registry_[device.device_id] = device;
        addToIndexes(device);

        // Add to health monitor
        if (health_monitor_) {
            health_monitor_->addDevice(device.device_id);
        }
    }

    // Load templates
    auto templates = storage_->loadAllTemplates();
    for (const auto& template_info : templates) {
        configuration_templates_[template_info.template_id] = template_info;
    }

    return true;
}

bool NetworkAudioDeviceRegistry::saveToStorage() {
    if (!storage_) {
        return false;
    }

    bool success = true;

    for (const auto& pair : device_registry_) {
        if (!storage_->saveDevice(pair.second)) {
            success = false;
        }
    }

    for (const auto& pair : configuration_templates_) {
        if (!storage_->saveTemplate(pair.second)) {
            success = false;
        }
    }

    return success;
}

bool NetworkAudioDeviceRegistry::createBackupFile(const std::string& backup_path) {
    std::ofstream file(backup_path);
    if (!file.is_open()) {
        return false;
    }

    // Create a comprehensive backup
    file << "{\n";
    file << "  \"backup_version\": \"1.0\",\n";
    file << "  \"created_at\": \"" << formatTimestamp(std::chrono::system_clock::now()) << "\",\n";
    file << "  \"registry_config\": " << "{},\n"; // Would serialize config here
    file << "  \"devices\": [\n";

    bool first_device = true;
    for (const auto& pair : device_registry_) {
        if (!first_device) file << ",\n";
        first_device = false;

        const RegistryDeviceInfo& device = pair.second;
        file << "    {\n";
        // Would serialize full device information here
        file << "      \"device_id\": \"" << device.device_id << "\",\n";
        file << "      \"device_name\": \"" << escapeJsonString(device.device_name) << "\"\n";
        file << "    }";
    }

    file << "\n  ],\n";
    file << "  \"configuration_templates\": [\n";

    bool first_template = true;
    for (const auto& pair : configuration_templates_) {
        if (!first_template) file << ",\n";
        first_template = false;

        const DeviceConfigurationTemplate& template_info = pair.second;
        file << "    {\n";
        file << "      \"template_id\": \"" << template_info.template_id << "\",\n";
        file << "      \"template_name\": \"" << escapeJsonString(template_info.template_name) << "\"\n";
        file << "    }";
    }

    file << "\n  ]\n";
    file << "}\n";

    file.close();
    return true;
}

bool NetworkAudioDeviceRegistry::restoreFromBackupFile(const std::string& backup_path) {
    // This would implement full JSON parsing and restoration
    // For now, just return false as placeholder
    std::ifstream file(backup_path);
    if (!file.is_open()) {
        return false;
    }

    file.close();
    return false; // Placeholder
}

void NetworkAudioDeviceRegistry::addToIndexes(const RegistryDeviceInfo& device) {
    class_index_[device.device_class].insert(device.device_id);
    priority_index_[device.priority].insert(device.device_id);
    health_index_[device.health_status].insert(device.device_id);
    manufacturer_index_[device.manufacturer].insert(device.device_id);

    for (const auto& group : device.groups) {
        group_index_[group].insert(device.device_id);
    }

    for (const auto& tag : device.tags) {
        tag_index_[tag].insert(device.device_id);
    }
}

void NetworkAudioDeviceRegistry::removeFromIndexes(const RegistryDeviceInfo& device) {
    class_index_[device.device_class].erase(device.device_id);
    priority_index_[device.priority].erase(device.device_id);
    health_index_[device.health_status].erase(device.device_id);
    manufacturer_index_[device.manufacturer].erase(device.device_id);

    for (const auto& group : device.groups) {
        auto it = group_index_.find(group);
        if (it != group_index_.end()) {
            it->second.erase(device.device_id);
            if (it->second.empty()) {
                group_index_.erase(it);
            }
        }
    }

    for (const auto& tag : device.tags) {
        auto it = tag_index_.find(tag);
        if (it != tag_index_.end()) {
            it->second.erase(device.device_id);
            if (it->second.empty()) {
                tag_index_.erase(it);
            }
        }
    }
}

void NetworkAudioDeviceRegistry::updateIndexes(const RegistryDeviceInfo& old_device, const RegistryDeviceInfo& new_device) {
    removeFromIndexes(old_device);
    addToIndexes(new_device);
}

std::string NetworkAudioDeviceRegistry::generateDeviceId(const RegistryDeviceInfo& device) const {
    if (!device.device_id.empty()) {
        return device.device_id;
    }

    std::ostringstream oss;
    oss << device.manufacturer << "_" << device.model << "_" << device.serial_number;

    std::string id = oss.str();
    std::replace(id.begin(), id.end(), ' ', '_');
    std::replace(id.begin(), id.end(), ':', '_');
    std::transform(id.begin(), id.end(), id.begin(), ::tolower);

    // Ensure uniqueness
    std::string base_id = id;
    int counter = 1;
    while (device_registry_.find(id) != device_registry_.end()) {
        oss.str("");
        oss << base_id << "_" << counter;
        id = oss.str();
        counter++;
    }

    return id;
}

std::string NetworkAudioDeviceRegistry::generateRegistrationToken() const {
    return generateUUID();
}

bool NetworkAudioDeviceRegistry::validateDevice(const RegistryDeviceInfo& device) const {
    if (device.device_name.empty() || device.manufacturer.empty()) {
        return false;
    }

    if (!isValidDeviceName(device.device_name)) {
        return false;
    }

    for (const auto& ip : device.ip_addresses) {
        if (!isValidIpAddress(ip)) {
            return false;
        }
    }

    if (!device.mac_address.empty() && !isValidMacAddress(device.mac_address)) {
        return false;
    }

    return true;
}

bool NetworkAudioDeviceRegistry::validateConfigurationTemplate(const DeviceConfigurationTemplate& template_info) const {
    if (template_info.template_id.empty() || template_info.template_name.empty()) {
        return false;
    }

    return true;
}

void NetworkAudioDeviceRegistry::updateStatistics() {
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    std::lock_guard<std::mutex> registry_lock(registry_mutex_);

    statistics_.total_devices = device_registry_.size();
    statistics_.registered_devices = 0;
    statistics_.online_devices = 0;
    statistics_.offline_devices = 0;
    statistics_.healthy_devices = 0;
    statistics_.unhealthy_devices = 0;
    statistics_.input_devices = 0;
    statistics_.output_devices = 0;
    statistics_.input_output_devices = 0;
    statistics_.high_priority_devices = 0;
    statistics_.critical_priority_devices = 0;

    double total_response_time = 0.0;
    double total_cpu_util = 0.0;
    double total_memory_util = 0.0;
    size_t responsive_devices = 0;

    for (const auto& pair : device_registry_) {
        const RegistryDeviceInfo& device = pair.second;

        if (device.registration_status == DeviceRegistrationStatus::REGISTERED) {
            statistics_.registered_devices++;
        }

        if (device.status == DeviceStatus::ONLINE) {
            statistics_.online_devices++;
        } else {
            statistics_.offline_devices++;
        }

        if (device.isHealthy()) {
            statistics_.healthy_devices++;
        } else {
            statistics_.unhealthy_devices++;
        }

        // Device class counts
        switch (device.device_class) {
            case DeviceClass::INPUT_DEVICE:
                statistics_.input_devices++;
                break;
            case DeviceClass::OUTPUT_DEVICE:
                statistics_.output_devices++;
                break;
            case DeviceClass::INPUT_OUTPUT_DEVICE:
                statistics_.input_output_devices++;
                break;
            default:
                break;
        }

        // Priority counts
        if (device.priority == DevicePriority::HIGH) {
            statistics_.high_priority_devices++;
        } else if (device.priority == DevicePriority::CRITICAL) {
            statistics_.critical_priority_devices++;
        }

        // Performance metrics
        if (device.average_response_time_ms > 0) {
            total_response_time += device.average_response_time_ms;
            responsive_devices++;
        }

        total_cpu_util += device.cpu_utilization_percent;
        total_memory_util += device.memory_utilization_percent;

        statistics_.total_requests += device.total_requests;
        statistics_.successful_requests += device.successful_requests;
        statistics_.failed_requests += device.failed_requests;
    }

    if (responsive_devices > 0) {
        statistics_.average_response_time_ms = total_response_time / responsive_devices;
    }

    if (statistics_.total_devices > 0) {
        statistics_.average_cpu_utilization = total_cpu_util / statistics_.total_devices;
        statistics_.average_memory_utilization = total_memory_util / statistics_.total_devices;
    }

    statistics_.last_update_time = std::chrono::system_clock::now();
    statistics_.uptime = std::chrono::duration_cast<std::chrono::seconds>(
        statistics_.last_update_time - registry_start_time_);
}

void NetworkAudioDeviceRegistry::recordEvent(const RegistryEvent& event) {
    std::lock_guard<std::mutex> lock(events_mutex_);
    event_queue_.push(event);
}

void NetworkAudioDeviceRegistry::notifyEventSubscribers(const RegistryEvent& event) {
    for (const auto& subscription : event_subscriptions_) {
        const std::vector<RegistryEventType>& event_types = subscription.second;

        for (auto event_type : event_types) {
            if (event.type == event_type) {
                // Would notify subscriber here
                break;
            }
        }
    }
}

bool NetworkAudioDeviceRegistry::matchesQuery(const RegistryDeviceInfo& device, const DeviceQuery& query) const {
    // Device class filter
    if (!query.device_classes.empty() &&
        std::find(query.device_classes.begin(), query.device_classes.end(), device.device_class) == query.device_classes.end()) {
        return false;
    }

    // Priority filter
    if (!query.priorities.empty() &&
        std::find(query.priorities.begin(), query.priorities.end(), device.priority) == query.priorities.end()) {
        return false;
    }

    // Health status filter
    if (!query.health_statuses.empty() &&
        std::find(query.health_statuses.begin(), query.health_statuses.end(), device.health_status) == query.health_statuses.end()) {
        return false;
    }

    // Registration status filter
    if (!query.registration_statuses.empty() &&
        std::find(query.registration_statuses.begin(), query.registration_statuses.end(), device.registration_status) == query.registration_statuses.end()) {
        return false;
    }

    // Manufacturer filter
    if (!query.manufacturers.empty() &&
        std::find(query.manufacturers.begin(), query.manufacturers.end(), device.manufacturer) == query.manufacturers.end()) {
        return false;
    }

    // Model filter
    if (!query.models.empty() &&
        std::find(query.models.begin(), query.models.end(), device.model) == query.models.end()) {
        return false;
    }

    // Connection type filter
    if (!query.connection_types.empty()) {
        bool has_connection_type = false;
        for (auto connection_type : query.connection_types) {
            if (std::find(device.connection_types.begin(), device.connection_types.end(), connection_type) != device.connection_types.end()) {
                has_connection_type = true;
                break;
            }
        }
        if (!has_connection_type) {
            return false;
        }
    }

    // Capabilities filter
    if (!query.capabilities.empty()) {
        bool has_all_capabilities = true;
        for (auto capability : query.capabilities) {
            if (std::find(device.capabilities.begin(), device.capabilities.end(), capability) == device.capabilities.end()) {
                has_all_capabilities = false;
                break;
            }
        }
        if (!has_all_capabilities) {
            return false;
        }
    }

    // Groups filter
    if (!query.groups.empty()) {
        bool has_group = false;
        for (const auto& group : query.groups) {
            if (std::find(device.groups.begin(), device.groups.end(), group) != device.groups.end()) {
                has_group = true;
                break;
            }
        }
        if (!has_group) {
            return false;
        }
    }

    // Tags filter
    if (!query.tags.empty()) {
        bool has_tag = false;
        for (const auto& tag : query.tags) {
            if (std::find(device.tags.begin(), device.tags.end(), tag) != device.tags.end()) {
                has_tag = true;
                break;
            }
        }
        if (!has_tag) {
            return false;
        }
    }

    // Numeric filters
    if (device.min_latency_ms < query.latency_range.first || device.min_latency_ms > query.latency_range.second) {
        return false;
    }

    if (device.max_channels < query.channel_range.first || device.max_channels > query.channel_range.second) {
        return false;
    }

    if (device.cpu_utilization_percent < query.cpu_utilization_range.first ||
        device.cpu_utilization_percent > query.cpu_utilization_range.second) {
        return false;
    }

    if (device.memory_utilization_percent < query.memory_utilization_range.first ||
        device.memory_utilization_percent > query.memory_utilization_range.second) {
        return false;
    }

    double success_rate = device.getSuccessRate();
    if (success_rate < query.success_rate_range.first || success_rate > query.success_rate_range.second) {
        return false;
    }

    // Text search
    if (!query.search_query.empty()) {
        std::string search_text = device.device_name + " " + device.manufacturer + " " + device.model + " " + device.description;
        if (query.case_sensitive) {
            if (query.exact_match) {
                if (search_text.find(query.search_query) == std::string::npos) {
                    return false;
                }
            } else {
                if (search_text.find(query.search_query) == std::string::npos) {
                    return false;
                }
            }
        } else {
            std::string lower_search_text = search_text;
            std::string lower_query = query.search_query;
            std::transform(lower_search_text.begin(), lower_search_text.end(), lower_search_text.begin(), ::tolower);
            std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(), ::tolower);

            if (lower_search_text.find(lower_query) == std::string::npos) {
                return false;
            }
        }
    }

    // Status filters
    if (query.online_only && device.status != DeviceStatus::ONLINE) {
        return false;
    }

    if (query.healthy_only && !device.isHealthy()) {
        return false;
    }

    if (query.registered_only && device.registration_status != DeviceRegistrationStatus::REGISTERED) {
        return false;
    }

    if (query.available_only && !device.isAvailable()) {
        return false;
    }

    // Time-based filters would go here

    return true;
}

std::vector<RegistryDeviceInfo> NetworkAudioDeviceRegistry::sortDevices(const std::vector<RegistryDeviceInfo>& devices,
                                                                        const std::string& sort_by, bool ascending) const {
    std::vector<RegistryDeviceInfo> sorted_devices = devices;

    if (sort_by == "device_name") {
        std::sort(sorted_devices.begin(), sorted_devices.end(),
                  [ascending](const RegistryDeviceInfo& a, const RegistryDeviceInfo& b) {
                      return ascending ? (a.device_name < b.device_name) : (a.device_name > b.device_name);
                  });
    } else if (sort_by == "manufacturer") {
        std::sort(sorted_devices.begin(), sorted_devices.end(),
                  [ascending](const RegistryDeviceInfo& a, const RegistryDeviceInfo& b) {
                      return ascending ? (a.manufacturer < b.manufacturer) : (a.manufacturer > b.manufacturer);
                  });
    } else if (sort_by == "priority") {
        std::sort(sorted_devices.begin(), sorted_devices.end(),
                  [ascending](const RegistryDeviceInfo& a, const RegistryDeviceInfo& b) {
                      return ascending ? (a.priority < b.priority) : (a.priority > b.priority);
                  });
    } else if (sort_by == "health") {
        std::sort(sorted_devices.begin(), sorted_devices.end(),
                  [ascending](const RegistryDeviceInfo& a, const RegistryDeviceInfo& b) {
                      return ascending ? (a.health_status < b.health_status) : (a.health_status > b.health_status);
                  });
    } else if (sort_by == "success_rate") {
        std::sort(sorted_devices.begin(), sorted_devices.end(),
                  [ascending](const RegistryDeviceInfo& a, const RegistryDeviceInfo& b) {
                      double rate_a = a.getSuccessRate();
                      double rate_b = b.getSuccessRate();
                      return ascending ? (rate_a < rate_b) : (rate_a > rate_b);
                  });
    } else if (sort_by == "response_time") {
        std::sort(sorted_devices.begin(), sorted_devices.end(),
                  [ascending](const RegistryDeviceInfo& a, const RegistryDeviceInfo& b) {
                      return ascending ? (a.average_response_time_ms < b.average_response_time_ms) :
                                       (a.average_response_time_ms > b.average_response_time_ms);
                  });
    } else if (sort_by == "last_seen") {
        std::sort(sorted_devices.begin(), sorted_devices.end(),
                  [ascending](const RegistryDeviceInfo& a, const RegistryDeviceInfo& b) {
                      return ascending ? (a.last_seen < b.last_seen) : (a.last_seen > b.last_seen);
                  });
    }

    return sorted_devices;
}

void NetworkAudioDeviceRegistry::checkDeviceRegistrations() {
    // Check for expired registrations
    auto now = std::chrono::system_clock::now();

    std::lock_guard<std::mutex> lock(registry_mutex_);
    for (auto& pair : device_registry_) {
        RegistryDeviceInfo& device = pair.second;

        if (device.registration_status == DeviceRegistrationStatus::REGISTERED) {
            auto registration_age = std::chrono::duration_cast<std::chrono::seconds>(now - device.registration_time);

            if (registration_age > device.registration_ttl) {
                device.registration_status = DeviceRegistrationStatus::REGISTRATION_EXPIRED;

                RegistryEvent event;
                event.type = RegistryEventType::DEVICE_HEALTH_CHANGED;
                event.device_id = device.device_id;
                event.device_name = device.device_name;
                event.message = "Device registration expired: " + device.device_name;
                event.timestamp = now;
                event.source = "DeviceRegistry";
                event.severity = "warning";
                recordEvent(event);
            }
        }
    }
}

void NetworkAudioDeviceRegistry::refreshDeviceInformation() {
    // This would refresh device information from the actual devices
    // For now, just update last seen times for online devices
    auto now = std::chrono::system_clock::now();

    std::lock_guard<std::mutex> lock(registry_mutex_);
    for (auto& pair : device_registry_) {
        RegistryDeviceInfo& device = pair.second;

        if (device.status == DeviceStatus::ONLINE) {
            device.last_seen = now;
        }
    }
}

void NetworkAudioDeviceRegistry::performDeviceHealthChecks() {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    for (auto& pair : device_registry_) {
        RegistryDeviceInfo& device = pair.second;

        // Perform health check based on various metrics
        DeviceHealth new_health = DeviceHealth::HEALTHY;

        // Check error rate
        double error_rate = device.total_requests > 0 ?
            (static_cast<double>(device.failed_requests) / device.total_requests) * 100.0 : 0.0;

        if (error_rate > 20.0) {
            new_health = DeviceHealth::CRITICAL;
        } else if (error_rate > 10.0) {
            new_health = DeviceHealth::DEGRADED;
        } else if (error_rate > 5.0) {
            new_health = DeviceHealth::WARNING;
        }

        // Check resource utilization
        if (device.cpu_utilization_percent > 95.0 || device.memory_utilization_percent > 95.0) {
            new_health = DeviceHealth::CRITICAL;
        } else if (device.cpu_utilization_percent > 80.0 || device.memory_utilization_percent > 80.0) {
            if (new_health == DeviceHealth::HEALTHY) {
                new_health = DeviceHealth::WARNING;
            }
        }

        // Check response time
        if (device.average_response_time_ms > 1000.0) {
            new_health = DeviceHealth::CRITICAL;
        } else if (device.average_response_time_ms > 500.0) {
            if (new_health == DeviceHealth::HEALTHY) {
                new_health = DeviceHealth::WARNING;
            }
        }

        // Check if status changed
        if (new_health != device.health_status) {
            updateDeviceHealth(device.device_id, new_health, "Automated health check");
        }
    }
}

std::string NetworkAudioDeviceRegistry::deviceClassToString(DeviceClass device_class) const {
    return RegistryUtils::deviceClassToString(device_class);
}

std::string NetworkAudioDeviceRegistry::priorityToString(DevicePriority priority) const {
    return RegistryUtils::priorityToString(priority);
}

std::string NetworkAudioDeviceRegistry::healthToString(DeviceHealth health) const {
    return RegistryUtils::healthToString(health);
}

DeviceClass NetworkAudioDeviceRegistry::stringToDeviceClass(const std::string& str) const {
    return RegistryUtils::stringToDeviceClass(str);
}

DevicePriority NetworkAudioDeviceRegistry::stringToPriority(const std::string& str) const {
    return RegistryUtils::stringToPriority(str);
}

DeviceHealth NetworkAudioDeviceRegistry::stringToHealth(const std::string& str) const {
    return RegistryUtils::stringToHealth(str);
}

bool NetworkAudioDeviceRegistry::matchesFilters(const AudioDeviceInfo& device) const {
    // Check if device matches registry filters
    for (const auto& capability : config_.required_capabilities) {
        if (std::find(device.capabilities.begin(), device.capabilities.end(), capability) == device.capabilities.end()) {
            return false;
        }
    }

    return true;
}

// ============================================================================
// RegistryStorage Implementation (Placeholder)
// ============================================================================

bool RegistryStorage::initialize(const RegistryConfig& config) {
    // Initialize storage implementation
    return true;
}

void RegistryStorage::shutdown() {
    // Shutdown storage implementation
}

bool RegistryStorage::saveDevice(const RegistryDeviceInfo& device) {
    // Save device to storage
    return true;
}

bool RegistryStorage::loadDevice(const std::string& device_id, RegistryDeviceInfo& device) {
    // Load device from storage
    return false;
}

bool RegistryStorage::deleteDevice(const std::string& device_id) {
    // Delete device from storage
    return true;
}

bool RegistryStorage::saveTemplate(const DeviceConfigurationTemplate& template_info) {
    // Save template to storage
    return true;
}

bool RegistryStorage::loadTemplate(const std::string& template_id, DeviceConfigurationTemplate& template_info) {
    // Load template from storage
    return false;
}

bool RegistryStorage::deleteTemplate(const std::string& template_id) {
    // Delete template from storage
    return true;
}

std::vector<RegistryDeviceInfo> RegistryStorage::loadAllDevices() {
    // Load all devices from storage
    return {};
}

std::vector<DeviceConfigurationTemplate> RegistryStorage::loadAllTemplates() {
    // Load all templates from storage
    return {};
}

bool RegistryStorage::createBackup(const std::string& backup_path) {
    // Create backup
    return true;
}

bool RegistryStorage::restoreFromBackup(const std::string& backup_path) {
    // Restore from backup
    return true;
}

bool RegistryStorage::cleanup() {
    // Cleanup storage
    return true;
}

// ============================================================================
// RegistryCache Implementation
// ============================================================================

RegistryCache::RegistryCache(size_t max_size, std::chrono::seconds ttl)
    : max_size_(max_size), ttl_(ttl) {
}

RegistryCache::~RegistryCache() {
    clear();
}

bool RegistryCache::get(const std::string& key, RegistryDeviceInfo& device) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    auto it = cache_.find(key);
    if (it != cache_.end()) {
        if (!isExpired(it->second.second)) {
            device = it->second.first;
            updateLRU(key);
            return true;
        } else {
            // Remove expired entry
            lru_list_.erase(lru_map_[key]);
            lru_map_.erase(key);
            cache_.erase(it);
        }
    }

    return false;
}

void RegistryCache::put(const std::string& key, const RegistryDeviceInfo& device) {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    auto now = std::chrono::steady_clock::now();
    cache_[key] = {device, now};

    // Update LRU
    updateLRU(key);

    // Check if we need to evict
    if (cache_.size() > max_size_) {
        evictOldest();
    }
}

void RegistryCache::invalidate(const std::string& key) {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    auto it = cache_.find(key);
    if (it != cache_.end()) {
        lru_list_.erase(lru_map_[key]);
        lru_map_.erase(key);
        cache_.erase(it);
    }
}

void RegistryCache::clear() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.clear();
    lru_list_.clear();
    lru_map_.clear();
}

size_t RegistryCache::size() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return cache_.size();
}

void RegistryCache::setMaxSize(size_t max_size) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    max_size_ = max_size;

    while (cache_.size() > max_size_) {
        evictOldest();
    }
}

void RegistryCache::setTTL(std::chrono::seconds ttl) {
    ttl_ = ttl;
}

void RegistryCache::evictOldest() {
    if (lru_list_.empty()) {
        return;
    }

    std::string oldest_key = lru_list_.front();
    lru_list_.pop_front();
    lru_map_.erase(oldest_key);
    cache_.erase(oldest_key);
}

void RegistryCache::updateLRU(const std::string& key) {
    auto it = lru_map_.find(key);
    if (it != lru_map_.end()) {
        lru_list_.erase(it->second);
    }

    lru_list_.push_back(key);
    lru_map_[key] = std::prev(lru_list_.end());
}

bool RegistryCache::isExpired(const std::chrono::steady_clock::time_point& timestamp) const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::seconds>(now - timestamp) > ttl_;
}

// ============================================================================
// RegistryHealthMonitor Implementation
// ============================================================================

RegistryHealthMonitor::RegistryHealthMonitor(const RegistryConfig& config)
    : config_(config) {
}

RegistryHealthMonitor::~RegistryHealthMonitor() {
    stop();
}

void RegistryHealthMonitor::start() {
    if (running_.load()) {
        return;
    }

    running_.store(true);
    monitor_thread_ = std::thread(&RegistryHealthMonitor::monitorThread, this);
}

void RegistryHealthMonitor::stop() {
    running_.store(false);
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
}

void RegistryHealthMonitor::addDevice(const std::string& device_id) {
    std::unique_lock<std::shared_mutex> lock(health_data_mutex_);
    health_data_[device_id] = DeviceHealthData{};
}

void RegistryHealthMonitor::removeDevice(const std::string& device_id) {
    std::unique_lock<std::shared_mutex> lock(health_data_mutex_);
    health_data_.erase(device_id);
}

void RegistryHealthMonitor::updateDeviceMetrics(const std::string& device_id, double cpu_util, double memory_util,
                                              double response_time, bool request_successful) {
    std::unique_lock<std::shared_mutex> lock(health_data_mutex_);

    auto it = health_data_.find(device_id);
    if (it == health_data_.end()) {
        return;
    }

    DeviceHealthData& data = it->second;

    // Update metrics history (keep last 100 entries)
    data.cpu_utilization_history.push_back(cpu_util);
    if (data.cpu_utilization_history.size() > 100) {
        data.cpu_utilization_history.pop_front();
    }

    data.memory_utilization_history.push_back(memory_util);
    if (data.memory_utilization_history.size() > 100) {
        data.memory_utilization_history.pop_front();
    }

    data.response_time_history.push_back(response_time);
    if (data.response_time_history.size() > 100) {
        data.response_time_history.pop_front();
    }

    data.request_success_history.push_back(request_successful);
    if (data.request_success_history.size() > 100) {
        data.request_success_history.pop_front();
    }

    // Update consecutive failures
    if (request_successful) {
        data.consecutive_failures = 0;
    } else {
        data.consecutive_failures++;
    }
}

void RegistryHealthMonitor::recordDeviceError(const std::string& device_id, const std::string& error) {
    std::unique_lock<std::shared_mutex> lock(health_data_mutex_);

    auto it = health_data_.find(device_id);
    if (it == health_data_.end()) {
        return;
    }

    DeviceHealthData& data = it->second;
    data.recent_errors.push_back(error);
    data.last_error_time = std::chrono::steady_clock::now();

    // Keep only last 20 errors
    if (data.recent_errors.size() > 20) {
        data.recent_errors.erase(data.recent_errors.begin());
    }
}

DeviceHealth RegistryHealthMonitor::getDeviceHealth(const std::string& device_id) const {
    std::shared_lock<std::shared_mutex> lock(health_data_mutex_);

    auto it = health_data_.find(device_id);
    if (it == health_data_.end()) {
        return DeviceHealth::UNKNOWN;
    }

    return it->second.current_health;
}

std::vector<std::string> RegistryHealthMonitor::getUnhealthyDevices() const {
    std::shared_lock<std::shared_mutex> lock(health_data_mutex_);

    std::vector<std::string> unhealthy_devices;

    for (const auto& pair : health_data_) {
        if (pair.second.current_health != DeviceHealth::HEALTHY) {
            unhealthy_devices.push_back(pair.first);
        }
    }

    return unhealthy_devices;
}

void RegistryHealthMonitor::setHealthCallback(std::function<void(const std::string&, DeviceHealth)> callback) {
    health_callback_ = callback;
}

void RegistryHealthMonitor::monitorThread() {
    while (running_.load()) {
        try {
            // Update health for all devices
            std::vector<std::string> device_ids;

            {
                std::shared_lock<std::shared_mutex> lock(health_data_mutex_);
                for (const auto& pair : health_data_) {
                    device_ids.push_back(pair.first);
                }
            }

            for (const auto& device_id : device_ids) {
                updateDeviceHealth(device_id);
            }

            // Cleanup old data
            cleanupOldData();

            std::this_thread::sleep_for(config_.health_check_interval);

        } catch (const std::exception& e) {
            std::cerr << "Health monitor thread error: " << e.what() << std::endl;
        }
    }
}

void RegistryHealthMonitor::updateDeviceHealth(const std::string& device_id) {
    std::unique_lock<std::shared_mutex> lock(health_data_mutex_);

    auto it = health_data_.find(device_id);
    if (it == health_data_.end()) {
        return;
    }

    DeviceHealthData& data = it->second;
    DeviceHealth new_health = calculateHealth(data);

    if (new_health != data.current_health) {
        DeviceHealth old_health = data.current_health;
        data.current_health = new_health;
        data.last_health_update = std::chrono::steady_clock::now();

        if (health_callback_) {
            health_callback_(device_id, new_health);
        }
    }
}

DeviceHealth RegistryHealthMonitor::calculateHealth(const DeviceHealthData& data) const {
    // Calculate overall health based on various metrics

    // Check consecutive failures
    if (data.consecutive_failures >= 5) {
        return DeviceHealth::CRITICAL;
    } else if (data.consecutive_failures >= 3) {
        return DeviceHealth::DEGRADED;
    }

    // Check CPU utilization
    if (!data.cpu_utilization_history.empty()) {
        double avg_cpu = calculateAverage(data.cpu_utilization_history);
        if (avg_cpu > 90.0) {
            return DeviceHealth::CRITICAL;
        } else if (avg_cpu > 80.0) {
            return DeviceHealth::WARNING;
        }
    }

    // Check memory utilization
    if (!data.memory_utilization_history.empty()) {
        double avg_memory = calculateAverage(data.memory_utilization_history);
        if (avg_memory > 90.0) {
            return DeviceHealth::CRITICAL;
        } else if (avg_memory > 80.0) {
            return DeviceHealth::WARNING;
        }
    }

    // Check response time
    if (!data.response_time_history.empty()) {
        double avg_response = calculateAverage(data.response_time_history);
        if (avg_response > 1000.0) {
            return DeviceHealth::CRITICAL;
        } else if (avg_response > 500.0) {
            return DeviceHealth::WARNING;
        }
    }

    // Check success rate
    if (!data.request_success_history.empty()) {
        size_t successful_requests = std::count(data.request_success_history.begin(),
                                               data.request_success_history.end(), true);
        double success_rate = (static_cast<double>(successful_requests) / data.request_success_history.size()) * 100.0;

        if (success_rate < 80.0) {
            return DeviceHealth::CRITICAL;
        } else if (success_rate < 95.0) {
            return DeviceHealth::WARNING;
        }
    }

    // Check recent errors
    if (data.recent_errors.size() > 10) {
        return DeviceHealth::WARNING;
    }

    return DeviceHealth::HEALTHY;
}

void RegistryHealthMonitor::cleanupOldData() {
    std::unique_lock<std::shared_mutex> lock(health_data_mutex_);

    auto cutoff_time = std::chrono::steady_clock::now() - config_.metrics_retention_period;

    for (auto& pair : health_data_) {
        DeviceHealthData& data = pair.second;

        // Clean up old error messages
        if (!data.recent_errors.empty() && data.last_error_time < cutoff_time) {
            data.recent_errors.clear();
        }
    }
}

double RegistryHealthMonitor::calculateAverage(const std::deque<double>& values) const {
    if (values.empty()) {
        return 0.0;
    }

    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }

    return sum / values.size();
}

// ============================================================================
// RegistryUtils Implementation
// ============================================================================

namespace RegistryUtils {

std::string deviceClassToString(DeviceClass device_class) {
    switch (device_class) {
        case DeviceClass::INPUT_DEVICE: return "Input Device";
        case DeviceClass::OUTPUT_DEVICE: return "Output Device";
        case DeviceClass::INPUT_OUTPUT_DEVICE: return "Input/Output Device";
        case DeviceClass::NETWORK_INTERFACE: return "Network Interface";
        case DeviceClass::MIXER: return "Mixer";
        case DeviceClass::PROCESSOR: return "Processor";
        case DeviceClass::CONVERTER: return "Converter";
        case DeviceClass::BRIDGE: return "Bridge";
        case DeviceClass::CONTROLLER: return "Controller";
        case DeviceClass::MONITOR: return "Monitor";
        case DeviceClass::RECORDER: return "Recorder";
        case DeviceClass::PLAYER: return "Player";
        default: return "Unknown";
    }
}

std::string priorityToString(DevicePriority priority) {
    switch (priority) {
        case DevicePriority::CRITICAL: return "Critical";
        case DevicePriority::HIGH: return "High";
        case DevicePriority::NORMAL: return "Normal";
        case DevicePriority::LOW: return "Low";
        case DevicePriority::BACKGROUND: return "Background";
        default: return "Unknown";
    }
}

std::string healthToString(DeviceHealth health) {
    switch (health) {
        case DeviceHealth::HEALTHY: return "Healthy";
        case DeviceHealth::WARNING: return "Warning";
        case DeviceHealth::DEGRADED: return "Degraded";
        case DeviceHealth::CRITICAL: return "Critical";
        case DeviceHealth::OFFLINE: return "Offline";
        default: return "Unknown";
    }
}

std::string registrationStatusToString(DeviceRegistrationStatus status) {
    switch (status) {
        case DeviceRegistrationStatus::REGISTERED: return "Registered";
        case DeviceRegistrationStatus::UNREGISTERED: return "Unregistered";
        case DeviceRegistrationStatus::PENDING_REGISTRATION: return "Pending Registration";
        case DeviceRegistrationStatus::REGISTRATION_FAILED: return "Registration Failed";
        case DeviceRegistrationStatus::REGISTRATION_EXPIRED: return "Registration Expired";
        case DeviceRegistrationStatus::TEMPORARY_UNAVAILABLE: return "Temporary Unavailable";
        default: return "Unknown";
    }
}

std::string eventTypeToString(RegistryEventType type) {
    switch (type) {
        case RegistryEventType::DEVICE_REGISTERED: return "Device Registered";
        case RegistryEventType::DEVICE_UNREGISTERED: return "Device Unregistered";
        case RegistryEventType::DEVICE_UPDATED: return "Device Updated";
        case RegistryEventType::DEVICE_HEALTH_CHANGED: return "Device Health Changed";
        case RegistryEventType::DEVICE_CONFIG_CHANGED: return "Device Config Changed";
        case RegistryEventType::DEVICE_PRIORITY_CHANGED: return "Device Priority Changed";
        case RegistryEventType::DEVICE_GROUP_CHANGED: return "Device Group Changed";
        case RegistryEventType::DEVICE_CONNECTED: return "Device Connected";
        case RegistryEventType::DEVICE_DISCONNECTED: return "Device Disconnected";
        case RegistryEventType::CONFIGURATION_TEMPLATE_CREATED: return "Configuration Template Created";
        case RegistryEventType::CONFIGURATION_TEMPLATE_UPDATED: return "Configuration Template Updated";
        case RegistryEventType::CONFIGURATION_TEMPLATE_DELETED: return "Configuration Template Deleted";
        case RegistryEventType::REGISTRY_BACKUP_CREATED: return "Registry Backup Created";
        case RegistryEventType::REGISTRY_RESTORED: return "Registry Restored";
        case RegistryEventType::REGISTRY_CLEANUP_PERFORMED: return "Registry Cleanup Performed";
        case RegistryEventType::ERROR_OCCURRED: return "Error Occurred";
        case RegistryEventType::WARNING_ISSUED: return "Warning Issued";
        default: return "Unknown";
    }
}

DeviceClass stringToDeviceClass(const std::string& str) {
    if (str == "Input Device" || str == "INPUT_DEVICE") return DeviceClass::INPUT_DEVICE;
    if (str == "Output Device" || str == "OUTPUT_DEVICE") return DeviceClass::OUTPUT_DEVICE;
    if (str == "Input/Output Device" || str == "INPUT_OUTPUT_DEVICE") return DeviceClass::INPUT_OUTPUT_DEVICE;
    if (str == "Network Interface" || str == "NETWORK_INTERFACE") return DeviceClass::NETWORK_INTERFACE;
    if (str == "Mixer") return DeviceClass::MIXER;
    if (str == "Processor") return DeviceClass::PROCESSOR;
    if (str == "Converter") return DeviceClass::CONVERTER;
    if (str == "Bridge") return DeviceClass::BRIDGE;
    if (str == "Controller") return DeviceClass::CONTROLLER;
    if (str == "Monitor") return DeviceClass::MONITOR;
    if (str == "Recorder") return DeviceClass::RECORDER;
    if (str == "Player") return DeviceClass::PLAYER;
    return DeviceClass::UNKNOWN;
}

DevicePriority stringToPriority(const std::string& str) {
    if (str == "Critical" || str == "CRITICAL") return DevicePriority::CRITICAL;
    if (str == "High" || str == "HIGH") return DevicePriority::HIGH;
    if (str == "Normal" || str == "NORMAL") return DevicePriority::NORMAL;
    if (str == "Low" || str == "LOW") return DevicePriority::LOW;
    if (str == "Background" || str == "BACKGROUND") return DevicePriority::BACKGROUND;
    return DevicePriority::NORMAL;
}

DeviceHealth stringToHealth(const std::string& str) {
    if (str == "Healthy" || str == "HEALTHY") return DeviceHealth::HEALTHY;
    if (str == "Warning" || str == "WARNING") return DeviceHealth::WARNING;
    if (str == "Degraded" || str == "DEGRADED") return DeviceHealth::DEGRADED;
    if (str == "Critical" || str == "CRITICAL") return DeviceHealth::CRITICAL;
    if (str == "Offline" || str == "OFFLINE") return DeviceHealth::OFFLINE;
    return DeviceHealth::UNKNOWN;
}

DeviceRegistrationStatus stringToRegistrationStatus(const std::string& str) {
    if (str == "Registered" || str == "REGISTERED") return DeviceRegistrationStatus::REGISTERED;
    if (str == "Unregistered" || str == "UNREGISTERED") return DeviceRegistrationStatus::UNREGISTERED;
    if (str == "Pending Registration" || str == "PENDING_REGISTRATION") return DeviceRegistrationStatus::PENDING_REGISTRATION;
    if (str == "Registration Failed" || str == "REGISTRATION_FAILED") return DeviceRegistrationStatus::REGISTRATION_FAILED;
    if (str == "Registration Expired" || str == "REGISTRATION_EXPIRED") return DeviceRegistrationStatus::REGISTRATION_EXPIRED;
    if (str == "Temporary Unavailable" || str == "TEMPORARY_UNAVAILABLE") return DeviceRegistrationStatus::TEMPORARY_UNAVAILABLE;
    return DeviceRegistrationStatus::UNREGISTERED;
}

RegistryEventType stringToEventType(const std::string& str) {
    if (str == "Device Registered" || str == "DEVICE_REGISTERED") return RegistryEventType::DEVICE_REGISTERED;
    if (str == "Device Unregistered" || str == "DEVICE_UNREGISTERED") return RegistryEventType::DEVICE_UNREGISTERED;
    if (str == "Device Updated" || str == "DEVICE_UPDATED") return RegistryEventType::DEVICE_UPDATED;
    if (str == "Device Health Changed" || str == "DEVICE_HEALTH_CHANGED") return RegistryEventType::DEVICE_HEALTH_CHANGED;
    if (str == "Device Config Changed" || str == "DEVICE_CONFIG_CHANGED") return RegistryEventType::DEVICE_CONFIG_CHANGED;
    // ... add more mappings as needed
    return RegistryEventType::DEVICE_UPDATED;
}

std::string generateDeviceId(const RegistryDeviceInfo& device) {
    std::ostringstream oss;
    oss << device.manufacturer << "_" << device.model << "_" << device.serial_number;

    std::string id = oss.str();
    std::replace(id.begin(), id.end(), ' ', '_');
    std::replace(id.begin(), id.end(), ':', '_');
    std::transform(id.begin(), id.end(), id.begin(), ::tolower);

    return id;
}

std::string generateRegistrationToken() {
    return generateUUID();
}

std::string generateUUID() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);

    std::stringstream ss;
    ss << std::hex;

    for (int i = 0; i < 32; ++i) {
        ss << std::setw(1) << std::setfill('0') << dis(gen);
        if (i == 7 || i == 11 || i == 15 || i == 19) {
            ss << '-';
        }
    }

    return ss.str();
}

bool isValidDeviceId(const std::string& device_id) {
    if (device_id.empty() || device_id.length() > 256) {
        return false;
    }

    // Check for valid characters (alphanumeric, underscore, hyphen)
    return std::all_of(device_id.begin(), device_id.end(),
                      [](char c) { return std::isalnum(c) || c == '_' || c == '-'; });
}

bool isValidDeviceName(const std::string& device_name) {
    if (device_name.empty() || device_name.length() > 512) {
        return false;
    }

    // Allow more characters for device names
    return true;
}

bool isValidIpAddress(const std::string& ip_address) {
    // Basic IPv4 validation
    std::regex ipv4_pattern(R"(^(\d{1,3}\.){3}\d{1,3}$)");
    return std::regex_match(ip_address, ipv4_pattern);
}

bool isValidMacAddress(const std::string& mac_address) {
    // Basic MAC address validation
    std::regex mac_pattern(R"(^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$)");
    return std::regex_match(mac_address, mac_pattern);
}

std::string escapeJsonString(const std::string& str) {
    std::string escaped;
    for (char c : str) {
        switch (c) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (c >= 0 && c < 32) {
                    escaped += "\\u";
                    char buf[5];
                    snprintf(buf, sizeof(buf), "%04x", static_cast<unsigned char>(c));
                    escaped += buf;
                } else {
                    escaped += c;
                }
                break;
        }
    }
    return escaped;
}

std::string formatTimestamp(const std::chrono::system_clock::time_point& timestamp) {
    auto time_t = std::chrono::system_clock::to_time_t(timestamp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        timestamp.time_since_epoch()) % 1000;

    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&time_t), "%Y-%m-%d %H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    oss << " UTC";

    return oss.str();
}

std::string formatDuration(const std::chrono::seconds& duration) {
    auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration % std::chrono::hours(1));
    auto seconds = duration % std::chrono::minutes(1);

    std::ostringstream oss;
    if (hours.count() > 0) {
        oss << hours.count() << "h ";
    }
    if (minutes.count() > 0) {
        oss << minutes.count() << "m ";
    }
    oss << seconds.count() << "s";

    return oss.str();
}

double calculateSuccessRate(uint64_t successful, uint64_t total) {
    return total > 0 ? (static_cast<double>(successful) / total) * 100.0 : 100.0;
}

double calculateAverage(const std::deque<double>& values) {
    if (values.empty()) {
        return 0.0;
    }

    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }

    return sum / values.size();
}

std::string calculateHash(const std::string& data) {
    // Simple hash function - in production, use a proper cryptographic hash
    std::hash<std::string> hasher;
    return std::to_string(hasher(data));
}

} // namespace RegistryUtils

} // namespace Network
} // namespace VortexGPU