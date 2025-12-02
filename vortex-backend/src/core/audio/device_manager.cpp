#include "core/audio/device_manager.hpp"
#include "core/dsp/audio_math.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>
#include <chrono>
#include <random>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <dsound.h>
#include <mmsystem.h>
#endif

#ifdef __APPLE__
#include <CoreAudio/CoreAudio.h>
#include <AudioUnit/AudioUnit.h>
#include <AudioToolbox/AudioServices.h>
#endif

#ifdef __linux__
#include <alsa/asoundlib.h>
#include <pulse/pulseaudio.h>
#include <jack/jack.h>
#include <sys/soundcard.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#endif

namespace vortex {
namespace core {
namespace audio {

// Constants
constexpr int DEFAULT_SAMPLE_RATE = 48000;
constexpr int DEFAULT_BUFFER_SIZE = 512;
constexpr double DEFAULT_LATENCY_MS = 10.0;
constexpr int MONITOR_UPDATE_INTERVAL_MS = 100;
constexpr int DISCOVERY_UPDATE_INTERVAL_MS = 1000;
constexpr int DEVICE_TEST_DURATION_MS = 2000;
constexpr float TEST_TONE_FREQUENCY = 1000.0f; // 1kHz test tone
constexpr float TEST_TONE_AMPLITUDE = 0.5f; // -6dBFS

DeviceManager::DeviceManager() : initialized_(false),
                                 hot_plug_detection_enabled_(true),
                                 device_discovery_active_(false),
                                 discovery_running_(false) {
}

DeviceManager::~DeviceManager() {
    shutdown();
}

bool DeviceManager::initialize(AudioDriverType driver_type) {
    if (initialized_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Initialize the specified driver
    bool driver_initialized = false;

    if (driver_type == AudioDriverType::AUTO) {
        // Try drivers in order of preference
        std::vector<AudioDriverType> preferred_drivers = getPlatformSupportedDrivers();

        for (auto preferred_driver : preferred_drivers) {
            switch (preferred_driver) {
                case AudioDriverType::WASAPI:
                    driver_initialized = initializeWASAPI();
                    break;
                case AudioDriverType::ASIO:
                    driver_initialized = initializeASIO();
                    break;
                case AudioDriverType::CORE_AUDIO:
                    driver_initialized = initializeCoreAudio();
                    break;
                case AudioDriverType::JACK:
                    driver_initialized = initializeJACK();
                    break;
                case AudioDriverType::ALSA:
                    driver_initialized = initializeALSA();
                    break;
                case AudioDriverType::PULSE:
                    driver_initialized = initializePulse();
                    break;
                case AudioDriverType::OSS:
                    driver_initialized = initializeOSS();
                    break;
                default:
                    continue;
            }

            if (driver_initialized) {
                active_driver_ = preferred_driver;
                break;
            }
        }
    } else {
        switch (driver_type) {
            case AudioDriverType::WASAPI:
                driver_initialized = initializeWASAPI();
                break;
            case AudioDriverType::ASIO:
                driver_initialized = initializeASIO();
                break;
            case AudioDriverType::CORE_AUDIO:
                driver_initialized = initializeCoreAudio();
                break;
            case AudioDriverType::JACK:
                driver_initialized = initializeJACK();
                break;
            case AudioDriverType::ALSA:
                driver_initialized = initializeALSA();
                break;
            case AudioDriverType::PULSE:
                driver_initialized = initializePulse();
                break;
            case AudioDriverType::OSS:
                driver_initialized = initializeOSS();
                break;
            default:
                driver_initialized = false;
                break;
        }

        if (driver_initialized) {
            active_driver_ = driver_type;
        }
    }

    if (!driver_initialized) {
        return false;
    }

    // Scan for devices
    scanDevices();

    // Set default devices
    auto devices = getAvailableDevices();
    for (const auto& device : devices) {
        if (device.device_type == AudioDeviceType::INPUT && default_input_device_id_ == 0) {
            default_input_device_id_ = device.id;
        }
        if (device.device_type == AudioDeviceType::OUTPUT && default_output_device_id_ == 0) {
            default_output_device_id_ = device.id;
        }
    }

    initialized_ = true;
    return true;
}

void DeviceManager::shutdown() {
    if (!initialized_) {
        return;
    }

    // Stop device discovery
    stopDeviceDiscovery();

    // Stop all device monitoring
    for (auto& [id, device_state] : devices_) {
        stopDeviceMonitorThread(id);
        closeDevice(id);
    }

    // Clear devices
    devices_.clear();

    // Shutdown drivers
    shutdownDrivers();

    initialized_ = false;
    active_driver_ = AudioDriverType::AUTO;
}

std::vector<DeviceInfo> DeviceManager::scanDevices() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        return {};
    }

    std::vector<DeviceInfo> devices;

    // Scan devices based on active driver
    switch (active_driver_) {
        case AudioDriverType::WASAPI:
            scanWASAPIDevices(devices);
            break;
        case AudioDriverType::ASIO:
            scanASIODevice(devices);
            break;
        case AudioDriverType::CORE_AUDIO:
            scanCoreAudioDevices(devices);
            break;
        case AudioDriverType::JACK:
            scanJACKDevices(devices);
            break;
        case AudioDriverType::ALSA:
            scanALSADevices(devices);
            break;
        case AudioDriverType::PULSE:
            scanPulseDevices(devices);
            break;
        case AudioDriverType::OSS:
            scanOSSDevices(devices);
            break;
        default:
            break;
    }

    // Update internal device list
    std::vector<uint32_t> current_ids;
    for (const auto& device : devices) {
        current_ids.push_back(device.id);

        auto it = devices_.find(device.id);
        if (it != devices_.end()) {
            // Update existing device
            it->second->info = device;
            it->second->info.last_seen = std::chrono::steady_clock::now();
        } else {
            // Add new device
            auto device_state = std::make_unique<DeviceStateInternal>();
            device_state->info = device;
            device_state->info.last_seen = std::chrono::steady_clock::now();
            device_state->monitor = DeviceMonitorInfo{};
            device_state->monitor.last_update = std::chrono::steady_clock::now();
            devices_[device.id] = std::move(device_state);
        }
    }

    // Remove devices that are no longer present
    auto it = devices_.begin();
    while (it != devices_.end()) {
        if (std::find(current_ids.begin(), current_ids.end(), it->first) == current_ids.end()) {
            it = devices_.erase(it);
        } else {
            ++it;
        }
    }

    // Update default devices if they changed
    for (const auto& device : devices) {
        if (device.is_default_input) {
            default_input_device_id_ = device.id;
        }
        if (device.is_default_output) {
            default_output_device_id_ = device.id;
        }
    }

    return devices;
}

std::vector<DeviceInfo> DeviceManager::getAvailableDevices() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<DeviceInfo> devices;
    devices.reserve(devices_.size());

    for (const auto& [id, device_state] : devices_) {
        devices.push_back(device_state->info);
    }

    return devices;
}

std::optional<DeviceInfo> DeviceManager::getDevice(uint32_t device_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = devices_.find(device_id);
    if (it != devices_.end()) {
        return it->second->info;
    }

    return std::nullopt;
}

std::optional<DeviceInfo> DeviceManager::getDefaultInputDevice() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (default_input_device_id_ != 0) {
        return getDevice(default_input_device_id_);
    }

    // Find first input device if no default set
    for (const auto& [id, device_state] : devices_) {
        if (device_state->info.device_type == AudioDeviceType::INPUT ||
            device_state->info.device_type == AudioDeviceType::INPUT_OUTPUT) {
            return device_state->info;
        }
    }

    return std::nullopt;
}

std::optional<DeviceInfo> DeviceManager::getDefaultOutputDevice() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (default_output_device_id_ != 0) {
        return getDevice(default_output_device_id_);
    }

    // Find first output device if no default set
    for (const auto& [id, device_state] : devices_) {
        if (device_state->info.device_type == AudioDeviceType::OUTPUT ||
            device_state->info.device_type == AudioDeviceType::INPUT_OUTPUT) {
            return device_state->info;
        }
    }

    return std::nullopt;
}

bool DeviceManager::setDefaultInputDevice(uint32_t device_id) {
    auto device = getDevice(device_id);
    if (!device || (device->device_type != AudioDeviceType::INPUT && device->device_type != AudioDeviceType::INPUT_OUTPUT)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Clear previous default
    if (default_input_device_id_ != 0) {
        auto prev_device = devices_.find(default_input_device_id_);
        if (prev_device != devices_.end()) {
            prev_device->second->info.is_default_input = false;
        }
    }

    // Set new default
    default_input_device_id_ = device_id;
    devices_[device_id]->info.is_default_input = true;

    return true;
}

bool DeviceManager::setDefaultOutputDevice(uint32_t device_id) {
    auto device = getDevice(device_id);
    if (!device || (device->device_type != AudioDeviceType::OUTPUT && device->device_type != AudioDeviceType::INPUT_OUTPUT)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Clear previous default
    if (default_output_device_id_ != 0) {
        auto prev_device = devices_.find(default_output_device_id_);
        if (prev_device != devices_.end()) {
            prev_device->second->info.is_default_output = false;
        }
    }

    // Set new default
    default_output_device_id_ = device_id;
    devices_[device_id]->info.is_default_output = true;

    return true;
}

bool DeviceManager::setDeviceEnabled(uint32_t device_id, bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = devices_.find(device_id);
    if (it == devices_.end()) {
        return false;
    }

    it->second->info.is_enabled = enabled;

    if (enabled && it->second->info.state == DeviceState::DISCONNECTED) {
        // Try to re-enable the device
        if (openDevice(device_id)) {
            updateDeviceState(device_id, DeviceState::IDLE);
        }
    } else if (!enabled && it->second->info.state != DeviceState::DISCONNECTED) {
        // Disable the device
        closeDevice(device_id);
        updateDeviceState(device_id, DeviceState::DISCONNECTED);
    }

    return true;
}

bool DeviceManager::configureDevice(uint32_t device_id, const DeviceConfiguration& config) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = devices_.find(device_id);
    if (it == devices_.end()) {
        return false;
    }

    // Validate configuration
    if (!validateConfiguration(device_id, config)) {
        return false;
    }

    // Configure device through driver
    if (!configureDeviceDriver(device_id, config)) {
        return false;
    }

    // Update device configuration
    it->second->info.current_config = config;

    return true;
}

bool DeviceManager::testDevice(uint32_t device_id, const DeviceConfiguration& config, int test_duration_ms) {
    auto device = getDevice(device_id);
    if (!device) {
        return false;
    }

    // Validate configuration
    if (!validateConfiguration(device_id, config)) {
        return false;
    }

    // Open device in test mode
    if (!openDevice(device_id)) {
        return false;
    }

    // Configure device
    if (!configureDeviceDriver(device_id, config)) {
        closeDevice(device_id);
        return false;
    }

    // Generate test signal
    int test_samples = (config.sample_rate * test_duration_ms) / 1000;
    std::vector<float> test_signal(test_samples * config.output_channels);

    if (!device_utils::generateTestTone(test_signal.data(), test_samples, config.sample_rate,
                                         TEST_TONE_FREQUENCY, TEST_TONE_AMPLITUDE)) {
        closeDevice(device_id);
        return false;
    }

    // Test device by playing test signal (simplified)
    bool test_passed = true;

    try {
        // Simulate device test
        std::this_thread::sleep_for(std::chrono::milliseconds(test_duration_ms));

        // Check for errors (would be implemented with actual device testing)
        auto monitor_info = getDeviceMonitorInfo(device_id);
        if (monitor_info) {
            // Check if device had excessive dropouts or clipping
            if (monitor_info->dropouts > 10 || monitor_info->clipping_samples > test_samples * 0.01) {
                test_passed = false;
            }
        }
    } catch (const std::exception& e) {
        test_passed = false;
    }

    // Close device
    closeDevice(device_id);

    return test_passed;
}

bool DeviceManager::startDeviceMonitoring(uint32_t device_id, DeviceMonitorCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = devices_.find(device_id);
    if (it == devices_.end()) {
        return false;
    }

    if (it->second->monitoring_active) {
        return true; // Already monitoring
    }

    it->second->monitor_callback = callback;
    it->second->monitoring_active = true;
    it->second->monitor_running = true;

    startDeviceMonitorThread(device_id);

    return true;
}

bool DeviceManager::stopDeviceMonitoring(uint32_t device_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = devices_.find(device_id);
    if (it == devices_.end()) {
        return false;
    }

    if (!it->second->monitoring_active) {
        return true; // Not monitoring
    }

    it->second->monitor_running = false;
    stopDeviceMonitorThread(device_id);
    it->second->monitoring_active = false;
    it->second->monitor_callback = nullptr;

    return true;
}

std::optional<DeviceMonitorInfo> DeviceManager::getDeviceMonitorInfo(uint32_t device_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = devices_.find(device_id);
    if (it == devices_.end()) {
        return std::nullopt;
    }

    return it->second->monitor;
}

std::vector<int> DeviceManager::getDeviceSampleRates(uint32_t device_id) const {
    auto device = getDevice(device_id);
    if (!device) {
        return {};
    }

    return device->capabilities.sample_rates;
}

std::vector<AudioBitDepth> DeviceManager::getDeviceBitDepths(uint32_t device_id) const {
    auto device = getDevice(device_id);
    if (!device) {
        return {};
    }

    return device->capabilities.bit_depths;
}

DeviceConfiguration DeviceManager::getOptimalConfiguration(uint32_t device_id, double target_latency_ms) const {
    auto device = getDevice(device_id);
    if (!device) {
        return {};
    }

    DeviceConfiguration config;
    const auto& caps = device->capabilities;

    // Find best sample rate (prefer 48kHz or higher)
    if (std::find(caps.sample_rates.begin(), caps.sample_rates.end(), 48000) != caps.sample_rates.end()) {
        config.sample_rate = 48000;
    } else if (std::find(caps.sample_rates.begin(), caps.sample_rates.end(), 44100) != caps.sample_rates.end()) {
        config.sample_rate = 44100;
    } else if (!caps.sample_rates.empty()) {
        config.sample_rate = caps.sample_rates[0];
    } else {
        config.sample_rate = DEFAULT_SAMPLE_RATE;
    }

    // Find best bit depth (prefer 32-bit float)
    if (std::find(caps.bit_depths.begin(), caps.bit_depths.end(), AudioBitDepth::FLOAT32) != caps.bit_depths.end()) {
        config.bit_depth = AudioBitDepth::FLOAT32;
    } else if (std::find(caps.bit_depths.begin(), caps.bit_depths.end(), AudioBitDepth::INT24) != caps.bit_depths.end()) {
        config.bit_depth = AudioBitDepth::INT24;
    } else if (!caps.bit_depths.empty()) {
        config.bit_depth = caps.bit_depths[0];
    } else {
        config.bit_depth = AudioBitDepth::FLOAT32;
    }

    // Set sample type based on bit depth
    config.sample_type = DeviceSampleType::FLOAT32; // Default

    // Calculate optimal buffer size for target latency
    config.buffer_size = device_utils::calculateOptimalBufferSize(config.sample_rate, target_latency_ms);

    // Set channel configuration
    if (device->device_type == AudioDeviceType::INPUT) {
        config.input_channels = std::min(2, static_cast<int>(caps.max_input_channels));
        config.output_channels = 0;
    } else if (device->device_type == AudioDeviceType::OUTPUT) {
        config.input_channels = 0;
        config.output_channels = std::min(2, static_cast<int>(caps.max_output_channels));
    } else {
        config.input_channels = std::min(2, static_cast<int>(caps.max_input_channels));
        config.output_channels = std::min(2, static_cast<int>(caps.max_output_channels));
    }

    // Set layout
    if (config.output_channels == 1) {
        config.layout = AudioChannelLayout::MONO;
    } else if (config.output_channels == 2) {
        config.layout = AudioChannelLayout::STEREO;
    } else {
        config.layout = AudioChannelLayout::STEREO; // Default fallback
    }

    // Set latency mode based on target
    if (target_latency_ms <= 5.0) {
        config.latency_mode = DeviceLatencyMode::LOWEST;
    } else if (target_latency_ms <= 10.0) {
        config.latency_mode = DeviceLatencyMode::LOW;
    } else if (target_latency_ms <= 20.0) {
        config.latencyMode = DeviceLatencyMode::MEDIUM;
    } else {
        config.latencyMode = DeviceLatencyMode::HIGH;
    }

    config.target_latency_ms = target_latency_ms;
    config.maximum_latency_ms = std::max(target_latency_ms * 2.0, caps.max_latency_ms);

    return config;
}

bool DeviceManager::synchronizeDevices(uint32_t master_device_id, const std::vector<uint32_t>& slave_device_ids) {
    // Device synchronization implementation would go here
    // This is a simplified placeholder
    return true;
}

bool DeviceManager::desynchronizeDevices(const std::vector<uint32_t>& device_ids) {
    // Device desynchronization implementation would go here
    return true;
}

bool DeviceManager::setHotPlugDetectionEnabled(bool enabled) {
    hot_plug_detection_enabled_ = enabled;

    if (enabled && !device_discovery_active_) {
        startDeviceDiscovery(nullptr);
    } else if (!enabled && device_discovery_active_) {
        stopDeviceDiscovery();
    }

    return true;
}

bool DeviceManager::isHotPlugDetectionEnabled() const {
    return hot_plug_detection_enabled_;
}

bool DeviceManager::startDeviceDiscovery(DeviceChangeCallback callback) {
    if (device_discovery_active_) {
        return true;
    }

    device_change_callback_ = callback;
    device_discovery_active_ = true;
    discovery_running_ = true;

    discovery_thread_ = std::thread(&DeviceManager::deviceDiscoveryLoop, this);

    return true;
}

void DeviceManager::stopDeviceDiscovery() {
    if (!device_discovery_active_) {
        return;
    }

    discovery_running_ = false;

    if (discovery_thread_.joinable()) {
        discovery_thread_.join();
    }

    device_discovery_active_ = false;
    device_change_callback_ = nullptr;
}

bool DeviceManager::isDeviceDiscoveryActive() const {
    return device_discovery_active_;
}

std::vector<DeviceInfo> DeviceManager::refreshDevices() {
    return scanDevices();
}

std::vector<DeviceInfo> DeviceManager::findDevicesByName(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<DeviceInfo> matching_devices;
    std::string lower_name = name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

    for (const auto& [id, device_state] : devices_) {
        std::string device_name = device_state->info.name;
        std::transform(device_name.begin(), device_name.end(), device_name.begin(), ::tolower);

        if (device_name.find(lower_name) != std::string::npos) {
            matching_devices.push_back(device_state->info);
        }
    }

    return matching_devices;
}

std::vector<DeviceInfo> DeviceManager::getDevicesByType(AudioDeviceType type) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<DeviceInfo> matching_devices;

    for (const auto& [id, device_state] : devices_) {
        if (device_state->info.device_type == type) {
            matching_devices.push_back(device_state->info);
        }
    }

    return matching_devices;
}

std::vector<DeviceInfo> DeviceManager::getDevicesByDriver(AudioDriverType driver) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<DeviceInfo> matching_devices;

    for (const auto& [id, device_state] : devices_) {
        if (device_state->info.driver_type == driver) {
            matching_devices.push_back(device_state->info);
        }
    }

    return matching_devices;
}

void DeviceManager::setDeviceStateCallback(DeviceStateCallback callback) {
    device_state_callback_ = callback;
}

std::vector<AudioDriverType> DeviceManager::getSupportedDrivers() const {
    return getPlatformSupportedDrivers();
}

bool DeviceManager::setActiveDriver(AudioDriverType driver) {
    if (initialized_) {
        // Need to reinitialize with new driver
        shutdown();
    }

    return initialize(driver);
}

AudioDriverType DeviceManager::getActiveDriver() const {
    return active_driver_;
}

bool DeviceManager::validateConfiguration(uint32_t device_id, const DeviceConfiguration& config) const {
    auto device = getDevice(device_id);
    if (!device) {
        return false;
    }

    return isConfigurationSupported(device->capabilities, config);
}

bool DeviceManager::resetDevice(uint32_t device_id) {
    auto device = getDevice(device_id);
    if (!device) {
        return false;
    }

    DeviceConfiguration default_config = getOptimalConfiguration(device_id);
    return configureDevice(device_id, default_config);
}

std::string DeviceManager::getDeviceInfoJSON(uint32_t device_id) const {
    auto device = getDevice(device_id);
    if (!device) {
        return "{}";
    }

    return device_utils::deviceInfoToJSON(*device);
}

bool DeviceManager::importConfiguration(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        return false;
    }

    std::string json((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());

    try {
        DeviceConfiguration config = device_utils::deviceConfigFromJSON(json);
        // Apply configuration to appropriate device
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool DeviceManager::exportConfiguration(const std::string& file_path) const {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        return false;
    }

    // Export all device configurations
    file << "{\n";
    file << "  \"devices\": [\n";

    bool first = true;
    for (const auto& [id, device_state] : devices_) {
        if (!first) {
            file << ",\n";
        }
        first = false;

        std::string device_json = device_utils::deviceInfoToJSON(device_state->info);
        file << "    " << device_json;
    }

    file << "\n  ]\n";
    file << "}\n";

    return true;
}

uint32_t DeviceManager::createVirtualDevice(const std::string& name, const DeviceConfiguration& config) {
    // Virtual device creation implementation would go here
    return 0;
}

bool DeviceManager::destroyVirtualDevice(uint32_t device_id) {
    // Virtual device destruction implementation would go here
    return true;
}

// Private methods
bool DeviceManager::initializeWASAPI() {
#ifdef _WIN32
    // WASAPI initialization code would go here
    return true;
#else
    return false;
#endif
}

bool DeviceManager::initializeASIO() {
#ifdef _WIN32
    // ASIO initialization code would go here
    return true;
#else
    return false;
#endif
}

bool DeviceManager::initializeCoreAudio() {
#ifdef __APPLE__
    // CoreAudio initialization code would go here
    return true;
#else
    return false;
#endif
}

bool DeviceManager::initializeJACK() {
#if defined(__linux__) || defined(_WIN32) || defined(__APPLE__)
    // JACK initialization code would go here
    return true;
#else
    return false;
#endif
}

bool DeviceManager::initializeALSA() {
#ifdef __linux__
    // ALSA initialization code would go here
    return true;
#else
    return false;
#endif
}

bool DeviceManager::initializePulse() {
#ifdef __linux__
    // PulseAudio initialization code would go here
    return true;
#else
    return false;
#endif
}

bool DeviceManager::initializeOSS() {
#ifdef __linux__
    // OSS initialization code would go here
    return true;
#else
    return false;
#endif
}

void DeviceManager::shutdownDrivers() {
    // Driver-specific shutdown code would go here
}

void DeviceManager::scanWASAPIDevices(std::vector<DeviceInfo>& devices) {
#ifdef _WIN32
    // WASAPI device scanning implementation would go here
    // This is a simplified placeholder
    DeviceInfo device;
    device.id = next_device_id_++;
    device.name = "WASAPI Default Output";
    device.driver_name = "WASAPI";
    device.driver_type = AudioDriverType::WASAPI;
    device.device_type = AudioDeviceType::OUTPUT;
    device.state = DeviceState::IDLE;
    device.capabilities = getDefaultCapabilities(AudioDeviceType::OUTPUT, AudioDriverType::WASAPI);
    device.current_config = getOptimalConfiguration(device.id);
    devices.push_back(device);
#endif
}

void DeviceManager::scanASIODevice(std::vector<DeviceInfo>& devices) {
#ifdef _WIN32
    // ASIO device scanning implementation would go here
#endif
}

void DeviceManager::scanCoreAudioDevices(std::vector<DeviceInfo>& devices) {
#ifdef __APPLE__
    // CoreAudio device scanning implementation would go here
    DeviceInfo device;
    device.id = next_device_id_++;
    device.name = "Built-in Output";
    device.driver_name = "CoreAudio";
    device.driver_type = AudioDriverType::CORE_AUDIO;
    device.device_type = AudioDeviceType::OUTPUT;
    device.state = DeviceState::IDLE;
    device.capabilities = getDefaultCapabilities(AudioDeviceType::OUTPUT, AudioDriverType::CORE_AUDIO);
    device.current_config = getOptimalConfiguration(device.id);
    device.is_default_output = true;
    devices.push_back(device);
#endif
}

void DeviceManager::scanJACKDevices(std::vector<DeviceInfo>& devices) {
#if defined(__linux__) || defined(_WIN32) || defined(__APPLE__)
    // JACK device scanning implementation would go here
#endif
}

void DeviceManager::scanALSADevices(std::vector<DeviceInfo>& devices) {
#ifdef __linux__
    // ALSA device scanning implementation would go here
    DeviceInfo device;
    device.id = next_device_id_++;
    device.name = "default";
    device.driver_name = "ALSA";
    device.driver_type = AudioDriverType::ALSA;
    device.device_type = AudioDeviceType::INPUT_OUTPUT;
    device.state = DeviceState::IDLE;
    device.capabilities = getDefaultCapabilities(AudioDeviceType::INPUT_OUTPUT, AudioDriverType::ALSA);
    device.current_config = getOptimalConfiguration(device.id);
    device.is_default_input = true;
    device.is_default_output = true;
    devices.push_back(device);
#endif
}

void DeviceManager::scanPulseDevices(std::vector<DeviceInfo>& devices) {
#ifdef __linux__
    // PulseAudio device scanning implementation would go here
#endif
}

void DeviceManager::scanOSSDevices(std::vector<DeviceInfo>& devices) {
#ifdef __linux__
    // OSS device scanning implementation would go here
#endif
}

bool DeviceManager::openDevice(uint32_t device_id) {
    auto it = devices_.find(device_id);
    if (it == devices_.end()) {
        return false;
    }

    switch (active_driver_) {
        case AudioDriverType::WASAPI:
            it->second->driver_handle = openWASAPIDevice(it->second->info);
            break;
        case AudioDriverType::ASIO:
            it->second->driver_handle = openASIODevice(it->second->info);
            break;
        default:
            return false;
    }

    return it->second->driver_handle != 0;
}

bool DeviceManager::closeDevice(uint32_t device_id) {
    auto it = devices_.find(device_id);
    if (it == devices_.end()) {
        return false;
    }

    if (it->second->driver_handle != 0) {
        switch (active_driver_) {
            case AudioDriverType::WASAPI:
                closeWASAPIDevice(it->second->driver_handle);
                break;
            case AudioDriverType::ASIO:
                closeASIODevice(it->second->driver_handle);
                break;
            default:
                break;
        }

        it->second->driver_handle = 0;
    }

    return true;
}

bool DeviceManager::configureDeviceDriver(uint32_t device_id, const DeviceConfiguration& config) {
    auto it = devices_.find(device_id);
    if (it == devices_.end() || it->second->driver_handle == 0) {
        return false;
    }

    switch (active_driver_) {
        case AudioDriverType::WASAPI:
            return configureWASAPIDevice(it->second->driver_handle, config);
        case AudioDriverType::ASIO:
            return configureASIODevice(it->second->driver_handle, config);
        default:
            return false;
    }
}

void DeviceManager::updateDeviceState(uint32_t device_id, DeviceState new_state) {
    auto it = devices_.find(device_id);
    if (it == devices_.end()) {
        return;
    }

    DeviceState old_state = it->second->info.state;
    it->second->info.state = new_state;

    if (device_state_callback_) {
        device_state_callback_(device_id, old_state, new_state);
    }
}

void DeviceManager::startDeviceMonitorThread(uint32_t device_id) {
    auto it = devices_.find(device_id);
    if (it == devices_.end()) {
        return;
    }

    if (it->second->monitor_thread.joinable()) {
        return;
    }

    it->second->monitor_thread = std::thread(&DeviceManager::deviceMonitorLoop, this, device_id);
}

void DeviceManager::stopDeviceMonitorThread(uint32_t device_id) {
    auto it = devices_.find(device_id);
    if (it == devices_.end()) {
        return;
    }

    if (it->second->monitor_thread.joinable()) {
        it->second->monitor_thread.join();
    }
}

void DeviceManager::deviceMonitorLoop(uint32_t device_id) {
    auto it = devices_.find(device_id);
    if (it == devices_.end()) {
        return;
    }

    while (it->second->monitor_running) {
        updateDeviceMonitorInfo(device_id);

        if (it->second->monitor_callback) {
            it->second->monitor_callback(device_id, it->second->monitor);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(MONITOR_UPDATE_INTERVAL_MS));
    }
}

void DeviceManager::updateDeviceMonitorInfo(uint32_t device_id) {
    auto it = devices_.find(device_id);
    if (it == devices_.end()) {
        return;
    }

    DeviceMonitorInfo& monitor = it->second->monitor;

    switch (active_driver_) {
        case AudioDriverType::WASAPI:
            monitor = getWASAPIMonitorInfo(it->second->driver_handle);
            break;
        case AudioDriverType::ASIO:
            monitor = getASIOMonitorInfo(it->second->driver_handle);
            break;
        default:
            // Default monitoring implementation
            monitor.state = it->second->info.state;
            monitor.last_update = std::chrono::steady_clock::now();
            break;
    }
}

void DeviceManager::deviceDiscoveryLoop() {
    while (discovery_running_) {
        auto current_devices = scanDevices();
        detectDeviceChanges(current_devices);

        std::this_thread::sleep_for(std::chrono::milliseconds(DISCOVERY_UPDATE_INTERVAL_MS));
    }
}

void DeviceManager::detectDeviceChanges(const std::vector<DeviceInfo>& current_devices) {
    // This would compare current devices with stored devices and call callbacks
    // Implementation would go here
}

DeviceCapabilities DeviceManager::getDefaultCapabilities(AudioDeviceType type, AudioDriverType driver) const {
    DeviceCapabilities caps;

    // Common sample rates
    caps.sample_rates = {8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 176400, 192000};

    // Common bit depths
    caps.bit_depths = {AudioBitDepth::INT16, AudioBitDepth::INT24, AudioBitDepth::INT32,
                       AudioBitDepth::FLOAT32};

    // Common sample types
    caps.sample_types = {DeviceSampleType::INT16, DeviceSampleType::INT24,
                        DeviceSampleType::INT32, DeviceSampleType::FLOAT32};

    // Common buffer sizes
    caps.buffer_sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096};

    // Common layouts
    caps.layouts = {AudioChannelLayout::MONO, AudioChannelLayout::STEREO,
                   AudioChannelLayout::TWO_POINT_ONE, AudioChannelLayout::FIVE_POINT_ONE,
                   AudioChannelLayout::SEVEN_POINT_ONE};

    // Channel counts
    switch (type) {
        case AudioDeviceType::INPUT:
            caps.max_input_channels = 8;
            caps.max_output_channels = 0;
            break;
        case AudioDeviceType::OUTPUT:
            caps.max_input_channels = 0;
            caps.max_output_channels = 8;
            break;
        case AudioDeviceType::INPUT_OUTPUT:
            caps.max_input_channels = 8;
            caps.max_output_channels = 8;
            break;
        default:
            caps.max_input_channels = 8;
            caps.max_output_channels = 8;
            break;
    }

    caps.max_total_channels = caps.max_input_channels + caps.max_output_channels;

    // Latency
    caps.min_latency_ms = 1.0;
    caps.max_latency_ms = 500.0;

    // Features
    caps.supports_shared_mode = true;
    caps.supports_event_driven = true;
    caps.supports_analog_io = true;
    caps.supports_midi = false;
    caps.supports_digital_io = (driver == AudioDriverType::ASIO);
    caps.supports_low_latency = (driver == AudioDriverType::ASIO || driver == AudioDriverType::WASAPI);

    // Clock sources
    caps.clock_sources = {DeviceClockSource::INTERNAL, DeviceClockSource::AUTOMATIC};

    // Audio quality (typical values)
    caps.dynamic_range_db = {60.0, 120.0, 96.0}; // Min, max, typical
    caps.thd_percent = 0.01;
    caps.snr_db = 90.0;

    return caps;
}

std::vector<AudioDriverType> DeviceManager::getPlatformSupportedDrivers() const {
    std::vector<AudioDriverType> drivers;

#ifdef _WIN32
    drivers.push_back(AudioDriverType::WASAPI);
    drivers.push_back(AudioDriverType::ASIO);
    drivers.push_back(AudioDriverType::DIRECTSOUND);
#endif

#ifdef __APPLE__
    drivers.push_back(AudioDriverType::CORE_AUDIO);
    drivers.push_back(AudioDriverType::JACK);
#endif

#ifdef __linux__
    drivers.push_back(AudioDriverType::ALSA);
    drivers.push_back(AudioDriverType::PULSE);
    drivers.push_back(AudioDriverType::JACK);
    drivers.push_back(AudioDriverType::OSS);
#endif

    // Add cross-platform drivers
    drivers.push_back(AudioDriverType::JACK);
    drivers.push_back(AudioDriverType::VIRTUAL);

    return drivers;
}

std::string DeviceManager::generateDeviceGUID(const DeviceInfo& device) const {
    // Generate a unique GUID for the device based on its properties
    std::string guid_str = device.driver_name + device.name + device.manufacturer + device.serial_number;

    std::array<uint8_t, 16> guid{};
    for (size_t i = 0; i < guid_str.length() && i < 16; ++i) {
        guid[i] = static_cast<uint8_t>(guid_str[i]);
    }

    return std::string(reinterpret_cast<const char*>(guid.data()), guid.size());
}

bool DeviceManager::compareDeviceGUID(const std::array<uint8_t, 16>& guid1, const std::array<uint8_t, 16>& guid2) const {
    return std::equal(guid1.begin(), guid1.end(), guid2.begin());
}

DeviceState DeviceManager::calculateDeviceState(const DeviceInfo& device) const {
    if (!device.is_enabled) {
        return DeviceState::DISABLED;
    }

    if (device.monitor_info.cpu_usage_percent > 90.0 || device.monitor_info.dropouts > 100) {
        return DeviceState::ERROR;
    }

    switch (device.monitor_info.state) {
        case DeviceState::ACTIVE:
        case DeviceState::IDLE:
            return device.monitor_info.state;
        default:
            return DeviceState::UNKNOWN;
    }
}

double DeviceManager::calculateOptimalLatency(const DeviceCapabilities& caps, double target_latency) const {
    return std::max(caps.min_latency_ms, std::min(target_latency, caps.max_latency_ms));
}

bool DeviceManager::isConfigurationSupported(const DeviceCapabilities& caps, const DeviceConfiguration& config) const {
    // Check sample rate
    if (std::find(caps.sample_rates.begin(), caps.sample_rates.end(), config.sample_rate) == caps.sample_rates.end()) {
        return false;
    }

    // Check bit depth
    if (std::find(caps.bit_depths.begin(), caps.bit_depths.end(), config.bit_depth) == caps.bit_depths.end()) {
        return false;
    }

    // Check buffer size
    if (std::find(caps.buffer_sizes.begin(), caps.buffer_sizes.end(), config.buffer_size) == caps.buffer_sizes.end()) {
        return false;
    }

    // Check channel counts
    if (config.input_channels > static_cast<int>(caps.max_input_channels) ||
        config.output_channels > static_cast<int>(caps.max_output_channels)) {
        return false;
    }

    // Check layout
    if (std::find(caps.layouts.begin(), caps.layouts.end(), config.layout) == caps.layouts.end()) {
        return false;
    }

    // Check latency
    double actual_latency = calculateBufferSizeMs(config.buffer_size, config.sample_rate);
    if (actual_latency < caps.min_latency_ms || actual_latency > caps.max_latency_ms) {
        return false;
    }

    // Check exclusive mode compatibility
    if (config.exclusive_mode && !caps.supports_exclusive_mode) {
        return false;
    }

    return true;
}

// Driver-specific helper methods (simplified placeholders)
uint32_t DeviceManager::openWASAPIDevice(const DeviceInfo& device) {
    // WASAPI device opening implementation
    return 1; // Return driver handle
}

bool DeviceManager::configureWASAPIDevice(uint32_t driver_handle, const DeviceConfiguration& config) {
    // WASAPI device configuration implementation
    return true;
}

void DeviceManager::closeWASAPIDevice(uint32_t driver_handle) {
    // WASAPI device closing implementation
}

DeviceMonitorInfo DeviceManager::getWASAPIMonitorInfo(uint32_t driver_handle) {
    DeviceMonitorInfo monitor;
    monitor.state = DeviceState::ACTIVE;
    monitor.last_update = std::chrono::steady_clock::now();
    return monitor;
}

uint32_t DeviceManager::openASIODevice(const DeviceInfo& device) {
    // ASIO device opening implementation
    return 2;
}

bool DeviceManager::configureASIODevice(uint32_t driver_handle, const DeviceConfiguration& config) {
    // ASIO device configuration implementation
    return true;
}

void DeviceManager::closeASIODevice(uint32_t driver_handle) {
    // ASIO device closing implementation
}

DeviceMonitorInfo DeviceManager::getASIOMonitorInfo(uint32_t driver_handle) {
    DeviceMonitorInfo monitor;
    monitor.state = DeviceState::ACTIVE;
    monitor.last_update = std::chrono::steady_clock::now();
    return monitor;
}

// DeviceManagerFactory implementation
std::unique_ptr<DeviceManager> DeviceManagerFactory::createPlatformDeviceManager() {
    AudioDriverType recommended = getRecommendedDriver();
    return createDeviceManager(recommended);
}

std::unique_ptr<DeviceManager> DeviceManagerFactory::createDeviceManager(AudioDriverType driver) {
    auto manager = std::make_unique<DeviceManager>();
    if (manager->initialize(driver)) {
        return manager;
    }
    return nullptr;
}

AudioDriverType DeviceManagerFactory::getRecommendedDriver() {
    auto drivers = getAvailableDrivers();
    if (drivers.empty()) {
        return AudioDriverType::AUTO;
    }

#ifdef _WIN32
    if (std::find(drivers.begin(), drivers.end(), AudioDriverType::WASAPI) != drivers.end()) {
        return AudioDriverType::WASAPI;
    }
#endif

#ifdef __APPLE__
    if (std::find(drivers.begin(), drivers.end(), AudioDriverType::CORE_AUDIO) != drivers.end()) {
        return AudioDriverType::CORE_AUDIO;
    }
#endif

#ifdef __linux__
    if (std::find(drivers.begin(), drivers.end(), AudioDriverType::PULSE) != drivers.end()) {
        return AudioDriverType::PULSE;
    }
#endif

    return drivers[0]; // Return first available
}

std::vector<AudioDriverType> DeviceManagerFactory::getAvailableDrivers() {
    return DeviceManagerFactory::detectAvailableDrivers();
}

bool DeviceManagerFactory::isDriverAvailable(AudioDriverType driver) {
    auto drivers = detectAvailableDrivers();
    return std::find(drivers.begin(), drivers.end(), driver) != drivers.end();
}

std::vector<AudioDriverType> DeviceManagerFactory::detectAvailableDrivers() {
    std::vector<AudioDriverType> drivers;

#ifdef _WIN32
    drivers.push_back(AudioDriverType::WASAPI);
    // ASIO detection would require checking for ASIO drivers
    drivers.push_back(AudioDriverType::DIRECTSOUND);
#endif

#ifdef __APPLE__
    drivers.push_back(AudioDriverType::CORE_AUDIO);
    // JACK detection would require checking if JACK is installed
#endif

#ifdef __linux__
    drivers.push_back(AudioDriverType::ALSA);
    // PulseAudio detection would require checking if PulseAudio is running
    // JACK detection would require checking if JACK is installed
#endif

    // Always available
    drivers.push_back(AudioDriverType::VIRTUAL);

    return drivers;
}

// Utility functions implementation
namespace device_utils {

std::string driverTypeToString(AudioDriverType driver) {
    switch (driver) {
        case AudioDriverType::AUTO: return "Auto";
        case AudioDriverType::WASAPI: return "WASAPI";
        case AudioDriverType::ASIO: return "ASIO";
        case AudioDriverType::DIRECTSOUND: return "DirectSound";
        case AudioDriverType::CORE_AUDIO: return "CoreAudio";
        case AudioDriverType::JACK: return "JACK";
        case AudioDriverType::ALSA: return "ALSA";
        case AudioDriverType::PULSE: return "PulseAudio";
        case AudioDriverType::OSS: return "OSS";
        case AudioDriverType::VIRTUAL: return "Virtual";
        case AudioDriverType::NETWORK: return "Network";
        default: return "Unknown";
    }
}

std::string deviceStateToString(DeviceState state) {
    switch (state) {
        case DeviceState::UNKNOWN: return "Unknown";
        case DeviceState::ACTIVE: return "Active";
        case DeviceState::IDLE: return "Idle";
        case DeviceState::DISCONNECTED: return "Disconnected";
        case DeviceState::ERROR: return "Error";
        case DeviceState::CONFIGURING: return "Configuring";
        case DeviceState::STARTING: return "Starting";
        case DeviceState::STOPPING: return "Stopping";
        case DeviceState::SUSPENDED: return "Suspended";
        default: return "Unknown";
    }
}

std::string latencyModeToString(DeviceLatencyMode mode) {
    switch (mode) {
        case DeviceLatencyMode::LOWEST: return "Lowest";
        case DeviceLatencyMode::LOW: return "Low";
        case DeviceLatencyMode::MEDIUM: return "Medium";
        case DeviceLatencyMode::HIGH: return "High";
        case DeviceLatencyMode::HIGHEST: return "Highest";
        case DeviceLatencyMode::CUSTOM: return "Custom";
        default: return "Unknown";
    }
}

std::string sampleTypeToString(DeviceSampleType type) {
    switch (type) {
        case DeviceSampleType::INT8: return "Int8";
        case DeviceSampleType::UINT8: return "UInt8";
        case DeviceSampleType::INT16: return "Int16";
        case DeviceSampleType::UINT16: return "UInt16";
        case DeviceSampleType::INT24: return "Int24";
        case DeviceSampleType::INT32: return "Int32";
        case DeviceSampleType::UINT32: return "UInt32";
        case DeviceSampleType::FLOAT32: return "Float32";
        case DeviceSampleType::FLOAT64: return "Float64";
        case DeviceSampleType::DSD8: return "DSD8";
        case DeviceSampleType::DSD16: return "DSD16";
        case DeviceSampleType::DSD32: return "DSD32";
        case DeviceSampleType::CUSTOM: return "Custom";
        default: return "Unknown";
    }
}

std::string clockSourceToString(DeviceClockSource source) {
    switch (source) {
        case DeviceClockSource::INTERNAL: return "Internal";
        case DeviceClockSource::EXTERNAL_WORD: return "External Word Clock";
        case DeviceClockSource::EXTERNAL_SPDIF: return "External S/PDIF";
        case DeviceClockSource::EXTERNAL_ADAT: return "External ADAT";
        case DeviceClockSource::EXTERNAL_AES_EBU: return "External AES/EBU";
        case DeviceClockSource::EXTERNAL_MIDI: return "External MIDI";
        case DeviceClockSource::NETWORK: return "Network";
        case DeviceClockSource::AUTOMATIC: return "Automatic";
        default: return "Unknown";
    }
}

AudioDriverType stringToDriverType(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "wasapi") return AudioDriverType::WASAPI;
    if (lower == "asio") return AudioDriverType::ASIO;
    if (lower == "directsound") return AudioDriverType::DIRECTSOUND;
    if (lower == "coreaudio" || lower == "core audio") return AudioDriverType::CORE_AUDIO;
    if (lower == "jack") return AudioDriverType::JACK;
    if (lower == "alsa") return AudioDriverType::ALSA;
    if (lower == "pulse" || lower == "pulseaudio") return AudioDriverType::PULSE;
    if (lower == "oss") return AudioDriverType::OSS;
    if (lower == "virtual") return AudioDriverType::VIRTUAL;
    if (lower == "network") return AudioDriverType::NETWORK;

    return AudioDriverType::AUTO;
}

DeviceState stringToDeviceState(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "active") return DeviceState::ACTIVE;
    if (lower == "idle") return DeviceState::IDLE;
    if (lower == "disconnected") return DeviceState::DISCONNECTED;
    if (lower == "error") return DeviceState::ERROR;
    if (lower == "configuring") return DeviceState::CONFIGURING;
    if (lower == "starting") return DeviceState::STARTING;
    if (lower == "stopping") return DeviceState::STOPPING;
    if (lower == "suspended") return DeviceState::SUSPENDED;

    return DeviceState::UNKNOWN;
}

DeviceLatencyMode stringToLatencyMode(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "lowest") return DeviceLatencyMode::LOWEST;
    if (lower == "low") return DeviceLatencyMode::LOW;
    if (lower == "medium") return DeviceLatencyMode::MEDIUM;
    if (lower == "high") return DeviceLatencyMode::HIGH;
    if (lower == "highest") return DeviceLatencyMode::HIGHEST;
    if (lower == "custom") return DeviceLatencyMode::CUSTOM;

    return DeviceLatencyMode::LOW;
}

bool isCompatibleConfiguration(const DeviceCapabilities& caps, const DeviceConfiguration& config) {
    // Check sample rate
    if (std::find(caps.sample_rates.begin(), caps.sample_rates.end(), config.sample_rate) == caps.sample_rates.end()) {
        return false;
    }

    // Check bit depth
    if (std::find(caps.bit_depths.begin(), caps.bit_depths.end(), config.bit_depth) == caps.bit_depths.end()) {
        return false;
    }

    // Check channel counts
    if (config.input_channels > static_cast<int>(caps.max_input_channels) ||
        config.output_channels > static_cast<int>(caps.max_output_channels)) {
        return false;
    }

    // Check buffer size (if supported)
    if (!caps.buffer_sizes.empty() &&
        std::find(caps.buffer_sizes.begin(), caps.buffer_sizes.end(), config.buffer_size) == caps.buffer_sizes.end()) {
        return false;
    }

    // Check layout
    if (std::find(caps.layouts.begin(), caps.layouts.end(), config.layout) == caps.layouts.end()) {
        return false;
    }

    // Check exclusive mode
    if (config.exclusive_mode && !caps.supports_exclusive_mode) {
        return false;
    }

    return true;
}

DeviceConfiguration findBestMatch(const DeviceCapabilities& caps, const std::vector<DeviceConfiguration>& options) {
    for (const auto& option : options) {
        if (isCompatibleConfiguration(caps, option)) {
            return option;
        }
    }

    // Return default configuration if no match found
    DeviceConfiguration default_config;
    default_config.sample_rate = caps.sample_rates.empty() ? 48000 : caps.sample_rates[0];
    default_config.bit_depth = caps.bit_depths.empty() ? AudioBitDepth::FLOAT32 : caps.bit_depths[0];
    default_config.buffer_size = caps.buffer_sizes.empty() ? 512 : caps.buffer_sizes[0];
    default_config.input_channels = std::min(2, static_cast<int>(caps.max_input_channels));
    default_config.output_channels = std::min(2, static_cast<int>(caps.max_output_channels));
    default_config.layout = caps.layouts.empty() ? AudioChannelLayout::STEREO : caps.layouts[0];

    return default_config;
}

std::vector<int> getCommonSampleRates(const std::vector<int>& rates1, const std::vector<int>& rates2) {
    std::vector<int> common_rates;
    std::set_intersection(rates1.begin(), rates1.end(),
                         rates2.begin(), rates2.end(),
                         std::back_inserter(common_rates));
    return common_rates;
}

double calculateBufferSizeMs(int buffer_size, int sample_rate) {
    return (static_cast<double>(buffer_size) / sample_rate) * 1000.0;
}

double calculateSampleRatePeriodMs(int sample_rate) {
    return 1000.0 / static_cast<double>(sample_rate);
}

double calculateBitsPerSecond(AudioBitDepth bit_depth, int sample_rate, int channels) {
    int bits_per_sample = 0;
    switch (bit_depth) {
        case AudioBitDepth::INT16: bits_per_sample = 16; break;
        case AudioBitDepth::INT24: bits_per_sample = 24; break;
        case AudioBitDepth::INT32:
        case AudioBitDepth::FLOAT32: bits_per_sample = 32; break;
        case AudioBitDepth::FLOAT64: bits_per_sample = 64; break;
        default: bits_per_sample = 32; break;
    }

    return static_cast<double>(bits_per_sample * sample_rate * channels);
}

double calculateBytesPerSecond(const DeviceConfiguration& config) {
    double bits_per_second = calculateBitsPerSecond(config.bit_depth, config.sample_rate,
                                                   config.input_channels + config.output_channels);
    return bits_per_second / 8.0;
}

int calculateOptimalBufferSize(int sample_rate, double target_latency_ms) {
    int buffer_size = static_cast<int>((target_latency_ms / 1000.0) * sample_rate);

    // Round to nearest power of 2
    int power_of_2 = 1;
    while (power_of_2 < buffer_size) {
        power_of_2 *= 2;
    }

    // Check if we should round down instead
    if ((power_of_2 - buffer_size) > (buffer_size - power_of_2 / 2)) {
        power_of_2 /= 2;
    }

    return std::max(32, std::min(8192, power_of_2));
}

bool generateTestTone(float* buffer, size_t samples, int sample_rate, float frequency, float amplitude) {
    if (!buffer || samples == 0) {
        return false;
    }

    for (size_t i = 0; i < samples; ++i) {
        double phase = 2.0 * M_PI * frequency * i / sample_rate;
        buffer[i] = static_cast<float>(amplitude * std::sin(phase));
    }

    return true;
}

bool analyzeTestSignal(const float* signal, size_t samples, int sample_rate, float& thd, float& snr) {
    if (!signal || samples == 0) {
        return false;
    }

    // Simplified THD and SNR calculation
    // Real implementation would use FFT analysis

    float signal_power = 0.0f;
    float noise_power = 0.0f;

    for (size_t i = 0; i < samples; ++i) {
        signal_power += signal[i] * signal[i];
    }

    signal_power /= static_cast<float>(samples);

    // Estimate noise (simplified)
    noise_power = signal_power * 0.001f; // Assume -60dB noise floor

    if (noise_power > 0.0f) {
        snr = 20.0f * std::log10f(signal_power / noise_power);
    } else {
        snr = 120.0f; // Perfect SNR
    }

    // THD calculation would require harmonic analysis
    thd = 0.01f; // 1% THD (placeholder)

    return true;
}

float calculateSignalToNoiseRatio(const float* signal, const float* noise, size_t samples) {
    if (!signal || !noise || samples == 0) {
        return 0.0f;
    }

    float signal_power = 0.0f;
    float noise_power = 0.0f;

    for (size_t i = 0; i < samples; ++i) {
        signal_power += signal[i] * signal[i];
        noise_power += noise[i] * noise[i];
    }

    signal_power /= static_cast<float>(samples);
    noise_power /= static_cast<float>(samples);

    if (noise_power > 0.0f) {
        return 20.0f * std::log10f(signal_power / noise_power);
    }

    return 120.0f; // Perfect SNR
}

float calculateTotalHarmonicDistortion(const float* signal, size_t samples, int sample_rate, int num_harmonics) {
    if (!signal || samples == 0) {
        return 0.0f;
    }

    // Simplified THD calculation
    // Real implementation would perform FFT and analyze harmonics
    return 0.01f; // 1% THD (placeholder)
}

std::string deviceInfoToJSON(const DeviceInfo& device) {
    std::ostringstream json;
    json << "{\n";
    json << "  \"id\": " << device.id << ",\n";
    json << "  \"name\": \"" << device.name << "\",\n";
    json << "  \"driver_name\": \"" << device.driver_name << "\",\n";
    json << "  \"driver_type\": \"" << driverTypeToString(device.driver_type) << "\",\n";
    json << "  \"device_type\": \"" << device_utils::getDeviceTypeString(device.device_type) << "\",\n";
    json << "  \"state\": \"" << deviceStateToString(device.state) << "\",\n";
    json << "  \"is_enabled\": " << (device.is_enabled ? "true" : "false") << ",\n";
    json << "  \"is_default_input\": " << (device.is_default_input ? "true" : "false") << ",\n";
    json << "  \"is_default_output\": " << (device.is_default_output ? "true" : "false") << "\n";
    json << "}";

    return json.str();
}

std::string deviceConfigToJSON(const DeviceConfiguration& config) {
    std::ostringstream json;
    json << "{\n";
    json << "  \"sample_rate\": " << config.sample_rate << ",\n";
    json << "  \"bit_depth\": \"" << getBitDepthString(config.bit_depth) << "\",\n";
    json << "  \"buffer_size\": " << config.buffer_size << ",\n";
    json << "  \"input_channels\": " << config.input_channels << ",\n";
    json << "  \"output_channels\": " << config.output_channels << ",\n";
    json << "  \"layout\": \"" << getLayoutString(config.layout) << "\"\n";
    json << "}";

    return json.str();
}

std::string deviceMonitorToJSON(const DeviceMonitorInfo& monitor) {
    std::ostringstream json;
    json << "{\n";
    json << "  \"input_peak_dbfs\": " << monitor.input_peak_dbfs << ",\n";
    json << "  \"output_peak_dbfs\": " << monitor.output_peak_dbfs << ",\n";
    json << "  \"input_rms_dbfs\": " << monitor.input_rms_dbfs << ",\n";
    json << "  \"output_rms_dbfs\": " << monitor.output_rms_dbfs << ",\n";
    json << "  \"state\": \"" << deviceStateToString(monitor.state) << "\",\n";
    json << "  \"dropouts\": " << monitor.dropouts << ",\n";
    json << "  \"cpu_usage_percent\": " << monitor.cpu_usage_percent << "\n";
    json << "}";

    return json.str();
}

DeviceInfo deviceInfoFromJSON(const std::string& json) {
    // JSON parsing implementation would go here
    // This is a simplified placeholder
    DeviceInfo device;
    return device;
}

DeviceConfiguration deviceConfigFromJSON(const std::string& json) {
    // JSON parsing implementation would go here
    // This is a simplified placeholder
    DeviceConfiguration config;
    return config;
}

std::string getPlatformName() {
#ifdef _WIN32
    return "Windows";
#elif __APPLE__
    return "macOS";
#elif __linux__
    return "Linux";
#else
    return "Unknown";
#endif
}

std::string getSystemAudioAPI() {
#ifdef _WIN32
    return "WASAPI";
#elif __APPLE__
    return "CoreAudio";
#elif __linux__
    return "ALSA";
#else
    return "Unknown";
#endif
}

bool isWindows() {
#ifdef _WIN32
    return true;
#else
    return false;
#endif
}

bool isMacOS() {
#ifdef __APPLE__
    return true;
#else
    return false;
#endif
}

bool isLinux() {
#ifdef __linux__
    return true;
#else
    return false;
#endif
}

bool isUnix() {
#if defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__))
    return true;
#else
    return false;
#endif
}

} // namespace device_utils

} // namespace audio
} // namespace core
} // namespace vortex