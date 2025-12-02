#include "output_manager.hpp"
#include "../core/audio_engine.hpp"
#include "../core/gpu/gpu_processor.hpp"
#include "../utils/logger.hpp"

#include <algorithm>
#include <random>
#include <sstream>
#include <iomanip>

#if defined(_WIN32)
#include <windows.h>
#include <mmsystem.h>
#include <dsound.h>
#include <asio.hpp>
#elif defined(__APPLE__)
#include <CoreAudio/CoreAudio.h>
#include <AudioToolbox/AudioToolbox.h>
#elif defined(__linux__)
#include <alsa/asoundlib.h>
#include <pulse/simple.h>
#include <jack/jack.h>
#endif

namespace vortex {

// Roon Bridge Server Implementation
class OutputManager::RoonBridgeServer {
public:
    struct RoonDeviceInfo {
        std::string deviceId;
        std::string displayName;
        std::pair<std::string, std::string> ipAddress;
        uint16_t port;
        bool isActive = false;
        std::string zoneName;
        std::map<std::string, std::string> capabilities;
    };

    RoonBridgeServer(const RoonBridgeConfig& config) : config_(config) {}
    virtual ~RoonBridgeServer() = default;

    virtual bool start() = 0;
    virtual void stop() = 0;
    virtual bool isActive() const = 0;
    virtual void setVolume(float volume) = 0;
    virtual void setMute(bool muted) = 0;

    RoonDeviceInfo getInfo() const {
        RoonDeviceInfo info;
        info.deviceId = config_.deviceId;
        info.displayName = config_.displayName;
        info.ipAddress = config_.ipAddress;
        info.port = config_.port;
        info.isActive = isActive();
        info.zoneName = config_.zoneName;

        info.capabilities["max_sample_rate"] = std::to_string(config_.maxSampleRate);
        info.capabilities["support_dsd"] = config_.supportDSD ? "true" : "false";
        info.capabilities["support_mqa"] = config_.supportMQA ? "true" : "false";
        info.capabilities["enable_raat"] = config_.enableRAAT ? "true" : "false";
        info.capabilities["enable_airplay"] = config_.enableAirPlay ? "true" : "false";

        return info;
    }

protected:
    RoonBridgeConfig config_;
    std::atomic<bool> running_{false};
};

// HQPlayer NAA Client Implementation
class OutputManager::HQPlayerNAAClient {
public:
    HQPlayerNAAClient(const HQPlayerConfig& config) : config_(config) {}
    virtual ~HQPlayerNAAClient() = default;

    virtual bool connect() = 0;
    virtual void disconnect() = 0;
    virtual bool isConnected() const = 0;
    virtual bool sendAudioFrame(const AudioFrame& frame) = 0;
    virtual void setVolume(float volume) = 0;

protected:
    HQPlayerConfig config_;
    std::atomic<bool> connected_{false};
};

// UPnP Renderer Implementation
class OutputManager::UPnPRenderer {
public:
    UPnPRenderer(const UPnPConfig& config) : config_(config) {}
    virtual ~UPnPRenderer() = default;

    virtual bool start() = 0;
    virtual void stop() = 0;
    virtual bool isRunning() const = 0;
    virtual bool streamAudio(const AudioFrame& frame) = 0;

protected:
    UPnPConfig config_;
    std::atomic<bool> running_{false};
};

// ASIO Device Implementation
class OutputManager::ASIODevice {
public:
    ASIODevice(const std::string& deviceId) : deviceId_(deviceId) {}
    virtual ~ASIODevice() = default;

    virtual bool initialize(uint32_t sampleRate, uint16_t channels, uint32_t bufferSize) = 0;
    virtual void shutdown() = 0;
    virtual bool start() = 0;
    virtual void stop() = 0;
    virtual bool writeAudio(const float* buffer, uint32_t frames) = 0;
    virtual void setVolume(float volume) = 0;

protected:
    std::string deviceId_;
    std::atomic<bool> active_{false};
};

// WASAPI Device Implementation
class OutputManager::WASAPIDevice {
public:
    WASAPIDevice(const std::string& deviceId) : deviceId_(deviceId) {}
    virtual ~WASAPIDevice() = default;

    virtual bool initialize(uint32_t sampleRate, uint16_t channels, uint32_t bufferSize) = 0;
    virtual void shutdown() = 0;
    virtual bool start() = 0;
    virtual void stop() = 0;
    virtual bool writeAudio(const float* buffer, uint32_t frames) = 0;
    virtual void setVolume(float volume) = 0;

protected:
    std::string deviceId_;
    std::atomic<bool> active_{false};
};

// OutputManager Implementation
OutputManager::OutputManager() {
    Logger::info("OutputManager: Initializing multi-device output manager");

    statistics_.startTime = std::chrono::steady_clock::now();
    statistics_.lastActivity = statistics_.startTime;
}

OutputManager::~OutputManager() {
    shutdown();
    Logger::info("OutputManager: Multi-device output manager destroyed");
}

bool OutputManager::initialize(std::shared_ptr<AudioEngine> audioEngine) {
    if (initialized_.load()) {
        Logger::warn("OutputManager: Already initialized");
        return true;
    }

    Logger::info("OutputManager: Initializing with audio engine");

    audioEngine_ = audioEngine;

    // Start background threads
    shouldShutdown_.store(false);

    try {
        discoveryThread_ = std::make_unique<std::thread>(&OutputManager::deviceDiscoveryThread, this);
        monitoringThread_ = std::make_unique<std::thread>(&OutputManager::deviceMonitoringThread, this);
        processingThread_ = std::make_unique<std::thread>(&OutputManager::audioProcessingThread, this);

        initialized_.store(true);
        Logger::info("OutputManager: Initialization complete");
        return true;

    } catch (const std::exception& e) {
        setError("Failed to initialize threads: " + std::string(e.what()));
        return false;
    }
}

void OutputManager::shutdown() {
    if (!initialized_.load()) {
        return;
    }

    Logger::info("OutputManager: Shutting down");

    shouldShutdown_.store(true);
    processingActive_.store(false);

    // Stop all devices
    {
        std::lock_guard<std::mutex> lock(devicesMutex_);
        for (auto& [deviceId, device] : devices_) {
            stopOutput(deviceId);
        }
    }

    // Wake up threads
    audioQueueCondition_.notify_all();

    // Wait for threads to finish
    if (discoveryThread_ && discoveryThread_->joinable()) {
        discoveryThread_->join();
    }
    if (monitoringThread_ && monitoringThread_->joinable()) {
        monitoringThread_->join();
    }
    if (processingThread_ && processingThread_->joinable()) {
        processingThread_->join();
    }

    // Clean up devices
    {
        std::lock_guard<std::mutex> lock(devicesMutex_);
        devices_.clear();
        deviceConfigs_.clear();

        // Clean up device-specific implementations
        asioDevices_.clear();
        wasapiDevices_.clear();
        coreAudioDevices_.clear();
        alsaDevices_.clear();
        pulseAudioDevices_.clear();
        jackDevices_.clear();
        roonBridges_.clear();
        hqPlayers_.clear();
        upnpRenderers_.clear();
    }

    {
        std::lock_guard<std::mutex> lock(zonesMutex_);
        zones_.clear();
    }

    initialized_.store(false);
    Logger::info("OutputManager: Shutdown complete");
}

bool OutputManager::isInitialized() const {
    return initialized_.load();
}

std::vector<OutputManager::OutputDevice> OutputManager::discoverDevices() {
    Logger::info("OutputManager: Discovering audio output devices");

    std::vector<OutputDevice> devices;

#if defined(_WIN32)
    // Windows ASIO devices
    devices.insert(devices.end(), discoverASIODevice());
    // Windows WASAPI devices
    devices.insert(devices.end(), discoverWASAPIDevices());
    // DirectSound devices
    devices.insert(devices.end(), discoverDirectSoundDevices());
#elif defined(__APPLE__)
    // macOS Core Audio devices
    devices.insert(devices.end(), discoverCoreAudioDevices());
#elif defined(__linux__)
    // Linux ALSA devices
    devices.insert(devices.end(), discoverALSADevices());
    // PulseAudio devices
    devices.insert(devices.end(), discoverPulseAudioDevices());
    // JACK devices
    devices.insert(devices.end(), discoverJACKDevices());
#endif

    // Network devices (platform-independent)
    devices.insert(devices.end(), discoverUPnPDevices());

    Logger::info("OutputManager: Discovered {} devices", devices.size());
    return devices;
}

std::vector<OutputManager::OutputDevice> OutputManager::getAvailableDevices() const {
    std::lock_guard<std::mutex> lock(devicesMutex_);
    std::vector<OutputDevice> available;

    for (const auto& [deviceId, device] : devices_) {
        if (device.status == DeviceStatus::AVAILABLE) {
            available.push_back(device);
        }
    }

    return available;
}

OutputManager::OutputDevice OutputManager::getDevice(const std::string& deviceId) const {
    std::lock_guard<std::mutex> lock(devicesMutex_);

    auto it = devices_.find(deviceId);
    if (it != devices_.end()) {
        return it->second;
    }

    return OutputDevice{};
}

bool OutputManager::addDevice(const OutputConfig& config) {
    if (!validateDeviceConfig(config)) {
        setError("Invalid device configuration");
        return false;
    }

    Logger::info("OutputManager: Adding device: {} ({})", config.deviceName, getDeviceTypeName(config.type));

    std::string deviceId = config.deviceIdentifier.empty() ? generateDeviceId() : config.deviceIdentifier;

    OutputDevice device;
    device.deviceId = deviceId;
    device.deviceName = config.deviceName;
    device.deviceDescription = "Output device";
    device.type = config.type;
    device.status = DeviceStatus::CONFIGURING;
    device.currentFormat = config.format;
    device.currentSampleRate = config.sampleRate;
    device.currentBitDepth = config.bitDepth;
    device.currentChannels = config.channels;

    {
        std::lock_guard<std::mutex> lock(devicesMutex_);
        devices_[deviceId] = device;
        deviceConfigs_[deviceId] = config;
    }

    // Initialize device-specific implementation
    bool success = false;
    switch (config.type) {
        case OutputType::ROON_BRIDGE:
            success = initializeRoonBridge(deviceId, config);
            break;
        case OutputType::HQPLAYER_NAA:
            success = initializeHQPlayerNAA(deviceId, config);
            break;
        case OutputType::UPNP_RENDERER:
            success = initializeUPnPRenderer(deviceId, config);
            break;
        case OutputType::ASIO_DEVICE:
            success = initializeASIODevice(deviceId, config);
            break;
        case OutputType::WASAPI_DEVICE:
            success = initializeWASAPIDevice(deviceId, config);
            break;
        default:
            setError("Unsupported device type");
            return false;
    }

    if (success) {
        {
            std::lock_guard<std::mutex> lock(devicesMutex_);
            devices_[deviceId].status = DeviceStatus::AVAILABLE;
        }
        notifyDeviceEvent("added", device);
        return true;
    } else {
        {
            std::lock_guard<std::mutex> lock(devicesMutex_);
            devices_.erase(deviceId);
            deviceConfigs_.erase(deviceId);
        }
        return false;
    }
}

bool OutputManager::removeDevice(const std::string& deviceId) {
    Logger::info("OutputManager: Removing device: {}", deviceId);

    {
        std::lock_guard<std::mutex> lock(devicesMutex_);
        auto it = devices_.find(deviceId);
        if (it == devices_.end()) {
            return false;
        }

        // Stop device if running
        if (it->second.status == DeviceStatus::AVAILABLE) {
            stopOutput(deviceId);
        }

        OutputDevice device = it->second;
        devices_.erase(it);
        deviceConfigs_.erase(deviceId);
    }

    // Clean up device-specific implementation
    switch (getDevice(deviceId).type) {
        case OutputType::ROON_BRIDGE:
            stopRoonBridge(deviceId);
            break;
        case OutputType::HQPLAYER_NAA:
            stopHQPlayerNAA(deviceId);
            break;
        case OutputType::UPNP_RENDERER:
            stopUPnPRenderer(deviceId);
            break;
        default:
            break;
    }

    notifyDeviceEvent("removed", getDevice(deviceId));
    return true;
}

bool OutputManager::updateDevice(const std::string& deviceId, const OutputConfig& config) {
    if (!validateDeviceConfig(config)) {
        setError("Invalid device configuration");
        return false;
    }

    std::lock_guard<std::mutex> lock(devicesMutex_);
    auto it = devices_.find(deviceId);
    if (it == devices_.end()) {
        return false;
    }

    // Stop device if running
    bool wasRunning = it->second.status == DeviceStatus::AVAILABLE;
    if (wasRunning) {
        stopOutput(deviceId);
    }

    // Update configuration
    deviceConfigs_[deviceId] = config;
    it->second.deviceName = config.deviceName;
    it->second.currentFormat = config.format;
    it->second.currentSampleRate = config.sampleRate;
    it->second.currentBitDepth = config.bitDepth;
    it->second.currentChannels = config.channels;
    it->second.status = DeviceStatus::CONFIGURING;

    // Reinitialize with new config
    bool success = false;
    switch (config.type) {
        case OutputType::ASIO_DEVICE:
            success = initializeASIODevice(deviceId, config);
            break;
        case OutputType::WASAPI_DEVICE:
            success = initializeWASAPIDevice(deviceId, config);
            break;
        default:
            success = true; // Other types don't need reinitialization
            break;
    }

    if (success) {
        it->second.status = DeviceStatus::AVAILABLE;
        if (wasRunning) {
            startOutput(deviceId);
        }
        notifyDeviceEvent("updated", it->second);
        return true;
    }

    it->second.status = DeviceStatus::ERROR;
    return false;
}

bool OutputManager::startOutput(const std::string& deviceId) {
    Logger::info("OutputManager: Starting output for device: {}", deviceId);

    std::lock_guard<std::mutex> lock(devicesMutex_);
    auto it = devices_.find(deviceId);
    if (it == devices_.end() || it->second.status != DeviceStatus::AVAILABLE) {
        return false;
    }

    // Implementation depends on device type
    bool success = false;
    switch (it->second.type) {
        case OutputType::ROON_BRIDGE:
            success = startRoonBridgeDevice(deviceId);
            break;
        case OutputType::HQPLAYER_NAA:
            success = startHQPlayerDevice(deviceId);
            break;
        case OutputType::UPNP_RENDERER:
            success = startUPnPRendererDevice(deviceId);
            break;
        case OutputType::ASIO_DEVICE:
            success = startASIODevice(deviceId);
            break;
        case OutputType::WASAPI_DEVICE:
            success = startWASAPIDevice(deviceId);
            break;
        default:
            setError("Unsupported device type for output");
            return false;
    }

    if (success) {
        it->second.status = DeviceStatus::AVAILABLE;
        it->second.lastActivity = std::chrono::steady_clock::now();
        notifyDeviceEvent("started", it->second);
        Logger::info("OutputManager: Device {} started successfully", deviceId);
        return true;
    }

    it->second.status = DeviceStatus::ERROR;
    return false;
}

bool OutputManager::stopOutput(const std::string& deviceId) {
    Logger::info("OutputManager: Stopping output for device: {}", deviceId);

    std::lock_guard<std::mutex> lock(devicesMutex_);
    auto it = devices_.find(deviceId);
    if (it == devices_.end()) {
        return false;
    }

    bool success = false;
    switch (it->second.type) {
        case OutputType::ROON_BRIDGE:
            stopRoonBridgeDevice(deviceId);
            success = true;
            break;
        case OutputType::HQPLAYER_NAA:
            stopHQPlayerDevice(deviceId);
            success = true;
            break;
        case OutputType::UPNP_RENDERER:
            stopUPnPRendererDevice(deviceId);
            success = true;
            break;
        case OutputType::ASIO_DEVICE:
            stopASIODevice(deviceId);
            success = true;
            break;
        case OutputType::WASAPI_DEVICE:
            stopWASAPIDevice(deviceId);
            success = true;
            break;
        default:
            success = true;
            break;
    }

    if (success) {
        it->second.status = DeviceStatus::AVAILABLE;
        notifyDeviceEvent("stopped", it->second);
    }

    return success;
}

bool OutputManager::pauseOutput(const std::string& deviceId) {
    // Implementation depends on device type
    // For now, just update status
    std::lock_guard<std::mutex> lock(devicesMutex_);
    auto it = devices_.find(deviceId);
    if (it != devices_.end()) {
        it->second.status = DeviceStatus::UNAVAILABLE;
        return true;
    }
    return false;
}

bool OutputManager::resumeOutput(const std::string& deviceId) {
    // Implementation depends on device type
    // For now, just update status
    std::lock_guard<std::mutex> lock(devicesMutex_);
    auto it = devices_.find(deviceId);
    if (it != devices_.end()) {
        it->second.status = DeviceStatus::AVAILABLE;
        return true;
    }
    return false;
}

bool OutputManager::createZone(const std::string& zoneName, const std::vector<std::string>& deviceIds) {
    Logger::info("OutputManager: Creating zone '{}' with {} devices", zoneName, deviceIds.size());

    // Verify all devices exist and are available
    {
        std::lock_guard<std::mutex> lock(devicesMutex_);
        for (const auto& deviceId : deviceIds) {
            auto it = devices_.find(deviceId);
            if (it == devices_.end() || it->second.status != DeviceStatus::AVAILABLE) {
                Logger::error("OutputManager: Device {} not available for zone", deviceId);
                return false;
            }
        }
    }

    std::lock_guard<std::mutex> lock(zonesMutex_);
    zones_[zoneName] = deviceIds;
    return true;
}

bool OutputManager::removeZone(const std::string& zoneName) {
    std::lock_guard<std::mutex> lock(zonesMutex_);
    auto it = zones_.find(zoneName);
    if (it != zones_.end()) {
        zones_.erase(it);
        Logger::info("OutputManager: Removed zone '{}'", zoneName);
        return true;
    }
    return false;
}

std::vector<std::string> OutputManager::getZoneDevices(const std::string& zoneName) const {
    std::lock_guard<std::mutex> lock(zonesMutex_);
    auto it = zones_.find(zoneName);
    if (it != zones_.end()) {
        return it->second;
    }
    return {};
}

bool OutputManager::playToZone(const std::string& zoneName, const AudioFrame& frame) {
    std::vector<std::string> deviceIds = getZoneDevices(zoneName);
    if (deviceIds.empty()) {
        return false;
    }

    bool allSuccess = true;
    for (const auto& deviceId : deviceIds) {
        if (!routeAudioToDevice(deviceId, frame)) {
            allSuccess = false;
        }
    }

    return allSuccess;
}

bool OutputManager::routeAudioToDevice(const std::string& deviceId, const AudioFrame& frame) {
    std::lock_guard<std::mutex> lock(devicesMutex_);
    auto it = devices_.find(deviceId);
    if (it == devices_.end() || it->second.status != DeviceStatus::AVAILABLE) {
        return false;
    }

    // Convert format if needed
    AudioFrame convertedFrame;
    const AudioFrame* targetFrame = &frame;

    if (frame.format != it->second.currentFormat ||
        frame.sampleRate != it->second.currentSampleRate) {

        if (!convertFormat(frame, convertedFrame, it->second)) {
            Logger::error("OutputManager: Failed to convert format for device {}", deviceId);
            return false;
        }
        targetFrame = &convertedFrame;
    }

    // Queue for processing
    {
        std::lock_guard<std::mutex> queueLock(audioQueueMutex_);
        audioQueue_.push(*targetFrame);
    }
    audioQueueCondition_.notify_one();

    return true;
}

bool OutputManager::routeAudioToZone(const std::string& zoneName, const AudioFrame& frame) {
    return playToZone(zoneName, frame);
}

bool OutputManager::setDeviceVolume(const std::string& deviceId, float volume) {
    std::lock_guard<std::mutex> lock(devicesMutex_);
    auto it = devices_.find(deviceId);
    if (it != devices_.end()) {
        it->second.currentVolume = std::clamp(volume, 0.0f, 1.0f);

        // Apply to device-specific implementation
        switch (it->second.type) {
            case OutputType::ROON_BRIDGE:
                setRoonBridgeVolume(deviceId, volume);
                break;
            case OutputType::HQPLAYER_NAA:
                setHQPlayerVolume(deviceId, volume);
                break;
            case OutputType::ASIO_DEVICE:
                setASIOVolume(deviceId, volume);
                break;
            case OutputType::WASAPI_DEVICE:
                setWASAPIVolume(deviceId, volume);
                break;
            default:
                break;
        }
        return true;
    }
    return false;
}

bool OutputManager::setDeviceMute(const std::string& deviceId, bool muted) {
    std::lock_guard<std::mutex> lock(devicesMutex_);
    auto it = devices_.find(deviceId);
    if (it != devices_.end()) {
        it->second.isMuted = muted;
        return true;
    }
    return false;
}

bool OutputManager::setZoneVolume(const std::string& zoneName, float volume) {
    std::vector<std::string> deviceIds = getZoneDevices(zoneName);
    bool allSuccess = true;

    for (const auto& deviceId : deviceIds) {
        if (!setDeviceVolume(deviceId, volume)) {
            allSuccess = false;
        }
    }

    return allSuccess;
}

bool OutputManager::setZoneMute(const std::string& zoneName, bool muted) {
    std::vector<std::string> deviceIds = getZoneDevices(zoneName);
    bool allSuccess = true;

    for (const auto& deviceId : deviceIds) {
        if (!setDeviceMute(deviceId, muted)) {
            allSuccess = false;
        }
    }

    return allSuccess;
}

bool OutputManager::convertFormat(const AudioFrame& input, AudioFrame& output,
                                  const OutputDevice& targetDevice) {
    output = input;

    // Handle format conversion
    if (input.format != targetDevice.currentFormat) {
        if (!convertPCMFormat(input, output, targetDevice.currentFormat, targetDevice.currentBitDepth)) {
            return false;
        }
    }

    // Handle sample rate conversion
    if (input.sampleRate != targetDevice.currentSampleRate) {
        AudioFrame resampledFrame;
        if (!resampleAudio(output, resampledFrame, targetDevice.currentSampleRate)) {
            return false;
        }
        output = resampledFrame;
    }

    // Handle channel conversion
    if (input.channels != targetDevice.currentChannels) {
        // For now, just use channel 0 for mono or duplicate for stereo
        if (targetDevice.currentChannels == 1 && input.channels >= 1) {
            // Convert to mono
            std::vector<float> monoData;
            monoData.reserve(output.frameCount);
            for (size_t i = 0; i < output.audioData.size(); i += input.channels) {
                float sum = 0.0f;
                for (size_t ch = 0; ch < input.channels && (i + ch) < output.audioData.size(); ++ch) {
                    sum += output.audioData[i + ch];
                }
                monoData.push_back(sum / input.channels);
            }
            output.audioData = monoData;
            output.channels = 1;
        } else if (targetDevice.currentChannels == 2 && input.channels == 1) {
            // Convert to stereo
            std::vector<float> stereoData;
            stereoData.reserve(output.audioData.size() * 2);
            for (float sample : output.audioData) {
                stereoData.push_back(sample);
                stereoData.push_back(sample);
            }
            output.audioData = stereoData;
            output.channels = 2;
        }
    }

    return true;
}

bool OutputManager::resampleAudio(const AudioFrame& input, AudioFrame& output,
                                  uint32_t targetSampleRate) {
    if (input.sampleRate == targetSampleRate) {
        output = input;
        return true;
    }

    // Simple linear resampling for demonstration
    // In production, use high-quality resampling library (libsamplerate, etc.)
    float ratio = static_cast<float>(targetSampleRate) / input.sampleRate;
    size_t outputFrames = static_cast<size_t>(input.frameCount * ratio);

    output.audioData.resize(outputFrames * input.channels);
    output.frameCount = outputFrames;
    output.sampleRate = targetSampleRate;
    output.format = input.format;
    output.channels = input.channels;
    output.timestamp = input.timestamp;

    // Linear interpolation
    for (size_t outFrame = 0; outFrame < outputFrames; ++outFrame) {
        float inFrameFloat = outFrame / ratio;
        size_t inFrame0 = static_cast<size_t>(inFrameFloat);
        size_t inFrame1 = std::min(inFrame0 + 1, input.frameCount - 1);
        float fraction = inFrameFloat - inFrame0;

        for (size_t ch = 0; ch < input.channels; ++ch) {
            float sample0 = input.audioData[inFrame0 * input.channels + ch];
            float sample1 = input.audioData[inFrame1 * input.channels + ch];
            float interpolated = sample0 + fraction * (sample1 - sample0);
            output.audioData[outFrame * input.channels + ch] = interpolated;
        }
    }

    return true;
}

bool OutputManager::convertToDSD(const AudioFrame& input, AudioFrame& output,
                                 uint32_t dsdRate) {
    // DSD conversion implementation
    // This is a simplified placeholder - real DSD conversion requires sophisticated modulation

    if (input.sampleRate == dsdRate && input.format == AudioFormat::DSD1) {
        output = input;
        return true;
    }

    // For demonstration, we'll just convert to high-rate PCM
    // Real implementation would use delta-sigma modulation
    output = input;
    output.format = AudioFormat::PCM_S24LE;
    output.sampleRate = dsdRate / 64; // Simplified conversion

    Logger::warn("OutputManager: DSD conversion not fully implemented - using PCM conversion");
    return true;
}

// Thread implementations
void OutputManager::deviceDiscoveryThread() {
    Logger::info("OutputManager: Device discovery thread started");

    while (!shouldShutdown_.load()) {
        try {
            if (enableDeviceDiscovery_.load()) {
                updateDeviceList();
            }

            std::this_thread::sleep_for(std::chrono::seconds(10));

        } catch (const std::exception& e) {
            Logger::error("OutputManager: Device discovery thread error: {}", e.what());
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    }

    Logger::info("OutputManager: Device discovery thread stopped");
}

void OutputManager::deviceMonitoringThread() {
    Logger::info("OutputManager: Device monitoring thread started");

    while (!shouldShutdown_.load()) {
        try {
            if (enableHealthMonitoring_.load()) {
                monitorDeviceHealth();
                updateDriftCorrection();
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(updateInterval_));

        } catch (const std::exception& e) {
            Logger::error("OutputManager: Device monitoring thread error: {}", e.what());
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    Logger::info("OutputManager: Device monitoring thread stopped");
}

void OutputManager::audioProcessingThread() {
    Logger::info("OutputManager: Audio processing thread started");

    processingActive_.store(true);

    while (!shouldShutdown_.load() && processingActive_.load()) {
        std::unique_lock<std::mutex> lock(audioQueueMutex_);
        audioQueueCondition_.wait(lock, [this] {
            return !audioQueue_.empty() || shouldShutdown_.load() || !processingActive_.load();
        });

        if (shouldShutdown_.load() || !processingActive_.load()) {
            break;
        }

        while (!audioQueue_.empty()) {
            AudioFrame frame = std::move(audioQueue_.front());
            audioQueue_.pop();
            lock.unlock();

            // Process frame for all available devices
            std::lock_guard<std::mutex> deviceLock(devicesMutex_);
            for (auto& [deviceId, device] : devices_) {
                if (device.status == DeviceStatus::AVAILABLE) {
                    processAudioFrame(frame, deviceId);
                }
            }

            lock.lock();
        }
    }

    processingActive_.store(false);
    Logger::info("OutputManager: Audio processing thread stopped");
}

void OutputManager::processAudioFrame(AudioFrame& frame, const std::string& deviceId) {
    auto it = devices_.find(deviceId);
    if (it == devices_.end()) {
        return;
    }

    OutputDevice& device = it->second;

    // Apply device-specific processing
    if (!applyDeviceProcessing(frame, device)) {
        device.errorCount++;
        return;
    }

    // Send to device based on type
    bool success = false;
    switch (device.type) {
        case OutputType::ROON_BRIDGE:
            success = sendToRoonBridge(deviceId, frame);
            break;
        case OutputType::HQPLAYER_NAA:
            success = sendToHQPlayer(deviceId, frame);
            break;
        case OutputType::UPNP_RENDERER:
            success = sendToUPnPRenderer(deviceId, frame);
            break;
        case OutputType::ASIO_DEVICE:
            success = sendToASIODevice(deviceId, frame);
            break;
        case OutputType::WASAPI_DEVICE:
            success = sendToWASAPIDevice(deviceId, frame);
            break;
        default:
            success = false;
            break;
    }

    if (success) {
        device.framesPlayed += frame.frameCount;
        device.lastActivity = std::chrono::steady_clock::now();

        // Update statistics
        {
            std::lock_guard<std::mutex> statsLock(statsMutex_);
            statistics_.totalFramesPlayed += frame.frameCount;
            statistics_.lastActivity = device.lastActivity;
        }

        // Notify callback
        if (audioOutputCallback_) {
            audioOutputCallback_(deviceId, frame);
        }
    } else {
        device.framesDropped += frame.frameCount;
        device.errorCount++;
    }
}

// Utility implementations
std::string OutputManager::generateDeviceId() const {
    static std::atomic<uint32_t> counter{0};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);

    return "device_" + std::to_string(counter++) + "_" + std::to_string(dis(gen));
}

std::string OutputManager::getDeviceTypeName(OutputType type) const {
    switch (type) {
        case OutputType::ROON_BRIDGE: return "Roon Bridge";
        case OutputType::HQPLAYER_NAA: return "HQPlayer NAA";
        case OutputType::UPNP_RENDERER: return "UPnP Renderer";
        case OutputType::ASIO_DEVICE: return "ASIO Device";
        case OutputType::WASAPI_DEVICE: return "WASAPI Device";
        case OutputType::DIRECTSOUND_DEVICE: return "DirectSound Device";
        case OutputType::COREAUDIO_DEVICE: return "Core Audio Device";
        case OutputType::ALSA_DEVICE: return "ALSA Device";
        case OutputType::PULSEAUDIO_DEVICE: return "PulseAudio Device";
        case OutputType::JACK_DEVICE: return "JACK Device";
        case OutputType::VIRTUAL_OUTPUT: return "Virtual Output";
        case OutputType::NETWORK_STREAM: return "Network Stream";
        case OutputType::FILE_OUTPUT: return "File Output";
        default: return "Unknown";
    }
}

bool OutputManager::validateDeviceConfig(const OutputConfig& config) const {
    if (config.deviceName.empty()) {
        setError("Device name cannot be empty");
        return false;
    }

    if (config.sampleRate == 0 || config.sampleRate > 768000) {
        setError("Invalid sample rate");
        return false;
    }

    if (config.channels == 0 || config.channels > 32) {
        setError("Invalid channel count");
        return false;
    }

    if (config.bitDepth < 8 || config.bitDepth > 64) {
        setError("Invalid bit depth");
        return false;
    }

    return true;
}

void OutputManager::notifyDeviceEvent(const std::string& eventType, const OutputDevice& device) {
    Logger::info("OutputManager: Device event: {} - {}", eventType, device.deviceName);

    if (eventType == "added" && deviceAddedCallback_) {
        deviceAddedCallback_(eventType, device);
    } else if (eventType == "removed" && deviceRemovedCallback_) {
        deviceRemovedCallback_(eventType, device);
    } else if ((eventType == "started" || eventType == "stopped" || eventType == "updated") &&
               deviceStateChangedCallback_) {
        deviceStateChangedCallback_(eventType, device);
    }
}

void OutputManager::notifyError(const std::string& deviceId, const std::string& error) {
    Logger::error("OutputManager: Device {} error: {}", deviceId, error);

    if (errorCallback_) {
        errorCallback_(deviceId, error);
    }
}

void OutputManager::setError(const std::string& error) const {
    lastError_ = error;
    Logger::error("OutputManager: {}", error);
}

// Placeholder implementations for device discovery and initialization
std::vector<OutputManager::OutputDevice> OutputManager::discoverASIODevice() {
    std::vector<OutputDevice> devices;
    // Implementation would use ASIO SDK
    return devices;
}

std::vector<OutputManager::OutputDevice> OutputManager::discoverWASAPIDevices() {
    std::vector<OutputDevice> devices;
    // Implementation would use WASAPI
    return devices;
}

std::vector<OutputManager::OutputDevice> OutputManager::discoverDirectSoundDevices() {
    std::vector<OutputDevice> devices;
    // Implementation would use DirectSound
    return devices;
}

std::vector<OutputManager::OutputDevice> OutputManager::discoverCoreAudioDevices() {
    std::vector<OutputDevice> devices;
    // Implementation would use Core Audio
    return devices;
}

std::vector<OutputManager::OutputDevice> OutputManager::discoverALSADevices() {
    std::vector<OutputDevice> devices;
    // Implementation would use ALSA
    return devices;
}

std::vector<OutputManager::OutputDevice> OutputManager::discoverPulseAudioDevices() {
    std::vector<OutputDevice> devices;
    // Implementation would use PulseAudio
    return devices;
}

std::vector<OutputManager::OutputDevice> OutputManager::discoverJACKDevices() {
    std::vector<OutputDevice> devices;
    // Implementation would use JACK
    return devices;
}

bool OutputManager::initializeRoonBridge(const std::string& deviceId, const OutputConfig& config) {
    // Implementation would initialize Roon Bridge
    return true;
}

bool OutputManager::initializeHQPlayerNAA(const std::string& deviceId, const OutputConfig& config) {
    // Implementation would initialize HQPlayer NAA
    return true;
}

bool OutputManager::initializeUPnPRenderer(const std::string& deviceId, const OutputConfig& config) {
    // Implementation would initialize UPnP renderer
    return true;
}

bool OutputManager::initializeASIODevice(const std::string& deviceId, const OutputConfig& config) {
    // Implementation would initialize ASIO device
    return true;
}

bool OutputManager::initializeWASAPIDevice(const std::string& deviceId, const OutputConfig& config) {
    // Implementation would initialize WASAPI device
    return true;
}

void OutputManager::updateDeviceList() {
    auto newDevices = discoverDevices();

    std::lock_guard<std::mutex> lock(devicesMutex_);
    for (const auto& device : newDevices) {
        if (devices_.find(device.deviceId) == devices_.end()) {
            devices_[device.deviceId] = device;
            notifyDeviceEvent("discovered", device);
        }
    }
}

void OutputManager::monitorDeviceHealth() {
    std::lock_guard<std::mutex> lock(devicesMutex_);

    for (auto& [deviceId, device] : devices_) {
        auto now = std::chrono::steady_clock::now();
        auto timeSinceActivity = std::chrono::duration_cast<std::chrono::seconds>(now - device.lastActivity);

        if (timeSinceActivity.count() > 30 && device.status == DeviceStatus::AVAILABLE) {
            device.needsRecovery = true;
            notifyError(deviceId, "Device appears unresponsive");
        }
    }
}

bool OutputManager::convertPCMFormat(const AudioFrame& input, AudioFrame& output,
                                     AudioFormat targetFormat, uint16_t targetBitDepth) {
    output = input;
    output.format = targetFormat;

    // Simplified format conversion
    if (targetBitDepth != 32) {
        // Convert bit depth
        float scale = std::pow(2.0f, targetBitDepth - 1) - 1.0f;
        for (float& sample : output.audioData) {
            sample = std::clamp(sample * scale, -scale, scale);
        }
    }

    return true;
}

bool OutputManager::resampleWithGPU(const AudioFrame& input, AudioFrame& output,
                                    uint32_t targetSampleRate) {
    auto gpuProcessor = audioEngine_.lock();
    if (!gpuProcessor) {
        return resampleAudio(input, output, targetSampleRate);
    }

    // Use GPU-accelerated resampling if available
    return resampleAudio(input, output, targetSampleRate); // Placeholder
}

bool OutputManager::applyVolumeControl(AudioFrame& frame, float volume) {
    if (volume >= 0.0f && volume <= 1.0f) {
        for (float& sample : frame.audioData) {
            sample *= volume;
        }
        return true;
    }
    return false;
}

bool OutputManager::applyDeviceProcessing(AudioFrame& frame, const OutputDevice& device) {
    if (!device.isMuted) {
        if (device.enableVolumeControl) {
            applyVolumeControl(frame, device.currentVolume);
        }
        return true;
    }
    return false;
}

void OutputManager::updateDriftCorrection() {
    // Implementation would handle device synchronization
}

OutputManager::OutputStatistics OutputManager::getStatistics() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    return statistics_;
}

OutputManager::OutputDevice OutputManager::getDeviceStatistics(const std::string& deviceId) const {
    std::lock_guard<std::mutex> lock(devicesMutex_);
    auto it = devices_.find(deviceId);
    return it != devices_.end() ? it->second : OutputDevice{};
}

void OutputManager::resetStatistics() {
    std::lock_guard<std::mutex> lock(statsMutex_);
    statistics_ = OutputStatistics{};
    statistics_.startTime = std::chrono::steady_clock::now();
    statistics_.lastActivity = statistics_.startTime;
}

OutputManager::HealthStatus OutputManager::getHealthStatus() const {
    HealthStatus status;

    std::lock_guard<std::mutex> lock(devicesMutex_);
    for (const auto& [deviceId, device] : devices_) {
        if (device.status == DeviceStatus::AVAILABLE) {
            status.activeDevices++;
            if (device.errorCount == 0) {
                status.healthyDevices++;
            } else if (device.errorCount < 5) {
                status.degradedDevices++;
            } else {
                status.failedDevices++;
            }
        }
    }

    status.isHealthy = (status.failedDevices == 0 && status.activeDevices > 0);

    return status;
}

bool OutputManager::isDeviceHealthy(const std::string& deviceId) const {
    auto device = getDeviceStatistics(deviceId);
    return device.status == DeviceStatus::AVAILABLE && device.errorCount < 5;
}

void OutputManager::recoverDevice(const std::string& deviceId) {
    Logger::info("OutputManager: Attempting to recover device {}", deviceId);

    // Implementation would attempt device recovery
    // For now, just restart the device
    stopOutput(deviceId);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    startOutput(deviceId);
}

void OutputManager::setMaxDevices(uint32_t maxDevices) {
    maxDevices_ = maxDevices;
}

uint32_t OutputManager::getMaxDevices() const {
    return maxDevices_;
}

void OutputManager::setUpdateInterval(uint32_t intervalMs) {
    updateInterval_ = intervalMs;
}

uint32_t OutputManager::getUpdateInterval() const {
    return updateInterval_;
}

// Callback setters
void OutputManager::setDeviceAddedCallback(DeviceEventCallback callback) {
    deviceAddedCallback_ = callback;
}

void OutputManager::setDeviceRemovedCallback(DeviceEventCallback callback) {
    deviceRemovedCallback_ = callback;
}

void OutputManager::setDeviceStateChangedCallback(DeviceEventCallback callback) {
    deviceStateChangedCallback_ = callback;
}

void OutputManager::setAudioOutputCallback(AudioEventCallback callback) {
    audioOutputCallback_ = callback;
}

void OutputManager::setErrorCallback(ErrorEventCallback callback) {
    errorCallback_ = callback;
}

} // namespace vortex