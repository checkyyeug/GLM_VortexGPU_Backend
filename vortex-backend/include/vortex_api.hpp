#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <optional>

#include "audio_types.hpp"
#include "network_types.hpp"

namespace vortex {

/**
 * @brief Main Vortex Audio API - Core interface for audio processing backend
 *
 * This header defines the primary API surface for the Vortex GPU Audio Backend.
 * All audio operations, GPU processing, and network services are accessible
 * through this interface.
 */
class VortexAPI {
public:
    using SystemClock = std::chrono::system_clock;
    using TimePoint = SystemClock::time_point;

    virtual ~VortexAPI() = default;

    // Core lifecycle
    virtual bool initialize(uint32_t sampleRate, uint32_t bufferSize) = 0;
    virtual void shutdown() = 0;
    virtual bool isInitialized() const = 0;

    // GPU Acceleration
    virtual bool enableGPUAcceleration(GPUBackend backend) = 0;
    virtual bool disableGPUAcceleration() = 0;
    virtual bool isGPUAccelerationEnabled() const = 0;
    virtual std::vector<GPUBackend> getAvailableGPUBackends() const = 0;
    virtual GPUStatus getGPUStatus() const = 0;

    // Audio File Operations
    virtual std::string uploadAudioFile(const std::string& filePath) = 0;
    virtual bool removeAudioFile(const std::string& fileId) = 0;
    virtual AudioFile getAudioFile(const std::string& fileId) const = 0;
    virtual std::vector<AudioFile> listAudioFiles() const = 0;
    virtual AudioMetadata extractMetadata(const std::string& fileId) = 0;

    // Audio Processing
    virtual bool startProcessing(const std::string& fileId) = 0;
    virtual bool stopProcessing() = 0;
    virtual bool isProcessing() const = 0;
    virtual ProcessingStatus getProcessingStatus(const std::string& fileId) const = 0;
    virtual float getProcessingProgress(const std::string& fileId) const = 0;

    // Processing Chain Management
    virtual std::string createProcessingChain(const std::string& name) = 0;
    virtual bool removeProcessingChain(const std::string& chainId) = 0;
    virtual ProcessingChain getProcessingChain(const std::string& chainId) const = 0;
    virtual bool setActiveProcessingChain(const std::string& chainId) = 0;
    virtual std::string addFilter(const std::string& chainId, FilterType type, uint32_t position) = 0;
    virtual bool removeFilter(const std::string& chainId, const std::string& filterId) = 0;
    virtual bool setFilterParameter(const std::string& chainId, const std::string& filterId,
                                   const std::string& parameter, float value) = 0;

    // Output Device Management
    virtual std::vector<OutputDevice> discoverOutputDevices() = 0;
    virtual bool selectOutputDevice(const std::string& deviceId) = 0;
    virtual OutputDevice getCurrentOutputDevice() const = 0;
    virtual std::vector<OutputDevice> getAvailableOutputDevices() const = 0;

    // Real-time Data Access
    virtual RealTimeData getRealTimeData() const = 0;
    virtual void setRealTimeDataCallback(std::function<void(const RealTimeData&)> callback) = 0;

    // Session Management
    virtual std::string createSession() = 0;
    virtual bool closeSession(const std::string& sessionId) = 0;
    virtual AudioSession getSession(const std::string& sessionId) const = 0;
    virtual bool loadAudioFileIntoSession(const std::string& sessionId, const std::string& fileId) = 0;
    virtual bool startPlayback(const std::string& sessionId) = 0;
    virtual bool pausePlayback(const std::string& sessionId) = 0;
    virtual bool stopPlayback(const std::string& sessionId) = 0;
    virtual bool seekPlayback(const std::string& sessionId, double position) = 0;

    // Configuration
    virtual bool loadConfiguration(const std::string& configPath) = 0;
    virtual bool saveConfiguration(const std::string& configPath) = 0;
    virtual SystemConfiguration getConfiguration() const = 0;
    virtual bool updateConfiguration(const SystemConfiguration& config) = 0;

    // System Monitoring
    virtual HardwareStatus getHardwareStatus() const = 0;
    virtual ProcessingMetrics getProcessingMetrics() const = 0;
    virtual bool isSystemHealthy() const = 0;

    // Network Services
    virtual bool startNetworkServices() = 0;
    virtual bool stopNetworkServices() = 0;
    virtual bool areNetworkServicesRunning() const = 0;
    virtual uint16_t getHTTPPort() const = 0;
    virtual uint16_t getWebSocketPort() const = 0;

    // Factory method
    static std::unique_ptr<VortexAPI> create();

protected:
    // Internal state
    bool m_initialized = false;
    bool m_gpuEnabled = false;
    GPUBackend m_currentGPUBackend = GPUBackend::CUDA;
};

// Error handling
enum class VortexError {
    None,
    InitializationFailed,
    GPUNotAvailable,
    FileNotFound,
    InvalidFormat,
    ProcessingFailed,
    DeviceNotFound,
    NetworkError,
    ConfigurationError,
    OutOfMemory,
    InvalidParameter
};

struct VortexResult {
    VortexError error = VortexError::None;
    std::string message;

    bool isSuccess() const { return error == VortexError::None; }
    bool isError() const { return error != VortexError::None; }
};

} // namespace vortex