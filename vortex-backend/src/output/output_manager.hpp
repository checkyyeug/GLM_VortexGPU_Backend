#pragma once

#include "../audio_types.hpp"
#include "../network_types.hpp"

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <condition_variable>
#include <functional>
#include <chrono>

namespace vortex {

// Forward declarations
class AudioEngine;
class GPUProcessor;

/**
 * @brief Multi-device audio output management system
 *
 * Provides professional multi-device output support including:
 * - Roon Bridge integration for high-quality audio streaming
 * - HQPlayer Network Audio Adapter (NAA) compatibility
 * - UPnP/DLNA renderer support
 * - ASIO/WASAPI/DirectSound low-latency output
 * - Device synchronization and sample rate conversion
 * - Hardware-accelerated upsampling and filtering
 * - Gapless playback and sample-accurate timing
 *
 * Features:
 * - Up to 8 simultaneous output devices
 * - Independent device configurations (sample rates, formats, bit depths)
 * - Real-time device hot-plugging support
 * - Automatic device discovery and enumeration
 * - Audio format conversion and resampling
 * - Multi-zone audio distribution
 * - Device health monitoring and recovery
 * - Bit-perfect output paths
 * - Hardware passthrough for DSD and high-res PCM
 */
class OutputManager {
public:
    enum class OutputType {
        NONE = 0,
        ROON_BRIDGE,           // Roon Bridge endpoint
        HQPLAYER_NAA,          // HQPlayer Network Audio Adapter
        UPNP_RENDERER,         // UPnP/DLNA renderer
        ASIO_DEVICE,           // ASIO low-latency audio device
        WASAPI_DEVICE,         // Windows Audio Session API
        DIRECTSOUND_DEVICE,    // DirectSound device
        COREAUDIO_DEVICE,      // macOS Core Audio
        ALSA_DEVICE,           // Linux ALSA
        PULSEAUDIO_DEVICE,     // PulseAudio
        JACK_DEVICE,           // JACK Audio Connection Kit
        VIRTUAL_OUTPUT,        // Virtual audio endpoint
        NETWORK_STREAM,        // Network audio stream
        FILE_OUTPUT           // File recording output
    };

    enum class DeviceStatus {
        UNKNOWN,
        AVAILABLE,
        UNAVAILABLE,
        BUSY,
        ERROR,
        DISCONNECTED,
        CONFIGURING
    };

    enum class SynchronizationMode {
        NONE,           // No synchronization
        MASTER,         // This device is master clock
        SLAVE,          // This device follows master clock
        FREE_RUNNING    // Independent timing
    };

    struct OutputDevice {
        std::string deviceId;
        std::string deviceName;
        std::string deviceDescription;
        OutputType type = OutputType::NONE;
        DeviceStatus status = DeviceStatus::UNKNOWN;

        // Audio capabilities
        std::vector<AudioFormat> supportedFormats;
        std::vector<uint32_t> supportedSampleRates;
        std::vector<uint16_t> supportedBitDepths;
        std::vector<uint16_t> supportedChannelCounts;
        uint32_t maxChannels = 2;
        uint32_t minChannels = 1;
        uint32_t maxSampleRate = 192000;
        uint32_t minSampleRate = 44100;
        uint16_t maxBitDepth = 32;
        uint16_t minBitDepth = 16;

        // Latency information
        uint32_t minLatencyMs = 0;
        uint32_t maxLatencyMs = 0;
        uint32_t currentLatencyMs = 0;
        uint32_t preferredLatencyMs = 0;

        // DSD capabilities
        bool supportsDSD = false;
        bool supportsDSD64 = false;
        bool supportsDSD128 = false;
        bool supportsDSD256 = false;
        bool supportsDSD512 = false;
        bool supportsDSD1024 = false;
        bool supportsDoP = false;    // DSD over PCM

        // Device configuration
        AudioFormat currentFormat = AudioFormat::UNKNOWN;
        uint32_t currentSampleRate = 0;
        uint16_t currentBitDepth = 0;
        uint16_t currentChannels = 2;
        float currentVolume = 1.0f;
        bool isMuted = false;
        bool isExclusive = false;   // Exclusive mode access

        // Synchronization
        SynchronizationMode syncMode = SynchronizationMode::NONE;
        std::string masterDeviceId;
        uint64_t driftSamples = 0;
        float driftRate = 0.0f;

        // Network information
        std::string networkAddress;
        uint16_t networkPort = 0;
        std::string protocolVersion;
        std::map<std::string, std::string> deviceMetadata;

        // Statistics
        uint64_t framesPlayed = 0;
        uint64_t framesDropped = 0;
        uint64_t underruns = 0;
        uint64_t overruns = 0;
        float averageLatency = 0.0f;
        std::chrono::steady_clock::time_point lastActivity;

        // Health monitoring
        uint32_t errorCount = 0;
        std::string lastError;
        std::chrono::steady_clock::time_point lastErrorTime;
        bool needsRecovery = false;

        // Audio processing
        bool enableUpsampling = false;
        uint32_t targetSampleRate = 0;
        bool enableDSDConversion = false;
        bool enableVolumeControl = true;
        bool enableEQ = false;
    };

    struct OutputConfig {
        OutputType type = OutputType::NONE;
        std::string deviceName;
        std::string deviceIdentifier;

        // Audio settings
        AudioFormat format = AudioFormat::PCM_S16LE;
        uint32_t sampleRate = 44100;
        uint16_t bitDepth = 16;
        uint16_t channels = 2;
        uint32_t bufferSize = 512;
        uint32_t bufferCount = 3;

        // Latency settings
        uint32_t targetLatencyMs = 100;
        bool enableExclusiveMode = false;
        bool enableBitPerfect = false;

        // Processing settings
        bool enableUpsampling = false;
        uint32_t upsampleTargetRate = 0;
        bool enableDSDConversion = false;
        bool enableVolumeControl = true;
        bool enableResampling = true;
        std::string resamplerQuality = "high";

        // Network settings
        std::string networkAddress;
        uint16_t networkPort = 0;
        std::string authentication;
        std::map<std::string, std::string> networkSettings;

        // Device-specific settings
        std::map<std::string, std::variant<int, float, bool, std::string>> deviceSettings;

        // Synchronization
        SynchronizationMode syncMode = SynchronizationMode::NONE;
        std::string masterDeviceId;
        uint32_t driftCorrectionInterval = 1000; // ms

        // Error handling
        uint32_t maxRetries = 3;
        uint32_t recoveryTimeoutMs = 5000;
        bool enableAutoRecovery = true;
    };

    struct AudioFrame {
        std::vector<float> audioData;      // Interleaved audio samples
        uint32_t sampleRate = 0;
        uint16_t channels = 0;
        uint32_t frameCount = 0;
        AudioFormat format = AudioFormat::PCM_FLOAT32;
        uint64_t timestamp = 0;            // Presentation timestamp
        bool isMarker = false;             // Marker frame for synchronization
        std::map<std::string, std::string> metadata;
    };

    OutputManager();
    ~OutputManager();

    // Initialization
    bool initialize(std::shared_ptr<AudioEngine> audioEngine);
    void shutdown();
    bool isInitialized() const;

    // Device discovery and management
    std::vector<OutputDevice> discoverDevices();
    std::vector<OutputDevice> getAvailableDevices() const;
    OutputDevice getDevice(const std::string& deviceId) const;
    bool addDevice(const OutputConfig& config);
    bool removeDevice(const std::string& deviceId);
    bool updateDevice(const std::string& deviceId, const OutputConfig& config);

    // Output management
    bool startOutput(const std::string& deviceId);
    bool stopOutput(const std::string& deviceId);
    bool pauseOutput(const std::string& deviceId);
    bool resumeOutput(const std::string& deviceId);

    // Multi-zone management
    bool createZone(const std::string& zoneName, const std::vector<std::string>& deviceIds);
    bool removeZone(const std::string& zoneName);
    std::vector<std::string> getZoneDevices(const std::string& zoneName) const;
    bool playToZone(const std::string& zoneName, const AudioFrame& frame);

    // Audio routing
    bool routeAudioToDevice(const std::string& deviceId, const AudioFrame& frame);
    bool routeAudioToZone(const std::string& zoneName, const AudioFrame& frame);
    bool setDeviceVolume(const std::string& deviceId, float volume);
    bool setDeviceMute(const std::string& deviceId, bool muted);
    bool setZoneVolume(const std::string& zoneName, float volume);
    bool setZoneMute(const std::string& zoneName, bool muted);

    // Format conversion and processing
    bool convertFormat(const AudioFrame& input, AudioFrame& output,
                       const OutputDevice& targetDevice);
    bool resampleAudio(const AudioFrame& input, AudioFrame& output,
                      uint32_t targetSampleRate);
    bool convertToDSD(const AudioFrame& input, AudioFrame& output,
                     uint32_t dsdRate);

    // Device synchronization
    bool synchronizeDevices(const std::vector<std::string>& deviceIds);
    bool setMasterDevice(const std::string& deviceId);
    bool calibrateDeviceLatency(const std::string& deviceId);
    void updateDriftCorrection();

    // Roon Bridge integration
    struct RoonBridgeConfig {
        std::string deviceId;
        std::string displayName;
        std::pair<std::string, std::string> ipAddress;
        uint16_t port = 9100;
        bool enableRAAT = true;           // Roon Advanced Audio Transport
        bool enableAirPlay = false;
        bool enableMeridian = false;
        uint32_t maxSampleRate = 192000;
        bool supportDSD = true;
        bool supportMQA = false;
        std::string zoneName;
    };

    bool setupRoonBridge(const RoonBridgeConfig& config);
    bool isRoonBridgeActive(const std::string& deviceId) const;
    std::vector<std::string> getRoonBridgeDevices() const;

    // HQPlayer NAA integration
    struct HQPlayerConfig {
        std::string deviceId;
        std::string displayName;
        std::pair<std::string, std::string> ipAddress;
        uint16_t port = 12345;
        bool enableDSD upsampling = true;
        uint32_t maxDSDRate = 1024000;
        bool enablePolySinc = true;
        std::string filterType = "poly-sinc-xtr-mp";
        bool enableVolumeControl = false;  // Bit-perfect mode
    };

    bool setupHQPlayerNAA(const HQPlayerConfig& config);
    bool isHQPlayerActive(const std::string& deviceId) const;

    // UPnP/DLNA support
    struct UPnPConfig {
        std::string deviceId;
        std::string friendlyName;
        std::string uuid;
        std::pair<std::string, std::string> ipAddress;
        uint16_t port = 49152;
        bool enableDLNA = true;
        bool supportLPCM = true;
        bool supportMP3 = true;
        bool supportFLAC = true;
        bool supportWAV = true;
    };

    bool setupUPnPRenderer(const UPnPConfig& config);
    bool discoverUPnPDevices();
    std::vector<UPnPConfig> getUPnPDevices() const;

    // Monitoring and statistics
    struct OutputStatistics {
        std::map<std::string, OutputDevice> devices;
        uint64_t totalFramesPlayed = 0;
        uint64_t totalFramesDropped = 0;
        float averageLatencyMs = 0.0f;
        uint32_t activeDevices = 0;
        uint32_t errorCount = 0;
        std::chrono::steady_clock::time_point startTime;
        std::chrono::steady_clock::time_point lastActivity;
        std::map<std::string, float> cpuUsageByDevice;
        std::map<std::string, uint64_t> memoryUsageByDevice;
    };

    OutputStatistics getStatistics() const;
    OutputDevice getDeviceStatistics(const std::string& deviceId) const;
    void resetStatistics();

    // Health monitoring
    struct HealthStatus {
        bool isHealthy = true;
        uint32_t activeDevices = 0;
        uint32_t healthyDevices = 0;
        uint32_t degradedDevices = 0;
        uint32_t failedDevices = 0;
        std::map<std::string, std::string> deviceIssues;
        std::vector<std::string> warnings;
        std::vector<std::string> errors;
    };

    HealthStatus getHealthStatus() const;
    bool isDeviceHealthy(const std::string& deviceId) const;
    void recoverDevice(const std::string& deviceId);

    // Configuration
    void setMaxDevices(uint32_t maxDevices);
    uint32_t getMaxDevices() const;
    void setUpdateInterval(uint32_t intervalMs);
    uint32_t getUpdateInterval() const;

    // Event callbacks
    using DeviceEventCallback = std::function<void(const std::string&, const OutputDevice&)>;
    using AudioEventCallback = std::function<void(const std::string&, const AudioFrame&)>;
    using ErrorEventCallback = std::function<void(const std::string&, const std::string&)>;

    void setDeviceAddedCallback(DeviceEventCallback callback);
    void setDeviceRemovedCallback(DeviceEventCallback callback);
    void setDeviceStateChangedCallback(DeviceEventCallback callback);
    void setAudioOutputCallback(AudioEventCallback callback);
    void setErrorCallback(ErrorEventCallback callback);

private:
    // Device management
    void deviceDiscoveryThread();
    void deviceMonitoringThread();
    void updateDeviceList();
    void monitorDeviceHealth();

    // Audio processing
    void audioProcessingThread();
    void processAudioFrame(AudioFrame& frame, const std::string& deviceId);
    void synchronizeFrameTiming(AudioFrame& frame, const std::vector<std::string>& deviceIds);

    // Format conversion
    bool convertPCMFormat(const AudioFrame& input, AudioFrame& output,
                          AudioFormat targetFormat, uint16_t targetBitDepth);
    bool resampleWithGPU(const AudioFrame& input, AudioFrame& output,
                        uint32_t targetSampleRate);
    bool applyVolumeControl(AudioFrame& frame, float volume);
    bool applyDeviceProcessing(AudioFrame& frame, const OutputDevice& device);

    // Roon Bridge implementation
    class RoonBridgeServer;
    std::map<std::string, std::unique_ptr<RoonBridgeServer>> roonBridges_;
    bool startRoonBridge(const RoonBridgeConfig& config);
    void stopRoonBridge(const std::string& deviceId);

    // HQPlayer NAA implementation
    class HQPlayerNAAClient;
    std::map<std::string, std::unique_ptr<HQPlayerNAAClient>> hqPlayers_;
    bool startHQPlayerNAA(const HQPlayerConfig& config);
    void stopHQPlayerNAA(const std::string& deviceId);

    // UPnP implementation
    class UPnPRenderer;
    std::map<std::string, std::unique_ptr<UPnPRenderer>> upnpRenderers_;
    bool startUPnPRenderer(const UPnPConfig& config);
    void stopUPnPRenderer(const std::string& deviceId);

    // Low-level audio interfaces
    class ASIODevice;
    class WASAPIDevice;
    class CoreAudioDevice;
    class ALSADevice;
    class PulseAudioDevice;
    class JACKDevice;

    // Device-specific implementations
    std::map<std::string, std::unique_ptr<ASIODevice>> asioDevices_;
    std::map<std::string, std::unique_ptr<WASAPIDevice>> wasapiDevices_;
    std::map<std::string, std::unique_ptr<CoreAudioDevice>> coreAudioDevices_;
    std::map<std::string, std::unique_ptr<ALSADevice>> alsaDevices_;
    std::map<std::string, std::unique_ptr<PulseAudioDevice>> pulseAudioDevices_;
    std::map<std::string, std::unique_ptr<JACKDevice>> jackDevices_;

    // State
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shouldShutdown_{false};

    // Device registry
    std::map<std::string, OutputDevice> devices_;
    std::map<std::string, OutputConfig> deviceConfigs_;
    mutable std::mutex devicesMutex_;

    // Zone management
    std::map<std::string, std::vector<std::string>> zones_;
    mutable std::mutex zonesMutex_;

    // Audio processing
    std::queue<AudioFrame> audioQueue_;
    std::mutex audioQueueMutex_;
    std::condition_variable audioQueueCondition_;

    // Threads
    std::unique_ptr<std::thread> discoveryThread_;
    std::unique_ptr<std::thread> monitoringThread_;
    std::unique_ptr<std::thread> processingThread_;
    std::atomic<bool> processingActive_{false};

    // Configuration
    uint32_t maxDevices_ = 8;
    uint32_t updateInterval_ = 100; // ms
    std::atomic<bool> enableDeviceDiscovery_{true};
    std::atomic<bool> enableHealthMonitoring_{true};

    // Audio engine reference
    std::weak_ptr<AudioEngine> audioEngine_;

    // Statistics
    mutable std::mutex statsMutex_;
    OutputStatistics statistics_;

    // Callbacks
    DeviceEventCallback deviceAddedCallback_;
    DeviceEventCallback deviceRemovedCallback_;
    DeviceEventCallback deviceStateChangedCallback_;
    AudioEventCallback audioOutputCallback_;
    ErrorEventCallback errorCallback_;

    // Error handling
    mutable std::string lastError_;
    void setError(const std::string& error) const;

    // Utility methods
    std::string generateDeviceId() const;
    std::string getDeviceTypeName(OutputType type) const;
    bool validateDeviceConfig(const OutputConfig& config) const;
    void notifyDeviceEvent(const std::string& eventType, const OutputDevice& device);
    void notifyError(const std::string& deviceId, const std::string& error);
};

} // namespace vortex