#pragma once

#include "../audio_types.hpp"
#include "../network_types.hpp"

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <functional>

namespace vortex {

/**
 * @brief Roon Bridge server implementation for high-quality audio streaming
 *
 * Implements the Roon Bridge protocol to enable VortexGPU to act as a Roon
 * endpoint, providing bit-perfect audio streaming with support for:
 * - Roon Advanced Audio Transport (RAAT)
 * - AirPlay 2 compatibility
 * - MQA (Master Quality Authenticated) playback
 * - DSD64/128/256/512/1024 support
 * - Gapless playback with sample-accurate timing
 * - Multi-zone support
 * - Device hot-plugging
 * - Automatic discovery via Roon ARC
 *
 * Features:
 * - Up to 768kHz PCM and DSD1024 support
 * - 32-bit float processing pipeline
 * - Zero-copy audio data path
 * - Real-time latency monitoring
 * - Multi-device synchronization
 * - Hardware passthrough modes
 * - Volume control bypass for bit-perfect output
 */
class RoonBridgeServer {
public:
    enum class TransportProtocol {
        RAAT,               // Roon Advanced Audio Transport
        AIRPLAY,            // Apple AirPlay 2
        MERIDIAN,           // Meridian Sooloos
        DLNA                // DLNA/UPnP fallback
    };

    enum class DeviceType {
        UNKNOWN,
        USB_DAC,
        NETWORK_RENDERER,
        BUILTIN_SPEAKERS,
        HEADPHONES,
        OPTICAL_OUTPUT,
        COAXIAL_OUTPUT,
        BALANCED_OUTPUT,
        VIRTUAL_DEVICE
    };

    enum class BitDepth {
        BIT_16,
        BIT_24,
        BIT_32,
        BIT_64
    };

    struct RoonDevice {
        std::string deviceId;
        std::string displayName;
        DeviceType type = DeviceType::UNKNOWN;
        bool isActive = false;

        // Audio capabilities
        std::vector<uint32_t> supportedSampleRates = {44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000, 705600, 768000};
        std::vector<BitDepth> supportedBitDepths = {BitDepth::BIT_16, BitDepth::BIT_24, BitDepth::BIT_32};
        std::vector<uint16_t> supportedChannels = {1, 2};
        bool supportsDSD = true;
        bool supportsMQA = false;
        bool supportsPCM = true;

        // Latency information
        uint32_t minLatencyMs = 50;
        uint32_t maxLatencyMs = 2000;
        uint32_t currentLatencyMs = 100;

        // Device settings
        bool exclusiveMode = false;
        bool volumeControlBypass = true;
        bool enableGapless = true;
        bool enableCrossfade = false;

        // Streaming settings
        TransportProtocol primaryProtocol = TransportProtocol::RAAT;
        std::vector<TransportProtocol> supportedProtocols;
        uint32_t maxBitrate = 0;      // 0 = unlimited
        bool enableCompression = false;

        // Device metadata
        std::string manufacturer;
        std::string model;
        std::string version;
        std::string serialNumber;
        std::map<std::string, std::string> customMetadata;

        // Statistics
        uint64_t tracksPlayed = 0;
        uint64_t bytesReceived = 0;
        uint64_t underruns = 0;
        uint64_t overruns = 0;
        std::chrono::steady_clock::time_point lastActivity;
    };

    struct RoonStream {
        std::string streamId;
        std::string deviceId;
        TransportProtocol protocol = TransportProtocol::RAAT;

        // Stream properties
        uint32_t sampleRate = 44100;
        uint16_t channels = 2;
        BitDepth bitDepth = BitDepth::BIT_16;
        bool isDSD = false;
        uint32_t dsdRate = 0;

        // Stream status
        bool isActive = false;
        bool isPaused = false;
        uint64_t position = 0;
        uint64_t duration = 0;
        float bufferHealth = 0.0f;

        // Track information
        std::string trackId;
        std::string albumId;
        std::string artistId;
        std::string title;
        std::string album;
        std::string artist;
        std::map<std::string, std::string> trackMetadata;

        // Audio data
        std::vector<uint8_t> audioBuffer;
        size_t bufferCapacity = 0;
        size_t bufferUsed = 0;

        // Timestamps
        uint64_t presentationTime = 0;
        uint64_t startTime = 0;
        uint64_t endTime = 0;
        std::chrono::steady_clock::time_point creationTime;
        std::chrono::steady_clock::time_point lastUpdate;
    };

    struct RoonBridgeConfig {
        std::string deviceId;
        std::string displayName = "VortexGPU Audio";
        std::pair<std::string, std::string> ipAddress = {"0.0.0.0", ""};
        uint16_t raatPort = 9100;
        uint16_t airplayPort = 5000;
        uint16_t httpPort = 9101;

        // Capabilities
        bool enableRAAT = true;
        bool enableAirPlay = false;
        bool enableMeridian = false;
        bool enableDLNA = false;
        bool supportMQA = false;
        bool supportDSD = true;
        uint32_t maxSampleRate = 768000;
        uint32_t maxDSDRate = 512;

        // Device settings
        bool exclusiveMode = false;
        bool volumeControlBypass = true;
        uint32_t maxLatencyMs = 2000;
        uint32_t preferredLatencyMs = 100;
        bool enableGapless = true;
        bool enableCrossfade = false;

        // Zone configuration
        std::string zoneName = "VortexGPU Zone";
        std::vector<std::string> zoneAliases;
        bool enableMultiZone = false;
        uint32_t maxZones = 1;

        // Network settings
        bool enableAutoDiscovery = true;
        bool requireAuthentication = false;
        std::string authToken;
        bool enableCompression = false;
        uint32_t maxBitrate = 0;

        // Audio processing
        bool enableVolumeControl = true;
        bool enableDSDConversion = true;
        bool enableUpsampling = false;
        uint32_t upsampleTargetRate = 0;
        std::string resamplerQuality = "high";

        // Logging and monitoring
        bool enableLogging = true;
        bool enableMetrics = true;
        uint32_t logLevel = 1;  // 0=error, 1=warn, 2=info, 3=debug
        bool enableHealthChecks = true;
        uint32_t healthCheckInterval = 5000; // ms
    };

    RoonBridgeServer(const RoonBridgeConfig& config);
    ~RoonBridgeServer();

    // Server lifecycle
    bool start();
    void stop();
    bool isRunning() const;
    bool isReady() const;

    // Device management
    bool registerDevice(const RoonDevice& device);
    bool unregisterDevice(const std::string& deviceId);
    RoonDevice getDevice(const std::string& deviceId) const;
    std::vector<RoonDevice> getRegisteredDevices() const;
    bool updateDeviceCapabilities(const std::string& deviceId, const RoonDevice& device);

    // Stream management
    std::string createStream(const std::string& deviceId, TransportProtocol protocol);
    bool startStream(const std::string& streamId);
    bool pauseStream(const std::string& streamId);
    bool stopStream(const std::string& streamId);
    bool seekStream(const std::string& streamId, uint64_t position);
    RoonStream getStream(const std::string& streamId) const;
    std::vector<RoonStream> getActiveStreams() const;

    // Audio processing
    bool receiveAudioData(const std::string& streamId, const std::vector<uint8_t>& audioData, uint64_t timestamp);
    bool processAudioFrame(const std::string& streamId, std::vector<float>& audioBuffer);
    void setStreamVolume(const std::string& streamId, float volume);
    void setStreamMute(const std::string& streamId, bool muted);

    // Roon protocol implementation
    bool handleRoonDiscovery();
    bool handleRoonAuthentication(const std::string& authToken);
    bool handleRoonDeviceInfo();
    bool handleRoonPlaybackCommand(const std::string& deviceId, const std::string& command);
    bool handleRoonVolumeCommand(const std::string& deviceId, const std::string& command, float value);

    // Statistics and monitoring
    struct BridgeStatistics {
        uint64_t totalStreamsCreated = 0;
        uint64_t totalAudioBytesReceived = 0;
        uint64_t totalTracksPlayed = 0;
        uint64_t totalUnderruns = 0;
        uint64_t totalOverruns = 0;
        uint32_t activeStreams = 0;
        uint32_t registeredDevices = 0;
        float averageLatency = 0.0f;
        std::chrono::steady_clock::time_point startTime;
        std::chrono::steady_clock::time_point lastActivity;
        std::map<std::string, uint64_t> streamsByDevice;
        std::map<TransportProtocol, uint64_t> streamsByProtocol;
    };

    BridgeStatistics getStatistics() const;
    void resetStatistics();

    // Health monitoring
    struct HealthStatus {
        bool isHealthy = true;
        bool raatServerRunning = false;
        bool airplayServerRunning = false;
        bool httpServerRunning = false;
        uint32_t activeStreams = 0;
        uint32_t healthyStreams = 0;
        std::vector<std::string> warnings;
        std::vector<std::string> errors;
        float cpuUsage = 0.0f;
        uint64_t memoryUsage = 0;
    };

    HealthStatus getHealthStatus() const;

    // Configuration
    void updateConfiguration(const RoonBridgeConfig& config);
    RoonBridgeConfig getConfiguration() const;

    // Event callbacks
    using StreamEventCallback = std::function<void(const std::string&, const RoonStream&)>;
    using DeviceEventCallback = std::function<void(const std::string&, const RoonDevice&)>;
    using AudioEventCallback = std::function<void(const std::string&, const std::vector<float>&)>;
    using ErrorCallback = std::function<void(const std::string&, const std::string&)>;

    void setStreamStartedCallback(StreamEventCallback callback);
    void setStreamStoppedCallback(StreamEventCallback callback);
    void setDeviceRegisteredCallback(DeviceEventCallback callback);
    void setDeviceUnregisteredCallback(DeviceEventCallback callback);
    void setAudioReceivedCallback(AudioEventCallback callback);
    void setErrorCallback(ErrorCallback callback);

private:
    // Network servers
    class RAATServer;
    class AirPlayServer;
    class HTTPServer;
    class DiscoveryServer;

    std::unique_ptr<RAATServer> raatServer_;
    std::unique_ptr<AirPlayServer> airplayServer_;
    std::unique_ptr<HTTPServer> httpServer_;
    std::unique_ptr<DiscoveryServer> discoveryServer_;

    // Server management
    bool startRAATServer();
    bool startAirPlayServer();
    bool startHTTPServer();
    bool startDiscoveryServer();

    void stopRAATServer();
    void stopAirPlayServer();
    void stopHTTPServer();
    void stopDiscoveryServer();

    // Audio processing
    void audioProcessingThread();
    void processRAATStream(const std::string& streamId);
    void processAirPlayStream(const std::string& streamId);

    // Protocol handling
    bool processRAATMessage(const std::vector<uint8_t>& message);
    bool processAirPlayMessage(const std::vector<uint8_t>& message);
    bool processHTTPRequest(const std::string& method, const std::string& path, const std::string& body);

    // Device discovery
    void deviceDiscoveryThread();
    void announceDevicePresence();
    void handleDeviceDisconnection();

    // Stream management
    std::string generateStreamId();
    bool validateStreamId(const std::string& streamId) const;
    void cleanupInactiveStreams();

    // Audio processing
    bool convertAudioFormat(const std::vector<uint8_t>& input, std::vector<float>& output,
                           const RoonStream& stream);
    bool resampleAudio(const std::vector<float>& input, std::vector<float>& output,
                      uint32_t inputRate, uint32_t outputRate);
    bool applyVolumeControl(std::vector<float>& audioData, float volume);

    // Configuration validation
    bool validateConfiguration(const RoonBridgeConfig& config) const;
    bool validateDevice(const RoonDevice& device) const;

    // Error handling
    void setError(const std::string& error);
    void handleStreamError(const std::string& streamId, const std::string& error);
    void handleDeviceError(const std::string& deviceId, const std::string& error);

    // Utility methods
    std::string getProtocolName(TransportProtocol protocol) const;
    std::string getDeviceTypeName(DeviceType type) const;
    std::string getBitDepthName(BitDepth bitDepth) const;
    uint32_t getBitDepthValue(BitDepth bitDepth) const;

    // State
    RoonBridgeConfig config_;
    std::atomic<bool> running_{false};
    std::atomic<bool> ready_{false};

    // Device registry
    std::map<std::string, RoonDevice> devices_;
    mutable std::mutex devicesMutex_;

    // Stream management
    std::map<std::string, RoonStream> streams_;
    mutable std::mutex streamsMutex_;

    // Audio processing
    std::unique_ptr<std::thread> processingThread_;
    std::unique_ptr<std::thread> discoveryThread_;
    std::queue<std::string> processingQueue_;
    std::mutex processingQueueMutex_;
    std::condition_variable processingCondition_;

    // Statistics
    mutable std::mutex statsMutex_;
    BridgeStatistics statistics_;

    // Callbacks
    StreamEventCallback streamStartedCallback_;
    StreamEventCallback streamStoppedCallback_;
    DeviceEventCallback deviceRegisteredCallback_;
    DeviceEventCallback deviceUnregisteredCallback_;
    AudioEventCallback audioReceivedCallback_;
    ErrorCallback errorCallback_;

    // Error handling
    mutable std::string lastError_;
};

/**
 * @brief Roon Advanced Audio Transport (RAAT) protocol server
 *
 * Implements the RAAT protocol for high-quality, bit-perfect audio streaming
 * from Roon to VortexGPU endpoints.
 */
class RoonBridgeServer::RAATServer {
public:
    RAATServer(uint16_t port, RoonBridgeServer* parent);
    ~RAATServer();

    bool start();
    void stop();
    bool isRunning() const;

    bool sendDeviceInfo();
    bool sendStreamStatus(const std::string& streamId);
    bool sendTimeSync();
    bool handleAuthentication(const std::string& authToken);

private:
    void serverThread();
    void handleClientConnection(int clientSocket);
    bool processMessage(const std::vector<uint8_t>& message);
    void sendMessage(const std::vector<uint8_t>& message);

    uint16_t port_;
    RoonBridgeServer* parent_;
    std::unique_ptr<std::thread> serverThread_;
    std::atomic<bool> running_{false};
    int serverSocket_ = -1;
    std::vector<int> clientSockets_;
    std::mutex clientSocketsMutex_;
};

/**
 * @brief Apple AirPlay 2 server implementation
 *
 * Implements AirPlay 2 protocol for compatibility with Apple devices
 * and Roon's AirPlay output.
 */
class RoonBridgeServer::AirPlayServer {
public:
    AirPlayServer(uint16_t port, RoonBridgeServer* parent);
    ~AirPlayServer();

    bool start();
    void stop();
    bool isRunning() const;

private:
    void serverThread();
    void handleAirPlayRequest(const std::string& request);
    void sendAirPlayResponse(const std::string& response);

    uint16_t port_;
    RoonBridgeServer* parent_;
    std::unique_ptr<std::thread> serverThread_;
    std::atomic<bool> running_{false};
};

/**
 * @brief HTTP server for Roon Bridge control interface
 *
 * Provides HTTP endpoints for device information, status monitoring,
 * and Roon Bridge control commands.
 */
class RoonBridgeServer::HTTPServer {
public:
    HTTPServer(uint16_t port, RoonBridgeServer* parent);
    ~HTTPServer();

    bool start();
    void stop();
    bool isRunning() const;

private:
    void serverThread();
    void handleHTTPRequest(const std::string& method, const std::string& path, const std::string& body);
    std::string generateDeviceInfoJSON();
    std::string generateStreamStatusJSON();
    std::string generateStatisticsJSON();

    uint16_t port_;
    RoonBridgeServer* parent_;
    std::unique_ptr<std::thread> serverThread_;
    std::atomic<bool> running_{false};
};

/**
 * @brief mDNS/Bonjour discovery server for Roon Bridge
 *
 * Handles automatic discovery by Roon via mDNS/Bonjour service announcements.
 */
class RoonBridgeServer::DiscoveryServer {
public:
    DiscoveryServer(RoonBridgeServer* parent);
    ~DiscoveryServer();

    bool start();
    void stop();
    bool isRunning() const;

    void announceService();
    void withdrawService();

private:
    void discoveryThread();
    void handleDiscoveryQuery(const std::string& query);
    void sendDiscoveryResponse(const std::string& response);

    RoonBridgeServer* parent_;
    std::unique_ptr<std::thread> discoveryThread_;
    std::atomic<bool> running_{false};
    bool serviceAnnounced_ = false;
};

} // namespace vortex