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
 * @brief HQPlayer Network Audio Adapter (NAA) client implementation
 *
 * Implements the HQPlayer NAA protocol for high-quality audio streaming from
 * HQPlayer to VortexGPU, supporting:
 * - Bit-perfect audio transport with HQPlayer
 * - DSD64/128/256/512/1024 support with native DSD streaming
 * - PCM up to 768kHz/32-bit
 * - Real-time streaming with <50ms latency
 * - Automatic sample rate and format detection
 * - Gapless playback support
 * - Multi-channel support (up to 8 channels)
 * - Hardware volume control bypass for bit-perfect mode
 *
 * Features:
 * - Native DSD transport (DoP and native DSD)
 * - Advanced upsampling support (poly-sinc, sinc-M, etc.)
 * - High-quality digital filters
 * - Real-time streaming optimization
 * - Zero-copy audio data path
 * - Advanced dithering algorithms
 * - Dynamic range optimization
 */
class HQPlayerNAAClient {
public:
    enum class ConnectionState {
        DISCONNECTED,
        CONNECTING,
        CONNECTED,
        STREAMING,
        ERROR,
        RECONNECTING
    };

    enum class TransportMode {
        TCP,                    // TCP transport for reliable delivery
        UDP,                    // UDP transport for low latency
        RTP,                    // RTP for synchronized delivery
        WEBSOCKET              // WebSocket for web-based delivery
    };

    enum class UpsamplingFilter {
        NONE,                   // No upsampling
        POLY_SINC,              // Poly-sinc filter
        POLY_SINC_XTR,          // Poly-sinc extra
        POLY_SINC_MP,           // Poly-sinc multi-stage
        SINC_M,                 // Sinc-M filter
        SINC_L,                 // Sinc-L filter
        MIN_PHASE,              // Minimum phase filter
        LINEAR_PHASE,           // Linear phase filter
        FIR_DITHER              // FIR with dithering
    };

    enum class ModulationType {
        NONE,
        SDM_5,                  // 5th order delta-sigma
        SDM_6,                  // 6th order delta-sigma
        SDM_7,                  // 7th order delta-sigma
        SDM_8,                  // 8th order delta-sigma
        PDM,                    // Pulse density modulation
        PWM                     // Pulse width modulation
    };

    struct NAADeviceInfo {
        std::string deviceId;
        std::string deviceName;
        std::string manufacturer;
        std::string model;
        std::string version;
        std::string serialNumber;

        // Capabilities
        std::vector<uint32_t> supportedSampleRates = {44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000, 705600, 768000};
        std::vector<uint16_t> supportedBitDepths = {16, 24, 32};
        std::vector<uint16_t> supportedChannels = {1, 2, 4, 6, 8};
        std::vector<UpsamplingFilter> supportedFilters = {UpsamplingFilter::NONE, UpsamplingFilter::POLY_SINC};

        // DSD capabilities
        bool supportsDSD = true;
        std::vector<uint32_t> dsdRates = {2822400, 5644800, 11289600, 22579200, 45158400}; // DSD64 to DSD1024
        bool supportsDoP = true;
        bool supportsNativeDSD = true;

        // Network capabilities
        std::vector<TransportMode> supportedTransports = {TransportMode::TCP};
        uint32_t maxBitrate = 0;      // 0 = unlimited
        uint32_t preferredLatencyMs = 100;
        uint32_t maxLatencyMs = 1000;

        // Processing capabilities
        bool supportsUpsampling = true;
        uint32_t maxUpsampleRate = 1536000; // 1.536 MHz
        bool supportsVolumeControl = false; // Bit-perfect mode
        bool supportsEQ = false;
        bool supportsCrossfade = true;

        // Device-specific settings
        std::map<std::string, std::variant<int, float, bool, std::string>> customSettings;

        // Status
        ConnectionState state = ConnectionState::DISCONNECTED;
        std::chrono::steady_clock::time_point lastActivity;
        uint64_t bytesReceived = 0;
        uint64_t packetsReceived = 0;
        uint64_t packetsLost = 0;
    };

    struct NAAStream {
        std::string streamId;
        std::string deviceId;

        // Audio properties
        uint32_t sampleRate = 44100;
        uint16_t channels = 2;
        uint16_t bitDepth = 24;
        AudioFormat format = AudioFormat::PCM_S24LE;
        bool isDSD = false;
        uint32_t dsdRate = 0;

        // Processing settings
        UpsamplingFilter upsamplingFilter = UpsamplingFilter::NONE;
        uint32_t targetSampleRate = 0;     // 0 = no upsampling
        ModulationType modulationType = ModulationType::NONE;
        bool enableDithering = false;
        std::string ditheringAlgorithm = "triangular";

        // Stream status
        ConnectionState state = ConnectionState::DISCONNECTED;
        bool isActive = false;
        bool isPaused = false;
        uint64_t position = 0;
        uint64_t duration = 0;
        float bufferHealth = 0.0f;

        // Track information
        std::string trackId;
        std::string title;
        std::string album;
        std::string artist;
        std::map<std::string, std::string> trackMetadata;

        // Network transport
        TransportMode transportMode = TransportMode::TCP;
        uint16_t dataPort = 12345;
        uint16_t controlPort = 12346;
        std::string remoteAddress;
        uint32_t packetSize = 4096;
        uint32_t packetCount = 0;

        // Timing and synchronization
        uint64_t startTime = 0;
        uint64_t currentTime = 0;
        uint64_t nextPacketTime = 0;
        float driftCorrection = 0.0f;
        uint32_t latencyMs = 100;

        // Audio buffer
        std::vector<uint8_t> audioBuffer;
        size_t bufferCapacity = 0;
        size_t bufferUsed = 0;
        std::queue<std::vector<uint8_t>> packetQueue;

        // Statistics
        uint64_t bytesReceived = 0;
        uint64_t packetsReceived = 0;
        uint64_t packetsLost = 0;
        uint64_t underruns = 0;
        uint64_t overruns = 0;
        float averageLatency = 0.0f;
        float bufferUtilization = 0.0f;

        // Timestamps
        std::chrono::steady_clock::time_point creationTime;
        std::chrono::steady_clock::time_point lastPacketTime;
        std::chrono::steady_clock::time_point lastUpdateTime;
    };

    struct NAAConfig {
        std::string deviceId;
        std::string deviceName = "VortexGPU NAA";
        std::pair<std::string, std::string> serverAddress = {"127.0.0.1", ""};
        uint16_t serverPort = 12345;

        // Connection settings
        TransportMode transportMode = TransportMode::TCP;
        bool enableAutoReconnect = true;
        uint32_t reconnectInterval = 5000; // ms
        uint32_t maxReconnectAttempts = 10;
        uint32_t connectionTimeout = 10000; // ms

        // Audio settings
        uint32_t maxSampleRate = 768000;
        uint16_t maxChannels = 8;
        uint16_t maxBitDepth = 32;
        bool enableBitPerfectMode = true;
        bool enableVolumeControl = false;

        // DSD settings
        bool enableDSDSupport = true;
        bool enableDoP = true;
        bool enableNativeDSD = true;
        uint32_t maxDSDRate = 1024000; // DSD1024

        // Upsampling settings
        bool enableUpsampling = true;
        UpsamplingFilter defaultFilter = UpsamplingFilter::POLY_SINC;
        uint32_t defaultTargetRate = 768000;
        bool enableAdaptiveFilter = true;
        bool enableHighQualityMode = true;

        // Buffer settings
        uint32_t bufferSize = 32768;      // bytes
        uint32_t bufferCount = 4;
        uint32_t packetSize = 4096;
        uint32_t targetLatency = 100;     // ms

        // Network settings
        bool enableCompression = false;
        bool enableEncryption = false;
        uint32_t networkTimeout = 5000;   // ms
        uint32_t keepAliveInterval = 30000; // ms

        // Processing settings
        bool enableDithering = false;
        std::string ditheringAlgorithm = "triangular";
        bool enableNoiseShaping = false;
        uint32_t noiseShapingOrder = 3;

        // Monitoring and logging
        bool enableMetrics = true;
        bool enableHealthMonitoring = true;
        uint32_t metricsInterval = 1000;  // ms
        bool enableDetailedLogging = false;
    };

    HQPlayerNAAClient(const NAAConfig& config);
    ~HQPlayerNAAClient();

    // Connection management
    bool connect();
    void disconnect();
    bool isConnected() const;
    ConnectionState getConnectionState() const;

    // Device management
    bool registerDevice(const NAADeviceInfo& device);
    NAADeviceInfo getDeviceInfo() const;
    bool updateDeviceInfo(const NAADeviceInfo& device);
    std::vector<UpsamplingFilter> getAvailableFilters() const;

    // Stream management
    std::string createStream();
    bool startStream(const std::string& streamId);
    bool pauseStream(const std::string& streamId);
    bool stopStream(const std::string& streamId);
    bool seekStream(const std::string& streamId, uint64_t position);
    NAAStream getStream(const std::string& streamId) const;
    std::vector<NAAStream> getActiveStreams() const;

    // Audio processing
    bool receiveAudioPacket(const std::string& streamId, const std::vector<uint8_t>& packet);
    bool processAudioBuffer(const std::string& streamId, std::vector<float>& audioBuffer);
    void configureUpsampling(const std::string& streamId, UpsamplingFilter filter, uint32_t targetRate);
    void setDithering(const std::string& streamId, bool enable, const std::string& algorithm);

    // HQPlayer protocol implementation
    bool handleHandshake();
    bool handleDeviceInfoRequest();
    bool handleStreamConfiguration(const std::string& streamId);
    bool handleVolumeControl(float volume);
    bool handleMuteControl(bool muted);
    bool handleSampleRateChange(uint32_t newRate);
    bool handleBitDepthChange(uint16_t newBitDepth);

    // Statistics and monitoring
    struct ClientStatistics {
        ConnectionState connectionState = ConnectionState::DISCONNECTED;
        uint64_t totalPacketsReceived = 0;
        uint64_t totalBytesReceived = 0;
        uint64_t totalPacketsLost = 0;
        uint64_t totalUnderruns = 0;
        uint64_t totalOverruns = 0;
        uint32_t activeStreams = 0;
        float averageLatency = 0.0f;
        float packetLossRate = 0.0f;
        float bufferUtilization = 0.0f;
        std::chrono::steady_clock::time_point connectionTime;
        std::chrono::steady_clock::time_point lastActivity;
        std::map<std::string, uint64_t> packetsByStream;
        std::map<UpsamplingFilter, uint64_t> streamsByFilter;
    };

    ClientStatistics getStatistics() const;
    void resetStatistics();

    // Health monitoring
    struct HealthStatus {
        bool isHealthy = true;
        bool isConnected = false;
        bool hasActiveStreams = false;
        bool packetLossWithinLimits = true;
        bool latencyWithinLimits = true;
        std::vector<std::string> warnings;
        std::vector<std::string> errors;
        float cpuUsage = 0.0f;
        uint64_t memoryUsage = 0;
        uint32_t reconnectAttempts = 0;
    };

    HealthStatus getHealthStatus() const;

    // Configuration
    void updateConfiguration(const NAAConfig& config);
    NAAConfig getConfiguration() const;

    // Event callbacks
    using ConnectionStateCallback = std::function<void(ConnectionState)>;
    using StreamEventCallback = std::function<void(const std::string&, const NAAStream&)>;
    using AudioEventCallback = std::function<void(const std::string&, const std::vector<float>&)>;
    using ErrorCallback = std::function<void(const std::string&, const std::string&)>;

    void setConnectionStateChangedCallback(ConnectionStateCallback callback);
    void setStreamStartedCallback(StreamEventCallback callback);
    void setStreamStoppedCallback(StreamEventCallback callback);
    void setAudioReceivedCallback(AudioEventCallback callback);
    void setErrorCallback(ErrorCallback callback);

private:
    // Network communication
    class NetworkTransport;
    class TCPTransport;
    class UDPTransport;
    class RTPTransport;
    class WebSocketTransport;

    std::unique_ptr<NetworkTransport> transport_;

    // Connection management
    void connectionThread();
    void monitorConnection();
    void attemptReconnection();
    void setConnectionState(ConnectionState state);

    // Audio processing
    void audioProcessingThread();
    void processStreamAudio(const std::string& streamId);
    bool applyUpsampling(const std::vector<float>& input, std::vector<float>& output,
                        uint32_t inputRate, uint32_t outputRate, UpsamplingFilter filter);
    bool applyDithering(std::vector<float>& audioData, const std::string& algorithm);
    bool applyNoiseShaping(std::vector<float>& audioData, uint32_t order);

    // Protocol handling
    bool processControlMessage(const std::vector<uint8_t>& message);
    bool processDataPacket(const std::string& streamId, const std::vector<uint8_t>& packet);
    std::vector<uint8_t> createControlMessage(const std::string& command, const std::map<std::string, std::string>& parameters);
    bool sendControlMessage(const std::vector<uint8_t>& message);

    // Stream management
    std::string generateStreamId();
    bool validateStreamId(const std::string& streamId) const;
    void cleanupInactiveStreams();
    void updateStreamStatistics(const std::string& streamId);

    // Buffer management
    bool initializeStreamBuffer(NAAStream& stream);
    void cleanupStreamBuffer(NAAStream& stream);
    bool writeToStreamBuffer(NAAStream& stream, const std::vector<uint8_t>& data);
    bool readFromStreamBuffer(NAAStream& stream, std::vector<uint8_t>& data, size_t bytes);

    // Timing and synchronization
    void synchronizeStreamTiming(NAAStream& stream);
    void calculateDriftCorrection(NAAStream& stream);
    void adjustPlaybackTiming(NAAStream& stream);

    // Configuration validation
    bool validateConfiguration(const NAAConfig& config) const;
    bool validateDeviceInfo(const NAADeviceInfo& device) const;

    // Error handling
    void setError(const std::string& error);
    void handleStreamError(const std::string& streamId, const std::string& error);
    void handleConnectionError(const std::string& error);

    // Utility methods
    std::string getConnectionStateName(ConnectionState state) const;
    std::string getTransportModeName(TransportMode mode) const;
    std::string getUpsamplingFilterName(UpsamplingFilter filter) const;
    uint32_t calculatePacketSize(uint32_t sampleRate, uint16_t channels, uint16_t bitDepth) const;

    // State
    NAAConfig config_;
    NAADeviceInfo deviceInfo_;
    std::atomic<ConnectionState> connectionState_{ConnectionState::DISCONNECTED};

    // Stream management
    std::map<std::string, NAAStream> streams_;
    mutable std::mutex streamsMutex_;

    // Audio processing
    std::unique_ptr<std::thread> connectionThread_;
    std::unique_ptr<std::thread> processingThread_;
    std::queue<std::string> processingQueue_;
    std::mutex processingQueueMutex_;
    std::condition_variable processingCondition_;
    std::atomic<bool> processingActive_{false};

    // Statistics
    mutable std::mutex statsMutex_;
    ClientStatistics statistics_;

    // Callbacks
    ConnectionStateCallback connectionStateChangedCallback_;
    StreamEventCallback streamStartedCallback_;
    StreamEventCallback streamStoppedCallback_;
    AudioEventCallback audioReceivedCallback_;
    ErrorCallback errorCallback_;

    // Error handling
    mutable std::string lastError_;
    std::atomic<uint32_t> reconnectAttempts_{0};
};

/**
 * @brief Network transport base class for HQPlayer NAA
 */
class HQPlayerNAAClient::NetworkTransport {
public:
    virtual ~NetworkTransport() = default;

    virtual bool connect(const std::pair<std::string, std::string>& address) = 0;
    virtual void disconnect() = 0;
    virtual bool isConnected() const = 0;
    virtual bool sendControlMessage(const std::vector<uint8_t>& message) = 0;
    virtual bool sendDataPacket(const std::vector<uint8_t>& data) = 0;
    virtual std::vector<uint8_t> receiveMessage(uint32_t timeoutMs) = 0;
    virtual uint32_t getLatency() const = 0;

protected:
    std::atomic<bool> connected_{false};
};

/**
 * @brief TCP transport implementation
 */
class HQPlayerNAAClient::TCPTransport : public NetworkTransport {
public:
    TCPTransport();
    ~TCPTransport() override;

    bool connect(const std::pair<std::string, std::string>& address) override;
    void disconnect() override;
    bool isConnected() const override;
    bool sendControlMessage(const std::vector<uint8_t>& message) override;
    bool sendDataPacket(const std::vector<uint8_t>& data) override;
    std::vector<uint8_t> receiveMessage(uint32_t timeoutMs) override;
    uint32_t getLatency() const override;

private:
    int socket_ = -1;
    std::string remoteHost_;
    uint16_t remotePort_ = 0;
    mutable std::mutex socketMutex_;
};

/**
 * @brief UDP transport implementation
 */
class HQPlayerNAAClient::UDPTransport : public NetworkTransport {
public:
    UDPTransport();
    ~UDPTransport() override;

    bool connect(const std::pair<std::string, std::string>& address) override;
    void disconnect() override;
    bool isConnected() const override;
    bool sendControlMessage(const std::vector<uint8_t>& message) override;
    bool sendDataPacket(const std::vector<uint8_t>& data) override;
    std::vector<uint8_t> receiveMessage(uint32_t timeoutMs) override;
    uint32_t getLatency() const override;

private:
    int socket_ = -1;
    sockaddr_in remoteAddr_;
    std::string remoteHost_;
    uint16_t remotePort_ = 0;
    mutable std::mutex socketMutex_;
};

} // namespace vortex