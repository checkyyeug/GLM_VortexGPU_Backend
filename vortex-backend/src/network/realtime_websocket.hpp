#pragma once

#include "../audio_types.hpp"
#include "../network_types.hpp"
#include "websocket_server.hpp"

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <condition_variable>
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

namespace vortex {

// Forward declarations
class SpectrumAnalyzer;
class WaveformProcessor;
class VUMeterProcessor;
class HardwareMonitor;

using WebSocketServer = websocketpp::server<websocketpp::config::asio>;
using ConnectionHdl = websocketpp::connection_hdl;

/**
 * @brief Real-time WebSocket data streaming for audio visualization
 *
 * Provides high-performance real-time streaming of audio visualization data
 * including spectrum analysis, waveform display, VU meters, and system monitoring.
 *
 * Features:
 * - 60fps real-time streaming with <50ms end-to-end latency
 * - Binary protocol with compression for bandwidth efficiency
 * - Multiple subscription types with adaptive quality
 * - Client-side rate limiting and adaptive quality scaling
 * - Hardware acceleration for data processing
 * - Automatic connection recovery and management
 * - Real-time hardware monitoring (GPU, CPU, memory)
 */
class RealTimeWebSocket {
public:
    enum class SubscriptionType {
        NONE = 0,
        SPECTRUM = 1 << 0,
        WAVEFORM = 1 << 1,
        VU_METERS = 1 << 2,
        HARDWARE = 1 << 3,
        METRICS = 1 << 4,
        AUDIO_STATUS = 1 << 5,
        PROCESSING_STATUS = 1 << 6,
        ALL = SPECTRUM | WAVEFORM | VU_METERS | HARDWARE | METRICS | AUDIO_STATUS | PROCESSING_STATUS
    };

    struct ClientSubscription {
        SubscriptionType types = SubscriptionType::NONE;
        uint32_t updateRate = 60;           // Updates per second
        std::vector<std::string> filters;     // Optional filters
        bool enableCompression = true;
        float qualityLevel = 1.0f;          // 0.0-1.0
        bool enableAdaptiveQuality = true;
        std::chrono::steady_clock::time_point lastUpdate;
        uint64_t messagesSent = 0;
        std::chrono::steady_clock::time_point subscriptionTime;
    };

    struct StreamingConfig {
        uint16_t port = 8081;
        std::string bindAddress = "0.0.0.0";
        uint32_t maxConnections = 1000;
        uint32_t connectionTimeout = 30;    // seconds
        bool enableCompression = true;
        bool enableBinaryProtocol = true;
        bool enableSSL = false;
        std::string sslCertificateFile;
        std::string sslPrivateKeyFile;
        uint32_t updateThreadInterval = 1; // milliseconds
        uint32_t maxMessageSize = 1024 * 1024; // 1MB
        float maxBandwidthPerClient = 1000.0f; // KB/s
        bool enableRateLimiting = true;
    };

    RealTimeWebSocket();
    ~RealTimeWebSocket();

    // Initialization
    bool initialize(const StreamingConfig& config);
    void shutdown();
    bool isInitialized() const;

    // Server control
    bool start();
    bool stop();
    bool isRunning() const;

    // Client management
    uint32_t getConnectedClients() const;
    std::vector<std::string> getClientIds() const;
    ClientSubscription getClientSubscription(const std::string& clientId) const;
    bool hasClient(const std::string& clientId) const;

    // Data processing integration
    void setSpectrumAnalyzer(std::shared_ptr<SpectrumAnalyzer> analyzer);
    void setWaveformProcessor(std::shared_ptr<WaveformProcessor> processor);
    void setVUMeterProcessor(std::shared_ptr<VUMeterProcessor> processor);
    void setHardwareMonitor(std::shared_ptr<HardwareMonitor> monitor);

    // Real-time data publishing
    void publishSpectrumData(const std::vector<float>& spectrum, const SpectrumData& metadata);
    void publishWaveformData(const std::vector<float>& leftChannel, const std::vector<float>& rightChannel,
                             const WaveformData& metadata);
    void publishVUMeterData(const VUMeterData& vuData);
    void publishHardwareStatus(const HardwareStatus& hardware);
    void publishProcessingMetrics(const ProcessingMetrics& metrics);
    void publishAudioStatus(const std::string& fileId, ProcessingStatus status, float progress);

    // Broadcast to all subscribers
    void broadcastToAll(SubscriptionType types, const std::string& dataType, const std::vector<uint8_t>& data);
    void broadcastToAll(const std::string& jsonMessage);

    // Statistics and monitoring
    struct StreamingStats {
        uint64_t totalConnections = 0;
        uint32_t currentConnections = 0;
        uint64_t messagesBroadcast = 0;
        uint64_t totalBytesTransmitted = 0;
        double averageLatency = 0.0;
        double maxLatency = 0.0;
        double minLatency = std::numeric_limits<double>::max();
        std::map<SubscriptionType, uint64_t> subscriptionCounts;
        std::map<std::string, uint64_t> messageCountsByType;
        std::chrono::steady_clock::time_point startTime;
        std::chrono::steady_clock::time_point lastActivity;
    };

    StreamingStats getStatistics() const;
    void resetStatistics();

    // Quality management
    struct QualityMetrics {
        float spectrumQuality = 1.0f;          // Frequency resolution
        float waveformQuality = 1.0f;         // Time resolution
        float vuMeterQuality = 1.0f;           // Dynamic range
        float updateRate = 60.0f;               // Actual update rate
        float bandwidthUtilization = 0.0f;      // % of allocated bandwidth
        uint32_t droppedMessages = 0;
        uint32_t compressedMessages = 0;
        float compressionRatio = 0.0f;
    };

    QualityMetrics getQualityMetrics() const;
    void setQualityTarget(float targetQuality);
    void enableAdaptiveQuality(bool enable);

    // Client-specific operations
    bool sendToClient(const std::string& clientId, const std::string& message);
    bool sendToClient(const std::string& clientId, const std::vector<uint8_t>& binaryData);
    bool kickClient(const std::string& clientId, const std::string& reason = "");
    bool updateClientSubscription(const std::string& clientId, const ClientSubscription& subscription);

private:
    // WebSocket server management
    struct ClientConnection {
        ConnectionHdl connection;
        std::string clientId;
        std::string remoteAddress;
        std::chrono::steady_clock::time_point connectTime;
        std::chrono::steady_clock::time_point lastActivity;
        ClientSubscription subscription;
        std::queue<std::chrono::steady_clock::time_point> messageTimestamps;
        uint64_t bytesTransmitted = 0;
        bool isAuthenticated = false;
        bool isRateLimited = false;
        std::chrono::steady_clock::time_point rateLimitReset;
        uint32_t messagesInPeriod = 0;
        bool shouldCompress = false;
    };

    bool initializeServer();
    void shutdownServer();
    void handleOpen(ConnectionHdl hdl);
    void handleClose(ConnectionHdl hdl);
    void handleMessage(ConnectionHdl hdl, WebSocketServer::message_ptr msg);
    void handleError(ConnectionHdl hdl);

    // Client management
    std::string generateClientId();
    void registerClient(ConnectionHdl hdl);
    void unregisterClient(const std::string& clientId);
    void updateClientActivity(const std::string& clientId);
    bool isValidClientId(const std::string& clientId) const;

    // Message handling
    void handleSubscriptionMessage(const std::string& clientId, const std::string& message);
    void handleUnsubscriptionMessage(const std::string& clientId, const std::string& message);
    void handlePingMessage(const std::string& clientId, const std::string& message);
    void handleCustomMessage(const std::string& clientId, const std::string& message);

    // Data processing and broadcasting
    void updateProcessingThread();
    void broadcastDataToSubscribers(SubscriptionType type, const std::string& dataType,
                                        const std::vector<uint8_t>& data);
    void processSubscriptionUpdates();

    // Rate limiting
    bool isRateLimited(const std::string& clientId);
    void updateRateLimiting(const std::string& clientId);
    void resetRateLimiting(const std::string& clientId);

    // Adaptive quality management
    void updateAdaptiveQuality(const std::string& clientId);
    float calculateOptimalQuality(const std::string& clientId);

    // Message formatting
    std::vector<uint8_t> formatSpectrumMessage(const std::vector<float>& spectrum, const SpectrumData& metadata);
    std::vector<uint8_t> formatWaveformMessage(const std::vector<float>& leftChannel,
                                                const std::vector<float>& rightChannel, const WaveformData& metadata);
    std::vector<uint8_t> formatVUMeterMessage(const VUMeterData& vuData);
    std::vector<uint8_t> formatHardwareMessage(const HardwareStatus& hardware);
    std::vector<uint8_t> formatMetricsMessage(const ProcessingMetrics& metrics);
    std::vector<uint8_t> formatStatusMessage(const std::string& fileId, ProcessingStatus status, float progress);
    std::vector<uint8_t> formatErrorMessage(const std::string& error);

    // JSON serialization
    std::string spectrumToJSON(const std::vector<float>& spectrum, const SpectrumData& metadata);
    std::string waveformToJSON(const std::vector<float>& leftChannel, const std::vector<float>& rightChannel,
                             const WaveformData& metadata);
    std::string vuMeterToJSON(const VUMeterData& vuData);
    std::string hardwareToJSON(const HardwareStatus& hardware);
    std::string metricsToJSON(const std::processing_metrics& metrics);
    std::string statusToJSON(const std::string& fileId, ProcessingStatus status, float progress);
    std::string errorToJSON(const std::string& error);

    // Binary protocol
    struct BinaryHeader {
        uint8_t magic[4] = {0x56, 0x54, 0x58, 0x58}; // "VTXX"
        uint16_t version = 1;
        uint16_t messageType = 0;
        uint32_t payloadSize = 0;
        uint64_t timestamp = 0;
        uint32_t sequenceNumber = 0;
        uint16_t flags = 0;
        uint16_t compression = 0;
    };

    std::vector<uint8_t> createBinaryMessage(uint16_t messageType, const std::vector<uint8_t>& payload);
    std::vector<uint8_t> compressData(const std::vector<uint8_t>& data);
    std::vector<uint8_t> decompressData(const std::vector<uint8_t>& data);

    // State
    std::unique_ptr<WebSocketServer> server_;
    std::unordered_map<std::string, std::unique_ptr<ClientConnection>> clients_;
    mutable std::mutex clientsMutex_;

    // Data processors
    std::shared_ptr<SpectrumAnalyzer> spectrumAnalyzer_;
    std::shared_ptr<WaveformProcessor> waveformProcessor_;
    std::shared_ptr<VUMeterProcessor> vuMeterProcessor_;
    std::shared_ptr<HardwareMonitor> hardwareMonitor_;

    // Configuration
    StreamingConfig config_;

    // Processing thread
    std::unique_ptr<std::thread> processingThread_;
    std::atomic<bool> shouldStopProcessing_{false};
    std::atomic<bool> processingActive_{false};

    // Latest data (for broadcasting)
    std::vector<float> latestSpectrum_;
    std::vector<float> latestLeftChannel_;
    std::vector<float> latestRightChannel_;
    VUMeterData latestVUMeter_;
    HardwareStatus latestHardware_;
    ProcessingMetrics latestMetrics_;
    std::map<std::string, ProcessingStatus> latestAudioStatus_;
    std::map<std::string, float> latestAudioProgress_;

    mutable std::mutex dataMutex_;
    std::chrono::steady_clock::time_point lastDataUpdate_;

    // Statistics
    mutable std::mutex statsMutex_;
    StreamingStats stats_;

    // Quality management
    float targetQuality_ = 1.0f;
    bool adaptiveQualityEnabled_ = true;
    std::map<std::string, QualityMetrics> clientQuality_;

    // Error handling
    mutable std::string lastError_;
    void setError(const std::string& error);

    // Utility methods
    std::string formatTimestamp(const std::chrono::steady_clock::time_point& timestamp) const;
    std::string formatBytes(uint64_t bytes) const;
    std::string formatDuration(std::chrono::milliseconds duration) const;
    uint64_t getCurrentTimestamp() const;
    std::string generateRandomId(size_t length = 16);

    // Connection lifecycle
    void setupConnection(ConnectionHdl hdl);
    void closeConnection(const std::string& clientId, const std::string& reason = "");
    void cleanupIdleConnections();

    // Message builders
    WebSocketServer::message_ptr createTextMessage(const std::string& text);
    WebSocketServer::message_ptr createBinaryMessage(const std::vector<uint8_t>& data);
};

/**
 * @brief Binary protocol constants and utilities
 */
class WebSocketProtocol {
public:
    enum MessageType : uint16_t {
        HEARTBEAT = 1,
        SUBSCRIBE = 2,
        UNSUBSCRIBE = 3,
        ERROR = 4,
        SPECTRUM = 10,
        WAVEFORM = 11,
        VU_METER = 12,
        HARDWARE = 13,
        METRICS = 14,
        STATUS = 15,
        AUDIO_DATA = 16,
        CONFIG = 17
    };

    enum class Flags : uint16_t {
        COMPRESSED = 0x01,
        ENCRYPTED = 0x02,
        FRAGMENTED = 0x04,
        HIGH_PRIORITY = 0x08,
        ACK_REQUIRED = 0x10
    };

    static constexpr char PROTOCOL_VERSION = 1;
    static constexpr uint32_t MAX_MESSAGE_SIZE = 1024 * 1024;
    static constexpr uint32_t HEADER_SIZE = 24;

    static std::vector<uint8_t> createHeader(uint16_t messageType, uint32_t payloadSize,
                                              uint64_t timestamp, uint32_t sequenceNumber,
                                              uint16_t flags = 0);
    static bool validateHeader(const std::vector<uint8_t>& data);
    static std::vector<uint8_t> serializeMessage(const WebSocketProtocol::BinaryHeader& header,
                                                  const std::vector<uint8_t>& payload);
};

} // namespace vortex