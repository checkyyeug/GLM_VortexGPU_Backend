#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <chrono>
#include <variant>

namespace vortex {

using SystemClock = std::chrono::system_clock;

// Network protocol enumeration
enum class NetworkProtocol {
    HTTP,
    WEBSOCKET,
    TCP,
    UDP,
    MDNS,
    UPNP_SSDP,
    Roon,
    HQPLAYER_NAA
};

// Message types for WebSocket communication
enum class WebSocketMessageType {
    SUBSCRIBE,
    UNSUBSCRIBE,
    DATA,
    ERROR,
    STATUS,
    HEARTBEAT,
    AUDIO_DATA,
    SPECTRUM_DATA,
    WAVEFORM_DATA,
    VU_METER_DATA,
    HARDWARE_DATA,
    ALERT_DATA,
    CONTROL_COMMAND,
    RESPONSE
};

// HTTP status codes
enum class HttpStatus {
    OK = 200,
    CREATED = 201,
    BAD_REQUEST = 400,
    UNAUTHORIZED = 401,
    FORBIDDEN = 403,
    NOT_FOUND = 404,
    METHOD_NOT_ALLOWED = 405,
    INTERNAL_ERROR = 500,
    SERVICE_UNAVAILABLE = 503
};

// WebSocket subscription types
enum class SubscriptionType {
    SPECTRUM,
    WAVEFORM,
    VU_METERS,
    HARDWARE,
    ALERTS,
    METRICS,
    AUDIO_STATUS,
    PROCESSING_STATUS
};

// Network connection status
enum class ConnectionStatus {
    DISCONNECTED,
    CONNECTING,
    CONNECTED,
    RECONNECTING,
    ERROR,
    CLOSED
};

// Authentication methods
enum class AuthMethod {
    NONE,
    TOKEN,
    BASIC,
    OAUTH2,
    CERTIFICATE
};

// API endpoint structure
struct APIEndpoint {
    std::string path;
    std::string method;
    std::string description;
    std::vector<std::string> parameters;
    std::string requestBody;
    std::string responseBody;
    std::vector<std::string> responses;
};

// WebSocket subscription request
struct WebSocketSubscription {
    std::string clientId;
    std::vector<SubscriptionType> dataTypes;
    uint32_t updateRate = 60;
    std::map<std::string, std::string> filters;
    SystemClock::time_point timestamp;
};

// WebSocket message structure
struct WebSocketMessage {
    WebSocketMessageType type;
    std::string sessionId;
    std::string clientId;
    std::variant<
        std::string,
        std::map<std::string, std::string>,
        std::vector<uint8_t>
    > payload;
    uint64_t timestamp;
    uint32_t sequenceNumber = 0;
    bool isCompressed = false;
    bool isEncrypted = false;
};

// HTTP request structure
struct HTTPRequest {
    std::string method;
    std::string path;
    std::map<std::string, std::string> headers;
    std::string body;
    std::map<std::string, std::string> queryParameters;
    std::string clientIP;
    std::string userAgent;
    SystemClock::time_point timestamp;
};

// HTTP response structure
struct HTTPResponse {
    HttpStatus status;
    std::map<std::string, std::string> headers;
    std::string body;
    std::string contentType;
    uint64_t contentLength = 0;
    bool isCompressed = false;
    SystemClock::time_point timestamp;
};

// Network client information
struct NetworkClient {
    std::string id;
    std::string ipAddress;
    uint16_t port;
    std::string userAgent;
    SystemClock::time_point connectedAt;
    SystemClock::time_point lastActivity;
    ConnectionStatus status;
    uint64_t bytesReceived = 0;
    uint64_t bytesSent = 0;
    std::map<std::string, std::string> metadata;
};

// Device discovery information
struct DeviceDiscoveryInfo {
    std::string deviceId;
    std::string deviceName;
    std::string deviceType;
    std::string ipAddress;
    uint16_t port;
    std::map<std::string, std::string> capabilities;
    SystemClock::time_point discoveredAt;
    SystemClock::time_point lastSeen;
    bool isOnline = false;
};

// Network statistics
struct NetworkStatistics {
    uint64_t totalConnections = 0;
    uint64_t activeConnections = 0;
    uint64_t bytesTransmitted = 0;
    uint64_t bytesReceived = 0;
    float averageLatency = 0.0f;
    float packetLossRate = 0.0f;
    uint32_t errors = 0;
    SystemClock::time_point startTime;
    SystemClock::time_point lastReset;
};

// Network configuration
struct NetworkConfig {
    std::string bindAddress = "0.0.0.0";
    uint16_t httpPort = 8080;
    uint16_t websocketPort = 8081;
    uint16_t discoveryPort = 8082;
    bool enableSSL = false;
    std::string sslCertificatePath;
    std::string sslPrivateKeyPath;
    uint32_t maxConnections = 100;
    uint32_t connectionTimeout = 30;
    bool enableCompression = true;
    AuthMethod authMethod = AuthMethod::NONE;
    std::map<std::string, std::string> authConfig;
};

// WebSocket client configuration
struct WebSocketClientConfig {
    std::string url;
    std::map<std::string, std::string> headers;
    bool autoReconnect = true;
    uint32_t reconnectInterval = 5;
    uint32_t maxReconnectAttempts = 10;
    bool enableCompression = true;
    bool enablePing = true;
    uint32_t pingInterval = 30;
};

// Protocol buffer message types
enum class ProtocolBufferType {
    AUDIO_METADATA,
    PROCESSING_STATUS,
    DEVICE_INFO,
    SYSTEM_METRICS,
    ALERT_MESSAGE,
    CONFIG_UPDATE
};

// Binary protocol header
struct BinaryProtocolHeader {
    uint32_t magic = 0x56545858; // "VTVX"
    uint16_t version = 1;
    uint16_t messageType = 0;
    uint32_t payloadSize = 0;
    uint64_t timestamp = 0;
    uint32_t sequenceNumber = 0;
    uint16_t flags = 0;
    uint16_t reserved = 0;
};

// Network event structure
struct NetworkEvent {
    enum class Type {
        CLIENT_CONNECTED,
        CLIENT_DISCONNECTED,
        MESSAGE_RECEIVED,
        ERROR_OCCURRED,
        CONFIG_UPDATED,
        DEVICE_DISCOVERED,
        DEVICE_LOST
    };

    Type type;
    std::string clientId;
    std::string eventData;
    SystemClock::time_point timestamp;
    std::map<std::string, std::string> metadata;
};

// Protocol handler interface
class ProtocolHandler {
public:
    virtual ~ProtocolHandler() = default;
    virtual bool initialize(const NetworkConfig& config) = 0;
    virtual void shutdown() = 0;
    virtual bool handleRequest(const HTTPRequest& request, HTTPResponse& response) = 0;
    virtual bool handleMessage(const WebSocketMessage& message, WebSocketMessage& response) = 0;
    virtual bool sendMessage(const WebSocketMessage& message) = 0;
    virtual NetworkStatistics getStatistics() const = 0;
};

// Discovery service interface
class DiscoveryService {
public:
    virtual ~DiscoveryService() = default;
    virtual bool start() = 0;
    virtual void stop() = 0;
    virtual std::vector<DeviceDiscoveryInfo> discoverDevices() = 0;
    virtual bool registerDevice(const DeviceDiscoveryInfo& device) = 0;
    virtual bool unregisterDevice(const std::string& deviceId) = 0;
    virtual void setDeviceUpdateCallback(std::function<void(const DeviceDiscoveryInfo&)> callback) = 0;
};

// Network server interface
class NetworkServer {
public:
    virtual ~NetworkServer() = default;
    virtual bool start(const NetworkConfig& config) = 0;
    virtual void stop() = 0;
    virtual bool isRunning() const = 0;
    virtual std::vector<NetworkClient> getConnectedClients() const = 0;
    virtual NetworkStatistics getStatistics() const = 0;
    virtual void setEventCallback(std::function<void(const NetworkEvent&)> callback) = 0;
};

// Utility functions
const char* protocolToString(NetworkProtocol protocol);
const char* messageTypeToString(WebSocketMessageType type);
const char* subscriptionTypeToString(SubscriptionType type);
const char* connectionStatusToString(ConnectionStatus status);
const char* authMethodToString(AuthMethod method);
const char* eventToString(const NetworkEvent::Type& type);

bool isValidIPAddress(const std::string& ip);
bool isValidPort(uint16_t port);
bool isValidURL(const std::string& url);

std::string generateClientId();
std::string generateSessionId();
std::string getCurrentTimestamp();
std::string formatBytes(uint64_t bytes);

// Message serialization functions
std::vector<uint8_t> serializeWebSocketMessage(const WebSocketMessage& message);
WebSocketMessage deserializeWebSocketMessage(const std::vector<uint8_t>& data);

std::string serializeHttpRequest(const HTTPRequest& request);
HTTPRequest deserializeHttpRequest(const std::string& data);

std::string serializeHttpResponse(const HTTPResponse& response);
HTTPResponse deserializeHttpResponse(const std::string& data);

// Binary protocol functions
std::vector<uint8_t> serializeBinaryHeader(const BinaryProtocolHeader& header);
BinaryProtocolHeader deserializeBinaryHeader(const std::vector<uint8_t>& data);

bool validateBinaryHeader(const BinaryProtocolHeader& header);
std::vector<uint8_t> createBinaryMessage(uint16_t messageType, const std::vector<uint8_t>& payload);

} // namespace vortex