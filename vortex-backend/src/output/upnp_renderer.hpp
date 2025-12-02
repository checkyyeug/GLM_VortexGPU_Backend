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
 * @brief UPnP/DLNA renderer implementation for VortexGPU audio backend
 *
 * Implements the UPnP AV and DLNA protocols to enable VortexGPU to function
 * as a network audio renderer, supporting:
 * - UPnP AV Media Renderer specification v1.0
 * - DLNA guidelines and interoperability
 * - Multi-format audio streaming (MP3, FLAC, WAV, AAC)
 * - Real-time streaming with buffer management
 * - Gapless playback support
 * - Device discovery via SSDP
 * - Content Directory Service integration
 * - Connection Manager service
 * - Rendering Control service
 *
 * Features:
 * - Automatic device discovery and announcement
 * - Multi-format audio codec support
 * - Real-time streaming with <100ms latency
 * - Adaptive bitrate and quality scaling
 * - Playlist management and queue control
 * - Transport state synchronization
 * - Volume and mute control
 * - Multi-zone support
 */
class UPnPRenderer {
public:
    enum class TransportState {
        STOPPED,
        PLAYING,
        PAUSED,
        TRANSITIONING,
        NO_MEDIA_PRESENT
    };

    enum class TransportStatus {
        OK,
        ERROR_OCCURRED,
        BUFFERING,
        CONNECTION_LOST
    };

    enum class PlaybackMode {
        NORMAL,
        REPEAT_ONE,
        REPEAT_ALL,
        SHUFFLE,
        SHUFFLE_REPEAT
    };

    enum class AudioCodec {
        UNKNOWN,
        PCM,
        MP3,
        FLAC,
        WAV,
        AAC,
        OGG,
        ALAC,
        WMA,
        DSD
    };

    struct MediaItem {
        std::string itemId;
        std::string title;
        std::string artist;
        std::string album;
        std::string genre;
        std::string duration;
        std::string uri;
        std::string mimeType;
        AudioCodec codec = AudioCodec::UNKNOWN;
        uint32_t sampleRate = 0;
        uint16_t channels = 0;
        uint16_t bitDepth = 0;
        uint32_t bitrate = 0;
        std::map<std::string, std::string> metadata;
        std::vector<std::string> albumArtURIs;
    };

    struct PlaybackInfo {
        TransportState state = TransportState::STOPPED;
        TransportStatus status = TransportStatus::OK;
        std::string currentURI;
        std::string nextURI;
        std::string currentURIMetaData;
        std::string nextURIMetaData;
        std::string playbackStorageMedium = "NONE";
        std::string recordStorageMedium = "NONE";
        std::string possiblePlaybackStorageMedia = "NETWORK, HDD";
        std::string possibleRecordStorageMedia = "NONE";
        PlaybackMode mode = PlaybackMode::NORMAL;
        float speed = 1.0f;
        uint64_t position = 0;
        uint64_t duration = 0;
        std::string absTime = "00:00:00";
        std::string relTime = "00:00:00";
        uint32_t absCount = 0;
        uint32_t relCount = 0;
    };

    struct UPnPConfig {
        std::string deviceId;
        std::string friendlyName = "VortexGPU Audio Renderer";
        std::string manufacturer = "VortexGPU";
        std::string manufacturerURL = "https://vortexgpu.com";
        std::string modelName = "Audio Backend";
        std::string modelNumber = "VG-1000";
        std::string modelDescription = "Professional Audio Processing Backend";
        std::string modelURL = "https://vortexgpu.com";
        std::string serialNumber = "VG-123456";
        std::string UDN = "";  // Will be generated if empty
        std::pair<std::string, std::string> ipAddress = {"0.0.0.0", ""};
        uint16_t port = 49152;
        std::string presentationURL;
        std::string iconURL;

        // Audio capabilities
        bool supportMP3 = true;
        bool supportFLAC = true;
        bool supportWAV = true;
        bool supportAAC = true;
        bool supportOGG = true;
        bool supportALAC = false;
        bool supportWMA = false;
        bool supportDSD = false;
        uint32_t maxSampleRate = 192000;
        uint16_t maxChannels = 2;
        uint16_t maxBitDepth = 24;

        // Streaming settings
        uint32_t bufferSize = 32768;      // bytes
        uint32_t bufferCount = 4;
        uint32_t targetLatency = 100;     // ms
        uint32_t networkTimeout = 5000;   // ms
        bool enableAdaptiveBitrate = true;
        bool enableGapless = true;

        // Service settings
        bool enableConnectionManager = true;
        bool enableRenderingControl = true;
        bool enableAVTransport = true;
        bool enableContentDirectory = false;

        // Security settings
        bool requireAuthentication = false;
        std::string authToken;
        std::vector<std::string> allowedIPs;

        // Logging and monitoring
        bool enableLogging = true;
        bool enableMetrics = true;
        uint32_t logLevel = 2;  // 0=error, 1=warn, 2=info, 3=debug
    };

    UPnPRenderer(const UPnPConfig& config);
    ~UPnPRenderer();

    // Server lifecycle
    bool start();
    void stop();
    bool isRunning() const;
    bool isReady() const;

    // Playback control
    bool setAVTransportURI(const std::string& instanceID, const std::string& currentURI,
                           const std::string& currentURIMetaData = "");
    bool setNextAVTransportURI(const std::string& instanceID, const std::string& nextURI,
                              const std::string& nextURIMetaData = "");
    bool play(const std::string& instanceID, const std::string& speed = "1");
    bool pause(const std::string& instanceID);
    bool stop(const std::string& instanceID);
    bool seek(const std::string& instanceID, const std::string& unit, const std::string& target);
    bool next(const std::string& instanceID);
    bool previous(const std::string& instanceID);

    // Rendering control
    bool setVolume(const std::string& instanceID, const std::string& channel, const std::string& desiredVolume);
    bool setMute(const std::string& instanceID, const std::string& channel, const std::string& desiredMute);
    bool getVolume(const std::string& instanceID, const std::string& channel, std::string& currentVolume);
    bool getMute(const std::string& instanceID, const std::string& channel, std::string& currentMute);

    // Connection management
    bool getProtocolInfo(std::string& protocolInfo);
    bool prepareForConnection(const std::string& connectionID, const std::string& peerConnectionID,
                             const std::string& direction, const std::string& protocolInfo);
    bool connectionComplete(const std::string& connectionID);

    // Content directory (optional)
    bool browse(const std::string& objectID, const std::string& browseFlag, const std::string& filter,
                const std::string& startingIndex, const std::string& requestedCount,
                const std::string& sortCriteria, std::string& result, std::string& numberReturned,
                std::string& totalMatches, std::string& updateID);

    // State management
    PlaybackInfo getPlaybackInfo() const;
    TransportState getTransportState() const;
    uint64_t getCurrentPosition() const;
    uint64_t getDuration() const;

    // Queue management
    bool addToQueue(const MediaItem& item);
    bool removeFromQueue(const std::string& itemId);
    bool clearQueue();
    bool setQueueMode(PlaybackMode mode);
    std::vector<MediaItem> getQueue() const;
    size_t getQueueSize() const;

    // Audio processing
    bool startStreaming(const std::string& uri);
    bool stopStreaming();
    bool pauseStreaming();
    bool resumeStreaming();
    bool seekStreaming(uint64_t position);

    // Statistics and monitoring
    struct RendererStatistics {
        bool isRunning = false;
        uint64_t totalConnections = 0;
        uint64_t activeConnections = 0;
        uint64_t totalBytesReceived = 0;
        uint64_t totalBytesPlayed = 0;
        uint64_t totalTracksPlayed = 0;
        uint64_t bufferUnderruns = 0;
        uint64_t bufferOverruns = 0;
        float averageLatency = 0.0f;
        float bufferUtilization = 0.0f;
        TransportState currentState = TransportState::STOPPED;
        uint32_t queueSize = 0;
        std::chrono::steady_clock::time_point startTime;
        std::chrono::steady_clock::time_point lastActivity;
        std::map<AudioCodec, uint64_t> playsByCodec;
        std::map<std::string, uint64_t> connectionsByClient;
    };

    RendererStatistics getStatistics() const;
    void resetStatistics();

    // Health monitoring
    struct HealthStatus {
        bool isHealthy = true;
        bool isRunning = false;
        bool hasActiveConnections = false;
        bool streamingStable = true;
        bool bufferWithinLimits = true;
        std::vector<std::string> warnings;
        std::vector<std::string> errors;
        float cpuUsage = 0.0f;
        uint64_t memoryUsage = 0;
        uint32_t networkLatency = 0;
    };

    HealthStatus getHealthStatus() const;

    // Configuration
    void updateConfiguration(const UPnPConfig& config);
    UPnPConfig getConfiguration() const;

    // Event callbacks
    using TransportStateCallback = std::function<void(TransportState)>;
    using MediaItemCallback = std::function<void(const MediaItem&)>;
    using ConnectionCallback = std::function<void(const std::string&, bool)>;
    using ErrorCallback = std::function<void(const std::string&, const std::string&)>;

    void setTransportStateChangedCallback(TransportStateCallback callback);
    void setMediaItemStartedCallback(MediaItemCallback callback);
    void setMediaItemEndedCallback(MediaItemCallback callback);
    void setConnectionCallback(ConnectionCallback callback);
    void setErrorCallback(ErrorCallback callback);

private:
    // Network services
    class SSDPServer;          // Simple Service Discovery Protocol
    class HTTPServer;          // HTTP server for SOAP requests
    class MediaReceiver;       // Media streaming receiver
    class ContentDirectory;    // Optional content directory service

    std::unique_ptr<SSDPServer> ssdpServer_;
    std::unique_ptr<HTTPServer> httpServer_;
    std::unique_ptr<MediaReceiver> mediaReceiver_;
    std::unique_ptr<ContentDirectory> contentDirectory_;

    // Server management
    bool startSSDPServer();
    bool startHTTPServer();
    bool startMediaReceiver();
    void stopSSDPServer();
    void stopHTTPServer();
    void stopMediaReceiver();

    // Device description
    std::string generateDeviceDescription() const;
    std::string generateServiceDescription(const std::string& serviceType) const;
    std::string generateIconData() const;

    // SOAP message handling
    std::string handleSOAPRequest(const std::string& method, const std::string& serviceType,
                                  const std::map<std::string, std::string>& arguments);
    std::string handleAVTransportAction(const std::string& action, const std::map<std::string, std::string>& arguments);
    std::string handleRenderingControlAction(const std::string& action, const std::map<std::string, std::string>& arguments);
    std::string handleConnectionManagerAction(const std::string& action, const std::map<std::string, std::string>& arguments);

    // Media processing
    void mediaProcessingThread();
    void processMediaQueue();
    bool loadMediaItem(const MediaItem& item);
    bool streamMediaData(const std::string& uri);
    void updatePlaybackPosition();

    // Format support
    bool isFormatSupported(const std::string& mimeType) const;
    AudioCodec detectCodec(const std::string& mimeType) const;
    std::string getProtocolInfo() const;

    // Queue management
    void advanceQueue();
    MediaItem getCurrentItem() const;
    MediaItem getNextItem() const;

    // State management
    void setTransportState(TransportState state);
    void setPlaybackPosition(uint64_t position);
    void updateDuration(uint64_t duration);

    // Eventing
    void sendEventNotification(const std::string& serviceType, const std::map<std::string, std::string>& variables);
    void subscribeToEvents(const std::string& serviceType, const std::string& callbackURL);
    void unsubscribeFromEvents(const std::string& serviceType, const std::string& sid);

    // Utility methods
    std::string generateUUID() const;
    std::string formatTime(uint64_t seconds) const;
    std::string formatDuration(uint64_t milliseconds) const;
    uint64_t parseTime(const std::string& timeStr) const;
    std::string escapeXML(const std::string& input) const;
    std::string unescapeXML(const std::string& input) const;

    // Configuration validation
    bool validateConfiguration(const UPnPConfig& config) const;

    // Error handling
    void setError(const std::string& error);
    void handleStreamingError(const std::string& error);

    // State
    UPnPConfig config_;
    std::atomic<bool> running_{false};
    std::atomic<bool> ready_{false};

    // Playback state
    PlaybackInfo playbackInfo_;
    std::vector<MediaItem> playbackQueue_;
    std::atomic<size_t> currentQueueIndex_{0};
    PlaybackMode queueMode_ = PlaybackMode::NORMAL;
    std::atomic<uint64_t> playbackPosition_{0};
    std::chrono::steady_clock::time_position playbackStartTime_;
    std::chrono::steady_clock::time_position lastPositionUpdate_;

    // Audio processing
    std::unique_ptr<std::thread> processingThread_;
    std::queue<std::string> processingQueue_;
    std::mutex processingQueueMutex_;
    std::condition_variable processingCondition_;
    std::atomic<bool> processingActive_{false};

    // Streaming
    std::atomic<bool> streamingActive_{false};
    std::string currentMediaURI_;
    std::vector<uint8_t> audioBuffer_;
    size_t bufferCapacity_ = 0;
    size_t bufferUsed_ = 0;
    std::mutex bufferMutex_;

    // Event subscriptions
    struct EventSubscription {
        std::string serviceType;
        std::string sid;
        std::string callbackURL;
        std::chrono::steady_clock::time_point expiry;
        uint32_t timeout = 1800; // seconds
        uint64_t sequenceNumber = 0;
    };

    std::map<std::string, EventSubscription> eventSubscriptions_;
    std::mutex eventSubscriptionsMutex_;

    // Statistics
    mutable std::mutex statsMutex_;
    RendererStatistics statistics_;

    // Callbacks
    TransportStateCallback transportStateChangedCallback_;
    MediaItemCallback mediaItemStartedCallback_;
    MediaItemCallback mediaItemEndedCallback_;
    ConnectionCallback connectionCallback_;
    ErrorCallback errorCallback_;

    // Error handling
    mutable std::string lastError_;
};

/**
 * @brief SSDP server for device discovery and announcement
 */
class UPnPRenderer::SSDPServer {
public:
    SSDPServer(const UPnPConfig& config, UPnPRenderer* parent);
    ~SSDPServer();

    bool start();
    void stop();
    bool isRunning() const;

    void announceDevice();
    void withdrawDevice();

private:
    void serverThread();
    void handleSearchRequest(const std::string& target, const std::string& mx);
    void sendDiscoveryResponse(const std::string& target, const std::string& location);

    UPnPConfig config_;
    UPnPRenderer* parent_;
    std::unique_ptr<std::thread> serverThread_;
    std::atomic<bool> running_{false};
    int socket_ = -1;
    std::string deviceUUID_;
};

/**
 * @brief HTTP server for SOAP and media requests
 */
class UPnPRenderer::HTTPServer {
public:
    HTTPServer(const UPnPConfig& config, UPnPRenderer* parent);
    ~HTTPServer();

    bool start();
    void stop();
    bool isRunning() const;

    std::string getDeviceURL() const;
    std::string getIconURL() const;

private:
    void serverThread();
    void handleHTTPRequest(const std::string& method, const std::string& path,
                           const std::map<std::string, std::string>& headers,
                           const std::string& body, std::string& response);
    std::string handleSOAPRequest(const std::string& soapAction, const std::string& soapBody);
    std::string handleMediaRequest(const std::string& path);
    std::string handleSubscriptionRequest(const std::string& method, const std::map<std::string, std::string>& headers,
                                         const std::string& body);

    UPnPConfig config_;
    UPnPRenderer* parent_;
    std::unique_ptr<std::thread> serverThread_;
    std::atomic<bool> running_{false};
    int serverSocket_ = -1;
    std::string baseURL_;
};

/**
 * @brief Media receiver for streaming audio content
 */
class UPnPRenderer::MediaReceiver {
public:
    MediaReceiver(const UPnPConfig& config, UPnPRenderer* parent);
    ~MediaReceiver();

    bool start();
    void stop();
    bool isRunning() const;

    bool receiveMedia(const std::string& uri);
    bool startPlayback();
    bool pausePlayback();
    bool stopPlayback();
    bool seekPlayback(uint64_t position);

    std::vector<uint8_t> getAudioBuffer(size_t bytes);
    float getBufferUtilization() const;

private:
    void streamingThread();
    void processMediaStream();
    bool openMediaStream(const std::string& uri);
    void closeMediaStream();

    UPnPConfig config_;
    UPnPRenderer* parent_;
    std::unique_ptr<std::thread> streamingThread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> active_{false};

    std::vector<uint8_t> audioBuffer_;
    size_t bufferCapacity_ = 0;
    size_t bufferUsed_ = 0;
    std::mutex bufferMutex_;
    std::condition_variable bufferCondition_;

    std::string currentURI_;
    FILE* mediaFile_ = nullptr;
    size_t filePosition_ = 0;
    size_t fileSize_ = 0;
    AudioCodec currentCodec_ = AudioCodec::UNKNOWN;

    uint64_t bufferUnderruns_ = 0;
    uint64_t bufferOverruns_ = 0;
};

/**
 * @brief Optional Content Directory service for media browsing
 */
class UPnPRenderer::ContentDirectory {
public:
    ContentDirectory(const UPnPConfig& config, UPnPRenderer* parent);
    ~ContentDirectory() = default;

    bool browse(const std::string& objectID, const std::string& browseFlag,
                const std::string& filter, const std::string& startingIndex,
                const std::string& requestedCount, const std::string& sortCriteria,
                std::string& result, std::string& numberReturned,
                std::string& totalMatches, std::string& updateID);

    bool search(const std::string& containerID, const std::string& searchCriteria,
                const std::string& filter, const std::string& startingIndex,
                const std::string& requestedCount, const std::string& sortCriteria,
                std::string& result, std::string& numberReturned,
                std::string& totalMatches, std::string& updateID);

    void addMediaItem(const MediaItem& item);
    void removeMediaItem(const std::string& itemId);
    void clearMediaLibrary();

private:
    std::string generateDIDLDocument(const std::vector<MediaItem>& items,
                                    const std::string& containerID,
                                    uint32_t numberReturned, uint32_t totalMatches,
                                    uint32_t updateID) const;

    UPnPConfig config_;
    UPnPRenderer* parent_;
    std::map<std::string, MediaItem> mediaLibrary_;
    std::mutex libraryMutex_;
    uint32_t systemUpdateID_ = 1;
};

} // namespace vortex