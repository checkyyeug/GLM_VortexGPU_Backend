#include "upnp_renderer.hpp"
#include "../utils/logger.hpp"

#include <random>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <ctime>
#include <regex>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "iphlpapi.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <net/if.h>
#include <ifaddrs.h>
#endif

namespace vortex {

// UPnPRenderer Implementation
UPnPRenderer::UPnPRenderer(const UPnPConfig& config) : config_(config) {
    Logger::info("UPnPRenderer: Initializing UPnP/DLNA renderer '{}'", config.friendlyName);

    // Validate configuration
    if (!validateConfiguration(config)) {
        throw std::invalid_argument("Invalid UPnP renderer configuration");
    }

    // Generate UDN if not provided
    if (config_.UDN.empty()) {
        config_.UDN = "uuid:" + generateUUID();
    }

    // Initialize playback state
    playbackInfo_.state = TransportState::STOPPED;
    playbackInfo_.status = TransportStatus::OK;
    playbackInfo_.playbackStorageMedium = "NETWORK";
    playbackInfo_.possiblePlaybackStorageMedia = "NETWORK, HDD";
    playbackInfo_.recordStorageMedium = "NONE";
    playbackInfo_.possibleRecordStorageMedia = "NONE";
    playbackInfo_.mode = PlaybackMode::NORMAL;
    playbackInfo_.speed = 1.0f;

    // Initialize buffer
    bufferCapacity_ = config_.bufferSize;
    audioBuffer_.reserve(bufferCapacity_);
    bufferUsed_ = 0;

    // Initialize statistics
    statistics_.startTime = std::chrono::steady_clock::now();
    statistics_.lastActivity = statistics_.startTime;
    statistics_.currentState = TransportState::STOPPED;

    // Create service components
    ssdpServer_ = std::make_unique<SSDPServer>(config_, this);
    httpServer_ = std::make_unique<HTTPServer>(config_, this);
    mediaReceiver_ = std::make_unique<MediaReceiver>(config_, this);

    if (config_.enableContentDirectory) {
        contentDirectory_ = std::make_unique<ContentDirectory>(config_, this);
    }

    Logger::info("UPnPRenderer: UPnP/DLNA renderer initialized with UDN: {}", config_.UDN);
}

UPnPRenderer::~UPnPRenderer() {
    stop();
    Logger::info("UPnPRenderer: UPnP/DLNA renderer destroyed");
}

bool UPnPRenderer::start() {
    if (running_.load()) {
        Logger::warn("UPnPRenderer: Already running");
        return true;
    }

    Logger::info("UPnPRenderer: Starting UPnP/DLNA renderer services");

    try {
        // Start SSDP server for device discovery
        if (!startSSDPServer()) {
            setError("Failed to start SSDP server");
            return false;
        }

        // Start HTTP server for SOAP and media requests
        if (!startHTTPServer()) {
            setError("Failed to start HTTP server");
            stopSSDPServer();
            return false;
        }

        // Start media receiver for streaming
        if (!startMediaReceiver()) {
            setError("Failed to start media receiver");
            stopHTTPServer();
            stopSSDPServer();
            return false;
        }

        // Start media processing thread
        processingActive_.store(true);
        processingThread_ = std::make_unique<std::thread>(&UPnPRenderer::mediaProcessingThread, this);

        // Announce device presence
        ssdpServer_->announceDevice();

        running_.store(true);
        ready_.store(true);

        Logger::info("UPnPRenderer: UPnP/DLNA renderer started successfully");
        return true;

    } catch (const std::exception& e) {
        setError("Failed to start renderer: " + std::string(e.what()));
        stop();
        return false;
    }
}

void UPnPRenderer::stop() {
    if (!running_.load()) {
        return;
    }

    Logger::info("UPnPRenderer: Stopping UPnP/DLNA renderer services");

    running_.store(false);
    ready_.store(false);
    processingActive_.store(false);

    // Stop streaming
    stopStreaming();

    // Wake up processing thread
    processingCondition_.notify_all();

    // Wait for threads to finish
    if (processingThread_ && processingThread_->joinable()) {
        processingThread_->join();
    }

    // Withdraw device presence
    ssdpServer_->withdrawDevice();

    // Stop services
    stopMediaReceiver();
    stopHTTPServer();
    stopSSDPServer();

    // Clear queues and buffers
    clearQueue();
    {
        std::lock_guard<std::mutex> lock(bufferMutex_);
        audioBuffer_.clear();
        bufferUsed_ = 0;
    }

    Logger::info("UPnPRenderer: UPnP/DLNA renderer stopped");
}

bool UPnPRenderer::isRunning() const {
    return running_.load();
}

bool UPnPRenderer::isReady() const {
    return ready_.load();
}

bool UPnPRenderer::setAVTransportURI(const std::string& instanceID, const std::string& currentURI,
                                     const std::string& currentURIMetaData) {
    Logger::info("UPnPRenderer: Setting AV Transport URI: {}", currentURI);

    // Validate URI
    if (currentURI.empty()) {
        setError("Empty URI provided");
        return false;
    }

    // Check if format is supported
    if (!isFormatSupported(currentURI)) {
        setError("Unsupported format: " + currentURI);
        return false;
    }

    // Clear current queue and add new media item
    clearQueue();

    MediaItem item;
    item.itemId = generateUUID();
    item.uri = currentURI;
    item.title = extractTitleFromMetadata(currentURIMetaData);
    item.artist = extractArtistFromMetadata(currentURIMetaData);
    item.album = extractAlbumFromMetadata(currentURIMetaData);
    item.codec = detectCodecFromURI(currentURI);

    if (!addToQueue(item)) {
        setError("Failed to add media item to queue");
        return false;
    }

    playbackInfo_.currentURI = currentURI;
    playbackInfo_.currentURIMetaData = currentURIMetaData;

    Logger::info("UPnPRenderer: AV Transport URI set successfully");
    return true;
}

bool UPnPRenderer::play(const std::string& instanceID, const std::string& speed) {
    Logger::info("UPnPRenderer: Playing at speed: {}", speed);

    if (playbackQueue_.empty()) {
        setError("No media in queue to play");
        return false;
    }

    try {
        float playbackSpeed = std::stof(speed);
        if (playbackSpeed <= 0.0f || playbackSpeed > 10.0f) {
            setError("Invalid playback speed: " + speed);
            return false;
        }
        playbackInfo_.speed = playbackSpeed;
    } catch (const std::exception&) {
        setError("Invalid speed format: " + speed);
        return false;
    }

    // Start streaming the current item
    MediaItem currentItem = getCurrentItem();
    if (!startStreaming(currentItem.uri)) {
        setError("Failed to start streaming");
        return false;
    }

    setTransportState(TransportState::PLAYING);
    playbackStartTime_ = std::chrono::steady_clock::now();
    setPlaybackPosition(0);

    // Notify callbacks
    if (mediaItemStartedCallback_) {
        mediaItemStartedCallback_(currentItem);
    }

    Logger::info("UPnPRenderer: Playback started");
    return true;
}

bool UPnPRenderer::pause(const std::string& instanceID) {
    Logger::info("UPnPRenderer: Pausing playback");

    if (playbackInfo_.state != TransportState::PLAYING) {
        Logger::warn("UPnPRenderer: Cannot pause - not currently playing");
        return false;
    }

    pauseStreaming();
    setTransportState(TransportState::PAUSED);

    Logger::info("UPnPRenderer: Playback paused");
    return true;
}

bool UPnPRenderer::stop(const std::string& instanceID) {
    Logger::info("UPnPRenderer: Stopping playback");

    stopStreaming();
    setTransportState(TransportState::STOPPED);
    setPlaybackPosition(0);

    Logger::info("UPnPRenderer: Playback stopped");
    return true;
}

bool UPnPRenderer::seek(const std::string& instanceID, const std::string& unit, const std::string& target) {
    Logger::info("UPnPRenderer: Seeking to {} {}", target, unit);

    if (playbackInfo_.state != TransportState::PLAYING && playbackInfo_.state != TransportState::PAUSED) {
        setError("Cannot seek - not currently playing or paused");
        return false;
    }

    uint64_t positionMs = 0;
    if (unit == "REL_TIME" || unit == "ABS_TIME") {
        positionMs = parseTime(target) * 1000;
    } else if (unit == "REL_COUNT" || unit == "ABS_COUNT") {
        try {
            positionMs = std::stoull(target);
        } catch (const std::exception&) {
            setError("Invalid seek position: " + target);
            return false;
        }
    } else {
        setError("Unsupported seek unit: " + unit);
        return false;
    }

    if (!seekStreaming(positionMs)) {
        setError("Failed to seek to position");
        return false;
    }

    setPlaybackPosition(positionMs);

    Logger::info("UPnPRenderer: Seek completed to position {}", positionMs);
    return true;
}

bool UPnPRenderer::setVolume(const std::string& instanceID, const std::string& channel, const std::string& desiredVolume) {
    Logger::info("UPnPRenderer: Setting volume for channel {} to {}", channel, desiredVolume);

    try {
        uint32_t volume = std::stoul(desiredVolume);
        if (volume > 100) {
            setError("Invalid volume level: " + desiredVolume);
            return false;
        }

        // Apply volume control to audio processing
        float volumeFactor = static_cast<float>(volume) / 100.0f;
        // This would be applied in the audio processing pipeline

        Logger::info("UPnPRenderer: Volume set to {}% ({})", volume, volumeFactor);
        return true;

    } catch (const std::exception&) {
        setError("Invalid volume format: " + desiredVolume);
        return false;
    }
}

bool UPnPRenderer::setMute(const std::string& instanceID, const std::string& channel, const std::string& desiredMute) {
    Logger::info("UPnPRenderer: Setting mute for channel {} to {}", channel, desiredMute);

    bool muted = false;
    if (desiredMute == "1" || desiredMute == "true" || desiredMute == "TRUE") {
        muted = true;
    } else if (desiredMute == "0" || desiredMute == "false" || desiredMute == "FALSE") {
        muted = false;
    } else {
        setError("Invalid mute value: " + desiredMute);
        return false;
    }

    // Apply mute to audio processing
    // This would be applied in the audio processing pipeline

    Logger::info("UPnPRenderer: Mute set to {}", muted ? "on" : "off");
    return true;
}

UPnPRenderer::PlaybackInfo UPnPRenderer::getPlaybackInfo() const {
    PlaybackInfo info = playbackInfo_;

    // Update timing information
    if (info.state == TransportState::PLAYING) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - playbackStartTime_);
        uint64_t currentPosition = playbackPosition_.load() + elapsed.count() * static_cast<uint64_t>(info.speed);

        info.position = currentPosition;
        info.relTime = formatTime(currentPosition / 1000);
        info.absTime = formatTime(currentPosition / 1000);
    }

    return info;
}

UPnPRenderer::TransportState UPnPRenderer::getTransportState() const {
    return playbackInfo_.state;
}

uint64_t UPnPRenderer::getCurrentPosition() const {
    if (playbackInfo_.state == TransportState::PLAYING) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - playbackStartTime_);
        return playbackPosition_.load() + elapsed.count() * static_cast<uint64_t>(playbackInfo_.speed);
    }
    return playbackPosition_.load();
}

uint64_t UPnPRenderer::getDuration() const {
    return playbackInfo_.duration;
}

bool UPnPRenderer::addToQueue(const MediaItem& item) {
    Logger::info("UPnPRenderer: Adding item to queue: {}", item.title);

    playbackQueue_.push_back(item);

    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        statistics_.queueSize = playbackQueue_.size();
    }

    return true;
}

bool UPnPRenderer::clearQueue() {
    Logger::info("UPnPRenderer: Clearing playback queue");

    playbackQueue_.clear();
    currentQueueIndex_.store(0);

    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        statistics_.queueSize = 0;
    }

    return true;
}

bool UPnPRenderer::setQueueMode(PlaybackMode mode) {
    Logger::info("UPnPRenderer: Setting queue mode to {}", static_cast<int>(mode));

    queueMode_ = mode;
    playbackInfo_.mode = mode;
    return true;
}

std::vector<UPnPRenderer::MediaItem> UPnPRenderer::getQueue() const {
    return playbackQueue_;
}

size_t UPnPRenderer::getQueueSize() const {
    return playbackQueue_.size();
}

bool UPnPRenderer::startStreaming(const std::string& uri) {
    if (streamingActive_.load()) {
        stopStreaming();
    }

    Logger::info("UPnPRenderer: Starting streaming from URI: {}", uri);

    if (!mediaReceiver_->receiveMedia(uri)) {
        setError("Failed to start media reception");
        return false;
    }

    if (!mediaReceiver_->startPlayback()) {
        setError("Failed to start playback");
        return false;
    }

    currentMediaURI_ = uri;
    streamingActive_.store(true);

    Logger::info("UPnPRenderer: Streaming started");
    return true;
}

bool UPnPRenderer::stopStreaming() {
    if (!streamingActive_.load()) {
        return true;
    }

    Logger::info("UPnPRenderer: Stopping streaming");

    mediaReceiver_->stopPlayback();
    streamingActive_.store(false);
    currentMediaURI_.clear();

    // Clear audio buffer
    {
        std::lock_guard<std::mutex> lock(bufferMutex_);
        audioBuffer_.clear();
        bufferUsed_ = 0;
    }

    Logger::info("UPnPRenderer: Streaming stopped");
    return true;
}

UPnPRenderer::RendererStatistics UPnPRenderer::getStatistics() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    RendererStatistics stats = statistics_;
    stats.isRunning = running_.load();
    stats.currentState = playbackInfo_.state;
    stats.queueSize = playbackQueue_.size();
    return stats;
}

void UPnPRenderer::resetStatistics() {
    std::lock_guard<std::mutex> lock(statsMutex_);
    statistics_ = RendererStatistics{};
    statistics_.startTime = std::chrono::steady_clock::now();
    statistics_.lastActivity = statistics_.startTime;
    statistics_.currentState = TransportState::STOPPED;
}

UPnPRenderer::HealthStatus UPnPRenderer::getHealthStatus() const {
    HealthStatus status;
    status.isRunning = running_.load();
    status.isHealthy = true;

    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        status.hasActiveConnections = (statistics_.activeConnections > 0);
        status.streamingStable = (statistics_.bufferUnderruns < 10);
        status.bufferWithinLimits = (statistics_.bufferUtilization < 0.9f);
        status.networkLatency = static_cast<uint32_t>(statistics_.averageLatency);
    }

    if (!status.streamingStable) {
        status.warnings.push_back("High buffer underrun count detected");
    }

    if (!status.bufferWithinLimits) {
        status.warnings.push_back("Buffer utilization above 90%");
    }

    status.isHealthy = status.isRunning && status.streamingStable && status.bufferWithinLimits;

    return status;
}

void UPnPRenderer::setTransportStateChangedCallback(TransportStateCallback callback) {
    transportStateChangedCallback_ = callback;
}

void UPnPRenderer::setMediaItemStartedCallback(MediaItemCallback callback) {
    mediaItemStartedCallback_ = callback;
}

void UPnPRenderer::setMediaItemEndedCallback(MediaItemCallback callback) {
    mediaItemEndedCallback_ = callback;
}

void UPnPRenderer::setConnectionCallback(ConnectionCallback callback) {
    connectionCallback_ = callback;
}

void UPnPRenderer::setErrorCallback(ErrorCallback callback) {
    errorCallback_ = callback;
}

// Private methods implementation
bool UPnPRenderer::startSSDPServer() {
    return ssdpServer_ && ssdpServer_->start();
}

bool UPnPRenderer::startHTTPServer() {
    return httpServer_ && httpServer_->start();
}

bool UPnPRenderer::startMediaReceiver() {
    return mediaReceiver_ && mediaReceiver_->start();
}

void UPnPRenderer::stopSSDPServer() {
    if (ssdpServer_) {
        ssdpServer_->stop();
    }
}

void UPnPRenderer::stopHTTPServer() {
    if (httpServer_) {
        httpServer_->stop();
    }
}

void UPnPRenderer::stopMediaReceiver() {
    if (mediaReceiver_) {
        mediaReceiver_->stop();
    }
}

void UPnPRenderer::mediaProcessingThread() {
    Logger::info("UPnPRenderer: Media processing thread started");

    while (processingActive_.load()) {
        std::unique_lock<std::mutex> lock(processingQueueMutex_);
        processingCondition_.wait(lock, [this] {
            return !processingQueue_.empty() || !processingActive_.load();
        });

        if (!processingActive_.load()) {
            break;
        }

        while (!processingQueue_.empty()) {
            std::string uri = processingQueue_.front();
            processingQueue_.pop();
            lock.unlock();

            processMediaQueue();

            lock.lock();
        }
    }

    Logger::info("UPnPRenderer: Media processing thread stopped");
}

void UPnPRenderer::processMediaQueue() {
    if (streamingActive_.load()) {
        // Update playback position
        updatePlaybackPosition();

        // Get audio data from media receiver
        std::vector<uint8_t> audioData = mediaReceiver_->getAudioBuffer(4096);
        if (!audioData.empty()) {
            std::lock_guard<std::mutex> lock(bufferMutex_);
            if (bufferUsed_ + audioData.size() <= bufferCapacity_) {
                audioBuffer_.insert(audioBuffer_.end(), audioData.begin(), audioData.end());
                bufferUsed_ += audioData.size();
            }
        }
    }
}

void UPnPRenderer::updatePlaybackPosition() {
    if (playbackInfo_.state == TransportState::PLAYING) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - playbackStartTime_);
        uint64_t newPosition = playbackPosition_.load() + elapsed.count() * static_cast<uint64_t>(playbackInfo_.speed);
        setPlaybackPosition(newPosition);
    }
}

void UPnPRenderer::setTransportState(TransportState state) {
    if (playbackInfo_.state != state) {
        TransportState oldState = playbackInfo_.state;
        playbackInfo_.state = state;
        Logger::info("UPnPRenderer: Transport state changed from {} to {}",
                     static_cast<int>(oldState), static_cast<int>(state));

        if (transportStateChangedCallback_) {
            transportStateChangedCallback_(state);
        }
    }
}

void UPnPRenderer::setPlaybackPosition(uint64_t position) {
    playbackPosition_.store(position);
    lastPositionUpdate_ = std::chrono::steady_clock::now();
}

UPnPRenderer::MediaItem UPnPRenderer::getCurrentItem() const {
    size_t index = currentQueueIndex_.load();
    if (index < playbackQueue_.size()) {
        return playbackQueue_[index];
    }
    return MediaItem{};
}

UPnPRenderer::MediaItem UPnPRenderer::getNextItem() const {
    switch (queueMode_) {
        case PlaybackMode::REPEAT_ONE:
            return getCurrentItem();
        case PlaybackMode::REPEAT_ALL:
        case PlaybackMode::SHUFFLE_REPEAT:
        case PlaybackMode::NORMAL:
        case PlaybackMode::SHUFFLE: {
            size_t index = currentQueueIndex_.load();
            if (index + 1 < playbackQueue_.size()) {
                return playbackQueue_[index + 1];
            }
            return MediaItem{};
        }
    }
    return MediaItem{};
}

bool UPnPRenderer::isFormatSupported(const std::string& mimeType) const {
    if (mimeType.empty()) {
        // Try to detect from URI extension
        return true; // Assume supported for now
    }

    std::string lowerMime = mimeType;
    std::transform(lowerMime.begin(), lowerMime.end(), lowerMime.begin(), ::tolower);

    if (config_.supportMP3 && (lowerMime.find("audio/mpeg") != std::string::npos ||
                                lowerMime.find("audio/mp3") != std::string::npos)) {
        return true;
    }
    if (config_.supportFLAC && lowerMime.find("audio/flac") != std::string::npos) {
        return true;
    }
    if (config_.supportWAV && (lowerMime.find("audio/wav") != std::string::npos ||
                               lowerMime.find("audio/wave") != std::string::npos)) {
        return true;
    }
    if (config_.supportAAC && lowerMime.find("audio/aac") != std::string::npos) {
        return true;
    }
    if (config_.supportOGG && lowerMime.find("audio/ogg") != std::string::npos) {
        return true;
    }

    return false;
}

std::string UPnPRenderer::generateUUID() const {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);

    std::ostringstream oss;
    oss << std::hex;

    for (int i = 0; i < 32; ++i) {
        if (i == 8 || i == 12 || i == 16 || i == 20) {
            oss << "-";
        }
        oss << std::setw(1) << dis(gen);
    }

    return oss.str();
}

std::string UPnPRenderer::formatTime(uint64_t seconds) const {
    uint64_t hours = seconds / 3600;
    uint64_t minutes = (seconds % 3600) / 60;
    uint64_t secs = seconds % 60;

    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(2) << hours << ":"
        << std::setfill('0') << std::setw(2) << minutes << ":"
        << std::setfill('0') << std::setw(2) << secs;

    return oss.str();
}

uint64_t UPnPRenderer::parseTime(const std::string& timeStr) const {
    std::regex timePattern(R"((\d+):(\d+):(\d+))");
    std::smatch match;

    if (std::regex_match(timeStr, match, timePattern)) {
        uint64_t hours = std::stoull(match[1].str());
        uint64_t minutes = std::stoull(match[2].str());
        uint64_t seconds = std::stoull(match[3].str());
        return hours * 3600 + minutes * 60 + seconds;
    }

    return 0;
}

bool UPnPRenderer::validateConfiguration(const UPnPConfig& config) const {
    if (config.friendlyName.empty()) {
        return false;
    }

    if (config.port == 0 || config.port > 65535) {
        return false;
    }

    if (config.maxSampleRate == 0 || config.maxSampleRate > 384000) {
        return false;
    }

    return true;
}

void UPnPRenderer::setError(const std::string& error) {
    lastError_ = error;
    Logger::error("UPnPRenderer: {}", error);

    if (errorCallback_) {
        errorCallback_("renderer", error);
    }
}

// Helper methods
std::string UPnPRenderer::extractTitleFromMetadata(const std::string& metadata) const {
    // Simple DIDL metadata parsing
    std::regex titlePattern(R"(<dc:title>([^<]+)</dc:title>)");
    std::smatch match;

    if (std::regex_search(metadata, match, titlePattern)) {
        return match[1].str();
    }

    return "Unknown Title";
}

std::string UPnPRenderer::extractArtistFromMetadata(const std::string& metadata) const {
    std::regex artistPattern(R"(<dc:creator>([^<]+)</dc:creator>)");
    std::smatch match;

    if (std::regex_search(metadata, match, artistPattern)) {
        return match[1].str();
    }

    return "";
}

std::string UPnPRenderer::extractAlbumFromMetadata(const std::string& metadata) const {
    std::regex albumPattern(R"(<upnp:album>([^<]+)</upnp:album>)");
    std::smatch match;

    if (std::regex_search(metadata, match, albumPattern)) {
        return match[1].str();
    }

    return "";
}

UPnPRenderer::AudioCodec UPnPRenderer::detectCodecFromURI(const std::string& uri) const {
    std::string lowerURI = uri;
    std::transform(lowerURI.begin(), lowerURI.end(), lowerURI.begin(), ::tolower);

    if (lowerURI.find(".mp3") != std::string::npos) {
        return AudioCodec::MP3;
    } else if (lowerURI.find(".flac") != std::string::npos) {
        return AudioCodec::FLAC;
    } else if (lowerURI.find(".wav") != std::string::npos) {
        return AudioCodec::WAV;
    } else if (lowerURI.find(".aac") != std::string::npos || lowerURI.find(".m4a") != std::string::npos) {
        return AudioCodec::AAC;
    } else if (lowerURI.find(".ogg") != std::string::npos) {
        return AudioCodec::OGG;
    }

    return AudioCodec::UNKNOWN;
}

// SSDPServer Implementation
UPnPRenderer::SSDPServer::SSDPServer(const UPnPConfig& config, UPnPRenderer* parent)
    : config_(config), parent_(parent) {
    deviceUUID_ = config_.UDN.substr(5); // Remove "uuid:" prefix
}

UPnPRenderer::SSDPServer::~SSDPServer() {
    stop();
}

bool UPnPRenderer::SSDPServer::start() {
    Logger::info("UPnPRenderer::SSDP: Starting SSDP server");

    socket_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_ < 0) {
        return false;
    }

    int opt = 1;
    if (setsockopt(socket_, SOL_SOCKET, SO_REUSEADDR,
                   reinterpret_cast<const char*>(&opt), sizeof(opt)) < 0) {
        return false;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(1900); // SSDP port

    if (bind(socket_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        return false;
    }

    running_.store(true);
    serverThread_ = std::make_unique<std::thread>(&SSDPServer::serverThread, this);

    Logger::info("UPnPRenderer::SSDP: SSDP server started");
    return true;
}

void UPnPRenderer::SSDPServer::stop() {
    if (!running_.load()) {
        return;
    }

    running_.store(false);

    if (socket_ >= 0) {
#ifdef _WIN32
        closesocket(socket_);
#else
        close(socket_);
#endif
        socket_ = -1;
    }

    if (serverThread_ && serverThread_->joinable()) {
        serverThread_->join();
    }

    Logger::info("UPnPRenderer::SSDP: SSDP server stopped");
}

bool UPnPRenderer::SSDPServer::isRunning() const {
    return running_.load();
}

void UPnPRenderer::SSDPServer::serverThread() {
    Logger::info("UPnPRenderer::SSDP: SSDP server thread started");

    char buffer[2048];
    sockaddr_in clientAddr{};
    socklen_t clientAddrLen = sizeof(clientAddr);

    while (running_.load()) {
        fd_set readSet;
        FD_ZERO(&readSet);
        FD_SET(socket_, &readSet);

        struct timeval timeout{};
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;

        int result = select(socket_ + 1, &readSet, nullptr, nullptr, &timeout);
        if (result > 0 && FD_ISSET(socket_, &readSet)) {
            ssize_t received = recvfrom(socket_, buffer, sizeof(buffer) - 1, 0,
                                      reinterpret_cast<sockaddr*>(&clientAddr), &clientAddrLen);
            if (received > 0) {
                buffer[received] = '\0';
                std::string request(buffer);

                if (request.find("M-SEARCH") != std::string::npos) {
                    // Handle search request
                    handleSearchRequest(request, clientAddr);
                }
            }
        }
    }

    Logger::info("UPnPRenderer::SSDP: SSDP server thread stopped");
}

void UPnPRenderer::SSDPServer::announceDevice() {
    if (!running_.load()) {
        return;
    }

    Logger::info("UPnPRenderer::SSDP: Announcing device presence");

    // Send NOTIFY messages
    std::string notifyMsg =
        "NOTIFY * HTTP/1.1\r\n"
        "HOST: 239.255.255.250:1900\r\n"
        "CACHE-CONTROL: max-age=1800\r\n"
        "LOCATION: http://" + config_.ipAddress.first + ":" + std::to_string(config_.port) + "/description.xml\r\n"
        "NT: upnp:rootdevice\r\n"
        "NTS: ssdp:alive\r\n"
        "SERVER: " + config_.manufacturer + "/" + config_.modelName + " 1.0\r\n"
        "USN: uuid:" + deviceUUID_ + "::upnp:rootdevice\r\n"
        "\r\n";

    // This would send the notification to multicast address
    // Implementation would involve UDP multicast sending
}

void UPnPRenderer::SSDPServer::withdrawDevice() {
    Logger::info("UPnPRenderer::SSDP: Withdrawing device presence");

    // Send ssdp:byebye NOTIFY messages
}

// HTTPServer Implementation
UPnPRenderer::HTTPServer::HTTPServer(const UPnPConfig& config, UPnPRenderer* parent)
    : config_(config), parent_(parent) {
    baseURL_ = "http://" + config_.ipAddress.first + ":" + std::to_string(config_.port);
}

UPnPRenderer::HTTPServer::~HTTPServer() {
    stop();
}

bool UPnPRenderer::HTTPServer::start() {
    Logger::info("UPnPRenderer::HTTP: Starting HTTP server on port {}", config_.port);

    serverSocket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket_ < 0) {
        return false;
    }

    int opt = 1;
    if (setsockopt(serverSocket_, SOL_SOCKET, SO_REUSEADDR,
                   reinterpret_cast<const char*>(&opt), sizeof(opt)) < 0) {
        return false;
    }

    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(config_.port);

    if (bind(serverSocket_, reinterpret_cast<sockaddr*>(&serverAddr), sizeof(serverAddr)) < 0) {
        return false;
    }

    if (listen(serverSocket_, 5) < 0) {
        return false;
    }

    running_.store(true);
    serverThread_ = std::make_unique<std::thread>(&HTTPServer::serverThread, this);

    Logger::info("UPnPRenderer::HTTP: HTTP server started");
    return true;
}

void UPnPRenderer::HTTPServer::stop() {
    if (!running_.load()) {
        return;
    }

    running_.store(false);

    if (serverSocket_ >= 0) {
#ifdef _WIN32
        closesocket(serverSocket_);
#else
        close(serverSocket_);
#endif
        serverSocket_ = -1;
    }

    if (serverThread_ && serverThread_->joinable()) {
        serverThread_->join();
    }

    Logger::info("UPnPRenderer::HTTP: HTTP server stopped");
}

bool UPnPRenderer::HTTPServer::isRunning() const {
    return running_.load();
}

void UPnPRenderer::HTTPServer::serverThread() {
    Logger::info("UPnPRenderer::HTTP: HTTP server thread started");

    while (running_.load()) {
        fd_set readSet;
        FD_ZERO(&readSet);
        FD_SET(serverSocket_, &readSet);

        struct timeval timeout{};
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;

        int result = select(serverSocket_ + 1, &readSet, nullptr, nullptr, &timeout);
        if (result > 0 && FD_ISSET(serverSocket_, &readSet)) {
            sockaddr_in clientAddr{};
            socklen_t clientAddrLen = sizeof(clientAddr);

            int clientSocket = accept(serverSocket_, reinterpret_cast<sockaddr*>(&clientAddr), &clientAddrLen);
            if (clientSocket >= 0) {
                // Handle client request
                char buffer[4096];
                ssize_t received = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);
                if (received > 0) {
                    buffer[received] = '\0';
                    std::string request(buffer);
                    std::string response;

                    // Parse HTTP request
                    std::istringstream iss(request);
                    std::string method, path, version;
                    iss >> method >> path >> version;

                    // Handle request
                    handleHTTPRequest(method, path, {}, "", response);

                    // Send response
                    send(clientSocket, response.c_str(), response.size(), 0);
                }

#ifdef _WIN32
                closesocket(clientSocket);
#else
                close(clientSocket);
#endif
            }
        }
    }

    Logger::info("UPnPRenderer::HTTP: HTTP server thread stopped");
}

void UPnPRenderer::HTTPServer::handleHTTPRequest(const std::string& method, const std::string& path,
                                                 const std::map<std::string, std::string>& headers,
                                                 const std::string& body, std::string& response) {
    if (path == "/description.xml") {
        // Device description
        std::string description = parent_->generateDeviceDescription();
        response = "HTTP/1.1 200 OK\r\n"
                  "Content-Type: text/xml\r\n"
                  "Content-Length: " + std::to_string(description.length()) + "\r\n"
                  "\r\n" + description;
    } else if (path == "/icon.png") {
        // Device icon
        std::string iconData = parent_->generateIconData();
        response = "HTTP/1.1 200 OK\r\n"
                  "Content-Type: image/png\r\n"
                  "Content-Length: " + std::to_string(iconData.length()) + "\r\n"
                  "\r\n" + iconData;
    } else {
        response = "HTTP/1.1 404 Not Found\r\n"
                  "Content-Type: text/plain\r\n"
                  "Content-Length: 13\r\n"
                  "\r\nNot Found";
    }
}

std::string UPnPRenderer::HTTPServer::getDeviceURL() const {
    return baseURL_ + "/description.xml";
}

std::string UPnPRenderer::HTTPServer::getIconURL() const {
    return baseURL_ + "/icon.png";
}

// MediaReceiver Implementation
UPnPRenderer::MediaReceiver::MediaReceiver(const UPnPConfig& config, UPnPRenderer* parent)
    : config_(config), parent_(parent) {
    bufferCapacity_ = config_.bufferSize;
    audioBuffer_.reserve(bufferCapacity_);
}

UPnPRenderer::MediaReceiver::~MediaReceiver() {
    stop();
}

bool UPnPRenderer::MediaReceiver::start() {
    running_.store(true);
    streamingThread_ = std::make_unique<std::thread>(&MediaReceiver::streamingThread, this);
    return true;
}

void UPnPRenderer::MediaReceiver::stop() {
    running_.store(false);
    active_.store(false);

    if (streamingThread_ && streamingThread_->joinable()) {
        streamingThread_->join();
    }

    closeMediaStream();
}

bool UPnPRenderer::MediaReceiver::isRunning() const {
    return running_.load();
}

bool UPnPRenderer::MediaReceiver::receiveMedia(const std::string& uri) {
    currentURI_ = uri;
    return openMediaStream(uri);
}

void UPnPRenderer::MediaReceiver::streamingThread() {
    Logger::info("UPnPRenderer::Media: Streaming thread started");

    while (running_.load()) {
        if (active_.load()) {
            processMediaStream();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    Logger::info("UPnPRenderer::Media: Streaming thread stopped");
}

void UPnPRenderer::MediaReceiver::processMediaStream() {
    if (!mediaFile_) {
        return;
    }

    // Read audio data from file
    std::vector<uint8_t> buffer(4096);
    size_t bytesRead = fread(buffer.data(), 1, buffer.size(), mediaFile_);

    if (bytesRead > 0) {
        std::lock_guard<std::mutex> lock(bufferMutex_);
        if (bufferUsed_ + bytesRead <= bufferCapacity_) {
            audioBuffer_.insert(audioBuffer_.end(), buffer.begin(), buffer.begin() + bytesRead);
            bufferUsed_ += bytesRead;
        } else {
            bufferOverruns_++;
        }
        filePosition_ += bytesRead;
    } else if (feof(mediaFile_)) {
        // End of file
        Logger::info("UPnPRenderer::Media: End of media stream");
        active_.store(false);
    }
}

bool UPnPRenderer::MediaReceiver::openMediaStream(const std::string& uri) {
    Logger::info("UPnPRenderer::Media: Opening media stream: {}", uri);

    // For simplicity, assume local file access
    // Real implementation would handle HTTP, network streams, etc.
    mediaFile_ = fopen(uri.c_str(), "rb");
    if (!mediaFile_) {
        Logger::error("UPnPRenderer::Media: Failed to open media file: {}", uri);
        return false;
    }

    // Get file size
    fseek(mediaFile_, 0, SEEK_END);
    fileSize_ = ftell(mediaFile_);
    fseek(mediaFile_, 0, SEEK_SET);
    filePosition_ = 0;

    return true;
}

void UPnPRenderer::MediaReceiver::closeMediaStream() {
    if (mediaFile_) {
        fclose(mediaFile_);
        mediaFile_ = nullptr;
    }
    filePosition_ = 0;
    fileSize_ = 0;
}

std::vector<uint8_t> UPnPRenderer::MediaReceiver::getAudioBuffer(size_t bytes) {
    std::lock_guard<std::mutex> lock(bufferMutex_);
    std::vector<uint8_t> result;

    if (bufferUsed_ >= bytes) {
        result.assign(audioBuffer_.begin(), audioBuffer_.begin() + bytes);
        audioBuffer_.erase(audioBuffer_.begin(), audioBuffer_.begin() + bytes);
        bufferUsed_ -= bytes;
    } else {
        result.assign(audioBuffer_.begin(), audioBuffer_.end());
        audioBuffer_.clear();
        bufferUsed_ = 0;
        bufferUnderruns_++;
    }

    return result;
}

float UPnPRenderer::MediaReceiver::getBufferUtilization() const {
    std::lock_guard<std::mutex> lock(bufferMutex_);
    return static_cast<float>(bufferUsed_) / static_cast<float>(bufferCapacity_);
}

} // namespace vortex