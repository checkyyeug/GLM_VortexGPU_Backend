#include "roon_bridge.hpp"
#include "../utils/logger.hpp"

#include <random>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstring>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#endif

namespace vortex {

// RoonBridgeServer Implementation
RoonBridgeServer::RoonBridgeServer(const RoonBridgeConfig& config) : config_(config) {
    Logger::info("RoonBridgeServer: Initializing Roon Bridge server '{}' on port {}",
                 config.displayName, config.raatPort);

    statistics_.startTime = std::chrono::steady_clock::now();
    statistics_.lastActivity = statistics_.startTime;

    // Validate configuration
    if (!validateConfiguration(config)) {
        throw std::invalid_argument("Invalid Roon Bridge configuration");
    }

    // Initialize server components
    raatServer_ = std::make_unique<RAATServer>(config.raatPort, this);

    if (config.enableAirPlay) {
        airplayServer_ = std::make_unique<AirPlayServer>(config.airplayPort, this);
    }

    httpServer_ = std::make_unique<HTTPServer>(config.httpPort, this);

    if (config.enableAutoDiscovery) {
        discoveryServer_ = std::make_unique<DiscoveryServer>(this);
    }
}

RoonBridgeServer::~RoonBridgeServer() {
    stop();
    Logger::info("RoonBridgeServer: Roon Bridge server destroyed");
}

bool RoonBridgeServer::start() {
    if (running_.load()) {
        Logger::warn("RoonBridgeServer: Already running");
        return true;
    }

    Logger::info("RoonBridgeServer: Starting Roon Bridge server");

    try {
        // Start core servers
        if (config_.enableRAAT && !startRAATServer()) {
            setError("Failed to start RAAT server");
            return false;
        }

        if (config_.enableAirPlay && !startAirPlayServer()) {
            setError("Failed to start AirPlay server");
            return false;
        }

        if (!startHTTPServer()) {
            setError("Failed to start HTTP server");
            return false;
        }

        if (config_.enableAutoDiscovery && !startDiscoveryServer()) {
            setError("Failed to start discovery server");
            return false;
        }

        // Start background threads
        processingThread_ = std::make_unique<std::thread>(&RoonBridgeServer::audioProcessingThread, this);

        if (config_.enableAutoDiscovery) {
            discoveryThread_ = std::make_unique<std::thread>(&RoonBridgeServer::deviceDiscoveryThread, this);
        }

        running_.store(true);
        ready_.store(true);

        Logger::info("RoonBridgeServer: Roon Bridge server started successfully");
        return true;

    } catch (const std::exception& e) {
        setError("Failed to start server: " + std::string(e.what()));
        return false;
    }
}

void RoonBridgeServer::stop() {
    if (!running_.load()) {
        return;
    }

    Logger::info("RoonBridgeServer: Stopping Roon Bridge server");

    running_.store(false);
    ready_.store(false);

    // Stop all streams
    {
        std::lock_guard<std::mutex> lock(streamsMutex_);
        for (auto& [streamId, stream] : streams_) {
            stopStream(streamId);
        }
        streams_.clear();
    }

    // Wake up threads
    processingCondition_.notify_all();

    // Wait for threads to finish
    if (processingThread_ && processingThread_->joinable()) {
        processingThread_->join();
    }
    if (discoveryThread_ && discoveryThread_->joinable()) {
        discoveryThread_->join();
    }

    // Stop servers
    stopRAATServer();
    stopAirPlayServer();
    stopHTTPServer();
    stopDiscoveryServer();

    Logger::info("RoonBridgeServer: Roon Bridge server stopped");
}

bool RoonBridgeServer::isRunning() const {
    return running_.load();
}

bool RoonBridgeServer::isReady() const {
    return ready_.load();
}

bool RoonBridgeServer::registerDevice(const RoonDevice& device) {
    if (!validateDevice(device)) {
        setError("Invalid device configuration");
        return false;
    }

    Logger::info("RoonBridgeServer: Registering device '{}' ({})", device.displayName, device.deviceId);

    {
        std::lock_guard<std::mutex> lock(devicesMutex_);
        devices_[device.deviceId] = device;

        // Update statistics
        std::lock_guard<std::mutex> statsLock(statsMutex_);
        statistics_.registeredDevices = devices_.size();
    }

    // Announce device to Roon
    if (discoveryServer_) {
        announceDevicePresence();
    }

    if (deviceRegisteredCallback_) {
        deviceRegisteredCallback_("registered", device);
    }

    return true;
}

bool RoonBridgeServer::unregisterDevice(const std::string& deviceId) {
    Logger::info("RoonBridgeServer: Unregistering device {}", deviceId);

    std::lock_guard<std::mutex> lock(devicesMutex_);
    auto it = devices_.find(deviceId);
    if (it == devices_.end()) {
        return false;
    }

    RoonDevice device = it->second;
    devices_.erase(it);

    // Update statistics
    {
        std::lock_guard<std::mutex> statsLock(statsMutex_);
        statistics_.registeredDevices = devices_.size();
    }

    if (deviceUnregisteredCallback_) {
        deviceUnregisteredCallback_("unregistered", device);
    }

    return true;
}

RoonBridgeServer::RoonDevice RoonBridgeServer::getDevice(const std::string& deviceId) const {
    std::lock_guard<std::mutex> lock(devicesMutex_);
    auto it = devices_.find(deviceId);
    return it != devices_.end() ? it->second : RoonDevice{};
}

std::vector<RoonBridgeServer::RoonDevice> RoonBridgeServer::getRegisteredDevices() const {
    std::lock_guard<std::mutex> lock(devicesMutex_);
    std::vector<RoonDevice> devices;

    for (const auto& [deviceId, device] : devices_) {
        devices.push_back(device);
    }

    return devices;
}

bool RoonBridgeServer::updateDeviceCapabilities(const std::string& deviceId, const RoonDevice& device) {
    if (!validateDevice(device)) {
        setError("Invalid device configuration");
        return false;
    }

    std::lock_guard<std::mutex> lock(devicesMutex_);
    auto it = devices_.find(deviceId);
    if (it == devices_.end()) {
        return false;
    }

    it->second = device;

    if (deviceRegisteredCallback_) {
        deviceRegisteredCallback_("updated", device);
    }

    return true;
}

std::string RoonBridgeServer::createStream(const std::string& deviceId, TransportProtocol protocol) {
    std::lock_guard<std::mutex> lock(devicesMutex_);
    auto deviceIt = devices_.find(deviceId);
    if (deviceIt == devices_.end()) {
        setError("Device not found: " + deviceId);
        return "";
    }

    std::string streamId = generateStreamId();

    RoonStream stream;
    stream.streamId = streamId;
    stream.deviceId = deviceId;
    stream.protocol = protocol;
    stream.sampleRate = 44100; // Default, will be updated by Roon
    stream.channels = 2;
    stream.bitDepth = BitDepth::BIT_16;
    stream.creationTime = std::chrono::steady_clock::now();
    stream.lastUpdate = stream.creationTime;
    stream.isActive = false;

    {
        std::lock_guard<std::mutex> streamLock(streamsMutex_);
        streams_[streamId] = stream;

        // Update statistics
        std::lock_guard<std::mutex> statsLock(statsMutex_);
        statistics_.totalStreamsCreated++;
        statistics_.streamsByDevice[deviceId]++;
        statistics_.streamsByProtocol[protocol]++;
    }

    Logger::info("RoonBridgeServer: Created stream {} for device {} using {}",
                 streamId, deviceId, getProtocolName(protocol));

    return streamId;
}

bool RoonBridgeServer::startStream(const std::string& streamId) {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    auto it = streams_.find(streamId);
    if (it == streams_.end()) {
        return false;
    }

    RoonStream& stream = it->second;
    stream.isActive = true;
    stream.isPaused = false;
    stream.startTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    // Add to processing queue
    {
        std::lock_guard<std::mutex> queueLock(processingQueueMutex_);
        processingQueue_.push(streamId);
    }
    processingCondition_.notify_one();

    if (streamStartedCallback_) {
        streamStartedCallback_(streamId, stream);
    }

    Logger::info("RoonBridgeServer: Started stream {}", streamId);
    return true;
}

bool RoonBridgeServer::pauseStream(const std::string& streamId) {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    auto it = streams_.find(streamId);
    if (it == streams_.end()) {
        return false;
    }

    it->second.isPaused = true;
    Logger::info("RoonBridgeServer: Paused stream {}", streamId);
    return true;
}

bool RoonBridgeServer::stopStream(const std::string& streamId) {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    auto it = streams_.find(streamId);
    if (it == streams_.end()) {
        return false;
    }

    RoonStream& stream = it->second;
    stream.isActive = false;
    stream.isPaused = false;
    stream.endTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    // Update statistics
    if (stream.position > 0) {
        std::lock_guard<std::mutex> statsLock(statsMutex_);
        statistics_.totalTracksPlayed++;
    }

    if (streamStoppedCallback_) {
        streamStoppedCallback_(streamId, stream);
    }

    Logger::info("RoonBridgeServer: Stopped stream {}", streamId);
    return true;
}

bool RoonBridgeServer::seekStream(const std::string& streamId, uint64_t position) {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    auto it = streams_.find(streamId);
    if (it == streams_.end()) {
        return false;
    }

    it->second.position = position;
    Logger::info("RoonBridgeServer: Stream {} seeked to position {}", streamId, position);
    return true;
}

RoonBridgeServer::RoonStream RoonBridgeServer::getStream(const std::string& streamId) const {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    auto it = streams_.find(streamId);
    return it != streams_.end() ? it->second : RoonStream{};
}

std::vector<RoonBridgeServer::RoonStream> RoonBridgeServer::getActiveStreams() const {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    std::vector<RoonStream> activeStreams;

    for (const auto& [streamId, stream] : streams_) {
        if (stream.isActive) {
            activeStreams.push_back(stream);
        }
    }

    return activeStreams;
}

bool RoonBridgeServer::receiveAudioData(const std::string& streamId, const std::vector<uint8_t>& audioData, uint64_t timestamp) {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    auto it = streams_.find(streamId);
    if (it == streams_.end() || !it->second.isActive) {
        return false;
    }

    RoonStream& stream = it->second;

    // Append to buffer
    size_t spaceAvailable = stream.bufferCapacity - stream.bufferUsed;
    if (audioData.size() > spaceAvailable) {
        // Buffer overflow
        stream.overruns++;
        Logger::warn("RoonBridgeServer: Buffer overflow for stream {}", streamId);
        return false;
    }

    stream.audioBuffer.insert(stream.audioBuffer.end(), audioData.begin(), audioData.end());
    stream.bufferUsed += audioData.size();
    stream.presentationTime = timestamp;
    stream.lastUpdate = std::chrono::steady_clock::now();

    // Update statistics
    {
        std::lock_guard<std::mutex> statsLock(statsMutex_);
        statistics_.totalAudioBytesReceived += audioData.size();
        statistics_.lastActivity = stream.lastUpdate;
    }

    // Notify audio received callback
    if (audioReceivedCallback_) {
        std::vector<float> audioFloat;
        if (convertAudioFormat(audioData, audioFloat, stream)) {
            audioReceivedCallback_(streamId, audioFloat);
        }
    }

    return true;
}

bool RoonBridgeServer::processAudioFrame(const std::string& streamId, std::vector<float>& audioBuffer) {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    auto it = streams_.find(streamId);
    if (it == streams_.end() || !it->second.isActive || it->second.isPaused) {
        return false;
    }

    RoonStream& stream = it->second;

    // Convert raw audio buffer to float
    if (stream.bufferUsed == 0) {
        // No data available
        stream.underruns++;
        return false;
    }

    if (!convertAudioFormat(stream.audioBuffer, audioBuffer, stream)) {
        return false;
    }

    // Clear processed data
    stream.audioBuffer.clear();
    stream.bufferUsed = 0;

    return true;
}

void RoonBridgeServer::setStreamVolume(const std::string& streamId, float volume) {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    auto it = streams_.find(streamId);
    if (it != streams_.end()) {
        // Volume control would be applied during audio processing
        // Store volume for later use
        Logger::info("RoonBridgeServer: Set stream {} volume to {:.2f}", streamId, volume);
    }
}

void RoonBridgeServer::setStreamMute(const std::string& streamId, bool muted) {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    auto it = streams_.find(streamId);
    if (it != streams_.end()) {
        // Mute state would be applied during audio processing
        Logger::info("RoonBridgeServer: Set stream {} mute to {}", streamId, muted ? "on" : "off");
    }
}

RoonBridgeServer::BridgeStatistics RoonBridgeServer::getStatistics() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    return statistics_;
}

void RoonBridgeServer::resetStatistics() {
    std::lock_guard<std::mutex> lock(statsMutex_);
    statistics_ = BridgeStatistics{};
    statistics_.startTime = std::chrono::steady_clock::now();
    statistics_.lastActivity = statistics_.startTime;

    // Reset per-device statistics
    {
        std::lock_guard<std::mutex> deviceLock(devicesMutex_);
        for (auto& [deviceId, device] : devices_) {
            device.tracksPlayed = 0;
            device.bytesReceived = 0;
            device.underruns = 0;
            device.overruns = 0;
        }
    }
}

RoonBridgeServer::HealthStatus RoonBridgeServer::getHealthStatus() const {
    HealthStatus status;

    status.raatServerRunning = raatServer_ && raatServer_->isRunning();
    status.airplayServerRunning = airplayServer_ && airplayServer_->isRunning();
    status.httpServerRunning = httpServer_ && httpServer_->isRunning();

    {
        std::lock_guard<std::mutex> lock(streamsMutex_);
        status.activeStreams = 0;
        status.healthyStreams = 0;

        for (const auto& [streamId, stream] : streams_) {
            if (stream.isActive) {
                status.activeStreams++;
                if (stream.underruns < 10 && stream.overruns < 10) {
                    status.healthyStreams++;
                } else {
                    status.errors.push_back("Stream " + streamId + " has excessive underruns/overruns");
                }
            }
        }
    }

    status.isHealthy = status.raatServerRunning && status.httpServerRunning &&
                      (status.activeStreams == 0 || status.healthyStreams == status.activeStreams);

    return status;
}

void RoonBridgeServer::updateConfiguration(const RoonBridgeConfig& config) {
    if (!validateConfiguration(config)) {
        setError("Invalid configuration");
        return;
    }

    config_ = config;
    Logger::info("RoonBridgeServer: Configuration updated");
}

RoonBridgeServer::RoonBridgeConfig RoonBridgeServer::getConfiguration() const {
    return config_;
}

void RoonBridgeServer::setStreamStartedCallback(StreamEventCallback callback) {
    streamStartedCallback_ = callback;
}

void RoonBridgeServer::setStreamStoppedCallback(StreamEventCallback callback) {
    streamStoppedCallback_ = callback;
}

void RoonBridgeServer::setDeviceRegisteredCallback(DeviceEventCallback callback) {
    deviceRegisteredCallback_ = callback;
}

void RoonBridgeServer::setDeviceUnregisteredCallback(DeviceEventCallback callback) {
    deviceUnregisteredCallback_ = callback;
}

void RoonBridgeServer::setAudioReceivedCallback(AudioEventCallback callback) {
    audioReceivedCallback_ = callback;
}

void RoonBridgeServer::setErrorCallback(ErrorCallback callback) {
    errorCallback_ = callback;
}

// Private methods implementation
bool RoonBridgeServer::startRAATServer() {
    if (!raatServer_) {
        return false;
    }

    return raatServer_->start();
}

bool RoonBridgeServer::startAirPlayServer() {
    if (!airplayServer_) {
        return false;
    }

    return airplayServer_->start();
}

bool RoonBridgeServer::startHTTPServer() {
    if (!httpServer_) {
        return false;
    }

    return httpServer_->start();
}

bool RoonBridgeServer::startDiscoveryServer() {
    if (!discoveryServer_) {
        return false;
    }

    return discoveryServer_->start();
}

void RoonBridgeServer::stopRAATServer() {
    if (raatServer_) {
        raatServer_->stop();
    }
}

void RoonBridgeServer::stopAirPlayServer() {
    if (airplayServer_) {
        airplayServer_->stop();
    }
}

void RoonBridgeServer::stopHTTPServer() {
    if (httpServer_) {
        httpServer_->stop();
    }
}

void RoonBridgeServer::stopDiscoveryServer() {
    if (discoveryServer_) {
        discoveryServer_->stop();
    }
}

void RoonBridgeServer::audioProcessingThread() {
    Logger::info("RoonBridgeServer: Audio processing thread started");

    while (running_.load()) {
        std::unique_lock<std::mutex> lock(processingQueueMutex_);
        processingCondition_.wait(lock, [this] {
            return !processingQueue_.empty() || !running_.load();
        });

        if (!running_.load()) {
            break;
        }

        while (!processingQueue_.empty()) {
            std::string streamId = processingQueue_.front();
            processingQueue_.pop();
            lock.unlock();

            // Process the stream
            {
                std::lock_guard<std::mutex> streamLock(streamsMutex_);
                auto it = streams_.find(streamId);
                if (it != streams_.end() && it->second.isActive) {
                    processRAATStream(streamId);
                }
            }

            lock.lock();
        }
    }

    Logger::info("RoonBridgeServer: Audio processing thread stopped");
}

void RoonBridgeServer::processRAATStream(const std::string& streamId) {
    // Implementation for processing RAAT stream
    // This would handle audio decoding, format conversion, and output
}

void RoonBridgeServer::deviceDiscoveryThread() {
    Logger::info("RoonBridgeServer: Device discovery thread started");

    while (running_.load()) {
        try {
            if (discoveryServer_) {
                announceDevicePresence();
            }

            std::this_thread::sleep_for(std::chrono::seconds(30));

        } catch (const std::exception& e) {
            Logger::error("RoonBridgeServer: Discovery thread error: {}", e.what());
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    }

    Logger::info("RoonBridgeServer: Device discovery thread stopped");
}

void RoonBridgeServer::announceDevicePresence() {
    if (discoveryServer_) {
        discoveryServer_->announceService();
    }
}

std::string RoonBridgeServer::generateStreamId() {
    static std::atomic<uint32_t> counter{0};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);

    return "stream_" + std::to_string(counter++) + "_" + std::to_string(dis(gen));
}

bool RoonBridgeServer::validateConfiguration(const RoonBridgeConfig& config) const {
    if (config.displayName.empty()) {
        return false;
    }

    if (config.raatPort == 0 || config.raatPort > 65535) {
        return false;
    }

    if (config.maxSampleRate == 0 || config.maxSampleRate > 768000) {
        return false;
    }

    return true;
}

bool RoonBridgeServer::validateDevice(const RoonDevice& device) const {
    if (device.deviceId.empty() || device.displayName.empty()) {
        return false;
    }

    if (device.supportedSampleRates.empty()) {
        return false;
    }

    return true;
}

bool RoonBridgeServer::convertAudioFormat(const std::vector<uint8_t>& input, std::vector<float>& output,
                                          const RoonStream& stream) const {
    // Convert audio data based on bit depth and format
    uint32_t bitDepth = getBitDepthValue(stream.bitDepth);
    size_t samplesPerChannel = input.size() / (bitDepth / 8 * stream.channels);

    output.clear();
    output.reserve(samplesPerChannel * stream.channels);

    switch (stream.bitDepth) {
        case BitDepth::BIT_16: {
            const int16_t* samples = reinterpret_cast<const int16_t*>(input.data());
            float scale = 1.0f / 32768.0f;
            for (size_t i = 0; i < samplesPerChannel * stream.channels; ++i) {
                output.push_back(samples[i] * scale);
            }
            break;
        }
        case BitDepth::BIT_24: {
            const uint8_t* samples = input.data();
            float scale = 1.0f / 8388608.0f;
            for (size_t i = 0; i < samplesPerChannel * stream.channels; ++i) {
                int32_t sample = (samples[i*3] << 8) | (samples[i*3+1] << 16) | (samples[i*3+2] << 24);
                output.push_back((sample >> 8) * scale);
            }
            break;
        }
        case BitDepth::BIT_32: {
            const int32_t* samples = reinterpret_cast<const int32_t*>(input.data());
            float scale = 1.0f / 2147483648.0f;
            for (size_t i = 0; i < samplesPerChannel * stream.channels; ++i) {
                output.push_back(samples[i] * scale);
            }
            break;
        }
        default:
            return false;
    }

    return true;
}

void RoonBridgeServer::setError(const std::string& error) {
    lastError_ = error;
    Logger::error("RoonBridgeServer: {}", error);

    if (errorCallback_) {
        errorCallback_("server", error);
    }
}

std::string RoonBridgeServer::getProtocolName(TransportProtocol protocol) const {
    switch (protocol) {
        case TransportProtocol::RAAT: return "RAAT";
        case TransportProtocol::AIRPLAY: return "AirPlay";
        case TransportProtocol::MERIDIAN: return "Meridian";
        case TransportProtocol::DLNA: return "DLNA";
        default: return "Unknown";
    }
}

uint32_t RoonBridgeServer::getBitDepthValue(BitDepth bitDepth) const {
    switch (bitDepth) {
        case BitDepth::BIT_16: return 16;
        case BitDepth::BIT_24: return 24;
        case BitDepth::BIT_32: return 32;
        case BitDepth::BIT_64: return 64;
        default: return 16;
    }
}

// RAATServer Implementation
RoonBridgeServer::RAATServer::RAATServer(uint16_t port, RoonBridgeServer* parent)
    : port_(port), parent_(parent) {
}

RoonBridgeServer::RAATServer::~RAATServer() {
    stop();
}

bool RoonBridgeServer::RAATServer::start() {
    Logger::info("RoonBridgeServer::RAAT: Starting RAAT server on port {}", port_);

#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        return false;
    }
#endif

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
    serverAddr.sin_port = htons(port_);

    if (bind(serverSocket_, reinterpret_cast<sockaddr*>(&serverAddr), sizeof(serverAddr)) < 0) {
        return false;
    }

    if (listen(serverSocket_, 5) < 0) {
        return false;
    }

    running_.store(true);
    serverThread_ = std::make_unique<std::thread>(&RAATServer::serverThread, this);

    Logger::info("RoonBridgeServer::RAAT: RAAT server started");
    return true;
}

void RoonBridgeServer::RAATServer::stop() {
    if (!running_.load()) {
        return;
    }

    Logger::info("RoonBridgeServer::RAAT: Stopping RAAT server");

    running_.store(false);

#ifdef _WIN32
    closesocket(serverSocket_);
    WSACleanup();
#else
    close(serverSocket_);
#endif

    if (serverThread_ && serverThread_->joinable()) {
        serverThread_->join();
    }

    {
        std::lock_guard<std::mutex> lock(clientSocketsMutex_);
        for (int clientSocket : clientSockets_) {
#ifdef _WIN32
            closesocket(clientSocket);
#else
            close(clientSocket);
#endif
        }
        clientSockets_.clear();
    }

    Logger::info("RoonBridgeServer::RAAT: RAAT server stopped");
}

bool RoonBridgeServer::RAATServer::isRunning() const {
    return running_.load();
}

void RoonBridgeServer::RAATServer::serverThread() {
    Logger::info("RoonBridgeServer::RAAT: RAAT server thread started");

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
                std::lock_guard<std::mutex> lock(clientSocketsMutex_);
                clientSockets_.push_back(clientSocket);

                // Handle client connection in a separate thread
                std::thread clientThread(&RAATServer::handleClientConnection, this, clientSocket);
                clientThread.detach();
            }
        }
    }

    Logger::info("RoonBridgeServer::RAAT: RAAT server thread stopped");
}

void RoonBridgeServer::RAATServer::handleClientConnection(int clientSocket) {
    Logger::info("RoonBridgeServer::RAAT: New client connection");

    std::vector<uint8_t> buffer(4096);

    while (running_.load()) {
        int bytesReceived = recv(clientSocket, reinterpret_cast<char*>(buffer.data()),
                                buffer.size(), 0);
        if (bytesReceived <= 0) {
            break;
        }

        buffer.resize(bytesReceived);
        processMessage(buffer);
        buffer.resize(4096);
    }

#ifdef _WIN32
    closesocket(clientSocket);
#else
    close(clientSocket);
#endif

    // Remove from client list
    {
        std::lock_guard<std::mutex> lock(clientSocketsMutex_);
        clientSockets_.erase(std::remove(clientSockets_.begin(), clientSockets_.end(), clientSocket),
                             clientSockets_.end());
    }
}

bool RoonBridgeServer::RAATServer::processMessage(const std::vector<uint8_t>& message) {
    // Placeholder for RAAT message processing
    // Real implementation would parse RAAT protocol messages
    return parent_->processRAATMessage(message);
}

// AirPlayServer Implementation
RoonBridgeServer::AirPlayServer::AirPlayServer(uint16_t port, RoonBridgeServer* parent)
    : port_(port), parent_(parent) {
}

RoonBridgeServer::AirPlayServer::~AirPlayServer() {
    stop();
}

bool RoonBridgeServer::AirPlayServer::start() {
    Logger::info("RoonBridgeServer::AirPlay: Starting AirPlay server on port {}", port_);

    running_.store(true);
    // Placeholder implementation
    return true;
}

void RoonBridgeServer::AirPlayServer::stop() {
    running_.store(false);
}

bool RoonBridgeServer::AirPlayServer::isRunning() const {
    return running_.load();
}

// HTTPServer Implementation
RoonBridgeServer::HTTPServer::HTTPServer(uint16_t port, RoonBridgeServer* parent)
    : port_(port), parent_(parent) {
}

RoonBridgeServer::HTTPServer::~HTTPServer() {
    stop();
}

bool RoonBridgeServer::HTTPServer::start() {
    Logger::info("RoonBridgeServer::HTTP: Starting HTTP server on port {}", port_);

    running_.store(true);
    serverThread_ = std::make_unique<std::thread>(&HTTPServer::serverThread, this);
    return true;
}

void RoonBridgeServer::HTTPServer::stop() {
    running_.store(false);

    if (serverThread_ && serverThread_->joinable()) {
        serverThread_->join();
    }
}

bool RoonBridgeServer::HTTPServer::isRunning() const {
    return running_.load();
}

void RoonBridgeServer::HTTPServer::serverThread() {
    Logger::info("RoonBridgeServer::HTTP: HTTP server thread started");

    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    Logger::info("RoonBridgeServer::HTTP: HTTP server thread stopped");
}

std::string RoonBridgeServer::HTTPServer::generateDeviceInfoJSON() {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"device_id\": \"" << parent_->config_.deviceId << "\",\n";
    oss << "  \"display_name\": \"" << parent_->config_.displayName << "\",\n";
    oss << "  \"raat_port\": " << parent_->config_.raatPort << ",\n";
    oss << "  \"http_port\": " << parent_->config_.httpPort << "\n";
    oss << "}";
    return oss.str();
}

// DiscoveryServer Implementation
RoonBridgeServer::DiscoveryServer::DiscoveryServer(RoonBridgeServer* parent)
    : parent_(parent) {
}

RoonBridgeServer::DiscoveryServer::~DiscoveryServer() {
    stop();
}

bool RoonBridgeServer::DiscoveryServer::start() {
    Logger::info("RoonBridgeServer::Discovery: Starting discovery service");

    running_.store(true);
    discoveryThread_ = std::make_unique<std::thread>(&DiscoveryServer::discoveryThread, this);
    return true;
}

void RoonBridgeServer::DiscoveryServer::stop() {
    running_.store(false);

    if (serviceAnnounced_) {
        withdrawService();
    }

    if (discoveryThread_ && discoveryThread_->joinable()) {
        discoveryThread_->join();
    }
}

bool RoonBridgeServer::DiscoveryServer::isRunning() const {
    return running_.load();
}

void RoonBridgeServer::DiscoveryServer::discoveryThread() {
    Logger::info("RoonBridgeServer::Discovery: Discovery thread started");

    while (running_.load()) {
        if (!serviceAnnounced_) {
            announceService();
        }

        std::this_thread::sleep_for(std::chrono::seconds(30));
    }

    Logger::info("RoonBridgeServer::Discovery: Discovery thread stopped");
}

void RoonBridgeServer::DiscoveryServer::announceService() {
    serviceAnnounced_ = true;
    Logger::info("RoonBridgeServer::Discovery: Service announced");
}

void RoonBridgeServer::DiscoveryServer::withdrawService() {
    serviceAnnounced_ = false;
    Logger::info("RoonBridgeServer::Discovery: Service withdrawn");
}

} // namespace vortex