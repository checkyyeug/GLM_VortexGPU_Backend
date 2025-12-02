#include "hqplayer_naa.hpp"
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
#include <fcntl.h>
#include <errno.h>
#endif

namespace vortex {

// HQPlayerNAAClient Implementation
HQPlayerNAAClient::HQPlayerNAAClient(const NAAConfig& config) : config_(config) {
    Logger::info("HQPlayerNAAClient: Initializing NAA client '{}' for server {}:{}",
                 config.deviceName, config.serverAddress.first, config.serverPort);

    // Validate configuration
    if (!validateConfiguration(config)) {
        throw std::invalid_argument("Invalid HQPlayer NAA configuration");
    }

    // Initialize device info
    deviceInfo_.deviceId = config.deviceId;
    deviceInfo_.deviceName = config.deviceName;
    deviceInfo_.manufacturer = "VortexGPU";
    deviceInfo_.model = "Audio Backend";
    deviceInfo_.version = "1.0.0";

    // Setup capabilities based on configuration
    deviceInfo_.supportedSampleRates = {44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000, 705600, 768000};
    if (config.maxSampleRate > 768000) {
        deviceInfo_.supportedSampleRates.push_back(1536000);
    }
    deviceInfo_.supportedBitDepths = {16, 24, 32};
    deviceInfo_.supportedChannels = {1, 2};
    if (config.maxChannels >= 4) {
        deviceInfo_.supportedChannels.push_back(4);
    }
    if (config.maxChannels >= 6) {
        deviceInfo_.supportedChannels.push_back(6);
    }
    if (config.maxChannels >= 8) {
        deviceInfo_.supportedChannels.push_back(8);
    }

    deviceInfo_.supportsDSD = config.enableDSDSupport;
    if (config.enableDSDSupport) {
        deviceInfo_.dsdRates = {2822400, 5644800, 11289600, 22579200, 45158400}; // DSD64 to DSD1024
        if (config.maxDSDRate >= 45158400) {
            deviceInfo_.dsdRates.push_back(90316800); // DSD2048
        }
        deviceInfo_.supportsDoP = config.enableDoP;
        deviceInfo_.supportsNativeDSD = config.enableNativeDSD;
    }

    deviceInfo_.supportsUpsampling = config.enableUpsampling;
    deviceInfo_.maxUpsampleRate = config.defaultTargetRate;
    deviceInfo_.supportsVolumeControl = config.enableVolumeControl;
    deviceInfo_.preferredLatencyMs = config.targetLatency;
    deviceInfo_.maxLatencyMs = config.targetLatency * 10;

    // Initialize statistics
    statistics_.connectionState = ConnectionState::DISCONNECTED;
    statistics_.connectionTime = std::chrono::steady_clock::now();
    statistics_.lastActivity = statistics_.connectionTime;

    // Create transport based on configuration
    switch (config.transportMode) {
        case TransportMode::TCP:
            transport_ = std::make_unique<TCPTransport>();
            break;
        case TransportMode::UDP:
            transport_ = std::make_unique<UDPTransport>();
            break;
        case TransportMode::RTP:
            // RTP transport would be implemented here
            transport_ = std::make_unique<TCPTransport>();
            break;
        case TransportMode::WEBSOCKET:
            // WebSocket transport would be implemented here
            transport_ = std::make_unique<TCPTransport>();
            break;
        default:
            transport_ = std::make_unique<TCPTransport>();
            break;
    }

    Logger::info("HQPlayerNAAClient: NAA client initialized with {} transport",
                 getTransportModeName(config.transportMode));
}

HQPlayerNAAClient::~HQPlayerNAAClient() {
    disconnect();
    Logger::info("HQPlayerNAAClient: NAA client destroyed");
}

bool HQPlayerNAAClient::connect() {
    if (isConnected()) {
        Logger::warn("HQPlayerNAAClient: Already connected");
        return true;
    }

    Logger::info("HQPlayerNAAClient: Connecting to HQPlayer server at {}:{}",
                 config_.serverAddress.first, config_.serverPort);

    setConnectionState(ConnectionState::CONNECTING);

    try {
        // Connect via transport
        if (!transport_->connect(config_.serverAddress)) {
            setError("Failed to connect via transport");
            setConnectionState(ConnectionState::ERROR);
            return false;
        }

        // Perform HQPlayer handshake
        if (!handleHandshake()) {
            setError("Failed to perform handshake");
            transport_->disconnect();
            setConnectionState(ConnectionState::ERROR);
            return false;
        }

        // Start connection monitoring
        connectionThread_ = std::make_unique<std::thread>(&HQPlayerNAAClient::connectionThread, this);

        // Start audio processing
        processingActive_.store(true);
        processingThread_ = std::make_unique<std::thread>(&HQPlayerNAAClient::audioProcessingThread, this);

        setConnectionState(ConnectionState::CONNECTED);
        statistics_.connectionTime = std::chrono::steady_clock::now();

        Logger::info("HQPlayerNAAClient: Successfully connected to HQPlayer server");
        return true;

    } catch (const std::exception& e) {
        setError("Connection failed: " + std::string(e.what()));
        transport_->disconnect();
        setConnectionState(ConnectionState::ERROR);
        return false;
    }
}

void HQPlayerNAAClient::disconnect() {
    if (!isConnected()) {
        return;
    }

    Logger::info("HQPlayerNAAClient: Disconnecting from HQPlayer server");

    setConnectionState(ConnectionState::DISCONNECTED);
    processingActive_.store(false);

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
    if (connectionThread_ && connectionThread_->joinable()) {
        connectionThread_->join();
    }

    // Disconnect transport
    transport_->disconnect();

    Logger::info("HQPlayerNAAClient: Disconnected from HQPlayer server");
}

bool HQPlayerNAAClient::isConnected() const {
    return connectionState_.load() == ConnectionState::CONNECTED ||
           connectionState_.load() == ConnectionState::STREAMING;
}

HQPlayerNAAClient::ConnectionState HQPlayerNAAClient::getConnectionState() const {
    return connectionState_.load();
}

bool HQPlayerNAAClient::registerDevice(const NAADeviceInfo& device) {
    if (!validateDeviceInfo(device)) {
        setError("Invalid device information");
        return false;
    }

    Logger::info("HQPlayerNAAClient: Registering device '{}' ({})", device.deviceName, device.deviceId);

    deviceInfo_ = device;

    // Send device info to HQPlayer if connected
    if (isConnected()) {
        return handleDeviceInfoRequest();
    }

    return true;
}

HQPlayerNAAClient::NAADeviceInfo HQPlayerNAAClient::getDeviceInfo() const {
    return deviceInfo_;
}

bool HQPlayerNAAClient::updateDeviceInfo(const NAADeviceInfo& device) {
    if (!validateDeviceInfo(device)) {
        setError("Invalid device information");
        return false;
    }

    deviceInfo_ = device;

    // Update server with new device info if connected
    if (isConnected()) {
        return handleDeviceInfoRequest();
    }

    return true;
}

std::vector<HQPlayerNAAClient::UpsamplingFilter> HQPlayerNAAClient::getAvailableFilters() const {
    return {UpsamplingFilter::NONE, UpsamplingFilter::POLY_SINC, UpsamplingFilter::POLY_SINC_XTR,
            UpsamplingFilter::POLY_SINC_MP, UpsamplingFilter::SINC_M, UpsamplingFilter::SINC_L};
}

std::string HQPlayerNAAClient::createStream() {
    std::string streamId = generateStreamId();

    NAAStream stream;
    stream.streamId = streamId;
    stream.deviceId = deviceInfo_.deviceId;
    stream.sampleRate = 44100;
    stream.channels = 2;
    stream.bitDepth = 24;
    stream.format = AudioFormat::PCM_S24LE;
    stream.isDSD = false;
    stream.dsdRate = 0;
    stream.upsamplingFilter = config_.defaultFilter;
    stream.targetSampleRate = config_.defaultTargetRate;
    stream.transportMode = config_.transportMode;
    stream.dataPort = config_.serverPort;
    stream.controlPort = config_.serverPort + 1;
    stream.remoteAddress = config_.serverAddress.first;
    stream.packetSize = config_.packetSize;
    stream.latencyMs = config_.targetLatency;
    stream.creationTime = std::chrono::steady_clock::now();

    // Initialize stream buffer
    if (!initializeStreamBuffer(stream)) {
        setError("Failed to initialize stream buffer");
        return "";
    }

    {
        std::lock_guard<std::mutex> lock(streamsMutex_);
        streams_[streamId] = stream;

        // Update statistics
        std::lock_guard<std::mutex> statsLock(statsMutex_);
        statistics_.packetsByStream[streamId] = 0;
    }

    Logger::info("HQPlayerNAAClient: Created stream {}", streamId);
    return streamId;
}

bool HQPlayerNAAClient::startStream(const std::string& streamId) {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    auto it = streams_.find(streamId);
    if (it == streams_.end()) {
        setError("Stream not found: " + streamId);
        return false;
    }

    NAAStream& stream = it->second;
    stream.isActive = true;
    stream.isPaused = false;
    stream.state = ConnectionState::STREAMING;
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

    Logger::info("HQPlayerNAAClient: Started stream {}", streamId);
    return true;
}

bool HQPlayerNAAClient::pauseStream(const std::string& streamId) {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    auto it = streams_.find(streamId);
    if (it == streams_.end()) {
        return false;
    }

    it->second.isPaused = true;
    it->second.state = ConnectionState::CONNECTED;
    Logger::info("HQPlayerNAAClient: Paused stream {}", streamId);
    return true;
}

bool HQPlayerNAAClient::stopStream(const std::string& streamId) {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    auto it = streams_.find(streamId);
    if (it == streams_.end()) {
        return false;
    }

    NAAStream& stream = it->second;
    stream.isActive = false;
    stream.isPaused = false;
    stream.state = ConnectionState::CONNECTED;
    stream.endTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    // Cleanup stream buffer
    cleanupStreamBuffer(stream);

    // Update statistics
    if (stream.position > 0) {
        std::lock_guard<std::mutex> statsLock(statsMutex_);
        // Update per-stream statistics as needed
    }

    if (streamStoppedCallback_) {
        streamStoppedCallback_(streamId, stream);
    }

    Logger::info("HQPlayerNAAClient: Stopped stream {}", streamId);
    return true;
}

bool HQPlayerNAAClient::seekStream(const std::string& streamId, uint64_t position) {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    auto it = streams_.find(streamId);
    if (it == streams_.end()) {
        return false;
    }

    it->second.position = position;
    Logger::info("HQPlayerNAAClient: Stream {} seeked to position {}", streamId, position);
    return true;
}

HQPlayerNAAClient::NAAStream HQPlayerNAAClient::getStream(const std::string& streamId) const {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    auto it = streams_.find(streamId);
    return it != streams_.end() ? it->second : NAAStream{};
}

std::vector<HQPlayerNAAClient::NAAStream> HQPlayerNAAClient::getActiveStreams() const {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    std::vector<NAAStream> activeStreams;

    for (const auto& [streamId, stream] : streams_) {
        if (stream.isActive) {
            activeStreams.push_back(stream);
        }
    }

    return activeStreams;
}

bool HQPlayerNAAClient::receiveAudioPacket(const std::string& streamId, const std::vector<uint8_t>& packet) {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    auto it = streams_.find(streamId);
    if (it == streams_.end() || !it->second.isActive) {
        return false;
    }

    NAAStream& stream = it->second;

    // Write to stream buffer
    if (!writeToStreamBuffer(stream, packet)) {
        stream.overruns++;
        Logger::warn("HQPlayerNAAClient: Buffer overflow for stream {}", streamId);
        return false;
    }

    stream.packetsReceived++;
    stream.bytesReceived += packet.size();
    stream.lastPacketTime = std::chrono::steady_clock::now();
    stream.nextPacketTime += calculatePacketSize(stream.sampleRate, stream.channels, stream.bitDepth);

    // Update statistics
    {
        std::lock_guard<std::mutex> statsLock(statsMutex_);
        statistics_.totalPacketsReceived++;
        statistics_.totalBytesReceived += packet.size();
        statistics_.packetsByStream[streamId]++;
        statistics_.lastActivity = stream.lastPacketTime;
    }

    return true;
}

bool HQPlayerNAAClient::processAudioBuffer(const std::string& streamId, std::vector<float>& audioBuffer) {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    auto it = streams_.find(streamId);
    if (it == streams_.end() || !it->second.isActive || it->second.isPaused) {
        return false;
    }

    NAAStream& stream = it->second;

    // Read from stream buffer
    std::vector<uint8_t> rawData;
    size_t bytesToRead = stream.packetSize;
    if (!readFromStreamBuffer(stream, rawData, bytesToRead)) {
        stream.underruns++;
        return false;
    }

    // Convert to float samples
    if (!convertAudioFormat(rawData, audioBuffer, stream.format, stream.channels)) {
        return false;
    }

    // Apply processing if needed
    if (stream.upsamplingFilter != UpsamplingFilter::NONE && stream.targetSampleRate > 0) {
        std::vector<float> upsampledAudio;
        if (applyUpsampling(audioBuffer, upsampledAudio, stream.sampleRate, stream.targetSampleRate, stream.upsamplingFilter)) {
            audioBuffer = std::move(upsampledAudio);
        }
    }

    if (stream.modulationType != ModulationType::NONE) {
        // Apply delta-sigma modulation for DSD conversion
        // This would be implemented based on the modulation type
    }

    if (stream.enableDithering) {
        applyDithering(audioBuffer, stream.ditheringAlgorithm);
    }

    return true;
}

void HQPlayerNAAClient::configureUpsampling(const std::string& streamId, UpsamplingFilter filter, uint32_t targetRate) {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    auto it = streams_.find(streamId);
    if (it != streams_.end()) {
        it->second.upsamplingFilter = filter;
        it->second.targetSampleRate = targetRate;
        Logger::info("HQPlayerNAAClient: Configured upsampling for stream {} with filter {} to {} Hz",
                     streamId, getUpsamplingFilterName(filter), targetRate);
    }
}

void HQPlayerNAAClient::setDithering(const std::string& streamId, bool enable, const std::string& algorithm) {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    auto it = streams_.find(streamId);
    if (it != streams_.end()) {
        it->second.enableDithering = enable;
        it->second.ditheringAlgorithm = algorithm;
        Logger::info("HQPlayerNAAClient: Set dithering for stream {} to {} ({})",
                     streamId, enable ? "enabled" : "disabled", algorithm);
    }
}

HQPlayerNAAClient::ClientStatistics HQPlayerNAAClient::getStatistics() const {
    std::lock_guard<std::mutex> lock(statsMutex_);

    ClientStatistics stats = statistics_;
    stats.connectionState = connectionState_.load();

    // Calculate derived statistics
    {
        std::lock_guard<std::mutex> streamLock(streamsMutex_);
        stats.activeStreams = 0;
        for (const auto& [streamId, stream] : streams_) {
            if (stream.isActive) {
                stats.activeStreams++;
            }
        }
    }

    if (statistics_.totalPacketsReceived > 0) {
        stats.packetLossRate = static_cast<float>(statistics_.totalPacketsLost) /
                               static_cast<float>(statistics_.totalPacketsReceived);
    }

    return stats;
}

void HQPlayerNAAClient::resetStatistics() {
    std::lock_guard<std::mutex> lock(statsMutex_);
    statistics_ = ClientStatistics{};
    statistics_.connectionState = connectionState_.load();
    statistics_.connectionTime = std::chrono::steady_clock::now();
    statistics_.lastActivity = statistics_.connectionTime;

    // Reset stream statistics
    {
        std::lock_guard<std::mutex> streamLock(streamsMutex_);
        for (auto& [streamId, stream] : streams_) {
            stream.bytesReceived = 0;
            stream.packetsReceived = 0;
            stream.packetsLost = 0;
            stream.underruns = 0;
            stream.overruns = 0;
        }
    }
}

HQPlayerNAAClient::HealthStatus HQPlayerNAAClient::getHealthStatus() const {
    HealthStatus status;

    status.isConnected = isConnected();
    status.connectionState = connectionState_.load();
    status.reconnectAttempts = reconnectAttempts_.load();

    {
        std::lock_guard<std::mutex> lock(streamsMutex_);
        status.hasActiveStreams = false;
        for (const auto& [streamId, stream] : streams_) {
            if (stream.isActive) {
                status.hasActiveStreams = true;
                if (stream.underruns > 10 || stream.overruns > 10) {
                    status.warnings.push_back("Stream " + streamId + " has excessive underruns/overruns");
                }
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        status.packetLossWithinLimits = (statistics_.totalPacketsLost < 10);
        status.latencyWithinLimits = (statistics_.averageLatency < config_.targetLatency * 2);
    }

    status.isHealthy = status.isConnected &&
                      (!status.hasActiveStreams ||
                       (status.packetLossWithinLimits && status.latencyWithinLimits));

    return status;
}

void HQPlayerNAAClient::updateConfiguration(const NAAConfig& config) {
    if (!validateConfiguration(config)) {
        setError("Invalid configuration");
        return;
    }

    config_ = config;
    Logger::info("HQPlayerNAAClient: Configuration updated");
}

HQPlayerNAAClient::NAAConfig HQPlayerNAAClient::getConfiguration() const {
    return config_;
}

void HQPlayerNAAClient::setConnectionStateChangedCallback(ConnectionStateCallback callback) {
    connectionStateChangedCallback_ = callback;
}

void HQPlayerNAAClient::setStreamStartedCallback(StreamEventCallback callback) {
    streamStartedCallback_ = callback;
}

void HQPlayerNAAClient::setStreamStoppedCallback(StreamEventCallback callback) {
    streamStoppedCallback_ = callback;
}

void HQPlayerNAAClient::setAudioReceivedCallback(AudioEventCallback callback) {
    audioReceivedCallback_ = callback;
}

void HQPlayerNAAClient::setErrorCallback(ErrorCallback callback) {
    errorCallback_ = callback;
}

// Private methods implementation
bool HQPlayerNAAClient::handleHandshake() {
    Logger::info("HQPlayerNAAClient: Performing handshake with HQPlayer server");

    // Send handshake message
    std::map<std::string, std::string> params;
    params["client_type"] = "NAA";
    params["client_version"] = "1.0.0";
    params["device_id"] = deviceInfo_.deviceId;
    params["device_name"] = deviceInfo_.deviceName;

    auto handshakeMsg = createControlMessage("HANDSHAKE", params);
    if (!sendControlMessage(handshakeMsg)) {
        return false;
    }

    // Wait for handshake response
    auto response = transport_->receiveMessage(config_.connectionTimeout);
    if (response.empty()) {
        setError("No handshake response from server");
        return false;
    }

    // Process handshake response
    return processControlMessage(response);
}

bool HQPlayerNAAClient::handleDeviceInfoRequest() {
    std::map<std::string, std::string> params;
    params["device_id"] = deviceInfo_.deviceId;
    params["device_name"] = deviceInfo_.deviceName;
    params["manufacturer"] = deviceInfo_.manufacturer;
    params["model"] = deviceInfo_.model;
    params["version"] = deviceInfo_.version;

    // Add capabilities
    std::ostringstream sampleRates;
    for (size_t i = 0; i < deviceInfo_.supportedSampleRates.size(); ++i) {
        if (i > 0) sampleRates << ",";
        sampleRates << deviceInfo_.supportedSampleRates[i];
    }
    params["sample_rates"] = sampleRates.str();

    std::ostringstream bitDepths;
    for (size_t i = 0; i < deviceInfo_.supportedBitDepths.size(); ++i) {
        if (i > 0) bitDepths << ",";
        bitDepths << deviceInfo_.supportedBitDepths[i];
    }
    params["bit_depths"] = bitDepths.str();

    params["supports_dsd"] = deviceInfo_.supportsDSD ? "true" : "false";
    params["supports_upsampling"] = deviceInfo_.supportsUpsampling ? "true" : "false";

    auto msg = createControlMessage("DEVICE_INFO", params);
    return sendControlMessage(msg);
}

void HQPlayerNAAClient::connectionThread() {
    Logger::info("HQPlayerNAAClient: Connection monitoring thread started");

    while (connectionState_.load() != ConnectionState::DISCONNECTED &&
           connectionState_.load() != ConnectionState::ERROR) {
        try {
            monitorConnection();
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        } catch (const std::exception& e) {
            Logger::error("HQPlayerNAAClient: Connection monitoring error: {}", e.what());
            handleConnectionError(e.what());
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    Logger::info("HQPlayerNAAClient: Connection monitoring thread stopped");
}

void HQPlayerNAAClient::monitorConnection() {
    if (!transport_->isConnected()) {
        handleConnectionError("Transport disconnected");
        return;
    }

    // Check for incoming control messages
    auto message = transport_->receiveMessage(100); // 100ms timeout
    if (!message.empty()) {
        processControlMessage(message);
    }

    // Cleanup inactive streams
    cleanupInactiveStreams();

    // Update connection state if streaming
    {
        std::lock_guard<std::mutex> lock(streamsMutex_);
        bool hasActiveStreams = false;
        for (const auto& [streamId, stream] : streams_) {
            if (stream.isActive) {
                hasActiveStreams = true;
                break;
            }
        }

        if (hasActiveStreams && connectionState_.load() == ConnectionState::CONNECTED) {
            setConnectionState(ConnectionState::STREAMING);
        } else if (!hasActiveStreams && connectionState_.load() == ConnectionState::STREAMING) {
            setConnectionState(ConnectionState::CONNECTED);
        }
    }
}

void HQPlayerNAAClient::audioProcessingThread() {
    Logger::info("HQPlayerNAAClient: Audio processing thread started");

    while (processingActive_.load()) {
        std::unique_lock<std::mutex> lock(processingQueueMutex_);
        processingCondition_.wait(lock, [this] {
            return !processingQueue_.empty() || !processingActive_.load();
        });

        if (!processingActive_.load()) {
            break;
        }

        while (!processingQueue_.empty()) {
            std::string streamId = processingQueue_.front();
            processingQueue_.pop();
            lock.unlock();

            processStreamAudio(streamId);

            lock.lock();
        }
    }

    Logger::info("HQPlayerNAAClient: Audio processing thread stopped");
}

void HQPlayerNAAClient::processStreamAudio(const std::string& streamId) {
    std::vector<float> audioBuffer;
    if (processAudioBuffer(streamId, audioBuffer) && !audioBuffer.empty()) {
        if (audioReceivedCallback_) {
            audioReceivedCallback_(streamId, audioBuffer);
        }
    }
}

std::string HQPlayerNAAClient::generateStreamId() {
    static std::atomic<uint32_t> counter{0};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);

    return "naa_stream_" + std::to_string(counter++) + "_" + std::to_string(dis(gen));
}

bool HQPlayerNAAClient::validateConfiguration(const NAAConfig& config) const {
    if (config.deviceName.empty()) {
        return false;
    }

    if (config.serverPort == 0 || config.serverPort > 65535) {
        return false;
    }

    if (config.maxSampleRate == 0 || config.maxSampleRate > 1536000) {
        return false;
    }

    if (config.maxChannels == 0 || config.maxChannels > 8) {
        return false;
    }

    return true;
}

bool HQPlayerNAAClient::validateDeviceInfo(const NAADeviceInfo& device) const {
    if (device.deviceId.empty() || device.deviceName.empty()) {
        return false;
    }

    if (device.supportedSampleRates.empty()) {
        return false;
    }

    return true;
}

std::vector<uint8_t> HQPlayerNAAClient::createControlMessage(const std::string& command,
                                                              const std::map<std::string, std::string>& parameters) {
    std::ostringstream oss;
    oss << command;

    for (const auto& [key, value] : parameters) {
        oss << "\n" << key << "=" << value;
    }

    std::string msgStr = oss.str();
    return std::vector<uint8_t>(msgStr.begin(), msgStr.end());
}

bool HQPlayerNAAClient::sendControlMessage(const std::vector<uint8_t>& message) {
    return transport_->sendControlMessage(message);
}

bool HQPlayerNAAClient::processControlMessage(const std::vector<uint8_t>& message) {
    std::string msgStr(message.begin(), message.end());
    std::istringstream iss(msgStr);
    std::string command;
    std::getline(iss, command);

    std::map<std::string, std::string> params;
    std::string line;
    while (std::getline(iss, line)) {
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            params[key] = value;
        }
    }

    // Process command
    if (command == "STREAM_START") {
        // Handle stream start
        return true;
    } else if (command == "STREAM_STOP") {
        // Handle stream stop
        return true;
    } else if (command == "SET_VOLUME") {
        auto it = params.find("volume");
        if (it != params.end()) {
            float volume = std::stof(it->second);
            handleVolumeControl(volume);
        }
        return true;
    } else if (command == "SET_MUTE") {
        auto it = params.find("mute");
        if (it != params.end()) {
            bool muted = (it->second == "true" || it->second == "1");
            handleMuteControl(muted);
        }
        return true;
    } else if (command == "SET_SAMPLE_RATE") {
        auto it = params.find("rate");
        if (it != params.end()) {
            uint32_t rate = std::stoul(it->second);
            handleSampleRateChange(rate);
        }
        return true;
    }

    Logger::info("HQPlayerNAAClient: Received control command: {}", command);
    return true;
}

bool HQPlayerNAAClient::handleVolumeControl(float volume) {
    Logger::info("HQPlayerNAAClient: Volume control set to {:.2f}", volume);
    return true;
}

bool HQPlayerNAAClient::handleMuteControl(bool muted) {
    Logger::info("HQPlayerNAAClient: Mute control set to {}", muted ? "on" : "off");
    return true;
}

bool HQPlayerNAAClient::handleSampleRateChange(uint32_t newRate) {
    Logger::info("HQPlayerNAAClient: Sample rate changed to {} Hz", newRate);
    return true;
}

void HQPlayerNAAClient::handleConnectionError(const std::string& error) {
    setError("Connection error: " + error);

    if (config_.enableAutoReconnect && reconnectAttempts_.load() < config_.maxReconnectAttempts) {
        setConnectionState(ConnectionState::RECONNECTING);
        attemptReconnection();
    } else {
        setConnectionState(ConnectionState::ERROR);
    }
}

void HQPlayerNAAClient::attemptReconnection() {
    reconnectAttempts_++;

    Logger::info("HQPlayerNAAClient: Attempting reconnection ({}/{})",
                 reconnectAttempts_.load(), config_.maxReconnectAttempts);

    // Disconnect and reconnect
    transport_->disconnect();
    std::this_thread::sleep_for(std::chrono::milliseconds(config_.reconnectInterval));

    if (transport_->connect(config_.serverAddress) && handleHandshake()) {
        setConnectionState(ConnectionState::CONNECTED);
        reconnectAttempts_.store(0);
        Logger::info("HQPlayerNAAClient: Reconnection successful");
    } else if (reconnectAttempts_.load() < config_.maxReconnectAttempts) {
        // Schedule another reconnection attempt
        std::thread([this]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.reconnectInterval));
            attemptReconnection();
        }).detach();
    } else {
        setConnectionState(ConnectionState::ERROR);
        Logger::error("HQPlayerNAAClient: Maximum reconnection attempts reached");
    }
}

void HQPlayerNAAClient::setConnectionState(ConnectionState state) {
    ConnectionState oldState = connectionState_.exchange(state);
    if (oldState != state) {
        Logger::info("HQPlayerNAAClient: Connection state changed from {} to {}",
                     getConnectionStateName(oldState), getConnectionStateName(state));

        if (connectionStateChangedCallback_) {
            connectionStateChangedCallback_(state);
        }
    }
}

bool HQPlayerNAAClient::initializeStreamBuffer(NAAStream& stream) {
    stream.bufferCapacity = config_.bufferSize;
    stream.audioBuffer.reserve(stream.bufferCapacity);
    stream.bufferUsed = 0;
    return true;
}

void HQPlayerNAAClient::cleanupStreamBuffer(NAAStream& stream) {
    stream.audioBuffer.clear();
    stream.bufferUsed = 0;
}

bool HQPlayerNAAClient::writeToStreamBuffer(NAAStream& stream, const std::vector<uint8_t>& data) {
    size_t spaceAvailable = stream.bufferCapacity - stream.bufferUsed;
    if (data.size() > spaceAvailable) {
        return false; // Buffer full
    }

    stream.audioBuffer.insert(stream.audioBuffer.end(), data.begin(), data.end());
    stream.bufferUsed += data.size();
    return true;
}

bool HQPlayerNAAClient::readFromStreamBuffer(NAAStream& stream, std::vector<uint8_t>& data, size_t bytes) {
    if (stream.bufferUsed < bytes) {
        return false; // Not enough data
    }

    data.assign(stream.audioBuffer.begin(), stream.audioBuffer.begin() + bytes);
    stream.audioBuffer.erase(stream.audioBuffer.begin(), stream.audioBuffer.begin() + bytes);
    stream.bufferUsed -= bytes;
    return true;
}

void HQPlayerNAAClient::cleanupInactiveStreams() {
    std::lock_guard<std::mutex> lock(streamsMutex_);
    auto now = std::chrono::steady_clock::now();

    for (auto it = streams_.begin(); it != streams_.end();) {
        const NAAStream& stream = it->second;
        if (!stream.isActive) {
            auto timeSinceCreation = std::chrono::duration_cast<std::chrono::seconds>(now - stream.creationTime);
            if (timeSinceCreation.count() > 300) { // 5 minutes
                Logger::info("HQPlayerNAAClient: Cleaning up inactive stream {}", stream.streamId);
                cleanupStreamBuffer(const_cast<NAAStream&>(stream));
                it = streams_.erase(it);
            } else {
                ++it;
            }
        } else {
            ++it;
        }
    }
}

bool HQPlayerNAAClient::convertAudioFormat(const std::vector<uint8_t>& input, std::vector<float>& output,
                                           AudioFormat format, uint16_t channels) const {
    // Convert audio data based on format and channels
    uint32_t bitDepth = 24; // Default
    size_t samplesPerChannel = input.size() / (bitDepth / 8 * channels);

    output.clear();
    output.reserve(samplesPerChannel * channels);

    switch (format) {
        case AudioFormat::PCM_S16LE: {
            const int16_t* samples = reinterpret_cast<const int16_t*>(input.data());
            float scale = 1.0f / 32768.0f;
            for (size_t i = 0; i < samplesPerChannel * channels; ++i) {
                output.push_back(samples[i] * scale);
            }
            break;
        }
        case AudioFormat::PCM_S24LE: {
            const uint8_t* samples = input.data();
            float scale = 1.0f / 8388608.0f;
            for (size_t i = 0; i < samplesPerChannel * channels; ++i) {
                int32_t sample = (samples[i*3] << 8) | (samples[i*3+1] << 16) | (samples[i*3+2] << 24);
                output.push_back((sample >> 8) * scale);
            }
            break;
        }
        case AudioFormat::PCM_S32LE: {
            const int32_t* samples = reinterpret_cast<const int32_t*>(input.data());
            float scale = 1.0f / 2147483648.0f;
            for (size_t i = 0; i < samplesPerChannel * channels; ++i) {
                output.push_back(samples[i] * scale);
            }
            break;
        }
        default:
            return false;
    }

    return true;
}

uint32_t HQPlayerNAAClient::calculatePacketSize(uint32_t sampleRate, uint16_t channels, uint16_t bitDepth) const {
    // Calculate packet size for 10ms of audio
    uint32_t samplesPerPacket = sampleRate / 100;
    uint32_t bytesPerSample = bitDepth / 8;
    return samplesPerPacket * channels * bytesPerSample;
}

void HQPlayerNAAClient::setError(const std::string& error) {
    lastError_ = error;
    Logger::error("HQPlayerNAAClient: {}", error);

    if (errorCallback_) {
        errorCallback_("client", error);
    }
}

std::string HQPlayerNAAClient::getConnectionStateName(ConnectionState state) const {
    switch (state) {
        case ConnectionState::DISCONNECTED: return "Disconnected";
        case ConnectionState::CONNECTING: return "Connecting";
        case ConnectionState::CONNECTED: return "Connected";
        case ConnectionState::STREAMING: return "Streaming";
        case ConnectionState::ERROR: return "Error";
        case ConnectionState::RECONNECTING: return "Reconnecting";
        default: return "Unknown";
    }
}

std::string HQPlayerNAAClient::getTransportModeName(TransportMode mode) const {
    switch (mode) {
        case TransportMode::TCP: return "TCP";
        case TransportMode::UDP: return "UDP";
        case TransportMode::RTP: return "RTP";
        case TransportMode::WEBSOCKET: return "WebSocket";
        default: return "Unknown";
    }
}

std::string HQPlayerNAAClient::getUpsamplingFilterName(UpsamplingFilter filter) const {
    switch (filter) {
        case UpsamplingFilter::NONE: return "None";
        case UpsamplingFilter::POLY_SINC: return "Poly-sinc";
        case UpsamplingFilter::POLY_SINC_XTR: return "Poly-sinc Xtra";
        case UpsamplingFilter::POLY_SINC_MP: return "Poly-sinc Multi";
        case UpsamplingFilter::SINC_M: return "Sinc-M";
        case UpsamplingFilter::SINC_L: return "Sinc-L";
        case UpsamplingFilter::MIN_PHASE: return "Minimum Phase";
        case UpsamplingFilter::LINEAR_PHASE: return "Linear Phase";
        case UpsamplingFilter::FIR_DITHER: return "FIR Dither";
        default: return "Unknown";
    }
}

// TCPTransport Implementation
HQPlayerNAAClient::TCPTransport::TCPTransport() {
#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
}

HQPlayerNAAClient::TCPTransport::~TCPTransport() {
    disconnect();
#ifdef _WIN32
    WSACleanup();
#endif
}

bool HQPlayerNAAClient::TCPTransport::connect(const std::pair<std::string, std::string>& address) {
    remoteHost_ = address.first;
    remotePort_ = static_cast<uint16_t>(std::stoul(address.second));

    socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_ < 0) {
        return false;
    }

    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(remotePort_);

    if (inet_pton(AF_INET, remoteHost_.c_str(), &serverAddr.sin_addr) <= 0) {
        // Resolve hostname
        struct hostent* host = gethostbyname(remoteHost_.c_str());
        if (host == nullptr) {
            return false;
        }
        memcpy(&serverAddr.sin_addr, host->h_addr, host->h_length);
    }

    if (::connect(socket_, reinterpret_cast<sockaddr*>(&serverAddr), sizeof(serverAddr)) < 0) {
        return false;
    }

    connected_.store(true);
    return true;
}

void HQPlayerNAAClient::TCPTransport::disconnect() {
    connected_.store(false);

    if (socket_ >= 0) {
#ifdef _WIN32
        closesocket(socket_);
#else
        close(socket_);
#endif
        socket_ = -1;
    }
}

bool HQPlayerNAAClient::TCPTransport::isConnected() const {
    return connected_.load() && socket_ >= 0;
}

bool HQPlayerNAAClient::TCPTransport::sendControlMessage(const std::vector<uint8_t>& message) {
    if (!isConnected()) {
        return false;
    }

    // Add length header
    uint32_t length = static_cast<uint32_t>(message.size());
    std::vector<uint8_t> packet(4 + message.size());
    packet[0] = (length >> 24) & 0xFF;
    packet[1] = (length >> 16) & 0xFF;
    packet[2] = (length >> 8) & 0xFF;
    packet[3] = length & 0xFF;
    std::copy(message.begin(), message.end(), packet.begin() + 4);

    return sendDataPacket(packet);
}

bool HQPlayerNAAClient::TCPTransport::sendDataPacket(const std::vector<uint8_t>& data) {
    if (!isConnected()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(socketMutex_);

    size_t totalSent = 0;
    while (totalSent < data.size()) {
        ssize_t sent = send(socket_, reinterpret_cast<const char*>(data.data() + totalSent),
                           data.size() - totalSent, 0);
        if (sent < 0) {
            connected_.store(false);
            return false;
        }
        totalSent += sent;
    }

    return true;
}

std::vector<uint8_t> HQPlayerNAAClient::TCPTransport::receiveMessage(uint32_t timeoutMs) {
    if (!isConnected()) {
        return {};
    }

    // Read length header
    uint8_t lengthBuffer[4];
    ssize_t received = recv(socket_, reinterpret_cast<char*>(lengthBuffer), 4, 0);
    if (received != 4) {
        if (received <= 0) {
            connected_.store(false);
        }
        return {};
    }

    uint32_t length = (lengthBuffer[0] << 24) | (lengthBuffer[1] << 16) | (lengthBuffer[2] << 8) | lengthBuffer[3];
    if (length > 1024 * 1024) { // 1MB limit
        return {};
    }

    // Read message body
    std::vector<uint8_t> message(length);
    size_t totalReceived = 0;
    while (totalReceived < length) {
        received = recv(socket_, reinterpret_cast<char*>(message.data() + totalReceived),
                       length - totalReceived, 0);
        if (received <= 0) {
            if (received < 0) {
                connected_.store(false);
            }
            return {};
        }
        totalReceived += received;
    }

    return message;
}

uint32_t HQPlayerNAAClient::TCPTransport::getLatency() const {
    return 10; // Typical TCP latency in ms
}

// UDPTransport Implementation
HQPlayerNAAClient::UDPTransport::UDPTransport() {
#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
}

HQPlayerNAAClient::UDPTransport::~UDPTransport() {
    disconnect();
#ifdef _WIN32
    WSACleanup();
#endif
}

bool HQPlayerNAAClient::UDPTransport::connect(const std::pair<std::string, std::string>& address) {
    remoteHost_ = address.first;
    remotePort_ = static_cast<uint16_t>(std::stoul(address.second));

    socket_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_ < 0) {
        return false;
    }

    memset(&remoteAddr_, 0, sizeof(remoteAddr_));
    remoteAddr_.sin_family = AF_INET;
    remoteAddr_.sin_port = htons(remotePort_);

    if (inet_pton(AF_INET, remoteHost_.c_str(), &remoteAddr_.sin_addr) <= 0) {
        struct hostent* host = gethostbyname(remoteHost_.c_str());
        if (host == nullptr) {
            return false;
        }
        memcpy(&remoteAddr_.sin_addr, host->h_addr, host->h_length);
    }

    connected_.store(true);
    return true;
}

void HQPlayerNAAClient::UDPTransport::disconnect() {
    connected_.store(false);

    if (socket_ >= 0) {
#ifdef _WIN32
        closesocket(socket_);
#else
        close(socket_);
#endif
        socket_ = -1;
    }
}

bool HQPlayerNAAClient::UDPTransport::isConnected() const {
    return connected_.load() && socket_ >= 0;
}

bool HQPlayerNAAClient::UDPTransport::sendControlMessage(const std::vector<uint8_t>& message) {
    return sendDataPacket(message);
}

bool HQPlayerNAAClient::UDPTransport::sendDataPacket(const std::vector<uint8_t>& data) {
    if (!isConnected()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(socketMutex_);

    ssize_t sent = sendto(socket_, reinterpret_cast<const char*>(data.data()), data.size(), 0,
                         reinterpret_cast<sockaddr*>(&remoteAddr_), sizeof(remoteAddr_));
    return sent == static_cast<ssize_t>(data.size());
}

std::vector<uint8_t> HQPlayerNAAClient::UDPTransport::receiveMessage(uint32_t timeoutMs) {
    if (!isConnected()) {
        return {};
    }

    // Set receive timeout
#ifdef _WIN32
    DWORD timeout = timeoutMs;
    setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout), sizeof(timeout));
#else
    struct timeval tv{};
    tv.tv_sec = timeoutMs / 1000;
    tv.tv_usec = (timeoutMs % 1000) * 1000;
    setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif

    std::vector<uint8_t> buffer(4096);
    sockaddr_in fromAddr{};
    socklen_t fromAddrLen = sizeof(fromAddr);

    ssize_t received = recvfrom(socket_, reinterpret_cast<char*>(buffer.data()), buffer.size(), 0,
                               reinterpret_cast<sockaddr*>(&fromAddr_), &fromAddrLen);

    if (received < 0) {
#ifdef _WIN32
        if (WSAGetLastError() != WSAETIMEDOUT) {
            connected_.store(false);
        }
#else
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            connected_.store(false);
        }
#endif
        return {};
    }

    buffer.resize(received);
    return buffer;
}

uint32_t HQPlayerNAAClient::UDPTransport::getLatency() const {
    return 5; // Typical UDP latency in ms
}

} // namespace vortex