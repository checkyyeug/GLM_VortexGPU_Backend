#include "network/audio_streaming_protocol.hpp"
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <algorithm>

namespace VortexGPU {
namespace Network {

// ============================================================================
// NetworkAudioStreamer Implementation
// ============================================================================

NetworkAudioStreamer::NetworkAudioStreamer()
    : config_()
    , initialized_(false)
    , next_session_id_(1)
    , global_stats_() {

    generate_session_token();
}

NetworkAudioStreamer::~NetworkAudioStreamer() {
    shutdown();
}

bool NetworkAudioStreamer::initialize(const AudioStreamConfig& config) {
    std::lock_guard<std::mutex> lock(streamer_mutex_);

    if (initialized_) {
        return true; // Already initialized
    }

    config_ = config;

    try {
        // Initialize protocol handlers
        for (auto protocol : config_.enabled_protocols) {
            switch (protocol) {
                case StreamingProtocol::RTP:
                    rtp_handler_.initialize(config_);
                    break;
                case StreamingProtocol::WebRTC:
                    webrtc_handler_.initialize(config_);
                    break;
                case StreamingProtocol::RTSP:
                    rtsp_handler_.initialize(config_);
                    break;
                case StreamingProtocol::SRT:
                    srt_handler_.initialize(config_);
                    break;
                case StreamingProtocol::NDI:
                    ndi_handler_.initialize(config_);
                    break;
                case StreamingProtocol::Dante:
                    dante_handler_.initialize(config_);
                    break;
                case StreamingProtocol::RAVENNA:
                    raven_handler_.initialize(config_);
                    break;
                case StreamingProtocol::AES67:
                    aes67_handler_.initialize(config_);
                    break;
                case StreamingProtocol::Custom:
                    custom_handler_.initialize(config_);
                    break;
            }
        }

        // Start background threads
        running_ = true;
        stats_thread_ = std::thread(&NetworkAudioStreamer::statisticsThread, this);
        cleanup_thread_ = std::thread(&NetworkAudioStreamer::cleanupThread, this);

        initialized_ = true;

        std::cout << "NetworkAudioStreamer initialized with protocols: ";
        for (auto protocol : config_.enabled_protocols) {
            std::cout << static_cast<int>(protocol) << " ";
        }
        std::cout << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize NetworkAudioStreamer: " << e.what() << std::endl;
        return false;
    }
}

void NetworkAudioStreamer::shutdown() {
    std::lock_guard<std::mutex> lock(streamer_mutex_);

    if (!initialized_) {
        return;
    }

    running_ = false;

    // Stop all sessions
    for (auto& session_pair : active_sessions_) {
        stopSessionInternal(session_pair.first);
    }

    // Wait for threads to finish
    if (stats_thread_.joinable()) {
        stats_thread_.join();
    }

    if (cleanup_thread_.joinable()) {
        cleanup_thread_.join();
    }

    active_sessions_.clear();
    initialized_ = false;

    std::cout << "NetworkAudioStreamer shut down" << std::endl;
}

uint32_t NetworkAudioStreamer::startSession(const std::string& session_name, bool is_sender) {
    if (!initialized_) {
        return 0;
    }

    std::lock_guard<std::mutex> lock(sessions_mutex_);

    uint32_t session_id = next_session_id_++;

    AudioStreamSession session;
    session.session_id = session_id;
    session.session_name = session_name;
    session.is_sender = is_sender;
    session.active = true;
    session.quality_level = StreamQuality::HIGH;

    // Select primary protocol
    if (!config_.enabled_protocols.empty()) {
        session.primary_protocol = config_.enabled_protocols[0];
    } else {
        session.primary_protocol = StreamingProtocol::RTP;
    }

    // Initialize quality manager
    session.quality_manager = std::make_unique<AudioStreamQualityManager>(config_, session.primary_protocol);

    // Initialize protocol-specific handlers
    initializeProtocolHandlers(session);

    // Setup initial configuration
    session.buffer_size = config_.buffer_size;
    session.sample_rate = config_.sample_rate;
    session.channels = config_.channels;
    session.bitrate = 64000; // Default bitrate

    active_sessions_[session_id] = std::move(session);

    std::cout << "Started audio streaming session: " << session_name
              << " (ID: " << session_id << ", " << (is_sender ? "Sender" : "Receiver") << ")" << std::endl;

    return session_id;
}

bool NetworkAudioStreamer::stopSession(uint32_t session_id) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    return stopSessionInternal(session_id);
}

bool NetworkAudioStreamer::stopSessionInternal(uint32_t session_id) {
    auto session_it = active_sessions_.find(session_id);
    if (session_it == active_sessions_.end()) {
        return false;
    }

    AudioStreamSession& session = session_it->second;
    session.active = false;

    // Close protocol-specific connections
    if (session.rtp_session) {
        rtp_handler_.closeSession(session.rtp_session.get());
    }

    if (session.webrtc_session) {
        webrtc_handler_.closeSession(session.webrtc_session.get());
    }

    if (session.srt_socket) {
        srt_handler_.closeSocket(session.srt_socket.get());
    }

    // Update global stats
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    global_stats_.total_sessions_sent++;

    std::cout << "Stopped audio streaming session: " << session.session_name
              << " (ID: " << session_id << ")" << std::endl;

    return true;
}

bool NetworkAudioStreamer::sendAudioData(uint32_t session_id, const float* audio_data,
                                        size_t num_samples, uint32_t timestamp) {
    if (!audio_data || num_samples == 0) {
        return false;
    }

    std::shared_lock<std::shared_mutex> lock(sessions_mutex_);

    auto session_it = active_sessions_.find(session_id);
    if (session_it == active_sessions_.end() || !session_it->second.active || !session_it->second.is_sender) {
        return false;
    }

    AudioStreamSession& session = session_it->second;

    // Apply quality management
    if (session.quality_manager) {
        session.quality_manager->updateQuality(session.stats);
    }

    // Encode audio data if needed
    std::vector<uint8_t> encoded_data;
    const uint8_t* data_to_send = reinterpret_cast<const uint8_t*>(audio_data);
    size_t data_size = num_samples * sizeof(float);

    if (config_.codec != AudioCodec::PCM) {
        if (!encodeAudioData(audio_data, num_samples, config_.codec, session.bitrate, encoded_data)) {
            return false;
        }
        data_to_send = encoded_data.data();
        data_size = encoded_data.size();
    }

    // Send via appropriate protocol
    bool success = false;

    switch (session.primary_protocol) {
        case StreamingProtocol::RTP:
            if (session.rtp_session) {
                success = rtp_handler_.sendPacket(session.rtp_session.get(), data_to_send, data_size, timestamp);
            }
            break;

        case StreamingProtocol::WebRTC:
            if (session.webrtc_session) {
                success = webrtc_handler_.sendData(session.webrtc_session.get(), data_to_send, data_size);
            }
            break;

        case StreamingProtocol::SRT:
            if (session.srt_socket) {
                success = srt_handler_.sendData(session.srt_socket.get(), data_to_send, data_size, 0);
            }
            break;

        default:
            // Fallback to RTP
            if (session.rtp_session) {
                success = rtp_handler_.sendPacket(session.rtp_session.get(), data_to_send, data_size, timestamp);
            }
            break;
    }

    if (success) {
        // Update session statistics
        session.stats.packets_sent++;
        session.stats.bytes_sent += data_size;
        session.stats.last_activity_time = std::chrono::steady_clock::now();
    }

    return success;
}

size_t NetworkAudioStreamer::receiveAudioData(uint32_t session_id, float* audio_buffer, size_t max_samples) {
    if (!audio_buffer || max_samples == 0) {
        return 0;
    }

    std::shared_lock<std::shared_mutex> lock(sessions_mutex_);

    auto session_it = active_sessions_.find(session_id);
    if (session_it == active_sessions_.end() || !session_it->second.active || session_it->second.is_sender) {
        return 0;
    }

    AudioStreamSession& session = session_it->second;

    // Receive data via appropriate protocol
    std::vector<uint8_t> received_data;
    uint32_t timestamp = 0;
    bool data_received = false;

    switch (session.primary_protocol) {
        case StreamingProtocol::RTP:
            if (session.rtp_session) {
                RTPPacket packet;
                if (rtp_handler_.receivePacket(session.rtp_session.get(), packet, 0)) {
                    received_data = packet.payload;
                    timestamp = packet.timestamp;
                    data_received = true;
                }
            }
            break;

        case StreamingProtocol::WebRTC:
            if (session.webrtc_session) {
                WebRTCDataPacket packet;
                if (webrtc_handler_.receiveData(session.webrtc_session.get(), packet, 0)) {
                    received_data = packet.data;
                    data_received = true;
                }
            }
            break;

        case StreamingProtocol::SRT:
            if (session.srt_socket) {
                SRTDataPacket packet;
                if (srt_handler_.receiveData(session.srt_socket.get(), packet, 0)) {
                    received_data = packet.data;
                    data_received = true;
                }
            }
            break;

        default:
            // Try RTP as fallback
            if (session.rtp_session) {
                RTPPacket packet;
                if (rtp_handler_.receivePacket(session.rtp_session.get(), packet, 0)) {
                    received_data = packet.payload;
                    timestamp = packet.timestamp;
                    data_received = true;
                }
            }
            break;
    }

    if (!data_received || received_data.empty()) {
        return 0;
    }

    // Decode audio data if needed
    size_t samples_decoded = 0;

    if (config_.codec == AudioCodec::PCM) {
        // Direct copy from PCM data
        size_t samples_available = received_data.size() / sizeof(float);
        samples_decoded = std::min(samples_available, max_samples);
        std::memcpy(audio_buffer, received_data.data(), samples_decoded * sizeof(float));
    } else {
        // Decode encoded data
        if (!decodeAudioData(received_data.data(), received_data.size(), config_.codec,
                           audio_buffer, max_samples, samples_decoded)) {
            return 0;
        }
    }

    // Update session statistics
    session.stats.packets_received++;
    session.stats.bytes_received += received_data.size();
    session.stats.last_activity_time = std::chrono::steady_clock::now();

    return samples_decoded;
}

bool NetworkAudioStreamer::setQualityLevel(uint32_t session_id, StreamQuality quality) {
    std::shared_lock<std::shared_mutex> lock(sessions_mutex_);

    auto session_it = active_sessions_.find(session_id);
    if (session_it == active_sessions_.end()) {
        return false;
    }

    AudioStreamSession& session = session_it->second;
    session.quality_level = quality;

    if (session.quality_manager) {
        session.quality_manager->forceQuality(quality);
    }

    // Update encoding bitrate based on quality
    switch (quality) {
        case StreamQuality::LOW:
            session.bitrate = 64000;
            break;
        case StreamQuality::MEDIUM:
            session.bitrate = 128000;
            break;
        case StreamQuality::HIGH:
            session.bitrate = 256000;
            break;
        case StreamQuality::LOSSLESS:
            session.bitrate = 1411000; // CD quality
            break;
    }

    return true;
}

AudioStreamStats NetworkAudioStreamer::getSessionStats(uint32_t session_id) const {
    std::shared_lock<std::shared_mutex> lock(sessions_mutex_);

    auto session_it = active_sessions_.find(session_id);
    if (session_it != active_sessions_.end()) {
        return session_it->second.stats;
    }

    return AudioStreamStats{};
}

std::vector<uint32_t> NetworkAudioStreamer::getActiveSessionIds() const {
    std::shared_lock<std::shared_mutex> lock(sessions_mutex_);

    std::vector<uint32_t> session_ids;
    for (const auto& session_pair : active_sessions_) {
        if (session_pair.second.active) {
            session_ids.push_back(session_pair.first);
        }
    }

    return session_ids;
}

std::vector<AudioStreamSession> NetworkAudioStreamer::getActiveSessions() const {
    std::shared_lock<std::shared_mutex> lock(sessions_mutex_);

    std::vector<AudioStreamSession> sessions;
    for (const auto& session_pair : active_sessions_) {
        if (session_pair.second.active) {
            sessions.push_back(session_pair.second);
        }
    }

    return sessions;
}

bool NetworkAudioStreamer::isSessionActive(uint32_t session_id) const {
    std::shared_lock<std::shared_mutex> lock(sessions_mutex_);

    auto session_it = active_sessions_.find(session_id);
    return session_it != active_sessions_.end() && session_it->second.active;
}

std::string NetworkAudioStreamer::getSessionToken() const {
    std::lock_guard<std::mutex> lock(streamer_mutex_);
    return session_token_;
}

NetworkAudioStreamerStats NetworkAudioStreamer::getGlobalStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return global_stats_;
}

bool NetworkAudioStreamer::enableEncryption(uint32_t session_id, EncryptionType encryption) {
    std::shared_lock<std::shared_mutex> lock(sessions_mutex_);

    auto session_it = active_sessions_.find(session_id);
    if (session_it == active_sessions_.end()) {
        return false;
    }

    AudioStreamSession& session = session_it->second;
    session.encryption_type = encryption;

    // Enable encryption for protocol handlers
    switch (session.primary_protocol) {
        case StreamingProtocol::RTP:
            if (session.rtp_session) {
                return rtp_handler_.enableEncryption(session.rtp_session.get(), encryption);
            }
            break;

        case StreamingProtocol::SRT:
            if (session.srt_socket) {
                return srt_handler_.enableEncryption(session.srt_socket.get(), encryption);
            }
            break;

        default:
            // Other protocols handle encryption internally
            break;
    }

    return true;
}

bool NetworkAudioStreamer::enableFEC(uint32_t session_id, FECLevel fec_level) {
    std::shared_lock<std::shared_mutex> lock(sessions_mutex_);

    auto session_it = active_sessions_.find(session_id);
    if (session_it == active_sessions_.end()) {
        return false;
    }

    AudioStreamSession& session = session_it->second;
    session.fec_enabled = true;
    session.fec_level = fec_level;

    // Enable FEC for protocol handlers that support it
    switch (session.primary_protocol) {
        case StreamingProtocol::RTP:
            if (session.rtp_session) {
                return rtp_handler_.enableFEC(session.rtp_session.get(), fec_level);
            }
            break;

        case StreamingProtocol::SRT:
            if (session.srt_socket) {
                return srt_handler_.enableFEC(session.srt_socket.get(), fec_level);
            }
            break;

        default:
            // Other protocols handle FEC internally
            break;
    }

    return true;
}

void NetworkAudioStreamer::generateSessionToken() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);

    std::stringstream ss;
    ss << std::hex;

    for (int i = 0; i < 32; ++i) {
        ss << std::setw(1) << std::setfill('0') << dis(gen);
        if (i == 7 || i == 11 || i == 15 || i == 19) {
            ss << '-';
        }
    }

    session_token_ = ss.str();
}

void NetworkAudioStreamer::initializeProtocolHandlers(AudioStreamSession& session) {
    // Initialize protocol-specific handlers based on enabled protocols
    for (auto protocol : config_.enabled_protocols) {
        switch (protocol) {
            case StreamingProtocol::RTP:
                if (!session.rtp_session) {
                    session.rtp_session = rtp_handler_.createSession();
                    if (session.rtp_session) {
                        rtp_handler_.configureSession(session.rtp_session.get(), session);
                    }
                }
                break;

            case StreamingProtocol::WebRTC:
                if (!session.webrtc_session) {
                    session.webrtc_session = webrtc_handler_.createSession();
                    if (session.webrtc_session) {
                        webrtc_handler_.configureSession(session.webrtc_session.get(), session);
                    }
                }
                break;

            case StreamingProtocol::RTSP:
                if (!session.rtsp_session) {
                    session.rtsp_session = rtsp_handler_.createSession();
                    if (session.rtsp_session) {
                        rtsp_handler_.configureSession(session.rtsp_session.get(), session);
                    }
                }
                break;

            case StreamingProtocol::SRT:
                if (!session.srt_socket) {
                    session.srt_socket = srt_handler_.createSocket();
                    if (session.srt_socket) {
                        srt_handler_.configureSocket(session.srt_socket.get(), session);
                    }
                }
                break;

            case StreamingProtocol::NDI:
                if (!session.ndi_sender && !session.ndi_receiver) {
                    if (session.is_sender) {
                        session.ndi_sender = ndi_handler_.createSender();
                        if (session.ndi_sender) {
                            ndi_handler_.configureSender(session.ndi_sender.get(), session);
                        }
                    } else {
                        session.ndi_receiver = ndi_handler_.createReceiver();
                        if (session.ndi_receiver) {
                            ndi_handler_.configureReceiver(session.ndi_receiver.get(), session);
                        }
                    }
                }
                break;

            case StreamingProtocol::Dante:
                if (!session.dante_interface) {
                    session.dante_interface = dante_handler_.createInterface();
                    if (session.dante_interface) {
                        dante_handler_.configureInterface(session.dante_interface.get(), session);
                    }
                }
                break;

            case StreamingProtocol::RAVENNA:
                if (!session.ravenna_session) {
                    session.ravenna_session = raven_handler_.createSession();
                    if (session.ravenna_session) {
                        raven_handler_.configureSession(session.ravenna_session.get(), session);
                    }
                }
                break;

            case StreamingProtocol::AES67:
                if (!session.aes67_interface) {
                    session.aes67_interface = aes67_handler_.createInterface();
                    if (session.aes67_interface) {
                        aes67_handler_.configureInterface(session.aes67_interface.get(), session);
                    }
                }
                break;

            case StreamingProtocol::Custom:
                if (!session.custom_session) {
                    session.custom_session = custom_handler_.createSession();
                    if (session.custom_session) {
                        custom_handler_.configureSession(session.custom_session.get(), session);
                    }
                }
                break;
        }
    }
}

bool NetworkAudioStreamer::encodeAudioData(const float* input, size_t num_samples,
                                          AudioCodec codec, uint32_t bitrate,
                                          std::vector<uint8_t>& output) {
    // Simplified encoding - in a real implementation, this would use actual codecs
    output.clear();

    switch (codec) {
        case AudioCodec::PCM:
            output.resize(num_samples * sizeof(float));
            std::memcpy(output.data(), input, num_samples * sizeof(float));
            return true;

        case AudioCodec::OPUS:
            // Placeholder for OPUS encoding
            return encodeOpus(input, num_samples, bitrate, output);

        case AudioCodec::AAC:
            // Placeholder for AAC encoding
            return encodeAAC(input, num_samples, bitrate, output);

        case AudioCodec::MP3:
            // Placeholder for MP3 encoding
            return encodeMP3(input, num_samples, bitrate, output);

        case AudioCodec::FLAC:
            // Placeholder for FLAC encoding
            return encodeFLAC(input, num_samples, output);

        case AudioCodec::ALAC:
            // Placeholder for ALAC encoding
            return encodeALAC(input, num_samples, output);

        default:
            return false;
    }
}

bool NetworkAudioStreamer::decodeAudioData(const uint8_t* input, size_t input_size,
                                          AudioCodec codec, float* output, size_t max_samples,
                                          size_t& samples_decoded) {
    samples_decoded = 0;

    switch (codec) {
        case AudioCodec::PCM:
            samples_decoded = std::min(input_size / sizeof(float), max_samples);
            std::memcpy(output, input, samples_decoded * sizeof(float));
            return true;

        case AudioCodec::OPUS:
            // Placeholder for OPUS decoding
            return decodeOpus(input, input_size, output, max_samples, samples_decoded);

        case AudioCodec::AAC:
            // Placeholder for AAC decoding
            return decodeAAC(input, input_size, output, max_samples, samples_decoded);

        case AudioCodec::MP3:
            // Placeholder for MP3 decoding
            return decodeMP3(input, input_size, output, max_samples, samples_decoded);

        case AudioCodec::FLAC:
            // Placeholder for FLAC decoding
            return decodeFLAC(input, input_size, output, max_samples, samples_decoded);

        case AudioCodec::ALAC:
            // Placeholder for ALAC decoding
            return decodeALAC(input, input_size, output, max_samples, samples_decoded);

        default:
            return false;
    }
}

void NetworkAudioStreamer::statisticsThread() {
    auto last_update = std::chrono::steady_clock::now();

    while (running_) {
        try {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update);

            if (elapsed.count() >= 100) { // Update every 100ms
                updateGlobalStatistics();
                last_update = now;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));

        } catch (const std::exception& e) {
            std::cerr << "Statistics thread error: " << e.what() << std::endl;
        }
    }
}

void NetworkAudioStreamer::cleanupThread() {
    auto last_cleanup = std::chrono::steady_clock::now();

    while (running_) {
        try {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_cleanup);

            if (elapsed.count() >= 60) { // Cleanup every minute
                cleanupInactiveSessions();
                last_cleanup = now;
            }

            std::this_thread::sleep_for(std::chrono::seconds(1));

        } catch (const std::exception& e) {
            std::cerr << "Cleanup thread error: " << e.what() << std::endl;
        }
    }
}

void NetworkAudioStreamer::updateGlobalStatistics() {
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);

    std::shared_lock<std::shared_mutex> sessions_lock(sessions_mutex_);

    size_t total_sessions = 0;
    size_t active_senders = 0;
    size_t active_receivers = 0;

    for (const auto& session_pair : active_sessions_) {
        if (session_pair.second.active) {
            total_sessions++;
            if (session_pair.second.is_sender) {
                active_senders++;
            } else {
                active_receivers++;
            }
        }
    }

    global_stats_.active_sessions = total_sessions;
    global_stats_.active_senders = active_senders;
    global_stats_.active_receivers = active_receivers;
    global_stats_.last_update_time = std::chrono::steady_clock::now();
}

void NetworkAudioStreamer::cleanupInactiveSessions() {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    auto now = std::chrono::steady_clock::now();
    auto timeout_threshold = std::chrono::minutes(5);

    auto it = active_sessions_.begin();
    while (it != active_sessions_.end()) {
        AudioStreamSession& session = it->second;

        if (session.active) {
            auto inactive_time = now - session.stats.last_activity_time;

            if (inactive_time > timeout_threshold) {
                std::cout << "Cleaning up inactive session: " << session.session_name
                          << " (inactive for " << std::chrono::duration_cast<std::chrono::minutes>(inactive_time).count()
                          << " minutes)" << std::endl;

                stopSessionInternal(it->first);
                it = active_sessions_.erase(it);
            } else {
                ++it;
            }
        } else {
            ++it;
        }
    }
}

// Placeholder codec implementations
bool NetworkAudioStreamer::encodeOpus(const float* input, size_t num_samples,
                                     uint32_t bitrate, std::vector<uint8_t>& output) {
    // Simplified OPUS encoding placeholder
    output.resize(num_samples * 2); // Approximate size
    for (size_t i = 0; i < num_samples; ++i) {
        int16_t sample = static_cast<int16_t>(input[i] * 32767.0f);
        output[i * 2] = sample & 0xFF;
        output[i * 2 + 1] = (sample >> 8) & 0xFF;
    }
    return true;
}

bool NetworkAudioStreamer::decodeOpus(const uint8_t* input, size_t input_size,
                                     float* output, size_t max_samples, size_t& samples_decoded) {
    samples_decoded = std::min(input_size / 2, max_samples);
    for (size_t i = 0; i < samples_decoded; ++i) {
        int16_t sample = (static_cast<int16_t>(input[i * 2 + 1]) << 8) | input[i * 2];
        output[i] = sample / 32767.0f;
    }
    return true;
}

bool NetworkAudioStreamer::encodeAAC(const float* input, size_t num_samples,
                                    uint32_t bitrate, std::vector<uint8_t>& output) {
    // Simplified AAC encoding placeholder
    return encodeOpus(input, num_samples, bitrate, output);
}

bool NetworkAudioStreamer::decodeAAC(const uint8_t* input, size_t input_size,
                                    float* output, size_t max_samples, size_t& samples_decoded) {
    return decodeOpus(input, input_size, output, max_samples, samples_decoded);
}

bool NetworkAudioStreamer::encodeMP3(const float* input, size_t num_samples,
                                    uint32_t bitrate, std::vector<uint8_t>& output) {
    // Simplified MP3 encoding placeholder
    return encodeOpus(input, num_samples, bitrate, output);
}

bool NetworkAudioStreamer::decodeMP3(const uint8_t* input, size_t input_size,
                                    float* output, size_t max_samples, size_t& samples_decoded) {
    return decodeOpus(input, input_size, output, max_samples, samples_decoded);
}

bool NetworkAudioStreamer::encodeFLAC(const float* input, size_t num_samples, std::vector<uint8_t>& output) {
    // Simplified FLAC encoding placeholder
    return encodeOpus(input, num_samples, 1411000, output);
}

bool NetworkAudioStreamer::decodeFLAC(const uint8_t* input, size_t input_size,
                                     float* output, size_t max_samples, size_t& samples_decoded) {
    return decodeOpus(input, input_size, output, max_samples, samples_decoded);
}

bool NetworkAudioStreamer::encodeALAC(const float* input, size_t num_samples, std::vector<uint8_t>& output) {
    // Simplified ALAC encoding placeholder
    return encodeOpus(input, num_samples, 1411000, output);
}

bool NetworkAudioStreamer::decodeALAC(const uint8_t* input, size_t input_size,
                                     float* output, size_t max_samples, size_t& samples_decoded) {
    return decodeOpus(input, input_size, output, max_samples, samples_decoded);
}

// ============================================================================
// AudioStreamDiscovery Implementation
// ============================================================================

AudioStreamDiscovery::AudioStreamDiscovery()
    : running_(false)
    , initialized_(false) {
}

AudioStreamDiscovery::~AudioStreamDiscovery() {
    shutdown();
}

bool AudioStreamDiscovery::initialize() {
    std::lock_guard<std::mutex> lock(discovery_mutex_);

    if (initialized_) {
        return true;
    }

    try {
        // Initialize discovery service
        discovery_service_.initialize("VortexGPU_Audio", 6000);

        running_ = true;
        advertising_thread_ = std::thread(&AudioStreamDiscovery::advertisingThread, this);
        browsing_thread_ = std::thread(&AudioStreamDiscovery::browsingThread, this);

        initialized_ = true;

        std::cout << "AudioStreamDiscovery initialized" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize AudioStreamDiscovery: " << e.what() << std::endl;
        return false;
    }
}

void AudioStreamDiscovery::shutdown() {
    std::lock_guard<std::mutex> lock(discovery_mutex_);

    if (!initialized_) {
        return;
    }

    running_ = false;

    if (advertising_thread_.joinable()) {
        advertising_thread_.join();
    }

    if (browsing_thread_.joinable()) {
        browsing_thread_.join();
    }

    discovery_service_.shutdown();
    advertised_streams_.clear();
    discovered_streams_.clear();

    initialized_ = false;

    std::cout << "AudioStreamDiscovery shut down" << std::endl;
}

void AudioStreamDiscovery::advertiseStream(const std::string& stream_name,
                                          const AudioStreamConfig& config,
                                          const std::string& description,
                                          const std::vector<std::string>& tags) {
    if (!initialized_) {
        return;
    }

    std::lock_guard<std::mutex> lock(discovery_mutex_);

    AudioStreamInfo info;
    info.stream_name = stream_name;
    info.host_name = config_.host_name;
    info.port = config_.port;
    info.protocol = config_.enabled_protocols.empty() ? StreamingProtocol::RTP : config_.enabled_protocols[0];
    info.sample_rate = config_.sample_rate;
    info.channels = config_.channels;
    info.codec = config_.codec;
    info.bitrate = 64000;
    info.description = description;
    info.tags = tags;
    info.is_sender = true;
    info.quality_level = StreamQuality::HIGH;

    advertised_streams_[stream_name] = info;

    std::cout << "Advertising audio stream: " << stream_name << std::endl;
}

void AudioStreamDiscovery::stopAdvertising(const std::string& stream_name) {
    std::lock_guard<std::mutex> lock(discovery_mutex_);

    auto it = advertised_streams_.find(stream_name);
    if (it != advertised_streams_.end()) {
        advertised_streams_.erase(it);
        std::cout << "Stopped advertising stream: " << stream_name << std::endl;
    }
}

void AudioStreamDiscovery::setStreamAvailable(const std::string& stream_name, bool available) {
    std::lock_guard<std::mutex> lock(discovery_mutex_);

    auto it = advertised_streams_.find(stream_name);
    if (it != advertised_streams_.end()) {
        it->second.available = available;
    }
}

std::vector<AudioStreamInfo> AudioStreamDiscovery::getDiscoveredStreams() const {
    std::lock_guard<std::mutex> lock(discovery_mutex_);

    std::vector<AudioStreamInfo> streams;
    streams.reserve(discovered_streams_.size());

    for (const auto& pair : discovered_streams_) {
        streams.push_back(pair.second);
    }

    return streams;
}

std::vector<AudioStreamInfo> AudioStreamDiscovery::getAdvertisedStreams() const {
    std::lock_guard<std::mutex> lock(discovery_mutex_);

    std::vector<AudioStreamInfo> streams;
    streams.reserve(advertised_streams_.size());

    for (const auto& pair : advertised_streams_) {
        streams.push_back(pair.second);
    }

    return streams;
}

void AudioStreamDiscovery::setDiscoveryCallback(std::function<void(const AudioStreamInfo&)> callback) {
    std::lock_guard<std::mutex> lock(discovery_mutex_);
    discovery_callback_ = callback;
}

void AudioStreamDiscovery::browseForStreams() {
    if (!initialized_) {
        return;
    }

    discovery_service_.browseForServices("_vortexgpu_audio._tcp");
}

void AudioStreamDiscovery::resolveStream(const std::string& stream_name) {
    if (!initialized_) {
        return;
    }

    discovery_service_.resolveService(stream_name);
}

void AudioStreamDiscovery::setUpdateInterval(std::chrono::seconds interval) {
    update_interval_ = interval;
}

void AudioStreamDiscovery::advertisingThread() {
    auto last_advertise = std::chrono::steady_clock::now();

    while (running_) {
        try {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_advertise);

            if (elapsed >= update_interval_) {
                std::lock_guard<std::mutex> lock(discovery_mutex_);

                for (const auto& pair : advertised_streams_) {
                    const AudioStreamInfo& info = pair.second;
                    if (info.available) {
                        std::string service_data = createServiceData(info);
                        discovery_service_.registerService(info.stream_name, "_vortexgpu_audio._tcp", info.port, service_data);
                    }
                }

                last_advertise = now;
            }

            std::this_thread::sleep_for(std::chrono::seconds(1));

        } catch (const std::exception& e) {
            std::cerr << "Advertising thread error: " << e.what() << std::endl;
        }
    }
}

void AudioStreamDiscovery::browsingThread() {
    while (running_) {
        try {
            discovery_service_.processServiceEvents();

            std::vector<DiscoveredService> services = discovery_service_.getDiscoveredServices();

            std::lock_guard<std::mutex> lock(discovery_mutex_);

            for (const DiscoveredService& service : services) {
                if (service.service_type == "_vortexgpu_audio._tcp") {
                    AudioStreamInfo info = parseServiceData(service);

                    // Check if this is a new stream or updated info
                    auto it = discovered_streams_.find(service.service_name);
                    if (it == discovered_streams_.end() || it->second.last_seen != service.last_seen) {
                        discovered_streams_[service.service_name] = info;

                        if (discovery_callback_) {
                            discovery_callback_(info);
                        }
                    }
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));

        } catch (const std::exception& e) {
            std::cerr << "Browsing thread error: " << e.what() << std::endl;
        }
    }
}

std::string AudioStreamDiscovery::createServiceData(const AudioStreamInfo& info) {
    // Create a JSON-like string with stream information
    std::ostringstream oss;
    oss << "{";
    oss << "\"name\":\"" << info.stream_name << "\",";
    oss << "\"host\":\"" << info.host_name << "\",";
    oss << "\"port\":" << info.port << ",";
    oss << "\"protocol\":" << static_cast<int>(info.protocol) << ",";
    oss << "\"sample_rate\":" << info.sample_rate << ",";
    oss << "\"channels\":" << info.channels << ",";
    oss << "\"codec\":" << static_cast<int>(info.codec) << ",";
    oss << "\"bitrate\":" << info.bitrate << ",";
    oss << "\"description\":\"" << info.description << "\",";
    oss << "\"sender\":" << (info.is_sender ? "true" : "false") << ",";
    oss << "\"quality\":" << static_cast<int>(info.quality_level);
    oss << "}";

    return oss.str();
}

AudioStreamInfo AudioStreamDiscovery::parseServiceData(const DiscoveredService& service) {
    AudioStreamInfo info;

    // Parse service data - simplified parsing
    info.stream_name = service.service_name;
    info.host_name = service.host_name;
    info.port = service.port;
    info.last_seen = service.last_seen;

    // Set default values for required fields
    info.protocol = StreamingProtocol::RTP;
    info.sample_rate = 48000;
    info.channels = 2;
    info.codec = AudioCodec::OPUS;
    info.bitrate = 128000;
    info.is_sender = false;
    info.quality_level = StreamQuality::MEDIUM;

    // Parse actual service data if available
    if (!service.txt_records.empty()) {
        std::string data = service.txt_records[0];
        // In a real implementation, this would properly parse JSON
        // For now, we use the default values above
    }

    return info;
}

} // namespace Network
} // namespace VortexGPU