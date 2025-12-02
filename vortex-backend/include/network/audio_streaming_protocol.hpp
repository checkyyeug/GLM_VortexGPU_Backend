#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <array>
#include <chrono>
#include <functional>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <queue>
#include <condition_variable>
#include <thread>

#ifdef VORTEX_ENABLE_ASIO
#include <asio.hpp>
#endif

namespace vortex {
namespace network {

/**
 * Network Audio Streaming Protocol
 * Low-latency, high-quality audio streaming over IP networks
 * Supports real-time bidirectional audio with adaptive quality and error recovery
 */

enum class StreamingProtocol {
    RTP,                 ///< Real-time Transport Protocol
    RTSP,                ///< Real-time Streaming Protocol
    WEBRTC,              ///< WebRTC audio streaming
    DSD_OVER_IP,         ///< DSD streaming over IP
    RAW_UDP,             ///< Raw UDP streaming
    RAW_TCP,             ///< Raw TCP streaming
    HTTP_LIVE_STREAMING, ///< HTTP Live Streaming (HLS)
    MPEG_DASH,           ///< Dynamic Adaptive Streaming over HTTP
    RTMP,                ///< Real-Time Messaging Protocol
    SIP,                 ///< Session Initiation Protocol
    SPOTIFY_CONNECT,     ///< Spotify Connect protocol
    AIRPLAY,             ///< Apple AirPlay protocol
    CHROMECAST,          ///< Google Chromecast protocol
    CUSTOM               ///< Custom protocol
};

enum class AudioCodec {
    PCM_S16LE,           ///< 16-bit PCM little-endian
    PCM_S24LE,           ///< 24-bit PCM little-endian
    PCM_S32LE,           ///< 32-bit PCM little-endian
    PCM_FLOAT32,         ///< 32-bit float
    OPUS,                ///< Opus codec
    AAC,                 ///< Advanced Audio Coding
    MP3,                 ///< MP3 codec
    FLAC,                ///< FLAC lossless codec
    ALAC,                ///< Apple Lossless Audio Codec
    VORBIS,              ///< Ogg Vorbis codec
    DSD64,               ///< DSD64 (1-bit 2.8224 MHz)
    DSD128,              ///< DSD128 (1-bit 5.6448 MHz)
    DSD256,              ///< DSD256 (1-bit 11.2896 MHz)
    DSD512,              ///< DSD512 (1-bit 22.5792 MHz)
    G711A,               ///< G.711 A-law
    G711U,               ///< G.711 μ-law
    G722,                ///< G.722 wideband codec
    G726,                ///< G.726 variable bitrate codec
    SPEEX,               ///< Speex codec
    CELT,                ///< CELT codec
    SILK,                ///< SILK codec
    CUSTOM_CODEC         ///< Custom codec
};

enum class TransportProtocol {
    UDP,                 ///< UDP transport
    TCP,                 ///< TCP transport
    SCTP,                ///< SCTP transport
    TLS_UDP,             ///< UDP with TLS
    TLS_TCP,             ///< TCP with TLS
    QUIC,                ///< QUIC protocol
    WEBSOCKET,           ///< WebSocket transport
    RAW_SOCKET,          ///< Raw socket transport
    CUSTOM_TRANSPORT     ///< Custom transport
};

enum class StreamingMode {
    LIVE,                ///< Live streaming
    ON_DEMAND,           ///< On-demand streaming
    FILE_STREAMING,      ///< File-based streaming
    BUFFER_STREAMING,     ///< Buffer-based streaming
    INTERACTIVE,          ///< Interactive streaming
    BROADCAST,           ///< Broadcast streaming
    UNICAST,             ///< Unicast streaming
    MULTICAST,           ///< Multicast streaming
    ANYCAST,             ///< Anycast streaming
    PEER_TO_PEER,        ///< Peer-to-peer streaming
    DISTRIBUTED          ///< Distributed streaming
};

enum class QualityLevel {
    LOW,                 ///< Low quality (8kHz mono)
    MEDIUM,              ///< Medium quality (22kHz mono)
    STANDARD,            ///< Standard quality (44kHz stereo)
    HIGH,                ///< High quality (48kHz stereo)
    PREMIUM,             ///< Premium quality (96kHz stereo)
    AUDIOPHILE,          ///< Audiophile quality (192kHz stereo)
    PROFESSIONAL,        ///< Professional quality (192kHz+ multichannel)
    CUSTOM               ///< Custom quality settings
};

enum class LatencyMode {
    ULTRA_LOW,           ///< Ultra-low latency (<5ms)
    LOW,                 ///< Low latency (5-20ms)
    MEDIUM,              ///< Medium latency (20-100ms)
    HIGH,                ///< High latency (100-500ms)
    FLEXIBLE,            ///< Flexible latency
    ADAPTIVE             ///< Adaptive latency
};

enum class PacketType {
    AUDIO_DATA,          ///< Audio data packet
    CONTROL,             ///< Control packet
    METADATA,            ///< Metadata packet
    SYNCHRONIZATION,     ///< Synchronization packet
    HEARTBEAT,           ///< Heartbeat packet
    ACK,                 ///< Acknowledgment packet
    NACK,                ///< Negative acknowledgment packet
    RECOVERY,            ///< Recovery packet
    QUALITY_REPORT,      ///< Quality report packet
    STATISTICS,          ///< Statistics packet
    CONFIGURATION,       ///< Configuration packet
    DISCOVERY,           ///< Discovery packet
    SESSION_START,       ///< Session start packet
    SESSION_END,         ///< Session end packet
    CUSTOM_PACKET        ///< Custom packet type
};

struct AudioStreamConfig {
    StreamingProtocol protocol = StreamingProtocol::RTP; ///< Streaming protocol
    AudioCodec codec = AudioCodec::OPUS;                   ///< Audio codec
    TransportProtocol transport = TransportProtocol::UDP;  ///< Transport protocol
    StreamingMode mode = StreamingMode::LIVE;            ///< Streaming mode
    QualityLevel quality = QualityLevel::HIGH;            ///< Quality level
    LatencyMode latency_mode = LatencyMode::LOW;          ///< Latency mode

    // Audio parameters
    int sample_rate = 48000;                               ///< Sample rate (Hz)
    int channels = 2;                                      ///< Number of channels
    int bit_depth = 16;                                    ///< Bit depth
    int bit_rate = 128000;                                 ///< Target bit rate (bps)
    int frame_size = 20;                                   ///< Frame size (ms)
    int buffer_size = 960;                                 ///< Buffer size (samples)

    // Network parameters
    std::string local_host = "0.0.0.0";                    ///< Local host address
    uint16_t local_port = 0;                               ///< Local port
    std::string remote_host;                                ///< Remote host address
    uint16_t remote_port = 0;                              ///< Remote port
    std::string multicast_address;                          ///< Multicast address
    uint16_t multicast_port = 0;                            ///< Multicast port
    int ttl = 64;                                           ///< Time-to-live for multicast
    int dscp = 0;                                          ///< DSCP value for QoS

    // Quality of service
    bool enable_redundancy = false;                        ///< Enable packet redundancy
    int redundancy_count = 2;                              ///< Number of redundant packets
    bool enable_fec = false;                               ///< Enable forward error correction
    int fec_percentage = 20;                              ///< FEC overhead percentage
    bool enable_adaptive_bitrate = false;                ///< Enable adaptive bitrate
    int min_bit_rate = 64000;                             ///< Minimum bit rate
    int max_bit_rate = 512000;                            ///< Maximum bit rate

    // Timing and synchronization
    bool enable_ntp_sync = false;                         ///< Enable NTP synchronization
    std::string ntp_server;                                ///< NTP server address
    bool enable_rtcp = true;                               ///< Enable RTCP
    int rtcp_interval_ms = 1000;                          ///< RTCP interval
    bool enable_clock_recovery = true;                     ///< Enable clock recovery
    double clock_drift_tolerance = 0.1;                    ///< Clock drift tolerance

    // Security
    bool enable_encryption = false;                        ///< Enable encryption
    std::string encryption_key;                            ///< Encryption key
    std::string encryption_algorithm;                      ///< Encryption algorithm
    bool enable_authentication = false;                     ///< Enable authentication
    std::string authentication_token;                       ///< Authentication token

    // Error handling
    bool enable_error_recovery = true;                    ///< Enable error recovery
    int max_missing_packets = 10;                          ///< Maximum missing packets before recovery
    int max_jitter_buffer_ms = 100;                       ///< Maximum jitter buffer size
    bool enable_concealment = true;                        ///< Enable packet loss concealment
    bool enable_interpolation = true;                      ///< Enable sample interpolation

    // Buffer management
    int min_buffer_ms = 20;                               ///< Minimum buffer size
    int max_buffer_ms = 200;                              ///< Maximum buffer size
    int target_buffer_ms = 60;                            ///< Target buffer size
    bool enable_adaptive_buffer = true;                   ///< Enable adaptive buffering
    float buffer_fill_ratio = 0.8f;                        ///< Target buffer fill ratio

    // Monitoring and statistics
    bool enable_monitoring = true;                         ///< Enable performance monitoring
    bool enable_statistics = true;                         ///< Enable statistics collection
    int statistics_interval_ms = 1000;                     ///< Statistics reporting interval
    bool enable_detailed_logging = false;                  ///< Enable detailed logging
    std::string log_level = "INFO";                        ///< Log level

    // Session management
    int session_timeout_ms = 30000;                        ///< Session timeout
    bool enable_keepalive = true;                           ///< Enable keepalive packets
    int keepalive_interval_ms = 5000;                      ///< Keepalive interval
    bool enable_auto_reconnect = true;                     ///< Enable automatic reconnection
    int reconnect_attempts = 3;                            ///< Reconnection attempts
    int reconnect_delay_ms = 1000;                         ///< Reconnection delay
};

struct AudioPacket {
    PacketType type = PacketType::AUDIO_DATA;              ///< Packet type
    uint32_t sequence_number = 0;                         ///< Sequence number
    uint32_t timestamp = 0;                                ///< Timestamp
    uint16_t ssrc = 0;                                     ///< Synchronization source identifier
    uint8_t payload_type = 0;                             ///< Payload type
    bool marker = false;                                   ///< Marker bit
    std::vector<uint8_t> payload;                         ///< Audio payload
    std::vector<uint8_t> header;                          ///< Packet header
    std::chrono::steady_clock::time_point arrival_time;    ///< Packet arrival time
    size_t packet_size = 0;                               ///< Total packet size
    double rtt_ms = 0.0;                                  ///< Round-trip time
    bool is_retransmitted = false;                        ///< Packet retransmission flag
    uint8_t redundancy_index = 0;                         ///< Redundancy index
};

struct StreamStatistics {
    // Packet statistics
    uint64_t packets_sent = 0;                            ///< Total packets sent
    uint64_t packets_received = 0;                         ///< Total packets received
    uint64_t packets_lost = 0;                             ///< Total packets lost
    uint64_t packets_discarded = 0;                        ///< Total packets discarded
    uint64_t bytes_sent = 0;                               ///< Total bytes sent
    uint64_t bytes_received = 0;                          ///< Total bytes received
    uint64_t retransmitted_packets = 0;                    ///< Retransmitted packets
    uint64_t out_of_order_packets = 0;                    ///< Out-of-order packets

    // Timing statistics
    double avg_rtt_ms = 0.0;                               ///< Average round-trip time
    double jitter_ms = 0.0;                                ///< Jitter in milliseconds
    double packet_loss_rate = 0.0;                        ///< Packet loss rate (percentage)
    double bandwidth_utilization = 0.0;                   ///< Bandwidth utilization
    double cpu_utilization = 0.0;                         ///< CPU utilization percentage

    // Audio quality statistics
    double audio_level_dbfs = -INFINITY;                  ///< Current audio level
    double peak_level_dbfs = -INFINITY;                   ///< Peak audio level
    double snr_db = 0.0;                                   ///< Signal-to-noise ratio
    double thd_percent = 0.0;                             ///< Total harmonic distortion
    double clock_drift_ppm = 0.0;                          ///< Clock drift in ppm
    double buffer_occupancy = 0.0;                         ///< Buffer occupancy percentage

    // Quality metrics
    double mos_score = 0.0;                               ///< Mean Opinion Score
    double r_factor = 0.0;                                 ///< R-factor quality metric
    double audio_quality = 0.0;                            ///< Audio quality score (0-1)

    // Session statistics
    std::chrono::steady_clock::time_point session_start;   ///< Session start time
    std::chrono::steady_clock::time_point last_update;     ///< Last update time
    double session_duration_seconds = 0.0;                 ///< Session duration

    // Error statistics
    uint64_t error_count = 0;                              ///< Total error count
    uint64_t timeout_count = 0;                            ///< Timeout count
    uint64_t buffer_underrun_count = 0;                    ///< Buffer underrun count
    uint64_t buffer_overrun_count = 0;                     ///< Buffer overrun count
};

struct StreamingSession {
    uint32_t session_id = 0;                               ///< Session ID
    std::string session_name;                              ///< Session name
    std::string description;                               ///< Session description
    AudioStreamConfig config;                              ///< Session configuration
    StreamStatistics statistics;                           ///< Session statistics
    bool is_active = false;                                ///< Session active flag
    bool is_sender = false;                                 ///< Sender/receiver flag
    std::string remote_endpoint;                           ///< Remote endpoint
    std::chrono::steady_clock::time_point creation_time;     ///< Session creation time
    std::chrono::steady_clock::time_point last_activity;   ///< Last activity time
    int timeout_ms = 30000;                                ///< Session timeout
};

using AudioPacketCallback = std::function<void(const AudioPacket& packet)>;
using StatisticsCallback = std::function<void(const StreamStatistics& stats)>;
using SessionEventCallback = std::function<void(const std::string& event, const std::string& details)>;
using QualityCallback = std::function<void(double audio_quality, double mos_score)>;

/**
 * Network Audio Streamer
 * Core audio streaming functionality with protocol support
 */
class NetworkAudioStreamer {
public:
    NetworkAudioStreamer();
    ~NetworkAudioStreamer();

    /**
     * Initialize audio streamer
     * @param config Stream configuration
     * @return True if initialization successful
     */
    bool initialize(const AudioStreamConfig& config);

    /**
     * Shutdown audio streamer
     */
    void shutdown();

    /**
     * Start streaming session
     * @param session_name Session name
     * @param is_sender True if this is a sender session
     * @return Session ID if successful, 0 otherwise
     */
    uint32_t startSession(const std::string& session_name, bool is_sender);

    /**
     * Stop streaming session
     * @param session_id Session ID
     * @return True if session stopped successfully
     */
    bool stopSession(uint32_t session_id);

    /**
     * Get session information
     * @param session_id Session ID
     * @return Session information if found
     */
    std::optional<StreamingSession> getSession(uint32_t session_id) const;

    /**
     * Send audio data
     * @param session_id Session ID
     * @param audio_data Audio buffer
     * @param num_samples Number of samples
     * @param timestamp Timestamp (0 for auto)
     * @return True if data sent successfully
     */
    bool sendAudioData(uint32_t session_id, const float* audio_data, size_t num_samples, uint32_t timestamp = 0);

    /**
     * Receive audio data
     * @param session_id Session ID
     * @param audio_buffer Audio buffer to receive into
     * @param max_samples Maximum samples to receive
     * @return Number of samples received
     */
    size_t receiveAudioData(uint32_t session_id, float* audio_buffer, size_t max_samples);

    /**
     * Get received audio data (callback-based)
     * @param session_id Session ID
     * @param callback Audio packet callback
     * @return True if callback registered successfully
     */
    bool setAudioCallback(uint32_t session_id, AudioPacketCallback callback);

    /**
     * Connect to remote endpoint
     * @param session_id Session ID
     * @param remote_host Remote host address
     * @param remote_port Remote port
     * @return True if connection successful
     */
    bool connect(uint32_t session_id, const std::string& remote_host, uint16_t remote_port);

    /**
     * Disconnect from remote endpoint
     * @param session_id Session ID
     * @return True if disconnection successful
     */
    bool disconnect(uint32_t session_id);

    /**
     * Send control message
     * @param session_id Session ID
     * @param message Control message
     * @return True if message sent successfully
     */
    bool sendControlMessage(uint32_t session_id, const std::string& message);

    /**
     * Send metadata
     * @param session_id Session ID
     * @param metadata Metadata dictionary
     * @return True if metadata sent successfully
     */
    bool sendMetadata(uint32_t session_id, const std::unordered_map<std::string, std::string>& metadata);

    /**
     * Update streaming configuration
     * @param session_id Session ID
     * @param new_config New configuration
     * @return True if update successful
     */
    bool updateConfiguration(uint32_t session_id, const AudioStreamConfig& new_config);

    /**
     * Get session statistics
     * @param session_id Session ID
     * @return Session statistics
     */
    StreamStatistics getSessionStatistics(uint32_t session_id) const;

    /**
     * Get all active sessions
     * @return List of active sessions
     */
    std::vector<StreamingSession> getActiveSessions() const;

    /**
     * Set statistics callback
     * @param callback Statistics callback
     */
    void setStatisticsCallback(StatisticsCallback callback);

    /**
     * Set quality callback
     * @param callback Quality callback
     */
    void setQualityCallback(QualityCallback callback);

    /**
     * Set session event callback
     * @param callback Session event callback
     */
    void setSessionEventCallback(SessionEventCallback callback);

    /**
     * Get network latency
     * @param session_id Session ID
     * @return Latency in milliseconds
     */
    double getLatency(uint32_t session_id) const;

    /**
     * Get audio quality metrics
     * @param session_id Session ID
     * @return Quality metrics
     */
    std::unordered_map<std::string, double> getQualityMetrics(uint32_t session_id) const;

    /**
     * Enable/disable adaptive quality
     * @param session_id Session ID
     * @param enable Enable adaptive quality
     * @return True if setting applied successfully
     */
    bool enableAdaptiveQuality(uint32_t session_id, bool enable);

    /**
     * Configure jitter buffer
     * @param session_id Session ID
     * @param target_size Target buffer size (ms)
     * @param max_size Maximum buffer size (ms)
     * @return True if configuration applied successfully
     */
    bool configureJitterBuffer(uint32_t session_id, int target_size, int max_size);

    /**
     * Enable packet loss concealment
     * @param session_id Session ID
     * @param enable Enable concealment
     * @return True if setting applied successfully
     */
    bool enablePacketLossConcealment(uint32_t session_id, bool enable);

    /**
     * Send heartbeat packet
     * @param session_id Session ID
     * @return True if heartbeat sent successfully
     */
    bool sendHeartbeat(uint32_t session_id);

    /**
     * Get recommended configuration
     * @param target_latency Target latency
     * @param network_bandwidth Available bandwidth (bps)
     * @return Recommended configuration
     */
    AudioStreamConfig getRecommendedConfiguration(double target_latency, uint64_t network_bandwidth) const;

private:
    struct SessionInternal {
        StreamingSession session;
        std::unique_ptr<std::thread> send_thread;
        std::unique_ptr<std::thread> receive_thread;
        std::unique_ptr<std::thread> stats_thread;
        std::atomic<bool> active{false};
        std::atomic<bool> sender{false};
        std::queue<AudioPacket> send_queue;
        std::queue<AudioPacket> receive_queue;
        std::mutex queue_mutex;
        std::condition_variable queue_cv;
        AudioPacketCallback audio_callback;
        std::vector<float> jitter_buffer;
        size_t jitter_buffer_size = 0;
        size_t jitter_buffer_write_pos = 0;
        size_t jitter_buffer_read_pos = 0;
        uint32_t expected_seq_num = 0;
        std::chrono::steady_clock::time_point last_packet_time;
    };

    // Core state
    bool initialized_ = false;
    AudioStreamConfig default_config_;
    mutable std::mutex sessions_mutex_;
    std::unordered_map<uint32_t, std::unique_ptr<SessionInternal>> sessions_;
    std::atomic<uint32_t> next_session_id_{1};

    // Network layer
#ifdef VORTEX_ENABLE_ASIO
    std::unique_ptr<asio::io_context> io_context_;
    std::unique_ptr<std::thread> io_thread_;
#endif

    // Callbacks
    StatisticsCallback stats_callback_;
    QualityCallback quality_callback_;
    SessionEventCallback event_callback_;

    // Monitoring
    std::atomic<bool> monitoring_active_{false};
    std::thread monitoring_thread_;

    // Internal methods
    void startSessionThread(SessionInternal& session);
    void stopSessionThread(SessionInternal& session);
    void sendThreadLoop(uint32_t session_id);
    void receiveThreadLoop(uint32_t session_id);
    void statisticsThreadLoop(uint32_t session_id);
    void monitoringLoop();

    // Packet processing
    AudioPacket createAudioPacket(const float* audio_data, size_t num_samples, uint32_t timestamp);
    AudioPacket createControlPacket(const std::string& message);
    AudioPacket createMetadataPacket(const std::unordered_map<std::string, std::string>& metadata);
    AudioPacket createHeartbeatPacket();

    // Audio processing
    std::vector<uint8_t> encodeAudio(const float* audio_data, size_t num_samples, AudioCodec codec);
    std::vector<float> decodeAudio(const std::vector<uint8_t>& encoded_data, AudioCodec codec);
    void applyPacketLossConcealment(std::vector<float>& buffer, size_t num_samples, PacketType lost_type);

    // Network operations
    bool sendPacket(uint32_t session_id, const AudioPacket& packet);
    bool receivePacket(uint32_t session_id, AudioPacket& packet);
    bool sendUDP(uint32_t session_id, const AudioPacket& packet);
    bool receiveUDP(uint32_t session_id, AudioPacket& packet);
    bool sendTCP(uint32_t session_id, const AudioPacket& packet);
    bool receiveTCP(uint32_t session_id, AudioPacket& packet);

    // Quality management
    void updateQualityMetrics(uint32_t session_id);
    void adjustBitRate(uint32_t session_id, double target_bitrate);
    void adjustJitterBuffer(uint32_t session_id, int target_size);

    // Jitter buffer management
    void addToJitterBuffer(SessionInternal& session, const AudioPacket& packet);
    bool readFromJitterBuffer(SessionInternal& session, AudioPacket& packet);
    void flushJitterBuffer(SessionInternal& session);

    // Statistics and monitoring
    void updateSessionStatistics(uint32_t session_id, const AudioPacket& packet, bool is_sent);
    void calculateQualityMetrics(uint32_t session_id);
    void notifySessionEvent(const std::string& event, const std::string& details);

    // Protocol-specific implementations
    bool initializeRTP(uint32_t session_id);
    bool initializeRTSP(uint32_t session_id);
    bool initializeWebRTC(uint32_t session_id);
    bool initializeDSDOverIP(uint32_t session_id);

    void processRTPPacket(AudioPacket& packet);
    void processRTCPPacket(AudioPacket& packet);
    void processSRTPPacket(AudioPacket& packet);
    void processDSDBuffer(const std::vector<uint8_t>& buffer);

    // Utility functions
    uint32_t generateSessionId() const;
    uint32_t generateSSRC() const;
    uint32_t getCurrentTimestamp() const;
    std::chrono::steady_clock::time_point getNTPTime() const;
    double calculateRTT(uint32_t session_id, uint32_t timestamp) const;
    double calculateJitter(const std::vector<std::chrono::steady_clock::time_point>& packet_times) const;
    double calculateMOS(double rtt, double jitter, double packet_loss) const;
};

/**
 * Audio Stream Discovery
 * Network service discovery for audio streaming
 */
class AudioStreamDiscovery {
public:
    AudioStreamDiscovery();
    ~AudioStreamDiscovery();

    /**
     * Initialize discovery service
     * @return True if initialization successful
     */
    bool initialize();

    /**
     * Shutdown discovery service
     */
    void shutdown();

    /**
     * Start broadcasting service availability
     * @param service_name Service name
     * @param service_type Service type
     * @param port Listening port
     * @return True if broadcasting started successfully
     */
    bool startBroadcast(const std::string& service_name, const std::string& service_type, uint16_t port);

    /**
     * Stop broadcasting
     */
    void stopBroadcast();

    /**
     * Search for available services
     * @param service_type Service type to search for
     * @param timeout_ms Search timeout in milliseconds
     * @return List of discovered services
     */
    std::vector<std::pair<std::string, std::string>> searchServices(
        const std::string& service_type, int timeout_ms = 5000);

    /**
     * Register service discovery callback
     * @param callback Discovery callback
     */
    void setDiscoveryCallback(std::function<void(const std::string& service, const std::string& info)> callback);

private:
    bool initialized_ = false;
    std::unique_ptr<std::thread> discovery_thread_;
    std::atomic<bool> discovery_active_{false};
    std::string service_name_;
    std::string service_type_;
    uint16_t port_;
    std::function<void(const std::string&, const std::string&)> discovery_callback_;

    void discoveryLoop();
};

// Utility functions
namespace audio_streaming_utils {

    // Protocol utilities
    std::string protocolToString(StreamingProtocol protocol);
    std::string codecToString(AudioCodec codec);
    std::string transportToString(TransportProtocol transport);
    std::string packetTypeToString(PacketType type);

    StreamingProtocol stringToProtocol(const std::string& str);
    AudioCodec stringToCodec(const std::string& str);
    TransportProtocol stringToTransport(const std::string& str);

    // Configuration utilities
    AudioStreamConfig createLowLatencyConfig();
    AudioStreamConfig createHighQualityConfig();
    AudioStreamConfig createAdaptiveConfig();
    AudioStreamConfig createDSDConfig();
    AudioStreamConfig createMulticastConfig();

    // Audio format conversion
    int getCodecSampleRate(AudioCodec codec);
    int getCodecChannels(AudioCodec codec);
    int getCodecBitDepth(AudioCodec codec);
    int getCodecFrameSize(AudioCodec codec);
    bool isLosslessCodec(AudioCodec codec);
    bool isRealTimeCodec(AudioCodec codec);

    // Quality calculations
    double calculateRequiredBandwidth(AudioCodec codec, int sample_rate, int channels, int bit_rate);
    double calculateLatency(int buffer_size_ms, int network_latency_ms, int processing_delay_ms);
    double calculateOptimalBitRate(AudioCodec codec, double target_quality, double network_bandwidth);
    double calculateAudioQuality(StreamStatistics& stats);
    double calculateMOS(StreamStatistics& stats);

    // Network utilities
    std::string getLocalIPAddress();
    std::vector<std::string> getNetworkInterfaces();
    bool isPortAvailable(uint16_t port);
    uint16_t findAvailablePort(uint16_t start_port = 49152);
    bool configureNetworkQoS(int socket, int dscp, int priority);

    // Time utilities
    uint32_t getNTPTimestamp();
    std::chrono::steady_clock::time_point convertNTPTime(uint32_t ntp_timestamp);
    double calculateClockDrift(uint32_t local_ts, uint32_t remote_ts);

    // Packet utilities
    AudioPacket createRTPPacket(const std::vector<uint8_t>& payload, uint8_t payload_type, uint32_t seq_num, uint32_t timestamp, uint32_t ssrc, bool marker);
    AudioPacket createRTCPPacket(const std::vector<uint8_t>& payload, uint32_t ssrc, uint8_t packet_type);
    std::vector<uint8_t> serializeRTPPacket(const AudioPacket& packet);
    AudioPacket deserializeRTPPacket(const std::vector<uint8_t>& data);

    // Error handling
    bool hasPacketLoss(const StreamStatistics& stats);
    bool hasHighLatency(const StreamStatistics& stats);
    bool hasPoorQuality(const StreamStatistics& stats);
    std::vector<std::string> diagnoseIssues(const StreamStatistics& stats);
    std::vector<std::string> suggestImprovements(const StreamStatistics& stats);

    // Validation utilities
    bool validateConfiguration(const AudioStreamConfig& config);
    bool validateAudioParameters(int sample_rate, int channels, int bit_depth);
    bool validateNetworkParameters(const std::string& host, uint16_t port);
    bool validateCodecParameters(AudioCodec codec, int sample_rate, int bit_rate);
}

} // namespace network
} // namespace core
} // namespace vortex