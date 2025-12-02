#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include <string>
#include <chrono>
#include <cstring>

namespace vortex::network::protocol {

// Visualization protocol version
constexpr uint8_t VISUALIZATION_PROTOCOL_VERSION = 1;

// Message flags
constexpr uint16_t FLAG_COMPRESSED = 0x0001;
constexpr uint16_t FLAG_CHECKSUM = 0x0002;
constexpr uint16_t FLAG_ENCRYPTED = 0x0004;
constexpr uint16_t FLAG_FRAGMENTED = 0x0008;
constexpr uint16_t FLAG_HIGH_PRECISION = 0x0010;
constexpr uint16_t FLAG_DELTA_ENCODED = 0x0020;

// Message types for visualization data
enum class VisualizationMessageType : uint16_t {
    UNKNOWN = 0,
    SPECTRUM_DATA = 1001,
    WAVEFORM_DATA = 1002,
    VU_LEVELS = 1003,
    PEAK_DATA = 1004,
    ZERO_CROSSINGS = 1005,
    ENVELOPE_DATA = 1006,
    PHASE_DATA = 1007,
    STEREO_IMAGE = 1008,
    CORRELATION_DATA = 1009,
    HISTOGRAM_DATA = 1010,
    WATERFALL_DATA = 1011,
    SPECTROGRAM_DATA = 1012,
    METADATA_UPDATE = 1013,
    PERFORMANCE_STATS = 1014,
    CONFIG_UPDATE = 1015,
    SUBSCRIPTION_ACK = 1016,
    ERROR_MESSAGE = 1017,
    HEARTBEAT = 1018,
    STREAM_START = 1019,
    STREAM_STOP = 1020,
    BUFFER_STATUS = 1021
};

// Data precision modes
enum class PrecisionMode : uint8_t {
    FLOAT_32 = 1,    // 32-bit float (default)
    FLOAT_16 = 2,    // 16-bit half precision
    FIXED_16 = 3,    // 16-bit fixed point
    UINT_8 = 4       // 8-bit unsigned (normalized 0-255)
};

// Frequency scaling modes
enum class FrequencyScale : uint8_t {
    LINEAR = 1,
    LOGARITHMIC = 2,
    MEL = 3,
    BARK = 4,
    OCTAVE = 5,
    ERB = 6  // Equivalent Rectangular Bandwidth
};

// Time scaling modes
enum class TimeScale : uint8_t {
    LINEAR = 1,
    LOGARITHMIC = 2,
    EXPONENTIAL = 3,
    PERCEPTUAL = 4
};

#pragma pack(push, 1)

/**
 * Message header for visualization protocol
 */
struct VisualizationMessageHeader {
    uint8_t version;                    // Protocol version
    uint16_t message_type;              // Message type ID
    uint16_t flags;                     // Message flags
    uint32_t sequence_number;           // Sequence number for ordering
    uint64_t timestamp;                 // Microsecond timestamp
    uint32_t payload_size;              // Size of payload in bytes
    uint32_t checksum;                  // CRC32 checksum
    uint8_t precision_mode;             // Data precision mode
    uint8_t reserved[3];                // Reserved for future use
};

/**
 * Spectrum data message
 */
struct SpectrumDataMessage {
    float sample_rate;                  // Audio sample rate in Hz
    uint32_t fft_size;                  // FFT size used
    uint32_t num_bins;                  // Number of frequency bins
    uint8_t window_type;                // Window function type
    uint8_t frequency_scale;            // Frequency scaling mode
    uint8_t amplitude_scale;            // Amplitude scaling mode
    uint8_t overlap_ratio;              // Overlap ratio (0-100%)
    float min_frequency;                // Minimum frequency in Hz
    float max_frequency;                // Maximum frequency in Hz
    float min_amplitude;                // Minimum amplitude in dB
    float max_amplitude;                // Maximum amplitude in dB
    uint32_t frequency_bin_count;       // Number of frequency bins
    uint32_t magnitude_bin_count;       // Number of magnitude bins
    // Followed by frequency_bins and magnitude_bins data
};

/**
 * Waveform data message
 */
struct WaveformDataMessage {
    float sample_rate;                  // Audio sample rate in Hz
    uint32_t num_samples;               // Number of waveform samples
    uint8_t display_mode;               // Display mode (peaks, rms, etc.)
    uint8_t time_scale;                 // Time scaling mode
    uint8_t compression_type;           // Compression type
    uint8_t reserved;                   // Reserved
    float window_duration;              // Window duration in seconds
    float decay_rate;                   // Peak decay rate
    uint32_t peak_count;                // Number of peaks detected
    uint32_t zero_crossing_count;       // Number of zero crossings
    // Followed by sample data and optional metadata
};

/**
 * VU levels message
 */
struct VULevelsMessage {
    float sample_rate;                  // Audio sample rate in Hz
    uint8_t meter_type;                 // VU meter type (peak, RMS, VU, etc.)
    uint8_t reference_level;            // Reference level type
    uint8_t integration_time;           // Integration time in ms (scaled)
    uint8_t channel_count;              // Number of channels
    float attack_time_ms;               // Attack time in milliseconds
    float release_time_ms;              // Release time in milliseconds
    float peak_hold_time_ms;            // Peak hold time in milliseconds
    float stereo_balance;               // Stereo balance (-1 to +1)
    float dynamic_range;                // Dynamic range in dB
    float crest_factor;                 // Crest factor
    // Followed by level data per channel
};

/**
 * Peak data message
 */
struct PeakDataMessage {
    float sample_rate;                  // Audio sample rate in Hz
    uint32_t num_peaks;                 // Number of peaks detected
    uint8_t detection_mode;             // Peak detection mode
    uint8_t threshold_mode;             // Threshold mode
    uint8_t interpolation_method;       // Interpolation method
    uint8_t reserved;                   // Reserved
    float threshold_dB;                 // Detection threshold in dB
    float min_peak_distance;            // Minimum distance between peaks
    float max_peaks_per_window;         // Maximum peaks per time window
    // Followed by peak frequency and amplitude data
};

/**
 * Zero crossing data message
 */
struct ZeroCrossingMessage {
    float sample_rate;                  // Audio sample rate in Hz
    uint32_t crossing_count;            // Number of zero crossings
    uint8_t detection_method;           // Detection method
    uint8_t hysteresis_enabled;         // Hysteresis enabled flag
    uint8_t interpolation_enabled;      // Interpolation enabled flag
    uint8_t reserved;                   // Reserved
    float hysteresis_threshold;         // Hysteresis threshold
    float estimated_frequency;          // Estimated fundamental frequency
    float confidence_score;             // Confidence score (0-1)
    // Followed by crossing timestamp data
};

/**
 * Stereo image data message
 */
struct StereoImageMessage {
    float sample_rate;                  // Audio sample rate in Hz
    uint8_t imaging_mode;               // Imaging mode (correlation, phase, etc.)
    uint8_t window_size;                // Analysis window size
    uint8_t frequency_bands;            // Number of frequency bands
    uint8_t reserved;                   // Reserved
    float correlation_coefficient;      // Stereo correlation coefficient
    float phase_difference_deg;         // Average phase difference in degrees
    float mid_side_ratio;               // Mid/Side ratio
    float width_percentage;             // Stereo width percentage
    float center_deviation;             // Center deviation
    // Followed by per-band imaging data
};

/**
 * Performance statistics message
 */
struct PerformanceStatsMessage {
    double processing_time_us;          // Processing time in microseconds
    double latency_us;                  // End-to-end latency in microseconds
    float current_fps;                  // Current frames per second
    float target_fps;                   // Target frames per second
    float cpu_usage_percent;            // CPU usage percentage
    float memory_usage_mb;              // Memory usage in MB
    uint32_t frames_processed;          // Total frames processed
    uint32_t frames_dropped;            // Total frames dropped
    uint32_t bytes_sent;                // Total bytes sent
    uint32_t active_subscriptions;      // Number of active subscriptions
    uint32_t queue_size;                // Current queue size
    double gpu_utilization_percent;     // GPU utilization percentage
    uint64_t uptime_seconds;            // Server uptime in seconds
};

/**
 * Subscription acknowledgment message
 */
struct SubscriptionAckMessage {
    uint8_t subscription_type;          // Type of subscription
    uint8_t status_code;                // Status code (0=success)
    uint8_t channel_count;              // Number of channels
    uint8_t reserved;                   // Reserved
    float update_frequency;             // Confirmed update frequency
    uint64_t subscription_id;           // Subscription identifier
    uint32_t max_message_size;          // Maximum message size
    uint32_t compression_level;         // Compression level
};

/**
 * Error message
 */
struct ErrorMessage {
    uint16_t error_code;                // Error code
    uint8_t severity;                   // Error severity (0=info, 1=warning, 2=error)
    uint8_t category;                   // Error category
    uint32_t context_data_size;         // Size of context data
    uint64_t error_timestamp;           // When error occurred
    // Followed by error message string and optional context data
};

#pragma pack(pop)

/**
 * High-performance binary protocol for audio visualization data
 */
class VisualizationProtocol {
public:
    VisualizationProtocol();
    ~VisualizationProtocol();

    // Configuration
    void setCompressionEnabled(bool enabled);
    void setCompressionLevel(int level);
    void setChecksumEnabled(bool enabled);
    void setPrecisionMode(PrecisionMode mode);
    void setDefaultFrequencyScale(FrequencyScale scale);
    void setDefaultTimeScale(TimeScale scale);

    // Serialization
    std::vector<uint8_t> serializeSpectrumData(const SpectrumDataMessage& header,
                                             const std::vector<float>& frequencies,
                                             const std::vector<float>& magnitudes);

    std::vector<uint8_t> serializeWaveformData(const WaveformDataMessage& header,
                                              const std::vector<float>& samples);

    std::vector<uint8_t> serializeVULevels(const VULevelsMessage& header,
                                          const std::vector<float>& levels);

    std::vector<uint8_t> serializePeakData(const PeakDataMessage& header,
                                         const std::vector<float>& peak_frequencies,
                                         const std::vector<float>& peak_magnitudes);

    std::vector<uint8_t> serializeStereoImage(const StereoImageMessage& header,
                                             const std::vector<float>& correlation_values);

    std::vector<uint8_t> serializePerformanceStats(const PerformanceStatsMessage& header);

    // Deserialization
    bool deserializeSpectrumData(const std::vector<uint8_t>& data,
                               SpectrumDataMessage& header,
                               std::vector<float>& frequencies,
                               std::vector<float>& magnitudes);

    bool deserializeWaveformData(const std::vector<uint8_t>& data,
                               WaveformDataMessage& header,
                               std::vector<float>& samples);

    bool deserializeVULevels(const std::vector<uint8_t>& data,
                           VULevelsMessage& header,
                           std::vector<float>& levels);

    // Message creation utilities
    std::vector<uint8_t> createSpectrumMessage(float sample_rate,
                                             uint32_t fft_size,
                                             const std::vector<float>& magnitudes,
                                             FrequencyScale freq_scale = FrequencyScale::LOGARITHMIC);

    std::vector<uint8_t> createWaveformMessage(float sample_rate,
                                              const std::vector<float>& samples,
                                              uint8_t display_mode);

    std::vector<uint8_t> createVUMessage(float sample_rate,
                                        const std::vector<float>& levels,
                                        uint8_t meter_type);

    std::vector<uint8_t> createPerformanceStatsMessage(double processing_time_us,
                                                      double latency_us,
                                                      float current_fps);

    // Message validation
    bool validateMessage(const std::vector<uint8_t>& data);
    bool validateSpectrumData(const SpectrumDataMessage& header,
                            const std::vector<float>& frequencies,
                            const std::vector<float>& magnitudes);

    // Compression and encoding
    std::vector<uint8_t> compressData(const std::vector<uint8_t>& data);
    std::vector<uint8_t> decompressData(const std::vector<uint8_t>& compressed);
    std::vector<uint8_t> deltaEncode(const std::vector<float>& data, float precision = 0.001f);
    std::vector<float> deltaDecode(const std::vector<uint8_t>& encoded, float precision = 0.001f);

    // Data conversion utilities
    std::vector<uint8_t> convertFloatsToPrecision(const std::vector<float>& data,
                                                 PrecisionMode precision);

    std::vector<float> convertPrecisionToFloats(const std::vector<uint8_t>& data,
                                               PrecisionMode precision);

    std::vector<float> generateFrequencyBins(float sample_rate,
                                           uint32_t fft_size,
                                           FrequencyScale scale,
                                           float min_freq = 20.0f,
                                           float max_freq = 20000.0f);

    // Performance optimization
    void setSerializationBuffer(size_t size);
    void enableFastMath(bool enabled);
    void enableSIMD(bool enabled);

private:
    // Configuration
    bool compression_enabled_;
    int compression_level_;
    bool checksum_enabled_;
    PrecisionMode precision_mode_;
    FrequencyScale default_freq_scale_;
    TimeScale default_time_scale_;
    bool fast_math_enabled_;
    bool simd_enabled_;

    // Buffers for performance
    std::vector<uint8_t> serialization_buffer_;
    size_t buffer_size_;

    // Internal serialization helpers
    std::vector<uint8_t> serializeHeader(const VisualizationMessageHeader& header);
    VisualizationMessageHeader deserializeHeader(const std::vector<uint8_t>& data);

    void writeFloat32(std::vector<uint8_t>& buffer, float value);
    void writeFloat16(std::vector<uint8_t>& buffer, float value);
    void writeFixed16(std::vector<uint8_t>& buffer, float value, float scale);
    void writeUInt8(std::vector<uint8_t>& buffer, float value);

    float readFloat32(const uint8_t* data);
    float readFloat16(const uint8_t* data);
    float readFixed16(const uint8_t* data, float scale);
    float readUInt8(const uint8_t* data);

    // Checksum calculation
    uint32_t calculateCRC32(const void* data, size_t size);
    uint32_t updateCRC32(uint32_t crc, uint8_t byte);

    // Compression helpers
    std::vector<uint8_t> compressZlib(const std::vector<uint8_t>& data);
    std::vector<uint8_t> decompressZlib(const std::vector<uint8_t>& compressed);

    // Fast math functions
    float fastLog2(float x);
    float fastExp2(float x);
    float fastSqrt(float x);

    // Current timestamp
    uint64_t getCurrentTimestamp() const;

    // Message validation
    bool validateHeader(const VisualizationMessageHeader& header);
    bool isValidMessageType(uint16_t type);
    bool isValidTimestamp(uint64_t timestamp);
};

/**
 * Factory for creating protocol instances with common configurations
 */
class VisualizationProtocolFactory {
public:
    static std::unique_ptr<VisualizationProtocol> createDefault();
    static std::unique_ptr<VisualizationProtocol> createHighPerformance();
    static std::unique_ptr<VisualizationProtocol> createLowLatency();
    static std::unique_ptr<VisualizationProtocol> createHighCompression();
};

/**
 * Protocol constants and limits
 */
namespace ProtocolConstants {
    constexpr size_t MAX_MESSAGE_SIZE = 16 * 1024 * 1024;  // 16MB
    constexpr size_t MAX_PAYLOAD_SIZE = MAX_MESSAGE_SIZE - sizeof(VisualizationMessageHeader);
    constexpr uint32_t MAX_SEQUENCE_NUMBER = 0xFFFFFFFF;
    constexpr uint32_t DEFAULT_CHECKSUM = 0;
    constexpr int DEFAULT_COMPRESSION_LEVEL = 6;
    constexpr size_t DEFAULT_BUFFER_SIZE = 64 * 1024;  // 64KB
    constexpr uint32_t MAX_FREQUENCY_BINS = 65536;
    constexpr uint32_t MAX_AUDIO_CHANNELS = 32;
    constexpr float MAX_SAMPLE_RATE = 192000.0f;
    constexpr float MIN_SAMPLE_RATE = 8000.0f;
    constexpr uint32_t MAX_FFT_SIZE = 1048576;  // 2^20
    constexpr float MAX_LATENCY_MS = 1000.0f;
    constexpr float MIN_UPDATE_FREQUENCY = 0.1f;
    constexpr float MAX_UPDATE_FREQUENCY = 1000.0f;
}

/**
 * Error codes for visualization protocol
 */
enum class ProtocolErrorCode : uint16_t {
    SUCCESS = 0,
    INVALID_MESSAGE = 1001,
    INVALID_HEADER = 1002,
    INVALID_CHECKSUM = 1003,
    INVALID_VERSION = 1004,
    INVALID_TIMESTAMP = 1005,
    INVALID_DATA_FORMAT = 1006,
    COMPRESSION_ERROR = 1007,
    DECOMPRESSION_ERROR = 1008,
    BUFFER_OVERFLOW = 1009,
    INVALID_FREQUENCY_DATA = 1010,
    INVALID_WAVEFORM_DATA = 1011,
    INVALID_LEVEL_DATA = 1012,
    SUBSCRIPTION_FAILED = 1013,
    RATE_LIMIT_EXCEEDED = 1014,
    SERVER_OVERLOAD = 1015,
    CONNECTION_CLOSED = 1016,
    TIMEOUT = 1017,
    AUTHENTICATION_FAILED = 1018,
    AUTHORIZATION_FAILED = 1019,
    RESOURCE_EXHAUSTED = 1020,
    INTERNAL_ERROR = 1021,
    UNKNOWN_ERROR = 9999
};

} // namespace vortex::network::protocol