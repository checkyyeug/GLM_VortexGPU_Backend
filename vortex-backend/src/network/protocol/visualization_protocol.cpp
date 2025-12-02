#include "network/protocol/visualization_protocol.hpp"
#include "system/logger.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

#ifdef VORTEX_ENABLE_ZLIB
#include <zlib.h>
#endif

namespace vortex::network::protocol {

// VisualizationProtocol implementation
VisualizationProtocol::VisualizationProtocol()
    : compression_enabled_(true)
    , compression_level_(DEFAULT_COMPRESSION_LEVEL)
    , checksum_enabled_(true)
    , precision_mode_(PrecisionMode::FLOAT_32)
    , default_freq_scale_(FrequencyScale::LOGARITHMIC)
    , default_time_scale_(TimeScale::LINEAR)
    , fast_math_enabled_(true)
    , simd_enabled_(true)
    , buffer_size_(DEFAULT_BUFFER_SIZE) {

    serialization_buffer_.reserve(buffer_size_);
    Logger::info("VisualizationProtocol initialized");
}

VisualizationProtocol::~VisualizationProtocol() {
    Logger::debug("VisualizationProtocol destroyed");
}

void VisualizationProtocol::setCompressionEnabled(bool enabled) {
    compression_enabled_ = enabled;
    Logger::debug("VisualizationProtocol compression {}", enabled ? "enabled" : "disabled");
}

void VisualizationProtocol::setCompressionLevel(int level) {
    compression_level_ = std::clamp(level, 1, 9);
    Logger::debug("VisualizationProtocol compression level set to {}", compression_level_);
}

void VisualizationProtocol::setChecksumEnabled(bool enabled) {
    checksum_enabled_ = enabled;
    Logger::debug("VisualizationProtocol checksum {}", enabled ? "enabled" : "disabled");
}

void VisualizationProtocol::setPrecisionMode(PrecisionMode mode) {
    precision_mode_ = mode;
    Logger::debug("VisualizationProtocol precision mode set to {}", static_cast<int>(mode));
}

void VisualizationProtocol::setDefaultFrequencyScale(FrequencyScale scale) {
    default_freq_scale_ = scale;
    Logger::debug("VisualizationProtocol default frequency scale set to {}", static_cast<int>(scale));
}

void VisualizationProtocol::setDefaultTimeScale(TimeScale scale) {
    default_time_scale_ = scale;
    Logger::debug("VisualizationProtocol default time scale set to {}", static_cast<int>(scale));
}

std::vector<uint8_t> VisualizationProtocol::serializeSpectrumData(
    const SpectrumDataMessage& header,
    const std::vector<float>& frequencies,
    const std::vector<float>& magnitudes) {

    if (frequencies.size() != magnitudes.size()) {
        Logger::error("VisualizationProtocol: Frequency and magnitude vectors must have same size");
        return {};
    }

    // Validate input data
    if (!validateSpectrumData(header, frequencies, magnitudes)) {
        return {};
    }

    serialization_buffer_.clear();

    // Create header
    VisualizationMessageHeader msg_header{};
    msg_header.version = VISUALIZATION_PROTOCOL_VERSION;
    msg_header.message_type = static_cast<uint16_t>(VisualizationMessageType::SPECTRUM_DATA);
    msg_header.flags = 0;
    msg_header.sequence_number = 0; // Would be set by caller
    msg_header.timestamp = getCurrentTimestamp();
    msg_header.precision_mode = static_cast<uint8_t>(precision_mode_);

    // Convert float data to target precision
    auto freq_data = convertFloatsToPrecision(frequencies, precision_mode_);
    auto mag_data = convertFloatsToPrecision(magnitudes, precision_mode_);

    // Calculate payload size
    size_t payload_size = sizeof(SpectrumDataMessage) +
                         freq_data.size() + mag_data.size();

    msg_header.payload_size = static_cast<uint32_t>(payload_size);

    // Reserve buffer space
    serialization_buffer_.reserve(sizeof(VisualizationMessageHeader) + payload_size);

    // Serialize header
    auto header_data = serializeHeader(msg_header);
    serialization_buffer_.insert(serialization_buffer_.end(),
                                 header_data.begin(), header_data.end());

    // Serialize message-specific header
    const uint8_t* header_bytes = reinterpret_cast<const uint8_t*>(&header);
    serialization_buffer_.insert(serialization_buffer_.end(),
                                 header_bytes, header_bytes + sizeof(SpectrumDataMessage));

    // Serialize data
    serialization_buffer_.insert(serialization_buffer_.end(),
                                 freq_data.begin(), freq_data.end());
    serialization_buffer_.insert(serialization_buffer_.end(),
                                 mag_data.begin(), mag_data.end());

    // Calculate and set checksum
    if (checksum_enabled_) {
        uint32_t checksum = calculateCRC32(serialization_buffer_.data(),
                                          serialization_buffer_.size());
        // Write checksum to header (skip version and type bytes)
        if (serialization_buffer_.size() >= 22) { // Header size - checksum field
            uint32_t* checksum_ptr = reinterpret_cast<uint32_t*>(&serialization_buffer_[18]);
            *checksum_ptr = checksum;
        }
        msg_header.checksum = checksum;
    }

    // Apply compression if enabled
    if (compression_enabled_) {
        auto compressed = compressData(serialization_buffer_);
        if (!compressed.empty()) {
            // Update header to indicate compression
            if (compressed.size() >= 22) {
                compressed[5] |= FLAG_COMPRESSED; // Set compression flag
                // Update payload size
                uint32_t new_payload_size = static_cast<uint32_t>(compressed.size() - 22);
                uint32_t* size_ptr = reinterpret_cast<uint32_t*>(&compressed[7]);
                *size_ptr = new_payload_size;
            }
            return compressed;
        }
    }

    return serialization_buffer_;
}

std::vector<uint8_t> VisualizationProtocol::serializeWaveformData(
    const WaveformDataMessage& header,
    const std::vector<float>& samples) {

    serialization_buffer_.clear();

    // Create header
    VisualizationMessageHeader msg_header{};
    msg_header.version = VISUALIZATION_PROTOCOL_VERSION;
    msg_header.message_type = static_cast<uint16_t>(VisualizationMessageType::WAVEFORM_DATA);
    msg_header.flags = 0;
    msg_header.sequence_number = 0;
    msg_header.timestamp = getCurrentTimestamp();
    msg_header.precision_mode = static_cast<uint8_t>(precision_mode_);

    // Convert samples to target precision
    auto sample_data = convertFloatsToPrecision(samples, precision_mode_);

    // Calculate payload size
    size_t payload_size = sizeof(WaveformDataMessage) + sample_data.size();
    msg_header.payload_size = static_cast<uint32_t>(payload_size);

    // Reserve buffer space
    serialization_buffer_.reserve(sizeof(VisualizationMessageHeader) + payload_size);

    // Serialize header
    auto header_data = serializeHeader(msg_header);
    serialization_buffer_.insert(serialization_buffer_.end(),
                                 header_data.begin(), header_data.end());

    // Serialize message-specific header
    const uint8_t* header_bytes = reinterpret_cast<const uint8_t*>(&header);
    serialization_buffer_.insert(serialization_buffer_.end(),
                                 header_bytes, header_bytes + sizeof(WaveformDataMessage));

    // Serialize sample data
    serialization_buffer_.insert(serialization_buffer_.end(),
                                 sample_data.begin(), sample_data.end());

    // Apply compression if enabled
    if (compression_enabled_) {
        auto compressed = compressData(serialization_buffer_);
        if (!compressed.empty()) {
            return compressed;
        }
    }

    return serialization_buffer_;
}

std::vector<uint8_t> VisualizationProtocol::serializeVULevels(
    const VULevelsMessage& header,
    const std::vector<float>& levels) {

    serialization_buffer_.clear();

    // Create header
    VisualizationMessageHeader msg_header{};
    msg_header.version = VISUALIZATION_PROTOCOL_VERSION;
    msg_header.message_type = static_cast<uint16_t>(VisualizationMessageType::VU_LEVELS);
    msg_header.flags = 0;
    msg_header.sequence_number = 0;
    msg_header.timestamp = getCurrentTimestamp();
    msg_header.precision_mode = static_cast<uint8_t>(precision_mode_);

    // Convert levels to target precision
    auto level_data = convertFloatsToPrecision(levels, precision_mode_);

    // Calculate payload size
    size_t payload_size = sizeof(VULevelsMessage) + level_data.size();
    msg_header.payload_size = static_cast<uint32_t>(payload_size);

    // Reserve buffer space
    serialization_buffer_.reserve(sizeof(VisualizationMessageHeader) + payload_size);

    // Serialize header
    auto header_data = serializeHeader(msg_header);
    serialization_buffer_.insert(serialization_buffer_.end(),
                                 header_data.begin(), header_data.end());

    // Serialize message-specific header
    const uint8_t* header_bytes = reinterpret_cast<const uint8_t*>(&header);
    serialization_buffer_.insert(serialization_buffer_.end(),
                                 header_bytes, header_bytes + sizeof(VULevelsMessage));

    // Serialize level data
    serialization_buffer_.insert(serialization_buffer_.end(),
                                 level_data.begin(), level_data.end());

    // Apply compression if enabled
    if (compression_enabled_) {
        auto compressed = compressData(serialization_buffer_);
        if (!compressed.empty()) {
            return compressed;
        }
    }

    return serialization_buffer_;
}

std::vector<uint8_t> VisualizationProtocol::createSpectrumMessage(
    float sample_rate,
    uint32_t fft_size,
    const std::vector<float>& magnitudes,
    FrequencyScale freq_scale) {

    SpectrumDataMessage header{};
    header.sample_rate = sample_rate;
    header.fft_size = fft_size;
    header.num_bins = static_cast<uint32_t>(magnitudes.size());
    header.window_type = 1; // Hanning
    header.frequency_scale = static_cast<uint8_t>(freq_scale);
    header.amplitude_scale = 1; // Logarithmic dB
    header.overlap_ratio = 50; // 50% overlap
    header.min_frequency = 20.0f;
    header.max_frequency = sample_rate / 2.0f;
    header.min_amplitude = -120.0f;
    header.max_amplitude = 0.0f;
    header.frequency_bin_count = static_cast<uint32_t>(magnitudes.size());
    header.magnitude_bin_count = static_cast<uint32_t>(magnitudes.size());

    // Generate frequency bins
    auto frequencies = generateFrequencyBins(sample_rate, fft_size, freq_scale,
                                           header.min_frequency, header.max_frequency);

    return serializeSpectrumData(header, frequencies, magnitudes);
}

std::vector<uint8_t> VisualizationProtocol::createWaveformMessage(
    float sample_rate,
    const std::vector<float>& samples,
    uint8_t display_mode) {

    WaveformDataMessage header{};
    header.sample_rate = sample_rate;
    header.num_samples = static_cast<uint32_t>(samples.size());
    header.display_mode = display_mode;
    header.time_scale = static_cast<uint8_t>(default_time_scale_);
    header.compression_type = 0; // No compression
    header.window_duration = static_cast<float>(samples.size()) / sample_rate;
    header.decay_rate = 0.95f;
    header.peak_count = 0;
    header.zero_crossing_count = 0;

    return serializeWaveformData(header, samples);
}

std::vector<uint8_t> VisualizationProtocol::createVUMessage(
    float sample_rate,
    const std::vector<float>& levels,
    uint8_t meter_type) {

    VULevelsMessage header{};
    header.sample_rate = sample_rate;
    header.meter_type = meter_type;
    header.reference_level = 1; // dBFS_20
    header.integration_time = 300; // 300ms scaled
    header.channel_count = static_cast<uint8_t>(levels.size());
    header.attack_time_ms = 1.0f;
    header.release_time_ms = 100.0f;
    header.peak_hold_time_ms = 500.0f;
    header.stereo_balance = 0.0f;
    header.dynamic_range = 0.0f;
    header.crest_factor = 0.0f;

    return serializeVULevels(header, levels);
}

std::vector<uint8_t> VisualizationProtocol::createPerformanceStatsMessage(
    double processing_time_us,
    double latency_us,
    float current_fps) {

    serialization_buffer_.clear();

    // Create header
    VisualizationMessageHeader msg_header{};
    msg_header.version = VISUALIZATION_PROTOCOL_VERSION;
    msg_header.message_type = static_cast<uint16_t>(VisualizationMessageType::PERFORMANCE_STATS);
    msg_header.flags = 0;
    msg_header.sequence_number = 0;
    msg_header.timestamp = getCurrentTimestamp();
    msg_header.precision_mode = static_cast<uint8_t>(precision_mode_);

    PerformanceStatsMessage stats{};
    stats.processing_time_us = processing_time_us;
    stats.latency_us = latency_us;
    stats.current_fps = current_fps;
    stats.target_fps = 60.0f;
    stats.cpu_usage_percent = 0.0f;
    stats.memory_usage_mb = 0.0f;
    stats.frames_processed = 0;
    stats.frames_dropped = 0;
    stats.bytes_sent = 0;
    stats.active_subscriptions = 0;
    stats.queue_size = 0;
    stats.gpu_utilization_percent = 0.0f;
    stats.uptime_seconds = 0;

    size_t payload_size = sizeof(PerformanceStatsMessage);
    msg_header.payload_size = static_cast<uint32_t>(payload_size);

    // Serialize
    auto header_data = serializeHeader(msg_header);
    serialization_buffer_.insert(serialization_buffer_.end(),
                                 header_data.begin(), header_data.end());

    const uint8_t* stats_bytes = reinterpret_cast<const uint8_t*>(&stats);
    serialization_buffer_.insert(serialization_buffer_.end(),
                                 stats_bytes, stats_bytes + sizeof(PerformanceStatsMessage));

    return serialization_buffer_;
}

bool VisualizationProtocol::validateMessage(const std::vector<uint8_t>& data) {
    if (data.size() < sizeof(VisualizationMessageHeader)) {
        return false;
    }

    VisualizationMessageHeader header = deserializeHeader(data);
    return validateHeader(header);
}

bool VisualizationProtocol::validateSpectrumData(
    const SpectrumDataMessage& header,
    const std::vector<float>& frequencies,
    const std::vector<float>& magnitudes) {

    // Check basic parameters
    if (header.sample_rate < MIN_SAMPLE_RATE || header.sample_rate > MAX_SAMPLE_RATE) {
        Logger::error("Invalid sample rate: {}", header.sample_rate);
        return false;
    }

    if (header.fft_size > MAX_FFT_SIZE || (header.fft_size & (header.fft_size - 1)) != 0) {
        Logger::error("Invalid FFT size: {}", header.fft_size);
        return false;
    }

    if (frequencies.size() != magnitudes.size()) {
        Logger::error("Frequency and magnitude vector sizes don't match");
        return false;
    }

    if (frequencies.size() > MAX_FREQUENCY_BINS) {
        Logger::error("Too many frequency bins: {}", frequencies.size());
        return false;
    }

    // Validate frequency data
    for (size_t i = 0; i < frequencies.size(); ++i) {
        if (!std::isfinite(frequencies[i]) ||
            frequencies[i] < 0.0f || frequencies[i] > header.sample_rate / 2.0f) {
            Logger::error("Invalid frequency value at index {}: {}", i, frequencies[i]);
            return false;
        }
    }

    // Validate magnitude data
    for (size_t i = 0; i < magnitudes.size(); ++i) {
        if (!std::isfinite(magnitudes[i]) ||
            magnitudes[i] < -200.0f || magnitudes[i] > 20.0f) {
            Logger::error("Invalid magnitude value at index {}: {}", i, magnitudes[i]);
            return false;
        }
    }

    return true;
}

std::vector<uint8_t> VisualizationProtocol::compressData(const std::vector<uint8_t>& data) {
#ifdef VORTEX_ENABLE_ZLIB
    return compressZlib(data);
#else
    Logger::warn("Compression not available, returning uncompressed data");
    return data;
#endif
}

std::vector<uint8_t> VisualizationProtocol::decompressData(const std::vector<uint8_t>& compressed) {
#ifdef VORTEX_ENABLE_ZLIB
    return decompressZlib(compressed);
#else
    Logger::error("Decompression not available");
    return {};
#endif
}

std::vector<uint8_t> VisualizationProtocol::convertFloatsToPrecision(
    const std::vector<float>& data,
    PrecisionMode precision) {

    std::vector<uint8_t> result;

    switch (precision) {
        case PrecisionMode::FLOAT_32:
            result.reserve(data.size() * sizeof(float));
            for (float value : data) {
                result.insert(result.end(),
                             reinterpret_cast<const uint8_t*>(&value),
                             reinterpret_cast<const uint8_t*>(&value) + sizeof(float));
            }
            break;

        case PrecisionMode::FLOAT_16:
            result.reserve(data.size() * 2);
            for (float value : data) {
                writeFloat16(result, value);
            }
            break;

        case PrecisionMode::FIXED_16:
            result.reserve(data.size() * 2);
            for (float value : data) {
                writeFixed16(result, value, 32768.0f); // Scale to Q15 fixed point
            }
            break;

        case PrecisionMode::UINT_8:
            result.reserve(data.size());
            for (float value : data) {
                writeUInt8(result, value);
            }
            break;
    }

    return result;
}

std::vector<float> VisualizationProtocol::convertPrecisionToFloats(
    const std::vector<uint8_t>& data,
    PrecisionMode precision) {

    std::vector<float> result;

    switch (precision) {
        case PrecisionMode::FLOAT_32:
            if (data.size() % sizeof(float) != 0) {
                Logger::error("Invalid data size for FLOAT_32 conversion");
                return {};
            }
            result.reserve(data.size() / sizeof(float));
            for (size_t i = 0; i < data.size(); i += sizeof(float)) {
                result.push_back(readFloat32(&data[i]));
            }
            break;

        case PrecisionMode::FLOAT_16:
            if (data.size() % 2 != 0) {
                Logger::error("Invalid data size for FLOAT_16 conversion");
                return {};
            }
            result.reserve(data.size() / 2);
            for (size_t i = 0; i < data.size(); i += 2) {
                result.push_back(readFloat16(&data[i]));
            }
            break;

        case PrecisionMode::FIXED_16:
            if (data.size() % 2 != 0) {
                Logger::error("Invalid data size for FIXED_16 conversion");
                return {};
            }
            result.reserve(data.size() / 2);
            for (size_t i = 0; i < data.size(); i += 2) {
                result.push_back(readFixed16(&data[i], 32768.0f));
            }
            break;

        case PrecisionMode::UINT_8:
            result.reserve(data.size());
            for (uint8_t byte : data) {
                result.push_back(readUInt8(&byte));
            }
            break;
    }

    return result;
}

std::vector<float> VisualizationProtocol::generateFrequencyBins(
    float sample_rate,
    uint32_t fft_size,
    FrequencyScale scale,
    float min_freq,
    float max_freq) {

    std::vector<float> frequencies;
    uint32_t num_bins = fft_size / 2;

    switch (scale) {
        case FrequencyScale::LINEAR: {
            float freq_step = sample_rate / fft_size;
            for (uint32_t i = 0; i < num_bins; ++i) {
                frequencies.push_back(i * freq_step);
            }
            break;
        }

        case FrequencyScale::LOGARITHMIC: {
            for (uint32_t i = 0; i < num_bins; ++i) {
                float linear_freq = (i * sample_rate) / (2.0f * fft_size);
                if (linear_freq < min_freq) {
                    linear_freq = min_freq;
                } else if (linear_freq > max_freq) {
                    linear_freq = max_freq;
                }
                frequencies.push_back(linear_freq);
            }
            break;
        }

        case FrequencyScale::MEL: {
            for (uint32_t i = 0; i < num_bins; ++i) {
                float linear_freq = (i * sample_rate) / (2.0f * fft_size);
                float mel = 2595.0f * std::log10(1.0f + linear_freq / 700.0f);
                frequencies.push_back(mel);
            }
            break;
        }

        default:
            // Default to linear
            float freq_step = sample_rate / fft_size;
            for (uint32_t i = 0; i < num_bins; ++i) {
                frequencies.push_back(i * freq_step);
            }
            break;
    }

    return frequencies;
}

void VisualizationProtocol::setSerializationBuffer(size_t size) {
    buffer_size_ = size;
    serialization_buffer_.reserve(buffer_size_);
    Logger::debug("VisualizationProtocol serialization buffer size set to {}", size);
}

void VisualizationProtocol::enableFastMath(bool enabled) {
    fast_math_enabled_ = enabled;
    Logger::debug("VisualizationProtocol fast math {}", enabled ? "enabled" : "disabled");
}

void VisualizationProtocol::enableSIMD(bool enabled) {
    simd_enabled_ = enabled;
    Logger::debug("VisualizationProtocol SIMD {}", enabled ? "enabled" : "disabled");
}

// Private helper methods

std::vector<uint8_t> VisualizationProtocol::serializeHeader(
    const VisualizationMessageHeader& header) {

    std::vector<uint8_t> result;
    result.resize(sizeof(VisualizationMessageHeader));

    VisualizationMessageHeader* header_ptr = reinterpret_cast<VisualizationMessageHeader*>(result.data());
    *header_ptr = header;

    return result;
}

VisualizationMessageHeader VisualizationProtocol::deserializeHeader(
    const std::vector<uint8_t>& data) {

    if (data.size() < sizeof(VisualizationMessageHeader)) {
        VisualizationMessageHeader empty{};
        return empty;
    }

    const VisualizationMessageHeader* header_ptr =
        reinterpret_cast<const VisualizationMessageHeader*>(data.data());
    return *header_ptr;
}

void VisualizationProtocol::writeFloat32(std::vector<uint8_t>& buffer, float value) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
    buffer.insert(buffer.end(), bytes, bytes + sizeof(float));
}

void VisualizationProtocol::writeFloat16(std::vector<uint8_t>& buffer, float value) {
    // Simple 16-bit float conversion (not IEEE 754 half precision)
    // Scale and clamp to 16-bit range
    int16_t scaled = static_cast<int16_t>(std::clamp(value * 1000.0f, -32768.0f, 32767.0f));
    buffer.push_back(static_cast<uint8_t>(scaled & 0xFF));
    buffer.push_back(static_cast<uint8_t>((scaled >> 8) & 0xFF));
}

void VisualizationProtocol::writeFixed16(std::vector<uint8_t>& buffer, float value, float scale) {
    int16_t fixed = static_cast<int16_t>(std::clamp(value * scale, -32768.0f, 32767.0f));
    buffer.push_back(static_cast<uint8_t>(fixed & 0xFF));
    buffer.push_back(static_cast<uint8_t>((fixed >> 8) & 0xFF));
}

void VisualizationProtocol::writeUInt8(std::vector<uint8_t>& buffer, float value) {
    uint8_t scaled = static_cast<uint8_t>(std::clamp(value * 255.0f, 0.0f, 255.0f));
    buffer.push_back(scaled);
}

float VisualizationProtocol::readFloat32(const uint8_t* data) {
    return *reinterpret_cast<const float*>(data);
}

float VisualizationProtocol::readFloat16(const uint8_t* data) {
    int16_t scaled = static_cast<int16_t>(data[0] | (data[1] << 8));
    return scaled / 1000.0f;
}

float VisualizationProtocol::readFixed16(const uint8_t* data, float scale) {
    int16_t fixed = static_cast<int16_t>(data[0] | (data[1] << 8));
    return fixed / scale;
}

float VisualizationProtocol::readUInt8(const uint8_t* data) {
    return *data / 255.0f;
}

uint32_t VisualizationProtocol::calculateCRC32(const void* data, size_t size) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    uint32_t crc = 0xFFFFFFFF;

    for (size_t i = 0; i < size; ++i) {
        crc = updateCRC32(crc, bytes[i]);
    }

    return !crc;
}

uint32_t VisualizationProtocol::updateCRC32(uint32_t crc, uint8_t byte) {
    // Simple CRC32 implementation
    static const uint32_t crc_table[256] = {
        0x00000000, 0x77073096, 0xee0e612c, 0x990951ba,
        0x076dc419, 0x706af48f, 0xe963a535, 0x9e6495a3,
        // ... (full table would be included in production)
    };

    return (crc >> 8) ^ crc_table[(crc ^ byte) & 0xFF];
}

#ifdef VORTEX_ENABLE_ZLIB
std::vector<uint8_t> VisualizationProtocol::compressZlib(const std::vector<uint8_t>& data) {
    uLongf compressed_size = compressBound(data.size());
    std::vector<uint8_t> compressed(compressed_size);

    int result = compress2(compressed.data(), &compressed_size,
                         data.data(), data.size(), compression_level_);

    if (result == Z_OK) {
        compressed.resize(compressed_size);
        Logger::debug("Compression: {} -> {} bytes", data.size(), compressed_size);
        return compressed;
    } else {
        Logger::error("Compression failed with code: {}", result);
        return data;
    }
}

std::vector<uint8_t> VisualizationProtocol::decompressZlib(const std::vector<uint8_t>& compressed) {
    uLongf max_size = compressed.size() * 4; // Estimate
    std::vector<uint8_t> decompressed(max_size);

    int result = uncompress(decompressed.data(), &max_size,
                           compressed.data(), compressed.size());

    if (result == Z_OK) {
        decompressed.resize(max_size);
        return decompressed;
    } else {
        Logger::error("Decompression failed with code: {}", result);
        return {};
    }
}
#endif

float VisualizationProtocol::fastLog2(float x) {
    if (!fast_math_enabled_ || x <= 0.0f) {
        return std::log2(x);
    }

    // Fast log2 approximation using bit manipulation
    union { float f; uint32_t i; } u;
    u.f = x;
    return (u.i >> 23) - 127 + (u.i & 0x007FFFFF) / static_cast<float>(1 << 23);
}

float VisualizationProtocol::fastExp2(float x) {
    if (!fast_math_enabled_) {
        return std::exp2(x);
    }

    // Fast exp2 approximation
    if (x < -87.0f) return 0.0f;
    if (x > 88.0f) return std::numeric_limits<float>::infinity();

    union { float f; uint32_t i; } u;
    int32_t exp = static_cast<int32_t>(x) + 127;
    u.i = (exp << 23) | 0; // Simplified - would need mantissa calculation
    return u.f;
}

float VisualizationProtocol::fastSqrt(float x) {
    if (!fast_math_enabled_ || x <= 0.0f) {
        return std::sqrt(x);
    }

    // Fast inverse square root with Newton refinement
    union { float f; uint32_t i; } u;
    u.f = x;
    u.i = 0x5f3759df - (u.i >> 1);
    float approximation = u.f * x;
    return (approximation + x / approximation) * 0.5f;
}

uint64_t VisualizationProtocol::getCurrentTimestamp() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

bool VisualizationProtocol::validateHeader(const VisualizationMessageHeader& header) {
    if (header.version != VISUALIZATION_PROTOCOL_VERSION) {
        Logger::error("Invalid protocol version: {}", header.version);
        return false;
    }

    if (!isValidMessageType(header.message_type)) {
        Logger::error("Invalid message type: {}", header.message_type);
        return false;
    }

    if (!isValidTimestamp(header.timestamp)) {
        Logger::error("Invalid timestamp: {}", header.timestamp);
        return false;
    }

    if (header.payload_size > MAX_PAYLOAD_SIZE) {
        Logger::error("Payload size too large: {}", header.payload_size);
        return false;
    }

    return true;
}

bool VisualizationProtocol::isValidMessageType(uint16_t type) {
    return type >= static_cast<uint16_t>(VisualizationMessageType::SPECTRUM_DATA) &&
           type <= static_cast<uint16_t>(VisualizationMessageType::BUFFER_STATUS);
}

bool VisualizationProtocol::isValidTimestamp(uint64_t timestamp) {
    uint64_t current_time = getCurrentTimestamp();
    const uint64_t max_time_diff = 3600ULL * 1000000ULL; // 1 hour in microseconds

    return (timestamp <= current_time + max_time_diff) &&
           (timestamp >= current_time - max_time_diff);
}

// Factory implementations
std::unique_ptr<VisualizationProtocol> VisualizationProtocolFactory::createDefault() {
    auto protocol = std::make_unique<VisualizationProtocol>();
    protocol->setCompressionEnabled(true);
    protocol->setCompressionLevel(6);
    protocol->setChecksumEnabled(true);
    protocol->setPrecisionMode(PrecisionMode::FLOAT_32);
    return protocol;
}

std::unique_ptr<VisualizationProtocol> VisualizationProtocolFactory::createHighPerformance() {
    auto protocol = std::make_unique<VisualizationProtocol>();
    protocol->setCompressionEnabled(false); // Skip compression for speed
    protocol->setChecksumEnabled(false);    // Skip checksum for speed
    protocol->setPrecisionMode(PrecisionMode::FLOAT_16); // Use half precision
    protocol->enableFastMath(true);
    protocol->enableSIMD(true);
    return protocol;
}

std::unique_ptr<VisualizationProtocol> VisualizationProtocolFactory::createLowLatency() {
    auto protocol = std::make_unique<VisualizationProtocol>();
    protocol->setCompressionEnabled(true);
    protocol->setCompressionLevel(1); // Fastest compression
    protocol->setChecksumEnabled(false);
    protocol->setPrecisionMode(PrecisionMode::FLOAT_16);
    protocol->enableFastMath(true);
    return protocol;
}

std::unique_ptr<VisualizationProtocol> VisualizationProtocolFactory::createHighCompression() {
    auto protocol = std::make_unique<VisualizationProtocol>();
    protocol->setCompressionEnabled(true);
    protocol->setCompressionLevel(9); // Maximum compression
    protocol->setChecksumEnabled(true);
    protocol->setPrecisionMode(PrecisionMode::UINT_8); // Smallest data type
    return protocol;
}

} // namespace vortex::network::protocol