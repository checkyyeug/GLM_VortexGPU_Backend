#pragma once

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <vector>
#include <string>
        ,
#include <unordered_map>
#include <queue>
#include <functional>
#include <array>
#include <deque>
#include <cmath>

#include "system/logger.hpp"
#include "core/dsp/realtime_effects_chain.hpp"

namespace vortex::core::dsp {

/**
 * Dynamic Range Processor for Vortex GPU Audio Backend
 *
 * This component provides professional dynamic range processing including
 * compressors, limiters, expanders, and gates with advanced features like
 * multiband processing, sidechain control, and adaptive response curves.
 *
 * Features:
 * - Sub-sample attack and release times
 * - Multiple compressor models (Opto, FET, VCA, Vari-Mu)
 * - Multi-band and parallel processing
 * - Sidechain filtering and ducking
 * - Look-ahead processing with zero latency compensation
 * - Adaptive knee and transfer functions
 * - Real-time gain reduction metering
 * - Comprehensive metering and analysis
 * - CPU/GPU acceleration options
 * - Professional audio standards (K-System, LUFS)
 * - Advanced automatic makeup gain
 */

/**
 * Dynamic range processor types
 */
enum class DynamicRangeType {
    COMPRESSOR,        // Dynamic range compressor
    LIMITER,           // Peak limiter
    EXPANDER,          // Dynamic range expander
    GATE,              // Noise gate
    DEESSER,           // De-esser
    MULTIBAND_COMP,    // Multi-band compressor
    UPWARD_COMP,       // Upward compressor
    PARALLEL_COMP,     // Parallel compression
    COMPRESSOR_LIMITER,// Compressor-limiter combo
    TRANSIENT_SHAPER,  // Transient shaper
    ADAPTIVE_DRC,      // Adaptive dynamic range controller
    AUTO_GAIN,         // Automatic gain control
    DUCKER,            // Sidechain ducker
    ENVELOPE_FOLLOWER, // Envelope follower
    RMS_COMPRESSOR,    // RMS-based compressor
    PEAK_COMPRESSOR,   // Peak-based compressor
    SMOOTH_LIMITER,    // Smooth limiter
    BRICKWALL_LIMITER, // Brickwall limiter
    CUSTOM             // Custom dynamic range processor
};

/**
 * Knee types for transfer functions
 */
enum class KneeType {
    HARD_KNEE,         // Hard knee transfer function
    SOFT_KNEE,         // Soft knee transfer function
    ADAPTIVE_KNEE,     // Adaptive knee based on signal
    LINEAR_KNEE,       // Linear transition
    EXPONENTIAL_KNEE,  // Exponential knee
    LOGARITHMIC_KNEE,  // Logarithmic knee
    CUSTOM_KNEE        // Custom knee curve
};

/**
 * Detection modes
 */
enum class DetectionMode {
    PEAK,              // Peak detection
    RMS,               // RMS detection
    AVERAGE,           // Average detection
    WEIGHTED_PEAK,     // Weighted peak detection
    PERCENTILE,        // Percentile detection
    ENVELOPE,          // Envelope detection
    ADAPTIVE,          // Adaptive detection
    SPECTRAL,          // Spectral detection
    MULTI_MODE         // Multiple detection modes
};

/**
 * Sidechain filter types
 */
enum class SidechainFilterType {
    NONE,              // No filtering
    LOW_PASS,          // Low-pass filter
    HIGH_PASS,         // High-pass filter
    BAND_PASS,         // Band-pass filter
    NOTCH,             // Notch filter
    PEAK,              // Peaking filter
    SHELF,             // Shelf filter
    CUSTOM             // Custom filter response
};

/**
 * Compression models
 */
enum class CompressionModel {
    FEED_FORWARD,       // Feed-forward compression
    FEED_BACK,          // Feedback compression
    OPTICAL,            // Optical compressor emulation
    VARI_MU,            // Variable-mu tube compressor
    FET,                // FET compressor emulation
    VCA,                // VCA compressor emulation
    DIGITAL,            // Digital compression
    HYBRID,             // Hybrid approach
    ADAPTIVE,           // Adaptive compression
    LOOK_AHEAD,         // Look-ahead compression
    PARALLEL,           // Parallel compression
    MID_SIDE,           // Mid/side processing
    MULTI_BAND,         // Multi-band processing
    SPECTRAL            // Spectral compression
};

/**
 * Dynamic range processing statistics
 */
struct DynamicRangeStatistics {
    uint64_t total_process_calls = 0;
    uint64_t successful_calls = 0;
    double avg_processing_time_us = 0.0;
    double max_processing_time_us = 0.0;
    double min_processing_time_us = std::numeric_limits<double>::max();

    // Gain reduction statistics
    float current_gain_reduction_db = 0.0f;
    float max_gain_reduction_db = 0.0f;
    float avg_gain_reduction_db = 0.0f;
    float gain_reduction_variance_db = 0.0f;

    // Level statistics
    float input_level_dbfs = 0.0f;
    float output_level_dbfs = 0.0f;
    float makeup_gain_db = 0.0f;
    float threshold_crossings_per_second = 0.0f;

    // Performance metrics
    double cpu_utilization_percent = 0.0;
    double gpu_utilization_percent = 0.0;
    float memory_usage_mb = 0.0f;
    uint64_t buffer_underruns = 0;
    uint64_t buffer_overruns = 0;
    uint64_t clipping_events = 0;

    // Real-time metrics
    float attack_rate_db_per_sec = 0.0f;
    float release_rate_db_per_sec = 0.0f;
    float envelope_follower_value = 0.0f;
    float sidechain_level_dbfs = 0.0f;

    // Timing
    double avg_latency_ms = 0.0;
    double max_latency_ms = 0.0;
    uint32_t look_ahead_samples = 0;

    // Quality metrics
    float distortion_thd_percent = 0.0f;
    float noise_floor_dbfs = -120.0f;
    float dynamic_range_db = 0.0f;
    float crest_factor_db = 0.0f;

    // State
    bool is_active = false;
    bool is_limiting = false;
    bool is_compressing = false;
    bool is_expanding = false;
    bool is_gating = false;

    std::chrono::steady_clock::time_point last_reset_time;
};

/**
 * Dynamic range processor parameters
 */
struct DynamicRangeParameters {
    // Basic parameters
    DynamicRangeType type = DynamicRangeType::COMPRESSOR;
    CompressionModel model = CompressionModel::DIGITAL;
    bool enabled = true;
    bool bypassed = false;

    // Threshold and ratio
    float threshold_dbfs = -20.0f;          // Processing threshold
    float ratio = 4.0f;                     // Compression/expansion ratio
    float knee_width_db = 2.0f;              // Knee width
    KneeType knee_type = KneeType::SOFT_KNEE;
    float range_db = 0.0f;                  // Range for expanders/gates

    // Attack and release
    float attack_time_ms = 5.0f;            // Attack time
    float release_time_ms = 100.0f;          // Release time
    float hold_time_ms = 0.0f;              // Hold time (for gates)
    float look_ahead_time_ms = 1.0f;        // Look-ahead time

    // Gain and makeup
    float makeup_gain_db = 0.0f;            // Makeup gain
    float auto_makeup_gain = false;          // Automatic makeup gain
    float target_level_dbfs = -1.0f;         // Target output level
    float ceiling_dbfs = -0.1f;              // Output ceiling
    bool enable_limiting = false;            // Enable limiting mode

    // Detection
    DetectionMode detection_mode = DetectionMode::RMS;
    float detection_window_ms = 10.0f;       // Detection window size
    float averaging_coefficient = 0.999f;    // RMS averaging coefficient
    float peak_hold_time_ms = 0.0f;         // Peak hold time
    bool enable_peak_detection = true;      // Enable peak detection

    // Sidechain
    bool enable_sidechain = false;          // Enable sidechain processing
    SidechainFilterType sidechain_filter_type = SidechainFilterType::NONE;
    float sidechain_freq_hz = 1000.0f;       // Sidechain filter frequency
    float sidechain_q = 1.0f;               // Sidechain filter Q
    float sidechain_gain_db = 0.0f;          // Sidechain gain
    bool enable_ducking = false;             // Enable ducking mode
    float ducking_threshold_dbfs = -20.0f;   // Ducking threshold
    float ducking_depth_db = 10.0f;          // Ducking depth

    // Multi-band processing
    bool enable_multiband = false;          // Enable multi-band processing
    uint32_t num_bands = 3;                 // Number of bands
    std::vector<float> band_frequencies_hz = {250.0f, 1000.0f, 4000.0f, 12000.0f};
    std::vector<float> band_thresholds_dbfs = {-20.0f, -18.0f, -16.0f, -15.0f};
    std::vector<float> band_ratios = {4.0f, 3.5f, 3.0f, 2.5f};
    std::vector<float> band_attack_ms = {5.0f, 3.0f, 2.0f, 1.0f};
    std::vector<float> band_release_ms = {100.0f, 80.0f, 60.0f, 40.0f};
    bool crossover_linear_phase = true;      // Linear-phase crossovers

    // Advanced parameters
    bool enable_adaptive_mode = false;       // Adaptive processing
    float adaptation_speed = 0.1f;           // Adaptation speed
    float program_dependency = 0.5f;        // Program dependency
    bool enable_parallel = false;           // Parallel processing
    float parallel_mix = 0.5f;               // Parallel mix ratio
    bool enable_upward = false;             // Upward compression

    // Noise gate specific
    float gate_threshold_dbfs = -40.0f;     // Gate threshold
    float gate_range_db = -60.0f;           // Gate range
    float gate_hysteresis_db = 2.0f;        // Gate hysteresis
    float gate_open_time_ms = 1.0f;         // Gate open time
    float gate_close_time_ms = 100.0f;       // Gate close time

    // De-esser specific
    float deesser_frequency_hz = 4000.0f;   // De-esser frequency
    float deesser_bandwidth_hz = 2000.0f;   // De-esser bandwidth
    float deesser_threshold_dbfs = -6.0f;   // De-esser threshold
    float deesser_ratio = 4.0f;             // De-esser ratio
    bool deesser_listen_mode = false;        // De-esser listen mode

    // Limiter specific
    float limiter_release_time_ms = 10.0f;  // Limiter release time
    bool limiter_brickwall = false;         // Brickwall limiting
    float limiter_ceiling_dbfs = -0.1f;     // Limiter ceiling

    // Metering and display
    bool enable_gain_reduction_meter = true;
    bool enable_input_output_meter = true;
    bool enable_envelope_follower = true;
    bool enable_histogram = false;
    uint32_t meter_update_rate_hz = 60;

    // Quality and performance
    bool high_precision_mode = true;         // 64-bit processing
    bool enable_dithering = false;          // Output dithering
    float dither_noise_dbfs = -120.0f;      // Dither noise level
    bool enable_clipping_protection = true; // Clipping protection
    float oversampling_factor = 1.0f;       // Oversampling factor

    // Real-time parameters
    float parameter_smooth_time_ms = 20.0f;  // Parameter smoothing
    bool enable_zero_latency = false;        // Zero latency mode
    float max_lookahead_samples = 0;         // Max lookahead samples
};

/**
 * Abstract base class for dynamic range processors
 */
class DynamicRangeProcessor {
public:
    virtual ~DynamicRangeProcessor() = default;

    // Basic lifecycle
    virtual bool initialize(uint32_t sample_rate, uint32_t channels, uint32_t max_frame_size) = 0;
    virtual void shutdown() = 0;
    virtual bool reset() = 0;

    // Processing
    virtual bool process(const float* input, float* output, uint32_t frame_count) = 0;
    virtual bool process_interleaved(const float* input, float* output, uint32_t frame_count) = 0;
    virtual bool process_with_sidechain(const float* input, const float* sidechain,
                                         float* output, uint32_t frame_count) = 0;

    // Parameters
    virtual bool set_parameters(const DynamicRangeParameters& params) = 0;
    virtual DynamicRangeParameters get_parameters() const = 0;
    virtual bool set_parameter(const std::string& name, float value) = 0;
    virtual float get_parameter(const std::string& name) const = 0;

    // Real-time controls
    virtual bool set_threshold(float threshold_dbfs) = 0;
    virtual bool set_ratio(float ratio) = 0;
    virtual bool set_attack_time(float attack_ms) = 0;
    virtual bool set_release_time(float release_ms) = 0;
    virtual bool set_makeup_gain(float gain_db) = 0;
    virtual bool set_knee_width(float width_db) = 0;

    // Effect controls
    virtual bool set_bypass(bool bypass) = 0;
    virtual bool is_bypassed() const = 0;
    virtual bool set_enabled(bool enabled) = 0;
    virtual bool is_enabled() const = 0;

    // Presets
    virtual bool save_preset(const std::string& name) = 0;
    virtual bool load_preset(const std::string& name) = 0;
    virtual std::vector<std::string> get_available_presets() const = 0;

    // Information
    virtual DynamicRangeType get_type() const = 0;
    virtual CompressionModel get_model() const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_version() const = 0;
    virtual std::string get_description() const = 0;
    virtual DynamicRangeStatistics get_statistics() const = 0;
    virtual void reset_statistics() = 0;

    // Advanced features
    virtual bool supports_real_time_parameter_changes() const = 0;
    virtual bool supports_gpu_acceleration() const = 0;
    virtual bool prefers_gpu_processing() const { return false; }
    virtual bool is_multiband() const = 0;
    virtual bool has_sidechain() const = 0;
    virtual double get_expected_latency_ms() const = 0;

    // Metering
    virtual float get_current_gain_reduction() const = 0;
    virtual float get_input_level() const = 0;
    virtual float get_output_level() const = 0;
    virtual float get_envelope_follower_value() const = 0;
    virtual bool is_clipping() const = 0;
};

/**
 * Core compressor implementation
 */
class Compressor : public DynamicRangeProcessor {
public:
    Compressor();
    virtual ~Compressor();

    // DynamicRangeProcessor interface
    bool initialize(uint32_t sample_rate, uint32_t channels, uint32_t max_frame_size) override;
    void shutdown() override;
    bool reset() override;

    bool process(const float* input, float* output, uint32_t frame_count) override;
    bool process_interleaved(const float* input, float* output, uint32_t frame_count) override;
    bool process_with_sidechain(const float* input, const float* sidechain,
                                 float* output, uint32_t frame_count) override;

    bool set_parameters(const DynamicRangeParameters& params) override;
    DynamicRangeParameters get_parameters() const override;
    bool set_parameter(const std::string& name, float value) override;
    float get_parameter(const std::string& name) const override;

    bool set_threshold(float threshold_dbfs) override;
    bool set_ratio(float ratio) override;
    bool set_attack_time(float attack_ms) override;
    bool set_release_time(float release_ms) override;
    bool set_makeup_gain(float gain_db) override;
    bool set_knee_width(float width_db) override;

    bool set_bypass(bool bypass) override;
    bool is_bypassed() const override;
    bool set_enabled(bool enabled) override;
    bool is_enabled() const override;

    bool save_preset(const std::string& name) override;
    bool load_preset(const std::string& name) override;
    std::vector<std::string> get_available_presets() const override;

    DynamicRangeType get_type() const override;
    CompressionModel get_model() const override;
    std::string get_name() const override;
    std::string get_version() const override;
    std::string get_description() const override;
    DynamicRangeStatistics get_statistics() const override;
    void reset_statistics() override;

    bool supports_real_time_parameter_changes() const override;
    bool supports_gpu_acceleration() const override;
    bool is_multiband() const override;
    bool has_sidechain() const override;
    double get_expected_latency_ms() const override;

    // Metering
    float get_current_gain_reduction() const override;
    float get_input_level() const override;
    float get_output_level() const override;
    float get_envelope_follower_value() const override;
    bool is_clipping() const override;

private:
    DynamicRangeParameters params_;
    DynamicRangeType type_;
    CompressionModel model_;
    bool bypassed_;
    bool enabled_;

    uint32_t sample_rate_;
    uint32_t channels_;
    uint32_t max_frame_size_;

    // Processing state
    float current_gain_reduction_db_;
    float envelope_value_;
    float input_level_dbfs_;
    float output_level_dbfs_;
    std::vector<float> channel_gain_reductions_;
    std::vector<float> channel_envelopes_;

    // Attack/release coefficients
    float attack_coefficient_;
    float release_coefficient_;
    float smoothing_coefficient_;

    // Look-ahead buffer
    std::vector<float> look_ahead_buffer_;
    std::vector<float> look_ahead_delay_buffer_;
    uint32_t look_ahead_samples_;
    uint32_t look_ahead_index_;

    // Sidechain processing
    std::vector<float> sidechain_buffer_;
    std::vector<float> sidechain_filter_state_;
    float sidechain_level_dbfs_;

    // Statistics
    mutable std::mutex stats_mutex_;
    DynamicRangeStatistics statistics_;

    // Processing buffers
    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;
    std::vector<float> gain_buffer_;

    // Preset management
    std::unordered_map<std::string, DynamicRangeParameters> presets_;

    // Internal methods
    void calculate_attack_release_coefficients();
    float calculate_gain_reduction(float input_level_db, float threshold_db, float ratio, float knee_width);
    void apply_soft_knee(float input_level_db, float threshold_db, float knee_width,
                         float& new_threshold, float& new_ratio);
    float process_sidechain_filter(float input, uint32_t channel);
    void update_statistics(const float* input, const float* output, uint32_t frame_count);
    float smooth_parameter(float current, float target, float coefficient);
    bool check_clipping(const float* buffer, uint32_t frame_count);
};

/**
 * Multi-band compressor implementation
 */
class MultiBandCompressor : public DynamicRangeProcessor {
public:
    MultiBandCompressor();
    virtual ~MultiBandCompressor();

    // DynamicRangeProcessor interface
    bool initialize(uint32_t sample_rate, uint32_t channels, uint32_t max_frame_size) override;
    void shutdown() override;
    bool reset() override;

    bool process(const float* input, float* output, uint32_t frame_count) override;
    bool process_interleaved(const float* input, float* output, uint32_t frame_count) override;
    bool process_with_sidechain(const float* input, const float* sidechain,
                                 float* output, uint32_t frame_count) override;

    bool set_parameters(const DynamicRangeParameters& params) override;
    DynamicRangeParameters get_parameters() const override;
    bool set_parameter(const std::string& name, float value) override;
    float get_parameter(const std::string& name) const override;

    bool set_threshold(float threshold_dbfs) override;
    bool set_ratio(float ratio) override;
    bool set_attack_time(float attack_ms) override;
    bool set_release_time(float release_ms) override;
    bool set_makeup_gain(float gain_db) override;
    bool set_knee_width(float width_db) override;

    bool set_bypass(bool bypass) override;
    bool is_bypassed() const override;
    bool set_enabled(bool enabled) override;
    bool is_enabled() const override;

    DynamicRangeType get_type() const override;
    CompressionModel get_model() const override;
    std::string get_name() const override;
    std::string get_version() const override;
    std::string get_description() const override;
    DynamicRangeStatistics get_statistics() const override;
    void reset_statistics() override;

    bool supports_real_time_parameter_changes() const override;
    bool supports_gpu_acceleration() const override;
    bool is_multiband() const override;
    bool has_sidechain() const override;
    double get_expected_latency_ms() const override;

    // Multi-band specific
    void set_num_bands(uint32_t num_bands);
    void set_band_parameters(uint32_t band_index, const DynamicRangeParameters& band_params);
    void set_band_crossover_frequency(uint32_t crossover_index, float frequency_hz);
    uint32_t get_num_bands() const;
    float get_band_gain_reduction(uint32_t band_index) const;

private:
    struct BandProcessor {
        std::unique_ptr<Compressor> compressor;
        std::vector<float> crossover_filters;
        std::vector<float> band_buffer;
        float frequency_hz;
        float current_gain_reduction;
        bool enabled;
    };

    DynamicRangeParameters params_;
    DynamicRangeType type_;
    CompressionModel model_;
    bool bypassed_;
    bool enabled_;

    uint32_t sample_rate_;
    uint32_t channels_;
    uint32_t max_frame_size_;

    // Band processors
    std::vector<BandProcessor> bands_;

    // Crossover filters
    std::vector<std::vector<float>> crossover_filters_;
    std::vector<std::vector<float>> crossover_states_;

    // Processing buffers
    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;
    std::vector<float> band_output_buffer_;

    // Statistics
    mutable std::mutex stats_mutex_;
    DynamicRangeStatistics statistics_;

    // Internal methods
    void initialize_bands();
    void update_crossover_filters();
    void apply_crossover_filter(const float* input, float* output, uint32_t frame_count,
                                uint32_t channel, float frequency_hz, bool is_low_pass);
    void process_band(const float* input, float* output, uint32_t frame_count,
                     uint32_t band_index);
    void combine_bands(uint32_t frame_count);
    float calculate_band_gain_reduction(uint32_t band_index) const;
};

/**
 * Limiter implementation
 */
class Limiter : public DynamicRangeProcessor {
public:
    Limiter();
    virtual ~Limiter();

    // DynamicRangeProcessor interface
    bool initialize(uint32_t sample_rate, uint32_t channels, uint32_t max_frame_size) override;
    void shutdown() override;
    bool reset() override;

    bool process(const float* input, float* output, uint32_t frame_count) override;
    bool process_interleaved(const float* input, float* output, uint32_t frame_count) override;
    bool process_with_sidechain(const float* input, const float* sidechain,
                                 float* output, uint32_t frame_count) override;

    bool set_parameters(const DynamicRangeParameters& params) override;
    DynamicRangeParameters get_parameters() const override;
    bool set_parameter(const std::string& name, float value) override;
    float get_parameter(const std::string& name) const override;

    bool set_threshold(float threshold_dbfs) override;
    bool set_ratio(float ratio) override;
    bool set_attack_time(float attack_ms) override;
    bool set_release_time(float release_ms) override;
    bool set_makeup_gain(float gain_db) override;
    bool set_knee_width(float width_db) override;

    bool set_bypass(bool bypass) override;
    bool is_bypassed() const override;
    bool set_enabled(bool enabled) override;
    bool is_enabled() const override;

    DynamicRangeType get_type() const override;
    CompressionModel get_model() const override;
    std::string get_name() const override;
    std::string get_version() const override;
    std::string get_description() const override;
    DynamicRangeStatistics get_statistics() const override;
    void reset_statistics() override;

    bool supports_real_time_parameter_changes() const override;
    bool supports_gpu_acceleration() const override;
    bool is_multiband() const override;
    bool has_sidechain() const override;
    double get_expected_latency_ms() const override;

    // Limiter specific
    void set_ceiling(float ceiling_dbfs);
    void set_release_time_ms(float release_ms);
    void set_brickwall_mode(bool enabled);
    bool is_brickwall_mode() const;

private:
    DynamicRangeParameters params_;
    DynamicRangeType type_;
    CompressionModel model_;
    bool bypassed_;
    bool enabled_;

    uint32_t sample_rate_;
    uint32_t channels_;
    uint32_t max_frame_size_;

    // Limiter state
    float ceiling_dbfs_;
    float current_gain_db_;
    float envelope_value_;
    bool brickwall_mode_;

    // Attack/release coefficients
    float attack_coefficient_;
    float release_coefficient_;

    // Look-ahead for brickwall limiting
    std::vector<float> look_ahead_buffer_;
    uint32_t look_ahead_samples_;
    uint32_t look_ahead_index_;

    // Statistics
    mutable std::mutex stats_mutex_;
    DynamicRangeStatistics statistics_;

    // Processing buffers
    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;

    // Internal methods
    float calculate_limiter_gain(float input_level_db, float ceiling_db, bool brickwall);
    void apply_brickwall_limiting(const float* input, float* output, uint32_t frame_count);
    void update_attack_release_coefficients();
};

/**
 * Dynamic range effects factory
 */
class DynamicRangeEffectsFactory {
public:
    // Compressors
    static std::unique_ptr<DynamicRangeProcessor> create_compressor();
    static std::unique_ptr<DynamicRangeProcessor> create_vocal_compressor();
    static std::unique_ptr<DynamicRangeProcessor> create_mastering_compressor();
    static std::unique_ptr<DynamicRangeProcessor> create_optical_compressor();
    static std::unique_ptr<DynamicRangeProcessor> create_fet_compressor();
    static std::unique_ptr<DynamicRangeProcessor> create_varimu_compressor();

    // Multi-band processors
    static std::unique_ptr<DynamicRangeProcessor> create_multiband_compressor(uint32_t bands = 3);
    static std::unique_ptr<DynamicRangeProcessor> create_dynamic_equalizer();

    // Limiters
    static std::unique_ptr<DynamicRangeProcessor> create_limiter();
    static std::unique_ptr<DynamicRangeProcessor> create_brickwall_limiter();
    static std::unique_ptr<DynamicRangeProcessor> create_maximizer();

    // Gates and expanders
    static std::unique_ptr<DynamicRangeProcessor> create_noise_gate();
    static std::unique_ptr<DynamicRangeProcessor> create_expander();
    static std::unique_ptr<DynamicRangeProcessor> create_upward_expander();

    // Specialized processors
    static std::unique_ptr<DynamicRangeProcessor> create_deesser();
    static std::unique_ptr<DynamicRangeProcessor> create_ducker();
    static std::unique_ptr<DynamicRangeProcessor> create_transient_shaper();
    static std::unique_ptr<DynamicRangeProcessor> create_parallel_compressor();

    // Utility methods
    static std::vector<std::string> get_available_effect_types();
    static std::vector<std::string> get_available_compression_models();
    static std::vector<std::string> get_available_knee_types();
    static std::vector<std::string> get_available_detection_modes();
};

/**
 * Utility functions for dynamic range processing
 */
namespace dynamic_range_utils {
    // Level conversion utilities
    float linear_to_db(float linear);
    float db_to_linear(float db);
    float dbfs_to_linear(float dbfs);
    float linear_to_dbfs(float linear);
    float rms_to_db(float rms, uint32_t num_samples);
    float peak_to_db(float peak);

    // Time constant utilities
    float time_constant_to_coefficient(float time_ms, uint32_t sample_rate);
    float coefficient_to_time_constant(float coefficient, uint32_t sample_rate);
    float attack_time_to_coefficient(float attack_ms, uint32_t sample_rate);
    float release_time_to_coefficient(float release_ms, uint32_t sample_rate);

    // Transfer function utilities
    float calculate_compression_gain(float input_db, float threshold_db, float ratio, float knee_width);
    float calculate_expansion_gain(float input_db, float threshold_db, float ratio, float range_db);
    float calculate_limiter_gain(float input_db, float ceiling_db);
    float calculate_gate_gain(float input_db, float threshold_db, float range_db, float hysteresis);

    // Knee function utilities
    float hard_knee(float input_db, float threshold_db);
    float soft_knee(float input_db, float threshold_db, float knee_width);
    float exponential_knee(float input_db, float threshold_db, float knee_width);
    float adaptive_knee(float input_db, float threshold_db, float knee_width, float signal_level);

    // Detection utilities
    float calculate_rms(const float* buffer, uint32_t frame_count);
    float calculate_peak(const float* buffer, uint32_t frame_count);
    float calculate_weighted_peak(const float* buffer, uint32_t frame_count, float weighting);
    float calculate_percentile(const float* buffer, uint32_t frame_count, float percentile);
    float calculate_envelope(const float* buffer, uint32_t frame_count, float attack_coeff, float release_coeff);

    // Sidechain utilities
    void apply_sidechain_filter(float* buffer, uint32_t frame_count, float frequency_hz,
                                float q, float sample_rate, SidechainFilterType filter_type);
    void calculate_lowpass_coefficients(float frequency_hz, float sample_rate, float q,
                                         float& b0, float& b1, float& b2, float& a1, float& a2);
    void calculate_highpass_coefficients(float frequency_hz, float sample_rate, float q,
                                          float& b0, float& b1, float& b2, float& a1, float& a2);

    // Crossover filter utilities
    void calculate_linkwitz_riley_crossover(float frequency_hz, float sample_rate,
                                             std::vector<float>& lowpass_coeffs,
                                             std::vector<float>& highpass_coeffs);
    void apply_crossover_filter(const float* input, float* low_output, float* high_output,
                                uint32_t frame_count, const float* coeffs, float* state);

    // Envelope follower utilities
    class EnvelopeFollower {
    public:
        EnvelopeFollower(float attack_coeff, float release_coeff, float hold_coeff = 0.0f);
        float process(float input);
        void reset();
        void set_attack_coefficient(float coeff);
        void set_release_coefficient(float coeff);
        void set_hold_coefficient(float coeff);

    private:
        float attack_coeff_;
        float release_coeff_;
        float hold_coeff_;
        float envelope_;
        float peak_value_;
        uint32_t hold_counter_;
    };

    // Look-ahead buffer utilities
    class LookAheadBuffer {
    public:
        LookAheadBuffer(uint32_t samples, uint32_t channels);
        ~LookAheadBuffer();

        void write(const float* input, uint32_t frame_count);
        void read(float* output, uint32_t frame_count);
        void write_sample(uint32_t channel, float sample);
        float read_sample(uint32_t channel);
        void clear();
        uint32_t get_size() const { return size_; }
        uint32_t get_channels() const { return channels_; }

    private:
        std::vector<std::vector<float>> buffer_;
        uint32_t size_;
        uint32_t channels_;
        std::vector<uint32_t> write_pos_;
        std::vector<uint32_t> read_pos_;
    };

    // Performance optimization utilities
    bool is_sse_supported();
    bool is_avx_supported();
    void* aligned_malloc(size_t size, size_t alignment = 16);
    void aligned_free(void* ptr);
    void apply_vectorized_gain(float* buffer, uint32_t frame_count, float gain);
    void apply_vectorized_compression(float* input, float* output, uint32_t frame_count, float gain);
    bool check_clipping_sse(const float* buffer, uint32_t frame_count);

    // Validation utilities
    bool validate_compression_parameters(const DynamicRangeParameters& params);
    bool validate_threshold(float threshold_dbfs);
    bool validate_ratio(float ratio);
    bool validate_time_constants(float attack_ms, float release_ms);
    bool is_valid_frequency_range(float freq_hz, uint32_t sample_rate);
}

} // namespace vortex::core::dsp