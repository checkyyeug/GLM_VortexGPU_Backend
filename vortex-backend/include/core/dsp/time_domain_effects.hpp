#pragma once

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <vector>
#include <string>
#include <unordered_map>
#include <queue>
#include <functional>
#include <array>
#include <deque>
#include <complex>

#include "system/logger.hpp"
#include "core/dsp/realtime_effects_chain.hpp"

namespace vortex::core::dsp {

/**
 * Time-Domain Audio Effects Processor for Vortex GPU Audio Backend
 *
 * This component provides high-quality time-domain audio effects with ultra-low
 * latency processing, suitable for professional audio applications. It supports
 * various time-based effects including delays, echoes, reverbs, and modulation
 * effects with CPU/GPU acceleration and real-time parameter control.
 *
 * Features:
 * - Sub-millisecond latency time-domain processing
 * - High-fidelity delay lines with interpolation
 * - Multi-tap delay and echo effects
 * - Algorithmic and convolution reverb
 * - Modulation effects (chorus, flanger, phaser)
 * - Pitch shifting and time stretching
 * - Dynamic range processing
 * - Real-time parameter smoothing
 * - GPU-accelerated processing where applicable
 * - Multi-channel support
 * - Professional audio quality (64-bit internal processing)
 * - Zero-latency monitoring options
 */

/**
 * Time-domain effect types
 */
enum class TimeDomainEffectType {
    DELAY,            // Simple delay/echo
    MULTI_TAP_DELAY,  // Multi-tap delay
    PING_PONG_DELAY,  // Ping-pong stereo delay
    REVERB,           // Algorithmic reverb
    CONVOLUTION_REVERB, // Convolution reverb
    CHORUS,           // Chorus effect
    FLANGER,          // Flanger effect
    PHASER,           // Phaser effect
    VIBRATO,          // Vibrato effect
    TREMOLO,          // Tremolo effect
    PITCH_SHIFTER,    // Pitch shifting
    TIME_STRETCH,     // Time stretching
    DOUBLER,          // Vocal doubler
    SLAPBACK_DELAY,   // Slapback delay
    FILTER_DELAY,     // Delay with filtering
    REVERSE_DELAY,    // Reverse delay
    SAMPLE_HOLD,      // Sample and hold
    RING_MODULATOR,   // Ring modulation
    AUTO_PAN,         // Auto-panning
    GATED_REVERB,     // Gated reverb
    SPRING_REVERB,    // Spring reverb simulation
    PLATE_REVERB,     // Plate reverb
    HALL_REVERB,      // Hall reverb
    ROOM_REVERB,      // Room reverb
    CUSTOM            // Custom time-domain effect
};

/**
 * Delay line interpolation types
 */
enum class InterpolationType {
    NONE,              // No interpolation (stepped)
    LINEAR,            // Linear interpolation
    COSINE,            // Cosine interpolation
    CUBIC,             // Cubic spline interpolation
    HERMITE,           // Hermite interpolation
    LAGRANGE,          // Lagrange interpolation
    ALLPASS,           // All-pass interpolation
    THIRAN,           // Thiran all-pass interpolation
    WINDOWED_SINC     // Windowed sinc interpolation
};

/**
 * Reverb algorithm types
 */
enum class ReverbAlgorithm {
    FREEVERB,          // Freeverb algorithm
    SCHROEDER,        // Schroeder reverb
    JOT_REVERB,       // Jot reverberator
    DATTORRO,         // Dattorro reverb
    CONVOLUTION,      // Convolution reverb
    FDN,              // Feedback delay network
    DIFFUSE_LOOPS,    // Diffuse loops
    PARTIAL,          // Partial convolution
    WAVEGUIDE,        // Waveguide reverb
    SPRING_SIMULATION // Spring reverb simulation
};

/**
 * Modulation waveform types
 */
enum class ModulationWaveform {
    SINE,             // Sine wave
    TRIANGLE,         // Triangle wave
    SAWTOOTH,         // Sawtooth wave
    SQUARE,           // Square wave
    SAMPLE_AND_HOLD,  // Sample and hold
    NOISE,            // Random noise
    EXPONENTIAL,      // Exponential curve
    LOGARITHMIC,      // Logarithmic curve
    CUSTOM            // Custom waveform
};

/**
 * Time-domain effect parameters
 */
struct TimeDomainParameters {
    // Delay parameters
    float delay_time_ms = 250.0f;
    float feedback_percent = 30.0f;
    float wet_mix_percent = 30.0f;
    float dry_mix_percent = 70.0f;
    float delay_filter_frequency = 20000.0f; // High-pass frequency for delay
    float delay_filter_resonance = 0.7f;
    bool delay_filter_enabled = false;

    // Multi-tap delay parameters
    std::vector<float> tap_delay_times_ms = {250.0f, 500.0f, 750.0f};
    std::vector<float> tap_gain_levels = {1.0f, 0.7f, 0.5f};
    std::vector<float> tap_feedback_levels = {0.3f, 0.2f, 0.1f};
    uint32_t num_taps = 3;

    // Reverb parameters
    float room_size = 0.5f;           // Room size (0-1)
    float damping = 0.5f;             // High-frequency damping
    float width = 1.0f;               // Stereo width
    float predelay_ms = 20.0f;        // Pre-delay time
    float diffusion = 0.7f;           // Diffusion
    float early_reflections_level = 0.3f; // Early reflections level
    float tail_level = 0.4f;          // Reverb tail level
    ReverbAlgorithm reverb_algorithm = ReverbAlgorithm::FREEVERB;

    // Convolution reverb parameters
    std::string impulse_response_path = "";
    float ir_gain_db = 0.0f;
    float ir_stretch = 1.0f;
    bool ir_reverse = false;

    // Modulation parameters
    float modulation_rate_hz = 1.0f;
    float modulation_depth = 0.5f;
    float modulation_phase = 0.0f;
    ModulationWaveform modulation_waveform = ModulationWaveform::SINE;
    float feedback_modulation = 0.0f;

    // Chorus/Flanger specific
    float chorus_delay_ms = 20.0f;
    float flanger_delay_ms = 1.0f;
    float stereo_spread_ms = 0.5f;

    // Phaser specific
    uint32_t phaser_stages = 8;
    float phaser_frequency = 1.0f;
    float phaser_spread = 0.5f;
    float phaser_feedback = 0.7f;

    // Pitch shifting parameters
    float pitch_shift_semitones = 0.0f;
    float pitch_shift_window_ms = 50.0f;
    float pitch_shift_crossfade_ms = 10.0f;
    bool pitch_shift_formant_preserve = true;

    // Time stretching parameters
    float time_stretch_ratio = 1.0f;
    bool time_stretch_preserve_pitch = true;
    float time_stretch_window_ms = 100.0f;

    // Tremolo/Vibrato parameters
    float tremolo_depth = 0.5f;
    float vibrato_depth = 0.1f;
    float vibrato_rate_hz = 5.0f;

    // Filter parameters
    float filter_cutoff_frequency = 1000.0f;
    float filter_resonance = 1.0f;
    float filter_drive = 0.0f;
    bool filter_enabled = false;

    // Stereo processing
    bool stereo_mode = true;
    float stereo_width = 1.0f;
    float stereo_pan = 0.0f; // -1 = left, 0 = center, 1 = right

    // Quality and performance
    bool high_quality_mode = true;
    InterpolationType interpolation_type = InterpolationType::CUBIC;
    uint32_t oversampling_factor = 1;
    bool enable_dc_blocking = true;

    // Advanced parameters
    float saturation_drive = 0.0f;
    float saturation_mix = 0.0f;
    bool enable_analog_simulation = false;
    float noise_floor_db = -120.0f;
    bool enable_dithering = false;

    // Real-time controls
    float tempo_sync_bpm = 120.0f;
    bool tempo_sync_enabled = false;
    uint32_t tempo_sync_numerator = 4;
    uint32_t tempo_sync_denominator = 4;
};

/**
 * Time-domain effect processing statistics
 */
struct TimeDomainStatistics {
    uint64_t total_process_calls = 0;
    uint64_t successful_calls = 0;
    double avg_processing_time_us = 0.0;
    double max_processing_time_us = 0.0;
    double min_processing_time_us = std::numeric_limits<double>::max();
    uint64_t buffer_underruns = 0;
    uint64_t buffer_overruns = 0;
    double cpu_utilization_percent = 0.0;
    double gpu_utilization_percent = 0.0;
    uint32_t delay_line_size_samples = 0;
    float memory_usage_mb = 0.0f;
    uint64_t convolution_operations = 0;
    uint64_t interpolation_operations = 0;
    std::chrono::steady_clock::time_point last_reset_time;
    bool is_active = false;
};

/**
 * Abstract base class for time-domain effects
 */
class TimeDomainEffect {
public:
    virtual ~TimeDomainEffect() = default;

    // Basic lifecycle
    virtual bool initialize(uint32_t sample_rate, uint32_t channels, uint32_t max_frame_size) = 0;
    virtual void shutdown() = 0;
    virtual bool reset() = 0;

    // Processing
    virtual bool process(const float* input, float* output, uint32_t frame_count) = 0;
    virtual bool process_interleaved(const float* input, float* output, uint32_t frame_count) = 0;

    // Parameters
    virtual bool set_parameters(const TimeDomainParameters& params) = 0;
    virtual TimeDomainParameters get_parameters() const = 0;
    virtual bool set_parameter(const std::string& name, float value) = 0;
    virtual float get_parameter(const std::string& name) const = 0;

    // Effect-specific controls
    virtual bool set_bypass(bool bypass) = 0;
    virtual bool is_bypassed() const = 0;
    virtual bool set_dry_wet_mix(float mix) = 0;

    // Presets
    virtual bool save_preset(const std::string& name) = 0;
    virtual bool load_preset(const std::string& name) = 0;
    virtual std::vector<std::string> get_available_presets() const = 0;

    // Information
    virtual TimeDomainEffectType get_type() const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_version() const = 0;
    virtual std::string get_description() const = 0;
    virtual TimeDomainStatistics get_statistics() const = 0;
    virtual void reset_statistics() = 0;

    // Advanced features
    virtual bool supports_real_time_parameter_changes() const = 0;
    virtual bool supports_gpu_acceleration() const = 0;
    virtual bool prefers_gpu_processing() const { return false; }
    virtual uint32_t get_delay_samples() const = 0;
    virtual double get_expected_latency_ms() const = 0;
};

/**
 * Delay line class with interpolation
 */
class DelayLine {
public:
    DelayLine();
    ~DelayLine();

    bool initialize(uint32_t max_delay_samples, uint32_t channels, InterpolationType interpolation);
    void shutdown();
    void clear();

    // Core processing
    float read_sample(uint32_t channel, float delay_samples) const;
    void write_sample(uint32_t channel, float sample);
    void process_block(const float* input, float* output, uint32_t frame_count,
                      const std::vector<float>& delay_times);

    // Controls
    void set_max_delay_samples(uint32_t max_delay_samples);
    void set_interpolation_type(InterpolationType type);
    void set_feedback(float feedback);
    void set_wet_mix(float mix);
    void set_filter_enabled(bool enabled);
    void set_filter_parameters(float frequency, float resonance);

    // State
    float get_feedback() const { return feedback_; }
    float get_wet_mix() const { return wet_mix_; }
    bool is_filter_enabled() const { return filter_enabled_; }
    uint32_t get_delay_line_size() const { return delay_line_size_; }

private:
    std::vector<std::vector<float>> delay_buffers_;
    std::vector<uint32_t> write_positions_;
    std::vector<std::vector<float>> filter_states_;

    uint32_t max_delay_samples_;
    uint32_t channels_;
    uint32_t delay_line_size_;
    InterpolationType interpolation_type_;

    float feedback_;
    float wet_mix_;

    // Filter parameters
    bool filter_enabled_;
    float filter_frequency_;
    float filter_resonance_;
    std::vector<float> filter_coefficients_;

    // Interpolation methods
    float interpolate_linear(uint32_t channel, float delay_samples) const;
    float interpolate_cubic(uint32_t channel, float delay_samples) const;
    float interpolate_hermite(uint32_t channel, float delay_samples) const;
    float interpolate_thiran(uint32_t channel, float delay_samples) const;

    // Filter processing
    void process_filter(uint32_t channel, float& sample);
    void update_filter_coefficients();
};

/**
 * Multi-tap delay processor
 */
class MultiTapDelayProcessor : public TimeDomainEffect {
public:
    MultiTapDelayProcessor();
    ~MultiTapDelayProcessor() override;

    // TimeDomainEffect interface
    bool initialize(uint32_t sample_rate, uint32_t channels, uint32_t max_frame_size) override;
    void shutdown() override;
    bool reset() override;

    bool process(const float* input, float* output, uint32_t frame_count) override;
    bool process_interleaved(const float* input, float* output, uint32_t frame_count) override;

    bool set_parameters(const TimeDomainParameters& params) override;
    TimeDomainParameters get_parameters() const override;
    bool set_parameter(const std::string& name, float value) override;
    float get_parameter(const std::string& name) const override;

    bool set_bypass(bool bypass) override;
    bool is_bypassed() const override;
    bool set_dry_wet_mix(float mix) override;

    bool save_preset(const std::string& name) override;
    bool load_preset(const std::string& name) override;
    std::vector<std::string> get_available_presets() const override;

    TimeDomainEffectType get_type() const override;
    std::string get_name() const override;
    std::string get_version() const override;
    std::string get_description() const override;
    TimeDomainStatistics get_statistics() const override;
    void reset_statistics() override;

    bool supports_real_time_parameter_changes() const override;
    bool supports_gpu_acceleration() const override;
    uint32_t get_delay_samples() const override;
    double get_expected_latency_ms() const override;

    // Multi-tap specific
    void set_num_taps(uint32_t num_taps);
    void set_tap_delay(uint32_t tap_index, float delay_ms);
    void set_tap_gain(uint32_t tap_index, float gain);
    void set_tap_feedback(uint32_t tap_index, float feedback);

private:
    struct DelayTap {
        float delay_time_ms;
        float gain;
        float feedback;
        uint32_t delay_samples;
        std::unique_ptr<DelayLine> delay_line;
    };

    std::vector<DelayTap> taps_;
    TimeDomainParameters parameters_;
    bool bypassed_;
    float dry_wet_mix_;

    uint32_t sample_rate_;
    uint32_t channels_;
    uint32_t max_frame_size_;

    // Statistics
    mutable std::mutex stats_mutex_;
    TimeDomainStatistics statistics_;

    // Processing buffers
    std::vector<float> wet_buffer_;
    std::vector<float> dry_buffer_;
    std::vector<float> tap_buffer_;

    // Preset management
    std::unordered_map<std::string, TimeDomainParameters> presets_;

    // Internal methods
    void update_tap_parameters();
    void process_taps(const float* input, float* output, uint32_t frame_count);
    void apply_dry_wet_mix(const float* dry, const float* wet, float* output, uint32_t frame_count);
};

/**
 * Reverb processor
 */
class ReverbProcessor : public TimeDomainEffect {
public:
    ReverbProcessor();
    ~ReverbProcessor() override;

    // TimeDomainEffect interface
    bool initialize(uint32_t sample_rate, uint32_t channels, uint32_t max_frame_size) override;
    void shutdown() override;
    bool reset() override;

    bool process(const float* input, float* output, uint32_t frame_count) override;
    bool process_interleaved(const float* input, float* output, uint32_t frame_count) override;

    bool set_parameters(const TimeDomainParameters& params) override;
    TimeDomainParameters get_parameters() const override;
    bool set_parameter(const std::string& name, float value) override;
    float get_parameter(const std::string& name) const override;

    bool set_bypass(bool bypass) override;
    bool is_bypassed() const override;
    bool set_dry_wet_mix(float mix) override;

    bool save_preset(const std::string& name) override;
    bool load_preset(const std::string& name) override;
    std::vector<std::string> get_available_presets() const override;

    TimeDomainEffectType get_type() const override;
    std::string get_name() const override;
    std::string get_version() const override;
    std::string get_description() const override;
    TimeDomainStatistics get_statistics() const override;
    void reset_statistics() override;

    bool supports_real_time_parameter_changes() const override;
    bool supports_gpu_acceleration() const override;
    uint32_t get_delay_samples() const override;
    double get_expected_latency_ms() const override;

    // Reverb specific
    bool load_impulse_response(const std::string& ir_path);
    bool set_reverb_algorithm(ReverbAlgorithm algorithm);
    void set_room_size(float size);
    void set_damping(float damping);
    void set_width(float width);

private:
    struct FreeverbParams {
        float room_size;
        float damping;
        float width;
        float wet_level;
        float dry_level;
        float mode;
        std::vector<std::vector<float>> comb_filters;
        std::vector<std::vector<float>> allpass_filters;
    };

    struct SchroederParams {
        std::vector<float> comb_delays;
        std::vector<float> comb_feedback;
        std::vector<float> allpass_delays;
        std::vector<float> allpass_feedback;
    };

    TimeDomainParameters parameters_;
    ReverbAlgorithm current_algorithm_;
    bool bypassed_;
    float dry_wet_mix_;

    uint32_t sample_rate_;
    uint32_t channels_;
    uint32_t max_frame_size_;

    // Algorithm-specific parameters
    FreeverbParams freeverb_params_;
    SchroederParams schroeder_params_;

    // Delay lines for reverb algorithms
    std::vector<std::unique_ptr<DelayLine>> comb_delay_lines_;
    std::vector<std::unique_ptr<DelayLine>> allpass_delay_lines_;

    // Convolution reverb
    std::vector<float> impulse_response_;
    std::vector<std::vector<float>> convolution_buffers_;
    uint32_t ir_length_;

    // Statistics
    mutable std::mutex stats_mutex_;
    TimeDomainStatistics statistics_;

    // Processing buffers
    std::vector<float> wet_buffer_;
    std::vector<float> dry_buffer_;
    std::vector<float> temp_buffer_;

    // Preset management
    std::unordered_map<std::string, TimeDomainParameters> presets_;

    // Internal methods
    void initialize_reverb_algorithm();
    void process_freeverb(const float* input, float* output, uint32_t frame_count);
    void process_schroeder(const float* input, float* output, uint32_t frame_count);
    void process_convolution(const float* input, float* output, uint32_t frame_count);
    void update_algorithm_parameters();
    void load_impulse_response_internal(const std::string& ir_path);
};

/**
 * Modulation effects processor (Chorus, Flanger, Phaser)
 */
class ModulationProcessor : public TimeDomainEffect {
public:
    ModulationProcessor();
    ~ModulationProcessor() override;

    // TimeDomainEffect interface
    bool initialize(uint32_t sample_rate, uint32_t channels, uint32_t max_frame_size) override;
    void shutdown() override;
    bool reset() override;

    bool process(const float* input, float* output, uint32_t frame_count) override;
    bool process_interleaved(const float* input, float* output, uint32_t frame_count) override;

    bool set_parameters(const TimeDomainParameters& params) override;
    TimeDomainParameters get_parameters() const override;
    bool set_parameter(const std::string& name, float value) override;
    float get_parameter(const std::string& name) const override;

    bool set_bypass(bool bypass) override;
    bool is_bypassed() const override;
    bool set_dry_wet_mix(float mix) override;

    bool save_preset(const std::string& name) override;
    bool load_preset(const std::string& name) override;
    std::vector<std::string> get_available_presets() const override;

    TimeDomainEffectType get_type() const override;
    std::string get_name() const override;
    std::string get_version() const override;
    std::string get_description() const override;
    TimeDomainStatistics get_statistics() const override;
    void reset_statistics() override;

    bool supports_real_time_parameter_changes() const override;
    bool supports_gpu_acceleration() const override;
    uint32_t get_delay_samples() const override;
    double get_expected_latency_ms() const override;

    // Modulation specific
    void set_modulation_type(TimeDomainEffectType type);
    void set_modulation_waveform(ModulationWaveform waveform);
    void set_lfo_rate(float rate_hz);
    void set_modulation_depth(float depth);

private:
    struct LFO {
        float phase;
        float frequency;
        float depth;
        ModulationWaveform waveform;
        std::vector<float> output_buffer;
    };

    struct AllpassFilter {
        float a1;
        float delay_line_samples;
        std::unique_ptr<DelayLine> delay_line;
    };

    TimeDomainParameters parameters_;
    TimeDomainEffectType modulation_type_;
    ModulationWaveform lfo_waveform_;
    bool bypassed_;
    float dry_wet_mix_;

    uint32_t sample_rate_;
    uint32_t channels_;
    uint32_t max_frame_size_;

    // LFOs for modulation
    std::vector<LFO> lfos_;
    std::vector<float> lfo_phase_offsets_;

    // Delay lines for modulation effects
    std::vector<std::unique_ptr<DelayLine>> delay_lines_;

    // Phaser all-pass filters
    std::vector<std::unique_ptr<AllpassFilter>> allpass_filters_;

    // Statistics
    mutable std::mutex stats_mutex_;
    TimeDomainStatistics statistics_;

    // Processing buffers
    std::vector<float> wet_buffer_;
    std::vector<float> dry_buffer_;
    std::vector<float> mod_buffer_;

    // Preset management
    std::unordered_map<std::string, TimeDomainParameters> presets_;

    // Internal methods
    void initialize_lfos();
    void initialize_delay_lines();
    void initialize_allpass_filters();
    void update_lfo_frequencies();
    void process_chorus(const float* input, float* output, uint32_t frame_count);
    void process_flanger(const float* input, float* output, uint32_t frame_count);
    void process_phaser(const float* input, float* output, uint32_t frame_count);
    void process_vibrato(const float* input, float* output, uint32_t frame_count);
    float generate_lfo_sample(uint32_t lfo_index, float phase_increment);
    float get_lfo_value(ModulationWaveform waveform, float phase);
};

/**
 * Time-domain effects factory
 */
class TimeDomainEffectsFactory {
public:
    // Delay effects
    static std::unique_ptr<TimeDomainEffect> create_simple_delay(float max_delay_ms = 2000.0f);
    static std::unique_ptr<TimeDomainEffect> create_multi_tap_delay(uint32_t num_taps = 4);
    static std::unique_ptr<TimeDomainEffect> create_ping_pong_delay(float max_delay_ms = 1000.0f);
    static std::unique_ptr<TimeDomainEffect> create_slapback_delay();

    // Reverb effects
    static std::unique_ptr<TimeDomainEffect> create_hall_reverb();
    static std::unique_ptr<TimeDomainEffect> create_room_reverb();
    static std::unique_ptr<TimeDomainEffect> create_plate_reverb();
    static std::unique_ptr<TimeDomainEffect> create_spring_reverb();
    static std::unique_ptr<TimeDomainEffect> create_convolution_reverb(const std::string& ir_path = "");

    // Modulation effects
    static std::unique_ptr<TimeDomainEffect> create_chorus();
    static std::unique_ptr<TimeDomainEffect> create_flanger();
    static std::unique_ptr<TimeDomainEffect> create_phaser();
    static std::unique_ptr<TimeDomainEffect> create_vibrato();
    static std::unique_ptr<TimeDomainEffect> create_tremolo();

    // Pitch and time effects
    static std::unique_ptr<TimeDomainEffect> create_pitch_shifter();
    static std::unique_ptr<TimeDomainEffect> create_time_stretch();
    static std::unique_ptr<TimeDomainEffect> create_doubler();

    // Specialized effects
    static std::unique_ptr<TimeDomainEffect> create_auto_pan();
    static std::unique_ptr<TimeDomainEffect> create_ring_modulator();
    static std::unique_ptr<TimeDomainEffect> create_sample_hold();

    // Utility methods
    static std::vector<std::string> get_available_effect_types();
    static std::vector<std::string> get_available_reverb_algorithms();
    static std::vector<std::string> get_available_modulation_waveforms();
};

/**
 * Utility functions for time-domain processing
 */
namespace time_domain_utils {
    // Time conversion utilities
    float milliseconds_to_samples(float time_ms, uint32_t sample_rate);
    float samples_to_milliseconds(uint32_t samples, uint32_t sample_rate);
    float tempo_ms_to_beats(float time_ms, float tempo_bpm);
    float beats_to_tempo_ms(float beats, float tempo_bpm);

    // Delay line utilities
    uint32_t calculate_delay_line_size(float max_delay_ms, uint32_t sample_rate);
    float calculate_feedback_time_constant(float feedback_percent, float sample_rate);
    bool is_power_of_two(uint32_t value);
    uint32_t next_power_of_two(uint32_t value);

    // Interpolation utilities
    float linear_interpolate(float a, float b, float fraction);
    float cubic_interpolate(float a, float b, float c, float d, float fraction);
    float hermite_interpolate(float a, float b, float c, float d, float fraction);
    float thiran_interpolate(float a, float b, float delay_fraction);

    // Filter utilities
    void calculate_low_pass_coefficients(float frequency, float sample_rate, float resonance,
                                         float& a0, float& a1, float& a2, float& b1, float& b2);
    void calculate_high_pass_coefficients(float frequency, float sample_rate, float resonance,
                                          float& a0, float& a1, float& a2, float& b1, float& b2);
    void calculate_band_pass_coefficients(float frequency, float sample_rate, float bandwidth,
                                          float& a0, float& a1, float& a2, float& b1, float& b2);

    // Reverb algorithm utilities
    std::vector<float> generate_comb_filter_tuning(float sample_rate, float spread_factor = 1.0f);
    std::vector<float> generate_allpass_filter_tuning(float sample_rate, float spread_factor = 1.0f);
    void calculate_freeverb_comb_tuning(float sample_rate, std::vector<float>& comb_tuning);
    void calculate_freeverb_allpass_tuning(float sample_rate, std::vector<float>& allpass_tuning);

    // LFO generation
    float generate_sine_wave(float phase);
    float generate_triangle_wave(float phase);
    float generate_sawtooth_wave(float phase);
    float generate_square_wave(float phase, float pulse_width = 0.5f);
    float generate_noise_wave();

    // Audio analysis utilities
    float calculate_correlation(const float* buffer1, const float* buffer2, uint32_t frame_count);
    void calculate_auto_correlation(const float* buffer, float* correlation, uint32_t frame_count, uint32_t max_lag);
    float find_pitch_period(const float* buffer, uint32_t frame_count, uint32_t min_period, uint32_t max_period);
    void apply_window(float* buffer, uint32_t frame_count, const char* window_type = "hann");

    // Convolution utilities
    bool load_impulse_response(const std::string& file_path, std::vector<float>& ir_buffer,
                              uint32_t& ir_length, uint32_t& ir_channels);
    void normalize_impulse_response(std::vector<float>& ir_buffer, float target_level_db = -1.0f);
    void trim_impulse_response(std::vector<float>& ir_buffer, float threshold_db = -60.0f);
    void reverse_impulse_response(std::vector<float>& ir_buffer);

    // Performance optimization utilities
    bool is_sse_supported();
    bool is_avx_supported();
    void* aligned_malloc(size_t size, size_t alignment = 16);
    void aligned_free(void* ptr);
    void apply_vectorized_gain(float* buffer, uint32_t frame_count, float gain);
    void apply_vectorized_mix(float* dest, const float* src, uint32_t frame_count, float mix_ratio);

    // Validation utilities
    bool validate_delay_parameters(const TimeDomainParameters& params);
    bool validate_reverb_parameters(const TimeDomainParameters& params);
    bool validate_modulation_parameters(const TimeDomainParameters& params);
    bool validate_impulse_response(const std::vector<float>& ir_buffer, uint32_t sample_rate);
}

} // namespace vortex::core::dsp