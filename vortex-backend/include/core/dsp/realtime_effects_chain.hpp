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

#include "system/logger.hpp"
#include "hardware/hardware_monitor.hpp"
#include "network/realtime_streaming.hpp"

namespace vortex::core::dsp {

/**
 * Real-time Audio Effects Chain for Vortex GPU Audio Backend
 *
 * This component provides a high-performance, low-latency audio effects processing
 * chain optimized for real-time applications. It supports multiple simultaneous
 * effects with CPU/GPU acceleration, automatic parameter interpolation, and
 * professional audio quality suitable for live performance and recording.
 *
 * Features:
 * - Sub-millisecond latency processing
 * - CPU and GPU acceleration support
 * - Real-time parameter interpolation and automation
 * - Professional audio effects (reverb, delay, EQ, compression, etc.)
 * - Effect chaining and routing with flexible signal flow
 * - Audio-rate modulation capabilities
 * - Preset management and parameter recall
 * - MIDI control integration
 * - Multi-channel processing support
 * - 64-bit internal processing precision
 * - Zero-crossing parameter smoothing
 * - Bypass and dry/wet mixing
 * - Real-time effect switching
 * - Dynamic CPU/GPU load balancing
 */

/**
 * Audio effect types
 */
enum class EffectType {
    EQUALIZER,         // Multi-band equalizer
    COMPRESSOR,        // Dynamic range compressor
    LIMITER,           // Peak limiter
    GATE,              // Noise gate
    EXPANDER,          // Dynamic range expander
    REVERB,            // Reverb (various algorithms)
    DELAY,             // Delay/echo effects
    CHORUS,            // Chorus effect
    FLANGER,           // Flanger effect
    PHASER,            // Phaser effect
    VIBRATO,           // Vibrato effect
    TREMOLO,           // Tremolo effect
    DISTORTION,        // Distortion/saturation
    FILTER,            // Various filter types
    PITCH_SHIFTER,     // Pitch shifting
    TIME_STRETCH,      // Time stretching
    STEREO_ENHANCER,   // Stereo enhancement
    LOUDNESS,          // Loudness maximizer
    DEESSER,           // De-esser
    MULTIBAND_COMP,    // Multi-band compressor
    SATURATION,        // Tape/tube saturation
    BITCRUSHER,        // Bit reduction
    RING_MODULATOR,    // Ring modulation
    AUTO_WAH,          // Auto-wah effect
    VOCODER,           // Vocoder
    CUSTOM             // Custom user-defined effect
};

/**
 * Effect processing modes
 */
enum class ProcessingMode {
    REAL_TIME,         // Real-time processing with lowest latency
    HIGH_QUALITY,      // High quality processing (higher latency)
    POWER_SAVING,      // Power-efficient processing
    GPU_ACCELERATED,   // Force GPU processing
    CPU_OPTIMIZED,     // Force CPU processing
    AUTO               // Automatically select best mode
};

/**
 * Parameter interpolation types
 */
enum class InterpolationType {
    NONE,              // Immediate parameter change
    LINEAR,            // Linear interpolation
    EXPONENTIAL,       // Exponential interpolation
    LOGARITHMIC,       // Logarithmic interpolation
    SMOOTH_STEP,       // Smoothstep interpolation
    SINE,              // Sine-based interpolation
    AUDIO_RATE,        // Audio-rate modulation
    CUSTOM             // Custom interpolation curve
};

/**
 * Audio buffer metadata
 */
struct AudioBufferMetadata {
    uint32_t sample_rate = 0;
    uint32_t channels = 0;
    uint32_t frame_count = 0;
    uint32_t bit_depth = 32;
    bool is_interleaved = true;
    double timestamp_seconds = 0.0;
    uint64_t frame_number = 0;
    bool is_real_time = true;
    float cpu_load_hint = 0.0f;
    float gpu_load_hint = 0.0f;
};

/**
 * Effect parameter definition
 */
struct EffectParameter {
    std::string name;
    std::string display_name;
    float default_value = 0.0f;
    float min_value = 0.0f;
    float max_value = 1.0f;
    float current_value = 0.0f;
    float target_value = 0.0f;
    float step_size = 0.001f;
    InterpolationType interpolation = InterpolationType::LINEAR;
    float interpolation_time_ms = 10.0f;
    bool is_automatable = true;
    bool is_modulatable = false;
    std::string unit = "";
    std::string description = "";

    // Real-time interpolation state
    float interpolation_progress = 0.0f;
    float interpolation_start_value = 0.0f;
    std::chrono::steady_clock::time_point interpolation_start_time;
    bool is_interpolating = false;
};

/**
 * MIDI control mapping
 */
struct MidiMapping {
    uint8_t channel = 0;
    uint8_t control_number = 0;
    std::string parameter_name;
    float min_value = 0.0f;
    float max_value = 1.0f;
    float scaling_factor = 1.0f;
    bool is_relative = false;
    bool is_learn_mode = false;
};

/**
 * Effect processing statistics
 */
struct EffectStatistics {
    uint64_t total_process_calls = 0;
    uint64_t successful_calls = 0;
    double avg_processing_time_us = 0.0;
    double max_processing_time_us = 0.0;
    double min_processing_time_us = std::numeric_limits<double>::max();
    uint64_t buffer_underruns = 0;
    uint64_t buffer_overruns = 0;
    double cpu_utilization_percent = 0.0;
    double gpu_utilization_percent = 0.0;
    std::chrono::steady_clock::time_point last_reset_time;
    bool is_active = false;
};

/**
 * Audio effect interface
 */
class AudioEffect {
public:
    virtual ~AudioEffect() = default;

    // Basic lifecycle
    virtual bool initialize(uint32_t sample_rate, uint32_t channels, uint32_t max_frame_size) = 0;
    virtual void shutdown() = 0;
    virtual bool reset() = 0;

    // Processing
    virtual bool process(const float* input, float* output, uint32_t frame_count) = 0;
    virtual bool process_interleaved(const float* input, float* output, uint32_t frame_count) = 0;

    // Parameters
    virtual bool set_parameter(const std::string& name, float value) = 0;
    virtual float get_parameter(const std::string& name) const = 0;
    virtual std::vector<EffectParameter> get_parameters() const = 0;
    virtual bool automate_parameter(const std::string& name, float target_value, float time_ms) = 0;

    // State management
    virtual bool set_bypass(bool bypass) = 0;
    virtual bool is_bypassed() const = 0;
    virtual bool set_dry_wet_mix(float mix) = 0; // 0.0 = dry, 1.0 = wet

    // MIDI control
    virtual bool add_midi_mapping(const MidiMapping& mapping) = 0;
    virtual bool remove_midi_mapping(const std::string& parameter_name) = 0;
    virtual bool process_midi_message(uint8_t status, uint8_t data1, uint8_t data2) = 0;

    // Presets
    virtual bool save_preset(const std::string& name) = 0;
    virtual bool load_preset(const std::string& name) = 0;
    virtual std::vector<std::string> get_available_presets() const = 0;

    // Information
    virtual EffectType get_type() const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_version() const = 0;
    virtual std::string get_description() const = 0;
    virtual EffectStatistics get_statistics() const = 0;
    virtual void reset_statistics() = 0;

    // Processing options
    virtual bool set_processing_mode(ProcessingMode mode) = 0;
    virtual ProcessingMode get_processing_mode() const = 0;
    virtual bool supports_gpu_acceleration() const = 0;
    virtual bool prefer_gpu_processing() const { return false; }
};

/**
 * Effects chain configuration
 */
struct EffectsChainConfig {
    // Processing settings
    ProcessingMode processing_mode = ProcessingMode::AUTO;
    uint32_t max_frame_size = 4096;
    uint32_t sample_rate = 44100;
    uint32_t channels = 2;
    uint32_t bit_depth = 32;

    // Performance settings
    double max_acceptable_latency_ms = 5.0;      // 5ms target latency
    float max_cpu_utilization_percent = 70.0f;   // CPU usage limit
    float max_gpu_utilization_percent = 80.0f;   // GPU usage limit
    bool enable_gpu_acceleration = true;
    bool enable_load_balancing = true;

    // Interpolation settings
    float default_parameter_smooth_time_ms = 10.0f;
    InterpolationType default_interpolation = InterpolationType::LINEAR;
    bool enable_audio_rate_modulation = true;
    uint32_t modulation_update_rate = 1000;       // Hz

    // Quality settings
    bool enable_high_precision_processing = true;
    bool enable_dithering = false;
    float output_gain_db = 0.0f;
    float input_gain_db = 0.0f;

    // Bypass and routing
    bool enable_global_bypass = false;
    bool enable_per_effect_bypass = true;
    bool enable_parallel_processing = false;
    bool enable_effect_muting = false;

    // Automation and control
    bool enable_parameter_automation = true;
    bool enable_midi_control = true;
    bool enable_osc_control = false;
    uint16_t osc_port = 9000;
    std::string osc_address_prefix = "/vortex/effects";

    // Preset management
    bool enable_preset_management = true;
    std::string preset_directory = "./presets/effects";
    bool auto_save_presets = false;
    uint32_t auto_preset_interval_seconds = 300;

    // Real-time features
    bool enable_real_time_effect_switching = true;
    bool enable_crossfade_switching = true;
    float crossfade_time_ms = 50.0f;
    bool enable_smooth_parameter_changes = true;

    // Monitoring and statistics
    bool enable_performance_monitoring = true;
    uint32_t statistics_update_interval_ms = 1000;
    bool enable_latency_measurement = true;
    bool enable_quality_monitoring = false;

    // Advanced settings
    bool enable_multi_threading = true;
    uint32_t thread_pool_size = 0; // 0 = auto
    bool enable_cache_optimization = true;
    bool enable_sse_optimization = true;
    bool enable_avx_optimization = true;
};

/**
 * Real-time effects chain
 */
class RealtimeEffectsChain {
public:
    RealtimeEffectsChain();
    ~RealtimeEffectsChain();

    // Lifecycle management
    bool initialize(const EffectsChainConfig& config);
    void shutdown();
    bool is_initialized() const { return initialized_; }
    bool is_processing() const { return processing_.load(); }

    // Configuration
    void update_config(const EffectsChainConfig& config);
    const EffectsChainConfig& get_config() const { return config_; }

    // Audio format
    bool set_audio_format(uint32_t sample_rate, uint32_t channels, uint32_t max_frame_size);
    AudioBufferMetadata get_audio_format() const;

    // Chain management
    bool add_effect(std::shared_ptr<AudioEffect> effect, int position = -1);
    bool remove_effect(const std::string& effect_name);
    bool remove_effect_at_position(int position);
    bool move_effect(const std::string& effect_name, int new_position);
    bool swap_effects(int position1, int position2);
    std::vector<std::shared_ptr<AudioEffect>> get_effects() const;
    std::shared_ptr<AudioEffect> get_effect(const std::string& name) const;
    std::shared_ptr<AudioEffect> get_effect_at_position(int position) const;
    size_t get_effect_count() const;

    // Processing control
    bool start_processing();
    void stop_processing();
    bool pause_processing();
    bool resume_processing();

    // Audio processing
    bool process_audio(const float* input, float* output, uint32_t frame_count);
    bool process_audio_interleaved(const float* input, float* output, uint32_t frame_count);
    bool process_audio_multi_channel(const std::vector<const float*>& inputs,
                                    std::vector<float*>& outputs,
                                    uint32_t frame_count);

    // Global controls
    bool set_global_bypass(bool bypass);
    bool is_globally_bypassed() const;
    bool set_dry_wet_mix(float mix); // 0.0 = dry, 1.0 = wet
    float get_dry_wet_mix() const { return dry_wet_mix_; }
    bool set_output_gain_db(float gain_db);
    float get_output_gain_db() const;
    bool set_input_gain_db(float gain_db);
    float get_input_gain_db() const;

    // Effect controls
    bool set_effect_bypass(const std::string& effect_name, bool bypass);
    bool set_effect_mute(const std::string& effect_name, bool mute);
    bool set_effect_dry_wet_mix(const std::string& effect_name, float mix);

    // Parameter automation
    bool automate_parameter(const std::string& effect_name, const std::string& parameter_name,
                          float target_value, float time_ms);
    bool automate_parameter_linear(const std::string& effect_name, const std::string& parameter_name,
                                  float start_value, float end_value, float duration_ms);
    bool stop_parameter_automation(const std::string& effect_name, const std::string& parameter_name);

    // MIDI control
    bool add_midi_mapping(const std::string& effect_name, const MidiMapping& mapping);
    bool remove_midi_mapping(const std::string& effect_name, const std::string& parameter_name);
    bool process_midi_message(uint8_t status, uint8_t data1, uint8_t data2);
    bool enter_midi_learn_mode(const std::string& effect_name, const std::string& parameter_name);
    bool exit_midi_learn_mode();

    // Preset management
    bool save_chain_preset(const std::string& name);
    bool load_chain_preset(const std::string& name);
    bool delete_chain_preset(const std::string& name);
    std::vector<std::string> get_available_presets() const;
    bool save_effect_preset(const std::string& effect_name, const std::string& preset_name);
    bool load_effect_preset(const std::string& effect_name, const std::string& preset_name);

    // Real-time monitoring
    struct ChainStatistics {
        uint64_t total_process_calls = 0;
        uint64_t successful_calls = 0;
        double avg_processing_time_us = 0.0;
        double max_processing_time_us = 0.0;
        double min_processing_time_us = std::numeric_limits<double>::max();
        double current_latency_ms = 0.0;
        double avg_latency_ms = 0.0;
        float cpu_utilization_percent = 0.0f;
        float gpu_utilization_percent = 0.0f;
        uint32_t active_effects = 0;
        uint32_t bypassed_effects = 0;
        std::chrono::steady_clock::time_point last_update;
    };

    ChainStatistics get_chain_statistics() const;
    std::vector<EffectStatistics> get_all_effect_statistics() const;
    EffectStatistics get_effect_statistics(const std::string& effect_name) const;
    void reset_statistics();

    // Performance optimization
    bool set_processing_mode(ProcessingMode mode);
    ProcessingMode get_processing_mode() const;
    bool enable_gpu_acceleration(bool enabled);
    bool is_gpu_acceleration_enabled() const;
    void optimize_for_latency();
    void optimize_for_quality();
    void optimize_for_power();

    // Advanced features
    bool enable_parallel_processing(bool enabled);
    bool enable_crossfade_effect_switching(bool enabled);
    bool set_crossfade_time(float time_ms);
    bool enable_audio_rate_modulation(bool enabled);
    bool set_modulation_rate(uint32_t rate_hz);

    // Hardware monitoring integration
    void set_hardware_monitor(std::shared_ptr<hardware::HardwareMonitor> monitor);
    void set_streaming_interface(std::shared_ptr<network::RealtimeStreamer> streamer);

    // Event callbacks
    using EffectAddedCallback = std::function<void(std::shared_ptr<AudioEffect>)>;
    using EffectRemovedCallback = std::function<void(const std::string&)>;
    using ParameterChangedCallback = std::function<void(const std::string&, const std::string&, float)>;
    using LatencyCallback = std::function<void(double)>;

    void set_effect_added_callback(EffectAddedCallback callback);
    void set_effect_removed_callback(EffectRemovedCallback callback);
    void set_parameter_changed_callback(ParameterChangedCallback callback);
    void set_latency_callback(LatencyCallback callback);

    // Diagnostics
    std::string get_diagnostics_report() const;
    bool validate_chain_setup() const;
    std::vector<std::string> test_chain_performance() const;

private:
    // Configuration and state
    EffectsChainConfig config_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> processing_{false};
    std::atomic<bool> paused_{false};

    // Audio format
    AudioBufferMetadata audio_format_;

    // Effects chain
    std::vector<std::shared_ptr<AudioEffect>> effects_chain_;
    mutable std::mutex effects_mutex_;
    std::unordered_map<std::string, std::shared_ptr<AudioEffect>> effects_map_;
    std::unordered_map<std::string, int> effect_positions_;

    // Global parameters
    std::atomic<bool> global_bypass_{false};
    std::atomic<float> dry_wet_mix_{1.0f}; // 1.0 = fully wet
    std::atomic<float> output_gain_linear_{1.0f};
    std::atomic<float> input_gain_linear_{1.0f};

    // Processing buffers
    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;
    std::vector<float> temp_buffer_;
    std::vector<float> dry_buffer_; // For dry/wet mixing
    mutable std::mutex buffers_mutex_;

    // Parameter interpolation
    struct ParameterAutomation {
        std::string effect_name;
        std::string parameter_name;
        float start_value;
        float target_value;
        float current_value;
        float duration_ms;
        std::chrono::steady_clock::time_point start_time;
        bool is_active = false;
    };

    std::vector<ParameterAutomation> automations_;
    mutable std::mutex automations_mutex_;

    // MIDI control
    std::vector<MidiMapping> midi_mappings_;
    mutable std::mutex midi_mutex_;
    bool midi_learn_mode_ = false;
    std::string midi_learn_effect_;
    std::string midi_learn_parameter_;

    // Performance monitoring
    mutable std::mutex stats_mutex_;
    ChainStatistics chain_stats_;
    std::chrono::steady_clock::time_point last_stats_update_;
    uint64_t process_call_count_ = 0;

    // Threading for parallel processing
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> parallel_processing_enabled_{false};
    std::condition_variable work_cv_;
    std::mutex work_mutex_;

    // Hardware monitoring
    std::shared_ptr<hardware::HardwareMonitor> hardware_monitor_;
    std::shared_ptr<network::RealtimeStreamer> streamer_;

    // Callbacks
    std::mutex callbacks_mutex_;
    EffectAddedCallback effect_added_callback_;
    EffectRemovedCallback effect_removed_callback_;
    ParameterChangedCallback parameter_changed_callback_;
    LatencyCallback latency_callback_;

    // Internal processing methods
    bool process_effects_chain(const float* input, float* output, uint32_t frame_count);
    bool process_effects_serial(const float* input, float* output, uint32_t frame_count);
    bool process_effects_parallel(const float* input, float* output, uint32_t frame_count);
    void update_parameter_automations();
    float interpolate_parameter(float current, float target, float progress, InterpolationType type);
    void process_midi_mapping(uint8_t status, uint8_t data1, uint8_t data2);
    void update_statistics(double processing_time_us, uint32_t frame_count);
    void apply_dry_wet_mix(const float* dry_input, const float* wet_output, float* final_output,
                          uint32_t frame_count);
    void apply_gain(float* audio, uint32_t frame_count, float gain);

    // Utility methods
    float db_to_linear(float db) const;
    float linear_to_db(float linear) const;
    bool validate_audio_format(const AudioBufferMetadata& format) const;
    void optimize_buffer_sizes(uint32_t max_frame_size);
    void notify_effect_added(std::shared_ptr<AudioEffect> effect);
    void notify_effect_removed(const std::string& name);
    void notify_parameter_changed(const std::string& effect_name, const std::string& param_name, float value);
    void notify_latency_changed(double latency_ms);
};

/**
 * Built-in audio effects factory
 */
class AudioEffectsFactory {
public:
    // Equalizers
    static std::shared_ptr<AudioEffect> create_parametric_eq(uint32_t bands = 4);
    static std::shared_ptr<AudioEffect> create_graphic_eq(uint32_t bands = 31);
    static std::shared_ptr<AudioEffect> create_shelving_eq();

    // Dynamics
    static std::shared_ptr<AudioEffect> create_compressor();
    static std::shared_ptr<AudioEffect> create_limiter();
    static std::shared_ptr<AudioEffect> create_gate();
    static std::shared_ptr<AudioEffect> create_expander();
    static std::shared_ptr<AudioEffect> create_multiband_compressor(uint32_t bands = 3);

    // Time-based effects
    static std::shared_ptr<AudioEffect> create_reverb(EffectType type = EffectType::REVERB);
    static std::shared_ptr<AudioEffect> create_delay(bool stereo = true);
    static std::shared_ptr<AudioEffect> create_echo();
    static std::shared_ptr<AudioEffect> create_multi_tap_delay(uint32_t taps = 4);

    // Modulation effects
    static std::shared_ptr<AudioEffect> create_chorus();
    static std::shared_ptr<AudioEffect> create_flanger();
    static std::shared_ptr<AudioEffect> create_phaser(int stages = 8);
    static std::shared_ptr<AudioEffect> create_vibrato();
    static std::shared_ptr<AudioEffect> create_tremolo();
    static std::shared_ptr<AudioEffect> create_ring_modulator();

    // Distortion and saturation
    static std::shared_ptr<AudioEffect> create_distortion();
    static std::shared_ptr<AudioEffect> create_saturation();
    static std::shared_ptr<AudioEffect> create_bitcrusher();
    static std::shared_ptr<AudioEffect> create_tube_saturation();
    static std::shared_ptr<AudioEffect> create_tape_saturation();

    // Filters
    static std::shared_ptr<AudioEffect> create_low_pass_filter();
    static std::shared_ptr<AudioEffect> create_high_pass_filter();
    static std::shared_ptr<AudioEffect> create_band_pass_filter();
    static std::shared_ptr<AudioEffect> create_notch_filter();
    static std::shared_ptr<AudioEffect> create_peak_filter();
    static std::shared_ptr<AudioEffect> create_all_pass_filter();

    // Specialized effects
    static std::shared_ptr<AudioEffect> create_pitch_shifter();
    static std::shared_ptr<AudioEffect> create_time_stretch();
    static std::shared_ptr<AudioEffect> create_stereo_enhancer();
    static std::shared_ptr<AudioEffect> create_loudness_maximizer();
    static std::shared_ptr<AudioEffect> create_deesser();
    static std::shared_ptr<AudioEffect> create_auto_wah();
    static std::shared_ptr<AudioEffect> create_vocoder();

    // Utility effects
    static std::shared_ptr<AudioEffect> create_gain_stage();
    static std::shared_ptr<AudioEffect> create_pan_control();
    static std::shared_ptr<AudioEffect> create_mute_switch();
    static std::shared_ptr<AudioEffect> create_phase_inverter();
};

/**
 * Utility functions for effects processing
 */
namespace effects_utils {
    // Parameter conversion utilities
    float db_to_linear(float db);
    float linear_to_db(float linear);
    float seconds_to_samples(float seconds, uint32_t sample_rate);
    float samples_to_seconds(uint32_t samples, uint32_t sample_rate);
    float frequency_to_midi_note(float frequency_hz);
    float midi_note_to_frequency(uint8_t midi_note);

    // Audio buffer utilities
    void copy_buffer(const float* source, float* dest, uint32_t frame_count);
    void clear_buffer(float* buffer, uint32_t frame_count);
    void apply_gain(float* buffer, uint32_t frame_count, float gain);
    void mix_buffers(const float* buffer1, const float* buffer2, float* output,
                    uint32_t frame_count, float mix_ratio = 0.5f);
    void crossfade_buffers(const float* buffer1, const float* buffer2, float* output,
                          uint32_t frame_count, float crossfade_progress);

    // Interpolation utilities
    float linear_interpolate(float a, float b, float t);
    float exponential_interpolate(float a, float b, float t);
    float logarithmic_interpolate(float a, float b, float t);
    float smoothstep_interpolate(float a, float b, float t);
    float sine_interpolate(float a, float b, float t);

    // Audio analysis utilities
    float calculate_rms_level(const float* buffer, uint32_t frame_count);
    float calculate_peak_level(const float* buffer, uint32_t frame_count);
    float calculate_crest_factor(const float* buffer, uint32_t frame_count);
    float calculate_zero_crossing_rate(const float* buffer, uint32_t frame_count);
    void calculate_frequency_spectrum(const float* buffer, uint32_t frame_count,
                                    float* spectrum_magnitude, float* spectrum_phase,
                                    uint32_t sample_rate);

    // Performance utilities
    bool is_sse_supported();
    bool is_avx_supported();
    bool is_avx2_supported();
    void* aligned_alloc(size_t size, size_t alignment = 32);
    void aligned_free(void* ptr);
}

} // namespace vortex::core::dsp