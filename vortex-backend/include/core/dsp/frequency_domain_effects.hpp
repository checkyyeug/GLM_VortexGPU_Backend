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
#include <complex>
#include <valarray>

#include "system/logger.hpp"
#include "core/dsp/realtime_effects_chain.hpp"

namespace vortex::core::dsp {

/**
 * Frequency-Domain Audio Effects Processor for Vortex GPU Audio Backend
 *
 * This component provides high-quality frequency-domain audio effects with
 * FFT-based processing, suitable for professional audio applications. It supports
 * various frequency-based effects including equalizers, filters, and spectral
 * processing with CPU/GPU acceleration and real-time parameter control.
 *
 * Features:
 * - FFT-based frequency-domain processing with low latency
 * - High-fidelity equalizers and filters
 * - Spectral manipulation and restoration
 * - Multi-band compression and expansion
 * - Dynamic EQ with real-time frequency tracking
 * - Phase correction and linear-phase filtering
 * - Spectral noise reduction and restoration
 * - Real-time parameter interpolation
 * - GPU-accelerated FFT processing
 * - Multi-channel support with phase coherence
 * - Professional audio quality (64-bit floating-point)
 * - Zero-latency monitoring options for certain modes
 */

/**
 * Frequency-domain effect types
 */
enum class FrequencyDomainEffectType {
    EQUALIZER,         // Parametric/graphic equalizer
    FILTER,            // Various filter types
    MULTI_BAND_COMP,   // Multi-band compression
    SPECTRAL_SHAPER,   // Spectral shaping
    NOISE_REDUCTION,   // Spectral noise reduction
    DEESSER,           // Spectral de-essing
    DYNAMIC_EQ,        // Dynamic equalization
    SPECTRAL_GATE,     // Spectral gating
    PHASE_CORRECTOR,   // Phase correction
    LINEAR_PHASE_FILT, // Linear phase filtering
    VOCODER,           // Vocoder/spectral processing
    PITCH_CORRECTOR,   // Spectral pitch correction
    HARMONIC_EXCITER,  // Harmonic excitation
    SPECTRAL_ENHANCER, // Spectral enhancement
    TRANSIENT_SHAPER,  // Transient shaping in frequency domain
    STEREO_IMAGER,     // Stereo imaging
    MID_SIDE_PROCESSOR, // Mid/side processing
    SPECTRAL_COMPRESSOR, // Spectral compression
    PHASE_VOCODER,     // Phase vocoder
    CONVOLUTION,       // Frequency-domain convolution
    CUSTOM             // Custom frequency-domain effect
};

/**
 * FFT window types
 */
enum class FFTWindowType {
    RECTANGULAR,       // Rectangular window
    HANNING,           // Hanning window
    HAMMING,           // Hamming window
    BLACKMAN,          // Blackman window
    BLACKMAN_HARRIS,   // Blackman-Harris window
    KAISER,            // Kaiser window
    FLAT_TOP,          // Flat top window
    DOLPH_CHEBYSHEV,   // Dolph-Chebyshev window
    GAUSSIAN,          // Gaussian window
    TUKEY,             // Tukey window
    CUSTOM             // Custom window
};

/**
 * Filter types for frequency-domain processing
 */
enum class FilterType {
    LOW_PASS,          // Low-pass filter
    HIGH_PASS,         // High-pass filter
    BAND_PASS,         // Band-pass filter
    BAND_STOP,         // Band-stop (notch) filter
    PEAK,              // Peaking filter
    LOW_SHELF,         // Low shelf filter
    HIGH_SHELF,        // High shelf filter
    ALL_PASS,          // All-pass filter
    BELL,              // Bell-shaped filter
    NOTCH,             // Notch filter
    COMB,              // Comb filter
    RESONANT,          // Resonant filter
    LINKWITZ_RILEY,    // Linkwitz-Riley crossover
    BUTTERWORTH,       // Butterworth filter
    CHEBYSHEV,         // Chebyshev filter
    ELLIPTIC,          // Elliptic (Cauer) filter
    BESSEL,            // Bessel filter
    CUSTOM             // Custom filter response
};

/**
 * Equalizer band types
 */
enum class EQBandType {
    PEAK,              // Peak/dip filter
    LOW_SHELF,         // Low shelf filter
    HIGH_SHELF,        // High shelf filter
    LOW_PASS,          // Low-pass filter
    HIGH_PASS,         // High-pass filter
    BAND_PASS,         // Band-pass filter
    NOTCH,             // Notch filter
    ALL_PASS,          // All-pass filter
    TILTING_SHELF,     // Tilting shelf filter
    VARIABLE_Q,        // Variable Q filter
    PASSIVE            // Passive EQ response
};

/**
 * Frequency-domain processing modes
 */
enum class ProcessingMode {
    REAL_TIME,         // Real-time processing with overlap-add
    BLOCK_BASED,       // Block-based processing
    STREAMING,         // Streaming processing
    OFFLINE,           // Offline processing (highest quality)
    INTERACTIVE,       // Interactive processing
    ANALYSIS,          // Analysis-only mode
    SYNTHESIS,         // Synthesis-only mode
    ADAPTIVE           // Adaptive processing based on content
};

/**
 * Frequency-domain effect parameters
 */
struct FrequencyDomainParameters {
    // FFT parameters
    uint32_t fft_size = 4096;               // FFT size (power of 2)
    uint32_t overlap_factor = 4;             // Overlap factor for OLA
    FFTWindowType window_type = FFTWindowType::HANNING;
    uint32_t window_size = 4096;             // Window size
    bool zero_padding = true;                // Zero padding for interpolation
    bool magnitude_smoothing = true;         // Smooth magnitude transitions
    bool phase_smoothing = true;             // Smooth phase transitions

    // Equalizer parameters
    struct EQBand {
        float frequency_hz = 1000.0f;        // Center frequency
        float gain_db = 0.0f;               // Gain in dB
        float q_factor = 1.0f;              // Q factor (bandwidth)
        EQBandType type = EQBandType::PEAK;  // Band type
        bool enabled = true;                 // Band enabled
        float frequency_smooth_ms = 10.0f;   // Frequency smoothing time
        float gain_smooth_ms = 10.0f;        // Gain smoothing time
        float q_smooth_ms = 10.0f;           // Q smoothing time
    };

    std::vector<EQBand> eq_bands = {
        {100.0f, 0.0f, 1.0f, EQBandType::LOW_SHELF},
        {1000.0f, 0.0f, 1.0f, EQBandType::PEAK},
        {10000.0f, 0.0f, 1.0f, EQBandType::HIGH_SHELF}
    };

    // Filter parameters
    FilterType filter_type = FilterType::LOW_PASS;
    float cutoff_frequency_hz = 1000.0f;    // Cutoff frequency
    float resonance = 1.0f;                 // Resonance/Q
    float slope_db_octave = 12.0f;           // Filter slope
    float order = 2.0f;                     // Filter order
    bool zero_phase = false;                 // Zero-phase filtering
    bool linear_phase = false;               // Linear-phase filtering

    // Multi-band compression parameters
    uint32_t num_bands = 3;                 // Number of frequency bands
    std::vector<float> band_frequencies_hz = {250.0f, 1000.0f, 4000.0f, 12000.0f};
    std::vector<float> band_thresholds_db = {-20.0f, -15.0f, -18.0f, -22.0f};
    std::vector<float> band_ratios = {3.0f, 2.5f, 4.0f, 2.0f};
    std::vector<float> band_attack_ms = {5.0f, 3.0f, 4.0f, 2.0f};
    std::vector<float> band_release_ms = {100.0f, 80.0f, 120.0f, 60.0f};
    std::vector<float> band_makeup_gain_db = {0.0f, 2.0f, 1.0f, 3.0f};
    bool crossover_linear_phase = true;      // Linear-phase crossovers

    // Spectral processing parameters
    float spectral_threshold_db = -60.0f;   // Noise threshold
    float spectral_reduction_db = 12.0f;    // Noise reduction amount
    float spectral_smoothing = 0.8f;        // Spectral smoothing factor
    bool transient_preservation = true;     // Preserve transients
    bool harmonic_preservation = true;      // Preserve harmonics
    float noise_floor_db = -120.0f;         // Estimated noise floor
    bool adaptive_threshold = true;         // Adaptive threshold

    // Dynamic EQ parameters
    float dynamic_threshold_db = -20.0f;    // Dynamic processing threshold
    float dynamic_range_db = 20.0f;         // Dynamic range
    float dynamic_ratio = 2.0f;             // Dynamic ratio
    float dynamic_attack_ms = 10.0f;        // Attack time
    float dynamic_release_ms = 100.0f;      // Release time
    bool frequency_tracking = true;         // Track frequency content
    float tracking_sensitivity = 0.5f;      // Tracking sensitivity

    // Stereo imaging parameters
    float stereo_width = 1.0f;              // Stereo width adjustment
    float mid_gain_db = 0.0f;               // Mid channel gain
    float side_gain_db = 0.0f;              // Side channel gain
    float stereo_rotation = 0.0f;           // Stereo rotation angle
    bool mid_side_mode = false;             // Mid/side processing mode
    float bass_mono_freq = 150.0f;          // Bass mono frequency
    float treble_mono_freq = 8000.0f;       // Treble mono frequency

    // Phase correction parameters
    float phase_correction_strength = 1.0f;  // Correction strength
    float phase_alignment = 0.0f;            // Phase alignment
    bool phase_smoothing_enabled = true;    // Phase smoothing
    float phase_threshold_rad = 0.1f;        // Phase correction threshold

    // Quality and performance
    bool high_precision_mode = true;         // 64-bit processing
    bool gpu_acceleration_enabled = true;    // GPU FFT acceleration
    bool multithreading_enabled = true;      // Multi-threading
    uint32_t num_threads = 0;                // Thread count (0 = auto)
    bool memory_optimization = true;         // Memory usage optimization
    bool cache_optimization = true;          // Cache-friendly processing

    // Advanced parameters
    bool magnitude_interpolation = true;     // Magnitude interpolation
    bool phase_interpolation = true;         // Phase interpolation
    float interpolation_quality = 0.8f;      // Interpolation quality (0-1)
    bool spectral_enhancement = false;      // Spectral enhancement
    float harmonic_generation = 0.0f;        // Harmonic generation amount
    bool transient_emphasis = false;         // Emphasize transients

    // Real-time parameters
    float parameter_smooth_time_ms = 20.0f;  // Parameter smoothing time
    bool zero_latency_mode = false;          // Zero latency mode (compromised quality)
    float lookahead_time_ms = 5.0f;           // Lookahead time
    bool auto_gain_correction = true;        // Automatic gain correction
    float target_level_dbfs = -1.0f;         // Target output level
};

/**
 * Frequency-domain effect processing statistics
 */
struct FrequencyDomainStatistics {
    uint64_t total_process_calls = 0;
    uint64_t successful_calls = 0;
    double avg_processing_time_us = 0.0;
    double max_processing_time_us = 0.0;
    double min_processing_time_us = std::numeric_limits<double>::max();
    uint64_t fft_operations = 0;
    uint64_t ifft_operations = 0;
    uint64_t spectral_modifications = 0;
    double cpu_utilization_percent = 0.0;
    double gpu_utilization_percent = 0.0;
    float memory_usage_mb = 0.0f;
    uint32_t fft_size = 0;
    uint32_t overlap_factor = 0;
    double avg_latency_ms = 0.0;
    uint64_t buffer_underruns = 0;
    uint64_t buffer_overruns = 0;
    std::chrono::steady_clock::time_point last_reset_time;
    bool is_active = false;
};

/**
 * Abstract base class for frequency-domain effects
 */
class FrequencyDomainEffect {
public:
    virtual ~FrequencyDomainEffect() = default;

    // Basic lifecycle
    virtual bool initialize(uint32_t sample_rate, uint32_t channels, uint32_t max_frame_size) = 0;
    virtual void shutdown() = 0;
    virtual bool reset() = 0;

    // Processing
    virtual bool process(const float* input, float* output, uint32_t frame_count) = 0;
    virtual bool process_interleaved(const float* input, float* output, uint32_t frame_count) = 0;

    // Parameters
    virtual bool set_parameters(const FrequencyDomainParameters& params) = 0;
    virtual FrequencyDomainParameters get_parameters() const = 0;
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
    virtual FrequencyDomainEffectType get_type() const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_version() const = 0;
    virtual std::string get_description() const = 0;
    virtual FrequencyDomainStatistics get_statistics() const = 0;
    virtual void reset_statistics() = 0;

    // Advanced features
    virtual bool supports_real_time_parameter_changes() const = 0;
    virtual bool supports_gpu_acceleration() const = 0;
    virtual bool prefers_gpu_processing() const { return false; }
    virtual uint32_t get_fft_size() const = 0;
    virtual double get_expected_latency_ms() const = 0;
    virtual bool is_linear_phase() const = 0;
};

/**
 * FFT processor class for frequency-domain operations
 */
class FFTProcessor {
public:
    FFTProcessor();
    ~FFTProcessor();

    bool initialize(uint32_t fft_size, uint32_t channels, bool gpu_acceleration = false);
    void shutdown();
    void clear();

    // Core FFT operations
    bool forward_fft(const float* input, std::complex<float>* output, uint32_t frame_size);
    bool inverse_fft(const std::complex<float>* input, float* output);
    bool process_overlap_add(const float* input, float* output, uint32_t frame_size);

    // Window functions
    bool apply_window(float* buffer, uint32_t size, FFTWindowType window_type);
    void generate_window(float* window, uint32_t size, FFTWindowType window_type);

    // FFT parameters
    void set_fft_size(uint32_t fft_size);
    void set_window_type(FFTWindowType window_type);
    void set_overlap_factor(uint32_t overlap_factor);
    uint32_t get_fft_size() const { return fft_size_; }
    uint32_t get_num_bins() const { return num_bins_; }
    uint32_t get_channels() const { return channels_; }

    // Performance
    bool set_gpu_acceleration(bool enabled);
    bool is_gpu_accelerated() const { return gpu_accelerated_; }

private:
    uint32_t fft_size_;
    uint32_t num_bins_;
    uint32_t channels_;
    uint32_t overlap_factor_;
    FFTWindowType window_type_;
    bool gpu_accelerated_;

    // Processing buffers
    std::vector<float> window_buffer_;
    std::vector<std::complex<float>> fft_buffer_;
    std::vector<std::complex<float>> ifft_buffer_;
    std::vector<float> overlap_buffer_;
    std::vector<float> input_history_;
    std::vector<float> output_history_;

    // FFT plans (would use FFTW, Intel MKL, or similar)
    void* fft_plan_;
    void* ifft_plan_;

    // GPU resources (if enabled)
    void* gpu_fft_context_;

    // Internal methods
    bool create_fft_plans();
    void destroy_fft_plans();
    bool initialize_gpu_fft();
    void shutdown_gpu_fft();
    void perform_cpu_fft(const float* input, std::complex<float>* output);
    void perform_cpu_ifft(const std::complex<float>* input, float* output);
    bool perform_gpu_fft(const float* input, std::complex<float>* output);
    bool perform_gpu_ifft(const std::complex<float>* input, float* output);
};

/**
 * Parametric equalizer processor
 */
class ParametricEqualizer : public FrequencyDomainEffect {
public:
    ParametricEqualizer();
    ~ParametricEqualizer() override;

    // FrequencyDomainEffect interface
    bool initialize(uint32_t sample_rate, uint32_t channels, uint32_t max_frame_size) override;
    void shutdown() override;
    bool reset() override;

    bool process(const float* input, float* output, uint32_t frame_count) override;
    bool process_interleaved(const float* input, float* output, uint32_t frame_count) override;

    bool set_parameters(const FrequencyDomainParameters& params) override;
    FrequencyDomainParameters get_parameters() const override;
    bool set_parameter(const std::string& name, float value) override;
    float get_parameter(const std::string& name) const override;

    bool set_bypass(bool bypass) override;
    bool is_bypassed() const override;
    bool set_dry_wet_mix(float mix) override;

    bool save_preset(const std::string& name) override;
    bool load_preset(const std::string& name) override;
    std::vector<std::string> get_available_presets() const override;

    FrequencyDomainEffectType get_type() const override;
    std::string get_name() const override;
    std::string get_version() const override;
    std::string get_description() const override;
    FrequencyDomainStatistics get_statistics() const override;
    void reset_statistics() override;

    bool supports_real_time_parameter_changes() const override;
    bool supports_gpu_acceleration() const override;
    uint32_t get_fft_size() const override;
    double get_expected_latency_ms() const override;
    bool is_linear_phase() const override;

    // Equalizer specific
    void add_band(float frequency_hz, float gain_db, float q_factor, EQBandType type);
    void remove_band(uint32_t band_index);
    void set_band_gain(uint32_t band_index, float gain_db);
    void set_band_frequency(uint32_t band_index, float frequency_hz);
    void set_band_q(uint32_t band_index, float q_factor);
    void set_band_type(uint32_t band_index, EQBandType type);
    void set_band_enabled(uint32_t band_index, bool enabled);
    uint32_t get_band_count() const;
    const FrequencyDomainParameters::EQBand& get_band(uint32_t band_index) const;

private:
    FrequencyDomainParameters parameters_;
    FrequencyDomainEffectType effect_type_;
    bool bypassed_;
    float dry_wet_mix_;

    uint32_t sample_rate_;
    uint32_t channels_;
    uint32_t max_frame_size_;

    // FFT processing
    std::unique_ptr<FFTProcessor> fft_processor_;

    // Frequency response
    std::vector<std::vector<float>> frequency_response_;
    std::vector<std::vector<std::complex<float>>> complex_response_;
    std::vector<std::vector<float>> smoothed_response_;
    std::vector<bool> response_dirty_;

    // Statistics
    mutable std::mutex stats_mutex_;
    FrequencyDomainStatistics statistics_;

    // Processing buffers
    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;
    std::vector<float> wet_buffer_;
    std::vector<float> dry_buffer_;
    std::vector<std::complex<float>> spectrum_;

    // Preset management
    std::unordered_map<std::string, FrequencyDomainParameters> presets_;

    // Internal methods
    void calculate_frequency_response();
    void calculate_band_response(FrequencyDomainParameters::EQBand& band,
                                std::vector<std::complex<float>>& response);
    void smooth_frequency_response(uint32_t channel);
    void apply_frequency_response(const std::vector<std::complex<float>>& input_spectrum,
                                 std::vector<std::complex<float>>& output_spectrum,
                                 uint32_t channel);
    void update_eq_band(uint32_t band_index);
    float interpolate_parameter(float current, float target, float progress);
};

/**
 * Multi-band compressor processor
 */
class MultiBandCompressor : public FrequencyDomainEffect {
public:
    MultiBandCompressor();
    ~MultiBandCompressor() override;

    // FrequencyDomainEffect interface
    bool initialize(uint32_t sample_rate, uint32_t channels, uint32_t max_frame_size) override;
    void shutdown() override;
    bool reset() override;

    bool process(const float* input, float* output, uint32_t frame_count) override;
    bool process_interleaved(const float* input, float* output, uint32_t frame_count) override;

    bool set_parameters(const FrequencyDomainParameters& params) override;
    FrequencyDomainParameters get_parameters() const override;
    bool set_parameter(const std::string& name, float value) override;
    float get_parameter(const std::string& name) const override;

    bool set_bypass(bool bypass) override;
    bool is_bypassed() const override;
    bool set_dry_wet_mix(float mix) override;

    bool save_preset(const std::string& name) override;
    bool load_preset(const std::string& name) override;
    std::vector<std::string> get_available_presets() const override;

    FrequencyDomainEffectType get_type() const override;
    std::string get_name() const override;
    std::string get_version() const override;
    std::string get_description() const override;
    FrequencyDomainStatistics get_statistics() const override;
    void reset_statistics() override;

    bool supports_real_time_parameter_changes() const override;
    bool supports_gpu_acceleration() const override;
    uint32_t get_fft_size() const override;
    double get_expected_latency_ms() const override;
    bool is_linear_phase() const override;

    // Multi-band specific
    void set_num_bands(uint32_t num_bands);
    void set_band_threshold(uint32_t band_index, float threshold_db);
    void set_band_ratio(uint32_t band_index, float ratio);
    void set_band_attack(uint32_t band_index, float attack_ms);
    void set_band_release(uint32_t band_index, float release_ms);
    void set_band_makeup_gain(uint32_t band_index, float makeup_gain_db);
    void set_band_enabled(uint32_t band_index, bool enabled);

private:
    struct CompressorBand {
        float threshold_db;
        float ratio;
        float attack_coeff;
        float release_coeff;
        float envelope;
        float gain_reduction;
        float makeup_gain_db;
        float makeup_gain_linear;
        bool enabled;
        uint32_t start_bin;
        uint32_t end_bin;
        std::vector<float> band_spectrum;
    };

    FrequencyDomainParameters parameters_;
    FrequencyDomainEffectType effect_type_;
    bool bypassed_;
    float dry_wet_mix_;

    uint32_t sample_rate_;
    uint32_t channels_;
    uint32_t max_frame_size_;

    // FFT processing
    std::unique_ptr<FFTProcessor> fft_processor_;

    // Compressor bands
    std::vector<CompressorBand> bands_;

    // Statistics
    mutable std::mutex stats_mutex_;
    FrequencyDomainStatistics statistics_;

    // Processing buffers
    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;
    std::vector<float> wet_buffer_;
    std::vector<std::complex<float>> spectrum_;

    // Preset management
    std::unordered_map<std::string, FrequencyDomainParameters> presets_;

    // Internal methods
    void initialize_bands();
    void calculate_band_ranges();
    void apply_compression(std::vector<std::complex<float>>& spectrum, uint32_t channel);
    float calculate_compression_gain(float level_db, const CompressorBand& band);
    void update_envelope_detector(CompressorBand& band, float input_level);
};

/**
 * Spectral noise reduction processor
 */
class SpectralNoiseReduction : public FrequencyDomainEffect {
public:
    SpectralNoiseReduction();
    ~SpectralNoiseReduction() override;

    // FrequencyDomainEffect interface
    bool initialize(uint32_t sample_rate, uint32_t channels, uint32_t max_frame_size) override;
    void shutdown() override;
    bool reset() override;

    bool process(const float* input, float* output, uint32_t frame_count) override;
    bool process_interleaved(const float* input, float* output, uint32_t frame_count) override;

    bool set_parameters(const FrequencyDomainParameters& params) override;
    FrequencyDomainParameters get_parameters() const override;
    bool set_parameter(const std::string& name, float value) override;
    float get_parameter(const std::string& name) const override;

    bool set_bypass(bool bypass) override;
    bool is_bypassed() const override;
    bool set_dry_wet_mix(float mix) override;

    FrequencyDomainEffectType get_type() const override;
    std::string get_name() const override;
    std::string get_version() const override;
    std::string get_description() const override;
    FrequencyDomainStatistics get_statistics() const override;
    void reset_statistics() override;

    bool supports_real_time_parameter_changes() const override;
    bool supports_gpu_acceleration() const override;
    uint32_t get_fft_size() const override;
    double get_expected_latency_ms() const override;
    bool is_linear_phase() const override;

    // Noise reduction specific
    bool capture_noise_profile(const float* noise_audio, uint32_t frame_count);
    bool load_noise_profile(const std::string& profile_path);
    bool save_noise_profile(const std::string& profile_path);
    void set_noise_threshold(float threshold_db);
    void set_reduction_amount(float reduction_db);

private:
    struct NoiseProfile {
        std::vector<float> magnitude_spectrum;
        std::vector<float> phase_spectrum;
        std::vector<float> noise_floor;
        uint32_t num_bins;
        uint32_t capture_frames;
        bool is_valid;
    };

    FrequencyDomainParameters parameters_;
    FrequencyDomainEffectType effect_type_;
    bool bypassed_;
    float dry_wet_mix_;

    uint32_t sample_rate_;
    uint32_t channels_;
    uint32_t max_frame_size_;

    // FFT processing
    std::unique_ptr<FFTProcessor> fft_processor_;

    // Noise profile
    NoiseProfile noise_profile_;
    bool noise_profile_captured_;

    // Statistics
    mutable std::mutex stats_mutex_;
    FrequencyDomainStatistics statistics_;

    // Processing buffers
    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;
    std::vector<float> wet_buffer_;
    std::vector<std::complex<float>> current_spectrum_;
    std::vector<float> magnitude_spectrum_;
    std::vector<float> phase_spectrum_;
    std::vector<float> gain_spectrum_;

    // Internal methods
    void capture_noise_sample(const std::vector<std::complex<float>>& spectrum);
    void apply_noise_reduction(std::vector<std::complex<float>>& spectrum);
    float calculate_spectral_gain(float noise_level, float signal_level);
    void update_noise_floor(const std::vector<float>& current_magnitude);
    bool is_noise_frequency(const std::vector<std::complex<float>>& spectrum, uint32_t bin);
};

/**
 * Frequency-domain effects factory
 */
class FrequencyDomainEffectsFactory {
public:
    // Equalizers
    static std::unique_ptr<FrequencyDomainEffect> create_parametric_eq(uint32_t bands = 3);
    static std::unique_ptr<FrequencyDomainEffect> create_graphic_eq(uint32_t bands = 31);
    static std::unique_ptr<FrequencyDomainEffect> create_linear_phase_eq();

    // Multi-band processors
    static std::unique_ptr<FrequencyDomainEffect> create_multi_band_compressor(uint32_t bands = 3);
    static std::unique_ptr<FrequencyDomainEffect> create_multi_band_expander(uint32_t bands = 3);
    static std::unique_ptr<FrequencyDomainEffect> create_crossover(uint32_t bands = 2);

    // Filters
    static std::unique_ptr<FrequencyDomainEffect> create_linear_phase_filter(FilterType type);
    static std::unique_ptr<FrequencyDomainEffect> create_adaptive_filter();
    static std::unique_ptr<FrequencyDomainEffect> create_phase_vocoder();

    // Spectral processors
    static std::unique_ptr<FrequencyDomainEffect> create_spectral_gate();
    static std::unique_ptr<FrequencyDomainEffect> create_spectral_enhancer();
    static std::unique_ptr<FrequencyDomainEffect> create_spectral_shaper();

    // Noise reduction
    static std::unique_ptr<FrequencyDomainEffect> create_spectral_noise_reduction();
    static std::unique_ptr<FrequencyDomainEffect> create_spectral_deesser();
    static std::unique_ptr<FrequencyDomainEffect> create_adaptive_noise_reduction();

    // Dynamics
    static std::unique_ptr<FrequencyDomainEffect> create_dynamic_eq();
    static std::unique_ptr<FrequencyDomainEffect> create_spectral_compressor();
    static std::unique_ptr<FrequencyDomainEffect> create_transient_shaper();

    // Stereo processing
    static std::unique_ptr<FrequencyDomainEffect> create_stereo_imager();
    static std::unique_ptr<FrequencyDomainEffect> create_mid_side_processor();

    // Specialized effects
    static std::unique_ptr<FrequencyDomainEffect> create_vocoder();
    static std::unique_ptr<FrequencyDomainEffect> create_pitch_corrector();
    static std::unique_ptr<FrequencyDomainEffect> create_harmonic_exciter();

    // Utility methods
    static std::vector<std::string> get_available_effect_types();
    static std::vector<std::string> get_available_window_types();
    static std::vector<std::string> get_available_filter_types();
    static std::vector<std::string> get_available_eq_band_types();
};

/**
 * Utility functions for frequency-domain processing
 */
namespace frequency_domain_utils {
    // Frequency conversion utilities
    float hz_to_bin(float frequency_hz, uint32_t fft_size, uint32_t sample_rate);
    float bin_to_hz(uint32_t bin, uint32_t fft_size, uint32_t sample_rate);
    float mel_to_hz(float mel);
    float hz_to_mel(float hz);
    float bark_to_hz(float bark);
    float hz_to_bark(float hz);
    float erb_to_hz(float erb);
    float hz_to_erb(float hz);

    // FFT utilities
    bool is_power_of_two(uint32_t value);
    uint32_t next_power_of_two(uint32_t value);
    uint32_t calculate_optimal_fft_size(uint32_t frame_size, uint32_t max_fft_size);
    std::vector<float> generate_window(uint32_t size, FFTWindowType type, float parameter = 0.0f);
    float calculate_window_correction_gain(uint32_t size, FFTWindowType type, uint32_t overlap_factor);

    // Complex number utilities
    float magnitude_to_db(float magnitude);
    float db_to_magnitude(float db);
    float phase_to_radians(float phase_degrees);
    float radians_to_phase(float radians);
    std::complex<float> polar_to_rectangular(float magnitude, float phase);
    void rectangular_to_polar(float real, float imag, float& magnitude, float& phase);

    // Filter design utilities
    std::vector<std::complex<float>> design_fir_filter(FilterType type, float cutoff_hz,
                                                     float sample_rate, uint32_t length,
                                                     float q_factor = 1.0f);
    std::vector<std::complex<float>> design_iir_filter(FilterType type, float cutoff_hz,
                                                     float sample_rate, float q_factor = 1.0f);
    std::vector<std::complex<float>> design_linear_phase_filter(FilterType type, float cutoff_hz,
                                                              float sample_rate, uint32_t length);

    // Spectral analysis utilities
    void calculate_magnitude_spectrum(const std::vector<std::complex<float>>& spectrum,
                                     std::vector<float>& magnitude);
    void calculate_phase_spectrum(const std::vector<std::complex<float>>& spectrum,
                                 std::vector<float>& phase);
    void calculate_power_spectrum(const std::vector<std::complex<float>>& spectrum,
                                 std::vector<float>& power);
    void unwrap_phase(std::vector<float>& phase);
    void smooth_spectrum(std::vector<float>& spectrum, uint32_t window_size);

    // Window function utilities
    void apply_window(float* buffer, uint32_t size, FFTWindowType type);
    void generate_hanning_window(float* window, uint32_t size);
    void generate_hamming_window(float* window, uint32_t size);
    void generate_blackman_window(float* window, uint32_t size);
    void generate_kaiser_window(float* window, uint32_t size, float beta);
    void generate_dolph_chebyshev_window(float* window, uint32_t size, float attenuation);

    // Overlap-add utilities
    bool overlap_add(const float* input, float* output, uint32_t frame_size,
                    const float* window, uint32_t window_size, uint32_t overlap_factor);
    void process_overlap_save(const float* input, float* output, uint32_t frame_size,
                             uint32_t window_size, uint32_t overlap_factor);

    // Audio analysis utilities
    float calculate_spectral_centroid(const std::vector<float>& magnitude_spectrum,
                                      uint32_t sample_rate);
    float calculate_spectral_bandwidth(const std::vector<float>& magnitude_spectrum,
                                       uint32_t sample_rate);
    float calculate_spectral_rolloff(const std::vector<float>& magnitude_spectrum,
                                     uint32_t sample_rate, float threshold = 0.85f);
    float calculate_spectral_flux(const std::vector<float>& current_magnitude,
                                  const std::vector<float>& previous_magnitude);
    float calculate_zero_crossing_rate(const float* audio, uint32_t frame_count);

    // Performance optimization utilities
    bool is_sse_supported();
    bool is_avx_supported();
    bool is_fma_supported();
    void* aligned_malloc(size_t size, size_t alignment = 32);
    void aligned_free(void* ptr);
    void apply_vectorized_gain(std::complex<float>* spectrum, uint32_t size, float gain);
    void apply_vectorized_multiply(std::complex<float>* dest, const std::complex<float>* src,
                                   uint32_t size);

    // Validation utilities
    bool validate_fft_parameters(uint32_t fft_size, uint32_t overlap_factor);
    bool validate_filter_parameters(FilterType type, float cutoff_hz, float sample_rate);
    bool validate_equalizer_parameters(const std::vector<FrequencyDomainParameters::EQBand>& bands,
                                      uint32_t sample_rate);
    bool is_frequency_response_stable(const std::vector<float>& response);
}

} // namespace vortex::core::dsp