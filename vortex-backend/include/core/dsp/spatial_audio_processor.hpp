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
#include <cmath>

#include "system/logger.hpp"
#include "core/dsp/realtime_effects_chain.hpp"

namespace vortex::core::dsp {

/**
 * Spatial Audio Processor for Vortex GPU Audio Backend
 *
 * This component provides advanced spatial audio processing including 3D positioning,
 * binaural rendering, ambisonics decoding, and spatial effects for immersive audio
 * experiences. It supports multiple speaker configurations, VR/AR applications,
 * and professional spatial audio workflows.
 *
 * Features:
 * - 3D audio positioning with real-time source tracking
 * - Binaural rendering with individualized HRTF
 * - Ambisonics encoding/decoding (First, Second, Third Order)
 * - Multiple speaker array configurations
 * - VR/AR spatial audio support
 * - Head-related transfer function (HRTF) processing
 * - Doppler effect and distance attenuation
 * - Room acoustics simulation
 * - Multi-channel audio routing
 * - Real-time source management
 * - GPU-accelerated spatial processing
 * - Professional audio quality (64-bit processing)
 * - Low-latency real-time spatial audio
 */

/**
 * Spatial audio processing types
 */
enum class SpatialAudioType {
    STEREO_PANNING,       // Stereo panning
    MULTI_CHANNEL_PANNING, // Multi-channel panning
    BINAURAL,             // Binaural rendering
    VBAP,                 // Vector Base Amplitude Panning
    AMBISONIC_DECODING,    // Ambisonics decoding
    OBJECT_BASED_AUDIO,     // Object-based audio
    WAVE_FIELD_SYNTHESIS,  // Wave field synthesis
    BINAURAL_ROOM_SIM,     // Binaural room simulation
    HRTF_CONVOLUTION,      // HRTF convolution
    VR_SPATIAL_AUDIO,      // VR/AR spatial audio
    GAME_AUDIO,            // Game audio spatialization
    CINEMA_AUDIO,          // Cinema audio processing
    LIVE_SOUND,            // Live sound reinforcement
    AUTOMOTIVE_AUDIO,      // Automotive spatial audio
    AUGMENTED_REALITY,     // AR spatial audio
    CUSTOM                 // Custom spatial processing
};

/**
 * Speaker configuration types
 */
enum class SpeakerConfiguration {
    MONO,                  // Mono (1.0)
    STEREO,                // Stereo (2.0)
    LCR,                   // Left-Center-Right (3.0)
    QUAD,                  // Quadraphonic (4.0)
    SURROUND_5_1,          // 5.1 surround
    SURROUND_7_1,          // 7.1 surround
    SURROUND_7_1_4,         // 7.1.4 Dolby Atmosmos
    DOLBY_ATMOS,           // Dolby Atmos (7.1.4 or more)
    DTS_X,                 // DTS:X (object-based)
    Auro_3D,               // Auro-3D (11.1)
    IMAX,                  // IMAX (multi-channel)
    BINAURAL,              // Binaural (2-channel headphones)
    AMBISONIC,             // Ambisonics (spherical harmonics)
    CUSTOM_ARRAY,          // Custom speaker array
    VR_AUDIO,              // VR audio setup
    GAME_CONSOLE,          // Game console setup
    AUTOMOTIVE,            // Automotive system
    LIVE_SOUND              // Live sound reinforcement
};

/**
 * Spatial audio source types
 */
enum class SourceType {
    POINT_SOURCE,           // Point source
    LINE_SOURCE,            // Line source
    PLANE_SOURCE,           // Plane source
    AREA_SOURCE,            // Area source
    OMNIDIRECTIONAL,        // Omnidirectional source
    DIRECTIONAL,            // Directional source
    ENVIRONMENT,            // Environmental source
    IMPULSE,                // Impulse source
    CONTINUOUS,             // Continuous source
    INTERACTIVE,            // Interactive source
    MOVING,                 // Moving source
    STATIC                  // Static source
};

/**
 * Room acoustics models
 */
enum class RoomAcousticsModel {
    FREE_FIELD,              // Free field (no reflections)
    SMALL_ROOM,             // Small room (reverb time < 0.5s)
    MEDIUM_ROOM,            // Medium room (0.5-1.5s)
    LARGE_ROOM,             // Large room (>1.5s)
    CONCERT_HALL,           // Concert hall
    CHURCH,                 // Church/acoustic space
    THEATER,                // Theater
    STADIUM,                // Stadium/arena
    OUTDOOR,                // Outdoor space
    CUSTOM_EARLY,           // Custom early reflections
    CUSTOM_LATE,            // Custom late reverb
    IMPULSE_RESPONSE,       // Impulse response based
    STATISTICAL,            // Statistical model
    GEOMETRIC_ACOUSTICS,    // Geometric acoustic modeling
    IMAGE_SOURCE,           // Image source method
    WAVEGUIDE,              // Waveguide synthesis
    HYBRID                  // Hybrid model
};

/**
 * HRTF interpolation types
 */
enum class HRTFInterpolation {
    NEAREST_NEIGHBOR,       // Nearest neighbor
    LINEAR,                 // Linear interpolation
    CUBIC,                  // Cubic interpolation
    SPLINE,                 // Spline interpolation
    WEIGHTED,               // Weighted interpolation
    FREQUENCY_DEPENDENT,     // Frequency-dependent
    ADAPTIVE,               // Adaptive interpolation
    MACHINE_LEARNING,       // Machine learning based
    CUSTOM                  // Custom interpolation
};

/**
 * 3D audio source definition
 */
struct AudioSource {
    uint32_t id = 0;                        // Unique source ID
    std::string name = "";                   // Source name
    SourceType type = SourceType::POINT_SOURCE;

    // Position (3D coordinates)
    float position_x = 0.0f;                  // X position (meters)
    float position_y = 0.0f;                  // Y position (meters)
    float position_z = 0.0f;                  // Z position (meters)

    // Velocity (for moving sources)
    float velocity_x = 0.0f;                  // X velocity (m/s)
    float velocity_y = 0.0f;                  // Y velocity (m/s)
    float velocity_z = 0.0f;                  // Z velocity (m/s)

    // Audio properties
    float gain = 1.0f;                        // Linear gain
    float gain_dbfs = 0.0f;                  // Gain in dBFS
    bool mute = false;                         // Source mute state
    bool solo = false;                         // Source solo state

    // Directivity
    float directivity_factor = 1.0f;          // Directivity factor (0-1)
    float directivity_pattern = 0.0f;          // Directivity pattern angle
    float azimuth_angle = 0.0f;                // Azimuth angle (radians)
    float elevation_angle = 0.0f;              // Elevation angle (radians)

    // Distance attenuation
    float reference_distance = 1.0f;           // Reference distance (meters)
    float rolloff_factor = 1.0f;               // Distance rolloff factor
    float max_distance = 100.0f;              // Maximum distance (meters)
    bool enable_distance_attenuation = true;  // Enable distance attenuation
    bool enable_air_absorption = false;       // Enable air absorption

    // Doppler effect
    bool enable_doppler = false;              // Enable Doppler effect
    float doppler_factor = 1.0f;               // Doppler scaling factor

    // Occlusion and obstruction
    bool enable_occlusion = false;             // Enable occlusion modeling
    float occlusion_factor = 1.0f;             // Occlusion factor (0-1)
    bool enable_obstruction = false;           // Enable obstruction modeling

    // Room acoustics
    bool enable_room_acoustics = false;        // Enable room acoustics
    float early_reflection_gain = 0.5f;       // Early reflections gain
    float late_reverb_gain = 0.3f;           // Late reverb gain
    float reverberation_time = 1.0f;          // Reverb time (seconds)

    // HRTF settings
    bool enable_hrtf = true;                  // Enable HRTF processing
    uint32_t hrtf_index = 0;                 // HRTF index
    HRTFInterpolation hrtf_interpolation = HRTFInterpolation::LINEAR;

    // Audio channels
    uint32_t num_channels = 1;                // Number of audio channels
    std::vector<float> channel_gains;          // Channel gains
    std::vector<bool> channel_mutes;           // Channel mute states

    // Timing
    std::chrono::steady_clock::time_point creation_time;
    float delay_ms = 0.0f;                     // Additional delay (ms)
    bool auto_update = true;                  // Auto-update position

    // State
    bool is_active = false;                   // Source is active
    bool is_moving = false;                   // Source is moving
    bool is_visible = true;                   // Source is visible
    float priority = 1.0f;                    // Source priority

    // Audio data
    std::vector<float> audio_buffer;          // Audio buffer
    uint32_t buffer_size = 0;                 // Buffer size
    uint32_t write_position = 0;              // Write position
    uint32_t read_position = 0;               // Read position
    bool is_buffer_full = false;              // Buffer full flag
};

/**
 * Spatial audio processing statistics
 */
struct SpatialAudioStatistics {
    uint64_t total_process_calls = 0;
    uint64_t successful_calls = 0;
    double avg_processing_time_us = 0.0;
    double max_processing_time_us = 0.0;
    double min_processing_time_us = std::numeric_limits<double>::max();

    // Source statistics
    uint32_t active_sources = 0;
    uint32_t moving_sources = 0;
    uint32_t static_sources = 0;
    uint32_t occluded_sources = 0;
    float avg_distance_meters = 0.0f;
    float max_distance_meters = 0.0f;

    // Performance metrics
    double cpu_utilization_percent = 0.0;
    double gpu_utilization_percent = 0.0;
    float memory_usage_mb = 0.0f;
    uint64_t buffer_underruns = 0;
    uint64_t buffer_overruns = 0;

    // HRTF statistics
    uint32_t hrtf_interpolations = 0;
    float avg_hrtf_latency_ms = 0.0f;
    uint32_t hrtf_cache_hits = 0;
    uint32_t hrtf_cache_misses = 0;

    // Room acoustics statistics
    uint64_t early_reflections_calculated = 0;
    uint64_t late_reverb_calculated = 0;
    float avg_reverb_time_seconds = 0.0f;
    float avg_early_reflection_gain = 0.0f;

    // Spatial processing statistics
    uint64_t distance_attenuations = 0;
    uint64_t doppler_calculations = 0;
    uint64_t occlusion_calculations = 0;
    float avg_spatial_gain = 0.0f;
    float avg_doppler_shift = 0.0f;

    // Real-time metrics
    double avg_latency_ms = 0.0;
    double max_latency_ms = 0.0;
    uint32_t frame_drop_count = 0;
    float audio_level_dbfs = 0.0f;
    float spatial_image_score = 0.0f;

    // Quality metrics
    float distortion_thd_percent = 0.0f;
    float spatial_accuracy = 0.0f;
    float localization_error_degrees = 0.0f;
    float channel_balance_db = 0.0f;

    // State
    bool is_active = false;
    bool is_processing = false;
    bool is_3d_enabled = false;

    std::chrono::steady_clock::time_point last_reset_time;
};

/**
 * Spatial audio processor configuration
 */
struct SpatialAudioConfig {
    // Basic configuration
    SpatialAudioType type = SpatialAudioType::BINAURAL;
    SpeakerConfiguration speaker_config = SpeakerConfiguration::STEREO;
    uint32_t sample_rate = 44100;
    uint32_t max_frame_size = 4096;
    uint32_t output_channels = 2;

    // HRTF configuration
    bool enable_hrtf = true;
    std::string hrtf_dataset_path = "./hrtf/";
    uint32_t hrtf_resolution = 512;
    HRTFInterpolation hrtf_interpolation = HRTFInterpolation::LINEAR;
    bool enable_individualized_hrtf = false;
    bool enable_hrtf_caching = true;
    uint32_t hrtf_cache_size = 1024;

    // Ambisonics configuration
    uint32_t ambisonic_order = 1;
    bool enable_ambisonics_encoding = false;
    bool enable_ambisonics_decoding = false;
    float ambisonic_gain_db = 0.0f;
    bool enable_nfc = false;  // Near-field compensation

    // Room acoustics configuration
    bool enable_room_acoustics = true;
    RoomAcousticsModel room_model = RoomAcousticsModel::MEDIUM_ROOM;
    float reverberation_time_seconds = 1.0f;
    float early_reflection_delay_ms = 20.0f;
    float late_reverb_delay_ms = 40.0f;
    float room_dimensions_x = 10.0f;    // Room dimensions (meters)
    float room_dimensions_y = 8.0f;
    float room_dimensions_z = 3.0f;
    float wall_absorption = 0.3f;       // Wall absorption coefficient
    float air_absorption_coefficient = 0.001f;

    // Distance attenuation configuration
    bool enable_distance_attenuation = true;
    float reference_distance_meters = 1.0f;
    float rolloff_factor = 1.0f;
    float max_distance_meters = 100.0f;
    bool enable_air_absorption = true;
    float air_absorption_low_freq = 1000.0f;  // Air absorption low frequency (Hz)
    float air_absorption_high_freq = 8000.0f; // Air absorption high frequency (Hz)

    // Doppler effect configuration
    bool enable_doppler = false;
    float doppler_factor = 1.0f;
    float speed_of_sound = 343.0f;            // Speed of sound (m/s)
    bool enable_frequency_dependent_doppler = false;

    // Occlusion and obstruction
    bool enable_occlusion = false;
    bool enable_obstruction = false;
    bool enable_diffraction = false;
    float occlusion_factor = 1.0f;
    float obstruction_factor = 1.0f;
    float diffraction_factor = 1.0f;

    // Directivity configuration
    bool enable_directivity = false;
    float default_directivity = 0.0f;
    bool enable_frequency_dependent_directivity = false;

    // Audio routing configuration
    bool enable_multi_channel = true;
    uint32_t max_audio_sources = 128;
    uint32_t max_concurrent_sources = 64;
    bool enable_source_grouping = false;
    bool enable_source_priority = true;

    // Quality and performance
    bool high_quality_mode = true;
    bool enable_gpu_acceleration = true;
    bool enable_multi_threading = true;
    uint32_t num_threads = 0;  // 0 = auto
    bool enable_cache_optimization = true;
    bool enable_precomputation = true;

    // Real-time configuration
    bool enable_real_time_positioning = true;
    uint32_t position_update_rate_hz = 60;
    bool enable_smooth_transitions = true;
    float transition_time_ms = 50.0f;
    bool enable_predictive_positioning = false;

    // VR/AR configuration
    bool enable_vr_audio = false;
    bool enable_head_tracking = false;
    float head_position_tolerance_meters = 0.01f;
    float head_rotation_tolerance_degrees = 1.0f;
    bool enable_6dof_tracking = false;

    // Advanced features
    bool enable_wave_field_synthesis = false;
    bool enable_object_based_audio = false;
    bool enable_spatial_coding = false;
    bool enable_multi_room_acoustics = false;
    bool enable_dynamic_room_acoustics = false;

    // Monitoring and diagnostics
    bool enable_spatial_visualization = false;
    bool enable_source_tracking = true;
    bool enable_performance_monitoring = true;
    uint32_t visualization_update_rate_hz = 30;
    bool enable_diagnostics = false;

    // Buffer management
    uint32_t audio_buffer_size = 4096;
    uint32_t hrtf_buffer_size = 1024;
    uint32_t reverb_buffer_size = 16384;
    bool enable_buffer_pooling = true;

    // Validation
    bool validate_positions = true;
    bool enable_spatial_validation = true;
    float max_valid_distance_meters = 1000.0f;
};

/**
 * Abstract base class for spatial audio processors
 */
class SpatialAudioProcessor {
public:
    virtual ~SpatialAudioProcessor() = default;

    // Basic lifecycle
    virtual bool initialize(const SpatialAudioConfig& config) = 0;
    virtual void shutdown() = 0;
    virtual bool reset() = 0;

    // Processing
    virtual bool process(const float** inputs, float* output, uint32_t frame_count,
                          const std::vector<AudioSource*>& sources) = 0;
    virtual bool process_interleaved(const float* input, float* output, uint32_t frame_count,
                                      const std::vector<AudioSource*>& sources) = 0;

    // Source management
    virtual uint32_t add_source(const AudioSource& source) = 0;
    virtual bool remove_source(uint32_t source_id) = 0;
    virtual bool update_source(uint32_t source_id, const AudioSource& source) = 0;
    virtual AudioSource* get_source(uint32_t source_id) = 0;
    virtual std::vector<AudioSource*> get_all_sources() = 0;
    virtual std::vector<AudioSource*> get_sources_in_range(const float* center, float radius) = 0;

    // Source positioning
    virtual bool set_source_position(uint32_t source_id, float x, float y, float z) = 0;
    virtual bool set_source_velocity(uint32_t source_id, float vx, float vy, float vz) = 0;
    virtual bool get_source_position(uint32_t source_id, float& x, float& y, float& z) = 0;
    virtual bool get_source_velocity(uint32_t source_id, float& vx, float& vy, float& vz) = 0;

    // Source controls
    virtual bool set_source_gain(uint32_t source_id, float gain) = 0;
    virtual bool set_source_mute(uint32_t source_id, bool mute) = 0;
    virtual bool set_source_solo(uint32_t source_id, bool solo) = 0;
    virtual bool set_source_directivity(uint32_t source_id, float factor, float azimuth, float elevation) = 0;

    // Listener configuration
    virtual bool set_listener_position(float x, float y, float z) = 0;
    virtual bool set_listener_orientation(float yaw, float pitch, float roll) = 0;
    virtual bool set_listener_velocity(float vx, float vy, float vz) = 0;
    virtual bool get_listener_position(float& x, float& y, float& z) = 0;
    virtual bool get_listener_orientation(float& yaw, float& pitch, float& roll) = 0;

    // Room acoustics
    virtual bool set_room_dimensions(float x, float y, float z) = 0;
    virtual bool set_reverberation_time(float reverb_time) = 0;
    virtual bool set_wall_absorption(float absorption) = 0;
    virtual bool set_air_absorption(float low_freq, float high_freq, float coefficient) = 0;

    // HRTF configuration
    virtual bool load_hrtf_dataset(const std::string& dataset_path) = 0;
    virtual bool set_hrtf_index(uint32_t source_id, uint32_t hrtf_index) = 0;
    virtual bool enable_hrtf_for_source(uint32_t source_id, bool enabled) = 0;
    virtual bool set_hrtf_interpolation(HRTFInterpolation interpolation) = 0;

    // Configuration
    virtual bool set_spatial_type(SpatialAudioType type) = 0;
    virtual bool set_speaker_configuration(SpeakerConfiguration config) = 0;
    virtual bool update_config(const SpatialAudioConfig& config) = 0;
    virtual const SpatialAudioConfig& get_config() const = 0;

    // Preset management
    virtual bool save_preset(const std::string& name) = 0;
    virtual bool load_preset(const std::string& name) = 0;
    virtual std::vector<std::string> get_available_presets() const = 0;

    // Information
    virtual SpatialAudioType get_type() const = 0;
    virtual SpeakerConfiguration get_speaker_configuration() const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_version() const = 0;
    virtual std::string get_description() const = 0;
    virtual SpatialAudioStatistics get_statistics() const = 0;
    virtual void reset_statistics() = 0;

    // Advanced features
    virtual bool supports_real_time_positioning() const = 0;
    virtual bool supports_hrtf() const = 0;
    virtual bool supports_ambisonics() const = 0;
    virtual bool supports_gpu_acceleration() const = 0;
    virtual bool prefers_gpu_processing() const { return false; }
    virtual uint32_t get_max_sources() const = 0;
    virtual double get_expected_latency_ms() const = 0;
    virtual bool is_3d_enabled() const = 0;

    // VR/AR support
    virtual bool enable_head_tracking(bool enabled) = 0;
    virtual bool set_head_transform(const float* transform_matrix) = 0;
    virtual bool enable_vr_mode(bool enabled) = 0;

    // Event callbacks
    using SourceAddedCallback = std::function<void(uint32_t source_id, const AudioSource& source)>;
    using SourceRemovedCallback = std::function<void(uint32_t source_id)>;
    using SourceMovedCallback = std::function<void(uint32_t source_id, float x, float y, float z)>;

    void set_source_added_callback(SourceAddedCallback callback);
    void set_source_removed_callback(SourceRemovedCallback callback);
    void set_source_moved_callback(SourceMovedCallback callback);

    // Diagnostics
    virtual std::string get_diagnostics_report() const = 0;
    virtual bool validate_configuration() const = 0;
    virtual std::vector<std::string> test_spatial_capabilities() const = 0;
};

/**
 * Binaural spatial audio processor with HRTF
 */
class BinauralProcessor : public SpatialAudioProcessor {
public:
    BinauralProcessor();
    virtual ~BinauralProcessor();

    // SpatialAudioProcessor interface
    bool initialize(const SpatialAudioConfig& config) override;
    void shutdown() override;
    bool reset() override;

    bool process(const float** inputs, float* output, uint32_t frame_count,
                  const std::vector<AudioSource*>& sources) override;
    bool process_interleaved(const float* input, float* output, uint32_t frame_count,
                              const std::vector<AudioSource*>& sources) override;

    uint32_t add_source(const AudioSource& source) override;
    bool remove_source(uint32_t source_id) override;
    bool update_source(uint32_t source_id, const AudioSource& source) override;
    AudioSource* get_source(uint32_t source_id) override;
    std::vector<AudioSource*> get_all_sources() override;
    std::vector<AudioSource*> get_sources_in_range(const float* center, float radius) override;

    bool set_source_position(uint32_t source_id, float x, float y, float z) override;
    bool set_source_velocity(uint32_t source_id, float vx, float vy, float vz) override;
    bool get_source_position(uint32_t source_id, float& x, float& y, float& z) override;
    bool get_source_velocity(uint32_t source_id, float& vx, float& vy, float& vz) override;

    bool set_source_gain(uint32_t source_id, float gain) override;
    bool set_source_mute(uint32_t source_id, bool mute) override;
    bool set_source_solo(uint32_t source_id, bool solo) override;
    bool set_source_directivity(uint32_t source_id, float factor, float azimuth, float elevation) override;

    bool set_listener_position(float x, float y, float z) override;
    bool set_listener_orientation(float yaw, float pitch, float roll) override;
    bool set_listener_velocity(float vx, float vy, float vz) override;
    bool get_listener_position(float& x, float& y, float& z) override;
    bool get_listener_orientation(float& yaw, float& pitch, float& roll) override;

    bool set_room_dimensions(float x, float y, float z) override;
    bool set_reverberation_time(float reverb_time) override;
    bool set_wall_absorption(float absorption) override;
    bool set_air_absorption(float low_freq, float high_freq, float coefficient) override;

    bool load_hrtf_dataset(const std::string& dataset_path) override;
    bool set_hrtf_index(uint32_t source_id, uint32_t hrtf_index) override;
    bool enable_hrtf_for_source(uint32_t source_id, bool enabled) override;
    bool set_hrtf_interpolation(HRTFInterpolation interpolation) override;

    bool set_spatial_type(SpatialAudioType type) override;
    bool set_speaker_configuration(SpeakerConfiguration config) override;
    bool update_config(const SpatialAudioConfig& config) override;
    const SpatialAudioConfig& get_config() const override;

    bool save_preset(const std::string& name) override;
    bool load_preset(const std::string& name) override;
    std::vector<std::string> get_available_presets() const override;

    SpatialAudioType get_type() const override;
    SpeakerConfiguration get_speaker_configuration() const override;
    std::string get_name() const override;
    std::string get_version() const override;
    std::string get_description() const override;
    SpatialAudioStatistics get_statistics() const override;
    void reset_statistics() override;

    bool supports_real_time_positioning() const override;
    bool supports_hrtf() const override;
    bool supports_ambisonics() const override;
    bool supports_gpu_acceleration() const override;
    uint32_t get_max_sources() const override;
    double get_expected_latency_ms() const override;
    bool is_3d_enabled() const override;

    bool enable_head_tracking(bool enabled) override;
    bool set_head_transform(const float* transform_matrix) override;
    bool enable_vr_mode(bool enabled) override;

    // Binaural specific
    bool enable_individualized_hrtf(bool enabled);
    bool measure_individualized_hrtf(uint32_t subject_id);
    bool load_individualized_hrtf(uint32_t subject_id, const std::string& hrtf_path);
    uint32_t get_hrtf_index(uint32_t source_id) const;
    float get_azimuth_angle(uint32_t source_id) const;
    float get_elevation_angle(uint32_t source_id) const;

private:
    SpatialAudioConfig config_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> processing_{false};

    // Audio sources
    std::unordered_map<uint32_t, std::unique_ptr<AudioSource>> sources_;
    std::mutex sources_mutex_;
    uint32_t next_source_id_;

    // Listener state
    struct ListenerState {
        float position_x, position_y, position_z;
        float orientation_yaw, orientation_pitch, orientation_roll;
        float velocity_x, velocity_y, velocity_z;
        float head_transform_matrix[16];
        bool head_tracking_enabled;
        bool vr_mode_enabled;
    } listener_state_;

    // HRTF data
    struct HRTFData {
        std::vector<std::vector<float>> left_ir;
        std::vector<std::vector<float>> right_ir;
        std::vector<std::vector<float>> left_phase;
        std::vector<std::vector<float>> right_phase;
        std::vector<float> azimuths;
        std::vector<float> elevations;
        uint32_t ir_length;
        uint32_t sample_rate;
        uint32_t num_measurements;
        bool is_individualized;
        std::string subject_name;
    };

    std::unordered_map<uint32_t, HRTFData> hrtf_datasets_;
    HRTFInterpolation hrtf_interpolation_;
    std::string hrtf_dataset_path_;
    bool enable_hrtf_caching_;
    std::unordered_map<uint32_t, uint32_t> hrtf_cache_;

    // Room acoustics
    struct RoomAcoustics {
        float dimensions_x, dimensions_y, dimensions_z;
        float reverberation_time;
        float wall_absorption;
        float air_absorption_low_freq;
        float air_absorption_high_freq;
        float air_absorption_coefficient;
        RoomAcousticsModel model;
        std::vector<std::vector<float>> early_reflections;
        std::vector<float> reverb_impulse_response;
    } room_acoustics_;

    // Processing buffers
    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;
    std::vector<float> hrtf_left_buffer_;
    std::vector<float> hrtf_right_buffer_;
    std::vector<float> distance_attenuation_buffer_;
    std::vector<float> doppler_buffer_;

    // Statistics
    mutable std::mutex stats_mutex_;
    SpatialAudioStatistics statistics_;

    // Callbacks
    std::mutex callbacks_mutex_;
    SourceAddedCallback source_added_callback_;
    SourceRemovedCallback source_removed_callback_;
    SourceMovedCallback source_moved_callback_;

    // Threading
    std::thread processing_thread_;
    std::atomic<bool> shutdown_requested_{false};

    // Internal methods
    void process_sources(const std::vector<AudioSource*>& sources, uint32_t frame_count);
    void apply_hrtf(AudioSource& source, uint32_t frame_count);
    void apply_distance_attenuation(AudioSource& source, uint32_t frame_count);
    void apply_doppler_effect(AudioSource& source, uint32_t frame_count);
    void apply_occlusion(AudioSource& source, uint32_t frame_count);
    void apply_room_acoustics(AudioSource& source, uint32_t frame_count);

    float calculate_source_gain(const AudioSource& source);
    float calculate_distance_attenuation(float distance, float reference_distance, float rolloff_factor);
    float calculate_doppler_shift(const AudioSource& source);
    float calculate_occlusion_factor(const AudioSource& source);

    void calculate_spatial_position(AudioSource& source, float& azimuth, float& elevation, float& distance);
    void interpolate_hrtf(uint32_t source_id, float azimuth, float elevation,
                            std::vector<float>& left_ir, std::vector<float>& right_ir);

    void update_statistics(uint32_t frame_count, double processing_time_us);
    void notify_source_added(uint32_t source_id, const AudioSource& source);
    void notify_source_removed(uint32_t source_id);
    void notify_source_moved(uint32_t source_id, float x, float y, float z);

    void processing_thread_function();
    bool load_hrtf_dataset_internal(const std::string& dataset_path, uint32_t dataset_id);
    void unload_hrtf_dataset(uint32_t dataset_id);
    bool validate_hrtf_index(uint32_t hrtf_index) const;
};

/**
 * Ambisonics processor for spherical harmonics
 */
class AmbisonicsProcessor : public SpatialAudioProcessor {
public:
    AmbisonicsProcessor();
    virtual ~AmbisonicsProcessor();

    // SpatialAudioProcessor interface (partial implementation)
    bool initialize(const SpatialAudioConfig& config) override;
    void shutdown() override;
    bool reset() override;

    bool process(const float** inputs, float* output, uint32_t frame_count,
                  const std::vector<AudioSource*>& sources) override;
    bool process_interleaved(const float* input, float* output, uint32_t frame_count,
                              const std::vector<AudioSource*>& sources) override;

    // Ambisonics specific
    bool encode_ambisonics(const float* input, float* ambisonics_output,
                           uint32_t frame_count, uint32_t input_channels,
                           const float* source_positions);
    bool decode_ambisonics(const float* ambisonics_input, float* output,
                           uint32_t frame_count, const float* speaker_positions,
                           uint32_t output_channels);

    bool set_ambisonic_order(uint32_t order);
    bool enable_near_field_compensation(bool enabled);
    bool enable_nfc_distance(float distance);

    uint32_t get_ambisonic_order() const;
    uint32_t get_ambisonic_channels() const;

private:
    SpatialAudioConfig config_;
    std::atomic<bool> initialized_{false};

    // Ambisonics parameters
    uint32_t ambisonic_order_;
    uint32_t ambisonic_channels_;
    bool enable_nfc_;
    float nfc_distance_;

    // Encoding/decoding matrices
    std::vector<std::vector<float>> encoding_matrix_;
    std::vector<std::vector<float>> decoding_matrix_;
    std::vector<float> nfc_filter_;

    // Processing buffers
    std::vector<float> ambisonics_buffer_;
    std::vector<float> work_buffer_;

    // Statistics
    mutable std::mutex stats_mutex_;
    SpatialAudioStatistics statistics_;
};

/**
 * Spatial audio effects factory
 */
class SpatialAudioProcessorFactory {
public:
    // Binaural processors
    static std::unique_ptr<SpatialAudioProcessor> create_binaural_processor();
    static std::unique_ptr<SpatialAudioProcessor> create_hrtf_processor(const std::string& hrtf_path = "");
    static std::unique_ptr<SpatialAudioProcessor> create_individualized_hrtf_processor();

    // Ambisonics processors
    static std::unique_ptr<SpatialAudioProcessor> create_ambisonics_decoder(uint32_t order = 1,
                                                                         SpeakerConfiguration config = SpeakerConfiguration::STEREO);
    static std::unique_ptr<SpatialAudioProcessor> create_ambisonics_encoder(uint32_t order = 1,
                                                                         uint32_t input_channels = 1);
    static std::unique_ptr<SpatialAudioProcessor> create_3d_audio_processor();

    // Multi-channel processors
    static std::unique_ptr<SpatialAudioProcessor> create_multi_channel_panner(SpeakerConfiguration config);
    static std::unique_ptr<SpatialAudioProcessor> create_vector_base_amplitude_panner(uint32_t speakers);
    static std::unique_ptr<SpatialAudioProcessor> create_dolby_atmos_processor();

    // Game audio processors
    static std::unique_ptr<SpatialAudioProcessor> create_game_audio_processor();
    static std::unique_ptr<SpatialAudioProcessor> create_console_audio_processor();
    static std::unique_ptr<SpatialAudioProcessor> create_mobile_audio_processor();

    // Professional audio processors
    std::unique_ptr<SpatialAudioProcessor> create_cinema_audio_processor();
    std::unique_ptr<SpatialAudioProcessor> create_live_sound_processor();
    std::unique_ptr<SpatialAudioProcessor> create_automotive_audio_processor();

    // VR/AR processors
    static std::unique_ptr<SpatialAudioProcessor> create_vr_audio_processor();
    static std::unique_ptr<SpatialAudioProcessor> create_ar_audio_processor();
    static std::unique_ptr<SpatialAudioProcessor> create_360_video_processor();

    // Specialized processors
    static std::unique_ptr<SpatialAudioProcessor> create_wave_field_synthesis_processor();
    static std::unique_ptr<SpatialAudioProcessor> create_object_based_audio_processor();
    static std::unique_ptr<SpatialAudioProcessor> create_multi_room_processor();

    // Utility methods
    static std::vector<std::string> get_available_processor_types();
    static std::vector<std::string> get_available_speaker_configurations();
    static std::vector<std::string> get_available_hrtf_datasets();
    static std::vector<std::string> get_available_ambisonic_orders();
};

/**
 * Utility functions for spatial audio processing
 */
namespace spatial_audio_utils {
    // Coordinate transformations
    void cartesian_to_spherical(float x, float y, float z,
                                  float& azimuth, float& elevation, float& distance);
    void spherical_to_cartesian(float azimuth, float elevation, float distance,
                                  float& x, float& y, float& z);
    void polar_to_cartesian(float radius, float azimuth, float elevation,
                            float& x, float& y, float& z);

    // HRTF utilities
    float calculate_hrtf_interpolation_weight(float angle1, float angle2, float target_angle);
    uint32_t find_nearest_hrtf_index(float azimuth, float elevation,
                                        const std::vector<float>& azimuths,
                                        const std::vector<float>& elevations);
    bool load_hrtf_measurements(const std::string& file_path,
                                 std::vector<float>& azimuths,
                                 std::vector<float>& elevations,
                                 std::vector<std::vector<float>>& left_irs,
                                 std::vector<std::vector<float>>& right_irs);
    bool validate_hrtf_dataset(const std::vector<float>& azimuths,
                                const std::vector<float>& elevations,
                                const std::vector<std::vector<float>>& left_irs,
                                const std::vector<std::vector<float>>& right_irs);

    // Ambisonics utilities
    void calculate_ambisonics_encoding_matrix(uint32_t order,
                                            std::vector<std::vector<float>>& matrix);
    void calculate_ambisonics_decoding_matrix(uint32_t order,
                                              SpeakerConfiguration config,
                                              const std::vector<float>& speaker_positions,
                                              std::vector<std::vector<float>>& matrix);
    void calculate_nfc_filter(uint32_t order, float distance,
                                std::vector<float>& filter_coefficients);
    uint32_t get_ambisonic_channel_count(uint32_t order);
    uint32_t get_spherical_harmonic_order(uint32_t channel);

    // Distance attenuation
    float calculate_distance_attenuation(float distance, float reference_distance,
                                         float rolloff_factor);
    float calculate_air_absorption(float frequency, float distance, float coefficient);
    float calculate_inverse_distance_attenuation(float gain, float reference_distance,
                                                   float rolloff_factor);

    // Doppler effect
    float calculate_doppler_shift(float source_frequency, float source_velocity,
                                   float listener_velocity, float speed_of_sound);
    float calculate_doppler_factor(float relative_velocity, float speed_of_sound);

    // Directivity pattern
    float calculate_cardioid_pattern(float angle, float directivity_factor);
    float calculate_hypercardioid_pattern(float angle, float directivity_factor);
    float calculate_supercardioid_pattern(float angle, float directivity_factor);
    float calculate_directivity_gain(float source_angle, float source_azimuth,
                                      float source_elevation, float directivity_factor);

    // Room acoustics
    float calculate_sabine_reverberation_time(float room_volume, float total_absorption);
    float calculate_early_reflection_delay(float room_dimension, float speed_of_sound);
    float calculate_mode_frequency(float room_dimension, uint32_t mode, float speed_of_sound);
    void calculate_room_impulse_response(float room_x, float room_y, float room_z,
                                           float reverb_time, float absorption,
                                           std::vector<float>& impulse_response);

    // Occlusion and obstruction
    float calculate_occlusion_factor(float source_distance, float obstruction_distance);
    float calculate_transmission_loss(float obstruction_thickness, float frequency,
                                        float material_density, float sound_speed);
    float calculate_diffraction_angle(float wavelength, float obstacle_size);

    // Speaker array calculations
    void calculate_vbap_gains(const std::vector<float>& speaker_positions,
                             const std::vector<float>& source_position,
                             std::vector<float>& gains);
    void calculate_speaker_array_positions(SpeakerConfiguration config,
                                           std::vector<float>& positions,
                                           float room_width, float room_depth,
                                           float height = 2.0f);

    // Performance optimization
    bool is_sse_supported();
    bool is_avx_supported();
    void* aligned_malloc(size_t size, size_t alignment = 16);
    void aligned_free(void* ptr);
    void apply_vectorized_gain(float* buffer, uint32_t size, float gain);
    void apply_vectorized_mix(float* dest, const float* src,
                               uint32_t size, float mix_ratio);

    // Validation utilities
    bool validate_spatial_position(float x, float y, float z, float max_distance);
    bool validate_audio_source(const AudioSource& source);
    bool validate_hrtf_dataset(const std::string& dataset_path);
    bool validate_ambisonics_order(uint32_t order);
    bool is_valid_speaker_configuration(SpeakerConfiguration config);

    // Conversion utilities
    float meters_to_feet(float meters);
    float feet_to_meters(float feet);
    float degrees_to_radians(float degrees);
    float radians_to_degrees(float radians);
    float hz_to_mel(float frequency_hz);
    float mel_to_hz(float mel);
}

} // namespace vortex::core::dsp