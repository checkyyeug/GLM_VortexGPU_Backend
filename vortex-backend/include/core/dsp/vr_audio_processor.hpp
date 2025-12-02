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
#include "core/dsp/spatial_audio_processor.hpp"
#include "core/dsp/spectrum_analyzer.hpp"
#include "core/dsp/waveform_processor.hpp"
#include "core/dsp/vu_meter_processor.hpp"

namespace vortex {
namespace core {
namespace dsp {

/**
 * VR Audio processor with 6DOF tracking and spatial audio
 * for VR/AR applications with head pose integration
 */

enum class VRRenderingMode {
    STEREO_PANNING,      ///< Basic stereo panning
    BINAURAL,           ///< Binaural HRTF rendering
    AMBISONIC_DECODING, ///< Ambisonics to binaural
    OBJECT_BASED,       ///< Object-based spatial audio
    HYBRID,             ///< Hybrid rendering mode
    ADAPTIVE            ///< Adaptive based on hardware
};

enum class HeadTrackingMode {
    DISABLED,           ///< No head tracking
    ORIENTATION_ONLY,   ///< Orientation tracking only
    POSITION_AND_ORIENTATION, ///< Full 6DOF tracking
    RELATIVE_POSITION, ///< Relative position tracking
    ROOM_SCALE         ///< Room-scale tracking
};

enum class AudioDistanceModel {
    NONE,               ///< No distance attenuation
    INVERSE,            ///< Inverse distance model
    INVERSE_CLAMPED,    ///< Inverse with clamping
    EXPONENTIAL,        ///< Exponential distance model
    EXPONENTIAL_CLAMPED, ///< Exponential with clamping
    LINEAR,             ///< Linear distance model
    LINEAR_CLAMPED,     ///< Linear with clamping
    PHYSICAL            ///< Physically accurate model
};

enum class OcclusionMethod {
    NONE,               ///< No occlusion
    RAY_CASTING,        ///< Ray casting occlusion
    PATH_TRACING,       ///< Path tracing for accuracy
    APPROXIMATE,        ///< Fast approximate method
    FREQUENCY_DEPENDENT ///< Frequency-dependent occlusion
};

struct HeadPose {
    float position[3] = {0.0f, 0.0f, 0.0f};        ///< World position (x, y, z) in meters
    float orientation[4] = {0.0f, 0.0f, 0.0f, 1.0f}; ///< Quaternion rotation (x, y, z, w)
    float velocity[3] = {0.0f, 0.0f, 0.0f};        ///< Linear velocity (m/s)
    float angular_velocity[3] = {0.0f, 0.0f, 0.0f}; ///< Angular velocity (rad/s)
    float acceleration[3] = {0.0f, 0.0f, 0.0f};    ///< Linear acceleration (m/s²)
    float angular_acceleration[3] = {0.0f, 0.0f, 0.0f}; ///< Angular acceleration (rad/s²)
    uint64_t timestamp = 0;                        ///< Timestamp in nanoseconds
    float prediction_time = 0.0f;                  ///< Prediction time in seconds
    bool is_valid = true;                          ///< Pose validity flag
    float confidence = 1.0f;                       ///< Tracking confidence (0-1)
};

struct VRAudioSource {
    uint32_t id = 0;                               ///< Unique source identifier
    float position[3] = {0.0f, 0.0f, 0.0f};        ///< 3D position (x, y, z) in meters
    float velocity[3] = {0.0f, 0.0f, 0.0f};        ///< Source velocity (m/s)
    float orientation[4] = {0.0f, 0.0f, 0.0f, 1.0f}; ///< Source orientation
    float gain = 1.0f;                             ///< Source gain (linear)
    float directivity = 0.0f;                      ///< Directivity factor (0=omnidirectional, 1=highly directional)
    float inner_cone_angle = 360.0f;               ///< Inner cone angle in degrees
    float outer_cone_angle = 360.0f;               ///< Outer cone angle in degrees
    float outer_cone_gain = 0.0f;                  ///< Gain outside outer cone
    float min_distance = 0.1f;                     ///< Minimum distance for attenuation
    float max_distance = 1000.0f;                  ///< Maximum distance for attenuation
    float reference_distance = 1.0f;               ///< Reference distance for attenuation
    float rolloff_factor = 1.0f;                   ///< Rolloff factor for distance model
    AudioDistanceModel distance_model = AudioDistanceModel::INVERSE;
    bool enable_doppler = true;                    ///< Enable Doppler effects
    bool enable_occlusion = false;                 ///< Enable occlusion processing
    bool enable_reflection = false;                ///< Enable room reflections
    bool enable_reverb = false;                    ///< Enable reverb processing
    float doppler_factor = 1.0f;                   ///< Doppler effect intensity
    float air_absorption = 0.0f;                   ///< Air absorption factor
    bool is_relative = false;                      ///< Relative to listener
    bool is_looping = false;                       ///< Looping source
    bool is_muted = false;                         ///< Source muted state
    bool is_spatialized = true;                    ///< Spatial processing enabled
    int8_t priority = 0;                           ///< Source priority (-128 to 127)
    float fade_time = 0.0f;                        ///< Fade in/out time
    uint32_t streaming_buffer_size = 4096;         ///< Streaming buffer size
    std::string name;                              ///< Source name
    std::string tags;                              ///< Source tags
};

struct VRRoomAcoustics {
    float room_dimensions[3] = {10.0f, 8.0f, 3.0f}; ///< Room dimensions (width, height, depth) in meters
    float reflection_coefficient = 0.5f;             ///< Wall reflection coefficient (0-1)
    float reverb_decay_time = 1.5f;                  ///< Reverb RT60 in seconds
    float diffusion = 0.8f;                          ///< Sound diffusion factor (0-1)
    float absorption[6] = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f}; ///< Wall absorption (left,right,front,back,top,bottom)
    float air_absorption = 0.1f;                     ///< Air absorption coefficient
    float temperature = 20.0f;                       ///< Room temperature in Celsius
    float humidity = 50.0f;                          ///< Humidity percentage
    float atmospheric_pressure = 101325.0f;          ///< Atmospheric pressure in Pa
    bool enable_early_reflections = true;           ///< Enable early reflections
    bool enable_late_reverb = true;                 ///< Enable late reverb
    bool enable_hrtf_occlusion = true;              ///< HRTF-based occlusion
    int max_reflection_order = 2;                   ///< Maximum reflection order
};

struct VRRenderingQuality {
    enum Level { ULTRA_HIGH, HIGH, MEDIUM, LOW, MINIMAL };

    Level hrtf_quality = HIGH;                      ///< HRTF processing quality
    Level reverb_quality = MEDIUM;                  ///< Reverb processing quality
    Level occlusion_quality = MEDIUM;               ///< Occlusion processing quality
    Level reflection_quality = LOW;                 ///< Reflection processing quality
    bool enable_gpu_acceleration = true;            ///< Enable GPU acceleration
    bool enable_multi_threading = true;             ///< Enable multi-threading
    int max_concurrent_sources = 32;                ///< Maximum concurrent sources
    float processing_latency_target = 10.0f;        ///< Target latency in milliseconds
    int spatial_sampling_rate = 48000;              ///< Spatial processing sample rate
    int hrtf_impulse_response_length = 256;         ///< HRTF IR length in samples
};

struct VRAudioMetrics {
    uint32_t active_sources = 0;                    ///< Number of active sources
    uint32_t rendered_sources = 0;                  ///< Number of rendered sources
    double processing_time_ms = 0.0;               ///< Processing time per frame
    double total_cpu_usage_percent = 0.0;          ///< Total CPU usage
    double audio_cpu_usage_percent = 0.0;          ///< Audio processing CPU usage
    uint64_t dropped_frames = 0;                    ///< Number of dropped frames
    double average_latency_ms = 0.0;               ///< Average audio latency
    double peak_latency_ms = 0.0;                  ///< Peak audio latency
    double memory_usage_mb = 0.0;                  ///< Memory usage in MB
    double gpu_utilization_percent = 0.0;          ///< GPU utilization percentage
    double gpu_memory_usage_mb = 0.0;              ///< GPU memory usage in MB
    uint32_t hrtf_cache_hits = 0;                  ///< HRTF cache hits
    uint32_t hrtf_cache_misses = 0;                ///< HRTF cache misses
    float hrtf_cache_hit_ratio = 0.0f;             ///< HRTF cache hit ratio
    uint32_t occlusion_raycasts = 0;                ///< Number of occlusion raycasts
    uint32_t reflection_paths = 0;                  ///< Number of reflection paths calculated
    bool quality_adaptation_active = false;        ///< Quality adaptation active
    int quality_level = 0;                         ///< Current quality level
};

using VRAudioMetricsCallback = std::function<void(const VRAudioMetrics& metrics)>;
using HeadPoseCallback = std::function<void(const HeadPose& pose)>;

/**
 * VR Audio Processor with 6DOF tracking
 * Optimized for VR/AR applications with low latency requirements
 */
class VRAudioProcessor {
public:
    VRAudioProcessor();
    ~VRAudioProcessor();

    /**
     * Initialize VR audio processor
     * @param sample_rate Audio sample rate
     * @param buffer_size Audio buffer size
     * @param channels Number of audio channels
     * @param quality Rendering quality settings
     * @return True if initialization successful
     */
    bool initialize(int sample_rate, int buffer_size, int channels,
                   const VRRenderingQuality& quality = VRRenderingQuality{});

    /**
     * Shutdown processor and cleanup resources
     */
    void shutdown();

    /**
     * Process audio frame with VR spatialization
     * @param input_buffer Input audio buffer
     * @param output_buffer Output audio buffer (spatialized)
     * @param num_samples Number of samples to process
     * @param current_head_pose Current head pose for rendering
     * @return True if processing successful
     */
    bool processAudioFrame(const float* input_buffer, float* output_buffer,
                          size_t num_samples, const HeadPose& current_head_pose);

    /**
     * Add VR audio source
     * @param source Audio source configuration
     * @return Source ID if successful, 0 otherwise
     */
    uint32_t addAudioSource(const VRAudioSource& source);

    /**
     * Remove audio source
     * @param source_id Source ID to remove
     * @return True if successful
     */
    bool removeAudioSource(uint32_t source_id);

    /**
     * Update audio source properties
     * @param source_id Source ID
     * @param source Updated source configuration
     * @return True if successful
     */
    bool updateAudioSource(uint32_t source_id, const VRAudioSource& source);

    /**
     * Get audio source information
     * @param source_id Source ID
     * @return Source information if found
     */
    std::optional<VRAudioSource> getAudioSource(uint32_t source_id) const;

    /**
     * Set room acoustics parameters
     * @param acoustics Room acoustics configuration
     */
    void setRoomAcoustics(const VRRoomAcoustics& acoustics);

    /**
     * Get current room acoustics
     * @return Room acoustics configuration
     */
    VRRoomAcoustics getRoomAcoustics() const;

    /**
     * Set VR rendering mode
     * @param mode Rendering mode
     */
    void setRenderingMode(VRRenderingMode mode);

    /**
     * Get current rendering mode
     * @return Current rendering mode
     */
    VRRenderingMode getRenderingMode() const;

    /**
     * Set head tracking mode
     * @param mode Head tracking mode
     */
    void setHeadTrackingMode(HeadTrackingMode mode);

    /**
     * Get head tracking mode
     * @return Current head tracking mode
     */
    HeadTrackingMode getHeadTrackingMode() const;

    /**
     * Configure HRTF parameters
     * @param hrtf_path Path to HRTF data files
     * @param enable_individualized Enable individualized HRTF
     * @param ear_distance Ear distance in meters
     * @return True if configuration successful
     */
    bool configureHRTF(const std::string& hrtf_path, bool enable_individualized = false,
                      float ear_distance = 0.17f);

    /**
     * Enable/disable Doppler effects
     * @param enable Enable flag
     */
    void setDopplerEnabled(bool enable);

    /**
     * Enable/disable occlusion processing
     * @param enable Enable flag
     * @param method Occlusion method
     */
    void setOcclusionEnabled(bool enable, OcclusionMethod method = OcclusionMethod::APPROXIMATE);

    /**
     * Enable/disable reflection processing
     * @param enable Enable flag
     * @param max_order Maximum reflection order
     */
    void setReflectionEnabled(bool enable, int max_order = 2);

    /**
     * Set quality adaptation settings
     * @param enable Enable automatic quality adaptation
     * @param target_latency Target latency in milliseconds
     * @param min_quality Minimum quality level
     */
    void setQualityAdaptation(bool enable, float target_latency = 10.0f,
                             VRRenderingQuality::Level min_quality = VRRenderingQuality::LOW);

    /**
     * Update head pose (called from VR system)
     * @param pose New head pose
     */
    void updateHeadPose(const HeadPose& pose);

    /**
     * Predict head pose for given time
     * @param future_time Future time in seconds
     * @return Predicted head pose
     */
    HeadPose predictHeadPose(float future_time) const;

    /**
     * Get current head pose
     * @return Current head pose
     */
    HeadPose getCurrentHeadPose() const;

    /**
     * Set audio source audio data
     * @param source_id Source ID
     * @param audio_data Audio data buffer
     * @param num_samples Number of samples
     * @return True if successful
     */
    bool setAudioSourceData(uint32_t source_id, const float* audio_data, size_t num_samples);

    /**
     * Stream audio to source
     * @param source_id Source ID
     * @param audio_data Audio data chunk
     * @param num_samples Number of samples in chunk
     * @return True if streaming successful
     */
    bool streamAudioToSource(uint32_t source_id, const float* audio_data, size_t num_samples);

    /**
     * Set listener position and orientation
     * @param position Listener position (x, y, z)
     * @param orientation Listener orientation (quaternion)
     */
    void setListenerPose(const float position[3], const float orientation[4]);

    /**
     * Get listener pose
     * @return Current listener pose
     */
    HeadPose getListenerPose() const;

    /**
     * Set rendering quality
     * @param quality Quality settings
     */
    void setRenderingQuality(const VRRenderingQuality& quality);

    /**
     * Get rendering quality
     * @return Current quality settings
     */
    VRRenderingQuality getRenderingQuality() const;

    /**
     * Get VR audio metrics
     * @return Audio processing metrics
     */
    VRAudioMetrics getMetrics() const;

    /**
     * Register metrics callback
     * @param callback Metrics callback function
     */
    void setMetricsCallback(VRAudioMetricsCallback callback);

    /**
     * Register head pose callback
     * @param callback Head pose callback function
     */
    void setHeadPoseCallback(HeadPoseCallback callback);

    /**
     * Preload HRTF data for specific positions
     * @param positions Positions to preload
     * @return True if preloading successful
     */
    bool preloadHRTFPositions(const std::vector<std::array<float, 3>>& positions);

    /**
     * Clear HRTF cache
     */
    void clearHRTFCache();

    /**
     * Get HRTF cache statistics
     * @return Cache hit ratio and other stats
     */
    std::pair<float, size_t> getHRTFCacheStats() const;

    /**
     * Perform ray casting for occlusion
     * @param from_pos Start position
     * @param to_pos End position
     * @return Occlusion factor (0=fully occluded, 1=fully audible)
     */
    float performOcclusionRaycast(const float from_pos[3], const float to_pos[3]);

    /**
     * Calculate early reflections
     * @param source_pos Source position
     * @param listener_pos Listener position
     * @param room_dims Room dimensions
     * @param reflection_coefficient Wall reflection coefficient
     * @return Early reflection delays and gains
     */
    std::vector<std::pair<float, float>> calculateEarlyReflections(
        const float source_pos[3], const float listener_pos[3],
        const float room_dims[3], float reflection_coefficient);

    /**
     * Validate processor configuration
     * @return True if configuration is valid
     */
    bool validateConfiguration() const;

    /**
     * Reset processor to default state
     */
    void reset();

private:
    struct VRAudioSourceState {
        VRAudioSource config;
        std::vector<float> audio_buffer;
        size_t buffer_position = 0;
        float last_distance = 0.0f;
        bool is_active = false;
        float current_gain = 0.0f;
        float target_gain = 0.0f;
        uint64_t last_update_time = 0;
        std::vector<float> hrtf_ir_left;
        std::vector<float> hrtf_ir_right;
        float last_occlusion = 1.0f;
        std::queue<float> audio_queue;
        std::mutex queue_mutex;
    };

    struct PoseHistory {
        HeadPose pose;
        uint64_t timestamp;
    };

    // Core components
    std::unique_ptr<SpatialAudioProcessor> spatial_processor_;
    std::unique_ptr<SpectrumAnalyzer> spectrum_analyzer_;
    std::unique_ptr<WaveformProcessor> waveform_processor_;
    std::unique_ptr<VUMeterProcessor> vu_processor_;

    // VR-specific processing
    VRRenderingMode rendering_mode_ = VRRenderingMode::BINAURAL;
    HeadTrackingMode head_tracking_mode_ = HeadTrackingMode::POSITION_AND_ORIENTATION;
    AudioDistanceModel default_distance_model_ = AudioDistanceModel::INVERSE;
    OcclusionMethod occlusion_method_ = OcclusionMethod::APPROXIMATE;
    VRRenderingQuality quality_settings_;
    VRRoomAcoustics room_acoustics_;

    // Audio sources management
    std::unordered_map<uint32_t, std::unique_ptr<VRAudioSourceState>> audio_sources_;
    std::mutex sources_mutex_;
    std::atomic<uint32_t> next_source_id_{1};

    // Head tracking and pose prediction
    HeadPose current_head_pose_;
    HeadPose last_head_pose_;
    std::vector<PoseHistory> pose_history_;
    std::mutex pose_mutex_;
    static constexpr size_t POSE_HISTORY_SIZE = 10;

    // Processing state
    int sample_rate_ = 48000;
    int buffer_size_ = 512;
    int channels_ = 2;
    bool initialized_ = false;
    bool doppler_enabled_ = true;
    bool occlusion_enabled_ = false;
    bool reflection_enabled_ = false;
    bool quality_adaptation_enabled_ = false;
    float target_latency_ms_ = 10.0f;
    VRRenderingQuality::Level min_quality_level_ = VRRenderingQuality::LOW;

    // HRTF processing
    std::string hrtf_data_path_;
    bool hrtf_individualized_ = false;
    float ear_distance_ = 0.17f;
    std::unordered_map<uint64_t, std::pair<std::vector<float>, std::vector<float>>> hrtf_cache_;
    std::mutex hrtf_mutex_;

    // Performance metrics
    mutable std::mutex metrics_mutex_;
    VRAudioMetrics metrics_;
    VRAudioMetricsCallback metrics_callback_;
    HeadPoseCallback head_pose_callback_;
    std::chrono::high_resolution_clock::time_point last_metrics_update_;
    uint64_t total_frames_processed_ = 0;

    // Quality adaptation
    std::atomic<int> current_quality_level_{2};
    std::chrono::high_resolution_clock::time_point last_quality_adaptation_;
    std::vector<double> recent_processing_times_;
    static constexpr size_t PROCESSING_TIME_HISTORY = 60;

    // GPU acceleration
    bool gpu_initialized_ = false;
    void* gpu_context_ = nullptr;
    void* hrtf_convolution_kernel_ = nullptr;
    void* occlusion_ray_kernel_ = nullptr;
    void* reflection_kernel_ = nullptr;

    // Internal methods
    bool initializeGPU();
    void shutdownGPU();
    bool loadHRTFData();
    void updateAudioSourceGains();
    void processAudioSources(float* output_buffer, size_t num_samples);
    void applyDopplerEffect(VRAudioSourceState& source, float* buffer, size_t num_samples);
    void applyOcclusion(VRAudioSourceState& source, float* buffer, size_t num_samples);
    void applyReflections(VRAudioSourceState& source, float* buffer, size_t num_samples);
    void applyReverb(float* buffer, size_t num_samples);
    std::pair<std::vector<float>, std::vector<float>> getHRTFImpulseResponse(
        const float source_pos[3], const float listener_pos[3], const float listener_orientation[4]);
    void interpolateHeadPose(const HeadPose& from, const HeadPose& to, float t, HeadPose& result);
    void updateMetrics();
    void adaptQuality();
    bool validateSource(const VRAudioSource& source) const;
    void calculateDistanceAttenuation(const VRAudioSource& source, float distance, float& gain);
    void calculateDirectivityGain(const VRAudioSource& source, const float listener_pos[3], float& gain);
    void processAudioCPU(float* output_buffer, size_t num_samples);
    void processAudioGPU(float* output_buffer, size_t num_samples);
    void mixSources(float* output_buffer, const std::vector<const float*>& source_buffers,
                   size_t num_samples);
};

/**
 * VR Audio Factory for creating specialized processors
 */
class VRAudioFactory {
public:
    /**
     * Create VR audio processor for specific use case
     * @param use_case VR use case (gaming, social, training, etc.)
     * @param sample_rate Audio sample rate
     * @param buffer_size Buffer size
     * @return VR audio processor instance
     */
    static std::unique_ptr<VRAudioProcessor> createVRProcessor(
        const std::string& use_case, int sample_rate = 48000, int buffer_size = 512);

    /**
     * Create optimized VR processor for gaming
     * @param sample_rate Audio sample rate
     * @param buffer_size Buffer size
     * @return Gaming VR processor
     */
    static std::unique_ptr<VRAudioProcessor> createGamingVRProcessor(
        int sample_rate = 48000, int buffer_size = 512);

    /**
     * Create high-quality VR processor for social applications
     * @param sample_rate Audio sample rate
     * @param buffer_size Buffer size
     * @return Social VR processor
     */
    static std::unique_ptr<VRAudioProcessor> createSocialVRProcessor(
        int sample_rate = 48000, int buffer_size = 512);

    /**
     * Create professional VR processor for training/simulation
     * @param sample_rate Audio sample rate
     * @param buffer_size Buffer size
     * @return Professional VR processor
     */
    static std::unique_ptr<VRAudioProcessor> createProfessionalVRProcessor(
        int sample_rate = 48000, int buffer_size = 512);

    /**
     * Get recommended quality settings for VR platform
     * @param platform Target platform (Oculus, Vive, etc.)
     * @param target_fps Target frame rate
     * @return Recommended quality settings
     */
    static VRRenderingQuality getRecommendedQuality(const std::string& platform, float target_fps = 90.0f);
};

// Utility functions for VR audio
namespace vr_audio_utils {

    // Coordinate transformations
    void worldToListenerSpace(const float world_pos[3], const float listener_pos[3],
                             const float listener_orientation[4], float result[3]);
    void listenerToWorldSpace(const float listener_pos[3], const float listener_orientation[4],
                             float result[3]);

    // Distance calculations
    float calculateDistance(const float pos1[3], const float pos2[3]);
    float calculateRelativeVelocity(const float vel1[3], const float vel2[3],
                                   const float direction[3]);

    // Audio calculations
    float calculateDopplerShift(float source_velocity, float listener_velocity,
                               float sound_speed, float relative_distance_change);
    float calculateAirAbsorption(float distance, float frequency, float temperature, float humidity);

    // Geometric calculations
    bool isPointInRoom(const float point[3], const float room_dims[3]);
    float calculateRoomVolume(const float room_dims[3]);
    float calculateSurfaceArea(const float room_dims[3]);

    // Quality calculations
    VRRenderingQuality calculateOptimalQuality(double cpu_usage, double gpu_usage,
                                             float target_latency, int source_count);
    bool isQualityLevelAcceptable(VRRenderingQuality::Level level,
                                 const VRRenderingQuality& settings);
}

} // namespace dsp
} // namespace core
} // namespace vortex