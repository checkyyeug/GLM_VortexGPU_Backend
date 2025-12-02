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

#include "system/logger.hpp"
#include "hardware/hardware_monitor.hpp"
#include "network/realtime_streaming.hpp"

namespace vortex::core::processing {

/**
 * Adaptive Audio Processing for Vortex GPU Audio Backend
 *
 * This component provides intelligent, adaptive audio processing that dynamically
 * adjusts processing parameters based on system load, audio characteristics,
 * and real-time performance requirements. It optimizes CPU/GPU utilization,
 * maintains real-time constraints, and ensures consistent audio quality.
 *
 * Features:
 * - Dynamic CPU/GPU workload balancing
 * - Adaptive processing algorithms based on content analysis
 * - Real-time performance optimization
 * - Multi-quality level processing with automatic fallback
 * - Intelligent buffer management and sizing
 * - Power-aware processing optimization
 * - Content-aware algorithm selection
 * - Automatic quality scaling based on system resources
 * - Predictive resource allocation
 * - Machine learning-based optimization
 */

/**
 * Audio processing quality levels
 */
enum class ProcessingQuality {
    ULTRA_HIGH,     // Maximum quality, highest resource usage
    HIGH,           // High quality, good resource usage
    MEDIUM,         // Balanced quality and performance
    LOW,            // Reduced quality for low-end systems
    MINIMAL,        // Minimum quality for extreme constraints
    AUTO            // Automatically select based on system
};

/**
 * Processing modes for different scenarios
 */
enum class ProcessingMode {
    REAL_TIME,      // Real-time processing with strict deadlines
    OFFLINE,        // Offline processing with high quality focus
    INTERACTIVE,    // Interactive processing with low latency priority
    BATCH,          // Batch processing for efficiency
    STREAMING,      // Streaming processing with continuity focus
    ADAPTIVE        // Adaptive mode based on system conditions
};

/**
 * Content types for specialized processing
 */
enum class ContentType {
    MUSIC,          // Music content with rich harmonics
    SPEECH,         // Speech content requiring clarity
    NOISE,          // Noise content with filtering needs
    MIXED,          // Mixed content type
    AMBIENT,        // Ambient/background audio
    SIGNAL,         // Signal processing content
    UNKNOWN         // Automatically determined
};

/**
 * Adaptive strategy for resource management
 */
enum class AdaptiveStrategy {
    CONSERVATIVE,   // Prioritize stability over performance
    BALANCED,       // Balanced approach
    AGGRESSIVE,     // Prioritize performance over stability
    POWER_EFFICIENT,// Prioritize power efficiency
    QUALITY_FOCUSED // Prioritize audio quality
};

/**
 * System performance metrics for adaptation
 */
struct SystemPerformanceMetrics {
    // CPU metrics
    double cpu_utilization_percent = 0.0;
    double cpu_load_average_1min = 0.0;
    uint32_t cpu_cores_active = 0;
    uint64_t cpu_frequency_hz = 0;
    double cpu_temperature_celsius = 0.0;

    // GPU metrics
    double gpu_utilization_percent = 0.0;
    double gpu_memory_utilization_percent = 0.0;
    double gpu_temperature_celsius = 0.0;
    uint64_t gpu_memory_used_bytes = 0;
    double gpu_power_consumption_watts = 0.0;

    // Memory metrics
    double memory_utilization_percent = 0.0;
    uint64_t memory_available_bytes = 0;
    double memory_pressure_score = 0.0;

    // Thermal and power metrics
    double system_temperature_celsius = 0.0;
    double power_consumption_watts = 0.0;
    bool is_thermal_throttling = false;
    bool is_power_throttling = false;

    // Performance metrics
    double real_time_performance_score = 0.0;
    double audio_latency_ms = 0.0;
    double deadline_miss_rate_percent = 0.0;
    uint64_t audio_glitches_count = 0;

    // Timestamp
    uint64_t timestamp_microseconds = 0;
    bool is_valid = false;
};

/**
 * Content analysis results
 */
struct ContentAnalysis {
    ContentType primary_content_type = ContentType::UNKNOWN;
    ContentType secondary_content_type = ContentType::UNKNOWN;
    double content_confidence = 0.0;

    // Spectral characteristics
    double spectral_centroid = 0.0;        // Frequency center of mass
    double spectral_bandwidth = 0.0;       // Frequency spread
    double spectral_rolloff = 0.0;         // Frequency below which 85% of energy lies
    double spectral_flux = 0.0;            // Rate of spectral change
    double zero_crossing_rate = 0.0;       // Zero crossing frequency

    // Temporal characteristics
    double rms_level = 0.0;               // RMS amplitude
    double peak_level = 0.0;              // Peak amplitude
    double crest_factor = 0.0;            // Ratio of peak to RMS
    double dynamic_range = 0.0;           // Dynamic range
    double tempo_estimation = 0.0;        // Estimated tempo (BPM)
    double onset_rate = 0.0;              // Rate of onsets per second

    // Harmonic characteristics
    double harmonic_content = 0.0;        // Harmonic vs noise ratio
    double fundamental_frequency = 0.0;   // Fundamental frequency (Hz)
    std::vector<double> harmonic_peaks;   // Harmonic frequencies
    double inharmonicity = 0.0;           // Deviation from harmonic series

    // Noise characteristics
    double noise_level = 0.0;             // Overall noise level
    double signal_to_noise_ratio = 0.0;   // SNR in dB
    double noise_color = 0.0;             // Noise color (white/pink/brown)

    // Quality metrics
    double content_complexity = 0.0;      // Processing complexity score
    double processing_difficulty = 0.0;   // Difficulty to process
    double quality_requirements = 0.0;    // Quality level needed

    // Metadata
    std::chrono::steady_clock::time_point analysis_time;
    uint64_t duration_microseconds = 0;
    bool is_stable = false;
};

/**
 * Adaptive processing parameters
 */
struct AdaptiveParameters {
    // Quality settings
    ProcessingQuality target_quality = ProcessingQuality::AUTO;
    ProcessingMode processing_mode = ProcessingMode::ADAPTIVE;
    AdaptiveStrategy strategy = AdaptiveStrategy::BALANCED;

    // Performance targets
    double max_acceptable_latency_ms = 16.0;      // Target latency
    double min_real_time_score = 80.0;            // Minimum performance score
    uint32_t max_glitches_per_minute = 1;         // Maximum allowed glitches

    // Resource constraints
    double max_cpu_utilization_percent = 80.0;    // CPU limit
    double max_gpu_utilization_percent = 85.0;    // GPU limit
    double max_memory_utilization_percent = 75.0; // Memory limit
    double max_temperature_celsius = 80.0;        // Temperature limit

    // Adaptive behavior
    bool enable_gpu_acceleration = true;          // Use GPU when available
    bool enable_adaptive_quality = true;          // Adjust quality dynamically
    bool enable_content_aware_processing = true;  // Adapt to content type
    bool enable_predictive_optimization = true;   // Predict resource needs
    bool enable_power_management = true;          // Optimize for power

    // Timing parameters
    uint32_t adaptation_interval_ms = 1000;       // How often to adapt
    uint32_t analysis_window_size = 4096;         // Content analysis window
    uint32_t performance_history_size = 100;      // Performance history to consider
    uint32_t adaptation_lookahead_ms = 500;       // Lookahead for adaptation

    // Quality scaling factors
    double cpu_load_scaling_factor = 1.5;         // CPU load impact on quality
    double gpu_load_scaling_factor = 1.2;         // GPU load impact on quality
    double temperature_scaling_factor = 2.0;      // Temperature impact on quality
    double latency_scaling_factor = 1.8;          // Latency impact on quality

    // Fallback thresholds
    struct FallbackThresholds {
        double high_cpu_threshold = 90.0;         // CPU utilization for fallback
        double high_gpu_threshold = 95.0;         // GPU utilization for fallback
        double high_memory_threshold = 85.0;      // Memory utilization for fallback
        double high_temp_threshold = 85.0;        // Temperature for fallback
        double high_latency_threshold = 20.0;     // Latency for fallback
        double low_performance_threshold = 60.0;  // Performance score for fallback
    };

    FallbackThresholds fallback_thresholds;

    // Machine learning parameters
    bool enable_ml_optimization = false;          // Use ML for optimization
    std::string ml_model_path = "";               // Path to ML model
    uint32_t ml_training_samples = 10000;         // Training data size
    double ml_confidence_threshold = 0.8;         // Minimum ML confidence
};

/**
 * Processing quality configuration
 */
struct QualityConfiguration {
    ProcessingQuality quality_level = ProcessingQuality::MEDIUM;

    // Processing parameters per quality level
    uint32_t fft_size = 4096;                     // FFT size
    uint32_t overlap_factor = 4;                  // Overlap factor
    uint32_t window_type = 0;                     // Window function type
    double frequency_resolution_hz = 0.0;         // Frequency resolution
    uint32_t num_frequency_bins = 2048;           // Number of frequency bins

    // Quality-specific parameters
    uint32_t spectrum_resolution = 1024;          // Spectrum analyzer resolution
    uint32_t waveform_resolution = 1000;          // Waveform display resolution
    uint32_t vu_meter_update_rate = 60;           // VU meter update rate (Hz)
    double peak_hold_time_seconds = 1.0;          // Peak hold time
    double ballistics_integration_time = 0.3;     // VU meter integration time

    // Processing flags
    bool enable_high_precision = true;            // Use high precision processing
    bool enable_advanced_filtering = false;       // Advanced filtering algorithms
    bool enable_noise_reduction = false;          // Noise reduction processing
    bool enable_dynamic_range_compression = false; // Dynamic range processing
    bool enable_harmonic_analysis = false;        // Harmonic analysis
    bool enable_spectral_enhancement = false;     // Spectral enhancement

    // Performance parameters
    bool prefer_gpu_processing = true;            // Use GPU when available
    bool enable_multi_threading = true;           // Use multi-threading
    uint32_t thread_count = 0;                    // Thread count (0 = auto)
    bool enable_vector_processing = true;         // Use SIMD operations
    bool enable_cache_optimization = true;        // Optimize for cache

    // Buffer management
    uint32_t input_buffer_size = 8192;            // Input buffer size
    uint32_t output_buffer_size = 8192;           // Output buffer size
    uint32_t processing_buffer_count = 4;         // Number of processing buffers
    bool enable_buffer_pooling = true;            // Use buffer pooling
};

/**
 * Adaptation decision result
 */
struct AdaptationDecision {
    bool should_adapt = false;                    // Whether adaptation is needed
    ProcessingQuality new_quality = ProcessingQuality::MEDIUM;    // New quality level
    ProcessingMode new_mode = ProcessingMode::ADAPTIVE;           // New processing mode

    // Reason for adaptation
    enum class Reason {
        NONE,                                   // No adaptation needed
        HIGH_CPU_LOAD,                          // High CPU utilization
        HIGH_GPU_LOAD,                          // High GPU utilization
        HIGH_MEMORY_USAGE,                      // High memory usage
        HIGH_TEMPERATURE,                       // High temperature
        HIGH_LATENCY,                           // High audio latency
        LOW_PERFORMANCE,                        // Low performance score
        CONTENT_CHANGE,                         // Content type changed
        POWER_CONSERVATION,                     // Power saving mode
        USER_REQUEST,                           // User requested change
        PREDICTIVE_OPTIMIZATION                 // Predictive adaptation
    };

    Reason reason = Reason::NONE;
    double confidence = 0.0;                     // Confidence in decision
    std::string description;                     // Human-readable description

    // Expected impact
    double expected_latency_improvement = 0.0;   // Expected latency change
    double expected_quality_impact = 0.0;        // Expected quality impact
    double expected_cpu_reduction = 0.0;         // Expected CPU reduction
    double expected_gpu_reduction = 0.0;         // Expected GPU reduction

    // Timing
    std::chrono::steady_clock::time_point decision_time;
    uint32_t adaptation_delay_ms = 0;            // Delay before applying
};

/**
 * Adaptive Audio Processor
 */
class AdaptiveAudioProcessor {
public:
    AdaptiveAudioProcessor();
    ~AdaptiveAudioProcessor();

    // Lifecycle management
    bool initialize(const AdaptiveParameters& params);
    void shutdown();
    bool is_initialized() const { return initialized_; }
    bool is_adapting() const { return adapting_.load(); }

    // Control
    bool start_adaptation();
    void stop_adaptation();
    bool pause_adaptation();
    bool resume_adaptation();

    // Configuration
    void update_parameters(const AdaptiveParameters& params);
    const AdaptiveParameters& get_parameters() const { return params_; }
    QualityConfiguration get_quality_config(ProcessingQuality quality) const;

    // Real-time monitoring
    SystemPerformanceMetrics get_system_metrics() const;
    ContentAnalysis get_content_analysis() const;
    AdaptationDecision get_last_adaptation_decision() const;
    std::vector<AdaptationDecision> get_adaptation_history(size_t count = 10) const;

    // Manual control
    bool set_quality_level(ProcessingQuality quality);
    bool set_processing_mode(ProcessingMode mode);
    bool set_adaptive_strategy(AdaptiveStrategy strategy);
    ProcessingQuality get_current_quality() const { return current_quality_; }
    ProcessingMode get_current_mode() const { return current_mode_; }

    // Content analysis
    ContentAnalysis analyze_audio_content(const float* audio_data, size_t num_samples,
                                         uint32_t sample_rate, uint32_t channels);
    void enable_content_aware_processing(bool enabled);
    bool is_content_aware_processing_enabled() const;

    // Performance monitoring
    struct AdaptivePerformance {
        uint64_t total_adaptations = 0;
        uint64_t successful_adaptations = 0;
        double avg_adaptation_time_ms = 0.0;
        double max_adaptation_time_ms = 0.0;
        uint64_t quality_degradations = 0;
        uint64_t quality_improvements = 0;
        double stability_score = 0.0;
        double adaptation_efficiency = 0.0;
        std::chrono::steady_clock::time_point start_time;
    };

    AdaptivePerformance get_performance_stats() const;
    void reset_performance_stats();

    // Hardware monitoring integration
    void set_hardware_monitor(std::shared_ptr<hardware::HardwareMonitor> monitor);
    void set_streaming_interface(std::shared_ptr<network::RealtimeStreamer> streamer);

    // Event callbacks
    using AdaptationCallback = std::function<void(const AdaptationDecision&)>;
    using PerformanceCallback = std::function<void(const SystemPerformanceMetrics&)>;
    using QualityChangeCallback = std::function<void(ProcessingQuality, ProcessingQuality)>;

    void set_adaptation_callback(AdaptationCallback callback);
    void set_performance_callback(PerformanceCallback callback);
    void set_quality_change_callback(QualityChangeCallback callback);

    // Advanced features
    void enable_predictive_optimization(bool enabled);
    void enable_ml_optimization(bool enabled);
    bool is_predictive_optimization_enabled() const;
    bool is_ml_optimization_enabled() const;

    // Diagnostics and validation
    std::string get_diagnostics_report() const;
    bool validate_adaptation_setup() const;
    std::vector<std::string> test_adaptation_capabilities() const;

    // Export/Import
    std::string export_adaptation_state() const;
    bool import_adaptation_state(const std::string& state_data);

private:
    // Configuration and state
    AdaptiveParameters params_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> adapting_{false};
    std::atomic<bool> paused_{false};
    std::atomic<bool> shutdown_requested_{false};

    // Current processing state
    std::atomic<ProcessingQuality> current_quality_{ProcessingQuality::MEDIUM};
    std::atomic<ProcessingMode> current_mode_{ProcessingMode::ADAPTIVE};
    std::atomic<AdaptiveStrategy> current_strategy_{AdaptiveStrategy::BALANCED};

    // Quality configurations
    std::unordered_map<ProcessingQuality, QualityConfiguration> quality_configs_;

    // System monitoring
    std::shared_ptr<hardware::HardwareMonitor> hardware_monitor_;
    std::shared_ptr<network::RealtimeStreamer> streamer_;

    // Threads
    std::thread adaptation_thread_;
    std::thread monitoring_thread_;
    std::thread analysis_thread_;

    // Data storage
    mutable std::mutex metrics_mutex_;
    SystemPerformanceMetrics latest_system_metrics_;
    ContentAnalysis latest_content_analysis_;
    AdaptationDecision last_adaptation_decision_;
    std::deque<AdaptationDecision> adaptation_history_;
    std::deque<SystemPerformanceMetrics> performance_history_;

    // Performance tracking
    mutable std::mutex performance_mutex_;
    AdaptivePerformance performance_stats_;

    // Callbacks
    std::mutex callbacks_mutex_;
    AdaptationCallback adaptation_callback_;
    PerformanceCallback performance_callback_;
    QualityChangeCallback quality_change_callback_;

    // Adaptation logic
    void adaptation_thread();
    void monitoring_thread();
    void analysis_thread();

    // Core adaptation algorithms
    AdaptationDecision evaluate_adaptation_needs(const SystemPerformanceMetrics& metrics,
                                               const ContentAnalysis& content);
    AdaptationDecision create_cpu_based_adaptation(double cpu_utilization);
    AdaptationDecision create_gpu_based_adaptation(double gpu_utilization);
    AdaptationDecision create_memory_based_adaptation(double memory_utilization);
    AdaptationDecision create_temperature_based_adaptation(double temperature);
    AdaptationDecision create_latency_based_adaptation(double latency);
    AdaptationDecision create_content_based_adaptation(const ContentAnalysis& content);
    AdaptationDecision create_predictive_adaptation();

    // Quality configuration management
    void initialize_quality_configurations();
    QualityConfiguration create_quality_config(ProcessingQuality quality);
    void apply_quality_configuration(const QualityConfiguration& config);

    // Content analysis algorithms
    ContentAnalysis perform_content_analysis(const float* audio_data, size_t num_samples,
                                           uint32_t sample_rate, uint32_t channels);
    ContentType determine_content_type(const ContentAnalysis& analysis);
    double calculate_spectral_centroid(const float* spectrum, size_t size, uint32_t sample_rate);
    double calculate_spectral_bandwidth(const float* spectrum, size_t size, uint32_t sample_rate,
                                       double centroid);
    double calculate_spectral_rolloff(const float* spectrum, size_t size, uint32_t sample_rate,
                                     double threshold = 0.85);
    double calculate_spectral_flux(const float* spectrum_prev, const float* spectrum_curr,
                                   size_t size);

    // Performance prediction
    struct PerformancePrediction {
        double predicted_cpu_utilization = 0.0;
        double predicted_gpu_utilization = 0.0;
        double predicted_latency_ms = 0.0;
        double predicted_quality_score = 0.0;
        double confidence = 0.0;
        std::chrono::steady_clock::time_point prediction_time;
    };

    PerformancePrediction predict_performance(ProcessingQuality quality,
                                             const SystemPerformanceMetrics& current_metrics);
    PerformancePrediction predict_performance_with_ml(ProcessingQuality quality,
                                                    const SystemPerformanceMetrics& current_metrics);

    // Decision making
    ProcessingQuality select_optimal_quality(const SystemPerformanceMetrics& metrics,
                                            const ContentAnalysis& content);
    ProcessingMode select_optimal_mode(const SystemPerformanceMetrics& metrics);
    bool should_adapt_quality(const SystemPerformanceMetrics& metrics,
                             const ContentAnalysis& content);
    bool should_adapt_mode(const SystemPerformanceMetrics& metrics);

    // Adaptation execution
    void execute_adaptation(const AdaptationDecision& decision);
    void transition_quality(ProcessingQuality old_quality, ProcessingQuality new_quality);
    void transition_mode(ProcessingMode old_mode, ProcessingMode new_mode);
    void notify_adaptation(const AdaptationDecision& decision);

    // Monitoring and validation
    void monitor_system_performance();
    void validate_adaptation_results(const AdaptationDecision& decision,
                                    const SystemPerformanceMetrics& new_metrics);

    // Utility methods
    uint64_t get_current_timestamp_microseconds() const;
    void update_performance_stats(const AdaptationDecision& decision, bool success);
    std::string quality_to_string(ProcessingQuality quality) const;
    std::string mode_to_string(ProcessingMode mode) const;
    std::string strategy_to_string(AdaptiveStrategy strategy) const;
    double calculate_adaptation_confidence(const AdaptationDecision& decision) const;
};

/**
 * Factory for creating adaptive audio processors
 */
class AdaptiveAudioProcessorFactory {
public:
    static std::unique_ptr<AdaptiveAudioProcessor> create_default();
    static std::unique_ptr<AdaptiveAudioProcessor> create_high_performance();
    static std::unique_ptr<AdaptiveAudioProcessor> create_power_efficient();
    static std::unique_ptr<AdaptiveAudioProcessor> create_quality_focused();
    static std::unique_ptr<AdaptiveAudioProcessor> create_balanced();
};

/**
 * Utility functions for adaptive audio processing
 */
namespace adaptive_utils {
    // Quality level conversions
    ProcessingQuality string_to_quality(const std::string& quality_str);
    std::string quality_to_string(ProcessingQuality quality);
    ProcessingMode string_to_mode(const std::string& mode_str);
    std::string mode_to_string(ProcessingMode mode);
    AdaptiveStrategy string_to_strategy(const std::string& strategy_str);
    std::string strategy_to_string(AdaptiveStrategy strategy);

    // Performance calculations
    double calculate_real_time_score(const SystemPerformanceMetrics& metrics);
    double calculate_quality_score(ProcessingQuality quality, const ContentAnalysis& content);
    double calculate_efficiency_score(const SystemPerformanceMetrics& metrics,
                                    ProcessingQuality quality);
    double calculate_stability_score(const std::vector<AdaptationDecision>& history);

    // Decision support
    bool is_quality_downgrade_acceptable(ProcessingQuality current, ProcessingQuality proposed,
                                        const SystemPerformanceMetrics& metrics);
    double estimate_quality_impact(ProcessingQuality old_quality, ProcessingQuality new_quality);
    double estimate_resource_savings(ProcessingQuality old_quality, ProcessingQuality new_quality);

    // Content analysis utilities
    ContentType detect_content_type(const float* audio_data, size_t num_samples,
                                   uint32_t sample_rate, uint32_t channels);
    double calculate_content_complexity(const ContentAnalysis& analysis);
    bool content_requires_high_quality(const ContentAnalysis& analysis);

    // Validation and testing
    bool validate_quality_configuration(const QualityConfiguration& config);
    bool validate_adaptation_parameters(const AdaptiveParameters& params);
    std::vector<std::string> get_supported_quality_levels();
    std::vector<std::string> get_supported_processing_modes();
}

} // namespace vortex::core::processing