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

#ifdef VORTEX_ENABLE_CUDA
#include <cuda_runtime.h>
#include <nvml.h>
#endif

#ifdef VORTEX_ENABLE_OPENCL
#include <CL/cl.h>
#endif

#ifdef VORTEX_ENABLE_VULKAN
#include <vulkan/vulkan.h>
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/time.h>
#endif

#include "system/logger.hpp"
#include "network/realtime_streaming.hpp"

namespace vortex::hardware {

/**
 * Real-time GPU utilization tracking for audio processing workloads
 *
 * This component provides detailed GPU monitoring with focus on audio
 * processing performance and real-time constraints.
 *
 * Features:
 * - Per-GPU utilization monitoring
 * - Memory usage tracking (VRAM)
 * - Thermal monitoring and throttling detection
 * - Power consumption monitoring
 * - Audio processing specific metrics
 * - Kernel execution tracking
 * - Real-time streaming of GPU metrics
 * - Multi-vendor support (NVIDIA, AMD, Intel)
 * - Cross-platform compatibility
 * - Low overhead monitoring
 */

struct GPUUtilizationMetrics {
    // Basic GPU information
    uint32_t device_id = 0;
    std::string device_name;
    std::string vendor_name;
    std::string driver_version;
    std::string cuda_capability;
    std::string opencl_version;
    std::string vulkan_version;
    uint64_t total_memory_bytes = 0;

    // Utilization metrics
    double gpu_utilization_percent = 0.0;      // Overall GPU utilization
    double graphics_utilization_percent = 0.0; // Graphics engine utilization
    double compute_utilization_percent = 0.0;   // Compute engine utilization
    double memory_utilization_percent = 0.0;    // Memory controller utilization
    double video_engine_utilization_percent = 0.0; // Video engine utilization

    // Memory metrics
    uint64_t memory_used_bytes = 0;
    uint64_t memory_free_bytes = 0;
    uint64_t memory_reserved_bytes = 0;
    double memory_bandwidth_utilization_percent = 0.0;
    uint64_t memory_bandwidth_mbps = 0;

    // Thermal metrics
    double temperature_gpu_celsius = 0.0;
    double temperature_memory_celsius = 0.0;
    bool is_thermal_throttling = false;
    uint32_t fan_speed_percent = 0;
    uint32_t fan_speed_rpm = 0;
    double power_limit_watts = 0.0;
    double power_usage_watts = 0.0;

    // Clock frequencies
    uint64_t graphics_clock_hz = 0;
    uint64_t memory_clock_hz = 0;
    uint64_t sm_clock_hz = 0;
    uint64_t video_clock_hz = 0;
    bool is_performance_state_maximum = false;
    uint32_t current_performance_state = 0;

    // Audio processing specific metrics
    struct AudioProcessingMetrics {
        uint32_t active_audio_kernels = 0;
        uint32_t total_audio_kernels = 0;
        double audio_kernel_utilization_percent = 0.0;
        uint64_t audio_memory_used_bytes = 0;
        double audio_processing_time_ms = 0.0;
        uint64_t audio_samples_processed = 0;
        uint64_t audio_samples_dropped = 0;
        double audio_throughput_samples_per_sec = 0.0;
        uint32_t audio_streams_active = 0;
        double audio_latency_ms = 0.0;
        uint64_t audio_buffer_utilization_percent = 0;
        bool is_audio_real_time_priority = false;
        double audio_efficiency_score = 0.0;
    };

    AudioProcessingMetrics audio_metrics;

    // Performance metrics
    struct PerformanceMetrics {
        double compute_efficiency = 0.0;        // FLOPS utilization vs theoretical
        double memory_efficiency = 0.0;        // Memory bandwidth utilization
        double power_efficiency = 0.0;         // Performance per watt
        uint64_t compute_cycles = 0;            // Total compute cycles
        uint64_t memory_transactions = 0;       // Total memory transactions
        double instruction_throughput_mips = 0.0; // Millions of instructions per second
        uint32_t sm_occupancy_percent = 0;      // Streaming multiprocessor occupancy
        double warp_efficiency_percent = 0.0;   // Warp scheduling efficiency
        uint64_t cache_hit_rate_percent = 0;     // L1/L2 cache hit rate
        uint64_t shared_memory_utilization_percent = 0; // Shared memory usage
    };

    PerformanceMetrics performance_metrics;

    // Real-time metrics
    struct RealTimeMetrics {
        double real_time_score = 0.0;          // Real-time performance score
        bool meets_real_time_deadlines = true;
        double deadline_miss_rate_percent = 0.0;
        uint64_t context_switches = 0;
        double preemptive_multitasking_overhead_percent = 0.0;
        bool is_in_power_save_mode = false;
        uint32_t pstate = 0;                  // Performance state
        double thermal_throttling_time_percent = 0.0;
        double clock_throttling_time_percent = 0.0;
        uint64_t power_budget_exceeded_events = 0;
    };

    RealTimeMetrics real_time_metrics;

    // Historical data
    struct HistoricalData {
        std::vector<double> utilization_history;      // Last N samples
        std::vector<double> temperature_history;
        std::vector<double> power_history;
        std::vector<double> memory_history;
        std::vector<double> audio_latency_history;
        std::chrono::steady_clock::time_point last_update;
        size_t max_history_size = 1000;
    };

    HistoricalData historical_data;

    // Timestamps and validity
    uint64_t timestamp_microseconds = 0;
    std::chrono::steady_clock::time_point collection_time;
    bool is_valid = false;
    double collection_duration_ms = 0.0;
};

/**
 * GPU utilization tracker configuration
 */
struct GPUTrackerConfig {
    // Collection intervals (in milliseconds)
    uint32_t utilization_interval_ms = 100;
    uint32_t memory_interval_ms = 200;
    uint32_t thermal_interval_ms = 1000;
    uint32_t power_interval_ms = 500;
    uint32_t audio_interval_ms = 50;
    uint32_t performance_interval_ms = 200;
    uint32_t clock_interval_ms = 1000;

    // Monitoring features
    bool enable_nvidia_ml = true;
    bool enable_nvidia_nsight = false;
    bool enable_amd_smi = true;
    bool enable_intel_gpu = true;
    bool enable_opencl_profiling = false;
    bool enable_vulkan_profiling = false;
    bool enable_cuda_profiling = false;
    bool enable_kernel_level_tracking = false;

    // Audio-specific monitoring
    bool enable_audio_kernel_tracking = true;
    bool enable_audio_memory_tracking = true;
    bool enable_audio_latency_tracking = true;
    bool enable_audio_efficiency_analysis = true;
    bool enable_real_time_priority_monitoring = true;

    // Performance optimization
    bool enable_low_overhead_mode = false;
    bool enable_caching = true;
    uint32_t cache_size = 1000;
    bool enable_batch_collection = true;
    uint32_t batch_size = 10;

    // Real-time streaming
    bool enable_realtime_streaming = true;
    uint32_t streaming_interval_ms = 100;
    bool enable_compression = true;
    bool enable_delta_encoding = true;

    // Thresholds and alerts
    struct Thresholds {
        double gpu_utilization_warning = 85.0;
        double gpu_utilization_critical = 95.0;
        double memory_utilization_warning = 80.0;
        double memory_utilization_critical = 90.0;
        double temperature_warning = 80.0;
        double temperature_critical = 90.0;
        double power_usage_warning = 0.8;  // 80% of power limit
        double power_usage_critical = 0.95; // 95% of power limit
        double audio_latency_warning = 5.0;
        double audio_latency_critical = 10.0;
        double real_time_score_warning = 70.0;
        double real_time_score_critical = 50.0;
    };

    Thresholds thresholds;

    // Advanced features
    bool enable_predictive_analysis = false;
    bool enable_anomaly_detection = false;
    bool enable_automatic_optimization = false;
    bool enable_detailed_profiling = false;

    // Logging and diagnostics
    bool enable_detailed_logging = false;
    bool enable_performance_profiling = true;
    std::string log_file_path = "";
    bool enable_kernel_tracing = false;
};

/**
 * GPU utilization tracker with real-time monitoring
 */
class GPUUtilizationTracker {
public:
    GPUUtilizationTracker();
    ~GPUUtilizationTracker();

    // Lifecycle management
    bool initialize(const GPUTrackerConfig& config);
    void shutdown();
    bool is_initialized() const { return initialized_; }
    bool is_tracking() const { return tracking_.load(); }

    // Control
    bool start_tracking();
    void stop_tracking();
    bool pause_tracking();
    bool resume_tracking();

    // Device management
    std::vector<uint32_t> get_available_devices() const;
    bool add_device(uint32_t device_id);
    bool remove_device(uint32_t device_id);
    bool is_device_available(uint32_t device_id) const;
    std::string get_device_name(uint32_t device_id) const;

    // Data collection
    GPUUtilizationMetrics collect_all_metrics(uint32_t device_id);
    GPUUtilizationMetrics collect_utilization_metrics(uint32_t device_id);
    GPUUtilizationMetrics collect_memory_metrics(uint32_t device_id);
    GPUUtilizationMetrics collect_thermal_metrics(uint32_t device_id);
    GPUUtilizationMetrics collect_power_metrics(uint32_t device_id);
    GPUUtilizationMetrics collect_audio_metrics(uint32_t device_id);
    GPUUtilizationMetrics collect_performance_metrics(uint32_t device_id);
    GPUUtilizationMetrics collect_clock_metrics(uint32_t device_id);

    // Multi-device operations
    std::vector<GPUUtilizationMetrics> collect_all_devices();
    std::unordered_map<uint32_t, GPUUtilizationMetrics> collect_all_devices_map();

    // Real-time data access
    GPUUtilizationMetrics get_latest_metrics(uint32_t device_id) const;
    std::vector<GPUUtilizationMetrics> get_metrics_history(uint32_t device_id, size_t count = 100) const;
    GPUUtilizationMetrics get_average_metrics(uint32_t device_id, std::chrono::seconds duration) const;

    // Audio processing specific functions
    void register_audio_kernel(const std::string& kernel_name, uint32_t device_id);
    void unregister_audio_kernel(const std::string& kernel_name);
    void start_audio_kernel_execution(const std::string& kernel_name, uint32_t device_id);
    void end_audio_kernel_execution(const std::string& kernel_name, uint32_t device_id);
    void track_audio_memory_allocation(uint32_t device_id, size_t bytes);
    void track_audio_memory_deallocation(uint32_t device_id, size_t bytes);

    // Performance analysis
    struct GPUPerformanceReport {
        std::string device_name;
        double average_utilization = 0.0;
        double peak_utilization = 0.0;
        double average_temperature = 0.0;
        double peak_temperature = 0.0;
        double average_power_usage = 0.0;
        double peak_power_usage = 0.0;
        double memory_efficiency = 0.0;
        double compute_efficiency = 0.0;
        double audio_efficiency = 0.0;
        double real_time_score = 0.0;
        uint64_t total_throttling_events = 0;
        std::chrono::steady_clock::time_point report_start;
        std::chrono::steady_clock::time_point report_end;
    };

    GPUPerformanceReport generate_performance_report(uint32_t device_id, std::chrono::seconds duration) const;

    // Real-time streaming integration
    void set_streaming_interface(std::shared_ptr<network::RealtimeStreamer> streamer);
    bool enable_realtime_streaming(bool enabled);
    bool is_realtime_streaming_enabled() const;

    // Configuration
    void update_config(const GPUTrackerConfig& config);
    const GPUTrackerConfig& get_config() const { return config_; }

    // Threshold management
    bool set_thresholds(const GPUTrackerConfig::Thresholds& thresholds);
    GPUTrackerConfig::Thresholds get_thresholds() const;
    std::vector<std::string> check_threshold_violations(uint32_t device_id) const;

    // Performance monitoring
    struct TrackerPerformance {
        uint64_t total_collections = 0;
        uint64_t successful_collections = 0;
        double avg_collection_time_ms = 0.0;
        double max_collection_time_ms = 0.0;
        uint64_t cache_hits = 0;
        uint64_t cache_misses = 0;
        double cache_hit_rate = 0.0;
        uint64_t streamed_messages = 0;
        uint64_t streaming_errors = 0;
        std::chrono::steady_clock::time_point start_time;
    };

    TrackerPerformance get_performance_stats() const;
    void reset_performance_stats();

    // Diagnostics
    std::string get_diagnostics_report() const;
    std::string get_device_diagnostics(uint32_t device_id) const;
    bool validate_tracking_setup() const;
    std::vector<std::string> test_tracking_capabilities() const;

    // Event callbacks
    using MetricsCallback = std::function<void(uint32_t, const GPUUtilizationMetrics&)>;
    using AlertCallback = std::function<void(uint32_t, const std::string& alert, const GPUUtilizationMetrics&)>;

    void set_metrics_callback(uint32_t device_id, MetricsCallback callback);
    void set_alert_callback(uint32_t device_id, AlertCallback callback);

    // Advanced features
    void enable_predictive_analysis(bool enabled);
    void enable_anomaly_detection(bool enabled);
    bool predict_utilization_trend(uint32_t device_id, std::chrono::seconds future_duration) const;
    std::vector<std::string> detect_anomalies(uint32_t device_id) const;

private:
    // Configuration and state
    GPUTrackerConfig config_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> tracking_{false};
    std::atomic<bool> paused_{false};
    std::atomic<bool> shutdown_requested_{false};

    // Device management
    std::vector<uint32_t> tracked_devices_;
    std::unordered_map<uint32_t, std::string> device_names_;
    mutable std::mutex devices_mutex_;

    // Metrics storage
    mutable std::mutex metrics_mutex_;
    std::unordered_map<uint32_t, GPUUtilizationMetrics> latest_metrics_;
    std::unordered_map<uint32_t, std::vector<GPUUtilizationMetrics>> metrics_history_;

    // Audio kernel tracking
    struct AudioKernelInfo {
        std::string name;
        uint32_t device_id;
        std::atomic<uint64_t> execution_count{0};
        std::atomic<double> total_execution_time_ms{0.0};
        std::atomic<double> max_execution_time_ms{0.0};
        std::atomic<bool> is_executing{false};
        std::chrono::steady_clock::time_point last_execution_start;
    };

    std::unordered_map<std::string, AudioKernelInfo> audio_kernels_;
    std::mutex audio_kernels_mutex_;

    // Streaming interface
    std::shared_ptr<network::RealtimeStreamer> streamer_;
    std::atomic<bool> realtime_streaming_enabled_{false};

    // Collection threads
    std::vector<std::thread> device_threads_;
    std::thread streaming_thread_;

    // Performance tracking
    mutable std::mutex performance_mutex_;
    TrackerPerformance performance_stats_;

    // Platform-specific handles
#ifdef VORTEX_ENABLE_NVML
    bool nvml_initialized_ = false;
    unsigned int nvml_device_count_ = 0;
    std::unordered_map<uint32_t, nvmlDevice_t> nvml_devices_;
#endif

#ifdef VORTEX_ENABLE_CUDA
    std::unordered_map<uint32_t, int> cuda_devices_;
#endif

#ifdef VORTEX_ENABLE_OPENCL
    cl_context opencl_context_;
    std::vector<cl_device_id> opencl_devices_;
#endif

    // Event callbacks
    std::unordered_map<uint32_t, MetricsCallback> metrics_callbacks_;
    std::unordered_map<uint32_t, AlertCallback> alert_callbacks_;
    mutable std::mutex callbacks_mutex_;

    // Collection methods
    void device_tracking_thread(uint32_t device_id);
    void streaming_thread();

    // Platform-specific implementations
    bool initialize_platform_monitors();
    void cleanup_platform_monitors();

    // NVIDIA GPU monitoring
    bool initialize_nvidia_monitoring();
    void cleanup_nvidia_monitoring();
    GPUUtilizationMetrics collect_nvidia_metrics(uint32_t device_id);

    // AMD GPU monitoring
    bool initialize_amd_monitoring();
    void cleanup_amd_monitoring();
    GPUUtilizationMetrics collect_amd_metrics(uint32_t device_id);

    // Intel GPU monitoring
    bool initialize_intel_monitoring();
    void cleanup_intel_monitoring();
    GPUUtilizationMetrics collect_intel_metrics(uint32_t device_id);

    // Audio-specific monitoring
    void update_audio_metrics(GPUUtilizationMetrics& metrics, uint32_t device_id);
    void calculate_audio_efficiency(GPUUtilizationMetrics& metrics);
    void calculate_real_time_score(GPUUtilizationMetrics& metrics);

    // Historical data management
    void update_historical_data(GPUUtilizationMetrics& metrics, uint32_t device_id);
    void cleanup_historical_data(uint32_t device_id);

    // Performance analysis
    void calculate_performance_metrics(GPUUtilizationMetrics& metrics, uint32_t device_id);
    void calculate_efficiency_scores(GPUUtilizationMetrics& metrics);

    // Caching
    void update_cache(const std::string& key, const GPUUtilizationMetrics& metrics);
    bool get_from_cache(const std::string& key, GPUUtilizationMetrics& metrics) const;

    // Streaming
    void stream_metrics(uint32_t device_id, const GPUUtilizationMetrics& metrics);
    std::vector<uint8_t> serialize_metrics(const GPUUtilizationMetrics& metrics) const;

    // Alert system
    void check_and_fire_alerts(uint32_t device_id, const GPUUtilizationMetrics& metrics);
    std::vector<std::string> analyze_threshold_violations(uint32_t device_id, const GPUUtilizationMetrics& metrics) const;

    // Utility methods
    uint64_t get_current_timestamp_microseconds() const;
    void update_performance_stats(double collection_time_ms, bool success);
    std::string generate_cache_key(uint32_t device_id, const std::string& metric_type) const;
};

/**
 * Factory for creating GPU utilization trackers
 */
class GPUUtilizationTrackerFactory {
public:
    static std::unique_ptr<GPUUtilizationTracker> create_default();
    static std::unique_ptr<GPUUtilizationTracker> create_high_performance();
    static std::unique_ptr<GPUUtilizationTracker> create_low_overhead();
    static std::unique_ptr<GPUUtilizationTracker> create_comprehensive();
};

/**
 * Utility functions for GPU utilization tracking
 */
namespace gpu_utils {
    // Conversion utilities
    double celsius_to_fahrenheit(double celsius);
    std::string format_memory_size(uint64_t bytes);
    std::string format_clock_frequency(uint64_t hz);
    std::string format_power_watts(double watts);

    // GPU calculation utilities
    double calculate_gpu_efficiency(double utilization, double temperature, double power);
    double calculate_thermal_headroom(double current_temp, double max_temp);
    double calculate_memory_bandwidth_utilization(uint64_t used_bandwidth, uint64_t max_bandwidth);

    // Vendor-specific utilities
    std::string get_gpu_vendor_name(uint32_t device_id);
    bool is_nvidia_gpu(uint32_t device_id);
    bool is_amd_gpu(uint32_t device_id);
    bool is_intel_gpu(uint32_t device_id);
    std::vector<std::string> get_gpu_capabilities(uint32_t device_id);

    // Audio processing utilities
    double calculate_audio_processing_score(const GPUUtilizationMetrics::AudioProcessingMetrics& audio);
    bool meets_real_time_audio_requirements(const GPUUtilizationMetrics& metrics);
    std::string analyze_audio_bottleneck(const GPUUtilizationMetrics& metrics);

    // Validation utilities
    bool is_valid_gpu_utilization(double utilization);
    bool is_valid_temperature(double temperature_celsius);
    bool is_valid_memory_usage(uint64_t used, uint64_t total);
    bool is_valid_gpu_metrics(const GPUUtilizationMetrics& metrics);
}

} // namespace vortex::hardware