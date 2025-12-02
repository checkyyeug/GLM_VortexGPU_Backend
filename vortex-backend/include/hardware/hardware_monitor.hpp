#pragma once

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>
#include <queue>

#ifdef VORTEX_ENABLE_NVML
#include <nvml.h>
#endif

#ifdef VORTEX_ENABLE_OPENCL
#include <CL/cl.h>
#endif

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <pdh.h>
#else
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

#include "system/logger.hpp"
#include "network/realtime_streaming.hpp"

namespace vortex::hardware {

/**
 * Real-time hardware monitoring system for Vortex GPU Audio Backend
 *
 * This component provides comprehensive hardware monitoring with real-time
 * data streaming capabilities for audio processing workloads.
 *
 * Features:
 * - CPU utilization and temperature monitoring
 * - GPU utilization, memory, temperature, and power monitoring
 * - Memory usage tracking (RAM, VRAM)
 * - Audio-specific metrics (DSP load, buffer utilization)
 * - Real-time data streaming via WebSocket
 * - Configurable monitoring intervals
 * - Performance optimization for minimal overhead
 * - Multi-GPU support
 * - Cross-platform compatibility
 */

struct HardwareMetrics {
    // CPU metrics
    double cpu_utilization_percent = 0.0;
    double cpu_temperature_celsius = 0.0;
    uint64_t cpu_frequency_hz = 0;
    uint32_t cpu_cores_active = 0;
    uint32_t cpu_cores_total = 0;
    double cpu_load_average_1min = 0.0;
    double cpu_load_average_5min = 0.0;
    double cpu_load_average_15min = 0.0;

    // Memory metrics
    uint64_t memory_total_bytes = 0;
    uint64_t memory_used_bytes = 0;
    uint64_t memory_available_bytes = 0;
    double memory_utilization_percent = 0.0;
    uint64_t memory_swap_total_bytes = 0;
    uint64_t memory_swap_used_bytes = 0;

    // GPU metrics (per GPU)
    struct GPUMetrics {
        uint32_t device_id = 0;
        std::string device_name;
        double gpu_utilization_percent = 0.0;
        double memory_utilization_percent = 0.0;
        uint64_t memory_total_bytes = 0;
        uint64_t memory_used_bytes = 0;
        uint64_t memory_free_bytes = 0;
        double temperature_celsius = 0.0;
        double power_consumption_watts = 0.0;
        uint64_t clock_frequency_hz = 0;
        uint32_t compute_utilization_percent = 0;
        uint64_t encoder_utilization_percent = 0;
        uint64_t decoder_utilization_percent = 0;
        double ecc_errors_corrected = 0.0;
        double ecc_errors_uncorrected = 0.0;
        uint32_t fan_speed_percent = 0;
        uint64_t performance_state = 0;
        std::string driver_version;
        std::string cuda_version;
        std::string vulkan_version;
    };

    std::vector<GPUMetrics> gpu_metrics;

    // Audio processing metrics
    struct AudioMetrics {
        double dsp_load_percent = 0.0;
        uint64_t samples_processed = 0;
        uint64_t samples_dropped = 0;
        double buffer_utilization_percent = 0.0;
        uint32_t active_channels = 0;
        uint32_t total_channels = 0;
        double processing_latency_ms = 0.0;
        double throughput_samples_per_second = 0.0;
        uint32_t real_time_priority_threads = 0;
        uint64_t audio_memory_usage_bytes = 0;
        double cpu_cycles_per_sample = 0.0;
        uint32_t gpu_kernels_executed = 0;
        double gpu_processing_time_ms = 0.0;
    };

    AudioMetrics audio_metrics;

    // System metrics
    struct SystemMetrics {
        uint64_t system_uptime_seconds = 0;
        uint64_t process_uptime_seconds = 0;
        uint64_t thread_count = 0;
        uint64_t handle_count = 0;
        double disk_utilization_percent = 0.0;
        uint64_t disk_read_bytes_per_sec = 0;
        uint64_t disk_write_bytes_per_sec = 0;
        double network_utilization_percent = 0.0;
        uint64_t network_bytes_in_per_sec = 0;
        uint64_t network_bytes_out_per_sec = 0;
        uint64_t page_faults = 0;
        uint64_t context_switches = 0;
    };

    SystemMetrics system_metrics;

    // Performance metrics
    struct PerformanceMetrics {
        double real_time_performance_score = 0.0;
        double audio_pipeline_efficiency = 0.0;
        double gpu_efficiency = 0.0;
        double memory_efficiency = 0.0;
        uint64_t interrupt_count = 0;
        double dpc_latency_us = 0.0;
        uint32_t cpu_cache_miss_rate_percent = 0;
        double io_wait_percent = 0.0;
        uint64_t thermal_throttling_events = 0;
        double power_efficiency_score = 0.0;
    };

    PerformanceMetrics performance_metrics;

    // Timestamps and validity
    uint64_t timestamp_microseconds = 0;
    std::chrono::steady_clock::time_point collection_time;
    bool is_valid = false;
    double collection_duration_ms = 0.0;
};

/**
 * Hardware monitoring configuration
 */
struct MonitorConfig {
    // Collection intervals (in milliseconds)
    uint32_t cpu_interval_ms = 1000;
    uint32_t gpu_interval_ms = 500;
    uint32_t memory_interval_ms = 2000;
    uint32_t audio_interval_ms = 100;
    uint32_t system_interval_ms = 5000;
    uint32_t performance_interval_ms = 1000;

    // GPU monitoring settings
    bool enable_nvidia_ml = true;
    bool enable_amd_smi = true;
    bool enable_intel_gpu = true;
    bool enable_opencl_monitoring = true;
    bool enable_vulkan_monitoring = true;
    bool enable_cuda_monitoring = true;

    // Audio-specific monitoring
    bool enable_dsp_monitoring = true;
    bool enable_buffer_monitoring = true;
    bool enable_latency_monitoring = true;
    bool enable_real_time_priority_monitoring = true;

    // System monitoring
    bool enable_disk_monitoring = true;
    bool enable_network_monitoring = true;
    bool enable_thermal_monitoring = true;
    bool enable_power_monitoring = true;

    // Performance tuning
    bool enable_high_precision_timing = true;
    bool enable_caching = true;
    uint32_t cache_size = 1000;
    bool enable_batch_collection = true;
    uint32_t batch_size = 10;

    // Streaming settings
    bool enable_realtime_streaming = true;
    uint32_t streaming_interval_ms = 100;
    bool enable_compression = true;
    uint8_t compression_level = 6;
    bool enable_differential_encoding = true;

    // Thresholds and alerts
    struct Thresholds {
        double cpu_utilization_warning = 80.0;
        double cpu_utilization_critical = 95.0;
        double memory_utilization_warning = 85.0;
        double memory_utilization_critical = 95.0;
        double gpu_utilization_warning = 85.0;
        double gpu_utilization_critical = 98.0;
        double gpu_temperature_warning = 80.0;
        double gpu_temperature_critical = 90.0;
        double audio_dsp_load_warning = 75.0;
        double audio_dsp_load_critical = 90.0;
        double audio_latency_warning = 5.0;
        double audio_latency_critical = 10.0;
    };

    Thresholds thresholds;

    // Logging and diagnostics
    bool enable_detailed_logging = false;
    bool enable_performance_profiling = false;
    uint32_t profiling_interval_seconds = 60;
    std::string log_file_path = "";
};

/**
 * Hardware monitor with real-time streaming capabilities
 */
class HardwareMonitor {
public:
    HardwareMonitor();
    ~HardwareMonitor();

    // Lifecycle management
    bool initialize(const MonitorConfig& config);
    void shutdown();
    bool is_initialized() const { return initialized_; }
    bool is_monitoring() const { return monitoring_.load(); }

    // Control
    bool start_monitoring();
    void stop_monitoring();
    bool pause_monitoring();
    bool resume_monitoring();

    // Data collection
    HardwareMetrics collect_all_metrics();
    HardwareMetrics collect_cpu_metrics();
    HardwareMetrics collect_gpu_metrics();
    HardwareMetrics collect_memory_metrics();
    HardwareMetrics collect_audio_metrics();
    HardwareMetrics collect_system_metrics();
    HardwareMetrics collect_performance_metrics();

    // Real-time data access
    HardwareMetrics get_latest_metrics() const;
    std::vector<HardwareMetrics> get_metrics_history(size_t count = 100) const;
    HardwareMetrics get_average_metrics(std::chrono::seconds duration) const;

    // GPU-specific functions
    std::vector<uint32_t> get_available_gpus() const;
    std::string get_gpu_name(uint32_t device_id) const;
    bool is_gpu_available(uint32_t device_id) const;

    // Threshold and alert management
    bool set_thresholds(const MonitorConfig::Thresholds& thresholds);
    MonitorConfig::Thresholds get_thresholds() const;
    std::vector<std::string> check_threshold_violations(const HardwareMetrics& metrics) const;

    // Real-time streaming integration
    void set_streaming_interface(std::shared_ptr<network::RealtimeStreamer> streamer);
    bool enable_realtime_streaming(bool enabled);
    bool is_realtime_streaming_enabled() const;

    // Configuration
    void update_config(const MonitorConfig& config);
    const MonitorConfig& get_config() const { return config_; }

    // Performance monitoring
    struct MonitorPerformance {
        uint64_t total_collections = 0;
        uint64_t successful_collections = 0;
        double avg_collection_time_ms = 0.0;
        double max_collection_time_ms = 0.0;
        double min_collection_time_ms = std::numeric_limits<double>::max();
        uint64_t cache_hits = 0;
        uint64_t cache_misses = 0;
        double cache_hit_rate = 0.0;
        uint64_t streamed_messages = 0;
        uint64_t streaming_errors = 0;
        std::chrono::steady_clock::time_point start_time;
    };

    MonitorPerformance get_performance_stats() const;
    void reset_performance_stats();

    // Diagnostics and validation
    std::string get_diagnostics_report() const;
    bool validate_monitoring_setup() const;
    std::vector<std::string> test_monitoring_capabilities() const;

    // Event callbacks
    using MetricsCallback = std::function<void(const HardwareMetrics&)>;
    using AlertCallback = std::function<void(const std::string& alert, const HardwareMetrics&)>;

    void set_metrics_callback(MetricsCallback callback);
    void set_alert_callback(AlertCallback callback);

    // Export/Import
    std::string export_metrics_json() const;
    bool import_metrics_json(const std::string& json_data);

    // Advanced features
    void enable_adaptive_monitoring(bool enabled);
    bool is_adaptive_monitoring_enabled() const;
    void set_monitoring_priority(int priority); // OS thread priority

private:
    // Configuration and state
    MonitorConfig config_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> monitoring_{false};
    std::atomic<bool> paused_{false};
    std::atomic<bool> shutdown_requested_{false};

    // Collection threads
    std::thread cpu_thread_;
    std::thread gpu_thread_;
    std::thread memory_thread_;
    std::thread audio_thread_;
    std::thread system_thread_;
    std::thread performance_thread_;
    std::thread streaming_thread_;

    // Synchronization
    mutable std::mutex metrics_mutex_;
    std::condition_variable metrics_cv_;
    std::queue<HardwareMetrics> metrics_queue_;
    std::vector<HardwareMetrics> metrics_history_;
    HardwareMetrics latest_metrics_;

    // Streaming interface
    std::shared_ptr<network::RealtimeStreamer> streamer_;
    std::atomic<bool> realtime_streaming_enabled_{false};

    // Performance tracking
    mutable std::mutex performance_mutex_;
    MonitorPerformance performance_stats_;

    // Cache for frequently accessed data
    mutable std::mutex cache_mutex_;
    std::unordered_map<std::string, std::pair<HardwareMetrics, std::chrono::steady_clock::time_point>> metrics_cache_;

    // Platform-specific handles
#ifdef _WIN32
    PDH_QUERY cpu_query_;
    PDH_HCOUNTER cpu_counter_;
    PDH_HCOUNTER memory_counter_;
    HANDLE process_handle_;
#else
    // Linux-specific handles
#endif

#ifdef VORTEX_ENABLE_NVML
    bool nvml_initialized_ = false;
    unsigned int nvml_device_count_ = 0;
#endif

    // Collection methods
    void cpu_collection_thread();
    void gpu_collection_thread();
    void memory_collection_thread();
    void audio_collection_thread();
    void system_collection_thread();
    void performance_collection_thread();
    void streaming_thread();

    // Platform-specific implementations
    bool initialize_platform_monitors();
    void cleanup_platform_monitors();

    // CPU monitoring
    double get_cpu_utilization();
    double get_cpu_temperature();
    uint64_t get_cpu_frequency();
    void get_cpu_load_averages(double& avg1, double& avg5, double& avg15);

    // Memory monitoring
    void get_memory_usage(uint64_t& total, uint64_t& used, uint64_t& available);
    void get_swap_usage(uint64_t& total, uint64_t& used);

    // GPU monitoring
    bool initialize_gpu_monitoring();
    void cleanup_gpu_monitoring();
    void collect_nvidia_gpu_metrics(std::vector<HardwareMetrics::GPUMetrics>& metrics);
    void collect_amd_gpu_metrics(std::vector<HardwareMetrics::GPUMetrics>& metrics);
    void collect_intel_gpu_metrics(std::vector<HardwareMetrics::GPUMetrics>& metrics);
    void collect_opencl_gpu_metrics(std::vector<HardwareMetrics::GPUMetrics>& metrics);

    // Audio metrics monitoring
    void collect_dsp_metrics(HardwareMetrics::AudioMetrics& metrics);
    void collect_buffer_metrics(HardwareMetrics::AudioMetrics& metrics);
    void collect_latency_metrics(HardwareMetrics::AudioMetrics& metrics);

    // System metrics
    void collect_system_info(HardwareMetrics::SystemMetrics& metrics);
    void collect_disk_metrics(HardwareMetrics::SystemMetrics& metrics);
    void collect_network_metrics(HardwareMetrics::SystemMetrics& metrics);
    void collect_thermal_metrics(HardwareMetrics& metrics);

    // Performance analysis
    void calculate_performance_metrics(HardwareMetrics& metrics);
    void calculate_real_time_performance_score(HardwareMetrics& metrics);
    void calculate_efficiency_metrics(HardwareMetrics& metrics);

    // Caching
    void update_cache(const std::string& key, const HardwareMetrics& metrics);
    bool get_from_cache(const std::string& key, HardwareMetrics& metrics) const;
    void cleanup_cache();

    // Streaming
    void stream_metrics(const HardwareMetrics& metrics);
    std::vector<uint8_t> serialize_metrics(const HardwareMetrics& metrics) const;

    // Adaptive monitoring
    void adjust_collection_intervals(const HardwareMetrics& metrics);
    bool should_increase_frequency(const HardwareMetrics& metrics) const;
    bool should_decrease_frequency(const HardwareMetrics& metrics) const;

    // Alert system
    void check_and_fire_alerts(const HardwareMetrics& metrics);
    std::vector<std::string> analyze_threshold_violations(const HardwareMetrics& metrics) const;

    // Thread priority and performance
    bool set_thread_priority(std::thread& thread, int priority);
    void optimize_thread_affinity();

    // Utility methods
    uint64_t get_current_timestamp_microseconds() const;
    double calculate_cache_hit_rate() const;
    void update_performance_stats(double collection_time_ms, bool success);
};

/**
 * Factory for creating hardware monitors with different configurations
 */
class HardwareMonitorFactory {
public:
    static std::unique_ptr<HardwareMonitor> create_default();
    static std::unique_ptr<HardwareMonitor> create_high_performance();
    static std::unique_ptr<HardwareMonitor> create_low_overhead();
    static std::unique_ptr<HardwareMonitor> create_comprehensive();
};

/**
 * Utility functions for hardware monitoring
 */
namespace hardware_utils {
    // Temperature conversion
    double celsius_to_fahrenheit(double celsius);
    double celsius_to_kelvin(double celsius);

    // Memory unit conversion
    std::string format_bytes(uint64_t bytes);
    std::string format_frequency(uint64_t hz);
    std::string format_duration(uint64_t microseconds);

    // Performance calculations
    double calculate_utilization_percentage(uint64_t used, uint64_t total);
    double calculate_efficiency_score(uint64_t useful_work, uint64_t total_work);
    double calculate_throughput(uint64_t items, std::chrono::milliseconds duration);

    // Hardware detection
    std::vector<std::string> detect_available_gpus();
    std::string detect_cpu_model();
    uint32_t detect_cpu_cores();
    uint64_t detect_total_memory();
    bool detect_nvidia_gpu_support();
    bool detect_amd_gpu_support();
    bool detect_intel_gpu_support();

    // Validation functions
    bool is_valid_cpu_utilization(double percentage);
    bool is_valid_temperature(double celsius);
    bool is_valid_memory_usage(uint64_t used, uint64_t total);
    bool is_valid_gpu_metrics(const HardwareMetrics::GPUMetrics& metrics);
}

} // namespace vortex::hardware