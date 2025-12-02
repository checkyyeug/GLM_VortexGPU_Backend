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

#include "system/logger.hpp"
#include "network/realtime_streaming.hpp"

namespace vortex::core::processing {

/**
 * Processing metrics collector for real-time audio processing performance
 *
 * This component collects detailed performance metrics for the audio processing
 * pipeline, enabling real-time monitoring and optimization of audio processing workloads.
 *
 * Features:
 * - Real-time audio processing latency tracking
 * - Throughput and buffer utilization monitoring
 * - DSP load and computational complexity analysis
 * - Memory usage tracking for audio buffers
 * - Thread utilization and scheduling analysis
 * - Pipeline efficiency measurement
 * - Quality of service metrics
 * - Performance bottleneck detection
 * - Real-time streaming integration
 * - Historical trend analysis
 * - Adaptive performance tuning
 */

struct ProcessingMetrics {
    // Timing metrics
    struct TimingMetrics {
        double total_processing_time_ms = 0.0;       // Total time for processing cycle
        double dsp_processing_time_ms = 0.0;       // DSP algorithm processing time
        double gpu_processing_time_ms = 0.0;       // GPU processing time
        double io_time_ms = 0.0;                     // I/O (file/network) time
        double overhead_time_ms = 0.0;               // System overhead time
        double lock_wait_time_ms = 0.0;              // Time waiting for locks
        double context_switch_time_ms = 0.0;         // Context switch overhead
        double scheduling_delay_ms = 0.0;           // Thread scheduling delay

        // Latency distribution
        std::vector<double> latency_samples;         // Individual sample latencies
        double min_latency_ms = std::numeric_limits<double>::max();
        double max_latency_ms = 0.0;
        double avg_latency_ms = 0.0;
        double p50_latency_ms = 0.0;                   // 50th percentile
        double p95_latency_ms = 0.0;                   // 95th percentile
        double p99_latency_ms = 0.0;                   // 99th percentile
        uint64_t deadline_misses = 0;                 // Number of missed deadlines
        double deadline_miss_rate_percent = 0.0;   // Percentage of missed deadlines

        // Jitter metrics
        double jitter_ms = 0.0;                        // Latency variation
        double max_jitter_ms = 0.0;
        double avg_jitter_ms = 0.0;
        uint64_t jitter_samples = 0;
    };

    TimingMetrics timing;

    // Throughput metrics
    struct ThroughputMetrics {
        uint64_t samples_processed = 0;            // Total samples processed
        uint64_t samples_dropped = 0;              // Samples dropped due to overload
        uint64_t samples_per_second = 0;            // Current samples/second
        double peak_samples_per_second = 0.0;         // Peak throughput
        double avg_samples_per_second = 0.0;          // Average throughput
        uint64_t blocks_processed = 0;               // Number of audio blocks
        uint64_t bytes_processed = 0;               // Bytes of audio data
        uint64_t bytes_per_second = 0;               // Current bytes/second

        // Quality metrics
        uint64_t clip_count = 0;                    // Number of clipping events
        double signal_to_noise_ratio_db = 0.0;       // SNR in dB
        double total_harmonic_distortion = 0.0;     // THD percentage
        double dynamic_range_db = 0.0;              // Dynamic range
        uint64_t underruns = 0;                     // Buffer underrun events
        uint64_t overruns = 0;                      // Buffer overrun events

        // Processing efficiency
        double cpu_efficiency = 0.0;                // CPU utilization efficiency
        double gpu_efficiency = 0.0;                // GPU utilization efficiency
        double memory_efficiency = 0.0;             // Memory bandwidth efficiency
        double cache_efficiency = 0.0;              // Cache hit rate

        // Real-time metrics
        bool meets_real_time_deadlines = true;
        double real_time_score = 0.0;                 // Real-time performance score
        uint32_t priority_inversions = 0;           // Priority inversion events
        double resource_contention_score = 0.0;     // Resource contention level
        uint64_t preemptive_events = 0;             // Preemptive multitasking events
    };

    ThroughputMetrics throughput;

    // Resource utilization metrics
    struct ResourceMetrics {
        double cpu_utilization_percent = 0.0;       // CPU usage percentage
        double memory_utilization_percent = 0.0;    // RAM usage percentage
        double disk_utilization_percent = 0.0;      // Disk I/O usage percentage
        double network_utilization_percent = 0.0;  // Network usage percentage

        // Thread metrics
        uint32_t active_threads = 0;               // Currently active threads
        uint32_t blocked_threads = 0;               // Threads waiting for resources
        uint32_t idle_threads = 0;                 // Idle threads
        double thread_utilization_efficiency = 0.0;   // Thread scheduling efficiency

        // Memory metrics
        uint64_t total_memory_allocated = 0;       // Total audio memory allocated
        uint64_t peak_memory_usage = 0;             // Peak memory usage
        uint64_t current_memory_usage = 0;          // Current memory usage
        double memory_allocation_rate = 0.0;         // Memory allocation rate
        uint64_t memory_fragmentation_count = 0;   // Memory fragmentation events

        // GPU metrics
        uint32_t gpu_active_kernels = 0;           // Active GPU kernels
        double gpu_utilization_percent = 0.0;        // GPU usage percentage
        uint64_t gpu_memory_used = 0;               // GPU memory used
        uint64_t gpu_bandwidth_utilized = 0;       // GPU bandwidth used
        double gpu_compute_efficiency = 0.0;       // GPU compute efficiency

        // I/O metrics
        uint64_t file_operations_per_second = 0;    // File I/O operations rate
        uint64_t network_packets_per_second = 0;   // Network packet rate
        double io_wait_percentage = 0.0;           // Time spent waiting for I/O
    };

    ResourceMetrics resources;

    // Pipeline efficiency metrics
    struct PipelineMetrics {
        uint32_t pipeline_stages = 0;               // Number of pipeline stages
        std::vector<double> stage_efficiency;      // Efficiency per stage
        double overall_pipeline_efficiency = 0.0;  // Overall pipeline efficiency
        uint32_t pipeline_stalls = 0;               // Pipeline stall events
        double pipeline_throughput = 0.0;           // Pipeline throughput
        uint64_t pipeline_cycles = 0;               // Total pipeline cycles
        double pipeline_cpi = 0.0;                   // Cycles per instruction
        uint64_t pipeline_bubbles = 0;               // Pipeline bubbles

        // Stage-specific metrics
        struct StageMetrics {
            std::string stage_name;
            double processing_time_ms = 0.0;
            double utilization_percent = 0.0;
            uint64_t items_processed = 0;
            uint64_t queue_depth = 0;
            double queue_utilization = 0.0;
            bool is_bottleneck = false;
        };

        std::vector<StageMetrics> stage_metrics;

        // Parallel processing metrics
        uint32_t parallel_workers = 0;               // Number of parallel workers
        double parallel_efficiency = 0.0;         // Parallel processing efficiency
        double load_balance_score = 0.0;           // Load balancing across workers
        uint64_t synchronization_overhead = 0;     // Synchronization overhead time
    };

    PipelineMetrics pipeline;

    // Quality of Service metrics
    struct QoSMetrics {
        uint32_t service_level_agreements_met = 0;    // SLAs met
        uint32_t service_level_agreements_violated = 0; // SLAs violated
        double service_level_agreement_compliance = 0.0; // SLA compliance percentage
        double user_experience_score = 0.0;          // User experience score
        uint64_t quality_events = 0;                  // Quality related events
        double quality_score = 0.0;                  // Overall quality score

        // Availability metrics
        double uptime_percentage = 0.0;            // System uptime
        double downtime_percentage = 0.0;          // System downtime
        uint64_t error_count = 0;                    // Total error count
        double mean_time_between_failures = 0.0;    // MTBF
        uint64_t recovery_time_ms = 0;              // Recovery time
    };

    QoSMetrics qos;

    // Historical data and trends
    struct HistoricalData {
        std::vector<double> latency_trend;         // Latency over time
        std::vector<double> throughput_trend;       // Throughput over time
        std::vector<double> efficiency_trend;      // Efficiency over time
        std::vector<double> resource_trend;         // Resource usage over time
        std::chrono::steady_clock::time_point last_update;
        size_t max_history_size = 1000;

        // Trend analysis
        double latency_trend_slope = 0.0;            // Slope of latency trend
        double throughput_trend_slope = 0.0;         // Slope of throughput trend
        bool is_performance_degrading = false;        // Performance degradation detected
        std::vector<std::string> anomalies;         // Detected anomalies
    };

    HistoricalData historical_data;

    // Metadata
    uint64_t timestamp_microseconds = 0;
    std::chrono::steady_clock::time_point collection_time;
    std::string processing_chain_id;
    std::string stage_name;
    bool is_valid = false;
    double collection_duration_ms = 0.0;
    uint32_t collection_version = 1;
};

/**
 * Processing metrics collector configuration
 */
struct ProcessingMetricsCollectorConfig {
    // Collection intervals (in milliseconds)
    uint32_t basic_interval_ms = 100;
    uint32_t detailed_interval_ms = 1000;
    uint32_t resource_interval_ms = 500;
    uint32_t pipeline_interval_ms = 2000;
    uint32_t qos_interval_ms = 5000;
    uint32_t trend_analysis_interval_ms = 10000;

    // Monitoring features
    bool enable_latency_tracking = true;
    bool enable_throughput_tracking = true;
    bool enable_resource_tracking = true;
    bool enable_pipeline_analysis = true;
    bool enable_qos_monitoring = true;
    bool enable_trend_analysis = true;
    bool enable_anomaly_detection = false;

    // Detailed monitoring options
    bool enable_detailed_latency_analysis = true;
    bool enable_detailed_profiling = false;
    bool enable_memory_tracking = true;
    bool enable_io_tracking = false;
    bool enable_thread_tracking = true;
    bool enable_gpu_tracking = true;

    // Performance optimization
    bool enable_low_overhead_mode = false;
    bool enable_caching = true;
    uint32_t cache_size = 1000;
    bool enable_batch_collection = true;
    uint32_t batch_size = 10;
    bool enable_sampling = false;
    double sampling_rate = 0.1; // 10% sampling rate

    // Real-time streaming
    bool enable_realtime_streaming = true;
    uint32_t streaming_interval_ms = 100;
    bool enable_compression = true;
    bool enable_delta_encoding = true;

    // Alert thresholds
    struct Thresholds {
        double latency_warning_ms = 5.0;
        double latency_critical_ms = 10.0;
        double throughput_warning_degradation = 10.0; // % degradation
        double throughput_critical_degradation = 25.0; // % degradation
        double cpu_utilization_warning = 80.0;
        double cpu_utilization_critical = 95.0;
        double memory_utilization_warning = 85.0;
        double memory_utilization_critical = 90.0;
        double deadline_miss_rate_warning = 1.0; // %
        double deadline_miss_rate_critical = 5.0; // %
        double pipeline_efficiency_warning = 70.0;
        double pipeline_efficiency_critical = 50.0;
    };

    Thresholds thresholds;

    // Advanced features
    bool enable_predictive_analysis = false;
    bool enable_adaptive_optimization = false;
    bool enable_automatic_tuning = false;
    bool enable_performance_profiling = true;
    uint32_t profiling_interval_seconds = 60;

    // Logging and diagnostics
    bool enable_detailed_logging = false;
    bool enable_performance_profiling = true;
    std::string log_file_path = "";
    bool enable_trace_collection = false;
};

/**
 * Processing metrics collector with real-time monitoring
 */
class ProcessingMetricsCollector {
public:
    ProcessingMetricsCollector();
    ~ProcessingMetricsCollector();

    // Lifecycle management
    bool initialize(const ProcessingMetricsCollectorConfig& config);
    void shutdown();
    bool is_initialized() const { return initialized_; }
    bool is_collecting() const { return collecting_.load(); }

    // Control
    bool start_collecting();
    void stop_collecting();
    bool pause_collecting();
    bool resume_collecting();

    // Data collection
    ProcessingMetrics collect_all_metrics();
    ProcessingMetrics collect_timing_metrics();
    ProcessingMetrics collect_throughput_metrics();
    ProcessingMetrics collect_resource_metrics();
    ProcessingMetrics collect_pipeline_metrics();
    ProcessingMetrics collect_qos_metrics();

    // Stage-specific collection
    void start_timing_measurement(const std::string& operation_id);
    void end_timing_measurement(const std::string& operation_id);
    void record_latency_sample(double latency_ms);
    void record_sample_processed();
    void record_sample_dropped();
    void record_clip_event();
    void record_buffer_underrun();
    void record_buffer_overrun();

    // Pipeline management
    void register_processing_stage(const std::string& stage_name, uint32_t stage_id);
    void set_pipeline_stage_count(uint32_t stage_count);
    void update_stage_metrics(uint32_t stage_id, double processing_time, bool is_bottleneck);

    // Real-time data access
    ProcessingMetrics get_latest_metrics() const;
    std::vector<ProcessingMetrics> get_metrics_history(size_t count = 100) const;
    ProcessingMetrics get_average_metrics(std::chrono::seconds duration) const;

    // Performance analysis
    struct PerformanceAnalysis {
        double overall_efficiency_score = 0.0;
        std::string primary_bottleneck;
        std::vector<std::string> optimization_suggestions;
        double performance_trend = 0.0;
        bool is_performance_degrading = false;
        std::chrono::steady_clock::time_point analysis_time;
        std::unordered_map<std::string, double> metric_scores;
    };

    PerformanceAnalysis analyze_performance() const;

    // Real-time streaming integration
    void set_streaming_interface(std::shared_ptr<network::RealtimeStreamer> streamer);
    bool enable_realtime_streaming(bool enabled);
    bool is_realtime_streaming_enabled() const;

    // Configuration
    void update_config(const ProcessingMetricsCollectorConfig& config);
    const ProcessingMetricsCollectorConfig& get_config() const { return config_; }

    // Threshold management
    bool set_thresholds(const ProcessingMetricsCollectorConfig::Thresholds& thresholds);
    ProcessingMetricsCollectorConfig::Thresholds get_thresholds() const;
    std::vector<std::string> check_threshold_violations(const ProcessingMetrics& metrics) const;

    // Alert system
    using MetricsCallback = std::function<void(const ProcessingMetrics&)>;
    using AlertCallback = std::function<void(const std::string& alert, const ProcessingMetrics&)>;

    void set_metrics_callback(MetricsCallback callback);
    void set_alert_callback(AlertCallback callback);

    // Performance monitoring
    struct CollectorPerformance {
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

    CollectorPerformance get_performance_stats() const;
    void reset_performance_stats();

    // Diagnostics
    std::string get_diagnostics_report() const;
    bool validate_collection_setup() const;
    std::vector<std::string> test_collection_capabilities() const;

    // Export/Import
    std::string export_metrics_json() const;
    std::string export_metrics_csv() const;
    bool import_metrics_json(const std::string& json_data);

    // Advanced features
    void enable_predictive_analysis(bool enabled);
    void enable_adaptive_optimization(bool enabled);
    void enable_automatic_tuning(bool enabled);
    std::vector<std::string> predict_performance_trends(std::chrono::seconds future_duration) const;
    std::vector<std::string> detect_anomalies() const;

private:
    // Configuration and state
    ProcessingMetricsCollectorConfig config_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> collecting_{false};
    std::atomic<bool> paused_{false};
    std::atomic<bool> shutdown_requested_{false};

    // Metrics storage
    mutable std::mutex metrics_mutex_;
    ProcessingMetrics latest_metrics_;
    std::vector<ProcessingMetrics> metrics_history_;

    // Timing measurements
    mutable std::mutex timing_mutex_;
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> timing_measurements_;

    // Pipeline stage information
    mutable std::mutex pipeline_mutex_;
    std::unordered_map<uint32_t, std::string> stage_names_;
    std::unordered_map<std::string, uint32_t> stage_ids_;
    uint32_t total_pipeline_stages_ = 0;

    // Streaming interface
    std::shared_ptr<network::RealtimeStreamer> streamer_;
    std::atomic<bool> realtime_streaming_enabled_{false};

    // Collection threads
    std::thread basic_thread_;
    std::thread detailed_thread_;
    std::thread resource_thread_;
    std::thread pipeline_thread_;
    std::thread qos_thread_;
    std::thread trend_analysis_thread_;
    std::thread streaming_thread_;

    // Performance tracking
    mutable std::mutex performance_mutex_;
    CollectorPerformance performance_stats_;

    // Caching
    mutable std::mutex cache_mutex_;
    std::unordered_map<std::string, ProcessingMetrics> metrics_cache_;

    // Event callbacks
    MetricsCallback metrics_callback_;
    AlertCallback alert_callback_;
    mutable std::mutex callbacks_mutex_;

    // Collection methods
    void basic_collection_thread();
    void detailed_collection_thread();
    void resource_collection_thread();
    void pipeline_collection_thread();
    void qos_collection_thread();
    void trend_analysis_thread();
    void streaming_thread();

    // Metrics collection implementations
    void collect_timing_data(ProcessingMetrics& metrics);
    void collect_throughput_data(ProcessingMetrics& metrics);
    void collect_resource_data(ProcessingMetrics& metrics);
    void collect_pipeline_data(ProcessingMetrics& metrics);
    void collect_qos_data(ProcessingMetrics& metrics);

    // Historical data management
    void update_historical_data(const ProcessingMetrics& metrics);
    void cleanup_historical_data();

    // Performance analysis
    void calculate_efficiency_scores(ProcessingMetrics& metrics);
    void detect_bottlenecks(ProcessingMetrics& metrics);
    void calculate_real_time_score(ProcessingMetrics& metrics);
    void analyze_trends(ProcessingMetrics& metrics);

    // Caching
    void update_cache(const std::string& key, const ProcessingMetrics& metrics);
    bool get_from_cache(const std::string& key, ProcessingMetrics& metrics) const;

    // Streaming
    void stream_metrics(const ProcessingMetrics& metrics);
    std::vector<uint8_t> serialize_metrics(const ProcessingMetrics& metrics) const;

    // Alert system
    void check_and_fire_alerts(const ProcessingMetrics& metrics);
    std::vector<std::string> analyze_threshold_violations(const ProcessingMetrics& metrics) const;

    // Statistical analysis
    void calculate_statistics(std::vector<double>& data, double& min_val, double& max_val, double& avg_val) const;
    void calculate_percentiles(const std::vector<double>& data, double& p50, double& p95, double& p99) const;
    double calculate_trend_slope(const std::vector<double>& data) const;

    // Utility methods
    uint64_t get_current_timestamp_microseconds() const;
    std::string generate_cache_key(const std::string& metric_type) const;
    void update_performance_stats(double collection_time_ms, bool success);
};

/**
 * Factory for creating processing metrics collectors
 */
class ProcessingMetricsCollectorFactory {
public:
    static std::unique_ptr<ProcessingMetricsCollector> create_default();
    static std::unique_ptr<ProcessingMetricsCollector> create_high_performance();
    static std::unique_ptr<ProcessingMetricsCollector> create_low_overhead();
    static std::unique_ptr<ProcessingMetricsCollector> create_comprehensive();
};

/**
 * Utility functions for processing metrics
 */
namespace processing_utils {
    // Time conversion utilities
    std::string format_duration(std::chrono::microseconds duration);
    std::string format_rate(uint64_t count, std::chrono::seconds period);
    std::string format_percentage(double percentage);

    // Statistical utilities
    double calculate_percentile(std::vector<double> data, double percentile);
    double calculate_standard_deviation(std::vector<double> data, double mean);
    double calculate_coefficient_of_variation(std::vector<double> data, double mean);

    // Performance calculation utilities
    double calculate_throughput_rate(uint64_t items, std::chrono::milliseconds duration);
    double calculate_efficiency_score(uint64_t useful_work, uint64_t total_work);
    double calculate_sla_compliance(uint32_t met_sla, uint32_t total_sla);
    double calculate_mtbf(uint64_t total_uptime, uint64_t failures);

    // Validation utilities
    bool is_valid_latency(double latency_ms);
    bool is_valid_throughput_rate(double rate);
    bool is_valid_utilization(double percentage);
    bool is_valid_processing_metrics(const ProcessingMetrics& metrics);

    // Analysis utilities
    std::string identify_primary_bottleneck(const ProcessingMetrics& metrics);
    std::vector<std::string> generate_optimization_suggestions(const ProcessingMetrics& metrics);
    bool is_performance_degrading(const ProcessingMetrics& metrics, std::chrono::seconds duration);
    double calculate_performance_score(const ProcessingMetrics& metrics);
}

} // namespace vortex::core::processing