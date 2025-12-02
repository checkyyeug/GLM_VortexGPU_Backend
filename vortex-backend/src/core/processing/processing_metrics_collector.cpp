#include "core/processing/processing_metrics_collector.hpp"
#include "system/logger.hpp"
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>

namespace vortex::core::processing {

ProcessingMetricsCollector::ProcessingMetricsCollector()
    : initialized_(false)
    , collecting_(false)
    , paused_(false)
    , shutdown_requested_(false)
    , collection_start_time_(std::chrono::steady_clock::now())
    , total_audio_samples_processed_(0)
    , total_frames_processed_(0) {

    // Initialize performance counters
    performance_stats_.total_collections = 0;
    performance_stats_.successful_collections = 0;
    performance_stats_.avg_collection_time_ms = 0.0;
    performance_stats_.max_collection_time_ms = 0.0;
    performance_stats_.min_collection_time_ms = std::numeric_limits<double>::max();
}

ProcessingMetricsCollector::~ProcessingMetricsCollector() {
    shutdown();
}

bool ProcessingMetricsCollector::initialize(const ProcessingMetricsConfig& config) {
    std::lock_guard<std::mutex> lock(config_mutex_);

    if (initialized_) {
        Logger::warn("ProcessingMetricsCollector already initialized");
        return true;
    }

    config_ = config;

    Logger::info("Initializing ProcessingMetricsCollector with {}ms collection interval",
                 config.collection_interval_ms);

    // Validate configuration
    if (!validate_config(config_)) {
        Logger::error("Invalid processing metrics collector configuration");
        return false;
    }

    // Initialize threads
    try {
        collection_thread_ = std::thread(&ProcessingMetricsCollector::collection_thread, this);
        analysis_thread_ = std::thread(&ProcessingMetricsCollector::analysis_thread, this);

        if (config_.enable_realtime_streaming) {
            streaming_thread_ = std::thread(&ProcessingMetricsCollector::streaming_thread, this);
        }

        initialized_ = true;
        Logger::info("ProcessingMetricsCollector initialized successfully");
        return true;
    }
    catch (const std::exception& e) {
        Logger::error("Failed to initialize ProcessingMetricsCollector threads: {}", e.what());
        return false;
    }
}

void ProcessingMetricsCollector::shutdown() {
    if (!initialized_) {
        return;
    }

    Logger::info("Shutting down ProcessingMetricsCollector");

    // Signal threads to stop
    shutdown_requested_ = true;
    collecting_ = false;

    // Wake up all threads
    collection_cv_.notify_all();
    analysis_cv_.notify_all();
    streaming_cv_.notify_all();

    // Join threads
    if (collection_thread_.joinable()) {
        collection_thread_.join();
    }
    if (analysis_thread_.joinable()) {
        analysis_thread_.join();
    }
    if (streaming_thread_.joinable()) {
        streaming_thread_.join();
    }

    // Final performance report
    generate_final_performance_report();

    initialized_ = false;
    Logger::info("ProcessingMetricsCollector shutdown complete");
}

bool ProcessingMetricsCollector::start_collection() {
    if (!initialized_) {
        Logger::error("ProcessingMetricsCollector not initialized");
        return false;
    }

    if (collecting_) {
        Logger::warn("ProcessingMetricsCollector already collecting");
        return true;
    }

    collecting_ = true;
    paused_ = false;
    collection_start_time_ = std::chrono::steady_clock::now();

    // Start collection thread
    collection_cv_.notify_one();

    Logger::info("Started processing metrics collection");
    return true;
}

void ProcessingMetricsCollector::stop_collection() {
    collecting_ = false;
    paused_ = false;
    Logger::info("Stopped processing metrics collection");
}

bool ProcessingMetricsCollector::pause_collection() {
    if (!collecting_ || paused_) {
        return false;
    }

    paused_ = true;
    Logger::info("Paused processing metrics collection");
    return true;
}

bool ProcessingMetricsCollector::resume_collection() {
    if (!collecting_ || !paused_) {
        return false;
    }

    paused_ = false;
    collection_cv_.notify_one();
    Logger::info("Resumed processing metrics collection");
    return true;
}

ProcessingMetrics ProcessingMetricsCollector::collect_current_metrics() {
    ProcessingMetrics metrics;

    if (!initialized_) {
        return metrics;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Collect timing metrics
    collect_timing_metrics(metrics.timing);

    // Collect throughput metrics
    collect_throughput_metrics(metrics.throughput);

    // Collect resource metrics
    collect_resource_metrics(metrics.resources);

    // Collect pipeline metrics
    collect_pipeline_metrics(metrics.pipeline);

    // Collect QoS metrics
    collect_qos_metrics(metrics.qos);

    // Set timestamps
    metrics.timestamp_microseconds = get_current_timestamp_microseconds();
    metrics.collection_time = std::chrono::steady_clock::now();
    metrics.is_valid = true;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    metrics.collection_duration_ms = duration.count() / 1000.0;

    // Update performance statistics
    update_performance_stats(metrics.collection_duration_ms, true);

    // Store latest metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        latest_metrics_ = metrics;

        // Update history
        metrics_history_.push_back(metrics);
        if (metrics_history_.size() > config_.history_size) {
            metrics_history_.pop_front();
        }
    }

    // Trigger callbacks
    trigger_metrics_callbacks(metrics);

    return metrics;
}

ProcessingMetrics ProcessingMetricsCollector::get_latest_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return latest_metrics_;
}

std::vector<ProcessingMetrics> ProcessingMetricsCollector::get_metrics_history(size_t count) const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    std::vector<ProcessingMetrics> history;
    size_t start_idx = (metrics_history_.size() > count) ? metrics_history_.size() - count : 0;

    for (size_t i = start_idx; i < metrics_history_.size(); ++i) {
        history.push_back(metrics_history_[i]);
    }

    return history;
}

ProcessingMetrics ProcessingMetricsCollector::get_average_metrics(std::chrono::seconds duration) const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    if (metrics_history_.empty()) {
        return ProcessingMetrics();
    }

    auto cutoff_time = std::chrono::steady_clock::now() - duration;

    ProcessingMetrics average;
    size_t count = 0;

    for (const auto& metrics : metrics_history_) {
        if (metrics.collection_time >= cutoff_time) {
            // Accumulate metrics (simplified - in real implementation would average each field)
            average.timing.frame_processing_time_ms += metrics.timing.frame_processing_time_ms;
            average.throughput.samples_per_second += metrics.throughput.samples_per_second;
            average.resources.cpu_utilization_percent += metrics.resources.cpu_utilization_percent;
            average.pipeline.pipeline_efficiency_percent += metrics.pipeline.pipeline_efficiency_percent;
            average.qos.real_time_score += metrics.qos.real_time_score;
            count++;
        }
    }

    if (count > 0) {
        average.timing.frame_processing_time_ms /= count;
        average.throughput.samples_per_second /= count;
        average.resources.cpu_utilization_percent /= count;
        average.pipeline.pipeline_efficiency_percent /= count;
        average.qos.real_time_score /= count;
        average.is_valid = true;
    }

    return average;
}

ProcessingMetricsCollector::ProcessingPerformanceReport ProcessingMetricsCollector::generate_performance_report(std::chrono::seconds duration) const {
    ProcessingPerformanceReport report;

    auto now = std::chrono::steady_clock::now();
    report.report_start = now - duration;
    report.report_end = now;

    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        std::lock_guard<std::mutex> perf_lock(performance_mutex_);

        report.total_collections = performance_stats_.total_collections;
        report.successful_collections = performance_stats_.successful_collections;
        report.collection_success_rate = report.successful_collections > 0 ?
            static_cast<double>(report.successful_collections) / report.total_collections : 0.0;
        report.avg_collection_time_ms = performance_stats_.avg_collection_time_ms;
        report.max_collection_time_ms = performance_stats_.max_collection_time_ms;
        report.cache_hit_rate = performance_stats_.cache_hit_rate;

        if (streamer_) {
            report.streamed_messages = performance_stats_.streamed_messages;
            report.streaming_errors = performance_stats_.streaming_errors;
        }
    }

    // Calculate averages from history
    auto history = get_metrics_history();
    if (!history.empty()) {
        for (const auto& metrics : history) {
            if (metrics.collection_time >= report.report_start) {
                report.avg_processing_latency_ms += metrics.timing.frame_processing_time_ms;
                report.avg_throughput_samples_per_sec += metrics.throughput.samples_per_second;
                report.avg_cpu_utilization_percent += metrics.resources.cpu_utilization_percent;
                report.avg_gpu_utilization_percent += metrics.resources.gpu_utilization_percent;
                report.avg_real_time_score += metrics.qos.real_time_score;
                report.avg_pipeline_efficiency_percent += metrics.pipeline.pipeline_efficiency_percent;
                report.report_sample_count++;
            }
        }

        if (report.report_sample_count > 0) {
            report.avg_processing_latency_ms /= report.report_sample_count;
            report.avg_throughput_samples_per_sec /= report.report_sample_count;
            report.avg_cpu_utilization_percent /= report.report_sample_count;
            report.avg_gpu_utilization_percent /= report.report_sample_count;
            report.avg_real_time_score /= report.report_sample_count;
            report.avg_pipeline_efficiency_percent /= report.report_sample_count;
        }
    }

    return report;
}

void ProcessingMetricsCollector::register_processing_stage(const std::string& stage_name,
                                                           const std::vector<std::string>& dependencies) {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);

    PipelineStage stage;
    stage.name = stage_name;
    stage.dependencies = dependencies;
    stage.processing_time_us = 0;
    stage.calls_count = 0;
    stage.successful_calls = 0;
    stage.average_processing_time_us = 0.0;
    stage.max_processing_time_us = 0;
    stage.is_active = false;
    stage.last_execution_time = std::chrono::steady_clock::time_point();
    stage.queue_size = 0;
    stage.drop_rate_percent = 0.0;

    pipeline_stages_[stage_name] = stage;

    Logger::info("Registered processing stage: {}", stage_name);
}

void ProcessingMetricsCollector::unregister_processing_stage(const std::string& stage_name) {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);

    auto it = pipeline_stages_.find(stage_name);
    if (it != pipeline_stages_.end()) {
        pipeline_stages_.erase(it);
        Logger::info("Unregistered processing stage: {}", stage_name);
    }
}

void ProcessingMetricsCollector::start_stage_timing(const std::string& stage_name) {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);

    auto it = pipeline_stages_.find(stage_name);
    if (it != pipeline_stages_.end()) {
        it->second.is_active = true;
        it->second.last_execution_time = std::chrono::steady_clock::now();
        it->second.calls_count++;
    }
}

void ProcessingMetricsCollector::end_stage_timing(const std::string& stage_name) {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);

    auto it = pipeline_stages_.find(stage_name);
    if (it != pipeline_stages_.end() && it->second.is_active) {
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - it->second.last_execution_time);

        it->second.is_active = false;
        it->second.processing_time_us += duration.count();
        it->second.successful_calls++;
        it->second.max_processing_time_us = std::max(it->second.max_processing_time_us, duration.count());
        it->second.average_processing_time_us = static_cast<double>(it->second.processing_time_us) /
                                               it->second.successful_calls;
    }
}

void ProcessingMetricsCollector::update_stage_queue_size(const std::string& stage_name, size_t queue_size) {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);

    auto it = pipeline_stages_.find(stage_name);
    if (it != pipeline_stages_.end()) {
        it->second.queue_size = queue_size;
    }
}

void ProcessingMetricsCollector::record_audio_frame_processed(uint32_t sample_rate, uint32_t channels,
                                                            uint32_t frame_size) {
    total_audio_samples_processed_ += frame_size * channels;
    total_frames_processed_++;

    // Update current audio parameters
    current_audio_sample_rate_ = sample_rate;
    current_audio_channels_ = channels;
    current_audio_frame_size_ = frame_size;
}

void ProcessingMetricsCollector::record_buffer_operation(size_t buffer_size, double operation_time_ms) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);

    buffer_metrics_.total_buffer_operations++;
    buffer_metrics_.total_bytes_processed += buffer_size;
    buffer_metrics_.total_operation_time_ms += operation_time_ms;

    if (operation_time_ms > buffer_metrics_.max_operation_time_ms) {
        buffer_metrics_.max_operation_time_ms = operation_time_ms;
    }

    buffer_metrics_.average_operation_time_ms = buffer_metrics_.total_operation_time_ms /
                                               buffer_metrics_.total_buffer_operations;

    buffer_metrics_.throughput_mb_per_sec = (buffer_metrics_.total_bytes_processed / 1024.0 / 1024.0) /
                                           (buffer_metrics_.total_operation_time_ms / 1000.0);
}

void ProcessingMetricsCollector::record_throughput_measurement(size_t samples_processed,
                                                             double processing_time_ms) {
    std::lock_guard<std::mutex> lock(throughput_mutex_);

    throughput_measurements_.push_back({samples_processed, processing_time_ms,
                                       std::chrono::steady_clock::now()});

    // Keep only recent measurements
    if (throughput_measurements_.size() > config_.max_throughput_samples) {
        throughput_measurements_.pop_front();
    }
}

void ProcessingMetricsCollector::set_streaming_interface(std::shared_ptr<network::RealtimeStreamer> streamer) {
    std::lock_guard<std::mutex> lock(streaming_mutex_);
    streamer_ = streamer;

    if (streamer_) {
        Logger::info("Processing metrics streaming interface configured");
    }
}

bool ProcessingMetricsCollector::enable_realtime_streaming(bool enabled) {
    std::lock_guard<std::mutex> lock(streaming_mutex_);

    if (enabled && !streamer_) {
        Logger::warn("Cannot enable streaming - no streaming interface configured");
        return false;
    }

    config_.enable_realtime_streaming = enabled;
    realtime_streaming_enabled_ = enabled;

    Logger::info("Real-time streaming {}", enabled ? "enabled" : "disabled");
    return true;
}

void ProcessingMetricsCollector::set_metrics_callback(MetricsCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    metrics_callback_ = callback;
}

void ProcessingMetricsCollector::set_alert_callback(AlertCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    alert_callback_ = callback;
}

void ProcessingMetricsCollector::collection_thread() {
    Logger::info("Processing metrics collection thread started");

    while (!shutdown_requested_) {
        std::unique_lock<std::mutex> lock(collection_mutex_);
        collection_cv_.wait(lock, [this] {
            return collecting_.load() && !paused_.load() || shutdown_requested_;
        });

        if (shutdown_requested_) {
            break;
        }

        // Collect metrics
        collect_current_metrics();

        // Sleep for collection interval
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.collection_interval_ms));
    }

    Logger::info("Processing metrics collection thread stopped");
}

void ProcessingMetricsCollector::analysis_thread() {
    Logger::info("Processing metrics analysis thread started");

    while (!shutdown_requested_) {
        std::unique_lock<std::mutex> lock(analysis_mutex_);
        analysis_cv_.wait(lock, [this] {
            return initialized_ || shutdown_requested_;
        });

        if (shutdown_requested_) {
            break;
        }

        // Perform analysis
        if (config_.enable_adaptive_analysis) {
            perform_adaptive_analysis();
        }

        if (config_.enable_anomaly_detection) {
            detect_anomalies();
        }

        if (config_.enable_prediction) {
            predict_performance_trends();
        }

        // Sleep for analysis interval
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.analysis_interval_ms));
    }

    Logger::info("Processing metrics analysis thread stopped");
}

void ProcessingMetricsCollector::streaming_thread() {
    Logger::info("Processing metrics streaming thread started");

    while (!shutdown_requested_) {
        std::unique_lock<std::mutex> lock(streaming_mutex_);
        streaming_cv_.wait(lock, [this] {
            return realtime_streaming_enabled_.load() || shutdown_requested_;
        });

        if (shutdown_requested_) {
            break;
        }

        if (realtime_streaming_enabled_ && streamer_) {
            auto metrics = get_latest_metrics();
            if (metrics.is_valid) {
                stream_processing_metrics(metrics);
            }
        }

        // Sleep for streaming interval
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.streaming_interval_ms));
    }

    Logger::info("Processing metrics streaming thread stopped");
}

void ProcessingMetricsCollector::collect_timing_metrics(ProcessingMetrics::TimingMetrics& timing) {
    auto now = std::chrono::steady_clock::now();

    // Calculate frame processing time
    timing.frame_processing_time_ms = calculate_average_frame_processing_time();
    timing.max_frame_processing_time_ms = calculate_max_frame_processing_time();
    timing.min_frame_processing_time_ms = calculate_min_frame_processing_time();

    // Calculate pipeline latency
    timing.pipeline_latency_ms = calculate_pipeline_latency();

    // Audio buffer metrics
    timing.audio_buffer_latency_ms = calculate_audio_buffer_latency();
    timing.audio_buffer_fill_ratio = calculate_audio_buffer_fill_ratio();

    // Memory allocation timing
    timing.memory_allocation_time_us = calculate_memory_allocation_time();
    timing.max_memory_allocation_time_us = calculate_max_memory_allocation_time();

    // Thread synchronization timing
    timing.thread_sync_time_us = calculate_thread_sync_time();
    timing.max_thread_sync_time_us = calculate_max_thread_sync_time();

    // Timestamps
    timing.timestamp = now;
}

void ProcessingMetricsCollector::collect_throughput_metrics(ProcessingMetrics::ThroughputMetrics& throughput) {
    // Sample processing throughput
    throughput.samples_per_second = calculate_samples_per_second();
    throughput.frames_per_second = calculate_frames_per_second();

    // Real-time factor
    throughput.real_time_factor = calculate_real_time_factor();

    // I/O throughput
    throughput.io_bytes_per_second = calculate_io_throughput();
    throughput.io_operations_per_second = calculate_io_operations_per_second();

    // CPU instruction throughput
    throughput.cpu_instructions_per_second = calculate_cpu_instruction_throughput();
    throughput.cpu_cache_hit_rate_percent = calculate_cpu_cache_hit_rate();

    // GPU throughput
    throughput.gpu_flops_per_second = calculate_gpu_flops_throughput();
    throughput.gpu_memory_bandwidth_mbps = calculate_gpu_memory_bandwidth();

    // Network throughput
    throughput.network_bytes_per_second = calculate_network_throughput();
    throughput.network_packets_per_second = calculate_network_packet_rate();

    // Storage throughput
    throughput.storage_read_mbps = calculate_storage_read_throughput();
    throughput.storage_write_mbps = calculate_storage_write_throughput();
}

void ProcessingMetricsCollector::collect_resource_metrics(ProcessingMetrics::ResourceMetrics& resources) {
    // CPU utilization
    resources.cpu_utilization_percent = calculate_cpu_utilization();
    resources.cpu_cores_active = calculate_active_cpu_cores();
    resources.cpu_frequency_hz = calculate_cpu_frequency();

    // Memory utilization
    resources.memory_used_bytes = calculate_memory_usage();
    resources.memory_total_bytes = get_total_system_memory();
    resources.memory_utilization_percent = resources.memory_total_bytes > 0 ?
        (static_cast<double>(resources.memory_used_bytes) / resources.memory_total_bytes) * 100.0 : 0.0;

    // GPU utilization
    resources.gpu_utilization_percent = calculate_gpu_utilization();
    resources.gpu_memory_used_bytes = calculate_gpu_memory_usage();
    resources.gpu_memory_total_bytes = get_total_gpu_memory();
    resources.gpu_memory_utilization_percent = resources.gpu_memory_total_bytes > 0 ?
        (static_cast<double>(resources.gpu_memory_used_bytes) / resources.gpu_memory_total_bytes) * 100.0 : 0.0;
    resources.gpu_temperature_celsius = calculate_gpu_temperature();

    // Thread utilization
    resources.active_threads = get_active_thread_count();
    resources.total_threads = get_total_thread_count();
    resources.thread_utilization_percent = resources.total_threads > 0 ?
        (static_cast<double>(resources.active_threads) / resources.total_threads) * 100.0 : 0.0;

    // Handle/utilization counts
    resources.handle_count = get_handle_count();
    resources.file_descriptor_count = get_file_descriptor_count();
}

void ProcessingMetricsCollector::collect_pipeline_metrics(ProcessingMetrics::PipelineMetrics& pipeline) {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);

    // Stage performance
    pipeline.active_stages = 0;
    pipeline.total_stages = pipeline_stages_.size();
    pipeline.pipeline_efficiency_percent = 0.0;

    double total_processing_time = 0.0;
    size_t active_count = 0;

    for (const auto& [name, stage] : pipeline_stages_) {
        if (stage.is_active) {
            pipeline.active_stages++;
            active_count++;
        }
        total_processing_time += stage.average_processing_time_us;
    }

    if (active_count > 0) {
        pipeline.average_stage_latency_us = total_processing_time / active_count;
    }

    // Calculate efficiency based on successful vs total calls
    uint64_t total_calls = 0, successful_calls = 0;
    for (const auto& [name, stage] : pipeline_stages_) {
        total_calls += stage.calls_count;
        successful_calls += stage.successful_calls;
    }

    if (total_calls > 0) {
        pipeline.pipeline_efficiency_percent = (static_cast<double>(successful_calls) / total_calls) * 100.0;
    }

    // Bottleneck analysis
    pipeline.bottleneck_stage = identify_bottleneck_stage();
    pipeline.max_stage_queue_size = calculate_max_queue_size();
    pipeline.average_queue_size = calculate_average_queue_size();

    // Real-time metrics
    pipeline.real_time_pipeline_score = calculate_real_time_pipeline_score();
    pipeline.deadline_miss_rate_percent = calculate_deadline_miss_rate();
    pipeline.pipeline_throughput_percent = calculate_pipeline_throughput();

    // Quality metrics
    pipeline.quality_score = calculate_pipeline_quality_score();
    pipeline.stability_index = calculate_pipeline_stability_index();
}

void ProcessingMetricsCollector::collect_qos_metrics(ProcessingMetrics::QoSMetrics& qos) {
    // Real-time performance score
    qos.real_time_score = calculate_real_time_performance_score();

    // Latency metrics
    qos.audio_latency_ms = calculate_audio_latency();
    qos.max_audio_latency_ms = calculate_max_audio_latency();
    qos.latency_jitter_ms = calculate_latency_jitter();

    // Deadline compliance
    qos.deadline_miss_rate_percent = calculate_deadline_miss_rate();
    qos.max_consecutive_deadline_misses = calculate_max_consecutive_deadline_misses();

    // Glitch detection
    qos.audio_glitches_per_minute = calculate_audio_glitch_rate();
    qos.max_silent_period_ms = calculate_max_silent_period();

    // Quality of Experience
    qos.quality_of_experience_score = calculate_qoe_score();
    qos.user_satisfaction_rating = calculate_user_satisfaction_rating();

    // Service level metrics
    qos.service_level_agreement_compliance_percent = calculate_sla_compliance();
    qos.availability_percentage = calculate_availability_percentage();
}

bool ProcessingMetricsCollector::validate_config(const ProcessingMetricsConfig& config) const {
    if (config.collection_interval_ms == 0 || config.analysis_interval_ms == 0) {
        Logger::error("Invalid interval configuration");
        return false;
    }

    if (config.history_size == 0 || config.max_throughput_samples == 0) {
        Logger::error("Invalid buffer size configuration");
        return false;
    }

    if (config.performance_window_ms == 0) {
        Logger::error("Invalid performance window configuration");
        return false;
    }

    // Validate thresholds
    const auto& thresholds = config.thresholds;
    if (thresholds.max_processing_latency_ms <= 0 ||
        thresholds.max_memory_utilization_percent <= 0 ||
        thresholds.max_cpu_utilization_percent <= 0 ||
        thresholds.max_gpu_utilization_percent <= 0 ||
        thresholds.min_real_time_score < 0 || thresholds.min_real_time_score > 100) {
        Logger::error("Invalid threshold configuration");
        return false;
    }

    return true;
}

uint64_t ProcessingMetricsCollector::get_current_timestamp_microseconds() const {
    auto now = std::chrono::steady_clock::now();
    auto epoch = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(epoch).count();
}

void ProcessingMetricsCollector::update_performance_stats(double collection_time_ms, bool success) {
    std::lock_guard<std::mutex> lock(performance_mutex_);

    performance_stats_.total_collections++;
    if (success) {
        performance_stats_.successful_collections++;
    }

    // Update timing statistics
    performance_stats_.avg_collection_time_ms =
        ((performance_stats_.avg_collection_time_ms * (performance_stats_.total_collections - 1)) +
         collection_time_ms) / performance_stats_.total_collections;

    performance_stats_.max_collection_time_ms =
        std::max(performance_stats_.max_collection_time_ms, collection_time_ms);
    performance_stats_.min_collection_time_ms =
        std::min(performance_stats_.min_collection_time_ms, collection_time_ms);
}

void ProcessingMetricsCollector::trigger_metrics_callbacks(const ProcessingMetrics& metrics) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);

    if (metrics_callback_) {
        try {
            metrics_callback_(metrics);
        } catch (const std::exception& e) {
            Logger::error("Metrics callback error: {}", e.what());
        }
    }

    // Check thresholds and trigger alerts
    check_and_fire_alerts(metrics);
}

void ProcessingMetricsCollector::check_and_fire_alerts(const ProcessingMetrics& metrics) {
    if (!alert_callback_) {
        return;
    }

    const auto& thresholds = config_.thresholds;

    // Check processing latency
    if (metrics.timing.frame_processing_time_ms > thresholds.max_processing_latency_ms) {
        alert_callback_(metrics, "High processing latency detected", "latency");
    }

    // Check memory utilization
    if (metrics.resources.memory_utilization_percent > thresholds.max_memory_utilization_percent) {
        alert_callback_(metrics, "High memory utilization detected", "memory");
    }

    // Check CPU utilization
    if (metrics.resources.cpu_utilization_percent > thresholds.max_cpu_utilization_percent) {
        alert_callback_(metrics, "High CPU utilization detected", "cpu");
    }

    // Check GPU utilization
    if (metrics.resources.gpu_utilization_percent > thresholds.max_gpu_utilization_percent) {
        alert_callback_(metrics, "High GPU utilization detected", "gpu");
    }

    // Check real-time score
    if (metrics.qos.real_time_score < thresholds.min_real_time_score) {
        alert_callback_(metrics, "Low real-time performance score", "realtime");
    }
}

void ProcessingMetricsCollector::stream_processing_metrics(const ProcessingMetrics& metrics) {
    if (!streamer_) {
        return;
    }

    try {
        // Serialize metrics to binary format
        auto serialized = serialize_processing_metrics(metrics);

        // Stream through real-time streamer
        streamer_->send_binary_data(serialized.data(), serialized.size(), "processing_metrics");

        // Update streaming statistics
        {
            std::lock_guard<std::mutex> lock(performance_mutex_);
            performance_stats_.streamed_messages++;
        }
    }
    catch (const std::exception& e) {
        Logger::error("Failed to stream processing metrics: {}", e.what());

        std::lock_guard<std::mutex> lock(performance_mutex_);
        performance_stats_.streaming_errors++;
    }
}

std::vector<uint8_t> ProcessingMetricsCollector::serialize_processing_metrics(const ProcessingMetrics& metrics) const {
    std::vector<uint8_t> data;

    // Simple serialization - in real implementation would use more efficient format
    data.resize(sizeof(ProcessingMetrics));
    std::memcpy(data.data(), &metrics, sizeof(ProcessingMetrics));

    return data;
}

void ProcessingMetricsCollector::perform_adaptive_analysis() {
    // Adaptive analysis implementation
    // Adjust collection intervals based on system load and performance requirements

    auto latest = get_latest_metrics();
    if (!latest.is_valid) {
        return;
    }

    // Increase frequency during high load
    if (latest.resources.cpu_utilization_percent > 80.0 ||
        latest.resources.gpu_utilization_percent > 80.0) {
        // Could reduce collection interval for more frequent monitoring
    }

    // Decrease frequency during low load for efficiency
    if (latest.resources.cpu_utilization_percent < 30.0 &&
        latest.resources.gpu_utilization_percent < 30.0) {
        // Could increase collection interval to reduce overhead
    }
}

void ProcessingMetricsCollector::detect_anomalies() {
    // Anomaly detection implementation
    // Look for unusual patterns in metrics that might indicate problems

    auto history = get_metrics_history(100); // Last 100 samples
    if (history.size() < 10) {
        return;
    }

    // Simple statistical anomaly detection
    // In real implementation would use more sophisticated algorithms

    for (const auto& metrics : history) {
        // Check for unusual processing times
        if (metrics.timing.frame_processing_time_ms > 100.0) { // 100ms threshold
            Logger::warn("Anomaly detected: High processing time {}ms",
                        metrics.timing.frame_processing_time_ms);
        }

        // Check for memory leaks
        if (metrics.resources.memory_utilization_percent > 95.0) {
            Logger::warn("Anomaly detected: Very high memory utilization {}%",
                        metrics.resources.memory_utilization_percent);
        }
    }
}

void ProcessingMetricsCollector::predict_performance_trends() {
    // Performance prediction implementation
    // Analyze trends to predict future performance issues

    auto history = get_metrics_history(1000); // Last 1000 samples
    if (history.size() < 100) {
        return;
    }

    // Simple linear trend analysis
    // In real implementation would use more sophisticated time series analysis

    // Predict memory usage trend
    std::vector<double> memory_usage;
    for (const auto& metrics : history) {
        memory_usage.push_back(metrics.resources.memory_utilization_percent);
    }

    // Calculate trend (simplified)
    double trend = calculate_trend(memory_usage);
    if (trend > 0.1) { // Increasing trend
        Logger::info("Memory usage trend: increasing (+{:.2f}%)", trend * 100);
    } else if (trend < -0.1) { // Decreasing trend
        Logger::info("Memory usage trend: decreasing ({:.2f}%)", trend * 100);
    }
}

// Helper method implementations (simplified for brevity)

double ProcessingMetricsCollector::calculate_average_frame_processing_time() const {
    // In real implementation, would calculate from actual timing data
    return 5.0; // 5ms average
}

double ProcessingMetricsCollector::calculate_max_frame_processing_time() const {
    return 15.0; // 15ms max
}

double ProcessingMetricsCollector::calculate_min_frame_processing_time() const {
    return 2.0; // 2ms min
}

double ProcessingMetricsCollector::calculate_pipeline_latency() const {
    return 8.0; // 8ms pipeline latency
}

double ProcessingMetricsCollector::calculate_samples_per_second() const {
    if (current_audio_sample_rate_ > 0 && current_audio_channels_ > 0) {
        return current_audio_sample_rate_ * current_audio_channels_;
    }
    return 0.0;
}

double ProcessingMetricsCollector::calculate_frames_per_second() const {
    if (current_audio_sample_rate_ > 0 && current_audio_frame_size_ > 0) {
        return static_cast<double>(current_audio_sample_rate_) / current_audio_frame_size_;
    }
    return 0.0;
}

double ProcessingMetricsCollector::calculate_real_time_factor() const {
    double processing_time = calculate_average_frame_processing_time();
    double real_time_limit = (static_cast<double>(current_audio_frame_size_) / current_audio_sample_rate_) * 1000.0;

    return real_time_limit > 0 ? processing_time / real_time_limit : 0.0;
}

double ProcessingMetricsCollector::calculate_cpu_utilization() const {
    // In real implementation, would use platform-specific APIs
    return 45.0; // 45% CPU utilization
}

uint64_t ProcessingMetricsCollector::calculate_memory_usage() const {
    // In real implementation, would use platform-specific APIs
    return 1024 * 1024 * 1024; // 1GB
}

uint64_t ProcessingMetricsCollector::get_total_system_memory() const {
    // In real implementation, would use platform-specific APIs
    return 16ULL * 1024 * 1024 * 1024; // 16GB total
}

double ProcessingMetricsCollector::calculate_gpu_utilization() const {
    // In real implementation, would use GPU monitoring APIs
    return 65.0; // 65% GPU utilization
}

uint64_t ProcessingMetricsCollector::calculate_gpu_memory_usage() const {
    // In real implementation, would use GPU monitoring APIs
    return 4ULL * 1024 * 1024 * 1024; // 4GB GPU memory
}

uint64_t ProcessingMetricsCollector::get_total_gpu_memory() const {
    // In real implementation, would use GPU monitoring APIs
    return 8ULL * 1024 * 1024 * 1024; // 8GB total GPU memory
}

double ProcessingMetricsCollector::calculate_gpu_temperature() const {
    // In real implementation, would use GPU monitoring APIs
    return 72.0; // 72°C GPU temperature
}

std::string ProcessingMetricsCollector::identify_bottleneck_stage() const {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);

    std::string bottleneck = "none";
    double max_time = 0.0;

    for (const auto& [name, stage] : pipeline_stages_) {
        if (stage.average_processing_time_us > max_time) {
            max_time = stage.average_processing_time_us;
            bottleneck = name;
        }
    }

    return bottleneck;
}

double ProcessingMetricsCollector::calculate_real_time_performance_score() const {
    auto latest = get_latest_metrics();
    if (!latest.is_valid) {
        return 0.0;
    }

    // Calculate real-time score based on multiple factors
    double latency_score = std::max(0.0, 100.0 - latest.timing.frame_processing_time_ms);
    double deadline_score = 100.0 - latest.qos.deadline_miss_rate_percent;
    double glitch_score = std::max(0.0, 100.0 - (latest.qos.audio_glitches_per_minute * 10));

    return (latency_score + deadline_score + glitch_score) / 3.0;
}

double ProcessingMetricsCollector::calculate_audio_latency() const {
    // In real implementation, would calculate from actual audio pipeline measurements
    return 12.0; // 12ms audio latency
}

double ProcessingMetricsCollector::calculate_deadline_miss_rate() const {
    // In real implementation, would calculate from actual deadline tracking
    return 0.5; // 0.5% deadline miss rate
}

double ProcessingMetricsCollector::calculate_audio_glitch_rate() const {
    // In real implementation, would calculate from actual glitch detection
    return 0.1; // 0.1 glitches per minute
}

double ProcessingMetricsCollector::calculate_trend(const std::vector<double>& values) const {
    if (values.size() < 2) {
        return 0.0;
    }

    // Simple linear trend calculation
    size_t n = values.size();
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;

    for (size_t i = 0; i < n; ++i) {
        sum_x += i;
        sum_y += values[i];
        sum_xy += i * values[i];
        sum_x2 += i * i;
    }

    double denominator = n * sum_x2 - sum_x * sum_x;
    if (denominator == 0) {
        return 0.0;
    }

    double slope = (n * sum_xy - sum_x * sum_y) / denominator;

    // Normalize by average value to get relative trend
    double avg_y = sum_y / n;
    return avg_y > 0 ? slope / avg_y : 0.0;
}

void ProcessingMetricsCollector::generate_final_performance_report() const {
    auto report = generate_performance_report(std::chrono::minutes(5));

    Logger::info("=== Processing Metrics Performance Report ===");
    Logger::info("Total collections: {}", report.total_collections);
    Logger::info("Successful collections: {}", report.successful_collections);
    Logger::info("Collection success rate: {:.2f}%", report.collection_success_rate * 100);
    Logger::info("Average collection time: {:.2f}ms", report.avg_collection_time_ms);
    Logger::info("Maximum collection time: {:.2f}ms", report.max_collection_time_ms);
    Logger::info("Average processing latency: {:.2f}ms", report.avg_processing_latency_ms);
    Logger::info("Average throughput: {:.0f} samples/sec", report.avg_throughput_samples_per_sec);
    Logger::info("Average CPU utilization: {:.1f}%", report.avg_cpu_utilization_percent);
    Logger::info("Average GPU utilization: {:.1f}%", report.avg_gpu_utilization_percent);
    Logger::info("Average real-time score: {:.1f}", report.avg_real_time_score);
    Logger::info("Average pipeline efficiency: {:.1f}%", report.avg_pipeline_efficiency_percent);

    if (streamer_) {
        Logger::info("Streamed messages: {}", report.streamed_messages);
        Logger::info("Streaming errors: {}", report.streaming_errors);
    }

    Logger::info("=== End Processing Metrics Report ===");
}

// Factory implementations
std::unique_ptr<ProcessingMetricsCollector> ProcessingMetricsCollectorFactory::create_default() {
    auto config = ProcessingMetricsConfig::create_default();
    auto collector = std::make_unique<ProcessingMetricsCollector>();

    if (!collector->initialize(config)) {
        return nullptr;
    }

    return collector;
}

std::unique_ptr<ProcessingMetricsCollector> ProcessingMetricsCollectorFactory::create_high_performance() {
    auto config = ProcessingMetricsConfig::create_high_performance();
    auto collector = std::make_unique<ProcessingMetricsCollector>();

    if (!collector->initialize(config)) {
        return nullptr;
    }

    return collector;
}

std::unique_ptr<ProcessingMetricsCollector> ProcessingMetricsCollectorFactory::create_low_overhead() {
    auto config = ProcessingMetricsConfig::create_low_overhead();
    auto collector = std::make_unique<ProcessingMetricsCollector>();

    if (!collector->initialize(config)) {
        return nullptr;
    }

    return collector;
}

std::unique_ptr<ProcessingMetricsCollector> ProcessingMetricsCollectorFactory::create_comprehensive() {
    auto config = ProcessingMetricsConfig::create_comprehensive();
    auto collector = std::make_unique<ProcessingMetricsCollector>();

    if (!collector->initialize(config)) {
        return nullptr;
    }

    return collector;
}

} // namespace vortex::core::processing