#include "hardware/hardware_monitor.hpp"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>

#ifdef VORTEX_ENABLE_NVML
#include <nvml.h>
#endif

#ifdef _WIN32
#include <intrin.h>
#pragma comment(lib, "pdh.lib")
#pragma comment(lib, "psapi.lib")
#else
#include <fstream>
#include <sstream>
#endif

namespace vortex::hardware {

HardwareMonitor::HardwareMonitor() {
    Logger::info("HardwareMonitor: Creating instance");
    performance_stats_.start_time = std::chrono::steady_clock::now();
}

HardwareMonitor::~HardwareMonitor() {
    shutdown();
}

bool HardwareMonitor::initialize(const MonitorConfig& config) {
    if (initialized_.load()) {
        Logger::warn("HardwareMonitor already initialized");
        return true;
    }

    config_ = config;

    Logger::info("HardwareMonitor: Initializing with CPU interval: {}ms, GPU interval: {}ms",
                 config_.cpu_interval_ms, config_.gpu_interval_ms);

    try {
        // Initialize platform-specific monitors
        if (!initialize_platform_monitors()) {
            Logger::error("HardwareMonitor: Failed to initialize platform monitors");
            return false;
        }

        // Initialize GPU monitoring
        if (config_.enable_nvidia_ml || config_.enable_amd_smi || config_.enable_intel_gpu) {
            if (!initialize_gpu_monitoring()) {
                Logger::warn("HardwareMonitor: GPU monitoring initialization failed, continuing without GPU monitoring");
            }
        }

        // Reserve history buffer
        metrics_history_.reserve(config_.cache_size);

        // Reserve cache
        metrics_cache_.reserve(config_.cache_size);

        initialized_.store(true);
        Logger::info("HardwareMonitor initialized successfully");
        return true;

    } catch (const std::exception& e) {
        Logger::error("HardwareMonitor: Exception during initialization: {}", e.what());
        return false;
    }
}

void HardwareMonitor::shutdown() {
    if (!initialized_.load()) {
        return;
    }

    Logger::info("HardwareMonitor: Shutting down");

    // Stop monitoring
    stop_monitoring();

    // Signal shutdown
    shutdown_requested_.store(true);
    metrics_cv_.notify_all();

    // Wait for all threads to finish
    if (cpu_thread_.joinable()) cpu_thread_.join();
    if (gpu_thread_.joinable()) gpu_thread_.join();
    if (memory_thread_.joinable()) memory_thread_.join();
    if (audio_thread_.joinable()) audio_thread_.join();
    if (system_thread_.joinable()) system_thread_.join();
    if (performance_thread_.joinable()) performance_thread_.join();
    if (streaming_thread_.joinable()) streaming_thread_.join();

    // Cleanup platform-specific resources
    cleanup_platform_monitors();
    cleanup_gpu_monitoring();

    // Clear data structures
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        std::queue<HardwareMetrics> empty;
        metrics_queue_.swap(empty);
        metrics_history_.clear();
    }

    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        metrics_cache_.clear();
    }

    initialized_.store(false);
    shutdown_requested_.store(false);

    // Log final performance statistics
    if (config_.enable_performance_profiling) {
        Logger::info("HardwareMonitor final stats: collections={}, success_rate={:.1f}%, avg_time={:.2f}ms",
                     performance_stats_.total_collections,
                     performance_stats_.total_collections > 0 ?
                         (performance_stats_.successful_collections * 100.0 / performance_stats_.total_collections) : 0.0,
                     performance_stats_.avg_collection_time_ms);
    }

    Logger::info("HardwareMonitor shutdown complete");
}

bool HardwareMonitor::start_monitoring() {
    if (!initialized_.load()) {
        Logger::error("HardwareMonitor: Cannot start monitoring - not initialized");
        return false;
    }

    if (monitoring_.load()) {
        Logger::warn("HardwareMonitor: Already monitoring");
        return true;
    }

    Logger::info("HardwareMonitor: Starting hardware monitoring");

    try {
        // Reset shutdown flag
        shutdown_requested_.store(false);
        paused_.store(false);

        // Start collection threads
        cpu_thread_ = std::thread(&HardwareMonitor::cpu_collection_thread, this);
        gpu_thread_ = std::thread(&HardwareMonitor::gpu_collection_thread, this);
        memory_thread_ = std::thread(&HardwareMonitor::memory_collection_thread, this);
        audio_thread_ = std::thread(&HardwareMonitor::audio_collection_thread, this);
        system_thread_ = std::thread(&HardwareMonitor::system_collection_thread, this);
        performance_thread_ = std::thread(&HardwareMonitor::performance_collection_thread, this);

        // Start streaming thread if enabled
        if (realtime_streaming_enabled_.load()) {
            streaming_thread_ = std::thread(&HardwareMonitor::streaming_thread, this);
        }

        // Set thread priorities for real-time performance
        if (config_.enable_real_time_priority_monitoring) {
            set_thread_priority(cpu_thread_, 2);
            set_thread_priority(gpu_thread_, 2);
            set_thread_priority(audio_thread_, 3); // Highest priority for audio
        }

        monitoring_.store(true);
        Logger::info("HardwareMonitor: Hardware monitoring started successfully");
        return true;

    } catch (const std::exception& e) {
        Logger::error("HardwareMonitor: Exception starting monitoring: {}", e.what());
        return false;
    }
}

void HardwareMonitor::stop_monitoring() {
    if (!monitoring_.load()) {
        return;
    }

    Logger::info("HardwareMonitor: Stopping hardware monitoring");

    monitoring_.store(false);
    shutdown_requested_.store(true);
    metrics_cv_.notify_all();

    // Wait for threads to finish
    if (cpu_thread_.joinable()) cpu_thread_.join();
    if (gpu_thread_.joinable()) gpu_thread_.join();
    if (memory_thread_.joinable()) memory_thread_.join();
    if (audio_thread_.joinable()) audio_thread_.join();
    if (system_thread_.joinable()) system_thread_.join();
    if (performance_thread_.joinable()) performance_thread_.join();
    if (streaming_thread_.joinable()) streaming_thread_.join();

    Logger::info("HardwareMonitor: Hardware monitoring stopped");
}

HardwareMetrics HardwareMonitor::collect_all_metrics() {
    HardwareMetrics metrics;
    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        // Collect all metric categories
        collect_cpu_metrics();
        collect_gpu_metrics();
        collect_memory_metrics();
        collect_audio_metrics();
        collect_system_metrics();
        collect_performance_metrics();

        // Get the latest metrics
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            metrics = latest_metrics_;
        }

        metrics.is_valid = true;

    } catch (const std::exception& e) {
        Logger::error("HardwareMonitor: Exception collecting metrics: {}", e.what());
        metrics.is_valid = false;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    metrics.collection_duration_ms = duration.count() / 1000.0;
    metrics.timestamp_microseconds = get_current_timestamp_microseconds();
    metrics.collection_time = std::chrono::steady_clock::now();

    // Update performance statistics
    update_performance_stats(metrics.collection_duration_ms, metrics.is_valid);

    return metrics;
}

HardwareMetrics HardwareMonitor::collect_cpu_metrics() {
    HardwareMetrics metrics;

    try {
        metrics.cpu_utilization_percent = get_cpu_utilization();
        metrics.cpu_temperature_celsius = get_cpu_temperature();
        metrics.cpu_frequency_hz = get_cpu_frequency();
        get_cpu_load_averages(metrics.cpu_load_average_1min,
                             metrics.cpu_load_average_5min,
                             metrics.cpu_load_average_15min);

#ifdef _WIN32
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        metrics.cpu_cores_total = sysInfo.dwNumberOfProcessors;
        metrics.cpu_cores_active = metrics.cpu_cores_total; // Simplified
#else
        // Linux implementation would read from /proc/cpuinfo and /proc/stat
        metrics.cpu_cores_total = std::thread::hardware_concurrency();
        metrics.cpu_cores_active = metrics.cpu_cores_total;
#endif

        metrics.is_valid = true;

    } catch (const std::exception& e) {
        Logger::error("HardwareMonitor: Exception collecting CPU metrics: {}", e.what());
        metrics.is_valid = false;
    }

    // Update latest metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        latest_metrics_.cpu_utilization_percent = metrics.cpu_utilization_percent;
        latest_metrics_.cpu_temperature_celsius = metrics.cpu_temperature_celsius;
        latest_metrics_.cpu_frequency_hz = metrics.cpu_frequency_hz;
        latest_metrics_.cpu_cores_active = metrics.cpu_cores_active;
        latest_metrics_.cpu_cores_total = metrics.cpu_cores_total;
        latest_metrics_.cpu_load_average_1min = metrics.cpu_load_average_1min;
        latest_metrics_.cpu_load_average_5min = metrics.cpu_load_average_5min;
        latest_metrics_.cpu_load_average_15min = metrics.cpu_load_average_15min;
    }

    return metrics;
}

HardwareMetrics HardwareMonitor::collect_gpu_metrics() {
    HardwareMetrics metrics;

    if (!config_.enable_nvidia_ml && !config_.enable_amd_smi && !config_.enable_intel_gpu) {
        return metrics; // GPU monitoring disabled
    }

    try {
        std::vector<HardwareMetrics::GPUMetrics> gpu_metrics;

        // Collect NVIDIA GPU metrics
        if (config_.enable_nvidia_ml) {
            collect_nvidia_gpu_metrics(gpu_metrics);
        }

        // Collect AMD GPU metrics
        if (config_.enable_amd_smi) {
            collect_amd_gpu_metrics(gpu_metrics);
        }

        // Collect Intel GPU metrics
        if (config_.enable_intel_gpu) {
            collect_intel_gpu_metrics(gpu_metrics);
        }

        // Collect OpenCL GPU metrics
        if (config_.enable_opencl_monitoring) {
            collect_opencl_gpu_metrics(gpu_metrics);
        }

        metrics.gpu_metrics = gpu_metrics;
        metrics.is_valid = !gpu_metrics.empty();

    } catch (const std::exception& e) {
        Logger::error("HardwareMonitor: Exception collecting GPU metrics: {}", e.what());
        metrics.is_valid = false;
    }

    // Update latest metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        latest_metrics_.gpu_metrics = metrics.gpu_metrics;
    }

    return metrics;
}

HardwareMetrics HardwareMonitor::collect_memory_metrics() {
    HardwareMetrics metrics;

    try {
        get_memory_usage(metrics.memory_total_bytes,
                       metrics.memory_used_bytes,
                       metrics.memory_available_bytes);

        if (metrics.memory_total_bytes > 0) {
            metrics.memory_utilization_percent =
                (metrics.memory_used_bytes * 100.0) / metrics.memory_total_bytes;
        }

        get_swap_usage(metrics.memory_swap_total_bytes, metrics.memory_swap_used_bytes);

        metrics.is_valid = true;

    } catch (const std::exception& e) {
        Logger::error("HardwareMonitor: Exception collecting memory metrics: {}", e.what());
        metrics.is_valid = false;
    }

    // Update latest metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        latest_metrics_.memory_total_bytes = metrics.memory_total_bytes;
        latest_metrics_.memory_used_bytes = metrics.memory_used_bytes;
        latest_metrics_.memory_available_bytes = metrics.memory_available_bytes;
        latest_metrics_.memory_utilization_percent = metrics.memory_utilization_percent;
        latest_metrics_.memory_swap_total_bytes = metrics.memory_swap_total_bytes;
        latest_metrics_.memory_swap_used_bytes = metrics.memory_swap_used_bytes;
    }

    return metrics;
}

HardwareMetrics HardwareMonitor::collect_audio_metrics() {
    HardwareMetrics metrics;

    if (!config_.enable_dsp_monitoring) {
        return metrics;
    }

    try {
        HardwareMetrics::AudioMetrics& audio_metrics = metrics.audio_metrics;

        collect_dsp_metrics(audio_metrics);

        if (config_.enable_buffer_monitoring) {
            collect_buffer_metrics(audio_metrics);
        }

        if (config_.enable_latency_monitoring) {
            collect_latency_metrics(audio_metrics);
        }

        metrics.is_valid = true;

    } catch (const std::exception& e) {
        Logger::error("HardwareMonitor: Exception collecting audio metrics: {}", e.what());
        metrics.is_valid = false;
    }

    // Update latest metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        latest_metrics_.audio_metrics = metrics.audio_metrics;
    }

    return metrics;
}

HardwareMetrics HardwareMonitor::collect_system_metrics() {
    HardwareMetrics metrics;

    try {
        collect_system_info(metrics.system_metrics);

        if (config_.enable_disk_monitoring) {
            collect_disk_metrics(metrics.system_metrics);
        }

        if (config_.enable_network_monitoring) {
            collect_network_metrics(metrics.system_metrics);
        }

        if (config_.enable_thermal_monitoring) {
            collect_thermal_metrics(metrics);
        }

        metrics.is_valid = true;

    } catch (const std::exception& e) {
        Logger::error("HardwareMonitor: Exception collecting system metrics: {}", e.what());
        metrics.is_valid = false;
    }

    // Update latest metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        latest_metrics_.system_metrics = metrics.system_metrics;
    }

    return metrics;
}

HardwareMetrics HardwareMonitor::collect_performance_metrics() {
    HardwareMetrics metrics;

    try {
        calculate_performance_metrics(metrics);
        metrics.is_valid = true;

    } catch (const std::exception& e) {
        Logger::error("HardwareMonitor: Exception collecting performance metrics: {}", e.what());
        metrics.is_valid = false;
    }

    // Update latest metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        latest_metrics_.performance_metrics = metrics.performance_metrics;
    }

    return metrics;
}

HardwareMetrics HardwareMonitor::get_latest_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return latest_metrics_;
}

std::vector<HardwareMetrics> HardwareMonitor::get_metrics_history(size_t count) const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    if (metrics_history_.empty()) {
        return {};
    }

    size_t start_index = (count >= metrics_history_.size()) ? 0 :
                        (metrics_history_.size() - count);

    return std::vector<HardwareMetrics>(metrics_history_.begin() + start_index,
                                        metrics_history_.end());
}

void HardwareMonitor::set_streaming_interface(std::shared_ptr<network::RealtimeStreamer> streamer) {
    streamer_ = streamer;
    Logger::info("HardwareMonitor: Streaming interface {}", streamer ? "set" : "cleared");
}

bool HardwareMonitor::enable_realtime_streaming(bool enabled) {
    realtime_streaming_enabled_.store(enabled);
    Logger::info("HardwareMonitor: Real-time streaming {}", enabled ? "enabled" : "disabled");
    return true;
}

void HardwareMonitor::update_config(const MonitorConfig& config) {
    config_ = config;
    Logger::info("HardwareMonitor: Configuration updated");
}

HardwareMonitor::MonitorPerformance HardwareMonitor::get_performance_stats() const {
    std::lock_guard<std::mutex> lock(performance_mutex_);
    MonitorPerformance stats = performance_stats_;
    stats.cache_hit_rate = calculate_cache_hit_rate();
    return stats;
}

std::string HardwareMonitor::get_diagnostics_report() const {
    std::ostringstream report;

    report << "=== HardwareMonitor Diagnostics Report ===\n";
    report << "Initialized: " << (initialized_.load() ? "Yes" : "No") << "\n";
    report << "Monitoring: " << (monitoring_.load() ? "Yes" : "No") << "\n";
    report << "Real-time streaming: " << (realtime_streaming_enabled_.load() ? "Yes" : "No") << "\n\n";

    auto perf = get_performance_stats();
    report << "Performance Statistics:\n";
    report << "  Total collections: " << perf.total_collections << "\n";
    report << "  Successful collections: " << perf.successful_collections << "\n";
    report << "  Success rate: " << std::fixed << std::setprecision(1)
           << (perf.total_collections > 0 ?
               (perf.successful_collections * 100.0 / perf.total_collections) : 0.0) << "%\n";
    report << "  Average collection time: " << std::setprecision(2) << perf.avg_collection_time_ms << "ms\n";
    report << "  Max collection time: " << perf.max_collection_time_ms << "ms\n";
    report << "  Cache hit rate: " << std::setprecision(1) << perf.cache_hit_rate << "%\n";

    auto current_metrics = get_latest_metrics();
    if (current_metrics.is_valid) {
        report << "\nCurrent System Status:\n";
        report << "  CPU Utilization: " << std::setprecision(1) << current_metrics.cpu_utilization_percent << "%\n";
        report << "  CPU Temperature: " << std::setprecision(1) << current_metrics.cpu_temperature_celsius << "°C\n";
        report << "  Memory Utilization: " << std::setprecision(1) << current_metrics.memory_utilization_percent << "%\n";

        if (!current_metrics.gpu_metrics.empty()) {
            report << "  GPUs detected: " << current_metrics.gpu_metrics.size() << "\n";
            for (size_t i = 0; i < current_metrics.gpu_metrics.size(); ++i) {
                const auto& gpu = current_metrics.gpu_metrics[i];
                report << "    GPU " << gpu.device_id << " (" << gpu.device_name << "):\n";
                report << "      Utilization: " << std::setprecision(1) << gpu.gpu_utilization_percent << "%\n";
                report << "      Temperature: " << std::setprecision(1) << gpu.temperature_celsius << "°C\n";
                report << "      Memory: " << gpu.memory_used_bytes / (1024*1024) << "MB / "
                       << gpu.memory_total_bytes / (1024*1024) << "MB\n";
            }
        }

        report << "  Audio DSP Load: " << std::setprecision(1) << current_metrics.audio_metrics.dsp_load_percent << "%\n";
        report << "  Audio Latency: " << std::setprecision(2) << current_metrics.audio_metrics.processing_latency_ms << "ms\n";
    }

    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - perf.start_time).count();
    report << "  Monitor uptime: " << uptime << " seconds\n";

    return report.str();
}

void HardwareMonitor::set_metrics_callback(MetricsCallback callback) {
    // This would store the callback for use when metrics are collected
    Logger::debug("HardwareMonitor: Metrics callback set");
}

void HardwareMonitor::set_alert_callback(AlertCallback callback) {
    // This would store the callback for use when thresholds are violated
    Logger::debug("HardwareMonitor: Alert callback set");
}

// Private implementation methods

void HardwareMonitor::cpu_collection_thread() {
    Logger::debug("HardwareMonitor: CPU collection thread started");

    while (!shutdown_requested_.load()) {
        if (paused_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        collect_cpu_metrics();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::this_thread::sleep_for(std::chrono::milliseconds(config_.cpu_interval_ms) - duration);
    }

    Logger::debug("HardwareMonitor: CPU collection thread stopped");
}

void HardwareMonitor::gpu_collection_thread() {
    Logger::debug("HardwareMonitor: GPU collection thread started");

    while (!shutdown_requested_.load()) {
        if (paused_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        collect_gpu_metrics();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::this_thread::sleep_for(std::chrono::milliseconds(config_.gpu_interval_ms) - duration);
    }

    Logger::debug("HardwareMonitor: GPU collection thread stopped");
}

void HardwareMonitor::memory_collection_thread() {
    Logger::debug("HardwareMonitor: Memory collection thread started");

    while (!shutdown_requested_.load()) {
        if (paused_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        collect_memory_metrics();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::this_thread::sleep_for(std::chrono::milliseconds(config_.memory_interval_ms) - duration);
    }

    Logger::debug("HardwareMonitor: Memory collection thread stopped");
}

void HardwareMonitor::audio_collection_thread() {
    Logger::debug("HardwareMonitor: Audio collection thread started");

    while (!shutdown_requested_.load()) {
        if (paused_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        collect_audio_metrics();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::this_thread::sleep_for(std::chrono::milliseconds(config_.audio_interval_ms) - duration);
    }

    Logger::debug("HardwareMonitor: Audio collection thread stopped");
}

void HardwareMonitor::system_collection_thread() {
    Logger::debug("HardwareMonitor: System collection thread started");

    while (!shutdown_requested_.load()) {
        if (paused_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        collect_system_metrics();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::this_thread::sleep_for(std::chrono::milliseconds(config_.system_interval_ms) - duration);
    }

    Logger::debug("HardwareMonitor: System collection thread stopped");
}

void HardwareMonitor::performance_collection_thread() {
    Logger::debug("HardwareMonitor: Performance collection thread started");

    while (!shutdown_requested_.load()) {
        if (paused_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        collect_performance_metrics();

        // Adaptive monitoring
        if (config_.enable_batch_collection) {
            adjust_collection_intervals(latest_metrics_);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::this_thread::sleep_for(std::chrono::milliseconds(config_.performance_interval_ms) - duration);
    }

    Logger::debug("HardwareMonitor: Performance collection thread stopped");
}

void HardwareMonitor::streaming_thread() {
    Logger::debug("HardwareMonitor: Streaming thread started");

    while (!shutdown_requested_.load()) {
        if (paused_.load() || !realtime_streaming_enabled_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        auto current_metrics = get_latest_metrics();
        if (current_metrics.is_valid) {
            stream_metrics(current_metrics);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(config_.streaming_interval_ms));
    }

    Logger::debug("HardwareMonitor: Streaming thread stopped");
}

bool HardwareMonitor::initialize_platform_monitors() {
#ifdef _WIN32
    // Initialize Windows Performance Counters
    PDH_STATUS status = PdhOpenQuery(&cpu_query_, 0);
    if (status != ERROR_SUCCESS) {
        Logger::error("HardwareMonitor: Failed to open PDH query: {}", status);
        return false;
    }

    status = PdhAddCounter(cpu_query_, L"\\Processor(_Total)\\% Processor Time", 0, &cpu_counter_);
    if (status != ERROR_SUCCESS) {
        Logger::warn("HardwareMonitor: Failed to add CPU counter: {}", status);
    }

    status = PdhAddCounter(cpu_query_, L"\\Memory\\Available MBytes", 0, &memory_counter_);
    if (status != ERROR_SUCCESS) {
        Logger::warn("HardwareMonitor: Failed to add memory counter: {}", status);
    }

    process_handle_ = GetCurrentProcess();
#else
    // Linux initialization would read from /proc/cpuinfo, /proc/meminfo, etc.
    Logger::debug("HardwareMonitor: Linux platform monitoring initialized");
#endif

    return true;
}

void HardwareMonitor::cleanup_platform_monitors() {
#ifdef _WIN32
    if (cpu_query_) {
        PdhCloseQuery(cpu_query_);
        cpu_query_ = nullptr;
    }

    if (process_handle_) {
        CloseHandle(process_handle_);
        process_handle_ = nullptr;
    }
#endif
}

double HardwareMonitor::get_cpu_utilization() {
#ifdef _WIN32
    if (cpu_query_ && cpu_counter_) {
        PDH_FMT_COUNTERVALUE counterValue;
        PDH_STATUS status = PdhCollectQueryData(cpu_query_);
        if (status == ERROR_SUCCESS) {
            status = PdhGetFormattedCounterValue(cpu_counter_, PDH_FMT_DOUBLE, nullptr, &counterValue);
            if (status == ERROR_SUCCESS) {
                return counterValue.doubleValue;
            }
        }
    }

    // Fallback method
    static ULARGE_INTEGER lastCPU, lastSysCPU, lastUserCPU;
    static int numProcessors = 0;
    static HANDLE self = GetCurrentProcess();

    if (numProcessors == 0) {
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        numProcessors = sysInfo.dwNumberOfProcessors;
    }

    ULARGE_INTEGER now, sys, user;
    if (GetProcessTimes(self, nullptr, nullptr, &user, &kernel) != 0) { // kernel needs to be defined
        now.QuadPart = kernel.QuadPart + user.QuadPart;
        sys.QuadPart = kernel.QuadPart;
        user.QuadPart = user.QuadPart;
    }

    double percent = (now.QuadPart - lastCPU.QuadPart) * 100.0 /
                    (sys.QuadPart - lastSysCPU.QuadPart + user.QuadPart - lastUserCPU.QuadPart);
    lastCPU = now;
    lastSysCPU = sys;
    lastUserCPU = user;

    return percent / numProcessors;
#else
    // Linux implementation - read from /proc/stat
    std::ifstream file("/proc/stat");
    if (!file.is_open()) {
        return 0.0;
    }

    std::string line;
    if (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string cpu_label;
        long user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice;

        if (iss >> cpu_label >> user >> nice >> system >> idle >> iowait >> irq >>
            softirq >> steal >> guest >> guest_nice) {

            long idle_time = idle + iowait;
            long non_idle_time = user + nice + system + irq + softirq + steal;
            long total_time = idle_time + non_idle_time;

            // This is simplified - proper implementation would store previous values
            return (total_time > 0) ? (non_idle_time * 100.0 / total_time) : 0.0;
        }
    }

    return 0.0;
#endif
}

double HardwareMonitor::get_cpu_temperature() {
#ifdef _WIN32
    // Windows - use WMI or other APIs
    // For now, return a placeholder
    return 45.0; // Placeholder temperature
#else
    // Linux - read from /sys/class/thermal/ or other sources
    std::ifstream file("/sys/class/thermal/thermal_zone0/temp");
    if (file.is_open()) {
        int temp_millidegrees;
        file >> temp_millidegrees;
        return temp_millidegrees / 1000.0;
    }
    return 0.0;
#endif
}

uint64_t HardwareMonitor::get_cpu_frequency() {
#ifdef _WIN32
    LARGE_INTEGER frequency;
    if (QueryPerformanceFrequency(&frequency)) {
        return frequency.QuadPart;
    }
    return 0;
#else
    // Linux - read from /proc/cpuinfo
    std::ifstream file("/proc/cpuinfo");
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("cpu MHz") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                double mhz = std::stod(line.substr(pos + 1));
                return static_cast<uint64_t>(mhz * 1000000);
            }
        }
    }
    return 0;
#endif
}

void HardwareMonitor::get_cpu_load_averages(double& avg1, double& avg5, double& avg15) {
#ifdef _WIN32
    // Windows doesn't have direct load averages
    // Use approximation based on CPU utilization
    double cpu_usage = get_cpu_utilization();
    avg1 = cpu_usage;
    avg5 = cpu_usage;
    avg15 = cpu_usage;
#else
    // Linux - read from /proc/loadavg
    std::ifstream file("/proc/loadavg");
    if (file.is_open()) {
        file >> avg1 >> avg5 >> avg15;
    } else {
        avg1 = avg5 = avg15 = 0.0;
    }
#endif
}

void HardwareMonitor::get_memory_usage(uint64_t& total, uint64_t& used, uint64_t& available) {
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memInfo)) {
        total = memInfo.ullTotalPhys;
        available = memInfo.ullAvailPhys;
        used = total - available;
        return;
    }
#else
    // Linux - read from /proc/meminfo
    std::ifstream file("/proc/meminfo");
    std::string line;
    uint64_t mem_total = 0, mem_free = 0, mem_available = 0, mem_buffers = 0, mem_cached = 0;

    while (std::getline(file, line)) {
        if (line.find("MemTotal:") != std::string::npos) {
            sscanf(line.c_str(), "MemTotal: %lu kB", &mem_total);
        } else if (line.find("MemFree:") != std::string::npos) {
            sscanf(line.c_str(), "MemFree: %lu kB", &mem_free);
        } else if (line.find("MemAvailable:") != std::string::npos) {
            sscanf(line.c_str(), "MemAvailable: %lu kB", &mem_available);
        } else if (line.find("Buffers:") != std::string::npos) {
            sscanf(line.c_str(), "Buffers: %lu kB", &mem_buffers);
        } else if (line.find("Cached:") != std::string::npos) {
            sscanf(line.c_str(), "Cached: %lu kB", &mem_cached);
        }
    }

    total = mem_total * 1024; // Convert to bytes
    if (mem_available > 0) {
        available = mem_available * 1024;
    } else {
        available = (mem_free + mem_buffers + mem_cached) * 1024;
    }
    used = total - available;
    return;
#endif

    // Fallback
    total = used = available = 0;
}

void HardwareMonitor::get_swap_usage(uint64_t& total, uint64_t& used) {
#ifdef _WIN32
    // Windows swap usage
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memInfo)) {
        total = memInfo.ullTotalPageFile;
        used = memInfo.ullTotalPageFile - memInfo.ullAvailPageFile;
        return;
    }
#else
    // Linux swap usage - read from /proc/meminfo
    std::ifstream file("/proc/meminfo");
    std::string line;
    uint64_t swap_total = 0, swap_free = 0;

    while (std::getline(file, line)) {
        if (line.find("SwapTotal:") != std::string::npos) {
            sscanf(line.c_str(), "SwapTotal: %lu kB", &swap_total);
        } else if (line.find("SwapFree:") != std::string::npos) {
            sscanf(line.c_str(), "SwapFree: %lu kB", &swap_free);
        }
    }

    total = swap_total * 1024;
    used = (swap_total - swap_free) * 1024;
    return;
#endif

    // Fallback
    total = used = 0;
}

bool HardwareMonitor::initialize_gpu_monitoring() {
#ifdef VORTEX_ENABLE_NVML
    nvmlReturn_t result = nvmlInit();
    if (result == NVML_SUCCESS) {
        nvml_initialized_ = true;
        result = nvmlDeviceGetCount(&nvml_device_count_);
        if (result == NVML_SUCCESS) {
            Logger::info("HardwareMonitor: NVML initialized with {} GPUs", nvml_device_count_);
            return true;
        }
    }
    Logger::warn("HardwareMonitor: NVML initialization failed: {}", nvmlErrorString(result));
#endif

    // Would initialize AMD and Intel GPU monitoring here
    Logger::debug("HardwareMonitor: GPU monitoring initialized");
    return true;
}

void HardwareMonitor::cleanup_gpu_monitoring() {
#ifdef VORTEX_ENABLE_NVML
    if (nvml_initialized_) {
        nvmlShutdown();
        nvml_initialized_ = false;
    }
#endif
}

void HardwareMonitor::collect_nvidia_gpu_metrics(std::vector<HardwareMetrics::GPUMetrics>& metrics) {
#ifdef VORTEX_ENABLE_NVML
    if (!nvml_initialized_) {
        return;
    }

    for (unsigned int i = 0; i < nvml_device_count_; ++i) {
        nvmlDevice_t device;
        if (nvmlDeviceGetHandleByIndex(i, &device) != NVML_SUCCESS) {
            continue;
        }

        HardwareMetrics::GPUMetrics gpu_metric;
        gpu_metric.device_id = i;

        // Device name
        char name[NVML_DEVICE_NAME_V2_BUFFER_SIZE];
        if (nvmlDeviceGetName(device, name, sizeof(name)) == NVML_SUCCESS) {
            gpu_metric.device_name = name;
        }

        // Utilization
        nvmlUtilization_t utilization;
        if (nvmlDeviceGetUtilizationRates(device, &utilization) == NVML_SUCCESS) {
            gpu_metric.gpu_utilization_percent = utilization.gpu;
            gpu_metric.memory_utilization_percent = utilization.memory;
        }

        // Memory
        nvmlMemory_t memory;
        if (nvmlDeviceGetMemoryInfo(device, &memory) == NVML_SUCCESS) {
            gpu_metric.memory_total_bytes = memory.total;
            gpu_metric.memory_used_bytes = memory.used;
            gpu_metric.memory_free_bytes = memory.free;
        }

        // Temperature
        unsigned int temp;
        if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
            gpu_metric.temperature_celsius = static_cast<double>(temp);
        }

        // Power consumption
        unsigned int power;
        if (nvmlDeviceGetPowerUsage(device, &power) == NVML_SUCCESS) {
            gpu_metric.power_consumption_watts = power / 1000.0; // Convert mW to W
        }

        // Clock frequency
        unsigned int clock;
        if (nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &clock) == NVML_SUCCESS) {
            gpu_metric.clock_frequency_hz = clock * 1000000ULL; // Convert MHz to Hz
        }

        // Driver version
        char driver_version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
        if (nvmlSystemGetDriverVersion(driver_version, sizeof(driver_version)) == NVML_SUCCESS) {
            gpu_metric.driver_version = driver_version;
        }

        metrics.push_back(gpu_metric);
    }
#endif
}

void HardwareMonitor::collect_amd_gpu_metrics(std::vector<HardwareMetrics::GPUMetrics>& metrics) {
    // AMD GPU metrics implementation would use ADL SDK or other APIs
    // Placeholder implementation
    Logger::debug("HardwareMonitor: AMD GPU metrics collection not yet implemented");
}

void HardwareMonitor::collect_intel_gpu_metrics(std::vector<HardwareMetrics::GPUMetrics>& metrics) {
    // Intel GPU metrics implementation would use Intel GPU APIs
    // Placeholder implementation
    Logger::debug("HardwareMonitor: Intel GPU metrics collection not yet implemented");
}

void HardwareMonitor::collect_opencl_gpu_metrics(std::vector<HardwareMetrics::GPUMetrics>& metrics) {
    // OpenCL GPU metrics implementation
    // Placeholder implementation
    Logger::debug("HardwareMonitor: OpenCL GPU metrics collection not yet implemented");
}

void HardwareMonitor::collect_dsp_metrics(HardwareMetrics::AudioMetrics& metrics) {
    // Collect DSP load metrics
    // This would interface with the audio processing components
    metrics.dsp_load_percent = 0.0; // Placeholder

    // Collect processing statistics
    metrics.samples_processed = 0;
    metrics.samples_dropped = 0;
    metrics.throughput_samples_per_second = 0.0;
    metrics.cpu_cycles_per_sample = 0.0;

    // Collect GPU audio processing metrics
    metrics.gpu_kernels_executed = 0;
    metrics.gpu_processing_time_ms = 0.0;
}

void HardwareMonitor::collect_buffer_metrics(HardwareMetrics::AudioMetrics& metrics) {
    // Collect audio buffer metrics
    metrics.buffer_utilization_percent = 0.0; // Placeholder
    metrics.active_channels = 2; // Default stereo
    metrics.total_channels = 2;
    metrics.audio_memory_usage_bytes = 0;
}

void HardwareMonitor::collect_latency_metrics(HardwareMetrics::AudioMetrics& metrics) {
    // Collect audio processing latency
    metrics.processing_latency_ms = 0.0; // Placeholder
    metrics.real_time_priority_threads = 0;
}

void HardwareMonitor::collect_system_info(HardwareMetrics::SystemMetrics& metrics) {
    // System uptime and process information
    metrics.system_uptime_seconds = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

#ifdef _WIN32
    FILETIME creationTime, exitTime, kernelTime, userTime;
    if (GetProcessTimes(GetCurrentProcess(), &creationTime, &exitTime, &kernelTime, &userTime)) {
        ULARGE_INTEGER kernel, user;
        kernel.LowPart = kernelTime.dwLowDateTime;
        kernel.HighPart = kernelTime.dwHighDateTime;
        user.LowPart = userTime.dwLowDateTime;
        user.HighPart = userTime.dwHighDateTime;

        metrics.process_uptime_seconds = (kernel.QuadPart + user.QuadPart) / 10000000ULL;
    }

    metrics.thread_count = 0; // Would need additional API calls
    metrics.handle_count = 0; // Would need additional API calls
#else
    // Linux implementation
    std::ifstream stat_file("/proc/stat");
    if (stat_file.is_open()) {
        // Read system uptime from first line of /proc/stat
        std::string line;
        std::getline(stat_file, line);
        std::istringstream iss(line);
        std::string label;
        double uptime_seconds;
        if (iss >> label >> uptime_seconds) {
            metrics.system_uptime_seconds = static_cast<uint64_t>(uptime_seconds);
        }
    }

    // Process information from /proc/self/status
    std::ifstream status_file("/proc/self/status");
    if (status_file.is_open()) {
        std::string line;
        while (std::getline(status_file, line)) {
            if (line.find("Threads:") != std::string::npos) {
                sscanf(line.c_str(), "Threads: %u", &metrics.thread_count);
            }
        }
    }
#endif
}

void HardwareMonitor::collect_disk_metrics(HardwareMetrics::SystemMetrics& metrics) {
    // Disk I/O metrics
    metrics.disk_utilization_percent = 0.0; // Placeholder
    metrics.disk_read_bytes_per_sec = 0;
    metrics.disk_write_bytes_per_sec = 0;
}

void HardwareMonitor::collect_network_metrics(HardwareMetrics::SystemMetrics& metrics) {
    // Network I/O metrics
    metrics.network_utilization_percent = 0.0; // Placeholder
    metrics.network_bytes_in_per_sec = 0;
    metrics.network_bytes_out_per_sec = 0;
}

void HardwareMonitor::collect_thermal_metrics(HardwareMetrics& metrics) {
    // Additional thermal monitoring beyond CPU/GPU
    // This could include motherboard temperature sensors, etc.
}

void HardwareMonitor::calculate_performance_metrics(HardwareMetrics& metrics) {
    auto& perf = metrics.performance_metrics;

    // Calculate real-time performance score
    calculate_real_time_performance_score(metrics);

    // Calculate audio pipeline efficiency
    if (metrics.audio_metrics.dsp_load_percent > 0) {
        perf.audio_pipeline_efficiency = std::min(100.0, 100.0 - metrics.audio_metrics.dsp_load_percent);
    }

    // Calculate GPU efficiency
    if (!metrics.gpu_metrics.empty()) {
        double total_gpu_efficiency = 0.0;
        for (const auto& gpu : metrics.gpu_metrics) {
            total_gpu_efficiency += (100.0 - gpu.gpu_utilization_percent);
        }
        perf.gpu_efficiency = total_gpu_efficiency / metrics.gpu_metrics.size();
    }

    // Calculate memory efficiency
    if (metrics.memory_total_bytes > 0) {
        perf.memory_efficiency = std::min(100.0, 100.0 - metrics.memory_utilization_percent);
    }

    // Calculate power efficiency score
    double total_power = 0.0;
    for (const auto& gpu : metrics.gpu_metrics) {
        total_power += gpu.power_consumption_watts;
    }

    if (total_power > 0) {
        // Higher efficiency for lower power consumption
        perf.power_efficiency_score = std::max(0.0, 100.0 - (total_power * 2.0)); // Simple formula
    }
}

void HardwareMonitor::calculate_real_time_performance_score(HardwareMetrics& metrics) {
    double score = 100.0;

    // Penalize high CPU usage
    if (metrics.cpu_utilization_percent > 80.0) {
        score -= (metrics.cpu_utilization_percent - 80.0) * 0.5;
    }

    // Penalize high memory usage
    if (metrics.memory_utilization_percent > 85.0) {
        score -= (metrics.memory_utilization_percent - 85.0) * 0.3;
    }

    // Penalize high temperatures
    if (metrics.cpu_temperature_celsius > 70.0) {
        score -= (metrics.cpu_temperature_celsius - 70.0) * 0.5;
    }

    // Penalize high audio DSP load
    if (metrics.audio_metrics.dsp_load_percent > 75.0) {
        score -= (metrics.audio_metrics.dsp_load_percent - 75.0) * 0.8;
    }

    // Penalize high audio latency
    if (metrics.audio_metrics.processing_latency_ms > 5.0) {
        score -= (metrics.audio_metrics.processing_latency_ms - 5.0) * 2.0;
    }

    // Penalize GPU issues
    for (const auto& gpu : metrics.gpu_metrics) {
        if (gpu.temperature_celsius > 80.0) {
            score -= (gpu.temperature_celsius - 80.0) * 0.4;
        }
        if (gpu.gpu_utilization_percent > 90.0) {
            score -= (gpu.gpu_utilization_percent - 90.0) * 0.3;
        }
    }

    metrics.performance_metrics.real_time_performance_score = std::max(0.0, score);
}

void HardwareMonitor::stream_metrics(const HardwareMetrics& metrics) {
    if (!streamer_ || !metrics.is_valid) {
        return;
    }

    try {
        // Serialize metrics for streaming
        auto serialized = serialize_metrics(metrics);

        // Send through the real-time streamer
        // This would interface with the subscription manager
        streamer->process_audio_frame(nullptr, 0); // Placeholder call

        performance_stats_.streamed_messages.fetch_add(1);

    } catch (const std::exception& e) {
        Logger::error("HardwareMonitor: Exception streaming metrics: {}", e.what());
        performance_stats_.streaming_errors.fetch_add(1);
    }
}

std::vector<uint8_t> HardwareMonitor::serialize_metrics(const HardwareMetrics& metrics) const {
    // Serialize metrics to binary format for streaming
    // This would use the visualization protocol
    std::vector<uint8_t> data;

    // For now, return empty placeholder
    // In a real implementation, this would use the VisualizationProtocol class

    return data;
}

void HardwareMonitor::adjust_collection_intervals(const HardwareMetrics& metrics) {
    if (!config_.enable_batch_collection) {
        return;
    }

    // Adaptive monitoring - adjust intervals based on system load
    bool should_increase = should_increase_frequency(metrics);
    bool should_decrease = should_decrease_frequency(metrics);

    if (should_increase) {
        // Increase collection frequency for critical metrics
        config_.audio_interval_ms = std::max(50u, config_.audio_interval_ms - 10u);
        config_.gpu_interval_ms = std::max(200u, config_.gpu_interval_ms - 50u);
    } else if (should_decrease) {
        // Decrease collection frequency to reduce overhead
        config_.audio_interval_ms = std::min(500u, config_.audio_interval_ms + 10u);
        config_.gpu_interval_ms = std::min(2000u, config_.gpu_interval_ms + 50u);
    }
}

bool HardwareMonitor::should_increase_frequency(const HardwareMetrics& metrics) const {
    // Increase frequency if system is under high load or approaching thresholds
    return (metrics.cpu_utilization_percent > 80.0) ||
           (metrics.audio_metrics.dsp_load_percent > 75.0) ||
           (metrics.audio_metrics.processing_latency_ms > 5.0);
}

bool HardwareMonitor::should_decrease_frequency(const HardwareMetrics& metrics) const {
    // Decrease frequency if system is idle and well within thresholds
    return (metrics.cpu_utilization_percent < 30.0) &&
           (metrics.audio_metrics.dsp_load_percent < 25.0) &&
           (metrics.audio_metrics.processing_latency_ms < 2.0);
}

bool HardwareMonitor::set_thread_priority(std::thread& thread, int priority) {
#ifdef _WIN32
    HANDLE handle = thread.native_handle();
    int priority_class = THREAD_PRIORITY_NORMAL;

    if (priority > 0) {
        priority_class = THREAD_PRIORITY_ABOVE_NORMAL;
    } else if (priority < 0) {
        priority_class = THREAD_PRIORITY_BELOW_NORMAL;
    }

    return SetThreadPriority(handle, priority_class) != FALSE;
#else
    // Linux implementation using pthread scheduling
    sched_param sch_params;
    sch_params.sched_priority = std::abs(priority);

    int policy = (priority > 0) ? SCHED_FIFO : SCHED_OTHER;

    return pthread_setschedparam(thread.native_handle(), policy, &sch_params) == 0;
#endif
}

uint64_t HardwareMonitor::get_current_timestamp_microseconds() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

void HardwareMonitor::update_performance_stats(double collection_time_ms, bool success) {
    std::lock_guard<std::mutex> lock(performance_mutex_);

    performance_stats_.total_collections++;
    if (success) {
        performance_stats_.successful_collections++;
    }

    // Update timing statistics
    if (collection_time_ms > performance_stats_.max_collection_time_ms) {
        performance_stats_.max_collection_time_ms = collection_time_ms;
    }

    if (collection_time_ms < performance_stats_.min_collection_time_ms) {
        performance_stats_.min_collection_time_ms = collection_time_ms;
    }

    // Update average
    performance_stats_.avg_collection_time_ms =
        (performance_stats_.avg_collection_time_ms * (performance_stats_.total_collections - 1) + collection_time_ms) /
        performance_stats_.total_collections;
}

// Factory implementations
std::unique_ptr<HardwareMonitor> HardwareMonitorFactory::create_default() {
    auto monitor = std::make_unique<HardwareMonitor>();
    MonitorConfig config;
    monitor->initialize(config);
    return monitor;
}

std::unique_ptr<HardwareMonitor> HardwareMonitorFactory::create_high_performance() {
    auto monitor = std::make_unique<HardwareMonitor>();
    MonitorConfig config;
    config.enable_high_precision_timing = true;
    config.enable_caching = true;
    config.cpu_interval_ms = 500;
    config.gpu_interval_ms = 250;
    config.audio_interval_ms = 50;
    monitor->initialize(config);
    return monitor;
}

std::unique_ptr<HardwareMonitor> HardwareMonitorFactory::create_low_overhead() {
    auto monitor = std::make_unique<HardwareMonitor>();
    MonitorConfig config;
    config.enable_caching = false;
    config.cpu_interval_ms = 2000;
    config.gpu_interval_ms = 1000;
    config.audio_interval_ms = 200;
    config.enable_detailed_logging = false;
    monitor->initialize(config);
    return monitor;
}

std::unique_ptr<HardwareMonitor> HardwareMonitorFactory::create_comprehensive() {
    auto monitor = std::make_unique<HardwareMonitor>();
    MonitorConfig config;
    config.enable_nvidia_ml = true;
    config.enable_amd_smi = true;
    config.enable_intel_gpu = true;
    config.enable_opencl_monitoring = true;
    config.enable_vulkan_monitoring = true;
    config.enable_cuda_monitoring = true;
    config.enable_disk_monitoring = true;
    config.enable_network_monitoring = true;
    config.enable_thermal_monitoring = true;
    config.enable_power_monitoring = true;
    config.enable_performance_profiling = true;
    monitor->initialize(config);
    return monitor;
}

// Utility function implementations
namespace hardware_utils {

double celsius_to_fahrenheit(double celsius) {
    return celsius * 9.0 / 5.0 + 32.0;
}

double celsius_to_kelvin(double celsius) {
    return celsius + 273.15;
}

std::string format_bytes(uint64_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit < 4) {
        size /= 1024.0;
        unit++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
    return oss.str();
}

std::string format_frequency(uint64_t hz) {
    const char* units[] = {"Hz", "kHz", "MHz", "GHz"};
    int unit = 0;
    double freq = static_cast<double>(hz);

    while (freq >= 1000.0 && unit < 3) {
        freq /= 1000.0;
        unit++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << freq << " " << units[unit];
    return oss.str();
}

std::string format_duration(uint64_t microseconds) {
    uint64_t seconds = microseconds / 1000000;
    uint64_t minutes = seconds / 60;
    uint64_t hours = minutes / 60;
    uint64_t days = hours / 24;

    seconds %= 60;
    minutes %= 60;
    hours %= 60;

    std::ostringstream oss;
    if (days > 0) {
        oss << days << "d ";
    }
    if (hours > 0) {
        oss << hours << "h ";
    }
    if (minutes > 0) {
        oss << minutes << "m ";
    }
    oss << seconds << "s";

    return oss.str();
}

double calculate_utilization_percentage(uint64_t used, uint64_t total) {
    if (total == 0) return 0.0;
    return (static_cast<double>(used) / static_cast<double>(total)) * 100.0;
}

double calculate_efficiency_score(uint64_t useful_work, uint64_t total_work) {
    if (total_work == 0) return 0.0;
    return (static_cast<double>(useful_work) / static_cast<double>(total_work)) * 100.0;
}

double calculate_throughput(uint64_t items, std::chrono::milliseconds duration) {
    if (duration.count() == 0) return 0.0;
    return static_cast<double>(items) / (static_cast<double>(duration.count()) / 1000.0);
}

} // namespace hardware_utils

} // namespace vortex::hardware