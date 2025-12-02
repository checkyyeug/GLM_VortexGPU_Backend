#include "hardware/gpu_utilization_tracker.hpp"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <fstream>

#ifdef VORTEX_ENABLE_NVML
#include <nvml.h>
#endif

#ifdef VORTEX_ENABLE_CUDA
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#endif

#ifdef _WIN32
#include <intrin.h>
#else
#include <unistd.h>
#include <sys/time.h>
#include <fstream>
#endif

namespace vortex::hardware {

GPUUtilizationTracker::GPUUtilizationTracker() {
    Logger::info("GPUUtilizationTracker: Creating instance");
    performance_stats_.start_time = std::chrono::steady_clock::now();
}

GPUUtilizationTracker::~GPUUtilizationTracker() {
    shutdown();
}

bool GPUUtilizationTracker::initialize(const GPUTrackerConfig& config) {
    if (initialized_.load()) {
        Logger::warn("GPUUtilizationTracker already initialized");
        return true;
    }

    config_ = config;

    Logger::info("GPUUtilizationTracker: Initializing with utilization interval: {}ms, audio interval: {}ms",
                 config_.utilization_interval_ms, config_.audio_interval_ms);

    try {
        // Initialize platform-specific monitors
        if (!initialize_platform_monitors()) {
            Logger::error("GPUUtilizationTracker: Failed to initialize platform monitors");
            return false;
        }

        // Initialize NVIDIA monitoring
        if (config_.enable_nvidia_ml && !initialize_nvidia_monitoring()) {
            Logger::warn("GPUUtilizationTracker: NVIDIA monitoring initialization failed");
        }

        // Initialize AMD monitoring
        if (config_.enable_amd_smi && !initialize_amd_monitoring()) {
            Logger::warn("GPUUtilizationTracker: AMD monitoring initialization failed");
        }

        // Initialize Intel monitoring
        if (config_.enable_intel_gpu && !initialize_intel_monitoring()) {
            Logger::warn("GPUUtilizationTracker: Intel monitoring initialization failed");
        }

        // Reserve space for device tracking
        device_threads_.reserve(16); // Reserve space for up to 16 GPUs

        initialized_.store(true);
        Logger::info("GPUUtilizationTracker initialized successfully");
        return true;

    } catch (const std::exception& e) {
        Logger::error("GPUUtilizationTracker: Exception during initialization: {}", e.what());
        return false;
    }
}

void GPUUtilizationTracker::shutdown() {
    if (!initialized_.load()) {
        return;
    }

    Logger::info("GPUUtilizationTracker: Shutting down");

    // Stop tracking
    stop_tracking();

    // Signal shutdown
    shutdown_requested_.store(true);

    // Wait for all threads to finish
    if (streaming_thread_.joinable()) {
        streaming_thread_.join();
    }

    for (auto& thread : device_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // Cleanup platform-specific resources
    cleanup_platform_monitors();
    cleanup_nvidia_monitoring();
    cleanup_amd_monitoring();
    cleanup_intel_monitoring();

    // Clear data structures
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        latest_metrics_.clear();
        metrics_history_.clear();
    }

    {
        std::lock_guard<std::mutex> lock(audio_kernels_mutex_);
        audio_kernels_.clear();
    }

    {
        std::lock_guard<std::mutex> lock(devices_mutex_);
        tracked_devices_.clear();
        device_names_.clear();
    }

    initialized_.store(false);
    shutdown_requested_.store(false);

    // Log final performance statistics
    if (config_.enable_performance_profiling) {
        Logger::info("GPUUtilizationTracker final stats: collections={}, success_rate={:.1f}%, avg_time={:.2f}ms",
                     performance_stats_.total_collections,
                     performance_stats_.total_collections > 0 ?
                         (performance_stats_.successful_collections * 100.0 / performance_stats_.total_collections) : 0.0,
                     performance_stats_.avg_collection_time_ms);
    }

    Logger::info("GPUUtilizationTracker shutdown complete");
}

bool GPUUtilizationTracker::start_tracking() {
    if (!initialized_.load()) {
        Logger::error("GPUUtilizationTracker: Cannot start tracking - not initialized");
        return false;
    }

    if (tracking_.load()) {
        Logger::warn("GPUUtilizationTracker: Already tracking");
        return true;
    }

    {
        std::lock_guard<std::mutex> lock(devices_mutex_);
        if (tracked_devices_.empty()) {
            Logger::warn("GPUUtilizationTracker: No devices to track");
            return false;
        }
    }

    Logger::info("GPUUtilizationTracker: Starting tracking for {} devices", tracked_devices_.size());

    try {
        // Reset shutdown flag
        shutdown_requested_.store(false);
        paused_.store(false);

        // Start device tracking threads
        device_threads_.clear();
        for (uint32_t device_id : tracked_devices_) {
            device_threads_.emplace_back(&GPUUtilizationTracker::device_tracking_thread, this, device_id);
        }

        // Start streaming thread if enabled
        if (realtime_streaming_enabled_.load()) {
            streaming_thread_ = std::thread(&GPUUtilizationTracker::streaming_thread, this);
        }

        // Set thread priorities for audio processing
        for (auto& thread : device_threads_) {
            set_thread_priority(thread, 1); // High priority for audio
        }

        tracking_.store(true);
        Logger::info("GPUUtilizationTracker: Tracking started successfully");
        return true;

    } catch (const std::exception& e) {
        Logger::error("GPUUtilizationTracker: Exception starting tracking: {}", e.what());
        return false;
    }
}

void GPUUtilizationTracker::stop_tracking() {
    if (!tracking_.load()) {
        return;
    }

    Logger::info("GPUUtilizationTracker: Stopping tracking");

    tracking_.store(false);
    shutdown_requested_.store(true);

    // Wait for all threads to finish
    if (streaming_thread_.joinable()) {
        streaming_thread_.join();
    }

    for (auto& thread : device_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    device_threads_.clear();
    Logger::info("GPUUtilizationTracker: Tracking stopped");
}

std::vector<uint32_t> GPUUtilizationTracker::get_available_devices() const {
    std::lock_guard<std::mutex> lock(devices_mutex_);
    return tracked_devices_;
}

bool GPUUtilizationTracker::add_device(uint32_t device_id) {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    // Check if device is already being tracked
    if (std::find(tracked_devices_.begin(), tracked_devices_.end(), device_id) != tracked_devices_.end()) {
        Logger::warn("GPUUtilizationTracker: Device {} already being tracked", device_id);
        return false;
    }

    // Validate device is available
    if (!is_device_available(device_id)) {
        Logger::error("GPUUtilizationTracker: Device {} is not available", device_id);
        return false;
    }

    // Add to tracking list
    tracked_devices_.push_back(device_id);

    // Initialize metrics storage for this device
    {
        std::lock_guard<std::mutex> metrics_lock(metrics_mutex_);
        latest_metrics_[device_id] = GPUUtilizationMetrics{};
        metrics_history_[device_id] = std::vector<GPUUtilizationMetrics>{};
        metrics_history_[device_id].reserve(config_.cache_size);
    }

    // Get device name
    std::string device_name = get_device_name(device_id);
    device_names_[device_id] = device_name;

    Logger::info("GPUUtilizationTracker: Added device {} ({})", device_id, device_name);
    return true;
}

bool GPUUtilizationTracker::remove_device(uint32_t device_id) {
    std::lock_guard<std::mutex> lock(devices_mutex_);

    auto it = std::find(tracked_devices_.begin(), tracked_devices_.end(), device_id);
    if (it == tracked_devices_.end()) {
        Logger::warn("GPUUtilizationTracker: Device {} not found in tracking list", device_id);
        return false;
    }

    // Remove from tracking list
    tracked_devices_.erase(it);
    device_names_.erase(device_id);

    // Remove from metrics storage
    {
        std::lock_guard<std::mutex> metrics_lock(metrics_mutex_);
        latest_metrics_.erase(device_id);
        metrics_history_.erase(device_id);
    }

    Logger::info("GPUUtilizationTracker: Removed device {}", device_id);
    return true;
}

bool GPUUtilizationTracker::is_device_available(uint32_t device_id) const {
#ifdef VORTEX_ENABLE_NVML
    if (config_.enable_nvidia_ml && nvml_initialized_) {
        nvmlDevice_t device;
        nvmlReturn_t result = nvmlDeviceGetHandleByIndex(device_id, &device);
        return result == NVML_SUCCESS;
    }
#endif

#ifdef VORTEX_ENABLE_CUDA
    if (config_.enable_cuda_profiling) {
        int device_count = 0;
        cudaError_t result = cudaGetDeviceCount(&device_count);
        if (result == cudaSuccess && device_id < device_count) {
            cudaDeviceProp prop;
            result = cudaGetDeviceProperties(&prop, device_id);
            return result == cudaSuccess;
        }
    }
#endif

    // Default fallback - assume device is available
    return true;
}

std::string GPUUtilizationTracker::get_device_name(uint32_t device_id) const {
    {
        std::lock_guard<std::mutex> lock(devices_mutex_);
        auto it = device_names_.find(device_id);
        if (it != device_names_.end()) {
            return it->second;
        }
    }

#ifdef VORTEX_ENABLE_NVML
    if (config_.enable_nvidia_ml && nvml_initialized_) {
        nvmlDevice_t device;
        if (nvmlDeviceGetHandleByIndex(device_id, &device) == NVML_SUCCESS) {
            char name[NVML_DEVICE_NAME_V2_BUFFER_SIZE];
            if (nvmlDeviceGetName(device, name, sizeof(name)) == NVML_SUCCESS) {
                return std::string(name);
            }
        }
    }
#endif

#ifdef VORTEX_ENABLE_CUDA
    if (config_.enable_cuda_profiling) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
            return std::string(prop.name);
        }
    }
#endif

    return "Unknown GPU";
}

GPUUtilizationTracker::GPUPerformanceReport GPUUtilizationTracker::generate_performance_report(
    uint32_t device_id, std::chrono::seconds duration) const {

    GPUPerformanceReport report;
    report.device_name = get_device_name(device_id);
    report.report_start = std::chrono::steady_clock::now() - duration;
    report.report_end = std::chrono::steady_clock::now();

    // Get metrics history for the specified duration
    auto metrics_history = get_metrics_history(device_id, config_.cache_size);
    if (metrics_history.empty()) {
        Logger::warn("GPUUtilizationTracker: No metrics history available for device {}", device_id);
        return report;
    }

    // Filter metrics by duration
    std::vector<GPUUtilizationMetrics> filtered_metrics;
    auto cutoff_time = std::chrono::steady_clock::now() - duration;
    for (const auto& metrics : metrics_history) {
        if (metrics.collection_time >= cutoff_time) {
            filtered_metrics.push_back(metrics);
        }
    }

    if (filtered_metrics.empty()) {
        return report;
    }

    // Calculate statistics
    double total_utilization = 0.0;
    double total_temperature = 0.0;
    double total_power = 0.0;
    double max_utilization = 0.0;
    double max_temperature = 0.0;
    double max_power = 0.0;
    uint64_t throttling_events = 0;

    for (const auto& metrics : filtered_metrics) {
        if (metrics.is_valid) {
            total_utilization += metrics.gpu_utilization_percent;
            total_temperature += metrics.temperature_gpu_celsius;
            total_power += metrics.power_usage_watts;

            max_utilization = std::max(max_utilization, metrics.gpu_utilization_percent);
            max_temperature = std::max(max_temperature, metrics.temperature_gpu_celsius);
            max_power = std::max(max_power, metrics.power_usage_watts);

            if (metrics.is_thermal_throttling) {
                throttling_events++;
            }
        }
    }

    size_t valid_metrics_count = filtered_metrics.size();
    if (valid_metrics_count > 0) {
        report.average_utilization = total_utilization / valid_metrics_count;
        report.average_temperature = total_temperature / valid_metrics_count;
        report.average_power_usage = total_power / valid_metrics_count;
    }

    report.peak_utilization = max_utilization;
    report.peak_temperature = max_temperature;
    report.peak_power_usage = max_power;
    report.total_throttling_events = throttling_events;

    // Calculate efficiency scores from the latest metrics
    if (!filtered_metrics.empty()) {
        const auto& latest = filtered_metrics.back();
        report.memory_efficiency = latest.performance_metrics.memory_efficiency;
        report.compute_efficiency = latest.performance_metrics.compute_efficiency;
        report.audio_efficiency = latest.audio_metrics.audio_efficiency_score;
        report.real_time_score = latest.real_time_metrics.real_time_score;
    }

    return report;
}

void GPUUtilizationTracker::set_streaming_interface(std::shared_ptr<network::RealtimeStreamer> streamer) {
    streamer_ = streamer;
    Logger::info("GPUUtilizationTracker: Streaming interface {}", streamer ? "set" : "cleared");
}

bool GPUUtilizationTracker::enable_realtime_streaming(bool enabled) {
    realtime_streaming_enabled_.store(enabled);
    Logger::info("GPUUtilizationTracker: Real-time streaming {}", enabled ? "enabled" : "disabled");
    return true;
}

GPUUtilizationTracker::TrackerPerformance GPUUtilizationTracker::get_performance_stats() const {
    std::lock_guard<std::mutex> lock(performance_mutex_);
    return performance_stats_;
}

std::string GPUUtilizationTracker::get_diagnostics_report() const {
    std::ostringstream report;

    report << "=== GPUUtilizationTracker Diagnostics Report ===\n";
    report << "Initialized: " << (initialized_.load() ? "Yes" : "No") << "\n";
    report << "Tracking: " << (tracking_.load() ? "Yes" : "No") << "\n";
    report << "Real-time streaming: " << (realtime_streaming_enabled_.load() ? "Yes" : "No") << "\n";
    report << "Tracked devices: " << tracked_devices_.size() << "\n\n";

    auto perf = get_performance_stats();
    report << "Performance Statistics:\n";
    report << "  Total collections: " << perf.total_collections << "\n";
    report << "  Successful collections: " << perf.successful_collections << "\n";
    report << "  Success rate: " << std::fixed << std::setprecision(1)
           << (perf.total_collections > 0 ?
               (perf.successful_collections * 100.0 / perf.total_collections) : 0.0) << "%\n";
    report << "  Average collection time: " << std::setprecision(2) << perf.avg_collection_time_ms << "ms\n";
    report << "  Max collection time: " << perf.max_collection_time_ms << "ms\n";

    report << "\nDevice Information:\n";
    {
        std::lock_guard<std::mutex> lock(devices_mutex_);
        for (uint32_t device_id : tracked_devices_) {
            auto it = device_names_.find(device_id);
            std::string device_name = (it != device_names_.end()) ? it->second : "Unknown";
            report << "  Device " << device_id << ": " << device_name << "\n";
        }
    }

    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - perf.start_time).count();
    report << "  Tracker uptime: " << uptime << " seconds\n";

    return report.str();
}

void GPUUtilizationTracker::register_audio_kernel(const std::string& kernel_name, uint32_t device_id) {
    std::lock_guard<std::mutex> lock(audio_kernels_mutex_);

    AudioKernelInfo info;
    info.name = kernel_name;
    info.device_id = device_id;
    info.execution_count.store(0);
    info.total_execution_time_ms.store(0.0);
    info.max_execution_time_ms.store(0.0);
    info.is_executing.store(false);

    audio_kernels_[kernel_name] = info;
    Logger::debug("GPUUtilizationTracker: Registered audio kernel {} on device {}", kernel_name, device_id);
}

void GPUUtilizationTracker::start_audio_kernel_execution(const std::string& kernel_name, uint32_t device_id) {
    std::lock_guard<std::mutex> lock(audio_kernels_mutex_);

    auto it = audio_kernels_.find(kernel_name);
    if (it != audio_kernels_.end()) {
        it->second.is_executing.store(true);
        it->second.last_execution_start = std::chrono::high_resolution_clock::now();
    }
}

void GPUUtilizationTracker::end_audio_kernel_execution(const std::string& kernel_name, uint32_t device_id) {
    std::lock_guard<std::mutex> lock(audio_kernels_mutex_);

    auto it = audio_kernels_.find(kernel_name);
    if (it != audio_kernels_.end() && it->second.is_executing.load()) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - it->second.last_execution_start).count();
        double duration_ms = duration / 1000.0;

        it->second.execution_count.fetch_add(1);
        it->second.total_execution_time_ms.fetch_add(duration_ms);

        // Update max execution time
        double current_max = it->second.max_execution_time_ms.load();
        while (duration_ms > current_max &&
               !it->second.max_execution_time_ms.compare_exchange_weak(current_max, duration_ms)) {
            // Retry if another thread updated the max value
        }

        it->second.is_executing.store(false);
    }
}

void GPUUtilizationTracker::track_audio_memory_allocation(uint32_t device_id, size_t bytes) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    auto it = latest_metrics_.find(device_id);
    if (it != latest_metrics_.end()) {
        it->second.audio_metrics.audio_memory_used_bytes.fetch_add(bytes);
    }
}

void GPUUtilizationTracker::track_audio_memory_deallocation(uint32_t device_id, size_t bytes) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    auto it = latest_metrics_.find(device_id);
    if (it != latest_metrics_.end()) {
        it->second.audio_metrics.audio_memory_used_bytes.fetch_sub(bytes);
    }
}

// Private implementation methods

void GPUUtilizationTracker::device_tracking_thread(uint32_t device_id) {
    Logger::debug("GPUUtilizationTracker: Device {} tracking thread started", device_id);

    while (!shutdown_requested_.load()) {
        if (paused_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        // Collect metrics based on configuration
        collect_all_metrics(device_id);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Adaptive monitoring
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.utilization_interval_ms) - duration);
    }

    Logger::debug("GPUUtilizationTracker: Device {} tracking thread stopped", device_id);
}

void GPUUtilizationTracker::streaming_thread() {
    Logger::debug("GPUUtilizationTracker: Streaming thread started");

    while (!shutdown_requested_.load()) {
        if (paused_.load() || !realtime_streaming_enabled_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // Stream metrics for all devices
        for (uint32_t device_id : tracked_devices_) {
            auto metrics = get_latest_metrics(device_id);
            if (metrics.is_valid) {
                stream_metrics(device_id, metrics);
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(config_.streaming_interval_ms));
    }

    Logger::debug("GPUUtilizationTracker: Streaming thread stopped");
}

bool GPUUtilizationTracker::initialize_platform_monitors() {
    // Platform-specific initialization would go here
    Logger::debug("GPUUtilizationTracker: Platform monitors initialized");
    return true;
}

void GPUUtilizationTracker::cleanup_platform_monitors() {
    // Platform-specific cleanup would go here
}

bool GPUUtilizationTracker::initialize_nvidia_monitoring() {
#ifdef VORTEX_ENABLE_NVML
    nvmlReturn_t result = nvmlInit();
    if (result == NVML_SUCCESS) {
        nvml_initialized_ = true;
        result = nvmlDeviceGetCount(&nvml_device_count_);
        if (result == NVML_SUCCESS) {
            Logger::info("GPUUtilizationTracker: NVML initialized with {} GPUs", nvml_device_count_);

            // Initialize device handles
            for (unsigned int i = 0; i < nvml_device_count_; ++i) {
                nvmlDevice_t device;
                if (nvmlDeviceGetHandleByIndex(i, &device) == NVML_SUCCESS) {
                    nvml_devices_[i] = device;
                }
            }

            return true;
        }
    }
    Logger::warn("GPUUtilizationTracker: NVML initialization failed: {}", nvmlErrorString(result));
#endif

    return false;
}

void GPUUtilizationTracker::cleanup_nvidia_monitoring() {
#ifdef VORTEX_ENABLE_NVML
    if (nvml_initialized_) {
        nvmlShutdown();
        nvml_initialized_ = false;
        nvml_device_count_ = 0;
        nvml_devices_.clear();
    }
#endif
}

GPUUtilizationMetrics GPUUtilizationTracker::collect_all_metrics(uint32_t device_id) {
    GPUUtilizationMetrics metrics;
    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        metrics.device_id = device_id;
        metrics.device_name = get_device_name(device_id);
        metrics.timestamp_microseconds = get_current_timestamp_microseconds();
        metrics.collection_time = start_time;

        // Collect all metric categories
        collect_utilization_metrics(device_id);
        collect_memory_metrics(device_id);
        collect_thermal_metrics(device_id);
        collect_power_metrics(device_id);
        collect_audio_metrics(device_id);
        collect_performance_metrics(device_id);
        collect_clock_metrics(device_id);

        // Update historical data
        update_historical_data(metrics, device_id);

        // Update latest metrics
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            latest_metrics_[device_id] = metrics;
        }

        metrics.is_valid = true;

    } catch (const std::exception& e) {
        Logger::error("GPUUtilizationTracker: Exception collecting metrics for device {}: {}", device_id, e.what());
        metrics.is_valid = false;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    metrics.collection_duration_ms = duration.count() / 1000.0;

    // Update performance statistics
    update_performance_stats(metrics.collection_duration_ms, metrics.is_valid);

    return metrics;
}

GPUUtilizationMetrics GPUUtilizationTracker::collect_utilization_metrics(uint32_t device_id) {
    GPUUtilizationMetrics metrics;

    // Try NVIDIA metrics first
    if (config_.enable_nvidia_ml && nvml_initialized_) {
        metrics = collect_nvidia_metrics(device_id);
        if (metrics.is_valid) {
            return metrics;
        }
    }

    // Fall back to AMD metrics
    if (config_.enable_amd_smi) {
        metrics = collect_amd_metrics(device_id);
        if (metrics.is_valid) {
            return metrics;
        }
    }

    // Fall back to Intel metrics
    if (config_.enable_intel_gpu) {
        metrics = collect_intel_metrics(device_id);
        if (metrics.is_valid) {
            return metrics;
        }
    }

    return metrics;
}

GPUUtilizationMetrics GPUUtilizationTracker::collect_nvidia_metrics(uint32_t device_id) {
    GPUUtilizationMetrics metrics;
    metrics.device_id = device_id;

#ifdef VORTEX_ENABLE_NVML
    if (!nvml_initialized_) {
        return metrics;
    }

    auto it = nvml_devices_.find(device_id);
    if (it == nvml_devices_.end()) {
        Logger::warn("GPUUtilizationTracker: NVML device {} not found", device_id);
        return metrics;
    }

    nvmlDevice_t device = it->second;

    // Device name
    char name[NVML_DEVICE_NAME_V2_BUFFER_SIZE];
    if (nvmlDeviceGetName(device, name, sizeof(name)) == NVML_SUCCESS) {
        metrics.device_name = name;
        metrics.vendor_name = "NVIDIA";
    }

    // Utilization
    nvmlUtilization_t utilization;
    if (nvmlDeviceGetUtilizationRates(device, &utilization) == NVML_SUCCESS) {
        metrics.gpu_utilization_percent = utilization.gpu;
        metrics.memory_utilization_percent = utilization.memory;
    }

    // Memory
    nvmlMemory_t memory;
    if (nvmlDeviceGetMemoryInfo(device, &memory) == NVML_SUCCESS) {
        metrics.memory_total_bytes = memory.total;
        metrics.memory_used_bytes = memory.used;
        metrics.memory_free_bytes = memory.free;
        if (memory.total > 0) {
            metrics.memory_utilization_percent = (memory.used * 100.0) / memory.total;
        }
    }

    // Temperature
    unsigned int temp;
    if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
        metrics.temperature_gpu_celsius = static_cast<double>(temp);
    }

    // Power consumption
    unsigned int power;
    if (nvmlDeviceGetPowerUsage(device, &power) == NVML_SUCCESS) {
        metrics.power_usage_watts = power / 1000.0; // Convert mW to W
    }

    // Clock frequencies
    unsigned int graphics_clock, memory_clock;
    if (nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &graphics_clock) == NVML_SUCCESS) {
        metrics.graphics_clock_hz = graphics_clock * 1000000ULL;
    }
    if (nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &memory_clock) == NVML_SUCCESS) {
        metrics.memory_clock_hz = memory_clock * 1000000ULL;
    }

    // Performance state
    if (nvmlDeviceGetPerformanceState(device, reinterpret_cast<nvmlPstates_t*>(&metrics.current_performance_state)) == NVML_SUCCESS) {
        metrics.is_performance_state_maximum = (metrics.current_performance_state == NVML_PSTATE_0);
    }

    metrics.is_valid = true;
    Logger::debug("GPUUtilizationTracker: Collected NVML metrics for device {}", device_id);

#else
    Logger::debug("GPUUtilizationTracker: NVML support not compiled in");
#endif

    return metrics;
}

GPUUtilizationMetrics GPUUtilizationTracker::collect_amd_metrics(uint32_t device_id) {
    GPUUtilizationMetrics metrics;
    metrics.device_id = device_id;

    // AMD GPU metrics implementation would use ADL SDK or other APIs
    // Placeholder implementation
    Logger::debug("GPUUtilizationTracker: AMD GPU metrics collection not yet implemented for device {}", device_id);

    return metrics;
}

GPUUtilizationMetrics GPUUtilizationTracker::collect_intel_metrics(uint32_t device_id) {
    GPUUtilizationMetrics metrics;
    metrics.device_id = device_id;

    // Intel GPU metrics implementation would use Intel GPU APIs
    // Placeholder implementation
    Logger::debug("GPUUtilizationTracker: Intel GPU metrics collection not yet implemented for device {}", device_id);

    return metrics;
}

void GPUUtilizationTracker::update_audio_metrics(GPUUtilizationMetrics& metrics, uint32_t device_id) {
    if (!config_.enable_audio_kernel_tracking) {
        return;
    }

    std::lock_guard<std::mutex> lock(audio_kernels_mutex_);

    // Count active audio kernels
    uint32_t active_kernels = 0;
    uint32_t total_kernels = 0;
    double total_execution_time = 0.0;
    double max_execution_time = 0.0;

    for (const auto& [name, info] : audio_kernels_) {
        if (info.device_id == device_id) {
            total_kernels++;
            if (info.is_executing.load()) {
                active_kernels++;
            }
            total_execution_time += info.total_execution_time_ms.load();
            max_execution_time = std::max(max_execution_time, info.max_execution_time_ms.load());
        }
    }

    auto& audio = metrics.audio_metrics;
    audio.active_audio_kernels = active_kernels;
    audio.total_audio_kernels = total_kernels;

    if (total_kernels > 0) {
        audio.audio_kernel_utilization_percent = (active_kernels * 100.0) / total_kernels;
    }

    audio.audio_processing_time_ms = total_execution_time;

    // Calculate efficiency score
    calculate_audio_efficiency(metrics);

    // Get audio memory usage
    audio.audio_memory_used_bytes = metrics.audio_metrics.audio_memory_used_bytes.load();
}

void GPUUtilizationTracker::calculate_audio_efficiency(GPUUtilizationMetrics& metrics) {
    if (!config_.enable_audio_efficiency_analysis) {
        return;
    }

    auto& audio = metrics.audio_metrics;

    // Calculate efficiency score based on multiple factors
    double efficiency_score = 100.0;

    // Penalize high latency
    if (audio.audio_latency_ms > 0.0) {
        efficiency_score -= std::min(50.0, audio.audio_latency_ms * 5.0); // -5 points per ms
    }

    // Penalize dropped samples
    if (audio.audio_samples_processed > 0) {
        double drop_rate = (audio.audio_samples_dropped * 100.0) / audio.audio_samples_processed;
        efficiency_score -= std::min(30.0, drop_rate); // -1 point per % dropped
    }

    // Reward good throughput
    if (audio.audio_throughput_samples_per_sec > 0.0) {
        // Bonus for high throughput (assuming target of 48000 samples/sec)
        double throughput_ratio = audio.audio_throughput_samples_per_sec / 48000.0;
        if (throughput_ratio > 0.0) {
            efficiency_score += std::min(20.0, throughput_ratio * 20.0); // +20 points for full target
        }
    }

    // Penalize high kernel execution time
    if (audio.audio_processing_time_ms > 0.0) {
        double avg_execution_time = audio.audio_processing_time_ms / std::max(1ULL, audio.total_audio_kernels);
        if (avg_execution_time > 1.0) { // If average kernel takes more than 1ms
            efficiency_score -= std::min(15.0, avg_execution_time);
        }
    }

    audio.audio_efficiency_score = std::max(0.0, std::min(100.0, efficiency_score));
}

void GPUUtilizationTracker::update_historical_data(GPUUtilizationMetrics& metrics, uint32_t device_id) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    auto& history = metrics_history_[device_id];

    // Add new metrics to history
    history.push_back(metrics);

    // Maintain maximum history size
    if (history.size() > config_.cache_size) {
        history.erase(history.begin());
    }

    // Update historical data vectors
    auto& hist_data = metrics.historical_data;
    hist_data.utilization_history.push_back(metrics.gpu_utilization_percent);
    hist_data.temperature_history.push_back(metrics.temperature_gpu_celsius);
    hist_data.power_history.push_back(metrics.power_usage_watts);
    hist_data.memory_history.push_back(metrics.memory_utilization_percent);
    hist_data.audio_latency_history.push_back(metrics.audio_metrics.audio_latency_ms);
    hist_data.last_update = metrics.collection_time;

    // Maintain maximum history size
    if (hist_data.utilization_history.size() > hist_data.max_history_size) {
        hist_data.utilization_history.erase(hist_data.utilization_history.begin());
    }
    if (hist_data.temperature_history.size() > hist_data.max_history_size) {
        hist_data.temperature_history.erase(hist_data.temperature_history.begin());
    }
    if (hist_data.power_history.size() > hist_data.max_history_size) {
        hist_data.power_history.erase(hist_data.power_history.begin());
    }
    if (hist_data.memory_history.size() > hist_data.max_history_size) {
        hist_data.memory_history.erase(hist_data.memory_history.begin());
    }
    if (hist_data.audio_latency_history.size() > hist_data.max_history_size) {
        hist_data.audio_latency_history.erase(hist_data.audio_latency_history.begin());
    }
}

GPUUtilizationMetrics GPUUtilizationTracker::get_latest_metrics(uint32_t device_id) const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    auto it = latest_metrics_.find(device_id);
    return (it != latest_metrics_.end()) ? it->second : GPUUtilizationMetrics{};
}

std::vector<GPUUtilizationMetrics> GPUUtilizationTracker::get_metrics_history(uint32_t device_id, size_t count) const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    auto it = metrics_history_.find(device_id);
    if (it == metrics_history_.end() || it->second.empty()) {
        return {};
    }

    const auto& history = it->second;
    size_t start_index = (count >= history.size()) ? 0 : (history.size() - count);

    return std::vector<GPUUtilizationMetrics>(history.begin() + start_index, history.end());
}

void GPUUtilizationTracker::stream_metrics(uint32_t device_id, const GPUUtilizationMetrics& metrics) {
    if (!streamer_ || !metrics.is_valid) {
        return;
    }

    try {
        // Serialize metrics for streaming
        auto serialized = serialize_metrics(metrics);

        // Send through the real-time streamer with channel-specific data
        std::ostringstream channel;
        channel << "gpu_metrics_" << device_id;

        // This would interface with the streaming system
        streamer->process_audio_frame(nullptr, 0); // Placeholder call

        performance_stats_.streamed_messages.fetch_add(1);

    } catch (const std::exception& e) {
        Logger::error("GPUUtilizationTracker: Exception streaming metrics for device {}: {}", device_id, e.what());
        performance_stats_.streaming_errors.fetch_add(1);
    }
}

std::vector<uint8_t> GPUUtilizationTracker::serialize_metrics(const GPUUtilizationMetrics& metrics) const {
    // Serialize metrics to binary format for streaming
    // This would use the visualization protocol
    std::vector<uint8_t> data;

    // For now, return empty placeholder
    // In a real implementation, this would use the VisualizationProtocol class

    return data;
}

uint64_t GPUUtilizationTracker::get_current_timestamp_microseconds() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

void GPUUtilizationTracker::update_performance_stats(double collection_time_ms, bool success) {
    std::lock_guard<std::mutex> lock(performance_mutex_);

    performance_stats_.total_collections++;
    if (success) {
        performance_stats_.successful_collections++;
    }

    // Update timing statistics
    if (collection_time_ms > performance_stats_.max_collection_time_ms) {
        performance_stats_.max_collection_time_ms = collection_time_ms;
    }

    // Update average
    performance_stats_.avg_collection_time_ms =
        (performance_stats_.avg_collection_time_ms * (performance_stats_.total_collections - 1) + collection_time_ms) /
        performance_stats_.total_collections;
}

// Factory implementations
std::unique_ptr<GPUUtilizationTracker> GPUUtilizationTrackerFactory::create_default() {
    auto tracker = std::make_unique<GPUUtilizationTracker>();
    GPUTrackerConfig config;
    tracker->initialize(config);
    return tracker;
}

std::unique_ptr<GPUUtilizationTracker> GPUUtilizationTrackerFactory::create_high_performance() {
    auto tracker = std::make_unique<GPUUtilizationTracker>();
    GPUTrackerConfig config;
    config.enable_high_precision_timing = true;
    config.enable_caching = true;
    config.utilization_interval_ms = 50;
    config.audio_interval_ms = 25;
    config.enable_detailed_profiling = true;
    tracker->initialize(config);
    return tracker;
}

std::unique_ptr<GPUUtilizationTracker> GPUUtilizationTrackerFactory::create_low_overhead() {
    auto tracker = std::make_unique<GPUUtilizationTracker>();
    GPUTrackerConfig config;
    config.enable_low_overhead_mode = true;
    config.utilization_interval_ms = 500;
    config.audio_interval_ms = 200;
    config.enable_detailed_logging = false;
    config.enable_kernel_level_tracking = false;
    tracker->initialize(config);
    return tracker;
}

std::unique_ptr<GPUUtilizationTracker> GPUUtilizationTrackerFactory::create_comprehensive() {
    auto tracker = std::make_unique<GPUUtilizationTracker>();
    GPUTrackerConfig config;
    config.enable_nvidia_ml = true;
    config.enable_amd_smi = true;
    config.enable_intel_gpu = true;
    config.enable_nvidia_nsight = true;
    config.enable_opencl_profiling = true;
    config.enable_vulkan_profiling = true;
    config.enable_cuda_profiling = true;
    config.enable_kernel_level_tracking = true;
    config.enable_predictive_analysis = true;
    config.enable_anomaly_detection = true;
    config.enable_automatic_optimization = true;
    config.enable_detailed_profiling = true;
    tracker->initialize(config);
    return tracker;
}

// Utility function implementations
namespace gpu_utils {

double celsius_to_fahrenheit(double celsius) {
    return celsius * 9.0 / 5.0 + 32.0;
}

std::string format_memory_size(uint64_t bytes) {
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

std::string format_clock_frequency(uint64_t hz) {
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

std::string format_power_watts(double watts) {
    if (watts < 1.0) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << (watts * 1000.0) << " mW";
        return oss.str();
    } else {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << watts << " W";
        return oss.str();
    }
}

double calculate_gpu_efficiency(double utilization, double temperature, double power) {
    if (power <= 0.0) return 0.0;

    // Normalize temperature (lower temp is better up to a point)
    double temp_factor = 1.0;
    if (temperature > 60.0) {
        temp_factor = std::max(0.0, 1.0 - (temperature - 60.0) / 40.0); // Penalty above 60°C
    }

    // Power efficiency (utilization per watt)
    double power_efficiency = utilization / power;

    // Overall efficiency score
    return utilization * temp_factor * std::min(2.0, power_efficiency);
}

double calculate_thermal_headroom(double current_temp, double max_temp) {
    if (max_temp <= current_temp) return 0.0;
    return ((max_temp - current_temp) / max_temp) * 100.0;
}

double calculate_memory_bandwidth_utilization(uint64_t used_bandwidth, uint64_t max_bandwidth) {
    if (max_bandwidth == 0) return 0.0;
    return (static_cast<double>(used_bandwidth) / static_cast<double>(max_bandwidth)) * 100.0;
}

double calculate_audio_processing_score(const GPUUtilizationMetrics::AudioProcessingMetrics& audio) {
    double score = 100.0;

    // Penalize high latency
    if (audio.audio_latency_ms > 0.0) {
        score -= std::min(50.0, audio.audio_latency_ms * 5.0);
    }

    // Penalize dropped samples
    if (audio.audio_samples_processed > 0) {
        double drop_rate = (audio.audio_samples_dropped * 100.0) / audio.audio_samples_processed;
        score -= std::min(30.0, drop_rate);
    }

    // Bonus for low latency
    if (audio.audio_latency_ms < 2.0) {
        score += 10.0;
    }

    // Bonus for no dropped samples
    if (audio.audio_samples_dropped == 0) {
        score += 10.0;
    }

    return std::max(0.0, std::min(100.0, score));
}

bool meets_real_time_audio_requirements(const GPUUtilizationMetrics& metrics) {
    const auto& audio = metrics.audio_metrics;

    // Check latency requirements
    if (audio.audio_latency_ms > 5.0) {
        return false;
    }

    // Check sample drop rate
    if (audio.audio_samples_processed > 0) {
        double drop_rate = (audio.audio_samples_dropped * 100.0) / audio.audio_samples_processed;
        if (drop_rate > 1.0) { // More than 1% dropped samples
            return false;
        }
    }

    // Check GPU utilization
    if (metrics.gpu_utilization_percent > 90.0) {
        return false;
    }

    // Check temperature
    if (metrics.temperature_gpu_celsius > 85.0) {
        return false;
    }

    return true;
}

std::string analyze_audio_bottleneck(const GPUUtilizationMetrics& metrics) {
    const auto& audio = metrics.audio_metrics;

    if (audio.audio_latency_ms > 5.0) {
        return "High audio latency (" + std::to_string(audio.audio_latency_ms) + "ms) detected";
    }

    if (audio.audio_samples_dropped > 0 && audio.audio_samples_processed > 0) {
        double drop_rate = (audio.audio_samples_dropped * 100.0) / audio.samples_processed;
        if (drop_rate > 1.0) {
            return "High sample drop rate (" + std::to_string(drop_rate) + "%) detected";
        }
    }

    if (metrics.gpu_utilization_percent > 90.0) {
        return "High GPU utilization (" + std::to_string(metrics.gpu_utilization_percent) + "%) causing bottleneck";
    }

    if (metrics.memory_utilization_percent > 85.0) {
        return "High memory utilization (" + std::to_string(metrics.memory_utilization_percent) + "%) causing bottleneck";
    }

    if (metrics.temperature_gpu_celsius > 85.0) {
        return "High GPU temperature (" + std::to_string(metrics.temperature_gpu_celsius) + "°C) causing thermal throttling";
    }

    return "No significant bottlenecks detected";
}

bool is_valid_gpu_utilization(double utilization) {
    return utilization >= 0.0 && utilization <= 100.0;
}

bool is_valid_temperature(double temperature_celsius) {
    return temperature_celsius >= -273.15 && temperature_celsius <= 150.0; // Reasonable temperature range
}

bool is_valid_memory_usage(uint64_t used, uint64_t total) {
    return total > 0 && used <= total;
}

bool is_valid_gpu_metrics(const GPUUtilizationMetrics& metrics) {
    return metrics.is_valid &&
           is_valid_gpu_utilization(metrics.gpu_utilization_percent) &&
           is_valid_temperature(metrics.temperature_gpu_celsius) &&
           is_valid_memory_usage(metrics.memory_used_bytes, metrics.memory_total_bytes);
}

} // namespace gpu_utils

} // namespace vortex::hardware