#include "core/processing/adaptive_audio_processor.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <sstream>
#include <random>

namespace vortex::core::processing {

AdaptiveAudioProcessor::AdaptiveAudioProcessor()
    : initialized_(false)
    , adapting_(false)
    , paused_(false)
    , shutdown_requested_(false)
    , current_quality_(ProcessingQuality::MEDIUM)
    , current_mode_(ProcessingMode::ADAPTIVE)
    , current_strategy_(AdaptiveStrategy::BALANCED) {

    // Initialize performance counters
    performance_stats_.total_adaptations = 0;
    performance_stats_.successful_adaptations = 0;
    performance_stats_.avg_adaptation_time_ms = 0.0;
    performance_stats_.max_adaptation_time_ms = 0.0;
    performance_stats_.quality_degradations = 0;
    performance_stats_.quality_improvements = 0;
    performance_stats_.stability_score = 0.0;
    performance_stats_.adaptation_efficiency = 0.0;
    performance_stats_.start_time = std::chrono::steady_clock::now();
}

AdaptiveAudioProcessor::~AdaptiveAudioProcessor() {
    shutdown();
}

bool AdaptiveAudioProcessor::initialize(const AdaptiveParameters& params) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);

    if (initialized_) {
        Logger::warn("AdaptiveAudioProcessor already initialized");
        return true;
    }

    params_ = params;

    Logger::info("Initializing AdaptiveAudioProcessor with {} quality and {} mode",
                 quality_to_string(params.target_quality),
                 mode_to_string(params.processing_mode));

    // Validate parameters
    if (!adaptive_utils::validate_adaptation_parameters(params_)) {
        Logger::error("Invalid adaptive audio processor parameters");
        return false;
    }

    // Initialize quality configurations
    initialize_quality_configurations();

    // Set initial quality and mode
    current_quality_ = params_.target_quality;
    current_mode_ = params_.processing_mode;
    current_strategy_ = params_.strategy;

    // Initialize threads
    try {
        adaptation_thread_ = std::thread(&AdaptiveAudioProcessor::adaptation_thread, this);
        monitoring_thread_ = std::thread(&AdaptiveAudioProcessor::monitoring_thread, this);
        analysis_thread_ = std::thread(&AdaptiveAudioProcessor::analysis_thread, this);

        initialized_ = true;
        Logger::info("AdaptiveAudioProcessor initialized successfully");
        return true;
    }
    catch (const std::exception& e) {
        Logger::error("Failed to initialize AdaptiveAudioProcessor threads: {}", e.what());
        return false;
    }
}

void AdaptiveAudioProcessor::shutdown() {
    if (!initialized_) {
        return;
    }

    Logger::info("Shutting down AdaptiveAudioProcessor");

    // Signal threads to stop
    shutdown_requested_ = true;
    adapting_ = false;

    // Wake up all threads
    // (In real implementation, would have condition variables for each thread)

    // Join threads
    if (adaptation_thread_.joinable()) {
        adaptation_thread_.join();
    }
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }
    if (analysis_thread_.joinable()) {
        analysis_thread_.join();
    }

    // Final performance report
    auto final_stats = get_performance_stats();
    Logger::info("AdaptiveAudioProcessor final report:");
    Logger::info("  Total adaptations: {}", final_stats.total_adaptations);
    Logger::info("  Successful adaptations: {}", final_stats.successful_adaptations);
    Logger::info("  Success rate: {:.2f}%",
                final_stats.total_adaptations > 0 ?
                (double(final_stats.successful_adaptations) / final_stats.total_adaptations) * 100.0 : 0.0);
    Logger::info("  Average adaptation time: {:.2f}ms", final_stats.avg_adaptation_time_ms);
    Logger::info("  Stability score: {:.2f}", final_stats.stability_score);

    initialized_ = false;
    Logger::info("AdaptiveAudioProcessor shutdown complete");
}

bool AdaptiveAudioProcessor::start_adaptation() {
    if (!initialized_) {
        Logger::error("AdaptiveAudioProcessor not initialized");
        return false;
    }

    if (adapting_) {
        Logger::warn("AdaptiveAudioProcessor already adapting");
        return true;
    }

    adapting_ = true;
    paused_ = false;

    Logger::info("Started adaptive audio processing");
    return true;
}

void AdaptiveAudioProcessor::stop_adaptation() {
    adapting_ = false;
    paused_ = false;
    Logger::info("Stopped adaptive audio processing");
}

bool AdaptiveAudioProcessor::pause_adaptation() {
    if (!adapting_ || paused_) {
        return false;
    }

    paused_ = true;
    Logger::info("Paused adaptive audio processing");
    return true;
}

bool AdaptiveAudioProcessor::resume_adaptation() {
    if (!adapting_ || !paused_) {
        return false;
    }

    paused_ = false;
    Logger::info("Resumed adaptive audio processing");
    return true;
}

void AdaptiveAudioProcessor::update_parameters(const AdaptiveParameters& params) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    params_ = params;

    // Reinitialize quality configurations if needed
    initialize_quality_configurations();

    Logger::info("Updated adaptive audio processor parameters");
}

QualityConfiguration AdaptiveAudioProcessor::get_quality_config(ProcessingQuality quality) const {
    auto it = quality_configs_.find(quality);
    if (it != quality_configs_.end()) {
        return it->second;
    }

    // Return default medium quality if not found
    return quality_configs_.at(ProcessingQuality::MEDIUM);
}

SystemPerformanceMetrics AdaptiveAudioProcessor::get_system_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return latest_system_metrics_;
}

ContentAnalysis AdaptiveAudioProcessor::get_content_analysis() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return latest_content_analysis_;
}

AdaptationDecision AdaptiveAudioProcessor::get_last_adaptation_decision() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return last_adaptation_decision_;
}

std::vector<AdaptationDecision> AdaptiveAudioProcessor::get_adaptation_history(size_t count) const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    std::vector<AdaptationDecision> history;
    size_t start_idx = (adaptation_history_.size() > count) ? adaptation_history_.size() - count : 0;

    for (size_t i = start_idx; i < adaptation_history_.size(); ++i) {
        history.push_back(adaptation_history_[i]);
    }

    return history;
}

bool AdaptiveAudioProcessor::set_quality_level(ProcessingQuality quality) {
    ProcessingQuality old_quality = current_quality_;

    if (old_quality == quality) {
        return true; // No change needed
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Create manual adaptation decision
    AdaptationDecision decision;
    decision.should_adapt = true;
    decision.new_quality = quality;
    decision.reason = AdaptationDecision::Reason::USER_REQUEST;
    decision.confidence = 1.0;
    decision.description = "Manual quality change from " + quality_to_string(old_quality) +
                          " to " + quality_to_string(quality);
    decision.decision_time = std::chrono::steady_clock::now();

    // Execute adaptation
    execute_adaptation(decision);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    // Update performance stats
    update_performance_stats(decision, true);

    Logger::info("Manually set quality level to {}", quality_to_string(quality));
    return true;
}

bool AdaptiveAudioProcessor::set_processing_mode(ProcessingMode mode) {
    ProcessingMode old_mode = current_mode_;

    if (old_mode == mode) {
        return true; // No change needed
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Create manual adaptation decision
    AdaptationDecision decision;
    decision.should_adapt = true;
    decision.new_mode = mode;
    decision.reason = AdaptationDecision::Reason::USER_REQUEST;
    decision.confidence = 1.0;
    decision.description = "Manual mode change from " + mode_to_string(old_mode) +
                          " to " + mode_to_string(mode);
    decision.decision_time = std::chrono::steady_clock::now();

    // Execute adaptation
    execute_adaptation(decision);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    // Update performance stats
    update_performance_stats(decision, true);

    Logger::info("Manually set processing mode to {}", mode_to_string(mode));
    return true;
}

bool AdaptiveAudioProcessor::set_adaptive_strategy(AdaptiveStrategy strategy) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    current_strategy_ = strategy;
    params_.strategy = strategy;

    Logger::info("Set adaptive strategy to {}", strategy_to_string(strategy));
    return true;
}

ContentAnalysis AdaptiveAudioProcessor::analyze_audio_content(const float* audio_data,
                                                            size_t num_samples,
                                                            uint32_t sample_rate,
                                                            uint32_t channels) {
    if (!audio_data || num_samples == 0 || sample_rate == 0 || channels == 0) {
        return ContentAnalysis();
    }

    auto analysis = perform_content_analysis(audio_data, num_samples, sample_rate, channels);

    // Store latest analysis
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        latest_content_analysis_ = analysis;
    }

    return analysis;
}

void AdaptiveAudioProcessor::enable_content_aware_processing(bool enabled) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    params_.enable_content_aware_processing = enabled;

    Logger::info("Content-aware processing {}", enabled ? "enabled" : "disabled");
}

bool AdaptiveAudioProcessor::is_content_aware_processing_enabled() const {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    return params_.enable_content_aware_processing;
}

AdaptiveAudioProcessor::AdaptivePerformance AdaptiveAudioProcessor::get_performance_stats() const {
    std::lock_guard<std::mutex> lock(performance_mutex_);
    return performance_stats_;
}

void AdaptiveAudioProcessor::reset_performance_stats() {
    std::lock_guard<std::mutex> lock(performance_mutex_);

    auto now = std::chrono::steady_clock::now();
    performance_stats_.total_adaptations = 0;
    performance_stats_.successful_adaptations = 0;
    performance_stats_.avg_adaptation_time_ms = 0.0;
    performance_stats_.max_adaptation_time_ms = 0.0;
    performance_stats_.quality_degradations = 0;
    performance_stats_.quality_improvements = 0;
    performance_stats_.stability_score = 0.0;
    performance_stats_.adaptation_efficiency = 0.0;
    performance_stats_.start_time = now;

    Logger::info("Reset adaptive audio processor performance statistics");
}

void AdaptiveAudioProcessor::set_hardware_monitor(std::shared_ptr<hardware::HardwareMonitor> monitor) {
    hardware_monitor_ = monitor;

    if (hardware_monitor_) {
        Logger::info("Hardware monitor interface configured");
    }
}

void AdaptiveAudioProcessor::set_streaming_interface(std::shared_ptr<network::RealtimeStreamer> streamer) {
    streamer_ = streamer;

    if (streamer_) {
        Logger::info("Real-time streaming interface configured");
    }
}

void AdaptiveAudioProcessor::set_adaptation_callback(AdaptationCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    adaptation_callback_ = callback;
}

void AdaptiveAudioProcessor::set_performance_callback(PerformanceCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    performance_callback_ = callback;
}

void AdaptiveAudioProcessor::set_quality_change_callback(QualityChangeCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    quality_change_callback_ = callback;
}

void AdaptiveAudioProcessor::enable_predictive_optimization(bool enabled) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    params_.enable_predictive_optimization = enabled;

    Logger::info("Predictive optimization {}", enabled ? "enabled" : "disabled");
}

void AdaptiveAudioProcessor::enable_ml_optimization(bool enabled) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    params_.enable_ml_optimization = enabled;

    Logger::info("Machine learning optimization {}", enabled ? "enabled" : "disabled");
}

bool AdaptiveAudioProcessor::is_predictive_optimization_enabled() const {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    return params_.enable_predictive_optimization;
}

bool AdaptiveAudioProcessor::is_ml_optimization_enabled() const {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    return params_.enable_ml_optimization;
}

std::string AdaptiveAudioProcessor::get_diagnostics_report() const {
    std::ostringstream report;

    report << "=== Adaptive Audio Processor Diagnostics ===\n";
    report << "Initialized: " << (initialized_ ? "Yes" : "No") << "\n";
    report << "Adapting: " << (adapting_.load() ? "Yes" : "No") << "\n";
    report << "Paused: " << (paused_.load() ? "Yes" : "No") << "\n";

    report << "\nCurrent State:\n";
    report << "  Quality: " << quality_to_string(current_quality_.load()) << "\n";
    report << "  Mode: " << mode_to_string(current_mode_.load()) << "\n";
    report << "  Strategy: " << strategy_to_string(current_strategy_.load()) << "\n";

    report << "\nParameters:\n";
    report << "  Max latency: " << params_.max_acceptable_latency_ms << "ms\n";
    report << "  Min performance score: " << params_.min_real_time_score << "\n";
    report << "  Adaptation interval: " << params_.adaptation_interval_ms << "ms\n";

    {
        std::lock_guard<std::mutex> lock(performance_mutex_);
        report << "\nPerformance:\n";
        report << "  Total adaptations: " << performance_stats_.total_adaptations << "\n";
        report << "  Successful: " << performance_stats_.successful_adaptations << "\n";
        report << "  Success rate: ";
        if (performance_stats_.total_adaptations > 0) {
            double rate = (double(performance_stats_.successful_adaptations) / performance_stats_.total_adaptations) * 100.0;
            report << rate << "%\n";
        } else {
            report << "N/A\n";
        }
        report << "  Average adaptation time: " << performance_stats_.avg_adaptation_time_ms << "ms\n";
        report << "  Stability score: " << performance_stats_.stability_score << "\n";
    }

    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        report << "\nLatest System Metrics:\n";
        report << "  CPU utilization: " << latest_system_metrics_.cpu_utilization_percent << "%\n";
        report << "  GPU utilization: " << latest_system_metrics_.gpu_utilization_percent << "%\n";
        report << "  Memory utilization: " << latest_system_metrics_.memory_utilization_percent << "%\n";
        report << "  Temperature: " << latest_system_metrics_.gpu_temperature_celsius << "°C\n";
        report << "  Real-time score: " << latest_system_metrics_.real_time_performance_score << "\n";
    }

    report << "\n=== End Diagnostics ===\n";

    return report.str();
}

bool AdaptiveAudioProcessor::validate_adaptation_setup() const {
    if (!initialized_) {
        return false;
    }

    // Validate parameters
    if (!adaptive_utils::validate_adaptation_parameters(params_)) {
        return false;
    }

    // Validate quality configurations
    for (const auto& [quality, config] : quality_configs_) {
        if (!adaptive_utils::validate_quality_configuration(config)) {
            return false;
        }
    }

    return true;
}

std::vector<std::string> AdaptiveAudioProcessor::test_adaptation_capabilities() const {
    std::vector<std::string> results;

    results.push_back("Testing adaptation capabilities...");

    // Test parameter validation
    bool params_valid = adaptive_utils::validate_adaptation_parameters(params_);
    results.push_back(params_valid ? "✓ Parameters valid" : "✗ Parameters invalid");

    // Test quality configurations
    bool configs_valid = true;
    for (const auto& [quality, config] : quality_configs_) {
        if (!adaptive_utils::validate_quality_configuration(config)) {
            configs_valid = false;
            break;
        }
    }
    results.push_back(configs_valid ? "✓ Quality configurations valid" : "✗ Quality configurations invalid");

    // Test hardware monitoring
    bool hardware_available = (hardware_monitor_ != nullptr);
    results.push_back(hardware_available ? "✓ Hardware monitor available" : "✗ Hardware monitor not available");

    // Test streaming interface
    bool streaming_available = (streamer_ != nullptr);
    results.push_back(streaming_available ? "✓ Streaming interface available" : "✗ Streaming interface not available");

    // Test adaptation decision logic
    SystemPerformanceMetrics test_metrics;
    test_metrics.cpu_utilization_percent = 50.0;
    test_metrics.gpu_utilization_percent = 60.0;
    test_metrics.memory_utilization_percent = 40.0;
    test_metrics.real_time_performance_score = 85.0;
    test_metrics.is_valid = true;

    ContentAnalysis test_content;
    test_content.content_confidence = 0.9;
    test_content.is_stable = true;

    AdaptationDecision test_decision = evaluate_adaptation_needs(test_metrics, test_content);
    results.push_back("✓ Adaptation decision logic functional");

    return results;
}

std::string AdaptiveAudioProcessor::export_adaptation_state() const {
    // Export current state to JSON-like format (simplified)
    std::ostringstream state;

    state << "{\n";
    state << "  \"current_quality\": \"" << quality_to_string(current_quality_.load()) << "\",\n";
    state << "  \"current_mode\": \"" << mode_to_string(current_mode_.load()) << "\",\n";
    state << "  \"current_strategy\": \"" << strategy_to_string(current_strategy_.load()) << "\",\n";
    state << "  \"is_adapting\": " << (adapting_.load() ? "true" : "false") << ",\n";
    state << "  \"is_paused\": " << (paused_.load() ? "true" : "false") << ",\n";

    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        state << "  \"system_metrics\": {\n";
        state << "    \"cpu_utilization\": " << latest_system_metrics_.cpu_utilization_percent << ",\n";
        state << "    \"gpu_utilization\": " << latest_system_metrics_.gpu_utilization_percent << ",\n";
        state << "    \"memory_utilization\": " << latest_system_metrics_.memory_utilization_percent << ",\n";
        state << "    \"real_time_score\": " << latest_system_metrics_.real_time_performance_score << "\n";
        state << "  },\n";
    }

    {
        std::lock_guard<std::mutex> lock(performance_mutex_);
        state << "  \"performance_stats\": {\n";
        state << "    \"total_adaptations\": " << performance_stats_.total_adaptations << ",\n";
        state << "    \"successful_adaptations\": " << performance_stats_.successful_adaptations << ",\n";
        state << "    \"stability_score\": " << performance_stats_.stability_score << "\n";
        state << "  }\n";
    }

    state << "}\n";

    return state.str();
}

bool AdaptiveAudioProcessor::import_adaptation_state(const std::string& state_data) {
    // Simple state import (in real implementation would parse JSON)
    // This is a placeholder that would restore the processor state

    Logger::info("Importing adaptation state ({} bytes)", state_data.size());

    // In a real implementation, this would parse the JSON and restore the state
    // For now, we just log that the import was attempted

    return true;
}

void AdaptiveAudioProcessor::adaptation_thread() {
    Logger::info("Adaptation thread started");

    while (!shutdown_requested_) {
        if (adapting_.load() && !paused_.load()) {
            auto start_time = std::chrono::high_resolution_clock::now();

            // Get current system metrics
            SystemPerformanceMetrics current_metrics;
            if (hardware_monitor_) {
                auto hw_metrics = hardware_monitor_->get_latest_metrics();
                if (hw_metrics.is_valid) {
                    current_metrics.cpu_utilization_percent = hw_metrics.cpu_utilization_percent;
                    current_metrics.gpu_utilization_percent = hw_metrics.gpu_metrics.empty() ? 0.0 :
                        hw_metrics.gpu_metrics[0].gpu_utilization_percent;
                    current_metrics.memory_utilization_percent = hw_metrics.memory_utilization_percent;
                    current_metrics.gpu_temperature_celsius = hw_metrics.gpu_metrics.empty() ? 0.0 :
                        hw_metrics.gpu_metrics[0].temperature_celsius;
                    current_metrics.real_time_performance_score = hw_metrics.performance_metrics.real_time_performance_score;
                    current_metrics.audio_latency_ms = hw_metrics.audio_metrics.processing_latency_ms;
                    current_metrics.is_valid = true;
                }
            }

            // Get current content analysis
            ContentAnalysis current_content;
            {
                std::lock_guard<std::mutex> lock(metrics_mutex_);
                current_content = latest_content_analysis_;
            }

            // Evaluate adaptation needs
            AdaptationDecision decision = evaluate_adaptation_needs(current_metrics, current_content);

            // Execute adaptation if needed
            if (decision.should_adapt) {
                execute_adaptation(decision);
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            // Sleep for adaptation interval
            std::this_thread::sleep_for(std::chrono::milliseconds(params_.adaptation_interval_ms));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    Logger::info("Adaptation thread stopped");
}

void AdaptiveAudioProcessor::monitoring_thread() {
    Logger::info("Monitoring thread started");

    while (!shutdown_requested_) {
        if (initialized_) {
            monitor_system_performance();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    Logger::info("Monitoring thread stopped");
}

void AdaptiveAudioProcessor::analysis_thread() {
    Logger::info("Analysis thread started");

    while (!shutdown_requested_) {
        if (initialized_ && params_.enable_content_aware_processing) {
            // Perform content analysis on available audio data
            // This would be triggered by audio processing pipeline
            // For now, we just simulate the timing
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    }

    Logger::info("Analysis thread stopped");
}

AdaptationDecision AdaptiveAudioProcessor::evaluate_adaptation_needs(
    const SystemPerformanceMetrics& metrics,
    const ContentAnalysis& content) {

    AdaptationDecision decision;
    decision.should_adapt = false;
    decision.new_quality = current_quality_;
    decision.new_mode = current_mode_;
    decision.confidence = 0.0;
    decision.decision_time = std::chrono::steady_clock::now();

    // Check CPU utilization
    if (metrics.cpu_utilization_percent > params_.fallback_thresholds.high_cpu_threshold) {
        auto cpu_decision = create_cpu_based_adaptation(metrics.cpu_utilization_percent);
        if (cpu_decision.should_adapt && cpu_decision.confidence > decision.confidence) {
            decision = cpu_decision;
        }
    }

    // Check GPU utilization
    if (metrics.gpu_utilization_percent > params_.fallback_thresholds.high_gpu_threshold) {
        auto gpu_decision = create_gpu_based_adaptation(metrics.gpu_utilization_percent);
        if (gpu_decision.should_adapt && gpu_decision.confidence > decision.confidence) {
            decision = gpu_decision;
        }
    }

    // Check memory utilization
    if (metrics.memory_utilization_percent > params_.fallback_thresholds.high_memory_threshold) {
        auto mem_decision = create_memory_based_adaptation(metrics.memory_utilization_percent);
        if (mem_decision.should_adapt && mem_decision.confidence > decision.confidence) {
            decision = mem_decision;
        }
    }

    // Check temperature
    if (metrics.gpu_temperature_celsius > params_.fallback_thresholds.high_temp_threshold) {
        auto temp_decision = create_temperature_based_adaptation(metrics.gpu_temperature_celsius);
        if (temp_decision.should_adapt && temp_decision.confidence > decision.confidence) {
            decision = temp_decision;
        }
    }

    // Check latency
    if (metrics.audio_latency_ms > params_.fallback_thresholds.high_latency_threshold) {
        auto latency_decision = create_latency_based_adaptation(metrics.audio_latency_ms);
        if (latency_decision.should_adapt && latency_decision.confidence > decision.confidence) {
            decision = latency_decision;
        }
    }

    // Check performance score
    if (metrics.real_time_performance_score < params_.fallback_thresholds.low_performance_threshold) {
        auto perf_decision = create_predictive_adaptation();
        if (perf_decision.should_adapt && perf_decision.confidence > decision.confidence) {
            decision = perf_decision;
        }
    }

    // Content-based adaptation
    if (params_.enable_content_aware_processing && content.is_stable) {
        auto content_decision = create_content_based_adaptation(content);
        if (content_decision.should_adapt && content_decision.confidence > decision.confidence) {
            decision = content_decision;
        }
    }

    // Predictive adaptation
    if (params_.enable_predictive_optimization && !decision.should_adapt) {
        auto predictive_decision = create_predictive_adaptation();
        if (predictive_decision.should_adapt && predictive_decision.confidence > decision.confidence) {
            decision = predictive_decision;
        }
    }

    return decision;
}

AdaptationDecision AdaptiveAudioProcessor::create_cpu_based_adaptation(double cpu_utilization) {
    AdaptationDecision decision;
    decision.reason = AdaptationDecision::Reason::HIGH_CPU_LOAD;

    if (cpu_utilization > 95.0) {
        decision.new_quality = ProcessingQuality::MINIMAL;
        decision.confidence = 0.9;
        decision.expected_cpu_reduction = 40.0;
        decision.expected_quality_impact = -50.0;
    } else if (cpu_utilization > 90.0) {
        decision.new_quality = ProcessingQuality::LOW;
        decision.confidence = 0.8;
        decision.expected_cpu_reduction = 30.0;
        decision.expected_quality_impact = -35.0;
    } else if (cpu_utilization > 85.0) {
        decision.new_quality = ProcessingQuality::MEDIUM;
        decision.confidence = 0.6;
        decision.expected_cpu_reduction = 20.0;
        decision.expected_quality_impact = -20.0;
    } else if (cpu_utilization > 80.0) {
        decision.new_quality = ProcessingQuality::HIGH;
        decision.confidence = 0.4;
        decision.expected_cpu_reduction = 10.0;
        decision.expected_quality_impact = -10.0;
    } else {
        decision.should_adapt = false;
        return decision;
    }

    decision.should_adapt = true;
    decision.description = "CPU utilization at " + std::to_string(cpu_utilization) + "% - reducing quality";
    decision.expected_latency_improvement = decision.expected_cpu_reduction * 0.3;

    return decision;
}

AdaptationDecision AdaptiveAudioProcessor::create_gpu_based_adaptation(double gpu_utilization) {
    AdaptationDecision decision;
    decision.reason = AdaptationDecision::Reason::HIGH_GPU_LOAD;

    if (gpu_utilization > 95.0) {
        decision.new_quality = ProcessingQuality::LOW;
        decision.confidence = 0.9;
        decision.expected_gpu_reduction = 50.0;
        decision.expected_quality_impact = -40.0;
    } else if (gpu_utilization > 90.0) {
        decision.new_quality = ProcessingQuality::MEDIUM;
        decision.confidence = 0.7;
        decision.expected_gpu_reduction = 35.0;
        decision.expected_quality_impact = -25.0;
    } else if (gpu_utilization > 85.0) {
        decision.new_mode = ProcessingMode::INTERACTIVE;
        decision.confidence = 0.5;
        decision.expected_gpu_reduction = 20.0;
        decision.expected_quality_impact = -15.0;
    } else {
        decision.should_adapt = false;
        return decision;
    }

    decision.should_adapt = true;
    decision.description = "GPU utilization at " + std::to_string(gpu_utilization) + "% - optimizing processing";
    decision.expected_latency_improvement = decision.expected_gpu_reduction * 0.2;

    return decision;
}

AdaptationDecision AdaptiveAudioProcessor::create_memory_based_adaptation(double memory_utilization) {
    AdaptationDecision decision;
    decision.reason = AdaptationDecision::Reason::HIGH_MEMORY_USAGE;

    if (memory_utilization > 90.0) {
        decision.new_quality = ProcessingQuality::LOW;
        decision.confidence = 0.8;
        decision.expected_quality_impact = -30.0;
    } else if (memory_utilization > 85.0) {
        decision.new_mode = ProcessingMode::INTERACTIVE;
        decision.confidence = 0.6;
        decision.expected_quality_impact = -20.0;
    } else {
        decision.should_adapt = false;
        return decision;
    }

    decision.should_adapt = true;
    decision.description = "Memory utilization at " + std::to_string(memory_utilization) + "% - optimizing memory usage";

    return decision;
}

AdaptationDecision AdaptiveAudioProcessor::create_temperature_based_adaptation(double temperature) {
    AdaptationDecision decision;
    decision.reason = AdaptationDecision::Reason::HIGH_TEMPERATURE;

    if (temperature > 85.0) {
        decision.new_quality = ProcessingQuality::MINIMAL;
        decision.new_mode = ProcessingMode::POWER_EFFICIENT;
        decision.confidence = 0.9;
        decision.expected_quality_impact = -45.0;
    } else if (temperature > 80.0) {
        decision.new_quality = ProcessingQuality::LOW;
        decision.confidence = 0.7;
        decision.expected_quality_impact = -35.0;
    } else {
        decision.should_adapt = false;
        return decision;
    }

    decision.should_adapt = true;
    decision.description = "GPU temperature at " + std::to_string(temperature) + "°C - thermal management";
    decision.expected_latency_improvement = 5.0; // May improve latency due to thermal throttling

    return decision;
}

AdaptationDecision AdaptiveAudioProcessor::create_latency_based_adaptation(double latency) {
    AdaptationDecision decision;
    decision.reason = AdaptationDecision::Reason::HIGH_LATENCY;

    if (latency > 20.0) {
        decision.new_quality = ProcessingQuality::LOW;
        decision.new_mode = ProcessingMode::REAL_TIME;
        decision.confidence = 0.9;
        decision.expected_latency_improvement = 10.0;
        decision.expected_quality_impact = -30.0;
    } else if (latency > 16.0) {
        decision.new_quality = ProcessingQuality::MEDIUM;
        decision.new_mode = ProcessingMode::REAL_TIME;
        decision.confidence = 0.7;
        decision.expected_latency_improvement = 5.0;
        decision.expected_quality_impact = -20.0;
    } else {
        decision.should_adapt = false;
        return decision;
    }

    decision.should_adapt = true;
    decision.description = "Audio latency at " + std::to_string(latency) + "ms - latency optimization";

    return decision;
}

AdaptationDecision AdaptiveAudioProcessor::create_content_based_adaptation(const ContentAnalysis& content) {
    AdaptationDecision decision;
    decision.reason = AdaptationDecision::Reason::CONTENT_CHANGE;

    // Adapt based on content type
    switch (content.primary_content_type) {
        case ContentType::SPEECH:
            // Speech needs clarity over quality
            if (current_quality_ > ProcessingQuality::HIGH) {
                decision.new_quality = ProcessingQuality::HIGH;
                decision.confidence = 0.7;
                decision.expected_quality_impact = -15.0;
            }
            break;

        case ContentType::MUSIC:
            // Music benefits from higher quality
            if (current_quality_ < ProcessingQuality::HIGH && content.quality_requirements > 0.7) {
                decision.new_quality = ProcessingQuality::HIGH;
                decision.confidence = 0.6;
                decision.expected_quality_impact = 15.0;
            }
            break;

        case ContentType::SIGNAL:
            // Signal processing needs precision
            if (current_quality_ < ProcessingQuality::ULTRA_HIGH) {
                decision.new_quality = ProcessingQuality::ULTRA_HIGH;
                decision.confidence = 0.8;
                decision.expected_quality_impact = 25.0;
            }
            break;

        case ContentType::NOISE:
            // Noise can use lower quality
            if (current_quality_ > ProcessingQuality::MEDIUM) {
                decision.new_quality = ProcessingQuality::MEDIUM;
                decision.confidence = 0.5;
                decision.expected_quality_impact = -10.0;
            }
            break;

        default:
            decision.should_adapt = false;
            return decision;
    }

    decision.should_adapt = true;
    decision.description = "Content type: " + std::to_string(static_cast<int>(content.primary_content_type)) +
                          " - content-aware adaptation";

    return decision;
}

AdaptationDecision AdaptiveAudioProcessor::create_predictive_adaptation() {
    AdaptationDecision decision;
    decision.reason = AdaptationDecision::Reason::PREDICTIVE_OPTIMIZATION;

    // Get current performance history
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    if (performance_history_.size() < 10) {
        decision.should_adapt = false;
        return decision;
    }

    // Simple trend analysis (in real implementation would use more sophisticated prediction)
    double avg_cpu = 0.0, avg_gpu = 0.0;
    for (const auto& metrics : performance_history_) {
        avg_cpu += metrics.cpu_utilization_percent;
        avg_gpu += metrics.gpu_utilization_percent;
    }
    avg_cpu /= performance_history_.size();
    avg_gpu /= performance_history_.size();

    // Predict if we'll exceed thresholds soon
    if (avg_cpu > 75.0 || avg_gpu > 75.0) {
        if (current_quality_ > ProcessingQuality::MEDIUM) {
            decision.new_quality = ProcessingQuality::MEDIUM;
            decision.confidence = 0.4;
            decision.expected_cpu_reduction = 15.0;
            decision.expected_gpu_reduction = 20.0;
            decision.expected_quality_impact = -25.0;
        }
    }

    if (decision.should_adapt) {
        decision.description = "Predictive optimization - trending toward high resource utilization";
    }

    return decision;
}

void AdaptiveAudioProcessor::initialize_quality_configurations() {
    // Initialize quality configurations for all levels
    quality_configs_[ProcessingQuality::ULTRA_HIGH] = create_quality_config(ProcessingQuality::ULTRA_HIGH);
    quality_configs_[ProcessingQuality::HIGH] = create_quality_config(ProcessingQuality::HIGH);
    quality_configs_[ProcessingQuality::MEDIUM] = create_quality_config(ProcessingQuality::MEDIUM);
    quality_configs_[ProcessingQuality::LOW] = create_quality_config(ProcessingQuality::LOW);
    quality_configs_[ProcessingQuality::MINIMAL] = create_quality_config(ProcessingQuality::MINIMAL);
}

QualityConfiguration AdaptiveAudioProcessor::create_quality_config(ProcessingQuality quality) {
    QualityConfiguration config;
    config.quality_level = quality;

    switch (quality) {
        case ProcessingQuality::ULTRA_HIGH:
            config.fft_size = 16384;
            config.overlap_factor = 8;
            config.spectrum_resolution = 4096;
            config.waveform_resolution = 2000;
            config.vu_meter_update_rate = 120;
            config.enable_high_precision = true;
            config.enable_advanced_filtering = true;
            config.enable_noise_reduction = true;
            config.enable_dynamic_range_compression = true;
            config.enable_harmonic_analysis = true;
            config.enable_spectral_enhancement = true;
            break;

        case ProcessingQuality::HIGH:
            config.fft_size = 8192;
            config.overlap_factor = 6;
            config.spectrum_resolution = 2048;
            config.waveform_resolution = 1500;
            config.vu_meter_update_rate = 90;
            config.enable_high_precision = true;
            config.enable_advanced_filtering = true;
            config.enable_noise_reduction = true;
            config.enable_dynamic_range_compression = false;
            config.enable_harmonic_analysis = true;
            config.enable_spectral_enhancement = false;
            break;

        case ProcessingQuality::MEDIUM:
            config.fft_size = 4096;
            config.overlap_factor = 4;
            config.spectrum_resolution = 1024;
            config.waveform_resolution = 1000;
            config.vu_meter_update_rate = 60;
            config.enable_high_precision = false;
            config.enable_advanced_filtering = false;
            config.enable_noise_reduction = false;
            config.enable_dynamic_range_compression = false;
            config.enable_harmonic_analysis = false;
            config.enable_spectral_enhancement = false;
            break;

        case ProcessingQuality::LOW:
            config.fft_size = 2048;
            config.overlap_factor = 2;
            config.spectrum_resolution = 512;
            config.waveform_resolution = 500;
            config.vu_meter_update_rate = 30;
            config.enable_high_precision = false;
            config.enable_advanced_filtering = false;
            config.enable_noise_reduction = false;
            config.enable_dynamic_range_compression = false;
            config.enable_harmonic_analysis = false;
            config.enable_spectral_enhancement = false;
            config.prefer_gpu_processing = false;
            break;

        case ProcessingQuality::MINIMAL:
            config.fft_size = 1024;
            config.overlap_factor = 1;
            config.spectrum_resolution = 256;
            config.waveform_resolution = 200;
            config.vu_meter_update_rate = 15;
            config.enable_high_precision = false;
            config.enable_advanced_filtering = false;
            config.enable_noise_reduction = false;
            config.enable_dynamic_range_compression = false;
            config.enable_harmonic_analysis = false;
            config.enable_spectral_enhancement = false;
            config.prefer_gpu_processing = false;
            config.enable_multi_threading = false;
            break;
    }

    return config;
}

ContentAnalysis AdaptiveAudioProcessor::perform_content_analysis(const float* audio_data,
                                                               size_t num_samples,
                                                               uint32_t sample_rate,
                                                               uint32_t channels) {
    ContentAnalysis analysis;

    if (!audio_data || num_samples == 0) {
        return analysis;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Convert to mono for analysis
    std::vector<float> mono_audio(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        if (channels == 1) {
            mono_audio[i] = audio_data[i];
        } else {
            float sum = 0.0f;
            for (uint32_t ch = 0; ch < channels; ++ch) {
                sum += audio_data[i * channels + ch];
            }
            mono_audio[i] = sum / channels;
        }
    }

    // Calculate basic audio metrics
    float sum_squares = 0.0f;
    float peak = 0.0f;
    uint32_t zero_crossings = 0;

    for (size_t i = 0; i < num_samples; ++i) {
        float sample = mono_audio[i];
        sum_squares += sample * sample;
        peak = std::max(peak, std::abs(sample));

        if (i > 0 && ((mono_audio[i-1] >= 0.0f && sample < 0.0f) ||
                      (mono_audio[i-1] < 0.0f && sample >= 0.0f))) {
            zero_crossings++;
        }
    }

    analysis.rms_level = std::sqrt(sum_squares / num_samples);
    analysis.peak_level = peak;
    analysis.crest_factor = (analysis.rms_level > 0.0f) ? (20.0f * std::log10(peak / analysis.rms_level)) : 0.0f;
    analysis.zero_crossing_rate = static_cast<double>(zero_crossings) / num_samples * sample_rate;

    // Perform simple FFT for spectral analysis
    size_t fft_size = std::min(num_samples, static_cast<size_t>(4096));
    fft_size = 1 << static_cast<size_t>(std::ceil(std::log2(fft_size))); // Next power of 2

    std::vector<float> spectrum(fft_size / 2 + 1);
    // In real implementation, would use FFT library
    // For now, we'll use a simple placeholder

    // Calculate spectral characteristics
    analysis.spectral_centroid = calculate_spectral_centroid(spectrum.data(), spectrum.size(), sample_rate);
    analysis.spectral_bandwidth = calculate_spectral_bandwidth(spectrum.data(), spectrum.size(),
                                                              sample_rate, analysis.spectral_centroid);
    analysis.spectral_rolloff = calculate_spectral_rolloff(spectrum.data(), spectrum.size(),
                                                          sample_rate, 0.85);

    // Determine content type
    analysis.primary_content_type = determine_content_type(analysis);
    analysis.content_confidence = 0.8; // Simplified confidence

    // Calculate content complexity and requirements
    analysis.content_complexity = std::min(1.0, analysis.spectral_bandwidth / 2000.0);
    analysis.processing_difficulty = analysis.content_complexity * (1.0 + analysis.harmonic_content);
    analysis.quality_requirements = analysis.processing_difficulty;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    analysis.duration_microseconds = duration.count();

    analysis.analysis_time = std::chrono::steady_clock::now();
    analysis.is_stable = true; // Simplified

    return analysis;
}

ContentType AdaptiveAudioProcessor::determine_content_type(const ContentAnalysis& analysis) {
    // Simple content type determination based on audio characteristics
    if (analysis.zero_crossing_rate > 5000.0 && analysis.spectral_centroid > 2000.0) {
        return ContentType::SPEECH;
    } else if (analysis.harmonic_content > 0.7 && analysis.spectral_centroid < 2000.0) {
        return ContentType::MUSIC;
    } else if (analysis.signal_to_noise_ratio < 10.0) {
        return ContentType::NOISE;
    } else if (analysis.harmonic_content > 0.9) {
        return ContentType::SIGNAL;
    } else {
        return ContentType::MIXED;
    }
}

double AdaptiveAudioProcessor::calculate_spectral_centroid(const float* spectrum, size_t size,
                                                         uint32_t sample_rate) {
    double weighted_sum = 0.0;
    double magnitude_sum = 0.0;

    for (size_t i = 0; i < size; ++i) {
        double freq = (static_cast<double>(i) / size) * (sample_rate / 2.0);
        weighted_sum += freq * spectrum[i];
        magnitude_sum += spectrum[i];
    }

    return magnitude_sum > 0.0 ? weighted_sum / magnitude_sum : 0.0;
}

double AdaptiveAudioProcessor::calculate_spectral_bandwidth(const float* spectrum, size_t size,
                                                          uint32_t sample_rate, double centroid) {
    double weighted_sum = 0.0;
    double magnitude_sum = 0.0;

    for (size_t i = 0; i < size; ++i) {
        double freq = (static_cast<double>(i) / size) * (sample_rate / 2.0);
        double deviation = freq - centroid;
        weighted_sum += deviation * deviation * spectrum[i];
        magnitude_sum += spectrum[i];
    }

    return magnitude_sum > 0.0 ? std::sqrt(weighted_sum / magnitude_sum) : 0.0;
}

double AdaptiveAudioProcessor::calculate_spectral_rolloff(const float* spectrum, size_t size,
                                                        uint32_t sample_rate, double threshold) {
    double total_energy = 0.0;
    for (size_t i = 0; i < size; ++i) {
        total_energy += spectrum[i] * spectrum[i];
    }

    double threshold_energy = total_energy * threshold;
    double cumulative_energy = 0.0;

    for (size_t i = 0; i < size; ++i) {
        cumulative_energy += spectrum[i] * spectrum[i];
        if (cumulative_energy >= threshold_energy) {
            return (static_cast<double>(i) / size) * (sample_rate / 2.0);
        }
    }

    return sample_rate / 2.0;
}

void AdaptiveAudioProcessor::execute_adaptation(const AdaptationDecision& decision) {
    if (!decision.should_adapt) {
        return;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Store old values
    ProcessingQuality old_quality = current_quality_;
    ProcessingMode old_mode = current_mode_;

    // Apply quality changes
    if (decision.new_quality != old_quality) {
        transition_quality(old_quality, decision.new_quality);
        current_quality_ = decision.new_quality;
    }

    // Apply mode changes
    if (decision.new_mode != old_mode) {
        transition_mode(old_mode, decision.new_mode);
        current_mode_ = decision.new_mode;
    }

    // Update performance stats
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        last_adaptation_decision_ = decision;
        adaptation_history_.push_back(decision);
        if (adaptation_history_.size() > 100) {
            adaptation_history_.pop_front();
        }
    }

    // Update performance tracking
    update_performance_stats(decision, true);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    Logger::info("Executed adaptation: {} -> {} ({}ms)",
                quality_to_string(old_quality), quality_to_string(decision.new_quality),
                duration.count() / 1000.0);

    // Notify callbacks
    notify_adaptation(decision);
}

void AdaptiveAudioProcessor::transition_quality(ProcessingQuality old_quality, ProcessingQuality new_quality) {
    // Apply new quality configuration
    QualityConfiguration config = get_quality_config(new_quality);
    apply_quality_configuration(config);

    // Trigger quality change callback
    {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        if (quality_change_callback_) {
            try {
                quality_change_callback_(old_quality, new_quality);
            } catch (const std::exception& e) {
                Logger::error("Quality change callback error: {}", e.what());
            }
        }
    }

    Logger::info("Quality transitioned from {} to {}",
                quality_to_string(old_quality), quality_to_string(new_quality));
}

void AdaptiveAudioProcessor::transition_mode(ProcessingMode old_mode, ProcessingMode new_mode) {
    // Apply mode-specific changes
    switch (new_mode) {
        case ProcessingMode::REAL_TIME:
            // Prioritize latency
            params_.max_acceptable_latency_ms = 10.0;
            break;
        case ProcessingMode::OFFLINE:
            // Prioritize quality
            params_.max_acceptable_latency_ms = 100.0;
            break;
        case ProcessingMode::INTERACTIVE:
            // Balance latency and quality
            params_.max_acceptable_latency_ms = 20.0;
            break;
        case ProcessingMode::STREAMING:
            // Prioritize continuity
            params_.max_acceptable_latency_ms = 30.0;
            break;
        default:
            // Use default values
            params_.max_acceptable_latency_ms = 16.0;
            break;
    }

    Logger::info("Processing mode transitioned from {} to {}",
                mode_to_string(old_mode), mode_to_string(new_mode));
}

void AdaptiveAudioProcessor::apply_quality_configuration(const QualityConfiguration& config) {
    // In a real implementation, this would apply the quality configuration
    // to the actual audio processing components

    Logger::debug("Applied quality configuration for {} quality",
                quality_to_string(config.quality_level));
}

void AdaptiveAudioProcessor::notify_adaptation(const AdaptationDecision& decision) {
    // Trigger adaptation callback
    {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        if (adaptation_callback_) {
            try {
                adaptation_callback_(decision);
            } catch (const std::exception& e) {
                Logger::error("Adaptation callback error: {}", e.what());
            }
        }
    }

    // Stream adaptation decision if streaming is enabled
    if (streamer_ && decision.should_adapt) {
        try {
            // Serialize adaptation decision and stream
            std::string adaptation_data = "adaptation:" + decision.description;
            streamer_->send_text_data(adaptation_data, "adaptation_event");
        } catch (const std::exception& e) {
            Logger::error("Failed to stream adaptation event: {}", e.what());
        }
    }
}

void AdaptiveAudioProcessor::monitor_system_performance() {
    if (!hardware_monitor_) {
        return;
    }

    // Get current system metrics
    auto hw_metrics = hardware_monitor_->get_latest_metrics();

    if (hw_metrics.is_valid) {
        SystemPerformanceMetrics metrics;
        metrics.cpu_utilization_percent = hw_metrics.cpu_utilization_percent;
        metrics.gpu_utilization_percent = hw_metrics.gpu_metrics.empty() ? 0.0 :
            hw_metrics.gpu_metrics[0].gpu_utilization_percent;
        metrics.memory_utilization_percent = hw_metrics.memory_utilization_percent;
        metrics.gpu_temperature_celsius = hw_metrics.gpu_metrics.empty() ? 0.0 :
            hw_metrics.gpu_metrics[0].temperature_celsius;
        metrics.real_time_performance_score = hw_metrics.performance_metrics.real_time_performance_score;
        metrics.audio_latency_ms = hw_metrics.audio_metrics.processing_latency_ms;
        metrics.timestamp_microseconds = get_current_timestamp_microseconds();
        metrics.is_valid = true;

        // Store latest metrics
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            latest_system_metrics_ = metrics;
            performance_history_.push_back(metrics);
            if (performance_history_.size() > 100) {
                performance_history_.pop_front();
            }
        }

        // Trigger performance callback
        {
            std::lock_guard<std::mutex> lock(callbacks_mutex_);
            if (performance_callback_) {
                try {
                    performance_callback_(metrics);
                } catch (const std::exception& e) {
                    Logger::error("Performance callback error: {}", e.what());
                }
            }
        }
    }
}

uint64_t AdaptiveAudioProcessor::get_current_timestamp_microseconds() const {
    auto now = std::chrono::steady_clock::now();
    auto epoch = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(epoch).count();
}

void AdaptiveAudioProcessor::update_performance_stats(const AdaptationDecision& decision, bool success) {
    std::lock_guard<std::mutex> lock(performance_mutex_);

    performance_stats_.total_adaptations++;
    if (success) {
        performance_stats_.successful_adaptations++;
    }

    // Update quality change counters
    if (decision.new_quality < current_quality_) {
        performance_stats_.quality_degradations++;
    } else if (decision.new_quality > current_quality_) {
        performance_stats_.quality_improvements++;
    }

    // Calculate stability score from adaptation history
    {
        std::lock_guard<std::mutex> metrics_lock(metrics_mutex_);
        if (adaptation_history_.size() > 10) {
            // Calculate how stable the system has been
            size_t stable_adaptations = 0;
            for (const auto& hist_decision : adaptation_history_) {
                if (hist_decision.confidence > 0.7) {
                    stable_adaptations++;
                }
            }
            performance_stats_.stability_score = static_cast<double>(stable_adaptations) / adaptation_history_.size();
        }
    }

    // Calculate adaptation efficiency
    if (performance_stats_.total_adaptations > 0) {
        performance_stats_.adaptation_efficiency =
            static_cast<double>(performance_stats_.successful_adaptations) / performance_stats_.total_adaptations;
    }
}

std::string AdaptiveAudioProcessor::quality_to_string(ProcessingQuality quality) const {
    switch (quality) {
        case ProcessingQuality::ULTRA_HIGH: return "ULTRA_HIGH";
        case ProcessingQuality::HIGH: return "HIGH";
        case ProcessingQuality::MEDIUM: return "MEDIUM";
        case ProcessingQuality::LOW: return "LOW";
        case ProcessingQuality::MINIMAL: return "MINIMAL";
        case ProcessingQuality::AUTO: return "AUTO";
        default: return "UNKNOWN";
    }
}

std::string AdaptiveAudioProcessor::mode_to_string(ProcessingMode mode) const {
    switch (mode) {
        case ProcessingMode::REAL_TIME: return "REAL_TIME";
        case ProcessingMode::OFFLINE: return "OFFLINE";
        case ProcessingMode::INTERACTIVE: return "INTERACTIVE";
        case ProcessingMode::BATCH: return "BATCH";
        case ProcessingMode::STREAMING: return "STREAMING";
        case ProcessingMode::ADAPTIVE: return "ADAPTIVE";
        default: return "UNKNOWN";
    }
}

std::string AdaptiveAudioProcessor::strategy_to_string(AdaptiveStrategy strategy) const {
    switch (strategy) {
        case AdaptiveStrategy::CONSERVATIVE: return "CONSERVATIVE";
        case AdaptiveStrategy::BALANCED: return "BALANCED";
        case AdaptiveStrategy::AGGRESSIVE: return "AGGRESSIVE";
        case AdaptiveStrategy::POWER_EFFICIENT: return "POWER_EFFICIENT";
        case AdaptiveStrategy::QUALITY_FOCUSED: return "QUALITY_FOCUSED";
        default: return "UNKNOWN";
    }
}

// Factory implementations
std::unique_ptr<AdaptiveAudioProcessor> AdaptiveAudioProcessorFactory::create_default() {
    auto params = AdaptiveParameters();
    params.target_quality = ProcessingQuality::MEDIUM;
    params.strategy = AdaptiveStrategy::BALANCED;

    auto processor = std::make_unique<AdaptiveAudioProcessor>();
    if (!processor->initialize(params)) {
        return nullptr;
    }

    return processor;
}

std::unique_ptr<AdaptiveAudioProcessor> AdaptiveAudioProcessorFactory::create_high_performance() {
    auto params = AdaptiveParameters();
    params.target_quality = ProcessingQuality::HIGH;
    params.strategy = AdaptiveStrategy::AGGRESSIVE;
    params.max_cpu_utilization_percent = 90.0;
    params.max_gpu_utilization_percent = 95.0;

    auto processor = std::make_unique<AdaptiveAudioProcessor>();
    if (!processor->initialize(params)) {
        return nullptr;
    }

    return processor;
}

std::unique_ptr<AdaptiveAudioProcessor> AdaptiveAudioProcessorFactory::create_power_efficient() {
    auto params = AdaptiveParameters();
    params.target_quality = ProcessingQuality::MEDIUM;
    params.strategy = AdaptiveStrategy::POWER_EFFICIENT;
    params.max_temperature_celsius = 70.0;
    params.enable_power_management = true;

    auto processor = std::make_unique<AdaptiveAudioProcessor>();
    if (!processor->initialize(params)) {
        return nullptr;
    }

    return processor;
}

std::unique_ptr<AdaptiveAudioProcessor> AdaptiveAudioProcessorFactory::create_quality_focused() {
    auto params = AdaptiveParameters();
    params.target_quality = ProcessingQuality::HIGH;
    params.strategy = AdaptiveStrategy::QUALITY_FOCUSED;
    params.min_real_time_score = 70.0; // Allow some flexibility for quality

    auto processor = std::make_unique<AdaptiveAudioProcessor>();
    if (!processor->initialize(params)) {
        return nullptr;
    }

    return processor;
}

std::unique_ptr<AdaptiveAudioProcessor> AdaptiveAudioProcessorFactory::create_balanced() {
    auto params = AdaptiveParameters();
    params.target_quality = ProcessingQuality::MEDIUM;
    params.strategy = AdaptiveStrategy::BALANCED;

    auto processor = std::make_unique<AdaptiveAudioProcessor>();
    if (!processor->initialize(params)) {
        return nullptr;
    }

    return processor;
}

// Utility namespace implementations
namespace adaptive_utils {

ProcessingQuality string_to_quality(const std::string& quality_str) {
    if (quality_str == "ULTRA_HIGH") return ProcessingQuality::ULTRA_HIGH;
    if (quality_str == "HIGH") return ProcessingQuality::HIGH;
    if (quality_str == "MEDIUM") return ProcessingQuality::MEDIUM;
    if (quality_str == "LOW") return ProcessingQuality::LOW;
    if (quality_str == "MINIMAL") return ProcessingQuality::MINIMAL;
    if (quality_str == "AUTO") return ProcessingQuality::AUTO;
    return ProcessingQuality::MEDIUM; // Default
}

std::string quality_to_string(ProcessingQuality quality) {
    switch (quality) {
        case ProcessingQuality::ULTRA_HIGH: return "ULTRA_HIGH";
        case ProcessingQuality::HIGH: return "HIGH";
        case ProcessingQuality::MEDIUM: return "MEDIUM";
        case ProcessingQuality::LOW: return "LOW";
        case ProcessingQuality::MINIMAL: return "MINIMAL";
        case ProcessingQuality::AUTO: return "AUTO";
        default: return "UNKNOWN";
    }
}

ProcessingMode string_to_mode(const std::string& mode_str) {
    if (mode_str == "REAL_TIME") return ProcessingMode::REAL_TIME;
    if (mode_str == "OFFLINE") return ProcessingMode::OFFLINE;
    if (mode_str == "INTERACTIVE") return ProcessingMode::INTERACTIVE;
    if (mode_str == "BATCH") return ProcessingMode::BATCH;
    if (mode_str == "STREAMING") return ProcessingMode::STREAMING;
    if (mode_str == "ADAPTIVE") return ProcessingMode::ADAPTIVE;
    return ProcessingMode::ADAPTIVE; // Default
}

std::string mode_to_string(ProcessingMode mode) {
    switch (mode) {
        case ProcessingMode::REAL_TIME: return "REAL_TIME";
        case ProcessingMode::OFFLINE: return "OFFLINE";
        case ProcessingMode::INTERACTIVE: return "INTERACTIVE";
        case ProcessingMode::BATCH: return "BATCH";
        case ProcessingMode::STREAMING: return "STREAMING";
        case ProcessingMode::ADAPTIVE: return "ADAPTIVE";
        default: return "UNKNOWN";
    }
}

AdaptiveStrategy string_to_strategy(const std::string& strategy_str) {
    if (strategy_str == "CONSERVATIVE") return AdaptiveStrategy::CONSERVATIVE;
    if (strategy_str == "BALANCED") return AdaptiveStrategy::BALANCED;
    if (strategy_str == "AGGRESSIVE") return AdaptiveStrategy::AGGRESSIVE;
    if (strategy_str == "POWER_EFFICIENT") return AdaptiveStrategy::POWER_EFFICIENT;
    if (strategy_str == "QUALITY_FOCUSED") return AdaptiveStrategy::QUALITY_FOCUSED;
    return AdaptiveStrategy::BALANCED; // Default
}

std::string strategy_to_string(AdaptiveStrategy strategy) {
    switch (strategy) {
        case AdaptiveStrategy::CONSERVATIVE: return "CONSERVATIVE";
        case AdaptiveStrategy::BALANCED: return "BALANCED";
        case AdaptiveStrategy::AGGRESSIVE: return "AGGRESSIVE";
        case AdaptiveStrategy::POWER_EFFICIENT: return "POWER_EFFICIENT";
        case AdaptiveStrategy::QUALITY_FOCUSED: return "QUALITY_FOCUSED";
        default: return "UNKNOWN";
    }
}

bool validate_adaptation_parameters(const AdaptiveParameters& params) {
    // Validate timing parameters
    if (params.max_acceptable_latency_ms <= 0.0 || params.adaptation_interval_ms == 0) {
        return false;
    }

    // Validate threshold parameters
    if (params.max_cpu_utilization_percent <= 0.0 || params.max_cpu_utilization_percent > 100.0 ||
        params.max_gpu_utilization_percent <= 0.0 || params.max_gpu_utilization_percent > 100.0 ||
        params.max_memory_utilization_percent <= 0.0 || params.max_memory_utilization_percent > 100.0) {
        return false;
    }

    // Validate fallback thresholds
    const auto& thresholds = params.fallback_thresholds;
    if (thresholds.high_cpu_threshold <= 0.0 || thresholds.high_cpu_threshold > 100.0 ||
        thresholds.high_gpu_threshold <= 0.0 || thresholds.high_gpu_threshold > 100.0 ||
        thresholds.high_memory_threshold <= 0.0 || thresholds.high_memory_threshold > 100.0) {
        return false;
    }

    return true;
}

bool validate_quality_configuration(const QualityConfiguration& config) {
    if (config.fft_size == 0 || config.overlap_factor == 0) {
        return false;
    }

    if (config.spectrum_resolution == 0 || config.waveform_resolution == 0) {
        return false;
    }

    if (config.vu_meter_update_rate == 0) {
        return false;
    }

    return true;
}

std::vector<std::string> get_supported_quality_levels() {
    return {"ULTRA_HIGH", "HIGH", "MEDIUM", "LOW", "MINIMAL", "AUTO"};
}

std::vector<std::string> get_supported_processing_modes() {
    return {"REAL_TIME", "OFFLINE", "INTERACTIVE", "BATCH", "STREAMING", "ADAPTIVE"};
}

} // namespace adaptive_utils

} // namespace vortex::core::processing