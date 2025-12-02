#include "network/audio_quality_manager.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>

namespace VortexGPU {
namespace Network {

// ============================================================================
// AudioQualityManager Implementation
// ============================================================================

AudioQualityManager::AudioQualityManager() {
}

AudioQualityManager::~AudioQualityManager() {
    shutdown();
}

bool AudioQualityManager::initialize(const QualityManagerConfig& config) {
    if (initialized_.load()) {
        return true;
    }

    config_ = config;

    try {
        // Initialize components
        predictor_ = std::make_unique<QualityPredictor>(config_);
        learner_ = std::make_unique<QualityLearner>(config_);
        analyzer_ = std::make_unique<QualityAnalyzer>(config_);

        running_.store(true);

        // Start background threads
        monitoring_thread_ = std::thread(&AudioQualityManager::monitoringThread, this);
        assessment_thread_ = std::thread(&AudioQualityManager::assessmentThread, this);
        adaptation_thread_ = std::thread(&AudioQualityManager::adaptationThread, this);
        event_processing_thread_ = std::thread(&AudioQualityManager::eventProcessingThread, this);

        // Initialize logging
        if (!config_.log_file_path.empty()) {
            std::ofstream log_file(config_.log_file_path, std::ios::app);
            if (log_file.is_open()) {
                log_file << "\n=== Audio Quality Manager Started at "
                        << std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::system_clock::now().time_since_epoch()).count() << " ===\n";
                log_file.close();
            }
        }

        initialized_.store(true);

        std::cout << "AudioQualityManager initialized with strategy: "
                  << strategyToString(config_.strategy) << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize AudioQualityManager: " << e.what() << std::endl;
        shutdown();
        return false;
    }
}

void AudioQualityManager::shutdown() {
    if (!initialized_.load()) {
        return;
    }

    running_.store(false);

    // Wait for threads to finish
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }

    if (assessment_thread_.joinable()) {
        assessment_thread_.join();
    }

    if (adaptation_thread_.joinable()) {
        adaptation_thread_.join();
    }

    if (event_processing_thread_.joinable()) {
        event_processing_thread_.join();
    }

    // Clear all streams
    streams_.clear();

    // Log shutdown
    if (!config_.log_file_path.empty()) {
        std::ofstream log_file(config_.log_file_path, std::ios::app);
        if (log_file.is_open()) {
            log_file << "\n=== Audio Quality Manager Stopped at "
                    << std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count() << " ===\n";
            log_file.close();
        }
    }

    initialized_.load() = false;

    std::cout << "AudioQualityManager shut down" << std::endl;
}

void AudioQualityManager::reset() {
    std::lock_guard<std::mutex> lock(global_mutex_);

    for (auto& pair : streams_) {
        std::lock_guard<std::mutex> stream_lock(pair.second->stream_mutex);
        pair.second->metrics.clear();
        pair.second->statistics.clear();
        pair.second->current_quality = AudioQualityLevel::GOOD;
        pair.second->current_quality_score = 50.0;
        pair.second->current_network_condition = NetworkCondition::UNKNOWN;
        pair.second->last_assessment.reset();
    }

    event_queue_ = {};
    event_history_.clear();

    writeToLog("Audio Quality Manager reset");
}

bool AudioQualityManager::addStream(const std::string& stream_id, uint32_t initial_bitrate, uint32_t sample_rate) {
    if (stream_id.empty()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(global_mutex_);

    if (streams_.find(stream_id) != streams_.end()) {
        return false; // Stream already exists
    }

    auto stream_data = std::make_unique<StreamData>();
    stream_data->stream_id = stream_id;
    stream_data->current_bitrate = initial_bitrate;
    stream_data->current_sample_rate = sample_rate;
    stream_data->current_buffer_size = 512; // Default buffer size
    stream_data->current_codec = AudioCodec::OPUS; // Default codec
    stream_data->adaptation_params = config_.adaptation_params;

    // Set initial bitrate index
    auto& bitrates = stream_data->adaptation_params.available_bitrates;
    auto bitrate_it = std::find(bitrates.begin(), bitrates.end(), initial_bitrate);
    if (bitrate_it != bitrates.end()) {
        stream_data->adaptation_params.current_bitrate_index =
            std::distance(bitrates.begin(), bitrate_it);
    }

    // Set initial sample rate index
    auto& sample_rates = stream_data->adaptation_params.available_sample_rates;
    auto rate_it = std::find(sample_rates.begin(), sample_rates.end(), sample_rate);
    if (rate_it != sample_rates.end()) {
        stream_data->adaptation_params.current_sample_rate_index =
            std::distance(sample_rates.begin(), rate_it);
    }

    stream_data->last_adaptation = std::chrono::steady_clock::now();
    stream_data->last_assessment_time = std::chrono::steady_clock::now();

    streams_[stream_id] = std::move(stream_data);

    // Record event
    QualityEvent event;
    event.type = QualityEventType::QUALITY_REPORT_GENERATED;
    event.stream_id = stream_id;
    event.message = "Stream added to quality monitoring: " + stream_id;
    event.timestamp = std::chrono::steady_clock::now();
    event.severity = "info";
    event.details["bitrate"] = std::to_string(initial_bitrate);
    event.details["sample_rate"] = std::to_string(sample_rate);
    recordEvent(event);

    writeToLog("Added stream: " + stream_id + " (bitrate: " + std::to_string(initial_bitrate) +
               ", sample_rate: " + std::to_string(sample_rate) + ")");

    return true;
}

bool AudioQualityManager::removeStream(const std::string& stream_id) {
    std::lock_guard<std::mutex> lock(global_mutex_);

    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return false;
    }

    // Record event
    QualityEvent event;
    event.type = QualityEventType::QUALITY_REPORT_GENERATED;
    event.stream_id = stream_id;
    event.message = "Stream removed from quality monitoring: " + stream_id;
    event.timestamp = std::chrono::steady_clock::now();
    event.severity = "info";
    event.details["final_quality"] = qualityLevelToString(it->second->current_quality);
    event.details["final_score"] = std::to_string(it->second->current_quality_score);
    recordEvent(event);

    streams_.erase(it);

    writeToLog("Removed stream: " + stream_id);

    return true;
}

bool AudioQualityManager::isStreamMonitored(const std::string& stream_id) const {
    std::lock_guard<std::mutex> lock(global_mutex_);
    return streams_.find(stream_id) != streams_.end();
}

std::vector<std::string> AudioQualityManager::getMonitoredStreams() const {
    std::lock_guard<std::mutex> lock(global_mutex_);

    std::vector<std::string> stream_ids;
    stream_ids.reserve(streams_.size());

    for (const auto& pair : streams_) {
        stream_ids.push_back(pair.first);
    }

    return stream_ids;
}

void AudioQualityManager::reportMetric(const std::string& stream_id, QualityMetric metric) {
    std::lock_guard<std::mutex> lock(global_mutex_);

    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return;
    }

    std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);

    // Add metric to stream data
    auto& metrics = it->second->metrics[metric.type];
    metrics.push_back(metric);

    // Limit metric history size
    if (metrics.size() > config_.max_samples_per_metric) {
        metrics.pop_front();
    }

    // Update statistics
    updateMetricStatistics(stream_id, metric.type);

    // Log detailed metrics if enabled
    if (config_.enable_detailed_logging) {
        writeToLog("Metric reported for " + stream_id + ": " + metricTypeToString(metric.type) +
                   " = " + std::to_string(metric.value) + " " + metric.unit);
    }
}

void AudioQualityManager::reportLatency(const std::string& stream_id, double latency_ms) {
    QualityMetric metric(QualityMetricType::LATENCY, latency_ms, "ms");
    reportMetric(stream_id, metric);
}

void AudioQualityManager::reportJitter(const std::string& stream_id, double jitter_ms) {
    QualityMetric metric(QualityMetricType::JITTER, jitter_ms, "ms");
    reportMetric(stream_id, metric);
}

void AudioQualityManager::reportPacketLoss(const std::string& stream_id, double loss_percent) {
    QualityMetric metric(QualityMetricType::PACKET_LOSS, loss_percent, "%");
    reportMetric(stream_id, metric);
}

void AudioQualityManager::reportThroughput(const std::string& stream_id, double throughput_mbps) {
    QualityMetric metric(QualityMetricType::THROUGHPUT, throughput_mbps, "Mbps");
    reportMetric(stream_id, metric);
}

void AudioQualityManager::reportAudioLevel(const std::string& stream_id, double peak_level_db, double rms_level_db) {
    QualityMetric peak_metric(QualityMetricType::PEAK_LEVEL, peak_level_db, "dB");
    QualityMetric rms_metric(QualityMetricType::RMS_LEVEL, rms_level_db, "dB");

    reportMetric(stream_id, peak_metric);
    reportMetric(stream_id, rms_metric);
}

void AudioQualityManager::reportResourceUtilization(const std::string& stream_id, double cpu_percent, double memory_percent) {
    QualityMetric cpu_metric(QualityMetricType::CPU_UTILIZATION, cpu_percent, "%");
    QualityMetric memory_metric(QualityMetricType::MEMORY_UTILIZATION, memory_percent, "%");

    reportMetric(stream_id, cpu_metric);
    reportMetric(stream_id, memory_metric);
}

void AudioQualityManager::reportBufferEvent(const std::string& stream_id, bool underrun, bool overrun) {
    QualityMetricType type = underrun ? QualityMetricType::BUFFER_UNDERRUN : QualityMetricType::BUFFER_OVERRUN;
    double value = underrun ? 1.0 : 1.0; // Count as occurrence
    std::string unit = "count";

    QualityMetric metric(type, value, unit);
    reportMetric(stream_id, metric);
}

QualityAssessment AudioQualityManager::assessQuality(const std::string& stream_id) const {
    std::lock_guard<std::mutex> lock(global_mutex_);

    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return QualityAssessment{};
    }

    std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);

    // Use analyzer to perform quality assessment
    return analyzer_->analyze(it->second->metrics, config_.thresholds);
}

std::map<std::string, QualityAssessment> AudioQualityManager::assessAllStreams() const {
    std::lock_guard<std::mutex> lock(global_mutex_);

    std::map<std::string, QualityAssessment> assessments;

    for (const auto& pair : streams_) {
        assessments[pair.first] = assessQuality(pair.first);
    }

    return assessments;
}

AudioQualityLevel AudioQualityManager::getCurrentQualityLevel(const std::string& stream_id) const {
    std::lock_guard<std::mutex> lock(global_mutex_);

    auto it = streams_.find(stream_id);
    if (it != streams_.end()) {
        std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);
        return it->second->current_quality;
    }

    return AudioQualityLevel::GOOD;
}

NetworkCondition AudioQualityManager::getCurrentNetworkCondition(const std::string& stream_id) const {
    std::lock_guard<std::mutex> lock(global_mutex_);

    auto it = streams_.find(stream_id);
    if (it != streams_.end()) {
        std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);
        return it->second->current_network_condition;
    }

    return NetworkCondition::UNKNOWN;
}

double AudioQualityManager::getQualityScore(const std::string& stream_id) const {
    std::lock_guard<std::mutex> lock(global_mutex_);

    auto it = streams_.find(stream_id);
    if (it != streams_.end()) {
        std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);
        return it->second->current_quality_score;
    }

    return 50.0;
}

QualityStatistics AudioQualityManager::getMetricStatistics(const std::string& stream_id, QualityMetricType metric_type) const {
    std::lock_guard<std::mutex> lock(global_mutex_);

    auto it = streams_.find(stream_id);
    if (it != streams_.end()) {
        std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);

        auto stat_it = it->second->statistics.find(metric_type);
        if (stat_it != it->second->statistics.end()) {
            return stat_it->second;
        }
    }

    return QualityStatistics{};
}

std::vector<QualityMetric> AudioQualityManager::getRecentMetrics(const std::string& stream_id, QualityMetricType metric_type, size_t count) const {
    std::lock_guard<std::mutex> lock(global_mutex_);

    auto it = streams_.find(stream_id);
    if (it != streams_.end()) {
        std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);

        auto metric_it = it->second->metrics.find(metric_type);
        if (metric_it != it->second->metrics.end()) {
            const auto& metrics = metric_it->second;
            size_t start_index = metrics.size() > count ? metrics.size() - count : 0;

            std::vector<QualityMetric> result;
            result.reserve(metrics.size() - start_index);

            for (size_t i = start_index; i < metrics.size(); ++i) {
                result.push_back(metrics[i]);
            }

            return result;
        }
    }

    return {};
}

std::string AudioQualityManager::generateQualityReport(const std::string& stream_id) const {
    std::ostringstream oss;
    auto assessment = assessQuality(stream_id);
    auto stats = getMetricStatistics(stream_id, QualityMetricType::LATENCY);

    oss << "=== Audio Quality Report for " << stream_id << " ===\n";
    oss << "Generated at: " << std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() << "\n\n";

    oss << "Overall Quality: " << qualityLevelToString(assessment.overall_quality) << "\n";
    oss << "Quality Score: " << std::fixed << std::setprecision(1) << assessment.quality_score << "/100\n";
    oss << "Network Condition: " << QualityUtils::networkConditionToString(assessment.network_condition) << "\n\n";

    oss << "Component Scores:\n";
    oss << "  Stability: " << std::fixed << std::setprecision(1) << assessment.stability_score << "/100\n";
    oss << "  Efficiency: " << std::fixed << std::setprecision(1) << assessment.efficiency_score << "/100\n";
    oss << "  User Experience: " << std::fixed << std::setprecision(1) << assessment.user_experience_score << "/100\n\n";

    if (!assessment.issues.empty()) {
        oss << "Issues Identified:\n";
        for (const auto& issue : assessment.issues) {
            oss << "  - " << issue << "\n";
        }
        oss << "\n";
    }

    if (!assessment.recommendations.empty()) {
        oss << "Recommendations:\n";
        for (const auto& recommendation : assessment.recommendations) {
            oss << "  - " << recommendation << "\n";
        }
        oss << "\n";
    }

    // Current configuration
    auto stream_it = streams_.find(stream_id);
    if (stream_it != streams_.end()) {
        std::lock_guard<std::mutex> stream_lock(stream_it->second->stream_mutex);
        const auto& stream_data = stream_it->second;

        oss << "Current Configuration:\n";
        oss << "  Bitrate: " << stream_data->current_bitrate << " bps\n";
        oss << "  Sample Rate: " << stream_data->current_sample_rate << " Hz\n";
        oss << "  Buffer Size: " << stream_data->current_buffer_size << " samples\n";
        oss << "  Codec: " << static_cast<int>(stream_data->current_codec) << "\n";
        oss << "  Adaptations Performed: " << stream_data->adaptation_count << "\n";
    }

    return oss.str();
}

std::string AudioQualityManager::generateOverallReport() const {
    std::ostringstream oss;
    auto assessments = assessAllStreams();

    oss << "=== Overall Audio Quality Report ===\n";
    oss << "Generated at: " << std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() << "\n\n";

    oss << "Monitored Streams: " << streams_.size() << "\n";

    if (!assessments.empty()) {
        // Calculate overall statistics
        double total_quality_score = 0.0;
        int healthy_streams = 0;
        std::map<AudioQualityLevel, int> quality_distribution;

        for (const auto& pair : assessments) {
            const QualityAssessment& assessment = pair.second;
            total_quality_score += assessment.quality_score;
            quality_distribution[assessment.overall_quality]++;

            if (assessment.overall_quality >= AudioQualityLevel::GOOD) {
                healthy_streams++;
            }
        }

        double average_quality_score = total_quality_score / assessments.size();

        oss << "Average Quality Score: " << std::fixed << std::setprecision(1) << average_quality_score << "/100\n";
        oss << "Healthy Streams: " << healthy_streams << "/" << assessments.size()
            << " (" << (static_cast<double>(healthy_streams) / assessments.size() * 100.0) << "%)\n\n";

        oss << "Quality Distribution:\n";
        for (const auto& pair : quality_distribution) {
            oss << "  " << qualityLevelToString(pair.first) << ": " << pair.second << " streams\n";
        }
        oss << "\n";

        // Stream-specific details
        oss << "Stream Details:\n";
        for (const auto& pair : assessments) {
            const std::string& stream_id = pair.first;
            const QualityAssessment& assessment = pair.second;

            oss << "  " << stream_id << ":\n";
            oss << "    Quality: " << qualityLevelToString(assessment.overall_quality)
                << " (" << std::fixed << std::setprecision(1) << assessment.quality_score << "/100)\n";
            oss << "    Network: " << QualityUtils::networkConditionToString(assessment.network_condition) << "\n";

            if (assessment.has_critical_issues()) {
                oss << "    Status: CRITICAL ISSUES DETECTED\n";
            } else if (assessment.needs_improvement()) {
                oss << "    Status: Needs improvement\n";
            } else {
                oss << "    Status: Good\n";
            }
            oss << "\n";
        }
    }

    // Recent events summary
    std::lock_guard<std::mutex> lock(global_mutex_);
    if (!event_history_.empty()) {
        oss << "Recent Events (last 10):\n";
        size_t start_index = event_history_.size() > 10 ? event_history_.size() - 10 : 0;
        for (size_t i = start_index; i < event_history_.size(); ++i) {
            const QualityEvent& event = event_history_[i];
            oss << "  " << QualityUtils::eventTypeToString(event.type)
                << " - " << event.stream_id << ": " << event.message << "\n";
        }
    }

    return oss.str();
}

void AudioQualityManager::exportMetrics(const std::string& stream_id, const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return;
    }

    std::lock_guard<std::mutex> lock(global_mutex_);

    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return;
    }

    std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);

    // CSV header
    file << "Timestamp,MetricType,Value,Unit,IsValid\n";

    // Export all metrics
    for (const auto& pair : it->second->metrics) {
        QualityMetricType metric_type = pair.first;
        const auto& metrics = pair.second;

        for (const auto& metric : metrics) {
            auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                metric.timestamp.time_since_epoch()).count();

            file << timestamp << ","
                 << QualityUtils::metricTypeToString(metric_type) << ","
                 << metric.value << ","
                 << metric.unit << ","
                 << (metric.is_valid ? "true" : "false") << "\n";
        }
    }

    file.close();
    writeToLog("Exported metrics for " + stream_id + " to " + filename);
}

bool AudioQualityManager::enableAdaptation(const std::string& stream_id, bool enabled) {
    std::lock_guard<std::mutex> lock(global_mutex_);

    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return false;
    }

    std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);
    it->second->adaptation_enabled = enabled;

    writeToLog("Adaptation " + std::string(enabled ? "enabled" : "disabled") + " for " + stream_id);
    return true;
}

bool AudioQualityManager::forceAdaptation(const std::string& stream_id, AudioQualityLevel target_quality) {
    std::lock_guard<std::mutex> lock(global_mutex_);

    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return false;
    }

    std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);

    AudioQualityLevel old_quality = it->second->current_quality;
    uint32_t old_bitrate = it->second->current_bitrate;

    // Force adaptation based on target quality
    switch (target_quality) {
        case AudioQualityLevel::POOR:
            // Set lowest quality settings
            adaptBitrate(stream_id, false);
            adaptSampleRate(stream_id, false);
            adaptBufferSize(stream_id, true); // Increase buffer size for stability
            break;

        case AudioQualityLevel::FAIR:
            // Set low-medium quality settings
            adaptBitrate(stream_id, false);
            adaptSampleRate(stream_id, false);
            break;

        case AudioQualityLevel::GOOD:
            // Set medium quality settings
            // Use default or current settings
            break;

        case AudioQualityLevel::EXCELLENT:
            // Set high quality settings
            adaptBitrate(stream_id, true);
            adaptSampleRate(stream_id, true);
            break;

        case AudioQualityLevel::STUDIO:
            // Set highest quality settings
            adaptBitrate(stream_id, true);
            adaptSampleRate(stream_id, true);
            adaptBufferSize(stream_id, false); // Smaller buffer for low latency
            break;
    }

    it->second->current_quality = target_quality;
    it->second->last_adaptation = std::chrono::steady_clock::now();
    it->second->adaptation_count++;

    // Record event
    QualityEvent event;
    event.type = QualityEventType::ADAPTATION_TRIGGERED;
    event.stream_id = stream_id;
    event.old_quality = old_quality;
    event.new_quality = target_quality;
    event.old_score = 0.0; // Could calculate previous score
    event.new_score = static_cast<double>(target_quality) * 25.0; // Simple mapping
    event.timestamp = std::chrono::steady_clock::now();
    event.severity = "info";
    event.message = "Force adaptation triggered: " + qualityLevelToString(old_quality) + " -> " + qualityLevelToString(target_quality);
    event.details["old_bitrate"] = std::to_string(old_bitrate);
    event.details["new_bitrate"] = std::to_string(it->second->current_bitrate);
    recordEvent(event);

    writeToLog("Force adaptation for " + stream_id + ": " +
               qualityLevelToString(old_quality) + " -> " + qualityLevelToString(target_quality));

    return true;
}

uint32_t AudioQualityManager::getCurrentBitrate(const std::string& stream_id) const {
    std::lock_guard<std::mutex> lock(global_mutex_);

    auto it = streams_.find(stream_id);
    if (it != streams_.end()) {
        std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);
        return it->second->current_bitrate;
    }

    return 0;
}

uint32_t AudioQualityManager::getCurrentSampleRate(const std::string& stream_id) const {
    std::lock_guard<std::mutex> lock(global_mutex_);

    auto it = streams_.find(stream_id);
    if (it != streams_.end()) {
        std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);
        return it->second->current_sample_rate;
    }

    return 0;
}

size_t AudioQualityManager::getCurrentBufferSize(const std::string& stream_id) const {
    std::lock_guard<std::mutex> lock(global_mutex_);

    auto it = streams_.find(stream_id);
    if (it != streams_.end()) {
        std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);
        return it->second->current_buffer_size;
    }

    return 0;
}

AudioCodec AudioQualityManager::getCurrentCodec(const std::string& stream_id) const {
    std::lock_guard<std::mutex> lock(global_mutex_);

    auto it = streams_.find(stream_id);
    if (it != streams_.end()) {
        std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);
        return it->second->current_codec;
    }

    return AudioCodec::OPUS;
}

std::pair<AudioQualityLevel, double> AudioQualityManager::predictQuality(const std::string& stream_id, std::chrono::seconds future_duration) const {
    std::lock_guard<std::mutex> lock(global_mutex_);

    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return {AudioQualityLevel::GOOD, 50.0};
    }

    std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);

    if (predictor_) {
        return predictor_->predictQuality(it->second->metrics, future_duration);
    }

    return {it->second->current_quality, it->second->current_quality_score};
}

std::vector<std::string> AudioQualityManager::getQualityRecommendations(const std::string& stream_id) const {
    auto assessment = assessQuality(stream_id);
    return analyzer_->generateRecommendations(assessment);
}

std::map<std::string, double> AudioQualityManager::analyzeTrends(const std::string& stream_id, std::chrono::seconds window) const {
    std::lock_guard<std::mutex> lock(global_mutex_);

    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return {};
    }

    std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);

    std::map<std::string, double> trends;
    auto cutoff_time = std::chrono::steady_clock::now() - window;

    for (const auto& pair : it->second->metrics) {
        QualityMetricType metric_type = pair.first;
        const auto& metrics = pair.second;

        // Filter metrics within the time window
        std::vector<double> recent_values;
        for (const auto& metric : metrics) {
            if (metric.timestamp >= cutoff_time) {
                recent_values.push_back(metric.value);
            }
        }

        if (recent_values.size() >= 2) {
            // Calculate trend using simple linear regression
            double slope = 0.0;
            if (recent_values.size() > 1) {
                double n = recent_values.size();
                double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;

                for (size_t i = 0; i < recent_values.size(); ++i) {
                    sum_x += i;
                    sum_y += recent_values[i];
                    sum_xy += i * recent_values[i];
                    sum_x2 += i * i;
                }

                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
            }

            trends[QualityUtils::metricTypeToString(metric_type)] = slope;
        }
    }

    return trends;
}

void AudioQualityManager::updateConfig(const QualityManagerConfig& config) {
    config_ = config;

    // Update components
    if (predictor_) {
        predictor_ = std::make_unique<QualityPredictor>(config_);
    }

    if (learner_) {
        learner_ = std::make_unique<QualityLearner>(config_);
    }

    if (analyzer_) {
        analyzer_ = std::make_unique<QualityAnalyzer>(config_);
    }

    writeToLog("Quality manager configuration updated");
}

void AudioQualityManager::updateThresholds(const QualityThresholds& thresholds) {
    config_.thresholds = thresholds;

    if (analyzer_) {
        analyzer_ = std::make_unique<QualityAnalyzer>(config_);
    }

    writeToLog("Quality thresholds updated");
}

void AudioQualityManager::setAdaptationStrategy(QualityAdaptationStrategy strategy) {
    config_.strategy = strategy;
    writeToLog("Adaptation strategy changed to: " + strategyToString(strategy));
}

void AudioQualityManager::setQualityPolicy(QualityPolicy policy) {
    config_.policy = policy;
    writeToLog("Quality policy changed to: " + std::to_string(static_cast<int>(policy)));
}

void AudioQualityManager::setEventCallback(std::function<void(const QualityEvent&)> callback) {
    std::lock_guard<std::mutex> lock(global_mutex_);
    event_callback_ = callback;
}

void AudioQualityManager::publishEvent(const QualityEvent& event) {
    recordEvent(event);
}

std::vector<QualityEvent> AudioQualityManager::getRecentEvents(const std::string& stream_id, size_t max_events) const {
    std::lock_guard<std::mutex> lock(global_mutex_);

    std::vector<QualityEvent> events;

    for (auto it = event_history_.rbegin(); it != event_history_.rend() && events.size() < max_events; ++it) {
        if (stream_id.empty() || it->stream_id == stream_id) {
            events.push_back(*it);
        }
    }

    return events;
}

bool AudioQualityManager::isHealthy(const std::string& stream_id) const {
    auto quality_level = getCurrentQualityLevel(stream_id);
    auto network_condition = getCurrentNetworkCondition(stream_id);
    double quality_score = getQualityScore(stream_id);

    return quality_level >= AudioQualityLevel::GOOD &&
           network_condition != NetworkCondition::VERY_POOR &&
           quality_score >= 70.0;
}

std::vector<std::string> AudioQualityManager::getUnhealthyStreams() const {
    std::lock_guard<std::mutex> lock(global_mutex_);

    std::vector<std::string> unhealthy_streams;

    for (const auto& pair : streams_) {
        if (!isHealthy(pair.first)) {
            unhealthy_streams.push_back(pair.first);
        }
    }

    return unhealthy_streams;
}

std::string AudioQualityManager::getDiagnosticInfo(const std::string& stream_id) const {
    std::ostringstream oss;

    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        oss << "Stream not found: " << stream_id;
        return oss.str();
    }

    std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);
    const auto& stream_data = it->second;

    oss << "=== Diagnostic Information for " << stream_id << " ===\n";
    oss << "Stream ID: " << stream_data->stream_id << "\n";
    oss << "Current Quality: " << qualityLevelToString(stream_data->current_quality) << "\n";
    oss << "Quality Score: " << std::fixed << std::setprecision(1) << stream_data->current_quality_score << "/100\n";
    oss << "Network Condition: " << QualityUtils::networkConditionToString(stream_data->current_network_condition) << "\n\n";

    oss << "Configuration:\n";
    oss << "  Bitrate: " << stream_data->current_bitrate << " bps\n";
    oss << "  Sample Rate: " << stream_data->current_sample_rate << " Hz\n";
    oss << "  Buffer Size: " << stream_data->current_buffer_size << " samples\n";
    oss << "  Codec: " << static_cast<int>(stream_data->current_codec) << "\n";
    oss << "  Adaptation Enabled: " << (stream_data->adaptation_enabled ? "Yes" : "No") << "\n";
    oss << "  Adaptations Performed: " << stream_data->adaptation_count << "\n\n";

    oss << "Recent Statistics:\n";
    for (const auto& stat_pair : stream_data->statistics) {
        const QualityStatistics& stats = stat_pair.second;
        if (stats.total_samples > 0) {
            oss << "  " << QualityUtils::metricTypeToString(stat_pair.first) << ":\n";
            oss << "    Samples: " << stats.total_samples << "\n";
            oss << "    Average: " << std::fixed << std::setprecision(2) << stats.average << "\n";
            oss << "    Min: " << std::fixed << std::setprecision(2) << stats.min_value << "\n";
            oss << "    Max: " << std::fixed << std::setprecision(2) << stats.max_value << "\n";
            oss << "    Std Dev: " << std::fixed << std::setprecision(2) << stats.standard_deviation << "\n";
        }
    }

    oss << "\nTiming Information:\n";
    oss << "  Last Assessment: " << std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - stream_data->last_assessment_time).count() << " seconds ago\n";
    oss << "  Last Adaptation: " << std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - stream_data->last_adaptation).count() << " seconds ago\n";

    return oss.str();
}

void AudioQualityManager::performHealthCheck() {
    auto unhealthy_streams = getUnhealthyStreams();

    if (!unhealthy_streams.empty()) {
        writeToLog("Health check found " + std::to_string(unhealthy_streams.size()) + " unhealthy streams");

        for (const auto& stream_id : unhealthy_streams) {
            // Generate health event
            QualityEvent event;
            event.type = QualityEventType::WARNING_ISSUED;
            event.stream_id = stream_id;
            event.message = "Stream health check failed";
            event.timestamp = std::chrono::steady_clock::now();
            event.severity = "warning";
            recordEvent(event);

            // Attempt recovery if adaptation is enabled
            auto it = streams_.find(stream_id);
            if (it != streams_.end() && it->second->adaptation_enabled) {
                forceAdaptation(stream_id, AudioQualityLevel::FAIR);
            }
        }
    }
}

void AudioQualityManager::optimizeQuality() {
    std::lock_guard<std::mutex> lock(global_mutex_);

    for (const auto& pair : streams_) {
        const std::string& stream_id = pair.first;

        if (pair.second->adaptation_enabled) {
            assessStreamQuality(stream_id);

            // Check if optimization is needed
            if (shouldAdaptUp(stream_id) || shouldAdaptDown(stream_id)) {
                adaptStreamQuality(stream_id);
            }
        }
    }

    writeToLog("Quality optimization completed");
}

// Private methods implementation

void AudioQualityManager::monitoringThread() {
    while (running_.load()) {
        try {
            // Periodic maintenance tasks
            if (config_.enable_monitoring) {
                // Clean up old metrics
                auto cutoff_time = std::chrono::steady_clock::now() - config_.statistics_window;

                std::lock_guard<std::mutex> lock(global_mutex_);
                for (auto& pair : streams_) {
                    std::lock_guard<std::mutex> stream_lock(pair.second->stream_mutex);

                    for (auto& metric_pair : pair.second->metrics) {
                        auto& metrics = metric_pair.second;
                        auto it = std::find_if(metrics.begin(), metrics.end(),
                            [cutoff_time](const QualityMetric& metric) {
                                return metric.timestamp >= cutoff_time;
                            });

                        if (it != metrics.begin()) {
                            metrics.erase(metrics.begin(), it);
                        }
                    }
                }
            }

            std::this_thread::sleep_for(config_.monitoring_interval);

        } catch (const std::exception& e) {
            std::cerr << "Monitoring thread error: " << e.what() << std::endl;
        }
    }
}

void AudioQualityManager::assessmentThread() {
    while (running_.load()) {
        try {
            // Assess quality for all streams
            if (config_.enable_monitoring) {
                std::lock_guard<std::mutex> lock(global_mutex_);

                for (const auto& pair : streams_) {
                    assessStreamQuality(pair.first);
                }
            }

            std::this_thread::sleep_for(config_.adaptation_params.quality_evaluation_interval);

        } catch (const std::exception& e) {
            std::cerr << "Assessment thread error: " << e.what() << std::endl;
        }
    }
}

void AudioQualityManager::adaptationThread() {
    while (running_.load()) {
        try {
            if (config_.enable_adaptation) {
                std::lock_guard<std::mutex> lock(global_mutex_);

                for (const auto& pair : streams_) {
                    const std::string& stream_id = pair.first;

                    if (pair.second->adaptation_enabled) {
                        // Check if adaptation is needed
                        auto now = std::chrono::steady_clock::now();
                        auto time_since_last_adaptation = std::chrono::duration_cast<std::chrono::seconds>(
                            now - pair.second->last_adaptation);

                        if (time_since_last_adaptation >= pair.second->adaptation_params.adaptation_cooldown) {
                            if (shouldAdaptUp(stream_id) || shouldAdaptDown(stream_id)) {
                                adaptStreamQuality(stream_id);
                            }

                            // Perform predictive adaptation if enabled
                            if (config_.enable_prediction && pair.second->adaptation_params.enable_predictive_adaptation) {
                                performPredictiveAdaptation(stream_id);
                            }
                        }
                    }
                }
            }

            std::this_thread::sleep_for(std::chrono::seconds(2));

        } catch (const std::exception& e) {
            std::cerr << "Adaptation thread error: " << e.what() << std::endl;
        }
    }
}

void AudioQualityManager::eventProcessingThread() {
    while (running_.load()) {
        try {
            std::queue<QualityEvent> events_to_process;

            {
                std::lock_guard<std::mutex> lock(global_mutex_);
                events_to_process.swap(event_queue_);
            }

            while (!events_to_process.empty()) {
                QualityEvent event = events_to_process.front();
                events_to_process.pop();

                // Add to history
                {
                    std::lock_guard<std::mutex> lock(global_mutex_);
                    event_history_.push_back(event);
                    if (event_history_.size() > 10000) {
                        event_history_.erase(event_history_.begin());
                    }
                }

                // Call callback if set
                if (event_callback_) {
                    event_callback_(event);
                }

                // Log event
                if (config_.enable_quality_events) {
                    writeToLog("Event: " + QualityUtils::eventTypeToString(event.type) +
                              " - " + event.stream_id + ": " + event.message);
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));

        } catch (const std::exception& e) {
            std::cerr << "Event processing thread error: " << e.what() << std::endl;
        }
    }
}

void AudioQualityManager::updateMetricStatistics(const std::string& stream_id, QualityMetricType metric_type) {
    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return;
    }

    std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);

    auto metrics_it = it->second->metrics.find(metric_type);
    if (metrics_it == it->second->metrics.end() || metrics_it->second.empty()) {
        return;
    }

    const auto& metrics = metrics_it->second;
    QualityStatistics& stats = it->second->statistics[metric_type];

    // Reset statistics
    stats.reset();
    stats.total_samples = metrics.size();

    // Calculate basic statistics
    for (const auto& metric : metrics) {
        if (metric.is_valid) {
            stats.min_value = std::min(stats.min_value, metric.value);
            stats.max_value = std::max(stats.max_value, metric.value);
            stats.average += metric.value;
        }
    }

    if (stats.total_samples > 0) {
        stats.average /= stats.total_samples;

        // Calculate standard deviation
        double sum_squared_diff = 0.0;
        for (const auto& metric : metrics) {
            if (metric.is_valid) {
                double diff = metric.value - stats.average;
                sum_squared_diff += diff * diff;
            }
        }
        stats.standard_deviation = std::sqrt(sum_squared_diff / stats.total_samples);

        // Calculate percentiles
        std::vector<double> sorted_values;
        for (const auto& metric : metrics) {
            if (metric.is_valid) {
                sorted_values.push_back(metric.value);
            }
        }

        if (!sorted_values.empty()) {
            std::sort(sorted_values.begin(), sorted_values.end());

            stats.median = sorted_values[sorted_values.size() / 2];

            if (sorted_values.size() >= 20) {
                size_t p95_index = static_cast<size_t>(sorted_values.size() * 0.95);
                size_t p99_index = static_cast<size_t>(sorted_values.size() * 0.99);
                stats.percentile_95 = sorted_values[std::min(p95_index, sorted_values.size() - 1)];
                stats.percentile_99 = sorted_values[std::min(p99_index, sorted_values.size() - 1)];
            }
        }
    }

    stats.last_update = std::chrono::steady_clock::now();
}

void AudioQualityManager::assessStreamQuality(const std::string& stream_id) {
    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return;
    }

    std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);

    // Perform quality assessment
    QualityAssessment assessment = analyzer_->analyze(it->second->metrics, config_.thresholds);

    // Update stream data
    AudioQualityLevel old_quality = it->second->current_quality;
    double old_score = it->second->current_quality_score;

    it->second->last_assessment = assessment;
    it->second->current_quality = assessment.overall_quality;
    it->second->current_quality_score = assessment.quality_score;
    it->second->current_network_condition = assessNetworkCondition(*it->second);
    it->second->last_assessment_time = std::chrono::steady_clock::now();

    // Generate event if quality changed significantly
    double quality_change = std::abs(assessment.quality_score - old_score);
    if (quality_change > config_.quality_change_threshold || old_quality != assessment.overall_quality) {
        QualityEvent event;
        event.type = (assessment.quality_score > old_score) ?
                     QualityEventType::QUALITY_IMPROVED : QualityEventType::QUALITY_DEGRADED;
        event.stream_id = stream_id;
        event.old_quality = old_quality;
        event.new_quality = assessment.overall_quality;
        event.old_score = old_score;
        event.new_score = assessment.quality_score;
        event.timestamp = std::chrono::steady_clock::now();
        event.severity = (assessment.has_critical_issues()) ? "error" : "info";
        event.message = "Quality " + std::string(assessment.quality_score > old_score ? "improved" : "degraded") +
                       " from " + std::to_string(old_score) + " to " + std::to_string(assessment.quality_score);
        recordEvent(event);
    }

    // Update learning model
    if (learner_ && config_.enable_learning) {
        learner_->addTrainingExample(it->second->statistics, assessment);
    }
}

void AudioQualityManager::adaptStreamQuality(const std::string& stream_id) {
    auto it = streams_.find(stream_id);
    if (it == streams_.end() || !it->second->adaptation_enabled) {
        return;
    }

    std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);

    AudioQualityLevel old_quality = it->second->current_quality;
    bool adapted = false;

    // Determine adaptation direction
    if (shouldAdaptDown(stream_id)) {
        // Reduce quality to improve stability
        if (adaptBitrate(stream_id, false)) adapted = true;
        if (adaptSampleRate(stream_id, false)) adapted = true;
        if (adaptBufferSize(stream_id, true)) adapted = true;
    } else if (shouldAdaptUp(stream_id)) {
        // Increase quality to improve experience
        if (adaptBufferSize(stream_id, false)) adapted = true;
        if (adaptSampleRate(stream_id, true)) adapted = true;
        if (adaptBitrate(stream_id, true)) adapted = true;
    }

    if (adapted) {
        it->second->last_adaptation = std::chrono::steady_clock::now();
        it->second->adaptation_count++;

        // Re-assess quality after adaptation
        assessStreamQuality(stream_id);

        // Record event
        QualityEvent event;
        event.type = QualityEventType::ADAPTATION_TRIGGERED;
        event.stream_id = stream_id;
        event.old_quality = old_quality;
        event.new_quality = it->second->current_quality;
        event.old_score = 0.0;
        event.new_score = it->second->current_quality_score;
        event.timestamp = std::chrono::steady_clock::now();
        event.severity = "info";
        event.message = "Automatic adaptation performed";
        event.details["adaptations_count"] = std::to_string(it->second->adaptation_count);
        recordEvent(event);

        writeToLog("Automatic adaptation for " + stream_id + ": " +
                   qualityLevelToString(old_quality) + " -> " +
                   qualityLevelToString(it->second->current_quality));
    }
}

void AudioQualityManager::performPredictiveAdaptation(const std::string& stream_id) {
    auto it = streams_.find(stream_id);
    if (it == streams_.end() || !predictor_) {
        return;
    }

    std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);

    // Predict quality for near future
    auto prediction = predictor_->predictQuality(it->second->metrics, config_.prediction_window);

    // If predicted quality is significantly worse, adapt proactively
    if (prediction.second < it->second->current_quality_score - 20.0) {
        forceAdaptation(stream_id, AudioQualityLevel::FAIR);

        QualityEvent event;
        event.type = QualityEventType::PREDICTION_MADE;
        event.stream_id = stream_id;
        event.new_quality = prediction.first;
        event.new_score = prediction.second;
        event.timestamp = std::chrono::steady_clock::now();
        event.severity = "warning";
        event.message = "Predictive adaptation triggered: predicted quality degradation";
        recordEvent(event);
    }
}

NetworkCondition AudioQualityManager::assessNetworkCondition(const StreamData& stream_data) const {
    auto latency_it = stream_data.statistics.find(QualityMetricType::LATENCY);
    auto jitter_it = stream_data.statistics.find(QualityMetricType::JITTER);
    auto loss_it = stream_data.statistics.find(QualityMetricType::PACKET_LOSS);

    double avg_latency = latency_it != stream_data.statistics.end() ? latency_it->second.average : 0.0;
    double avg_jitter = jitter_it != stream_data.statistics.end() ? jitter_it->second.average : 0.0;
    double avg_loss = loss_it != stream_data.statistics.end() ? loss_it->second.average : 0.0;

    // Assess network condition based on thresholds
    if (avg_loss < 1.0 && avg_jitter < 5.0 && avg_latency < 10.0) {
        return NetworkCondition::EXCELLENT;
    } else if (avg_loss < 3.0 && avg_jitter < 10.0 && avg_latency < 25.0) {
        return NetworkCondition::GOOD;
    } else if (avg_loss < 5.0 && avg_jitter < 20.0 && avg_latency < 50.0) {
        return NetworkCondition::FAIR;
    } else if (avg_loss < 10.0 && avg_jitter < 50.0 && avg_latency < 100.0) {
        return NetworkCondition::POOR;
    } else {
        return NetworkCondition::VERY_POOR;
    }
}

double AudioQualityManager::calculateQualityScore(const QualityAssessment& assessment) const {
    return assessment.quality_score;
}

bool AudioQualityManager::shouldAdaptUp(const std::string& stream_id) const {
    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return false;
    }

    std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);

    // Check if quality is good and can be improved
    if (it->second->current_quality_score < config_.adaptation_params.up_quality_threshold) {
        return false;
    }

    // Check adaptation cooldown
    auto time_since_last_adaptation = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - it->second->last_adaptation);
    if (time_since_last_adaptation < it->second->adaptation_params.adaptation_cooldown) {
        return false;
    }

    // Check if we're not already at maximum quality
    if (it->second->current_quality >= AudioQualityLevel::EXCELLENT) {
        return false;
    }

    return true;
}

bool AudioQualityManager::shouldAdaptDown(const std::string& stream_id) const {
    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return false;
    }

    std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);

    // Check if quality is poor and needs improvement
    if (it->second->current_quality_score > config_.adaptation_params.down_quality_threshold) {
        return false;
    }

    // Check adaptation cooldown
    auto time_since_last_adaptation = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - it->second->last_adaptation);
    if (time_since_last_adaptation < it->second->adaptation_params.adaptation_cooldown) {
        return false;
    }

    // Check if we're not already at minimum quality
    if (it->second->current_quality <= AudioQualityLevel::POOR) {
        return false;
    }

    return true;
}

bool AudioQualityManager::adaptBitrate(const std::string& stream_id, bool increase) {
    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return false;
    }

    std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);

    auto& params = it->second->adaptation_params;
    auto& bitrates = params.available_bitrates;
    size_t& current_index = params.current_bitrate_index;

    if (increase && current_index < bitrates.size() - 1) {
        current_index++;
        it->second->current_bitrate = bitrates[current_index];
        return true;
    } else if (!increase && current_index > 0) {
        current_index--;
        it->second->current_bitrate = bitrates[current_index];
        return true;
    }

    return false;
}

bool AudioQualityManager::adaptSampleRate(const std::string& stream_id, bool increase) {
    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return false;
    }

    std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);

    auto& params = it->second->adaptation_params;
    auto& sample_rates = params.available_sample_rates;
    size_t& current_index = params.current_sample_rate_index;

    if (increase && current_index < sample_rates.size() - 1) {
        current_index++;
        it->second->current_sample_rate = sample_rates[current_index];
        return true;
    } else if (!increase && current_index > 0) {
        current_index--;
        it->second->current_sample_rate = sample_rates[current_index];
        return true;
    }

    return false;
}

bool AudioQualityManager::adaptBufferSize(const std::string& stream_id, bool increase) {
    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return false;
    }

    std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);

    auto& params = it->second->adaptation_params;
    auto& buffer_sizes = params.available_buffer_sizes;
    size_t& current_index = params.current_buffer_size_index;

    if (increase && current_index < buffer_sizes.size() - 1) {
        current_index++;
        it->second->current_buffer_size = buffer_sizes[current_index];
        return true;
    } else if (!increase && current_index > 0) {
        current_index--;
        it->second->current_buffer_size = buffer_sizes[current_index];
        return true;
    }

    return false;
}

bool AudioQualityManager::adaptCodec(const std::string& stream_id) {
    // Codec adaptation is more complex and would require codec negotiation
    // This is a placeholder implementation
    return false;
}

void AudioQualityManager::recordEvent(const QualityEvent& event) {
    std::lock_guard<std::mutex> lock(global_mutex_);
    event_queue_.push(event);
}

std::string AudioQualityManager::metricTypeToString(QualityMetricType type) const {
    return QualityUtils::metricTypeToString(type);
}

std::string AudioQualityManager::qualityLevelToString(AudioQualityLevel level) const {
    return QualityUtils::qualityLevelToString(level);
}

std::string AudioQualityManager::strategyToString(QualityAdaptationStrategy strategy) const {
    return QualityUtils::strategyToString(strategy);
}

void AudioQualityManager::writeToLog(const std::string& message) {
    if (!config_.log_file_path.empty()) {
        std::ofstream log_file(config_.log_file_path, std::ios::app);
        if (log_file.is_open()) {
            auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            log_file << "[" << timestamp << "] " << message << std::endl;
            log_file.close();
        }
    }
}

void AudioQualityManager::saveConfiguration() {
    // Implementation for saving configuration to file
    // This would serialize the config_ object
}

void AudioQualityManager::loadConfiguration() {
    // Implementation for loading configuration from file
    // This would deserialize into the config_ object
}

// ============================================================================
// QualityPredictor Implementation
// ============================================================================

QualityPredictor::QualityPredictor(const QualityManagerConfig& config)
    : prediction_window_(config.prediction_window),
      confidence_threshold_(config.prediction_confidence_threshold),
      history_size_(config.prediction_history_size) {
}

std::pair<AudioQualityLevel, double> QualityPredictor::predictQuality(
    const std::unordered_map<QualityMetricType, std::deque<QualityMetric>>& metrics,
    std::chrono::seconds future_duration) const {

    // Simple linear regression-based prediction
    // In a real implementation, this would use more sophisticated models

    auto latency_it = metrics.find(QualityMetricType::LATENCY);
    auto jitter_it = metrics.find(QualityMetricType::JITTER);
    auto loss_it = metrics.find(QualityMetricType::PACKET_LOSS);

    if (latency_it == metrics.end() || jitter_it == metrics.end() || loss_it == metrics.end()) {
        return {AudioQualityLevel::GOOD, 50.0};
    }

    // Predict future values
    auto [predicted_latency, latency_confidence] = predictLatency(latency_it->second, future_duration);
    auto predicted_loss = predictPacketLoss(loss_it->second, future_duration);

    // Calculate predicted quality score based on predicted metrics
    double quality_score = 100.0;

    // Penalty for high latency
    if (predicted_latency > 50.0) {
        quality_score -= (predicted_latency - 50.0) * 2.0;
    }

    // Penalty for packet loss
    quality_score -= predicted_loss * 10.0;

    // Clamp to valid range
    quality_score = std::max(0.0, std::min(100.0, quality_score));

    // Determine quality level
    AudioQualityLevel quality_level;
    if (quality_score >= 90.0) {
        quality_level = AudioQualityLevel::EXCELLENT;
    } else if (quality_score >= 75.0) {
        quality_level = AudioQualityLevel::GOOD;
    } else if (quality_score >= 60.0) {
        quality_level = AudioQualityLevel::FAIR;
    } else {
        quality_level = AudioQualityLevel::POOR;
    }

    return {quality_level, quality_score};
}

std::pair<double, double> QualityPredictor::predictLatency(
    const std::deque<QualityMetric>& latency_metrics,
    std::chrono::seconds future_duration) const {

    if (latency_metrics.size() < 2) {
        return {10.0, 0.0}; // Default values
    }

    // Simple linear regression
    double slope = linearRegression(latency_metrics, future_duration);
    double current_latency = latency_metrics.back().value;
    double predicted_latency = current_latency + slope;

    // Calculate confidence based on data consistency
    double variance = 0.0;
    double mean = current_latency;

    for (const auto& metric : latency_metrics) {
        variance += std::pow(metric.value - mean, 2);
    }
    variance /= latency_metrics.size();

    double confidence = std::exp(-variance / 100.0); // Simple confidence metric

    return {std::max(0.0, predicted_latency), confidence};
}

double QualityPredictor::predictPacketLoss(
    const std::deque<QualityMetric>& loss_metrics,
    std::chrono::seconds future_duration) const {

    if (loss_metrics.empty()) {
        return 0.0;
    }

    // Use exponential smoothing for packet loss prediction
    return exponentialSmoothing(loss_metrics, 0.3);
}

void QualityPredictor::updateModel(
    const std::unordered_map<QualityMetricType, std::deque<QualityMetric>>& metrics,
    const QualityAssessment& actual_outcome) {
    // Update model parameters based on prediction accuracy
    // In a real implementation, this would adjust model weights
}

double QualityPredictor::linearRegression(const std::deque<QualityMetric>& metrics, std::chrono::seconds future) const {
    if (metrics.size() < 2) {
        return 0.0;
    }

    size_t n = metrics.size();
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double x = static_cast<double>(i);
        double y = metrics[i].value;
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
    }

    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);

    // Predict future value
    return slope * static_cast<double>(future.count());
}

double QualityPredictor::movingAverage(const std::deque<QualityMetric>& metrics, size_t window) const {
    if (metrics.empty()) {
        return 0.0;
    }

    size_t actual_window = std::min(window, metrics.size());
    auto start_it = metrics.end() - actual_window;

    double sum = 0.0;
    for (auto it = start_it; it != metrics.end(); ++it) {
        sum += it->value;
    }

    return sum / actual_window;
}

double QualityPredictor::exponentialSmoothing(const std::deque<QualityMetric>& metrics, double alpha) const {
    if (metrics.empty()) {
        return 0.0;
    }

    double smoothed = metrics[0].value;
    for (size_t i = 1; i < metrics.size(); ++i) {
        smoothed = alpha * metrics[i].value + (1.0 - alpha) * smoothed;
    }

    return smoothed;
}

// ============================================================================
// QualityLearner Implementation
// ============================================================================

QualityLearner::QualityLearner(const QualityManagerConfig& config)
    : learning_rate_(config.learning_rate),
      batch_size_(config.training_batch_size),
      model_file_path_(config.model_file_path) {
    initializeNetwork();
}

QualityLearner::~QualityLearner() {
}

void QualityLearner::addTrainingExample(
    const std::unordered_map<QualityMetricType, QualityStatistics>& features,
    const QualityAssessment& outcome) {

    training_data_.push_back({features, outcome});

    // Limit training data size
    if (training_data_.size() > 1000) {
        training_data_.erase(training_data_.begin());
    }
}

std::pair<AudioQualityLevel, double> QualityLearner::predict(
    const std::unordered_map<QualityMetricType, QualityStatistics>& features) {

    // Simple neural network prediction
    double output = forwardPass(features);

    // Convert output to quality level and score
    double score = std::max(0.0, std::min(100.0, output * 100.0));

    AudioQualityLevel level;
    if (score >= 90.0) {
        level = AudioQualityLevel::EXCELLENT;
    } else if (score >= 75.0) {
        level = AudioQualityLevel::GOOD;
    } else if (score >= 60.0) {
        level = AudioQualityLevel::FAIR;
    } else {
        level = AudioQualityLevel::POOR;
    }

    return {level, score};
}

void QualityLearner::train() {
    if (training_data_.size() < batch_size_) {
        return;
    }

    // Simple stochastic gradient descent
    for (size_t epoch = 0; epoch < 10; ++epoch) {
        for (const auto& example : training_data_) {
            double output = forwardPass(example.first);
            double target = example.second.quality_score / 100.0;
            backwardPass(example.first, target);
        }
    }
}

void QualityLearner::saveModel(const std::string& filename) const {
    // Save neural network weights to file
    // This is a simplified implementation
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "Neural Network Model\n";
        file << "Input-Hidden Weights:\n";
        for (const auto& row : model_.weights_input_hidden) {
            for (double weight : row) {
                file << weight << " ";
            }
            file << "\n";
        }
        file << "Hidden-Output Weights:\n";
        for (double weight : model_.weights_hidden_output) {
            file << weight << " ";
        }
        file << "\n";
        file.close();
    }
}

void QualityLearner::loadModel(const std::string& filename) {
    // Load neural network weights from file
    // This is a simplified implementation
    std::ifstream file(filename);
    if (file.is_open()) {
        std::string line;
        std::getline(file, line); // Skip header
        std::getline(file, line); // Skip section header

        // Load input-hidden weights (simplified)
        for (auto& row : model_.weights_input_hidden) {
            std::getline(file, line);
            std::istringstream iss(line);
            for (double& weight : row) {
                iss >> weight;
            }
        }

        file.close();
    }
}

double QualityLearner::forwardPass(const std::unordered_map<QualityMetricType, QualityStatistics>& features) {
    // Convert features to input vector
    std::vector<double> inputs;

    // Add key metrics as inputs (simplified)
    inputs.push_back(features.count(QualityMetricType::LATENCY) ?
                    features.at(QualityMetricType::LATENCY).average / 100.0 : 0.0);
    inputs.push_back(features.count(QualityMetricType::PACKET_LOSS) ?
                    features.at(QualityMetricType::PACKET_LOSS).average / 10.0 : 0.0);
    inputs.push_back(features.count(QualityMetricType::JITTER) ?
                    features.at(QualityMetricType::JITTER).average / 20.0 : 0.0);
    inputs.push_back(features.count(QualityMetricType::THROUGHPUT) ?
                    features.at(QualityMetricType::THROUGHPUT).average / 100.0 : 0.0);

    // Neural network forward pass (simplified single neuron)
    double sum = model_.output_bias;
    for (size_t i = 0; i < inputs.size() && i < model_.weights_input_hidden[0].size(); ++i) {
        sum += inputs[i] * model_.weights_input_hidden[0][i];
    }

    return sigmoid(sum);
}

void QualityLearner::backwardPass(const std::unordered_map<QualityMetricType, QualityStatistics>& features, double target) {
    // Simplified backpropagation
    double output = forwardPass(features);
    double error = target - output;

    // Update weights (simplified)
    for (size_t i = 0; i < model_.weights_input_hidden[0].size(); ++i) {
        double input = 0.0;
        switch (i) {
            case 0: input = features.count(QualityMetricType::LATENCY) ?
                          features.at(QualityMetricType::LATENCY).average / 100.0 : 0.0; break;
            case 1: input = features.count(QualityMetricType::PACKET_LOSS) ?
                          features.at(QualityMetricType::PACKET_LOSS).average / 10.0 : 0.0; break;
            case 2: input = features.count(QualityMetricType::JITTER) ?
                          features.at(QualityMetricType::JITTER).average / 20.0 : 0.0; break;
            case 3: input = features.count(QualityMetricType::THROUGHPUT) ?
                          features.at(QualityMetricType::THROUGHPUT).average / 100.0 : 0.0; break;
        }

        model_.weights_input_hidden[0][i] += learning_rate_ * error * input * sigmoidDerivative(output);
    }
}

double QualityLearner::sigmoid(double x) const {
    return 1.0 / (1.0 + std::exp(-x));
}

double QualityLearner::sigmoidDerivative(double x) const {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

void QualityLearner::initializeNetwork() {
    // Initialize simple neural network
    model_.weights_input_hidden.resize(1, std::vector<double>(4)); // 4 input features
    model_.weights_hidden_output.resize(4); // 4 hidden neurons
    model_.hidden_bias.resize(4, 0.0);
    model_.output_bias = 0.0;

    // Initialize weights with small random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    for (auto& row : model_.weights_input_hidden) {
        for (double& weight : row) {
            weight = dis(gen);
        }
    }

    for (double& weight : model_.weights_hidden_output) {
        weight = dis(gen);
    }
}

// ============================================================================
// QualityAnalyzer Implementation
// ============================================================================

QualityAnalyzer::QualityAnalyzer(const QualityManagerConfig& config)
    : thresholds_(config.thresholds) {
}

QualityAssessment QualityAnalyzer::analyze(
    const std::unordered_map<QualityMetricType, std::deque<QualityMetric>>& metrics,
    const QualityThresholds& thresholds) {

    QualityAssessment assessment;
    assessment.assessment_time = std::chrono::steady_clock::now();

    // Calculate statistics for each metric type
    for (const auto& pair : metrics) {
        QualityMetricType metric_type = pair.first;
        const auto& metric_values = pair.second;

        if (metric_values.empty()) {
            continue;
        }

        QualityStatistics& stats = assessment.metric_statistics[metric_type];
        stats.total_samples = metric_values.size();

        // Calculate basic statistics
        for (const auto& metric : metric_values) {
            if (metric.is_valid) {
                stats.min_value = std::min(stats.min_value, metric.value);
                stats.max_value = std::max(stats.max_value, metric.value);
                stats.average += metric.value;
            }
        }

        if (stats.total_samples > 0) {
            stats.average /= stats.total_samples;

            // Calculate standard deviation
            double sum_squared_diff = 0.0;
            for (const auto& metric : metric_values) {
                if (metric.is_valid) {
                    double diff = metric.value - stats.average;
                    sum_squared_diff += diff * diff;
                }
            }
            stats.standard_deviation = std::sqrt(sum_squared_diff / stats.total_samples);

            // Calculate percentiles
            std::vector<double> sorted_values;
            for (const auto& metric : metric_values) {
                if (metric.is_valid) {
                    sorted_values.push_back(metric.value);
                }
            }

            if (!sorted_values.empty()) {
                std::sort(sorted_values.begin(), sorted_values.end());
                stats.median = sorted_values[sorted_values.size() / 2];

                if (sorted_values.size() >= 20) {
                    size_t p95_index = static_cast<size_t>(sorted_values.size() * 0.95);
                    size_t p99_index = static_cast<size_t>(sorted_values.size() * 0.99);
                    stats.percentile_95 = sorted_values[std::min(p95_index, sorted_values.size() - 1)];
                    stats.percentile_99 = sorted_values[std::min(p99_index, sorted_values.size() - 1)];
                }
            }
        }

        stats.last_update = std::chrono::steady_clock::now();
    }

    // Identify issues
    assessment.issues = identifyIssues(assessment, thresholds);

    // Generate recommendations
    assessment.recommendations = generateRecommendations(assessment);

    // Calculate scores
    assessment.stability_score = calculateStabilityScore(metrics);
    assessment.efficiency_score = calculateEfficiencyScore(metrics);
    assessment.user_experience_score = calculateUserExperienceScore(assessment);

    // Determine overall quality level and score
    assessment.quality_score = (assessment.stability_score * 0.3 +
                               assessment.efficiency_score * 0.3 +
                               assessment.user_experience_score * 0.4);

    // Determine quality level based on score
    if (assessment.quality_score >= 90.0) {
        assessment.overall_quality = AudioQualityLevel::EXCELLENT;
    } else if (assessment.quality_score >= 75.0) {
        assessment.overall_quality = AudioQualityLevel::GOOD;
    } else if (assessment.quality_score >= 60.0) {
        assessment.overall_quality = AudioQualityLevel::FAIR;
    } else {
        assessment.overall_quality = AudioQualityLevel::POOR;
    }

    // Assess network condition
    auto latency_it = assessment.metric_statistics.find(QualityMetricType::LATENCY);
    auto jitter_it = assessment.metric_statistics.find(QualityMetricType::JITTER);
    auto loss_it = assessment.metric_statistics.find(QualityMetricType::PACKET_LOSS);

    double avg_latency = latency_it != assessment.metric_statistics.end() ? latency_it->second.average : 0.0;
    double avg_jitter = jitter_it != assessment.metric_statistics.end() ? jitter_it->second.average : 0.0;
    double avg_loss = loss_it != assessment.metric_statistics.end() ? loss_it->second.average : 0.0;

    if (avg_loss < 1.0 && avg_jitter < 5.0 && avg_latency < 10.0) {
        assessment.network_condition = NetworkCondition::EXCELLENT;
    } else if (avg_loss < 3.0 && avg_jitter < 10.0 && avg_latency < 25.0) {
        assessment.network_condition = NetworkCondition::GOOD;
    } else if (avg_loss < 5.0 && avg_jitter < 20.0 && avg_latency < 50.0) {
        assessment.network_condition = NetworkCondition::FAIR;
    } else if (avg_loss < 10.0 && avg_jitter < 50.0 && avg_latency < 100.0) {
        assessment.network_condition = NetworkCondition::POOR;
    } else {
        assessment.network_condition = NetworkCondition::VERY_POOR;
    }

    return assessment;
}

std::vector<std::string> QualityAnalyzer::identifyIssues(
    const QualityAssessment& assessment,
    const QualityThresholds& thresholds) {

    std::vector<std::string> issues;

    // Check latency
    auto latency_it = assessment.metric_statistics.find(QualityMetricType::LATENCY);
    if (latency_it != assessment.metric_statistics.end()) {
        if (latency_it->second.average > thresholds.critical_latency_ms) {
            issues.push_back("Critical: High latency detected (" +
                           std::to_string(latency_it->second.average) + "ms)");
        } else if (latency_it->second.average > thresholds.max_acceptable_latency_ms) {
            issues.push_back("Warning: Latency above acceptable threshold (" +
                           std::to_string(latency_it->second.average) + "ms)");
        }
    }

    // Check packet loss
    auto loss_it = assessment.metric_statistics.find(QualityMetricType::PACKET_LOSS);
    if (loss_it != assessment.metric_statistics.end()) {
        if (loss_it->second.average > thresholds.critical_packet_loss_percent) {
            issues.push_back("Critical: High packet loss detected (" +
                           std::to_string(loss_it->second.average) + "%)");
        } else if (loss_it->second.average > thresholds.max_acceptable_packet_loss_percent) {
            issues.push_back("Warning: Packet loss above acceptable threshold (" +
                           std::to_string(loss_it->second.average) + "%)");
        }
    }

    // Check jitter
    auto jitter_it = assessment.metric_statistics.find(QualityMetricType::JITTER);
    if (jitter_it != assessment.metric_statistics.end()) {
        if (jitter_it->second.average > thresholds.critical_jitter_ms) {
            issues.push_back("Critical: High jitter detected (" +
                           std::to_string(jitter_it->second.average) + "ms)");
        } else if (jitter_it->second.average > thresholds.max_acceptable_jitter_ms) {
            issues.push_back("Warning: Jitter above acceptable threshold (" +
                           std::to_string(jitter_it->second.average) + "ms)");
        }
    }

    // Check resource utilization
    auto cpu_it = assessment.metric_statistics.find(QualityMetricType::CPU_UTILIZATION);
    if (cpu_it != assessment.metric_statistics.end()) {
        if (cpu_it->second.average > thresholds.cpu_utilization_threshold) {
            issues.push_back("Warning: High CPU utilization (" +
                           std::to_string(cpu_it->second.average) + "%)");
        }
    }

    auto memory_it = assessment.metric_statistics.find(QualityMetricType::MEMORY_UTILIZATION);
    if (memory_it != assessment.metric_statistics.end()) {
        if (memory_it->second.average > thresholds.memory_utilization_threshold) {
            issues.push_back("Warning: High memory utilization (" +
                           std::to_string(memory_it->second.average) + "%)");
        }
    }

    return issues;
}

std::vector<std::string> QualityAnalyzer::generateRecommendations(const QualityAssessment& assessment) {
    std::vector<std::string> recommendations;

    // Quality-based recommendations
    if (assessment.overall_quality == AudioQualityLevel::POOR) {
        recommendations.push_back("Consider reducing bitrate to improve stability");
        recommendations.push_back("Increase buffer size to reduce underruns/overruns");
        recommendations.push_back("Check network connection quality");
    } else if (assessment.overall_quality == AudioQualityLevel::FAIR) {
        recommendations.push_back("Monitor network conditions closely");
        recommendations.push_back("Consider enabling adaptive bitrate");
    }

    // Metric-specific recommendations
    auto latency_it = assessment.metric_statistics.find(QualityMetricType::LATENCY);
    if (latency_it != assessment.metric_statistics.end() && latency_it->second.average > 50.0) {
        recommendations.push_back("Reduce buffer size for lower latency");
        recommendations.push_back("Consider using more efficient codec");
    }

    auto loss_it = assessment.metric_statistics.find(QualityMetricType::PACKET_LOSS);
    if (loss_it != assessment.metric_statistics.end() && loss_it->second.average > 2.0) {
        recommendations.push_back("Enable forward error correction (FEC)");
        recommendations.push_back("Reduce bitrate to match network capacity");
    }

    auto jitter_it = assessment.metric_statistics.find(QualityMetricType::JITTER);
    if (jitter_it != assessment.metric_statistics.end() && jitter_it->second.standard_deviation > 10.0) {
        recommendations.push_back("Increase jitter buffer size");
        recommendations.push_back("Use adaptive jitter buffer");
    }

    if (assessment.stability_score < 70.0) {
        recommendations.push_back("Enable network quality monitoring");
        recommendations.push_back("Consider redundant network paths");
    }

    return recommendations;
}

double QualityAnalyzer::calculateStabilityScore(
    const std::unordered_map<QualityMetricType, std::deque<QualityMetric>>& metrics) {

    double stability_score = 100.0;

    // Calculate stability based on metric variance
    for (const auto& pair : metrics) {
        const auto& metric_values = pair.second;
        if (metric_values.size() < 2) {
            continue;
        }

        double variance = calculateVariance(metric_values);
        double avg_value = 0.0;
        for (const auto& metric : metric_values) {
            avg_value += metric.value;
        }
        avg_value /= metric_values.size();

        // Normalize variance
        double normalized_variance = avg_value > 0 ? variance / (avg_value * avg_value) : 0.0;

        // Penalize high variance
        switch (pair.first) {
            case QualityMetricType::LATENCY:
                stability_score -= normalized_variance * 30.0;
                break;
            case QualityMetricType::PACKET_LOSS:
                stability_score -= normalized_variance * 50.0;
                break;
            case QualityMetricType::JITTER:
                stability_score -= normalized_variance * 40.0;
                break;
            default:
                stability_score -= normalized_variance * 20.0;
                break;
        }
    }

    return std::max(0.0, std::min(100.0, stability_score));
}

double QualityAnalyzer::calculateEfficiencyScore(
    const std::unordered_map<QualityMetricType, std::deque<QualityMetric>>& metrics) {

    double efficiency_score = 50.0; // Base score

    // Calculate efficiency based on resource utilization vs quality
    auto cpu_it = metrics.find(QualityMetricType::CPU_UTILIZATION);
    auto memory_it = metrics.find(QualityMetricType::MEMORY_UTILIZATION);
    auto throughput_it = metrics.find(QualityMetricType::THROUGHPUT);

    // CPU efficiency (lower is better)
    if (cpu_it != metrics.end() && !cpu_it->second.empty()) {
        double avg_cpu = 0.0;
        for (const auto& metric : cpu_it->second) {
            avg_cpu += metric.value;
        }
        avg_cpu /= cpu_it->second.size();

        if (avg_cpu < 50.0) {
            efficiency_score += 20.0;
        } else if (avg_cpu < 80.0) {
            efficiency_score += 10.0;
        }
    }

    // Memory efficiency
    if (memory_it != metrics.end() && !memory_it->second.empty()) {
        double avg_memory = 0.0;
        for (const auto& metric : memory_it->second) {
            avg_memory += metric.value;
        }
        avg_memory /= memory_it->second.size();

        if (avg_memory < 50.0) {
            efficiency_score += 15.0;
        } else if (avg_memory < 80.0) {
            efficiency_score += 5.0;
        }
    }

    // Throughput efficiency
    if (throughput_it != metrics.end() && !throughput_it->second.empty()) {
        double avg_throughput = 0.0;
        for (const auto& metric : throughput_it->second) {
            avg_throughput += metric.value;
        }
        avg_throughput /= throughput_it->second.size();

        if (avg_throughput > 10.0) {
            efficiency_score += 15.0;
        } else if (avg_throughput > 5.0) {
            efficiency_score += 5.0;
        }
    }

    return std::max(0.0, std::min(100.0, efficiency_score));
}

double QualityAnalyzer::calculateUserExperienceScore(const QualityAssessment& assessment) {
    double ux_score = assessment.quality_score;

    // Penalize for critical issues
    for (const auto& issue : assessment.issues) {
        if (issue.find("Critical") != std::string::npos) {
            ux_score -= 15.0;
        } else if (issue.find("Warning") != std::string::npos) {
            ux_score -= 5.0;
        }
    }

    // Bonus for good network conditions
    if (assessment.network_condition == NetworkCondition::EXCELLENT) {
        ux_score += 10.0;
    } else if (assessment.network_condition == NetworkCondition::GOOD) {
        ux_score += 5.0;
    }

    return std::max(0.0, std::min(100.0, ux_score));
}

double QualityAnalyzer::calculateVariance(const std::deque<QualityMetric>& metrics) {
    if (metrics.empty()) {
        return 0.0;
    }

    double mean = 0.0;
    for (const auto& metric : metrics) {
        mean += metric.value;
    }
    mean /= metrics.size();

    double sum_squared_diff = 0.0;
    for (const auto& metric : metrics) {
        double diff = metric.value - mean;
        sum_squared_diff += diff * diff;
    }

    return sum_squared_diff / metrics.size();
}

// ============================================================================
// QualityUtils Implementation
// ============================================================================

namespace QualityUtils {

std::string metricTypeToString(QualityMetricType type) {
    switch (type) {
        case QualityMetricType::LATENCY: return "Latency";
        case QualityMetricType::JITTER: return "Jitter";
        case QualityMetricType::PACKET_LOSS: return "Packet Loss";
        case QualityMetricType::THROUGHPUT: return "Throughput";
        case QualityMetricType::SIGNAL_TO_NOISE_RATIO: return "SNR";
        case QualityMetricType::TOTAL_HARMONIC_DISTORTION: return "THD";
        case QualityMetricType::DYNAMIC_RANGE: return "Dynamic Range";
        case QualityMetricType::PEAK_LEVEL: return "Peak Level";
        case QualityMetricType::RMS_LEVEL: return "RMS Level";
        case QualityMetricType::CLOCK_DRIFT: return "Clock Drift";
        case QualityMetricType::BUFFER_UNDERRUN: return "Buffer Underrun";
        case QualityMetricType::BUFFER_OVERRUN: return "Buffer Overrun";
        case QualityMetricType::CPU_UTILIZATION: return "CPU Utilization";
        case QualityMetricType::MEMORY_UTILIZATION: return "Memory Utilization";
        case QualityMetricType::NETWORK_UTILIZATION: return "Network Utilization";
        case QualityMetricType::ERROR_RATE: return "Error Rate";
        case QualityMetricType::QUALITY_INDEX: return "Quality Index";
        case QualityMetricType::USER_EXPERIENCE_SCORE: return "User Experience Score";
        default: return "Unknown";
    }
}

std::string qualityLevelToString(AudioQualityLevel level) {
    switch (level) {
        case AudioQualityLevel::POOR: return "Poor";
        case AudioQualityLevel::FAIR: return "Fair";
        case AudioQualityLevel::GOOD: return "Good";
        case AudioQualityLevel::EXCELLENT: return "Excellent";
        case AudioQualityLevel::STUDIO: return "Studio";
        default: return "Unknown";
    }
}

std::string networkConditionToString(NetworkCondition condition) {
    switch (condition) {
        case NetworkCondition::EXCELLENT: return "Excellent";
        case NetworkCondition::GOOD: return "Good";
        case NetworkCondition::FAIR: return "Fair";
        case NetworkCondition::POOR: return "Poor";
        case NetworkCondition::VERY_POOR: return "Very Poor";
        case NetworkCondition::UNKNOWN: return "Unknown";
        default: return "Unknown";
    }
}

std::string strategyToString(QualityAdaptationStrategy strategy) {
    switch (strategy) {
        case QualityAdaptationStrategy::MANUAL: return "Manual";
        case QualityAdaptationStrategy::AUTOMATIC_UP: return "Automatic Up";
        case QualityAdaptationStrategy::AUTOMATIC_DOWN: return "Automatic Down";
        case QualityAdaptationStrategy::FULL_AUTOMATIC: return "Full Automatic";
        case QualityAdaptationStrategy::PREDICTIVE: return "Predictive";
        case QualityAdaptationStrategy::LEARNING_BASED: return "Learning Based";
        case QualityAdaptationStrategy::HYBRID: return "Hybrid";
        default: return "Unknown";
    }
}

std::string eventTypeToString(QualityEventType type) {
    switch (type) {
        case QualityEventType::QUALITY_IMPROVED: return "Quality Improved";
        case QualityEventType::QUALITY_DEGRADED: return "Quality Degraded";
        case QualityEventType::ADAPTATION_TRIGGERED: return "Adaptation Triggered";
        case QualityEventType::THRESHOLD_BREACHED: return "Threshold Breached";
        case QualityEventType::NETWORK_CONDITION_CHANGED: return "Network Condition Changed";
        case QualityEventType::PREDICTION_MADE: return "Prediction Made";
        case QualityEventType::LEARNING_MODEL_UPDATED: return "Learning Model Updated";
        case QualityEventType::QUALITY_REPORT_GENERATED: return "Quality Report Generated";
        case QualityEventType::ERROR_DETECTED: return "Error Detected";
        case QualityEventType::WARNING_ISSUED: return "Warning Issued";
        default: return "Unknown";
    }
}

QualityMetricType stringToMetricType(const std::string& str) {
    if (str == "Latency" || str == "LATENCY") return QualityMetricType::LATENCY;
    if (str == "Jitter" || str == "JITTER") return QualityMetricType::JITTER;
    if (str == "Packet Loss" || str == "PACKET_LOSS") return QualityMetricType::PACKET_LOSS;
    if (str == "Throughput" || str == "THROUGHPUT") return QualityMetricType::THROUGHPUT;
    // ... add more mappings as needed
    return QualityMetricType::QUALITY_INDEX; // Default
}

AudioQualityLevel stringToQualityLevel(const std::string& str) {
    if (str == "Poor" || str == "POOR") return AudioQualityLevel::POOR;
    if (str == "Fair" || str == "FAIR") return AudioQualityLevel::FAIR;
    if (str == "Good" || str == "GOOD") return AudioQualityLevel::GOOD;
    if (str == "Excellent" || str == "EXCELLENT") return AudioQualityLevel::EXCELLENT;
    if (str == "Studio" || str == "STUDIO") return AudioQualityLevel::STUDIO;
    return AudioQualityLevel::GOOD; // Default
}

NetworkCondition stringToNetworkCondition(const std::string& str) {
    if (str == "Excellent" || str == "EXCELLENT") return NetworkCondition::EXCELLENT;
    if (str == "Good" || str == "GOOD") return NetworkCondition::GOOD;
    if (str == "Fair" || str == "FAIR") return NetworkCondition::FAIR;
    if (str == "Poor" || str == "POOR") return NetworkCondition::POOR;
    if (str == "Very Poor" || str == "VERY_POOR") return NetworkCondition::VERY_POOR;
    return NetworkCondition::UNKNOWN; // Default
}

QualityAdaptationStrategy stringToStrategy(const std::string& str) {
    if (str == "Manual" || str == "MANUAL") return QualityAdaptationStrategy::MANUAL;
    if (str == "Automatic Up" || str == "AUTOMATIC_UP") return QualityAdaptationStrategy::AUTOMATIC_UP;
    if (str == "Automatic Down" || str == "AUTOMATIC_DOWN") return QualityAdaptationStrategy::AUTOMATIC_DOWN;
    if (str == "Full Automatic" || str == "FULL_AUTOMATIC") return QualityAdaptationStrategy::FULL_AUTOMATIC;
    if (str == "Predictive" || str == "PREDICTIVE") return QualityAdaptationStrategy::PREDICTIVE;
    if (str == "Learning Based" || str == "LEARNING_BASED") return QualityAdaptationStrategy::LEARNING_BASED;
    if (str == "Hybrid" || str == "HYBRID") return QualityAdaptationStrategy::HYBRID;
    return QualityAdaptationStrategy::FULL_AUTOMATIC; // Default
}

double normalizeValue(double value, double min_val, double max_val) {
    if (max_val - min_val == 0) {
        return 0.0;
    }
    return (value - min_val) / (max_val - min_val);
}

double calculateMOS(double r_factor) {
    // E-model R-factor to MOS conversion
    if (r_factor < 0) return 1.0;
    if (r_factor > 100) return 4.5;

    if (r_factor < 50) {
        return 1.0 + (r_factor - 0) * 0.04;
    } else if (r_factor < 80) {
        return 3.0 + (r_factor - 50) * 0.033;
    } else {
        return 4.0 + (r_factor - 80) * 0.017;
    }
}

double calculateRFactor(double latency, double packet_loss, double jitter) {
    // Simplified R-factor calculation
    double r_factor = 94.2; // Base R-factor

    // Latency penalty
    r_factor -= latency * 0.024;

    // Packet loss penalty
    r_factor -= packet_loss * 2.5;

    // Jitter penalty
    r_factor -= jitter * 0.1;

    return std::max(0.0, std::min(100.0, r_factor));
}

double calculateEModel(double latency, double packet_loss, double jitter, double codec_mos) {
    double r_factor = calculateRFactor(latency, packet_loss, jitter);
    return calculateMOS(r_factor);
}

std::vector<double> smoothData(const std::deque<QualityMetric>& metrics, size_t window_size) {
    std::vector<double> smoothed;
    if (metrics.empty()) {
        return smoothed;
    }

    for (size_t i = 0; i < metrics.size(); ++i) {
        size_t start = (i >= window_size) ? i - window_size : 0;
        double sum = 0.0;
        size_t count = 0;

        for (size_t j = start; j <= i; ++j) {
            sum += metrics[j].value;
            count++;
        }

        smoothed.push_back(sum / count);
    }

    return smoothed;
}

std::pair<double, double> calculateConfidenceInterval(const std::deque<QualityMetric>& metrics, double confidence) {
    if (metrics.size() < 2) {
        return {0.0, 0.0};
    }

    // Calculate mean and standard deviation
    double mean = 0.0;
    for (const auto& metric : metrics) {
        mean += metric.value;
    }
    mean /= metrics.size();

    double variance = 0.0;
    for (const auto& metric : metrics) {
        variance += std::pow(metric.value - mean, 2);
    }
    variance /= (metrics.size() - 1);
    double std_dev = std::sqrt(variance);

    // Calculate confidence interval (simplified, assuming normal distribution)
    double z_score = 1.96; // For 95% confidence
    double margin_error = z_score * std_dev / std::sqrt(metrics.size());

    return {mean - margin_error, mean + margin_error};
}

bool isWithinThreshold(double value, double target, double tolerance_percent) {
    double tolerance = target * tolerance_percent;
    return std::abs(value - target) <= tolerance;
}

double calculateDeviation(double actual, double expected) {
    if (expected == 0) {
        return 0.0;
    }
    return std::abs((actual - expected) / expected) * 100.0;
}

std::string formatQualityValue(double value, QualityMetricType type) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    switch (type) {
        case QualityMetricType::LATENCY:
        case QualityMetricType::JITTER:
            oss << value << " ms";
            break;
        case QualityMetricType::PACKET_LOSS:
        case QualityMetricType::CPU_UTILIZATION:
        case QualityMetricType::MEMORY_UTILIZATION:
        case QualityMetricType::NETWORK_UTILIZATION:
        case QualityMetricType::ERROR_RATE:
            oss << value << " %";
            break;
        case QualityMetricType::THROUGHPUT:
            oss << value << " Mbps";
            break;
        case QualityMetricType::SIGNAL_TO_NOISE_RATIO:
        case QualityMetricType::DYNAMIC_RANGE:
        case QualityMetricType::PEAK_LEVEL:
        case QualityMetricType::RMS_LEVEL:
            oss << value << " dB";
            break;
        case QualityMetricType::TOTAL_HARMONIC_DISTORTION:
            oss << value << " %";
            break;
        default:
            oss << value;
            break;
    }

    return oss.str();
}

} // namespace QualityUtils

} // namespace Network
} // namespace VortexGPU