#pragma once

#include "network/audio_streaming_protocol.hpp"
#include "network/audio_discovery.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <condition_variable>
#include <deque>
#include <array>

namespace VortexGPU {
namespace Network {

// Quality metrics types
enum class QualityMetricType {
    LATENCY,
    JITTER,
    PACKET_LOSS,
    THROUGHPUT,
    SIGNAL_TO_NOISE_RATIO,
    TOTAL_HARMONIC_DISTORTION,
    DYNAMIC_RANGE,
    PEAK_LEVEL,
    RMS_LEVEL,
    CLOCK_DRIFT,
    BUFFER_UNDERRUN,
    BUFFER_OVERRUN,
    CPU_UTILIZATION,
    MEMORY_UTILIZATION,
    NETWORK_UTILIZATION,
    ERROR_RATE,
    QUALITY_INDEX,
    USER_EXPERIENCE_SCORE
};

// Quality levels
enum class AudioQualityLevel {
    POOR = 0,
    FAIR = 1,
    GOOD = 2,
    EXCELLENT = 3,
    STUDIO = 4
};

// Quality adaptation strategies
enum class QualityAdaptationStrategy {
    MANUAL,           // Manual control only
    AUTOMATIC_UP,     // Only improve quality automatically
    AUTOMATIC_DOWN,   // Only reduce quality automatically
    FULL_AUTOMATIC,   // Full bidirectional adaptation
    PREDICTIVE,       // Predictive adaptation based on trends
    LEARNING_BASED,   // Machine learning based adaptation
    HYBRID            // Combination of strategies
};

// Quality management policies
enum class QualityPolicy {
    PRIORITY_STABILITY,    // Prioritize connection stability
    PRIORITY_QUALITY,      // Prioritize audio quality
    PRIORITY_LATENCY,      // Prioritize low latency
    PRIORITY_BANDWIDTH,    // Prioritize bandwidth efficiency
    BALANCED,              // Balanced approach
    ADAPTIVE               // Adaptive based on context
};

// Network condition categories
enum class NetworkCondition {
    EXCELLENT,       // < 1% loss, < 5ms jitter
    GOOD,           // < 3% loss, < 10ms jitter
    FAIR,           // < 5% loss, < 20ms jitter
    POOR,           // < 10% loss, < 50ms jitter
    VERY_POOR,      // >= 10% loss, >= 50ms jitter
    UNKNOWN
};

// Audio quality thresholds
struct QualityThresholds {
    // Latency thresholds (ms)
    double target_latency_ms = 10.0;
    double max_acceptable_latency_ms = 50.0;
    double critical_latency_ms = 100.0;

    // Jitter thresholds (ms)
    double target_jitter_ms = 2.0;
    double max_acceptable_jitter_ms = 10.0;
    double critical_jitter_ms = 20.0;

    // Packet loss thresholds (%)
    double target_packet_loss_percent = 0.1;
    double max_acceptable_packet_loss_percent = 1.0;
    double critical_packet_loss_percent = 5.0;

    // Throughput thresholds (Mbps)
    double target_throughput_mbps = 10.0;
    double min_throughput_mbps = 1.0;
    double critical_throughput_mbps = 0.5;

    // Audio quality thresholds
    double target_snr_db = 80.0;
    double min_acceptable_snr_db = 60.0;
    double target_thd_percent = 0.01;
    double max_acceptable_thd_percent = 0.1;

    // Buffer thresholds
    double buffer_underrun_threshold_percent = 5.0;
    double buffer_overrun_threshold_percent = 5.0;

    // Resource utilization thresholds (%)
    double cpu_utilization_threshold = 80.0;
    double memory_utilization_threshold = 80.0;
    double network_utilization_threshold = 80.0;
};

// Quality metric data
struct QualityMetric {
    QualityMetricType type;
    double value = 0.0;
    std::chrono::steady_clock::time_point timestamp;
    std::string unit;
    bool is_valid = true;
    std::map<std::string, std::string> metadata;

    QualityMetric() : timestamp(std::chrono::steady_clock::now()) {}
    QualityMetric(QualityMetricType t, double v, const std::string& u = "")
        : type(t), value(v), timestamp(std::chrono::steady_clock::now()), unit(u) {}
};

// Quality statistics
struct QualityStatistics {
    uint64_t total_samples = 0;
    double min_value = std::numeric_limits<double>::max();
    double max_value = std::numeric_limits<double>::lowest();
    double average = 0.0;
    double median = 0.0;
    double standard_deviation = 0.0;
    double percentile_95 = 0.0;
    double percentile_99 = 0.0;
    std::chrono::steady_clock::time_point last_update;

    void reset() {
        total_samples = 0;
        min_value = std::numeric_limits<double>::max();
        max_value = std::numeric_limits<double>::lowest();
        average = 0.0;
        median = 0.0;
        standard_deviation = 0.0;
        percentile_95 = 0.0;
        percentile_99 = 0.0;
        last_update = std::chrono::steady_clock::now();
    }
};

// Quality assessment result
struct QualityAssessment {
    AudioQualityLevel overall_quality = AudioQualityLevel::GOOD;
    NetworkCondition network_condition = NetworkCondition::UNKNOWN;
    double quality_score = 0.0;        // 0-100
    double stability_score = 0.0;       // 0-100
    double efficiency_score = 0.0;      // 0-100
    double user_experience_score = 0.0; // 0-100

    std::map<QualityMetricType, QualityStatistics> metric_statistics;
    std::vector<std::string> issues;
    std::vector<std::string> recommendations;
    std::chrono::steady_clock::time_point assessment_time;

    bool needs_improvement() const {
        return overall_quality < AudioQualityLevel::GOOD || quality_score < 70.0;
    }

    bool has_critical_issues() const {
        return std::any_of(issues.begin(), issues.end(),
                          [](const std::string& issue) {
                              return issue.find("Critical") != std::string::npos;
                          });
    }
};

// Quality adaptation parameters
struct QualityAdaptationParameters {
    // Bitrate adaptation
    std::vector<uint32_t> available_bitrates = {64000, 96000, 128000, 192000, 256000, 320000, 512000, 1024000, 1411200};
    size_t current_bitrate_index = 4; // Start at 256kbps

    // Sample rate adaptation
    std::vector<uint32_t> available_sample_rates = {44100, 48000, 88200, 96000, 176400, 192000};
    size_t current_sample_rate_index = 1; // Start at 48kHz

    // Buffer size adaptation
    std::vector<size_t> available_buffer_sizes = {64, 128, 256, 512, 1024, 2048, 4096};
    size_t current_buffer_size_index = 3; // Start at 512

    // Codec priority
    std::vector<AudioCodec> codec_priority = {AudioCodec::OPUS, AudioCodec::AAC, AudioCodec::MP3, AudioCodec::FLAC};

    // Adaptation timing
    std::chrono::seconds adaptation_cooldown = std::chrono::seconds(5);
    std::chrono::seconds quality_evaluation_interval = std::chrono::seconds(2);
    std::chrono::seconds trend_analysis_window = std::chrono::seconds(30);

    // Adaptation thresholds
    double up_quality_threshold = 0.8;      // Improve quality if score > 80%
    double down_quality_threshold = 0.6;    // Reduce quality if score < 60%
    double stability_factor = 0.1;          // Minimum improvement to justify adaptation

    // Adaptive parameters
    bool enable_bitrate_adaptation = true;
    bool enable_sample_rate_adaptation = true;
    bool enable_buffer_size_adaptation = true;
    bool enable_codec_adaptation = false;
    bool enable_predictive_adaptation = true;
};

// Quality manager configuration
struct QualityManagerConfig {
    // General settings
    bool enable_monitoring = true;
    bool enable_adaptation = true;
    bool enable_prediction = true;
    bool enable_learning = false;

    // Monitoring configuration
    std::chrono::seconds monitoring_interval = std::chrono::seconds(1);
    std::chrono::seconds statistics_window = std::chrono::minutes(5);
    size_t max_samples_per_metric = 1000;
    std::vector<QualityMetricType> enabled_metrics = {
        QualityMetricType::LATENCY,
        QualityMetricType::JITTER,
        QualityMetricType::PACKET_LOSS,
        QualityMetricType::THROUGHPUT,
        QualityMetricType::CPU_UTILIZATION,
        QualityMetricType::MEMORY_UTILIZATION
    };

    // Quality thresholds
    QualityThresholds thresholds;

    // Adaptation settings
    QualityAdaptationStrategy strategy = QualityAdaptationStrategy::FULL_AUTOMATIC;
    QualityPolicy policy = QualityPolicy::BALANCED;
    QualityAdaptationParameters adaptation_params;

    // Prediction settings
    std::chrono::seconds prediction_window = std::chrono::seconds(10);
    double prediction_confidence_threshold = 0.7;
    size_t prediction_history_size = 100;

    // Learning settings (if ML-based adaptation is enabled)
    std::string model_file_path = "";
    double learning_rate = 0.01;
    size_t training_batch_size = 32;
    std::chrono::seconds model_update_interval = std::chrono::hours(1);

    // Reporting and logging
    bool enable_quality_reports = true;
    std::chrono::seconds report_interval = std::chrono::minutes(5);
    std::string log_file_path = "audio_quality.log";
    bool enable_detailed_logging = false;

    // Event notifications
    bool enable_quality_events = true;
    std::vector<std::string> event_subscribers;
    double quality_change_threshold = 0.1; // Notify on 10% quality change
};

// Quality events
enum class QualityEventType {
    QUALITY_IMPROVED,
    QUALITY_DEGRADED,
    ADAPTATION_TRIGGERED,
    THRESHOLD_BREACHED,
    NETWORK_CONDITION_CHANGED,
    PREDICTION_MADE,
    LEARNING_MODEL_UPDATED,
    QUALITY_REPORT_GENERATED,
    ERROR_DETECTED,
    WARNING_ISSUED
};

// Quality event
struct QualityEvent {
    QualityEventType type;
    std::string stream_id;
    std::string message;
    AudioQualityLevel old_quality = AudioQualityLevel::GOOD;
    AudioQualityLevel new_quality = AudioQualityLevel::GOOD;
    double old_score = 0.0;
    double new_score = 0.0;
    std::chrono::steady_clock::time_point timestamp;
    std::string severity;
    std::map<std::string, std::string> details;
};

// Forward declarations
class QualityPredictor;
class QualityLearner;
class QualityAnalyzer;

// Main audio quality manager
class AudioQualityManager {
public:
    AudioQualityManager();
    ~AudioQualityManager();

    // Initialization and lifecycle
    bool initialize(const QualityManagerConfig& config = {});
    void shutdown();
    bool isInitialized() const { return initialized_; }
    void reset();

    // Stream management
    bool addStream(const std::string& stream_id, uint32_t initial_bitrate = 256000, uint32_t sample_rate = 48000);
    bool removeStream(const std::string& stream_id);
    bool isStreamMonitored(const std::string& stream_id) const;
    std::vector<std::string> getMonitoredStreams() const;

    // Metrics collection
    void reportMetric(const std::string& stream_id, QualityMetric metric);
    void reportLatency(const std::string& stream_id, double latency_ms);
    void reportJitter(const std::string& stream_id, double jitter_ms);
    void reportPacketLoss(const std::string& stream_id, double loss_percent);
    void reportThroughput(const std::string& stream_id, double throughput_mbps);
    void reportAudioLevel(const std::string& stream_id, double peak_level_db, double rms_level_db);
    void reportResourceUtilization(const std::string& stream_id, double cpu_percent, double memory_percent);
    void reportBufferEvent(const std::string& stream_id, bool underrun, bool overrun);

    // Quality assessment
    QualityAssessment assessQuality(const std::string& stream_id) const;
    std::map<std::string, QualityAssessment> assessAllStreams() const;
    AudioQualityLevel getCurrentQualityLevel(const std::string& stream_id) const;
    NetworkCondition getCurrentNetworkCondition(const std::string& stream_id) const;
    double getQualityScore(const std::string& stream_id) const;

    // Statistics and reporting
    QualityStatistics getMetricStatistics(const std::string& stream_id, QualityMetricType metric_type) const;
    std::vector<QualityMetric> getRecentMetrics(const std::string& stream_id, QualityMetricType metric_type, size_t count = 100) const;
    std::string generateQualityReport(const std::string& stream_id) const;
    std::string generateOverallReport() const;
    void exportMetrics(const std::string& stream_id, const std::string& filename) const;

    // Quality adaptation
    bool enableAdaptation(const std::string& stream_id, bool enabled = true);
    bool forceAdaptation(const std::string& stream_id, AudioQualityLevel target_quality);
    uint32_t getCurrentBitrate(const std::string& stream_id) const;
    uint32_t getCurrentSampleRate(const std::string& stream_id) const;
    size_t getCurrentBufferSize(const std::string& stream_id) const;
    AudioCodec getCurrentCodec(const std::string& stream_id) const;

    // Prediction and analysis
    std::pair<AudioQualityLevel, double> predictQuality(const std::string& stream_id, std::chrono::seconds future_duration) const;
    std::vector<std::string> getQualityRecommendations(const std::string& stream_id) const;
    std::map<std::string, double> analyzeTrends(const std::string& stream_id, std::chrono::seconds window) const;

    // Configuration management
    void updateConfig(const QualityManagerConfig& config);
    QualityManagerConfig getConfig() const { return config_; }
    void updateThresholds(const QualityThresholds& thresholds);
    QualityThresholds getThresholds() const { return config_.thresholds; }
    void setAdaptationStrategy(QualityAdaptationStrategy strategy);
    void setQualityPolicy(QualityPolicy policy);

    // Event handling
    void setEventCallback(std::function<void(const QualityEvent&)> callback);
    void publishEvent(const QualityEvent& event);
    std::vector<QualityEvent> getRecentEvents(const std::string& stream_id = "", size_t max_events = 100) const;

    // Health and diagnostics
    bool isHealthy(const std::string& stream_id) const;
    std::vector<std::string> getUnhealthyStreams() const;
    std::string getDiagnosticInfo(const std::string& stream_id) const;
    void performHealthCheck();
    void optimizeQuality();

private:
    // Core components
    std::unique_ptr<QualityPredictor> predictor_;
    std::unique_ptr<QualityLearner> learner_;
    std::unique_ptr<QualityAnalyzer> analyzer_;

    // Configuration and state
    QualityManagerConfig config_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> running_{false};

    // Stream-specific data
    struct StreamData {
        std::string stream_id;
        uint32_t current_bitrate;
        uint32_t current_sample_rate;
        size_t current_buffer_size;
        AudioCodec current_codec;
        bool adaptation_enabled = true;

        QualityAdaptationParameters adaptation_params;
        std::unordered_map<QualityMetricType, std::deque<QualityMetric>> metrics;
        std::unordered_map<QualityMetricType, QualityStatistics> statistics;

        QualityAssessment last_assessment;
        AudioQualityLevel current_quality = AudioQualityLevel::GOOD;
        double current_quality_score = 50.0;
        NetworkCondition current_network_condition = NetworkCondition::UNKNOWN;

        std::chrono::steady_clock::time_point last_adaptation;
        std::chrono::steady_clock::time_point last_assessment_time;
        uint32_t adaptation_count = 0;

        mutable std::mutex stream_mutex;
    };

    std::unordered_map<std::string, std::unique_ptr<StreamData>> streams_;

    // Global metrics and events
    mutable std::mutex global_mutex_;
    std::queue<QualityEvent> event_queue_;
    std::vector<QualityEvent> event_history_;
    std::function<void(const QualityEvent&)> event_callback_;

    // Background threads
    std::thread monitoring_thread_;
    std::thread assessment_thread_;
    std::thread adaptation_thread_;
    std::thread event_processing_thread_;

    // Internal methods
    void monitoringThread();
    void assessmentThread();
    void adaptationThread();
    void eventProcessingThread();

    void updateMetricStatistics(const std::string& stream_id, QualityMetricType metric_type);
    void assessStreamQuality(const std::string& stream_id);
    void adaptStreamQuality(const std::string& stream_id);
    void performPredictiveAdaptation(const std::string& stream_id);

    QualityLevel calculateQualityLevel(const QualityAssessment& assessment) const;
    NetworkCondition assessNetworkCondition(const StreamData& stream_data) const;
    double calculateQualityScore(const QualityAssessment& assessment) const;

    bool shouldAdaptUp(const std::string& stream_id) const;
    bool shouldAdaptDown(const std::string& stream_id) const;
    bool adaptBitrate(const std::string& stream_id, bool increase);
    bool adaptSampleRate(const std::string& stream_id, bool increase);
    bool adaptBufferSize(const std::string& stream_id, bool increase);
    bool adaptCodec(const std::string& stream_id);

    void recordEvent(const QualityEvent& event);
    void notifyEventSubscribers(const QualityEvent& event);

    std::string metricTypeToString(QualityMetricType type) const;
    std::string qualityLevelToString(AudioQualityLevel level) const;
    std::string strategyToString(QualityAdaptationStrategy strategy) const;

    void writeToLog(const std::string& message);
    void saveConfiguration();
    void loadConfiguration();
};

// Quality predictor for predictive adaptation
class QualityPredictor {
public:
    QualityPredictor(const QualityManagerConfig& config);
    ~QualityPredictor();

    std::pair<AudioQualityLevel, double> predictQuality(const std::unordered_map<QualityMetricType, std::deque<QualityMetric>>& metrics,
                                                        std::chrono::seconds future_duration) const;

    std::pair<double, double> predictLatency(const std::deque<QualityMetric>& latency_metrics,
                                            std::chrono::seconds future_duration) const;

    double predictPacketLoss(const std::deque<QualityMetric>& loss_metrics,
                            std::chrono::seconds future_duration) const;

    void updateModel(const std::unordered_map<QualityMetricType, std::deque<QualityMetric>>& metrics,
                    const QualityAssessment& actual_outcome);

private:
    std::chrono::seconds prediction_window_;
    double confidence_threshold_;
    size_t history_size_;

    double linearRegression(const std::deque<QualityMetric>& metrics, std::chrono::seconds future) const;
    double movingAverage(const std::deque<QualityMetric>& metrics, size_t window) const;
    double exponentialSmoothing(const std::deque<QualityMetric>& metrics, double alpha = 0.3) const;
};

// Quality learner for ML-based adaptation
class QualityLearner {
public:
    QualityLearner(const QualityManagerConfig& config);
    ~QualityLearner();

    void addTrainingExample(const std::unordered_map<QualityMetricType, QualityStatistics>& features,
                           const QualityAssessment& outcome);

    std::pair<AudioQualityLevel, double> predict(const std::unordered_map<QualityMetricType, QualityStatistics>& features);

    void train();
    void saveModel(const std::string& filename) const;
    void loadModel(const std::string& filename);

private:
    double learning_rate_;
    size_t batch_size_;
    std::string model_file_path_;

    // Simplified neural network representation
    struct NeuralNetwork {
        std::vector<std::vector<double>> weights_input_hidden;
        std::vector<double> weights_hidden_output;
        std::vector<double> hidden_bias;
        double output_bias;
    };

    NeuralNetwork model_;
    std::vector<std::pair<std::unordered_map<QualityMetricType, QualityStatistics>, QualityAssessment>> training_data_;

    double forwardPass(const std::unordered_map<QualityMetricType, QualityStatistics>& features);
    void backwardPass(const std::unordered_map<QualityMetricType, QualityStatistics>& features, double target);
    double sigmoid(double x) const;
    double sigmoidDerivative(double x) const;
    void initializeNetwork();
};

// Quality analyzer for detailed analysis
class QualityAnalyzer {
public:
    QualityAnalyzer(const QualityManagerConfig& config);

    QualityAssessment analyze(const std::unordered_map<QualityMetricType, std::deque<QualityMetric>>& metrics,
                             const QualityThresholds& thresholds);

    std::vector<std::string> identifyIssues(const QualityAssessment& assessment,
                                           const QualityThresholds& thresholds);

    std::vector<std::string> generateRecommendations(const QualityAssessment& assessment);

    double calculateStabilityScore(const std::unordered_map<QualityMetricType, std::deque<QualityMetric>>& metrics);

    double calculateEfficiencyScore(const std::unordered_map<QualityMetricType, std::deque<QualityMetric>>& metrics);

    double calculateUserExperienceScore(const QualityAssessment& assessment);

private:
    QualityThresholds thresholds_;

    double calculateVariance(const std::deque<QualityMetric>& metrics);
    double calculateTrend(const std::deque<QualityMetric>& metrics);
    std::map<double, double> calculateDistribution(const std::deque<QualityMetric>& metrics);
    double calculatePercentile(const std::deque<QualityMetric>& metrics, double percentile);
};

// Utility functions
namespace QualityUtils {
    std::string metricTypeToString(QualityMetricType type);
    std::string qualityLevelToString(AudioQualityLevel level);
    std::string networkConditionToString(NetworkCondition condition);
    std::string strategyToString(QualityAdaptationStrategy strategy);
    std::string eventTypeToString(QualityEventType type);

    QualityMetricType stringToMetricType(const std::string& str);
    AudioQualityLevel stringToQualityLevel(const std::string& str);
    NetworkCondition stringToNetworkCondition(const std::string& str);
    QualityAdaptationStrategy stringToStrategy(const std::string& str);

    double normalizeValue(double value, double min_val, double max_val);
    double calculateMOS(double r_factor); // Mean Opinion Score from R-factor
    double calculateRFactor(double latency, double packet_loss, double jitter);
    double calculateEModel(double latency, double packet_loss, double jitter, double codec_mos);

    std::vector<double> smoothData(const std::deque<QualityMetric>& metrics, size_t window_size = 5);
    std::pair<double, double> calculateConfidenceInterval(const std::deque<QualityMetric>& metrics, double confidence = 0.95);

    bool isWithinThreshold(double value, double target, double tolerance_percent = 0.1);
    double calculateDeviation(double actual, double expected);
    std::string formatQualityValue(double value, QualityMetricType type);
}

} // namespace Network
} // namespace VortexGPU