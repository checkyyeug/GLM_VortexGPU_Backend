#pragma once

#include "network/audio_streaming_protocol.hpp"
#include "network/audio_discovery.hpp"
#include "network/audio_quality_manager.hpp"
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

// Synchronization modes
enum class SyncMode {
    FREE_RUNNING,        // No external sync, internal clock only
    MASTER_SLAVE,        // One device acts as master, others as slaves
    PEER_TO_PEER,        // All devices are peers, negotiate timing
    IEEE_1588_PTP,       // Precision Time Protocol (PTP/IEEE 1588)
    NTP,                // Network Time Protocol
    GPS_DISCIPLINED,     // GPS-disciplined clock
    WORD_CLOCK,          // Word clock synchronization
    ADAT_SYNC,           // ADAT synchronization
    AES_SYNC,            // AES/EBU synchronization
    MIDI_CLOCK,          // MIDI clock synchronization
    SAMPLE_ACCURATE      // Sample-accurate synchronization
};

// Clock types
enum class ClockType {
    INTERNAL,            // Internal system clock
   _EXTERNAL_WORD_CLOCK, // External word clock input
    AES_EBU_CLOCK,      // AES/EBU clock
    ADAT_CLOCK,         // ADAT clock
    MIDI_CLOCK,         // MIDI clock
    PTP_CLOCK,          // IEEE 1588 PTP clock
    NTP_CLOCK,          // NTP-synchronized clock
    GPS_CLOCK,          // GPS-disciplined clock
    SOFTWARE_CLOCK      // Software-based clock
};

// Clock quality levels (IEEE 1588)
enum class ClockQuality {
    GRANDMASTER = 0,
    BOUNDARY_CLOCK = 1,
    ORDINARY_CLOCK = 2,
    SLAVE_ONLY = 3,
    PASSIVE = 4,
    UNKNOWN = 255
};

// Synchronization status
enum class SyncStatus {
    LOCKED,              // Synchronized and locked
    ACQUIRING,           // Acquiring synchronization
    HOLDOVER,            // In holdover mode (lost sync source)
    FREERUN,            // Free running (no sync source)
    FAULT,              // Fault condition
    UNKNOWN
};

// Sync source types
enum class SyncSourceType {
    INTERNAL_CLOCK,
    EXTERNAL_WORD_CLOCK,
    AES_EBU,
    ADAT,
    S/PDIF,
    MIDI,
    PTP_MASTER,
    NTP_SERVER,
    GPS,
    NETWORK_PEER,
    SOFTWARE_GENERATOR
};

// Sync lock parameters
struct SyncLockParameters {
    double lock_threshold_ppm = 10.0;        // Lock threshold in ppm
    double unlock_threshold_ppm = 50.0;      // Unlock threshold in ppm
    std::chrono::milliseconds lock_time{5000};   // Time to confirm lock
    std::chrono::milliseconds holdover_timeout{10000}; // Holdover timeout
    uint32_t min_lock_samples = 1000;        // Minimum samples to confirm lock
    double wander_filter_coefficient = 0.01; // Wander filter coefficient
    double jitter_filter_coefficient = 0.1;   // Jitter filter coefficient
};

// Clock adjustment parameters
struct ClockAdjustmentParameters {
    double max_adjustment_ppm = 100.0;      // Maximum adjustment rate
    double adjustment_rate = 0.1;            // Adjustment rate coefficient
    uint32_t adjustment_window_size = 1024;  // Window size for averaging
    bool enable_smooth_adjustment = true;     // Enable smooth adjustment
    double smooth_adjustment_alpha = 0.1;    // Smooth adjustment filter alpha
    bool enable_phase_adj = false;           // Enable phase adjustment
    double max_phase_correction_samples = 2.0; // Max phase correction in samples
};

// PTP (IEEE 1588) configuration
struct PTPConfiguration {
    uint8_t domain_number = 0;
    uint8_t priority1 = 128;
    uint8_t priority2 = 128;
    ClockQuality clock_quality = ClockQuality::ORDINARY_CLOCK;
    bool enable_two_step = true;
    std::chrono::milliseconds sync_interval{1000};
    std::chrono::milliseconds delay_request_interval{1000};
    uint16_t port_number = 319;              // PTP event port
    uint16_t general_port = 320;             // PTP general port
    std::vector<std::string> multicast_addresses = {"224.0.1.129"};
    bool enable_unicast = false;
    std::chrono::seconds announce_interval{2};
    std::chrono::seconds announce_receipt_timeout{3};
};

// Timestamp format
struct AudioTimestamp {
    uint64_t sample_count = 0;               // Sample position
    double seconds = 0.0;                    // Seconds (fractional)
    uint32_t nanoseconds = 0;                // Nanoseconds part
    std::chrono::system_clock::time_point system_time; // System time
    uint32_t timestamp_format_version = 1;   // Timestamp format version
    bool is_valid = true;                    // Timestamp validity flag

    AudioTimestamp() {
        system_time = std::chrono::system_clock::now();
    }

    double getTotalSeconds() const {
        return seconds + (nanoseconds / 1e9);
    }

    bool operator<(const AudioTimestamp& other) const {
        return getTotalSeconds() < other.getTotalSeconds();
    }

    bool operator==(const AudioTimestamp& other) const {
        return sample_count == other.sample_count &&
               getTotalSeconds() == other.getTotalSeconds();
    }
};

// Synchronization delay measurement
struct SyncDelayMeasurement {
    AudioTimestamp send_timestamp;
    AudioTimestamp receive_timestamp;
    double measured_delay_ms = 0.0;
    double jitter_ms = 0.0;
    bool is_valid = true;
    std::chrono::system_clock::time_point measurement_time;
};

// Clock statistics
struct ClockStatistics {
    ClockType clock_type = ClockType::INTERNAL;
    SyncStatus sync_status = SyncStatus::UNKNOWN;
    double clock_offset_ppm = 0.0;           // Clock offset in ppm
    double frequency_error_ppm = 0.0;        // Frequency error in ppm
    double phase_error_samples = 0.0;        // Phase error in samples
    double jitter_ms = 0.0;                  // Clock jitter
    uint32_t sync_loss_count = 0;            // Number of sync losses
    std::chrono::steady_clock::time_point last_sync_time;
    std::chrono::seconds total_sync_time{0};
    uint64_t total_samples_processed = 0;
    double average_drift_ppm = 0.0;
    std::chrono::milliseconds max_holdover_time{0};
};

// Sync event types
enum class SyncEventType {
    CLOCK_LOCKED,
    CLOCK_UNLOCKED,
    SYNC_SOURCE_CHANGED,
    MASTER_CHANGED,
    PHASE_ADJUSTMENT,
    FREQUENCY_ADJUSTMENT,
    HOLDOVER_ENTERED,
    HOLDOVER_EXITED,
    PTP_MASTER_SELECTED,
    PTP_MASTER_LOST,
    TIMESTAMP_RECEIVED,
    DELAY_MEASUREMENT_COMPLETE,
    SYNC_FAULT_DETECTED,
    SYNC_RECOVERY
};

// Sync event
struct SyncEvent {
    SyncEventType type;
    std::string stream_id;
    std::string message;
    AudioTimestamp timestamp;
    SyncSourceType old_source = SyncSourceType::INTERNAL_CLOCK;
    SyncSourceType new_source = SyncSourceType::INTERNAL_CLOCK;
    double old_offset_ppm = 0.0;
    double new_offset_ppm = 0.0;
    std::chrono::steady_clock::time_point event_time;
    std::string severity;
    std::map<std::string, std::string> metadata;
};

// Sync manager configuration
struct SyncManagerConfig {
    // General settings
    SyncMode default_mode = SyncMode::FREE_RUNNING;
    ClockType primary_clock = ClockType::INTERNAL;
    bool enable_cross_device_sync = false;
    bool enable_sample_accurate_sync = false;

    // Source priorities (lower number = higher priority)
    std::map<SyncSourceType, uint8_t> source_priorities = {
        {SyncSourceType::GPS, 1},
        {SyncSourceType::PTP_MASTER, 2},
        {SyncSourceType::EXTERNAL_WORD_CLOCK, 3},
        {SyncSourceType::AES_EBU, 4},
        {SyncSourceType::ADAT, 5},
        {SyncSourceType::S/PDIF, 6},
        {SyncSourceType::MIDI_CLOCK, 7},
        {SyncSourceType::NTP_SERVER, 8},
        {SyncSourceType::INTERNAL_CLOCK, 9},
        {SyncSourceType::SOFTWARE_GENERATOR, 10}
    };

    // Lock parameters
    SyncLockParameters lock_parameters;

    // Adjustment parameters
    ClockAdjustmentParameters adjustment_parameters;

    // PTP configuration
    PTPConfiguration ptp_config;

    // Timing configuration
    std::chrono::milliseconds sync_check_interval{100};
    std::chrono::milliseconds delay_measurement_interval{1000};
    std::chrono::seconds statistics_update_interval{5};
    uint32_t max_delay_measurements = 100;
    bool enable_wander_filtering = true;
    bool enable_jitter_filtering = true;

    // Redundancy and failover
    bool enable_primary_backup_sync = false;
    uint32_t max_sync_sources = 3;
    std::chrono::seconds source_failover_timeout{5};
    bool enable_automatic_source_switching = true;

    // Monitoring and logging
    bool enable_sync_monitoring = true;
    bool enable_detailed_logging = false;
    std::string log_file_path = "audio_sync.log";
    bool enable_sync_events = true;
    std::vector<std::string> event_subscribers;

    // Audio specific
    uint32_t sample_rate = 48000;
    uint16_t channels = 2;
    uint32_t buffer_size = 512;
    bool enable_drift_compensation = true;
    double max_drift_compensation_samples = 2.0;
    bool enable_sample_rate_conversion = false;
};

// Forward declarations
class PTPClock;
class NTPClock;
class ClockServo;
class SyncMonitor;
class SampleRateConverter;

// Main audio synchronization manager
class AudioSynchronizationManager {
public:
    AudioSynchronizationManager();
    ~AudioSynchronizationManager();

    // Initialization and lifecycle
    bool initialize(const SyncManagerConfig& config = {});
    void shutdown();
    bool isInitialized() const { return initialized_; }
    void reset();

    // Clock management
    bool setPrimaryClock(ClockType clock_type, SyncSourceType source_type = SyncSourceType::INTERNAL_CLOCK);
    ClockType getPrimaryClock() const;
    SyncSourceType getPrimarySource() const;
    std::vector<SyncSourceType> getAvailableSources() const;
    bool addSyncSource(SyncSourceType source_type, const std::string& source_info = "");
    bool removeSyncSource(SyncSourceType source_type);
    void setSourcePriority(SyncSourceType source_type, uint8_t priority);

    // Synchronization control
    bool startSynchronization(SyncMode mode);
    bool stopSynchronization();
    SyncMode getCurrentSyncMode() const;
    SyncStatus getSyncStatus() const;
    bool isSynchronized() const;

    // Stream synchronization
    bool addSynchronizedStream(const std::string& stream_id, uint32_t sample_rate = 48000);
    bool removeSynchronizedStream(const std::string& stream_id);
    std::vector<std::string> getSynchronizedStreams() const;
    bool setStreamClock(const std::string& stream_id, ClockType clock_type);
    ClockType getStreamClock(const std::string& stream_id) const;

    // Timestamp handling
    AudioTimestamp getCurrentTimestamp() const;
    AudioTimestamp generateTimestamp(uint64_t sample_count) const;
    bool synchronizeTimestamp(AudioTimestamp& timestamp);
    AudioTimestamp calculateStreamDelay(const std::string& stream_id) const;
    std::vector<SyncDelayMeasurement> getDelayMeasurements(const std::string& stream_id = "") const;

    // PTP (IEEE 1588) support
    bool enablePTP(const PTPConfiguration& config = {});
    bool disablePTP();
    bool isPTPEnabled() const;
    bool isPTPMaster() const;
    std::vector<std::string> getPTPPeers() const;

    // Sample rate conversion and alignment
    bool enableSampleRateConversion(const std::string& stream_id, double target_ratio = 1.0);
    bool disableSampleRateConversion(const std::string& stream_id);
    double getSampleRateRatio(const std::string& stream_id) const;
    bool alignSampleRates(const std::vector<std::string>& stream_ids);

    // Drift compensation
    bool enableDriftCompensation(const std::string& stream_id);
    bool disableDriftCompensation(const std::string& stream_id);
    double getDriftCompensation(const std::string& stream_id) const;
    void calibrateDrift(const std::string& stream_id, std::chrono::seconds duration = std::chrono::seconds{30});

    // Clock adjustment and servo control
    bool setClockAdjustment(double offset_ppm);
    double getClockAdjustment() const;
    bool enableClockServo(bool enabled = true);
    bool isClockServoEnabled() const;

    // Statistics and monitoring
    ClockStatistics getClockStatistics(ClockType clock_type = ClockType::INTERNAL) const;
    std::map<std::string, ClockStatistics> getAllStreamStatistics() const;
    std::vector<SyncDelayMeasurement> getDelayHistory(size_t max_measurements = 100) const;
    double getCurrentOffsetPPM() const;
    double getCurrentJitterMs() const;
    std::chrono::seconds getSyncUptime() const;

    // Configuration
    void updateConfig(const SyncManagerConfig& config);
    SyncManagerConfig getConfig() const { return config_; }
    void updateLockParameters(const SyncLockParameters& params);
    void updateAdjustmentParameters(const ClockAdjustmentParameters& params);
    void updatePTPConfiguration(const PTPConfiguration& config);

    // Event handling
    void setEventCallback(std::function<void(const SyncEvent&)> callback);
    void publishEvent(const SyncEvent& event);
    std::vector<SyncEvent> getRecentEvents(const std::string& stream_id = "", size_t max_events = 100) const;

    // Advanced features
    bool enableCrossDeviceSync(const std::vector<std::string>& device_ids);
    bool disableCrossDeviceSync();
    bool performPhaseAlignment(const std::vector<std::string>& stream_ids);
    bool performFrequencyAlignment(const std::vector<std::string>& stream_ids);
    bool switchSyncSource(SyncSourceType new_source);
    void enterHoldoverMode();
    void exitHoldoverMode();
    bool isInHoldover() const;

    // Diagnostics and testing
    std::string generateSyncReport() const;
    std::string generateDiagnosticReport() const;
    void performSyncTest(std::chrono::seconds duration = std::chrono::seconds{60});
    bool validateTimestampAccuracy();
    void calibrateClockServo();

private:
    // Core components
    std::unique_ptr<PTPClock> ptp_clock_;
    std::unique_ptr<NTPClock> ntp_clock_;
    std::unique_ptr<ClockServo> clock_servo_;
    std::unique_ptr<SyncMonitor> sync_monitor_;

    // Configuration and state
    SyncManagerConfig config_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> running_{false};

    // Synchronization state
    SyncMode current_sync_mode_ = SyncMode::FREE_RUNNING;
    ClockType primary_clock_type_ = ClockType::INTERNAL;
    SyncSourceType primary_source_type_ = SyncSourceType::INTERNAL_CLOCK;
    SyncStatus sync_status_ = SyncStatus::UNKNOWN;
    bool holdover_mode_ = false;

    // Stream data
    struct StreamSyncData {
        std::string stream_id;
        ClockType clock_type = ClockType::INTERNAL;
        uint32_t sample_rate = 48000;
        uint16_t channels = 2;
        uint64_t sample_count = 0;
        AudioTimestamp last_timestamp;
        std::deque<SyncDelayMeasurement> delay_measurements;
        double sample_rate_ratio = 1.0;
        bool src_enabled = false;
        bool drift_compensation_enabled = false;
        double drift_compensation_value = 0.0;
        ClockStatistics statistics;
        mutable std::mutex stream_mutex;
    };

    std::unordered_map<std::string, std::unique_ptr<StreamSyncData>> synchronized_streams_;

    // Sync sources management
    std::vector<SyncSourceType> available_sources_;
    std::unordered_map<SyncSourceType, std::string> source_info_;
    std::map<SyncSourceType, uint8_t> source_priorities_;

    // Global clock data
    ClockStatistics primary_clock_statistics_;
    std::deque<SyncDelayMeasurement> global_delay_measurements_;
    AudioTimestamp current_timestamp_;

    // Event system
    mutable std::mutex events_mutex_;
    std::queue<SyncEvent> event_queue_;
    std::vector<SyncEvent> event_history_;
    std::function<void(const SyncEvent&)> event_callback_;

    // Background threads
    std::thread sync_thread_;
    std::thread monitoring_thread_;
    std::thread timestamp_thread_;
    std::thread event_processing_thread_;

    // Timing
    std::chrono::steady_clock::time_point sync_start_time_;
    std::chrono::steady_clock::time_point last_sync_check_;

    // Internal methods
    void syncThread();
    void monitoringThread();
    void timestampThread();
    void eventProcessingThread();

    bool selectBestSyncSource();
    bool initializeClock(ClockType clock_type, SyncSourceType source_type);
    void updatePrimaryClock();
    void performClockAdjustment();
    void checkSyncHealth();

    AudioTimestamp generateTimestampInternal() const;
    bool synchronizeTimestampInternal(AudioTimestamp& timestamp);
    void measureDelay(const std::string& stream_id);
    void updateStreamStatistics(const std::string& stream_id);

    bool isSourceAvailable(SyncSourceType source_type) const;
    SyncSourceType getHighestPrioritySource() const;
    double calculateSourceQuality(SyncSourceType source_type) const;

    void recordEvent(const SyncEvent& event);
    void notifyEventSubscribers(const SyncEvent& event);

    std::string clockTypeToString(ClockType type) const;
    std::string sourceTypeToString(SyncSourceType type) const;
    std::string syncModeToString(SyncMode mode) const;
    std::string syncStatusToString(SyncStatus status) const;

    void writeToLog(const std::string& message);
    void saveConfiguration();
    void loadConfiguration();
};

// PTP (IEEE 1588) Clock implementation
class PTPClock {
public:
    PTPClock(const PTPConfiguration& config);
    ~PTPClock();

    bool initialize();
    void shutdown();
    bool isRunning() const { return running_; }

    bool becomeMaster();
    bool becomeSlave(const std::string& master_address);
    bool isMaster() const { return is_master_; }
    bool isSlave() const { return is_slave_; }

    AudioTimestamp getCurrentTime() const;
    void setTime(const AudioTimestamp& timestamp);
    bool synchronizeWith(const std::string& peer_address);

    void processSyncMessage(const uint8_t* message, size_t length);
    void processDelayReqMessage(const uint8_t* message, size_t length);
    void processFollowUpMessage(const uint8_t* message, size_t length);
    void processDelayRespMessage(const uint8_t* message, size_t length);

    std::vector<std::string> getDiscoveredPeers() const;
    PTPConfiguration getConfig() const { return config_; }

private:
    PTPConfiguration config_;
    std::atomic<bool> running_{false};
    std::atomic<bool> is_master_{false};
    std::atomic<bool> is_slave_{false};

    AudioTimestamp current_time_;
    double clock_offset_ppm_ = 0.0;
    std::chrono::steady_clock::time_point last_sync_;

    std::thread ptp_thread_;
    std::mutex ptp_mutex_;

    struct PTPPeer {
        std::string address;
        AudioTimestamp last_sync_time;
        double measured_delay_ms = 0.0;
        double measured_offset_ppm = 0.0;
        bool is_master = false;
    };

    std::unordered_map<std::string, PTPPeer> peers_;

    void ptpThread();
    void sendSyncMessage();
    void sendDelayReqMessage();
    void sendFollowUpMessage(const AudioTimestamp& original_timestamp);
    void sendDelayRespMessage(const std::string& requester_address, const AudioTimestamp& receive_timestamp);
    void announcePresence();

    void updateClockFromSync(const AudioTimestamp& master_time, const AudioTimestamp& slave_time);
    void calculatePeerDelay(const std::string& peer_address);
};

// NTP Clock implementation
class NTPClock {
public:
    NTPClock(const std::string& server_address);
    ~NTPClock();

    bool initialize();
    void shutdown();
    bool synchronize();
    bool isSynchronized() const { return synchronized_; }

    AudioTimestamp getCurrentTime() const;
    double getOffset() const { return offset_ms_; }
    double getJitter() const { return jitter_ms_; }

private:
    std::string server_address_;
    std::atomic<bool> synchronized_{false};
    double offset_ms_ = 0.0;
    double jitter_ms_ = 0.0;
    std::chrono::steady_clock::time_point last_sync_;

    struct NTPPacket {
        uint32_t leap_version_mode;
        uint8_t stratum;
        uint8_t poll;
        int8_t precision;
        uint32_t root_delay;
        uint32_t root_dispersion;
        uint32_t reference_id;
        uint64_t reference_timestamp;
        uint64_t originate_timestamp;
        uint64_t receive_timestamp;
        uint64_t transmit_timestamp;
    };

    bool sendNTPRequest();
    bool receiveNTPResponse(NTPPacket& packet);
    uint64_t getCurrentNTPTime() const;
    AudioTimestamp convertNTPToTimestamp(uint64_t ntp_time) const;
};

// Clock servo for precise clock control
class ClockServo {
public:
    ClockServo(const ClockAdjustmentParameters& params);
    ~ClockServo();

    void configure(const ClockAdjustmentParameters& params);
    void setTargetOffset(double offset_ppm);
    void updateMeasurement(double measured_offset_ppm, double jitter_ms);
    double getAdjustment() const;
    bool isLocked() const;
    void reset();

private:
    ClockAdjustmentParameters params_;
    double target_offset_ppm_ = 0.0;
    double current_adjustment_ppm_ = 0.0;
    double filter_state_ = 0.0;
    std::deque<double> measurement_history_;
    double accumulated_error_ = 0.0;
    std::chrono::steady_clock::time_point last_adjustment_;
    bool locked_ = false;

    void applyPIControl();
    void applyKalmanFilter(double measurement);
    double calculateLockStatus() const;
};

// Sync monitor for health and performance monitoring
class SyncMonitor {
public:
    SyncMonitor(const SyncManagerConfig& config);
    ~SyncMonitor();

    void updateClockStatistics(const ClockStatistics& stats);
    void recordDelayMeasurement(const SyncDelayMeasurement& measurement);
    void recordSyncEvent(const SyncEvent& event);

    std::vector<std::string> getHealthWarnings() const;
    double calculateSyncQuality() const;
    std::string generateHealthReport() const;

private:
    SyncManagerConfig config_;
    std::deque<ClockStatistics> statistics_history_;
    std::deque<SyncDelayMeasurement> delay_history_;
    std::deque<SyncEvent> event_history_;
    std::chrono::steady_clock::time_point last_analysis_;

    void analyzeTrends();
    double calculateJitterTrend() const;
    double calculateStabilityMetric() const;
};

// Sample rate converter for aligning different sample rates
class SampleRateConverter {
public:
    SampleRateConverter(uint32_t input_rate, uint32_t output_rate);
    ~SampleRateConverter();

    void setRatio(double ratio);
    double getRatio() const { return ratio_; }

    size_t convertSamples(const float* input, float* output, size_t input_samples);
    void reset();

private:
    uint32_t input_rate_;
    uint32_t output_rate_;
    double ratio_ = 1.0;
    double phase_accumulator_ = 0.0;
    std::vector<float> filter_coefficients_;
    size_t filter_order_;
    std::deque<float> input_history_;

    void initializeFilter();
    double interpolateSample(const std::deque<float>& history, double phase) const;
};

// Utility functions
namespace SyncUtils {
    std::string clockTypeToString(ClockType type);
    std::string sourceTypeToString(SyncSourceType type);
    std::string syncModeToString(SyncMode mode);
    std::string syncStatusToString(SyncStatus status);
    std::string eventTypeToString(SyncEventType type);

    ClockType stringToClockType(const std::string& str);
    SyncSourceType stringToSourceType(const std::string& str);
    SyncMode stringToSyncMode(const std::string& str);
    SyncStatus stringToSyncStatus(const std::string& str);
    SyncEventType stringToEventType(const std::string& str);

    AudioTimestamp createTimestamp(uint64_t sample_count, uint32_t sample_rate);
    double calculateClockOffset(const AudioTimestamp& local, const AudioTimestamp& remote);
    double calculateDelay(const AudioTimestamp& send_time, const AudioTimestamp& receive_time);
    bool compareTimestamps(const AudioTimestamp& a, const AudioTimestamp& b, double tolerance_ms = 1.0);

    double ppmToRatio(double ppm);
    double ratioToPPM(double ratio);
    double calculatePPMDifference(double frequency1, double frequency2);

    uint64_t timestampToNanoseconds(const AudioTimestamp& timestamp);
    AudioTimestamp nanosecondsToTimestamp(uint64_t nanoseconds);
    std::chrono::nanoseconds timestampToDuration(const AudioTimestamp& timestamp);
    AudioTimestamp durationToTimestamp(std::chrono::nanoseconds duration);

    double calculatePhaseAlignment(const std::vector<double>& phases);
    std::vector<double> performPhaseCorrection(const std::vector<double>& phases, double target_phase);
    bool validateTimestampSequence(const std::vector<AudioTimestamp>& timestamps);

    std::string formatTimestamp(const AudioTimestamp& timestamp);
    std::string formatDuration(std::chrono::nanoseconds duration);
    std::chrono::system_clock::time_point parseTimestamp(const std::string& timestamp_str);
}

} // namespace Network
} // namespace VortexGPU