#include "network/audio_synchronization.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cstring>

namespace VortexGPU {
namespace Network {

// ============================================================================
// AudioSynchronizationManager Implementation
// ============================================================================

AudioSynchronizationManager::AudioSynchronizationManager() {
    sync_start_time_ = std::chrono::steady_clock::now();
    last_sync_check_ = sync_start_time_;
    current_timestamp_ = AudioTimestamp{};
}

AudioSynchronizationManager::~AudioSynchronizationManager() {
    shutdown();
}

bool AudioSynchronizationManager::initialize(const SyncManagerConfig& config) {
    if (initialized_.load()) {
        return true;
    }

    config_ = config;

    try {
        // Initialize primary clock
        if (!initializeClock(config_.primary_clock, config_.primary_clock)) {
            std::cerr << "Failed to initialize primary clock" << std::endl;
            return false;
        }

        // Initialize clock servo
        clock_servo_ = std::make_unique<ClockServo>(config_.adjustment_parameters);

        // Initialize sync monitor
        sync_monitor_ = std::make_unique<SyncMonitor>(config_);

        // Initialize available sync sources based on configuration
        available_sources_.clear();
        source_priorities_ = config_.source_priorities;

        running_.store(true);

        // Start background threads
        sync_thread_ = std::thread(&AudioSynchronizationManager::syncThread, this);
        monitoring_thread_ = std::thread(&AudioSynchronizationManager::monitoringThread, this);
        timestamp_thread_ = std::thread(&AudioSynchronizationManager::timestampThread, this);
        event_processing_thread_ = std::thread(&AudioSynchronizationManager::eventProcessingThread, this);

        // Initialize logging
        if (!config_.log_file_path.empty()) {
            std::ofstream log_file(config_.log_file_path, std::ios::app);
            if (log_file.is_open()) {
                log_file << "\n=== Audio Synchronization Manager Started at "
                        << std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::system_clock::now().time_since_epoch()).count() << " ===\n";
                log_file.close();
            }
        }

        initialized_.store(true);

        std::cout << "AudioSynchronizationManager initialized with "
                  << clockTypeToString(config_.primary_clock) << " as primary clock" << std::endl;

        // Record initialization event
        SyncEvent event;
        event.type = SyncEventType::CLOCK_LOCKED;
        event.message = "Audio synchronization manager initialized";
        event.timestamp = current_timestamp_;
        event.event_time = std::chrono::steady_clock::now();
        event.severity = "info";
        event.new_source = config_.primary_clock;
        recordEvent(event);

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize AudioSynchronizationManager: " << e.what() << std::endl;
        shutdown();
        return false;
    }
}

void AudioSynchronizationManager::shutdown() {
    if (!initialized_.load()) {
        return;
    }

    running_.store(false);

    // Wait for threads to finish
    if (sync_thread_.joinable()) {
        sync_thread_.join();
    }

    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }

    if (timestamp_thread_.joinable()) {
        timestamp_thread_.join();
    }

    if (event_processing_thread_.joinable()) {
        event_processing_thread_.join();
    }

    // Clear all streams
    synchronized_streams_.clear();

    // Shutdown components
    if (ptp_clock_) {
        ptp_clock_->shutdown();
        ptp_clock_.reset();
    }

    if (ntp_clock_) {
        ntp_clock_.reset();
    }

    if (clock_servo_) {
        clock_servo_.reset();
    }

    if (sync_monitor_) {
        sync_monitor_.reset();
    }

    // Log shutdown
    if (!config_.log_file_path.empty()) {
        std::ofstream log_file(config_.log_file_path, std::ios::app);
        if (log_file.is_open()) {
            log_file << "\n=== Audio Synchronization Manager Stopped at "
                    << std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count() << " ===\n";
            log_file.close();
        }
    }

    initialized_.load() = false;

    std::cout << "AudioSynchronizationManager shut down" << std::endl;
}

void AudioSynchronizationManager::reset() {
    std::lock_guard<std::mutex> events_lock(events_mutex_);

    // Reset all stream data
    for (auto& pair : synchronized_streams_) {
        std::lock_guard<std::mutex> stream_lock(pair.second->stream_mutex);
        pair.second->sample_count = 0;
        pair.second->last_timestamp = AudioTimestamp{};
        pair.second->delay_measurements.clear();
        pair.second->sample_rate_ratio = 1.0;
        pair.second->drift_compensation_value = 0.0;
        pair.second->statistics = ClockStatistics{};
    }

    // Reset global state
    current_timestamp_ = AudioTimestamp{};
    sync_status_ = SyncStatus::UNKNOWN;
    holdover_mode_ = false;
    global_delay_measurements_.clear();
    event_queue_ = {};
    event_history_.clear();

    // Reset clock servo
    if (clock_servo_) {
        clock_servo_->reset();
    }

    // Reset timing
    sync_start_time_ = std::chrono::steady_clock::now();
    last_sync_check_ = sync_start_time_;

    writeToLog("Audio Synchronization Manager reset");
}

bool AudioSynchronizationManager::setPrimaryClock(ClockType clock_type, SyncSourceType source_type) {
    std::lock_guard<std::mutex> lock(events_mutex_);

    ClockType old_clock_type = primary_clock_type_;
    SyncSourceType old_source_type = primary_source_type_;

    // Initialize the new clock
    if (!initializeClock(clock_type, source_type)) {
        writeToLog("Failed to initialize new primary clock: " + clockTypeToString(clock_type));
        return false;
    }

    primary_clock_type_ = clock_type;
    primary_source_type_ = source_type;

    // Update clock servo target
    if (clock_servo_) {
        clock_servo_->setTargetOffset(0.0);
    }

    // Record event
    SyncEvent event;
    event.type = SyncEventType::SYNC_SOURCE_CHANGED;
    event.message = "Primary clock changed";
    event.timestamp = current_timestamp_;
    event.event_time = std::chrono::steady_clock::now();
    event.severity = "info";
    event.old_source = old_source_type;
    event.new_source = source_type;
    recordEvent(event);

    writeToLog("Primary clock changed: " + clockTypeToString(old_clock_type) + " -> " + clockTypeToString(clock_type));

    return true;
}

ClockType AudioSynchronizationManager::getPrimaryClock() const {
    std::lock_guard<std::mutex> lock(events_mutex_);
    return primary_clock_type_;
}

SyncSourceType AudioSynchronizationManager::getPrimarySource() const {
    std::lock_guard<std::mutex> lock(events_mutex_);
    return primary_source_type_;
}

std::vector<SyncSourceType> AudioSynchronizationManager::getAvailableSources() const {
    std::lock_guard<std::mutex> lock(events_mutex_);
    return available_sources_;
}

bool AudioSynchronizationManager::addSyncSource(SyncSourceType source_type, const std::string& source_info) {
    std::lock_guard<std::mutex> lock(events_mutex_);

    if (std::find(available_sources_.begin(), available_sources_.end(), source_type) != available_sources_.end()) {
        return false; // Source already exists
    }

    available_sources_.push_back(source_type);
    source_info_[source_type] = source_info;

    // Try to initialize the new source if it's higher priority than current
    if (selectBestSyncSource()) {
        updatePrimaryClock();
    }

    writeToLog("Added sync source: " + sourceTypeToString(source_type));
    return true;
}

bool AudioSynchronizationManager::removeSyncSource(SyncSourceType source_type) {
    std::lock_guard<std::mutex> lock(events_mutex_);

    auto it = std::find(available_sources_.begin(), available_sources_.end(), source_type);
    if (it == available_sources_.end()) {
        return false; // Source doesn't exist
    }

    // Can't remove if it's the current primary source and no backup available
    if (primary_source_type_ == source_type && available_sources_.size() == 1) {
        return false;
    }

    available_sources_.erase(it);
    source_info_.erase(source_type);

    // Switch to best available source if we removed the current primary
    if (primary_source_type_ == source_type) {
        if (selectBestSyncSource()) {
            updatePrimaryClock();
        }
    }

    writeToLog("Removed sync source: " + sourceTypeToString(source_type));
    return true;
}

void AudioSynchronizationManager::setSourcePriority(SyncSourceType source_type, uint8_t priority) {
    std::lock_guard<std::mutex> lock(events_mutex_);
    source_priorities_[source_type] = priority;

    // Re-evaluate best source if priority changed
    if (selectBestSyncSource()) {
        updatePrimaryClock();
    }

    writeToLog("Updated priority for source " + sourceTypeToString(source_type) + ": " + std::to_string(priority));
}

bool AudioSynchronizationManager::startSynchronization(SyncMode mode) {
    std::lock_guard<std::mutex> lock(events_mutex_);

    current_sync_mode_ = mode;
    sync_status_ = SyncStatus::ACQUIRING;

    // Initialize mode-specific components
    switch (mode) {
        case SyncMode::IEEE_1588_PTP:
            if (!enablePTP(config_.ptp_config)) {
                return false;
            }
            break;

        case SyncMode::NTP:
            // Initialize NTP clock
            break;

        case SyncMode::PEER_TO_PEER:
            // Initialize peer discovery
            break;

        case SyncMode::MASTER_SLAVE:
            // Initialize master/slave relationship
            break;

        default:
            break;
    }

    // Start clock servo
    if (clock_servo_) {
        clock_servo_->configure(config_.adjustment_parameters);
    }

    // Record event
    SyncEvent event;
    event.type = SyncEventType::CLOCK_LOCKED;
    event.message = "Synchronization started";
    event.timestamp = current_timestamp_;
    event.event_time = std::chrono::steady_clock::now();
    event.severity = "info";
    event.new_source = primary_source_type_;
    recordEvent(event);

    writeToLog("Synchronization started: " + syncModeToString(mode));
    return true;
}

bool AudioSynchronizationManager::stopSynchronization() {
    std::lock_guard<std::mutex> lock(events_mutex_);

    // Shutdown mode-specific components
    if (current_sync_mode_ == SyncMode::IEEE_1588_PTP) {
        disablePTP();
    }

    current_sync_mode_ = SyncMode::FREE_RUNNING;
    sync_status_ = SyncStatus::FREERUN;

    // Reset clock servo
    if (clock_servo_) {
        clock_servo_->reset();
    }

    // Record event
    SyncEvent event;
    event.type = SyncEventType::CLOCK_UNLOCKED;
    event.message = "Synchronization stopped";
    event.timestamp = current_timestamp_;
    event.event_time = std::chrono::steady_clock::now();
    event.severity = "info";
    event.old_source = primary_source_type_;
    recordEvent(event);

    writeToLog("Synchronization stopped");
    return true;
}

SyncMode AudioSynchronizationManager::getCurrentSyncMode() const {
    std::lock_guard<std::mutex> lock(events_mutex_);
    return current_sync_mode_;
}

SyncStatus AudioSynchronizationManager::getSyncStatus() const {
    std::lock_guard<std::mutex> lock(events_mutex_);
    return sync_status_;
}

bool AudioSynchronizationManager::isSynchronized() const {
    std::lock_guard<std::mutex> lock(events_mutex_);
    return sync_status_ == SyncStatus::LOCKED || sync_status_ == SyncStatus::HOLDOVER;
}

bool AudioSynchronizationManager::addSynchronizedStream(const std::string& stream_id, uint32_t sample_rate) {
    if (stream_id.empty()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(events_mutex_);

    if (synchronized_streams_.find(stream_id) != synchronized_streams_.end()) {
        return false; // Stream already exists
    }

    auto stream_data = std::make_unique<StreamSyncData>();
    stream_data->stream_id = stream_id;
    stream_data->clock_type = primary_clock_type_;
    stream_data->sample_rate = sample_rate;
    stream_data->channels = config_.channels;
    stream_data->last_timestamp = current_timestamp_;

    synchronized_streams_[stream_id] = std::move(stream_data);

    // Record event
    SyncEvent event;
    event.type = SyncEventType::CLOCK_LOCKED;
    event.stream_id = stream_id;
    event.message = "Stream added to synchronization: " + stream_id;
    event.timestamp = current_timestamp_;
    event.event_time = std::chrono::steady_clock::now();
    event.severity = "info";
    event.new_source = primary_source_type_;
    recordEvent(event);

    writeToLog("Added synchronized stream: " + stream_id + " (sample_rate: " + std::to_string(sample_rate) + ")");
    return true;
}

bool AudioSynchronizationManager::removeSynchronizedStream(const std::string& stream_id) {
    std::lock_guard<std::mutex> lock(events_mutex_);

    auto it = synchronized_streams_.find(stream_id);
    if (it == synchronized_streams_.end()) {
        return false;
    }

    // Record event
    SyncEvent event;
    event.type = SyncEventType::CLOCK_UNLOCKED;
    event.stream_id = stream_id;
    event.message = "Stream removed from synchronization: " + stream_id;
    event.timestamp = current_timestamp_;
    event.event_time = std::chrono::steady_clock::now();
    event.severity = "info";
    event.old_source = it->second->clock_type;
    recordEvent(event);

    synchronized_streams_.erase(it);
    writeToLog("Removed synchronized stream: " + stream_id);
    return true;
}

std::vector<std::string> AudioSynchronizationManager::getSynchronizedStreams() const {
    std::lock_guard<std::mutex> lock(events_mutex_);

    std::vector<std::string> stream_ids;
    stream_ids.reserve(synchronized_streams_.size());

    for (const auto& pair : synchronized_streams_) {
        stream_ids.push_back(pair.first);
    }

    return stream_ids;
}

bool AudioSynchronizationManager::setStreamClock(const std::string& stream_id, ClockType clock_type) {
    std::lock_guard<std::mutex> lock(events_mutex_);

    auto it = synchronized_streams_.find(stream_id);
    if (it == synchronized_streams_.end()) {
        return false;
    }

    std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);
    it->second->clock_type = clock_type;

    writeToLog("Set clock for stream " + stream_id + ": " + clockTypeToString(clock_type));
    return true;
}

ClockType AudioSynchronizationManager::getStreamClock(const std::string& stream_id) const {
    std::lock_guard<std::mutex> lock(events_mutex_);

    auto it = synchronized_streams_.find(stream_id);
    if (it != synchronized_streams_.end()) {
        std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);
        return it->second->clock_type;
    }

    return ClockType::INTERNAL;
}

AudioTimestamp AudioSynchronizationManager::getCurrentTimestamp() const {
    std::lock_guard<std::mutex> lock(events_mutex_);
    return current_timestamp_;
}

AudioTimestamp AudioSynchronizationManager::generateTimestamp(uint64_t sample_count) const {
    std::lock_guard<std::mutex> lock(events_mutex_);

    AudioTimestamp timestamp;
    timestamp.sample_count = sample_count;
    timestamp.seconds = static_cast<double>(sample_count) / config_.sample_rate;
    timestamp.nanoseconds = static_cast<uint32_t>(
        (timestamp.seconds - std::floor(timestamp.seconds)) * 1e9);
    timestamp.system_time = std::chrono::system_clock::now();
    timestamp.timestamp_format_version = 1;
    timestamp.is_valid = true;

    return timestamp;
}

bool AudioSynchronizationManager::synchronizeTimestamp(AudioTimestamp& timestamp) {
    return synchronizeTimestampInternal(timestamp);
}

AudioTimestamp AudioSynchronizationManager::calculateStreamDelay(const std::string& stream_id) const {
    std::lock_guard<std::mutex> lock(events_mutex_);

    auto it = synchronized_streams_.find(stream_id);
    if (it == synchronized_streams_.end()) {
        return AudioTimestamp{};
    }

    std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);
    const auto& delay_measurements = it->second->delay_measurements;

    if (delay_measurements.empty()) {
        return AudioTimestamp{};
    }

    // Calculate average delay
    double total_delay = 0.0;
    for (const auto& measurement : delay_measurements) {
        total_delay += measurement.measured_delay_ms;
    }

    double average_delay = total_delay / delay_measurements.size();
    return SyncUtils::durationToTimestamp(std::chrono::milliseconds(
        static_cast<int64_t>(average_delay)));
}

std::vector<SyncDelayMeasurement> AudioSynchronizationManager::getDelayMeasurements(const std::string& stream_id) const {
    std::lock_guard<std::mutex> lock(events_mutex_);

    if (stream_id.empty()) {
        return std::vector<SyncDelayMeasurement>(global_delay_measurements_.begin(),
                                                 global_delay_measurements_.end());
    }

    auto it = synchronized_streams_.find(stream_id);
    if (it != synchronized_streams_.end()) {
        std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);
        return std::vector<SyncDelayMeasurement>(it->second->delay_measurements.begin(),
                                                 it->second->delay_measurements.end());
    }

    return {};
}

bool AudioSynchronizationManager::enablePTP(const PTPConfiguration& config) {
    try {
        ptp_clock_ = std::make_unique<PTPClock>(config);
        if (!ptp_clock_->initialize()) {
            ptp_clock_.reset();
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to enable PTP: " << e.what() << std::endl;
        return false;
    }
}

bool AudioSynchronizationManager::disablePTP() {
    if (ptp_clock_) {
        ptp_clock_->shutdown();
        ptp_clock_.reset();
        return true;
    }
    return false;
}

bool AudioSynchronizationManager::isPTPEnabled() const {
    return ptp_clock_ && ptp_clock_->isRunning();
}

bool AudioSynchronizationManager::isPTPMaster() const {
    return ptp_clock_ && ptp_clock_->isMaster();
}

std::vector<std::string> AudioSynchronizationManager::getPTPPeers() const {
    if (ptp_clock_) {
        return ptp_clock_->getDiscoveredPeers();
    }
    return {};
}

bool AudioSynchronizationManager::enableSampleRateConversion(const std::string& stream_id, double target_ratio) {
    std::lock_guard<std::mutex> lock(events_mutex_);

    auto it = synchronized_streams_.find(stream_id);
    if (it == synchronized_streams_.end()) {
        return false;
    }

    std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);
    it->second->src_enabled = true;
    it->second->sample_rate_ratio = target_ratio;

    writeToLog("Enabled sample rate conversion for " + stream_id + " (ratio: " + std::to_string(target_ratio) + ")");
    return true;
}

bool AudioSynchronizationManager::disableSampleRateConversion(const std::string& stream_id) {
    std::lock_guard<std::mutex> lock(events_mutex_);

    auto it = synchronized_streams_.find(stream_id);
    if (it == synchronized_streams_.end()) {
        return false;
    }

    std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);
    it->second->src_enabled = false;
    it->second->sample_rate_ratio = 1.0;

    writeToLog("Disabled sample rate conversion for " + stream_id);
    return true;
}

double AudioSynchronizationManager::getSampleRateRatio(const std::string& stream_id) const {
    std::lock_guard<std::mutex> lock(events_mutex_);

    auto it = synchronized_streams_.find(stream_id);
    if (it != synchronized_streams_.end()) {
        std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);
        return it->second->sample_rate_ratio;
    }

    return 1.0;
}

bool AudioSynchronizationManager::alignSampleRates(const std::vector<std::string>& stream_ids) {
    // This would implement complex sample rate alignment logic
    // For now, just return true as a placeholder
    writeToLog("Sample rate alignment requested for " + std::to_string(stream_ids.size()) + " streams");
    return true;
}

bool AudioSynchronizationManager::enableDriftCompensation(const std::string& stream_id) {
    std::lock_guard<std::mutex> lock(events_mutex_);

    auto it = synchronized_streams_.find(stream_id);
    if (it == synchronized_streams_.end()) {
        return false;
    }

    std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);
    it->second->drift_compensation_enabled = true;

    writeToLog("Enabled drift compensation for " + stream_id);
    return true;
}

bool AudioSynchronizationManager::disableDriftCompensation(const std::string& stream_id) {
    std::lock_guard<std::mutex> lock(events_mutex_);

    auto it = synchronized_streams_.find(stream_id);
    if (it == synchronized_streams_.end()) {
        return false;
    }

    std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);
    it->second->drift_compensation_enabled = false;
    it->second->drift_compensation_value = 0.0;

    writeToLog("Disabled drift compensation for " + stream_id);
    return true;
}

double AudioSynchronizationManager::getDriftCompensation(const std::string& stream_id) const {
    std::lock_guard<std::mutex> lock(events_mutex_);

    auto it = synchronized_streams_.find(stream_id);
    if (it != synchronized_streams_.end()) {
        std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);
        return it->second->drift_compensation_value;
    }

    return 0.0;
}

void AudioSynchronizationManager::calibrateDrift(const std::string& stream_id, std::chrono::seconds duration) {
    // Start calibration process
    writeToLog("Calibrating drift for " + stream_id + " (duration: " + std::to_string(duration.count()) + "s)");

    // This would implement actual drift calibration logic
    // For now, just set a default compensation value
    auto it = synchronized_streams_.find(stream_id);
    if (it != synchronized_streams_.end()) {
        std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);
        it->second->drift_compensation_value = 0.001; // 0.1% drift compensation
    }
}

bool AudioSynchronizationManager::setClockAdjustment(double offset_ppm) {
    std::lock_guard<std::mutex> lock(events_mutex_);

    if (clock_servo_) {
        clock_servo_->setTargetOffset(offset_ppm);
        return true;
    }

    return false;
}

double AudioSynchronizationManager::getClockAdjustment() const {
    std::lock_guard<std::mutex> lock(events_mutex_);

    if (clock_servo_) {
        return clock_servo_->getAdjustment();
    }

    return 0.0;
}

bool AudioSynchronizationManager::enableClockServo(bool enabled) {
    // Clock servo is always enabled when initialized
    return clock_servo_ != nullptr;
}

bool AudioSynchronizationManager::isClockServoEnabled() const {
    return clock_servo_ != nullptr;
}

ClockStatistics AudioSynchronizationManager::getClockStatistics(ClockType clock_type) const {
    std::lock_guard<std::mutex> lock(events_mutex_);

    if (clock_type == primary_clock_type_) {
        return primary_clock_statistics_;
    }

    // Return empty statistics for non-primary clocks
    return ClockStatistics{};
}

std::map<std::string, ClockStatistics> AudioSynchronizationManager::getAllStreamStatistics() const {
    std::lock_guard<std::mutex> lock(events_mutex_);

    std::map<std::string, ClockStatistics> statistics;

    for (const auto& pair : synchronized_streams_) {
        std::lock_guard<std::mutex> stream_lock(pair.second->stream_mutex);
        statistics[pair.first] = pair.second->statistics;
    }

    return statistics;
}

std::vector<SyncDelayMeasurement> AudioSynchronizationManager::getDelayHistory(size_t max_measurements) const {
    std::lock_guard<std::mutex> lock(events_mutex_);

    size_t start_index = global_delay_measurements_.size() > max_measurements ?
                       global_delay_measurements_.size() - max_measurements : 0;

    return std::vector<SyncDelayMeasurement>(global_delay_measurements_.begin() + start_index,
                                             global_delay_measurements_.end());
}

double AudioSynchronizationManager::getCurrentOffsetPPM() const {
    std::lock_guard<std::mutex> lock(events_mutex_);
    return primary_clock_statistics_.clock_offset_ppm;
}

double AudioSynchronizationManager::getCurrentJitterMs() const {
    std::lock_guard<std::mutex> lock(events_mutex_);
    return primary_clock_statistics_.jitter_ms;
}

std::chrono::seconds AudioSynchronizationManager::getSyncUptime() const {
    std::lock_guard<std::mutex> lock(events_mutex_);
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::seconds>(now - sync_start_time_);
}

void AudioSynchronizationManager::updateConfig(const SyncManagerConfig& config) {
    std::lock_guard<std::mutex> lock(events_mutex_);
    config_ = config;

    // Update component configurations
    if (clock_servo_) {
        clock_servo_->configure(config_.adjustment_parameters);
    }

    if (ptp_clock_) {
        ptp_clock_.reset(); // Will be reinitialized if needed
    }

    if (sync_monitor_) {
        sync_monitor_.reset(); // Will be reinitialized if needed
    }

    writeToLog("Synchronization manager configuration updated");
}

void AudioSynchronizationManager::updateLockParameters(const SyncLockParameters& params) {
    std::lock_guard<std::mutex> lock(events_mutex_);
    config_.lock_parameters = params;
    writeToLog("Lock parameters updated");
}

void AudioSynchronizationManager::updateAdjustmentParameters(const ClockAdjustmentParameters& params) {
    std::lock_guard<std::mutex> lock(events_mutex_);
    config_.adjustment_parameters = params;

    if (clock_servo_) {
        clock_servo_->configure(params);
    }

    writeToLog("Clock adjustment parameters updated");
}

void AudioSynchronizationManager::updatePTPConfiguration(const PTPConfiguration& config) {
    std::lock_guard<std::mutex> lock(events_mutex_);
    config_.ptp_config = config;

    if (ptp_clock_) {
        ptp_clock_.reset(); // Will be reinitialized with new config
        if (current_sync_mode_ == SyncMode::IEEE_1588_PTP) {
            enablePTP(config);
        }
    }

    writeToLog("PTP configuration updated");
}

void AudioSynchronizationManager::setEventCallback(std::function<void(const SyncEvent&)> callback) {
    std::lock_guard<std::mutex> lock(events_mutex_);
    event_callback_ = callback;
}

void AudioSynchronizationManager::publishEvent(const SyncEvent& event) {
    recordEvent(event);
}

std::vector<SyncEvent> AudioSynchronizationManager::getRecentEvents(const std::string& stream_id, size_t max_events) const {
    std::lock_guard<std::mutex> lock(events_mutex_);

    std::vector<SyncEvent> events;

    for (auto it = event_history_.rbegin(); it != event_history_.rend() && events.size() < max_events; ++it) {
        if (stream_id.empty() || it->stream_id == stream_id) {
            events.push_back(*it);
        }
    }

    return events;
}

bool AudioSynchronizationManager::enableCrossDeviceSync(const std::vector<std::string>& device_ids) {
    // This would implement cross-device synchronization
    // For now, just return true as a placeholder
    writeToLog("Cross-device sync enabled for " + std::to_string(device_ids.size()) + " devices");
    return true;
}

bool AudioSynchronizationManager::disableCrossDeviceSync() {
    writeToLog("Cross-device sync disabled");
    return true;
}

bool AudioSynchronizationManager::performPhaseAlignment(const std::vector<std::string>& stream_ids) {
    // This would implement phase alignment between streams
    // For now, just return true as a placeholder
    writeToLog("Phase alignment performed for " + std::to_string(stream_ids.size()) + " streams");
    return true;
}

bool AudioSynchronizationManager::performFrequencyAlignment(const std::vector<std::string>& stream_ids) {
    // This would implement frequency alignment between streams
    // For now, just return true as a placeholder
    writeToLog("Frequency alignment performed for " + std::to_string(stream_ids.size()) + " streams");
    return true;
}

bool AudioSynchronizationManager::switchSyncSource(SyncSourceType new_source) {
    std::lock_guard<std::mutex> lock(events_mutex_);

    if (!isSourceAvailable(new_source)) {
        return false;
    }

    SyncSourceType old_source = primary_source_type_;
    primary_source_type_ = new_source;

    // Re-initialize clock with new source
    if (!initializeClock(primary_clock_type_, new_source)) {
        primary_source_type_ = old_source;
        return false;
    }

    // Record event
    SyncEvent event;
    event.type = SyncEventType::SYNC_SOURCE_CHANGED;
    event.message = "Sync source switched";
    event.timestamp = current_timestamp_;
    event.event_time = std::chrono::steady_clock::now();
    event.severity = "info";
    event.old_source = old_source;
    event.new_source = new_source;
    recordEvent(event);

    writeToLog("Sync source switched: " + sourceTypeToString(old_source) + " -> " + sourceTypeToString(new_source));
    return true;
}

void AudioSynchronizationManager::enterHoldoverMode() {
    std::lock_guard<std::mutex> lock(events_mutex_);

    if (!holdover_mode_) {
        holdover_mode_ = true;
        sync_status_ = SyncStatus::HOLDOVER;

        SyncEvent event;
        event.type = SyncEventType::HOLDOVER_ENTERED;
        event.message = "Entered holdover mode";
        event.timestamp = current_timestamp_;
        event.event_time = std::chrono::steady_clock::now();
        event.severity = "warning";
        recordEvent(event);

        writeToLog("Entered holdover mode");
    }
}

void AudioSynchronizationManager::exitHoldoverMode() {
    std::lock_guard<std::mutex> lock(events_mutex_);

    if (holdover_mode_) {
        holdover_mode_ = false;
        sync_status_ = SyncStatus::LOCKED;

        SyncEvent event;
        event.type = SyncEventType::HOLDOVER_EXITED;
        event.message = "Exited holdover mode";
        event.timestamp = current_timestamp_;
        event.event_time = std::chrono::steady_clock::now();
        event.severity = "info";
        recordEvent(event);

        writeToLog("Exited holdover mode");
    }
}

bool AudioSynchronizationManager::isInHoldover() const {
    std::lock_guard<std::mutex> lock(events_mutex_);
    return holdover_mode_;
}

std::string AudioSynchronizationManager::generateSyncReport() const {
    std::ostringstream oss;

    oss << "=== Audio Synchronization Report ===\n";
    oss << "Generated at: " << std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() << "\n\n";

    oss << "Synchronization Status:\n";
    oss << "  Primary Clock: " << clockTypeToString(primary_clock_type_) << "\n";
    oss << "  Primary Source: " << sourceTypeToString(primary_source_type_) << "\n";
    oss << "  Sync Mode: " << syncModeToString(current_sync_mode_) << "\n";
    oss << "  Sync Status: " << syncStatusToString(sync_status_) << "\n";
    oss << "  Holdover Mode: " << (holdover_mode_ ? "Yes" : "No") << "\n";
    oss << "  Uptime: " << getSyncUptime().count() << " seconds\n\n";

    oss << "Clock Statistics:\n";
    oss << "  Clock Offset: " << std::fixed << std::setprecision(3) << primary_clock_statistics_.clock_offset_ppm << " ppm\n";
    oss << "  Frequency Error: " << std::fixed << std::setprecision(3) << primary_clock_statistics_.frequency_error_ppm << " ppm\n";
    oss << "  Phase Error: " << std::fixed << std::setprecision(3) << primary_clock_statistics_.phase_error_samples << " samples\n";
    oss << "  Jitter: " << std::fixed << std::setprecision(3) << primary_clock_statistics_.jitter_ms << " ms\n";
    oss << "  Sync Loss Count: " << primary_clock_statistics_.sync_loss_count << "\n\n";

    oss << "Available Sync Sources:\n";
    for (auto source : available_sources_) {
        uint8_t priority = source_priorities_.count(source) ? source_priorities_.at(source) : 255;
        oss << "  " << sourceTypeToString(source) << " (Priority: " << static_cast<int>(priority) << ")\n";
    }
    oss << "\n";

    oss << "Synchronized Streams (" << synchronized_streams_.size() << "):\n";
    for (const auto& pair : synchronized_streams_) {
        const auto& stream = pair.second;
        std::lock_guard<std::mutex> stream_lock(stream->stream_mutex);

        oss << "  " << stream->stream_id << ":\n";
        oss << "    Clock: " << clockTypeToString(stream->clock_type) << "\n";
        oss << "    Sample Rate: " << stream->sample_rate << " Hz\n";
        oss << "    Channels: " << stream->channels << "\n";
        oss << "    Sample Count: " << stream->sample_count << "\n";
        oss << "    SRC Enabled: " << (stream->src_enabled ? "Yes" : "No") << "\n";
        oss << "    SRC Ratio: " << std::fixed << std::setprecision(6) << stream->sample_rate_ratio << "\n";
        oss << "    Drift Compensation: " << (stream->drift_compensation_enabled ? "Yes" : "No") << "\n";
        if (stream->drift_compensation_enabled) {
            oss << "    Drift Value: " << std::fixed << std::setprecision(6) << stream->drift_compensation_value << "\n";
        }
        oss << "    Delay Measurements: " << stream->delay_measurements.size() << "\n";
    }
    oss << "\n";

    oss << "Global Delay Measurements: " << global_delay_measurements_.size() << " total\n";

    return oss.str();
}

std::string AudioSynchronizationManager::generateDiagnosticReport() const {
    std::ostringstream oss;
    oss << generateSyncReport();

    // Add detailed diagnostic information
    oss << "\n=== Diagnostic Information ===\n";

    // PTP status
    if (isPTPEnabled()) {
        oss << "PTP Status: Enabled\n";
        oss << "PTP Master: " << (isPTPMaster() ? "Yes" : "No") << "\n";
        auto ptp_peers = getPTPPeers();
        oss << "PTP Peers: " << ptp_peers.size() << "\n";
        for (const auto& peer : ptp_peers) {
            oss << "  - " << peer << "\n";
        }
    } else {
        oss << "PTP Status: Disabled\n";
    }

    // Clock servo status
    oss << "Clock Servo: " << (isClockServoEnabled() ? "Enabled" : "Disabled") << "\n";
    if (isClockServoEnabled()) {
        oss << "Current Adjustment: " << std::fixed << std::setprecision(3) << getClockAdjustment() << " ppm\n";
    }

    // Recent events
    auto recent_events = getRecentEvents("", 10);
    oss << "\nRecent Events (last 10):\n";
    for (const auto& event : recent_events) {
        oss << "  " << SyncUtils::eventTypeToString(event.type)
            << " - " << event.stream_id << ": " << event.message << "\n";
    }

    return oss.str();
}

void AudioSynchronizationManager::performSyncTest(std::chrono::seconds duration) {
    writeToLog("Starting synchronization test for " + std::to_string(duration.count()) + " seconds");

    // This would implement comprehensive sync testing
    // For now, just simulate a basic test

    auto start_time = std::chrono::steady_clock::now();
    auto end_time = start_time + duration;

    while (std::chrono::steady_clock::now() < end_time && running_.load()) {
        // Simulate sync test activities
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    writeToLog("Synchronization test completed");
}

bool AudioSynchronizationManager::validateTimestampAccuracy() {
    // This would implement timestamp accuracy validation
    // For now, just return true as a placeholder
    writeToLog("Timestamp accuracy validation performed");
    return true;
}

void AudioSynchronizationManager::calibrateClockServo() {
    if (clock_servo_) {
        clock_servo_->reset();
        writeToLog("Clock servo calibrated");
    }
}

// Private methods implementation

void AudioSynchronizationManager::syncThread() {
    while (running_.load()) {
        try {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_sync_check_);

            if (elapsed >= config_.sync_check_interval) {
                checkSyncHealth();
                last_sync_check_ = now;
            }

            // Perform clock adjustments if needed
            performClockAdjustment();

            // Update primary clock
            updatePrimaryClock();

            // Measure delays for streams
            for (const auto& pair : synchronized_streams_) {
                if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_sync_check_).count() >= config_.delay_measurement_interval.count()) {
                    measureDelay(pair.first);
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));

        } catch (const std::exception& e) {
            std::cerr << "Sync thread error: " << e.what() << std::endl;
        }
    }
}

void AudioSynchronizationManager::monitoringThread() {
    while (running_.load()) {
        try {
            // Update statistics
            for (const auto& pair : synchronized_streams_) {
                updateStreamStatistics(pair.first);
            }

            // Update primary clock statistics
            if (clock_servo_) {
                primary_clock_statistics_.clock_offset_ppm = clock_servo_->getAdjustment();
                primary_clock_statistics_.last_sync_time = std::chrono::steady_clock::now();
                primary_clock_statistics_.total_sync_time = getSyncUptime();
            }

            // Update sync monitor
            if (sync_monitor_) {
                sync_monitor_->updateClockStatistics(primary_clock_statistics_);
                for (const auto& measurement : global_delay_measurements_) {
                    sync_monitor_->recordDelayMeasurement(measurement);
                }
            }

            std::this_thread::sleep_for(config_.statistics_update_interval);

        } catch (const std::exception& e) {
            std::cerr << "Monitoring thread error: " << e.what() << std::endl;
        }
    }
}

void AudioSynchronizationManager::timestampThread() {
    while (running_.load()) {
        try {
            // Update current timestamp
            current_timestamp_ = generateTimestampInternal();

            std::this_thread::sleep_for(std::chrono::milliseconds(1));

        } catch (const std::exception& e) {
            std::cerr << "Timestamp thread error: " << e.what() << std::endl;
        }
    }
}

void AudioSynchronizationManager::eventProcessingThread() {
    while (running_.load()) {
        try {
            std::queue<SyncEvent> events_to_process;

            {
                std::lock_guard<std::mutex> lock(events_mutex_);
                events_to_process.swap(event_queue_);
            }

            while (!events_to_process.empty()) {
                SyncEvent event = events_to_process.front();
                events_to_process.pop();

                // Add to history
                {
                    std::lock_guard<std::mutex> lock(events_mutex_);
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
                if (config_.enable_sync_events) {
                    writeToLog("Event: " + SyncUtils::eventTypeToString(event.type) +
                              " - " + event.stream_id + ": " + event.message);
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));

        } catch (const std::exception& e) {
            std::cerr << "Event processing thread error: " << e.what() << std::endl;
        }
    }
}

bool AudioSynchronizationManager::selectBestSyncSource() {
    if (available_sources_.empty()) {
        primary_source_type_ = SyncSourceType::INTERNAL_CLOCK;
        return false;
    }

    // Find source with highest priority (lowest priority number)
    SyncSourceType best_source = SyncSourceType::INTERNAL_CLOCK;
    uint8_t best_priority = 255;

    for (auto source : available_sources_) {
        uint8_t priority = source_priorities_.count(source) ? source_priorities_.at(source) : 255;
        if (priority < best_priority) {
            best_priority = priority;
            best_source = source;
        }
    }

    if (best_source != primary_source_type_) {
        primary_source_type_ = best_source;
        return true;
    }

    return false;
}

bool AudioSynchronizationManager::initializeClock(ClockType clock_type, SyncSourceType source_type) {
    // This would implement clock initialization based on type
    // For now, just return true for internal clock
    return clock_type == ClockType::INTERNAL;
}

void AudioSynchronizationManager::updatePrimaryClock() {
    // This would implement primary clock updates
    // For now, just update the timestamp
}

void AudioSynchronizationManager::performClockAdjustment() {
    if (clock_servo_ && clock_servo_->isLocked()) {
        // Apply clock adjustment from servo
    }
}

void AudioSynchronizationManager::checkSyncHealth() {
    SyncStatus old_status = sync_status_;

    // Check sync health based on various factors
    if (holdover_mode_) {
        sync_status_ = SyncStatus::HOLDOVER;
    } else if (std::abs(primary_clock_statistics_.clock_offset_ppm) > config_.lock_parameters.unlock_threshold_ppm) {
        sync_status_ = SyncStatus::FREERUN;
        enterHoldoverMode();
    } else if (std::abs(primary_clock_statistics_.clock_offset_ppm) < config_.lock_parameters.lock_threshold_ppm) {
        sync_status_ = SyncStatus::LOCKED;
        exitHoldoverMode();
    } else {
        sync_status_ = SyncStatus::ACQUIRING;
    }

    // Record status change event
    if (old_status != sync_status_) {
        SyncEvent event;
        event.type = (sync_status_ == SyncStatus::LOCKED) ? SyncEventType::CLOCK_LOCKED : SyncEventType::CLOCK_UNLOCKED;
        event.message = "Sync status changed: " + syncStatusToString(old_status) + " -> " + syncStatusToString(sync_status_);
        event.timestamp = current_timestamp_;
        event.event_time = std::chrono::steady_clock::now();
        event.severity = (sync_status_ == SyncStatus::LOCKED) ? "info" : "warning";
        recordEvent(event);
    }
}

AudioTimestamp AudioSynchronizationManager::generateTimestampInternal() const {
    AudioTimestamp timestamp;
    timestamp.sample_count = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() * config_.sample_rate / 1e9);
    timestamp.seconds = static_cast<double>(timestamp.sample_count) / config_.sample_rate;
    timestamp.nanoseconds = static_cast<uint32_t>(
        (timestamp.seconds - std::floor(timestamp.seconds)) * 1e9);
    timestamp.system_time = std::chrono::system_clock::now();
    timestamp.timestamp_format_version = 1;
    timestamp.is_valid = true;

    return timestamp;
}

bool AudioSynchronizationManager::synchronizeTimestampInternal(AudioTimestamp& timestamp) {
    // Apply clock adjustment to timestamp
    if (clock_servo_) {
        double adjustment_ppm = clock_servo_->getAdjustment();
        timestamp.sample_count = static_cast<uint64_t>(timestamp.sample_count * (1.0 + adjustment_ppm / 1e6));
        timestamp.seconds = static_cast<double>(timestamp.sample_count) / config_.sample_rate;
    }

    return true;
}

void AudioSynchronizationManager::measureDelay(const std::string& stream_id) {
    // This would implement actual delay measurement
    // For now, create a simulated measurement
    SyncDelayMeasurement measurement;
    measurement.send_timestamp = current_timestamp_;
    measurement.receive_timestamp = current_timestamp_;
    measurement.measured_delay_ms = 5.0 + (std::rand() % 1000) / 1000.0; // 5-6ms with some variation
    measurement.jitter_ms = 0.5;
    measurement.is_valid = true;
    measurement.measurement_time = std::chrono::steady_clock::now();

    // Add to global measurements
    {
        std::lock_guard<std::mutex> lock(events_mutex_);
        global_delay_measurements_.push_back(measurement);
        if (global_delay_measurements_.size() > config_.max_delay_measurements) {
            global_delay_measurements_.pop_front();
        }
    }

    // Add to stream measurements
    auto it = synchronized_streams_.find(stream_id);
    if (it != synchronized_streams_.end()) {
        std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);
        it->second->delay_measurements.push_back(measurement);
        if (it->second->delay_measurements.size() > config_.max_delay_measurements) {
            it->second->delay_measurements.pop_front();
        }
    }
}

void AudioSynchronizationManager::updateStreamStatistics(const std::string& stream_id) {
    auto it = synchronized_streams_.find(stream_id);
    if (it == synchronized_streams_.end()) {
        return;
    }

    std::lock_guard<std::mutex> stream_lock(it->second->stream_mutex);
    auto& stats = it->second->statistics;
    auto& stream_data = it->second;

    // Update basic statistics
    stats.clock_type = stream_data->clock_type;
    stats.sync_status = sync_status_;
    stats.last_sync_time = std::chrono::steady_clock::now();
    stats.total_samples_processed = stream_data->sample_count;

    // Calculate delay statistics
    if (!stream_data->delay_measurements.empty()) {
        double total_delay = 0.0;
        for (const auto& measurement : stream_data->delay_measurements) {
            total_delay += measurement.measured_delay_ms;
        }
        double average_delay = total_delay / stream_data->delay_measurements.size();

        // Calculate jitter (standard deviation)
        double jitter = 0.0;
        for (const auto& measurement : stream_data->delay_measurements) {
            jitter += std::pow(measurement.measured_delay_ms - average_delay, 2);
        }
        jitter = std::sqrt(jitter / stream_data->delay_measurements.size());

        stats.jitter_ms = jitter;
    }
}

bool AudioSynchronizationManager::isSourceAvailable(SyncSourceType source_type) const {
    return std::find(available_sources_.begin(), available_sources_.end(), source_type) != available_sources_.end();
}

SyncSourceType AudioSynchronizationManager::getHighestPrioritySource() const {
    if (available_sources_.empty()) {
        return SyncSourceType::INTERNAL_CLOCK;
    }

    SyncSourceType best_source = SyncSourceType::INTERNAL_CLOCK;
    uint8_t best_priority = 255;

    for (auto source : available_sources_) {
        uint8_t priority = source_priorities_.count(source) ? source_priorities_.at(source) : 255;
        if (priority < best_priority) {
            best_priority = priority;
            best_source = source;
        }
    }

    return best_source;
}

double AudioSynchronizationManager::calculateSourceQuality(SyncSourceType source_type) const {
    // This would implement source quality calculation
    // For now, return a simple quality based on source type
    switch (source_type) {
        case SyncSourceType::GPS:
        case SyncSourceType::PTP_MASTER:
            return 1.0;
        case SyncSourceType::EXTERNAL_WORD_CLOCK:
        case SyncSourceType::AES_EBU:
            return 0.9;
        case SyncSourceType::ADAT:
        case SyncSourceType::S/PDIF:
            return 0.8;
        case SyncSourceType::NTP_SERVER:
            return 0.7;
        case SyncSourceType::MIDI_CLOCK:
            return 0.6;
        case SyncSourceType::INTERNAL_CLOCK:
            return 0.5;
        case SyncSourceType::NETWORK_PEER:
            return 0.4;
        case SyncSourceType::SOFTWARE_GENERATOR:
            return 0.3;
        default:
            return 0.0;
    }
}

void AudioSynchronizationManager::recordEvent(const SyncEvent& event) {
    std::lock_guard<std::mutex> lock(events_mutex_);
    event_queue_.push(event);
}

std::string AudioSynchronizationManager::clockTypeToString(ClockType type) const {
    return SyncUtils::clockTypeToString(type);
}

std::string AudioSynchronizationManager::sourceTypeToString(SyncSourceType type) const {
    return SyncUtils::sourceTypeToString(type);
}

std::string AudioSynchronizationManager::syncModeToString(SyncMode mode) const {
    return SyncUtils::syncModeToString(mode);
}

std::string AudioSynchronizationManager::syncStatusToString(SyncStatus status) const {
    return SyncUtils::syncStatusToString(status);
}

void AudioSynchronizationManager::writeToLog(const std::string& message) {
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

void AudioSynchronizationManager::saveConfiguration() {
    // Implementation for saving configuration to file
}

void AudioSynchronizationManager::loadConfiguration() {
    // Implementation for loading configuration from file
}

// ============================================================================
// PTPClock Implementation
// ============================================================================

PTPClock::PTPClock(const PTPConfiguration& config)
    : config_(config) {
}

PTPClock::~PTPClock() {
    shutdown();
}

bool PTPClock::initialize() {
    running_.store(true);
    ptp_thread_ = std::thread(&PTPClock::ptpThread, this);
    return true;
}

void PTPClock::shutdown() {
    running_.store(false);
    if (ptp_thread_.joinable()) {
        ptp_thread_.join();
    }
}

bool PTPClock::becomeMaster() {
    std::lock_guard<std::mutex> lock(ptp_mutex_);
    is_master_.store(true);
    is_slave_.store(false);
    return true;
}

bool PTPClock::becomeSlave(const std::string& master_address) {
    std::lock_guard<std::mutex> lock(ptp_mutex_);
    is_master_.store(false);
    is_slave_.store(true);

    // Add master to peers
    PTPPeer peer;
    peer.address = master_address;
    peer.is_master = true;
    peers_[master_address] = peer;

    return true;
}

AudioTimestamp PTPClock::getCurrentTime() const {
    std::lock_guard<std::mutex> lock(ptp_mutex_);
    return current_time_;
}

void PTPClock::setTime(const AudioTimestamp& timestamp) {
    std::lock_guard<std::mutex> lock(ptp_mutex_);
    current_time_ = timestamp;
    last_sync_ = std::chrono::steady_clock::now();
}

bool PTPClock::synchronizeWith(const std::string& peer_address) {
    // This would implement PTP synchronization with a peer
    return true;
}

void PTPClock::ptpThread() {
    while (running_.load()) {
        try {
            if (is_master_.load()) {
                sendSyncMessage();
            } else if (is_slave_.load()) {
                sendDelayReqMessage();
            }

            std::this_thread::sleep_for(config_.sync_interval);

        } catch (const std::exception& e) {
            std::cerr << "PTP thread error: " << e.what() << std::endl;
        }
    }
}

void PTPClock::sendSyncMessage() {
    // This would implement PTP sync message sending
}

void PTPClock::sendDelayReqMessage() {
    // This would implement PTP delay request message sending
}

std::vector<std::string> PTPClock::getDiscoveredPeers() const {
    std::lock_guard<std::mutex> lock(ptp_mutex_);

    std::vector<std::string> peer_addresses;
    for (const auto& pair : peers_) {
        peer_addresses.push_back(pair.first);
    }

    return peer_addresses;
}

// ============================================================================
// NTPClock Implementation
// ============================================================================

NTPClock::NTPClock(const std::string& server_address)
    : server_address_(server_address) {
}

NTPClock::~NTPClock() {
}

bool NTPClock::initialize() {
    return true;
}

bool NTPClock::synchronize() {
    if (sendNTPRequest()) {
        NTPPacket packet;
        if (receiveNTPResponse(packet)) {
            // Process NTP response and calculate offset
            synchronized_.store(true);
            last_sync_ = std::chrono::steady_clock::now();
            return true;
        }
    }
    return false;
}

AudioTimestamp NTPClock::getCurrentTime() const {
    // Return current time adjusted by NTP offset
    AudioTimestamp timestamp;
    timestamp.seconds = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    timestamp.system_time = std::chrono::system_clock::now();
    return timestamp;
}

bool NTPClock::sendNTPRequest() {
    // This would implement NTP request sending
    return true;
}

bool NTPClock::receiveNTPResponse(NTPPacket& packet) {
    // This would implement NTP response receiving
    return true;
}

// ============================================================================
// ClockServo Implementation
// ============================================================================

ClockServo::ClockServo(const ClockAdjustmentParameters& params)
    : params_(params) {
    filter_state_ = 0.0;
    last_adjustment_ = std::chrono::steady_clock::now();
}

ClockServo::~ClockServo() {
}

void ClockServo::configure(const ClockAdjustmentParameters& params) {
    std::lock_guard<std::mutex> lock(ptp_mutex_);
    params_ = params;
}

void ClockServo::setTargetOffset(double offset_ppm) {
    std::lock_guard<std::mutex> lock(ptp_mutex_);
    target_offset_ppm_ = offset_ppm;
}

void ClockServo::updateMeasurement(double measured_offset_ppm, double jitter_ms) {
    std::lock_guard<std::mutex> lock(ptp_mutex_);

    measurement_history_.push_back(measured_offset_ppm);
    if (measurement_history_.size() > params_.adjustment_window_size) {
        measurement_history_.pop_front();
    }

    // Apply Kalman filter
    applyKalmanFilter(measured_offset_ppm);

    // Apply PI control
    applyPIControl();

    // Check lock status
    locked_ = calculateLockStatus();
}

double ClockServo::getAdjustment() const {
    std::lock_guard<std::mutex> lock(ptp_mutex_);
    return current_adjustment_ppm_;
}

bool ClockServo::isLocked() const {
    std::lock_guard<std::mutex> lock(ptp_mutex_);
    return locked_;
}

void ClockServo::reset() {
    std::lock_guard<std::mutex> lock(ptp_mutex_);
    current_adjustment_ppm_ = 0.0;
    filter_state_ = 0.0;
    accumulated_error_ = 0.0;
    measurement_history_.clear();
    locked_ = false;
    last_adjustment_ = std::chrono::steady_clock::now();
}

void ClockServo::applyPIControl() {
    if (measurement_history_.empty()) {
        return;
    }

    double error = target_offset_ppm_ - measurement_history_.back();
    accumulated_error_ += error;

    // Proportional term
    double p_term = params_.adjustment_rate * error;

    // Integral term
    double i_term = params_.adjustment_rate * 0.1 * accumulated_error_;

    current_adjustment_ppm_ = p_term + i_term;

    // Clamp to maximum adjustment
    current_adjustment_ppm_ = std::max(-params_.max_adjustment_ppm,
                                        std::min(params_.max_adjustment_ppm, current_adjustment_ppm_));
}

void ClockServo::applyKalmanFilter(double measurement) {
    // Simple Kalman filter implementation
    double process_noise = 0.1;
    double measurement_noise = jitter_ms / 1000.0; // Convert ms to ppm equivalent

    double prediction = filter_state_;
    double prediction_covariance = process_noise;

    double kalman_gain = prediction_covariance / (prediction_covariance + measurement_noise);
    filter_state_ = prediction + kalman_gain * (measurement - prediction);
    // Prediction covariance updated for next iteration
}

double ClockServo::calculateLockStatus() const {
    if (measurement_history_.size() < 10) {
        return false;
    }

    double variance = 0.0;
    double mean = 0.0;

    for (double measurement : measurement_history_) {
        mean += measurement;
    }
    mean /= measurement_history_.size();

    for (double measurement : measurement_history_) {
        variance += std::pow(measurement - mean, 2);
    }
    variance /= measurement_history_.size();

    return std::sqrt(variance) < 1.0; // Locked if variance < 1 ppm
}

// ============================================================================
// SyncMonitor Implementation
// ============================================================================

SyncMonitor::SyncMonitor(const SyncManagerConfig& config)
    : config_(config) {
    last_analysis_ = std::chrono::steady_clock::now();
}

SyncMonitor::~SyncMonitor() {
}

void SyncMonitor::updateClockStatistics(const ClockStatistics& stats) {
    statistics_history_.push_back(stats);
    if (statistics_history_.size() > 100) {
        statistics_history_.pop_front();
    }

    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_analysis_).count() >= 60) {
        analyzeTrends();
        last_analysis_ = now;
    }
}

void SyncMonitor::recordDelayMeasurement(const SyncDelayMeasurement& measurement) {
    delay_history_.push_back(measurement);
    if (delay_history_.size() > 1000) {
        delay_history_.pop_front();
    }
}

void SyncMonitor::recordSyncEvent(const SyncEvent& event) {
    event_history_.push_back(event);
    if (event_history_.size() > 500) {
        event_history_.pop_front();
    }
}

std::vector<std::string> SyncMonitor::getHealthWarnings() const {
    std::vector<std::string> warnings;

    if (!statistics_history_.empty()) {
        const auto& latest = statistics_history_.back();
        if (std::abs(latest.clock_offset_ppm) > 50.0) {
            warnings.push_back("High clock offset detected");
        }
        if (latest.jitter_ms > 10.0) {
            warnings.push_back("High jitter detected");
        }
    }

    return warnings;
}

double SyncMonitor::calculateSyncQuality() const {
    if (statistics_history_.empty()) {
        return 50.0; // Unknown quality
    }

    const auto& latest = statistics_history_.back();
    double quality = 100.0;

    // Penalize for high offset
    quality -= std::abs(latest.clock_offset_ppm) * 0.5;

    // Penalize for high jitter
    quality -= latest.jitter_ms * 2.0;

    return std::max(0.0, std::min(100.0, quality));
}

std::string SyncMonitor::generateHealthReport() const {
    std::ostringstream oss;
    oss << "=== Synchronization Health Report ===\n";
    oss << "Sync Quality: " << std::fixed << std::setprecision(1) << calculateSyncQuality() << "%\n";

    auto warnings = getHealthWarnings();
    if (!warnings.empty()) {
        oss << "Health Warnings:\n";
        for (const auto& warning : warnings) {
            oss << "  - " << warning << "\n";
        }
    } else {
        oss << "No health warnings\n";
    }

    return oss.str();
}

void SyncMonitor::analyzeTrends() {
    // This would implement trend analysis
}

double SyncMonitor::calculateJitterTrend() const {
    return 0.0;
}

double SyncMonitor::calculateStabilityMetric() const {
    return 100.0;
}

// ============================================================================
// SampleRateConverter Implementation
// ============================================================================

SampleRateConverter::SampleRateConverter(uint32_t input_rate, uint32_t output_rate)
    : input_rate_(input_rate), output_rate_(output_rate) {
    ratio_ = static_cast<double>(output_rate) / static_cast<double>(input_rate);
    initializeFilter();
}

SampleRateConverter::~SampleRateConverter() {
}

void SampleRateConverter::setRatio(double ratio) {
    ratio_ = ratio;
}

size_t SampleRateConverter::convertSamples(const float* input, float* output, size_t input_samples) {
    if (!input || !output || input_samples == 0) {
        return 0;
    }

    // This is a simplified implementation
    // In practice, you would use a proper sample rate conversion algorithm
    size_t output_samples = 0;
    for (size_t i = 0; i < input_samples; ++i) {
        while (phase_accumulator_ < 1.0) {
            output[output_samples++] = input[i];
            phase_accumulator_ += ratio_;
        }
        phase_accumulator_ -= 1.0;
    }

    return output_samples;
}

void SampleRateConverter::reset() {
    phase_accumulator_ = 0.0;
    input_history_.clear();
}

void SampleRateConverter::initializeFilter() {
    // Initialize filter coefficients for sample rate conversion
    filter_order_ = 64;
    filter_coefficients_.resize(filter_order_);

    // Simple windowed sinc filter
    for (size_t i = 0; i < filter_order_; ++i) {
        double n = static_cast<double>(i) - filter_order_ / 2.0;
        if (n != 0) {
            filter_coefficients_[i] = std::sin(M_PI * n / filter_order_) / (M_PI * n / filter_order_);
        } else {
            filter_coefficients_[i] = 1.0;
        }
    }
}

// ============================================================================
// SyncUtils Implementation
// ============================================================================

namespace SyncUtils {

std::string clockTypeToString(ClockType type) {
    switch (type) {
        case ClockType::INTERNAL: return "Internal";
        case ClockType::EXTERNAL_WORD_CLOCK: return "External Word Clock";
        case ClockType::AES_EBU_CLOCK: return "AES/EBU";
        case ClockType::ADAT_CLOCK: return "ADAT";
        case ClockType::MIDI_CLOCK: return "MIDI Clock";
        case ClockType::PTP_CLOCK: return "PTP";
        case ClockType::NTP_CLOCK: return "NTP";
        case ClockType::GPS_CLOCK: return "GPS";
        case ClockType::SOFTWARE_CLOCK: return "Software";
        default: return "Unknown";
    }
}

std::string sourceTypeToString(SyncSourceType type) {
    switch (type) {
        case SyncSourceType::INTERNAL_CLOCK: return "Internal Clock";
        case SyncSourceType::EXTERNAL_WORD_CLOCK: return "External Word Clock";
        case SyncSourceType::AES_EBU: return "AES/EBU";
        case SyncSourceType::ADAT: return "ADAT";
        case SyncSourceType::S/PDIF: return "S/PDIF";
        case SyncSourceType::MIDI: return "MIDI";
        case SyncSourceType::PTP_MASTER: return "PTP Master";
        case SyncSourceType::NTP_SERVER: return "NTP Server";
        case SyncSourceType::GPS: return "GPS";
        case SyncSourceType::NETWORK_PEER: return "Network Peer";
        case SyncSourceType::SOFTWARE_GENERATOR: return "Software Generator";
        default: return "Unknown";
    }
}

std::string syncModeToString(SyncMode mode) {
    switch (mode) {
        case SyncMode::FREE_RUNNING: return "Free Running";
        case SyncMode::MASTER_SLAVE: return "Master/Slave";
        case SyncMode::PEER_TO_PEER: return "Peer to Peer";
        case SyncMode::IEEE_1588_PTP: return "IEEE 1588 PTP";
        case SyncMode::NTP: return "NTP";
        case SyncMode::GPS_DISCIPLINED: return "GPS Disciplined";
        case SyncMode::WORD_CLOCK: return "Word Clock";
        case SyncMode::ADAT_SYNC: return "ADAT Sync";
        case SyncMode::AES_SYNC: return "AES Sync";
        case SyncMode::MIDI_CLOCK: return "MIDI Clock";
        case SyncMode::SAMPLE_ACCURATE: return "Sample Accurate";
        default: return "Unknown";
    }
}

std::string syncStatusToString(SyncStatus status) {
    switch (status) {
        case SyncStatus::LOCKED: return "Locked";
        case SyncStatus::ACQUIRING: return "Acquiring";
        case SyncStatus::HOLDOVER: return "Holdover";
        case SyncStatus::FREERUN: return "Free Running";
        case SyncStatus::FAULT: return "Fault";
        case SyncStatus::UNKNOWN: return "Unknown";
        default: return "Unknown";
    }
}

std::string eventTypeToString(SyncEventType type) {
    switch (type) {
        case SyncEventType::CLOCK_LOCKED: return "Clock Locked";
        case SyncEventType::CLOCK_UNLOCKED: return "Clock Unlocked";
        case SyncEventType::SYNC_SOURCE_CHANGED: return "Sync Source Changed";
        case SyncEventType::MASTER_CHANGED: return "Master Changed";
        case SyncEventType::PHASE_ADJUSTMENT: return "Phase Adjustment";
        case SyncEventType::FREQUENCY_ADJUSTMENT: return "Frequency Adjustment";
        case SyncEventType::HOLDOVER_ENTERED: return "Holdover Entered";
        case SyncEventType::HOLDOVER_EXITED: return "Holdover Exited";
        case SyncEventType::PTP_MASTER_SELECTED: return "PTP Master Selected";
        case SyncEventType::PTP_MASTER_LOST: return "PTP Master Lost";
        case SyncEventType::TIMESTAMP_RECEIVED: return "Timestamp Received";
        case SyncEventType::DELAY_MEASUREMENT_COMPLETE: return "Delay Measurement Complete";
        case SyncEventType::SYNC_FAULT_DETECTED: return "Sync Fault Detected";
        case SyncEventType::SYNC_RECOVERY: return "Sync Recovery";
        default: return "Unknown";
    }
}

ClockType stringToClockType(const std::string& str) {
    if (str == "Internal" || str == "INTERNAL") return ClockType::INTERNAL;
    if (str == "External Word Clock" || str == "EXTERNAL_WORD_CLOCK") return ClockType::EXTERNAL_WORD_CLOCK;
    if (str == "AES/EBU" || str == "AES_EBU_CLOCK") return ClockType::AES_EBU_CLOCK;
    if (str == "ADAT" || str == "ADAT_CLOCK") return ClockType::ADAT_CLOCK;
    if (str == "MIDI Clock" || str == "MIDI_CLOCK") return ClockType::MIDI_CLOCK;
    if (str == "PTP" || str == "PTP_CLOCK") return ClockType::PTP_CLOCK;
    if (str == "NTP" || str == "NTP_CLOCK") return ClockType::NTP_CLOCK;
    if (str == "GPS" || str == "GPS_CLOCK") return ClockType::GPS_CLOCK;
    if (str == "Software" || str == "SOFTWARE_CLOCK") return ClockType::SOFTWARE_CLOCK;
    return ClockType::INTERNAL; // Default
}

SyncSourceType stringToSourceType(const std::string& str) {
    if (str == "Internal Clock" || str == "INTERNAL_CLOCK") return SyncSourceType::INTERNAL_CLOCK;
    if (str == "External Word Clock" || str == "EXTERNAL_WORD_CLOCK") return SyncSourceType::EXTERNAL_WORD_CLOCK;
    if (str == "AES/EBU" || str == "AES_EBU") return SyncSourceType::AES_EBU;
    if (str == "ADAT" || str == "ADAT") return SyncSourceType::ADAT;
    if (str == "S/PDIF" || str == "S/PDIF") return SyncSourceType::S/PDIF;
    if (str == "MIDI" || str == "MIDI") return SyncSourceType::MIDI;
    if (str == "PTP Master" || str == "PTP_MASTER") return SyncSourceType::PTP_MASTER;
    if (str == "NTP Server" || str == "NTP_SERVER") return SyncSourceType::NTP_SERVER;
    if (str == "GPS" || str == "GPS") return SyncSourceType::GPS;
    if (str == "Network Peer" || str == "NETWORK_PEER") return SyncSourceType::NETWORK_PEER;
    if (str == "Software Generator" || str == "SOFTWARE_GENERATOR") return SyncSourceType::SOFTWARE_GENERATOR;
    return SyncSourceType::INTERNAL_CLOCK; // Default
}

SyncMode stringToSyncMode(const std::string& str) {
    if (str == "Free Running" || str == "FREE_RUNNING") return SyncMode::FREE_RUNNING;
    if (str == "Master/Slave" || str == "MASTER_SLAVE") return SyncMode::MASTER_SLAVE;
    if (str == "Peer to Peer" || str == "PEER_TO_PEER") return SyncMode::PEER_TO_PEER;
    if (str == "IEEE 1588 PTP" || str == "IEEE_1588_PTP") return SyncMode::IEEE_1588_PTP;
    if (str == "NTP" || str == "NTP") return SyncMode::NTP;
    if (str == "GPS Disciplined" || str == "GPS_DISCIPLINED") return SyncMode::GPS_DISCIPLINED;
    if (str == "Word Clock" || str == "WORD_CLOCK") return SyncMode::WORD_CLOCK;
    if (str == "ADAT Sync" || str == "ADAT_SYNC") return SyncMode::ADAT_SYNC;
    if (str == "AES Sync" || str == "AES_SYNC") return SyncMode::AES_SYNC;
    if (str == "MIDI Clock" || str == "MIDI_CLOCK") return SyncMode::MIDI_CLOCK;
    if (str == "Sample Accurate" || str == "SAMPLE_ACCURATE") return SyncMode::SAMPLE_ACCURATE;
    return SyncMode::FREE_RUNNING; // Default
}

SyncStatus stringToSyncStatus(const std::string& str) {
    if (str == "Locked" || str == "LOCKED") return SyncStatus::LOCKED;
    if (str == "Acquiring" || str == "ACQUIRING") return SyncStatus::ACQUIRING;
    if (str == "Holdover" || str == "HOLDOVER") return SyncStatus::HOLDOVER;
    if (str == "Free Running" || str == "FREERUN") return SyncStatus::FREERUN;
    if (str == "Fault" || str == "FAULT") return SyncStatus::FAULT;
    if (str == "Unknown" || str == "UNKNOWN") return SyncStatus::UNKNOWN;
    return SyncStatus::UNKNOWN; // Default
}

SyncEventType stringToEventType(const std::string& str) {
    if (str == "Clock Locked" || str == "CLOCK_LOCKED") return SyncEventType::CLOCK_LOCKED;
    if (str == "Clock Unlocked" || str == "CLOCK_UNLOCKED") return SyncEventType::CLOCK_UNLOCKED;
    // ... add more mappings as needed
    return SyncEventType::CLOCK_LOCKED; // Default
}

AudioTimestamp createTimestamp(uint64_t sample_count, uint32_t sample_rate) {
    AudioTimestamp timestamp;
    timestamp.sample_count = sample_count;
    timestamp.seconds = static_cast<double>(sample_count) / sample_rate;
    timestamp.nanoseconds = static_cast<uint32_t>(
        (timestamp.seconds - std::floor(timestamp.seconds)) * 1e9);
    timestamp.system_time = std::chrono::system_clock::now();
    timestamp.timestamp_format_version = 1;
    timestamp.is_valid = true;

    return timestamp;
}

double calculateClockOffset(const AudioTimestamp& local, const AudioTimestamp& remote) {
    double local_seconds = local.getTotalSeconds();
    double remote_seconds = remote.getTotalSeconds();
    return (remote_seconds - local_seconds) * 1e6; // Convert to ppm
}

double calculateDelay(const AudioTimestamp& send_time, const AudioTimestamp& receive_time) {
    double delay_seconds = receive_time.getTotalSeconds() - send_time.getTotalSeconds();
    return delay_seconds * 1000.0; // Convert to milliseconds
}

bool compareTimestamps(const AudioTimestamp& a, const AudioTimestamp& b, double tolerance_ms) {
    double diff_ms = std::abs(a.getTotalSeconds() - b.getTotalSeconds()) * 1000.0;
    return diff_ms <= tolerance_ms;
}

double ppmToRatio(double ppm) {
    return 1.0 + ppm / 1e6;
}

double ratioToPPM(double ratio) {
    return (ratio - 1.0) * 1e6;
}

double calculatePPMDifference(double frequency1, double frequency2) {
    return ((frequency1 - frequency2) / frequency2) * 1e6;
}

uint64_t timestampToNanoseconds(const AudioTimestamp& timestamp) {
    return static_cast<uint64_t>(timestamp.getTotalSeconds() * 1e9);
}

AudioTimestamp nanosecondsToTimestamp(uint64_t nanoseconds) {
    AudioTimestamp timestamp;
    timestamp.seconds = static_cast<double>(nanoseconds) / 1e9;
    timestamp.nanoseconds = static_cast<uint32_t>(nanoseconds % 1000000000ULL);
    timestamp.system_time = std::chrono::system_clock::now();
    timestamp.timestamp_format_version = 1;
    timestamp.is_valid = true;

    return timestamp;
}

std::chrono::nanoseconds timestampToDuration(const AudioTimestamp& timestamp) {
    return std::chrono::nanoseconds(timestampToNanoseconds(timestamp));
}

AudioTimestamp durationToTimestamp(std::chrono::nanoseconds duration) {
    return nanosecondsToTimestamp(duration.count());
}

std::string formatTimestamp(const AudioTimestamp& timestamp) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << timestamp.getTotalSeconds()
        << "s (sample: " << timestamp.sample_count << ")";
    return oss.str();
}

std::string formatDuration(std::chrono::nanoseconds duration) {
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration % std::chrono::seconds(1));

    std::ostringstream oss;
    oss << seconds.count() << "." << std::setfill('0') << std::setw(3) << milliseconds.count() << "s";
    return oss.str();
}

} // namespace SyncUtils

} // namespace Network
} // namespace VortexGPU