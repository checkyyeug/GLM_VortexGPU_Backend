#include "core/audio/session_manager.hpp"
#include "core/dsp/audio_math.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <unistd.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

namespace vortex {
namespace core {
namespace audio {

// Constants
constexpr int DEFAULT_TIMEOUT_SECONDS = 300; // 5 minutes
constexpr int METRICS_UPDATE_INTERVAL_MS = 1000;
constexpr int CLEANUP_INTERVAL_SECONDS = 30;
constexpr int MONITORING_INTERVAL_MS = 100;
constexpr float DEFAULT_CPU_USAGE_THRESHOLD = 80.0f; // 80% CPU threshold
constexpr size_t DEFAULT_MEMORY_THRESHOLD_MB = 1024; // 1GB memory threshold
constexpr int MAX_HISTORY_SIZE = 10000;
constexpr double MIN_SESSION_DURATION_SECONDS = 1.0; // Minimum session duration
constexpr double MAX_SESSION_DURATION_SECONDS = 86400.0; // 24 hours max duration

// Missing definitions from header (would be defined in their respective headers)
struct SystemMetrics {
    double total_cpu_usage_percent = 0.0;
    size_t total_memory_usage_mb = 0;
    int active_sessions = 0;
    int suspended_sessions = 0;
    uint64_t total_processed_frames = 0;
    double average_latency_ms = 0.0;
    uint64_t total_errors = 0;
    std::chrono::steady_clock::time_point last_update;
};

struct SystemResources {
    int max_input_channels = 128;
    int max_output_channels = 128;
    int max_sample_rate = 192000;
    int min_sample_rate = 8000;
    int max_buffer_size = 8192;
    int min_buffer_size = 16;
    double max_latency_ms = 100.0;
    int max_dsp_cores = 16;
    size_t max_memory_mb = 8192;
    int max_concurrent_sessions = 64;
    std::vector<AudioBitDepth> supported_bit_depths;
};

struct SystemHealthStatus {
    bool system_healthy = true;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
    double system_load_percent = 0.0;
    size_t memory_usage_percent = 0;
    int active_sessions = 0;
    std::chrono::steady_clock::time_point last_check;
};

struct SessionStatistics {
    uint64_t total_sessions = 0;
    uint64_t active_sessions = 0;
    uint64_t completed_sessions = 0;
    uint64_t failed_sessions = 0;
    uint64_t preempted_sessions = 0;
    double average_session_duration_seconds = 0.0;
    double total_cpu_time_seconds = 0.0;
    size_t total_memory_usage_mb = 0;
    uint64_t total_errors = 0;
    std::chrono::steady_clock::time_point last_reset;
};

SessionManager::SessionManager() : max_sessions_(64),
                                 last_metrics_update_(std::chrono::steady_clock::now()),
                                 next_session_id_(1) {
    // Initialize system resources with defaults
    system_resources_.max_input_channels = 128;
    system_resources_.max_output_channels = 128;
    system_resources_.max_sample_rate = 192000;
    system_resources_.min_sample_rate = 8000;
    system_resources_.max_buffer_size = 8192;
    system_resources_.min_buffer_size = 16;
    system_resources_.max_latency_ms = 100.0;
    system_resources_.max_dsp_cores = 16;
    system_resources_.max_memory_mb = 8192;
    system_resources_.max_concurrent_sessions = max_sessions_;
    system_resources_.supported_bit_depths = {
        AudioBitDepth::INT16, AudioBitDepth::INT24, AudioBitDepth::INT32,
        AudioBitDepth::FLOAT32, AudioBitDepth::FLOAT64
    };
}

SessionManager::~SessionManager() {
    shutdown();
}

bool SessionManager::initialize(int max_sessions) {
    if (initialized_) {
        return false;
    }

    max_sessions_ = max_sessions;
    system_resources_.max_concurrent_sessions = max_sessions;

    // Load session templates
    if (!loadSessionTemplates()) {
        // Create default templates if loading fails
        session_templates_["Recording"] = session_utils::createDefaultTemplate(SessionType::RECORDING);
        session_templates_["Playback"] = session_utils::createDefaultTemplate(SessionType::PLAYBACK);
        session_templates_["Duplex"] = session_utils::createDefaultTemplate(SessionType::DUPLEX);
        session_templates_["Streaming"] = session_utils::createDefaultTemplate(SessionType::STREAMING);
    }

    // Start background threads
    metrics_active_ = true;
    metrics_thread_ = std::thread(&SessionManager::metricsUpdateLoop, this);

    cleanup_active_ = true;
    cleanup_thread_ = std::thread(&SessionManager::cleanupLoop, this);

    initialized_ = true;
    return true;
}

void SessionManager::shutdown() {
    if (!initialized_) {
        return;
    }

    // Stop background threads
    metrics_active_ = false;
    cleanup_active_ = false;

    if (metrics_thread_.joinable()) {
        metrics_thread_.join();
    }

    if (cleanup_thread_.joinable()) {
        cleanup_thread_.join();
    }

    // Destroy all sessions
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    for (auto& [session_id, session] : sessions_) {
        stopSessionMonitoring(*session);
    }
    sessions_.clear();

    // Save session templates
    saveSessionTemplates();

    // Clear callbacks
    state_callback_ = nullptr;
    metrics_callback_ = nullptr;
    error_callback_ = nullptr;
    resource_callback_ = nullptr;

    initialized_ = false;
}

uint64_t SessionManager::createSession(const SessionInfo& session_info) {
    if (!initialized_) {
        return 0;
    }

    // Validate session configuration
    if (!validateSessionConfiguration(session_info)) {
        return 0;
    }

    // Check if we can create another session
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    if (sessions_.size() >= static_cast<size_t>(max_sessions_)) {
        return 0;
    }

    // Check resource availability
    if (!canAllocateResources(session_info.resource_request)) {
        return 0;
    }

    // Generate unique session ID
    uint64_t session_id = generateSessionId();

    // Create session internal structure
    auto session_internal = std::make_unique<SessionInternal>();
    session_internal->info = session_info;
    session_internal->info.session_id = session_id;
    session_internal->info.state = SessionState::PREPARING;
    session_internal->info.creation_time = std::chrono::steady_clock::now();
    session_internal->info.last_activity = session_internal->info.creation_time;
    session_internal->process_count = 0;
    session_internal->error_count = 0;

    // Allocate resources
    SessionAllocation allocation = allocateSessionResources(session_info.resource_request);
    if (!allocation.resources_sufficient) {
        return 0;
    }

    session_internal->info.allocation = allocation;
    session_internal->last_activity = std::chrono::steady_clock::now();

    // Initialize session components
    try {
        initializeSessionEngine(*session_internal);
        initializeSessionRouter(*session_internal);
        initializeSessionDeviceManager(*session_internal);
    } catch (const std::exception& e) {
        deallocateSessionResources(allocation);
        notifySessionError(session_id, "Failed to initialize session components: " + std::string(e.what()));
        return 0;
    }

    // Start monitoring
    startSessionMonitoring(*session_internal);

    // Add to sessions map
    sessions_[session_id] = std::move(session_internal);

    // Activate session
    if (activateSession(session_id)) {
        // Update resource allocation
        {
            std::lock_guard<std::mutex> resource_lock(resources_mutex_);
            allocateResources(session_id, session_info.resource_request);
        }

        notifySessionStateChange(session_id, SessionState::PREPARING, SessionState::ACTIVE);
        return session_id;
    } else {
        // Cleanup on failure
        stopSessionMonitoring(*sessions_[session_id]);
        deallocateResources(session_id);
        sessions_.erase(session_id);
        return 0;
    }
}

bool SessionManager::destroySession(uint64_t session_id) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return false;
    }

    auto& session = *it->second;

    // Stop monitoring
    stopSessionMonitoring(session);

    // Update state
    SessionState old_state = session.info.state;
    session.info.state = SessionState::TERMINATING;
    session.info.end_time = std::chrono::steady_clock::now();

    // Calculate final duration
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        session.info.end_time - session.info.creation_time).count();
    session.info.metrics.total_duration_seconds = duration;

    // Deallocate resources
    deallocateResources(session_id);

    // Update final metrics
    session.info.metrics = calculateSessionMetrics(session_id);

    // Notify state change
    notifySessionStateChange(session_id, old_state, SessionState::COMPLETED);

    // Remove from sessions
    sessions_.erase(it);

    return true;
}

std::optional<SessionInfo> SessionManager::getSession(uint64_t session_id) const {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
        return it->second->info;
    }

    return std::nullopt;
}

bool SessionManager::updateSession(uint64_t session_id, const SessionInfo& session_info) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return false;
    }

    auto& session = *it->second;

    // Validate new configuration
    if (!validateSessionConfiguration(session_info)) {
        return false;
    }

    // Check if resource change is needed
    if (session_info.resource_request.input_channels != session.info.resource_request.input_channels ||
        session_info.resource_request.output_channels != session.info.resource_request.output_channels ||
        session_info.resource_request.sample_rate != session.info.resource_request.sample_rate ||
        session_info.resource_request.buffer_size != session.info.resource_request.buffer_size ||
        session_info.resource_request.bit_depth != session.info.resource_request.bit_depth) {

        // Request resource change
        if (!requestResourceChange(session_id, session_info.resource_request)) {
            return false;
        }
    }

    // Update session info (preserving critical fields)
    SessionInfo old_info = session.info;
    session.info = session_info;
    session.info.session_id = session_id;
    session.info.creation_time = old_info.creation_time;
    session.info.allocation = old_info.allocation;
    session.info.metrics = old_info.metrics;

    // Update priority if changed
    if (old_info.priority != session_info.priority) {
        updateSessionPriority(session_id, session_info.priority);
    }

    updateSessionActivity(session_id);

    return true;
}

bool SessionManager::activateSession(uint64_t session_id) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return false;
    }

    auto& session = *it->second;

    if (session.info.state == SessionState::ACTIVE) {
        return true; // Already active
    }

    SessionState old_state = session.info.state;

    try {
        // Start audio engine
        if (session.engine) {
            if (!session.engine->start()) {
                throw std::runtime_error("Failed to start audio engine");
            }
        }

        // Set state to active
        session.info.state = SessionState::ACTIVE;
        session.info.metrics.start_time = std::chrono::steady_clock::now();

        updateSessionActivity(session_id);
        notifySessionStateChange(session_id, old_state, SessionState::ACTIVE);

        return true;
    } catch (const std::exception& e) {
        session.info.state = SessionState::ERROR;
        notifySessionError(session_id, "Failed to activate session: " + std::string(e.what()));
        return false;
    }
}

bool SessionManager::deactivateSession(uint64_t session_id) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return false;
    }

    auto& session = *it->second;

    if (session.info.state == SessionState::INACTIVE) {
        return true; // Already inactive
    }

    SessionState old_state = session.info.state;

    try {
        // Stop audio engine
        if (session.engine) {
            session.engine->stop();
        }

        // Set state to inactive
        session.info.state = SessionState::INACTIVE;
        session.info.metrics.end_time = std::chrono::steady_clock::now();

        // Update active duration
        if (session.info.metrics.start_time.time_since_epoch().count() > 0) {
            auto active_duration = std::chrono::duration_cast<std::chrono::seconds>(
                session.info.metrics.end_time - session.info.metrics.start_time).count();
            session.info.metrics.active_duration_seconds += active_duration;
        }

        notifySessionStateChange(session_id, old_state, SessionState::INACTIVE);

        return true;
    } catch (const std::exception& e) {
        session.info.state = SessionState::ERROR;
        notifySessionError(session_id, "Failed to deactivate session: " + std::string(e.what()));
        return false;
    }
}

bool SessionManager::pauseSession(uint64_t session_id) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return false;
    }

    auto& session = *it->second;

    if (session.info.state != SessionState::ACTIVE) {
        return false;
    }

    SessionState old_state = session.info.state;
    session.info.state = SessionState::PAUSED;

    // Update active duration so far
    session.info.metrics.active_duration_seconds += std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - session.info.metrics.start_time).count();

    notifySessionStateChange(session_id, old_state, SessionState::PAUSED);
    return true;
}

bool SessionManager::resumeSession(uint64_t session_id) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return false;
    }

    auto& session = *it->second;

    if (session.info.state != SessionState::PAUSED) {
        return false;
    }

    SessionState old_state = session.info.state;
    session.info.state = SessionState::ACTIVE;
    session.info.metrics.start_time = std::chrono::steady_clock::now();

    notifySessionStateChange(session_id, old_state, SessionState::ACTIVE);
    return true;
}

bool SessionManager::suspendSession(uint64_t session_id, const std::string& reason) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return false;
    }

    auto& session = *it->second;

    if (session.info.state == SessionState::SUSPENDED) {
        return true; // Already suspended
    }

    SessionState old_state = session.info.state;

    try {
        // Temporarily deallocate resources for high-priority sessions
        if (session.engine) {
            session.engine->stop();
        }

        session.info.state = SessionState::SUSPENDED;
        session.info.metadata["suspension_reason"] = reason;
        session.info.metadata["suspension_time"] = std::to_string(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());

        notifySessionStateChange(session_id, old_state, SessionState::SUSPENDED);
        return true;
    } catch (const std::exception& e) {
        session.info.state = SessionState::ERROR;
        notifySessionError(session_id, "Failed to suspend session: " + std::string(e.what()));
        return false;
    }
}

std::vector<SessionInfo> SessionManager::getActiveSessions() const {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    std::vector<SessionInfo> active_sessions;
    for (const auto& [session_id, session] : sessions_) {
        if (session->info.state == SessionState::ACTIVE) {
            active_sessions.push_back(session->info);
        }
    }

    return active_sessions;
}

std::vector<SessionInfo> SessionManager::getSessionsByType(SessionType type) const {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    std::vector<SessionInfo> matching_sessions;
    for (const auto& [session_id, session] : sessions_) {
        if (session->info.type == type) {
            matching_sessions.push_back(session->info);
        }
    }

    return matching_sessions;
}

std::vector<SessionInfo> SessionManager::getSessionsByPriority(SessionPriority priority) const {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    std::vector<SessionInfo> matching_sessions;
    for (const auto& [session_id, session] : sessions_) {
        if (session->info.priority == priority) {
            matching_sessions.push_back(session->info);
        }
    }

    return matching_sessions;
}

std::vector<SessionInfo> SessionManager::getSessionsByProcess(const std::string& process_id) const {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    std::vector<SessionInfo> matching_sessions;
    for (const auto& [session_id, session] : sessions_) {
        if (session->info.owner_process_id == process_id) {
            matching_sessions.push_back(session->info);
        }
    }

    return matching_sessions;
}

std::vector<SessionInfo> SessionManager::getSessionsByUser(const std::string& user_id) const {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    std::vector<SessionInfo> matching_sessions;
    for (const auto& [session_id, session] : sessions_) {
        if (session->info.owner_user_id == user_id) {
            matching_sessions.push_back(session->info);
        }
    }

    return matching_sessions;
}

std::vector<SessionInfo> SessionManager::findSessionsByName(const std::string& name) const {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    std::vector<SessionInfo> matching_sessions;
    std::string lower_name = name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

    for (const auto& [session_id, session] : sessions_) {
        std::string session_name = session->info.session_name;
        std::transform(session_name.begin(), session_name.end(), session_name.begin(), ::tolower);

        if (session_name.find(lower_name) != std::string::npos) {
            matching_sessions.push_back(session->info);
        }
    }

    return matching_sessions;
}

std::optional<SessionMetrics> SessionManager::getSessionMetrics(uint64_t session_id) const {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
        return it->second->info.metrics;
    }

    return std::nullopt;
}

std::vector<SessionMetrics> SessionManager::getAllSessionMetrics() const {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    std::vector<SessionMetrics> all_metrics;
    for (const auto& [session_id, session] : sessions_) {
        all_metrics.push_back(session->info.metrics);
    }

    return all_metrics;
}

SystemMetrics SessionManager::getSystemMetrics() const {
    return calculateSystemMetrics();
}

bool SessionManager::setSessionPriority(uint64_t session_id, SessionPriority priority) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return false;
    }

    SessionPriority old_priority = it->second->info.priority;
    it->second->info.priority = priority;

    updateSessionPriority(session_id, priority);
    updateSessionActivity(session_id);

    return true;
}

bool SessionManager::requestResourceChange(uint64_t session_id, const SessionResourceRequest& new_request) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return false;
    }

    auto& session = *it->second;

    // Validate new request
    if (!validateResourceRequest(new_request)) {
        return false;
    }

    // Check if we can allocate new resources
    if (!canAllocateResources(new_request)) {
        // Try preempting lower priority sessions
        if (!preemptSessions(session_id, new_request)) {
            return false;
        }
    }

    // Deallocate old resources
    deallocateResources(session_id);

    // Allocate new resources
    SessionAllocation new_allocation = allocateSessionResources(new_request);
    if (!new_allocation.resources_sufficient) {
        // Try to restore old allocation
        SessionAllocation old_allocation = allocateSessionResources(session.info.resource_request);
        if (old_allocation.resources_sufficient) {
            session.info.allocation = old_allocation;
            allocateResources(session_id, session.info.resource_request);
        }
        return false;
    }

    // Apply new allocation
    SessionAllocation old_allocation = session.info.allocation;
    session.info.allocation = new_allocation;
    session.info.resource_request = new_request;

    // Update actual allocation in resource tracker
    {
        std::lock_guard<std::mutex> resource_lock(resources_mutex_);
        allocateResources(session_id, new_request);
    }

    // Notify resource change
    notifyResourceChange(session_id, new_allocation);

    // Reconfigure audio components if needed
    try {
        if (session.engine && (old_allocation.actual_sample_rate != new_allocation.actual_sample_rate ||
                              old_allocation.actual_buffer_size != new_allocation.actual_buffer_size)) {
            // Reinitialize engine with new parameters
            AudioSession engine_session = session.engine->getSession();
            engine_session.sample_rate = new_allocation.actual_sample_rate;
            engine_session.buffer_size = new_allocation.actual_buffer_size;
            session.engine->initialize(engine_session);
        }
    } catch (const std::exception& e) {
        notifySessionError(session_id, "Failed to reconfigure audio components: " + std::string(e.what()));
        return false;
    }

    return true;
}

bool SessionManager::preemptSessions(uint64_t session_id, const SessionResourceRequest& required_resources) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    // Find sessions to preempt (lower priority)
    std::vector<uint64_t> sessions_to_preempt;
    for (const auto& [other_session_id, other_session] : sessions_) {
        if (other_session_id == session_id) {
            continue;
        }

        if (other_session->info.priority < sessions_.at(session_id)->info.priority &&
            (other_session->info.state == SessionState::ACTIVE ||
             other_session->info.state == SessionState::PAUSED)) {
            sessions_to_preempt.push_back(other_session_id);
        }
    }

    if (sessions_to_preempt.empty()) {
        return false; // No sessions to preempt
    }

    // Suspend lower priority sessions
    return preemptSessionsInternal(sessions_to_preempt);
}

bool SessionManager::setSessionScheduling(uint64_t session_id, SessionScheduling scheduling) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return false;
    }

    it->second->info.scheduling = scheduling;
    updateSessionActivity(session_id);

    // Re-schedule sessions based on new policy
    scheduleSessions();

    return true;
}

bool SessionManager::setSessionIsolation(uint64_t session_id, SessionIsolation isolation) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return false;
    }

    it->second->info.isolation = isolation;
    updateSessionActivity(session_id);

    // Apply isolation settings
    // This would involve configuring device access and resource isolation
    // Implementation would depend on the specific isolation level

    return true;
}

bool SessionManager::forceTerminateSession(uint64_t session_id, const std::string& reason) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return false;
    }

    auto& session = *it->second;

    // Force termination regardless of state
    SessionState old_state = session.info.state;
    session.info.state = SessionState::TERMINATING;
    session.info.metadata["termination_reason"] = reason;
    session.info.metadata["forced_termination"] = "true";

    try {
        // Stop all components forcefully
        if (session.engine) {
            session.engine->stop();
        }

        // Deallocate resources immediately
        deallocateResources(session_id);

        // Update metrics
        session.info.metrics = calculateSessionMetrics(session_id);

        // Notify state change
        notifySessionStateChange(session_id, old_state, SessionState::TERMINATING);

        // Remove from sessions
        sessions_.erase(it);

        return true;
    } catch (const std::exception& e) {
        notifySessionError(session_id, "Failed to force terminate session: " + std::string(e.what()));
        return false;
    }
}

void SessionManager::setSessionStateCallback(SessionStateCallback callback) {
    state_callback_ = callback;
}

void SessionManager::setSessionMetricsCallback(SessionMetricsCallback callback) {
    metrics_callback_ = callback;
}

void SessionManager::setSessionErrorCallback(SessionErrorCallback callback) {
    error_callback_ = callback;
}

void SessionManager::setResourceChangeCallback(ResourceChangeCallback callback) {
    resource_callback_ = callback;
}

bool SessionManager::canCreateSession(const SessionResourceRequest& resource_request) const {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    if (sessions_.size() >= static_cast<size_t>(max_sessions_)) {
        return false;
    }

    return canAllocateResources(resource_request);
}

SystemResources SessionManager::getAvailableResources() const {
    std::lock_guard<std::mutex> lock(resources_mutex_);

    SystemResources available = system_resources_;

    // Calculate remaining resources
    for (uint32_t device_id : allocated_input_devices_) {
        // This would subtract device-specific resources
    }

    // Subtract allocated DSP cores
    available.max_dsp_cores -= allocated_dsp_cores_;

    // Subtract allocated memory
    available.max_memory_mb -= allocated_memory_mb_;

    return available;
}

bool SessionManager::setResourceConstraints(const SystemResources& constraints) {
    std::lock_guard<std::mutex> lock(resources_mutex_);

    system_resources_ = constraints;
    max_sessions_ = constraints.max_concurrent_sessions;

    return true;
}

bool SessionManager::setSessionAutoSuspend(uint64_t session_id, bool enable, int timeout) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return false;
    }

    it->second->info.auto_suspend = enable;
    it->second->info.timeout_seconds = timeout;

    return true;
}

SessionStatistics SessionManager::getSessionStatistics(uint64_t session_id) const {
    return calculateSessionStatistics(session_id);
}

std::string SessionManager::exportSessionConfiguration(uint64_t session_id) const {
    auto session = getSession(session_id);
    if (!session) {
        return "{}";
    }

    return session_utils::sessionInfoToJSON(*session);
}

uint64_t SessionManager::importSessionConfiguration(const std::string& config_json) {
    try {
        SessionInfo session_info = session_utils::sessionInfoFromJSON(config_json);
        return createSession(session_info);
    } catch (const std::exception& e) {
        return 0;
    }
}

bool SessionManager::createSessionTemplate(uint64_t session_id, const std::string& template_name) {
    auto session = getSession(session_id);
    if (!session) {
        return false;
    }

    std::lock_guard<std::mutex> lock(templates_mutex_);
    session_templates_[template_name] = *session;

    // Save templates
    saveSessionTemplates();

    return true;
}

uint64_t SessionManager::createSessionFromTemplate(const std::string& template_name, const std::string& session_name) {
    std::lock_guard<std::mutex> lock(templates_mutex_);

    auto it = session_templates_.find(template_name);
    if (it == session_templates_.end()) {
        return 0;
    }

    SessionInfo session_info = it->second;
    session_info.session_name = session_name;
    session_info.session_id = 0; // Will be assigned by createSession
    session_info.state = SessionState::INACTIVE;
    session_info.creation_time = std::chrono::steady_clock::now();
    session_info.last_activity = session_info.creation_time;

    return createSession(session_info);
}

bool SessionManager::validateSessionConfiguration(const SessionInfo& session_info) const {
    // Validate session name
    if (!session_utils::isValidSessionName(session_info.session_name)) {
        return false;
    }

    // Validate process and user IDs
    if (!session_utils::isValidProcessId(session_info.owner_process_id) ||
        !session_utils::isValidUserId(session_info.owner_user_id)) {
        return false;
    }

    // Validate resource request
    if (!validateResourceRequest(session_info.resource_request)) {
        return false;
    }

    // Validate timeout
    if (session_info.timeout_seconds < 0 || session_info.timeout_seconds > 86400) {
        return false;
    }

    // Validate priority and isolation combination
    if (session_info.priority == SessionPriority::CRITICAL &&
        session_info.isolation == SessionIsolation::NONE) {
        return false; // Critical sessions need some isolation
    }

    return true;
}

SystemHealthStatus SessionManager::getSystemHealthStatus() const {
    return calculateSystemHealth();
}

// Private methods
bool SessionManager::allocateResources(uint64_t session_id, const SessionResourceRequest& request) {
    std::lock_guard<std::mutex> lock(resources_mutex_);

    // Mark resources as allocated
    allocated_dsp_cores_ += std::max(0, request.dsp_load_percent / 25); // Rough estimate
    allocated_memory_mb_ += request.memory_mb;

    return true;
}

void SessionManager::deallocateResources(uint64_t session_id) {
    std::lock_guard<std::mutex> lock(resources_mutex_);
    std::lock_guard<std::mutex> sessions_lock(sessions_mutex_);

    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
        const auto& request = it->second->info.resource_request;

        // Free resources
        allocated_dsp_cores_ -= std::max(0, request.dsp_load_percent / 25);
        allocated_memory_mb_ -= request.memory_mb;
        if (allocated_memory_mb < 0) allocated_memory_mb = 0;
        if (allocated_dsp_cores_ < 0) allocated_dsp_cores_ = 0;

        // Free devices
        for (uint32_t device_id : it->second->info.allocation.allocated_input_devices) {
            allocated_input_devices_.erase(device_id);
        }
        for (uint32_t device_id : it->second->info.allocation.allocated_output_devices) {
            allocated_output_devices_.erase(device_id);
        }
    }
}

bool SessionManager::canAllocateResources(const SessionResourceRequest& request) const {
    std::lock_guard<std::mutex> lock(resources_mutex_);

    // Check DSP cores
    int required_cores = std::max(0, request.dsp_load_percent / 25);
    if (allocated_dsp_cores_ + required_cores > system_resources_.max_dsp_cores) {
        return false;
    }

    // Check memory
    if (allocated_memory_mb_ + request.memory_mb > system_resources_.max_memory_mb) {
        return false;
    }

    // Check sample rate
    if (request.sample_rate < system_resources_.min_sample_rate ||
        request.sample_rate > system_resources_.max_sample_rate) {
        return false;
    }

    // Check buffer size
    if (request.buffer_size < system_resources_.min_buffer_size ||
        request.buffer_size > system_resources_.max_buffer_size) {
        return false;
    }

    // Check latency
    if (request.target_latency_ms > system_resources_.max_latency_ms) {
        return false;
    }

    return true;
}

void SessionManager::updateSessionPriority(uint64_t session_id, SessionPriority priority) {
    std::lock_guard<std::mutex> lock(scheduling_mutex_);

    // Update priority queue
    // This would involve removing and re-inserting the session in the priority queue
    scheduleSessions();
}

void SessionManager::scheduleSessions() {
    std::lock_guard<std::mutex> lock(scheduling_mutex_);

    // Implement session scheduling based on the policy
    // This would affect which sessions get CPU time and when
}

void SessionManager::cleanupInactiveSessions() {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    auto now = std::chrono::steady_clock::now();
    std::vector<uint64_t> sessions_to_remove;

    for (auto& [session_id, session] : sessions_) {
        bool should_remove = false;
        std::string reason;

        // Check timeout for inactive sessions
        if (session->info.auto_suspend &&
            (session->info.state == SessionState::INACTIVE ||
             session->info.state == SessionState::PAUSED)) {

            auto inactive_duration = std::chrono::duration_cast<std::chrono::seconds>(
                now - session->info.last_activity).count();

            if (inactive_duration > session->info.timeout_seconds) {
                should_remove = true;
                reason = "Session timeout";
            }
        }

        // Check for sessions in error state
        if (session->info.state == SessionState::ERROR) {
            auto error_duration = std::chrono::duration_cast<std::chrono::seconds>(
                now - session->info.last_activity).count();

            if (error_duration > 300) { // 5 minutes in error state
                should_remove = true;
                reason = "Session in error state too long";
            }
        }

        // Check for excessive errors
        if (session->error_count > 100) {
            should_remove = true;
            reason = "Excessive errors";
        }

        if (should_remove) {
            sessions_to_remove.push_back(session_id);
            notifySessionError(session_id, reason);
        }
    }

    // Remove sessions
    for (uint64_t session_id : sessions_to_remove) {
        forceTerminateSession(session_id, "Automatic cleanup: " + sessions_to_remove.empty() ? "" : "timeout");
    }
}

void SessionManager::updateAllSessionMetrics() {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_metrics_update_).count() < METRICS_UPDATE_INTERVAL_MS) {
        return;
    }

    for (auto& [session_id, session] : sessions_) {
        session->info.metrics = calculateSessionMetrics(session_id);
        session->info.metrics.last_update = now;

        // Store in history
        historical_metrics_.push_back(session->info.metrics);
        if (historical_metrics_.size() > MAX_HISTORY_SIZE) {
            historical_metrics_.erase(historical_metrics_.begin());
        }
    }

    last_metrics_update_ = now;
}

void SessionManager::sessionMonitoringLoop(uint64_t session_id) {
    auto session_it = sessions_.find(session_id);
    if (session_it == sessions_.end()) {
        return;
    }

    auto& session = *session_it;

    while (session.monitoring_active) {
        try {
            // Update session metrics
            session.info.metrics = calculateSessionMetrics(session_id);

            // Check for errors
            if (session.info.metrics.audio_underruns > 0 || session.info.metrics.audio_overruns > 0) {
                session.error_count++;
                if (session.error_count % 10 == 0) { // Report every 10 errors
                    notifySessionError(session_id, "Audio underruns/overruns detected");
                }
            }

            // Check for high CPU usage
            if (session.info.metrics.average_cpu_usage_percent > DEFAULT_CPU_USAGE_THRESHOLD) {
                session.error_count++;
                if (session.error_count % 20 == 0) { // Report every 20 high CPU events
                    notifySessionError(session_id, "High CPU usage detected");
                }
            }

            // Update activity timestamp
            if (session.info.state == SessionState::ACTIVE) {
                session.last_activity = std::chrono::steady_clock::now();
            }

        } catch (const std::exception& e) {
            notifySessionError(session_id, "Monitoring error: " + std::string(e.what()));
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(MONITORING_INTERVAL_MS));
    }
}

void SessionManager::metricsUpdateLoop() {
    while (metrics_active_) {
        try {
            updateAllSessionMetrics();

            // Calculate system metrics
            SystemMetrics system_metrics = calculateSystemMetrics();

            // Notify metrics callback
            if (metrics_callback_) {
                for (const auto& [session_id, session] : sessions_) {
                    metrics_callback_(session->info.metrics);
                }
            }

        } catch (const std::exception& e) {
            // Log error but continue monitoring
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(METRICS_UPDATE_INTERVAL_MS));
    }
}

void SessionManager::cleanupLoop() {
    while (cleanup_active_) {
        try {
            cleanupInactiveSessions();
        } catch (const std::exception& e) {
            // Log error but continue cleanup
        }

        std::this_thread::sleep_for(std::chrono::seconds(CLEANUP_INTERVAL_SECONDS));
    }
}

void SessionManager::notifySessionStateChange(uint64_t session_id, SessionState old_state, SessionState new_state) {
    if (state_callback_) {
        state_callback_(session_id, old_state, new_state);
    }
}

void SessionManager::notifySessionError(uint64_t session_id, const std::string& error_message) {
    if (error_callback_) {
        error_callback_(session_id, error_message);
    }
}

void SessionManager::notifyResourceChange(uint64_t session_id, const SessionAllocation& allocation) {
    if (resource_callback_) {
        resource_callback_(session_id, allocation);
    }
}

SessionMetrics SessionManager::calculateSessionMetrics(uint64_t session_id) const {
    SessionMetrics metrics;
    metrics.session_id = session_id;

    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return metrics;
    }

    const auto& session = *it->second;

    // Basic info
    metrics.type = session.info.type;
    metrics.state = session.info.state;
    metrics.start_time = session.info.metrics.start_time;
    metrics.end_time = session.info.metrics.end_time;

    // Calculate duration
    auto now = std::chrono::steady_clock::now();
    if (session.info.state == SessionState::ACTIVE && metrics.start_time.time_since_epoch().count() > 0) {
        metrics.active_duration_seconds = std::chrono::duration_cast<std::chrono::seconds>(
            now - metrics.start_time).count();
    }

    if (metrics.start_time.time_since_epoch().count() > 0) {
        metrics.total_duration_seconds = std::chrono::duration_cast<std::chrono::seconds>(
            now - session.info.creation_time).count();
    }

    // Audio metrics (would be collected from audio engine)
    metrics.processed_frames = session.process_count;
    metrics.average_latency_ms = session.info.allocation.actual_latency_ms;
    metrics.peak_latency_ms = session.info.allocation.actual_latency_ms * 1.5; // Estimate

    // CPU usage (simplified)
    metrics.average_cpu_usage_percent = session.info.resource_request.dsp_load_percent;
    metrics.peak_cpu_usage_percent = metrics.average_cpu_usage_percent * 1.2;

    // Memory usage
    metrics.peak_memory_usage_mb = session.info.resource_request.memory_mb;

    // Error counts
    metrics.audio_underruns = session.info.metrics.audio_underruns;
    metrics.audio_overruns = session.info.metrics.audio_overruns;
    metrics.dropouts = session.info.metrics.dropouts;

    // Audio levels (simplified)
    metrics.peak_level_dbfs = -3.0f; // Example value
    metrics.average_level_dbfs = -12.0f; // Example value
    metrics.signal_to_noise_ratio_db = 90.0; // Example value

    // Per-channel metrics
    metrics.per_channel_cpu_usage.resize(session.info.resource_request.input_channels + session.info.resource_request.output_channels);
    metrics.per_channel_levels_dbfs.resize(session.info.resource_request.input_channels + session.info.resource_request.output_channels);

    for (size_t i = 0; i < metrics.per_channel_cpu_usage.size(); ++i) {
        metrics.per_channel_cpu_usage[i] = metrics.average_cpu_usage_percent / metrics.per_channel_cpu_usage.size();
        metrics.per_channel_levels_dbfs[i] = metrics.average_level_dbfs;
    }

    metrics.last_update = now;

    return metrics;
}

bool SessionManager::validateResourceRequest(const SessionResourceRequest& request) const {
    if (request.input_channels < 0 || request.input_channels > system_resources_.max_input_channels) {
        return false;
    }

    if (request.output_channels < 0 || request.output_channels > system_resources_.max_output_channels) {
        return false;
    }

    if (request.sample_rate < system_resources_.min_sample_rate ||
        request.sample_rate > system_resources_.max_sample_rate) {
        return false;
    }

    if (request.buffer_size < system_resources_.min_buffer_size ||
        request.buffer_size > system_resources_.max_buffer_size) {
        return false;
    }

    if (request.dsp_load_percent < 0 || request.dsp_load_percent > 100) {
        return false;
    }

    if (request.memory_mb < 0 || request.memory_mb > system_resources_.max_memory_mb) {
        return false;
    }

    return true;
}

SessionAllocation SessionManager::allocateSessionResources(const SessionResourceRequest& request) {
    SessionAllocation allocation;

    // Find compatible sample rate and buffer size
    allocation.actual_sample_rate = request.sample_rate;
    allocation.actual_buffer_size = request.buffer_size;
    allocation.actual_bit_depth = request.bit_depth;

    // Calculate actual latency
    allocation.actual_latency_ms = audio_routing_utils::calculateLatency(request.buffer_size, request.sample_rate);

    // Allocate DSP cores (simplified)
    allocation.allocated_dsp_cores = std::max(0, request.dsp_load_percent / 25);

    // Allocate memory
    allocation.allocated_memory_mb = request.memory_mb;

    // Find available devices
    if (!request.required_devices.empty()) {
        // Use required devices
        // This would involve checking device availability
    } else {
        // Use preferred or default devices
        allocation.allocated_input_devices.push_back(0); // Default input
        allocation.allocated_output_devices.push_back(0); // Default output
    }

    allocation.allocation_time = std::chrono::steady_clock::now();
    allocation.resources_sufficient = true;
    allocation.allocation_details = "Resources allocated successfully";

    return allocation;
}

void SessionManager::deallocateSessionResources(const SessionAllocation& allocation) {
    // This would actually deallocate the resources
    // For now, it's handled in deallocateResources()
}

std::vector<uint64_t> SessionManager::getSessionsToPreempt(SessionPriority min_priority, const SessionResourceRequest& required) {
    std::vector<uint64_t> sessions_to_preempt;

    std::lock_guard<std::mutex> lock(sessions_mutex_);
    for (const auto& [session_id, session] : sessions_) {
        if (session->info.priority < min_priority &&
            (session->info.state == SessionState::ACTIVE ||
             session->info.state == SessionState::PAUSED)) {
            sessions_to_preempt.push_back(session_id);
        }
    }

    // Sort by priority (lowest first)
    std::sort(sessions_to_preempt.begin(), sessions_to_preempt.end(),
              [this](uint64_t a, uint64_t b) {
                  auto session_a = sessions_.at(a);
                  auto session_b = sessions_.at(b);
                  return session_a->info.priority < session_b->info.priority;
              });

    return sessions_to_preempt;
}

bool SessionManager::preemptSessionsInternal(const std::vector<uint64_t>& session_ids) {
    for (uint64_t session_id : session_ids) {
        if (!suspendSession(session_id, "Resource preemption")) {
            return false;
        }
    }
    return true;
}

uint64_t SessionManager::generateSessionId() const {
    static std::atomic<uint64_t> counter{1};
    return counter.fetch_add(1);
}

void SessionManager::initializeSessionEngine(SessionInternal& session) {
    session.engine = std::make_unique<MultiChannelEngine>();

    AudioSession engine_session;
    engine_session.sample_rate = session.info.allocation.actual_sample_rate;
    engine_session.buffer_size = session.info.allocation.actual_buffer_size;
    engine_session.channels = session.info.resource_request.input_channels + session.info.resource_request.output_channels;
    engine_session.bit_depth = session.info.allocation.actual_bit_depth;
    engine_session.name = session.info.session_name + " Engine";

    if (!session.engine->initialize(engine_session)) {
        throw std::runtime_error("Failed to initialize audio engine");
    }

    session.info.allocation.engine_id = session.engine.get(); // Store engine ID
}

void SessionManager::initializeSessionRouter(SessionInternal& session) {
    session.router = std::make_unique<AudioRouter>();

    RoutingConfig config;
    config.sample_rate = session.info.allocation.actual_sample_rate;
    config.buffer_size = session.info.allocation.actual_buffer_size;
    config.max_channels = session.info.resource_request.input_channels + session.info.resource_request.output_channels;
    config.target_latency_ms = session.info.resource_request.target_latency_ms;

    if (!session.router->initialize(config)) {
        throw std::runtime_error("Failed to initialize audio router");
    }

    session.info.allocation.router_id = session.router.get(); // Store router ID
}

void SessionManager::initializeSessionDeviceManager(SessionInternal& session) {
    session.device_manager = std::make_unique<AudioDeviceManager>();

    if (!session.device_manager->initialize()) {
        throw std::runtime_error("Failed to initialize device manager");
    }

    // Configure devices based on allocation
    for (uint32_t device_id : session.info.allocation.allocated_input_devices) {
        session.device_manager->addInputDevice(device_id);
    }
    for (uint32_t device_id : session.info.allocation.allocated_output_devices) {
        session.device_manager->addOutputDevice(device_id);
    }
}

void SessionManager::startSessionMonitoring(SessionInternal& session) {
    session.monitoring_active = true;
    session.monitoring_thread = std::thread(&SessionManager::sessionMonitoringLoop, this, session.info.session_id);
}

void SessionManager::stopSessionMonitoring(SessionInternal& session) {
    session.monitoring_active = false;
    if (session.monitoring_thread.joinable()) {
        session.monitoring_thread.join();
    }
}

bool SessionManager::isSessionActive(const SessionInfo& session) const {
    return session.state == SessionState::ACTIVE ||
           session.state == SessionState::PAUSED ||
           session.state == SessionState::SUSPENDED;
}

void SessionManager::updateSessionActivity(uint64_t session_id) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
        it->second->info.last_activity = std::chrono::steady_clock::now();
        it->second->process_count++;

        // Notify activity
        std::unique_lock<std::mutex> activity_lock(it->second->activity_mutex);
        it->second->activity_cv.notify_all();
    }
}

SystemMetrics SessionManager::calculateSystemMetrics() const {
    SystemMetrics metrics;
    metrics.last_update = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> lock(sessions_mutex_);
    std::lock_guard<std::mutex> resource_lock(resources_mutex_);

    metrics.active_sessions = 0;
    metrics.total_processed_frames = 0;
    metrics.total_errors = 0;
    double total_cpu = 0.0;

    for (const auto& [session_id, session] : sessions_) {
        if (session->info.state == SessionState::ACTIVE) {
            metrics.active_sessions++;
        }
        if (session->info.state == SessionState::SUSPENDED) {
            metrics.suspended_sessions++;
        }

        metrics.total_processed_frames += session->info.metrics.processed_frames;
        metrics.total_errors += session->error_count;
        total_cpu += session->info.metrics.average_cpu_usage_percent;
    }

    metrics.total_cpu_usage_percent = total_cpu / std::max(1.0, static_cast<double>(sessions_.size()));
    metrics.total_memory_usage_mb = allocated_memory_mb_;

    // Calculate average latency
    if (metrics.active_sessions > 0) {
        double total_latency = 0.0;
        int latency_count = 0;
        for (const auto& [session_id, session] : sessions_) {
            if (session->info.state == SessionState::ACTIVE) {
                total_latency += session->info.metrics.average_latency_ms;
                latency_count++;
            }
        }
        metrics.average_latency_ms = total_latency / latency_count;
    }

    return metrics;
}

SystemHealthStatus SessionManager::calculateSystemHealth() const {
    SystemHealthStatus status;
    status.last_check = std::chrono::steady_clock::now();
    status.active_sessions = 0;

    SystemMetrics metrics = calculateSystemMetrics();
    status.system_load_percent = metrics.total_cpu_usage_percent;
    status.memory_usage_percent = (static_cast<double>(metrics.total_memory_usage_mb) / system_resources_.max_memory_mb) * 100.0;
    status.active_sessions = metrics.active_sessions;

    status.system_healthy = true;

    // Check system health
    if (status.system_load_percent > DEFAULT_CPU_USAGE_THRESHOLD) {
        status.warnings.push_back("High CPU usage");
        if (status.system_load_percent > 95.0) {
            status.errors.push_back("Critical CPU usage");
            status.system_healthy = false;
        }
    }

    if (status.memory_usage_percent > 80.0) {
        status.warnings.push_back("High memory usage");
        if (status.memory_usage_percent > 95.0) {
            status.errors.push_back("Critical memory usage");
            status.system_healthy = false;
        }
    }

    if (metrics.active_sessions >= system_resources_.max_concurrent_sessions) {
        status.warnings.push_back("High session count");
    }

    if (metrics.total_errors > 100) {
        status.errors.push_back("High error rate");
        status.system_healthy = false;
    }

    return status;
}

SessionStatistics SessionManager::calculateSessionStatistics(uint64_t session_id) const {
    SessionStatistics stats;
    stats.last_reset = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> lock(sessions_mutex_);

    stats.total_sessions = sessions_.size();

    for (const auto& [id, session] : sessions_) {
        if (session->info.state == SessionState::ACTIVE) {
            stats.active_sessions++;
        } else if (session->info.state == SessionState::COMPLETED) {
            stats.completed_sessions++;
        } else if (session->info.state == SessionState::ERROR ||
                  session->info.state == SessionState::TERMINATING) {
            stats.failed_sessions++;
        } else if (session->info.state == SessionState::SUSPENDED) {
            stats.preempted_sessions++;
        }

        stats.total_cpu_time_seconds += session->info.metrics.average_cpu_usage_percent *
                                        session->info.metrics.total_duration_seconds / 100.0;
        stats.total_memory_usage_mb += session->info.metrics.peak_memory_usage_mb;
        stats.total_errors += session->error_count;

        if (session->info.metrics.total_duration_seconds > 0) {
            stats.average_session_duration_seconds += session->info.metrics.total_duration_seconds;
        }
    }

    if (stats.completed_sessions > 0) {
        stats.average_session_duration_seconds /= stats.completed_sessions;
    }

    return stats;
}

bool SessionManager::loadSessionTemplates() {
    // Load session templates from file
    // This would read from a configuration file or database
    std::string templates_file = "session_templates.json";

    try {
        if (std::filesystem::exists(templates_file)) {
            std::ifstream file(templates_file);
            if (file.is_open()) {
                // Parse JSON and load templates
                // This would require a JSON library
                return true;
            }
        }
    } catch (const std::exception& e) {
        // Continue with default templates
    }

    return false;
}

bool SessionManager::saveSessionTemplates() const {
    // Save session templates to file
    std::string templates_file = "session_templates.json";

    try {
        std::ofstream file(templates_file);
        if (file.is_open()) {
            // Write JSON templates
            // This would require a JSON library
            return true;
        }
    } catch (const std::exception& e) {
        return false;
    }

    return false;
}

// SessionManagerFactory implementation
std::unique_ptr<SessionManager> SessionManagerFactory::createProfessionalWorkstationManager(int max_sessions) {
    auto constraints = createProfessionalWorkstationConstraints();
    auto manager = std::make_unique<SessionManager>();
    if (manager->initialize(max_sessions)) {
        manager->setResourceConstraints(constraints);
        return manager;
    }
    return nullptr;
}

std::unique_ptr<SessionManager> SessionManagerFactory::createStreamingServiceManager(int max_sessions) {
    auto constraints = createStreamingServiceConstraints();
    auto manager = std::make_unique<SessionManager>();
    if (manager->initialize(max_sessions)) {
        manager->setResourceConstraints(constraints);
        return manager;
    }
    return nullptr;
}

std::unique_ptr<SessionManager> SessionManagerFactory::createLivePerformanceManager(int max_sessions) {
    auto constraints = createLivePerformanceConstraints();
    auto manager = std::make_unique<SessionManager>();
    if (manager->initialize(max_sessions)) {
        manager->setResourceConstraints(constraints);
        return manager;
    }
    return nullptr;
}

std::unique_ptr<SessionManager> SessionManagerFactory::createGamingSessionManager(int max_sessions) {
    auto constraints = createGamingConstraints();
    auto manager = std::make_unique<SessionManager>();
    if (manager->initialize(max_sessions)) {
        manager->setResourceConstraints(constraints);
        return manager;
    }
    return nullptr;
}

std::unique_ptr<SessionManager> SessionManagerFactory::createBroadcastManager(int max_sessions) {
    auto constraints = createBroadcastConstraints();
    auto manager = std::make_unique<SessionManager>();
    if (manager->initialize(max_sessions)) {
        manager->setResourceConstraints(constraints);
        return manager;
    }
    return nullptr;
}

std::unique_ptr<SessionManager> SessionManagerFactory::createCustomManager(const SystemResources& constraints,
                                                                        int max_sessions) {
    auto manager = std::make_unique<SessionManager>();
    if (manager->initialize(max_sessions)) {
        manager->setResourceConstraints(constraints);
        return manager;
    }
    return nullptr;
}

SystemResources SessionManagerFactory::createProfessionalWorkstationConstraints() {
    SystemResources constraints;
    constraints.max_input_channels = 64;
    constraints.max_output_channels = 64;
    constraints.max_sample_rate = 192000;
    constraints.min_sample_rate = 44100;
    constraints.max_buffer_size = 4096;
    constraints.min_buffer_size = 32;
    constraints.max_latency_ms = 10.0;
    constraints.max_dsp_cores = 8;
    constraints.max_memory_mb = 4096;
    constraints.max_concurrent_sessions = 32;
    constraints.supported_bit_depths = {
        AudioBitDepth::INT24, AudioBitDepth::INT32,
        AudioBitDepth::FLOAT32, AudioBitDepth::FLOAT64
    };
    return constraints;
}

SystemResources SessionManagerFactory::createStreamingServiceConstraints() {
    SystemResources constraints;
    constraints.max_input_channels = 2;
    constraints.max_output_channels = 8;
    constraints.max_sample_rate = 96000;
    constraints.min_sample_rate = 44100;
    constraints.max_buffer_size = 2048;
    constraints.min_buffer_size = 128;
    constraints.max_latency_ms = 50.0;
    constraints.max_dsp_cores = 4;
    constraints.max_memory_mb = 2048;
    constraints.max_concurrent_sessions = 1000;
    constraints.supported_bit_depths = {
        AudioBitDepth::INT16, AudioBitDepth::FLOAT32
    };
    return constraints;
}

SystemResources SessionManagerFactory::createLivePerformanceConstraints() {
    SystemResources constraints;
    constraints.max_input_channels = 32;
    constraints.max_output_channels = 32;
    constraints.max_sample_rate = 96000;
    constraints.min_sample_rate = 44100;
    constraints.max_buffer_size = 512;
    constraints.min_buffer_size = 16;
    constraints.max_latency_ms = 5.0;
    constraints.max_dsp_cores = 16;
    constraints.max_memory_mb = 8192;
    constraints.max_concurrent_sessions = 16;
    constraints.supported_bit_depths = {
        AudioBitDepth::INT24, AudioBitDepth::INT32,
        AudioBitDepth::FLOAT32
    };
    return constraints;
}

SystemResources SessionManagerFactory::createGamingConstraints() {
    SystemResources constraints;
    constraints.max_input_channels = 8;
    constraints.max_output_channels = 8;
    constraints.max_sample_rate = 48000;
    constraints.min_sample_rate = 22050;
    constraints.max_buffer_size = 1024;
    constraints.min_buffer_size = 64;
    constraints.max_latency_ms = 20.0;
    constraints.max_dsp_cores = 4;
    constraints.max_memory_mb = 1024;
    constraints.max_concurrent_sessions = 8;
    constraints.supported_bit_depths = {
        AudioBitDepth::INT16, AudioBitDepth::FLOAT32
    };
    return constraints;
}

SystemResources SessionManagerFactory::createBroadcastConstraints() {
    SystemResources constraints;
    constraints.max_input_channels = 16;
    constraints.max_output_channels = 16;
    constraints.max_sample_rate = 48000;
    constraints.min_sample_rate = 44100;
    constraints.max_buffer_size = 1024;
    constraints.min_buffer_size = 64;
    constraints.max_latency_ms = 15.0;
    constraints.max_dsp_cores = 8;
    constraints.max_memory_mb = 2048;
    constraints.max_concurrent_sessions = 24;
    constraints.supported_bit_depths = {
        AudioBitDepth::INT24, AudioBitDepth::INT32,
        AudioBitDepth::FLOAT32
    };
    return constraints;
}

// Utility functions implementation
namespace session_utils {

std::string sessionTypeToString(SessionType type) {
    switch (type) {
        case SessionType::RECORDING: return "Recording";
        case SessionType::PLAYBACK: return "Playback";
        case SessionType::DUPLEX: return "Duplex";
        case SessionType::MONITORING: return "Monitoring";
        case SessionType::STREAMING: return "Streaming";
        case SessionType::PROCESSING: return "Processing";
        case SessionType::ANALYSIS: return "Analysis";
        case SessionType::MIXDOWN: return "Mixdown";
        case SessionType::MASTERING: return "Mastering";
        case SessionType::BROADCAST: return "Broadcast";
        case SessionType::CONFERENCE: return "Conference";
        case SessionType::VIRTUAL_INSTRUMENT: return "Virtual Instrument";
        case SessionType::SAMPLE_PREVIEW: return "Sample Preview";
        case SessionType::SYSTEM_AUDIO: return "System Audio";
        case SessionType::CUSTOM: return "Custom";
        default: return "Unknown";
    }
}

std::string sessionPriorityToString(SessionPriority priority) {
    switch (priority) {
        case SessionPriority::CRITICAL: return "Critical";
        case SessionPriority::HIGH: return "High";
        case SessionPriority::NORMAL: return "Normal";
        case SessionPriority::LOW: return "Low";
        case SessionPriority::BACKGROUND: return "Background";
        default: return "Unknown";
    }
}

std::string sessionStateToString(SessionState state) {
    switch (state) {
        case SessionState::INACTIVE: return "Inactive";
        case SessionState::PREPARING: return "Preparing";
        case SessionState::ACTIVE: return "Active";
        case SessionState::SUSPENDED: return "Suspended";
        case SessionState::PAUSED: return "Paused";
        case SessionState::ERROR: return "Error";
        case SessionState::TERMINATING: return "Terminating";
        case SessionState::COMPLETED: return "Completed";
        default: return "Unknown";
    }
}

std::string sessionIsolationToString(SessionIsolation isolation) {
    switch (isolation) {
        case SessionIsolation::NONE: return "None";
        case SessionIsolation::PROCESS: return "Process";
        case SessionIsolation::USER: return "User";
        case SessionIsolation::APPLICATION: return "Application";
        case SessionIsolation::DEVICE: return "Device";
        case SessionIsolation::FULL: return "Full";
        default: return "Unknown";
    }
}

std::string sessionSchedulingToString(SessionScheduling scheduling) {
    switch (scheduling) {
        case SessionScheduling::FIFO: return "FIFO";
        case SessionScheduling::PRIORITY: return "Priority";
        case SessionScheduling::ROUND_ROBIN: return "Round Robin";
        case SessionScheduling::FAIR_SHARE: return "Fair Share";
        case SessionScheduling::REAL_TIME: return "Real-time";
        case SessionScheduling::BATCH: return "Batch";
        case SessionScheduling::ADAPTIVE: return "Adaptive";
        default: return "Unknown";
    }
}

std::string resourceAllocationToString(ResourceAllocation allocation) {
    switch (allocation) {
        case ResourceAllocation::DEDICATED: return "Dedicated";
        case ResourceAllocation::SHARED: return "Shared";
        case ResourceAllocation::ON_DEMAND: return "On Demand";
        case ResourceAllocation::POOL: return "Pool";
        case ResourceAllocation::PRIORITY_BASED: return "Priority Based";
        case ResourceAllocation::BEST_EFFORT: return "Best Effort";
        default: return "Unknown";
    }
}

SessionType stringToSessionType(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "recording") return SessionType::RECORDING;
    if (lower == "playback") return SessionType::PLAYBACK;
    if (lower == "duplex") return SessionType::DUPLEX;
    if (lower == "monitoring") return SessionType::MONITORING;
    if (lower == "streaming") return SessionType::STREAMING;
    if (lower == "processing") return SessionType::PROCESSING;
    if (lower == "analysis") return SessionType::ANALYSIS;
    if (lower == "mixdown") return SessionType::MIXDOWN;
    if (lower == "mastering") return SessionType::MASTERING;
    if (lower == "broadcast") return SessionType::BROADCAST;
    if (lower == "conference") return SessionType::CONFERENCE;
    if (lower == "virtual instrument") return SessionType::VIRTUAL_INSTRUMENT;
    if (lower == "sample preview") return SessionType::SAMPLE_PREVIEW;
    if (lower == "system audio") return SessionType::SYSTEM_AUDIO;

    return SessionType::CUSTOM;
}

SessionPriority stringToSessionPriority(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "critical") return SessionPriority::CRITICAL;
    if (lower == "high") return SessionPriority::HIGH;
    if (lower == "normal") return SessionPriority::NORMAL;
    if (lower == "low") return SessionPriority::LOW;
    if (lower == "background") return SessionPriority::BACKGROUND;

    return SessionPriority::NORMAL;
}

SessionState stringToSessionState(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "inactive") return SessionState::INACTIVE;
    if (lower == "preparing") return SessionState::PREPARING;
    if (lower == "active") return SessionState::ACTIVE;
    if (lower == "suspended") return SessionState::SUSPENDED;
    if (lower == "paused") return SessionState::PAUSED;
    if (lower == "error") return SessionState::ERROR;
    if (lower == "terminating") return SessionState::TERMINATING;
    if (lower == "completed") return SessionState::COMPLETED;

    return SessionState::INACTIVE;
}

uint64_t generateSessionId() {
    static std::atomic<uint64_t> counter{1};
    auto now = std::chrono::high_resolution_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();

    return (static_cast<uint64_t>(timestamp) << 32) | counter.fetch_add(1);
}

bool isSessionStateTransitionValid(SessionState from, SessionState to) {
    // Define valid state transitions
    switch (from) {
        case SessionState::INACTIVE:
            return to == SessionState::PREPARING || to == SessionState::ERROR;

        case SessionState::PREPARING:
            return to == SessionState::ACTIVE || to == SessionState::ERROR || to == SessionState::TERMINATING;

        case SessionState::ACTIVE:
            return to == SessionState::PAUSED || to == SessionState::SUSPENDED || to == SessionState::INACTIVE || to == SessionState::ERROR || to == SessionState::TERMINATING;

        case SessionState::SUSPENDED:
            return to == SessionState::ACTIVE || to == SessionState::TERMINATING || to == SessionState::ERROR;

        case SessionState::PAUSED:
            return to == SessionState::ACTIVE || to == SessionState::TERMINATING || to == SessionState::ERROR;

        case SessionState::ERROR:
            return to == SessionState::INACTIVE || to == SessionState::TERMINATING;

        case SessionState::TERMINATING:
            return to == SessionState::COMPLETED;

        case SessionState::COMPLETED:
            return to == SessionState::INACTIVE;

        default:
            return false;
    }
}

SessionState getHighestPriorityState(const std::vector<SessionState>& states) {
    if (states.empty()) {
        return SessionState::INACTIVE;
    }

    // Define priority order (highest to lowest)
    std::vector<SessionState> priority_order = {
        SessionState::ACTIVE,
        SessionState::PAUSED,
        SessionState::SUSPENDED,
        SessionState::PREPARING,
        SessionState::ERROR,
        SessionState::TERMINATING,
        SessionState::COMPLETED,
        SessionState::INACTIVE
    };

    for (SessionState priority_state : priority_order) {
        if (std::find(states.begin(), states.end(), priority_state) != states.end()) {
            return priority_state;
        }
    }

    return SessionState::INACTIVE;
}

SessionPriority getHighestPriority(const std::vector<SessionPriority>& priorities) {
    if (priorities.empty()) {
        return SessionPriority::NORMAL;
    }

    return *std::min_element(priorities.begin(), priorities.end(),
        [](SessionPriority a, SessionPriority b) {
            return static_cast<int>(a) < static_cast<int>(b);
        });
}

bool areResourcesCompatible(const SessionResourceRequest& req1, const SessionResourceRequest& req2) {
    // Check if two resource requests can coexist
    return req1.sample_rate == req2.sample_rate &&
           req1.buffer_size == req2.buffer_size &&
           req1.bit_depth == req2.bit_depth &&
           req1.exclusive_device_access == false &&
           req2.exclusive_device_access == false;
}

SessionResourceRequest mergeResourceRequests(const std::vector<SessionResourceRequest>& requests) {
    SessionResourceRequest merged;

    if (requests.empty()) {
        return merged;
    }

    // Use the first request as base
    merged = requests[0];

    // Merge channels
    for (const auto& req : requests) {
        merged.input_channels = std::max(merged.input_channels, req.input_channels);
        merged.output_channels = std::max(merged.output_channels, req.output_channels);
        merged.dsp_load_percent += req.dsp_load_percent;
        merged.memory_mb += req.memory_mb;
    }

    // Take minimum target latency
    for (const auto& req : requests) {
        merged.target_latency_ms = std::min(merged.target_latency_ms, req.target_latency_ms);
    }

    // Any exclusive access requirement makes the merged request exclusive
    for (const auto& req : requests) {
        if (req.exclusive_device_access) {
            merged.exclusive_device_access = true;
            break;
        }
    }

    return merged;
}

SessionResourceRequest minimizeResourceRequest(const SessionResourceRequest& request) {
    SessionResourceRequest minimized = request;

    // Reduce channels to minimum required
    minimized.input_channels = std::max(1, request.input_channels / 2);
    minimized.output_channels = std::max(1, request.output_channels / 2);

    // Use smaller buffer size if possible
    minimized.buffer_size = std::max(64, request.buffer_size / 2);

    // Reduce quality
    if (minimized.bit_depth == AudioBitDepth::FLOAT64) {
        minimized.bit_depth = AudioBitDepth::FLOAT32;
    } else if (minimized.bit_depth == AudioBitDepth::INT32) {
        minimized.bit_depth = AudioBitDepth::INT24;
    }

    return minimized;
}

double calculateResourceUtilization(const SessionResourceRequest& used, const SessionResourceRequest& available) {
    if (available.input_channels == 0 || available.output_channels == 0) {
        return 1.0;
    }

    double channel_util = (static_cast<double>(used.input_channels) / available.input_channels +
                           static_cast<double>(used.output_channels) / available.output_channels) / 2.0;

    double dsp_util = available.dsp_load_percent > 0 ?
                      static_cast<double>(used.dsp_load_percent) / available.dsp_load_percent : 0.0;

    double memory_util = available.memory_mb > 0 ?
                       static_cast<double>(used.memory_mb) / available.memory_mb : 0.0;

    return std::max({channel_util, dsp_util, memory_util});
}

double calculateSessionEfficiency(const SessionMetrics& metrics) {
    if (metrics.total_duration_seconds == 0 || metrics.active_duration_seconds == 0) {
        return 0.0;
    }

    double efficiency = metrics.active_duration_seconds / metrics.total_duration_seconds;

    // Penalize high error rates
    if (metrics.processed_frames > 0) {
        double error_rate = static_cast<double>(metrics.audio_underruns + metrics.audio_overruns + metrics.dropouts) /
                          metrics.processed_frames;
        efficiency *= (1.0 - error_rate);
    }

    return std::max(0.0, std::min(1.0, efficiency));
}

double calculateSessionStability(const SessionMetrics& metrics) {
    if (metrics.processed_frames == 0) {
        return 0.0;
    }

    // Calculate stability based on error rates and consistency
    double xrun_rate = static_cast<double>(metrics.audio_underruns + metrics.audio_overruns) / metrics.processed_frames;
    double dropout_rate = static_cast<double>(metrics.dropouts) / metrics.processed_frames;

    double stability = 1.0 - (xrun_rate * 10.0) - (dropout_rate * 50.0);
    stability = std::max(0.0, std::min(1.0, stability));

    // Consider CPU consistency
    if (metrics.average_cpu_usage_percent > 0 && metrics.peak_cpu_usage_percent > 0) {
        double cpu_consistency = metrics.average_cpu_usage_percent / metrics.peak_cpu_usage_percent;
        stability *= cpu_consistency;
    }

    return stability;
}

double calculateSessionQuality(const SessionMetrics& metrics) {
    double efficiency = calculateSessionEfficiency(metrics);
    double stability = calculateSessionStability(metrics);

    // Audio quality factors
    double audio_quality = 0.0;
    if (metrics.signal_to_noise_ratio_db > 0) {
        audio_quality = std::min(1.0, metrics.signal_to_noise_ratio_db / 90.0); // 90dB = excellent
    }

    // Latency quality
    double latency_quality = 1.0;
    if (metrics.average_latency_ms > 20.0) {
        latency_quality = std::max(0.1, 20.0 / metrics.average_latency_ms);
    }

    // Weighted combination
    double quality = (efficiency * 0.3) + (stability * 0.3) + (audio_quality * 0.2) + (latency_quality * 0.2);

    return std::max(0.0, std::min(1.0, quality));
}

bool isSessionPerformingWell(const SessionMetrics& metrics) {
    double efficiency = calculateSessionEfficiency(metrics);
    double stability = calculateSessionStability(metrics);
    double quality = calculateSessionQuality(metrics);

    return efficiency > 0.8 && stability > 0.8 && quality > 0.7;
}

std::vector<std::string> detectSessionIssues(const SessionMetrics& metrics) {
    std::vector<std::string> issues;

    if (metrics.audio_underruns > 0) {
        issues.push_back("Audio underruns detected");
    }

    if (metrics.audio_overruns > 0) {
        issues.push_back("Audio overruns detected");
    }

    if (metrics.dropouts > 10) {
        issues.push_back("Excessive audio dropouts");
    }

    if (metrics.average_cpu_usage_percent > 80.0) {
        issues.push_back("High CPU usage");
    }

    if (metrics.peak_latency_ms > 50.0) {
        issues.push_back("High latency peaks");
    }

    if (metrics.peak_level_dbfs > -0.1f) {
        issues.push_back("Audio clipping detected");
    }

    if (metrics.signal_to_noise_ratio_db < 60.0) {
        issues.push_back("Low signal-to-noise ratio");
    }

    double efficiency = calculateSessionEfficiency(metrics);
    if (efficiency < 0.5) {
        issues.push_back("Low session efficiency");
    }

    return issues;
}

std::string sessionInfoToJSON(const SessionInfo& session) {
    // JSON implementation would go here
    // This is a simplified placeholder
    std::ostringstream json;
    json << "{\n";
    json << "  \"session_id\": " << session.session_id << ",\n";
    json << "  \"session_name\": \"" << session.session_name << "\",\n";
    json << "  \"type\": \"" << sessionTypeToString(session.type) << "\",\n";
    json << "  \"priority\": \"" << sessionPriorityToString(session.priority) << "\",\n";
    json << "  \"state\": \"" << sessionStateToString(session.state) << "\"\n";
    json << "}";
    return json.str();
}

std::string sessionMetricsToJSON(const SessionMetrics& metrics) {
    std::ostringstream json;
    json << "{\n";
    json << "  \"session_id\": " << metrics.session_id << ",\n";
    json << "  \"processed_frames\": " << metrics.processed_frames << ",\n";
    json << "  \"average_cpu_usage_percent\": " << metrics.average_cpu_usage_percent << ",\n";
    json << "  \"peak_cpu_usage_percent\": " << metrics.peak_cpu_usage_percent << ",\n";
    json << "  \"average_latency_ms\": " << metrics.average_latency_ms << ",\n";
    json << "  \"peak_latency_ms\": " << metrics.peak_latency_ms << "\n";
    json << "}";
    return json.str();
}

std::string resourceRequestToJSON(const SessionResourceRequest& request) {
    std::ostringstream json;
    json << "{\n";
    json << "  \"input_channels\": " << request.input_channels << ",\n";
    json << "  \"output_channels\": " << request.output_channels << ",\n";
    json << "  \"sample_rate\": " << request.sample_rate << ",\n";
    json << "  \"buffer_size\": " << request.buffer_size << ",\n";
    json << "  \"target_latency_ms\": " << request.target_latency_ms << ",\n";
    json << "  \"dsp_load_percent\": " << request.dsp_load_percent << ",\n";
    json << "  \"memory_mb\": " << request.memory_mb << "\n";
    json << "}";
    return json.str();
}

SessionInfo sessionInfoFromJSON(const std::string& json) {
    // JSON parsing implementation would go here
    SessionInfo session;
    return session;
}

SessionMetrics sessionMetricsFromJSON(const std::string& json) {
    // JSON parsing implementation would go here
    SessionMetrics metrics;
    return metrics;
}

SessionResourceRequest resourceRequestFromJSON(const std::string& json) {
    // JSON parsing implementation would go here
    SessionResourceRequest request;
    return request;
}

std::string formatDuration(uint64_t seconds) {
    uint64_t hours = seconds / 3600;
    uint64_t minutes = (seconds % 3600) / 60;
    uint64_t secs = seconds % 60;

    std::ostringstream oss;
    if (hours > 0) {
        oss << hours << "h ";
    }
    if (minutes > 0 || hours > 0) {
        oss << minutes << "m ";
    }
    oss << secs << "s";

    return oss.str();
}

std::chrono::steady_clock::time_point parseTimestamp(const std::string& timestamp) {
    // Timestamp parsing implementation would go here
    return std::chrono::steady_clock::now();
}

std::string formatTimestamp(const std::chrono::steady_clock::time_point& timestamp) {
    auto time_t = std::chrono::duration_cast<std::chrono::seconds>(
        timestamp.time_since_epoch()).count();

    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

bool isValidSessionName(const std::string& name) {
    return !name.empty() && name.length() <= 256 &&
           std::all_of(name.begin(), name.end(),
                       [](char c) { return std::isalnum(c) || std::isspace(c) || c == '_' || c == '-'; });
}

bool isValidProcessId(const std::string& process_id) {
    return !process_id.empty() && process_id.length() <= 64;
}

bool isValidUserId(const std::string& user_id) {
    return !user_id.empty() && user_id.length() <= 64;
}

bool isValidSessionId(uint64_t session_id) {
    return session_id != 0;
}

std::vector<std::string> getDefaultSessionTemplates() {
    return {"Recording", "Playback", "Duplex", "Streaming", "Processing"};
}

SessionInfo createDefaultTemplate(SessionType type) {
    SessionInfo session;
    session.type = type;
    session.session_name = sessionTypeToString(type) + " Template";
    session.priority = SessionPriority::NORMAL;
    session.isolation = SessionIsolation::PROCESS;
    session.scheduling = SessionScheduling::PRIORITY;
    session.allocation_mode = ResourceAllocation::SHARED;
    session.auto_suspend = true;
    session.timeout_seconds = DEFAULT_TIMEOUT_SECONDS;

    // Set resource requirements based on type
    switch (type) {
        case SessionType::RECORDING:
            session.resource_request.input_channels = 2;
            session.resource_request.output_channels = 0;
            session.resource_request.sample_rate = 48000;
            session.resource_request.buffer_size = 512;
            session.resource_request.target_latency_ms = 10.0;
            session.resource_request.low_latency_mode = true;
            break;

        case SessionType::PLAYBACK:
            session.resource_request.input_channels = 0;
            session.resource_request.output_channels = 2;
            session.resource_request.sample_rate = 44100;
            session.resource_request.buffer_size = 1024;
            session.resource_request.target_latency_ms = 20.0;
            break;

        case SessionType::DUPLEX:
            session.resource_request.input_channels = 2;
            session.resource_request.output_channels = 2;
            session.resource_request.sample_rate = 48000;
            session.resource_request.buffer_size = 256;
            session.resource_request.target_latency_ms = 5.0;
            session.resource_request.low_latency_mode = true;
            break;

        case SessionType::STREAMING:
            session.resource_request.input_channels = 2;
            session.resource_request.output_channels = 2;
            session.resource_request.sample_rate = 48000;
            session.resource_request.buffer_size = 1024;
            session.resource_request.target_latency_ms = 50.0;
            break;

        default:
            session.resource_request.input_channels = 2;
            session.resource_request.output_channels = 2;
            session.resource_request.sample_rate = 48000;
            session.resource_request.buffer_size = 512;
            session.resource_request.target_latency_ms = 10.0;
            break;
    }

    session.resource_request.bit_depth = AudioBitDepth::FLOAT32;
    session.resource_request.max_acceptable_latency_ms = session.resource_request.target_latency_ms * 2.0;
    session.resource_request.dsp_load_percent = 20;
    session.resource_request.memory_mb = 256;

    return session;
}

bool validateTemplate(const SessionInfo& template_info) {
    return isValidSessionName(template_info.session_name) &&
           template_info.type != SessionType::CUSTOM &&
           validateResourceRequest(template_info.resource_request);
}

} // namespace session_utils

} // namespace audio
} // namespace core
} // namespace vortex