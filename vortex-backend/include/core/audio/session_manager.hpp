#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <array>
#include <chrono>
#include <functional>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <queue>
#include <condition_variable>
#include "core/audio/multi_channel_engine.hpp"
#include "core/audio/device_manager.hpp"
#include "core/audio/audio_routing.hpp"

namespace vortex {
namespace core {
namespace audio {

/**
 * Audio Session Management System
 * Manages concurrent audio sessions with resource allocation, scheduling,
 * and isolation. Supports multiple applications and users sharing audio resources.
 */

enum class SessionType {
    RECORDING,          ///< Recording session
    PLAYBACK,           ///< Playback session
    DUPLEX,             ///< Simultaneous recording and playback
    MONITORING,         ///< Monitoring only
    STREAMING,          ///< Network streaming session
    PROCESSING,         ///< Audio processing session
    ANALYSIS,           ///< Audio analysis session
    MIXDOWN,            ///< Mixdown session
    MASTERING,          ///< Mastering session
    BROADCAST,          ///< Live broadcast session
    CONFERENCE,         ///< Audio conference session
    VIRTUAL_INSTRUMENT, ///< Virtual instrument session
    SAMPLE_PREVIEW,     ///< Sample preview session
    SYSTEM_AUDIO,       ///< System audio session
    CUSTOM              ///< Custom session type
};

enum class SessionPriority {
    CRITICAL,           ///< Critical session (interrupts others)
    HIGH,               ///< High priority session
    NORMAL,             ///< Normal priority
    LOW,                ///< Low priority
    BACKGROUND          ///< Background processing only
};

enum class SessionState {
    INACTIVE,           ///< Session not active
    PREPARING,          ///< Session preparing resources
    ACTIVE,             ///< Session active
    SUSPENDED,          ///< Session temporarily suspended
    PAUSED,             ///< Session paused by user
    ERROR,              ///< Session in error state
    TERMINATING,        ///< Session terminating
    COMPLETED           ///< Session completed successfully
};

enum class SessionIsolation {
    NONE,               ///< No isolation
    PROCESS,            ///< Process-level isolation
    USER,               ///< User-level isolation
    APPLICATION,        ///< Application-level isolation
    DEVICE,             ///< Device-level isolation
    FULL                ///< Full isolation
};

enum class SessionScheduling {
    FIFO,               ///< First-in-first-out scheduling
    PRIORITY,           ///< Priority-based scheduling
    ROUND_ROBIN,        ///< Round-robin scheduling
    FAIR_SHARE,         ///< Fair share scheduling
    REAL_TIME,          ///< Real-time scheduling
    BATCH,              ///< Batch processing
    ADAPTIVE            ///< Adaptive scheduling
};

enum class ResourceAllocation {
    DEDICATED,          ///< Dedicated resources
    SHARED,             ///< Shared resources
    ON_DEMAND,          ///< Resources on demand
    POOL,               ///< Resource pool
    PRIORITY_BASED,     ///< Priority-based allocation
    BEST_EFFORT         ///< Best effort allocation
};

struct SessionResourceRequest {
    int input_channels = 2;                    ///< Requested input channels
    int output_channels = 2;                   ///< Requested output channels
    int sample_rate = 48000;                   ///< Requested sample rate
    int buffer_size = 512;                     ///< Requested buffer size
    AudioBitDepth bit_depth = AudioBitDepth::FLOAT32; ///< Bit depth
    double target_latency_ms = 10.0;            ///< Target latency
    double max_acceptable_latency_ms = 50.0;   ///< Max acceptable latency
    int dsp_load_percent = 0;                  ///< DSP load requirement (0-100)
    int memory_mb = 0;                        ///< Memory requirement in MB
    bool exclusive_device_access = false;      ///< Exclusive device access required
    bool low_latency_mode = true;              ///< Low latency mode required
    bool persistent_resources = false;         ///< Keep resources when inactive
    std::vector<std::string> required_devices; ///< Specific device requirements
    std::vector<std::string> preferred_devices; ///< Preferred device list
};

struct SessionAllocation {
    std::vector<uint32_t> allocated_input_devices;    ///< Allocated input devices
    std::vector<uint32_t> allocated_output_devices;   ///< Allocated output devices
    int actual_sample_rate = 0;                      ///< Actual sample rate
    int actual_buffer_size = 0;                       ///< Actual buffer size
    AudioBitDepth actual_bit_depth = AudioBitDepth::FLOAT32; ///< Actual bit depth
    double actual_latency_ms = 0.0;                   ///< Actual latency achieved
    int allocated_dsp_cores = 0;                      ///< Allocated DSP cores
    size_t allocated_memory_mb = 0;                   ///< Allocated memory in MB
    std::chrono::steady_clock::time_point allocation_time;
    uint32_t router_id = 0;                          ///< Router instance ID
    uint32_t engine_id = 0;                          ///< Engine instance ID
    bool resources_sufficient = true;                 ///< Resource sufficiency
    std::string allocation_details;                  ///< Allocation details
};

struct SessionMetrics {
    uint64_t session_id = 0;                        ///< Session ID
    SessionType type = SessionType::CUSTOM;          ///< Session type
    SessionState state = SessionState::INACTIVE;     ///< Current state
    std::chrono::steady_clock::time_point start_time; ///< Session start time
    std::chrono::steady_clock::time_point end_time;   ///< Session end time
    uint64_t total_duration_seconds = 0;             ///< Total duration in seconds
    uint64_t active_duration_seconds = 0;            ///< Active duration in seconds
    uint64_t processed_frames = 0;                   ///< Total processed frames
    double average_cpu_usage_percent = 0.0;          ///< Average CPU usage
    double peak_cpu_usage_percent = 0.0;             ///< Peak CPU usage
    size_t peak_memory_usage_mb = 0;                 ///< Peak memory usage
    uint64_t audio_underruns = 0;                    ///< Audio underruns
    uint64_t audio_overruns = 0;                     ///< Audio overruns
    double average_latency_ms = 0.0;                 ///< Average latency
    double peak_latency_ms = 0.0;                    ///< Peak latency
    uint32_t dropouts = 0;                           ///< Audio dropouts
    double signal_to_noise_ratio_db = 0.0;           ///< Signal-to-noise ratio
    float peak_level_dbfs = -INFINITY;               ///< Peak level (dBFS)
    float average_level_dbfs = -INFINITY;            ///< Average level (dBFS)
    std::vector<double> per_channel_cpu_usage;       ///< Per-channel CPU usage
    std::vector<float> per_channel_levels_dbfs;      ///< Per-channel levels
    std::chrono::steady_clock::time_point last_update;
};

struct SessionInfo {
    uint64_t session_id = 0;                        ///< Unique session ID
    std::string session_name;                        ///< Session name
    std::string description;                         ///< Session description
    std::string owner_process_id;                   ///< Owner process ID
    std::string owner_user_id;                       ///< Owner user ID
    std::string owner_application;                   ///< Owner application name
    SessionType type = SessionType::CUSTOM;          ///< Session type
    SessionPriority priority = SessionPriority::NORMAL; ///< Priority level
    SessionState state = SessionState::INACTIVE;     ///< Current state
    SessionIsolation isolation = SessionIsolation::NONE; ///< Isolation level
    SessionScheduling scheduling = SessionScheduling::PRIORITY; ///< Scheduling mode
    ResourceAllocation allocation_mode = ResourceAllocation::SHARED; ///< Resource allocation
    SessionResourceRequest resource_request;        ///< Resource requirements
    SessionAllocation allocation;                   ///< Resource allocation
    SessionMetrics metrics;                         ///< Session metrics
    bool auto_suspend = true;                       ///< Auto-suspend when inactive
    int timeout_seconds = 300;                      ///< Inactivity timeout
    std::vector<std::string> tags;                  ///< Session tags
    std::unordered_map<std::string, std::string> metadata; ///< Session metadata
    std::chrono::steady_clock::time_point creation_time;
    std::chrono::steady_clock::time_point last_activity;
};

using SessionStateCallback = std::function<void(uint64_t session_id, SessionState old_state, SessionState new_state)>;
using SessionMetricsCallback = std::function<void(const SessionMetrics& metrics)>;
using SessionErrorCallback = std::function<void(uint64_t session_id, const std::string& error_message)>;
using ResourceChangeCallback = std::function<void(uint64_t session_id, const SessionAllocation& allocation)>;

/**
 * Audio Session Manager
 * Manages multiple concurrent audio sessions with resource allocation and scheduling
 */
class SessionManager {
public:
    SessionManager();
    ~SessionManager();

    /**
     * Initialize session manager
     * @param max_sessions Maximum concurrent sessions
     * @param system_resources System resource constraints
     * @return True if initialization successful
     */
    bool initialize(int max_sessions = 64);

    /**
     * Shutdown session manager
     */
    void shutdown();

    /**
     * Create new audio session
     * @param session_info Session configuration
     * @return Session ID if successful, 0 otherwise
     */
    uint64_t createSession(const SessionInfo& session_info);

    /**
     * Destroy audio session
     * @param session_id Session ID
     * @return True if session destroyed successfully
     */
    bool destroySession(uint64_t session_id);

    /**
     * Get session information
     * @param session_id Session ID
     * @return Session information if found
     */
    std::optional<SessionInfo> getSession(uint64_t session_id) const;

    /**
     * Update session information
     * @param session_id Session ID
     * @param session_info Updated session information
     * @return True if update successful
     */
    bool updateSession(uint64_t session_id, const SessionInfo& session_info);

    /**
     * Activate session
     * @param session_id Session ID
     * @return True if activation successful
     */
    bool activateSession(uint64_t session_id);

    /**
     * Deactivate session
     * @param session_id Session ID
     * @return True if deactivation successful
     */
    bool deactivateSession(uint64_t session_id);

    /**
     * Pause session
     * @param session_id Session ID
     * @return True if pause successful
     */
    bool pauseSession(uint64_t session_id);

    /**
     * Resume session
     * @param session_id Session ID
     * @return True if resume successful
     */
    bool resumeSession(uint64_t session_id);

    /**
     * Suspend session
     * @param session_id Session ID
     * @param reason Suspension reason
     * @return True if suspension successful
     */
    bool suspendSession(uint64_t session_id, const std::string& reason = "System suspension");

    /**
     * Get all active sessions
     * @return List of active sessions
     */
    std::vector<SessionInfo> getActiveSessions() const;

    /**
     * Get sessions by type
     * @param type Session type
     * @return List of sessions of specified type
     */
    std::vector<SessionInfo> getSessionsByType(SessionType type) const;

    /**
     * Get sessions by priority
     * @param priority Session priority
     * @return List of sessions with specified priority
     */
    std::vector<SessionInfo> getSessionsByPriority(SessionPriority priority) const;

    /**
     * Get sessions for process
     * @param process_id Process ID
     * @return List of sessions owned by process
     */
    std::vector<SessionInfo> getSessionsByProcess(const std::string& process_id) const;

    /**
     * Get sessions for user
     * @param user_id User ID
     * @return List of sessions owned by user
     */
    std::vector<SessionInfo> getSessionsByUser(const std::string& user_id) const;

    /**
     * Find sessions by name
     * @param name Session name (partial match allowed)
     * @return Matching sessions
     */
    std::vector<SessionInfo> findSessionsByName(const std::string& name) const;

    /**
     * Get session metrics
     * @param session_id Session ID
     * @return Session metrics if found
     */
    std::optional<SessionMetrics> getSessionMetrics(uint64_t session_id) const;

    /**
     * Get all session metrics
     * @return List of all session metrics
     */
    std::vector<SessionMetrics> getAllSessionMetrics() const;

    /**
     * Get system-wide metrics
     * @return System metrics
     */
    SystemMetrics getSystemMetrics() const;

    /**
     * Set session priority
     * @param session_id Session ID
     * @param priority New priority level
     * @return True if priority set successfully
     */
    bool setSessionPriority(uint64_t session_id, SessionPriority priority);

    /**
     * Request resource change for session
     * @param session_id Session ID
     * @param new_request New resource request
     * @return True if change request accepted
     */
    bool requestResourceChange(uint64_t session_id, const SessionResourceRequest& new_request);

    /**
     * Preempt lower priority sessions
     * @param session_id High-priority session ID
     * @param required_resources Resources needed
     * @return True if preemption successful
     */
    bool preemptSessions(uint64_t session_id, const SessionResourceRequest& required_resources);

    /**
     * Set session scheduling policy
     * @param session_id Session ID
     * @param scheduling Scheduling mode
     * @return True if scheduling set successfully
     */
    bool setSessionScheduling(uint64_t session_id, SessionScheduling scheduling);

    /**
     * Set session isolation level
     * @param session_id Session ID
     * @param isolation Isolation level
     * @return True if isolation set successfully
     */
    bool setSessionIsolation(uint64_t session_id, SessionIsolation isolation);

    /**
     * Force session termination
     * @param session_id Session ID
     * @param reason Termination reason
     * @return True if termination successful
     */
    bool forceTerminateSession(uint64_t session_id, const std::string& reason = "System termination");

    /**
     * Register session state callback
     * @param callback State change callback
     */
    void setSessionStateCallback(SessionStateCallback callback);

    /**
     * Register session metrics callback
     * @param callback Metrics update callback
     */
    void setSessionMetricsCallback(SessionMetricsCallback callback);

    /**
     * Register session error callback
     * @param callback Error callback
     */
    void setSessionErrorCallback(SessionErrorCallback callback);

    /**
     * Register resource change callback
     * @param callback Resource change callback
     */
    void setResourceChangeCallback(ResourceChangeCallback callback);

    /**
     * Check if session can be created
     * @param resource_request Resource requirements
     * @return True if resources are available
     */
    bool canCreateSession(const SessionResourceRequest& resource_request) const;

    /**
     * Get available resources
     * @return Available system resources
     */
    SystemResources getAvailableResources() const;

    /**
     * Set resource constraints
     * @param constraints Resource constraints
     * @return True if constraints set successfully
     */
    bool setResourceConstraints(const SystemResources& constraints);

    /**
     * Enable/disable session auto-suspension
     * @param session_id Session ID
     * @param enable Auto-suspension enabled
     * @param timeout Suspension timeout in seconds
     * @return True if configuration successful
     */
    bool setSessionAutoSuspend(uint64_t session_id, bool enable, int timeout = 300);

    /**
     * Get session statistics
     * @param session_id Session ID
     * @return Session statistics
     */
    SessionStatistics getSessionStatistics(uint64_t session_id) const;

    /**
     * Export session configuration
     * @param session_id Session ID
     * @return JSON string with session configuration
     */
    std::string exportSessionConfiguration(uint64_t session_id) const;

    /**
     * Import session configuration
     * @param config_json JSON configuration string
     * @return Session ID if import successful, 0 otherwise
     */
    uint64_t importSessionConfiguration(const std::string& config_json);

    /**
     * Create session template
     * @param session_id Session ID
     * @param template_name Template name
     * @return True if template created successfully
     */
    bool createSessionTemplate(uint64_t session_id, const std::string& template_name);

    /**
     * Create session from template
     * @param template_name Template name
     * @param session_name New session name
     * @return Session ID if creation successful, 0 otherwise
     */
    uint64_t createSessionFromTemplate(const std::string& template_name, const std::string& session_name);

    /**
     * Validate session configuration
     * @param session_info Session configuration
     * @return True if configuration is valid
     */
    bool validateSessionConfiguration(const SessionInfo& session_info) const;

    /**
     * Get system health status
     * @return System health status
     */
    SystemHealthStatus getSystemHealthStatus() const;

private:
    struct SystemResources {
        int max_input_channels = 128;              ///< Maximum input channels
        int max_output_channels = 128;             ///< Maximum output channels
        int max_sample_rate = 192000;              ///< Maximum sample rate
        int min_sample_rate = 8000;                ///< Minimum sample rate
        int max_buffer_size = 8192;                ///< Maximum buffer size
        int min_buffer_size = 16;                  ///< Minimum buffer size
        double max_latency_ms = 100.0;             ///< Maximum acceptable latency
        int max_dsp_cores = 16;                    ///< Maximum DSP cores
        size_t max_memory_mb = 8192;               ///< Maximum memory in MB
        int max_concurrent_sessions = 64;          ///< Maximum concurrent sessions
        std::vector<AudioBitDepth> supported_bit_depths; ///< Supported bit depths
    };

    struct SessionInternal {
        SessionInfo info;
        std::unique_ptr<MultiChannelEngine> engine;
        std::unique_ptr<AudioRouter> router;
        std::unique_ptr<AudioDeviceManager> device_manager;
        std::chrono::steady_clock::time_point last_activity;
        std::thread monitoring_thread;
        std::atomic<bool> monitoring_active{false};
        std::condition_variable activity_cv;
        std::mutex activity_mutex;
        uint64_t process_count = 0;
        uint64_t error_count = 0;
        std::queue<std::string> error_messages;
    };

    // Core state
    bool initialized_ = false;
    mutable std::mutex sessions_mutex_;
    std::unordered_map<uint64_t, std::unique_ptr<SessionInternal>> sessions_;
    std::atomic<uint64_t> next_session_id_{1};
    int max_sessions_ = 64;
    SystemResources system_resources_;

    // Resource management
    mutable std::mutex resources_mutex_;
    std::unordered_set<uint32_t> allocated_input_devices_;
    std::unordered_set<uint32_t> allocated_output_devices_;
    std::unordered_set<int> allocated_sample_rates_;
    std::unordered_set<int> allocated_buffer_sizes_;
    int allocated_dsp_cores_ = 0;
    size_t allocated_memory_mb_ = 0;

    // Scheduling and priority management
    std::priority_queue<std::pair<SessionPriority, uint64_t>> priority_queue_;
    mutable std::mutex scheduling_mutex_;

    // Callbacks
    SessionStateCallback state_callback_;
    SessionMetricsCallback metrics_callback_;
    SessionErrorCallback error_callback_;
    ResourceChangeCallback resource_callback_;

    // Monitoring and metrics
    std::thread metrics_thread_;
    std::atomic<bool> metrics_active_{false};
    std::chrono::steady_clock::time_point last_metrics_update_;
    std::vector<SessionMetrics> historical_metrics_;

    // Background tasks
    std::thread cleanup_thread_;
    std::atomic<bool> cleanup_active_{false};
    std::condition_variable cleanup_cv_;
    std::mutex cleanup_mutex_;

    // Session templates
    std::unordered_map<std::string, SessionInfo> session_templates_;
    mutable std::mutex templates_mutex_;

    // Internal methods
    bool allocateResources(uint64_t session_id, const SessionResourceRequest& request);
    void deallocateResources(uint64_t session_id);
    bool canAllocateResources(const SessionResourceRequest& request) const;
    void updateSessionPriority(uint64_t session_id, SessionPriority priority);
    void scheduleSessions();
    void cleanupInactiveSessions();
    void updateAllSessionMetrics();
    void sessionMonitoringLoop(uint64_t session_id);
    void metricsUpdateLoop();
    void cleanupLoop();
    void notifySessionStateChange(uint64_t session_id, SessionState old_state, SessionState new_state);
    void notifySessionError(uint64_t session_id, const std::string& error);
    void notifyResourceChange(uint64_t session_id, const SessionAllocation& allocation);
    SessionMetrics calculateSessionMetrics(uint64_t session_id) const;
    bool validateResourceRequest(const SessionResourceRequest& request) const;
    SessionAllocation allocateSessionResources(const SessionResourceRequest& request);
    void deallocateSessionResources(const SessionAllocation& allocation);
    std::vector<uint64_t> getSessionsToPreempt(SessionPriority min_priority, const SessionResourceRequest& required);
    bool preemptSessionsInternal(const std::vector<uint64_t>& session_ids);
    uint64_t generateSessionId() const;
    void initializeSessionEngine(SessionInternal& session);
    void initializeSessionRouter(SessionInternal& session);
    void initializeSessionDeviceManager(SessionInternal& session);
    void startSessionMonitoring(SessionInternal& session);
    void stopSessionMonitoring(SessionInternal& session);
    bool isSessionActive(const SessionInfo& session) const;
    void updateSessionActivity(uint64_t session_id);
    SystemMetrics calculateSystemMetrics() const;
    SystemHealthStatus calculateSystemHealth() const;
    SessionStatistics calculateSessionStatistics(uint64_t session_id) const;
    bool loadSessionTemplates();
    bool saveSessionTemplates() const;
};

/**
 * Session Manager Factory
 * Creates pre-configured session managers for different use cases
 */
class SessionManagerFactory {
public:
    /**
     * Create session manager for professional audio workstation
     * @param max_sessions Maximum concurrent sessions
     * @return Configured session manager
     */
    static std::unique_ptr<SessionManager> createProfessionalWorkstationManager(int max_sessions = 32);

    /**
     * Create session manager for streaming service
     * @param max_sessions Maximum concurrent sessions
     * @return Configured session manager
     */
    static std::unique_ptr<SessionManager> createStreamingServiceManager(int max_sessions = 1000);

    /**
     * Create session manager for live performance
     * @param max_sessions Maximum concurrent sessions
     * @return Configured session manager
     */
    static std::unique_ptr<SessionManager> createLivePerformanceManager(int max_sessions = 16);

    /**
     * Create session manager for gaming
     * @param max_sessions Maximum concurrent sessions
     * @return Configured session manager
     */
    static std::unique_ptr<SessionManager> createGamingSessionManager(int max_sessions = 8);

    /**
     * Create session manager for broadcast
     * @param max_sessions Maximum concurrent sessions
     * @return Configured session manager
     */
    static std::unique_ptr<SessionManager> createBroadcastManager(int max_sessions = 24);

    /**
     * Create session manager with custom configuration
     * @param constraints Resource constraints
     * @param max_sessions Maximum concurrent sessions
     * @return Configured session manager
     */
    static std::unique_ptr<SessionManager> createCustomManager(const SystemResources& constraints,
                                                            int max_sessions = 64);

private:
    static SystemResources createProfessionalWorkstationConstraints();
    static SystemResources createStreamingServiceConstraints();
    static SystemResources createLivePerformanceConstraints();
    static SystemResources createGamingConstraints();
    static SystemResources createBroadcastConstraints();
};

// Utility functions
namespace session_utils {

    // Session type utilities
    std::string sessionTypeToString(SessionType type);
    std::string sessionPriorityToString(SessionPriority priority);
    std::string sessionStateToString(SessionState state);
    std::string sessionIsolationToString(SessionIsolation isolation);
    std::string sessionSchedulingToString(SessionScheduling scheduling);
    std::string resourceAllocationToString(ResourceAllocation allocation);

    SessionType stringToSessionType(const std::string& str);
    SessionPriority stringToSessionPriority(const std::string& str);
    SessionState stringToSessionState(const std::string& str);

    // Session utilities
    uint64_t generateSessionId();
    bool isSessionStateTransitionValid(SessionState from, SessionState to);
    SessionState getHighestPriorityState(const std::vector<SessionState>& states);
    SessionPriority getHighestPriority(const std::vector<SessionPriority>& priorities);

    // Resource utilities
    bool areResourcesCompatible(const SessionResourceRequest& req1, const SessionResourceRequest& req2);
    SessionResourceRequest mergeResourceRequests(const std::vector<SessionResourceRequest>& requests);
    SessionResourceRequest minimizeResourceRequest(const SessionResourceRequest& request);
    double calculateResourceUtilization(const SessionResourceRequest& used, const SessionResourceRequest& available);

    // Performance utilities
    double calculateSessionEfficiency(const SessionMetrics& metrics);
    double calculateSessionStability(const SessionMetrics& metrics);
    double calculateSessionQuality(const SessionMetrics& metrics);
    bool isSessionPerformingWell(const SessionMetrics& metrics);
    std::vector<std::string> detectSessionIssues(const SessionMetrics& metrics);

    // Configuration utilities
    std::string sessionInfoToJSON(const SessionInfo& session);
    std::string sessionMetricsToJSON(const SessionMetrics& metrics);
    std::string resourceRequestToJSON(const SessionResourceRequest& request);
    SessionInfo sessionInfoFromJSON(const std::string& json);
    SessionMetrics sessionMetricsFromJSON(const std::string& json);
    SessionResourceRequest resourceRequestFromJSON(const std::string& json);

    // Time utilities
    std::string formatDuration(uint64_t seconds);
    std::chrono::steady_clock::time_point parseTimestamp(const std::string& timestamp);
    std::string formatTimestamp(const std::chrono::steady_clock::time_point& timestamp);

    // Validation utilities
    bool isValidSessionName(const std::string& name);
    bool isValidProcessId(const std::string& process_id);
    bool isValidUserId(const std::string& user_id);
    bool isValidSessionId(uint64_t session_id);

    // Template utilities
    std::vector<std::string> getDefaultSessionTemplates();
    SessionInfo createDefaultTemplate(SessionType type);
    bool validateTemplate(const SessionInfo& template_info);
}

} // namespace audio
} // namespace core
} // namespace vortex