#pragma once

#include "audio_types.hpp"
#include "network_types.hpp"

#include <string>
#include <memory>
#include <mutex>
#include <atomic>
#include <functional>

namespace vortex {

/**
 * @brief Configuration manager for system settings and parameters
 *
 * This class handles loading, saving, and managing system configuration
 * with support for hot-reloading and validation.
 */
class ConfigManager {
public:
    ConfigManager();
    ~ConfigManager();

    // Initialization
    bool initialize(const std::string& configPath = "config/default.json");
    void shutdown();
    bool isInitialized() const;

    // Configuration loading
    bool loadFromFile(const std::string& filePath);
    bool loadFromString(const std::string& json);
    bool loadFromEnvironment();
    bool reload();

    // Configuration saving
    bool saveToFile(const std::string& filePath) const;
    std::string saveToString() const;

    // Configuration access
    SystemConfiguration getConfiguration() const;
    bool updateConfiguration(const SystemConfiguration& config);

    // Individual component access
    AudioSettings getAudioSettings() const;
    bool setAudioSettings(const AudioSettings& settings);

    NetworkSettings getNetworkSettings() const;
    bool setNetworkSettings(const NetworkSettings& settings);

    GPUPreferences getGPUPreferences() const;
    bool setGPUPreferences(const GPUPreferences& preferences);

    MonitoringSettings getMonitoringSettings() const;
    bool setMonitoringSettings(const MonitoringSettings& settings);

    UserPreferences getUserPreferences() const;
    bool setUserPreferences(const UserPreferences& preferences);

    // Configuration validation
    bool validateConfiguration(const SystemConfiguration& config) const;
    bool validateAudioSettings(const AudioSettings& settings) const;
    bool validateNetworkSettings(const NetworkSettings& settings) const;
    bool validateGPUPreferences(const GPUPreferences& preferences) const;

    // Hot-reloading support
    bool enableHotReload(bool enable);
    bool isHotReloadEnabled() const;
    void setReloadCallback(std::function<void(const SystemConfiguration&)> callback);

    // Configuration merging
    bool mergeFromFile(const std::string& filePath);
    bool mergeFromString(const std::string& json);
    bool mergeConfiguration(const SystemConfiguration& overlay);

    // Default configuration generation
    static SystemConfiguration createDefaultConfiguration();
    static SystemConfiguration createHighPerformanceConfiguration();
    static SystemConfiguration createLowLatencyConfiguration();
    static SystemConfiguration createDevelopmentConfiguration();

    // Configuration templates
    enum class Profile {
        DEFAULT,
        HIGH_PERFORMANCE,
        LOW_LATENCY,
        DEVELOPMENT,
        PRODUCTION,
        TESTING
    };

    bool loadProfile(Profile profile);
    bool saveProfile(Profile profile, const std::string& filePath) const;

    // Configuration diff and patch
    struct ConfigDiff {
        std::vector<std::string> added;
        std::vector<std::string> modified;
        std::vector<std::string> removed;
    };

    ConfigDiff diff(const SystemConfiguration& other) const;
    bool applyPatch(const std::string& patchJson);

    // Configuration backup and restore
    bool createBackup(const std::string& backupPath) const;
    bool restoreFromBackup(const std::string& backupPath);
    std::vector<std::string> listBackups() const;

    // Configuration locking (for production environments)
    bool lockConfiguration(const std::string& password);
    bool unlockConfiguration(const std::string& password);
    bool isConfigurationLocked() const;

    // Environment-specific configuration
    bool setEnvironment(const std::string& environment);
    std::string getEnvironment() const;

    // Configuration statistics
    struct ConfigStats {
        uint64_t loadCount = 0;
        uint64_t saveCount = 0;
        uint64_t reloadCount = 0;
        uint64_t validationFailures = 0;
        std::chrono::steady_clock::time_point lastLoad;
        std::chrono::steady_clock::time_point lastSave;
        std::chrono::steady_clock::time_point lastValidation;
    };

    ConfigStats getStatistics() const;
    void resetStatistics();

    // Configuration schema validation
    bool validateAgainstSchema(const std::string& schemaPath) const;
    std::string getConfigurationSchema() const;

protected:
    // JSON parsing helpers
    bool parseConfigurationJson(const std::string& json, SystemConfiguration& config) const;
    std::string serializeConfigurationJson(const SystemConfiguration& config) const;

    // File watching for hot-reload
    void startFileWatcher();
    void stopFileWatcher();
    void onFileChanged();

    // Configuration migration
    bool migrateConfiguration(const std::string& fromVersion, const std::string& toVersion);

private:
    // Internal state
    mutable std::mutex m_mutex;
    SystemConfiguration m_configuration;
    std::string m_configFilePath;
    std::string m_environment = "production";

    // Initialization state
    std::atomic<bool> m_initialized{false};

    // Hot-reloading
    std::atomic<bool> m_hotReloadEnabled{false};
    std::function<void(const SystemConfiguration&)> m_reloadCallback;
    std::unique_ptr<std::thread> m_fileWatcherThread;
    std::atomic<bool> m_fileWatcherRunning{false};

    // Configuration locking
    std::atomic<bool> m_locked{false};
    std::string m_lockPassword;

    // Statistics
    mutable ConfigStats m_stats;

    // Default values
    static constexpr uint32_t DEFAULT_SAMPLE_RATE = 44100;
    static constexpr uint16_t DEFAULT_BIT_DEPTH = 24;
    static constexpr uint16_t DEFAULT_CHANNELS = 2;
    static constexpr uint32_t DEFAULT_BUFFER_SIZE = 4096;
    static constexpr float DEFAULT_MAX_LATENCY = 10.0f;
    static constexpr uint16_t DEFAULT_HTTP_PORT = 8080;
    static constexpr uint16_t DEFAULT_WEBSOCKET_PORT = 8081;

    // Validation helpers
    bool isValidSampleRate(uint32_t sampleRate) const;
    bool isValidBitDepth(uint16_t bitDepth) const;
    bool isValidPort(uint16_t port) const;
    bool isValidGPUBackend(GPUBackend backend) const;
};

/**
 * @brief Configuration validator and sanitizer
 */
class ConfigValidator {
public:
    struct ValidationResult {
        bool isValid = true;
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
        std::vector<std::string> suggestions;
    };

    static ValidationResult validateAudioSettings(const AudioSettings& settings);
    static ValidationResult validateNetworkSettings(const NetworkSettings& settings);
    static ValidationResult validateGPUPreferences(const GPUPreferences& preferences);
    static ValidationResult validateMonitoringSettings(const MonitoringSettings& settings);
    static ValidationResult validateSystemConfiguration(const SystemConfiguration& config);

    static AudioSettings sanitizeAudioSettings(const AudioSettings& settings);
    static NetworkSettings sanitizeNetworkSettings(const NetworkSettings& settings);
    static GPUPreferences sanitizeGPUPreferences(const GPUPreferences& preferences);
    static SystemConfiguration sanitizeSystemConfiguration(const SystemConfiguration& config);
};

/**
 * @brief Configuration profile manager
 */
class ConfigProfileManager {
public:
    struct ProfileInfo {
        std::string name;
        std::string description;
        Profile type;
        std::string filePath;
        SystemConfiguration configuration;
        std::chrono::steady_clock::time_point created;
        std::chrono::steady_clock::time_point lastModified;
    };

    static bool saveProfile(const std::string& name, const SystemConfiguration& config,
                           const std::string& description = "");
    static bool loadProfile(const std::string& name, SystemConfiguration& config);
    static bool deleteProfile(const std::string& name);
    static std::vector<ProfileInfo> listProfiles();
    static bool exportProfile(const std::string& name, const std::string& exportPath);
    static bool importProfile(const std::string& importPath, const std::string& name);

private:
    static std::string getProfileDirectory();
    static std::string getProfilePath(const std::string& name);
};

} // namespace vortex