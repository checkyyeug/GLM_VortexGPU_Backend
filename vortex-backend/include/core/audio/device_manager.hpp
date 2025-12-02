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
#include "core/audio/multi_channel_engine.hpp"

namespace vortex {
namespace core {
namespace audio {

/**
 * Audio Device Manager
 * Handles discovery, configuration, and management of audio devices
 * Supports multi-platform audio drivers and device hot-plugging
 */

enum class AudioDriverType {
    AUTO,               ///< Automatic driver selection
    WASAPI,             ///< Windows Audio Session API (Windows 10+)
    ASIO,               ///< Audio Stream Input/Output (Windows)
    DIRECTSOUND,        ///< DirectSound (Windows legacy)
    CORE_AUDIO,         ///< Core Audio (macOS/iOS)
    JACK,               ///< Jack Audio Connection Kit (Linux/Windows/macOS)
    ALSA,               ///< Advanced Linux Sound Architecture (Linux)
    PULSE,              ///< PulseAudio (Linux)
    OSS,                ///< Open Sound System (Unix/Linux)
    VIRTUAL,            ///< Virtual audio driver
    NETWORK             ///< Network audio driver
};

enum class DeviceState {
    UNKNOWN,            ///< Device state unknown
    ACTIVE,             ///< Device active and running
    IDLE,               ///< Device idle but available
    DISCONNECTED,       ///< Device disconnected
    ERROR,              ///< Device in error state
    CONFIGURING,        ///< Device being configured
    STARTING,           ///< Device starting up
    STOPPING,           ///< Device stopping
    SUSPENDED           ///< Device suspended (power save)
};

enum class DeviceLatencyMode {
    LOWEST,             ///< Lowest possible latency
    LOW,                ///< Low latency (default)
    MEDIUM,             ///< Medium latency
    HIGH,               ///< High latency
    HIGHEST,            ///< Highest latency
    CUSTOM              ///< Custom latency settings
};

enum class DeviceSampleType {
    INT8,               ///< 8-bit integer
    UINT8,              ///< 8-bit unsigned integer
    INT16,              ///< 16-bit integer
    UINT16,             ///< 16-bit unsigned integer
    INT24,              ///< 24-bit integer
    INT32,              ///< 32-bit integer
    UINT32,             ///< 32-bit unsigned integer
    FLOAT32,            ///< 32-bit float
    FLOAT64,            ///< 64-bit float
    DSD8,               ///< DSD 8-bit (1-bit)
    DSD16,              ///< DSD 16-bit
    DSD32,              ///< DSD 32-bit
    CUSTOM              ///< Custom sample type
};

enum class DeviceClockSource {
    INTERNAL,           ///< Internal clock
    EXTERNAL_WORD,      ///< External word clock
    EXTERNAL_SPDIF,     ///< External S/PDIF clock
    EXTERNAL_ADAT,      ///< External ADAT clock
    EXTERNAL_AES_EBU,   ///< External AES/EBU clock
    EXTERNAL_MIDI,      ///< External MIDI clock
    NETWORK,            ///< Network clock sync
    AUTOMATIC           ///< Automatic selection
};

struct DeviceCapabilities {
    std::vector<int> sample_rates;              ///< Supported sample rates
    std::vector<AudioBitDepth> bit_depths;       ///< Supported bit depths
    std::vector<DeviceSampleType> sample_types; ///< Supported sample types
    std::vector<int> buffer_sizes;              ///< Supported buffer sizes
    std::vector<AudioChannelLayout> layouts;   ///< Supported channel layouts
    uint32_t max_input_channels = 0;           ///< Maximum input channels
    uint32_t max_output_channels = 0;          ///< Maximum output channels
    uint32_t max_total_channels = 0;           ///< Maximum total channels
    double min_latency_ms = 0.0;               ///< Minimum achievable latency
    double max_latency_ms = 0.0;               ///< Maximum allowed latency
    bool supports_exclusive_mode = false;      ///< Exclusive mode support
    bool supports_shared_mode = true;          ///< Shared mode support
    bool supports_event_driven = true;         ///< Event-driven callback support
    bool supports_polling = false;             ///< Polling mode support
    bool supports_midi = false;                ///< MIDI support
    bool supports_digital_io = false;          ///< Digital I/O support
    bool supports_analog_io = true;            ///< Analog I/O support
    bool supports_low_latency = false;         ///< Low-latency support
    bool supports_synchronization = false;     ///< Device synchronization
    bool supports_hot_plug = false;            ///< Hot-plug detection
    std::vector<DeviceClockSource> clock_sources; ///< Supported clock sources
    std::array<double, 3> dynamic_range_db{0.0, 0.0, 0.0}; ///< Min, max, typical dynamic range
    double thd_percent = 0.0;                  ///< Total harmonic distortion
    double snr_db = 0.0;                       ///< Signal-to-noise ratio
};

struct DeviceConfiguration {
    int sample_rate = 48000;                    ///< Sample rate
    AudioBitDepth bit_depth = AudioBitDepth::FLOAT32; ///< Bit depth
    DeviceSampleType sample_type = DeviceSampleType::FLOAT32; ///< Sample type
    int buffer_size = 512;                      ///< Buffer size
    AudioChannelLayout layout = AudioChannelLayout::STEREO; ///< Channel layout
    int input_channels = 2;                     ///< Number of input channels
    int output_channels = 2;                    ///< Number of output channels
    DeviceLatencyMode latency_mode = DeviceLatencyMode::LOW; ///< Latency mode
    DeviceClockSource clock_source = DeviceClockSource::INTERNAL; ///< Clock source
    bool exclusive_mode = false;                ///< Exclusive access mode
    bool event_driven = true;                   ///< Event-driven callback
    double target_latency_ms = 10.0;           ///< Target latency
    double maximum_latency_ms = 50.0;          ///< Maximum acceptable latency
    int priority = 0;                          ///< Thread priority (-10 to 10)
    bool enable_monitoring = false;             ///< Input monitoring
    float monitoring_level = 1.0f;             ///< Monitoring level
    std::string device_name;                    ///< Device name override
    std::string driver_name;                    ///< Driver name override
    std::vector<uint8_t> driver_specific_data; ///< Driver-specific configuration
};

struct DeviceMonitorInfo {
    float input_peak_dbfs = -INFINITY;          ///< Input peak level (dBFS)
    float output_peak_dbfs = -INFINITY;         ///< Output peak level (dBFS)
    float input_rms_dbfs = -INFINITY;           ///< Input RMS level (dBFS)
    float output_rms_dbfs = -INFINITY;          ///< Output RMS level (dBFS)
    float input_dc_offset = 0.0f;               ///< Input DC offset
    float output_dc_offset = 0.0f;              ///< Output DC offset
    uint32_t clipping_samples = 0;              ///< Number of clipping samples
    uint32_t dropouts = 0;                      ///< Number of audio dropouts
    uint64_t total_samples_processed = 0;       ///< Total samples processed
    double cpu_usage_percent = 0.0;             ///< Device CPU usage
    double buffer_utilization = 0.0;            ///< Buffer utilization
    DeviceState state = DeviceState::UNKNOWN;   ///< Current device state
    std::chrono::steady_clock::time_point last_update;
    std::array<float, 32> per_channel_levels{-INFINITY}; ///< Per-channel levels
    std::vector<std::string> error_messages;    ///< Recent error messages
};

struct DeviceInfo {
    uint32_t id = 0;                            ///< Unique device ID
    std::string name;                           ///< Device name
    std::string driver_name;                    ///< Driver name
    std::string manufacturer;                   ///< Manufacturer
    std::string version;                        ///< Device version
    std::string serial_number;                  ///< Serial number
    AudioDriverType driver_type = AudioDriverType::AUTO;
    AudioDeviceType device_type = AudioDeviceType::OUTPUT;
    DeviceState state = DeviceState::UNKNOWN;
    DeviceCapabilities capabilities;
    DeviceConfiguration current_config;
    DeviceMonitorInfo monitor_info;
    bool is_default_input = false;              ///< Default input device
    bool is_default_output = false;             ///< Default output device
    bool is_enabled = true;                     ///< Device enabled
    std::chrono::steady_clock::time_point last_seen;
    uint64_t total_uptime_seconds = 0;          ///< Total uptime
    std::array<uint8_t, 16> device_guid{0};     ///< Device GUID
    std::vector<std::string> supported_formats; ///< Supported audio formats
    std::string hardware_id;                    ///< Hardware ID
    std::string location;                       ///< Physical location
};

using DeviceChangeCallback = std::function<void(const DeviceInfo& device, bool added)>;
using DeviceStateCallback = std::function<void(uint32_t device_id, DeviceState old_state, DeviceState new_state)>;
using DeviceMonitorCallback = std::function<void(uint32_t device_id, const DeviceMonitorInfo& monitor_info)>;

/**
 * Audio Device Manager
 * Handles all aspects of audio device management
 */
class DeviceManager {
public:
    DeviceManager();
    ~DeviceManager();

    /**
     * Initialize device manager
     * @param driver_type Preferred driver type (AUTO for automatic)
     * @return True if initialization successful
     */
    bool initialize(AudioDriverType driver_type = AudioDriverType::AUTO);

    /**
     * Shutdown device manager and cleanup
     */
    void shutdown();

    /**
     * Scan for available audio devices
     * @return List of available devices
     */
    std::vector<DeviceInfo> scanDevices();

    /**
     * Get all available devices
     * @return List of all devices
     */
    std::vector<DeviceInfo> getAvailableDevices() const;

    /**
     * Get device by ID
     * @param device_id Device ID
     * @return Device info if found
     */
    std::optional<DeviceInfo> getDevice(uint32_t device_id) const;

    /**
     * Get default input device
     * @return Default input device info
     */
    std::optional<DeviceInfo> getDefaultInputDevice() const;

    /**
     * Get default output device
     * @return Default output device info
     */
    std::optional<DeviceInfo> getDefaultOutputDevice() const;

    /**
     * Set default input device
     * @param device_id Device ID
     * @return True if successful
     */
    bool setDefaultInputDevice(uint32_t device_id);

    /**
     * Set default output device
     * @param device_id Device ID
     * @return True if successful
     */
    bool setDefaultOutputDevice(uint32_t device_id);

    /**
     * Enable/disable device
     * @param device_id Device ID
     * @param enabled Enable state
     * @return True if successful
     */
    bool setDeviceEnabled(uint32_t device_id, bool enabled);

    /**
     * Configure device
     * @param device_id Device ID
     * @param config Device configuration
     * @return True if configuration successful
     */
    bool configureDevice(uint32_t device_id, const DeviceConfiguration& config);

    /**
     * Test device with specific configuration
     * @param device_id Device ID
     * @param config Test configuration
     * @param test_duration_ms Test duration in milliseconds
     * @return True if test passed
     */
    bool testDevice(uint32_t device_id, const DeviceConfiguration& config, int test_duration_ms = 1000);

    /**
     * Start device monitoring
     * @param device_id Device ID
     * @param callback Monitor callback function
     * @return True if monitoring started
     */
    bool startDeviceMonitoring(uint32_t device_id, DeviceMonitorCallback callback);

    /**
     * Stop device monitoring
     * @param device_id Device ID
     * @return True if monitoring stopped
     */
    bool stopDeviceMonitoring(uint32_t device_id);

    /**
     * Get device monitoring info
     * @param device_id Device ID
     * @return Monitor info if available
     */
    std::optional<DeviceMonitorInfo> getDeviceMonitorInfo(uint32_t device_id) const;

    /**
     * Get available sample rates for device
     * @param device_id Device ID
     * @return List of supported sample rates
     */
    std::vector<int> getDeviceSampleRates(uint32_t device_id) const;

    /**
     * Get available bit depths for device
     * @param device_id Device ID
     * @return List of supported bit depths
     */
    std::vector<AudioBitDepth> getDeviceBitDepths(uint32_t device_id) const;

    /**
     * Get optimal configuration for device
     * @param device_id Device ID
     * @param target_latency_ms Target latency
     * @return Optimal configuration
     */
    DeviceConfiguration getOptimalConfiguration(uint32_t device_id, double target_latency_ms = 10.0) const;

    /**
     * Synchronize devices
     * @param master_device_id Master device ID
     * @param slave_device_ids Slave device IDs
     * @return True if synchronization successful
     */
    bool synchronizeDevices(uint32_t master_device_id, const std::vector<uint32_t>& slave_device_ids);

    /**
     * Desynchronize devices
     * @param device_ids Device IDs to desynchronize
     * @return True if desynchronization successful
     */
    bool desynchronizeDevices(const std::vector<uint32_t>& device_ids);

    /**
     * Enable/disable hot-plug detection
     * @param enabled Enable hot-plug detection
     * @return True if configuration successful
     */
    bool setHotPlugDetectionEnabled(bool enabled);

    /**
     * Is hot-plug detection enabled
     * @return True if hot-plug detection is enabled
     */
    bool isHotPlugDetectionEnabled() const;

    /**
     * Start device discovery (background thread)
     * @param callback Device change callback
     * @return True if discovery started
     */
    bool startDeviceDiscovery(DeviceChangeCallback callback);

    /**
     * Stop device discovery
     */
    void stopDeviceDiscovery();

    /**
     * Is device discovery active
     * @return True if discovery is active
     */
    bool isDeviceDiscoveryActive() const;

    /**
     * Refresh device list
     * @return Updated device list
     */
    std::vector<DeviceInfo> refreshDevices();

    /**
     * Get device by name
     * @param name Device name (partial match allowed)
     * @return Matching devices
     */
    std::vector<DeviceInfo> findDevicesByName(const std::string& name) const;

    /**
     * Get devices by type
     * @param type Device type
     * @return Matching devices
     */
    std::vector<DeviceInfo> getDevicesByType(AudioDeviceType type) const;

    /**
     * Get devices by driver
     * @param driver Driver type
     * @return Matching devices
     */
    std::vector<DeviceInfo> getDevicesByDriver(AudioDriverType driver) const;

    /**
     * Register device state callback
     * @param callback State change callback
     */
    void setDeviceStateCallback(DeviceStateCallback callback);

    /**
     * Get supported drivers for platform
     * @return List of supported drivers
     */
    std::vector<AudioDriverType> getSupportedDrivers() const;

    /**
     * Set active driver
     * @param driver Driver type
     * @return True if driver set successfully
     */
    bool setActiveDriver(AudioDriverType driver);

    /**
     * Get active driver
     * @return Active driver type
     */
    AudioDriverType getActiveDriver() const;

    /**
     * Validate device configuration
     * @param device_id Device ID
     * @param config Configuration to validate
     * @return True if configuration is valid
     */
    bool validateConfiguration(uint32_t device_id, const DeviceConfiguration& config) const;

    /**
     * Reset device to default configuration
     * @param device_id Device ID
     * @return True if reset successful
     */
    bool resetDevice(uint32_t device_id);

    /**
     * Get device information as JSON
     * @param device_id Device ID
     * @return JSON string with device info
     */
    std::string getDeviceInfoJSON(uint32_t device_id) const;

    /**
     * Import device configuration from file
     * @param file_path Configuration file path
     * @return True if import successful
     */
    bool importConfiguration(const std::string& file_path);

    /**
     * Export device configuration to file
     * @param file_path Configuration file path
     * @return True if export successful
     */
    bool exportConfiguration(const std::string& file_path) const;

    /**
     * Create virtual device
     * @param name Device name
     * @param config Device configuration
     * @return Virtual device ID
     */
    uint32_t createVirtualDevice(const std::string& name, const DeviceConfiguration& config);

    /**
     * Destroy virtual device
     * @param device_id Device ID
     * @return True if destruction successful
     */
    bool destroyVirtualDevice(uint32_t device_id);

private:
    struct DeviceStateInternal {
        DeviceInfo info;
        DeviceMonitorInfo monitor;
        bool monitoring_active = false;
        DeviceMonitorCallback monitor_callback;
        std::chrono::steady_clock::time_point last_monitor_update;
        std::thread monitor_thread;
        std::atomic<bool> monitor_running{false};
        uint32_t driver_handle = 0; // Driver-specific handle
        void* driver_context = nullptr; // Driver-specific context
    };

    // Core state
    bool initialized_ = false;
    AudioDriverType active_driver_ = AudioDriverType::AUTO;
    std::atomic<uint32_t> next_device_id_{1};
    mutable std::mutex mutex_;

    // Device storage
    std::unordered_map<uint32_t, std::unique_ptr<DeviceStateInternal>> devices_;
    uint32_t default_input_device_id_ = 0;
    uint32_t default_output_device_id_ = 0;

    // Hot-plug detection
    bool hot_plug_detection_enabled_ = true;
    bool device_discovery_active_ = false;
    std::thread discovery_thread_;
    std::atomic<bool> discovery_running_{false};
    DeviceChangeCallback device_change_callback_;

    // Callbacks
    DeviceStateCallback device_state_callback_;

    // Driver-specific implementations
    bool initializeWASAPI();
    bool initializeASIO();
    bool initializeCoreAudio();
    bool initializeJACK();
    bool initializeALSA();
    bool initializePulse();
    bool initializeOSS();

    void shutdownDrivers();
    void scanWASAPIDevices(std::vector<DeviceInfo>& devices);
    void scanASIODevice(std::vector<DeviceInfo>& devices);
    void scanCoreAudioDevices(std::vector<DeviceInfo>& devices);
    void scanJACKDevices(std::vector<DeviceInfo>& devices);
    void scanALSADevices(std::vector<DeviceInfo>& devices);
    void scanPulseDevices(std::vector<DeviceInfo>& devices);
    void scanOSSDevices(std::vector<DeviceInfo>& devices);

    // Device management
    bool openDevice(uint32_t device_id);
    bool closeDevice(uint32_t device_id);
    bool configureDeviceDriver(uint32_t device_id, const DeviceConfiguration& config);
    void updateDeviceState(uint32_t device_id, DeviceState new_state);

    // Monitoring
    void startDeviceMonitorThread(uint32_t device_id);
    void stopDeviceMonitorThread(uint32_t device_id);
    void deviceMonitorLoop(uint32_t device_id);
    void updateDeviceMonitorInfo(uint32_t device_id);

    // Discovery
    void deviceDiscoveryLoop();
    void detectDeviceChanges(const std::vector<DeviceInfo>& current_devices);

    // Utility functions
    DeviceCapabilities getDefaultCapabilities(AudioDeviceType type, AudioDriverType driver) const;
    std::vector<AudioDriverType> getPlatformSupportedDrivers() const;
    std::string generateDeviceGUID(const DeviceInfo& device) const;
    bool compareDeviceGUID(const std::array<uint8_t, 16>& guid1, const std::array<uint8_t, 16>& guid2) const;
    DeviceState calculateDeviceState(const DeviceInfo& device) const;
    double calculateOptimalLatency(const DeviceCapabilities& caps, double target_latency) const;
    bool isConfigurationSupported(const DeviceCapabilities& caps, const DeviceConfiguration& config) const;

    // Driver-specific helper functions
    uint32_t openWASAPIDevice(const DeviceInfo& device);
    bool configureWASAPIDevice(uint32_t driver_handle, const DeviceConfiguration& config);
    void closeWASAPIDevice(uint32_t driver_handle);
    DeviceMonitorInfo getWASAPIMonitorInfo(uint32_t driver_handle);

    uint32_t openASIODevice(const DeviceInfo& device);
    bool configureASIODevice(uint32_t driver_handle, const DeviceConfiguration& config);
    void closeASIODevice(uint32_t driver_handle);
    DeviceMonitorInfo getASIOMonitorInfo(uint32_t driver_handle);

    // Similar methods would exist for other drivers...
};

/**
 * Audio Device Factory
 * Creates and configures device managers for different platforms
 */
class DeviceManagerFactory {
public:
    /**
     * Create device manager for current platform
     * @return Device manager instance
     */
    static std::unique_ptr<DeviceManager> createPlatformDeviceManager();

    /**
     * Create device manager with specific driver
     * @param driver Driver type
     * @return Device manager instance
     */
    static std::unique_ptr<DeviceManager> createDeviceManager(AudioDriverType driver);

    /**
     * Get recommended driver for platform
     * @return Recommended driver type
     */
    static AudioDriverType getRecommendedDriver();

    /**
     * Get all available drivers for platform
     * @return List of available drivers
     */
    static std::vector<AudioDriverType> getAvailableDrivers();

    /**
     * Check if driver is available on platform
     * @param driver Driver type
     * @return True if driver is available
     */
    static bool isDriverAvailable(AudioDriverType driver);

private:
    static std::vector<AudioDriverType> detectAvailableDrivers();
    static AudioDriverType getBestAvailableDriver();
};

// Utility functions
namespace device_utils {

    // Format conversions
    std::string driverTypeToString(AudioDriverType driver);
    std::string deviceStateToString(DeviceState state);
    std::string latencyModeToString(DeviceLatencyMode mode);
    std::string sampleTypeToString(DeviceSampleType type);
    std::string clockSourceToString(DeviceClockSource source);

    AudioDriverType stringToDriverType(const std::string& str);
    DeviceState stringToDeviceState(const std::string& str);
    DeviceLatencyMode stringToLatencyMode(const std::string& str);

    // Device compatibility
    bool isCompatibleConfiguration(const DeviceCapabilities& caps, const DeviceConfiguration& config);
    DeviceConfiguration findBestMatch(const DeviceCapabilities& caps, const std::vector<DeviceConfiguration>& options);
    std::vector<int> getCommonSampleRates(const std::vector<int>& rates1, const std::vector<int>& rates2);

    // Audio calculations
    double calculateBufferSizeMs(int buffer_size, int sample_rate);
    double calculateSampleRatePeriodMs(int sample_rate);
    double calculateBitsPerSecond(AudioBitDepth bit_depth, int sample_rate, int channels);
    double calculateBytesPerSecond(const DeviceConfiguration& config);
    int calculateOptimalBufferSize(int sample_rate, double target_latency_ms);

    // Device testing
    bool generateTestTone(float* buffer, size_t samples, int sample_rate, float frequency, float amplitude);
    bool analyzeTestSignal(const float* buffer, size_t samples, int sample_rate, float& thd, float& snr);
    float calculateSignalToNoiseRatio(const float* signal, const float* noise, size_t samples);
    float calculateTotalHarmonicDistortion(const float* signal, size_t samples, int sample_rate, int num_harmonics = 10);

    // JSON utilities
    std::string deviceInfoToJSON(const DeviceInfo& device);
    std::string deviceConfigToJSON(const DeviceConfiguration& config);
    std::string deviceMonitorToJSON(const DeviceMonitorInfo& monitor);
    DeviceInfo deviceInfoFromJSON(const std::string& json);
    DeviceConfiguration deviceConfigFromJSON(const std::string& json);

    // Platform detection
    std::string getPlatformName();
    std::string getSystemAudioAPI();
    bool isWindows();
    bool isMacOS();
    bool isLinux();
    bool isUnix();

}

} // namespace audio
} // namespace core
} // namespace vortex