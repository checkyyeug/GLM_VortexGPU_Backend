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
#include "core/dsp/spatial_audio_processor.hpp"
#include "core/dsp/vr_audio_processor.hpp"
#include "core/dsp/spectrum_analyzer.hpp"
#include "core/dsp/waveform_processor.hpp"
#include "core/dsp/vu_meter_processor.hpp"
#include "core/dsp/realtime_effects_chain.hpp"

namespace vortex {
namespace core {
namespace audio {

/**
 * Multi-channel audio engine for professional audio applications
 * Supports high-channel-count audio processing with routing, mixing, and real-time effects
 */

enum class AudioChannelLayout {
    MONO,               ///< 1.0 - Mono
    STEREO,             ///< 2.0 - Stereo
    TWO_POINT_ONE,      ///< 2.1 - Stereo + LFE
    THREE_POINT_ONE,    ///< 3.1 - L,R,C + LFE
    FIVE_POINT_ONE,     ///< 5.1 - L,R,C,LFE,LS,RS
    SEVEN_POINT_ONE,    ///< 7.1 - L,R,C,LFE,LS,RS,LR,RR
    SEVEN_POINT_ONE_FOUR, ///< 7.1.4 - 7.1 + 4 height channels
    NINE_POINT_ONE_FOUR,  ///< 9.1.4 - 7.1 + front wide + 4 height
    DOLBY_ATMOS,       ///< Dolby Atmos setup
    DTS_X,            ///< DTS:X setup
    AURO_3D,          ///< Auro-3D setup
    CUSTOM            ///< Custom channel configuration
};

enum class AudioBitDepth {
    INT16,             ///< 16-bit integer
    INT24,             ///< 24-bit integer
    INT32,             ///< 32-bit integer
    FLOAT32,           ///< 32-bit float
    FLOAT64,           ///< 64-bit float
    DSD64,             ///< DSD64 (1-bit 2.8224 MHz)
    DSD128,            ///< DSD128 (1-bit 5.6448 MHz)
    DSD256,            ///< DSD256 (1-bit 11.2896 MHz)
    DSD512             ///< DSD512 (1-bit 22.5792 MHz)
};

enum class AudioDeviceType {
    INPUT,             ///< Input device (microphone, line-in)
    OUTPUT,            ///< Output device (speakers, headphones)
    INPUT_OUTPUT,      ///< Full duplex device
    VIRTUAL,           ///< Virtual device/software endpoint
    NETWORK            ///< Network audio device
};

enum class AudioSyncMode {
    NONE,              ///< No synchronization
    CLOCK_MASTER,      ///< Device is clock master
    CLOCK_SLAVE,       ///< Device is clock slave
    WORD_CLOCK,        ///< External word clock sync
    ADAT_SYNC,         ///< ADAT synchronization
    SPDIF_SYNC,        ///< S/PDIF synchronization
    MTC,               ///< MIDI Time Code
    LTC,               ///< Linear Time Code
    JAM_SYNC           ///< Jam sync (synchronize on start)
};

enum class AudioRoutingMode {
    DIRECT,            ///< Direct routing
    CROSSFADE,         ///< Crossfade routing
    MATRIX,            ///< Matrix mixing
    BUS_SUMMING,       ///< Bus summing
    AUX_SEND,          ///< Auxiliary sends
    INSERT,            ///< Insert effects
    SIDECHAIN,         ///< Sidechain routing
    METERING           ///< Metering only
};

struct AudioChannel {
    int index = 0;                              ///< Channel index
    std::string name;                           ///< Channel name
    std::string description;                    ///< Channel description
    float gain = 1.0f;                         ///< Channel gain (linear)
    float pan = 0.0f;                          ///< Pan position (-1.0 to 1.0)
    bool muted = false;                        ///< Mute state
    bool soloed = false;                       ///< Solo state
    float phase = 0.0f;                        ///< Phase offset (radians)
    bool phase_inverted = false;               ///< Phase invert
    std::string group;                         ///< Channel group
    std::vector<std::string> tags;             ///< Channel tags
    int32_t delay_samples = 0;                 ///< Delay in samples
    bool enable_limiter = false;               ///< Enable channel limiter
    float limiter_threshold = 0.0f;            ///< Limiter threshold (dB)
    bool record_enabled = false;               ///< Enable recording
    std::string record_file_path;              ///< Recording file path
    uint32_t output_routing = 0;              ///< Output routing mask
    std::vector<uint32_t> send_routing;       ///< Send routing destinations
    std::vector<float> send_levels;           ///< Send levels
};

struct AudioDevice {
    uint32_t id = 0;                           ///< Device ID
    std::string name;                          ///< Device name
    std::string manufacturer;                  ///< Manufacturer name
    std::string driver;                        ///< Driver name (ASIO, CoreAudio, etc.)
    AudioDeviceType type = AudioDeviceType::OUTPUT;
    AudioChannelLayout channel_layout = AudioChannelLayout::STEREO;
    int max_channels = 2;                      ///< Maximum supported channels
    int max_input_channels = 0;                ///< Maximum input channels
    int max_output_channels = 2;               ///< Maximum output channels
    std::vector<int> supported_sample_rates;  ///< Supported sample rates
    std::vector<AudioBitDepth> supported_bit_depths;
    int default_sample_rate = 48000;           ///< Default sample rate
    AudioBitDepth default_bit_depth = AudioBitDepth::FLOAT32;
    int preferred_buffer_size = 512;           ///< Preferred buffer size
    double default_latency_ms = 10.0;         ///< Default latency
    bool supports_low_latency = true;          ///< Low latency support
    bool supports_exclusive_mode = true;      ///< Exclusive mode support
    bool supports_digital_io = false;         ///< Digital I/O support
    std::array<float, 8> current_levels{0.0f}; ///< Current input/output levels
    bool is_active = false;                    ///< Device active state
    std::chrono::steady_clock::time_point last_activity;
};

struct AudioBus {
    uint32_t id = 0;                           ///< Bus ID
    std::string name;                          ///< Bus name
    std::vector<uint32_t> input_channels;      ///< Input channel indices
    std::vector<uint32_t> output_channels;     ///< Output channel indices
    float master_gain = 1.0f;                 ///< Master gain
    std::vector<float> channel_gains;          ///< Individual channel gains
    bool muted = false;                        ///< Mute state
    bool soloed = false;                       ///< Solo state
    AudioRoutingMode routing_mode = AudioRoutingMode::DIRECT;
    std::unique_ptr<RealtimeEffectsChain> effects_chain;
    bool enable_recording = false;            ///< Enable bus recording
    std::string record_file_path;             ///< Recording file path
};

struct AudioRoutingMatrix {
    std::vector<std::vector<float>> matrix;    ///< Routing matrix [input][output]
    int input_channels = 0;                   ///< Number of input channels
    int output_channels = 0;                  ///< Number of output channels
    bool enabled = true;                      ///< Matrix enabled
};

struct AudioSession {
    uint32_t id = 0;                           ///< Session ID
    std::string name;                          ///< Session name
    int sample_rate = 48000;                   ///< Sample rate
    AudioBitDepth bit_depth = AudioBitDepth::FLOAT32;
    int buffer_size = 512;                     ///< Buffer size
    int channels = 2;                          ///< Number of channels
    AudioChannelLayout layout = AudioChannelLayout::STEREO;
    AudioSyncMode sync_mode = AudioSyncMode::NONE;
    uint32_t master_device_id = 0;            ///< Master output device
    std::vector<uint32_t> input_device_ids;   ///< Input device IDs
    std::vector<uint32_t> output_device_ids;  ///< Output device IDs
    bool auto_crossfade_devices = true;       ///< Auto crossfade between devices
    float crossfade_duration_ms = 1000.0f;    ///< Crossfade duration
    bool enable_monitoring = true;            ///< Enable input monitoring
    float monitoring_level = 1.0f;            ///< Monitoring level
    bool enable_dithering = false;            ///< Enable dithering for lower bit depths
    std::chrono::steady_clock::time_point start_time;
    uint64_t processed_samples = 0;           ///< Total processed samples
};

struct AudioEngineMetrics {
    uint64_t processed_frames = 0;             ///< Total processed frames
    uint64_t dropped_frames = 0;              ///< Dropped frames count
    double average_latency_ms = 0.0;          ///< Average processing latency
    double peak_latency_ms = 0.0;             ///< Peak processing latency
    double cpu_usage_percent = 0.0;           ///< CPU usage percentage
    double gpu_usage_percent = 0.0;           ///< GPU usage percentage
    double memory_usage_mb = 0.0;             ///< Memory usage in MB
    std::vector<float> input_levels;          ///< Current input levels
    std::vector<float> output_levels;         ///< Current output levels
    uint32_t active_channels = 0;             ///< Number of active channels
    uint32_t active_buses = 0;                ///< Number of active buses
    std::chrono::steady_clock::time_point last_update;
    bool real_time_stable = true;             ///< Real-time processing stable
    int xruns_count = 0;                     ///< Buffer underruns/overruns count
};

using AudioDeviceCallback = std::function<void(const AudioDevice& device)>;
using AudioSessionCallback = std::function<void(const AudioSession& session)>;
using AudioMetricsCallback = std::function<void(const AudioEngineMetrics& metrics)>;
using AudioLevelCallback = std::function<void(int channel, float level)>;

/**
 * Multi-channel Audio Engine
 * High-performance audio processing with multi-device support
 */
class MultiChannelEngine {
public:
    MultiChannelEngine();
    ~MultiChannelEngine();

    /**
     * Initialize audio engine
     * @param session Audio session configuration
     * @return True if initialization successful
     */
    bool initialize(const AudioSession& session);

    /**
     * Shutdown audio engine and cleanup resources
     */
    void shutdown();

    /**
     * Start audio processing
     * @return True if started successfully
     */
    bool start();

    /**
     * Stop audio processing
     */
    void stop();

    /**
     * Process audio callback (called by audio driver)
     * @param input_buffer Input audio buffer
     * @param output_buffer Output audio buffer
     * @param num_samples Number of samples to process
     * @return True if processing successful
     */
    bool processAudioCallback(const float* input_buffer, float* output_buffer, size_t num_samples);

    /**
     * Scan for available audio devices
     * @return List of available devices
     */
    std::vector<AudioDevice> scanDevices();

    /**
     * Get available audio devices
     * @return List of available devices
     */
    std::vector<AudioDevice> getAvailableDevices() const;

    /**
     * Get device by ID
     * @param device_id Device ID
     * @return Device if found
     */
    std::optional<AudioDevice> getDevice(uint32_t device_id) const;

    /**
     * Set master output device
     * @param device_id Device ID
     * @return True if successful
     */
    bool setMasterDevice(uint32_t device_id);

    /**
     * Add input device
     * @param device_id Device ID
     * @return True if successful
     */
    bool addInputDevice(uint32_t device_id);

    /**
     * Add output device
     * @param device_id Device ID
     * @return True if successful
     */
    bool addOutputDevice(uint32_t device_id);

    /**
     * Remove device
     * @param device_id Device ID
     * @return True if successful
     */
    bool removeDevice(uint32_t device_id);

    /**
     * Create audio bus
     * @param name Bus name
     * @param input_channels Input channels
     * @param output_channels Output channels
     * @return Bus ID if successful, 0 otherwise
     */
    uint32_t createBus(const std::string& name,
                      const std::vector<uint32_t>& input_channels,
                      const std::vector<uint32_t>& output_channels);

    /**
     * Remove audio bus
     * @param bus_id Bus ID
     * @return True if successful
     */
    bool removeBus(uint32_t bus_id);

    /**
     * Get bus configuration
     * @param bus_id Bus ID
     * @return Bus configuration if found
     */
    std::optional<AudioBus> getBus(uint32_t bus_id) const;

    /**
     * Add effect chain to bus
     * @param bus_id Bus ID
     * @param effects Effects chain
     * @return True if successful
     */
    bool addEffectsToBus(uint32_t bus_id, std::unique_ptr<RealtimeEffectsChain> effects);

    /**
     * Configure audio channels
     * @param channels Channel configurations
     * @return True if successful
     */
    bool configureChannels(const std::vector<AudioChannel>& channels);

    /**
     * Get channel configuration
     * @param channel_index Channel index
     * @return Channel configuration if found
     */
    std::optional<AudioChannel> getChannel(int channel_index) const;

    /**
     * Set routing matrix
     * @param matrix Routing matrix
     */
    void setRoutingMatrix(const AudioRoutingMatrix& matrix);

    /**
     * Get current routing matrix
     * @return Routing matrix
     */
    AudioRoutingMatrix getRoutingMatrix() const;

    /**
     * Set channel gain
     * @param channel_index Channel index
     * @param gain Gain value (linear)
     * @return True if successful
     */
    bool setChannelGain(int channel_index, float gain);

    /**
     * Set channel pan
     * @param channel_index Channel index
     * @param pan Pan value (-1.0 to 1.0)
     * @return True if successful
     */
    bool setChannelPan(int channel_index, float pan);

    /**
     * Mute/unmute channel
     * @param channel_index Channel index
     * @param muted Mute state
     * @return True if successful
     */
    bool setChannelMuted(int channel_index, bool muted);

    /**
     * Solo/unsolo channel
     * @param channel_index Channel index
     * @param soloed Solo state
     * @return True if successful
     */
    bool setChannelSoloed(int channel_index, bool soloed);

    /**
     * Enable/disable channel recording
     * @param channel_index Channel index
     * @param enabled Recording enabled
     * @param file_path Recording file path
     * @return True if successful
     */
    bool setChannelRecording(int channel_index, bool enabled, const std::string& file_path = "");

    /**
     * Set bus gain
     * @param bus_id Bus ID
     * @param gain Gain value (linear)
     * @return True if successful
     */
    bool setBusGain(uint32_t bus_id, float gain);

    /**
     * Mute/unmute bus
     * @param bus_id Bus ID
     * @param muted Mute state
     * @return True if successful
     */
    bool setBusMuted(uint32_t bus_id, bool muted);

    /**
     * Solo/unsolo bus
     * @param bus_id Bus ID
     * @param soloed Solo state
     * @return True if successful
     */
    bool setBusSoloed(uint32_t bus_id, bool soloed);

    /**
     * Enable/disable bus recording
     * @param bus_id Bus ID
     * @param enabled Recording enabled
     * @param file_path Recording file path
     * @return True if successful
     */
    bool setBusRecording(uint32_t bus_id, bool enabled, const std::string& file_path = "");

    /**
     * Set routing between channels
     * @param from_channel Source channel
     * @param to_channel Destination channel
     * @param gain Routing gain
     * @return True if successful
     */
    bool setChannelRouting(int from_channel, int to_channel, float gain);

    /**
     * Add send routing
     * @param from_channel Source channel
     * @param to_bus Destination bus
     * @param gain Send gain
     * @return True if successful
     */
    bool addSendRouting(int from_channel, uint32_t to_bus, float gain);

    /**
     * Remove send routing
     * @param from_channel Source channel
     * @param to_bus Destination bus
     * @return True if successful
     */
    bool removeSendRouting(int from_channel, uint32_t to_bus);

    /**
     * Get current audio metrics
     * @return Audio engine metrics
     */
    AudioEngineMetrics getMetrics() const;

    /**
     * Register device callback
     * @param callback Device change callback
     */
    void setDeviceCallback(AudioDeviceCallback callback);

    /**
     * Register session callback
     * @param callback Session change callback
     */
    void setSessionCallback(AudioSessionCallback callback);

    /**
     * Register metrics callback
     * @param callback Metrics callback
     */
    void setMetricsCallback(AudioMetricsCallback callback);

    /**
     * Register audio level callback
     * @param callback Level change callback
     */
    void setLevelCallback(AudioLevelCallback callback);

    /**
     * Set channel layout
     * @param layout Channel layout
     */
    void setChannelLayout(AudioChannelLayout layout);

    /**
     * Get current channel layout
     * @return Channel layout
     */
    AudioChannelLayout getChannelLayout() const;

    /**
     * Set bit depth
     * @param bit_depth Audio bit depth
     * @return True if successful
     */
    bool setBitDepth(AudioBitDepth bit_depth);

    /**
     * Get current bit depth
     * @return Audio bit depth
     */
    AudioBitDepth getBitDepth() const;

    /**
     * Enable/disable dithering
     * @param enabled Dithering enabled
     */
    void setDitheringEnabled(bool enabled);

    /**
     * Is dithering enabled
     * @return True if dithering is enabled
     */
    bool isDitheringEnabled() const;

    /**
     * Set synchronization mode
     * @param mode Sync mode
     * @return True if successful
     */
    bool setSyncMode(AudioSyncMode mode);

    /**
     * Get current sync mode
     * @return Sync mode
     */
    AudioSyncMode getSyncMode() const;

    /**
     * Validate engine configuration
     * @return True if configuration is valid
     */
    bool validateConfiguration() const;

    /**
     * Reset engine to default state
     */
    void reset();

private:
    struct ChannelState {
        AudioChannel config;
        std::vector<float> buffer;
        float current_gain = 1.0f;
        float target_gain = 1.0f;
        float current_pan = 0.0f;
        float target_pan = 0.0f;
        bool ramp_gain = false;
        bool ramp_pan = false;
        uint32_t record_handle = 0;
    };

    struct BusState {
        AudioBus config;
        std::vector<float> mix_buffer;
        std::vector<float> output_buffer;
        float current_master_gain = 1.0f;
        float target_master_gain = 1.0f;
        bool ramp_gain = false;
        uint32_t record_handle = 0;
    };

    // Core session state
    AudioSession session_;
    bool initialized_ = false;
    bool running_ = false;
    mutable std::mutex session_mutex_;

    // Device management
    std::unordered_map<uint32_t, AudioDevice> available_devices_;
    std::unordered_map<uint32_t, AudioDevice> active_devices_;
    uint32_t master_device_id_ = 0;
    std::vector<uint32_t> input_device_ids_;
    std::vector<uint32_t> output_device_ids_;
    mutable std::mutex devices_mutex_;

    // Channel management
    std::vector<ChannelState> channels_;
    mutable std::mutex channels_mutex_;

    // Bus management
    std::unordered_map<uint32_t, std::unique_ptr<BusState>> buses_;
    std::atomic<uint32_t> next_bus_id_{1};
    mutable std::mutex buses_mutex_;

    // Routing
    AudioRoutingMatrix routing_matrix_;
    std::vector<std::vector<std::pair<uint32_t, float>>> send_routing_; // [channel][bus_id, gain]
    mutable std::mutex routing_mutex_;

    // Audio processing components
    std::unique_ptr<SpatialAudioProcessor> spatial_processor_;
    std::unique_ptr<SpectrumAnalyzer> spectrum_analyzer_;
    std::unique_ptr<WaveformProcessor> waveform_processor_;
    std::unique_ptr<VUMeterProcessor> vu_processor_;

    // Buffers
    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;
    std::vector<float> mix_buffer_;
    std::vector<float> temp_buffer_;

    // Metrics and callbacks
    mutable std::mutex metrics_mutex_;
    AudioEngineMetrics metrics_;
    AudioDeviceCallback device_callback_;
    AudioSessionCallback session_callback_;
    AudioMetricsCallback metrics_callback_;
    AudioLevelCallback level_callback_;
    std::chrono::steady_clock::time_point last_metrics_update_;
    std::vector<float> peak_levels_;
    uint64_t xruns_count_ = 0;

    // Real-time processing state
    std::atomic<bool> processing_active_{false};
    std::atomic<uint64_t> processed_frames_{0};
    std::chrono::high_resolution_clock::time_point start_time_;

    // Internal methods
    bool initializeDevices();
    bool initializeAudioDrivers();
    void shutdownAudioDrivers();
    bool validateDeviceCompatibility(const std::vector<AudioDevice>& devices) const;
    void processChannels(float* input_buffer, float* output_buffer, size_t num_samples);
    void processBuses(float* output_buffer, size_t num_samples);
    void applyRouting(float* input_buffer, float* output_buffer, size_t num_samples);
    void applyDithering(float* buffer, size_t num_samples);
    void updateChannelLevels(float* buffer, size_t num_samples);
    void updateMetrics();
    void startRecording();
    void stopRecording();
    void initializeChannelLayout();
    int getChannelCountForLayout(AudioChannelLayout layout) const;
    std::vector<float> getChannelGainsForLayout(AudioChannelLayout layout) const;
    void performDeviceCrossfade(uint32_t old_device_id, uint32_t new_device_id);
    bool startDevice(uint32_t device_id);
    bool stopDevice(uint32_t device_id);
    void processDeviceAudio(AudioDevice& device, float* buffer, size_t num_samples, bool is_input);
    float calculateRMS(float* buffer, size_t samples);
    float calculatePeak(float* buffer, size_t samples);
    void applyGainRamp(float* buffer, size_t samples, float start_gain, float end_gain);
    void applyPan(float* buffer, size_t samples, float pan, int channels);
    void syncAllDevices();
    void detectXRuns();
};

/**
 * Audio Device Manager
 * Handles device enumeration and management
 */
class AudioDeviceManager {
public:
    AudioDeviceManager();
    ~AudioDeviceManager();

    /**
     * Initialize device manager
     * @return True if successful
     */
    bool initialize();

    /**
     * Scan for available devices
     * @return List of available devices
     */
    std::vector<AudioDevice> scanDevices();

    /**
     * Get device capabilities
     * @param device_id Device ID
     * @return Device capabilities
     */
    std::vector<int> getDeviceSupportedSampleRates(uint32_t device_id) const;

    /**
     * Get device supported bit depths
     * @param device_id Device ID
     * @return Supported bit depths
     */
    std::vector<AudioBitDepth> getDeviceSupportedBitDepths(uint32_t device_id) const;

    /**
     * Test device
     * @param device_id Device ID
     * @param sample_rate Test sample rate
     * @param buffer_size Test buffer size
     * @return Test result
     */
    bool testDevice(uint32_t device_id, int sample_rate, int buffer_size);

    /**
     * Get default input device
     * @return Default input device ID
     */
    uint32_t getDefaultInputDevice() const;

    /**
     * Get default output device
     * @return Default output device ID
     */
    uint32_t getDefaultOutputDevice() const;

private:
    std::unordered_map<uint32_t, AudioDevice> devices_;
    std::atomic<uint32_t> next_device_id_{1};
    bool initialized_ = false;
    mutable std::mutex mutex_;

    bool loadDeviceDrivers();
    void unloadDeviceDrivers();
    bool initializeASIO();
    bool initializeCoreAudio();
    bool initializeWASAPI();
    bool initializeALSA();
    bool initializeJack();
};

// Utility functions
namespace audio_engine_utils {

    // Channel layout utilities
    int getChannelCount(AudioChannelLayout layout);
    std::vector<std::string> getChannelNames(AudioChannelLayout layout);
    std::vector<float> getDefaultChannelPanning(AudioChannelLayout layout);
    std::vector<float> getDefaultSpeakerPositions(AudioChannelLayout layout);

    // Bit depth utilities
    int getBitDepthSize(AudioBitDepth bit_depth);
    double getBitDepthDynamicRange(AudioBitDepth bit_depth);
    bool requiresDithering(AudioBitDepth source, AudioBitDepth target);

    // Device utilities
    std::string getDeviceTypeString(AudioDeviceType type);
    std::string getLayoutString(AudioChannelLayout layout);
    std::string getBitDepthString(AudioBitDepth bit_depth);
    std::string getSyncModeString(AudioSyncMode mode);

    // Conversion utilities
    void convertBitDepth(const void* input, void* output, size_t samples,
                        AudioBitDepth from, AudioBitDepth to, bool dither = false);
    void interleaveChannels(const float** input, float* output, size_t samples, int channels);
    void deinterleaveChannels(const float* input, float** output, size_t samples, int channels);

    // Audio analysis utilities
    float calculateLUFS(const float* buffer, size_t samples, int sample_rate);
    float calculateTruePeak(const float* buffer, size_t samples, int sample_rate);
    std::vector<float> calculateFrequencyBands(const float* buffer, size_t samples, int sample_rate,
                                              const std::vector<float>& frequencies);
}

} // namespace audio
} // namespace core
} // namespace vortex