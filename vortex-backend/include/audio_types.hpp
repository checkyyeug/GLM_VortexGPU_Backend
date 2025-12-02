#pragma once

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <cstdint>
#include <optional>

namespace vortex {

using SystemClock = std::chrono::system_clock;
using Duration = std::chrono::duration<double>;

// Audio format enumeration
enum class AudioFormat {
    UNKNOWN,
    MP3,
    WAV,
    FLAC,
    ALAC,
    AAC,
    OGG,
    M4A,
    DSD64,
    DSD128,
    DSD256,
    DSD512,
    DSD1024,
    DSF,
    DFF,
    PCM16,
    PCM24,
    PCM32
};

// Processing status
enum class ProcessingStatus {
    IDLE,
    LOADING,
    READY,
    PROCESSING,
    COMPLETED,
    ERROR
};

// GPU backend types
enum class GPUBackend {
    NONE,
    CUDA,
    OPENCL,
    VULKAN
};

// Filter types for processing chain
enum class FilterType {
    EQUALIZER,
    CONVOLUTION,
    DSD_PROCESSOR,
    RESAMPLER,
    COMPRESSOR,
    LIMITER,
    DELAY,
    REVERB,
    GATE,
    CUSTOM
};

// Output device types
enum class OutputDeviceType {
    LOCAL,
    ROON_BRIDGE,
    HQPLAYER_NAA,
    UPNP_RENDERER,
    NETWORK_STREAM
};

// Playback state
enum class PlaybackState {
    STOPPED,
    PLAYING,
    PAUSED,
    SEEKING,
    BUFFERING,
    ERROR
};

// Playback mode
enum class PlaybackMode {
    PLAY_ONCE,
    LOOP_TRACK,
    LOOP_PLAYLIST,
    SHUFFLE
};

// Audio file metadata structure
struct AudioMetadata {
    // Technical metadata
    AudioFormat format = AudioFormat::UNKNOWN;
    std::string codec;
    uint32_t bitrate = 0;
    uint32_t sampleRate = 0;
    uint16_t bitDepth = 0;
    uint16_t channels = 0;
    Duration duration{0};

    // Artistic metadata
    std::optional<std::string> title;
    std::optional<std::string> artist;
    std::optional<std::string> album;
    std::optional<uint16_t> year;
    std::optional<std::string> genre;
    std::optional<uint16_t> track;
    std::optional<std::string> albumArt;

    // Quality metrics
    float dynamicRange = 0.0f;
    float peakLevel = 0.0f;
    std::optional<float> replayGain;
};

// Audio file structure
struct AudioFile {
    std::string id;
    std::string name;
    std::string filePath;
    AudioFormat format = AudioFormat::UNKNOWN;
    AudioMetadata metadata;
    ProcessingStatus status = ProcessingStatus::IDLE;
    float loadingProgress = 0.0f;
    uint64_t fileSize = 0;
    Duration duration{0};
    SystemClock::time_point createdAt;
    SystemClock::time_point updatedAt;
    std::optional<std::string> errorMessage;
};

// Filter module for processing chain
struct FilterModule {
    std::string id;
    std::string name;
    FilterType type;
    bool isEnabled = true;
    bool isBypassed = false;
    bool isSolo = false;
    float wetMix = 1.0f;
    std::map<std::string, float> parameters;
    uint32_t position = 0;
    std::optional<std::string> pluginId;
};

// Processing chain structure
struct ProcessingChain {
    std::string id;
    std::string name;
    std::vector<FilterModule> filters;
    bool isActive = true;
    float masterVolume = 1.0f;
    bool bypassChain = false;
    SystemClock::time_point lastModified;
};

// Device capabilities
struct DeviceCapabilities {
    bool supportsDSD = false;
    bool supportsPCM768 = false;
    bool supportsMultiChannel = false;
    bool supportsRealtimeControl = false;
    uint32_t maxBitrate = 0;
    std::vector<std::string> exclusiveFeatures;
};

// Output device structure
struct OutputDevice {
    std::string id;
    std::string name;
    OutputDeviceType type;
    std::string ipAddress;
    uint16_t port = 0;
    std::vector<AudioFormat> supportedFormats;
    std::vector<uint32_t> supportedSampleRates;
    std::vector<uint16_t> supportedBitDepths;
    uint32_t maxChannels = 0;
    uint32_t latency = 0;
    bool isConnected = false;
    DeviceCapabilities capabilities;
    SystemClock::time_point lastSeen;
};

// Audio session structure
struct AudioSession {
    std::string id;
    std::optional<std::string> audioFileId;
    std::optional<std::string> outputDeviceId;
    std::optional<std::string> processingChainId;
    PlaybackState state = PlaybackState::STOPPED;
    Duration currentTime{0};
    float volume = 1.0f;
    bool isMuted = false;
    PlaybackMode mode = PlaybackMode::PLAY_ONCE;
    SystemClock::time_point sessionStart;
    SystemClock::time_point lastActivity;
};

// Spectrum data structure
struct SpectrumData {
    std::vector<float> bins;
    float frequencyRange[2] = {20.0f, 20000.0f};
    float amplitudeRange[2] = {0.0f, 1.0f};
    uint32_t fftSize = 2048;
    float windowOverlap = 0.75f;
};

// Waveform data structure
struct WaveformData {
    std::vector<float> leftChannel;
    std::vector<float> rightChannel;
    uint32_t sampleRate = 0;
    uint16_t bitDepth = 0;
    float peakLevels[2] = {0.0f, 0.0f};
};

// VU meter data structure
struct VUMeterData {
    float vuLevels[2] = {-60.0f, -60.0f};
    float peakLevels[2] = {-60.0f, -60.0f};
    float rmsLevels[2] = {-60.0f, -60.0f};
    float stereoCorrelation = 0.0f;
    float dynamicRange = 0.0f;
};

// GPU status structure
struct GPUStatus {
    float utilization = 0.0f;
    uint64_t memoryUsed = 0;
    uint64_t memoryTotal = 0;
    float temperature = 0.0f;
    float powerUsage = 0.0f;
    float clockSpeed = 0.0f;
    uint32_t activeCores = 0;
};

// NPU status structure
struct NPUStatus {
    float utilization = 0.0f;
    uint64_t memoryUsed = 0;
    uint64_t memoryTotal = 0;
    float temperature = 0.0f;
    float powerUsage = 0.0f;
};

// CPU status structure
struct CPUStatus {
    float utilization = 0.0f;
    uint64_t memoryUsed = 0;
    uint64_t memoryTotal = 0;
    float temperature = 0.0f;
    uint32_t activeCores = 0;
    uint32_t totalCores = 0;
};

// Memory status structure
struct MemoryStatus {
    uint64_t total = 0;
    uint64_t used = 0;
    uint64_t available = 0;
    float utilization = 0.0f;
};

// Latency analysis structure
struct LatencyAnalysis {
    float audioProcessing = 0.0f;
    float networkTransfer = 0.0f;
    float gpuCompute = 0.0f;
    float diskIO = 0.0f;
    float total = 0.0f;
};

// Hardware status structure
struct HardwareStatus {
    GPUStatus gpu;
    NPUStatus npu;
    CPUStatus cpu;
    MemoryStatus memory;
    LatencyAnalysis latency;
};

// Processing metrics structure
struct ProcessingMetrics {
    float processingLatency = 0.0f;
    float throughput = 0.0f;
    uint32_t droppedFrames = 0;
    float cpuUsage = 0.0f;
    uint64_t samplesProcessed = 0;
    float qualityScore = 0.0f;
};

// Connection status structure
struct ConnectionStatus {
    bool isConnected = false;
    uint32_t activeConnections = 0;
    float latency = 0.0f;
    uint64_t bytesTransferred = 0;
    uint64_t packetsLost = 0;
};

// System alert structure
struct SystemAlert {
    std::string id;
    std::string type;
    std::string severity;
    std::string message;
    SystemClock::time_point timestamp;
    bool acknowledged = false;
};

// Real-time data structure
struct RealTimeData {
    uint64_t timestamp;
    SpectrumData spectrum;
    WaveformData waveform;
    VUMeterData vuMeters;
    HardwareStatus hardware;
    ProcessingMetrics processing;
    ConnectionStatus network;
    std::vector<SystemAlert> alerts;
};

// Audio settings structure
struct AudioSettings {
    uint32_t defaultSampleRate = 44100;
    uint16_t defaultBitDepth = 24;
    uint16_t defaultChannels = 2;
    uint32_t bufferSize = 4096;
    float maxLatency = 10.0f;
    bool enableDSDProcessing = true;
    bool enableGPUAcceleration = true;
};

// Network settings structure
struct NetworkSettings {
    uint16_t httpPort = 8080;
    uint16_t websocketPort = 8081;
    uint16_t discoveryPort = 8082;
    bool enableSSL = false;
    uint32_t maxConnections = 100;
    uint32_t connectionTimeout = 30;
};

// GPU preferences structure
struct GPUPreferences {
    std::vector<GPUBackend> preferredBackends = {GPUBackend::CUDA, GPUBackend::OPENCL, GPUBackend::VULKAN};
    bool enableMultiGPU = true;
    uint64_t memoryLimit = 8192;
    float utilizationTarget = 80.0f;
    bool enableFallback = true;
};

// Monitoring settings structure
struct MonitoringSettings {
    bool enableHardwareMonitoring = true;
    bool enablePerformanceMetrics = true;
    bool enableAlerts = true;
    uint32_t updateRate = 60;
    float gpuUtilizationThreshold = 90.0f;
    float cpuUtilizationThreshold = 80.0f;
    float memoryUtilizationThreshold = 85.0f;
};

// User preferences structure
struct UserPreferences {
    std::string theme = "dark";
    std::string language = "en";
    bool enableNotifications = true;
    bool enableAutoSave = true;
    uint32_t autoSaveInterval = 300;
    bool enableBackup = true;
};

// System configuration structure
struct SystemConfiguration {
    AudioSettings audio;
    NetworkSettings network;
    GPUPreferences gpu;
    MonitoringSettings monitoring;
    UserPreferences user;
    SystemClock::time_point lastModified;
};

// Audio buffer structure
struct AudioBuffer {
    std::vector<float> data;
    uint32_t sampleRate = 0;
    uint16_t channels = 0;
    uint32_t frames = 0;
    bool interleaved = true;
};

// Format utility functions
const char* formatToString(AudioFormat format);
const char* statusToString(ProcessingStatus status);
const char* backendToString(GPUBackend backend);
const char* filterTypeToString(FilterType type);
const char* deviceTypeToString(OutputDeviceType type);
const char* playbackStateToString(PlaybackState state);
const char* playbackModeToString(PlaybackMode mode);

bool isLosslessFormat(AudioFormat format);
bool isLossyFormat(AudioFormat format);
bool isDSDFormat(AudioFormat format);
bool isPCMFormat(AudioFormat format);

uint32_t formatToBitrate(AudioFormat format);
uint32_t formatToSampleRate(AudioFormat format);
uint16_t formatToBitDepth(AudioFormat format);

} // namespace vortex